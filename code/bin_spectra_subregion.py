"""Bin LC-MS mzML files into focused-subregion PNGs (the bottom-left
quadrant of m/z x RT, used for the subregion ablation in Section 3.7).

Algorithm matches bin_spectra_per_sample.py except that only spectra/peaks
within the lower half of the per-file m/z and RT ranges are kept. Per-file
ranges are used (not global), mirroring the per-sample normalization choice
of the headline pipeline.

For each input mzML, one PNG is written per requested binning resolution:
``<output>/focused_zoom_<RES>/<sample>.png``.

Example:
    python bin_spectra_subregion.py \\
        --mzml data/mzml \\
        --output data/raw \\
        --bins 256 512 1024 --workers 2
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from matplotlib.pyplot import imsave

EXCLUDE_FILES = {"Fetuin_final", "S2B-5x-0302"}
EPSILON = 1e-6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_mzml(mzml_dir: Path):
    samples = {}
    for sub in (mzml_dir, mzml_dir / "classA", mzml_dir / "classB"):
        if sub.is_dir():
            for p in sub.glob("*.mzML"):
                stem = p.stem
                if stem in EXCLUDE_FILES:
                    continue
                samples[stem] = p
    return samples


def get_filtered_spectra(spectra, hard_min_peaks: int, percentile: float):
    peak_counts = [s.size() for s in spectra]
    cutoff = max(hard_min_peaks, float(np.percentile(peak_counts, percentile)))
    return [s for s in spectra if s.size() >= cutoff]


def load_spectra(src: Path, streaming: bool):
    if streaming:
        from pyopenms import OnDiscMSExperiment
        exp = OnDiscMSExperiment()
        if not exp.openFile(str(src)):
            raise RuntimeError(f"OnDiscMSExperiment failed to open {src}")
        return [exp.getSpectrum(i) for i in range(exp.getNrSpectra())]
    from pyopenms import MSExperiment, MzMLFile
    exp = MSExperiment()
    MzMLFile().load(str(src), exp)
    return list(exp.getSpectra())


def bin_one(args):
    (sample, src, out_paths, bins_list, hard_min_peaks, percentile,
     streaming) = args
    if all(p.exists() for p in out_paths):
        return f"already binned: {sample}"
    t0 = time.time()
    try:
        spectra = load_spectra(src, streaming)
    except Exception as e:
        return f"FAILED to open {sample}: {e}"
    filtered = get_filtered_spectra(spectra, hard_min_peaks, percentile)
    if not filtered:
        return f"SKIP {sample}: no scans after filter (had {len(spectra)} total)"

    mz_min, mz_max = float("inf"), 0.0
    rt_min, rt_max = float("inf"), 0.0
    for s in filtered:
        mz, _ = s.get_peaks()
        rt = s.getRT()
        if len(mz) == 0:
            continue
        mz_min = min(mz_min, float(mz.min()))
        mz_max = max(mz_max, float(mz.max()))
        rt_min = min(rt_min, rt)
        rt_max = max(rt_max, rt)
    mz_range = (mz_min, mz_min + 0.5 * (mz_max - mz_min))
    rt_range = (rt_min, rt_min + 0.5 * (rt_max - rt_min))

    log_min, log_max = float("inf"), -float("inf")
    buffer = []
    for s in filtered:
        rt = s.getRT()
        if not rt_range[0] <= rt <= rt_range[1]:
            continue
        mz, intensity = s.get_peaks()
        if len(mz) == 0:
            continue
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz = mz[mask]
        intensity = intensity[mask]
        if len(mz) == 0:
            continue
        log_intensity = np.log10(intensity + EPSILON)
        log_min = min(log_min, float(log_intensity.min()))
        log_max = max(log_max, float(log_intensity.max()))
        buffer.append((mz, rt, log_intensity))

    if not buffer:
        return f"SKIP {sample}: no peaks in subregion"

    for bins, out in zip(bins_list, out_paths):
        image = np.zeros((bins, bins), dtype=np.float64)
        counts = np.zeros((bins, bins), dtype=np.float64)
        for mz, rt, log_intensity in buffer:
            norm_log = (log_intensity - log_min) / (log_max - log_min)
            x_idx = ((mz - mz_range[0]) / (mz_range[1] - mz_range[0])
                     * (bins - 1)).astype(int)
            y_idx = int((rt - rt_range[0]) / (rt_range[1] - rt_range[0])
                        * (bins - 1))
            for i in range(len(mz)):
                image[y_idx, x_idx[i]] += norm_log[i]
                counts[y_idx, x_idx[i]] += 1
        counts[counts == 0] = 1
        image /= counts
        out.parent.mkdir(parents=True, exist_ok=True)
        imsave(str(out), image, cmap="viridis", origin="lower")
    return f"binned: {sample} ({time.time()-t0:.1f}s)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mzml", required=True)
    ap.add_argument("--output", required=True,
                    help="Output base; PNGs go to <output>/focused_zoom_<RES>/")
    ap.add_argument("--bins", type=int, nargs="+", default=[256, 512, 1024],
                    help="One or more output side lengths (default: 256 512 1024)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hard-min-peaks", type=int, default=5000)
    ap.add_argument("--percentile", type=float, default=10.0)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    mzml_dir = Path(args.mzml)
    out_base = Path(args.output)

    samples = discover_mzml(mzml_dir)
    if not samples:
        raise SystemExit(f"No mzML files found under {mzml_dir}")
    log(f"Found {len(samples)} mzML files; binning to "
        f"{out_base}/focused_zoom_<{','.join(map(str, args.bins))}>/ "
        f"({args.workers} workers, streaming={args.streaming})")

    work = []
    for sid, src in sorted(samples.items()):
        out_paths = [out_base / f"focused_zoom_{b}" / f"{sid}.png"
                     for b in args.bins]
        work.append((sid, src, out_paths, args.bins,
                     args.hard_min_peaks, args.percentile, args.streaming))

    with Pool(max(1, args.workers)) as pool:
        for i, msg in enumerate(pool.imap_unordered(bin_one, work), 1):
            log(f"  [{i}/{len(work)}] {msg}")
    log("Done.")


if __name__ == "__main__":
    main()
