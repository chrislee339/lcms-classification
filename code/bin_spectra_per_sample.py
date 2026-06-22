"""Bin LC-MS mzML files into 2D spectral PNG images using per-sample
normalization (the headline pipeline of the manuscript).

This is the canonical binning algorithm used to generate the
``data/raw/spectra_filter_<N>/`` PNGs that all primary results were trained
on. The algorithm:

  1. Drop the bottom-percentile scans by peak count
     (cutoff = max(hard_min_peaks, P-th percentile of peak counts)).
     Default thresholds: hard_min_peaks=5000, percentile=10.
  2. Compute per-file m/z, retention-time, and log10-intensity ranges.
  3. Bin peaks onto an N x N grid using those per-file ranges, accumulating
     normalized log-intensities and per-bin counts.
  4. Average per bin (count-normalized) and write a viridis PNG.

Pass ``--streaming`` to use OnDiscMSExperiment for memory safety on
workstations that cannot fit a full MSExperiment in RAM.

Example:
    python bin_spectra_per_sample.py \\
        --mzml data/mzml \\
        --output data/raw/spectra_filter_512 \\
        --bins 512 --workers 4
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
    sample, src, out, bins, hard_min_peaks, percentile, streaming = args
    if out.exists():
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
    log_min, log_max = float("inf"), -float("inf")
    for s in filtered:
        mz, intensity = s.get_peaks()
        rt = s.getRT()
        if len(mz) == 0:
            continue
        mz_min = min(mz_min, float(mz.min()))
        mz_max = max(mz_max, float(mz.max()))
        rt_min = min(rt_min, rt)
        rt_max = max(rt_max, rt)
        log_intensity = np.log10(intensity + EPSILON)
        log_min = min(log_min, float(log_intensity.min()))
        log_max = max(log_max, float(log_intensity.max()))

    image = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)
    for s in filtered:
        mz, intensity = s.get_peaks()
        rt = s.getRT()
        if len(mz) == 0:
            continue
        log_intensity = np.log10(intensity + EPSILON)
        norm_log = (log_intensity - log_min) / (log_max - log_min)
        x_idx = ((mz - mz_min) / (mz_max - mz_min) * (bins - 1)).astype(int)
        y_idx = int((rt - rt_min) / (rt_max - rt_min) * (bins - 1))
        for i in range(len(mz)):
            image[y_idx, x_idx[i]] += norm_log[i]
            counts[y_idx, x_idx[i]] += 1

    counts[counts == 0] = 1
    image /= counts
    imsave(str(out), image, cmap="viridis", origin="lower")
    return f"binned: {sample} ({time.time()-t0:.1f}s)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mzml", required=True,
                    help="Input directory containing .mzML files")
    ap.add_argument("--output", required=True,
                    help="Output directory for PNGs")
    ap.add_argument("--bins", type=int, default=512,
                    help="Output image side length (default: 512)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Number of parallel workers (default: 2)")
    ap.add_argument("--hard-min-peaks", type=int, default=5000,
                    help="Hard floor on peaks-per-scan (default: 5000)")
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="Drop scans below this percentile (default: 10)")
    ap.add_argument("--streaming", action="store_true",
                    help="Use OnDiscMSExperiment (low-RAM mode)")
    args = ap.parse_args()

    mzml_dir = Path(args.mzml)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_mzml(mzml_dir)
    if not samples:
        raise SystemExit(f"No mzML files found under {mzml_dir}")
    log(f"Found {len(samples)} mzML files; binning to {out_dir} "
        f"({args.bins}x{args.bins}, {args.workers} workers, "
        f"streaming={args.streaming})")

    work = [
        (sid, src, out_dir / f"{sid}.png", args.bins,
         args.hard_min_peaks, args.percentile, args.streaming)
        for sid, src in sorted(samples.items())
    ]
    with Pool(max(1, args.workers)) as pool:
        for i, msg in enumerate(pool.imap_unordered(bin_one, work), 1):
            log(f"  [{i}/{len(work)}] {msg}")
    log(f"Done. PNGs at {out_dir}")


if __name__ == "__main__":
    main()
