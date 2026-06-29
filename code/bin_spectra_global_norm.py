"""Re-bin LC-MS mzML files using globally aligned m/z and RT axes.

This is the data-preparation step for the per-sample vs global normalization
head-to-head experiment. It replaces the per-file min/max normalization performed in
`bin_spectra_per_sample.py` with a single cohort-wide axis range:

  * Pass 1 (over all mzML files): compute global m/z range, RT range, and
    log-intensity range across every spectrum in every file. Also records
    a per-file peak-count distribution so we can replicate the original
    "drop spectra with < 5000 peaks" filter without holding all spectra in
    memory at once.
  * Pass 2 (over all mzML files): bin each file into the SAME 2D grid using
    the global ranges from pass 1. Log intensities are normalized using the
    global log_min / log_max so absolute m/z positions and intensities are
    comparable across runs.

Memory-frugal implementation: spectra are streamed one at a time via
`OnDiscMSExperiment`, so each pass holds at most one decoded spectrum in
memory regardless of mzML file size. This keeps memory use low on machines
that cannot hold a full mzML in RAM.

Usage:
    python3 bin_spectra_global_norm.py --mzml /path/to/mzml \
        --output data/raw/spectra_filter_global_512 \
        --bins 512 --workers 1

After running, train via `run_global_norm_comparison.py` (which builds the
kfold split internally).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from matplotlib.pyplot import imsave


EPS = 1e-6
HARD_MIN_PEAKS = 5000
PEAK_PERCENTILE = 10


def _open_on_disk(path: Path):
    from pyopenms import OnDiscMSExperiment
    e = OnDiscMSExperiment()
    e.setSkipXMLChecks(True)
    if not e.openFile(str(path)):
        raise IOError(f"OnDiscMSExperiment.openFile failed for {path}")
    return e


def _peak_count_cutoff(exp) -> int:
    """Two-pass-on-the-same-file cost is fine: getSpectrum is fast on the
    on-disk index. We record peak counts to set a per-file cutoff matching
    the original per-sample binning behavior."""
    n = exp.getNrSpectra()
    counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        counts[i] = exp.getSpectrum(i).size()
    return int(max(HARD_MIN_PEAKS, np.percentile(counts, PEAK_PERCENTILE)))


def file_ranges(mzml_path: Path):
    """Pass 1: stream once through the file and return
    (mz_min, mz_max, rt_min, rt_max, log_min, log_max, n_filtered)."""
    exp = _open_on_disk(mzml_path)
    cutoff = _peak_count_cutoff(exp)
    n_total = exp.getNrSpectra()

    mz_min, mz_max = float("inf"), 0.0
    rt_min, rt_max = float("inf"), 0.0
    log_min, log_max = float("inf"), -float("inf")
    n_filtered = 0
    for i in range(n_total):
        s = exp.getSpectrum(i)
        if s.size() < cutoff:
            continue
        mz, intensity = s.get_peaks()
        if len(mz) == 0:
            continue
        rt = float(s.getRT())
        mz_min = min(mz_min, float(mz.min()))
        mz_max = max(mz_max, float(mz.max()))
        rt_min = min(rt_min, rt)
        rt_max = max(rt_max, rt)
        log_intensity = np.log10(intensity + EPS)
        log_min = min(log_min, float(log_intensity.min()))
        log_max = max(log_max, float(log_intensity.max()))
        n_filtered += 1
    return (mz_min, mz_max, rt_min, rt_max, log_min, log_max, n_filtered)


def bin_with_global_ranges(mzml_path: Path, out_dir: Path, bins: int,
                           mz_min: float, mz_max: float,
                           rt_min: float, rt_max: float,
                           log_min: float, log_max: float) -> None:
    """Pass 2: stream once, accumulate into a (bins, bins) image using the
    global ranges. Constant memory regardless of file size."""
    exp = _open_on_disk(mzml_path)
    cutoff = _peak_count_cutoff(exp)
    n_total = exp.getNrSpectra()

    image = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.int64)
    mz_span = mz_max - mz_min
    rt_span = rt_max - rt_min
    log_span = log_max - log_min

    for i in range(n_total):
        s = exp.getSpectrum(i)
        if s.size() < cutoff:
            continue
        mz, intensity = s.get_peaks()
        if len(mz) == 0:
            continue
        rt = float(s.getRT())
        if not (rt_min <= rt <= rt_max):
            continue
        m_in = (mz >= mz_min) & (mz <= mz_max)
        if not m_in.any():
            continue
        mz_in = mz[m_in]
        intensity_in = intensity[m_in]
        log_int = np.log10(intensity_in + EPS)
        norm_log = np.clip((log_int - log_min) / log_span, 0.0, 1.0)
        x_idx = ((mz_in - mz_min) / mz_span * (bins - 1)).astype(np.int64)
        y_idx = int((rt - rt_min) / rt_span * (bins - 1))

        np.add.at(image[y_idx], x_idx, norm_log)
        np.add.at(counts[y_idx], x_idx, 1)

    counts[counts == 0] = 1
    image = image / counts

    out = out_dir / mzml_path.name.replace(".mzML", ".png")
    imsave(out, image, cmap="viridis", origin="lower")


def _pass1_worker(args):
    path, idx, total = args
    t0 = time.time()
    res = file_ranges(path)
    print(f"  [{idx + 1}/{total}] {path.name} ({time.time() - t0:.1f}s) "
          f"mz=[{res[0]:.1f},{res[1]:.1f}] rt=[{res[2]:.1f},{res[3]:.1f}] "
          f"n_filtered={res[6]}", flush=True)
    return path.name, res


def _pass2_worker(args):
    (path, out_dir, bins,
     mz_min, mz_max, rt_min, rt_max, log_min, log_max,
     idx, total) = args
    t0 = time.time()
    bin_with_global_ranges(path, out_dir, bins,
                           mz_min, mz_max, rt_min, rt_max,
                           log_min, log_max)
    print(f"  [{idx + 1}/{total}] {path.name} ({time.time() - t0:.1f}s)",
          flush=True)
    return path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mzml", required=True,
                    help="Directory containing classA/ and classB/ subdirs of .mzML files")
    ap.add_argument("--output", required=True,
                    help="Output directory for the global-normalized PNGs")
    ap.add_argument("--bins", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ranges-cache", default=None,
                    help="JSON file to cache per-file ranges (saves a re-run if pass 2 fails)")
    args = ap.parse_args()

    mzml_root = Path(args.mzml)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)





    all_files = []
    nested = (mzml_root / "classA").is_dir() or (mzml_root / "classB").is_dir()
    if nested:
        for cls in ("classA", "classB"):
            d = mzml_root / cls
            if not d.is_dir():
                print(f"  WARNING: {d} not found, skipping")
                continue
            for f in sorted(d.iterdir()):
                if f.suffix == ".mzML":
                    all_files.append(f)
    else:
        for f in sorted(mzml_root.iterdir()):
            if f.suffix == ".mzML":
                all_files.append(f)
    if not all_files:
        print(f"No mzML files found under {mzml_root}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(all_files)} mzML files.")


    cache = Path(args.ranges_cache) if args.ranges_cache else None
    if cache and cache.exists():
        print(f"Using cached pass-1 ranges from {cache}")
        with cache.open() as f:
            file_meta = {k: tuple(v) for k, v in json.load(f).items()}
    else:
        print("Pass 1: computing per-file axis and intensity ranges...")
        tasks = [(p, i, len(all_files)) for i, p in enumerate(all_files)]
        with Pool(args.workers) as pool:
            results = pool.map(_pass1_worker, tasks)
        file_meta = dict(results)
        if cache:
            with cache.open("w") as f:
                json.dump({k: list(v) for k, v in file_meta.items()}, f, indent=2)
            print(f"Cached ranges to {cache}")


    g_mz_min = min(v[0] for v in file_meta.values())
    g_mz_max = max(v[1] for v in file_meta.values())
    g_rt_min = min(v[2] for v in file_meta.values())
    g_rt_max = max(v[3] for v in file_meta.values())
    g_log_min = min(v[4] for v in file_meta.values())
    g_log_max = max(v[5] for v in file_meta.values())

    print("\nGlobal ranges:")
    print(f"  m/z: [{g_mz_min:.2f}, {g_mz_max:.2f}]")
    print(f"  RT : [{g_rt_min:.2f}, {g_rt_max:.2f}]")
    print(f"  log10(intensity): [{g_log_min:.3f}, {g_log_max:.3f}]")


    with (out_root / "global_ranges.json").open("w") as f:
        json.dump({
            "mz_min": g_mz_min, "mz_max": g_mz_max,
            "rt_min": g_rt_min, "rt_max": g_rt_max,
            "log_min": g_log_min, "log_max": g_log_max,
            "bins": args.bins,
            "n_files": len(all_files),
        }, f, indent=2)


    print("\nPass 2: binning to global grid...")
    pass2_args = [
        (p, out_root, args.bins,
         g_mz_min, g_mz_max, g_rt_min, g_rt_max, g_log_min, g_log_max,
         i, len(all_files))
        for i, p in enumerate(all_files)
    ]
    with Pool(args.workers) as pool:
        pool.map(_pass2_worker, pass2_args)

    print(f"\nDone. PNGs at {out_root}")


if __name__ == "__main__":
    main()
