"""Per-run MS-level and scan-rate summary from indexed mzML scan headers.

Reads only each spectrum's header via the mzML index (a few KB per scan), so it never loads a run
into memory. For each run it reports the scans per MS level, the most peaks in any scan of that level,
how many scans of each level pass the image generator's peak filter (code/bin_spectra_per_sample.py:
cutoff = max(5000, 10th percentile of all scans' peak counts)), and the acquisition rate.

usage: python3 mzml_scan_audit.py out.csv file1.mzML [file2.mzML ...]
"""
import csv
import re
import statistics
import sys

import numpy as np


def scan_headers(path):
    with open(path, "rb") as f:
        f.seek(-4000, 2)
        offset = int(re.search(rb"<indexListOffset>(\d+)<", f.read()).group(1))
        f.seek(offset)
        index = f.read().split(b'<index name="chromatogram"')[0]
        rows = []
        for o in map(int, re.findall(rb'<offset idRef="[^"]*">(\d+)</offset>', index)):
            f.seek(o)
            head = f.read(4000)
            peaks = int(re.search(rb'defaultArrayLength="(\d+)"', head).group(1))
            level = int(re.search(rb'name="ms level" value="(\d+)"', head).group(1))
            m = re.search(rb'name="scan start time" value="([0-9.eE+-]+)"[^>]*unitName="(\w+)"', head)
            t = float(m.group(1)) * (60.0 if m.group(2) == b"minute" else 1.0)
            rows.append((t, level, peaks))
    return sorted(rows)


def audit(path):
    rows = scan_headers(path)
    counts = np.array([p for _, _, p in rows])
    cutoff = max(5000, np.percentile(counts, 10))
    times = [t for t, _, _ in rows]
    ms1_times = [t for t, lvl, _ in rows if lvl == 1]
    out = {
        "run": path.rsplit("/", 1)[-1].replace(".mzML", ""),
        "n_spectra": len(rows),
        "cutoff": cutoff,
        "duration_min": round((times[-1] - times[0]) / 60, 2),
        "spectra_per_s": round(len(rows) / (times[-1] - times[0]), 3),
        "median_interval_ms": round(1000 * statistics.median(np.diff(times)), 1),
        "median_ms1_interval_ms": round(1000 * statistics.median(np.diff(ms1_times)), 1),
    }
    for level in (1, 2):
        peaks = [p for _, lvl, p in rows if lvl == level]
        out[f"ms{level}_n"] = len(peaks)
        out[f"ms{level}_max_peaks"] = max(peaks) if peaks else 0
        out[f"ms{level}_passing"] = sum(p >= cutoff for p in peaks)
    out["other_levels_n"] = sum(lvl not in (1, 2) for _, lvl, _ in rows)
    return out


if __name__ == "__main__":
    out_csv, paths = sys.argv[1], sys.argv[2:]
    results = []
    for p in paths:
        results.append(audit(p))
        print(results[-1], flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
