"""
Model-independent image-level QC criterion hunt for patient S11.

Computes per-image QC metrics on the 512x512 binned spectra (independent
of any trained model) and flags outliers. Goal: see if S11 is identifiable
as anomalous from the raw image alone, before any model training.

QC metrics computed per image:
  - total_intensity:    sum of all pixel intensities (proxy for total ion current)
  - sparsity:           fraction of pixels with intensity < 0.01
  - peak_density:       fraction of pixels above 0.5 (bright peaks)
  - mean_intensity:     mean pixel intensity
  - std_intensity:      std of pixel intensity
  - shannon_entropy:    H over the intensity histogram (32 bins)
  - centroid_x / y:     intensity-weighted centroid (in [0,1])
  - centroid_radial:    distance of centroid from image center
  - low_mz_low_rt_frac: fraction of total signal in the bottom-left quadrant
  - off_diagonal_ratio: signal in the GradCAM region vs the rest
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import argparse
import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
_DEFAULT_IMG = REPO_ROOT / "data/raw/spectra_filter_512"
_DEFAULT_OUT = REPO_ROOT / "results"
_ap = argparse.ArgumentParser()
_ap.add_argument("--img-dir", default=str(_DEFAULT_IMG))
_ap.add_argument("--out-dir", default=str(_DEFAULT_OUT))
_args, _ = _ap.parse_known_args()
IMG_DIR = Path(_args.img_dir)
OUT_DIR = Path(_args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load an image as a normalized [0,1] grayscale array.

    The binned spectra are saved as RGB but the spectral signal is on the
    intensity channel; we collapse to luminance.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_metrics(arr: np.ndarray) -> dict:
    """Compute model-independent QC metrics for a single image.

    Convention for the binned spectra: x-axis = m/z (low to high, left to right);
    y-axis with origin at top-left, but spectra are typically rendered with
    low retention time at the top and high RT at the bottom. The "bottom-left"
    that GradCAM highlights corresponds to (low m/z, low RT) which in image
    coordinates is the TOP-LEFT quadrant of a numpy array. We use that
    convention here.
    """
    h, w = arr.shape
    total = float(arr.sum())
    nz = arr > 0.01

    metrics = {
        "h": h,
        "w": w,
        "total_intensity": total,
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "sparsity": float((arr < 0.01).mean()),
        "peak_density": float((arr > 0.5).mean()),
    }


    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    metrics["shannon_entropy"] = float(-np.sum(p * np.log2(p)))


    if total > 0:
        ys, xs = np.indices(arr.shape)
        cx = float((arr * xs).sum() / total) / (w - 1)
        cy = float((arr * ys).sum() / total) / (h - 1)
    else:
        cx = cy = 0.5
    metrics["centroid_x"] = cx
    metrics["centroid_y"] = cy
    metrics["centroid_radial"] = float(np.hypot(cx - 0.5, cy - 0.5))


    q = arr[: h // 2, : w // 2]
    metrics["low_mz_low_rt_frac"] = float(q.sum() / total) if total > 0 else 0.0

    return metrics


def main() -> None:

    files = sorted(p for p in IMG_DIR.iterdir()
                   if p.suffix == ".png" and "-" not in p.stem)
    rows = []
    for f in files:
        stem = f.stem
        cls = stem[-1]
        if cls not in ("A", "B"):
            continue
        patient = stem[:-1]
        arr = load_image(f)
        metrics = compute_metrics(arr)
        metrics["patient"] = patient
        metrics["class"] = cls
        metrics["filename"] = f.name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} images covering {df['patient'].nunique()} patients.")
    out_per_image = OUT_DIR / "s11_qc_per_image.csv"
    df.to_csv(out_per_image, index=False)
    print(f"Per-image metrics: {out_per_image}")


    metric_cols = [
        "total_intensity", "mean_intensity", "std_intensity",
        "sparsity", "peak_density", "shannon_entropy",
        "centroid_x", "centroid_y", "centroid_radial",
        "low_mz_low_rt_frac",
    ]
    pat_df = df.groupby("patient")[metric_cols].mean().reset_index()

    z = pat_df.copy()
    for col in metric_cols:
        mu = pat_df[col].mean()
        sd = pat_df[col].std(ddof=0)
        z[f"{col}_z"] = (pat_df[col] - mu) / sd if sd > 0 else 0.0

    out_per_patient = OUT_DIR / "s11_qc_per_patient.csv"
    z.to_csv(out_per_patient, index=False)
    print(f"Per-patient metrics+z-scores: {out_per_patient}")


    z_cols = [c for c in z.columns if c.endswith("_z")]
    z["max_abs_z"] = z[z_cols].abs().max(axis=1)
    z["max_abs_z_metric"] = z[z_cols].abs().idxmax(axis=1)
    ranking = z.sort_values("max_abs_z", ascending=False)[
        ["patient", "max_abs_z", "max_abs_z_metric"] + z_cols
    ]
    print("\nTop 8 patients by max |z| across QC metrics (S11 italicized if present):")
    top = ranking.head(8).copy()
    for col in z_cols:
        top[col] = top[col].round(2)
    top["max_abs_z"] = top["max_abs_z"].round(2)
    print(top.to_string(index=False))


    s11 = z[z["patient"] == "S11"].iloc[0]
    print("\nS11 detail:")
    for col in metric_cols:
        raw = s11[col]
        zval = s11[f"{col}_z"]
        rank = (z[col].rank().loc[s11.name])
        print(f"  {col:>22s}: raw={raw: .4f}  z={zval:+.2f}  rank={int(rank)}/{len(z)}")


    for col in metric_cols:
        flagged = z[z[f"{col}_z"].abs() > 3]["patient"].tolist()
        if flagged:
            print(f"\n|z|>3 on {col}: {flagged}")
    flagged_2_5 = z[(z[z_cols].abs() > 2.5).any(axis=1)]["patient"].tolist()
    print(f"\nPatients with any |z|>2.5: {flagged_2_5}")


if __name__ == "__main__":
    main()
