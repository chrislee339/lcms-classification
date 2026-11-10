"""GradCAM stability of the 38-patient full-spectrum 512 x 512 ResNet-34 models.

Runs the GradCAM code of code/run_gradcam_analysis.py (layer4, true-class target, ReLU, max-scaled), unchanged, on
every test sample of the 38-patient full-spectrum 512 model, for all three seeds, misclassified samples
included. Keeps each map's native 7 x 7 grid (recomputed from the stored activations and gradients, the
same computation before upsampling) and the 224 x 224 upsampled map.

GC_MODELS (checkpoint folder pattern, {s} = seed, {f} = fold) and GC_KFOLD (test folder pattern) choose the models.
The defaults are the runs used in the paper, in which S11 is always in a training set:
  GC_MODELS=$LCMS_BASE/models/resnet_v2/fullspec_resnet34_512_withS11rot_seed{s}_seed{s}_fold{f}
  GC_KFOLD=$LCMS_BASE/data/kfold/v2_512_by_patient_withS11_rot/fold_{f}/test

Outputs in <out>: maps.npz and maps_meta.csv (native and upsampled maps, per-map metadata), stability_summary.csv,
location_per_map.csv, location_summary.csv, mean_maps.npz (mean maps for Figure 3).

usage: python3 gradcam_stability.py <out dir>
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gradcam_analysis as rga  # noqa: E402  (original GradCAM code, unchanged)

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
OUT = sys.argv[1]
MODELS = os.environ.get("GC_MODELS",  # GC_MODELS / GC_KFOLD choose the runs (see above)
                        os.path.join(BASE, "models/resnet_v2/fullspec_resnet34_512_withS11rot_seed{s}_seed{s}_fold{f}"))
KFOLD = os.environ.get("GC_KFOLD", os.path.join(BASE, "data/kfold/v2_512_by_patient_withS11_rot/fold_{f}/test"))
SEEDS, N_FOLDS, N_PERM, TOP = (42, 69, 137), 6, 1000, 10
rng = np.random.default_rng(2026)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT, exist_ok=True)

# ---- 1. maps -----------------------------------------------------------------------------------
meta, native, full = [], [], []
transform = rga.get_transform()
for seed in SEEDS:
    for fold in range(N_FOLDS):
        model = rga.load_model(MODELS.format(s=seed, f=fold), "resnet34", device)
        cam = rga.GradCAM(model, model.backbone.layer4)
        test_dir = KFOLD.format(f=fold)
        for fname in sorted(f for f in os.listdir(test_dir) if f.endswith(".png")):
            y = rga.get_label_from_filename(fname)
            x = transform(Image.open(os.path.join(test_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
            up, logit = cam(x, target_class=y)
            w = cam.gradients.mean(dim=(2, 3), keepdim=True)
            nat = F.relu((w * cam.activations).sum(dim=1)).squeeze().detach().cpu().numpy()
            nat = nat / nat.max() if nat.max() > 0 else nat
            prob = 1 / (1 + np.exp(-logit))
            meta.append({"filename": fname, "patient": fname[:-5], "y_true": y, "seed": seed, "fold": fold,
                         "prob": prob, "correct": (prob > 0.5) == (y == 1), "all_zero": nat.max() == 0})
            native.append(nat)
            full.append(up)
        cam.remove_hooks()
meta = pd.DataFrame(meta)
native, full = np.array(native), np.array(full)
assert native.shape[1:] == (7, 7) and len(meta) == 228, (native.shape, len(meta))
np.savez_compressed(f"{OUT}/maps.npz", native=native, full=full)
meta.to_csv(f"{OUT}/maps_meta.csv", index=False)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return np.nan if d == 0 else float((a * b).sum() / d)


def iou(a, b, k=TOP):
    ta, tb = set(np.argsort(a.ravel())[-k:]), set(np.argsort(b.ravel())[-k:])
    return len(ta & tb) / len(ta | tb)


def null(pairs):
    """Same statistics after randomly permuting the cells of the second map."""
    rs, js = [], []
    for a, b in pairs:
        for _ in range(N_PERM):
            pb = rng.permutation(b.ravel()).reshape(b.shape)
            rs.append(corr(a, pb))
            js.append(iou(a, pb))
    return np.array(rs), np.array(js)


def describe(name, pairs, subset):
    r = np.array([corr(a, b) for a, b in pairs])
    j = np.array([iou(a, b) for a, b in pairs])
    nr, nj = null(pairs)
    r, nr = r[~np.isnan(r)], nr[~np.isnan(nr)]
    return {"comparison": name, "subset": subset, "n_pairs": len(pairs),
            "r_median": np.median(r), "r_q1": np.percentile(r, 25), "r_q3": np.percentile(r, 75),
            "null_r_median": np.median(nr), "null_r_95th": np.percentile(nr, 95),
            "share_above_null_95th": np.mean(r > np.percentile(nr, 95)),
            "iou_median": np.median(j), "null_iou_median": np.median(nj), "null_iou_95th": np.percentile(nj, 95)}


rows = []
# ---- 2. seed stability: same sample, different seeds ---------------------------------------------
by_sample = {f: native[meta.index[meta.filename == f]] for f in meta.filename.unique()}
wrong_any = set(meta.filename[~meta.correct])
for subset, keep in (("all samples", lambda f: True), ("correct in all seeds", lambda f: f not in wrong_any),
                     ("misclassified by at least one seed", lambda f: f in wrong_any)):
    pairs = [(m[i], m[j]) for f, m in by_sample.items() if keep(f) for i, j in itertools.combinations(range(3), 2)]
    if pairs:
        rows.append(describe("same sample, different seeds", pairs, subset))

# ---- 3-4. cross-patient agreement on seed-averaged maps ------------------------------------------
avg = meta.groupby("filename").agg(patient=("patient", "first"), y=("y_true", "first"), prob=("prob", "mean")).reset_index()
avg["correct"] = (avg.prob > 0.5) == (avg.y == 1)
avg_map = {f: by_sample[f].mean(0) for f in avg.filename}
for subset, sel in (("all samples", avg), ("correct (seed-averaged)", avg[avg.correct])):
    A, B = sel[sel.y == 0], sel[sel.y == 1]
    within = lambda S: [(avg_map[a], avg_map[b]) for a, b in itertools.combinations(S.filename, 2)]
    rows.append(describe("different patients, adjacent normal", within(A), subset))
    rows.append(describe("different patients, tumor", within(B), subset))
    between = [(avg_map[a], avg_map[b]) for a in A.filename for b in B.filename
               if A.set_index("filename").patient[a] != B.set_index("filename").patient[b]]
    rows.append(describe("different patients, adjacent normal vs tumor", between, subset))
wrong = avg[~avg.correct]
mixed = [(avg_map[w], avg_map[c]) for w in wrong.filename for c in avg[avg.correct & (avg.y == avg.set_index("filename").y[w])].filename]
if mixed:
    rows.append(describe("misclassified sample vs correctly classified samples of its class", mixed, "misclassified"))
pd.DataFrame(rows).to_csv(f"{OUT}/stability_summary.csv", index=False)

# ---- 6. location: attention inside the bottom-left quadrant ---------------------------------------
# images are saved with origin="lower", so the bottom half of the array is low retention time
h, w = full.shape[1:]
bl = (slice(h // 2, h), slice(0, w // 2))
loc = meta.copy()
loc["bl_share"] = [m[bl].sum() / m.sum() if m.sum() > 0 else np.nan for m in full]
loc["peak_in_bl"] = [np.unravel_index(m.argmax(), m.shape)[0] >= h // 2 and np.unravel_index(m.argmax(), m.shape)[1] < w // 2
                     for m in full]
loc.to_csv(f"{OUT}/location_per_map.csv", index=False)
summ = []
for subset, sel in (("all maps", loc), ("correct", loc[loc.correct]), ("misclassified", loc[~loc.correct])):
    for cls, g in (("both", sel), ("adjacent normal", sel[sel.y_true == 0]), ("tumor", sel[sel.y_true == 1])):
        summ.append({"subset": subset, "class": cls, "n_maps": len(g), "bl_share_median": g.bl_share.median(),
                     "bl_share_q1": g.bl_share.quantile(0.25), "bl_share_q3": g.bl_share.quantile(0.75),
                     "peak_in_bl": g.peak_in_bl.mean(), "all_zero_maps": int(g.all_zero.sum())})
pd.DataFrame(summ).to_csv(f"{OUT}/location_summary.csv", index=False)

# ---- 8. mean maps for Figure 3 (all samples, and correctly classified samples only) ----------------
avg_full = {f: full[meta.index[meta.filename == f]].mean(0) for f in avg.filename}
np.savez_compressed(f"{OUT}/mean_maps.npz",
                    all_A=np.mean([avg_full[f] for f in avg.filename[avg.y == 0]], 0),
                    all_B=np.mean([avg_full[f] for f in avg.filename[avg.y == 1]], 0),
                    correct_A=np.mean([avg_full[f] for f in avg.filename[(avg.y == 0) & avg.correct]], 0),
                    correct_B=np.mean([avg_full[f] for f in avg.filename[(avg.y == 1) & avg.correct]], 0))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(summ).round(3).to_string(index=False))
