"""Generate the occlusion-test figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = Path(os.environ.get("LCMS_FIGURES", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(RESULTS_DIR / "gradcam_occlusion_summary.csv")
agg = (summary.groupby("condition")[["accuracy", "roc_auc"]]
       .agg(["mean", "std"]).reset_index())

quadrants = ["TL", "TR", "BL", "BR"]
brightness = {"TL": 0.4653, "TR": 0.4402, "BL": 0.4770, "BR": 0.4383}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))


ax = axes[0]
labels = ["Full"] + [f"Mask {q}" for q in quadrants]
acc_means, acc_stds = [], []
for cond in ["full", "mask_TL", "mask_TR", "mask_BL", "mask_BR"]:
    row = agg[agg["condition"] == cond].iloc[0]
    acc_means.append(row[("accuracy", "mean")])
    acc_stds.append(row[("accuracy", "std")])
colors = ["#2E7D32", "#90CAF9", "#90CAF9", "#E53935", "#90CAF9"]
ax.bar(range(len(labels)), acc_means, yerr=acc_stds, color=colors,
       edgecolor="black", linewidth=0.8, capsize=4)
ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.7)
ax.set_ylim(0.0, 1.0)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Test accuracy")
ax.set_title("(a) Accuracy when each quadrant is masked", fontsize=11)
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(acc_means):
    ax.text(i, m + 0.04, f"{m:.2f}", ha="center", fontsize=8)


ax = axes[1]
labels_b = ["Full"] + [f"Only {q}" for q in quadrants]
aucs, auc_stds = [], []
for cond in ["full", "only_TL", "only_TR", "only_BL", "only_BR"]:
    row = agg[agg["condition"] == cond].iloc[0]
    aucs.append(row[("roc_auc", "mean")])
    auc_stds.append(row[("roc_auc", "std")])
colors_b = ["#2E7D32", "#FFB74D", "#FFB74D", "#E53935", "#FFB74D"]
ax.bar(range(len(labels_b)), aucs, yerr=auc_stds, color=colors_b,
       edgecolor="black", linewidth=0.8, capsize=4)
ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.7)
ax.set_ylim(0.0, 1.05)
ax.set_xticks(range(len(labels_b)))
ax.set_xticklabels(labels_b, rotation=20, ha="right")
ax.set_ylabel("Test AUC-ROC")
ax.set_title("(b) AUC when only one quadrant is shown", fontsize=11)
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(aucs):
    ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=8)


ax = axes[2]
acc_drops = {}
full_acc = agg[agg["condition"] == "full"].iloc[0][("accuracy", "mean")]
for q in quadrants:
    cond_acc = agg[agg["condition"] == f"mask_{q}"].iloc[0][("accuracy", "mean")]
    acc_drops[q] = full_acc - cond_acc

xs = [brightness[q] for q in quadrants]
ys = [acc_drops[q] for q in quadrants]
colors_c = {"TL": "#FFB300", "TR": "#1E88E5", "BL": "#E53935", "BR": "#43A047"}
for q in quadrants:
    ax.scatter(brightness[q], acc_drops[q], s=200, color=colors_c[q],
               edgecolor="black", zorder=3, label=q)
    ax.annotate(q, (brightness[q], acc_drops[q]),
                textcoords="offset points", xytext=(8, 4), fontsize=11,
                fontweight="bold")

m, b = np.polyfit(xs, ys, 1)
xx = np.linspace(min(xs) - 0.005, max(xs) + 0.005, 50)
ax.plot(xx, m * xx + b, ls="--", color="gray", lw=1,
        label=f"Linear fit (slope={m:+.2f})")
ax.set_xlabel("Mean quadrant intensity")
ax.set_ylabel("Accuracy drop when quadrant masked")
ax.set_title("(c) Discriminative impact vs brightness", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

fig.suptitle(
    "GradCAM-region occlusion control (ResNet-34, full-spectrum 512x512)",
    fontsize=12.5, fontweight="bold", y=1.02,
)
fig.tight_layout()
out = FIG_DIR / "gradcam_occlusion_test.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out}")
