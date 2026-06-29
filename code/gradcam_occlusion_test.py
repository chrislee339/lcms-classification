"""
GradCAM-region occlusion test: distinguish discriminative regions from
merely bright ones.

Re-runs inference with each of the four image quadrants masked out (zeroed)
in turn, plus an "only quadrant" control where the rest of the image is
zeroed. Three quantities are compared:

  1. accuracy/AUC drop when each quadrant is occluded (a discriminative
     quadrant causes the largest drop when masked out);
  2. mean image intensity in each quadrant (the brightness baseline);
  3. accuracy/AUC retained when only one quadrant is shown.

If attention merely tracked brightness, masking the brightest quadrant
would produce the largest drop. If the GradCAM-highlighted quadrant is
discriminative beyond its brightness, masking it drops accuracy more than
masking other, similarly bright quadrants.

Output:
  results/gradcam_occlusion.csv          per-image predictions per condition
  results/gradcam_occlusion_summary.csv  per-fold/seed accuracy by condition
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

import os
BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
KFOLD = BASE / "data/kfold/v2_512_by_patient_rmS11"
MODEL_BASE = BASE / "models/resnet_v2"
RESULTS_DIR = BASE / "results"

EXP = "fullspec_resnet34_512"
SEEDS = [42, 69, 137]
N_FOLDS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")







QUADRANTS = ["TL", "TR", "BL", "BR"]
CONDITIONS = ["full"] + [f"mask_{q}" for q in QUADRANTS] + [f"only_{q}" for q in QUADRANTS]


def quadrant_mask(h: int, w: int, name: str, mode: str) -> np.ndarray:
    """Return a {0,1} mask of shape (h, w). mode='mask' zeroes the named
    quadrant; mode='only' keeps only the named quadrant."""
    m = np.ones((h, w), dtype=np.float32)
    rows = slice(0, h // 2) if name in ("TL", "TR") else slice(h // 2, h)
    cols = slice(0, w // 2) if name in ("TL", "BL") else slice(w // 2, w)
    if mode == "mask":
        m[rows, cols] = 0.0
    elif mode == "only":
        m_keep = np.zeros((h, w), dtype=np.float32)
        m_keep[rows, cols] = 1.0
        m = m_keep
    return m


class LCMSDataset(Dataset):
    """Returns the resized 224x224 image (no ImageNet normalization yet) as
    a [3, 224, 224] tensor in [0,1], plus the label and filename. Occlusion
    is applied AFTER this in the inference loop."""

    def __init__(self, img_dir: Path):
        self.img_dir = Path(img_dir)
        self.files = sorted(p.name for p in self.img_dir.iterdir()
                            if p.suffix == ".png")
        self.labels = []
        for f in self.files:
            base = Path(f).stem
            if "_aug" in base:
                base = base.split("_aug")[0]
            self.labels.append(0 if base[-1] == "A" else 1)
        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_dir / self.files[idx]).convert("RGB")
        return self.resize(img), self.labels[idx], self.files[idx]


class TransferResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)



IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def occlude(images: torch.Tensor, condition: str) -> torch.Tensor:
    """Apply occlusion to a [B, 3, 224, 224] tensor in [0,1].

    Conditions:
      'full'       : no occlusion
      'mask_<Q>'   : zero the named quadrant
      'only_<Q>'   : keep only the named quadrant
    """
    if condition == "full":
        return images
    parts = condition.split("_")
    mode, qname = parts[0], parts[1]
    h, w = images.shape[-2], images.shape[-1]
    mask = quadrant_mask(h, w, qname, mode)
    mask_t = torch.from_numpy(mask).to(images.device)
    return images * mask_t


def quadrant_brightness(images: torch.Tensor) -> dict:
    """Mean intensity per quadrant for a batch (averaged across batch and
    channel)."""
    h, w = images.shape[-2], images.shape[-1]
    out = {}
    for q in QUADRANTS:
        m = torch.from_numpy(quadrant_mask(h, w, q, "only")).to(images.device)

        keep_pix = m.sum().item()
        if keep_pix == 0:
            out[q] = float("nan")
        else:
            out[q] = float((images * m).sum().item()
                           / (images.shape[0] * images.shape[1] * keep_pix))
    return out


def main() -> None:
    records = []
    brightness_records = []
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            test_dir = KFOLD / f"fold_{fold}" / "test"
            if not test_dir.is_dir():
                print(f"  SKIP: {test_dir}")
                continue
            ds = LCMSDataset(test_dir)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

            model_dir = MODEL_BASE / f"{EXP}_seed{seed}_seed{seed}_fold{fold}"
            ckpt = model_dir / "best_model_1.pt"
            if not ckpt.is_file():
                print(f"  SKIP: {ckpt}")
                continue

            model = TransferResNet().to(DEVICE)
            sd = torch.load(ckpt, map_location=DEVICE)
            if any(k.startswith("module.") for k in sd):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
            model.eval()

            with torch.no_grad():
                for images, labels, filenames in loader:
                    images = images.to(DEVICE)


                    bright = quadrant_brightness(images)
                    for q in QUADRANTS:
                        brightness_records.append({
                            "seed": seed, "fold": fold, "quadrant": q,
                            "mean_intensity": bright[q],
                            "n_images": images.shape[0],
                        })

                    for cond in CONDITIONS:
                        occluded = occlude(images, cond)
                        x = normalize(occluded)
                        logits = model(x)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        for i, fname in enumerate(filenames):
                            records.append({
                                "seed": seed,
                                "fold": fold,
                                "condition": cond,
                                "filename": fname,
                                "y_true": int(labels[i]),
                                "y_prob": float(probs[i]),
                            })

            print(f"  seed={seed} fold={fold} done ({len(ds)} images x {len(CONDITIONS)} conds)")

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "gradcam_occlusion.csv"
    df.to_csv(out, index=False)
    print(f"Wrote per-image predictions: {out} ({len(df)} rows)")

    bright_df = pd.DataFrame(brightness_records).groupby("quadrant")["mean_intensity"].mean()
    print("\nMean intensity per quadrant (averaged over all test images):")
    print(bright_df.round(4).to_string())


    summary = []
    for (seed, fold, cond), grp in df.groupby(["seed", "fold", "condition"]):
        y_true = grp["y_true"].values
        y_prob = grp["y_prob"].values
        y_pred = (y_prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        summary.append({
            "seed": seed, "fold": fold, "condition": cond,
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": auc,
            "n": len(y_true),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "gradcam_occlusion_summary.csv", index=False)


    agg = summary_df.groupby("condition")[["accuracy", "roc_auc"]].agg(["mean", "std"])
    print("\nMean accuracy / AUC by condition (mean +/- std across 18 seed-fold pairs):")
    out_lines = []
    for cond in CONDITIONS:
        row = agg.loc[cond]
        line = (f"  {cond:>10s}: acc={row[('accuracy','mean')]:.3f} +/- "
                f"{row[('accuracy','std')]:.3f}, "
                f"auc={row[('roc_auc','mean')]:.3f} +/- "
                f"{row[('roc_auc','std')]:.3f}")
        print(line)
        out_lines.append(line)


if __name__ == "__main__":
    main()
