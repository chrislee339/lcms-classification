"""Evaluate saved checkpoints: per-sample test predictions and validation loss.

For every seed and fold of one configuration, loads best_model_1..3.pt from
<model_base>/<name>_seed<s>_seed<s>_fold<k>/ (the top 3 checkpoints by validation loss, as kept by the original
training code, code/run_all_experiments.py) and writes
  <out>/<name>_predictions.csv  fold, seed, filename, y_true, y_prob (best_model_1), y_prob_ens (mean of 3)
  <out>/<name>_valloss.csv      seed, fold, val_loss, val_acc, n_val  (best_model_1 on that fold's
                                validation patients, loss summed over batches of 8 exactly as the training
                                code does when it ranks checkpoints)
Uses the training code's own dataset class, model class and transform, so nothing is re-implemented.

usage: python3 evaluate_checkpoints.py <name> <variant> <kfold_dir> <model_base> <out_dir> [--cpu]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_all_experiments as base  # noqa: E402

name, variant, kfold_dir, model_base, out_dir = sys.argv[1:6]
device = torch.device("cpu" if "--cpu" in sys.argv or not torch.cuda.is_available() else "cuda")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
criterion = nn.BCEWithLogitsLoss()


def load(path):
    model = base.TransferResNet(variant=variant)
    state = {k.replace("module.", ""): v for k, v in torch.load(path, map_location=device).items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run(model, loader):
    probs, labels, files, loss = [], [], [], 0.0
    with torch.no_grad():
        for images, y, f in loader:
            out = model(images.to(device))
            loss += criterion(out, y.float().to(device)).item()
            probs += torch.sigmoid(out).cpu().numpy().ravel().tolist()
            labels += y.numpy().ravel().tolist()
            files += list(f)
    return np.array(probs), np.array(labels), files, loss


pred_rows, val_rows = [], []
for seed in base.SEEDS:
    for fold in range(base.N_FOLDS):
        ckpt_dir = os.path.join(model_base, f"{name}_seed{seed}_seed{seed}_fold{fold}")
        models = [load(os.path.join(ckpt_dir, f"best_model_{i}.pt")) for i in (1, 2, 3)]
        loaders = {split: DataLoader(base.LCMSDataset(os.path.join(kfold_dir, f"fold_{fold}", split), transform=transform),
                                     batch_size=base.BATCH_SIZE_EVAL, shuffle=False, num_workers=0)
                   for split in ("val", "test")}
        vp, vy, _, vloss = run(models[0], loaders["val"])
        val_rows.append({"seed": seed, "fold": fold, "val_loss": vloss,
                         "val_acc": float(((vp > 0.5) == vy).mean()), "n_val": len(vy)})
        p1, y, files, _ = run(models[0], loaders["test"])
        pens = np.mean([p1] + [run(m, loaders["test"])[0] for m in models[1:]], axis=0)
        pred_rows += [{"fold": fold, "seed": seed, "filename": f, "y_true": int(t), "y_prob": a, "y_prob_ens": b}
                      for f, t, a, b in zip(files, y, p1, pens)]
        print(f"{name} seed {seed} fold {fold}: val_loss {vloss:.4f}", flush=True)

os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(pred_rows).to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)
pd.DataFrame(val_rows).to_csv(os.path.join(out_dir, f"{name}_valloss.csv"), index=False)
print("wrote", os.path.join(out_dir, f"{name}_predictions.csv"), "and _valloss.csv")
