"""Train and evaluate one (candidate, outer fold, seed) unit of the nested cross-validation on 38 patients (S11 always
in a training set).

Runs run_single_experiment of the original training code (code/run_all_experiments.py) unchanged on the 5 inner folds
of one outer fold (nested_build_inner_folds.py), then writes each inner-test sample's prediction from best_model_1
(single model) and the mean of best_model_1..3, the same way evaluate_checkpoints.py does for the outer folds. Checks
that the recomputed single-model accuracy matches what the training code logged for every inner fold.
Writes $NESTED/preds/<candidate>__outer<k>_seed<s>.csv (only once the unit is complete) and a _folds.json beside it;
checkpoints go to $NESTED/models/. The full run is 10 candidates x 6 outer folds x 3 seeds (42, 69, 137) = 180 units.

NESTED_SMOKE=1 runs 1 inner fold for 3 epochs into $NESTED/smoke/ (a test of the plumbing only).
LCMS_BASE (default: the repository root), NESTED (default $LCMS_BASE/nested) and LCMS_CODE (the folder holding
run_all_experiments.py, default code/) set the locations.

usage: python3 nested_train_inner.py <candidate> <outer fold> <seed>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
NESTED = os.environ.get("NESTED", os.path.join(BASE, "nested"))
sys.path.insert(0, os.environ.get("LCMS_CODE", str(Path(__file__).resolve().parents[1])))
import run_all_experiments as base  # noqa: E402

CANDIDATES = {  # name (as in the outer S11-in-training runs): (variant, inner k-fold source)
    **{f"arch_{v}_256_withS11rot": (v, "256") for v in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")},
    "fullspec_resnet34_512_withS11rot": ("resnet34", "512"),
    "fullspec_resnet34_1024_withS11rot": ("resnet34", "1024"),
    **{f"subregion_resnet34_{r}_withS11rot": ("resnet34", f"sub{r}") for r in (256, 512, 1024)},
}

name, outer, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
smoke = os.environ.get("NESTED_SMOKE") == "1"
root = os.path.join(NESTED, "smoke") if smoke else NESTED
variant, source = CANDIDATES[name]
kfold_dir = os.path.join(NESTED, "kfold", source, f"outer{outer}")
assert os.path.exists(os.path.join(NESTED, "kfold", source, "BUILD_DONE")), kfold_dir

base.N_FOLDS = 1 if smoke else 5
if smoke:
    base.NUM_EPOCHS, base.WARMUP_EPOCHS = 3, 1  # 3 epochs so that 3 checkpoints exist
base.MODEL_BASE = os.path.join(root, "models")
unit = f"{name}__outer{outer}"
out_csv = os.path.join(root, "preds", f"{unit}_seed{seed}.csv")
if os.path.exists(out_csv):
    print("already done:", out_csv)
    sys.exit(0)
for j in range(base.N_FOLDS):  # leftovers of an interrupted attempt
    shutil.rmtree(os.path.join(base.MODEL_BASE, f"{unit}_seed{seed}_fold{j}"), ignore_errors=True)

base.log(f"nested CV unit: {name}, outer fold {outer}, seed {seed}, {kfold_dir}, {base.N_FOLDS} inner folds")
fold_results = base.run_single_experiment(variant, kfold_dir, seed, unit)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
criterion = nn.BCEWithLogitsLoss()


def load(path):
    model = base.TransferResNet(variant=variant)
    model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(path, map_location=device).items()},
                          strict=True)
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


rows, val = [], []
for j in range(base.N_FOLDS):
    ckpt = os.path.join(base.MODEL_BASE, f"{unit}_seed{seed}_fold{j}")
    models = [load(os.path.join(ckpt, f"best_model_{i}.pt")) for i in (1, 2, 3)]
    loaders = {s: DataLoader(base.LCMSDataset(os.path.join(kfold_dir, f"fold_{j}", s), transform=transform),
                             batch_size=base.BATCH_SIZE_EVAL, shuffle=False, num_workers=0) for s in ("val", "test")}
    _, _, _, vloss = run(models[0], loaders["val"])
    p1, y, files, _ = run(models[0], loaders["test"])
    pens = np.mean([p1] + [run(m, loaders["test"])[0] for m in models[1:]], axis=0)
    acc = float(((p1 > 0.5) == y).mean())
    assert abs(acc - fold_results[j]["accuracy"]) < 1e-9, (j, acc, fold_results[j]["accuracy"])
    val.append({"inner_fold": j, "val_loss": vloss, "harness_accuracy": fold_results[j]["accuracy"],
                "harness_auc": fold_results[j]["roc_auc"]})
    rows += [{"candidate": name, "outer": outer, "inner_fold": j, "seed": seed, "filename": f, "y_true": int(t),
              "y_prob": a, "y_prob_ens": b} for f, t, a, b in zip(files, y, p1, pens)]

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv.replace(".csv", "_folds.json"), "w") as fh:
    json.dump(val, fh, indent=1)
pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
os.replace(out_csv + ".tmp", out_csv)  # the .csv appears only when the unit is complete
base.log(f"wrote {out_csv} ({len(rows)} inner-test predictions)")
