"""Train ResNet-34 on the globally-normalized 512x512 LC-MS images and
write detailed/summary CSVs comparable to the main per-sample-normalized
results.

PRE-REQUISITES:
  1. Raw mzML files must be present at data/mzml/{classA,classB}/.
  2. Run:
       python3 code/bin_spectra_global_norm.py \
           --mzml data/mzml --output data/raw/spectra_filter_global_512 \
           --bins 512
  3. Run this script. It builds the S11-excluded kfold split (using
     patient_splits.json) and trains.

Outputs:
  results/fullspec_resnet34_512_globalnorm_detailed.csv
  results/fullspec_resnet34_512_globalnorm_summary.csv

The headline per-sample-normalized result lives in
  results/fullspec_resnet34_512_detailed.csv

so the comparison is a paired test on (seed, fold) between these two CSVs.
Both sides exclude S11.
"""
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all_experiments as base

RAW = BASE / "data/raw/spectra_filter_global_512"
KFOLD = BASE / "data/kfold/v2_512_by_patient_globalnorm_rmS11"
PATIENT_SPLITS = BASE / "data/kfold/patient_splits.json"

EXCLUDE_FILES = {"Fetuin_final.png", "S2B-5x-0302.png"}
N_FOLDS = 6
N_AUGMENTS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_kfold() -> None:
    if not RAW.is_dir() or not any(RAW.iterdir()):
        log(f"ERROR: globally-normalized PNGs not found at {RAW}.")
        log("       Run bin_spectra_global_norm.py first.")
        sys.exit(1)

    if (KFOLD / "fold_5" / "test").is_dir() and any((KFOLD / "fold_5" / "test").iterdir()):
        log(f"  Kfold dir already exists: {KFOLD}")
        return

    log(f"  Building kfold dir: {KFOLD}")
    with PATIENT_SPLITS.open() as f:
        splits = {int(k): {s: set(v) for s, v in d.items()} for k, d in json.load(f).items()}

    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    noise_fn = transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))

    class RandomSubsetAug:
        def __init__(self, tlist, min_ops=1, max_ops=3):
            self.tlist, self.min_ops, self.max_ops = tlist, min_ops, max_ops
        def __call__(self, img):
            n = random.randint(self.min_ops, self.max_ops)
            return transforms.Compose(random.sample(self.tlist, n))(img)

    aug_pipeline = transforms.Compose([
        transforms.ToTensor(),
        RandomSubsetAug([color_jitter, gaussian_blur, noise_fn]),
        transforms.ToPILImage(),
    ])

    available = {}
    for f in os.listdir(RAW):
        if not f.endswith(".png") or f in EXCLUDE_FILES:
            continue
        stem = f[:-4]
        if "-" in stem:
            continue
        if stem[:-1] in base.EXCLUDE_PATIENTS:
            continue
        available[stem] = RAW / f

    for fold in range(N_FOLDS):
        for split in ("train", "val", "test"):
            d = KFOLD / f"fold_{fold}" / split
            d.mkdir(parents=True, exist_ok=True)
            patients = splits[fold][split]
            for sample, src in available.items():
                if sample[:-1] not in patients:
                    continue
                shutil.copy2(src, d / f"{sample}.png")
                if split == "train":
                    img = Image.open(src).convert("RGB")
                    for k in range(1, N_AUGMENTS + 1):
                        seed = 69 + k + hash(sample) % 10000
                        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
                        aug = aug_pipeline(img)
                        aug.save(d / f"{sample}_aug{k}.png")
    log(f"  Done: {KFOLD}")


def main() -> None:
    log("Global-normalization head-to-head: ResNet-34 / 512x512 (S11 excluded)")
    log("=" * 60)
    build_kfold()
    experiments = [{
        "name": "fullspec_resnet34_512_globalnorm",
        "variant": "resnet34",
        "image_dir": str(RAW),
        "kfold_dir": str(KFOLD),
    }]
    base.run_experiment_group(experiments, active_seeds=base.SEEDS)
    log("\nGlobal-norm training complete. Compare against "
        "results/fullspec_resnet34_512_detailed.csv (per-sample baseline).")


if __name__ == "__main__":
    main()
