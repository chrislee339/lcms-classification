"""Build kfold directories with S11 included.

Mirrors prepare_kfold_dirs.py but reads patient_splits_withS11.json and
includes S11 in the cohort. Outputs to v2_<res>_by_patient_withS11/ and the
subregion and global-normalization variants listed in CONFIGS.
"""
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RAW_BASE = BASE / "data/raw"
KFOLD_BASE = BASE / "data/kfold"
SPLITS_JSON = KFOLD_BASE / "patient_splits_withS11.json"

N_FOLDS = 6
N_AUGMENTS = 20
EXCLUDE_FILES = {"Fetuin_final.png", "S2B-5x-0302.png"}


CONFIGS = [
    ("spectra_filter_256", "v2_256_by_patient_withS11"),
    ("spectra_filter_512", "v2_512_by_patient_withS11"),
    ("spectra_filter_1024", "v2_1024_by_patient_withS11"),
    ("focused_zoom_1024", "v2_subregion_1024_by_patient_withS11"),
    ("spectra_filter_global_512", "v2_512_by_patient_globalnorm_withS11"),
    ("focused_zoom_512", "v2_subregion_512_by_patient_withS11"),
    ("focused_zoom_256", "v2_subregion_256_by_patient_withS11"),
]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    with SPLITS_JSON.open() as f:
        splits = {int(k): {s: set(v) for s, v in d.items()} for k, d in json.load(f).items()}

    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    noise_fn = transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))

    class RandomSubsetAug:
        def __init__(self, tlist, min_ops=1, max_ops=3):
            self.tlist = tlist
            self.min_ops = min_ops
            self.max_ops = max_ops
        def __call__(self, img):
            n = random.randint(self.min_ops, self.max_ops)
            sel = random.sample(self.tlist, n)
            return transforms.Compose(sel)(img)

    aug_pipeline = transforms.Compose([
        transforms.ToTensor(),
        RandomSubsetAug([color_jitter, gaussian_blur, noise_fn]),
        transforms.ToPILImage(),
    ])

    for image_subdir, kfold_subdir in CONFIGS:
        image_dir = RAW_BASE / image_subdir
        kfold_dir = KFOLD_BASE / kfold_subdir
        if not image_dir.is_dir():
            log(f"  SKIP (no images): {image_dir}")
            continue

        marker = kfold_dir / "fold_5" / "test"
        if marker.is_dir() and any(marker.iterdir()):
            log(f"  Already exists: {kfold_dir}")
            continue
        log(f"  Creating: {kfold_dir}")

        available = {}
        for f in os.listdir(image_dir):
            if not f.endswith(".png") or f in EXCLUDE_FILES:
                continue
            stem = f[:-4]
            if "-" in stem:
                continue
            available[stem] = image_dir / f

        for fold in range(N_FOLDS):
            for split_name in ["train", "val", "test"]:
                split_dir = kfold_dir / f"fold_{fold}" / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                patients = splits[fold][split_name]

                for sample_name, src_path in available.items():
                    pid = sample_name[:-1]
                    if pid not in patients:
                        continue
                    dst = split_dir / f"{sample_name}.png"
                    shutil.copy2(src_path, dst)
                    if split_name == "train":
                        img = Image.open(src_path).convert("RGB")
                        for aug_i in range(1, N_AUGMENTS + 1):
                            set_seed(69 + aug_i + hash(sample_name) % 10000)
                            aug_img = aug_pipeline(img)
                            aug_dst = split_dir / f"{sample_name}_aug{aug_i}.png"
                            aug_img.save(aug_dst)
        log(f"  Done: {kfold_dir}")


if __name__ == "__main__":
    main()
