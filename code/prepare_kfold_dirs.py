
"""
Pre-create all kfold directories with augmentations.
Run this ONCE before launching experiments.
"""

import os
import random
import shutil
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from pathlib import Path
BASE_DIR = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parent.parent))
RAW_BASE = os.path.join(BASE_DIR, "data/raw")
KFOLD_BASE = os.path.join(BASE_DIR, "data/kfold")
REFERENCE_KFOLD = os.path.join(KFOLD_BASE, "256_by_patient_rmS11")

N_FOLDS = 6
N_AUGMENTS = 20
EXCLUDE_PATIENTS = {"S11"}
EXCLUDE_FILES = {"Fetuin_final.png", "S2B-5x-0302.png"}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_patient_splits():
    splits = {}
    for fold in range(N_FOLDS):
        splits[fold] = {}
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(REFERENCE_KFOLD, f"fold_{fold}", split_name)
            files = os.listdir(split_dir)
            patients = set()
            for f in files:
                name = f.replace(".png", "")
                if "_aug" in name:
                    name = name.split("_aug")[0]
                patient_id = name[:-1]
                patients.add(patient_id)
            splits[fold][split_name] = patients
    return splits


def create_kfold(image_source_dir, kfold_output_dir, patient_splits):

    test_path = os.path.join(kfold_output_dir, "fold_5", "test")
    if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
        log(f"  Already exists: {kfold_output_dir}")
        return

    log(f"  Creating: {kfold_output_dir}")

    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    noise_fn = transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))

    class RandomSubsetAugmentation:
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
        RandomSubsetAugmentation([color_jitter, gaussian_blur, noise_fn]),
        transforms.ToPILImage()
    ])

    available = {}
    for f in os.listdir(image_source_dir):
        if not f.endswith('.png') or f in EXCLUDE_FILES:
            continue
        name = f.replace(".png", "")
        patient_id = name[:-1]
        if patient_id in EXCLUDE_PATIENTS:
            continue
        available[name] = os.path.join(image_source_dir, f)

    for fold in range(N_FOLDS):
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(kfold_output_dir, f"fold_{fold}", split_name)
            os.makedirs(split_dir, exist_ok=True)

            patients = patient_splits[fold][split_name]

            for sample_name, src_path in available.items():
                patient_id = sample_name[:-1]
                if patient_id not in patients:
                    continue

                dst = os.path.join(split_dir, f"{sample_name}.png")
                shutil.copy2(src_path, dst)

                if split_name == "train":
                    img = Image.open(src_path).convert("RGB")
                    for aug_i in range(1, N_AUGMENTS + 1):
                        set_seed(69 + aug_i + hash(sample_name) % 10000)
                        aug_img = aug_pipeline(img)
                        aug_dst = os.path.join(split_dir, f"{sample_name}_aug{aug_i}.png")
                        aug_img.save(aug_dst)

    log(f"  Done: {kfold_output_dir}")


def main():
    log("Creating all kfold directories...")
    patient_splits = get_patient_splits()


    configs = [

        ("spectra_filter_256", "v2_256_by_patient_rmS11"),
        ("spectra_filter_512", "v2_512_by_patient_rmS11"),
        ("spectra_filter_1024", "v2_1024_by_patient_rmS11"),

        ("focused_zoom_256", "v2_subregion_256_by_patient_rmS11"),
        ("focused_zoom_512", "v2_subregion_512_by_patient_rmS11"),
        ("focused_zoom_1024", "v2_subregion_1024_by_patient_rmS11"),
    ]

    for image_subdir, kfold_subdir in configs:
        image_dir = os.path.join(RAW_BASE, image_subdir)
        kfold_dir = os.path.join(KFOLD_BASE, kfold_subdir)

        if not os.path.exists(image_dir):
            log(f"  SKIP (no images): {image_dir}")
            continue

        create_kfold(image_dir, kfold_dir, patient_splits)

    log("All kfold directories created!")


if __name__ == "__main__":
    main()
