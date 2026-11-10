"""Build 38-patient k-fold folders in which patient S11 is always in a training set.

In data/kfold/patient_splits_withS11.json, S11 is in the validation set of folds 0, 1, 2, 3 and 5 and in the test set
of fold 4, so it is never trained on. Here S11 is moved from validation to training in those five folds, the most
common placement in these splits (19 of the other 37 patients are in the training set of every fold in which they
are not tested). The new splits are written to data/kfold/patient_splits_withS11_rot.json. Every other file is a
symlink into the source's existing 38-patient k-fold folder, so the validation sets are the same as in the
37-patient runs and all other patients keep their folds. Each output folder is named <source k-fold folder>_rot and
gets a BUILD_DONE marker when complete.

Sources (SOURCES below): the 38-patient k-fold folders data/kfold/v2_*_by_patient_withS11, built with the procedure
of code/prepare_kfold_dirs_withS11.py (which lists 256 and 512; the 1024, subregion and global-normalization folders
apply the same procedure to their image folders), and the image folders under data/raw.

S11's 20 augmented copies per sample are written to data/s11_aug/<source>/ with the k-fold builders' own procedure
(same transforms, same set_seed(69 + aug_i + hash(name) % 10000)); run with PYTHONHASHSEED=0 so that hash() gives the
same seeds in every run. --check <source> <sample> regenerates an existing sample's copies (in data/s11_aug_check/)
and compares them byte for byte with that source folder; they match exactly for a source folder built with
PYTHONHASHSEED=0. The 256 and 512 source folders used in the paper were built with a different hash seed, so S11's
copies there are new draws of the same procedure.

All paths are under $LCMS_BASE (default: the repository root).

usage: PYTHONHASHSEED=0 python build_kfold.py [<source> ...]          (default: all sources)
       PYTHONHASHSEED=0 python build_kfold.py --check 1024 S20A
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DATA = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parents[2])) / "data"
SOURCES = {  # key: (existing 38-patient k-fold folder, raw image folder, S11 augmentation folder)
    "512": (DATA / "kfold/v2_512_by_patient_withS11", DATA / "raw/spectra_filter_512", DATA / "s11_aug/512"),
    "256": (DATA / "kfold/v2_256_by_patient_withS11", DATA / "raw/spectra_filter_256", DATA / "s11_aug/256"),
    "1024": (DATA / "kfold/v2_1024_by_patient_withS11", DATA / "raw/spectra_filter_1024", DATA / "s11_aug/1024"),
    "sub256": (DATA / "kfold/v2_subregion_256_by_patient_withS11", DATA / "raw/focused_zoom_256", DATA / "s11_aug/sub256"),
    "sub512": (DATA / "kfold/v2_subregion_512_by_patient_withS11", DATA / "raw/focused_zoom_512", DATA / "s11_aug/sub512"),
    "sub1024": (DATA / "kfold/v2_subregion_1024_by_patient_withS11", DATA / "raw/focused_zoom_1024", DATA / "s11_aug/sub1024"),
    "global512": (DATA / "kfold/v2_512_by_patient_globalnorm_withS11", DATA / "raw/spectra_filter_global_512", DATA / "s11_aug/global512"),
}
N_FOLDS, N_AUG = 6, 20
assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
dst_of = lambda key: DATA / "kfold" / (SOURCES[key][0].name + "_rot")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RandomSubsetAug:
    def __init__(self, tlist, min_ops=1, max_ops=3):
        self.tlist, self.min_ops, self.max_ops = tlist, min_ops, max_ops

    def __call__(self, img):
        n = random.randint(self.min_ops, self.max_ops)
        return transforms.Compose(random.sample(self.tlist, n))(img)


aug_pipeline = transforms.Compose([
    transforms.ToTensor(),
    RandomSubsetAug([transforms.ColorJitter(brightness=0.1, contrast=0.1),
                     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                     transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))]),
    transforms.ToPILImage(),
])
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()


def make_augs(raw, sample, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(raw / f"{sample}.png").convert("RGB")
    for i in range(1, N_AUG + 1):
        set_seed(69 + i + hash(sample) % 10000)
        aug_pipeline(img).save(out_dir / f"{sample}_aug{i}.png")


if "--check" in sys.argv:
    key, sample = sys.argv[sys.argv.index("--check") + 1:sys.argv.index("--check") + 3]
    src, raw, _ = SOURCES[key]
    tmp = DATA / "s11_aug_check" / key
    make_augs(raw, sample, tmp)
    ref = next(src / f"fold_{k}" / "train" for k in range(N_FOLDS) if (src / f"fold_{k}" / "train" / f"{sample}_aug1.png").exists())
    same = sum(md5(tmp / f"{sample}_aug{i}.png") == md5(ref / f"{sample}_aug{i}.png") for i in range(1, N_AUG + 1))
    print(f"check {key} {sample}: {same} of {N_AUG} augmented copies byte-identical to {ref}")
    sys.exit(0)

splits = json.loads((DATA / "kfold/patient_splits_withS11.json").read_text())
rot = {}
for k, d in splits.items():
    d = {s: list(v) for s, v in d.items()}
    if "S11" in d["val"]:
        d["val"].remove("S11")
        d["train"] = sorted(d["train"] + ["S11"], key=lambda p: int(p[1:]))
    rot[k] = d
(DATA / "kfold/patient_splits_withS11_rot.json").write_text(json.dumps(rot, indent=4))

for key in (sys.argv[1:] or list(SOURCES)):
    src, raw, aug = SOURCES[key]
    dst = dst_of(key)
    if (dst / "BUILD_DONE").exists():
        print(f"{key}: already built ({dst})")
        continue
    for s in ("S11A", "S11B"):
        if not (aug / f"{s}_aug{N_AUG}.png").exists():
            make_augs(raw, s, aug)
    for k in range(N_FOLDS):
        for split in ("train", "val", "test"):
            d = dst / f"fold_{k}" / split
            d.mkdir(parents=True, exist_ok=True)
            srcd = (src / f"fold_{k}" / split).resolve()
            for f in os.listdir(srcd):
                if f.endswith(".png") and not f.startswith("S11") and not (d / f).is_symlink():
                    (d / f).symlink_to(srcd / f)
            if "S11" in rot[str(k)][split]:
                for s in ("S11A", "S11B"):
                    if not (d / f"{s}.png").is_symlink():
                        (d / f"{s}.png").symlink_to(raw / f"{s}.png")
                    if split == "train":
                        for i in range(1, N_AUG + 1):
                            if not (d / f"{s}_aug{i}.png").is_symlink():
                                (d / f"{s}_aug{i}.png").symlink_to(aug / f"{s}_aug{i}.png")
            n = len(os.listdir(d))
            assert n == len(rot[str(k)][split]) * 2 * (N_AUG + 1 if split == "train" else 1), (key, k, split, n)
            assert ("S11A.png" in os.listdir(d)) == ("S11" in rot[str(k)][split]), (key, k, split)
    (dst / "BUILD_DONE").touch()
    print(f"{key}: built {dst}")
for k, d in rot.items():
    where = next(s for s in ("train", "val", "test") if "S11" in d[s])
    print(f"fold {k}: S11 in {where}; train {len(d['train'])}, val {len(d['val'])}, test {len(d['test'])}")
