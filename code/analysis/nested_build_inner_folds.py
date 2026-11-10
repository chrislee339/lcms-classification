"""Inner folds for the nested cross-validation on 38 patients (S11 always in a training set).

For each outer fold k of data/kfold/patient_splits_withS11_rot.json (written by build_kfold.py), the
outer non-test patients (train + val, 31 or 32) are shuffled with random.Random(2026 + k) and dealt into 5 groups
(sizes differ by at most one). Inner fold j tests on group j, picks checkpoints on group (j + 1) mod 5 and trains on
the other three. Outer test patients never enter an inner fold. S11 is an ordinary patient here, as in the outer
splits.

Folders hold symlinks into the 38-patient k-fold folders (data/kfold/*_withS11_rot, from build_kfold.py): a sample's
augmented copies are byte-identical in every outer fold where it is in training, so the inner runs train on exactly
the images of the outer runs.

writes $NESTED/kfold/<source>/outer<k>/fold_<j>/{train,val,test}/, inner_splits.json and BUILD_DONE per source;
sources whose k-fold folder is not present are skipped. LCMS_BASE (default: the repository root) and NESTED
(default $LCMS_BASE/nested) set the locations.

usage: python3 nested_build_inner_folds.py [--splits-only]
"""
import json
import os
import random
import sys
from pathlib import Path

BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("NESTED", BASE / "nested")) / "kfold"
SPLITS = json.loads((BASE / "data/kfold/patient_splits_withS11_rot.json").read_text())
SOURCES = {
    "256": BASE / "data/kfold/v2_256_by_patient_withS11_rot",
    "512": BASE / "data/kfold/v2_512_by_patient_withS11_rot",
    "1024": BASE / "data/kfold/v2_1024_by_patient_withS11_rot",
    "sub256": BASE / "data/kfold/v2_subregion_256_by_patient_withS11_rot",
    "sub512": BASE / "data/kfold/v2_subregion_512_by_patient_withS11_rot",
    "sub1024": BASE / "data/kfold/v2_subregion_1024_by_patient_withS11_rot",
}
N_OUTER, N_INNER, N_AUG = 6, 5, 20
num = lambda p: int(p[1:])


def inner_splits(k):
    pts = sorted(SPLITS[str(k)]["train"] + SPLITS[str(k)]["val"], key=num)
    random.Random(2026 + k).shuffle(pts)
    groups = [sorted(pts[j::N_INNER], key=num) for j in range(N_INNER)]
    return {j: {"train": sorted(sum((groups[i] for i in range(N_INNER) if i not in (j, (j + 1) % N_INNER)), []), key=num),
                "val": groups[(j + 1) % N_INNER], "test": groups[j]} for j in range(N_INNER)}


def index(src):
    """filename -> real path, originals from any split, augmented copies from training splits."""
    files = {}
    for fold in range(N_OUTER):
        for split in ("train", "val", "test"):
            d = (src / f"fold_{fold}" / split).resolve()
            for f in os.listdir(d):
                if f.endswith(".png") and f not in files:
                    files[f] = d / f
    return files


splits = {k: inner_splits(k) for k in range(N_OUTER)}
for k, inner in splits.items():
    outer_test, non_test = set(SPLITS[str(k)]["test"]), set(SPLITS[str(k)]["train"] + SPLITS[str(k)]["val"])
    tested = sorted(p for j in inner for p in inner[j]["test"])
    assert sorted(tested) == sorted(non_test) and len(tested) == len(set(tested)), k  # each non-test patient once
    for j, s in inner.items():
        assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"]))
        assert not (set(s["train"]) | set(s["val"]) | set(s["test"])) & outer_test
        assert set(s["train"]) | set(s["val"]) | set(s["test"]) == non_test

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inner_splits.json").write_text(json.dumps({str(k): {str(j): v for j, v in d.items()} for k, d in splits.items()},
                                                  indent=1))
if "--splits-only" in sys.argv:
    sys.exit(0)
for name, src in SOURCES.items():
    if not src.exists():
        print("skipped (k-fold folder not on this machine):", name)
        continue
    if (OUT / name / "BUILD_DONE").exists():
        print("already built:", name)
        continue
    files = index(src)
    for k, inner in splits.items():
        for j, s in inner.items():
            for split, patients in s.items():
                d = OUT / name / f"outer{k}" / f"fold_{j}" / split
                d.mkdir(parents=True, exist_ok=True)
                wanted = [f"{p}{c}" for p in patients for c in "AB"]
                if split == "train":
                    wanted += [f"{w}_aug{i}" for w in list(wanted) for i in range(1, N_AUG + 1)]
                for w in wanted:
                    link = d / f"{w}.png"
                    if not link.is_symlink():
                        link.symlink_to(files[f"{w}.png"])  # KeyError here = a missing source image
                n = len(os.listdir(d))
                assert n == len(patients) * 2 * (N_AUG + 1 if split == "train" else 1), (name, k, j, split, n)
    (OUT / name / "BUILD_DONE").touch()
    print("built:", name, "from", src)
