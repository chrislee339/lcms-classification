"""Inner folds for the nested cross-validation on 37 patients (S11 excluded).

The 38-patient inner splits (inner_splits.json from nested_build_inner_folds.py) with S11 removed; nothing else
changes. In outer fold 4, S11 is an outer test patient, so that fold's inner folds are identical to the 38-patient
run's: they are written to inner_splits.json unchanged but not built, because those units are reused.
Images are symlinks to the same files the 38-patient inner folds use. A source is either a 38-patient k-fold folder
(data/kfold/*_withS11_rot, from build_kfold.py) or, with POOL=<dir>, a flat folder <dir>/<source>/ holding one copy
of every image.

writes $NESTED/kfold/<source>/outer<k>/fold_<j>/{train,val,test}/, inner_splits.json and BUILD_DONE per source;
sources that are not present are skipped. LCMS_BASE (default: the repository root), NESTED (default
$LCMS_BASE/nested37) and SPLITS38 (default $LCMS_BASE/nested/kfold/inner_splits.json) set the locations.

usage: python3 nested_build_inner_folds_37.py [--splits-only]
"""
import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("NESTED", BASE / "nested37")) / "kfold"
SPLITS38 = json.loads(Path(os.environ.get("SPLITS38", BASE / "nested/kfold/inner_splits.json")).read_text())
POOL = os.environ.get("POOL")
SOURCES = {
    "256": BASE / "data/kfold/v2_256_by_patient_withS11_rot",
    "512": BASE / "data/kfold/v2_512_by_patient_withS11_rot",
    "1024": BASE / "data/kfold/v2_1024_by_patient_withS11_rot",
    "sub256": BASE / "data/kfold/v2_subregion_256_by_patient_withS11_rot",
    "sub512": BASE / "data/kfold/v2_subregion_512_by_patient_withS11_rot",
    "sub1024": BASE / "data/kfold/v2_subregion_1024_by_patient_withS11_rot",
}
N_OUTER, N_INNER, N_AUG = 6, 5, 20
BUILT = (0, 1, 2, 3, 5)  # outer fold 4 is reused from the 38-patient run
DROP = "S11"

splits = {k: {j: {s: [p for p in v[s] if p != DROP] for s in ("train", "val", "test")}
              for j, v in SPLITS38[str(k)].items()} for k in range(N_OUTER)}
for k in range(N_OUTER):
    pool38 = {p for v in SPLITS38[str(k)].values() for p in v["test"]}
    assert (DROP in pool38) == (k in BUILT), k  # S11 is a non-test patient exactly in the rebuilt outer folds
    non_test = pool38 - {DROP}
    tested = [p for v in splits[k].values() for p in v["test"]]
    assert sorted(tested) == sorted(non_test), k  # each non-test patient tested once
    for j, s in splits[k].items():
        a, b, c = (set(s[x]) for x in ("train", "val", "test"))
        assert a and b and c and not (a & b or a & c or b & c) and a | b | c == non_test, (k, j)
        assert all(set(s[x]) == set(SPLITS38[str(k)][j][x]) - {DROP} for x in s), (k, j)
assert splits[4] == {j: {s: v[s] for s in ("train", "val", "test")} for j, v in SPLITS38["4"].items()}


def index(name, src):
    """filename -> real path, originals from any split, augmented copies from training splits."""
    if POOL:
        d = Path(POOL) / name
        return {f: d / f for f in os.listdir(d) if f.endswith(".png")}
    files = {}
    for fold in range(N_OUTER):
        for split in ("train", "val", "test"):
            d = (src / f"fold_{fold}" / split).resolve()
            for f in os.listdir(d):
                if f.endswith(".png") and f not in files:
                    files[f] = d / f
    return files


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inner_splits.json").write_text(json.dumps({str(k): {str(j): v for j, v in d.items()} for k, d in splits.items()},
                                                  indent=1))
if "--splits-only" in sys.argv:
    sys.exit(0)
for name, src in SOURCES.items():
    if not (Path(POOL) / name if POOL else src).exists():
        print("skipped (images not on this machine):", name)
        continue
    if (OUT / name / "BUILD_DONE").exists():
        print("already built:", name)
        continue
    files = index(name, src)
    for k in BUILT:
        for j, s in splits[k].items():
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
    print("built:", name, "from", Path(POOL) / name if POOL else src)
