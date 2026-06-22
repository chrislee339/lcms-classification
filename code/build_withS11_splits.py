"""Build patient_splits_withS11.json by adding S11 to the existing rmS11
split using the original 256_by_patient placement (S11 in test for fold 4,
val for the other folds)."""
import json
from pathlib import Path

import os
BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
KFOLD = BASE / "data/kfold"
src = KFOLD / "patient_splits.json"
dst = KFOLD / "patient_splits_withS11.json"

with src.open() as f:
    splits = json.load(f)


S11_PLACEMENT = {
    "0": "val",
    "1": "val",
    "2": "val",
    "3": "val",
    "4": "test",
    "5": "val",
}

for fold, where in S11_PLACEMENT.items():
    if "S11" in splits[fold].get(where, []):
        continue
    splits[fold].setdefault(where, []).append("S11")
    splits[fold][where] = sorted(splits[fold][where])

with dst.open("w") as f:
    json.dump(splits, f, indent=4)


for k, v in splits.items():
    s11 = next(s for s in ["train", "val", "test"] if "S11" in v.get(s, []))
    print(f"fold {k}: S11 in {s11}; train={len(v['train'])} "
          f"val={len(v['val'])} test={len(v['test'])}")
print(f"\nWrote {dst}")
