"""Train the custom CNN baseline (custom_cnn_model.py) on the 38-patient folds in which patient S11 is always in a
training set, one seed per call, with run_single_experiment of the original training code
(code/run_all_experiments.py) unchanged: the same 256 x 256 full-spectrum folds as the architecture comparison, and the
same seeds, augmentation, transform, optimizer, schedule, epochs and top-3 checkpointing as the ResNets. The only
difference is the model, and the 5-epoch frozen-backbone phase freezes nothing because there is no pretrained backbone.
Checkpoints go to models/resnet_v2/custom_cnn_256_withS11rot_seed<s>_seed<s>_fold<k>/ and the per-fold metrics to
results/analysis/training/custom_cnn_256_withS11rot_seed<s>.json (both under $LCMS_BASE, default: the repository root).

usage: python train_custom_cnn.py <seed> [--smoke <out dir>]      (seeds 42, 69, 137)
  --smoke  2 epochs, fold 0 only, checkpoints and JSON under <out dir> (run it twice to check determinism)
"""
import json
import os
import sys
from pathlib import Path

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
NAME = "custom_cnn_256_withS11rot"
KFOLD = f"{BASE}/data/kfold/v2_256_by_patient_withS11_rot"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_experiments as base  # noqa: E402
import custom_cnn_model  # noqa: E402

custom_cnn_model.install(base)
seed = int(sys.argv[1])
out = f"{BASE}/results/analysis/training/{NAME}_seed{seed}.json"
if "--smoke" in sys.argv:
    smoke = sys.argv[sys.argv.index("--smoke") + 1]
    base.NUM_EPOCHS, base.N_FOLDS, base.MODEL_BASE = 2, 1, os.path.join(smoke, "models")
    out = os.path.join(smoke, f"{NAME}_seed{seed}.json")
if os.path.exists(out):
    sys.exit(f"already done: {out}")
assert os.path.exists(os.path.join(KFOLD, "BUILD_DONE")), KFOLD
os.makedirs(os.path.dirname(out), exist_ok=True)
base.log(f"38-patient run, S11 in training: {NAME}, seed {seed}, {KFOLD}")
res = base.run_single_experiment("custom_cnn", KFOLD, seed, f"{NAME}_seed{seed}")
with open(out + ".tmp", "w") as fh:
    json.dump(res, fh, default=float, indent=1)
os.replace(out + ".tmp", out)
base.log(f"wrote {out}")
