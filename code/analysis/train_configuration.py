"""Train one (configuration, seed) on the 38-patient folds in which patient S11 is always in a training set
(build_kfold.py), with run_single_experiment of the original training code (code/run_all_experiments.py), unchanged.
Checkpoints go to models/resnet_v2/<name>_seed<s>_seed<s>_fold<k>/, the naming of the original training code, so
evaluate_checkpoints.py scores them the same way; the per-fold metrics returned by run_single_experiment go to
results/analysis/training/<name>_seed<s>.json (both under $LCMS_BASE, default: the repository root).

Configurations (CONFIGS below): arch_resnet{18,34,50,101,152}_256_withS11rot, fullspec_resnet34_{512,1024}_withS11rot,
subregion_resnet34_{256,512,1024}_withS11rot and fullspec_resnet34_512_globalnorm_withS11rot; seeds 42, 69, 137.

usage: python train_configuration.py <configuration> <seed>      (configuration ends in "rot")
       python train_configuration.py --kfold-dir <configuration>  (prints the configuration's k-fold folder)
"""
import json
import os
import sys
from pathlib import Path

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
KF = f"{BASE}/data/kfold"
CONFIGS = {  # name: (variant, k-fold folder)
    **{f"arch_{v}_256_withS11rot": (v, f"{KF}/v2_256_by_patient_withS11_rot")
       for v in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")},
    "fullspec_resnet34_512_withS11rot": ("resnet34", f"{KF}/v2_512_by_patient_withS11_rot"),
    "fullspec_resnet34_1024_withS11rot": ("resnet34", f"{KF}/v2_1024_by_patient_withS11_rot"),
    **{f"subregion_resnet34_{r}_withS11rot": ("resnet34", f"{KF}/v2_subregion_{r}_by_patient_withS11_rot")
       for r in (256, 512, 1024)},
    "fullspec_resnet34_512_globalnorm_withS11rot": ("resnet34", f"{KF}/v2_512_by_patient_globalnorm_withS11_rot"),
}
if sys.argv[1] == "--kfold-dir":
    print(CONFIGS[sys.argv[2]][1])
    sys.exit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_all_experiments as base  # noqa: E402

name, seed = sys.argv[1], int(sys.argv[2])
variant, kfold = CONFIGS[name]
out = f"{BASE}/results/analysis/training/{name}_seed{seed}.json"
if os.path.exists(out):
    sys.exit(f"already done: {out}")
assert os.path.exists(os.path.join(kfold, "BUILD_DONE")), kfold
os.makedirs(os.path.dirname(out), exist_ok=True)
base.log(f"38-patient run, S11 in training: {name}, seed {seed}, {kfold}")
res = base.run_single_experiment(variant, kfold, seed, f"{name}_seed{seed}")
with open(out + ".tmp", "w") as fh:
    json.dump(res, fh, default=float, indent=1)
os.replace(out + ".tmp", out)
base.log(f"wrote {out}")
