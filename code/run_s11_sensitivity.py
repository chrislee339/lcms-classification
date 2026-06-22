"""Run the S11-included sensitivity training experiments.

Reuses the training plumbing in run_all_experiments.py (run_experiment_group)
but points each experiment at a v2_*_by_patient_withS11 kfold directory and
saves results with names ending in `_withS11`. The output CSVs land alongside
the existing rmS11 results in `results/`.

We are deliberately scoping the rerun to the experiments the headline claims
depend on:
  - fullspec_resnet34_512_withS11   (the headline configuration)
  - arch_resnet{18,34,50,101,152}_256_withS11  (the architecture ablation)

This is enough to support a sensitivity-analysis table in the paper without
re-running the full design space.
"""
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_all_experiments as base





base.EXCLUDE_PATIENTS = set()

BASE = os.environ.get("LCMS_BASE", str(__import__("pathlib").Path(__file__).resolve().parent.parent))
RAW = os.path.join(BASE, "data/raw")
KFOLD = os.path.join(BASE, "data/kfold")

EXPERIMENTS = [

    {
        "name": "fullspec_resnet34_512_withS11",
        "variant": "resnet34",
        "image_dir": os.path.join(RAW, "spectra_filter_512"),
        "kfold_dir": os.path.join(KFOLD, "v2_512_by_patient_withS11"),
    },

    {
        "name": "arch_resnet18_256_withS11",
        "variant": "resnet18",
        "image_dir": os.path.join(RAW, "spectra_filter_256"),
        "kfold_dir": os.path.join(KFOLD, "v2_256_by_patient_withS11"),
    },
    {
        "name": "arch_resnet34_256_withS11",
        "variant": "resnet34",
        "image_dir": os.path.join(RAW, "spectra_filter_256"),
        "kfold_dir": os.path.join(KFOLD, "v2_256_by_patient_withS11"),
    },
    {
        "name": "arch_resnet50_256_withS11",
        "variant": "resnet50",
        "image_dir": os.path.join(RAW, "spectra_filter_256"),
        "kfold_dir": os.path.join(KFOLD, "v2_256_by_patient_withS11"),
    },
    {
        "name": "arch_resnet101_256_withS11",
        "variant": "resnet101",
        "image_dir": os.path.join(RAW, "spectra_filter_256"),
        "kfold_dir": os.path.join(KFOLD, "v2_256_by_patient_withS11"),
    },
    {
        "name": "arch_resnet152_256_withS11",
        "variant": "resnet152",
        "image_dir": os.path.join(RAW, "spectra_filter_256"),
        "kfold_dir": os.path.join(KFOLD, "v2_256_by_patient_withS11"),
    },
]


def main() -> None:
    base.log("=" * 60)
    base.log("S11-included sensitivity rerun")
    base.log("=" * 60)
    base.log(f"Configurations: {len(EXPERIMENTS)}")
    for exp in EXPERIMENTS:
        base.log(f"  - {exp['name']}  ({exp['variant']}, {exp['kfold_dir']})")
    t0 = time.time()
    base.run_experiment_group(EXPERIMENTS, active_seeds=base.SEEDS)
    base.log(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
