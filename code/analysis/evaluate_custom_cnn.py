"""Score the custom CNN checkpoints with evaluate_checkpoints.py, unchanged, after installing the custom model class.
Paths are under $LCMS_BASE (default: the repository root).

usage: python evaluate_custom_cnn.py   (writes results/analysis/custom_cnn/custom_cnn_256_withS11rot_predictions.csv)
"""
import os
import runpy
import sys
from pathlib import Path

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_experiments as base  # noqa: E402
import custom_cnn_model  # noqa: E402

custom_cnn_model.install(base)
sys.argv = ["evaluate_checkpoints.py", "custom_cnn_256_withS11rot", "custom_cnn",
            f"{BASE}/data/kfold/v2_256_by_patient_withS11_rot", f"{BASE}/models/resnet_v2",
            f"{BASE}/results/analysis/custom_cnn"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_checkpoints.py"), run_name="__main__")
