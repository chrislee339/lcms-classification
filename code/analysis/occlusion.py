"""Occlusion control (Figure 4) on the 38-patient full-spectrum 512 x 512 models, using code/gradcam_occlusion_test.py
unchanged.

Imports gradcam_occlusion_test.py and points it at a 38-patient k-fold folder and its checkpoints, writing
to <out> instead of results/. Then splits the per-sample predictions into one file per
condition (same format as the CNN predictions) for single_model_stats.py.

OC_KFOLD, OC_EXP and OC_MODEL_BASE choose the k-fold folder, the configuration and the checkpoint folder. The
defaults are the run used in the paper, in which S11 is always in a training set
(data/kfold/v2_512_by_patient_withS11_rot, fullspec_resnet34_512_withS11rot, models/resnet_v2).
make_figures.py reads the mean quadrant intensities that this script prints from logs/occlusion.txt, so save the
output there.

usage: python3 occlusion.py <out dir>
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradcam_occlusion_test as oc  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
oc.KFOLD = Path(os.environ.get("OC_KFOLD", oc.BASE / "data/kfold/v2_512_by_patient_withS11_rot"))  # OC_* choose
oc.EXP = os.environ.get("OC_EXP", "fullspec_resnet34_512_withS11rot")                            # the run to test
oc.MODEL_BASE = Path(os.environ.get("OC_MODEL_BASE", oc.MODEL_BASE))                           # (see above)
oc.RESULTS_DIR = out
oc.main()

df = pd.read_csv(out / "gradcam_occlusion.csv")
assert df.groupby("condition").size().eq(228).all(), df.groupby("condition").size()
for cond, g in df.groupby("condition"):
    g = g.assign(y_prob_ens=g.y_prob)[["fold", "seed", "filename", "y_true", "y_prob", "y_prob_ens"]]
    g.to_csv(out / f"occlusion_{cond}_predictions.csv", index=False)
print("conditions:", sorted(df.condition.unique()))
