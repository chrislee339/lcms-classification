"""CNN vs classical baselines, patient-level and paired, with the same functions as
patient_stats.py (seed-averaged predictions, patient bootstrap, patient-level permutation, Holm within
the family of four comparisons).

The prediction folder must hold <cnn name>_predictions.csv and the four
classical_{rf,plsda}_{fixed,tuned}_<cohort>_predictions.csv files written by classical_foldlocal.py.

usage: python3 cnn_vs_classical.py <dir with all prediction files> <cnn name> <cohort 37|38> <out csv>
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pred_dir, cnn, cohort, out_csv = sys.argv[1:5]
sys.argv = [sys.argv[0], pred_dir, out_csv + "_unused"]
import patient_stats as ps  # noqa: E402

rows = []
for clf in ("rf_fixed", "rf_tuned", "plsda_fixed", "plsda_tuned"):
    for col in ("y_prob", "y_prob_ens"):
        rows += [dict(r, family="CNN vs classical") for r in ps.compare(f"classical_{clf}_{cohort}", cnn, col)]
res = ps.holm(pd.DataFrame(rows))
res.to_csv(out_csv, index=False)
print(res[res.model == "y_prob"][["a", "b", "metric", "b_minus_a", "lo", "hi", "p_perm", "p_holm"]].round(3).to_string(index=False))
