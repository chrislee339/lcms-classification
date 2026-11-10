#!/bin/bash
# Recompute every statistics table in results/analysis/stats (and the occlusion statistics) from the per-sample
# predictions shipped in results/analysis. CPU only; no images, checkpoints or raw data needed. With the default
# 10,000 bootstrap and permutation resamples this takes about two hours on four CPU cores.
#
# usage (from the repository root): bash code/analysis/reproduce_statistics.sh <out dir>
set -euo pipefail
OUT=${1:?usage: reproduce_statistics.sh <out dir>}
PY=${PYTHON:-python3}
C=code/analysis; R=results/analysis; P=$R/predictions; K=$R/classical; N=$R/nested_inner
mkdir -p "$OUT"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# 38 patients, S11 always in a training set: single-model and three-seed-ensemble statistics
$PY $C/single_model_stats.py "$OUT/single38rot.csv" $P/*_withS11rot_predictions.csv
$PY $C/patient_stats.py $P "$OUT/stats38rot" --rot

# 37 patients (S11 excluded): the full-spectrum and subregion runs in results/, the architectures in predictions_37
mkdir -p "$TMP/p37"
for f in results/{fullspec,subregion}_resnet34_{256,512,1024}_predictions.csv; do ln -s "$PWD/$f" "$TMP/p37/"; done
$PY $C/single_model_stats.py "$OUT/single37.csv" "$TMP"/p37/*_predictions.csv
$PY $C/single_model_stats.py "$OUT/single37_arch.csv" $R/predictions_37/arch_resnet{18,34,50,101,152}_256_predictions.csv
$PY $C/patient_stats.py "$TMP/p37" "$OUT/stats37" --37
$PY $C/patient_stats.py $R/predictions_37 "$OUT/stats37_arch" --37

# Nested cross-validation (the configuration chosen in each outer fold from the inner-fold predictions)
$PY $C/nested_result.py $N/38 $P $C data/kfold/patient_splits_withS11_rot.json "$OUT/nested"
$PY $C/nested_result_37.py $N/37 $N/38 $R/predictions_37 results $C data/kfold/patient_splits_withS11_rot.json \
  "$OUT/nested37"

# Classical baselines and their comparison with the CNN
$PY $C/single_model_stats.py "$OUT/single_classical38.csv" $K/c38/classical_*_38_predictions.csv
$PY $C/single_model_stats.py "$OUT/single_classical37.csv" $K/c37/classical_*_37_predictions.csv
FIXED38="$K/c38/classical_rf_fixed_38_predictions.csv $K/c38/classical_plsda_fixed_38_predictions.csv"
$PY $C/single_model_compare.py "$OUT/nested/nested_withS11rot_predictions.csv" "$OUT/single_cmp_fixed38_nested.csv" $FIXED38
$PY $C/single_model_compare.py $P/fullspec_resnet34_512_withS11rot_predictions.csv "$OUT/single_cmp_fixed38_512.csv" $FIXED38
$PY $C/single_model_compare.py "$OUT/nested/nested_withS11rot_predictions.csv" "$OUT/single_cmp_all38_nested.csv" \
  $K/c38/classical_{rf,plsda}_{fixed,tuned}_38_predictions.csv
$PY $C/single_model_compare.py "$OUT/nested37/nested_37_predictions.csv" "$OUT/nested37/single_cmp37_nested.csv" \
  $K/c37/classical_{rf,plsda}_{fixed,tuned}_37_predictions.csv
mkdir -p "$TMP/c38"
for f in $K/c38/classical_*_38_predictions.csv $P/fullspec_resnet34_512_withS11rot_predictions.csv \
         "$OUT/nested/nested_withS11rot_predictions.csv"; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$TMP/c38/"; done
$PY $C/cnn_vs_classical.py "$TMP/c38" nested_withS11rot 38 "$OUT/cnn_vs_classical38_nested.csv"
$PY $C/cnn_vs_classical.py "$TMP/c38" fullspec_resnet34_512_withS11rot 38 "$OUT/cnn_vs_classical38_512.csv"

# Custom CNN, Table S7 (all five metrics) and the occlusion control
$PY $C/single_model_stats.py "$OUT/single_custom_cnn38rot.csv" $R/custom_cnn/custom_cnn_256_withS11rot_predictions.csv
$PY $C/table_s7_all_metrics.py "$OUT/s7_all_metrics.csv"
$PY $C/single_model_stats.py "$OUT/occlusion_stats.csv" $R/occlusion/occlusion_*_predictions.csv
echo "done: $OUT"
