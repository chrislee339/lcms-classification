# LC-MS Spectral Image Classification

Companion code for the manuscript *"An End-to-End Deep Learning Framework for
Classification of LC-MS Spectral Images: A Systematic Evaluation on Paired
Breast Tumor and Adjacent Normal Tissue."*

This project classifies liquid chromatography-mass spectrometry (LC-MS) runs
of breast tissue as tumor or adjacent-normal. Instead of the usual peak-picking
and feature-engineering pipeline, it turns each raw run into a 2D image (mass-
to-charge ratio on one axis, retention time on the other) and trains a standard
image classifier (an ImageNet-pretrained ResNet) directly on those images. The
repository contains the full pipeline (image generation, model training, the
ablation experiments, interpretability analysis, and a classical baseline for
comparison), along with the cross-validation splits and the per-sample
predictions and statistics behind the paper's tables and figures.

## What's in here

```
code/                 The pipeline: image generation, the training code
                      (run_all_experiments.py), GradCAM, occlusion, and the
                      classical baseline
code/analysis/        The analyses reported in the paper: fold construction,
                      training and checkpoint evaluation, nested
                      cross-validation, patient-level statistics, classical
                      baselines, GradCAM agreement, occlusion, and figures
data/kfold/           Patient-to-fold assignments (see Cross-validation splits)
results/analysis/     Per-sample predictions and the statistics behind the
                      paper's tables and figures (see below)
results/              Per-fold results of the earlier runs, including the
                      37-patient runs (S11 excluded) used in the sensitivity
                      analysis, and the leave-one-patient-out predictions
                      (lopo_predictions.csv)
```

The raw LC-MS files and trained model weights are **not** included. Sharing of
the underlying patient data is governed by the study's IRB protocol and
participant consent; inquiries can be directed to the corresponding author.

## Study design in brief

- 38 patients, each with one adjacent-normal (A) and one tumor (B) LC-MS run
  (76 runs).
- Six-fold cross-validation split by patient: the training, validation and
  test patients of a fold never overlap, and every patient is a test patient
  in exactly one fold. Each configuration is trained with three seeds
  (42, 69, 137).
- Primary analysis: all 38 patients, with the folds in
  `data/kfold/patient_splits_withS11_rot.json`. Patient S11 is in the training
  split for folds 0, 1, 2, 3 and 5 and in the test split for fold 4. Files from
  this layout end in `_withS11rot`.
- The configuration (backbone, binning, full spectrum or subregion) is chosen
  in each fold by nested cross-validation: the fold's non-test patients are
  split into five inner folds, the configuration with the highest inner
  AUC-ROC is chosen, and it is scored on the fold's test patients.
- Sensitivity analysis: the same folds with S11 removed (37 patients,
  `data/kfold/patient_splits.json`; files without a suffix).
- Statistics are computed at the patient level: 95% intervals from a patient
  cluster bootstrap, paired permutation tests, and Holm correction within each
  family of comparisons.

## Installation

Tested with Python 3.9-3.12.

```
git clone <this-repo>
cd <this-repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`pyopenms` (needed only for the image-generation and classical-baseline steps)
occasionally fails to install via pip on some platforms; if so, use
`conda install -c bioconda pyopenms`. A CUDA GPU is recommended for training
but not for anything else; ResNet-34 at 512x512 fits on a single 24 GB card.
The reported models were trained with Python 3.8, PyTorch 2.2.2 (CUDA 12.1),
torchvision 0.17.2 and Pillow 10.3.0 on NVIDIA RTX 4090 GPUs; other GPUs and
library versions can give slightly different trained models.

Scripts find the repository root automatically, so they can be run from
anywhere inside the project. To point them elsewhere, set `LCMS_BASE`.

## Reproduce the paper's statistics

`results/analysis/` holds the per-sample test predictions of every trained
configuration, so all statistics can be recomputed without images,
checkpoints or a GPU:

```
bash code/analysis/reproduce_statistics.sh <out dir>
```

This rebuilds every table in `results/analysis/stats/`, including the nested
cross-validation estimate from the inner-fold predictions, and the occlusion
statistics. With the default 10,000 resamples it takes about two hours on four
CPU cores.

Contents of `results/analysis/`:

```
predictions/        Per-sample test predictions (*_predictions.csv) and
                    validation losses (*_valloss.csv) of each configuration,
                    38 patients
predictions_37/     The same for the five architectures on 37 patients (the
                    other 37-patient predictions are in results/)
custom_cnn/         The custom CNN baseline, 38 patients
nested_inner/       Inner-fold predictions of the nested cross-validation
                    (38/ and 37/), with the inner fold assignments
classical/          Random Forest and PLS-DA predictions (c38/, c37/) and the
                    per-fold feature counts and chosen settings
gradcam/            GradCAM attention agreement and location statistics
occlusion/          Quadrant occlusion predictions and statistics
stats/              Single-model and ensemble statistics, paired comparisons,
                    the nested cross-validation result (nested/, nested37/),
                    CNN vs classical comparisons, and all metrics for Table S7
mzml_scan_audit.csv Scan levels and acquisition intervals of every run
```

## Full pipeline

Regenerating the spectral images or training models requires the raw LC-MS runs
(not included; see above). With the mzML files in `data/mzml/`, named
`S<patient><A|B>.mzML` (A = adjacent normal, B = tumor; e.g. `S24A.mzML`,
`S24B.mzML`):

```
# 1. Generate spectral images (per-sample normalization; repeat for 256 and 1024)
python code/bin_spectra_per_sample.py --mzml data/mzml \
    --output data/raw/spectra_filter_512 --bins 512 --workers 2

# 2. Build patient-stratified cross-validation folds
python code/prepare_kfold_dirs.py            # 37 patients (S11 excluded)
python code/prepare_kfold_dirs_withS11.py    # 38 patients
PYTHONHASHSEED=0 python code/analysis/build_kfold.py   # 38 patients, S11 in training

# 3. Train the 37-patient experiments (3 seeds x 6 folds per configuration;
#    ~1-3 days on a single modern GPU)
python code/run_all_experiments.py
```

Supporting steps used for specific analyses:

- `bin_spectra_subregion.py`: focused bottom-left-quadrant images
- `bin_spectra_global_norm.py`: images on common m/z and retention-time axes
  (Table S7)
- `pick_peaks_classical.py`: the peak-picking + Random Forest / PLS-DA baseline
- `s11_qc_hunt.py`: model-independent image quality-control metrics
  (Supplementary Section S4; results in `results/s11_qc_per_*.csv`)
- `run_gradcam_analysis.py`, `gradcam_occlusion_test.py`: attention maps and
  the occlusion control (these need trained model weights, available from the
  corresponding author)

The 38-patient analyses in the paper are run with the scripts in
`code/analysis/`, in this order (each script's docstring gives its inputs,
outputs and options):

1. `train_configuration.py <configuration>_withS11rot <seed>` for the eleven
   configurations (ResNet-18/34/50/101/152 at 256 x 256; full spectrum at 512
   and 1024; subregion at 256, 512 and 1024; full spectrum 512 on common axes)
   and the seeds 42, 69 and 137; `train_custom_cnn.py <seed>` for the custom CNN.
2. `evaluate_checkpoints.py` for each configuration (per-sample predictions and
   validation losses), and `evaluate_custom_cnn.py`.
3. Nested cross-validation: `nested_build_inner_folds.py`, then
   `nested_train_inner.py <configuration>_withS11rot <outer fold> <seed>` for
   the ten configurations, six outer folds and three seeds; the 37-patient
   version uses `nested_build_inner_folds_37.py` and `nested_train_inner_37.py`.
   `nested_result.py` and `nested_result_37.py` then choose the configuration
   in each outer fold and score it.
4. `classical_foldlocal.py`: Random Forest and PLS-DA with the feature matrix
   built and the settings tuned inside each fold.
5. `gradcam_stability.py` and `occlusion.py` on the full-spectrum 512 models.
6. The statistics, as in `reproduce_statistics.sh`, and `make_figures.py`.
7. `mzml_scan_audit.py`: scan levels and acquisition intervals from the mzML
   headers.

## Cross-validation splits

`data/kfold/*.json` give the exact patient-to-fold assignments:

```
patient_splits_withS11_rot.json  38 patients, S11 in training (primary analysis)
patient_splits.json              37 patients, S11 excluded (sensitivity analysis)
patient_splits_withS11.json      38 patients, S11 in validation (earlier layout;
                                 the source of the folds above)
nested_inner_splits_38.json      inner folds of the nested cross-validation,
nested_inner_splits_37.json      per outer fold, 38 and 37 patients
```

The three outer layouts have the same test patients in every fold. To load:

```python
import json
with open("data/kfold/patient_splits_withS11_rot.json") as f:
    splits = {int(k): {s: set(v) for s, v in d.items()}
              for k, d in json.load(f).items()}
# splits[fold]['train' | 'val' | 'test'] -> set of patient IDs, e.g. "S24"
```

## License

MIT. See `LICENSE`.
