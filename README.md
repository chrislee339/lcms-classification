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
comparison), along with the cross-validation splits and the result tables
behind the paper's figures.

## What's in here

```
code/                 Python scripts: image generation, training, ablations,
                      GradCAM, classical baseline, statistical comparisons
data/kfold/           Patient-to-fold assignments used in the paper
                        patient_splits.json          primary cohort
                        patient_splits_withS11.json  sensitivity cohort
results/              Per-fold metrics, bootstrap CIs, and paired-comparison
                      tables that back the paper's tables and figures
```

The raw LC-MS files and trained model weights are **not** included. Sharing of
the underlying patient data is governed by the study's IRB protocol and
participant consent; inquiries can be directed to the corresponding author.

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

Scripts find the repository root automatically, so they can be run from
anywhere inside the project. To point them elsewhere, set `LCMS_BASE`.

## Reproduce the paper's results

The tables in `results/` contain every per-fold metric reported in the paper.
These scripts read them and recompute the statistics. CPU only, no raw data
needed:

```
python code/bootstrap_ablations.py          # architecture/resolution stats
python code/compare_s11_sensitivity.py      # S11 in/out sensitivity
python code/compare_globalnorm.py           # per-sample vs global normalization
python code/compare_classical_baselines.py  # CNN vs Random Forest / PLS-DA
```

## Full pipeline

Regenerating the spectral images or training models requires the raw LC-MS runs
(not included; see above). With the mzML files in `data/mzml/`, named
`S<patient><A|B>.mzML` (A = adjacent normal, B = tumor; e.g. `S24A.mzML`,
`S24B.mzML`):

```
# 1. Generate spectral images (per-sample normalization, the headline setup)
python code/bin_spectra_per_sample.py --mzml data/mzml \
    --output data/raw/spectra_filter_512 --bins 512 --workers 2

# 2. Build patient-stratified cross-validation folds
python code/prepare_kfold_dirs.py            # primary cohort
python code/prepare_kfold_dirs_withS11.py    # sensitivity cohort

# 3. Train the full set of experiments (3 seeds x 6 folds per configuration;
#    ~1-3 days on a single modern GPU)
python code/run_all_experiments.py
```

Supporting steps used for specific analyses:

- `bin_spectra_subregion.py`: focused bottom-left-quadrant images
- `bin_spectra_global_norm.py` + `run_global_norm_comparison.py`: the
  global-axis normalization comparison
- `run_s11_sensitivity.py`: retrains with the excluded patient included
- `pick_peaks_classical.py`: the peak-picking + Random Forest / PLS-DA baseline
- `s11_qc_hunt.py`: model-independent image quality-control metrics
- `run_gradcam_analysis.py`, `gradcam_occlusion_test.py`,
  `make_occlusion_figure.py`: attention maps and the occlusion control
  (these need trained model weights, available from the corresponding author)

## Cross-validation splits

`data/kfold/*.json` give the exact patient-to-fold assignments. To load:

```python
import json
with open("data/kfold/patient_splits.json") as f:
    splits = {int(k): {s: set(v) for s, v in d.items()}
              for k, d in json.load(f).items()}
# splits[fold]['train' | 'val' | 'test'] -> set of patient IDs, e.g. "S24"
```

## License

MIT. See `LICENSE`.
