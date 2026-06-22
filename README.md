# LC-MS Spectral Image Classification with CNNs

End-to-end deep-learning framework for binary tumor / adjacent-normal
classification of paired breast tissue LC-MS runs. Companion code for the
manuscript *"An End-to-End Deep Learning Framework for Classification of
LC-MS Spectral Images: A Systematic Evaluation on Paired Breast Tumor and
Adjacent Normal Tissue"* (under review at BMC Bioinformatics).

The pipeline rasterizes raw mzML spectra into 2D m/z x retention-time images
and fine-tunes ImageNet-pretrained ResNets on the resulting representations,
without peak picking or feature engineering. The headline configuration
(ResNet-34, 512x512, full-spectrum) achieves 91.7% accuracy and 96.8% pooled
AUC-ROC on a 37-patient cohort under patient-stratified six-fold
cross-validation, and significantly outperforms a classical peak-picking +
Random Forest / PLS-DA baseline (paired Wilcoxon p < 0.01 on every metric).

## Repository contents

```
.
|-- code/                Python scripts (training, ablations, comparisons,
|                        GradCAM, classical baseline)
|-- data/kfold/
|   |-- patient_splits.json              S11-excluded splits (primary cohort)
|   `-- patient_splits_withS11.json      S11-included splits (sensitivity)
|-- results/             Aggregate CSVs (per-fold metrics, bootstrap CIs,
|                        paired comparisons) used to populate the paper's
|                        tables and supplementary tables.
|-- README.md
|-- LICENSE
`-- requirements.txt
```

The raw LC-MS mzML files and trained model weights are not included; both are
available from the corresponding author on reasonable request, subject to
patient consent and institutional data-sharing policies.

## What is and isn't reproducible from this repository

What you **can** reproduce without raw data, using only the bundled CSVs:
* All numerical claims in the paper's tables and supplementary tables.
* The paired bootstrap CIs and Wilcoxon tests reported in the comparison
  sections.
* All paper figures derived from the metrics (everything except the raw
  spectral image panels).

What you **cannot** reproduce without raw mzML files:
* Generating the per-sample-normalized PNG images from raw spectra.
* Training new model checkpoints from scratch.
* GradCAM and occlusion analyses, which depend on saved model weights and
  per-sample images.

What you **can** reproduce with raw data + this code:
* The full end-to-end pipeline: image generation, kfold split assembly,
  training, evaluation, GradCAM, occlusion analysis.

## Installation

Tested with Python 3.9 - 3.12.

```
git clone <this-repo>
cd <this-repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

If `pip install pyopenms` fails on your platform (it occasionally does on
non-x86_64 or unusual glibc versions), install pyopenms via conda instead:
`conda install -c bioconda pyopenms`. Only the binning and classical-baseline
scripts need it; the comparison scripts in `results/`-only mode do not.

Path handling: every script resolves the repo root from its own location, so
running scripts from anywhere inside the repo works without configuration.
To override (e.g. when running on cluster scratch), set `LCMS_BASE`:

```
export LCMS_BASE=/path/to/repo
```

A CUDA-capable GPU is recommended for training; ResNet-34 at 512x512 fits
comfortably on a single 24 GB card. Inference, GradCAM, and the comparison
scripts run on CPU.

## Reproducing the paper's analyses without retraining

The `results/` directory contains every per-fold metric the paper reports.
The comparison scripts read those CSVs and perform paired bootstrap CIs and
Wilcoxon tests:

```
python code/bootstrap_ablations.py            # writes Supplementary Table S6
python code/compare_s11_sensitivity.py        # writes Supplementary Table S5
python code/compare_globalnorm.py             # writes Supplementary Table S7
python code/compare_classical_baselines.py    # writes Supplementary Table S8
```

Each script reads from `results/`, performs its analysis, prints a summary
table to stdout, and writes the corresponding `*_comparison.csv` back into
`results/`. No GPU or raw data required.

## Reproducing the analyses end-to-end (requires raw mzML)

Place mzML files in `data/mzml/`. Sample naming convention:
`S<patient_index><A|B>.mzML` (e.g. `S24A.mzML`, `S24B.mzML`).

### 1. Generate per-sample-normalized spectral images (the headline pipeline)

```
python code/bin_spectra_per_sample.py \
    --mzml data/mzml \
    --output data/raw/spectra_filter_512 \
    --bins 512 --workers 2
```

This is the canonical binning algorithm (matches `hpc/create_plot.py` from
the original HPC submissions). For each mzML, peaks-per-scan are filtered
(hard floor 5000, drop bottom 10th percentile), per-file m/z, RT, and
log10-intensity ranges are computed, peaks are binned onto an N x N grid,
and a count-normalized average is saved as a viridis PNG.

To generate the subregion (focused-zoom) images used by the subregion
ablation in Section 3.7:

```
python code/bin_spectra_subregion.py \
    --mzml data/mzml \
    --output data/raw \
    --bins 256 512 1024 --workers 2
```

This produces `data/raw/focused_zoom_<RES>/` directories using the same
per-file ranges restricted to the bottom-left half of the m/z and RT axes.

By default both binning scripts load each mzML fully into RAM via pyOpenMS
`MSExperiment`. On workstations with insufficient memory (the original
authors needed an HPC partition for this step), pass `--streaming` to use
`OnDiscMSExperiment` and read spectra one at a time. Output is identical.

### 2. Build patient-stratified kfold directories

```
python code/prepare_kfold_dirs.py             # primary cohort (S11 excluded)
python code/prepare_kfold_dirs_withS11.py     # sensitivity cohort (S11 incl)
```

These read `data/kfold/patient_splits.json` (or `_withS11.json`), populate
`data/kfold/v2_<RES>_by_patient_rmS11/fold_{0..5}/{train,val,test}/` with
augmented PNGs (20 augmentations per training image).

### 3. Train

```
python code/run_all_experiments.py
```

Loops over the configurations in the paper (architecture ablation, binning
ablation, full vs subregion). Each configuration runs three seeds (42, 69,
137) over six folds. Total wall time on a single RTX 4090 is roughly 1-3
days depending on which configurations are enabled.

### 4. Sensitivity analyses (S5, S7)

```
python code/run_s11_sensitivity.py            # retrains 6 configs with S11 in
python code/bin_spectra_global_norm.py \      # global-axis binning
    --mzml data/mzml \
    --output data/raw/spectra_filter_global_512 \
    --bins 512 --workers 4
python code/run_global_norm_comparison.py     # trains on the global-norm imgs
```

### 5. Classical metabolomics baseline (S8)

```
python code/pick_peaks_classical.py \
    --mzml data/mzml \
    --output results \
    --patient-splits data/kfold/patient_splits.json \
    --workers 8
```

Runs FeatureFindingMetabo across all mzMLs, builds a samples x features
intensity matrix, and trains Random Forest + PLS-DA under the same
patient-stratified 6-fold splits as the CNN.

### 6. GradCAM and occlusion control (Section 3.6)

These scripts require both saved model checkpoints and the per-sample PNG
images. Model weights are not included in the repository; request them from
the corresponding author.

```
python code/run_gradcam_analysis.py
python code/gradcam_occlusion_test.py
python code/make_occlusion_figure.py
```

## Cross-validation splits

`data/kfold/patient_splits.json` and `patient_splits_withS11.json` define
the exact patient-to-fold assignments used in the paper. To reuse them:

```python
import json
with open("data/kfold/patient_splits.json") as f:
    splits = {int(k): {s: set(v) for s, v in d.items()}
              for k, d in json.load(f).items()}
# splits[fold_idx]['train' | 'val' | 'test'] -> set of patient IDs (e.g. "S24")
```

## Citation

If you build on this work, please cite the manuscript (under review). A
bibtex entry will be added on acceptance.

## License

MIT - see `LICENSE`.
