# Sea Ice Segmentation Fusion

Train and evaluate sea ice type segmentation models using ALOS-2 L-band and
Sentinel-1 C-band SAR imagery. Experiments are selected by an INI file. The
repository contains one training entry point (`main.py`), one evaluation entry
point (`evaluate.py`), and one model module (`models.py`).

## Installation

```bash
conda env create -f env.yml
conda activate sea-ice-segment-fusion
python -m pytest -q tests
```

Alternatively, install Python dependencies with `pip install -r requirements.txt`.

On macOS, installing geospatial dependencies through conda-forge may help:

```bash
conda install -c conda-forge rasterio rioxarray
```

The Belgica Bank raster images, rasterized labels, and train/validation CSV
files must be supplied separately.

## Data and output directories

Set environment variables to locate data and write results:

```bash
export OUTPUT_ROOT="results"
export DATASET_ROOT="/Users/jenmacdonald/Documents/datasets/Belgica_Bank"
```

Relative root values are resolved relative to the repository directory. In the
INI files, `[paths]` defines fallback directories when the environment
variables are not set:

```ini
[paths]
default_dataset_root = D:/jema2085/data/Belgica_Bank
default_output_root = D:/jema2085/data/Belgica_Bank/results
```

Path resolution examples when the environment variables above are set:

| Configured path | Resolved path |
|---|---|
| `results/SA/alos2-jan` | `./results/SA/alos2-jan` |
| `results/BB_IO-SA-alos2-jan.csv` | `$DATASET_ROOT/BB_IO-SA-alos2-jan.csv` |
| `denoised_resampled/example.tif` | `$DATASET_ROOT/denoised_resampled/example.tif` |
| `labels_rasterized/SA/example.tif` | `$DATASET_ROOT/labels_rasterized/SA/example.tif` |

For `run_dirs` and `csv_files`, the portion after `results/` is appended to
the corresponding root. CSVs must be placed directly under `DATASET_ROOT`.
SAR and label raster paths are resolved under `DATASET_ROOT`; absolute paths
in CSVs that begin with the configured dataset directory are relocated there.

On Windows PowerShell, environment variables may be set for the terminal
session as follows:

```powershell
$env:OUTPUT_ROOT = "D:/my-experiment-results"
$env:DATASET_ROOT = "D:/my-datasets/Belgica_Bank"
```

On macOS and Linux, both environment variables must be set when the configured
fallback directories refer to a Windows drive.

## Experiment configurations

There are 16 INI files: one training/evaluation pair for each experiment.

| Experiment | Training INI | Evaluation INI |
|---|---|---|
| L-band only | `configs/SA_single_alos.ini` | `configs/eval_SA_single_alos2.ini` |
| C-band only | `configs/SA_single_s1.ini` | `configs/eval_SA_single_s1.ini` |
| Early fusion | `configs/SA_early.ini` | `configs/eval_SA_early.ini` |
| Late fusion | `configs/SA_late.ini` | `configs/eval_SA_late.ini` |
| Late fusion (L + L) | `configs/SA_late_alos2_alos2.ini` | `configs/eval_SA_late_alos2_alos2.ini` |
| Late fusion (C + C) | `configs/SA_late_s1_s1.ini` | `configs/eval_SA_late_s1_s1.ini` |
| Gated, L-band target | `configs/SA_gated_alos2.ini` | `configs/eval_SA_gated_alos2.ini` |
| Gated, C-band target | `configs/SA_gated_s1.ini` | `configs/eval_SA_gated_s1.ini` |

The late-fusion L+L and C+C controls use the same two independently trained
encoders and concatenation decoder as L+C late fusion, with identical trainable
parameter counts. Each encoder receives the same source band for its control.
Both controls use the paired-scene training CSVs, so the training and test
scenes and labels remain the same as for L+C; the unused SAR source is not
fed into the model. Each encoder's input is normalized with the selected
source's fold statistics. Their runs, checkpoints, and evaluation summaries
are stored separately from the L+C experiment.

Run the extra controls with:

```bash
python main.py --config-file configs/SA_late_alos2_alos2.ini --seed 0
python evaluate.py --config-file configs/eval_SA_late_alos2_alos2.ini --seed 0
python main.py --config-file configs/SA_late_s1_s1.ini --seed 0
python evaluate.py --config-file configs/eval_SA_late_s1_s1.ini --seed 0
```

The `[experiment]` and `[model]` sections define the model and sensor behavior.
For gated models, `gated_target_band` selects the sensor used for validation
and target-only inference, and `gated_switch_epoch` selects the transition to
target-only training. The `[eval]` section specifies the evaluated sensor.

## Training and evaluation

Development run (one epoch, one fold, one sample per scene, batch size one,
CPU by default):

```bash
python main.py --config-file configs/SA_late.ini --dev
python evaluate.py --config-file configs/eval_SA_late.ini --dev
```

Full run (five folds and two test scenes per fold):

```bash
python main.py --config-file configs/SA_late.ini --seed 0
python evaluate.py --config-file configs/eval_SA_late.ini --seed 0
```

### Class-weighted cross-entropy

By default, training uses **standard cross-entropy** (CE). Pass `--class-weighted`
for class-weighted CE. Per-fold weights are computed using **training labels only**:
weights are inverse-square-root class frequencies, normalized to an average of
1 over classes present in training, and capped at 10. Classes absent from the
training split receive weight 0. The labels marked with `ignore_index` do not
contribute. No other loss functions or class-aware sampling strategies are
available. Training and validation use the selected CE loss for `val_loss` and
early stopping.

```bash
python main.py --config-file configs/SA_late.ini --seed 0 --class-weighted
python evaluate.py --config-file configs/eval_SA_late.ini --seed 0 --class-weighted
```

Use the flag on **both** commands for weighted runs. Standard CE and weighted
CE save to separate directories (`seed_0/` and `seed_0/class_weighted/`). Aggregate F1 summaries are also
separated by this directory suffix. The
training manifest and checkpoint record the selected objective and exact weights;
evaluation rejects mismatches. Other settings and the training patch sampler
are unchanged.

`--config-file` is required and may be a path or a filename in `configs/`.
Both entry points accept `--seed` (default `0`), `--dev`, `--class-weighted`, and
`--device` (`auto`, `cpu`, `cuda`, or `mps`). In development mode, model
pretraining downloads are disabled. In full mode, model pretraining follows
the selected INI. Training settings such as crop length, epoch limits, batch
size, and sample counts are controlled by the INI.

### Gated fusion ablations

Gated models support `--ablation` with `none` (default), `no-curriculum`, and
`no-gate-centering`. The standard gate applies a fixed centering adjustment:

```text
g_adjusted = g - 0.5 * (g - 0.5)^2 * sign(g - 0.5)
f_fused = g_adjusted * L + (1 - g_adjusted) * C
```

With the standard curriculum, both sensors contribute during the first 15
epochs, followed by target-only training. In the `no-curriculum` ablation,
both encoders contribute throughout training while validation still uses the
target sensor. In the `no-gate-centering` ablation, the sigmoid gate is used
directly.

```bash
# Standard gated L-band target
python main.py --config-file configs/SA_gated_alos2.ini --seed 0
python evaluate.py --config-file configs/eval_SA_gated_alos2.ini --seed 0

# Gated L-band target without curriculum switching
python main.py --config-file configs/SA_gated_alos2.ini --ablation no-curriculum --seed 0
python evaluate.py --config-file configs/eval_SA_gated_alos2.ini --ablation no-curriculum --seed 0

# Gated L-band target without gate-centering
python main.py --config-file configs/SA_gated_alos2.ini --ablation no-gate-centering --seed 0
python evaluate.py --config-file configs/eval_SA_gated_alos2.ini --ablation no-gate-centering --seed 0
```

Use the C-band gated configuration pair to run the same variants for the C-band
target. Training and evaluation must use the same ablation value. Ablation
artifacts are stored in separate folders to prevent checkpoint collisions.

## Outputs and metrics

Per-fold training outputs are stored under
`OUTPUT_ROOT/SA/<configured-run-directory>/<dev|full>/seed_<seed>/`, including
`model.pth`, a Lightning checkpoint, `run.json`, `train.cfg`, and training logs.
Evaluation writes class rasters, probability rasters (if enabled), correctness
rasters, per-scene metrics, and confusion matrices alongside the checkpoint.

Evaluation also produces aggregate metrics at
`OUTPUT_ROOT/SA/metrics/<experiment>/<dev|full>/seed_<seed>/evaluation_<sensor>/`:

- `scene_f1.csv`: weighted, macro, and per-class F1 with ground-truth supports,
  for every evaluated scene.
- `scene_weighted_f1.csv`: per-scene weighted F1 scores.
- `f1_summary.csv`: scene-level mean, sample standard deviation, number of
  contributing scenes, and pixel-pooled F1.
- `metrics_average.txt`: readable scene-level and pooled summary.
- `confusion_matrix_average.pdf`: pooled confusion matrix visualization.

A class that has no labeled pixels in a scene is reported as N/A for that
scene; it is excluded from that scene's macro F1 and across-scene per-class
mean. The summaries distinguish scene-level descriptive statistics from
metrics pooled across all labeled pixels. A single seed does not provide an
independent-seed confidence interval.

Training and evaluation INIs contain identical per-fold normalization
statistics. The evaluator also checks them against the saved `run.json`
manifest before processing any scene. Raster inference operates on entire
scenes in a single forward pass and can require substantial CPU/GPU memory.
During training, invalid SAR pixels are replaced with each channel's
patch-local mean before normalization.

## Determinism

Python, NumPy, PyTorch, and Lightning RNGs are seeded. Deterministic PyTorch
algorithms and cuDNN settings are enabled, and CUDA TF32 is disabled. One RNG
stream is maintained across the five-fold run. Exact numerical agreement
across operating systems, GPU types, or library versions is not guaranteed.
The manifest records the software environment and selected settings.

## Utilities

`utils/compute_mean_std.py` calculates band-wise means and standard deviations
from raster paths listed in a CSV file. Its CLI options are documented with
`python utils/compute_mean_std.py -h`.
