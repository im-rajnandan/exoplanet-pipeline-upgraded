# Kaggle Real-Data Runbook

This is the recommended path for running the pipeline on real TESS light curves
and a curated labeled dataset. Synthetic scripts are only smoke tests.

## What The Code Expects

Use these inputs on Kaggle:

- A curated label CSV, for example `/kaggle/input/curated/curated_labels.csv`.
- TESS light-curve FITS files, either attached as a Kaggle dataset or downloaded
  from MAST during the notebook run.
- Optional TIC/CTL catalog files from STScI when selecting science targets by
  TIC ID instead of starting from a local FITS directory.

The curated label CSV must contain:

```text
tic_id,label
```

It should contain these optional columns when available:

```text
sector,period_days,epoch_time,duration_days,duration_hours,depth_ppm
```

Supported label aliases include planet/transit/exoplanet, eb/eclipsing_binary,
blend/background_eb, starspot/stellar_variability, systematic/artifact, and
no_signal/noise. Unknown labels are intentionally dropped by the training code
instead of being guessed.

Epoch note: TESS light-curve FITS times are usually BTJD. If the curated labels
store BJD-like epochs such as 2459000.x, pass `--epoch-offset 2457000`. If you
normalize TOI metadata through `--source tess-toi`, the code converts those
epochs to BTJD for you.

## Install In A Kaggle Notebook

Enable internet if you plan to download from MAST or NASA Exoplanet Archive.
If internet is disabled, attach a Kaggle dataset containing the repo, FITS
files, and any predownloaded metadata.

```bash
cd /kaggle/working
git clone <your-repo-url> exoplanet-pipeline
cd exoplanet-pipeline
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mast,deep]"
```

Use the `deep` extra only if you will train or run the CNN. The sklearn tabular
classifier does not need a GPU.

## First Smoke Test

Run this before touching the real data:

```bash
python scripts/run_parts_9_10_synthetic_batch.py \
  --output-dir /kaggle/working/smoke_outputs \
  --n-periods 500
```

Expected output:

```text
/kaggle/working/smoke_outputs/batch_final_candidate_catalog.csv
/kaggle/working/smoke_outputs/submission_assets/
```

## Build A Tabular AI Model From Curated Labels

This is the main supervised classifier path for the project objective. It turns
curated TIC labels plus FITS files into the same Parts 1-5 feature columns used
by the science pipeline, then trains the Part 6 sklearn model.

```bash
python scripts/build_labeled_candidate_catalog.py \
  /kaggle/input/curated/curated_labels.csv \
  --label-col label \
  --lightcurve-dir /kaggle/input/tess-lightcurves \
  --output-dir /kaggle/working/outputs_labeled_candidates \
  --epoch-offset 2457000 \
  --n-periods 3000 \
  --no-variants
```

If light curves are not attached locally and Kaggle internet is enabled, use a
writable cache and allow downloads:

```bash
python scripts/build_labeled_candidate_catalog.py \
  /kaggle/input/curated/curated_labels.csv \
  --label-col label \
  --lightcurve-dir /kaggle/working/data/lightcurves \
  --download-missing \
  --output-dir /kaggle/working/outputs_labeled_candidates \
  --epoch-offset 2457000 \
  --n-periods 3000 \
  --no-variants
```

Check the manifest before training:

```bash
python - <<'PY'
import pandas as pd
manifest = pd.read_csv('/kaggle/working/outputs_labeled_candidates/labeled_candidate_manifest.csv')
print(manifest['status'].value_counts(dropna=False))
features = pd.read_csv('/kaggle/working/outputs_labeled_candidates/labeled_candidate_features.csv')
print(features['label'].value_counts(dropna=False))
print(features.shape)
PY
```

Train the model:

```bash
python scripts/train_ai_classifier_from_catalog.py \
  /kaggle/working/outputs_labeled_candidates/labeled_candidate_features.csv \
  --label-col label \
  --output-dir /kaggle/working/outputs_part6_ai \
  --model-type random_forest
```

Primary artifact:

```text
/kaggle/working/outputs_part6_ai/part6_ai_classifier.joblib
```

## Optional CNN Training

Use this if you want the CNN triage score in addition to the tabular model. This
path uses public TOI metadata and MAST light curves.

```bash
python scripts/download_public_metadata.py \
  --source tess-toi \
  --output-dir /kaggle/working/data/public \
  --top 2000

python scripts/build_public_cnn_examples.py \
  /kaggle/working/data/public/metadata/tess-toi_normalized_metadata.csv \
  --output-dir /kaggle/working/data/public/cnn_examples \
  --lightcurve-dir /kaggle/working/data/public/lightcurves \
  --download-missing \
  --max-rows 2000 \
  --n-workers 8

python scripts/train_cnn_classifier.py \
  /kaggle/working/data/public/cnn_examples \
  --output-dir /kaggle/working/outputs_cnn \
  --epochs 20 \
  --batch-size 32 \
  --device cuda
```

Primary artifact:

```text
/kaggle/working/outputs_cnn/cnn_model.pt
```

## Run On Science FITS Files

Use this when the science light curves are already attached as a Kaggle dataset.

```bash
python scripts/run_parts_9_10_fits_directory.py \
  /kaggle/input/science-fits \
  --output-dir /kaggle/working/outputs_science \
  --ai-model /kaggle/working/outputs_part6_ai/part6_ai_classifier.joblib \
  --cnn-model /kaggle/working/outputs_cnn \
  --n-workers 2 \
  --n-periods 4000 \
  --timeout-seconds 300 \
  --no-variants
```

If you are not using the CNN, remove `--cnn-model`. If you are not using the
sklearn model, remove `--ai-model`.

Main outputs:

```text
/kaggle/working/outputs_science/batch_final_candidate_catalog.csv
/kaggle/working/outputs_science/batch_target_summary.csv
/kaggle/working/outputs_science/batch_failure_log.csv
/kaggle/working/outputs_science/submission_assets/
```

## Run From TIC/CTL And Download From MAST

Use this if the science run starts from STScI TIC/CTL rows instead of a local
FITS directory.

```bash
python scripts/build_tic_ctl_target_list.py \
  /kaggle/input/tic-ctl/exo_CTL_08.01.csv \
  --catalog-type ctl \
  --max-targets 1000 \
  --output-csv /kaggle/working/data/tic_ctl_targets.csv

python scripts/run_tic_ctl_pipeline.py \
  /kaggle/working/data/tic_ctl_targets.csv \
  --catalog-type target-list \
  --download-dir /kaggle/working/data/lightcurves \
  --output-dir /kaggle/working/outputs_tic_ctl \
  --max-targets 1000 \
  --ai-model /kaggle/working/outputs_part6_ai/part6_ai_classifier.joblib \
  --n-workers 4 \
  --n-periods 4000 \
  --timeout-seconds 300
```

For a first real run, start with `--max-targets 50` or `--max-targets 100`.
Scale only after the download manifest and failure rate look sane.

## Validate Against Curated Labels

If your science catalog contains labels, or you join labels back onto the final
candidate catalog by TIC ID/candidate, run:

```bash
python scripts/validate_candidate_catalog.py \
  --catalog /kaggle/working/outputs_science/batch_final_candidate_catalog_labeled.csv \
  --label-col label \
  --pred-col final_predicted_class \
  --out-dir /kaggle/working/outputs_validation
```

Then regenerate report assets with validation metrics included:

```bash
python scripts/generate_final_submission_assets.py \
  /kaggle/working/outputs_science/batch_final_candidate_catalog.csv \
  --output-dir /kaggle/working/outputs_science/submission_assets_validated \
  --validation-report /kaggle/working/outputs_validation/validation_report.json
```

## What To Record For The Final Report

Keep these files from each Kaggle run:

- Input data source names, sector numbers, TIC/CTL file names, and row counts.
- `labeled_candidate_summary.json` and `labeled_candidate_manifest.csv`.
- `part6_ai_classifier_metrics.json`,
  `part6_ai_classifier_feature_importance.csv`, confusion matrix, and feature
  importance plot.
- CNN metrics if the CNN was used.
- `batch_run_manifest.json`, `batch_target_summary.csv`, and
  `batch_failure_log.csv`.
- `batch_final_candidate_catalog.csv` and `submission_assets/`.
- Validation report JSON and plots when labels are available.

## Remaining Work Against The Objective

The repository now has the executable plumbing for the objective, but the
scientific result still depends on running it on the expected data:

- Download or attach a real TESS high-cadence sector or a representative subset.
- Attach the curated organizer labels with TIC IDs and, ideally, ephemerides.
- Build the labeled feature catalog and confirm enough rows survive per class.
- Train the supervised tabular model and inspect holdout metrics.
- Optionally train the CNN on public TOI examples.
- Run the trained model on science FITS files or a TIC/CTL-selected target list.
- Validate detection, classification, period, duration, depth, and confidence
  calibration on held-out labeled data.
- Write the final 3-page report from the generated methodology, metrics, and
  uncertainty artifacts.

The highest-risk open item is data coverage: TIC/CTL itself is only a target
catalog, not labeled light-curve data. The curated labels and actual FITS files
must be present or downloadable for a meaningful Kaggle run.
