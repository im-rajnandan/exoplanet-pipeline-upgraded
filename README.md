# Exoplanet Pipeline Parts 1-10

Modular pipeline for AI-enabled detection and triage of exoplanet-like signals in noisy TESS light curves.

The package keeps detection, physical vetting, AI classification, uncertainty, validation, and final catalog generation as separate stages. It outputs candidates for review; it does not astronomically validate planets by itself.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev,mast]"
```

For plain requirements-based environments:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements.txt
```

To run the walkthrough notebooks, install the optional notebook extra:

```bash
python3 -m pip install -e ".[notebook]"
```

To train or run the optional CNN vetter, install the deep extra. Use this on
GPU notebook runtimes such as Kaggle or Google Colab when training:

```bash
python3 -m pip install -e ".[deep]"
```

The maintained default path uses Astropy BLS detection and rolling-median detrending. `transitleastsquares` and `wotan` remain runtime-optional: if compatible versions are installed separately, `detection_method="tls"` and `detrend_method="wotan_biweight"` can use them.

## Main Commands

Run the Parts 1-5 synthetic detection, fitting, vetting, and baseline-classification demo:

```bash
python3 scripts/run_parts_1_to_5_synthetic.py
```

Run Parts 1-5 on one local TESS FITS light curve:

```bash
python3 scripts/run_parts_1_to_5_fits.py path/to/lightcurve.fits --method bls
```

Train or demo the Part 6 AI classifier:

```bash
python3 scripts/run_part6_synthetic_ai_demo.py
python3 scripts/train_ai_classifier_from_catalog.py curated_labeled_catalog.csv --label-col label --output-dir outputs_part6_ai
python3 scripts/predict_ai_classifier_catalog.py outputs_part6_ai/part6_ai_classifier.joblib candidate_catalog.csv --output-csv outputs_part6_ai/science_predictions.csv
```

Build and train the optional CNN vetter from official public data:

```bash
# If the required starting point is STScI TIC/CTL, first build a target list.
python3 scripts/build_tic_ctl_target_list.py path/to/exo_CTL_08.01.csv --catalog-type ctl --max-targets 1000 --output-csv data/public/tic_ctl_targets.csv
python3 scripts/run_tic_ctl_pipeline.py data/public/tic_ctl_targets.csv --catalog-type target-list --output-dir outputs_tic_ctl --max-targets 100 --cnn-model outputs_cnn

# Metadata from NASA Exoplanet Archive TAP. Use --top for smoke tests.
python3 scripts/download_public_metadata.py --source tess-toi --output-dir data/public
python3 scripts/download_public_metadata.py --source kepler-dr25 --output-dir data/public

# TESS/TOI CNN examples from official TOI metadata plus MAST light-curve files.
# Add --download-missing in an environment with astroquery/network access.
python3 scripts/build_public_cnn_examples.py data/public/metadata/tess-toi_normalized_metadata.csv --output-dir data/public/cnn_examples --lightcurve-dir data/public/lightcurves --download-missing --max-rows 100

# Train on CPU or a notebook GPU runtime. The bundle is state_dict + JSON sidecars.
python3 scripts/train_cnn_classifier.py data/public/cnn_examples --output-dir outputs_cnn --epochs 20 --device cuda
python3 scripts/predict_cnn_examples.py outputs_cnn data/public/cnn_examples --output-csv outputs_cnn/cnn_predictions.csv --device cuda
```

See [docs/public_cnn_training.md](docs/public_cnn_training.md) for the public-data-only Colab/Kaggle workflow and source notes.

Run uncertainty and validation utilities:

```bash
python3 scripts/run_parts_1_to_8_synthetic_single.py
python3 scripts/run_parts_7_8_synthetic_validation.py
python3 scripts/validate_candidate_catalog.py --catalog predictions.csv --label-col label --pred-col final_predicted_class --out-dir outputs_validation
```

Run Parts 9-10 batch processing and final submission asset generation:

```bash
python3 scripts/run_parts_9_10_synthetic_batch.py --output-dir outputs_parts_9_10 --n-periods 500
python3 scripts/run_parts_9_10_fits_directory.py /path/to/fits_dir --output-dir outputs_sector --max-targets 100 --cnn-model outputs_cnn
python3 scripts/generate_final_submission_assets.py outputs_sector/batch_final_candidate_catalog.csv --output-dir submission_assets
```

Generated `outputs*`, `data`, and `plots` directories are intentionally gitignored. Re-run the scripts to recreate demo catalogs, plots, model artifacts, reports, and final submission files.

## Package Layout

```text
src/exoplanet_pipeline/
  config.py                  central pipeline config
  schema.py                  dataclasses for light curves, candidates, fits, vetting, classes
  ingest.py                  local TESS FITS loading and optional MAST download helper
  quality.py                 TESS quality-mask helpers
  preprocess.py              flux selection, normalization, detrending, QC metrics
  detect.py                  BLS/TLS/NumPy-box periodic dip detection
  fit.py                     first-pass transit parameter refinement
  vetting.py                 odd/even, secondary, centroid, crowding, and shape features
  classify.py                transparent rule-based baseline classifier
  ml.py, ml_synthetic.py     supervised classifier and synthetic feature data
  cnn_views.py, cnn.py        optional CNN view generation, model, bundle APIs
  public_data.py             public metadata, TIC/CTL target lists, TAP helpers
  uncertainty.py             uncertainty and final confidence estimates
  validation.py              labeled-catalog and injection-recovery metrics
  batch.py                   batch execution, resume cache, failure logs
  final_catalog.py           harmonized final catalog and priority ranking
  final_outputs.py           review tables, plots, summaries, report draft

scripts/                      maintained command-line entry points
tests/                        unit and integration tests
notebooks/                    optional walkthrough notebooks
```

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests --tb=short --disable-warnings
```

Current audited status:

```text
base without PyTorch: 44 passed
deep smoke: 3 passed
```

## Scientific Limits

The final confidence is a triage confidence, not a formal validation probability. High-value or crowded-field candidates still need target-pixel difference imaging, Gaia/nearby-source checks, and domain review before being treated as validated planets.
