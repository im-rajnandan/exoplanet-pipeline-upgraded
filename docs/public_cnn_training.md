# Public-Data-Only CNN Training Workflow

This project should run from original public TIC/CTL target catalogs and public
MAST light curves. Synthetic examples remain useful only for fast smoke tests.

## Official Sources

- NASA Exoplanet Archive TAP service: official programmatic access for TOI and
  Kepler DR25 metadata.
  <https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html>
- TOI table column definitions, including `tid`, `tfopwg_disp`, `pl_orbper`,
  `pl_tranmid`, `pl_trandurh`, `pl_trandep`, and `sectors`.
  <https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html>
- NASA Exoplanet Archive program-interface notes for Kepler DR25 KOI/TCE TAP
  table names such as `q1_q17_dr25_koi` and `q1_q17_dr25_tce`.
  <https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html>
- MAST/astroquery documentation for TESS data products and DV products.
  <https://astroquery.readthedocs.io/en/latest/mast/mast.html>
- STScI TIC/CTL downloads, including xCTL and TIC declination chunks.
  <https://archive.stsci.edu/tess/tic_ctl.html>

## Recommended GPU Notebook Setup

Use Kaggle or Google Colab for training. Keep downloaded data under ignored
paths and save final bundles to cloud storage if needed.

```bash
git clone <your-repo-url>
cd exoplanet-detection-using-light-curves
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mast,deep]"
```

For Colab, choose `Runtime > Change runtime type > T4 GPU` when available. For
Kaggle, enable GPU in notebook settings. Training scripts default to CPU, so pass
`--device cuda` when a GPU is available.

## Start From TIC/CTL Target Catalogs

TIC/CTL files are target/stellar catalogs, not light curves. The pipeline uses
them to choose TIC IDs, then downloads matching TESS light-curve FITS files from
MAST before running detection and classification.

Build a target list from the 4-column xCTL file:

```bash
python scripts/build_tic_ctl_target_list.py \
  data/public/metadata/exo_CTL_08.01.csv \
  --catalog-type ctl \
  --max-targets 1000 \
  --output-csv data/public/tic_ctl_targets.csv
```

Build a target list from a TIC declination chunk:

```bash
python scripts/build_tic_ctl_target_list.py \
  data/public/metadata/tic_dec88_00N__90_00N.csv.gz \
  --catalog-type tic \
  --nrows 10000 \
  --max-targets 1000 \
  --output-csv data/public/tic_targets.csv
```

Run the full pipeline from a TIC/CTL target list:

```bash
python scripts/run_tic_ctl_pipeline.py \
  data/public/tic_ctl_targets.csv \
  --catalog-type target-list \
  --download-dir data/public/lightcurves \
  --output-dir outputs_tic_ctl \
  --max-targets 100 \
  --cnn-model outputs_cnn
```

This produces the normal final candidate catalog, visual assets, and report
assets after light curves have been downloaded.

## Optional Label Metadata For Training

```bash
python scripts/download_public_metadata.py --source tess-toi --output-dir data/public
python scripts/download_public_metadata.py --source kepler-dr25 --output-dir data/public
```

For quick notebook smoke tests:

```bash
python scripts/download_public_metadata.py --source tess-toi --output-dir data/public --top 50
```

The downloader writes:

```text
data/public/metadata/<source>_raw_metadata.csv
data/public/metadata/<source>_normalized_metadata.csv
data/public/metadata/<source>_manifest.json
data/public/metadata/<source>_tap_query.txt
```

## Build TESS/TOI CNN Examples For Supervised Training

The automatic supervised example builder targets TESS TOIs because TOI metadata
provides public dispositions and ephemerides. TIC/CTL alone does not provide
planet/false-positive labels.

```bash
python scripts/build_public_cnn_examples.py \
  data/public/metadata/tess-toi_normalized_metadata.csv \
  --output-dir data/public/cnn_examples \
  --lightcurve-dir data/public/lightcurves \
  --download-missing \
  --max-rows 1000
```

Notes:

- Labels come from public TOI dispositions. Broad false-positive dispositions are
  used as binary `false_positive_or_other` labels unless a reliable canonical
  subtype is present.
- The script seeds candidates from public ephemerides and builds fixed CNN views
  from official MAST light curves.
- Missing or failed downloads are recorded in
  `data/public/cnn_examples/cnn_example_manifest.csv`.

## Train On A Free GPU

```bash
python scripts/train_cnn_classifier.py \
  data/public/cnn_examples \
  --output-dir outputs_cnn \
  --epochs 20 \
  --batch-size 32 \
  --device cuda
```

The bundle format is:

```text
outputs_cnn/cnn_model.pt
outputs_cnn/cnn_config.json
outputs_cnn/cnn_scalar_scaler.json
outputs_cnn/cnn_label_map.json
outputs_cnn/cnn_metrics.json
```

Run predictions on saved examples:

```bash
python scripts/predict_cnn_examples.py \
  outputs_cnn \
  data/public/cnn_examples \
  --output-csv outputs_cnn/cnn_predictions.csv \
  --device cuda
```

Use the CNN bundle in the existing pipeline:

```bash
python scripts/run_parts_9_10_fits_directory.py \
  data/public/lightcurves \
  --output-dir outputs_sector_cnn \
  --cnn-model outputs_cnn
```

## What Still Needs Scientific Validation

Before treating the CNN as a scientific model, run target-split validation by
`tic_id`, inspect false positives and false negatives, check calibration, and
document the exact metadata snapshot and MAST products used. CNN confidence is a
triage score, not astronomical validation probability.
