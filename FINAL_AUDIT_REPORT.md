# Final Audit Report — Exoplanet Pipeline Parts 1–10

## Audit scope

This audit rechecked the package structure, code syntax, test suite, demo execution, generated-output workflow, and scientific design consistency for the TESS exoplanet detection/classification problem statement.

## What was checked

1. **Repository structure**
   - Source package under `src/exoplanet_pipeline/`.
   - Scripts under `scripts/`.
   - Demo notebooks under `notebooks/`.
   - Tests under `tests/`.
   - Historical partial design plans, partial report drafts, generated demo outputs, and Python bytecode were removed from version control.

2. **Syntax/import sanity**
   - Ran `PYTHONPYCACHEPREFIX=.context/pycache .context/audit-venv/bin/python -m compileall -q src scripts tests`.
   - Result: passed.

3. **Unit/integration tests**
   - Ran:
     ```bash
     PYTHONDONTWRITEBYTECODE=1 .context/audit-venv/bin/python -m pytest --disable-warnings
     ```
   - Result: `31 passed`.

4. **Packaging sanity**
   - Ran `.context/audit-venv/bin/python -m pip check`.
   - Result: passed after moving runtime-optional TLS/wotan integrations out of mandatory dependencies.

5. **Notebook integrity**
   - All notebooks in `notebooks/` are valid JSON.

6. **Synthetic end-to-end demo**
   - Ran:
     ```bash
     .context/audit-venv/bin/python scripts/run_parts_9_10_synthetic_batch.py --output-dir .context/audit_parts_9_10 --n-periods 500
     ```
   - Result: completed successfully.
   - Processed 3 synthetic targets and produced 3 final candidates.
   - Correctly separated the demonstration cases into:
     - planetary transit candidate,
     - eclipsing binary,
     - blend/contaminated signal.

7. **Repository hygiene**
   - Removed tracked `__pycache__`/`.pyc` files.
   - Removed tracked generated `outputs_*` artifacts; demos regenerate them as needed.
   - Removed superseded partial design/report drafts while keeping the final report, README, audit note, tests, scripts, and notebooks.
   - Removed legacy duplicate scripts covered by maintained Parts 1-5 commands.
   - Added `.gitignore` rules for Python caches, egg-info, local virtualenvs, `.context`, and generated data/plot/output directories.

8. **Compatibility fixes from this audit**
   - Clipped BLS duration grids so Astropy does not reject duration values longer than the shortest trial period.
   - Added a transparent NumPy-box fallback when Astropy BLS fails at runtime.
   - Replaced `DataFrame.to_markdown()` in final candidate review generation with a small dependency-free markdown table writer.
   - Added final-catalog schema validation before writing submission catalogs.
   - Aligned uncertainty rows by `candidate_id` instead of row order.
   - Fixed the Parts 1-8 single-target demo plotting call.
   - Added executable CLI/demo smoke tests and generated TESS-like FITS ingestion tests.
   - Made the compact Parts 7-8 validation demo class-balanced.
   - Aligned injection-recovery uncertainty estimates to the selected `candidate_id`.
   - Reported optional `wotan` detrending fallbacks in preprocessing warnings.
   - Added a GitHub Actions test workflow for Python 3.10 and 3.12.
   - Added `exoplanet_pipeline.__version__`.
   - Made top-level package exports explicit instead of hiding import errors behind broad `try/except` blocks.
   - Updated install/test docs for `python3 -m ...` and editable installs.

## Scientific/design checks

### Good

- Detection is separated from classification, which is the correct architecture.
- The pipeline preserves raw, normalized, and detrended flux.
- SAP/PDCSAP flux source is recorded honestly.
- Synthetic fallback is not used for real-data failures.
- CROWDSAP is treated as a contamination-risk feature, not an automatic rejection.
- Secondary-eclipse logic searches phase 0.5 and does not reuse the primary-transit phase.
- Centroid shift uses residual centroid motion, not normalized centroid ratios.
- Final catalog contains period, duration, depth, SNR, confidence, class, risk summaries, and recommended action.
- Validation framework supports curated labels and injection-recovery experiments.
- AI classifier includes physical guardrails for strong EB/blend/systematic evidence.

### Important limitations to keep honest

- The fast default demo uses a small BLS grid for speed; real sector runs should use larger grids. TLS remains runtime-optional because the current `transitleastsquares` dependency stack is not reliable on modern Python.
- The transit fitting is a robust box/profile refinement, not a full limb-darkened physical transit model.
- Planet radius is only meaningful when reliable stellar radius metadata exists.
- Crowded-field candidates still need stronger target-pixel-file difference imaging and Gaia nearby-source checks for high confidence.
- The AI classifier is a framework; real performance depends on the organizer-provided curated labeled dataset.
- Confidence is a triage confidence, not formal astronomical validation probability.

## Final status

The package is now a smaller, coherent, tested, submission-grade skeleton for the full problem statement. It is ready for:

1. running on the organizer's curated labeled dataset,
2. running on local TESS FITS light curves,
3. generating a final candidate catalog and submission assets,
4. extending with target-pixel-file/Gaia vetting if time remains.
