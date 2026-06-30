from pathlib import Path
import os
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_main_scripts_show_help():
    scripts = [
        "scripts/run_parts_1_to_5_fits.py",
        "scripts/run_parts_9_10_fits_directory.py",
        "scripts/train_ai_classifier_from_catalog.py",
        "scripts/validate_candidate_catalog.py",
    ]
    for script in scripts:
        result = _run_script(script, "--help")
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_fits_directory_cli_fails_fast_on_empty_directory(tmp_path: Path):
    result = _run_script("scripts/run_parts_9_10_fits_directory.py", str(tmp_path))
    assert result.returncode == 2
    assert "No FITS files found" in result.stderr


def test_parts_1_to_8_single_demo_executes(tmp_path: Path):
    result = _run_script(str(ROOT / "scripts/run_parts_1_to_8_synthetic_single.py"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs_parts_1_to_8_single" / "parts_1_to_8_single_catalog.csv").exists()
    assert (tmp_path / "outputs_parts_1_to_8_single" / "single_detection.png").exists()


def test_parts_7_8_validation_demo_executes_with_multiple_classes(tmp_path: Path):
    result = _run_script(str(ROOT / "scripts/run_parts_7_8_synthetic_validation.py"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    catalog = tmp_path / "outputs_parts_7_8" / "parts_7_8_injection_recovery_catalog.csv"
    assert catalog.exists()
    labels = set(pd.read_csv(catalog)["label"])
    assert {"PLANETARY_TRANSIT_CANDIDATE", "ECLIPSING_BINARY", "BLEND_OR_CONTAMINATED_SIGNAL"}.issubset(labels)
