import argparse
from pathlib import Path

from exoplanet_pipeline import PipelineConfig
from exoplanet_pipeline.pipeline import run_parts_1_to_5_from_fits
from exoplanet_pipeline.diagnostics import plot_preprocessing, plot_detection, plot_vetting_summary


def main():
    parser = argparse.ArgumentParser(description="Run Parts 1-5 on one local TESS light-curve FITS file.")
    parser.add_argument("fits_file")
    parser.add_argument("--out", default="outputs_parts_1_to_5")
    parser.add_argument("--method", choices=["bls", "tls", "both"], default="bls")
    args = parser.parse_args()

    cfg = PipelineConfig(detection_method=args.method)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_parts_1_to_5_from_fits(args.fits_file, config=cfg)
    clean = result["clean"]
    detection = result["detection"]
    stem = f"TIC_{clean.tic_id or 'unknown'}_S{clean.sector or 'unknown'}"

    plot_preprocessing(clean, out_dir / f"{stem}_preprocessing.png")
    plot_detection(clean, detection, out_dir / f"{stem}_detection.png")
    if not result["catalog"].empty:
        result["catalog"].to_csv(out_dir / f"{stem}_candidate_catalog_parts_1_to_5.csv", index=False)
        plot_vetting_summary(
            clean,
            detection.candidates[0],
            result["fit_results"][0],
            result["vetting_results"][0],
            result["classification_results"][0],
            out_dir / f"{stem}_vetting_summary.png",
        )
    print(result["catalog"].T if not result["catalog"].empty else detection.status)


if __name__ == "__main__":
    main()
