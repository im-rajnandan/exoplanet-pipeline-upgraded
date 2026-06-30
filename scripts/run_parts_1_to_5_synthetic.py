from pathlib import Path
import pandas as pd

from exoplanet_pipeline import PipelineConfig
from exoplanet_pipeline.synthetic import make_synthetic_transit_lc, make_synthetic_eb_lc, make_synthetic_blend_lc
from exoplanet_pipeline.pipeline import run_parts_1_to_5_from_raw
from exoplanet_pipeline.diagnostics import plot_preprocessing, plot_detection, plot_vetting_summary


def main():
    cfg = PipelineConfig(detection_method="bls", n_periods=900, n_durations=8, detection_use_variants=False)
    out_dir = Path("outputs_parts_1_to_5")
    out_dir.mkdir(exist_ok=True)

    samples = [
        ("planet", make_synthetic_transit_lc()),
        ("eb", make_synthetic_eb_lc()),
        ("blend", make_synthetic_blend_lc()),
    ]

    all_rows = []
    for name, raw in samples:
        result = run_parts_1_to_5_from_raw(raw, config=cfg)
        clean = result["clean"]
        detection = result["detection"]
        plot_preprocessing(clean, out_dir / f"{name}_preprocessing.png")
        plot_detection(clean, detection, out_dir / f"{name}_detection.png")
        if not result["catalog"].empty:
            all_rows.append(result["catalog"].assign(sample=name))
            # Plot the top fully processed candidate.
            plot_vetting_summary(
                clean,
                detection.candidates[0],
                result["fit_results"][0],
                result["vetting_results"][0],
                result["classification_results"][0],
                out_dir / f"{name}_vetting_summary.png",
            )
        print(f"{name}: clean={clean.status}, detection={detection.status}")
        if result["classification_results"]:
            cls = result["classification_results"][0]
            print(f"  class={cls.predicted_class}, confidence={cls.confidence:.3f}, evidence={cls.evidence}")

    if all_rows:
        catalog = pd.concat(all_rows, ignore_index=True)
        catalog.to_csv(out_dir / "parts_1_to_5_synthetic_catalog.csv", index=False)
        print(f"Wrote {out_dir / 'parts_1_to_5_synthetic_catalog.csv'}")


if __name__ == "__main__":
    main()
