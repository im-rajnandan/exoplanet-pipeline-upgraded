#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet_pipeline.classify import classify_candidate_rule_based
from exoplanet_pipeline.cnn import (
    predict_cnn_examples,
    save_cnn_bundle,
    train_cnn_classifier,
    training_example_from_candidate_views,
)
from exoplanet_pipeline.cnn_views import build_cnn_candidate_views, save_cnn_example_npz
from exoplanet_pipeline.config import PipelineConfig
from exoplanet_pipeline.fit import refine_candidate_parameters
from exoplanet_pipeline.preprocess import preprocess_raw_lightcurve
from exoplanet_pipeline.schema import CandidateSignal
from exoplanet_pipeline.synthetic import make_synthetic_blend_lc, make_synthetic_eb_lc, make_synthetic_transit_lc
from exoplanet_pipeline.vetting import extract_vetting_features


def _candidate_from_truth(raw, period: float, t0: float, duration_hours: float, depth_ppm: float) -> CandidateSignal:
    return CandidateSignal(
        tic_id=raw.tic_id,
        sector=raw.sector,
        candidate_id=1,
        period_days=period,
        epoch_time=t0,
        duration_days=duration_hours / 24.0,
        depth_fraction=depth_ppm * 1e-6,
        depth_ppm=depth_ppm,
        snr=20.0,
        local_snr=20.0,
        sde=12.0,
        fap=None,
        n_transits=6,
        n_full_transits=6,
        n_in_transit_points=100,
        detection_method="truth_seeded_demo",
        flux_source="PDCSAP",
        detrend_variant="default",
        status="STRONG_DETECTION",
    )


def _demo_examples() -> list:
    specs = [
        ("PLANETARY_TRANSIT_CANDIDATE", make_synthetic_transit_lc, {"period_days": 3.0, "depth_ppm": 1300, "duration_hours": 2.0}),
        ("ECLIPSING_BINARY", make_synthetic_eb_lc, {"period_days": 4.0, "primary_depth_ppm": 18000, "secondary_depth_ppm": 5000, "duration_hours": 3.0}),
        ("BLEND_OR_CONTAMINATED_SIGNAL", make_synthetic_blend_lc, {"period_days": 3.5, "observed_depth_ppm": 1200, "duration_hours": 2.0}),
    ]
    examples = []
    for i in range(4):
        for label, factory, kwargs in specs:
            raw = factory(tic_id=990000 + len(examples), random_seed=100 + len(examples), **kwargs)
            clean = preprocess_raw_lightcurve(raw, PipelineConfig(detrend_method="none"))
            if label == "ECLIPSING_BINARY":
                depth = kwargs["primary_depth_ppm"]
            elif label == "BLEND_OR_CONTAMINATED_SIGNAL":
                depth = kwargs["observed_depth_ppm"]
            else:
                depth = kwargs["depth_ppm"]
            cand = _candidate_from_truth(raw, kwargs["period_days"], 1.0, kwargs["duration_hours"], depth)
            fit = refine_candidate_parameters(clean, cand, n_bootstrap=20)
            vet = extract_vetting_features(clean, cand, fit)
            cls = classify_candidate_rule_based(cand, fit, vet)
            view = build_cnn_candidate_views(clean, cand, fit, vet)
            view.metadata.update({
                "class_predicted_class": cls.predicted_class,
                "vet_secondary_sigma": vet.secondary_sigma,
                "vet_odd_even_sigma": vet.odd_even_sigma,
                "vet_centroid_shift_sigma": vet.centroid_shift_sigma,
                "vet_crowding_risk": vet.crowding_risk,
                "vet_data_quality_score": vet.data_quality_score,
                "fit_snr": fit.snr,
            })
            examples.append(training_example_from_candidate_views(view, canonical_label=label))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny CPU CNN vetter and write demo prediction columns.")
    parser.add_argument("--output-dir", default="outputs_cnn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = Path(args.output_dir)
    examples_dir = out / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples = _demo_examples()
    for i, ex in enumerate(examples):
        proxy = type("_ExampleProxy", (), {
            "views": {"global_flux": ex.global_flux},
            "local_tensor": lambda self, ex=ex: ex.local_views,
            "scalar_vector": lambda self, names=None, ex=ex: ex.scalar_features,
            "metadata": ex.metadata,
        })()
        save_cnn_example_npz(proxy, str(examples_dir / f"demo_{i:03d}.npz"), canonical_label=ex.canonical_label, binary_label=ex.binary_label)

    result = train_cnn_classifier(examples, epochs=args.epochs, batch_size=6, seed=42, device=args.device)
    paths = save_cnn_bundle(
        result.model,
        out,
        result.config,
        result.scalar_scaler,
        result.label_map,
        {**result.metrics, "warnings": result.warnings},
    )
    pred = predict_cnn_examples(result.bundle(), examples, device=args.device)
    pred_path = out / "cnn_demo_predictions.csv"
    pred.to_csv(pred_path, index=False)
    print("Part 6 CNN demo complete.")
    print(f"Examples: {examples_dir}")
    print(f"Predictions: {pred_path}")
    print("Final metrics:", result.metrics.get("final", {}))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
