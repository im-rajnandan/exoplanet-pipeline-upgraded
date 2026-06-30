from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .schema import CleanLightCurve, CandidateSignal, DetectionResult
from .detect import make_transit_mask


def plot_preprocessing(clean: CleanLightCurve, output_path: str | Path | None = None):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    title = (
        f"TIC {clean.tic_id} | Sector {clean.sector} | {clean.selected_flux_source} | "
        f"status={clean.status} | noise={clean.qc.get('robust_noise_ppm', float('nan')):.0f} ppm | "
        f"CROWDSAP={clean.qc.get('crowdsap', 'NA')}"
    )
    fig.suptitle(title, fontsize=12)

    axes[0].plot(clean.time, clean.flux_raw_selected, ".", ms=2)
    axes[0].set_ylabel("raw selected flux")

    axes[1].plot(clean.time, clean.flux_normalized, ".", ms=2)
    if clean.trend is not None:
        axes[1].plot(clean.time, clean.trend, lw=1)
    axes[1].set_ylabel("normalized + trend")

    axes[2].plot(clean.time, clean.flux_detrended, ".", ms=2)
    axes[2].axhline(1.0, lw=1)
    axes[2].set_ylabel("detrended flux")

    residual = clean.flux_detrended - np.nanmedian(clean.flux_detrended)
    axes[3].hist(residual[np.isfinite(residual)] * 1e6, bins=80)
    axes[3].set_xlabel("residual ppm")
    axes[3].set_ylabel("count")

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_phase(phase: np.ndarray, flux: np.ndarray, bins: int = 100):
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(bins, np.nan)
    for i in range(bins):
        m = (phase >= edges[i]) & (phase < edges[i + 1])
        if m.sum() >= 3:
            binned[i] = np.nanmedian(flux[m])
    return centers, binned


def plot_detection(clean: CleanLightCurve, result: DetectionResult, output_path: str | Path | None = None):
    candidate = result.best_candidate
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"TIC {clean.tic_id} | Sector {clean.sector} | detection={result.status}", fontsize=12)

    axes[0].plot(clean.time, clean.flux_detrended, ".", ms=2)
    axes[0].axhline(1.0, lw=1)
    axes[0].set_ylabel("detrended flux")

    if candidate is not None:
        in_tr = make_transit_mask(clean.time, candidate.period_days, candidate.epoch_time, candidate.duration_days, width_factor=1.0)
        axes[0].plot(clean.time[in_tr], clean.flux_detrended[in_tr], ".", ms=3)
        axes[0].set_title(
            f"Best: P={candidate.period_days:.5f} d, duration={candidate.duration_days*24:.2f} hr, "
            f"depth={candidate.depth_ppm:.0f} ppm, SNR={candidate.local_snr:.2f}, SDE={candidate.sde}"
        )

        phase, folded_flux = phase_fold(clean.time, clean.flux_detrended, candidate.period_days, candidate.epoch_time)
        axes[1].plot(phase, folded_flux, ".", ms=2, alpha=0.5)
        centers, binned = bin_phase(phase, folded_flux, bins=120)
        axes[1].plot(centers, binned, "o", ms=3)
        half_dur_phase = 0.5 * candidate.duration_days / candidate.period_days
        axes[1].axvspan(-half_dur_phase, half_dur_phase, alpha=0.15)
        axes[1].set_ylabel("phase-folded flux")
        axes[1].set_xlabel("phase")

        zoom = np.abs(phase) < max(0.05, 3 * candidate.duration_days / candidate.period_days)
        axes[2].plot(phase[zoom], folded_flux[zoom], ".", ms=2, alpha=0.6)
        zc, zb = bin_phase(phase[zoom], folded_flux[zoom], bins=60)
        axes[2].plot(zc, zb, "o", ms=3)
        axes[2].axvspan(-half_dur_phase, half_dur_phase, alpha=0.15)
        axes[2].set_ylabel("transit zoom")
        axes[2].set_xlabel("phase")
    else:
        axes[1].text(0.5, 0.5, "No candidate", ha="center", va="center", transform=axes[1].transAxes)
        axes[2].axis("off")

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_vetting_summary(
    clean: CleanLightCurve,
    candidate: CandidateSignal,
    fit,
    vetting,
    classification,
    output_path: str | Path | None = None,
):
    """Create a compact diagnostic panel for Parts 3-5."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 13))
    title = (
        f"TIC {clean.tic_id} | S{clean.sector} | {classification.predicted_class} "
        f"conf={classification.confidence:.2f} | P={fit.period_days:.5f} d | "
        f"depth={fit.depth_ppm:.0f} ppm | SNR={fit.snr:.1f}"
    )
    fig.suptitle(title, fontsize=12)

    # 1. Detrended light curve and transit marks.
    axes[0].plot(clean.time, clean.flux_detrended, ".", ms=2, alpha=0.7)
    in_tr = make_transit_mask(clean.time, fit.period_days, fit.epoch_time, fit.duration_days, width_factor=1.0)
    axes[0].plot(clean.time[in_tr], clean.flux_detrended[in_tr], ".", ms=3)
    axes[0].axhline(1.0, lw=1)
    axes[0].set_ylabel("detrended flux")
    axes[0].set_title("Detected transit windows")

    # 2. Phase fold.
    phase, folded_flux = phase_fold(clean.time, clean.flux_detrended, fit.period_days, fit.epoch_time)
    axes[1].plot(phase, folded_flux, ".", ms=2, alpha=0.5)
    centers, binned = bin_phase(phase, folded_flux, bins=140)
    axes[1].plot(centers, binned, "o", ms=3)
    half = 0.5 * fit.duration_days / fit.period_days
    axes[1].axvspan(-half, half, alpha=0.15)
    axes[1].set_ylabel("folded flux")
    axes[1].set_xlabel("phase")
    axes[1].set_title("Phase-folded transit")

    # 3. Secondary region around phase 0.5.
    phase01 = ((clean.time - fit.epoch_time) / fit.period_days) % 1.0
    order = np.argsort(phase01)
    axes[2].plot(phase01[order], clean.flux_detrended[order], ".", ms=2, alpha=0.35)
    sec_phase = vetting.secondary_phase if np.isfinite(vetting.secondary_phase) else 0.5
    sec_half = 0.5 * fit.duration_days / fit.period_days
    axes[2].axvspan(max(0, sec_phase - sec_half), min(1, sec_phase + sec_half), alpha=0.15)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylabel("flux")
    axes[2].set_xlabel("phase [0,1)")
    axes[2].set_title(f"Secondary search: sigma={vetting.secondary_sigma:.2f}, phase={vetting.secondary_phase}")

    # 4. Evidence summary text.
    axes[3].axis("off")
    evidence = "\n".join([
        f"Class: {classification.predicted_class}",
        f"Confidence: {classification.confidence:.3f}",
        f"Planet score: {classification.planet_score:.3f}",
        f"EB score: {classification.eb_score:.3f}",
        f"Blend score: {classification.blend_score:.3f}",
        f"Odd/even sigma: {vetting.odd_even_sigma:.2f}",
        f"Secondary sigma: {vetting.secondary_sigma:.2f}",
        f"Centroid shift sigma: {vetting.centroid_shift_sigma:.2f}",
        f"CROWDSAP: {vetting.crowdsap}",
        f"V-shape score: {vetting.v_shape_score:.3f}",
        "Evidence: " + ", ".join(classification.evidence[:8]),
        "Warnings: " + ", ".join(classification.warnings[:8]),
    ])
    axes[3].text(0.01, 0.98, evidence, va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig
