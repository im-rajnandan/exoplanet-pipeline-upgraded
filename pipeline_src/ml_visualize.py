"""
================================================================
 ML VISUALIZATION — Diagnostic Dashboard + Sector Summary
================================================================
 Generates:
   1. Per-candidate diagnostic figure (6 panels)
   2. Sector summary dashboard (confusion matrix, ROC, P-R diagram)
   3. Training history plots
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Plot style
DARK = '#0d1117'; PANEL = '#161b22'; BORD = '#30363d'
TPRI = '#c9d1d9'; TSEC = '#8b949e'
BLUE = '#58a6ff'; GREEN = '#3fb950'; RED = '#f78166'
TEAL = '#2b8a3e'; ORANGE = '#d97706'; AMB = '#e3b341'; PUR = '#d2a8ff'

CLASS_COLORS = {
    'PLANET': GREEN, 'EB': RED, 'BLEND': ORANGE, 'OTHER': TSEC,
    0: GREEN, 1: RED, 2: ORANGE, 3: TSEC
}
CLASS_NAMES = ['PLANET', 'EB', 'BLEND', 'OTHER']

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORD, 'axes.labelcolor': TPRI,
    'xtick.color': TSEC, 'ytick.color': TSEC,
    'text.color': TPRI, 'grid.color': BORD,
    'grid.alpha': 0.3, 'font.family': 'monospace',
    'font.size': 10, 'legend.facecolor': PANEL,
    'legend.edgecolor': BORD,
})

OUT_DIR = './tess_pipeline_output/ml_plots'


# ═══════════════════════════════════════════════════════════════
# 1. PER-CANDIDATE DIAGNOSTIC FIGURE
# ═══════════════════════════════════════════════════════════════

def plot_candidate_diagnostic(tic_id, sector, time, flux, phase, flux_phase,
                                model_phase=None, model_flux=None,
                                probabilities=None, uncertainty=None,
                                params=None, counterfactual=None,
                                needs_review=False, mc_probs=None,
                                save=True):
    """
    6-panel diagnostic figure for a single candidate.
    
    Panels:
      1. Raw light curve with transit windows
      2. Phase-folded with best-fit model
      3. Classification probability bar chart
      4. MC Dropout uncertainty violin
      5. Counterfactual feature importance
      6. Parameter summary box
    """
    fig = plt.figure(figsize=(22, 18), facecolor=DARK)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)
    
    pred_class = CLASS_NAMES[probabilities.argmax()] if probabilities is not None else '?'
    pred_prob = probabilities.max() if probabilities is not None else 0
    review_str = "  ⚠️ NEEDS REVIEW" if needs_review else ""
    
    fig.suptitle(
        f"TIC {tic_id}  Sector {sector}  ·  {pred_class} ({pred_prob:.1%}){review_str}",
        fontsize=14, y=0.97, color=CLASS_COLORS.get(pred_class, TPRI)
    )
    
    # ── Panel 1: Raw Light Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, flux, color=BLUE, lw=0.4, alpha=0.7, rasterized=True)
    if params and 'period' in params and 't0' in params:
        period = params['period']
        t0 = params['t0']
        for i in range(int((time[-1] - t0) / period) + 2):
            tc = t0 + i * period
            if time[0] <= tc <= time[-1]:
                dur = params.get('duration_hrs', 2.0) / 24
                ax1.axvspan(tc - dur, tc + dur, alpha=0.15, color=GREEN, lw=0)
    ax1.set_title('① Cleaned Light Curve (green = transit windows)', pad=6)
    ax1.set(xlabel='BTJD [days]', ylabel='Relative Flux')
    ax1.grid(True, alpha=0.2)
    
    # ── Panel 2: Phase-Folded with Model
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(phase, flux_phase, s=2, color=BLUE, alpha=0.3, rasterized=True)
    if model_phase is not None and model_flux is not None:
        ax2.plot(model_phase, model_flux, color=GREEN, lw=2.5, zorder=5,
                 label='Batman model')
    ax2.set_xlim(-0.08, 0.08)
    period_str = f"P={params['period']:.5f}d" if params else ""
    ax2.set_title(f'② Phase-Folded Transit  {period_str}', pad=6)
    ax2.set(xlabel='Phase', ylabel='Flux')
    if model_phase is not None:
        ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    # ── Panel 3: Classification Probabilities
    ax3 = fig.add_subplot(gs[1, 1])
    if probabilities is not None:
        bars = ax3.barh(CLASS_NAMES, probabilities, 
                        color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                        edgecolor=BORD, linewidth=0.5)
        ax3.set_xlim(0, 1.05)
        for bar, prob in zip(bars, probabilities):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}', va='center', color=TPRI, fontsize=11)
        if needs_review:
            ax3.text(0.5, -0.15, '⚠️ FLAGGED FOR MANUAL REVIEW',
                    transform=ax3.transAxes, ha='center', fontsize=11,
                    color=AMB, fontweight='bold')
    ax3.set_title('③ Classification Probabilities (calibrated)', pad=6)
    ax3.grid(True, alpha=0.2, axis='x')
    
    # ── Panel 4: MC Dropout Uncertainty
    ax4 = fig.add_subplot(gs[2, 0])
    if mc_probs is not None and len(mc_probs.shape) == 2:
        # mc_probs shape: (mc_samples, n_classes) for this single target
        parts = ax4.violinplot(mc_probs, positions=range(4), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(CLASS_COLORS[i])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color(TPRI)
        parts['cmedians'].set_color(AMB)
        ax4.set_xticks(range(4))
        ax4.set_xticklabels(CLASS_NAMES)
        unc_val = uncertainty if uncertainty is not None else 0
        ax4.set_title(f'④ MC Dropout Uncertainty (σ={unc_val:.4f})', pad=6)
    else:
        ax4.text(0.5, 0.5, 'MC Dropout\nNot Available', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, color=TSEC)
        ax4.set_title('④ MC Dropout Uncertainty', pad=6)
    ax4.set(ylabel='Probability')
    ax4.grid(True, alpha=0.2)
    
    # ── Panel 5: Counterfactual Importance
    ax5 = fig.add_subplot(gs[2, 1])
    if counterfactual:
        groups = list(counterfactual.keys())
        impacts = [counterfactual[g].get('n_changed', 0) for g in groups]
        colors = [RED if imp > 0 else TSEC for imp in impacts]
        
        # For single target, show if this specific target's prediction changed
        labels_display = []
        for g in groups:
            changed_idx = counterfactual[g].get('changed_indices', [])
            if 0 in changed_idx:  # Index 0 = this target (assuming single)
                labels_display.append(f"{g} ★")
            else:
                labels_display.append(g)
        
        ax5.barh(labels_display, impacts, color=colors, edgecolor=BORD)
        ax5.set_title('⑤ Counterfactual Importance (N predictions changed)', pad=6)
        ax5.set(xlabel='# Predictions Changed if Removed')
    else:
        ax5.text(0.5, 0.5, 'Counterfactual\nNot Available', transform=ax5.transAxes,
                ha='center', va='center', fontsize=14, color=TSEC)
        ax5.set_title('⑤ Feature Importance', pad=6)
    ax5.grid(True, alpha=0.2, axis='x')
    
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, f'TIC{tic_id}_S{sector}_ml_diagnostic.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close()
        print(f"  Saved: {out}")
        return out
    
    return fig


# ═══════════════════════════════════════════════════════════════
# 2. PARAMETER SUMMARY CARD
# ═══════════════════════════════════════════════════════════════

def plot_parameter_card(tic_id, sector, params, probabilities=None, save=True):
    """
    Compact parameter summary card for a planet candidate.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK)
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    pred_class = 'PLANET'
    if probabilities is not None:
        pred_class = CLASS_NAMES[probabilities.argmax()]
    
    title = f"TIC {tic_id}  Sector {sector}"
    ax.text(0.5, 0.95, title, transform=ax.transAxes, ha='center', 
            fontsize=16, fontweight='bold', color=CLASS_COLORS.get(pred_class, GREEN))
    
    lines = [
        f"Period       :  {params['period']:.6f} ± {params['period_err']:.6f} days",
        f"Transit Depth:  {params['depth_ppm']:.1f} ± {params['depth_ppm_err']:.1f} ppm",
        f"Duration     :  {params['duration_hrs']:.2f} ± {params['duration_hrs_err']:.2f} hrs",
        f"Planet Radius:  {params['Rp_earth']:.2f} ± {params['Rp_earth_err']:.2f} R⊕",
        f"Rp/R*        :  {params['rp_rs']:.6f} ± {params['rp_rs_err']:.6f}",
        f"Inclination  :  {params['inc']:.2f} ± {params['inc_err']:.2f}°",
        f"a/R*         :  {params['a_rs']:.2f} ± {params['a_rs_err']:.2f}",
        f"Impact (b)   :  {params['b']:.4f} ± {params['b_err']:.4f}",
        f"a (AU)       :  {params['a_AU']:.5f} ± {params['a_AU_err']:.5f}",
        f"Fit Method   :  {params['fit_method']}",
        f"χ²_red       :  {params['chi2_red']:.4f}",
    ]
    
    text = "\n".join(lines)
    ax.text(0.1, 0.8, text, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', color=TPRI, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=DARK, edgecolor=BORD))
    
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, f'TIC{tic_id}_S{sector}_params.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close()
        return out
    return fig


# ═══════════════════════════════════════════════════════════════
# 3. SECTOR SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════

def plot_sector_dashboard(predictions, true_labels=None, probabilities=None,
                           params_list=None, save=True):
    """
    Sector-wide summary dashboard:
      - Classification distribution
      - Confusion matrix (if labels available)
      - ROC curves per class
      - Period-Radius diagram
    """
    n_panels = 4 if true_labels is not None else 2
    fig = plt.figure(figsize=(22, 12 if n_panels == 4 else 6), facecolor=DARK)
    
    if n_panels == 4:
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Sector Classification Summary', fontsize=16, y=0.97, color=TPRI)
    
    # ── Panel 1: Classification Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pred_counts = Counter(predictions)
    labels = CLASS_NAMES
    counts = [pred_counts.get(i, 0) for i in range(4)]
    colors = [CLASS_COLORS[i] for i in range(4)]
    
    bars = ax1.bar(labels, counts, color=colors, edgecolor=BORD, linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', color=TPRI, fontsize=12)
    ax1.set_title('Classification Distribution', pad=6)
    ax1.set(ylabel='Count')
    ax1.grid(True, alpha=0.2, axis='y')
    
    # ── Panel 2: Confusion Matrix (if labels available)
    if true_labels is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        cm = confusion_matrix(true_labels, predictions, labels=range(4))
        im = ax2.imshow(cm, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax2.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax2.set_title('Confusion Matrix', pad=6)
        ax2.set(xlabel='Predicted', ylabel='True')
        
        # Annotate cells
        for i in range(4):
            for j in range(4):
                color = DARK if cm[i, j] > cm.max() / 2 else TPRI
                ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # ── Panel 3: ROC Curves
    if true_labels is not None and probabilities is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        y_true_bin = label_binarize(true_labels, classes=range(4))
        
        for i in range(4):
            if y_true_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                ax3.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                        label=f'{CLASS_NAMES[i]} (AUC={roc_auc:.3f})')
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax3.set_title('ROC Curves per Class', pad=6)
        ax3.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.2)
    
    # ── Panel 4: Period-Radius Diagram
    panel_idx = gs[1, 1] if n_panels == 4 else gs[0, 1]
    ax4 = fig.add_subplot(panel_idx)
    
    if params_list:
        periods = [p['period'] for p in params_list if p.get('Rp_earth', 0) > 0]
        radii = [p['Rp_earth'] for p in params_list if p.get('Rp_earth', 0) > 0]
        radii_err = [p.get('Rp_earth_err', 0) for p in params_list if p.get('Rp_earth', 0) > 0]
        
        if periods:
            ax4.errorbar(periods, radii, yerr=radii_err, fmt='o',
                        color=GREEN, markersize=8, capsize=3, ecolor=TSEC,
                        markeredgecolor=DARK, markeredgewidth=0.5)
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            
            # Reference lines
            ax4.axhline(1.0, color=BLUE, ls='--', alpha=0.3, label='1 R⊕')
            ax4.axhline(3.5, color=TEAL, ls='--', alpha=0.3, label='3.5 R⊕ (sub-Neptune)')
            ax4.axhline(11.2, color=ORANGE, ls='--', alpha=0.3, label='11.2 R⊕ (Jupiter)')
    
    ax4.set_title('Period-Radius Diagram', pad=6)
    ax4.set(xlabel='Orbital Period (days)', ylabel='Planet Radius (R⊕)')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.2)
    
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, 'sector_summary_dashboard.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close()
        print(f"  Saved: {out}")
        return out
    return fig


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════

def plot_training_history(train_history, save=True):
    """Plot training loss, validation loss, and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
    
    epochs = range(1, len(train_history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, train_history['train_loss'], color=BLUE, lw=2, label='Train')
    ax1.plot(epochs, train_history['val_loss'], color=RED, lw=2, label='Val')
    ax1.set_title('Training & Validation Loss', pad=6)
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Accuracy
    ax2.plot(epochs, train_history['val_acc'], color=GREEN, lw=2, label='Val Accuracy')
    best_epoch = np.argmax(train_history['val_acc']) + 1
    best_acc = max(train_history['val_acc'])
    ax2.axvline(best_epoch, color=AMB, ls='--', alpha=0.5, 
                label=f'Best: {best_acc:.4f} @ epoch {best_epoch}')
    ax2.set_title('Validation Accuracy', pad=6)
    ax2.set(xlabel='Epoch', ylabel='Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, 'training_history.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close()
        print(f"  Saved: {out}")
        return out
    return fig


# ═══════════════════════════════════════════════════════════════
# 5. CALIBRATION PLOT
# ═══════════════════════════════════════════════════════════════

def plot_calibration_curve(probabilities, true_labels, n_bins=10, save=True):
    """
    Reliability diagram: predicted probability vs observed frequency.
    A well-calibrated model follows the diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=DARK)
    
    for c in range(4):
        if (true_labels == c).sum() == 0:
            continue
        
        probs_c = probabilities[:, c]
        true_c = (true_labels == c).astype(float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_pred = []
        bin_true = []
        
        for j in range(n_bins):
            mask = (probs_c >= bin_edges[j]) & (probs_c < bin_edges[j+1])
            if mask.sum() > 0:
                bin_pred.append(probs_c[mask].mean())
                bin_true.append(true_c[mask].mean())
        
        if bin_pred:
            ax.plot(bin_pred, bin_true, 'o-', color=CLASS_COLORS[c],
                   label=CLASS_NAMES[c], markersize=6, lw=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.set_title('Calibration Curve (Reliability Diagram)', pad=6)
    ax.set(xlabel='Mean Predicted Probability', ylabel='Observed Frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, 'calibration_curve.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
        plt.close()
        print(f"  Saved: {out}")
        return out
    return fig


# ═══════════════════════════════════════════════════════════════
# 6. CONVENIENCE: Generate all visualizations
# ═══════════════════════════════════════════════════════════════

def generate_all_visualizations(ml_results, params_list=None, 
                                  true_labels=None, train_history=None):
    """
    Generate all visualization outputs.
    
    Args:
        ml_results: dict from DualStreamTrainer.predict()
        params_list: list of param dicts from parameter estimation
        true_labels: ground truth labels (if available)
        train_history: training history dict
    """
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Training history
    if train_history:
        plot_training_history(train_history)
    
    # Sector dashboard
    plot_sector_dashboard(
        ml_results['predictions'],
        true_labels=true_labels,
        probabilities=ml_results.get('probabilities'),
        params_list=params_list,
    )
    
    # Calibration curve
    if true_labels is not None and 'probabilities' in ml_results:
        plot_calibration_curve(ml_results['probabilities'], true_labels)
    
    print(f"\n[VIZ] All plots saved to {OUT_DIR}/")


if __name__ == '__main__':
    print("[VIZ] This module is meant to be imported. Run ml_pipeline.py instead.")
    print(f"  Output directory: {OUT_DIR}")
