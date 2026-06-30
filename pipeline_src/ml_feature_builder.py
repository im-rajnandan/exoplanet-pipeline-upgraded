"""
================================================================
 ML FEATURE BUILDER — Phase-Folded Views + Scalar Features
================================================================
 Reads the preprocessing pipeline's batch CSV + regenerates light
 curves to extract:
   • Global view  (201-pt phase-folded, full orbit)
   • Local view   (61-pt phase-folded, zoomed ±2× transit duration)
   • 25 scalar features for LightGBM Stream B
   • Stellar conditioning features (Teff, logg, R_star)

 Output → tess_pipeline_output/ml_features/
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
import logging

from dataprepro2 import (
    make_synthetic_tess_lc, stage1_ingest, stage2_quality_mask,
    stage3_detrend, stage4_sigma_clip, stage5_tls, stage6_fap,
    stage7_centroid, stage8_vetoes, stage_crowdsap_check,
    determine_verdict, transit_mask, SDE_THRESHOLD
)

# ── Config
ML_FEATURES_DIR = './tess_pipeline_output/ml_features'
os.makedirs(ML_FEATURES_DIR, exist_ok=True)

N_GLOBAL_BINS = 201   # Full orbit phase-folded bins
N_LOCAL_BINS  = 61    # Zoomed transit bins
LOCAL_PHASE_WIDTH = 0.08  # ±0.08 phase around transit center


def phase_fold(time, flux, period, T0):
    """Phase-fold a light curve. Returns sorted phase and flux arrays."""
    phase = ((time - T0 + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def median_bin(phase, flux, n_bins, phase_min=-0.5, phase_max=0.5):
    """
    Bin a phase-folded light curve into n_bins using median.
    Returns binned flux array of shape (n_bins,).
    Empty bins are filled with 1.0 (baseline).
    """
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)
    binned = np.ones(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() >= 3:
            binned[i] = np.nanmedian(flux[mask])
        elif mask.sum() > 0:
            binned[i] = np.nanmean(flux[mask])
    
    return binned


def build_global_view(phase, flux, n_bins=N_GLOBAL_BINS):
    """Full-orbit phase-folded view (201 bins, -0.5 to +0.5)."""
    return median_bin(phase, flux, n_bins, phase_min=-0.5, phase_max=0.5)


def build_local_view(phase, flux, n_bins=N_LOCAL_BINS, width=LOCAL_PHASE_WIDTH):
    """Zoomed transit view (61 bins, centered on phase=0)."""
    return median_bin(phase, flux, n_bins, 
                      phase_min=-width, phase_max=width)


def compute_extra_features(time_c, flux_c, err_c, tls_res, meta, crowdsap_result):
    """
    Compute additional derived scalar features beyond what the
    preprocessing pipeline already provides.
    """
    features = {}
    
    # ── Depth-to-noise ratio
    residual_std = np.nanstd(flux_c)
    fractional_depth = abs(1.0 - tls_res.depth)
    features['depth_to_noise'] = (fractional_depth / residual_std 
                                   if residual_std > 0 else 0.0)
    
    # ── Transit SNR per transit
    n_transits = len(tls_res.transit_times)
    features['n_transits'] = n_transits
    features['snr_per_transit'] = (tls_res.snr / np.sqrt(max(n_transits, 1)))
    
    # ── Period ratio (period / data span) — helps detect long-period false alarms
    data_span = time_c[-1] - time_c[0]
    features['period_to_span'] = tls_res.period / data_span if data_span > 0 else 1.0
    
    # ── In-transit fraction (how much of the data is in transit)
    in_tr = transit_mask(time_c, period=tls_res.period, 
                         duration=tls_res.duration * 1.5, T0=tls_res.T0)
    features['in_transit_fraction'] = in_tr.sum() / len(time_c) if len(time_c) > 0 else 0.0
    
    # ── Transit shape: ingress/egress symmetry via folded model
    # Compare first half vs second half of transit in folded view
    phase, flux_ph = phase_fold(time_c, flux_c, tls_res.period, tls_res.T0)
    dur_phase = (tls_res.duration / tls_res.period) / 2
    
    left_mask  = (phase >= -dur_phase) & (phase < 0)
    right_mask = (phase >= 0) & (phase <= dur_phase)
    
    if left_mask.sum() > 3 and right_mask.sum() > 3:
        left_mean  = np.nanmedian(flux_ph[left_mask])
        right_mean = np.nanmedian(flux_ph[right_mask])
        features['transit_asymmetry'] = abs(left_mean - right_mean) / max(fractional_depth, 1e-8)
    else:
        features['transit_asymmetry'] = 0.0
    
    # ── Out-of-transit variability (RMS of out-of-transit flux)
    out_tr = ~in_tr
    if out_tr.sum() > 10:
        features['oot_rms_ppm'] = np.nanstd(flux_c[out_tr]) * 1e6
    else:
        features['oot_rms_ppm'] = 0.0
    
    # ── Stellar parameters (from TIC or defaults)
    features['stellar_rad'] = meta.get('rad', 1.0)
    features['stellar_teff'] = meta.get('TEFF', 5500.0)  # K
    features['stellar_logg'] = meta.get('LOGG', 4.4)     # cgs
    
    # ── Residual scatter after removing transit model (ppm)
    features['residual_ppm'] = residual_std * 1e6
    
    return features


def extract_features_for_target(tic_id, sector, use_network=False, run_multi_planet_search=False):
    """
    Run the preprocessing pipeline for a single target and extract
    all ML features: global view, local view, scalar feature dict.
    
    Returns:
        global_view (201,), local_view (61,), scalar_dict, status
    """
    try:
        # ── Stage 1: Ingest
        data, meta, source = stage1_ingest(tic_id, sector, use_network)
        
        # ── Crowding check
        crowdsap_result = stage_crowdsap_check(meta)
        
        # ── Stage 2: Quality mask
        (time_q, flux_q, err_q, cc_q, cr_q, n_bad, n_good) = stage2_quality_mask(data)
        
        # ── Stage 3: Detrend (Pass 1)
        flux_src = meta.get('flux_source', 'PDCSAP')
        flat1, trend1, res_ppm1, win1, meth1 = stage3_detrend(time_q, flux_q, flux_src)
        
        # ── Stage 4: Sigma clip (Pass 1)
        (time_c, flux_c, err_c, ccol_c, crow_c, n_removed) = stage4_sigma_clip(
            time_q, flat1, err_q, cc_q, cr_q, sigma_upper=5.0)
        
        if len(time_c) < 500:
            return None, None, None, 'insufficient_data'
        
        # ── Stage 5: TLS (Pass 1)
        tls_res, pmin, pmax = stage5_tls(time_c, flux_c, err_c)
        
        if tls_res.SDE < 3.0:  # Very low SDE → still extract features for autoencoder
            # Phase-fold with best period anyway (for autoencoder training)
            phase, flux_ph = phase_fold(time_c, flux_c, tls_res.period, tls_res.T0)
            global_view = build_global_view(phase, flux_ph)
            local_view  = build_local_view(phase, flux_ph)
            
            scalar = {
                'tic_id': tic_id, 'sector': sector,
                'SDE': tls_res.SDE, 'SNR': tls_res.snr,
                'FAP': 1.0, 'period': tls_res.period,
                'duration_hrs': tls_res.duration * 24,
                'depth_ppm_obs': abs(1.0 - tls_res.depth) * 1e6,
                'depth_ppm_corr': abs(1.0 - tls_res.depth) * 1e6,
                'Rp_earth': 0.0,
                'crowdsap': crowdsap_result['crowdsap'],
                'flfrcsap': crowdsap_result['flfrcsap'],
                'crowdsap_flag': int(crowdsap_result['crowdsap_flag']),
                'harmonic_flag': 0, 'has_multiple_planets': 0,
                'centroid_shift_pix': 0.0, 'centroid_p_col': 1.0,
                'centroid_blend_flag': 0,
                'odd_even_sigma': 0.0, 'odd_even_flag': 0,
                'secondary_sigma': 0.0, 'secondary_flag': 0,
                'verdict_score': 0,
                'is_significant': 0,
            }
            extras = compute_extra_features(time_c, flux_c, err_c, tls_res, meta, crowdsap_result)
            scalar.update(extras)
            return global_view, local_view, scalar, 'low_sde'
        
        # ── Pass 2: Iterative detrending
        t_mask = transit_mask(time_q, period=tls_res.period, 
                               duration=tls_res.duration * 1.5, T0=tls_res.T0)
        adaptive_window = max(3.0 * tls_res.duration, 0.5)
        flat, trend, res_ppm, win, meth = stage3_detrend(
            time_q, flux_q, flux_src, transit_mask=t_mask, window_override=adaptive_window)
        
        (time_c, flux_c, err_c, ccol_c, crow_c, n_removed) = stage4_sigma_clip(
            time_q, flat, err_q, cc_q, cr_q, sigma_upper=5.0)
        
        tls_res, pmin, pmax = stage5_tls(time_c, flux_c, err_c)
        
        # ── Phase fold
        phase, flux_ph = phase_fold(time_c, flux_c, tls_res.period, tls_res.T0)
        global_view = build_global_view(phase, flux_ph)
        local_view  = build_local_view(phase, flux_ph)
        
        # ── Full veto pipeline for scalars
        FAP, null_sdes = stage6_fap(tls_res)
        centroid_result = stage7_centroid(
            time_c, ccol_c, crow_c,
            tls_res.period, tls_res.T0, tls_res.duration)
        oe_result = stage8_vetoes(time_c, flux_c, tls_res)
        
        fractional_depth = abs(1.0 - tls_res.depth)
        corrected_depth = (fractional_depth / crowdsap_result['crowdsap_correction']
                           if crowdsap_result['crowdsap_correction'] > 0 else fractional_depth)
        rp = np.sqrt(abs(corrected_depth)) * meta.get('rad', 1.0) * 109.076
        
        # Multi-planet search
        has_multiple = 0
        if run_multi_planet_search:
            t_mask2 = transit_mask(time_c, period=tls_res.period, 
                                   duration=tls_res.duration * 1.5, T0=tls_res.T0)
            time_m = time_c[~t_mask2]
            flux_m = flux_c[~t_mask2]
            err_m  = err_c[~t_mask2]
            if len(time_m) > 500:
                try:
                    tls_multi, _, _ = stage5_tls(time_m, flux_m, err_m)
                    has_multiple = int(tls_multi.SDE >= SDE_THRESHOLD)
                except Exception:
                    pass
        
        # ── Build scalar dict
        scalar = {
            'tic_id': tic_id, 'sector': sector,
            'SDE': tls_res.SDE, 'SNR': tls_res.snr,
            'FAP': FAP, 'period': tls_res.period,
            'duration_hrs': tls_res.duration * 24,
            'depth_ppm_obs': fractional_depth * 1e6,
            'depth_ppm_corr': corrected_depth * 1e6,
            'Rp_earth': rp,
            'crowdsap': crowdsap_result['crowdsap'],
            'flfrcsap': crowdsap_result['flfrcsap'],
            'crowdsap_flag': int(crowdsap_result['crowdsap_flag']),
            'harmonic_flag': int(tls_res.harmonic_flag),
            'has_multiple_planets': has_multiple,
            'centroid_shift_pix': centroid_result['shift_pix'],
            'centroid_p_col': centroid_result['p_col'],
            'centroid_blend_flag': int(centroid_result['blend_suspect']),
            'odd_even_sigma': oe_result['odd_even_sigma'],
            'odd_even_flag': int(oe_result['odd_even_flag']),
            'secondary_sigma': oe_result['secondary_sigma'],
            'secondary_flag': int(oe_result['secondary_flag']),
            'verdict_score': 0,  # Will be set by determine_verdict
            'is_significant': 1,
        }
        
        # ── Extra derived features
        extras = compute_extra_features(time_c, flux_c, err_c, tls_res, meta, crowdsap_result)
        scalar.update(extras)
        
        return global_view, local_view, scalar, 'ok'
    
    except Exception as e:
        logging.error(f"Feature extraction failed for TIC {tic_id}: {e}")
        return None, None, None, 'error'


def build_features_from_batch(batch_csv, sector=1, use_network=False):
    """
    Read the batch results CSV and re-process each target to extract
    ML features. Can also work with a list of TIC IDs directly.
    
    Saves outputs to ML_FEATURES_DIR.
    """
    df = pd.read_csv(batch_csv)
    tic_ids = df['tic_id'].unique()
    
    print(f"\n[ML FEATURES] Extracting features for {len(tic_ids)} targets...")
    
    global_views = []
    local_views  = []
    scalars      = []
    metadata     = []
    
    for i, tic_id in enumerate(tic_ids):
        if (i + 1) % max(1, len(tic_ids) // 10) == 0:
            print(f"  {i+1}/{len(tic_ids)}")
        
        gv, lv, sc, status = extract_features_for_target(
            int(tic_id), sector, use_network, run_multi_planet_search=True)
        
        if gv is not None:
            global_views.append(gv)
            local_views.append(lv)
            scalars.append(sc)
            
            # Get original verdict from batch CSV
            row = df[df['tic_id'] == tic_id].iloc[0]
            metadata.append({
                'tic_id': tic_id,
                'sector': sector,
                'original_verdict': row.get('verdict', 'UNKNOWN'),
                'extraction_status': status,
            })
    
    if len(global_views) == 0:
        print("[ML FEATURES] No features extracted!")
        return
    
    # ── Save arrays
    global_arr = np.array(global_views, dtype=np.float32)
    local_arr  = np.array(local_views, dtype=np.float32)
    
    np.save(os.path.join(ML_FEATURES_DIR, 'global_views.npy'), global_arr)
    np.save(os.path.join(ML_FEATURES_DIR, 'local_views.npy'), local_arr)
    
    scalar_df = pd.DataFrame(scalars)
    scalar_df.to_csv(os.path.join(ML_FEATURES_DIR, 'scalar_features.csv'), index=False)
    
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(ML_FEATURES_DIR, 'metadata.csv'), index=False)
    
    print(f"\n[ML FEATURES] Saved:")
    print(f"  global_views.npy  : {global_arr.shape}")
    print(f"  local_views.npy   : {local_arr.shape}")
    print(f"  scalar_features.csv : {len(scalar_df)} rows × {len(scalar_df.columns)} cols")
    print(f"  metadata.csv      : {len(meta_df)} rows")
    print(f"  Output dir: {ML_FEATURES_DIR}")
    
    return global_arr, local_arr, scalar_df, meta_df


def build_features_from_tic_list(tic_ids, sector=1, use_network=False):
    """
    Extract features directly from a list of TIC IDs (without needing
    a pre-existing batch CSV).
    """
    print(f"\n[ML FEATURES] Extracting features for {len(tic_ids)} targets...")
    
    global_views = []
    local_views  = []
    scalars      = []
    
    for i, tic_id in enumerate(tic_ids):
        if (i + 1) % max(1, len(tic_ids) // 10) == 0:
            print(f"  {i+1}/{len(tic_ids)}")
        
        gv, lv, sc, status = extract_features_for_target(
            int(tic_id), sector, use_network)
        
        if gv is not None:
            global_views.append(gv)
            local_views.append(lv)
            scalars.append(sc)
    
    if len(global_views) == 0:
        return None, None, None
    
    return (np.array(global_views, dtype=np.float32),
            np.array(local_views, dtype=np.float32),
            pd.DataFrame(scalars))


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    batch_csv = './tess_pipeline_output/batch_sector1_results.csv'
    
    if os.path.exists(batch_csv):
        build_features_from_batch(batch_csv, sector=1, use_network=False)
    else:
        # Demo: build features from a few synthetic targets
        print("[ML FEATURES] No batch CSV found. Running demo with synthetic targets.")
        demo_tics = [261136679, 350622204, 100100827, 441462736, 307210830]
        gv, lv, sc = build_features_from_tic_list(demo_tics, sector=1)
        if gv is not None:
            print(f"  Global views: {gv.shape}")
            print(f"  Local views:  {lv.shape}")
            print(f"  Scalars:      {sc.shape}")
