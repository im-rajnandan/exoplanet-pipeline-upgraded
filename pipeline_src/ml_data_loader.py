"""
================================================================
 ML DATA LOADER — Training Data Assembly + Splits
================================================================
 Assembles labeled training data from three sources:
   1. Curated dataset (known planets, EBs, false positives)
   2. Kepler DR25 TCE dispositions (supplement)
   3. Synthetic injection-recovery (augmentation)
 
 Produces stratified train/val/test splits with class balancing.
 Output → tess_pipeline_output/ml_splits/
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
import logging
from collections import Counter

from dataprepro2 import make_synthetic_tess_lc, transit_mask
from ml_feature_builder import (
    extract_features_for_target, phase_fold,
    build_global_view, build_local_view, compute_extra_features,
    ML_FEATURES_DIR, N_GLOBAL_BINS, N_LOCAL_BINS
)

# ── Config
ML_SPLITS_DIR = './tess_pipeline_output/ml_splits'
os.makedirs(ML_SPLITS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Class label mapping
LABEL_MAP = {
    'PLANET': 0,
    'EB': 1,
    'BLEND': 2,
    'OTHER': 3,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


# ═══════════════════════════════════════════════════════════════
# SOURCE 1: Curated Dataset
# ═══════════════════════════════════════════════════════════════

def load_curated_dataset(curated_csv_path=None):
    """
    Load the provided curated dataset of labeled exoplanets,
    false positives, and eclipsing binaries.
    
    Expected CSV columns: tic_id, sector, label
    Where label ∈ {PLANET, EB, BLEND, OTHER}
    
    If no file provided, returns empty DataFrame.
    """
    if curated_csv_path and os.path.exists(curated_csv_path):
        df = pd.read_csv(curated_csv_path)
        print(f"[DATA LOADER] Loaded curated dataset: {len(df)} entries")
        
        # Standardize label names
        label_aliases = {
            'planet': 'PLANET', 'confirmed': 'PLANET', 'candidate': 'PLANET',
            'PC': 'PLANET', 'KP': 'PLANET',
            'eb': 'EB', 'eclipsing_binary': 'EB', 'eclipsing binary': 'EB',
            'FP': 'OTHER', 'false_positive': 'OTHER', 'false positive': 'OTHER',
            'blend': 'BLEND', 'NTP': 'BLEND',
            'noise': 'OTHER', 'variable': 'OTHER', 'other': 'OTHER',
            'FA': 'OTHER',
        }
        if 'label' in df.columns:
            df['label'] = df['label'].str.strip().map(
                lambda x: label_aliases.get(x.lower(), x.upper()))
        
        print(f"  Label distribution: {dict(Counter(df['label']))}")
        return df
    else:
        print("[DATA LOADER] No curated dataset provided. Using synthetic data only.")
        return pd.DataFrame(columns=['tic_id', 'sector', 'label'])


# ═══════════════════════════════════════════════════════════════
# SOURCE 2: Kepler DR25 Dispositions (via NASA Exoplanet Archive)
# ═══════════════════════════════════════════════════════════════

def load_kepler_dr25_dispositions(max_samples=2000):
    """
    Download Kepler DR25 TCE dispositions from NASA Exoplanet Archive.
    Maps Kepler dispositions to our 4-class schema.
    
    This is optional supplementary data — phase-folded signal shapes
    are instrument-agnostic enough to be useful training examples.
    
    Returns DataFrame with columns matching our schema.
    """
    cache_path = os.path.join(ML_SPLITS_DIR, 'kepler_dr25_cache.csv')
    
    if os.path.exists(cache_path):
        print(f"[DATA LOADER] Loading cached Kepler DR25 from {cache_path}")
        return pd.read_csv(cache_path)
    
    try:
        # TAP query to NASA Exoplanet Archive
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        
        print("[DATA LOADER] Downloading Kepler DR25 dispositions from NASA Exoplanet Archive...")
        table = NasaExoplanetArchive.query_criteria(
            table='q1_q17_dr25_tce',
            select='kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,'
                   'tce_depth,tce_prad,tce_pdisposition,tce_maxsesd',
            where=f"tce_pdisposition is not null",
            order='kepid',
        )
        df = table.to_pandas()
        
        # Map dispositions
        disp_map = {
            'PC': 'PLANET',  # Planet Candidate
            'FP': 'EB',      # False Positive (majority are EBs)
            'FA': 'OTHER',   # False Alarm
        }
        df['label'] = df['tce_pdisposition'].map(disp_map).fillna('OTHER')
        
        # Refine FP → BLEND vs EB using secondary eclipse significance
        # High tce_maxsesd indicates eclipsing binary
        blend_mask = (df['label'] == 'EB') & (df['tce_maxsesd'] < 3.0)
        df.loc[blend_mask, 'label'] = 'BLEND'
        
        # Subsample
        if len(df) > max_samples:
            df = df.groupby('label').apply(
                lambda x: x.sample(min(len(x), max_samples // 4), 
                                   random_state=RANDOM_STATE)
            ).reset_index(drop=True)
        
        df.to_csv(cache_path, index=False)
        print(f"  Loaded {len(df)} Kepler DR25 dispositions")
        print(f"  Distribution: {dict(Counter(df['label']))}")
        return df
        
    except Exception as e:
        print(f"[DATA LOADER] Could not load Kepler DR25: {e}")
        print("  Continuing without Kepler supplementary data.")
        return pd.DataFrame(columns=['label'])


# ═══════════════════════════════════════════════════════════════
# SOURCE 3: Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════

import multiprocessing as mp
from functools import partial

def _generate_synthetic_wrapper(args):
    try:
        return _generate_one_synthetic(*args)
    except Exception:
        return None, None, None

def generate_synthetic_training_data(n_per_class=500, sector=1):
    """
    Generate synthetic light curves for each class using multiprocessing.
    """
    print(f"\n[DATA LOADER] Generating synthetic training data ({n_per_class} per class)...")
    
    global_views = []
    local_views  = []
    scalars      = []
    labels       = []
    
    pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))
    
    # ── CLASS: PLANET
    print("  Generating PLANET class...")
    tasks = [(1000000 + i, sector, np.random.uniform(0.5, 12.0), np.random.uniform(100e-6, 5000e-6), np.random.uniform(200, 1000), np.random.uniform(0.85, 1.0), False, 'PLANET') for i in range(n_per_class)]
    for res in pool.imap_unordered(_generate_synthetic_wrapper, tasks):
        if res[0] is not None:
            global_views.append(res[0]); local_views.append(res[1]); scalars.append(res[2]); labels.append(LABEL_MAP['PLANET'])
            
    # ── CLASS: EB
    print("  Generating EB class...")
    tasks = [(2000000 + i, sector, np.random.uniform(0.5, 10.0), np.random.uniform(5000e-6, 0.15), np.random.uniform(200, 800), np.random.uniform(0.85, 1.0), False, 'EB', True, np.random.uniform(0.2, 0.8)) for i in range(n_per_class)]
    for res in pool.imap_unordered(_generate_synthetic_wrapper, tasks):
        if res[0] is not None:
            global_views.append(res[0]); local_views.append(res[1]); scalars.append(res[2]); labels.append(LABEL_MAP['EB'])
            
    # ── CLASS: BLEND
    print("  Generating BLEND class...")
                noise_ppm=np.random.uniform(300, 900),
                crowdsap=np.random.uniform(0.5, 0.85),  # Low CROWDSAP = blended
                inject_blend=True,
                label='BLEND'
            )
            if gv is not None:
                global_views.append(gv)
                local_views.append(lv)
                scalars.append(sc)
                labels.append(LABEL_MAP['BLEND'])
        except Exception:
            continue
    print(f"    Generated {labels.count(LABEL_MAP['BLEND'])} BLEND samples")
    
    # ── CLASS: OTHER (noise, variability, systematics)
    print("  Generating OTHER class...")
    for i in range(n_per_class):
        try:
            gv, lv, sc = _generate_one_synthetic(
                tic_seed=4000000 + i, sector=sector,
                period=np.random.uniform(0.5, 12.0),
                depth=np.random.uniform(0, 50e-6),  # Very shallow or zero depth
                noise_ppm=np.random.uniform(500, 2000),  # High noise
                crowdsap=np.random.uniform(0.7, 1.0),
                inject_blend=False,
                label='OTHER',
                add_variability=True
            )
            if gv is not None:
                global_views.append(gv)
                local_views.append(lv)
                scalars.append(sc)
                labels.append(LABEL_MAP['OTHER'])
        except Exception:
            continue
    print(f"    Generated {labels.count(LABEL_MAP['OTHER'])} OTHER samples")
    
    return (np.array(global_views, dtype=np.float32),
            np.array(local_views, dtype=np.float32),
            pd.DataFrame(scalars),
            np.array(labels, dtype=np.int64))


def _generate_one_synthetic(tic_seed, sector, period, depth, noise_ppm,
                             crowdsap, inject_blend, label,
                             inject_secondary=False, secondary_depth_ratio=0.0,
                             add_variability=False):
    """
    Generate one synthetic light curve and extract its ML features.
    Returns (global_view, local_view, scalar_dict) or (None, None, None).
    """
    np.random.seed(tic_seed)
    
    duration_hrs = max(0.5, min(period * 0.06 * 24, 8.0))  # Rough scaling
    
    data, meta = make_synthetic_tess_lc(
        tic_id=tic_seed, sector=sector,
        period=period, depth=depth,
        duration_hrs=duration_hrs,
        noise_ppm=noise_ppm,
        crowdsap=crowdsap,
        inject_blend=inject_blend
    )
    
    # Inject secondary eclipse for EBs
    if inject_secondary and secondary_depth_ratio > 0:
        t = data['time']
        baseline = 185000.0
        sec_depth = depth * secondary_depth_ratio
        T0_first = t[0] + period * 1.5
        dur_d = duration_hrs / 24.0
        
        for i in range(int((t[-1] - T0_first) / period) + 1):
            tc = T0_first + i * period + period * 0.5  # Phase 0.5
            in_sec = np.abs(t - tc) < dur_d / 2
            data['pdcsap_flux'][in_sec] -= sec_depth * baseline
            data['sap_flux'][in_sec] -= sec_depth * baseline
    
    # Add extra stellar variability for OTHER class
    if add_variability:
        t = data['time']
        baseline = 185000.0
        # Strong quasi-periodic variability
        var_period = np.random.uniform(1.0, 5.0)
        var_amp = np.random.uniform(0.002, 0.01) * baseline
        variability = var_amp * np.sin(2 * np.pi * t / var_period + np.random.uniform(0, 6))
        variability += var_amp * 0.3 * np.sin(4 * np.pi * t / var_period)
        data['pdcsap_flux'] += variability
        data['sap_flux'] += variability
    
    meta['flux_source'] = 'PDCSAP'
    meta['rad'] = np.random.uniform(0.5, 2.5)  # Random stellar radius
    meta['TEFF'] = np.random.uniform(3500, 7000)
    meta['LOGG'] = np.random.uniform(3.5, 5.0)
    
    # Run through feature extraction (simplified — no network)
    from dataprepro2 import (stage2_quality_mask, stage3_detrend,
                             stage4_sigma_clip, stage5_tls, stage6_fap,
                             stage7_centroid, stage8_vetoes, stage_crowdsap_check)
    
    crowdsap_result = stage_crowdsap_check(meta)
    (time_q, flux_q, err_q, cc_q, cr_q, _, _) = stage2_quality_mask(data)
    flat, trend, _, _, _ = stage3_detrend(time_q, flux_q, 'PDCSAP')
    (time_c, flux_c, err_c, ccol_c, crow_c, _) = stage4_sigma_clip(
        time_q, flat, err_q, cc_q, cr_q, sigma_upper=5.0)
    
    if len(time_c) < 500:
        return None, None, None
    
    tls_res, _, _ = stage5_tls(time_c, flux_c, err_c)
    
    # Phase fold
    phase = ((time_c - tls_res.T0 + 0.5 * tls_res.period) % tls_res.period) / tls_res.period - 0.5
    sort_idx = np.argsort(phase)
    phase_s, flux_s = phase[sort_idx], flux_c[sort_idx]
    
    from ml_feature_builder import build_global_view, build_local_view
    global_view = build_global_view(phase_s, flux_s)
    local_view  = build_local_view(phase_s, flux_s)
    
    # Build scalar features
    FAP, _ = stage6_fap(tls_res)
    centroid_result = stage7_centroid(
        time_c, ccol_c, crow_c,
        tls_res.period, tls_res.T0, tls_res.duration)
    oe_result = stage8_vetoes(time_c, flux_c, tls_res)
    
    fractional_depth = abs(1.0 - tls_res.depth)
    corrected_depth = (fractional_depth / crowdsap_result['crowdsap_correction']
                       if crowdsap_result['crowdsap_correction'] > 0 else fractional_depth)
    rp = np.sqrt(abs(corrected_depth)) * meta.get('rad', 1.0) * 109.076
    
    scalar = {
        'tic_id': tic_seed, 'sector': sector,
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
        'has_multiple_planets': 0,
        'centroid_shift_pix': centroid_result['shift_pix'],
        'centroid_p_col': centroid_result['p_col'],
        'centroid_blend_flag': int(centroid_result['blend_suspect']),
        'odd_even_sigma': oe_result['odd_even_sigma'],
        'odd_even_flag': int(oe_result['odd_even_flag']),
        'secondary_sigma': oe_result['secondary_sigma'],
        'secondary_flag': int(oe_result['secondary_flag']),
        'verdict_score': 0,
        'is_significant': int(tls_res.SDE >= 3.0),
        'label': label,
    }
    
    # Extra features
    extras = compute_extra_features(time_c, flux_c, err_c, tls_res, meta, crowdsap_result)
    scalar.update(extras)
    
    return global_view, local_view, scalar


# ═══════════════════════════════════════════════════════════════
# SPLIT CREATION
# ═══════════════════════════════════════════════════════════════

def create_stratified_splits(global_views, local_views, scalar_df, labels,
                              train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """
    Create stratified train/val/test splits.
    Ensures each split has proportional class representation.
    """
    from sklearn.model_selection import train_test_split
    
    n = len(labels)
    indices = np.arange(n)
    
    # First split: train vs (val + test)
    train_idx, rest_idx = train_test_split(
        indices, test_size=(val_frac + test_frac),
        stratify=labels, random_state=RANDOM_STATE
    )
    
    # Second split: val vs test
    rest_labels = labels[rest_idx]
    val_rel = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=(1 - val_rel),
        stratify=rest_labels, random_state=RANDOM_STATE
    )
    
    splits = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[name] = {
            'global_views': global_views[idx],
            'local_views': local_views[idx],
            'scalars': scalar_df.iloc[idx].reset_index(drop=True),
            'labels': labels[idx],
        }
    
    return splits


def save_splits(splits):
    """Save train/val/test splits to disk."""
    for name, data in splits.items():
        prefix = os.path.join(ML_SPLITS_DIR, name)
        np.savez_compressed(
            f'{prefix}.npz',
            global_views=data['global_views'],
            local_views=data['local_views'],
            labels=data['labels'],
        )
        data['scalars'].to_csv(f'{prefix}_scalars.csv', index=False)
    
    print(f"\n[DATA LOADER] Splits saved to {ML_SPLITS_DIR}/")
    for name, data in splits.items():
        dist = dict(Counter(data['labels']))
        readable = {LABEL_NAMES.get(k, k): v for k, v in dist.items()}
        print(f"  {name}: {len(data['labels'])} samples — {readable}")


def load_splits():
    """Load pre-saved splits from disk."""
    splits = {}
    for name in ['train', 'val', 'test']:
        prefix = os.path.join(ML_SPLITS_DIR, name)
        npz = np.load(f'{prefix}.npz')
        scalars = pd.read_csv(f'{prefix}_scalars.csv')
        splits[name] = {
            'global_views': npz['global_views'],
            'local_views': npz['local_views'],
            'scalars': scalars,
            'labels': npz['labels'],
        }
    return splits


# ═══════════════════════════════════════════════════════════════
# MAIN: Assemble all data sources + create splits
# ═══════════════════════════════════════════════════════════════

def prepare_training_data(curated_csv=None, use_kepler_dr25=False, 
                           n_synthetic_per_class=300, sector=1):
    """
    Main entry point: assembles training data from all sources,
    creates stratified splits, saves to disk.
    
    Args:
        curated_csv: Path to labeled curated dataset (optional)
        use_kepler_dr25: Whether to download Kepler DR25 dispositions
        n_synthetic_per_class: Number of synthetic examples per class
        sector: TESS sector number
    
    Returns:
        splits dict with train/val/test data
    """
    print("\n" + "=" * 60)
    print("  ML DATA LOADER — Assembling Training Data")
    print("=" * 60)
    
    all_global = []
    all_local  = []
    all_scalars = []
    all_labels = []
    
    # ── Source 1: Curated dataset
    if curated_csv:
        curated_df = load_curated_dataset(curated_csv)
        if len(curated_df) > 0:
            print("\n[DATA LOADER] Extracting features from curated dataset...")
            for _, row in curated_df.iterrows():
                gv, lv, sc, status = extract_features_for_target(
                    int(row['tic_id']), int(row.get('sector', sector)),
                    use_network=True
                )
                if gv is not None:
                    all_global.append(gv)
                    all_local.append(lv)
                    sc['label'] = row['label']
                    all_scalars.append(sc)
                    all_labels.append(LABEL_MAP.get(row['label'], 3))
    
    # ── Source 2: Kepler DR25 (optional supplement)
    # Note: This requires downloading Kepler light curves which is slow.
    # Skip for hackathon unless you have time.
    
    # ── Source 3: Synthetic data (always available)
    gv_syn, lv_syn, sc_syn, lab_syn = generate_synthetic_training_data(
        n_per_class=n_synthetic_per_class, sector=sector
    )
    
    if len(gv_syn) > 0:
        all_global.append(gv_syn)
        all_local.append(lv_syn)
        all_scalars.append(sc_syn)
        all_labels.append(lab_syn)
    
    # ── Combine all sources
    if len(all_global) == 0:
        raise ValueError("No training data generated!")
    
    # Handle mix of lists and arrays
    global_views = np.concatenate([g if isinstance(g, np.ndarray) and g.ndim == 2 
                                    else np.array([g]) for g in all_global])
    local_views  = np.concatenate([l if isinstance(l, np.ndarray) and l.ndim == 2 
                                    else np.array([l]) for l in all_local])
    labels = np.concatenate([l if isinstance(l, np.ndarray) 
                             else np.array([l]) for l in all_labels])
    scalar_df = pd.concat(all_scalars, ignore_index=True)
    
    print(f"\n[DATA LOADER] Total assembled: {len(labels)} samples")
    print(f"  Distribution: {dict(Counter(labels))}")
    readable = {LABEL_NAMES.get(k, k): v for k, v in Counter(labels).items()}
    print(f"  Classes: {readable}")
    
    # ── Create splits
    splits = create_stratified_splits(global_views, local_views, scalar_df, labels)
    save_splits(splits)
    
    return splits


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ML Data Loader')
    parser.add_argument('--curated', type=str, default=None,
                        help='Path to curated labeled dataset CSV')
    parser.add_argument('--n-synthetic', type=int, default=300,
                        help='Number of synthetic samples per class')
    parser.add_argument('--sector', type=int, default=1,
                        help='TESS sector number')
    args = parser.parse_args()
    
    splits = prepare_training_data(
        curated_csv=args.curated,
        n_synthetic_per_class=args.n_synthetic,
        sector=args.sector
    )
    
    print("\n[DATA LOADER] Done! Splits ready for training.")
