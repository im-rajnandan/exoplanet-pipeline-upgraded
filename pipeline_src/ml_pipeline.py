"""
================================================================
 ML PIPELINE — End-to-End Orchestrator
================================================================
 Runs the complete ML pipeline:
   Step 1: Build features from preprocessing output
   Step 2: Assemble training data + splits
   Step 3: Train autoencoder anomaly pre-filter (Stage 1)
   Step 4: Train dual-stream classifier (Stage 2)
   Step 5: Calibrated inference with uncertainty (Stage 3)
   Step 6: Parameter estimation for planet candidates
   Step 7: Generate visualizations
   Step 8: Export final catalog

 Usage:
   python ml_pipeline.py                    # Full pipeline
   python ml_pipeline.py --skip-training    # Inference only (load models)
   python ml_pipeline.py --n-synthetic 100  # Quick demo
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
import time
import argparse
import json

# Pipeline modules
from ml_feature_builder import (build_features_from_batch, build_features_from_tic_list,
                                 extract_features_for_target, ML_FEATURES_DIR)
from ml_data_loader import (prepare_training_data, load_splits, 
                              ML_SPLITS_DIR, LABEL_MAP, LABEL_NAMES)
from ml_stage1_autoencoder import AutoencoderTrainer, MODEL_DIR
from ml_stage2_classifier import DualStreamTrainer, CLASS_NAMES
from ml_parameter_estimation import estimate_parameters, estimate_all_candidates
from ml_visualize import (plot_sector_dashboard, plot_training_history,
                           plot_candidate_diagnostic, plot_parameter_card,
                           plot_calibration_curve, generate_all_visualizations,
                           OUT_DIR)

# ── Config
RESULTS_DIR = './tess_pipeline_output/ml_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_full_pipeline(batch_csv=None, curated_csv=None, sector=1,
                       n_synthetic=300, skip_training=False,
                       use_network=False, mc_samples=50):
    """
    Run the complete ML pipeline end-to-end.
    
    Args:
        batch_csv: Path to preprocessing batch results CSV
        curated_csv: Path to labeled curated dataset
        sector: TESS sector number
        n_synthetic: Number of synthetic samples per class for training
        skip_training: If True, load pre-trained models
        use_network: Whether to download real TESS data
        mc_samples: Number of MC Dropout samples for uncertainty
    """
    
    t_start = time.time()
    
    print("\n" + "═" * 65)
    print("  EXOPLANET ML CLASSIFICATION PIPELINE")
    print("  Hierarchical Dual-Stream Ensemble")
    print("═" * 65)
    print(f"  Sector: {sector}")
    print(f"  Batch CSV: {batch_csv or 'None (synthetic only)'}")
    print(f"  Curated labels: {curated_csv or 'None'}")
    print(f"  Synthetic per class: {n_synthetic}")
    print(f"  Mode: {'Inference' if skip_training else 'Training + Inference'}")
    print("═" * 65)
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: Build Features
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 1: Feature Extraction")
    print("▓" * 60)
    
    if batch_csv and os.path.exists(batch_csv):
        gv_path = os.path.join(ML_FEATURES_DIR, 'global_views.npy')
        sc_path = os.path.join(ML_FEATURES_DIR, 'scalar_features.csv')
        
        if os.path.exists(gv_path) and os.path.exists(sc_path):
            print("[STEP 1] Loading cached features...")
            science_global = np.load(gv_path)
            science_local = np.load(os.path.join(ML_FEATURES_DIR, 'local_views.npy'))
            science_scalars = pd.read_csv(sc_path)
            science_meta = pd.read_csv(os.path.join(ML_FEATURES_DIR, 'metadata.csv'))
        else:
            science_global, science_local, science_scalars, science_meta = \
                build_features_from_batch(batch_csv, sector=sector, use_network=use_network)
    else:
        print("[STEP 1] No batch CSV — will use synthetic data only.")
        science_global = None
        science_local = None
        science_scalars = None
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: Training Data Assembly
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 2: Training Data Assembly")
    print("▓" * 60)
    
    splits_exist = os.path.exists(os.path.join(ML_SPLITS_DIR, 'train.npz'))
    
    if not skip_training or not splits_exist:
        splits = prepare_training_data(
            curated_csv=curated_csv,
            n_synthetic_per_class=n_synthetic,
            sector=sector
        )
    else:
        print("[STEP 2] Loading cached splits...")
        splits = load_splits()
        for name in ['train', 'val', 'test']:
            print(f"  {name}: {len(splits[name]['labels'])} samples")
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: Autoencoder Anomaly Pre-filter (Stage 1)
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 3: Autoencoder Anomaly Pre-filter")
    print("▓" * 60)
    
    ae_trainer = AutoencoderTrainer(
        input_length=splits['train']['global_views'].shape[1],
        latent_dim=32,
        epochs=100,
        patience=10,
        batch_size=64,
    )
    
    ae_model_path = os.path.join(MODEL_DIR, 'autoencoder_best.pth')
    
    if skip_training and os.path.exists(ae_model_path):
        print("[STEP 3] Loading pre-trained autoencoder...")
        ae_trainer.load()
    else:
        # Train on ALL training data (autoencoder doesn't need labels)
        all_global = np.concatenate([
            splits['train']['global_views'],
            splits['val']['global_views']
        ])
        all_sde = None
        if 'SDE' in splits['train']['scalars'].columns:
            all_sde = np.concatenate([
                splits['train']['scalars']['SDE'].values,
                splits['val']['scalars']['SDE'].values
            ])
        
        ae_trainer.train(all_global, sde_values=all_sde, sde_threshold=6.0)
    
    # Get anomaly scores for test set
    test_scores, test_anomalous = ae_trainer.predict_anomaly_scores(
        splits['test']['global_views']
    )
    
    # Also score science data if available
    if science_global is not None:
        sci_scores, sci_anomalous = ae_trainer.predict_anomaly_scores(science_global)
        print(f"  Science data: {sci_anomalous.sum()}/{len(sci_anomalous)} anomalous")
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: Dual-Stream Classifier (Stage 2)
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 4: Dual-Stream Classifier")
    print("▓" * 60)
    
    classifier = DualStreamTrainer(
        cnn_epochs=50,
        cnn_patience=7,
        batch_size=64,
        mc_dropout_samples=mc_samples,
    )
    
    fusion_model_path = os.path.join(MODEL_DIR, 'fusion_best.pth')
    
    if skip_training and os.path.exists(fusion_model_path):
        print("[STEP 4] Loading pre-trained classifier...")
        classifier.load()
    else:
        best_acc = classifier.train(splits)
        print(f"\n  Best validation accuracy: {best_acc:.4f}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: Evaluation + Calibrated Inference (Stage 3)
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 5: Calibrated Inference + Uncertainty")
    print("▓" * 60)
    
    # Evaluate on test split
    test_acc, test_results = classifier.evaluate(splits, 'test')
    
    # Classify science data if available
    science_results = None
    if science_global is not None and science_scalars is not None:
        print("\n  Classifying science data...")
        science_results = classifier.predict(
            science_global, science_local, science_scalars,
            mc_samples=mc_samples
        )
        
        # Combine autoencoder + classifier results
        # Override removed (Bug 3 fix) - the autoencoder anomaly threshold was filtering out valid synthetic planets
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: Parameter Estimation for Planet Candidates
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 6: Parameter Estimation")
    print("▓" * 60)
    
    params_list = []
    
    # Find planet candidates (from test set or science data)
    if science_results is not None:
        planet_mask = science_results['predictions'] == LABEL_MAP['PLANET']
        planet_indices = np.where(planet_mask)[0]
        
        if len(planet_indices) > 0:
            print(f"\n  Found {len(planet_indices)} planet candidates in science data")
            planet_df = science_scalars.iloc[planet_indices]
            
            if 'tic_id' in planet_df.columns:
                params_list = estimate_all_candidates(
                    planet_df, sector=sector, use_network=use_network)
            else:
                print("  No TIC IDs in scalar features — skipping parameter estimation")
        else:
            print("  No planet candidates found in science data")
    
    # Also estimate for test set planets (for validation)
    test_planet_mask = (test_results['predictions'] == LABEL_MAP['PLANET'])
    test_planet_count = test_planet_mask.sum()
    print(f"  Test set: {test_planet_count} classified as PLANET")
    
    # ══════════════════════════════════════════════════════════
    # STEP 7: Visualization
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 7: Generating Visualizations")
    print("▓" * 60)
    
    # Training history
    if classifier.train_history.get('train_loss'):
        plot_training_history(classifier.train_history)
    
    # Test set dashboard
    plot_sector_dashboard(
        test_results['predictions'],
        true_labels=splits['test']['labels'],
        probabilities=test_results.get('probabilities'),
        params_list=params_list,
    )
    
    # Calibration curve
    if 'probabilities' in test_results:
        plot_calibration_curve(
            test_results['probabilities'],
            splits['test']['labels']
        )
    
    # Science data dashboard
    if science_results is not None:
        plot_sector_dashboard(
            science_results['predictions'],
            probabilities=science_results.get('probabilities'),
            params_list=params_list,
        )
    
    # ══════════════════════════════════════════════════════════
    # STEP 8: Export Final Catalog
    # ══════════════════════════════════════════════════════════
    
    print("\n\n" + "▓" * 60)
    print("  STEP 8: Exporting Final Catalog")
    print("▓" * 60)
    
    catalog = export_catalog(
        test_results, splits['test'],
        science_results, science_scalars,
        params_list,
        test_anomaly_scores=test_scores,
    )
    
    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    
    elapsed = time.time() - t_start
    
    print("\n\n" + "═" * 65)
    print("  PIPELINE COMPLETE")
    print("═" * 65)
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Planet candidates (test): {test_planet_count}")
    if science_results is not None:
        sci_planets = (science_results['predictions'] == LABEL_MAP['PLANET']).sum()
        sci_review = science_results['needs_review'].sum()
        print(f"  Planet candidates (science): {sci_planets}")
        print(f"  Needs manual review: {sci_review}")
    if params_list:
        print(f"  Parameters estimated: {len(params_list)} candidates")
    print(f"\n  Outputs:")
    print(f"    Models:     {MODEL_DIR}/")
    print(f"    Results:    {RESULTS_DIR}/")
    print(f"    Plots:      {OUT_DIR}/")
    print("═" * 65)
    
    return catalog


def export_catalog(test_results, test_data, science_results=None,
                    science_scalars=None, params_list=None,
                    test_anomaly_scores=None):
    """
    Export the final classification catalog.
    """
    rows = []
    
    # Test set catalog
    for i in range(len(test_results['predictions'])):
        row = {
            'dataset': 'test',
            'prediction': CLASS_NAMES[test_results['predictions'][i]],
            'confidence': float(test_results['probabilities'][i].max()),
            'uncertainty': float(test_results['uncertainty'][i]),
            'needs_review': bool(test_results['needs_review'][i]),
            'true_label': LABEL_NAMES.get(test_data['labels'][i], '?'),
        }
        
        # Add probabilities
        for c, name in enumerate(CLASS_NAMES):
            row[f'prob_{name}'] = float(test_results['probabilities'][i][c])
        
        # Add anomaly score
        if test_anomaly_scores is not None:
            row['anomaly_score'] = float(test_anomaly_scores[i])
        
        rows.append(row)
    
    # Science data catalog
    if science_results is not None and science_scalars is not None:
        for i in range(len(science_results['predictions'])):
            row = {
                'dataset': 'science',
                'prediction': CLASS_NAMES[science_results['predictions'][i]],
                'confidence': float(science_results['probabilities'][i].max()),
                'uncertainty': float(science_results['uncertainty'][i]),
                'needs_review': bool(science_results['needs_review'][i]),
            }
            
            # Add TIC ID if available
            if 'tic_id' in science_scalars.columns:
                row['tic_id'] = int(science_scalars.iloc[i]['tic_id'])
            
            for c, name in enumerate(CLASS_NAMES):
                row[f'prob_{name}'] = float(science_results['probabilities'][i][c])
            
            rows.append(row)
    
    catalog_df = pd.DataFrame(rows)
    
    # Add parameter estimates
    if params_list:
        params_df = pd.DataFrame(params_list)
        params_path = os.path.join(RESULTS_DIR, 'parameter_estimates.csv')
        params_df.to_csv(params_path, index=False)
        print(f"  Parameters: {params_path}")
    
    catalog_path = os.path.join(RESULTS_DIR, 'classification_catalog.csv')
    catalog_df.to_csv(catalog_path, index=False)
    
    print(f"\n  Final catalog: {catalog_path}")
    print(f"  Total entries: {len(catalog_df)}")
    print(f"  Class distribution:")
    for cls in CLASS_NAMES:
        n = (catalog_df['prediction'] == cls).sum()
        print(f"    {cls:8s}: {n}")
    n_review = catalog_df['needs_review'].sum()
    print(f"  Needs review: {n_review}")
    
    return catalog_df


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exoplanet ML Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_pipeline.py                              # Full training + inference
  python ml_pipeline.py --n-synthetic 100            # Quick demo (fewer samples)
  python ml_pipeline.py --skip-training              # Load saved models
  python ml_pipeline.py --batch-csv results.csv      # Use real preprocessing output
  python ml_pipeline.py --curated-csv labels.csv     # Include labeled training data
        """
    )
    parser.add_argument('--batch-csv', type=str, 
                        default='./tess_pipeline_output/batch_sector1_results.csv',
                        help='Path to preprocessing batch results CSV')
    parser.add_argument('--curated-csv', type=str, default=None,
                        help='Path to curated labeled dataset CSV')
    parser.add_argument('--sector', type=int, default=1,
                        help='TESS sector number')
    parser.add_argument('--n-synthetic', type=int, default=300,
                        help='Number of synthetic samples per class')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, load pre-trained models')
    parser.add_argument('--use-network', action='store_true',
                        help='Download real TESS data from MAST')
    parser.add_argument('--mc-samples', type=int, default=50,
                        help='Number of MC Dropout samples for uncertainty')
    
    args = parser.parse_args()
    
    catalog = run_full_pipeline(
        batch_csv=args.batch_csv if os.path.exists(args.batch_csv) else None,
        curated_csv=args.curated_csv,
        sector=args.sector,
        n_synthetic=args.n_synthetic,
        skip_training=args.skip_training,
        use_network=args.use_network,
        mc_samples=args.mc_samples,
    )
