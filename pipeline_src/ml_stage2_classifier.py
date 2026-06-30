"""
================================================================
 ML STAGE 2 — DUAL-STREAM CLASSIFIER + CALIBRATION
================================================================
 Two parallel streams:
   Stream A: 1D CNN (AstroNet two-tower) on phase-folded views
   Stream B: LightGBM on scalar physics features
 
 Fusion: CNN embedding + LightGBM leaf embedding → stacker
 
 Stage 3 integrated:
   - Temperature scaling for calibrated probabilities
   - MC Dropout for epistemic uncertainty
   - "Requires manual review" routing
   - Counterfactual importance reporting
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
import json
import logging
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import lightgbm as lgb
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
import joblib

# ── Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                       'mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_DIR = './tess_pipeline_output/ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)

N_CLASSES = 4
CLASS_NAMES = ['PLANET', 'EB', 'BLEND', 'OTHER']

# Scalar features used by LightGBM (Stream B)
SCALAR_FEATURE_COLS = [
    'SDE', 'SNR', 'FAP', 'period', 'duration_hrs',
    'depth_ppm_obs', 'depth_ppm_corr', 'Rp_earth',
    'crowdsap', 'flfrcsap', 'crowdsap_flag',
    'harmonic_flag', 'has_multiple_planets',
    'centroid_shift_pix', 'centroid_p_col', 'centroid_blend_flag',
    'odd_even_sigma', 'odd_even_flag',
    'secondary_sigma', 'secondary_flag',
    'depth_to_noise', 'n_transits', 'snr_per_transit',
    'period_to_span', 'in_transit_fraction',
    'transit_asymmetry', 'oot_rms_ppm',
    'stellar_rad', 'stellar_teff', 'stellar_logg',
    'residual_ppm',
]


# ═══════════════════════════════════════════════════════════════
# FOCAL LOSS (handles class imbalance better than cross-entropy)
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Reduces the loss contribution from easy examples, focusing
    training on hard misclassified samples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # Per-class weights
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ═══════════════════════════════════════════════════════════════
# STREAM A: 1D CNN (AstroNet-style Two-Tower)
# ═══════════════════════════════════════════════════════════════

class MPSPool1d(nn.Module):
    """Custom pool to replace AdaptiveAvgPool1d(1) which crashes on MPS."""
    def forward(self, x):
        return x.mean(dim=2, keepdim=True)

class GlobalTower(nn.Module):
    """Processes full-orbit phase-folded view (201 points)."""
    def __init__(self, input_length=201):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.1),
            MPSPool1d(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))


class LocalTower(nn.Module):
    """Processes zoomed transit view (61 points)."""
    def __init__(self, input_length=61):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.1),
            MPSPool1d(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))


class CNNStreamA(nn.Module):
    """
    Stream A: Two-tower CNN combining global and local views.
    Output: 96-dimensional embedding (64 global + 32 local).
    """
    def __init__(self, global_len=201, local_len=61):
        super().__init__()
        self.global_tower = GlobalTower(global_len)
        self.local_tower = LocalTower(local_len)
        self.combined = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
    
    def forward(self, global_view, local_view):
        g = self.global_tower(global_view)
        l = self.local_tower(local_view)
        combined = torch.cat([g, l], dim=1)
        return self.combined(combined)


# ═══════════════════════════════════════════════════════════════
# FUSION MODEL (CNN + LightGBM embeddings → final prediction)
# ═══════════════════════════════════════════════════════════════

class FusionClassifier(nn.Module):
    """
    Combines CNN Stream A output with LightGBM leaf embeddings.
    
    Input: CNN_embedding(64) + LGBM_leaf_embedding(n_trees)
    Output: logits(4)
    """
    def __init__(self, cnn_dim=64, lgbm_dim=500, n_classes=4, max_leaves=100, emb_dim=2):
        super().__init__()
        self.cnn_stream = CNNStreamA()
        self.leaf_emb = nn.Embedding(max_leaves, emb_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + lgbm_dim * emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        self.n_classes = n_classes
    
    def forward(self, global_view, local_view, lgbm_features):
        cnn_emb = self.cnn_stream(global_view, local_view)
        l_emb = self.leaf_emb(lgbm_features.long())
        l_emb = l_emb.view(l_emb.size(0), -1)
        combined = torch.cat([cnn_emb, l_emb], dim=1)
        logits = self.fusion(combined)
        return logits
    
    def get_cnn_embedding(self, global_view, local_view):
        """Extract CNN embedding without fusion (for analysis)."""
        return self.cnn_stream(global_view, local_view)


# ═══════════════════════════════════════════════════════════════
# TEMPERATURE SCALING (Stage 3 — Calibration)
# ═══════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar T to calibrate softmax probabilities.
    calibrated_prob = softmax(logits / T)
    
    Fit on validation set AFTER training the classifier.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Optimize temperature on validation set."""
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.LongTensor(labels)
        
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_tensor)
            loss = criterion(scaled, labels_tensor)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        print(f"[CALIBRATION] Optimal temperature: {self.temperature.item():.4f}")
        return self


# ═══════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def augment_light_curve(global_view, local_view):
    """
    Apply random augmentations to phase-folded light curves.
    - Random phase shift (±0.02)
    - Gaussian noise injection
    - Random vertical scaling
    """
    gv = global_view.copy()
    lv = local_view.copy()
    
    # Phase shift (circular roll)
    shift = np.random.randint(-4, 5)  # ±4 bins ≈ ±0.02 phase
    gv = np.roll(gv, shift, axis=-1)
    
    # Gaussian noise injection
    noise_level = np.random.uniform(0.0001, 0.001)
    gv += np.random.normal(0, noise_level, gv.shape)
    lv += np.random.normal(0, noise_level, lv.shape)
    
    # Random vertical scaling (±5%)
    scale = np.random.uniform(0.95, 1.05)
    gv = 1.0 + (gv - 1.0) * scale
    lv = 1.0 + (lv - 1.0) * scale
    
    return gv, lv


class AugmentedTensorDataset(Dataset):
    def __init__(self, gv, lv, lf, y, augment=True):
        self.gv = gv
        self.lv = lv
        self.lf = lf
        self.y = y
        self.augment = augment
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        gv = self.gv[idx]
        lv = self.lv[idx]
        lf = self.lf[idx]
        y = self.y[idx]
        
        if self.augment:
            gv_np = gv.cpu().numpy().squeeze(0)
            lv_np = lv.cpu().numpy().squeeze(0)
            gv_aug, lv_aug = augment_light_curve(gv_np, lv_np)
            gv = torch.FloatTensor(gv_aug).unsqueeze(0).to(self.gv.device)
            lv = torch.FloatTensor(lv_aug).unsqueeze(0).to(self.lv.device)
            
        return gv, lv, lf, y


# ═══════════════════════════════════════════════════════════════
# DUAL-STREAM TRAINER
# ═══════════════════════════════════════════════════════════════

class DualStreamTrainer:
    """
    Complete training pipeline for the dual-stream classifier.
    Handles Stream B (LightGBM), Stream A (CNN), fusion, and calibration.
    """
    
    def __init__(self, n_classes=N_CLASSES, cnn_epochs=50, cnn_patience=7,
                 cnn_lr=1e-3, batch_size=64, mc_dropout_samples=50):
        self.n_classes = n_classes
        self.cnn_epochs = cnn_epochs
        self.cnn_patience = cnn_patience
        self.cnn_lr = cnn_lr
        self.batch_size = batch_size
        self.mc_dropout_samples = mc_dropout_samples
        
        self.lgbm_model = None
        self.fusion_model = None
        self.scaler = StandardScaler()
        self.temp_scaler = TemperatureScaler()
        
        self.lgbm_n_trees = 500
        self.train_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def _get_scalar_features(self, scalar_df):
        """Extract and impute scalar feature matrix from DataFrame."""
        available_cols = [c for c in SCALAR_FEATURE_COLS if c in scalar_df.columns]
        X = scalar_df[available_cols].copy()
        
        # Fill NaN/inf with column medians
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        
        return X.values.astype(np.float32), available_cols
    
    # ── Stream B: LightGBM ──────────────────────────────────────
    
    def train_stream_b(self, train_scalars, train_labels, val_scalars, val_labels):
        """Train LightGBM classifier on scalar features."""
        print("\n" + "─" * 50)
        print("  Stream B: Training LightGBM on scalar features")
        print("─" * 50)
        
        X_train, cols = self._get_scalar_features(train_scalars)
        X_val, _ = self._get_scalar_features(val_scalars)
        
        # Fit scaler
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Compute class weights
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        sample_weights = np.array([total / (self.n_classes * class_counts[l]) 
                                    for l in train_labels])
        
        self.lgbm_model = lgb.LGBMClassifier(
            n_estimators=self.lgbm_n_trees,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        
        self.lgbm_model.fit(
            X_train, train_labels,
            eval_set=[(X_val, val_labels)],
            callbacks=[lgb.log_evaluation(50)],
        )
        
        # Evaluate
        val_pred = self.lgbm_model.predict(X_val)
        val_acc = accuracy_score(val_labels, val_pred)
        print(f"\n  LightGBM Validation Accuracy: {val_acc:.4f}")
        print(classification_report(val_labels, val_pred, 
                                     target_names=CLASS_NAMES, zero_division=0))
        
        # Feature importance
        importances = self.lgbm_model.feature_importances_
        feat_imp = sorted(zip(cols, importances), key=lambda x: -x[1])
        print("  Top 10 features:")
        for name, imp in feat_imp[:10]:
            print(f"    {name:25s} : {imp:.0f}")
        
        # Save
        joblib.dump(self.lgbm_model, os.path.join(MODEL_DIR, 'lgbm_stream_b.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scalar_scaler.pkl'))
        
        return val_acc
    
    def get_lgbm_leaf_embeddings(self, scalar_df):
        """
        Extract LightGBM leaf index embeddings.
        Shape: (N, n_trees) — each entry is the leaf index for that tree.
        """
        X, _ = self._get_scalar_features(scalar_df)
        X = self.scaler.transform(X)
        leaf_preds = self.lgbm_model.predict(X, pred_leaf=True)
        
        # Returned as raw integers for nn.Embedding
        return leaf_preds
    
    # ── Full Pipeline Training ──────────────────────────────────
    
    def train(self, splits):
        """
        Train the complete dual-stream pipeline.
        
        Args:
            splits: dict with 'train', 'val', 'test' keys,
                    each containing global_views, local_views, scalars, labels
        """
        print("\n" + "=" * 60)
        print("  STAGE 2: Training Dual-Stream Classifier")
        print(f"  Device: {DEVICE}")
        print("=" * 60)
        
        train = splits['train']
        val = splits['val']
        
        # ── Step 1: Train Stream B (LightGBM)
        self.train_stream_b(
            train['scalars'], train['labels'],
            val['scalars'], val['labels']
        )
        
        # ── Step 2: Get LightGBM leaf embeddings for all data
        train_leaves = self.get_lgbm_leaf_embeddings(train['scalars'])
        val_leaves = self.get_lgbm_leaf_embeddings(val['scalars'])
        
        self.lgbm_n_trees = train_leaves.shape[1]  # Actual tree count
        
        # ── Step 3: Train Fusion Model (CNN + LightGBM)
        print("\n" + "─" * 50)
        print("  Training Fusion Model (CNN + LightGBM)")
        print("─" * 50)
        
        self.fusion_model = FusionClassifier(
            cnn_dim=64, lgbm_dim=self.lgbm_n_trees, n_classes=self.n_classes
        ).to(DEVICE)
        
        # Prepare tensors
        train_gv = torch.FloatTensor(train['global_views'][:, np.newaxis, :])
        train_lv = torch.FloatTensor(train['local_views'][:, np.newaxis, :])
        train_lf = torch.FloatTensor(train_leaves)
        train_y  = torch.LongTensor(train['labels'])
        
        # Split validation into early stopping (val_stop) and calibration (val_calib)
        N_val = len(val['labels'])
        indices = np.random.permutation(N_val)
        split_idx = N_val // 2
        
        val_stop_idx = indices[:split_idx]
        val_calib_idx = indices[split_idx:]
        
        val_gv = torch.FloatTensor(val['global_views'][:, np.newaxis, :])
        val_lv = torch.FloatTensor(val['local_views'][:, np.newaxis, :])
        val_lf = torch.FloatTensor(val_leaves)
        val_y  = torch.LongTensor(val['labels'])
        
        val_stop_gv, val_stop_lv = val_gv[val_stop_idx], val_lv[val_stop_idx]
        val_stop_lf, val_stop_y = val_lf[val_stop_idx], val_y[val_stop_idx]
        
        val_calib_gv, val_calib_lv = val_gv[val_calib_idx], val_lv[val_calib_idx]
        val_calib_lf, val_calib_y = val_lf[val_calib_idx], val_y[val_calib_idx]
        
        # Class weights for focal loss
        class_counts = Counter(train['labels'].tolist())
        total = sum(class_counts.values())
        weights = torch.FloatTensor([total / (self.n_classes * class_counts.get(i, 1)) 
                                      for i in range(self.n_classes)]).to(DEVICE)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        
        optimizer = optim.AdamW(self.fusion_model.parameters(), 
                                lr=self.cnn_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cnn_epochs)
        
        train_dataset = AugmentedTensorDataset(train_gv, train_lv, train_lf, train_y, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                  shuffle=True, drop_last=False)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.cnn_epochs):
            # ── Train
            self.fusion_model.train()
            train_loss = 0.0
            correct = 0
            total_samples = 0
            
            for batch_gv, batch_lv, batch_lf, batch_y in train_loader:
                batch_gv, batch_lv = batch_gv.to(DEVICE), batch_lv.to(DEVICE)
                batch_lf, batch_y = batch_lf.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                logits = self.fusion_model(batch_gv, batch_lv, batch_lf)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)
            
            scheduler.step()
            train_loss /= total_samples
            train_acc = correct / total_samples
            
            # ── Validate
            self.fusion_model.eval()
            val_stop_dataset = TensorDataset(val_stop_gv, val_stop_lv, val_stop_lf, val_stop_y)
            val_stop_loader = DataLoader(val_stop_dataset, batch_size=self.batch_size, shuffle=False)
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for v_gv, v_lv, v_lf, v_y in val_stop_loader:
                    v_gv, v_lv, v_lf, v_y = v_gv.to(DEVICE), v_lv.to(DEVICE), v_lf.to(DEVICE), v_y.to(DEVICE)
                    logits = self.fusion_model(v_gv, v_lv, v_lf)
                    loss = criterion(logits, v_y)
                    val_loss += loss.item() * len(v_y)
                    preds = logits.argmax(dim=1)
                    correct += (preds == v_y).sum().item()
            val_loss /= len(val_stop_y)
            val_acc = correct / len(val_stop_y)
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.fusion_model.state_dict(),
                          os.path.join(MODEL_DIR, 'fusion_best.pth'))
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.cnn_epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"Best: {best_val_acc:.4f}")
            
            if patience_counter >= self.cnn_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.fusion_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, 'fusion_best.pth'),
                       map_location=DEVICE, weights_only=True))
        
        # ── Step 4: Temperature Scaling on Calibration Set
        print("\n  Calibrating Probabilities (Temperature Scaling)...")
        self.fusion_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'fusion_best.pth')))
        self.fusion_model.eval()
        
        calib_logits_list = []
        val_calib_dataset = TensorDataset(val_calib_gv, val_calib_lv, val_calib_lf)
        val_calib_loader = DataLoader(val_calib_dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for v_gv, v_lv, v_lf in val_calib_loader:
                v_gv, v_lv, v_lf = v_gv.to(DEVICE), v_lv.to(DEVICE), v_lf.to(DEVICE)
                calib_logits_list.append(self.fusion_model(v_gv, v_lv, v_lf).cpu().numpy())
        
        calib_logits = np.concatenate(calib_logits_list, axis=0)
        self.temp_scaler = TemperatureScaler().to(DEVICE)
        self.temp_scaler.fit(calib_logits, val_calib_y.numpy())
        
        # Final calibrated val predictions
        with torch.no_grad():
            val_logits = self.fusion_model(val_gv, val_lv, val_lf)
            calibrated_logits = self.temp_scaler(val_logits)
            calibrated_probs = F.softmax(calibrated_logits, dim=1).cpu().numpy()
            val_preds_final = calibrated_probs.argmax(axis=1)
        
        print(f"\n  Final Calibrated Validation Results:")
        print(classification_report(val['labels'], val_preds_final,
                                     target_names=CLASS_NAMES, zero_division=0))
        
        # Save everything
        self._save_all()
        
        return best_val_acc
    
    # ── Inference with Uncertainty ──────────────────────────────
    
    def predict(self, global_views, local_views, scalar_df,
                mc_samples=None, return_details=True):
        """
        Run calibrated inference with MC Dropout uncertainty.
        
        Returns dict with:
          - probabilities: (N, 4) calibrated class probabilities
          - predictions: (N,) predicted class indices
          - uncertainty: (N,) epistemic uncertainty (MC Dropout std)
          - needs_review: (N,) boolean — flagged for manual review
          - class_names: (N,) string predictions
        """
        if mc_samples is None:
            mc_samples = self.mc_dropout_samples
        
        # Get LightGBM embeddings
        leaves = self.get_lgbm_leaf_embeddings(scalar_df)
        
        # Prepare tensors
        gv = torch.FloatTensor(global_views[:, np.newaxis, :]).to(DEVICE)
        lv = torch.FloatTensor(local_views[:, np.newaxis, :]).to(DEVICE)
        lf = torch.FloatTensor(leaves).to(DEVICE)
        
        # ── MC Dropout: run N forward passes with dropout active
        all_probs = []
        self.fusion_model.train()  # Keep dropout active
        
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.fusion_model(gv, lv, lf)
                # Apply temperature scaling
                scaled_logits = self.temp_scaler(logits.cpu())
                probs = F.softmax(scaled_logits, dim=1).numpy()
                all_probs.append(probs)
        
        self.fusion_model.eval()  # Reset to eval mode
        
        all_probs = np.array(all_probs)  # (mc_samples, N, n_classes)
        
        # Mean prediction (calibrated)
        mean_probs = all_probs.mean(axis=0)  # (N, n_classes)
        
        # Epistemic uncertainty (std across MC samples)
        epistemic_std = all_probs.std(axis=0)  # (N, n_classes)
        uncertainty = epistemic_std.mean(axis=1)  # (N,) — average across classes
        
        # Predictions
        predictions = mean_probs.argmax(axis=1)
        
        # ── "Requires Manual Review" routing
        top2 = np.sort(mean_probs, axis=1)[:, -2:]  # Top-2 probabilities
        top2_diff = top2[:, 1] - top2[:, 0]          # Difference between top 2
        
        needs_review = (
            (top2_diff < 0.15) |       # Top-2 classes too close
            (uncertainty > 0.1) |       # High epistemic uncertainty
            (mean_probs.max(axis=1) < 0.5)  # No confident prediction
        )
        
        result = {
            'probabilities': mean_probs,
            'predictions': predictions,
            'class_names': np.array([CLASS_NAMES[p] for p in predictions]),
            'uncertainty': uncertainty,
            'needs_review': needs_review,
            'mc_probs_all': all_probs,  # Full MC samples for analysis
        }
        
        # ── Counterfactual importance (if requested)
        if return_details:
            result['counterfactual'] = self._compute_counterfactual_importance(
                global_views, local_views, scalar_df, predictions)
        
        n_review = needs_review.sum()
        print(f"\n[CLASSIFIER] Predictions: {len(predictions)} targets")
        print(f"  Class distribution: {dict(Counter(predictions))}")
        readable = {CLASS_NAMES[k]: v for k, v in Counter(predictions).items()}
        print(f"  → {readable}")
        print(f"  Needs manual review: {n_review} ({100*n_review/len(predictions):.1f}%)")
        print(f"  Mean uncertainty: {uncertainty.mean():.4f}")
        
        return result
    
    # ── Counterfactual Importance ────────────────────────────────
    
    def _compute_counterfactual_importance(self, global_views, local_views, 
                                           scalar_df, original_preds):
        """
        For each prediction, ablate feature groups and see if the
        verdict changes. Reports which evidence is "load-bearing".
        
        Feature groups:
          1. Centroid info (centroid_shift_pix, centroid_p_col, centroid_blend_flag)
          2. Odd/even info (odd_even_sigma, odd_even_flag)
          3. Secondary eclipse (secondary_sigma, secondary_flag)
          4. Stellar params (stellar_rad, stellar_teff, stellar_logg)
          5. Crowding (crowdsap, flfrcsap, crowdsap_flag)
        """
        feature_groups = {
            'centroid': ['centroid_shift_pix', 'centroid_p_col', 'centroid_blend_flag'],
            'odd_even': ['odd_even_sigma', 'odd_even_flag'],
            'secondary': ['secondary_sigma', 'secondary_flag'],
            'stellar': ['stellar_rad', 'stellar_teff', 'stellar_logg'],
            'crowding': ['crowdsap', 'flfrcsap', 'crowdsap_flag'],
        }
        
        counterfactual = {}
        
        for group_name, cols in feature_groups.items():
            # Replace feature group with realistic baselines
            ablated_df = scalar_df.copy()
            for col in cols:
                if col in ablated_df.columns:
                    if col in ['crowdsap', 'flfrcsap', 'centroid_p_col', 'stellar_rad']:
                        ablated_df[col] = 1.0
                    elif col == 'stellar_teff':
                        ablated_df[col] = 5500.0
                    elif col == 'stellar_logg':
                        ablated_df[col] = 4.4
                    else:
                        ablated_df[col] = 0.0
            
            # Re-predict with ablated features
            leaves = self.get_lgbm_leaf_embeddings(ablated_df)
            gv = torch.FloatTensor(global_views[:, np.newaxis, :]).to(DEVICE)
            lv = torch.FloatTensor(local_views[:, np.newaxis, :]).to(DEVICE)
            lf = torch.FloatTensor(leaves).to(DEVICE)
            
            self.fusion_model.eval()
            with torch.no_grad():
                logits = self.fusion_model(gv, lv, lf)
                scaled = self.temp_scaler(logits.cpu())
                ablated_preds = F.softmax(scaled, dim=1).numpy().argmax(axis=1)
            
            # Record which predictions changed
            changed = ablated_preds != original_preds
            counterfactual[group_name] = {
                'n_changed': int(changed.sum()),
                'changed_indices': np.where(changed)[0].tolist(),
                'ablated_preds': ablated_preds,
            }
        
        return counterfactual
    
    # ── Evaluation ───────────────────────────────────────────────
    
    def evaluate(self, splits, split_name='test'):
        """Evaluate on a specific split."""
        data = splits[split_name]
        result = self.predict(
            data['global_views'], data['local_views'], data['scalars'],
            return_details=False
        )
        
        print(f"\n{'='*50}")
        print(f"  Evaluation on {split_name} set ({len(data['labels'])} samples)")
        print(f"{'='*50}")
        
        acc = accuracy_score(data['labels'], result['predictions'])
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(data['labels'], result['predictions'],
                                     target_names=CLASS_NAMES, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(data['labels'], result['predictions'])
        print("  Confusion Matrix:")
        print(f"  {'':12s}" + "".join(f"{c:>8s}" for c in CLASS_NAMES))
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i]:12s}" + "".join(f"{v:8d}" for v in row))
        
        return acc, result
    
    # ── Save / Load ──────────────────────────────────────────────
    
    def _save_all(self):
        """Save all model artifacts."""
        torch.save(self.fusion_model.state_dict(),
                  os.path.join(MODEL_DIR, 'fusion_best.pth'))
        torch.save(self.temp_scaler.state_dict(),
                  os.path.join(MODEL_DIR, 'temp_scaler.pth'))
        joblib.dump(self.lgbm_model, os.path.join(MODEL_DIR, 'lgbm_stream_b.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scalar_scaler.pkl'))
        
        # Save config
        config = {
            'n_classes': self.n_classes,
            'lgbm_n_trees': self.lgbm_n_trees,
            'class_names': CLASS_NAMES,
            'scalar_feature_cols': SCALAR_FEATURE_COLS,
            'train_history': self.train_history,
        }
        with open(os.path.join(MODEL_DIR, 'classifier_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[CLASSIFIER] All models saved to {MODEL_DIR}/")
    
    def load(self):
        """Load all model artifacts."""
        # Load config
        with open(os.path.join(MODEL_DIR, 'classifier_config.json')) as f:
            config = json.load(f)
        self.lgbm_n_trees = config['lgbm_n_trees']
        
        # Load LightGBM
        self.lgbm_model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_stream_b.pkl'))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scalar_scaler.pkl'))
        
        # Load fusion model
        self.fusion_model = FusionClassifier(
            cnn_dim=64, lgbm_dim=self.lgbm_n_trees, n_classes=self.n_classes
        ).to(DEVICE)
        self.fusion_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, 'fusion_best.pth'),
                       map_location=DEVICE, weights_only=True))
        
        # Load temperature scaler
        self.temp_scaler = TemperatureScaler()
        self.temp_scaler.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, 'temp_scaler.pth'),
                       map_location=DEVICE, weights_only=True))
        
        return self


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from ml_data_loader import load_splits, ML_SPLITS_DIR
    
    splits_exist = os.path.exists(os.path.join(ML_SPLITS_DIR, 'train.npz'))
    
    if splits_exist:
        print("[CLASSIFIER] Loading pre-built splits...")
        splits = load_splits()
        
        trainer = DualStreamTrainer(
            cnn_epochs=50,
            cnn_patience=7,
            batch_size=64,
        )
        
        best_acc = trainer.train(splits)
        print(f"\n[CLASSIFIER] Best validation accuracy: {best_acc:.4f}")
        
        # Evaluate on test set
        test_acc, test_result = trainer.evaluate(splits, 'test')
        print(f"\n[CLASSIFIER] Test accuracy: {test_acc:.4f}")
    else:
        print("[CLASSIFIER] No splits found. Run ml_data_loader.py first.")
