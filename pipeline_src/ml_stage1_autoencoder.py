"""
================================================================
 ML STAGE 1 — AUTOENCODER ANOMALY PRE-FILTER (Unsupervised)
================================================================
 1D convolutional autoencoder trained ONLY on "boring" non-varying
 stars (SDE < 6). High reconstruction error → anomalous → passed
 to Stage 2 classifier.

 Scientific motivation: catches non-box-shaped signals (single
 transits, grazing geometries, starspot modulations) that TLS's
 box model might underweight.
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import os
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                       'mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_DIR = './tess_pipeline_output/ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# AUTOENCODER ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class Conv1DAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for light curve anomaly detection.
    
    Input: (batch, 1, 201) — global phase-folded view
    Latent: 32-dimensional embedding
    Output: (batch, 1, 201) — reconstructed view
    """
    
    def __init__(self, input_length=201, latent_dim=32):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # ── Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Calculate encoded spatial size
        self._encoded_length = self._get_encoded_length(input_length)
        self._flat_size = 64 * self._encoded_length
        
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # ── Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self._flat_size),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
        
        # Use F.interpolate instead of AdaptiveAvgPool1d for MPS compatibility
        self.input_length = input_length
    
    def _get_encoded_length(self, L):
        """Calculate spatial dim after encoder convolutions."""
        L = (L + 2 * 3 - 7) // 2 + 1   # Conv1 with k=7, s=2, p=3
        L = (L + 2 * 2 - 5) // 2 + 1   # Conv2 with k=5, s=2, p=2
        L = (L + 2 * 1 - 3) // 2 + 1   # Conv3 with k=3, s=2, p=1
        return L
    
    def encode(self, x):
        """Encode input to latent representation."""
        h = self.encoder(x)
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(-1, 64, self._encoded_length)
        out = self.decoder(h)
        out = F.interpolate(out, size=self.input_length, mode='linear', align_corners=False)
        return out
    
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class AutoencoderTrainer:
    """
    Trains the autoencoder on "boring" (non-significant) light curves
    and uses reconstruction error as an anomaly score.
    """
    
    def __init__(self, input_length=201, latent_dim=32, lr=1e-3, 
                 batch_size=64, epochs=100, patience=10):
        self.model = Conv1DAutoencoder(input_length, latent_dim).to(DEVICE)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold = None  # Set after training
        self.train_losses = []
        self.val_losses = []
    
    def _prepare_data(self, global_views, sde_values=None, sde_threshold=6.0):
        """
        Select "boring" population for training.
        If sde_values provided, filter to SDE < sde_threshold.
        """
        if sde_values is not None:
            # Only train on non-significant detections
            boring_mask = sde_values < sde_threshold
            train_data = global_views[boring_mask]
            print(f"[AUTOENCODER] Training on {len(train_data)} boring curves "
                  f"(SDE < {sde_threshold}) out of {len(global_views)} total")
        else:
            train_data = global_views
            print(f"[AUTOENCODER] Training on all {len(train_data)} curves")
        
        # Normalize to [0, 1] range per sample
        mins = train_data.min(axis=1, keepdims=True)
        maxs = train_data.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        train_data = (train_data - mins) / ranges
        
        # Add channel dimension: (N, 201) → (N, 1, 201)
        train_data = train_data[:, np.newaxis, :]
        
        # Split 90/10 for train/val within boring population
        n = len(train_data)
        n_val = max(int(n * 0.1), 1)
        perm = np.random.permutation(n)
        val_data = train_data[perm[:n_val]]
        train_data = train_data[perm[n_val:]]
        
        return train_data, val_data
    
    def train(self, global_views, sde_values=None, sde_threshold=6.0):
        """Train the autoencoder."""
        print("\n" + "=" * 60)
        print("  STAGE 1: Training Autoencoder Anomaly Pre-filter")
        print(f"  Device: {DEVICE}")
        print("=" * 60)
        
        train_data, val_data = self._prepare_data(global_views, sde_values, sde_threshold)
        
        train_tensor = torch.FloatTensor(train_data).to(DEVICE)
        val_tensor = torch.FloatTensor(val_data).to(DEVICE)
        
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                  shuffle=True, drop_last=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # ── Train
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(train_tensor)
            
            # ── Validate
            self.model.eval()
            with torch.no_grad():
                recon_val, _ = self.model(val_tensor)
                val_loss = criterion(recon_val, val_tensor).item()
            
            scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          os.path.join(MODEL_DIR, 'autoencoder_best.pth'))
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f} | Patience: {patience_counter}/{self.patience}")
            
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, 'autoencoder_best.pth'),
                       map_location=DEVICE, weights_only=True))
        
        # ── Set anomaly threshold (95th percentile of training errors)
        self.model.eval()
        with torch.no_grad():
            recon_train, _ = self.model(train_tensor)
            train_errors = torch.mean((recon_train - train_tensor) ** 2, 
                                       dim=(1, 2)).cpu().numpy()
        
        self.threshold = np.percentile(train_errors, 95)
        print(f"\n  Anomaly threshold (95th pctile): {self.threshold:.6f}")
        print(f"  Training complete. Best val loss: {best_val_loss:.6f}")
        
        # Save threshold
        config = {
            'threshold': float(self.threshold),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1,
            'n_train': len(train_tensor),
            'n_val': len(val_tensor),
        }
        with open(os.path.join(MODEL_DIR, 'autoencoder_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        return self
    
    def predict_anomaly_scores(self, global_views):
        """
        Compute reconstruction error (anomaly score) for each sample.
        
        Returns:
            scores: (N,) array of reconstruction MSE
            is_anomalous: (N,) boolean array
        """
        self.model.eval()
        
        # Normalize
        data = global_views.copy()
        mins = data.min(axis=1, keepdims=True)
        maxs = data.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        data = (data - mins) / ranges
        data = data[:, np.newaxis, :]  # Add channel dim
        
        tensor = torch.FloatTensor(data).to(DEVICE)
        
        with torch.no_grad():
            recon, latent = self.model(tensor)
            scores = torch.mean((recon - tensor) ** 2, dim=(1, 2)).cpu().numpy()
        
        is_anomalous = scores > self.threshold
        
        n_anom = is_anomalous.sum()
        print(f"[AUTOENCODER] Anomaly detection: {n_anom}/{len(scores)} "
              f"({100*n_anom/len(scores):.1f}%) flagged as anomalous")
        
        return scores, is_anomalous
    
    def get_latent_embeddings(self, global_views):
        """Extract latent embeddings for downstream use."""
        self.model.eval()
        
        data = global_views.copy()
        mins = data.min(axis=1, keepdims=True)
        maxs = data.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        data = (data - mins) / ranges
        data = data[:, np.newaxis, :]
        
        tensor = torch.FloatTensor(data).to(DEVICE)
        
        with torch.no_grad():
            z = self.model.encode(tensor).cpu().numpy()
        
        return z
    
    def save(self, path=None):
        """Save the trained model."""
        if path is None:
            path = os.path.join(MODEL_DIR, 'autoencoder_best.pth')
        torch.save(self.model.state_dict(), path)
    
    def load(self, path=None):
        """Load a trained model."""
        if path is None:
            path = os.path.join(MODEL_DIR, 'autoencoder_best.pth')
        self.model.load_state_dict(
            torch.load(path, map_location=DEVICE, weights_only=True))
        
        config_path = os.path.join(MODEL_DIR, 'autoencoder_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.threshold = config['threshold']
        
        return self


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ML_FEATURES_DIR = './tess_pipeline_output/ml_features'
    
    # Load features
    gv_path = os.path.join(ML_FEATURES_DIR, 'global_views.npy')
    sc_path = os.path.join(ML_FEATURES_DIR, 'scalar_features.csv')
    
    if os.path.exists(gv_path):
        import pandas as pd
        global_views = np.load(gv_path)
        scalars = pd.read_csv(sc_path)
        sde_values = scalars['SDE'].values if 'SDE' in scalars.columns else None
        
        trainer = AutoencoderTrainer(
            input_length=global_views.shape[1],
            latent_dim=32,
            epochs=100,
            patience=10,
        )
        trainer.train(global_views, sde_values)
        scores, is_anom = trainer.predict_anomaly_scores(global_views)
        
        print(f"\nAnomaly score stats:")
        print(f"  Mean:   {scores.mean():.6f}")
        print(f"  Median: {np.median(scores):.6f}")
        print(f"  Max:    {scores.max():.6f}")
        print(f"  Threshold: {trainer.threshold:.6f}")
    else:
        print("[AUTOENCODER] No features found. Run ml_feature_builder.py first.")
        print("  Running with synthetic demo data instead...")
        
        # Demo with random data
        demo_data = np.random.randn(200, 201).astype(np.float32) * 0.01 + 1.0
        trainer = AutoencoderTrainer(input_length=201, epochs=30, patience=5)
        trainer.train(demo_data)
        scores, is_anom = trainer.predict_anomaly_scores(demo_data)
