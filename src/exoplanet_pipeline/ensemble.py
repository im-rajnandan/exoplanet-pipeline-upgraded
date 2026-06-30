"""
================================================================
 exoplanet_pipeline/ensemble.py
 Dual-stream ensemble classifier and LightGBM-CNN fusion.
================================================================
"""

import numpy as np
import pandas as pd
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


if HAS_TORCH:
    class CNNStreamA(nn.Module):
        """
        CNN Stream A extracting latent features from global and local views.
        """
        def __init__(self, cnn_dim=64):
            super().__init__()
            # Define global and local feature encoders using standard 1D CNN branches
            self.global_branch = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(32 * 50, cnn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            
            self.local_branch = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(32 * 10, cnn_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            
            self.fc = nn.Sequential(
                nn.Linear(cnn_dim + cnn_dim // 2, cnn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )

        def forward(self, global_view, local_view):
            g_emb = self.global_branch(global_view)
            l_emb = self.local_branch(local_view)
            combined = torch.cat([g_emb, l_emb], dim=1)
            return self.fc(combined)


    class FusionClassifier(nn.Module):
        """
        Combines CNN Stream A output with LightGBM leaf embeddings.
        
        Input: CNN_embedding(64) + LGBM_leaf_embedding(n_trees * emb_dim)
        Output: logits(4)
        """
        def __init__(self, cnn_dim=64, lgbm_dim=500, n_classes=4, max_leaves=100, emb_dim=2):
            super().__init__()
            self.cnn_stream = CNNStreamA(cnn_dim)
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
            return self.cnn_stream(global_view, local_view)
else:
    class FusionClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for FusionClassifier.")


def blend_predictions(cnn_probs: dict[str, float], ml_probs: dict[str, float], cnn_weight: float = 0.5) -> dict[str, float]:
    """
    Standard ensemble average of CNN and Tabular ML probabilities.
    """
    blended = {}
    classes = set(cnn_probs.keys()).union(ml_probs.keys())
    for cls in classes:
        p_cnn = cnn_probs.get(cls, 0.0)
        p_ml = ml_probs.get(cls, 0.0)
        blended[cls] = float(cnn_weight * p_cnn + (1.0 - cnn_weight) * p_ml)
    
    # Renormalize to ensure sum is exactly 1.0
    s = sum(blended.values())
    if s > 0:
        for cls in blended:
            blended[cls] /= s
    else:
        for cls in blended:
            blended[cls] = 1.0 / len(blended)
    return blended
