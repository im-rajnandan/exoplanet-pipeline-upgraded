from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from .cnn_views import (
    CNN_SCALAR_FEATURE_COLUMNS,
    GLOBAL_VIEW_BINS,
    LOCAL_VIEW_BINS,
    LOCAL_VIEW_NAMES,
    CNNCandidateViews,
)
from .classification_policy import CANONICAL_CLASSES, finalize_probabilities, renormalize_probs as _renormalize_probs

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass
    F = None



PLANET_CLASS = "PLANETARY_TRANSIT_CANDIDATE"
FALSE_POSITIVE_BINARY_LABEL = "false_positive_or_other"
PLANET_BINARY_LABEL = "planet_like"


@dataclass
class CNNModelConfig:
    """Serializable architecture and feature contract for the CNN vetter."""

    canonical_classes: list[str] = field(default_factory=lambda: list(CANONICAL_CLASSES))
    scalar_feature_names: list[str] = field(default_factory=lambda: list(CNN_SCALAR_FEATURE_COLUMNS))
    local_view_names: list[str] = field(default_factory=lambda: list(LOCAL_VIEW_NAMES))
    global_bins: int = GLOBAL_VIEW_BINS
    local_bins: int = LOCAL_VIEW_BINS
    input_channels: int = 2
    conv_channels: int = 32
    view_embedding_dim: int = 64
    scalar_embedding_dim: int = 32
    fusion_hidden_dim: int = 96
    dropout: float = 0.10
    model_version: str = "cnn_v1"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CNNModelConfig":
        if not data:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)


@dataclass
class CNNTrainingExample:
    global_flux: np.ndarray
    local_views: np.ndarray
    scalar_features: np.ndarray
    scalar_feature_names: list[str]
    canonical_label: str | None = None
    binary_label: str | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CNNTrainingResult:
    model: Any
    config: CNNModelConfig
    scalar_scaler: dict[str, Any]
    label_map: dict[str, Any]
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def bundle(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "config": self.config,
            "scalar_scaler": self.scalar_scaler,
            "label_map": self.label_map,
            "metrics": self.metrics,
        }


def _require_torch():
    if not HAS_TORCH:
        raise ImportError('CNN vetting requires PyTorch. Install with: pip install -e ".[deep]"')
    import torch
    from torch import nn
    import torch.nn.functional as F
    return torch, nn, F


class _ConvBranch(nn.Module):
    def __init__(self, in_channels: int, channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels * 2),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _CandidateVetterCNN(nn.Module):
    def __init__(self, model_config: CNNModelConfig):
        super().__init__()
        self.config = model_config
        self.global_branch = _ConvBranch(
            model_config.input_channels,
            model_config.conv_channels,
            model_config.view_embedding_dim,
            model_config.dropout,
        )
        self.local_branch = _ConvBranch(
            model_config.input_channels,
            model_config.conv_channels,
            model_config.view_embedding_dim,
            model_config.dropout,
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(len(model_config.scalar_feature_names), model_config.scalar_embedding_dim),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.scalar_embedding_dim, model_config.scalar_embedding_dim),
            nn.ReLU(),
        )
        fusion_in = model_config.view_embedding_dim * 2 + model_config.scalar_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, model_config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.fusion_hidden_dim, model_config.fusion_hidden_dim // 2),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(model_config.fusion_hidden_dim // 2, len(model_config.canonical_classes))
        self.binary_head = nn.Linear(model_config.fusion_hidden_dim // 2, 1)

    def forward(self, global_flux, local_views, scalars):
        global_emb = self.global_branch(global_flux)
        batch, n_views, channels, n_bins = local_views.shape
        flat_local = local_views.reshape(batch * n_views, channels, n_bins)
        local_emb = self.local_branch(flat_local).reshape(batch, n_views, -1)
        view_valid = (local_views[:, :, 1, :].sum(dim=-1) > 0).float().unsqueeze(-1)
        denom = view_valid.sum(dim=1).clamp_min(1.0)
        local_emb = (local_emb * view_valid).sum(dim=1) / denom
        scalar_emb = self.scalar_branch(scalars)
        fused = self.fusion(torch.cat([global_emb, local_emb, scalar_emb], dim=1))
        return {
            "class_logits": self.class_head(fused),
            "binary_logit": self.binary_head(fused).squeeze(-1),
        }


def create_cnn_model(config: CNNModelConfig | dict[str, Any] | None = None):
    """Instantiate the optional PyTorch CNN model on demand."""
    _require_torch()
    cfg = CNNModelConfig.from_dict(config) if isinstance(config, dict) or config is None else config
    return _CandidateVetterCNN(cfg)


def fit_scalar_scaler(matrix: np.ndarray, feature_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("scalar matrix must be 2-dimensional")
    impute = np.nanmedian(values, axis=0)
    impute = np.where(np.isfinite(impute), impute, 0.0)
    filled = np.where(np.isfinite(values), values, impute)
    mean = np.nanmean(filled, axis=0)
    scale = np.nanstd(filled, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return {
        "feature_names": list(feature_names),
        "impute": impute.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def transform_scalars(vector_or_matrix: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    values = np.asarray(vector_or_matrix, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    impute = np.asarray(scaler.get("impute", np.zeros(values.shape[1])), dtype=np.float32)
    mean = np.asarray(scaler.get("mean", np.zeros(values.shape[1])), dtype=np.float32)
    scale = np.asarray(scaler.get("scale", np.ones(values.shape[1])), dtype=np.float32)
    values = np.where(np.isfinite(values), values, impute)
    out = (values - mean) / np.where(scale > 0, scale, 1.0)
    return out[0] if single else out


def training_example_from_candidate_views(
    example: CNNCandidateViews,
    *,
    canonical_label: str | None = None,
    binary_label: str | int | float | None = None,
    config: CNNModelConfig | None = None,
) -> CNNTrainingExample:
    cfg = config or CNNModelConfig()
    return CNNTrainingExample(
        global_flux=example.global_tensor(),
        local_views=example.local_tensor(),
        scalar_features=example.scalar_vector(cfg.scalar_feature_names),
        scalar_feature_names=list(cfg.scalar_feature_names),
        canonical_label=canonical_label,
        binary_label=binary_label,
        metadata=dict(example.metadata),
    )


def load_cnn_example_npz(path: str | Path, config: CNNModelConfig | None = None) -> CNNTrainingExample:
    cfg = config or CNNModelConfig()
    with np.load(path, allow_pickle=True) as data:
        scalar_features = np.asarray(data["scalar_features"], dtype=np.float32)
        names = [str(x) for x in data.get("scalar_feature_names", np.asarray(cfg.scalar_feature_names)).tolist()]
        scalar_features = _align_scalar_vector(scalar_features, names, cfg.scalar_feature_names)
        metadata: dict[str, Any] = {}
        if "metadata" in data:
            raw_meta = data["metadata"]
            try:
                metadata = dict(raw_meta.tolist()[0])
            except Exception:
                metadata = {}
        canonical_label = str(data["canonical_label"].tolist()) if "canonical_label" in data else None
        binary_label = str(data["binary_label"].tolist()) if "binary_label" in data else None
        return CNNTrainingExample(
            global_flux=np.asarray(data["global_flux"], dtype=np.float32),
            local_views=np.asarray(data["local_views"], dtype=np.float32),
            scalar_features=scalar_features.astype(np.float32),
            scalar_feature_names=list(cfg.scalar_feature_names),
            canonical_label=canonical_label,
            binary_label=binary_label,
            metadata=metadata,
        )


def load_cnn_examples(paths_or_dir: str | Path | Iterable[str | Path], config: CNNModelConfig | None = None) -> list[CNNTrainingExample]:
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        paths = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    else:
        paths = [Path(p) for p in paths_or_dir]
    return [load_cnn_example_npz(path, config=config) for path in paths]


def train_cnn_classifier(
    examples: list[CNNTrainingExample],
    *,
    config: CNNModelConfig | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.20,
    seed: int = 42,
    device: str = "cpu",
) -> CNNTrainingResult:
    """Train the V1 CNN vetter with masked canonical and binary losses."""
    torch, nn, F = _require_torch()
    if not examples:
        raise ValueError("No CNN examples provided.")
    cfg = config or CNNModelConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    label_map = {
        "class_to_idx": {cls: i for i, cls in enumerate(cfg.canonical_classes)},
        "idx_to_class": {str(i): cls for i, cls in enumerate(cfg.canonical_classes)},
        "binary_labels": {PLANET_BINARY_LABEL: 1, FALSE_POSITIVE_BINARY_LABEL: 0},
    }
    global_x = np.stack([ex.global_flux for ex in examples]).astype(np.float32)
    local_x = np.stack([ex.local_views for ex in examples]).astype(np.float32)
    scalar_x = np.stack([_align_scalar_vector(ex.scalar_features, ex.scalar_feature_names, cfg.scalar_feature_names) for ex in examples]).astype(np.float32)
    scaler = fit_scalar_scaler(scalar_x, cfg.scalar_feature_names)
    scalar_x = transform_scalars(scalar_x, scaler).astype(np.float32)
    y_class = np.asarray([label_map["class_to_idx"].get(str(ex.canonical_label), -100) if ex.canonical_label else -100 for ex in examples], dtype=np.int64)
    y_binary = np.asarray([_binary_label_value(ex.binary_label, ex.canonical_label) for ex in examples], dtype=np.float32)
    if np.all(y_class == -100) and np.all(~np.isfinite(y_binary)):
        raise ValueError("At least one canonical_label or binary_label is required to train the CNN classifier.")

    train_idx, val_idx = _group_train_val_split(examples, validation_fraction, rng)
    if val_idx.size == 0:
        val_idx = train_idx.copy()

    model = create_cnn_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ignore_index = ignore_index

        def forward(self, inputs, targets):
            ce_loss_val = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
            pt = torch.exp(-ce_loss_val)
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss_val
            if self.alpha is not None:
                alpha_t = self.alpha[targets[valid_mask]]
                focal_loss = alpha_t * focal_loss[valid_mask]
                return focal_loss.mean()
            else:
                return focal_loss[valid_mask].mean()

    valid_y = y_class[y_class != -100]
    if len(valid_y) > 0:
        counts = np.bincount(valid_y, minlength=len(cfg.canonical_classes))
        total = len(valid_y)
        alpha_weights = [total / max(counts[i], 1) for i in range(len(cfg.canonical_classes))]
        alpha_sum = sum(alpha_weights)
        alpha_weights = [w / alpha_sum * len(cfg.canonical_classes) for w in alpha_weights]
        alpha_tensor = torch.FloatTensor(alpha_weights).to(device)
    else:
        alpha_tensor = None

    ce_loss = FocalLoss(alpha=alpha_tensor, gamma=2.0, ignore_index=-100)

    warnings: list[str] = []
    if np.all(y_class == -100):
        warnings.append("No canonical labels provided; training used binary labels only.")
    if np.all(~np.isfinite(y_binary)):
        warnings.append("No binary labels provided; training used canonical labels only.")

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_idx)
        losses: list[float] = []
        for batch_idx in _batch_indices(train_idx, batch_size):
            tensors = _make_batch_tensors(global_x, local_x, scalar_x, y_class, y_binary, batch_idx, torch, device, augment=True)
            optimizer.zero_grad()
            outputs = model(tensors["global"], tensors["local"], tensors["scalars"])
            loss = _masked_total_loss(outputs, tensors["class"], tensors["binary"], ce_loss, F)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics = _evaluate_model(model, global_x, local_x, scalar_x, y_class, y_binary, val_idx, torch, device)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        history.append(metrics)

    # Temperature Scaling Calibration
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            return logits / self.temperature

        def calibrate(self, logits, labels):
            opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
            crit = nn.CrossEntropyLoss()
            mask = labels != -100
            if mask.sum() == 0:
                return self
            val_logits = logits[mask]
            val_labels = labels[mask]
            def eval_loss():
                opt.zero_grad()
                loss = crit(self.forward(val_logits), val_labels)
                loss.backward()
                return loss
            opt.step(eval_loss)
            return self

    model.eval()
    with torch.no_grad():
        val_tensors = _make_batch_tensors(global_x, local_x, scalar_x, y_class, y_binary, val_idx, torch, device)
        val_outputs = model(val_tensors["global"], val_tensors["local"], val_tensors["scalars"])
        val_class_logits = val_outputs["class_logits"]
        val_labels = val_tensors["class"]

    temp_scaler = TemperatureScaler().to(device)
    try:
        temp_scaler.calibrate(val_class_logits, val_labels)
        temp_val = float(temp_scaler.temperature.item())
    except Exception:
        temp_val = 1.0

    metrics = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "n_examples": int(len(examples)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "history": history,
        "final": history[-1] if history else {},
        "temperature": temp_val,
    }
    return CNNTrainingResult(model=model, config=cfg, scalar_scaler=scaler, label_map=label_map, metrics=metrics, warnings=warnings)


def save_cnn_bundle(
    model: Any,
    output_dir: str | Path,
    config: CNNModelConfig,
    scalar_scaler: dict[str, Any],
    label_map: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save the CNN bundle as state_dict plus JSON sidecars."""
    torch, _, _ = _require_torch()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_map = label_map or {
        "class_to_idx": {cls: i for i, cls in enumerate(config.canonical_classes)},
        "idx_to_class": {str(i): cls for i, cls in enumerate(config.canonical_classes)},
        "binary_labels": {PLANET_BINARY_LABEL: 1, FALSE_POSITIVE_BINARY_LABEL: 0},
    }
    metrics = metrics or {}
    paths = {
        "model": output / "cnn_model.pt",
        "config": output / "cnn_config.json",
        "scalar_scaler": output / "cnn_scalar_scaler.json",
        "label_map": output / "cnn_label_map.json",
        "metrics": output / "cnn_metrics.json",
    }
    torch.save(model.state_dict(), paths["model"])
    _write_json(paths["config"], asdict(config))
    _write_json(paths["scalar_scaler"], scalar_scaler)
    _write_json(paths["label_map"], label_map)
    _write_json(paths["metrics"], metrics)
    return paths


def load_cnn_bundle(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a V1 CNN bundle directory or direct ``cnn_model.pt`` path."""
    torch, _, _ = _require_torch()
    p = Path(path)
    bundle_dir = p.parent if p.is_file() else p
    config = CNNModelConfig.from_dict(_read_json(bundle_dir / "cnn_config.json"))
    model = create_cnn_model(config)
    state = torch.load(bundle_dir / "cnn_model.pt", map_location=map_location)
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    return {
        "model": model,
        "config": config,
        "scalar_scaler": _read_json(bundle_dir / "cnn_scalar_scaler.json"),
        "label_map": _read_json(bundle_dir / "cnn_label_map.json"),
        "metrics": _read_json(bundle_dir / "cnn_metrics.json"),
        "bundle_dir": str(bundle_dir),
    }


def mc_dropout_predict(
    model: Any,
    global_tensor: torch.Tensor,
    local_tensor: torch.Tensor,
    scalar_tensor: torch.Tensor,
    temperature: float = 1.0,
    mc_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run MC Dropout predictions on a single target to get mean probabilities, class stds, and epistemic uncertainty."""
    torch, _, F = _require_torch()
    model.train()  # Keep dropout active!

    mc_probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            outputs = model(global_tensor, local_tensor, scalar_tensor)
            logits = outputs["class_logits"] / temperature
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            mc_probs.append(probs)

    mc_probs = np.array(mc_probs)  # (mc_samples, n_classes)
    mean_probs = mc_probs.mean(axis=0)
    std_probs = mc_probs.std(axis=0)

    pred_idx = mean_probs.argmax()
    epistemic_uncertainty = float(std_probs[pred_idx])

    model.eval()  # restore eval mode
    return mean_probs, std_probs, epistemic_uncertainty


def predict_cnn_candidate_views(
    cnn_bundle: dict[str, Any] | str | Path,
    example: CNNCandidateViews | CNNTrainingExample,
    *,
    catalog_row: pd.Series | dict[str, Any] | None = None,
    apply_physical_guardrails: bool = True,
    device: str | None = None,
    mc_samples: int = 50,
) -> dict[str, Any]:
    """Return public CNN and final-classifier columns for one candidate."""
    torch, _, F = _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_cnn_bundle(cnn_bundle, map_location=device) if isinstance(cnn_bundle, (str, Path)) else cnn_bundle
    model = bundle["model"]
    if hasattr(model, "to"):
        model.to(device)
    config = bundle.get("config") or CNNModelConfig()
    if isinstance(config, dict):
        config = CNNModelConfig.from_dict(config)
    scaler = bundle.get("scalar_scaler") or fit_scalar_scaler(np.zeros((1, len(config.scalar_feature_names))), config.scalar_feature_names)

    global_x, local_x, scalar_x = _example_arrays(example, config)
    scalar_x = transform_scalars(scalar_x, scaler)
    
    temperature = bundle.get("metrics", {}).get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        temperature = 1.0

    global_tensor = torch.as_tensor(global_x[None, :, :], dtype=torch.float32, device=device)
    local_tensor = torch.as_tensor(local_x[None, :, :, :], dtype=torch.float32, device=device)
    scalar_tensor = torch.as_tensor(scalar_x[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(global_tensor, local_tensor, scalar_tensor)
        logits = outputs["class_logits"] / temperature
        probs_arr = F.softmax(logits, dim=1).cpu().numpy()[0]
        binary_prob = float(torch.sigmoid(outputs["binary_logit"]).cpu().numpy()[0])

    # Run MC Dropout
    epistemic_unc = 0.0
    if mc_samples > 0:
        try:
            _, _, epistemic_unc = mc_dropout_predict(
                model, global_tensor, local_tensor, scalar_tensor,
                temperature=temperature, mc_samples=mc_samples
            )
        except Exception:
            epistemic_unc = 0.0

    probs = {cls: float(probs_arr[i]) for i, cls in enumerate(config.canonical_classes)}
    for cls in CANONICAL_CLASSES:
        probs.setdefault(cls, 0.0)
    probs = _renormalize_probs(probs)
    cnn_pred = max(probs, key=probs.get)
    cnn_conf = float(probs[cnn_pred])

    final_probs, final_pred, final_conf, warnings_here = finalize_probabilities(
        probs,
        row=catalog_row,
        apply_guardrails=apply_physical_guardrails and catalog_row is not None,
        low_margin_warning="low_cnn_margin_downgraded_to_uncertain",
    )
    method = "cnn_plus_physical_guardrails" if apply_physical_guardrails and catalog_row is not None else "cnn"

    out: dict[str, Any] = {
        "cnn_predicted_class": cnn_pred,
        "cnn_confidence": cnn_conf,
        "cnn_binary_planet_probability": binary_prob,
        "cnn_model_version": config.model_version,
        "cnn_epistemic_uncertainty": float(epistemic_unc),
        "final_predicted_class": final_pred,
        "final_confidence": final_conf,
        "final_classifier_method": method,
        "final_classifier_warnings": ";".join(warnings_here),
    }
    for cls in CANONICAL_CLASSES:
        out[f"cnn_prob_{cls}"] = probs.get(cls, 0.0)
        out[f"final_prob_{cls}"] = final_probs.get(cls, 0.0)
    return out


def predict_cnn_examples(cnn_bundle: dict[str, Any] | str | Path, examples: list[CNNTrainingExample], *, device: str = "cpu") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        row = dict(ex.metadata)
        row.update(predict_cnn_candidate_views(cnn_bundle, ex, catalog_row=ex.metadata, device=device))
        rows.append(row)
    return pd.DataFrame(rows)


def _example_arrays(example: CNNCandidateViews | CNNTrainingExample, config: CNNModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(example, CNNCandidateViews):
        return (
            example.global_tensor().astype(np.float32),
            example.local_tensor().astype(np.float32),
            example.scalar_vector(config.scalar_feature_names).astype(np.float32),
        )
    return (
        np.asarray(example.global_flux, dtype=np.float32),
        np.asarray(example.local_views, dtype=np.float32),
        _align_scalar_vector(example.scalar_features, example.scalar_feature_names, config.scalar_feature_names).astype(np.float32),
    )


def _align_scalar_vector(values: np.ndarray, source_names: list[str], target_names: list[str]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    mapping = {name: arr[i] for i, name in enumerate(source_names) if i < len(arr)}
    return np.asarray([mapping.get(name, np.nan) for name in target_names], dtype=np.float32)


def _binary_label_value(binary_label: str | int | float | None, canonical_label: str | None = None) -> float:
    if binary_label is None or (isinstance(binary_label, float) and not np.isfinite(binary_label)):
        if canonical_label is None:
            return np.nan
        return 1.0 if str(canonical_label) == PLANET_CLASS else 0.0
    raw = str(binary_label).strip().lower()
    if raw in {"1", "true", "planet", "planet_like", "pc", "candidate"}:
        return 1.0
    if raw in {"0", "false", "false_positive", "false_positive_or_other", "fp", "other"}:
        return 0.0
    return np.nan


def _group_train_val_split(examples: list[CNNTrainingExample], validation_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(examples)
    idx = np.arange(n)
    if n < 3 or validation_fraction <= 0:
        return idx, np.asarray([], dtype=int)
    groups = np.asarray([str(ex.metadata.get("tic_id", f"row_{i}")) for i, ex in enumerate(examples)])
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_val_groups = max(1, int(round(len(unique) * validation_fraction))) if len(unique) > 1 else 0
    val_groups = set(unique[:n_val_groups])
    val = idx[np.asarray([g in val_groups for g in groups])]
    train = idx[np.asarray([g not in val_groups for g in groups])]
    if train.size == 0 or val.size == 0:
        rng.shuffle(idx)
        n_val = max(1, int(round(n * validation_fraction)))
        val = idx[:n_val]
        train = idx[n_val:]
    return train.astype(int), val.astype(int)


def _batch_indices(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), max(1, int(batch_size))):
        yield indices[start : start + max(1, int(batch_size))]


def augment_light_curve_numpy(flux, phase_shift_bins=4, noise_sigma=0.0005, scale_pct=0.05):
    augmented = flux.copy()
    if phase_shift_bins > 0:
        shift = np.random.randint(-phase_shift_bins, phase_shift_bins + 1)
        augmented = np.roll(augmented, shift)
    if noise_sigma > 0:
        noise = np.random.normal(0, np.random.uniform(0.1, 1.0) * noise_sigma, len(flux))
        augmented += noise
    if scale_pct > 0:
        scale = 1.0 + np.random.uniform(-scale_pct, scale_pct)
        augmented = (augmented - 1.0) * scale + 1.0
    return augmented


def _make_batch_tensors(global_x, local_x, scalar_x, y_class, y_binary, batch_idx, torch, device, augment=False):
    g_batch = global_x[batch_idx].copy()
    l_batch = local_x[batch_idx].copy()

    if augment:
        for i in range(len(batch_idx)):
            g_batch[i, 0] = augment_light_curve_numpy(g_batch[i, 0], phase_shift_bins=4, noise_sigma=0.0005, scale_pct=0.05)
            for view_idx in range(l_batch.shape[1]):
                l_batch[i, view_idx, 0] = augment_light_curve_numpy(l_batch[i, view_idx, 0], phase_shift_bins=2, noise_sigma=0.0005, scale_pct=0.05)

    return {
        "global": torch.as_tensor(g_batch, dtype=torch.float32, device=device),
        "local": torch.as_tensor(l_batch, dtype=torch.float32, device=device),
        "scalars": torch.as_tensor(scalar_x[batch_idx], dtype=torch.float32, device=device),
        "class": torch.as_tensor(y_class[batch_idx], dtype=torch.long, device=device),
        "binary": torch.as_tensor(y_binary[batch_idx], dtype=torch.float32, device=device),
    }


def _masked_total_loss(outputs, y_class, y_binary, ce_loss, F) -> Any:
    loss = ce_loss(outputs["class_logits"], y_class)
    binary_mask = torch_isfinite(y_binary)
    if binary_mask.any():
        binary_loss = F.binary_cross_entropy_with_logits(outputs["binary_logit"][binary_mask], y_binary[binary_mask])
        if torch_isfinite(loss):
            loss = loss + 0.5 * binary_loss
        else:
            loss = 0.5 * binary_loss
    return loss


def torch_isfinite(x):
    import torch

    return torch.isfinite(x)


def _evaluate_model(model, global_x, local_x, scalar_x, y_class, y_binary, indices, torch, device) -> dict[str, float]:
    if len(indices) == 0:
        return {"val_accuracy": float("nan"), "val_binary_accuracy": float("nan")}
    model.eval()
    with torch.no_grad():
        tensors = _make_batch_tensors(global_x, local_x, scalar_x, y_class, y_binary, indices, torch, device)
        outputs = model(tensors["global"], tensors["local"], tensors["scalars"])
        pred = outputs["class_logits"].argmax(dim=1)
        class_mask = tensors["class"] >= 0
        if class_mask.any():
            val_acc = float((pred[class_mask] == tensors["class"][class_mask]).float().mean().cpu().item())
        else:
            val_acc = float("nan")
        bin_mask = torch.isfinite(tensors["binary"])
        if bin_mask.any():
            bin_pred = (torch.sigmoid(outputs["binary_logit"][bin_mask]) >= 0.5).float()
            bin_acc = float((bin_pred == tensors["binary"][bin_mask]).float().mean().cpu().item())
        else:
            bin_acc = float("nan")
    model.train()
    return {"val_accuracy": val_acc, "val_binary_accuracy": bin_acc}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
