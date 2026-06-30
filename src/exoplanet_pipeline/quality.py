from __future__ import annotations

import numpy as np

# TESS quality flags are bit-coded. These presets deliberately avoid hiding a
# magic integer inside preprocessing. Adjust after checking the exact data product.
# The exact flag meanings are documented by TESS/SPOC and are exposed by Lightkurve.
QUALITY_BITS = {
    1: "Attitude tweak",
    2: "Safe mode",
    4: "Coarse point",
    8: "Earth/Moon in camera",
    16: "Reaction wheel desaturation",
    32: "Cosmic ray in optimal aperture",
    64: "Manual exclude",
    128: "Discontinuity corrected",
    256: "Impulsive outlier",
    512: "Argabrightening",
    1024: "Cosmic ray in collateral data",
    2048: "Straylight",
    4096: "Straylight 2",
    8192: "Planet search exclude",
    16384: "Bad calibration exclude",
}

QUALITY_MASK_PRESETS = {
    "none": 0,
    # Minimal: remove only severe spacecraft/pointing/scattered-light events.
    "minimal": 2 | 4 | 8 | 64 | 2048 | 4096 | 8192 | 16384,
    # Conservative: also remove strong outlier/discontinuity/collateral flags.
    "conservative": 2 | 4 | 8 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384,
    # Strict: remove every known nonzero flag in the dictionary.
    "strict": sum(QUALITY_BITS.keys()),
}


def quality_mask_from_flags(quality: np.ndarray | None, mode: str = "conservative") -> np.ndarray:
    """Return True for cadences kept by the selected quality mask preset."""
    if quality is None:
        return np.array([], dtype=bool)
    quality = np.asarray(quality)
    if mode not in QUALITY_MASK_PRESETS:
        raise ValueError(f"Unknown quality mask mode: {mode}. Choose from {list(QUALITY_MASK_PRESETS)}")
    mask_value = QUALITY_MASK_PRESETS[mode]
    if mask_value == 0:
        return np.ones_like(quality, dtype=bool)
    return (quality.astype(int) & mask_value) == 0


def summarize_quality_flags(quality: np.ndarray | None) -> dict[str, int]:
    """Count how often each known quality bit is set."""
    if quality is None:
        return {}
    q = np.asarray(quality).astype(int)
    out = {}
    for bit, name in QUALITY_BITS.items():
        count = int(np.sum((q & bit) != 0))
        if count:
            out[f"{bit}:{name}"] = count
    return out
