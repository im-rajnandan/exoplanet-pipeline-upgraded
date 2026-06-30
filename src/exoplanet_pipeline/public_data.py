from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .ml import CANONICAL_CLASSES, normalize_label


PLANET_CLASS = "PLANETARY_TRANSIT_CANDIDATE"
EB_CLASS = "ECLIPSING_BINARY"
SYSTEMATIC_CLASS = "INSTRUMENTAL_OR_LOW_QUALITY_SYSTEMATIC"
UNCERTAIN_CLASS = "UNCERTAIN_TRANSIT_LIKE_SIGNAL"
PLANET_BINARY = "planet_like"
OTHER_BINARY = "false_positive_or_other"


@dataclass(frozen=True)
class PublicSource:
    source: str
    description: str
    metadata_url: str
    notes: str


PUBLIC_SOURCES: dict[str, PublicSource] = {
    "kepler-dr25": PublicSource(
        source="kepler-dr25",
        description="NASA Exoplanet Archive Kepler DR25 KOI/TCE metadata",
        metadata_url="https://exoplanetarchive.ipac.caltech.edu/",
        notes="Use KOI/TCE dispositions for binary labels; use canonical labels only where subtype is reliable.",
    ),
    "tess-toi": PublicSource(
        source="tess-toi",
        description="NASA Exoplanet Archive / TESS Project Candidate TOI metadata",
        metadata_url="https://exoplanetarchive.ipac.caltech.edu/",
        notes="TOI dispositions generally support planet-like vs false-positive-or-other labels.",
    ),
    "tess-dv": PublicSource(
        source="tess-dv",
        description="MAST TESS Data Validation / Data Alert products and TCE/DVT metadata",
        metadata_url="https://mast.stsci.edu/",
        notes="DV products are used for cached examples and metadata joins; labels usually come from TOI/TCE dispositions.",
    ),
}

EXOPLANET_ARCHIVE_TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_ARCHIVE_TAP_TABLES: dict[str, str] = {
    "tess-toi": "toi",
    "kepler-dr25": "q1_q17_dr25_koi",
}

STSCI_TIC_CTL_PAGE_URL = "https://archive.stsci.edu/tess/tic_ctl.html"
STSCI_XCTL_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01.csv"
STSCI_XCTL_HEADER_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01_header.csv"
STSCI_XCTL_TIC_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01xTIC_v8.1.csv"
STSCI_XCTL_TIC_HEADER_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01xTIC_v8.1_header.csv"
STSCI_TIC_COLUMN_DESCRIPTION_URL = "https://archive.stsci.edu/missions/tess/catalogs/tic_v81/tic_column_description.txt"

CTL_COLUMN_NAMES: list[str] = ["ID", "priority", "splists", "objID"]
TIC_MINIMAL_USECOLS: dict[int, str] = {
    0: "ID",
    13: "ra",
    14: "dec",
    60: "Tmag",
    64: "Teff",
    66: "logg",
    70: "rad",
    72: "mass",
    74: "rho",
    77: "lum",
    79: "d",
    87: "priority",
    124: "objID",
}


def supported_public_sources() -> list[str]:
    return sorted(PUBLIC_SOURCES)


def build_exoplanet_archive_tap_query(
    source: str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    top: int | None = None,
    where: str | None = None,
) -> str:
    """Build an ADQL query for the official NASA Exoplanet Archive TAP service."""
    source = _normalize_source(source)
    if source not in EXOPLANET_ARCHIVE_TAP_TABLES:
        raise ValueError(f"{source!r} is not available through the NASA Exoplanet Archive TAP helper.")
    table = EXOPLANET_ARCHIVE_TAP_TABLES[source]
    select_cols = ", ".join(columns) if columns else "*"
    top_clause = f"TOP {int(top)} " if top is not None else ""
    query = f"SELECT {top_clause}{select_cols} FROM {table}"
    if where:
        query += f" WHERE {where}"
    return query


def build_exoplanet_archive_tap_url(query: str, *, output_format: str = "csv") -> str:
    return EXOPLANET_ARCHIVE_TAP_SYNC_URL + "?" + urlencode({"query": query, "format": output_format})


def download_exoplanet_archive_metadata(
    source: str,
    output_dir: str | Path = "data/public",
    *,
    top: int | None = None,
    where: str | None = None,
    timeout: int = 180,
) -> dict[str, Path]:
    """Download and normalize official NASA Exoplanet Archive metadata.

    This helper intentionally downloads metadata only. Light-curve and DV files
    can be much larger and are handled separately through MAST/astroquery.
    """
    source = _normalize_source(source)
    query = build_exoplanet_archive_tap_query(source, top=top, where=where)
    url = build_exoplanet_archive_tap_url(query)
    output = Path(output_dir)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_path = metadata_dir / f"{source}_raw_metadata.csv"
    with urlopen(url, timeout=timeout) as response:
        raw_path.write_bytes(response.read())
    normalized = read_public_metadata(raw_path, source=source)
    paths = write_public_metadata_artifacts(normalized, output, source=source)
    query_path = metadata_dir / f"{source}_tap_query.txt"
    query_path.write_text(query + "\n" + url + "\n", encoding="utf-8")
    return {"raw_metadata": raw_path, "tap_query": query_path, **paths}


def read_tic_ctl_catalog(
    path: str | Path,
    *,
    catalog_type: str = "auto",
    nrows: int | None = None,
    minimal_tic: bool = True,
    header_file: str | Path | None = None,
) -> pd.DataFrame:
    """Read a local STScI TIC/CTL catalog file.

    STScI TIC/CTL catalog files are target catalogs, not light curves. They are
    used here to create target lists whose TIC IDs can be passed to MAST light
    curve download/search.
    """
    path = Path(path)
    catalog_type = _infer_tic_ctl_type(path, catalog_type)
    if catalog_type == "ctl":
        df = pd.read_csv(path, header=None, names=CTL_COLUMN_NAMES, nrows=nrows)
    elif catalog_type == "tic":
        if minimal_tic:
            usecols = sorted(TIC_MINIMAL_USECOLS)
            df = pd.read_csv(
                path,
                header=None,
                usecols=usecols,
                nrows=nrows,
                compression="infer",
            )
            df = df.rename(columns={i: TIC_MINIMAL_USECOLS[i] for i in usecols})
        else:
            names = _read_stsci_header_names(header_file) if header_file is not None else list(TIC_MINIMAL_USECOLS.values())
            df = pd.read_csv(path, header=None, names=names, nrows=nrows, compression="infer")
    elif catalog_type == "xctl":
        names = _read_stsci_header_names(header_file) if header_file is not None else None
        df = pd.read_csv(path, header=None if names else "infer", names=names, nrows=nrows, compression="infer")
    else:
        raise ValueError("catalog_type must be one of: auto, ctl, tic, xctl")
    return standardize_tic_ctl_catalog(df, catalog_type=catalog_type)


def standardize_tic_ctl_catalog(df: pd.DataFrame, *, catalog_type: str = "tic") -> pd.DataFrame:
    """Normalize TIC/CTL target catalogs into a pipeline target-list table."""
    out = df.copy()
    out.columns = [str(c).strip().strip("[]") for c in out.columns]
    lower = {c.lower(): c for c in out.columns}

    tic_col = lower.get("id") or lower.get("tic_id") or lower.get("ticid") or lower.get("tid")
    if tic_col is None:
        raise ValueError("TIC/CTL catalog must contain an ID/TIC ID column.")
    out["tic_id"] = pd.to_numeric(out[tic_col], errors="coerce").astype("Int64")
    out["source_catalog"] = catalog_type

    rename_map = {
        "ra": "ra",
        "dec": "dec",
        "tmag": "tmag",
        "teff": "teff",
        "logg": "logg",
        "rad": "stellar_radius",
        "mass": "stellar_mass",
        "rho": "stellar_density",
        "lum": "stellar_luminosity",
        "d": "distance_pc",
        "priority": "ctl_priority",
        "splists": "ctl_splists",
        "objid": "ctl_obj_id",
    }
    for src_lower, dst in rename_map.items():
        src = lower.get(src_lower)
        if src is not None and dst not in out.columns:
            out[dst] = out[src]

    numeric_cols = [
        "ra",
        "dec",
        "tmag",
        "teff",
        "logg",
        "stellar_radius",
        "stellar_mass",
        "stellar_density",
        "stellar_luminosity",
        "distance_pc",
        "ctl_priority",
        "ctl_obj_id",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "ctl_splists" in out.columns:
        out["ctl_splists"] = out["ctl_splists"].fillna("").astype(str)

    front = [
        "tic_id",
        "source_catalog",
        "ra",
        "dec",
        "tmag",
        "teff",
        "logg",
        "stellar_radius",
        "stellar_mass",
        "ctl_priority",
        "ctl_splists",
        "ctl_obj_id",
    ]
    ordered = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    return out[ordered]


def write_tic_ctl_target_list(df: pd.DataFrame, output_path: str | Path, *, max_targets: int | None = None) -> Path:
    out = df.copy()
    out = out[out["tic_id"].notna()].copy()
    if "ctl_priority" in out.columns:
        out = out.sort_values("ctl_priority", ascending=False, na_position="last")
    if max_targets is not None:
        out = out.head(max_targets)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def read_public_metadata(path: str | Path, source: str) -> pd.DataFrame:
    """Read a local public metadata table and add normalized project labels."""
    source = _normalize_source(source)
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return standardize_public_metadata(df, source=source)


def standardize_public_metadata(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize public catalog metadata into a training-manifest table.

    The function is intentionally local-data only. It does not contact NASA or
    MAST, which keeps tests deterministic and prevents accidental large downloads.
    """
    source = _normalize_source(source)
    if df is None:
        raise ValueError("metadata dataframe is None")
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    labels = [_labels_for_row(row, source) for _, row in out.iterrows()]
    out["source"] = source
    out["canonical_label"] = [x["canonical_label"] for x in labels]
    out["binary_label"] = [x["binary_label"] for x in labels]
    out["label_notes"] = [x["label_notes"] for x in labels]
    _add_if_missing(out, "tic_id", _first_numeric(out, ["tic_id", "tid", "toi_tic", "tic"]))
    _add_if_missing(out, "kepid", _first_numeric(out, ["kepid", "kep_id", "kic", "target_id"]))
    _add_if_missing(out, "period_days", _first_numeric(out, ["period_days", "pl_orbper", "koi_period", "tce_period", "toi_period"]))
    _add_if_missing(out, "epoch_time", _first_numeric(out, ["epoch_time", "pl_tranmid", "koi_time0bk", "tce_time0bk", "toi_epoch"]))
    _add_if_missing(out, "duration_days", _duration_days(out))
    _add_if_missing(out, "depth_ppm", _first_numeric(out, ["depth_ppm", "koi_depth", "tce_depth", "toi_depth", "pl_trandep"]))
    if source == "tess-toi":
        if "pl_tranmid" in out.columns and "epoch_time_bjd" not in out.columns:
            out["epoch_time_bjd"] = pd.to_numeric(out["pl_tranmid"], errors="coerce")
        out["epoch_time"] = _convert_large_bjd_epoch(out["epoch_time"], offset=2457000.0)
        out["epoch_time_system"] = "BTJD"
    elif source == "kepler-dr25":
        if "koi_time0" in out.columns and "epoch_time_bjd" not in out.columns:
            out["epoch_time_bjd"] = pd.to_numeric(out["koi_time0"], errors="coerce")
        out["epoch_time"] = _convert_large_bjd_epoch(out["epoch_time"], offset=2454833.0)
        out["epoch_time_system"] = "BKJD"
    return out


def write_public_metadata_artifacts(df: pd.DataFrame, output_dir: str | Path, source: str) -> dict[str, Path]:
    """Write normalized public metadata and a small manifest under an ignored path."""
    source = _normalize_source(source)
    output = Path(output_dir)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = metadata_dir / f"{source}_normalized_metadata.csv"
    manifest_path = metadata_dir / f"{source}_manifest.json"
    df.to_csv(normalized_path, index=False)
    manifest = {
        "source": asdict(PUBLIC_SOURCES[source]),
        "n_rows": int(len(df)),
        "canonical_label_counts": _value_counts(df.get("canonical_label")),
        "binary_label_counts": _value_counts(df.get("binary_label")),
        "columns": list(df.columns),
        "cache_policy": "Downloaded data, generated examples, and trained model outputs should remain under ignored data/public or outputs_cnn paths.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"normalized_metadata": normalized_path, "manifest": manifest_path}


def _labels_for_row(row: pd.Series, source: str) -> dict[str, str | None]:
    raw = _first_text(row, _disposition_columns(source))
    raw_norm = raw.strip().lower().replace(" ", "_").replace("-", "_") if raw else ""
    canonical: str | None = None
    binary: str | None = None
    notes: list[str] = []

    if source == "tess-toi":
        if raw_norm in {"pc", "apc", "cp", "kp", "known_planet", "confirmed", "candidate", "planet_candidate"}:
            canonical = PLANET_CLASS
            binary = PLANET_BINARY
        elif raw_norm in {"fp", "false_positive", "false_positive_or_other", "fa", "false_alarm"}:
            binary = OTHER_BINARY
            notes.append("toi_false_positive_subtype_not_assumed")
        elif raw_norm in {"ambiguous", "unknown", "unk", ""}:
            canonical = None
            binary = None
    elif source == "kepler-dr25":
        if raw_norm in {"confirmed", "candidate", "planet_candidate", "pc"}:
            canonical = PLANET_CLASS
            binary = PLANET_BINARY
        elif raw_norm in {"false_positive", "fp"}:
            binary = OTHER_BINARY
            notes.append("kepler_false_positive_subtype_not_assumed")
        elif raw_norm in {"afp", "astrophysical_false_positive"}:
            canonical = EB_CLASS
            binary = OTHER_BINARY
        elif raw_norm in {"ntp", "non_transiting_phenomenon", "false_alarm"}:
            canonical = SYSTEMATIC_CLASS
            binary = OTHER_BINARY
    elif source == "tess-dv":
        if raw_norm in {"planet_candidate", "pc", "candidate", "threshold_crossing_event_planet_candidate"}:
            canonical = PLANET_CLASS
            binary = PLANET_BINARY
        elif raw_norm in {"false_positive", "fp", "false_alarm", "fa"}:
            binary = OTHER_BINARY
            notes.append("dv_false_positive_subtype_not_assumed")

    if canonical is None and binary is None:
        aliased = normalize_label(raw)
        if aliased in CANONICAL_CLASSES:
            canonical = aliased
            binary = PLANET_BINARY if aliased == PLANET_CLASS else OTHER_BINARY
    return {
        "canonical_label": canonical,
        "binary_label": binary,
        "label_notes": ";".join(notes),
    }


def _normalize_source(source: str) -> str:
    source = str(source).strip().lower()
    aliases = {
        "toi": "tess-toi",
        "tess_project_candidates": "tess-toi",
        "tess-project-candidates": "tess-toi",
        "kepler": "kepler-dr25",
        "kepler_dr25": "kepler-dr25",
        "tess_dv": "tess-dv",
    }
    source = aliases.get(source, source)
    if source not in PUBLIC_SOURCES:
        raise ValueError(f"Unsupported public source {source!r}. Supported sources: {', '.join(supported_public_sources())}")
    return source


def _infer_tic_ctl_type(path: Path, catalog_type: str) -> str:
    catalog_type = str(catalog_type).strip().lower()
    if catalog_type != "auto":
        return catalog_type
    name = path.name.lower()
    if "xctl" in name and "xtic" in name:
        return "xctl"
    if name.startswith("tic_") or "tic_dec" in name:
        return "tic"
    if "ctl" in name:
        return "ctl"
    return "tic"


def _read_stsci_header_names(header_file: str | Path | None) -> list[str]:
    if header_file is None:
        raise ValueError("header_file is required to read the full TIC/xCTL catalog.")
    text = Path(header_file).read_text(encoding="utf-8")
    names: list[str] = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part.startswith("["):
            continue
        end = part.find("]")
        if end > 1:
            names.append(part[1:end])
    if not names:
        raise ValueError(f"No column names found in STScI header file {header_file!r}.")
    return names


def _disposition_columns(source: str) -> list[str]:
    common = ["disposition", "disp", "label", "canonical_label"]
    if source == "tess-toi":
        return ["tfopwg_disp", "toi_disposition", "toi_disp", "disposition", *common]
    if source == "kepler-dr25":
        return ["koi_disposition", "koi_pdisposition", "av_training_set", "tce_label", *common]
    return ["dv_disposition", "tce_disposition", "tfopwg_disp", *common]


def _first_text(row: pd.Series, columns: list[str]) -> str:
    lower_map = {str(c).lower(): c for c in row.index}
    for col in columns:
        actual = lower_map.get(col.lower())
        if actual is None:
            continue
        value = row.get(actual)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _first_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for col in columns:
        actual = lower_map.get(col.lower())
        if actual is not None:
            vals = pd.to_numeric(df[actual], errors="coerce")
            if vals.notna().any():
                return vals
    return None


def _duration_days(df: pd.DataFrame) -> pd.Series | None:
    days = _first_numeric(df, ["duration_days", "koi_duration_days", "tce_duration_days"])
    if days is not None:
        return days
    hours = _first_numeric(df, ["duration_hours", "koi_duration", "tce_duration", "toi_duration", "pl_trandurh"])
    if hours is not None:
        return hours / 24.0
    return None


def _add_if_missing(df: pd.DataFrame, column: str, values: pd.Series | None) -> None:
    if column not in df.columns:
        df[column] = values if values is not None else np.nan


def _convert_large_bjd_epoch(values: Any, offset: float) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    return vals.where(~(vals > 2_400_000.0), vals - offset)


def _value_counts(series: Any) -> dict[str, int]:
    if series is None:
        return {}
    return {str(k): int(v) for k, v in pd.Series(series).fillna("UNLABELED").value_counts().sort_index().items()}
