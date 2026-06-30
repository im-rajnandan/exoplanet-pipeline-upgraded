from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .schema import RawLightCurve


def _read_header_value(header: Any, keys: list[str], default=None):
    for key in keys:
        if key in header:
            value = header.get(key)
            if value not in (None, ""):
                return value
    return default


def _get_col(data: Any, names: list[str]) -> np.ndarray | None:
    available = {name.upper(): name for name in getattr(data, "names", [])}
    for name in names:
        key = name.upper()
        if key in available:
            arr = np.asarray(data[available[key]], dtype=float)
            return arr
    return None


def load_tess_fits(file_path: str | Path) -> RawLightCurve:
    """Load a local TESS light-curve FITS file.

    This function intentionally supports local FITS first because competition data
    may be provided as files. Download/search helpers can be added around it.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:
        return RawLightCurve(
            tic_id=None,
            sector=None,
            time=np.array([]),
            status="MISSING_ASTROPY",
            error="Install astropy to read FITS files.",
        )

    path = Path(file_path)
    if not path.exists():
        return RawLightCurve(None, None, np.array([]), status="FILE_NOT_FOUND", error=str(path))

    try:
        with fits.open(path, memmap=False) as hdul:
            primary_header = hdul[0].header
            # Most SPOC light curves store the time series in extension 1.
            lightcurve_hdu = hdul[1]
            header = lightcurve_hdu.header
            data = lightcurve_hdu.data

            time = _get_col(data, ["TIME"])
            if time is None:
                return RawLightCurve(None, None, np.array([]), status="NO_TIME_COLUMN", error=str(path))

            sap_flux = _get_col(data, ["SAP_FLUX"])
            sap_flux_err = _get_col(data, ["SAP_FLUX_ERR"])
            pdcsap_flux = _get_col(data, ["PDCSAP_FLUX"])
            pdcsap_flux_err = _get_col(data, ["PDCSAP_FLUX_ERR"])
            centroid_col = _get_col(data, ["MOM_CENTR1", "CENTR1", "POS_CORR1"])
            centroid_row = _get_col(data, ["MOM_CENTR2", "CENTR2", "POS_CORR2"])

            quality = None
            available = {name.upper(): name for name in getattr(data, "names", [])}
            if "QUALITY" in available:
                quality = np.asarray(data[available["QUALITY"]]).astype(int)

            tic_id = _read_header_value(primary_header, ["TICID", "TIC_ID", "OBJECT"], None)
            if isinstance(tic_id, str):
                digits = "".join(ch for ch in tic_id if ch.isdigit())
                tic_id = int(digits) if digits else None
            elif tic_id is not None:
                try:
                    tic_id = int(tic_id)
                except Exception:
                    tic_id = None

            sector = _read_header_value(primary_header, ["SECTOR"], _read_header_value(header, ["SECTOR"], None))
            try:
                sector = int(sector) if sector is not None else None
            except Exception:
                sector = None

            metadata = {
                "file_path": str(path),
                "tic_id": tic_id,
                "sector": sector,
                "camera": _read_header_value(primary_header, ["CAMERA"], _read_header_value(header, ["CAMERA"], None)),
                "ccd": _read_header_value(primary_header, ["CCD"], _read_header_value(header, ["CCD"], None)),
                "object": _read_header_value(primary_header, ["OBJECT"], None),
                "tess_mag": _read_header_value(primary_header, ["TESSMAG", "TESS_MAG"], None),
                "teff": _read_header_value(primary_header, ["TEFF"], None),
                "logg": _read_header_value(primary_header, ["LOGG"], None),
                "stellar_radius": _read_header_value(primary_header, ["RADIUS", "RSTAR"], None),
                "stellar_mass": _read_header_value(primary_header, ["MASS", "MSTAR"], None),
                "crowdsap": _read_header_value(header, ["CROWDSAP"], _read_header_value(primary_header, ["CROWDSAP"], None)),
                "flfrcsap": _read_header_value(header, ["FLFRCSAP"], _read_header_value(primary_header, ["FLFRCSAP"], None)),
                "available_columns": list(getattr(data, "names", [])),
            }

            return RawLightCurve(
                tic_id=tic_id,
                sector=sector,
                time=np.asarray(time, dtype=float),
                sap_flux=sap_flux,
                sap_flux_err=sap_flux_err,
                pdcsap_flux=pdcsap_flux,
                pdcsap_flux_err=pdcsap_flux_err,
                quality=quality,
                centroid_col=centroid_col,
                centroid_row=centroid_row,
                metadata=metadata,
                status="RAW_LOADED",
            )
    except Exception as exc:
        return RawLightCurve(None, None, np.array([]), status="FITS_READ_FAILED", error=repr(exc))


def search_and_download_tess_lc(tic_id: int, sector: int | None = None, download_dir: str | Path = "data/raw") -> list[Path]:
    """Optional helper for downloading TESS light curves with astroquery.

    This is intentionally separate from preprocessing so failures remain explicit.
    It returns local file paths or an empty list. It never generates synthetic data.
    """
    try:
        from astroquery.mast import Observations
    except ImportError as exc:
        raise RuntimeError("Install astroquery to download from MAST.") from exc

    criteria = {
        "target_name": f"TIC {tic_id}",
        "dataproduct_type": "timeseries",
        "obs_collection": "TESS",
    }
    if sector is not None:
        criteria["sequence_number"] = int(sector)

    obs = Observations.query_criteria(**criteria)
    if len(obs) == 0:
        return []
    products = Observations.get_product_list(obs)
    mask = ["lc.fits" in str(row["productFilename"]).lower() for row in products]
    lc_products = products[mask]
    if len(lc_products) == 0:
        return []
    manifest = Observations.download_products(lc_products, download_dir=str(download_dir), mrp_only=False)
    return [Path(p) for p in manifest["Local Path"] if p]
