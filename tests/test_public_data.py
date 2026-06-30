import pandas as pd

from exoplanet_pipeline.public_data import (
    build_exoplanet_archive_tap_query,
    build_exoplanet_archive_tap_url,
    read_tic_ctl_catalog,
    read_public_metadata,
    standardize_public_metadata,
    write_tic_ctl_target_list,
)


def test_tess_toi_public_metadata_binary_labels():
    df = pd.DataFrame(
        {
            "TIC_ID": [1001, 1002, 1003],
            "TFOPWG_DISP": ["PC", "FP", "Ambiguous"],
            "toi_period": [3.1, 4.2, 5.3],
            "toi_duration": [2.4, 3.0, 4.8],
        }
    )
    out = standardize_public_metadata(df, source="tess-toi")
    assert out.loc[0, "canonical_label"] == "PLANETARY_TRANSIT_CANDIDATE"
    assert out.loc[0, "binary_label"] == "planet_like"
    assert pd.isna(out.loc[1, "canonical_label"])
    assert out.loc[1, "binary_label"] == "false_positive_or_other"
    assert abs(out.loc[0, "duration_days"] - 0.1) < 1e-9


def test_tess_toi_public_metadata_official_column_aliases():
    df = pd.DataFrame(
        {
            "tid": [123456789],
            "tfopwg_disp": ["PC"],
            "pl_orbper": [2.75],
            "pl_tranmid": [2458900.25],
            "pl_trandurh": [3.6],
            "pl_trandep": [850.0],
        }
    )
    out = standardize_public_metadata(df, source="tess-toi")
    assert out.loc[0, "tic_id"] == 123456789
    assert out.loc[0, "period_days"] == 2.75
    assert out.loc[0, "epoch_time"] == 1900.25
    assert out.loc[0, "epoch_time_system"] == "BTJD"
    assert out.loc[0, "epoch_time_bjd"] == 2458900.25
    assert abs(out.loc[0, "duration_days"] - 0.15) < 1e-9
    assert out.loc[0, "depth_ppm"] == 850.0


def test_kepler_dr25_public_metadata_parser_from_csv(tmp_path):
    path = tmp_path / "kepler.csv"
    pd.DataFrame(
        {
            "kepid": [1, 2],
            "koi_disposition": ["CONFIRMED", "FALSE POSITIVE"],
            "koi_period": [10.0, 20.0],
            "koi_duration": [4.8, 6.0],
        }
    ).to_csv(path, index=False)
    out = read_public_metadata(path, source="kepler-dr25")
    assert list(out["binary_label"]) == ["planet_like", "false_positive_or_other"]
    assert out.loc[0, "canonical_label"] == "PLANETARY_TRANSIT_CANDIDATE"
    assert pd.isna(out.loc[1, "canonical_label"])


def test_exoplanet_archive_tap_query_builder():
    query = build_exoplanet_archive_tap_query("tess-toi", top=5, where="tfopwg_disp is not null")
    assert query == "SELECT TOP 5 * FROM toi WHERE tfopwg_disp is not null"
    url = build_exoplanet_archive_tap_url(query)
    assert "exoplanetarchive.ipac.caltech.edu/TAP/sync" in url
    assert "format=csv" in url


def test_ctl_catalog_target_list_parser(tmp_path):
    path = tmp_path / "exo_CTL_08.01.csv"
    path.write_text("12345,0.91,planetcandidate,77\n67890,0.12,cooldwarfs_v8,88\n", encoding="utf-8")
    out = read_tic_ctl_catalog(path, catalog_type="ctl")
    assert list(out["tic_id"].astype(int)) == [12345, 67890]
    assert out.loc[0, "ctl_priority"] == 0.91
    assert out.loc[0, "ctl_splists"] == "planetcandidate"
    target_path = write_tic_ctl_target_list(out, tmp_path / "targets.csv", max_targets=1)
    targets = pd.read_csv(target_path)
    assert targets.shape[0] == 1
    assert targets.loc[0, "tic_id"] == 12345


def test_tic_chunk_minimal_parser(tmp_path):
    path = tmp_path / "tic_dec88_00N__90_00N.csv"
    row = [""] * 125
    row[0] = "334299896"
    row[13] = "80.2525"
    row[14] = "89.0986"
    row[60] = "19.7762"
    row[64] = "3335"
    row[66] = "4.5"
    row[70] = "0.41"
    row[72] = "0.43"
    row[87] = "0.75"
    row[124] = "80103050"
    path.write_text(",".join(row) + "\n", encoding="utf-8")
    out = read_tic_ctl_catalog(path, catalog_type="tic", nrows=1)
    assert out.loc[0, "tic_id"] == 334299896
    assert out.loc[0, "ra"] == 80.2525
    assert out.loc[0, "dec"] == 89.0986
    assert out.loc[0, "tmag"] == 19.7762
    assert out.loc[0, "stellar_radius"] == 0.41
