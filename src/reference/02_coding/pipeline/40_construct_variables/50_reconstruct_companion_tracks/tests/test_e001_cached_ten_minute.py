import pandas as pd

from track_core.e001_cached_ten_minute import strict_grid_seconds


def test_strict_grid_excludes_observed_boundaries():
    got = strict_grid_seconds(pd.Timestamp("2025-01-01 00:03:00Z"),
                              pd.Timestamp("2025-01-01 00:31:00Z"))
    assert pd.to_datetime(got, unit="s", utc=True).tolist() == [
        pd.Timestamp("2025-01-01 00:10:00Z"),
        pd.Timestamp("2025-01-01 00:20:00Z"),
        pd.Timestamp("2025-01-01 00:30:00Z"),
    ]


def test_strict_grid_excludes_exact_boundary_bins():
    got = strict_grid_seconds(pd.Timestamp("2025-01-01 00:00:00Z"),
                              pd.Timestamp("2025-01-01 00:20:00Z"))
    assert pd.to_datetime(got, unit="s", utc=True).tolist() == [
        pd.Timestamp("2025-01-01 00:10:00Z")]
