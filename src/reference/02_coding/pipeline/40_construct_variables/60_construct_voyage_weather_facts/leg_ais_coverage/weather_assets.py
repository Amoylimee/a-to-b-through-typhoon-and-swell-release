"""Resolve the frozen 2025 ERA5 assets for route-free coverage diagnostics."""

from __future__ import annotations

import json
import calendar
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WEATHER_STAGE = HERE.parent
CONFIG = WEATHER_STAGE / "config/2025_ten_minute_weather.json"


def resolve_monthly_era5_assets(*, year: int, month: int) -> list[Path]:
    if year != 2025 or not 1 <= month <= 12:
        raise ValueError("E001 coverage bounds use the frozen 2025 ERA5 calendar only")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = Path(config["wave_directory"]) / config["wave_filename_template"].format(
        month=month
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


def validate_month_time_grid(matcher, year: int, month: int) -> None:
    expected = pd.date_range(
        f"{year}-{month:02d}-01T00:00:00Z",
        periods=calendar.monthrange(year, month)[1] * 24,
        freq="h",
    )
    if not matcher.times.equals(expected):
        raise ValueError(f"ERA5 asset does not contain the exact {year}-{month:02d} UTC grid")
