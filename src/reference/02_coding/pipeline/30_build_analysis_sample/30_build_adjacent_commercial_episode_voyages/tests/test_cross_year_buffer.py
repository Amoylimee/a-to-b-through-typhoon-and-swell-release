"""Unit checks for the accepted cross-year identity and follow-up rules."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pandas as pd
import pytest


HERE = Path(__file__).resolve()
DRIVER = HERE.parents[1] / "20_build_cross_year_buffered_adjacent_commercial_episode_voyages.py"
SPEC = importlib.util.spec_from_file_location("cross_year_driver", DRIVER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_namespace_visit_ids_preserves_blank_and_missing_values() -> None:
    frame = pd.DataFrame({"visit_id": ["123", " 456 ", "", None]})
    result = MODULE.namespace_visit_ids(frame, 2023)
    assert result["visit_id"].tolist()[:3] == ["2023:123", "2023: 456 ", ""]
    assert pd.isna(result.loc[3, "visit_id"])


def test_namespace_visit_ids_is_idempotent_across_mixed_input() -> None:
    frame = pd.DataFrame(
        {
            "visit_id": [
                "1010590_v00075",
                "2023:1010590_v00001",
                "2022:already-prefixed",
                "",
                None,
            ]
        }
    )
    once = MODULE.namespace_visit_ids(frame, 2022)
    twice = MODULE.namespace_visit_ids(once, 2022)
    assert once["visit_id"].tolist()[:4] == [
        "2022:1010590_v00075",
        "2023:1010590_v00001",
        "2022:already-prefixed",
        "",
    ]
    assert pd.isna(once.loc[4, "visit_id"])
    pd.testing.assert_frame_equal(once, twice)


def test_common_followup_uses_calendar_year_and_prior_scope() -> None:
    frame = pd.DataFrame(
        {
            "departure_time": [
                "2023-02-28T00:00:00Z",
                "2023-02-28T00:00:00Z",
                "2024-02-29T00:00:00Z",
            ],
            "arrival_time": [
                "2024-02-28T00:00:00Z",
                "2024-02-29T00:00:00Z",
                "2025-02-28T00:00:00Z",
            ],
            "temporal_scope_eligible": [True, True, False],
        }
    )
    result = MODULE.apply_common_followup_window(frame)
    assert result["e017_common_followup_eligible"].tolist() == [True, False, True]
    assert result["temporal_scope_eligible"].tolist() == [True, False, False]


def test_formal_years_and_worker_cap_are_frozen() -> None:
    assert MODULE.DEFAULT_YEARS == (2021, 2022, 2023, 2024, 2025)
    assert MODULE.BUFFERED_YEARS == (2021, 2022, 2023, 2024)
    assert MODULE.SUPPORTED_YEARS == frozenset((2021, 2022, 2023, 2024, 2025))
    assert MODULE.SPEC_TAG == "next_calendar_year_v4_common_one_year_followup"


def test_formal_output_and_frozen_legacy_support_are_separate() -> None:
    builder = MODULE.p0
    assert builder.DEFAULT_OUTPUT_ROOT == (
        builder.CODING_ROOT
        / "data/processed/products/adjacent_commercial_episode_voyages"
    )
    assert builder.FROZEN_SUPPORT_ROOT == (
        builder.CODING_ROOT
        / "data/frozen_inputs/stage30_support_2025/data/processed/build_multicall_first_passage_voyages/main"
    )
    assert builder.DEFAULT_OUTPUT_ROOT != builder.FROZEN_SUPPORT_ROOT
    assert (
        builder.FROZEN_SUPPORT_ROOT
        / "support/frozen_2021/city_snap_table_2021.csv"
    ).is_file()
    assert (
        builder.FROZEN_SUPPORT_ROOT
        / "support/endpoint_2025/city_snap_table_c5_2025.csv"
    ).is_file()


def test_every_downstream_builder_requires_the_formal_mode() -> None:
    builder = MODULE.p0
    for function in (
        builder.materialize_c2,
        builder.downstream_config,
        builder.build_final_year,
        builder.run_full_pipeline,
        builder.build_delivery_products,
    ):
        assert "candidate_enumeration_mode" in inspect.signature(
            function
        ).parameters
    config = builder.downstream_config(
        2021,
        builder.CODING_ROOT / "support.feather",
        16,
        builder.candidate_pairs.MAIN_CANDIDATE_ENUMERATION_MODE,
    )
    assert (
        config["candidate_enumeration_mode"]
        == "adjacent_commercial_episodes"
    )


def test_city_support_scope_is_explicit_and_covers_analysis_years() -> None:
    builder = MODULE.p0
    signature = inspect.signature(builder.run_full_pipeline)
    assert "city_support_years" in signature.parameters
    assert builder.DEFAULT_YEARS == (2021, 2025)
    assert builder.DEFAULT_CITY_SUPPORT_YEARS == (2021, 2022, 2023, 2024, 2025)
    assert builder.validate_city_support_years(
        (2021, 2025),
        (2021, 2022, 2023, 2024, 2025),
    ) == (2021, 2022, 2023, 2024, 2025)
    with pytest.raises(ValueError, match="contain every analysis year"):
        builder.validate_city_support_years((2021, 2025), (2021, 2022, 2023))


def test_cli_output_paths_resolve_from_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert MODULE.resolve_cli_output(Path("02_coding/.runs/formal")) == (
        tmp_path / "02_coding/.runs/formal"
    )
