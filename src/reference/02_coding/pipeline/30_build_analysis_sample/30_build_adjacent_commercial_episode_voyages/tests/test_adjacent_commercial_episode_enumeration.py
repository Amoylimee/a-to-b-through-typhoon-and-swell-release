"""Unit checks for the accepted adjacent-episode main rule."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


HERE = Path(__file__).resolve()
STAGE_ROOT = HERE.parents[1]
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

from core.candidate_core import enumerate_vessel_candidates  # noqa: E402
from core.candidate_pairs import (  # noqa: E402
    ADJACENT_COMMERCIAL_EPISODES,
    MAIN_CANDIDATE_ENUMERATION_MODE,
    destination_episode_indices,
)
from core.city_episodes import enumerate_adjacent_episode_pairs  # noqa: E402


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    base = pd.Timestamp("2021-01-01T00:00:00Z")
    rows = []
    for sequence, city in enumerate(("AAA:A", "BBB:B", "CCC:C", "DDD:D")):
        arrival = base + pd.Timedelta(hours=sequence * 24)
        departure = pd.NaT if sequence == 3 else arrival + pd.Timedelta(hours=2)
        rows.append(
            {
                "mmsi": "123456789",
                "episode_sequence": sequence,
                "imo": "IMO1234567",
                "vessel_uid_clean": "ship-1",
                "vessel_name": "TEST VESSEL",
                "detailed_vessel_type_normalized": "CONTAINER SHIP",
                "ship_type_sample": "main",
                "topology_city_id": city,
                "passage_arrival_time": arrival,
                "passage_departure_time": departure,
                "episode_id": f"episode-{sequence}",
                "first_call_id": f"call-{sequence}",
                "last_call_id": f"call-{sequence}",
            }
        )
    episodes = pd.DataFrame(rows)
    legs = pd.DataFrame(
        [
            {
                "mmsi": "123456789",
                "leg_sequence": sequence,
                "trip_id": f"leg-{sequence}",
                "departure_call_id": f"call-{sequence}",
                "departure_time": episodes.loc[sequence, "passage_departure_time"],
                "arrival_call_id": f"call-{sequence + 1}",
                "arrival_time": episodes.loc[sequence + 1, "passage_arrival_time"],
            }
            for sequence in range(3)
        ]
    )
    call_meta = {
        f"call-{sequence}": {
            "topology_city_id": row["topology_city_id"],
            "episode_id": row["episode_id"],
        }
        for sequence, row in episodes.iterrows()
    }
    return episodes, legs, call_meta


def enumerate_candidates(mode: str | None = None) -> pd.DataFrame:
    episodes, legs, call_meta = synthetic_inputs()
    args = (
        episodes,
        legs,
        call_meta,
        [
            {
                "period_id": "2021_full_year",
                "start": pd.Timestamp("2021-01-01T00:00:00Z"),
                "end": pd.Timestamp("2022-01-01T00:00:00Z"),
            }
        ],
        {"CONTAINER SHIP"},
        set(),
    )
    candidates, _ = (
        enumerate_vessel_candidates(*args)
        if mode is None
        else enumerate_vessel_candidates(*args, mode)
    )
    return candidates


def test_main_rule_emits_only_consecutive_commercial_episodes() -> None:
    default = enumerate_candidates()
    explicit = enumerate_candidates(ADJACENT_COMMERCIAL_EPISODES)

    assert MAIN_CANDIDATE_ENUMERATION_MODE == ADJACENT_COMMERCIAL_EPISODES
    assert len(default) == 3
    assert default["candidate_id"].tolist() == explicit["candidate_id"].tolist()
    assert default["n_intermediate_city_episodes"].eq(0).all()
    assert default["dest_episode_sequence"].sub(
        default["origin_episode_sequence"]
    ).eq(1).all()
    assert default["pair_id"].tolist() == [
        "AAA:A__BBB:B",
        "BBB:B__CCC:C",
        "CCC:C__DDD:D",
    ]


def test_pair_manifest_uses_the_same_main_rule() -> None:
    episodes, _, _ = synthetic_inputs()
    manifest, diagnostics = enumerate_adjacent_episode_pairs(episodes)

    assert diagnostics["candidate_enumeration_mode"] == (
        ADJACENT_COMMERCIAL_EPISODES
    )
    assert diagnostics["candidate_occurrences"] == 3
    assert set(manifest["pair_id"]) == {
        "AAA:A__BBB:B",
        "BBB:B__CCC:C",
        "CCC:C__DDD:D",
    }


def test_destination_rule_is_exactly_one_episode_forward() -> None:
    assert list(destination_episode_indices(["A", "B", "C"], 0)) == [1]


@pytest.mark.parametrize("mode", ["fanout_first_passage", "unknown"])
def test_non_main_modes_fail_before_enumeration(mode: str) -> None:
    with pytest.raises(ValueError, match="formal main pipeline"):
        list(destination_episode_indices(["A", "B"], 0, mode))
