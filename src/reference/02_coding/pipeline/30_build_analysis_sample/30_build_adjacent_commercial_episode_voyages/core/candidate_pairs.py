"""The fixed adjacent-commercial-episode rule for main voyage candidates."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


ADJACENT_COMMERCIAL_EPISODES = "adjacent_commercial_episodes"
MAIN_CANDIDATE_ENUMERATION_MODE = ADJACENT_COMMERCIAL_EPISODES


def validate_candidate_enumeration_mode(mode: str) -> str:
    """Fail before work begins if an input is not the formal main mode."""

    value = str(mode)
    if value != MAIN_CANDIDATE_ENUMERATION_MODE:
        raise ValueError(
            "the formal main pipeline requires candidate_enumeration_mode="
            f"{MAIN_CANDIDATE_ENUMERATION_MODE!r}; got {value!r}. Historical "
            "fan-out reproduction must use the archived code snapshot."
        )
    return value


def destination_episode_indices(
    cities: Sequence[str],
    origin_index: int,
    mode: str = MAIN_CANDIDATE_ENUMERATION_MODE,
) -> Iterator[int]:
    """Yield only the immediately following commercial city episode."""

    validate_candidate_enumeration_mode(mode)
    destination_index = origin_index + 1
    if destination_index < len(cities):
        yield destination_index
