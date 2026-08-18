"""Spatial-caption augmentation used by the DISSE implementation.

The exact descriptor lexicon is an implementation detail and is not specified
in the EUSIPCO paper. Pass a seeded ``random.Random`` instance for reproducible
caption generation.
"""

from __future__ import annotations

import random
from typing import Mapping


TEMPLATES = (
    "The sound: {orig} is coming from the {loc} of a {room}.",
    "You can hear {orig} from the {loc}, inside a {room}.",
    "The sound: {orig} originates in the {loc} of a {room}.",
)


def _direction(azimuth: float, rng: random.Random) -> str:
    if -35 <= azimuth <= 35:
        return rng.choice(("front", "in front"))
    if 55 <= azimuth <= 125:
        return rng.choice(("right", "to the right"))
    if -125 <= azimuth <= -55:
        return rng.choice(("left", "to the left"))
    if azimuth >= 145 or azimuth <= -145:
        return rng.choice(("back", "behind"))
    return ""


def _elevation(elevation: float, rng: random.Random) -> str:
    if elevation > 40:
        return rng.choice(("up", "above"))
    if elevation < -40:
        return rng.choice(("down", "below"))
    return ""


def _distance(distance: float, rng: random.Random) -> str:
    if distance < 1:
        return rng.choice(("near", "close", "nearby"))
    if distance > 2:
        return rng.choice(("far", "distant"))
    return ""


def _room_size(area: float, rng: random.Random) -> str:
    if area < 50:
        return rng.choice(("small", "tiny"))
    if area > 100:
        return rng.choice(("large", "spacious"))
    return rng.choice(("mid-sized", "medium"))


def _reverberation(t30_ms: float, rng: random.Random) -> str:
    if t30_ms < 200:
        return rng.choice(("acoustically dampened", "dry-sounding"))
    if t30_ms > 1000:
        return rng.choice(("highly reverberant", "echoey"))
    return ""


def augment_caption(
    original: str,
    metadata: Mapping[str, float],
    *,
    rng: random.Random | None = None,
) -> str:
    """Inject implementation-matched spatial descriptors into a caption."""
    rng = rng or random.Random()
    # Keep the random draws in the same order as the research implementation:
    # distance, direction, elevation, room size, reverberation, template.
    distance = _distance(float(metadata["source_distance_m"]), rng)
    direction = _direction(float(metadata["azimuth_deg"]), rng)
    elevation = _elevation(float(metadata["elevation_deg"]), rng)
    location = " ".join(
        part
        for part in (
            distance,
            elevation,
            direction,
        )
        if part
    )
    room = _room_size(float(metadata["area_m2"]), rng)
    reverb = _reverberation(float(metadata["fullband_T30_ms"]), rng)
    if reverb:
        room = f"{room} {reverb}"
    room = f"{room} room"
    template = rng.choice(TEMPLATES)
    return template.format(orig=original, loc=location, room=room)
