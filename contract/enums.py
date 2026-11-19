"""Enumerations and value patterns of the data contract."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field


class Kind(StrEnum):
    """What a reading is: a sensor measurement, a reference label, or an external series."""

    measurement = "measurement"
    label = "label"
    external = "external"


class Quality(StrEnum):
    """Quality flag. Ordered good < suspect < bad; QC only ever moves rightwards."""

    good = "good"
    suspect = "suspect"
    bad = "bad"


# Who produced a reading. Free-form slug so each project names its own sources ("sensor",
# "lab", "logger-export", ...). One value is reserved: "simulator" marks invented data and
# drives the dashboard banner.
SOURCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$"
Source = Annotated[str, Field(pattern=SOURCE_PATTERN, description="who produced the reading")]
SIMULATOR = "simulator"

QUALITY_RANK: dict[Quality, int] = {Quality.good: 0, Quality.suspect: 1, Quality.bad: 2}


def worst_quality(a: Quality, b: Quality) -> Quality:
    """Return the worse of two quality flags."""
    return a if QUALITY_RANK[a] >= QUALITY_RANK[b] else b
