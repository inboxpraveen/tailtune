"""Installed package versions, compared with the ranges tailtune is tested against.

Reads metadata only. Nothing here imports the packages it reports on.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Literal

from packaging.version import InvalidVersion, Version

Standing = Literal["tested", "newer", "older", "unsupported", "missing", "unknown"]


@dataclass(frozen=True)
class TestedRange:
    """`low <= v < high` is tested. Below `floor` is unsupported outright."""

    package: str
    low: str
    high: str
    floor: str | None = None


# The ranges v0.1 will be tested against (docs/plan/11-engineering-and-release.md section 2).
# 0.0.1 requires none of these packages; doctor reports them so users can see where they stand.
TESTED: dict[str, TestedRange] = {
    "torch": TestedRange("torch", "2.14", "2.16", floor="2.14"),
    "transformers": TestedRange("transformers", "5.22", "5.23", floor="5.0"),
    "accelerate": TestedRange("accelerate", "1.14", "2.0", floor="1.14"),
    "peft": TestedRange("peft", "0.21.2", "0.22"),
    "trl": TestedRange("trl", "1.17", "1.18"),
}


@dataclass(frozen=True)
class Installed:
    package: str
    version: str | None
    standing: Standing


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _release(v: Version) -> Version:
    """Drop local and pre-release parts, so `2.14.1+cu130` compares as `2.14.1`."""
    return Version(v.base_version)


def standing(package: str, version: str | None = None) -> Installed:
    """Where an installed version sits against TESTED. `version` overrides the lookup."""
    found = version if version is not None else installed_version(package)
    if found is None:
        return Installed(package, None, "missing")
    rng = TESTED.get(package)
    if rng is None:
        return Installed(package, found, "unknown")
    try:
        v = _release(Version(found))
    except InvalidVersion:
        return Installed(package, found, "unknown")
    if rng.floor is not None and v < Version(rng.floor):
        return Installed(package, found, "unsupported")
    if v < Version(rng.low):
        return Installed(package, found, "older")
    if v >= Version(rng.high):
        return Installed(package, found, "newer")
    return Installed(package, found, "tested")


def describe_range(package: str) -> str:
    rng = TESTED[package]
    return f">={rng.low},<{rng.high}"
