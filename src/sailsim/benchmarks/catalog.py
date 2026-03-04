"""Benchmark catalog loader — reads TOML catalog, applies gate overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from sailsim.core.config import QualityGateConfig


@dataclass
class BenchmarkEntry:
    """A single benchmark definition from the catalog."""

    name: str
    scenario: str
    autopilot: str
    yacht: str = "default"
    gate_overrides: dict[str, float] = field(default_factory=dict)


def load_catalog(path: str | Path) -> list[BenchmarkEntry]:
    """Load benchmark entries from a TOML catalog file.

    Expected format::

        [[benchmark]]
        name = "pypilot_calm"
        scenario = "calm_heading_hold"
        autopilot = "pypilot"
        yacht = "default"
        [benchmark.gate_overrides]
        max_heading_deviation_deg = 20.0
    """
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)

    entries: list[BenchmarkEntry] = []
    for item in data.get("benchmark", []):
        entries.append(
            BenchmarkEntry(
                name=item["name"],
                scenario=item["scenario"],
                autopilot=item["autopilot"],
                yacht=item.get("yacht", "default"),
                gate_overrides=item.get("gate_overrides", {}),
            )
        )
    return entries


def apply_gate_overrides(
    gates: QualityGateConfig,
    overrides: dict[str, float],
) -> QualityGateConfig:
    """Return a new QualityGateConfig with overridden values.

    Only fields that exist on QualityGateConfig are applied; unknown
    keys are silently ignored.
    """
    data = gates.model_dump()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    return QualityGateConfig(**data)
