"""Generische Autopilot-Benchmark-Tests — liest den TOML-Katalog."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.autopilot.factory import create_autopilot
from sailsim.benchmarks.catalog import apply_gate_overrides, load_catalog
from sailsim.core.config import load_autopilot, load_scenario, load_yacht
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_heading_hold

catalog = load_catalog("configs/benchmarks/autopilot_catalog.toml")


@pytest.mark.docker
@pytest.mark.parametrize("entry", catalog, ids=[e.name for e in catalog])
def test_benchmark(entry, pypilot_service, tmp_path):
    """Run a benchmark entry: scenario + autopilot + quality gate overrides."""
    config = load_scenario(entry.scenario)
    config.yacht = load_yacht(entry.yacht)
    if entry.gate_overrides:
        config.quality_gates = apply_gate_overrides(config.quality_gates, entry.gate_overrides)

    ap_config = load_autopilot(entry.autopilot)
    autopilot = create_autopilot(ap_config, yacht=config.yacht)

    recorder = run_scenario(config, autopilot)
    # Save JSON for later analysis with sailsim --view
    recorder.to_json(tmp_path / f"{entry.name}.json")

    result = evaluate_heading_hold(
        recorder,
        np.degrees(config.target_heading),
        config.quality_gates,
    )
    assert result.passed, f"{entry.name} failed:\n{result.summary()}"
