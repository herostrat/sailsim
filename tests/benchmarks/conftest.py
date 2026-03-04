"""Fixtures for benchmark integration tests."""

from __future__ import annotations

import pytest

from sailsim.benchmarks.docker_manager import DockerService


@pytest.fixture(scope="session")
def pypilot_service():
    """Start pypilot Docker container for the test session."""
    svc = DockerService(
        compose_file="docker/docker-compose.yml",
        service="pypilot",
        health_port=23322,
    )
    svc.start(timeout=30)
    yield svc
    svc.stop()
