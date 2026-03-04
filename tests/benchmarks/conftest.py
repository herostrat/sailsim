"""Fixtures for benchmark integration tests."""

from __future__ import annotations

import contextlib
import logging
import socket
import time

import pytest

from sailsim.benchmarks.docker_manager import DockerService

logger = logging.getLogger(__name__)


def _wait_for_pypilot_ready(
    host: str = "localhost",
    port: int = 23322,
    timeout: float = 30.0,
) -> None:
    """Wait until pypilot's JSON-TCP protocol is responsive.

    A bare TCP connection can succeed before pypilot's autopilot logic
    is initialized.  This probe sends a ``watch`` command and waits for
    pypilot to reply with actual data, confirming it is ready.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.settimeout(5.0)
            # Ask pypilot to stream a well-known value
            sock.sendall(b'watch={"ap.enabled":true}\n')
            data = sock.recv(4096)
            sock.close()
            if b"ap.enabled" in data:
                logger.info("pypilot JSON-TCP ready")
                return
        except OSError:
            pass
        with contextlib.suppress(OSError):
            sock.close()  # type: ignore[possibly-undefined]
        time.sleep(1.0)
    raise TimeoutError(f"pypilot not ready after {timeout}s")


@pytest.fixture(scope="session")
def pypilot_service():
    """Start pypilot Docker container for the test session."""
    svc = DockerService(
        compose_file="docker/docker-compose.yml",
        service="pypilot",
        health_port=23322,
    )
    svc.start(timeout=30)
    # Wait for pypilot protocol to be fully functional, not just the port
    _wait_for_pypilot_ready(timeout=30)
    yield svc
    svc.stop()
