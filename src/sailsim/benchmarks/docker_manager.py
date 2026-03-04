"""Docker container lifecycle management for benchmark tests."""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerService:
    """Manages a Docker Compose service lifecycle. Use as context manager."""

    def __init__(
        self,
        compose_file: str | Path,
        service: str,
        health_port: int | None = None,
    ) -> None:
        self.compose_file = str(Path(compose_file).resolve())
        self.service = service
        self.health_port = health_port
        self._started = False

    def start(self, timeout: float = 30.0) -> None:
        """Build and start the service, wait until healthy."""
        logger.info("Starting Docker service %s ...", self.service)
        subprocess.run(
            [
                "docker", "compose", "-f", self.compose_file,
                "up", "-d", "--build", self.service,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._started = True

        if self.health_port is not None:
            self._wait_for_port(self.health_port, timeout)

    def stop(self) -> None:
        """Stop and remove the service container."""
        if not self._started:
            return
        logger.info("Stopping Docker service %s ...", self.service)
        subprocess.run(
            [
                "docker", "compose", "-f", self.compose_file,
                "down", self.service,
            ],
            capture_output=True,
            text=True,
        )
        self._started = False

    def is_healthy(self) -> bool:
        """Check if the service port is reachable."""
        if self.health_port is None:
            return self._started
        try:
            with socket.create_connection(("localhost", self.health_port), timeout=2):
                return True
        except OSError:
            return False

    def __enter__(self) -> DockerService:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @staticmethod
    def _wait_for_port(port: int, timeout: float) -> None:
        """Wait until a TCP port is reachable on localhost."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=2):
                    logger.info("Port %d is ready", port)
                    return
            except OSError:
                time.sleep(1.0)
        raise TimeoutError(f"Port {port} not reachable after {timeout}s")
