"""Spectral wave environment models.

Provides JONSWAP and Pierson-Moskowitz spectrum-based wave fields.
Wave elevation is computed as a superposition of regular wave components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sailsim.core.types import WaveState

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Gravity and water density
G = 9.81
RHO_WATER = 1025.0


def pierson_moskowitz_spectrum(
    omega: NDArray[np.float64],
    Hs: float,
    Tp: float,
) -> NDArray[np.float64]:
    """Pierson-Moskowitz wave spectrum S(omega).

    Args:
        omega: angular frequencies [rad/s]
        Hs: significant wave height [m]
        Tp: peak period [s]

    Returns:
        Spectral density S(omega) [m^2 s/rad]
    """
    omega_p = 2.0 * np.pi / Tp
    alpha = 5.0 / 16.0 * Hs**2 * omega_p**4

    S = np.zeros_like(omega)
    mask = omega > 0
    S[mask] = (alpha / omega[mask] ** 5) * np.exp(-1.25 * (omega_p / omega[mask]) ** 4)
    return S


def jonswap_spectrum(
    omega: NDArray[np.float64],
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
) -> NDArray[np.float64]:
    """JONSWAP wave spectrum S(omega).

    Pierson-Moskowitz x peak enhancement factor.

    Args:
        omega: angular frequencies [rad/s]
        Hs: significant wave height [m]
        Tp: peak period [s]
        gamma: peak enhancement factor (default 3.3)

    Returns:
        Spectral density S(omega) [m^2 s/rad]
    """
    omega_p = 2.0 * np.pi / Tp
    S_pm = pierson_moskowitz_spectrum(omega, Hs, Tp)

    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    mask = omega > 0
    exp_arg = np.zeros_like(omega)
    exp_arg[mask] = -0.5 * ((omega[mask] - omega_p) / (sigma[mask] * omega_p)) ** 2
    enhancement = gamma ** np.exp(exp_arg)

    # Normalize to preserve Hs
    S_j = S_pm * enhancement
    m0 = np.trapezoid(S_j, omega) if len(omega) > 1 else 0.0
    target_m0 = (Hs / 4.0) ** 2
    if m0 > 0:
        S_j *= target_m0 / m0

    return S_j


@dataclass
class WaveComponents:
    """Pre-generated wave component data for time-domain realization."""

    amplitudes: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    wavenumbers: NDArray[np.float64]
    phases: NDArray[np.float64]
    direction: float


def generate_wave_components(
    Hs: float,
    Tp: float,
    direction: float,
    n_components: int = 50,
    spectrum: str = "jonswap",
    gamma: float = 3.3,
    seed: int | None = None,
) -> WaveComponents:
    """Generate wave components from spectrum for time-domain simulation.

    Args:
        Hs: significant wave height [m]
        Tp: peak period [s]
        direction: wave propagation direction [rad]
        n_components: number of spectral components
        spectrum: "jonswap" or "pm"
        gamma: JONSWAP peak enhancement factor
        seed: random seed for reproducibility

    Returns:
        WaveComponents for time-domain elevation calculation.
    """
    rng = np.random.default_rng(seed)

    # Frequency range: 0.5*omega_p to 3*omega_p
    omega_p = 2.0 * np.pi / Tp
    omega_min = 0.3 * omega_p
    omega_max = 3.0 * omega_p
    omega = np.linspace(omega_min, omega_max, n_components)
    d_omega = omega[1] - omega[0] if n_components > 1 else 1.0

    if spectrum == "jonswap":
        S = jonswap_spectrum(omega, Hs, Tp, gamma)
    else:
        S = pierson_moskowitz_spectrum(omega, Hs, Tp)

    # Amplitudes from spectrum
    amplitudes = np.sqrt(2.0 * S * d_omega)

    # Deep water dispersion: omega^2 = g * k
    wavenumbers = omega**2 / G

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, n_components)

    return WaveComponents(
        amplitudes=amplitudes,
        frequencies=omega,
        wavenumbers=wavenumbers,
        phases=phases,
        direction=direction,
    )


def wave_elevation(
    components: WaveComponents,
    x: float,
    y: float,
    t: float,
) -> float:
    """Compute wave elevation at position (x, y) and time t.

    Uses linear superposition: eta = sum(a_i * cos(k_i*xi - omega_i*t + phi_i))
    where xi is the position projected onto wave direction.
    """
    # Project position onto wave propagation direction
    xi = x * np.cos(components.direction) + y * np.sin(components.direction)

    phase_args = components.wavenumbers * xi - components.frequencies * t + components.phases
    return float(np.sum(components.amplitudes * np.cos(phase_args)))


class NoWaves:
    """Null-object: no wave effects."""

    def get(self, t: float) -> WaveState:
        return WaveState()

    def set_boat_position(self, x: float, y: float) -> None:
        pass


class SpectralWaves:
    """Spectral wave model with time-domain realization."""

    def __init__(
        self,
        Hs: float,
        Tp: float,
        direction: float,
        n_components: int = 50,
        spectrum: str = "jonswap",
        gamma: float = 3.3,
        seed: int | None = None,
    ) -> None:
        self._Hs = Hs
        self._Tp = Tp
        self._direction = direction
        self._components = generate_wave_components(
            Hs,
            Tp,
            direction,
            n_components,
            spectrum,
            gamma,
            seed,
        )
        self._x = 0.0
        self._y = 0.0

    def set_boat_position(self, x: float, y: float) -> None:
        """Update boat position for elevation calculation."""
        self._x = x
        self._y = y

    def get(self, t: float) -> WaveState:
        """Return wave state at current boat position and time t."""
        elevation = wave_elevation(self._components, self._x, self._y, t)
        return WaveState(
            Hs=self._Hs,
            Tp=self._Tp,
            direction=self._direction,
            elevation=elevation,
        )
