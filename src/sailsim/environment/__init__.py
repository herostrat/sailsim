"""Environment models (wind, current, waves)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sailsim.environment.current import ConstantCurrent, NoCurrent, TidalCurrent
from sailsim.environment.waves import NoWaves, SpectralWaves
from sailsim.environment.wind import ConstantWind, GustWind, ShiftingWind

if TYPE_CHECKING:
    from sailsim.core.config import CurrentConfig, WaveConfig, WindConfig


def build_wind_model(cfg: WindConfig) -> ConstantWind | GustWind | ShiftingWind:
    """Create a wind model from configuration."""
    if cfg.model == "gust":
        return GustWind(
            base_speed=cfg.speed,
            direction=cfg.direction,
            gust_intensity=cfg.gust_intensity,
            gust_tau=cfg.gust_tau,
            seed=cfg.gust_seed,
        )
    elif cfg.model == "shifting":
        return ShiftingWind(
            speed=cfg.speed,
            base_direction=cfg.direction,
            mode=cfg.shift_mode,
            rate=cfg.shift_rate,
            amplitude=cfg.shift_amplitude,
            period=cfg.shift_period,
        )
    else:
        return ConstantWind(cfg.speed, cfg.direction)


def build_current_model(cfg: CurrentConfig) -> NoCurrent | ConstantCurrent | TidalCurrent:
    """Create a current model from configuration."""
    if cfg.model == "constant":
        return ConstantCurrent(speed=cfg.speed, direction=cfg.direction)
    elif cfg.model == "tidal":
        return TidalCurrent(
            base_speed=cfg.speed,
            amplitude=cfg.tidal_amplitude,
            period=cfg.tidal_period,
            direction=cfg.direction,
            phase=cfg.tidal_phase,
        )
    else:
        return NoCurrent()


def build_wave_model(cfg: WaveConfig) -> NoWaves | SpectralWaves:
    """Create a wave model from configuration."""
    if cfg.model == "spectral" and cfg.Hs > 0:
        return SpectralWaves(
            Hs=cfg.Hs,
            Tp=cfg.Tp,
            direction=cfg.direction,
            n_components=cfg.n_components,
            spectrum=cfg.spectrum,
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
    else:
        return NoWaves()
