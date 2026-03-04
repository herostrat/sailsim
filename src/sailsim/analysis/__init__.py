"""Control-theoretic stability analysis for autopilots.

Two modes:
- **Analytical** (model-based): builds transfer functions from Nomoto
  parameters, computes Bode/Nyquist/pole-zero, stability margins, speed sweep.
- **Empirical** (data-driven): analyses recorded simulations for step-response
  metrics, spectral estimation, rudder activity.
"""

from sailsim.analysis.linear import (
    ClosedLoopPoles,
    LinearAnalysisResult,
    SpeedSweepResult,
    StabilityMargins,
    analyze_at_speed,
    build_controller_tf,
    build_plant_tf,
    build_plant_tf_2nd,
    compute_margins,
    describing_function_rate_limiter,
    sweep_speed,
)

__all__ = [
    "ClosedLoopPoles",
    "LinearAnalysisResult",
    "SpeedSweepResult",
    "StabilityMargins",
    "analyze_at_speed",
    "build_controller_tf",
    "build_plant_tf",
    "build_plant_tf_2nd",
    "compute_margins",
    "describing_function_rate_limiter",
    "sweep_speed",
]
