"""Nomoto-based nautical course controller with pole placement.

Implements the 1st-order Nomoto model for yaw dynamics and derives
PID-like gains from pole placement (Fossen Ch. 7/12).  The Nomoto
parameters K and T are estimated from the yacht's sway-yaw
hydrodynamic derivatives, giving physically motivated gains that
automatically adapt to vessel speed.

References:
    Nomoto et al. (1957) "On the steering qualities of ships"
    Fossen (2021) "Handbook of Marine Craft Hydrodynamics and
        Motion Control" Ch. 7.2, 12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sailsim.core.types import ControlCommand, SensorData

if TYPE_CHECKING:
    from sailsim.core.config import YachtConfig

# Water density [kg/m^3]
RHO_WATER = 1025.0

# Lift-curve slope for the rudder [1/rad], consistent with
# hydrodynamics.py default cl_slope for rudder_forces_3dof.
CL_ALPHA_RUDDER = 3.0

# Wake fraction — effective inflow is u * WAKE_FACTOR,
# matching the 0.9 factor in rudder_forces_3dof.
WAKE_FACTOR = 0.9


@dataclass
class NomotoParams:
    """Result of Nomoto parameter estimation.

    Attributes:
        K: Steady-state rudder gain [1/s].
        T: 1st-order time constant [s].
        T1: Eigenvalue time constant 1 [s].
        T2: Eigenvalue time constant 2 [s].
        T3: Numerator zero time constant [s].
        U_ref: Reference speed at which these were computed [m/s].
    """

    K: float
    T: float
    T1: float
    T2: float
    T3: float
    U_ref: float


def estimate_nomoto_params(yacht: YachtConfig, U: float) -> NomotoParams:
    """Estimate 1st-order Nomoto K and T from yacht hydrodynamics.

    Builds the sway-yaw mass and damping sub-matrices from the yacht's
    hydrodynamic derivatives and solves for the 2nd-order Nomoto
    transfer function, then reduces to the standard 1st-order
    approximation.

    Args:
        yacht: Yacht configuration with hydrodynamic coefficients.
        U: Forward speed [m/s] (must be > 0).

    Returns:
        NomotoParams with K, T, T1, T2, T3 and reference speed.
    """
    m = yacht.mass
    xg = yacht.xg

    # --- Sway-Yaw mass sub-matrix M' = M_RB[1:,1:] + M_A[1:,1:] ---
    # Rigid body (sway-yaw block of Fossen M_RB):
    #   [[m,      m*xg],
    #    [m*xg,   Iz  ]]
    # Added mass (sway-yaw block, signs: M_A = -diag(derivatives)):
    #   [[-Y_vdot,  -Y_rdot],
    #    [-N_vdot,  -N_rdot]]
    M = np.array([
        [m - yacht.Y_vdot, m * xg - yacht.Y_rdot],
        [m * xg - yacht.N_vdot, yacht.Iz - yacht.N_rdot],
    ])

    # --- Damping sub-matrix D' (linearised at straight-line cruise) ---
    # D'[0,0] = -Yv
    # D'[0,1] = -(Yr - m*U)   (Coriolis linearisation: C_RB[1,2] = m*u)
    # D'[1,0] = -Nv
    # D'[1,1] = -Nr
    D = np.array([
        [-yacht.Yv, -(yacht.Yr - m * U)],
        [-yacht.Nv, -yacht.Nr],
    ])

    # --- Rudder force derivatives (linearised) ---
    u_eff = WAKE_FACTOR * U
    q = 0.5 * RHO_WATER * u_eff**2
    Y_delta = -q * yacht.rudder_area * CL_ALPHA_RUDDER
    N_delta = Y_delta * yacht.rudder_x

    # --- State-space: x_dot = A*x + B*delta ---
    # A = -M'^{-1} * D'
    # B =  M'^{-1} * [Y_delta, N_delta]^T
    M_inv = np.linalg.inv(M)
    A = -M_inv @ D

    # --- Eigenvalues → T1, T2 ---
    eigvals = np.linalg.eigvals(A)
    # Eigenvalues should be real and negative for a stable vessel
    lam = np.sort(np.real(eigvals))  # most negative first
    T1 = -1.0 / lam[0]
    T2 = -1.0 / lam[1]

    # --- Numerator zero → T3, and steady-state gain K ---
    # Transfer function r(s)/delta(s) via Cramer's rule:
    #   numerator = b1*s + b0
    #   b1 = M'[0,0]*N_delta - M'[1,0]*Y_delta
    #   b0 = D'[0,0]*N_delta - D'[1,0]*Y_delta
    b1 = M[0, 0] * N_delta - M[1, 0] * Y_delta
    b0 = D[0, 0] * N_delta - D[1, 0] * Y_delta

    T3 = b1 / b0
    det_D = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]
    K = b0 / det_D

    # --- 1st-order approximation ---
    T = T1 + T2 - T3

    return NomotoParams(K=K, T=T, T1=T1, T2=T2, T3=T3, U_ref=U)


def _compute_gains(
    K: float, T: float, omega_n: float, zeta: float,
) -> tuple[float, float, float]:
    """Pole-placement gains from Nomoto K, T and design specs.

    Closed-loop characteristic polynomial with 1st-order Nomoto:
        T*s^2 + (1 + K*Kd)*s + K*Kp = 0
    Matching to desired 2nd-order: s^2 + 2*zeta*omega_n*s + omega_n^2.

    Returns:
        (Kp, Kd, Ki) controller gains.
    """
    Kp = omega_n**2 * T / K
    Kd = (2.0 * zeta * omega_n * T - 1.0) / K
    Ki = omega_n * Kp / 10.0
    return Kp, Kd, Ki


class NomotoAutopilot:
    """Heading autopilot based on Nomoto model with pole placement.

    Gains are computed from yacht hydrodynamics and adapt to the
    current vessel speed (gain scheduling).  The controller also
    applies rudder rate limiting for realistic actuator behaviour.

    Implements ``AutopilotProtocol``.
    """

    def __init__(
        self,
        yacht: YachtConfig,
        omega_n: float = 0.4,
        zeta: float = 0.8,
        rudder_rate_max: float = 0.087,  # ~5 deg/s
        U_min: float = 0.5,
        auto_sail_trim: bool = False,
    ) -> None:
        self.yacht = yacht
        self.omega_n = omega_n
        self.zeta = zeta
        self.rudder_rate_max = rudder_rate_max
        self.rudder_max = yacht.rudder_max
        self.U_min = U_min
        self.auto_sail_trim = auto_sail_trim

        self.target_heading: float = 0.0
        self._integral: float = 0.0
        self._prev_rudder: float = 0.0

        # Pre-compute T once (speed-independent in 1st-order approx)
        params_ref = estimate_nomoto_params(yacht, U=2.0)
        self._T = params_ref.T

    def set_target_heading(self, heading: float) -> None:
        """Set the desired heading [rad] and reset the integrator."""
        self.target_heading = heading
        self._integral = 0.0

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Compute rudder command from sensor data.

        The Nomoto gain K is recomputed at the current speed for
        automatic gain scheduling.  T is kept constant (it depends
        mainly on hull geometry, not speed).
        """
        U = max(sensors.speed_through_water, self.U_min)

        # Speed-dependent K
        u_eff = WAKE_FACTOR * U
        q = 0.5 * RHO_WATER * u_eff**2
        Y_delta = -q * self.yacht.rudder_area * CL_ALPHA_RUDDER
        N_delta = Y_delta * self.yacht.rudder_x

        # Damping matrix determinant for K
        m = self.yacht.mass
        D00 = -self.yacht.Yv
        D01 = -(self.yacht.Yr - m * U)
        D10 = -self.yacht.Nv
        D11 = -self.yacht.Nr
        det_D = D00 * D11 - D01 * D10

        b0 = D00 * N_delta - D10 * Y_delta
        K = b0 / det_D

        # Pole-placement gains
        Kp, Kd, Ki = _compute_gains(K, self._T, self.omega_n, self.zeta)

        # Heading error (wrapped to [-pi, pi])
        error = self.target_heading - sensors.heading
        error = np.arctan2(np.sin(error), np.cos(error))

        # Integral with anti-windup
        self._integral += error * dt
        integral_max = 1.0
        self._integral = np.clip(self._integral, -integral_max, integral_max)

        # Control law: delta = Kp*error + Ki*integral - Kd*r
        rudder = Kp * error + Ki * self._integral - Kd * sensors.yaw_rate

        # Rudder position limit
        rudder = np.clip(rudder, -self.rudder_max, self.rudder_max)

        # Rudder rate limit
        max_change = self.rudder_rate_max * dt
        rudder_change = rudder - self._prev_rudder
        rudder_change = np.clip(rudder_change, -max_change, max_change)
        rudder = self._prev_rudder + rudder_change
        self._prev_rudder = float(rudder)

        # Sail trim
        if self.auto_sail_trim:
            from sailsim.physics.aerodynamics import optimal_sail_trim

            trim = optimal_sail_trim(sensors.apparent_wind_angle)
        else:
            trim = 0.5

        return ControlCommand(rudder_angle=float(rudder), sail_trim=trim)
