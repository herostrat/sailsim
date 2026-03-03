"""3-DOF sailing yacht model (surge, sway, yaw).

Composes all physics modules into a complete vessel model that can be
stepped forward in time. The state vector is always 6-DOF for compatibility
with downstream code, but only surge (u), sway (v), and yaw (r) are active.

Coordinate conventions:
    - Body frame: x forward, y starboard, z down
    - NED frame: x north, y east, z down
    - Positive rudder angle → starboard turn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sailsim.core.types import (
    ControlCommand,
    CurrentState,
    ForceData,
    VesselState,
    WaveState,
    WindState,
)
from sailsim.physics.aerodynamics import apparent_wind, sail_forces_3dof
from sailsim.physics.dynamics import (
    coriolis_matrix_3dof,
    damping_matrix_3dof,
    equations_of_motion_3dof,
    mass_matrix_3dof,
    rotation_matrix_3dof,
)
from sailsim.physics.hydrodynamics import keel_forces_3dof, rudder_forces_3dof
from sailsim.physics.integration import rk4_step
from sailsim.physics.wave_forces import wave_forces_3dof

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class YachtParams:
    """Physical parameters for a 3-DOF sailing yacht model.

    Default values represent a typical ~10m (33ft) cruising yacht,
    ~4000 kg displacement.
    """

    # Mass properties
    mass: float = 4000.0  # total mass [kg]
    Iz: float = 25000.0  # yaw moment of inertia [kg*m^2]
    xg: float = 0.0  # CG offset from CO [m] (at origin for simplicity)

    # Added mass coefficients (negative by convention)
    X_udot: float = -200.0  # surge added mass [kg]
    Y_vdot: float = -4000.0  # sway added mass [kg]
    Y_rdot: float = -400.0  # sway-yaw coupling [kg*m]
    N_vdot: float = -400.0  # yaw-sway coupling [kg*m]
    N_rdot: float = -15000.0  # yaw added mass [kg*m^2] (deep keel)

    # Linear damping coefficients (negative by convention)
    Xu: float = -100.0  # surge linear damping [N*s/m]
    Yv: float = -2000.0  # sway linear damping [N*s/m] (keel resists leeway)
    Yr: float = -500.0  # sway-yaw linear damping [N*s/m*rad]
    Nv: float = -500.0  # yaw-sway linear damping [N*s/m]
    Nr: float = -20000.0  # yaw linear damping [N*m*s/rad] (deep keel)

    # Quadratic damping coefficients (negative by convention)
    Xuu: float = -50.0  # surge quadratic damping [N*s^2/m^2]
    Yvv: float = -3000.0  # sway quadratic damping [N*s^2/m^2]
    Yrr: float = -500.0  # [N*s^2/rad^2]
    Nvv: float = -500.0  # [N*s^2/m^2]
    Nrr: float = -50000.0  # yaw quadratic damping [N*m*s^2/rad^2]

    # Sail properties
    sail_area: float = 50.0  # total sail area (main + genoa) [m^2]
    mast_height: float = 12.0  # center of effort height [m]
    # Sail CE longitudinal position [m]. Negative = aft of CO.
    # Weather helm requires CE aft of keel CLR (sail_ce_x < keel_x).
    sail_ce_x: float = -0.20

    # Rudder properties
    rudder_area: float = 0.25  # rudder planform area [m^2]
    rudder_x: float = -4.5  # rudder position from CO [m] (negative = aft)
    rudder_max: float = 0.52  # max rudder deflection [rad] (~30°)

    # Keel properties
    keel_area: float = 1.5  # keel lateral area [m^2]
    # Keel CLR longitudinal position [m]. Positive = forward of CO.
    # Weather helm requires CLR forward of sail CE (keel_x > sail_ce_x).
    keel_x: float = 0.15


class Yacht3DOF:
    """3-DOF sailing yacht simulation model."""

    def __init__(self, params: YachtParams | None = None) -> None:
        self.params = params or YachtParams()
        self.state = VesselState()

        # Precompute constant matrices
        p = self.params
        self._M = mass_matrix_3dof(
            p.mass,
            p.Iz,
            p.xg,
            p.X_udot,
            p.Y_vdot,
            p.Y_rdot,
            p.N_vdot,
            p.N_rdot,
        )
        self._M_inv = np.linalg.inv(self._M)

    def reset(self, state: VesselState | None = None) -> None:
        """Reset the yacht to a given state (or zero)."""
        self.state = state or VesselState()

    def _extract_3dof(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract 3-DOF subvectors from full 6-DOF state."""
        eta3 = np.array([self.state.eta[0], self.state.eta[1], self.state.eta[5]])
        nu3 = np.array([self.state.nu[0], self.state.nu[1], self.state.nu[5]])
        return eta3, nu3

    def _write_3dof(self, eta3: NDArray[np.float64], nu3: NDArray[np.float64]) -> None:
        """Write 3-DOF results back into full 6-DOF state."""
        self.state.eta[0] = eta3[0]
        self.state.eta[1] = eta3[1]
        self.state.eta[5] = eta3[2]
        self.state.nu[0] = nu3[0]
        self.state.nu[1] = nu3[1]
        self.state.nu[5] = nu3[2]

    def _compute_forces(
        self,
        nu3: NDArray[np.float64],
        psi: float,
        wind: WindState,
        control: ControlCommand,
        waves: WaveState | None = None,
    ) -> NDArray[np.float64]:
        """Sum all external forces acting on the yacht."""
        p = self.params
        u, v, r = nu3

        # Sail forces
        aw_speed, aw_angle = apparent_wind(
            wind.speed,
            wind.direction,
            u,
            v,
            psi,
        )
        tau_sail = sail_forces_3dof(
            aw_speed,
            aw_angle,
            p.sail_area,
            p.sail_ce_x,
            control.sail_trim,
        )

        # Rudder forces (clamp rudder angle)
        delta = np.clip(control.rudder_angle, -p.rudder_max, p.rudder_max)
        tau_rudder = rudder_forces_3dof(
            delta,
            u,
            v,
            r,
            p.rudder_area,
            p.rudder_x,
        )

        # Keel forces
        tau_keel = keel_forces_3dof(u, v, p.keel_area, p.keel_x)

        tau = tau_sail + tau_rudder + tau_keel

        # Wave forces
        if waves is not None and waves.Hs > 0:
            tau += wave_forces_3dof(waves, psi, u)

        return tau

    def compute_forces(
        self,
        wind: WindState,
        control: ControlCommand,
        waves: WaveState | None = None,
    ) -> ForceData:
        """Compute force breakdown at current state (for recording).

        Pure read — does not modify state. Call before step().
        """
        _, nu3 = self._extract_3dof()
        psi = self.state.psi
        p = self.params
        u, v, r = nu3

        aw_speed, aw_angle = apparent_wind(
            wind.speed,
            wind.direction,
            u,
            v,
            psi,
        )
        tau_sail = sail_forces_3dof(
            aw_speed,
            aw_angle,
            p.sail_area,
            p.sail_ce_x,
            control.sail_trim,
        )

        delta = np.clip(control.rudder_angle, -p.rudder_max, p.rudder_max)
        tau_rudder = rudder_forces_3dof(
            delta,
            u,
            v,
            r,
            p.rudder_area,
            p.rudder_x,
        )
        tau_keel = keel_forces_3dof(u, v, p.keel_area, p.keel_x)

        tau_waves = np.zeros(3)
        if waves is not None and waves.Hs > 0:
            tau_waves = wave_forces_3dof(waves, psi, u)

        return ForceData(
            sail=tau_sail.copy(),
            rudder=tau_rudder.copy(),
            keel=tau_keel.copy(),
            waves=tau_waves.copy(),
        )

    def step(
        self,
        wind: WindState,
        control: ControlCommand,
        dt: float,
        current: CurrentState | None = None,
        waves: WaveState | None = None,
    ) -> VesselState:
        """Advance the yacht state by one time step.

        Args:
            wind: current wind conditions
            control: autopilot control command
            dt: time step [s]
            current: ocean current (affects position only, not forces)
            waves: wave state (adds wave forces)

        Returns:
            Updated vessel state.
        """
        p = self.params
        eta3, nu3 = self._extract_3dof()
        eta3[2]

        # Current velocity in NED (only affects kinematics)
        if current is not None and current.speed > 0:
            current_ned = np.array(
                [
                    current.speed * np.cos(current.direction),
                    current.speed * np.sin(current.direction),
                    0.0,
                ]
            )
        else:
            current_ned = np.zeros(3)

        # Build state-dependent matrices
        coriolis_matrix_3dof(
            p.mass,
            p.Iz,
            p.xg,
            p.X_udot,
            p.Y_vdot,
            p.Y_rdot,
            p.N_vdot,
            p.N_rdot,
            nu3,
        )
        damping_matrix_3dof(
            p.Xu,
            p.Yv,
            p.Yr,
            p.Nv,
            p.Nr,
            p.Xuu,
            p.Yvv,
            p.Yrr,
            p.Nvv,
            p.Nrr,
            nu3,
        )

        # Combined state: [eta3, nu3] = 6 elements
        combined = np.concatenate([eta3, nu3])

        def derivatives(_t: float, state_vec: NDArray[np.float64]) -> NDArray[np.float64]:
            e3 = state_vec[:3]
            n3 = state_vec[3:]
            current_psi = e3[2]

            tau = self._compute_forces(n3, current_psi, wind, control, waves)

            # Recompute C and D for current nu (important for RK4 substeps)
            C_local = coriolis_matrix_3dof(
                p.mass,
                p.Iz,
                p.xg,
                p.X_udot,
                p.Y_vdot,
                p.Y_rdot,
                p.N_vdot,
                p.N_rdot,
                n3,
            )
            D_local = damping_matrix_3dof(
                p.Xu,
                p.Yv,
                p.Yr,
                p.Nv,
                p.Nr,
                p.Xuu,
                p.Yvv,
                p.Yrr,
                p.Nvv,
                p.Nrr,
                n3,
            )

            nu_dot = equations_of_motion_3dof(self._M_inv, C_local, D_local, n3, tau)  # type: ignore[arg-type]
            R = rotation_matrix_3dof(current_psi)
            eta_dot = R @ n3 + current_ned  # current only in kinematics

            return np.concatenate([eta_dot, nu_dot])

        # Integrate
        new_combined = rk4_step(derivatives, 0.0, combined, dt)

        # Normalize heading to [-pi, pi]
        new_combined[2] = np.arctan2(np.sin(new_combined[2]), np.cos(new_combined[2]))

        # Clamp velocities to prevent numerical divergence
        max_speed = 20.0  # m/s
        max_yaw_rate = 1.0  # rad/s
        new_combined[3] = np.clip(new_combined[3], -max_speed, max_speed)
        new_combined[4] = np.clip(new_combined[4], -max_speed, max_speed)
        new_combined[5] = np.clip(new_combined[5], -max_yaw_rate, max_yaw_rate)

        self._write_3dof(new_combined[:3], new_combined[3:])
        return self.state
