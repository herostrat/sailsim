"""6-DOF sailing yacht model (surge, sway, heave, roll, pitch, yaw).

Full Fossen 6-DOF rigid body model with hydrostatic restoring forces.
Extends the 3-DOF model with roll (heeling), pitch, and heave.
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
from sailsim.physics.aerodynamics import apparent_wind, sail_forces_6dof
from sailsim.physics.dynamics import (
    coriolis_matrix_6dof,
    damping_matrix_6dof,
    equations_of_motion_6dof,
    mass_matrix_6dof,
    rotation_matrix_6dof,
)
from sailsim.physics.hydrodynamics import keel_forces_6dof, rudder_forces_6dof
from sailsim.physics.hydrostatics import restoring_forces_6dof
from sailsim.physics.integration import rk4_step
from sailsim.physics.wave_forces import wave_forces_6dof

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class YachtParams6DOF:
    """Physical parameters for a 6-DOF sailing yacht model.

    Extends YachtParams with roll, pitch, heave parameters.
    """

    # Mass properties
    mass: float = 4000.0
    Ix: float = 8000.0  # roll moment of inertia [kg*m^2]
    Iy: float = 30000.0  # pitch moment of inertia [kg*m^2]
    Iz: float = 25000.0  # yaw moment of inertia [kg*m^2]
    xg: float = 0.0
    zg: float = 0.2  # vertical CG above CO [m]

    # Added mass (surge, sway, heave, roll, pitch, yaw)
    X_udot: float = -200.0
    Y_vdot: float = -4000.0
    Z_wdot: float = -4000.0
    K_pdot: float = -500.0
    M_qdot: float = -20000.0
    N_rdot: float = -15000.0
    Y_rdot: float = -400.0
    N_vdot: float = -400.0

    # Linear damping
    Xu: float = -100.0
    Yv: float = -2000.0
    Zw: float = -3000.0
    Kp: float = -5000.0  # strong roll damping (keel + hull)
    Mq: float = -30000.0  # pitch damping
    Nr: float = -20000.0
    Yr: float = -500.0
    Nv: float = -500.0

    # Quadratic damping
    Xuu: float = -50.0
    Yvv: float = -3000.0
    Zww: float = -3000.0
    Kpp: float = -3000.0
    Mqq: float = -20000.0
    Nrr: float = -50000.0

    # Sail properties
    sail_area: float = 50.0
    mast_height: float = 12.0
    sail_ce_x: float = 0.3
    sail_ce_z: float = 5.0  # CE height above WL [m]

    # Rudder properties
    rudder_area: float = 0.25
    rudder_x: float = -4.5
    rudder_z: float = 0.8  # rudder center depth below WL [m]
    rudder_max: float = 0.52

    # Keel properties
    keel_area: float = 1.5
    keel_x: float = -0.3
    keel_z: float = 1.5  # keel CLR depth below WL [m]

    # Hydrostatic
    GM_T: float = 1.2  # transverse metacentric height [m]
    GM_L: float = 15.0  # longitudinal metacentric height [m]
    Aw: float = 22.0  # waterplane area [m^2]


class Yacht6DOF:
    """6-DOF sailing yacht simulation model."""

    def __init__(self, params: YachtParams6DOF | None = None) -> None:
        self.params = params or YachtParams6DOF()
        self.state = VesselState()

        p = self.params
        self._M = mass_matrix_6dof(
            p.mass,
            p.Ix,
            p.Iy,
            p.Iz,
            p.xg,
            p.zg,
            p.X_udot,
            p.Y_vdot,
            p.Z_wdot,
            p.K_pdot,
            p.M_qdot,
            p.N_rdot,
            p.Y_rdot,
            p.N_vdot,
        )
        self._M_inv = np.linalg.inv(self._M)

    def reset(self, state: VesselState | None = None) -> None:
        """Reset the yacht to a given state (or zero)."""
        self.state = state or VesselState()

    def _compute_forces(
        self,
        nu6: NDArray[np.float64],
        eta6: NDArray[np.float64],
        wind: WindState,
        control: ControlCommand,
        waves: WaveState | None = None,
    ) -> NDArray[np.float64]:
        """Sum all external forces (6-DOF)."""
        p = self.params
        u, v, _w, _roll_p, _pitch_q, r = nu6
        psi = eta6[5]

        # Sail forces
        aw_speed, aw_angle = apparent_wind(
            wind.speed,
            wind.direction,
            u,
            v,
            psi,
        )
        tau_sail = sail_forces_6dof(
            aw_speed,
            aw_angle,
            p.sail_area,
            p.sail_ce_x,
            p.sail_ce_z,
            control.sail_trim,
        )

        # Rudder forces
        delta = np.clip(control.rudder_angle, -p.rudder_max, p.rudder_max)
        tau_rudder = rudder_forces_6dof(
            delta,
            u,
            v,
            r,
            p.rudder_area,
            p.rudder_x,
            p.rudder_z,
        )

        # Keel forces
        tau_keel = keel_forces_6dof(u, v, p.keel_area, p.keel_x, p.keel_z)

        tau = tau_sail + tau_rudder + tau_keel

        # Wave forces
        if waves is not None and waves.Hs > 0:
            tau += wave_forces_6dof(waves, psi, u)

        return tau

    def compute_forces(
        self,
        wind: WindState,
        control: ControlCommand,
        waves: WaveState | None = None,
    ) -> ForceData:
        """Compute force breakdown at current state (for recording)."""
        p = self.params
        nu6 = self.state.nu
        psi = self.state.psi
        u, v = nu6[0], nu6[1]
        r = nu6[5]

        aw_speed, aw_angle = apparent_wind(
            wind.speed,
            wind.direction,
            u,
            v,
            psi,
        )
        tau_sail = sail_forces_6dof(
            aw_speed,
            aw_angle,
            p.sail_area,
            p.sail_ce_x,
            p.sail_ce_z,
            control.sail_trim,
        )

        delta = np.clip(control.rudder_angle, -p.rudder_max, p.rudder_max)
        tau_rudder = rudder_forces_6dof(
            delta,
            u,
            v,
            r,
            p.rudder_area,
            p.rudder_x,
            p.rudder_z,
        )
        tau_keel = keel_forces_6dof(u, v, p.keel_area, p.keel_x, p.keel_z)

        tau_waves = np.zeros(6)
        if waves is not None and waves.Hs > 0:
            tau_waves = wave_forces_6dof(waves, psi, u)

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
        """Advance the yacht state by one time step (6-DOF)."""
        p = self.params
        eta6 = self.state.eta.copy()
        nu6 = self.state.nu.copy()

        # Current velocity in NED
        if current is not None and current.speed > 0:
            current_ned = np.array(
                [
                    current.speed * np.cos(current.direction),
                    current.speed * np.sin(current.direction),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        else:
            current_ned = np.zeros(6)

        # Combined state: [eta6, nu6] = 12 elements
        combined = np.concatenate([eta6, nu6])

        def derivatives(_t: float, state_vec: NDArray[np.float64]) -> NDArray[np.float64]:
            e6 = state_vec[:6]
            n6 = state_vec[6:]
            phi = e6[3]
            theta = e6[4]
            current_psi = e6[5]

            tau = self._compute_forces(n6, e6, wind, control, waves)

            # Hydrostatic restoring forces
            g_eta = restoring_forces_6dof(
                e6[2],
                phi,
                theta,
                p.mass,
                p.GM_T,
                p.GM_L,
                p.Aw,
            )

            # State-dependent matrices
            C = coriolis_matrix_6dof(
                p.mass,
                p.Ix,
                p.Iy,
                p.Iz,
                p.xg,
                p.zg,
                p.X_udot,
                p.Y_vdot,
                p.Z_wdot,
                p.K_pdot,
                p.M_qdot,
                p.N_rdot,
                n6,
            )
            D = damping_matrix_6dof(
                p.Xu,
                p.Yv,
                p.Zw,
                p.Kp,
                p.Mq,
                p.Nr,
                p.Xuu,
                p.Yvv,
                p.Zww,
                p.Kpp,
                p.Mqq,
                p.Nrr,
                p.Yr,
                p.Nv,
                n6,
            )

            nu_dot = equations_of_motion_6dof(self._M_inv, C, D, n6, tau, g_eta)  # type: ignore[arg-type]
            J = rotation_matrix_6dof(phi, theta, current_psi)
            eta_dot = J @ n6 + current_ned

            return np.concatenate([eta_dot, nu_dot])

        # Integrate
        new_combined = rk4_step(derivatives, 0.0, combined, dt)

        # Normalize angles to [-pi, pi]
        for i in [3, 4, 5]:  # phi, theta, psi
            new_combined[i] = np.arctan2(np.sin(new_combined[i]), np.cos(new_combined[i]))

        # Clamp velocities
        max_speed = 20.0
        max_heave = 5.0
        max_rot = 2.0
        new_combined[6] = np.clip(new_combined[6], -max_speed, max_speed)  # u
        new_combined[7] = np.clip(new_combined[7], -max_speed, max_speed)  # v
        new_combined[8] = np.clip(new_combined[8], -max_heave, max_heave)  # w
        new_combined[9] = np.clip(new_combined[9], -max_rot, max_rot)  # p
        new_combined[10] = np.clip(new_combined[10], -max_rot, max_rot)  # q
        new_combined[11] = np.clip(new_combined[11], -max_rot, max_rot)  # r

        self.state.eta[:] = new_combined[:6]
        self.state.nu[:] = new_combined[6:]
        return self.state
