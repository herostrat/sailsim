#!/usr/bin/env python3
"""Estimate sailing yacht hydrodynamic coefficients from basic dimensions.

Methods:
  - Added mass: Clarke et al. (1983) + Soeding (1982)
  - Damping: Clarke (1983) linear + cross-flow quadratic rule of thumb
  - Inertia: ITTC radius-of-gyration rules
  - Hydrostatics: standard naval architecture (Larsson & Eliasson)
  - 6-DOF extensions: scaling rules from strip-theory analogy

See docs/yacht_coefficient_estimation.md for full derivations and references.

Usage:
    python scripts/estimate_yacht_coefficients.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TextIO

RHO_SW = 1025.0  # seawater density [kg/m^3]
G = 9.81


@dataclass
class YachtDimensions:
    """Basic yacht dimensions — the input to coefficient estimation."""

    name: str
    description: str

    LOA: float  # length overall [m]
    LWL: float  # waterline length [m]
    BWL: float  # waterline beam [m]
    T_canoe: float  # canoe-body draft [m]
    T_total: float  # total draft incl. keel [m]
    mass: float  # displacement [kg]
    Cb: float  # block coefficient
    Cp: float  # prismatic coefficient

    sail_area: float  # upwind sail area [m^2]
    mast_height: float  # mast height above deck [m]
    # Sail CE position relative to CO [m]. Negative = aft.
    # Weather helm: sail_ce_x < keel_x (CE behind CLR).
    # Rule of thumb: lead = keel_x - sail_ce_x ≈ 3-5% of LWL.
    sail_ce_x: float = -0.20
    sail_ce_z: float = 0.0  # will be computed if zero

    keel_area: float = 1.0  # keel planform area [m^2]
    keel_span: float = 0.0  # will be computed from T_total - T_canoe
    # Keel CLR position relative to CO [m]. Positive = forward.
    keel_x: float = 0.15
    keel_z: float = 0.0  # will be computed

    rudder_area: float = 0.25  # rudder planform area [m^2]
    rudder_x: float = -4.5  # rudder longitudinal position [m]
    rudder_z: float = 0.0  # will be computed
    rudder_max: float = 0.52  # max deflection [rad] (~30 deg)
    rudder_cp_offset: float = 0.08  # stock-to-CP [m]

    ballast_fraction: float = 0.35  # fraction of mass in keel ballast
    Cwp: float = 0.68  # waterplane coefficient
    U_design: float = 3.0  # design speed [m/s]
    zeta_roll: float = 0.08  # roll damping ratio

    # Hydrostatics — use published/estimated values directly.
    # The standard BM formula (Cwp^2·L·B^3/12 / ∇) overestimates for
    # fine-lined yacht waterplanes by ~3-4x.  These should come from
    # inclining test data, published specifications, or expert estimate.
    GM_T: float = 1.0  # transverse metacentric height [m]
    GM_L: float = 0.0  # longitudinal (0 = auto-estimate)

    # Computed fields (filled by estimate())
    results: dict = field(default_factory=dict)


def estimate(d: YachtDimensions) -> dict:
    """Estimate all YachtConfig parameters from basic dimensions.

    The Clarke (1983) formulas use TOTAL draft (hull + keel), because for
    sailing yachts the keel is the dominant contributor to the lateral plane
    and must be included in the maneuvering derivatives.
    """
    L = d.LWL
    B = d.BWL
    T = d.T_total  # total draft for Clarke formulas
    m = d.mass
    rho = RHO_SW
    nabla = m / rho  # volume displacement [m^3]

    # Fill derived geometry
    if d.keel_span <= 0:
        d.keel_span = d.T_total - d.T_canoe
    if d.keel_z <= 0:
        d.keel_z = d.T_canoe + d.keel_span * 0.45
    if d.rudder_z <= 0:
        d.rudder_z = d.T_canoe * 0.6
    if d.sail_ce_z <= 0:
        d.sail_ce_z = 0.39 * d.mast_height + 1.0  # +1m boom height

    # ── Inertia ──────────────────────────────────────────
    k_xx = 0.38 * B
    k_yy = 0.27 * L
    k_zz = 0.25 * L

    Ix = m * k_xx**2
    Iy = m * k_yy**2
    Iz = m * k_zz**2

    # Parallel axis correction for ballast keel
    m_ballast = m * d.ballast_fraction
    d_keel = d.keel_span * 0.7  # effective depth of ballast bulb
    Ix += m_ballast * d_keel**2

    xg = 0.0
    zg = 0.15 * d.T_canoe  # conservative

    # ── Clarke (1983) Added Mass ─────────────────────────
    S = math.pi * (T / L) ** 2

    Y_vdot_p = -S * (1 + 0.16 * d.Cb * B / T - 5.1 * (B / L) ** 2)
    Y_rdot_p = -S * (0.67 * B / L - 0.0033 * (B / T) ** 2)
    N_vdot_p = -S * (1.1 * B / L - 0.041 * B / T)
    N_rdot_p = -S * (1 / 12 + 0.017 * d.Cb * B / T - 0.33 * B / L)

    Y_vdot = Y_vdot_p * 0.5 * rho * L**3
    Y_rdot = Y_rdot_p * 0.5 * rho * L**4
    N_vdot = N_vdot_p * 0.5 * rho * L**4
    N_rdot = N_rdot_p * 0.5 * rho * L**5

    # Hull fraction correction for added mass.
    # Clarke with T_total overestimates sway/yaw added mass because the keel
    # is included.  The Munk moment (Y_vdot - X_udot)*v*u in the Coriolis
    # matrix creates a destabilising yaw moment proportional to Y_vdot.
    # Since the simulator models the keel separately, we reduce Y_vdot and
    # N_rdot to hull-only values using hull_frac.
    hull_frac_mass = d.T_canoe / d.T_total
    Y_vdot *= hull_frac_mass
    N_rdot *= hull_frac_mass

    # Ensure correct signs and minimum magnitudes (added mass must resist motion).
    # Clarke extrapolation can underestimate for extreme yacht proportions.
    Y_vdot = min(Y_vdot, -0.5 * m)  # sway added mass >= 50% of hull mass
    N_rdot = min(N_rdot, -0.10 * Iz)  # yaw added inertia >= 10% of Iz

    # Cross-coupling added mass: cap to ensure positive-definite mass matrix.
    # |Y_rdot|, |N_vdot| <= 0.25 * sqrt(|Y_vdot * N_rdot|)
    max_cross_mass = 0.25 * math.sqrt(abs(Y_vdot * N_rdot))
    Y_rdot = max(Y_rdot, -max_cross_mass)
    N_vdot = max(N_vdot, -max_cross_mass)

    # Surge added mass (Soeding)
    X_udot = -2.7 * rho * nabla ** (5 / 3) / L**2
    # Clamp: should be between -3% and -15% of mass
    X_udot = max(X_udot, -0.15 * m)
    X_udot = min(X_udot, -0.03 * m)

    # 6-DOF extensions
    Z_wdot = min(Y_vdot * 0.8, -0.05 * m)
    K_pdot = -0.15 * Ix
    M_qdot = min(N_rdot * 0.9, -0.05 * Iy)

    # ── Clarke (1983) Linear Damping ─────────────────────
    U = d.U_design

    Y_v_p = -S * (1 + 0.4 * d.Cb * B / T)
    Y_r_p = -S * (-0.5 + 2.2 * B / L - 0.08 * B / T)
    N_v_p = -S * (0.5 + 2.4 * T / L)
    N_r_p = -S * (0.25 + 0.039 * B / T - 0.56 * B / L)

    Yv = Y_v_p * 0.5 * rho * L**2 * U
    Yr = Y_r_p * 0.5 * rho * L**3 * U
    Nv = N_v_p * 0.5 * rho * L**3 * U
    Nr = N_r_p * 0.5 * rho * L**4 * U

    # ── Hull Fraction Correction ───────────────────────
    # Clarke formulas with T_total include the keel in the "hull" derivatives.
    # Since the simulator models keel and rudder forces SEPARATELY (in
    # hydrodynamics.py), we must reduce the Clarke values to hull-only.
    # Damping scales roughly with lateral area^1.5 (flow interaction),
    # so hull_frac^1.5 gives good hull-only estimates.
    hull_frac = d.T_canoe / d.T_total
    Yv *= hull_frac
    Yr *= hull_frac
    Nv *= hull_frac
    Nr *= hull_frac

    # Ensure damping is dissipative (negative)
    Yv = min(Yv, -50.0)
    Nr = min(Nr, -100.0)

    # Cross-coupling cap: ensure Nomoto directional stability (Yv*Nr > Nv*Yr)
    # for hull-only terms.  Cap |Yr|, |Nv| ≤ 0.15 * sqrt(|Yv*Nr|).
    max_cross = 0.15 * math.sqrt(abs(Yv * Nr))
    Yr = max(Yr, -max_cross)
    Nv = max(Nv, -max_cross)

    # Surge damping
    T_surge = 20.0
    Xu = -(m - X_udot) / T_surge

    # ── Quadratic Damping ────────────────────────────────
    # Wetted surface (Larsson & Eliasson)
    Cm = d.Cb / d.Cp if d.Cp > 0 else 0.5
    if Cm > 0 and nabla > 0 and L > 0:
        S_wet = math.sqrt(nabla * L) * (0.65 / Cm) ** (1 / 3)
    else:
        S_wet = 2.5 * L * T  # fallback

    Re = L * U / 1.19e-6
    if Re > 0:
        Cf = 0.075 / (math.log10(Re) - 2) ** 2
    else:
        Cf = 0.003
    # Total resistance includes wave-making + pressure drag, not just friction.
    # For sailing yachts, residuary resistance is typically 5-10x frictional.
    # Use a total resistance factor of ~8 (Cf_total ≈ 8·Cf_friction).
    Cr_factor = 8.0
    Xuu = -0.5 * rho * S_wet * (1 + Cr_factor) * Cf

    # Quadratic terms from hull-fraction-corrected linear damping
    Yvv = -2.5 * abs(Yv) / max(U, 0.5)
    Yrr = -2.5 * abs(Yr) / max(U / L, 0.1)
    Nvv = -2.5 * abs(Nv) / max(U, 0.5)
    Nrr = -2.5 * abs(Nr) / max(U / L, 0.1)

    # ── Hydrostatics ─────────────────────────────────────
    # Use provided GM_T directly (computed BM overestimates for yacht shapes)
    GM_T = d.GM_T
    Aw = d.Cwp * L * B
    GM_L = d.GM_L if d.GM_L > 0 else _estimate_GM_L(L, B, d.T_canoe, m, d.Cwp)

    # ── 6-DOF Damping ───────────────────────────────────
    Zw = Yv * 0.8
    Mq = Nr * 0.9

    # Roll damping from target damping ratio
    # Kp = -zeta * 2 * sqrt(C44 * (Ix + |K_pdot|))
    C44 = rho * G * nabla * GM_T  # roll restoring stiffness
    Kp_mag = d.zeta_roll * 2 * math.sqrt(abs(C44) * (Ix + abs(K_pdot)))
    Kp = -Kp_mag

    Zww = Yvv * 0.8
    Kpp = -2.5 * abs(Kp) / max(U / B, 0.3)
    Mqq = Nrr * 0.9

    # ── Assemble results ─────────────────────────────────
    r = {
        "mass": _r(m),
        "Iz": _r(Iz),
        "xg": _r(xg),
        # Added mass
        "X_udot": _r(X_udot),
        "Y_vdot": _r(Y_vdot),
        "Y_rdot": _r(Y_rdot),
        "N_vdot": _r(N_vdot),
        "N_rdot": _r(N_rdot),
        # Linear damping
        "Xu": _r(Xu),
        "Yv": _r(Yv),
        "Yr": _r(Yr),
        "Nv": _r(Nv),
        "Nr": _r(Nr),
        # Quadratic damping
        "Xuu": _r(Xuu),
        "Yvv": _r(Yvv),
        "Yrr": _r(Yrr),
        "Nvv": _r(Nvv),
        "Nrr": _r(Nrr),
        # Sail
        "sail_area": _r(d.sail_area),
        "mast_height": _r(d.mast_height),
        "sail_ce_x": _r(d.sail_ce_x),
        # Rudder
        "rudder_area": _r(d.rudder_area),
        "rudder_x": _r(d.rudder_x),
        "rudder_max": _r(d.rudder_max),
        "rudder_cp_offset": _r(d.rudder_cp_offset),
        # Keel
        "keel_area": _r(d.keel_area),
        "keel_x": _r(d.keel_x),
        # 6-DOF
        "Ix": _r(Ix),
        "Iy": _r(Iy),
        "zg": _r(zg),
        "Z_wdot": _r(Z_wdot),
        "K_pdot": _r(K_pdot),
        "M_qdot": _r(M_qdot),
        "Zw": _r(Zw),
        "Kp": _r(Kp),
        "Mq": _r(Mq),
        "Zww": _r(Zww),
        "Kpp": _r(Kpp),
        "Mqq": _r(Mqq),
        "sail_ce_z": _r(d.sail_ce_z),
        "rudder_z": _r(d.rudder_z),
        "keel_z": _r(d.keel_z),
        "GM_T": _r(GM_T),
        "GM_L": _r(GM_L),
        "Aw": _r(Aw),
    }
    d.results = r
    return r


def _r(v: float) -> float:
    """Round to reasonable precision."""
    if abs(v) >= 1000:
        return round(v, 0)
    elif abs(v) >= 10:
        return round(v, 1)
    elif abs(v) >= 1:
        return round(v, 2)
    else:
        return round(v, 3)


def _estimate_GM_L(L, B, T, m, Cwp) -> float:
    """Estimate longitudinal metacentric height [m]."""
    rho = RHO_SW
    nabla = m / rho
    IL = Cwp * L**3 * B / 12.0
    BM_L = IL / nabla if nabla > 0 else 10.0
    KB = 0.58 * T
    KG = T * 0.6
    return KB + BM_L - KG


def write_toml(d: YachtDimensions, f: TextIO) -> None:
    """Write a yacht profile TOML file."""
    r = d.results
    if not r:
        r = estimate(d)

    f.write(f"# {d.name}\n")
    f.write(f"# {d.description}\n")
    f.write("#\n")
    f.write(f"# Estimated from: LOA={d.LOA}m, LWL={d.LWL}m, BWL={d.BWL}m,\n")
    f.write(f"#   T={d.T_total}m, Displacement={d.mass}kg, Cb={d.Cb}\n")
    f.write("# Method: Clarke (1983) + ITTC rules of thumb\n")
    f.write("# See docs/yacht_coefficient_estimation.md\n")
    f.write("\n")

    _w(f, "mass", r["mass"], "kg")
    _w(f, "Iz", r["Iz"], "yaw moment of inertia [kg*m^2]")
    _w(f, "xg", r["xg"], "longitudinal CG position [m]")
    f.write("\n# Added mass coefficients\n")
    _w(f, "X_udot", r["X_udot"], "")
    _w(f, "Y_vdot", r["Y_vdot"], "")
    _w(f, "Y_rdot", r["Y_rdot"], "")
    _w(f, "N_vdot", r["N_vdot"], "")
    _w(f, "N_rdot", r["N_rdot"], "")
    f.write("\n# Linear damping\n")
    _w(f, "Xu", r["Xu"], "")
    _w(f, "Yv", r["Yv"], "")
    _w(f, "Yr", r["Yr"], "")
    _w(f, "Nv", r["Nv"], "")
    _w(f, "Nr", r["Nr"], "")
    f.write("\n# Quadratic damping\n")
    _w(f, "Xuu", r["Xuu"], "")
    _w(f, "Yvv", r["Yvv"], "")
    _w(f, "Yrr", r["Yrr"], "")
    _w(f, "Nvv", r["Nvv"], "")
    _w(f, "Nrr", r["Nrr"], "")
    f.write("\n# Sail properties\n")
    _w(f, "sail_area", r["sail_area"], "m^2")
    _w(f, "mast_height", r["mast_height"], "m")
    _w(f, "sail_ce_x", r["sail_ce_x"], "longitudinal centre of effort [m]")
    f.write("\n# Rudder properties\n")
    _w(f, "rudder_area", r["rudder_area"], "m^2")
    _w(f, "rudder_x", r["rudder_x"], "longitudinal position [m]")
    _w(f, "rudder_max", r["rudder_max"], "max deflection [rad]")
    _w(f, "rudder_cp_offset", r["rudder_cp_offset"], "stock-to-centre-of-pressure [m]")
    f.write("\n# Keel properties\n")
    _w(f, "keel_area", r["keel_area"], "m^2")
    _w(f, "keel_x", r["keel_x"], "longitudinal position [m]")
    f.write("\n# --- 6-DOF extensions ---\n")
    _w(f, "Ix", r["Ix"], "roll moment of inertia [kg*m^2]")
    _w(f, "Iy", r["Iy"], "pitch moment of inertia [kg*m^2]")
    _w(f, "zg", r["zg"], "vertical CG above origin [m]")
    f.write("\n# 6-DOF added mass\n")
    _w(f, "Z_wdot", r["Z_wdot"], "")
    _w(f, "K_pdot", r["K_pdot"], "")
    _w(f, "M_qdot", r["M_qdot"], "")
    f.write("\n# 6-DOF linear damping\n")
    _w(f, "Zw", r["Zw"], "")
    _w(f, "Kp", r["Kp"], "")
    _w(f, "Mq", r["Mq"], "")
    f.write("\n# 6-DOF quadratic damping\n")
    _w(f, "Zww", r["Zww"], "")
    _w(f, "Kpp", r["Kpp"], "")
    _w(f, "Mqq", r["Mqq"], "")
    f.write("\n# Appendage depths (6-DOF moment arms)\n")
    _w(f, "sail_ce_z", r["sail_ce_z"], "sail CE height above WL [m]")
    _w(f, "rudder_z", r["rudder_z"], "rudder center depth below WL [m]")
    _w(f, "keel_z", r["keel_z"], "keel CLR depth below WL [m]")
    f.write("\n# Hydrostatics\n")
    _w(f, "GM_T", r["GM_T"], "transverse metacentric height [m]")
    _w(f, "GM_L", r["GM_L"], "longitudinal metacentric height [m]")
    _w(f, "Aw", r["Aw"], "waterplane area [m^2]")


def _w(f: TextIO, key: str, value: float, comment: str) -> None:
    """Write a TOML key = value line."""
    # Format: align values and comments
    val_str = f"{value}"
    if isinstance(value, float) and value == int(value) and abs(value) >= 1:
        val_str = f"{value:.1f}"
    line = f"{key} = {val_str}"
    if comment:
        line = f"{line:<30s}  # {comment}"
    f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Representative yacht definitions
# ═══════════════════════════════════════════════════════════════════════

YACHTS = [
    YachtDimensions(
        name="Mini 6.50",
        description="Production Class Mini 6.50 — ultra-light offshore racer",
        LOA=6.50,
        LWL=5.80,
        BWL=3.00,
        T_canoe=0.40,
        T_total=2.00,
        mass=950,
        Cb=0.18,
        Cp=0.52,
        sail_area=40.0,
        mast_height=11.0,
        sail_ce_x=-0.15,
        keel_area=0.45,
        keel_x=0.10,
        rudder_area=0.12,
        rudder_x=-2.8,
        rudder_cp_offset=0.05,
        ballast_fraction=0.42,
        Cwp=0.62,
        U_design=3.5,
        zeta_roll=0.06,
        GM_T=0.7,
        GM_L=8.0,
    ),
    YachtDimensions(
        name="J/24",
        description="J/24 one-design racing keelboat (7.3m, 1406 kg)",
        LOA=7.32,
        LWL=6.10,
        BWL=2.72,
        T_canoe=0.45,
        T_total=1.22,
        mass=1406,
        Cb=0.24,
        Cp=0.54,
        sail_area=32.0,
        mast_height=8.5,
        sail_ce_x=-0.15,
        keel_area=0.50,
        keel_x=0.10,
        rudder_area=0.10,
        rudder_x=-2.9,
        rudder_cp_offset=0.06,
        ballast_fraction=0.31,
        Cwp=0.65,
        U_design=2.8,
        zeta_roll=0.07,
        GM_T=0.8,
        GM_L=10.0,
    ),
    YachtDimensions(
        name="Dehler 34",
        description="Dehler 34 cruiser/racer (10.7m, 5950 kg, L-keel with bulb)",
        LOA=10.70,
        LWL=9.60,
        BWL=3.60,
        T_canoe=0.60,
        T_total=1.95,
        mass=5950,
        Cb=0.28,
        Cp=0.55,
        sail_area=65.0,
        mast_height=15.0,
        sail_ce_x=-0.20,
        keel_area=1.20,
        keel_x=0.15,
        rudder_area=0.25,
        rudder_x=-4.5,
        rudder_cp_offset=0.08,
        ballast_fraction=0.35,
        Cwp=0.68,
        U_design=3.0,
        zeta_roll=0.08,
        GM_T=1.1,
        GM_L=14.0,
    ),
    YachtDimensions(
        name="Swan 45",
        description="Nautor Swan 45 performance cruiser (13.8m, 9850 kg)",
        LOA=13.82,
        LWL=12.04,
        BWL=3.86,
        T_canoe=0.70,
        T_total=2.80,
        mass=9850,
        Cb=0.30,
        Cp=0.56,
        sail_area=113.0,
        mast_height=18.5,
        sail_ce_x=-0.25,
        keel_area=2.00,
        keel_x=0.20,
        rudder_area=0.35,
        rudder_x=-5.5,
        rudder_cp_offset=0.10,
        ballast_fraction=0.40,
        Cwp=0.68,
        U_design=3.5,
        zeta_roll=0.09,
        GM_T=1.3,
        GM_L=18.0,
    ),
    YachtDimensions(
        name="Hallberg-Rassy 62",
        description="Hallberg-Rassy 62 bluewater cruiser (18.9m, 33000 kg)",
        LOA=18.88,
        LWL=15.30,
        BWL=5.15,
        T_canoe=1.00,
        T_total=2.50,
        mass=33000,
        Cb=0.38,
        Cp=0.58,
        sail_area=176.0,
        mast_height=24.5,
        sail_ce_x=-0.30,
        keel_area=3.50,
        keel_x=0.25,
        rudder_area=0.55,
        rudder_x=-7.0,
        rudder_cp_offset=0.12,
        ballast_fraction=0.33,
        Cwp=0.72,
        U_design=3.5,
        zeta_roll=0.10,
        GM_T=1.6,
        GM_L=25.0,
    ),
    YachtDimensions(
        name="IMOCA 60",
        description="IMOCA 60 offshore racer (18.3m, 8500 kg, canting keel + foils)",
        LOA=18.28,
        LWL=17.50,
        BWL=5.85,
        T_canoe=0.50,
        T_total=4.50,
        mass=8500,
        Cb=0.16,
        Cp=0.51,
        sail_area=275.0,
        mast_height=28.0,
        sail_ce_x=-0.35,
        keel_area=1.50,
        keel_x=0.30,
        rudder_area=0.45,
        rudder_x=-8.5,
        rudder_cp_offset=0.10,
        ballast_fraction=0.36,
        Cwp=0.60,
        U_design=5.0,
        zeta_roll=0.06,
        GM_T=1.0,
        GM_L=20.0,
    ),
]


def main() -> None:
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "configs" / "yachts"
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["mini650", "j24", "dehler34", "swan45", "hr62", "imoca60"]

    for yacht, fname in zip(YACHTS, filenames, strict=True):
        estimate(yacht)
        path = out_dir / f"{fname}.toml"
        with path.open("w") as f:
            write_toml(yacht, f)
        print(f"Wrote {path}")

        # Print summary to stdout
        r = yacht.results
        print(f"  {yacht.name}: mass={r['mass']}, Iz={r['Iz']}, Nr={r['Nr']}, GM_T={r['GM_T']}")

    # Also print full table for comparison
    print("\n── Comparison Table ──")
    header = f"{'Parameter':<14s}"
    for y in YACHTS:
        header += f"  {y.name:>12s}"
    print(header)
    print("─" * len(header))
    for key in [
        "mass",
        "Iz",
        "Ix",
        "X_udot",
        "Y_vdot",
        "N_rdot",
        "Xu",
        "Yv",
        "Nr",
        "Nrr",
        "Kp",
        "GM_T",
        "GM_L",
        "Aw",
    ]:
        row = f"{key:<14s}"
        for y in YACHTS:
            v = y.results[key]
            if abs(v) >= 1000:
                row += f"  {v:>12.0f}"
            elif abs(v) >= 1:
                row += f"  {v:>12.1f}"
            else:
                row += f"  {v:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
