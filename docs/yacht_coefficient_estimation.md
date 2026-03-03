# Yacht Coefficient Estimation — From Hull Dimensions to Simulation Parameters

This document describes the methods used to derive the `YachtConfig` parameter set
from basic yacht dimensions. All formulas are implemented in
`scripts/estimate_yacht_coefficients.py`.

---

## 1. Input Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| LOA | Length overall | m |
| LWL | Waterline length | m |
| BWL | Waterline beam | m |
| T | Canoe-body draft (excl. keel appendage) | m |
| T_total | Total draft (incl. keel) | m |
| m | Displacement mass | kg |
| Cb | Block coefficient | — |
| Cp | Prismatic coefficient | — |
| SA | Sail area upwind (main + jib) | m^2 |
| H_mast | Mast height above deck | m |
| A_keel | Keel planform area | m^2 |
| A_rudder | Rudder planform area | m^2 |
| x_keel | Keel longitudinal position from midship (neg = aft) | m |
| x_rudder | Rudder longitudinal position from midship (neg = aft) | m |

### Block Coefficient Ranges for Sailing Yachts

| Type | Cb range | Cp range |
|------|----------|----------|
| Ultra-light racer (IMOCA, Mini) | 0.15 – 0.22 | 0.50 – 0.54 |
| Light racer (J/24) | 0.20 – 0.28 | 0.52 – 0.56 |
| Cruiser/racer | 0.25 – 0.35 | 0.54 – 0.58 |
| Heavy cruiser | 0.35 – 0.42 | 0.56 – 0.62 |

Source: Larsson & Eliasson, *Principles of Yacht Design*, 4th ed.; Dudley Dix
yacht design coefficients.

---

## 2. Mass & Inertia

### 2.1 Moments of Inertia (Radius of Gyration)

ITTC Recommended Procedures 7.5-02-07-04.4 give the following rules of thumb,
refined for sailing yachts with ballast keels:

```
k_xx = 0.35..0.40 × BWL     (roll)
k_yy = 0.25..0.30 × LWL     (pitch)
k_zz = 0.24..0.26 × LWL     (yaw)
```

Moments of inertia:

```
Ix = m × k_xx^2
Iy = m × k_yy^2
Iz = m × k_zz^2
```

For yachts with deep, heavy ballast keels the parallel-axis theorem adds a
correction: `Ix_corr = Ix + m_ballast × d_keel^2`.

Source: ITTC 7.5-02-07-04.4; Ottosson & Brown (2011); Larsson & Eliasson.

### 2.2 Centre of Gravity Height

```
xg ≈ 0  (midship, fine for most yachts)
zg ≈ 0.1..0.3 × T  (above waterline origin; depends on ballast fraction)
```

---

## 3. Added Mass — Clarke et al. (1983)

The most widely used empirical regression for maneuvering added mass. Developed
for conventional ships (Cb 0.5–0.85), but extrapolates reasonably to yachts as a
starting estimate.

### 3.1 Scale Factor

```
S = π × (T/LWL)^2
```

### 3.2 Nondimensional Added Mass (Prime System)

```
Y'_vdot = -S × (1 + 0.16·Cb·BWL/T - 5.1·(BWL/LWL)^2)
Y'_rdot = -S × (0.67·BWL/LWL - 0.0033·(BWL/T)^2)
N'_vdot = -S × (1.1·BWL/LWL - 0.041·BWL/T)
N'_rdot = -S × (1/12 + 0.017·Cb·BWL/T - 0.33·BWL/LWL)
```

### 3.3 Dimensionalization and Hull Fraction

```
Y_vdot = Y'_vdot × 0.5·ρ·LWL^3
Y_rdot = Y'_rdot × 0.5·ρ·LWL^4
N_vdot = N'_vdot × 0.5·ρ·LWL^4
N_rdot = N'_rdot × 0.5·ρ·LWL^5
```

Since the simulator models keel forces separately, the diagonal added mass is
reduced to hull-only: `Y_vdot *= T_canoe/T_total`, `N_rdot *= T_canoe/T_total`.
This prevents the Munk moment `(Y_vdot - X_udot)·v·u` from being overestimated.

Cross-coupling added mass is capped to ensure a positive-definite mass matrix:
`|Y_rdot|, |N_vdot| ≤ 0.25·√(|Y_vdot·N_rdot|)`.

Minimum floors: `|Y_vdot| ≥ 0.5·m`, `|N_rdot| ≥ 0.10·Iz`.

### 3.4 Surge Added Mass

Soeding (1982):

```
X_udot = -2.7·ρ·∇^(5/3) / LWL^2
```

Simpler rule of thumb (Fossen):

```
X_udot ≈ -(0.05..0.10) × m
```

### 3.5 6-DOF Extensions

No well-established empirical set exists for heave/roll/pitch added mass of
sailing yachts. Rules of thumb from strip-theory analogy:

```
Z_wdot ≈ 0.8 × Y_vdot       (heave ≈ sway, slightly less for slender hull)
K_pdot ≈ -(0.10..0.25) × Ix  (roll added inertia)
M_qdot ≈ 0.9 × N_rdot        (pitch ≈ yaw added inertia)
```

Source: Clarke, Gedling & Hine (1983), RINA Transactions; Soeding (1982),
Schiffstechnik; Fossen MSS toolbox (github.com/cybergalactic/MSS).

---

## 4. Damping — Clarke et al. (1983) + Cross-Flow

### 4.1 Linear Damping (Hull Only, Nondimensional)

Same scale factor S = π·(T_total/LWL)^2:

```
Y'_v = -S × (1 + 0.4·Cb·BWL/T)
Y'_r = -S × (-0.5 + 2.2·BWL/LWL - 0.08·BWL/T)
N'_v = -S × (0.5 + 2.4·T/LWL)
N'_r = -S × (0.25 + 0.039·BWL/T - 0.56·BWL/LWL)
```

Dimensionalization (speed-dependent):

```
Yv = Y'_v × 0.5·ρ·LWL^2·U_design
Yr = Y'_r × 0.5·ρ·LWL^3·U_design
Nv = N'_v × 0.5·ρ·LWL^3·U_design
Nr = N'_r × 0.5·ρ·LWL^4·U_design
```

### 4.1.1 Hull Fraction Correction

**Important:** Clarke formulas with T_total include the keel contribution, but the
simulator models keel and rudder forces *separately* (`hydrodynamics.py`). To avoid
double-counting, all sway/yaw damping is reduced to hull-only values:

```
hull_frac = T_canoe / T_total
Yv *= hull_frac   (and similarly Yr, Nv, Nr)
```

This also reduces the Munk moment `(Y_vdot - X_udot)·v·u` in the Coriolis matrix,
which causes destabilising weather helm. The same hull_frac is applied to Y_vdot and
N_rdot.

Additionally, cross-coupling is capped for directional stability (Nomoto criterion):

```
max_cross = 0.15 × √(|Yv × Nr|)
|Yr| ≤ max_cross,  |Nv| ≤ max_cross
```

### 4.2 Surge Damping

From surge time constant T_surge ≈ 15–25 s:

```
Xu = -(m - X_udot) / T_surge
```

Quadratic surge from ITTC friction plus wave-making resistance:

```
Cf = 0.075 / (log10(Re) - 2)^2
Xuu = -0.5·ρ·S_wet·(1 + Cr_factor)·Cf
```

where Cr_factor ≈ 8 (residuary-to-friction resistance ratio; ITTC skin friction
alone underestimates total yacht drag by ~8–10×).

### 4.3 Quadratic Damping (Sway/Yaw)

Rule of thumb (from cross-flow drag analogy), using hull-fraction-corrected values:

```
Yvv = -2.5 × |Yv| / U_design
Yrr = -2.5 × |Yr| / (U_design/LWL)
Nvv = -2.5 × |Nv| / U_design
Nrr = -2.5 × |Nr| / (U_design/LWL)
```

Note: all quadratic damping coefficients are **negative** (dissipative).

### 4.4 6-DOF Damping

```
Zw  ≈ 0.8 × Yv
Kp  = -ζ_roll × 2 × √(ρ·g·GM_T·∇ × (Ix + |K_pdot|))
Mq  ≈ 0.9 × Nr

Zww ≈ 0.8 × Yvv
Kpp ≈ 2.5 × |Kp| / (U_design/BWL)
Mqq ≈ 0.9 × Nrr
```

Where ζ_roll ≈ 0.05–0.15 (roll damping ratio; higher for deep-keel yachts).

Source: Clarke et al. (1983); Fossen (2021), *Handbook of Marine Craft
Hydrodynamics*, Ch. 6–8; Hoerner (1965), *Fluid-Dynamic Drag*.

---

## 5. Appendage Forces

### 5.1 Keel Lift (Helmbold Formula)

For a finite-span foil mounted on the hull (mirror-image effect doubles AR):

```
AR_geo = span^2 / A_keel
AR_eff = 2 × AR_geo

dCL/dα = 2π·AR_eff / (2 + √(AR_eff^2 + 4))
```

The keel contribution to Yv is essentially:

```
Yv_keel = -0.5·ρ·U·A_keel·(dCL/dα)
Nv_keel = Yv_keel × x_keel
```

### 5.2 Rudder Force

Normal force coefficient (Fujii/Molland):

```
CN = 6.13·AR_eff / (AR_eff + 2.25)
```

Rudder forces at deflection δ:

```
F_N = 0.5·ρ·A_rudder·CN·U_R^2·sin(δ)
Y_rudder = -(1 + a_H)·F_N·cos(δ)         a_H ≈ 0.3–0.5
N_rudder = -(x_rudder + a_H·x_H)·F_N·cos(δ)
```

### 5.3 Aspect Ratios (Typical Values)

| Keel type | AR_geo | AR_eff | dCL/dα [/rad] |
|-----------|--------|--------|---------------|
| Long shallow | 0.5–1.5 | 1.0–3.0 | 1.5–3.5 |
| Moderate fin | 1.5–3.0 | 3.0–6.0 | 3.5–4.8 |
| High-perf fin | 3.0–5.0 | 6.0–10.0 | 4.8–5.5 |
| Spade rudder | 2.0–4.0 | 4.0–8.0 | 4.0–5.3 |

Source: Whicker & Fehlner (1958), DTMB Report 933; Larsson & Eliasson.

---

## 6. Hydrostatics

### 6.1 Transverse Metacentric Height (GM_T)

**Recommended:** Use published or measured GM_T values directly. The standard BM
formula (`IT = Cwp²·L·B³/12`) overestimates by 3–4× for fine-lined yacht waterplanes
because Cwp poorly represents the transverse second moment of a narrow, pointed
waterplane shape.

Fallback estimate (for reference only):

```
IT = Cwp^2 × LWL × BWL^3 / 12       (overestimates for yachts!)
∇ = m / ρ
BM = IT / ∇
KB ≈ 0.58 × T
KG ≈ T - ballast_fraction × keel_depth
GM_T = KB + BM - KG
```

Typical GM_T ranges for sailing yachts: 0.6–1.8 m.

### 6.2 Longitudinal Metacentric Height (GM_L)

```
IL = Cwp × LWL^3 × BWL / 12
BM_L = IL / ∇
GM_L = KB + BM_L - KG               (typically 10–30 × GM_T)
```

### 6.3 Waterplane Area

```
Aw = Cwp × LWL × BWL
```

### 6.4 Wetted Surface Area

Larsson & Eliasson:

```
Cm = Cb / Cp
S_wet = √(∇ × LWL) × (0.65/Cm)^(1/3)
```

Source: Larsson & Eliasson; standard naval architecture (Rawson & Tupper).

---

## 7. Roll Damping — Ikeda Decomposition

For sailing yachts, the keel appendage damping dominates:

```
B_keel = 0.5·ρ·U·A_keel·(dCL/dα)·z_keel^2·ω_roll
B_rudder = 0.5·ρ·U·A_rudder·(dCL/dα_r)·z_rudder^2·ω_roll
```

Natural roll frequency:

```
ω_n = √(ρ·g·∇·GM_T / (Ix + |K_pdot|))
T_roll = 2π / ω_n ≈ 2·BWL / √GM_T   (Larsson rule of thumb)
```

Typical sailing yacht roll damping ratio: ζ = 0.05–0.15.

Source: Ikeda, Himeno & Tanaka (1978); ITTC 7.5-02-07-04.5; Kawahara et al.
(2011).

---

## 8. Sail Forces

### Lift/Drag Model

Combined mainsail + jib CL/CD vs apparent wind angle (AWA), derived from
Hazen/ORC VPP data:

| AWA [deg] | CL | CD |
|-----------|------|------|
| 20 | 0.90 | 0.04 |
| 27 | 1.20 | 0.05 |
| 40 | 1.50 | 0.08 |
| 60 | 1.45 | 0.17 |
| 90 | 1.00 | 0.40 |
| 120 | 0.50 | 0.70 |
| 150 | 0.25 | 0.90 |
| 180 | 0.00 | 1.10 |

Centre of effort height:

```
z_CE ≈ 0.39 × mast_height + boom_height
```

Source: Marchaj (1979), *Aero-Hydrodynamics of Sailing*; Hazen (1980), SNAME
CSYS; ORC VPP Documentation 2023.

---

## 9. Caveats and Limitations

1. **Clarke (1983) extrapolation**: The regression was developed for ships with
   Cb = 0.5–0.85 and L/B = 5–8. Sailing yachts (Cb = 0.15–0.40, L/B = 2.5–4)
   are outside this range. Results are starting estimates only.

2. **Hull fraction correction**: Since the simulator models keel and rudder
   separately, all Clarke derivatives are reduced by `hull_frac = T_canoe / T_total`
   to avoid double-counting the keel's contribution. Cross-coupling is further
   capped for directional stability. See Sections 3.3 and 4.1.1.

3. **Munk moment**: The Coriolis term `(Y_vdot - X_udot)·v·u` creates a
   destabilising yaw moment (weather helm tendency). With uncorrected Y_vdot from
   Clarke+T_total, this can be 5–10× too strong. The hull_frac correction on
   Y_vdot keeps the Munk moment proportional to yacht mass.

4. **Speed dependence**: The linear damping from Clarke scales with U_design.
   At low speeds, quadratic (cross-flow) damping dominates. The simulator uses
   both linear and quadratic terms, which partially compensates.

5. **No appendage interaction**: The formulas treat hull, keel, and rudder
   independently. In reality, keel downwash affects rudder efficiency and the
   hull boundary layer modifies appendage lift.

6. **6-DOF extensions are approximate**: No empirical regression comparable to
   Clarke exists for heave/roll/pitch of sailing yachts. The scaling rules
   (Z ≈ 0.8×Y, M ≈ 0.9×N) are order-of-magnitude.

7. **Per-yacht autopilot tuning required**: Different yachts have different
   rudder authority (rudder_area × |rudder_x| / Iz). Larger yachts with higher
   inertia need higher PID gains. Ultra-light racing yachts (Mini 6.50, IMOCA 60)
   may capsize in 6-DOF due to extreme sail-area-to-displacement ratios.

8. **Tuning validation**: Estimates should be checked against known behavior —
   turning circle (2–4 boat lengths tactical diameter), roll period, hull speed,
   weather helm tendency.

---

## 10. References

1. Clarke, D., Gedling, P. & Hine, G. (1983). "The Application of Manoeuvring
   Criteria in Hull Design Using Linear Theory," RINA Transactions.
2. Soeding, H. (1982). "Prediction of Ship Steering Capabilities,"
   Schiffstechnik.
3. Fossen, T.I. (2021). *Handbook of Marine Craft Hydrodynamics and Motion
   Control*, 2nd ed., Wiley.
4. Keuning, J.A. & Sonnenberg, U.B. (1998). "Approximation of the Hydrodynamic
   Forces on a Sailing Yacht based on the DSYHS," TU Delft Report 1175-P.
5. Keuning, J.A. & Katgert, M. (2008). "A Bare Hull Resistance Prediction
   Method Derived from the DSYHS Extended to Higher Speeds."
6. Larsson, L. & Eliasson, R. (2014). *Principles of Yacht Design*, 4th ed.,
   Bloomsbury.
7. Angelou, M. & Spyrou, K.J. (2017). "A New Mathematical Model for
   Investigating Course Stability and Maneuvering Motions of Sailing Yachts,"
   J. Sailing Technology, 2(01).
8. Whicker, L.F. & Fehlner, L.F. (1958). "Free-stream Characteristics of a
   Family of Low-Aspect-Ratio Control Surfaces," DTMB Report 933.
9. Ikeda, Y., Himeno, Y. & Tanaka, N. (1978). "Components of Roll Damping of
   Ship at Forward Speed," J. SNAJ, Vol. 143.
10. ITTC Recommended Procedures 7.5-02-07-04.4 (radius of gyration),
    7.5-02-07-04.5 (roll damping estimation).
11. Marchaj, C.A. (1979/2000). *Aero-Hydrodynamics of Sailing*, 3rd ed.
12. ORC VPP Documentation (2023), orc.org.
13. Hoerner, S.F. (1965). *Fluid-Dynamic Drag*.
14. Dudley Dix Yacht Design — hull coefficient tables, dixdesign.com.
