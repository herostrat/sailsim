# Stability Analysis — Control-Theoretic Autopilot Evaluation

This document describes the stability analysis tools in `sailsim.analysis`.
The module provides both **model-based** (analytical) and **data-driven**
(empirical) methods to evaluate autopilot performance beyond simple pass/fail
quality gates.

---

## 1. Motivation

The existing quality gates (`recording.analysis.evaluate_heading_hold`) answer
"did the simulation pass?" but not "how much margin is there before instability?"
or "how does robustness change with speed?". Classical control theory provides
quantitative answers through gain/phase margins, pole analysis, and frequency
response estimation.

Two complementary approaches are implemented:

| Mode | Input | Works with | Output |
|------|-------|-----------|--------|
| **Analytical** | `YachtConfig` + autopilot params | Nomoto only | Transfer functions, Bode/Nyquist, poles, margins |
| **Empirical** | Recorded JSON from `Recorder` | Any autopilot (Nomoto, pypilot, SignalK) | Step response metrics, spectral estimate, rudder activity |

---

## 2. Analytical Mode — Transfer Function Analysis

### 2.1 Plant Model (Nomoto)

The 1st-order Nomoto model relates rudder angle delta to heading psi:

```
G(s) = K / (s * (T*s + 1))
```

The `1/s` integrator converts yaw rate `r` to heading `psi`. Parameters `K`
(rudder gain) and `T` (time constant) are estimated from the yacht's sway-yaw
hydrodynamic derivatives via `estimate_nomoto_params()`.

The 2nd-order Nomoto model is also available:

```
G(s) = K * (T3*s + 1) / (s * (T1*s + 1) * (T2*s + 1))
```

### 2.2 Controller Model

The Nomoto autopilot uses a PID-like control law (see `nomoto.py` line 234):

```
delta = Kp * error + Ki * integral(error) - Kd * yaw_rate
```

As a transfer function:

```
C(s) = (Kd*s^2 + Kp*s + Ki) / s
```

Note: `Kd` acts on yaw rate `r` (measured), not on differentiated heading error.

### 2.3 Closed-Loop Analysis

- **Open-loop**: `L(s) = C(s) * G(s)`
- **Closed-loop**: `T(s) = L(s) / (1 + L(s))`

The characteristic polynomial of the closed-loop (with 1st-order Nomoto):

```
T*s^2 + (1 + K*Kd)*s + K*Kp = 0
```

The gains `Kp`, `Kd` are chosen by `_compute_gains()` to match a desired
2nd-order response `s^2 + 2*zeta*omega_n*s + omega_n^2 = 0`. The integral gain
is set to `Ki = omega_n * Kp / 10`.

### 2.4 Stability Margins

From the open-loop frequency response `L(jw)`:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Gain margin (GM)** | 1/|L(jw)| at phase = -180 deg | How much gain can increase before instability |
| **Phase margin (PM)** | 180 + angle(L(jw)) at |L|=1 | How much phase lag can increase before instability |
| **Delay margin** | PM / w_gc | Maximum tolerable loop delay [s] |
| **Bandwidth** | Frequency where |T(jw)| drops 3dB | Closed-loop speed of response |

Rules of thumb for good design: GM > 6 dB, PM > 30 deg.

### 2.5 Frozen-Point Speed Sweep

Nomoto K is speed-dependent (K proportional to U^2), making the system linear
parameter-varying (LPV). The `sweep_speed()` function performs frozen-point
analysis: compute margins at each speed independently. This is standard practice
in marine control (Fossen Ch. 12).

The speed sweep shows how stability margins degrade (or improve) across the
operating envelope.

### 2.6 Describing Function for Rate Limiter

The rudder rate limiter is a nonlinearity. For sinusoidal input
`delta(t) = A*sin(wt)` with rate limit R:

- If `A*w <= R`: no clipping, `N = 1`
- If `A*w > R`: gain < 1 with phase lag

On the Nyquist plot, limit cycles exist where `L(jw)` intersects `-1/N(A,w)`.
This predicts oscillations caused by rudder rate saturation.

---

## 3. Empirical Mode — Data-Driven Analysis

### 3.1 Step Response Extraction

`extract_step_responses()` detects setpoint changes in `target_heading` and
measures the heading response:

| Metric | How it is measured |
|--------|--------------------|
| Rise time | Time to reach 90% of the step |
| Settling time | Time after which error stays within 5% (min 1 deg) |
| Overshoot | Peak beyond target as % of step size |
| Steady-state error | Mean absolute error in last 20% of window |
| Peak rudder | Maximum absolute rudder angle during response |

Target heading is unwrapped before step detection. Heading error uses
`(error + 180) % 360 - 180` wrapping.

### 3.2 Spectral Transfer Function Estimation

`estimate_transfer_function()` estimates `H(f) = Sxy(f) / Sxx(f)` using the
Welch cross-spectral method:

- Input signal: rudder angle
- Output signal: heading
- Method: `scipy.signal.csd` / `scipy.signal.welch`
- Coherence: `gamma^2 = |Sxy|^2 / (Sxx * Syy)`

**Coherence as quality indicator:**
- `gamma^2 ~ 1`: linear model valid at this frequency
- `gamma^2 ~ 0`: nonlinearities or noise dominate

The empirical Bode plot can be overlaid with the analytical model to identify
where the linear approximation breaks down.

### 3.3 Rudder Activity Analysis

`analyze_rudder_activity()` computes:

- Rudder PSD via Welch method — identifies dominant oscillation frequencies
- Total RMS rudder angle
- Fraction of time at position limit (>= 95% of max)
- Fraction of time at rate limit (>= 95% of max rate)

High saturation fractions indicate the controller is demanding more than the
actuator can deliver, which degrades performance.

---

## 4. Plots

All plot functions are in `sailsim.analysis.plots` and return a matplotlib
`Figure`.

### Model-based

| Function | Description |
|----------|-------------|
| `plot_bode(result, which)` | Magnitude + phase, GM/PM annotations for open-loop |
| `plot_nyquist(result)` | Nyquist contour with (-1,0) point, optional describing function overlay |
| `plot_root_locus(result)` | Pole trajectories as gain varies |
| `plot_pole_zero_map(result)` | s-plane with damping lines and frequency circles |
| `plot_speed_sensitivity(sweep)` | 4-panel: K/T, gains, GM/PM, pole trajectories vs speed |

### Empirical

| Function | Description |
|----------|-------------|
| `plot_step_response(steps, recorder)` | Heading + rudder with rise time/overshoot annotations |
| `plot_estimated_bode(estimate, model_result)` | Empirical Bode + coherence + optional model overlay |
| `plot_rudder_spectrum(analysis)` | Rudder PSD with dominant frequency |
| `plot_saturation_summary(analysis)` | Bar chart of position/rate saturation |

---

## 5. Reports

Text summaries for terminal output:

- `summarize_linear(result)` — Nomoto params, gains, margins, poles, stability verdict
- `summarize_speed_sweep(sweep)` — Table with critical-value markers (! for GM < 6 dB or PM < 30 deg)
- `summarize_empirical(steps, rudder)` — Step response table + rudder activity stats

---

## 6. CLI Usage

```bash
# Single operating point (default yacht, 3 m/s)
python scripts/analyze_autopilot.py analytical --yacht default --speed 3.0

# Speed sweep 0.5-8 m/s
python scripts/analyze_autopilot.py analytical --yacht default --sweep

# Specific autopilot tuning
python scripts/analyze_autopilot.py analytical --yacht j24 --autopilot tack --speed 4.0

# Override design parameters
python scripts/analyze_autopilot.py analytical --omega-n 0.6 --zeta 0.7

# Analyse recorded simulation
python scripts/analyze_autopilot.py empirical recording.json

# Compare two recordings
python scripts/analyze_autopilot.py empirical nomoto.json pypilot.json

# Run simulation + both analyses
python scripts/analyze_autopilot.py full --scenario calm_heading_hold

# Save plots without display
python scripts/analyze_autopilot.py analytical --sweep --save-dir /tmp/plots --no-plot
```

---

## 7. Programmatic API

```python
from sailsim.analysis.linear import analyze_at_speed, sweep_speed
from sailsim.analysis.empirical import (
    extract_step_responses,
    estimate_transfer_function,
    analyze_rudder_activity,
)
from sailsim.analysis.plots import plot_bode, plot_nyquist
from sailsim.analysis.report import summarize_linear
from sailsim.core.config import YachtConfig

import numpy as np

# Analytical
yacht = YachtConfig()
result = analyze_at_speed(yacht, U=3.0, omega_n=0.5, zeta=0.8)
print(summarize_linear(result))
plot_bode(result, which="open_loop")

# Speed sweep
sweep = sweep_speed(yacht, np.linspace(1, 8, 16))

# Empirical (from recorded JSON)
from sailsim.recording.recorder import Recorder
rec = Recorder.from_json("recording.json")
steps = extract_step_responses(rec)
spectral = estimate_transfer_function(rec)
rudder = analyze_rudder_activity(rec)
```

---

## 8. Interpreting Results

### Typical output for default yacht at 3 m/s

```
Nomoto:  K = 0.789  T = 3.13 s
Gains:   Kp = 0.991  Kd = 1.902  Ki = 0.050
GM = inf dB   PM = 74.1 deg   BW = 0.67 rad/s
Poles: -0.370 +/- 0.262j  (zeta = 0.82, wn = 0.45)
System: STABLE
```

**Reading the margins:**
- `GM = inf`: No phase crossover — the system cannot go unstable from pure gain increase. This is typical for well-damped 2nd-order systems with integral action.
- `PM = 74 deg`: Generous phase margin. Values above 45 deg indicate robust stability.
- `BW = 0.67 rad/s`: Closed-loop bandwidth of about 0.1 Hz — appropriate for heading control (responds to disturbances below ~10 s period).

**Reading the poles:**
- Complex pair at `-0.37 +/- 0.26j` with damping zeta=0.82 — well-damped, no oscillatory tendency.
- All poles in the left half-plane — stable.

### Warning signs
- **PM < 30 deg**: Close to instability, expect oscillations
- **Large overshoot (> 20%)**: Reduce omega_n or increase zeta
- **High rudder saturation (> 30%)**: Controller demands exceed actuator capability
- **Low coherence at design bandwidth**: Nonlinearities dominate, linear model unreliable

---

## 9. Theory Background

### Nomoto Model

The Nomoto model is a reduced-order manoeuvring model. Starting from the
linearised sway-yaw equations (Fossen Ch. 7), the sway dynamics are eliminated
to get a scalar transfer function from rudder to yaw rate, which is then
augmented by a `1/s` integrator for heading.

Key properties:
- `K` is proportional to `U^2` (through rudder force)
- `T` is mainly hull geometry (nearly speed-independent in 1st-order approximation)
- The `1/s` integrator means the plant has zero DC gain margin — pure heading control requires integral action or velocity feedback

### Pole Placement

Given the characteristic polynomial `T*s^2 + (1+K*Kd)*s + K*Kp = 0`, matching
to the desired form `s^2 + 2*zeta*wn*s + wn^2 = 0` yields:

```
Kp = wn^2 * T / K
Kd = (2*zeta*wn*T - 1) / K
```

Both gains scale inversely with K, so as speed increases and K grows,
the gains automatically decrease (gain scheduling).

### References

1. Nomoto, K. et al. (1957). "On the steering qualities of ships." J. Zosen Kiokai.
2. Fossen, T.I. (2021). *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., Wiley. Ch. 7, 12.
3. Gelb, A. & Vander Velde, W.E. (1968). *Multiple-Input Describing Functions and Nonlinear System Design*, McGraw-Hill.
4. Astrom, K.J. & Hagglund, T. (2006). *Advanced PID Control*, ISA.
