# sailsim — Sailing Yacht Autopilot Simulator

Note: This project is in early development. The core physics engine and viewer are functional, but the autopilot and scenario library are still being built out. Contributions and feedback are welcome!
A physics-based simulation framework for developing and testing sailing yacht autopilots. It models the coupled dynamics of wind, waves, current, and vessel response so that autopilot controllers can be evaluated under realistic conditions — without going on the water.

## Purpose

Autopilot tuning on a real sailboat is slow, weather-dependent, and hard to reproduce. This simulator provides a deterministic, repeatable environment where you can:

- **Compare controller tunings** side-by-side under identical conditions
- **Run regression tests** — scenarios with quality gates catch regressions automatically
- **Explore edge cases** — heavy gusts, beam seas, tacking/gybing — that are dangerous or rare in practice
- **Visualize everything** — an interactive playback viewer shows heading, forces, roll, wind, and more

## What is Simulated

### Vessel Dynamics

The core physics engine solves the rigid-body equations of motion in the body-fixed frame. Two fidelity levels are available:

| Mode | Degrees of Freedom | States | Use Case |
|------|-------------------|--------|----------|
| **3-DOF** | Surge, Sway, Yaw (x, y, psi) | Position + heading in the horizontal plane | Fast iteration, controller tuning |
| **6-DOF** | + Heave, Roll, Pitch (z, phi, theta) | Full spatial motion including heel and trim | Realistic dynamics, wave response |

Both modes use Fossen's marine vessel convention with added-mass, linear damping, and quadratic damping terms.

### Force Models

| Component | What it Computes |
|-----------|-----------------|
| **Sail aerodynamics** | Lift/drag from apparent wind via flat-plate CL/CD model, applied at the sail's centre of effort |
| **Rudder hydrodynamics** | Lift/drag on the rudder blade from local flow, including boat speed and yaw rate contributions |
| **Keel hydrodynamics** | Lateral resistance from the keel, modelled with the same lift/drag approach as the rudder |
| **Wave forces** | 1st-order wave excitation from a spectral sea state (JONSWAP or Pierson-Moskowitz), resolved per frequency component |
| **Hydrostatics** (6-DOF) | Restoring forces in heave, roll, and pitch following Fossen's g(eta) formulation |

### Environment

- **Wind**: constant, gust (Ornstein-Uhlenbeck process), or shifting (linear / sinusoidal direction changes)
- **Waves**: spectral model with configurable Hs, Tp, direction, and random seed
- **Current**: none, constant, or tidal (sinusoidal speed variation)

### Autopilot

A PID heading controller with configurable gains (Kp, Ki, Kd). Optional automatic sail trim adjusts the sheet based on apparent wind angle.

### Navigation

- **Heading hold** — maintain a fixed compass heading
- **Scheduled maneuvers** — heading changes at specific times to simulate tacking and gybing sequences
- **Waypoint following** — line-of-sight guidance steers the boat through an ordered list of waypoints, advancing to the next when within a configurable tolerance radius

## Installation

Requires Python 3.12+.

```bash
git clone <repo-url>
cd sailing_autopilot_simultator
pip install -e ".[dev]"
```

## Usage

### Run a scenario

```bash
sailsim --scenario configs/scenarios/calm_heading_hold.toml
```

This runs the simulation, evaluates quality gates, and prints a pass/fail summary. Exit code 0 means all gates passed.

### Run and view interactively

```bash
sailsim --scenario configs/scenarios/rough_6dof.toml --view
```

Opens the playback viewer after the simulation completes. Use the timeline slider to scrub through the recording.

### Save and compare recordings

```bash
# Save two runs with different tunings
sailsim --scenario configs/scenarios/compare_conservative.toml --save-json /tmp/conservative.json
sailsim --scenario configs/scenarios/compare_aggressive.toml --save-json /tmp/aggressive.json

# Compare side-by-side in the viewer
sailsim --view /tmp/conservative.json /tmp/aggressive.json
```

### Run a waypoint route

```bash
sailsim --scenario configs/scenarios/waypoint_triangle.toml --save-json /tmp/triangle.json --view
```

Waypoints are defined in the scenario TOML:

```toml
[[route.waypoints]]
x = 100.0    # North [m]
y = 0.0      # East [m]
tolerance = 15.0  # reached when within 15 m

[[route.waypoints]]
x = 100.0
y = 100.0
tolerance = 15.0
```

### Export telemetry to CSV

```bash
sailsim --scenario configs/scenarios/calm_heading_hold.toml --output telemetry.csv
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--scenario PATH` | Load scenario from a TOML file |
| `--view [JSON...]` | Launch viewer. No args: run + view. With args: load and compare JSON files |
| `--save-json PATH` | Save recording as JSON |
| `--output PATH` | Export telemetry to CSV |
| `--quiet` | Suppress progress output |

## Viewer

The interactive playback viewer has three pages, switchable via buttons or keyboard (`1` / `2` / `3`):

**Steering** — Heading (with target), rudder angle, roll, speed through water, yaw rate.

**Environment** — True wind speed and direction, wave elevation, speed over ground, sail trim.

**Forces** — Per-component force breakdown (sail, rudder, keel, waves) for surge, sway, and yaw moment, plus pitch.

The left column shows the vessel trajectory with a desired track line (dotted), waypoint markers and tolerance circles (when applicable), attitude indicators (yacht cross-section views for roll, pitch, and yaw), and a numeric readout at the cursor position. Multiple recordings overlay with distinct colours for comparison.

## Scenarios

Scenarios are defined in TOML files under `configs/scenarios/`. Each file specifies the yacht model, environment, autopilot gains, initial conditions, and quality gates.

| Scenario | Description |
|----------|-------------|
| `calm_heading_hold.toml` | 3-DOF, light wind, basic heading hold |
| `moderate_beam_reach.toml` | 3-DOF, moderate beam reach |
| `rough_6dof.toml` | 6-DOF with gusts, waves (Hs=1.0m), and current |
| `tack_port_to_starboard.toml` | Tacking maneuver with scheduled heading change |
| `gybe_starboard_to_port.toml` | Gybing maneuver |
| `compare_conservative.toml` | 6-DOF, conservative PID (Kp=0.3, Ki=0.1, Kd=4.0) |
| `compare_aggressive.toml` | 6-DOF, aggressive PID (Kp=2.0, Ki=0.3, Kd=1.0) |
| `waypoint_triangle.toml` | 3 waypoints forming a triangle, line-of-sight guidance |

### Quality gates

Each scenario defines pass/fail thresholds:

- **Max heading deviation** — largest instantaneous error from target
- **Max settling time** — time until heading stays within tolerance
- **Max mean heading error** — average absolute error over the run
- **Max rudder rate** — limits actuator speed to prevent unrealistic control

## Project Structure

```
src/sailsim/
  autopilot/       PID controller and base class
  core/            Config loading, simulation runner, data types
  environment/     Wind, wave, and current models
  physics/         Aerodynamics, hydrodynamics, hydrostatics, wave forces, integration
  recording/       Time-series recorder, JSON/CSV export, quality gate analysis
  sensors/         Sensor data extraction from vessel state
  vessel/          3-DOF and 6-DOF yacht models
  viewer/          Interactive matplotlib playback viewer
  cli.py           Command-line interface

configs/
  scenarios/       TOML scenario definitions
  yachts/          Yacht parameter files

tests/
  unit/            Unit tests for individual modules
  integration/     Physics validation and recording tests
  scenarios/       End-to-end scenario tests with quality gates
```

## License

MIT
