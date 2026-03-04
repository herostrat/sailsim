"""Interactive matplotlib-based playback viewer for simulation recordings.

Three-page layout (one parameter per panel, no twin axes):
  Page 1 (Steering): Heading+target, Rudder, Roll, STW, Yaw rate
  Page 2 (Environment): TWS, TWD, Wave elevation, SOG, Sail trim
  Page 3 (Forces): Force X, Force Y, Moment N, Pitch, Rudder torque + energy

Left column = trajectory map (top) + attitude indicators (middle) + readout (bottom).
Top = global run-color legend.
Bottom = timeline slider + play/pause/step/speed/page controls.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button, Slider

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.text import Text

    from sailsim.recording.recorder import Recorder

# Color palette for overlaying multiple runs
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

NUM_PAGES = 3
PAGE_LABELS = ["Steering", "Env", "Forces"]

# ── Yacht silhouette polygons (normalised, centred at origin) ──────────

# Bow view (Spantriss) — looking from forward, shows roll
HULL_BOW = np.array(
    [
        [-0.45, 0.00],  # deck port
        [-0.42, -0.08],
        [-0.35, -0.18],  # chine
        [-0.20, -0.32],
        [0.00, -0.38],  # keel root
        [0.20, -0.32],
        [0.35, -0.18],
        [0.42, -0.08],
        [0.45, 0.00],  # deck starboard
    ]
)
KEEL_BOW = np.array([[0.00, -0.38], [0.00, -0.68]])
MAST_BOW = np.array([[0.00, 0.00], [0.00, 0.62]])

# Side view (Seitenriss) — looking from starboard, shows pitch
HULL_SIDE = np.array(
    [
        [-0.48, 0.04],  # stern deck
        [-0.50, 0.00],  # transom
        [-0.48, -0.06],
        [-0.30, -0.16],
        [-0.05, -0.22],  # deepest hull
        [0.25, -0.16],
        [0.45, -0.04],  # forefoot
        [0.50, 0.06],  # stem head
        [0.42, 0.08],  # bow deck
        [-0.48, 0.04],  # back to stern
    ]
)
KEEL_SIDE = np.array(
    [
        [-0.08, -0.22],
        [-0.12, -0.55],
        [0.04, -0.55],
        [0.08, -0.22],
    ]
)
RUDDER_SIDE = np.array(
    [
        [-0.46, -0.06],
        [-0.48, -0.32],
        [-0.44, -0.32],
        [-0.42, -0.06],
    ]
)
MAST_SIDE = np.array([[0.08, 0.08], [0.08, 0.62]])

# Top view (Wasserlinienriss) — looking from above, shows yaw
# x = forward (north at 0° heading), y = starboard (east)
HULL_TOP = np.array(
    [
        [0.50, 0.00],  # bow tip
        [0.35, 0.12],
        [0.15, 0.20],
        [0.00, 0.22],  # max beam
        [-0.20, 0.20],
        [-0.40, 0.14],
        [-0.50, 0.08],  # stern starboard
        [-0.50, -0.08],  # stern port
        [-0.40, -0.14],
        [-0.20, -0.20],
        [0.00, -0.22],
        [0.15, -0.20],
        [0.35, -0.12],
    ]
)
KEEL_TOP = np.array([[0.10, 0.00], [-0.15, 0.00]])


@dataclass
class _AttitudeElements:
    """Typed container for attitude indicator artists per run."""

    bow_hull: MplPolygon
    bow_keel: Line2D
    bow_mast: Line2D
    side_hull: MplPolygon
    side_keel: MplPolygon
    side_rudder: MplPolygon
    side_mast: Line2D
    top_hull: MplPolygon
    top_keel: Line2D
    bow_label: Text
    side_label: Text
    top_label: Text


def _force_n_index(force_array) -> int:
    """Return the index of the N (yaw moment) component.

    3-DOF forces: [X, Y, N] -> index 2
    6-DOF forces: [X, Y, Z, K, M, N] -> index 5
    """
    return 2 if len(force_array) == 3 else 5


class PlaybackViewer:
    """Interactive playback viewer for one or more simulation recordings."""

    def __init__(self, runs: list[tuple[str, Recorder]]) -> None:
        self._runs = runs
        self._current_frame = 0
        self._playing = False
        self._max_frames = max(len(r.steps) for _, r in runs)
        self._updating_slider = False
        self._speed = 1
        self._active_page = 0  # 0=Steering, 1=Environment, 2=Forces

        self._build_figure()
        self._plot_static()
        self._create_cursors()

        self._anim = FuncAnimation(
            self._fig,
            self._update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Layout ────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        """Build figure with three-page time-series layout."""
        self._fig = plt.figure(figsize=(16, 10))

        # Global run-color legend at top
        for idx, (label, _) in enumerate(self._runs):
            color = COLORS[idx % len(COLORS)]
            self._fig.text(
                0.06 + idx * 0.18,
                0.975,
                f"\u25cf {label}",
                fontsize=10,
                color=color,
                fontweight="bold",
                va="top",
            )

        # Main grid: 2 columns
        outer = gridspec.GridSpec(
            1,
            2,
            figure=self._fig,
            width_ratios=[1, 1],
            left=0.06,
            right=0.97,
            top=0.955,
            bottom=0.15,
            wspace=0.25,
        )

        # Left: trajectory (top) + attitude indicators (middle) + readout (bottom)
        left = gridspec.GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=outer[0, 0],
            height_ratios=[5, 2.5, 1.5],
            hspace=0.12,
        )
        self._ax_traj = self._fig.add_subplot(left[0])

        # 3 attitude sub-axes: bow (roll), side (pitch), top (yaw)
        att_inner = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=left[1],
            wspace=0.15,
        )
        self._ax_bow = self._fig.add_subplot(att_inner[0])
        self._ax_side = self._fig.add_subplot(att_inner[1])
        self._ax_top = self._fig.add_subplot(att_inner[2])

        for ax in (self._ax_bow, self._ax_side, self._ax_top):
            ax.set_aspect("equal")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        self._ax_readout = self._fig.add_subplot(left[2])
        self._ax_readout.axis("off")

        # Right: 5 subplots for Page 1 (Steering)
        inner_right = gridspec.GridSpecFromSubplotSpec(
            5,
            1,
            subplot_spec=outer[0, 1],
            hspace=0.45,
        )
        self._ax_heading = self._fig.add_subplot(inner_right[0])
        self._ax_rudder = self._fig.add_subplot(inner_right[1], sharex=self._ax_heading)
        self._ax_roll = self._fig.add_subplot(inner_right[2], sharex=self._ax_heading)
        self._ax_stw = self._fig.add_subplot(inner_right[3], sharex=self._ax_heading)
        self._ax_yawrate = self._fig.add_subplot(inner_right[4], sharex=self._ax_heading)

        page1 = [
            self._ax_heading,
            self._ax_rudder,
            self._ax_roll,
            self._ax_stw,
            self._ax_yawrate,
        ]

        # Page 2 (Environment) — overlapping axes at same positions
        positions = [ax.get_position().bounds for ax in page1]

        self._ax_tws = self._fig.add_axes(positions[0])
        self._ax_twd = self._fig.add_axes(positions[1], sharex=self._ax_tws)
        self._ax_wave_el = self._fig.add_axes(positions[2], sharex=self._ax_tws)
        self._ax_sog = self._fig.add_axes(positions[3], sharex=self._ax_tws)
        self._ax_sailtrim = self._fig.add_axes(positions[4], sharex=self._ax_tws)

        page2 = [
            self._ax_tws,
            self._ax_twd,
            self._ax_wave_el,
            self._ax_sog,
            self._ax_sailtrim,
        ]

        # Page 3 (Forces) — overlapping axes at same positions
        self._ax_fx = self._fig.add_axes(positions[0])
        self._ax_fy = self._fig.add_axes(positions[1], sharex=self._ax_fx)
        self._ax_fn = self._fig.add_axes(positions[2], sharex=self._ax_fx)
        self._ax_pitch = self._fig.add_axes(positions[3], sharex=self._ax_fx)
        self._ax_torque = self._fig.add_axes(positions[4], sharex=self._ax_fx)
        self._ax_energy = self._ax_torque.twinx()

        page3 = [
            self._ax_fx,
            self._ax_fy,
            self._ax_fn,
            self._ax_pitch,
            self._ax_torque,
        ]

        self._page_axes = [page1, page2, page3]

        # Initially hide pages 2 and 3
        for ax in page2 + page3:
            ax.set_visible(False)
        self._ax_energy.set_visible(False)

        # Collect all time-series axes for cursor lines
        self._all_ts_axes = page1 + page2 + page3

        # Slider
        self._ax_slider = self._fig.add_axes((0.15, 0.055, 0.7, 0.025))
        self._slider = Slider(
            self._ax_slider,
            "Time",
            0,
            self._max_frames - 1,
            valinit=0,
            valstep=1,
        )
        self._slider.on_changed(self._on_slider)

        # Transport buttons
        self._ax_play = self._fig.add_axes((0.15, 0.01, 0.07, 0.035))
        self._ax_step_back = self._fig.add_axes((0.23, 0.01, 0.07, 0.035))
        self._ax_step_fwd = self._fig.add_axes((0.31, 0.01, 0.07, 0.035))
        self._ax_speed_btn = self._fig.add_axes((0.39, 0.01, 0.07, 0.035))

        self._btn_play = Button(self._ax_play, "Play")
        self._btn_step_back = Button(self._ax_step_back, "< Step")
        self._btn_step_fwd = Button(self._ax_step_fwd, "Step >")
        self._btn_speed = Button(self._ax_speed_btn, "1x")

        self._btn_play.on_clicked(self._toggle_play)
        self._btn_step_back.on_clicked(self._step_back)
        self._btn_step_fwd.on_clicked(self._step_fwd)
        self._btn_speed.on_clicked(self._cycle_speed)

        # Page switch buttons (3 buttons)
        self._page_buttons: list[Button] = []
        btn_x = 0.55
        for i, lbl in enumerate(PAGE_LABELS):
            ax_btn = self._fig.add_axes((btn_x + i * 0.09, 0.01, 0.08, 0.035))
            btn = Button(ax_btn, lbl)

            def _on_page(_event: object, page: int = i) -> None:
                self._switch_page(page)

            btn.on_clicked(_on_page)
            self._page_buttons.append(btn)
        self._page_buttons[0].label.set_fontweight("bold")

    # ── Static traces ─────────────────────────────────────────────────

    def _plot_static(self) -> None:
        """Plot all static traces (drawn once, one parameter per panel)."""
        has_target = any(r.steps[0].target_heading is not None for _, r in self._runs)

        for idx, (_label, rec) in enumerate(self._runs):
            color = COLORS[idx % len(COLORS)]
            times = [s.t for s in rec.steps]

            # ── Trajectory ──
            xs = [s.state.x for s in rec.steps]
            ys = [s.state.y for s in rec.steps]
            self._ax_traj.plot(ys, xs, color=color, alpha=0.8, lw=1)
            self._ax_traj.plot(ys[0], xs[0], "s", color=color, ms=5)

            # Desired track (Soll-Linie)
            dt_xs, dt_ys = rec.desired_track()
            if dt_xs:
                self._ax_traj.plot(
                    dt_ys,
                    dt_xs,
                    color=color,
                    ls=":",
                    lw=1.2,
                    alpha=0.4,
                    label="Desired trk" if idx == 0 else None,
                )

            # Waypoints and route (drawn once for first run that has route)
            if rec.route and idx == 0:
                wp_xs = [wp.x for wp in rec.route]
                wp_ys = [wp.y for wp in rec.route]
                # Start → first waypoint
                self._ax_traj.plot(
                    [ys[0], wp_ys[0]],
                    [xs[0], wp_xs[0]],
                    color="#aaaaaa",
                    ls="-.",
                    lw=1.0,
                    alpha=0.4,
                    zorder=2,
                )
                # Route lines between waypoints
                self._ax_traj.plot(
                    wp_ys,
                    wp_xs,
                    color="#aaaaaa",
                    ls="-.",
                    lw=1.0,
                    alpha=0.6,
                    zorder=2,
                )
                # Waypoint markers
                self._ax_traj.plot(
                    wp_ys,
                    wp_xs,
                    "D",
                    color="#ff4444",
                    ms=7,
                    alpha=0.8,
                    zorder=9,
                    label="Waypoints",
                )
                # Tolerance circles
                for wp in rec.route:
                    circle = plt.Circle(
                        (wp.y, wp.x),
                        wp.tolerance,
                        fill=False,
                        color="#ff4444",
                        ls="--",
                        lw=0.8,
                        alpha=0.4,
                    )
                    self._ax_traj.add_patch(circle)

            # ── Page 1: Steering ──

            # Heading + target (same parameter: actual vs. desired)
            headings = [np.degrees(s.sensors.heading) for s in rec.steps]
            self._ax_heading.plot(times, headings, color=color, lw=1)

            if rec.steps[0].target_heading is not None and idx == 0:
                targets = [
                    np.degrees(s.target_heading) for s in rec.steps if s.target_heading is not None
                ]
                target_times = [s.t for s in rec.steps if s.target_heading is not None]
                self._ax_heading.plot(
                    target_times,
                    targets,
                    color="gray",
                    ls="--",
                    lw=1.5,
                    alpha=0.7,
                    label="target",
                )

            # Rudder angle
            rudders = [np.degrees(s.control.rudder_angle) for s in rec.steps]
            self._ax_rudder.plot(times, rudders, color=color, lw=1)

            # Roll
            rolls = [np.degrees(s.state.phi) for s in rec.steps]
            self._ax_roll.plot(times, rolls, color=color, lw=1)

            # Speed through water
            stw = [s.sensors.speed_through_water for s in rec.steps]
            self._ax_stw.plot(times, stw, color=color, lw=1)

            # Yaw rate
            yaw_rates = [np.degrees(s.state.r) for s in rec.steps]
            self._ax_yawrate.plot(times, yaw_rates, color=color, lw=1)

            # ── Page 2: Environment ──

            # True wind speed
            tws = [s.wind.speed for s in rec.steps]
            self._ax_tws.plot(times, tws, color=color, lw=1)

            # True wind direction
            twd = [np.degrees(s.wind.direction) for s in rec.steps]
            self._ax_twd.plot(times, twd, color=color, lw=1)

            # Wave elevation
            if rec.steps[0].waves is not None:
                elevations = [s.waves.elevation if s.waves else 0.0 for s in rec.steps]
            else:
                elevations = [0.0] * len(rec.steps)
            self._ax_wave_el.plot(times, elevations, color=color, lw=1)

            # Speed over ground
            sog = [s.sensors.speed_over_ground for s in rec.steps]
            self._ax_sog.plot(times, sog, color=color, lw=1)

            # Sail trim
            trims = [s.control.sail_trim for s in rec.steps]
            self._ax_sailtrim.plot(times, trims, color=color, lw=1)

            # ── Page 3: Forces ──

            if rec.steps[0].forces is not None:
                for component, ls_style in [
                    ("sail", "-"),
                    ("rudder", "--"),
                    ("keel", ":"),
                    ("waves", "-."),
                ]:
                    vals_x = [getattr(s.forces, component)[0] for s in rec.steps]
                    self._ax_fx.plot(times, vals_x, color=color, ls=ls_style, lw=1, alpha=0.8)

                    vals_y = [getattr(s.forces, component)[1] for s in rec.steps]
                    self._ax_fy.plot(times, vals_y, color=color, ls=ls_style, lw=1, alpha=0.8)

                    vals_n = [
                        getattr(s.forces, component)[_force_n_index(getattr(s.forces, component))]
                        for s in rec.steps
                    ]
                    self._ax_fn.plot(times, vals_n, color=color, ls=ls_style, lw=1, alpha=0.8)

            # Pitch
            pitches = [np.degrees(s.state.theta) for s in rec.steps]
            self._ax_pitch.plot(times, pitches, color=color, lw=1)

            # Rudder torque + cumulative energy
            torques = [s.rudder_torque if s.rudder_torque is not None else 0.0 for s in rec.steps]
            self._ax_torque.plot(times, torques, color=color, lw=1)
            energy = np.cumsum(
                [
                    abs(t)
                    * abs(rec.steps[i].control.rudder_angle - rec.steps[i - 1].control.rudder_angle)
                    if i > 0
                    else 0.0
                    for i, t in enumerate(torques)
                ]
            )
            self._ax_energy.plot(
                times,
                energy,
                color=color,
                ls="--",
                lw=1,
                alpha=0.7,
            )

        # ── Trajectory decorations ──
        self._ax_traj.plot([], [], color="#17becf", lw=2.5, label="Wind")
        self._ax_traj.plot([], [], color="#ff6600", lw=2.5, label="Current")
        self._ax_traj.plot([], [], color="gray", ls="--", lw=1.5, label="Target hdg")
        self._ax_traj.set_xlabel("East [m]")
        self._ax_traj.set_ylabel("North [m]")
        self._ax_traj.set_aspect("equal", adjustable="datalim")
        self._ax_traj.grid(True, alpha=0.3)
        self._ax_traj.legend(fontsize=8, loc="lower left")

        # ── Page 1 labels ──
        self._ax_heading.set_ylabel("Heading [\u00b0]")
        self._ax_heading.grid(True, alpha=0.3)
        if has_target:
            self._ax_heading.legend(fontsize=7, loc="upper right")

        self._ax_rudder.set_ylabel("Rudder [\u00b0]")
        self._ax_rudder.grid(True, alpha=0.3)

        self._ax_roll.set_ylabel("Roll [\u00b0]")
        self._ax_roll.grid(True, alpha=0.3)

        self._ax_stw.set_ylabel("STW [m/s]")
        self._ax_stw.grid(True, alpha=0.3)

        self._ax_yawrate.set_ylabel("Yaw rate [\u00b0/s]")
        self._ax_yawrate.set_xlabel("Time [s]")
        self._ax_yawrate.grid(True, alpha=0.3)

        # ── Page 2 labels ──
        self._ax_tws.set_ylabel("TWS [m/s]")
        self._ax_tws.grid(True, alpha=0.3)

        self._ax_twd.set_ylabel("TWD [\u00b0]")
        self._ax_twd.grid(True, alpha=0.3)

        self._ax_wave_el.set_ylabel("Wave elev. [m]")
        self._ax_wave_el.grid(True, alpha=0.3)

        self._ax_sog.set_ylabel("SOG [m/s]")
        self._ax_sog.grid(True, alpha=0.3)

        self._ax_sailtrim.set_ylabel("Sail trim")
        self._ax_sailtrim.set_ylim(-0.05, 1.05)
        self._ax_sailtrim.set_xlabel("Time [s]")
        self._ax_sailtrim.grid(True, alpha=0.3)

        # ── Page 3 labels ──
        self._ax_fx.set_ylabel("Force X [N]")
        self._ax_fx.grid(True, alpha=0.3)
        style_handles = [
            Line2D([0], [0], color="gray", ls="-", lw=1, label="sail"),
            Line2D([0], [0], color="gray", ls="--", lw=1, label="rudder"),
            Line2D([0], [0], color="gray", ls=":", lw=1, label="keel"),
            Line2D([0], [0], color="gray", ls="-.", lw=1, label="waves"),
        ]
        self._ax_fx.legend(
            handles=style_handles,
            fontsize=7,
            loc="upper right",
            ncol=4,
            handlelength=2.0,
            framealpha=0.7,
        )

        self._ax_fy.set_ylabel("Force Y [N]")
        self._ax_fy.grid(True, alpha=0.3)

        self._ax_fn.set_ylabel("Moment N [Nm]")
        self._ax_fn.grid(True, alpha=0.3)

        self._ax_pitch.set_ylabel("Pitch [\u00b0]")
        self._ax_pitch.grid(True, alpha=0.3)

        self._ax_torque.set_ylabel("Torque [Nm]")
        self._ax_torque.set_xlabel("Time [s]")
        self._ax_torque.grid(True, alpha=0.3)
        self._ax_energy.set_ylabel("Energy [J]")

        # ── Attitude indicator static elements ──
        for ax, title in [(self._ax_bow, "Roll"), (self._ax_side, "Pitch"), (self._ax_top, "Yaw")]:
            # Coordinate crosshairs
            ax.plot([-0.9, 0.9], [0, 0], color="gray", lw=0.5, alpha=0.4)
            ax.plot([0, 0], [-0.9, 0.9], color="gray", lw=0.5, alpha=0.4)
            ax.set_title(title, fontsize=9, pad=3)

        # Waterline (blue dashed) for bow and side views
        for ax in (self._ax_bow, self._ax_side):
            ax.plot([-0.9, 0.9], [0, 0], color="#1f77b4", lw=1, ls="--", alpha=0.35, zorder=1)

        # Hide x-tick labels on upper axes of each page
        for page in self._page_axes:
            for ax in page[:-1]:
                plt.setp(ax.get_xticklabels(), visible=False)

    # ── Dynamic cursors & arrows ──────────────────────────────────────

    def _create_cursors(self) -> None:
        """Create cursor lines, arrows, target heading lines, and readout."""
        self._cursor_lines = []
        self._boat_arrows = []
        self._wind_arrows = []
        self._current_arrows = []
        self._target_lines = []

        for ax in self._all_ts_axes:
            line = ax.axvline(x=0, color="red", lw=1, alpha=0.6)
            self._cursor_lines.append(line)

        for idx, (_, _rec) in enumerate(self._runs):
            color = COLORS[idx % len(COLORS)]

            # Boat heading arrow
            boat = self._ax_traj.quiver(
                0,
                0,
                0,
                0,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.9,
                width=0.008,
                zorder=10,
                headwidth=3.5,
                headlength=4.5,
            )
            self._boat_arrows.append(boat)

            # Wind arrow (teal)
            w = self._ax_traj.quiver(
                0,
                0,
                0,
                0,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#17becf",
                alpha=0.7,
                width=0.005,
                zorder=8,
                headwidth=4,
                headlength=5,
            )
            self._wind_arrows.append(w)

            # Current arrow (orange)
            c = self._ax_traj.quiver(
                0,
                0,
                0,
                0,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#ff6600",
                alpha=0.7,
                width=0.005,
                zorder=8,
                headwidth=4,
                headlength=5,
            )
            c.set_visible(False)
            self._current_arrows.append(c)

            # Target heading line (dashed)
            (target_line,) = self._ax_traj.plot(
                [],
                [],
                color=color,
                ls="--",
                lw=1.5,
                alpha=0.5,
                zorder=7,
            )
            self._target_lines.append(target_line)

        # ── Attitude indicator patches per run ──
        self._attitude_patches: list[_AttitudeElements] = []
        for idx, (_, _rec) in enumerate(self._runs):
            color = COLORS[idx % len(COLORS)]
            alpha = max(0.3, 0.8 - 0.15 * idx)

            # Bow view
            bow_hull = MplPolygon(
                HULL_BOW.copy(),
                closed=False,
                fill=False,
                edgecolor=color,
                lw=2,
                alpha=alpha,
                zorder=5,
            )
            self._ax_bow.add_patch(bow_hull)
            (bow_keel,) = self._ax_bow.plot(
                KEEL_BOW[:, 0],
                KEEL_BOW[:, 1],
                color=color,
                lw=2.5,
                alpha=alpha,
                zorder=4,
            )
            (bow_mast,) = self._ax_bow.plot(
                MAST_BOW[:, 0],
                MAST_BOW[:, 1],
                color=color,
                lw=1.5,
                alpha=alpha,
                zorder=4,
            )

            # Side view
            side_hull = MplPolygon(
                HULL_SIDE.copy(),
                closed=True,
                fill=False,
                edgecolor=color,
                lw=2,
                alpha=alpha,
                zorder=5,
            )
            self._ax_side.add_patch(side_hull)
            side_keel = MplPolygon(
                KEEL_SIDE.copy(),
                closed=True,
                fill=True,
                facecolor=color,
                edgecolor=color,
                lw=1,
                alpha=alpha * 0.5,
                zorder=4,
            )
            self._ax_side.add_patch(side_keel)
            side_rudder = MplPolygon(
                RUDDER_SIDE.copy(),
                closed=True,
                fill=True,
                facecolor=color,
                edgecolor=color,
                lw=1,
                alpha=alpha * 0.5,
                zorder=4,
            )
            self._ax_side.add_patch(side_rudder)
            (side_mast,) = self._ax_side.plot(
                MAST_SIDE[:, 0],
                MAST_SIDE[:, 1],
                color=color,
                lw=1.5,
                alpha=alpha,
                zorder=4,
            )

            # Top view
            top_hull = MplPolygon(
                HULL_TOP.copy(),
                closed=True,
                fill=False,
                edgecolor=color,
                lw=2,
                alpha=alpha,
                zorder=5,
            )
            self._ax_top.add_patch(top_hull)
            (top_keel,) = self._ax_top.plot(
                KEEL_TOP[:, 0],
                KEEL_TOP[:, 1],
                color=color,
                lw=2,
                alpha=alpha,
                zorder=4,
            )

            # Angle labels (placed at bottom-right of each attitude axis)
            bow_label = self._ax_bow.text(
                0.95,
                -0.90,
                "",
                fontsize=8,
                color=color,
                ha="right",
                va="bottom",
                fontweight="bold",
            )
            side_label = self._ax_side.text(
                0.95,
                -0.90,
                "",
                fontsize=8,
                color=color,
                ha="right",
                va="bottom",
                fontweight="bold",
            )
            top_label = self._ax_top.text(
                0.95,
                -0.90,
                "",
                fontsize=8,
                color=color,
                ha="right",
                va="bottom",
                fontweight="bold",
            )

            self._attitude_patches.append(
                _AttitudeElements(
                    bow_hull=bow_hull,
                    bow_keel=bow_keel,
                    bow_mast=bow_mast,
                    side_hull=side_hull,
                    side_keel=side_keel,
                    side_rudder=side_rudder,
                    side_mast=side_mast,
                    top_hull=top_hull,
                    top_keel=top_keel,
                    bow_label=bow_label,
                    side_label=side_label,
                    top_label=top_label,
                )
            )

        # Value readout in dedicated panel below trajectory
        self._readout_text = self._ax_readout.text(
            0.0,
            1.0,
            "",
            transform=self._ax_readout.transAxes,
            fontsize=8.5,
            family="monospace",
            va="top",
            ha="left",
        )

        # Time readout near slider
        self._time_text = self._fig.text(
            0.88,
            0.015,
            "t = 0.00 s",
            fontsize=10,
            family="monospace",
        )

    @staticmethod
    def _rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
        """Rotate 2D points around the origin by *angle* radians."""
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        result: np.ndarray = points @ rot.T
        return result

    def _arrow_scale(self) -> float:
        """Arrow length scale based on trajectory extent."""
        xlim = self._ax_traj.get_xlim()
        ylim = self._ax_traj.get_ylim()
        extent = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), 10.0)
        return extent * 0.08

    def _update_cursors(self) -> None:
        """Move cursors, arrows, target lines, and readout to current frame."""
        ref_rec = self._runs[0][1]
        frame = min(self._current_frame, len(ref_rec.steps) - 1)
        t_now = ref_rec.steps[frame].t

        for line in self._cursor_lines:
            line.set_xdata([t_now, t_now])

        scale = self._arrow_scale()

        for idx, (_, rec) in enumerate(self._runs):
            f = min(self._current_frame, len(rec.steps) - 1)
            step = rec.steps[f]
            pos_e = step.state.y  # East = plot x
            pos_n = step.state.x  # North = plot y

            # Boat heading arrow
            heading = step.state.eta[5]
            hdg_e = np.sin(heading)
            hdg_n = np.cos(heading)
            b_len = scale * 0.7
            self._boat_arrows[idx].set_offsets([[pos_e, pos_n]])
            self._boat_arrows[idx].set_UVC(hdg_e * b_len, hdg_n * b_len)

            # Wind arrow
            tw_dir = step.wind.direction
            tw_spd = step.wind.speed
            wind_e = -np.sin(tw_dir)
            wind_n = -np.cos(tw_dir)
            w_len = scale * min(tw_spd / 10.0, 1.5)
            self._wind_arrows[idx].set_offsets([[pos_e, pos_n]])
            self._wind_arrows[idx].set_UVC(wind_e * w_len, wind_n * w_len)

            # Current arrow
            if step.current is not None and step.current.speed > 0:
                cur_dir = step.current.direction
                cur_e = np.sin(cur_dir)
                cur_n = np.cos(cur_dir)
                c_len = scale * min(step.current.speed / 2.0, 1.0)
                self._current_arrows[idx].set_offsets([[pos_e, pos_n]])
                self._current_arrows[idx].set_UVC(cur_e * c_len, cur_n * c_len)
                self._current_arrows[idx].set_visible(True)
            else:
                self._current_arrows[idx].set_visible(False)

            # Target heading line
            if step.target_heading is not None:
                tgt = step.target_heading
                tgt_e = np.sin(tgt)
                tgt_n = np.cos(tgt)
                t_len = scale * 1.5
                self._target_lines[idx].set_data(
                    [pos_e, pos_e + tgt_e * t_len],
                    [pos_n, pos_n + tgt_n * t_len],
                )
            else:
                self._target_lines[idx].set_data([], [])

            # ── Attitude indicators ──
            att = self._attitude_patches[idx]
            roll = step.state.phi  # positive = starboard down
            pitch = -step.state.theta  # negate so bow-up rotates visually up
            yaw = step.state.eta[5]

            # Bow view — rotate by roll
            rotated = self._rotate_2d(HULL_BOW, roll)
            att.bow_hull.set_xy(rotated)
            rk = self._rotate_2d(KEEL_BOW, roll)
            att.bow_keel.set_data(rk[:, 0], rk[:, 1])
            rm = self._rotate_2d(MAST_BOW, roll)
            att.bow_mast.set_data(rm[:, 0], rm[:, 1])

            # Side view — rotate by pitch
            rotated = self._rotate_2d(HULL_SIDE, pitch)
            att.side_hull.set_xy(rotated)
            rk = self._rotate_2d(KEEL_SIDE, pitch)
            att.side_keel.set_xy(rk)
            rr = self._rotate_2d(RUDDER_SIDE, pitch)
            att.side_rudder.set_xy(rr)
            rm = self._rotate_2d(MAST_SIDE, pitch)
            att.side_mast.set_data(rm[:, 0], rm[:, 1])

            # Top view — rotate by yaw
            rotated = self._rotate_2d(HULL_TOP, -yaw)
            att.top_hull.set_xy(rotated)
            rk = self._rotate_2d(KEEL_TOP, -yaw)
            att.top_keel.set_data(rk[:, 0], rk[:, 1])

            # Angle labels
            att.bow_label.set_text(f"{np.degrees(step.state.phi):+.1f}\u00b0")
            att.side_label.set_text(f"{np.degrees(step.state.theta):+.1f}\u00b0")
            att.top_label.set_text(f"{np.degrees(yaw):+.1f}\u00b0")

        self._time_text.set_text(f"t = {t_now:.2f} s")
        self._update_readout()

        self._updating_slider = True
        self._slider.set_val(self._current_frame)
        self._updating_slider = False

    # ── Value readout ─────────────────────────────────────────────────

    def _update_readout(self) -> None:
        """Update the value readout panel (page-dependent)."""
        parts = []
        for _idx, (label, rec) in enumerate(self._runs):
            f = min(self._current_frame, len(rec.steps) - 1)
            step = rec.steps[f]

            if len(self._runs) > 1:
                parts.append(f"\u2500\u2500 {label} \u2500\u2500")

            if self._active_page == 0:
                self._readout_steering(parts, step)
            elif self._active_page == 1:
                self._readout_environment(parts, step)
            else:
                self._readout_forces(parts, step)

        self._readout_text.set_text("\n".join(parts))

    def _readout_steering(self, parts: list[str], step) -> None:
        """Steering readout content."""
        heading_deg = np.degrees(step.sensors.heading)
        rudder_deg = np.degrees(step.control.rudder_angle)
        roll_deg = np.degrees(step.state.phi)
        stw = step.sensors.speed_through_water
        yaw_rate = np.degrees(step.state.r)

        target_str = ""
        if step.target_heading is not None:
            target_str = f"  Tgt {np.degrees(step.target_heading):+.1f}\u00b0"

        parts.append(
            f"Pos ({step.state.x:+.1f}, {step.state.y:+.1f})  "
            f"Hdg {heading_deg:+.1f}\u00b0{target_str}"
        )
        parts.append(
            f"Rud {rudder_deg:+.1f}\u00b0  Roll {roll_deg:+.1f}\u00b0  "
            f"STW {stw:.2f} m/s  r {yaw_rate:+.2f}\u00b0/s"
        )
        if step.active_waypoint_index is not None:
            parts.append(f"Waypoint: #{step.active_waypoint_index}")

    def _readout_environment(self, parts: list[str], step) -> None:
        """Environment readout content."""
        tw_spd = step.wind.speed
        tw_dir_deg = np.degrees(step.wind.direction)
        sog = step.sensors.speed_over_ground
        sail_trim = step.control.sail_trim
        wave_el = step.waves.elevation if step.waves else 0.0

        parts.append(f"Wind {tw_spd:.1f} m/s @ {tw_dir_deg:.0f}\u00b0  Wave {wave_el:+.3f} m")
        parts.append(f"SOG {sog:.2f} m/s  Trim {sail_trim:.2f}")
        if step.current is not None and step.current.speed > 0:
            parts.append(
                f"Current {step.current.speed:.2f} m/s "
                f"@ {np.degrees(step.current.direction):.0f}\u00b0"
            )

    def _readout_forces(self, parts: list[str], step) -> None:
        """Forces readout content."""
        if step.forces is not None:
            n_idx = _force_n_index(step.forces.sail)
            parts.append(
                f"Fx: sail={step.forces.sail[0]:+.0f}"
                f" rud={step.forces.rudder[0]:+.0f}"
                f" keel={step.forces.keel[0]:+.0f}"
                f" wav={step.forces.waves[0]:+.0f} N"
            )
            parts.append(
                f"Fy: sail={step.forces.sail[1]:+.0f}"
                f" rud={step.forces.rudder[1]:+.0f}"
                f" keel={step.forces.keel[1]:+.0f}"
                f" wav={step.forces.waves[1]:+.0f} N"
            )
            parts.append(
                f"Mz: sail={step.forces.sail[n_idx]:+.0f}"
                f" rud={step.forces.rudder[n_idx]:+.0f}"
                f" keel={step.forces.keel[n_idx]:+.0f}"
                f" wav={step.forces.waves[n_idx]:+.0f} Nm"
            )

        pitch_deg = np.degrees(step.state.theta)
        wave_el = step.waves.elevation if step.waves else 0.0
        torque = step.rudder_torque if step.rudder_torque is not None else 0.0
        parts.append(f"Pitch {pitch_deg:+.2f}\u00b0  Wave elev {wave_el:+.3f} m")
        parts.append(f"Rudder torque {torque:+.1f} Nm")

    # ── Page switching ─────────────────────────────────────────────────

    def _switch_page(self, page: int) -> None:
        """Switch between Steering (0), Environment (1), and Forces (2)."""
        if page == self._active_page:
            return
        self._active_page = page

        for i, axes in enumerate(self._page_axes):
            for ax in axes:
                ax.set_visible(i == page)

        # Twin axis for energy must be toggled explicitly
        self._ax_energy.set_visible(page == 2)

        for i, btn in enumerate(self._page_buttons):
            btn.label.set_fontweight("bold" if i == page else "normal")

        self._update_readout()
        self._fig.canvas.draw_idle()

    # ── Animation & controls ──────────────────────────────────────────

    def _update(self, _frame_num: int) -> list[Artist]:
        if self._playing:
            self._current_frame += self._speed
            if self._current_frame >= self._max_frames:
                self._current_frame = self._max_frames - 1
                self._playing = False
                self._btn_play.label.set_text("Play")
            self._update_cursors()
        return []

    def _on_slider(self, val: float) -> None:
        if self._updating_slider:
            return
        self._current_frame = int(val)
        self._update_cursors()
        self._fig.canvas.draw_idle()

    def _toggle_play(self, _event: object) -> None:
        self._playing = not self._playing
        self._btn_play.label.set_text("Pause" if self._playing else "Play")

    def _step_back(self, _event: object) -> None:
        self._playing = False
        self._btn_play.label.set_text("Play")
        self._current_frame = max(0, self._current_frame - 1)
        self._update_cursors()
        self._fig.canvas.draw_idle()

    def _step_fwd(self, _event: object) -> None:
        self._playing = False
        self._btn_play.label.set_text("Play")
        self._current_frame = min(self._max_frames - 1, self._current_frame + 1)
        self._update_cursors()
        self._fig.canvas.draw_idle()

    def _cycle_speed(self, _event: object) -> None:
        speeds = [1, 2, 5, 10]
        try:
            i = speeds.index(self._speed)
            self._speed = speeds[(i + 1) % len(speeds)]
        except ValueError:
            self._speed = 1
        self._btn_speed.label.set_text(f"{self._speed}x")

    def _on_key(self, event) -> None:
        if event.key == "1":
            self._switch_page(0)
        elif event.key == "2":
            self._switch_page(1)
        elif event.key == "3":
            self._switch_page(2)

    def _on_close(self, _event: object) -> None:
        if self._anim.event_source is not None:
            self._anim.event_source.stop()


def show(runs: list[tuple[str, Recorder]]) -> None:
    """Launch the interactive playback viewer."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _viewer = PlaybackViewer(runs)
    plt.show()
