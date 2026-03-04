"""Apply simulation heading patch to pypilot's pilot.py.

Appends a patched compute_heading() that falls back to GPS track
when the IMU heading is unavailable (no physical IMU in simulation).
"""

import glob
import sys

# Find pilot.py in site-packages
pilot_files = glob.glob(
    "/usr/local/lib/python3.*/site-packages/pypilot/pilots/pilot.py"
)

if not pilot_files:
    print("ERROR: Could not find pypilot/pilots/pilot.py", file=sys.stderr)
    sys.exit(1)

pilot_file = pilot_files[0]

# The patch: when IMU heading is unavailable (compass is False),
# fall back to gps.track.value (set from NMEA RMC sentences).
# This is the unfiltered GPS track which pypilot sets directly
# in sensors.py gps.update() from NMEA data.
patch_code = '''

# --- Simulation heading patch: use GPS track as heading fallback ---
def _patched_compute_heading(self):
    from pypilot.resolv import resolv
    ap = self.ap
    compass = ap.boatimu.SensorValues["heading_lowpass"].value
    if compass is False:
        # No IMU available — use GPS track from NMEA as heading
        gps_track = ap.sensors.gps.track.value
        if gps_track is False or gps_track is None:
            return
        compass = gps_track
    if ap.mode.value == "true wind":
        true_wind = resolv(ap.true_wind_compass_offset.value - compass)
        ap.heading.set(true_wind)
    elif ap.mode.value == "wind":
        wind = resolv(ap.wind_compass_offset.value - compass)
        ap.heading.set(wind)
    elif ap.mode.value == "gps" or ap.mode.value == "nav":
        gps = resolv(compass + ap.gps_compass_offset.value, 180)
        ap.heading.set(gps)
    elif ap.mode.value == "compass":
        ap.heading.set(compass)

AutopilotPilot.compute_heading = _patched_compute_heading
'''

with open(pilot_file, "a") as f:
    f.write(patch_code)

print(f"Patched {pilot_file}")
