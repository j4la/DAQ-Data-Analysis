"""
TDMS Sensor Data Plotter
========================
Reads sensor data from a TDMS file (sheet: "Untitled") and generates:
  - One plot per channel vs time (reconstructed from known sample rates)
  - A zoomed "fire event" plot for Box Thrust

Sensor sample rates
-------------------
  10 Hz  : N2O Temperature (C), Vent Temperature (C)
  100 Hz : everything else

Usage
-----
  python tdms_plotter.py <path_to_file.tdms>

Output
------
  A folder called  <tdms_filename>_plots/  containing one PNG per channel
  plus  fire_event_thrust.png.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nptdms import TdmsFile

# ── Configuration ────────────────────────────────────────────────────────────

CHANNELS_10HZ = {"N2O Temperature (C)", "Vent Temperature (C)"}

ALL_CHANNELS = [
    "N2O Temperature (C)",
    "Vent Temperature (C)",
    "Bottle Pressure (Bar)",
    "Tank Pressure (Bar)",
    "Box Thrust (kg)",
    "O2 Pressure (Bar)",
    "RTD Bottom",
    "RTD Middle",
    "RTD Top",
]

CHANNEL_UNITS = {
    "N2O Temperature (C)":  "Temperature (°C)",
    "Vent Temperature (C)": "Temperature (°C)",
    "Bottle Pressure (Bar)":"Pressure (Bar)",
    "Tank Pressure (Bar)":  "Pressure (Bar)",
    "Box Thrust (kg)":      "Thrust (kg)",
    "O2 Pressure (Bar)":    "Pressure (Bar)",
    "RTD Bottom":           "Temperature (°C)",
    "RTD Middle":           "Temperature (°C)",
    "RTD Top":              "Temperature (°C)",
}

GROUP_NAME   = "Untitled"   # second sheet name in the TDMS file
PLOT_DPI     = 150
STYLE_COLOR  = "#1f77b4"    # default line colour
FIRE_COLOR   = "#d62728"    # thrust fire-event colour

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tdms(path: str) -> dict[str, np.ndarray]:
    """Return {channel_name: numpy_array} for the Untitled group."""
    tdms = TdmsFile.read(path)

    # Locate the group (case-insensitive fallback)
    group_names = [g.name for g in tdms.groups()]
    if GROUP_NAME in group_names:
        group = tdms[GROUP_NAME]
    else:
        # Try to find a group that matches loosely
        matches = [n for n in group_names if n.strip().lower() == GROUP_NAME.lower()]
        if not matches:
            raise ValueError(
                f"Could not find group '{GROUP_NAME}' in {path}.\n"
                f"Available groups: {group_names}"
            )
        group = tdms[matches[0]]

    data = {}
    available = {ch.name for ch in group.channels()}
    for name in ALL_CHANNELS:
        if name in available:
            data[name] = group[name][:]
        else:
            print(f"  [WARNING] Channel '{name}' not found – skipping.")

    return data


def build_time(n_samples: int, hz: float) -> np.ndarray:
    """Return a time axis in seconds starting at 0."""
    return np.arange(n_samples) / hz


def make_output_dir(tdms_path: str) -> str:
    base = os.path.splitext(os.path.basename(tdms_path))[0]
    out  = os.path.join(os.path.dirname(tdms_path), f"{base}_plots")
    os.makedirs(out, exist_ok=True)
    return out


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_channel(name: str, values: np.ndarray, time: np.ndarray, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, values, color=STYLE_COLOR, linewidth=0.8, label=name)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=11)
    ax.set_title(f"{name}  –  {'10 Hz' if name in CHANNELS_10HZ else '100 Hz'}", fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{safe_filename(name)}.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


def detect_fire_event(
    thrust: np.ndarray,
    time: np.ndarray,
    baseline_window: float = 1.0,   # seconds to estimate baseline
    threshold_factor: float = 3.0,  # how many std-devs above baseline = "fire"
    hz: float = 100.0,
    pad: float = 2.0,               # seconds of padding around the event
) -> tuple[float, float]:
    """
    Returns (t_start, t_end) of the fire event.

    Strategy
    --------
    1. Estimate baseline mean & std from the first `baseline_window` seconds.
    2. Fire starts at the first sample that crosses  baseline_mean + threshold * std
       AND is still positive (above zero).
    3. Fire ends when the signal returns to within 1 std of baseline.
    """
    n_base = int(baseline_window * hz)
    baseline = thrust[:n_base]
    b_mean   = np.mean(baseline)
    b_std    = np.std(baseline)

    # Use a threshold that is above zero and above baseline noise
    threshold_hi = max(b_mean + threshold_factor * b_std, 0.0)
    threshold_lo = b_mean + b_std   # re-entry threshold

    fired = False
    i_start = i_end = None

    for i, v in enumerate(thrust):
        if not fired:
            if v > threshold_hi:
                fired   = True
                i_start = i
        else:
            if v <= threshold_lo:
                i_end = i
                break

    if i_start is None:
        raise RuntimeError("Could not detect the start of the fire event. "
                           "Check that Box Thrust data contains a positive thrust pulse.")
    if i_end is None:
        i_end = len(thrust) - 1  # event runs to end of data

    t_start = time[i_start]
    t_end   = time[i_end]
    return t_start, t_end


def plot_fire_event(thrust: np.ndarray, time: np.ndarray, out_dir: str, pad: float = 2.0):
    """Detect and plot the fire event window."""
    hz = 100.0

    try:
        t_start, t_end = detect_fire_event(thrust, time, hz=hz, pad=pad)
    except RuntimeError as err:
        print(f"  [WARNING] Fire event detection failed: {err}")
        return

    # Add padding around the event
    t_lo = max(t_start - pad, time[0])
    t_hi = min(t_end   + pad, time[-1])

    mask = (time >= t_lo) & (time <= t_hi)
    t_win = time[mask]
    v_win = thrust[mask]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_win, v_win, color=FIRE_COLOR, linewidth=1.0, label="Box Thrust")

    # Mark start / end of fire event
    ax.axvline(t_start, color="green",  linestyle="--", linewidth=1.2, label=f"Fire start  ({t_start:.3f} s)")
    ax.axvline(t_end,   color="orange", linestyle="--", linewidth=1.2, label=f"Fire end    ({t_end:.3f} s)")
    ax.axhline(0,       color="grey",   linestyle=":",  linewidth=0.8, alpha=0.7)

    duration = t_end - t_start
    peak_idx  = np.argmax(v_win)
    peak_val  = v_win[peak_idx]

    # Annotate peak
    ax.annotate(
        f"Peak: {peak_val:.2f} kg",
        xy=(t_win[peak_idx], peak_val),
        xytext=(t_win[peak_idx], peak_val * 0.85),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, ha="center",
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Thrust (kg)", fontsize=11)
    ax.set_title(
        f"Box Thrust – Fire Event  "
        f"(start={t_start:.3f} s,  end={t_end:.3f} s,  duration={duration:.3f} s)",
        fontsize=12,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fname = os.path.join(out_dir, "fire_event_thrust.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Fire event: start={t_start:.4f} s, end={t_end:.4f} s, "
          f"duration={duration:.4f} s, peak={peak_val:.3f} kg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python tdms_plotter.py <path_to_file.tdms>")
        sys.exit(1)

    tdms_path = sys.argv[1]
    if not os.path.isfile(tdms_path):
        print(f"Error: File not found – {tdms_path}")
        sys.exit(1)

    print(f"\nReading: {tdms_path}")
    data    = load_tdms(tdms_path)
    out_dir = make_output_dir(tdms_path)
    print(f"Output directory: {out_dir}\n")

    # ── Plot every channel ─────────────────────────────────────────────────
    for name in ALL_CHANNELS:
        if name not in data:
            continue

        values = data[name]
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        time   = build_time(len(values), hz)

        print(f"  Plotting '{name}'  ({len(values)} samples @ {hz:.0f} Hz, "
              f"duration={time[-1]:.2f} s)")
        plot_channel(name, values, time, out_dir)

    # ── Fire-event plot ────────────────────────────────────────────────────
    if "Box Thrust (kg)" in data:
        print("\nDetecting and plotting fire event …")
        thrust = data["Box Thrust (kg)"]
        time   = build_time(len(thrust), 100.0)
        plot_fire_event(thrust, time, out_dir)
    else:
        print("\n[WARNING] 'Box Thrust (kg)' not in data – fire event plot skipped.")

    print(f"\nDone. {len(data)} channel(s) plotted → {out_dir}/")


if __name__ == "__main__":
    main()