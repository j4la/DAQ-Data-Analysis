"""
TDMS Sensor Data Plotter  (v2)
===============================
Reads sensor data from a TDMS file (group: "Untitled") and generates:

Individual channel plots
------------------------
  • One PNG per channel vs absolute time (time-axis tick spacing = 30 s)
  • X-axis major ticks every 30 s for ALL full-run plots

Combined / derived plots
------------------------
  • all_channels_stacked.png  – every channel stacked vertically, shared x-axis
  • vent_and_RTDs.png         – Vent Temperature + RTD Bottom/Middle/Top, shared x-axis
  • all_RTDs.png              – RTD Bottom / Middle / Top on a single axes
  • fire_event_thrust.png     – Box Thrust zoomed to the burn, time axis re-zeroed
                                to the fire-start, end-time tick shown explicitly
  • impulse_summary.png       – text card showing total impulse calculation

Sample rates
------------
  10 Hz  : N2O Temperature (C), Vent Temperature (C)
  100 Hz : everything else

Usage
-----
  python tdms_plotter.py  <path_to_file.tdms>

Output
------
  <tdms_filename>_plots/   (created next to the TDMS file)
"""

import sys
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nptdms import TdmsFile

# ── Configuration ─────────────────────────────────────────────────────────────

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

RTD_CHANNELS  = ["RTD Bottom", "RTD Middle", "RTD Top"]
RTD_COLORS    = ["#e6194b", "#f58231", "#3cb44b"]   # red / orange / green

CHANNEL_UNITS = {
    "N2O Temperature (C)":  "N2O Temp (°C)",
    "Vent Temperature (C)": "Vent Temp (°C)",
    "Bottle Pressure (Bar)":"Bottle P (Bar)",
    "Tank Pressure (Bar)":  "Tank P (Bar)",
    "Box Thrust (kg)":      "Thrust (kg)",
    "O2 Pressure (Bar)":    "O2 P (Bar)",
    "RTD Bottom":           "RTD Bot (°C)",
    "RTD Middle":           "RTD Mid (°C)",
    "RTD Top":              "RTD Top (°C)",
}

CHANNEL_COLORS = {
    "N2O Temperature (C)":  "#1f77b4",
    "Vent Temperature (C)": "#ff7f0e",
    "Bottle Pressure (Bar)":"#2ca02c",
    "Tank Pressure (Bar)":  "#9467bd",
    "Box Thrust (kg)":      "#d62728",
    "O2 Pressure (Bar)":    "#8c564b",
    "RTD Bottom":           "#e6194b",
    "RTD Middle":           "#f58231",
    "RTD Top":              "#3cb44b",
}

GROUP_NAME  = "Untitled"
PLOT_DPI    = 150
TICK_STEP   = 30          # seconds between major x-axis ticks for full-run plots

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tdms(path: str) -> dict[str, np.ndarray]:
    tdms = TdmsFile.read(path)
    group_names = [g.name for g in tdms.groups()]
    if GROUP_NAME in group_names:
        group = tdms[GROUP_NAME]
    else:
        matches = [n for n in group_names if n.strip().lower() == GROUP_NAME.lower()]
        if not matches:
            raise ValueError(
                f"Could not find group '{GROUP_NAME}' in {path}.\n"
                f"Available groups: {group_names}"
            )
        group = tdms[matches[0]]

    data: dict[str, np.ndarray] = {}
    available = {ch.name for ch in group.channels()}
    for name in ALL_CHANNELS:
        if name in available:
            data[name] = group[name][:]
        else:
            print(f"  [WARNING] Channel '{name}' not found – skipping.")
    return data


def build_time(n_samples: int, hz: float) -> np.ndarray:
    return np.arange(n_samples) / hz


def make_output_dir(tdms_path: str) -> str:
    base = os.path.splitext(os.path.basename(tdms_path))[0]
    out  = os.path.join(os.path.dirname(tdms_path), f"{base}_plots")
    os.makedirs(out, exist_ok=True)
    return out


def safe_filename(name: str) -> str:
    return (name.replace(" ", "_").replace("/", "_")
                .replace("(", "").replace(")", ""))


def apply_30s_ticks(ax, t_max: float):
    """Major tick every 30 s, minor every 5 s, on the given axes."""
    ax.xaxis.set_major_locator(ticker.MultipleLocator(TICK_STEP))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.set_xlim(left=0, right=t_max)
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.2)


def style_ax(ax):
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=6)


# ── Fire-event detection ──────────────────────────────────────────────────────

def detect_fire_event(
    thrust: np.ndarray,
    time: np.ndarray,
    baseline_window: float = 1.0,
    threshold_factor: float = 3.0,
    hz: float = 100.0,
) -> tuple[int, int]:
    """Return (i_start, i_end) sample indices of the thrust event."""
    n_base = int(baseline_window * hz)
    baseline = thrust[:n_base]
    b_mean   = float(np.mean(baseline))
    b_std    = float(np.std(baseline))

    threshold_hi = max(b_mean + threshold_factor * b_std, 0.0)
    threshold_lo = b_mean + b_std

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
        raise RuntimeError(
            "Could not detect the start of the fire event. "
            "Check that Box Thrust data contains a positive thrust pulse."
        )
    if i_end is None:
        i_end = len(thrust) - 1

    return i_start, i_end


# ── Individual channel plot ───────────────────────────────────────────────────

def plot_channel(name: str, values: np.ndarray, time: np.ndarray,
                 out_dir: str, t_max_global: float):
    fig, ax = plt.subplots(figsize=(14, 4))
    color = CHANNEL_COLORS.get(name, "#1f77b4")
    ax.plot(time, values, color=color, linewidth=0.8, label=name)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=11)
    hz_label = "10 Hz" if name in CHANNELS_10HZ else "100 Hz"
    ax.set_title(f"{name}  –  {hz_label}", fontsize=12)
    apply_30s_ticks(ax, t_max_global)
    style_ax(ax)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{safe_filename(name)}.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── All channels stacked ──────────────────────────────────────────────────────

def plot_all_stacked(data: dict[str, np.ndarray], out_dir: str,
                     t_max_global: float):
    """One subplot per channel, all sharing the same x-axis."""
    available = [c for c in ALL_CHANNELS if c in data]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3 * n), sharex=True
    )
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        values = data[name]
        time   = build_time(len(values), hz)
        color  = CHANNEL_COLORS.get(name, "#1f77b4")
        ax.plot(time, values, color=color, linewidth=0.7, label=name)
        ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        style_ax(ax)
        ax.grid(which="major", linestyle="--", alpha=0.5)
        ax.grid(which="minor", linestyle=":",  alpha=0.2)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Apply 30 s ticks only to the shared x-axis (bottom subplot)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(TICK_STEP))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axes[-1].set_xlim(left=0, right=t_max_global)
    axes[-1].set_xlabel("Time (s)", fontsize=11)

    fig.suptitle("All Channels – Full Run", fontsize=13, y=1.002)
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_channels_stacked.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Vent temp + RTDs stacked ──────────────────────────────────────────────────

def plot_vent_and_rtds(data: dict[str, np.ndarray], out_dir: str,
                       t_max_global: float):
    """
    Vent Temperature and the three RTDs on separate subplots with a shared
    x-axis so they are directly comparable in time.
    """
    channels = ["Vent Temperature (C)"] + RTD_CHANNELS
    available = [c for c in channels if c in data]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {
        "Vent Temperature (C)": "#ff7f0e",
        "RTD Bottom":           "#e6194b",
        "RTD Middle":           "#f58231",
        "RTD Top":              "#3cb44b",
    }

    for ax, name in zip(axes, available):
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        values = data[name]
        time   = build_time(len(values), hz)
        ax.plot(time, values, color=colors.get(name, "#333333"),
                linewidth=0.8, label=name)
        ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        style_ax(ax)
        ax.grid(which="major", linestyle="--", alpha=0.5)
        ax.grid(which="minor", linestyle=":",  alpha=0.2)

    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(TICK_STEP))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axes[-1].set_xlim(left=0, right=t_max_global)
    axes[-1].set_xlabel("Time (s)", fontsize=11)

    fig.suptitle("Vent Temperature & RTDs – Full Run", fontsize=13, y=1.002)
    fig.tight_layout()
    fname = os.path.join(out_dir, "vent_and_RTDs.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── All RTDs on one axes ──────────────────────────────────────────────────────

def plot_all_rtds(data: dict[str, np.ndarray], out_dir: str,
                  t_max_global: float):
    available = [c for c in RTD_CHANNELS if c in data]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    for name, color in zip(RTD_CHANNELS, RTD_COLORS):
        if name not in data:
            continue
        values = data[name]
        time   = build_time(len(values), 100.0)
        ax.plot(time, values, color=color, linewidth=0.9, label=name)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("All RTDs – Full Run", fontsize=12)
    apply_30s_ticks(ax, t_max_global)
    style_ax(ax)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_RTDs.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Fire-event thrust plot ────────────────────────────────────────────────────

def plot_fire_event(thrust: np.ndarray, time: np.ndarray, out_dir: str,
                    pad: float = 2.0) -> tuple[float, float] | None:
    """
    Plot the thrust fire event with:
      - time axis re-zeroed so fire-start = 0
      - x-axis starts at 0 (fire-start)
      - explicit tick at the exact fire-end time
      - total impulse annotation
    Returns (burn_time, avg_thrust) or None on failure.
    """
    hz = 100.0
    try:
        i_start, i_end = detect_fire_event(thrust, time, hz=hz)
    except RuntimeError as err:
        print(f"  [WARNING] Fire event detection failed: {err}")
        return None

    t_start = time[i_start]
    t_end   = time[i_end]
    duration = t_end - t_start

    # ── Impulse metrics ────────────────────────────────────────────────────
    fire_values   = thrust[i_start : i_end + 1]
    avg_thrust_kg = float(np.mean(fire_values))
    burn_time_s   = duration
    # Total impulse in N·s  (1 kgf ≈ 9.80665 N)
    total_impulse_Ns = avg_thrust_kg * 9.80665 * burn_time_s
    peak_thrust_kg   = float(np.max(fire_values))

    # ── Build the window ───────────────────────────────────────────────────
    # Include `pad` seconds before fire-start and after fire-end
    i_lo = max(i_start - int(pad * hz), 0)
    i_hi = min(i_end   + int(pad * hz), len(thrust) - 1)

    t_win = time[i_lo : i_hi + 1] - t_start   # re-zero so fire-start = 0
    v_win = thrust[i_lo : i_hi + 1]

    t_fire_end_shifted = t_end - t_start        # end time on re-zeroed axis

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_win, v_win, color="#d62728", linewidth=1.2, label="Box Thrust")

    # Zero reference line
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)

    # Fire-start marker (now at t=0)
    ax.axvline(0,               color="green",  linestyle="--", linewidth=1.2,
               label=f"Fire start (t = 0 s)")
    # Fire-end marker
    ax.axvline(t_fire_end_shifted, color="orange", linestyle="--", linewidth=1.2,
               label=f"Fire end (t = {t_fire_end_shifted:.3f} s)")

    # Peak annotation
    peak_idx_local = int(np.argmax(v_win))
    ax.annotate(
        f"Peak: {peak_thrust_kg:.2f} kg",
        xy=(t_win[peak_idx_local], v_win[peak_idx_local]),
        xytext=(t_win[peak_idx_local],
                v_win[peak_idx_local] * 0.80),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, ha="center",
    )

    # Impulse text box (top-left)
    info = (
        f"Burn time:     {burn_time_s:.3f} s\n"
        f"Avg thrust:    {avg_thrust_kg:.2f} kg\n"
        f"Total impulse: {total_impulse_Ns:.1f} N·s"
    )
    ax.text(
        0.02, 0.97, info,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9),
        family="monospace",
    )

    # ── X-axis: start from 0, add explicit end-time tick ──────────────────
    x_lo = t_win[0]          # slightly before 0 (negative pad)
    x_hi = t_win[-1]

    # Build ticks: every 1 s auto, but always include 0 and t_fire_end_shifted
    auto_ticks = np.arange(
        np.ceil(x_lo),
        np.floor(x_hi) + 1,
        1.0
    )
    all_ticks = np.unique(
        np.concatenate([auto_ticks, [0.0, t_fire_end_shifted]])
    )
    ax.set_xticks(all_ticks)
    # Format all ticks; highlight the fire-end tick label
    labels = []
    for t in all_ticks:
        if abs(t - t_fire_end_shifted) < 1e-6:
            labels.append(f"{t:.3f}\n(end)")
        else:
            labels.append(f"{t:.1f}")
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_xlim(left=0, right=x_hi)      # start axis at 0 (fire-start)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.2)

    ax.set_xlabel("Time relative to fire start (s)", fontsize=11)
    ax.set_ylabel("Thrust (kg)", fontsize=11)
    ax.set_title(
        f"Box Thrust – Fire Event  "
        f"(burn = {burn_time_s:.3f} s,  "
        f"avg = {avg_thrust_kg:.2f} kg,  "
        f"impulse = {total_impulse_Ns:.1f} N·s)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    fname = os.path.join(out_dir, "fire_event_thrust.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Fire event: start={t_start:.4f} s, end={t_end:.4f} s, "
          f"duration={burn_time_s:.4f} s, peak={peak_thrust_kg:.3f} kg, "
          f"impulse={total_impulse_Ns:.2f} N·s")

    return burn_time_s, avg_thrust_kg


# ── Total impulse summary card ────────────────────────────────────────────────

def plot_impulse_summary(burn_time: float, avg_thrust_kg: float, out_dir: str):
    """Render a clean summary card showing the impulse calculation."""
    total_impulse_Ns  = avg_thrust_kg * 9.80665 * burn_time
    avg_thrust_N      = avg_thrust_kg * 9.80665

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    title = "Total Impulse Summary"
    body  = textwrap.dedent(f"""\
        Formula:    Total Impulse  =  Average Thrust  ×  Burn Time

        Burn Time       =  {burn_time:.4f} s
        Average Thrust  =  {avg_thrust_kg:.4f} kg  ({avg_thrust_N:.3f} N)

        Total Impulse   =  {avg_thrust_kg:.4f} kg  ×  9.80665  ×  {burn_time:.4f} s
                        =  {total_impulse_Ns:.3f} N·s
    """)

    ax.text(
        0.5, 0.90, title,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        ha="center", va="top",
    )
    ax.text(
        0.05, 0.72, body,
        transform=ax.transAxes,
        fontsize=12, va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4ff",
                  edgecolor="#4466aa", linewidth=1.5),
    )

    fig.tight_layout()
    fname = os.path.join(out_dir, "impulse_summary.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


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

    # Determine global time maximum (longest channel duration)
    t_max_global = 0.0
    for name, values in data.items():
        hz   = 10.0 if name in CHANNELS_10HZ else 100.0
        t_ch = (len(values) - 1) / hz
        t_max_global = max(t_max_global, t_ch)

    # Round up to the next 30 s boundary so the last tick is always clean
    t_max_plot = np.ceil(t_max_global / TICK_STEP) * TICK_STEP

    # ── Individual channel plots ───────────────────────────────────────────
    print("── Individual channel plots ──")
    for name in ALL_CHANNELS:
        if name not in data:
            continue
        values = data[name]
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        time   = build_time(len(values), hz)
        print(f"  Plotting '{name}'  ({len(values)} samples @ {hz:.0f} Hz, "
              f"duration={time[-1]:.2f} s)")
        plot_channel(name, values, time, out_dir, t_max_plot)

    # ── Combined plots ─────────────────────────────────────────────────────
    print("\n── Combined plots ──")
    plot_all_stacked(data, out_dir, t_max_plot)
    plot_vent_and_rtds(data, out_dir, t_max_plot)
    plot_all_rtds(data, out_dir, t_max_plot)

    # ── Fire event + impulse ───────────────────────────────────────────────
    print("\n── Fire event & impulse ──")
    if "Box Thrust (kg)" in data:
        thrust = data["Box Thrust (kg)"]
        time   = build_time(len(thrust), 100.0)
        result = plot_fire_event(thrust, time, out_dir)
        if result is not None:
            burn_time, avg_thrust = result
            plot_impulse_summary(burn_time, avg_thrust, out_dir)
    else:
        print("  [WARNING] 'Box Thrust (kg)' not in data – fire event skipped.")

    print(f"\nDone. All plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()