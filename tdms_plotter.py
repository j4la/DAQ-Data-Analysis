"""
TDMS Sensor Data Plotter  (v3)
================================
Reads sensor data from a TDMS file (group: "Untitled") and generates:

Individual channel plots
------------------------
  • One PNG per channel vs absolute time
  • X-axis always has exactly 10 evenly-spaced tick marks

Combined / derived plots
------------------------
  • all_channels_stacked.png      – every channel stacked, shared x-axis (10 ticks)
  • vent_and_RTDs.png             – Vent Temperature + RTDs, shared x-axis (10 ticks)
  • all_RTDs.png                  – all three RTDs overlaid (10 ticks)
  • fire_event_thrust.png         – burn window, re-zeroed time, 10 ticks + exact end tick
  • impulse_summary.png           – impulse calculation card
  • rail_exit_velocity.png        – velocity & position vs time through the launch rail

Exported files
--------------
  • <name>.eng                    – RASP/OpenRocket engine file for the thrust curve

Physics
-------
  Rail exit velocity integrates F_net = F_thrust - m*g - 0.5*rho*Cd*A*v^2
  over the measured thrust curve until the rocket base clears the top of the rail.
  Rail length = 10 m, rocket length = 3 m  =>  travel distance = 7 m.

  IMPORTANT: Update ROCKET_MASS_KG, ROCKET_CD, and ROCKET_DIAM_M at the top
  of this file to match your actual rocket before trusting the velocity result.

Sample rates
------------
  10 Hz  : N2O Temperature (C), Vent Temperature (C)
  100 Hz : everything else

Usage
-----
  python tdms_plotter.py  <path_to_file.tdms>
"""

import sys
import os
import textwrap
import math

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

RTD_CHANNELS = ["RTD Bottom", "RTD Middle", "RTD Top"]
RTD_COLORS   = ["#e6194b", "#f58231", "#3cb44b"]

CHANNEL_UNITS = {
    "N2O Temperature (C)":  "N2O Temp (C)",
    "Vent Temperature (C)": "Vent Temp (C)",
    "Bottle Pressure (Bar)":"Bottle P (Bar)",
    "Tank Pressure (Bar)":  "Tank P (Bar)",
    "Box Thrust (kg)":      "Thrust (kg)",
    "O2 Pressure (Bar)":    "O2 P (Bar)",
    "RTD Bottom":           "RTD Bot (C)",
    "RTD Middle":           "RTD Mid (C)",
    "RTD Top":              "RTD Top (C)",
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

GROUP_NAME = "Untitled"
PLOT_DPI   = 150

# ── Launch rail / rocket parameters (update these for your rocket) ─────────────
RAIL_HEIGHT_M   = 10.0    # height of the top of the launch rail (m)
ROCKET_LENGTH_M =  3.0    # length of the rocket body (m)
RAIL_TRAVEL_M   = RAIL_HEIGHT_M - ROCKET_LENGTH_M   # distance rocket travels on rail = 7 m

ROCKET_MASS_KG  = 15.0    # total wet launch mass (kg)  <-- UPDATE
ROCKET_CD       =  0.5    # drag coefficient             <-- UPDATE
ROCKET_DIAM_M   =  0.075  # body tube outer diameter (m) <-- UPDATE

# Derived
G           = 9.80665
RHO_AIR     = 1.225       # kg/m^3, sea-level ISA
ROCKET_AREA = math.pi * (ROCKET_DIAM_M / 2.0) ** 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tdms(path):
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

    data = {}
    available = {ch.name for ch in group.channels()}
    for name in ALL_CHANNELS:
        if name in available:
            data[name] = group[name][:]
        else:
            print(f"  [WARNING] Channel '{name}' not found - skipping.")
    return data


def build_time(n_samples, hz):
    return np.arange(n_samples) / hz


def make_output_dir(tdms_path):
    base = os.path.splitext(os.path.basename(tdms_path))[0]
    out  = os.path.join(os.path.dirname(tdms_path), f"{base}_plots")
    os.makedirs(out, exist_ok=True)
    return out


def safe_filename(name):
    return (name.replace(" ", "_").replace("/", "_")
                .replace("(", "").replace(")", ""))


def apply_10_ticks(ax, t_min, t_max, extra_ticks=None, extra_labels=None):
    """
    Place exactly 10 evenly-spaced major ticks from t_min to t_max.
    Optionally append extra tick positions (e.g. the exact burn-end time)
    labelled via extra_labels = {value: label_string}.
    Minor ticks sit at midpoints between each adjacent pair of major ticks.
    """
    major = np.linspace(t_min, t_max, 10)
    if extra_ticks:
        major = np.unique(np.concatenate([major, extra_ticks]))

    ax.set_xticks(major)

    if extra_labels:
        lbls = []
        for t in major:
            hit = None
            for ev, lbl in extra_labels.items():
                if abs(t - ev) < 1e-9:
                    hit = lbl
                    break
            lbls.append(hit if hit is not None else f"{t:.1f}")
        ax.set_xticklabels(lbls, fontsize=8)
    else:
        ax.set_xticklabels([f"{t:.1f}" for t in major], fontsize=8)

    if len(major) > 1:
        midpoints = (major[:-1] + major[1:]) / 2
        ax.set_xticks(midpoints, minor=True)

    ax.set_xlim(left=t_min, right=t_max)
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.2)


def style_ax(ax):
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=6)


# ── Fire-event detection ──────────────────────────────────────────────────────

def detect_fire_event(thrust, time, baseline_window=1.0, threshold_factor=3.0, hz=100.0):
    """Return (i_start, i_end) sample indices bounding the thrust event."""
    n_base = int(baseline_window * hz)
    baseline  = thrust[:n_base]
    b_mean    = float(np.mean(baseline))
    b_std     = float(np.std(baseline))

    threshold_hi = max(b_mean + threshold_factor * b_std, 0.0)
    threshold_lo = b_mean + b_std

    fired   = False
    i_start = None
    i_end   = None

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
            "Could not detect fire event start. "
            "Ensure Box Thrust data contains a positive thrust pulse."
        )
    if i_end is None:
        i_end = len(thrust) - 1

    return i_start, i_end


# ── Individual channel plot ───────────────────────────────────────────────────

def plot_channel(name, values, time, out_dir, t_max_global):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, values, color=CHANNEL_COLORS.get(name, "#1f77b4"),
            linewidth=0.8, label=name)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=11)
    ax.set_title(f"{name}  -  {'10 Hz' if name in CHANNELS_10HZ else '100 Hz'}",
                 fontsize=12)
    apply_10_ticks(ax, 0.0, t_max_global)
    style_ax(ax)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{safe_filename(name)}.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── All channels stacked ──────────────────────────────────────────────────────

def plot_all_stacked(data, out_dir, t_max_global):
    available = [c for c in ALL_CHANNELS if c in data]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        values = data[name]
        time   = build_time(len(values), hz)
        ax.plot(time, values, color=CHANNEL_COLORS.get(name, "#1f77b4"),
                linewidth=0.7, label=name)
        ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        style_ax(ax)
        ax.grid(which="major", linestyle="--", alpha=0.5)
        ax.grid(which="minor", linestyle=":",  alpha=0.2)

    apply_10_ticks(axes[-1], 0.0, t_max_global)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle("All Channels - Full Run", fontsize=13, y=1.002)
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_channels_stacked.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Vent temp + RTDs stacked (shared x-axis) ──────────────────────────────────

def plot_vent_and_rtds(data, out_dir, t_max_global):
    channels  = ["Vent Temperature (C)"] + RTD_CHANNELS
    available = [c for c in channels if c in data]
    n = len(available)
    if n == 0:
        return

    vent_rtd_colors = {
        "Vent Temperature (C)": "#ff7f0e",
        "RTD Bottom":           "#e6194b",
        "RTD Middle":           "#f58231",
        "RTD Top":              "#3cb44b",
    }

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        values = data[name]
        time   = build_time(len(values), hz)
        ax.plot(time, values, color=vent_rtd_colors.get(name, "#333333"),
                linewidth=0.8, label=name)
        ax.set_ylabel(CHANNEL_UNITS.get(name, "Value"), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        style_ax(ax)
        ax.grid(which="major", linestyle="--", alpha=0.5)
        ax.grid(which="minor", linestyle=":",  alpha=0.2)

    apply_10_ticks(axes[-1], 0.0, t_max_global)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle("Vent Temperature & RTDs - Full Run", fontsize=13, y=1.002)
    fig.tight_layout()
    fname = os.path.join(out_dir, "vent_and_RTDs.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── All RTDs overlaid ─────────────────────────────────────────────────────────

def plot_all_rtds(data, out_dir, t_max_global):
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
    ax.set_ylabel("Temperature (C)", fontsize=11)
    ax.set_title("All RTDs - Full Run", fontsize=12)
    apply_10_ticks(ax, 0.0, t_max_global)
    style_ax(ax)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_RTDs.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Fire-event thrust plot ────────────────────────────────────────────────────

def plot_fire_event(thrust, time, out_dir, pad=2.0):
    """
    Returns a dict of fire-event metrics, or None on failure.
    Time axis is re-zeroed so fire-start = 0 s.
    10 evenly-spaced ticks + one extra tick at the exact burn-end time.
    """
    hz = 100.0
    try:
        i_start, i_end = detect_fire_event(thrust, time, hz=hz)
    except RuntimeError as err:
        print(f"  [WARNING] {err}")
        return None

    t_start  = time[i_start]
    t_end    = time[i_end]
    duration = t_end - t_start

    fire_values      = thrust[i_start : i_end + 1]
    avg_thrust_kg    = float(np.mean(fire_values))
    peak_thrust_kg   = float(np.max(fire_values))
    total_impulse_Ns = avg_thrust_kg * G * duration

    # Windowed view with padding
    i_lo  = max(i_start - int(pad * hz), 0)
    i_hi  = min(i_end   + int(pad * hz), len(thrust) - 1)
    t_win = time[i_lo : i_hi + 1] - t_start   # re-zero
    v_win = thrust[i_lo : i_hi + 1]

    t_end_rel = duration      # fire-end on re-zeroed axis
    x_hi      = t_win[-1]    # = pad s after fire-end

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_win, v_win, color="#d62728", linewidth=1.2, label="Box Thrust")
    ax.axhline(0,         color="grey",   linestyle=":",  linewidth=0.9, alpha=0.7)
    ax.axvline(0,         color="green",  linestyle="--", linewidth=1.2,
               label="Fire start (t = 0 s)")
    ax.axvline(t_end_rel, color="orange", linestyle="--", linewidth=1.2,
               label=f"Fire end  (t = {t_end_rel:.4f} s)")

    peak_idx = int(np.argmax(v_win))
    ax.annotate(
        f"Peak: {peak_thrust_kg:.2f} kg",
        xy=(t_win[peak_idx], v_win[peak_idx]),
        xytext=(t_win[peak_idx], v_win[peak_idx] * 0.78),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, ha="center",
    )

    info = (
        f"Burn time:     {duration:.4f} s\n"
        f"Avg thrust:    {avg_thrust_kg:.2f} kg\n"
        f"Total impulse: {total_impulse_Ns:.1f} N*s"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9))

    apply_10_ticks(
        ax, t_min=0.0, t_max=x_hi,
        extra_ticks=[t_end_rel],
        extra_labels={t_end_rel: f"{t_end_rel:.4f}\n(end)"},
    )
    style_ax(ax)
    ax.set_xlabel("Time from fire start (s)", fontsize=11)
    ax.set_ylabel("Thrust (kg)", fontsize=11)
    ax.set_title(
        f"Box Thrust - Fire Event  "
        f"(burn={duration:.4f} s,  avg={avg_thrust_kg:.2f} kg,  "
        f"impulse={total_impulse_Ns:.1f} N*s)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    fname = os.path.join(out_dir, "fire_event_thrust.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Fire event: start={t_start:.4f} s  end={t_end:.4f} s  "
          f"burn={duration:.4f} s  peak={peak_thrust_kg:.3f} kg  "
          f"impulse={total_impulse_Ns:.2f} N*s")

    return {
        "i_start":          i_start,
        "i_end":            i_end,
        "t_start":          t_start,
        "t_end":            t_end,
        "duration":         duration,
        "avg_thrust_kg":    avg_thrust_kg,
        "peak_thrust_kg":   peak_thrust_kg,
        "total_impulse_Ns": total_impulse_Ns,
        "fire_values":      fire_values,
    }


# ── Total impulse summary card ────────────────────────────────────────────────

def plot_impulse_summary(fire, out_dir):
    burn_time     = fire["duration"]
    avg_thrust_kg = fire["avg_thrust_kg"]
    avg_thrust_N  = avg_thrust_kg * G
    total_impulse = fire["total_impulse_Ns"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.93, "Total Impulse Summary",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="center", va="top")

    body = textwrap.dedent(f"""\
        Formula:   Total Impulse  =  Avg Thrust (N)  x  Burn Time (s)

        Burn Time      =  {burn_time:.4f} s
        Avg Thrust     =  {avg_thrust_kg:.4f} kgf  ({avg_thrust_N:.3f} N)

        Total Impulse  =  {avg_thrust_N:.3f} N  x  {burn_time:.4f} s
                       =  {total_impulse:.3f} N*s
    """)
    ax.text(0.05, 0.74, body, transform=ax.transAxes,
            fontsize=12, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4ff",
                      edgecolor="#4466aa", linewidth=1.5))

    fig.tight_layout()
    fname = os.path.join(out_dir, "impulse_summary.png")
    fig.savefig(fname, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── .eng file export ──────────────────────────────────────────────────────────

def write_eng_file(fire, thrust_full, tdms_path, out_dir):
    """
    Write a RASP/OpenRocket .eng engine file.

    Time axis: 0 (fire-start) to burn-end.
    Thrust:    Newtons (converted from kgf).
    """
    i_start       = fire["i_start"]
    i_end         = fire["i_end"]
    duration      = fire["duration"]
    total_impulse = fire["total_impulse_Ns"]
    avg_thrust_N  = fire["avg_thrust_kg"] * G
    peak_thrust_N = fire["peak_thrust_kg"] * G

    fire_thrust_N = np.maximum(thrust_full[i_start : i_end + 1] * G, 0.0)
    dt    = 1.0 / 100.0
    t_rel = np.arange(len(fire_thrust_N)) * dt

    # Determine RASP letter class from total impulse
    classes = [
        (2.5, "A"), (5, "B"), (10, "C"), (20, "D"), (40, "E"),
        (80, "F"), (160, "G"), (320, "H"), (640, "I"), (1280, "J"),
        (2560, "K"), (5120, "L"), (10240, "M"), (20480, "N"),
    ]
    motor_class = "X"
    for limit, letter in classes:
        if total_impulse <= limit:
            motor_class = letter
            break
    motor_name = f"{motor_class}{int(avg_thrust_N)}"

    base     = os.path.splitext(os.path.basename(tdms_path))[0]
    eng_path = os.path.join(out_dir, f"{base}.eng")

    with open(eng_path, "w") as f:
        f.write("; RASP engine file generated from TDMS data\n")
        f.write(f"; Source:         {os.path.basename(tdms_path)}\n")
        f.write(f"; Burn time:      {duration:.4f} s\n")
        f.write(f"; Total impulse:  {total_impulse:.3f} N*s\n")
        f.write(f"; Avg thrust:     {avg_thrust_N:.3f} N\n")
        f.write(f"; Peak thrust:    {peak_thrust_N:.3f} N\n")
        f.write(f"; Motor class:    {motor_class}\n")
        f.write(";\n")
        # RASP header: name diam_mm len_mm delays prop_mass total_mass manufacturer
        f.write(
            f"{motor_name} {int(ROCKET_DIAM_M*1000)} "
            f"{int(ROCKET_LENGTH_M*1000)} 0 0 {ROCKET_MASS_KG:.3f} TDMS\n"
        )
        # Data: time(s) thrust(N)
        for t, fn in zip(t_rel, fire_thrust_N):
            f.write(f"   {t:.4f} {fn:.4f}\n")
        # Ensure clean zero at end
        last_t = t_rel[-1]
        if fire_thrust_N[-1] > 0.001:
            f.write(f"   {last_t + dt:.4f} 0.0000\n")
        f.write(";\n")

    print(f"  Saved: {eng_path}  "
          f"(class {motor_class}, {len(t_rel)} points, {duration:.4f} s)")


# ── Rail-exit velocity ────────────────────────────────────────────────────────

def compute_rail_exit(fire, thrust_full):
    """
    Numerically integrate the equation of motion along the launch rail:

        F_net = F_thrust - m*g - 0.5*rho*Cd*A*v^2

    starting from v=0, x=0 at ignition.  Integration stops when the rocket
    base has travelled RAIL_TRAVEL_M (= RAIL_HEIGHT_M - ROCKET_LENGTH_M).

    Returns a dict with time/velocity/position arrays and exit speed.
    """
    i_start = fire["i_start"]
    i_end   = fire["i_end"]

    fire_thrust_N = np.maximum(thrust_full[i_start : i_end + 1] * G, 0.0)
    dt = 1.0 / 100.0

    v, x, t = 0.0, 0.0, 0.0
    v_hist = [v]
    x_hist = [x]
    t_hist = [t]
    rail_exit_v = None
    rail_exit_t = None

    for fn in fire_thrust_N:
        drag  = 0.5 * RHO_AIR * ROCKET_CD * ROCKET_AREA * v * abs(v)
        f_net = fn - ROCKET_MASS_KG * G - drag
        a     = f_net / ROCKET_MASS_KG

        v = max(v + a * dt, 0.0)   # cannot go negative on the rail
        x += v * dt
        t += dt

        v_hist.append(v)
        x_hist.append(x)
        t_hist.append(t)

        if rail_exit_v is None and x >= RAIL_TRAVEL_M:
            rail_exit_v = v
            rail_exit_t = t
            break

    return {
        "t_arr":       np.array(t_hist),
        "v_arr":       np.array(v_hist),
        "x_arr":       np.array(x_hist),
        "rail_exit_v": rail_exit_v,
        "rail_exit_t": rail_exit_t,
    }


def plot_rail_exit(rail, out_dir):
    t_arr  = rail["t_arr"]
    v_arr  = rail["v_arr"]
    x_arr  = rail["x_arr"]
    v_exit = rail["rail_exit_v"]
    t_exit = rail["rail_exit_t"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ── Velocity ───────────────────────────────────────────────────────────
    ax1.plot(t_arr, v_arr, color="#1f77b4", linewidth=1.2, label="Velocity")
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_title(
        f"Rail Exit Velocity Analysis\n"
        f"Rail = {RAIL_HEIGHT_M} m  |  Rocket = {ROCKET_LENGTH_M} m  |  "
        f"Travel = {RAIL_TRAVEL_M} m  |  Mass = {ROCKET_MASS_KG} kg  |  "
        f"Cd = {ROCKET_CD}  |  Diam = {ROCKET_DIAM_M*100:.1f} cm",
        fontsize=10,
    )
    if v_exit is not None:
        ax1.axvline(t_exit, color="orange", linestyle="--", linewidth=1.2,
                    label=f"Rail exit  t = {t_exit:.3f} s")
        ax1.axhline(v_exit, color="green",  linestyle=":",  linewidth=1.0,
                    label=f"Exit velocity = {v_exit:.2f} m/s  "
                          f"({v_exit*3.6:.1f} km/h)")
        ax1.annotate(
            f"Exit: {v_exit:.2f} m/s\n({v_exit*3.6:.1f} km/h)",
            xy=(t_exit, v_exit),
            xytext=(t_exit + t_arr[-1] * 0.06, v_exit * 0.78),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9),
        )
    ax1.legend(fontsize=9, loc="upper left")
    style_ax(ax1)
    ax1.grid(which="major", linestyle="--", alpha=0.5)
    ax1.grid(which="minor", linestyle=":",  alpha=0.2)

    # ── Position ───────────────────────────────────────────────────────────
    ax2.plot(t_arr, x_arr, color="#9467bd", linewidth=1.2, label="Position")
    ax2.axhline(RAIL_TRAVEL_M, color="orange", linestyle="--", linewidth=1.0,
                label=f"Rail clearance ({RAIL_TRAVEL_M} m)")
    ax2.set_ylabel("Position along rail (m)", fontsize=11)
    ax2.set_xlabel("Time from ignition (s)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    style_ax(ax2)
    ax2.grid(which="major", linestyle="--", alpha=0.5)
    ax2.grid(which="minor", linestyle=":",  alpha=0.2)

    t_max_rail = t_arr[-1]
    extra_t = [t_exit] if t_exit is not None else None
    apply_10_ticks(ax2, 0.0, t_max_rail, extra_ticks=extra_t)

    fig.tight_layout()
    fname = os.path.join(out_dir, "rail_exit_velocity.png")
    fig.savefig(fname, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"  Saved: {fname}")

    if v_exit is not None:
        print(f"  Rail exit velocity : {v_exit:.3f} m/s  ({v_exit*3.6:.2f} km/h)")
        print(f"  Rail exit time     : {t_exit:.4f} s from ignition")
    else:
        print("  [WARNING] Rocket did not clear the rail during the measured burn.")
        print("            Check ROCKET_MASS_KG, ROCKET_CD, and thrust data.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python tdms_plotter.py <path_to_file.tdms>")
        sys.exit(1)

    tdms_path = sys.argv[1]
    if not os.path.isfile(tdms_path):
        print(f"Error: File not found - {tdms_path}")
        sys.exit(1)

    print(f"\nReading: {tdms_path}")
    data    = load_tdms(tdms_path)
    out_dir = make_output_dir(tdms_path)
    print(f"Output directory: {out_dir}\n")

    # Global time span across all channels
    t_max_global = 0.0
    for name, values in data.items():
        hz   = 10.0 if name in CHANNELS_10HZ else 100.0
        t_ch = (len(values) - 1) / hz
        t_max_global = max(t_max_global, t_ch)

    # ── Individual channel plots ───────────────────────────────────────────
    print("---- Individual channel plots ----")
    for name in ALL_CHANNELS:
        if name not in data:
            continue
        values = data[name]
        hz     = 10.0 if name in CHANNELS_10HZ else 100.0
        time   = build_time(len(values), hz)
        print(f"  Plotting '{name}'  ({len(values)} samples @ {hz:.0f} Hz, "
              f"duration={time[-1]:.2f} s)")
        plot_channel(name, values, time, out_dir, t_max_global)

    # ── Combined overview plots ────────────────────────────────────────────
    print("\n---- Combined plots ----")
    plot_all_stacked(data, out_dir, t_max_global)
    plot_vent_and_rtds(data, out_dir, t_max_global)
    plot_all_rtds(data, out_dir, t_max_global)

    # ── Fire event pipeline ────────────────────────────────────────────────
    print("\n---- Fire event, impulse, .eng, rail velocity ----")
    if "Box Thrust (kg)" not in data:
        print("  [WARNING] 'Box Thrust (kg)' not found - skipping fire analysis.")
        print(f"\nDone. Plots saved to: {out_dir}/")
        return

    thrust = data["Box Thrust (kg)"]
    time   = build_time(len(thrust), 100.0)

    fire = plot_fire_event(thrust, time, out_dir)
    if fire is None:
        print(f"\nDone (fire event not detected). Plots saved to: {out_dir}/")
        return

    plot_impulse_summary(fire, out_dir)

    print("\n  Writing .eng file ...")
    write_eng_file(fire, thrust, tdms_path, out_dir)

    print("\n  Computing rail exit velocity ...")
    rail = compute_rail_exit(fire, thrust)
    plot_rail_exit(rail, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
