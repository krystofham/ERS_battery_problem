"""
ERS Boost Zone Estimator – 2026 F1 Regulations
===============================================
Physical estimation of ERS deployment from FastF1 telemetry data.
Since the 2026 F1 API does not provide binary ERS/aero flags, 
we reconstruct activation from force dynamics: F_ERS = m·a + F_drag − F_ICE_baseline.

Improvements over the original version:
  - Robust gear shift stability detection (rolling std instead of diff == 0)
  - More accurate ERS energy estimation (integral over the zone, not avg * avg)
  - Dual-threshold detection: Straight Line Mode vs. Overtake Mode
  - 2025 vs. 2026 comparative overlay plot
  - Zone export to CSV
  - Distance normalization (0–1) for track comparison
  - Error handling for session loading
  - Comments on physical constant uncertainty
"""

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Cache 
CACHE_PATH = Path("./fastf1_cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))

# PHYSICAL CONSTANTS (Estimated)
# Uncertainties are noted – they directly affect baseline accuracy.

AIR_DENSITY  = 1.225   # kg/m³ – Standard atmosphere (±0.05 at 20 °C / 500m ASL)
CAR_MASS     = 768     # kg   – Minimum mass incl. driver (±5 kg)
CD_A         = 0.55    # m²   – Cd * Frontal Area; depends on aero setup (±0.15 m²)
                       #        affects baseline by ~8% at 250 km/h
ROLLING_COEF = 0.015   # –    – Rolling resistance (±0.003)
G            = 9.81    # m/s²

# DETECTION PARAMETERS

MIN_SPEED_KMH        = 200    # km/h – Minimum speed for analysis (straights)
THROTTLE_MIN         = 96     # %    – Full throttle (WOT condition)
GEAR_STABLE_WINDOW   = 5      # samples – Window for gear stability detection
SG_WINDOW            = 15     # samples – Savitzky-Golay window (must be odd)
SG_POLYORDER         = 3

BASELINE_DEG         = 2      # Polynomial degree for ICE baseline
BASELINE_N_BINS      = 20     # Number of speed bins
BASELINE_PERCENTILE  = 35     # Force percentile in each bin → "ICE Floor"

# Dual-threshold detection – Straight Line Mode vs. Overtake Mode
# Overtake Mode = significantly higher ERS output → higher percentile
SLM_RESIDUAL_PERCENTILE  = 65   # Straight Line Mode threshold
OTM_RESIDUAL_PERCENTILE  = 88   # Overtake Mode threshold

MIN_ZONE_METERS      = 50     # m – Minimum zone length (noise filter)
MERGE_GAP_METERS     = 150     # m – Merging threshold for adjacent zones


# TELEMETRY CLEANING

def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans telemetry and adds derived columns.
    New: Gear stability detection via rolling std (more robust against index gaps).
    """
    cols = ["Speed", "Throttle", "Brake", "nGear", "Time", "Distance"]
    df = df[cols].dropna().copy().reset_index(drop=True)
    df["time_s"] = df["Time"].dt.total_seconds()

    # 1. Calculate raw acceleration
    df["accel_raw"] = df["Speed"].diff() / df["time_s"].diff().replace(0, np.nan)

    # 2. Explicitly drop rows where acceleration couldn't be calculated 
    # (This handles the first row AND any mid-lap time-gaps safely)
    df = df.dropna(subset=["accel_raw"]).copy()

    # 3. Filter out physically impossible sensor spikes (>150 km/h/s)
    valid_accel_mask = df["accel_raw"].abs() <= 150
    df = df[valid_accel_mask].reset_index(drop=True)
  
    # Savitzky-Golay acceleration smoothing
    if len(df) > SG_WINDOW:
        df["accel_smooth"] = savgol_filter(
            df["accel_raw"].fillna(0),
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER,
        )
    else:
        df["accel_smooth"] = df["accel_raw"]

    # Gear Stability Detection
    # Rolling std, True if gear didn't change within GEAR_STABLE_WINDOW
    gear_std = df["nGear"].rolling(GEAR_STABLE_WINDOW, center=True, min_periods=1).std()
    df["gear_stable"] = (gear_std == 0)

    # Normalized distance 0–1 for track comparisons
    d_min, d_max = df["Distance"].min(), df["Distance"].max()
    df["dist_norm"] = (df["Distance"] - d_min) / (d_max - d_min) if d_max > d_min else 0.0

    return df


# PHYSICAL MODEL

def drag_force(v_kmh: pd.Series) -> pd.Series:
    """Aerodynamic + Rolling resistance [N]."""
    v_ms = v_kmh / 3.6
    return 0.5 * AIR_DENSITY * CD_A * v_ms**2 + ROLLING_COEF * CAR_MASS * G


def compute_propulsive_force(df: pd.DataFrame) -> pd.Series:
    """Total propulsive force [N] = m·a + F_drag."""
    return CAR_MASS * (df["accel_smooth"] / 3.6) + drag_force(df["Speed"])


def fit_engine_baseline(speeds: pd.Series, forces: pd.Series):
    """
    Robust ICE baseline via percentiles in speed bins.
    """
    v = speeds.values
    f = forces.values
    bins = np.linspace(v.min(), v.max(), BASELINE_N_BINS + 1)
    bin_v, bin_f = [], []
    for i in range(BASELINE_N_BINS):
        mask = (v >= bins[i]) & (v < bins[i + 1])
        if mask.sum() > 3:
            bin_v.append(v[mask].mean())
            bin_f.append(np.percentile(f[mask], BASELINE_PERCENTILE))

    if len(bin_v) < BASELINE_DEG + 1:
        c = np.polyfit(v, f, deg=BASELINE_DEG)
        return np.polyval(c, v), c

    c = np.polyfit(bin_v, bin_f, deg=BASELINE_DEG)
    return np.polyval(c, v), c


# ERS ENERGY CALCULATION (Integral approach)

def zone_energy_kj(cand: pd.DataFrame, d_start: float, d_end: float) -> float:
    """
    Estimated ERS energy in the zone [kJ].
    Logic: Integral P_ERS dt ≈ Σ (F_residual × v) × Δt
    """
    mask = (cand["Distance"] >= d_start) & (cand["Distance"] <= d_end)
    z = cand[mask].copy()
    if z.empty or "time_s" not in z.columns:
        return float("nan")
    v_ms = z["Speed"] / 3.6
    power_w = z["F_residual"] * v_ms
    dt = z["time_s"].diff().fillna(0).clip(lower=0, upper=1)
    energy_j = (power_w * dt).sum()
    return energy_j / 1000  # kJ


# ERS ZONE DETECTION (Dual-Threshold: SLM + OTM)

def detect_zones_at_threshold(cand: pd.DataFrame, df: pd.DataFrame, threshold: float, label: str):
    """Helper function – returns zones for a specific threshold."""
    above = cand[cand["F_residual"] > threshold]
    if above.empty:
        return []

    raw_zones = []
    g_start = prev = above.index[0]
    for idx in above.index[1:]:
        if idx - prev > 5:
            raw_zones.append((g_start, prev))
            g_start = idx
        prev = idx
    raw_zones.append((g_start, prev))

    zones_m = []
    for s, e in raw_zones:
        d_start = df.loc[s, "Distance"]
        d_end   = df.loc[e, "Distance"]
        if d_end - d_start >= MIN_ZONE_METERS:
            zones_m.append((d_start, d_end, s, e))

    if not zones_m:
        return []

    merged = [zones_m[0]]
    for zone in zones_m[1:]:
        if zone[0] - merged[-1][1] <= MERGE_GAP_METERS:
            merged[-1] = (merged[-1][0], zone[1], merged[-1][2], zone[3])
        else:
            merged.append(zone)

    return merged


def find_ers_zones(df: pd.DataFrame):
    """
    Main detection logic. Returns:
      slm_zones  – Straight Line Mode zones
      otm_zones  – Overtake Mode zones (subset of SLM)
      cand       – Candidates DataFrame
      thr_slm    – SLM threshold [N]
      thr_otm    – OTM threshold [N]
    """
    mask = (
        (df["Throttle"] >= THROTTLE_MIN) &
        (df["Brake"] == 0) &
        (df["Speed"] >= MIN_SPEED_KMH) &
        (df["gear_stable"])
    )
    cand = df[mask].copy()

    if len(cand) < BASELINE_DEG + 2:
        print("⚠ Insufficient data for analysis.")
        return [], [], None, None, None

    cand["F_propulsive"] = compute_propulsive_force(cand)
    baseline, _          = fit_engine_baseline(cand["Speed"], cand["F_propulsive"])
    cand["F_baseline"]   = baseline
    cand["F_residual"]   = cand["F_propulsive"] - cand["F_baseline"]

    thr_slm = np.percentile(cand["F_residual"], SLM_RESIDUAL_PERCENTILE)
    thr_otm = np.percentile(cand["F_residual"], OTM_RESIDUAL_PERCENTILE)

    print(f"\nResidual Force Statistics -------")
    print(f"  Median : {np.median(cand['F_residual']):+.0f} N")
    print(f"  IQR    : {np.percentile(cand['F_residual'],25):.0f} … {np.percentile(cand['F_residual'],75):.0f} N")
    print(f"  Std    : {cand['F_residual'].std():.0f} N")
    print(f"  SLM Threshold ({SLM_RESIDUAL_PERCENTILE}th perc.) : {thr_slm:.0f} N")
    print(f"  OTM Threshold ({OTM_RESIDUAL_PERCENTILE}th perc.) : {thr_otm:.0f} N")

    slm_zones = detect_zones_at_threshold(cand, df, thr_slm, "SLM")
    otm_zones = detect_zones_at_threshold(cand, df, thr_otm, "OTM")

    print(f"\nDetected Zones -------")
    for label, zones in [("Straight Line Mode", slm_zones), ("Overtake Mode", otm_zones)]:
        print(f"\n  {label}: {len(zones)} zones")
        for i, (d_start, d_end, *_) in enumerate(zones):
            ekj = zone_energy_kj(cand, d_start, d_end)
            zone_mask = (cand["Distance"] >= d_start) & (cand["Distance"] <= d_end)
            mean_v = cand.loc[zone_mask, "Speed"].mean() if zone_mask.any() else float("nan")
            print(f"    Zone {i+1}: {d_start:.0f}m → {d_end:.0f}m  "
                  f"(Length: {d_end-d_start:.0f}m, "
                  f"Avg Speed: {mean_v:.0f} km/h, "
                  f"Energy: {ekj:.1f} kJ)")

    return slm_zones, otm_zones, cand, thr_slm, thr_otm


# CSV EXPORT

def export_zones_csv(slm_zones, otm_zones, cand: pd.DataFrame,
                     driver: str, session_name: str, out_path: Path):
    """Exports detected zones to CSV for external analysis."""
    rows = []
    for mode, zones in [("SLM", slm_zones), ("OTM", otm_zones)]:
        for d_start, d_end, *_ in zones:
            zone_mask = (cand["Distance"] >= d_start) & (cand["Distance"] <= d_end)
            z = cand[zone_mask]
            rows.append({
                "driver":         driver,
                "session":        session_name,
                "mode":           mode,
                "d_start_m":      round(d_start, 1),
                "d_end_m":        round(d_end, 1),
                "length_m":       round(d_end - d_start, 1),
                "mean_speed_kmh": round(z["Speed"].mean(), 1) if not z.empty else None,
                "peak_force_N":   round(z["F_residual"].max(), 0) if not z.empty else None,
                "energy_kJ":      round(zone_energy_kj(cand, d_start, d_end), 2),
            })
    if rows:
        df_out = pd.DataFrame(rows)
        csv_path = out_path / f"ers_zones_{driver}_{session_name}.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\n✓ Exported: {csv_path}")
        return df_out
    return pd.DataFrame()


# visualization
def plot_telemetry(df, slm_zones, otm_zones, candidate, thr_slm, thr_otm,
                   driver_name, session_name="2026 R1"):
    """Speed, Acceleration + Throttle, Residual Force."""
    fig = plt.figure(figsize=(17, 12), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.07,
                             top=0.93, bottom=0.06, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax1, ax2, ax3 = axes

    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#cccccc")

    SLM_COLOR = "#f5a623"   # Orange – Straight Line Mode
    OTM_COLOR = "#e63946"   # Red    – Overtake Mode
    dist = df["Distance"]

    # Panel 1: Speed 
    ax1.plot(dist, df["Speed"], color="#4da6ff", lw=1.4, label="Speed")
    ax1.axhline(MIN_SPEED_KMH, color="#555", ls="--", lw=0.8,
                label=f"Min. {MIN_SPEED_KMH} km/h")
    ax1.set_ylabel("Speed (km/h)", color="#cccccc")
    ax1.set_ylim(0, df["Speed"].max() * 1.1)

    # Panel 2: Acceleration + Throttle
    ax2.plot(dist, df["accel_raw"],    color="#333333", lw=0.6, label="Accel (Raw)")
    ax2.plot(dist, df["accel_smooth"], color="#4da6ff", lw=1.3, label="Accel (Filter)")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.set_ylabel("Acceleration (km/h/s)", color="#cccccc")
    ax2.set_ylim(-35, 90)

    # Throttle overlay on right axis
    ax2b = ax2.twinx()
    ax2b.plot(dist, df["Throttle"], color="#cc4444", alpha=0.45, lw=0.9, label="Throttle (%)")
    ax2b.set_ylim(0, 160)
    ax2b.set_ylabel("Throttle (%)", color="#cc4444")
    ax2b.tick_params(colors="#cc4444", labelsize=7)

    # Panel 3: Residual Force
    if candidate is not None and "F_residual" in candidate.columns:
        cdist = candidate["Distance"]
        res   = candidate["F_residual"]

        ax3.plot(cdist, candidate["F_propulsive"], color="#333333", lw=0.9, label="Total Propulsive Force")
        ax3.plot(cdist, candidate["F_baseline"], color="#44aa66", lw=1.5, ls="--", label="ICE Baseline")
        ax3.plot(cdist, res, color="#4da6ff", lw=1.1, label="Residual Force (F − Baseline)")
        ax3.fill_between(cdist, 0, res, where=res > 0, color="#4da6ff", alpha=0.15)

        if thr_slm is not None:
            ax3.axhline(thr_slm, color=SLM_COLOR, ls=":", lw=1.2,
                        label=f"SLM Threshold ({SLM_RESIDUAL_PERCENTILE}th perc. = {thr_slm:.0f} N)")
        if thr_otm is not None:
            ax3.axhline(thr_otm, color=OTM_COLOR, ls=":", lw=1.2,
                        label=f"OTM Threshold ({OTM_RESIDUAL_PERCENTILE}th perc. = {thr_otm:.0f} N)")
        ax3.axhline(0, color="#444", lw=0.5)
        ax3.set_ylabel("Force (N)", color="#cccccc")
        ax3.set_xlabel("Distance (m)", color="#aaaaaa")

    # Highlight Zones 
    for i, (d_start, d_end, *_) in enumerate(slm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=SLM_COLOR, alpha=0.18, label="SLM" if (i == 0 and ax is ax1) else "")

    for i, (d_start, d_end, *_) in enumerate(otm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=OTM_COLOR, alpha=0.30, label="OTM" if (i == 0 and ax is ax1) else "")

    # Shared Legends
    legend_patches = [
        Patch(color=SLM_COLOR, alpha=0.6, label=f"Straight Line Mode ({len(slm_zones)} zones)"),
        Patch(color=OTM_COLOR, alpha=0.8, label=f"Overtake Mode ({len(otm_zones)} zones)"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][:2],
               fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")
    ax2.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")
    ax3.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")

    fig.suptitle(
        f"ERS Telemetry Analysis · {driver_name} · {session_name} · Physical Estimation (2026 Regs)",
        fontsize=12, color="#eeeeee", y=0.97
    )

    out = Path("./graphs") / f"ers_telemetry_{driver_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Chart saved: {out}")
    plt.show()


# VISUALIZATION – 2025 vs. 2026 Overlay

def plot_overlay_2025_vs_2026(df_2025: pd.DataFrame | None,
                               df_2026: pd.DataFrame,
                               slm_zones_2026,
                               straight_label: str = "Start/Finish Straight"):
    """
    Comparative plot of speed profiles 2025 vs. 2026.
    Uses normalized distance (0-1) to align different lap lengths.
    """
    fig, (ax_spd, ax_acc) = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0f0f0f", sharex=True)
    for ax in [ax_spd, ax_acc]:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#333333")

    x_2026 = df_2026["dist_norm"]
    ax_spd.plot(x_2026, df_2026["Speed"], color="#e63946", lw=1.5, label="2026")
    ax_acc.plot(x_2026, df_2026["accel_smooth"], color="#e63946", lw=1.3, label="2026")

    if df_2025 is not None:
        x_2025 = df_2025["dist_norm"]
        ax_spd.plot(x_2025, df_2025["Speed"], color="#4da6ff", lw=1.5, ls="--", label="2025")
        ax_acc.plot(x_2025, df_2025["accel_smooth"], color="#4da6ff", lw=1.3, ls="--", label="2025")

    # Highlight SLM zones (normalized)
    d_min, d_max = df_2026["Distance"].min(), df_2026["Distance"].max()
    for i, (d_start, d_end, *_) in enumerate(slm_zones_2026):
        n_start = (d_start - d_min) / (d_max - d_min) if d_max > d_min else 0
        n_end   = (d_end   - d_min) / (d_max - d_min) if d_max > d_min else 1
        for ax in [ax_spd, ax_acc]:
            ax.axvspan(n_start, n_end, color="#f5a623", alpha=0.20,
                       label="Est. SLM (2026)" if i == 0 and ax is ax_spd else "")

    ax_spd.set_ylabel("Speed (km/h)", color="#cccccc")
    ax_acc.set_ylabel("Acceleration (km/h/s)", color="#cccccc")
    ax_acc.set_xlabel("Normalized Distance (0 = Start, 1 = Finish)", color="#aaaaaa")
    ax_spd.axhline(MIN_SPEED_KMH, color="#555", ls=":", lw=0.8)
    ax_acc.axhline(0, color="#555", lw=0.6)

    for ax in [ax_spd, ax_acc]:
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

    fig.suptitle(f"Comparison 2025 vs. 2026 – {straight_label}", fontsize=12, color="#eeeeee", y=0.97)
    fig.subplots_adjust(hspace=0.06, top=0.93, bottom=0.08)

    out = Path("./graphs") / "overlay_2025_vs_2026.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Overlay saved: {out}")
    plt.show()


# MAIN EXECUTION

def load_session_safe(year: int, event, session_type: str):
    """Loads session with error handling."""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"❌ Loading session {year} / {event} / {session_type} failed: {e}")
        return None

def plot_baseline_diagnostic(cand: pd.DataFrame, driver_name: str):
    """
    Scatter: F_propulsive vs. Speed.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")

    ax.scatter(cand["Speed"], cand["F_propulsive"],
               s=2, alpha=0.3, color="#4da6ff", label="F_propulsive")
    ax.scatter(cand["Speed"], cand["F_baseline"],
               s=2, alpha=0.6, color="#44aa66", label="ICE baseline")

    # Zvýrazni spike_suspect zóny jinou barvou
    suspect_mask = cand["F_residual"] > cand["F_residual"].quantile(0.95)
    ax.scatter(cand.loc[suspect_mask, "Speed"],
               cand.loc[suspect_mask, "F_propulsive"],
               s=8, color="#e63946", alpha=0.7, label="Top 5% residual")

    ax.set_xlabel("Speed (km/h)", color="#aaaaaa")
    ax.set_ylabel("Force (N)", color="#cccccc")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")
    ax.set_title(f"Baseline Diagnostic – {driver_name}", color="#eeeeee")

    out = Path("./graphs") / f"baseline_diagnostic_{driver_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Baseline diagnostic: {out}")
    plt.show()
def main():
    OUTPUT_DIR   = Path("./graphs")
    DRIVER_CODE  = "HAM"
    SESSION_YEAR = 2026
    SESSION_EVT  = 1
    SESSION_TYPE = "R"

    session_2026 = load_session_safe(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
    if session_2026 is None:
        print("Exiting due to missing data.")
        return

    lap_2026 = session_2026.laps.pick_drivers(DRIVER_CODE).pick_fastest()
    tel_2026  = lap_2026.get_telemetry()
    df_2026   = clean_telemetry(tel_2026)

    slm_zones, otm_zones, candidate, thr_slm, thr_otm = find_ers_zones(df_2026)

    session_label = f"{SESSION_YEAR} R{SESSION_EVT}"
    plot_telemetry(df_2026, slm_zones, otm_zones, candidate, thr_slm, thr_otm,
                   driver_name=DRIVER_CODE, session_name=session_label)

    if candidate is not None:
        export_zones_csv(slm_zones, otm_zones, candidate,
                         driver=DRIVER_CODE,
                         session_name=session_label.replace(" ", "_"),
                         out_path=OUTPUT_DIR)

    df_2025 = None
    session_2025 = load_session_safe(2025, SESSION_EVT, SESSION_TYPE)
    if session_2025 is not None:
        try:
            lap_2025 = session_2025.laps.pick_drivers(DRIVER_CODE).pick_fastest()
            df_2025  = clean_telemetry(lap_2025.get_telemetry())
        except Exception as e:
            print(f"2025 lap not loaded: {e}")

    plot_overlay_2025_vs_2026(df_2025, df_2026, slm_zones,
                               straight_label=f"GP {SESSION_EVT} – Fastest Lap")

    # Cross-check Output
    total_slm_kj = sum(zone_energy_kj(candidate, d, e) for d, e, *_ in slm_zones)
    print(f"Total estimated ERS energy per lap: {total_slm_kj:.0f} kJ")
    # Reference: 2026 battery limit ~4 MJ (4000 kJ) per lap deployment.
    # If output is significantly higher, baseline is too low (CD_A likely underestimated).
    if candidate is not None:
        plot_baseline_diagnostic(candidate, DRIVER_CODE)
if __name__ == "__main__":
    main()
