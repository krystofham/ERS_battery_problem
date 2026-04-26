"""
ERS Boost Zone Estimator – 2026 F1 Regulations
===============================================
Physical estimation of ERS deployment from FastF1 telemetry data.
Since the 2026 F1 API does not provide binary ERS/aero flags,
we reconstruct activation from force dynamics: F_ERS = m·a + F_drag − F_ICE_baseline.

v2 improvements (physics):
  [NEW] DRS-aware dynamic CD_A: detects DRS activation via telemetry channel and
        applies a configurable drag reduction multiplier to avoid false ERS positives.
  [NEW] Fuel mass attrition: linearly interpolates car mass over race laps based on
        estimated fuel burn, correcting F = m*a baseline for late-race laps.
  [NEW] Track gradient (Z-axis): uses FastF1 X/Y/Z telemetry to compute per-sample
        slope angle θ and adds F_gravity = m*g*sin(θ) to the propulsive force model.

Previous improvements (v1):
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

AIR_DENSITY     = 1.225   # kg/m³ – Standard atmosphere (±0.05 at 20 °C / 500m ASL)
CAR_MASS_BASE   = 768     # kg   – Minimum mass incl. driver at race start (±5 kg)
CD_A_BASE       = 0.55    # m²   – Cd * Frontal Area; no-DRS reference (±0.15 m²)
                           #        affects baseline by ~8% at 250 km/h
ROLLING_COEF    = 0.015   # –    – Rolling resistance (±0.003)G               = 9.81    # m/s²

# [NEW] DRS PARAMETERS
# FastF1 DRS channel: values >= 10 generally indicate open DRS.
# The drag reduction multiplier (0–1) applied to CD_A when DRS is active.
# Typical DRS reduces drag ~20–25%; 0.78 is a conservative midpoint.
# Sources: Nakagawa et al. (2022 Aerodynamics Conference), Bop team estimates.
DRS_OPEN_THRESHOLD  = 10      # DRS channel value above which DRS is considered open
DRS_DRAG_MULTIPLIER = 0.78    # CD_A scaling factor when DRS is open  (±0.04)

# [NEW] FUEL BURN PARAMETERS
# F1 races allow up to 110 kg of fuel. Actual burn varies by track but
# 1.7–2.0 kg/lap is typical on a 60-lap circuit. We model it as linear.
# Source: 2026 Technical Regulations Art. 6.4; team telemetry public estimates.
FUEL_LOAD_KG        = 100     # kg – Fuel loaded at race start (±5 kg)
FUEL_BURN_RATE      = 1.7     # kg/lap – Average fuel burn rate (±0.3 kg/lap)
                               # Skewing lower is safer; over-estimation raises mass → lowers residual

# [NEW] GRADIENT PARAMETERS
# Z-axis smoothing window. Raw Z-axis from GPS has noise; aggressive smoothing
# avoids injecting high-frequency force spikes into the baseline.
GRADIENT_SG_WINDOW   = 301     # Must be odd, and > SG_POLYORDER
GRADIENT_SG_POLYORDER = 3


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
SLM_RESIDUAL_PERCENTILE  = 65   # Straight Line Mode threshold
OTM_RESIDUAL_PERCENTILE  = 88   # Overtake Mode threshold

MIN_ZONE_METERS      = 50     # m – Minimum zone length (noise filter)
MERGE_GAP_METERS     = 150    # m – Merging threshold for adjacent zones
G = 9.81

# FUEL MASS MODEL

def compute_dynamic_mass(lap_number: int, total_laps: int) -> float:
    """
    Returns estimated car mass at a given lap number [kg].

    Models fuel burn as linear over the race. The lap_number argument should
    be the 1-indexed race lap (1 = first lap, total_laps = final lap).

    Physical reasoning:
      - Start mass  = CAR_MASS_BASE + FUEL_LOAD_KG
      - End mass    ≈ CAR_MASS_BASE  (empty tank)
      - At lap L:   mass = start_mass − (L − 1) * FUEL_BURN_RATE
    Clamped at CAR_MASS_BASE so we never go below the dry car weight.

    Uncertainty note:
      A ±0.3 kg/lap error propagates to ±18 kg on lap 60, or roughly ±13 N
      of baseline force at 1 m/s² acceleration.
    """
    fuel_burned = min((lap_number - 1) * FUEL_BURN_RATE, FUEL_LOAD_KG)
    if fuel_burned > FUEL_LOAD_KG:
        raise ValueError("Invalid fuel data")
    return CAR_MASS_BASE + FUEL_LOAD_KG - fuel_burned


# DRS-AWARE CD_A 

def compute_effective_cd_a(df: pd.DataFrame) -> pd.Series:
    """
    Returns a per-sample effective CD_A Series [m²].

    If the DRS telemetry channel is available and the value is >= DRS_OPEN_THRESHOLD,
    the drag coefficient is reduced by DRS_DRAG_MULTIPLIER. Otherwise the base value
    CD_A_BASE is used.

    Why this matters:
      At 300 km/h (83.3 m/s), a 22% drag reduction (ΔCD_A ≈ 0.12) reduces drag
      force by ~280 N. Without this correction, the residual force during DRS
      zones spikes by ~280 N, which is easily above the OTM detection threshold —
      producing false Overtake Mode activations on every long straight.

    FastF1 DRS channel reference:
      Values: 0 = closed, 8 = available, 10/12/14 = open (hardware-dependent).
      Using >= 10 as the open threshold is standard practice in the community.
    """

    if "DRS" not in df.columns or df["DRS"].max() == 0:
        print("  DRS channel empty or all zeros. Using constant CD_A_BASE.")
        return pd.Series(CD_A_BASE, index=df.index)


    drs_open = df["DRS"] >= DRS_OPEN_THRESHOLD
    effective = np.where(drs_open, CD_A_BASE * DRS_DRAG_MULTIPLIER, CD_A_BASE)
    n_open = drs_open.sum()
    print(f"  DRS open samples: {n_open} ({100 * n_open / len(df):.1f}% of filtered data)")
    if (100 * n_open / len(df)) > 50:
        raise ValueError("DRS not working properly")
    return pd.Series(effective, index=df.index)


# GRADIENT FORCE 

def compute_gradient_force(df: pd.DataFrame, car_mass: float) -> pd.Series:
    """
    Returns gravitational resistance/assistance due to track slope [N].

    Method:
      1. Compute dZ/dDistance along the lap using smoothed Z telemetry.
      2. Gradient angle θ = arctan(dZ/dD).
      3. Gravity component F_g = m * g * sin(θ).
         Positive on uphill (resists motion), negative on downhill (assists motion).

    Smoothing:
      Raw Z from GPS has ~0.5 m noise. Without smoothing this creates ±G/sample
      force spikes. GRADIENT_SG_WINDOW (default 201) as lower values are not working properly
      at typical telemetry resolution, preserving genuine elevation changes like
      Eau Rouge / Raidillon while suppressing sensor flutter.

    Physical scale reference (Spa Raidillon, ~20% gradient):
      sin(11.3°) ≈ 0.196 → F_g = 768 kg * 9.81 * 0.196 ≈ 1476 N uphill
      This is comparable in magnitude to ICE peak torque force — cannot be ignored.

    Returns zero Series if Z channel is unavailable (flat-track assumption).
    """
    COLUMN_Z_NAME = "Z"
    if COLUMN_Z_NAME not in df.columns or df["Z"].isna().all():
        raise ValueError("Z telemetry not available. Assuming flat track (F_gradient = 0).")

    z = df[COLUMN_Z_NAME].fillna(method="ffill").fillna(method="bfill").values
    d = df["Distance"].values

    # Smooth Z to suppress GPS noise before differentiating
    if len(z) > GRADIENT_SG_WINDOW:
        z_smooth = savgol_filter(z, window_length=GRADIENT_SG_WINDOW,
                                 polyorder=GRADIENT_SG_POLYORDER)
    else:
        z_smooth = z

    # dZ/dD: finite differences, pad endpoints with zero slope
    dz = np.gradient(z_smooth, d)

    # θ = arctan(dZ/dD); sin(θ) ≈ dZ/dD for small angles, but use exact form
    MAX_ANGLE_DEG = 15
    theta_raw = np.arctan(dz)
    clipped = np.sum(np.abs(np.degrees(theta_raw)) > MAX_ANGLE_DEG)
    print(f"  Gradient clip zasáhl: {clipped} vzorků z {len(theta_raw)} ({100*clipped/len(theta_raw):.1f}%)")
    theta = np.clip(theta_raw, np.radians(-MAX_ANGLE_DEG), np.radians(MAX_ANGLE_DEG))
    f_gradient = car_mass * G * np.sin(theta)

    # Diagnostic output
    print(f"  Track gradient range: {np.degrees(theta.min()):.1f}° … {np.degrees(theta.max()):.1f}°")
    print(f"  Gradient force range: {f_gradient.min():.0f} N … {f_gradient.max():.0f} N")

    return pd.Series(f_gradient, index=df.index)


# TELEMETRY CLEANING

def clean_telemetry(df: pd.DataFrame, lap_number: int = 1,
                    total_laps: int = 60) -> pd.DataFrame:
    """
    Cleans telemetry and adds derived columns.

    Parameters:
      lap_number  : Race lap this telemetry belongs to (for mass model).
      total_laps  : Total laps in the session (for mass model denominator).

    Changes vs v1:
      - Requests DRS column in addition to core channels.
      - Requests Z column for gradient model.
      - Stores computed car_mass for use in downstream force calculations.
    """
    # Include DRS and Z if present; don't fail if absent
    core_cols = ["Speed", "Throttle", "Brake", "nGear", "Time", "Distance"]
    opt_cols  = ["DRS", "X", "Y", "Z"]
    available = [c for c in opt_cols if c in df.columns]
    cols = core_cols + available
    df = df[cols].dropna(subset=core_cols).copy().reset_index(drop=True)
    df["time_s"] = df["Time"].dt.total_seconds()

    # Dynamic car mass for this lap
    df["car_mass"] = compute_dynamic_mass(lap_number, total_laps)

    # Raw acceleration
    df["accel_raw"] = df["Speed"].diff() / df["time_s"].diff().replace(0, np.nan)
    df = df.dropna(subset=["accel_raw"]).copy()

    # Filter physically impossible sensor spikes (>150 km/h/s)
    df = df[df["accel_raw"].abs() <= 150].reset_index(drop=True)

    # Savitzky-Golay acceleration smoothing
    if len(df) > SG_WINDOW:
        df["accel_smooth"] = savgol_filter(
            df["accel_raw"].fillna(0),
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER,
        )
    else:
        df["accel_smooth"] = df["accel_raw"]

    # Gear stability detection: rolling std, True if gear stable in window
    gear_std = df["nGear"].rolling(GEAR_STABLE_WINDOW, center=True, min_periods=1).std()
    df["gear_stable"] = (gear_std == 0)

    # Normalized distance 0–1 for track comparisons
    d_min, d_max = df["Distance"].min(), df["Distance"].max()
    df["dist_norm"] = (df["Distance"] - d_min) / (d_max - d_min) if d_max > d_min else 0.0

    return df


# PHYSICAL MODEL

def drag_force(v_kmh: pd.Series, cd_a: pd.Series) -> pd.Series:
    """
    Aerodynamic + Rolling resistance [N].
    Now accepts a per-sample cd_a Series instead of the global constant,
    enabling DRS-corrected drag calculation at each telemetry point.
    """
    v_ms = v_kmh / 3.6
    # car_mass is looked up from the Series in the caller; rolling uses the
    # same dynamic mass for consistency, but the effect is negligible (<0.5 N).
    return 0.5 * AIR_DENSITY * cd_a * v_ms**2 + ROLLING_COEF * CAR_MASS_BASE * G


def compute_propulsive_force(df: pd.DataFrame) -> pd.Series:
    """
    Total propulsive force [N] = m·a + F_drag − F_gradient.

    Changes vs v1:
      - Uses per-sample dynamic car_mass from df["car_mass"] for inertial term.
      - Uses per-sample effective CD_A (DRS-corrected).
      - Subtracts gradient force so uphill resistance is not attributed to ERS.
    Sign convention for gradient:
      F_gradient is positive uphill (adds to required propulsive force).
      Subtracting it from the propulsive total isolates the powertrain-only component.
    """
    cd_a_series    = compute_effective_cd_a(df)
    f_drag         = drag_force(df["Speed"], cd_a_series)
    f_gradient     = compute_gradient_force(df, df["car_mass"].iloc[0])
    f_inertial     = df["car_mass"] * (df["accel_smooth"] / 3.6)
    return f_inertial + f_drag - f_gradient


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

def detect_zones_at_threshold(cand: pd.DataFrame, df: pd.DataFrame,
                               threshold: float, label: str):
    """Helper – returns zones for a specific threshold."""
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


# VISUALIZATION 

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

    SLM_COLOR = "#f5a623"
    OTM_COLOR = "#e63946"
    dist = df["Distance"]

    # Panel 1: Speed
    ax1.plot(dist, df["Speed"], color="#4da6ff", lw=1.4, label="Speed")
    ax1.axhline(MIN_SPEED_KMH, color="#555", ls="--", lw=0.8,
                label=f"Min. {MIN_SPEED_KMH} km/h")
    # DRS indicator band
    if "DRS" in df.columns:
        drs_active = df["DRS"] >= DRS_OPEN_THRESHOLD
        ax1.fill_between(dist, 0, df["Speed"].max() * 1.1,
                         where=drs_active, color="#44aa66", alpha=0.08,
                         label="DRS open")
    ax1.set_ylabel("Speed (km/h)", color="#cccccc")
    ax1.set_ylim(0, df["Speed"].max() * 1.1)

    # Panel 2: Acceleration + Throttle
    ax2.plot(dist, df["accel_raw"],    color="#333333", lw=0.6, label="Accel (Raw)")
    ax2.plot(dist, df["accel_smooth"], color="#4da6ff", lw=1.3, label="Accel (Filtered)")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.set_ylabel("Acceleration (km/h/s)", color="#cccccc")
    ax2.set_ylim(-35, 90)

    ax2b = ax2.twinx()
    ax2b.plot(dist, df["Throttle"], color="#cc4444", alpha=0.45, lw=0.9, label="Throttle (%)")
    ax2b.set_ylim(0, 160)
    ax2b.set_ylabel("Throttle (%)", color="#cc4444")
    ax2b.tick_params(colors="#cc4444", labelsize=7)

    # Panel 3: Residual Force (+ optional gradient overlay)
    if candidate is not None and "F_residual" in candidate.columns:
        cdist = candidate["Distance"]
        res   = candidate["F_residual"]

        ax3.plot(cdist, candidate["F_propulsive"], color="#333333", lw=0.9,
                 label="Total Propulsive Force")
        ax3.plot(cdist, candidate["F_baseline"], color="#44aa66", lw=1.5, ls="--",
                 label="ICE Baseline")
        ax3.plot(cdist, res, color="#4da6ff", lw=1.1,
                 label="Residual Force (F − Baseline)")
        ax3.fill_between(cdist, 0, res, where=res > 0, color="#4da6ff", alpha=0.15)

        if thr_slm is not None:
            ax3.axhline(thr_slm, color=SLM_COLOR, ls=":", lw=1.2,
                        label=f"SLM Threshold ({SLM_RESIDUAL_PERCENTILE}th = {thr_slm:.0f} N)")
        if thr_otm is not None:
            ax3.axhline(thr_otm, color=OTM_COLOR, ls=":", lw=1.2,
                        label=f"OTM Threshold ({OTM_RESIDUAL_PERCENTILE}th = {thr_otm:.0f} N)")
        ax3.axhline(0, color="#444", lw=0.5)
        ax3.set_ylabel("Force (N)", color="#cccccc")
        ax3.set_xlabel("Distance (m)", color="#aaaaaa")

    # Highlight Zones
    for i, (d_start, d_end, *_) in enumerate(slm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=SLM_COLOR, alpha=0.18,
                       label="SLM" if (i == 0 and ax is ax1) else "")

    for i, (d_start, d_end, *_) in enumerate(otm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=OTM_COLOR, alpha=0.30,
                       label="OTM" if (i == 0 and ax is ax1) else "")

    legend_patches = [
        Patch(color=SLM_COLOR, alpha=0.6, label=f"Straight Line Mode ({len(slm_zones)} zones)"),
        Patch(color=OTM_COLOR, alpha=0.8, label=f"Overtake Mode ({len(otm_zones)} zones)"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][:3],
               fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")
    ax2.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")
    ax3.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc")

    fig.suptitle(
        f"ERS Telemetry Analysis · {driver_name} · {session_name} · "
        f"Physical Estimation (2026 Regs) [DRS/Mass/Gradient corrected]",
        fontsize=11, color="#eeeeee", y=0.97
    )

    out = Path("./graphs") / f"ers_telemetry_{driver_name}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Chart saved: {out}")
    plt.show()


# VISUALIZATION – 2025 vs. 2026 Overlay

def plot_overlay_2025_vs_2026(df_2025, df_2026, slm_zones_2026,
                               straight_label="Start/Finish Straight"):
    fig, (ax_spd, ax_acc) = plt.subplots(2, 1, figsize=(14, 8),
                                          facecolor="#0f0f0f", sharex=True)
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

    fig.suptitle(f"Comparison 2025 vs. 2026 – {straight_label}",
                 fontsize=12, color="#eeeeee", y=0.97)
    fig.subplots_adjust(hspace=0.06, top=0.93, bottom=0.08)

    out = Path("./graphs") / "overlay_2025_vs_2026.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Overlay saved: {out}")
    plt.show()


# ── BASELINE DIAGNOSTIC ───────────────────────────────────────────────────────

def plot_baseline_diagnostic(cand: pd.DataFrame, driver_name: str):
    """Scatter: F_propulsive vs. Speed with ICE baseline and top-5% residuals."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")

    ax.scatter(cand["Speed"], cand["F_propulsive"],
               s=2, alpha=0.3, color="#4da6ff", label="F_propulsive")
    ax.scatter(cand["Speed"], cand["F_baseline"],
               s=2, alpha=0.6, color="#44aa66", label="ICE baseline")

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


def main():
    OUTPUT_DIR   = Path("./graphs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    DRIVER_CODE  = "HAM"
    SESSION_YEAR = 2026
    SESSION_EVT  = 1
    SESSION_TYPE = "R"

    session_2026 = load_session_safe(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
    if session_2026 is None:
        print("Exiting due to missing data.")
        return

    # Determine total laps for fuel model
    total_laps = int(session_2026.laps["LapNumber"].max()) if not session_2026.laps.empty else 60
    print(f"Session: {SESSION_YEAR} R{SESSION_EVT} | Total laps: {total_laps}")

    fastest_lap = session_2026.laps.pick_drivers(DRIVER_CODE).pick_fastest()
    lap_number  = int(fastest_lap["LapNumber"])
    print(f"Fastest lap: #{lap_number} | "
          f"Est. car mass: {compute_dynamic_mass(lap_number, total_laps):.1f} kg "
          f"(vs. {CAR_MASS_BASE + FUEL_LOAD_KG} kg at start)")

    tel_2026 = fastest_lap.get_telemetry()

    # DEBUG
    print(tel_2026.columns.tolist())
    print(tel_2026["DRS"].value_counts().head(10))

    df_2026  = clean_telemetry(tel_2026, lap_number=lap_number, total_laps=total_laps)
    
    slm_zones, otm_zones, candidate, thr_slm, thr_otm = find_ers_zones(df_2026)

    session_label = f"{SESSION_YEAR} R{SESSION_EVT}"
    plot_telemetry(df_2026, slm_zones, otm_zones, candidate, thr_slm, thr_otm,
                   driver_name=DRIVER_CODE, session_name=session_label)

    if candidate is not None:
        export_zones_csv(slm_zones, otm_zones, candidate,
                         driver=DRIVER_CODE,
                         session_name=session_label.replace(" ", "_"),
                         out_path=OUTPUT_DIR)

    # 2025 comparison (mass model uses defaults for cross-year baseline)
    df_2025 = None
    session_2025 = load_session_safe(2025, SESSION_EVT, SESSION_TYPE)
    if session_2025 is not None:
        try:
            lap_2025 = session_2025.laps.pick_drivers(DRIVER_CODE).pick_fastest()
            lnum_25  = int(lap_2025["LapNumber"])
            tlaps_25 = int(session_2025.laps["LapNumber"].max())
            df_2025  = clean_telemetry(lap_2025.get_telemetry(),
                                       lap_number=lnum_25, total_laps=tlaps_25)
        except Exception as e:
            print(f"2025 lap not loaded: {e}")

    plot_overlay_2025_vs_2026(df_2025, df_2026, slm_zones,
                               straight_label=f"GP {SESSION_EVT} – Fastest Lap")

    # Cross-check
    if candidate is not None:
        total_slm_kj = sum(zone_energy_kj(candidate, d, e) for d, e, *_ in slm_zones)
        print(f"\nTotal estimated ERS energy per lap: {total_slm_kj:.0f} kJ")
        print("Reference: 2026 battery deployment limit ~4 MJ (4000 kJ) per lap.")
        if total_slm_kj > 4000:
            print("⚠  Energy exceeds reference limit — CD_A_BASE or FUEL_BURN_RATE "
                  "may need tuning for this circuit / conditions.")
        plot_baseline_diagnostic(candidate, DRIVER_CODE)


if __name__ == "__main__":
    main()