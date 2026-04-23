"""
ERS Boost Zone Estimator – 2026 F1 Regulations
===============================================
Fyzikální odhad nasazení ERS z telemetrických dat FastF1.
Protože F1 API v roce 2026 neposkytuje binární ERS/aero flag,
rekonstruujeme aktivaci z dynamiky síly: F_ERS = m·a + F_drag − F_ICE_baseline.

Vylepšení oproti původní verzi:
  - Robustní detekce stability řazení (rolling std místo diff == 0)
  - Přesnější odhad energie ERS (integrál přes zónu, ne průměr × průměr)
  - Dvojprahová detekce: Straight Line Mode vs. Overtake Mode
  - Overlay 2025 vs 2026 srovnávací plot
  - Export zón do CSV
  - Normalizace vzdálenosti na 0–1 pro porovnání tratí
  - Ošetření chyb při načítání session
  - Komentáře o nejistotě fyzikálních konstant
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


# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_PATH = Path("./fastf1_cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# FYZIKÁLNÍ KONSTANTY (odhad pro 2026 předpisy)
# Nejistoty jsou uvedeny – mají přímý vliv na přesnost baseline.
# ══════════════════════════════════════════════════════════════════════════════

AIR_DENSITY  = 1.225   # kg/m³ – standardní atmosféra (±0.05 při 20 °C / 500 m n.m.)
CAR_MASS     = 768     # kg   – minimální hmotnost vč. jezdce (±5 kg)
CD_A         = 0.65    # m²   – Cd × čelní plocha; závisí na nastavení aero (±0.15 m²)
                       #        → ovlivňuje baseline ~8 % při 250 km/h
ROLLING_COEF = 0.015   # –    – valivý odpor (±0.003)
G            = 9.81    # m/s²


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRY DETEKCE
# ══════════════════════════════════════════════════════════════════════════════

MIN_SPEED_KMH        = 200    # km/h – minimální rychlost pro analýzu (přímky)
THROTTLE_MIN         = 99     # %    – plný plyn (WOT podmínka)
GEAR_STABLE_WINDOW   = 5      # vzorků – okno pro detekci stability řazení
SG_WINDOW            = 15     # vzorků – Savitzky-Golay okno (musí být liché)
SG_POLYORDER         = 3

BASELINE_DEG         = 3      # stupeň polynomu pro ICE baseline
BASELINE_N_BINS      = 20     # počet rychlostních binů
BASELINE_PERCENTILE  = 25     # percentil síly v každém binu → "podlaha ICE"

# Dvojprahová detekce – Straight Line Mode vs. Overtake Mode
# Overtake Mode = výrazně vyšší ERS výkon → vyšší percentil
SLM_RESIDUAL_PERCENTILE  = 65   # Straight Line Mode práh
OTM_RESIDUAL_PERCENTILE  = 88   # Overtake Mode práh

MIN_ZONE_METERS      = 50     # m – minimální délka zóny (filtr šumu)
MERGE_GAP_METERS     = 50     # m – sloučení sousedních zón


# ══════════════════════════════════════════════════════════════════════════════
# ČIŠTĚNÍ TELEMETRIE
# ══════════════════════════════════════════════════════════════════════════════

def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vyčistí telemetrii a přidá odvozené sloupce.
    Nově: detekce stability řazení přes rolling std (odolnější vůči index gaps).
    """
    cols = ["Speed", "Throttle", "Brake", "nGear", "Time", "Distance"]
    df = df[cols].dropna().copy().reset_index(drop=True)
    df["time_s"] = df["Time"].dt.total_seconds()

    dv = df["Speed"].diff()
    dt = df["time_s"].diff().replace(0, np.nan)
    df["accel_raw"] = dv / dt

    # Odstraň fyzikálně nesmyslné špičky (>150 km/h/s ≈ 41 m/s²)
    invalid = df["accel_raw"].abs() > 150
    df = df.iloc[1:][~invalid].reset_index(drop=True)

    # Savitzky-Golay vyhlazení zrychlení
    if len(df) > SG_WINDOW:
        df["accel_smooth"] = savgol_filter(
            df["accel_raw"].fillna(0),
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER,
        )
    else:
        df["accel_smooth"] = df["accel_raw"]

    # ── Detekce stability řazení ──────────────────────────────────────────────
    # Původní: diff().abs() == 0 → vynechává body při skoku indexu
    # Nově: rolling std → True pokud se řadicí stupeň nezměnil v okně GEAR_STABLE_WINDOW
    gear_std = df["nGear"].rolling(GEAR_STABLE_WINDOW, center=True, min_periods=1).std()
    df["gear_stable"] = (gear_std == 0)

    # Normalizovaná vzdálenost 0–1 pro porovnání tratí
    d_min, d_max = df["Distance"].min(), df["Distance"].max()
    df["dist_norm"] = (df["Distance"] - d_min) / (d_max - d_min) if d_max > d_min else 0.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FYZIKÁLNÍ MODEL
# ══════════════════════════════════════════════════════════════════════════════

def drag_force(v_kmh: pd.Series) -> pd.Series:
    """Aerodynamický + valivý odpor [N]."""
    v_ms = v_kmh / 3.6
    return 0.5 * AIR_DENSITY * CD_A * v_ms**2 + ROLLING_COEF * CAR_MASS * G


def compute_propulsive_force(df: pd.DataFrame) -> pd.Series:
    """Celková hnací síla [N] = m·a + F_odpor."""
    return CAR_MASS * (df["accel_smooth"] / 3.6) + drag_force(df["Speed"])


def fit_engine_baseline(speeds: pd.Series, forces: pd.Series):
    """
    Robustní ICE baseline přes percentily v rychlostních binech.
    Viz. komentář v původním kódu – tato metoda zůstává, je správná.
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


# ══════════════════════════════════════════════════════════════════════════════
# VÝPOČET ERS ENERGIE V ZÓNĚ (přesnější než průměr × průměr)
# ══════════════════════════════════════════════════════════════════════════════

def zone_energy_kj(cand: pd.DataFrame, d_start: float, d_end: float) -> float:
    """
    Odhadovaná ERS energie v zóně [kJ].

    Původně: mean_residual_force × mean_velocity × mean_duration
    Nově:    integrál P_ERS dt ≈ Σ (F_residual × v) × Δt
             → přesnější, protože F a v se v zóně mění

    P_ERS [W] = F_residual [N] × v [m/s]
    E_ERS [J] = ∫ P_ERS dt ≈ Σ P_ERS_i × Δt_i
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


# ══════════════════════════════════════════════════════════════════════════════
# DETEKCE ERS ZÓN (dvojprahová: SLM + OTM)
# ══════════════════════════════════════════════════════════════════════════════

def detect_zones_at_threshold(cand: pd.DataFrame, df: pd.DataFrame, threshold: float, label: str):
    """Pomocná funkce – vrátí zóny pro daný práh."""
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
    Hlavní funkce detekce. Vrací:
      slm_zones  – Straight Line Mode zóny
      otm_zones  – Overtake Mode zóny (podmnožina SLM)
      cand       – DataFrame kandidátů
      thr_slm    – práh SLM [N]
      thr_otm    – práh OTM [N]
    """
    mask = (
        (df["Throttle"] >= THROTTLE_MIN) &
        (df["Brake"] == 0) &
        (df["Speed"] >= MIN_SPEED_KMH) &
        (df["gear_stable"])  # <- nová detekce stability řazení
    )
    cand = df[mask].copy()

    if len(cand) < BASELINE_DEG + 2:
        print("⚠  Nedostatek dat pro analýzu.")
        return [], [], None, None, None

    cand["F_propulsive"] = compute_propulsive_force(cand)
    baseline, _          = fit_engine_baseline(cand["Speed"], cand["F_propulsive"])
    cand["F_baseline"]   = baseline
    cand["F_residual"]   = cand["F_propulsive"] - cand["F_baseline"]

    thr_slm = np.percentile(cand["F_residual"], SLM_RESIDUAL_PERCENTILE)
    thr_otm = np.percentile(cand["F_residual"], OTM_RESIDUAL_PERCENTILE)

    print(f"\n── Reziduální síla ──────────────────────────────────────")
    print(f"  Medián : {np.median(cand['F_residual']):+.0f} N")
    print(f"  IQR    : {np.percentile(cand['F_residual'],25):.0f} … {np.percentile(cand['F_residual'],75):.0f} N")
    print(f"  Std    : {cand['F_residual'].std():.0f} N")
    print(f"  Práh SLM ({SLM_RESIDUAL_PERCENTILE}. percentil) : {thr_slm:.0f} N")
    print(f"  Práh OTM ({OTM_RESIDUAL_PERCENTILE}. percentil) : {thr_otm:.0f} N")

    slm_zones = detect_zones_at_threshold(cand, df, thr_slm, "SLM")
    otm_zones = detect_zones_at_threshold(cand, df, thr_otm, "OTM")

    print(f"\n── Detekované zóny ──────────────────────────────────────")
    for label, zones in [("Straight Line Mode", slm_zones), ("Overtake Mode", otm_zones)]:
        print(f"\n  {label}: {len(zones)} zón")
        for i, (d_start, d_end, *_) in enumerate(zones):
            ekj = zone_energy_kj(cand, d_start, d_end)
            zone_mask = (cand["Distance"] >= d_start) & (cand["Distance"] <= d_end)
            mean_v = cand.loc[zone_mask, "Speed"].mean() if zone_mask.any() else float("nan")
            print(f"    Zóna {i+1}: {d_start:.0f} m → {d_end:.0f} m  "
                  f"(délka: {d_end-d_start:.0f} m, "
                  f"∅ rychlost: {mean_v:.0f} km/h, "
                  f"energie: {ekj:.1f} kJ)")

    return slm_zones, otm_zones, cand, thr_slm, thr_otm


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT DO CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_zones_csv(slm_zones, otm_zones, cand: pd.DataFrame,
                     driver: str, session_name: str, out_path: Path):
    """
    Exportuje detekované zóny do CSV pro další analýzu mimo matplotlib.
    Sloupce: driver, session, mode, d_start_m, d_end_m, length_m,
             mean_speed_kmh, peak_force_N, energy_kJ
    """
    rows = []
    for mode, zones, in [("SLM", slm_zones), ("OTM", otm_zones)]:
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
        print(f"\n✓ Export: {csv_path}")
        return df_out
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# VIZUALIZACE – hlavní telemetrický plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_telemetry(df, slm_zones, otm_zones, candidate, thr_slm, thr_otm,
                   driver_name="HAM", session_name="2026 R1"):
    """
    3panelový plot: rychlost | zrychlení + plyn | reziduální síla.
    Nově: overlay plynu na ose zrychlení, dvojbarevné zvýraznění zón.
    """
    fig = plt.figure(figsize=(17, 12), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.07,
                             top=0.93, bottom=0.06, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax1, ax2, ax3 = axes

    # Styl
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#cccccc")

    SLM_COLOR = "#f5a623"   # oranžová – Straight Line Mode
    OTM_COLOR = "#e63946"   # červená  – Overtake Mode
    dist = df["Distance"]

    # ── Panel 1: Rychlost ──────────────────────────────────────────────────
    ax1.plot(dist, df["Speed"], color="#4da6ff", lw=1.4, label="Rychlost")
    ax1.axhline(MIN_SPEED_KMH, color="#555", ls="--", lw=0.8,
                label=f"Min. {MIN_SPEED_KMH} km/h")
    ax1.set_ylabel("Rychlost (km/h)", color="#cccccc")
    ax1.set_ylim(0, df["Speed"].max() * 1.1)

    # ── Panel 2: Zrychlení + plyn ──────────────────────────────────────────
    ax2.plot(dist, df["accel_raw"],    color="#333333", lw=0.6, label="Zrychlení (raw)")
    ax2.plot(dist, df["accel_smooth"], color="#4da6ff", lw=1.3,
             label="Zrychlení (SG filtr)")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.set_ylabel("Zrychlení (km/h/s)", color="#cccccc")
    ax2.set_ylim(-35, 90)

    # Overlay plynu na pravé ose
    ax2b = ax2.twinx()
    ax2b.plot(dist, df["Throttle"], color="#cc4444", alpha=0.45, lw=0.9,
              label="Plyn (%)")
    ax2b.set_ylim(0, 160)
    ax2b.set_ylabel("Plyn (%)", color="#cc4444")
    ax2b.tick_params(colors="#cc4444", labelsize=7)
    ax2b.spines[:].set_color("#333333")

    # ── Panel 3: Reziduální síla ───────────────────────────────────────────
    if candidate is not None and "F_residual" in candidate.columns:
        cdist = candidate["Distance"]
        res   = candidate["F_residual"]

        ax3.plot(cdist, candidate["F_propulsive"],
                 color="#333333", lw=0.9, label="Celková hnací síla")
        ax3.plot(cdist, candidate["F_baseline"],
                 color="#44aa66", lw=1.5, ls="--", label="ICE baseline (percentil)")
        ax3.plot(cdist, res, color="#4da6ff", lw=1.1,
                 label="Reziduální síla (F − baseline)")
        ax3.fill_between(cdist, 0, res, where=res > 0,
                         color="#4da6ff", alpha=0.15)

        if thr_slm is not None:
            ax3.axhline(thr_slm, color=SLM_COLOR, ls=":", lw=1.2,
                        label=f"Práh SLM ({SLM_RESIDUAL_PERCENTILE}. perc. = {thr_slm:.0f} N)")
        if thr_otm is not None:
            ax3.axhline(thr_otm, color=OTM_COLOR, ls=":", lw=1.2,
                        label=f"Práh OTM ({OTM_RESIDUAL_PERCENTILE}. perc. = {thr_otm:.0f} N)")
        ax3.axhline(0, color="#444", lw=0.5)
        ax3.set_ylabel("Síla (N)", color="#cccccc")
        ax3.set_xlabel("Vzdálenost (m)", color="#aaaaaa")

    # ── Zvýraznění zón ────────────────────────────────────────────────────
    for i, (d_start, d_end, *_) in enumerate(slm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=SLM_COLOR, alpha=0.18,
                       label="SLM" if (i == 0 and ax is ax1) else "")

    for i, (d_start, d_end, *_) in enumerate(otm_zones):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(d_start, d_end, color=OTM_COLOR, alpha=0.30,
                       label="OTM" if (i == 0 and ax is ax1) else "")

    # Sdílené legendy
    legend_patches = [
        Patch(color=SLM_COLOR, alpha=0.6, label=f"Straight Line Mode ({len(slm_zones)} zón)"),
        Patch(color=OTM_COLOR, alpha=0.8, label=f"Overtake Mode ({len(otm_zones)} zón)"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][:2],
               fontsize=8, loc="upper left",
               facecolor="#1a1a1a", edgecolor="#333", labelcolor="#cccccc")
    ax2.legend(fontsize=8, loc="upper left",
               facecolor="#1a1a1a", edgecolor="#333", labelcolor="#cccccc")
    ax3.legend(fontsize=8, loc="upper left",
               facecolor="#1a1a1a", edgecolor="#333", labelcolor="#cccccc")

    fig.suptitle(
        f"Telemetrická analýza ERS  ·  {driver_name}  ·  {session_name}  "
        f"·  fyzikální odhad (2026 předpisy)",
        fontsize=12, color="#eeeeee", y=0.97
    )

    out = Path("./fastf1_cache") / f"ers_telemetry_{driver_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Graf uložen: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# VIZUALIZACE – overlay 2025 vs. 2026
# ══════════════════════════════════════════════════════════════════════════════

def plot_overlay_2025_vs_2026(df_2025: pd.DataFrame | None,
                               df_2026: pd.DataFrame,
                               slm_zones_2026,
                               straight_label: str = "Start/Finish Straight"):
    """
    Srovnávací plot rychlostních profilů 2025 vs. 2026.

    Protože nemáme ERS flag, ukazujeme, jak se změnil tvar
    akcelerace na přímce – viditelný efekt 'Straight Line Mode'.

    df_2025: volitelný (None = zobrazí se jen 2026)
    Normalizovaná vzdálenost (0–1) umožňuje porovnání i přes různé
    délky kola nebo pozice detektoru.
    """
    fig, (ax_spd, ax_acc) = plt.subplots(2, 1, figsize=(14, 8),
                                          facecolor="#0f0f0f", sharex=True)
    for ax in [ax_spd, ax_acc]:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#333333")

    x_2026 = df_2026["dist_norm"]

    ax_spd.plot(x_2026, df_2026["Speed"],
                color="#e63946", lw=1.5, label="2026")
    ax_acc.plot(x_2026, df_2026["accel_smooth"],
                color="#e63946", lw=1.3, label="2026")

    if df_2025 is not None:
        x_2025 = df_2025["dist_norm"]
        ax_spd.plot(x_2025, df_2025["Speed"],
                    color="#4da6ff", lw=1.5, ls="--", label="2025")
        ax_acc.plot(x_2025, df_2025["accel_smooth"],
                    color="#4da6ff", lw=1.3, ls="--", label="2025")

    # Zvýraznění SLM zón (normalizovaně)
    d_min = df_2026["Distance"].min()
    d_max = df_2026["Distance"].max()
    for i, (d_start, d_end, *_) in enumerate(slm_zones_2026):
        n_start = (d_start - d_min) / (d_max - d_min) if d_max > d_min else 0
        n_end   = (d_end   - d_min) / (d_max - d_min) if d_max > d_min else 1
        for ax in [ax_spd, ax_acc]:
            ax.axvspan(n_start, n_end, color="#f5a623", alpha=0.20,
                       label="Est. SLM (2026)" if i == 0 and ax is ax_spd else "")

    ax_spd.set_ylabel("Rychlost (km/h)", color="#cccccc")
    ax_acc.set_ylabel("Zrychlení (km/h/s)", color="#cccccc")
    ax_acc.set_xlabel("Normalizovaná vzdálenost (0 = start, 1 = cíl)", color="#aaaaaa")
    ax_spd.axhline(MIN_SPEED_KMH, color="#555", ls=":", lw=0.8)
    ax_acc.axhline(0, color="#555", lw=0.6)

    for ax in [ax_spd, ax_acc]:
        ax.legend(fontsize=8, facecolor="#1a1a1a",
                  edgecolor="#333", labelcolor="#cccccc")

    fig.suptitle(f"Srovnání 2025 vs. 2026 – {straight_label}",
                 fontsize=12, color="#eeeeee", y=0.97)
    fig.subplots_adjust(hspace=0.06, top=0.93, bottom=0.08)

    out = Path("./fastf1_cache") / "overlay_2025_vs_2026.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"✓ Overlay uložen: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# HLAVNÍ SPUŠTĚNÍ
# ══════════════════════════════════════════════════════════════════════════════

def load_session_safe(year: int, event, session_type: str):
    """Načte session s ošetřením chyb."""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"❌ Načtení session {year} / {event} / {session_type} selhalo: {e}")
        return None


def main():
    OUTPUT_DIR   = Path("./fastf1_cache")
    DRIVER_CODE  = "HAM"
    SESSION_YEAR = 2026
    SESSION_EVT  = 1          # číslo nebo název závodu
    SESSION_TYPE = "R"

    # ── Načtení 2026 ──────────────────────────────────────────────────────
    session_2026 = load_session_safe(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
    if session_2026 is None:
        print("Spouštím bez dat – ukončuji.")
        return

    lap_2026 = session_2026.laps.pick_drivers(DRIVER_CODE).pick_fastest()
    tel_2026  = lap_2026.get_telemetry()
    df_2026   = clean_telemetry(tel_2026)

    # ── Detekce ERS zón ───────────────────────────────────────────────────
    slm_zones, otm_zones, candidate, thr_slm, thr_otm = find_ers_zones(df_2026)

    # ── Telemetrický plot ─────────────────────────────────────────────────
    session_label = f"{SESSION_YEAR} R{SESSION_EVT}"
    plot_telemetry(df_2026, slm_zones, otm_zones, candidate,
                   thr_slm, thr_otm,
                   driver_name=DRIVER_CODE,
                   session_name=session_label)

    # ── Export CSV ────────────────────────────────────────────────────────
    if candidate is not None:
        export_zones_csv(slm_zones, otm_zones, candidate,
                         driver=DRIVER_CODE,
                         session_name=session_label.replace(" ", "_"),
                         out_path=OUTPUT_DIR)

    # ── Overlay 2025 vs. 2026 (volitelné) ────────────────────────────────
    # Pokud máš 2025 data stejného závodu, načti je tady:
    df_2025 = None
    session_2025 = load_session_safe(2025, SESSION_EVT, SESSION_TYPE)
    if session_2025 is not None:
        try:
            lap_2025 = session_2025.laps.pick_drivers(DRIVER_CODE).pick_fastest()
            df_2025  = clean_telemetry(lap_2025.get_telemetry())
        except Exception as e:
            print(f"⚠  2025 kolo nenačteno: {e}")

    plot_overlay_2025_vs_2026(df_2025, df_2026, slm_zones,
                               straight_label=f"GP {SESSION_EVT} – nejrychlejší kolo")

    # V main() přidej výpis pro cross-check:
    total_slm_kj = sum(
        zone_energy_kj(candidate, d, e)
        for d, e, *_ in slm_zones
    )
    print(f"Celková odhadovaná ERS energie za kolo: {total_slm_kj:.0f} kJ")
    # Porovnej s limitem: 2026 baterie ~4 MJ za kolo deployment
    # Pokud vychází výrazně přes 4000 kJ → baseline je příliš nízká (CD_A podhodnoceno)
if __name__ == "__main__":
    main()