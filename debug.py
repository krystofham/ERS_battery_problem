"""
Debug: DRS a Z kanál – FastF1 telemetrie
=========================================
v2: přidán Z noise cleaning pipeline
    - outlier removal (spike > 1m mezi vzorky)
    - median filter
    - SG smoothing
    Porovnání gradientu před a po čištění.
"""

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from pathlib import Path

# ── Konfigurace ───────────────────────────────────────────────────────────────
CACHE_PATH   = Path("./fastf1_cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))

SESSION_YEAR = 2026
SESSION_EVT  = 1
SESSION_TYPE = "R"
DRIVER_CODE  = "HAM"

SPIKE_THRESHOLD_M  = 2.0   # m – max realistický skok Z mezi sousedními vzorky
MEDIAN_KERNEL      = 11    # median filter kernel (musí být liché)
SG_WINDOW_FINAL    = 401   # SG po čištění
SG_POLYORDER       = 2

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Načítám session...")
session = fastf1.get_session(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
session.load()

lap = session.laps.pick_drivers(DRIVER_CODE).pick_fastest()
tel = lap.get_telemetry()

print(f"\nDriver : {DRIVER_CODE}")
print(f"Kolo   : #{int(lap['LapNumber'])}")
print(f"Vzorků : {len(tel)}")


# ── SEKCE 1: Všechny kanály ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DOSTUPNÉ KANÁLY")
print("=" * 60)
for col in tel.columns:
    n_null = tel[col].isna().sum()
    try:
        sample = tel[col].dropna().iloc[0]
    except IndexError:
        sample = "PRÁZDNÝ"
    print(f"  {col:<30} dtype={str(tel[col].dtype):<12} null={n_null:<4} ukázka={sample}")


# ── SEKCE 2: DRS ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DRS ANALÝZA")
print("=" * 60)

if "DRS" not in tel.columns:
    print("  ✗ DRS kanál NEEXISTUJE.")
else:
    drs = tel["DRS"]
    print(f"  Min / Max     : {drs.min()} / {drs.max()}")
    print(f"  Null hodnoty  : {drs.isna().sum()}")
    print()
    print("  Rozložení hodnot:")
    for val, cnt in drs.value_counts().sort_index().items():
        pct = 100 * cnt / len(drs)
        bar = "█" * int(pct / 2)
        print(f"    DRS={val:<4}  {cnt:>4} vzorků  ({pct:5.1f}%)  {bar}")
    print()
    for thr in [1, 8, 10]:
        n = (drs >= thr).sum()
        print(f"    >= {thr:<3}: {n} vzorků ({100*n/len(drs):.1f}%)")


# ── SEKCE 3: Z kanál – raw analýza ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Z KANÁL – RAW ANALÝZA")
print("=" * 60)

if "Z" not in tel.columns:
    print("  ✗ Z kanál NEEXISTUJE.")
else:
    z_vals = tel["Z"].ffill().bfill().values
    d_vals = tel["Distance"].values

    print(f"  Min výška      : {z_vals.min():.2f} m")
    print(f"  Max výška      : {z_vals.max():.2f} m")
    print(f"  Výškový rozsah : {z_vals.max() - z_vals.min():.2f} m")
    print(f"  Std            : {z_vals.std():.2f} m")

    dz_raw = np.abs(np.diff(z_vals))
    print(f"\n  Skok mezi sousedními vzorky (|ΔZ|):")
    print(f"    Max skok  : {dz_raw.max():.3f} m")
    print(f"    Průměr    : {dz_raw.mean():.3f} m")
    print(f"    > {SPIKE_THRESHOLD_M} m   : {(dz_raw > SPIKE_THRESHOLD_M).sum()} vzorků "
          f"({100*(dz_raw > SPIKE_THRESHOLD_M).mean():.1f}%) ← spiky")

    print(f"\n  Gradient (raw SG) při různých oknech:")
    print(f"  {'SG okno':<10} {'max °':>8} {'min °':>8} {'std °':>8}  komentář")
    print("  " + "-" * 55)
    for window in [31, 101, 201, 301, 501]:
        if window >= len(z_vals):
            continue
        z_s = savgol_filter(z_vals, window_length=window, polyorder=SG_POLYORDER)
        deg = np.degrees(np.arctan(np.gradient(z_s, d_vals)))
        tag = "✓ OK" if deg.max() < 5 else ("~ hraniční" if deg.max() < 15 else "⚠ šum")
        print(f"  {window:<10} {deg.max():>8.1f} {deg.min():>8.1f} {deg.std():>8.2f}  {tag}")


# ── SEKCE 4: Z noise cleaning pipeline ───────────────────────────────────────
print("\n" + "=" * 60)
print("Z NOISE CLEANING PIPELINE")
print("=" * 60)

if "Z" in tel.columns:

    # Krok 1: Spike outlier removal
    print(f"\n  Krok 1: Spike removal (threshold = {SPIKE_THRESHOLD_M} m)")
    z_step1 = z_vals.copy().astype(float)
    dz      = np.diff(z_step1)
    spikes  = np.abs(dz) > SPIKE_THRESHOLD_M
    spike_indices = np.where(spikes)[0] + 1
    z_step1[spike_indices] = np.nan
    n_removed = spikes.sum()
    print(f"    Odstraněno   : {n_removed} vzorků ({100*n_removed/len(z_vals):.1f}%)")
    z_step1 = pd.Series(z_step1).interpolate("linear").values
    print(f"    Interpolováno lineárně.")

    # Krok 2: Median filter
    print(f"\n  Krok 2: Median filter (kernel = {MEDIAN_KERNEL})")
    z_step2 = medfilt(z_step1, kernel_size=MEDIAN_KERNEL)
    diff_median = np.abs(z_step2 - z_step1).mean()
    print(f"    Průměrná změna oproti kroku 1: {diff_median:.3f} m")

    # Krok 3: SG smoothing
    print(f"\n  Krok 3: SG smoothing (okno = {SG_WINDOW_FINAL})")
    z_step3 = savgol_filter(z_step2, window_length=SG_WINDOW_FINAL, polyorder=SG_POLYORDER)
    diff_sg = np.abs(z_step3 - z_step2).mean()
    print(f"    Průměrná změna oproti kroku 2: {diff_sg:.3f} m")

    # Výsledný gradient
    grad_raw   = np.degrees(np.arctan(np.gradient(
                   savgol_filter(z_vals, 301, SG_POLYORDER), d_vals)))
    grad_clean = np.degrees(np.arctan(np.gradient(z_step3, d_vals)))

    print(f"\n  Porovnání gradientu před / po čištění:")
    print(f"  {'':25} {'před':>10} {'po':>10}")
    print(f"  {'Max gradient':25} {grad_raw.max():>10.1f}° {grad_clean.max():>10.1f}°")
    print(f"  {'Min gradient':25} {grad_raw.min():>10.1f}° {grad_clean.min():>10.1f}°")
    print(f"  {'Std':25} {grad_raw.std():>10.2f}° {grad_clean.std():>10.2f}°")

    mass = 776.2
    f_max_raw   = mass * 9.81 * np.sin(np.radians(abs(grad_raw.max())))
    f_max_clean = mass * 9.81 * np.sin(np.radians(abs(grad_clean.max())))
    print(f"\n  Max gradient síla (m = {mass} kg):")
    print(f"    Před čištěním : {f_max_raw:.0f} N")
    print(f"    Po čištění    : {f_max_clean:.0f} N")
    if grad_clean.max() < 5:
        print(f"    ✓ Gradient po čištění je fyzikálně věrohodný.")
    else:
        print(f"    ⚠ Gradient stále přesahuje 5° – zvažte deaktivaci Z korekce.")


# ── SEKCE 5: Grafy ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generuji grafy...")

Path("./graphs").mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 1, figsize=(14, 14), facecolor="#0f0f0f")
fig.suptitle(f"DEBUG Z noise cleaning · {DRIVER_CODE} · {SESSION_YEAR} R{SESSION_EVT}",
             color="#eeeeee", fontsize=12)

for ax in axes:
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.spines[:].set_color("#333333")

dist = tel["Distance"].values

# Panel 1: DRS
ax = axes[0]
if "DRS" in tel.columns:
    ax.plot(dist, tel["DRS"].values, color="#f5a623", lw=1.2, label="DRS raw")
    ax.axhline(10, color="#e63946", ls="--", lw=0.8, label=">= 10 open")
    ax.axhline(8,  color="#ffcc00", ls=":",  lw=0.8, label=">= 8 available")
ax.set_ylabel("DRS", color="#cccccc")
ax.set_title("DRS kanál", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

# Panel 2: Z výška – kroky čištění
ax = axes[1]
if "Z" in tel.columns:
    ax.plot(dist, z_vals,  color="#555555", lw=0.7, label="Raw Z")
    ax.plot(dist, z_step1, color="#4da6ff", lw=1.0, label="Po spike removal")
    ax.plot(dist, z_step2, color="#44aa66", lw=1.0, label=f"Po median ({MEDIAN_KERNEL})")
    ax.plot(dist, z_step3, color="#f5a623", lw=1.5, label=f"Po SG ({SG_WINDOW_FINAL})")
ax.set_ylabel("Výška (m)", color="#cccccc")
ax.set_title("Z výška – kroky čištění", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

# Panel 3: Gradient před čištěním
ax = axes[2]
if "Z" in tel.columns:
    for window, color, lw in [(31, "#4da6ff", 0.8), (101, "#44aa66", 1.0), (301, "#f5a623", 1.2)]:
        z_s = savgol_filter(z_vals, window_length=window, polyorder=SG_POLYORDER)
        ax.plot(dist, np.degrees(np.arctan(np.gradient(z_s, dist))),
                color=color, lw=lw, label=f"SG={window} (raw)")
    ax.axhline(0,   color="#444", lw=0.5)
    ax.axhline(5,   color="#555", ls=":", lw=0.8, label="±5°")
    ax.axhline(-5,  color="#555", ls=":", lw=0.8)
    ax.axhline(15,  color="#e63946", ls="--", lw=0.8, label="±15° clip")
    ax.axhline(-15, color="#e63946", ls="--", lw=0.8)
ax.set_ylabel("Gradient (°)", color="#cccccc")
ax.set_title("Gradient PŘED čištěním", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

# Panel 4: Gradient po čištění
ax = axes[3]
if "Z" in tel.columns and 'grad_clean' in dir():
    ax.plot(dist, grad_clean, color="#f5a623", lw=1.3, label="Po cleaning pipeline")
    ax.plot(dist, grad_raw,   color="#333333", lw=0.7, alpha=0.6, label="Před (SG=301)")
    ax.axhline(0,   color="#444", lw=0.5)
    ax.axhline(5,   color="#555", ls=":", lw=0.8, label="±5°")
    ax.axhline(-5,  color="#555", ls=":", lw=0.8)
    ax.axhline(15,  color="#e63946", ls="--", lw=0.8, label="±15°")
    ax.axhline(-15, color="#e63946", ls="--", lw=0.8)
ax.set_ylabel("Gradient (°)", color="#cccccc")
ax.set_xlabel("Distance (m)", color="#aaaaaa")
ax.set_title("Gradient PO čištění", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("./graphs/debug_drs_z.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"✓ Graf uložen: {out}")
plt.show()

print("\n" + "=" * 60)
print("Debug hotov.")