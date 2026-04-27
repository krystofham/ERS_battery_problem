"""
Debug: DRS a Z kanál – FastF1 telemetrie
=========================================
v3: Z_CLAMPED kill-switch na úrovni modulu
    - definován ihned po načtení telemetrie
    - sekce 4 a grafy podmíněny not Z_CLAMPED
"""

import fastf1
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d  
from pathlib import Path

# ── Konfigurace ───────────────────────────────────────────────────────────────
CACHE_PATH   = Path("./fastf1_cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))

SESSION_YEAR = 2026
SESSION_EVT  = 1
SESSION_TYPE = "R"
DRIVER_CODE  = "VER"

SPIKE_THRESHOLD_M  = 2.0
MEDIAN_KERNEL      = 11
SG_WINDOW_FINAL    = 101
SG_POLYORDER       = 2

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Načítám session...")
session = fastf1.get_session(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
session.load()


lap = (session.laps
       .pick_drivers(DRIVER_CODE)
       .loc[session.laps["TrackStatus"] == "1"]
       .pick_fastest())
tel = lap.get_telemetry()

# ── Z track median přes všechna kola + smooth track map ──────────────────────
all_tel = session.laps.pick_drivers(DRIVER_CODE).get_telemetry()
BIN_SIZE = 10
bins = np.arange(0, all_tel["Distance"].max() + BIN_SIZE, BIN_SIZE)
all_tel["DistBin"] = pd.cut(all_tel["Distance"], bins=bins, labels=bins[:-1])

# OPRAVA: Použijeme pouze observed=False, aby délka odpovídala bins[:-1]
z_track_series = all_tel.groupby("DistBin", observed=False)["Z"].median()

# Vytvoření tabulky mapy trati
track_map = pd.DataFrame({
    "DistBin": bins[:-1].astype(float),
    "Z_raw": z_track_series.values
})

# Vyplnění děr (pokud v binu nikdo nejel) a vyhlazení
track_map["Z_raw"] = track_map["Z_raw"].ffill().bfill()
track_map["Z_smooth"] = gaussian_filter1d(track_map["Z_raw"].values, sigma=3)

# Interpolace výšky z mapy trati do telemetrie konkrétního kola
z_vals_clean = np.interp(
    tel["Distance"].values, 
    track_map["DistBin"].values, 
    track_map["Z_smooth"].values
)
# ── Z_CLAMPED – definováno na úrovni modulu ───────────────────────────────────
Z_CLAMPED = "Z" in tel.columns and tel["Z"].max() >= 99.9
Z_SOURCE   = "track_median" if Z_CLAMPED else "raw"

print(f"\nDriver : {DRIVER_CODE}")
print(f"Kolo   : #{int(lap['LapNumber'])}")
print(f"Vzorků : {len(tel)}")
print(f"Z_CLAMPED: {Z_CLAMPED}")

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
    z_vals = None
    d_vals = None
else:
    z_vals = tel["Z"].ffill().bfill().values
    d_vals = tel["Distance"].values

    print(f"  Min výška      : {z_vals.min():.2f} m")
    print(f"  Max výška      : {z_vals.max():.2f} m")
    print(f"  Výškový rozsah : {z_vals.max() - z_vals.min():.2f} m")
    print(f"  Std            : {z_vals.std():.2f} m")

    if Z_CLAMPED:
        print(f"\n  ✗ POZOR: Max Z = přesně {z_vals.max():.1f} m – datový clamp detekován!")
        print(f"    Z data jsou nepoužitelná. Pipeline a Z korekce VYPNUTY.")
    else:
        dz_raw = np.abs(np.diff(z_vals))
        print(f"\n  Skok mezi sousedními vzorky (|ΔZ|):")
        print(f"    Max skok  : {dz_raw.max():.3f} m")
        print(f"    Průměr    : {dz_raw.mean():.3f} m")
        print(f"    > {SPIKE_THRESHOLD_M} m   : {(dz_raw > SPIKE_THRESHOLD_M).sum()} vzorků "
              f"({100*(dz_raw > SPIKE_THRESHOLD_M).mean():.1f}%) ← spiky")

        print(f"\n  Gradient (raw SG) při různých oknech:")
        print(f"  {'SG okno':<10} {'max °':>8} {'min °':>8} {'std °':>8}  komentář")
        print("  " + "-" * 55)
        for window in [201, 401, 601, 801]:
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

grad_clean = None
grad_raw   = None

if "Z" not in tel.columns:
    print("  ✗ Z kanál chybí – pipeline přeskočena.")
else:
    # 0. Příprava vstupních dat
    if Z_CLAMPED:
        print(f"  ⚠ Z data clamped – vstup nahrazen track medianem.")
        z_input = z_vals_clean   # Délka odpovídá tel
    else:
        z_input = tel["Z"].ffill().bfill().values

    # Vzdálenosti pro gradient
    dist_tel = tel["Distance"].values
    dd = np.gradient(dist_tel)
    dd[dd == 0] = 0.1 # Ochrana proti dělení nulou

    # VÝPOČET GRAD_RAW (pro porovnání v tabulce)
    dz_raw_in = np.gradient(z_input)
    grad_raw = np.degrees(np.arctan2(dz_raw_in, dd))

    # Krok 1: Spike outlier removal
    print(f"\n  Krok 1: Spike removal (threshold = {SPIKE_THRESHOLD_M} m)")
    z_step1 = z_input.copy().astype(float)
    dz_diff = np.diff(z_step1, prepend=z_step1[0])
    spikes  = np.abs(dz_diff) > SPIKE_THRESHOLD_M
    z_step1[spikes] = np.nan
    n_removed = np.isnan(z_step1).sum()
    z_step1 = pd.Series(z_step1).interpolate("linear").ffill().bfill().values
    print(f"    Odstraněno   : {n_removed} vzorků")

    # Krok 2: Median filter
    print(f"  Krok 2: Median filter (kernel = {MEDIAN_KERNEL})")
    z_step2 = medfilt(z_step1, kernel_size=MEDIAN_KERNEL)

    # Krok 3: Gaussian smoothing
    print(f"  Krok 3: Gaussian smoothing (sigma=105)")
    z_step3 = gaussian_filter1d(z_step2, sigma=105)
    
    # Uložení finální výšky
    tel["Z_final"] = z_step3

    # Krok 4: Výpočet vyčištěného gradientu
    dz_clean = np.gradient(tel["Z_final"].values)
    grad_clean = np.degrees(np.arctan2(dz_clean, dd))
    tel["Gradient_Deg"] = grad_clean

    # --- KONTROLA A VÝPISY ---
    critical = (np.abs(grad_clean) > 15).sum()
    print(f"\n  Kritické body (>15°): {critical}")
    print(f"  Průměrná vzdálenost mezi vzorky: {dd.mean():.4f} m")

    print(f"\n  Porovnání gradientu před / po čištění:")
    print(f"  {'':25} {'před':>10} {'po':>10}")
    # Teď už grad_raw i grad_clean existují jako pole
    print(f"  {'Max gradient':25} {grad_raw.max():>10.1f}° {grad_clean.max():>10.1f}°")
    print(f"  {'Min gradient':25} {grad_raw.min():>10.1f}° {grad_clean.min():>10.1f}°")
    print(f"  {'Std':25} {grad_raw.std():>10.2f}° {grad_clean.std():>10.2f}°")
    print(f"  {'Kritické body >15°':25} {'—':>10} {critical:>10}")

    mass = 776.2
    f_max_clean = mass * 9.81 * np.sin(np.radians(abs(grad_clean.max())))
    print(f"\n  Max gradient síla (m = {mass} kg):")
    print(f"    Po čištění    : {f_max_clean:.0f} N")

# ── SEKCE 5: Grafy ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generuji grafy...")

Path("./graphs").mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 1, figsize=(14, 14), facecolor="#0f0f0f")
fig.suptitle(f"DEBUG Z noise cleaning · {DRIVER_CODE} · {SESSION_YEAR} R{SESSION_EVT}"
             + (" [Z CLAMPED – VYPNUTO]" if Z_CLAMPED else ""),
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

# Panel 2: Z výška
ax = axes[1]
if "Z" in tel.columns and grad_clean is not None:
    ax.plot(dist, z_input,      color="#555555", lw=0.7, label=f"Vstup ({'track median' if Z_CLAMPED else 'raw Z'})")
    ax.plot(dist, z_vals_clean, color="#4da6ff", lw=1.2, label="Median přes všechna kola")
    ax.plot(dist, z_step3,      color="#f5a623", lw=1.5, label=f"Po SG ({SG_WINDOW_FINAL})")
ax.set_ylabel("Výška (m)", color="#cccccc")
ax.set_title("Z výška – kroky čištění", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")

# Panel 3: Gradient před čištěním
ax = axes[2]
if "Z" in tel.columns and grad_raw is not None:
    # Použijeme jen opravdu velká okna, menší nemají smysl
    for window, color, lw in [(201, "#4da6ff", 0.8), (501, "#44aa66", 1.0), (801, "#f5a623", 1.2)]:
        if window < len(z_input):
            z_s = savgol_filter(z_input, window_length=window, polyorder=2)
            # DŮLEŽITÉ: Používáme np.arctan2 pro stabilitu
            temp_grad = np.degrees(np.arctan2(np.gradient(z_s), np.gradient(dist)))
            ax.plot(dist, temp_grad, color=color, lw=lw, label=f"SG={window}")
    
    ax.set_ylim(-20, 20)  # OŘÍZNUTÍ OSY Y - neuvidíš ty 60° nesmysly
    ax.axhline(15,  color="#e63946", ls="--", lw=0.8, label="±15° Limit")
    ax.axhline(-15, color="#e63946", ls="--", lw=0.8)
    ax.set_title("Gradient před čištěním (Savgol filtry - limitováno na ±20°)", color="#aaaaaa", fontsize=9)
# Panel 4: Gradient po čištění
ax = axes[3]
if grad_clean is not None:
    ax.plot(dist, grad_clean, color="#f5a623", lw=1.3, label="Po cleaning pipeline")
    ax.plot(dist, grad_raw,   color="#333333", lw=0.7, alpha=0.6, label="Před (SG=501)")
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