"""
Debug: DRS a Z kanál – FastF1 telemetrie
=========================================
Vypíše podrobnou analýzu DRS a Z (výška) kanálu pro nejrychlejší kolo.
Spusť samostatně, nezávisle na hlavním modelu.
"""

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path

# ── Konfigurace ───────────────────────────────────────────────────────────────
CACHE_PATH   = Path("./fastf1_cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))

SESSION_YEAR = 2026
SESSION_EVT  = 1
SESSION_TYPE = "R"
DRIVER_CODE  = "HAM"

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Načítám session...")
session = fastf1.get_session(SESSION_YEAR, SESSION_EVT, SESSION_TYPE)
session.load()

lap = session.laps.pick_drivers(DRIVER_CODE).pick_fastest()
tel = lap.get_telemetry()

print(f"\nDriver : {DRIVER_CODE}")
print(f"Kolo   : #{int(lap['LapNumber'])}")
print(f"Čas    : {lap['LapTime']}")
print(f"Vzorků : {len(tel)}")


# ── SEKCE 1: Všechny dostupné kanály ─────────────────────────────────────────
print("\n" + "=" * 60)
print("DOSTUPNÉ KANÁLY")
print("=" * 60)
for col in tel.columns:
    dtype = tel[col].dtype
    n_null = tel[col].isna().sum()
    try:
        sample = tel[col].dropna().iloc[0]
    except IndexError:
        sample = "PRÁZDNÝ"
    print(f"  {col:<30} dtype={str(dtype):<12} null={n_null:<4} ukázka={sample}")


# ── SEKCE 2: DRS analýza ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DRS ANALÝZA")
print("=" * 60)

if "DRS" not in tel.columns:
    print("  ✗ DRS kanál NEEXISTUJE v telemetrii.")
else:
    drs = tel["DRS"]
    print(f"  Datový typ    : {drs.dtype}")
    print(f"  Null hodnoty  : {drs.isna().sum()} z {len(drs)}")
    print(f"  Min / Max     : {drs.min()} / {drs.max()}")
    print(f"  Průměr        : {drs.mean():.2f}")
    print()
    print("  Rozložení hodnot (value_counts):")
    vc = drs.value_counts().sort_index()
    for val, cnt in vc.items():
        pct = 100 * cnt / len(drs)
        bar = "█" * int(pct / 2)
        print(f"    DRS={val:<4}  {cnt:>4} vzorků  ({pct:5.1f}%)  {bar}")

    print()
    # Interpretace podle FastF1 konvence
    print("  Interpretace FastF1 DRS hodnot:")
    print("    0        = DRS zavřené / nedostupné")
    print("    8        = DRS dostupné (detection point překročen)")
    print("    10/12/14 = DRS otevřené (hardware-závislé)")

    for threshold in [1, 8, 10]:
        n = (drs >= threshold).sum()
        print(f"    >= {threshold:<3}: {n} vzorků ({100*n/len(drs):.1f}%)")


# ── SEKCE 3: Z kanál (výška) analýza ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Z KANÁL (VÝŠKA TRATI) ANALÝZA")
print("=" * 60)

if "Z" not in tel.columns:
    print("  ✗ Z kanál NEEXISTUJE v telemetrii.")
else:
    z = tel["Z"].dropna()
    d = tel["Distance"].dropna()

    print(f"  Datový typ       : {tel['Z'].dtype}")
    print(f"  Null hodnoty     : {tel['Z'].isna().sum()} z {len(tel)}")
    print(f"  Min výška        : {z.min():.2f} m")
    print(f"  Max výška        : {z.max():.2f} m")
    print(f"  Výškový rozsah  : {z.max() - z.min():.2f} m")
    print(f"  Průměr           : {z.mean():.2f} m")
    print(f"  Std              : {z.std():.2f} m")

    # Gradient analýza při různých SG oknech
    print()
    print("  Gradient při různých SG smoothing oknech:")
    print(f"  {'SG okno':<10} {'max °':>8} {'min °':>8} {'std °':>8}  komentář")
    print("  " + "-" * 55)

    z_vals = tel["Z"].fillna(method="ffill").fillna(method="bfill").values
    d_vals = tel["Distance"].values

    for window in [11, 31, 51, 101, 201, 301, 501]:
        if window >= len(z_vals):
            continue
        z_s = savgol_filter(z_vals, window_length=window, polyorder=3)
        dz  = np.gradient(z_s, d_vals)
        deg = np.degrees(np.arctan(dz))
        comment = ""
        if deg.max() > 15:
            comment = "⚠ přesahuje 15° → stále šum"
        elif deg.max() > 5:
            comment = "⚠ přesahuje 5° → možný šum"
        elif deg.max() < 3:
            comment = "✓ realistické pro rovnou trať"
        else:
            comment = "~ hraniční"
        print(f"  {window:<10} {deg.max():>8.1f} {deg.min():>8.1f} {deg.std():>8.2f}  {comment}")

    # Najdi optimální okno (první kde max < 5°)
    print()
    for window in range(11, 600, 10):
        if window >= len(z_vals):
            break
        z_s = savgol_filter(z_vals, window_length=window, polyorder=3)
        dz  = np.gradient(z_s, d_vals)
        deg = np.degrees(np.arctan(dz))
        if deg.max() < 5.0:
            print(f"  → Doporučené min. okno pro max < 5°: SG_WINDOW = {window}")
            break
    else:
        print("  → Ani okno 600 nedosáhlo max < 5°. Z kanál je pravděpodobně šum.")


# ── SEKCE 4: Grafy ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generuji grafy...")

Path("./graphs").mkdir(exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#0f0f0f")
fig.suptitle(f"DEBUG – DRS + Z kanál · {DRIVER_CODE} · {SESSION_YEAR} R{SESSION_EVT}",
             color="#eeeeee", fontsize=12)

for ax in axes:
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.spines[:].set_color("#333333")

dist = tel["Distance"]

# Panel 1: DRS přes celé kolo
ax1 = axes[0]
if "DRS" in tel.columns:
    ax1.plot(dist, tel["DRS"], color="#f5a623", lw=1.2, label="DRS raw hodnota")
    ax1.axhline(10, color="#e63946", ls="--", lw=0.8, label="Threshold >= 10 (open)")
    ax1.axhline(8,  color="#ffcc00", ls=":",  lw=0.8, label="Threshold >= 8 (available)")
    ax1.set_ylabel("DRS hodnota", color="#cccccc")
    ax1.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")
ax1.set_title("DRS kanál", color="#aaaaaa", fontsize=9)

# Panel 2: Z výška surová vs. vyhlazená
ax2 = axes[1]
if "Z" in tel.columns:
    z_raw = tel["Z"].fillna(method="ffill").fillna(method="bfill").values
    d_vals = tel["Distance"].values
    ax2.plot(dist, z_raw, color="#333333", lw=0.8, label="Z raw")
    for window, color in [(31, "#4da6ff"), (101, "#44aa66"), (301, "#f5a623")]:
        if window < len(z_raw):
            z_s = savgol_filter(z_raw, window_length=window, polyorder=3)
            ax2.plot(d_vals, z_s, lw=1.2, color=color, label=f"SG={window}")
    ax2.set_ylabel("Výška (m)", color="#cccccc")
    ax2.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")
ax2.set_title("Z výška – raw vs. vyhlazená", color="#aaaaaa", fontsize=9)

# Panel 3: Gradient úhel při různých oknech
ax3 = axes[2]
if "Z" in tel.columns:
    for window, color in [(31, "#4da6ff"), (101, "#44aa66"), (301, "#f5a623")]:
        if window < len(z_raw):
            z_s  = savgol_filter(z_raw, window_length=window, polyorder=3)
            dz   = np.gradient(z_s, d_vals)
            deg  = np.degrees(np.arctan(dz))
            ax3.plot(d_vals, deg, lw=1.0, color=color, label=f"SG={window}")
    ax3.axhline(0,  color="#444", lw=0.5)
    ax3.axhline(15, color="#e63946", ls="--", lw=0.8, label="±15° clip")
    ax3.axhline(-15,color="#e63946", ls="--", lw=0.8)
    ax3.axhline(3,  color="#555",   ls=":",  lw=0.8, label="±3° (realistic AUS)")
    ax3.axhline(-3, color="#555",   ls=":",  lw=0.8)
    ax3.set_ylabel("Gradient (°)", color="#cccccc")
    ax3.set_xlabel("Distance (m)", color="#aaaaaa")
    ax3.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")
ax3.set_title("Gradient úhel trati", color="#aaaaaa", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("./graphs/debug_drs_z.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"✓ Graf uložen: {out}")
plt.show()

print("\n" + "=" * 60)
print("Debug hotov.")