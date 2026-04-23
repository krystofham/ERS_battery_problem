# ERS Boost Zone Estimator – 2026 F1 Regulations
> This readme is mostly done with AI. Claude Sonnet 3.6
Physical estimation of ERS deployment zones from FastF1 telemetry.  
Developed as a response to [Fast-F1 issue #864](https://github.com/theOehrly/Fast-F1/issues/864).

---

## Problem

The 2026 F1 API does not expose binary ERS/aero mode flags (Straight Line Mode, Overtake Mode).  
This script estimates ERS activation zones from publicly available telemetry using a physical force model.

---

## Method

```
F_ERS = m·a + F_drag − F_ICE_baseline
```

1. **Acceleration** is computed from speed telemetry and smoothed with a Savitzky-Golay filter
2. **Total propulsive force** is derived from `F = m·a + F_aero + F_rolling`
3. **ICE baseline** is fitted using speed-bin percentiles — robust against ERS deployment itself
4. **Residual force** above the baseline is attributed to ERS
5. **Dual-threshold detection** separates two modes:
   - Straight Line Mode (SLM) — 65th percentile threshold
   - Overtake Mode (OTM) — 88th percentile threshold
6. **Energy** is estimated via `E = Σ (F_residual × v × Δt)` [kJ]

---

## Validation

Tested on 2026 Australian GP, VER and HAM fastest laps.

| Check | Result |
|---|---|
| Main straights detected | 3/3 matched within 35 m across both drivers |
| Estimated ERS energy | 2300–2940 kJ/lap (below 4000 kJ regulation limit) |
| Peak power (most zones) | 196–341 kW (below 350 kW MGU-K limit) |
| Cross-driver consistency | SLM zone starts agree to < 35 m (≈ telemetry sampling resolution) |

---

## Output

- **Telemetry plot** — speed, acceleration + throttle overlay, residual force with zone highlights
- **2025 vs. 2026 overlay** — normalized distance for cross-year comparison
- **CSV export** — zone metrics including `peak_power_kw` and `spike_suspect` flag
- **Baseline diagnostic plot** — F_propulsive vs. speed scatter for model validation

---

## Limitations

- Physical constants (`CD_A`, `CAR_MASS`) are estimates — see comments in code
- Baseline is relative, so `CD_A` has minimal effect on residuals (baseline absorbs the shift)
- Polynomial baseline can extrapolate between sparse speed bins — use `BASELINE_DEG = 2` for stability
- No ground truth available until the F1 API exposes binary ERS flags
- `spike_suspect = True` flags zones where estimated peak power exceeds 350 kW

---

## Requirements

```
fastf1
numpy
pandas
matplotlib
scipy
```

```bash
pip install fastf1 numpy pandas matplotlib scipy
```

---

## Usage

```python
# Edit these three lines in main():
DRIVER_CODE  = "VER"
SESSION_YEAR = 2026
SESSION_EVT  = 1        # race number or name, e.g. "Bahrain"
```

```bash
python ers_analysis_2026.py
```

Cache is stored in `./fastf1_cache/`. Outputs (PNG, CSV) are saved there too.

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `SLM_RESIDUAL_PERCENTILE` | 65 | Lower = more zones detected |
| `OTM_RESIDUAL_PERCENTILE` | 88 | Higher = only strong bursts |
| `MIN_ZONE_METERS` | 50 | Minimum zone length, filters noise |
| `MERGE_GAP_METERS` | 50 | Merges zones closer than this |
| `BASELINE_DEG` | 2 | Polynomial degree, 1–3 |
| `BASELINE_PERCENTILE` | 35 | Higher = more conservative baseline |

---


## License

MIT License — Copyright (c) 2026 Kryštof H.  
See [license](lincense) for full text.