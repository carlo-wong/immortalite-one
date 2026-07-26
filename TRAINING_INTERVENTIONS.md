# Training interventions & gates (successful)

Official log of **kept** one-knob changes and **PASS** strength gates that define the promote line.

Failed / inconclusive rewinds (e.g. 400 vs 380 tries) are **not** listed here — see `tmp/400_vs_380_tries.csv`.

Promote baseline: **`ckpt_iter_0380.pt`**.  
Rule: **one TRAIN knob per 20-iter block**.

---

## Kept interventions (quality / clip era → 380)

| When | Lever | Gate | Elo (CI) | Notes |
|------|--------|------|----------|-------|
| 161 | `value_target=root_q` | (series) | — | Live label recipe |
| 241 | move T=4 / 10 plies | (series PASS) | — | Sampling only |
| 261 | SP sims → **150** (gate 100) | (series PASS) | — | |
| 321 | LR → 2.0e-4 | 340 vs 320 | +92 [+38,+150] | Clip-cure ladder |
| 341 | LR → 1.5e-4 | 360 vs 340 @256 | +45 [+6,+85] | |
| 361 | LR → 1.0e-4; no ply cap | **380 vs 360** | **+67 [+28,+108]** | **Promote 380** |

---

## Official PASS gates (recent)

| A | B | Games | W–D–L | Elo | 95% CI | Verdict |
|---|---|-------|-------|-----|--------|---------|
| 360 | 340 | 256 | 123–43–90 | +45.04 | [+6.36, +84.87] | **PASS** |
| 380 | 360 | 256 | 132–41–83 | +67.33 | [+28.42, +107.99] | **PASS** |

Source of truth for machine gates: `results/metrics_gates.csv` (PASS/historical only as you choose to log).

---

## Current recipe (in flight — not yet gated)

| Lever | Status |
|--------|--------|
| `value_coef=1.5`, `root_q`, steps 800, sims 150 | Running (rewind from 380); gate 400 vs 380 pending |

On PASS, add a row above and append the gate to `metrics_gates.csv`. On FAIL/INC, put the gate row in `tmp/400_vs_380_tries.csv` only.

Last updated: 2026-07-26.
