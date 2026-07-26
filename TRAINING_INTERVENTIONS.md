# Training interventions & gates

Living log of **one-knob experiments** and strength-gate outcomes.  
Sources: `results/metrics_gates.csv`, `TRAINING_CHANGELOG.md`, chat pastes (noted).

Promote baseline after 380: **`ckpt_iter_0380.pt`**.  
Rule: **one TRAIN knob per 20-iter block**. Forbidden: sims↑ bundled with other levers, `grad_clip`↑, `claim_draw=False`, `gumbel_c_scale≠0.1`, optimizer reset.

---

## Interventions (recent quality / clip era → now)

| When | Lever (one change) | vs baseline | Gate | Elo (CI) | Outcome | Notes |
|------|-------------------|-------------|------|----------|---------|-------|
| 161§ | sims 200 + bugs (`c_scale=1.0`, worst-child root Q) | rewind 160 | — | — | **discarded** | Corrupted π/Q targets |
| 161¶ | `claim_draw=False` in search | rewind 160 | 180 vs 160 | **−112** | **FAIL / discard** | 17× repetition draws |
| 161 | `value_target=root_q` | outcome labels | (kept) | — | **kept** | Live since 161 |
| 241 | move T=4 / 10 plies | root_q | (series PASS) | — | **kept** | Sampling only |
| 261 | SP sims 100→**150** (gate 100) | 241 | (series PASS) | — | **kept** | |
| 321 | LR → 2.0e-4 | 261 | 340 vs 320 | +92 [+38,+150] | **PASS** | Clip-cure ladder |
| 341 | LR → 1.5e-4 | 321 | 360 vs 340 @256 | +45 [+6,+85] | **PASS** | Grad still hot |
| 361 | LR → 1.0e-4; no ply cap | 341 | 380 vs 360 | +67 [+28,+108] | **PASS** | Promote **380** |
| **381ª** | soft **Q+Z α=0.5** + cold shard flush | 380 / root_q | **400 vs 380** | **−92 [−133,−53]** | **FAIL** | Chat paste; not in CSV until backfill. α shock + empty buffer |
| **381ᵇ** | `train_steps` 800→**1200** | rewind 380 / root_q | **400 vs 380** | **+18 [−21,+57]** | **INCONCLUSIVE** | Chat paste. Mild vl cool; grads hotter (~16). Reverted steps |
| **381ᶜ** | **`value_coef` 1.0→1.5** | rewind 380 / root_q / steps 800 | 400 vs 380 | *pending* | **running** | Card 2A; loss = π + 1.5·v |

Older reverted trials (see changelog): Phase 2A replay/LR (161‡), scale-up 120→122, etc.

---

## Gates (Elo CI protocol)

PASS = `elo_lower > 0`. FAIL = `elo_upper < 0`. Else INCONCLUSIVE.

| A | B | Games | W–D–L | Elo | 95% CI | Verdict | Source |
|---|---|-------|-------|-----|--------|---------|--------|
| 360 | 340 | 256 | 123–43–90 | +45.04 | [+6.36, +84.87] | **PASS** | `metrics_gates.csv` |
| 380 | 360 | 256 | 132–41–83 | +67.33 | [+28.42, +107.99] | **PASS** | `metrics_gates.csv` |
| 400 | 380 | 256 | 74–42–140 | **−91.64** | [−133.05, −52.64] | **FAIL** | chat (Q+Z α=0.5) |
| 400 | 380 | 256 | 114–41–101 | **+17.66** | [−21.33, +57.10] | **INCONCLUSIVE** | chat (train_steps 1200) |

Full machine log (may lag chat pastes): `results/metrics_gates.csv`.

---

## Catalog still untried (from quality-alternatives @380)

| Rank | Lever | Status |
|------|--------|--------|
| 1 | Soft Q+Z α=0.5 | **Tried → FAIL** (retry only if α≥0.75–0.9 + warm buffer) |
| 2 | `value_coef`↑ | **In progress** @1.5 |
| 3 | Playout-cap (KataGo) | Not tried |
| 4 | `train_steps`↑ | **Tried → weak / INCONCLUSIVE** |
| 5 | Board augment | Not tried |
| 6 | Replay 200k→150k | Standby |
| 7 | Surprise / policy-KL upsample | Not tried |

---

## Ops reminders

- After FAIL/weak rewinds: `cp results/ckpt_iter_0380.pt results/latest.pt`
- `q_z` shards skip when `value_target=root_q`; **root_q** 381–400 shards from a failed block **do** warm — archive if you want a clean buffer
- Do not promote on INCONCLUSIVE alone

Last updated: 2026-07-26.
