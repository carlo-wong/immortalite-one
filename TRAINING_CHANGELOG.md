# Training recipe changelog

Compatible with Immortalite Zero checkpoints / shards (`ENCODING_VERSION = 2`). New checkpoints and shards stamp `value_target`; when resuming a legacy artifact that lacks that field, pass `--value-target` explicitly (Colab/Lightning already pass `root_q`).

**2026-07-25 cutover:** Immortalite One was renamed to Immortalite Zero (hybrid C++/Python). The pure-Python tree is archived as [`immortalite-zero-python`](https://github.com/carlo-wong/immortalite-zero-python). Colab Drive (`immortalite_zero_checkpoints`) and Lightning sibling `results/` / `syzygy345/` paths are unchanged.

Resume from the listed **start iter** with the `TRAIN` settings below (`colab/train.ipynb` cell 6 or `lightning-ai/run_train.py`). Training-parameter changes were aligned to **every 20 iterations** so each gate compares against a checkpoint trained on the same recipe.

Gates run every 20 iters vs the checkpoint **20 iters ago**. Edit only the `TRAIN` dict when moving to a new row.

| Start iter | Games | Train steps | Concurrency | Workers | Replay | Gate | LR | Notes |
|------------|-------|-------------|-------------|---------|--------|------|-----|-------|
| **0** | 64 | 400 | 64 | 1 | 50k | 64 games, winrate | cosine 6e-4→1e-4 | draw 1/3, 100 sims, resign off |
| **20** | 64 | 400 | 64 | 1 | 50k | 64 | (cosine) | same recipe |
| **40** | 64 | 400 | 64 | 1 | 50k | 64 | (cosine) | same recipe |
| **60** | 64 | 400 | **128** | 1 | **200k** | 64 | (cosine) | MCTS batch throughput |
| **61**† | **128** | **800** | 128 | 1 | 200k | 64 | (cosine) | scale games + steps with concurrency |
| **80** | 128 | 800 | 128 | 1 | 200k | 64 | **~6e-4 flat** | resume from `ckpt_iter_0080` |
| **100** | 128 | 800 | 128 | 1 | 200k | 64 | **2.5e-4 flat** | consolidate after hot LR |
| **120** | 256 | 1600 | 256 | 1 | 200k | 512 SPRT | 2.5e-4 flat | scale-up trial; reverted at 122 |
| **122** | **128** | **800** | **128** | **1** | 200k | **128 SPRT** | 2.5e-4 flat | faster s/game; gate cap matches batch |
| **161**‡ | **128** | **800** | **128** | **1** | **120k** | **128 SPRT** | **5e-4→2e-4** (161–196) | Phase 2A — **reverted** (regressed ~69 Elo vs 160); shards/checkpoints 161–180 removed |
| **161**§ | **128** | **800** | **128** | **1** | **200k** | **256 SPRT** | **2.5e-4 flat** | rewind to `ckpt_iter_0160`; 200 sims only; resign off — **reverted** (one-hot policy targets from c_scale=1.0 bug + worst-child root-Q bug; iters 161–180 discarded) |
| **161**¶ | **128** | **800** | **128** | **2** | **200k** | **128 games** | **2.5e-4 flat** | claim_draw off in search — **reverted** (17× repetition draws corrupted value targets; gate 180 vs 160 −112 Elo FAIL; iters 161–180 discarded) |
| **161** | **128** | **800** | **128** | **2** | **200k** | **128 games** (Elo CI) | **2.5e-4 flat** | **100 sims**; resign off; **claim_draw on** in search; **value_target=root_q** (per-ply MCTS Q); Gumbel c_scale 0.1 + root-Q fixes; encoding vectorized |
| **241** | **128** | **800** | **128** | **2**/4 | **200k** | **128 games** (Elo CI) | **2.5e-4 flat** | same as 161 + **move_temperature=4.0** for first **10** plies (sampling only); log `metrics_first_moves.csv`; **100 sims** |
| **261** | **128** | **800** | **128** | **2**/4 | **200k** | **128 games** (Elo CI), **gate_sims=100** | **2.5e-4 flat** | same as 241 + **self-play sims 150** (gate stays 100); keep T=4 / 10 plies |
| **321** | **128** | **800** | **128** | **2**/4 | **200k** | **128 games** (Elo CI), **gate_sims=100** | **2.0e-4 flat** | same as 261; LR step-down 2.5e-4 → 2.0e-4 |
| **341** | **128** | **800** | **128** | **2**/4 | **200k** | **128 / 256 games** (Elo CI), **gate_sims=100** | **1.5e-4 flat** | same as 321; LR 2.0e-4 → 1.5e-4; gate **360 vs 340** PASS @256 (+45 Elo, LB +6.4) |
| **361** | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | same as 341; LR → 1.0e-4; **no ply cap**; gate **380 vs 360** PASS (+67 Elo) |
| **381** | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | **`value_target=q_z` @0.5** — **reverted** (gate 400 vs 380 FAIL −92 Elo; cold buffer + α shock) |
| **381**¶ | **128** | **1200** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | rewind 380; `root_q`; **train_steps→1200** — **weak** (gate 400 vs 380 INCONCLUSIVE +18 Elo; grads hotter) |
| **381**‖ | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | rewind 380; `root_q`; steps **800**; **`value_coef=1.5`**; gate **400 vs 380** PASS (+67 Elo); **401–420** INC vs 400 — not kept |
| **401** | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | rewind 400; `root_q`; **`value_coef` 1.5→1.0** — INC ~0.51 vs 400; not kept |
| **401**¶ | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | rewind 400; **`policy_surprise_data_weight=0.5`** — INC near FAIL (−38 Elo); not kept |
| **401**‖ | **128** | **800** | **128** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | rewind 400; surprise **off**; **sims 150→200** (games/steps held); next gate **420 vs 400** |
| **481** | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | games 128→160 (steps held); gate **500 vs 480** PASS (+63 Elo) |
| **501** | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **1.0e-4 flat** | **`c_puct` 1.5→1.25** (gates keep 1.5); gate **520 vs 500** soft PASS (+41 Elo) |
| **521** | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **7.5e-5 flat** | one TRAIN knob: **LR / lr_min 1.0e-4→7.5e-5**; hold games/sims/c_puct/buffer; gate **540 vs 520** hard PASS (+84 Elo) |
| **541**¶ | **160** | **800** | **160** | **2**/4 | **150k** | **256 games** (Elo CI), **gate_sims=100** | **7.5e-5 flat** | buffer/window 200k→150k — **discarded** (White first-move collapse; c2c4→~0.61 at tip 560; iters 541–580 removed from metrics) |
| **561**¶ | **160** | **800** | **160** | **2**/4 | **150k** | **256 games** (Elo CI), **gate_sims=100** | **7.5e-5 flat** | `dirichlet_epsilon` 0.25→0.30 — **discarded** (diversity worsened; archived) |
| **561**§ | **160** | **800** | **160** | **2**/4 | **150k** | **256 games** (Elo CI), **gate_sims=100** | **7.5e-5 flat** | `move_temperature` 4→5 — **discarded** (diversity still CRITICAL; lag tip erosion vs 520) |
| **561**‖ | **160** | **800** | **160** | **2**/4 | **150k** | **256 games** (Elo CI), **gate_sims=100** | **7.5e-5 flat** | `random_opening_plies=1` — **discarded** (startpos sample starvation; gate 580 vs 560 −20 Elo INC) |
| **541** | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **5e-5 flat** | rewind tip **540**; restore buffer **200k** / `random_opening_plies=0`; one TRAIN knob: **LR / lr_min 7.5e-5→5e-5**; next gate **560 vs 540** |
| **561**¶¶ | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **5e-5 flat** | `dirichlet_alpha` **0.30→0.15** (ε held 0.25) — **discarded** (diversity CRITICAL: H≈1.64 / c2c4≈0.59 / 20/20 top1; gate **580 vs 560** +12 Elo INC; metrics 561–580 cleared) |
| **561** | **160** | **800** | **160** | **2**/4 | **200k** | **256 games** (Elo CI), **gate_sims=100** | **5e-5 flat** | resume tip **560**; restore `dirichlet_alpha` **0.30** (**hygiene**, not a new knob); HOLD LR **5e-5** / sims **200** / buffer **200k**; washout canary then recovery block; next gate **580 vs 560** |

**Current row:** resume tip **560** (`latest.pt` / `ckpt_iter_0560.pt`), `value_target=root_q`, **`sims=200`**, games/steps **160/800**, **`c_puct=1.25`**, buffer/window **200k**, **LR 5e-5 flat**, **`dirichlet_alpha=0.30`** (restore after discarded alpha=0.15 tip), **`dirichlet_epsilon=0.25`**, **`move_temperature=4`** for 10 plies, **`random_opening_plies=0`**, **`value_coef=1.0`**, **`policy_surprise_data_weight=0`**. Do **not** use `--reset-optimizer`. **No new TRAIN knob** — alpha restore is discard hygiene. Canary through 565 (abort if mean c2c4 ≥ 0.60 / H ≤ 1.70 sustained); manual gate after the block: **580 vs 560**.

Resume keeps **checkpoint net architecture** (8×96, 51 value bins). Fresh net only with a new `--checkpoint-dir`.

---

## Major changes by start iter

### Iter 0 — baseline recipe

- Flat `TRAIN` dict in Colab; resume-on-by-default from `latest.pt`.
- **64 self-play games** per iteration, **400 train steps**, **128 batch** (~6× sample reuse).
- **100 MCTS sims** per move (training and gates); no sim ramp.
- **Draw penalty 1/3** in self-play (football 3-1-0); gates use normal WDL (draw contempt 0).
- **Resign off** during self-play and gates.
- **Syzygy** adjudication in self-play and gate matches.
- **Cosine LR** from 6e-4 down toward 1e-4 over the schedule horizon.
- **50k replay** buffer/window; grows to cap as shards accumulate.
- Auto-gate every 20 iters vs checkpoint 20 iters ago; **64 games**, winrate thresholds (~0.55 / 0.45).

### Iter 20 / 40 — hold steady

- No `TRAIN` dict changes; LR continues on the cosine schedule.
- Lets each 20-iter gate block compare nets trained on identical data scale.

### Iter 60 — throughput (label recipe unchanged)

- **Concurrency 128** so MCTS batches full GPU width while still playing 64 games/iter.
- **Replay buffer/window 200k** — more history without changing how positions are labeled.
- MCTS batching optimizations in the engine (faster eval throughput).
- Still 64 games / 400 steps until the scale-up row below.

### Iter 61 — data scale-up

- **128 games** and **800 train steps** scaled together with concurrency 128.
- Keeps ~6× sample reuse (more games → proportionally more train steps).
- Replay stays at 200k (~12 iters of history at 128 games/iter before later changes).

### Iter 80 — LR reset from anchor checkpoint

- Resume from **`ckpt_iter_0080`** with LR raised to a **flat ~6e-4** (end of previous cosine was too cold).
- Same 128 / 800 / 200k otherwise; grad clip 10.
- Intended as a consolidation block before the next LR drop.

### Iter 100 — cooler flat LR

- **LR and lr_min both 2.5e-4** — effective rate stays constant (no cosine decay).
- Same games, steps, replay, and 64-game winrate gates.
- Policy was still improving but loss/noise suggested the hotter 6e-4 block had run its course.

### Iter 120 — 256-game scale-up (reverted at 122)

- **256 games / 1600 train steps** — trial; ~17% slower per game vs 128 on T4 (see iter 121 metrics).
- **`selfplay_workers: 1`**, **`concurrency: 256`**, SPRT cap 512.

### Iter 122 — back to 128 + smaller SPRT cap

- **128 games / 800 train steps**, **concurrency 128**, **`selfplay_workers: 1`** — best measured s/game on single GPU.
- Gate cap **128**; LR 2.5e-4 flat; replay 200k (~12 iters at 128 games); draw 1/3; resign off; gate sims 100.

### Iter 161 — Phase 2A (reverted)

- Bundled LR warm restart, optimizer reset, and 120k replay — gate 180 vs 160 **−69 Elo**; training metrics improved but strength collapsed (entropy collapse). Run discarded; resume from **`ckpt_iter_0160`**.

### Iter 161 — sims 200 experiment (reverted)

- Rewind to **`ckpt_iter_0160`**; delete shards/checkpoints/metrics for iters 161–180.
- **Only change vs 141–160 recipe:** self-play and gate **200 MCTS sims** (was 100).
- **256-game SPRT cap** (was 128) for tighter gate estimates.
- **200k replay**, **2.5e-4 flat LR**, resign off, optimizer state preserved.

### Iter 161 — bug-fix restart with claim_draw=False (reverted)

- **Sims-200 run discarded:** two bugs introduced by the Jul 6 fix commit corrupted training targets. (1) Gumbel improved-policy collapsed to one-hot because `gumbel_c_scale` was set to 1.0 instead of 0.1 (argmax sigma dominates). (2) `searched_root_q` returned the worst child's value instead of the visit-weighted mean, corrupting truncation value labels. Both bugs are now fixed with regression tests.
- Rewound to **`ckpt_iter_0160`** with **100 sims**, workers **2**, vectorized encoding, Gumbel c_scale 0.1 + root-Q fixes.
- **`claim_draw=False` in MCTS search** (intended as a speedup) made search blind to threefold/fifty-move draws while the game loop still adjudicated them → 17× more repetition draws, corrupted value targets (−1/3 on winning positions), value_std collapse, gate 180 vs 160 **−112 Elo FAIL**.
- **Third discard of iters 161–180.** Resume from **`ckpt_iter_0160`**.

### Iter 161 — claim_draw restored (superseded by root_q labels)

- Same recipe as the bug-fix restart **except** search keeps **`claim_draw=True`** (Config default; no train.py override).
- Gate logging uses **Elo 95% CI verdict** (PASS if lower bound > 0, FAIL if upper bound < 0, else INCONCLUSIVE) — no H₀/H₁ LLR columns in `metrics_gates.csv`.
- **Resignation off**; workers 2; 200k replay; 2.5e-4 flat LR; 100 sims.

### Iter 161 — value_target=root_q

- **One change** vs claim_draw-restored recipe: self-play value labels use per-ply **`searched_root_q`** (`--value-target root_q`) instead of terminal game outcome (±1 / −draw_penalty).
- Policy targets unchanged (Gumbel improved policy). Search still uses `draw_contempt = draw_penalty` and `claim_draw=True`. Gates unchanged (WDL outcomes).
- Wired in `colab/train.ipynb` and `lightning-ai/run_train.py` TRAIN dicts.
- Abort watch: `value_std` should rise or stay high (not collapse toward 0); threefold count must stay ~2/128; do not trust train loss alone.

### Iter 241 — move temperature

- Same recipe as root_q row, plus early-ply **move sampling temperature T=4 for first 10 plies** (`--move-temperature 4 --move-temperature-plies 10`).
- Sampling only during exploration plies; stored policy targets stay untempered. Gate exploration / gate temperature unchanged.
- Each self-play iter appends `metrics_first_moves.csv`:
  `iter,n,entropy,top1..top5_uci/share,main_share,flank_share`
  (`main`={e4,d4,Nf3,c4}; `flank`=wing/fianchetto set). An older CSV header is rotated to
  `metrics_first_moves_legacy.csv` on first write after upgrade.

### Iter 261 — self-play sims 150

- Same as 241 (T=4 / 10 plies, root_q, claim_draw on) except **self-play `--sims 150`**.
- **`gate_sims` stays 100** for lag-20 Elo-CI gates (comparable protocol).
- Rationale: first-move diversity recovered under T=4; gate Elo near 128-game noise floor — raise search depth for stronger training targets without jumping to 200.
- Lightning: `lightning-ai/run_train.py` or combined `lightning-ai/run_train_and_gate.py` (train to next ×20, then gate vs −20).

### Iter 321 — LR 2.0e-4 flat

- Same as 261 except **LR / lr_min = 2.0e-4** (partial clip-cure step). Native One self-play throughput regime.

### Iter 341 — LR 1.5e-4 flat

- Same as 321 except **LR / lr_min = 1.5e-4** (finish the prior clip-cure step). Hold sims/T/buffer/games/steps.
- Gate **360 vs 340**: INCONCLUSIVE at 128 games; **PASS at 256** (+45.04 Elo, LB +6.36). Grad clip saturation did not cool (mean grad_norm ~12.35).

### Iter 361 — LR 1.0e-4 flat

- Same as 341 except **LR / lr_min = 1.0e-4** (last LR rung before floor; clip abort after 1.5e-4 failed to restore headroom).
- Hold sims 150 / gate_sims 100 / T=4 / buffer 200k / games 128 / steps 800 / `root_q`.
- **Remove training ply cap:** `max_game_moves` **200 → 10_000** (matches strength-gate policy). Games resolve by checkmate / Syzygy / fifty-move / threefold instead of `max_moves` truncation labels.
- **Gate concurrency 128 → 256** (native dual-net path; `gate_workers` still ignored).
- Gate **380 vs 360**: PASS (+67 Elo).

### Iter 381 — value_target=q_z (reverted)

- Soft Q+Z (`α=0.5`) + hard skip of stamped `root_q` shards (cold buffer).
- Gate **400 vs 380**: **FAIL** (−91.6 Elo, CI [−133, −53], 74–42–140). Train curves also aborted (vl late−early, top1 collapse, sign_acc drip). Not a label-path bug — α shock + flush confound.

### Iter 381 — rewind + train_steps 1200 (weak)

- Restore **`ckpt_iter_0380.pt`**; `root_q`; **train_steps 800→1200**.
- Gate **400 vs 380**: **INCONCLUSIVE** (+18 Elo, CI [−21, +57]). Mild vl cool; **grad_norm worse** (~16). Not kept.

### Iter 381 — rewind + value_coef 1.5

- Restore **`ckpt_iter_0380.pt`** as `latest.pt`. **`value_target=root_q`**, **`train_steps=800`** (reverted), sims **150**.
- **One TRAIN knob:** **`value_coef` 1.0 → 1.5** (`loss = policy_loss + 1.5 * value_loss`). Hold LR 1e-4 / grad_clip 10 / T=4 / buffer 200k / games 128.
- Gate **400 vs 380**: **PASS** (+67 Elo). Kept **400**.
- Block **401–420** under coef 1.5: gate **420 vs 400** **INCONCLUSIVE** twice (~+11 / +33 Elo). Flat policy, soft value drift, grad_norm ~25. Not kept; metrics/shards for 401–420 discarded.

### Iter 401 — rewind + value_coef 1.0

- Restore **`ckpt_iter_0400.pt`** as `latest.pt`. Hold `root_q` / steps 800 / sims 150 / LR 1e-4.
- **One TRAIN knob:** **`value_coef` 1.5 → 1.0** (equal policy/value weight).
- Gate **420 vs 400**: **INCONCLUSIVE** (~0.51). Not kept.

### Iter 401 — rewind + policy surprise 0.5 (weak)

- Restore **`ckpt_iter_0400.pt`** as `latest.pt`. Hold `value_coef=1.0` / `root_q` / steps 800 / sims 150.
- **One TRAIN knob:** **`policy_surprise_data_weight` 0 → 0.5** (KataGo write-time: half uniform, half ∝ `KL(π_target ‖ π_prior)`; replicate at ingest).
- Train curves looked healthier (pl/vl↓, top1↑; grad_norm ~16–17).
- Gate **420 vs 400**: **INCONCLUSIVE** near FAIL — **94–40–122**, score **0.445**, **−38 Elo**, CI **[−78, +0.8]**. Worse than prior 420-slot tries. Not kept.

### Iter 401 — rewind + sims 200

- Restore **`ckpt_iter_0400.pt`** as `latest.pt`. Surprise **off**. Hold `value_coef=1.0` / `root_q` / games 128 / steps 800 / LR 1e-4 / buffer **200k**.
- **One TRAIN knob:** **self-play `sims` 150→200** (gate stays **100**).
- Gate **420 vs 400**: PASS. Kept.

### Iter 481 — games 160

- **One TRAIN knob:** self-play **games 128→160** (concurrency 160; steps held at 800).
- Gate **500 vs 480**: PASS (+63 Elo).

### Iter 501 — c_puct 1.25

- **One TRAIN knob:** self-play **`c_puct` 1.5→1.25** (gates still force Config default 1.5).
- Gate **520 vs 500**: soft PASS (+41 Elo, CI LB +3.6).

### Iter 521 — LR 7.5e-5 flat

- Resume from tip **520**. Hold games 160 / sims 200 / steps 800 / buffer 200k / `c_puct=1.25` / `root_q`.
- **One TRAIN knob:** **LR / lr_min 1.0e-4 → 7.5e-5** (flat).
- Gate **540 vs 520**: hard PASS (+84 Elo, CI LB +47). Kept.

### Iter 541 — replay 150k (discarded)

- Resume from tip **540**. Hold games 160 / sims 200 / steps 800 / LR 7.5e-5 / `c_puct=1.25` / `root_q`.
- **One TRAIN knob:** **replay buffer + window 200k → 150k** (fresher root_q labels).
- Gate **560 vs 540**: INCONCLUSIVE (+27 Elo). Lag **560 vs 520**: hard PASS (+96 Elo).
- Opening diversity collapsed during the block (White first-move H ≈ 1.62, c2c4 ≈ 0.61 at tip 560 vs healthy H ≈ 2.29 / c2c4 ≈ 0.04 in 521–540). Discarded; rewind tip **540**, restore buffer **200k**. Metrics rows 541–580 cleared.

### Iter 561 — dirichlet_epsilon 0.30 (discarded)

- Ran from tip **560** under the 150k-buffer recipe. Diversity worsened (H ≈ 1.36, c2c4 ≈ 0.68). Archived with the 541+ scrap.

### Iter 561 — move temperature 5 (discarded)

- Ran from tip **560**. Diversity still CRITICAL; lag 580 vs 520 +73 vs 560's +96 tip erosion. Archived with the 541+ scrap.

### Iter 561 — random opening prefix k=1 (discarded)

- Ran from tip **560**. Prefix wrote no training samples → White startpos starvation. Gate **580 vs 560** −20 Elo INC. Archived with the 541+ scrap.

### Iter 541 — rewind tip 540 + LR 5e-5

- Restore **`ckpt_iter_0540.pt`** as `latest.pt`. Discard buffer-150k / ε / T=5 / opening-prefix branches (metrics cleared past 540).
- Hold games 160 / sims 200 / steps 800 / buffer **200k** / `c_puct=1.25` / T=4 / plies=10 / ε=0.25 / `random_opening_plies=0` / `root_q`.
- **One TRAIN knob:** **LR / lr_min 7.5e-5 → 5e-5** (flat).
- Watch White first-move diversity (abort if H ≲ 2.0 or c2c4 ≳ 0.40 sustained). Manual gate after the block: **560 vs 540**; lag hygiene **560 vs 520** if INC.

### Iter 561 — dirichlet_alpha 0.15 (discarded)

- Resume from tip **560**. Hold LR 5e-5 / sims 200 / buffer 200k / ε=0.25 / T=4.
- **One TRAIN knob:** `dirichlet_alpha` **0.30→0.15**.
- Gate **580 vs 560**: INCONCLUSIVE (+12 Elo). Diversity CRITICAL (block H≈1.64 / c2c4≈0.59 / top1=c2c4 20/20).
- Discarded; rewind tip **560**; restore α **0.30**. Metrics rows 561–580 cleared.

### Iter 561 — alpha restore washout / recovery (current)

- Restore **`ckpt_iter_0560.pt`** as `latest.pt`.
- Hold games 160 / sims 200 / steps 800 / buffer **200k** / LR **5e-5** / `c_puct=1.25` / T=4 / plies=10 / ε=0.25 / `random_opening_plies=0` / `root_q`.
- **Hygiene (not an experiment):** `dirichlet_alpha` **0.15→0.30**.
- **No new TRAIN knob.** Canary through 565; if green, full recovery block to 580.
- Manual gate after the block: **580 vs 560**; if INC, champ context **580 vs 540** and optional lag **580 vs 520**.

Last updated: 2026-07-31.
