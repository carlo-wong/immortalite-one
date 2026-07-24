# Training Immortalite One on Google Colab

Step-by-step guide for the free Colab GPU workflow. Open `colab/train.ipynb` and run cells in order.

> **What you're doing:** clone the repo in Colab, build the native C++ extension, run self-play training on a free GPU, save checkpoints to Google Drive every iteration, and download `latest.pt` for local analysis.

---

## Before you start

- A **Google account** (Colab + Drive).
- Code on GitHub: `github.com/carlo-wong/immortalite-one` — `git push` after local changes so Colab can `git pull`.
- **Drive first:** cell 2 mounts Drive so you can verify access before the slow install.
- **Native extension:** cell 3 installs pinned `cmake>=3.26,<4.0` and runs `pip install -e . --no-deps` so Colab’s CUDA torch is not replaced. Training will not run without `engine._native`. See [docs/BUILD.md](../docs/BUILD.md) for wheels / troubleshooting.
- **Existing training data:** during migration, the notebook reads and writes `MyDrive/immortalite_zero_checkpoints` directly.
- **Syzygy:** cell 4 copies `syzygy345/` from Drive if present, or downloads once into your checkpoint folder.

---

## Step 1 — Open the notebook

**https://colab.research.google.com/github/carlo-wong/immortalite-one/blob/main/colab/train.ipynb**

Or: [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** → `carlo-wong/immortalite-one`.

## Step 2 — Enable GPU

**Runtime → Change runtime type → Hardware accelerator → GPU → Save.**

## Step 3 — Run cells

| Cell | What it does |
|------|--------------|
| 1 | Clone repo + `git pull` |
| 2 | Mount Drive → existing `MyDrive/immortalite_zero_checkpoints` (verify, then AFK) |
| 3 | Pin cmake + **build native extension** (`pip install -e . --no-deps`) |
| 4 | Syzygy tablebases (Drive cache or download) |
| 5 | **Train** — always `--device cuda --gpu`; edit `TRAIN` dict only; auto-stops at iters 160, 180, … |
| 6 | Optional **manual gate** (SPRT, 128 games / 100 sims) |
| 7 | Plot `metrics.csv` + gate results |

## Step 4 — Current `TRAIN` defaults (cell 5)

Current recipe: iter **361+** — same as `lightning-ai/run_train.py` except workers **2** (Colab) vs **4** (Lightning). See `TRAINING_CHANGELOG.md`.

| Key | Value | Notes |
|-----|-------|-------|
| `sims` | **150** | flat MCTS sims/move |
| `move_temperature` / `move_temperature_plies` | **4.0** / **10** | early-ply sampling only; targets untempered |
| `value_target` | **root_q** | per-ply MCTS root Q labels |
| `games` | 128 | full GPU batch width (`concurrency` matches) |
| `train_steps` | 800 | ~6× sample reuse at 128 games |
| `concurrency` | 128 | batched MCTS eval width (one GPU owner) |
| `selfplay_workers` / `gate_workers` | **2** / **2** | self-play: central inference; gates: see note below |
| `replay_buffer` / `replay_window` | **200k** | ~12 iters at 128 games |
| `draw_penalty` | 1/3 | football 3-1-0 shaping |
| `resign` | False | off |
| `lr` / `lr_min` | **1.0e-4** | flat (row 361+) |
| `gate_games` / `gate_sims` | **256 / 100** | manual gate cell 6 only |
| `gate_concurrency` | **256** | gate parallelization knob (native path) |
| `gate_exploration_moves` | **0** | after masters book (no temperature) |
| `gate_openings` | **masters** | 128 prefix-free lines × both colors (=256) |
| `save_every` | 10 | numbered snapshots |
| `resume` | True | loads `latest.pt` automatically |

**Why gates ignore `gate_workers`:** Immortalite One’s fast gate path loads **two** nets (A and B) on **one** CUDA process and runs many games via `gate_concurrency`. Splitting across workers would either duplicate both nets per worker (VRAM blow-up) or need a dual-net central-inference server (not implemented). Self-play can use multiple workers because it has a **single** net + central inference. To speed gates, raise `gate_concurrency` (now 256), not `gate_workers`.

Training auto-stops after completing an iter that is a multiple of **20** (240, 260, …). Re-run cell 5 for the next span. No in-loop auto-gate.

With CUDA and more than one self-play worker, central inference is enabled by default: the training process owns CUDA and workers do CPU/native search. Legal-only transfer with reusable pinned buffers, native game actors, and CUDA Graphs auto mode are selected when supported; each has a direct eager or non-central fallback.

## Step 5 — What good looks like

```
iter  40 | sims 150 | games 128 | samples 18500 | buffer 200000 | policy_loss 2.1 | value_loss 0.4 | lr 1.000e-04 | 420.0s
```

- **policy_loss** should trend down over many iterations (not every single iter).
- **value_loss** should stay meaningful — games need real outcomes, not only max-move truncations.
- **Next manual gate (cell 6):** **380 vs 360** (`CHECKPOINT_A=380`, `CHECKPOINT_B=360`).
- **SPRT PASS** in a manual gate means significant improvement; **INCONCLUSIVE** is normal on short runs.
- Cell 7 plots are the clearest long-run signal.

Pure self-play on a free GPU targets club-level strength, not Stockfish.

## Step 6 — Disconnects and resuming

Checkpoints save to Drive **every iteration** (`latest.pt`, `metrics.csv`, sample shards).

| Goal | Action |
|------|--------|
| **Resume** after disconnect | Re-run cells 1→5. `resume: True` loads `latest.pt`. |
| **Fresh run** | Empty Drive checkpoint folder, re-run 1→5. |
| **Compare checkpoints** | Use cell 6 manual gate or download `ckpt_iter_XXXX.pt`. |

Numbered snapshots: `ckpt_iter_0000.pt`, `ckpt_iter_0010.pt`, … every `save_every` iters.

**metrics_gates.csv:** if you upgraded from an older recipe, delete or rotate the file — the header now includes the Fishtest-style SPRT (`llr`, `decision`, `verdict`) plus a logistic `elo` estimate with a 95% CI (`elo_lower`/`elo_upper`) and `los`.

## Step 7 — Update code from your machine

```bash
git add -A && git commit -m "your change" && git push
```

In Colab, re-run **cell 1** (`git pull`) and continue training.

## Step 8 — Use locally

1. Download `latest.pt` (and shards / metrics if needed) from Drive.
2. Verify encoding / metadata:

```bash
python -m engine.inspect_encoding --checkpoint-dir checkpoints
```

3. Optional UCI analysis (fixed sims):

```bash
python -m uci.uci_engine checkpoints/latest.pt
```

Legacy Zero checkpoints lack `value_target` metadata: resume with `--value-target root_q` (cell 5 already passes it).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `engine._native` import error | Re-run cell 3; ensure pinned cmake / ninja / build tools installed |
| CUDA torch became CPU-only | Do not `pip install -r requirements.txt` on Colab; reinstall GPU torch and use `--no-deps` |
| Drive auth failed | Re-run cell 2 |
| Training "stuck" | One iter can take several minutes at 128 games; watch `metrics.csv` |
| OOM | Lower `games` / `concurrency` together, or reduce net in checkpoint (fresh start only) |
| SPRT always INCONCLUSIVE | Normal early; need more gate games or stronger signal |
| Old gate CSV garbled | Delete `metrics_gates.csv` and let it recreate |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
