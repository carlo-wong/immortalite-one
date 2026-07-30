# Training Immortalite Zero on Lightning AI

Self-play on a Lightning AI Studio GPU via `run_train.py`, `run_gate.py`, and `run_train_and_gate.py`. Sibling `results/` and `syzygy345/` folders are reused across sessions.

---

## Workspace layout

```
parent/
├── immortalite-zero/       # git clone (canonical hybrid product)
│   └── lightning-ai/
│       ├── run_train.py
│       ├── run_gate.py
│       ├── run_train_and_gate.py
│       └── paths.py
├── immortalite-zero-python/  # optional archived pure-Python oracle
├── results/                # checkpoints/shards; reused directly
│   ├── latest.pt
│   ├── metrics.csv
│   ├── metrics_gates.csv
│   └── ckpt_iter_XXXX.pt
└── syzygy345/              # tablebases; reused directly
```

Build Syzygy locally once:

```bash
python scripts/download_syzygy345.py --out syzygy345
```

---

## Before you start

- Lightning AI account with GPU studio.
- Keep `results/` and `syzygy345/` as **siblings** of the repo; do not rename or copy them.
- Push engine changes to GitHub; `git pull` before `run_train.py`.
- **Native extension required:** build `engine._native` once per studio with `pip install -e . --no-deps` (see Step 1). Do **not** `pip install -r requirements.txt` on Lightning — that can replace the studio CUDA torch.

---

## Step 1 — Studio setup (copy-paste)

A compiled C++ extension (`engine._native`) is required. Running `python lightning-ai/run_train*.py` alone is not enough until this succeeds. Re-run after a fresh studio, env wipe, or C++ changes after `git pull`.

```bash
cd /teamspace/studios/this_studio/immortalite-zero
sudo apt-get install -y build-essential ninja-build python3-dev
pip install -q "cmake>=3.26,<4.0" ninja pybind11 scikit-build-core
pip install -q python-chess numpy tqdm
pip install -q -e . --no-deps
python -c "import torch; from engine import _native; print(torch.__version__, torch.version.cuda, _native.version())"
```

`--no-deps` compiles `cpp/` into `engine._native` without replacing Lightning’s preinstalled CUDA torch. Do **not** `pip install -r requirements.txt` on the studio.

If import fails with a “circular import” message for `_native`, the extension is missing — re-run the block above (not a real circular import).

## Step 2 — Train

Edit `TRAIN` in `lightning-ai/run_train.py` if needed, then:

```bash
cd immortalite-zero
nohup python lightning-ai/run_train.py > ../results/train.log 2>&1 &
tail -f ../results/train.log
```

Or train to the next ×20 milestone **and** gate automatically:

```bash
nohup python lightning-ai/run_train_and_gate.py > ../results/train_and_gate.log 2>&1 &
tail -f ../results/train_and_gate.log
```

Writes `latest.pt`, `metrics.csv`, shards every iteration to `../results/`. Training survives browser close (~4h studio limit still applies).

After the job finishes (success or failure), scripts call `Studio().stop()` when `SLEEP_STUDIO=True` in `lightning-ai/studio_sleep.py` (default on), after a **5-minute delay** (`SLEEP_STUDIO_DELAY_SEC`). That ends GPU billing instead of waiting for idle auto-sleep. Set `SLEEP_STUDIO = False` if you are debugging interactively, or `SLEEP_STUDIO_DELAY_SEC = 0` to stop immediately. Requires `pip install lightning-sdk` once per studio (usually preinstalled).

### Current `TRAIN` defaults

Same recipe as Colab except `selfplay_workers=4` / `gate_workers=4` (Lightning T4 has 4 vCPUs; Colab is 2). Current row: **rewind to 400**, retrain **401–420** with **sims 150→200** (games/steps stay 128/800, buffer **200k**, surprise **off**). Next gate **420 vs 400**. See `colab/README.md` and `TRAINING_CHANGELOG.md`.

**Rewind ops:** `cp ../results/ckpt_iter_0400.pt ../results/latest.pt` before train. Keep older shards/ckpts (stamp-skipped mismatches); new iters overwrite the same filenames.

| Key | Value |
|-----|-------|
| `sims` | **200** (self-play; was 150) |
| `games` | 128 |
| `train_steps` | **800** |
| `concurrency` | 128 |
| `selfplay_workers` / `gate_workers` | 4 / 4 |
| `value_target` | `root_q` |
| `value_coef` | **1.0** (loss = π + v) |
| `policy_surprise_data_weight` | **0.0** (0.5 failed; off) |
| `move_temperature` / `move_temperature_plies` | **4.0** / **10** (sampling only) |
| `random_opening_plies` | **1** (uniform legal first ply; SP only) |
| `resign` | off |
| `replay_buffer` / `replay_window` | **150k** |
| `gate_games` / `gate_sims` | **256** / **100** (gates stay 100) |
| `gate_concurrency` | **256** (native gate parallelization) |
| `gate_exploration_moves` / `gate_openings` | 0 / masters (128×2 colors) |
| `lr` / `lr_min` | **7.5e-5** flat |
| Training span | auto-stops at iters 360, 380, … (multiples of 20) |
| `RESET_OPTIMIZER` | `False` |

### Gating

In-loop gating is off (`--gate-every 0`). Manual gate:

```bash
# edit CHECKPOINT_A / CHECKPOINT_B in run_gate.py first (defaults: 380 vs 360)
python lightning-ai/run_gate.py
```

Or use Colab cell 6. Gates use the native actor + dual-evaluator path when `engine._native` is built.

**Why `gate_workers` is ignored on the native path:** a strength gate needs **two** nets on GPU. The fast path keeps both evaluators in one process and packs games with `gate_concurrency`. Multi-worker split would either copy both nets into every worker (VRAM) or require dual-net central inference (not built). Self-play multi-worker works because it is a **single** net + central inference. Speed gates with `gate_concurrency`, not more `gate_workers`.

## Step 3 — Sessions and resuming

When a studio ends, **download the updated `results/` folder**.

| Goal | Action |
|------|--------|
| Resume | Re-upload `results/` with `latest.pt`, re-run train |
| Fresh start | Empty `results/` (no `latest.pt`) |

Rotate old `metrics_gates.csv` if upgrading from pre-SPRT recipes.

## Step 5 — Use locally

```bash
python -m engine.inspect_encoding --checkpoint-dir results
python -m uci.uci_engine results/latest.pt
```

Legacy checkpoints that lack `value_target` metadata: resume with an explicit `--value-target root_q` (Lightning’s `run_train.py` already passes it). Mismatched stamped shards (e.g. leftover `q_z`) are skipped on warm-up.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `engine._native` import error | Re-run Step 1 (`pip install -e . --no-deps` + pinned cmake) |
| Syzygy incomplete | All 145 `.rtbw` in `syzygy345/` sibling folder |
| `results/` not found | Sibling of repo, not inside it |
| Slow self-play | Keep `concurrency` = `games`; `selfplay_workers=4` on Lightning (4 vCPUs). Colab bench only tested 2 |
| Slow gates / workers ignored | Expected on native path — raise `gate_concurrency` |
| OOM | Lower `games` and `concurrency` together |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
