# Self-play throughput notes (Immortalite One)

Goal: **fastest games/hour on one GPU** without changing search quality (sims/move, net, MCTS semantics).

Immortalite One moves board / encoding / MCTS into **C++** (`engine._native`) and batches NN evals via `evaluate_planes`. Recipe knobs below inherit Immortalite Zero measurements; **re-benchmark after major native changes**.

Local microbench (CPU, uniform logits — search only, no torch):

```bash
python scripts/bench_mcts.py --sims 32 --searches 30
```

---

## Recommended defaults (current cloud recipe)

Same as Zero’s late recipe unless a One-specific bench says otherwise:

| Parameter | Colab | Lightning | Why |
|-----------|-------|-----------|-----|
| `games` | **128** | **128** | Best measured s/game class on T4 for Zero |
| `concurrency` | **128** | **128** | Match `games` for full batch width |
| `selfplay_workers` | **2** | **4** | Match vCPU count; each worker has its own CUDA context |
| `sims` | **150** | **150** | Flat sims/move (gate often 100) |
| `train_steps` | **800** | **800** | ~6× sample reuse at batch 128 |
| `value_target` | `root_q` | `root_q` | Per-ply searched root Q |

See `lightning-ai/run_train.py` and `colab/train.ipynb` `TRAIN` dicts.

---

## What Zero taught us (still relevant)

From Immortalite Zero T4 runs (`SELFPLAY_THROUGHPUT.md` in Chess AI):

1. **64 → 128 games** helped s/game (wider GPU batches).
2. **128 → 256** hurt on T4 — CPU (MCTS/encoding/Syzygy) dominated; One’s native MCTS should shift this curve — **re-test before raising games**.
3. Prefer **s/game** (and games/hour), not only wall-clock per iter.
4. Keep `concurrency == games` when using one GPU owner per worker.

---

## One-specific notes

| Topic | Guidance |
|-------|----------|
| Native vs Python MCTS | Train path uses `play_games_batched_native`. Force Python with `IMMORTALITE_ONE_FORCE_PYTHON=1` only for debug. |
| FEN + move history | Pass UCI history into native sessions so EP / repetition / claim_draw match python-chess (facade already does this). |
| Gate / match path | Still uses Python `search_gen` in places; self-play train path is native. Expect gates slower until ported. |
| Compile cost | Colab/Lightning pay once per runtime for `pip install -e .`; use CI wheels when available ([docs/BUILD.md](docs/BUILD.md)). |
| Syzygy | Same as Zero — optional for smoke; on for production recipe. |

---

## Before changing games / concurrency / workers

- [ ] Compare **s/game** from the iter log (`selfplay X.Xs`), not just total iter time.
- [ ] Run ≥ **3 iters** after a change (compile / CUDA warmup).
- [ ] Log whether native was used (`MCTS.using_native` / absence of Python-fallback warning).
- [ ] Re-run `scripts/bench_mcts.py` if C++ search changed.
- [ ] Suspect regression if s/game rises >~5% vs prior block median.

---

## Not worth revisiting (until new bench)

- `games: 256` on T4 without a fresh One measurement  
- `selfplay_workers >` vCPU count on one GPU  
- Chasing TensorRT / ONNX while MCTS still dominates wall time  

---

Last updated: 2026-07-23 (Immortalite One Phase 3; recipe numbers from Zero T4 era — re-bench on One).
