# Self-play throughput notes (Immortalite One)

Goal: **fastest games/hour on one GPU** without changing search quality (sims/move, net, or MCTS semantics).

Immortalite One moves board, encoding, and MCTS into **C++** (`engine._native`). Cloud self-play uses one CUDA owner for multi-worker training, while workers perform CPU/native search.

## Recommended defaults (current cloud recipe)

| Parameter | Colab | Lightning | Why |
|-----------|-------|-----------|-----|
| `games` | **128** | **128** | Full production game batch |
| `concurrency` | **128** | **128** | Full batch width |
| `selfplay_workers` | **2** | **4** | Match available CPU/vCPUs; CUDA stays in the training process |
| `sims` | **150** | **150** | Flat sims/move (gate often 100) |
| `train_steps` | **800** | **800** | ~6× sample reuse at batch 128 |
| `value_target` | `root_q` | `root_q` | Per-ply searched root Q |

`engine.train` enables central inference by default when `device` starts with `cuda` and `selfplay_workers > 1`. The training process owns the sole CUDA evaluator; workers run CPU/native search and send inference requests to it. CPU, one-worker, and broker-unavailable runs use direct batching instead—central inference never silently falls back to competing per-worker CUDA contexts.

## Strength gates

Manual SPRT gates (`play_match` / Colab cell 6) use the same native `GameActorBatch` stack as self-play when the extension is built: dual `NetEvaluator` in one process (single CUDA owner), masters openings via per-actor `start_moves`, and `pending_net_ids` for A/B routing. `gate_workers>1` is ignored on this path — set `gate_concurrency` (default 128) instead. Falls back to the legacy dual-worker CUDA path only if native actors are unavailable.

## Fast paths and fallbacks

- **Legal-only transfer and reusable buffers:** when native APIs and `NetEvaluator.evaluate_legal` are available, only legal policy logits cross device-to-host and pinned buffers are reused.
- **Native actors:** `GameActorBatch` is selected automatically when available through `play_games_batched_native_actors`.
- **CUDA Graphs:** `NetEvaluator` / `CudaBatchExecutor` use `graph_mode="auto"`; unsupported or failed batch buckets run eagerly without changing search semantics.
- **Python fallback:** `IMMORTALITE_ONE_FORCE_PYTHON=1` is for debugging only, not a production throughput baseline.

See `lightning-ai/run_train.py` and `colab/train.ipynb` for the matching `TRAIN` recipes.

## Benchmarking

Use `scripts/bench_throughput.py` and `colab/benchmark_throughput.ipynb` for the controlled T4 A0 check, worker × concurrency matrix, and central-inference comparison. Keep checkpoint, recipe, and Syzygy inputs fixed when comparing lanes.

Benchmark artifacts append to JSONL (`benchmark_throughput.jsonl`); they do not read or modify `metrics.csv`. Compare **seconds/game** and games/hour after warm-up, not only total wall-clock time.

For local CPU search-only checks:

```bash
python scripts/bench_mcts.py --sims 32 --searches 30
```

## Before changing games / concurrency / workers

- [ ] Run at least three warmed iterations or benchmark repeats.
- [ ] Keep `games`, `concurrency`, sims, checkpoint, and Syzygy fixed for a throughput comparison.
- [ ] Confirm the native actor path is active and record central-inference mode.
- [ ] Re-run `scripts/bench_mcts.py` if C++ search changes.
- [ ] Treat a >~5% seconds/game increase versus the prior median as a regression to investigate.

## Not worth revisiting without a fresh benchmark

- `games: 256` on T4
- More workers than available vCPUs on one GPU
- TensorRT / ONNX while MCTS remains the bottleneck

Last updated: 2026-07-24.
