# Immortalite One

Hybrid rewrite of Immortalite Zero: **C++ board / movegen / encoding / MCTS** (pybind11) plus **Python / PyTorch** training, UCI, Colab, and Lightning.

Compatible with Immortalite Zero `.pt` checkpoints, `samples_iter_*.npz` shards, and `ENCODING_VERSION = 2`.

## Layout

| Path | Role |
|------|------|
| `cpp/` | Native core (board, movegen, encoding, MCTS, bindings) |
| `engine/` | Python package — `python -m engine.train` |
| `uci/` | UCI front-end (`python -m uci.uci_engine [ckpt.pt]`) |
| `colab/` | Colab notebook (Drive checkpoints + native build) |
| `lightning-ai/` | Lightning AI `run_train.py` |
| `tests/` | Encoding / MCTS parity vs Python oracle |
| `docs/BUILD.md` | Editable install, Colab, prebuilt wheels |
| `SELFPLAY_THROUGHPUT.md` | Games/concurrency/worker guidance |
| `../Chess AI` | Immortalite Zero — read-only oracle |

## Native extension (required)

```bash
# Local
pip install -r requirements.txt
pip install -e .

# Colab / Lightning (keep preinstalled CUDA torch)
pip install "cmake>=3.26,<4.0" ninja pybind11 scikit-build-core python-chess numpy tqdm
pip install -e . --no-deps
```

Details: [docs/BUILD.md](docs/BUILD.md).

- **Windows:** MSVC x64 Developer shell; CMake **&lt; 4.0** (pinned in `requirements.txt`)
- **Linux / Colab / Lightning:** `build-essential` / `g++`, then `pip install -e . --no-deps` on managed GPUs
- **Wheels:** CI can build artifacts via [`.github/workflows/wheels.yml`](.github/workflows/wheels.yml) (not on PyPI yet)

## Quick start

```bash
# Parity
python -m pytest tests/ -v

# MCTS microbench (native vs Python, CPU uniform logits)
python scripts/bench_mcts.py --sims 32 --searches 40

# UCI (Arena / CuteChess / Lichess local)
python -m uci.uci_engine path/to/latest.pt

# Short train smoke (CPU)
python -m engine.train --iterations 1 --device cpu --light \
  --games 2 --train-steps 4 --concurrency 2 --selfplay-workers 1 \
  --sims 16 --value-target outcome --gate-every 0 --quick-eval-games 0 \
  --checkpoint-dir checkpoints_smoke
```

Self-play uses native `MctsSession` + batched `evaluate_planes` when `engine._native` is available. Force Python MCTS with `IMMORTALITE_ONE_FORCE_PYTHON=1`.

### Microbench (example, Windows CPU)

Uniform logits, 40 searches × 32 sims (no torch forward in the compare):

| Backend | ms / search | Relative |
|---------|-------------|----------|
| Native C++ MCTS | ~0.6 | **~13×** vs Python |
| Python MCTS | ~7.8 | 1× |

With a small CPU `ChessNet`, forward cost dominates (~100 ms/search); on GPU self-play the native search still cuts the CPU share of wall time. Re-run after search changes.

## Cloud

- **Lightning:** `python lightning-ai/run_train.py` (sibling `../results`, `../syzygy345`) — see [lightning-ai/README.md](lightning-ai/README.md)
- **Colab:** [colab/train.ipynb](colab/train.ipynb) — `pip install -e . --no-deps`, Drive `immortalite_one_checkpoints`

Throughput knobs: [SELFPLAY_THROUGHPUT.md](SELFPLAY_THROUGHPUT.md).

## CI

GitHub Actions [`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds the native extension and runs `pytest` on **Ubuntu** and **Windows** (Python 3.11).

## Status

- Phase 1: C++ core, parity tests, UCI/analyze
- Phase 2: self-play, train, shards, Colab/Lightning entrypoints
- Phase 3: CI, wheel workflow, build/throughput docs, MCTS microbench
