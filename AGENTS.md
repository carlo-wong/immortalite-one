# Preferred subagents — Immortalite One

Project specialists under [`.cursor/agents/`](.cursor/agents/), selected from the global library at `~/.cursor/agents-library` (plus the Immortalite Zero set).

Immortalite One is a hybrid rewrite: **C++ board/MCTS/encoding** (pybind11) + **Python/PyTorch** training, with compatibility for Immortalite Zero checkpoints (`.pt`), shards (`.npz`), UCI, and Colab/Lightning entrypoints.

Invoke with `/agent-name` or via the Task tool (`subagent_type`).

| Agent | Use when |
|-------|----------|
| `cpp-pro` | Bitboards, movegen, MCTS tree, encoding hot path, modern C++20 |
| `build-engineer` | CMake, scikit-build-core, pybind11 wheels, Colab/Lightning compile story |
| `tooling-engineer` | Native extension packaging, CLIs, developer tooling around the C++/Python boundary |
| `python-pro` | Python bridge (`engine/`), pybind wrappers, type-safe tests |
| `reinforcement-learning-engineer` | Self-play loops, policy/value targets, MCTS↔training integration |
| `ml-engineer` | Training scripts, checkpointing, progressive sims, data generation |
| `performance-engineer` | Self-play throughput, CPU vs GPU bottlenecks, sim budgets |
| `test-automator` | Parity suites vs Immortalite Zero (encoding, MCTS, terminals), CI |
| `software-architect` | Hybrid boundary decisions, compatibility contracts, module layout |
| `fastapi-developer` | Analysis API (`server/`) when ported |
| `data-scientist` | Training curves, self-play metrics, gating/eval stats |
| `ab-test-analysis` | SPRT gating — accept/reject candidate nets |
| `debugger` | Training crashes, encoding mismatches, wrong eval, UCI bugs, native segfaults |

Built-in Task types (not in this folder): `explore`, `shell`, `bugbot`, `security-review`, `ci-investigator`, `best-of-n-runner`.

---

## Source of agents

| Location | Role |
|----------|------|
| `C:\Users\user\.cursor\agents-library\` | Global catalog (~350 agents); **source** for copies below |
| `C:\Users\user\.cursor\agents\` | Global install dir (currently empty) |
| [`Chess AI/.cursor/agents/`](../Chess%20AI/.cursor/agents/) | Immortalite Zero project set (subset of the library) |
| [`.cursor/agents/`](.cursor/agents/) | This project’s installed specialists |

---

## Planned file map

| Area | Paths |
|------|-------|
| C++ core | `cpp/` (board, movegen, encoding, MCTS, pybind bindings) |
| Python ML / train | `engine/` or `python/immortalite_one/` (`network.py`, `train.py`, `selfplay.py`, …) |
| Native extension | pybind11 module via CMake / scikit-build-core |
| Cloud training | `colab/train.ipynb`, `lightning-ai/` |
| UCI | `uci/uci_engine.py` |
| Analysis API | `server/` (later) |
| Tests / parity | `tests/` (encoding + MCTS vs Immortalite Zero oracle) |
| Compat artifacts | `.pt` checkpoints, `samples_iter_*.npz`, `ENCODING_VERSION = 2` |

Oracle / reference (read-only): sibling Immortalite Zero repo at `../Chess AI`.
