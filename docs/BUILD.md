# Building Immortalite Zero

## Local editable install (developers)

```bash
# Linux
sudo apt-get install -y build-essential ninja-build python3-dev
pip install -r requirements.txt
pip install -e .
python -c "from engine import _native; print(_native.version())"
```

```powershell
# Windows — use "x64 Native Tools" / Developer PowerShell (MSVC on PATH)
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins `cmake>=3.26,<4.0` and installs PyTorch for local/CPU setups.

**Requirements**

| Piece | Notes |
|-------|--------|
| CMake | `>=3.26,<4.0` (CMake 4.4+ nested MSVC checks can fail) |
| Ninja | Used by `pyproject.toml` (`cmake.args = ["-G", "Ninja"]`) |
| C++20 | MSVC on Windows; g++/clang on Linux |
| PyTorch | Runtime dep; not linked into the C++ extension |

The native module is **CPU-only** (board / MCTS / encoding). NN inference stays in PyTorch.

## Colab / Lightning (keep preinstalled CUDA torch)

Do **not** run `pip install -r requirements.txt` on managed GPU runtimes — that can replace the studio’s CUDA PyTorch with a CPU wheel from PyPI.

```bash
sudo apt-get install -y build-essential ninja-build python3-dev
pip install -q "cmake>=3.26,<4.0" ninja pybind11 scikit-build-core
pip install -q python-chess numpy tqdm
pip install -q -e . --no-deps
python -c "import torch; from engine import _native; print(torch.__version__, torch.cuda.is_available(), _native.version())"
```

`colab/train.ipynb` cell 2 and [lightning-ai/README.md](../lightning-ai/README.md) follow this recipe. Rebuild after pulling C++ changes. Drive / sibling checkpoint folders do not include the compiled extension — each runtime compiles once (or install a matching wheel, below).

## Prebuilt wheels (future / CI artifacts)

CI workflow [`.github/workflows/wheels.yml`](../.github/workflows/wheels.yml) builds:

- Linux wheel on `ubuntu-latest` (platform tag from the runner)
- Windows `win_amd64` wheel

Trigger: `workflow_dispatch` or push tag `v*`. Download the artifact and:

```bash
pip install immortalite_zero-*.whl --no-deps
```

**Not on PyPI yet.** Until wheels are published for your exact Python/ABI:

- Prefer `pip install -e . --no-deps` on Colab/Lightning after `git pull`
- Or download a CI wheel that matches `cp311` + OS

### Planned manylinux story

1. Switch wheel job to [`cibuildwheel`](https://cibuildwheel.readthedocs.io/) for `manylinux_2_28_x86_64` / `musllinux` as needed  
2. Publish to GitHub Releases or PyPI on version tags  
3. Colab cell: `pip install` wheel URL first (`--no-deps`), fall back to `pip install -e . --no-deps`

## Spawn / multiprocessing

Self-play workers use `multiprocessing` **spawn**. `engine._native` must import cleanly in child processes (no leftover non-picklable global board state). The extension is load-once per process; weight reloads are Python/PyTorch only.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: engine._native` | Build tools missing or wrong shell; re-run `pip install -e . --no-deps` (cloud) or `pip install -e .` (local) |
| CMake 4.x fail on Windows | `pip install "cmake>=3.26,<4.0"` |
| CUDA torch replaced by CPU build | Reinstall Colab/Lightning torch, then use `--no-deps` for `-e .` |
| MinGW `.pyd` won't load | Build with MSVC, not MinGW |
| Colab compile OOM / timeout | Retry; or install a CI wheel for cp3xx-linux |
