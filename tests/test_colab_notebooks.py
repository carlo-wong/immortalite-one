"""Static compatibility checks for executable Colab notebook commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "colab" / "train.ipynb",
    ROOT / "colab" / "benchmark_throughput.ipynb",
)


def _code(notebook: Path) -> str:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    cells = document["cells"]
    assert isinstance(cells, list) and cells
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    assert code_cells
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in code_cells
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_colab_notebook_builds_native_extension_editably(notebook: Path) -> None:
    code = _code(notebook)

    assert '"cmake>=3.26,<4.0"' in code
    assert "!pip install -q -e . --no-deps" in code
    assert "from engine import _native" in code


def test_training_notebook_uses_engine_train_entrypoint() -> None:
    code = _code(ROOT / "colab" / "train.ipynb")

    assert (ROOT / "engine" / "train.py").is_file()
    assert "!python -m engine.train " in code
    assert "metrics_training.csv" in code
    assert "metrics_performance.csv" in code
    for option in (
        "--device cuda",
        "--checkpoint-dir",
        "--value-target",
        "--selfplay-workers",
        "--concurrency",
        "--replay-buffer",
        "--resume",
    ):
        assert option in code


def test_benchmark_notebook_uses_throughput_entrypoint() -> None:
    code = _code(ROOT / "colab" / "benchmark_throughput.ipynb")

    assert (ROOT / "scripts" / "bench_throughput.py").is_file()
    assert "'python', 'scripts/bench_throughput.py'" in code
    assert "subprocess.run(" in code
    for option in ("--checkpoint", "--output", "--device", "--workers", "--concurrency"):
        assert f"'{option}'" in code
