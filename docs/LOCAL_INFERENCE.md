# Local inference (localhost)

One-shot: start the FastAPI analysis server + web GUI from the repo root (after `pip install -e .`).

## One-shot (auto-discover checkpoint)

```bash
python -m uvicorn server.app:app --port 8000
```

Open: **http://localhost:8000/app/**

Checkpoint search order (first that exists wins):

1. `results/latest.pt`
2. `checkpoints/latest.pt`
3. `results/immortalite_zero_checkpoints/latest.pt`
4. `results/immortalite_one_checkpoints/latest.pt` (legacy folder name)

Or set `IMMORTALITE_ZERO_CHECKPOINT` (alias: `IMMORTALITE_ONE_CHECKPOINT`). Device defaults to CPU; set `IMMORTALITE_ZERO_DEVICE=cuda` for GPU.

## One-shot with an explicit checkpoint (PowerShell)

```powershell
$env:IMMORTALITE_ZERO_CHECKPOINT="results\latest.pt"; $env:IMMORTALITE_ZERO_DEVICE="cpu"; python -m uvicorn server.app:app --port 8000
```

From sibling Lightning / archived weights:

```powershell
$env:IMMORTALITE_ZERO_CHECKPOINT="..\results\latest.pt"; python -m uvicorn server.app:app --port 8000
```

## Quick checks

```bash
curl http://localhost:8000/health
```

Reload on code edits: add `--reload`.

## UCI (optional)

For Arena / CuteChess / Lichess local (not the web GUI):

```bash
python -m uci.uci_engine path/to/latest.pt
```
