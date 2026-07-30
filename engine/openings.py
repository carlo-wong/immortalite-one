"""Opening prefixes: masters gate book + tabula-rasa self-play prefixes."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import chess
import numpy as np

DEFAULT_MASTERS_OPENINGS_PATH = (
    Path(__file__).resolve().parent / "data" / "masters_prefix_free_top128.tsv"
)

_SAN_MOVE_RE = re.compile(r"[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?|O-O-O|O-O")


def _parse_pgn_sans(pgn: str) -> list[str]:
    """Extract SAN tokens from a numbered PGN fragment (no result / comments)."""
    clean = re.sub(r"\d+\.+", " ", pgn)
    clean = re.sub(r"[()]", " ", clean)
    toks: list[str] = []
    for tok in clean.split():
        tok = tok.rstrip("+#")  # check / mate markers
        if _SAN_MOVE_RE.fullmatch(tok):
            toks.append(tok)
    return toks


def pgn_to_uci(pgn: str) -> list[str]:
    """Convert a short PGN move list to UCI. Raises ValueError if illegal."""
    board = chess.Board()
    uci: list[str] = []
    for san in _parse_pgn_sans(pgn):
        try:
            move = board.parse_san(san)
        except ValueError as exc:
            raise ValueError(f"illegal SAN {san!r} in {pgn!r}: {exc}") from exc
        board.push(move)
        uci.append(move.uci())
    if not uci:
        raise ValueError(f"no moves parsed from {pgn!r}")
    return uci


def load_opening_book(path: str | Path | None = None) -> list[list[str]]:
    """Load prefix-free masters TSV → list of UCI move lists (one per opening)."""
    book_path = Path(path) if path is not None else DEFAULT_MASTERS_OPENINGS_PATH
    if not book_path.is_file():
        raise FileNotFoundError(f"opening book not found: {book_path}")

    openings: list[list[str]] = []
    with book_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "pgn" not in reader.fieldnames:
            raise ValueError(f"opening book missing 'pgn' column: {book_path}")
        for row in reader:
            pgn = (row.get("pgn") or "").strip()
            if not pgn:
                continue
            openings.append(pgn_to_uci(pgn))
    if not openings:
        raise ValueError(f"opening book empty: {book_path}")
    return openings


def load_default_gate_openings() -> list[list[str]]:
    """128 masters lines for gate_games=256 (each line × both colors)."""
    return load_opening_book(DEFAULT_MASTERS_OPENINGS_PATH)


def opening_for_game(openings: list[list[str]] | None, game_idx: int) -> list[str] | None:
    """Map game index → opening: opening_idx = game_idx // 2 (color-paired)."""
    if not openings:
        return None
    return openings[(game_idx // 2) % len(openings)]


def random_legal_opening_prefixes(
    num_games: int,
    plies: int,
    *,
    rng: Any | None = None,
) -> list[list[str]]:
    """Uniform-random legal UCI prefixes for self-play (no human book).

    Each game independently samples ``plies`` legal moves from the start
    position (and subsequent positions). Empty when ``plies <= 0``.
    """
    if num_games < 0:
        raise ValueError("num_games must be >= 0")
    if plies < 0:
        raise ValueError("plies must be >= 0")
    if num_games == 0 or plies == 0:
        return [[] for _ in range(num_games)]
    gen = rng if rng is not None else np.random.default_rng()
    out: list[list[str]] = []
    for _ in range(num_games):
        board = chess.Board()
        moves: list[str] = []
        for _ply in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(gen.integers(0, len(legal)))]
            board.push(move)
            moves.append(move.uci())
        out.append(moves)
    return out


def diversity_move_uci(moves: list[str], random_opening_plies: int) -> str | None:
    """First MCTS-chosen UCI for diversity logging (skips forced prefix plies)."""
    k = max(0, int(random_opening_plies))
    if k < len(moves):
        return moves[k]
    return None
