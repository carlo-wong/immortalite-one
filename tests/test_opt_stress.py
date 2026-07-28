"""Rigorous stress coverage for semantics-neutral search/inference opts."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from engine import _native
from engine.config import Config, MCTSConfig
from engine.encoding import POLICY_SIZE, board_to_planes
from engine.mcts import MCTS
from engine._python_mcts import PythonMCTS
from engine.network import NetEvaluator
from engine.selfplay import play_games_batched_native_actors


class UniformEvaluator:
    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        return np.zeros(POLICY_SIZE, dtype=np.float32), self.value

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        return (
            np.zeros((n, POLICY_SIZE), dtype=np.float32),
            np.full(n, self.value, dtype=np.float32),
        )

    def evaluate_legal(
        self,
        planes: np.ndarray,
        legal_indices: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(planes.shape[0])
        gathered = np.zeros(len(legal_indices), dtype=np.float32)
        values = np.full(n, self.value, dtype=np.float32)
        return gathered, values


def _random_reachable_positions(seed: int, games: int, max_plies: int) -> list[chess.Board]:
    rng = np.random.default_rng(seed)
    boards: list[chess.Board] = []
    for g in range(games):
        board = chess.Board()
        plies = int(rng.integers(1, max_plies + 1))
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[int(rng.integers(0, len(legal)))])
            if board.is_game_over(claim_draw=True):
                break
            # Snapshot after each ply for denser coverage.
            boards.append(board.copy(stack=True))
        if g == 0 and not boards:
            boards.append(chess.Board())
    return boards


def test_zobrist_random_games_make_unmake_matches_fresh_fen() -> None:
    rng = np.random.default_rng(20260728)
    for game_i in range(40):
        board = chess.Board()
        moves: list[str] = []
        for _ in range(int(rng.integers(8, 60))):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(rng.integers(0, len(legal)))]
            board.push(move)
            moves.append(move.uci())
            if board.is_game_over(claim_draw=True):
                break
        fen = chess.STARTING_FEN
        expected = [int(_native.transposition_key_fen(chess.Board(fen).fen(en_passant="fen")))]
        b = chess.Board(fen)
        for uci in moves:
            b.push_uci(uci)
            expected.append(int(_native.transposition_key_fen(b.fen(en_passant="fen"))))
        forward, backward = _native.zobrist_trace_fen(fen, moves)
        assert [int(k) for k in forward] == expected, (game_i, moves)
        assert [int(k) for k in backward] == list(reversed(expected)), (game_i, moves)


def test_legal_moves_set_and_order_vs_pythonchess_random() -> None:
    boards = _random_reachable_positions(seed=11, games=25, max_plies=40)
    # Include tricky tactical FENs.
    boards.extend(
        [
            chess.Board(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            ),
            chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
            chess.Board("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
            chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
            chess.Board("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        ]
    )
    for board in boards:
        fen = board.fen()
        native = [uci for _, uci in _native.legal_move_indices_fen(fen)]
        # python-chess order: same generator semantics the engine targets.
        py = [m.uci() for m in board.legal_moves]
        assert native == py, fen
        assert len(native) == len(set(native))


@pytest.mark.parametrize("sims", [16, 48])
def test_mcts_visit_parity_random_midgames(sims: int) -> None:
    boards = _random_reachable_positions(seed=99, games=8, max_plies=25)
    # Keep a few midgame positions that still have branching.
    positions = [b for b in boards if not b.is_game_over(claim_draw=True)][:12]
    cfg = MCTSConfig(simulations=sims, claim_draw=True, draw_contempt=1 / 3)
    ev = UniformEvaluator(0.0)
    for board in positions:
        py = PythonMCTS(ev, cfg).run(board.copy(), simulations=sims, add_noise=False)
        native = MCTS(ev, cfg)
        assert native.using_native
        cpp = native.run(board.copy(), simulations=sims, add_noise=False)
        py_visits = {m.uci(): int(n) for m, n in zip(py.moves, py.visits)}
        cpp_visits = {m.uci(): int(n) for m, n in zip(cpp.moves, cpp.visits)}
        assert cpp_visits == py_visits, board.fen()
        assert abs(cpp.root_value - py.root_value) < 1e-5


def test_evaluate_legal_matches_dense_gather_cpu() -> None:
    net = __import__("engine.network", fromlist=["ChessNet"]).ChessNet()
    ev = NetEvaluator(net, device="cpu", graph_mode="off")
    boards = _random_reachable_positions(seed=7, games=6, max_plies=20)[:8]
    planes = np.stack([board_to_planes(b) for b in boards], axis=0).astype(np.float32)
    indices: list[int] = []
    offsets = [0]
    for b in boards:
        idxs = [idx for idx, _ in _native.legal_move_indices_fen(b.fen())]
        indices.extend(idxs)
        offsets.append(len(indices))
    legal_indices = np.asarray(indices, dtype=np.int32)
    legal_offsets = np.asarray(offsets, dtype=np.int32)

    dense_logits, dense_values = ev.evaluate_planes(planes)
    gathered, legal_values = ev.evaluate_legal(planes, legal_indices, legal_offsets)

    expected = []
    for i in range(len(boards)):
        a, b = int(legal_offsets[i]), int(legal_offsets[i + 1])
        expected.append(dense_logits[i, legal_indices[a:b]])
    expected_arr = (
        np.concatenate(expected).astype(np.float32)
        if expected
        else np.empty(0, dtype=np.float32)
    )
    np.testing.assert_allclose(gathered, expected_arr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(legal_values, dense_values, rtol=1e-5, atol=1e-5)


def test_native_actors_uniform_selfplay_completes_and_is_sane() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 24
    cfg.mcts.simulations = 24
    cfg.mcts.claim_draw = True
    games = play_games_batched_native_actors(
        UniformEvaluator(0.0),
        cfg,
        simulations=24,
        num_games=4,
        concurrency=4,
        add_noise=False,
        exploration_moves=0,
    )
    assert len(games) == 4
    for game in games:
        assert game.samples
        assert len(game.samples) == len(game.moves)
        for s in game.samples:
            assert s.planes.shape == (20, 8, 8)
            assert s.policy.shape == (POLICY_SIZE,)
            assert np.isfinite(s.policy).all()
            assert abs(float(s.policy.sum()) - 1.0) < 1e-3
            assert -1.0 <= float(s.value) <= 1.0


def test_ep_and_castling_rights_zobrist_edge_matrix() -> None:
    cases = [
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", ["e4d3"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8g8"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1c1", "e8c8"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["a1a8", "h8h1"]),
        ("4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1", ["b7b8q", "b2b1r"]),
        ("4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1", ["b7b8n", "b2b1b"]),
        ("2n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1", ["b7c8q"]),
    ]
    for fen, moves in cases:
        legal_moves: list[str] = []
        b = chess.Board(fen)
        for uci in moves:
            move = chess.Move.from_uci(uci)
            if move not in b.legal_moves:
                break
            b.push(move)
            legal_moves.append(uci)
        assert legal_moves, (fen, moves)
        expected = [int(_native.transposition_key_fen(chess.Board(fen).fen(en_passant="fen")))]
        b = chess.Board(fen)
        for uci in legal_moves:
            b.push_uci(uci)
            expected.append(int(_native.transposition_key_fen(b.fen(en_passant="fen"))))
        forward, backward = _native.zobrist_trace_fen(fen, legal_moves)
        assert [int(k) for k in forward] == expected
        assert [int(k) for k in backward] == list(reversed(expected))


def test_deeper_perft_start_and_kiwipete() -> None:
    # Depth 4 startpos is the classic 197k nodes — catches legality/unmake bugs.
    board = chess.Board()

    def native_perft(b: chess.Board, depth: int) -> int:
        if depth == 0:
            return 1
        moves = [chess.Move.from_uci(u) for _, u in _native.legal_move_indices_fen(b.fen())]
        assert set(moves) == set(b.legal_moves)
        if depth == 1:
            return len(moves)
        nodes = 0
        for m in moves:
            b.push(m)
            nodes += native_perft(b, depth - 1)
            b.pop()
        return nodes

    assert native_perft(board, 4) == 197_281
    kiwi = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    assert native_perft(kiwi, 3) == 97_862
