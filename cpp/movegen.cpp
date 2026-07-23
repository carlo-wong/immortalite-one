#include "movegen.hpp"

#include "board.hpp"

#include <array>
#include <cassert>

namespace immortalite {
namespace {

Bitboard kKnightAttacks[64];
Bitboard kKingAttacks[64];
Bitboard kPawnAttacks[2][64];
Bitboard kRayAttacks[64][8];  // N, NE, E, SE, S, SW, W, NW

constexpr int kDirDelta[8] = {8, 9, 1, -7, -8, -9, -1, 7};

bool valid_step(Square from, int df, int dr) {
  int f = file_of(from) + df;
  int r = rank_of(from) + dr;
  return f >= 0 && f <= 7 && r >= 0 && r <= 7;
}

void init_tables_once() {
  static bool done = false;
  if (done) return;
  done = true;

  constexpr int kN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
  constexpr int kK[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
  constexpr int kDir[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

  for (Square s = 0; s < 64; ++s) {
    Bitboard n = 0, k = 0;
    for (auto& d : kN) {
      if (valid_step(s, d[0], d[1])) n |= bit(square_of(file_of(s) + d[0], rank_of(s) + d[1]));
    }
    for (auto& d : kK) {
      if (valid_step(s, d[0], d[1])) k |= bit(square_of(file_of(s) + d[0], rank_of(s) + d[1]));
    }
    kKnightAttacks[s] = n;
    kKingAttacks[s] = k;

    Bitboard wp = 0, bp = 0;
    if (valid_step(s, -1, 1)) wp |= bit(square_of(file_of(s) - 1, rank_of(s) + 1));
    if (valid_step(s, 1, 1)) wp |= bit(square_of(file_of(s) + 1, rank_of(s) + 1));
    if (valid_step(s, -1, -1)) bp |= bit(square_of(file_of(s) - 1, rank_of(s) - 1));
    if (valid_step(s, 1, -1)) bp |= bit(square_of(file_of(s) + 1, rank_of(s) - 1));
    kPawnAttacks[WHITE][s] = wp;
    kPawnAttacks[BLACK][s] = bp;

    for (int dir = 0; dir < 8; ++dir) {
      Bitboard ray = 0;
      int f = file_of(s) + kDir[dir][0];
      int r = rank_of(s) + kDir[dir][1];
      while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        ray |= bit(square_of(f, r));
        f += kDir[dir][0];
        r += kDir[dir][1];
      }
      kRayAttacks[s][dir] = ray;
    }
  }
}

Bitboard sliding_attacks(Square s, Bitboard occ, const int* dirs, int n_dirs) {
  Bitboard attacks = 0;
  for (int i = 0; i < n_dirs; ++i) {
    int dir = dirs[i];
    Bitboard ray = kRayAttacks[s][dir];
    Bitboard blockers = ray & occ;
    if (blockers) {
      // Positive dirs: N,NE,E,SE use lowest blocker; negative use highest.
      Square block;
      if (kDirDelta[dir] > 0) {
        block = lsb(blockers);
      } else {
        block = msb(blockers);
      }
      attacks |= kRayAttacks[s][dir] ^ kRayAttacks[block][dir];
    } else {
      attacks |= ray;
    }
  }
  return attacks;
}

constexpr int kBishopDirs[4] = {1, 3, 5, 7};
constexpr int kRookDirs[4] = {0, 2, 4, 6};

void add_move(Move* out, int& n, int max_out, Square from, Square to, PieceType promo,
              std::uint8_t flags) {
  if (n < max_out) out[n] = Move::make(from, to, promo, flags);
  ++n;
}

Bitboard attacks_from(const Position& pos, Square from, Bitboard occ) {
  Piece p = pos.piece_on(from);
  switch (type_of(p)) {
    case PAWN:
      return kPawnAttacks[color_of(p)][from];
    case KNIGHT:
      return kKnightAttacks[from];
    case BISHOP:
      return bishop_attacks_bb(from, occ);
    case ROOK:
      return rook_attacks_bb(from, occ);
    case QUEEN:
      return queen_attacks_bb(from, occ);
    case KING:
      return kKingAttacks[from];
    default:
      return 0;
  }
}

int gen_pseudo_legal(const Position& pos, Move* out, int max_out) {
  // Order mirrors python-chess Board.generate_pseudo_legal_moves (scan_reversed).
  init_tables_once();
  int n = 0;
  const Color us = pos.side_to_move();
  const Color them = ~us;
  const Bitboard ours = pos.pieces(us);
  const Bitboard theirs = pos.pieces(them);
  const Bitboard occ = pos.occupied();
  const Bitboard empty = ~occ;
  Bitboard pawns = pos.pieces(us, PAWN);

  // Non-pawn piece moves (msb-first from/to).
  Bitboard non_pawns = ours & ~pawns;
  while (non_pawns) {
    Square from = popmsb(non_pawns);
    Bitboard att = attacks_from(pos, from, occ) & ~ours;
    while (att) {
      Square to = popmsb(att);
      std::uint8_t fl = (theirs & bit(to)) ? MF_CAPTURE : MF_NONE;
      add_move(out, n, max_out, from, to, NO_PIECE_TYPE, fl);
    }
  }

  // Castling
  if (!pos.in_check()) {
    if (us == WHITE) {
      if (pos.has_kingside_castling(WHITE) && !(occ & (bit(5) | bit(6))) &&
          !pos.is_attacked(5, BLACK) && !pos.is_attacked(6, BLACK)) {
        add_move(out, n, max_out, 4, 6, NO_PIECE_TYPE, MF_CASTLE);
      }
      if (pos.has_queenside_castling(WHITE) && !(occ & (bit(1) | bit(2) | bit(3))) &&
          !pos.is_attacked(3, BLACK) && !pos.is_attacked(2, BLACK)) {
        add_move(out, n, max_out, 4, 2, NO_PIECE_TYPE, MF_CASTLE);
      }
    } else {
      if (pos.has_kingside_castling(BLACK) && !(occ & (bit(61) | bit(62))) &&
          !pos.is_attacked(61, WHITE) && !pos.is_attacked(62, WHITE)) {
        add_move(out, n, max_out, 60, 62, NO_PIECE_TYPE, MF_CASTLE);
      }
      if (pos.has_queenside_castling(BLACK) && !(occ & (bit(57) | bit(58) | bit(59))) &&
          !pos.is_attacked(59, WHITE) && !pos.is_attacked(58, WHITE)) {
        add_move(out, n, max_out, 60, 58, NO_PIECE_TYPE, MF_CASTLE);
      }
    }
  }

  // Pawn captures (and promo captures), msb-first
  Bitboard cap_pawns = pawns;
  while (cap_pawns) {
    Square from = popmsb(cap_pawns);
    Bitboard att = kPawnAttacks[us][from] & theirs;
    while (att) {
      Square to = popmsb(att);
      int to_rank = rank_of(to);
      if (to_rank == 0 || to_rank == 7) {
        for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
          add_move(out, n, max_out, from, to, pt,
                   static_cast<std::uint8_t>(MF_PROMOTION | MF_CAPTURE));
        }
      } else {
        add_move(out, n, max_out, from, to, NO_PIECE_TYPE, MF_CAPTURE);
      }
    }
  }

  // Single / double pushes
  const int push = us == WHITE ? 8 : -8;
  Bitboard single = us == WHITE ? (pawns << 8) & empty : (pawns >> 8) & empty;
  Bitboard doubles =
      us == WHITE ? ((single & RANK_3) << 8) & empty : ((single & RANK_6) >> 8) & empty;

  Bitboard single_moves = single;
  while (single_moves) {
    Square to = popmsb(single_moves);
    Square from = to - push;
    int to_rank = rank_of(to);
    if (to_rank == 0 || to_rank == 7) {
      for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
        add_move(out, n, max_out, from, to, pt, MF_PROMOTION);
      }
    } else {
      add_move(out, n, max_out, from, to, NO_PIECE_TYPE, MF_NONE);
    }
  }
  while (doubles) {
    Square to = popmsb(doubles);
    Square from = to - 2 * push;
    add_move(out, n, max_out, from, to, NO_PIECE_TYPE, MF_DOUBLE_PAWN);
  }

  // En passant
  if (pos.ep_square() != NO_SQUARE) {
    Square ep = pos.ep_square();
    Bitboard attackers = kPawnAttacks[them][ep] & pawns;
    while (attackers) {
      Square from = popmsb(attackers);
      add_move(out, n, max_out, from, ep, NO_PIECE_TYPE, MF_EP | MF_CAPTURE);
    }
  }

  return n;
}

bool leaves_king_in_check(Position& pos, Move m) {
  Color us = pos.side_to_move();
  pos.make_move(m);
  bool bad = pos.is_attacked(pos.king_square(us), pos.side_to_move());
  pos.unmake_move();
  return bad;
}

}  // namespace

void init_attack_tables() { init_tables_once(); }

Bitboard pawn_attacks_bb(Color c, Square s) {
  init_tables_once();
  return kPawnAttacks[c][s];
}
Bitboard knight_attacks_bb(Square s) {
  init_tables_once();
  return kKnightAttacks[s];
}
Bitboard king_attacks_bb(Square s) {
  init_tables_once();
  return kKingAttacks[s];
}
Bitboard bishop_attacks_bb(Square s, Bitboard occ) {
  init_tables_once();
  return sliding_attacks(s, occ, kBishopDirs, 4);
}
Bitboard rook_attacks_bb(Square s, Bitboard occ) {
  init_tables_once();
  return sliding_attacks(s, occ, kRookDirs, 4);
}
Bitboard queen_attacks_bb(Square s, Bitboard occ) {
  return bishop_attacks_bb(s, occ) | rook_attacks_bb(s, occ);
}

int generate_pseudo_ep(const Position& pos, Move* out, int max_out) {
  init_tables_once();
  int n = 0;
  if (pos.ep_square() == NO_SQUARE) return 0;
  Color us = pos.side_to_move();
  Color them = ~us;
  Square ep = pos.ep_square();
  Bitboard attackers = kPawnAttacks[them][ep] & pos.pieces(us, PAWN);
  while (attackers) {
    Square from = poplsb(attackers);
    add_move(out, n, max_out, from, ep, NO_PIECE_TYPE, MF_EP | MF_CAPTURE);
  }
  return n;
}

int generate_legal_moves(const Position& pos, Move* out, int max_out) {
  Move buf[256];
  int pseudo = gen_pseudo_legal(pos, buf, 256);
  // Need mutable copy for make/unmake legality filter.
  Position tmp = pos;
  int n = 0;
  for (int i = 0; i < pseudo; ++i) {
    if (!leaves_king_in_check(tmp, buf[i])) {
      if (n < max_out) out[n] = buf[i];
      ++n;
    }
  }
  // Match python-chess Board.generate_legal_moves / _generate_evasions:
  // when in check, king flights come first, then other evasions, preserving
  // relative order within each group (affects PUCT first-max tie-breaks).
  if (pos.in_check() && n > 1) {
    const int write_n = n < max_out ? n : max_out;
    Square ksq = pos.king_square(pos.side_to_move());
    Move reordered[256];
    int w = 0;
    for (int i = 0; i < write_n; ++i) {
      if (out[i].from() == ksq) reordered[w++] = out[i];
    }
    for (int i = 0; i < write_n; ++i) {
      if (out[i].from() != ksq) reordered[w++] = out[i];
    }
    for (int i = 0; i < write_n; ++i) out[i] = reordered[i];
  }
  return n;
}

int generate_legal_moves(const Position& pos, std::vector<Move>& out) {
  Move buf[256];
  int n = generate_legal_moves(pos, buf, 256);
  out.assign(buf, buf + n);
  return n;
}

}  // namespace immortalite
