#include "encoding.hpp"

#include "movegen.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace immortalite {
namespace {

// Queen dirs (d_rank, d_file): N, NE, E, SE, S, SW, W, NW
constexpr int kQueenDirs[8][2] = {
    {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

// Knight offsets (d_rank, d_file)
constexpr int kKnightOffsets[8][2] = {
    {2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};

constexpr PieceType kUnderpromo[3] = {KNIGHT, BISHOP, ROOK};

int sign(int x) { return (x > 0) - (x < 0); }

int queen_dir_index(int dr, int df) {
  for (int i = 0; i < 8; ++i)
    if (kQueenDirs[i][0] == dr && kQueenDirs[i][1] == df) return i;
  return -1;
}

int knight_index(int dr, int df) {
  for (int i = 0; i < 8; ++i)
    if (kKnightOffsets[i][0] == dr && kKnightOffsets[i][1] == df) return i;
  return -1;
}

inline void set_plane(float* planes, int p, int rank, int file, float v) {
  planes[(p * 8 + rank) * 8 + file] = v;
}

inline void fill_plane(float* planes, int p, float v) {
  float* dst = planes + p * 64;
  for (int i = 0; i < 64; ++i) dst[i] = v;
}

}  // namespace

void fill_planes(const Position& pos, float* out) {
  std::memset(out, 0, sizeof(float) * NUM_INPUT_PLANES * 64);

  const bool white_tm = pos.side_to_move() == WHITE;

  for (Square s = 0; s < 64; ++s) {
    Piece p = pos.piece_on(s);
    if (p == NO_PIECE) continue;
    Square canon = white_tm ? s : square_mirror(s);
    int rank = rank_of(canon);
    int file = file_of(canon);
    PieceType pt = type_of(p);
    Color pc = color_of(p);
    // us = side to move
    bool us = white_tm ? (pc == WHITE) : (pc == BLACK);
    int plane = static_cast<int>(pt) + (us ? 0 : 6);
    set_plane(out, plane, rank, file, 1.0f);
  }

  if (white_tm) {
    if (pos.has_kingside_castling(WHITE)) fill_plane(out, 12, 1.0f);
    if (pos.has_queenside_castling(WHITE)) fill_plane(out, 13, 1.0f);
    if (pos.has_kingside_castling(BLACK)) fill_plane(out, 14, 1.0f);
    if (pos.has_queenside_castling(BLACK)) fill_plane(out, 15, 1.0f);
    if (pos.ep_square() != NO_SQUARE) {
      Square ep = pos.ep_square();
      set_plane(out, 16, rank_of(ep), file_of(ep), 1.0f);
    }
  } else {
    if (pos.has_kingside_castling(BLACK)) fill_plane(out, 12, 1.0f);
    if (pos.has_queenside_castling(BLACK)) fill_plane(out, 13, 1.0f);
    if (pos.has_kingside_castling(WHITE)) fill_plane(out, 14, 1.0f);
    if (pos.has_queenside_castling(WHITE)) fill_plane(out, 15, 1.0f);
    if (pos.ep_square() != NO_SQUARE) {
      Square ep = square_mirror(pos.ep_square());
      set_plane(out, 16, rank_of(ep), file_of(ep), 1.0f);
    }
  }

  if (pos.is_repetition(3)) {
    fill_plane(out, 17, 1.0f);
    fill_plane(out, 18, 1.0f);
  } else if (pos.is_repetition(2)) {
    fill_plane(out, 17, 1.0f);
  }

  float hm = std::min(static_cast<float>(pos.halfmove_clock()) / 100.0f, 1.0f);
  fill_plane(out, 19, hm);
}

int move_to_index(const Position& pos, Move m) {
  Square from_sq = m.from();
  Square to_sq = m.to();
  if (pos.side_to_move() == BLACK) {
    from_sq = square_mirror(from_sq);
    to_sq = square_mirror(to_sq);
  }
  int fr = rank_of(from_sq), ff = file_of(from_sq);
  int tr = rank_of(to_sq), tf = file_of(to_sq);
  int d_rank = tr - fr, d_file = tf - ff;

  PieceType promo = m.promotion();
  if (promo != NO_PIECE_TYPE && promo != QUEEN) {
    int dir_idx = sign(d_file) + 1;
    int piece_idx = 0;
    if (promo == BISHOP) piece_idx = 1;
    else if (promo == ROOK) piece_idx = 2;
    int plane = 64 + dir_idx * 3 + piece_idx;
    return from_sq * PLANES_PER_FROM + plane;
  }

  int ki = knight_index(d_rank, d_file);
  if (ki >= 0) {
    return from_sq * PLANES_PER_FROM + 56 + ki;
  }

  int step_r = sign(d_rank), step_f = sign(d_file);
  int distance = std::max(std::abs(d_rank), std::abs(d_file));
  int dir_idx = queen_dir_index(step_r, step_f);
  int plane = dir_idx * 7 + (distance - 1);
  return from_sq * PLANES_PER_FROM + plane;
}

Move index_to_move(const Position& pos, int index) {
  if (index < 0 || index >= POLICY_SIZE) return NULL_MOVE;
  int from_canon = index / PLANES_PER_FROM;
  int plane = index % PLANES_PER_FROM;
  int fr = rank_of(from_canon), ff = file_of(from_canon);
  int tr = 0, tf = 0;
  PieceType promotion = NO_PIECE_TYPE;

  if (plane < 56) {
    int dir_idx = plane / 7;
    int distance = (plane % 7) + 1;
    tr = fr + kQueenDirs[dir_idx][0] * distance;
    tf = ff + kQueenDirs[dir_idx][1] * distance;
  } else if (plane < 64) {
    int ki = plane - 56;
    tr = fr + kKnightOffsets[ki][0];
    tf = ff + kKnightOffsets[ki][1];
  } else {
    int p = plane - 64;
    int dir_idx = p / 3;
    int piece_idx = p % 3;
    int d_file = dir_idx - 1;
    tr = fr + 1;
    tf = ff + d_file;
    promotion = kUnderpromo[piece_idx];
  }

  if (tr < 0 || tr > 7 || tf < 0 || tf > 7) return NULL_MOVE;
  Square to_canon = square_of(tf, tr);
  Square from_sq = from_canon;
  Square to_sq = to_canon;
  if (pos.side_to_move() == BLACK) {
    from_sq = square_mirror(from_canon);
    to_sq = square_mirror(to_canon);
  }

  if (promotion == NO_PIECE_TYPE) {
    Piece piece = pos.piece_on(from_sq);
    if (piece != NO_PIECE && type_of(piece) == PAWN) {
      int to_rank = rank_of(to_sq);
      if ((color_of(piece) == WHITE && to_rank == 7) ||
          (color_of(piece) == BLACK && to_rank == 0)) {
        promotion = QUEEN;
      }
    }
  }

  Move buf[256];
  int n = generate_legal_moves(pos, buf, 256);
  for (int i = 0; i < n; ++i) {
    if (buf[i].from() == from_sq && buf[i].to() == to_sq) {
      if (promotion == NO_PIECE_TYPE) {
        if (buf[i].promotion() == NO_PIECE_TYPE) return buf[i];
      } else if (buf[i].promotion() == promotion) {
        return buf[i];
      }
    }
  }
  return NULL_MOVE;
}

std::vector<std::pair<int, Move>> legal_move_indices(const Position& pos) {
  std::vector<Move> moves;
  generate_legal_moves(pos, moves);
  std::vector<std::pair<int, Move>> out;
  out.reserve(moves.size());
  for (Move m : moves) {
    out.emplace_back(move_to_index(pos, m), m);
  }
  return out;
}

}  // namespace immortalite
