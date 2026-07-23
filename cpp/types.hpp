#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace immortalite {

inline constexpr int ENCODING_VERSION = 2;
inline constexpr int NUM_INPUT_PLANES = 20;
inline constexpr int POLICY_SIZE = 64 * 73;  // 4672
inline constexpr int PLANES_PER_FROM = 73;

using Bitboard = std::uint64_t;
using Square = int;  // 0..63, a1=0, h1=7, a8=56, h8=63

enum Color : int { WHITE = 0, BLACK = 1 };

inline constexpr Color operator~(Color c) {
  return c == WHITE ? BLACK : WHITE;
}

enum PieceType : int {
  PAWN = 0,
  KNIGHT = 1,
  BISHOP = 2,
  ROOK = 3,
  QUEEN = 4,
  KING = 5,
  NO_PIECE_TYPE = 6
};

enum Piece : int {
  NO_PIECE = 0,
  W_PAWN = 1,
  W_KNIGHT = 2,
  W_BISHOP = 3,
  W_ROOK = 4,
  W_QUEEN = 5,
  W_KING = 6,
  B_PAWN = 7,
  B_KNIGHT = 8,
  B_BISHOP = 9,
  B_ROOK = 10,
  B_QUEEN = 11,
  B_KING = 12
};

inline constexpr Piece make_piece(Color c, PieceType pt) {
  return static_cast<Piece>(1 + static_cast<int>(pt) + (c == BLACK ? 6 : 0));
}

inline constexpr Color color_of(Piece p) {
  return p <= W_KING ? WHITE : BLACK;
}

inline constexpr PieceType type_of(Piece p) {
  return static_cast<PieceType>((static_cast<int>(p) - 1) % 6);
}

enum CastlingRight : std::uint8_t {
  WHITE_OO = 1,
  WHITE_OOO = 2,
  BLACK_OO = 4,
  BLACK_OOO = 8
};

enum MoveFlags : std::uint8_t {
  MF_NONE = 0,
  MF_CAPTURE = 1 << 0,
  MF_DOUBLE_PAWN = 1 << 1,
  MF_EP = 1 << 2,
  MF_CASTLE = 1 << 3,
  MF_PROMOTION = 1 << 4
};

// from:6 | to:6 | promo:3 | flags:5  (promo 0..5, 7 = none)
struct Move {
  std::uint32_t raw = 0;

  Move() = default;

  static constexpr Move make(Square from, Square to, PieceType promo = NO_PIECE_TYPE,
                             std::uint8_t flags = MF_NONE) {
    Move m;
    const unsigned pr = (promo == NO_PIECE_TYPE) ? 7u : static_cast<unsigned>(promo);
    m.raw = static_cast<std::uint32_t>((from & 63) | ((to & 63) << 6) | (pr << 12) |
                                       (static_cast<unsigned>(flags) << 15));
    return m;
  }

  Square from() const { return static_cast<Square>(raw & 63u); }
  Square to() const { return static_cast<Square>((raw >> 6) & 63u); }
  PieceType promotion() const {
    unsigned p = (raw >> 12) & 7u;
    return p >= 6 ? NO_PIECE_TYPE : static_cast<PieceType>(p);
  }
  std::uint8_t flags() const { return static_cast<std::uint8_t>((raw >> 15) & 0x1Fu); }

  bool null() const { return raw == 0; }

  bool operator==(Move o) const { return raw == o.raw; }
  bool operator!=(Move o) const { return raw != o.raw; }
};

inline constexpr Move NULL_MOVE{};

inline constexpr Square square_of(int file, int rank) { return rank * 8 + file; }
inline constexpr int file_of(Square s) { return s & 7; }
inline constexpr int rank_of(Square s) { return s >> 3; }
inline constexpr Square square_mirror(Square s) { return s ^ 56; }

inline constexpr Bitboard bit(Square s) {
  return Bitboard{1} << s;
}

inline constexpr Square NO_SQUARE = -1;

inline constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
inline constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
inline constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
inline constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
inline constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
inline constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
inline constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
inline constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;
inline constexpr Bitboard FILE_A = 0x0101010101010101ULL;
inline constexpr Bitboard FILE_B = 0x0202020202020202ULL;
inline constexpr Bitboard FILE_G = 0x4040404040404040ULL;
inline constexpr Bitboard FILE_H = 0x8080808080808080ULL;

inline int popcount(Bitboard b) {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt64(b));
#else
  return __builtin_popcountll(b);
#endif
}

inline Square lsb(Bitboard b) {
#if defined(_MSC_VER)
  unsigned long idx;
  _BitScanForward64(&idx, b);
  return static_cast<Square>(idx);
#else
  return static_cast<Square>(__builtin_ctzll(b));
#endif
}

inline Square poplsb(Bitboard& b) {
  Square s = lsb(b);
  b &= b - 1;
  return s;
}

inline Square msb(Bitboard b) {
#if defined(_MSC_VER)
  unsigned long idx;
  _BitScanReverse64(&idx, b);
  return static_cast<Square>(idx);
#else
  return static_cast<Square>(63 - __builtin_clzll(b));
#endif
}

// python-chess scan_reversed: yield highest set bits first.
inline Square popmsb(Bitboard& b) {
  Square s = msb(b);
  b ^= bit(s);
  return s;
}

inline std::string square_to_string(Square s) {
  char buf[3] = {static_cast<char>('a' + file_of(s)), static_cast<char>('1' + rank_of(s)), 0};
  return buf;
}

inline Square square_from_string(std::string_view s) {
  if (s.size() < 2) return NO_SQUARE;
  int f = s[0] - 'a';
  int r = s[1] - '1';
  if (f < 0 || f > 7 || r < 0 || r > 7) return NO_SQUARE;
  return square_of(f, r);
}

inline std::string move_to_uci(Move m) {
  std::string u = square_to_string(m.from()) + square_to_string(m.to());
  if (m.promotion() != NO_PIECE_TYPE) {
    static constexpr char kPromo[] = "pnbrqk";
    u.push_back(kPromo[m.promotion()]);
  }
  return u;
}

enum class Termination {
  None,
  Checkmate,
  Stalemate,
  InsufficientMaterial,
  SeventyfiveMoves,
  FivefoldRepetition,
  FiftyMoves,
  ThreefoldRepetition
};

struct Outcome {
  Termination termination = Termination::None;
  int winner = -1;  // 0 white, 1 black, -1 draw
  bool is_terminal() const { return termination != Termination::None; }
};

}  // namespace immortalite
