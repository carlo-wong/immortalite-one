#pragma once

#include "types.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace immortalite {

struct Undo {
  Move move;
  Piece captured = NO_PIECE;
  std::uint8_t castling = 0;
  Square ep_square = NO_SQUARE;
  int halfmove = 0;
  std::uint64_t key_before = 0;  // transposition key before the move
  bool ep_keyed_before = false;
  bool irreversible = false;
};

class Position {
 public:
  Position();
  explicit Position(std::string_view fen);

  void clear();
  bool set_fen(std::string_view fen);
  std::string get_fen() const;

  // Apply UCI moves from the current position (builds repetition history).
  bool apply_uci(std::string_view uci);
  bool apply_uci_list(const std::vector<std::string>& ucis);

  Position copy() const { return *this; }
  // Fixed-size board state only. Legal filtering does not need repetition history.
  Position copy_without_history() const;

  Color side_to_move() const { return side_; }
  int halfmove_clock() const { return halfmove_; }
  int fullmove_number() const { return fullmove_; }
  Square ep_square() const { return ep_square_; }
  std::uint8_t castling_rights() const { return castling_; }

  Bitboard pieces(PieceType pt) const { return type_bb_[pt]; }
  Bitboard pieces(Color c) const { return color_bb_[c]; }
  Bitboard pieces(Color c, PieceType pt) const { return type_bb_[pt] & color_bb_[c]; }
  Bitboard occupied() const { return occupied_; }
  Piece piece_on(Square s) const { return board_[s]; }
  Square king_square(Color c) const { return king_sq_[c]; }

  void put_piece(Piece p, Square s);
  void remove_piece(Square s);
  void move_piece(Square from, Square to);

  void make_move(Move m);
  // Legality probes only: updates board state and remains unmakeable, but skips
  // repetition bookkeeping and Zobrist work that the probe cannot observe.
  void make_move_transient(Move m);
  void unmake_move();

  bool in_check() const;
  bool is_attacked(Square s, Color by) const;
  Bitboard attackers_to(Square s, Bitboard occ) const;

  // Legal moves into `out` (cleared first). Returns count.
  int legal_moves(std::vector<Move>& out) const;
  int legal_moves(Move* out, int max_out) const;

  bool is_legal(Move m) const;
  Move parse_uci(std::string_view uci) const;

  // Repetition: matches python-chess Board.is_repetition(count).
  bool is_repetition(int count) const;
  int repetition_count() const;  // occurrences of current position (>=1)

  bool has_insufficient_material() const;
  bool has_insufficient_material(Color c) const;

  bool can_claim_fifty_moves() const;
  bool can_claim_threefold_repetition() const;

  // Matches python-chess Board.outcome(claim_draw=...).
  Outcome outcome(bool claim_draw) const;

  // Key matching python-chess _transposition_key components (Zobrist).
  std::uint64_t transposition_key() const { return key_; }

  bool has_kingside_castling(Color c) const {
    return castling_ & (c == WHITE ? WHITE_OO : BLACK_OO);
  }
  bool has_queenside_castling(Color c) const {
    return castling_ & (c == WHITE ? WHITE_OOO : BLACK_OOO);
  }

 private:
  struct NoInitTag {};
  explicit Position(NoInitTag) {}

  friend void init_attack_tables();
  friend class MoveGen;

  void make_move_impl(Move m, bool update_key_and_history);
  void update_key_ep();
  bool has_legal_en_passant() const;
  bool is_zeroing(Move m) const;
  bool reduces_castling_rights(Move m) const;
  bool is_irreversible(Move m) const;
  void set_castling_from_fen_char(char c);
  char fen_piece_char(Piece p) const;

  std::array<Piece, 64> board_{};
  std::array<Bitboard, 6> type_bb_{};
  std::array<Bitboard, 2> color_bb_{};
  Bitboard occupied_ = 0;
  Color side_ = WHITE;
  std::uint8_t castling_ = 0;
  Square ep_square_ = NO_SQUARE;
  int halfmove_ = 0;
  int fullmove_ = 1;
  std::array<Square, 2> king_sq_{NO_SQUARE, NO_SQUARE};
  std::uint64_t key_ = 0;
  bool ep_keyed_ = false;
  std::vector<Undo> history_;
};

void init_zobrist();

}  // namespace immortalite
