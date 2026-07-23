#pragma once

#include "types.hpp"

#include <vector>

namespace immortalite {

class Position;

// Attack / occupancy helpers used by board and move generation.
void init_attack_tables();

Bitboard pawn_attacks_bb(Color c, Square s);
Bitboard knight_attacks_bb(Square s);
Bitboard king_attacks_bb(Square s);
Bitboard bishop_attacks_bb(Square s, Bitboard occ);
Bitboard rook_attacks_bb(Square s, Bitboard occ);
Bitboard queen_attacks_bb(Square s, Bitboard occ);

// Generate all legal moves for `pos` into `out`.
int generate_legal_moves(const Position& pos, std::vector<Move>& out);
int generate_legal_moves(const Position& pos, Move* out, int max_out);

// Pseudo-legal EP captures only (for has_legal_en_passant).
int generate_pseudo_ep(const Position& pos, Move* out, int max_out);

}  // namespace immortalite
