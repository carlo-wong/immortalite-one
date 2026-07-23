#pragma once

#include "board.hpp"
#include "types.hpp"

#include <utility>
#include <vector>

namespace immortalite {

// Fill out[20*8*8] row-major planes[p][rank][file] as float32.
void fill_planes(const Position& pos, float* out);

int move_to_index(const Position& pos, Move m);
// Inverse; returns NULL_MOVE if decoded move is illegal in pos.
Move index_to_move(const Position& pos, int index);

// Map legal move index -> move.
std::vector<std::pair<int, Move>> legal_move_indices(const Position& pos);

}  // namespace immortalite
