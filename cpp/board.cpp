#include "board.hpp"

#include "movegen.hpp"

#include <cctype>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace immortalite {
namespace {

std::uint64_t Z_PIECE[12][64];
std::uint64_t Z_CASTLING[16];
std::uint64_t Z_EP[8];
std::uint64_t Z_SIDE;
bool zobrist_ready = false;

void ensure_zobrist() {
  if (zobrist_ready) return;
  std::mt19937_64 rng(0xC0FFEEULL);
  for (int p = 0; p < 12; ++p)
    for (int s = 0; s < 64; ++s) Z_PIECE[p][s] = rng();
  for (int i = 0; i < 16; ++i) Z_CASTLING[i] = rng();
  for (int i = 0; i < 8; ++i) Z_EP[i] = rng();
  Z_SIDE = rng();
  zobrist_ready = true;
}

int piece_z_index(Piece p) { return static_cast<int>(p) - 1; }

}  // namespace

void init_zobrist() { ensure_zobrist(); }

Position::Position() { set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

Position::Position(std::string_view fen) {
  if (!set_fen(fen)) {
    throw std::invalid_argument("invalid FEN");
  }
}

void Position::clear() {
  board_.fill(NO_PIECE);
  type_bb_.fill(0);
  color_bb_.fill(0);
  occupied_ = 0;
  side_ = WHITE;
  castling_ = 0;
  ep_square_ = NO_SQUARE;
  halfmove_ = 0;
  fullmove_ = 1;
  king_sq_ = {NO_SQUARE, NO_SQUARE};
  key_ = 0;
  history_.clear();
}

void Position::put_piece(Piece p, Square s) {
  board_[s] = p;
  Bitboard b = bit(s);
  type_bb_[type_of(p)] |= b;
  color_bb_[color_of(p)] |= b;
  occupied_ |= b;
  if (type_of(p) == KING) king_sq_[color_of(p)] = s;
}

void Position::remove_piece(Square s) {
  Piece p = board_[s];
  if (p == NO_PIECE) return;
  Bitboard b = bit(s);
  type_bb_[type_of(p)] &= ~b;
  color_bb_[color_of(p)] &= ~b;
  occupied_ &= ~b;
  board_[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
  Piece p = board_[from];
  Bitboard fb = bit(from), tb = bit(to);
  board_[from] = NO_PIECE;
  board_[to] = p;
  type_bb_[type_of(p)] ^= fb | tb;
  color_bb_[color_of(p)] ^= fb | tb;
  occupied_ ^= fb | tb;
  if (type_of(p) == KING) king_sq_[color_of(p)] = to;
}

void Position::set_castling_from_fen_char(char c) {
  switch (c) {
    case 'K': castling_ |= WHITE_OO; break;
    case 'Q': castling_ |= WHITE_OOO; break;
    case 'k': castling_ |= BLACK_OO; break;
    case 'q': castling_ |= BLACK_OOO; break;
    default: break;
  }
}

bool Position::set_fen(std::string_view fen) {
  ensure_zobrist();
  init_attack_tables();
  clear();

  std::string s(fen);
  std::istringstream ss(s);
  std::string board, stm, castle, ep;
  int half = 0, full = 1;
  if (!(ss >> board >> stm >> castle >> ep)) return false;
  if (!(ss >> half)) half = 0;
  if (!(ss >> full)) full = 1;

  Square sq = 56;  // a8
  for (char c : board) {
    if (c == '/') {
      sq -= 16;
      continue;
    }
    if (c >= '1' && c <= '8') {
      sq += c - '0';
      continue;
    }
    Piece p = NO_PIECE;
    switch (c) {
      case 'P': p = W_PAWN; break;
      case 'N': p = W_KNIGHT; break;
      case 'B': p = W_BISHOP; break;
      case 'R': p = W_ROOK; break;
      case 'Q': p = W_QUEEN; break;
      case 'K': p = W_KING; break;
      case 'p': p = B_PAWN; break;
      case 'n': p = B_KNIGHT; break;
      case 'b': p = B_BISHOP; break;
      case 'r': p = B_ROOK; break;
      case 'q': p = B_QUEEN; break;
      case 'k': p = B_KING; break;
      default: return false;
    }
    if (sq < 0 || sq > 63) return false;
    put_piece(p, sq);
    ++sq;
  }

  if (stm == "w") side_ = WHITE;
  else if (stm == "b") side_ = BLACK;
  else return false;

  castling_ = 0;
  if (castle != "-") {
    for (char c : castle) set_castling_from_fen_char(c);
  }

  ep_square_ = NO_SQUARE;
  if (ep != "-") {
    Square e = square_from_string(ep);
    if (e == NO_SQUARE) return false;
    ep_square_ = e;
  }

  halfmove_ = half;
  fullmove_ = full;

  // Compute key
  key_ = 0;
  for (Square s = 0; s < 64; ++s) {
    Piece p = board_[s];
    if (p != NO_PIECE) key_ ^= Z_PIECE[piece_z_index(p)][s];
  }
  key_ ^= Z_CASTLING[castling_ & 15];
  if (side_ == BLACK) key_ ^= Z_SIDE;
  update_key_ep();
  return true;
}

void Position::update_key_ep() {
  // Key already has no EP component; add only if legal EP exists (python-chess).
  // Caller must not have EP xor applied. We rebuild EP part from scratch by
  // ensuring we track whether EP is in the key via recomputation of full key EP.
  // Simpler: strip any previous EP by recomputing from pieces each time we need
  // accuracy — here we only call after set_fen / make / unmake when ep changes.
  // We store key without EP, then XOR ep file if legal.
  // Actually our key_ may already contain EP from previous state. Rebuild cleanly:
  std::uint64_t k = 0;
  for (Square s = 0; s < 64; ++s) {
    Piece p = board_[s];
    if (p != NO_PIECE) k ^= Z_PIECE[piece_z_index(p)][s];
  }
  k ^= Z_CASTLING[castling_ & 15];
  if (side_ == BLACK) k ^= Z_SIDE;
  if (ep_square_ != NO_SQUARE && has_legal_en_passant()) {
    k ^= Z_EP[file_of(ep_square_)];
  }
  key_ = k;
}

char Position::fen_piece_char(Piece p) const {
  static constexpr char kChars[] = ".PNBRQKpnbrqk";
  return kChars[static_cast<int>(p)];
}

bool Position::apply_uci(std::string_view uci) {
  Move m = parse_uci(uci);
  if (m.null()) return false;
  make_move(m);
  return true;
}

bool Position::apply_uci_list(const std::vector<std::string>& ucis) {
  for (const auto& u : ucis) {
    if (!apply_uci(u)) return false;
  }
  return true;
}

std::string Position::get_fen() const {
  std::string out;
  for (int r = 7; r >= 0; --r) {
    int empty = 0;
    for (int f = 0; f < 8; ++f) {
      Piece p = board_[square_of(f, r)];
      if (p == NO_PIECE) {
        ++empty;
      } else {
        if (empty) {
          out.push_back(static_cast<char>('0' + empty));
          empty = 0;
        }
        out.push_back(fen_piece_char(p));
      }
    }
    if (empty) out.push_back(static_cast<char>('0' + empty));
    if (r) out.push_back('/');
  }
  out.push_back(' ');
  out.push_back(side_ == WHITE ? 'w' : 'b');
  out.push_back(' ');
  std::string castle;
  if (castling_ & WHITE_OO) castle.push_back('K');
  if (castling_ & WHITE_OOO) castle.push_back('Q');
  if (castling_ & BLACK_OO) castle.push_back('k');
  if (castling_ & BLACK_OOO) castle.push_back('q');
  out += castle.empty() ? "-" : castle;
  out.push_back(' ');
  out += ep_square_ == NO_SQUARE ? "-" : square_to_string(ep_square_);
  out.push_back(' ');
  out += std::to_string(halfmove_);
  out.push_back(' ');
  out += std::to_string(fullmove_);
  return out;
}

bool Position::is_attacked(Square s, Color by) const {
  Bitboard occ = occupied_;
  if (pawn_attacks_bb(~by, s) & pieces(by, PAWN)) return true;
  if (knight_attacks_bb(s) & pieces(by, KNIGHT)) return true;
  if (king_attacks_bb(s) & pieces(by, KING)) return true;
  if (bishop_attacks_bb(s, occ) & (pieces(by, BISHOP) | pieces(by, QUEEN))) return true;
  if (rook_attacks_bb(s, occ) & (pieces(by, ROOK) | pieces(by, QUEEN))) return true;
  return false;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
  Bitboard attackers = 0;
  attackers |= pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN);
  attackers |= pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN);
  attackers |= knight_attacks_bb(s) & type_bb_[KNIGHT];
  attackers |= king_attacks_bb(s) & type_bb_[KING];
  attackers |= bishop_attacks_bb(s, occ) & (type_bb_[BISHOP] | type_bb_[QUEEN]);
  attackers |= rook_attacks_bb(s, occ) & (type_bb_[ROOK] | type_bb_[QUEEN]);
  return attackers & occupied_;
}

bool Position::in_check() const {
  return is_attacked(king_sq_[side_], ~side_);
}

bool Position::is_zeroing(Move m) const {
  if (m.flags() & (MF_CAPTURE | MF_EP)) return true;
  Piece p = board_[m.from()];
  return p != NO_PIECE && type_of(p) == PAWN;
}

bool Position::reduces_castling_rights(Move m) const {
  std::uint8_t before = castling_;
  if (!before) return false;
  std::uint8_t after = before;
  Square from = m.from(), to = m.to();
  Piece p = board_[from];
  if (type_of(p) == KING) {
    if (color_of(p) == WHITE) after &= static_cast<std::uint8_t>(~(WHITE_OO | WHITE_OOO));
    else after &= static_cast<std::uint8_t>(~(BLACK_OO | BLACK_OOO));
  }
  if (from == 0 || to == 0) after &= static_cast<std::uint8_t>(~WHITE_OOO);
  if (from == 7 || to == 7) after &= static_cast<std::uint8_t>(~WHITE_OO);
  if (from == 56 || to == 56) after &= static_cast<std::uint8_t>(~BLACK_OOO);
  if (from == 63 || to == 63) after &= static_cast<std::uint8_t>(~BLACK_OO);
  return after != before;
}

bool Position::is_irreversible(Move m) const {
  // Match python-chess: zeroing OR reduces castling OR has_legal_en_passant (current).
  return is_zeroing(m) || reduces_castling_rights(m) || has_legal_en_passant();
}

bool Position::has_legal_en_passant() const {
  if (ep_square_ == NO_SQUARE) return false;
  Move buf[8];
  int n = generate_pseudo_ep(*this, buf, 8);
  Position tmp = *this;
  for (int i = 0; i < n; ++i) {
    Color us = tmp.side_to_move();
    tmp.make_move(buf[i]);
    bool ok = !tmp.is_attacked(tmp.king_square(us), tmp.side_to_move());
    tmp.unmake_move();
    if (ok) return true;
  }
  return false;
}

void Position::make_move(Move m) {
  ensure_zobrist();
  Undo u;
  u.move = m;
  u.castling = castling_;
  u.ep_square = ep_square_;
  u.halfmove = halfmove_;
  u.key_before = key_;
  u.irreversible = is_irreversible(m);
  u.captured = NO_PIECE;

  const Color us = side_;
  const Color them = ~us;
  const Square from = m.from();
  const Square to = m.to();
  Piece piece = board_[from];

  // Clear EP from key path by full rebuild at end; track captured.
  if (m.flags() & MF_EP) {
    Square cap_sq = to + (us == WHITE ? -8 : 8);
    u.captured = board_[cap_sq];
    remove_piece(cap_sq);
  } else if (board_[to] != NO_PIECE) {
    u.captured = board_[to];
    remove_piece(to);
  }

  // Castling rook move
  if (m.flags() & MF_CASTLE) {
    if (to == 6) {  // white O-O
      move_piece(7, 5);
    } else if (to == 2) {
      move_piece(0, 3);
    } else if (to == 62) {
      move_piece(63, 61);
    } else if (to == 58) {
      move_piece(56, 59);
    }
  }

  move_piece(from, to);

  if (m.flags() & MF_PROMOTION) {
    remove_piece(to);
    put_piece(make_piece(us, m.promotion()), to);
  }

  // Update castling rights
  if (type_of(piece) == KING) {
    if (us == WHITE) castling_ &= static_cast<std::uint8_t>(~(WHITE_OO | WHITE_OOO));
    else castling_ &= static_cast<std::uint8_t>(~(BLACK_OO | BLACK_OOO));
  }
  auto clear_rook = [&](Square s) {
    if (s == 0) castling_ &= static_cast<std::uint8_t>(~WHITE_OOO);
    if (s == 7) castling_ &= static_cast<std::uint8_t>(~WHITE_OO);
    if (s == 56) castling_ &= static_cast<std::uint8_t>(~BLACK_OOO);
    if (s == 63) castling_ &= static_cast<std::uint8_t>(~BLACK_OO);
  };
  clear_rook(from);
  clear_rook(to);

  // Halfmove (use pre-move piece / flags; board_[from] is already empty)
  if ((m.flags() & (MF_CAPTURE | MF_EP)) || type_of(piece) == PAWN) halfmove_ = 0;
  else ++halfmove_;

  // EP square
  ep_square_ = NO_SQUARE;
  if (m.flags() & MF_DOUBLE_PAWN) {
    ep_square_ = static_cast<Square>((from + to) / 2);
  }

  if (us == BLACK) ++fullmove_;
  side_ = them;

  history_.push_back(u);
  update_key_ep();
}

void Position::unmake_move() {
  Undo u = history_.back();
  history_.pop_back();
  Move m = u.move;
  const Color us = ~side_;  // side that made the move
  const Square from = m.from();
  const Square to = m.to();

  side_ = us;
  if (us == BLACK) --fullmove_;
  castling_ = u.castling;
  ep_square_ = u.ep_square;
  halfmove_ = u.halfmove;

  // Undo promotion
  if (m.flags() & MF_PROMOTION) {
    remove_piece(to);
    put_piece(make_piece(us, PAWN), to);
  }

  move_piece(to, from);

  if (m.flags() & MF_CASTLE) {
    if (to == 6) move_piece(5, 7);
    else if (to == 2) move_piece(3, 0);
    else if (to == 62) move_piece(61, 63);
    else if (to == 58) move_piece(59, 56);
  }

  if (m.flags() & MF_EP) {
    Square cap_sq = to + (us == WHITE ? -8 : 8);
    put_piece(u.captured, cap_sq);
  } else if (u.captured != NO_PIECE) {
    put_piece(u.captured, to);
  }

  key_ = u.key_before;
}

int Position::legal_moves(std::vector<Move>& out) const {
  return generate_legal_moves(*this, out);
}

int Position::legal_moves(Move* out, int max_out) const {
  return generate_legal_moves(*this, out, max_out);
}

bool Position::is_legal(Move m) const {
  Move buf[256];
  int n = generate_legal_moves(*this, buf, 256);
  for (int i = 0; i < n; ++i)
    if (buf[i] == m) return true;
  return false;
}

Move Position::parse_uci(std::string_view uci) const {
  if (uci.size() < 4) return NULL_MOVE;
  Square from = square_from_string(uci.substr(0, 2));
  Square to = square_from_string(uci.substr(2, 2));
  if (from == NO_SQUARE || to == NO_SQUARE) return NULL_MOVE;
  PieceType promo = NO_PIECE_TYPE;
  if (uci.size() >= 5) {
    switch (uci[4]) {
      case 'n': promo = KNIGHT; break;
      case 'b': promo = BISHOP; break;
      case 'r': promo = ROOK; break;
      case 'q': promo = QUEEN; break;
      default: return NULL_MOVE;
    }
  }
  Move buf[256];
  int n = generate_legal_moves(*this, buf, 256);
  for (int i = 0; i < n; ++i) {
    if (buf[i].from() == from && buf[i].to() == to) {
      if (promo == NO_PIECE_TYPE) {
        if (buf[i].promotion() == NO_PIECE_TYPE || buf[i].promotion() == QUEEN) return buf[i];
      } else if (buf[i].promotion() == promo) {
        return buf[i];
      }
    }
  }
  return NULL_MOVE;
}

bool Position::is_repetition(int count) const {
  if (count <= 1) return true;
  // Match python-chess: current position counts once; each prior match decrements.
  std::uint64_t key = transposition_key();
  int need = count;
  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
    if (need <= 1) return true;
    const Undo& u = history_[i];
    if (u.irreversible) break;
    if (u.key_before == key) {
      --need;
      if (need <= 1) return true;
    }
  }
  return need <= 1;
}

int Position::repetition_count() const {
  int c = 1;
  std::uint64_t key = key_;
  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
    const Undo& u = history_[i];
    if (u.irreversible) break;
    if (u.key_before == key) ++c;
  }
  return c;
}

bool Position::has_insufficient_material(Color color) const {
  // Match python-chess Board.has_insufficient_material.
  if (pieces(color) & (type_bb_[PAWN] | type_bb_[ROOK] | type_bb_[QUEEN])) return false;

  if (pieces(color) & type_bb_[KNIGHT]) {
    return popcount(pieces(color)) <= 2 &&
           !(pieces(~color) & ~type_bb_[KING] & ~type_bb_[QUEEN]);
  }

  if (pieces(color) & type_bb_[BISHOP]) {
    constexpr Bitboard DARK = 0xAA55AA55AA55AA55ULL;
    Bitboard bishops = type_bb_[BISHOP];
    bool same_color = !(bishops & DARK) || !(bishops & ~DARK);
    return same_color && !type_bb_[PAWN] && !type_bb_[KNIGHT];
  }

  return true;
}

bool Position::has_insufficient_material() const {
  return has_insufficient_material(WHITE) && has_insufficient_material(BLACK);
}

bool Position::can_claim_fifty_moves() const {
  if (halfmove_ >= 100) {
    // Prefer other ends, but claim check itself is just the clock when used from outcome.
    return true;
  }
  if (halfmove_ >= 99) {
    Move buf[256];
    int n = generate_legal_moves(*this, buf, 256);
    Position tmp = *this;
    for (int i = 0; i < n; ++i) {
      if (!tmp.is_zeroing(buf[i])) {
        tmp.make_move(buf[i]);
        bool ok = tmp.halfmove_clock() >= 100;
        tmp.unmake_move();
        if (ok) return true;
      }
    }
  }
  return false;
}

bool Position::can_claim_threefold_repetition() const {
  // Match python-chess Board.can_claim_threefold_repetition.
  std::uint64_t cur = transposition_key();
  std::vector<std::uint64_t> keys;
  keys.push_back(cur);
  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
    const Undo& u = history_[i];
    if (u.irreversible) break;
    keys.push_back(u.key_before);
  }

  int count = 0;
  for (auto k : keys)
    if (k == cur) ++count;
  if (count >= 3) return true;

  Move buf[256];
  int n = generate_legal_moves(*this, buf, 256);
  Position tmp = *this;
  for (int i = 0; i < n; ++i) {
    tmp.make_move(buf[i]);
    std::uint64_t k = tmp.transposition_key();
    int c = 0;
    for (auto x : keys)
      if (x == k) ++c;
    tmp.unmake_move();
    if (c >= 2) return true;
  }
  return false;
}

Outcome Position::outcome(bool claim_draw) const {
  Outcome o;
  Move buf[256];
  int n = generate_legal_moves(*this, buf, 256);
  bool check = in_check();

  if (n == 0) {
    if (check) {
      o.termination = Termination::Checkmate;
      o.winner = static_cast<int>(~side_);
    } else {
      o.termination = Termination::Stalemate;
    }
    return o;
  }

  if (has_insufficient_material()) {
    o.termination = Termination::InsufficientMaterial;
    return o;
  }

  if (halfmove_ >= 150) {
    o.termination = Termination::SeventyfiveMoves;
    return o;
  }
  if (is_repetition(5)) {
    o.termination = Termination::FivefoldRepetition;
    return o;
  }

  if (claim_draw) {
    if (can_claim_fifty_moves()) {
      o.termination = Termination::FiftyMoves;
      return o;
    }
    if (can_claim_threefold_repetition()) {
      o.termination = Termination::ThreefoldRepetition;
      return o;
    }
  }
  return o;
}

}  // namespace immortalite
