#include "mcts.hpp"

#include "encoding.hpp"
#include "movegen.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace immortalite {
namespace {

std::vector<float> softmax_take(const float* logits, const std::vector<int>& idxs) {
  std::vector<float> x;
  x.reserve(idxs.size());
  float maxv = -std::numeric_limits<float>::infinity();
  for (int i : idxs) {
    float v = logits[i];
    x.push_back(v);
    if (v > maxv) maxv = v;
  }
  float sum = 0.0f;
  for (float& v : x) {
    v = std::exp(v - maxv);
    sum += v;
  }
  if (sum > 0) {
    for (float& v : x) v /= sum;
  }
  return x;
}

std::vector<double> softmax_d(const std::vector<double>& logits) {
  double maxv = *std::max_element(logits.begin(), logits.end());
  std::vector<double> e(logits.size());
  double sum = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    e[i] = std::exp(logits[i] - maxv);
    sum += e[i];
  }
  if (sum > 0) {
    for (double& v : e) v /= sum;
  }
  return e;
}

}  // namespace

std::vector<double> MctsResult::improved_policy(const MctsConfig& cfg) const {
  if (visits.empty()) return {};
  double max_n = *std::max_element(visits.begin(), visits.end());
  double sigma = (cfg.gumbel_c_visit + max_n) * cfg.gumbel_c_scale;
  double qmin = *std::min_element(q_values.begin(), q_values.end());
  double qmax = *std::max_element(q_values.begin(), q_values.end());
  double span = qmax - qmin;
  std::vector<double> logits(visits.size());
  for (size_t i = 0; i < visits.size(); ++i) {
    double qn = span > 0 ? (q_values[i] - qmin) / span : 0.0;
    double cp = std::max(clean_priors[i], 1e-9);
    logits[i] = std::log(cp) + sigma * qn;
  }
  return softmax_d(logits);
}

void MctsSession::begin_search_from_root() {
  path_board_ = root_board_;
  root_turn_ = root_board_.side_to_move();
  root_ = std::make_unique<Node>();
  clear_pending_legal();
  pending_leaves_.clear();

  auto [term, tval] = terminal_eval(*root_, root_board_);
  if (term) {
    root_value_ = tval;
    collect_result();
    done_ = true;
    need_root_eval_ = false;
    waiting_for_eval_ = false;
    return;
  }
  need_root_eval_ = true;
  waiting_for_eval_ = true;
  cache_pending_legal_indices(root_board_);
}

MctsSession::MctsSession(std::string_view fen, int simulations, const MctsConfig& cfg,
                         bool add_noise, const std::vector<std::string>& moves_from_fen)
    : cfg_(cfg),
      sims_target_(simulations > 0 ? simulations : cfg.simulations),
      add_noise_(add_noise) {
  init_attack_tables();
  init_zobrist();
  if (!root_board_.set_fen(fen)) {
    throw std::invalid_argument("MctsSession: invalid FEN");
  }
  for (const auto& uci : moves_from_fen) {
    Move m = root_board_.parse_uci(uci);
    if (m.null()) {
      throw std::invalid_argument("MctsSession: illegal history move " + uci);
    }
    root_board_.make_move(m);
  }
  begin_search_from_root();
}

MctsSession::MctsSession(const Position& root, int simulations, const MctsConfig& cfg,
                         bool add_noise)
    : cfg_(cfg),
      sims_target_(simulations > 0 ? simulations : cfg.simulations),
      add_noise_(add_noise),
      root_board_(root) {
  init_attack_tables();
  init_zobrist();
  begin_search_from_root();
}

std::pair<bool, float> MctsSession::terminal_eval(Node& node, const Position& board) {
  if (node.terminal_checked) return {node.is_terminal, node.terminal_value};
  Outcome o = board.outcome(cfg_.claim_draw);
  node.is_terminal = o.is_terminal();
  if (node.is_terminal) {
    node.terminal_value = value_from_outcome(o, board);
  } else {
    node.terminal_value = 0.0f;
  }
  node.terminal_checked = true;
  return {node.is_terminal, node.terminal_value};
}

float MctsSession::value_from_outcome(const Outcome& outcome, const Position& board) const {
  if (outcome.termination == Termination::Checkmate) {
    return -1.0f;  // side to move has been mated
  }
  float contempt = cfg_.draw_contempt;
  return board.side_to_move() == root_turn_ ? -contempt : contempt;
}

std::pair<int, MctsSession::Node*> MctsSession::select_child(Node& node) const {
  float c_puct = cfg_.c_puct;
  // N_virtual is always 0 when virtual loss is off → identical to prior PUCT.
  float sqrt_n = std::sqrt(static_cast<float>(node.N + node.N_virtual));
  float best = -std::numeric_limits<float>::infinity();
  int best_idx = -1;
  Node* best_child = nullptr;
  for (auto& [idx, child] : node.children) {
    float q = static_cast<float>(-child->Q());
    const float child_n = static_cast<float>(child->N + child->N_virtual);
    float u = c_puct * child->prior * sqrt_n / (1.0f + child_n);
    float score = q + u;
    if (score > best) {
      best = score;
      best_idx = idx;
      best_child = child.get();
    }
  }
  return {best_idx, best_child};
}

void MctsSession::expand_from_eval(Node& node, const Position& board, const float* logits,
                                   float value) {
  const std::vector<std::pair<int, Move>>* mapping_ptr = &pending_legal_mapping_;
  std::vector<std::pair<int, Move>> generated;
  if (pending_legal_mapping_.empty()) {
    generated = legal_move_indices(board);
    mapping_ptr = &generated;
  }
  const auto& mapping = *mapping_ptr;
  if (mapping.empty()) {
    (void)value;
    return;
  }
  ++nodes_expanded_;
  std::vector<int> idxs;
  idxs.reserve(mapping.size());
  for (const auto& [idx, mv] : mapping) {
    (void)mv;
    idxs.push_back(idx);
  }
  expand_with_priors(node, mapping, softmax_take(logits, idxs));
}

void MctsSession::expand_from_legal_eval(Node& node, const Position& board,
                                         const float* legal_logits, float value) {
  (void)board;
  if (pending_legal_mapping_.empty()) {
    throw std::logic_error("expand_from_legal_eval without cached legal mapping");
  }
  const auto& mapping = pending_legal_mapping_;
  if (mapping.size() != pending_legal_indices_.size()) {
    throw std::logic_error("pending legal mapping/index size mismatch");
  }
  for (size_t i = 0; i < mapping.size(); ++i) {
    if (mapping[i].first != pending_legal_indices_[i]) {
      throw std::logic_error("pending legal move indices no longer match leaf");
    }
  }
  if (mapping.empty()) {
    (void)value;
    return;
  }
  ++nodes_expanded_;
  std::vector<float> priors(legal_logits, legal_logits + mapping.size());
  const float maxv = *std::max_element(priors.begin(), priors.end());
  float sum = 0.0f;
  for (float& prior : priors) {
    prior = std::exp(prior - maxv);
    sum += prior;
  }
  if (sum > 0.0f) {
    for (float& prior : priors) prior /= sum;
  }
  expand_with_priors(node, mapping, priors);
}

void MctsSession::expand_with_priors(
    Node& node, const std::vector<std::pair<int, Move>>& mapping,
    const std::vector<float>& priors) {
  if (mapping.size() != priors.size()) {
    throw std::logic_error("policy priors do not match legal moves");
  }
  node.children.reserve(mapping.size());
  for (size_t i = 0; i < mapping.size(); ++i) {
    auto child = std::make_unique<Node>();
    child->prior = priors[i];
    child->move = mapping[i].second;
    node.children.emplace_back(mapping[i].first, std::move(child));
  }
}

void MctsSession::cache_pending_legal_indices(const Position& board) {
  pending_legal_mapping_ = legal_move_indices(board);
  pending_legal_indices_.clear();
  pending_legal_indices_.reserve(pending_legal_mapping_.size());
  for (const auto& [idx, move] : pending_legal_mapping_) {
    (void)move;
    pending_legal_indices_.push_back(idx);
  }
}

void MctsSession::clear_pending_legal() {
  pending_legal_indices_.clear();
  pending_legal_mapping_.clear();
}

void MctsSession::add_dirichlet_noise(Node& root) {
  if (root.children.empty()) return;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
  std::vector<double> noise(root.children.size());
  double sum = 0;
  for (size_t i = 0; i < root.children.size(); ++i) {
    noise[i] = gamma(gen);
    if (noise[i] <= 0) noise[i] = 1e-20;
    sum += noise[i];
  }
  float eps = cfg_.dirichlet_epsilon;
  for (size_t i = 0; i < root.children.size(); ++i) {
    float n = static_cast<float>(noise[i] / sum);
    auto& child = root.children[i].second;
    child->prior = (1.0f - eps) * child->prior + eps * n;
  }
}

void MctsSession::apply_virtual_loss_on_path() {
  const int vl = cfg_.virtual_loss;
  for (Node* n : path_) n->N_virtual += vl;
}

void MctsSession::remove_virtual_loss_on_path(const std::vector<Node*>& path) {
  const int vl = cfg_.virtual_loss;
  for (Node* n : path) n->N_virtual -= vl;
}

void MctsSession::backup(float value) {
  for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
    Node* n = *it;
    n->N += 1;
    n->W += value;
    value = -value;
  }
  for (int i = 0; i < path_depth_; ++i) path_board_.unmake_move();
  path_.clear();
  path_depth_ = 0;
  assert(path_board_.transposition_key() == root_board_.transposition_key());
}

void MctsSession::backup_pending(PendingLeaf& leaf, float value) {
  remove_virtual_loss_on_path(leaf.path);
  for (auto it = leaf.path.rbegin(); it != leaf.path.rend(); ++it) {
    Node* n = *it;
    n->N += 1;
    n->W += value;
    value = -value;
  }
}

void MctsSession::select_to_leaf() {
  // Every completed path is unmade by backup(), so path_board_ is already at
  // the root. Reassigning here deep-copied the full game history per simulation.
  assert(path_board_.transposition_key() == root_board_.transposition_key());
  path_.clear();
  path_.push_back(root_.get());
  path_depth_ = 0;

  Node* node = root_.get();
  bool is_terminal = false;
  float terminal_value = 0.0f;

  while (node->expanded() && !is_terminal) {
    auto [idx, child] = select_child(*node);
    if (child == nullptr) break;
    Move move = child->move;
    if (move.null()) {
      move = index_to_move(path_board_, idx);
    }
    if (move.null()) break;
    path_board_.make_move(move);
    ++path_depth_;
    node = child;
    path_.push_back(node);
    auto te = terminal_eval(*node, path_board_);
    is_terminal = te.first;
    terminal_value = te.second;
  }

  if (is_terminal) {
    backup(terminal_value);
    ++sims_done_;
    waiting_for_eval_ = false;
  } else {
    waiting_for_eval_ = true;
    cache_pending_legal_indices(path_board_);
  }
}

bool MctsSession::select_one_leaf_with_virtual_loss() {
  assert(path_board_.transposition_key() == root_board_.transposition_key());
  path_.clear();
  path_.push_back(root_.get());
  path_depth_ = 0;

  Node* node = root_.get();
  bool is_terminal = false;
  float terminal_value = 0.0f;

  while (node->expanded() && !is_terminal) {
    auto [idx, child] = select_child(*node);
    if (child == nullptr) break;
    Move move = child->move;
    if (move.null()) {
      move = index_to_move(path_board_, idx);
    }
    if (move.null()) break;
    path_board_.make_move(move);
    ++path_depth_;
    node = child;
    path_.push_back(node);
    auto te = terminal_eval(*node, path_board_);
    is_terminal = te.first;
    terminal_value = te.second;
  }

  if (is_terminal) {
    backup(terminal_value);
    ++sims_done_;
    return false;
  }

  // Unexpanded leaf already in-flight (virtual loss still outstanding).
  if (node->N_virtual > 0) {
    for (int i = 0; i < path_depth_; ++i) path_board_.unmake_move();
    path_.clear();
    path_depth_ = 0;
    return false;
  }

  cache_pending_legal_indices(path_board_);
  PendingLeaf leaf;
  leaf.path = path_;
  leaf.path_depth = path_depth_;
  leaf.board = path_board_;
  leaf.legal_indices = pending_legal_indices_;
  leaf.legal_mapping = pending_legal_mapping_;
  apply_virtual_loss_on_path();
  for (int i = 0; i < path_depth_; ++i) path_board_.unmake_move();
  path_.clear();
  path_depth_ = 0;
  clear_pending_legal();
  pending_leaves_.push_back(std::move(leaf));
  return true;
}

void MctsSession::select_leaves(int max_k) {
  if (max_k <= 0) return;
  while (static_cast<int>(pending_leaves_.size()) < max_k && sims_done_ < sims_target_) {
    const int sims_before = sims_done_;
    if (!select_one_leaf_with_virtual_loss()) {
      if (sims_done_ > sims_before) continue;  // terminal backup; try another leaf
      break;                                   // in-flight collision — stop this wave
    }
  }
  waiting_for_eval_ = !pending_leaves_.empty();
  if (pending_leaves_.empty() && sims_done_ >= sims_target_) {
    collect_result();
    done_ = true;
    waiting_for_eval_ = false;
  }
}

void MctsSession::advance_after_expand() {
  if (multi_leaf_mode()) {
    // Defer multi-leaf selection to positions_needing_eval (honors caller max_n).
    if (sims_done_ >= sims_target_) {
      collect_result();
      done_ = true;
      waiting_for_eval_ = false;
    } else {
      waiting_for_eval_ = false;
    }
    return;
  }
  // Continue simulations until we need another eval or finish.
  while (sims_done_ < sims_target_) {
    select_to_leaf();
    if (waiting_for_eval_) return;  // leaf needs NN
    // else: backed up a terminal; continue
  }
  collect_result();
  done_ = true;
  waiting_for_eval_ = false;
}

int MctsSession::pending_eval_count() const {
  if (need_root_eval_ && waiting_for_eval_) return 1;
  if (!pending_leaves_.empty()) return static_cast<int>(pending_leaves_.size());
  if (waiting_for_eval_) return 1;
  return 0;
}

const std::vector<int>& MctsSession::pending_legal_indices(int leaf) const {
  if (!pending_leaves_.empty()) {
    if (leaf < 0 || leaf >= static_cast<int>(pending_leaves_.size())) {
      throw std::out_of_range("pending legal leaf index");
    }
    return pending_leaves_[static_cast<size_t>(leaf)].legal_indices;
  }
  if (leaf != 0) throw std::out_of_range("pending legal leaf index");
  return pending_legal_indices_;
}

int MctsSession::positions_needing_eval(float* out_planes, int max_n) {
  if (done_ || max_n <= 0) return 0;

  if (!multi_leaf_mode()) {
    if (!waiting_for_eval_) return 0;
    if (pending_legal_indices_.empty()) cache_pending_legal_indices(path_board_);
    fill_planes(path_board_, out_planes);
    return 1;
  }

  // Root eval is always a single position.
  if (need_root_eval_) {
    if (!waiting_for_eval_) return 0;
    if (pending_legal_indices_.empty()) cache_pending_legal_indices(root_board_);
    fill_planes(root_board_, out_planes);
    return 1;
  }

  if (pending_leaves_.empty()) {
    const int budget = std::min({max_n, cfg_.max_leaves_per_eval, sims_target_ - sims_done_});
    if (budget <= 0) {
      if (sims_done_ >= sims_target_) {
        collect_result();
        done_ = true;
      }
      return 0;
    }
    select_leaves(budget);
  }

  if (pending_leaves_.empty()) return 0;

  const int k = std::min(max_n, static_cast<int>(pending_leaves_.size()));
  // Only expose a prefix when capacity is tight; remaining leaves stay pending
  // with virtual loss applied until a later wave applies them.
  for (int i = 0; i < k; ++i) {
    fill_planes(pending_leaves_[static_cast<size_t>(i)].board,
                out_planes + static_cast<size_t>(i) * NUM_INPUT_PLANES * 64);
  }
  waiting_for_eval_ = true;
  return k;
}

void MctsSession::apply_eval(const float* logits, const float* values, int n) {
  if (done_ || (!waiting_for_eval_ && pending_leaves_.empty())) return;
  if (n < 1 || logits == nullptr || values == nullptr) {
    throw std::invalid_argument("apply_eval requires n>=1");
  }

  if (need_root_eval_) {
    ++steps_;
    root_value_ = values[0];
    expand_from_eval(*root_, root_board_, logits, values[0]);
    clear_pending_legal();
    root_clean_priors_.clear();
    for (auto& [idx, child] : root_->children) {
      root_clean_priors_[idx] = child->prior;
    }
    if (add_noise_) add_dirichlet_noise(*root_);
    need_root_eval_ = false;
    waiting_for_eval_ = false;
    path_board_ = root_board_;
    advance_after_expand();
    return;
  }

  if (multi_leaf_mode() && !pending_leaves_.empty()) {
    // Apply first n pending leaves (n may be < pending when batch capacity truncated).
    if (n > static_cast<int>(pending_leaves_.size())) {
      throw std::invalid_argument("apply_eval leaf count mismatch");
    }
    ++steps_;
    for (int i = 0; i < n; ++i) {
      PendingLeaf& leaf = pending_leaves_[static_cast<size_t>(i)];
      pending_legal_indices_ = leaf.legal_indices;
      pending_legal_mapping_ = leaf.legal_mapping;
      expand_from_eval(*leaf.path.back(), leaf.board,
                       logits + static_cast<size_t>(i) * POLICY_SIZE, values[i]);
      clear_pending_legal();
      backup_pending(leaf, values[i]);
      ++sims_done_;
    }
    pending_leaves_.erase(pending_leaves_.begin(), pending_leaves_.begin() + n);
    waiting_for_eval_ = !pending_leaves_.empty();
    if (!pending_leaves_.empty()) return;
    if (sims_done_ >= sims_target_) {
      collect_result();
      done_ = true;
      waiting_for_eval_ = false;
      return;
    }
    advance_after_expand();
    return;
  }

  // Single-leaf path (bit-identical to pre-VL behavior).
  ++steps_;
  Node* leaf = path_.back();
  float value = values[0];
  expand_from_eval(*leaf, path_board_, logits, value);
  clear_pending_legal();
  backup(value);
  ++sims_done_;
  waiting_for_eval_ = false;

  if (sims_done_ >= sims_target_) {
    collect_result();
    done_ = true;
    return;
  }
  advance_after_expand();
}

void MctsSession::apply_eval_legal(const float* legal_logits, int legal_count, float value) {
  int offsets[2] = {0, legal_count};
  apply_eval_legal(legal_logits, offsets, &value, 1);
}

void MctsSession::apply_eval_legal(const float* legal_logits, const int* offsets,
                                   const float* values, int n) {
  if (done_ || (!waiting_for_eval_ && pending_leaves_.empty())) return;
  if (legal_logits == nullptr || offsets == nullptr || values == nullptr || n < 1) {
    throw std::invalid_argument("apply_eval_legal requires n>=1");
  }
  if (offsets[0] != 0) throw std::invalid_argument("legal offsets must start at zero");

  if (need_root_eval_) {
    if (n != 1) throw std::invalid_argument("root eval expects a single legal row");
    const int legal_count = offsets[1] - offsets[0];
    if (legal_count != static_cast<int>(pending_legal_indices_.size())) {
      throw std::invalid_argument("legal logits must match pending legal move count");
    }
    ++steps_;
    root_value_ = values[0];
    expand_from_legal_eval(*root_, root_board_, legal_logits, values[0]);
    clear_pending_legal();
    root_clean_priors_.clear();
    for (auto& [idx, child] : root_->children) {
      root_clean_priors_[idx] = child->prior;
    }
    if (add_noise_) add_dirichlet_noise(*root_);
    need_root_eval_ = false;
    waiting_for_eval_ = false;
    path_board_ = root_board_;
    advance_after_expand();
    return;
  }

  if (multi_leaf_mode() && !pending_leaves_.empty()) {
    if (n > static_cast<int>(pending_leaves_.size())) {
      throw std::invalid_argument("apply_eval_legal leaf count mismatch");
    }
    for (int i = 0; i < n; ++i) {
      const int start = offsets[i];
      const int end = offsets[i + 1];
      if (start < 0 || end < start ||
          end - start !=
              static_cast<int>(pending_leaves_[static_cast<size_t>(i)].legal_indices.size())) {
        throw std::invalid_argument("legal logits must match pending legal move count");
      }
    }
    ++steps_;
    for (int i = 0; i < n; ++i) {
      PendingLeaf& leaf = pending_leaves_[static_cast<size_t>(i)];
      pending_legal_indices_ = leaf.legal_indices;
      pending_legal_mapping_ = leaf.legal_mapping;
      const int start = offsets[i];
      const int end = offsets[i + 1];
      expand_from_legal_eval(*leaf.path.back(), leaf.board, legal_logits + start, values[i]);
      clear_pending_legal();
      backup_pending(leaf, values[i]);
      ++sims_done_;
      (void)end;
    }
    pending_leaves_.erase(pending_leaves_.begin(), pending_leaves_.begin() + n);
    waiting_for_eval_ = !pending_leaves_.empty();
    if (!pending_leaves_.empty()) return;
    if (sims_done_ >= sims_target_) {
      collect_result();
      done_ = true;
      waiting_for_eval_ = false;
      return;
    }
    advance_after_expand();
    return;
  }

  // Single-leaf path.
  if (n != 1) throw std::invalid_argument("single-leaf apply_eval_legal expects n==1");
  const int legal_count = offsets[1] - offsets[0];
  if (legal_count != static_cast<int>(pending_legal_indices_.size())) {
    throw std::invalid_argument("legal logits must match pending legal move count");
  }
  ++steps_;
  Node* leaf = path_.back();
  expand_from_legal_eval(*leaf, path_board_, legal_logits, values[0]);
  clear_pending_legal();
  backup(values[0]);
  ++sims_done_;
  waiting_for_eval_ = false;

  if (sims_done_ >= sims_target_) {
    collect_result();
    done_ = true;
    return;
  }
  advance_after_expand();
}

int MctsSession::total_virtual_loss_node(const Node& node) const {
  int total = node.N_virtual;
  for (const auto& [idx, child] : node.children) {
    (void)idx;
    if (child) total += total_virtual_loss_node(*child);
  }
  return total;
}

int MctsSession::total_virtual_loss() const {
  if (!root_) return 0;
  return total_virtual_loss_node(*root_);
}

void MctsSession::collect_result() {
  result_ = MctsResult{};
  result_.root_value = root_value_;
  for (auto& [idx, child] : root_->children) {
    if (child->move.null()) continue;
    result_.moves_uci.push_back(move_to_uci(child->move));
    result_.native_moves.push_back(child->move);
    result_.indices.push_back(idx);
    result_.visits.push_back(static_cast<double>(child->N));
    double q = child->N > 0 ? -child->Q() : static_cast<double>(root_value_);
    result_.q_values.push_back(q);
    result_.priors.push_back(static_cast<double>(child->prior));
    auto it = root_clean_priors_.find(idx);
    result_.clean_priors.push_back(it != root_clean_priors_.end() ? it->second
                                                                  : child->prior);
  }
}

ExportedNode MctsSession::export_node(int idx, const Node& node, int depth_left) {
  ExportedNode out;
  out.index = idx;
  if (!node.move.null()) out.move_uci = move_to_uci(node.move);
  out.N = node.N;
  out.W = node.W;
  out.prior = node.prior;
  if (depth_left <= 0) return out;
  out.children.reserve(node.children.size());
  for (const auto& [cidx, child] : node.children) {
    if (!child) continue;
    out.children.push_back(export_node(cidx, *child, depth_left - 1));
  }
  return out;
}

std::vector<ExportedNode> MctsSession::export_tree(int max_depth) const {
  std::vector<ExportedNode> out;
  if (!root_ || max_depth <= 0) return out;
  out.reserve(root_->children.size());
  for (const auto& [idx, child] : root_->children) {
    if (!child) continue;
    out.push_back(export_node(idx, *child, max_depth - 1));
  }
  return out;
}

}  // namespace immortalite
