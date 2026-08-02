#pragma once

#include "board.hpp"
#include "types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace immortalite {

struct MctsConfig {
  int simulations = 100;
  float c_puct = 1.5f;
  float dirichlet_alpha = 0.3f;
  float dirichlet_epsilon = 0.25f;
  float gumbel_c_visit = 50.0f;
  float gumbel_c_scale = 0.1f;
  float draw_contempt = 1.0f / 3.0f;
  bool claim_draw = true;
  // Multi-leaf / virtual-loss. Both must be enabled (VL>0 and max_leaves>1)
  // to leave the single-leaf path; defaults preserve bit-identical search.
  int virtual_loss = 0;           // 0 = OFF
  int max_leaves_per_eval = 1;    // 1 = current single-leaf behavior
};

struct MctsResult {
  std::vector<std::string> moves_uci;
  // Native callers can apply the selected root move without reparsing UCI and
  // rescanning the same legal move list. Python-facing result shape is unchanged.
  std::vector<Move> native_moves;
  std::vector<int> indices;
  std::vector<double> visits;
  std::vector<double> q_values;
  std::vector<double> priors;
  std::vector<double> clean_priors;
  float root_value = 0.0f;

  // Gumbel completed-Q improved policy over considered moves.
  std::vector<double> improved_policy(const MctsConfig& cfg) const;
};

// Serializable MCTS node for Python PV reconstruction (policy index keyed).
struct ExportedNode {
  int index = -1;
  std::string move_uci;
  int N = 0;
  double W = 0.0;
  float prior = 0.0f;
  std::vector<ExportedNode> children;
};

struct MctsSessionStats {
  std::uint64_t steps = 0;
  std::uint64_t simulations_completed = 0;
  std::uint64_t nodes_expanded = 0;
};

class MctsSession {
 public:
  // `fen` is the position to search. Optional `moves_from_fen` is applied first
  // (building repetition / 50-move history) when the true root is fen+moves.
  // For self-play with prior game history, pass the game's start FEN and the
  // UCI move list so claim_draw / repetition planes match python-chess.
  MctsSession(std::string_view fen, int simulations, const MctsConfig& cfg, bool add_noise,
              const std::vector<std::string>& moves_from_fen = {});

  // Prefer this for self-play: copies an already-advanced Position (including
  // undo/repetition history) so each ply does not replay start-FEN + UCI.
  MctsSession(const Position& root, int simulations, const MctsConfig& cfg, bool add_noise);

  bool done() const { return done_; }

  // Returns number of positions needing eval (0 if done or waiting).
  // Writes planes into out_planes as (N, 20, 8, 8) row-major float32.
  // Caller provides buffer with capacity >= N * 20 * 64; N is usually 1.
  // When virtual-loss multi-leaf is enabled, N may be up to min(max_n,
  // max_leaves_per_eval).
  int positions_needing_eval(float* out_planes, int max_n);

  // Apply NN outputs. logits: (N, 4672), values: (N,).
  // N must match the number of pending eval positions currently exposed.
  void apply_eval(const float* logits, const float* values, int n);

  // Legal policy indices for pending eval leaf `i` (i=0 is the only leaf when
  // multi-leaf is off).
  const std::vector<int>& pending_legal_indices(int leaf = 0) const;

  int pending_eval_count() const;

  // Apply logits ordered as pending_legal_indices(), avoiding a full policy transfer.
  void apply_eval_legal(const float* legal_logits, int legal_count, float value);

  // Batch legal apply for n pending leaves. offsets has n+1 entries (CSR).
  void apply_eval_legal(const float* legal_logits, const int* offsets, const float* values,
                        int n);

  // Sum of outstanding virtual-loss counters across the tree (0 when idle/done).
  int total_virtual_loss() const;

  const MctsResult& result() const { return result_; }
  MctsSessionStats stats() const {
    return {
        steps_,
        static_cast<std::uint64_t>(sims_done_),
        nodes_expanded_,
    };
  }
  MctsConfig& config() { return cfg_; }
  const MctsConfig& config() const { return cfg_; }

  // Export root children (and expanded subtrees) for Python PV walking.
  // max_depth = max plies below the root (default covers GUI pv_len=8).
  std::vector<ExportedNode> export_tree(int max_depth = 32) const;

 private:
  struct Node {
    float prior = 0.0f;
    Move move = NULL_MOVE;
    int N = 0;
    int N_virtual = 0;  // outstanding virtual loss (PUCT only; cleared on backup)
    double W = 0.0;
    bool terminal_checked = false;
    bool is_terminal = false;
    float terminal_value = 0.0f;
    // Insertion order = legal-move order (matches python-chess / Zero expand).
    std::vector<std::pair<int, std::unique_ptr<Node>>> children;

    double Q() const { return N > 0 ? W / N : 0.0; }
    bool expanded() const { return !children.empty(); }
  };

  // One in-flight leaf awaiting NN eval (virtual-loss multi-leaf mode only).
  struct PendingLeaf {
    std::vector<Node*> path;
    int path_depth = 0;
    Position board;
    std::vector<int> legal_indices;
    std::vector<std::pair<int, Move>> legal_mapping;
  };

  bool multi_leaf_mode() const {
    return cfg_.virtual_loss > 0 && cfg_.max_leaves_per_eval > 1;
  }

  void select_to_leaf();
  // Select up to max_k distinct leaves, applying virtual loss so paths diverge.
  void select_leaves(int max_k);
  // Returns true if a new pending leaf was parked; false if a terminal was
  // backed up or an in-flight leaf was hit (caller should stop).
  bool select_one_leaf_with_virtual_loss();
  void apply_virtual_loss_on_path();
  void remove_virtual_loss_on_path(const std::vector<Node*>& path);
  void backup(float value);
  void backup_pending(PendingLeaf& leaf, float value);
  void expand_from_eval(Node& node, const Position& board, const float* logits, float value);
  void expand_from_legal_eval(Node& node, const Position& board, const float* legal_logits,
                              float value);
  void expand_with_priors(Node& node, const std::vector<std::pair<int, Move>>& mapping,
                          const std::vector<float>& priors);
  void cache_pending_legal_indices(const Position& board);
  void clear_pending_legal();
  void begin_search_from_root();
  void add_dirichlet_noise(Node& root);
  std::pair<bool, float> terminal_eval(Node& node, const Position& board);
  float value_from_outcome(const Outcome& outcome, const Position& board) const;
  std::pair<int, Node*> select_child(Node& node) const;
  void collect_result();
  void advance_after_expand();
  int total_virtual_loss_node(const Node& node) const;
  static ExportedNode export_node(int idx, const Node& node, int depth_left);

  MctsConfig cfg_;
  int sims_target_ = 0;
  int sims_done_ = 0;
  bool add_noise_ = false;
  bool done_ = false;
  bool need_root_eval_ = true;
  bool waiting_for_eval_ = false;
  float root_value_ = 0.0f;
  Color root_turn_ = WHITE;

  Position root_board_;
  Position path_board_;
  std::unique_ptr<Node> root_;
  std::vector<Node*> path_;
  int path_depth_ = 0;
  std::vector<int> pending_legal_indices_;
  // Full (policy_index, Move) mapping cached with pending_legal_indices_ so
  // expand does not regenerate legal moves a second time.
  std::vector<std::pair<int, Move>> pending_legal_mapping_;
  std::vector<PendingLeaf> pending_leaves_;
  std::unordered_map<int, float> root_clean_priors_;
  MctsResult result_;
  std::uint64_t steps_ = 0;
  std::uint64_t nodes_expanded_ = 0;
};

}  // namespace immortalite
