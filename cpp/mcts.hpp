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
};

struct MctsResult {
  std::vector<std::string> moves_uci;
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

  bool done() const { return done_; }

  // Returns number of positions needing eval (0 if done or waiting).
  // Writes planes into out_planes as (N, 20, 8, 8) row-major float32.
  // Caller provides buffer with capacity >= N * 20 * 64; N is usually 1.
  int positions_needing_eval(float* out_planes, int max_n);

  // Apply NN outputs. logits: (N, 4672), values: (N,).
  void apply_eval(const float* logits, const float* values, int n);

  // Legal policy indices for the single position currently awaiting evaluation.
  const std::vector<int>& pending_legal_indices() const { return pending_legal_indices_; }

  // Apply logits ordered as pending_legal_indices(), avoiding a full policy transfer.
  void apply_eval_legal(const float* legal_logits, int legal_count, float value);

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
    double W = 0.0;
    bool terminal_checked = false;
    bool is_terminal = false;
    float terminal_value = 0.0f;
    // Insertion order = legal-move order (matches python-chess / Zero expand).
    std::vector<std::pair<int, std::unique_ptr<Node>>> children;

    double Q() const { return N > 0 ? W / N : 0.0; }
    bool expanded() const { return !children.empty(); }
  };

  void select_to_leaf();
  void backup(float value);
  void expand_from_eval(Node& node, const Position& board, const float* logits, float value);
  void expand_from_legal_eval(Node& node, const Position& board, const float* legal_logits,
                              float value);
  void expand_with_priors(Node& node, const std::vector<std::pair<int, Move>>& mapping,
                          const std::vector<float>& priors);
  void cache_pending_legal_indices(const Position& board);
  void add_dirichlet_noise(Node& root);
  std::pair<bool, float> terminal_eval(Node& node, const Position& board);
  float value_from_outcome(const Outcome& outcome, const Position& board) const;
  std::pair<int, Node*> select_child(Node& node) const;
  void collect_result();
  void advance_after_expand();
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
  std::unordered_map<int, float> root_clean_priors_;
  MctsResult result_;
  std::uint64_t steps_ = 0;
  std::uint64_t nodes_expanded_ = 0;
};

}  // namespace immortalite
