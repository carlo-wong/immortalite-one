#pragma once

#include "mcts.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace immortalite {

enum class GameActorState { Playing, NeedTablebase, NeedEval, Completed };
enum class TablebaseOutcome { Unavailable, WinSTM, WinOpp, Draw };

struct GameActorConfig {
  int simulations = 100;
  int max_game_moves = 10000;
  bool claim_draw = true;
  float draw_contempt = 1.0f / 3.0f;
  float draw_penalty = 1.0f / 3.0f;
  std::string value_target = "outcome";
  float value_q_ratio = 0.5f;  // weight on root_q when value_target == "q_z"
  float resign_threshold = -1.1f;
  int resign_plies = 0;
  int resign_min_moves = 0;
  float move_temperature = 1.0f;
  int move_temperature_plies = 0;
  int exploration_moves = 20;
  int tb_max_pieces = 5;
  float fast_mate_bonus = 0.0f;
  bool add_noise = true;
};

struct TablebaseRequest {
  int actor_id;
  std::string fen;
  int piece_count;
};

struct LegalCsr {
  std::vector<int> indices;
  std::vector<int> offsets;
};

struct GameActorSample {
  std::vector<float> planes;
  std::vector<float> policy;
  Color player = WHITE;
  float root_q = 0.0f;
  float value = 0.0f;
  // KL(π_target ‖ π_prior) over legal moves at write time (0 if unavailable).
  float policy_surprise = 0.0f;
};

struct CompletedGame {
  int actor_id = -1;
  std::vector<GameActorSample> samples;
  std::string termination;
  int winner = -1;
  std::vector<std::string> moves;
};

class GameActorBatch {
 public:
  GameActorBatch(int game_count, const GameActorConfig& actor_cfg, const MctsConfig& mcts_cfg,
                 std::uint64_t base_seed,
                 const std::vector<std::vector<std::string>>& start_moves = {},
                 const std::vector<std::uint8_t>& a_is_white = {});
  ~GameActorBatch();

  std::vector<TablebaseRequest> tablebase_requests() const;
  void apply_tablebase(const std::vector<int>& actor_ids,
                       const std::vector<TablebaseOutcome>& outcomes);
  int positions_needing_eval(float* out_planes, int capacity, std::vector<int>& actor_ids);
  std::vector<int> pending_net_ids() const;
  LegalCsr pending_legal_csr() const;
  void apply_eval(const std::vector<int>& actor_ids, const float* logits, const float* values,
                  int n);
  void apply_eval_legal(const std::vector<int>& actor_ids, const float* legal_logits,
                        const int* offsets, const float* values, int n);
  std::vector<CompletedGame> take_completed();

 private:
  struct Actor;
  void advance(Actor& actor);
  void complete(Actor& actor);
  void finish_search(Actor& actor);

  GameActorConfig actor_cfg_;
  MctsConfig mcts_cfg_;
  std::vector<std::unique_ptr<Actor>> actors_;
  std::vector<int> last_eval_actor_ids_;
};

}  // namespace immortalite
