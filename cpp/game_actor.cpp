#include "game_actor.hpp"

#include "encoding.hpp"
#include "movegen.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace immortalite {
namespace {

double policy_kl_target_from_prior(const std::vector<double>& prior,
                                   const std::vector<double>& target) {
  if (prior.empty() || prior.size() != target.size()) return 0.0;
  double prior_sum = 0.0;
  double target_sum = 0.0;
  for (size_t i = 0; i < prior.size(); ++i) {
    if (prior[i] < 0.0 || target[i] < 0.0) return 0.0;
    prior_sum += prior[i];
    target_sum += target[i];
  }
  if (prior_sum <= 0.0 || target_sum <= 0.0) return 0.0;
  double kl = 0.0;
  for (size_t i = 0; i < prior.size(); ++i) {
    const double p = std::max(prior[i] / prior_sum, 1e-12);
    const double t = std::max(target[i] / target_sum, 1e-12);
    kl += t * std::log(t / p);
  }
  return kl;
}

std::string termination_name(const Outcome& outcome) {
  switch (outcome.termination) {
    case Termination::Checkmate: return "checkmate";
    case Termination::Stalemate: return "stalemate";
    case Termination::InsufficientMaterial: return "insufficient_material";
    case Termination::SeventyfiveMoves: return "seventyfive_moves";
    case Termination::FivefoldRepetition: return "fivefold_repetition";
    case Termination::FiftyMoves: return "fifty_moves";
    case Termination::ThreefoldRepetition: return "threefold_repetition";
    default: return "no_legal_moves";
  }
}

}  // namespace

struct GameActorBatch::Actor {
  int id = 0;
  Position board;
  std::vector<std::string> moves;
  std::unique_ptr<MctsSession> session;
  GameActorState state = GameActorState::Playing;
  std::vector<GameActorSample> samples;
  int move_count = 0;
  int low_value_streak[2] = {0, 0};
  float last_root_value = 0.0f;
  std::string termination;
  int winner = -1;
  bool completed_taken = false;
  bool a_is_white = true;
  std::mt19937_64 rng;
};

GameActorBatch::GameActorBatch(int game_count, const GameActorConfig& actor_cfg,
                               const MctsConfig& mcts_cfg, std::uint64_t base_seed,
                               const std::vector<std::vector<std::string>>& start_moves,
                               const std::vector<std::uint8_t>& a_is_white)
    : actor_cfg_(actor_cfg), mcts_cfg_(mcts_cfg) {
  if (game_count < 0) throw std::invalid_argument("game_count must be non-negative");
  if (!start_moves.empty() && static_cast<int>(start_moves.size()) != game_count) {
    throw std::invalid_argument("start_moves length must match game_count");
  }
  if (!a_is_white.empty() && static_cast<int>(a_is_white.size()) != game_count) {
    throw std::invalid_argument("a_is_white length must match game_count");
  }
  if (actor_cfg_.value_target != "outcome" && actor_cfg_.value_target != "root_q" &&
      actor_cfg_.value_target != "q_z") {
    throw std::invalid_argument("value_target must be 'outcome', 'root_q', or 'q_z'");
  }
  if (actor_cfg_.value_q_ratio < 0.0f || actor_cfg_.value_q_ratio > 1.0f) {
    throw std::invalid_argument("value_q_ratio must be in [0, 1]");
  }
  mcts_cfg_.claim_draw = actor_cfg_.claim_draw;
  mcts_cfg_.draw_contempt = actor_cfg_.draw_contempt;
  init_attack_tables();
  init_zobrist();
  actors_.reserve(static_cast<size_t>(game_count));
  for (int id = 0; id < game_count; ++id) {
    auto actor = std::make_unique<Actor>();
    actor->id = id;
    actor->rng.seed(base_seed + static_cast<std::uint64_t>(id));
    if (!start_moves.empty()) {
      const auto& opening = start_moves[static_cast<size_t>(id)];
      if (!opening.empty()) {
        if (!actor->board.apply_uci_list(opening)) {
          throw std::invalid_argument("illegal opening move for actor " + std::to_string(id));
        }
        actor->moves = opening;
        actor->move_count = static_cast<int>(opening.size());
      }
    }
    if (!a_is_white.empty()) {
      actor->a_is_white = a_is_white[static_cast<size_t>(id)] != 0;
    }
    actors_.push_back(std::move(actor));
    advance(*actors_.back());
  }
}

GameActorBatch::~GameActorBatch() = default;

void GameActorBatch::advance(Actor& actor) {
  if (actor.state == GameActorState::Completed) return;
  const Outcome outcome = actor.board.outcome(actor_cfg_.claim_draw);
  if (outcome.is_terminal()) {
    actor.termination = termination_name(outcome);
    actor.winner = outcome.winner;
    complete(actor);
    return;
  }
  if (actor.move_count >= actor_cfg_.max_game_moves) {
    actor.termination = "max_moves";
    complete(actor);
    return;
  }
  if (actor_cfg_.tb_max_pieces > 0 && popcount(actor.board.occupied()) <= actor_cfg_.tb_max_pieces) {
    actor.state = GameActorState::NeedTablebase;
    return;
  }
  // Copy the live board (with repetition / 50-move history) instead of
  // replaying start-FEN + full UCI list every ply.
  actor.session = std::make_unique<MctsSession>(actor.board, actor_cfg_.simulations, mcts_cfg_,
                                                actor_cfg_.add_noise);
  actor.state = GameActorState::NeedEval;
}

std::vector<TablebaseRequest> GameActorBatch::tablebase_requests() const {
  std::vector<TablebaseRequest> out;
  for (const auto& actor : actors_) {
    if (actor->state == GameActorState::NeedTablebase) {
      out.push_back({actor->id, actor->board.get_fen(), popcount(actor->board.occupied())});
    }
  }
  return out;
}

void GameActorBatch::apply_tablebase(const std::vector<int>& actor_ids,
                                     const std::vector<TablebaseOutcome>& outcomes) {
  if (actor_ids.size() != outcomes.size()) throw std::invalid_argument("tablebase result length");
  for (size_t i = 0; i < actor_ids.size(); ++i) {
    const int id = actor_ids[i];
    if (id < 0 || id >= static_cast<int>(actors_.size())) throw std::invalid_argument("actor id");
    Actor& actor = *actors_[static_cast<size_t>(id)];
    if (actor.state != GameActorState::NeedTablebase) continue;
    switch (outcomes[i]) {
      case TablebaseOutcome::WinSTM:
        actor.termination = "tablebase_win";
        actor.winner = actor.board.side_to_move();
        complete(actor);
        break;
      case TablebaseOutcome::WinOpp:
        actor.termination = "tablebase_win";
        actor.winner = ~actor.board.side_to_move();
        complete(actor);
        break;
      case TablebaseOutcome::Draw:
        actor.termination = "tablebase_draw";
        complete(actor);
        break;
      case TablebaseOutcome::Unavailable:
        actor.session = std::make_unique<MctsSession>(actor.board, actor_cfg_.simulations, mcts_cfg_,
                                                      actor_cfg_.add_noise);
        actor.state = GameActorState::NeedEval;
        break;
    }
  }
}

int GameActorBatch::positions_needing_eval(float* out_planes, int capacity,
                                           std::vector<int>& actor_ids) {
  actor_ids.clear();
  last_eval_actor_ids_.clear();
  if (capacity < 0) throw std::invalid_argument("negative plane capacity");
  int count = 0;
  for (auto& actor : actors_) {
    if (actor->state != GameActorState::NeedEval || !actor->session || actor->session->done()) continue;
    if (count >= capacity) break;
    if (actor->session->positions_needing_eval(out_planes + count * NUM_INPUT_PLANES * 64, 1) > 0) {
      actor_ids.push_back(actor->id);
      last_eval_actor_ids_.push_back(actor->id);
      ++count;
    }
  }
  return count;
}

std::vector<int> GameActorBatch::pending_net_ids() const {
  std::vector<int> out;
  out.reserve(last_eval_actor_ids_.size());
  for (int id : last_eval_actor_ids_) {
    const Actor& actor = *actors_[static_cast<size_t>(id)];
    const bool use_a = (actor.board.side_to_move() == WHITE) == actor.a_is_white;
    out.push_back(use_a ? 0 : 1);
  }
  return out;
}

LegalCsr GameActorBatch::pending_legal_csr() const {
  LegalCsr out;
  out.offsets.push_back(0);
  for (int id : last_eval_actor_ids_) {
    const auto& actor = *actors_[static_cast<size_t>(id)];
    const auto& legal = actor.session->pending_legal_indices();
    out.indices.insert(out.indices.end(), legal.begin(), legal.end());
    out.offsets.push_back(static_cast<int>(out.indices.size()));
  }
  return out;
}

void GameActorBatch::finish_search(Actor& actor) {
  const MctsResult& result = actor.session->result();
  if (result.moves_uci.empty()) {
    actor.termination = "no_legal_moves";
    complete(actor);
    return;
  }
  GameActorSample sample;
  sample.planes.resize(NUM_INPUT_PLANES * 64);
  fill_planes(actor.board, sample.planes.data());
  sample.policy.assign(POLICY_SIZE, 0.0f);
  const auto improved = result.improved_policy(mcts_cfg_);
  for (size_t i = 0; i < result.indices.size(); ++i) {
    sample.policy[static_cast<size_t>(result.indices[i])] = static_cast<float>(improved[i]);
  }
  sample.player = actor.board.side_to_move();
  double visit_sum = std::accumulate(result.visits.begin(), result.visits.end(), 0.0);
  double q_sum = 0.0;
  for (size_t i = 0; i < result.visits.size(); ++i) q_sum += result.visits[i] * result.q_values[i];
  sample.root_q = static_cast<float>(visit_sum > 0.0 ? q_sum / visit_sum : result.root_value);
  sample.policy_surprise = static_cast<float>(
      policy_kl_target_from_prior(result.clean_priors, improved));
  actor.last_root_value = sample.root_q;
  actor.samples.push_back(std::move(sample));

  const Color player = actor.board.side_to_move();
  const bool resign_enabled = actor_cfg_.resign_plies > 0 && actor_cfg_.resign_threshold >= -1.0f;
  if (resign_enabled && actor.move_count >= actor_cfg_.resign_min_moves) {
    int& streak = actor.low_value_streak[static_cast<int>(player)];
    streak = actor.last_root_value <= actor_cfg_.resign_threshold ? streak + 1 : 0;
    if (streak >= actor_cfg_.resign_plies) {
      actor.termination = "resign";
      actor.winner = ~player;
      complete(actor);
      return;
    }
  }

  size_t chosen = 0;
  if (actor.move_count < actor_cfg_.exploration_moves) {
    std::vector<double> probs = improved;
    if (actor_cfg_.move_temperature_plies > 0 &&
        actor.move_count < actor_cfg_.move_temperature_plies &&
        std::abs(actor_cfg_.move_temperature - 1.0f) > 1e-12f) {
      double sum = 0.0;
      for (double& p : probs) {
        p = std::pow(std::max(p, 1e-12), 1.0 / actor_cfg_.move_temperature);
        sum += p;
      }
      for (double& p : probs) p /= sum;
    }
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    chosen = dist(actor.rng);
  } else {
    chosen = static_cast<size_t>(std::distance(
        result.visits.begin(), std::max_element(result.visits.begin(), result.visits.end())));
  }
  const Move move = actor.board.parse_uci(result.moves_uci[chosen]);
  if (move.null()) throw std::logic_error("MCTS returned illegal root move");
  actor.board.make_move(move);
  actor.moves.push_back(result.moves_uci[chosen]);
  ++actor.move_count;
  actor.session.reset();
  actor.state = GameActorState::Playing;
  advance(actor);
}

void GameActorBatch::apply_eval(const std::vector<int>& actor_ids, const float* logits,
                                const float* values, int n) {
  if (n != static_cast<int>(actor_ids.size()) || logits == nullptr || values == nullptr) {
    throw std::invalid_argument("eval batch shape");
  }
  std::unordered_set<int> seen;
  for (int id : actor_ids) {
    if (id < 0 || id >= static_cast<int>(actors_.size())) throw std::invalid_argument("actor id");
    if (!seen.insert(id).second) throw std::invalid_argument("duplicate actor id");
    const Actor& actor = *actors_[static_cast<size_t>(id)];
    if (actor.state != GameActorState::NeedEval || !actor.session ||
        std::find(last_eval_actor_ids_.begin(), last_eval_actor_ids_.end(), id) ==
            last_eval_actor_ids_.end()) {
      throw std::invalid_argument("actor id is not pending evaluation");
    }
  }
  for (int row = 0; row < n; ++row) {
    const int id = actor_ids[static_cast<size_t>(row)];
    Actor& actor = *actors_[static_cast<size_t>(id)];
    actor.session->apply_eval(logits + static_cast<size_t>(row) * POLICY_SIZE, values + row, 1);
    if (actor.session->done()) finish_search(actor);
  }
  std::erase_if(last_eval_actor_ids_, [&](int id) { return seen.contains(id); });
}

void GameActorBatch::apply_eval_legal(const std::vector<int>& actor_ids, const float* legal_logits,
                                      const int* offsets, const float* values, int n) {
  if (n != static_cast<int>(actor_ids.size()) || legal_logits == nullptr || offsets == nullptr ||
      values == nullptr) throw std::invalid_argument("legal eval batch shape");
  if (offsets[0] != 0) throw std::invalid_argument("legal offsets must start at zero");
  std::unordered_set<int> seen;
  for (int row = 0; row < n; ++row) {
    const int id = actor_ids[static_cast<size_t>(row)];
    if (id < 0 || id >= static_cast<int>(actors_.size())) throw std::invalid_argument("actor id");
    if (!seen.insert(id).second) throw std::invalid_argument("duplicate actor id");
    const Actor& actor = *actors_[static_cast<size_t>(id)];
    if (actor.state != GameActorState::NeedEval || !actor.session ||
        std::find(last_eval_actor_ids_.begin(), last_eval_actor_ids_.end(), id) ==
            last_eval_actor_ids_.end()) {
      throw std::invalid_argument("actor id is not pending evaluation");
    }
    const int start = offsets[row], end = offsets[row + 1];
    if (start < 0 || end < start ||
        end - start != static_cast<int>(actor.session->pending_legal_indices().size())) {
      throw std::invalid_argument("legal offsets do not match pending legal moves");
    }
  }
  for (int row = 0; row < n; ++row) {
    Actor& actor = *actors_[static_cast<size_t>(actor_ids[static_cast<size_t>(row)])];
    const int start = offsets[row], end = offsets[row + 1];
    actor.session->apply_eval_legal(legal_logits + start, end - start, values[row]);
    if (actor.session->done()) finish_search(actor);
  }
  std::erase_if(last_eval_actor_ids_, [&](int id) { return seen.contains(id); });
}

void GameActorBatch::complete(Actor& actor) {
  if (actor.state == GameActorState::Completed) return;
  if (actor_cfg_.value_target == "root_q") {
    for (auto& sample : actor.samples) sample.value = sample.root_q;
  } else {
    // Outcome z (also the Z side of soft Q+Z).
    if (actor.termination == "max_moves" && !actor.samples.empty()) {
      const Color last = actor.samples.back().player;
      for (auto& sample : actor.samples)
        sample.value = sample.player == last ? actor.last_root_value : -actor.last_root_value;
    } else {
      float target = 0.0f;
      if (actor.termination == "checkmate" || actor.termination == "resign" ||
          actor.termination == "tablebase_win") {
        target = 1.0f;
        if (actor.termination == "checkmate" && actor_cfg_.fast_mate_bonus > 0.0f) {
          target += actor_cfg_.fast_mate_bonus / std::max(1, actor.move_count);
        }
        for (auto& sample : actor.samples)
          sample.value = sample.player == actor.winner ? target : -target;
      } else {
        if (actor.termination == "stalemate" || actor.termination == "insufficient_material" ||
            actor.termination == "fifty_moves" || actor.termination == "seventyfive_moves" ||
            actor.termination == "threefold_repetition" || actor.termination == "fivefold_repetition" ||
            actor.termination == "tablebase_draw")
          target = -actor_cfg_.draw_penalty;
        for (auto& sample : actor.samples) sample.value = target;
      }
    }
    if (actor_cfg_.value_target == "q_z") {
      const float alpha = actor_cfg_.value_q_ratio;
      for (auto& sample : actor.samples) {
        sample.value = alpha * sample.root_q + (1.0f - alpha) * sample.value;
      }
    }
  }
  actor.session.reset();
  actor.state = GameActorState::Completed;
}

std::vector<CompletedGame> GameActorBatch::take_completed() {
  std::vector<CompletedGame> out;
  for (auto& actor : actors_) {
    if (actor->state != GameActorState::Completed || actor->completed_taken) continue;
    out.push_back({actor->id, std::move(actor->samples), actor->termination, actor->winner, actor->moves});
    actor->completed_taken = true;
  }
  return out;
}

}  // namespace immortalite
