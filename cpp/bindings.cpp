#include "board.hpp"
#include "encoding.hpp"
#include "game_actor.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "types.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<double> to_numpy(const std::vector<double>& v) {
  py::array_t<double> arr(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) {
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(double));
  }
  return arr;
}

immortalite::MctsConfig config_from_dict(const py::dict& d) {
  immortalite::MctsConfig cfg;
  auto get_f = [&](const char* key, float& dst) {
    if (d.contains(key)) dst = d[key].cast<float>();
  };
  auto get_i = [&](const char* key, int& dst) {
    if (d.contains(key)) dst = d[key].cast<int>();
  };
  auto get_b = [&](const char* key, bool& dst) {
    if (d.contains(key)) dst = d[key].cast<bool>();
  };
  get_i("simulations", cfg.simulations);
  get_f("c_puct", cfg.c_puct);
  get_f("dirichlet_alpha", cfg.dirichlet_alpha);
  get_f("dirichlet_epsilon", cfg.dirichlet_epsilon);
  get_f("gumbel_c_visit", cfg.gumbel_c_visit);
  get_f("gumbel_c_scale", cfg.gumbel_c_scale);
  get_f("draw_contempt", cfg.draw_contempt);
  get_b("claim_draw", cfg.claim_draw);
  return cfg;
}

std::vector<std::vector<std::string>> start_moves_from_py(py::object obj, int game_count) {
  std::vector<std::vector<std::string>> out;
  if (obj.is_none()) return out;
  const py::sequence seq = obj.cast<py::sequence>();
  if (static_cast<int>(seq.size()) != game_count) {
    throw std::invalid_argument("start_moves length must match game_count");
  }
  out.reserve(static_cast<size_t>(game_count));
  for (py::ssize_t i = 0; i < seq.size(); ++i) {
    const py::sequence moves = seq[i].cast<py::sequence>();
    std::vector<std::string> row;
    row.reserve(static_cast<size_t>(moves.size()));
    for (py::ssize_t j = 0; j < moves.size(); ++j) {
      row.push_back(moves[j].cast<std::string>());
    }
    out.push_back(std::move(row));
  }
  return out;
}

std::vector<std::uint8_t> a_is_white_from_py(py::object obj, int game_count) {
  std::vector<std::uint8_t> out;
  if (obj.is_none()) return out;
  if (py::isinstance<py::array>(obj)) {
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr = obj;
    if (arr.ndim() != 1 || static_cast<int>(arr.shape(0)) != game_count) {
      throw std::invalid_argument("a_is_white length must match game_count");
    }
    out.assign(arr.data(), arr.data() + game_count);
    return out;
  }
  const py::sequence seq = obj.cast<py::sequence>();
  if (static_cast<int>(seq.size()) != game_count) {
    throw std::invalid_argument("a_is_white length must match game_count");
  }
  out.reserve(static_cast<size_t>(game_count));
  for (py::ssize_t i = 0; i < seq.size(); ++i) {
    out.push_back(seq[i].cast<bool>() ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0));
  }
  return out;
}

py::array_t<float> fill_planes_fen(const std::string& fen,
                                   const std::optional<std::vector<std::string>>& moves) {
  immortalite::init_attack_tables();
  immortalite::init_zobrist();
  immortalite::Position pos;
  if (!pos.set_fen(fen)) throw std::invalid_argument("invalid FEN");
  if (moves) {
    if (!pos.apply_uci_list(*moves)) throw std::invalid_argument("illegal move in history");
  }
  py::array_t<float> arr({immortalite::NUM_INPUT_PLANES, 8, 8});
  auto buf = arr.mutable_unchecked<3>();
  std::vector<float> tmp(static_cast<size_t>(immortalite::NUM_INPUT_PLANES * 64));
  immortalite::fill_planes(pos, tmp.data());
  for (int p = 0; p < immortalite::NUM_INPUT_PLANES; ++p)
    for (int r = 0; r < 8; ++r)
      for (int f = 0; f < 8; ++f) buf(p, r, f) = tmp[static_cast<size_t>((p * 8 + r) * 8 + f)];
  return arr;
}

py::list legal_move_indices_fen(const std::string& fen) {
  immortalite::init_attack_tables();
  immortalite::init_zobrist();
  immortalite::Position pos;
  if (!pos.set_fen(fen)) throw std::invalid_argument("invalid FEN");
  auto mapping = immortalite::legal_move_indices(pos);
  py::list out;
  for (auto& [idx, mv] : mapping) {
    out.append(py::make_tuple(idx, immortalite::move_to_uci(mv)));
  }
  return out;
}

int move_to_index_fen(const std::string& fen, const std::string& uci) {
  immortalite::init_attack_tables();
  immortalite::init_zobrist();
  immortalite::Position pos;
  if (!pos.set_fen(fen)) throw std::invalid_argument("invalid FEN");
  immortalite::Move m = pos.parse_uci(uci);
  if (m.null()) throw std::invalid_argument("illegal or invalid UCI move");
  return immortalite::move_to_index(pos, m);
}

class PyMctsSession {
 public:
  PyMctsSession(const std::string& fen, int simulations, const py::dict& config, bool add_noise,
                const std::optional<std::vector<std::string>>& moves)
      : session_(std::make_unique<immortalite::MctsSession>(
            fen, simulations, config_from_dict(config), add_noise,
            moves ? *moves : std::vector<std::string>{})) {}

  bool done() const { return session_->done(); }

  py::dict stats() const {
    const auto stats = session_->stats();
    py::dict d;
    d["steps"] = stats.steps;
    d["simulations_completed"] = stats.simulations_completed;
    d["nodes_expanded"] = stats.nodes_expanded;
    return d;
  }

  py::array_t<float> positions_needing_eval() {
    std::vector<float> tmp(static_cast<size_t>(immortalite::NUM_INPUT_PLANES * 64));
    int n = session_->positions_needing_eval(tmp.data(), 1);
    if (n <= 0) {
      return py::array_t<float>(std::vector<py::ssize_t>{0, immortalite::NUM_INPUT_PLANES, 8, 8});
    }
    py::array_t<float> arr({1, immortalite::NUM_INPUT_PLANES, 8, 8});
    auto buf = arr.mutable_unchecked<4>();
    for (int p = 0; p < immortalite::NUM_INPUT_PLANES; ++p)
      for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 8; ++f)
          buf(0, p, r, f) = tmp[static_cast<size_t>((p * 8 + r) * 8 + f)];
    return arr;
  }

  bool positions_needing_eval_into(py::array_t<float, py::array::c_style> out, int row) {
    if (!out.writeable()) throw std::invalid_argument("out must be writable");
    if (out.ndim() != 4 || out.shape(1) != immortalite::NUM_INPUT_PLANES ||
        out.shape(2) != 8 || out.shape(3) != 8) {
      throw std::invalid_argument("out must have shape (N, 20, 8, 8)");
    }
    if (row < 0 || row >= out.shape(0)) {
      throw std::invalid_argument("row must be within out's first dimension");
    }
    auto buffer = out.mutable_unchecked<4>();
    return session_->positions_needing_eval(&buffer(row, 0, 0, 0), 1) > 0;
  }

  py::array_t<int> pending_legal_indices() const {
    const auto& indices = session_->pending_legal_indices();
    py::array_t<int> out(static_cast<py::ssize_t>(indices.size()));
    if (!indices.empty()) {
      std::memcpy(out.mutable_data(), indices.data(), indices.size() * sizeof(int));
    }
    return out;
  }

  void apply_eval(py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                  py::array_t<float, py::array::c_style | py::array::forcecast> values) {
    auto l = logits.unchecked();
    auto v = values.unchecked();
    if (l.ndim() != 2 || l.shape(1) != immortalite::POLICY_SIZE) {
      throw std::invalid_argument("logits must have shape (N, 4672)");
    }
    if (v.ndim() != 1 || v.shape(0) != l.shape(0)) {
      throw std::invalid_argument("values must have shape (N,)");
    }
    int n = static_cast<int>(l.shape(0));
    if (n < 1) return;
    std::vector<float> lbuf(static_cast<size_t>(n * immortalite::POLICY_SIZE));
    std::vector<float> vbuf(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      vbuf[static_cast<size_t>(i)] = v(i);
      for (int j = 0; j < immortalite::POLICY_SIZE; ++j)
        lbuf[static_cast<size_t>(i * immortalite::POLICY_SIZE + j)] = l(i, j);
    }
    session_->apply_eval(lbuf.data(), vbuf.data(), n);
  }

  void apply_eval_legal(py::array_t<float, py::array::c_style> legal_logits, float value) {
    if (legal_logits.ndim() != 1) {
      throw std::invalid_argument("legal_logits must be a one-dimensional float32 array");
    }
    session_->apply_eval_legal(legal_logits.data(), static_cast<int>(legal_logits.shape(0)),
                               value);
  }

  py::dict result() const {
    const auto& r = session_->result();
    py::dict d;
    d["moves"] = r.moves_uci;
    d["indices"] = r.indices;
    d["visits"] = to_numpy(r.visits);
    d["q_values"] = to_numpy(r.q_values);
    d["priors"] = to_numpy(r.priors);
    d["clean_priors"] = to_numpy(r.clean_priors);
    d["root_value"] = r.root_value;
    d["improved_policy"] = to_numpy(r.improved_policy(session_->config()));
    d["tree"] = tree_to_list(session_->export_tree(32));
    return d;
  }

 private:
  static py::dict exported_node_to_dict(const immortalite::ExportedNode& n) {
    py::dict d;
    d["index"] = n.index;
    d["move"] = n.move_uci;
    d["N"] = n.N;
    d["W"] = n.W;
    d["prior"] = n.prior;
    py::list children;
    for (const auto& c : n.children) children.append(exported_node_to_dict(c));
    d["children"] = children;
    return d;
  }

  static py::list tree_to_list(const std::vector<immortalite::ExportedNode>& nodes) {
    py::list out;
    for (const auto& n : nodes) out.append(exported_node_to_dict(n));
    return out;
  }

  std::unique_ptr<immortalite::MctsSession> session_;
};

class PyGameActorBatch {
 public:
  PyGameActorBatch(int game_count, const py::dict& actor_config, const py::dict& mcts_config,
                   std::uint64_t base_seed, py::object start_moves = py::none(),
                   py::object a_is_white = py::none())
      : batch_(game_count, actor_config_from_dict(actor_config), config_from_dict(mcts_config),
               base_seed, start_moves_from_py(start_moves, game_count),
               a_is_white_from_py(a_is_white, game_count)) {}

  py::list tablebase_requests() const {
    py::list out;
    for (const auto& request : batch_.tablebase_requests()) {
      py::dict d;
      d["actor_id"] = request.actor_id;
      d["fen"] = request.fen;
      d["piece_count"] = request.piece_count;
      out.append(d);
    }
    return out;
  }

  void apply_tablebase(py::array_t<int, py::array::c_style | py::array::forcecast> ids,
                       py::array_t<int, py::array::c_style | py::array::forcecast> outcomes) {
    if (ids.ndim() != 1 || outcomes.ndim() != 1 || ids.shape(0) != outcomes.shape(0))
      throw std::invalid_argument("actor_ids and outcomes must be equal-length vectors");
    std::vector<int> actor_ids(ids.data(), ids.data() + ids.shape(0));
    std::vector<immortalite::TablebaseOutcome> values;
    values.reserve(static_cast<size_t>(outcomes.shape(0)));
    for (py::ssize_t i = 0; i < outcomes.shape(0); ++i) {
      const int value = outcomes.data()[i];
      if (value < 0 || value > 3) throw std::invalid_argument("invalid tablebase outcome");
      values.push_back(static_cast<immortalite::TablebaseOutcome>(value));
    }
    batch_.apply_tablebase(actor_ids, values);
  }

  py::tuple positions_needing_eval(py::object out = py::none()) {
    py::array_t<float> owned;
    py::array_t<float, py::array::c_style> target;
    if (out.is_none()) {
      owned = py::array_t<float>(py::array::ShapeContainer{
          static_cast<py::ssize_t>(1024), static_cast<py::ssize_t>(immortalite::NUM_INPUT_PLANES),
          static_cast<py::ssize_t>(8), static_cast<py::ssize_t>(8)});
      target = owned;
    } else {
      target = out.cast<py::array_t<float, py::array::c_style>>();
      if (!target.writeable() || target.ndim() != 4 || target.shape(1) != immortalite::NUM_INPUT_PLANES ||
          target.shape(2) != 8 || target.shape(3) != 8) {
        throw std::invalid_argument("out must be writable shape (N, 20, 8, 8)");
      }
    }
    std::vector<int> ids;
    const int n = batch_.positions_needing_eval(target.mutable_data(), static_cast<int>(target.shape(0)), ids);
    py::array_t<int> actor_ids(static_cast<py::ssize_t>(ids.size()));
    if (!ids.empty()) std::memcpy(actor_ids.mutable_data(), ids.data(), ids.size() * sizeof(int));
    py::array planes = target[py::make_tuple(py::slice(0, n, 1), py::ellipsis())];
    return py::make_tuple(actor_ids, planes);
  }

  py::array_t<std::int32_t> pending_net_ids() const {
    const auto net_ids = batch_.pending_net_ids();
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(net_ids.size()));
    if (!net_ids.empty()) {
      auto* data = out.mutable_data();
      for (size_t i = 0; i < net_ids.size(); ++i) data[i] = net_ids[i];
    }
    return out;
  }

  py::tuple pending_legal_csr() const {
    const auto csr = batch_.pending_legal_csr();
    py::array_t<int> indices(static_cast<py::ssize_t>(csr.indices.size()));
    py::array_t<int> offsets(static_cast<py::ssize_t>(csr.offsets.size()));
    if (!csr.indices.empty()) std::memcpy(indices.mutable_data(), csr.indices.data(), csr.indices.size() * sizeof(int));
    std::memcpy(offsets.mutable_data(), csr.offsets.data(), csr.offsets.size() * sizeof(int));
    return py::make_tuple(indices, offsets);
  }

  void apply_eval(py::array_t<int, py::array::c_style | py::array::forcecast> ids,
                  py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                  py::array_t<float, py::array::c_style | py::array::forcecast> values) {
    if (ids.ndim() != 1 || logits.ndim() != 2 || logits.shape(1) != immortalite::POLICY_SIZE ||
        values.ndim() != 1 || ids.shape(0) != logits.shape(0) || ids.shape(0) != values.shape(0))
      throw std::invalid_argument("invalid eval batch shapes");
    std::vector<int> actor_ids(ids.data(), ids.data() + ids.shape(0));
    batch_.apply_eval(actor_ids, logits.data(), values.data(), static_cast<int>(ids.shape(0)));
  }

  void apply_eval_legal(py::array_t<int, py::array::c_style | py::array::forcecast> ids,
                        py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                        py::array_t<int, py::array::c_style | py::array::forcecast> offsets,
                        py::array_t<float, py::array::c_style | py::array::forcecast> values) {
    if (ids.ndim() != 1 || logits.ndim() != 1 || offsets.ndim() != 1 ||
        values.ndim() != 1 || offsets.shape(0) != ids.shape(0) + 1 || values.shape(0) != ids.shape(0))
      throw std::invalid_argument("invalid legal eval batch shapes");
    if (offsets.data()[0] != 0 ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<int>(logits.shape(0))) {
      throw std::invalid_argument("legal offsets must span legal logits");
    }
    std::vector<int> actor_ids(ids.data(), ids.data() + ids.shape(0));
    batch_.apply_eval_legal(actor_ids, logits.data(), offsets.data(), values.data(),
                            static_cast<int>(ids.shape(0)));
  }

  py::dict take_completed() {
    const auto games = batch_.take_completed();
    size_t sample_count = 0;
    for (const auto& game : games) sample_count += game.samples.size();
    py::array_t<float> planes(py::array::ShapeContainer{
        static_cast<py::ssize_t>(sample_count),
        static_cast<py::ssize_t>(immortalite::NUM_INPUT_PLANES),
        static_cast<py::ssize_t>(8), static_cast<py::ssize_t>(8)});
    py::array_t<float> policies(py::array::ShapeContainer{
        static_cast<py::ssize_t>(sample_count),
        static_cast<py::ssize_t>(immortalite::POLICY_SIZE)});
    py::array_t<int> players(static_cast<py::ssize_t>(sample_count));
    py::array_t<float> values(static_cast<py::ssize_t>(sample_count));
    py::array_t<float> root_q(static_cast<py::ssize_t>(sample_count));
    py::list metadata;
    size_t row = 0;
    for (const auto& game : games) {
      py::dict meta;
      meta["actor_id"] = game.actor_id;
      meta["sample_start"] = row;
      meta["sample_end"] = row + game.samples.size();
      meta["termination"] = game.termination;
      meta["winner"] = game.winner;
      meta["moves"] = game.moves;
      metadata.append(meta);
      for (const auto& sample : game.samples) {
        std::memcpy(planes.mutable_data(row, 0, 0, 0), sample.planes.data(), sample.planes.size() * sizeof(float));
        std::memcpy(policies.mutable_data(row, 0), sample.policy.data(), sample.policy.size() * sizeof(float));
        players.mutable_data()[row] = static_cast<int>(sample.player);
        values.mutable_data()[row] = sample.value;
        root_q.mutable_data()[row] = sample.root_q;
        ++row;
      }
    }
    py::dict out;
    out["planes"] = planes; out["policies"] = policies; out["players"] = players;
    out["values"] = values; out["root_q"] = root_q; out["games"] = metadata;
    return out;
  }

 private:
  static immortalite::GameActorConfig actor_config_from_dict(const py::dict& d) {
    immortalite::GameActorConfig cfg;
    auto get_i = [&](const char* key, int& value) { if (d.contains(key)) value = d[key].cast<int>(); };
    auto get_f = [&](const char* key, float& value) { if (d.contains(key)) value = d[key].cast<float>(); };
    auto get_b = [&](const char* key, bool& value) { if (d.contains(key)) value = d[key].cast<bool>(); };
    get_i("simulations", cfg.simulations); get_i("max_game_moves", cfg.max_game_moves);
    get_b("claim_draw", cfg.claim_draw); get_f("draw_contempt", cfg.draw_contempt);
    get_f("draw_penalty", cfg.draw_penalty); if (d.contains("value_target")) cfg.value_target = d["value_target"].cast<std::string>();
    get_f("value_q_ratio", cfg.value_q_ratio);
    get_f("resign_threshold", cfg.resign_threshold); get_i("resign_plies", cfg.resign_plies);
    get_i("resign_min_moves", cfg.resign_min_moves); get_f("move_temperature", cfg.move_temperature);
    get_i("move_temperature_plies", cfg.move_temperature_plies); get_i("exploration_moves", cfg.exploration_moves);
    get_i("tb_max_pieces", cfg.tb_max_pieces); get_f("fast_mate_bonus", cfg.fast_mate_bonus);
    get_b("add_noise", cfg.add_noise);
    return cfg;
  }

  immortalite::GameActorBatch batch_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Immortalite One native core (board, encoding, MCTS)";
  m.attr("ENCODING_VERSION") = immortalite::ENCODING_VERSION;
  m.attr("NUM_INPUT_PLANES") = immortalite::NUM_INPUT_PLANES;
  m.attr("POLICY_SIZE") = immortalite::POLICY_SIZE;

  m.def("version", []() { return "0.1.0"; }, "Native module version string");

  m.def("fill_planes_fen", &fill_planes_fen, py::arg("fen"),
        py::arg("moves") = py::none(),
        "Encode FEN to (20,8,8) float32 planes. Optional moves: UCI list applied after fen "
        "(builds repetition history).");
  m.def("legal_move_indices_fen", &legal_move_indices_fen, py::arg("fen"),
        "List of (policy_index, uci) for legal moves");
  m.def("move_to_index_fen", &move_to_index_fen, py::arg("fen"), py::arg("uci"),
        "Map UCI move to policy index");

  py::class_<PyMctsSession>(m, "MctsSession")
      .def(py::init<const std::string&, int, const py::dict&, bool,
                    const std::optional<std::vector<std::string>>&>(),
           py::arg("fen"), py::arg("simulations"), py::arg("config") = py::dict(),
           py::arg("add_noise") = false, py::arg("moves") = py::none(),
           "Search root = fen after optional UCI moves (history preserved for claim_draw).")
      .def("done", &PyMctsSession::done)
      .def("stats", &PyMctsSession::stats)
      .def("positions_needing_eval", &PyMctsSession::positions_needing_eval)
      .def("positions_needing_eval_into", &PyMctsSession::positions_needing_eval_into,
           py::arg("out"), py::arg("row"))
      .def("pending_legal_indices", &PyMctsSession::pending_legal_indices)
      .def("apply_eval", &PyMctsSession::apply_eval, py::arg("logits"), py::arg("values"))
      .def("apply_eval_legal", &PyMctsSession::apply_eval_legal,
           py::arg("legal_logits"), py::arg("value"))
      .def("result", &PyMctsSession::result);

  py::class_<PyGameActorBatch>(m, "GameActorBatch")
      .def(py::init<int, const py::dict&, const py::dict&, std::uint64_t, py::object, py::object>(),
           py::arg("game_count"), py::arg("actor_config") = py::dict(),
           py::arg("mcts_config") = py::dict(), py::arg("base_seed") = 0,
           py::arg("start_moves") = py::none(), py::arg("a_is_white") = py::none())
      .def("tablebase_requests", &PyGameActorBatch::tablebase_requests)
      .def("apply_tablebase", &PyGameActorBatch::apply_tablebase)
      .def("positions_needing_eval", &PyGameActorBatch::positions_needing_eval,
           py::arg("out") = py::none())
      .def("pending_net_ids", &PyGameActorBatch::pending_net_ids)
      .def("pending_legal_csr", &PyGameActorBatch::pending_legal_csr)
      .def("apply_eval", &PyGameActorBatch::apply_eval)
      .def("apply_eval_legal", &PyGameActorBatch::apply_eval_legal)
      .def("take_completed", &PyGameActorBatch::take_completed);
}
