#include "board.hpp"
#include "encoding.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "types.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    // Contiguous copies for pointer API
    std::vector<float> lbuf(static_cast<size_t>(n * immortalite::POLICY_SIZE));
    std::vector<float> vbuf(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      vbuf[static_cast<size_t>(i)] = v(i);
      for (int j = 0; j < immortalite::POLICY_SIZE; ++j)
        lbuf[static_cast<size_t>(i * immortalite::POLICY_SIZE + j)] = l(i, j);
    }
    session_->apply_eval(lbuf.data(), vbuf.data(), n);
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
    return d;
  }

 private:
  std::unique_ptr<immortalite::MctsSession> session_;
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
      .def("positions_needing_eval", &PyMctsSession::positions_needing_eval)
      .def("apply_eval", &PyMctsSession::apply_eval, py::arg("logits"), py::arg("values"))
      .def("result", &PyMctsSession::result);
}
