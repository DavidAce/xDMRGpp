#include "impl.h"

using Scalar = cx32;

/* clang-format off */

template void tools::finite::mps::apply_gate(StateFinite<Scalar> &state, const qm::Gate &gate, GateOp gop, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_gates(StateFinite<Scalar> &state, const std::vector<Eigen::Tensor<Scalar, 2>> &nsite_tensors, size_t gate_size, CircuitOp cop, bool moveback, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_gates(StateFinite<Scalar> &state, const std::vector<qm::Gate> &gates, CircuitOp cop, bool moveback, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_circuit(StateFinite<Scalar> &state, const std::vector<std::vector<qm::Gate>> &circuit, CircuitOp cop, bool moveback, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::swap_sites(StateFinite<Scalar> &state, size_t posL, size_t posR, std::vector<size_t> &sites, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_swap_gate(StateFinite<Scalar> &state, const qm::SwapGate &gate, GateOp gop, std::vector<size_t> &sites, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_swap_gates(StateFinite<Scalar> &state, std::vector<qm::SwapGate> &gates, CircuitOp cop, GateMove gm, std::optional<svd::config> svd_cfg);

template void tools::finite::mps::apply_swap_gates(StateFinite<Scalar> &state, const std::vector<qm::SwapGate> &gates, CircuitOp cop, GateMove gm, std::optional<svd::config> svd_cfg);
