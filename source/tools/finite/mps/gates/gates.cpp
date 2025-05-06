#include "impl.h"

/* clang-format off */

template std::vector<size_t> tools::finite::mps::generate_gate_sequence(long state_len, long state_pos, const std::vector<qm::Gate> &gates, CircuitOp cop, bool range_long_to_short);

template std::vector<size_t> tools::finite::mps::generate_gate_sequence(long state_len, long state_pos, const std::vector<qm::SwapGate> &gates, CircuitOp cop, bool range_long_to_short);

