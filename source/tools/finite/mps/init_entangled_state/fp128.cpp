#include "impl.h"

using Scalar = fp128;

/* clang-format off */

template void tools::finite::mps::init::set_midchain_singlet_neel_state(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_random_entangled_state_haar(StateFinite<Scalar> &state, StateInitType type, long bond_lim);

template void tools::finite::mps::init::random_entangled_state(StateFinite<Scalar> &state, StateInitType type, [[maybe_unused]] std::string_view axis, [[maybe_unused]] bool use_eigenspinors, long bond_lim);

template void tools::finite::mps::init::set_random_entangled_state_with_random_spinors(StateFinite<Scalar> &state, StateInitType type, long bond_lim);

template void tools::finite::mps::init::set_random_entangled_state_on_axes_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type, const std::vector<std::string> &axes, long bond_lim);

template void tools::finite::mps::init::set_random_entangled_state_on_axis_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, long bond_lim);

template void tools::finite::mps::init::randomize_given_state(StateFinite<Scalar> &state, StateInitType type);
