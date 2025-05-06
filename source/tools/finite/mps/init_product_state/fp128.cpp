#include "impl.h"

using Scalar = fp128;

/* clang-format off */

template void tools::finite::mps::init::random_product_state(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, bool use_eigenspinors, std::string &pattern);

template void tools::finite::mps::init::set_product_state_neel_shuffled(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_product_state_neel_dislocated(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_product_state_domain_wall(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_product_state_aligned(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_product_state_neel(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_random_product_state_with_random_spinors(StateFinite<Scalar> &state, StateInitType type, std::string &pattern);

template void tools::finite::mps::init::set_product_state_on_axis_using_pattern(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_sum_of_random_product_states(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_random_product_state_on_axis_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);

template void tools::finite::mps::init::set_random_product_state_on_axis(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern);



