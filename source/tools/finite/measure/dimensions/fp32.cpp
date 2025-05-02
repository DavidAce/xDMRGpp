#include "../dimensions.impl.h"

using Scalar = fp32;

/* clang-format off */

template long tools::finite::measure::bond_dimension_current(const StateFinite<Scalar> &state);

template long tools::finite::measure::bond_dimension_midchain(const StateFinite<Scalar> &state);

template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<Scalar> &state, size_t pos);

template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<Scalar> &state);

template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<Scalar> &state);

template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<Scalar> &state);

/* clang-format on */