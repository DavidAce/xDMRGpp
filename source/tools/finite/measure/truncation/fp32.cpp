#include "../truncation.impl.h"

using Scalar = fp32;

/* clang-format off */
template std::vector<fp64> tools::finite::measure::truncation_errors(const StateFinite<Scalar> &state);

template std::vector<fp64> tools::finite::measure::truncation_errors_active(const StateFinite<Scalar> &state);
/* clang-format on */