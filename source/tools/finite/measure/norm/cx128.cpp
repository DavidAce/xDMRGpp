#include "../norm.impl.h"

using Scalar = cx128;

/* clang-format off */

template RealScalar<Scalar>  tools::finite::measure::norm_1site(const StateFinite<Scalar> &state);
template RealScalar<Scalar>  tools::finite::measure::norm_state(const StateFinite<Scalar> &state);