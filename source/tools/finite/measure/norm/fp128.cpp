#include "../norm.impl.h"

using Scalar = fp128;

/* clang-format off */

template RealScalar<Scalar>  tools::finite::measure::norm(const StateFinite<Scalar> &state, bool full);
