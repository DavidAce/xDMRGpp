#include "../number_entropy.impl.h"

using Scalar = fp128;

/* clang-format off */

template std::vector<RealScalar<Scalar>>  tools::finite::measure::number_entropies(const StateFinite<Scalar> &state);

template RealScalar<Scalar>  tools::finite::measure::number_entropy_current(const StateFinite<Scalar> &state);

template RealScalar<Scalar>  tools::finite::measure::number_entropy_midchain(const StateFinite<Scalar> &state);

/* clang-format on */
