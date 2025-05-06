#include "impl.h"

using Scalar = cx128;

/* clang-format off */

template size_t              tools::infinite::measure::length(const TensorsInfinite<Scalar> &tensors);
template size_t              tools::infinite::measure::length(const EdgesInfinite<Scalar> &edges);
template RealScalar<Scalar>  tools::infinite::measure::norm(const StateInfinite<Scalar> &state);
template long                tools::infinite::measure::bond_dimension(const StateInfinite<Scalar> &state);
template double              tools::infinite::measure::truncation_error(const StateInfinite<Scalar> &state);
template RealScalar<Scalar>  tools::infinite::measure::entanglement_entropy(const StateInfinite<Scalar> &state);
