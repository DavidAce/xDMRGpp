#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template RealScalar<Scalar>  tools::infinite::measure::energy_per_site_ham(const TensorsInfinite<Scalar> &tensors);
template RealScalar<Scalar>  tools::infinite::measure::energy_variance_per_site_ham(const TensorsInfinite<Scalar> &tensors);
