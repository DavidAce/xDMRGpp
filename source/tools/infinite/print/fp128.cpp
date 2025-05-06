#include "impl.h"

using Scalar = fp128;

/* clang-format off */

template void tools::infinite::print::print_hamiltonians(const ModelInfinite<Scalar> &model);
template void tools::infinite::print::dimensions(const TensorsInfinite<Scalar> &tensors);
