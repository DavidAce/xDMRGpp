#include "impl.h"

using Scalar = fp32;

/* clang-format off */
template void tools::finite::print::dimensions(const TensorsFinite<Scalar> &tensors);

template void tools::finite::print::model(const ModelFinite<Scalar> &model);
