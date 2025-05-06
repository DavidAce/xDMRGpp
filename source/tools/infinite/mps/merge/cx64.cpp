#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template void tools::infinite::mps::merge_twosite_tensor(StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent, std::optional<svd::config> svd_cfg);
