#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template size_t tools::finite::mps::merge_multisite_mps(StateFinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<size_t> &sites, long center_position, MergeEvent mevent, std::optional<svd::config> svd_cfg, std::optional<LogPolicy> logPolicy);
