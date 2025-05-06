#include "../split.impl.h"

using Scalar = cx64;
using T = cx64;

/* clang-format off */
template std::vector<MpsSite<Scalar>>  tools::common::split::split_mps(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, std::optional<svd::config> svd_cfg);

template std::vector<MpsSite<Scalar>> tools::common::split::internal::split_mps_into_As(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);

template std::deque<MpsSite<Scalar>> tools::common::split::internal::split_mps_into_Bs(const Eigen::Tensor<T, 3> &multisite_mps, const std::vector<long> &spin_dims, const std::vector<size_t> &positions, long center_position, svd::config &svd_cfg);
