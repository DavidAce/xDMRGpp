#include "../../EdgesFinite.impl.h"

using Scalar = fp64;
using T      = fp32;

template env_pair<Eigen::Tensor<T, 3>>  EdgesFinite<Scalar>::get_env_ene_blk_as<T>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<T, 3>>  EdgesFinite<Scalar>::get_env_var_blk_as<T>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<T, 3>>  EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<T>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<T, 3>>  EdgesFinite<Scalar>::get_multisite_env_var_blk_as<T>(std::optional<std::vector<size_t>> sites) const;

