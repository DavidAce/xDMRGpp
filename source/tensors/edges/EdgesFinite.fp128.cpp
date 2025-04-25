#include "EdgesFinite.impl.h"

using Scalar = fp128;

template env_pair<Eigen::Tensor<fp32, 3>>  EdgesFinite<Scalar>::get_env_ene_blk_as<fp32>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<fp64, 3>>  EdgesFinite<Scalar>::get_env_ene_blk_as<fp64>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<fp128, 3>> EdgesFinite<Scalar>::get_env_ene_blk_as<fp128>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx32, 3>>  EdgesFinite<Scalar>::get_env_ene_blk_as<cx32>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx64, 3>>  EdgesFinite<Scalar>::get_env_ene_blk_as<cx64>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx128, 3>> EdgesFinite<Scalar>::get_env_ene_blk_as<cx128>(size_t posL, size_t posR) const;

template env_pair<Eigen::Tensor<fp32, 3>>  EdgesFinite<Scalar>::get_env_var_blk_as<fp32>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<fp64, 3>>  EdgesFinite<Scalar>::get_env_var_blk_as<fp64>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<fp128, 3>> EdgesFinite<Scalar>::get_env_var_blk_as<fp128>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx32, 3>>  EdgesFinite<Scalar>::get_env_var_blk_as<cx32>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx64, 3>>  EdgesFinite<Scalar>::get_env_var_blk_as<cx64>(size_t posL, size_t posR) const;
template env_pair<Eigen::Tensor<cx128, 3>> EdgesFinite<Scalar>::get_env_var_blk_as<cx128>(size_t posL, size_t posR) const;

template env_pair<Eigen::Tensor<fp32, 3>>  EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<fp32>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<fp64, 3>>  EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<fp64>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<fp128, 3>> EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<fp128>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx32, 3>>  EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<cx32>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx64, 3>>  EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<cx64>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx128, 3>> EdgesFinite<Scalar>::get_multisite_env_ene_blk_as<cx128>(std::optional<std::vector<size_t>> sites) const;

template env_pair<Eigen::Tensor<fp32, 3>>  EdgesFinite<Scalar>::get_multisite_env_var_blk_as<fp32>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<fp64, 3>>  EdgesFinite<Scalar>::get_multisite_env_var_blk_as<fp64>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<fp128, 3>> EdgesFinite<Scalar>::get_multisite_env_var_blk_as<fp128>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx32, 3>>  EdgesFinite<Scalar>::get_multisite_env_var_blk_as<cx32>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx64, 3>>  EdgesFinite<Scalar>::get_multisite_env_var_blk_as<cx64>(std::optional<std::vector<size_t>> sites) const;
template env_pair<Eigen::Tensor<cx128, 3>> EdgesFinite<Scalar>::get_multisite_env_var_blk_as<cx128>(std::optional<std::vector<size_t>> sites) const;