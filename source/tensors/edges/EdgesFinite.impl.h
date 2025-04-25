#pragma once
#include "EdgesFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvPair.h"
#include "tensors/site/env/EnvVar.h"
template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesFinite<Scalar>::get_env_ene_blk_as(size_t posL, size_t posR) const {
    return {get_env_ene(posL).L.template get_block_as<T>(), get_env_ene(posR).R.template get_block_as<T>()};
}

template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesFinite<Scalar>::get_env_var_blk_as(size_t posL, size_t posR) const {
    return {get_env_var(posL).L.template get_block_as<T>(), get_env_var(posR).R.template get_block_as<T>()};
}

template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesFinite<Scalar>::get_multisite_env_ene_blk_as(std::optional<std::vector<size_t>> sites) const {
    const auto &envs = get_multisite_env_ene(std::move(sites));
    return {envs.L.template get_block_as<T>(), envs.R.template get_block_as<T>()};
}

template<typename Scalar>
template<typename T>
env_pair<Eigen::Tensor<T, 3>> EdgesFinite<Scalar>::get_multisite_env_var_blk_as(std::optional<std::vector<size_t>> sites) const {
    const auto &envs = get_multisite_env_var(std::move(sites));
    return {envs.L.template get_block_as<T>(), envs.R.template get_block_as<T>()};
}
