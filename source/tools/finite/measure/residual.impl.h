#pragma once
#include "residual.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

using tools::finite::measure::RealScalar;

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo,
                                                         const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpo, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}



template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                         const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpos, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                         const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                         const env_pair<const EnvEne<Scalar> &>                           &envs) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_block_as<Scalar>(), envs.R.template get_block_as<Scalar>());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                         const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                         const env_pair<const EnvVar<Scalar> &>                           &envs) {
    // Calculate the residual_norm r = |H²v - E²v|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO2_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_block_as<Scalar>(), envs.R.template get_block_as<Scalar>());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    return residual_norm(mps, mpo, env);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H1<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H1<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_var(sites);
    return residual_norm(mps, mpo, env);
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H2<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H2<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                              const EdgesFinite<Scalar> &edges) {
    // Calculate the residual_norm r = |Hv - Ev|, where H is the full Hamiltonian and v is the full mps
    // Note that the full residual norm is equal to the sqrt(Var(H)) = Std(H)
    tools::log->info("Calculating residual norm with full system");
    auto        sites = num::range<size_t>(0, state.get_length());
    const auto &mps   = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo   = model.get_mpo(sites);
    const auto &env   = edges.get_multisite_env_ene(sites);
    return residual_norm<Scalar>(mps, mpo, env);
}


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const TensorsFinite<Scalar> &tensors) {
    return residual_norm_full<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
