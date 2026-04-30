#pragma once
#include "../residual.h"
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
#include "tools/common/contraction/matrix_vector_product.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/ops.h"
#include <algorithm>
#include <cmath>
#include <string_view>

using tools::finite::measure::RealScalar;

namespace {
    template<typename Scalar>
    bool sites_are_full_system(const std::vector<size_t> &sites, const StateFinite<Scalar> &state) {
        if(sites.empty()) return true;
        if(sites.size() != state.template get_length<size_t>()) return false;
        const auto full_sites = num::range<size_t>(0, state.get_length());
        return std::equal(sites.begin(), sites.end(), full_sites.begin());
    }

    template<typename Scalar>
    void throw_large_partial_residual(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, std::string_view name) {
        constexpr size_t full_tensor_site_limit = 12;
        if(sites.size() <= full_tensor_site_limit) return;
        if(sites_are_full_system(sites, state)) return;
        throw except::runtime_error(
            "{} requested {} sites, which is too large for the full-multisite tensor route. Zip-up residuals are only available for the "
            "full system; use fewer sites or request the full system explicitly.",
            name, sites.size());
    }

    template<typename Scalar>
    void throw_large_full_h2_residual(const std::vector<size_t> &sites, const StateFinite<Scalar> &state) {
        constexpr size_t full_tensor_site_limit = 12;
        if(sites.size() <= full_tensor_site_limit) return;
        if(not sites_are_full_system(sites, state)) return;
        throw except::runtime_error(
            "residual_norm_H2 requested the full system over {} sites, which is too large for the full-multisite tensor route. There is no "
            "H2 zip-up residual because the current MPO2 shift API squares a shifted H instead of shifting H2 after squaring.",
            sites.size());
    }

    inline svd::config default_residual_zipup_svd_config() {
        auto svd_cfg    = svd::config(8192, 1e-20);
        svd_cfg.svd_lib = svd::lib::lapack;
        svd_cfg.svd_rtn = svd::rtn::gesdd;
        return svd_cfg;
    }

    template<typename Scalar>
    RealScalar<Scalar> sqrt_nonnegative(RealScalar<Scalar> value, RealScalar<Scalar> scale, std::string_view name) {
        using Real = RealScalar<Scalar>;
        if(value < Real{0}) {
            const auto tol = Real{1000} * std::numeric_limits<Real>::epsilon() * std::max(scale, Real{1});
            if(std::abs(value) > tol) tools::log->warn("{} produced a negative norm squared {:.16e}; clamping to zero", name, value);
            value = Real{0};
        }
        return std::sqrt(value);
    }
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo,
                                                         const x2::Tensor<Scalar, 3> &envL, const x2::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    auto Hv = tools::common::contraction::matrix_vector_product(mps, mpo, envL, envR);
    auto E  = tools::common::contraction::contract_mps_overlap(mps, Hv);
    return (tenx::VectorMap(Hv) - E * tenx::VectorMap(mps)).norm();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3> &mps, const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                         const x2::Tensor<Scalar, 3> &envL, const x2::Tensor<Scalar, 3> &envR) {
    // Calculate the residual_norm r = |Hv - Ev|
    if(mpos.size() == 1) return residual_norm(mps, mpos.front(), envL, envR);

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
    mpo_vec.reserve(mpo_refs.size());
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_blkx2_as<Scalar>(), envs.R.template get_blkx2_as<Scalar>());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                         const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                         const env_pair<const EnvVar<Scalar> &>                           &envs) {
    // Calculate the residual_norm r = |H²v - E²v|
    auto mpo_vec = std::vector<Eigen::Tensor<Scalar, 4>>();
    mpo_vec.reserve(mpo_refs.size());
    for(const auto &mpo : mpo_refs) mpo_vec.emplace_back(mpo.get().template MPO2_as<Scalar>());
    return residual_norm(mps, mpo_vec, envs.L.template get_blkx2_as<Scalar>(), envs.R.template get_blkx2_as<Scalar>());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H1(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(sites.empty()) return residual_norm_zip_up(state, model, edges, default_residual_zipup_svd_config());
    throw_large_partial_residual(sites, state, "residual_norm_H1");
    if(sites_are_full_system(sites, state) and sites.size() > 12) return residual_norm_zip_up(state, model, edges, default_residual_zipup_svd_config());
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
    tensors.assert_edges_ene();
    if(auto cache = tensors.measurements.get_cached_residual_norm_h1(tensors); cache) return cache->value();
    auto value = residual_norm_H1<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
    tensors.measurements.set_cached_residual_norm_h1(value, tensors);
    return value;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_zip_up(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                const EdgesFinite<Scalar> &edges, const svd::config &svd_cfg) {
    // Zip-up application of (H - <H>) to the MPS. This avoids forming the full d^L state vector used by residual_norm_full.
    auto       residual_state  = state;
    const auto energy          = tools::finite::measure::energy(state, model, edges);
    const auto length          = state.template get_length<RealScalar<Scalar>>();
    const auto energy_per_site = static_cast<Scalar>(energy / length);
    const auto mpos_shifted    = model.get_mpo_tensors(energy_per_site, MposWithEdges::ON, MpoCompress::DPL);
    tools::finite::ops::apply_mpos_general(residual_state, mpos_shifted, svd_cfg);
    const auto norm2 = tools::finite::measure::norm_state(residual_state);
    return sqrt_nonnegative<Scalar>(norm2, norm2, "residual_norm_zip_up");
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_zip_up(const TensorsFinite<Scalar> &tensors, const svd::config &svd_cfg) {
    tensors.assert_edges_ene();
    // Use energy(tensors) so the measurement cache remains the source of truth when the value is already known.
    auto       residual_state  = tensors.get_state();
    const auto energy          = tools::finite::measure::energy(tensors);
    const auto length          = tensors.template get_length<RealScalar<Scalar>>();
    const auto energy_per_site = static_cast<Scalar>(energy / length);
    const auto mpos_shifted    = tensors.get_model().get_mpo_tensors(energy_per_site, MposWithEdges::ON, MpoCompress::DPL);
    tools::finite::ops::apply_mpos_general(residual_state, mpos_shifted, svd_cfg);
    const auto norm2 = tools::finite::measure::norm_state(residual_state);
    return sqrt_nonnegative<Scalar>(norm2, norm2, "residual_norm_zip_up");
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                            const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    const auto residual_sites = sites.empty() ? num::range<size_t>(0, state.get_length()) : sites;
    throw_large_partial_residual(residual_sites, state, "residual_norm_H2");
    throw_large_full_h2_residual(residual_sites, state);
    const auto &mps = state.template get_multisite_mps<Scalar>(residual_sites);
    const auto &mpo = model.get_mpo(residual_sites);
    const auto &env = edges.get_multisite_env_var(residual_sites);
    return residual_norm(mps, mpo, env);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const std::vector<size_t> &sites, const TensorsFinite<Scalar> &tensors) {
    return residual_norm_H2<Scalar>(sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_H2(const TensorsFinite<Scalar> &tensors) {
    tensors.assert_edges_var();
    if(auto cache = tensors.measurements.get_cached_residual_norm_h2(tensors); cache) return cache->value();
    auto value = residual_norm_H2<Scalar>(tensors.active_sites, tensors.get_state(), tensors.get_model(), tensors.get_edges());
    tensors.measurements.set_cached_residual_norm_h2(value, tensors);
    return value;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                              const EdgesFinite<Scalar> &edges) {
    // Calculate the residual_norm r = |Hv - Ev|, where H is the full Hamiltonian and v is the full mps
    // Note that the full residual norm is equal to the sqrt(Var(H)) = Std(H)
    tools::log->info("Calculating residual norm with full system");
    auto sites = num::range<size_t>(0, state.get_length());
    throw_large_partial_residual(sites, state, "residual_norm_full");
    if(sites.size() > 12) return residual_norm_zip_up(state, model, edges, default_residual_zipup_svd_config());
    const auto &mps = state.template get_multisite_mps<Scalar>(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    return residual_norm<Scalar>(mps, mpo, env);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::residual_norm_full(const TensorsFinite<Scalar> &tensors) {
    tensors.assert_edges_ene();
    if(auto cache = tensors.measurements.get_cached_residual_norm_full(tensors); cache) return cache->value();
    auto value = residual_norm_full<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
    tensors.measurements.set_cached_residual_norm_full(value, tensors);
    return value;
}
