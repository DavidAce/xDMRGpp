#pragma once
#include "config/settings.h"
#include "debug/info.h"
#include "expectation_value/contract.h"
#include "io/fmt_custom.h"
#include "math/tenx.h"
#include "norm.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include <algorithm>

using tools::finite::measure::RealScalar;

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::norm_1site(const StateFinite<Scalar> &state) {
    // if(state.measurements.norm) return state.measurements.norm.value();

    auto t_norm = tid::tic_scope("norm", tid::level::highest);
    // We know the all sites are normalized. We can check that the current position is normalized
    const auto  pos = std::clamp(state.template get_position<long>(), 0l, state.template get_length<long>() - 1l);
    const auto &mps = state.get_mps_site(pos);
    tools::log->trace("Measuring norm using site {} with dimensions {}", pos, mps.dimensions());
    Scalar norm = tools::common::contraction::contract_mps_norm(mps.get_M());

    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    auto normErr = std::abs(norm - RealScalar<Scalar>{1});

    if(normErr > normTol) tools::log->debug("norm_1site: far from unity: {:.5e}", fp(normErr));
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::norm_state(const StateFinite<Scalar> &state) {
    if(state.measurements.norm) return state.measurements.norm.value();
    auto t_norm = tid::tic_scope("norm", tid::level::highest);
    tools::log->trace("Measuring norm on the full chain");
    Eigen::Tensor<Scalar, 2> chain;
    Eigen::Tensor<Scalar, 2> temp;
    bool                     first   = true;
    auto                    &threads = tenx::threads::get();

    for(const auto &mps : state.mps_sites) {
        const auto &M = mps->get_M();
        if(first) {
            chain = tools::common::contraction::contract_mps_partial<std::array{0l, 1l}>(M);
            first = false;
            continue;
        }
        chain = contract_chain_M_Mconj_0_1_01_10(chain, M, threads);
    }
    Scalar norm = tenx::MatrixMap(chain).trace();

    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    auto normErr = std::abs(norm - RealScalar<Scalar>{1});

    if(normErr > normTol) tools::log->debug("norm_state: far from unity: {:.5e}", fp(normErr));
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}

template<typename Scalar>
Eigen::Tensor<Scalar, 2> tools::finite::measure::isometry_left(const StateFinite<Scalar> &state, Eigen::Index pos) {
    auto t_iso = tid::tic_scope("isometry_left", tid::level::highest);

    const auto L = state.template get_length<Eigen::Index>();
    if(L == 0) return Eigen::Tensor<Scalar, 2>();

    // Clamp pos into a sensible range (assumes positions are 0..L-1)
    pos = std::clamp<Eigen::Index>(pos, 0, L - 1);

    auto &threads = tenx::threads::get();

    // Start with identity on the left boundary bond dimension of the first site
    const auto &M0  = state.mps_sites.front()->get_M();
    const auto  chi = M0.dimension(1);

    Eigen::Tensor<Scalar, 2> chain(chi, chi);
    chain.setZero();
    for(Eigen::Index i = 0; i < chi; ++i) chain(i, i) = Scalar{1};

    Eigen::Tensor<Scalar, 2> temp;

    for(const auto &mps : state.mps_sites) {
        const auto  mps_pos = mps->template get_position<Eigen::Index>();
        const auto &M       = mps->get_M();

        // Sanity checks
        assert(chain.dimension(0) == chain.dimension(1));
        assert(chain.dimension(0) == M.dimension(1));

        chain = contract_chain_M_Mconj_0_1_01_10(chain, M, threads);

        // Stop after including site `pos`
        if(mps_pos >= pos) break;
    }

    return chain;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 2> tools::finite::measure::isometry_right(const StateFinite<Scalar> &state, Eigen::Index pos) {
    auto t_iso = tid::tic_scope("isometry_right", tid::level::highest);

    const auto L = state.template get_length<Eigen::Index>();
    if(L == 0) return Eigen::Tensor<Scalar, 2>();

    // Clamp pos into a sensible range (assumes positions are 0..L-1)
    pos = std::clamp<Eigen::Index>(pos, 0, L - 1);

    auto &threads = tenx::threads::get();

    // Start with identity on the right boundary bond dimension of the last site
    const auto &ML  = state.mps_sites.back()->get_M();
    const auto  chi = ML.dimension(2);

    Eigen::Tensor<Scalar, 2> chain(chi, chi);
    chain.setZero();
    for(Eigen::Index i = 0; i < chi; ++i) chain(i, i) = Scalar{1};

    Eigen::Tensor<Scalar, 2> temp;

    // Iterate from right to left
    for(auto it = state.mps_sites.rbegin(); it != state.mps_sites.rend(); ++it) {
        const auto &mps     = *it;
        const auto  mps_pos = mps->template get_position<Eigen::Index>();
        const auto &M       = mps->get_M();

        // Sanity checks
        assert(chain.dimension(0) == chain.dimension(1));
        assert(chain.dimension(0) == M.dimension(2));
        chain = contract_chain_M_Mconj_0_2_01_20(chain, M, threads);

        // Stop after including site `pos`
        if(mps_pos <= pos) break;
    }

    return chain;
}