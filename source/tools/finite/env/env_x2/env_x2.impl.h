#pragma once
#include "../../env.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/linalg/matrix/to_string.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction/env.h"
#include "tools/common/log.h"

namespace settings {
    inline constexpr bool debug_edges = false;
}

template<typename Scalar>
void tools::finite::env::rebuild_edges_ene_x2(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_ene_x2: fLBIT algorithm should never rebuild energy edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto         t_reb   = tid::tic_scope("rebuild_edges_ene_x2", tid::higher);
    const size_t L       = state.template get_length<size_t>();
    const size_t min_pos = 0;
    const size_t max_pos = L - 1;

    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_ene_x2: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this rebuild after\n"
                                    " activating sites.");

    const long current_position = state.template get_position<long>();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    const size_t posL_active = edges.active_sites.back();
    const size_t posR_active = edges.active_sites.front();
    assert(posL_active < L && posL_active <= posR_active);
    assert(posR_active < L && posR_active >= posL_active);
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene_x2: pos {} | dir {} | "
                          "inspecting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    std::vector<size_t>   env_pos_log;
    x2::Tensor<Scalar, 3> new_x2, old_x2;

    { // Seed left boundary
        auto &env0 = edges.get_env_eneL(0);
        env0.set_edge_dims(state.get_mps_site(0), model.get_mpo(0));
        old_x2 = env0.get_blkx2();
    }
    const size_t stopL = std::min(posL_active, L - 1);
    for(size_t pos = min_pos; pos < stopL; pos++) {
        const auto &env_here = edges.get_env_eneL(pos);
        auto       &env_rght = edges.get_env_eneL(pos + 1);
        const auto &mps      = state.get_mps_site(pos);
        const auto &mpo      = model.get_mpo(pos);
        const auto &M        = mps.get_M_bare();
        const auto &W        = mpo.MPO();
        auto        id_here  = env_here.get_unique_id();
        auto        id_rght  = env_rght.get_unique_id();

        // Determine whether the block has already been built
        // We refresh this block if any of these conditions hold:
        //   unique_id_env != env.unique_id;
        //   unique_id_mps != mps.unique_id;
        //   unique_id_mpo != mpo.unique_id;

        bool env_stale = env_here.get_unique_id() != env_rght.get_unique_id_env().value_or(-1ul);
        bool mps_stale = mps.get_unique_id() != env_rght.get_unique_id_mps().value_or(-1ul);
        bool mpo_stale = mpo.get_unique_id() != env_rght.get_unique_id_mpo().value_or(-1ul);
        bool refresh   = env_stale or mps_stale or mpo_stale;

        if(refresh) {
            new_x2.resize(mps.get_chiR(), mps.get_chiR(), mpo.MPO().dimension(1));
            tools::common::contraction::contract_envL_mps_mpo(new_x2, old_x2, M, W);
            env_rght.set_blkx2(new_x2, env_here, mps, mpo);
            old_x2 = env_rght.get_blkx2();
        } else {
            old_x2 = env_rght.get_blkx2();
            old_x2.renorm();
        }
        env_rght.assert_unique_id(env_here, mps, mpo);
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene_x2: rebuilt eneL edges: {}", env_pos_log);

    env_pos_log.clear();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene_x2: pos {} | dir {} | "
                          "inspecting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    new_x2 = x2::Tensor<Scalar, 3>();
    old_x2 = x2::Tensor<Scalar, 3>();
    { // Seed right boundary (this is where set_edge_dims belongs)

        auto &envN = edges.get_env_eneR(L - 1);
        envN.set_edge_dims(state.get_mps_site(L - 1), model.get_mpo(L - 1));
        old_x2 = envN.get_blkx2();
    }
    const size_t stopR = std::min(posR_active, L - 1); // smallest valid envR index

    for(size_t pos = max_pos; pos > stopR; --pos) {
        const auto &env_here = edges.get_env_eneR(pos);
        auto       &env_left = edges.get_env_eneR(pos - 1);
        const auto &mps      = state.get_mps_site(pos);
        const auto &mpo      = model.get_mpo(pos);
        const auto &M        = mps.get_M_bare();
        const auto &W        = mpo.MPO();
        auto        id_here  = env_here.get_unique_id();
        auto        id_left  = env_left.get_unique_id();

        // Determine whether the block has already been built
        // We refresh this block if any of these conditions hold:
        //   unique_id_env != env.unique_id;
        //   unique_id_mps != mps.unique_id;
        //   unique_id_mpo != mpo.unique_id;

        bool env_stale = env_here.get_unique_id() != env_left.get_unique_id_env().value_or(-1ul);
        bool mps_stale = mps.get_unique_id() != env_left.get_unique_id_mps().value_or(-1ul);
        bool mpo_stale = mpo.get_unique_id() != env_left.get_unique_id_mpo().value_or(-1ul);
        bool refresh   = env_stale or mps_stale or mpo_stale;
        if(refresh) {
            new_x2.resize(mps.get_chiL(), mps.get_chiL(), mpo.MPO().dimension(0));
            tools::common::contraction::contract_envR_mps_mpo(new_x2, old_x2, M, W);
            env_left.set_blkx2(new_x2, env_here, mps, mpo);
            old_x2 = env_left.get_blkx2();
        } else {
            old_x2 = env_left.get_blkx2();
            old_x2.renorm();
        }
        env_left.assert_unique_id(env_here, mps, mpo);
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene_x2: rebuilt eneR edges: {}", env_pos_log);
    if(not edges.get_env_eneL(posL_active).has_block()) throw except::logic_error("rebuild_edges_ene_x2: active env eneL has undefined block");
    if(not edges.get_env_eneR(posR_active).has_block()) throw except::logic_error("rebuild_edges_ene_x2: active env eneR has undefined block");
}

template<typename Scalar>
void tools::finite::env::rebuild_edges_var_x2(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_var_x2: fLBIT algorithm should never rebuild energy edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto         t_reb   = tid::tic_scope("rebuild_edges_var_x2", tid::higher);
    const size_t L       = state.template get_length<size_t>();
    const size_t min_pos = 0;
    const size_t max_pos = L - 1;

    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_var_x2: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this rebuild after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    const size_t posL_active = edges.active_sites.back();
    const size_t posR_active = edges.active_sites.front();
    assert(posL_active < L && posL_active <= posR_active);
    assert(posR_active < L && posR_active >= posL_active);
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_var_x2: pos {} | dir {} | "
                          "inspecting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    std::vector<size_t>   env_pos_log;
    x2::Tensor<Scalar, 3> new_x2, old_x2;

    { // Seed left boundary
        auto &env0 = edges.get_env_varL(0);
        env0.set_edge_dims(state.get_mps_site(0), model.get_mpo(0));
        old_x2 = env0.get_blkx2();
    }
    const size_t stopL = std::min(posL_active, L - 1);
    for(size_t pos = min_pos; pos < stopL; pos++) {
        const auto &env_here = edges.get_env_varL(pos);
        auto       &env_rght = edges.get_env_varL(pos + 1);
        const auto &mps      = state.get_mps_site(pos);
        const auto &mpo      = model.get_mpo(pos);
        const auto &M        = mps.get_M_bare();
        const auto &W        = mpo.MPO2();
        auto        id_here  = env_here.get_unique_id();
        auto        id_rght  = env_rght.get_unique_id();

        // Determine whether the block has already been built
        // We refresh this block if any of these conditions hold:
        //   not has_block()
        //   unique_id_env != env.unique_id;
        //   unique_id_mps != mps.unique_id;
        //   unique_id_mpo != mpo.unique_id;

        bool env_stale = env_here.get_unique_id() != env_rght.get_unique_id_env().value_or(-1ul);
        bool mps_stale = mps.get_unique_id() != env_rght.get_unique_id_mps().value_or(-1ul);
        bool mpo_stale = mpo.get_unique_id_sq() != env_rght.get_unique_id_mpo().value_or(-1ul);
        bool refresh   = env_stale or mps_stale or mpo_stale;
        if(refresh) {
            new_x2.resize(mps.get_chiR(), mps.get_chiR(), mpo.MPO2().dimension(1));
            tools::common::contraction::contract_envL_mps_mpo(new_x2, old_x2, M, W);
            env_rght.set_blkx2(new_x2, env_here, mps, mpo);
            old_x2 = env_rght.get_blkx2();
        } else {
            old_x2 = env_rght.get_blkx2();
            old_x2.renorm();
        }
        env_rght.assert_unique_id(env_here, mps, mpo);
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var_x2: rebuilt varL edges: {}", env_pos_log);

    env_pos_log.clear();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_var_x2: pos {} | dir {} | "
                          "inspecting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    new_x2 = x2::Tensor<Scalar, 3>();
    old_x2 = x2::Tensor<Scalar, 3>();

    { // Seed right boundary
        auto &envN = edges.get_env_varR(L - 1);
        envN.set_edge_dims(state.get_mps_site(L - 1), model.get_mpo(L - 1));
        old_x2 = envN.get_blkx2();
    }
    const size_t stopR = std::min(posR_active, L - 1); // smallest valid envR index
    for(size_t pos = max_pos; pos > stopR; --pos) {
        const auto &env_here = edges.get_env_varR(pos);
        auto       &env_left = edges.get_env_varR(pos - 1);
        const auto &mps      = state.get_mps_site(pos);
        const auto &mpo      = model.get_mpo(pos);
        const auto &M        = mps.get_M_bare();
        const auto &W        = mpo.MPO2();
        auto        id_here  = env_here.get_unique_id();
        auto        id_left  = env_left.get_unique_id();

        // Determine whether the block has already been built
        // We refresh this block if any of these conditions hold:
        //   not has_block()
        //   unique_id_env != env.unique_id;
        //   unique_id_mps != mps.unique_id;
        //   unique_id_mpo != mpo.unique_id;

        bool env_stale = env_here.get_unique_id() != env_left.get_unique_id_env().value_or(-1ul);
        bool mps_stale = mps.get_unique_id() != env_left.get_unique_id_mps().value_or(-1ul);
        bool mpo_stale = mpo.get_unique_id_sq() != env_left.get_unique_id_mpo().value_or(-1ul);
        bool refresh   = env_stale or mps_stale or mpo_stale;
        if(refresh) {
            new_x2.resize(mps.get_chiL(), mps.get_chiL(), mpo.MPO2().dimension(0));
            tools::common::contraction::contract_envR_mps_mpo(new_x2, old_x2, M, W);
            env_left.set_blkx2(new_x2, env_here, mps, mpo);
            old_x2 = env_left.get_blkx2();

        } else {
            old_x2 = env_left.get_blkx2();
            old_x2.renorm();
        }
        env_left.assert_unique_id(env_here, mps, mpo);
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var_x2: rebuilt varR edges: {}", env_pos_log);
    if(not edges.get_env_varL(posL_active).has_block()) throw except::logic_error("rebuild_edges_var_x2: active env varL has undefined block");
    if(not edges.get_env_varR(posR_active).has_block()) throw except::logic_error("rebuild_edges_var_x2: active env varR has undefined block");
}
