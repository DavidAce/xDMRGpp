#pragma once

#include "math/tenx.h"
// -- (textra first)
#include "../../mps.h"
#include "config/enums.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/svd.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/norm.h"

namespace settings {
    static constexpr bool debug_normalization = true;
}

using tools::finite::mps::RealScalar;

template<typename Scalar>
bool tools::finite::mps::normalize_state(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy) {
    // When a state needs to be normalized it's enough to "move" the center position around the whole chain.
    // Each move performs an SVD decomposition which leaves unitaries behind, effectively normalizing the state.
    // NOTE! It IS important to start with the current position.
    constexpr auto eps                = std::numeric_limits<RealScalar<Scalar>>::epsilon();
    const auto     slack              = static_cast<RealScalar<Scalar>>(settings::precision::max_norm_slack);
    const auto     normErrorTolerance = eps * slack;
    const auto     normErrorTrigger   = eps * std::sqrt(slack);
    if(norm_policy == NormPolicy::IFNEEDED) {
        // We may only go ahead with a normalization if it's really needed.
        if(state.is_normalized_on_all_sites(normErrorTrigger)) {
            tools::log->trace("normalize_state: not needed");
            return false; // Return false, i.e. did "not" perform a normalization.
        }
        // Otherwise, we just do the normalization
    }

    // Save the current position, direction and center status
    auto dir   = state.get_direction();
    auto pos   = state.template get_position<long>();
    auto cnt   = pos >= 0;
    auto steps = 0;
    if(tools::log->level() <= spdlog::level::debug)
        tools::log->debug("normalize_state: {} old local norm = {:.16f} | pos {} | dir {} | bond dims {}", enum2sv(norm_policy),
                          fp(tools::finite::measure::norm_state(state)), pos, dir, tools::finite::measure::bond_dimensions(state));

    // Start with SVD at the current center position
    // NOTE: You have thought that this is unnecessary and removed it, only to find bugs much later.
    //       In particular, the bond dimension will shrink too much when doing projections, if this step is skipped.
    //       This makes sure chiL and chiR differ at most by factor spin_dim when we start the normalization
    if(pos >= 0) {
        auto &mps = state.get_mps_site(pos);
        // Make sure that the bond dimension does not increase faster than spin_dim per site
        tools::finite::mps::merge_multisite_mps(state, mps.get_M(), {static_cast<size_t>(pos)}, pos, MergeEvent::NORM, svd_cfg, LogPolicy::SILENT);
    }
    // Now we can move around the chain until we return to the original status
    constexpr int maxtrials = 3;
    for(int trial = 0; trial < maxtrials; ++trial) {
        if constexpr(settings::debug_normalization) {
            auto norm = tools::finite::measure::norm_state(state);
            tools::log->debug("Normalization trial {}/{} | current state norm {:.16f}", trial, maxtrials, fp(norm));
        }
        while(steps++ < 2 or not state.position_is_at(pos, dir, cnt)) move_center_point_single_site(state, svd_cfg);
        state.assert_validity();
        state.clear_measurements();
        state.clear_cache();
        if(state.is_normalized_on_all_sites(normErrorTolerance)) break;
    }

    if(not state.is_normalized_on_all_sites(normErrorTolerance)) {
        auto norm_state = tools::finite::measure::norm_state(state);
        auto norm_1site = tools::finite::measure::norm_1site(state);
        auto norm_error = std::abs(RealScalar<Scalar>(1) - norm_state);
        for(const auto &mps : state.mps_sites) {
            tools::log->warn("{} | norm  err {:.16f} | is_normalized {:<7} | L norm {:.16f}", mps->get_tag(), fp(mps->get_norm_error()),
                             mps->is_normalized(normErrorTolerance), fp(tenx::VectorMap(mps->get_L()).norm()));
            if(mps->isCenter()) tools::log->warn("LC({}) | norm {:.16f}", mps->get_position(), fp(tenx::VectorMap(mps->get_LC()).norm()));
        }
        throw except::runtime_error(
            "normalize_state: normalization failed. state norm {:.16f} | site norm {:.16f} | norm error {:.5e} | max allowed norm error {:.5e}", fp(norm_state),
            fp(norm_1site), fp(norm_error), fp(normErrorTolerance));
    }
    if(svd_cfg and svd_cfg->rank_max and state.get_largest_bond() > svd_cfg->rank_max.value())
        throw except::logic_error("normalize_state: a bond dimension exceeds bond limit: {} > {}", tools::finite::measure::bond_dimensions(state),
                                  svd_cfg->rank_max.value());
    if(tools::log->level() <= spdlog::level::debug) {
        auto norm = tools::finite::measure::norm_state(state);
        tools::log->debug("normalize_state: new local norm = {:.16f} | pos {} | dir {} | bond dims {}", fp(norm), pos, dir,
                          tools::finite::measure::bond_dimensions(state));
    }
    return true;
}
