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

using tools::finite::mps::RealScalar;

template<typename Scalar>
bool tools::finite::mps::normalize_state(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy) {
    // When a state needs to be normalized it's enough to "move" the center position around the whole chain.
    // Each move performs an SVD decomposition which leaves unitaries behind, effectively normalizing the state.
    // NOTE! It IS important to start with the current position.

    if(norm_policy == NormPolicy::IFNEEDED) {
        // We may only go ahead with a normalization if it's really needed.
        tools::log->trace("normalize_state: checking if needed");
        if(state.is_normalized_on_all_sites()) return false; // Return false, i.e. did "not" perform a normalization.
        // Otherwise, we just do the normalization
    }

    // Save the current position, direction and center status
    auto dir   = state.get_direction();
    auto pos   = state.template get_position<long>();
    auto cnt   = pos >= 0;
    auto steps = 0;
    if(tools::log->level() <= spdlog::level::debug)
        tools::log->debug("normalize_state: old local norm = {:.16f} | pos {} | dir {} | bond dims {}", fp(tools::finite::measure::norm(state)), pos, dir,
                          tools::finite::measure::bond_dimensions(state));

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
    while(steps++ < 2 or not state.position_is_at(pos, dir, cnt)) move_center_point_single_site(state, svd_cfg);
    state.assert_validity();
    state.clear_measurements();
    state.clear_cache();

    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    if(not state.is_normalized_on_all_sites(normTol)) {
        for(const auto &mps : state.mps_sites) {
            bool normalized_tag = state.get_normalization_tags()[mps->template get_position<size_t>()];
            tools::log->warn("{} | is_normalized {:<7} | L norm {:.16f} | norm tag {}", mps->get_tag(), mps->is_normalized(),
                             fp(tenx::VectorMap(mps->get_L()).norm()), normalized_tag);
            if(mps->isCenter()) tools::log->warn("LC({}) | norm {:.16f}", mps->get_position(), fp(tenx::VectorMap(mps->get_LC()).norm()));
        }
        auto norm_error = std::abs(tools::finite::measure::norm(state) - RealScalar<Scalar>{1});
        throw except::runtime_error("normalize_state: normalization failed. state norm error {:.3e} | max allowed norm error {:.3e} | norm tags {}",
                                    fp(norm_error), fp(normTol), state.get_normalization_tags());
    }

    if(svd_cfg and svd_cfg->rank_max and state.get_largest_bond() > svd_cfg->rank_max.value())
        throw except::logic_error("normalize_state: a bond dimension exceeds bond limit: {} > {}", tools::finite::measure::bond_dimensions(state),
                                  svd_cfg->rank_max.value());
    if(tools::log->level() <= spdlog::level::debug)
        tools::log->debug("normalize_state: new local norm = {:.16f} | pos {} | dir {} | bond dims {}", fp(tools::finite::measure::norm(state)), pos, dir,
                          tools::finite::measure::bond_dimensions(state));
    return true;
}
