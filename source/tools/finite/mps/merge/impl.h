#pragma once

#include "math/tenx.h"
// -- (textra first)
#include "../../mps.h"
#include "config/enums.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/linalg/tensor/to_string.h"
#include "math/num.h"
#include "math/svd.h"
#include "qm/mpo.h"
#include "qm/spin.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/common/split.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/ops.h"
#include <fmt/ranges.h>
namespace settings {
    inline constexpr bool debug_merge   = false;
    inline constexpr bool verbose_merge = false;
}

using tools::finite::mps::RealScalar;

template<typename Scalar>
size_t tools::finite::mps::merge_multisite_mps(StateFinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &multisite_mps, const std::vector<size_t> &sites,
                                               long center_position, MergeEvent mevent, std::optional<svd::config> svd_cfg,
                                               std::optional<LogPolicy> logPolicy) {
    auto t_merge          = tid::tic_scope("merge", tid::level::higher);
    auto current_position = state.template get_position<long>();
    auto moves            = static_cast<size_t>(std::abs(center_position - current_position));
    if constexpr(settings::debug)
        if(logPolicy == LogPolicy::VERBOSE)
            tools::log->trace("merge_multisite_mps: sites {} | dimensions {} | center {} -> {} | {}", sites, multisite_mps.dimensions(), current_position,
                              center_position, state.get_labels());
    if constexpr(settings::verbose_merge)
        tools::log->trace("merge_multisite_mps: sites {} | dimensions {} | center {} -> {} | {}", sites, multisite_mps.dimensions(), current_position,
                          center_position, state.get_labels());

    // Some sanity checks
    if(multisite_mps.dimension(1) != state.get_mps_site(sites.front()).get_chiL())
        throw except::logic_error("merge_multisite_mps: mps dim1 {} != chiL {} on left-most site", multisite_mps.dimension(1),
                                  state.get_mps_site(sites.front()).get_chiL(), sites.front());

    if(multisite_mps.dimension(2) != state.get_mps_site(sites.back()).get_chiR())
        throw except::logic_error("merge_multisite_mps: mps dim2 {} != chiR {} on right-most site", multisite_mps.dimension(2),
                                  state.get_mps_site(sites.back()).get_chiR(), sites.back());
    if constexpr(settings::debug_merge or settings::debug) {
        auto t_dbg = tid::tic_scope("debug");
        // Never allow nan's in the multisite_mps
        if(tenx::hasNaN(multisite_mps))
            throw except::runtime_error("merge_multisite_mps: multisite_mps has nan's:\n"
                                        "sites            :{}\n"
                                        "center_position  :{}\n"
                                        "current_position :{}\n"
                                        "multisite_mps    :\n{}",
                                        sites, center_position, current_position, linalg::tensor::to_string(multisite_mps, 3, 6));

        if(state.has_nan())
            throw except::runtime_error("merge_multisite_mps: state has nan's:\n"
                                        "sites            :{}\n"
                                        "center_position  :{}\n"
                                        "current_position :{}\n"
                                        "multisite_mps    :\n{}",
                                        sites, center_position, current_position, linalg::tensor::to_string(multisite_mps, 3, 6));

        // We have to allow non-normalized multisite mps! Otherwise, we won't be able to make them normalized
        auto norm      = tenx::norm(multisite_mps);
        auto normError = std::abs(norm - Scalar{1});
        auto normTol   = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
        if(normError > normTol)
            tools::log->debug("merge_multisite_mps: Multisite mps for positions {} has norm far from unity. Norm error: {:.5e}", sites, fp(normError));
    }

    // Can't set center on one of sites if the current center is too far away: we would end up with interleaved A's and B sites
    bool center_in_sites = center_position == std::clamp<long>(center_position, safe_cast<long>(sites.front()), safe_cast<long>(sites.back()));
    bool center_in_range = current_position == std::clamp<long>(current_position, safe_cast<long>(sites.front()) - 1, safe_cast<long>(sites.back()));
    if(center_in_sites and not center_in_range)
        throw except::runtime_error("merge_multisite_mps: cannot merge multisite_mps {} with new center at {}: current center {} is too far", sites,
                                    center_position, current_position);

    long              spin_prod = 1;
    std::vector<long> spin_dims;
    spin_dims.reserve(sites.size());
    for(const auto &pos : sites) {
        spin_dims.emplace_back(state.get_mps_site(pos).spin_dim());
        spin_prod *= spin_dims.back();
    }
    if(spin_prod != multisite_mps.dimension(0))
        throw except::runtime_error("merge_multisite_mps: multisite mps dim0 {} != spin_prod {}", multisite_mps.dimension(0), spin_prod);

    // Hold LC if moving. This should be placed in an L-slot later
    std::optional<stash<Eigen::Tensor<Scalar, 1>>> lc_move = std::nullopt;
    if(center_position != current_position and current_position >= 0) {
        auto &mps      = state.get_mps_site(current_position); // Guaranteed to have LC since that is the definition of current_position
        auto  pos_back = safe_cast<long>(sites.back());
        auto  pos_frnt = safe_cast<long>(sites.front());
        auto  pos_curr = safe_cast<size_t>(current_position);

        // Detect right-move
        if(center_position > current_position) { // This AC will become an A (AC moves to the right)
            if(center_position != std::clamp(center_position, pos_frnt, pos_back))
                throw except::logic_error("merge_multisite_mps: right-moving new center position {} must be in sites {}", center_position, sites);

            // Case 1, right-move: LC[3]B[4] -> L[4]A[4]LC[4]V[5], current_position == 3, center_position == 4. Then LC[3] becomes L[4] on A[4]
            // Case 2, swap-move: A[3]LC[3]B[4] -> A[3]A[4]LC[4]V, current_position == 3, center_position == 4. Then LC[3] is thrown away
            // Case 4, deep-move: A[3]A[4]LC[4]B[5]B[6]B[7] -> A[3]A[4]A[5]A[6]LC[6]B[7], current_position == 5, center_position == 6. Then LC[4] is thrown
            // Takeaway: LC is only held when LC is on the left edge, turning a B into an A which needs an L
            // It's important that the last V is a diagonal matrix, otherwise it would truncate the site to the right.
            if(current_position + 1 == pos_frnt) lc_move = stash<Eigen::Tensor<Scalar, 1>>{mps.get_LC(), mps.get_truncation_error_LC(), sites.front()};
        }
        // Detect left-move
        if(center_position < current_position) { // This AC position will become a B (AC moves to the left)
            if(center_position < pos_frnt - 1)
                throw except::logic_error("merge_multisite_mps: left-moving new center position {} is out of range [{}]+{}", center_position, pos_frnt - 1,
                                          sites);
            if(current_position > pos_back + 1)
                throw except::logic_error("merge_multisite_mps: left-moving current position {} is out of range {}+[{}]", current_position, sites,
                                          pos_back + 1);

            // Case 1, left-move: A[3]LC[3]     -> U[2]LC[2]B[3]    , current_position == 3, center_position == 2. Then LC[3] becomes L[3] on B[3]
            // Case 2, swap-move: A[3]A[4]LC[4] -> A[3]LC[3]B[4]    , current_position == 3, center_position == 4. Then LC[4] becomes L[4] on B[4]
            // Case 3, full-move: A[3]A[4]LC[4] -> U[2]LC[2]B[3]B[4], current_position == 3, center_position == 4. Then LC[4] becomes L[4] on B[4]
            // Case 4, deep-move: A[3]A[4]LC[4]B[5]B[6]B[7] -> A[3]LC[4]B[4]B[5]B[6]B[7], current_position == 4 center_position == 3. Then LC[4] is thrown
            // Takeaway: LC is only held when LC is on the right edge, turning an AC into a B which needs an L.
            // It's important that the front U is a diagonal matrix, otherwise it would truncate the site to the left.

            if(current_position == pos_back) lc_move = stash<Eigen::Tensor<Scalar, 1>>{mps.get_LC(), mps.get_truncation_error_LC(), pos_curr};
        }
        // Note that one of the positions on the split may contain a new center, so we need to unset
        // the center in our current state so we don't get duplicate centers
        mps.unset_LC();
    }

    if constexpr(settings::verbose_merge)
        if(svd_cfg) tools::log->trace("merge_multisite_mps: splitting sites {} | {}", sites, svd_cfg->to_string());

    // Split the multisite mps into single-site mps objects
    auto mps_list = tools::common::split::split_mps<Scalar>(multisite_mps, spin_dims, sites, center_position, svd_cfg);
    // Sanity checks
    if(sites.size() != mps_list.size())
        throw std::runtime_error(
            fmt::format("merge_multisite_mps: number of sites mismatch: sites.size() {} != mps_list.size() {}", sites.size(), mps_list.size()));

    bool keepTruncationErrors = true;
    switch(mevent) {
        case MergeEvent::MOVE: [[fallthrough]];
        case MergeEvent::NORM: [[fallthrough]];
        case MergeEvent::EXP: [[fallthrough]];
        case MergeEvent::SWAP: keepTruncationErrors = false; break;
        case MergeEvent::OPT: keepTruncationErrors = true; break;
        default: break;
    }

    // Fuse the split-up mps components into the current state
    for(auto &mps_src : mps_list) {
        auto  pos     = mps_src.get_position();
        auto &mps_tgt = state.get_mps_site(pos);
        if(!keepTruncationErrors) {
            // The truncation errors are ignored
            mps_src.unset_truncation_error();
            mps_src.unset_truncation_error_LC();
            mps_src.drop_stashed_errors();
        }

        // inject lc_move if there is any waiting
        if(lc_move and pos == lc_move->pos_dst) { mps_src.set_L(lc_move->data, lc_move->error); }
        mps_tgt.fuse_mps(mps_src);
        state.tag_site_normalized(pos, true); // Fused site is normalized

        // Now take stashes for neighboring sites
        // Note that if U or V are rectangular and pushed onto the next site, that next site loses its normalization, unless
        // we are pushing across the center matrix.
        // Tagging it as non-normalized lets us determine whether a full normalization is required later.
        if(pos < state.get_length() - 1) {
            auto &mps_nbr  = state.get_mps_site(pos + 1);
            auto  old_chiL = mps_nbr.get_chiL();
            mps_nbr.take_stash(mps_src);                                                                                 // Take stashed S,V (and possibly LC)
            if(mps_nbr.get_label() != "B" and mps_nbr.get_chiL() != old_chiL) state.tag_site_normalized(pos + 1, false); // Normalization may have been ruined
        }
        if(pos > 0) {
            auto &mps_nbr  = state.get_mps_site(pos - 1);
            auto  old_chiR = mps_nbr.get_chiR();
            mps_nbr.take_stash(mps_src);                                                                                 // Take stashed U,S (and possibly LC)
            if(mps_nbr.get_label() == "B" and mps_nbr.get_chiR() != old_chiR) state.tag_site_normalized(pos - 1, false); // Normalization may have been ruined
        }
        mps_src.drop_stash(); // Discard whatever is left stashed at the edge (this normalizes the state)
    }

    current_position = state.template get_position<long>();
    if(current_position != center_position)
        throw except::logic_error("Center position mismatch {} ! {}\nLabels: {}", current_position, center_position, state.get_labels());
    state.clear_cache();
    state.clear_measurements();
    if constexpr(settings::debug or settings::debug_merge) {
        auto t_dbg = tid::tic_scope("debug");
        for(const auto &pos : sites) state.get_mps_site(pos).assert_validity();
        for(const auto &pos : sites) state.get_mps_site(pos).assert_normalized();
    }
    return moves;
}

template<typename Scalar>
bool tools::finite::mps::normalize_state(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy) {
    // When a state needs to be normalized it's enough to "move" the center position around the whole chain.
    // Each move performs an SVD decomposition which leaves unitaries behind, effectively normalizing the state.
    // NOTE! It IS important to start with the current position.
    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    if(norm_policy == NormPolicy::IFNEEDED) {
        // We may only go ahead with a normalization if it's really needed.
        tools::log->trace("normalize_state: checking if needed");
        if(state.is_normalized_on_all_sites(normTol)) return false; // Return false, i.e. did "not" perform a normalization.
        // Otherwise, we just do the normalization
    }

    // Save the current position, direction and center status
    auto dir   = state.get_direction();
    auto pos   = state.template get_position<long>();
    auto cnt   = pos >= 0;
    auto steps = 0;
    if(tools::log->level() <= spdlog::level::debug)
        tools::log->debug("normalize_state: {} old local norm = {:.16f} | pos {} | dir {} | bond dims {}", enum2sv(norm_policy),
                          fp(tools::finite::measure::norm(state)), pos, dir, tools::finite::measure::bond_dimensions(state));

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
template bool tools::finite::mps::normalize_state(StateFinite<fp32> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
template bool tools::finite::mps::normalize_state(StateFinite<fp64> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
template bool tools::finite::mps::normalize_state(StateFinite<fp128> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
template bool tools::finite::mps::normalize_state(StateFinite<cx32> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
template bool tools::finite::mps::normalize_state(StateFinite<cx64> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);
template bool tools::finite::mps::normalize_state(StateFinite<cx128> &state, std::optional<svd::config> svd_cfg, NormPolicy norm_policy);

template<typename Scalar>
void tools::finite::mps::apply_random_paulis(StateFinite<Scalar> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices) {
    auto [mpos, L, R] = qm::mpo::sum_of_pauli_mpo<Scalar>(paulimatrices, state.get_length(), RandomizerMode::SELECT1);
    tools::finite::ops::apply_mpos(state, mpos, L, R);
}
template void tools::finite::mps::apply_random_paulis(StateFinite<fp32> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);
template void tools::finite::mps::apply_random_paulis(StateFinite<fp64> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);
template void tools::finite::mps::apply_random_paulis(StateFinite<fp128> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx32> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx64> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx128> &state, const std::vector<Eigen::MatrixXcd> &paulimatrices);

template<typename Scalar>
void tools::finite::mps::apply_random_paulis(StateFinite<Scalar> &state, const std::vector<std::string> &paulistrings) {
    std::vector<Eigen::MatrixXcd> paulimatrices;
    for(const auto &str : paulistrings) paulimatrices.emplace_back(qm::spin::half::get_pauli(str));
    apply_random_paulis(state, paulimatrices);
}
template void tools::finite::mps::apply_random_paulis(StateFinite<fp32> &state, const std::vector<std::string> &paulistrings);
template void tools::finite::mps::apply_random_paulis(StateFinite<fp64> &state, const std::vector<std::string> &paulistrings);
template void tools::finite::mps::apply_random_paulis(StateFinite<fp128> &state, const std::vector<std::string> &paulistrings);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx32> &state, const std::vector<std::string> &paulistrings);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx64> &state, const std::vector<std::string> &paulistrings);
template void tools::finite::mps::apply_random_paulis(StateFinite<cx128> &state, const std::vector<std::string> &paulistrings);