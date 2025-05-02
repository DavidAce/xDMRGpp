#include "math/tenx.h"
// -- (textra first)
#include "../mps.h"
#include "math/svd.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
namespace settings {
    inline constexpr bool debug_moves   = true;
    inline constexpr bool verbose_moves = true;
}

template<typename Scalar>
size_t tools::finite::mps::move_center_point_single_site(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg) {
    auto t_move = tid::tic_scope("move", tid::level::higher);
    if(state.position_is_outward_edge()) {
        if(state.get_direction() == -1 and state.get_mps_site(0l).get_chiL() != 1)
            throw except::logic_error("chiL at position 0 must have dimension 1, but it has dimension {}. Mps dims {}", state.get_mps_site(0l).get_chiL(),
                                      state.get_mps_site(0l).dimensions());
        if(state.get_direction() == 1 and state.get_mps_site().get_chiR() != 1)
            throw except::logic_error("chiR at position {} must have dimension 1, but it has dimension {}. Mps dims {}", state.get_position(),
                                      state.get_mps_site().get_chiR(), state.get_mps_site().dimensions());
        state.flip_direction(); // Instead of moving out of the chain, just flip the direction and return
        return 0;               // No moves this time, return 0
    } else {
        long pos  = state.template get_position<long>(); // If all sites are B's, then this is -1. Otherwise, this is the current "A*LC" site
        long posC = pos + state.get_direction();         // This is the site which becomes the new center position
        if(pos < -1 or pos >= state.template get_length<long>()) throw except::range_error("pos out of bounds: {}", pos);
        if(posC < -1 or posC >= state.template get_length<long>()) throw except::range_error("posC out of bounds: {}", posC);
        if(state.get_direction() != posC - pos) throw except::logic_error("Expected posC - pos == {}. Got {}", state.get_direction(), posC - pos);

        if constexpr(settings::verbose_moves) {
            if(posC > pos) tools::log->trace("Moving {} -> {}", pos, posC);
            if(posC < pos) tools::log->trace("Moving {} <- {}", posC, pos);
        }

        Eigen::Tensor<Scalar, 1> LC(1);
        LC.setConstant(Scalar{1}); // Store the LC bond in a temporary. It will become a regular "L" bond later
        if(pos >= 0) LC = state.get_mps_site(pos).get_LC();

        if(state.get_direction() == 1) {
            auto  posC_ul = safe_cast<size_t>(posC);     // Cast to unsigned
            auto &mpsC    = state.get_mps_site(posC);    // This becomes the new AC (currently B)
            auto  trnc    = mpsC.get_truncation_error(); // Truncation error of the old B/new AC, i.e. bond to the right of posC,
            // Construct a single-site tensor. This is equivalent to state.get_multisite_mps(...) but avoid normalization checks.
            auto onesite_tensor = tools::common::contraction::contract_bnd_mps(LC, mpsC.get_M());
            tools::finite::mps::merge_multisite_mps(state, onesite_tensor, {posC_ul}, posC, MergeEvent::MOVE, svd_cfg, LogPolicy::VERBOSE);
            mpsC.set_truncation_error_LC(std::max(trnc, mpsC.get_truncation_error_LC()));
        } else if(state.get_direction() == -1) {
            auto  pos_ul = safe_cast<size_t>(pos);     // Cast to unsigned
            auto &mps    = state.get_mps_site(pos);    // This AC becomes the new B
            auto  trnc   = mps.get_truncation_error(); // Truncation error of old AC/new B, i.e. bond to the left of pos,
            // No need to contract anything this time. Note that we must take a copy! Not a reference (since M, LC are unset in merge)
            const auto onesite_tensor = mps.get_M();
            tools::finite::mps::merge_multisite_mps(state, onesite_tensor, {pos_ul}, posC, MergeEvent::MOVE, svd_cfg, LogPolicy::VERBOSE);
            if(posC >= 0) {
                auto &mpsC = state.get_mps_site(posC); // This old A is now an AC
                mpsC.set_truncation_error_LC(std::max(trnc, mpsC.get_truncation_error_LC()));
            }
        }
        state.clear_cache();
        state.clear_measurements();
        return 1; // Moved once, so return 1
    }
}
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<fp32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<fp64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<fp128> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<cx32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<cx64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_single_site(StateFinite<cx128> &state, std::optional<svd::config> svd_cfg);

template<typename Scalar>
size_t tools::finite::mps::move_center_point(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg) {
    auto t_move = tid::tic_scope("move");
    if(state.position_is_outward_edge(2)) {
        state.flip_direction(); // Instead of moving out of the chain, just flip the direction and return
        return 0;               // No moves this time, return 0
    } else {
        long pos = state.template get_position<long>();
        if(pos < -1 or pos >= state.template get_length<long>()) throw except::range_error("pos out of bounds: {}", pos);

        long  posL    = state.get_direction() == 1 ? pos + 1 : pos - 1;
        long  posR    = state.get_direction() == 1 ? pos + 2 : pos;
        auto  posL_ul = safe_cast<size_t>(posL);
        auto  posR_ul = safe_cast<size_t>(posR);
        auto &mps     = state.get_mps_site();
        auto &mpsL    = state.get_mps_site(posL); // Becomes the new center position
        auto &mpsR    = state.get_mps_site(posR); // The site to the right of the new center position.

        // Store the special LC bond in a temporary. It needs to be put back afterward
        // Do the same with its truncation error
        Eigen::Tensor<Scalar, 1> LC                  = mps.template get_LC_as<Scalar>();
        double                   truncation_error_LC = mps.get_truncation_error_LC();
        auto                     twosite_tensor      = state.template get_multisite_mps<Scalar>({posL_ul, posR_ul});
        tools::finite::mps::merge_multisite_mps(state, twosite_tensor, {static_cast<size_t>(posL), static_cast<size_t>(posR)}, safe_cast<long>(posL),
                                                MergeEvent::MOVE, svd_cfg, LogPolicy::SILENT);
        state.clear_cache();
        state.clear_measurements();

        // Put LC where it belongs.
        // Recall that mpsL, mpsR are on the new position, not the old one!
        if(state.get_direction() == 1)
            mpsL.set_L(LC, truncation_error_LC);
        else
            mpsR.set_L(LC, truncation_error_LC);
        return 1; // Moved once, so return 1
    }
}
template size_t tools::finite::mps::move_center_point(StateFinite<fp32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point(StateFinite<fp64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point(StateFinite<fp128> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point(StateFinite<cx32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point(StateFinite<cx64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point(StateFinite<cx128> &state, std::optional<svd::config> svd_cfg);

template<typename Scalar>
size_t tools::finite::mps::move_center_point_to_pos(StateFinite<Scalar> &state, long pos, std::optional<svd::config> svd_cfg) {
    auto position = state.template get_position<long>();
    if(pos != std::clamp<long>(pos, -1l, position - 1))
        throw except::logic_error("move_center_point_to_pos: Given pos [{}]. Expected range [-1,{}]", pos, position - 1);
    if((state.get_direction() < 0 and pos > position) or //
       (state.get_direction() > 0 and pos < position))   //
        state.flip_direction();                          // Turn direction towards new position

    size_t moves = 0;
    while(not state.position_is_at(pos)) moves += move_center_point_single_site(state, svd_cfg);
    return moves;
}
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<fp32> &state, long pos, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<fp64> &state, long pos, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<fp128> &state, long pos, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<cx32> &state, long pos, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<cx64> &state, long pos, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos(StateFinite<cx128> &state, long pos, std::optional<svd::config> svd_cfg);

template<typename Scalar>
size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<Scalar> &state, long pos, int dir, std::optional<svd::config> svd_cfg) {
    if(pos != std::clamp<long>(pos, -1l, state.template get_length<long>() - 1))
        throw except::logic_error("move_center_point_to_pos_dir: Given pos [{}]. Expected range [-1,{}]", pos, state.template get_length<long>() - 1);
    if(std::abs(dir) != 1) throw except::logic_error("move_center_point_to_pos_dir: dir must be 1 or -1");
    auto position = state.template get_position<long>();
    if((state.get_direction() < 0 and pos > position) or //
       (state.get_direction() > 0 and pos < position))   //
        state.flip_direction();                          // Turn direction towards new position
    size_t moves = 0;
    while(not state.position_is_at(pos)) moves += move_center_point_single_site(state, svd_cfg);
    if(dir != state.get_direction()) state.flip_direction();
    return moves;
}
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<fp32> &state, long pos, int dir, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<fp64> &state, long pos, int dir, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<fp128> &state, long pos, int dir, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<cx32> &state, long pos, int dir, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<cx64> &state, long pos, int dir, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_pos_dir(StateFinite<cx128> &state, long pos, int dir, std::optional<svd::config> svd_cfg);

template<typename Scalar>
size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg) {
    // Flip if past middle and going in the other direction
    auto position = state.template get_position<long>();
    if(state.get_direction() < 0 and position > state.template get_length<long>() / 2) state.flip_direction();
    if(state.get_direction() > 0 and position < state.template get_length<long>() / 2) state.flip_direction();
    size_t moves = 0;
    while(not state.position_is_inward_edge()) moves += move_center_point_single_site(state, svd_cfg);
    return moves;
}
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<fp32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<fp64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<fp128> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<cx32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<cx64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_inward_edge(StateFinite<cx128> &state, std::optional<svd::config> svd_cfg);

template<typename Scalar>
size_t tools::finite::mps::move_center_point_to_middle(StateFinite<Scalar> &state, std::optional<svd::config> svd_cfg) {
    size_t moves = 0;
    while(not state.position_is_the_middle_any_direction()) moves += move_center_point_single_site(state, svd_cfg);
    return moves;
}
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<fp32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<fp64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<fp128> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<cx32> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<cx64> &state, std::optional<svd::config> svd_cfg);
template size_t tools::finite::mps::move_center_point_to_middle(StateFinite<cx128> &state, std::optional<svd::config> svd_cfg);