#pragma once
#include "../../mps.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateInfinite.h"
#include "tools/common/split.h"

template<typename Scalar>
void tools::infinite::mps::merge_twosite_tensor(StateInfinite<Scalar> &state, const Eigen::Tensor<Scalar, 3> &twosite_tensor, MergeEvent mevent,
                                                std::optional<svd::config> svd_cfg) {
    long   dA       = state.get_spin_dimA();
    long   dB       = state.get_spin_dimB();
    size_t posA     = state.get_positionA();
    size_t posB     = state.get_positionB();
    auto   mps_list = tools::common::split::split_mps<Scalar>(twosite_tensor, {dA, dB}, {posA, posB}, safe_cast<long>(posA), svd_cfg);
    if(mps_list.size() != 2) throw except::logic_error("Got {} MPS sites from two-site tensor.", mps_list.size());

    bool keepTruncationErrors = true;
    switch(mevent) {
        case MergeEvent::MOVE:
        case MergeEvent::NORM:
        case MergeEvent::SWAP: keepTruncationErrors = false; break;
        default: break;
    }

    if(!keepTruncationErrors) {
        // The truncation errors are ignored by setting them to a negative value
        mps_list.front().unset_truncation_error();
        mps_list.front().unset_truncation_error_LC();
        mps_list.front().drop_stashed_errors();
        mps_list.back().unset_truncation_error();
        mps_list.back().unset_truncation_error_LC();
        mps_list.back().drop_stashed_errors();
    }
    state.get_mps_siteA().fuse_mps(mps_list.front());
    state.get_mps_siteB().fuse_mps(mps_list.back());
}


