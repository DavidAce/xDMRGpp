#pragma once
#include "truncation.h"
#include "debug/exceptions.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"

template<typename Scalar>
std::vector<fp64> tools::finite::measure::truncation_errors(const StateFinite<Scalar> &state) {
    if(state.measurements.truncation_errors) return state.measurements.truncation_errors.value();
    auto              t_chi = tid::tic_scope("trunc", tid::level::highest);
    std::vector<fp64> truncation_errors;
    if(not state.has_center_point()) truncation_errors.emplace_back(0);
    for(const auto &mps : state.mps_sites) {
        truncation_errors.emplace_back(mps->get_truncation_error());
        if(mps->isCenter()) truncation_errors.emplace_back(mps->get_truncation_error_LC());
    }
    if(truncation_errors.size() != state.get_length() + 1) throw except::logic_error("truncation_errors.size() should be length+1");
    state.measurements.truncation_errors = truncation_errors;
    return state.measurements.truncation_errors.value();
}

template<typename Scalar>
std::vector<fp64> tools::finite::measure::truncation_errors_active(const StateFinite<Scalar> &state) {
    // Here we get the truncation erros of the bonds that were merged into the full state in the last step
    // For instance, if the active sites are {2, 3, 4, 5, 6}, this returns the 4 bonds connecting {2,3}, {3,4}, {4,5} and {5,6}
    // If active_sites is just {4}, it returns the bond between {4,5} when going right, and {3,4} when going left.
    if(state.active_sites.empty()) throw except::logic_error("truncation_errors_active(): active_sites is empty");
    if(state.active_sites.size() == 1) {
        // In single-site DMRG the active site is a center "AC" site:
        // If we do forward expansion:
        //  * Going left-to-right, the right bond (LC) is truncated after optimization
        //  * Going right-to-left, the left bond (L) is truncated after optimization.
        // if(state.get_direction() == +1) return {state.get_mps_site(state.active_sites[0]).get_truncation_error_LC()};
        // if(state.get_direction() == -1) return {state.get_mps_site(state.active_sites[0]).get_truncation_error()};
        return {state.get_mps_site(state.active_sites[0]).get_truncation_error_last()};
    }
    if(state.active_sites.size() == 2) return {state.get_mps_site(state.active_sites[0]).get_truncation_error_LC()};
    std::vector<fp64> truncation_errors;
    for(const auto &pos : state.active_sites) {
        if(&pos == &state.active_sites.front()) continue;
        const auto &mps = state.get_mps_site(pos);
        truncation_errors.push_back(mps.get_truncation_error_last());
    }
    return truncation_errors;
}
