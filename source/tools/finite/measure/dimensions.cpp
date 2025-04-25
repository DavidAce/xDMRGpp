
#include "dimensions.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"

template<typename Scalar>
long tools::finite::measure::bond_dimension_current(const StateFinite<Scalar> &state) {
    if(state.measurements.bond_dim) return state.measurements.bond_dim.value();
    if(state.has_center_point())
        state.measurements.bond_dim = state.current_bond().dimension(0);
    else
        state.measurements.bond_dim = 1;
    return state.measurements.bond_dim.value();
}
template long tools::finite::measure::bond_dimension_current(const StateFinite<fp32> &state);
template long tools::finite::measure::bond_dimension_current(const StateFinite<fp64> &state);
template long tools::finite::measure::bond_dimension_current(const StateFinite<fp128> &state);
template long tools::finite::measure::bond_dimension_current(const StateFinite<cx32> &state);
template long tools::finite::measure::bond_dimension_current(const StateFinite<cx64> &state);
template long tools::finite::measure::bond_dimension_current(const StateFinite<cx128> &state);

template<typename Scalar>
long tools::finite::measure::bond_dimension_midchain(const StateFinite<Scalar> &state) {
    if(state.measurements.bond_mid) return state.measurements.bond_mid.value();
    state.measurements.bond_mid = state.get_midchain_bond().dimension(0);
    return state.measurements.bond_mid.value();
}
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<fp32> &state);
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<fp64> &state);
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<fp128> &state);
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<cx32> &state);
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<cx64> &state);
template long tools::finite::measure::bond_dimension_midchain(const StateFinite<cx128> &state);

template<typename Scalar>
std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<Scalar> &state, size_t pos) {
    auto t_bond = tid::tic_scope("bond_dimensions", tid::level::highest);
    assert(pos < state.template get_length<size_t>());
    return {state.mps_sites[pos]->get_chiL(), state.mps_sites[pos]->get_chiR()};
}
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<fp32> &state, size_t pos);
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<fp64> &state, size_t pos);
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<fp128> &state, size_t pos);
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<cx32> &state, size_t pos);
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<cx64> &state, size_t pos);
template std::pair<long, long> tools::finite::measure::bond_dimensions(const StateFinite<cx128> &state, size_t pos);

template<typename Scalar>
std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<Scalar> &state) {
    if(state.measurements.bond_dimensions) return state.measurements.bond_dimensions.value();
    auto              t_bond = tid::tic_scope("bond_dimensions", tid::level::highest);
    std::vector<long> bond_dimensions;
    bond_dimensions.reserve(state.get_length() + 1);
    if(not state.has_center_point()) bond_dimensions.emplace_back(state.mps_sites.front()->get_chiL());
    for(const auto &mps : state.mps_sites) {
        bond_dimensions.emplace_back(mps->get_L().dimension(0));
        if(mps->isCenter()) { bond_dimensions.emplace_back(mps->get_LC().dimension(0)); }
    }
    if(bond_dimensions.size() != state.get_length() + 1) throw except::logic_error("bond_dimensions.size() should be length+1");
    state.measurements.bond_dimensions = bond_dimensions;
    return state.measurements.bond_dimensions.value();
}

template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<fp32> &state);
template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<fp64> &state);
template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<fp128> &state);
template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<cx32> &state);
template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<cx64> &state);
template std::vector<long> tools::finite::measure::bond_dimensions(const StateFinite<cx128> &state);

template<typename Scalar>
std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<Scalar> &state) {
    // Here we get the bond dimensions of the bonds that were merged into the full state in the last step
    // For instance, if the active sites are {2,3,4,5,6} this returns the 4 bonds connecting {2,3}, {3,4}, {4,5} and {5,6}
    // If active sites is just {4}, it returns the bond between {4,5} when going left or right, or between {3,4} wen going right to left
    auto t_chi = tid::tic_scope("bond_merged", tid::level::highest);
    if(state.active_sites.empty()) return {};
    if(state.active_sites.size() == 1) {
        // In single-site DMRG the active site is a center "AC" site:
        //  * Going left-to-right, the forward (right) bond is expanded, and this same bond is truncated when merging
        //  * Going right-to-left, the forward (left) bond is expanded (L), but LC is still the one truncated when merging.
        if(state.get_direction() == +1) return {state.get_mps_site(state.active_sites[0]).get_chiR()};
        if(state.get_direction() == -1) return {state.get_mps_site(state.active_sites[0]).get_chiL()};
    }
    if(state.active_sites.size() == 2) return {state.get_mps_site(state.active_sites[0]).get_chiR()};
    std::vector<long> bond_dimensions;
    for(const auto &pos : state.active_sites) {
        if(&pos == &state.active_sites.front()) continue;
        const auto &mps = state.get_mps_site(pos);
        bond_dimensions.push_back(mps.get_chiL());
    }
    return bond_dimensions;
}
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<fp32> &state);
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<fp64> &state);
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<fp128> &state);
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<cx32> &state);
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<cx64> &state);
template std::vector<long> tools::finite::measure::bond_dimensions_active(const StateFinite<cx128> &state);

template<typename Scalar>
std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<Scalar> &state) {
    std::vector<long> spin_dimensions;
    spin_dimensions.reserve(state.get_length());
    for(const auto &mps : state.mps_sites) { spin_dimensions.emplace_back(mps->spin_dim()); }
    return spin_dimensions;
}

template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<fp32> &state);
template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<fp64> &state);
template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<fp128> &state);
template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<cx32> &state);
template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<cx64> &state);
template std::vector<long> tools::finite::measure::spin_dimensions(const StateFinite<cx128> &state);