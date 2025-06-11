#pragma once
#include "math/tenx.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/infinite/measure.h"
using tools::infinite::measure::RealScalar;
template<typename Scalar>
size_t tools::infinite::measure::length(const TensorsInfinite<Scalar> &tensors) {
    return tensors.edges->get_length();
}

template<typename Scalar>
size_t tools::infinite::measure::length(const EdgesInfinite<Scalar> &edges) {
    return edges.get_length();
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::norm(const StateInfinite<Scalar> &state) {
    if(state.measurements.norm) return state.measurements.norm.value();
    auto norm    = tenx::norm(state.get_2site_mps());
    auto normTol = std::numeric_limits<RealScalar<Scalar>>::epsilon() * settings::precision::max_norm_slack;
    auto normErr = std::abs(norm - RealScalar<Scalar>{1});
    if(normErr > normTol) tools::log->debug("norm: far from unity: {:.5e}", fp(normErr));
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}

template<typename Scalar>
long tools::infinite::measure::bond_dimension(const StateInfinite<Scalar> &state) {
    if(state.measurements.bond_dim) return state.measurements.bond_dim.value();
    state.measurements.bond_dim = state.chiC();
    return state.measurements.bond_dim.value();
}

template<typename Scalar>
double tools::infinite::measure::truncation_error(const StateInfinite<Scalar> &state) {
    if(state.measurements.truncation_error) return state.measurements.truncation_error.value();
    state.measurements.truncation_error = state.get_truncation_error();
    return state.measurements.truncation_error.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::entanglement_entropy(const StateInfinite<Scalar> &state) {
    if(state.measurements.entanglement_entropy) return state.measurements.entanglement_entropy.value();
    auto                     t_ent          = tid::tic_token("ent");
    const auto              &LC             = state.LC();
    Eigen::Tensor<Scalar, 0> SA             = -LC.square().contract(LC.square().log().eval(), tenx::idx({0}, {0}));
    state.measurements.entanglement_entropy = std::real(SA(0));
    return state.measurements.entanglement_entropy.value();
}
