#pragma once
#include "spin.h"
#include "config/debug.h"
#include "correlation.h"
#include "debug/exceptions.h"
#include "expectation_value.h"
#include "math/num.h"
#include "math/tenx.h"
#include "qm/mpo.h"
#include "qm/spin.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

using tools::finite::measure::RealScalar;

template<typename Scalar>
std::array<RealScalar<Scalar>, 3> tools::finite::measure::spin_components(const StateFinite<Scalar> &state) {
    if(state.measurements.spin_components) return state.measurements.spin_components.value();
    RealScalar<Scalar> spin_x          = measure::spin_component(state, qm::spin::half::sx);
    RealScalar<Scalar> spin_y          = measure::spin_component(state, qm::spin::half::sy);
    RealScalar<Scalar> spin_z          = measure::spin_component(state, qm::spin::half::sz);
    state.measurements.spin_components = {spin_x, spin_y, spin_z};
    return state.measurements.spin_components.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::spin_component(const StateFinite<Scalar> &state, const Eigen::Matrix2cd &paulimatrix) {
    auto t_spn       = tid::tic_scope("spin", tid::level::highest);
    auto [mpo, L, R] = qm::mpo::pauli_mpo<Scalar>(paulimatrix);
    Eigen::Tensor<Scalar, 3> temp;
    for(const auto &mps : state.mps_sites) {
        tools::common::contraction::contract_env_mps_mpo(temp, L, mps->get_M(), mpo);
        L = temp;
    }

    if(L.dimensions() != R.dimensions()) throw except::runtime_error("spin_component(): L and R dimension mismatch");
    Eigen::Tensor<Scalar, 0> spin_tmp = L.contract(R, tenx::idx({0, 1, 2}, {0, 1, 2}));
    RealScalar<Scalar>       spin     = std::real(spin_tmp(0));
    return spin;
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::spin_component(const StateFinite<Scalar> &state, std::string_view axis) {
    if(axis.find('x') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sx);
    if(axis.find('y') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sy);
    if(axis.find('z') != std::string_view::npos) return measure::spin_component(state, qm::spin::half::sz);
    throw except::logic_error("unexpected axis [{}]", axis);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::spin_alignment(const StateFinite<Scalar> &state, std::string_view axis) {
    if(not qm::spin::half::is_valid_axis(axis)) throw except::logic_error("unexpected axis [{}]", axis);
    auto spin_component_along_axis = tools::finite::measure::spin_component(state, qm::spin::half::get_pauli(axis));
    auto sign                      = qm::spin::half::get_sign(axis);
    return sign * spin_component_along_axis;
}

template<typename Scalar>
int tools::finite::measure::spin_sign(const StateFinite<Scalar> &state, std::string_view axis) {
    // The sign on the axis string is ignored here
    auto spin_component_along_axis = tools::finite::measure::spin_component(state, qm::spin::half::get_pauli(axis));
    return num::sign(spin_component_along_axis);
}

template<typename Scalar>
std::array<Eigen::Tensor<RealScalar<Scalar>, 1>, 3> tools::finite::measure::spin_expectation_values_xyz(const StateFinite<Scalar> &state) {
    if(not state.measurements.expectation_values_sx)
        state.measurements.expectation_values_sx = measure::expectation_values<Scalar>(state, qm::spin::half::sx).real();
    if(not state.measurements.expectation_values_sy)
        state.measurements.expectation_values_sy = measure::expectation_values<Scalar>(state, qm::spin::half::sy).real();
    if(not state.measurements.expectation_values_sz)
        state.measurements.expectation_values_sz = measure::expectation_values<Scalar>(state, qm::spin::half::sz).real();
    return {state.measurements.expectation_values_sx.value(), state.measurements.expectation_values_sy.value(),
            state.measurements.expectation_values_sz.value()};
}

template<typename Scalar>
std::array<RealScalar<Scalar>, 3> tools::finite::measure::spin_expectation_value_xyz(const StateFinite<Scalar> &state) {
    if constexpr(settings::debug) tools::log->trace("Measuring spin expectation_value_xyz");
    auto sx = qm::spin::half::tensor::sx;
    auto sy = qm::spin::half::tensor::sy;
    auto sz = qm::spin::half::tensor::sz;

    auto pos = (state.template get_length<long>() - 1) / 2;
    return {std::real(measure::expectation_value<Scalar>(state, std::vector{LocalObservableOp{sx, pos}})),
            std::real(measure::expectation_value<Scalar>(state, std::vector{LocalObservableOp{sy, pos}})),
            std::real(measure::expectation_value<Scalar>(state, std::vector{LocalObservableOp{sz, pos}}))};
}

template<typename Scalar>
std::array<Eigen::Tensor<RealScalar<Scalar>, 2>, 3> tools::finite::measure::spin_correlation_matrix_xyz(const StateFinite<Scalar> &state) {
    if constexpr(settings::debug) tools::log->trace("Measuring spin correlation_matrix_xyz");
    auto sx = qm::spin::half::tensor::sx;
    auto sy = qm::spin::half::tensor::sy;
    auto sz = qm::spin::half::tensor::sz;
    if(not state.measurements.correlation_matrix_sx) state.measurements.correlation_matrix_sx = measure::correlation_matrix<Scalar>(state, sx, sx).real();
    if(not state.measurements.correlation_matrix_sy) state.measurements.correlation_matrix_sy = measure::correlation_matrix<Scalar>(state, sy, sy).real();
    if(not state.measurements.correlation_matrix_sz) state.measurements.correlation_matrix_sz = measure::correlation_matrix<Scalar>(state, sz, sz).real();
    return {state.measurements.correlation_matrix_sx.value(), state.measurements.correlation_matrix_sy.value(),
            state.measurements.correlation_matrix_sz.value()};
}

template<typename Scalar>
std::array<RealScalar<Scalar>, 3> tools::finite::measure::spin_structure_factor_xyz(const StateFinite<Scalar> &state) {
    auto sx = qm::spin::half::tensor::sx;
    auto sy = qm::spin::half::tensor::sy;
    auto sz = qm::spin::half::tensor::sz;

    if(not state.measurements.structure_factor_x) {
        auto correlation_matrix_sx = measure::correlation_matrix<Scalar>(state, sx, sx);
        if(not state.measurements.correlation_matrix_sx) state.measurements.correlation_matrix_sx = correlation_matrix_sx.real();
        state.measurements.structure_factor_x = measure::structure_factor(state, correlation_matrix_sx);
    }
    if(not state.measurements.structure_factor_y) {
        auto correlation_matrix_sy = measure::correlation_matrix<Scalar>(state, sy, sy);
        if(not state.measurements.correlation_matrix_sy) state.measurements.correlation_matrix_sy = correlation_matrix_sy.real();
        state.measurements.structure_factor_y = measure::structure_factor(state, correlation_matrix_sy);
    }
    if(not state.measurements.structure_factor_z) {
        auto correlation_matrix_sz = measure::correlation_matrix<Scalar>(state, sz, sz);
        if(not state.measurements.correlation_matrix_sz) state.measurements.correlation_matrix_sz = correlation_matrix_sz.real();
        state.measurements.structure_factor_z = measure::structure_factor(state, correlation_matrix_sz);
    }

    return {state.measurements.structure_factor_x.value(), state.measurements.structure_factor_y.value(), state.measurements.structure_factor_z.value()};
}
