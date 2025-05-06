#pragma once
#include "tools/infinite/measure.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
using tools::infinite::measure::RealScalar;

template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_minus_energy_shift(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                       const EdgesInfinite<Scalar> &edges) {
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite<Scalar>>) {
        return tools::infinite::measure::energy_minus_energy_shift(state.get_2site_mps(), model, edges);
    } else {
        tools::log->trace("Measuring energy mpo");
        const auto &mpo          = model.get_2site_mpo_AB();
        const auto &env          = edges.get_env_ene_blk();
        auto        t_ene        = tid::tic_scope("ene");
        auto        e_minus_ered = tools::common::contraction::expectation_value(state, mpo, env.L, env.R);
        assert(std::abs(std::imag(e_minus_ered)) < RealScalar<Scalar>{1e-10f});
        return std::real(e_minus_ered);
    }
}


template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                        const EdgesInfinite<Scalar> &edges) {
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite<Scalar>>)
        return tools::infinite::measure::energy_mpo(state.get_2site_mps(), model, edges);
    else
        return tools::infinite::measure::energy_minus_energy_shift(state, model, edges) +
               std::real(model.get_energy_shift_per_site()) * static_cast<RealScalar<Scalar>>(edges.get_length());
}


template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                 const EdgesInfinite<Scalar> &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_mpo(state, model, edges) / RealScalar<Scalar>{2};
}


template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                 const EdgesInfinite<Scalar> &edges) {
    // Depending on whether the mpo's are energy-shifted or not we get different formulas.
    // If mpo's are shifted:
    //      Var H = <(H-E_shf)²> - <(H-E_shf)>² = <H²> - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                          = H²   - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                          = H²   - E²
    //      so Var H = <(H-E_red)²> - energy_minus_energy_shift² = H² - ~0
    //      where H² is computed with shifted mpo's. Note that ~0 is not exactly zero
    //      because E_shf != E necessarily (though they are supposed to be very close)
    // Else:
    //      Var H = <(H - 0)²> - <H - 0>² = H2 - E²
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite<Scalar>>) {
        return tools::infinite::measure::energy_variance_mpo(state.get_2site_mps(), model, edges);
    } else {
        tools::log->trace("Measuring energy variance mpo");
        RealScalar<Scalar> energy = 0;
        if(model.is_shifted())
            energy = tools::infinite::measure::energy_minus_energy_shift(state, model, edges);
        else
            energy = tools::infinite::measure::energy_mpo(state, model, edges);
        RealScalar<Scalar> E2  = energy * energy;
        const auto        &mpo = model.get_2site_mpo_AB();
        const auto        &env = edges.get_env_var_blk();
        tools::log->trace("Measuring energy variance mpo");
        auto t_var = tid::tic_scope("var");
        auto H2    = tools::common::contraction::expectation_value(state, mpo, env.L, env.R);
        assert(std::abs(std::imag(H2)) < RealScalar<Scalar>{1e-10f});
        return std::abs(H2 - E2);
    }
}


template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                          const EdgesInfinite<Scalar> &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_variance_mpo(state, model, edges) / RealScalar<Scalar>{2};
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_mpo) return tensors.measurements.energy_mpo.value();
    tensors.measurements.energy_mpo = tools::infinite::measure::energy_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_mpo.value();
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_per_site_mpo) return tensors.measurements.energy_per_site_mpo.value();
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    auto L                                   = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_per_site_mpo = tools::infinite::measure::energy_mpo(tensors) / static_cast<RealScalar<Scalar>>(L);
    return tensors.measurements.energy_per_site_mpo.value();
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_mpo) return tensors.measurements.energy_variance_mpo.value();
    tensors.measurements.energy_variance_mpo = tools::infinite::measure::energy_variance_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_variance_mpo.value();
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_per_site_mpo) return tensors.measurements.energy_variance_per_site_mpo.value();
    auto L                                            = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_variance_per_site_mpo = tools::infinite::measure::energy_variance_mpo(tensors) / static_cast<RealScalar<Scalar>>(L);
    return tensors.measurements.energy_variance_per_site_mpo.value();
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_mpo(mps, *tensors.model, *tensors.edges);
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_per_site_mpo(mps, *tensors.model, *tensors.edges);
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_variance_mpo(mps, *tensors.model, *tensors.edges);
}


template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_variance_per_site_mpo(mps, *tensors.model, *tensors.edges);
}
