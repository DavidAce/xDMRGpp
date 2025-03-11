#include "tools/infinite/measure.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

size_t tools::infinite::measure::length(const TensorsInfinite &tensors) { return tensors.edges->get_length(); }
size_t tools::infinite::measure::length(const EdgesInfinite &edges) { return edges.get_length(); }

double tools::infinite::measure::norm(const StateInfinite &state) {
    if(state.measurements.norm) return state.measurements.norm.value();
    cx64 norm = tools::common::contraction::contract_mps_norm(state.get_2site_mps());
    if(std::abs(norm - 1.0) > settings::precision::max_norm_error) tools::log->debug("norm: far from unity: {:.16f}{:+.16f}i", norm.real(), norm.imag());
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}

long tools::infinite::measure::bond_dimension(const StateInfinite &state) {
    if(state.measurements.bond_dim) return state.measurements.bond_dim.value();
    state.measurements.bond_dim = state.chiC();
    return state.measurements.bond_dim.value();
}

double tools::infinite::measure::truncation_error(const StateInfinite &state) {
    if(state.measurements.truncation_error) return state.measurements.truncation_error.value();
    state.measurements.truncation_error = state.get_truncation_error();
    return state.measurements.truncation_error.value();
}

double tools::infinite::measure::entanglement_entropy(const StateInfinite &state) {
    if(state.measurements.entanglement_entropy) return state.measurements.entanglement_entropy.value();
    auto                     t_ent          = tid::tic_token("ent");
    const auto              &LC             = state.LC();
    Eigen::Tensor<Scalar, 0> SA             = -LC.square().contract(LC.square().log().eval(), tenx::idx({0}, {0}));
    state.measurements.entanglement_entropy = std::real(SA(0));
    return state.measurements.entanglement_entropy.value();
}

template<typename state_or_mps_type>
double tools::infinite::measure::energy_minus_energy_shift(const state_or_mps_type &state, const ModelInfinite &model, const EdgesInfinite &edges) {
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite>) {
        return tools::infinite::measure::energy_minus_energy_shift(state.get_2site_mps(), model, edges);
    } else {
        tools::log->trace("Measuring energy mpo");
        const auto &mpo          = model.get_2site_mpo_AB();
        const auto &env          = edges.get_env_ene_blk();
        auto        t_ene        = tid::tic_scope("ene");
        auto        e_minus_ered = tools::common::contraction::expectation_value(state, mpo, env.L, env.R);
        assert(std::abs(std::imag(e_minus_ered)) < 1e-10);
        return std::real(e_minus_ered);
    }
}

template double tools::infinite::measure::energy_minus_energy_shift(const StateInfinite &, const ModelInfinite &model, const EdgesInfinite &edges);
template double tools::infinite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &, const ModelInfinite &model, const EdgesInfinite &edges);

template<typename state_or_mps_type>
double tools::infinite::measure::energy_mpo(const state_or_mps_type &state, const ModelInfinite &model, const EdgesInfinite &edges) {
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite>)
        return tools::infinite::measure::energy_mpo(state.get_2site_mps(), model, edges);
    else
        return tools::infinite::measure::energy_minus_energy_shift(state, model, edges) +
               model.get_energy_shift_per_site() * static_cast<double>(edges.get_length());
}

template double tools::infinite::measure::energy_mpo(const StateInfinite &, const ModelInfinite &model, const EdgesInfinite &edges);
template double tools::infinite::measure::energy_mpo(const Eigen::Tensor<Scalar, 3> &, const ModelInfinite &model, const EdgesInfinite &edges);

template<typename state_or_mps_type>
double tools::infinite::measure::energy_per_site_mpo(const state_or_mps_type &state, const ModelInfinite &model, const EdgesInfinite &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_mpo(state, model, edges) / static_cast<double>(2);
}

template double tools::infinite::measure::energy_per_site_mpo(const StateInfinite &, const ModelInfinite &model, const EdgesInfinite &edges);
template double tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<Scalar, 3> &, const ModelInfinite &model, const EdgesInfinite &edges);

template<typename state_or_mps_type>
double tools::infinite::measure::energy_variance_mpo(const state_or_mps_type &state, const ModelInfinite &model, const EdgesInfinite &edges) {
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
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite>) {
        return tools::infinite::measure::energy_variance_mpo(state.get_2site_mps(), model, edges);
    } else {
        tools::log->trace("Measuring energy variance mpo");
        double energy = 0;
        if(model.is_shifted())
            energy = tools::infinite::measure::energy_minus_energy_shift(state, model, edges);
        else
            energy = tools::infinite::measure::energy_mpo(state, model, edges);
        double      E2  = energy * energy;
        const auto &mpo = model.get_2site_mpo_AB();
        const auto &env = edges.get_env_var_blk();
        tools::log->trace("Measuring energy variance mpo");
        auto t_var = tid::tic_scope("var");
        auto H2    = tools::common::contraction::expectation_value(state, mpo, env.L, env.R);
        assert(std::abs(std::imag(H2)) < 1e-10);
        return std::abs(H2 - E2);
    }
}

template double tools::infinite::measure::energy_variance_mpo(const StateInfinite &, const ModelInfinite &model, const EdgesInfinite &edges);
template double tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<Scalar, 3> &, const ModelInfinite &model, const EdgesInfinite &edges);

template<typename state_or_mps_type>
double tools::infinite::measure::energy_variance_per_site_mpo(const state_or_mps_type &state, const ModelInfinite &model, const EdgesInfinite &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_variance_mpo(state, model, edges) / static_cast<double>(2);
}

template double tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite &, const ModelInfinite &model, const EdgesInfinite &edges);
template double tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<Scalar, 3> &, const ModelInfinite &model,
                                                                       const EdgesInfinite &edges);

double tools::infinite::measure::energy_mpo(const TensorsInfinite &tensors) {
    if(tensors.measurements.energy_mpo) return tensors.measurements.energy_mpo.value();
    tensors.measurements.energy_mpo = tools::infinite::measure::energy_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_mpo.value();
}

double tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite &tensors) {
    if(tensors.measurements.energy_per_site_mpo) return tensors.measurements.energy_per_site_mpo.value();
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    auto L                                   = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_per_site_mpo = tools::infinite::measure::energy_mpo(tensors) / static_cast<double>(L);
    return tensors.measurements.energy_per_site_mpo.value();
}

double tools::infinite::measure::energy_variance_mpo(const TensorsInfinite &tensors) {
    if(tensors.measurements.energy_variance_mpo) return tensors.measurements.energy_variance_mpo.value();
    tensors.measurements.energy_variance_mpo = tools::infinite::measure::energy_variance_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_variance_mpo.value();
}

double tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite &tensors) {
    if(tensors.measurements.energy_variance_per_site_mpo) return tensors.measurements.energy_variance_per_site_mpo.value();
    auto L                                            = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_variance_per_site_mpo = tools::infinite::measure::energy_variance_mpo(tensors) / static_cast<double>(L);
    return tensors.measurements.energy_variance_per_site_mpo.value();
}

double tools::infinite::measure::energy_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite &tensors) {
    return tools::infinite::measure::energy_mpo(mps, *tensors.model, *tensors.edges);
}
double tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite &tensors) {
    return tools::infinite::measure::energy_per_site_mpo(mps, *tensors.model, *tensors.edges);
}
double tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite &tensors) {
    return tools::infinite::measure::energy_variance_mpo(mps, *tensors.model, *tensors.edges);
}
double tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite &tensors) {
    return tools::infinite::measure::energy_variance_per_site_mpo(mps, *tensors.model, *tensors.edges);
}
