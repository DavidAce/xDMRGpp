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
template<typename Scalar>
size_t tools::infinite::measure::length(const TensorsInfinite<Scalar> &tensors) {
    return tensors.edges->get_length();
}
template size_t tools::infinite::measure::length(const TensorsInfinite<fp32> &tensors);
template size_t tools::infinite::measure::length(const TensorsInfinite<fp64> &tensors);
template size_t tools::infinite::measure::length(const TensorsInfinite<fp128> &tensors);
template size_t tools::infinite::measure::length(const TensorsInfinite<cx32> &tensors);
template size_t tools::infinite::measure::length(const TensorsInfinite<cx64> &tensors);
template size_t tools::infinite::measure::length(const TensorsInfinite<cx128> &tensors);

template<typename Scalar>
size_t tools::infinite::measure::length(const EdgesInfinite<Scalar> &edges) {
    return edges.get_length();
}
template size_t tools::infinite::measure::length(const EdgesInfinite<fp32> &edges);
template size_t tools::infinite::measure::length(const EdgesInfinite<fp64> &edges);
template size_t tools::infinite::measure::length(const EdgesInfinite<fp128> &edges);
template size_t tools::infinite::measure::length(const EdgesInfinite<cx32> &edges);
template size_t tools::infinite::measure::length(const EdgesInfinite<cx64> &edges);
template size_t tools::infinite::measure::length(const EdgesInfinite<cx128> &edges);

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::norm(const StateInfinite<Scalar> &state) {
    if(state.measurements.norm) return state.measurements.norm.value();
    Scalar norm = tools::common::contraction::contract_mps_norm(state.get_2site_mps());
    if(std::abs(norm - RealScalar<Scalar>{1}) > static_cast<RealScalar<Scalar>>(settings::precision::max_norm_error))
        tools::log->debug("norm: far from unity: {:.16f}", fp(norm));
    state.measurements.norm = std::abs(norm);
    return state.measurements.norm.value();
}
template fp32  tools::infinite::measure::norm(const StateInfinite<fp32> &state);
template fp64  tools::infinite::measure::norm(const StateInfinite<fp64> &state);
template fp128 tools::infinite::measure::norm(const StateInfinite<fp128> &state);
template fp32  tools::infinite::measure::norm(const StateInfinite<cx32> &state);
template fp64  tools::infinite::measure::norm(const StateInfinite<cx64> &state);
template fp128 tools::infinite::measure::norm(const StateInfinite<cx128> &state);

template<typename Scalar>
long tools::infinite::measure::bond_dimension(const StateInfinite<Scalar> &state) {
    if(state.measurements.bond_dim) return state.measurements.bond_dim.value();
    state.measurements.bond_dim = state.chiC();
    return state.measurements.bond_dim.value();
}
template long tools::infinite::measure::bond_dimension(const StateInfinite<fp32> &state);
template long tools::infinite::measure::bond_dimension(const StateInfinite<fp64> &state);
template long tools::infinite::measure::bond_dimension(const StateInfinite<fp128> &state);
template long tools::infinite::measure::bond_dimension(const StateInfinite<cx32> &state);
template long tools::infinite::measure::bond_dimension(const StateInfinite<cx64> &state);
template long tools::infinite::measure::bond_dimension(const StateInfinite<cx128> &state);

template<typename Scalar>
double tools::infinite::measure::truncation_error(const StateInfinite<Scalar> &state) {
    if(state.measurements.truncation_error) return state.measurements.truncation_error.value();
    state.measurements.truncation_error = state.get_truncation_error();
    return state.measurements.truncation_error.value();
}
template double tools::infinite::measure::truncation_error(const StateInfinite<fp32> &state);
template double tools::infinite::measure::truncation_error(const StateInfinite<fp64> &state);
template double tools::infinite::measure::truncation_error(const StateInfinite<fp128> &state);
template double tools::infinite::measure::truncation_error(const StateInfinite<cx32> &state);
template double tools::infinite::measure::truncation_error(const StateInfinite<cx64> &state);
template double tools::infinite::measure::truncation_error(const StateInfinite<cx128> &state);

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::entanglement_entropy(const StateInfinite<Scalar> &state) {
    if(state.measurements.entanglement_entropy) return state.measurements.entanglement_entropy.value();
    auto                     t_ent          = tid::tic_token("ent");
    const auto              &LC             = state.LC();
    Eigen::Tensor<Scalar, 0> SA             = -LC.square().contract(LC.square().log().eval(), tenx::idx({0}, {0}));
    state.measurements.entanglement_entropy = std::real(SA(0));
    return state.measurements.entanglement_entropy.value();
}
template RealScalar<fp32>  tools::infinite::measure::entanglement_entropy(const StateInfinite<fp32> &state);
template RealScalar<fp64>  tools::infinite::measure::entanglement_entropy(const StateInfinite<fp64> &state);
template RealScalar<fp128> tools::infinite::measure::entanglement_entropy(const StateInfinite<fp128> &state);
template RealScalar<fp32>  tools::infinite::measure::entanglement_entropy(const StateInfinite<cx32> &state);
template RealScalar<fp64>  tools::infinite::measure::entanglement_entropy(const StateInfinite<cx64> &state);
template RealScalar<fp128> tools::infinite::measure::entanglement_entropy(const StateInfinite<cx128> &state);

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

/* clang-format off */
template fp32  tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<fp32> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<fp64> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<fp128> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<cx32> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<cx64> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<cx128> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
template fp32  tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<fp32, 3> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<fp64, 3> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<fp128, 3> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<cx32, 3> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<cx64, 3> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<cx128, 3> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
/* clang-format on */

template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                        const EdgesInfinite<Scalar> &edges) {
    if constexpr(std::is_same_v<state_or_mps_type, StateInfinite<Scalar>>)
        return tools::infinite::measure::energy_mpo(state.get_2site_mps(), model, edges);
    else
        return tools::infinite::measure::energy_minus_energy_shift(state, model, edges) +
               std::real(model.get_energy_shift_per_site()) * static_cast<RealScalar<Scalar>>(edges.get_length());
}

/* clang-format off */
template fp32  tools::infinite::measure::energy_mpo(const StateInfinite<fp32> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_mpo(const StateInfinite<fp64> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_mpo(const StateInfinite<fp128> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_mpo(const StateInfinite<cx32> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_mpo(const StateInfinite<cx64> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_mpo(const StateInfinite<cx128> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
template fp32  tools::infinite::measure::energy_mpo(const  Eigen::Tensor<fp32, 3> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_mpo(const  Eigen::Tensor<fp64, 3> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_mpo(const  Eigen::Tensor<fp128, 3> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_mpo(const  Eigen::Tensor<cx32, 3> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_mpo(const  Eigen::Tensor<cx64, 3> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_mpo(const  Eigen::Tensor<cx128, 3> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
/* clang-format on */

template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                 const EdgesInfinite<Scalar> &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_mpo(state, model, edges) / RealScalar<Scalar>{2};
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_per_site_mpo(const StateInfinite<fp32> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_per_site_mpo(const StateInfinite<fp64> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_per_site_mpo(const StateInfinite<fp128> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_per_site_mpo(const StateInfinite<cx32> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_per_site_mpo(const StateInfinite<cx64> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_per_site_mpo(const StateInfinite<cx128> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
template fp32  tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<fp32, 3> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<fp64, 3> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<fp128, 3> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<cx32, 3> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<cx64, 3> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<cx128, 3> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
/* clang-format on */

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
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_mpo(const StateInfinite<fp32> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_variance_mpo(const StateInfinite<fp64> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_variance_mpo(const StateInfinite<fp128> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_variance_mpo(const StateInfinite<cx32> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_variance_mpo(const StateInfinite<cx64> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_variance_mpo(const StateInfinite<cx128> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
template fp32  tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<fp32, 3> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<fp64, 3> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<fp128, 3> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<cx32, 3> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<cx64, 3> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<cx128, 3> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
/* clang-format on */

template<typename state_or_mps_type, typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const state_or_mps_type &state, const ModelInfinite<Scalar> &model,
                                                                          const EdgesInfinite<Scalar> &edges) {
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    return tools::infinite::measure::energy_variance_mpo(state, model, edges) / RealScalar<Scalar>{2};
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<fp32> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<fp64> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<fp128> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<cx32> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<cx64> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<cx128> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<fp32, 3> &, const ModelInfinite<fp32> &model, const EdgesInfinite<fp32> &edges);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<fp64, 3> &, const ModelInfinite<fp64> &model, const EdgesInfinite<fp64> &edges);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<fp128, 3> &, const ModelInfinite<fp128> &model, const EdgesInfinite<fp128> &edges);
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<cx32, 3> &, const ModelInfinite<cx32> &model, const EdgesInfinite<cx32> &edges);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<cx64, 3> &, const ModelInfinite<cx64> &model, const EdgesInfinite<cx64> &edges);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<cx128, 3> &, const ModelInfinite<cx128> &model, const EdgesInfinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_mpo) return tensors.measurements.energy_mpo.value();
    tensors.measurements.energy_mpo = tools::infinite::measure::energy_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_mpo.value();
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_mpo(const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_mpo(const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_mpo(const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_mpo(const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_mpo(const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_mpo(const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_per_site_mpo) return tensors.measurements.energy_per_site_mpo.value();
    tools::log->warn("energy_per_site_mpo: CHECK DIVISION");
    auto L                                   = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_per_site_mpo = tools::infinite::measure::energy_mpo(tensors) / static_cast<RealScalar<Scalar>>(L);
    return tensors.measurements.energy_per_site_mpo.value();
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_mpo) return tensors.measurements.energy_variance_mpo.value();
    tensors.measurements.energy_variance_mpo = tools::infinite::measure::energy_variance_mpo(*tensors.state, *tensors.model, *tensors.edges);
    return tensors.measurements.energy_variance_mpo.value();
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_per_site_mpo) return tensors.measurements.energy_variance_per_site_mpo.value();
    auto L                                            = tools::infinite::measure::length(tensors);
    tensors.measurements.energy_variance_per_site_mpo = tools::infinite::measure::energy_variance_mpo(tensors) / static_cast<RealScalar<Scalar>>(L);
    return tensors.measurements.energy_variance_per_site_mpo.value();
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_mpo(mps, *tensors.model, *tensors.edges);
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_mpo(const Eigen::Tensor<fp32,3> &mps, const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_mpo(const Eigen::Tensor<fp64,3> &mps, const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_mpo(const Eigen::Tensor<fp128,3> &mps, const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_mpo(const Eigen::Tensor<cx32,3> &mps, const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_mpo(const Eigen::Tensor<cx64,3> &mps, const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_mpo(const Eigen::Tensor<cx128,3> &mps, const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_per_site_mpo(mps, *tensors.model, *tensors.edges);
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<fp32,3> &mps, const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<fp64,3> &mps, const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<fp128,3> &mps, const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<cx32,3> &mps, const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<cx64,3> &mps, const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<cx128,3> &mps, const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_variance_mpo(mps, *tensors.model, *tensors.edges);
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<fp32,3> &mps, const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<fp64,3> &mps, const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<fp128,3> &mps, const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<cx32,3> &mps, const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<cx64,3> &mps, const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<cx128,3> &mps, const TensorsInfinite<cx128> &);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<Scalar, 3> &mps, const TensorsInfinite<Scalar> &tensors) {
    return tools::infinite::measure::energy_variance_per_site_mpo(mps, *tensors.model, *tensors.edges);
}
/* clang-format off */
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<fp32,3> &mps, const TensorsInfinite<fp32> &);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<fp64,3> &mps, const TensorsInfinite<fp64> &);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<fp128,3> &mps, const TensorsInfinite<fp128> &);
template fp32  tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<cx32,3> &mps, const TensorsInfinite<cx32> &);
template fp64  tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<cx64,3> &mps, const TensorsInfinite<cx64> &);
template fp128 tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<cx128,3> &mps, const TensorsInfinite<cx128> &);
/* clang-format on */
