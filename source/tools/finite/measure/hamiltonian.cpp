#include "hamiltonian.h"
#include "config/settings.h"
#include "expectation_value.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"

namespace settings {
    constexpr bool debug_hamiltonian = false;
}

using tools::finite::measure::RealScalar;

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_ene_active();
    auto t_H2 = tid::tic_scope("H", tid::level::highest);

    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}
/* clang-format off */
template cx64 tools::finite::measure::expval_hamiltonian<cx64>  (const Eigen::Tensor<cx64, 3> &mps , const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian<cx128>(const Eigen::Tensor<cx128, 3> &mps , const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    auto mps = state.get_mps_active();
    auto mpo = model.get_mpo_active();
    auto env = edges.get_ene_active();
    auto t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}
/* clang-format off */
template cx64 tools::finite::measure::expval_hamiltonian<cx64>(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian<cx128>(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                  const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.get_mps(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_ene(sites);
    auto        t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}
/* clang-format off */
template cx64 tools::finite::measure::expval_hamiltonian<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                  const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                  const env_pair<const EnvEne<Scalar> &>                           &envs) {
    auto t_H = tid::tic_scope("H", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo_refs, envs);
}

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3> &mps, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    auto mpo = model.get_mpo_active();
    auto env = edges.get_var_active();
    auto t_H = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}

/* clang-format off */
template cx64  tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx64, 3> &mps, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx128, 3> &mps, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    auto mps  = state.get_mps_active();
    auto mpo  = model.get_mpo_active();
    auto env  = edges.get_var_active();
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}
/* clang-format off */
template cx64  tools::finite::measure::expval_hamiltonian_squared<cx64>(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian_squared<cx128>(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const std::vector<size_t> &sites, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    const auto &mps = state.get_mps(sites);
    const auto &mpo = model.get_mpo(sites);
    const auto &env = edges.get_multisite_env_var(sites);
    auto        t_H = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo, env);
}
/* clang-format off */
template cx64 tools::finite::measure::expval_hamiltonian_squared<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges);
template cx128 tools::finite::measure::expval_hamiltonian_squared<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<Scalar, 3>                                   &mps,
                                                          const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpo_refs,
                                                          const env_pair<const EnvVar<Scalar> &>                           &envs) {
    auto t_H2 = tid::tic_scope("H2", tid::level::highest);
    return tools::finite::measure::expectation_value<Scalar>(mps, mps, mpo_refs, envs);
}
/* clang-format off */
template cx64  tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpo_refs, const env_pair<const EnvVar<cx64> &>                           &envs);
template cx128 tools::finite::measure::expval_hamiltonian_squared(const Eigen::Tensor<cx128, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpo_refs, const env_pair<const EnvVar<cx128> &>                           &envs);
/* clang-format on */

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::measure::expval_hamiltonian<Scalar>(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}
template cx64  tools::finite::measure::expval_hamiltonian(const TensorsFinite<cx64> &tensors);
template cx128 tools::finite::measure::expval_hamiltonian(const TensorsFinite<cx128> &tensors);

template<typename Scalar>
Scalar tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::measure::expval_hamiltonian_squared<Scalar>(*tensors.state, *tensors.model, *tensors.edges);
}
template cx64  tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite<cx64> &tensors);
template cx128 tools::finite::measure::expval_hamiltonian_squared(const TensorsFinite<cx128> &tensors);

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                     const EdgesFinite<Scalar> &edges, MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_hamiltonian) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    if constexpr(settings::debug) tools::log->trace("Measuring energy: sites {}", state.active_sites);
    auto e_minus_ered = expval_hamiltonian<Scalar>(state, model, edges);
    if constexpr(settings::debug_hamiltonian) {
        const auto &multisite_mps = state.template get_multisite_mps<Scalar>();
        const auto &multisite_mpo = model.template get_multisite_mpo<Scalar>();
        const auto &multisite_env = edges.get_multisite_env_ene_blk();
        auto        edbg          = tools::common::contraction::expectation_value(multisite_mps, multisite_mpo, multisite_env.L, multisite_env.R);
        tools::log->trace("e_minus_ered: {:.16f}", fp(e_minus_ered));
        tools::log->trace("e_minus_edbg: {:.16f}", fp(edbg));
        if(measurements != nullptr and measurements->energy_minus_energy_shift) {
            tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
            assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
        }
        assert(std::abs(e_minus_ered - edbg) < 1e-14);
    }

    assert(std::abs(std::imag(e_minus_ered)) < 1e-10);
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}
/* clang-format off */
template fp64 tools::finite::measure::energy_minus_energy_shift(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy_minus_energy_shift(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                                     const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                                     MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy_minus_energy_shift) {
        if constexpr(!settings::debug_hamiltonian) {
            // Return the cache hit when not debugging. Otherwise, check that it is correct!
            // tools::log->trace("energy_minus_energy_shift: cache hit: {:.16f}", measurements->energy_minus_energy_shift.value());
            return measurements->energy_minus_energy_shift.value();
        }
    }
    auto t_ene = tid::tic_scope("ene", tid::level::highest);
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(model.active_sites, edges.active_sites));
    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    auto e_minus_ered = Scalar(std::numeric_limits<RealScalar<Scalar>>::quiet_NaN(), std::numeric_limits<RealScalar<Scalar>>::quiet_NaN());
    if(model.active_sites.size() <= 3) {
        // Contract directly
        const auto &mpo = model.template get_multisite_mpo<Scalar>();
        const auto &env = edges.get_multisite_env_ene_blk();
        if constexpr(settings::debug_hamiltonian)
            tools::log->trace("Measuring energy: multisite_mps dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo.dimensions(), edges.active_sites, env.L.dimensions(), env.R.dimensions());
        e_minus_ered = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
        if constexpr(settings::debug_hamiltonian) {
            // Split the multisite mps first
            const auto mpos = model.get_mpo_active();
            const auto envs = edges.get_ene_active();
            const auto edbg = tools::finite::measure::expectation_value<Scalar>(multisite_mps, mpos, envs, svd_cfg);
            tools::log->trace("e_minus_ered: {:.16f}{:+.16f}i", e_minus_ered.real(), e_minus_ered.imag());
            tools::log->trace("e_minus_edbg: {:.16f}{:+.16f}i", edbg.real(), edbg.imag());
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
            }
            assert(std::abs(e_minus_ered - edbg) < 1e-14);
        }
    } else {
        model.clear_cache();
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_ene_active();
        tools::log->trace("Measuring energy: multisite_mps dims {} | sites {} | eshift {:.16f} | norm {:.16f}", multisite_mps.dimensions(), model.active_sites,
                          fp(model.get_energy_shift_mpo()), fp(tenx::norm(multisite_mps)));
        e_minus_ered = tools::finite::measure::expectation_value<Scalar>(multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_hamiltonian) {
            const auto &mpo  = model.template get_multisite_mpo<cx64>();
            const auto &env  = edges.get_multisite_env_ene_blk();
            const auto  edbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("e_minus_ered: {:.16f}", fp(e_minus_ered));
            tools::log->trace("e_minus_edbg: {:.16f}", fp(edbg));
            if(measurements != nullptr and measurements->energy_minus_energy_shift) {
                tools::log->trace("e_minus_ehit: {:.16f}", measurements->energy_minus_energy_shift.value());
                assert(std::abs(e_minus_ered - measurements->energy_minus_energy_shift.value()) < 1e-14);
            }
            assert(std::abs(e_minus_ered - edbg) < 1e-10);
        }
    }
    assert(std::abs(std::imag(e_minus_ered)) < 1e-10);
    if(measurements != nullptr) measurements->energy_minus_energy_shift = std::real(e_minus_ered);
    return std::real(e_minus_ered);
}
/* clang-format off */
template fp64 tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<cx128, 3> &multisite_mps, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(state, model, edges, measurements);
    auto eshift         = std::real(model.get_energy_shift_mpo());
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}
/* clang-format off */
template fp64 tools::finite::measure::energy (const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                  const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                  MeasurementsTensorsFinite<Scalar> *measurements) {
    if(measurements != nullptr and measurements->energy) return measurements->energy.value();
    // This measures the actual energy of the system regardless of the energy shift in the MPO's
    // If they are shifted, then
    //      "Actual energy" = (E - E_shift) + E_shift = (~0) + E_shift = E
    // Else
    //      "Actual energy" = (E - E_shift) + E_shift = E  + 0 = E
    auto e_minus_eshift = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    auto eshift         = std::real(model.get_energy_shift_mpo());
    auto energy         = e_minus_eshift + eshift;
    if(measurements != nullptr) measurements->energy = energy;
    return energy;
}
/* clang-format off */
template fp64  tools::finite::measure::energy(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy(const Eigen::Tensor<cx128, 3> &multisite_mps, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                           MeasurementsTensorsFinite<Scalar> *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo²'s is equivalent to the usual way.
    // If mpo's are shifted in the mpo²:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not state.active_sites.empty());
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    assert(num::all_equal(state.active_sites, model.active_sites, edges.active_sites));
    if constexpr(settings::debug_hamiltonian) tools::log->trace("Measuring energy variance: sites {}", state.active_sites);
    auto E  = expval_hamiltonian<Scalar>(state, model, edges);
    auto E2 = E * E;
    auto H2 = expval_hamiltonian_squared<Scalar>(state, model, edges);
    // #pragma message "remove minus one for double parity shift test"
    // H2-=1.0;
    assert(std::abs(std::imag(H2)) < 1e-10);
    RealScalar<Scalar> var = std::abs(H2 - E2);
    if constexpr(settings::debug_hamiltonian) tools::log->trace("Variance |H2-E2| = |{:.16f} - {:.16f}| = {:.16f}", std::real(H2), E2, var);
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}
/* clang-format off */
template fp64 tools::finite::measure::energy_variance(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy_variance(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format on */

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                           const EdgesFinite<Scalar> &edges, std::optional<svd::config> svd_cfg,
                                                           MeasurementsTensorsFinite<Scalar> *measurements) {
    // Here we show that the variance calculated with energy-shifted mpo's is equivalent to the usual way.
    // If mpo's are shifted:
    //      Var H = <(H-E_shf)²> - <H-E_shf>²     = <H²>  - 2<H>E_shf + E_shf² - (<H> - E_shf)²
    //                                            = H²    - 2*E*E_shf + E_shf² - E² + 2*E*E_shf - E_shf²
    //                                            = H²    - E²
    //      Note that in the last line, H²-E² is a subtraction of two large numbers --> catastrophic cancellation --> loss of precision.
    //      On the other hand Var H = <(H-E_shf)²> - energy_minus_energy_shift² = <(H-E_red)²> - ~dE², where both terms are always  << 1.
    //      The first term is computed from a double-layer of shifted mpo's.
    //      In the second term dE is usually very small, in fact identically zero immediately after an energy-reduction operation,
    //      but may grow if the optimization steps make significant progress refining E. Thus wethe first term is a good approximation to
    //      the variance by itself.
    //
    // Else, if E_shf = 0 (i.e. not shifted) we get the usual formula:
    //      Var H = <(H - 0)²> - <H - 0>² = H² - E²
    if(measurements != nullptr and measurements->energy_variance) return measurements->energy_variance.value();
    assert(not model.active_sites.empty());
    assert(not edges.active_sites.empty());
    if(not num::all_equal(model.active_sites, edges.active_sites))
        throw std::runtime_error(
            fmt::format("Could not compute energy variance: active sites are not equal: model {} | edges {}", model.active_sites, edges.active_sites));
    RealScalar<Scalar> energy = tools::finite::measure::energy_minus_energy_shift(multisite_mps, model, edges, svd_cfg, measurements);
    RealScalar<Scalar> E2     = energy * energy;

    auto t_var = tid::tic_scope("var", tid::level::highest);

    // Check if we can contract directly or if we need to use the split method
    // Normally it's only worth splitting the multisite mps when it has more than 3 sites
    Scalar H2 = Scalar(std::numeric_limits<RealScalar<Scalar>>::quiet_NaN(), std::numeric_limits<RealScalar<Scalar>>::quiet_NaN());
    if(model.active_sites.size() <= 3) {
        // Direct contraction
        const auto &mpo2 = model.template get_multisite_mpo_squared<Scalar>();
        const auto &env2 = edges.get_multisite_env_var_blk();
        if constexpr(settings::debug)
            tools::log->trace("Measuring energy variance: state dims {} | model sites {} dims {} | edges sites {} dims [L{} R{}]", multisite_mps.dimensions(),
                              model.active_sites, mpo2.dimensions(), edges.active_sites, env2.L.dimensions(), env2.R.dimensions());

        if(multisite_mps.dimension(0) != mpo2.dimension(2))
            throw std::runtime_error(fmt::format("State and model have incompatible physical dimension: state dim {} | model dim {}",
                                                 multisite_mps.dimension(0), mpo2.dimension(2)));
        H2 = tools::common::contraction::expectation_value(multisite_mps, mpo2, env2.L, env2.R);
    } else {
        // Split the multisite mps first
        const auto mpos = model.get_mpo_active();
        const auto envs = edges.get_var_active();
        tools::log->trace("Measuring energy variance: state dims {} | sites {}", multisite_mps.dimensions(), model.active_sites);
        H2 = tools::finite::measure::expectation_value<Scalar>(multisite_mps, multisite_mps, mpos, envs, svd_cfg);
        if constexpr(settings::debug_hamiltonian) {
            const auto &mpo   = model.template get_multisite_mpo_squared<cx64>();
            const auto &env   = edges.get_multisite_env_var_blk();
            const auto  H2dbg = tools::common::contraction::expectation_value(multisite_mps, mpo, env.L, env.R);
            tools::log->trace("H2   : {:.16f}{:+.16f}i", H2.real(), H2.imag());
            tools::log->trace("H2dbg: {:.16f}{:+.16f}i", H2dbg.real(), H2dbg.imag());
            assert(std::abs(H2 - H2dbg) < 1e-10);
        }
    }
    assert(std::abs(std::imag(H2)) < 1e-10);
    // #pragma message "remove minus one for double parity shift test"
    // H2-=1.0;
    RealScalar<Scalar> var = std::abs(H2 - E2);
    // tools::log->info("Var H = H² - E² = {:.16f} - {:.16f} = {:.16f}", std::real(H2), E2, var);
    if(measurements != nullptr) measurements->energy_variance = var;
    return var;
}
/* clang-format off */
template fp64  tools::finite::measure::energy_variance(const Eigen::Tensor<cx64, 3> &multisite_mps, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy_variance(const Eigen::Tensor<cx128, 3> &multisite_mps, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format off */


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                 RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max, MeasurementsTensorsFinite<Scalar> *measurements) {
    return (tools::finite::measure::energy(state, model, edges, measurements) - energy_min) / (energy_max - energy_min);
}
/* clang-format off */
template fp64  tools::finite::measure::energy_normalized(const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, fp64 energy_min, fp64 energy_max, MeasurementsTensorsFinite<cx64> *measurements);
template fp128 tools::finite::measure::energy_normalized(const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, fp128 energy_min, fp128 energy_max, MeasurementsTensorsFinite<cx128> *measurements);
/* clang-format off */


template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &multisite_mps, const ModelFinite<Scalar> &model,
                                                 const EdgesFinite<Scalar> &edges, RealScalar<Scalar> energy_min, RealScalar<Scalar> energy_max, std::optional<svd::config> svd_cfg,
                                                 MeasurementsTensorsFinite<Scalar> *measurements) {
    return (tools::finite::measure::energy(multisite_mps, model, edges, svd_cfg, measurements) - energy_min) / (energy_max - energy_min);
}

template<typename Scalar>
RealScalar<Scalar>
tools::finite::measure::energy_shift(const TensorsFinite<Scalar> &tensors) { return tensors.model->get_energy_shift_mpo(); }

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const TensorsFinite<Scalar> &tensors) {
    tensors.assert_edges_ene();
    return energy_minus_energy_shift(*tensors.state, tensors.get_model(), tensors.get_edges(), &tensors.measurements);
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy(const TensorsFinite<Scalar> &tensors) {
    if(not tensors.measurements.energy) {
        tensors.assert_edges_ene();
        tensors.measurements.energy = tools::finite::measure::energy(tensors.get_state(), tensors.get_model(), tensors.get_edges(), &tensors.measurements);
    }
    return tensors.measurements.energy.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::finite::measure::energy_variance(const TensorsFinite<Scalar> &tensors) {
    if(not tensors.measurements.energy_variance) {
        tensors.assert_edges_var();
        tensors.measurements.energy_variance = tools::finite::measure::energy_variance(*tensors.state, *tensors.model, *tensors.edges, &tensors.measurements);
    }
    return tensors.measurements.energy_variance.value();
}



    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_normalized(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin, RealScalar<Scalar> emax) {
        tensors.assert_edges_ene();
        return energy_normalized(*tensors.state, tensors.get_model(), tensors.get_edges(), emin, emax);
    }

    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors,
                                                 MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_minus_energy_shift(state, tensors.get_model(), tensors.get_edges(), measurements);
    }
    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors, MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy(state, tensors.get_model(), tensors.get_edges(), measurements);
    }

    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_variance(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors,
                                       MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_variance(state, tensors.get_model(), tensors.get_edges(), measurements);
    }

    template<typename Scalar>
    RealScalar<Scalar> energy_normalized(const StateFinite<Scalar> &state, const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin,
                                         RealScalar<Scalar> emax, MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_normalized(state, tensors.get_model(), tensors.get_edges(), emin, emax, measurements);
    }

    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_minus_energy_shift(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg,
                                                 MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_minus_energy_shift(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
    }
    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg,
                              MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
    }

    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_variance(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg,
                                       MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_variance(mps, tensors.get_model(), tensors.get_edges(), svd_cfg, measurements);
    }

    template<typename Scalar>
    RealScalar<Scalar> tools::finite::measure::energy_normalized(const Eigen::Tensor<Scalar, 3> &mps, const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> emin,
                                         RealScalar<Scalar> emax, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite<Scalar> *measurements) {
        return energy_normalized(mps, tensors.get_model(), tensors.get_edges(), emin, emax, svd_cfg, measurements);
    }
