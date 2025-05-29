#include "../flbit.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "flbit.tmpl.h"
#include "general/iter.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/num.h"
#include "math/svd.h"
#include "qm/lbit.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/entanglement_entropy.h"
#include "tools/finite/measure/number_entropy.h"
#include "tools/finite/mps.h"
#include "tools/finite/ops.h"
#include <complex>
#include <h5pp/h5pp.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

template<typename Scalar>
std::pair<StateFinite<Scalar>, AlgorithmStatus> flbit_tmpl::update_state(const size_t time_index, cx128 time_point, const StateFinite<Scalar> &state_lbit_init,
                                                                         const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                                                         const std::vector<std::vector<qm::Gate>>     &unitary_circuit,
                                                                         const AlgorithmStatus                        &status_init) {
    /*!
     * \fn void update_state()
     */
    tools::log->debug("Starting fLBIT: iter {} | t = {:.2e}", time_index + 1, fp(time_point));
    auto t_step = tid::tic_scope("upd");

    // Time evolve from 0 to time_point[iter] here
    auto state_tevo = time_evolve_lbit_state(state_lbit_init, gates_tevo, status_init);
    state_tevo      = transform_to_real_basis(state_tevo, unitary_circuit, status_init);

    AlgorithmStatus status_tevo = status_init; // Time evolved status
    status_tevo.phys_time       = abs(time_point);
    status_tevo.iter            = time_index + 1;
    status_tevo.step            = time_index * settings::model::model_size;
    status_tevo.position        = state_tevo.template get_position<long>();
    status_tevo.direction       = state_tevo.get_direction();
    return {state_tevo, check_convergence(status_tevo)};
}


template<typename Scalar>
StateFinite<Scalar> flbit_tmpl::time_evolve_lbit_state(const StateFinite<Scalar> &state_lbit_init, const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                                       const AlgorithmStatus &status) {
    auto t_evo      = tid::tic_scope("time_evo", tid::level::normal);
    auto svd_cfg    = svd::config(status.bond_lim, status.trnc_lim);
    svd_cfg.svd_lib = svd::lib::lapacke;
    svd_cfg.svd_rtn = svd::rtn::geauto;
    auto delta_t    = status.delta_t.to_floating_point<cx128>();
    tools::log->debug("Applying time evolution swap gates Δt = {:.2e}", fp(delta_t));
    auto state_lbit_tevo = state_lbit_init;
    for(const auto &gates : gates_tevo) { tools::finite::mps::apply_swap_gates(state_lbit_tevo, gates, CircuitOp::NONE, GateMove::AUTO, svd_cfg); }

    if constexpr(settings::debug) {
        // Check that we would get back the original state if we time evolved backwards
        auto state_lbit_init_debug = state_lbit_tevo;
        for(const auto &gates : iter::reverse(gates_tevo)) {
            tools::finite::mps::apply_swap_gates(state_lbit_init_debug, gates, CircuitOp::ADJ, GateMove::AUTO, svd_cfg);
        }
        tools::finite::mps::normalize_state(state_lbit_init_debug, std::nullopt, NormPolicy::IFNEEDED);
        auto overlap = tools::finite::ops::overlap<Scalar>(state_lbit_init, state_lbit_init_debug);
        tools::log->info("Debug overlap after time evolution: {:.16f}", fp(overlap));
        if(std::abs(overlap - 1.0) > 10 * status.trnc_lim)
            throw except::runtime_error("State overlap after backwards time evolution is not 1: Got {:.16f}", fp(overlap));
    }
    return state_lbit_tevo;
}
template<typename Scalar>
StateFinite<Scalar> flbit_tmpl::transform_to_real_basis(const StateFinite<Scalar> &state_lbit, const std::vector<std::vector<qm::Gate>> &unitary_circuit,
                                                        const AlgorithmStatus &status) {
    assert(unitary_circuit.size() == settings::model::lbit::u_depth);
    auto svd_cfg    = svd::config(status.bond_lim, status.trnc_lim);
    svd_cfg.svd_lib = svd::lib::lapacke;
    svd_cfg.svd_rtn = svd::rtn::geauto;
    return qm::lbit::transform_to_real_basis(state_lbit, unitary_circuit, svd_cfg);
}

inline AlgorithmStatus flbit_tmpl::check_convergence(const AlgorithmStatus &status_init) {
    auto status = status_init;
    if(status.entanglement_saturated_for > 0)
        status.algorithm_saturated_for++;
    else
        status.algorithm_saturated_for = 0;

    status.algorithm_converged_for = status.iter + 1 - std::min(settings::flbit::time_num_steps, status.iter + 1);
    status.algorithm_has_succeeded = status.algorithm_converged_for > 0;
    if(status.algorithm_saturated_for > 0 and status.algorithm_converged_for == 0)
        status.algorithm_has_stuck_for++;
    else
        status.algorithm_has_stuck_for = 0;

    status.algorithm_has_to_stop = false; // Never stop due to saturation

    tools::log->debug("Simulation report: converged {} | saturated {} | stuck {} | succeeded {} | has to stop {}", status.algorithm_converged_for,
                      status.algorithm_saturated_for, status.algorithm_has_stuck_for, status.algorithm_has_succeeded, status.algorithm_has_to_stop);
    status.algo_stop = AlgorithmStop::NONE;
    if(status.iter >= settings::flbit::iter_min) {
        if(status.iter >= settings::flbit::iter_max) status.algo_stop = AlgorithmStop::MAX_ITERS;
        if(status.iter >= settings::flbit::time_num_steps) status.algo_stop = AlgorithmStop::SUCCESS;
        if(status.algorithm_has_to_stop) status.algo_stop = AlgorithmStop::SATURATED;
    }
    return status;
}

template<typename Scalar>
void flbit_tmpl::print_status(const StateFinite<Scalar> &state_real, const AlgorithmStatus &status) {
    if(num::mod(status.iter, settings::print_freq(status.algo_type)) != 0) return;
    if(settings::print_freq(status.algo_type) == 0) return;
    auto        t_print = tid::tic_scope("print", tid::level::normal);
    std::string report;
    report += fmt::format("{:<} ", state_real.get_name());
    report += fmt::format("iter:{:<4} ", status.iter);
    report += fmt::format("step:{:<5} ", status.step);
    report += fmt::format("t:{:>2}/{:<2}", omp_get_thread_num(), omp_get_num_threads());
    report += fmt::format("L:{} ", state_real.get_length());
    report += fmt::format("l:{:<2} ", state_real.get_position());
    report += fmt::format("ε:{:<8.2e} ", state_real.get_truncation_error_midchain());
    report += fmt::format("Sₑ(L/2):{:<18.16f} ", fp(tools::finite::measure::entanglement_entropy_midchain(state_real)));
    report += fmt::format("Sₙ(L/2):{:<18.16f} ", fp(tools::finite::measure::number_entropy_midchain<Scalar>(state_real)));
    report += fmt::format("χ:{:<3}|{:<3}|{:<3} ", status.bond_max, status.bond_lim, tools::finite::measure::bond_dimension_midchain(state_real));
    if(settings::flbit::time_scale == TimeScale::LOGSPACED)
        report += fmt::format("ptime:{:<} ", fmt::format("{:>.2e}s", status.phys_time.to_floating_point<fp64>()));
    if(settings::flbit::time_scale == TimeScale::LINSPACED)
        report += fmt::format("ptime:{:<} ", fmt::format("{:>.6f}s", status.phys_time.to_floating_point<fp64>()));
    report += fmt::format("wtime:{:<} ", fmt::format("{:>.1f}s", status.wall_time));
    report += fmt::format("mem[rss {:<.1f}|peak {:<.1f}|vm {:<.1f}]MB ", debug::mem_rss_in_mb(), debug::mem_hwm_in_mb(), debug::mem_vm_in_mb());
    tools::log->info(report);
}
