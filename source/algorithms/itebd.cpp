#include "itebd.h"
#include "config/settings.h"
#include "qm/time.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/infinite/opt.h"

// template class itebd<fp32>;
// template class itebd<fp64>;
// template class itebd<fp128>;
// template class itebd<cx32>;
template class itebd<cx64>;
// template class itebd<cx128>;

template<typename Scalar>
itebd<Scalar>::itebd(std::shared_ptr<h5pp::File> h5ppFile_) : AlgorithmInfinite<Scalar>(std::move(h5ppFile_), OptRitz::NONE, AlgorithmType::iTEBD) {
    tools::log->trace("Constructing class_itebd");
}

template<typename Scalar> void itebd<Scalar>::run_preprocessing() {
    tools::log->info("Running {} preprocessing", status.algo_type_sv());
    auto t_pre = tid::tic_scope("pre");
    init_bond_dimension_limits();
    init_truncation_error_limits();
    initialize_model(); // First use of random!
    auto t_init    = tid::tic_scope("init");
    status.delta_t = cx128(static_cast<fp128>(settings::itebd::time_step_init_real), static_cast<fp128>(settings::itebd::time_step_init_imag));
    h_evn          = tensors.model->get_2site_ham_AB();
    h_odd          = tensors.model->get_2site_ham_BA();

    unitary_time_evolving_operators =
        qm::time::get_twosite_time_evolution_operators<CplxScalar>(status.delta_t.template to_floating_point<cx128>(), settings::itebd::suzuki_order, h_evn, h_odd);
    tools::log->info("Finished {} preprocessing", status.algo_type_sv());
}

template<typename Scalar> void itebd<Scalar>::run_algorithm() {
    tools::log->info("Starting {} simulation", status.algo_type_sv());
    auto t_run = tid::tic_scope("run");
    while(status.iter < settings::itebd::iter_max and status.algorithm_converged_for == 0) {
        update_state();
        check_convergence();
        print_status();
        write_to_file();

        status.iter++;
        status.wall_time = tid::get_unscoped("t_tot").get_time();
        status.algo_time = t_run->get_time();
    }
}

template<typename Scalar> void itebd<Scalar>::run_postprocessing() {
    auto t_pos = tid::tic_scope("post");
    print_status_full();
}

template<typename Scalar> void itebd<Scalar>::update_state() {
    /*!
     * \fn update_state()
     * \brief infinite Time evolving block decimation.
     */
    auto t_step = tid::tic_scope("step");
    for(auto &U : unitary_time_evolving_operators) {
        Eigen::Tensor<cx64, 3> twosite_tensor = tools::infinite::opt::time_evolve_state(*tensors.state, U);
        tensors.merge_twosite_tensor(twosite_tensor, MergeEvent::GATE, svd::config(status.bond_lim, status.trnc_lim));
        if(&U != &unitary_time_evolving_operators.back()) { tensors.state->swap_AB(); }
    }
    status.phys_time += abs(status.delta_t.template to_floating_point<cx128>());
    tensors.clear_measurements();
}

template<typename Scalar> void itebd<Scalar>::check_convergence() {
    auto t_con = tid::tic_scope("conv");
    check_convergence_entg_entropy();
    check_convergence_variance_ham();
    check_convergence_variance_mom();
    update_bond_dimension_limit();
    check_convergence_time_step();
    if(status.entanglement_converged_for > 0 and status.variance_ham_converged_for > 0 and status.variance_mom_converged_for > 0 and
       status.bond_limit_has_reached_max and status.time_step_has_converged) {
        status.algorithm_converged_for++;
    } else
        status.algorithm_converged_for = 0;
}

template<typename Scalar> void itebd<Scalar>::check_convergence_time_step() {
    auto delta_t = status.delta_t.template to_floating_point<cx128>();
    if(abs(delta_t) <= static_cast<fp128>(settings::itebd::time_step_min)) {
        status.time_step_has_converged = true;
    } else if(status.bond_limit_has_reached_max and status.entanglement_converged_for > 0) {
        // TODO : This step is not compatible with switching between real/imag time evolution... I think?
        status.delta_t                  = std::max<fp128>(static_cast<fp128>(settings::itebd::time_step_min), abs(delta_t) * static_cast<fp128>(0.5));
        unitary_time_evolving_operators = qm::time::get_twosite_time_evolution_operators<CplxScalar>(delta_t, settings::itebd::suzuki_order, h_evn, h_odd);
        //        state->H->update_evolution_step_size(-status.delta_t, settings::itebd::suzuki_order);
        clear_convergence_status();
    }
}
