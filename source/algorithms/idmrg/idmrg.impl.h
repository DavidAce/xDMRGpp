#pragma once
#include "../idmrg.h"
#include "config/settings.h"
#include "tensors/edges/EdgesInfinite.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/state/StateInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/infinite/opt.h"

template<typename Scalar>
idmrg<Scalar>::idmrg(std::shared_ptr<h5pp::File> h5ppFile_) : AlgorithmInfinite<Scalar>(std::move(h5ppFile_), OptRitz::SR, AlgorithmType::iDMRG) {
    tools::log->trace("Constructing class_idmrg");
    tensors.initialize(settings::model::model_type);
}

template<typename Scalar>
void idmrg<Scalar>::run_algorithm() {
    if(status.opt_ritz == OptRitz::SR)
        tensors.state->set_name("state_emin");
    else
        tensors.state->set_name("state_emax");
    tools::log->info("Starting {} simulation of model [{}] for state [{}]", status.algo_type_sv(), enum2sv(settings::model::model_type),
                     tensors.state->get_name());
    auto t_algo = tid::tic_scope(status.algo_type_sv());
    while(true) {
        update_state();
        check_convergence();
        print_status();
        write_to_file();

        // It's important not to perform the last move.
        // That last state would not get optimized
        if(status.iter >= settings::idmrg::iter_max) {
            status.algo_stop = AlgorithmStop::MAX_ITERS;
            break;
        }
        if(status.algorithm_has_succeeded) {
            status.algo_stop = AlgorithmStop::SUCCESS;
            break;
        }
        if(status.algorithm_has_to_stop) {
            status.algo_stop = AlgorithmStop::SATURATED;
            break;
        }
        if(status.iter >= settings::idmrg::iter_max) {
            status.algo_stop = AlgorithmStop::MAX_ITERS;
            break;
        }
        if(status.algorithm_has_succeeded) {
            status.algo_stop = AlgorithmStop::SUCCESS;
            break;
        }
        if(status.algorithm_has_to_stop) {
            status.algo_stop = AlgorithmStop::SATURATED;
            break;
        }

        update_bond_dimension_limit();   // Will update bond dimension if the state precision is being limited by bond dimension
        update_truncation_error_limit(); // Will update truncation error limit if the state is being truncated
        tensors.enlarge();
        status.iter++;
        status.step++;
        status.wall_time = tid::get_unscoped("t_tot").get_time();
        status.algo_time = t_algo->get_time();
    }
    tools::log->info("Finished {} simulation -- reason: {}", status.algo_type_sv(), status.algo_stop_sv());
}

template<typename Scalar>
void idmrg<Scalar>::update_state() {
    /*!
     * \fn void single_DMRG_step()
     */
    tools::log->trace("Starting single iDMRG step with ritz: [{}]", enum2sv(status.opt_ritz));
    Eigen::Tensor<Scalar, 3> twosite_tensor = tools::infinite::opt::find_ground_state(tensors, status.opt_ritz);
    tensors.merge_twosite_tensor(twosite_tensor, MergeEvent::OPT, svd::config(status.bond_lim, status.trnc_lim));
}

template<typename Scalar>
void idmrg<Scalar>::check_convergence() {
    tools::log->trace("Checking convergence");
    auto t_con = tid::tic_scope("conv");
    check_convergence_entanglement();
    check_convergence_variance_mpo();
    check_convergence_variance_ham();
    check_convergence_variance_mom();
    if(status.energy_mpo_saturated_for > 0 and status.variance_mpo_converged_for > 0 and status.variance_ham_converged_for > 0 and
       status.variance_mom_converged_for > 0 and status.bond_limit_has_reached_max) {
        status.algorithm_converged_for++;
    } else
        status.algorithm_converged_for = 0;
}
