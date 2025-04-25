#pragma once
#include "AlgorithmFinite.h"
#include <deque>
class class_h5table_measurements_finite;
namespace tools::finite::opt {
    template<typename Scalar> class opt_mps;
    struct OptMeta;
}

template<typename Scalar> class StateFinite;

/*!
// * \brief Class that runs the finite DMRG algorithm.
 */

template<typename Scalar>
class fdmrg : public AlgorithmFinite<Scalar> {
    public:
    using RealScalar = typename AlgorithmFinite<Scalar>::RealScalar;
    using OptMeta    = typename AlgorithmFinite<Scalar>::OptMeta;

    std::optional<RealScalar> variance_before_step = std::nullopt;
    std::string_view          get_state_name() const;

    // Inherit the constructor of class_algorithm_base
    fdmrg();
    explicit fdmrg(std::shared_ptr<h5pp::File> h5file_);
    void resume() final;
    void run_task_list(std::deque<fdmrg_task> &task_list);
    void run_default_task_list() final;
    void run_preprocessing() final;
    void run_algorithm() final;
    void update_state() final;

    using AlgorithmFinite<Scalar>::AlgorithmFinite;
    using AlgorithmFinite<Scalar>::status;
    using AlgorithmFinite<Scalar>::h5file;
    using AlgorithmFinite<Scalar>::tensors;
    using AlgorithmFinite<Scalar>::set_parity_shift_mpo;
    using AlgorithmFinite<Scalar>::set_parity_shift_mpo_squared;
    using AlgorithmFinite<Scalar>::set_energy_shift_mpo;
    using AlgorithmFinite<Scalar>::rebuild_tensors;
    using AlgorithmFinite<Scalar>::update_precision_limit;
    using AlgorithmFinite<Scalar>::update_dmrg_blocksize;
    using AlgorithmFinite<Scalar>::initialize_model;
    using AlgorithmFinite<Scalar>::initialize_state;
    using AlgorithmFinite<Scalar>::init_bond_dimension_limits;
    using AlgorithmFinite<Scalar>::init_truncation_error_limits;
    using AlgorithmFinite<Scalar>::get_opt_meta;
    using AlgorithmFinite<Scalar>::last_optsolver;
    using AlgorithmFinite<Scalar>::expand_bonds;
    using AlgorithmFinite<Scalar>::write_to_file;
    using AlgorithmFinite<Scalar>::check_convergence;
    using AlgorithmFinite<Scalar>::clear_convergence_status;
    using AlgorithmFinite<Scalar>::run_rbds_analysis;
    using AlgorithmFinite<Scalar>::run_rtes_analysis;
    using AlgorithmFinite<Scalar>::run_postprocessing;
    using AlgorithmFinite<Scalar>::print_status_full;
    using AlgorithmFinite<Scalar>::print_status;
    using AlgorithmFinite<Scalar>::update_bond_dimension_limit;
    using AlgorithmFinite<Scalar>::update_truncation_error_limit;
    using AlgorithmFinite<Scalar>::try_projection;
    using AlgorithmFinite<Scalar>::move_center_point;
    using AlgorithmFinite<Scalar>::var_delta;
    using AlgorithmFinite<Scalar>::ene_delta;
    using AlgorithmFinite<Scalar>::var_change;
    using AlgorithmFinite<Scalar>::var_latest;
    using AlgorithmFinite<Scalar>::ene_latest;
    using AlgorithmFinite<Scalar>::last_optalgo;
};
