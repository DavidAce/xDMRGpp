#pragma once

#include "AlgorithmFinite.h"
#include "qm/gate.h"
#include <deque>

namespace tools::finite::opt {
    template<typename Scalar> class opt_mps;
}
template<typename Scalar> class StateFinite;

/*!
 * \brief Class that runs the excited-state DMRG algorithm.
 */
template<typename Scalar>
class xdmrg : public AlgorithmFinite<Scalar> {
    public:
    using OptMeta    = typename AlgorithmFinite<Scalar>::OptMeta;
    using RealScalar = typename AlgorithmFinite<Scalar>::RealScalar;
    using CplxScalar = std::complex<RealScalar>;
    using VecType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VecReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VecCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, 1>;
    using MatType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic>;

    // Inherit the constructor of class_algorithm_base
    explicit xdmrg(std::shared_ptr<h5pp::File> h5ppFile_);
    void find_energy_range();
    void init_energy_target(std::optional<double> energy_density_target = std::nullopt);
    void run_task_list(std::deque<xdmrg_task> &task_list);
    void run_preprocessing() final;
    void run_default_task_list() final;
    void run_algorithm() final;
    void update_state() final;
    void resume() final;
    void update_time_step();
    void set_energy_shift_mpo() final;

    using AlgorithmFinite<Scalar>::AlgorithmFinite;
    using AlgorithmFinite<Scalar>::status;
    using AlgorithmFinite<Scalar>::h5file;
    using AlgorithmFinite<Scalar>::tensors;
    using AlgorithmFinite<Scalar>::set_parity_shift_mpo;
    using AlgorithmFinite<Scalar>::set_parity_shift_mpo_squared;
    using AlgorithmFinite<Scalar>::set_energy_shift_mpo;
    using AlgorithmFinite<Scalar>::rebuild_tensors;
    using AlgorithmFinite<Scalar>::update_precision_limit;
    using AlgorithmFinite<Scalar>::update_eigs_tolerance;
    using AlgorithmFinite<Scalar>::update_bond_dimension_limit;
    using AlgorithmFinite<Scalar>::update_truncation_error_limit;
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

    using AlgorithmFinite<Scalar>::try_projection;
    using AlgorithmFinite<Scalar>::move_center_point;
    using AlgorithmFinite<Scalar>::var_delta;
    using AlgorithmFinite<Scalar>::ene_delta;
    using AlgorithmFinite<Scalar>::var_change;
    using AlgorithmFinite<Scalar>::var_latest;
    using AlgorithmFinite<Scalar>::ene_latest;
    using AlgorithmFinite<Scalar>::last_optalgo;
};
