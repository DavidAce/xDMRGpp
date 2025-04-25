#pragma once
#include "AlgorithmFinite.h"
#include "qm/gate.h"
#include "qm/lbit.h"
#include <deque>

/*!
// * \brief Class that runs the finite LBIT algorithm.
 */

template<typename Scalar> class StateFinite;
template<typename Scalar>
class flbit : public AlgorithmFinite<Scalar> {
    using RealScalar = typename AlgorithmFinite<Scalar>::RealScalar;
    using OptMeta    = typename AlgorithmFinite<Scalar>::OptMeta;
    using VecType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VecReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VecCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, 1>;
    using MatType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic>;

    using AlgorithmFinite<Scalar>::AlgorithmFinite;
    using AlgorithmFinite<Scalar>::status;
    using AlgorithmFinite<Scalar>::h5file;
    using AlgorithmFinite<Scalar>::tensors;
    using AlgorithmFinite<Scalar>::initialize_model;
    using AlgorithmFinite<Scalar>::initialize_state;
    using AlgorithmFinite<Scalar>::init_bond_dimension_limits;
    using AlgorithmFinite<Scalar>::init_truncation_error_limits;
    using AlgorithmFinite<Scalar>::clear_convergence_status;
    using AlgorithmFinite<Scalar>::run_postprocessing;
    using AlgorithmFinite<Scalar>::print_status_full;
    using AlgorithmFinite<Scalar>::print_status;

    public:
    std::unique_ptr<StateFinite<Scalar>>             state_lbit, state_lbit_init, state_real_init;
    std::vector<qm::Gate>                            ham_gates_1body, time_gates_1body;
    std::vector<qm::Gate>                            ham_gates_2body, time_gates_2body;
    std::vector<qm::Gate>                            ham_gates_3body, time_gates_3body;
    std::vector<qm::Gate>                            ham_gates_Lbody, time_gates_Lbody;
    std::vector<qm::SwapGate>                        ham_swap_gates_1body, time_swap_gates_1body;
    std::vector<qm::SwapGate>                        ham_swap_gates_2body, time_swap_gates_2body;
    std::vector<qm::SwapGate>                        ham_swap_gates_3body, time_swap_gates_3body;
    std::vector<qm::SwapGate>                        ham_swap_gates_Lbody, time_swap_gates_Lbody;
    std::vector<std::vector<qm::Gate>>               unitary_gates_2site_layers;
    std::vector<Eigen::Tensor<cx64, 4>>              unitary_gates_mpo_layer_full;
    std::vector<std::vector<Eigen::Tensor<cx64, 4>>> unitary_gates_mpo_layers;
    qm::lbit::UnitaryGateProperties                  uprop;
    Eigen::Tensor<std::complex<double>, 1>           ledge, redge;
    std::vector<cx128>                               time_points;
    Eigen::Tensor<cx64, 2>                           lbit_overlap; // The real-space support of the l-bits

    // Inherit the constructor of class_algorithm_base
    explicit flbit(std::shared_ptr<h5pp::File> h5file_);
    void update_time_step();
    void resume() final;
    void run_task_list(std::deque<flbit_task> &task_list);
    void run_default_task_list() final;
    void run_preprocessing() final;
    void run_algorithm() final;
    void run_algorithm_parallel();
    void run_algorithm2();
    void update_state() final;
    void check_convergence() final;
    void create_time_points();
    void create_hamiltonian_gates();
    //    void create_hamiltonian_swap_gates();
    void update_time_evolution_gates();
    void create_unitary_circuit_gates();
    void time_evolve_lbit_state();
    void transform_to_real_basis();
    void transform_to_lbit_basis();
    void write_to_file(StorageEvent storage_event = StorageEvent::ITERATION, CopyPolicy copy_policy = CopyPolicy::TRY) final;
    void print_status() final;
    void print_status(const AlgorithmStatus &status, const TensorsFinite<Scalar> &tensors);
};
