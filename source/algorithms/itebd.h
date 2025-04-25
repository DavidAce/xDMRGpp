
#pragma once
#include "AlgorithmInfinite.h"
#include <unsupported/Eigen/CXX11/Tensor>

/*!
 * \brief Class that runs the imaginary TEBD algorithm.
 */
template<typename Scalar>
class itebd : public AlgorithmInfinite<Scalar> {
    using RealScalar = typename AlgorithmInfinite<Scalar>::RealScalar;
    using CplxScalar = std::complex<RealScalar>;
    using VecType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VecReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VecCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, 1>;
    using MatType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatReal    = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatCplx    = Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic>;

    using AlgorithmInfinite<Scalar>::AlgorithmInfinite;
    using AlgorithmInfinite<Scalar>::status;
    using AlgorithmInfinite<Scalar>::h5file;
    using AlgorithmInfinite<Scalar>::tensors;
    using AlgorithmInfinite<Scalar>::initialize_model;
    using AlgorithmInfinite<Scalar>::initialize_state;
    using AlgorithmInfinite<Scalar>::init_bond_dimension_limits;
    using AlgorithmInfinite<Scalar>::init_truncation_error_limits;
    using AlgorithmInfinite<Scalar>::write_to_file;
    using AlgorithmInfinite<Scalar>::print_status;
    using AlgorithmInfinite<Scalar>::print_status_full;
    using AlgorithmInfinite<Scalar>::update_state;
    using AlgorithmInfinite<Scalar>::update_truncation_error_limit;
    using AlgorithmInfinite<Scalar>::update_bond_dimension_limit;
    using AlgorithmInfinite<Scalar>::check_convergence_entg_entropy;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_mpo;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_ham;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_mom;
    using AlgorithmInfinite<Scalar>::clear_convergence_status;

    public:
    explicit itebd(std::shared_ptr<h5pp::File> h5ppFile_);

    std::vector<Eigen::Tensor<CplxScalar, 2>> unitary_time_evolving_operators;
    Eigen::Tensor<Scalar, 2>                  h_evn, h_odd;

    void run_algorithm() final;
    void run_preprocessing() final;
    void run_postprocessing() final;
    void update_state() final;
    void check_convergence_time_step();
    void check_convergence() final;
};
