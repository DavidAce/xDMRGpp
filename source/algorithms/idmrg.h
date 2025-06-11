#pragma once

#include "AlgorithmInfinite.h"

/*!
 * \brief Class that runs the infinite DMRG algorithm.
 */
template<typename Scalar>
class idmrg : public AlgorithmInfinite<Scalar> {
    using RealScalar = typename AlgorithmInfinite<Scalar>::RealScalar;

    using AlgorithmInfinite<Scalar>::AlgorithmInfinite;
    using AlgorithmInfinite<Scalar>::status;
    using AlgorithmInfinite<Scalar>::h5file;
    using AlgorithmInfinite<Scalar>::tensors;
    using AlgorithmInfinite<Scalar>::write_to_file;
    using AlgorithmInfinite<Scalar>::print_status;
    using AlgorithmInfinite<Scalar>::print_status_full;
    using AlgorithmInfinite<Scalar>::update_state;
    using AlgorithmInfinite<Scalar>::update_truncation_error_limit;
    using AlgorithmInfinite<Scalar>::update_bond_dimension_limit;
    using AlgorithmInfinite<Scalar>::check_convergence_entanglement;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_mpo;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_ham;
    using AlgorithmInfinite<Scalar>::check_convergence_variance_mom;

    public:
    // Inherit the constructor of class_algorithm_base
    explicit idmrg(std::shared_ptr<h5pp::File> h5ppFile_);
    void run_algorithm() final;
    void update_state() final;
    void check_convergence() final;
};
