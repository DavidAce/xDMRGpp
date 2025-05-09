#include "../log.h"
#include "../solver.h"
#include "math/cast.h"
#include <chrono>
#include <complex>
#include <Eigen/Eigenvalues>

using namespace eig;

int eig::solver::wgeev_eigen(cx128 *matrix, size_type L) {
    eig::log->trace("Starting eig wgeev_eigen");
    using Scalar                      = cx128;
    using Real       = typename Eigen::NumTraits<Scalar>::Real;
    using Cplx       = std::complex<Real>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixCplx = Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorCplx = Eigen::Matrix<Cplx, Eigen::Dynamic, 1>;

    auto t_start    = std::chrono::high_resolution_clock::now();
    auto matrix_map = Eigen::Map<MatrixType>(matrix, L, L);
    int  n          = safe_cast<int>(L);
    int  info       = 0;
    auto t_prep     = std::chrono::high_resolution_clock::now();

    auto solver = Eigen::ComplexEigenSolver<MatrixType>(matrix_map, config.compute_eigvecs == Vecs::ON );
    if(solver.info() == Eigen::ComputationInfo::Success) {
        auto &eigvals = result.get_eigvals<Scalar, Form::NSYM>();
        eigvals.resize(safe_cast<size_t>(L));
        auto evals_map = Eigen::Map<VectorCplx>(eigvals.data(), L);
        evals_map      = solver.eigenvalues();

        if(config.compute_eigvecs == Vecs::ON) {
            auto &eigvecs = result.get_eigvecs<Scalar, Form::NSYM>();
            eigvecs.resize(static_cast<size_t>(L * L));
            auto evecs_map = Eigen::Map<MatrixCplx>(eigvecs.data(), L, L);
            evecs_map      = solver.eigenvectors();
        }
    }
    else{
        throw std::runtime_error("Eigen wgeev_eigen failed with error: " + std::to_string(solver.info()));
    }

    auto t_total               = std::chrono::high_resolution_clock::now();
    result.meta.eigvecsR_found = solver.info() == Eigen::ComputationInfo::Success and config.compute_eigvecs == Vecs::ON;
    result.meta.eigvals_found  = true;
    result.meta.rows           = L;
    result.meta.cols           = L;
    result.meta.nev            = n;
    result.meta.nev_converged  = n;
    result.meta.n              = L;
    result.meta.form           = Form::NSYM;
    result.meta.type           = Type::CX128;
    result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
    result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();

    return info;
}
