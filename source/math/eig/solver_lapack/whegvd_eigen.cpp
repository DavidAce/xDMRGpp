#include "../log.h"
#include "../solver.h"
#include "math/cast.h"
#include <chrono>
#include <complex>
#include <Eigen/Eigenvalues>

using namespace eig;

int eig::solver::whegvd_eigen(cx128 *matrixA, cx128 *matrixB, size_type L) {
    eig::log->trace("Starting eig whegvd_eigen");
    using Scalar = cx128;
    using Real   = typename Eigen::NumTraits<Scalar>::Real;
    // using Cplx       = std::complex<Real>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    // using MatrixReal = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    auto t_start     = std::chrono::high_resolution_clock::now();
    auto matrixA_map = Eigen::Map<MatrixType>(matrixA, L, L);
    auto matrixB_map = Eigen::Map<MatrixType>(matrixB, L, L);
    int  n           = safe_cast<int>(L);
    int  info        = 0;
    int  options     = config.compute_eigvecs == Vecs::ON ? Eigen::ComputeEigenvectors | Eigen::Ax_lBx : Eigen::EigenvaluesOnly | Eigen::Ax_lBx;
    auto t_prep      = std::chrono::high_resolution_clock::now();

    auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>(matrixA_map, matrixB_map, options);
    if(solver.info() == Eigen::ComputationInfo::Success) {
        auto &eigvals = result.get_eigvals<Scalar, Form::SYMM>();
        eigvals.resize(safe_cast<size_t>(L));
        auto evals_map = Eigen::Map<VectorReal>(eigvals.data(), L);
        evals_map      = solver.eigenvalues();

        if(config.compute_eigvecs == Vecs::ON) {
            auto &eigvecs = result.get_eigvecs<Scalar, Form::SYMM>();
            eigvecs.resize(static_cast<size_t>(L * L));
            auto evecs_map = Eigen::Map<MatrixType>(eigvecs.data(), L, L);
            evecs_map      = solver.eigenvectors();
        }
    } else {
        throw std::runtime_error("Eigen whegvd_eigen failed with error: " + std::to_string(solver.info()));
    }

    auto t_total               = std::chrono::high_resolution_clock::now();
    result.meta.eigvecsR_found = solver.info() == Eigen::ComputationInfo::Success and config.compute_eigvecs == Vecs::ON;
    result.meta.eigvals_found  = true;
    result.meta.rows           = L;
    result.meta.cols           = L;
    result.meta.nev            = n;
    result.meta.nev_converged  = n;
    result.meta.n              = L;
    result.meta.form           = Form::SYMM;
    result.meta.type           = Type::CX128;
    result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
    result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();

    return info;
}
