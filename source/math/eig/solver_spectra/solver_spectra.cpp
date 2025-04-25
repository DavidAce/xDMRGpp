#include "solver_spectra.h"
#include "../matvec/matvec_dense.h"
#include "../matvec/matvec_mpo.h"
#include "../matvec/matvec_mpos.h"
#include "../matvec/matvec_sparse.h"
#include "../matvec/matvec_zero.h"
#include "math/cast.h"
#include "tid/tid.h"
#include <debug/exceptions.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/HermEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>

template class eig::solver_spectra<MatVecMPOS<fp32>>;
template class eig::solver_spectra<MatVecMPOS<fp64>>;
template class eig::solver_spectra<MatVecMPOS<fp128>>;
template class eig::solver_spectra<MatVecMPOS<cx32>>;
template class eig::solver_spectra<MatVecMPOS<cx64>>;
template class eig::solver_spectra<MatVecMPOS<cx128>>;
//
template class eig::solver_spectra<MatVecMPO<fp32>>;
template class eig::solver_spectra<MatVecMPO<fp64>>;
template class eig::solver_spectra<MatVecMPO<fp128>>;
template class eig::solver_spectra<MatVecMPO<cx32>>;
template class eig::solver_spectra<MatVecMPO<cx64>>;
template class eig::solver_spectra<MatVecMPO<cx128>>;
//
template class eig::solver_spectra<MatVecDense<fp32>>;
template class eig::solver_spectra<MatVecDense<fp64>>;
template class eig::solver_spectra<MatVecDense<fp128>>;
template class eig::solver_spectra<MatVecDense<cx32>>;
template class eig::solver_spectra<MatVecDense<cx64>>;
template class eig::solver_spectra<MatVecDense<cx128>>;
//
template class eig::solver_spectra<MatVecSparse<fp32>>;
template class eig::solver_spectra<MatVecSparse<fp64>>;
template class eig::solver_spectra<MatVecSparse<fp128>>;
template class eig::solver_spectra<MatVecSparse<cx32>>;
template class eig::solver_spectra<MatVecSparse<cx64>>;
template class eig::solver_spectra<MatVecSparse<cx128>>;
//
template class eig::solver_spectra<MatVecZero<fp32>>;
template class eig::solver_spectra<MatVecZero<fp64>>;
template class eig::solver_spectra<MatVecZero<fp128>>;
template class eig::solver_spectra<MatVecZero<cx32>>;
template class eig::solver_spectra<MatVecZero<cx64>>;
template class eig::solver_spectra<MatVecZero<cx128>>;
//

Spectra::SortRule get_spectra_sort_rule(eig::Ritz ritz) {
    switch(ritz) {
        case eig::Ritz::LA: return Spectra::SortRule::LargestAlge;
        case eig::Ritz::SA: return Spectra::SortRule::SmallestAlge;
        case eig::Ritz::LM: return Spectra::SortRule::LargestMagn;
        case eig::Ritz::SM: return Spectra::SortRule::SmallestMagn;
        case eig::Ritz::LR: return Spectra::SortRule::LargestReal;
        case eig::Ritz::SR: return Spectra::SortRule::SmallestReal;
        case eig::Ritz::LI: return Spectra::SortRule::LargestImag;
        case eig::Ritz::SI: return Spectra::SortRule::SmallestImag;
        case eig::Ritz::BE: return Spectra::SortRule::BothEnds;
        case eig::Ritz::primme_smallest: return Spectra::SortRule::SmallestMagn;
        case eig::Ritz::primme_largest: return Spectra::SortRule::LargestMagn;
        case eig::Ritz::primme_closest_geq: return Spectra::SortRule::SmallestMagn;
        case eig::Ritz::primme_closest_leq: return Spectra::SortRule::SmallestMagn;
        case eig::Ritz::primme_closest_abs: return Spectra::SortRule::SmallestMagn;
        case eig::Ritz::primme_largest_abs: return Spectra::SortRule::LargestMagn;
        default: throw except::runtime_error("Unknown ritz value");
    }
}

template<typename MatrixType>
eig::solver_spectra<MatrixType>::solver_spectra(MatrixType &matrix_, eig::settings &config_, eig::solution &result_)
    : matrix(matrix_), config(config_), result(result_) {
    if(not config.initial_guess.empty()) residual = static_cast<Scalar *>(config.initial_guess[0].ptr); // Can only take one (the first) residual_norm pointer
    nev_internal = std::clamp<int>(safe_cast<int>(config.maxNev.value()), 1, matrix.rows() / 2);
    ncv_internal = std::clamp<int>(safe_cast<int>(config.maxNcv.value()), safe_cast<int>(config.maxNev.value()) + 1, matrix.rows());
}

template<typename MatrixType>
void eig::solver_spectra<MatrixType>::eigs() {
    auto t_arp = tid::tic_scope("spectra");
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Cplx = std::complex<Real>;
    result.reset();
    nev_internal = std::clamp<int>(safe_cast<int>(config.maxNev.value()), 1, matrix.rows() / 2);
    ncv_internal = std::clamp<int>(safe_cast<int>(config.maxNcv.value()), safe_cast<int>(config.maxNev.value()) + 1, matrix.rows());

    config.checkRitz();
    matrix.set_mode(config.form.value());
    matrix.set_side(config.side.value());

    constexpr Type type = eig::ScalarToType<Scalar>();
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        if(matrix.get_form() == Form::SYMM) {
            Spectra::HermEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            sol.init(residual);
            sol.compute(get_spectra_sort_rule(config.ritz.value()), config.maxIter.value_or(1000), static_cast<Real>(config.tol.value_or(1e-12)));
            if(sol.info() == Spectra::CompInfo::Successful) {
                auto                                               &eigvecs         = result.get_eigvecs<Form::SYMM, type>();
                auto                                               &eigvals         = result.get_eigvals<Form::SYMM, type>();
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> eigvecs_spectra = sol.eigenvectors();
                Eigen::Matrix<Real, Eigen::Dynamic, 1>              eigvals_spectra = sol.eigenvalues();
                eigvecs.resize(eigvecs_spectra.size());
                eigvals.resize(eigvals_spectra.size());
                std::copy(eigvecs_spectra.data(), eigvecs_spectra.data() + eigvecs_spectra.size(), eigvecs.data());
                std::copy(eigvals_spectra.data(), eigvals_spectra.data() + eigvals_spectra.size(), eigvals.data());
            }
        }
        if(matrix.get_form() == Form::NSYM) { throw except::runtime_error("Spectra does not support NSYM on complex matrices"); }

    } else {
        if(matrix.get_form() == Form::SYMM) {
            Spectra::SymEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            sol.init(residual);
            sol.compute(get_spectra_sort_rule(config.ritz.value()), config.maxIter.value_or(1000), static_cast<Real>(config.tol.value_or(1e-12)));
            if(sol.info() == Spectra::CompInfo::Successful) {
                auto                                               &eigvecs         = result.get_eigvecs<Form::SYMM, type>();
                auto                                               &eigvals         = result.get_eigvals<Form::SYMM, type>();
                Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> eigvecs_spectra = sol.eigenvectors();
                Eigen::Matrix<Real, Eigen::Dynamic, 1>              eigvals_spectra = sol.eigenvalues();
                eigvecs.resize(eigvecs_spectra.size());
                eigvals.resize(eigvals_spectra.size());
                std::copy(eigvecs_spectra.data(), eigvecs_spectra.data() + eigvecs_spectra.size(), eigvecs.data());
                std::copy(eigvals_spectra.data(), eigvals_spectra.data() + eigvals_spectra.size(), eigvals.data());
            }
        }
        if(matrix.get_form() == Form::NSYM) {
            Spectra::GenEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            sol.init(residual);
            sol.compute(get_spectra_sort_rule(config.ritz.value()), config.maxIter.value_or(1000), static_cast<Real>(config.tol.value_or(1e-12)));
            if(sol.info() == Spectra::CompInfo::Successful) {
                auto                                               &eigvecs         = result.get_eigvecs<Form::NSYM, type>();
                auto                                               &eigvals         = result.get_eigvals<Form::NSYM, type>();
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> eigvecs_spectra = sol.eigenvectors();
                Eigen::Matrix<Cplx, Eigen::Dynamic, 1>              eigvals_spectra = sol.eigenvalues();
                eigvecs.resize(eigvecs_spectra.size());
                eigvals.resize(eigvals_spectra.size());
                std::copy(eigvecs_spectra.data(), eigvecs_spectra.data() + eigvecs_spectra.size(), eigvecs.data());
                std::copy(eigvals_spectra.data(), eigvals_spectra.data() + eigvals_spectra.size(), eigvals.data());
            }
        }
    }
}