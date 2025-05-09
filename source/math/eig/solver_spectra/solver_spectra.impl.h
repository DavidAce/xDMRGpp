#pragma once
#include "../matvec/matvec_dense.h"
#include "../matvec/matvec_mpo.h"
#include "../matvec/matvec_mpos.h"
#include "../matvec/matvec_sparse.h"
#include "../matvec/matvec_zero.h"
#include "math/cast.h"
#include "math/eig/log.h"
#include "solver_spectra.h"
#include "tid/tid.h"
#include <debug/exceptions.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/HermEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>

#include "GenMatVec.h"


inline Spectra::SortRule get_spectra_sort_rule(eig::Ritz ritz) {
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

inline std::string_view SpectraInfoToString(Spectra::CompInfo info) {
    switch(info) {
        case Spectra::CompInfo::Successful: return "Successful";
        case Spectra::CompInfo::NotComputed: return "NotComputed";
        case Spectra::CompInfo::NotConverging: return "NotConverging";
        case Spectra::CompInfo::NumericalIssue: return "NumericalIssue";
        default: return "Unknown Spectra::CompInfo";
    }
}




template<typename MatrixType>
void eig::solver_spectra<MatrixType>::eigs() {
    auto t_spectra = tid::tic_scope("spectra");
    using Real     = decltype(std::real(std::declval<Scalar>()));
    // using Cplx     = std::complex<Real>;
    result.reset();
    config.checkRitz();
    nev_internal = std::clamp<int>(safe_cast<int>(config.maxNev.value()), 1, matrix.rows() / 2);
    ncv_internal = std::clamp<int>(safe_cast<int>(config.maxNcv.value()), safe_cast<int>(config.maxNev.value()) + 1, matrix.rows());

    auto eps     = std::numeric_limits<Real>::epsilon();
    auto tol_def = eps * Real{1e4};
    auto tol_min = eps * Real{1e1};
    auto tol_cfg = config.tol.value_or(static_cast<double>(tol_def));
    auto tol     = std::max<Real>(static_cast<Real>(tol_cfg), tol_min); /*!< 1e-12 is good, see link above. */
    auto maxiter = config.maxIter.value_or(1000);


    auto sort    = get_spectra_sort_rule(config.ritz.value());
    matrix.set_mode(config.form.value());
    matrix.set_side(config.side.value());

    auto run = [&](auto &sol, auto &eigvals, auto &eigvecs) {
        sol.init(residual);
        sol.compute(sort, maxiter, tol);
        if(sol.info() != Spectra::CompInfo::Successful) {
            eig::log->debug("{}: {}", sfinae::type_name<std::remove_cvref_t<decltype(sol)>>(), SpectraInfoToString(sol.info()));
            sol.compute(sort, maxiter, Real{1e20f}); // Should succeed due to high tol
        }
        if(sol.info() != Spectra::CompInfo::Successful) throw except::runtime_error("Spectra failed to converge");
        auto eigvecs_spectra = sol.eigenvectors(nev_internal);
        auto eigvals_spectra = sol.eigenvalues();
        eigvecs.resize(eigvecs_spectra.size());
        eigvals.resize(eigvals_spectra.size());
        std::copy(eigvecs_spectra.data(), eigvecs_spectra.data() + eigvecs_spectra.size(), eigvecs.data());
        std::copy(eigvals_spectra.data(), eigvals_spectra.data() + eigvals_spectra.size(), eigvals.data());

        result.meta.eigvecsR_found = true; // We can use partial results
        result.meta.eigvals_found  = true; // We can use partial results
        result.meta.rows           = matrix.rows();
        result.meta.cols           = eigvecs_spectra.cols();
        result.meta.nev            = nev_internal;
        result.meta.nev_converged  = eigvals.size(); // Holds the number of converged eigenpairs on exit
        result.meta.ncv            = ncv_internal;
        result.meta.tol            = static_cast<double>(tol);
        result.meta.iter           = sol.num_iterations() + sol.num_operations();
        result.meta.num_mv         = matrix.num_mv;
        result.meta.num_pc         = matrix.num_pc;
        result.meta.num_op         = matrix.num_op;
        result.meta.time_mv        = matrix.t_multAx->get_time();
        result.meta.time_pc        = matrix.t_multPc->get_time();
        result.meta.time_op        = matrix.t_multOPv->get_time();
        result.meta.n              = matrix.rows();
        result.meta.tag            = config.tag;
        result.meta.ritz           = eig::RitzToShortString(config.ritz.value());
        result.meta.form           = config.form.value();
        result.meta.type           = config.type.value();
        result.meta.time_total     = t_spectra->get_time();
    };

    constexpr Type type = eig::ScalarToType<Scalar>();
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        if(matrix.get_form() == Form::SYMM) {
            Spectra::HermEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            return run(sol, result.get_eigvals<Form::SYMM, type>(), result.get_eigvecs<Form::SYMM, type>());
        }
        if(matrix.get_form() == Form::NSYM) { throw except::runtime_error("Spectra does not support NSYM on complex matrices"); }

    } else {
        if(matrix.get_form() == Form::SYMM) {
            if constexpr(std::is_same_v<MatrixType, MatVecMPOS<Scalar>>) {
                if (config.primme_massMatrixMatvec.has_value()) {
                    auto matrixB = GenMatVec<MatrixType>(matrix);
                    matrixB.set_maxiters(200000);
                    matrixB.set_tolerance(1e-10f);
                    Spectra::SymGEigsSolver<MatrixType, GenMatVec<MatrixType>, Spectra::GEigsMode::RegularInverse> sol(matrix,matrixB, nev_internal, ncv_internal);
                    return run(sol, result.get_eigvals<Form::SYMM, type>(), result.get_eigvecs<Form::SYMM, type>());
                }
            }
            Spectra::SymEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            return run(sol, result.get_eigvals<Form::SYMM, type>(), result.get_eigvecs<Form::SYMM, type>());
        }
        if(matrix.get_form() == Form::NSYM) {
            Spectra::GenEigsSolver<MatrixType> sol(matrix, nev_internal, ncv_internal);
            return run(sol, result.get_eigvals<Form::NSYM, type>(), result.get_eigvecs<Form::NSYM, type>());
        }
    }
}