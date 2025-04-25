#include <complex>

#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

#if defined(MKL_AVAILABLE)
    #include <mkl_lapacke.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/lapacke.h>
#else
    #include <lapacke.h>
#endif
#include "../log.h"
#include "../solver.h"
#include "math/cast.h"
#include <chrono>

using namespace eig;

int eig::solver::dsyevd(fp64 *matrix, size_type L) {
    eig::log->trace("Starting eig dsyevd");
    auto t_start = std::chrono::high_resolution_clock::now();

    auto &eigvals = result.get_eigvals<Form::SYMM, Type::FP64>();
    eigvals.resize(safe_cast<size_t>(L));

    // Call lapack solver
    int    info = 0;
    char   jobz = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';
    char   uplo = 'U';
    double lwork_query[1];
    int    liwork_query[1];
    int    n = safe_cast<int>(L);

    info = LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, jobz, uplo, n, matrix, n, eigvals.data(), lwork_query, -1, liwork_query, -1);
    if(info < 0) throw std::runtime_error("LAPACKE_dsyevd_work query: info" + std::to_string(info));

    int lwork  = safe_cast<int>(lwork_query[0]);
    int liwork = safe_cast<int>(liwork_query[0]);
    eig::log->trace(" lwork  = {}", lwork);
    eig::log->trace(" liwork = {}", liwork);

    std::vector<double> work(safe_cast<size_t>(lwork));
    std::vector<int>    iwork(safe_cast<size_t>(liwork));
    auto                t_prep = std::chrono::high_resolution_clock::now();
    info                       = LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, jobz, uplo, n, matrix, n, eigvals.data(), work.data(), lwork, iwork.data(), liwork);
    if(info < 0) throw std::runtime_error("LAPACKE_dsyevd_work: info" + std::to_string(info));

    auto t_total = std::chrono::high_resolution_clock::now();
    if(info == 0) {
        if(config.compute_eigvecs == Vecs::ON) {
            auto &eigvecs = result.get_eigvecs<Form::SYMM, Type::FP64>();
            eigvecs.resize(static_cast<size_t>(L * L));
            std::copy(matrix, matrix + L * L, eigvecs.begin());
        }
        result.meta.eigvecsR_found = config.compute_eigvecs == Vecs::ON;
        result.meta.eigvals_found  = true;
        result.meta.rows           = L;
        result.meta.cols           = L;
        result.meta.nev            = n;
        result.meta.nev_converged  = n;
        result.meta.n              = L;
        result.meta.form           = Form::SYMM;
        result.meta.type           = Type::FP64;
        result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
        result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();
    } else {
        throw std::runtime_error("LAPACK dsyevd failed with error: " + std::to_string(info));
    }
    return info;
}
