#include <complex>

#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

// complex must be included before lapacke!
#if defined(MKL_AVAILABLE)
    #include <mkl_lapacke.h>
#elif defined(OPENBLAS_AVAILABLE)
    #include <openblas/lapacke.h>
#else
    #include <lapacke.h>
#endif

#include "../log.h"
#include "../solver.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include <chrono>

int eig::solver::cheev(cx32 *matrix, size_type L) {
    eig::log->trace("Starting eig zheev. Eigvecs: {}", config.compute_eigvecs.value() == eig::Vecs::ON);
    auto  t_start = std::chrono::high_resolution_clock::now();
    auto &eigvals = result.get_eigvals<Form::SYMM, Type::CX32>();
    auto &eigvecs = result.get_eigvecs<Form::SYMM, Type::CX32>();
    eigvals.resize(safe_cast<size_t>(L));

    int  n             = safe_cast<int>(L);
    char jobz          = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';
    auto t_prep        = std::chrono::high_resolution_clock::now();
    cx32 work_query[1] = {};
    int  lrwork        = std::max(1, 3 * n - 2);
    auto rwork         = std::vector<fp32>(static_cast<size_t>(lrwork));

    int info =
        LAPACKE_cheev_work(LAPACK_COL_MAJOR, jobz, 'U', n, reinterpret_cast<lapack_complex_float *>(matrix), n, eigvals.data(), work_query, -1, rwork.data());
    if(info != 0) throw std::runtime_error("LAPACK zheev query failed with error: " + std::to_string(info));

    int  lwork = safe_cast<int>(std::abs(work_query[0]));
    auto work  = std::vector<cx32>(safe_cast<size_t>(lwork));

    info = LAPACKE_cheev_work(LAPACK_COL_MAJOR, jobz, 'U', n, reinterpret_cast<lapack_complex_float *>(matrix), n, eigvals.data(),
                              reinterpret_cast<lapack_complex_float *>(work.data()), lwork, rwork.data());
    if(info != 0) throw std::runtime_error("LAPACK zheev failed with error: " + std::to_string(info));
    if(config.compute_eigvecs == Vecs::ON) {
        eigvecs.resize(static_cast<size_t>(L * L));
        std::copy(matrix, matrix + L * L, eigvecs.begin());
    }

    auto t_total = std::chrono::high_resolution_clock::now();

    result.meta.eigvecsR_found = config.compute_eigvecs == Vecs::ON;
    result.meta.eigvals_found  = true;
    result.meta.rows           = L;
    result.meta.cols           = L;
    result.meta.nev            = n;
    result.meta.nev_converged  = n;
    result.meta.n              = L;
    result.meta.form           = Form::SYMM;
    result.meta.type           = Type::CX32;
    result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
    result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();

    return info;
}

int eig::solver::cheevd(cx32 *matrix, size_type L) {
    eig::log->trace("Starting eig cheevd. Eigvecs: {}", config.compute_eigvecs.value() == eig::Vecs::ON);
    auto  t_start = std::chrono::high_resolution_clock::now();
    auto &eigvals = result.get_eigvals<Form::SYMM, Type::CX32>();
    auto &eigvecs = result.get_eigvecs<Form::SYMM, Type::CX32>();
    eigvals.resize(static_cast<size_t>(L));

    int  n              = safe_cast<int>(L);
    char jobz           = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';
    auto t_prep         = std::chrono::high_resolution_clock::now();
    cx32 work_query[1]  = {};
    fp32 rwork_query[1] = {};
    int  iwork_query[1] = {};
    int  info = LAPACKE_cheevd_work(LAPACK_COL_MAJOR, jobz, 'U', n, reinterpret_cast<lapack_complex_float *>(matrix), n, eigvals.data(), work_query, -1,
                                    rwork_query, -1, iwork_query, -1);
    if(info != 0) throw std::runtime_error("LAPACK cheevd query failed with error: " + std::to_string(info));

    // These nice values are inspired from armadillo. The prefactors give good performance.
    int  lwork  = safe_cast<int>(std::abs(work_query[0])); // 2 * (2 * n + n * n);
    int  lrwork = safe_cast<int>(rwork_query[0]);          // 2 * (1 + 5 * n + 2 * (n * n));
    int  liwork = safe_cast<int>(iwork_query[0]);          // 3 * (3 + 5 * n);
    auto work   = std::vector<cx32>(safe_cast<size_t>(lwork));
    auto rwork  = std::vector<fp32>(safe_cast<size_t>(lrwork));
    auto iwork  = std::vector<int>(safe_cast<size_t>(liwork));

    info = LAPACKE_cheevd_work(LAPACK_COL_MAJOR, jobz, 'U', n, reinterpret_cast<lapack_complex_float *>(matrix), n, eigvals.data(),
                               reinterpret_cast<lapack_complex_float *>(work.data()), lwork, rwork.data(), lrwork, iwork.data(), liwork);
    if(info != 0) throw std::runtime_error("LAPACK cheevd failed with error: " + std::to_string(info));
    if(config.compute_eigvecs == Vecs::ON) {
        eigvecs.resize(safe_cast<size_t>(L * L));
        std::copy(matrix, matrix + L * L, eigvecs.begin());
    }

    auto t_total = std::chrono::high_resolution_clock::now();

    result.meta.eigvecsR_found = config.compute_eigvecs == Vecs::ON;
    result.meta.eigvals_found  = true;
    result.meta.rows           = L;
    result.meta.cols           = L;
    result.meta.nev            = n;
    result.meta.nev_converged  = n;
    result.meta.n              = L;
    result.meta.form           = Form::SYMM;
    result.meta.type           = Type::CX32;
    result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
    result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();

    return info;
}
