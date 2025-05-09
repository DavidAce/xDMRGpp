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
#include "../enums.h"
#include "../log.h"
#include "../solver.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include <chrono>

using namespace eig;

int eig::solver::zheevr(cx64 *matrix /*!< gets destroyed */, size_type L, char range, int il, int iu, fp64 vl, fp64 vu) {
    eig::log->trace("Starting eig zheevr | range {} | i [{},{}] | v [{},{}]", range, il, iu, vl, vu);
    auto t_start = std::chrono::high_resolution_clock::now();

    //    auto A    = std::vector<cx64>(matrix, matrix + L * L);
    char jobz  = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';
    int  info  = 0;
    int  n     = safe_cast<int>(L);
    int  lda   = std::max(1, n);
    int  ldz   = std::max(1, n);
    int  m_req = n; // For range == 'V' we don't know how many eigenvalues will be found in (vl,vu]
    if(range == 'I') m_req = std::max(iu, il) - std::min(iu, il) + 1;
    m_req = std::clamp(m_req, 1, std::min(m_req, n));

    int              m_found = 0;
    cx64             lwork_query[1];
    fp64             rwork_query[1];
    int              iwork_query[1];
    std::vector<int> isuppz(static_cast<size_t>(2 * m_req));
    std::vector<int> ifail(static_cast<unsigned long>(L));

    auto &eigvals = result.get_eigvals<Form::SYMM, Type::CX64>();
    auto &eigvecs = result.get_eigvecs<Form::SYMM, Type::CX64>();
    eigvals.resize(static_cast<size_t>(ldz));
    if(config.compute_eigvecs == Vecs::ON)
        eigvecs.resize(safe_cast<size_t>(ldz * m_req)); // Docs claim ldz * m, but it segfaults when 'V' finds more than m eigvals

    info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, jobz, range, 'U', lda, matrix, lda, //
                               vl, vu, il, iu, 2 * LAPACKE_dlamch('S'),              //
                               &m_found, eigvals.data(), eigvecs.data(),             //
                               ldz, isuppz.data(), lwork_query, -1, rwork_query, -1, iwork_query, -1);
    if(info < 0) throw except::runtime_error("LAPACKE_dsyevr_work: info {}", info);

    int lwork  = safe_cast<int>(std::abs(lwork_query[0].real()));
    int lrwork = safe_cast<int>(rwork_query[0]);
    int liwork = safe_cast<int>(iwork_query[0]);

    eig::log->trace(" lwork  = {}", lwork);
    eig::log->trace(" lrwork = {}", lrwork);
    eig::log->trace(" liwork = {}", liwork);
    std::vector<cx64>   work(static_cast<size_t>(lwork));
    std::vector<double> rwork(static_cast<size_t>(lrwork));
    std::vector<int>    iwork(static_cast<size_t>(liwork));
    auto                t_prep = std::chrono::high_resolution_clock::now();
    info                       = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, jobz, range, 'U', lda, matrix, lda, //
                                                     vl, vu, il, iu, 2 * LAPACKE_dlamch('S'),              //
                                                     &m_found, eigvals.data(), eigvecs.data(),             //
                                                     ldz, isuppz.data(), lwork_query, lwork, rwork_query, lrwork, iwork_query, liwork);

    auto t_total = std::chrono::high_resolution_clock::now();
    if(info != 0) throw std::runtime_error("LAPACK zheevr failed with error: " + std::to_string(info));
    eig::log->trace("Found {} eigenvalues", m_found);
    eigvals.resize(static_cast<size_t>(m_found));
    eigvecs.resize(static_cast<size_t>(m_found) * static_cast<size_t>(ldz));
    result.meta.eigvecsR_found = true;
    result.meta.eigvals_found  = true;
    result.meta.rows           = L;
    result.meta.cols           = m_found;
    result.meta.nev            = m_found;
    result.meta.nev_converged  = m_found;
    result.meta.n              = L;
    result.meta.form           = Form::SYMM;
    result.meta.type           = Type::CX64;
    result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
    result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();
    return info;
}
