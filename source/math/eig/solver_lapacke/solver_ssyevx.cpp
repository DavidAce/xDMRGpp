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

int eig::solver::ssyevx(fp32 *matrix, size_type L, char range, int il, int iu, fp32 vl, fp32 vu) {
    eig::log->trace("Starting eig ssyevx | range {} | i [{},{}] | v [{},{}]", range, il, iu, vl, vu);
    auto t_start = std::chrono::high_resolution_clock::now();
    char jobz    = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';
    int  info    = 0;
    int  n       = safe_cast<int>(L);
    int  lda     = std::max(1, n);
    int  ldz     = std::max(1, n);
    int  liwork  = std::max(1, 5 * n);
    auto lifail  = std::max(1ul, static_cast<size_t>(n));
    int  m_req   = n; // For range == 'V' we don't know how many eigenvalues will be found in (vl,vu]
    if(range == 'I') m_req = std::max(iu, il) - std::min(iu, il) + 1;
    m_req = std::clamp(m_req, 1, std::min(m_req, n));

    int              m_found = 0;
    fp32             lwork_query[1];
    std::vector<int> iwork(static_cast<size_t>(liwork));
    std::vector<int> ifail(lifail);

    auto &eigvals = result.get_eigvals<Form::SYMM, Type::FP32>();
    auto &eigvecs = result.get_eigvecs<Form::SYMM, Type::FP32>();
    eigvals.resize(static_cast<size_t>(ldz));
    if(config.compute_eigvecs == Vecs::ON) {
        eigvecs.resize(static_cast<size_t>(ldz * m_req)); // Docs claim ldz * m, but it segfaults when 'V' finds more than m eigvals
    }

    info = LAPACKE_ssyevx_work(LAPACK_COL_MAJOR, jobz, range, 'U', n, matrix, lda, vl, vu, il, iu, 2 * LAPACKE_dlamch('S'), &m_found, eigvals.data(),
                               eigvecs.data(), ldz, lwork_query, -1, iwork.data(), ifail.data());

    int lwork = safe_cast<int>(lwork_query[0]);
    eig::log->trace(" lwork  = {}", lwork);
    eig::log->trace(" liwork = {}", liwork);
    std::vector<fp32> work(static_cast<size_t>(lwork));

    auto t_prep = std::chrono::high_resolution_clock::now();

    info = LAPACKE_ssyevx_work(LAPACK_COL_MAJOR, jobz, range, 'U', n, matrix, lda, vl, vu, il, iu, 2 * LAPACKE_dlamch('S'), &m_found, eigvals.data(),
                               eigvecs.data(), ldz, work.data(), lwork, iwork.data(), ifail.data());

    auto t_total = std::chrono::high_resolution_clock::now();
    if(info == 0) {
        eig::log->trace("Found {} eigenvalues", m_found);
        eigvals.resize(static_cast<size_t>(m_found));
        eigvecs.resize(static_cast<size_t>(ldz * m_found));
        result.meta.eigvecsR_found = config.compute_eigvecs == Vecs::ON;
        result.meta.eigvals_found  = true;
        result.meta.rows           = L;
        result.meta.cols           = m_found;
        result.meta.nev            = m_found;
        result.meta.nev_converged  = m_found;
        result.meta.n              = L;
        result.meta.form           = Form::SYMM;
        result.meta.type           = Type::FP32;
        result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
        result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();
    } else {
        throw std::runtime_error("LAPACK ssyevx failed with error: " + std::to_string(info));
    }
    return info;
}
