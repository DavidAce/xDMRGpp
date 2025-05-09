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

int eig::solver::sgeev(fp32 *matrix, size_type L) {
    eig::log->trace("Starting eig_sgeev (non-optimized)");
    auto  t_start      = std::chrono::high_resolution_clock::now();
    auto &eigvals_real = result.eigvals_real_fp32;
    auto &eigvals_imag = result.eigvals_imag_fp32;
    eigvals_real.resize(safe_cast<size_t>(L));
    eigvals_imag.resize(safe_cast<size_t>(L));
    std::vector<fp32> eigvecsR_tmp(safe_cast<size_t>(L * L));
    std::vector<fp32> eigvecsL_tmp(safe_cast<size_t>(L * L));

    int  info = 0;
    int  Lint = safe_cast<int>(L);
    fp32 lwork_query;
    char jobz = config.compute_eigvecs == Vecs::ON ? 'V' : 'N';

    info = LAPACKE_sgeev_work(LAPACK_COL_MAJOR, jobz, jobz, Lint, matrix, safe_cast<int>(L), eigvals_real.data(), eigvals_imag.data(), eigvecsL_tmp.data(),
                              Lint, eigvecsR_tmp.data(), safe_cast<int>(L), &lwork_query, -1);
    int               lwork = (int) std::real(2.0f * lwork_query); // Make it twice as big for performance.
    std::vector<fp32> work(safe_cast<size_t>(lwork));
    auto              t_prep = std::chrono::high_resolution_clock::now();
    info = LAPACKE_sgeev_work(LAPACK_COL_MAJOR, jobz, jobz, Lint, matrix, safe_cast<int>(L), eigvals_real.data(), eigvals_imag.data(), eigvecsL_tmp.data(),
                              Lint, eigvecsR_tmp.data(), safe_cast<int>(L), work.data(), lwork);
    auto t_total = std::chrono::high_resolution_clock::now();
    if(info == 0) {
        result.meta.eigvecsR_found = true;
        result.meta.eigvecsL_found = true;
        result.meta.eigvals_found  = true;
        result.meta.rows           = L;
        result.meta.cols           = L;
        result.meta.nev            = Lint;
        result.meta.nev_converged  = Lint;
        result.meta.n              = L;
        result.meta.form           = Form::NSYM;
        result.meta.type           = Type::FP32;
        result.meta.time_prep      = std::chrono::duration<double>(t_prep - t_start).count();
        result.meta.time_total     = std::chrono::duration<double>(t_total - t_start).count();
    } else {
        throw std::runtime_error("LAPACK sgeev failed with error: " + std::to_string(info));
    }

    auto &eigvals  = result.eigvals_cx32;
    auto &eigvecsR = result.eigvecsR_cx32;
    auto &eigvecsL = result.eigvecsL_cx32;
    eigvals.resize(safe_cast<size_t>(L));
    eigvecsR.resize(safe_cast<size_t>(L * L));
    eigvecsL.resize(safe_cast<size_t>(L * L));

    // Copy eigenvalues
    for(size_t i = 0; i < safe_cast<size_t>(L); i++) eigvals[i] = cx32(eigvals_real[i], eigvals_imag[i]);
    // Release real/imag parts
    eigvals_real.clear();
    eigvals_imag.clear();

    // Copy eigenvectors
    auto count = 0ul;
    auto rows  = safe_cast<size_t>(L);
    auto cols  = safe_cast<size_t>(L);
    for(size_t i = 0; i < rows; i++) {
        size_t j = 0;
        while(j < cols) {
            if(eigvals_imag[j] == 0.0f) {
                eigvecsR[count] = eigvecsR_tmp[i + j * rows]; //(i,j)
                eigvecsL[count] = eigvecsL_tmp[i + j * rows]; //(i,j)
                count++;
                j++;
            } else {
                eigvecsR[count] = cx32(eigvecsR_tmp[i + j * rows], eigvecsR_tmp[i + (j + 1) * rows]);
                eigvecsL[count] = cx32(eigvecsL_tmp[i + j * rows], eigvecsL_tmp[i + (j + 1) * rows]);
                count++;
                eigvecsR[count] = cx32(eigvecsR_tmp[i + j * rows], -eigvecsR_tmp[i + (j + 1) * rows]);
                eigvecsL[count] = cx32(eigvecsL_tmp[i + j * rows], -eigvecsL_tmp[i + (j + 1) * rows]);
                count++;
                j += 2;
            }
        }
    }

    return info;
}
