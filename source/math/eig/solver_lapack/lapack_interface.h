#pragma once

#include "math/lapack/lapack_fortran_symbols.h"
#include <complex>
#include <cstddef>

#ifndef lapack_complex_float
    #define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
    #define lapack_complex_double std::complex<double>
#endif

#ifndef LAPACK_COL_MAJOR
    #define LAPACK_COL_MAJOR 102
#endif

using lapack_int = int;

namespace eig::internal::lapack_interface {
    inline constexpr std::size_t fortran_charlen = 1;

    [[nodiscard]] inline int require_col_major(int matrix_layout) {
        return matrix_layout == LAPACK_COL_MAJOR ? 0 : -1;
    }
}

// Keep the ABI-sensitive LAPACK details in one place:
// - symbol spelling from FortranCInterface
// - hidden CHARACTER length arguments
// - the current lapack_int choice
// - the invariant that callers use column-major storage only

extern "C" {
    void DMRG_LAPACK_sgeev(const char *jobvl, const char *jobvr, const lapack_int *n, float *a, const lapack_int *lda, float *wr, float *wi, float *vl,
                           const lapack_int *ldvl, float *vr, const lapack_int *ldvr, float *work, const lapack_int *lwork, lapack_int *info,
                           std::size_t jobvl_len, std::size_t jobvr_len);
    void DMRG_LAPACK_dgeev(const char *jobvl, const char *jobvr, const lapack_int *n, double *a, const lapack_int *lda, double *wr, double *wi, double *vl,
                           const lapack_int *ldvl, double *vr, const lapack_int *ldvr, double *work, const lapack_int *lwork, lapack_int *info,
                           std::size_t jobvl_len, std::size_t jobvr_len);
    void DMRG_LAPACK_cgeev(const char *jobvl, const char *jobvr, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda, lapack_complex_float *w,
                           lapack_complex_float *vl, const lapack_int *ldvl, lapack_complex_float *vr, const lapack_int *ldvr, lapack_complex_float *work,
                           const lapack_int *lwork, float *rwork, lapack_int *info, std::size_t jobvl_len, std::size_t jobvr_len);
    void DMRG_LAPACK_zgeev(const char *jobvl, const char *jobvr, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda,
                           lapack_complex_double *w, lapack_complex_double *vl, const lapack_int *ldvl, lapack_complex_double *vr, const lapack_int *ldvr,
                           lapack_complex_double *work, const lapack_int *lwork, double *rwork, lapack_int *info, std::size_t jobvl_len,
                           std::size_t jobvr_len);

    void DMRG_LAPACK_ssyevd(const char *jobz, const char *uplo, const lapack_int *n, float *a, const lapack_int *lda, float *w, float *work,
                            const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork, lapack_int *info, std::size_t jobz_len,
                            std::size_t uplo_len);
    void DMRG_LAPACK_dsyevd(const char *jobz, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda, double *w, double *work,
                            const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork, lapack_int *info, std::size_t jobz_len,
                            std::size_t uplo_len);
    void DMRG_LAPACK_cheev(const char *jobz, const char *uplo, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda, float *w,
                           lapack_complex_float *work, const lapack_int *lwork, float *rwork, lapack_int *info, std::size_t jobz_len,
                           std::size_t uplo_len);
    void DMRG_LAPACK_zheev(const char *jobz, const char *uplo, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda, double *w,
                           lapack_complex_double *work, const lapack_int *lwork, double *rwork, lapack_int *info, std::size_t jobz_len,
                           std::size_t uplo_len);
    void DMRG_LAPACK_cheevd(const char *jobz, const char *uplo, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda, float *w,
                            lapack_complex_float *work, const lapack_int *lwork, float *rwork, const lapack_int *lrwork, lapack_int *iwork,
                            const lapack_int *liwork, lapack_int *info, std::size_t jobz_len, std::size_t uplo_len);
    void DMRG_LAPACK_zheevd(const char *jobz, const char *uplo, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda, double *w,
                            lapack_complex_double *work, const lapack_int *lwork, double *rwork, const lapack_int *lrwork, lapack_int *iwork,
                            const lapack_int *liwork, lapack_int *info, std::size_t jobz_len, std::size_t uplo_len);

    void DMRG_LAPACK_ssyevr(const char *jobz, const char *range, const char *uplo, const lapack_int *n, float *a, const lapack_int *lda, const float *vl,
                            const float *vu, const lapack_int *il, const lapack_int *iu, const float *abstol, lapack_int *m, float *w, float *z,
                            const lapack_int *ldz, lapack_int *isuppz, float *work, const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork,
                            lapack_int *info, std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);
    void DMRG_LAPACK_dsyevr(const char *jobz, const char *range, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda,
                            const double *vl, const double *vu, const lapack_int *il, const lapack_int *iu, const double *abstol, lapack_int *m,
                            double *w, double *z, const lapack_int *ldz, lapack_int *isuppz, double *work, const lapack_int *lwork, lapack_int *iwork,
                            const lapack_int *liwork, lapack_int *info, std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);
    void DMRG_LAPACK_cheevr(const char *jobz, const char *range, const char *uplo, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda,
                            const float *vl, const float *vu, const lapack_int *il, const lapack_int *iu, const float *abstol, lapack_int *m, float *w,
                            lapack_complex_float *z, const lapack_int *ldz, lapack_int *isuppz, lapack_complex_float *work, const lapack_int *lwork,
                            float *rwork, const lapack_int *lrwork, lapack_int *iwork, const lapack_int *liwork, lapack_int *info, std::size_t jobz_len,
                            std::size_t range_len, std::size_t uplo_len);
    void DMRG_LAPACK_zheevr(const char *jobz, const char *range, const char *uplo, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda,
                            const double *vl, const double *vu, const lapack_int *il, const lapack_int *iu, const double *abstol, lapack_int *m,
                            double *w, lapack_complex_double *z, const lapack_int *ldz, lapack_int *isuppz, lapack_complex_double *work,
                            const lapack_int *lwork, double *rwork, const lapack_int *lrwork, lapack_int *iwork, const lapack_int *liwork,
                            lapack_int *info, std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);

    void DMRG_LAPACK_ssyevx(const char *jobz, const char *range, const char *uplo, const lapack_int *n, float *a, const lapack_int *lda, const float *vl,
                            const float *vu, const lapack_int *il, const lapack_int *iu, const float *abstol, lapack_int *m, float *w, float *z,
                            const lapack_int *ldz, float *work, const lapack_int *lwork, lapack_int *iwork, lapack_int *ifail, lapack_int *info,
                            std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);
    void DMRG_LAPACK_dsyevx(const char *jobz, const char *range, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda,
                            const double *vl, const double *vu, const lapack_int *il, const lapack_int *iu, const double *abstol, lapack_int *m,
                            double *w, double *z, const lapack_int *ldz, double *work, const lapack_int *lwork, lapack_int *iwork, lapack_int *ifail,
                            lapack_int *info, std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);

    void DMRG_LAPACK_ssygvd(const lapack_int *itype, const char *jobz, const char *uplo, const lapack_int *n, float *a, const lapack_int *lda, float *b,
                            const lapack_int *ldb, float *w, float *work, const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork,
                            lapack_int *info, std::size_t jobz_len, std::size_t uplo_len);
    void DMRG_LAPACK_dsygvd(const lapack_int *itype, const char *jobz, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda, double *b,
                            const lapack_int *ldb, double *w, double *work, const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork,
                            lapack_int *info, std::size_t jobz_len, std::size_t uplo_len);
    void DMRG_LAPACK_ssygvx(const lapack_int *itype, const char *jobz, const char *range, const char *uplo, const lapack_int *n, float *a,
                            const lapack_int *lda, float *b, const lapack_int *ldb, const float *vl, const float *vu, const lapack_int *il,
                            const lapack_int *iu, const float *abstol, lapack_int *m, float *w, float *z, const lapack_int *ldz, float *work,
                            const lapack_int *lwork, lapack_int *iwork, lapack_int *ifail, lapack_int *info, std::size_t jobz_len, std::size_t range_len,
                            std::size_t uplo_len);
    void DMRG_LAPACK_dsygvx(const lapack_int *itype, const char *jobz, const char *range, const char *uplo, const lapack_int *n, double *a,
                            const lapack_int *lda, double *b, const lapack_int *ldb, const double *vl, const double *vu, const lapack_int *il,
                            const lapack_int *iu, const double *abstol, lapack_int *m, double *w, double *z, const lapack_int *ldz, double *work,
                            const lapack_int *lwork, lapack_int *iwork, lapack_int *ifail, lapack_int *info, std::size_t jobz_len, std::size_t range_len,
                            std::size_t uplo_len);

    float  DMRG_LAPACK_slamch(const char *cmach, std::size_t cmach_len);
    double DMRG_LAPACK_dlamch(const char *cmach, std::size_t cmach_len);
}

inline float DMRG_slamch(char cmach) { return DMRG_LAPACK_slamch(&cmach, eig::internal::lapack_interface::fortran_charlen); }
inline double DMRG_dlamch(char cmach) { return DMRG_LAPACK_dlamch(&cmach, eig::internal::lapack_interface::fortran_charlen); }

inline int DMRG_sgeev_work(int matrix_layout, char jobvl, char jobvr, lapack_int n, float *a, lapack_int lda, float *wr, float *wi, float *vl,
                              lapack_int ldvl, float *vr, lapack_int ldvr, float *work, lapack_int lwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgeev(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, eig::internal::lapack_interface::fortran_charlen,
                      eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgeev_work(int matrix_layout, char jobvl, char jobvr, lapack_int n, double *a, lapack_int lda, double *wr, double *wi, double *vl,
                              lapack_int ldvl, double *vr, lapack_int ldvr, double *work, lapack_int lwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgeev(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, eig::internal::lapack_interface::fortran_charlen,
                      eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgeev_work(int matrix_layout, char jobvl, char jobvr, lapack_int n, lapack_complex_float *a, lapack_int lda, lapack_complex_float *w,
                              lapack_complex_float *vl, lapack_int ldvl, lapack_complex_float *vr, lapack_int ldvr, lapack_complex_float *work,
                              lapack_int lwork, float *rwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgeev(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info,
                      eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgeev_work(int matrix_layout, char jobvl, char jobvr, lapack_int n, lapack_complex_double *a, lapack_int lda, lapack_complex_double *w,
                              lapack_complex_double *vl, lapack_int ldvl, lapack_complex_double *vr, lapack_int ldvr, lapack_complex_double *work,
                              lapack_int lwork, double *rwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgeev(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info,
                      eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_ssyevd_work(int matrix_layout, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w, float *work, lapack_int lwork,
                               lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_ssyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dsyevd_work(int matrix_layout, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w, double *work, lapack_int lwork,
                               lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dsyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cheev_work(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float *a, lapack_int lda, float *w,
                              lapack_complex_float *work, lapack_int lwork, float *rwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cheev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info, eig::internal::lapack_interface::fortran_charlen,
                      eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zheev_work(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double *a, lapack_int lda, double *w,
                              lapack_complex_double *work, lapack_int lwork, double *rwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zheev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info, eig::internal::lapack_interface::fortran_charlen,
                      eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cheevd_work(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float *a, lapack_int lda, float *w,
                               lapack_complex_float *work, lapack_int lwork, float *rwork, lapack_int lrwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cheevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zheevd_work(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double *a, lapack_int lda, double *w,
                               lapack_complex_double *work, lapack_int lwork, double *rwork, lapack_int lrwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zheevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_ssyevr_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, float *a, lapack_int lda, float vl, float vu,
                               lapack_int il, lapack_int iu, float abstol, lapack_int *m, float *w, float *z, lapack_int ldz, lapack_int *isuppz,
                               float *work, lapack_int lwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_ssyevr(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dsyevr_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, double *a, lapack_int lda, double vl, double vu,
                               lapack_int il, lapack_int iu, double abstol, lapack_int *m, double *w, double *z, lapack_int ldz, lapack_int *isuppz,
                               double *work, lapack_int lwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dsyevr(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cheevr_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, lapack_complex_float *a, lapack_int lda, float vl,
                               float vu, lapack_int il, lapack_int iu, float abstol, lapack_int *m, float *w, lapack_complex_float *z, lapack_int ldz,
                               lapack_int *isuppz, lapack_complex_float *work, lapack_int lwork, float *rwork, lapack_int lrwork, lapack_int *iwork,
                               lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cheevr(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork,
                       &liwork, &info, eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zheevr_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, lapack_complex_double *a, lapack_int lda, double vl,
                               double vu, lapack_int il, lapack_int iu, double abstol, lapack_int *m, double *w, lapack_complex_double *z, lapack_int ldz,
                               lapack_int *isuppz, lapack_complex_double *work, lapack_int lwork, double *rwork, lapack_int lrwork, lapack_int *iwork,
                               lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zheevr(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork,
                       &liwork, &info, eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_ssyevx_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, float *a, lapack_int lda, float vl, float vu,
                               lapack_int il, lapack_int iu, float abstol, lapack_int *m, float *w, float *z, lapack_int ldz, float *work,
                               lapack_int lwork, lapack_int *iwork, lapack_int *ifail) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_ssyevx(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dsyevx_work(int matrix_layout, char jobz, char range, char uplo, lapack_int n, double *a, lapack_int lda, double vl, double vu,
                               lapack_int il, lapack_int iu, double abstol, lapack_int *m, double *w, double *z, lapack_int ldz, double *work,
                               lapack_int lwork, lapack_int *iwork, lapack_int *ifail) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dsyevx(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail, &info,
                       eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_ssygvd_work(int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *b,
                               lapack_int ldb, float *w, float *work, lapack_int lwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_ssygvd(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dsygvd_work(int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *b,
                               lapack_int ldb, double *w, double *work, lapack_int lwork, lapack_int *iwork, lapack_int liwork) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dsygvd(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_ssygvx_work(int matrix_layout, lapack_int itype, char jobz, char range, char uplo, lapack_int n, float *a, lapack_int lda, float *b,
                               lapack_int ldb, float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int *m, float *w, float *z,
                               lapack_int ldz, float *work, lapack_int lwork, lapack_int *iwork, lapack_int *ifail) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_ssygvx(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail,
                       &info, eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dsygvx_work(int matrix_layout, lapack_int itype, char jobz, char range, char uplo, lapack_int n, double *a, lapack_int lda, double *b,
                               lapack_int ldb, double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int *m, double *w, double *z,
                               lapack_int ldz, double *work, lapack_int lwork, lapack_int *iwork, lapack_int *ifail) {
    if(auto layout_status = eig::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dsygvx(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail,
                       &info, eig::internal::lapack_interface::fortran_charlen, eig::internal::lapack_interface::fortran_charlen,
                       eig::internal::lapack_interface::fortran_charlen);
    return info;
}
