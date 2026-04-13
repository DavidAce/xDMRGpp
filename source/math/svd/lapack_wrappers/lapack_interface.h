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

namespace svd::internal::lapack_interface {
    inline constexpr std::size_t fortran_charlen = 1;

    [[nodiscard]] inline int require_col_major(int matrix_layout) {
        return matrix_layout == LAPACK_COL_MAJOR ? 0 : -1;
    }
}

extern "C" {
    void DMRG_LAPACK_sgesvd(const char *jobu, const char *jobvt, const lapack_int *m, const lapack_int *n, float *a, const lapack_int *lda, float *s,
                            float *u, const lapack_int *ldu, float *vt, const lapack_int *ldvt, float *work, const lapack_int *lwork, lapack_int *info,
                            std::size_t jobu_len, std::size_t jobvt_len);
    void DMRG_LAPACK_dgesvd(const char *jobu, const char *jobvt, const lapack_int *m, const lapack_int *n, double *a, const lapack_int *lda, double *s,
                            double *u, const lapack_int *ldu, double *vt, const lapack_int *ldvt, double *work, const lapack_int *lwork,
                            lapack_int *info, std::size_t jobu_len, std::size_t jobvt_len);
    void DMRG_LAPACK_cgesvd(const char *jobu, const char *jobvt, const lapack_int *m, const lapack_int *n, lapack_complex_float *a,
                            const lapack_int *lda, float *s, lapack_complex_float *u, const lapack_int *ldu, lapack_complex_float *vt,
                            const lapack_int *ldvt, lapack_complex_float *work, const lapack_int *lwork, float *rwork, lapack_int *info,
                            std::size_t jobu_len, std::size_t jobvt_len);
    void DMRG_LAPACK_zgesvd(const char *jobu, const char *jobvt, const lapack_int *m, const lapack_int *n, lapack_complex_double *a,
                            const lapack_int *lda, double *s, lapack_complex_double *u, const lapack_int *ldu, lapack_complex_double *vt,
                            const lapack_int *ldvt, lapack_complex_double *work, const lapack_int *lwork, double *rwork, lapack_int *info,
                            std::size_t jobu_len, std::size_t jobvt_len);

    void DMRG_LAPACK_sgesvj(const char *joba, const char *jobu, const char *jobv, const lapack_int *m, const lapack_int *n, float *a,
                            const lapack_int *lda, float *sva, const lapack_int *mv, float *v, const lapack_int *ldv, float *work,
                            const lapack_int *lwork, lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len);
    void DMRG_LAPACK_dgesvj(const char *joba, const char *jobu, const char *jobv, const lapack_int *m, const lapack_int *n, double *a,
                            const lapack_int *lda, double *sva, const lapack_int *mv, double *v, const lapack_int *ldv, double *work,
                            const lapack_int *lwork, lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len);
    void DMRG_LAPACK_cgesvj(const char *joba, const char *jobu, const char *jobv, const lapack_int *m, const lapack_int *n, lapack_complex_float *a,
                            const lapack_int *lda, float *sva, const lapack_int *mv, lapack_complex_float *v, const lapack_int *ldv,
                            lapack_complex_float *cwork, const lapack_int *lwork, float *rwork, const lapack_int *lrwork, lapack_int *info,
                            std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len);
    void DMRG_LAPACK_zgesvj(const char *joba, const char *jobu, const char *jobv, const lapack_int *m, const lapack_int *n,
                            lapack_complex_double *a, const lapack_int *lda, double *sva, const lapack_int *mv, lapack_complex_double *v,
                            const lapack_int *ldv, lapack_complex_double *cwork, const lapack_int *lwork, double *rwork,
                            const lapack_int *lrwork, lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len);

    void DMRG_LAPACK_sgejsv(const char *joba, const char *jobu, const char *jobv, const char *jobr, const char *jobt, const char *jobp,
                            const lapack_int *m, const lapack_int *n, float *a, const lapack_int *lda, float *sva, float *u,
                            const lapack_int *ldu, float *v, const lapack_int *ldv, float *work, const lapack_int *lwork, lapack_int *iwork,
                            lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len, std::size_t jobr_len,
                            std::size_t jobt_len, std::size_t jobp_len);
    void DMRG_LAPACK_dgejsv(const char *joba, const char *jobu, const char *jobv, const char *jobr, const char *jobt, const char *jobp,
                            const lapack_int *m, const lapack_int *n, double *a, const lapack_int *lda, double *sva, double *u,
                            const lapack_int *ldu, double *v, const lapack_int *ldv, double *work, const lapack_int *lwork, lapack_int *iwork,
                            lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len, std::size_t jobr_len,
                            std::size_t jobt_len, std::size_t jobp_len);
    void DMRG_LAPACK_cgejsv(const char *joba, const char *jobu, const char *jobv, const char *jobr, const char *jobt, const char *jobp,
                            const lapack_int *m, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda, float *sva,
                            lapack_complex_float *u, const lapack_int *ldu, lapack_complex_float *v, const lapack_int *ldv,
                            lapack_complex_float *cwork, const lapack_int *lwork, float *rwork, const lapack_int *lrwork, lapack_int *iwork,
                            lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len, std::size_t jobr_len,
                            std::size_t jobt_len, std::size_t jobp_len);
    void DMRG_LAPACK_zgejsv(const char *joba, const char *jobu, const char *jobv, const char *jobr, const char *jobt, const char *jobp,
                            const lapack_int *m, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda, double *sva,
                            lapack_complex_double *u, const lapack_int *ldu, lapack_complex_double *v, const lapack_int *ldv,
                            lapack_complex_double *cwork, const lapack_int *lwork, double *rwork, const lapack_int *lrwork, lapack_int *iwork,
                            lapack_int *info, std::size_t joba_len, std::size_t jobu_len, std::size_t jobv_len, std::size_t jobr_len,
                            std::size_t jobt_len, std::size_t jobp_len);

    void DMRG_LAPACK_sgesdd(const char *jobz, const lapack_int *m, const lapack_int *n, float *a, const lapack_int *lda, float *s, float *u,
                            const lapack_int *ldu, float *vt, const lapack_int *ldvt, float *work, const lapack_int *lwork, lapack_int *iwork,
                            lapack_int *info, std::size_t jobz_len);
    void DMRG_LAPACK_dgesdd(const char *jobz, const lapack_int *m, const lapack_int *n, double *a, const lapack_int *lda, double *s, double *u,
                            const lapack_int *ldu, double *vt, const lapack_int *ldvt, double *work, const lapack_int *lwork, lapack_int *iwork,
                            lapack_int *info, std::size_t jobz_len);
    void DMRG_LAPACK_cgesdd(const char *jobz, const lapack_int *m, const lapack_int *n, lapack_complex_float *a, const lapack_int *lda, float *s,
                            lapack_complex_float *u, const lapack_int *ldu, lapack_complex_float *vt, const lapack_int *ldvt,
                            lapack_complex_float *work, const lapack_int *lwork, float *rwork, lapack_int *iwork, lapack_int *info,
                            std::size_t jobz_len);
    void DMRG_LAPACK_zgesdd(const char *jobz, const lapack_int *m, const lapack_int *n, lapack_complex_double *a, const lapack_int *lda, double *s,
                            lapack_complex_double *u, const lapack_int *ldu, lapack_complex_double *vt, const lapack_int *ldvt,
                            lapack_complex_double *work, const lapack_int *lwork, double *rwork, lapack_int *iwork, lapack_int *info,
                            std::size_t jobz_len);

    void DMRG_LAPACK_sgesvdx(const char *jobu, const char *jobvt, const char *range, const lapack_int *m, const lapack_int *n, float *a,
                             const lapack_int *lda, const float *vl, const float *vu, const lapack_int *il, const lapack_int *iu, lapack_int *ns,
                             float *s, float *u, const lapack_int *ldu, float *vt, const lapack_int *ldvt, float *work, const lapack_int *lwork,
                             lapack_int *iwork, lapack_int *info, std::size_t jobu_len, std::size_t jobvt_len, std::size_t range_len);
    void DMRG_LAPACK_dgesvdx(const char *jobu, const char *jobvt, const char *range, const lapack_int *m, const lapack_int *n, double *a,
                             const lapack_int *lda, const double *vl, const double *vu, const lapack_int *il, const lapack_int *iu, lapack_int *ns,
                             double *s, double *u, const lapack_int *ldu, double *vt, const lapack_int *ldvt, double *work,
                             const lapack_int *lwork, lapack_int *iwork, lapack_int *info, std::size_t jobu_len, std::size_t jobvt_len,
                             std::size_t range_len);
    void DMRG_LAPACK_cgesvdx(const char *jobu, const char *jobvt, const char *range, const lapack_int *m, const lapack_int *n,
                             lapack_complex_float *a, const lapack_int *lda, const float *vl, const float *vu, const lapack_int *il,
                             const lapack_int *iu, lapack_int *ns, float *s, lapack_complex_float *u, const lapack_int *ldu,
                             lapack_complex_float *vt, const lapack_int *ldvt, lapack_complex_float *work, const lapack_int *lwork, float *rwork,
                             lapack_int *iwork, lapack_int *info, std::size_t jobu_len, std::size_t jobvt_len, std::size_t range_len);
    void DMRG_LAPACK_zgesvdx(const char *jobu, const char *jobvt, const char *range, const lapack_int *m, const lapack_int *n,
                             lapack_complex_double *a, const lapack_int *lda, const double *vl, const double *vu, const lapack_int *il,
                             const lapack_int *iu, lapack_int *ns, double *s, lapack_complex_double *u, const lapack_int *ldu,
                             lapack_complex_double *vt, const lapack_int *ldvt, lapack_complex_double *work, const lapack_int *lwork, double *rwork,
                             lapack_int *iwork, lapack_int *info, std::size_t jobu_len, std::size_t jobvt_len, std::size_t range_len);
}

inline int DMRG_sgesvd_work(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, float *a, lapack_int lda, float *s, float *u,
                            lapack_int ldu, float *vt, lapack_int ldvt, float *work, lapack_int lwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgesvd_work(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s, double *u,
                            lapack_int ldu, double *vt, lapack_int ldvt, double *work, lapack_int lwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgesvd_work(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, lapack_complex_float *a, lapack_int lda, float *s,
                            lapack_complex_float *u, lapack_int ldu, lapack_complex_float *vt, lapack_int ldvt, lapack_complex_float *work,
                            lapack_int lwork, float *rwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgesvd_work(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, lapack_complex_double *a, lapack_int lda, double *s,
                            lapack_complex_double *u, lapack_int ldu, lapack_complex_double *vt, lapack_int ldvt, lapack_complex_double *work,
                            lapack_int lwork, double *rwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_sgesvj_work(int matrix_layout, char joba, char jobu, char jobv, lapack_int m, lapack_int n, float *a, lapack_int lda, float *sva,
                            lapack_int mv, float *v, lapack_int ldv, float *work, lapack_int lwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgesvj(&joba, &jobu, &jobv, &m, &n, a, &lda, sva, &mv, v, &ldv, work, &lwork, &info, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgesvj_work(int matrix_layout, char joba, char jobu, char jobv, lapack_int m, lapack_int n, double *a, lapack_int lda, double *sva,
                            lapack_int mv, double *v, lapack_int ldv, double *work, lapack_int lwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgesvj(&joba, &jobu, &jobv, &m, &n, a, &lda, sva, &mv, v, &ldv, work, &lwork, &info, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgesvj_work(int matrix_layout, char joba, char jobu, char jobv, lapack_int m, lapack_int n, lapack_complex_float *a, lapack_int lda,
                            float *sva, lapack_int mv, lapack_complex_float *v, lapack_int ldv, lapack_complex_float *cwork, lapack_int lwork,
                            float *rwork, lapack_int lrwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgesvj(&joba, &jobu, &jobv, &m, &n, a, &lda, sva, &mv, v, &ldv, cwork, &lwork, rwork, &lrwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgesvj_work(int matrix_layout, char joba, char jobu, char jobv, lapack_int m, lapack_int n, lapack_complex_double *a, lapack_int lda,
                            double *sva, lapack_int mv, lapack_complex_double *v, lapack_int ldv, lapack_complex_double *cwork, lapack_int lwork,
                            double *rwork, lapack_int lrwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgesvj(&joba, &jobu, &jobv, &m, &n, a, &lda, sva, &mv, v, &ldv, cwork, &lwork, rwork, &lrwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_sgejsv_work(int matrix_layout, char joba, char jobu, char jobv, char jobr, char jobt, char jobp, lapack_int m, lapack_int n, float *a,
                            lapack_int lda, float *sva, float *u, lapack_int ldu, float *v, lapack_int ldv, float *work, lapack_int lwork,
                            lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgejsv(&joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda, sva, u, &ldu, v, &ldv, work, &lwork, iwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgejsv_work(int matrix_layout, char joba, char jobu, char jobv, char jobr, char jobt, char jobp, lapack_int m, lapack_int n, double *a,
                            lapack_int lda, double *sva, double *u, lapack_int ldu, double *v, lapack_int ldv, double *work, lapack_int lwork,
                            lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgejsv(&joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda, sva, u, &ldu, v, &ldv, work, &lwork, iwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgejsv_work(int matrix_layout, char joba, char jobu, char jobv, char jobr, char jobt, char jobp, lapack_int m, lapack_int n,
                            lapack_complex_float *a, lapack_int lda, float *sva, lapack_complex_float *u, lapack_int ldu, lapack_complex_float *v,
                            lapack_int ldv, lapack_complex_float *cwork, lapack_int lwork, float *rwork, lapack_int lrwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgejsv(&joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda, sva, u, &ldu, v, &ldv, cwork, &lwork, rwork, &lrwork, iwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgejsv_work(int matrix_layout, char joba, char jobu, char jobv, char jobr, char jobt, char jobp, lapack_int m, lapack_int n,
                            lapack_complex_double *a, lapack_int lda, double *sva, lapack_complex_double *u, lapack_int ldu, lapack_complex_double *v,
                            lapack_int ldv, lapack_complex_double *cwork, lapack_int lwork, double *rwork, lapack_int lrwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgejsv(&joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda, sva, u, &ldu, v, &ldv, cwork, &lwork, rwork, &lrwork, iwork, &info,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                       svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_sgesdd_work(int matrix_layout, char jobz, lapack_int m, lapack_int n, float *a, lapack_int lda, float *s, float *u, lapack_int ldu,
                            float *vt, lapack_int ldvt, float *work, lapack_int lwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgesdd(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgesdd_work(int matrix_layout, char jobz, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s, double *u, lapack_int ldu,
                            double *vt, lapack_int ldvt, double *work, lapack_int lwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgesdd(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgesdd_work(int matrix_layout, char jobz, lapack_int m, lapack_int n, lapack_complex_float *a, lapack_int lda, float *s,
                            lapack_complex_float *u, lapack_int ldu, lapack_complex_float *vt, lapack_int ldvt, lapack_complex_float *work,
                            lapack_int lwork, float *rwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgesdd(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgesdd_work(int matrix_layout, char jobz, lapack_int m, lapack_int n, lapack_complex_double *a, lapack_int lda, double *s,
                            lapack_complex_double *u, lapack_int ldu, lapack_complex_double *vt, lapack_int ldvt, lapack_complex_double *work,
                            lapack_int lwork, double *rwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgesdd(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info, svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_sgesvdx_work(int matrix_layout, char jobu, char jobvt, char range, lapack_int m, lapack_int n, float *a, lapack_int lda, float vl,
                             float vu, lapack_int il, lapack_int iu, lapack_int *ns, float *s, float *u, lapack_int ldu, float *vt,
                             lapack_int ldvt, float *work, lapack_int lwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_sgesvdx(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, ns, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info,
                        svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                        svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_dgesvdx_work(int matrix_layout, char jobu, char jobvt, char range, lapack_int m, lapack_int n, double *a, lapack_int lda, double vl,
                             double vu, lapack_int il, lapack_int iu, lapack_int *ns, double *s, double *u, lapack_int ldu, double *vt,
                             lapack_int ldvt, double *work, lapack_int lwork, lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_dgesvdx(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, ns, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info,
                        svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                        svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_cgesvdx_work(int matrix_layout, char jobu, char jobvt, char range, lapack_int m, lapack_int n, lapack_complex_float *a, lapack_int lda,
                             float vl, float vu, lapack_int il, lapack_int iu, lapack_int *ns, float *s, lapack_complex_float *u, lapack_int ldu,
                             lapack_complex_float *vt, lapack_int ldvt, lapack_complex_float *work, lapack_int lwork, float *rwork,
                             lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_cgesvdx(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, ns, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info,
                        svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                        svd::internal::lapack_interface::fortran_charlen);
    return info;
}

inline int DMRG_zgesvdx_work(int matrix_layout, char jobu, char jobvt, char range, lapack_int m, lapack_int n, lapack_complex_double *a, lapack_int lda,
                             double vl, double vu, lapack_int il, lapack_int iu, lapack_int *ns, double *s, lapack_complex_double *u, lapack_int ldu,
                             lapack_complex_double *vt, lapack_int ldvt, lapack_complex_double *work, lapack_int lwork, double *rwork,
                             lapack_int *iwork) {
    if(auto layout_status = svd::internal::lapack_interface::require_col_major(matrix_layout); layout_status != 0) return layout_status;
    lapack_int info = 0;
    DMRG_LAPACK_zgesvdx(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, ns, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info,
                        svd::internal::lapack_interface::fortran_charlen, svd::internal::lapack_interface::fortran_charlen,
                        svd::internal::lapack_interface::fortran_charlen);
    return info;
}
