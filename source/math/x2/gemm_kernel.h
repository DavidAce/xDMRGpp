#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <type_traits>
#include <utility>

namespace x2_detail {

    template<typename Scalar>
    using RealT                                   = decltype(std::real(std::declval<Scalar>()));
    template<typename T> constexpr bool is_real_v = Eigen::NumTraits<T>::IsComplex == 0;
    template<typename T> constexpr bool is_cplx_v = Eigen::NumTraits<T>::IsComplex == 1;

    template<typename Scalar>
    EIGEN_STRONG_INLINE static void add_with_residual(const Scalar &a, const Scalar &b, Scalar &sum, Scalar &err) {
        if constexpr(is_real_v<Scalar>) {
            // Error-free transform for sum in IEEE arithmetic (componentwise for complex).
            sum             = a + b;
            const Scalar bb = sum - a;
            err             = (a - (sum - bb)) + (b - bb);
        } else {
            using Real = RealT<Scalar>;
            Real sum_r, err_r;
            Real sum_i, err_i;

            add_with_residual<Real>(std::real(a), std::real(b), sum_r, err_r);
            add_with_residual<Real>(std::imag(a), std::imag(b), sum_i, err_i);

            sum = Scalar{sum_r, sum_i};
            err = Scalar{err_r, err_i};
        }
    }

    template<typename Scalar>
    EIGEN_STRONG_INLINE static Scalar mul_residual(const Scalar &a, const Scalar &b, const Scalar &p) {
        if constexpr(is_real_v<Scalar>) {
            // For real types: exact product residual via fma
            return Scalar(std::fma(a, b, -p));
        } else {
            // Returns e such that (a*b) = p + e, where p is the rounded complex product.
            // We compute the exact residual from:
            //   ar*br - ai*bi  and  ar*bi + ai*br
            // using std::fma for multiplication residuals and add_with_residual for the add/sub residual.
            using Real = RealT<Scalar>;

            const Real ar = std::real(a);
            const Real ai = std::imag(a);
            const Real br = std::real(b);
            const Real bi = std::imag(b);

            // Rounded partial products (hi parts)
            const Real p1r = ar * br; // ar*br
            const Real p2r = ai * bi; // ai*bi
            const Real p1i = ar * bi; // ar*bi
            const Real p2i = ai * br; // ai*br

            // Exact residuals of the multiplications via fma (lo parts)
            const Real p1r_err = std::fma(ar, br, -p1r);
            const Real p2r_err = std::fma(ai, bi, -p2r);
            const Real p1i_err = std::fma(ar, bi, -p1i);
            const Real p2i_err = std::fma(ai, br, -p2i);

            // Form the rounded "sum" the same way as the standard complex formula:
            //   real: p1r - p2r
            //   imag: p1i + p2i
            // Capture the exact rounding error of these add/sub operations with add_with_residual.
            const Scalar x{p1r, p1i};
            const Scalar y{-p2r, p2i};

            Scalar sum_hi, sum_add_err;
            add_with_residual<Scalar>(x, y, sum_hi, sum_add_err);

            // sum_hi is our rounded reconstruction; p is the actual rounded product we were given.
            const Real delta_r = std::real(sum_hi) - std::real(p);
            const Real delta_i = std::imag(sum_hi) - std::imag(p);

            // Combine all low parts:
            // exact(a*b) = (sum_hi + sum_add_err) + mult_err_terms
            // residual   = exact(a*b) - p
            const Real mult_err_r = p1r_err - p2r_err;
            const Real mult_err_i = p1i_err + p2i_err;

            const Real err_r = delta_r + std::real(sum_add_err) + mult_err_r;
            const Real err_i = delta_i + std::imag(sum_add_err) + mult_err_i;

            return Scalar{err_r, err_i};
        }
    }

    template<typename Scalar>
    EIGEN_STRONG_INLINE static void accumulate(Scalar &sum_hi, Scalar &sum_lo, const Scalar &p_hi, const Scalar &p_lo) {
        // Add p_hi into sum_hi with add_with_residual; push all error into sum_lo (plus p_lo).
        Scalar s, e;
        add_with_residual(sum_hi, p_hi, s, e);
        sum_hi = s;
        sum_lo += (e + p_lo);
    }

} // namespace x2_detail

template<typename DerivedC, typename DerivedA, typename DerivedB>
void gemm_x2_kernel_fused_packed(Eigen::DenseBase<DerivedC> &C_hi, Eigen::DenseBase<DerivedC> &C_lo, const Eigen::DenseBase<DerivedA> &A,
                                 const Eigen::DenseBase<DerivedB> &B) {
    using Scalar = typename DerivedC::Scalar;
    static_assert(std::is_same_v<Scalar, typename DerivedA::Scalar>);
    static_assert(std::is_same_v<Scalar, typename DerivedB::Scalar>);
    static_assert(std::is_same_v<Scalar, typename DerivedC::Scalar>);

    static_assert((DerivedA::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor
    static_assert((DerivedB::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor
    static_assert((DerivedC::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor

    static_assert((DerivedA::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access
    static_assert((DerivedB::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access
    static_assert((DerivedC::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access

    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A.rows();
    const Eigen::Index k = A.cols();
    const Eigen::Index n = B.cols();

    assert(A.rows() == m && A.cols() == k);
    assert(B.rows() == k);
    assert(B.rows() == k && B.cols() == n);

    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);

    const Scalar *Ap = A.derived().data();
    const Scalar *Bp = B.derived().data();
    Scalar       *Ch = C_hi.derived().data();
    Scalar       *Cl = C_lo.derived().data();

    const Eigen::Index lda = A.outerStride();
    const Eigen::Index ldb = B.outerStride();
    const Eigen::Index ldc = C_hi.outerStride();
    assert(lda == A.outerStride());
    assert(ldb == B.outerStride());
    assert(ldc == C_lo.outerStride());

    assert(A.innerStride() == 1);
    assert(B.innerStride() == 1);
    assert(C_hi.innerStride() == 1);
    assert(C_lo.innerStride() == 1);

#pragma omp parallel
    {
        alignas(64) Scalar sum_hi_buf[JB * IB]; // index: [j * IB + i]
        alignas(64) Scalar sum_lo_buf[JB * IB]; // index: [j * IB + i]

        // Packed B panels: (kb x jb) stored row-major: [kk][j]
        alignas(64) Scalar Bpack[BK * JB];

#pragma omp for collapse(2) schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);
                const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                // column-wise zero the active block
                for(Eigen::Index j = 0; j < jb; ++j) {
                    std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                    std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                }

                for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                    const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                    // ---- PACK (B_hi, B_lo) PANEL (kb x jb) ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Eigen::Index k_idx = k0 + kk;
                        Scalar            *row_h = Bpack + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Eigen::Index idx = k_idx + (j0 + j) * ldb;
                            row_h[j]               = Bp[idx];
                        }
                    }

                    // ---- COMPUTE USING PACKED B ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *bp_row = Bpack + kk * JB;
                        const Scalar *ap_col = Ap + i0 + (k0 + kk) * lda;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bp = bp_row[j];

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar ap = ap_col[i];

                                const Scalar p_hi = ap * bp;
                                const Scalar p_lo = x2_detail::mul_residual(ap, bp, p_hi);

                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);
                            }
                        }
                    }
                }
                // Store block back (new layout: sum_[j * IB + i])
                for(Eigen::Index j = 0; j < jb; ++j) {
                    Scalar *__restrict ch_col = Ch + i0 + (j0 + j) * ldc;
                    Scalar *__restrict cl_col = Cl + i0 + (j0 + j) * ldc;

                    const Scalar *__restrict sh = sum_hi_buf + j * IB;
                    const Scalar *__restrict sl = sum_lo_buf + j * IB;

                    for(Eigen::Index i = 0; i < ib; ++i) {
                        ch_col[i] = sh[i];
                        cl_col[i] = sl[i];
                    }
                }
            }
        }
    }

    assert(C_hi.allFinite());
    assert(C_lo.allFinite());
}

template<typename DerivedC, typename DerivedA, typename DerivedB>
void gemm_x2_kernel_fused_packed(Eigen::DenseBase<DerivedC> &C_hi, Eigen::DenseBase<DerivedC> &C_lo, const Eigen::DenseBase<DerivedA> &A_hi,
                                 const Eigen::DenseBase<DerivedA> &A_lo, const Eigen::DenseBase<DerivedB> &B_hi, const Eigen::DenseBase<DerivedB> &B_lo) {
    using Scalar = typename DerivedA::Scalar;
    static_assert(std::is_same_v<Scalar, typename DerivedA::Scalar>);
    static_assert(std::is_same_v<Scalar, typename DerivedB::Scalar>);
    static_assert(std::is_same_v<Scalar, typename DerivedC::Scalar>);

    static_assert((DerivedA::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access
    static_assert((DerivedB::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access
    static_assert((DerivedC::Flags & Eigen::DirectAccessBit) != 0); // Require .data() access

    static_assert((DerivedA::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor
    static_assert((DerivedB::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor
    static_assert((DerivedC::Flags & Eigen::RowMajorBit) == 0); // Require ColMajor

    constexpr Eigen::Index IB = 64;
    constexpr Eigen::Index JB = 8;
    constexpr Eigen::Index BK = 128;

    const Eigen::Index m = A_hi.rows();
    const Eigen::Index k = A_hi.cols();
    const Eigen::Index n = B_hi.cols();

    assert(A_lo.rows() == m && A_lo.cols() == k);
    assert(B_hi.rows() == k);
    assert(B_lo.rows() == k && B_lo.cols() == n);

    assert(C_hi.rows() == m && C_hi.cols() == n);
    assert(C_lo.rows() == m && C_lo.cols() == n);

    const Scalar *Ah = A_hi.derived().data();
    const Scalar *Al = A_lo.derived().data();
    const Scalar *Bh = B_hi.derived().data();
    const Scalar *Bl = B_lo.derived().data();
    Scalar       *Ch = C_hi.derived().data();
    Scalar       *Cl = C_lo.derived().data();

    const Eigen::Index lda = A_hi.outerStride();
    const Eigen::Index ldb = B_hi.outerStride();
    const Eigen::Index ldc = C_hi.outerStride();
    assert(lda == A_lo.outerStride());
    assert(ldb == B_lo.outerStride());
    assert(ldc == C_lo.outerStride());

    assert(A_hi.innerStride() == 1);
    assert(A_lo.innerStride() == 1);
    assert(B_hi.innerStride() == 1);
    assert(B_lo.innerStride() == 1);
    assert(C_hi.innerStride() == 1);
    assert(C_lo.innerStride() == 1);

#pragma omp parallel
    {
        alignas(64) Scalar sum_hi_buf[JB * IB]; // index: [j * IB + i]
        alignas(64) Scalar sum_lo_buf[JB * IB]; // index: [j * IB + i]

        // Packed B panels: (kb x jb) stored row-major: [kk][j]
        alignas(64) Scalar Bpack_hi[BK * JB];
        alignas(64) Scalar Bpack_lo[BK * JB];

#pragma omp for collapse(2) schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);
                const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                // column-wise zero the active block
                for(Eigen::Index j = 0; j < jb; ++j) {
                    std::fill_n(sum_hi_buf + j * IB, ib, Scalar(0));
                    std::fill_n(sum_lo_buf + j * IB, ib, Scalar(0));
                }

                for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                    const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                    // ---- PACK (B_hi, B_lo) PANEL (kb x jb) ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Eigen::Index k_idx = k0 + kk;
                        Scalar            *row_h = Bpack_hi + kk * JB;
                        Scalar            *row_l = Bpack_lo + kk * JB;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Eigen::Index idx = k_idx + (j0 + j) * ldb;
                            row_h[j]               = Bh[idx];
                            row_l[j]               = Bl[idx];
                        }
                    }

                    // ---- COMPUTE USING PACKED B ----
                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Scalar *brow_h = Bpack_hi + kk * JB;
                        const Scalar *brow_l = Bpack_lo + kk * JB;

                        const Scalar *ah_col = Ah + i0 + (k0 + kk) * lda;
                        const Scalar *al_col = Al + i0 + (k0 + kk) * lda;

                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar bh    = brow_h[j];
                            const Scalar bl    = brow_l[j];
                            const Scalar bh_bl = bh + bl;

                            Scalar *__restrict sh = sum_hi_buf + j * IB;
                            Scalar *__restrict sl = sum_lo_buf + j * IB;

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar ah = ah_col[i];
                                const Scalar al = al_col[i];

                                const Scalar p_hi = ah * bh;
                                const Scalar p_lo = x2_detail::mul_residual(ah, bh, p_hi);

                                x2_detail::accumulate(sh[i], sl[i], p_hi, p_lo);
                                if constexpr(std::is_floating_point_v<Scalar>) {
                                    sl[i] = std::fma(ah, bl, sl[i]);
                                    sl[i] = std::fma(al, bh_bl, sl[i]);
                                } else {
                                    sl[i] += ah * bl;
                                    sl[i] += al * bh_bl;
                                }
                            }
                        }
                    }
                }
                // Store block back (new layout: sum_[j * IB + i])
                for(Eigen::Index j = 0; j < jb; ++j) {
                    Scalar *__restrict ch_col = Ch + i0 + (j0 + j) * ldc;
                    Scalar *__restrict cl_col = Cl + i0 + (j0 + j) * ldc;

                    const Scalar *__restrict sh = sum_hi_buf + j * IB;
                    const Scalar *__restrict sl = sum_lo_buf + j * IB;

                    for(Eigen::Index i = 0; i < ib; ++i) {
                        ch_col[i] = sh[i];
                        cl_col[i] = sl[i];
                    }
                }
            }
        }
    }

    assert(C_hi.allFinite());
    assert(C_lo.allFinite());
}
