#include <algorithm>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <type_traits>
#include <utility>
#include <cassert>

namespace x2_detail {

    template<typename Scalar>
    using RealT                                   = decltype(std::real(std::declval<Scalar>()));
    template<typename T> constexpr bool is_real_v = Eigen::NumTraits<T>::IsComplex == 0;
    template<typename T> constexpr bool is_cplx_v = Eigen::NumTraits<T>::IsComplex == 1;

    template<typename Scalar>
    static void add_with_residual(const Scalar &a, const Scalar &b, Scalar &sum, Scalar &err) {
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
    static Scalar mul_residual(const Scalar &a, const Scalar &b, const Scalar &p) {
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
    static void dd_accumulate(Scalar &sum_hi, Scalar &sum_lo, const Scalar &p_hi, const Scalar &p_lo) {
        // Add p_hi into sum_hi with add_with_residual; push all error into sum_lo (plus p_lo).
        Scalar s, e;
        add_with_residual(sum_hi, p_hi, s, e);
        sum_hi = s;
        sum_lo += (e + p_lo);
    }

} // namespace x2_detail

template<typename DerivedA, typename DerivedB, typename DerivedC>
void gemm_x2_kernel(Eigen::DenseBase<DerivedC> &C_hi, Eigen::DenseBase<DerivedC> &C_lo, const Eigen::DenseBase<DerivedA> &A_hi,
                    const Eigen::DenseBase<DerivedB> &B_hi, Eigen::Index IB, Eigen::Index JB, Eigen::Index BK) {
    using ScalarA = typename DerivedA::Scalar;
    using ScalarB = typename DerivedB::Scalar;
    using ScalarC = typename DerivedC::Scalar;
    static_assert(std::is_same_v<ScalarA, ScalarB>);
    static_assert(std::is_same_v<ScalarA, ScalarC>);
    static_assert((DerivedA::Flags & Eigen::DirectAccessBit) != 0); // Check that we have direct access to buffer via .data()
    static_assert((DerivedB::Flags & Eigen::DirectAccessBit) != 0); // Check that we have direct access to buffer via .data()
    static_assert((DerivedC::Flags & Eigen::DirectAccessBit) != 0); // Check that we have direct access to buffer via .data()

    using Scalar = ScalarA;

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index m = A_hi.rows();
    const Eigen::Index k = A_hi.cols();
    const Eigen::Index n = B_hi.cols();
    assert(B_hi.rows() == k);
    assert(IB >= 1 && JB >= 1 && BK >= 1);
    assert(C_hi.rows() == m);
    assert(C_hi.cols() == n);
    assert(C_lo.rows() == m);
    assert(C_lo.cols() == n);


    const Scalar *Aptr = A_hi.derived().data();
    const Scalar *Bptr = B_hi.derived().data();
    Scalar       *Ch   = C_hi.derived().data();
    Scalar       *Cl   = C_lo.derived().data();

    // Assume column-major (your codebase does). Use outerStride to be safe with Ref.
    const Eigen::Index lda = A_hi.outerStride();
    const Eigen::Index ldb = B_hi.outerStride();
    const Eigen::Index ldc = C_hi.outerStride();
    assert(ldc == C_lo.outerStride());

    assert(A_hi.innerStride() == 1);
    assert(B_hi.innerStride() == 1);
    assert(C_hi.innerStride() == 1);
    assert(C_lo.innerStride() == 1);

#pragma omp parallel
    {
        // Thread-local block accumulators (reused to avoid alloc churn)
        MatrixType sum_hi_buf;
        MatrixType sum_lo_buf;
        sum_hi_buf.resize(IB, JB);
        sum_lo_buf.resize(IB, JB);

#pragma omp for collapse(2) schedule(static)
        for(Eigen::Index j0 = 0; j0 < n; j0 += JB) {
            for(Eigen::Index i0 = 0; i0 < m; i0 += IB) {
                const Eigen::Index jb = std::min<Eigen::Index>(JB, n - j0);
                const Eigen::Index ib = std::min<Eigen::Index>(IB, m - i0);

                // Only touch the active top-left corner
                sum_hi_buf.topLeftCorner(ib, jb).setZero();
                sum_lo_buf.topLeftCorner(ib, jb).setZero();

                // Iterate k in increasing order (stable, predictable accuracy)
                for(Eigen::Index k0 = 0; k0 < k; k0 += BK) {
                    const Eigen::Index kb = std::min<Eigen::Index>(BK, k - k0);

                    for(Eigen::Index kk = 0; kk < kb; ++kk) {
                        const Eigen::Index k_idx = k0 + kk;

                        // B(k_idx, j) is contiguous in k only if row-major.
                        // Since we are column-major, access B by columns j.
                        for(Eigen::Index j = 0; j < jb; ++j) {
                            const Scalar b = Bptr[k_idx + (j0 + j) * ldb];

                            for(Eigen::Index i = 0; i < ib; ++i) {
                                const Scalar a    = Aptr[(i0 + i) + k_idx * lda];
                                const Scalar p_hi = a * b;
                                const Scalar p_lo = x2_detail::mul_residual(a, b, p_hi);

                                Scalar &sh = sum_hi_buf(i, j);
                                Scalar &sl = sum_lo_buf(i, j);
                                x2_detail::dd_accumulate(sh, sl, p_hi, p_lo);
                            }
                        }
                    }
                }

                // Store block back
                for(Eigen::Index j = 0; j < jb; ++j) {
                    for(Eigen::Index i = 0; i < ib; ++i) {
                        Ch[(i0 + i) + (j0 + j) * ldc] = sum_hi_buf(i, j);
                        Cl[(i0 + i) + (j0 + j) * ldc] = sum_lo_buf(i, j);
                    }
                }
            }
        }
    }
    assert(C_hi.allFinite());
    assert(C_lo.allFinite());
}
