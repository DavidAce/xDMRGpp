#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
namespace x2 {
    enum class Access : unsigned char { ReadOnly, ReadWrite };

    template<typename Scalar_>
    inline static typename Eigen::NumTraits<Scalar_>::Real norm(const Scalar_ *const hi_ptr, const Scalar_ *const lo_ptr, Eigen::Index n) {
        // Frobenius norm of (hi + lo)
        using Scalar     = Scalar_;
        using RealScalar = Eigen::NumTraits<Scalar>::Real;

        if(n == 0) return RealScalar{0};

        using RealAcc = std::conditional_t<sizeof(RealScalar) <= sizeof(double), long double, fp128>;
        RealAcc sum   = RealAcc{0};

#pragma omp parallel for reduction(+ : sum) schedule(static)
        for(Eigen::Index i = 0; i < n; ++i) {
            const Scalar a = hi_ptr[i];
            const Scalar b = lo_ptr[i];

            if constexpr(std::is_floating_point_v<Scalar>) {
                const RealAcc ar = static_cast<RealAcc>(a);
                const RealAcc br = static_cast<RealAcc>(b);

                const RealAcc aa = ar * ar;
                const RealAcc bb = br * br;
                const RealAcc ab = ar * br;

                sum += aa + RealAcc{2} * ab + bb;
            } else {
                // |a|^2 + 2 Re(conj(a)b) + |b|^2
                const RealAcc ar = static_cast<RealAcc>(std::real(a));
                const RealAcc ai = static_cast<RealAcc>(std::imag(a));
                const RealAcc br = static_cast<RealAcc>(std::real(b));
                const RealAcc bi = static_cast<RealAcc>(std::imag(b));

                const RealAcc aa = ar * ar + ai * ai;
                const RealAcc bb = br * br + bi * bi;
                const RealAcc ab = ar * br + ai * bi; // Re(conj(a)*b)

                sum += aa + RealAcc{2} * ab + bb;
            }
        }

        if(sum < RealAcc{0}) sum = RealAcc{0}; // guard tiny negative from rounding
        return static_cast<RealScalar>(std::sqrt(sum));
    }
}

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

    template<auto rank>
    EIGEN_STRONG_INLINE static Eigen::Index size_from_dims(const std::array<Eigen::Index, rank> &d) {
        Eigen::Index s = 1;
        using rtype    = decltype(rank);
        for(rtype i = 0; i < rank; ++i) s *= d[static_cast<size_t>(i)];
        return s;
    }
    template<typename Scalar>
    EIGEN_STRONG_INLINE static void accumulate(Scalar &sum_hi, Scalar &sum_lo, const Scalar &p_hi, const Scalar &p_lo) {
        // Add p_hi into sum_hi with add_with_residual; push all error into sum_lo (plus p_lo).
        Scalar s, e;
        add_with_residual(sum_hi, p_hi, s, e);
        sum_hi  = s;
        sum_lo += (e + p_lo);
    }

    template<typename Scalar>
    EIGEN_STRONG_INLINE static void accumulate3(Scalar &sum_hi, Scalar &sum_lo, Scalar &sum_ll, const Scalar &p_hi, const Scalar &p_lo) {
        // Step 1: sum_hi += p_hi, capture rounding residue into hi_err
        Scalar hi_new, hi_err;
        add_with_residual(sum_hi, p_hi, hi_new, hi_err);
        sum_hi = hi_new;

        // Step 2: t = hi_err + p_lo, capture residue t_err
        Scalar t, t_err;
        add_with_residual(hi_err, p_lo, t, t_err);

        // Step 3: sum_lo += t, capture rounding residue into lo_err
        Scalar lo_new, lo_err;
        add_with_residual(sum_lo, t, lo_new, lo_err);
        sum_lo = lo_new;

        // Step 4: push leftovers to sum_ll
        sum_ll += (lo_err + t_err);
    }

    // Update low channel with compensation into sum_ll
    template<typename Scalar>
    EIGEN_STRONG_INLINE static void accumulate_lo3(Scalar &sum_lo, Scalar &sum_ll, const Scalar &q_hi, const Scalar &q_lo) {
        Scalar lo_new, lo_err;
        add_with_residual(sum_lo, q_hi, lo_new, lo_err);
        sum_lo  = lo_new;
        sum_ll += (lo_err + q_lo);
    }

    // Collapse (hi, lo, ll) into a renormalized x2 pair (out_hi, out_lo)
    template<typename Scalar>
    EIGEN_STRONG_INLINE static void renorm3_to_x2(const Scalar &hi, const Scalar &lo, const Scalar &ll, Scalar &out_hi, Scalar &out_lo) {
        // Merge lo + ll
        Scalar lo1, lo1_err;
        add_with_residual(lo, ll, lo1, lo1_err);

        // Merge hi + lo1
        Scalar hi1, hi1_err;
        add_with_residual(hi, lo1, hi1, hi1_err);

        // Combine remaining residues
        Scalar e, e_err;
        add_with_residual(hi1_err, lo1_err, e, e_err);

        // Final pack into (out_hi, out_lo)
        add_with_residual(hi1, e, out_hi, out_lo);
        out_lo += e_err;

        // One final renorm for good measure
        add_with_residual(out_hi, out_lo, out_hi, out_lo);
    }

    template<typename Scalar>
    EIGEN_STRONG_INLINE static void renorm(Scalar *__restrict hp, Scalar *__restrict lp, const Eigen::Index size) {
#pragma omp parallel for schedule(static)
        for(Eigen::Index idx = 0; idx < size; ++idx) {
            Scalar s, e;
            x2_detail::add_with_residual(hp[idx], lp[idx], s, e);
            hp[idx] = s;
            lp[idx] = e;
        }
    }

    template<int rank>
    EIGEN_STRONG_INLINE static std::array<Eigen::Index, rank> to_array_dims(const Eigen::DSizes<Eigen::Index, rank> &d) {
        std::array<Eigen::Index, rank> out{};
        for(int i = 0; i < rank; ++i) out[static_cast<size_t>(i)] = d[static_cast<size_t>(i)];
        return out;
    }

    template<int rank>
    EIGEN_STRONG_INLINE static std::array<Eigen::Index, rank> colmajor_strides(const std::array<Eigen::Index, rank> &dims) {
        std::array<Eigen::Index, rank> s{};
        s[0] = 1;
        for(int ax = 1; ax < rank; ++ax) s[static_cast<size_t>(ax)] = s[static_cast<size_t>(ax - 1)] * dims[static_cast<size_t>(ax - 1)];
        return s;
    }

    template<int rank>
    EIGEN_STRONG_INLINE static void fill_axes(std::array<int, rank> &dst, int &count, std::initializer_list<int> axes) {
        assert(static_cast<int>(axes.size()) <= rank);
        count = 0;
        for(int ax : axes) {
            dst[static_cast<size_t>(count)] = ax;
            ++count;
        }
    }

}