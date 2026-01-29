#pragma once
#include "x2_kernel.h"
#include "math/tenx.h"
#include <cassert>
#include <general/sfinae.h>

template<typename Scalar, int Rank>
struct TensorX2 {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using TensorType = Eigen::Tensor<Scalar, Rank>;
    TensorType hi, lo;

    TensorX2() = default;
    TensorX2(const Eigen::array<Eigen::Index, Rank> &dims) {
        hi.resize(dims);
        lo.resize(dims);
    }
    template<typename... Dims>
    requires(std::integral<Dims> && ...)
    TensorX2(Dims... dims) {
        static_assert(sizeof...(Dims) == Rank);
        hi.resize(dims...);
        lo.resize(dims...);
    }
    TensorX2(const Eigen::TensorRef<TensorType> &A) {
        hi = A;
        lo.resize(A.dimensions());
        lo.setZero();
    }
    void resize(const Eigen::array<Eigen::Index, Rank> &dims) {
        hi.resize(dims);
        lo.resize(dims);
    }
    template<typename... Dims>
    void resize(Dims... dims) {
        static_assert(sizeof...(Dims) == Rank);
        hi.resize(dims...);
        lo.resize(dims...);
    }

    Eigen::Index size() const { return hi.size(); }

    void setZero() {
        hi.setZero();
        lo.setZero();
    }
    bool allFinite() const {
        for(Eigen::Index i = 0; i < hi.size(); ++i) {
            Scalar elem = hi.data()[i];
            if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
                if(!std::isfinite(std::real(elem)) or !std::isfinite(std::imag(elem))) return false;
            } else {
                if(!std::isfinite(elem)) return false;
            }
        }
        for(Eigen::Index i = 0; i < lo.size(); ++i) {
            Scalar elem = lo.data()[i];
            if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
                if(!std::isfinite(std::real(elem)) or !std::isfinite(std::imag(elem))) return false;
            } else {
                if(!std::isfinite(elem)) return false;
            }
        }
        return true;
    }

    // Final downcast (do not use for intermediates)
    TensorType to_TensorType() const { return (hi + lo); }

    // Cheap renormalization: enforce hi carries the leading bits
    void renorm() {
        // two_sum per entry: (hi, lo) := hi + lo exactly split back into hi+lo
        // Vectorized-ish form:
        TensorType s = (hi + lo);
        TensorType e = (lo - (s - hi));
        hi.swap(s);
        lo.swap(e);
        assert(allFinite());
    }

    void shuffle(const Eigen::array<int, Rank> &perm) {
        hi = Eigen::Tensor<Scalar, Rank>(hi.shuffle(perm));
        lo = Eigen::Tensor<Scalar, Rank>(lo.shuffle(perm));
        assert(allFinite());
    }

    // Frobenius norm of (hi + lo)
    RealScalar norm() const {
        assert(hi.size() == lo.size());
        const Eigen::Index n = hi.size();
        if(n == 0) return RealScalar{0};

        const Scalar *hi_ptr = hi.data();
        const Scalar *lo_ptr = lo.data();

        using RealAcc = std::conditional_t<sizeof(RealScalar) <= sizeof(double), long double, fp128>;
        RealAcc sum   = RealAcc{0};

#pragma omp parallel for reduction(+ : sum) schedule(static)
        for(Eigen::Index i = 0; i < n; ++i) {
            const Scalar a = hi_ptr[i];
            const Scalar b = lo_ptr[i];

            if constexpr(tenx::sfinae::is_std_complex_v<Scalar>) {
                // |a|^2 + 2 Re(conj(a)b) + |b|^2
                const RealAcc ar = static_cast<RealAcc>(std::real(a));
                const RealAcc ai = static_cast<RealAcc>(std::imag(a));
                const RealAcc br = static_cast<RealAcc>(std::real(b));
                const RealAcc bi = static_cast<RealAcc>(std::imag(b));

                const RealAcc aa = ar * ar + ai * ai;
                const RealAcc bb = br * br + bi * bi;
                const RealAcc ab = ar * br + ai * bi; // Re(conj(a)*b)

                sum += aa + RealAcc{2} * ab + bb;
            } else {
                const RealAcc ar = static_cast<RealAcc>(a);
                const RealAcc br = static_cast<RealAcc>(b);

                const RealAcc aa = ar * ar;
                const RealAcc bb = br * br;
                const RealAcc ab = ar * br;

                sum += aa + RealAcc{2} * ab + bb;
            }
        }

        if(sum < RealAcc{0}) sum = RealAcc{0}; // guard tiny negative from rounding
        return static_cast<RealScalar>(std::sqrt(sum));
    }
};

template<typename Scalar, int Rank, typename Perm>
void shuffle_inplace(TensorX2<Scalar, Rank> &T, const Perm &perm) {
    // Avoid aliasing issues by shuffling into temporaries and swapping
    auto hi_new = Eigen::Tensor<Scalar, Rank>(T.hi.shuffle(perm));
    auto lo_new = Eigen::Tensor<Scalar, Rank>(T.lo.shuffle(perm));
    T.hi        = std::move(hi_new);
    T.lo        = std::move(lo_new);
}

template<typename Scalar>
struct MatrixX2 {
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    MatrixType hi;
    MatrixType lo;

    MatrixX2() = default;

    MatrixX2(Eigen::Index rows, Eigen::Index cols) {
        hi.setZero(rows, cols);
        lo.setZero(rows, cols);
    }

    MatrixX2(const Eigen::Ref<const MatrixType> &A) {
        hi = A;
        lo.setZero(A.rows(), A.cols());
    }

    Eigen::Index rows() const { return hi.rows(); }
    Eigen::Index cols() const { return hi.cols(); }

    void resize(Eigen::Index rows, Eigen::Index cols) {
        hi.resize(rows, cols);
        lo.resize(rows, cols);
    }

    void setZero() {
        hi.setZero();
        lo.setZero();
    }

    bool allFinite() const { return hi.allFinite() && lo.allFinite(); }

    // Final downcast (do not use for intermediates)
    MatrixType to_MatrixType() const { return (hi + lo); }

    // Cheap renormalization: enforce hi carries the leading bits
    void renorm() {
        // two_sum per entry: (hi, lo) := hi + lo exactly split back into hi+lo
        // Vectorized-ish form:
        MatrixType s = (hi + lo);
        MatrixType e = (lo - (s - hi));
        hi.swap(s);
        lo.swap(e);
    }
};

template<typename Scalar>
struct MatrixX2Map {
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<MatrixType> hi;
    Eigen::Map<MatrixType> lo;

    MatrixX2Map(Scalar *hi_ptr, Scalar *lo_ptr, Eigen::Index rows, Eigen::Index cols) : hi(hi_ptr, rows, cols), lo(lo_ptr, rows, cols) {}

    // Cheap renormalization: enforce hi carries the leading bits
    void renorm() {
        // two_sum per entry: (hi, lo) := hi + lo exactly split back into hi+lo
        // Vectorized-ish form:
        MatrixType s = (hi + lo);
        MatrixType e = (lo - (s - hi));
        hi           = s;
        lo           = e;
    }
    Eigen::Index rows() const {
        assert(hi.rows() == lo.rows());
        return hi.rows();
    }
    Eigen::Index cols() const {
        assert(hi.cols() == lo.cols());
        return hi.cols();
    }
};

template<typename Scalar>
struct ConstMatrixX2Map {
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const MatrixType> hi;
    Eigen::Map<const MatrixType> lo;

    ConstMatrixX2Map(const Scalar *hi_ptr, const Scalar *lo_ptr, Eigen::Index rows, Eigen::Index cols) : hi(hi_ptr, rows, cols), lo(lo_ptr, rows, cols) {}
    Eigen::Index rows() const {
        assert(hi.rows() == lo.rows());
        return hi.rows();
    }
    Eigen::Index cols() const {
        assert(hi.cols() == lo.cols());
        return hi.cols();
    }
};

template<typename Scalar>
static inline Scalar x2_split_hi(const Scalar &a, const decltype(std::real(a)) split) {
    using Real = decltype(std::real(a));
    if constexpr(tenx::sfinae::is_std_complex_v<Scalar>) {
        Real ar = std::real(a), ai = std::imag(a);
        Real cr  = split * ar;
        Real ci  = split * ai;
        Real ahr = cr - (cr - ar);
        Real ahi = ci - (ci - ai);
        return Scalar{ahr, ahi};
    } else {
        auto c  = split * a;
        auto ah = c - (c - a);
        return ah;
    }
}

template<typename Scalar>
static inline Scalar x2_split_lo(const Scalar &a, const Scalar &ah) {
    return a - ah;
}

template<typename Scalar>
void x2_split(MatrixX2<Scalar> &A_dd, const typename MatrixX2<Scalar>::MatrixType &A) {
    using Real = typename MatrixX2<Scalar>::RealScalar;

    // For double: digits=53, half=27 gives 2^27+1
    constexpr int half_bits = (std::numeric_limits<Real>::digits + 1) / 2;
    const Real    split     = std::ldexp(Real{1}, half_bits) + Real{1};

    A_dd.resize(A.rows(), A.cols());
    A_dd.hi = A.unaryExpr([&](const Scalar &x) { return x2_split_hi(x, split); });
    A_dd.lo = (A - A_dd.hi); // exact residual in fp64 arithmetic
}

template<typename Scalar>
void gemm_x2(MatrixX2<Scalar> &C_out, const MatrixX2<Scalar> &A_in, const MatrixX2<Scalar> &B_in) {
    const Eigen::Index m = A_in.rows();
    const Eigen::Index k = A_in.cols();
    const Eigen::Index n = B_in.cols();
    assert(B_in.rows() == k);
    assert(A_in.allFinite());
    assert(B_in.allFinite());

    C_out.resize(m, n);
    gemm_x2_kernel(C_out.hi, C_out.lo, A_in.hi, B_in.hi, /*IB=*/64, /*JB=*/8, /*BK=*/128);
    // P0 = Ah*Bh
    // C_out.hi.noalias() = A_in.hi * B_in.hi;
    // lo accumulates the cross terms
    C_out.lo.noalias() += A_in.hi * B_in.lo;
    C_out.lo.noalias() += A_in.lo * B_in.hi;
    C_out.lo.noalias() += A_in.lo * B_in.lo;

    // Renormalize (i.e. redo the hi-lo split)
    C_out.renorm();
    assert(C_out.allFinite());
}

template<typename Scalar>
void gemm_x2(MatrixX2Map<Scalar> &C_out, const ConstMatrixX2Map<Scalar> &A_in, const ConstMatrixX2Map<Scalar> &B_in) {
    assert(B_in.rows() == A_in.cols());
    assert(C_out.rows() == A_in.rows());
    assert(C_out.cols() == B_in.cols());

    // P0 = Ah*Bh
    gemm_x2_kernel(C_out.hi, C_out.lo, A_in.hi, B_in.hi, /*IB=*/64, /*JB=*/8, /*BK=*/128);

    // lo accumulates the cross terms
    C_out.lo.noalias() += A_in.hi * B_in.lo;
    C_out.lo.noalias() += A_in.lo * B_in.hi;
    C_out.lo.noalias() += A_in.lo * B_in.lo;

    // Renormalize (i.e. redo the hi-lo split)
    C_out.renorm();
}

template<typename Scalar>
void gemm_x2(MatrixX2<Scalar> &C_out, const typename MatrixX2<Scalar>::MatrixType &A_in, const typename MatrixX2<Scalar>::MatrixType &B_in) {
    MatrixX2<Scalar> A_dd, B_dd;
    x2_split(A_dd, A_in);
    x2_split(B_dd, B_in);
    gemm_x2(C_out, A_dd, B_dd);
}

