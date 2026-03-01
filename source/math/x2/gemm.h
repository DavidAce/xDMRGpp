#pragma once
#include "gemm_kernel.h"
#include "Matrix.h"
#include "Tensor.h"
#include "view.h"
#include <cassert>
#include <limits>

namespace x2 {
    template<typename Scalar>
    static inline Scalar x2_split_hi(const Scalar &a, const decltype(std::real(a)) split) {
        using Real = decltype(std::real(a));
        if constexpr(std::is_floating_point_v<Scalar>) {
            auto c  = split * a;
            auto ah = c - (c - a);
            return ah;
        } else {
            Real ar = std::real(a), ai = std::imag(a);
            Real cr  = split * ar;
            Real ci  = split * ai;
            Real ahr = cr - (cr - ar);
            Real ahi = ci - (ci - ai);
            return Scalar{ahr, ahi};
        }
    }

    template<typename Scalar>
    static inline Scalar x2_split_lo(const Scalar &a, const Scalar &ah) {
        return a - ah;
    }

    template<typename Scalar>
    void x2_split(Matrix<Scalar> &A_dd, const typename Matrix<Scalar>::MatrixType &A) {
        using Real = typename Matrix<Scalar>::RealScalar;

        // For double: digits=53, half=27 gives 2^27+1
        constexpr int half_bits = (std::numeric_limits<Real>::digits + 1) / 2;
        const Real    split     = std::ldexp(Real{1}, half_bits) + Real{1};

        A_dd.resize(A.rows(), A.cols());
        A_dd.hi = A.unaryExpr([&](const Scalar &x) { return x2_split_hi(x, split); });
        A_dd.lo = (A - A_dd.hi()); // exact residual in fp64 arithmetic
    }


    template<typename Scalar, int rankC, int rankA, int rankB>
    void gemm_x2(X2TensorAsMatrixView<Scalar, rankC> &C_out, const ConstX2TensorAsMatrixView<Scalar, rankA> &A_in,
                 const ConstX2TensorAsMatrixView<Scalar, rankB> &B_in) {
        assert(C_out.rows() == A_in.rows());
        assert(C_out.cols() == B_in.cols());
        assert(B_in.rows() == A_in.cols());
        gemm_x2_kernel_fused_packed_Cx2_Ax2_Bx2(C_out.hi, C_out.lo, A_in.hi, A_in.lo, B_in.hi, B_in.lo);
    }

    template<typename Scalar, int rankC, int rankA, int rankB>
    void gemm_x2(X2TensorAsMatrixView<Scalar, rankC> &C_out, const ConstTensorAsMatrixView<Scalar, rankA> &A_in,
                 const ConstX2TensorAsMatrixView<Scalar, rankB> &B_in) {
        assert(C_out.rows() == A_in.rows());
        assert(C_out.cols() == B_in.cols());
        assert(B_in.rows() == A_in.cols());
        gemm_x2_kernel_fused_packed_Cx2_A_Bx2(C_out.hi, C_out.lo, A_in, B_in.hi, B_in.lo);
    }

    template<typename Scalar, int rankC, int rankA, int rankB>
    void gemm_x2(X2TensorAsMatrixView<Scalar, rankC> &C_out, const ConstX2TensorAsMatrixView<Scalar, rankA> &A_in,
                 const ConstTensorAsMatrixView<Scalar, rankB> &B_in) {
        assert(C_out.rows() == A_in.rows());
        assert(C_out.cols() == B_in.cols());
        assert(B_in.rows() == A_in.cols());
        gemm_x2_kernel_fused_packed_Cx2_Ax2_B(C_out.hi, C_out.lo, A_in.hi, A_in.lo, B_in);
    }

    template<typename Scalar, int rankC, int rankA, int rankB>
    void gemm_x2(X2TensorAsMatrixView<Scalar, rankC> &C_out, const ConstTensorAsMatrixView<Scalar, rankA> &A_in,
                 const ConstTensorAsMatrixView<Scalar, rankB> &B_in) {
        assert(C_out.rows() == A_in.rows());
        assert(C_out.cols() == B_in.cols());
        assert(B_in.rows() == A_in.cols());
        gemm_x2_kernel_fused_packed_Cx2_A_B(C_out.hi, C_out.lo, A_in, B_in);
    }
}