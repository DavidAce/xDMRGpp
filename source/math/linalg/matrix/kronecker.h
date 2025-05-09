#pragma once
#include "../common.h"
#include "unsupported/Eigen/KroneckerProduct"
#include <Eigen/Core>

namespace linalg::matrix {
    template<auto N, auto M>
    constexpr auto multiply() {
        if constexpr(N == -1 or M == -1)
            return -1;
        else
            return N * M;
    }

    template<typename DA, typename DB>
    using KroneckerResultType = Eigen::Matrix<cplx_or_real_t<typename DA::Scalar, typename DB::Scalar>, multiply<DA::RowsAtCompileTime, DB::RowsAtCompileTime>(),
                                              multiply<DA::ColsAtCompileTime, DB::ColsAtCompileTime>(), Eigen::ColMajor>;

    template<typename DerivedA, typename DerivedB>
    KroneckerResultType<DerivedA, DerivedB> kronecker(const Eigen::PlainObjectBase<DerivedA> &A, const Eigen::PlainObjectBase<DerivedB> &B, bool mirror) {
        if(mirror)
            return Eigen::kroneckerProduct(B.derived(), A.derived());
        else
            return Eigen::kroneckerProduct(A.derived(), B.derived());
    }

    template<typename DerivedA, typename DerivedB>
    auto kronecker(const Eigen::EigenBase<DerivedA> &A, const Eigen::EigenBase<DerivedB> &B, bool mirror = false) {
        if constexpr(is_PlainObject<DerivedA>::value and is_PlainObject<DerivedB>::value)
            return kronecker(A, B, mirror);
        else { return kronecker(A.derived().eval(), B.derived().eval(), mirror); }
    }

}
