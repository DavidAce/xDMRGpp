#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
template<typename Scalar>
class StateFinite;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename T>  using RealScalar = decltype(std::real(std::declval<T>()));
    template<typename CalcType, typename Scalar, typename OpType> [[nodiscard]] extern CalcType                   correlation            (const StateFinite<Scalar> & state, const Eigen::Tensor<OpType, 2> &op1, const Eigen::Tensor<OpType,2> &op2, long pos1, long pos2);
    template<typename CalcType, typename Scalar, typename OpType> [[nodiscard]] extern Eigen::Tensor<CalcType, 2> correlation_matrix     (const StateFinite<Scalar> & state, const Eigen::Tensor<OpType, 2> &op1, const Eigen::Tensor<OpType,2> &op2);
    template<typename Scalar> [[nodiscard]]                                     extern RealScalar<Scalar>         structure_factor       (const StateFinite<Scalar> & state, const Eigen::Tensor<Scalar, 2> &correlation_matrix);
    /* clang-format on */

}