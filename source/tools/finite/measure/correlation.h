#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
template<typename Scalar>
class StateFinite;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar>
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    template<typename Scalar> [[nodiscard]] extern Scalar                     correlation            (const StateFinite<Scalar> & state, const Eigen::Tensor<Scalar, 2> &op1, const Eigen::Tensor<Scalar,2> &op2, long pos1, long pos2);
    template<typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar, 2>   correlation_matrix     (const StateFinite<Scalar> & state, const Eigen::Tensor<Scalar, 2> &op1, const Eigen::Tensor<Scalar,2> &op2);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>         structure_factor       (const StateFinite<Scalar> & state, const Eigen::Tensor<Scalar, 2> &correlation_matrix);
    /* clang-format on */

}