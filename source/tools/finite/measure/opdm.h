#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include <Eigen/src/Core/NumTraits.h>
template<typename Scalar>
class StateFinite;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar, 2>   opdm                             (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern Eigen::Tensor<RealScalar<Scalar>, 1> opdm_spectrum          (const StateFinite<Scalar> & state);
    /* clang-format on */

}