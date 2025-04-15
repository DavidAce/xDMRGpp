#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
template<typename Scalar>
class StateFinite;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> [[nodiscard]] extern Eigen::Tensor<cx64, 2>   opdm                   (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern Eigen::Tensor<double, 1> opdm_spectrum          (const StateFinite<Scalar> & state);
    /* clang-format on */

}