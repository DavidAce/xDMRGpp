#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
template<typename Scalar>
class StateFinite;
enum class ModelType;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 2>             opdm          (const StateFinite<Scalar> & state, ModelType modelType);
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 2>             opdm_u1_sym   (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar, 2>             opdm_general  (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] Eigen::Tensor<RealScalar<Scalar>, 1> opdm_spectrum (const StateFinite<Scalar> & state, ModelType modelType);
    /* clang-format on */

}