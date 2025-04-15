#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include <array>
#include <Eigen/src/Core/NumTraits.h>
#include <string_view>

template<typename Scalar>
class StateFinite;
namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar>
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar> [[nodiscard]] extern std::array<RealScalar<Scalar>,3>                    spin_components        (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>                                  spin_component         (const StateFinite<Scalar> & state, const Eigen::Matrix2cd &paulimatrix);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>                                  spin_component         (const StateFinite<Scalar> & state, std::string_view axis);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>                                  spin_alignment         (const StateFinite<Scalar> & state, std::string_view axis);
    template<typename Scalar> [[nodiscard]] extern int                                                 spin_sign              (const StateFinite<Scalar> & state, std::string_view axis);
    template<typename Scalar> [[nodiscard]] extern std::array<Eigen::Tensor<RealScalar<Scalar>, 1>, 3> expectation_values_xyz (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::array<RealScalar<Scalar>, 3>                   expectation_value_xyz  (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::array<Eigen::Tensor<RealScalar<Scalar>, 2>, 3> correlation_matrix_xyz (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::array<RealScalar<Scalar>, 3>                   structure_factor_xyz   (const StateFinite<Scalar> & state);
    /* clang-format on */
}