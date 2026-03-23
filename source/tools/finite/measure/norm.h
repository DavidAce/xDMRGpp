#pragma once

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename Scalar>
class StateFinite;

namespace tools::finite::measure {

    /* clang-format off */
  template<typename Scalar>  using RealScalar = decltype(std::real(std::declval<Scalar>()));
  template<typename Scalar> [[nodiscard]] RealScalar<Scalar>      norm_1site    (const StateFinite<Scalar> & state);
  template<typename Scalar> [[nodiscard]] RealScalar<Scalar>      norm_state    (const StateFinite<Scalar> & state);
  template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar,2> isometry_left (const StateFinite<Scalar> & state, Eigen::Index pos);
  template<typename Scalar> [[nodiscard]] Eigen::Tensor<Scalar,2> isometry_right(const StateFinite<Scalar> & state, Eigen::Index pos);

  //  [[nodiscard]]  extern double norm_fast                                   (const StateFinite & state);
}

/* clang-format on */
