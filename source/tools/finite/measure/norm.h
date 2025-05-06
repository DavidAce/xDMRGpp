#pragma once

#include <complex>

template<typename Scalar>
class StateFinite;


namespace tools::finite::measure {

  /* clang-format off */
  template<typename Scalar>  using RealScalar = decltype(std::real(std::declval<Scalar>()));
  template<typename Scalar> [[nodiscard]] RealScalar<Scalar> norm    (const StateFinite<Scalar> & state, bool full = false);

  //  [[nodiscard]]  extern double norm_fast                                   (const StateFinite & state);
}

/* clang-format on */
