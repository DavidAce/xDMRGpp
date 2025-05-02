#pragma once
#include "math/tenx.h"
#include <vector>
template<typename Scalar>
class StateFinite;
namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>              number_entropy_current  (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>              number_entropy_midchain (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::vector<RealScalar<Scalar>> number_entropies        (const StateFinite<Scalar> & state);
    /* clang-format on */
}