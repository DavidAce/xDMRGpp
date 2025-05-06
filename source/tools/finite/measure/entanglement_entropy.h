#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include <vector>

template<typename Scalar>
class StateFinite;
namespace tools::finite::measure {
    template<typename Scalar>
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    /* clang-format off */
    template <typename Scalar> [[nodiscard]] RealScalar<Scalar> entanglement_entropy                                    (const Eigen::Tensor<Scalar,1> & bond);
    template <typename Scalar> [[nodiscard]] RealScalar<Scalar> entanglement_entropy_current                            (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] RealScalar<Scalar> entanglement_entropy_midchain                           (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] std::vector<RealScalar<Scalar>> entanglement_entropies                     (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] RealScalar<Scalar>              entanglement_entropy_log2                  (const StateFinite<Scalar> & state, size_t nsites);
    template <typename Scalar> [[nodiscard]] std::vector<RealScalar<Scalar>> entanglement_entropies_log2                (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] RealScalar<Scalar>              renyi_entropy_midchain                     (const StateFinite<Scalar> & state, double q);
    template <typename Scalar> [[nodiscard]] std::vector<RealScalar<Scalar>> renyi_entropies                            (const StateFinite<Scalar> & state, double q);
    /* clang-format on */

}