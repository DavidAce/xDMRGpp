#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include <Eigen/src/Core/NumTraits.h>
#include <vector>

template<typename Scalar>
class StateFinite;
namespace tools::finite::measure {
    template<typename Scalar>
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    /* clang-format off */
    template <typename Scalar> [[nodiscard]] extern RealScalar<Scalar> entanglement_entropy                                    (const Eigen::Tensor<Scalar,1> & bond);
    template <typename Scalar> [[nodiscard]] extern RealScalar<Scalar> entanglement_entropy_current                            (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] extern RealScalar<Scalar> entanglement_entropy_midchain                           (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] extern std::vector<RealScalar<Scalar>> entanglement_entropies                     (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] extern RealScalar<Scalar>              entanglement_entropy_log2                  (const StateFinite<Scalar> & state, size_t nsites);
    template <typename Scalar> [[nodiscard]] extern std::vector<RealScalar<Scalar>> entanglement_entropies_log2                (const StateFinite<Scalar> & state);
    template <typename Scalar> [[nodiscard]] extern RealScalar<Scalar>              renyi_entropy_midchain                     (const StateFinite<Scalar> & state, double q);
    template <typename Scalar> [[nodiscard]] extern std::vector<RealScalar<Scalar>> renyi_entropies                            (const StateFinite<Scalar> & state, double q);
    /* clang-format on */

}