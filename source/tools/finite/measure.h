#pragma once

#include "math/float.h"
#include "math/svd/config.h"
#include "measure/correlation.h"
#include "measure/dimensions.h"
#include "measure/entanglement_entropy.h"
#include "measure/expectation_value.h"
#include "measure/hamiltonian.h"
#include "measure/information.h"
#include "measure/number_entropy.h"
#include "measure/opdm.h"
#include "measure/residual.h"
#include "measure/spin.h"
#include "measure/truncation.h"
#include <complex>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class TensorsFinite;
template<typename Scalar>
class MpoSite;
template<typename Scalar>
class MpsSite;
class AlgorithmStatus;
template<typename Scalar>
struct MeasurementsTensorsFinite;
template<typename T>
struct env_pair;
template<typename Scalar>
class EnvEne;
template<typename Scalar>
class EnvVar;
enum class RDM;

namespace tools::finite::measure {

    /* clang-format off */
    template<typename Scalar>  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    template<typename Scalar> [[nodiscard]] extern size_t             length  (const ModelFinite<Scalar> & model);
    template<typename Scalar> [[nodiscard]] extern size_t             length  (const TensorsFinite<Scalar> & tensors);
    template<typename Scalar> [[nodiscard]] extern size_t             length  (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern size_t             length  (const EdgesFinite<Scalar> & edges);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar> norm    (const StateFinite<Scalar> & state, bool full = false);

//  [[nodiscard]]  extern double norm_fast                                   (const StateFinite & state);



}

/* clang-format on */
