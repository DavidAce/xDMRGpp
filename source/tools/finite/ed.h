#pragma once

#include "math/tenx/fwd_decl.h"

template<typename Scalar> class StateFinite;
template<typename Scalar> class ModelFinite;
template<typename Scalar> class EdgesFinite;
template<typename Scalar> class TensorsFinite;
class AlgorithmStatus;
class ur;
namespace eig {
    class solver;
}

namespace tools::finite::ed {
    template<typename Scalar> extern StateFinite<Scalar> find_exact_state(const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status);
}