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

namespace tools::finite::opt {
    template<typename Scalar> class opt_mps;
    struct OptMeta;
    template<typename Scalar> extern opt_mps<Scalar> get_opt_initial_mps(const TensorsFinite<Scalar> &tensors, const OptMeta &meta);
    template<typename Scalar> extern opt_mps<Scalar> get_updated_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_state,
                                                                       const AlgorithmStatus &status, OptMeta &meta);
    template<typename Scalar> extern opt_mps<Scalar> find_ground_state(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_state,
                                                                       const AlgorithmStatus &status, OptMeta &meta);
}
