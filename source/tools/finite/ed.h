#pragma once


template<typename Scalar> class StateFinite;
template<typename Scalar> class TensorsFinite;
class AlgorithmStatus;
namespace eig {
    class solver;
}
namespace tools::finite::opt::reports {
    template<typename Scalar> struct eigs_log;
}

namespace tools::finite::ed {
    template<typename Scalar>  StateFinite<Scalar> find_exact_state(const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status,
                                                                          tools::finite::opt::reports::eigs_log<Scalar> &elog);
}