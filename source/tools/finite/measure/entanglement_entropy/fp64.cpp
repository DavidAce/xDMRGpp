#include "../entanglement_entropy.impl.h"

using Scalar = fp64;
using Real   = fp64;

/* clang-format off */

template Real  tools::finite::measure::entanglement_entropy(const Eigen::Tensor<Scalar, 1> &);

template Real  tools::finite::measure::entanglement_entropy_current(const StateFinite<Scalar> &);

template Real  tools::finite::measure::entanglement_entropy_midchain(const StateFinite<Scalar> &);

template std::vector<Real>  tools::finite::measure::entanglement_entropies(const StateFinite<Scalar> &);

template Real  tools::finite::measure::entanglement_entropy_log2(const StateFinite<Scalar> &, size_t);

template std::vector<Real>  tools::finite::measure::entanglement_entropies_log2(const StateFinite<Scalar> &);

template std::vector<Real>  tools::finite::measure::renyi_entropies(const StateFinite<Scalar> &, double);

template Real  tools::finite::measure::renyi_entropy_midchain(const StateFinite<Scalar> &, double);

/* clang-format on */



