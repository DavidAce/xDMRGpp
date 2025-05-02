#include "../opdm.impl.h"

using Scalar = cx128;

template Eigen::Tensor<Scalar, 2>  tools::finite::measure::opdm(const StateFinite<Scalar> &state);

template Eigen::Tensor<RealScalar<Scalar>, 1>  tools::finite::measure::opdm_spectrum(const StateFinite<Scalar> &state);
