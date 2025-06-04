#include "../opdm.impl.h"

using Scalar = cx128;

template Eigen::Tensor<Scalar, 2>  tools::finite::measure::opdm(const StateFinite<Scalar> &state, ModelType model_type);
template Eigen::Tensor<Scalar, 2>  tools::finite::measure::opdm_u1_sym(const StateFinite<Scalar> &state);
template Eigen::Tensor<Scalar, 2>  tools::finite::measure::opdm_general(const StateFinite<Scalar> &state);

template Eigen::Tensor<RealScalar<Scalar>, 1>  tools::finite::measure::opdm_spectrum(const StateFinite<Scalar> &state, ModelType model_type);
