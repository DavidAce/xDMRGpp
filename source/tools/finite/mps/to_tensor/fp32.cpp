#include "impl.h"

using Scalar = fp32;
using CalcType = fp32;
/* clang-format off */

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CalcType>(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name);

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CalcType>(const StateFinite<Scalar> &state);