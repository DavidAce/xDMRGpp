#include "impl.h"

using Scalar = cx32;
using RealType = fp32;
using CplxType = cx32;
/* clang-format off */

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CplxType>(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name);
template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<RealType>(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name);

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CplxType>(const StateFinite<Scalar> &state);
template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<RealType>(const StateFinite<Scalar> &state);