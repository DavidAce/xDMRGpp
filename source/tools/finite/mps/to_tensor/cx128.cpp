#include "impl.h"

using Scalar = cx128;
using RealType = fp128;
using CplxType = cx128;
/* clang-format off */

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CplxType>(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name);
template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<RealType>(const std::vector<std::unique_ptr<MpsSite<Scalar>>> &mps_sites, std::string_view name);

template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<CplxType>(const StateFinite<Scalar> &state);
template Eigen::Tensor<Scalar, 1> tools::finite::mps::mps2tensor<RealType>(const StateFinite<Scalar> &state);