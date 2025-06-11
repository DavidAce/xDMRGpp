#include "../EnvBase.impl.h"

using Scalar = cx128;
using Real   = fp128;
template class EnvBase<Scalar>;

template Eigen::Tensor<Scalar, 3> EnvBase<Scalar>::get_expansion_term<Scalar>(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const;
template Eigen::Tensor<Real, 3>   EnvBase<Scalar>::get_expansion_term<Real>(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const;
template Eigen::Tensor<Scalar, 3> EnvBase<Scalar>::get_expansion_term<Scalar>(const Eigen::Tensor<Scalar, 3> &mps, const MpoSite<Scalar> &mpo) const;
template Eigen::Tensor<Real, 3>   EnvBase<Scalar>::get_expansion_term<Real>(const Eigen::Tensor<Scalar, 3> &mps, const MpoSite<Scalar> &mpo) const;
