#include "../EnvBase.impl.h"

using Scalar = fp128;
template class EnvBase<Scalar>;

template Eigen::Tensor<Scalar, 3> EnvBase<Scalar>::get_expansion_term<Scalar>(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const;
