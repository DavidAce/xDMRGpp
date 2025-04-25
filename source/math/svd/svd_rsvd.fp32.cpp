#include "svd_rsvd.impl.h"

using Scalar = fp32;
template std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>>    svd::solver::do_svd_rsvd(const Scalar *, long, long) const;
