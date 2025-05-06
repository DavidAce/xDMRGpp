#include "../svd_eigen.impl.h"

using Scalar = cx128;
template std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>>    svd::solver::do_svd_eigen(const Scalar *, long, long) const;
