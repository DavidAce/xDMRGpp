#include "../svd_lapacke.impl.h"

using Scalar = cx32;

template std::tuple<svd::MatrixType<Scalar>, svd::VectorType<Scalar>, svd::MatrixType<Scalar>>    svd::solver::do_svd_lapacke(const Scalar *, long, long) const;
