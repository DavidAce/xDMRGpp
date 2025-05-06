#include "../svd.impl.h"

using Scalar = fp32;

template std::tuple<svd::MatrixType<Scalar>,  svd::VectorType<Scalar>,  svd::MatrixType<Scalar>>  svd::solver::do_svd_ptr(const Scalar *, long, long, const svd::config &);

template void svd::solver::print_matrix<Scalar>(const Scalar *vec_ptr, long rows, long cols, std::string_view tag, long dec) const;

template void svd::solver::print_vector<Scalar>(const Scalar *vec_ptr, long size, std::string_view tag, long dec) const;

template std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 2>>   svd::solver::split_mpo_l2r(const Eigen::Tensor<Scalar, 4> &mpo, const svd::config &svd_cfg);

template std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 4>>   svd::solver::split_mpo_r2l(const Eigen::Tensor<Scalar, 4> &mpo, const svd::config &svd_cfg);
