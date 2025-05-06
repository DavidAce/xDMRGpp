#include "../MpoSite.tmpl.h"

using Scalar = cx32;
using T = fp128;

/* clang-format off */

template Eigen::Tensor<T, 4>  MpoSite<Scalar>::apply_edge_left(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeL) const;

template Eigen::Tensor<T, 4>  MpoSite<Scalar>::apply_edge_right(const Eigen::Tensor<T, 4> &mpo, const Eigen::Tensor<T, 1> &edgeR) const;

template Eigen::Tensor<T, 4>  MpoSite<Scalar>::get_parity_shifted_mpo(const Eigen::Tensor<T, 4> &mpo_build) const;

template Eigen::Tensor<T, 1>  MpoSite<Scalar>::get_MPO_edge_left(const Eigen::Tensor<T, 4> &mpo) const;

template Eigen::Tensor<T, 1>  MpoSite<Scalar>::get_MPO_edge_right(const Eigen::Tensor<T, 4> &mpo) const;

template Eigen::Tensor<T, 1>  MpoSite<Scalar>::get_MPO2_edge_left() const;

template Eigen::Tensor<T, 1>  MpoSite<Scalar>::get_MPO2_edge_right() const;

