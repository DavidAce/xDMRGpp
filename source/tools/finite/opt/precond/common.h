#pragma once
#include "math/tenx.h"
#include <Eigen/Dense>

namespace tools::finite::opt::precond::common {
    template<typename Scalar> using Real           = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> using Mat            = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename Scalar> using Vec            = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar, auto rank> using Ten = Eigen::Tensor<Scalar, rank>;

    template<typename Scalar>
    Ten<Scalar, 3> transform_env(const Ten<Scalar, 3> &blk, const Mat<Scalar> &M_transf, Real<Scalar> kappa = Real<Scalar>{1});

    template<typename Scalar>
    Ten<Scalar, 3> transform_tensor(const Ten<Scalar, 3> &psi, const Mat<Scalar> &ML, const Mat<Scalar> &MR);

    template<typename Scalar>
    Vec<Scalar> transform_vector(const Vec<Scalar> &psi, std::array<Eigen::Index, 3> psi_dims, const Mat<Scalar> &ML, const Mat<Scalar> &MR);

    template<typename Scalar>
    Mat<Scalar> transform_matrix(const Mat<Scalar> &V, const std::array<Eigen::Index, 3> psi_shape, const Mat<Scalar> &ML, const Mat<Scalar> &MR);
}