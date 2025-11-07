#include "common.impl.h"

using Scalar = cx64;

/* clang-format off */
using namespace tools::finite::opt::precond::common;

template Ten<Scalar,3> tools::finite::opt::precond::common::transform_env(const Ten<Scalar,3> &blk, const Mat<Scalar> &M_transf, Real<Scalar> kappa);

template Ten<Scalar,3> tools::finite::opt::precond::common::transform_tensor(const Ten<Scalar,3> &psi, const Mat<Scalar> &ML, const Mat<Scalar> &MR);

template Vec<Scalar>  tools::finite::opt::precond::common::transform_vector(const Vec<Scalar> &psi, std::array<Eigen::Index, 3> psi_dims, const Mat<Scalar> &ML, const Mat<Scalar> &MR);

template Mat<Scalar>  tools::finite::opt::precond::common::transform_matrix(const Mat<Scalar> &V, const std::array<Eigen::Index, 3> psi_shape, const Mat<Scalar> &ML, const Mat<Scalar> &MR);

