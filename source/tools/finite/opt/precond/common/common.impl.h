#pragma once
#include "../common.h"

namespace settings {
#if defined(NDEBUG)
    inline constexpr bool debug_common = false;
#else
    inline constexpr bool debug_common = true;
#endif
}

using namespace tools::finite::opt::precond::common;

template<typename Scalar>
Ten<Scalar, 3> tools::finite::opt::precond::common::transform_env(const Ten<Scalar, 3> &blk, const Mat<Scalar> &M_transf, Real<Scalar> kappa) {
    // Sandwich each virtual-bond (beta) slice of env between the transformation operators M
    // auto &threads  = tenx::threads::get();
    auto M        = tenx::TensorCast(M_transf);
    auto Mh       = tenx::TensorCast(M_transf.adjoint());
    auto bc_dim   = std::array<Eigen::Index, 3>{M.dimension(0), M.dimension(0), blk.dimension(2)};
    auto bc_env   = Ten<Scalar, 3>(bc_dim);
    auto dim_beta = bc_dim[2];
    for(Eigen::Index b = 0; b < dim_beta; ++b) {
        auto env_slice = blk.chip(b, 2);
        // bc_env.chip(b, 2) = Mh.contract(env_slice, tenx::idx({1}, {0})).contract(M, tenx::idx({1}, {0}));
        bc_env.chip(b, 2) = M.contract(env_slice, tenx::idx({1}, {0})).contract(Mh, tenx::idx({1}, {0}));
    }
    return bc_env * bc_env.constant(kappa);
}

template<typename Scalar>
Ten<Scalar, 3> tools::finite::opt::precond::common::transform_tensor(const Ten<Scalar, 3> &psi, const Mat<Scalar> &ML, const Mat<Scalar> &MR) {
    auto  MLm       = tenx::TensorMap(ML);
    auto  MRm       = tenx::TensorMap(MR);
    auto &threads   = tenx::threads::get();
    auto  init_dims = psi.dimensions();
    auto  bc_psi    = Ten<Scalar, 3>();
    bc_psi.resize(init_dims[0], MLm.dimension(0), MRm.dimension(0));
    bc_psi.device(*threads->dev) = MLm.contract(psi, tenx::idx({1}, {1})).contract(MRm, tenx::idx({2}, {0})).shuffle(std::array<Eigen::Index, 3>{1, 0, 2});
    return bc_psi;
}

template<typename Scalar>
Vec<Scalar> tools::finite::opt::precond::common::transform_vector(const Vec<Scalar> &psi, std::array<Eigen::Index, 3> psi_dims, const Mat<Scalar> &ML,
                                                                  const Mat<Scalar> &MR) {
    Ten<Scalar, 3> psi_tensor             = tenx::TensorCast(psi, psi_dims);
    Ten<Scalar, 3> transformed_psi_tensor = transform_tensor(psi_tensor, ML, MR);
    return tenx::VectorCast(transformed_psi_tensor);
}

template<typename Scalar>
Mat<Scalar> tools::finite::opt::precond::common::transform_matrix(const Mat<Scalar> &V, const std::array<Eigen::Index, 3> psi_shape, const Mat<Scalar> &ML,
                                                                  const Mat<Scalar> &MR) {
    // V has many psi (one per column) with dimensions of psi_shape
    if(ML.size() == 0 or MR.size() == 0) return V;
    auto  MLm     = tenx::TensorMap(ML);
    auto  MRm     = tenx::TensorMap(MR);
    auto &threads = tenx::threads::get();

    auto psi_shape_new = std::array<Eigen::Index, 3>{psi_shape[0], ML.rows(), MR.rows()};
    auto V_rows_new    = psi_shape_new[0] * psi_shape_new[1] * psi_shape_new[2];
    auto V_cols_new    = V.cols(); // No change
    auto ML_V_MR       = Mat<Scalar>(V_rows_new, V_cols_new);

    for(Eigen::Index c = 0; c < V.cols(); ++c) {
        auto psi_old = Eigen::TensorMap<const Ten<Scalar, 3>>(V.col(c).data(), psi_shape);
        auto psi_new = Eigen::TensorMap<Ten<Scalar, 3>>(ML_V_MR.col(c).data(), psi_shape_new);

        psi_new.device(*threads->dev) =
            MLm.contract(psi_old, tenx::idx({1}, {1})).contract(MRm, tenx::idx({2}, {0})).shuffle(std::array<Eigen::Index, 3>{1, 0, 2});
        // ML_V_MR.col(c).normalize();
    }
    return ML_V_MR;
}
