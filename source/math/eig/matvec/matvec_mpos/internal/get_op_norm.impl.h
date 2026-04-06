#pragma once

#include "../../matvec_mpos.h"
#include "tools/common/contraction/contraction_policy.h"
#include <Eigen/Eigenvalues>
#include <cmath>

template<typename Scalar>
typename MatVecMPOS<Scalar>::RealScalar MatVecMPOS<Scalar>::get_op_norm(Eigen::Index max_op_norm_iters, RealScalar reltol) const {
    if(!std::isnan(op_norm_krylov) and max_op_norm_iters <= op_norm_krylov_iters) return op_norm_krylov;
    auto shape_mpo = std::array<Eigen::Index, 4>{envL_A.dimension(2), envR_A.dimension(2), shape_mps[0], shape_mps[0]};

    {
        auto h1info = tools::common::contraction::internal::get_info_h1mv();
        auto h2info = tools::common::contraction::internal::get_info_h2mv();
        if(h1info.H1_local_dims == shape_mpo) {
            op_norm_krylov = static_cast<RealScalar>(h1info.H1_local_norm);
            return op_norm_krylov;
        }
        if(h2info.H2_local_dims == shape_mpo) {
            op_norm_krylov = static_cast<RealScalar>(h2info.H2_local_norm);
            return op_norm_krylov;
        }
    }

    using Real    = RealScalar;
    using VecType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecReal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    auto h1info = SetH1MvInfo(ContractionBackend::TBLIS, shape_mpo);
    auto h2info = SetH2MvInfo(ContractionBackend::TBLIS, shape_mpo);

    Eigen::Tensor<Scalar, 3> vt(shape_mps);
    Eigen::Tensor<Scalar, 3> wt(shape_mps);
    auto                     v_map = Eigen::Map<VecType>(vt.data(), size_mps);
    auto                     w_map = Eigen::Map<VecType>(wt.data(), size_mps);

    VecType v = VecType::Random(size_mps).normalized();

    Real         lambda    = Real{0};
    Eigen::Index krylovdim = 3;
    Eigen::Index iter      = 0;
    for(iter = 0; iter < max_op_norm_iters; ++iter) {
        const Eigen::Index p = std::max<Eigen::Index>(2, krylovdim);

        MatType K(size_mps, p);
        K.col(0) = v;

        for(Eigen::Index j = 1; j < p; ++j) {
            v_map = K.col(j - 1);
            MultAx(vt.data(), wt.data());
            K.col(j) = w_map;
        }

        Eigen::HouseholderQR<MatType> qr(K);
        const MatType                 QR = qr.matrixQR().topLeftCorner(p, p);
        VecReal                       diag_abs(p);
        for(Eigen::Index j = 0; j < p; ++j) diag_abs(j) = std::abs(QR(j, j));

        const Real diag0    = (p > 0) ? diag_abs(0) : Real{0};
        const Real drop_tol = std::max(Real{1e-20f}, Real{1e-12f} * diag0);

        Eigen::Index k_eff = 0;
        for(Eigen::Index j = 0; j < p; ++j) {
            if(diag_abs(j) > drop_tol)
                ++k_eff;
            else
                break;
        }
        k_eff = std::max<Eigen::Index>(k_eff, 1);

        MatType Q = qr.householderQ() * MatType::Identity(size_mps, k_eff).eval();

        MatType W(size_mps, k_eff);
        for(Eigen::Index j = 0; j < k_eff; ++j) {
            v_map = Q.col(j);
            MultAx(vt.data(), wt.data());
            W.col(j) = w_map;
        }

        MatType T = (Q.adjoint() * W).eval();
        T         = ((T + T.adjoint()) / Real{2}).eval();

        Eigen::SelfAdjointEigenSolver<MatType> es(T);
        if(es.info() != Eigen::Success) break;

        const auto &evals = es.eigenvalues();
        const auto &evecs = es.eigenvectors();

        Eigen::Index idx = 0;
        evals.cwiseAbs().maxCoeff(&idx);

        const Scalar theta      = Scalar(evals(idx));
        const Real   lambda_new = std::abs(evals(idx));

        VecType    y      = (Q * evecs.col(idx)).eval();
        const Real y_norm = y.norm();
        if(y_norm > Real{0}) y /= y_norm;

        VecType    r      = (W * evecs.col(idx) - (Q * evecs.col(idx)) * theta).eval();
        const Real r_norm = r.norm();

        v      = std::move(y);
        lambda = lambda_new;
        if(lambda > Real{0} && r_norm < static_cast<Real>(reltol) * lambda) break;
    }
    op_norm_krylov_iters = max_op_norm_iters;
    op_norm_krylov       = lambda;
    return lambda;
}
