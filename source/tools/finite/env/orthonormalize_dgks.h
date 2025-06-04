#pragma once
#include "assertions.h"
#include <Eigen/QR>
#include <Eigen/Core>
template<typename MatrixTypeX, typename MatrixTypeY>
void orthonormalize_dgks(const MatrixTypeX &X, MatrixTypeY &Y) {
    assert(X.rows() == Y.rows());
    if(X.cols() == 0 || Y.cols() == 0) return;
    static_assert(std::is_same_v<typename MatrixTypeX::Scalar, typename MatrixTypeY::Scalar>, "MatrixTypeX and MatrixTypeY must have the same scalar type");
    using Scalar     = typename MatrixTypeX::Scalar;
    using RealScalar = typename MatrixTypeX::RealScalar;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    assert_allfinite(X);
    assert_allfinite(Y);
    // DGKS clean Y against X and orthonormalize Y
    // We do not assume that X or Y are normalized!
    RealScalar threshold = std::numeric_limits<RealScalar>::epsilon() * Y.rows() * 5;
    for(int rep = 0; rep < 50; ++rep) {
        auto maxProjX = RealScalar{0};
        for(Eigen::Index col_y = 0; col_y < Y.cols(); ++col_y) {
            auto Ycol = Y.col(col_y);
            for(Eigen::Index col_x = 0; col_x < X.cols(); ++col_x) {
                auto Xcol = X.col(col_x);
                auto Xsqn = Xcol.squaredNorm();
                if(Xsqn < threshold) { continue; }
                MatrixType proj = (Xcol.adjoint() * Ycol) / Xcol.squaredNorm();
                Ycol.noalias() -= Xcol * proj;
                maxProjX = std::max(maxProjX, proj.colwise().norm().maxCoeff());
            }
        }
        if(maxProjX < threshold) break;
        if(rep > 2) tools::log->info("dgks: rep = {}: maxProjX = {:.5e} | threshold {:.5e}", rep, maxProjX, threshold);
    }
    assert_allfinite(X);
    assert_allfinite(Y);
}
