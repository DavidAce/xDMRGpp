#pragma once
#include "../../env.h"
#include "../assertions.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "../expansion_terms.h"
#include "math/tenx.h"
#include <Eigen/Core>
#include <Eigen/QR>

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
        if(rep > 2) tools::log->info("dgks: rep = {}: maxProjX = {:.5e} | threshold {:.5e}", rep, fp(maxProjX), fp(threshold));
    }
    assert_allfinite(X);
    assert_allfinite(Y);
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>>
    tools::finite::env::internal::get_expansion_terms_MP_N0(const Eigen::Tensor<T, 3>                     &M,   // Gets expanded
                                                            const Eigen::Tensor<T, 3>                     &N,   // Gets padded
                                                            const Eigen::Tensor<T, 3>                     &P1,  //
                                                            const Eigen::Tensor<T, 3>                     &P2,  //
                                                            [[maybe_unused]] const BondExpansionResult<T> &res, //
                                                            [[maybe_unused]] const Eigen::Index            bond_max) {
    /*
        We form M_P = [M | P] by concatenating along dimension 2.
        In matrix-language, the columns of P are added as new columns to M.
        The columns of M and P are normalized (M is bare).

        We form P by taking P1=H1*M and P2=H2*M, where the effective operators act only to the left of the M.
        We then subtract from P1 and P2 their projections against M and each other. After stacking P = [P1 | P2],
        we sort the columns of P with a column-pivoting householder QR.
      */
    assert(M.dimension(2) == N.dimension(1));
    assert(P1.size() == 0 or P1.dimension(1) == M.dimension(1));
    assert(P1.size() == 0 or P1.dimension(0) == M.dimension(0));
    assert(P2.size() == 0 or P2.dimension(1) == M.dimension(1));
    assert(P2.size() == 0 or P2.dimension(0) == M.dimension(0));
    using MatrixT    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = typename MatrixT::RealScalar;
    using VectorR    = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    assert_orthonormal<2>(M); // "
    assert_orthonormal<1>(N);

    auto max_cols_keep = std::min<Eigen::Index>({M.dimension(0) * M.dimension(1), N.dimension(0) * N.dimension(2)});
    max_cols_keep      = std::min(max_cols_keep, bond_max);
    if(max_cols_keep <= M.dimension(2)) {
        // We can't add more columns beyond the ones that are already fixed
        return {M, N};
    }

    // Now let's start calculating residuals.
    const auto M_matrix = tenx::MatrixMap(M, M.dimension(0) * M.dimension(1), M.dimension(2));
    MatrixT    P_matrix;
    if(P1.size() > 0) {
        P_matrix              = tenx::MatrixMap(P1, P1.dimension(0) * P1.dimension(1), P1.dimension(2));
        RealScalar maxcolnorm = P_matrix.colwise().norm().maxCoeff();
        P_matrix /= maxcolnorm;
    }
    if(P2.size() > 0) {
        P_matrix.conservativeResize(M_matrix.rows(), P_matrix.cols() + P2.dimension(2));
        P_matrix.rightCols(P2.dimension(2)) = tenx::MatrixMap(P2, P2.dimension(0) * P2.dimension(1), P2.dimension(2));
        RealScalar maxcolnorm               = P_matrix.rightCols(P2.dimension(2)).colwise().norm().maxCoeff();
        P_matrix.rightCols(P2.dimension(2)) /= maxcolnorm;
    }
    VectorR P_norms = P_matrix.colwise().norm();
    // tools::log->info("P norms: {::.3e}", fv(P_norms));
    orthonormalize_dgks(M_matrix, P_matrix);
    RealScalar maxPnorm = std::max<RealScalar>(RealScalar{1}, P_matrix.colwise().norm().maxCoeff());

    Eigen::ColPivHouseholderQR<MatrixT> cpqr(P_matrix);
    auto                                max_cols_keep_P = std::min<Eigen::Index>(cpqr.rank(), std::max<Eigen::Index>(0, max_cols_keep - M.dimension(2)));
    P_matrix                                            = cpqr.householderQ().setLength(cpqr.rank()) * MatrixT::Identity(P_matrix.rows(), max_cols_keep_P);

    Eigen::Index dim0 = M.dimension(0);
    Eigen::Index dim1 = M.dimension(1);
    Eigen::Index dim2 = M_matrix.cols() + P_matrix.cols();

    auto extM = std::array<long, 3>{M.dimension(0), M.dimension(1), M_matrix.cols()};
    auto extP = std::array<long, 3>{M.dimension(0), M.dimension(1), P_matrix.cols()};
    auto offM = std::array<long, 3>{0, 0, 0};
    auto offP = std::array<long, 3>{0, 0, M_matrix.cols()};

    auto M_P = Eigen::Tensor<T, 3>(dim0, dim1, dim2);

    if(extM[2] > 0) M_P.slice(offM, extM) = M;
    if(extP[2] > 0) M_P.slice(offP, extP) = tenx::TensorMap(P_matrix, extP);

    auto N_0 = Eigen::Tensor<T, 3>(N.dimension(0), M_P.dimension(2), N.dimension(2));
    N_0.setZero();
    auto extN_0                              = std::array<long, 3>{N.dimension(0), std::min(N.dimension(1), M_P.dimension(2)), N.dimension(2)};
    N_0.slice(tenx::array3{0, 0, 0}, extN_0) = N.slice(tenx::array3{0, 0, 0}, extN_0); // Copy N into N_0

    // Sanity checks
    auto M_P_matrix = tenx::MatrixMap(M_P, M_P.dimension(0) * M_P.dimension(1), M_P.dimension(2));

    assert_allfinite(M_P_matrix);
    assert_orthonormal<2>(M_P, std::numeric_limits<RealScalar>::epsilon() * maxPnorm);

    return {M_P, N_0};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> tools::finite::env::internal::get_expansion_terms_N0_MP(const Eigen::Tensor<T, 3>    &N,  // Gets padded
                                                                                                            const Eigen::Tensor<T, 3>    &M,  // Gets expanded
                                                                                                            const Eigen::Tensor<T, 3>    &P1, //
                                                                                                            const Eigen::Tensor<T, 3>    &P2, //
                                                                                                            const BondExpansionResult<T> &res,
                                                                                                            const Eigen::Index            bond_max) {
    constexpr auto shf = std::array<long, 3>{0, 2, 1};
    assert(N.dimension(2) == M.dimension(1));
    assert_orthonormal<2>(N); // N is an "A"
    assert_orthonormal<1>(M); // M is a "B"

    auto N_           = Eigen::Tensor<T, 3>(N.shuffle(shf));
    auto M_           = Eigen::Tensor<T, 3>(M.shuffle(shf));
    auto P1_          = Eigen::Tensor<T, 3>(P1.shuffle(shf));
    auto P2_          = Eigen::Tensor<T, 3>(P2.shuffle(shf));
    auto [MP, N0] = get_expansion_terms_MP_N0(M_, N_, P1_, P2_, res, bond_max);
    return {N0.shuffle(shf), MP.shuffle(shf)};
}
