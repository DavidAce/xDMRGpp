#pragma once
#include "../RGB.h"
#include "../StopReason.h"
#include "io/fmt_custom.h"
#include "math/eig/log.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include <Eigen/Eigenvalues>

namespace settings {
#if defined(NDEBUG)
    constexpr bool debug_RGB = false;
#else
    constexpr bool debug_RGB = true;
#endif
    constexpr bool print_q = false;
}

template<typename Scalar>
void RGB<Scalar>::delete_blocks_from_left_until_orthogonal(const Eigen::Ref<const MatrixType> X, MatrixType &Y, MatrixType &HY, Eigen::Index maxBlocks,
                                                           RealScalar threshold) {
    assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(HY.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(Y.cols() == HY.cols() && "Y and HY must have the same number of columns");
    assert(Y.rows() == HY.rows() && "Y and HY must have the same number of rows");
    assert(X.rows() == Y.rows() && "X and Y must have the same number of rows");
    assert(X.rows() == HY.rows() && "X and Y must have the same number of rows");
    const Eigen::Index n_blocks_y = Y.cols() / b;

    Eigen::Index n_blocks_new = 0;
    for(Eigen::Index blk = 0; blk < n_blocks_y; ++blk) {
        Eigen::Index n_blocks_try = n_blocks_y - blk;
        if(n_blocks_try > maxBlocks) continue;          // No point in checking orthogonality, delete
        auto Yrc       = Y.rightCols(n_blocks_try * b); //
        auto XYrc_norm = (X.adjoint() * Yrc).norm();
        if(XYrc_norm < threshold) {
            // Keep n_blocks_new
            n_blocks_new = n_blocks_try;
            break;
        }
    }
    if(n_blocks_new != n_blocks_y) {
        // Truncate Y and HY. They may become empty!
        Y  = Y.rightCols(n_blocks_new * b).eval();  //;
        HY = HY.rightCols(n_blocks_new * b).eval(); //
    }
}
// Y is N x ycols
// X is N x xcols
// mask is length n_blocks = ycols / blockWidth
template<typename Scalar>
void RGB<Scalar>::selective_orthonormalize(const Eigen::Ref<const MatrixType> X,            // (N, xcols)
                                           Eigen::Ref<MatrixType>             Y,            // (N, ycols)
                                           RealScalar                         breakdownTol, // The smallest allowed norm
                                           VectorIdxT                        &mask          // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == Y.cols() / b && "Mask size must match number of blocks in Y.");

    const Eigen::Index n_blocks_y = Y.cols() / b;
    const Eigen::Index n_blocks_x = X.cols() / b;
    const Eigen::Index xcols      = X.cols();
    const Eigen::Index ycols      = Y.cols();

    if(xcols == 0 || ycols == 0) return;

    // Compute Gram matrix G = X^H Y (xcols x ycols)
    MatrixType G = X.adjoint() * Y;

    for(Eigen::Index blk = 0; blk < n_blocks_y; ++blk) {
        bool bad = false;
        // Check overlap for any entry in this block
        for(Eigen::Index prev_blk = 0; prev_blk < n_blocks_x; ++prev_blk) {
            const auto G_block = G.block(prev_blk * b, blk * b, b, b);
            if(G_block.cwiseAbs().maxCoeff() > orthTolQ) {
                bad = true;
                break;
            }
        }

        if(bad) {
            // Clean this block
            auto Yblock = Y.middleCols(blk * b, b);
            for(int rep = 0; rep < 2; rep++) {
                Yblock -= X * (X.adjoint() * Yblock).eval(); // (N, blockWidth) - (N, xcols) * (xcols, blockWidth)
            }
        }
    }

    // Orthonormalize all columns of Y together
    hhqr.compute(Y);
    Y = hhqr.householderQ().setLength(Y.cols()) * MatrixType::Identity(Y.rows(), Y.cols());

    // Update mask
    for(Eigen::Index blk = 0; blk < n_blocks_y; ++blk) {
        auto block_norm = Y.middleCols(blk * b, b).norm(); // Frobenius norm of block
        if(block_norm < breakdownTol) mask(blk) = 0;
    }
}

template<typename Scalar> typename RGB<Scalar>::MatrixType RGB<Scalar>::get_wBlock() {
    // We add Lanczos-style residual blocks
    W = HV;
    assert(W.allFinite());
    A                    = V.adjoint() * W;
    status.max_eval_est = std::max({status.max_eval_est, A.norm() * std::abs(std::sqrt<RealScalar>(b))});

    // 3) Subtract projections to A and B once
    W.noalias() -= V * A; // Qi * Qi.adjoint()*H*Qi
    if(V_prev.rows() == N and V_prev.cols() == b) {
        B = V_prev.adjoint() * W;
        W.noalias() -= V_prev * B.adjoint();
        status.max_eval_est = std::max({status.max_eval_est, B.norm() * std::abs(std::sqrt<RealScalar>(b))});
    }
    if constexpr(settings::debug_RGB) {
        bool allFinite = W.allFinite();
        if(!allFinite) { eig::log->warn("W is not all finite: \n{}\n", linalg::matrix::to_string(W, 8)); }
        assert(allFinite);
    }
    return W;
}
template<typename Scalar> typename RGB<Scalar>::MatrixType RGB<Scalar>::get_mBlock() {
    // M are the b next-best ritz vectors from the previous iteration
    assert_allfinite(M);
    return M;
}
template<typename Scalar> typename RGB<Scalar>::MatrixType RGB<Scalar>::get_sBlock() {
    // Make a residual block "S = (HQ-λQ)"
    MatrixType S = HV - V * T_evals(status.optIdx).asDiagonal(); // Put the residual "R" in S.
    assert_allfinite(S);
    return S;
}
template<typename Scalar> typename RGB<Scalar>::MatrixType RGB<Scalar>::get_rBlock() {
    // Get a random block
    return MatrixType::Random(N, b);
}

template<typename Scalar>
void RGB<Scalar>::build_Q_enr_i() {
    const Eigen::Index N = H1.rows();
    assert(V.cols() == b);
    assert_orthonormal(V, normTolQ);

    // Now V has b orthonormalized ritz vectors

    // Start defining the blocks of Q
    Eigen::Index wBlocks_old = wBlocks;
    Eigen::Index mBlocks_old = mBlocks;
    Eigen::Index sBlocks_old = sBlocks;
    // Eigen::Index rBlocks_old = rBlocks;

    auto get_total_blocks = [&]() { return wBlocks + mBlocks + sBlocks + rBlocks; };

    wBlocks = std::min(max_wBlocks, wBlocks_old + 1);                     // Add space for one more W block
    mBlocks = M.cols() == b ? std::min(max_mBlocks, mBlocks_old + 1) : 0; // Add space for one more M block
    sBlocks = std::min(max_sBlocks, sBlocks_old + 1);                     // Add space for one more S block
    rBlocks = (inject_randomness and status.iter > 20 and status.iter % 20 == 0 and get_total_blocks() * b <= N) ? 1 : 0;

    // Try to keep W and S if possible, drop R, M first
    while(N - V.cols() < get_total_blocks() * b) {
        /* clang-format off */
        if(rBlocks > 0) { rBlocks--; continue; }
        if(mBlocks > 0) { mBlocks--; continue; }
        if(wBlocks > 0) { wBlocks--; continue; }
        if(sBlocks > 0) { sBlocks--; continue; }
        break; // If all are at min, break to avoid infinite loop
        /* clang-format on */
    }

    Q_enr_i.conservativeResize(N, (wBlocks + mBlocks + sBlocks + rBlocks) * b);
    assert(N >= Q_enr_i.cols() + V.cols());
    if(Q_enr_i.cols() == 0) return;

    Eigen::Index wOffset = 0;
    Eigen::Index mOffset = wBlocks;
    Eigen::Index sOffset = wBlocks + mBlocks;
    Eigen::Index rOffset = wBlocks + mBlocks + sBlocks;

    if(wBlocks > 0) Q_enr_i.middleCols(wOffset * b, b) = get_wBlock();
    if(mBlocks > 0) Q_enr_i.middleCols(mOffset * b, b) = get_mBlock();
    if(sBlocks > 0) Q_enr_i.middleCols(sOffset * b, b) = get_sBlock();
    if(rBlocks > 0) Q_enr_i.middleCols(rOffset * b, b) = get_rBlock();

    if(use_preconditioner) {
        // Precondition the latest W, S and R,
        /* clang-format off */
        if(wBlocks > 0) {Q_enr_i.middleCols(wOffset * b, b) = -MultPX(Q_enr_i.middleCols(wOffset * b, b));}
        if(sBlocks > 0) {Q_enr_i.middleCols(sOffset * b, b) = -MultPX(Q_enr_i.middleCols(sOffset * b, b));}
        if(rBlocks > 0) {Q_enr_i.middleCols(rOffset * b, b) = -MultPX(Q_enr_i.middleCols(rOffset * b, b));}
        /* clang-format on */
    }
    if(chebyshev_filter_degree >= 1) {
        // Apply the chebyshev filter on newly generated residual and random blocks
        auto W = wBlocks > 0 ? Q_enr_i.middleCols(wOffset * b, b) : Q_enr_i.middleCols(wOffset * b, 0);
        auto S = sBlocks > 0 ? Q_enr_i.middleCols(sOffset * b, b) : Q_enr_i.middleCols(sOffset * b, 0);
        auto R = rBlocks > 0 ? Q_enr_i.middleCols(rOffset * b, b) : Q_enr_i.middleCols(rOffset * b, 0);

        /* clang-format off */
        if(wBlocks > 0) {W = qr_and_chebyshevFilter(W);}
        if(sBlocks > 0) {S = qr_and_chebyshevFilter(S);}
        if(rBlocks > 0) {R = qr_and_chebyshevFilter(R);}
        /* clang-format on */
    }
    // pick a relative breakdown tolerance:
    auto       breakdownTol      = eps * 10 * std::max({status.H_norm_est()});
    VectorIdxT active_block_mask = VectorIdxT::Ones(wBlocks + mBlocks + sBlocks + rBlocks);

    // orthonormalize(Q_enr, Q_enr_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
    orthonormalize(V, Q_enr_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
    assert_orthogonal(V, Q_enr_i, breakdownTol);
    compress_cols(Q_enr_i, active_block_mask);

    if(Q_enr_i.cols() == 0) {
        // // Happy breakdown!
        eig::log->warn("optVal {::.16f}: ", fv(status.optVal));
        eig::log->warn("T_evals {::.16f}: ", fv(T_evals));
        eig::log->debug("saturated basis");
        status.stopReason |= StopReason::saturated_basis;
        status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
        return;
    }

    if constexpr(settings::print_q) eig::log->warn("Q_enr_i after compression: \n{}\n", linalg::matrix::to_string(Q_enr_i, 8));
    assert_allfinite(Q_enr_i);
    assert_orthonormal(Q_enr_i, normTolQ);
}

template<typename Scalar>
void RGB<Scalar>::build() {
    // Form a fresh HV
    HV = MultHX(V);

    build_Q_enr_i();
    assert(Q_enr_i.rows() == N);
    if(status.stopReason != StopReason::none) return;
    // Roll until satisfying |Q_cur.adjoint() * Q_enr| < orthTolQ and Q_enr.cols()/b < max Blocks

    // Append the enrichment for this iteration

    // eig::log->warn("Q_enr maxBLocks {}: \n{}\n", maxBlocks, linalg::matrix::to_string(Q_enr, 8));
    // eig::log->warn("Q_enr_i : \n{}\n", linalg::matrix::to_string(Q_enr_i, 8));
    auto oldBlocks = Q_enr.cols() / b;
    auto newBlocks = std::max<Eigen::Index>(1, std::min<Eigen::Index>({(Q_enr.cols() + Q_enr_i.cols()) / b, (N / b - 1), (maxBasisBlocks - 1)}));
    if(oldBlocks != newBlocks) {
        Q_enr.conservativeResize(N, newBlocks * b);
        Q_enr.rightCols(oldBlocks * b) = Q_enr.leftCols(oldBlocks * b).eval();
    }
    auto copyBlocks                = std::min<Eigen::Index>(Q_enr.cols() / b, Q_enr_i.cols() / b);
    Q_enr.leftCols(copyBlocks * b) = Q_enr_i.leftCols(copyBlocks * b);

    // Append also the enrichment onto HQ_enr
    // MatrixType HQ_enr_i = MultHX(Q_enr_i);
    // HQ_enr.conservativeResize(N, HQ_enr.cols() + HQ_enr_i.cols());
    // HQ_enr.rightCols(HQ_enr_i.cols()) = HQ_enr_i;

    // eig::log->info("before delete: V.cols() = {} | Q_enr.cols() = {} | Q_enr_i.cols() = {}", V.cols(), Q_enr.cols(), Q_enr_i.cols());
    // delete_blocks_from_left_until_orthogonal(V, Q_enr, HQ_enr, maxBasisBlocks - 1, 1e-6);
    // eig::log->info("after  delete: V.cols() = {} | Q_enr.cols() = {} | Q_enr_i.cols() = {}", V.cols(), Q_enr.cols(), Q_enr_i.cols());

    if(Q_enr.cols() == 0) {
        // Happy breakdown!
        eig::log->warn("optVal {::.16f}: ", fv(status.optVal));
        eig::log->warn("T_evals {::.16f}: ", fv(T_evals));
        eig::log->debug("saturated basis");
        status.stopReason |= StopReason::saturated_basis;
        status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
        return;
    }
    // eig::log->warn("Q_enr : \n{}\n", linalg::matrix::to_string(Q_enr, 8));

    // Form Q
    Q.resize(N, V.cols() + Q_enr.cols());
    Q.leftCols(V.cols())      = V;
    Q.rightCols(Q_enr.cols()) = Q_enr;
    // eig::log->warn("Q : \n{}\n", linalg::matrix::to_string(Q, 8));

    Eigen::ColPivHouseholderQR<MatrixType> cphq(Q); // Compute QR decomposition of Q
    auto                                   rank = cphq.rank();
    Q                                           = hhqr.householderQ().setLength(rank) * MatrixType::Identity(Q.rows(), rank); // Recover Q

    // auto       nblocks = Q.cols() / b;
    // VectorIdxT mask    = VectorIdxT::Ones(nblocks);
    // for(Eigen::Index blk = 0; blk < nblocks; ++blk) {
    //     auto Qblock = Q.middleCols(blk * b, b);
    //     mask(blk)   = Qblock.norm() > RealScalar{0.5f};
    // }
    // compress(Q, mask);

    // Form HQ
    HQ = MultHX(Q);
    // HQ.resize(N, HV.cols() + HQ_enr.cols());
    // HQ.leftCols(HV.cols())      = HV;
    // HQ.rightCols(HQ_enr.cols()) = HQ_enr;

    // eig::log->warn("Q : \n{}\n", linalg::matrix::to_string(Q, 8));
    // eig::log->warn("HQ : \n{}\n", linalg::matrix::to_string(HQ, 8));

    assert(Q.colwise().norm().minCoeff() > eps);
    assert(HQ.colwise().norm().minCoeff() > eps);

    // Form G
    G = Q.adjoint() * Q;                             // Gram matrix
    G = RealScalar{0.5f} * (G + G.adjoint()).eval(); // Symmetrize

    assert_orthonormal(G, RealScalar{2e-1f});

    // Form T
    T = Q.adjoint() * HQ;
    T = RealScalar{0.5f} * (T + T.adjoint()).eval(); // Symmetrize
    assert(T.colwise().norm().minCoeff() > eps);

    if constexpr(settings::print_q) eig::log->warn("T : \n{}\n", linalg::matrix::to_string(T, 8));

    // Solve T by calling diagonalizeT() elsewhere
}

template<typename Scalar>
void RGB<Scalar>::diagonalizeT() {
    if(status.stopReason != StopReason::none) return;
    if(T.rows() == 0) return;
    assert(T.colwise().norm().minCoeff() != 0);
    // We expect Q to have some non-orthogonality, so we should
    // use solve a generalized eigenvalue problem instead.
    // The ritz vectors V = (Q * eigenvectors) that we get are G-orthonormal,
    // meaning they are the usual ritz vectors if G == I. But if G != I, we should
    // orthonormalize as V = QR(V)

    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T, G, Eigen::Ax_lBx);
    if(es.info() == Eigen::Success and T_evals.minCoeff() > 0) {
        T_evals = es.eigenvalues();
        T_evecs = es.eigenvectors().colwise().normalized();
    } else {
        eig::log->warn("G: \n{}\n", linalg::matrix::to_string(G, 8));
        Eigen::SelfAdjointEigenSolver<MatrixType> es2(T);

        T_evals = es2.eigenvalues();
        T_evecs = es2.eigenvectors();
        assert_allfinite(T_evecs);
    }
}

template<typename Scalar>
void RGB<Scalar>::extractRitzVectors() {
    if(status.stopReason != StopReason::none) return;
    if(T.rows() < b) return;
    // Here we DO NOT assume that Q is orthonormal.

    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
    auto Z        = T_evecs(Eigen::all, status.optIdx);
    // Refined extraction
    if(use_refined_rayleigh_ritz) {
        // Get the regular ritz vector residual
        status.rNorms = (HQ * Z - Q * T_evals.asDiagonal() * Z).colwise().norm();

        for(Eigen::Index j = 0; j < static_cast<Eigen::Index>(status.optIdx.size()); ++j) {
            const auto &theta = T_evals(status.optIdx[j]);

            MatrixType M = HQ - theta * Q;  // Residual
            MatrixType G = Q.adjoint() * Q; // Gram Matrix
            MatrixType K = M.adjoint() * M; // Squared residual

            // Symmetrize
            G = RealScalar{0.5f} * (G + G.adjoint()).eval();
            K = RealScalar{0.5f} * (K + K.adjoint()).eval();

            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> gsolver(K, G, Eigen::Ax_lBx);

            RealScalar refinedRnorm = std::sqrt(std::abs(gsolver.eigenvalues()(0)));
            if(gsolver.info() == Eigen::Success and refinedRnorm < 10 * status.rNorms(j)) {
                // Accept the solution
                auto Z_ref       = gsolver.eigenvectors().col(0); // eigenvector for smallest eigenvalue
                V.col(j)         = (Q * Z_ref).normalized();
                status.rNorms(j) = refinedRnorm;
            } else {
                auto GnormError = (G.adjoint() * G - MatrixType::Identity(G.cols(), G.cols())).norm();
                eig::log->info("refinement failed on ritz vector {} | rnorm: refined={:.5e}, standard={:.5e} | GnormError {:.5e} ", j, refinedRnorm,
                               status.rNorms(j), GnormError);
                eig::log->info("gsolver eigenvalues: {}", linalg::matrix::to_string(gsolver.eigenvalues().transpose(), 8));
                eig::log->info("G\n{}\n", linalg::matrix::to_string(G, 8));
                eig::log->info("T_evecs\n{}\n", linalg::matrix::to_string(T_evecs, 8));
                eig::log->warn("T_evals: \n{}\n", linalg::matrix::to_string(T_evals, 8));
                V.col(j)   = Q * T_evecs(Eigen::all, status.optIdx[j]);
                T_evals(j) = (V.col(j).adjoint() * MultHX(V.col(j))).real().coeff(0);
            }
        }
    }

    else {
        // Regular Rayleigh-Ritz
        V = Q * Z;
    }

    // Orthonormalize because Q may not be fully orthonormal
    hhqr.compute(V);
    V = hhqr.householderQ().setLength(V.cols()) * MatrixType::Identity(V.rows(), V.cols());

    if(use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b) {
        auto top_2b_indices = get_ritz_indices(ritz, b, b, T_evals);
        M                   = Q * T_evecs(Eigen::all, top_2b_indices);
        assert_allfinite(M);
        // Orthonormalize because Q may not be fully orthonormal
        hhqr.compute(M);
        M = hhqr.householderQ().setLength(M.cols()) * MatrixType::Identity(M.rows(), M.cols());
        assert_allfinite(M);
    }
}

template<typename Scalar>
void RGB<Scalar>::extractResidualNorms() {
    if(status.stopReason != StopReason::none) return;
    if(T.rows() < b) return;
    assert_orthonormal(V, normTolQ);
    // Here we assume V is already orthonormal (even if Q was not a good orthonormal basis)
    // We can also do this calculation even if we already have residual norms from
    // refined extraction. These are more accurate, and we need HV later anyway.

    // Step 1: Apply H to new V
    HV = MultHX(V);

    // Step 5: Residuals
    status.rNorms = (HV - V * T_evals(status.optIdx).asDiagonal()).colwise().norm();
}

template<typename Scalar>
void RGB<Scalar>::set_maxLanczosResidualHistory(Eigen::Index w) {
    w = std::clamp<Eigen::Index>(w, 0, 1); // wBlocks are not required
    if(w != max_wBlocks) eig::log->debug("RGB: max_wBlocks = {}", max_wBlocks);
    max_wBlocks = w;
}

template<typename Scalar>
void RGB<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    m = std::clamp<Eigen::Index>(m, 0, 1); // mBlocks are not required
    if(m != max_mBlocks) eig::log->debug("RGB: max_mBlocks = {}", max_mBlocks);
    max_mBlocks                              = m;
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void RGB<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    s = std::clamp<Eigen::Index>(s, 0, 1); // sBlocks are not required
    if(s != max_sBlocks) eig::log->debug("RGB: max_sBlocks = {}", max_sBlocks);
    max_sBlocks = s;
}

template<typename Scalar>
void RGB<Scalar>::set_maxBasisBlocks(Eigen::Index bs) {
    if(bs == 0) throw except::runtime_error("bs must be at least 2 | it is {}", bs);
    b  = std::min(std::max(nev, b), N / 2);
    bs = std::min<Eigen::Index>(bs, N / b);
    bs = std::max<Eigen::Index>(bs, 1);
    if(bs != maxBasisBlocks) eig::log->debug("RGB: maxBasisBlocks = {}", bs);
    maxBasisBlocks = bs;
    if(maxBasisBlocks == 0) throw except::runtime_error("maxBasisBlocks must be at least 1 | it is {}", maxBasisBlocks);
}