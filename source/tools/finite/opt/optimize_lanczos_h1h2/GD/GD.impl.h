#pragma once
#include "../GD.h"
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
    constexpr bool debug_GD = false;
#else
    constexpr bool debug_GD = true;
#endif
    constexpr bool print_q = false;
}

template<typename Scalar>
void GD<Scalar>::delete_blocks_from_left_until_orthogonal(const Eigen::Ref<const MatrixType> X, MatrixType &Y, MatrixType &HY, Eigen::Index maxBlocks,
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
void GD<Scalar>::selective_orthonormalize(const Eigen::Ref<const MatrixType> X,            // (N, xcols)
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

template<typename Scalar>
typename GD<Scalar>::MatrixType GD<Scalar>::get_Q_res(std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MultPX) {
    assert(V.rows() == N);
    assert(V.cols() == b);
    assert_orthonormal(V, orthTolQ);
    // Now V has b orthonormalized ritz vectors

    // Start defining the residual blocks (we have ritz-type and lanczos type)
    Eigen::Index wBlocks_old = wBlocks;
    Eigen::Index sBlocks_old = sBlocks;

    auto get_total_blocks = [&]() { return wBlocks + sBlocks; };

    wBlocks = std::min(max_wBlocks, wBlocks_old + 1); // Add space for one more W block
    sBlocks = std::min(max_sBlocks, sBlocks_old + 1); // Add space for one more S block

    // Try to keep W and S if possible, drop R, M first
    while(N - V.cols() < get_total_blocks() * b) {
        /* clang-format off */
        if(wBlocks > 0) { wBlocks--; continue; }
        if(sBlocks > 0) { sBlocks--; continue; }
        break; // If all are at min, break to avoid infinite loop
        /* clang-format on */
    }
    MatrixType Q_res(N, (wBlocks + sBlocks) * b);

    assert(N >= Q_res.cols() + V.cols());
    // eig::log->warn("V \n{}\n", linalg::matrix::to_string(V, 8));
    // eig::log->warn("HV \n{}\n", linalg::matrix::to_string(HV, 8));

    Eigen::Index wOffset = 0;
    Eigen::Index sOffset = wBlocks + mBlocks;
    if(wBlocks > 0) Q_res.middleCols(wOffset * b, b) = get_wBlock();
    if(sBlocks > 0) Q_res.middleCols(sOffset * b, b) = get_sBlock();
    // eig::log->warn("Q_res before filtering: \n{}\n", linalg::matrix::to_string(Q_res, 8));

    if(chebyshev_filter_degree >= 1) {
        // Apply the chebyshev filter on newly generated residual and random blocks

        auto W = wBlocks > 0 ? Q_res.middleCols(wOffset * b, b) : Q_res.middleCols(wOffset * b, 0);
        auto S = sBlocks > 0 ? Q_res.middleCols(sOffset * b, b) : Q_res.middleCols(sOffset * b, 0);

        /* clang-format off */
        if(wBlocks > 0) {W = qr_and_chebyshevFilter(W);}
        if(sBlocks > 0) {S = qr_and_chebyshevFilter(S);}
        /* clang-format on */
    }
    if(use_preconditioner) {
        // Precondition the latest W, S and R,
        /* clang-format off */
        if(wBlocks > 0) {Q_res.middleCols(wOffset * b, b) = MultPX(Q_res.middleCols(wOffset * b, b));}
        if(sBlocks > 0) {
            if(residual_correction_type == ResidualCorrectionType::NONE or residual_correction_type == ResidualCorrectionType::CHEAP_OLSEN ) {
                Q_res.middleCols(sOffset * b, b) = MultPX(Q_res.middleCols(sOffset * b, b));
            }
        }

        /* clang-format on */
    }

    // pick a relative breakdown tolerance:
    auto       breakdownTol      = eps * 10 * std::max({RealScalar{1}, status.max_eval_estimate()});
    VectorIdxT active_block_mask = VectorIdxT::Ones(wBlocks + sBlocks);
    // eig::log->warn("Q_res before compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));
    // orthonormalize(Q_enr, Q_enr_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
    orthonormalize(V, Q_res, breakdownTol, 10000 * breakdownTol, active_block_mask);
    assert_orthogonal(V, Q_res, breakdownTol);
    orthonormalize(Q, Q_res, breakdownTol, 10000 * breakdownTol, active_block_mask);
    assert_orthogonal(Q, Q_res, breakdownTol);
    compress_cols(Q_res, active_block_mask);
    // eig::log->warn("Q_res after  compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));

    assert_allfinite(Q_res);
    assert_orthonormal(Q_res, orthTolQ);
    if constexpr(settings::print_q) eig::log->warn("Q_res after compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));
    return Q_res;
}

template<typename Scalar>
void GD<Scalar>::build() {
    switch(algo) {
        case OptAlgo::DMRG: [[fallthrough]];
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
        case OptAlgo::XDMRG: {
            auto Q_res = get_Q_res([this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return this->MultPX(X); });
            build(Q_res, Q, HQ, [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return this->MultHX(X); });
            break;
        }
        case OptAlgo::GDMRG: {
            auto Q_res = get_Q_res([this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return this->MultP2X(X); });
            build(Q_res, Q, H1Q, H2Q);
            break;
        }
        default: throw except::runtime_error("invalid algo {}", enum2sv(algo));
    }
}

template<typename Scalar>
void GD<Scalar>::build(MatrixType &Q_res, MatrixType &Q, MatrixType &HQ, std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MultHX) {
    if(status.stopReason != StopReason::none) return;
    assert(Q_res.rows() == N);
    // Roll until satisfying |Q_cur.adjoint() * Q_enr| < orthTolQ and Q_enr.cols()/b < max Blocks

    // Append the enrichment for this iteration

    // eig::log->warn("Q_enr maxBLocks {}: \n{}\n", maxBlocks, linalg::matrix::to_string(Q_enr, 8));
    auto oldBlocks = Q.cols() / b;
    auto newBlocks = std::max<Eigen::Index>(1, std::min<Eigen::Index>({(Q.cols() + Q_res.cols()) / b, (N / b)}));
    if(newBlocks > maxBasisBlocks or Q.cols() == 0 or Q_res.cols() == 0) {
        // (re)start
        if(Q_res.cols() == 0 and status.iter == status.iter_last_restart + 1) {
            // Failed to add a nonzero residual
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
            return;
        }

        status.iter_last_restart = status.iter;
        Eigen::Index vBlocks     = V.cols() / b;
        Eigen::Index mBlocks     = use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b ? 1 : 0;
        Eigen::Index rBlocks     = inject_randomness ? 1 : 0;
        Eigen::Index kBlocks     = std::max<Eigen::Index>(0, std::min<Eigen::Index>(Q.cols() / b - vBlocks + mBlocks + rBlocks, maxRetainBlocks));
        Eigen::Index qBlocks     = Q_res.cols() / b;
        MatrixType   Q_keep      = Q.rightCols(kBlocks * b);
        MatrixType   HQ_keep     = HQ.rightCols(kBlocks * b);
        if(mBlocks > 0) M = get_mBlock(); // Before modifying Q
        Q.conservativeResize(N, (vBlocks + mBlocks + rBlocks + kBlocks) * b + Q_res.cols());
        HQ.conservativeResize(N, (vBlocks + mBlocks + rBlocks + kBlocks) * b + Q_res.cols());
        auto vOffset = 0;
        auto mOffset = vBlocks;
        auto kOffset = vBlocks + mBlocks;
        auto rOffset = vBlocks + mBlocks + kBlocks;
        auto qOffset = vBlocks + mBlocks + kBlocks + rBlocks;
        if(vBlocks > 0) Q.middleCols(vOffset * b, vBlocks * b) = V;
        if(mBlocks > 0) Q.middleCols(mOffset * b, mBlocks * b) = M;
        if(kBlocks > 0) Q.middleCols(kOffset * b, kBlocks * b) = Q_keep;
        if(rBlocks > 0) Q.middleCols(rOffset * b, rBlocks * b) = get_rBlock();
        if(qBlocks > 0) Q.middleCols(qOffset * b, qBlocks * b) = Q_res;
        MatrixType Qold = Q;
        hhqr.compute(Q);
        Q = hhqr.householderQ().setLength(Q.cols()) * MatrixType::Identity(N, Q.cols());

        VectorReal D = (Qold.conjugate().array() * Q.array()).colwise().sum().real();
        VectorReal E = (D.cwiseAbs() - VectorType::Ones(D.size())).cwiseAbs();

        // Start building HQ
        if(vBlocks > 0) HQ.middleCols(vOffset * b, vBlocks * b) = HV;
        if(mBlocks > 0) HQ.middleCols(mOffset * b, mBlocks * b) = HM; // HM was set during the last set_mBlock() call
        if(kBlocks > 0) HQ.middleCols(kOffset * b, kBlocks * b) = HQ_keep;
        if(rBlocks > 0) HQ.middleCols(rOffset * b, rBlocks * b) = MultHX(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided
        if(qBlocks > 0) HQ.middleCols(qOffset * b, qBlocks * b) = MultHX(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided
        // We may need to rebuild columns of HQ that were affected by QR,
        // that is, if the orthonormalization of Q rotated those columns.
        // To know for sure, we have to check E
        for(Eigen::Index i = 0; i < Q.cols(); ++i) {
            if(rOffset * b <= i && i < (rOffset + rBlocks) * b) continue; // Skip because it was already built
            if(qOffset * b <= i && i < (qOffset + qBlocks) * b) continue; // Skip because it was already built
            if(E(i) < normTolQ) {
                RealScalar sign = D(i) > 0 ? RealScalar{1} : RealScalar{-1};
                HQ.col(i) *= sign; // Q and HQ columns were preserved up to a sign
            } else {
                HQ.col(i) = MultHX(Q.col(i)); // HQ column was not preserved in QR
            }
        }
    } else if(oldBlocks != newBlocks) {
        // Append enrichment
        Q.conservativeResize(N, newBlocks * b);
        HQ.conservativeResize(N, newBlocks * b);
        auto copyBlocks              = std::min<Eigen::Index>(Q.cols() / b, Q_res.cols() / b);
        Q.rightCols(copyBlocks * b)  = Q_res.leftCols(copyBlocks * b);
        HQ.rightCols(copyBlocks * b) = MultHX(Q_res);
    }

    assert(Q.colwise().norm().minCoeff() > eps);
    assert(HQ.colwise().norm().minCoeff() > eps);
}

template<typename Scalar>
void GD<Scalar>::build(MatrixType &Q1_res, MatrixType &Q2_res, MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q) {
    if(status.stopReason != StopReason::none) return;
    assert(Q1_res.rows() == N);
    assert(Q2_res.rows() == N);

    auto       breakdownTol = eps * 100 * std::max({RealScalar{1}, status.max_eval_estimate()});
    VectorIdxT mask         = VectorIdxT::Ones(Q2_res.cols() / b);
    orthonormalize(Q1_res, Q2_res, breakdownTol * 1000, breakdownTol, mask);
    compress_cols(Q2_res, mask);

    // Roll until satisfying |Q_cur.adjoint() * Q_enr| < orthTolQ and Q_enr.cols()/b < max Blocks

    // Append the enrichment for this iteration

    // eig::log->warn("Q_enr maxBLocks {}: \n{}\n", maxBlocks, linalg::matrix::to_string(Q_enr, 8));
    auto oldBlocks = Q.cols() / b;
    auto newBlocks = std::max<Eigen::Index>(1, std::min<Eigen::Index>({(Q.cols() + Q1_res.cols() + Q2_res.cols()) / b, (N / b)}));
    if(newBlocks > maxBasisBlocks or Q.cols() == 0 or Q1_res.cols() == 0 or Q2_res.cols() == 0) {
        // (re)start
        if(Q1_res.cols() + Q2_res.cols() == 0 and status.iter == status.iter_last_restart + 1) {
            // Failed to add a nonzero residual
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
            return;
        }

        status.iter_last_restart = status.iter;
        Eigen::Index vBlocks     = V.cols() / b;
        Eigen::Index mBlocks     = use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b ? 1 : 0;
        Eigen::Index rBlocks     = inject_randomness ? 1 : 0;
        Eigen::Index kBlocks     = std::max<Eigen::Index>(
            0, std::min<Eigen::Index>(Q.cols() / b - (vBlocks + mBlocks + rBlocks) - Q1_res.cols() / b - Q2_res.cols() / b, maxRetainBlocks));
        Eigen::Index qBlocks  = (Q1_res.cols() + Q2_res.cols()) / b;
        MatrixType   Q_keep   = Q.rightCols(kBlocks * b);
        MatrixType   H1Q_keep = H1Q.rightCols(kBlocks * b);
        MatrixType   H2Q_keep = H2Q.rightCols(kBlocks * b);
        if(mBlocks > 0) M = get_mBlock(); // Before modifying Q
        Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q1_res.cols() + Q2_res.cols());
        H1Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q1_res.cols() + Q2_res.cols());
        H2Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q1_res.cols() + Q2_res.cols());
        auto vOffset = 0;
        auto mOffset = vBlocks;
        auto kOffset = vBlocks + mBlocks;
        auto rOffset = vBlocks + mBlocks + kBlocks;
        auto qOffset = vBlocks + mBlocks + kBlocks + rBlocks;
        if(vBlocks > 0) Q.middleCols(vOffset * b, vBlocks * b) = V;
        if(mBlocks > 0) Q.middleCols(mOffset * b, mBlocks * b) = M;
        if(kBlocks > 0) Q.middleCols(kOffset * b, kBlocks * b) = Q_keep;
        if(rBlocks > 0) Q.middleCols(rOffset * b, rBlocks * b) = get_rBlock();
        if(qBlocks > 0) {
            Q.middleCols(qOffset * b, qBlocks * b).leftCols(Q1_res.cols())  = Q1_res;
            Q.middleCols(qOffset * b, qBlocks * b).rightCols(Q2_res.cols()) = Q2_res;
        }
        MatrixType Qold = Q;
        hhqr.compute(Q);
        Q = hhqr.householderQ().setLength(Q.cols()) * MatrixType::Identity(N, Q.cols());

        VectorReal D = (Qold.conjugate().array() * Q.array()).colwise().sum().real();
        VectorReal E = (D.cwiseAbs() - VectorType::Ones(D.size())).cwiseAbs();

        // Build H1Q and H2Q
        if(vBlocks > 0) H1Q.middleCols(vOffset * b, vBlocks * b) = H1V;
        if(vBlocks > 0) H2Q.middleCols(vOffset * b, vBlocks * b) = H2V;

        if(mBlocks > 0) H1Q.middleCols(mOffset * b, mBlocks * b) = H1M; // Set during the last call to set_mBlock(...)
        if(mBlocks > 0) H2Q.middleCols(mOffset * b, mBlocks * b) = H2M; // Set during the last call to set_mBlock(...)

        if(kBlocks > 0) H1Q.middleCols(kOffset * b, kBlocks * b) = H1Q_keep;
        if(kBlocks > 0) H2Q.middleCols(kOffset * b, kBlocks * b) = H2Q_keep;

        if(rBlocks > 0) H1Q.middleCols(rOffset * b, rBlocks * b) = MultH1X(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided
        if(rBlocks > 0) H2Q.middleCols(rOffset * b, rBlocks * b) = MultH2X(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided

        if(qBlocks > 0) H1Q.middleCols(qOffset * b, qBlocks * b) = MultH1X(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided
        if(qBlocks > 0) H2Q.middleCols(qOffset * b, qBlocks * b) = MultH2X(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided

        // We may need to rebuild columns of H1Q and H2Q that were affected by QR,
        // that is, if the orthonormalization of Q rotated those columns.
        // To know for sure, we have to check E
        for(Eigen::Index i = 0; i < Q.cols(); ++i) {
            if(rOffset * b <= i && i < (rOffset + rBlocks) * b) continue; // Skip because it was already built
            if(qOffset * b <= i && i < (qOffset + qBlocks) * b) continue; // Skip because it was already built
            if(E(i) < normTolQ) {
                RealScalar sign = D(i) > 0 ? RealScalar{1} : RealScalar{-1};
                H1Q.col(i) *= sign; // Q and HQ columns were preserved up to a sign
                H2Q.col(i) *= sign; // Q and HQ columns were preserved up to a sign
            } else {
                H1Q.col(i) = MultH1X(Q.col(i)); // HQ column was not preserved in QR
                H2Q.col(i) = MultH2X(Q.col(i)); // HQ column was not preserved in QR
            }
        }
    } else if(oldBlocks != newBlocks) {
        // Append enrichment
        Q.conservativeResize(N, newBlocks * b);
        H1Q.conservativeResize(N, newBlocks * b);
        H2Q.conservativeResize(N, newBlocks * b);

        auto copyBlocks                                      = std::min<Eigen::Index>(Q.cols() / b, (Q1_res.cols() + Q1_res.cols()) / b);
        Q.rightCols(copyBlocks * b).leftCols(Q1_res.cols())  = Q1_res;
        Q.rightCols(copyBlocks * b).rightCols(Q2_res.cols()) = Q2_res;

        H1Q.rightCols(copyBlocks * b).leftCols(Q1_res.cols())  = MultH1X(Q1_res);
        H1Q.rightCols(copyBlocks * b).rightCols(Q2_res.cols()) = MultH1X(Q2_res);

        H2Q.rightCols(copyBlocks * b).leftCols(Q1_res.cols())  = MultH2X(Q1_res);
        H2Q.rightCols(copyBlocks * b).rightCols(Q2_res.cols()) = MultH2X(Q2_res);
    }

    assert(Q.colwise().norm().minCoeff() > eps);
    assert(H1Q.colwise().norm().minCoeff() > eps);
    assert(H2Q.colwise().norm().minCoeff() > eps);
}

template<typename Scalar>
void GD<Scalar>::build(MatrixType &Q_res, MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q) {
    if(status.stopReason != StopReason::none) return;
    assert(Q_res.rows() == N);

    // Roll until satisfying |Q_cur.adjoint() * Q_enr| < orthTolQ and Q_enr.cols()/b < max Blocks

    // Append the enrichment for this iteration

    // eig::log->warn("Q_enr maxBLocks {}: \n{}\n", maxBlocks, linalg::matrix::to_string(Q_enr, 8));
    auto oldBlocks = Q.cols() / b;
    auto newBlocks = std::max<Eigen::Index>(1, std::min<Eigen::Index>({(Q.cols() + Q_res.cols()) / b, (N / b)}));
    if(newBlocks > maxBasisBlocks or Q.cols() == 0 or Q_res.cols() == 0) {
        // (re)start
        // (re)start
        if(Q_res.cols() == 0 and status.iter == status.iter_last_restart + 1) {
            // Failed to add a nonzero residual
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
            return;
        }

        status.iter_last_restart = status.iter;
        Eigen::Index vBlocks     = V.cols() / b;
        Eigen::Index mBlocks     = use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b ? 1 : 0;
        Eigen::Index rBlocks     = inject_randomness ? 1 : 0;
        Eigen::Index kBlocks =
            std::max<Eigen::Index>(0, std::min<Eigen::Index>(Q.cols() / b - (vBlocks + mBlocks + rBlocks) - Q_res.cols() / b, maxRetainBlocks));
        Eigen::Index qBlocks  = (Q_res.cols()) / b;
        MatrixType   Q_keep   = Q.rightCols(kBlocks * b);
        MatrixType   H1Q_keep = H1Q.rightCols(kBlocks * b);
        MatrixType   H2Q_keep = H2Q.rightCols(kBlocks * b);
        if(mBlocks > 0) M = get_mBlock(); // Before modifying Q
        Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q_res.cols());
        H1Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q_res.cols());
        H2Q.conservativeResize(N, (vBlocks + mBlocks + kBlocks + rBlocks) * b + Q_res.cols());
        auto vOffset = 0;
        auto mOffset = vBlocks;
        auto kOffset = vBlocks + mBlocks;
        auto rOffset = vBlocks + mBlocks + kBlocks;
        auto qOffset = vBlocks + mBlocks + kBlocks + rBlocks;
        if(vBlocks > 0) Q.middleCols(vOffset * b, vBlocks * b) = V;
        if(mBlocks > 0) Q.middleCols(mOffset * b, mBlocks * b) = M;
        if(kBlocks > 0) Q.middleCols(kOffset * b, kBlocks * b) = Q_keep;
        if(rBlocks > 0) Q.middleCols(rOffset * b, rBlocks * b) = get_rBlock();
        if(qBlocks > 0) Q.middleCols(qOffset * b, qBlocks * b) = Q_res;

        MatrixType Qold = Q;
        hhqr.compute(Q);
        Q = hhqr.householderQ().setLength(Q.cols()) * MatrixType::Identity(N, Q.cols());

        VectorReal D = (Qold.conjugate().array() * Q.array()).colwise().sum().real();
        VectorReal E = (D.cwiseAbs() - VectorType::Ones(D.size())).cwiseAbs();

        // Build H1Q and H2Q
        if(vBlocks > 0) H1Q.middleCols(vOffset * b, vBlocks * b) = H1V;
        if(vBlocks > 0) H2Q.middleCols(vOffset * b, vBlocks * b) = H2V;

        if(mBlocks > 0) H1Q.middleCols(mOffset * b, mBlocks * b) = H1M; // Set during the last call to set_mBlock(...)
        if(mBlocks > 0) H2Q.middleCols(mOffset * b, mBlocks * b) = H2M; // Set during the last call to set_mBlock(...)

        if(kBlocks > 0) H1Q.middleCols(kOffset * b, kBlocks * b) = H1Q_keep;
        if(kBlocks > 0) H2Q.middleCols(kOffset * b, kBlocks * b) = H2Q_keep;

        if(rBlocks > 0) H1Q.middleCols(rOffset * b, rBlocks * b) = MultH1X(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided
        if(rBlocks > 0) H2Q.middleCols(rOffset * b, rBlocks * b) = MultH2X(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided

        if(qBlocks > 0) H1Q.middleCols(qOffset * b, qBlocks * b) = MultH1X(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided
        if(qBlocks > 0) H2Q.middleCols(qOffset * b, qBlocks * b) = MultH2X(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided

        // We may need to rebuild columns of H1Q and H2Q that were affected by QR,
        // that is, if the orthonormalization of Q rotated those columns.
        // To know for sure, we have to check E
        for(Eigen::Index i = 0; i < Q.cols(); ++i) {
            if(rOffset * b <= i && i < (rOffset + rBlocks) * b) continue; // Skip because it was already built
            if(qOffset * b <= i && i < (qOffset + qBlocks) * b) continue; // Skip because it was already built
            if(E(i) < normTolQ) {
                RealScalar sign = D(i) > 0 ? RealScalar{1} : RealScalar{-1};
                H1Q.col(i) *= sign; // Q and HQ columns were preserved up to a sign
                H2Q.col(i) *= sign; // Q and HQ columns were preserved up to a sign
            } else {
                H1Q.col(i) = MultH1X(Q.col(i)); // HQ column was not preserved in QR
                H2Q.col(i) = MultH2X(Q.col(i)); // HQ column was not preserved in QR
            }
        }
    } else if(oldBlocks != newBlocks) {
        // Append enrichment
        Q.conservativeResize(N, newBlocks * b);
        H1Q.conservativeResize(N, newBlocks * b);
        H2Q.conservativeResize(N, newBlocks * b);

        auto copyBlocks               = std::min<Eigen::Index>(Q.cols() / b, (Q_res.cols()) / b);
        Q.rightCols(copyBlocks * b)   = Q_res.leftCols(copyBlocks * b);
        H1Q.rightCols(copyBlocks * b) = MultH1X(Q_res.leftCols(copyBlocks * b));
        H2Q.rightCols(copyBlocks * b) = MultH2X(Q_res.leftCols(copyBlocks * b));
    }

    assert(Q.colwise().norm().minCoeff() > eps);
    assert(H1Q.colwise().norm().minCoeff() > eps);
    assert(H2Q.colwise().norm().minCoeff() > eps);
}

template<typename Scalar>
void GD<Scalar>::set_maxLanczosResidualHistory(Eigen::Index w) {
    w = std::clamp<Eigen::Index>(w, 0, 1); // wBlocks are not required
    if(w != max_wBlocks) eig::log->debug("GD: max_wBlocks = {}", max_wBlocks);
    max_wBlocks = w;
}

template<typename Scalar>
void GD<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    m = std::clamp<Eigen::Index>(m, 0, 1); // mBlocks are not required
    if(m != max_mBlocks) eig::log->debug("GD: max_mBlocks = {}", max_mBlocks);
    max_mBlocks                              = m;
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void GD<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    s = std::clamp<Eigen::Index>(s, 0, 1); // sBlocks are not required
    if(s != max_sBlocks) eig::log->debug("GD: max_sBlocks = {}", max_sBlocks);
    max_sBlocks = s;
}

template<typename Scalar>
void GD<Scalar>::set_maxBasisBlocks(Eigen::Index bb) {
    if(bb == 0) throw except::runtime_error("maxBasisBlocks must be at least 2 | it is {}", bb);
    b  = std::min(std::max(nev, b), N / 2);
    bb = std::min<Eigen::Index>(bb, N / b);
    bb = std::max<Eigen::Index>(bb, 1);
    if(bb != maxBasisBlocks) eig::log->debug("GD: maxBasisBlocks = {}", bb);
    maxBasisBlocks = bb;
    if(maxBasisBlocks == 0) throw except::runtime_error("maxBasisBlocks must be at least 1 | it is {}", maxBasisBlocks);
}

template<typename Scalar>
void GD<Scalar>::set_maxRetainBlocks(Eigen::Index rb) {
    b  = std::min(std::max(nev, b), N / 2);
    rb = std::min<Eigen::Index>(rb, N / b);
    if(rb != maxRetainBlocks) eig::log->debug("GD: maxRetainBlocks = {}", rb);
    maxRetainBlocks = rb;
}
//
// template<typename Scalar>
// typename GD<Scalar>::MatrixType GD<Scalar>::get_Q_res(const MatrixType &Q, const MatrixType &HV,
//                                                       std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MultPX) {
//     assert(V.rows() == N);
//     assert(V.cols() == b);
//     assert_orthonormal(V, orthTolQ);
//     // Now V has b orthonormalized ritz vectors
//
//     // Start defining the residual blocks (we have ritz-type and lanczos type)
//     Eigen::Index wBlocks_old = wBlocks;
//     Eigen::Index sBlocks_old = sBlocks;
//
//     auto get_total_blocks = [&]() { return wBlocks + sBlocks; };
//
//     wBlocks = std::min(max_wBlocks, wBlocks_old + 1); // Add space for one more W block
//     sBlocks = std::min(max_sBlocks, sBlocks_old + 1); // Add space for one more S block
//
//     // Try to keep W and S if possible, drop R, M first
//     while(N - V.cols() < get_total_blocks() * b) {
//         /* clang-format off */
//         if(wBlocks > 0) { wBlocks--; continue; }
//         if(sBlocks > 0) { sBlocks--; continue; }
//         break; // If all are at min, break to avoid infinite loop
//         /* clang-format on */
//     }
//     MatrixType Q_res(N, (wBlocks + sBlocks) * b);
//
//     assert(N >= Q_res.cols() + V.cols());
//     // eig::log->warn("V \n{}\n", linalg::matrix::to_string(V, 8));
//     // eig::log->warn("HV \n{}\n", linalg::matrix::to_string(HV, 8));
//
//     Eigen::Index wOffset = 0;
//     Eigen::Index sOffset = wBlocks + mBlocks;
//     if(wBlocks > 0) Q_res.middleCols(wOffset * b, b) = get_wBlock();
//     if(sBlocks > 0) Q_res.middleCols(sOffset * b, b) = get_sBlock();
//     // eig::log->warn("Q_res before filtering: \n{}\n", linalg::matrix::to_string(Q_res, 8));
//
//     if(chebyshev_filter_degree >= 1) {
//         // Apply the chebyshev filter on newly generated residual and random blocks
//
//         auto W = wBlocks > 0 ? Q_res.middleCols(wOffset * b, b) : Q_res.middleCols(wOffset * b, 0);
//         auto S = sBlocks > 0 ? Q_res.middleCols(sOffset * b, b) : Q_res.middleCols(sOffset * b, 0);
//
//         /* clang-format off */
//         if(wBlocks > 0) {W = qr_and_chebyshevFilter(W);}
//         if(sBlocks > 0) {S = qr_and_chebyshevFilter(S);}
//         /* clang-format on */
//     }
//     if(use_preconditioner) {
//         // Precondition the latest W, S and R,
//         /* clang-format off */
//         if(wBlocks > 0) {Q_res.middleCols(wOffset * b, b) = -MultPX(Q_res.middleCols(wOffset * b, b));}
//         if(sBlocks > 0) {Q_res.middleCols(sOffset * b, b) = -MultPX(Q_res.middleCols(sOffset * b, b));}
//         /* clang-format on */
//     }
//
//     // pick a relative breakdown tolerance:
//     auto       breakdownTol      = eps * 10 * std::max({RealScalar{1}, status.H_norm_est()});
//     VectorIdxT active_block_mask = VectorIdxT::Ones(wBlocks + sBlocks);
//     // eig::log->warn("Q_res before compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));
//     // orthonormalize(Q_enr, Q_enr_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     orthonormalize(V, Q_res, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     assert_orthogonal(V, Q_res, breakdownTol);
//     orthonormalize(Q, Q_res, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     assert_orthogonal(Q, Q_res, breakdownTol);
//     compress_cols(Q_res, active_block_mask);
//     // eig::log->warn("Q_res after  compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));
//
//     assert_allfinite(Q_res);
//     assert_orthonormal(Q_res, orthTolQ);
//     if constexpr(settings::print_q) eig::log->warn("Q_res after compression: \n{}\n", linalg::matrix::to_string(Q_res, 8));
//     return Q_res;
// }

// template<typename Scalar>
// void GD<Scalar>::build_Q_res_i() {
//     const Eigen::Index N = H1.rows();
//     assert(V.cols() == b);
//     assert_orthonormal(V, orthTolQ);
//
//     // Now V has b orthonormalized ritz vectors
//
//     // Start defining the residual blocks (we have ritz-type and lanczos type)
//     Eigen::Index wBlocks_old = wBlocks;
//     Eigen::Index sBlocks_old = sBlocks;
//
//     auto get_total_blocks = [&]() { return wBlocks + sBlocks; };
//
//     wBlocks = std::min(max_wBlocks, wBlocks_old + 1); // Add space for one more W block
//     sBlocks = std::min(max_sBlocks, sBlocks_old + 1); // Add space for one more S block
//
//     // Try to keep W and S if possible, drop R, M first
//     while(N - V.cols() < get_total_blocks() * b) {
//         /* clang-format off */
//         if(wBlocks > 0) { wBlocks--; continue; }
//         if(sBlocks > 0) { sBlocks--; continue; }
//         break; // If all are at min, break to avoid infinite loop
//         /* clang-format on */
//     }
//
//     Q_res_i.conservativeResize(N, (wBlocks + sBlocks) * b);
//     assert(N >= Q_res_i.cols() + V.cols());
//     if(Q_res_i.cols() == 0) return;
//
//     Eigen::Index wOffset = 0;
//     Eigen::Index sOffset = wBlocks + mBlocks;
//
//     if(wBlocks > 0) Q_res_i.middleCols(wOffset * b, b) = get_wBlock();
//     if(sBlocks > 0) Q_res_i.middleCols(sOffset * b, b) = get_sBlock();
//     if(chebyshev_filter_degree >= 1) {
//         // Apply the chebyshev filter on newly generated residual and random blocks
//
//         auto W = wBlocks > 0 ? Q_res_i.middleCols(wOffset * b, b) : Q_res_i.middleCols(wOffset * b, 0);
//         auto S = sBlocks > 0 ? Q_res_i.middleCols(sOffset * b, b) : Q_res_i.middleCols(sOffset * b, 0);
//
//         /* clang-format off */
//         if(wBlocks > 0) {W = qr_and_chebyshevFilter(W);}
//         if(sBlocks > 0) {S = qr_and_chebyshevFilter(S);}
//         /* clang-format on */
//     }
//     if(use_preconditioner) {
//         // Precondition the latest W, S and R,
//         /* clang-format off */
//         if(wBlocks > 0) {Q_res_i.middleCols(wOffset * b, b) = -MultPX(Q_res_i.middleCols(wOffset * b, b));}
//         if(sBlocks > 0) {Q_res_i.middleCols(sOffset * b, b) = -MultPX(Q_res_i.middleCols(sOffset * b, b));}
//         /* clang-format on */
//     }
//
//     // pick a relative breakdown tolerance:
//     auto       breakdownTol      = eps * 10 * std::max({RealScalar{1}, status.H_norm_est()});
//     VectorIdxT active_block_mask = VectorIdxT::Ones(wBlocks + mBlocks + sBlocks + rBlocks);
//
//     // orthonormalize(Q_enr, Q_enr_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     orthonormalize(V, Q_res_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     assert_orthogonal(V, Q_res_i, breakdownTol);
//     orthonormalize(Q, Q_res_i, breakdownTol, 10000 * breakdownTol, active_block_mask);
//     assert_orthogonal(Q, Q_res_i, breakdownTol);
//     compress_cols(Q_res_i, active_block_mask);
//
//     if(Q_res_i.cols() == 0) {
//         // // Happy breakdown!
//         eig::log->warn("optVal {::.16f}: ", fv(status.optVal));
//         eig::log->warn("T_evals {::.16f}: ", fv(T_evals));
//         eig::log->debug("saturated basis");
//         status.stopReason |= StopReason::saturated_basis;
//         status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
//         return;
//     }
//
//     if constexpr(settings::print_q) eig::log->warn("Q_enr_i after compression: \n{}\n", linalg::matrix::to_string(Q_res_i, 8));
//     assert_allfinite(Q_res_i);
//     assert_orthonormal(Q_res_i, orthTolQ);
// }

// template<typename Scalar>
// void GD<Scalar>::build() {
//     build_Q_res_i();
//     assert(Q_res_i.rows() == N);
//     if(status.stopReason != StopReason::none) return;
//     // Roll until satisfying |Q_cur.adjoint() * Q_enr| < orthTolQ and Q_enr.cols()/b < max Blocks
//
//     // Append the enrichment for this iteration
//
//     // eig::log->warn("Q_enr maxBLocks {}: \n{}\n", maxBlocks, linalg::matrix::to_string(Q_enr, 8));
//     auto oldBlocks = Q.cols() / b;
//     auto newBlocks = std::max<Eigen::Index>(1, std::min<Eigen::Index>({(Q.cols() + Q_res_i.cols()) / b, (N / b)}));
//     if(newBlocks > maxBasisBlocks or Q.cols() == 0 or Q_res_i.cols() == 0) {
//         // (re)start
//
//         Eigen::Index vBlocks = V.cols() / b;
//         Eigen::Index mBlocks = use_extra_ritz_vectors_in_the_next_basis ? M.cols() / b : 0;
//         Eigen::Index rBlocks = inject_randomness ? 1 : 0;
//         Eigen::Index kBlocks = std::max<Eigen::Index>(0, std::min<Eigen::Index>(Q.cols() / b - vBlocks + mBlocks + rBlocks, maxRetainBlocks));
//         Eigen::Index qBlocks = Q_res_i.cols() / b;
//         MatrixType   Q_keep  = Q.rightCols(kBlocks * b);
//         MatrixType   HQ_keep = HQ.rightCols(kBlocks * b);
//         Q.conservativeResize(N, (vBlocks + mBlocks + rBlocks + kBlocks) * b + Q_res_i.cols());
//         HQ.conservativeResize(N, (vBlocks + mBlocks + rBlocks + kBlocks) * b + Q_res_i.cols());
//         auto vOffset = 0;
//         auto mOffset = vBlocks;
//         auto rOffset = vBlocks + mBlocks;
//         auto kOffset = vBlocks + mBlocks + rBlocks;
//         auto qOffset = vBlocks + mBlocks + rBlocks + kBlocks;
//         if(vBlocks > 0) Q.middleCols(vOffset * b, vBlocks * b) = V;
//         if(mBlocks > 0) Q.middleCols(mOffset * b, mBlocks * b) = get_mBlock();
//         if(rBlocks > 0) Q.middleCols(rOffset * b, rBlocks * b) = get_rBlock();
//         if(kBlocks > 0) Q.middleCols(kOffset * b, kBlocks * b) = Q_keep;
//         if(qBlocks > 0) Q.middleCols(qOffset * b, qBlocks * b) = Q_res_i;
//         MatrixType Qold = Q;
//         hhqr.compute(Q);
//         Q = hhqr.householderQ().setLength(Q.cols()) * MatrixType::Identity(N, Q.cols());
//
//         VectorReal D = (Qold.conjugate().array() * Q.array()).colwise().sum().real();
//         VectorReal E = (D.cwiseAbs() - VectorType::Ones(D.size())).cwiseAbs();
//
//         // Start building HQ
//         if(vBlocks > 0) HQ.middleCols(vOffset * b, vBlocks * b) = HV;
//         if(mBlocks > 0) HQ.middleCols(mOffset * b, mBlocks * b) = MultHX(Q.middleCols(mOffset * b, mBlocks * b)); // No chance that this can be avoided
//         if(rBlocks > 0) HQ.middleCols(rOffset * b, rBlocks * b) = MultHX(Q.middleCols(rOffset * b, rBlocks * b)); // No chance that this can be avoided
//         if(kBlocks > 0) HQ.middleCols(kOffset * b, kBlocks * b) = HQ_keep;
//         if(qBlocks > 0) HQ.middleCols(qOffset * b, qBlocks * b) = MultHX(Q.middleCols(qOffset * b, qBlocks * b)); // Can't be avoided
//         // We may need to rebuild columns of HQ that were affected by QR,
//         // that is, if the orthonormalization of Q rotated those columns.
//         // To know for sure, we have to check E
//         for(Eigen::Index i = 0; i < Q.cols(); ++i) {
//             if(mOffset * b <= i && i < (mOffset + mBlocks) * b) continue; // Skip because it was already built
//             if(rOffset * b <= i && i < (rOffset + rBlocks) * b) continue; // Skip because it was already built
//             if(qOffset * b <= i && i < (qOffset + qBlocks) * b) continue; // Skip because it was already built
//             if(E(i) < normTolQ) {
//                 RealScalar sign = D(i) > 0 ? RealScalar{1} : RealScalar{-1};
//                 HQ.col(i) *= sign; // Q and HQ columns were preserved up to a sign
//             } else {
//                 HQ.col(i) = MultHX(Q.col(i)); // HQ column was not preserved in QR
//             }
//         }
//     } else if(oldBlocks != newBlocks) {
//         // Append enrichment
//         Q.conservativeResize(N, newBlocks * b);
//         HQ.conservativeResize(N, newBlocks * b);
//         auto copyBlocks              = std::min<Eigen::Index>(Q.cols() / b, Q_res_i.cols() / b);
//         Q.rightCols(copyBlocks * b)  = Q_res_i.leftCols(copyBlocks * b);
//         HQ.rightCols(copyBlocks * b) = MultHX(Q_res_i);
//     }
//
//     assert(Q.colwise().norm().minCoeff() > eps);
//     assert(HQ.colwise().norm().minCoeff() > eps);
//
//     // Form T
//     T = Q.adjoint() * HQ;
//     assert(T.colwise().norm().minCoeff() > eps);
//
//     if constexpr(settings::print_q) eig::log->warn("T : \n{}\n", linalg::matrix::to_string(T, 8));
//
//     // Solve T by calling diagonalizeT() elsewhere
// }
