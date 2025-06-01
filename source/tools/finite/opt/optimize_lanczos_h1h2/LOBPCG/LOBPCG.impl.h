#pragma once
#include "../LOBPCG.h"
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
    constexpr bool debug_lobpcg = false;
#else
    constexpr bool debug_lobpcg = true;
#endif
    constexpr bool print_q = false;
}

template<typename Scalar>
void LOBPCG<Scalar>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
    auto from = matrix.middleCols(offset_old * b, extent * b);
    auto to   = matrix.middleCols(offset_new * b, extent * b);
    to        = from.eval();
}

template<typename Scalar>
void LOBPCG<Scalar>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
    for(int k = extent - 1; k > 0; --k) {
        // Takes [M0 | M1 | M2 | M3] to [M0 | M0 | M1| M2 ] So that we can overwrite M0 (the oldest)
        auto K0 = matrix.middleCols((offset + k + 0) * b, b);
        auto K1 = matrix.middleCols((offset + k - 1) * b, b);
        K0      = K1;
    }
}

template<typename Scalar>
std::pair<typename LOBPCG<Scalar>::VectorIdxT, typename LOBPCG<Scalar>::VectorIdxT> LOBPCG<Scalar>::selective_orthonormalize() {
    using Index = Eigen::Index;
    Index N     = Q.rows();
    Index k     = Q.cols();
    assert(k % b == 0 && "Q.cols() must be divisible by block size b.");
    Index n_blocks = k / b;

    // Compute Gram matrix

    MatrixType G = Q.adjoint() * Q;

    // auto Gnorm = (G - MatrixType::Identity(Q.cols(), Q.cols())).norm();
    eig::log->warn("G = \n{}\n", linalg::matrix::to_string(G, 8));

    // Identify blocks needing re-orthonormalization
    std::vector<Index> needs_reortho;
    for(Index blk = 0; blk < n_blocks; ++blk) {
        Index col_start = blk * b;
        bool  bad       = false;
        // Check against all previous blocks
        for(Index prev_blk = 0; prev_blk < blk; ++prev_blk) {
            Index prev_col_start = prev_blk * b;
            auto  G_block        = G.block(prev_col_start, col_start, b, b);
            if(G_block.cwiseAbs().maxCoeff() > orthTolQ) {
                bad = true;
                break;
            }
        }
        // Check internal orthonormality (diagonal block)
        if(!bad) {
            auto       G_diag = G.block(col_start, col_start, b, b);
            MatrixType I      = MatrixType::Identity(b, b);
            if((G_diag - I).cwiseAbs().maxCoeff() > normTolQ) { bad = true; }
        }

        if(bad) needs_reortho.push_back(blk);
    }

    VectorIdxT active_block_mask = VectorIdxT::Ones(n_blocks);
    VectorIdxT change_block_mask = VectorIdxT::Zero(n_blocks);
    // Re-orthonormalize bad blocks
    for(Index blk : needs_reortho) {
        Index col_start = blk * b;
        auto  Qk        = Q.middleCols(col_start, b);

        // Orthogonalize Qk against previous blocks
        for(Index prev_blk = 0; prev_blk < blk; ++prev_blk) {
            Index prev_col_start = prev_blk * b;
            auto  Qj             = Q.middleCols(prev_col_start, b);
            // Project and subtract
            MatrixType proj = Qj.adjoint() * Qk;
            Qk -= Qj * proj;
        }
        active_block_mask(blk) = Qk.norm() > normTolQ;
        change_block_mask(blk) = 1;
        // Orthonormalize block using HouseholderQR
        Eigen::HouseholderQR<MatrixType> hhqr(Qk);
        Qk = hhqr.householderQ().setLength(Qk.cols()) * MatrixType::Identity(N, b);
    }
    return {active_block_mask, change_block_mask};
}

template<typename Scalar>
void LOBPCG<Scalar>::build() {
    const Eigen::Index N = H1.rows();
    // Eigen::Index       max_steps = std::min<Eigen::Index>(ncv, N);
    set_maxLanczosResidualHistory(max_wBlocks);
    set_maxExtraRitzHistory(max_mBlocks);
    set_maxRitzResidualHistory(max_sBlocks);
    assert(V.cols() == b);
    // assert(max_wBlocks + max_sBlocks >= 1); // we need at least one of them for basis exhaustion tests

    // Now V has b orthonormalized ritz vectors

    // Start defining the blocks of Q
    Eigen::Index qBlocks_old = qBlocks;
    Eigen::Index wBlocks_old = wBlocks;
    Eigen::Index mBlocks_old = mBlocks;
    Eigen::Index sBlocks_old = sBlocks;
    Eigen::Index rBlocks_old = rBlocks;

    auto get_total_blocks = [&]() { return qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; };

    qBlocks = status.iter == 0 ? 1 : 2;                                   // For V_prev, V
    wBlocks = std::min(max_wBlocks, wBlocks_old + 1);                     // Add space for one more W block
    mBlocks = M.cols() == b ? std::min(max_mBlocks, mBlocks_old + 1) : 0; // Add space for one more M block
    sBlocks = std::min(max_sBlocks, sBlocks_old + 1);                     // Add space for one more S block
    rBlocks = (inject_randomness and status.iter > 20 and status.iter % 20 == 0 and get_total_blocks() * b <= N) ? 1 : 0;

    // Try to keep W and S if possible, drop R, M first
    while(N < get_total_blocks() * b) {
        /* clang-format off */
        if(rBlocks > 0) { rBlocks--; continue; }
        if(mBlocks > 0) { mBlocks--; continue; }
        if(wBlocks > 0) { wBlocks--; continue; }
        if(sBlocks > 1) { sBlocks--; continue; }
        break; // If all are at min, break to avoid infinite loop
        /* clang-format on */
    }

    // if(get_total_blocks() * b != Q.cols()) { unset_HQ(); }
    Q.conservativeResize(N, (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);
    // HQ.conservativeResize(N, (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);

    if constexpr(settings::print_q) eig::log->warn("Q after conservativeResize: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(status.iter == 0) {
        Q.leftCols(b) = V; // Copy the V panel as an initial guess
        // HQ.leftCols(b) = HV; // Copy the HV panel that was computed in init()
    } else {
        // eig::log->info("Q before resize: \n{}\n", linalg::matrix::to_string(Q,8));
        if(qBlocks_old != qBlocks and wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We made room for more qBlocks!
            // All the blocks after qBlocks must shift to the right
            Eigen::Index offset_old = qBlocks_old;
            Eigen::Index offset_new = qBlocks;
            Eigen::Index extent     = (wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting W->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after shifting W->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(wBlocks_old != wBlocks and mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We changed the number of wBlocks!
            // All the blocks after wBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks;
            Eigen::Index extent     = std::min(mBlocks_old, mBlocks) + std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting M->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after shifting M->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(mBlocks_old < mBlocks and sBlocks_old + rBlocks_old > 0) {
            // We changed the number of mBlocks!
            // All the blocks after mBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks;
            Eigen::Index extent     = std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting S->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after shifting S->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(sBlocks_old < sBlocks and rBlocks_old > 0) {
            // We changed the number of sBlocks!
            // All the blocks after sBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks + sBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks + sBlocks;
            Eigen::Index extent     = std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting R->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after shifting R->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll V_prev and V
        if(qBlocks == 2) {
            // Set V_prev(i) = V(i-1), where i is the iteration number
            Q.middleCols(0, b) = Q.middleCols(b, b);
            // HQ.middleCols(0, b) = HQ.middleCols(b, b);
        }
        assert(qBlocks >= 1);
        Q.middleCols((qBlocks - 1) * b, b) = V;
        // HQ.middleCols((qBlocks - 1) * b, b) = HV;
        if constexpr(settings::print_q) eig::log->warn("Q after rolling Q: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after rolling Q: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll W's
        roll_blocks_left(Q, qBlocks, wBlocks);
        // roll_blocks_left(HQ, qBlocks, wBlocks);

        if constexpr(settings::print_q) eig::log->warn("Q after rolling W: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after rolling W: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll M's
        roll_blocks_left(Q, qBlocks + wBlocks, mBlocks);
        // roll_blocks_left(HQ, qBlocks + wBlocks, mBlocks);

        if constexpr(settings::print_q) eig::log->warn("Q after rolling M: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after rolling M: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll S's
        roll_blocks_left(Q, qBlocks + wBlocks + mBlocks, sBlocks);
        // roll_blocks_left(HQ, qBlocks + wBlocks + mBlocks, sBlocks);

        if constexpr(settings::print_q) eig::log->warn("Q after rolling S: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eig::log->warn("HQ after rolling S: \n{}\n", linalg::matrix::to_string(HQ, 8));
    }
    assert(Q.cols() == (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);
    // assert(HQ.cols() == (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);
    // MatrixType G_old_for_HQ = Q.adjoint() * Q;

    /*! Main LOBPCG step.
        In LOBPCG loop, we always update three basis blocks at a time [V_prev, V, and Q_next] = LOPBCG (V_prev, V, W)
    */
    // Eigen::Index i = std::min<Eigen::Index>(status.iter, 1);

    // Eigen::Index qOffset = 0;
    Eigen::Index wOffset = qBlocks;
    Eigen::Index mOffset = qBlocks + wBlocks;
    Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
    Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;

    // Keep track of which Q blocks change, so we know which HQ blocks to recompute later
    VectorIdxT change_block_mask = VectorIdxT::Ones(qBlocks + wBlocks + mBlocks + sBlocks + rBlocks);
    /* clang-format off */
    if(wBlocks > 0) { Q.middleCols(wOffset * b, b) = get_wBlock(); change_block_mask(wOffset) = 1; }
    if(mBlocks > 0) { Q.middleCols(mOffset * b, b) = get_mBlock(); change_block_mask(mOffset) = 1; }
    if(sBlocks > 0) { Q.middleCols(sOffset * b, b) = get_sBlock(); change_block_mask(sOffset) = 1; }
    if(rBlocks > 0) { Q.middleCols(rOffset * b, b) = get_rBlock(); change_block_mask(rOffset) = 1; }
    /* clang-format off */

    //
    // if(wBlocks > 0) {
    //     const auto V_prev  = i == 0 ? Q.middleCols(0, 0) : Q.middleCols(0, b);
    //     const auto V   = Q.middleCols(V_prev.cols(), b);
    //     const auto HV_prev = i == 0 ? HQ.middleCols(0, 0) : HQ.middleCols(0, b);
    //     const auto HV  = HQ.middleCols(HV_prev.cols(), b);
    //
    //     auto W  = Q.middleCols(qBlocks * b, b);
    //     auto HW = HQ.middleCols(qBlocks * b, b);
    //     // We add Lanczos-style residual blocks
    //     W = HV;
    //     assert(W.allFinite());
    //     A                    = V.adjoint() * W;
    //     B                    = V_prev.adjoint() * W;
    //     status.H_norm_approx = std::max({
    //         status.H_norm_approx,                          //
    //         A.norm() * std::abs(std::sqrt<RealScalar>(b)), //
    //         B.norm() * std::abs(std::sqrt<RealScalar>(b))  //
    //     });
    //
    //     // 3) Subtract projections to A and B once
    //     W.noalias() -= V * A; // Qi * Qi.adjoint()*H*Qi
    //     if(i > 0) { W.noalias() -= V_prev * B.adjoint(); }
    //     HW = MultHX(W); // Update HW also
    // }
    // if constexpr(settings::print_q) eig::log->warn("Q after filling W: \n{}\n", linalg::matrix::to_string(Q, 8));
    // if constexpr(settings::print_q) eig::log->warn("HQ after filling W: \n{}\n", linalg::matrix::to_string(HQ, 8));
    //
    // // Inject additional ritz vectors in the GD+k style
    // if(mBlocks > 0) {
    //     // M are the b next-best ritz vectors from the previous iteration
    //     auto M_new                           = Q.middleCols((qBlocks + wBlocks) * b, b);
    //     M_new                                = M;
    //     change_block_mask(qBlocks + wBlocks) = 1;
    // }
    // if constexpr(settings::print_q) eig::log->warn("Q after filling M: \n{}\n", linalg::matrix::to_string(Q, 8));
    // if constexpr(settings::print_q) eig::log->warn("HQ after filling M: \n{}\n", linalg::matrix::to_string(HQ, 8));
    //
    // if(sBlocks > 0) {
    //     // We add a residual block "S = (HQ-λQ)" Hold on with HS
    //     const auto V  = Q.middleCols((qBlocks - 1) * b, b);
    //     const auto HV = HQ.middleCols((qBlocks - 1) * b, b);
    //     auto       S      = Q.middleCols((qBlocks + wBlocks + mBlocks) * b, b);   // Residual block
    //     S                 = HV - V * T_evals(status.optIdx).asDiagonal(); // Put the residual "R" in S.
    // }
    // if(rBlocks > 0) {
    //     // We add a residual block "S = (HQ-λQ)". Hold on with HS until after preconditioning
    //     auto R = Q.middleCols((qBlocks + wBlocks + mBlocks + sBlocks) * b, b);
    //     R.setRandom();
    //     change_block_mask(qBlocks + wBlocks + mBlocks + sBlocks) = 1;
    // }
    //
    // if constexpr(settings::print_q) eig::log->warn("Q after filling S: \n{}\n", linalg::matrix::to_string(Q, 8));
    // if constexpr(settings::print_q) eig::log->warn("HQ after filling S: \n{}\n", linalg::matrix::to_string(HQ, 8));
    if constexpr(settings::print_q) eig::log->warn("Q after preconditioning W,S,R: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eig::log->warn("HQ after preconditioning W,S,R: \n{}\n", linalg::matrix::to_string(HQ, 8));
    // eig::log->info("change_block_mask after preconditioner: {}", change_block_mask.transpose());
    if(chebyshev_filter_degree >= 1) {
        // Apply the chebyshev filter on newly generated residual and random blocks
        auto W = wBlocks > 0 ? Q.middleCols(wOffset * b, b) : Q.middleCols(wOffset * b, 0);
        auto S = sBlocks > 0 ? Q.middleCols(sOffset * b, b) : Q.middleCols(sOffset * b, 0);
        auto R = rBlocks > 0 ? Q.middleCols(rOffset * b, b) : Q.middleCols(rOffset * b, 0);

        /* clang-format off */
        if(wBlocks > 0) {W = qr_and_chebyshevFilter(W); change_block_mask(wOffset) = 1;}
        if(sBlocks > 0) {S = qr_and_chebyshevFilter(S); change_block_mask(sOffset) = 1;}
        if(rBlocks > 0) {R = qr_and_chebyshevFilter(R); change_block_mask(rOffset) = 1;}
        /* clang-format on */
    }
    if(use_preconditioner) {
        // Precondition the latest W, S and R,
        /* clang-format off */
        if(wBlocks > 0) {Q.middleCols(wOffset * b, b) = -MultPX(Q.middleCols(wOffset * b, b)); change_block_mask(wOffset) = 1;}
        if(sBlocks > 0) {Q.middleCols(sOffset * b, b) = -MultPX(Q.middleCols(sOffset * b, b)); change_block_mask(sOffset) = 1;}
        if(rBlocks > 0) {Q.middleCols(rOffset * b, b) = -MultPX(Q.middleCols(rOffset * b, b)); change_block_mask(rOffset) = 1;}
        /* clang-format on */
    }

    if constexpr(settings::print_q) eig::log->warn("Q after chebyshev W,S,R: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eig::log->warn("HQ after chebyshev W,S,R: \n{}\n", linalg::matrix::to_string(HQ, 8));

    // Run QR to orthonormalize [V, V_prev, W]
    // This lets the DGKS below to clean W, R and S quickly against [V, V_prev]
    hhqr.compute(Q.leftCols(qBlocks * b));
    Q.leftCols(qBlocks * b) = hhqr.householderQ().setLength(qBlocks * b) * MatrixType::Identity(N, qBlocks * b); //
    change_block_mask.middleRows(0, qBlocks).setOnes();

    assert_allfinite(Q);

    // eig::log->warn("Q before DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));

    // pick a relative breakdown tolerance:
    auto       breakdownTol      = eps * 10 * std::max({status.H_norm_est()});
    VectorIdxT active_block_mask = VectorIdxT::Ones(qBlocks + wBlocks + mBlocks + sBlocks + rBlocks);

    assert(active_block_mask.size() * b == Q.cols());
    assert(change_block_mask.size() * b == Q.cols());

    // DGKS on each of X = M, W, R, S to remove overlap with [V, V_prev]
    orthonormalize(Q.leftCols(qBlocks * b),                                  //[V_prev, V]
                   Q.rightCols((mBlocks + wBlocks + sBlocks + rBlocks) * b), // all the others
                   breakdownTol,                                             // For block normalization checks
                   1000 * breakdownTol,                                      // for pairwise block orthonormalization checks
                   active_block_mask.bottomRows(mBlocks + wBlocks + sBlocks + rBlocks));

    compress_cols(Q, active_block_mask);
    // compress_cols(HQ, active_block_mask);
    // compress_rows_and_cols(G_old_for_HQ, active_block_mask);
    // compress_rows_and_cols(G, active_block_mask);

    // MatrixType G_diff_for_HQ = (Q.adjoint() * Q - G_old_for_HQ).cwiseAbs();
    // eig::log->info("G diff: \n{}\n", linalg::matrix::to_string(G_diff_for_HQ,8));

    // compress(HQ, active_block_mask);

    // for(int k = qBlocks; k < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++k) {
    //     auto Xk        = Q.middleCols(k * b, b);
    //     auto XkNormNew = Xk.norm();
    //     for(int rep = 0; rep < 10; ++rep) { // two DGKS passes
    //         if(active_block_mask(k) == 0) continue;
    //         auto XjXknormMax = RealScalar{0};
    //         for(Eigen::Index j = 0; j < k; ++j) { // Clean every Xk against Qj...Q{k-1} V only
    //             if(active_block_mask(j) == 0) continue;
    //             // if(change_block_mask(k) == 0) continue;
    //             auto       Xj   = Q.middleCols(j * b, b);
    //             MatrixType XjXk = (Xj.adjoint() * Xk);
    //             Xk -= Xj * XjXk;
    //             // if(k == qBlocks + wBlocks + mBlocks + sBlocks + rBlocks - 1) {
    //             //     eig::log->info("|X{}*(X{}.adjoint()*S0)|={:.5e} new S0 norm: {:.5e}",j, j, XjXk.norm(), Xk.norm());
    //             // }
    //             auto XjXknorm = XjXk.norm();
    //             XjXknormMax   = std::max(XjXknorm, XjXknorm);
    //         }
    //         XkNormNew = Xk.norm();
    //         if(XkNormNew < breakdownTol) {
    //             // This Wk has been zeroed out! Disable and go to next
    //             active_block_mask(k) = 0;
    //             eig::log->info("active_block_mask: {}", active_block_mask.transpose());
    //             break;
    //         }
    //         // eig::log->info("max overlap Q(0...{}).adjoint() * X({}) = {:.16f} rep {}", k-1, k, QjXknorm, rep);
    //         if(XjXknormMax < normTolQ) break; // Orthonormal enough, go to next Wk
    //     }
    //
    //     // auto XknewXknorm = (Xk_new.normalized().adjoint() * Xk.normalized()).norm();
    //     // eig::log->info("change_block_mask DGKS: {} {} | XknewXkNorm {:.5e}", k, change_block_mask.transpose() , XknewXknorm);
    //     // if(std::abs(XknewXknorm - 1) > orthTolQ or XkNormNew > normTolQ) {
    //     //     change_block_mask(k) = 1;
    //     //
    //     // }
    //
    //     //     // We have now cleaned Xk. Let's orthonormalize it so we can use it to clean others
    //     hhqr.compute(Xk);
    //     Xk = hhqr.householderQ().setLength(Xk.cols()) * MatrixType::Identity(N, b);
    // }

    // std::tie(active_block_mask, change_block_mask) = selective_orthonormalize();
    // eig::log->info("change_block_mask after selective_orthonormalize: {}", change_block_mask.transpose());
    // eig::log->info("active_block_mask after selective_orthonormalize: {}", active_block_mask.transpose());

    // change_block_mask = selective_orthonormalize(change_block_mask, orthTolQ);
    // eig::log->info("Reortho: {}", reortho);
    // if constexpr(settings::debug_lobpcg) {
    //     // W's should not have overlap with previous blocks
    //     for(int k = qBlocks; k < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++k) {
    //         if(active_block_mask[k] == 0) continue;
    //         const auto Xk = Q.middleCols(k * b, b);
    //         for(Eigen::Index j = 0; j < k; ++j) {
    //             if(active_block_mask[j] == 0) continue;
    //             const auto Xj       = Q.middleCols(j * b, b);
    //             auto       XjXknorm = (Xj.adjoint() * Xk).norm();
    //             if(XjXknorm > orthTolQ) {
    //                 // eig::log->info("Q after DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));
    //                 eig::log->warn("overlap X({}).adjoint() * X({}) = {:.16f} ", j, k, XjXknorm);
    //                 // throw except::runtime_error("overlap Q({}).adjoint() * W = {:.16f} ", j, QjWnorm);
    //             }
    //         }
    //     }
    // }

    // Compress Q if any block got zeroed
    // assert(active_block_mask.size() * b == Q.cols());
    // Eigen::Index num_active_blocks = std::accumulate(active_block_mask.begin(), active_block_mask.end(), 0);
    // if(num_active_blocks < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks)
    // eig::log->info("should compress {} < {}", num_active_blocks, qBlocks + wBlocks + mBlocks + sBlocks + rBlocks);
    // if(num_active_blocks < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) {
    //     Eigen::Index qOffset = 0;
    //     Eigen::Index wOffset = qBlocks;
    //     Eigen::Index mOffset = qBlocks + wBlocks;
    //     Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
    //     Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;
    //
    //     // Check if that the new search directions in the latest W and S have nonzero norm nonzero
    //     const auto W = wBlocks > 0 ? Q.middleCols(wOffset * b, b) : Q.middleCols(wOffset * b, 0);
    //     const auto S = sBlocks > 0 ? Q.middleCols(sOffset * b, b) : Q.middleCols(sOffset * b, 0);
    //
    //     auto minWnorm = wBlocks > 0 ? W.colwise().norm().minCoeff() : 1;
    //     auto minSnorm = sBlocks > 0 ? S.colwise().norm().minCoeff() : 1;
    //
    //     bool isZeroWnorm = wBlocks > 0 ? minWnorm < breakdownTol : false;
    //     bool isZeroSnorm = sBlocks > 0 ? minSnorm < breakdownTol : false;
    //     if(isZeroWnorm or isZeroSnorm) {
    //         // Happy breakdown, reached the invariant subspace:
    //         // We could not add new search directions
    //         eig::log->warn("optVal {::.16f}: ", fv(status.optVal));
    //         eig::log->warn("T_evals {::.16f}: ", fv(T_evals));
    //         eig::log->debug("saturated basis");
    //         status.stopReason |= StopReason::saturated_basis;
    //         status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
    //         return;
    //     }
    //
    //     // We can now squeeze out blocks zeroed out by DGKS
    //     // Get the block indices that we should keep
    //     std::vector<Eigen::Index> active_blocks;
    //     std::vector<Eigen::Index> active_columns;
    //     active_blocks.reserve((qBlocks + wBlocks + mBlocks + sBlocks + rBlocks));
    //     active_columns.reserve((qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);
    //     for(Eigen::Index j = 0; j < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++j) {
    //         if(active_block_mask(j) == 1) {
    //             for(Eigen::Index k = 0; k < b; ++k) active_columns.push_back(j * b + k);
    //             active_blocks.push_back(j);
    //         }
    //     }
    //     active_columns.shrink_to_fit();
    //     // eig::log->info("active_columns: {}", active_columns);
    //     assert(Q.cols() == HQ.cols());
    //     if(active_columns.size() != static_cast<size_t>(Q.cols())) {
    //         Q                 = Q(Eigen::all, active_columns).eval();  // Shrink keeping only nonzeros
    //         HQ                = HQ(Eigen::all, active_columns).eval(); // Shrink keeping only nonzeros
    //         change_block_mask = change_block_mask(active_blocks);
    //
    //         Eigen::Index qBlocks_new = std::accumulate(active_block_mask.begin() + qOffset, active_block_mask.begin() + qOffset + qBlocks, 0);
    //         Eigen::Index wBlocks_new = std::accumulate(active_block_mask.begin() + wOffset, active_block_mask.begin() + wOffset + wBlocks, 0);
    //         Eigen::Index mBlocks_new = std::accumulate(active_block_mask.begin() + mOffset, active_block_mask.begin() + mOffset + mBlocks, 0);
    //         Eigen::Index sBlocks_new = std::accumulate(active_block_mask.begin() + sOffset, active_block_mask.begin() + sOffset + sBlocks, 0);
    //         Eigen::Index rBlocks_new = std::accumulate(active_block_mask.begin() + rOffset, active_block_mask.begin() + rOffset + rBlocks, 0);
    //
    //         if(qBlocks_new != qBlocks) eig::log->info("new qBlocks : {} -> {}", qBlocks, qBlocks_new);
    //         if(wBlocks_new != wBlocks) eig::log->info("new wBlocks : {} -> {}", wBlocks, wBlocks_new);
    //         if(mBlocks_new != mBlocks) eig::log->info("new mBlocks : {} -> {}", mBlocks, mBlocks_new);
    //         if(sBlocks_new != sBlocks) eig::log->info("new sBlocks : {} -> {}", sBlocks, sBlocks_new);
    //         if(rBlocks_new != rBlocks) eig::log->info("new rBlocks : {} -> {}", rBlocks, rBlocks_new);
    //
    //         qBlocks = qBlocks_new;
    //         wBlocks = wBlocks_new;
    //         mBlocks = mBlocks_new;
    //         sBlocks = sBlocks_new;
    //         rBlocks = rBlocks_new;
    //     }
    // }

    if constexpr(settings::print_q) eig::log->warn("Q after compression: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eig::log->warn("HQ after compression: \n{}\n", linalg::matrix::to_string(HQ, 8));

    assert_allfinite(Q);
    assert_orthonormal(Q, normTolQ);

    hhqr.compute(Q);
    Q = hhqr.householderQ().setLength(Q.cols()) * MatrixType::Identity(N, Q.cols()); //

    assert_allfinite(Q);
    assert_orthonormal(Q, normTolQ);

    // unset_HQ();

    // It's time to update HQ
    // for(Eigen::Index j = 0; j < change_block_mask.size(); ++j) {
    //     eig::log->info("Updating HQ block {}", j);
    //     HQ.middleCols(j * b, b) = MultHX(Q.middleCols(j * b, b));
    // }
    HQ = MultHX(Q);

    if constexpr(settings::print_q) eig::log->warn("Q after householder: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eig::log->warn("HQ after householder: \n{}\n", linalg::matrix::to_string(HQ, 8));

    assert(Q.allFinite());
    assert(Q.colwise().norm().minCoeff() != 0);
    // Form the Gram Matrix
    // G          = Q.adjoint() * Q;
    // G          = RealScalar{0.5f} * (G + G.adjoint());
    // auto Gnorm = (G - MatrixType::Identity(Q.cols(), Q.cols())).norm();
    // eig::log->warn("G = \n{}\n", linalg::matrix::to_string(G, 8));
    // eig::log->warn("G = Q*Q before T: norm {:.5}", Gnorm);

    // Form T
    T = Q.adjoint() * HQ; // get_HQ(); // (blocks*b)×(blocks*b)
    assert(T.colwise().norm().minCoeff() != 0);
    if constexpr(settings::print_q) eig::log->warn("T : \n{}\n", linalg::matrix::to_string(T, 8));

    // Solve T by calling diagonalizeT() elsewhere
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxLanczosResidualHistory(Eigen::Index w) {
    w = std::clamp<Eigen::Index>(w, 0, N / b); // wBlocks are not required
    if(w != max_wBlocks) eig::log->debug("LOBPCG: max_wBlocks = {}", max_wBlocks);
    max_wBlocks = w;
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    m = std::clamp<Eigen::Index>(m, 0, N / b); // mBlocks are not required
    if(m != max_mBlocks) eig::log->debug("LOBPCG: max_mBlocks = {}", max_mBlocks);
    max_mBlocks                              = m;
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    s = std::clamp<Eigen::Index>(s, 0, N / b); // sBlocks are not required
    if(s != max_sBlocks) eig::log->debug("RGB: max_sBlocks = {}", max_sBlocks);
    max_sBlocks = s;
}