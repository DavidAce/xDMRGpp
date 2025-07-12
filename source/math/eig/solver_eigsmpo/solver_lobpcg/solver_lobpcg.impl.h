#pragma once
#include "../solver_lobpcg.h"
#include "../StopReason.h"
#include "io/fmt_custom.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include <Eigen/Eigenvalues>
#include <spdlog/spdlog.h>
namespace settings {
#if defined(NDEBUG)
    constexpr bool debug_lobpcg = false;
#else
    constexpr bool debug_lobpcg = true;
#endif
    constexpr bool print_q = false;
}

template<typename Scalar>
void solver_lobpcg<Scalar>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
    auto from = matrix.middleCols(offset_old * b, extent * b);
    auto to   = matrix.middleCols(offset_new * b, extent * b);
    to        = from.eval();
}

template<typename Scalar>
void solver_lobpcg<Scalar>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
    for(int k = extent - 1; k > 0; --k) {
        // Takes [M0 | M1 | M2 | M3] to [M0 | M0 | M1| M2 ] So that we can overwrite M0 (the oldest)
        auto K0 = matrix.middleCols((offset + k + 0) * b, b);
        auto K1 = matrix.middleCols((offset + k - 1) * b, b);
        K0      = K1;
    }
}

template<typename Scalar>
std::pair<typename solver_lobpcg<Scalar>::VectorIdxT, typename solver_lobpcg<Scalar>::VectorIdxT> solver_lobpcg<Scalar>::selective_orthonormalize() {
    using Index = Eigen::Index;
    Index N     = Q.rows();
    Index k     = Q.cols();
    assert(k % b == 0 && "Q.cols() must be divisible by block size b.");
    Index n_blocks = k / b;

    // Compute Gram matrix

    MatrixType G = Q.adjoint() * Q;

    // auto Gnorm = (G - MatrixType::Identity(Q.cols(), Q.cols())).norm();
    eiglog->warn("G = \n{}\n", linalg::matrix::to_string(G, 8));

    // Identify blocks needing re-orthonormalization
    std::vector<Index> needs_reortho;
    for(Index blk = 0; blk < n_blocks; ++blk) {
        Index col_start = blk * b;
        bool  bad       = false;
        // Check against all previous blocks
        for(Index prev_blk = 0; prev_blk < blk; ++prev_blk) {
            Index prev_col_start = prev_blk * b;
            auto  G_block        = G.block(prev_col_start, col_start, b, b);
            if(G_block.cwiseAbs().maxCoeff() > orthTol) {
                bad = true;
                break;
            }
        }
        // Check internal orthonormality (diagonal block)
        if(!bad) {
            auto       G_diag = G.block(col_start, col_start, b, b);
            MatrixType I      = MatrixType::Identity(b, b);
            if((G_diag - I).cwiseAbs().maxCoeff() > normTol) { bad = true; }
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
        active_block_mask(blk) = Qk.norm() > normTol;
        change_block_mask(blk) = 1;
        // Orthonormalize block using HouseholderQR
        Eigen::HouseholderQR<MatrixType> hhqr(Qk);
        Qk = hhqr.householderQ().setLength(Qk.cols()) * MatrixType::Identity(N, b);
    }
    return {active_block_mask, change_block_mask};
}

template<typename Scalar>
void solver_lobpcg<Scalar>::build() {
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

    if constexpr(settings::print_q) eiglog->warn("Q after conservativeResize: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(status.iter == 0) {
        Q.leftCols(b) = V; // Copy the V panel as an initial guess
        // HQ.leftCols(b) = HV; // Copy the HV panel that was computed in init()
    } else {
        // eiglog->info("Q before resize: \n{}\n", linalg::matrix::to_string(Q,8));
        if(qBlocks_old != qBlocks and wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We made room for more qBlocks!
            // All the blocks after qBlocks must shift to the right
            Eigen::Index offset_old = qBlocks_old;
            Eigen::Index offset_new = qBlocks;
            Eigen::Index extent     = (wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eiglog->warn("Q after shifting W->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after shifting W->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(wBlocks_old != wBlocks and mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We changed the number of wBlocks!
            // All the blocks after wBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks;
            Eigen::Index extent     = std::min(mBlocks_old, mBlocks) + std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eiglog->warn("Q after shifting M->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after shifting M->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(mBlocks_old < mBlocks and sBlocks_old + rBlocks_old > 0) {
            // We changed the number of mBlocks!
            // All the blocks after mBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks;
            Eigen::Index extent     = std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eiglog->warn("Q after shifting S->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after shifting S->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        if(sBlocks_old < sBlocks and rBlocks_old > 0) {
            // We changed the number of sBlocks!
            // All the blocks after sBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks + sBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks + sBlocks;
            Eigen::Index extent     = std::min(rBlocks_old, rBlocks);
            shift_blocks_right(Q, offset_old, offset_new, extent);
            // shift_blocks_right(HQ, offset_old, offset_new, extent);
        }
        if constexpr(settings::print_q) eiglog->warn("Q after shifting R->: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after shifting R->: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll V_prev and V
        if(qBlocks == 2) {
            // Set V_prev(i) = V(i-1), where i is the iteration number
            Q.middleCols(0, b) = Q.middleCols(b, b);
            // HQ.middleCols(0, b) = HQ.middleCols(b, b);
        }
        assert(qBlocks >= 1);
        Q.middleCols((qBlocks - 1) * b, b) = V;
        // HQ.middleCols((qBlocks - 1) * b, b) = HV;
        if constexpr(settings::print_q) eiglog->warn("Q after rolling Q: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after rolling Q: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll W's
        roll_blocks_left(Q, qBlocks, wBlocks);
        // roll_blocks_left(HQ, qBlocks, wBlocks);

        if constexpr(settings::print_q) eiglog->warn("Q after rolling W: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after rolling W: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll M's
        roll_blocks_left(Q, qBlocks + wBlocks, mBlocks);
        // roll_blocks_left(HQ, qBlocks + wBlocks, mBlocks);

        if constexpr(settings::print_q) eiglog->warn("Q after rolling M: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after rolling M: \n{}\n", linalg::matrix::to_string(HQ, 8));

        // Roll S's
        roll_blocks_left(Q, qBlocks + wBlocks + mBlocks, sBlocks);
        // roll_blocks_left(HQ, qBlocks + wBlocks + mBlocks, sBlocks);

        if constexpr(settings::print_q) eiglog->warn("Q after rolling S: \n{}\n", linalg::matrix::to_string(Q, 8));
        if constexpr(settings::print_q) eiglog->warn("HQ after rolling S: \n{}\n", linalg::matrix::to_string(HQ, 8));
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

    auto MultP = [this](const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                        std::optional<const Eigen::Ref<const MatrixType>> iG) -> MatrixType {
        if(algo == OptAlgo::GDMRG)
            return this->MultP2(X, evals, iG);
        else
            return this->MultP(X, evals, iG);
    };

    /* clang-format off */
    if(wBlocks > 0) { Q.middleCols(wOffset * b, b) = get_wBlock(MultP); change_block_mask(wOffset) = 1; }
    if(mBlocks > 0) { Q.middleCols(mOffset * b, b) = get_mBlock();       change_block_mask(mOffset) = 1; }
    if(sBlocks > 0) { Q.middleCols(sOffset * b, b) = get_sBlock(S, MultP); change_block_mask(sOffset) = 1; }
    if(rBlocks > 0) { Q.middleCols(rOffset * b, b) = get_rBlock();       change_block_mask(rOffset) = 1; }
    /* clang-format off */


    if constexpr(settings::print_q) eiglog->warn("Q after preconditioning W,S,R: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eiglog->warn("HQ after preconditioning W,S,R: \n{}\n", linalg::matrix::to_string(HQ, 8));
    // eiglog->info("change_block_mask after preconditioner: {}", change_block_mask.transpose());


    // Run QR to orthonormalize [V, V_prev, W]
    // This lets the DGKS below to clean W, R and S quickly against [V, V_prev]
    hhqr.compute(Q.leftCols(qBlocks * b));
    Q.leftCols(qBlocks * b) = hhqr.householderQ().setLength(qBlocks * b) * MatrixType::Identity(N, qBlocks * b); //

    assert_allFinite(Q);

    // eiglog->warn("Q before DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));

    // pick a relative breakdown tolerance:


    // DGKS on each of X = M, W, R, S to remove overlap with [V, V_prev]

    OrthMeta mQL, mQR;
    mQL.maskTol = eps * 10 * std::max({status.max_eval_estimate()});
    mQR.maskTol = eps * 10 * std::max({status.max_eval_estimate()});
    MatrixType QL =  Q.leftCols(qBlocks * b);//[V_prev, V]
    MatrixType QR =  Q.rightCols((mBlocks + wBlocks + sBlocks + rBlocks) * b);// all the others
    MatrixType HQL = HQ.leftCols(qBlocks * b);
    MatrixType HQR = HQ.rightCols((mBlocks + wBlocks + sBlocks + rBlocks) * b);// all the others;
    block_l2_orthonormalize(QL, HQL, mQL);
    block_l2_orthogonalize(QL, HQL, QR, HQR, mQR);
    // Copy back
    Q.resize(Eigen::NoChange, QL.cols() + QR.cols());
    Q.leftCols(QL.cols()) = QL;
    Q.rightCols(QR.cols()) = QR;

    HQ.resize(Eigen::NoChange, HQL.cols() + HQR.cols());
    HQ.leftCols(HQL.cols()) = HQL;
    HQ.rightCols(HQR.cols()) = HQR;

    if constexpr(settings::print_q) eiglog->warn("Q after compression: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eiglog->warn("HQ after compression: \n{}\n", linalg::matrix::to_string(HQ, 8));

    assert_allFinite(Q);
    assert_l2_orthonormal(Q);

    // unset_HQ();

    // It's time to update HQ
    // for(Eigen::Index j = 0; j < change_block_mask.size(); ++j) {
    //     eiglog->info("Updating HQ block {}", j);
    //     HQ.middleCols(j * b, b) = MultHX(Q.middleCols(j * b, b));
    // }
    // HQ = MultH(Q);

    if constexpr(settings::print_q) eiglog->warn("Q after householder: \n{}\n", linalg::matrix::to_string(Q, 8));
    if constexpr(settings::print_q) eiglog->warn("HQ after householder: \n{}\n", linalg::matrix::to_string(HQ, 8));

    assert(Q.allFinite());
    assert(Q.colwise().norm().minCoeff() != 0);
    // Form the Gram Matrix
    // G          = Q.adjoint() * Q;
    // G          = RealScalar{0.5f} * (G + G.adjoint());
    // auto Gnorm = (G - MatrixType::Identity(Q.cols(), Q.cols())).norm();
    // eiglog->warn("G = \n{}\n", linalg::matrix::to_string(G, 8));
    // eiglog->warn("G = Q*Q before T: norm {:.5}", Gnorm);

    // Form T
    T = Q.adjoint() * HQ; // get_HQ(); // (blocks*b)×(blocks*b)
    assert(T.colwise().norm().minCoeff() != 0);
    if constexpr(settings::print_q) eiglog->warn("T : \n{}\n", linalg::matrix::to_string(T, 8));

    // Solve T by calling diagonalizeT() elsewhere
}

template<typename Scalar>
void solver_lobpcg<Scalar>::set_maxLanczosResidualHistory(Eigen::Index w) {
    w = std::clamp<Eigen::Index>(w, 0, N / b); // wBlocks are not required
    if(w != max_wBlocks) eiglog->debug("LOBPCG: max_wBlocks = {}", max_wBlocks);
    max_wBlocks = w;
}

template<typename Scalar>
void solver_lobpcg<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    m = std::clamp<Eigen::Index>(m, 0, N / b); // mBlocks are not required
    if(m != max_mBlocks) eiglog->debug("LOBPCG: max_mBlocks = {}", max_mBlocks);
    max_mBlocks                              = m;
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void solver_lobpcg<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    s = std::clamp<Eigen::Index>(s, 0, N / b); // sBlocks are not required
    if(s != max_sBlocks) eiglog->debug("RGB: max_sBlocks = {}", max_sBlocks);
    max_sBlocks = s;
}