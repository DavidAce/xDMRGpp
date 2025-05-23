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
typename LOBPCG<Scalar>::MatrixType LOBPCG<Scalar>::chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       // input Q (orthonormal)
                                                                    RealScalar                          lambda_min, // estimated smallest eigenvalue
                                                                    RealScalar                          lambda_max, // estimated largest eigenvalue
                                                                    RealScalar                          lambda_cut, // cut-off (e.g. λmin for low-end)
                                                                    int                                 degree      // polynomial degree k,
) {
    if(Qref.cols() == 0) { return Qref; }
    if(degree == 0) { return Qref; }

    int N = Qref.rows();

    // Map spectrum [λ_min, λ_max] to [-1,1]
    RealScalar av = (lambda_max + lambda_min) / RealScalar{2};
    RealScalar bv = (lambda_max - lambda_min) / RealScalar{2};

    RealScalar safety = RealScalar{1e-8f} * (lambda_max - lambda_min);
    lambda_cut        = std::clamp(lambda_cut, lambda_min + safety, lambda_max - safety);

    if(lambda_cut != std::clamp(lambda_cut, lambda_min, lambda_max)) {
        eig::log->warn("lambda_cut outside range [lambda_min, lambda_max]");
        return Qref;
    }
    if(bv < eps * std::abs(av)) {
        eig::log->warn("bv < eps");
        return Qref;
    }
    // eig::log->info("lambdas: {:.5e} {:.5e} {:.5e}", lambda_min, lambda_cut, lambda_max);

    // Recurrence: T0 = Qref, T1 = ((H - a*I)/bspec)*Qref

    if(degree == 1) { return (MultHX(Qref) - av * Qref) * (RealScalar{1} / bv); }

    RealScalar x0 = (lambda_cut - av) / bv;

    // Clamp x0 into [-1,1] to avoid NaN
    x0              = std::clamp(x0, RealScalar{-1}, RealScalar{1});
    RealScalar norm = std::cos(degree * std::acos(x0)); // = T_k(x0)

    if(std::abs(norm) < eps or !std::isfinite(norm)) {
        // normalization too small; skip filtering
        eig::log->warn("norm invalid {:.5e}", norm);
        return Qref;
    }

    // Chebyshev recurrence: T_k = 2*( (H - aI)/bspec ) T_{k-1} - T_{k-2}
    MatrixType Tkm2 = Qref;
    MatrixType Tkm1 = (MultHX(Qref) - av * Qref) * (RealScalar{1} / bv);
    MatrixType Tcur(N, Qref.cols());
    for(int k = 2; k <= degree; ++k) {
        Tcur = (MultHX(Tkm1) - av * Tkm1) * (RealScalar{2} / bv) - Tkm2;
        assert(std::isfinite(Tcur.norm()));
        Tkm2 = std::move(Tkm1);
        Tkm1 = std::move(Tcur);
    }
    return Tkm1 * (Scalar{1} / norm);
}

template<typename Scalar>
typename LOBPCG<Scalar>::MatrixType LOBPCG<Scalar>::qr_and_chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       // input Q (orthonormal)
                                                                           RealScalar                          lambda_min, // estimated smallest eigenvalue
                                                                           RealScalar                          lambda_max, // estimated largest eigenvalue
                                                                           RealScalar                          lambda_cut, // cut-off (e.g. λmin for low-end)
                                                                           int                                 degree      // polynomial degree k,
) {
    if(Qref.cols() == 0) return Qref;
    if(degree == 0) return Qref;

    // Re orthogonalize
    MatrixType Qnew = Qref;
    hhqr.compute(Qnew);
    Qnew = hhqr.householderQ() * MatrixType::Identity(N, Qnew.cols()); //
    return chebyshevFilter(Qnew, lambda_min, lambda_max, lambda_cut, degree);
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

    qBlocks = status.iter == 0 ? 1 : 2;                                   // For Q_prev, Q_cur
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

    if(get_total_blocks() * b != Q.cols()) { unset_HQ(); }
    Q.conservativeResize(N, (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);

    if constexpr(settings::print_q) eig::log->warn("Q after conservativeResize: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(status.iter == 0) {
        Q.leftCols(b) = V; // Copy the V panel as an initial guess
        unset_HQ();
    } else {
        // eig::log->info("Q before resize: \n{}\n", linalg::matrix::to_string(Q,8));
        if(qBlocks_old != qBlocks and wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We made room for more qBlocks!
            // All the blocks after qBlocks must shift to the right
            auto from = Q.middleCols(qBlocks_old * b, (wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old) * b);
            auto to   = Q.middleCols(qBlocks * b, (wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old) * b);
            to        = from.eval();
            unset_HQ();
            // if(qBlocks > qBlocks_old) Q.middleCols(qBlocks_old * b, (qBlocks - qBlocks_old) * b).setZero();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting W->: \n{}\n", linalg::matrix::to_string(Q, 8));

        if(wBlocks_old != wBlocks and mBlocks_old + sBlocks_old + rBlocks_old > 0) {
            // We changed the number of wBlocks!
            // All the blocks after wBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks;
            Eigen::Index extent     = std::min(mBlocks_old, mBlocks) + std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            auto         from       = Q.middleCols(offset_old * b, extent * b);
            auto         to         = Q.middleCols(offset_new * b, extent * b);
            to                      = from.eval();
            unset_HQ();
            // if(wBlocks > wBlocks_old) Q.middleCols(offset_old * b, (wBlocks - wBlocks_old) * b).setZero();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting M->: \n{}\n", linalg::matrix::to_string(Q, 8));

        if(mBlocks_old < mBlocks and sBlocks_old + rBlocks_old > 0) {
            // We changed the number of mBlocks!
            // All the blocks after mBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks;
            Eigen::Index extent     = std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks);
            auto         from       = Q.middleCols(offset_old * b, extent * b);
            auto         to         = Q.middleCols(offset_new * b, extent * b);
            to                      = from.eval();
            unset_HQ();
            // if(mBlocks > mBlocks_old) Q.middleCols(offset_old * b, (mBlocks - mBlocks_old) * b).setZero();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting S->: \n{}\n", linalg::matrix::to_string(Q, 8));

        if(sBlocks_old < sBlocks and rBlocks_old > 0) {
            // We changed the number of sBlocks!
            // All the blocks after sBlocks must shift accordingly
            Eigen::Index offset_old = qBlocks + wBlocks + mBlocks + sBlocks_old;
            Eigen::Index offset_new = qBlocks + wBlocks + mBlocks + sBlocks;
            Eigen::Index extent     = std::min(rBlocks_old, rBlocks);
            auto         from       = Q.middleCols(offset_old * b, extent * b);
            auto         to         = Q.middleCols(offset_new * b, extent * b);
            to                      = from.eval();
            unset_HQ();
            // if(sBlocks > sBlocks_old) Q.middleCols(offset_old * b, (sBlocks - sBlocks_old) * b).setZero();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after shifting R->: \n{}\n", linalg::matrix::to_string(Q, 8));

        // Roll Q_prev and Q_cur
        if(qBlocks == 2) {
            // Set Q_prev(i) = Q_cur(i-1), where i is the iteration number
            Q.middleCols(0, b) = Q.middleCols(b, b);
            unset_HQ();
        }
        assert(qBlocks >= 1);
        Q.middleCols((qBlocks - 1) * b, b) = V;
        unset_HQ_cur();

        if constexpr(settings::print_q) eig::log->warn("Q after rolling Q: \n{}\n", linalg::matrix::to_string(Q, 8));

        // Roll W's
        Eigen::Index wOffset = qBlocks; // Thanks to the shift, this is always true
        for(int k = wBlocks - 1; k > 0; --k) {
            // Takes [W0 | W1 | W2 | W3] to [W0 | W0 | W1| W2 ] So that we can overwrite W0 (the oldest)
            auto Wk0 = Q.middleCols((wOffset + k + 0) * b, b);
            auto Wk1 = Q.middleCols((wOffset + k - 1) * b, b);
            Wk0      = Wk1;
            unset_HQ();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after rolling W: \n{}\n", linalg::matrix::to_string(Q, 8));

        // Roll M's
        Eigen::Index mOffset = qBlocks + wBlocks; // Thanks to the shift, this is always true
        for(int k = mBlocks - 1; k > 0; --k) {
            // Takes [M0 | M1 | M2 | M3] to [M0 | M0 | M1| M2 ] So that we can overwrite X0 (the oldest)
            auto Mk0 = Q.middleCols((mOffset + k + 0) * b, b);
            auto Mk1 = Q.middleCols((mOffset + k - 1) * b, b);
            Mk0      = Mk1;
            unset_HQ();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after rolling M: \n{}\n", linalg::matrix::to_string(Q, 8));

        // Roll S's
        Eigen::Index sOffset = qBlocks + wBlocks + mBlocks; // Thanks to the shift, this is always true
        for(int k = sBlocks - 1; k > 0; --k) {
            // Takes [S0 | S1 | S2 | S3] to [S0 | S0 | S1| S2 ] So that we can overwrite X0 (the oldest)
            auto Sk0 = Q.middleCols((sOffset + k + 0) * b, b);
            auto Sk1 = Q.middleCols((sOffset + k - 1) * b, b);
            Sk0      = Sk1;
            unset_HQ();
        }
        if constexpr(settings::print_q) eig::log->warn("Q after rolling S: \n{}\n", linalg::matrix::to_string(Q, 8));
    }
    assert(Q.cols() == (qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);

    /*! Main LOBPCG step.
        In LOBPCG loop, we always update three basis blocks at a time [Q_prev, Q_cur, and Q_next] = LOPBCG (Q_prev, Q_cur, W)
    */
    Eigen::Index i      = std::min<Eigen::Index>(status.iter, 1);
    const auto   Q_prev = i == 0 ? Q.middleCols(0, 0) : Q.middleCols(0, b);
    const auto   Q_cur  = Q.middleCols(Q_prev.cols(), b);

    if(wBlocks > 0) {
        auto W = Q.middleCols(qBlocks * b, b);
        // We add Lanczos-style residual blocks
        W = get_HQ_cur();

        assert(W.allFinite());
        A                    = Q_cur.adjoint() * W;
        B                    = Q_prev.adjoint() * W;
        status.H_norm_approx = std::max({
            status.H_norm_approx,                          //
            A.norm() * std::abs(std::sqrt<RealScalar>(b)), //
            B.norm() * std::abs(std::sqrt<RealScalar>(b))  //
        });

        // 3) Subtract projections to A and B once
        W.noalias() -= Q_cur * A; // Qi * Qi.adjoint()*H*Qi
        if(i > 0) { W.noalias() -= Q_prev * B.adjoint(); }
    }
    if constexpr(settings::print_q) eig::log->warn("Q after filling W: \n{}\n", linalg::matrix::to_string(Q, 8));

    // Inject additional ritz vectors in the GD+k style
    if(mBlocks > 0) {
        // M are the b next-best ritz vectors from the previous iteration
        auto M_new = Q.middleCols((qBlocks + wBlocks) * b, b);
        M_new      = M;
        unset_HQ();
    }
    if constexpr(settings::print_q) eig::log->warn("Q after filling M: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(sBlocks > 0) {
        // We add a residual block "S = (HQ-λQ)"
        auto S = Q.middleCols((qBlocks + wBlocks + mBlocks) * b, b);         // Residual block
        S      = get_HQ_cur() - Q_cur * T_evals(status.optIdx).asDiagonal(); // Put the residual "R" in S.
        eig::log->info("S norm: {:.5e}", S.norm());
        unset_HQ();
    }
    if constexpr(settings::print_q) eig::log->warn("Q after filling S: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(use_preconditioner) {
        // Precondition the latest W, S and R,
        Eigen::Index wOffset = qBlocks;
        Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
        Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;
        /* clang-format off */
        if(wBlocks > 0) {Q.middleCols(wOffset * b, b) = -MultPX(Q.middleCols(wOffset * b, b)); unset_HQ();}
        if(sBlocks > 0) {Q.middleCols(sOffset * b, b) = -MultPX(Q.middleCols(sOffset * b, b)); unset_HQ();}
        if(rBlocks > 0) {Q.middleCols(rOffset * b, b) = -MultPX(Q.middleCols(rOffset * b, b)); unset_HQ();}
        /* clang-format on */
    }
    if constexpr(settings::print_q) eig::log->warn("Q after preconditioning W,S,R: \n{}\n", linalg::matrix::to_string(Q, 8));

    if(chebyshev_filter_degree >= 1) {
        // Apply the chebyshev filter on newly generated residual and random blocks
        RealScalar lambda_cut = status.optVal(0) * (1 + RealScalar{1e-3f});
        if(T_evals.size() > 1) {
            auto       select_2 = get_ritz_indices(ritz, 0, 2, T_evals);
            VectorReal evals    = T_evals(select_2);
            lambda_cut          = std::lerp(evals(0), evals(1), RealScalar{1e-1f});
        }

        Eigen::Index wOffset = qBlocks;
        Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
        Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;

        auto W = wBlocks > 0 ? Q.middleCols(wOffset * b, b) : Q.middleCols(wOffset * b, 0);
        auto S = sBlocks > 0 ? Q.middleCols(sOffset * b, b) : Q.middleCols(sOffset * b, 0);
        auto R = rBlocks > 0 ? Q.middleCols(rOffset * b, b) : Q.middleCols(rOffset * b, 0);

        /* clang-format off */
        if(wBlocks > 0) {W = qr_and_chebyshevFilter(W, 0, status.H_norm_approx * RealScalar{1.05f}, lambda_cut, chebyshev_filter_degree); unset_HQ();}
        if(sBlocks > 0) {S = qr_and_chebyshevFilter(S, 0, status.H_norm_approx * RealScalar{1.05f}, lambda_cut, chebyshev_filter_degree); unset_HQ();}
        if(rBlocks > 0) {R = qr_and_chebyshevFilter(R, 0, status.H_norm_approx * RealScalar{1.05f}, lambda_cut, chebyshev_filter_degree); unset_HQ();}
        /* clang-format on */
    }
    if constexpr(settings::print_q) eig::log->warn("Q after chebyshev W,S,R: \n{}\n", linalg::matrix::to_string(Q, 8));

    // Run QR to orthonormalize [Q_cur, Q_prev, W]
    // This lets the DGKS below to clean W, R and S quickly against [Q_cur, Q_prev]
    hhqr.compute(Q.leftCols(qBlocks * b));
    Q.leftCols(qBlocks * b) = hhqr.householderQ() * MatrixType::Identity(N, qBlocks * b); //
    unset_HQ_cur();

    assert(Q.allFinite());
    // eig::log->warn("Q before DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));

    // pick a relative breakdown tolerance:
    auto       breakdownTol      = eps * 10 * std::max({status.H_norm_approx});
    VectorIdxT active_block_mask = VectorIdxT::Ones(qBlocks + wBlocks + mBlocks + sBlocks + rBlocks);
    assert(active_block_mask.size() * b == Q.cols());
    // DGKS on each of X = W, R, S to remove overlap with [Q_cur, Q_prev]
    for(int k = qBlocks; k < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++k) {
        auto Xk = Q.middleCols(k * b, b);
        for(int rep = 0; rep < 10; ++rep) { // two DGKS passes
            if(active_block_mask[k] == 0) continue;
            auto XjXknorm = RealScalar{0};
            for(Eigen::Index j = 0; j < k; ++j) { // Clean every Xk against Qj...Q{k-1} Q_cur only
                if(active_block_mask[j] == 0) continue;
                auto       Xj   = Q.middleCols(j * b, b);
                MatrixType XjXk = (Xj.adjoint() * Xk);
                Xk -= Xj * XjXk;
                // if(k == qBlocks + wBlocks + mBlocks + sBlocks + rBlocks - 1) {
                //     eig::log->info("|X{}*(X{}.adjoint()*S0)|={:.5e} new S0 norm: {:.5e}",j, j, XjXk.norm(), Xk.norm());
                // }
                XjXknorm = std::max(XjXknorm, XjXk.norm());
            }
            auto Xknorm = Xk.norm();
            if(Xknorm < breakdownTol) {
                // This Wk has been zeroed out! Disable and go to next
                active_block_mask(k) = 0;
                eig::log->info("active_block_mask: {}", active_block_mask.transpose());
                break;
            }
            // eig::log->info("max overlap Q(0...{}).adjoint() * X({}) = {:.16f} rep {}", k-1, k, QjXknorm, rep);
            if(XjXknorm < normTolQ) break; // Orthonormal enough, go to next Wk
        }

        // We have now cleaned Xk. Let's orthonormalize it so we can use it to clean others
        hhqr.compute(Xk);
        Xk = hhqr.householderQ() * MatrixType::Identity(N, b);
    }

    if constexpr(settings::debug_lobpcg) {
        // W's should not have overlap with previous blocks
        for(int k = qBlocks; k < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++k) {
            if(active_block_mask[k] == 0) continue;
            const auto Xk = Q.middleCols(k * b, b);
            for(Eigen::Index j = 0; j < k; ++j) {
                if(active_block_mask[j] == 0) continue;
                const auto Xj       = Q.middleCols(j * b, b);
                auto       XjXknorm = (Xj.adjoint() * Xk).norm();
                if(XjXknorm > orthTolQ) {
                    // eig::log->info("Q after DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));
                    eig::log->warn("overlap X({}).adjoint() * X({}) = {:.16f} ", j, k, XjXknorm);
                    // throw except::runtime_error("overlap Q({}).adjoint() * W = {:.16f} ", j, QjWnorm);
                }
            }
        }
    }

    assert(Q.allFinite());

    // Compress Q if any block got zeroed
    assert(active_block_mask.size() * b == Q.cols());
    Eigen::Index num_active_blocks = std::accumulate(active_block_mask.begin(), active_block_mask.end(), 0);
    // if(num_active_blocks < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks)
        // eig::log->info("should compress {} < {}", num_active_blocks, qBlocks + wBlocks + mBlocks + sBlocks + rBlocks);
    if(num_active_blocks < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) {
        Eigen::Index qOffset = 0;
        Eigen::Index wOffset = qBlocks;
        Eigen::Index mOffset = qBlocks + wBlocks;
        Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
        Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;

        // Check if that the new search directions in the latest W and S have nonzero norm nonzero
        const auto W = wBlocks > 0 ? Q.middleCols(wOffset * b, b) : Q.middleCols(wOffset * b, 0);
        const auto S = sBlocks > 0 ? Q.middleCols(sOffset * b, b) : Q.middleCols(sOffset * b, 0);

        auto minWnorm = wBlocks > 0 ? W.colwise().norm().minCoeff() : 1;
        auto minSnorm = sBlocks > 0 ? S.colwise().norm().minCoeff() : 1;

        bool isZeroWnorm = wBlocks > 0 ? minWnorm < breakdownTol : false;
        bool isZeroSnorm = sBlocks > 0 ? minSnorm < breakdownTol : false;
        if(isZeroWnorm or isZeroSnorm) {
            // Happy breakdown, reached the invariant subspace:
            // We could not add new search directions
            eig::log->warn("optVal {::.16f}: ", fv(status.optVal));
            eig::log->warn("T_evals {::.16f}: ", fv(T_evals));
            eig::log->debug("saturated basis");
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
            return;
        }

        // We can now squeeze out blocks zeroed out by DGKS
        // Get the block indices that we should keep
        std::vector<Eigen::Index> active_columns;
        active_columns.reserve((qBlocks + wBlocks + mBlocks + sBlocks + rBlocks) * b);
        for(Eigen::Index j = 0; j < qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; ++j) {
            if(active_block_mask(j) == 1) {
                for(Eigen::Index k = 0; k < b; ++k) active_columns.push_back(j * b + k);
            }
        }
        active_columns.shrink_to_fit();
        // eig::log->info("active_columns: {}", active_columns);
        if(active_columns.size() != static_cast<size_t>(Q.cols())) {
            Q = Q(Eigen::all, active_columns).eval(); // Shrink keeping only nonzeros

            unset_HQ();
            unset_HQ_cur();

            Eigen::Index qBlocks_new = std::accumulate(active_block_mask.begin() + qOffset, active_block_mask.begin() + qOffset + qBlocks, 0);
            Eigen::Index wBlocks_new = std::accumulate(active_block_mask.begin() + wOffset, active_block_mask.begin() + wOffset + wBlocks, 0);
            Eigen::Index mBlocks_new = std::accumulate(active_block_mask.begin() + mOffset, active_block_mask.begin() + mOffset + mBlocks, 0);
            Eigen::Index sBlocks_new = std::accumulate(active_block_mask.begin() + sOffset, active_block_mask.begin() + sOffset + sBlocks, 0);
            Eigen::Index rBlocks_new = std::accumulate(active_block_mask.begin() + rOffset, active_block_mask.begin() + rOffset + rBlocks, 0);

            if(qBlocks_new != qBlocks) eig::log->info("new qBlocks : {} -> {}", qBlocks, qBlocks_new);
            if(wBlocks_new != wBlocks) eig::log->info("new wBlocks : {} -> {}", wBlocks, wBlocks_new);
            if(mBlocks_new != mBlocks) eig::log->info("new mBlocks : {} -> {}", mBlocks, mBlocks_new);
            if(sBlocks_new != sBlocks) eig::log->info("new sBlocks : {} -> {}", sBlocks, sBlocks_new);
            if(rBlocks_new != rBlocks) eig::log->info("new rBlocks : {} -> {}", rBlocks, rBlocks_new);

            qBlocks = qBlocks_new;
            wBlocks = wBlocks_new;
            mBlocks = mBlocks_new;
            sBlocks = sBlocks_new;
            rBlocks = rBlocks_new;
        }
    }

    if constexpr(settings::print_q) eig::log->warn("Q after compression: \n{}\n", linalg::matrix::to_string(Q, 8));

    assert(Q.colwise().norm().minCoeff() != 0);

    hhqr.compute(Q);
    Q = hhqr.householderQ() * MatrixType::Identity(N, Q.cols()); //
    unset_HQ();
    unset_HQ_cur();

    if constexpr(settings::print_q) eig::log->warn("Q after householder: \n{}\n", linalg::matrix::to_string(Q, 8));

    assert(Q.allFinite());
    assert(Q.colwise().norm().minCoeff() != 0);

    // Form T
    T = Q.adjoint() * get_HQ(); // (blocks*b)×(blocks*b)
    assert(T.colwise().norm().minCoeff() != 0);

    // Solve T by calling diagonalizeT() elsewhere
}

template<typename Scalar>
void LOBPCG<Scalar>::extractResidualNorms() {
    if(status.stopReason != StopReason::none) return;
    if(T.rows() < b) return;

    // Basically, since
    //     a) ritz evecs: V = Q*Z, where Z are a selection of columns from T,
    //     b) ritz evals: TZ =  Λ*Z (eigenvalue problem)
    // we can do |HV - ΛV| = |(H(QZ) - (QZ)Λ| = |(HQ)Z - Q(TZ)| = |(HQ - QT)*Z|,
    // where Z are the desired ritz-eigenvector columns of T.
    // For this to be effective, we need to have HQ already.

    // eig::log->info("rnorms: {::.5e}", fv(status.rNorms));
    if(!use_refined_rayleigh_ritz) {
        Eigen::Index qcols = std::min(Q.cols(), T_evecs.rows());
        auto         Z     = T_evecs(Eigen::all, status.optIdx);
        status.rNorms      = ((get_HQ().leftCols(qcols) - Q.leftCols(qcols) * T) * Z).colwise().norm();
    }

    // eig::log->info("rnorms: {::.5e}", fv(status.rNorms));
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxLanczosResidualHistory(Eigen::Index k) {
    // We can't have more linearly independent basis columns than there are N (rows in H).
    // The structure of LOBPCG is [Q_prev, Q_cur, W0...W{k-1}, M0...M{m-1}, S0...S{s-1}], each of width b.
    // The total width shouldn't exceed N.
    k           = std::min(k, N / b);
    max_wBlocks = std::max<Eigen::Index>(0, k); // Not strictly necessary
    eig::log->debug("LOBPCG: max_wBlocks = {}", max_wBlocks);
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    // We can't have more linearly independent basis columns than there are N (rows in H).
    // The structure of LOBPCG is [Q_prev, Q_cur, W0...W{k-1}, M0...M{m-1}, S0...S{s-1}], each of width b.
    // The total width shouldn't exceed N.
    m           = std::min(m, N / b);
    max_mBlocks = std::max<Eigen::Index>(0, m); // mBlocks are not required
    eig::log->debug("LOBPCG: max_mBlocks = {}", max_mBlocks);
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void LOBPCG<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    // We can't have more linearly independent basis columns than there are N (rows in H).
    // The structure of LOBPCG is [Q_prev, Q_cur, W0...W{k-1}, M0...M{m-1}, S0...S{s-1}], each of width b.
    // The total width shouldn't exceed N.
    s           = std::min(s, N / b);
    max_sBlocks = std::max<Eigen::Index>(0, s); // Not strictly necessary
    eig::log->debug("LOBPCG: max_sBlocks = {}", max_sBlocks);
}