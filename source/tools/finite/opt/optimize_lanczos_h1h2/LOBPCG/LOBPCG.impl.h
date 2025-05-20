#pragma once
#include "../LOBPCG.h"
#include "../SolverExit.h"
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
}

template<typename Scalar>
typename LOBPCG<Scalar>::MatrixType LOBPCG<Scalar>::chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       // input Q (orthonormal)
                                                                    const Eigen::Ref<const MatrixType> &HQref,      // input H * Q
                                                                    RealScalar                          lambda_min, // estimated smallest eigenvalue
                                                                    RealScalar                          lambda_max, // estimated largest eigenvalue
                                                                    RealScalar                          lambda_cut, // cut-off (e.g. λmin for low-end)
                                                                    int                                 degree      // polynomial degree k,
) {
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

    if(degree == 0) { return Qref; }
    if(degree == 1) { return (HQref - av * Qref) * (RealScalar{1} / bv); }

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
    MatrixType Tkm1 = (HQref - av * Qref) * (RealScalar{1} / bv);
    MatrixType Tcur(N, Qref.cols());
    for(int k = 2; k <= degree; ++k) {
        Tcur = (MultHX(Tkm1) - av * Tkm1) * (RealScalar{2} / bv) - Tkm2;
        if(!std::isfinite(Tcur.norm())) { throw except::runtime_error("Tcur.norm() is not finite"); }
        Tkm2 = std::move(Tkm1);
        Tkm1 = std::move(Tcur);
    }
    return Tkm1 * (Scalar{1} / norm);
}

template<typename Scalar>
void LOBPCG<Scalar>::build() {
    const Eigen::Index N = H1.rows();
    // Eigen::Index       max_steps = std::min<Eigen::Index>(ncv, N);
    assert(V.cols() == b);
    assert(max_wBlocks >= 1);
    use_chebyshev_basis_during_ritz_extraction = false;
    // Now V has b orthonormalized ritz vectors

    // Start defining the first blocks of Q

    Eigen::Index qBlocks  = status.iter == 0 ? 1 : 2;                                                     // For Q_prev, Q_cur
    Eigen::Index wBlocks  = std::min(max_wBlocks, std::max<Eigen::Index>(1, Q.cols() / b - qBlocks + 1)); // Add space for 1 W block
    Eigen::Index qwBlocks = qBlocks + wBlocks;                                                            // Total number of blocks

    // Eigen::Index wBlocks  = std::min(status.iter + 1, max_wBlocks); // For the W's, we get one per iter
    if(status.iter == 0 or T_evecs.rows() == 0) {
        Q.setZero(N, qwBlocks * b); // Allocate for 2 blocks Q_prev, Q_cur
        Q.leftCols(b) = V;          // Copy the V panel as an initial guess
    } else if(T_evecs.rows() > 0) {
        // eig::log->info("Q before resize: \n{}\n", linalg::matrix::to_string(Q,8));

        if(Q.cols() != qwBlocks * b) {
            Eigen::Index qBlocks_old = status.iter <= 1 ? 1 : 2;
            Eigen::Index wBlocks_old = std::max<Eigen::Index>(1, Q.cols() / b - qBlocks_old); //   std::min(status.iter, max_wBlocks);
            // Move the W blocks to the right
            // happens at iteration 1, when nBlocks goes from 1 -> 2
            // or when we compress Q
            MatrixType W_backup = Q.rightCols(wBlocks_old * b);
            // eig::log->info("W_backup: \n{}\n", linalg::matrix::to_string(W_backup,8));
            assert(W_backup.colwise().norm().minCoeff() > 10 * normTolQ);
            assert(W_backup.allFinite());
            Q.conservativeResize(N, qwBlocks * b);
            // Q.rightCols(Q.cols() - qBlocks_old).setZero(); // Fill with zeros
            Q.middleCols(qBlocks * b, W_backup.cols()) = W_backup;
        }
        assert(qBlocks + wBlocks == qwBlocks);
        assert((qBlocks + wBlocks) * b == Q.cols());
        assert(qwBlocks * b == Q.cols());

        // Roll Q_prev and Q_cur
        {
            // eig::log->info("Q before roll: \n{}\n", linalg::matrix::to_string(Q,8));
            Q.middleCols(0, b) = Q.middleCols(b, b);
            Q.middleCols(b, b) = V;
        }
        // Roll W's
        for(int k = wBlocks - 1; k > 0; --k) {
            // Takes [W0 | W1 | W2 | W3] to [W0 | W0 | W1| W2 ] So that we can overwrite W0 (the oldest)
            auto Wk0 = Q.middleCols((qBlocks + k + 0) * b, b);
            auto Wk1 = Q.middleCols((qBlocks + k - 1) * b, b);
            Wk0      = Wk1;
        }

        // eig::log->info("Q after roll: \n{}\n", linalg::matrix::to_string(Q,8));
        if(Q.colwise().norm().minCoeff() < eps) throw except::runtime_error("Too close to zero!");
    }

    /*! Main LOBPCG step.
        In LOBPCG loop, we always update three basis blocks at a time [Q_prev, Q_cur, and Q_next] = LOPBCG (Q_prev, Q_cur, W)
    */
    Eigen::Index i      = std::min<Eigen::Index>(status.iter, 1);
    auto         Q_prev = i == 0 ? Q.middleCols(0, 0) : Q.middleCols(0, b);
    auto         Q_cur  = Q.middleCols(Q_prev.cols(), b);
    auto         W      = Q.middleCols(qBlocks * b, b);


    // 1) Apply the operator and form W = [f(H1,H2)*Q_cur]
    W = MultHX(Q_cur);

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

    // measure the *smallest* new direction in W:
    auto minWnorm = W.colwise().norm().minCoeff(); // Krylov residual! Not Ritz residual (do not use as ritz-vector residual norms)
    // pick a relative breakdown tolerance:
    auto breakdownTol = eps * 10 * std::max({A.norm(), B.norm(), status.H_norm_approx});
    if(minWnorm < breakdownTol) {
        // Happy breakdown, reached the invariant subspace:
        //      The norm of W is sufficiently small that it would not continue the three-term recurrence
        eig::log->warn("Q breakdown: \n{}\n", linalg::matrix::to_string(Q, 8));

        if(i == 0) {
            eig::log->info("converged subspace");
            status.exit |= SolverExit::converged_subspace;
            status.exitMsg.emplace_back("Converged: the current solution is likely exact");
        } else {
            eig::log->info("saturated subspace");
            status.exit |= SolverExit::saturated_subspace;
            status.exitMsg.emplace_back("Converged: exhausted the subspace directions");
        }
        return;
    }

    if(use_preconditioner) {
        Q.rightCols(wBlocks) = MultPX(Q.rightCols(wBlocks));
        // W = MultPX(W);
    }

    if(Q.colwise().norm().minCoeff() == 0) {
        eig::log->warn("Q before DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));
        assert(Q.colwise().norm().minCoeff() != 0);
    }
    assert(Q.allFinite());
    // eig::log->warn("Q before DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));

    // Run QR to orthonormalize [Q_cur, Q_prev]
    // This lets the DGKS below to clean W quickly against [Q_cur, Q_prev]
    hhqr.compute(Q.leftCols(qBlocks * b));
    Q.leftCols(qBlocks * b) = hhqr.householderQ() * MatrixType::Identity(N, qBlocks * b); //

    VectorIdxT active_block_mask = VectorIdxT::Ones(qBlocks + wBlocks);
    // DGKS on each W to remove overlap with the previous blocks
    for(int k = 0; k < wBlocks; ++k) {
        auto Wk = Q.middleCols((qBlocks + k) * b, b);
        for(int rep = 0; rep < 2; ++rep) { // two DGKS passes
            auto QjWknorm = RealScalar{0};
            for(Eigen::Index j = 0; j < qBlocks; ++j) { // Clean every Wk against Q_prev and Q_cur only
                if(active_block_mask[j] == 0) continue;
                auto       Qj   = Q.middleCols(j * b, b);
                MatrixType QjWk = (Qj.adjoint() * Wk);
                Wk -= Qj * QjWk;
                QjWknorm = std::max(QjWknorm, QjWk.norm());
            }
            auto Wknorm = Wk.norm();
            if(Wknorm < breakdownTol) {
                // This Wk has been zeroed out! Disable and go to next
                active_block_mask(qBlocks + k) = 0;
                eig::log->info("active_block_mask: {}", active_block_mask.transpose());
                break;
            }
            // eig::log->info("max overlap rep Q(0...{}).adjoint() * W({}) = {:.16f} rep {}", qBlocks + k - 1, k, QjWknorm, rep);
            if(QjWknorm < normTolQ) break; // Orthonormal enough, go to next Wk
        }
    }

    // Compress Q
    std::vector<Eigen::Index> active_blocks;
    active_blocks.reserve(qBlocks + wBlocks);
    for(Eigen::Index j = 0; j < qBlocks + wBlocks; ++j) {
        if(active_block_mask(j) == 1) active_blocks.push_back(j);
    }
    // eig::log->info("active_block_mask: {}", active_block_mask.transpose());
    // eig::log->info("active_blocks    : {}");
    // eig::log->info("qBlocks          : {}", qBlocks);
    // eig::log->info("wBlocks          : {}", wBlocks);
    if(active_blocks.size() < qBlocks + wBlocks) {
        Q       = Q(Eigen::all, active_blocks).eval();
        wBlocks = active_block_mask.bottomRows(wBlocks).count(); // Update wBlocks.
        eig::log->info("new wBlocks          : {}", wBlocks);
    }

    if constexpr(settings::debug_lobpcg) {
        // W's should not have overlap with previous blocks
        for(int k = 0; k < wBlocks; ++k) {
            auto Wk = Q.middleCols((qBlocks + k) * b, b);
            for(Eigen::Index j = 0; j < qBlocks; ++j) {
                auto Qj       = Q.middleCols(j * b, b);
                auto QjWknorm = (Qj.adjoint() * Wk).norm();
                if(QjWknorm > orthTolQ) {
                    // eig::log->info("Q after DGKS: \n{}\n", linalg::matrix::to_string(Q, 8));
                    eig::log->warn("overlap Q({}).adjoint() * W({}) = {:.16f} ", j, k, QjWknorm);
                    // throw except::runtime_error("overlap Q({}).adjoint() * W = {:.16f} ", j, QjWnorm);
                }
            }
        }
    }
    assert(Q.allFinite());

    minWnorm = W.colwise().norm().minCoeff(); // Krylov residual! Not Ritz residual (do not use as ritz-vector residual norms)
    // pick a relative breakdown tolerance:
    if(minWnorm < breakdownTol) {
        // Happy breakdown, reached the invariant subspace:
        //      The norm of W is sufficiently small that it would not continue the three-term recurrence

        if(Q.colwise().norm().minCoeff() < eps) {
            eig::log->warn("Q breakdown: \n{}\n", linalg::matrix::to_string(Q, 8));
            throw except::runtime_error("Too close to zero!");
        }
        if(i == 0) {
            eig::log->info("converged subspace");
            status.exit |= SolverExit::converged_subspace;
            status.exitMsg.emplace_back("Converged: the current solution is likely exact");
        } else {
            eig::log->info("saturated subspace");
            status.exit |= SolverExit::saturated_subspace;
            status.exitMsg.emplace_back("Converged: exhausted the subspace directions");
        }
        return;
    }
    assert(Q.allFinite());
    assert(Q.colwise().norm().minCoeff() != 0);

    // Now W does not contain any overlap toward the previous two basis directions Q_prev and Q_cur

    // Form the new LOBPCG search space by appending W

    // Run QR to block orthogonalize Q
    hhqr.compute(Q.leftCols(qwBlocks * b));
    Q.leftCols(qwBlocks * b) = hhqr.householderQ() * MatrixType::Identity(N, qwBlocks * b); //

    assert(Q.allFinite());
    assert(Q.colwise().norm().minCoeff() != 0);

    // MatrixType G         = Q.adjoint() * Q;
    // RealScalar orthError = (G - MatrixType::Identity(Q.cols(), Q.cols())).norm();
    // if(orthError > 1000 * orthTolQ) {
    //     eig::log->info("|G - I| {:.5e}", orthError);
    //     eig::log->info("G is not identity: \n{}\n", linalg::matrix::to_string(G, 8));
    //     throw except::runtime_error("G is not identity");
    // }

    if(chebyshev_filter_degree >= 1 and status.iter > 1) {
        RealScalar lambda_cut = status.optVal(0) * (1 + RealScalar{1e-3f});
        if(T_evals.size() > 1) {
            auto       select_2 = get_ritz_indices(ritz, 2, T_evals);
            VectorReal evals    = T_evals(select_2);
            if(std::abs(evals(0)) < RealScalar{1e-1f}) chebyshev_filter_degree = 2;
            if(std::abs(evals(0)) < RealScalar{1e-2f}) chebyshev_filter_degree = 4;
            lambda_cut = std::lerp(evals(0), evals(1), 1e-3);
        }
        HQ = MultHX(Q);
        X = chebyshevFilter(Q, HQ, 0, status.H_norm_approx * 1.05, lambda_cut, 1);

        // Re orthogonalize
        hhqr.compute(X);
        X = hhqr.householderQ() * MatrixType::Identity(N, X.cols()); //

        assert(X.allFinite());
        assert(X.colwise().norm().minCoeff() != 0);

        // Form T
        use_chebyshev_basis_during_ritz_extraction = true;
        HX                                         = MultHX(X);
        T                                          = X.adjoint() * HX; // (blocks*b)×(blocks*b)
        // eig::log->info("Q: \n{}\n", linalg::matrix::to_string(Q,8));
        // eig::log->info("T: \n{}\n", linalg::matrix::to_string(T,8));
        assert(T.colwise().norm().minCoeff() != 0);

    } else {
        // Form T
        HQ = MultHX(Q);
        T  = Q.adjoint() * HQ; // (blocks*b)×(blocks*b)
        // eig::log->info("Q: \n{}\n", linalg::matrix::to_string(Q,8));
        // eig::log->info("T: \n{}\n", linalg::matrix::to_string(T,8));
        assert(T.colwise().norm().minCoeff() != 0);
    }

    // Solve T by calling diagonalizeT() elsewhere

    if constexpr(settings::debug_lobpcg) {
        auto                  Q0        = Q.middleCols(0 * b, b);
        [[maybe_unused]] auto Q0Q0_norm = (Q0.adjoint() * Q0).norm();
        assert(std::abs(Q0Q0_norm - std::sqrt<RealScalar>(b)) < orthTolQ);

        if(i > 0) {
            auto                  Q1        = Q.middleCols(1 * b, b);
            [[maybe_unused]] auto Q1Q1_norm = (Q1.adjoint() * Q1).norm();
            [[maybe_unused]] auto Q0Q1_norm = (Q0.adjoint() * Q1).norm();
            assert(std::abs(Q1Q1_norm - std::sqrt<RealScalar>(b)) < orthTolQ);
            assert(Q0Q1_norm < orthTolQ * 10000);
        }
    }
}

template<typename Scalar>
void LOBPCG<Scalar>::extractResidualNorms() {
    bool saturated = has_any_flags(status.exit, SolverExit::converged_subspace, SolverExit::saturated_subspace);
    if(saturated and status.iter == 0) {
        for(Eigen::Index i = 0; i < nev; ++i) { // Only consider nev rnorms
            switch(algo) {
                case OptAlgo::DMRG: [[fallthrough]];
                case OptAlgo::DMRGX: [[fallthrough]];
                case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
                case OptAlgo::XDMRG: {
                    status.rNorms(i) = (MultHX(V.col(i)) - status.optVal(i) * V.col(i)).template norm();
                    break;
                }
                case OptAlgo::GDMRG: {
                    status.rNorms(i) = (H1.MultAX(V.col(i)) - status.optVal(i) * H2.MultAX(V.col(i))).template lpNorm<Eigen::Infinity>();
                    break;
                }
                default: throw except::runtime_error("Unknown algorithm");
            }
        }
    }
    if(saturated) return;
    if(T.rows() < b) return;

    // Basically, since
    //     a) ritz evecs: V = Q*Z, where Z are a selection of columns from T,
    //     b) ritz evals: TZ =  Λ*Z (eigenvalue problem)
    // we can do |HV - ΛV| = |(H(QZ) - (QZ)Λ| = |(HQ)Z - Q(TZ)| = |(HQ - QT)*Z|,
    // where Z are the desired ritz-eigenvector columns of T.
    // For this to be effective, we need to have HQ already.

    // for(Eigen::Index i = 0; i < nev; ++i) { // Only consider nev rnorms
    //     if(algo == OptAlgo::DMRG)
    //         status.rNorms(i) = (H1.MultAX(V.col(i)) - status.optVal(i) * V.col(i)).template lpNorm<Eigen::Infinity>();
    //     else if(algo == OptAlgo::GDMRG)
    //         status.rNorms(i) = (H1.MultAX(V.col(i)) - status.optVal(i) * H2.MultAX(V.col(i))).template lpNorm<Eigen::Infinity>();
    //     else
    //         status.rNorms(i) = (H2.MultAX(V.col(i)) - status.optVal(i) * V.col(i)).template lpNorm<Eigen::Infinity>();
    // }
    // eig::log->info("rnorms: {::.5e}", fv(status.rNorms));
    if(!use_refined_rayleigh_ritz) {
        const auto  &Qa    = use_chebyshev_basis_during_ritz_extraction ? X : Q;
        const auto  &HQa   = use_chebyshev_basis_during_ritz_extraction ? HX : HQ;
        Eigen::Index qcols = std::min(Qa.cols(), T_evecs.rows());
        auto         Z     = T_evecs(Eigen::all, status.optIdx);
        status.rNorms      = ((HQa.leftCols(qcols) - Qa.leftCols(qcols) * T) * Z).colwise().norm();
    }

    // eig::log->info("rnorms: {::.5e}", fv(status.rNorms));
}

template<typename Scalar>
void LOBPCG<Scalar>::set_ResidualHistoryLength(Eigen::Index k) {
    // We can't have more linearly independent W's than there are N (rows in H).
    // The structure of LOBPCG is [Q_prev, Q_cur, W0...W{k-1}], each of width b.
    // The need at least 1 W, and the total width shouldn't exceed N.
    k           = std::min(k, N / b - 2);
    max_wBlocks = std::max<Eigen::Index>(1, k);
    eig::log->debug("LOBPCG: max_wBlocks = {}", max_wBlocks);
}
