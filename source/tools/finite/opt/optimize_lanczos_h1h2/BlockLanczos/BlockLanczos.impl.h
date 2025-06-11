#pragma once
#include "../BlockLanczos.h"
#include "../StopReason.h"
#include "io/fmt_custom.h"
#include "math/eig/log.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include <Eigen/Eigenvalues>

namespace settings {
#if defined(NDEBUG)
    constexpr bool debug_lanczos = false;
#else
    constexpr bool debug_lanczos = true;
#endif
}
template<typename Scalar>
void BlockLanczos<Scalar>::write_Q_next_B_DGKS(Eigen::Index i) {
    // 4) Subtract the projections of W onto all previous basis blocks
    for(int rep = 0; rep < 2; ++rep) {
        MatrixType QjW;
        for(Eigen::Index j = i; j >= 0; --j) {
            // if(i - j > 3) continue; // Only subtracts projections onto Q_prev and Q_cur
            auto Qj = Q.middleCols(j * b, b);
            QjW     = (Qj.adjoint() * W);
            W.noalias() -= Qj * QjW;
            // if(QjW.norm() < orthTolQ) break;
        }
    }

    // if constexpr(settings::debug_lanczos) {
    //     // W should not have overlap with Q_cur or Q_prev
    //     for(Eigen::Index j = 0; j <= i; ++j) {
    //         // if(i - j > 1) continue;
    //         auto Qj      = Q.middleCols(j * b, b); // Q_prev and Q_cur
    //         auto QjWnorm = (Qj.adjoint() * W).norm();
    //         // assert(QjWnorm < orthTolQ);
    //     }
    // }

    // Run QR(W) = Q_next * B.
    hhqr.compute(W); // Gives us Q_next * B,  B = Q_next.adjoint() * W, where W = (H*Q_cur - projections)
    Q.middleCols((i + 1) * b, b) = hhqr.householderQ().setLength(W.cols()) * MatrixType::Identity(N, b);        // Q_next
    B                            = hhqr.matrixQR().topLeftCorner(b, b).template triangularView<Eigen::Upper>(); // B
}

template<typename Scalar>
void BlockLanczos<Scalar>::build() {
    const Eigen::Index N = H1.rows();
    Eigen::Index       m = ncv;

    assert(m >= 1);
    assert(V.cols() == b);
    // Now V has b orthonormalized ritz vectors
    // Start defining the first blocks of Q
    Q.setZero(N, m * b); // Account for the b2 initial vectors
    Q.leftCols(b) = V;   // Copy the V panel as initial guess

    assert(Q.leftCols(b).allFinite());
    assert(std::abs((Q.leftCols(b).adjoint() * Q.leftCols(b)).norm() - std::sqrt<RealScalar>(b)) < orthTolQ);

    T.setZero(Q.cols(), Q.cols());

    /*! Main block-Lanczos loop. We have already filled istart out of m blocks.
        At this point we have Q = [Q0=V] and will produce Q = [Q0,...Q[m-1]].
        We also end up with a block-tridiagonals T:
        T = [ A₀     B₁ᴴ     0       0       ...        0     ]
            [ B₁     A₁      B₂ᴴ     0       ...        0     ]
            [ 0      B₂      A₂      B₃ᴴ     ...        0     ]
            [ 0      0       B₃      A₃      ...        0     ]
            [ ...   ...     ...     ...      ...       ...    ]
            [ 0       0      0       0       Bₘ-1     Aₘ-1    ]
    */

    for(Eigen::Index i = 0; i < m; ++i) {
        const auto Q_cur = Q.middleCols(i * b, b);

        // 1) Apply the operator and form W = [f(H1,H2)*Q_cur]
        W  = MultHX(Q_cur);
        HQ = W; // Save for later, when updating B

        assert(W.allFinite());

        // 2) Compute A and add it to T
        A                           = Q_cur.adjoint() * W;
        T.block(i * b, i * b, b, b) = A;

        // 3) Subtract projections to A and B once
        W.noalias() -= Q_cur * A; // Qi * Qi.adjoint()*H*Qi
        if(i > 0) {
            auto Q_prev = Q.middleCols((i - 1) * b, b);
            W.noalias() -= Q_prev * B.adjoint(); // Qj * (Qi.adjoint()*H*Qj).adjoint(), j<i // B is also from the previous iteration
        }

        // measure the *smallest* new direction in W:
        auto min_rnorm = W.colwise().norm().minCoeff(); // Krylov residual! Not Ritz residual (do not use as ritz-vector residual norms)
        // pick a *relative* breakdown tolerance:
        auto breakdownTol = eps * 10 * std::max(A.norm(), B.norm());
        if(min_rnorm < breakdownTol) {
            // Happy breakdown, reached the invariant subspace:
            //      The norm of W is sufficiently small that it would not continue the three-term recurrence
            Eigen::Index doneBlocks = (i + 1);
            // shrink down to the blocks that were built
            Q.conservativeResize(N, doneBlocks * b);
            T.conservativeResize(doneBlocks * b, doneBlocks * b);

            // We need B for what would have been the next iteration to calculate residual norms
            hhqr.compute(W); // Gives us Q_next * B
            B = hhqr.matrixQR().topLeftCorner(b, b).template triangularView<Eigen::Upper>();

            eig::log->debug("saturated basis");
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
            break;
        }

        if(use_preconditioner) { W = MultPX(W); }

        write_Q_next_B_DGKS(i);

        if(i + 1 == m) break; // We only need Q0... Q[m-1], not the last Q[m]. But we do need B[m] for residual norm calculations.

        // 6) Add B to T
        T.block(i * b, (i + 1) * b, b, b) = B.adjoint();
        T.block((i + 1) * b, i * b, b, b) = B;

        if constexpr(settings::debug_lanczos) {
            if(bIsOK(B, A)) {
                auto                  Q1        = Q.middleCols(i * b, b);
                auto                  Q2        = Q.middleCols((i + 1) * b, b);
                [[maybe_unused]] auto Q1Q1_norm = (Q1.adjoint() * Q1).norm();
                [[maybe_unused]] auto Q2Q2_norm = (Q2.adjoint() * Q2).norm();
                [[maybe_unused]] auto Q1Q2_norm = (Q1.adjoint() * Q2).norm();
                assert(std::abs(Q1Q1_norm - std::sqrt<RealScalar>(b)) < orthTolQ);
                assert(std::abs(Q2Q2_norm - std::sqrt<RealScalar>(b)) < orthTolQ);
                assert(Q1Q2_norm < orthTolQ);

                if(i > 0) {
                    auto                  Q0         = Q.middleCols((i - 1) * b, b);
                    [[maybe_unused]] auto Q0Q0_norm  = (Q0.adjoint() * Q0).norm();
                    [[maybe_unused]] auto Q0Q1_norm  = (Q0.adjoint() * Q1).norm();
                    [[maybe_unused]] auto Q0Q2_norm  = (Q0.adjoint() * Q2).norm();
                    [[maybe_unused]] auto Q0HQ0_norm = (Q0.adjoint() * HQ).norm(); // A0
                    assert(std::abs(Q0Q0_norm - std::sqrt<RealScalar>(b)) < orthTolQ);
                    assert(Q0Q1_norm < orthTolQ * 10000);
                    assert(Q0Q2_norm < orthTolQ * 10000);
                    // assert(Q1HQ0_norm < normTol);
                }
                // Check orthogonality explicitly
                auto Q_next = Q.middleCols((i + 1) * b, b);
                assert(Q_next.allFinite());
                if constexpr(settings::debug_lanczos) {
                    for(Eigen::Index j = i; j >= 0; --j) {
                        // if(i - j > 1) continue;
                        auto Qj           = Q.middleCols(j * b, b); // Q_prev and Q_cur
                        auto QjQnext_norm = (Qj.adjoint() * Q_next).norm();
                        // eig::log->info("overlap Q({}).adjoint() * Q({}) = {:.16f} ", j, i + 1, QjQnext_norm);
                        assert(QjQnext_norm < orthTolQ * 10000);
                    }
                }
            }
        }
    }

    if constexpr(settings::debug_lanczos) {
        if(status.iter % 10 == 0) {
            Eigen::Index tcols    = T.rows();
            MatrixType   T_direct = Q.leftCols(tcols).adjoint() * MultHX(Q.leftCols(tcols));
            // 3) Compare with your assembled T2:
            MatrixType diff      = T_direct - T;
            RealScalar diffNorm  = diff.norm();
            MatrixType G         = Q.leftCols(tcols).adjoint() * Q.leftCols(tcols);
            RealScalar orthError = (G - MatrixType::Identity(tcols, tcols)).norm();

            if(diffNorm > RealScalar{1e-6f} or orthError > RealScalar{1e-6f}) {
                eig::log->info("T: \n{}\n", linalg::matrix::to_string(T, 8));
                eig::log->info("T_direct: \n{}\n", linalg::matrix::to_string(T_direct, 8));
                eig::log->info("G = Q.adjoint()*Q = \n{}\n", linalg::matrix::to_string(G, 8));
                for(long j = 0; j < diff.cols(); ++j) {
                    for(long i = 0; i < diff.rows(); ++i) {
                        if(std::abs(diff(i, j)) > RealScalar{1e-6f}) { eig::log->info("diff({},{}) = {:.16f}", i, j, fp(diff(i, j))); }
                    }
                }
            }
            eig::log->info("‖T_direct – T‖ = {:.4e} | ‖G-I‖ = {:.4e}", fp(diffNorm), fp(orthError));
            if(diffNorm > RealScalar{1e-4f} or orthError > RealScalar{1e-4f}) { throw except::runtime_error("Lanczos error"); }
        }
    }
}
