#pragma once
#include "../solver_gdplusk.h"
#include "../StopReason.h"
#include "io/fmt_custom.h"
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
    constexpr bool debug_gdplusk = true;
#endif
    constexpr bool print_q = false;
}

template<typename Scalar>
void solver_gdplusk<Scalar>::make_new_Q_block(fMultP_t fMultP) {
    assert(V.rows() == N);
    assert(V.cols() == b);

    // Start defining the residual blocks (we have ritz-type and lanczos type)
    Eigen::Index sBlocks_old = sBlocks;

    auto get_total_blocks = [&]() { return sBlocks; };

    sBlocks = std::min(max_sBlocks, sBlocks_old + 1); // Add space for one more S block

    // Try to keep W and S if possible, drop R, M first
    while(N - V.cols() < get_total_blocks() * b) {
        /* clang-format off */
        if(sBlocks > 0) { sBlocks--; continue; }
        break; // If all are at min, break to avoid infinite loop
        /* clang-format on */
    }

    Q_new.resize(N, get_total_blocks() * b);

    assert(N >= Q_new.cols() + V.cols());

    Eigen::Index sOffset = mBlocks;
    if(sBlocks > 0) Q_new.middleCols(sOffset * b, b).noalias() = get_sBlock(S, fMultP);
    OrthMeta m;
    m.maskPolicy = MaskPolicy::COMPRESS;

    auto orthognalize_Q_new = [&]() {
        HQ_new  = MatrixType();
        H1Q_new = MatrixType();
        H2Q_new = MatrixType();
        if(algo == OptAlgo::GDMRG) {
            if(use_h2_inner_product) {
                block_h2_orthogonalize(V, H1V, H2V, Q_new, H1Q_new, H2Q_new, m);
                block_h2_orthogonalize(Q, H1Q, H2Q, Q_new, H1Q_new, H2Q_new, m);
            } else {
                block_l2_orthogonalize(V, H1V, H2V, Q_new, H1Q_new, H2Q_new, m);
                block_l2_orthogonalize(Q, H1Q, H2Q, Q_new, H1Q_new, H2Q_new, m);
            }
        } else {
            block_l2_orthogonalize(V, HV, Q_new, HQ_new, m);
            block_l2_orthogonalize(Q, HQ, Q_new, HQ_new, m);
        }
    };

    orthognalize_Q_new();
    if(Q_new.cols() == 0 and inject_randomness) {
        eiglog->debug("Replacing Q_new with a random vector");
        Q_new = get_rBlock();
        orthognalize_Q_new();
    }

    if constexpr(settings::print_q) eiglog->warn("Q_new after compression: \n{}\n", linalg::matrix::to_string(Q_new, 8));
}

template<typename Scalar>
void solver_gdplusk<Scalar>::build() {
    switch(algo) {
        case OptAlgo::DMRG: [[fallthrough]];
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
        case OptAlgo::XDMRG: {
            fMultP_t fMultP = [this](const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                                     std::optional<const Eigen::Ref<const MatrixType>> iG) -> MatrixType { return this->MultP(X, evals, iG); };
            // fMultH_t fMultH = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return this->MultH(X); };
            make_new_Q_block(fMultP);
            build(Q, HQ, Q_new, HQ_new);
            break;
        }
        case OptAlgo::GDMRG: {
            fMultP_t fMultP = [this](const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                                     std::optional<const Eigen::Ref<const MatrixType>> iG) -> MatrixType {
                if(use_h1h2_preconditioner)
                    return this->MultP1P2(X, evals, iG);
                else
                    return this->MultP2(X, evals, iG);
            };
            make_new_Q_block(fMultP);
            build(Q, H1Q, H2Q, Q_new, H1Q_new, H2Q_new);
            break;
        }
        default: throw except::runtime_error("invalid algo {}", enum2sv(algo));
    }
}

template<typename Scalar>
void solver_gdplusk<Scalar>::build(MatrixType &Q, MatrixType &HQ, const MatrixType &Q_new, const MatrixType &HQ_new) {
    if(status.stopReason != StopReason::none) return;
    assert(Q_new.rows() == N);

    // Append the enrichment for this iteration
    auto oldCols = Q.cols();
    auto newCols = std::max<Eigen::Index>(1, std::min<Eigen::Index>({Q.cols() + Q_new.cols(), N}));
    if(newCols > maxBasisBlocks or Q.cols() == 0 or Q_new.cols() == 0) {
        // (re)start
        if(Q_new.cols() == 0 and status.iter <= status.iter_last_restart + 2) {
            // Failed to add a nonzero residual
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back(fmt::format("saturated basis: exhausted subspace search | iter {} | mv {} | {:.3e} s", status.iter,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            return;
        }

        // Our Q has a layout [V, M, K, Q_new], where
        // V: the ritz vector block after the last restart
        // M: the next set ritz vector blocks (i.e. the next b top ritz vectors)
        // K: Preconditioned residual blocks (Q_new blocks) in Q to keep during restart
        // Q_new: the latest block of preconditioned residuals

        status.iter_last_restart = status.iter;
        Eigen::Index qCols_old   = std::max<Eigen::Index>(0, std::min<Eigen::Index>(Q.cols() - vBlocks * b + mBlocks * b, maxRetainBlocks * b));
        Eigen::Index vCols       = V.cols();
        Eigen::Index mCols       = use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b ? b : 0;
        Eigen::Index kCols       = qCols_old;
        Eigen::Index qCols       = Q_new.cols();

        auto vOffset = 0;
        auto mOffset = vCols;
        auto kOffset = vCols + mCols;
        auto qOffset = vCols + mCols + kCols;

        MatrixType Q_keep  = Q.middleCols(kOffset, kCols);
        MatrixType HQ_keep = HQ.middleCols(kOffset, kCols);

        if(mCols > 0) M = get_mBlock(); // Generates M and HM before modifying Q
        Q.conservativeResize(N, (vCols + mCols + kCols) + Q_new.cols());
        HQ.conservativeResize(N, (vCols + mCols + kCols) + Q_new.cols());

        if(vCols > 0) Q.middleCols(vOffset, vCols) = V;
        if(mCols > 0) Q.middleCols(mOffset, mCols) = M;
        if(kCols > 0) Q.middleCols(kOffset, kCols) = Q_keep;
        if(qCols > 0) Q.middleCols(qOffset, qCols) = Q_new;

        if(vCols > 0) HQ.middleCols(vOffset, vCols) = HV;
        if(mCols > 0) HQ.middleCols(mOffset, mCols) = HM;
        if(kCols > 0) HQ.middleCols(kOffset, kCols) = HQ_keep;
        if(qCols > 0) HQ.middleCols(qOffset, qCols) = HQ_new;

        OrthMeta m;
        m.maskPolicy = MaskPolicy::COMPRESS;
        block_l2_orthonormalize(Q, HQ, m);

    } else if(oldCols != newCols) {
        // Append enrichment
        Q.conservativeResize(N, newCols);
        HQ.conservativeResize(N, newCols);
        auto copyCols          = std::min<Eigen::Index>(Q.cols(), Q_new.cols());
        Q.rightCols(copyCols)  = Q_new.leftCols(copyCols);
        HQ.rightCols(copyCols) = HQ_new;
    }
    assert_l2_orthonormal(Q);
    assert(Q.colwise().norm().minCoeff() > eps);
    assert(HQ.colwise().norm().minCoeff() > eps);
}

template<typename Scalar>
void solver_gdplusk<Scalar>::build(MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q, const MatrixType &Q_new, const MatrixType &H1Q_new,
                                   const MatrixType &H2Q_new) {
    if(status.stopReason != StopReason::none) return;
    assert(Q_new.rows() == N);
    assert(algo == OptAlgo::GDMRG);

    // Append the enrichment for this iteration

    auto oldCols = Q.cols();
    auto newCols = std::max<Eigen::Index>(1, std::min<Eigen::Index>({Q.cols() + Q_new.cols(), N}));
    if(newCols > maxBasisBlocks * b or Q.cols() == 0 or Q_new.cols() == 0) {
        // (re)start
        if(Q_new.cols() == 0 and status.iter <= status.iter_last_restart + 2) {
            // Failed to add a nonzero residual
            status.stopReason |= StopReason::saturated_basis;
            status.stopMessage.emplace_back(fmt::format("saturated basis: exhausted subspace search | iter {} | mv {} | {:.3e} s", status.iter,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            return;
        }

        status.iter_last_restart = status.iter;
        Eigen::Index qCols_old   = std::max<Eigen::Index>(0, std::min<Eigen::Index>(Q.cols() - vBlocks * b + mBlocks * b, maxRetainBlocks * b));
        Eigen::Index vCols       = V.cols();
        Eigen::Index mCols       = use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b ? b : 0;
        Eigen::Index kCols       = qCols_old;
        Eigen::Index qCols       = Q_new.cols();

        auto vOffset = 0;
        auto mOffset = vCols;
        auto kOffset = vCols + mCols;
        auto qOffset = vCols + mCols + kCols;

        MatrixType Q_keep = Q.rightCols(kCols);

        if(mCols > 0) M = get_mBlock(); // Generates M, H1M, H2M before modifying Q
        Q.conservativeResize(N, vCols + mCols + kCols + Q_new.cols());

        if(vCols > 0) Q.middleCols(vOffset, vCols) = V;
        if(mCols > 0) Q.middleCols(mOffset, mCols) = M;
        if(kCols > 0) Q.middleCols(kOffset, kCols) = Q_keep;
        if(qCols > 0) Q.middleCols(qOffset, qCols) = Q_new;

        OrthMeta m;
        m.maskPolicy = MaskPolicy::COMPRESS;
        H1Q          = MatrixType();
        H2Q          = MatrixType();
        if(use_h2_inner_product) {
            block_h2_orthonormalize_llt(Q, H1Q, H2Q, m);
        } else {
            block_l2_orthonormalize(Q, H1Q, H2Q, m);
        }

    } else if(oldCols != newCols) {
        // Append enrichment
        Q.conservativeResize(N, newCols);
        H1Q.conservativeResize(N, newCols);
        H2Q.conservativeResize(N, newCols);

        auto copyCols           = std::min<Eigen::Index>(Q.cols(), Q_new.cols());
        Q.rightCols(copyCols)   = Q_new.leftCols(copyCols);
        H1Q.rightCols(copyCols) = H1Q_new.leftCols(copyCols);
        H2Q.rightCols(copyCols) = H2Q_new.leftCols(copyCols);
        //
        OrthMeta m;
        m.maskPolicy = MaskPolicy::COMPRESS;
        m.Gram =  Q.adjoint() * H2Q;
        m.orthError = (m.Gram.cwiseAbs() - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
        if(m.orthError > normTol * std::sqrt(status.op_norm_estimate)) {
            if(use_h2_inner_product) {
                block_h2_orthonormalize_llt(Q, H1Q, H2Q, m);
            } else {
                block_l2_orthonormalize(Q, H1Q, H2Q, m);
            }
        }

    }
    if(use_h2_inner_product) {
        assert_h2_orthonormal(Q, H2Q);
    } else {
        assert_l2_orthonormal(Q);
    }
    assert(Q.colwise().norm().minCoeff() > eps);
    assert(H1Q.colwise().norm().minCoeff() > eps);
    assert(H2Q.colwise().norm().minCoeff() > eps);
}

template<typename Scalar>
void solver_gdplusk<Scalar>::set_maxExtraRitzHistory(Eigen::Index m) {
    m = std::clamp<Eigen::Index>(m, 0, 1); // mBlocks are not required
    if(m != max_mBlocks) eiglog->trace("gdplusk: max_mBlocks = {}", max_mBlocks);
    max_mBlocks                              = m;
    use_extra_ritz_vectors_in_the_next_basis = max_mBlocks > 0;
}

template<typename Scalar>
void solver_gdplusk<Scalar>::set_maxRitzResidualHistory(Eigen::Index s) {
    s = std::clamp<Eigen::Index>(s, 0, 1); // sBlocks are not required
    if(s != max_sBlocks) eiglog->trace("gdplusk: max_sBlocks = {}", max_sBlocks);
    max_sBlocks = s;
}

template<typename Scalar>
void solver_gdplusk<Scalar>::set_maxBasisBlocks(Eigen::Index bb) {
    if(bb == 0) throw except::runtime_error("maxBasisBlocks must be at least 2 | it is {}", bb);
    b  = std::min(std::max(nev, b), N / 2);
    bb = std::min<Eigen::Index>(bb, N / b);
    bb = std::max<Eigen::Index>(bb, 1);
    if(bb != maxBasisBlocks) eiglog->trace("gdplusk: maxBasisBlocks = {}", bb);
    maxBasisBlocks = bb;
    if(maxBasisBlocks == 0) throw except::runtime_error("maxBasisBlocks must be at least 1 | it is {}", maxBasisBlocks);
}

template<typename Scalar>
void solver_gdplusk<Scalar>::set_maxRetainBlocks(Eigen::Index rb) {
    b  = std::min(std::max(nev, b), N / 2);
    rb = std::min<Eigen::Index>(rb, N / b);
    if(rb != maxRetainBlocks) eiglog->trace("gdplusk: maxRetainBlocks = {}", rb);
    maxRetainBlocks = rb;
}
