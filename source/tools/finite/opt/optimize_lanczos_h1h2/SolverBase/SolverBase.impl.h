#pragma once
#include "../SolverBase.h"
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
    constexpr bool debug_solver = false;
#else
    constexpr bool debug_solver = true;
#endif
}

template<typename Scalar>
SolverBase<Scalar>::SolverBase(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1,
                               MatVecMPOS<Scalar> &H2)
    : nev(nev),   //
      ncv(ncv),   //
      algo(algo), //
      ritz(ritz), //
      H1(H1),     //
      H2(H2),     //
      V(V) {
    N         = H1.get_size();
    mps_size  = H1.get_size();
    mps_shape = H1.get_shape_mps();
    nev       = std::min(nev, N);
    ncv       = std::min(std::max(nev, ncv), N);
    b         = std::min(std::max(nev, b), N / 2);
    status.rNorms.setOnes(nev);
    status.optVal.setOnes(nev);
    status.oldVal.setOnes(nev);
    status.absDiff.setOnes(nev);
    status.relDiff.setOnes(nev);

    assert(mps_size == H1.rows());
    assert(mps_size == H2.rows());
}

template<typename Scalar>
void SolverBase<Scalar>::set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize) {
    use_preconditioner = jcbMaxBlockSize > 0;
    if(use_preconditioner) {
        H1.preconditioner = eig::Preconditioner::JACOBI;
        H2.preconditioner = eig::Preconditioner::JACOBI;
        H1.factorization  = eig::Factorization::LU;
        H2.factorization  = eig::Factorization::LLT;
        H1.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
    }
}

template<typename Scalar>
void SolverBase<Scalar>::set_chebyshevFilterRelGapThreshold(RealScalar threshold) {
    assert(threshold >= 0);
    if(threshold >= 0) { chebyshev_filter_relative_gap_threshold = threshold; }
}

template<typename Scalar>
void SolverBase<Scalar>::set_chebyshevFilterLambdaCutBias(RealScalar bias) {
    chebyshev_filter_lambda_cut_bias = std::clamp<RealScalar>(bias, eps, 1 - eps);
}

template<typename Scalar>
void SolverBase<Scalar>::set_chebyshevFilterDegree(Eigen::Index degree) {
    if(degree > 0) { chebyshev_filter_degree = degree; }
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       // input Q (orthonormal)
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

    if(lambda_cut != std::clamp(lambda_cut, lambda_min, lambda_max)) {
        eig::log->warn("lambda_cut outside range [lambda_min, lambda_max]");
        return Qref;
    }
    if(bv < eps * std::abs(av)) {
        eig::log->warn("bv < eps");
        return Qref;
    }

    RealScalar x0 = (lambda_cut - av) / bv;
    // Clamp x0 into [-1,1] to avoid NaN
    x0              = std::clamp(x0, RealScalar{-1}, RealScalar{1});
    RealScalar norm = std::cos(degree * std::acos(x0)); // = T_k(x0)

    if(degree == 1) { return (MultHX(Qref) - av * Qref) * (RealScalar{1} / bv / norm); }

    // eig::log->info("Chebyshev filter: x0={:.5e} norm={:.5e} lambda_min={:.5e} lambda_cut={:.5e} lambda_max={:.5e}", x0, norm, lambda_min, lambda_cut,
    // lambda_max);
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
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::qr_and_chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref) {
    if(Qref.cols() == 0) return Qref;
    if(chebyshev_filter_degree == 0) return Qref;
    if(T_evals.size() <= 1) return Qref;
    // calculate the gap and relative gap
    auto       select_2 = get_ritz_indices(ritz, 0, 2, T_evals);
    VectorReal evals    = T_evals(select_2);

    auto absgap = std::abs(evals(1) - evals(0));
    auto relgap = absgap / status.H_norm_est();
    assert(std::isfinite(relgap));

    if(relgap > chebyshev_filter_relative_gap_threshold) return Qref;

    RealScalar factor_more = RealScalar{1.01f};
    RealScalar factor_less = RealScalar{0.99f};
    RealScalar lambda_min  = status.min_eval_est < 0 ? status.min_eval_est * factor_more : status.min_eval_est * factor_less;
    RealScalar lambda_max  = status.max_eval_est < 0 ? status.max_eval_est * factor_less : status.max_eval_est * factor_more;
    // RealScalar lambda_cut  = std::lerp(lambda_min, lambda_max, chebyshev_filter_lambda_cut_bias);
    RealScalar lambda_cut = lambda_min + chebyshev_filter_lambda_cut_bias * (lambda_max - lambda_min);
    // RealScalar lambda_cut  = std::lerp(evals(0), evals(1), chebyshev_filter_lambda_cut_bias);
    lambda_cut = std::clamp(lambda_cut, lambda_min, lambda_max);
    // eig::log->info("Applying the chebyshev filter | gap: abs={:.5e} rel={:.5e}", absgap, relgap);
    // Re orthogonalize

    assert_allfinite(Qref);
    MatrixType Qnew = Qref;
    hhqr.compute(Qnew);
    Qnew = hhqr.householderQ().setLength(Qnew.cols()) * MatrixType::Identity(N, Qnew.cols()); //
    assert_allfinite(Qnew);
    Qnew = chebyshevFilter(Qnew, lambda_min, lambda_max, lambda_cut, chebyshev_filter_degree);
    assert_allfinite(Qnew);
    return Qnew;
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultHX(const Eigen::Ref<const MatrixType> &X) {
    status.num_matvecs += X.cols();
    switch(algo) {
        case OptAlgo::DMRG: return H1.MultAX(X);
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: {
            MatrixType H2X = H2.MultAX(X);
            MatrixType H1X = H1.MultAX(X);
            return H2X - H1.MultAX(H1X);
        }
        case OptAlgo::XDMRG: return H2.MultAX(X);
        case OptAlgo::GDMRG: return H1.MultAX(X);
        default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
    }
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultH1X(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH1X: should only be called by GDMRG");
    status.num_matvecs += X.cols();
    return H1.MultAX(X);
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultH2X(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH2X: should only be called by GDMRG");
    status.num_matvecs += X.cols();
    return H2.MultAX(X);
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultPX(const Eigen::Ref<const MatrixType> &X) {
    // Preconditioning
    status.num_precond += X.cols();
    switch(algo) {
        case OptAlgo::DMRG: return H1.MultPX(X);
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: return H2.MultPX(X);
        case OptAlgo::XDMRG: return H2.MultPX(X);
        case OptAlgo::GDMRG: throw except::runtime_error("LOBPCG: GDMRG is not suitable, use GeneralizedLanczos instead");
        default: throw except::runtime_error("LOBPCG: unknown algorithm {}", enum2sv(algo));
    }
}
template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultP1X(const Eigen::Ref<const MatrixType> &X) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP1X: should only be called by GDMRG");
    status.num_precond += X.cols();
    return H1.MultPX(X);
}

template<typename Scalar>
typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::MultP2X(const Eigen::Ref<const MatrixType> &X) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP2X: should only be called by GDMRG");
    status.num_precond += X.cols();
    // return X;
    return H2.MultPX(X);
}

template<typename Scalar> typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::get_wBlock() {
    // We add Lanczos-style residual blocks
    W = (algo == OptAlgo::GDMRG) ? H2V : HV;
    A = V.adjoint() * W;

    // 3) Subtract projections to A and B once
    W.noalias() -= V * A; // Qi * Qi.adjoint()*H*Qi
    if(V_prev.rows() == N and V_prev.cols() == b) {
        B = V_prev.adjoint() * W;
        W.noalias() -= V_prev * B.adjoint();
    }
    assert_allfinite(W);
    return W;
}

template<typename Scalar> typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::get_mBlock() {
    // M are the b next-best ritz vectors from the previous iteration
    if(use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b) {
        auto top_2b_indices = get_ritz_indices(ritz, b, b, T_evals);
        auto Z              = T_evecs(Eigen::all, top_2b_indices); // Selected subspace eigenvectors
        M                   = Q * Z;                               // Regular Rayleigh-Ritz
        // Transform the basis with applied operators
        if(algo == OptAlgo::GDMRG) {
            H1M = H1Q * Z;
            H2M = H2Q * Z;
        } else {
            HM = HQ * Z;
        }
    }
    return M;
}

template<typename Scalar> typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::get_sBlock() {
    // Make a residual block "S = (HQ-λQ)"
    if(S.cols() != b) {
        auto Y = T_evals(status.optIdx);
        if(algo == OptAlgo::GDMRG) {
            S = HV - V * Y.asDiagonal();
        } else {
            S = H1V - H2V * Y.asDiagonal();
        }
    }
    assert_allfinite(S);
    return S;
}

template<typename Scalar> typename SolverBase<Scalar>::MatrixType SolverBase<Scalar>::get_rBlock() {
    // Get a random block
    return MatrixType::Random(N, b);
}

template<typename Scalar>
const typename SolverBase<Scalar>::MatrixType &SolverBase<Scalar>::get_HQ() {
    // HQ   = MultHX(Q);
    // return HQ;
    if(status.iter == i_HQ) {
        // assert((HQ - MultHX(Q)).norm() < 100 * eps);
        return HQ;
    }
    i_HQ = status.iter;
    HQ   = MultHX(Q);
    return HQ;
}

template<typename Scalar>
const typename SolverBase<Scalar>::MatrixType &SolverBase<Scalar>::get_HQ_cur() {
    // HQ_cur   = MultHX(Q.middleCols((qBlocks - 1)*b, b));
    // return HQ_cur;
    assert(qBlocks >= 1);
    if(status.iter == i_HQ) {
        HQ_cur = HQ.middleCols((qBlocks - 1) * b, b);
        // assert((HQ_cur - MultHX(Q.middleCols((qBlocks - 1) * b, b))).norm() < 100 * eps);
        return HQ_cur;
    }
    if(status.iter == i_HQ_cur) {
        // assert((HQ_cur - MultHX(Q.middleCols((qBlocks - 1) * b, b))).norm() < 100 * eps);
        return HQ_cur;
    }
    i_HQ_cur = status.iter;
    HQ_cur   = MultHX(Q.middleCols((qBlocks - 1) * b, b));
    return HQ_cur;
}

template<typename Scalar>
void SolverBase<Scalar>::unset_HQ() {
    i_HQ = -1;
}
template<typename Scalar>
void SolverBase<Scalar>::unset_HQ_cur() {
    i_HQ_cur = -1;
    i_HQ     = -1;
}

template<typename Scalar>
void SolverBase<Scalar>::orthonormalize(const Eigen::Ref<const MatrixType> X,       // (N, xcols)
                                        Eigen::Ref<MatrixType>             Y,       // (N, ycols)
                                        RealScalar                         normTol, // The largest allowed norm error
                                        RealScalar                         orthTol, // The largest allowed orthonormality error
                                        Eigen::Ref<VectorIdxT>             mask     // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == Y.cols() / b && "Mask size must match number of blocks in Y.");
    // eig::log->info("mask: {}", mask);
    if constexpr(settings::debug_solver) {
        MatrixType XX         = X.adjoint() * X;
        auto       XorthError = (XX - MatrixType::Identity(X.cols(), X.cols())).norm();
        if(XorthError >= orthTol) {
            eig::log->info("XX: \n{}\n", linalg::matrix::to_string(XX, 8));
            eig::log->info("X normError: {:.5e}", XorthError);
        }
        // assert(XnormError < normTol);
    }
    if constexpr(settings::debug_solver) {
        bool allFinite = X.allFinite();
        if(!allFinite) { eig::log->warn("X is not all finite: \n{}\n", linalg::matrix::to_string(X, 8)); }
        assert(allFinite);
    }
    if constexpr(settings::debug_solver) {
        bool allFinite = Y.allFinite();
        if(!allFinite) { eig::log->warn("Y is not all finite: \n{}\n", linalg::matrix::to_string(Y, 8)); }
        assert(allFinite);
    }

    // if constexpr(settings::debug_solver) {
    // auto XYnormError = (X.adjoint() * Y).cwiseAbs().maxCoeff();
    // eig::log->info("X: \n{}\n", linalg::matrix::to_string(X, 8));
    // eig::log->info("Y: \n{}\n", linalg::matrix::to_string(Y, 8));
    // eig::log->info("XY normError before cleaning: {:.5e}", XYnormError);
    // }

    const Eigen::Index n_blocks_y = Y.cols() / b;
    const Eigen::Index n_blocks_x = X.cols() / b;
    const Eigen::Index xcols      = X.cols();
    const Eigen::Index ycols      = Y.cols();

    if(xcols == 0 || ycols == 0) return;

    // DGKS clean Y against X and orthonormalize Y
    for(int rep = 0; rep < 2; ++rep) {
        for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
            if(mask(blk_y) == 0) continue;
            auto Yblock = Y.middleCols(blk_y * b, b);
            // Mask if numerically zero
            if(Yblock.colwise().norm().minCoeff() < normTol) {
                mask(blk_y) = 0;
                Yblock.setZero();
                eig::log->info("mask before DGKS X: {}", mask);
                continue;
            }
            for(Eigen::Index blk_x = 0; blk_x < n_blocks_x; ++blk_x) {
                auto Xblock = X.middleCols(blk_x * b, b);
                Yblock.noalias() -= Xblock * (Xblock.adjoint() * Yblock).eval(); // Remove projection
            }

            // Mask if numerically zero
            if(Yblock.colwise().norm().minCoeff() < normTol) {
                mask(blk_y) = 0;
                Yblock.setZero();
                eig::log->info("mask after DGKS X: {}", mask);
                continue;
            }

            // Orthonormalize this block
            hhqr.compute(Yblock);
            Yblock = hhqr.householderQ().setLength(Yblock.cols()) * MatrixType::Identity(Yblock.rows(), Yblock.cols());

            // Mask if numerically zero
            if(Yblock.colwise().norm().minCoeff() < normTol) {
                mask(blk_y) = 0;
                Yblock.setZero();
                eig::log->info("mask after QR y: {}", mask);
                continue;
            }

            // Forward MGS

            for(Eigen::Index blk2_y = blk_y + 1; blk2_y < n_blocks_y; ++blk2_y) {
                if(mask(blk2_y) == 0) continue;
                auto Yblock2 = Y.middleCols(blk2_y * b, b);
                Yblock2.noalias() -= Yblock * (Yblock.adjoint() * Yblock2);
                if(Yblock2.colwise().norm().minCoeff() < normTol) {
                    mask(blk2_y) = 0;
                    Yblock2.setZero();
                    continue;
                }
            }
        }
    }

    std::vector<Eigen::Index> active_ycols;
    for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
        if(mask(j) == 1) {
            for(Eigen::Index col = 0; col < b; ++col) active_ycols.push_back(j * b + col);
        }
    }
    // eig::log->info("mask final: {}", mask);
    // eig::log->info("active_ycols: {}", active_ycols);
    if(active_ycols.empty()) { return; }
    auto Ymask = Y(Eigen::all, active_ycols);

    if constexpr(settings::debug_solver) {
        auto YorthError = (Ymask.adjoint() * Ymask - MatrixType::Identity(Ymask.cols(), Ymask.cols())).norm();
        if(YorthError > orthTol) eig::log->info("Y normError: {:.5e}", YorthError);
        assert(YorthError <= orthTol);
    }

    if constexpr(settings::debug_solver) {
        MatrixType XY          = X.adjoint() * Ymask;
        auto       XYorthError = XY.cwiseAbs().maxCoeff();
        if(XYorthError > orthTol) {
            eig::log->info("X: \n{}\n", linalg::matrix::to_string(X, 8));
            eig::log->info("Y: \n{}\n", linalg::matrix::to_string(Y, 8));
            eig::log->info("XY: \n{}\n", linalg::matrix::to_string(XY, 8));
            eig::log->info("XY orthError after DGKS: {:.5e}", XYorthError);
        }
        assert(XYorthError <= orthTol);
    }
    if constexpr(settings::debug_solver) {
        bool allFinite = Ymask.allFinite();
        if(!allFinite) { eig::log->warn("Y is not all finite: \n{}\n", linalg::matrix::to_string(Y, 8)); }
        assert(allFinite);
    }
}

template<typename Scalar>
void SolverBase<Scalar>::compress_cols(MatrixType       &X,   // (N, ycols)
                                       const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == X.cols() / b && "Mask size must match number of blocks in X.");
    const Eigen::Index n_blocks_x = X.cols() / b;

    // We can now squeeze out blocks zeroed out by DGKS
    // Get the block indices that we should keep
    std::vector<Eigen::Index> active_columns;
    active_columns.reserve(n_blocks_x * b);
    for(Eigen::Index j = 0; j < n_blocks_x; ++j) {
        if(mask(j) == 1) {
            for(Eigen::Index k = 0; k < b; ++k) active_columns.push_back(j * b + k);
        }
    }
    active_columns.shrink_to_fit();
    if(active_columns.size() != static_cast<size_t>(X.cols())) {
        X = X(Eigen::all, active_columns).eval(); // Shrink keeping only nonzeros
    }
}

template<typename Scalar>
void SolverBase<Scalar>::compress_rows_and_cols(MatrixType       &X,   // (N, ycols)
                                                const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == X.cols() / b && "Mask size must match number of blocks in X.");
    assert(mask.size() == X.rows() / b && "Mask size must match number of blocks in X.");
    const Eigen::Index n_blocks_x = X.cols() / b;

    // We can now squeeze out blocks zeroed out by DGKS
    // Get the block indices that we should keep
    std::vector<Eigen::Index> active_indices;
    active_indices.reserve(n_blocks_x * b);
    for(Eigen::Index j = 0; j < n_blocks_x; ++j) {
        if(mask(j) == 1) {
            for(Eigen::Index k = 0; k < b; ++k) active_indices.push_back(j * b + k);
        }
    }
    active_indices.shrink_to_fit();
    if(active_indices.size() != static_cast<size_t>(X.cols())) {
        X = X(active_indices, active_indices).eval(); // Shrink keeping only nonzeros
    }
    assert_allfinite(X);
}

template<typename Scalar> void SolverBase<Scalar>::assert_allfinite(const Eigen::Ref<const MatrixType> X, const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;
        bool allFinite = X.allFinite();
        if(!allFinite) {
            eig::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            eig::log->warn("X is not all finite: \n{}\n", linalg::matrix::to_string(X, 8));
            throw except::runtime_error("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name());
        }
    }
}

template<typename Scalar> void SolverBase<Scalar>::assert_orthonormal(const Eigen::Ref<const MatrixType> X, RealScalar threshold,
                                                                      const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;
        MatrixType XX        = X.adjoint() * X;
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            eig::log->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            eig::log->warn("X orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), orthError, threshold);
        }
    }
}
template<typename Scalar> void SolverBase<Scalar>::assert_orthogonal(const Eigen::Ref<const MatrixType> X, const Eigen::Ref<const MatrixType> Y,
                                                                     RealScalar threshold, const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0 || Y.cols() == 0) return;
        MatrixType XY          = X.adjoint() * Y;
        auto       XYorthError = XY.cwiseAbs().maxCoeff();
        if(XYorthError > threshold) {
            eig::log->info("XY: \n{}\n", linalg::matrix::to_string(XY, 8));
            eig::log->info("XY orthError: {:.5e}", XYorthError);
            throw except::runtime_error("{}:{}: {}: matrices are not orthogonal: error = {:.5e} > threshold {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), XYorthError, threshold);
        }
    }
}

template<typename Scalar>
std::vector<Eigen::Index> SolverBase<Scalar>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) {
    // Select eigenvalues
    std::vector<Eigen::Index> indices;
    assert(num <= evals.size());
    auto ritz_internal = ritz;
    // if(algo == OptAlgo::GDMRG) {
    //     // Map to opposite ritz
    //     switch(ritz) {
    //         case OptRitz::LM: ritz_internal = OptRitz::SM; break;
    //         case OptRitz::LR: ritz_internal = OptRitz::SM; break;
    //         case OptRitz::SM: ritz_internal = OptRitz::LM; break;
    //         case OptRitz::SR: ritz_internal = OptRitz::LR; break;
    //         default: break;
    //     }
    // }
    switch(ritz_internal) {
        case OptRitz::SR: indices = getIndices(evals, offset, num, std::less<RealScalar>()); break;
        case OptRitz::LR: indices = getIndices(evals, offset, num, std::greater<RealScalar>()); break;
        case OptRitz::SM: indices = getIndices(evals.cwiseAbs(), offset, num, std::less<RealScalar>()); break;
        case OptRitz::LM: indices = getIndices(evals.cwiseAbs(), offset, num, std::greater<RealScalar>()); break;
        case OptRitz::IS: [[fallthrough]];
        case OptRitz::TE: [[fallthrough]];
        case OptRitz::NONE: {
            if(std::isnan(status.initVal))
                throw except::runtime_error("Ritz [{} ({})] does not work when lanczos.status.initVal is nan", enum2sv(ritz), enum2sv(ritz_internal));
            indices = getIndices((evals.array() - status.initVal).cwiseAbs(), offset, num, std::less<RealScalar>());
            break;
        }
        default: throw except::runtime_error("unhandled ritz: [{} ({})]", enum2sv(ritz), enum2sv(ritz_internal));
    }
    return indices;
}

template<typename Scalar>
void SolverBase<Scalar>::init() {
    assert(H1.rows() == H1.cols() && "H1 must be square");
    assert(H2.rows() == H2.cols() && "H2 must be square");
    assert(N == H1.rows() && "H1 and H2 must have same dimension");
    assert(N == H2.rows() && "H1 and H2 must have same dimension");
    nev = std::min(nev, N);
    ncv = std::min(std::max(nev, ncv), N);
    b   = std::min(std::max(nev, b), N / 2);
    Eigen::ColPivHouseholderQR<MatrixType> cpqr;

    // Step 0: Construct and orthonormalize the initial block V.
    // We aim to construct V = [v[0]...v[b-1]], where v are ritz eigenvectors,
    // If V has fewer than b columns, we pad it with random vectors and orthonormalize with ColPivHouseholderQR.
    // If V has more than b columns, we discard the overshooting columns after QR.
    // If after QR we have fewer than b columns, we pad again (this is a very unlikely event)
    assert(V.size() == 0 or N == V.rows());
    for(long i = 0; i < 2; ++i) {
        if(V.cols() < b) {
            // Pad with random vectors
            auto vc = V.cols();
            V.conservativeResize(N, b);
            V.rightCols(b - vc).setRandom();
        }
        // Orthonormalize V.
        // Discard columns if there are more than b (this is not expected, but also not an error)
        cpqr.compute(V);
        cpqr.setThreshold(orthTolQ);
        auto rank = std::min(cpqr.rank(), b);
        V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank) * cpqr.colsPermutation().transpose();
        if(V.cols() == b) break;
    }

    assert(V.cols() == b);
    assert_orthonormal(V, orthTolQ);
    if(status.iter == 0) {
        // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
        if(algo == OptAlgo::GDMRG) {
            Q                                                       = V;
            H1Q                                                     = MultH1X(Q);
            H2Q                                                     = MultH2X(Q);
            MatrixType                                           T1 = Q.adjoint() * H1Q;
            MatrixType                                           T2 = Q.adjoint() * H2Q;
            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es_seed(T1, T2, Eigen::Ax_lBx);
            T_evecs       = es_seed.eigenvectors().colwise().normalized();
            T_evals       = es_seed.eigenvalues();
            status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
            auto Z        = T_evecs(Eigen::all, status.optIdx);
            auto Y        = T_evals(status.optIdx);
            V             = (Q * Z).normalized(); // Now V has b columns mixed according to the selected columns in T_evecs
            H1V           = H1Q * Z;
            H2V           = H2Q * Z;
            S             = H1V - H2V * Y.asDiagonal();
            status.rNorms = S.colwise().norm();
            status.optVal = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
            Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
            status.max_eval_est = std::max({status.max_eval_est, es1.eigenvalues().cwiseAbs().maxCoeff(), es2.eigenvalues().cwiseAbs().maxCoeff()});
            status.min_eval_est = std::min({status.min_eval_est, es1.eigenvalues().cwiseAbs().minCoeff(), es2.eigenvalues().cwiseAbs().minCoeff()});
            RealScalar min_sep =
                T_evals.size() <= 1 ? RealScalar{1} : (T_evals.bottomRows(T_evals.size() - 1) - T_evals.topRows(T_evals.size() - 1)).cwiseAbs().minCoeff();
            auto select1 = get_ritz_indices(ritz, 0, 1, T_evals);
            status.condition =
                (es1.eigenvalues().cwiseAbs().maxCoeff() + T_evals(select1).cwiseAbs().coeff(0) * es2.eigenvalues().cwiseAbs().maxCoeff()) / min_sep;
        } else {
            Q  = V;
            HQ = MultHX(V);
            T  = Q.adjoint() * HQ;
            Eigen::SelfAdjointEigenSolver<MatrixType> es_seed(T);
            T_evecs             = es_seed.eigenvectors();
            T_evals             = es_seed.eigenvalues();
            status.optIdx       = get_ritz_indices(ritz, 0, b, T_evals);
            auto Z              = T_evecs(Eigen::all, status.optIdx);
            auto Y              = T_evals(status.optIdx);
            V                   = (Q * Z).normalized(); // Now V has b columns mixed according to the selected columns in T_evecs
            HV                  = HQ * Z;
            S                   = HV - V * Y.asDiagonal();
            status.rNorms       = S.colwise().norm();
            status.optVal       = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            status.max_eval_est = T_evals.maxCoeff();
            status.min_eval_est = T_evals.minCoeff();
            status.condition    = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
        }
    }

    assert(V.cols() == b);
    assert(V.allFinite());
    assert(std::abs((V.adjoint() * V).norm() - std::sqrt<RealScalar>(b)) < orthTolQ);
    eig::log->info("iter -1| mv {:5} | optVal {::.16f} | blk {:2} | b {} | ritz {} | rNormTol {:.3e} | tol {:.2e} | rNorms = {::.8e}", status.num_matvecs,
                   fv(status.optVal), Q.cols() / b, b, enum2sv(ritz), rnormTol(), tol, fv(VectorReal(status.rNorms.topRows(nev))));

    // Now V has b orthonormalized ritz vectors
}

template<typename Scalar>
void SolverBase<Scalar>::diagonalizeT() {
    if(algo == OptAlgo::GDMRG) return diagonalizeT1T2();
    if(status.stopReason != StopReason::none) return;
    if(Q.cols() == 0) return;
    if(HQ.cols() == 0) return;
    assert(Q.cols() == HQ.cols());
    status.rNorms = {};

    MatrixType T = Q.adjoint() * HQ;
    T            = RealScalar{0.5f} * (T + T.adjoint()).eval(); // Symmetrize
    assert(T.colwise().norm().minCoeff() != 0);

    Eigen::SelfAdjointEigenSolver<MatrixType> es(T, Eigen::ComputeEigenvectors);
    T_evals = es.eigenvalues();
    T_evecs = es.eigenvectors();

    status.max_eval_est = std::max(status.max_eval_est, T_evals.cwiseAbs().maxCoeff());
    status.min_eval_est = std::min(status.min_eval_est, T_evals.cwiseAbs().minCoeff());
    status.condition    = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
}

template<typename Scalar>
void SolverBase<Scalar>::diagonalizeT1T2() {
    if(status.stopReason != StopReason::none) return;
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("diagonalizeT1T2() is only implemented for GDMRG");
    status.rNorms = {};

    MatrixType T1 = Q.adjoint() * H1Q;
    MatrixType T2 = Q.adjoint() * H2Q;

    // Symmetrize
    T1 = RealScalar{0.5f} * (T1 + T1.adjoint()).eval();
    T2 = RealScalar{0.5f} * (T2 + T2.adjoint()).eval();
    assert(T1.rows() == T2.rows());
    assert(T1.cols() == T2.cols());

    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T1, T2, Eigen::Ax_lBx);
    if(es.info() == Eigen::Success) {
        T_evals = es.eigenvalues();
        T_evecs = es.eigenvectors().colwise().normalized();
    } else {
        // Minimize variance instead, but we should invert the spectrum
        eig::log->warn("Generalized eigenvalue problem failed, using variance minimization instead. \n");
        Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2 - T1 * T1);
        T_evals = es2.eigenvalues().inverse();
        T_evecs = es2.eigenvectors();
        assert_allfinite(T_evecs);
    }

    Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
    Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
    status.max_eval_est = std::max({status.max_eval_est, es1.eigenvalues().cwiseAbs().maxCoeff(), es2.eigenvalues().cwiseAbs().maxCoeff()});
    status.min_eval_est = std::min({status.min_eval_est, es1.eigenvalues().cwiseAbs().minCoeff(), es2.eigenvalues().cwiseAbs().minCoeff()});

    status.condition = std::max(es1.eigenvalues().cwiseAbs().maxCoeff() / es1.eigenvalues().cwiseAbs().minCoeff(),
                                es2.eigenvalues().cwiseAbs().maxCoeff() / es2.eigenvalues().cwiseAbs().minCoeff());

    RealScalar min_sep = T_evals.size() <= 1 ? RealScalar{1} : //
                             (T_evals.bottomRows(T_evals.size() - 1) - T_evals.topRows(T_evals.size() - 1)).cwiseAbs().minCoeff();
    auto       select1 = get_ritz_indices(ritz, 0, 1, T_evals);
    status.condition   = (es1.eigenvalues().cwiseAbs().maxCoeff() + T_evals(select1).cwiseAbs().coeff(0) * es2.eigenvalues().cwiseAbs().maxCoeff()) / min_sep;
}

template<typename Scalar>
void SolverBase<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    auto Z = T_evecs(Eigen::all, optIdx); // Selected subspace eigenvectors
    auto Y = T_evals(optIdx);             // Selected subspace eigenvalues
    V      = Q * Z;                       // Regular Rayleigh-Ritz

    // Transform the basis with applied operators
    HV = HQ * Z;

    S      = HV - V * Y.asDiagonal(); // Residual vector
    rNorms = S.colwise().norm();      // Residual norm
}

template<typename Scalar>
void SolverBase<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                            VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    auto Z = T_evecs(Eigen::all, optIdx); // Selected subspace eigenvectors
    auto Y = T_evals(optIdx);             // Selected subspace eigenvalues
    V      = Q * Z;                       // Regular Rayleigh-Ritz

    // Transform the basis with applied operators
    H1V = H1Q * Z;
    H2V = H2Q * Z;

    S      = H1V - H2V * Y.asDiagonal(); // Residual vector
    rNorms = S.colwise().norm();         // Residual norm
}

/*!
 * Extract Ritz vectors, optionally performing refined Ritz extraction.
 * If chebyshev filtering is enabled, use the filtered basis (X/HX);
 * otherwise use the unfiltered basis (Q/HQ).
 * The refined Ritz extraction uses SVD to minimize the residual norm
 * in the projected subspace.
 */
template<typename Scalar>
void SolverBase<Scalar>::extractRitzVectors() {
    if(status.stopReason != StopReason::none) return;
    if(T_evals.size() < b) return;
    assert_orthonormal(Q, orthTolQ);
    // Here we assume that Q is orthonormal.

    // Get the indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
    if(algo == OptAlgo::GDMRG) {
        extractRitzVectors(status.optIdx, V, H1V, H2V, S, status.rNorms);
    } else {
        extractRitzVectors(status.optIdx, V, HV, S, status.rNorms);
    }
}

template<typename Scalar>
void SolverBase<Scalar>::refineRitzVectors(MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S, VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    for(Eigen::Index j = 0; j < V.cols(); ++j) {
        const auto &theta = T_evals(status.optIdx[j]);
        auto        v     = V.col(j);
        auto        h1v   = H1V.col(j);
        auto        h2v   = H2V.col(j);
        auto        s     = S.col(j);
        auto       &rNorm = rNorms(j);
        MatrixType  M     = H1Q - theta * H2Q;

        svd.compute(M, Eigen::ComputeThinV);

        Eigen::Index min_idx;
        svd.singularValues().minCoeff(&min_idx);

        RealScalar refinedRnorm = svd.singularValues()(min_idx);
        if(svd.info() == Eigen::Success and refinedRnorm < 10 * rNorm) {
            // Accept the solution
            auto Z_ref = svd.matrixV().col(min_idx);
            v          = (Q * Z_ref).normalized();
            h1v        = (H1Q * Z_ref);
            h2v        = (H2Q * Z_ref);
            s          = (h1v - h2v * theta);
            rNorm      = refinedRnorm;
        } else {
            eig::log->warn("refinement failed on ritz vector {} | rnorm: refined={:.5e}, standard={:.5e} | info {} ", j, rNorm, refinedRnorm,
                           static_cast<int>(svd.info()));
        }
    }
}
template<typename Scalar>
void SolverBase<Scalar>::refineRitzVectors(MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    for(Eigen::Index j = 0; j < V.cols(); ++j) {
        const auto &theta = T_evals(status.optIdx[j]);
        auto        v     = V.col(j);
        auto        hv    = HV.col(j);
        auto        s     = S.col(j);
        auto       &rNorm = rNorms(j);
        MatrixType  M     = HQ - theta * Q;

        svd.compute(M, Eigen::ComputeThinV);

        Eigen::Index min_idx;
        svd.singularValues().minCoeff(&min_idx);

        RealScalar refinedRnorm = svd.singularValues()(min_idx);
        if(svd.info() == Eigen::Success and refinedRnorm < 10 * rNorm) {
            // Accept the solution
            auto Z_ref = svd.matrixV().col(min_idx);
            v          = (Q * Z_ref).normalized();
            hv         = (HQ * Z_ref);
            s          = (hv - v * theta);
            rNorm      = refinedRnorm;
        } else {
            eig::log->warn("refinement failed on ritz vector {} | rnorm: refined={:.5e}, standard={:.5e} | info {} ", j, rNorm, refinedRnorm,
                           static_cast<int>(svd.info()));
        }
    }
}

template<typename Scalar>
void SolverBase<Scalar>::refineRitzVectors() {
    if(!use_refined_rayleigh_ritz) return;
    if(status.rNorms.size() == 0) throw except::runtime_error("refineRitzVectors() called before extractRitzVectors()");
    // Refined extraction
    if(algo == OptAlgo::GDMRG) {
        refineRitzVectors(V, H1V, H2V, S, status.rNorms);
    } else {
        refineRitzVectors(V, HV, S, status.rNorms);
    }
}

template<typename Scalar>
void SolverBase<Scalar>::updateStatus() {
    if(status.stopReason != StopReason::none) return;
    // Eigenvalues are sorted in ascending order.
    status.oldVal  = status.optVal;
    status.optVal  = T_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
    status.absDiff = (status.optVal - status.oldVal).cwiseAbs();
    status.relDiff = status.absDiff.array() / (RealScalar{0.5} * (status.optVal + status.oldVal).array());

    if(status.rNorms.topRows(nev).maxCoeff() < rnormTol()) {
        status.stopMessage.emplace_back(fmt::format("converged rNorms {::.3e} < tol {:.3e}", fv(status.rNorms), rnormTol()));
        status.stopReason |= StopReason::converged_rnormTol;
    }
    if(status.iter >= std::max(1l, max_iters)) {
        status.stopMessage.emplace_back(fmt::format("iter ({}) >= maxiter ({})", status.iter, max_iters));
        status.stopReason |= StopReason::max_iterations;
    }
    if(status.num_matvecs >= std::max(1l, max_matvecs)) {
        status.stopMessage.emplace_back(fmt::format("num_matvecs ({}) >= max_matvecs ({})", status.num_matvecs, max_matvecs));
        status.stopReason |= StopReason::max_matvecs;
    }

    RealScalar absgap = std::numeric_limits<RealScalar>::quiet_NaN();
    RealScalar relgap = std::numeric_limits<RealScalar>::quiet_NaN();

    if(T_evals.size() >= 2) {
        auto       select_2 = get_ritz_indices(ritz, 0, 2, T_evals);
        VectorReal evals    = T_evals(select_2);
        absgap              = std::abs(evals(1) - evals(0));
        relgap              = absgap / status.H_norm_est();
    }

    eig::log->info("iter {} | mv {:5} | optVal {::.16f} | blk {:2} | b {} | ritz {} | rNormTol {:.3e} | tol {:.2e} | rNorms = {::.8e} | cond {:.2e} | gap "
                   "{:.3e} (rel {:.3e})",
                   status.iter, status.num_matvecs, fv(status.optVal), Q.cols() / b, b, enum2sv(ritz), rnormTol(), tol,
                   fv(VectorReal(status.rNorms.topRows(nev))), status.condition, absgap, relgap);
}