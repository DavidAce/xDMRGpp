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
void SolverBase<Scalar>::set_chebyshevFilterDegree(Eigen::Index degree) {
    if(degree > 0) { chebyshev_filter_degree = degree; }
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
        case OptAlgo::GDMRG: throw except::runtime_error("LOBPCG: GDMRG is not suitable, use GeneralizedLanczos instead");
        default: throw except::runtime_error("LOBPCG: unknown algorithm {}", enum2sv(algo));
    }
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
std::vector<Eigen::Index> SolverBase<Scalar>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) {
    // Select eigenvalues
    std::vector<Eigen::Index> indices;
    assert(num <= evals.size());
    switch(ritz) {
        case OptRitz::SR: indices = getIndices(evals, offset, num, std::less<RealScalar>()); break;
        case OptRitz::LR: indices = getIndices(evals, offset, num, std::greater<RealScalar>()); break;
        case OptRitz::SM: indices = getIndices(evals.cwiseAbs(), offset, num, std::less<RealScalar>()); break;
        case OptRitz::LM: indices = getIndices(evals.cwiseAbs(), offset, num, std::greater<RealScalar>()); break;
        case OptRitz::IS: [[fallthrough]];
        case OptRitz::TE: [[fallthrough]];
        case OptRitz::NONE: {
            if(std::isnan(status.initVal))
                throw except::runtime_error("Ritz [{} ({})] does not work when lanczos.status.initVal is nan", enum2sv(ritz), enum2sv(ritz));
            indices = getIndices((evals.array() - status.initVal).cwiseAbs(), offset, num, std::less<RealScalar>());
            break;
        }
        default: throw except::runtime_error("unhandled ritz: [{} ({})]", enum2sv(ritz), enum2sv(ritz));
    }
    return indices;
}

template<typename Scalar>
void SolverBase<Scalar>::init() {
    assert(H1.rows() == H1.cols() && "H1 must be square");
    assert(H2.rows() == H2.cols() && "H2 must be square");
    assert(N == H1.rows() && "H1 and H2 must have same dimension");
    assert(N == H2.rows() && "H1 and H2 must have same dimension");
    nev                                      = std::min(nev, N);
    ncv                                      = std::min(std::max(nev, ncv), N);
    b                                        = std::min(std::max(nev, b), N / 2);
    Eigen::Index                           m = ncv;
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
    assert(std::abs((V.adjoint() * V).norm() - std::sqrt<RealScalar>(b)) < orthTolQ);
    if(status.iter == 0) {
        // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
        MatrixType HV = MultHX(V);
        T = V.adjoint() * HV;
        Eigen::SelfAdjointEigenSolver<MatrixType> es_seed(T);
        T_evecs       = es_seed.eigenvectors();
        T_evals       = es_seed.eigenvalues();
        status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
        auto Z        = T_evecs(Eigen::all, status.optIdx);
        V             = (V * Z).eval(); // Now V has b columns mixed according to the selected columns in T_evecs

        // HQ       = MultHX(V);
        // HQ_cur   = HQ;
        // qBlocks  = 1;
        // i_HQ     = status.iter;
        // i_HQ_cur = status.iter;

        status.rNorms        = (HV * Z - V * T_evals.asDiagonal()).colwise().norm();
        status.optVal        = T_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
        status.H_norm_approx = std::max(status.H_norm_approx, T_evals.cwiseAbs().maxCoeff());
    }

    assert(m >= 1);
    assert(V.cols() == b);
    assert(V.allFinite());
    assert(std::abs((V.adjoint() * V).norm() - std::sqrt<RealScalar>(b)) < orthTolQ);

    // Now V has b orthonormalized ritz vectors
}

template<typename Scalar>
void SolverBase<Scalar>::diagonalizeT() {
    if(status.stopReason != StopReason::none) return;
    if(T.rows() == 0) return;
    assert(T.colwise().norm().minCoeff() != 0);

    // Eigen::SelfAdjointEigenSolver<MatrixType> es(T, Eigen::ComputeEigenvectors);
    // T_evals = es.eigenvalues();
    // T_evecs = es.eigenvectors();

    // MatrixType S = HQ.adjoint() * HQ;
    // Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T, S, Eigen::Ax_lBx);
    // T_evals              = es.eigenvalues().cwiseInverse().reverse();
    // T_evecs              = es.eigenvectors().rowwise().reverse();

    auto       solver = eig::solver();
    MatrixType T_temp = T; // T_temp is destroyed
    solver.eig<eig::Form::SYMM>(T_temp.data(), T_temp.rows(), eig::Vecs::ON);
    T_evals = eig::view::get_eigvals<RealScalar>(solver.result);
    T_evecs = eig::view::get_eigvecs<Scalar>(solver.result);
    // assert(T_evals.cwiseAbs().minCoeff() != 0);
    status.H_norm_approx = std::max(status.H_norm_approx, T_evals.cwiseAbs().maxCoeff());
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
    if(T.rows() < b) return;

    // Get the indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);

    // Refined extraction
    Eigen::JacobiSVD<MatrixType> svd;
    if(use_refined_rayleigh_ritz) {
        auto Z = T_evecs(Eigen::all, status.optIdx);
        // Get the regular ritz vector residual
        status.rNorms = ((get_HQ() - Q * T) * Z).colwise().norm();
        for(Eigen::Index j = 0; j < static_cast<Eigen::Index>(status.optIdx.size()); ++j) {
            auto &theta = T_evals(status.optIdx[j]);

            MatrixType M = HQ - theta * Q;
            svd.compute(M, Eigen::ComputeThinV);
            Eigen::Index min_idx;
            svd.singularValues().minCoeff(&min_idx);

            auto refinedRnorm = svd.singularValues()(min_idx); // We got it for free here, no need to calculate it later
            if(refinedRnorm < 10 * status.rNorms(j)) {
                // Accept the solution
                V.col(j)         = (Q * svd.matrixV().col(min_idx)).normalized();
                status.rNorms(j) = refinedRnorm;
                // The refined eigenvalues aren't really needed, so we do not spend a matvec on them
                // RealScalar newTheta = (V.col(j).adjoint() * MultHX(V.col(j))).real().coeff(0);
                // eig::log->info("evals({0}) = {1:.16f} | V({0})*HV({0}) = {2:.16f} | sv {3:.5e}", j, theta, newTheta, svd.singularValues()(min_idx));
                // theta = newTheta;
            } else {
                eig::log->trace("refinement failed on ritz vector {} | rnorm: ref={:.5e}, reg={:.5e}  ", j, refinedRnorm, status.rNorms(j));
                V.col(j) = Q * T_evecs(Eigen::all, status.optIdx[j]);
            }
        }
    } else {
        // Regular Rayleigh-Ritz
        V = Q * T_evecs(Eigen::all, status.optIdx);
    }

    if(use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b) {
        auto top_2b_indices = get_ritz_indices(ritz, b, b, T_evals);
        M                   = Q * T_evecs(Eigen::all, top_2b_indices);
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

    // if(status.iter >= 1) {
    //     if(status.absDiff.maxCoeff() < absDiffTol and status.iter >= 3) {
    //         status.stopMessage.emplace_back(fmt::format("saturated: abs diff {::.3e} < tol {:.3e}", fv(status.absDiff), absDiffTol));
    //         status.stopReason |= StopReason::saturated_absDiffTol;
    //     }
    //     if(status.relDiff.maxCoeff() < relDiffTol and status.iter >= 3) {
    //         status.stopMessage.emplace_back(fmt::format("saturated: rel diff {::.3e} < {:.3e}", fv(status.relDiff), relDiffTol));
    //         status.stopReason |= StopReason::saturated_relDiffTol;
    //     }
    // }

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
    eig::log->info("iter {} | mv {:5} | optVal {::.16f} | blk {} | b {} | ritz {} | rNormTol {:.3e} | tol {:.2e} | rNorms = {::.8e}", status.iter,
                   status.num_matvecs, fv(status.optVal), Q.cols() / b, b, enum2sv(ritz), rnormTol(), tol, fv(VectorReal(status.rNorms.topRows(nev))));
}