#include "../env.h"
#include "BondExpansionResult.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/matrix.h"
#include "math/linalg/tensor.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/mps.h"
#include "tools/finite/opt_meta.h"
#include <Eigen/Eigenvalues>

namespace settings {
    static constexpr bool debug_edges     = false;
    static constexpr bool debug_expansion = false;
}

std::tuple<long, double, bool> get_optimal_eigenvalue(long numZeroEigenvalues, Eigen::Vector3d evals, double oldVal, OptRitz ritz) {
    long   optIdx    = 0;
    double optVal    = 0;
    bool   saturated = false;
    auto   eps       = std::max(1.0, evals.cwiseAbs().maxCoeff()) * std::numeric_limits<double>::epsilon();
    switch(ritz) {
        case OptRitz::SR: {
            if(numZeroEigenvalues > 0) { evals = (evals.cwiseAbs().array() < eps).select(evals.maxCoeff() + 1, evals).eval(); }
            [[maybe_unused]] auto tmp = evals.minCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = oldVal <= optVal;
            break;
        }
        case OptRitz::SM: {
            optIdx    = numZeroEigenvalues;
            optVal    = evals.coeff(optIdx);
            saturated = std::abs(oldVal) <= std::abs(optVal);
            break;
        }
        case OptRitz::LR: {
            if(numZeroEigenvalues > 0) evals = (evals.cwiseAbs().array() < eps).select(evals.minCoeff() - 1, evals).eval();
            [[maybe_unused]] auto tmp = evals.maxCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = oldVal >= optVal;
            break;
        }
        case OptRitz::LM: {
            [[maybe_unused]] auto tmp = evals.cwiseAbs().maxCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = std::abs(oldVal) >= std::abs(optVal);
            break;
        }
        default: {
            // Take the closest to the old value
            optVal    = (evals.array() - oldVal).cwiseAbs().minCoeff(&optIdx);
            optVal    = evals.coeff(optIdx);
            saturated = std::abs(oldVal - optVal) < std::numeric_limits<fp64>::epsilon();

            break;
        }
    }
    return {optIdx, optVal, saturated};
}

template<typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, std::vector<long>>
    modified_gram_schmidt(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q) {
    auto t_gramSchmidt = tid::tic_scope("gramschmidt");

    // Orthonormalize with Modified Gram Schmidt
    using MatrixType  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Real        = typename Eigen::NumTraits<Scalar>::Real;
    auto nonOrthoCols = std::vector<long>();
    auto validCols    = std::vector<long>();
    validCols.reserve(Q.cols());
    nonOrthoCols.reserve(Q.cols());
    constexpr auto max_norm_error  = std::numeric_limits<Real>::epsilon() * 100;
    constexpr auto max_ortho_error = std::numeric_limits<Real>::epsilon() * 100;

    for(long i = 0; i < Q.cols(); ++i) {
        auto norm = Q.col(i).norm();
        if(std::abs(norm) < max_norm_error) { continue; }
        Q.col(i) /= norm;
        for(long j = i + 1; j < Q.cols(); ++j) { Q.col(j) -= Q.col(i).dot(Q.col(j)) * Q.col(i); }
    }
    MatrixType Qid = MatrixType::Zero(Q.cols(), Q.cols());
    for(long j = 0; j < Qid.cols(); ++j) {
        for(long i = 0; i <= j; ++i) {
            Qid(i, j) = Q.col(i).dot(Q.col(j));
            Qid(j, i) = Qid(i, j);
        }
        if(j == 0 and std::abs(Qid(j, j) - Real{1}) > max_ortho_error) {
            nonOrthoCols.emplace_back(j);
        } else if(j > 0 and std::abs(Qid(j, j) - Real{1} + Qid.col(j).topRows(j).cwiseAbs().sum()) > max_ortho_error) {
            nonOrthoCols.emplace_back(j);
        }

        if(j == 0 and std::abs(Qid(j, j) - Real{1}) <= max_ortho_error) {
            validCols.emplace_back(j);
        } else if(j > 0 and std::abs(Qid(j, j) - Real{1} + Qid.col(j).topRows(j).cwiseAbs().sum()) <= max_ortho_error) {
            validCols.emplace_back(j);
        }
    }

    if(!Qid(validCols, validCols).isIdentity(max_norm_error)) {
        tools::log->info("Qid \n{}\n", linalg::matrix::to_string(Qid, 8));
        tools::log->info("vc  {}", validCols);
        tools::log->info("noc {}", nonOrthoCols);
        throw except::runtime_error("Q has non orthonormal columns: \n{}\n"
                                    " validCols   : {}\n"
                                    " nonOrthoCols: {}",
                                    linalg::matrix::to_string(Qid, 8), validCols, nonOrthoCols);
    }

    // Q(Eigen::all, nonOrthoCols).setZero();
    Q.colwise().normalize();
    for(long j = 0; j < Q.cols(); ++j) {
        if(!Q.col(j).allFinite()) Q.col(j).setZero();
    }
    return {Q, nonOrthoCols};
}

template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 2>, std::vector<long>> modified_gram_schmidt(const Eigen::Tensor<Scalar, 2> &Q) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Qin = tenx::MatrixMap(Q);
    auto [Qout, nonOrthoCols]                                 = modified_gram_schmidt(Qin);
    return {tenx::TensorMap(Qout), nonOrthoCols};
}

template<typename Scalar>
struct MGSResult {
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    MatrixType   Q;
    VectorReal   Rdiag;
    VectorIdxT   permutation;
    Eigen::Index ncfix = 0;
};

/*! Performs the Modified Gram-Schmidt (MGS) process on a matrix Q --> Q*R,
    while leaving the "ncfix" left-most columns in Q unchanged, and sorting P in order of decreasing |R(i,i)| value.

    The input matrix has the form Q = [M P]. During bond expansion, the columns of
        - M are orthogonal (not orthonormal),
        - P are neither orthogonal nor normalized.
    We want to preserve the columns in M as they are, but sort the
    columns of P in decreasing "|R(i,i)|" values (the "R" from the MGS or a QR decomposition).
    In other words, we want to find the columns in P that would contribute the most to
    the space already spanned by M.

    This is similar in spirit to a column-pivoting QR decomposition, but where we fix "ncfix" left-most columns.
*/
template<typename MatrixT>
MGSResult<typename MatrixT::Scalar> modified_gram_schmidt_Rsort(const MatrixT &A, long ncfix) {
    auto t_gramSchmidt = tid::tic_scope("gramschmidt");

    using IdxT       = Eigen::Index;
    using Scalar     = typename MatrixT::Scalar;
    using MatrixType = typename MGSResult<Scalar>::MatrixType;
    using RealScalar = typename MGSResult<Scalar>::RealScalar;
    using VectorReal = typename MGSResult<Scalar>::VectorReal;

    // STEP 1: Sort the P columns (columns from index ncfix onward) by decreasing norm
    const IdxT ncols       = A.cols();
    auto       permutation = Eigen::Matrix<IdxT, Eigen::Dynamic, 1>(A.cols());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin() + ncfix, permutation.end(), [&A](IdxT i, IdxT j) -> bool { return A.col(i).norm() > A.col(j).norm(); });
    MatrixType Q = A(Eigen::all, permutation);

    // STEP 2: Perform the Modified Gram-Schmidt process.
    // The first ncfix columns (from M) are locked.
    VectorReal Rdiag     = VectorReal::Zero(Q.cols());
    auto       threshold = 1e2 * std::numeric_limits<RealScalar>::epsilon();
    for(IdxT i = 0; i < ncols; ++i) {
        const auto colNorm = Q.col(i).norm();
        Rdiag(i)           = colNorm;

        // Avoid normalizing a near-zero vector.
        if(std::abs(colNorm) < threshold) { continue; }
        Q.col(i) /= colNorm;
        // For subsequent columns (only for columns that belong to P, not the locked columns)
        for(long j = i + 1; j < ncols; ++j) {
            if(j < ncfix) {
                continue; // Skip modification for locked columns from M
            }
            // Subtract projection onto Q.col(i); using complex dot product.
            Q.col(j) -= Q.col(i).dot(Q.col(j)) * Q.col(i);
        }
    }

    // STEP 3: Re-sort the columns corresponding to P by decreasing Rdiag
    auto repermutation = Eigen::Matrix<long, Eigen::Dynamic, 1>(Q.cols());
    std::iota(repermutation.begin(), repermutation.end(), 0);
    std::sort(repermutation.begin() + ncfix, repermutation.end(), [&Rdiag](long i, long j) -> bool { return Rdiag(i) > Rdiag(j); });

    // Apply the second permutation to both Q and Rdiag.
    // Note: The overall permutation becomes the composition of both permutations.
    permutation = permutation(repermutation).eval();
    Q           = Q(Eigen::all, repermutation).eval();
    Rdiag       = Rdiag(repermutation).eval();
    return MGSResult(Q, Rdiag, permutation, ncfix);
}

/*! Performs the Modified Gram-Schmidt (MGS) process on a matrix Q --> Q*R,
    while leaving the "ncfix" left-most columns in Q unchanged, and sorting P
    in order of decreasing |R(i,i)| value, by using column pivoting.

    The input matrix has the form Q = [M P]. During bond expansion, the columns of
        - M are orthogonal (not orthonormal),
        - P are neither orthogonal nor normalized.
    We want to preserve the columns in M as they are, but sort the
    columns of P in decreasing "|R(i,i)|" values (the "R" from the MGS or a QR decomposition).
    In other words, we want to find the columns in P that would contribute the most to
    the space already spanned by M.

    This is similar in spirit to a column-pivoting QR decomposition, but where we fix "ncfix" left-most columns.
*/
template<typename MatrixT>
MGSResult<typename MatrixT::Scalar> modified_gram_schmidt_colpiv(const MatrixT &A, long ncfix) {
    auto t_gramSchmidt = tid::tic_scope("mgs-colpiv");

    using IdxT       = Eigen::Index;
    using Scalar     = typename MatrixT::Scalar;
    using MatrixType = typename MGSResult<Scalar>::MatrixType;
    using RealScalar = typename MGSResult<Scalar>::RealScalar;
    using VectorReal = typename MGSResult<Scalar>::VectorReal;
    using VectorIdxT = typename MGSResult<Scalar>::VectorIdxT;

    // Initialize a permutation vector that keeps track of column pivoting
    const IdxT ncols       = A.cols();
    VectorIdxT permutation = VectorIdxT::LinSpaced(ncols, 0, ncols - 1);
    MatrixType Q           = A;

    // Perform the Modified Gram-Schmidt process with column pivoting
    // The first ncfix columns (from M) are locked.
    VectorReal Rdiag     = VectorReal::Zero(Q.cols());
    auto       threshold = std::numeric_limits<RealScalar>::epsilon() * RealScalar{100};
    for(IdxT i = 0; i < ncols; ++i) {
        // For columns that are not locked (i >= ncfix), perform dynamic pivoting.
        if(i >= ncfix) {
            // Determine the index of the column (from i to end) with the maximal norm.
            IdxT       pivot   = i;
            RealScalar maxNorm = Q.col(i).norm();
            for(IdxT j = i + 1; j < ncols; ++j) {
                RealScalar norm_j = Q.col(j).norm();
                if(norm_j > maxNorm) {
                    maxNorm = norm_j;
                    pivot   = j;
                }
            }
            // Swap the current column with the column having maximum residual norm.
            if(pivot != i) {
                Q.col(i).swap(Q.col(pivot));
                std::swap(permutation(i), permutation(pivot));
            }
        }

        const auto colNorm = Q.col(i).norm();
        Rdiag(i)           = colNorm;

        // Avoid normalizing a near-zero vector.
        if(std::abs(colNorm) < threshold) { continue; }
        Q.col(i) /= colNorm;

        // For subsequent columns (only for columns that belong to P, not the locked columns)
        for(long j = i + 1; j < ncols; ++j) {
            if(j < ncfix) {
                continue; // Skip modification for locked columns from M
            }
            // Subtract projection onto Q.col(i); using complex dot product.
            Q.col(j) -= Q.col(i).dot(Q.col(j)) * Q.col(i);
        }
    }

    return MGSResult(Q, Rdiag, permutation, ncfix);
}

std::pair<fp64, fp64> get_scaling_factors(const Eigen::Tensor<cx64, 3> &M, const Eigen::Tensor<cx64, 3> &P1, const Eigen::Tensor<cx64, 3> &P2,
                                          const std::array<long, 2> &fixedAxes) {
    auto gavg = [](const Eigen::Tensor<cx64, 3> &X, const std::array<long, 2> &axes) -> fp64 {
        if(X.size() == 0) return 0.0;
        Eigen::Tensor<fp64, 0> gavg = X.square().sum(axes).abs().sqrt().log().mean().exp();
        return gavg.coeff(0);
    };
    auto maxnorm = [](const Eigen::Tensor<cx64, 3> &X, const std::array<long, 2> &axes) -> fp64 {
        if(X.size() == 0) return 0.0;
        Eigen::Tensor<fp64, 0> maxn = X.square().sum(axes).abs().sqrt().maximum();
        return maxn.coeff(0);
    };
    auto gavg_M = gavg(M, fixedAxes);
    // auto gavg_P1 = gavg(P1, fixedAxes);
    // auto gavg_P2 = gavg(P2, fixedAxes);
    // auto factor1 = gavg_P1 != 0 ? gavg_M / gavg_P1 : 0.0;
    // auto factor2 = gavg_P2 != 0 ? gavg_M / gavg_P2 : 0.0;
    auto maxn_P1 = maxnorm(P1, fixedAxes);
    auto maxn_P2 = maxnorm(P2, fixedAxes);
    auto factor1 = maxn_P1 != 0 ? gavg_M / maxn_P1 : 0.0;
    auto factor2 = maxn_P2 != 0 ? gavg_M / maxn_P2 : 0.0;
    return {factor1, factor2};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>>
    delinearize_expansion_terms_l2r(const Eigen::Tensor<T, 3> &M, const Eigen::Tensor<T, 3> &N, const Eigen::Tensor<T, 3> &P1, const Eigen::Tensor<T, 3> &P2,
                                    const BondExpansionResult<T> &res, [[maybe_unused]] const svd::config &svd_cfg) {
    /*
        We form M_P = [α₀M | α₁P1 | α₂P2] by concatenating along dimension 2.
        The scaling factors α control the amount of perturbation provided by P1 and P2.
        In matrix-language, the rescaled columns of P1 and P2 are added as new columns to M.

        We then apply a column-pivoting QR factorization of M_P.
        The factorization sorts the columns in M_P such that they contribute decreasingly to the basis formed by previous columns.
        We want to preserve the column order of M, but use this sorting mechanism on the columns from P1 and P2.
        Since M arises from an SVD (typically U * S), its columns are already orthogonal (although not orthonormal due to S).


     */
    assert(M.dimension(2) == N.dimension(1));
    // assert(std::min(M.dimension(0) * M.dimension(1), N.dimension(0) * N.dimension(2)) >= M.dimension(2));
    assert(P1.size() == 0 or P1.dimension(1) == M.dimension(1));
    assert(P1.size() == 0 or P1.dimension(0) == M.dimension(0));
    assert(P2.size() == 0 or P2.dimension(1) == M.dimension(1));
    assert(P2.size() == 0 or P2.dimension(0) == M.dimension(0));
    using R = typename Eigen::NumTraits<T>::Real;
    // using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using VectorR = Eigen::Matrix<R, Eigen::Dynamic, 1>;
    auto offM     = Eigen::DSizes<long, 3>{0, 0, 0};
    auto offP1    = Eigen::DSizes<long, 3>{0, 0, M.dimension(2)};
    auto offP2    = Eigen::DSizes<long, 3>{0, 0, M.dimension(2) + P1.dimension(2)};

    auto M_P = Eigen::Tensor<T, 3>(M.dimension(0), M.dimension(1), M.dimension(2) + P1.dimension(2) + P2.dimension(2));
    M_P.setZero();
    if(M.size() > 0) M_P.slice(offM, M.dimensions()) = M;
    if(P1.size() > 0) M_P.slice(offP1, P1.dimensions()) = P1;
    if(P2.size() > 0) M_P.slice(offP2, P2.dimensions()) = P2;

    auto dim0           = M_P.dimension(0);
    auto dim1           = M_P.dimension(1);
    auto dim2           = M_P.dimension(2);
    long nFixedLeftCols = M.dimension(2);

    VectorR norms_before = tenx::MatrixMap(M_P, dim0 * dim1, dim2).colwise().norm();
    auto    mgsr         = modified_gram_schmidt_colpiv(tenx::MatrixMap(M_P, dim0 * dim1, dim2), nFixedLeftCols);
    auto    max_keep     = std::min(mgsr.permutation.size(), dim0 * dim1);

    // Get the values of Rdiag corresponding to M and P separately
    auto rvals_M = mgsr.Rdiag.topRows(nFixedLeftCols);
    auto rvals_P = mgsr.Rdiag.middleRows(nFixedLeftCols, max_keep - nFixedLeftCols);

    // Calculate scaling factors
    [[maybe_unused]] auto gavg_M = std::exp(rvals_M.array().log().mean()); // Geometric average of R(i,i) values corresponding to M
    auto                  maxr_P = rvals_P.maxCoeff();                     // Max R value

    // Rescale the values in P.
    R factor = std::max(static_cast<R>(res.alpha_h1v), static_cast<R>(res.alpha_h2v)) / maxr_P;
    // fp64 factor = std::max(res.alpha_h1v, res.alpha_h2v)  / maxr_P;
    // Form the new M_P
    M_P       = tenx::TensorCast(tenx::MatrixMap(M_P, dim0 * dim1, dim2)(Eigen::all, mgsr.permutation.topRows(max_keep)), dim0, dim1, max_keep);
    auto M_Ps = M_P.slice(std::array<long, 3>{0, 0, M.dimension(2)}, std::array<long, 3>{dim0, dim1, M_P.dimension(2) - M.dimension(2)});
    M_Ps      = M_Ps.unaryExpr([&factor](const auto v) { return v * factor; });

    // if(max_keep <= 32) {
    //     auto            matrix_dep = tenx::MatrixMap(M_P, M_P.dimension(0) * M_P.dimension(1), M_P.dimension(2));
    //     Eigen::MatrixXd norms_fmt(dim2, 5);
    //     norms_fmt.setZero();
    //     norms_fmt.col(0)                                  = Eigen::VectorXd::LinSpaced(dim2, 0, dim2 - 1);
    //     norms_fmt.col(1)                                  = norms_before;
    //     norms_fmt.col(2).topRows(mgsr.Rdiag.size())       = mgsr.Rdiag;
    //     norms_fmt.col(3).topRows(mgsr.permutation.size()) = mgsr.permutation.cast<double>();
    //     norms_fmt.col(4).topRows(matrix_dep.cols())       = matrix_dep.colwise().norm();
    //
    //     tools::log->info("factor: {:.3e}", factor);
    //     tools::log->info("alpha   : {:.3e}", std::max(res.alpha_h1v, res.alpha_h2v));
    //     tools::log->info("gavg_M  : {:.3e}", gavg_M);
    //     tools::log->info("maxr_P  : {:.3e}", maxr_P);
    //     tools::log->info("norms: \n{}\n", linalg::matrix::to_string(norms_fmt, 8));
    // }

    auto N_0 = Eigen::Tensor<T, 3>(N.dimension(0), M_P.dimension(2), N.dimension(2));
    N_0.setZero();
    auto extN_0                              = std::array<long, 3>{N.dimension(0), std::min(N.dimension(1), M_P.dimension(2)), N.dimension(2)};
    N_0.slice(tenx::array3{0, 0, 0}, extN_0) = N.slice(tenx::array3{0, 0, 0}, extN_0); // Copy N into N_0
    return {M_P, N_0};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> delinearize_expansion_terms_r2l(const Eigen::Tensor<T, 3> &M, const Eigen::Tensor<T, 3> &N,
                                                                                    const Eigen::Tensor<T, 3> &P1, const Eigen::Tensor<T, 3> &P2,
                                                                                    const BondExpansionResult<T> &res, const svd::config &svd_cfg) {
    constexpr auto shf = std::array<long, 3>{0, 2, 1};
    assert(N.dimension(2) == M.dimension(1));
    // assert(std::min(M.dimension(0) * M.dimension(2), N.dimension(0) * N.dimension(1)) >= M.dimension(1));

    auto M_           = Eigen::Tensor<T, 3>(M.shuffle(shf));
    auto N_           = Eigen::Tensor<T, 3>(N.shuffle(shf));
    auto P1_          = Eigen::Tensor<T, 3>(P1.shuffle(shf));
    auto P2_          = Eigen::Tensor<T, 3>(P2.shuffle(shf));
    auto [M_P_, N_0_] = delinearize_expansion_terms_l2r(M_, N_, P1_, P2_, res, svd_cfg);
    return {M_P_.shuffle(shf), N_0_.shuffle(shf)};
}

template<typename T, typename Scalar>
void tools::finite::env::internal::get_optimally_mixed_block(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                             const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta,
                                                             BondExpansionResult<Scalar> &res) {
    if constexpr(sfinae::is_std_complex_v<T>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block<fp64>(sites, state, model, edges, opt_meta, res);
        }
    }

    auto t_mixblk = tid::tic_scope("mixblk");
    auto K1_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG);
    auto K2_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG);

    MatVecMPOS<T> H1 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    MatVecMPOS<T> H2 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    using R          = typename Eigen::NumTraits<T>::Real;
    using MatrixT    = typename MatVecMPOS<T>::MatrixType;
    // using MatrixR     = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorR     = Eigen::Matrix<R, Eigen::Dynamic, 1>;
    auto nonOrthoCols = std::vector<long>();

    auto mps_size  = H1.get_size();
    auto mps_shape = H1.get_shape_mps();
    long ncv       = std::clamp(opt_meta.bondexp_nkrylov, 3ul, 256ul);

    auto H1V = MatrixT();
    auto H2V = MatrixT();
    if(K1_on) H1V.resize(mps_size, ncv);
    if(K2_on) H2V.resize(mps_size, ncv);

    // Default solution
    res.mixed_blk = state.template get_multisite_mps<Scalar>(sites);
    res.alpha_mps = 1.0;
    res.alpha_h1v = 0.0;
    res.alpha_h2v = 0.0;

    // Initialize Krylov vector 0
    auto V   = MatrixT(mps_size, ncv);
    V.col(0) = tenx::asScalarType<T>(tenx::VectorCast(res.mixed_blk));

    R                  optVal = std::numeric_limits<R>::quiet_NaN();
    long               optIdx = 0;
    R                  tol    = static_cast<R>(opt_meta.eigs_tol.value_or(settings::precision::eigs_tol_max));
    R                  absTol = std::numeric_limits<R>::epsilon() * 100;
    R                  relTol = R{1e-4f};
    R                  rnorm  = R{1};
    [[maybe_unused]] R snorm  = R{1}; // Estimate the matrix norm from the largest singular value/eigenvalue. Converged if  rnorm  < snorm * tol
    size_t             iter   = 0;
    size_t             ngs    = 0;
    std::string        msg;
    while(true) {
        // Define the krylov subspace
        for(long i = 0; i + 1 < ncv; ++i) {
            if(i < ncv / 2) {
                H1.MultAx(V.col(i).data(), V.col(i + 1).data());
            } else if(i == ncv / 2) {
                H2.MultAx(V.col(0).data(), V.col(i + 1).data());
            } else {
                H2.MultAx(V.col(i).data(), V.col(i + 1).data());
            }
        }

        // Orthonormalize with Modified Gram Schmidt
        for(size_t igs = 0; igs <= 5; ++igs) {
            std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
            ngs++;
            if(nonOrthoCols.empty()) break;
        }

        // V should now have orthonormal vectors
        if(K1_on) {
            for(long i = 0; i < ncv; ++i) H1.MultAx(V.col(i).data(), H1V.col(i).data());
        }
        if(K2_on) {
            for(long i = 0; i < ncv; ++i) H2.MultAx(V.col(i).data(), H2V.col(i).data());
        }
        if(!std::isnan(optVal)) {
            if(opt_meta.optAlgo == OptAlgo::DMRG)
                rnorm = (H1V.col(0) - optVal * V.col(0)).cwiseAbs().maxCoeff();
            else if(opt_meta.optAlgo == OptAlgo::GDMRG)
                rnorm = (H1V.col(0) - optVal * H2V.col(0)).cwiseAbs().maxCoeff();
            else
                rnorm = (H2V.col(0) - optVal * V.col(0)).cwiseAbs().maxCoeff();
        }

        if(iter >= 1ul and rnorm < tol /* * snorm */) {
            msg = fmt::format("converged rnorm {:.3e} < tol {:.3e}", fp(rnorm), fp(tol));
            break;
        }
        auto t_dotprod = tid::tic_scope("dotprod");

        MatrixT K1 = MatrixT::Zero(ncv, ncv);
        if(K1_on) {
            for(long j = 0; j < ncv; ++j) {
                for(long i = j; i < ncv; ++i) { K1(i, j) = V.col(i).dot(H1V.col(j)); }
            }
            K1 = K1.template selfadjointView<Eigen::Lower>();
        }

        MatrixT K2 = MatrixT::Zero(ncv, ncv);
        if(K2_on) {
            // Use abs to avoid negative near-zero values
            for(long j = 0; j < ncv; ++j) {
                for(long i = j; i < ncv; ++i) {
                    if(i == j)
                        K2(i, j) = std::abs(V.col(i).dot(H2V.col(j)));
                    else
                        K2(i, j) = V.col(i).dot(H2V.col(j));
                }
            }
            K2 = K2.template selfadjointView<Eigen::Lower>();
        }

        t_dotprod.toc();
        auto t_eigsol = tid::tic_scope("eigsol");
        // auto    eps           = std::numeric_limits<R>::epsilon();
        long    numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() < std::numeric_limits<double>::epsilon()).count();
        long    numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() < std::numeric_limits<double>::epsilon()).count();
        long    numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
        VectorR evals; // Eigen::VectorXd ::Zero();
        MatrixT evecs; // Eigen::MatrixXcd::Zero();
        OptRitz ritz_internal = opt_meta.optRitz;
        switch(opt_meta.optAlgo) {
            using enum OptAlgo;
            case DMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K1, Eigen::ComputeEigenvectors);
                if(solver.info() == Eigen::ComputationInfo::Success) {
                    evals = solver.eigenvalues();
                    evecs = solver.eigenvectors();
                } else {
                    tools::log->info("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
                    tools::log->warn("Diagonalization of K1 exited with info {}", static_cast<int>(solver.info()));
                }

                if(evals.hasNaN()) throw except::runtime_error("found nan eigenvalues");
                break;
            }
            case DMRGX: [[fallthrough]];
            case HYBRID_DMRGX: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case XDMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case GDMRG: {
                if(nonOrthoCols.empty() and numZeroRows == 0) {
                    auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixT>(
                        K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                    evals = solver.eigenvalues().real();
                    evecs = solver.eigenvectors().colwise().normalized();
                } else {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                    evals       = solver.eigenvalues();
                    evecs       = solver.eigenvectors();
                    if(opt_meta.optRitz == OptRitz::LM) ritz_internal = OptRitz::SM;
                    if(opt_meta.optRitz == OptRitz::LR) ritz_internal = OptRitz::SM;
                    if(opt_meta.optRitz == OptRitz::SM) ritz_internal = OptRitz::LM;
                    if(opt_meta.optRitz == OptRitz::SR) ritz_internal = OptRitz::LR;
                }

                break;
            }
        }
        auto t_checks      = tid::tic_scope("checks");
        snorm              = static_cast<R>(evals.cwiseAbs().maxCoeff());
        V                  = (V * evecs.real()).eval(); // Now V has ncv columns mixed according to evecs
        VectorR mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
        auto    mixedColOk = std::vector<long>();       // New states with acceptable norm and eigenvalue
        mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
        for(long i = 0; i < mixedNorms.size(); ++i) {
            if(std::abs(mixedNorms(i) - R{1}) > static_cast<R>(settings::precision::max_norm_error)) continue;
            // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
            // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues of
            // GDMRG are not
            mixedColOk.emplace_back(i);
        }
        if constexpr(!tenx::sfinae::is_quadruple_prec_v<T>) {
            if(mixedColOk.size() <= 1) {
                tools::log->debug("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
                tools::log->debug("K2                     : \n{}\n", linalg::matrix::to_string(K2, 8));
                tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
                // tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(evecs, 8));
                // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
                tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
                tools::log->debug("mixedColOk             = {}", mixedColOk);
                tools::log->debug("numZeroRowsK1          = {}", numZeroRowsK1);
                tools::log->debug("numZeroRowsK2          = {}", numZeroRowsK2);
                tools::log->debug("nonOrthoCols           = {}", nonOrthoCols);
                tools::log->debug("ngramSchmidt           = {}", ngs);
                if(opt_meta.optAlgo == OptAlgo::GDMRG) {
                    H2.MultAx(V.col(0).data(), H2V.col(0).data());
                    H2.MultAx(V.col(1).data(), H2V.col(1).data());
                    H2.MultAx(V.col(2).data(), H2V.col(2).data());
                    tools::log->debug("V.col(0).dot(H2*V.col(1)) = {:.16f}", V.col(0).dot(H2V.col(1)));
                    tools::log->debug("V.col(0).dot(H2*V.col(2)) = {:.16f}", V.col(0).dot(H2V.col(2)));
                    tools::log->debug("V.col(1).dot(H2*V.col(2)) = {:.16f}", V.col(1).dot(H2V.col(2)));
                } else {
                    tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
                    tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
                    tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
                }
            }
        }
        if(mixedColOk.empty()) {
            msg = fmt::format("mixedColOk is empty");
            break;
        }
        // tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
        // Eigenvalues are sorted in ascending order.
        long colIdx = 0;
        switch(ritz_internal) {
            case OptRitz::SR: {
                evals(mixedColOk).minCoeff(&colIdx);
                break;
            }
            case OptRitz::LR: {
                evals(mixedColOk).maxCoeff(&colIdx);
                break;
            }
            case OptRitz::SM: {
                evals(mixedColOk).cwiseAbs().minCoeff(&colIdx);
                break;
            }
            case OptRitz::LM: {
                evals(mixedColOk).cwiseAbs().maxCoeff(&colIdx);
                break;
            }
            case OptRitz::IS: [[fallthrough]];
            case OptRitz::TE: [[fallthrough]];
            case OptRitz::NONE: {
                (evals(mixedColOk).array() - static_cast<R>(res.ene_old)).cwiseAbs().minCoeff(&colIdx);
            }
        }
        optIdx = mixedColOk[colIdx];

        auto oldVal = optVal;
        optVal      = evals(optIdx);
        auto relval = std::abs((oldVal - optVal) / (R{0.5} * (optVal + oldVal)));

        // Check convergence
        if(std::abs(oldVal - optVal) < absTol) {
            msg = fmt::format("saturated: abs change {:.3e} < 1e-14", fp(std::abs(oldVal - optVal)));
            break;
        }
        if(relval < relTol) {
            msg = fmt::format("saturated: rel change ({:.3e}) < 1e-4", fp(relval));
            break;
        }

        // If we make it here: update the solution
        res.mixed_blk = tenx::asScalarType<Scalar>(tenx::TensorCast(V.col(optIdx), mps_shape));
        VectorR col   = evecs.col(optIdx).real();
        res.alpha_mps = static_cast<double>(col.coeff(0));
        res.alpha_h1v = static_cast<double>(col.coeff(1));
        res.alpha_h2v = static_cast<double>(col.coeff(ncv / 2 + 1));

        if(mixedColOk.size() == 1) {
            msg = fmt::format("saturated: only one valid eigenvector");
            break;
        }

        if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others)
        if(iter + 1 < opt_meta.bondexp_maxiter)
            tools::log->debug(
                "bond expansion result: {:.16f} [{}] | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters {} | "
                "{:.3e} it/s |  {:.3e} s",
                fp(optVal), optIdx, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, mps_size, fp(V.col(0).norm()), fp(rnorm), ngs, iter,
                iter / t_mixblk->get_last_interval(), t_mixblk->get_last_interval());

        iter++;
        if(iter >= std::max(1ul, opt_meta.bondexp_maxiter)) {
            msg = fmt::format("iter ({}) >= maxiter ({})", iter, opt_meta.bondexp_maxiter);
            break;
        }
    }

    tools::log->debug("mixed state result: {:.16f} [{}] | ncv {} | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters "
                      "{} | {:.3e} s | {}",
                      fp(optVal), optIdx, ncv, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, mps_size, fp(V.col(0).norm()), fp(rnorm), ngs, iter,
                      t_mixblk->get_last_interval(), msg);
}
/* clang-format off */
template void tools::finite::env::internal::get_optimally_mixed_block<fp32>(const std::vector<size_t> &sites, const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges, const OptMeta &opt_meta, BondExpansionResult<fp32> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<fp64>(const std::vector<size_t> &sites, const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges, const OptMeta &opt_meta, BondExpansionResult<fp64> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<fp128>(const std::vector<size_t> &sites, const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges, const OptMeta &opt_meta, BondExpansionResult<fp128> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<cx32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<fp32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<fp64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::get_optimally_mixed_block<fp128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
/* clang-format on */

template<typename T, typename Scalar>
void tools::finite::env::internal::set_mixing_factors_to_rnorm(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                               const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta,
                                                               BondExpansionResult<Scalar> &res) {
    assert(opt_meta.bondexp_minalpha <= opt_meta.bondexp_maxalpha);
    assert(opt_meta.bondexp_minalpha >= 0);
    assert(opt_meta.bondexp_maxalpha > 0);
    // using R = typename Eigen::NumTraits<Scalar>::Real;
    // using MatrixT      = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    // using MatrixR      = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorR      = Eigen::Matrix<R, Eigen::Dynamic, 1>;
    auto multisite_mps = state.template get_multisite_mps<Scalar>(sites);
    using R            = typename Eigen::NumTraits<T>::Real;
    auto V             = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(multisite_mps.size(), 3);
    V.setZero();
    V.col(0)      = tenx::asScalarType<T>(tenx::VectorCast(multisite_mps));
    res.alpha_mps = 1;
    res.alpha_h1v = 0;
    res.alpha_h2v = 0;

    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::H1)) {
        auto rnorm_h1 = tools::finite::measure::residual_norm_H1<Scalar>(sites, state, model, edges);
        res.alpha_h1v = std::clamp(static_cast<R>(rnorm_h1), static_cast<R>(opt_meta.bondexp_minalpha), static_cast<R>(opt_meta.bondexp_maxalpha));
    }
    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::H2)) {
        auto rnorm_h2 = tools::finite::measure::residual_norm_H2<Scalar>(sites, state, model, edges);
        res.alpha_h2v = std::clamp(static_cast<R>(rnorm_h2), static_cast<R>(opt_meta.bondexp_minalpha), static_cast<R>(opt_meta.bondexp_maxalpha));
    }

    res.alpha_mps = std::sqrt(1.0 - std::pow(res.alpha_h1v, 2.0) + std::pow(res.alpha_h2v, 2.0));

    Eigen::Matrix<T, 3, 1> col;
    col(0) = static_cast<R>(res.alpha_mps);
    col(1) = static_cast<R>(res.alpha_h1v);
    col(2) = static_cast<R>(res.alpha_h2v);

    // Define new vectors based on the eigenvalues
    V.col(0)      = (V * col).eval();
    res.mixed_blk = tenx::asScalarType<Scalar>(tenx::TensorCast(V.col(0), multisite_mps.dimensions()));
}

/* clang-format off */
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp32>(const std::vector<size_t> &sites, const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges, const OptMeta &opt_meta, BondExpansionResult<fp32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp64>(const std::vector<size_t> &sites, const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges, const OptMeta &opt_meta, BondExpansionResult<fp64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp128>(const std::vector<size_t> &sites, const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges, const OptMeta &opt_meta, BondExpansionResult<fp128> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<cx32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_rnorm<fp128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
/* clang-format on */

template<typename T, typename Scalar>
void tools::finite::env::internal::set_mixing_factors_to_stdv_H(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                                const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const OptMeta &opt_meta,
                                                                BondExpansionResult<Scalar> &res) {
    auto multisite_mps = state.template get_multisite_mps<Scalar>(sites);
    using R            = typename Eigen::NumTraits<T>::Real;
    assert(opt_meta.bondexp_minalpha <= opt_meta.bondexp_maxalpha);
    assert(opt_meta.bondexp_minalpha >= 0);
    assert(opt_meta.bondexp_maxalpha > 0);
    res.alpha_mps = 1;
    res.alpha_h1v = 0;
    res.alpha_h2v = 0;
    auto var      = tools::finite::measure::energy_variance(state, model, edges);
    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::H1)) {
        res.alpha_h1v = std::clamp(static_cast<R>(std::sqrt(var)), static_cast<R>(opt_meta.bondexp_minalpha), static_cast<R>(opt_meta.bondexp_maxalpha));
    }
    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::H2)) {
        res.alpha_h2v = std::clamp(static_cast<R>(std::sqrt(var)), static_cast<R>(opt_meta.bondexp_minalpha), static_cast<R>(opt_meta.bondexp_maxalpha));
    }
}

/* clang-format off */
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp32>(const std::vector<size_t> &sites, const StateFinite<fp32> &state, const ModelFinite<fp32> &model, const EdgesFinite<fp32> &edges, const OptMeta &opt_meta, BondExpansionResult<fp32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp64>(const std::vector<size_t> &sites, const StateFinite<fp64> &state, const ModelFinite<fp64> &model, const EdgesFinite<fp64> &edges, const OptMeta &opt_meta, BondExpansionResult<fp64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp128>(const std::vector<size_t> &sites, const StateFinite<fp128> &state, const ModelFinite<fp128> &model, const EdgesFinite<fp128> &edges, const OptMeta &opt_meta, BondExpansionResult<fp128> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<cx32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<cx64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<cx128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp32>(const std::vector<size_t> &sites, const StateFinite<cx32> &state, const ModelFinite<cx32> &model, const EdgesFinite<cx32> &edges, const OptMeta &opt_meta, BondExpansionResult<cx32> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp64>(const std::vector<size_t> &sites, const StateFinite<cx64> &state, const ModelFinite<cx64> &model, const EdgesFinite<cx64> &edges, const OptMeta &opt_meta, BondExpansionResult<cx64> &res);
template void tools::finite::env::internal::set_mixing_factors_to_stdv_H<fp128>(const std::vector<size_t> &sites, const StateFinite<cx128> &state, const ModelFinite<cx128> &model, const EdgesFinite<cx128> &edges, const OptMeta &opt_meta, BondExpansionResult<cx128> &res);
/* clang-format on */

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::get_mixing_factors_postopt_rnorm(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                                                 const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                                                 const OptMeta &opt_meta) {
    using R = typename Eigen::NumTraits<Scalar>::Real;
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
    auto res           = BondExpansionResult<Scalar>();
    res.direction      = state.get_direction();
    res.sites          = sites;
    res.dims_old       = state.get_mps_dims(sites);
    res.bond_old       = state.get_bond_dims(sites);
    res.posL           = safe_cast<long>(sites.front());
    res.posR           = safe_cast<long>(sites.back());
    const auto &mpsL   = state.get_mps_site(res.posL);
    const auto &mpsR   = state.get_mps_site(res.posR);
    res.dimL_old       = mpsL.dimensions();
    res.dimR_old       = mpsR.dimensions();
    res.bondexp_factor = static_cast<R>(opt_meta.bondexp_factor);

    res.ene_old = tools::finite::measure::energy(state, model, edges);
    res.var_old = tools::finite::measure::energy_variance(state, model, edges);
    /*! In single-site bond expansion we should only use single-site techniques.
        The POSTOPT_1SITE is identical to the DMRG3S method up to the choice of mixing factor:
            - the expansion occurs after the optimization step, before moving
            - the current site is enriched as  Ψ' =  (P⁰ + P¹ + P²)Ψ , where:
                  - P⁰ = α₀ = sqrt(1-α₁²-α₂²),
                  - P¹ = α₁H¹Ψ, α₁ = |(H¹ - <H¹>)Ψ| (residual norm wrt H¹)
                  - P² = α₂H²Ψ, α₂ = |(H² - <H²>)Ψ| (residual norm wrt H²)
                  - H¹,H²,Ψ denote the local "effective" parts of the MPS/MPO corresponding to the current site.
            - the next site ahead is zero-padded to match the new dimensions.
            - this works well because the residuals vanish as we approach an eigenstate.
    */
    auto pos            = state.template get_position<size_t>();
    auto bonds          = state.get_bond_dims(sites);
    auto minbond        = *std::min_element(bonds.begin(), bonds.end());
    auto sites_oneortwo = minbond < 4 ? sites : std::vector{pos}; // We need the bond dimension to be larger than the local 1site dimension
    switch(opt_meta.optType) {
        case OptType::FP64: {
            internal::set_mixing_factors_to_rnorm<fp64>(sites_oneortwo, state, model, edges, opt_meta, res);
            break;
        }
        case OptType::CX64: {
            internal::set_mixing_factors_to_rnorm<cx64>(sites_oneortwo, state, model, edges, opt_meta, res);
            break;
        }
        default: throw except::runtime_error("get_mixing_factors_postopt_rnorm: not implemented for type {}", enum2sv(opt_meta.optType));
    }
    return res;
}
template BondExpansionResult<fp64> tools::finite::env::get_mixing_factors_postopt_rnorm(const std::vector<size_t> &, const StateFinite<fp64> &,
                                                                                        const ModelFinite<fp64> &, const EdgesFinite<fp64> &, const OptMeta &);
template BondExpansionResult<cx64> tools::finite::env::get_mixing_factors_postopt_rnorm(const std::vector<size_t> &, const StateFinite<cx64> &,
                                                                                        const ModelFinite<cx64> &, const EdgesFinite<cx64> &, const OptMeta &);

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::get_mixing_factors_preopt_krylov(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                                                 const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges,
                                                                                 const OptMeta &opt_meta) {
    using R = typename Eigen::NumTraits<Scalar>::Real;
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
    auto res           = BondExpansionResult<Scalar>();
    res.sites          = sites;
    res.dims_old       = state.get_mps_dims(sites);
    res.bond_old       = state.get_bond_dims(sites);
    res.posL           = safe_cast<long>(sites.front());
    res.posR           = safe_cast<long>(sites.back());
    const auto &mpsL   = state.get_mps_site(res.posL);
    const auto &mpsR   = state.get_mps_site(res.posR);
    res.dimL_old       = mpsL.dimensions();
    res.dimR_old       = mpsR.dimensions();
    res.bondexp_factor = static_cast<R>(opt_meta.bondexp_factor);

    res.ene_old = tools::finite::measure::energy(state, model, edges);
    res.var_old = tools::finite::measure::energy_variance(state, model, edges);
    /*! For PREOPT_NSITE_REAR and PREOPT_NSITE_FORE:
        - the expansion occurs just before the main DMRG optimization step.
        - the expansion involves [active sites] plus sites behind or ahead.
        - at least two sites are used, the upper limit depends on dmrg_blocksize
        - on these sites, we find α that minimizes f(Ψ'), where Ψ' = (α₀ + α₁ H¹ + α₂ H²)Ψ, and f is
          the relevant objective function (energy, variance or <H>/<H²>).
        - note that no zero-padding is used here.
       In multisie DMRG we can afford to estimate mixing factors using multiple sites.
        This technique should use at least two sites. Note that it is possible to use this technique
        with a single site in principle, but it underestimates the mixing factors by several orders of magnitude,
        leading to poor convergence.
     */
    // auto opt_meta2 = opt_meta;
    // opt_meta2.optRitz = OptRitz::SM;
    // opt_meta2.optAlgo = OptAlgo::XDMRG;
    switch(opt_meta.optType) {
        case OptType::FP64: {
            internal::get_optimally_mixed_block<fp64>(sites, state, model, edges, opt_meta, res);
            break;
        }
        case OptType::CX64: {
            internal::get_optimally_mixed_block<cx64>(sites, state, model, edges, opt_meta, res);
            break;
        }
        default: throw except::runtime_error("get_mixing_factors_preopt_krylov: not implemented for type {}", enum2sv(opt_meta.optType));
    }
    return res;
}
template BondExpansionResult<fp64> tools::finite::env::get_mixing_factors_preopt_krylov(const std::vector<size_t> &, const StateFinite<fp64> &,
                                                                                        const ModelFinite<fp64> &, const EdgesFinite<fp64> &, const OptMeta &);
template BondExpansionResult<cx64> tools::finite::env::get_mixing_factors_preopt_krylov(const std::vector<size_t> &, const StateFinite<cx64> &,
                                                                                        const ModelFinite<cx64> &, const EdgesFinite<cx64> &, const OptMeta &);

template<typename Scalar>
void merge_expansion_terms_r2l(const StateFinite<Scalar> &state, MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &ML_PL, MpsSite<Scalar> &mpsR,
                               const Eigen::Tensor<Scalar, 3> &MR_PR, const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR
    // We zero-pad mpsL and enrich mpsR.
    // During forward expansion -->
    //      * mpsL is AC(i), or B(i) during multisite dmrg
    //      * mpsR is B(i+1)
    // During backward expansion <--
    //      * mpsL is A(i-1) always
    //      * mpsR is AC(i) always
    // Thus, the possible situations are [AC,B] or [B,B] or [A,AC]
    // After USV = SVD(MR_PR):
    //      If [AC,B]:
    //           * ML_PL is [AC(i), PL]            <--- Note that we use full AC(i)! Not bare A(i)
    //           * MR_PR is [B(i+1), PR]^T
    //           * mpsL:  AC(i)   = ML_PL * U     <---- lost left normalization, but it is not needed during optimization
    //           * mpsL:  C(i)   = S
    //           * mpsR:  B(i+1) = V
    //      If [B,B]:
    //           * ML_PL is [B(i), PL]
    //           * MR_PR is [B(i+1), PR]^T
    //           * mpsL:  B(i)   = ML_PL * U * S  <-- lost right normalization, but it is not needed during optimization
    //           * mpsL:  Λ(i)   = S
    //           * mpsR:  B(i+1) = V
    //      If [A,AC]:
    //           * ML_PL is [A(i-1), PL]             <--- (usually PL = P0)
    //           * MR_PR is [AC(i), PR]^T  -> USV    <--- Note that we use AC(i)! Not bare A(i)
    //           * mpsL:  A(i-1) = ML_PL * U         <--- lost left normalization, but it will not be needed during the optimization later
    //           * mpsR:  Λ(i) = S, (will become a center later)
    //           * mpsR:  A(i) = S * V (note that we replace bare A(i), not AC(i)!)
    //           * The old C(i) living in mpsR is outdated, but it incorporated into AC(i) when building MR_PR.
    //           * We can simply replace C(i) with an identity matrix. It will not be needed after we transform AC(i) into a B(i) later.
    //

    tools::log->trace("merge_expansion_term_PR: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    svd::solver svd;
    auto        posL = mpsL.get_position();
    auto        labL = mpsL.get_label();
    auto        labR = mpsR.get_label();
    auto [U, S, V]   = svd.schmidt_into_right_normalized(MR_PR, mpsR.spin_dim(), svd_cfg);
    if(labL == "AC" and labR == "B") {
        mpsR.set_M(V);
        mpsR.stash_U(U, posL);
        mpsR.stash_C(S, -1.0, posL); // Set a negative truncation error to ignore it.
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR); // normalization of mpsL is lost here.
    } else if(labL == "B" and labR == "B") {
        auto US = tenx::asDiagonalContract(S, U, 2); // Eigen::Tensor<cx64, 3>(U.contract(tenx::asDiagonal(S), tenx::idx({2}, {0})));
        mpsR.set_M(V);
        mpsR.stash_U(US, posL);
        mpsR.stash_S(S, -1.0, posL); // Set a negative truncation error to ignore it.
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR); // normalization of mpsL is lost here.
    } else if(labL == "A" and labR == "AC") {
        Eigen::Tensor<Scalar, 3> SV_LCinv = tenx::asDiagonalContract(S, tenx::asDiagonalInverseContract(mpsR.get_LC(), V, 2), 1);
        // .contract(tenx::asDiagonalInversed(mpsR.get_LC()), tenx::idx({2}, {0}))
        // .shuffle(tenx::array3{1, 0, 2});
        mpsR.set_M(SV_LCinv);
        mpsR.set_L(S, -1.0);
        mpsR.stash_U(U, posL);
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR);

        // Eigen::Tensor<cx64, 3> SV = tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2});
        // Eigen::Tensor<cx64, 1> Id(V.dimension(2));
        // Id.setConstant(cx64(1.0, 0.0));
        // mpsR.set_M(SV);
        // mpsR.set_L(S, -1.0);
        // mpsR.set_LC(Id, -1.0);
        // mpsR.stash_U(U, posL);
        // mpsL.set_M(ML_PL);
        // mpsL.take_stash(mpsR);

    } else {
        throw except::runtime_error("merge_expansion_term_PR: could not match case: [{},{}]", labL, labR);
    }
    state.clear_cache();
    state.clear_measurements();
    {
        // Make mpsL normalized so that later checks can succeed
        auto                     multisite_mpsL = state.template get_multisite_mps<Scalar>({posL});
        auto                     norm_old       = tools::common::contraction::contract_mps_norm(multisite_mpsL);
        Eigen::Tensor<Scalar, 3> M_tmp          = mpsL.get_M_bare() * mpsL.get_M_bare().constant(std::pow(norm_old, Real{-0.5})); // Rescale
        mpsL.set_M(M_tmp);
        state.clear_cache();
        state.clear_measurements();
        // if constexpr(settings::debug_expansion) {
        auto mpsL_final = state.template get_multisite_mps<Scalar>({posL});
        auto norm_new   = tools::common::contraction::contract_mps_norm(mpsL_final);
        tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsL.get_label(), mpsL.get_position(), fp(std::abs(norm_old)),
                          fp(std::abs(norm_new)));
        // }
    }
}

template<typename Scalar>
void merge_expansion_terms_l2r(const StateFinite<Scalar> &state, MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &ML_PL, MpsSite<Scalar> &mpsR,
                               const Eigen::Tensor<Scalar, 3> &MR_PR, const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR.
    // During forward expansion <--
    //      * mpsL is A(i-1) always
    //      * mpsR is AC(i), or A(i) during multisite dmrg
    // During backward expansion -->
    //      * mpsL is AC(i) always
    //      * mpsR is B(i+1) always
    // Thus, the possible situations are  [A, AC], [A,A] or [AC,B]
    // After USV = SVD(ML_PL):
    //      If [A, AC]:
    //           * ML_PL is [A(i-1), PL]
    //           * MR_PR is [AC(i),  0]^T                        <--- Note that we use bare A(i)! Not AC(i)
    //           * mpsL:  A(i-1) = U
    //           * mpsR:  Λ(i)   = S                             <--- takes stash S
    //           * mpsR:  A      = S * V * MR_PR * C(i)^-1       <--- takes stash S,V and loses left-right normalization
    //           * mpsR:  C(i)                                   <--- does not change
    //      If [A,A]:
    //           * ML_PL is [A(i-1), PL]^T
    //           * MR_PR is [AC(i), 0]^T             <--- Note that we use bare A(i)! Not AC(i)
    //           * mpsL:  A(i-1) = U
    //           * mpsR:  Λ(i)   = S (takes stash S)
    //           * mpsR:  A(i) = S * V * MR_PR (takes stash S,SV and loses left normalization)
    //      If [AC,B]:
    //           * ML_PL is [AC(i), PL]^T
    //           * MR_PR is [B(i+1), 0]^T
    //           * mpsL:  A(i) = U
    //           * mpsL:  C(i) = S
    //           * mpsR:  B(i+1) = V * MR_PR (loses right normalization, but that is not be needed during the next optimization)
    //

    tools::log->trace("merge_expansion_term_PL: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());
    svd::solver svd;
    using Real     = typename Eigen::NumTraits<Scalar>::Real;
    auto posR      = mpsR.get_position();
    auto labL      = mpsL.get_label();
    auto labR      = mpsR.get_label();
    auto [U, S, V] = svd.schmidt_into_left_normalized(ML_PL, mpsL.spin_dim(), svd_cfg);
    if(labL == "A" and labR == "AC") {
        // Eigen::Tensor<cx64, 3> SV          = tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2});
        // Eigen::Tensor<cx64, 3> MR_PR_LCinv = MR_PR.contract(tenx::asDiagonalInversed(mpsR.get_LC()), tenx::idx({2}, {0}));
        auto SV          = tenx::asDiagonalContract(S, V, 1);
        auto MR_PR_LCinv = tenx::asDiagonalInverseContract(mpsR.get_LC(), MR_PR, 2);
        mpsL.set_M(U);
        mpsL.stash_S(S, -1.0, posR); // Set a negative truncation error to ignore it.
        mpsL.stash_V(SV, posR);
        mpsR.set_M(MR_PR_LCinv);
        mpsR.take_stash(mpsL); // normalization of mpsR is lost here

    } else if(labL == "A" and labR == "A") {
        // auto SV = Eigen::Tensor<cx64, 3>(tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2}));
        auto SV = tenx::asDiagonalContract(S, V, 1);
        mpsL.set_M(U);
        mpsL.stash_S(S, -1.0, posR); // Set a negative truncation error to ignore it.
        mpsL.stash_V(SV, posR);
        mpsR.set_M(MR_PR);
        mpsR.take_stash(mpsL); // normalization of mpsR is lost here
    } else if(labL == "AC" and labR == "B") {
        mpsL.set_M(U);
        mpsL.set_LC(S);
        mpsL.stash_V(V, posR);
        mpsR.set_M(MR_PR);
        mpsR.take_stash(mpsL);
    } else {
        throw except::runtime_error("merge_expansion_term_PL: could not match case: [{},{}]", labL, labR);
    }
    state.clear_cache();
    state.clear_measurements();
    {
        // Make mpsR normalized so that later checks can succeed
        auto                     multisite_mpsR = state.template get_multisite_mps<Scalar>({posR});
        auto                     norm_old       = tools::common::contraction::contract_mps_norm(multisite_mpsR);
        Eigen::Tensor<Scalar, 3> M_tmp          = mpsR.get_M_bare() * mpsR.get_M_bare().constant(std::pow(norm_old, -Real(0.5))); // Rescale by the norm
        mpsR.set_M(M_tmp);
        state.clear_cache();
        state.clear_measurements();
        // if constexpr(settings::debug_expansion) {
        auto mpsR_final = state.template get_multisite_mps<Scalar>({mpsR.get_position()});
        auto norm_new   = tools::common::contraction::contract_mps_norm(mpsR_final);
        tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsR.get_label(), mpsR.get_position(), fp(std::abs(norm_old)),
                          fp(std::abs(norm_new)));
        // }
    }
}
template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::expand_bond_postopt_1site(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                          EdgesFinite<Scalar> &edges, const OptMeta &opt_meta) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_postopt_1site: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_bond_postopt_1site: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for bond expansion");

    if(!has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::POSTOPT_1SITE))
        throw except::logic_error("expand_bond_postopt_1site: opt_meta.bondexp_policy must have BondExpansionPolicy::POSTOPT set");
    if(opt_meta.optExit == OptExit::NONE) throw except::logic_error("expand_bond_postopt_1site: requires opt_meta.optExit != OptExit::NONE");

    // POSTOPT enriches the current site and zero-pads the upcoming site: Therefore it can only do REAR
    // This method adds noise to the bond when expanding. Therefore, we rarely benefit from this method once
    // the bond dimension has already grown to its theoretical maximum: If bonds are numbered l=0,1,2...L
    // the maximum bond is d^min(l,L-l), where d is the spin dimension

    // Case list
    // (a)     [ML, P] [MR 0]^T : postopt_rear (AC,B) -->
    // (b)     [ML, 0] [MR P]^T : postopt_rear (A,AC) <--

    std::vector<size_t> pos_expanded;
    auto                pos = state.template get_position<size_t>();
    if(state.get_direction() > 0 and pos == std::clamp<size_t>(pos, 0, state.template get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};
    if(state.get_direction() < 0 and pos == std::clamp<size_t>(pos, 1, state.template get_length<size_t>() - 1)) pos_expanded = {pos - 1, pos};

    if(pos_expanded.empty()) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("No positions to expand: active sites {} | mode {}", state.active_sites, flag2str(opt_meta.bondexp_policy));
        return res; // No update
    }

    size_t posL = pos_expanded.front();
    size_t posR = pos_expanded.back();
    auto  &mpsL = state.get_mps_site(posL);
    auto  &mpsR = state.get_mps_site(posR);
    if(state.num_bonds_at_maximum(pos_expanded) == 1) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("The bond upper limit has been reached for site pair [{}-{}] | mode {}", mpsL.get_tag(), mpsR.get_tag(),
                               flag2str(opt_meta.bondexp_policy));
        return res; // No update
    }

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    // assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    size_t posP = pos;
    size_t pos0 = state.get_direction() > 0 ? posR : posL;
    auto  &mpsP = state.get_mps_site(posP);
    auto  &mps0 = state.get_mps_site(pos0);

    auto dimL_old = mpsL.dimensions();
    auto dimR_old = mpsR.dimensions();
    auto dimP_old = mpsP.dimensions();

    auto res = get_mixing_factors_postopt_rnorm(pos_expanded, state, model, edges, opt_meta);

    if(res.alpha_h1v == 0 and res.alpha_h2v == 0) {
        res.msg = fmt::format("Expansion canceled: {}{} - {}{} | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                              mpsR.dimensions(), res.alpha_mps, res.alpha_h1v, res.alpha_h2v);
        return res;
    }

    tools::log->debug("Expanding {}{} - {}{} | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e} | factor {:.1e}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                      mpsR.dimensions(), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, opt_meta.bondexp_factor);

    // Set up the SVD
    // Bond dimension can't grow faster than x spin_dim.
    auto svd_cfg             = opt_meta.svd_cfg.value();
    auto bond_max            = std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR());
    svd_cfg.truncation_limit = svd_cfg.truncation_limit.value_or(settings::precision::svd_truncation_min);
    svd_cfg.rank_max         = std::min(bond_max, svd_cfg.rank_max.value_or(bond_max));

    decltype(auto) M     = mpsP.template get_M_as<Scalar>();
    decltype(auto) N     = mps0.template get_M_as<Scalar>();
    const auto    &mpoP  = model.get_mpo(posP);
    const auto    &envP1 = state.get_direction() > 0 ? edges.get_env_eneL(posP) : edges.get_env_eneR(posP);
    const auto    &envP2 = state.get_direction() > 0 ? edges.get_env_varL(posP) : edges.get_env_varR(posP);
    const auto     P1    = res.alpha_h1v == 0 ? Eigen::Tensor<Scalar, 3>() : envP1.template get_expansion_term<Scalar>(mpsP, mpoP);
    const auto     P2    = res.alpha_h2v == 0 ? Eigen::Tensor<Scalar, 3>() : envP2.template get_expansion_term<Scalar>(mpsP, mpoP);

    if(state.get_direction() > 0) {
        auto [M_P_del, N_0] = delinearize_expansion_terms_l2r(M, N, P1, P2, res, svd_cfg);
        merge_expansion_terms_l2r(state, mpsP, M_P_del, mps0, N_0, svd_cfg);
        tools::log->debug("Bond expansion l2r {} | {} α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.3e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded,
                          flag2str(opt_meta.bondexp_policy), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(),
                          svd_cfg.rank_max.value(), dimP_old, M.dimensions(), res.dimMP, mpsP.dimensions());
    } else {
        auto [M_P_del, N_0] = delinearize_expansion_terms_r2l(M, N, P1, P2, res, svd_cfg);
        merge_expansion_terms_r2l(state, mps0, N_0, mpsP, M_P_del, svd_cfg);
        tools::log->debug("Bond expansion r2l {} | {} α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.3e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded,
                          flag2str(opt_meta.bondexp_policy), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(),
                          svd_cfg.rank_max.value(), dimP_old, M.dimensions(), M_P_del.dimensions(), mpsP.dimensions());
    }

    if(mpsP.dimensions()[0] * std::min(mpsP.dimensions()[1], mpsP.dimensions()[2]) < std::max(mpsP.dimensions()[1], mpsP.dimensions()[2])) {
        tools::log->warn("Bond expansion failed: {} -> {}", dimP_old, mpsP.dimensions());
    }

    if(dimL_old[1] != mpsL.get_chiL()) throw except::runtime_error("mpsL changed chiL during bond expansion: {} -> {}", dimL_old, mpsL.dimensions());
    if(dimR_old[2] != mpsR.get_chiR()) throw except::runtime_error("mpsR changed chiR during bond expansion: {} -> {}", dimR_old, mpsR.dimensions());
    if constexpr(settings::debug_expansion) mpsL.assert_normalized();
    if constexpr(settings::debug_expansion) mpsR.assert_normalized();
    state.clear_cache();
    state.clear_measurements();
    env::rebuild_edges(state, model, edges);

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}
/* clang-format off */
template BondExpansionResult<fp32>  tools::finite::env::expand_bond_postopt_1site(StateFinite<fp32> &, const ModelFinite<fp32> &, EdgesFinite<fp32> &, const OptMeta &);
template BondExpansionResult<fp64>  tools::finite::env::expand_bond_postopt_1site(StateFinite<fp64> &, const ModelFinite<fp64> &, EdgesFinite<fp64> &, const OptMeta &);
template BondExpansionResult<fp128> tools::finite::env::expand_bond_postopt_1site(StateFinite<fp128> &, const ModelFinite<fp128> &, EdgesFinite<fp128> &, const OptMeta &);
template BondExpansionResult<cx32>  tools::finite::env::expand_bond_postopt_1site(StateFinite<cx32> &, const ModelFinite<cx32> &, EdgesFinite<cx32> &, const OptMeta &);
template BondExpansionResult<cx64>  tools::finite::env::expand_bond_postopt_1site(StateFinite<cx64> &, const ModelFinite<cx64> &, EdgesFinite<cx64> &, const OptMeta &);
template BondExpansionResult<cx128> tools::finite::env::expand_bond_postopt_1site(StateFinite<cx128> &, const ModelFinite<cx128> &, EdgesFinite<cx128> &, const OptMeta &);
/* clang-format on */

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                         EdgesFinite<Scalar> &edges, const OptMeta &opt_meta) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_forward_nsite: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_bond_forward_nsite: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for bond expansion");
    if(not opt_meta.svd_cfg.has_value()) throw except::logic_error("opt_meta.svd_cfg has no been set");

    bool nopreopt = !has_any_flags(opt_meta.bondexp_policy, BondExpansionPolicy::PREOPT_NSITE_REAR, BondExpansionPolicy::PREOPT_NSITE_FORE);
    if(nopreopt)
        throw except::logic_error("expand_bond_ssite_preopt: BondExpansionPolicy::PREOPT_NSITE_REAR|PREOPT_NSITE_FORE was not set in opt_meta.bondexp_policy");
    if(opt_meta.optExit != OptExit::NONE) throw except::logic_error("expand_bond_ssite_preopt: bond expansion requires opt_meta.optExit == OptExit::NONE");

    // Determine which bond to expand
    // We need at least 1 extra site, apart from the active site(s), to expand the environment for the upcoming optimization.
    size_t blocksize = std::max(opt_meta.bondexp_blocksize, state.active_sites.size() + 1);
    size_t posL      = state.active_sites.front();
    size_t posR      = state.active_sites.back();
    size_t length    = state.template get_length<size_t>();

    // Grow the posL and posR boundary until they cover the block size
    long poslL      = safe_cast<long>(posL);
    long poslR      = safe_cast<long>(posR);
    long lengthl    = safe_cast<long>(length);
    long blocksizel = safe_cast<long>(blocksize);
    if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::PREOPT_NSITE_FORE)) {
        if(state.get_direction() > 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
        if(state.get_direction() < 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
    } else if(has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::PREOPT_NSITE_REAR)) {
        if(state.get_direction() > 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
        if(state.get_direction() < 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
    }

    posL = safe_cast<size_t>(poslL);
    posR = safe_cast<size_t>(poslR);
    if(posR - posL + 1 > blocksize) throw except::logic_error("error in block size selection | posL {} to posR {} != blocksize {}", posL, posR, blocksize);

    auto pos_active_and_expanded = num::range<size_t>(posL, posR + 1);

    if(pos_active_and_expanded.size() < 2ul) { return BondExpansionResult<Scalar>(); }

    // Define the left and right mps that will get modified
    auto        res  = get_mixing_factors_preopt_krylov(pos_active_and_expanded, state, model, edges, opt_meta);
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    assert(mpsL.get_chiR() == mpsR.get_chiL());
    // assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    res.ene_old = tools::finite::measure::energy(state, model, edges);
    res.var_old = tools::finite::measure::energy_variance(state, model, edges);

    if(res.alpha_h1v == 0 and res.alpha_h2v == 0) {
        tools::log->debug("Expansion canceled: {}({}) - {}({}) | αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(),
                          mpsR.get_position(), res.alpha_h1v, res.alpha_h2v);
        return res;
    }

    tools::log->debug("Expanding {}({}) - {}({}) | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(), mpsR.get_position(),
                      res.alpha_mps, res.alpha_h1v, res.alpha_h2v);

    mps::merge_multisite_mps(state, res.mixed_blk, pos_active_and_expanded, state.template get_position<long>(), MergeEvent::EXP, opt_meta.svd_cfg.value());

    res.dims_new = state.get_mps_dims(pos_active_and_expanded);
    res.bond_new = state.get_bond_dims(pos_active_and_expanded);

    tools::log->debug("Bond expansion pos {} | {} {} | αₑ:{:.2e} αᵥ:{:.2e} | svd_ε {:.2e} | χlim {} | χ {} -> {}", pos_active_and_expanded,
                      enum2sv(opt_meta.optAlgo), enum2sv(opt_meta.optRitz), res.alpha_h1v, res.alpha_h2v, opt_meta.svd_cfg->truncation_limit.value(),
                      opt_meta.svd_cfg->rank_max.value(), res.bond_old, res.bond_new);
    state.clear_cache();
    state.clear_measurements();
    for(const auto &mps : state.get_mps(pos_active_and_expanded)) mps.get().assert_normalized();
    env::rebuild_edges(state, model, edges);

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    assert(num::prod(res.dimL_new) > 0);
    assert(num::prod(res.dimR_new) > 0);
    assert(mpsL.get_chiR() == mpsR.get_chiL());

    res.ene_new = tools::finite::measure::energy(state, model, edges);
    res.var_new = tools::finite::measure::energy_variance(state, model, edges);
    if(std::isnan(res.ene_new)) throw except::runtime_error("res.ene_new is nan");
    if(std::isnan(res.var_new)) throw except::runtime_error("res.var_new is nan");
    res.ok = true;
    return res;
}
/* clang-format off */
template BondExpansionResult<fp32>  tools::finite::env::expand_bond_preopt_nsite(StateFinite<fp32> &, const ModelFinite<fp32> &, EdgesFinite<fp32> &, const OptMeta &);
template BondExpansionResult<fp64>  tools::finite::env::expand_bond_preopt_nsite(StateFinite<fp64> &, const ModelFinite<fp64> &, EdgesFinite<fp64> &, const OptMeta &);
template BondExpansionResult<fp128> tools::finite::env::expand_bond_preopt_nsite(StateFinite<fp128> &, const ModelFinite<fp128> &, EdgesFinite<fp128> &, const OptMeta &);
template BondExpansionResult<cx32>  tools::finite::env::expand_bond_preopt_nsite(StateFinite<cx32> &, const ModelFinite<cx32> &, EdgesFinite<cx32> &, const OptMeta &);
template BondExpansionResult<cx64>  tools::finite::env::expand_bond_preopt_nsite(StateFinite<cx64> &, const ModelFinite<cx64> &, EdgesFinite<cx64> &, const OptMeta &);
template BondExpansionResult<cx128> tools::finite::env::expand_bond_preopt_nsite(StateFinite<cx128> &, const ModelFinite<cx128> &, EdgesFinite<cx128> &, const OptMeta &);
/* clang-format on */

template<typename Scalar>
void tools::finite::env::assert_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert energy edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites, we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();

    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);

    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &ene = edges.get_env_eneL(pos);
        if(pos == 0 and not ene.has_block()) throw except::runtime_error("ene L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_next = edges.get_env_eneL(pos + 1);
        ene_next.assert_unique_id(ene, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);

    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &ene = edges.get_env_eneR(pos);
        if(pos == state.get_length() - 1 and not ene.has_block()) throw except::runtime_error("ene R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_prev = edges.get_env_eneR(pos - 1);
        ene_prev.assert_unique_id(ene, mps, mpo);
    }
}
template void tools::finite::env::assert_edges_ene(const StateFinite<fp32> &, const ModelFinite<fp32> &, const EdgesFinite<fp32> &);
template void tools::finite::env::assert_edges_ene(const StateFinite<fp64> &, const ModelFinite<fp64> &, const EdgesFinite<fp64> &);
template void tools::finite::env::assert_edges_ene(const StateFinite<fp128> &, const ModelFinite<fp128> &, const EdgesFinite<fp128> &);
template void tools::finite::env::assert_edges_ene(const StateFinite<cx32> &, const ModelFinite<cx32> &, const EdgesFinite<cx32> &);
template void tools::finite::env::assert_edges_ene(const StateFinite<cx64> &, const ModelFinite<cx64> &, const EdgesFinite<cx64> &);
template void tools::finite::env::assert_edges_ene(const StateFinite<cx128> &, const ModelFinite<cx128> &, const EdgesFinite<cx128> &);

template<typename Scalar>
void tools::finite::env::assert_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert variance edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites, we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();

    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &var = edges.get_env_varL(pos);
        if(pos == 0 and not var.has_block()) throw except::runtime_error("var L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_next = edges.get_env_varL(pos + 1);
        var_next.assert_unique_id(var, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &var = edges.get_env_varR(pos);
        if(pos == state.get_length() - 1 and not var.has_block()) throw except::runtime_error("var R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_prev = edges.get_env_varR(pos - 1);
        var_prev.assert_unique_id(var, mps, mpo);
    }
}
template void tools::finite::env::assert_edges_var(const StateFinite<fp32> &, const ModelFinite<fp32> &, const EdgesFinite<fp32> &);
template void tools::finite::env::assert_edges_var(const StateFinite<fp64> &, const ModelFinite<fp64> &, const EdgesFinite<fp64> &);
template void tools::finite::env::assert_edges_var(const StateFinite<fp128> &, const ModelFinite<fp128> &, const EdgesFinite<fp128> &);
template void tools::finite::env::assert_edges_var(const StateFinite<cx32> &, const ModelFinite<cx32> &, const EdgesFinite<cx32> &);
template void tools::finite::env::assert_edges_var(const StateFinite<cx64> &, const ModelFinite<cx64> &, const EdgesFinite<cx64> &);
template void tools::finite::env::assert_edges_var(const StateFinite<cx128> &, const ModelFinite<cx128> &, const EdgesFinite<cx128> &);

template<typename Scalar>
void tools::finite::env::assert_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
}
template void tools::finite::env::assert_edges(const StateFinite<fp32> &, const ModelFinite<fp32> &, const EdgesFinite<fp32> &);
template void tools::finite::env::assert_edges(const StateFinite<fp64> &, const ModelFinite<fp64> &, const EdgesFinite<fp64> &);
template void tools::finite::env::assert_edges(const StateFinite<fp128> &, const ModelFinite<fp128> &, const EdgesFinite<fp128> &);
template void tools::finite::env::assert_edges(const StateFinite<cx32> &, const ModelFinite<cx32> &, const EdgesFinite<cx32> &);
template void tools::finite::env::assert_edges(const StateFinite<cx64> &, const ModelFinite<cx64> &, const EdgesFinite<cx64> &);
template void tools::finite::env::assert_edges(const StateFinite<cx128> &, const ModelFinite<cx128> &, const EdgesFinite<cx128> &);

template<typename Scalar>
void tools::finite::env::rebuild_edges_ene(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_ene: fLBIT algorithm should never rebuild energy edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto   t_reb   = tid::tic_scope("rebuild_edges_ene", tid::higher);
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites then we can build up until current position
    /*
     * LOG:
     * - 2021-10-14:
     *      Just had a terribly annoying bug:
     *      Moving the center position clears active_sites, which caused problems when turning back from the right edge.
     *          1)  active_sites [A(L-1), AC(L)] are updated, left edge exist for A(L-1), right edge exists for AC(L)
     *          2)  move dir -1, clear active sites
     *          3)  assert_edges checks up to AC(L-1), but this site has a stale right edge.
     *      Therefore, one would have to rebuild edges between steps 2) and 3) to solve this issue
     *
     *      One solution would be to always rebuild edges up to the current position from both sides, but that would be
     *      wasteful. Instead, we could just accept that some edges are stale after moving the center-point,
     *      as long as we rebuild those when sites get activated again.
     *
     * - 2024-07-23
     *      Just found a way to calculate the optimal mixing factor for subspace expansion.
     *      In forward expansion we need H_eff including one site beyond active_sites.
     *      Therefore, we need to build more environments than we have needed previously.
     *      Examples:
     *          - Forward, direction == 1, active_sites = [5,6,7,8].
     *            Then we expand bond [8,9], and so we need envL[8] and envR[9].
     *          - Forward, direction == -1, active_sites = [5,6,7,8].
     *            Then we expand bond [4,5], and so we need envL[4] and envR[5].
     *      These environments weren't built before.
     *      Therefore, we must now rebuild
     *          - envL 0 to active_sites.back()
     *          - envR L to active_sites.front()
     */

    // If there are no active sites we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this rebuild after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    std::vector<size_t> env_pos_log;
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &env_here = edges.get_env_eneL(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == 0) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_ene: No eneL block detected at pos {}", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &env_rght = edges.get_env_eneL(pos + 1);
        auto  id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneL edges: {}", env_pos_log);

    env_pos_log.clear();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &env_here = edges.get_env_eneR(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == state.get_length() - 1) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_ene: No eneR block detected at pos {}", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &env_left = edges.get_env_eneR(pos - 1);
        auto  id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneR edges: {}", env_pos_log);
    if(not edges.get_env_eneL(posL_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneL has undefined block");
    if(not edges.get_env_eneR(posR_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneR has undefined block");
}
template void tools::finite::env::rebuild_edges_ene(const StateFinite<fp32> &, const ModelFinite<fp32> &, EdgesFinite<fp32> &);
template void tools::finite::env::rebuild_edges_ene(const StateFinite<fp64> &, const ModelFinite<fp64> &, EdgesFinite<fp64> &);
template void tools::finite::env::rebuild_edges_ene(const StateFinite<fp128> &, const ModelFinite<fp128> &, EdgesFinite<fp128> &);
template void tools::finite::env::rebuild_edges_ene(const StateFinite<cx32> &, const ModelFinite<cx32> &, EdgesFinite<cx32> &);
template void tools::finite::env::rebuild_edges_ene(const StateFinite<cx64> &, const ModelFinite<cx64> &, EdgesFinite<cx64> &);
template void tools::finite::env::rebuild_edges_ene(const StateFinite<cx128> &, const ModelFinite<cx128> &, EdgesFinite<cx128> &);

template<typename Scalar>
void tools::finite::env::rebuild_edges_var(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_var: fLBIT algorithm should never rebuild variance edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("rebuild_edges_var: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("rebuild_edges_var: All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto t_reb = tid::tic_scope("rebuild_edges_var", tid::level::higher);

    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites, we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of the center site.
    // We could always keep all edges refreshed, but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.template get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    }

    std::vector<size_t> env_pos_log;
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &env_here = edges.get_env_varL(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == 0) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_var: No varL block detected at pos {}", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &env_rght = edges.get_env_varL(pos + 1);
        auto  id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }

    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varL edges: {}", env_pos_log);
    env_pos_log.clear();
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    }

    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &env_here = edges.get_env_varR(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == state.get_length() - 1) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_var: No varR block detected at pos {}", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &env_left = edges.get_env_varR(pos - 1);
        auto  id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varR edges: {}", env_pos_log);
    if(not edges.get_env_varL(posL_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varL has undefined block");
    if(not edges.get_env_varR(posR_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varR has undefined block");
}
template void tools::finite::env::rebuild_edges_var(const StateFinite<fp32> &, const ModelFinite<fp32> &, EdgesFinite<fp32> &);
template void tools::finite::env::rebuild_edges_var(const StateFinite<fp64> &, const ModelFinite<fp64> &, EdgesFinite<fp64> &);
template void tools::finite::env::rebuild_edges_var(const StateFinite<fp128> &, const ModelFinite<fp128> &, EdgesFinite<fp128> &);
template void tools::finite::env::rebuild_edges_var(const StateFinite<cx32> &, const ModelFinite<cx32> &, EdgesFinite<cx32> &);
template void tools::finite::env::rebuild_edges_var(const StateFinite<cx64> &, const ModelFinite<cx64> &, EdgesFinite<cx64> &);
template void tools::finite::env::rebuild_edges_var(const StateFinite<cx128> &, const ModelFinite<cx128> &, EdgesFinite<cx128> &);

template<typename Scalar>
void tools::finite::env::rebuild_edges(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    rebuild_edges_ene(state, model, edges);
    rebuild_edges_var(state, model, edges);
}
template void tools::finite::env::rebuild_edges(const StateFinite<fp32> &, const ModelFinite<fp32> &, EdgesFinite<fp32> &);
template void tools::finite::env::rebuild_edges(const StateFinite<fp64> &, const ModelFinite<fp64> &, EdgesFinite<fp64> &);
template void tools::finite::env::rebuild_edges(const StateFinite<fp128> &, const ModelFinite<fp128> &, EdgesFinite<fp128> &);
template void tools::finite::env::rebuild_edges(const StateFinite<cx32> &, const ModelFinite<cx32> &, EdgesFinite<cx32> &);
template void tools::finite::env::rebuild_edges(const StateFinite<cx64> &, const ModelFinite<cx64> &, EdgesFinite<cx64> &);
template void tools::finite::env::rebuild_edges(const StateFinite<cx128> &, const ModelFinite<cx128> &, EdgesFinite<cx128> &);
