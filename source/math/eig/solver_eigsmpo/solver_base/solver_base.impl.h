#pragma once
#include "../solver_base.h"
#include "../StopReason.h"
#include "io/fmt_custom.h"
#include "JacobiDavidsonOperator.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/linalg/tensor/to_string.h"
#include "math/tenx.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/contraction_policy.h"
#include "tools/finite/opt_mps.h"
#include <Eigen/Eigenvalues>
#include <spdlog/sinks/stdout_color_sinks.h>
namespace settings {
#if defined(NDEBUG)
    inline constexpr bool debug_solver = false;
#else
    inline constexpr bool debug_solver = true;
#endif
}

template<typename Scalar> void solver_base<Scalar>::OrthMeta::analyze_l2_orthonormality(const Eigen::Ref<const MatrixType> &Y) {
    if(Y.cols() == 0) return;
    MatrixType I = MatrixType::Identity(Y.cols(), Y.cols());
    Gram         = Y.adjoint() * Y;
    Gram_symm    = Gram;
    Gram_skew    = Gram;
    orthError    = (Gram - I).norm();
    symmError    = orthError;
    skewError    = orthError;
    Rdiag        = Gram_symm.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
}
template<typename Scalar> void solver_base<Scalar>::OrthMeta::analyze_h2_orthonormality(const Eigen::Ref<const MatrixType> &Y,
                                                                                        const Eigen::Ref<const MatrixType> &H2Y) {
    if(Y.cols() != H2Y.cols() || Y.rows() != H2Y.rows()) return;

    MatrixType I = MatrixType::Identity(Y.cols(), Y.cols());

    MatrixType G1 = Y.adjoint() * H2Y;
    MatrixType G2 = H2Y.adjoint() * Y;

    Gram      = G1;
    Gram_symm = (G1 + G2) * half;
    Gram_skew = (G1 - G2) * half;

    orthError     = (Gram - I).norm();
    symmError     = (Gram_symm - I).norm();
    skewError     = Gram_skew.norm();
    skewError_fwd = skewError;
    Rdiag         = Gram_symm.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
}

template<typename Scalar> void solver_base<Scalar>::OrthMeta::analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X,
                                                                                       const Eigen::Ref<const MatrixType> &Y) {
    if(Y.cols() == 0) return;
    Gram = X.adjoint() * Y;

    Gram_symm = Gram;
    Gram_skew = Gram;
    orthError = Gram.norm();
    symmError = orthError;
    skewError = orthError;
    Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
}
template<typename Scalar> void solver_base<Scalar>::OrthMeta::analyze_h2_orthogonality(const Eigen::Ref<const MatrixType> &X,
                                                                                       const Eigen::Ref<const MatrixType> &H2X,
                                                                                       const Eigen::Ref<const MatrixType> &Y,
                                                                                       const Eigen::Ref<const MatrixType> &H2Y) {
    if(Y.cols() != H2Y.cols() || Y.rows() != H2Y.rows()) return;
    if(X.cols() != H2X.cols() || X.rows() != H2X.rows()) return;
    if(Y.rows() != X.rows()) return;

    MatrixType G1 = X.adjoint() * H2Y;
    MatrixType G2 = H2X.adjoint() * Y;

    MatrixType I = MatrixType::Identity(G1.rows(), G1.cols());

    Gram      = G1;
    Gram_symm = (G1 + G2) * half;
    Gram_skew = (G1 - G2) * half;

    orthError = (Gram - I).norm();
    symmError = (Gram_symm - I).norm();
    skewError = Gram_skew.norm();
    Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
}

template<typename Scalar>
void solver_base<Scalar>::setLogger(spdlog::level::level_enum level, const std::string &name) {
    eiglog = spdlog::get(name);
    if(!eiglog) {
        eiglog = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
        eiglog->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%n]%^[%=8l]%$ %v");
        eiglog->set_level(level);
    } else {
        if(level != eiglog->level()) { eiglog->set_level(level); }
    }
}

template<typename Scalar>
solver_base<Scalar>::solver_base(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1,
                                 MatVecMPOS<Scalar> &H2, MatVecMPOS<Scalar> &H1H2, spdlog::level::level_enum logLevel_)
    : logLevel(logLevel_), //
      nev(nev),            //
      ncv(ncv),            //
      algo(algo),          //
      ritz(ritz),          //
      H1(H1),              //
      H2(H2),              //
      H1H2(H1H2),          //
      V(V) {
    setLogger(logLevel, fmt::format("eigs|{}", enum2sv(algo)));
    N         = H1.get_size();
    mps_size  = H1.get_size();
    mps_shape = H1.get_shape_mps();
    nev       = std::min(nev, N);
    ncv       = std::min(std::max(nev, ncv), N);
    b         = std::min(std::max(nev, b), N / 2);
    status.rNorms.setOnes(nev);
    status.eigVal.setOnes(nev);
    status.oldVal.setOnes(nev);
    status.absDiff.setOnes(nev);
    status.relDiff.setOnes(nev);

    assert(mps_size == H1.rows());
    assert(mps_size == H2.rows());
    assert(mps_size == H1H2.rows());
    set_preconditioner_params();
}

template<typename Scalar>
auto solver_base<Scalar>::get_residuals(const Eigen::Ref<VectorReal> &Y, const Eigen::Ref<MatrixType> &H1V, const Eigen::Ref<MatrixType> &H2V)
    -> std::pair<MatrixType, VectorReal> {
    MatrixType S = H1V - H2V * Y.asDiagonal();
    VectorReal N = S.colwise().norm();
    VectorReal D = H1V.colwise().norm().array() + H2V.colwise().norm().array() * Y.cwiseAbs().array();

    constexpr auto dmin = std::numeric_limits<RealScalar>::min();
    D                   = D.cwiseMax(VectorReal::Constant(D.size(), dmin));

    VectorReal R = N.cwiseQuotient(D);
    return {S, R};
};

template<typename Scalar>
solver_base<Scalar>::RealScalar solver_base<Scalar>::rNormTol([[maybe_unused]] Eigen::Index n) const {
    assert(abstol > RealScalar{0});
    assert(abstol < RealScalar{1});
    auto tol = abstol;

    if(reltol > RealScalar{0}) {
        assert(reltol < RealScalar{1});
        assert(n < status.rNorms_init.size());
        assert(status.rNorms.size() == status.rNorms_init.size());
        tol = std::clamp(reltol * status.rNorms_init(n), tol, RealScalar{0.99f});
    }
    return tol;
}

template<typename Scalar>
solver_base<Scalar>::VectorReal solver_base<Scalar>::rNormTols() const {
    VectorReal rNormTols(nev);
    for(Eigen::Index n = 0; n < nev; ++n) { rNormTols(n) = rNormTol(n); }
    return rNormTols;
}

template<typename Scalar>
void solver_base<Scalar>::set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize) {
    if(jcbMaxBlockSize >= 0) {
        H1.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H1H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H1.factorization   = eig::Factorization::LU;
        H2.factorization   = eig::Factorization::LLT;
        H1H2.factorization = eig::Factorization::LU;
    }
}

template<typename Scalar>
void solver_base<Scalar>::set_jcbOverlapSize(Eigen::Index jcbOverlapSize) {
    if(jcbOverlapSize >= 0) {
        H1.set_jcbOverlapSize(jcbOverlapSize);
        H2.set_jcbOverlapSize(jcbOverlapSize);
        H1H2.set_jcbOverlapSize(jcbOverlapSize);
    }
}

template<typename Scalar>
void solver_base<Scalar>::set_jcbNumPasses(Eigen::Index jcbNumPasses) {
    if(jcbNumPasses >= 0) {
        H1.set_jcbNumPasses(jcbNumPasses);
        H2.set_jcbNumPasses(jcbNumPasses);
        H1H2.set_jcbNumPasses(jcbNumPasses);
    }
}

template<typename Scalar>
Eigen::Index solver_base<Scalar>::get_jcbMaxBlockSize() const {
    assert(H1.get_jcbMaxBlockSize() == H2.get_jcbMaxBlockSize());
    return H1.get_jcbMaxBlockSize();
}
template<typename Scalar>
Eigen::Index solver_base<Scalar>::get_jcbOverlapSize() const {
    assert(H1.get_jcbOverlapSize() == H2.get_jcbOverlapSize());
    return H1.get_jcbOverlapSize();
}

template<typename Scalar>
Eigen::Index solver_base<Scalar>::get_jcbNumPasses() const {
    assert(H1.get_jcbNumPasses() == H2.get_jcbNumPasses());
    return H1.get_jcbNumPasses();
}

template<typename Scalar>
void solver_base<Scalar>::set_preconditioner_type(eig::Preconditioner preconditioner_type_) {
    preconditioner_type = preconditioner_type_;
    H1.preconditioner   = preconditioner_type;
    H2.preconditioner   = preconditioner_type;
    H1H2.preconditioner = preconditioner_type;
    use_preconditioner  = preconditioner_type != eig::Preconditioner::NONE;
}
template<typename Scalar>
void solver_base<Scalar>::set_preconditioner_params(Eigen::Index maxiters, RealScalar initialTol, Eigen::Index jcbMaxBlockSize) {
    assert(initialTol > 0);
    use_preconditioner = preconditioner_type != eig::Preconditioner::NONE;
    H1.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::IND);
    H2.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::IND); // IND (MINRES) is often faster than DEF (CG) with H2 (which is PSD)
    H1H2.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::IND);
    H1.set_jcbMaxBlockSize(jcbMaxBlockSize);
    H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
    H1H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
    H1.factorization   = eig::Factorization::LU;
    H2.factorization   = eig::Factorization::LLT;
    H1H2.factorization = eig::Factorization::LU;
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::get_op_norm_estimate(std::optional<RealScalar> eigval) const {
    switch(algo) {
        case OptAlgo::DMRG: {
            auto H_maxeval = std::max({std::abs(status.T_min_eval), std::abs(status.T_max_eval)}); // Largest seen eigenvalue of T2 (projected H2)
            auto H_maxnorm = HQ.norm() / Q.norm();                                                 // Estimate the op norm from the current basis
            auto H_pownorm = H1.get_op_norm();                                                     // Norm estaimte from power iteration
            return std::max({H_maxeval, H_maxnorm, H_pownorm});
        }
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
        case OptAlgo::XDMRG: {
            auto H_maxeval = std::max({std::abs(status.T_min_eval), std::abs(status.T_max_eval)}); // Largest seen eigenvalue of T2 (projected H2)
            auto H_maxnorm = HQ.norm() / Q.norm();                                                 // Estimate the op norm from the current basis
            auto H_pownorm = H2.get_op_norm();                                                     // Norm estaimte from power iteration
            return std::max({H_maxeval, H_maxnorm, H_pownorm});
        }
        case OptAlgo::GDMRG: {
            auto H1_maxeval = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval)}); // Largest seen eigenvalue of T1 (projected H1)
            auto H2_maxeval = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval)}); // Largest seen eigenvalue of T2 (projected H2)
            auto H1_maxnorm = H1Q.norm() / Q.norm();                                                  // Estimate the op norm from the current basis
            auto H2_maxnorm = H2Q.norm() / Q.norm();                                                  // Estimate the op norm from the current basis
            auto H1_pownorm = H1.get_op_norm();                                                       // Norm estaimte from power iteration
            auto H2_pownorm = H2.get_op_norm();                                                       // Norm estaimte from power iteration
            auto H1_normest = std::max({H1_maxeval, H1_maxnorm, H1_pownorm});
            auto H2_normest = std::max({H2_maxeval, H2_maxnorm, H2_pownorm});
            if(not eigval.has_value()) { eigval = RealScalar{1}; }
            auto abs_lambda = std::abs(eigval.value());

            // RealScalar H1_vnorm = H1V.norm() / V.norm();
            // RealScalar H2_vnorm = H2V.norm() / V.norm();
            //  tools::log->debug("Op norm H1: max eval = {:.3e},  |H1Q|/|Q| = {:.3e}, |H1V|/|V| = {:.3e} pow iter estimate = {:.3e}", fp(H1_maxeval),
            //                    fp(H1_maxnorm), fp(H1_vnorm), fp(H1_pownorm));
            //  tools::log->debug("Op norm H2: max eval = {:.3e},  |H2Q|/|Q| = {:.3e}, |H2V|/|V| = {:.3e} pow iter estimate = {:.3e}", fp(H2_maxeval),
            //                    fp(H2_maxnorfm), fp(H2_vnorm), fp(H2_pownorm));
            //  tools::log->debug("Op norm  |H1Q|/|H2Q|  = {:.3e},  |H1V|/|H2V|  = {:.3e}", fp(H1_maxnorm / H2_maxnorm), fp(H1_vnorm / H2_vnorm));
            //  tools::log->debug("Op norm  |H1Q|+|H2Q|λ = {:.3e},  |H1V|+|H2V|λ = {:.3e}", fp(H1_maxnorm + abs_lambda * H2_maxnorm),
            //                    fp(H1_vnorm + abs_lambda * H2_vnorm));

            return H1_normest + abs_lambda * H2_normest;
        }

        default: throw except::runtime_error("unrecognized algo");
    }
}

template<typename Scalar>
typename solver_base<Scalar>::VectorReal solver_base<Scalar>::get_op_norm_estimates(Eigen::Ref<VectorReal> eigvals) const {
    VectorReal opnorms(eigvals.size());
    for(Eigen::Index i = 0; i < eigvals.size(); ++i) opnorms(i) = get_op_norm_estimate(eigvals(i));
    return opnorms;
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::Status::max_eval_estimate() const {
    auto it = std::max_element(max_eval_history.begin(), max_eval_history.end());
    if(it != max_eval_history.end()) { return std::max(RealScalar{1}, *it); }
    throw except::runtime_error("max_eval_history is empty");
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::Status::min_eval_estimate() const {
    auto it = std::min_element(min_eval_history.begin(), min_eval_history.end());
    if(it != min_eval_history.end()) { return *it; }
    throw except::runtime_error("min_eval_history is empty");
}

template<typename Scalar>
void solver_base<Scalar>::Status::commit_evals(RealScalar min_eval, RealScalar max_eval) {
    max_eval_history.push_back(max_eval);
    min_eval_history.push_back(min_eval);
    while(max_eval_history.size() > max_history_size) { max_eval_history.pop_front(); }
    while(min_eval_history.size() > max_history_size) { min_eval_history.pop_front(); }
}

template<typename Scalar>
void solver_base<Scalar>::set_chebyshevFilterRelGapThreshold(RealScalar threshold) {
    assert(threshold >= 0);
    if(threshold >= 0) { chebyshev_filter_relative_gap_threshold = threshold; }
}

template<typename Scalar>
void solver_base<Scalar>::set_chebyshevFilterLambdaCutBias(RealScalar bias) {
    chebyshev_filter_lambda_cut_bias = std::clamp<RealScalar>(bias, eps, 1 - eps);
}

template<typename Scalar>
void solver_base<Scalar>::set_chebyshevFilterDegree(Eigen::Index degree) {
    if(degree > 0) { chebyshev_filter_degree = degree; }
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       // input Q (orthonormal)
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
        eiglog->warn("lambda_cut outside range [lambda_min, lambda_max]");
        return Qref;
    }
    if(bv < eps * std::abs(av)) {
        eiglog->warn("bv < eps");
        return Qref;
    }

    RealScalar x0 = (lambda_cut - av) / bv;
    // Clamp x0 into [-1,1] to avoid NaN
    x0              = std::clamp(x0, RealScalar{-1}, RealScalar{1});
    RealScalar norm = std::cos(degree * std::acos(x0)); // = T_k(x0)

    if(degree == 1) { return (MultH(Qref) - av * Qref) * (RealScalar{1} / bv / norm); }

    // eiglog->info("Chebyshev filter: x0={:.5e} norm={:.5e} lambda_min={:.5e} lambda_cut={:.5e} lambda_max={:.5e}", x0, norm, lambda_min, lambda_cut,
    // lambda_max);
    if(std::abs(norm) < eps or !std::isfinite(norm)) {
        // normalization too small; skip filtering
        eiglog->warn("norm invalid {:.5e}", fp(norm));
        return Qref;
    }

    // Chebyshev recurrence: T_k = 2*( (H - aI)/bspec ) T_{k-1} - T_{k-2}
    MatrixType Tkm2 = Qref;
    MatrixType Tkm1 = (MultH(Qref) - av * Qref) * (RealScalar{1} / bv);
    MatrixType Tcur(N, Qref.cols());
    for(int k = 2; k <= degree; ++k) {
        Tcur = (MultH(Tkm1) - av * Tkm1) * (RealScalar{2} / bv) - Tkm2;
        assert(std::isfinite(Tcur.norm()));
        Tkm2 = std::move(Tkm1);
        Tkm1 = std::move(Tcur);
    }
    return Tkm1 * (Scalar{1} / norm);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::qr_and_chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref) {
    if(Qref.cols() == 0) return Qref;
    if(chebyshev_filter_degree == 0) return Qref;
    if(T_evals.size() <= 1) return Qref;
    // calculate the gap and relative gap
    auto       select_2 = get_ritz_indices(ritz, 0, 2, T_evals);
    VectorReal evals    = T_evals(select_2);

    auto absgap = std::abs(evals(1) - evals(0));
    auto relgap = absgap / status.max_eval_estimate();
    assert(std::isfinite(relgap));
    if(relgap > chebyshev_filter_relative_gap_threshold) return Qref;
    auto bias = chebyshev_filter_lambda_cut_bias;
    if(ritz == OptRitz::LM or ritz == OptRitz::LR) { bias = RealScalar{1} - bias; }

    RealScalar lambda_min = status.min_eval_estimate() * RealScalar{0.99f};
    RealScalar lambda_max = status.max_eval_estimate() * RealScalar{1.01f};
    if(lambda_min > lambda_max) std::swap(lambda_min, lambda_max);
    RealScalar lambda_cut = lambda_min + bias * (lambda_max - lambda_min);
    lambda_cut            = std::clamp(lambda_cut, lambda_min, lambda_max);

    // eiglog->info("Applying the chebyshev filter | gap: abs={:.5e} rel={:.5e}", absgap, relgap);
    // Re orthogonalize

    assert_allFinite(Qref);
    MatrixType Qnew = Qref;
    hhqr.compute(Qnew);
    Qnew = hhqr.householderQ().setLength(Qnew.cols()) * MatrixType::Identity(N, Qnew.cols()); //
    assert_allFinite(Qnew);
    Qnew = chebyshevFilter(Qnew, lambda_min, lambda_max, lambda_cut, chebyshev_filter_degree);
    assert_allFinite(Qnew);
    return Qnew;
}

// namespace Eigen {
//     template<>
//     struct NumTraits<std::float128_t> : NumTraits<double> {
//         typedef std::float128_t Real;
//         typedef std::float128_t NonInteger;
//         typedef std::float128_t Nested;
//         enum {
//             IsComplex = 0,
//             IsInteger = 0,
//             IsSigned = 1,
//             RequireInitialization = 1,
//             ReadCost = 1,
//             AddCost = 1,
//             MulCost = 3
//         };
//         static inline Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
//         static inline Real dummy_precision() { return static_cast<Real>(1e-30); }
//         static inline Real highest() { return std::numeric_limits<Real>::max(); }
//         static inline Real lowest() { return std::numeric_limits<Real>::lowest(); }
//         static inline Real infinity() { return std::numeric_limits<Real>::infinity(); }
//     };
// }

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::get_rNorms_log10_change_per_iteration() {
    if(status.rNorms_history.size() < 2ul) return RealScalar{0};
    // If the residual norm is decreasing, this function returns a negative value, otherwise positive
    auto rNorm_change = status.rNorms_history.back().array() / status.rNorms_history.front().array();
    return std::log10(rNorm_change.minCoeff()) / static_cast<RealScalar>(status.rNorms_history.size());
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::get_rNorms_log10_change_per_matvec() {
    if(status.rNorms_history.size() < 2ul) return RealScalar{0};
    // If the residual norm is decreasing, this function returns a negative value, otherwise positive
    auto size = status.rNorms_history.size();
    assert(size == status.matvecs_history.size());
    auto rNorm_change = status.rNorms_history[size - 1].array() / status.rNorms_history[size - 2].array();
    auto sum_matvecs  = status.matvecs_history[size - 1] + status.matvecs_history[size - 2];
    // auto sum_matvecs  = std::accumulate(status.matvecs_history.begin(), status.matvecs_history.end(), Eigen::Index{0});
    return std::log10(rNorm_change.minCoeff()) / static_cast<RealScalar>(sum_matvecs);
}

template<typename Scalar>
typename solver_base<Scalar>::VectorReal solver_base<Scalar>::get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10) {
    if(v.empty()) return {};
    auto       cols   = static_cast<Eigen::Index>(v.size());
    auto       rows   = static_cast<Eigen::Index>(v.front().size());
    MatrixReal matrix = MatrixReal::Zero(rows, cols);
    for(size_t idx = 0; idx < v.size(); ++idx) {
        if(v[idx].size() < rows) { throw except::runtime_error("v has unequal size vectors"); }
        if(apply_log10)
            matrix.col(idx) = v[idx].topRows(rows).array().log10();
        else
            matrix.col(idx) = v[idx].topRows(rows).array();
    }
    VectorReal means  = matrix.rowwise().mean();
    VectorReal stddev = (((matrix.colwise() - means).array().square().rowwise().sum()) / static_cast<RealScalar>((matrix.cols() - 1))).sqrt();
    return stddev;
}

template<typename Scalar>
typename solver_base<Scalar>::VectorReal solver_base<Scalar>::get_slopes(const std::deque<VectorReal> &v, bool apply_log10) {
    // Least-squares slope for equally spaced t = 0..m-1, per column:
    auto get_slope = [](const VectorReal &x, const VectorReal &y) -> RealScalar {
        assert(x.size() == y.size());
        assert(x.size() >= 2);
        auto xmean = x.mean();
        auto ymean = y.mean();
        auto sxy   = (x.array() - xmean).matrix().dot((y.array() - ymean).matrix());
        auto sxx   = (x.array() - xmean).matrix().dot((x.array() - xmean).matrix());
        return sxy / sxx;
    };
    if(v.empty()) return {};

    auto       m = static_cast<Eigen::Index>(v.size());
    auto       n = static_cast<Eigen::Index>(v.front().size());
    VectorReal x = VectorReal::LinSpaced(m, RealScalar(0), RealScalar(m - 1));
    VectorReal slopes(n);
    for(Eigen::Index j = 0; j < n; ++j) {
        VectorReal y(m);
        for(Eigen::Index i = 0; i < m; ++i) {
            const VectorReal &eigVals_i = v.at(i);
            assert(eigVals_i.size() == n);
            y(i) = eigVals_i[j];
        }
        if(apply_log10)
            slopes(j) = get_slope(x, y.array().log10());
        else
            slopes(j) = get_slope(x, y);
    }

    return slopes;
}

template<typename Scalar>
bool solver_base<Scalar>::rNorms_have_saturated() {
    // Check if there is less than 1% fluctuation in the (order of magnitude of) latest residual norms.
    Eigen::Index min_history_size = std::min<Eigen::Index>(status.max_history_size, 2);
    if(status.iter < min_history_size) return false;
    if(status.rNorms_history.size() < static_cast<size_t>(min_history_size)) return false;

    VectorReal &vals           = status.rNorms;
    VectorReal  stds           = get_standard_deviations(status.rNorms_history, false);
    VectorIdxT  stds_saturated = (stds.array() < vals.array()).template cast<Eigen::Index>(); // Saturated if the fluctuations are smaller than the value itself
    // eiglog->info("rNorm stds {::.5e} {}", fv(stds), stds_saturated);

    return stds_saturated.all();
}

template<typename Scalar>
bool solver_base<Scalar>::eigVals_have_saturated() {
    // Check if there is less than 1% fluctuation in the latest eigVals.
    Eigen::Index min_history_size = std::min<Eigen::Index>(status.max_history_size, 2);
    if(status.iter < min_history_size) return false;
    if(status.eigVals_history.size() < static_cast<size_t>(min_history_size)) return false;
    VectorReal vals             = status.eigVal.cwiseAbs().array() + eps;
    VectorReal stds             = get_standard_deviations(status.eigVals_history, false);
    VectorReal rels             = stds.cwiseQuotient(vals);
    VectorIdxT stds_saturated   = (stds.array() < RealScalar{1e-2f}).template cast<Eigen::Index>();
    VectorIdxT rels_saturated   = (rels.array() < RealScalar{1e-5f}).template cast<Eigen::Index>();
    VectorReal slopes           = get_slopes(status.eigVals_history, false);
    VectorIdxT slopes_saturated = (slopes.cwiseAbs().array() < RealScalar{1e-2f}).template cast<Eigen::Index>();

    // eiglog->info("eigVal stds {::.5e} {} rels {::.5e} {} slopes {::.5e} {}", fv(stds), stds_saturated, fv(rels), rels_saturated, fv(slopes),
    // slopes_saturated);
    return stds_saturated.all() or rels_saturated.all() or slopes_saturated.all();
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_tolerance([[maybe_unused]] const Eigen::Ref<const MatrixType> &S) {
    // if(status.iter_last_preconditioner_tolerance_adjustment == status.iter) return;
    H1.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.T1_max_eval), std::abs(status.T1_min_eval)) / std::min(std::abs(status.T1_max_eval), std::abs(status.T1_min_eval));
    H2.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.T2_max_eval), std::abs(status.T2_min_eval)) / std::min(std::abs(status.T2_max_eval), std::abs(status.T2_min_eval));
    H1H2.get_iterativeLinearSolverConfig().jacobi.cond = status.condition;

    if(!use_adaptive_inner_tolerance) return;
    // auto Snorm = S.leftCols(nev).colwise().norm().minCoeff();

    auto set_cfg = [&](IterativeLinearSolverConfig<Scalar> &cfg) {
        auto oldtol = std::max(eps, cfg.tolerance);
        auto oldits = status.num_iters_inner_prev;

        cfg.tolerance = oldtol; // std::min<RealScalar>({oldtol, std::sqrt(Snorm)});
        if(status.iter > 0) {
            if(oldits < 100l) cfg.tolerance *= std::sqrt(half);
            if(oldits > cfg.maxiters / 2) cfg.tolerance *= std::sqrt(RealScalar{2});
        }

        cfg.tolerance = std::clamp(cfg.tolerance, eps, RealScalar{0.75f});
        // RealScalar maxiters = RealScalar{50l} / cfg.tolerance;
        cfg.maxiters = 2000l; // std::clamp(safe_cast<long>(maxiters), 50l, 200l);

        // RealScalar tol_rnorm = std::pow(Snorm, RealScalar{0.382f});
        RealScalar tol_rnorm = RealScalar{1e-4f}; // std::pow(Snorm, RealScalar{0.5f});
        // RealScalar tol_old   = cfg.tolerance;
        // RealScalar tol_min = RealScalar{0.1f}; // std::sqrt(eps);
        // RealScalar tol_max = RealScalar{0.1f};
        RealScalar tol_min = RealScalar{1e-20f}; // std::sqrt(eps);
        RealScalar tol_max = RealScalar{1e-1f};

        // if(status.iter > 0) {
        //     if(oldits < 50) cfg.tolerance = std::min(cfg.tolerance, oldtol * half);
        //     if(oldits > cfg.maxiters / 2) cfg.tolerance *= RealScalar{2};
        // }
        cfg.tolerance = std::clamp(tol_rnorm, tol_min, tol_max);

        // cfg.tolerance = RealScalar{1e-2f};
        // eiglog->info("tol {:.2e} maxit {} oldtol {:.2e} oldits {}", fp(cfg.tolerance), cfg.maxiters, fp(oldtol), oldits);
    };

    set_cfg(H1.get_iterativeLinearSolverConfig());
    set_cfg(H2.get_iterativeLinearSolverConfig());
    set_cfg(H1H2.get_iterativeLinearSolverConfig());

    status.iter_last_preconditioner_tolerance_adjustment = status.iter;
    // eiglog->info("max iters H1 {} | H2 {} | H1H2 {}", H1.get_iterativeLinearSolverConfig().maxiters, H2.get_iterativeLinearSolverConfig().maxiters,
    // H1H2.get_iterativeLinearSolverConfig().maxiters);
    return;
    auto rNorm_log10_decrease = get_rNorms_log10_change_per_iteration();
    if(rNorm_log10_decrease == RealScalar{0}) return;
    if(rNorm_log10_decrease > RealScalar{-0.9f}) {
        // Decreasing less than a quarter of an order of magnitude per iteration,
        // We could spend more time in the inner solver, so we tighten the tolerance
        H1.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{0.5f};
        H2.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{0.5f};
        H1H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
    }

    if(rNorm_log10_decrease < RealScalar{-3.0f}) {
        // Decreasing more than two orders of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{5};
        H2.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{5};
        H1H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
    } else if(rNorm_log10_decrease < RealScalar{-2.1f}) {
        // Decreasing more than one order of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{2};
        H2.get_iterativeLinearSolverConfig().tolerance   *= RealScalar{2};
        H1H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{2};
    }
    /* clang-format off */
    H1.get_iterativeLinearSolverConfig().tolerance = std::clamp<RealScalar>(H1.get_iterativeLinearSolverConfig().tolerance, RealScalar{5e-12f}, RealScalar{0.25f});
    H2.get_iterativeLinearSolverConfig().tolerance = std::clamp<RealScalar>(H2.get_iterativeLinearSolverConfig().tolerance, RealScalar{5e-12f}, RealScalar{0.25f});
    H1H2.get_iterativeLinearSolverConfig().tolerance = std::clamp<RealScalar>(H1H2.get_iterativeLinearSolverConfig().tolerance, RealScalar{5e-12f}, RealScalar{0.25f});
    /* clang-format on */
    status.iter_last_preconditioner_tolerance_adjustment = status.iter;
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_H1_limits() {
    if(status.iter_last_preconditioner_H1_limit_adjustment == status.iter) return;
    H1.get_iterativeLinearSolverConfig().precondType = PreconditionerType::JACOBI;
    if(H1.get_iterativeLinearSolverConfig().precondType == PreconditionerType::CHEBYSHEV) {
        RealScalar lambda_min                                     = status.T1_min_eval * RealScalar{0.9f};
        RealScalar lambda_max                                     = status.T1_max_eval * RealScalar{1.1f};
        H1.get_iterativeLinearSolverConfig().chebyshev.lambda_min = lambda_min;
        H1.get_iterativeLinearSolverConfig().chebyshev.lambda_max = lambda_max;
        H1.get_iterativeLinearSolverConfig().chebyshev.degree     = 5;
    }
    status.iter_last_preconditioner_H1_limit_adjustment = status.iter;
}

template<typename Scalar>
void solver_base<Scalar>::adjust_residual_correction_type() {
    auto mintol                       = std::min(H1.get_iterativeLinearSolverConfig().tolerance, H2.get_iterativeLinearSolverConfig().tolerance);
    residual_correction_type_internal = residual_correction_type;
    if(residual_correction_type_internal == ResidualCorrectionType::AUTO) {
        residual_correction_type_internal = ResidualCorrectionType::NONE;
        if(mintol < RealScalar{1e-1f} or status.num_matvecs_inner > 300) { residual_correction_type_internal = ResidualCorrectionType::CHEAP_OLSEN; }
        if(mintol < RealScalar{1e-3f} or status.num_matvecs_inner > 1000) { residual_correction_type_internal = ResidualCorrectionType::FULL_OLSEN; }
        if(mintol < RealScalar{1e-5f} or status.num_matvecs_inner > 2000) { residual_correction_type_internal = ResidualCorrectionType::JACOBI_DAVIDSON; }
    }
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_H2_limits() {
    if(status.iter_last_preconditioner_H2_limit_adjustment == status.iter) return;
    H2.get_iterativeLinearSolverConfig().precondType   = PreconditionerType::JACOBI;
    H1H2.get_iterativeLinearSolverConfig().precondType = PreconditionerType::JACOBI;

    if(H2.get_iterativeLinearSolverConfig().precondType == PreconditionerType::CHEBYSHEV) {
        RealScalar lambda_min                                     = RealScalar{0}; // status.H2_min_eval * RealScalar{0.9f};
        RealScalar lambda_max                                     = status.T2_max_eval * RealScalar{1.01f};
        H2.get_iterativeLinearSolverConfig().chebyshev.lambda_min = lambda_min;
        H2.get_iterativeLinearSolverConfig().chebyshev.lambda_max = lambda_max;
        H2.get_iterativeLinearSolverConfig().chebyshev.degree     = 2;
    }
    status.iter_last_preconditioner_H2_limit_adjustment = status.iter;
}

template<typename Scalar>
void solver_base<Scalar>::save_preconditioner_stats(IterativeLinearSolverConfig<Scalar> &cfg) {
    auto &res                    = cfg.result;
    status.num_iters_inner      += res.iters;
    status.num_matvecs_inner    += res.matvecs;
    status.num_precond_inner    += res.precond;
    status.time_matvecs_inner   += res.time_matvecs;
    status.time_precond_inner   += res.time_precond;
    status.time_jacobi_inner    += res.time_jacobi;
    status.time_chebyshev_inner += res.time_chebyshev;
    status.inner_error_last      = std::max(status.inner_error_last, res.error);
    status.inner_tol_last        = std::max(status.inner_tol_last, cfg.tolerance);
    res.reset();
}

template<typename Scalar>
void solver_base<Scalar>::save_jd_stats(const IterativeLinearSolverConfig<Scalar> &cfg) {
    auto &res                    = cfg.result;
    status.num_iters_inner      += res.iters;
    status.num_precond_inner    += res.precond;
    status.num_jdops_inner      += res.matvecs;
    status.time_precond_inner   += res.time_precond;
    status.time_jdops_inner     += res.time_matvecs;
    status.time_jacobi_inner    += res.time_jacobi;
    status.time_chebyshev_inner += res.time_chebyshev;
    status.inner_error_last      = std::max(status.inner_error_last, res.error);
    status.inner_tol_last        = std::max(status.inner_tol_last, cfg.tolerance);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH(const Eigen::Ref<const MatrixType> &X) {
    auto       token_matvecs = status.time_matvecs.tic_token();
    MatrixType HX;
    switch(algo) {
        case OptAlgo::DMRG:
            HX                  = H1.MultAX(X);
            status.num_matvecs += X.cols();
            break;
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: {
            MatrixType H2X      = H2.MultAX(X);
            MatrixType H1X      = H1.MultAX(X);
            HX                  = H2X - H1.MultAX(H1X);
            status.num_matvecs += 3 * X.cols(); // two more matvecs
            break;
        }
        case OptAlgo::XDMRG:
            HX                  = H2.MultAX(X);
            status.num_matvecs += X.cols();
            break;
        case OptAlgo::GDMRG: throw except::runtime_error("MultH: GDMRG is not suitable, use MultH1X or MultH2X instead");
        default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
    }
    return HX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH1(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH1: should only be called by GDMRG");
    auto token_matvecs  = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H1.MultAX(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH2(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH2: should only be called by GDMRG");
    auto token_matvecs  = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H2.MultAX(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH2_hp(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH2_hp: should only be called by GDMRG");
    auto token_matvecs  = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H2.MultAX_hp(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP(const Eigen::Ref<const MatrixType>                  &X,
                                                                    [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                    std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    auto       token_precond = status.time_precond.tic_token();
    MatrixType HPX;
    switch(algo) {
        case OptAlgo::DMRG: {
            H1.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX                                               = H1.MultPX(X);
            break;
        }
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: {
            H2.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX                                               = H2.MultPX(X);
            break;
        }
        case OptAlgo::XDMRG: {
            H2.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX                                               = H2.MultPX(X);
            break;
        }
        case OptAlgo::GDMRG: throw except::runtime_error("MultPX: GDMRG is not suitable, use MultP1X or MultP2X instead");
        default: throw except::runtime_error("MultPX: unknown algorithm {}", enum2sv(algo));
    }
    save_preconditioner_stats(H1.get_iterativeLinearSolverConfig());
    save_preconditioner_stats(H2.get_iterativeLinearSolverConfig());
    return HPX;
}
template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP1(const Eigen::Ref<const MatrixType>                  &X,
                                                                     [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                     std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    auto token_precond                                   = status.time_precond.tic_token();
    H1.get_iterativeLinearSolverConfig().initialGuess    = initialGuess.value_or(MatrixType{});
    H1.get_iterativeLinearSolverConfig().jacobi.skipjcb  = dev_skipjcb;
    MatrixType HPX                                       = H1.MultPX(X);
    status.num_precond                                  += X.cols();
    save_preconditioner_stats(H1.get_iterativeLinearSolverConfig());
    return HPX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP2(const Eigen::Ref<const MatrixType>                  &X,
                                                                     [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                     std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP2: should only be called by GDMRG");
    auto token_precond = status.time_precond.tic_token();
    assert(X.allFinite());
    H2.get_iterativeLinearSolverConfig().initialGuess    = initialGuess.value_or(MatrixType{});
    H2.get_iterativeLinearSolverConfig().jacobi.skipjcb  = dev_skipjcb;
    MatrixType HPX                                       = H2.MultPX(X);
    status.num_precond                                  += X.cols();
    save_preconditioner_stats(H2.get_iterativeLinearSolverConfig());
    return HPX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP1P2(const Eigen::Ref<const MatrixType>                  &X,
                                                                       [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                       std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP1P2: should only be called by GDMRG");
    auto token_precond                                     = status.time_precond.tic_token();
    H1H2.get_iterativeLinearSolverConfig().initialGuess    = initialGuess.value_or(MatrixType{});
    H1H2.get_iterativeLinearSolverConfig().jacobi.skipjcb  = dev_skipjcb;
    MatrixType H1H2PX                                      = H1H2.MultPX(X, evals);
    auto      &H1H2ir                                      = H1H2.get_iterativeLinearSolverConfig().result;
    status.num_precond                                    += X.cols();
    save_preconditioner_stats(H1H2.get_iterativeLinearSolverConfig());
    H1H2ir.reset();
    return H1H2PX;
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_mBlock() {
    // M are the b next-best ritz vectors from the previous iteration
    if(use_extra_ritz_vectors_in_the_next_basis and T_evals.size() >= 2 * b) {
        auto top_2b_indices = get_ritz_indices(ritz, b, b, T_evals);
        auto Z              = T_evecs(Eigen::placeholders::all, top_2b_indices); // Selected subspace eigenvectors
        M                   = Q * Z;                                             // Regular Rayleigh-Ritz
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

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::cheap_Olsen_correction(const MatrixType &V, const MatrixType &S) {
    MatrixType D(S.rows(), S.cols());

    // Generate the cheap olsen correction (S - Δ*V),
    // Where Δ is a diagonal matrix that Δ𝑖,𝑖 holds an estimation of the error of
    // the approximate eigenvalue Λ𝑖,𝑖.
    assert(V.allFinite());
    assert(S.allFinite());
    for(long i = 0; i < S.cols(); ++i) {
        auto d           = D.col(i);
        auto v           = V.col(i);
        auto s           = S.col(i);
        auto numerator   = Scalar{1};
        auto denominator = Scalar{1};

        if(algo == OptAlgo::GDMRG) {
            // For generalized eigenvalue problems
            auto h2v    = H2V.col(i);
            numerator   = h2v.dot(s); // v^H * B * s
            denominator = h2v.dot(v); // v^H * B * v
        } else {
            // For standard eigenvalue problems
            numerator   = v.dot(s); // v^H * s
            denominator = v.dot(v); // v^H * v
        }

        auto delta  = std::abs(denominator) > eps * 100 ? numerator / denominator : RealScalar{0};
        d.noalias() = s - delta * v; // Gets preconditioned later
    }
    return D;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::full_Olsen_correction(const MatrixType &V, const MatrixType &S) {
    // Precondition V and S blockwise
    MatrixType MV;
    MatrixType MS;
    MatrixType coeffs;
    auto       Y = T_evals(status.optIdx);

    if(algo == OptAlgo::GDMRG and use_h2_inner_product) {
        MV.noalias() = use_preconditioner ? MultP2(V, Y, std::nullopt) : V;
        MS.noalias() = use_preconditioner ? MultP2(S, Y, std::nullopt) : S;

        // Gram matrix in H2-inner product: G = V^H * B * MV  = ( (B*V).adjoint() * MV ) (b x b)

        // MatrixType B_MV = MultH2X(MV);
        // MatrixType G    = V.adjoint() * B_MV;

        // Coefficients: G^{-1} * (V^H * B * MS) = (B*V).adjoint() * MS
        // MatrixType H2_MS    = MultH2X(MS);
        // MatrixType VT_H2_MS = V.adjoint() * H2_MS;

        MatrixType G        = H2V.adjoint() * MV;
        MatrixType VT_H2_MS = H2V.adjoint() * MS;
        coeffs              = G.ldlt().solve(VT_H2_MS);
    } else {
        MV.noalias() = use_preconditioner ? MultP(V, Y, std::nullopt) : V;
        MS.noalias() = use_preconditioner ? MultP(S, Y, std::nullopt) : S;

        // Gram matrix in preconditioned metric: G = V^T * MV (b x b)
        MatrixType G = V.adjoint() * MV; // symmetric if M is HPD

        // Projection coefficients: C = G^{-1} * (V^T * MS)
        MatrixType VT_MS = V.adjoint() * MS;
        coeffs           = G.ldlt().solve(VT_MS); // robust inversion
    }
    // Olsen correction
    return MS - MV * coeffs; // N x b
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S,
                                                                                                                      const VectorReal &evals) {
    assert(V.rows() == S.rows());
    assert(V.cols() == S.cols());
    assert(!use_h2_inner_product);

    // Apply the jacobi davidson correction equation on the projected residual S
    //     ProjectOpL * ResidualOp * ProjectOpR = -RHS = -ProjectOpL(S)
    // where in the generalized problem:
    //      ProjectOpL(X) = (X - V * V.adjoint() * X),
    //      ResidualOp(X) = (H1*X - Theta * H2*X)
    //      ProjectOpR(X) = (X - V * V.adjoint() * X) = ProjectOpL(X),
    // while in the standard problem we set the left hands id operator H = H1 or H2, so
    //      ResidualOp = (H - Theta * I)
    // Note that in practice we treat this in a column-by column way, not the whole block S.
    // Note also that the inner MINRES solver is preconditioned with jacobi blocks Minv.
    // To avoid leakage  we project those blocks too, as ProjectOpR * Minv * ProjectOpL * X.
    // Notice that the projectors are reversed!

    // Define the matrix-vector operator for the H2 operator
    auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        status.num_matvecs_inner += X.cols();
        if(algo == OptAlgo::DMRG)
            return H1.MultAX(X);
        else
            return H2.MultAX(X);
    };

    auto ProjectOpL = [&V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
        auto                    t_pl = tid::tic_token("ProjectOpL", tid::level::higher);
        thread_local MatrixType T;
        T.resize(V.cols(), X.cols());
        Y.resize(X.rows(), X.cols());
        T.noalias()  = V.adjoint() * X;
        Y.noalias()  = X;
        Y.noalias() -= V * T;
    };
    auto ProjectOpL_tmp = [ProjectOpL](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        MatrixType Y(X.rows(), X.cols());
        ProjectOpL(X, Y);
        return Y;
    };

    auto ProjectOpR = [&V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
        auto                    t_pr = tid::tic_token("ProjectOpR", tid::level::higher);
        thread_local MatrixType T;
        T.resize(V.cols(), X.cols());
        Y.resize(X.rows(), X.cols());
        T.noalias()  = V.adjoint() * X;
        Y.noalias()  = X;
        Y.noalias() -= V * T;
    };
    auto ProjectOpR_tmp = [ProjectOpR](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        MatrixType Y(X.rows(), X.cols());
        ProjectOpR(X, Y);
        return Y;
    };

    auto t_jdl2 = tid::tic_scope("jdl2");

    // Right-hand side (projected)
    MatrixType RHS = -ProjectOpL_tmp(S);

    if(D.size() != RHS.size()) D.setZero(RHS.rows(), RHS.cols());

    for(Eigen::Index i = 0; i < S.cols(); ++i) { // We use block size b
        // v: current Ritz vector, s: current residual, Bv: either I*v (Euclidean) or H2*v (H2-orthonormal)
        auto d = D.col(i); // The solution vector i
        // auto              x0  = X0.col(i);  // The solution vector i
        RealScalar        th  = evals(i);   // The ritz value for this ritz vector
        const VectorType &rhs = RHS.col(i); // Right-hand side (projected)

        if(use_shifted_jd_eigenvalue) {
            // Find the index k that minimizes |T_evals(k) - th|. We expect T_evals(k) == th.
            Eigen::Index k      = 0;
            RealScalar   mingap = std::abs(T_evals(k) - th);
            for(Eigen::Index j = 0; j < T_evals.size(); ++j) {
                RealScalar gap = std::abs(T_evals(j) - th);
                if(gap < mingap) {
                    k      = j;
                    mingap = gap;
                }
            }

            auto kappa_small = [&](RealScalar tau) {
                RealScalar alpha = std::numeric_limits<RealScalar>::infinity();
                RealScalar beta  = RealScalar(0);
                for(Eigen::Index j = 0; j < T_evals.size(); ++j) {
                    if(j == k) continue;
                    RealScalar diff = std::abs(T_evals(j) - tau);
                    beta            = std::max(beta, diff);
                    alpha           = std::min(alpha, diff);
                }
                // guard
                if(!(alpha > RealScalar(0))) alpha = RealScalar(1);
                // eiglog->debug("alpha={:.3e}", "beta={:.3e}", alpha,beta);
                return beta / alpha;
            };

            // Build exvals = T_evals \ {th}
            std::vector<RealScalar> exvals;
            exvals.reserve(T_evals.size() - 1);
            exvals.assign(T_evals.data(), T_evals.data() + T_evals.size());

            exvals.erase(exvals.begin() + k); // Remove the closest (k points to the active theta)

            // Guard tiny/degenerate cases
            RealScalar tau = th;
            if(!exvals.empty()) {
                // nearest-neighbor gap g and hull center c
                RealScalar g    = std::numeric_limits<RealScalar>::infinity();
                RealScalar smin = exvals.front(), smax = exvals.front();
                for(const auto &mu : exvals) {
                    smin = std::min(smin, mu);
                    smax = std::max(smax, mu);
                    g    = std::min(g, std::abs(mu - th));
                }
                if(!std::isfinite(g) || g == RealScalar(0)) g = RealScalar(1); // fallback
                const RealScalar c = RealScalar(0.5) * (smin + smax);

                // blend + clip
                constexpr RealScalar eta   = RealScalar(0.5); // pull halfway toward c
                constexpr RealScalar alpha = RealScalar(0.5); // stay within 0.5 * gap
                RealScalar           delta = eta * (c - th);
                const RealScalar     bound = alpha * g;
                if(delta > bound) delta = bound;
                if(delta < -bound) delta = -bound;

                tau = th + delta;
            }
            // Before/after
            const RealScalar kappa_at_theta = kappa_small(th);
            const RealScalar kappa_at_tau   = kappa_small(tau);
            // eiglog->info("T_evals: {::.3e}, k = {}", fv(T_evals), k);
            eiglog->trace("[JD] kappa_small(theta)={:.3e}, kappa_small(tau)={:.3e}, theta={:.6e}, tau={:.6e}", fp(kappa_at_theta), fp(kappa_at_tau), fp(th),
                          fp(tau));
            th = tau;
        }

        if(i > 0) {
            // This residual is not in the "active" set. Default to Cheap Olsen + CG instead
            const VectorType &s  = S.col(i); // The residual vector i
            const VectorType &v  = V.col(i); // The residual vector i
            auto              ev = evals.middleRows(i, 1);
            D.col(i).noalias()   = algo == OptAlgo::DMRG ? MultP1(s, ev) : (use_h1h2_jcb_preconditioner ? MultP1P2(s, ev) : MultP2(s, ev));
            D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
        } else {
            auto  token_precond = status.time_precond.tic_token();
            auto &H             = algo == OptAlgo::DMRG ? H1 : (use_h1h2_jcb_preconditioner ? H1H2 : H2);

            auto t_calc = tid::tic_scope("CalcPc");
            H.CalcPc(th); // Compute the block-jacobi preconditioner (do llt/ldlt on all blocks)
            t_calc.toc();

            IterativeLinearSolverConfig<Scalar> cfg = H.get_iterativeLinearSolverConfig(); // Get the jacobi blocks
            cfg.result                              = {};
            cfg.matdef                              = MatDef::IND; // IND (MINRES) is often faster than DEF (CG) with H2 (which is PSD)
            cfg.precondType                         = PreconditionerType::JACOBI;
            cfg.jacobi.skipjcb                      = dev_skipjcb;

            // Define the residual matrix-vector operator depending on the different DMRG algorithms
            auto ResidualOp = [this, th, &H](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                auto t_rop = tid::tic_scope("ResidualOp", tid::level::higher);
                auto t_mvi = status.time_matvecs_inner.tic_token();
                HX.resize(X.rows(), X.cols());
                switch(algo) {
                    case OptAlgo::DMRG: [[fallthrough]];
                    case OptAlgo::DMRGX: [[fallthrough]];
                    case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
                    case OptAlgo::XDMRG: {
                        status.num_matvecs_inner += X.cols();
                        HX.noalias()              = H.MultAX(X) - th * X;
                        break;
                    }
                    case OptAlgo::GDMRG: {
                        // Generalized problem
                        if(use_jd_h2_only) {
                            HX.noalias()              = H2.MultAX(X);
                            status.num_matvecs_inner += 1 * X.cols();
                            auto t_h2                 = tid::tic_token("H2X", tid::higher, H2.t_multAx->get_last_interval());
                        } else {
                            HX.noalias()              = H1.MultAX(X) - th * H2.MultAX(X);
                            status.num_matvecs_inner += 2 * X.cols();
                            auto t_h1                 = tid::tic_token("H1X", tid::higher, H1.t_multAx->get_last_interval());
                            auto t_h2                 = tid::tic_token("H2X", tid::higher, H2.t_multAx->get_last_interval());
                        }

                        break;
                    }
                    default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
                }
            };
            if(use_jd_initial_guess) cfg.initialGuess = d;

            auto JDop = JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR, MatrixOp);

            d.noalias() = JacobiDavidsonSolver(JDop, rhs, cfg);
            d.noalias() = ProjectOpR_tmp(d);
            save_jd_stats(cfg);
            cfg.result.reset();
        }
    }
    status.num_precond += b; // This routine is a preconditioner
    // D_prec = D;              // Store the result so we can use it in the next iteration.
    return D; // N x b, enrichment directions
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType
    solver_base<Scalar>::jacobi_davidson_h2_correction(const MatrixType &V, const MatrixType &H2V, const MatrixType &S, const VectorReal &evals) {
    assert(algo == OptAlgo::GDMRG);
    assert(use_h2_inner_product);
    assert(V.rows() == S.rows());
    assert(V.cols() == S.cols());
    assert(H1V.size() == V.size());
    assert(H2V.size() == V.size());

    // Define the residual S as:
    //      S = H1V-H2V*Theta.asDiagonal():
    // Apply the jacobi davidson correction equation on the projected residual S
    //     ProjectOpL * ResidualOp * ProjectOpR = -RHS = -ProjectOpL(S)
    // where in the generalized problem:
    //      ProjectOpL(X) = (X - H2 * V * V.adjoint() * X) = (X - H2V * (V.adjoint()*X)),
    //      ResidualOp(X) = (H1*X - Theta * H2*X)
    //      ProjectOpR(X) = (X - V * V.adjoint() * H2 * X) = (X - V * (H2V.adjoint() * X)),
    // where in the last line, the last equality is used to avoid a matvec with H2.
    // Note that in practice we treat this in a column-by column way, not the whole block S.
    // Note also that the inner MINRES solver is preconditioned with jacobi blocks Minv.
    // To avoid leakage  we project those blocks too, as ProjectOpR * Minv * ProjectOpL * X.
    // Notice that the projectors are reversed!

    auto ProjectOpL = [&V, &H2V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
        auto                    t_pl = tid::tic_token("ProjectOpL", tid::level::higher);
        thread_local MatrixType T;
        T.resize(V.cols(), X.cols());
        Y.resize(X.rows(), X.cols());
        T.noalias()  = V.adjoint() * X;
        Y.noalias()  = X;
        Y.noalias() -= H2V * T;
    };
    auto ProjectOpL_tmp = [ProjectOpL](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        MatrixType Y(X.rows(), X.cols());
        ProjectOpL(X, Y);
        return Y;
    };

    auto ProjectOpR = [&V, &H2V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
        auto                    t_pr = tid::tic_token("ProjectOpR", tid::level::higher);
        thread_local MatrixType T;
        T.resize(H2V.cols(), X.cols());
        Y.resize(X.rows(), X.cols());
        T.noalias()  = H2V.adjoint() * X;
        Y.noalias()  = X;
        Y.noalias() -= V * T;
    };
    auto ProjectOpR_tmp = [ProjectOpR](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        MatrixType Y(X.rows(), X.cols());
        ProjectOpR(X, Y);
        return Y;
    };
    auto t_jdh2 = tid::tic_scope("jdh2");

    // Right-hand side (projected)
    MatrixType RHS = -ProjectOpL_tmp(S);

    MatrixType D(S.rows(), S.cols()); // accumulate the result from the JD correction equation

    for(Eigen::Index i = 0; i < S.cols(); ++i) {
        auto              d   = D.col(i);   // The solution vector i
        const RealScalar &th  = evals(i);   // The ritz value for this ritz vector
        const VectorType &rhs = RHS.col(i); // Right-hand side (projected residual)
        if(i >= nev) {
            // This residual is not in the "active" set. Default to Cheap Olsen + CG instead
            const VectorType &s  = S.col(i); // The residual vector i
            const VectorType &v  = V.col(i); // The residual vector i
            auto              ev = evals.middleRows(i, 1);
            D.col(i).noalias()   = algo == OptAlgo::DMRG ? MultP1(s, ev) : (use_h1h2_jcb_preconditioner ? MultP1P2(s, ev) : MultP2(s, ev));
            D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
        } else {
            auto  token_precond = status.time_precond.tic_token();
            auto &H             = use_h1h2_jcb_preconditioner ? H1H2 : H2; // Typically use_h1h2_preconditioner is false

            auto t_calc = tid::tic_scope("calc");
            H.CalcPc(th); // Compute the block-jacobi preconditioner
            t_calc.toc();

            IterativeLinearSolverConfig<Scalar> cfg = H.get_iterativeLinearSolverConfig(); // Get the jacobi blocks
            cfg.result                              = {};
            cfg.matdef                              = use_jd_def_solver ? MatDef::DEF : MatDef::IND;
            cfg.precondType                         = PreconditionerType::JACOBI;
            cfg.jacobi.skipjcb                      = dev_skipjcb;
            // Define the matrix-vector operator for the H2 operator
            auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                auto t_mvi                = status.time_matvecs_inner.tic_token();
                status.num_matvecs_inner += X.cols();
                return H2.MultAX(X);
            };

            // Define the residual matrix-vector operator depending on the different DMRG algorithms
            auto ResidualOp_noalloc = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                auto t_rop = tid::tic_token("ResidualOp", tid::level::higher);
                auto t_mvi = status.time_matvecs_inner.tic_token();
                // Generalized problem
                HX.resize(X.rows(), X.cols());
                if(use_jd_h2_only) {
                    HX.noalias()              = H2.MultAX(X);
                    status.num_matvecs_inner += 1 * X.cols();
                    auto t_h2                 = tid::tic_token("H2X", tid::higher, H2.t_multAx->get_last_interval());
                } else {
                    HX.noalias()              = H1.MultAX(X) - th * H2.MultAX(X);
                    status.num_matvecs_inner += 2 * X.cols();
                    auto t_h1                 = tid::tic_token("H1X", tid::higher, H1.t_multAx->get_last_interval());
                    auto t_h2                 = tid::tic_token("H2X", tid::higher, H2.t_multAx->get_last_interval());
                }
            };

            // auto Kop = [&](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return ProjectOpL(ResidualOp(ProjectOpR(X))); };

            auto JDop = JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp_noalloc, ProjectOpL, ProjectOpR, MatrixOp);

            d.noalias() = JacobiDavidsonSolver(JDop, rhs, cfg);
            d.noalias() = ProjectOpR_tmp(d);
            save_jd_stats(cfg);
            cfg.result.reset();
        }
    }
    status.num_precond += b; // This routine is a preconditioner
    return D;                // N x b, enrichment directions
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_sBlock(const MatrixType &S_in, fMultP_t MultP) {
    // Make a residual block "S = (HQ-λQ)"
    MatrixType S = S_in;
    assert(S.allFinite());
    assert(S.cols() > 0);
    auto Y = T_evals(status.optIdx);

    if(chebyshev_filter_degree >= 1) S = qr_and_chebyshevFilter(S);
    switch(residual_correction_type_internal) {
        case ResidualCorrectionType::NONE:
            if(use_preconditioner) { S = MultP(S, Y, std::nullopt); }
            break;
        case ResidualCorrectionType::AUTO: [[fallthrough]];
        case ResidualCorrectionType::CHEAP_OLSEN:
            if(use_preconditioner) { S = MultP(S, Y, std::nullopt); }
            S.noalias() = cheap_Olsen_correction(V, S);
            break;
        case ResidualCorrectionType::FULL_OLSEN:
            // This has an internal preconditioner
            S.noalias() = full_Olsen_correction(V, S);
            break;
        case ResidualCorrectionType::JACOBI_DAVIDSON:
            // This is an internal preconditioner
            assert(use_preconditioner && " Jacobi Davidson correction needs use_preconditioner == true");
            if(algo == OptAlgo::GDMRG && use_h2_inner_product) {
                S.noalias() = jacobi_davidson_h2_correction(V, H2V, S, Y);
            } else {
                S.noalias() = jacobi_davidson_l2_correction(V, S, Y);
            }
            break;
    }
    assert_allFinite(S);
    return S;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_wBlock(fMultP_t MultP) {
    // We add Lanczos-style residual blocks
    W = (algo == OptAlgo::GDMRG) ? H2V : HV;
    A = V.adjoint() * W;

    // 3) Subtract projections to A and B once
    W.noalias() -= V * A; // Qi * Qi.adjoint()*H*Qi
    if(V_prev.rows() == N and V_prev.cols() == b) {
        B            = V_prev.adjoint() * W;
        W.noalias() -= V_prev * B.adjoint();
    }
    assert_allFinite(W);
    if(use_preconditioner) {
        auto       select_b = get_ritz_indices(ritz, 0, b, T_evals);
        VectorReal evals    = T_evals(select_b);
        W                   = MultP(W, evals, std::nullopt);
    }
    return W;
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_rBlock() {
    // Get a random block
    return Eigen::MatrixXf::Random(N, b).cast<Scalar>();
}

template<typename Scalar>
const typename solver_base<Scalar>::MatrixType &solver_base<Scalar>::get_HQ() {
    // HQ   = MultHX(Q);
    // return HQ;
    if(status.iter == i_HQ) {
        // assert((HQ - MultHX(Q)).norm() < 100 * eps);
        return HQ;
    }
    i_HQ = status.iter;
    HQ   = MultH(Q);
    return HQ;
}

template<typename Scalar>
const typename solver_base<Scalar>::MatrixType &solver_base<Scalar>::get_HQ_cur() {
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
    HQ_cur   = MultH(Q.middleCols((qBlocks - 1) * b, b));
    return HQ_cur;
}

template<typename Scalar>
void solver_base<Scalar>::unset_HQ() {
    i_HQ = -1;
}
template<typename Scalar>
void solver_base<Scalar>::unset_HQ_cur() {
    i_HQ_cur = -1;
    i_HQ     = -1;
}

template<typename Scalar>
void solver_base<Scalar>::mask_col_blocks(Eigen::Ref<MatrixType> Y, OrthMeta &m) {
    Eigen::Index n_blocks_y = Y.cols() / b;
    if(m.mask.size() != n_blocks_y) throw except::runtime_error("mask_col_blocks: mask size must match the number of blocks in Y");
    assert(m.Rdiag.size() == Y.cols());
    assert(m.mask.size() == n_blocks_y);
    for(Eigen::Index i = 0; i < n_blocks_y; ++i) {
        if(m.mask(i) == 0) continue;
        auto ri = m.Rdiag.middleRows(i * b, b);
        if(ri.minCoeff() < m.maskTol) {
            auto yi = Y.middleCols(i * b, b);
            yi.setZero();
            m.mask(i) = 0;
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::mask_cols(Eigen::Ref<MatrixType> Y, OrthMeta &m) {
    if(m.mask.size() != Y.cols()) throw except::runtime_error("mask_cols: mask size must match the number of columns in Y");

    assert(m.Rdiag.size() == Y.cols());
    assert(m.mask.size() == Y.cols());
    for(Eigen::Index i = 0; i < Y.cols(); ++i) {
        if(m.mask(i) == 0) continue;
        auto ri = m.Rdiag.row(i);
        if(ri.minCoeff() < m.maskTol) {
            auto yi = Y.col(i);
            yi.setZero();
            m.mask(i) = 0;
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::compress_col_blocks(MatrixType       &X,   // (N, ycols)
                                              const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == X.cols() / b && "Mask size must match number of blocks in X.");
    const Eigen::Index n_blocks_x = X.cols() / b;
    if(mask.sum() == n_blocks_x) return;

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
        X = X(Eigen::placeholders::all, active_columns).eval(); // Shrink keeping only nonzeros
    }
}

template<typename Scalar>
void solver_base<Scalar>::compress_cols(MatrixType       &X,   // (N, ycols)
                                        const VectorIdxT &mask // block norm mask, size = ycols
) {
    assert(mask.size() == X.cols() && "Mask size must match number of columns in X.");
    if(mask.sum() == X.cols()) return;

    // We can now squeeze out blocks zeroed out by DGKS
    // Get the block indices that we should keep
    std::vector<Eigen::Index> active_columns;
    active_columns.reserve(X.cols());
    for(Eigen::Index j = 0; j < X.cols(); ++j) {
        if(mask(j) == 1) { active_columns.push_back(j); }
    }
    active_columns.shrink_to_fit();
    X = X(Eigen::placeholders::all, active_columns).eval(); // Shrink keeping only nonzeros
}

template<typename Scalar>
void solver_base<Scalar>::compress_row_blocks(VectorReal       &X,   // (, ycols)
                                              const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    const Eigen::Index n_blocks_x = X.rows();
    assert(mask.size() == X.rows() && "Mask size must match number of rows in X.");
    if(mask.sum() == n_blocks_x) return;

    // We can now squeeze out blocks zeroed out by DGKS
    // Get the block indices that we should keep
    std::vector<Eigen::Index> active_rows;
    active_rows.reserve(n_blocks_x);
    for(Eigen::Index j = 0; j < n_blocks_x; ++j) {
        if(mask(j) == 1) { active_rows.push_back(j); }
    }
    active_rows.shrink_to_fit();
    if(active_rows.size() != static_cast<size_t>(X.rows())) {
        X = X(active_rows).eval(); // Shrink keeping only nonzeros
    }
}

template<typename Scalar>
void solver_base<Scalar>::compress_rows(VectorReal       &X,   // (, ycols)
                                        const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(mask.size() == X.rows() && "Mask size must match number of rows in X.");
    if(mask.sum() == X.rows()) return;

    // We can now squeeze out blocks zeroed out by DGKS
    // Get the block indices that we should keep
    std::vector<Eigen::Index> active_rows;
    active_rows.reserve(X.rows());
    for(Eigen::Index j = 0; j < X.rows(); ++j) {
        if(mask(j) == 1) { active_rows.push_back(j); }
    }
    active_rows.shrink_to_fit();
    if(active_rows.size() != static_cast<size_t>(X.rows())) {
        X = X(active_rows).eval(); // Shrink keeping only nonzeros
    }
}

template<typename Scalar>
void solver_base<Scalar>::compress_rows_and_cols(MatrixType       &X,   // (N, ycols)
                                                 const VectorIdxT &mask // block norm mask, size = n_blocks = ycols / blockWidth
) {
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == X.cols() / b && "Mask size must match number of blocks in X.");
    assert(mask.size() == X.rows() / b && "Mask size must match number of blocks in X.");
    const Eigen::Index n_blocks_x = X.cols() / b;
    if(mask.sum() == n_blocks_x) return;

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
    assert_allFinite(X);
}

// Right-unitary 2x2 rotation on columns i,j to equalize ‖Q.col(i)‖₂ and ‖Q.col(j)‖₂.
// Applies the same transform to H2Q so Q* remains H2-orthonormal in exact arithmetic.
template<typename Scalar>
void solver_base<Scalar>::balance_pair(Eigen::Ref<MatrixType> Y, Eigen::Ref<MatrixType> H2Y, Eigen::Index i, Eigen::Index j) {
    // Diagonals and cross term of the 2x2 Gram block (in L2, *not* H2)
    const RealScalar a     = Y.col(i).squaredNorm();
    const RealScalar b     = Y.col(j).squaredNorm();
    const Scalar     c     = Y.col(i).adjoint() * Y.col(j);
    const RealScalar abs_c = std::abs(c);

    // If already very close (or orthogonal with a <= b), nothing to do
    if(abs_c == RealScalar(0) && a <= b) return;

    // Compute unitary U that makes the two L2 column norms closer:
    //   tan(2θ) = (a - b) / (2|c|)
    const RealScalar two_theta = std::atan2(a - b, RealScalar(2) * std::max(abs_c, RealScalar(0)));
    const RealScalar ct        = std::cos(RealScalar(0.5) * two_theta);
    const RealScalar st        = std::sin(RealScalar(0.5) * two_theta);
    const Scalar     phase     = (abs_c > RealScalar(0)) ? c / Scalar(abs_c) : Scalar(1);

    // U = [ ct,           -phase*st
    //       conj(phase)*st,  ct      ]
    Scalar u00 = Scalar(ct);
    Scalar u01 = -phase * Scalar(st);
    Scalar u10 = -Eigen::numext::conj(u01); // <-- ensures u10 = -conj(u01)
    Scalar u11 = Scalar(ct);

    // Apply on the right to the (i,j) column pair for both Y and H2Y
    VectorType Yi = Y.col(i);
    VectorType Yj = Y.col(j);
    VectorType Hi = H2Y.col(i);
    VectorType Hj = H2Y.col(j);

    Y.col(i)   = Yi * u00 + Yj * u10;
    Y.col(j)   = Yi * u01 + Yj * u11;
    H2Y.col(i) = Hi * u00 + Hj * u10;
    H2Y.col(j) = Hi * u01 + Hj * u11;
}

// Sweep: pair largest with smallest L2-norm columns and balance them.
// - num_sweeps: how many global passes
// - max_pairs_per_sweep: limit pairs per pass (<= m/2). Use -1 for all pairs.
// - target_ratio: early-stop if (max_norm/min_norm) ≤ target_ratio
template<typename Scalar>
void solver_base<Scalar>::balance_columns_sweep(Eigen::Ref<MatrixType>                  Y,                   //
                                                Eigen::Ref<MatrixType>                  H2Y,                 //
                                                Eigen::Index                            num_sweeps,          //
                                                Eigen::Index                            max_pairs_per_sweep, //
                                                typename Eigen::NumTraits<Scalar>::Real target_ratio) {
    // #pragma message "Reenable balancing?"
    return;
    using Index = Eigen::Index;

    assert(Y.rows() == H2Y.rows());
    assert(Y.cols() == H2Y.cols());
    const Index m = Y.cols();
    if(m < 2 || num_sweeps <= 0) return;
    VectorReal y_norms   = Y.colwise().norm();
    VectorReal h2y_norms = H2Y.colwise().norm();
    for(int sweep = 0; sweep < num_sweeps; ++sweep) {
        // Compute current column norms
        VectorReal cn = Y.colwise().norm();

        // Early stop if already balanced enough
        RealScalar maxn = cn.maxCoeff();
        RealScalar minn = cn.minCoeff();
        if(minn > RealScalar(0) && maxn / minn <= target_ratio) break;

        // Order indices by norm ascending
        std::vector<Index> idx(m);
        std::iota(idx.begin(), idx.end(), Index(0));
        std::sort(idx.begin(), idx.end(), [&](Index i, Index j) { return cn(i) < cn(j); });

        // How many pairs this sweep?
        Index pairs = m / 2;
        if(max_pairs_per_sweep >= 0) pairs = std::min<Index>(pairs, max_pairs_per_sweep);

        for(Index k = 0; k < pairs; ++k) {
            Index i = idx[k];         // small
            Index j = idx[m - 1 - k]; // large
            if(i == j) break;

            balance_pair(Y, H2Y, i, j);
        }
    }
    VectorReal y_norms_new   = Y.colwise().norm();
    VectorReal h2y_norms_new = H2Y.colwise().norm();
    eiglog->info("norms   Y {::.4e} -> {::.4e}", fv(y_norms), fv(y_norms_new));
    eiglog->info("norms H2Y {::.4e} -> {::.4e}", fv(h2y_norms), fv(h2y_norms_new));
}

template<typename Scalar> void solver_base<Scalar>::assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;
        bool allFinite = X.allFinite();
        if(!allFinite) {
            eiglog->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            eiglog->warn("X is not all finite: \n{}\n", linalg::matrix::to_string(X, 8));
            throw except::runtime_error("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name());
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m, const std::source_location &location) {
    assert(!(use_h2_inner_product and algo == OptAlgo::GDMRG) && "assert_l2_orthonormal is for the L2 inner product");
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;

        MatrixType Gram      = X.adjoint() * X;
        RealScalar orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        RealScalar xnorm     = X.norm();
        RealScalar t_abs     = X.size() * eps * (xnorm + xnorm);
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : normTol * X.cols();
        RealScalar finalTol  = std::max({t_abs, normTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol) {
            eiglog->info("mask      = {} ", m.mask);
            eiglog->info("t_abs     = {} ", fp(t_abs));
            eiglog->info("normTol   = {} ", fp(normTol));
            eiglog->info("maskTol   = {}", fp(maskTol));
            eiglog->info("finalTol  = {} ", fp(finalTol));
            eiglog->info("orthError = {} ", fp(orthError));
            eiglog->info("gram matrix: \n{}", linalg::matrix::to_string(Gram, 16));

            eiglog->warn("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > tol = {:.5e}", location.file_name(), location.line(), location.function_name(),
                         fp(orthError), fp(finalTol));
            if(orthError > 1000 * finalTol) {
                throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > tol = {:.5e}", location.file_name(), location.line(),
                                            location.function_name(), fp(orthError), fp(finalTol));
            }
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m,
                                               const std::source_location &location) {
    assert(!(use_h2_inner_product and algo == OptAlgo::GDMRG) && "assert_l2_orthonormal is for the L2 inner product");
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0 || Y.cols() == 0) return;
        if(m.mask.size() > 0 and m.mask.sum() == 0) return;

        MatrixType Gram      = X.adjoint() * Y;
        RealScalar orthError = Gram.norm();
        RealScalar xnorm     = X.norm();
        RealScalar ynorm     = Y.norm();
        RealScalar t_abs     = X.size() * eps * (xnorm + ynorm);
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol * X.cols();
        RealScalar finalTol  = std::max({t_abs, orthTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol) {
            eiglog->info("mask      = {} ", m.mask);
            eiglog->info("t_abs     = {} ", fp(t_abs));
            eiglog->info("orthTol   = {} ", fp(orthTol));
            eiglog->info("maskTol   = {}", fp(maskTol));
            eiglog->info("finalTol  = {} ", fp(finalTol));
            eiglog->info("orthError = {} ", fp(orthError));
            eiglog->info("gram matrix: \n{}", linalg::matrix::to_string(Gram, 16));
            eiglog->warn("{}:{}: {}: matrices are not orthogonal: error = {:.5e} > tol = {:.5e}", location.file_name(), location.line(),
                         location.function_name(), fp(orthError), fp(finalTol));
            if(orthError > 1000 * finalTol)
                throw except::runtime_error("{}:{}: {}: matrices are not orthogonal: error = {:.5e} > tol = {:.5e}", location.file_name(), location.line(),
                                            location.function_name(), fp(orthError), fp(finalTol));
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::assert_h2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &H2Y, const OrthMeta &m,
                                               const std::source_location &location) {
    assert(use_h2_inner_product and algo == OptAlgo::GDMRG && "assert_h2_orthonormal is for the H2 inner product");
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0 || H2Y.cols() == 0) return;

        MatrixType Gram      = X.adjoint() * H2Y;
        auto       orthError = Gram.norm();
        RealScalar xnorm     = X.norm();
        RealScalar h2ynorm   = H2Y.norm();
        RealScalar h2norm    = std::isfinite(status.T2_max_eval) ? status.T2_max_eval : RealScalar{1};
        RealScalar t_abs     = orthTol * X.cols() * (xnorm + h2ynorm);
        RealScalar h2Tol     = orthTol * X.cols() * h2norm;
        RealScalar opTol     = orthTol * X.cols() * get_op_norm_estimate();
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol;

        RealScalar finalTol = std::max({t_abs, orthTol, opTol, h2Tol, maskTol}) * RealScalar{10};
        if(orthError > finalTol) {
            eiglog->info("mask      = {}", m.mask);
            eiglog->info("xnorm     = {}", fp(xnorm));
            eiglog->info("h2ynorm   = {}", fp(h2ynorm));
            eiglog->info("t_abs     = {}", fp(t_abs));
            eiglog->info("orthTol   = {}", fp(orthTol));
            eiglog->info("h2Tol     = {}", fp(h2Tol));
            eiglog->info("opTol     = {}", fp(opTol));
            eiglog->info("maskTol   = {}", fp(maskTol));
            eiglog->info("finalTol  = {}", fp(finalTol));
            eiglog->info("orthError = {}", fp(orthError));
            eiglog->info("gram matrix: \n{}", linalg::matrix::to_string(Gram, 16));
            eiglog->warn("{}:{}: {}: matrices are not orthogonal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                         location.function_name(), fp(orthError), fp(finalTol));
            // if(orthError > 1000 * finalTol) {
            // throw except::runtime_error("{}:{}: {}: matrices are not orthogormal: error = {:.5e} > threshold = {:.5e}", location.file_name(),
            // location.line(), location.function_name(), fp(orthError), fp(finalTol));
            // }
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::assert_h2_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &H2X, const OrthMeta &m,
                                                const std::source_location &location) {
    assert(use_h2_inner_product and algo == OptAlgo::GDMRG && "assert_h2_orthonormal is for the H2 inner product");
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;
        MatrixType G1        = X.adjoint() * H2X;
        MatrixType G2        = H2X.adjoint() * X;
        MatrixType Gram      = G1;
        MatrixType Gram_symm = (G1 + G2) * half;
        MatrixType Gram_skew = (G1 - G2) * half;
        MatrixType I         = MatrixType::Identity(Gram.rows(), Gram.cols());
        RealScalar orthError = (Gram - I).norm();
        RealScalar symmError = (Gram_symm - I).norm();
        RealScalar skewError = Gram_skew.norm();

        RealScalar xnorm   = X.norm();
        RealScalar h2xnorm = H2X.norm();

        Eigen::SelfAdjointEigenSolver<MatrixType> esG(Gram_symm);
        VectorReal                                evG_abs   = esG.eigenvalues().cwiseAbs();
        RealScalar                                evG_max   = evG_abs.maxCoeff();
        RealScalar                                evG_min   = evG_abs.minCoeff();
        RealScalar                                normG_max = std::sqrt(evG_max);
        // RealScalar                                normG_min = std::sqrt(evG_min);

        RealScalar c_abs     = X.size();
        RealScalar c_rel     = X.size();
        RealScalar t_abs     = c_abs * eps * (xnorm + h2xnorm);
        RealScalar t_rel     = c_rel * std::sqrt(eps) * normG_max;
        RealScalar kappaG    = evG_max / evG_min;
        RealScalar kappaGTol = 20 * eps * kappaG;
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        RealScalar finalTol  = std::max({t_abs, t_rel, orthTol, kappaGTol, maskTol}) * RealScalar{10};

        if(skewError > RealScalar{1e-2f}) {
            eiglog->warn("{}:{}: {}: Skew-symmetric gram matrix: skewError = {:.4e} (G1-G2)/2 = \n{}", location.file_name(), location.line(),
                         location.function_name(), fp(skewError), linalg::matrix::to_string(Gram_skew, 8));
            // throw except::runtime_error("{}:{}: {}: Skew-symmetric gram matrix: skewError = {:.4e} (G1-G2)/2 = \n{}", location.file_name(), location.line(),
            // location.function_name(), skewError, linalg::matrix::to_string(Gram_skew, 8));
        }
        if(symmError > finalTol) {
            eiglog->info("evG min   = {}", fp(evG_min));
            eiglog->info("evG max   = {}", fp(evG_max));
            eiglog->info("kappaG    = {} ", fp(kappaG));
            eiglog->info("xnorm     = {} ", fp(xnorm));
            eiglog->info("bxnorm    = {} ", fp(h2xnorm));
            eiglog->info("t_rel     = {} ", fp(t_rel));
            eiglog->info("t_abs     = {} ", fp(t_abs));
            eiglog->info("kappaGTol = {} ", fp(kappaGTol));
            eiglog->info("finalTol  = {} ", fp(finalTol));
            eiglog->info("orthTol   = {} ", fp(orthTol));
            eiglog->info("maskTol   = {} ", fp(maskTol));
            eiglog->info("orthError = {} ", fp(orthError));
            eiglog->info("gram matrix: \n{}", linalg::matrix::to_string(Gram, 16));
            eiglog->warn("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                         location.function_name(), fp(orthError), fp(finalTol));
            if(orthError > 1000 * finalTol) {
                throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                                            location.function_name(), fp(orthError), fp(finalTol));
            }
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) {
        H1Y.resizeLike(Y);
        H2Y.resizeLike(Y);
        return;
    }
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask = VectorIdxT::Ones(Y.cols());
    if(std::isnan(m.maskTol)) m.maskTol = normTol * Y.cols(); // * get_op_norm_estimate();

    auto handle_masked_columns = [&]() {
        if(m.mask.sum() != Y.cols()) {
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS: {
                    eiglog->warn("block_l2_orthonormalize: Compressing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    compress_cols(Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                }
                case MaskPolicy::RANDOMIZE: {
                    eiglog->warn("block_l2_orthonormalize: Randomizing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) { Y.col(j) = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>(); }
                    }
                    break;
                }
                default: throw except::runtime_error("Unrecognized mask policy");
            }
        }
    };

    // Initial mask
    m.Rdiag = VectorReal::Zero(Y.cols());
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto yj    = Y.col(j);
        m.Rdiag(j) = yj.norm();
        if(m.Rdiag(j) < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(m.Rdiag(j)), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
        }
    }
    // Compress or randomize
    handle_masked_columns();
    if(Y.cols() == 0) {
        H1Y.resizeLike(Y);
        H2Y.resizeLike(Y);
        return;
    }

    // Orthonormalize
    hhqr.compute(Y);
    Y       = hhqr.householderQ().setLength(Y.cols()) * MatrixType::Identity(Y.rows(), Y.cols());
    m.Rdiag = hhqr.matrixQR().diagonal().cwiseAbs().topRows(Y.cols());
    // Initial mask
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto       yj   = Y.col(j);
        RealScalar norm = yj.norm();
        if(norm < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(norm), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
        }
    }

    // Compress or randomize
    handle_masked_columns();
    auto h1info = SetH1MvInfo(ContractionBackend::X2); // Use high-precision matvec
    auto h2info = SetH2MvInfo(ContractionBackend::X2); // Use high-precision matvec
    H1Y         = MultH1(Y);
    H2Y         = MultH2(Y);
    assert_l2_orthonormal(Y, m);
}

/*! Orthonormalize Z in the appropriate metric
    Z is typically a set of eigenvectors for the small eigenvalue problem, used for Ritz extraction, e.g. V = Q * Z;

    Directly after solving T1*x = l*T2*x:
         - In L2 mode: Z.adjoint()*T2*Z = I, T2 != I
         - In H2 mode: Z.adjoint()*T2*Z ~ Z.adjoint()*Z ~ I because T2 ~ I.

    We can orthonormalize Z with QR in both L2 and H2 modes, because
         - In L2 mode: We want Z.adjoint()*Z = I, so we can take Householder QR to L2-orthonormalize Z directly.
         - In H2 mode, we want Z.adjoint() * T2 * Z = I, but we already have that T2 ~ I, and therefore householder QR works here too,
           but it is not strictly needed. We can check if T2 is actually an identity first.
*/
template<typename Scalar>
void solver_base<Scalar>::orthonormalize_Z(Eigen::Ref<MatrixType> Z, const Eigen::Ref<const MatrixType> &T2) {
    if(!use_h2_inner_product) {
        hhqr.compute(Z);
        Z = hhqr.householderQ().setLength(Z.cols()) * MatrixType::Identity(Z.rows(), Z.cols()); //
    } else {
        MatrixType G    = Z.adjoint() * T2 * Z;
        G               = (G + G.adjoint()).eval() * half;
        auto       es   = Eigen::SelfAdjointEigenSolver<MatrixType>(G);
        VectorReal D    = es.eigenvalues();
        MatrixType U    = es.eigenvectors();
        RealScalar cut  = 100 * eps * D.size() * D.cwiseAbs().maxCoeff();
        RealScalar cut2 = cut * cut;
        for(Eigen::Index j = 0; j < D.size(); ++j) {
            if(D(j) < cut2) {
                eiglog->warn("flooring D({})={:.5e} -> {:.5e}", j, fp(D(j)), fp(cut2));
                D(j) = std::max(D(j), cut2);
            }
        }
        Z *= U * D.cwiseInverse().cwiseSqrt().asDiagonal() * U.adjoint();
    }
}

template<typename Scalar>
void solver_base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &HY, OrthMeta &m) {
    if(Y.cols() == 0) {
        HY.resizeLike(Y);
        return;
    }
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo != OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask = VectorIdxT::Ones(Y.cols());
    if(std::isnan(m.maskTol)) m.maskTol = normTol * Y.cols(); // * get_op_norm_estimate();

    auto handle_masked_columns = [&]() {
        if(m.mask.sum() != Y.cols()) {
            VectorReal norms = (Y.adjoint() * Y).diagonal().cwiseAbs();
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS: {
                    eiglog->debug("block_l2_orthonormalize: Compressing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(norms), fp(m.maskTol));
                    compress_cols(Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                }
                case MaskPolicy::RANDOMIZE: {
                    eiglog->debug("block_l2_orthonormalize: Randomizing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(norms), fp(m.maskTol));
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) { Y.col(j) = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>(); }
                    }
                    break;
                }
                default: throw except::runtime_error("Unrecognized mask policy");
            }
        }
    };

    // Initial mask
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto       yj   = Y.col(j);
        RealScalar norm = yj.norm();
        if(norm < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(norm), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
        }
    }
    // Compress or randomize
    handle_masked_columns();
    if(Y.cols() == 0) {
        HY.resizeLike(Y);
        return;
    }

    // Orthonormalize
    hhqr.compute(Y);
    Y       = hhqr.householderQ().setLength(Y.cols()) * MatrixType::Identity(Y.rows(), Y.cols());
    m.Rdiag = hhqr.matrixQR().diagonal().cwiseAbs().topRows(Y.cols());
    // Initial mask
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto       yj   = Y.col(j);
        RealScalar norm = yj.norm();
        if(norm < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(norm), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
        }
    }

    // Compress or randomize
    handle_masked_columns();
    auto h1info = SetH1MvInfo(ContractionBackend::X2); // Use high-precision matvec
    auto h2info = SetH2MvInfo(ContractionBackend::X2); // Use high-precision matvec
    HY          = MultH(Y);
    assert_l2_orthonormal(Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &HX, MatrixType &Y, MatrixType &HY, OrthMeta &m) {
    if(X.cols() == 0 || Y.cols() == 0) {
        HY.resizeLike(Y);
        return;
    }
    if(m.mask.size() > 0 && m.mask.sum() == 0) return;
    assert(algo != OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    assert_allFinite(X);
    assert_allFinite(HX);
    assert_allFinite(Y);
    assert_allFinite(HY);
    assert_l2_orthonormal(X);

    if(std::isnan(m.orthTol)) m.orthTol = normTol * Y.cols();

    m.Gram      = X.adjoint() * Y;
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
    m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

    MatrixType Gxx = X.adjoint() * X;

    // DGKS clean Y against X
    Eigen::Index maxReps = 2;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        MatrixType W  = Gxx.ldlt().solve(m.Gram);
        Y.noalias()  -= X * W;

        m.Gram      = X.adjoint() * Y;
        m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
        m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

        // DGKS drop test – skip next rep if it already cleaned well
        bool orth_converged = m.orthError < m.orthTol;
        if(orth_converged or Y.cols() == 0) break;
    }
    if constexpr(settings::debug_solver)
        eiglog->trace("rep {} orthError after l2 orthonormalization: {:.3e} | orthTol {:.3e}", rep, fp(m.orthError), fp(m.orthTol));
    assert_l2_orthogonal(X, Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y,
                                                 MatrixType &H2Y, OrthMeta &m) {
    if(X.cols() == 0 || Y.cols() == 0) {
        H1Y.resizeLike(Y);
        H2Y.resizeLike(Y);
        return;
    }
    if(m.mask.size() > 0 && m.mask.sum() == 0) return;
    assert(algo == OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    assert_allFinite(X);
    assert_allFinite(H1X);
    assert_allFinite(H2X);
    assert_allFinite(Y);
    assert_allFinite(H1Y);
    assert_allFinite(H2Y);
    assert_l2_orthonormal(X);

    if(std::isnan(m.orthTol)) m.orthTol = orthTol * Y.cols();
    m.orthTol   = std::max(m.orthTol, orthTol * Y.cols());
    m.Gram      = X.adjoint() * Y;
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
    m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

    MatrixType Gxx = X.adjoint() * X;

    // DGKS clean Y against X
    Eigen::Index maxReps = 2;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        MatrixType W  = Gxx.ldlt().solve(m.Gram);
        Y.noalias()  -= X * W;

        m.Gram      = X.adjoint() * Y;
        m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
        m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

        // DGKS drop test – skip next rep if it already cleaned well
        bool orth_converged = m.orthError < m.orthTol;
        if(orth_converged or Y.cols() == 0) break;
    }
    if constexpr(settings::debug_solver)
        eiglog->trace("rep {} orthError after l2 orthonormalization: {:.3e} | orthTol {:.3e}", rep, fp(m.orthError), fp(m.orthTol));

    H1Y = MultH1(Y);
    assert_l2_orthogonal(X, Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthonormalize_dgks(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG and use_h2_inner_product);
    auto h1info                = SetH1MvInfo(ContractionBackend::X2); // Use more accurate matvec
    auto h2info                = SetH2MvInfo(ContractionBackend::X2); // Use more accurate matvec
    auto handle_masked_columns = [&]() {
        if(m.mask.sum() != Y.cols()) {
            MatrixType GI = m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols());
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS: {
                    eiglog->debug("block_h2_orthonormalize_dgks_x2: Compressing Y. Mask: {} | maskTol {:.3e} | G - I: \n{}\n", m.mask, fp(m.maskTol),
                                  linalg::matrix::to_string(GI, 8));
                    compress_cols(Y, m.mask);
                    compress_cols(H2Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    m.analyze_h2_orthonormality(Y, H2Y);
                    break;
                }
                case MaskPolicy::RANDOMIZE: {
                    eiglog->debug("block_h2_orthonormalize_dgks_x2: Randomizing Y. Mask: {} | maskTol {:.3e} | Gsym - I: \n", m.mask, fp(m.maskTol),
                                  linalg::matrix::to_string(GI, 8));
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) {
                            Y.col(j)   = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>();
                            H2Y.col(j) = MultH2(Y.col(j));
                        }
                    }
                    m.analyze_h2_orthonormality(Y, H2Y);
                    break;
                }
                default: throw except::runtime_error("Unrecognized mask policy");
            }
        }
    };

    auto dot_fp80 = [](Eigen::Ref<VectorType> a, Eigen::Ref<VectorType> b) -> Scalar {
        assert(a.size() == b.size());
        using LScalar = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
        return static_cast<Scalar>(a.template cast<LScalar>().dot(b.template cast<LScalar>()));
    };

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I
    m.mask        = VectorIdxT::Ones(Y.cols());
    m.proj_sum_h2 = VectorReal::Zero(Y.cols());
    m.scale_log   = VectorReal::Zero(Y.cols());
    if(std::isnan(m.maskTol)) m.maskTol = normTol * Y.cols(); // * get_op_norm_estimate();

    m.analyze_h2_orthonormality(Y, H2Y);

    // Orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I
    bool should_refresh_h2y = m.refresh_h2y or           // explicitly asked for
                              m.skewError > m.skewTol or // stale
                              Y.cols() != H2Y.cols() or  // bad size
                              Y.rows() != H2Y.rows();    // bad size

    // Orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I
    H2Y = MultH2(Y);
    eiglog->debug("block_h2_orthonormalize_dgks_x2: Refreshed H2Y");
    if(should_refresh_h2y) {}
    m.analyze_h2_orthonormality(Y, H2Y);

    assert_allFinite(H2Y);

    m.refresh_h2y = false;

    eiglog->info("block_h2_orthonormalize_dgks_x2: initial  orthError {:.4e} symmError {:.4e} skewError {:.4e} gram matrix: \n{}\n", fp(m.orthError),
                 fp(m.symmError), fp(m.skewError), linalg::matrix::to_string(m.Gram, 8));

    // Initial mask
    m.Rdiag = VectorReal::Zero(Y.cols());
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto yj   = Y.col(j);
        auto h2yj = H2Y.col(j);
        // 2) Norm & mask‐check
        auto normSq1 = std::real(dot_fp80(yj, h2yj));
        auto normSq2 = std::real(dot_fp80(h2yj, yj));
        auto norm    = std::sqrt(std::max<RealScalar>(0, (normSq1 + normSq2) * half));
        m.Rdiag(j)   = norm;
        if(norm < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(norm), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
            h2yj.setZero();
        }
    }
    // Compress or randomize
    handle_masked_columns();
    if(Y.cols() == 0) return;

    // DGKS passes
    Eigen::Index maxReps = 2;
    for(int rep = 0; rep < maxReps; ++rep) {
        VectorReal      normSqs = VectorReal::Zero(Y.cols());
        Eigen::VectorXi have    = Eigen::VectorXi::Zero(Y.cols()); // 0/1

        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            if(m.mask(j) == 0) continue;

            auto yj   = Y.col(j);
            auto h2yj = H2Y.col(j);

            // 1) Clean against i<j
            for(Eigen::Index i = 0; i < j; ++i) {
                if(m.mask(i) == 0) continue;
                auto yi   = Y.col(i);
                auto h2yi = H2Y.col(i);

                if(have(i) == 0) {
                    normSqs(i) = std::max<RealScalar>(0, std::real(dot_fp80(yi, h2yi)));
                    have(i)    = 1;
                }

                // auto       proj_ij = dot_fp80(yi, h2yj);
                RealScalar normSq  = normSqs(i);         // std::real(dot_fp80(yi, h2yi)); // yi^* H2 yj
                Scalar     proj1   = dot_fp80(yi, h2yj); // yi^* H2 yj
                Scalar     proj2   = dot_fp80(h2yi, yj); // (yi H2)^* yj
                Scalar     proj_ij = (proj1 + proj2) / (RealScalar{2} * normSq);
                // Scalar proj_ij = proj1 ;
                eiglog->info("(i:{:3}, j:{:3}): p1 = {:.4e} | p2 = {:.4e} | |p1-p2| = {:.4e} |yi| = {:.4e} |yj| = {:.4e} |h2yi| = {:.4e} |h2yj| = {:.4e}", i, j,
                             fp(proj1), fp(proj2), fp(std::abs(proj1 - proj2)), fp(yi.norm()), fp(yj.norm()), fp(h2yi.norm()), fp(h2yj.norm()));

                // subtract
                yj.noalias()   -= yi * proj_ij;
                h2yj.noalias() -= h2yi * proj_ij;
            }

            // 2) Norm & mask‐check
            auto normSq = std::real(dot_fp80(yj, h2yj));
            auto norm   = std::sqrt(std::max<RealScalar>(0, normSq));
            if(norm <= m.maskTol) {
                eiglog->trace("Masking column {}: normSq = {:.4e} | norm {:.4e} | maskTol = {:.4e}", j, fp(normSq), fp(norm), fp(m.maskTol));
                m.mask(j) = 0;
                yj.setZero();
                h2yj.setZero();
                continue;
            }

            // 3) Normalize
            eiglog->info("(j:{:3}) norm error = {:.4e}", j, fp(std::abs(norm - RealScalar{1})));
            yj   /= norm;
            h2yj /= norm;

            // Cache diagonals for this column j (now “final” for this rep)
            normSqs(j) = std::max<RealScalar>(0, std::real(dot_fp80(yj, h2yj)));
        }
        m.analyze_h2_orthonormality(Y, H2Y);
        handle_masked_columns(); // Compress or randomize

        if constexpr(settings::debug_solver) {
            MatrixType GI = m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols());
            eiglog->trace(
                "block_h2_orthonormalize_dgks_x2: dgks rep {}: orthError = {:.4e} symmError = {:.4e}  skewError = {:.4e}  |H2Y| = {:.4e} Y.cols() = {}  "
                "|H2| = {:.3e} G - I:\n{}\n",
                rep, fp(m.orthError), fp(m.symmError), fp(m.skewError), fp(H2Y.norm()), Y.cols(), fp(H2.get_op_norm()), linalg::matrix::to_string(GI, 8));
        }
        if(m.orthError < m.orthTol) break;
    }

    H1Y = MultH1(Y);
    assert_h2_orthonormal(Y, H2Y, m);
}

template<typename LScalar>
struct EigOrthoStepMeta {
    using RealLScalar = decltype(std::real(std::declval<LScalar>()));
    using MatrixLType = Eigen::Matrix<LScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorIdxT  = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    using VectorLReal = Eigen::Matrix<RealLScalar, Eigen::Dynamic, 1>;
    MatrixLType Y, H2Y;
    MatrixLType G;
    RealLScalar symmError;
    template<typename Scalar>
    EigOrthoStepMeta(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Y_Scalar, //
                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H2Y_Scalar)
        : Y(Y_Scalar.template cast<LScalar>()), H2Y(H2Y_Scalar.template cast<LScalar>()) {}
};

template<typename Scalar, typename RealScalar, typename LScalar>
void do_eig_orthonormalization_step(
    EigOrthoStepMeta<LScalar> &m,
    std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &)>
                                                     applyH2,
    [[maybe_unused]] std::shared_ptr<spdlog::logger> eiglog) {
    using RealLScalar = typename EigOrthoStepMeta<LScalar>::RealLScalar;
    using MatrixLType = typename EigOrthoStepMeta<LScalar>::MatrixLType;
    using VectorLReal = typename EigOrthoStepMeta<LScalar>::VectorLReal;

    auto &G         = m.G;
    auto &Y         = m.Y;
    auto &H2Y       = m.H2Y;
    auto &symmError = m.symmError;

    static constexpr auto half = RealLScalar{1} / RealLScalar{2};

    auto assert_finite = [&]() {
        if constexpr(settings::debug_solver) {
            bool ynan   = !Y.allFinite();
            bool h2ynan = !H2Y.allFinite();
            if(ynan or h2ynan) eiglog->info("do_eig_orthonormalization_step: G:\n{}\n", linalg::matrix::to_string(G, 8));
            if(ynan) throw except::runtime_error("do_eig_orthonormalization_step: Y has nan or inf");
            if(h2ynan) throw except::runtime_error("do_eig_orthonormalization_step: H2Y has nan or inf");
        }
    };
    MatrixLType G1          = Y.adjoint() * H2Y;
    G                       = (G1 + G1.adjoint()) * half; // The Gram matrix must be hermitian (and PSD)
    symmError               = (G - MatrixLType::Identity(G.rows(), G.cols())).norm();
    VectorLReal Gdiag       = G.real().diagonal();
    VectorLReal scaleErrors = Gdiag - VectorLReal::Ones(Gdiag.size());
    assert_finite();
    if constexpr(settings::debug_solver) { eiglog->trace("do_eig_orthonormalization_step: Scale errors diag(G)-I: {::.5e}", fv(scaleErrors)); }

    if(Y.cols() == 0) {
        // Nothing left to orthonormalize
        if constexpr(settings::debug_solver) eiglog->trace("do_eig_orthonormalization_step: no columns left");
        G         = MatrixLType();
        symmError = RealLScalar{0};
        return;
    }

    // Step 1: Compute the eigenvalues of G
    auto esG = Eigen::SelfAdjointEigenSolver<MatrixLType>(G);
    if(esG.info() != Eigen::Success) throw except::runtime_error("do_eig_orthonormalization_step: eig failed. G = \n{}\n", linalg::matrix::to_string(G, 8));
    VectorLReal lG = esG.eigenvalues();
    if constexpr(settings::debug_solver) eiglog->trace("do_eig_orthonormalization_step: λ(G) = {::.5e}", fv(lG));

    // Step 2: Drop eigenvalues of G that are too small (these correspond to nearly collinear vectors in H2-norm)
    RealLScalar eps100 = std::numeric_limits<RealLScalar>::epsilon() * RealLScalar(100);
    RealLScalar tol    = eps100 * std::max<RealLScalar>(RealLScalar(1), lG.cwiseAbs().maxCoeff());

    std::vector<Eigen::Index> keep;
    for(Eigen::Index j = 0; j < lG.size(); ++j) {
        if(lG(j) > tol) {
            keep.push_back(j); // Keep only positive eigenvalues
        } else {
            // if constexpr(settings::debug_solver)
            eiglog->trace("do_eig_orthonormalization_step: dropping eigenvalue {} of {}: evs: {::.5e}", j, G.rows(), fv(lG));
        }
    }

    if(keep.empty()) {
        Y.resize(Y.rows(), 0);
        H2Y.resize(H2Y.rows(), 0);
        G         = MatrixLType();
        symmError = RealLScalar{0};
        return;
    }

    VectorLReal D = lG(keep);
    MatrixLType U = esG.eigenvectors()(Eigen::placeholders::all, keep);
    MatrixLType W = U * D.cwiseInverse().cwiseSqrt().asDiagonal();

    // Step 3: Compress and normalize Y and H2Y in one shot. Note that if keep = {} (empty), then Y and H2Y become empty.
    Y   = (Y * W).eval();   // Note that W may not be square due to pruning
    H2Y = (H2Y * W).eval(); // Note that W may not be square due to pruning
    if constexpr(settings::debug_solver) {
        MatrixLType H2_YW = applyH2(Y.template cast<Scalar>()).template cast<LScalar>(); // Refresh
        MatrixLType Delta = H2_YW - H2Y;
        for(Eigen::Index i = 0; i < H2Y.cols(); i++) {
            auto yw = Y.col(i);
            // auto        h2y_w      = H2Y.col(i);
            auto        h2_yw      = H2_YW.col(i);
            RealLScalar delta      = Delta.col(i).norm();
            RealLScalar eta_lin    = delta / h2_yw.norm();
            RealLScalar yw_norm    = yw.norm();
            RealLScalar h2_yw_norm = h2_yw.norm();
            eiglog->debug("[{:2}]: eta_lin={:.4e} |yw|={:.4e} |h2_yw|={:.4e} |Δ|={:.4e}", i, fp(eta_lin), fp(yw_norm), fp(h2_yw_norm), fp(delta));
        }
        MatrixLType E_predict = Y.adjoint() * Delta;
        eiglog->debug("E = YW^*Δ (prediction) = \n{}\n", linalg::matrix::to_string(E_predict, 8));
        eiglog->debug("symmError   = {:.5e}", fp(E_predict.norm()));
    }

    // Refresh the Gram matrix
    G1 = Y.adjoint() * H2Y;
    G  = (G1 + G1.adjoint()) * half;

    MatrixLType E_H2Y_W = (G - MatrixLType::Identity(G.rows(), G.cols()));
    symmError           = E_H2Y_W.norm();
    assert_finite();
    if constexpr(settings::debug_solver) {
        eiglog->debug("E = Gsymm - I (from H2Y_W) = \n{}\n", linalg::matrix::to_string(E_H2Y_W, 8));
        eiglog->debug("symmError   = |Gsymm - I|: {:.5e}", fp(symmError));
    }
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthonormalize_eig(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG and use_h2_inner_product);
    assert(m.maskPolicy == MaskPolicy::COMPRESS); // This operation does not preserve column order

    auto h1info = SetH1MvInfo(ContractionBackend::X2); // Use high-precision matvec
    auto h2info = SetH2MvInfo(ContractionBackend::X2); // Use high-precision matvec

    // Orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I
    m.analyze_h2_orthonormality(Y, H2Y);
    if(m.refresh_h2y or Y.cols() != H2Y.cols() or Y.rows() != H2Y.rows() or m.skewError > m.skewTol) {
        H2Y = MultH2(Y);
        m.analyze_h2_orthonormality(Y, H2Y);
        if constexpr(settings::debug_solver) {
            eiglog->debug("block_h2_orthonormalize_eig: Refreshed H2Y");

            OrthMeta   mdbg    = m;
            MatrixType H2Y_dbg = MultH2(Y);
            mdbg.analyze_h2_orthonormality(Y, H2Y_dbg);
            RealScalar H2Y_err = (H2Y - H2Y_dbg).norm();
            eiglog->debug("block_h2_orthonormalize_eig: high precision error mitigation: {:.4e} | skewErrors hp={:.4e} dbg={:.4e}", fp(H2Y_err),
                          fp(m.skewError), fp(mdbg.skewError));
        }

    } else {
        assert_allFinite(H2Y);
        if constexpr(settings::debug_solver) {
            MatrixType H2Y_dbg = MultH2(Y);
            RealScalar H2Y_err = (H2Y - H2Y_dbg).norm();
            // if(H2Y_err > 1e8 * eps) throw except::runtime_error("block_h2_orthonormalize_eig: H2Y mismatch: err {:.4e}", fp(H2Y_err));
            if(H2Y_err > std::sqrt(eps)) eiglog->warn("block_h2_orthonormalize_eig: H2Y mismatch: err {:.4e}", fp(H2Y_err));
        }
    }

    m.refresh_h2y = false;

    if constexpr(settings::debug_solver)
        eiglog->trace("block_h2_orthonormalize_eig: initial ortherror={:.4e} symmError={:.4e} skewError={:.4e} Gyy: \n{}\n", fp(m.orthError), fp(m.symmError),
                      fp(m.skewError), linalg::matrix::to_string(m.Gram, 8));

    if(m.symmError < m.orthTol) {
        H1Y = MultH1(Y);
        assert_h2_orthonormal(Y, H2Y, m);
        return; // No need to orthonormalize
    }
    auto assert_finite = [&]() {
        if constexpr(settings::debug_solver) {
            if(!Y.allFinite()) throw except::runtime_error("block_h2_orthonormalize_eig: Y has nan or inf");
            if(!H2Y.allFinite()) throw except::runtime_error("block_h2_orthonormalize_eig: H2Y has nan or inf");
        }
    };

    // using ScalarL              = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<double>, double>;
    // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
    // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
    fMultH_t fMultH2 = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return this->MultH2(X); };

    Eigen::Index maxReps = 1;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        // balance_columns_sweep(Y, H2Y, /*num_sweeps=*/2, /*max_pairs_per_sweep=*/-1, /*target_ratio=*/2.0);
        assert_finite();

        auto eosm = EigOrthoStepMeta<Scalar>(Y, H2Y);
        do_eig_orthonormalization_step<Scalar, RealScalar, Scalar>(eosm, fMultH2, eiglog);

        assert_finite();

        if(eosm.Y.cols() == 0) {
            if constexpr(settings::debug_solver) eiglog->trace("block_h2_orthonormalize_eig: 0/{} cols remain in Y", m.Gram.cols());
            Y   = MatrixType();
            H1Y = MatrixType();
            H2Y = MatrixType();
            m   = OrthMeta();
            return;
        }

        // Extract the solution
        Y   = eosm.Y.template cast<Scalar>();
        H2Y = eosm.H2Y.template cast<Scalar>();
        m.analyze_h2_orthonormality(Y, H2Y);
        assert_finite();

        if constexpr(settings::debug_solver) {
            eiglog->trace("block_h2_orthonormalize_eig: eig rep {}: orthError = {:.4e} symmError {:.4e} skewError {:.4e} | tol {:.5e}", rep, fp(m.orthError),
                          fp(m.symmError), fp(m.skewError), fp(normTol));
        }
    }
    if(m.skewError >= RealScalar{1e-3f}) {
        MatrixType GramError = m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols());
        eiglog->warn("block_h2_orthonormalize_eig: very large skew error on rep {}: orthError = {:.4e} symmError = {:.4e} skewError = {:.4e} "
                     "cols {}  | \n G - I: \n{}",
                     rep, fp(m.orthError), fp(m.symmError), fp(m.skewError), Y.cols(), linalg::matrix::to_string(GramError, 8));
    }

    H1Y = MultH1(Y);
    assert_h2_orthonormal(Y, H2Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y,
                                                 MatrixType &H2Y, OrthMeta &m) {
    if(X.cols() == 0 || Y.cols() == 0) return;
    if(m.mask.size() > 0 && m.mask.sum() == 0) return;
    assert(algo == OptAlgo::GDMRG and use_h2_inner_product && "block_h2_orthogonalize is for H2 inner product");
    auto h1info = SetH1MvInfo(ContractionBackend::X2); // Use high-precision matvec
    auto h2info = SetH2MvInfo(ContractionBackend::X2); // Use high-precision matvec
    assert_allFinite(X);
    assert_allFinite(H1X);
    assert_allFinite(H2X);
    assert_allFinite(Y);
    assert_h2_orthonormal(X, H2X);

    if(std::isnan(m.orthTol)) m.orthTol = orthTol * X.cols();
    m.orthTol = std::max(m.orthTol, eps * std::sqrt(status.op_norm_estimate));
    if(!std::isfinite(m.orthTol))
        throw except::runtime_error("block_h2_orthogonalize: invalid value: m.orthTol={:.3e} | status.op_norm_estimate={:.3e}", fp(m.orthTol),
                                    fp(status.op_norm_estimate));
    bool has_refreshed_h2y = false;
    if(m.refresh_h2y or Y.size() != H2Y.size()) {
        H2Y               = MultH2(Y);
        has_refreshed_h2y = true;
        eiglog->trace("Refreshed H2Y");
    } else {
        assert_allFinite(H2Y);
        if constexpr(settings::debug_solver) {
            MatrixType H2Y_dbg = MultH2(Y);
            RealScalar H2Y_err = (H2Y - H2Y_dbg).norm();
            // if(H2Y_err > 1e8 * eps) throw except::runtime_error("block_h2_orthogonalize: H2Y mismatch: err {:.4e}", fp(H2Y_err));
            if(H2Y_err > std::sqrt(eps)) eiglog->warn("block_h2_orthogonalize: H2Y mismatch: err {:.4e}", fp(H2Y_err));
        }
    }

    m.analyze_h2_orthogonality(X, H2X, Y, H2Y);

    MatrixType Gyy = Y.adjoint() * H2Y;
    Gyy            = (Gyy + Gyy.adjoint()).eval() / RealScalar{2};
    RealScalar Eyy = (Gyy - MatrixType::Identity(Gyy.cols(), Gyy.rows())).norm();

    MatrixType Gxx = X.adjoint() * H2X;
    Gxx            = (Gxx + Gxx.adjoint()).eval() / RealScalar{2};
    RealScalar Exx = (Gxx - MatrixType::Identity(Gxx.cols(), Gxx.rows())).norm();

    if(m.skewError > std::sqrt(m.orthTol) and !has_refreshed_h2y) {
        eiglog->debug("block_h2_orthogonalize: initial orthError = {:.4e} symmError = {:.4e} skewError = {:.4e} Exx {:.4e} Eyy {:.4e}", fp(m.orthError),
                      fp(m.symmError), fp(m.skewError), fp(Exx), fp(Eyy));
        MatrixType H2Y_new = MultH2(Y);
        OrthMeta   m_new   = m;
        m_new.analyze_h2_orthogonality(X, H2X, Y, H2Y_new);
        if(m_new.skewError < m.skewError) {
            eiglog->debug("block_h2_orthogonalize: initial orthError = {:.4e} symmError = {:.4e} skewError = {:.4e} Exx {:.4e} Eyy {:.4e} (after H2Y refresh)",
                          fp(m.orthError), fp(m.symmError), fp(m.skewError), fp(Exx), fp(Eyy));

            H2Y.swap(H2Y_new);
            m                 = m_new;
            has_refreshed_h2y = true;
        }
    }

    if constexpr(settings::debug_solver) {
        eiglog->trace("block_h2_orthogonalize:          rep-1: orthError = {:.4e} symmError = {:.4e} skewError = {:.4e} Exx {:.4e} Eyy {:.4e}", fp(m.orthError),
                      fp(m.symmError), fp(m.skewError), fp(Exx), fp(Eyy));
    }
    if(std::isfinite(m.orthTol) and std::max(m.symmError, m.skewError) < m.orthTol) {
        if(has_refreshed_h2y or m.refresh_h2y or Y.size() != H1Y.size()) H1Y = MultH1(Y);
        // if constexpr(settings::debug_solver)
        eiglog->trace("block_h2_orthogonalize: no need: orthError = {:.4e} symmError = {:.4e} skewError = {:.4e},  Eyy = {:.4e} < orthTol {:.4e}",
                      fp(m.orthError), fp(m.symmError), fp(m.skewError), fp(Eyy), fp(m.orthTol));
        return; // No need to orthogonalize or orthonormalize
    }
    m.refresh_h2y = false;

    if(Exx > m.orthTol) { eiglog->debug("block_h2_orthogonalize: X is not sufficiently H2-orthonormal: xOrthError= {:.4e}", fp(Exx)); }
    if(Exx > 10000 * m.orthTol) {
        eiglog->warn("block_h2_orthogonalize: X is not sufficiently H2-orthonormal: xOrthError= {:.4e}: Gxx = \n{}\n", fp(Exx),
                     linalg::matrix::to_string(Gxx, 8));
    }

    // DGKS clean Y against X
    Eigen::Index maxReps = 2;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        if(m.mask.size() != Y.cols()) m.mask = VectorIdxT::Ones(Y.cols());
        if(m.proj_sum_h1.size() != Y.cols()) m.proj_sum_h1 = VectorReal::Zero(Y.cols());
        if(m.proj_sum_h2.size() != Y.cols()) m.proj_sum_h2 = VectorReal::Zero(Y.cols());
        if(m.scale_log.size() != Y.cols()) m.scale_log = VectorReal::Zero(Y.cols());

        MatrixType W = Gxx.ldlt().solve(m.Gram_symm);

        Y.noalias()   -= X * W;
        H2Y.noalias() -= H2X * W;

        if constexpr(settings::debug_solver) {
            RealScalar E_proj = (X.adjoint() * (MultH2(Y) - H2Y)).norm();
            eiglog->trace(
                "block_h2_orthogonalize:          rep {}: orthError = {:.4e} symmError = {:.4e} skewError = {:.4e} E_proj={:.4e} Exx {:.4e} Eyy {:.4e}", rep,
                fp(m.orthError), fp(m.symmError), fp(m.skewError), fp(E_proj), fp(Exx), fp(Eyy));
        }
        if(rep >= 1) {
            // DGKS drop test – skip next rep if it already cleaned well
            bool orth_converged = m.orthError < m.orthTol;
            if(orth_converged) break;
        }
    }
    assert_h2_orthogonal(X, H2Y, m);
    assert_h2_orthogonal(H2X, Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::pad_and_orthonormalize(MatrixType &Y, MatrixType &HY, Eigen::Index nBlocks, OrthMeta &m) {
    Eigen::Index reps = 0;
    while(reps++ == 0 or Y.cols() / b < nBlocks) {
        if(Y.cols() < nBlocks * b) {
            // Pad with random vectors
            auto vc = Y.cols();
            Y.conservativeResize(Y.rows(), nBlocks * b);
            auto Yrc = Y.rightCols(nBlocks * b - vc);
            for(auto yj : Yrc.colwise()) { yj = Eigen::VectorXf::Random(yj.size()).template cast<Scalar>(); }
        }
        block_l2_orthonormalize(Y, HY, m);
    }
}

template<typename Scalar>
void solver_base<Scalar>::pad_and_orthonormalize(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, Eigen::Index nBlocks, OrthMeta &m) {
    Eigen::Index reps = 0;
    while(reps++ == 0 or Y.cols() < nBlocks * b) {
        if(Y.cols() < nBlocks * b) {
            // Pad with random vectors
            auto vc = Y.cols();
            Y.conservativeResize(Y.rows(), nBlocks * b);
            auto Yrc = Y.rightCols(nBlocks * b - vc);
            for(auto yj : Yrc.colwise()) { yj = Eigen::VectorXf::Random(yj.size()).template cast<Scalar>(); }
            eiglog->info("Randomizing {} blocks", nBlocks * b - vc);
            m.refresh_h2y = true;
        }

        if(algo == OptAlgo::GDMRG) {
            if(use_h2_inner_product) {
                block_h2_orthonormalize_eig(Y, H1Y, H2Y, m);
            } else {
                // V is expected to be H2-orthonormal, so we L2 orthonormalize it
                block_l2_orthonormalize(Y, H1Y, H2Y, m);
            }
        }
    }
}

template<typename Scalar>
std::vector<Eigen::Index> solver_base<Scalar>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const {
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
void solver_base<Scalar>::init() {
    auto t_init = tid::tic_scope("init");
    assert(H1.rows() == H1.cols() && "H1 must be square");
    assert(H2.rows() == H2.cols() && "H2 must be square");
    assert(N == H1.rows() && "H1 and H2 must have same dimension");
    assert(N == H2.rows() && "H1 and H2 must have same dimension");
    nev                         = std::min(nev, N);
    ncv                         = std::min(std::max(nev, ncv), N);
    b                           = std::min(std::max(nev, b), N / 2);
    status.saturation_count_max = ncv;
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
            auto Vrc = V.rightCols(b - vc);
            for(auto vj : Vrc.colwise()) { vj = Eigen::VectorXf::Random(vj.size()).template cast<Scalar>(); }
        }
        // Orthonormalize V.
        // Discard columns if there are more than b (this is not expected, but also not an error)
        cpqr.compute(V);
        auto rank = std::min(cpqr.rank(), b);
        V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank) * cpqr.colsPermutation().transpose();
        if(V.cols() == b) break;
    }

    auto block_orthonormalize = [&] {
        auto m        = OrthMeta();
        m.refresh_h2y = true;
        m.maskPolicy  = MaskPolicy::COMPRESS;
        if(algo == OptAlgo::GDMRG) {
            if(use_h2_inner_product) {
                block_h2_orthonormalize_dgks(V, H1V, H2V, m);
            } else {
                block_l2_orthonormalize(V, H1V, H2V, m);
            }
        } else {
            block_l2_orthonormalize(V, HV, m);
        }
    };

    assert(V.cols() == b);
    if(status.iter == 0) {
        // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
        if(algo == OptAlgo::GDMRG) {
            block_orthonormalize();
            Q             = V;
            H1Q           = H1V;
            H2Q           = H2V;
            MatrixType T1 = Q.adjoint() * H1Q;
            MatrixType T2 = Q.adjoint() * H2Q;
            T1            = RealScalar{0.5f} * (T1.adjoint() + T1); // Symmetrize
            T2            = RealScalar{0.5f} * (T2.adjoint() + T2); // Symmetrize
            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es_seed(T1, T2, Eigen::Ax_lBx);
            T_evecs       = es_seed.eigenvectors();
            T_evals       = es_seed.eigenvalues();
            status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
            MatrixType Z  = T_evecs(Eigen::placeholders::all, status.optIdx);
            VectorReal Y  = T_evals(status.optIdx);
            V             = Q * Z;   // Now V has b columns mixed according to the selected columns in T_evecs
            H1V           = H1Q * Z; // Now H1V has b columns mixed according to the selected columns in T_evecs
            H2V           = H2Q * Z; // Now H2V has b columns mixed according to the selected columns in T_evecs

            status.commit_evals(T_evals.cwiseAbs().minCoeff(), T_evals.cwiseAbs().maxCoeff());
            Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
            Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
            status.T1_evals    = es1.eigenvalues();
            status.T2_evals    = es2.eigenvalues();
            status.T1_min_eval = es1.eigenvalues().minCoeff();
            status.T1_max_eval = es1.eigenvalues().maxCoeff();
            status.T2_min_eval = es2.eigenvalues().minCoeff();
            status.T2_max_eval = es2.eigenvalues().maxCoeff();
            RealScalar min_sep =
                T_evals.size() <= 1 ? RealScalar{1} : (T_evals.tail(T_evals.size() - 1) - T_evals.head(T_evals.size() - 1)).cwiseAbs().minCoeff();
            auto select1       = get_ritz_indices(ritz, 0, 1, T_evals);
            auto H1_max_abs    = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
            auto H2_max_abs    = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});
            status.sensitivity = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;

            status.T_max_eval       = T_evals.maxCoeff();
            status.T_min_eval       = T_evals.minCoeff();
            auto H1H2_max_abs       = std::max(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
            auto H1H2_min_abs       = std::min(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
            status.condition        = H1H2_max_abs / H1H2_min_abs;
            status.op_norm_estimate = get_op_norm_estimate();
            // We may need to orthonormalize V in GDMRG
            block_orthonormalize();

            std::tie(S, status.rNorms) = get_residuals(Y, H1V, H2V);
            status.eigVal              = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b

        } else {
            block_orthonormalize();
            Q  = V;
            HQ = MultH(V);
            T  = Q.adjoint() * HQ;
            T  = RealScalar{0.5f} * (T.adjoint() + T); // Symmetrize
            Eigen::SelfAdjointEigenSolver<MatrixType> es(T);
            T_evecs                    = es.eigenvectors();
            T_evals                    = es.eigenvalues();
            status.optIdx              = get_ritz_indices(ritz, 0, b, T_evals);
            MatrixType Z               = T_evecs(Eigen::placeholders::all, status.optIdx);
            VectorReal Y               = T_evals(status.optIdx);
            V                          = Q * Z; // Now V has b columns mixed according to the selected columns in T_evecs
            HV                         = HQ * Z;
            std::tie(S, status.rNorms) = get_residuals(Y, HV, V);
            status.eigVal              = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            status.T1_evals            = es.eigenvalues();
            status.T2_evals            = es.eigenvalues();
            status.T1_min_eval         = T_evals.minCoeff();
            status.T2_min_eval         = T_evals.minCoeff();
            status.T1_max_eval         = T_evals.maxCoeff();
            status.T2_max_eval         = T_evals.maxCoeff();
            status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());

            status.T_max_eval       = T_evals.maxCoeff();
            status.T_min_eval       = T_evals.minCoeff();
            auto H1H2_max_abs       = std::max(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
            auto H1H2_min_abs       = std::min(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
            status.condition        = H1H2_max_abs / H1H2_min_abs;
            status.op_norm_estimate = get_op_norm_estimate();
        }
    }
    status.rNorms_init = status.rNorms;
    assert(V.cols() == b);
    assert_allFinite(V);
    last_log_time.tic();
    last_log_time.start_lap();
}

template<typename Scalar>
void solver_base<Scalar>::diagonalizeT() {
    if(algo == OptAlgo::GDMRG) return diagonalizeT1T2();
    if(status.stopReason != StopReason::none) return;
    if(Q.cols() == 0) return;
    if(HQ.cols() == 0) return;
    auto t_diag = tid::tic_scope("diagonalizeT");

    assert(Q.cols() == HQ.cols());

    MatrixType T = Q.adjoint() * HQ;
    T            = (T + T.adjoint()).eval() / RealScalar{2}; // Symmetrize
    assert(T.colwise().norm().minCoeff() != 0);

    Eigen::SelfAdjointEigenSolver<MatrixType> es(T, Eigen::ComputeEigenvectors);
    T_evals            = es.eigenvalues();
    T_evecs            = es.eigenvectors();
    status.T1_evals    = es.eigenvalues();
    status.T2_evals    = es.eigenvalues();
    status.T1_min_eval = std::min(status.T1_min_eval, T_evals.minCoeff());
    status.T1_max_eval = std::max(status.T1_max_eval, T_evals.maxCoeff());
    status.T2_min_eval = std::min(status.T2_min_eval, T_evals.minCoeff());
    status.T2_max_eval = std::max(status.T2_max_eval, T_evals.maxCoeff());

    auto diff = [](const VectorReal &x) -> VectorReal {
        if(x.size() <= 1) return VectorReal::Ones(1);
        return x.tail(x.size() - 1) - x.head(x.size() - 1);
    };
    if(T_evals.size() >= std::max(b, nev + 1)) {
        auto select2 = get_ritz_indices(ritz, 0, nev + 1, T_evals);
        status.gap   = diff(T_evals(select2)).cwiseAbs().minCoeff();
    }

    status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
    status.T_min_eval       = std::min(status.T_min_eval, T_evals.minCoeff());
    status.T_max_eval       = std::max(status.T_max_eval, T_evals.maxCoeff());
    auto H1H2_max_abs       = std::max(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
    auto H1H2_min_abs       = std::min(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
    status.condition        = H1H2_max_abs / H1H2_min_abs;
    status.op_norm_estimate = get_op_norm_estimate();
    if(status.iter > 1 and use_deflated_inner_preconditioner) {
        auto Z                  = es.eigenvectors().leftCols(1);
        auto jcbCfg             = algo == OptAlgo::DMRG ? H1.get_iterativeLinearSolverConfig().jacobi : H2.get_iterativeLinearSolverConfig().jacobi;
        jcbCfg.deflationEigVecs = Q * Z;
        jcbCfg.deflationEigInvs = es.eigenvalues().topRows(1).cwiseInverse();
    }
}

template<typename Scalar>
void solver_base<Scalar>::diagonalizeT1T2() {
    if(status.stopReason != StopReason::none) return;
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("diagonalizeT1T2() is only implemented for GDMRG");
    auto t_diag = tid::tic_scope("diagonalizeT1T2");

    status.rNorms              = {};
    static constexpr auto half = RealScalar{1} / RealScalar{2};
    if(Q.cols() == 0) throw except::runtime_error("Q has no columns");
    if(H1Q.cols() == 0) throw except::runtime_error("H1Q has no columns");
    if(H2Q.cols() == 0) throw except::runtime_error("H2Q has no columns");
    T1 = Q.adjoint() * H1Q;
    T2 = Q.adjoint() * H2Q;

    // Symmetrize
    T1 = (T1 + T1.adjoint()).eval() * half;
    T2 = (T2 + T2.adjoint()).eval() * half;
    assert(T1.rows() == T2.rows());
    assert(T1.cols() == T2.cols());
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T1, T2, Eigen::Ax_lBx);
    // {
    //     auto             es1     = Eigen::SelfAdjointEigenSolver<MatrixType>(T1, Eigen::EigenvaluesOnly);
    //     auto             es2     = Eigen::SelfAdjointEigenSolver<MatrixType>(T2, Eigen::EigenvaluesOnly);
    //     auto             idx_es1 = get_ritz_indices(OptRitz::SM, 0, 1, es1.eigenvalues());
    //     auto             idx_es2 = get_ritz_indices(OptRitz::SM, 0, 1, es2.eigenvalues());
    //     const RealScalar eval1   = es1.eigenvalues()(idx_es1[0]);
    //     const RealScalar eval2   = es2.eigenvalues()(idx_es2[0]);
    //     eiglog->info("es1: {:.16e} | es2: {:.16e}", fp(eval1), fp(eval2));
    // }

    if(es.info() == Eigen::Success) {
        T_evals = es.eigenvalues();
        T_evecs = es.eigenvectors();
        orthonormalize_Z(T_evecs, T2);
    } else {
        // Failed to add a nonzero residual
        status.stopReason |= StopReason::no_valid_eigenvector;
        status.stopMessage.emplace_back(fmt::format("Eigen::GeneralizedSelfAdjointEigenSolver failed | iter {} | mv {} | {:.3e} s", status.iter,
                                                    status.num_matvecs_total, status.time_elapsed.get_time()));
        return;
    }
    status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
    status.T_min_eval = std::min(status.T_min_eval, T_evals.minCoeff());
    status.T_max_eval = std::max(status.T_max_eval, T_evals.maxCoeff());
    auto H1H2_max_abs = std::max(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
    auto H1H2_min_abs = std::min(std::abs(status.T_min_eval), std::abs(status.T_max_eval));
    status.condition  = H1H2_max_abs / H1H2_min_abs;
    // Calculate the gap
    auto diff = [](const VectorReal &x) -> VectorReal {
        if(x.size() <= 1) return VectorReal::Ones(1);
        return x.tail(x.size() - 1) - x.head(x.size() - 1);
    };
    if(T_evals.size() >= std::max(b, nev + 1)) {
        auto select = get_ritz_indices(ritz, 0, std::max(b, nev + 1), T_evals);
        status.gap  = diff(T_evals(select)).cwiseAbs().minCoeff();
    }

    // Calculate min max eigenvalues of H1 and H2 and condition number

    Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
    Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
    {
        status.T1_evals       = es1.eigenvalues();
        status.T2_evals       = es2.eigenvalues();
        status.T1_min_eval    = std::min(status.T1_min_eval, es1.eigenvalues().minCoeff());
        status.T1_max_eval    = std::max(status.T1_max_eval, es1.eigenvalues().maxCoeff());
        status.T2_min_eval    = std::min(status.T2_min_eval, es2.eigenvalues().minCoeff());
        status.T2_max_eval    = std::max(status.T2_max_eval, es2.eigenvalues().maxCoeff());
        RealScalar min_sep    = diff(T_evals).cwiseAbs().minCoeff();
        auto       select1    = get_ritz_indices(ritz, 0, 1, T_evals);
        auto       H1_max_abs = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
        auto       H2_max_abs = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});
        status.sensitivity    = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;

        // auto       select_b = get_ritz_indices(ritz, 0, b, T_evals);
        // VectorReal evals    = T_evals(select_b);
        // eiglog->debug("Op evals {::.5e}", fv(evals));
        // eiglog->debug("H1 evals {::.5e}", fv(es1.eigenvalues()));
        // eiglog->debug("H2 evals {::.5e}", fv(es2.eigenvalues()));
    }

    // Register deflation and coarse space vectors

    if(status.iter + 1 >= 2 and use_deflated_inner_preconditioner) {
        Eigen::Index              nDefl   = std::min(5l, es2.eigenvalues().size());
        MatrixType                Z       = es2.eigenvectors().leftCols(nDefl);
        VectorReal                Y       = es2.eigenvalues().topRows(nDefl);
        MatrixType                Vdefl   = Q * Z;
        VectorReal                rnorms  = (H2Q * Z - Vdefl * Y.asDiagonal()).colwise().norm();
        std::vector<Eigen::Index> deflIdx = {};
        for(Eigen::Index idx = 0; idx < nDefl; ++idx) {
            if(rnorms(idx) < RealScalar{1e-5f} and Y(idx) < RealScalar{1e-2f}) deflIdx.emplace_back(idx);
        }
        if(deflIdx.size() > 0) {
            eiglog->trace("deflating idx {} | eigv {} | rnorms {}", deflIdx, fv(Y), fv(rnorms));
            // one-time B-orthonormalisation of Z
            Z                             = Z(Eigen::placeholders::all, deflIdx).eval();
            Vdefl                         = Vdefl(Eigen::placeholders::all, deflIdx).eval();
            Y                             = Y(deflIdx).eval();
            rnorms                        = rnorms(deflIdx).eval();
            MatrixType             GramH2 = Vdefl.adjoint() * (H2Q * Z); // small p×p matrix
            Eigen::LLT<MatrixType> llt(GramH2);
            Vdefl = (Vdefl * llt.matrixL().solve(MatrixType::Identity(GramH2.rows(), GramH2.cols()))).eval(); // now Zᵀ B Z = I
            H2.get_iterativeLinearSolverConfig().jacobi.deflationEigVecs = Vdefl;
            H2.get_iterativeLinearSolverConfig().jacobi.deflationEigInvs = Y.cwiseInverse();
        } else {
            H2.get_iterativeLinearSolverConfig().jacobi.deflationEigVecs = MatrixType();
            H2.get_iterativeLinearSolverConfig().jacobi.deflationEigInvs = VectorType();
        }
    }
    if(use_coarse_inner_preconditioner and status.iter >= 1 and T_evals.size() > 5l) {
        // We add a coarse preconditioning term to the block jacobi solver:
        //      M^{-1}_2lvl = M^{-1}_BJ + Z(Z*BZ)^{-1}Z*   (* means adjoint)
        // where
        //      - M^{-1}_BJ is the curren block jacobi preconditioner.
        //      - Z(Z*BZ)^{-1}Z* is the B-pseudo-inverse on span(Z)
        //      - Z is a tall B-orthonormal coarse basis that captures slow modes in the eigenvalue problem.
        //      - B is the matrix out of which you made the block jacobi: H2 in GDMRG, else H.
        //
        //
        const auto &BQ         = algo == OptAlgo::GDMRG ? H2Q : HQ;
        const auto &BV         = algo == OptAlgo::GDMRG ? H2V : HV;
        auto        nCoarse    = std::min(5l, T_evals.size());
        auto        nCoarseIdx = get_ritz_indices(ritz, 1, nCoarse, T_evals);

        auto &jcbCfg    = algo == OptAlgo::DMRG ? H1.get_iterativeLinearSolverConfig().jacobi : H2.get_iterativeLinearSolverConfig().jacobi;
        jcbCfg.coarseZ  = {};
        jcbCfg.coarseBZ = {};
        if(nCoarseIdx.size() > 0) {
            MatrixType Z = T_evecs(Eigen::placeholders::all, nCoarseIdx);
            VectorReal Y = T_evals(Eigen::placeholders::all, nCoarseIdx);

            eiglog->trace("coarsening idx {} | eigv {}", nCoarseIdx, fv(Y));

            MatrixType coarseZ  = Q * Z;
            MatrixType coarseBZ = BQ * Z;

            // Build Gv = Vᵀ B V and RHS = Vᵀ B Z
            MatrixType Gv  = V.adjoint() * BV;
            MatrixType RHS = V.adjoint() * coarseBZ;

            Eigen::LLT<MatrixType> lltV(Gv);
            if(lltV.info() == Eigen::Success) {
                MatrixType coeffs   = lltV.solve(RHS); // (VᵀBV)^{-1} (VᵀBZ)
                coarseZ.noalias()  -= V * coeffs;      // Z ← Z − V (VᵀBV)^{-1} Vᵀ B Z
                coarseBZ.noalias() -= BV * coeffs;     // BZ ← BZ − BV (VᵀBV)^{-1} Vᵀ B Z
            } else {
                eiglog->warn("LLTV failed to create the coarse operator from Gv");
                return;
            }
            MatrixType Gram = coarseZ.adjoint() * coarseBZ; // small p×p matrix

            Eigen::LLT<MatrixType> llt(Gram);
            if(llt.info() != Eigen::Success) {
                // tiny diagonal bump; skip coarse if it still fails
                Gram.diagonal().array() += RealScalar(1e-12);
                llt.compute(Gram);
                if(llt.info() != Eigen::Success) {
                    eiglog->warn("LLT failed to create the coarse operator from Gram");
                    jcbCfg.coarseZ  = {};
                    jcbCfg.coarseBZ = {};
                    return;
                }
            }
            const MatrixType Rinv = llt.matrixU().solve(MatrixType::Identity(Gram.rows(), Gram.cols()));
            jcbCfg.coarseZ        = coarseZ * Rinv;  // now Zᵀ B Z ≈ I
            jcbCfg.coarseBZ       = coarseBZ * Rinv; // keep BZ = B·Z consistent

            if constexpr(settings::debug_solver) {
                // Sanity checks
                MatrixType VtBZ = V.adjoint() * jcbCfg.coarseBZ; // Vᵀ B Z
                RealScalar leak = VtBZ.norm();
                eiglog->debug("[coarse] ‖VᵀBVZ‖    = {:.3e}", fp(leak));

                MatrixType Id        = jcbCfg.coarseZ.adjoint() * jcbCfg.coarseBZ; // Zᵀ B Z
                RealScalar ortho_err = (Id - MatrixType::Identity(Id.rows(), Id.cols())).norm();
                eiglog->debug("[coarse] ‖ZᵀBZ - I‖ = {:.3e}", fp(ortho_err));
            }
        }
    } else {
        auto &jcbCfg    = algo == OptAlgo::DMRG ? H1.get_iterativeLinearSolverConfig().jacobi : H2.get_iterativeLinearSolverConfig().jacobi;
        jcbCfg.coarseZ  = {};
        jcbCfg.coarseBZ = {};
    }
}

template<typename Scalar>
void solver_base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx); // Selected subspace eigenvectors
    VectorReal Y = T_evals(optIdx);                           // Selected subspace eigenvalues

    // Transform the basis
    V                   = Q * Z; // Regular Rayleigh-Ritz
    HV                  = HQ * Z;
    std::tie(S, rNorms) = get_residuals(Y, HV, V);
}

template<typename Scalar>
void solver_base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx); // Selected subspace eigenvectors
    VectorReal Y = T_evals(optIdx);                           // Selected subspace eigenvalues

    // Transform the basis
    V.noalias()         = Q * Z; // Regular Rayleigh-Ritz
    H1V.noalias()       = H1Q * Z;
    H2V.noalias()       = H2Q * Z;
    std::tie(S, rNorms) = get_residuals(Y, H1V, H2V);
}

/*!
 * Extract Ritz vectors, optionally performing refined Ritz extraction.
 * If chebyshev filtering is enabled, use the filtered basis (X/HX);
 * otherwise use the unfiltered basis (Q/HQ).
 * The refined Ritz extraction uses SVD to minimize the residual norm
 * in the projected subspace.
 */
template<typename Scalar>
void solver_base<Scalar>::extractRitzVectors() {
    if(status.stopReason != StopReason::none) return;
    if(T_evals.size() < b) return;
    auto t_extract = tid::tic_scope("extractRitzVectors");
    // Here we assume that Q is orthonormal.

    // Determine how many ritz indices to get
    Eigen::Index k     = std::min(maxPrevBlocks * b, T_evals.size());
    Eigen::Index nritz = std::max({nev, b, k});

    // Get the indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    status.optIdx = get_ritz_indices(ritz, 0, nritz, T_evals);

    if(use_refined_rayleigh_ritz) {
        // Refined extraction
        if(algo == OptAlgo::GDMRG) {
            refinedRitzVectors(status.optIdx, V, H1V, H2V, S, status.rNorms);
        } else {
            refinedRitzVectors(status.optIdx, V, HV, S, status.rNorms);
        }
    } else {
        if(algo == OptAlgo::GDMRG) {
            extractRitzVectors(status.optIdx, V, H1V, H2V, S, status.rNorms);
        } else {
            extractRitzVectors(status.optIdx, V, HV, S, status.rNorms);
        }
    }

    // Get the "prev" part
    K_prev = K;
    K      = V.leftCols(k);

    // Keep b columns
    if(k > b) {
        V.conservativeResize(Eigen::NoChange, b);
        if(algo == OptAlgo::GDMRG) {
            H1V.conservativeResize(Eigen::NoChange, b);
            H2V.conservativeResize(Eigen::NoChange, b);
        } else {
            HV.conservativeResize(Eigen::NoChange, b);
        }
        S.conservativeResize(Eigen::NoChange, b);
        status.rNorms.conservativeResize(b);
    }
}

template<typename Scalar>
solver_base<Scalar>::MatrixType solver_base<Scalar>::get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z,
                                                                                       const Eigen::Ref<const VectorReal> &Y, const MatrixType &H1Q,
                                                                                       const MatrixType &H2Q) {
    assert(algo == OptAlgo::GDMRG);
    // assert(static_cast<size_t>(V.cols()) == optIdx.size());
    assert(Z.cols() == Y.size());
    Eigen::JacobiSVD<MatrixType, Eigen::ComputeThinV> svd;
    MatrixType                                        Z_ref(Z.rows(), Z.cols());
    MatrixType                                        T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols()); // cache H2*zj
    for(Eigen::Index j = 0; j < Y.size(); ++j) {
        const auto &theta = Y(j);
        MatrixType  M     = (H1Q - theta * H2Q);

        svd.compute(M);

        if(svd.info() == Eigen::Success) {
            Eigen::Index min_idx;
            svd.singularValues().minCoeff(&min_idx);

            // Accept the solution
            auto zj   = Z_ref.col(j);
            auto t2zj = T2Z_ref.col(j);
            zj        = svd.matrixV().col(min_idx); // overwrite

            //----------------------------------------------------------------
            // orthogonalize zj against previously accepted columns
            //----------------------------------------------------------------
            if(use_h2_inner_product) {
                t2zj = T2 * zj; // T2 is b×b, cheap
            } else {
                t2zj = zj;
            }

            if(j > 0) {
                auto Z_prev   = Z_ref.leftCols(j);
                auto T2Z_prev = T2Z_ref.leftCols(j);

                MatrixType Gxx = Z_prev.adjoint() * T2Z_prev;
                Gxx            = (Gxx + Gxx.adjoint()).eval() * half;

                MatrixType Gxy = Z_prev.adjoint() * t2zj; // Gram between previous and current

                // Solve Gxx * w = g
                MatrixType W = Gxx.ldlt().solve(Gxy);

                // Project out
                zj.noalias()   -= Z_prev * W;
                t2zj.noalias() -= T2Z_prev * W;
            }

            // for(Eigen::Index i = 0; i < j; ++i) {
            //     auto   zi   = Z_ref.col(i);
            //     auto   t2zi = T2Z_ref.col(i);
            //     Scalar proj = zi.dot(t2zj); // (z_i)† T2 zj
            //     zj.noalias() -= zi * proj;
            //     t2zj.noalias() -= t2zi * proj; // keep cache consistent
            // }

            //-----------------------------------------------------------------------------------------------------
            // Normalize w.r.t. T2-norm  ‖z‖_{2} = sqrt(abs(zj.adjoint()*T2*zj)) (when using the H2 inner product)
            //----------------------------------------------------------------------------------------------------
            RealScalar norm = std::sqrt(std::max<RealScalar>(0, std::real(zj.dot(t2zj))));
            if(norm < normTol) { // * get_op_norm_estimate()) {
                // Column numerically null → zero-out but keep slot
                zj.setZero();
                t2zj.setZero();
                continue;
            }
            zj   /= norm;
            t2zj /= norm;

        } else {
            Z_ref.col(j) = Z.col(j);
            eiglog->warn("refinement failed on ritz vector {} | info {} ", j, static_cast<int>(svd.info()));
        }
    }
    return Z_ref;
}
//

template<typename Scalar>
std::pair<typename solver_base<Scalar>::MatrixType, typename solver_base<Scalar>::MatrixType>
    solver_base<Scalar>::get_h2_normalizer_for_the_projected_pencil(const MatrixType &T2) {
    MatrixType T2h = (T2 + T2.adjoint()) / RealScalar{2};

    auto es = Eigen::SelfAdjointEigenSolver<MatrixType>(T2h, Eigen::ComputeEigenvectors);
    if(es.info() != Eigen::Success) throw except::runtime_error("get_h2_normalizer_for_the_projected_pencil: eigensolver failed");

    auto U = es.eigenvectors();
    auto D = es.eigenvalues();

    const RealScalar Dmax = std::max<RealScalar>(RealScalar{1}, D.cwiseAbs().maxCoeff());
    const RealScalar tau  = RealScalar{10} * eps * Dmax;

    if(D.minCoeff() <= RealScalar{0}) { eiglog->warn("Projected T2 is numerically indefinite: min eval {:.3e}", fp(D.minCoeff())); }

    for(Eigen::Index k = 0; k < D.size(); ++k) D(k) = std::max(D(k), tau);

    return {U * D.cwiseInverse().cwiseSqrt().asDiagonal() * U.adjoint(), //
            U * D.cwiseSqrt().asDiagonal() * U.adjoint()};
}

template<typename Scalar>
solver_base<Scalar>::MatrixType solver_base<Scalar>::get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1,
                                                                                      const MatrixType &T2) {
    assert(Z_rr.size() > 0);
    assert(Z_rr.rows() == Z_ref.rows());
    assert(Z_rr.cols() == Z_ref.cols());
    assert(Z_rr.rows() == T1.rows());
    assert(Z_rr.rows() == T2.rows());
    MatrixType Z(Z_rr.rows(), Z_rr.cols());

    // Symmetrize
    MatrixType T1h = (T1.adjoint() + T1) * half;
    MatrixType T2h = (T2.adjoint() + T2) * half;

    MatrixType I = MatrixType::Identity(2, 2);
    for(Eigen::Index k = 0; k < Z.cols(); ++k) {
        using M2Type = Eigen::Matrix<Scalar, 2, 2>;
        M2Type     A(2, 2), B(2, 2);
        VectorType z0 = Z_rr.col(k);
        VectorType z1 = Z_ref.col(k);
        A(0, 0)       = z0.adjoint() * T1h * z0;
        A(1, 0)       = z1.adjoint() * T1h * z0;
        A(0, 1)       = z0.adjoint() * T1h * z1;
        A(1, 1)       = z1.adjoint() * T1h * z1;

        B(0, 0) = z0.adjoint() * T2h * z0;
        B(1, 0) = z1.adjoint() * T2h * z0;
        B(0, 1) = z0.adjoint() * T2h * z1;
        B(1, 1) = z1.adjoint() * T2h * z1;

        // Make sure B is positive definite
        // RealScalar tau = 10 * eps * std::max(RealScalar{1}, WT2W.norm());
        RealScalar tau  = 10 * eps * std::max(RealScalar{1}, std::real(B.trace()) * half);
        B              += I * tau;

        // Symmetrize
        A = (A.adjoint() + A) * half;
        B = (B.adjoint() + B) * half;

        auto ges = Eigen::GeneralizedSelfAdjointEigenSolver<M2Type>(A, B, Eigen::Ax_lBx);
        if(ges.info() == Eigen::Success) {
            auto select1 = get_ritz_indices(ritz, 0, 1, ges.eigenvalues());
            auto v       = ges.eigenvectors().col(select1.at(0));
            Z.col(k)     = z0 * v(0) + z1 * v(1);
        } else {
            eiglog->warn("ges failed");
            eiglog->warn("A \n{}", linalg::matrix::to_string(A, 8));
            eiglog->warn("B \n{}", linalg::matrix::to_string(B, 8));
            Z.col(k) = z0; // Default to RR in case of failure
        }
    }

    orthonormalize_Z(Z, T2h);

    return Z;
}

template<typename Scalar>
solver_base<Scalar>::MatrixType solver_base<Scalar>::get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z,
                                                                                       const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                                                       const MatrixType &HQ) {
    assert(algo != OptAlgo::GDMRG);
    assert(Z.cols() == Y.size());
    Eigen::JacobiSVD<MatrixType, Eigen::ComputeThinV> svd;
    MatrixType                                        Z_ref(Z.rows(), Z.cols());
    MatrixType                                        T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols()); // cache H2*zj
    for(Eigen::Index j = 0; j < Y.size(); ++j) {
        const auto &theta = Y(j);
        MatrixType  M     = HQ - theta * Q;
        svd.compute(M);

        Eigen::Index min_idx;
        svd.singularValues().minCoeff(&min_idx);

        if(svd.info() == Eigen::Success) {
            // Accept the solution
            Z_ref.col(j) = svd.matrixV().col(min_idx);
        } else {
            Z_ref.col(j)            = Z.col(j);
            RealScalar refinedRnorm = svd.singularValues()(min_idx);
            eiglog->warn("refinement failed on ritz vector {} | refined rnorm={:.5e} | info {} ", j, fp(refinedRnorm), static_cast<int>(svd.info()));
        }
    }
    return Z_ref;
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    assert(algo == OptAlgo::GDMRG);
    VectorReal Y     = T_evals(optIdx);
    MatrixType Z_rr  = T_evecs(Eigen::placeholders::all, optIdx);
    MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, H1Q, H2Q);
    MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, T1, T2); // Gives an optimal combination of Z_rr and Z_ref

    // if(algo == OptAlgo::GDMRG) {
    //     Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
    //     auto                                      T2idx   = get_ritz_indices(OptRitz::SM, 0, optIdx.size(), es2.eigenvalues());
    //     VectorReal                                YT2     = es2.eigenvalues()(T2idx);
    //     MatrixType                                ZT2     = es2.eigenvectors()(Eigen::placeholders::all, T2idx);
    //     VectorType                                V_opt   = Q * Z_opt;
    //     VectorType                                H2V_opt = H2Q * Z_opt;
    //     VectorReal                                YT2_opt = (V_opt.adjoint() * H2V_opt).diagonal().real();
    //
    //     for(Eigen::Index idx = 0; idx < YT2.size(); ++idx) {
    //         if(YT2(idx) < YT2_opt(idx)) {
    //             eiglog->info("idx {} | T2 eval {:.16f} -> {:.16f}", idx, fp(YT2_opt(idx)), fp(YT2(idx)));
    //             Z_opt(Eigen::placeholders::all, idx) = ZT2(Eigen::placeholders::all, idx);
    //         }
    //     }
    // }

    // Transform
    V.noalias()   = Q * Z_opt;
    H1V.noalias() = H1Q * Z_opt;
    H2V.noalias() = H2Q * Z_opt;

    if(use_rayleigh_quotients_instead_of_evals) {
        // We replace the eigenvalues in T_evals by their rayleigh quotients, to sync V and Y in the residual vector calculation
        VectorReal rq1  = (V.adjoint() * H1V).diagonal().real();
        VectorReal rq2  = (V.adjoint() * H2V).diagonal().real();
        T_evals(optIdx) = rq1.cwiseQuotient(rq2);
        Y               = T_evals(optIdx);
    }

    std::tie(S, rNorms) = get_residuals(Y, H1V, H2V);
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    MatrixType                   Z     = T_evecs(Eigen::placeholders::all, optIdx);
    VectorReal                   Y     = T_evals(optIdx);
    MatrixType                   Z_ref = get_refined_ritz_eigenvectors_std(Z, Y, Q, HQ);

    // Transform the basis with applied operators
    V  = Q * Z_ref;
    HV = HQ * Z_ref;

    if(use_rayleigh_quotients_instead_of_evals) {
        // We replace the eigenvalues in T_evals by their rayleigh quotients
        Y = (V.adjoint() * HV).diagonal().real();
    }
    std::tie(S, rNorms) = get_residuals(Y, HV, V);
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors() {
    if(!use_refined_rayleigh_ritz) return;
    if(status.rNorms.size() == 0) throw except::runtime_error("refineRitzVectors() called before extractRitzVectors()");
    auto t_refined = tid::tic_scope("refinedRitzVectors");
    // Refined extraction
    if(algo == OptAlgo::GDMRG) {
        refinedRitzVectors(status.optIdx, V, H1V, H2V, S, status.rNorms);
    } else {
        refinedRitzVectors(status.optIdx, V, HV, S, status.rNorms);
    }
}

template<typename Scalar>
void solver_base<Scalar>::preamble() {
    // Prepare for the next iteration
    status.num_iters_inner_prev = status.num_iters_inner;
    status.num_matvecs          = 0;
    status.num_precond          = 0;
    status.num_iters_inner      = 0;
    status.num_matvecs_inner    = 0;
    status.num_precond_inner    = 0;
    status.num_jdops_inner      = 0;

    status.inner_error_last = RealScalar{0};
    status.inner_tol_last   = RealScalar{0};

    status.time_jdops_inner.reset();
    status.time_jacobi_inner.reset();
    status.time_chebyshev_inner.reset();

    status.time_matvecs.reset();
    status.time_precond.reset();
    status.time_matvecs_inner.reset();
    status.time_precond_inner.reset();

    adjust_preconditioner_tolerance(S);
    adjust_residual_correction_type();
    adjust_preconditioner_H1_limits();
    adjust_preconditioner_H2_limits();
}

template<typename Scalar>
void solver_base<Scalar>::updateStatus() {
    // Accumulate counters from the inner solvre
    status.num_matvecs_total  += status.num_matvecs + status.num_matvecs_inner;
    status.num_precond_total  += status.num_precond + status.num_precond_inner;
    status.time_matvecs_total += status.time_matvecs.get_time() + status.time_matvecs_inner.get_time();
    status.time_precond_total += status.time_precond.get_time() + status.time_precond_inner.get_time();

    // Eigenvalues are sorted in ascending order.
    status.oldVal  = status.eigVal.topRows(nev);
    status.eigVal  = T_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
    status.absDiff = (status.eigVal - status.oldVal).cwiseAbs();

    VectorReal denom = (RealScalar{0.5} * (status.eigVal + status.oldVal).array().abs()).matrix();
    denom            = denom.cwiseMax(VectorReal::Constant(denom.size(), std::numeric_limits<RealScalar>::min()));
    status.relDiff   = status.absDiff.cwiseQuotient(denom);

    status.rNorms_history.push_back(status.rNorms.topRows(nev));
    status.eigVals_history.push_back(status.eigVal.topRows(nev));
    status.matvecs_history.push_back(status.num_matvecs + status.num_matvecs_inner);
    while(status.rNorms_history.size() > status.max_history_size) status.rNorms_history.pop_front();
    while(status.eigVals_history.size() > status.max_history_size) status.eigVals_history.pop_front();
    while(status.matvecs_history.size() > status.max_history_size) status.matvecs_history.pop_front();
    if(eigVals_have_saturated())
        status.saturation_count_eigVal++;
    else
        status.saturation_count_eigVal = 0;

    if(rNorms_have_saturated())
        status.saturation_count_rNorm++;
    else
        status.saturation_count_rNorm = 0;

    constexpr auto beta   = RealScalar{0.5f};
    VectorReal     rNorms = status.rNorms.topRows(nev); // The current residual norms
    RealScalar     relGap = status.gap * get_op_norm_estimate();
    if(rNorms.size() != rNormTols().size()) throw except::logic_error("unequal sizes");
    status.rNorm_below_rnormTol = (rNorms.array() < rNormTols().array()).all(); // Residual norm condition
    status.rNorm_below_gap      = rNorms.maxCoeff() < beta * relGap;            // Gap condition for the currently selected operator (H1, H2, or H1/H2)
    // tools::log->info("rNorms {::.3e} | rNormTols {::.3e} | relGap {:.5e}", fv(status.rNorms),  fv(rNormTols()), fp(relGap));

    if(status.rNorm_below_rnormTol) {
        std::string msg_rnorm_gap = fmt::format(" | gap {:.3e} (rel {:.3e})", fp(status.gap), fp(relGap));
        if constexpr(settings::debug_solver) {
            if(algo == OptAlgo::GDMRG and dev_append_extra_blocks_to_basis) {
                msg_rnorm_gap = fmt::format(" | H1|H2: norm {:.2e}|{:.2e}", fp(H1.get_op_norm()), fp(H2.get_op_norm()));
            }
        }
        status.stopMessage.emplace_back(fmt::format("converged rNorm {::.3e} < tol {::.3e}{} | iters {} | mv {} | {:.3e} s",
                                                    fv(VectorReal(status.rNorms.topRows(nev))), fv(rNormTols()), msg_rnorm_gap, status.iter + 1,
                                                    status.num_matvecs_total, status.time_elapsed.get_time()));
        status.stopReason |= StopReason::converged_rNorms;
    }

    if(max_iters >= 0l and status.iter + 1 >= max_iters) {
        status.stopMessage.emplace_back(
            fmt::format("iters ({}) >= maxiter ({}) | mv {} | {:.3e} s", status.iter + 1, max_iters, status.num_matvecs_total, status.time_elapsed.get_time()));
        status.stopReason |= StopReason::max_iterations;
    }
    if(max_matvecs >= 0l and status.num_matvecs_total >= max_matvecs) {
        status.stopMessage.emplace_back(
            fmt::format("num_matvecs_total ({}) >= max_matvecs ({}) | {:.3e} s", status.num_matvecs_total, max_matvecs, status.time_elapsed.get_time()));
        status.stopReason |= StopReason::max_matvecs;
    }

    if(std::min(status.saturation_count_eigVal, status.saturation_count_rNorm) >= status.saturation_count_max) {
        status.stopMessage.emplace_back(fmt::format("saturation_count (eigVal {} rNorm {}) >= saturation_count_max ({}) | it {} | mv {} | {:.3e} s",
                                                    status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max, status.iter + 1,
                                                    status.num_matvecs_total, status.time_elapsed.get_time()));
        status.stopReason |= StopReason::saturated_eigVals;
        status.stopReason |= StopReason::saturated_rNorms;
    } else if(status.saturation_count_eigVal >= status.saturation_count_max * 2) {
        status.stopMessage.emplace_back(fmt::format("saturation_count eigVal {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                    status.saturation_count_eigVal, status.saturation_count_max, status.iter + 1, status.num_matvecs_total,
                                                    status.time_elapsed.get_time()));
        status.stopReason |= StopReason::saturated_eigVals;
    } else if(status.saturation_count_eigVal > 2 and status.saturation_count_rNorm >= status.saturation_count_max * 2) {
        // Probably eigVal is stuck in some kind of cycle.
        status.stopMessage.emplace_back(fmt::format("saturation_count rNorm {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                    status.saturation_count_rNorm, status.saturation_count_max, status.iter + 1, status.num_matvecs_total,
                                                    status.time_elapsed.get_time()));
        status.stopReason |= StopReason::saturated_rNorms;
    }
}

template<typename Scalar>
void solver_base<Scalar>::printStatus() {
    std::string msg_rnorm_gap = fmt::format(" | gap {:.3e}", fp(status.gap));
    if constexpr(settings::debug_solver) {
        if(algo == OptAlgo::GDMRG) { msg_rnorm_gap = fmt::format(" | H1|H2: norm {:.2e}|{:.2e}", fp(status.T1_max_eval), fp(status.T2_max_eval)); }
    }

    std::string rCorrMsg;
    switch(residual_correction_type_internal) {
        case ResidualCorrectionType::NONE: rCorrMsg = "NO"; break;
        case ResidualCorrectionType::CHEAP_OLSEN: rCorrMsg = "CO"; break;
        case ResidualCorrectionType::FULL_OLSEN: rCorrMsg = "FO"; break;
        case ResidualCorrectionType::JACOBI_DAVIDSON: rCorrMsg = "JD"; break;
        case ResidualCorrectionType::AUTO: rCorrMsg = "AU"; break;
    }
    std::string innerMsg;
    if(status.num_matvecs_inner > 0 || status.num_jdops_inner > 0 || status.num_precond_inner > 0) {
        innerMsg = fmt::format("[inner: ({}) mv {:5} jd {:5} pc {:5} err {:.2e} tol {:.2e} "
                               "mv {:.1e}s jd {:.1e}s pc {:.1e}s] ",
                               rCorrMsg,                                 //
                               status.num_matvecs_inner,                 //
                               status.num_jdops_inner,                   //
                               status.num_precond_inner,                 //
                               fp(status.inner_error_last),              //
                               fp(status.inner_tol_last),                //
                               fp(status.time_matvecs_inner.get_time()), //
                               fp(status.time_jdops_inner.get_time()),   //
                               fp(status.time_precond_inner.get_time()));
    }
    MatrixType Gram       = use_h2_inner_product ? Q.adjoint() * H2Q : Q.adjoint() * Q;
    Gram                  = (Gram + Gram.adjoint()).eval() / RealScalar{2};
    RealScalar  orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
    std::string evMsg;
    if(algo == OptAlgo::GDMRG) {
        VectorReal VH1V = (V.adjoint() * H1V).real();
        VectorReal VH2V = (V.adjoint() * H2V).real();
        evMsg           = fmt::format(" {::.16f} / {::.16f}", fv(VH1V), fv(VH2V));
    }

    // bool log_low_maxiter = max_iters < 10;
    // bool log_jacobi_prec   = preconditioner_type == eig::Preconditioner::JACOBI and status.iter % 100 == 0;
    // bool log_solve_prec    = preconditioner_type == eig::Preconditioner::SOLVE;
    bool                      log_long_time    = last_log_time.get_lap() > 10.0;
    bool                      log_every_ten_it = (status.iter + 1) % 10 == 0;
    spdlog::level::level_enum loglevel         = eiglog->level();
    if(loglevel < spdlog::level::info and (log_every_ten_it or log_long_time)) loglevel = spdlog::level::info;
    if(loglevel >= eiglog->level()) {
        [[maybe_unused]] auto lap = last_log_time.restart_lap();
        eiglog->log(loglevel,
                    "it {:3} mv {:3} pc {:3} t {:.1e}s dim {}={} {}"
                    "eigVal {::.16f}{} "
                    "oErr {:.3e} rNorms {::.8e} rNormTol {::.3e} tol {:.2e} (rel {:.2e}) "
                    "({:9.2e}/mv) sat {}:{}/{} col {:2} b {} ritz {} "
                    "op norm {:.2e} cond {:.2e} sens {:.2e}{}",
                    status.iter + 1,                   //
                    status.num_matvecs,                //
                    status.num_precond,                //
                    status.time_elapsed.restart_lap(), //
                    mps_shape,                         //
                    N,                                 //
                    innerMsg,                          //
                    fv(status.eigVal),                 //
                    evMsg,                             //
                    fp(orthError),                     //
                    // fv(VectorReal(status.rNorms.topRows(nev))), //
                    fv(VectorReal(status.rNorms)),            //
                    fv(rNormTols()),                          //
                    fp(abstol),                               //
                    fp(reltol),                               //
                    fp(get_rNorms_log10_change_per_matvec()), //
                    status.saturation_count_eigVal,           //
                    status.saturation_count_rNorm,            //
                    status.saturation_count_max,              //
                    Q.cols(),                                 //
                    b,                                        //
                    enum2sv(ritz),                            //
                    fp(status.op_norm_estimate),              //
                    fp(status.condition),                     //
                    fp(status.sensitivity),                   //
                    msg_rnorm_gap);
    }
}

template<typename Scalar>
void solver_base<Scalar>::set_maxPrevBlocks(Eigen::Index pb) {
    b  = std::min(std::max(nev, b), N / 2);
    pb = std::min<Eigen::Index>(pb, N / b);
    if(pb != maxPrevBlocks) eiglog->trace("gdplusk: maxPrevBlocks = {}", pb);
    maxPrevBlocks = pb;
}

template<typename Scalar>
void solver_base<Scalar>::debug_check_H2_symmetry(int nsamples) {
    if constexpr(settings::debug_solver) {
        using L     = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
        using R     = decltype(std::real(std::declval<L>()));
        auto dot_hp = [](const VectorType &a, const VectorType &b) -> L { return (a.template cast<L>().dot(b.template cast<L>())); };

        for(int s = 0; s < nsamples; ++s) {
            VectorType u = VectorType::Random(mps_size);
            VectorType v = VectorType::Random(mps_size);
            // Apply H2 in the same way your solver does.
            VectorType H2u = MultH2(u); // use hp here to isolate "operator" not "matvec noise"
            VectorType H2v = MultH2(v);

            L p1 = dot_hp(u, H2v);
            L p2 = dot_hp(H2u, v);

            R denom = std::max<R>(u.norm() * H2v.norm() + H2u.norm() * v.norm(), R(1e-300));
            R rel   = std::abs(p1 - p2) / denom;

            eiglog->info("H2 symmetry test sample {}: |u| = {:.3e} |v| = {:.3e} |H2u| = {:.3e} |H2v| = {:.3e} |p1-p2|/(|u|*|H2v| + |H2u|*|v|) = {:.3e}", s,
                         fp(u.norm()), fp(v.norm()), fp(H2u.norm()), fp(H2v.norm()), fp(rel));
        }
    }
}

template<typename Scalar>
void solver_base<Scalar>::debug_check_H2_symmetry(const MatrixType &Y, int nsamples) {
    if constexpr(settings::debug_solver) {
        using L     = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
        using R     = decltype(std::real(std::declval<L>()));
        auto dot_hp = [](const VectorType &a, const VectorType &b) -> L { return (a.template cast<L>().dot(b.template cast<L>())); };

        for(int s = 0; s < nsamples; ++s) {
            for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                auto       v       = Y.col(j);
                VectorType H2v     = MultH2(v);
                RealScalar vnorm   = v.norm();
                RealScalar H2vnorm = H2v.norm();
                for(Eigen::Index i = 0; i < j; ++i) {
                    auto       u     = Y.col(i);
                    RealScalar unorm = u.norm();
                    // Apply H2 in the same way your solver does.
                    VectorType H2u     = MultH2(u); // use hp here to isolate "operator" not "matvec noise"
                    RealScalar H2unorm = H2u.norm();
                    L          p1      = dot_hp(u, H2v);
                    L          p2      = dot_hp(H2u, v);

                    R denom = std::max<R>(u.norm() * H2v.norm() + H2u.norm() * v.norm(), R{1e-300});
                    R rel   = std::abs(p1 - p2) / denom;

                    eiglog->info("H2 symmetry test sample {}: |u| = {:.3e} |v| = {:.3e} |H2u| = {:.3e} |H2v| = {:.3e} |p1-p2|/(|u|*|H2v| + |H2u|*|v|) = {:.3e}",
                                 s, fp(unorm), fp(vnorm), fp(H2unorm), fp(H2vnorm), fp(rel));
                }
            }
        }
    }
}

//
//
// auto gemm_highprecision_fp80 = [](const Eigen::Ref<const MatrixType> &A_in, const Eigen::Ref<const MatrixType> &B_in, Eigen::Index BK) -> MatrixType {
//     // Multiply in FP64, accumulate in long double, return FP64.
//     // - If BK == 1: do scalar FMAs (double multiply) into long double accumulator.
//     // - If BK  > 1: do GEMM in double for each k-block, then add that block result into long double accumulator.
//     //
//     // Requirements: A.cols() == B.rows().
//
//     const Eigen::Index m = A_in.rows();
//     const Eigen::Index k = A_in.cols();
//     const Eigen::Index n = B_in.cols();
//
//     assert(B_in.rows() == A_in.cols());
//     assert(BK >= 1);
//     using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
//
//     // long double accumulator (no upcast of A/B storage; only the running sum is long double)
//     Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> acc(m, n);
//     acc.setZero();
//
//     // Special case: BK == 1 uses scalar updates (double multiply, long double add)
//     if(BK == 1) {
//         // Access as plain matrices (still views, no copy)
//         const auto &A   = A_in.derived();
//         const auto &B   = B_in.derived();
//         auto        Bkk = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(n);
//         for(Eigen::Index kk = 0; kk < k; ++kk) {
//             const auto a_col = A.col(kk); // length m
//             Bkk              = B.row(kk);
//             for(Eigen::Index j = 0; j < n; ++j) { acc.col(j).noalias() += (a_col * Bkk(j)).template cast<ScalarL>(); }
//         }
//         return acc.template cast<Scalar>();
//     }
//
//     // General case: BK > 1
//     // Reusable FP64 buffer for each block contribution.
//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> P(m, n);
//     P.setZero();
//
//     for(Eigen::Index kk = 0; kk < k; kk += BK) {
//         const Eigen::Index kb = std::min<Eigen::Index>(BK, k - kk);
//
//         P.noalias() = A_in.middleCols(kk, kb) * B_in.middleRows(kk, kb); // FP64 block GEMM
//
//         acc.noalias() += P.template cast<ScalarL>(); // Accumulate block result in long double
//     }
//
//     // Downcast final result back to FP64
//     return acc.template cast<Scalar>();
// };
// auto gemm_highprecision_fp128 = [](const Eigen::Ref<const MatrixType> &A_in, const Eigen::Ref<const MatrixType> &B_in, Eigen::Index BK) -> MatrixType {
//     // Multiply in FP64, accumulate in long double, return FP64.
//     // - If BK == 1: do scalar FMAs (double multiply) into long double accumulator.
//     // - If BK  > 1: do GEMM in double for each k-block, then add that block result into long double accumulator.
//     //
//     // Requirements: A.cols() == B.rows().
//
//     const Eigen::Index m = A_in.rows();
//     const Eigen::Index k = A_in.cols();
//     const Eigen::Index n = B_in.cols();
//
//     assert(B_in.rows() == A_in.cols());
//     assert(BK >= 1);
//     using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
//
//     // long double accumulator (no upcast of A/B storage; only the running sum is long double)
//     Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic> acc(m, n);
//     acc.setZero();
//
//     // Special case: BK == 1 uses scalar updates (double multiply, long double add)
//     if(BK == 1) {
//         // Access as plain matrices (still views, no copy)
//         const auto &A = A_in.derived();
//         const auto &B = B_in.derived();
//
//         for(Eigen::Index kk = 0; kk < k; kk += BK) {
//             const Eigen::Index kb = std::min<Eigen::Index>(BK, k - kk);
//             // For each (i,j): compute the dot over the current k-block in FP64,
//             // then accumulate into long double.
//             for(Eigen::Index i = 0; i < m; ++i) {
//                 const auto a_seg = A_in.row(i).segment(kk, kb); // row segment view: length kb
//                 for(Eigen::Index j = 0; j < n; ++j) {
//                     const auto b_seg = B_in.col(j).segment(kk, kb);      // col segment view: length kb
//                     acc(i, j) += static_cast<ScalarL>(a_seg.dot(b_seg)); // dot() is FP64, the add is long double
//                 }
//             }
//         }
//
//         return acc.template cast<Scalar>();
//     }
//
//     // General case: BK > 1
//     // Reusable FP64 buffer for each block contribution.
//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> P(m, n);
//
//     for(Eigen::Index kk = 0; kk < k; kk += BK) {
//         const Eigen::Index kb = std::min<Eigen::Index>(BK, k - kk);
//
//         P.noalias() = A_in.middleCols(kk, kb) * B_in.middleRows(kk, kb); // FP64 block GEMM
//
//         acc.noalias() += P.template cast<ScalarL>(); // Accumulate block result in long double
//     }
//
//     // Downcast final result back to FP64
//     return acc.template cast<Scalar>();
// };
//
// auto applyH2_highprecision1 = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
//     const auto &envL         = H2.get_envL();
//     const auto &envR         = H2.get_envR();
//     const auto &mpo2         = H2.get_mpos().front(); // From double layer of mpos
//     auto        get_positive = [](Scalar val) {
//         if constexpr(std::is_same_v<Scalar, RealScalar>) { // Real values
//             return val > RealScalar{0} ? val : RealScalar{0};
//         } else { // Complex values (how should this be resolved? Perhaps magnitude above or below 1?)
//             return std::real(val) > RealScalar{0} ? val : RealScalar{0};
//         }
//     };
//     auto get_negative = [](Scalar val) {
//         if constexpr(std::is_same_v<Scalar, RealScalar>) { // Real values
//             return val < RealScalar{0} ? val : RealScalar{0};
//         } else { // Complex values (how should this be resolved? Perhaps magnitude above or below 1?)
//             return std::real(val) > RealScalar{0} ? val : RealScalar{0};
//         }
//     };
//
//     auto envL_split = std::array<Eigen::Tensor<Scalar, 3>, 2>{envL.unaryExpr(get_positive), envL.unaryExpr(get_negative)};
//     auto envR_split = std::array<Eigen::Tensor<Scalar, 3>, 2>{envR.unaryExpr(get_positive), envR.unaryExpr(get_negative)};
//     auto mpo2_split = std::array<Eigen::Tensor<Scalar, 4>, 2>{mpo2.unaryExpr(get_positive), mpo2.unaryExpr(get_negative)};
//
//     // using ScalarL              = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<double>, double>;
//     // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
//
//     using AccScalar = std::conditional_t<std::is_same_v<Scalar, RealScalar>, long double, std::complex<long double>>;
//     using VectorAcc = Eigen::Matrix<AccScalar, Eigen::Dynamic, 1>;
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto tmp = Eigen::Tensor<Scalar, 3>(mps_shape);                                          // Temporary for accumulating
//
//         auto      mps_split = std::array<Eigen::Tensor<Scalar, 3>, 2>{mps.unaryExpr(get_positive), mps.unaryExpr(get_negative)};
//         VectorAcc accv      = VectorAcc::Zero(mps_size);
//
//         for(const auto &envL_half : envL_split) {
//             for(const auto &envR_half : envR_split) {
//                 for(const auto &mpo2_half : mpo2_split) {
//                     for(const auto &mps_half : mps_split) {
//                         tools::common::contraction::matrix_vector_product(tmp, mps_half, mpo2_half, envL_half, envR_half);
//                         // HERE WE HAVE A CHANCE TO ADD tmp TO res in a more accurate way!
//
//                         auto tmpv = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tmp.data(), mps_size);
//                         accv += tmpv.template cast<AccScalar>();
//                     }
//                 }
//             }
//         }
//         auto resv = Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1>>(Y.col(i).data(), mps_size); // data out
//         resv      = accv.template cast<Scalar>();
//     }
//
//     return Y;
// };
// auto applyH2_highprecision3 = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
//     using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
//     // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
//     using MatrixTypeL                    = Eigen::Matrix<ScalarL, Eigen::Dynamic, Eigen::Dynamic>;
//     const Eigen::Tensor<ScalarL, 3> envL = H2.get_envL().template cast<ScalarL>();
//     const Eigen::Tensor<ScalarL, 3> envR = H2.get_envR().template cast<ScalarL>();
//     const Eigen::Tensor<ScalarL, 4> mpo2 = H2.get_mpos().front().template cast<ScalarL>(); // From double layer of mpos
//     const MatrixTypeL               XL   = X.template cast<ScalarL>();
//     MatrixTypeL                     Y(X.rows(), X.cols()); // Result
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<ScalarL, 3>>(XL.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<ScalarL, 3>>(Y.col(i).data(), mps_shape);        // data out
//         tools::common::contraction::matrix_vector_product(res, mps, mpo2, envL, envR);
//     }
//
//     return Y.template cast<Scalar>(); // Downcast
// };
// auto applyH2_highprecisionL = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply envR last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//     eiglog->info("max envL    : {:.16e}", fp(get_max(envL)));
//     eiglog->info("max envR    : {:.16e}", fp(get_max(envR)));
//     eiglog->info("max mpo2    : {:.16e}", fp(get_max(mpo2)));
//
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         {
//             Eigen::Tensor<Scalar, 4> mpsenvL(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
//             Eigen::Tensor<Scalar, 4> mpsenvLmpo2(mps.dimension(2), envL.dimension(1), mpo2.dimension(1), mpo2.dimension(3));
//             mpsenvL.device(*threads->dev)     = mps.contract(envL, tenx::idx({1}, {0}));
//             mpsenvLmpo2.device(*threads->dev) = mpsenvL.contract(mpo2, tenx::idx({3, 0}, {0, 2}));
//             res.device(*threads->dev)         = mpsenvLmpo2.contract(envR, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
//             eiglog->info("L: i={:2} max mps: {:20.16e} T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mps)), fp(get_max(mpsenvL)),
//                          fp(get_max(mpsenvLmpo2)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionR = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply envL last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         {
//             Eigen::Tensor<Scalar, 4> mpsenvR(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
//             Eigen::Tensor<Scalar, 4> mpsenvRmpo(mps.dimension(1), envR.dimension(1), mpo2.dimension(0), mpo2.dimension(3));
//             mpsenvR.device(*threads->dev)    = mps.contract(envR, tenx::idx({2}, {0}));
//             mpsenvRmpo.device(*threads->dev) = mpsenvR.contract(mpo2, tenx::idx({3, 0}, {1, 2}));
//             res.device(*threads->dev)        = mpsenvRmpo.contract(envL, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 2, 0});
//             eiglog->info("R: i={:2} max mps: {:20.16e} T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mps)), fp(get_max(mpsenvR)),
//                          fp(get_max(mpsenvRmpo)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionM = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply mpo2 last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto                     mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto                     res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         auto                     mps_max = get_max(mps);
//         Eigen::Tensor<Scalar, 3> nps     = mps * mps.constant(RealScalar{1} / mps_max);
//         {
//             Eigen::Tensor<Scalar, 4> mpsenvR(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
//             Eigen::Tensor<Scalar, 5> mpsenvRenvL(mps.dimension(0), envR.dimension(1), envR.dimension(2), envL.dimension(1), envL.dimension(2));
//             mpsenvR.device(*threads->dev)     = nps.contract(envR, tenx::idx({2}, {0}));
//             mpsenvRenvL.device(*threads->dev) = mpsenvR.contract(envL, tenx::idx({1}, {0}));
//             res.device(*threads->dev)         = mpsenvRenvL.contract(mpo2, tenx::idx({0, 2, 4}, {2, 1, 0})).shuffle(tenx::array3{2, 1, 0});
//             res *= res.constant(mps_max);
//             eiglog->info("M: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsenvR)), fp(get_max(mpsenvRenvL)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionN = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // QR, apply mpo2 last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//     // Split environments with Householder QR along the virtual bond index
//     auto envL_map = Eigen::Map<const MatrixType>(envL.data(), envL.dimension(0) * envL.dimension(1), envL.dimension(2));
//     auto envR_map = Eigen::Map<const MatrixType>(envR.data(), envR.dimension(0) * envR.dimension(1), envR.dimension(2));
//
//     auto hhqrL = Eigen::HouseholderQR<MatrixType>(envL_map);
//     auto hhqrR = Eigen::HouseholderQR<MatrixType>(envR_map);
//
//     MatrixType qnvL_matrix = hhqrL.householderQ().setLength(envL_map.cols()) * MatrixType::Identity(envL_map.rows(), envL_map.cols()); //
//     MatrixType qnvR_matrix = hhqrR.householderQ().setLength(envR_map.cols()) * MatrixType::Identity(envR_map.rows(), envR_map.cols()); //
//
//     MatrixType rnvL_matrix = hhqrL.matrixQR().topLeftCorner(envL_map.cols(), envL_map.cols()).template triangularView<Eigen::Upper>(); // B
//     MatrixType rnvR_matrix = hhqrR.matrixQR().topLeftCorner(envR_map.cols(), envR_map.cols()).template triangularView<Eigen::Upper>(); // B
//
//     // Multiply the mpo from both sides by the "R" matrix coming from QR.
//     auto qnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvL_matrix.data(), envL.dimension(0), envL.dimension(1), qnvL_matrix.cols());
//     auto qnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvR_matrix.data(), envR.dimension(0), envR.dimension(1), qnvR_matrix.cols());
//     auto rnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvL_matrix.data(), rnvL_matrix.rows(), rnvL_matrix.cols());
//     auto rnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvR_matrix.data(), rnvR_matrix.rows(), rnvR_matrix.cols());
//
//     Eigen::Tensor<Scalar, 4> qpo2 =
//         rnvL.contract(mpo2, tenx::idx({1}, {0})).contract(rnvR.conjugate(), tenx::idx({1}, {1})).shuffle(std::array{0, 3, 1, 2});
//     RealScalar qpo2_max = get_max(qpo2);
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         auto mps_max = get_max(mps);
//         {
//             Eigen::Tensor<Scalar, 4> mpsqnvR(mps.dimension(0), mps.dimension(1), qnvR.dimension(1), qnvR.dimension(2));
//             Eigen::Tensor<Scalar, 5> mpsqnvRqnvL(mps.dimension(0), qnvR.dimension(1), qnvR.dimension(2), qnvL.dimension(1), qnvL.dimension(2));
//             mpsqnvR.device(*threads->dev)     = mps.contract(qnvR, tenx::idx({2}, {0}));
//             mpsqnvRqnvL.device(*threads->dev) = mpsqnvR.contract(qnvL, tenx::idx({1}, {0}));
//             res.device(*threads->dev)         = mpsqnvRqnvL.contract(qpo2, tenx::idx({0, 2, 4}, {2, 1, 0})).shuffle(tenx::array3{2, 1, 0});
//             eiglog->info("N: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsqnvR)), fp(get_max(mpsqnvRqnvL)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionQ = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // QR + rescale mpo only
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//
//     // Split environments with Householder QR along the virtual bond index
//     auto envL_map = Eigen::Map<const MatrixType>(envL.data(), envL.dimension(0) * envL.dimension(1), envL.dimension(2));
//     auto envR_map = Eigen::Map<const MatrixType>(envR.data(), envR.dimension(0) * envR.dimension(1), envR.dimension(2));
//
//     auto hhqrL = Eigen::HouseholderQR<MatrixType>(envL_map);
//     auto hhqrR = Eigen::HouseholderQR<MatrixType>(envR_map);
//
//     MatrixType qnvL_matrix = hhqrL.householderQ().setLength(envL_map.cols()) * MatrixType::Identity(envL_map.rows(), envL_map.cols()); //
//     MatrixType qnvR_matrix = hhqrR.householderQ().setLength(envR_map.cols()) * MatrixType::Identity(envR_map.rows(), envR_map.cols()); //
//
//     MatrixType rnvL_matrix = hhqrL.matrixQR().topLeftCorner(envL_map.cols(), envL_map.cols()).template triangularView<Eigen::Upper>(); // B
//     MatrixType rnvR_matrix = hhqrR.matrixQR().topLeftCorner(envR_map.cols(), envR_map.cols()).template triangularView<Eigen::Upper>(); // B
//
//     // Multiply the mpo from both sides by the "R" matrix coming from QR.
//     auto qnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvL_matrix.data(), envL.dimension(0), envL.dimension(1), qnvL_matrix.cols());
//     auto qnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvR_matrix.data(), envR.dimension(0), envR.dimension(1), qnvR_matrix.cols());
//     auto rnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvL_matrix.data(), rnvL_matrix.rows(), rnvL_matrix.cols());
//     auto rnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvR_matrix.data(), rnvR_matrix.rows(), rnvR_matrix.cols());
//
//     Eigen::Tensor<Scalar, 4> qpo2 =
//         rnvL.contract(mpo2, tenx::idx({1}, {0})).contract(rnvR.conjugate(), tenx::idx({1}, {1})).shuffle(std::array{0, 3, 1, 2});
//     RealScalar qpo2_max = get_max(qpo2);
//     qpo2 *= qpo2.constant(RealScalar{1} / qpo2_max);
//
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         {
//             Eigen::Tensor<Scalar, 4> mpsqnvL(mps.dimension(0), mps.dimension(2), qnvL.dimension(1), qnvL.dimension(2));
//             Eigen::Tensor<Scalar, 4> mpsqnvLqpo2(mps.dimension(2), qnvL.dimension(1), qpo2.dimension(1), qpo2.dimension(3));
//             mpsqnvL.device(*threads->dev)     = mps.contract(qnvL, tenx::idx({1}, {0}));
//             mpsqnvLqpo2.device(*threads->dev) = mpsqnvL.contract(qpo2, tenx::idx({3, 0}, {0, 2}));
//             res.device(*threads->dev)         = mpsqnvLqpo2.contract(qnvR, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
//             res *= res.constant(qpo2_max);
//             eiglog->info("Q: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsqnvL)), fp(get_max(mpsqnvLqpo2)), fp(get_max(res)));
//             // } else {
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionX = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // QR + rescale mpo and mps
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//     // Split environments with Householder QR along the virtual bond index
//     auto envL_map = Eigen::Map<const MatrixType>(envL.data(), envL.dimension(0) * envL.dimension(1), envL.dimension(2));
//     auto envR_map = Eigen::Map<const MatrixType>(envR.data(), envR.dimension(0) * envR.dimension(1), envR.dimension(2));
//
//     auto hhqrL = Eigen::HouseholderQR<MatrixType>(envL_map);
//     auto hhqrR = Eigen::HouseholderQR<MatrixType>(envR_map);
//
//     MatrixType qnvL_matrix = hhqrL.householderQ().setLength(envL_map.cols()) * MatrixType::Identity(envL_map.rows(), envL_map.cols()); //
//     MatrixType qnvR_matrix = hhqrR.householderQ().setLength(envR_map.cols()) * MatrixType::Identity(envR_map.rows(), envR_map.cols()); //
//
//     MatrixType rnvL_matrix = hhqrL.matrixQR().topLeftCorner(envL_map.cols(), envL_map.cols()).template triangularView<Eigen::Upper>(); // B
//     MatrixType rnvR_matrix = hhqrR.matrixQR().topLeftCorner(envR_map.cols(), envR_map.cols()).template triangularView<Eigen::Upper>(); // B
//
//     // Multiply the mpo from both sides by the "R" matrix coming from QR.
//     auto qnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvL_matrix.data(), envL.dimension(0), envL.dimension(1), qnvL_matrix.cols());
//     auto qnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(qnvR_matrix.data(), envR.dimension(0), envR.dimension(1), qnvR_matrix.cols());
//     auto rnvL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvL_matrix.data(), rnvL_matrix.rows(), rnvL_matrix.cols());
//     auto rnvR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(rnvR_matrix.data(), rnvR_matrix.rows(), rnvR_matrix.cols());
//
//     Eigen::Tensor<Scalar, 4> qpo2 =
//         rnvL.contract(mpo2, tenx::idx({1}, {0})).contract(rnvR.conjugate(), tenx::idx({1}, {1})).shuffle(std::array{0, 3, 1, 2});
//     RealScalar qpo2_max = get_max(qpo2);
//     qpo2 *= qpo2.constant(RealScalar{1} / qpo2_max);
//
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto                     mps     = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto                     res     = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         auto                     mps_max = get_max(mps);
//         Eigen::Tensor<Scalar, 3> nps     = mps * mps.constant(RealScalar{1} / mps_max);
//         {
//             Eigen::Tensor<Scalar, 4> mpsqnvL(mps.dimension(0), mps.dimension(2), qnvL.dimension(1), qnvL.dimension(2));
//             Eigen::Tensor<Scalar, 4> mpsqnvLqpo2(mps.dimension(2), qnvL.dimension(1), qpo2.dimension(1), qpo2.dimension(3));
//             mpsqnvL.device(*threads->dev)     = nps.contract(qnvL, tenx::idx({1}, {0}));
//             mpsqnvLqpo2.device(*threads->dev) = mpsqnvL.contract(qpo2, tenx::idx({3, 0}, {0, 2}));
//             res.device(*threads->dev)         = mpsqnvLqpo2.contract(qnvR, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
//             res *= res.constant(qpo2_max * mps_max);
//             eiglog->info("X: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsqnvL)), fp(get_max(mpsqnvLqpo2)), fp(get_max(res)));
//             // } else {
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionY = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply mpo2 last
//     const auto &envL                      = H2.get_envL();
//     const auto &envR                      = H2.get_envR();
//     const auto &mpo2                      = H2.get_mpos().front(); // From double layer of mpos
//     using ScalarL                         = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
//     const Eigen::Tensor<ScalarL, 4> mpo2L = mpo2.template cast<ScalarL>(); // From double layer of mpos
//     auto                            resL  = Eigen::Tensor<ScalarL, 3>(mps_shape);
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         {
//             Eigen::Tensor<Scalar, 4> mpsenvL(mps.dimension(0), mps.dimension(2), envL.dimension(1), envL.dimension(2));
//             Eigen::Tensor<Scalar, 5> mpsenvLenvR(mps.dimension(0), envL.dimension(1), envL.dimension(2), envR.dimension(1), envR.dimension(2));
//             mpsenvL.device(*threads->dev)     = mps.contract(envL, tenx::idx({1}, {0}));
//             mpsenvLenvR.device(*threads->dev) = mpsenvL.contract(envR, tenx::idx({1}, {0}));
//             resL.device(*threads->dev)        = mpo2L.contract(mpsenvLenvR.template cast<ScalarL>(), tenx::idx({2, 0, 1}, {0, 2, 4}));
//             res                               = resL.template cast<Scalar>();
//             eiglog->info("Y: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsenvL)), fp(get_max(mpsenvLenvR)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionZ = [this, gemm_highprecision_fp80](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply envL last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//
//     Eigen::Tensor<Scalar, 4> mpo2_shf = mpo2.shuffle(std::array{0, 3, 2, 1});
//     Eigen::Tensor<Scalar, 3> envL_shf = envL.shuffle(std::array{0, 2, 1});
//
//     Eigen::Index md       = mps_shape[0];
//     Eigen::Index mL       = mps_shape[1];
//     Eigen::Index mR       = mps_shape[2];
//     Eigen::Index wL       = mpo2.dimension(0);
//     Eigen::Index wR       = mpo2.dimension(1);
//     Eigen::Index wd       = mpo2.dimension(3);
//     auto         envR_mat = Eigen::Map<const MatrixType>(envR.data(), mR, mR * wR);
//     auto         envL_mat = Eigen::Map<const MatrixType>(envL_shf.data(), mL * wR, mL);
//     auto         res_shf  = Eigen::Tensor<Scalar, 3>(wd, mR, mL);
//     auto         res_mat  = Eigen::Map<MatrixType>(res_shf.data(), wd * mR, mL);
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//
//         {
//             Eigen::Tensor<Scalar, 4> T1(md, mL, mR, wR);
//             Eigen::Tensor<Scalar, 4> T2(wL, wd, mL, mR);
//
//             auto mps_mat = Eigen::Map<const MatrixType>(mps.data(), md * mL, mR);
//
//             {
//                 auto T1_mat = Eigen::Map<MatrixType>(T1.data(), md * mL, mR * wR);
//                 T1_mat      = gemm_highprecision_fp80(mps_mat, envR_mat, 1);
//             }
//
//             {
//                 T1            = Eigen::Tensor<Scalar, 4>(T1.shuffle(std::array{0, 3, 1, 2}));
//                 auto T1_mat   = Eigen::Map<const MatrixType>(T1.data(), md * wR, mL * mR);
//                 auto T2_mat   = Eigen::Map<MatrixType>(T2.data(), wR * wd, mL * mR);
//                 auto mpo2_mat = Eigen::Map<const MatrixType>(mpo2_shf.data(), wL * wd, md * wR);
//                 T2_mat        = gemm_highprecision_fp80(mpo2_mat, T1_mat, 1);
//             }
//
//             {
//                 T2          = Eigen::Tensor<Scalar, 4>(T2.shuffle(std::array{1, 3, 2, 0}));
//                 auto T2_mat = Eigen::Map<const MatrixType>(T2.data(), wd * mR, mL * wL);
//                 res_mat     = gemm_highprecision_fp80(T2_mat, envL_mat, 1);
//                 res         = res_shf.shuffle(std::array{0, 2, 1});
//             }
//
//             eiglog->info("Z: i={:2} max mps: {:20.16e} T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mps)), fp(get_max(T1)), fp(get_max(T2)),
//                          fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
// auto applyH2_highprecisionS = [this, gemm_highprecision_fp128](const Eigen::Ref<const MatrixType> &X) -> MatrixType { // Apply envL last
//     const auto &envL = H2.get_envL();
//     const auto &envR = H2.get_envR();
//     const auto &mpo2 = H2.get_mpos().front(); // From double layer of mpos
//
//     MatrixType Y(X.rows(), X.cols()); // Result
//     auto      &threads = tenx::threads::get();
//     auto       get_max = [](const auto &obj) -> RealScalar {
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(obj.data(), obj.size()).cwiseAbs().maxCoeff();
//     };
//
//     for(Eigen::Index i = 0; i < X.cols(); ++i) {
//         auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(X.col(i).data(), mps_shape); // data in
//         auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(Y.col(i).data(), mps_shape);       // data out
//         {
//             Eigen::Tensor<Scalar, 4> mpsenvR(mps.dimension(0), mps.dimension(1), envR.dimension(1), envR.dimension(2));
//             Eigen::Tensor<Scalar, 4> mpsenvRmpo(mps.dimension(1), envR.dimension(1), mpo2.dimension(0), mpo2.dimension(3));
//
//             auto mps_mat     = Eigen::Map<const MatrixType>(mps.data(), mps.dimension(0) * mps.dimension(1), mps.dimension(2));
//             auto envR_mat    = Eigen::Map<const MatrixType>(envR.data(), envR.dimension(0), envR.dimension(1) * mps.dimension(2));
//             auto mpsenvR_mat = Eigen::Map<MatrixType>(mpsenvR.data(), mps.dimension(0) * mps.dimension(1), envR.dimension(1) * envR.dimension(2));
//             mpsenvR_mat      = gemm_highprecision_fp128(mps_mat, envR_mat, 1);
//             // mpsenvR.device(*threads->dev)    = mps.contract(envR, tenx::idx({2}, {0}));
//             mpsenvRmpo.device(*threads->dev) = mpsenvR.contract(mpo2, tenx::idx({3, 0}, {1, 2}));
//             res.device(*threads->dev)        = mpsenvRmpo.contract(envL, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 2, 0});
//             eiglog->info("Z: i={:2} max T1: {:20.16e} T2: {:20.16e} res: {:20.16e}", i, fp(get_max(mpsenvR)), fp(get_max(mpsenvRmpo)), fp(get_max(res)));
//         }
//     }
//
//     return Y; // Downcast
// };
