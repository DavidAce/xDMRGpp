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
#include "tools/finite/opt_mps.h"
#include <Eigen/Eigenvalues>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace settings {
#if defined(NDEBUG)
    constexpr bool debug_solver = false;
#else
    constexpr bool debug_solver = true;
#endif
}

template<typename Scalar>
void solver_base<Scalar>::setLogger(spdlog::level::level_enum level, const std::string &name) {
    if(!eiglog) {
        eiglog = spdlog::get(name);
        if(!eiglog) {
            eiglog = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
            eiglog->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%n]%^[%=8l]%$ %v");
        }
    } else {
        if(level != eiglog->level()) { eiglog->set_level(level); }
    }
}

template<typename Scalar>
solver_base<Scalar>::solver_base(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1,
                                 MatVecMPOS<Scalar> &H2, spdlog::level::level_enum logLevel_)
    : logLevel(logLevel_), //
      nev(nev),            //
      ncv(ncv),            //
      algo(algo),          //
      ritz(ritz),          //
      H1(H1),              //
      H2(H2),              //
      V(V) {
    setLogger(logLevel, fmt::format("eigs|{}", enum2sv(algo)));
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
    set_preconditioner_params();
}

template<typename Scalar>
void solver_base<Scalar>::set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize) {
    if(jcbMaxBlockSize >= 0) {
        H1.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
        H1.factorization = eig::Factorization::LU;
        H2.factorization = eig::Factorization::LLT;
    }
}

template<typename Scalar>
Eigen::Index solver_base<Scalar>::get_jcbMaxBlockSize() const {
    assert(H1.get_jcbMaxBlockSize() == H2.get_jcbMaxBlockSize());
    return H1.get_jcbMaxBlockSize();
}
template<typename Scalar>
void solver_base<Scalar>::set_preconditioner_type(eig::Preconditioner preconditioner_type_) {
    preconditioner_type = preconditioner_type_;
    H1.preconditioner   = preconditioner_type;
    H2.preconditioner   = preconditioner_type;
    use_preconditioner  = preconditioner_type != eig::Preconditioner::NONE;
}
template<typename Scalar>
void solver_base<Scalar>::set_preconditioner_params(Eigen::Index maxiters, RealScalar initialTol, Eigen::Index jcbMaxBlockSize) {
    assert(initialTol > 0);
    use_preconditioner = preconditioner_type != eig::Preconditioner::NONE;
    H1.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::IND);
    H2.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::DEF);
    H1.set_jcbMaxBlockSize(jcbMaxBlockSize);
    H2.set_jcbMaxBlockSize(jcbMaxBlockSize);
    H1.factorization = eig::Factorization::LU;
    H2.factorization = eig::Factorization::LLT;
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::Status::op_norm_estimate(OptAlgo algo) const {
    switch(algo) {
        case OptAlgo::DMRG: return H1_max_eval;
        case OptAlgo::DMRGX: return H1_max_eval;
        case OptAlgo::HYBRID_DMRGX: return H1_max_eval;
        case OptAlgo::XDMRG: return H2_max_eval;
        case OptAlgo::GDMRG: return std::max(std::abs(H1_max_eval), std::abs(H1_min_eval)) / std::abs(H2_min_eval);
        default: throw except::runtime_error("unrecognized algo");
    }
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::Status::max_eval_estimate() const {
    auto it = std::max_element(max_eval_history.begin(), max_eval_history.end());
    if(it != max_eval_history.end()) { return std::max(RealScalar{1}, *it); }
    throw except::runtime_error("max_eval_history is empty");
}

template<typename Scalar>
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::Status::min_eval_estimate() const {
    auto it = std::max_element(min_eval_history.begin(), min_eval_history.end());
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

    if(degree == 1) { return (MultHX(Qref) - av * Qref) * (RealScalar{1} / bv / norm); }

    // eiglog->info("Chebyshev filter: x0={:.5e} norm={:.5e} lambda_min={:.5e} lambda_cut={:.5e} lambda_max={:.5e}", x0, norm, lambda_min, lambda_cut,
    // lambda_max);
    if(std::abs(norm) < eps or !std::isfinite(norm)) {
        // normalization too small; skip filtering
        eiglog->warn("norm invalid {:.5e}", fp(norm));
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

    RealScalar lambda_min = status.min_eval_estimate() * RealScalar{1.01f};
    RealScalar lambda_max = status.min_eval_estimate() * RealScalar{0.99f};
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
typename solver_base<Scalar>::RealScalar solver_base<Scalar>::get_max_standard_deviation(const std::deque<VectorReal> &v, bool apply_log10) {
    if(v.empty()) return std::numeric_limits<RealScalar>::quiet_NaN();
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
    return stddev.maxCoeff();
}

template<typename Scalar>
bool solver_base<Scalar>::rNorm_has_saturated() {
    // Check if there is less than 1% fluctuation in the (order of magnitude of) latest residual norms.
    Eigen::Index min_history_size = std::min<Eigen::Index>(status.max_history_size, 2);
    return status.iter >= min_history_size and status.rNorms_history.size() >= static_cast<size_t>(min_history_size) and
           get_max_standard_deviation(status.rNorms_history, true) < RealScalar{0.01f};
}

template<typename Scalar>
bool solver_base<Scalar>::optVal_has_saturated(RealScalar threshold) {
    // Check if there is less than 1% fluctuation in the latest optVals.
    Eigen::Index min_history_size = std::min<Eigen::Index>(status.max_history_size, 2);
    if(status.optVals_history.size() < static_cast<size_t>(min_history_size)) return false;
    auto optVal_avg     = status.optVal.cwiseAbs().mean();
    auto optVal_std     = get_max_standard_deviation(status.optVals_history, false);
    auto optVal_std_rel = optVal_std / optVal_avg;
    threshold           = std::max(threshold, rnormTol() * 10 / optVal_avg);
    return optVal_std_rel < threshold;
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_tolerance(const Eigen::Ref<const MatrixType> &S) {
    // if(status.iter_last_preconditioner_tolerance_adjustment == status.iter) return;
    H1.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.H1_max_eval), std::abs(status.H1_min_eval)) / std::min(std::abs(status.H1_max_eval), std::abs(status.H1_min_eval));
    H2.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.H2_max_eval), std::abs(status.H2_min_eval)) / std::min(std::abs(status.H2_max_eval), std::abs(status.H2_min_eval));

    if(!use_adaptive_inner_tolerance) return;
    auto Snorm                                     = S.colwise().norm().minCoeff();
    H1.get_iterativeLinearSolverConfig().tolerance = std::min<RealScalar>(RealScalar{0.1f}, std::sqrt(Snorm));
    H2.get_iterativeLinearSolverConfig().tolerance = std::min<RealScalar>(RealScalar{0.1f}, std::sqrt(Snorm));

    auto maxiters1 = std::max(100l, safe_cast<long>(100 / H1.get_iterativeLinearSolverConfig().tolerance));
    auto maxiters2 = std::max(100l, safe_cast<long>(100 / H2.get_iterativeLinearSolverConfig().tolerance));
    maxiters1      = std::min(maxiters1, 40000l);
    maxiters2      = std::min(maxiters2, 40000l);

    H1.get_iterativeLinearSolverConfig().maxiters = maxiters1;
    H2.get_iterativeLinearSolverConfig().maxiters = maxiters2;

    status.iter_last_preconditioner_tolerance_adjustment = status.iter;

    return;
    auto rNorm_log10_decrease = get_rNorms_log10_change_per_iteration();
    if(rNorm_log10_decrease == RealScalar{0}) return;
    if(rNorm_log10_decrease > RealScalar{-0.9f}) {
        // Decreasing less than a quarter of an order of magnitude per iteration,
        // We could spend more time in the inner solver, so we tighten the tolerance
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
    }

    if(rNorm_log10_decrease < RealScalar{-3.0f}) {
        // Decreasing more than two orders of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
    } else if(rNorm_log10_decrease < RealScalar{-2.1f}) {
        // Decreasing more than one order of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{2};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{2};
    }
    /* clang-format off */
    H1.get_iterativeLinearSolverConfig().tolerance = std::clamp<RealScalar>(H1.get_iterativeLinearSolverConfig().tolerance, RealScalar{5e-12f}, RealScalar{0.25f});
    H2.get_iterativeLinearSolverConfig().tolerance = std::clamp<RealScalar>(H2.get_iterativeLinearSolverConfig().tolerance, RealScalar{5e-12f}, RealScalar{0.25f});
    /* clang-format on */
    status.iter_last_preconditioner_tolerance_adjustment = status.iter;
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_H1_limits() {
    if(status.iter_last_preconditioner_H1_limit_adjustment == status.iter) return;
    H1.get_iterativeLinearSolverConfig().precondType = PreconditionerType::JACOBI;
    if(H1.get_iterativeLinearSolverConfig().precondType == PreconditionerType::CHEBYSHEV) {
        RealScalar lambda_min                                     = status.H1_min_eval * RealScalar{0.9f};
        RealScalar lambda_max                                     = status.H1_max_eval * RealScalar{1.1f};
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
    H2.get_iterativeLinearSolverConfig().precondType = PreconditionerType::JACOBI;

    if(H2.get_iterativeLinearSolverConfig().precondType == PreconditionerType::CHEBYSHEV) {
        RealScalar lambda_min                                     = RealScalar{0}; // status.H2_min_eval * RealScalar{0.9f};
        RealScalar lambda_max                                     = status.H2_max_eval * RealScalar{1.01f};
        H2.get_iterativeLinearSolverConfig().chebyshev.lambda_min = lambda_min;
        H2.get_iterativeLinearSolverConfig().chebyshev.lambda_max = lambda_max;
        H2.get_iterativeLinearSolverConfig().chebyshev.degree     = 2;
    }
    status.iter_last_preconditioner_H2_limit_adjustment = status.iter;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultHX(const Eigen::Ref<const MatrixType> &X) {
    auto       token_matvecs = status.time_matvecs.tic_token();
    MatrixType HX;
    switch(algo) {
        case OptAlgo::DMRG:
            HX = H1.MultAX(X);
            status.num_matvecs += X.cols();
            break;
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: {
            MatrixType H2X = H2.MultAX(X);
            MatrixType H1X = H1.MultAX(X);
            HX             = H2X - H1.MultAX(H1X);
            status.num_matvecs += 3 * X.cols(); // two more matvecs
            break;
        }
        case OptAlgo::XDMRG:
            HX = H2.MultAX(X);
            status.num_matvecs += X.cols();
            break;
        case OptAlgo::GDMRG: throw except::runtime_error("MultHX: GDMRG is not suitable, use MultH1X or MultH2X instead");
        default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
    }
    return HX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH1X(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH1X: should only be called by GDMRG");
    auto token_matvecs = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H1.MultAX(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH2X(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH2X: should only be called by GDMRG");
    auto token_matvecs = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H2.MultAX(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultPX(const Eigen::Ref<const MatrixType>               &X,
                                                                     std::optional<const Eigen::Ref<const MatrixType>> initialGuess) {
    // Preconditioning
    auto       token_precond = status.time_precond.tic_token();
    MatrixType HPX;
    switch(algo) {
        case OptAlgo::DMRG: {
            if(use_initial_guess) H1.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX = H1.MultPX(X);
            break;
        }
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: {
            if(use_initial_guess) H2.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX = H2.MultPX(X);
            break;
        }
        case OptAlgo::XDMRG: {
            if(use_initial_guess) H2.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
            HPX = H2.MultPX(X);
            break;
        }
        case OptAlgo::GDMRG: throw except::runtime_error("MultPX: GDMRG is not suitable, use MultP1X or MultP2X instead");
        default: throw except::runtime_error("MultPX: unknown algorithm {}", enum2sv(algo));
    }

    auto &H1ir = H1.get_iterativeLinearSolverConfig().result;
    auto &H2ir = H2.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_matvecs_inner += H1ir.matvecs + H2ir.matvecs;
    status.num_precond_inner += H1ir.precond + H2ir.precond;
    status.time_matvecs_inner += H1ir.time_matvecs + H2ir.time_matvecs;
    status.time_precond_inner += H1ir.time_precond + H2ir.time_precond;
    H1ir.reset();
    H2ir.reset();
    return HPX;
}
template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP1X(const Eigen::Ref<const MatrixType>               &X,
                                                                      std::optional<const Eigen::Ref<const MatrixType>> initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP1X: should only be called by GDMRG");
    auto token_precond = status.time_precond.tic_token();
    if(use_initial_guess) H1.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
    MatrixType HPX  = H1.MultPX(X);
    auto      &H1ir = H1.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_matvecs_inner += H1ir.matvecs;
    status.num_precond_inner += H1ir.precond;
    status.time_matvecs_inner += H1ir.time_matvecs;
    status.time_precond_inner += H1ir.time_precond;
    H1ir.reset();
    return HPX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP2X(const Eigen::Ref<const MatrixType>               &X,
                                                                      std::optional<const Eigen::Ref<const MatrixType>> initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP2X: should only be called by GDMRG");
    auto token_precond = status.time_precond.tic_token();
    if(use_initial_guess) H2.get_iterativeLinearSolverConfig().initialGuess = initialGuess.value_or(MatrixType{});
    MatrixType HPX  = H2.MultPX(X);
    auto      &H2ir = H2.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_matvecs_inner += H2ir.matvecs;
    status.num_precond_inner += H2ir.precond;
    status.time_matvecs_inner += H2ir.time_matvecs;
    status.time_precond_inner += H2ir.time_precond;
    H2ir.reset();
    return HPX;
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_mBlock() {
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

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::cheap_Olsen_correction(const MatrixType &V, const MatrixType &S) {
    MatrixType D(S.rows(), S.cols());

    // Generate the cheap olsen correction (S - Δ*V),
    // Where Δ is a diagonal matrix that Δ𝑖,𝑖 holds an estimation of the error of
    // the approximate eigenvalue Λ𝑖,𝑖.
    for(long i = 0; i < S.cols(); ++i) {
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

        auto delta         = std::abs(denominator) > eps * 100 ? numerator / denominator : RealScalar{0};
        D.col(i).noalias() = s - delta * v; // Gets preconditioned later
    }
    return D;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::full_Olsen_correction(const MatrixType &V, const MatrixType &S) {
    // Precondition V and S blockwise
    MatrixType MV;
    MatrixType MS;
    MatrixType coeffs;
    if(algo == OptAlgo::GDMRG) {
        MV.noalias() = use_preconditioner ? MultP2X(V, V_prec) : V;
        MS.noalias() = use_preconditioner ? MultP2X(S, S_prec) : S;
        if(use_preconditioner and use_initial_guess) V_prec = MV;
        if(use_preconditioner and use_initial_guess) S_prec = MS;

        // Gram matrix in H2-inner product: G = V^H * B * MV (b x b)
        MatrixType B_MV = MultH2X(MV);
        MatrixType G    = V.adjoint() * B_MV;

        // Coefficients: G^{-1} * (V^H * B * MS)
        MatrixType H2_MS    = MultH2X(MS);
        MatrixType VT_H2_MS = V.adjoint() * H2_MS;
        coeffs              = G.ldlt().solve(VT_H2_MS);
    } else {
        MV.noalias() = use_preconditioner ? MultPX(V, V_prec) : V;
        MS.noalias() = use_preconditioner ? MultPX(S, S_prec) : S;

        if(use_preconditioner and use_initial_guess) V_prec = MV;
        if(use_preconditioner and use_initial_guess) S_prec = MS;
        // Gram matrix in preconditioned metric: G = V^T * MV (b x b)
        MatrixType G = V.adjoint() * MV; // symmetric if M is HPD

        // Projection coefficients: C = G^{-1} * (V^T * MS)
        MatrixType VT_MS = V.adjoint() * MS;
        coeffs           = G.ldlt().solve(VT_MS); // robust inversion
    }
    // Olsen correction
    return MS - MV * coeffs; // N x b
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::jacobi_davidson_correction(const MatrixType &V, const MatrixType &S) {
    assert(V.rows() == S.rows());
    assert(V.cols() == S.cols());

    auto &H = algo == OptAlgo::DMRG ? H1 : H2;
    H.CalcPc(); // Compute the block-jacobi preconditioner (do llt on all blocks)
    bool use_b = use_b_orthonormal_jd_projection and algo == OptAlgo::GDMRG and H2V.cols() == V.cols();

    VectorReal        Y   = T_evals(status.optIdx);
    MatrixType        RHS = -S;
    const MatrixType &BV  = use_b ? std::as_const(H2V) : V;
    // Right-hand side (projected)
    if(use_b) {
        auto       token_precond = status.time_precond.tic_token();
        MatrixType Gram          = V.adjoint() * BV;                     // k × k  Gram matrix  G = Vᴴ H2 V      (Hermitian, pos.def.)
        MatrixType projCoeff     = Gram.ldlt().solve(V.adjoint() * RHS); // solve   G ⋅ coeff = Vᴴ RHS     (coeff is k × b)
        RHS.noalias() -= BV * projCoeff;                                 // RHS ← RHS − H2 V ⋅ coeff      (H₂-orthogonal projector)
        if(b > 1) {                                                      // (optional second DGKS pass for extra safety)
            projCoeff = Gram.ldlt().solve(V.adjoint() * RHS);
            RHS.noalias() -= BV * projCoeff;
        }
    } else {
        auto token_precond = status.time_precond.tic_token();

        RHS -= V * (V.adjoint() * RHS).eval();           // Use euclidean inner product
        if(b > 1) RHS -= V * (V.adjoint() * RHS).eval(); // Second pass for better precision
    }

    auto MatrixOp = [this, &H](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        status.num_matvecs_inner += X.cols();
        return H.MultAX(X);
    };
    MatrixType D(S.rows(), S.cols()); // accumulate the result from the JD correction equation

    for(Eigen::Index i = 0; i < b; ++i) { // We use block size b
        // v: current Ritz vector, s: current residual, Bv: either I*v (Euclidean) or H2*v (H2-orthonormal)
        const VectorType &s   = S.col(i);   // The residual vector i
        const VectorType &v   = V.col(i);   // The ritz vector i
        const VectorType &Bv  = BV.col(i);  // H.MultAX(v);   // we just take the euclidean basis here
        const RealScalar &th  = Y(i);       // The ritz value for this ritz vector
        const VectorType &rhs = RHS.col(i); // Right-hand side (projected)

        if(i > 0) {
            // Do Cheap Olsen + CG instead
            D.col(i).noalias() = algo == OptAlgo::DMRG ? MultP1X(s) : MultP2X(s);
            D.col(i).noalias() = cheap_Olsen_correction(v, D.col(i));
        } else {
            auto token_precond = status.time_precond.tic_token();

            IterativeLinearSolverConfig<Scalar> cfg = H.get_iterativeLinearSolverConfig(); // Get the jacobi blocks
            cfg.result                              = {};
            cfg.matdef                              = MatDef ::IND;
            cfg.precondType                         = PreconditionerType::JACOBI;

            // Define the residual matrix-vector operator depending on the different DMRG algorithms
            auto ResidualOp = [this, i, th, &H](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                MatrixType HX(X.rows(), X.cols());
                switch(algo) {
                    case OptAlgo::DMRG: [[fallthrough]];
                    case OptAlgo::DMRGX: [[fallthrough]];
                    case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
                    case OptAlgo::XDMRG:
                        HX.noalias() = H.MultAX(X) - th * X;
                        status.num_matvecs_inner += X.cols();
                        break;
                    case OptAlgo::GDMRG: // Generalized problem
                        HX.noalias() = H1.MultAX(X) - th * H2.MultAX(X);
                        status.num_matvecs_inner += 2 * X.cols();
                        break;
                    default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
                }
                return HX;
            };

            auto JDop = JacobiDavidsonOperator<Scalar>(v, Bv, s, ResidualOp, MatrixOp);
            if(D_prec.cols() == b and use_initial_guess) {
                D_prec.col(i) -= Bv * v.dot(D_prec.col(i)); // Project out the current direction
                // Check that the initial guess is good
                RealScalar ratio = (JDop * D_prec.col(i) - rhs).norm() / rhs.norm();
                if(ratio < 1) {
                    eiglog->info("ratio: {:.3e}", fp(ratio));
                    cfg.initialGuess = D_prec.col(i);
                } else {
                    cfg.initialGuess = MatrixType{};
                }
            } else {
                cfg.initialGuess = MatrixType{};
            }
            if(dev_cheap_olsen_as_jd_initial_guess) cfg.initialGuess = cheap_Olsen_correction(V, RHS);

            D.col(i).noalias() = JacobiDavidsonSolver(JDop, rhs, cfg);

            // status.num_matvecs_inner += cfg.result.matvecs;
            status.num_precond_inner += cfg.result.precond;
            status.time_matvecs_inner += cfg.result.time_matvecs;
            status.time_precond_inner += cfg.result.time_precond;

            H.get_iterativeLinearSolverConfig().result += cfg.result;
        }
    }
    status.num_precond += b; // This routine is a preconditioner
    // D_prec = D;              // Store the result so we can use it in the next iteration.
    return D; // N x b, enrichment directions
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_sBlock(const MatrixType &S_in, fMultPX_t MultPX) {
    // Make a residual block "S = (HQ-λQ)"
    MatrixType S = S_in;
    // if(S.cols() != b) {
    //     auto Y = T_evals(status.optIdx);
    //     if(algo == OptAlgo::GDMRG) {
    //         S.noalias() = H1V - H2V * Y.asDiagonal();
    //     } else {
    //         S.noalias() = HV - V * Y.asDiagonal();
    //     }
    // }
    assert(S.cols() > 0);
    adjust_preconditioner_tolerance(S);

    if(chebyshev_filter_degree >= 1) S = qr_and_chebyshevFilter(S);
    switch(residual_correction_type_internal) {
        case ResidualCorrectionType::NONE:
            if(use_preconditioner) {
                S = MultPX(S, S_prec);
                if(use_initial_guess) S_prec = S;
            }
            break;
        case ResidualCorrectionType::AUTO: [[fallthrough]];
        case ResidualCorrectionType::CHEAP_OLSEN:
            if(use_preconditioner) {
                S = MultPX(S, S_prec);
                if(use_initial_guess) S_prec = S;
            }
            S.noalias() = cheap_Olsen_correction(V, S);
            break;
        case ResidualCorrectionType::FULL_OLSEN:
            // This has an internal preconditioner
            S.noalias() = full_Olsen_correction(V, S);
            break;
        case ResidualCorrectionType::JACOBI_DAVIDSON:
            // This is an internal preconditioner
            assert(use_preconditioner && " Jacobi Davidson correction needs use_preconditioner == true");
            S.noalias() = jacobi_davidson_correction(V, S);
            break;
    }
    assert_allFinite(S);
    return S;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_wBlock(fMultPX_t MultPX) {
    // We add Lanczos-style residual blocks
    W = (algo == OptAlgo::GDMRG) ? H2V : HV;
    A = V.adjoint() * W;

    // 3) Subtract projections to A and B once
    W.noalias() -= V * A; // Qi * Qi.adjoint()*H*Qi
    if(V_prev.rows() == N and V_prev.cols() == b) {
        B = V_prev.adjoint() * W;
        W.noalias() -= V_prev * B.adjoint();
    }
    assert_allFinite(W);
    if(use_preconditioner) {
        W = MultPX(W, W_prec);
        if(use_initial_guess) W_prec = W;
    }
    return W;
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_rBlock() {
    // Get a random block
    return MatrixType::Random(N, b);
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
    HQ   = MultHX(Q);
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
    HQ_cur   = MultHX(Q.middleCols((qBlocks - 1) * b, b));
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
void solver_base<Scalar>::orthogonalize(const Eigen::Ref<const MatrixType> &X,       // (N, xcols)
                                        Eigen::Ref<MatrixType>              Y,       // (N, ycols)
                                        RealScalar                          normTol, // The largest allowed norm error
                                        Eigen::Ref<VectorIdxT>              mask     // block norm mask, size = n_blocks = ycols / blockWidth
) {
    if(X.cols() == 0 || Y.cols() == 0) return;

    assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == Y.cols() / b && "Mask size must match number of blocks in Y.");

    assert_allFinite(X);
    assert_allFinite(Y);
    assert_orthonormal(X, normTol * 10000);

    const Eigen::Index n_blocks_y = Y.cols() / b;
    const Eigen::Index n_blocks_x = X.cols() / b;

    auto set_mask = [&](Eigen::Ref<MatrixType> matrix, Eigen::Index &maskItem) -> bool {
        if(maskItem == 0) return true;
        VectorReal norms = matrix.colwise().norm();
        if(norms.minCoeff() < normTol) {
            eiglog->debug("masked {::.3e} < {:.3e}", fv(norms), fp(normTol));
            maskItem = 0;
            matrix.setZero();
            return true;
        }
        return false;
    };

    // DGKS clean Y against X and orthogonalize within Y
    for(int rep = 0; rep < 2; ++rep) {
        for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
            if(mask(blk_y) == 0) continue;
            auto Yblock = Y.middleCols(blk_y * b, b);

            if(set_mask(Yblock, mask(blk_y))) continue; // Mask if numerically zero

            for(Eigen::Index blk_x = 0; blk_x < n_blocks_x; ++blk_x) {
                auto Xblock = X.middleCols(blk_x * b, b);
                Yblock.noalias() -= Xblock * (Xblock.adjoint() * Yblock).eval(); // Remove projection
            }

            if(set_mask(Yblock, mask(blk_y))) continue; // Mask if numerically zero

            // Forward MGS to orthogonalize Y internally
            for(Eigen::Index blk2_y = blk_y + 1; blk2_y < n_blocks_y; ++blk2_y) {
                if(mask(blk2_y) == 0) continue;
                auto Yblock2 = Y.middleCols(blk2_y * b, b);
                auto G       = Yblock.adjoint() * Yblock;  // Gram matrix b x b
                auto C       = Yblock.adjoint() * Yblock2; // Gram matrix b x b
                Yblock2.noalias() -= Yblock * G.ldlt().solve(C);
                if(set_mask(Yblock2, mask(blk2_y))) continue; // Mask if numerically zero
            }
        }
    }

    if constexpr(settings::debug_solver) {
        // Sanity checks on the remaining columns of Y that have not been masked out (i.e. that are still active)
        std::vector<Eigen::Index> active_ycols;
        for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
            if(mask(j) == 1) {
                for(Eigen::Index col = 0; col < b; ++col) active_ycols.push_back(j * b + col);
            }
        }
        if(active_ycols.empty()) { return; }
        auto Ymask = Y(Eigen::all, active_ycols);

        // Check that Y is orthonormal
        assert_orthogonal(X, Ymask, normTol * 10000);
        assert_allFinite(Ymask);
    }
}

template<typename Scalar>
void solver_base<Scalar>::orthonormalize(const Eigen::Ref<const MatrixType> &X,       // (N, xcols)
                                         Eigen::Ref<MatrixType>              Y,       // (N, ycols)
                                         RealScalar                          normTol, // The largest allowed norm error
                                         Eigen::Ref<VectorIdxT>              mask     // block norm mask, size = n_blocks = ycols / blockWidth
) {
    if(X.cols() == 0 || Y.cols() == 0) return;
    assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
    assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
    assert(mask.size() == Y.cols() / b && "Mask size must match the number of blocks in Y.");

    const Eigen::Index n_blocks_y = Y.cols() / b;
    const Eigen::Index n_blocks_x = X.cols() / b;

    if(mask.sum() == 0) return;
    auto orthTol = 10000 * normTol;

    assert_allFinite(X);
    assert_allFinite(Y);
    assert_orthonormal(X, orthTol);

    auto set_mask = [&](Eigen::Ref<MatrixType> matrix, Eigen::Index &maskItem, VectorReal norms = {}) -> bool {
        if(maskItem == 0) return true;
        if(norms.size() == 0) norms = matrix.colwise().norm();
        if(norms.minCoeff() < normTol) {
            // eiglog->debug("masked {::.3e} < {:.3e}", fv(norms), fp(normTol));
            maskItem = 0;
            matrix.setZero();
            return true;
        }
        return false;
    };

    auto get_Ymasked = [&]() {
        std::vector<Eigen::Index> active_ycols;
        for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
            if(mask(j) == 1) {
                for(Eigen::Index col = 0; col < b; ++col) active_ycols.push_back(j * b + col);
            }
        }
        // if(active_ycols.empty()) { return; }
        return Y(Eigen::all, active_ycols);
    };

    // DGKS clean Y against X and orthonormalize Y
    for(int rep = 0; rep < 2; ++rep) {
        for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
            if(mask(blk_y) == 0) continue;
            auto Yblock = Y.middleCols(blk_y * b, b);
            for(Eigen::Index blk_x = 0; blk_x < n_blocks_x; ++blk_x) {
                auto Xblock = X.middleCols(blk_x * b, b);
                Yblock.noalias() -= Xblock * (Xblock.adjoint() * Yblock).eval(); // Remove projection
            }
        }
    }

    VectorReal norms = Y.colwise().norm();
    // eiglog->info("Y norms after DGKS: {::.3e} | current mask: {}", fv(norms), mask);
    for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
        if(mask(blk_y) == 0) continue;
        auto Yblock = Y.middleCols(blk_y * b, b);
        set_mask(Yblock, mask(blk_y));
    }
    if(mask.sum() == 0) return;

    norms = Y.colwise().norm();
    // eiglog->info("Y norms after mask 1: {::.3e} | current mask: {}", fv(norms), mask);

    // Orthonormalize the active columns of Y
    {
        MatrixType Ymask = get_Ymasked();
        hhqr.compute(Ymask);
        VectorReal Rdiag = hhqr.matrixQR().diagonal().cwiseAbs().topRows(Ymask.cols());
        // eiglog->info("R diag  after QR    : {::.3e} | current mask: {}", fv(Rdiag), mask);
        Ymask                       = hhqr.householderQ().setLength(Ymask.cols()) * MatrixType::Identity(Ymask.rows(), Ymask.cols());
        Eigen::Index n_blocks_ymask = Ymask.cols() / b;
        for(Eigen::Index blk_y = 0; blk_y < n_blocks_ymask; ++blk_y) {
            auto Yblock = Ymask.middleCols(blk_y * b, b);
            auto Rblock = Rdiag.middleRows(blk_y * b, b);
            if(Rblock.minCoeff() < normTol) { Yblock.setZero(); }
        }
        norms = Ymask.colwise().norm();
        // eiglog->info("Ymask norms after QR    : {::.3e} | current mask: {}", fv(norms), mask);
        get_Ymasked() = Ymask;
    }

    // Mask blocks that contain columns with near-zero norm
    norms = Y.colwise().norm();
    // eiglog->info("Y norms after QR    : {::.3e} | current mask: {}", fv(norms), mask);

    for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
        if(mask(blk_y) == 0) continue;
        auto Yblock = Y.middleCols(blk_y * b, b);
        set_mask(Yblock, mask(blk_y));
    }
    if(mask.sum() == 0) return;
    norms = Y.colwise().norm();
    // eiglog->info("Y norms after mask 2: {::.3e} | current mask: {}", fv(norms), mask);

    // Finally, compute the gram matrix X.adjoint()*Yblock to make sure that normalization didn't blow up the error too much.
    for(Eigen::Index blk_y = 0; blk_y < n_blocks_y; ++blk_y) {
        if(mask(blk_y) == 0) continue;
        auto       Yblock       = Y.middleCols(blk_y * b, b);
        MatrixType XYblock      = X.adjoint() * Yblock;
        RealScalar XYblockError = XYblock.cwiseAbs().maxCoeff();
        if(XYblockError >= orthTol) {
            eiglog->debug("blk {}: masked {:.3e} < {:.3e} (orthonormality test)", blk_y, fp(XYblockError), fp(orthTol));
            Yblock.setZero();
            mask(blk_y) = 0;
        }
    }

    if constexpr(settings::debug_solver) {
        if(mask.sum() == 0) return;
        // Sanity checks on the remaining columns of Y that have not been masked out (i.e. that are still active)
        auto Ymask = get_Ymasked();

        assert_orthonormal(Ymask, normTol);
        assert_orthogonal(X, Ymask, orthTol);
        assert_allFinite(Ymask);
    }
}

template<typename Scalar>
void solver_base<Scalar>::compress_cols(MatrixType       &X,   // (N, ycols)
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
void solver_base<Scalar>::compress_rows_and_cols(MatrixType       &X,   // (N, ycols)
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
    assert_allFinite(X);
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

template<typename Scalar> void solver_base<Scalar>::assert_orthonormal(const Eigen::Ref<const MatrixType> &X, RealScalar threshold,
                                                                       const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0) return;
        MatrixType XX        = X.adjoint() * X;
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            eiglog->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            eiglog->warn("X orthError: {:.5e}", fp(orthError));
            throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), fp(orthError), fp(threshold));
        }
    }
}
template<typename Scalar> void solver_base<Scalar>::assert_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y,
                                                                      RealScalar threshold, const std::source_location &location) {
    if constexpr(settings::debug_solver) {
        if(X.cols() == 0 || Y.cols() == 0) return;
        MatrixType XY          = X.adjoint() * Y;
        auto       XYorthError = XY.cwiseAbs().maxCoeff();
        if(XYorthError > threshold) {
            eiglog->info("XY: \n{}\n", linalg::matrix::to_string(XY, 8));
            eiglog->info("XY orthError: {:.5e}", fp(XYorthError));
            throw except::runtime_error("{}:{}: {}: matrices are not orthogonal: error = {:.5e} > threshold {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), fp(XYorthError), fp(threshold));
        }
    }
}

template<typename Scalar>
std::vector<Eigen::Index> solver_base<Scalar>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) {
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
        // cpqr.setThreshold(orthTolQ);
        auto rank = std::min(cpqr.rank(), b);
        V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank) * cpqr.colsPermutation().transpose();
        if(V.cols() == b) break;
    }

    assert(V.cols() == b);
    assert_orthonormal(V, orthTolQ);
    if(status.iter == 0) {
        // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
        if(algo == OptAlgo::GDMRG) {
            Q             = V;
            H1Q           = MultH1X(Q);
            H2Q           = MultH2X(Q);
            MatrixType T1 = Q.adjoint() * H1Q;
            MatrixType T2 = Q.adjoint() * H2Q;
            T1            = RealScalar{0.5f} * (T1.adjoint() + T1); // Symmetrize
            T2            = RealScalar{0.5f} * (T2.adjoint() + T2); // Symmetrize
            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es_seed(T1, T2, Eigen::Ax_lBx);
            T_evecs       = es_seed.eigenvectors().colwise().normalized();
            T_evals       = es_seed.eigenvalues();
            status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
            auto Z        = T_evecs(Eigen::all, status.optIdx);
            auto Y        = T_evals(status.optIdx);
            V             = Q * Z; // Now V has b columns mixed according to the selected columns in T_evecs
            if(b > 1) {
                // V is H2-orthonormal, so we orthonormalize it to get regular orthonormality
                hhqr.compute(V);
                V   = hhqr.householderQ().setLength(b) * MatrixType::Identity(N, b);
                H1V = MultH1X(V);
                H2V = MultH2X(V);
            } else {
                // Transform the basis with applied operators
                H1V = H1Q * Z;
                H2V = H2Q * Z;
            }
            S             = H1V - H2V * Y.asDiagonal();
            status.rNorms = S.colwise().norm();
            status.optVal = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            status.commit_evals(T_evals.cwiseAbs().minCoeff(), T_evals.cwiseAbs().maxCoeff());
            Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
            Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);

            status.H1_min_eval = es1.eigenvalues().minCoeff();
            status.H1_max_eval = es1.eigenvalues().maxCoeff();
            status.H2_min_eval = es2.eigenvalues().minCoeff();
            status.H2_max_eval = es2.eigenvalues().maxCoeff();
            RealScalar min_sep =
                T_evals.size() <= 1 ? RealScalar{1} : (T_evals.bottomRows(T_evals.size() - 1) - T_evals.topRows(T_evals.size() - 1)).cwiseAbs().minCoeff();
            auto select1     = get_ritz_indices(ritz, 0, 1, T_evals);
            auto H1_max_abs  = std::max(std::abs(status.H1_min_eval), std::abs(status.H1_max_eval));
            auto H2_max_abs  = std::max(std::abs(status.H2_min_eval), std::abs(status.H2_max_eval));
            status.condition = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;

        } else {
            Q  = V;
            HQ = MultHX(V);
            T  = Q.adjoint() * HQ;
            T  = RealScalar{0.5f} * (T.adjoint() + T); // Symmetrize
            Eigen::SelfAdjointEigenSolver<MatrixType> es_seed(T);
            T_evecs       = es_seed.eigenvectors();
            T_evals       = es_seed.eigenvalues();
            status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
            auto Z        = T_evecs(Eigen::all, status.optIdx);
            auto Y        = T_evals(status.optIdx);
            V             = Q * Z; // Now V has b columns mixed according to the selected columns in T_evecs
            HV            = HQ * Z;
            S             = HV - V * Y.asDiagonal();
            status.rNorms = S.colwise().norm();
            status.optVal = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b

            status.H1_min_eval = T_evals.minCoeff();
            status.H2_min_eval = T_evals.minCoeff();
            status.H1_max_eval = T_evals.maxCoeff();
            status.H2_max_eval = T_evals.maxCoeff();
            status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
            status.condition = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
        }
    }

    assert(V.cols() == b);
    assert_allFinite(V);
    assert_orthonormal(V, orthTolQ);
    // adjust_preconditioner_tolerance();
    // adjust_residual_correction_type();
    // adjust_preconditioner_H1_limits();
    // adjust_preconditioner_H2_limits();
    // eiglog->debug("iter 0| mv {:5} | optVal {::.16f} | blk {:2} | b {} | ritz {} | rNormTol {:.3e} | tol {:.2e} | rNorms = {::.8e}", status.num_matvecs,
    //              fv(status.optVal), Q.cols() / b, b, enum2sv(ritz), fp(rnormTol()), fp(tol), fv(VectorReal(status.rNorms.topRows(nev))));

    // Now V has b orthonormalized ritz vectors
}

template<typename Scalar>
void solver_base<Scalar>::diagonalizeT() {
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

    status.H1_min_eval = std::min(status.H1_min_eval, T_evals.minCoeff());
    status.H1_max_eval = std::max(status.H1_max_eval, T_evals.maxCoeff());
    status.H2_min_eval = std::min(status.H2_min_eval, T_evals.minCoeff());
    status.H2_max_eval = std::max(status.H2_max_eval, T_evals.maxCoeff());

    auto diff = [](const VectorReal &x) -> VectorReal {
        if(x.size() <= 1) return VectorReal::Ones(1);
        return x.bottomRows(x.size() - 1) - x.topRows(x.size() - 1);
    };
    auto select2 = get_ritz_indices(ritz, 0, nev + 1, T_evals);
    status.gap   = diff(T_evals(select2)).cwiseAbs().minCoeff();

    status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
    status.condition = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
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
    status.rNorms = {};

    // if(use_b_orthonormal_jd_projection) {
    //     Eigen::Index k    = Q.cols();
    //     MatrixType   G    = Q.adjoint() * H2Q;
    //     MatrixType   Linv = G.llt().solve(MatrixType::Identity(k, k));
    //     Q *= Linv; // Q is now B-orthonormal
    //     H1Q *= Linv;
    //     H2Q *= Linv;
    // }

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
        if(use_b_orthonormal_jd_projection)
            T_evecs = es.eigenvectors();
        else
            T_evecs = es.eigenvectors().colwise().normalized();
    } else {
        // Minimize variance instead, but we should invert the spectrum
        eiglog->warn("Generalized eigenvalue problem failed, using variance minimization instead. \n");
        Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2 - T1 * T1);
        T_evals = es2.eigenvalues().inverse();
        T_evecs = es2.eigenvectors();
        assert_allFinite(T_evecs);
    }
    status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());

    // Calculate the gap
    auto diff = [](const VectorReal &x) -> VectorReal {
        if(x.size() <= 1) return VectorReal::Ones(1);
        return x.bottomRows(x.size() - 1) - x.topRows(x.size() - 1);
    };
    auto select = get_ritz_indices(ritz, 0, std::max(b, nev + 1), T_evals);
    status.gap  = diff(T_evals(select)).cwiseAbs().minCoeff();

    if(dev_append_extra_blocks_to_basis) {
        // ----- Rayleigh quotients, gaps and residuals for H1 and H2 -------------------------------------
        auto       Z             = T_evecs(Eigen::all, select);
        MatrixType QZ            = Q * Z;
        MatrixType H1QZ          = H1Q * Z;
        MatrixType H2QZ          = H2Q * Z;
        VectorReal QZsquaredNorm = QZ.colwise().squaredNorm();
        VectorReal Y1            = (QZ.adjoint() * H1QZ).diagonal().real().eval().cwiseQuotient(QZsquaredNorm); // (u† H1 u)/(u†u)
        VectorReal Y2            = (QZ.adjoint() * H2QZ).diagonal().real().eval().cwiseQuotient(QZsquaredNorm); // (u† H2 u)/(u†u)

        // ----- Gaps ----------------------------------------------
        status.gap_H1 = diff(Y1).cwiseAbs().minCoeff();
        status.gap_H2 = diff(Y2).cwiseAbs().minCoeff();

        if(b != nev + 1) {
            Y1.conservativeResize(b);
            Y2.conservativeResize(b);
            QZ.conservativeResize(Eigen::NoChange, b);
            H1QZ.conservativeResize(Eigen::NoChange, b);
            H2QZ.conservativeResize(Eigen::NoChange, b);
        }

        S1 = H1QZ - QZ * Y1.asDiagonal();
        S2 = H2QZ - QZ * Y2.asDiagonal();

        // ----- Residuals ----------------------------------------------
        status.rNorms_H1 = S1.colwise().norm();
        status.rNorms_H2 = S2.colwise().norm();
    }

    // Calculate min max eigenvalues of H1 and H2 and condition number

    Eigen::SelfAdjointEigenSolver<MatrixType> es1(T1);
    Eigen::SelfAdjointEigenSolver<MatrixType> es2(T2);
    {
        status.H1_min_eval    = std::min(status.H1_min_eval, es1.eigenvalues().minCoeff());
        status.H1_max_eval    = std::max(status.H1_max_eval, es1.eigenvalues().maxCoeff());
        status.H2_min_eval    = std::min(status.H2_min_eval, es2.eigenvalues().minCoeff());
        status.H2_max_eval    = std::max(status.H2_max_eval, es2.eigenvalues().maxCoeff());
        RealScalar min_sep    = diff(T_evals).cwiseAbs().minCoeff();
        auto       select1    = get_ritz_indices(ritz, 0, 1, T_evals);
        auto       H1_max_abs = std::max(std::abs(status.H1_min_eval), std::abs(status.H1_max_eval));
        auto       H2_max_abs = std::max(std::abs(status.H2_min_eval), std::abs(status.H2_max_eval));
        status.condition      = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;
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
            Z                             = Z(Eigen::all, deflIdx).eval();
            Vdefl                         = Vdefl(Eigen::all, deflIdx).eval();
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
    if(use_coarse_inner_preconditioner and status.iter >= 1) {
        Eigen::Index              nCoarse    = std::min(5l, es2.eigenvalues().size());
        MatrixType                Z          = es2.eigenvectors().leftCols(nCoarse);
        VectorReal                Y          = es2.eigenvalues().topRows(nCoarse);
        MatrixType                coarseZ    = Q * Z;
        MatrixType                coarseHZ   = algo == OptAlgo::GDMRG ? H2Q * Z : HQ * Z;
        VectorReal                rnorms     = (coarseHZ - coarseZ * Y.asDiagonal()).colwise().norm();
        std::vector<Eigen::Index> nCoarseIdx = {};
        for(Eigen::Index idx = 0; idx < nCoarse; ++idx) {
            if(rnorms(idx) < RealScalar{1e-5f}) nCoarseIdx.emplace_back(idx);
        }
        auto &jcbCfg = algo == OptAlgo::DMRG ? H1.get_iterativeLinearSolverConfig().jacobi : H2.get_iterativeLinearSolverConfig().jacobi;

        if(nCoarseIdx.size() > 0) {
            eiglog->trace("coarsening idx {} | eigv {} | rnorms {}", nCoarseIdx, fv(Y), fv(rnorms));
            // one-time B-orthonormalisation of Z
            Z                             = Z(Eigen::all, nCoarseIdx).eval();
            coarseZ                       = coarseZ(Eigen::all, nCoarseIdx).eval();
            coarseHZ                      = coarseHZ(Eigen::all, nCoarseIdx).eval();
            rnorms                        = rnorms(nCoarseIdx).eval();
            MatrixType             GramH2 = coarseZ.adjoint() * coarseHZ; // small p×p matrix
            Eigen::LLT<MatrixType> llt(GramH2);
            coarseZ         = (coarseZ * llt.matrixL().solve(MatrixType::Identity(GramH2.rows(), GramH2.cols()))).eval(); // now Zᵀ B Z = I
            jcbCfg.coarseZ  = coarseZ;
            jcbCfg.coarseHZ = coarseHZ;

        } else {
            jcbCfg.coarseZ  = {};
            jcbCfg.coarseHZ = {};
        }
    } else {
        auto &jcbCfg    = algo == OptAlgo::DMRG ? H1.get_iterativeLinearSolverConfig().jacobi : H2.get_iterativeLinearSolverConfig().jacobi;
        jcbCfg.coarseZ  = {};
        jcbCfg.coarseHZ = {};
    }
}

template<typename Scalar>
void solver_base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
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
void solver_base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    auto Z      = T_evecs(Eigen::all, optIdx); // Selected subspace eigenvectors
    auto Y      = T_evals(optIdx);             // Selected subspace eigenvalues
    V.noalias() = Q * Z;                       // Regular Rayleigh-Ritz
    if(b > 1) {
        // V is H2-orthonormal, so we orthonormalize it to get regular orthonormality
        hhqr.compute(V);
        V.noalias() = hhqr.householderQ().setLength(b) * MatrixType::Identity(N, b);
        // Transform the basis with applied operators
        H1V.noalias() = MultH1X(V);
        H2V.noalias() = MultH2X(V);
    } else {
        // Transform the basis with applied operators
        H1V.noalias() = H1Q * Z;
        H2V.noalias() = H2Q * Z;
    }

    S.noalias()      = H1V - H2V * Y.asDiagonal(); // Residual vector
    rNorms.noalias() = S.colwise().norm();         // Residual norm
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
    assert_orthonormal(Q, orthTolQ);
    // Here we assume that Q is orthonormal.

    // Get the indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);

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
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    assert(static_cast<size_t>(V.cols()) == optIdx.size());
    auto       Z = T_evecs(Eigen::all, optIdx);
    auto       Y = T_evals(optIdx);
    MatrixType Z_ref(T_evecs.rows(), V.cols());
    for(Eigen::Index j = 0; j < V.cols(); ++j) {
        const auto &theta = Y(j);
        MatrixType  M     = H1Q - theta * H2Q;

        svd.compute(M, Eigen::ComputeThinV);

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
    V = Q * Z_ref;
    if(b > 1) {
        // V is H2-orthonormal, so we orthonormalize it to get regular orthonormality
        hhqr.compute(V);
        V = hhqr.householderQ().setLength(b) * MatrixType::Identity(N, b);
        // Transform the basis with applied operators
        H1V = MultH1X(V);
        H2V = MultH2X(V);
    } else {
        // Transform the basis with applied operators
        H1V = H1Q * Z_ref;
        H2V = H2Q * Z_ref;
    }

    if(use_rayleigh_quotients_instead_of_evals) {
        // We replace the eigenvalues in T_evals by their rayleigh quotients
        VectorReal rq1 = (V.adjoint() * H1V).diagonal().real();
        VectorReal rq2 = (V.adjoint() * H2V).diagonal().real();
        Y              = rq1.cwiseQuotient(rq2);
    }

    S      = H1V - H2V * Y.asDiagonal();
    rNorms = S.colwise().norm();
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    auto                         Z = T_evecs(Eigen::all, optIdx);
    auto                         Y = T_evals(optIdx);
    MatrixType                   Z_ref(T_evecs.rows(), V.cols());
    for(Eigen::Index j = 0; j < V.cols(); ++j) {
        const auto &theta = Y(j);
        MatrixType  M     = HQ - theta * Q;

        svd.compute(M, Eigen::ComputeThinV);

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
    V = Q * Z_ref;
    if(b > 1) {
        // V is H2-orthonormal, so we orthonormalize it to get regular orthonormality
        hhqr.compute(V);
        V = hhqr.householderQ().setLength(b) * MatrixType::Identity(N, b);
        // Transform the basis with applied operators
        HV = MultHX(V);
    } else {
        // Transform the basis with applied operators
        HV = HQ * Z_ref;
    }

    if(use_rayleigh_quotients_instead_of_evals) {
        // We replace the eigenvalues in T_evals by their rayleigh quotients
        Y = (V.adjoint() * HV).diagonal().real();
    }

    S      = HV - V * Y.asDiagonal();
    rNorms = S.colwise().norm();
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors() {
    if(!use_refined_rayleigh_ritz) return;
    if(status.rNorms.size() == 0) throw except::runtime_error("refineRitzVectors() called before extractRitzVectors()");
    // Refined extraction
    if(algo == OptAlgo::GDMRG) {
        refinedRitzVectors(status.optIdx, V, H1V, H2V, S, status.rNorms);
    } else {
        refinedRitzVectors(status.optIdx, V, HV, S, status.rNorms);
    }
    assert_orthonormal(V, orthTolQ);
}

template<typename Scalar>
void solver_base<Scalar>::preamble() {
    // Prepare for the next iteration
    status.num_matvecs       = 0;
    status.num_precond       = 0;
    status.num_matvecs_inner = 0;
    status.num_precond_inner = 0;

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
    auto H1ir = H1.get_iterativeLinearSolverConfig();
    auto H2ir = H2.get_iterativeLinearSolverConfig();

    // Accumulate counters from the inner solvre
    status.num_matvecs_total += status.num_matvecs + status.num_matvecs_inner;
    status.num_precond_total += status.num_precond + status.num_precond_inner;
    status.time_matvecs_total += status.time_matvecs.get_time() + status.time_matvecs_inner.get_time();
    status.time_precond_total += status.time_precond.get_time() + status.time_precond_inner.get_time();

    // Eigenvalues are sorted in ascending order.
    status.oldVal  = status.optVal;
    status.optVal  = T_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
    status.absDiff = (status.optVal - status.oldVal).cwiseAbs();
    status.relDiff = status.absDiff.array() / (RealScalar{0.5} * (status.optVal + status.oldVal).array());

    status.rNorms_history.push_back(status.rNorms.topRows(nev));
    status.optVals_history.push_back(status.optVal.topRows(nev));
    status.matvecs_history.push_back(status.num_matvecs + status.num_matvecs_inner);
    while(status.rNorms_history.size() > status.max_history_size) status.rNorms_history.pop_front();
    while(status.optVals_history.size() > status.max_history_size) status.optVals_history.pop_front();
    while(status.matvecs_history.size() > status.max_history_size) status.matvecs_history.pop_front();

    constexpr auto beta         = RealScalar{0.5f};
    RealScalar     rNorm        = status.rNorms.topRows(nev).maxCoeff();                                      // The current worst-case residual norm
    RealScalar     rNorm_H1     = status.rNorms_H1.size() > 0 ? status.rNorms_H1.topRows(nev).maxCoeff() : 0; // The current worst-case residual norm
    RealScalar     rNorm_H2     = status.rNorms_H2.size() > 0 ? status.rNorms_H2.topRows(nev).maxCoeff() : 0; // The current worst-case residual norm
    RealScalar     opNorm_H     = algo == OptAlgo::GDMRG ? RealScalar{1.0f} : status.op_norm_estimate(algo);  // Use op norm on H1 and H2 but not H1/H2
    status.rNorm_below_rnormTol = rNorm < rnormTol();                                                         // Residual norm condition
    status.rNorm_below_gap      = rNorm < beta * status.gap * opNorm_H; // Gap condition for the currently selected operator (H1, H2, or H1/H2)
    status.rNorm_below_gapH1    = rNorm_H1 < beta * status.gap_H1 * std::abs(status.H1_max_eval); // Gap condition for operator H1
    status.rNorm_below_gapH2    = rNorm_H2 < beta * status.gap_H2 * std::abs(status.H2_max_eval); // Gap condition for operator H2

    if(status.rNorm_below_rnormTol and status.rNorm_below_gap) {
        std::string msg_rnorm_gap = fmt::format(" | gap {:.3e}", fp(status.gap));
        if constexpr(settings::debug_solver) {
            if(algo == OptAlgo::GDMRG and dev_append_extra_blocks_to_basis) {
                msg_rnorm_gap =
                    fmt::format(" | H1|H2: rNorm {:.2e}|{:.2e} gap {:.3e}|{:.3e}", fp(rNorm_H1), fp(rNorm_H2), fp(status.gap_H1), fp(status.gap_H2));
            }
        }
        status.stopMessage.emplace_back(
            fmt::format("converged rNorm {::.3e} < tol {:.3e}{}", fv(VectorReal(status.rNorms.topRows(nev))), fp(rnormTol()), msg_rnorm_gap));
        status.stopReason |= StopReason::converged_rNorm;
    }
    // if(use_adaptive_inner_tolerance and optVal_has_saturated() and rNorm_has_saturated()) {
    // status.stopMessage.emplace_back(fmt::format("saturated rNorm {::.3e} (tol {:.3e})", fv(VectorReal(status.rNorms.topRows(nev))), fp(rnormTol())));
    // status.stopReason |= StopReason::saturated_rNorm;
    // }

    if(max_iters >= 0l and status.iter + 1 >= max_iters) {
        status.stopMessage.emplace_back(fmt::format("iters ({}) >= maxiter ({})", status.iter + 1, max_iters));
        status.stopReason |= StopReason::max_iterations;
    }
    if(max_matvecs >= 0l and status.num_matvecs_total >= max_matvecs) {
        status.stopMessage.emplace_back(fmt::format("num_matvecs_total ({}) >= max_matvecs ({})", status.num_matvecs_total, max_matvecs));
        status.stopReason |= StopReason::max_matvecs;
    }

    if(status.stopReason != StopReason::none) return;
}

template<typename Scalar>
void solver_base<Scalar>::printStatus() {
    RealScalar  rNorm_H1      = status.rNorms_H1.size() > 0 ? status.rNorms_H1.topRows(nev).maxCoeff() : 0; // The current worst-case residual norm
    RealScalar  rNorm_H2      = status.rNorms_H2.size() > 0 ? status.rNorms_H2.topRows(nev).maxCoeff() : 0; // The current worst-case residual norm
    std::string msg_rnorm_gap = fmt::format(" | gap {:.3e}", fp(status.gap));
    if constexpr(settings::debug_solver) {
        if(algo == OptAlgo::GDMRG) {
            msg_rnorm_gap = fmt::format(" | H1|H2: rNorm {:.2e}|{:.2e} gap {:.3e}|{:.3e}", fp(rNorm_H1), fp(rNorm_H2), fp(status.gap_H1), fp(status.gap_H2));
        }
    }

    std::string optValMsg = optVal_has_saturated() ? "SAT" : "";
    std::string rNormMsg  = rNorm_has_saturated() ? "SAT" : "";
    std::string rCorrMsg;
    switch(residual_correction_type_internal) {
        case ResidualCorrectionType::NONE: rCorrMsg = "NO"; break;
        case ResidualCorrectionType::CHEAP_OLSEN: rCorrMsg = "CO"; break;
        case ResidualCorrectionType::FULL_OLSEN: rCorrMsg = "FO"; break;
        case ResidualCorrectionType::JACOBI_DAVIDSON: rCorrMsg = "JD"; break;
        case ResidualCorrectionType::AUTO: rCorrMsg = "AU"; break;
    }
    auto        H1ir            = H1.get_iterativeLinearSolverConfig();
    auto        H2ir            = H2.get_iterativeLinearSolverConfig();
    std::string innerMsg        = status.num_matvecs_inner == 0 ? std::string()
                                                                : fmt::format("[inner: ({}) mv {:5} err {:.2e} tol {:.2e} t {:.1e}s] ", //
                                                                              rCorrMsg,                                                 //
                                                                              status.num_matvecs_inner,                                 //
                                                                              fp(std::max(H1ir.result.error, H2ir.result.error)),       //
                                                                              fp(std::max(H1ir.tolerance, H2ir.tolerance)),             //
                                                                              fp(H1ir.result.time + H2ir.result.time));
    bool        log_low_maxiter = max_iters < 10;
    bool        log_jacobi_prec = preconditioner_type == eig::Preconditioner::JACOBI and status.iter % 100 == 0;
    bool        log_solve_prec  = preconditioner_type == eig::Preconditioner::SOLVE;
    if(log_low_maxiter or log_jacobi_prec or log_solve_prec)
        eiglog->debug("it {:3} mv {:3} pc {:3} t {:.1e}s {}"
                      "optVal {::.16f}{} rNorms {::.8e}{} ({:9.2e}/mv) col {:2} x {} ritz {} rNormTol {:.3e} tol {:.2e} "
                      "op norm {:.2e} cond {:.2e}{}",
                      status.iter + 1,                   //
                      status.num_matvecs,                //
                      status.num_precond,                //
                      status.time_elapsed.restart_lap(), //
                      innerMsg,
                      fv(status.optVal),                          //
                      optValMsg,                                  //
                      fv(VectorReal(status.rNorms.topRows(nev))), //
                      rNormMsg,                                   //
                      fp(get_rNorms_log10_change_per_matvec()),   //
                      Q.cols() / b,                               //
                      b,                                          //
                      enum2sv(ritz),                              //
                      fp(rnormTol()),                             //
                      fp(tol),                                    //
                      fp(status.op_norm_estimate(algo)),          //
                      fp(status.condition),                       //
                      msg_rnorm_gap);
}