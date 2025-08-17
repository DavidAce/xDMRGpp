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
//
// template<typename Scalar>
// solver_base<Scalar>::VectorReal solver_base<Scalar>::rnormTol(Eigen::Ref<VectorReal> evals) const {
//     if(use_relative_rnorm_tolerance) {
//         return get_op_norm_estimates(evals) * tol * RealScalar{2};
//     }
//
//     else
//         return VectorReal::Ones(evals.size()) * tol * RealScalar{2};
// }

template<typename Scalar>
solver_base<Scalar>::RealScalar solver_base<Scalar>::rNormTol(Eigen::Index n) const {
    if(n < 0 or n > nev) throw except::runtime_error("rNormTol: n == {} is out of bounds 0 <= n <= nev[{}]", n, nev);
    RealScalar tol_eff = std::max(tol, 10 * orthTol);
    RealScalar lambda  = std::abs(status.eigVal(n));
    if(use_relative_rnorm_tolerance) {
        switch(algo) {
            case OptAlgo::DMRG: [[fallthrough]];
            case OptAlgo::DMRGX: [[fallthrough]];
            case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
            case OptAlgo::XDMRG: {
                RealScalar hvnorm = std::max(RealScalar{1e-30f}, HV.col(n).norm());
                return tol_eff * std::max({RealScalar{1}, hvnorm, lambda});
            }
            case OptAlgo::GDMRG: {
                RealScalar h1vnorm = std::max(RealScalar{1e-30f}, H1V.col(n).norm());
                RealScalar h2vnorm = std::max(RealScalar{1e-30f}, H2V.col(n).norm());
                RealScalar maxnorm = std::max({RealScalar{1}, h1vnorm, h2vnorm * lambda});
                return tol_eff * maxnorm;
            }
            default: throw except::logic_error("rNormTol: unhandled algo");
        }
    } else {
        return tol_eff;
    }
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
Eigen::Index solver_base<Scalar>::get_jcbMaxBlockSize() const {
    assert(H1.get_jcbMaxBlockSize() == H2.get_jcbMaxBlockSize());
    return H1.get_jcbMaxBlockSize();
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
    H2.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::DEF);
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
            return std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
        }
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
        case OptAlgo::XDMRG: {
            return std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});
        }
        case OptAlgo::GDMRG: {
            auto H1_norm = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
            auto H2_norm = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});

            if(not eigval.has_value()) {
                eigval = RealScalar{1};
                if(T_evals.size() > 0) {
                    auto select_1 = get_ritz_indices(ritz, 0, 1, T_evals);
                    eigval        = T_evals(select_1)[0];
                }
            }
            return H1_norm + std::abs(eigval.value()) * H2_norm;
        }

        default: throw except::runtime_error("unrecognized algo");
    }
}

template<typename Scalar>
typename solver_base<Scalar>::VectorReal solver_base<Scalar>::get_op_norm_estimates(Eigen::Ref<VectorReal> eigvals) const {
    switch(algo) {
        case OptAlgo::DMRG: {
            RealScalar H1_norm = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
            VectorReal H1_norms(eigvals.size());
            H1_norms.setConstant(H1_norm);
            return H1_norms;
        }
        case OptAlgo::DMRGX: [[fallthrough]];
        case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
        case OptAlgo::XDMRG: {
            RealScalar H2_norm = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});
            VectorReal H2_norms(eigvals.size());
            H2_norms.setConstant(H2_norm);
            return H2_norms;
        }
        case OptAlgo::GDMRG: {
            RealScalar H1_norm = std::max({std::abs(status.T1_min_eval), std::abs(status.T1_max_eval), H1.get_op_norm()});
            RealScalar H2_norm = std::max({std::abs(status.T2_min_eval), std::abs(status.T2_max_eval), H2.get_op_norm()});
            return H1_norm + eigvals.cwiseAbs().array() * H2_norm;
        }

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
    VectorReal vals           = status.eigVal.cwiseAbs().array() + eps;
    VectorReal stds           = get_standard_deviations(status.eigVals_history, false);
    VectorReal rels           = stds.cwiseQuotient(vals);
    VectorIdxT stds_saturated = (stds.array() < RealScalar{1e-2f}).template cast<Eigen::Index>();
    VectorIdxT rels_saturated = (rels.array() < RealScalar{1e-5f}).template cast<Eigen::Index>();
    // eiglog->info("eigVal stds {::.5e} {} rels {::.5e} {}", fv(stds), stds_saturated, fv(rels), rels_saturated);
    return stds_saturated.all() or rels_saturated.all();
}

template<typename Scalar>
void solver_base<Scalar>::adjust_preconditioner_tolerance(const Eigen::Ref<const MatrixType> &S) {
    // if(status.iter_last_preconditioner_tolerance_adjustment == status.iter) return;
    H1.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.T1_max_eval), std::abs(status.T1_min_eval)) / std::min(std::abs(status.T1_max_eval), std::abs(status.T1_min_eval));
    H2.get_iterativeLinearSolverConfig().jacobi.cond =
        std::max(std::abs(status.T2_max_eval), std::abs(status.T2_min_eval)) / std::min(std::abs(status.T2_max_eval), std::abs(status.T2_min_eval));
    H1H2.get_iterativeLinearSolverConfig().jacobi.cond = status.condition;

    if(!use_adaptive_inner_tolerance) return;
    auto Snorm = S.leftCols(nev).colwise().norm().minCoeff();
    if(dev_thick_jd_projector) Snorm = std::pow(Snorm, RealScalar{2});

    auto set_cfg = [&](IterativeLinearSolverConfig<Scalar> &cfg) {
        auto oldtol = std::max(eps, cfg.tolerance);
        auto oldits = status.num_iters_inner_prev;

        cfg.tolerance = oldtol; // std::min<RealScalar>({oldtol, std::sqrt(Snorm)});

        if(oldits > 0 and oldits < 200l) cfg.tolerance *= half;
        if(oldits > 1000l) cfg.tolerance *= RealScalar{2};

        cfg.tolerance = std::clamp(cfg.tolerance, eps, RealScalar{5e-1f});
        // RealScalar maxiters = RealScalar{50l} / cfg.tolerance;
        cfg.maxiters = 2000l; // std::clamp(safe_cast<long>(maxiters), 50l, 200l);
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
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
        H1H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{0.5f};
    }

    if(rNorm_log10_decrease < RealScalar{-3.0f}) {
        // Decreasing more than two orders of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
        H1H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{5};
    } else if(rNorm_log10_decrease < RealScalar{-2.1f}) {
        // Decreasing more than one order of magnitude per iteration,
        // We don't really need to decrease that fast, we are likely spending too many iterations.
        H1.get_iterativeLinearSolverConfig().tolerance *= RealScalar{2};
        H2.get_iterativeLinearSolverConfig().tolerance *= RealScalar{2};
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
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH(const Eigen::Ref<const MatrixType> &X) {
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
        case OptAlgo::GDMRG: throw except::runtime_error("MultH: GDMRG is not suitable, use MultH1X or MultH2X instead");
        default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
    }
    return HX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH1(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH1: should only be called by GDMRG");
    auto token_matvecs = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H1.MultAX(X);
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultH2(const Eigen::Ref<const MatrixType> &X) {
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultH2: should only be called by GDMRG");
    auto token_matvecs = status.time_matvecs.tic_token();
    status.num_matvecs += X.cols();
    return H2.MultAX(X);
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

    auto &H1ir = H1.get_iterativeLinearSolverConfig().result;
    auto &H2ir = H2.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_iters_inner += H1ir.iters + H2ir.iters;
    status.num_matvecs_inner += H1ir.matvecs + H2ir.matvecs;
    status.num_precond_inner += H1ir.precond + H2ir.precond;
    status.time_matvecs_inner += H1ir.time_matvecs + H2ir.time_matvecs;
    status.time_precond_inner += H1ir.time_precond + H2ir.time_precond;
    H1ir.reset();
    H2ir.reset();
    return HPX;
}
template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP1(const Eigen::Ref<const MatrixType>                  &X,
                                                                     [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                     std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    auto token_precond                                  = status.time_precond.tic_token();
    H1.get_iterativeLinearSolverConfig().initialGuess   = initialGuess.value_or(MatrixType{});
    H1.get_iterativeLinearSolverConfig().jacobi.skipjcb = dev_skipjcb;
    MatrixType HPX                                      = H1.MultPX(X);
    auto      &H1ir                                     = H1.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_iters_inner += H1ir.iters;
    status.num_matvecs_inner += H1ir.matvecs;
    status.num_precond_inner += H1ir.precond;
    status.time_matvecs_inner += H1ir.time_matvecs;
    status.time_precond_inner += H1ir.time_precond;
    H1ir.reset();
    return HPX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP2(const Eigen::Ref<const MatrixType>                  &X,
                                                                     [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                     std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP2: should only be called by GDMRG");
    auto token_precond                                  = status.time_precond.tic_token();
    H2.get_iterativeLinearSolverConfig().initialGuess   = initialGuess.value_or(MatrixType{});
    H2.get_iterativeLinearSolverConfig().jacobi.skipjcb = dev_skipjcb;
    MatrixType HPX                                      = H2.MultPX(X);
    auto      &H2ir                                     = H2.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_iters_inner += H2ir.iters;
    status.num_matvecs_inner += H2ir.matvecs;
    status.num_precond_inner += H2ir.precond;
    status.time_matvecs_inner += H2ir.time_matvecs;
    status.time_precond_inner += H2ir.time_precond;
    H2ir.reset();
    return HPX;
}

template<typename Scalar>
typename solver_base<Scalar>::MatrixType solver_base<Scalar>::MultP1P2(const Eigen::Ref<const MatrixType>                  &X,
                                                                       [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                                       std::optional<const Eigen::Ref<const MatrixType>>    initialGuess) {
    // Preconditioning
    if(algo != OptAlgo::GDMRG) throw except::runtime_error("MultP1P2: should only be called by GDMRG");
    auto token_precond                                    = status.time_precond.tic_token();
    H1H2.get_iterativeLinearSolverConfig().initialGuess   = initialGuess.value_or(MatrixType{});
    H1H2.get_iterativeLinearSolverConfig().jacobi.skipjcb = dev_skipjcb;
    MatrixType H1H2PX                                     = H1H2.MultPX(X, evals);
    auto      &H1H2ir                                     = H1H2.get_iterativeLinearSolverConfig().result;
    status.num_precond += X.cols();
    status.num_iters_inner += H1H2ir.iters;
    status.num_matvecs_inner += H1H2ir.matvecs;
    status.num_precond_inner += H1H2ir.precond;
    status.time_matvecs_inner += H1H2ir.time_matvecs;
    status.time_precond_inner += H1H2ir.time_precond;
    H1H2ir.reset();
    return H1H2PX;
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
    auto ProjectOpL = [this, &V](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        auto t_pl = tid::tic_token("ProjectOpL", tid::level::higher);
        return X - V * (V.adjoint() * X).eval();
    };
    auto ProjectOpR = [this, &V](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        auto t_pr = tid::tic_token("ProjectOpR", tid::level::higher);
        return X - V * (V.adjoint() * X).eval();
    };

    // Right-hand side (projected)
    MatrixType RHS = -ProjectOpL(S);

    MatrixType D(S.rows(), S.cols()); // accumulate the result from the JD correction equation

    for(Eigen::Index i = 0; i < S.cols(); ++i) { // We use block size b
        // v: current Ritz vector, s: current residual, Bv: either I*v (Euclidean) or H2*v (H2-orthonormal)
        auto              d   = D.col(i);   // The solution vector i
        const RealScalar &th  = evals(i);   // The ritz value for this ritz vector
        const VectorType &rhs = RHS.col(i); // Right-hand side (projected)

        if(i > 0) {
            // This residual is not in the "active" set. Default to Cheap Olsen + CG instead
            const VectorType &s  = S.col(i); // The residual vector i
            const VectorType &v  = V.col(i); // The residual vector i
            auto              ev = evals.middleRows(i, 1);
            D.col(i).noalias()   = algo == OptAlgo::DMRG ? MultP1(s, ev) : (use_h1h2_preconditioner ? MultP1P2(s, ev) : MultP2(s, ev));
            D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
        } else {
            auto  token_precond = status.time_precond.tic_token();
            auto &H             = algo == OptAlgo::DMRG ? H1 : (use_h1h2_preconditioner ? H1H2 : H2);
            H.CalcPc(th); // Compute the block-jacobi preconditioner (do llt/ldlt on all blocks)

            IterativeLinearSolverConfig<Scalar> cfg = H.get_iterativeLinearSolverConfig(); // Get the jacobi blocks
            cfg.result                              = {};
            cfg.matdef                              = MatDef::IND;
            cfg.precondType                         = PreconditionerType::JACOBI;
            cfg.jacobi.skipjcb                      = dev_skipjcb;

            // Define the residual matrix-vector operator depending on the different DMRG algorithms
            auto ResidualOp = [this, th, &H](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                auto       t_residualop = tid::tic_scope("ResidualOp", tid::level::higher);
                MatrixType HX(X.rows(), X.cols());
                switch(algo) {
                    case OptAlgo::DMRG: [[fallthrough]];
                    case OptAlgo::DMRGX: [[fallthrough]];
                    case OptAlgo::HYBRID_DMRGX: [[fallthrough]];
                    case OptAlgo::XDMRG: {
                        HX.noalias() = H.MultAX(X) - th * X;
                        status.num_matvecs_inner += X.cols();
                        break;
                    }
                    case OptAlgo::GDMRG: {
                        // Generalized problem
                        HX.noalias() = H1.MultAX(X) - th * H2.MultAX(X);
                        status.num_matvecs_inner += 2 * X.cols();
                        auto t_h1 = tid::tic_token("H1X", tid::higher, H1.t_multAx->get_last_interval());
                        auto t_h2 = tid::tic_token("H2X", tid::higher, H2.t_multAx->get_last_interval());
                        break;
                    }
                    default: throw except::runtime_error("unknown algorithm {}", enum2sv(algo));
                }
                return HX;
            };

            auto JDop = JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR, MatrixOp);

            d.noalias() = JacobiDavidsonSolver(JDop, rhs, cfg);
            d.noalias() = ProjectOpR(d);

            // status.num_matvecs_inner += cfg.result.matvecs;
            status.num_iters_inner += cfg.result.iters;
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

    auto ProjectOpL = [this, &V, &H2V](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        auto t_pl = tid::tic_token("ProjectOpL", tid::level::higher);
        return X - H2V * (V.adjoint() * X).eval();
    };
    auto ProjectOpR = [this, &V, &H2V](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
        auto t_pr = tid::tic_token("ProjectOpR", tid::level::higher);
        return X - V * (H2V.adjoint() * X).eval();
    };

    // Right-hand side (projected)
    MatrixType RHS = -ProjectOpL(S);

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
            D.col(i).noalias()   = algo == OptAlgo::DMRG ? MultP1(s, ev) : (use_h1h2_preconditioner ? MultP1P2(s, ev) : MultP2(s, ev));
            D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
        } else {
            auto  token_precond = status.time_precond.tic_token();
            auto &H             = use_h1h2_preconditioner ? H1H2 : H2; // Typically use_h1h2_preconditioner is false
            H.CalcPc(th);                                              // Compute the block-jacobi preconditioner

            IterativeLinearSolverConfig<Scalar> cfg = H.get_iterativeLinearSolverConfig(); // Get the jacobi blocks
            cfg.result                              = {};
            cfg.matdef                              = MatDef::IND;
            cfg.precondType                         = PreconditionerType::JACOBI;
            cfg.jacobi.skipjcb                      = dev_skipjcb;
            // Define the matrix-vector operator for the H2 operator
            auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                status.num_matvecs_inner += X.cols();
                return H2.MultAX(X);
            };

            // Define the residual matrix-vector operator depending on the different DMRG algorithms
            auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                auto t_rop = tid::tic_token("ResidualOp", tid::level::higher);
                status.num_matvecs_inner += 2 * X.cols();
                return H1.MultAX(X) - th * H2.MultAX(X);
            };

            // auto Kop = [&](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return ProjectOpL(ResidualOp(ProjectOpR(X))); };

            auto JDop = JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR, MatrixOp);

            d.noalias() = JacobiDavidsonSolver(JDop, rhs, cfg);
            d.noalias() = ProjectOpR(d);

            status.num_iters_inner += cfg.result.iters;
            status.num_precond_inner += cfg.result.precond;
            status.time_matvecs_inner += cfg.result.time_matvecs;
            status.time_precond_inner += cfg.result.time_precond;

            H.get_iterativeLinearSolverConfig().result += cfg.result;
            // RealScalar error = (Kop(d) - rhs).norm() / rhs.norm();
            // eiglog->info("JD solve {}: |rhs|={:.5e}  |d|={:.5e} |Kd-rhs|/|rhs|={:.5e} error={:.5e} iters={} mvs {}", i, fp(rhs.norm()), fp(d.norm()),
            // fp(error), fp(cfg.result.error), cfg.result.iters, 2 * cfg.result.matvecs);
        }
    }
    status.num_precond += b; // This routine is a preconditioner
    return D;                // N x b, enrichment directions
}

template<typename Scalar> typename solver_base<Scalar>::MatrixType solver_base<Scalar>::get_sBlock(const MatrixType &S_in, fMultP_t MultP) {
    // Make a residual block "S = (HQ-λQ)"
    MatrixType S = S_in;

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
        B = V_prev.adjoint() * W;
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
        X = X(Eigen::all, active_columns).eval(); // Shrink keeping only nonzeros
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
    X = X(Eigen::all, active_columns).eval(); // Shrink keeping only nonzeros
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
    using Index = Eigen::Index;

    assert(Y.rows() == H2Y.rows());
    assert(Y.cols() == H2Y.cols());
    const Index m = Y.cols();
    if(m < 2 || num_sweeps <= 0) return;

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
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : normTol;
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
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
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
        MatrixType Gram      = X.adjoint() * H2X;
        auto       orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        RealScalar xnorm     = X.norm();
        RealScalar h2xnorm   = H2X.norm();

        Eigen::SelfAdjointEigenSolver<MatrixType> esG(Gram);
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

        if(orthError > finalTol) {
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
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask    = VectorIdxT::Ones(Y.cols());
    m.maskTol = std::max(m.maskTol, normTol * Y.cols());

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
                        if(m.mask(j) == 0) { Y.col(j).setRandom(); }
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
    if(Y.cols() == 0) return;

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

    H1Y = MultH1(Y);
    H2Y = MultH2(Y);
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
        G               = (G + G.adjoint()) * half;
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
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo != OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask    = VectorIdxT::Ones(Y.cols());
    m.maskTol = std::max(m.maskTol, normTol * Y.cols());

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
                        if(m.mask(j) == 0) { Y.col(j).setRandom(); }
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
    if(Y.cols() == 0) return;

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

    HY = MultH(Y);
    assert_l2_orthonormal(Y, m);
}
// template<typename Scalar>
// void solver_base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &HY, OrthMeta &m) {
//     // Orthonormalization with respect to the Euclidean inner product, i.e. Y.adjoint()*Y = I
//
//     if(Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
//     assert(!(algo == OptAlgo::GDMRG and use_h2_inner_product) && "block_l2_orthonormalize is for the L2 inner product");
//     assert_allFinite(Y);
//
//     // DGKS clean every block against previous blocks
//     Eigen::Index n_blocks_y = Y.cols() / b;
//     int          maxreps    = 2;
//     if(m.proj_sum_h.size() != n_blocks_y) m.proj_sum_h = VectorReal::Zero(n_blocks_y);
//     if(m.scale_log.size() != n_blocks_y) m.scale_log = VectorReal::Zero(n_blocks_y);
//     for(int rep = 0; rep < maxreps; ++rep) {
//         for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
//             if(m.mask(j) == 0) continue;
//             auto yj = Y.middleCols(j * b, b);
//             for(Eigen::Index i = 0; i < j; ++i) {
//                 if(m.mask(i) == 0) continue; // skip dropped block
//                 auto       yi   = Y.middleCols(i * b, b);
//                 MatrixType proj = yi.adjoint() * yj;
//                 yj.noalias() -= yi * proj; // Remove projection
//             }
//
//             // Normalize the block itself
//             hhqr.compute(yj);
//             yj            = hhqr.householderQ().setLength(yj.cols()) * MatrixType::Identity(yj.rows(), yj.cols());
//             VectorReal rj = hhqr.matrixQR().diagonal().cwiseAbs().topRows(yj.cols());
//             if(rj.minCoeff() < m.maskTol) {
//                 m.mask(j) = 0;
//                 yj.setZero();
//             }
//         }
//         m.Gram      = Y.adjoint() * Y;                          // m×m  (Hermitian PD)
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//
//         // Determine whether to do another rep
//         if(rep + 1 < maxreps) {
//             RealScalar Gmax = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//             if(Gmax < m.maskTol) break; // skip the next rep
//         }
//     }
//
//     // Mask
//     auto mask_old_size = m.mask.size();
//     mask_col_blocks(Y, m);
//     if(m.maskPolicy == MaskPolicy::COMPRESS) {
//         compress_col_blocks(Y, m.mask);
//         m.mask = VectorIdxT::Ones(Y.cols() / b);
//     }
//
//     if(m.mask.size() != mask_old_size and m.mask.sum() > 0 and Y.cols() >= b) {
//         m.Gram      = Y.adjoint() * Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//     }
//
//     HY = MultH(Y);
//     assert_l2_orthonormal(Y, m);
// }

// template<typename Scalar>
// void solver_base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &HY, OrthMeta &m) {
//     // Orthonormalization with respect to the Euclidean inner product, i.e. Y.adjoint()*Y = I
//
//     if(Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
//     assert(!(algo == OptAlgo::GDMRG and use_h2_inner_product) && "block_l2_orthonormalize is for the L2 inner product");
//     assert_allFinite(Y);
//
//     // DGKS clean every block against previous blocks
//     Eigen::Index n_blocks_y = Y.cols() / b;
//     int          maxreps    = 2;
//     if(m.proj_sum_h.size() != n_blocks_y) m.proj_sum_h = VectorReal::Zero(n_blocks_y);
//     if(m.scale_log.size() != n_blocks_y) m.scale_log = VectorReal::Zero(n_blocks_y);
//     for(int rep = 0; rep < maxreps; ++rep) {
//         for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
//             if(m.mask(j) == 0) continue;
//             auto yj = Y.middleCols(j * b, b);
//             for(Eigen::Index i = 0; i < j; ++i) {
//                 if(m.mask(i) == 0) continue; // skip dropped block
//                 auto       yi   = Y.middleCols(i * b, b);
//                 MatrixType proj = yi.adjoint() * yj;
//                 yj.noalias() -= yi * proj; // Remove projection
//             }
//
//             // Normalize the block itself
//             hhqr.compute(yj);
//             yj            = hhqr.householderQ().setLength(yj.cols()) * MatrixType::Identity(yj.rows(), yj.cols());
//             VectorReal rj = hhqr.matrixQR().diagonal().cwiseAbs().topRows(yj.cols());
//             if(rj.minCoeff() < m.maskTol) {
//                 m.mask(j) = 0;
//                 yj.setZero();
//             }
//         }
//         m.Gram      = Y.adjoint() * Y;                          // m×m  (Hermitian PD)
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//
//         // Determine whether to do another rep
//         if(rep + 1 < maxreps) {
//             RealScalar Gmax = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//             if(Gmax < m.maskTol) break; // skip the next rep
//         }
//     }
//
//     // Mask
//     auto mask_old_size = m.mask.size();
//     mask_col_blocks(Y, m);
//     if(m.maskPolicy == MaskPolicy::COMPRESS) {
//         compress_col_blocks(Y, m.mask);
//         m.mask = VectorIdxT::Ones(Y.cols() / b);
//     }
//
//     if(m.mask.size() != mask_old_size and m.mask.sum() > 0 and Y.cols() >= b) {
//         m.Gram      = Y.adjoint() * Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//     }
//
//     HY = MultH(Y);
//     assert_l2_orthonormal(Y, m);
// }
//
// template<typename Scalar>
// void solver_base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
//     // Orthonormalization with respect to the Euclidean inner product, i.e. Y.adjoint()*Y = I
//
//     if(Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
//     assert(!(algo == OptAlgo::GDMRG and use_h2_inner_product) && "block_l2_orthonormalize is for the L2 inner product");
//     assert_allFinite(Y);
//
//     // DGKS clean every block against previous blocks
//     Eigen::Index n_blocks_y = Y.cols() / b;
//     if(m.mask.size() != n_blocks_y) m.mask = VectorIdxT::Ones(n_blocks_y);
//     int maxreps = 3;
//     for(int rep = 0; rep < maxreps; ++rep) {
//         for(Eigen::Index j = 0; j < n_blocks_y; ++j) {
//             if(m.mask(j) == 0) continue;
//             auto yj = Y.middleCols(j * b, b);
//             for(Eigen::Index i = 0; i < j; ++i) {
//                 if(m.mask(i) == 0) continue; // skip dropped block
//                 auto       yi   = Y.middleCols(i * b, b);
//                 MatrixType proj = yi.adjoint() * yj;
//                 yj.noalias() -= yi * proj; // Remove projection
//             }
//
//             // Normalize the block itself
//             hhqr.compute(yj);
//             yj            = hhqr.householderQ().setLength(yj.cols()) * MatrixType::Identity(yj.rows(), yj.cols());
//             VectorReal rj = hhqr.matrixQR().diagonal().cwiseAbs().topRows(yj.cols());
//             if(rj.minCoeff() < m.maskTol) {
//                 m.mask(j) = 0;
//                 yj.setZero();
//             }
//         }
//         m.Gram      = Y.adjoint() * Y;                          // m×m  (Hermitian PD)
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//
//         // Determine whether to do another rep
//         if(rep + 1 < maxreps) {
//             RealScalar Gmax = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//             if(Gmax < m.maskTol) break; // skip the next rep
//         }
//     }
//
//     // Mask
//     auto mask_old_size = m.mask.size();
//     mask_col_blocks(Y, m);
//     if(m.maskPolicy == MaskPolicy::COMPRESS) {
//         compress_col_blocks(Y, m.mask);
//         m.mask = VectorIdxT::Ones(Y.cols() / b);
//     }
//
//     if(m.mask.size() != mask_old_size and m.mask.sum() > 0 and Y.cols() >= b) {
//         m.Gram      = Y.adjoint() * Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//     }
//
//     H1Y = MultH1(Y);
//     H2Y = MultH2(Y);
//     assert_l2_orthonormal(Y, m);
// }
//
// template<typename Scalar>
// void solver_base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &HX, MatrixType &Y, MatrixType &HY, OrthMeta &m) {
//     if(X.cols() == 0 || Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
//     assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
//     assert(!(algo == OptAlgo::GDMRG and use_h2_inner_product) && "block_l2_orthogonalize is for the L2 inner product");
//
//     assert_allFinite(X);
//     assert_allFinite(HX);
//     assert_allFinite(Y);
//     assert_l2_orthonormal(X);
//
//     // DGKS clean Y against X
//     auto maxReps = 3;
//     for(int rep = 0; rep < 4; ++rep) {
//         MatrixType proj = X.adjoint() * Y;
//         Y.noalias() -= X * proj; // Remove projection
//         // DGKS drop test – skip next pass if it already cleaned well
//         if(rep + 1 < maxReps) {
//             if(proj.norm() < 10 * eps) break;
//         }
//     }
//
//     m.Gram      = Y.adjoint() * Y;
//     m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//     m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//
//     // Mask and compress
//     auto mask_old_size = m.mask.size();
//     mask_col_blocks(Y, m);
//     if(m.maskPolicy == MaskPolicy::COMPRESS) {
//         compress_col_blocks(Y, m.mask);
//         compress_row_blocks(m.scale_log, m.mask);
//         m.mask = VectorIdxT::Ones(Y.cols() / b);
//     }
//
//     if(m.mask.size() != mask_old_size and m.mask.sum() > 0 and Y.cols() >= b) {
//         m.Gram      = Y.adjoint() * Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//     }
//
//     // orthonormalize Y and refresh HY
//     block_l2_orthonormalize(Y, HY, m);
//     assert_l2_orthogonal(X, Y);
// }

template<typename Scalar>
void solver_base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &HX, MatrixType &Y, MatrixType &HY, OrthMeta &m) {
    if(X.cols() == 0 || Y.cols() == 0) return;
    if(m.mask.size() > 0 && m.mask.sum() == 0) return;
    assert(algo != OptAlgo::GDMRG);
    assert(!use_h2_inner_product);

    assert_allFinite(X);
    assert_allFinite(HX);
    assert_allFinite(Y);
    assert_allFinite(HY);
    assert_l2_orthonormal(X);

    m.orthTol = std::max(m.orthTol, orthTol * Y.cols());

    m.Gram      = X.adjoint() * Y;
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
    m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

    MatrixType xGram = X.adjoint() * X;
    // RealScalar xOrthError = (xGram - MatrixType::Identity(xGram.cols(), xGram.rows())).norm();
    // DGKS clean Y against X
    Eigen::Index maxReps = 3;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        Y.noalias() -= X * m.Gram;

        // orthonormalize Y and refresh HY
        block_l2_orthonormalize(Y, HY, m);

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
    if(X.cols() == 0 || Y.cols() == 0) return;
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

    m.orthTol   = std::max(m.orthTol, orthTol * Y.cols());
    m.Gram      = X.adjoint() * Y;
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
    m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

    MatrixType xGram = X.adjoint() * X;
    // RealScalar xOrthError = xGram.norm();

    // DGKS clean Y against X
    Eigen::Index maxReps = 3;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        Y.noalias() -= X * m.Gram;

        // orthonormalize Y and refresh H1Y and H2Y
        block_l2_orthonormalize(Y, H1Y, H2Y, m);

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

// template<typename Scalar>
// void solver_base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y,
//                                                  MatrixType &H2Y, OrthMeta &m) {
//     if(X.cols() == 0 || Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(Y.cols() % b == 0 && "Y's column count must be a multiple of the block width b.");
//     assert(X.cols() % b == 0 && "X's column count must be a multiple of the block width b.");
//     assert(!(algo == OptAlgo::GDMRG and use_h2_inner_product) && "block_l2_orthogonalize is for the L2 inner product");
//
//     assert_allFinite(X);
//     assert_allFinite(H1X);
//     assert_allFinite(H2X);
//     assert_allFinite(Y);
//     assert_l2_orthonormal(X);
//
//     // DGKS clean Y against X
//     auto maxReps = 3;
//     for(int rep = 0; rep < maxReps; ++rep) {
//         MatrixType proj = X.adjoint() * Y; // X.cols() x Y.cols()
//         Y.noalias() -= X * proj;           // Remove projection
//
//         // DGKS drop test – skip 2nd pass if it already cleaned well
//         if(rep + 1 < maxReps) {
//             if(proj.norm() < 10 * eps) break;
//         }
//     }
//
//     m.Gram      = Y.adjoint() * Y;
//     m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//     m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//
//     // Mask and compress
//     Eigen::Index n_blocks_y = Y.cols() / b;
//     if(m.mask.size() != n_blocks_y) m.mask = VectorIdxT::Ones(n_blocks_y);
//     auto mask_old_size = m.mask.size();
//     mask_col_blocks(Y, m);
//     if(m.maskPolicy == MaskPolicy::COMPRESS) {
//         compress_col_blocks(Y, m.mask);
//         m.mask = VectorIdxT::Ones(Y.cols() / b);
//     }
//
//     if(m.mask.size() != mask_old_size and m.mask.sum() > 0 and Y.cols() >= b) {
//         m.Gram      = Y.adjoint() * Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
//         m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
//     }
//
//     // orthonormalize Y and refresh H1Y and H2Y
//     block_l2_orthonormalize(Y, H1Y, H2Y, m);
//     assert_l2_orthogonal(X, Y, m);
// }

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthonormalize_dgks(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG and use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask        = VectorIdxT::Ones(Y.cols());
    m.proj_sum_h2 = VectorReal::Zero(Y.cols());
    m.scale_log   = VectorReal::Zero(Y.cols());

    H2Y       = MultH2(Y);
    m.maskTol = std::max(m.maskTol, normTol * std::sqrt(status.op_norm_estimate));

    auto handle_masked_columns = [&]() {
        if(m.mask.sum() != Y.cols()) {
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS: {
                    eiglog->debug("block_h2_orthonormalize_dgks: Compressing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    compress_cols(Y, m.mask);
                    compress_cols(H2Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                }
                case MaskPolicy::RANDOMIZE: {
                    eiglog->debug("block_h2_orthonormalize_dgks: Randomizing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) {
                            Y.col(j).setRandom();
                            H2Y.col(j) = MultH2(Y.col(j));
                        }
                    }
                    break;
                }
                default: throw except::runtime_error("Unrecognized mask policy");
            }
        }
    };
    auto dot_fp128 = [](Eigen::Ref<VectorType> a, Eigen::Ref<VectorType> b) -> Scalar {
        // high-precision inner product
        using LScalar = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
        return static_cast<Scalar>(a.template cast<LScalar>().dot(b.template cast<LScalar>()));
    };
    // Initial mask
    m.Rdiag = VectorReal::Zero(Y.cols());
    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto yj    = Y.col(j);
        auto h2yj  = H2Y.col(j);
        m.Rdiag(j) = std::sqrt(std::abs(yj.dot(h2yj)));
        if(m.Rdiag(j) < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(m.Rdiag(j)), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
            h2yj.setZero();
        }
    }
    // Compress or randomize
    handle_masked_columns();
    if(Y.cols() == 0) return;

    // DGKS passes
    Eigen::Index maxReps = 3;
    for(int rep = 0; rep < maxReps; ++rep) {
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            if(m.mask(j) == 0) continue;

            auto yj   = Y.col(j);
            auto h2yj = H2Y.col(j);

            // 1) Clean against i<j
            for(Eigen::Index i = 0; i < j; ++i) {
                if(m.mask(i) == 0) continue;
                auto yi   = Y.col(i);
                auto h2yi = H2Y.col(i);

                // projection hij = yiᴴ H2 yj
                // Scalar proj = yi.dot(h2yj);
                Scalar proj = dot_fp128(yi, h2yj);

                // subtract
                yj.noalias() -= yi * proj;
                h2yj.noalias() -= h2yi * proj;
            }

            // 2) Norm & mask‐check
            RealScalar norm = std::sqrt(std::real(yj.dot(h2yj)));
            if(norm <= m.maskTol) {
                m.mask(j) = 0;
                yj.setZero();
                h2yj.setZero();
                continue;
            }

            // 3) Normalize
            yj /= norm;
            h2yj /= norm;
        }
        // Compress or randomize
        handle_masked_columns();

        // Refresh gram matrix
        H2Y.noalias() = MultH2(Y);
        m.Gram        = Y.adjoint() * H2Y;
        m.Gram        = RealScalar{0.5f} * (m.Gram + m.Gram.adjoint()); // The Gram matrix must be hermitian (and PSD)
        m.Rdiag       = m.Gram.diagonal().cwiseAbs().cwiseSqrt();       // Equivalent to diag(R), with R from QR
        m.orthError   = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
        if constexpr(settings::debug_solver)
            eiglog->trace("block_h2_orthonormalize_dgks: dgks rep {}: orthError {:.3e} | H2Y.norm() = {:.3e} | Y.cols() {} | H2 norm {:.3e}", rep,
                          fp(m.orthError), fp(H2Y.norm()), Y.cols(), fp(H2.get_op_norm()));

        if(m.orthError < normTol) break;
    }

    H1Y = MultH1(Y);
    assert_h2_orthonormal(Y, H2Y, m);
}

template<typename LScalar>
struct LLTOrthoStepMeta {
    using RealLScalar = decltype(std::real(std::declval<LScalar>()));
    using MatrixLType = Eigen::Matrix<LScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorIdxT  = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    using VectorLReal = Eigen::Matrix<RealLScalar, Eigen::Dynamic, 1>;

    MatrixLType Y;
    MatrixLType H2Y;
    VectorIdxT  mask;
    RealLScalar maskTol;
    MaskPolicy  maskPolicy;
    MatrixLType G;
    VectorLReal Rdiag;
    RealLScalar orthError;
    template<typename Scalar>
    LLTOrthoStepMeta(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Y_Scalar,   //
                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H2Y_Scalar, //
                     Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>        &mask_,      //
                     decltype(std::real(std::declval<Scalar>())) maskTol_Scalar, MaskPolicy maskPolicy_)
        : Y(Y_Scalar.template cast<LScalar>()), H2Y(H2Y_Scalar.template cast<LScalar>()), mask(mask_), maskTol(static_cast<RealLScalar>(maskTol_Scalar)),
          maskPolicy(maskPolicy_) {}
};

template<typename LScalar>
void do_llt_orthonormalization_step(LLTOrthoStepMeta<LScalar> &m, [[maybe_unused]] std::shared_ptr<spdlog::logger> eiglog) {
    using RealLScalar = typename LLTOrthoStepMeta<LScalar>::RealLScalar;
    using MatrixLType = typename LLTOrthoStepMeta<LScalar>::MatrixLType;
    using VectorIdxT  = typename LLTOrthoStepMeta<LScalar>::VectorIdxT;
    // using VectorLReal = typename LLTOrthoStepMeta<LScalar>::VectorLReal;

    auto &G         = m.G;
    auto &Y         = m.Y;
    auto &H2Y       = m.H2Y;
    auto &mask      = m.mask;
    auto &maskTol   = m.maskTol;
    auto &orthError = m.orthError;

    auto colMask2ColIndex = [](const VectorIdxT &mask) -> std::vector<Eigen::Index> {
        std::vector<Eigen::Index> index;
        for(Eigen::Index j = 0; j < mask.size(); ++j) {
            if(mask(j) == 1) { index.push_back(j); }
        }
        return index;
    };

    G         = Y.adjoint() * H2Y;
    G         = (G + G.adjoint()) / RealLScalar{2};  // The Gram matrix must be hermitian (and PSD)
    m.Rdiag   = G.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
    orthError = (G.cwiseAbs() - MatrixLType::Identity(G.rows(), G.cols())).norm();

    Eigen::LLT<MatrixLType> llt(G);
    if(llt.info() != Eigen::Success) {
        RealLScalar sigma = G.cwiseAbs().rowwise().sum().maxCoeff();
        RealLScalar delta = Y.cols() * std::numeric_limits<RealLScalar>::epsilon() * sigma;

        for(Eigen::Index attempt = 1; attempt < 3; ++attempt) {
            // eiglog->warn("LLT failed, adding ridge δ = {:.3e}", fp(delta));
            G.diagonal().array() += delta;
            llt.compute(G);
            if(llt.info() == Eigen::Success) break;
            delta *= RealLScalar{10};
        }
        if(llt.info() != Eigen::Success)
            throw except::runtime_error("llt failed even after ridge escalation to δ = {:.3e}. G: \n{}", fp(delta), linalg::matrix::to_string(G, 8));
    }

    MatrixLType Rinv = llt.matrixU().solve(MatrixLType::Identity(Y.cols(), Y.cols()));
    Y *= Rinv;
    H2Y *= Rinv;

    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
        auto yj    = Y.col(j);
        auto h2yj  = H2Y.col(j);
        auto norm  = std::sqrt(std::abs(yj.dot(h2yj)));
        m.Rdiag(j) = norm;
        if(norm < maskTol) {
            mask(j) = 0;
            yj.setZero();
            h2yj.setZero();
            continue;
        }
        yj /= norm;
        h2yj /= norm;
    }

    // Refresh
    auto idx  = colMask2ColIndex(mask);
    auto Ym   = Y(Eigen::all, idx);
    auto H2Ym = H2Y(Eigen::all, idx);
    G         = Ym.adjoint() * H2Ym;
    G         = (G + G.adjoint()) / RealLScalar{2};  // The Gram matrix must be hermitian (and PSD)
    m.Rdiag   = G.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
    orthError = (G.cwiseAbs() - MatrixLType::Identity(G.rows(), G.cols())).norm();
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthonormalize_llt(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG and use_h2_inner_product);

    // Column-wise orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I

    m.mask        = VectorIdxT::Ones(Y.cols());
    m.proj_sum_h2 = VectorReal::Zero(Y.cols());
    m.scale_log   = VectorReal::Zero(Y.cols());

    H2Y       = MultH2(Y);
    m.maskTol = std::max(m.maskTol, normTol * std::sqrt(status.op_norm_estimate));

    auto handle_masked_columns = [&]() {
        if(m.mask.sum() != Y.cols()) {
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS: {
                    eiglog->debug("block_h2_orthonormalize_llt: Compressing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    compress_cols(Y, m.mask);
                    compress_cols(H2Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                }
                case MaskPolicy::RANDOMIZE: {
                    eiglog->debug("block_h2_orthonormalize_llt: Randomizing Y. Mask: {} | norms {::.3e} | maskTol {:.3e}", m.mask, fv(m.Rdiag), fp(m.maskTol));
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) {
                            Y.col(j).setRandom();
                            H2Y.col(j) = MultH2(Y.col(j));
                        }
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
        auto h2yj  = H2Y.col(j);
        m.Rdiag(j) = std::sqrt(std::abs(yj.dot(h2yj)));
        if(m.Rdiag(j) < m.maskTol) {
            eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, fp(m.Rdiag(j)), fp(m.maskTol));
            m.mask(j) = 0;
            yj.setZero();
            h2yj.setZero();
        }
    }
    // Compress or randomize
    handle_masked_columns();
    if(Y.cols() == 0) return;
    // Refresh gram matrix
    m.Gram      = Y.adjoint() * H2Y;
    m.Gram      = RealScalar{0.5f} * (m.Gram + m.Gram.adjoint()); // The Gram matrix must be hermitian (and PSD)
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();       // Equivalent to diag(R), with R from QR
    m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();

    if constexpr(settings::debug_solver)
        eiglog->trace("block_h2_orthonormalize_llt: before llt: orthError {:.5e} | cols {} | H2Y.norm() =  {:.3e}", fp(m.orthError), Y.cols(), fp(H2Y.norm()));

    // using Scalar64  = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<double>, double>;
    // using Scalar80  = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;
    // using Scalar128 = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;

    // eiglog->debug("gram - I : \n{}", linalg::matrix::to_string(m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols()), 16));

    Eigen::Index maxReps = 3;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        auto lm = LLTOrthoStepMeta<Scalar>(Y, H2Y, m.mask, m.maskTol, m.maskPolicy);
        // auto l80  = LLTOrthoStepMeta<Scalar80>(Y, H2Y, m.mask, m.maskTol, m.maskPolicy);
        // auto l128 = LLTOrthoStepMeta<Scalar128>(Y, H2Y, m.mask, m.maskTol, m.maskPolicy);
        do_llt_orthonormalization_step(lm, eiglog);
        // do_llt_orthonormalization_step(l80, eiglog);
        // do_llt_orthonormalization_step(l128, eiglog);

        // Extract the solution
        Y           = lm.Y.template cast<Scalar>();
        H2Y         = lm.H2Y.template cast<Scalar>();
        m.mask      = lm.mask;
        m.Gram      = lm.G;
        m.Rdiag     = lm.Rdiag; // Equivalent to diag(R), with R from QR
        m.orthError = lm.orthError;

        handle_masked_columns(); // Compress or randomize

        // Refresh gram matrix

        // eiglog->debug("block_h2_orthonormalize_llt: llt rep {}: orthError l64: {:.5e} | l80: {:.5e} | l128: {:.5e} | final {:.5e}", rep, fp(l64.orthError),
        // fp(l80.orthError), fp(l128.orthError), fp(m.orthError));

        // eiglog->debug("block_h2_orthonormalize_llt: llt rep {}: orthError {:.5e} | tol {:.5e}", rep, fp(m.orthError), fp(normTol));
        // eiglog->debug("gram - I: \n{}", linalg::matrix::to_string(m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols()), 16));
        if constexpr(settings::debug_solver)
            eiglog->trace("block_h2_orthonormalize_llt: llt rep {}: orthError {:.5e} | tol {:.5e}", rep, fp(m.orthError), fp(normTol));
        if(rep >= 1 and m.orthError < normTol) break;
    }
    // eiglog->debug("block_h2_orthonormalize_llt: llt rep {}: orthError {:.5e} | tol {:.5e}", rep, fp(m.orthError), fp(normTol));

    H1Y = MultH1(Y);
    assert_h2_orthonormal(Y, H2Y, m);
}

template<typename LScalar>
struct EigOrthoStepMeta {
    using RealLScalar = decltype(std::real(std::declval<LScalar>()));
    using MatrixLType = Eigen::Matrix<LScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorIdxT  = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    using VectorLReal = Eigen::Matrix<RealLScalar, Eigen::Dynamic, 1>;

    MatrixLType Y;
    MatrixLType H2Y;
    MatrixLType G;
    VectorLReal Rdiag;
    RealLScalar orthError;
    template<typename Scalar>
    EigOrthoStepMeta(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Y_Scalar, //
                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H2Y_Scalar)
        : Y(Y_Scalar.template cast<LScalar>()), H2Y(H2Y_Scalar.template cast<LScalar>()) {}
};

template<typename RealScalar, typename LScalar>
void do_eig_orthonormalization_step(EigOrthoStepMeta<LScalar> &m, [[maybe_unused]] std::shared_ptr<spdlog::logger> eiglog) {
    using RealLScalar = typename EigOrthoStepMeta<LScalar>::RealLScalar;
    using MatrixLType = typename EigOrthoStepMeta<LScalar>::MatrixLType;
    using VectorLReal = typename EigOrthoStepMeta<LScalar>::VectorLReal;

    auto &G         = m.G;
    auto &R         = m.Rdiag;
    auto &Y         = m.Y;
    auto &H2Y       = m.H2Y;
    auto &orthError = m.orthError;

    static constexpr auto half = RealLScalar{1} / RealLScalar{2};
    static constexpr auto epsL = static_cast<RealLScalar>(std::numeric_limits<RealScalar>::epsilon());

    G                       = Y.adjoint() * H2Y;
    G                       = (G + G.adjoint()) * half; // The Gram matrix must be hermitian (and PSD)
    orthError               = (G - MatrixLType::Identity(G.rows(), G.cols())).norm();
    VectorLReal Gdiag       = G.real().diagonal();
    VectorLReal scaleErrors = Gdiag - VectorLReal::Ones(Gdiag.size());
    if constexpr(settings::debug_solver) eiglog->trace("do_eig_orthonormalization_step: Scale errors diag(G)-I: {::.5e}", fv(scaleErrors));

    if(scaleErrors.cwiseAbs().maxCoeff() > RealScalar{10000 * epsL}) {
        // Step 0: Pre-scale (turns the gram matrix into a correlation matrix)
        //         We drop tiny-norm columns so that we don't amplify errors later.
        VectorLReal               absGdiag = Gdiag.cwiseAbs();
        RealLScalar               tol_drop = 100 * epsL * G.rows() * std::max(RealLScalar{1}, absGdiag.maxCoeff());
        std::vector<Eigen::Index> keep;
        for(Eigen::Index j = 0; j < Gdiag.size(); ++j) {
            auto d = std::sqrt(absGdiag(j));

            if(d > tol_drop) {
                Y.col(j) /= d;
                H2Y.col(j) /= d;
                keep.push_back(j);
            } else {
                RealLScalar ynorm   = Y.col(j).norm();
                RealLScalar h2ynorm = H2Y.col(j).norm();
                // if constexpr(settings::debug_solver) eiglog->trace("do_eig_orthonormalization_step: dropping |G({},{})| = {:.5e}", j, j, absGdiag(j));
                eiglog->trace("do_eig_orthonormalization_step: dropping |G({},{})| = {:.5e}  |Y(j)| = {:.5e}  |H2Y(j)| = {:.5e}", j, j, fp(absGdiag(j)),
                              fp(ynorm), fp(h2ynorm));
            }
        }

        Y   = Y(Eigen::all, keep).eval();
        H2Y = H2Y(Eigen::all, keep).eval();

        // Refresh the Gram matrix
        G = Y.adjoint() * H2Y;
        G = (G + G.adjoint()) * half; // Symmetrize
    }

    if(Y.cols() == 0) {
        // Nothing left to orthonormalize
        if constexpr(settings::debug_solver) eiglog->trace("do_eig_orthonormalization_step: no columns left");
        G         = MatrixLType();
        R         = VectorLReal();
        orthError = RealLScalar{0};
        return;
    }

    if(G.size() == 1) {
        // We normalize a single column directly
        auto absG = std::abs(G(0, 0));
        auto d    = std::sqrt(absG);
        Y /= d;
        H2Y /= d;
        G         = Y.adjoint() * H2Y;
        R         = G.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
        orthError = (G - MatrixLType::Identity(G.rows(), G.cols())).norm();
        return;
    }

    // Step 1: Drop eigenvalues of G that are too small (these correspond to nearly collinear vectors in H2-norm)

    auto        esG = Eigen::SelfAdjointEigenSolver<MatrixLType>(G);
    VectorLReal lG  = esG.eigenvalues();
    VectorLReal lGI = lG - VectorLReal::Ones(lG.size());
    if constexpr(settings::debug_solver) eiglog->trace("λ(G) - I = {::.5e}", fv(lGI));

    RealLScalar               tol_drop = 100 * epsL * G.rows() * std::max(RealLScalar{1}, esG.eigenvalues().maxCoeff());
    std::vector<Eigen::Index> keep;

    for(Eigen::Index j = 0; j < esG.eigenvalues().size(); ++j) {
        if(std::abs(esG.eigenvalues()(j)) > tol_drop) {
            keep.push_back(j);
        } else {
            // if constexpr(settings::debug_solver) eiglog->trace("dropping col {} of {}: evs: {::.5e}", j, G.rows(), fv(esG.eigenvalues()));
            eiglog->trace("dropping col {} of {}: evs: {::.5e}", j, G.rows(), fv(esG.eigenvalues()));
        }
    }
    // Step 2: Define the normalizing matrix W using only the safe components of G
    VectorLReal D = esG.eigenvalues()(keep);
    MatrixLType U = esG.eigenvectors()(Eigen::all, keep);
    MatrixLType W = U * D.cwiseInverse().cwiseSqrt().asDiagonal();

    // Compress and normalize in one shot
    Y *= W;
    H2Y *= W;

    // Refresh
    G         = Y.adjoint() * H2Y;
    G         = (G + G.adjoint()) * half;            // The Gram matrix must be hermitian (and PSD)
    R         = G.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
    orthError = (G - MatrixLType::Identity(G.rows(), G.cols())).norm();
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthonormalize_eig(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m) {
    if(Y.cols() == 0) return;
    if(m.mask.size() > 0 and m.mask.sum() == 0) return;

    assert(algo == OptAlgo::GDMRG and use_h2_inner_product);
    assert(m.maskPolicy == MaskPolicy::COMPRESS); // This operation does not preserve column order

    // Orthonormalization with respect to the H2 inner product, i.e. Y.adjoint()*H2*Y = I
    balance_columns_sweep(Y, H2Y, /*num_sweeps=*/2, /*max_pairs_per_sweep=*/-1, /*target_ratio=*/2.0);
    H2Y = MultH2(Y);

    using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<double>, double>;
    // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<fp128>, fp128>;
    // using ScalarL = std::conditional_t<tenx::sfinae::is_std_complex_v<Scalar>, std::complex<long double>, long double>;

    Eigen::Index maxReps = 2;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        auto eosm = EigOrthoStepMeta<ScalarL>(Y, H2Y);
        do_eig_orthonormalization_step<RealScalar, ScalarL>(eosm, eiglog);

        if(eosm.Y.cols() == 0) {
            if constexpr(settings::debug_solver) eiglog->trace("block_h2_orthonormalize_eig: 0/{} cols remain in Y", m.Gram.cols());
            Y           = MatrixType();
            H1Y         = MatrixType();
            H2Y         = MatrixType();
            m.Gram      = MatrixType();
            m.Rdiag     = VectorReal();
            m.orthError = 0;
            return;
        }

        // Extract the solution
        Y   = eosm.Y.template cast<Scalar>();
        H2Y = eosm.H2Y.template cast<Scalar>();
        if(rep == 0) {
            balance_columns_sweep(Y, H2Y, /*num_sweeps=*/2, /*max_pairs_per_sweep=*/-1, /*target_ratio=*/2.0);
            H2Y = MultH2(Y);
        }
        m.Gram      = eosm.G.template cast<Scalar>();
        m.Rdiag     = eosm.Rdiag.template cast<RealScalar>(); // Equivalent to diag(R), with R from QR
        m.orthError = static_cast<RealScalar>(eosm.orthError);

        if constexpr(settings::debug_solver)
            eiglog->trace("block_h2_orthonormalize_eig: eig rep {}: orthError {:.5e} | tol {:.5e}", rep, fp(m.orthError), fp(normTol));

        if(m.orthError >= RealScalar{1e-1f})
            throw except::runtime_error("block_h2_orthonormalize_eig: very large error on rep {}: orthError {:.5e} | cols {} | H2Y.norm() =  {:.3e} \n G: \n{}",
                                        rep, fp(m.orthError), Y.cols(), fp(H2Y.norm()), linalg::matrix::to_string(m.Gram, 8));

        if(rep >= 1 and m.orthError < normTol) break;
    }
    balance_columns_sweep(Y, H2Y, /*num_sweeps=*/2, /*max_pairs_per_sweep=*/-1, /*target_ratio=*/2.0);

    // Refresh
    m.Gram      = Y.adjoint() * H2Y;
    m.Gram      = (m.Gram + m.Gram.adjoint()) * half;       // The Gram matrix must be hermitian (and PSD)
    m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt(); // Equivalent to diag(R), with R from QR
    m.orthError = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
    // eiglog->info("ortherror in final: {:.5e}", fp(m.orthError));

    H1Y = MultH1(Y);
    assert_h2_orthonormal(Y, H2Y, m);
}

template<typename Scalar>
void solver_base<Scalar>::block_h2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y,
                                                 MatrixType &H2Y, OrthMeta &m) {
    if(X.cols() == 0 || Y.cols() == 0) return;
    if(m.mask.size() > 0 && m.mask.sum() == 0) return;
    assert(algo == OptAlgo::GDMRG and use_h2_inner_product && "block_h2_orthogonalize is for H2 inner product");

    assert_allFinite(X);
    assert_allFinite(H1X);
    assert_allFinite(H2X);
    assert_allFinite(Y);
    // assert_allFinite(H1Y);
    // assert_allFinite(H2Y);
    assert_h2_orthonormal(X, H2X);

    m.orthTol = std::max(m.orthTol, eps * std::sqrt(status.op_norm_estimate));

    H2Y.noalias() = MultH2(Y);
    m.Gram        = X.adjoint() * H2Y;
    m.Rdiag       = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
    m.orthError   = m.Gram.size() > 0 ? m.Gram.norm() : 0;

    MatrixType xGram      = X.adjoint() * H2X;
    RealScalar xOrthError = (xGram - MatrixType::Identity(xGram.cols(), xGram.rows())).norm();

    // DGKS clean Y against X
    Eigen::Index maxReps = 2;
    Eigen::Index rep     = 0;
    for(rep = 0; rep < maxReps; ++rep) {
        if(m.mask.size() != Y.cols()) m.mask = VectorIdxT::Ones(Y.cols());
        if(m.proj_sum_h1.size() != Y.cols()) m.proj_sum_h1 = VectorReal::Zero(Y.cols());
        if(m.proj_sum_h2.size() != Y.cols()) m.proj_sum_h2 = VectorReal::Zero(Y.cols());
        if(m.scale_log.size() != Y.cols()) m.scale_log = VectorReal::Zero(Y.cols());

        Y.noalias() -= X * m.Gram;
        H2Y.noalias() -= H2X * m.Gram;

        // orthonormalize Y and refresh H1Y and H2Y
        block_h2_orthonormalize_eig(Y, H1Y, H2Y, m);
        // H2Y.noalias() = MultH2(Y);
        if(Y.cols() > 0) {
            m.Gram      = X.adjoint() * H2Y;
            m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
            m.orthError = m.Gram.norm();
        } else {
            if constexpr(settings::debug_solver) eiglog->trace("block_h2_orthogonalize: 0/{} columns remain in Y", m.Gram.rows());
            m.Gram      = MatrixType();
            m.Rdiag     = VectorReal();
            m.orthError = 0;
            break;
        }

        // if constexpr(settings::debug_solver)
        if constexpr(settings::debug_solver)
            eiglog->trace("block_h2_orthogonalize           rep {}: orthError {:.5e} | tol {:.3e} | xOrthError {:.5e}", rep, fp(m.orthError), fp(m.orthTol),
                          fp(xOrthError));
        if(rep >= 1) {
            // DGKS drop test – skip next rep if it already cleaned well
            bool orth_converged = m.orthError < m.orthTol;
            if(orth_converged) break;
        }
    }
    assert_h2_orthogonal(X, H2Y, m);
}

//
// template<typename Scalar>
// void solver_base<Scalar>::block_h2_orthogonalize_old(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y,
//                                                  MatrixType &H2Y, OrthMeta &m) {
//     if(X.cols() == 0 || Y.cols() == 0) return;
//     if(m.mask.size() > 0 && m.mask.sum() == 0) return;
//     assert(algo == OptAlgo::GDMRG and use_h2_inner_product && "block_h2_orthogonalize is for H2 inner product");
//
//     assert_allFinite(X);
//     assert_allFinite(H1X);
//     assert_allFinite(H2X);
//     assert_allFinite(Y);
//     assert_allFinite(H1Y);
//     assert_allFinite(H2Y);
//     assert_h2_orthonormal(X, H2X);
//
//     m.orthTol = std::max(m.orthTol, eps * std::sqrt(status.op_norm_estimate));
//
//     H2Y.noalias() = MultH2(Y);
//     m.Gram        = X.adjoint() * H2Y;
//     m.Rdiag       = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
//     m.orthError   = m.Gram.size() > 0 ? m.Gram.norm() : 0;
//
//     MatrixType xGram = X.adjoint() * H2X;
//     // RealScalar xOrthError = (xGram - MatrixType::Identity(xGram.cols(), xGram.rows())).norm();
//
//     // DGKS clean Y against X
//     Eigen::Index maxReps = 3;
//     Eigen::Index rep     = 0;
//     for(rep = 0; rep < maxReps; ++rep) {
//         if(m.mask.size() != Y.cols()) m.mask = VectorIdxT::Ones(Y.cols());
//         if(m.proj_sum_h1.size() != Y.cols()) m.proj_sum_h1 = VectorReal::Zero(Y.cols());
//         if(m.proj_sum_h2.size() != Y.cols()) m.proj_sum_h2 = VectorReal::Zero(Y.cols());
//         if(m.scale_log.size() != Y.cols()) m.scale_log = VectorReal::Zero(Y.cols());
//
//         for(Eigen::Index j = 0; j < Y.cols(); ++j) {
//             auto       yj   = Y.col(j);
//             auto       h2yj = H2Y.col(j);
//             MatrixType gj   = m.Gram.col(j);
//             // RealScalar yj_norm = yj.norm();
//             // RealScalar gj_norm           = gj.norm();
//             // RealScalar refresh_threshold = RealScalar{0.1f} * yj_norm; // compares current sizes
//             yj.noalias() -= X * gj; // Remove projection
//             h2yj = MultH2(yj);
//             // if(gj_norm > refresh_threshold) {
//             //     eiglog->info("rep {} block_h2_orthogonalize: recomputing h2y block {} | gj_norm {} > refresh threshold {}", rep, j, fp(gj_norm),
//             //                  fp(refresh_threshold));
//             //     h2yj.noalias()   = MultH2(yj); // Compute from scratch
//             //     m.proj_sum_h2(j) = 0;
//             //     m.scale_log.setZero();
//             // } else {
//             //     MatrixType H2X_pj = H2X * gj;
//             //     h2yj.noalias() -= H2X_pj;          // Remove projection
//             //     m.proj_sum_h2(j) += H2X_pj.norm(); // Accumulate norms
//             // }
//         }
//
//         // orthonormalize Y and refresh H1Y and H2Y
//
//         block_h2_orthonormalize_eig(Y, H1Y, H2Y, m);
//
//         m.Gram      = X.adjoint() * H2Y;
//         m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
//         m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;
//         // if constexpr(settings::debug_solver)
//         eiglog->debug("block_h2_orthogonalize           rep {}: orthError {:.5e} | tol {:.3e}", rep, fp(m.orthError), fp(m.orthTol));
//         if(rep >= 1) {
//             // DGKS drop test – skip next rep if it already cleaned well
//             bool orth_converged = m.orthError < m.orthTol;
//             if(orth_converged) break;
//         }
//
//
//     }
//
//     assert_h2_orthogonal(X, H2Y, m);
// }

template<typename Scalar>
void solver_base<Scalar>::pad_and_orthonormalize(MatrixType &Y, MatrixType &HY, Eigen::Index nBlocks, OrthMeta &m) {
    Eigen::Index reps = 0;
    while(reps++ == 0 or Y.cols() / b < nBlocks) {
        if(Y.cols() < nBlocks * b) {
            // Pad with random vectors
            auto vc = Y.cols();
            Y.conservativeResize(Y.rows(), nBlocks * b);
            Y.rightCols(nBlocks * b - vc).setRandom();
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
            Y.rightCols(nBlocks * b - vc).setRandom();
            eiglog->info("Randomizing {} blocks", nBlocks * b - vc);
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
            V.rightCols(b - vc).setRandom();
        }
        // Orthonormalize V.
        // Discard columns if there are more than b (this is not expected, but also not an error)
        cpqr.compute(V);
        auto rank = std::min(cpqr.rank(), b);
        V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank) * cpqr.colsPermutation().transpose();
        if(V.cols() == b) break;
    }

    auto block_orthonormalize = [&] {
        auto m = OrthMeta();
        if(algo == OptAlgo::GDMRG) {
            if(use_h2_inner_product) {
                block_h2_orthonormalize_eig(V, H1V, H2V, m);
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
            auto Z        = T_evecs(Eigen::all, status.optIdx);
            auto Y        = T_evals(status.optIdx);
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
                T_evals.size() <= 1 ? RealScalar{1} : (T_evals.bottomRows(T_evals.size() - 1) - T_evals.topRows(T_evals.size() - 1)).cwiseAbs().minCoeff();
            auto select1            = get_ritz_indices(ritz, 0, 1, T_evals);
            auto H1_max_abs         = std::max(std::abs(status.T1_min_eval), std::abs(status.T1_max_eval));
            auto H2_max_abs         = std::max(std::abs(status.T2_min_eval), std::abs(status.T2_max_eval));
            status.condition        = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;
            status.op_norm_estimate = get_op_norm_estimate();
            // We may need to orthonormalize V in GDMRG
            block_orthonormalize();

            S             = H1V - H2V * Y.asDiagonal();
            status.rNorms = S.colwise().norm();
            status.eigVal = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b

        } else {
            block_orthonormalize();
            Q  = V;
            HQ = MultH(V);
            T  = Q.adjoint() * HQ;
            T  = RealScalar{0.5f} * (T.adjoint() + T); // Symmetrize
            Eigen::SelfAdjointEigenSolver<MatrixType> es(T);
            T_evecs            = es.eigenvectors();
            T_evals            = es.eigenvalues();
            status.optIdx      = get_ritz_indices(ritz, 0, b, T_evals);
            auto Z             = T_evecs(Eigen::all, status.optIdx);
            auto Y             = T_evals(status.optIdx);
            V                  = Q * Z; // Now V has b columns mixed according to the selected columns in T_evecs
            HV                 = HQ * Z;
            S                  = HV - V * Y.asDiagonal();
            status.rNorms      = S.colwise().norm();
            status.eigVal      = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            status.T1_evals    = es.eigenvalues();
            status.T2_evals    = es.eigenvalues();
            status.T1_min_eval = T_evals.minCoeff();
            status.T2_min_eval = T_evals.minCoeff();
            status.T1_max_eval = T_evals.maxCoeff();
            status.T2_max_eval = T_evals.maxCoeff();
            status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
            status.condition        = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
            status.op_norm_estimate = get_op_norm_estimate();
        }
    }

    assert(V.cols() == b);
    assert_allFinite(V);
}

template<typename Scalar>
void solver_base<Scalar>::diagonalizeT() {
    if(algo == OptAlgo::GDMRG) return diagonalizeT1T2();
    if(status.stopReason != StopReason::none) return;
    if(Q.cols() == 0) return;
    if(HQ.cols() == 0) return;
    assert(Q.cols() == HQ.cols());

    MatrixType T = Q.adjoint() * HQ;
    T            = RealScalar{0.5f} * (T + T.adjoint()).eval(); // Symmetrize
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
        return x.bottomRows(x.size() - 1) - x.topRows(x.size() - 1);
    };
    auto select2 = get_ritz_indices(ritz, 0, nev + 1, T_evals);
    status.gap   = diff(T_evals(select2)).cwiseAbs().minCoeff();

    status.commit_evals(T_evals.minCoeff(), T_evals.maxCoeff());
    status.condition        = T_evals.cwiseAbs().maxCoeff() / T_evals.cwiseAbs().minCoeff();
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
    status.rNorms              = {};
    static constexpr auto half = RealScalar{1} / RealScalar{2};

    T1 = Q.adjoint() * H1Q;
    T2 = Q.adjoint() * H2Q;

    // Symmetrize
    T1 = (T1 + T1.adjoint()).eval() * half;
    T2 = (T2 + T2.adjoint()).eval() * half;
    assert(T1.rows() == T2.rows());
    assert(T1.cols() == T2.cols());

    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T1, T2, Eigen::Ax_lBx);
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

    // Calculate the gap
    auto diff = [](const VectorReal &x) -> VectorReal {
        if(x.size() <= 1) return VectorReal::Ones(1);
        return x.bottomRows(x.size() - 1) - x.topRows(x.size() - 1);
    };
    auto select = get_ritz_indices(ritz, 0, std::max(b, nev + 1), T_evals);
    status.gap  = diff(T_evals(select)).cwiseAbs().minCoeff();

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
        auto       H1_max_abs = std::max(std::abs(status.T1_min_eval), std::abs(status.T1_max_eval));
        auto       H2_max_abs = std::max(std::abs(status.T2_min_eval), std::abs(status.T2_max_eval));
        status.condition      = (H1_max_abs + T_evals(select1).cwiseAbs().coeff(0) * H2_max_abs) / min_sep;

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

    // Transform the basis
    V  = Q * Z; // Regular Rayleigh-Ritz
    HV = HQ * Z;

    S      = HV - V * Y.asDiagonal(); // Residual vector
    rNorms = S.colwise().norm();      // Residual norm
}

template<typename Scalar>
void solver_base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    // Get indices of the top b (the block size) eigenvalues as a std::vector<Eigen::Index>
    auto Z = T_evecs(Eigen::all, optIdx); // Selected subspace eigenvectors
    auto Y = T_evals(optIdx);             // Selected subspace eigenvalues

    // Transform the basis
    V.noalias()   = Q * Z; // Regular Rayleigh-Ritz
    H1V.noalias() = H1Q * Z;
    H2V.noalias() = H2Q * Z;

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

    // auto get_s_fma = [](Eigen::Ref<VectorType> Av, Eigen::Ref<VectorType> Bv, RealScalar lambda) -> VectorType {
    //     // Av, Bv : n-vectors; lambda : Scalar
    //     const RealScalar g  = std::hypot(Av.norm(), std::abs(lambda) * Bv.norm());
    //     const RealScalar ig = g > RealScalar(0) ? RealScalar(1) / g : RealScalar(1);
    //     VectorType       s(Av.size());
    //     for(Eigen::Index j = 0; j < s.size(); ++j) {
    //         if constexpr(std::is_arithmetic_v<Scalar>) {
    //             s[j] = std::fma(-lambda, Bv[j] * ig, Av[j] * ig) * g; // 1 rounding per entry
    //
    //         } else {
    //             RealScalar Avr = std::real(Av[j]);
    //             RealScalar Avi = std::imag(Av[j]);
    //
    //             RealScalar Bvr = std::real(Bv[j]);
    //             RealScalar Bvi = std::imag(Bv[j]);
    //
    //             RealScalar sr = std::fma(-lambda, Bvr * ig, Avr * ig) * g; // 1 rounding per entry
    //             RealScalar si = std::fma(-lambda, Bvi * ig, Avi * ig) * g; // 1 rounding per entry
    //             s[j]          = {sr, si};
    //         }
    //     }
    //     return s;
    // };
    // if(algo == OptAlgo::GDMRG) {
    // auto get_S_fma = [&]() -> MatrixType {
    //     MatrixType S_fma(N, S.cols());
    //     auto       lambdas = T_evals(status.optIdx);
    //     for(Eigen::Index i = 0; i < S.cols(); ++i) S_fma.col(i) = get_s_fma(H1V.col(i), H2V.col(i), lambdas[i]);
    //     return S_fma;
    // };

    // VectorReal Qnorm      = Q.colwise().norm();
    // VectorReal H2Qnorm    = H2Q.colwise().norm();
    // VectorReal Vnorm      = V.colwise().norm();
    // VectorReal H2Vnorm    = H2V.colwise().norm();
    // MatrixType S_fma      = get_S_fma();
    // VectorReal rnorm_fma  = S_fma.colwise().norm();
    // VectorReal rnorm_diff = status.rNorms - rnorm_fma;
    // eiglog->info("|V| = {::.5e} |H2V| = {::.5e}  |S| = {::.5e}  |S|_fma = {::.5e} (diff = {::.5e})", fv(Vnorm), fv(H2Vnorm), fv(status.rNorms),
    //              fv(rnorm_fma), fv(rnorm_diff));
    // eiglog->info("|Q|   = {::.5e}", fv(Qnorm));
    // eiglog->info("|H2Q| = {::.5e}", fv(H2Qnorm));
    // }
}

template<typename Scalar>
solver_base<Scalar>::MatrixType solver_base<Scalar>::get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z,
                                                                                       const Eigen::Ref<const VectorReal> &Y, const MatrixType &H1Q,
                                                                                       const MatrixType &H2Q) {
    assert(algo == OptAlgo::GDMRG);
    // assert(static_cast<size_t>(V.cols()) == optIdx.size());
    assert(Z.cols() == Y.size());
    Eigen::JacobiSVD<MatrixType> svd;
    MatrixType                   Z_ref(Z.rows(), Z.cols());
    MatrixType                   T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols()); // cache H2*zj
    for(Eigen::Index j = 0; j < Y.size(); ++j) {
        const auto &theta = Y(j);
        MatrixType  M     = (H1Q - theta * H2Q);

        svd.compute(M, Eigen::ComputeThinV);

        Eigen::Index min_idx;
        svd.singularValues().minCoeff(&min_idx);

        if(svd.info() == Eigen::Success) {
            // Accept the solution
            auto zj   = Z_ref.col(j);
            auto t2zj = T2Z_ref.col(j);
            zj        = svd.matrixV().col(min_idx);

            //----------------------------------------------------------------
            // orthogonalize zj against previously accepted columns
            //----------------------------------------------------------------
            if(use_h2_inner_product) {
                t2zj = T2 * zj; // T2 is b×b, cheap
            } else {
                t2zj = zj;
            }
            for(Eigen::Index i = 0; i < j; ++i) {
                auto   zi   = Z_ref.col(i);
                auto   t2zi = T2Z_ref.col(i);
                Scalar proj = zi.dot(t2zj); // (z_i)† T2 zj
                zj.noalias() -= zi * proj;
                t2zj.noalias() -= t2zi * proj; // keep cache consistent
            }

            //-----------------------------------------------------------------------------------------------------
            // Normalize w.r.t. T2-norm  ‖z‖_{2} = sqrt(abs(zj.adjoint()*T2*zj)) (when using the H2 inner product)
            //----------------------------------------------------------------------------------------------------
            RealScalar norm = std::sqrt(std::abs(zj.dot(t2zj)));
            if(norm < normTol) {
                // Column numerically null → zero-out but keep slot
                zj.setZero();
                t2zj.setZero();
                continue;
            }
            zj /= norm;
            t2zj /= norm;

        } else {
            Z_ref.col(j)            = Z.col(j);
            RealScalar refinedRnorm = svd.singularValues()(min_idx);
            eiglog->warn("refinement failed on ritz vector {} | refined rnorm={:.5e} | info {} ", j, fp(refinedRnorm), static_cast<int>(svd.info()));
        }
    }
    return Z_ref;
}

template<typename Scalar>
std::pair<typename solver_base<Scalar>::MatrixType, typename solver_base<Scalar>::MatrixType>
    solver_base<Scalar>::get_h2_normalizer_for_the_projected_pencil(const MatrixType &T2) {
    static constexpr auto half = RealScalar{1} / RealScalar{2};
    MatrixType            T2h  = (T2 + T2.adjoint()) * half;
    auto                  es   = Eigen::SelfAdjointEigenSolver<MatrixType>(T2h, Eigen::ComputeEigenvectors);
    if(es.info() != Eigen::Success) { throw except::runtime_error("get_h2_normalizer_for_the_projected_pencil: eigensolver failed"); }
    auto U    = es.eigenvectors();
    auto D    = es.eigenvalues();
    auto Dmax = std::max<RealScalar>(1, D.maxCoeff());
    for(Eigen::Index k = 0; k < D.size(); ++k) { D(k) = std::max(D(k), eps * Dmax); }
    return {U * D.cwiseInverse().cwiseSqrt().asDiagonal() * U.adjoint(), U * D.cwiseSqrt().asDiagonal() * U.adjoint()};
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
        RealScalar tau = 10 * eps * std::max(RealScalar{1}, std::real(B.trace()) * half);
        B += I * tau;

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
    Eigen::JacobiSVD<MatrixType> svd;
    MatrixType                   Z_ref(Z.rows(), Z.cols());
    MatrixType                   T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols()); // cache H2*zj
    for(Eigen::Index j = 0; j < Y.size(); ++j) {
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
    return Z_ref;
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S,
                                             VectorReal &rNorms) {
    assert(algo == OptAlgo::GDMRG);
    VectorReal Y     = T_evals(optIdx);
    MatrixType Z_rr  = T_evecs(Eigen::all, optIdx);
    MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, H1Q, H2Q);
    MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, T1, T2); // Gives an optimal combination of Z_rr and Z_ref

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

    S      = H1V - H2V * Y.asDiagonal();
    rNorms = S.colwise().norm();
}

template<typename Scalar>
void solver_base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms) {
    Eigen::JacobiSVD<MatrixType> svd;
    auto                         Z     = T_evecs(Eigen::all, optIdx);
    auto                         Y     = T_evals(optIdx);
    auto                         Z_ref = get_refined_ritz_eigenvectors_std(Z, Y, Q, HQ);

    // Transform the basis with applied operators
    V  = Q * Z_ref;
    HV = HQ * Z_ref;

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
    status.num_matvecs_total += status.num_matvecs + status.num_matvecs_inner;
    status.num_precond_total += status.num_precond + status.num_precond_inner;
    status.time_matvecs_total += status.time_matvecs.get_time() + status.time_matvecs_inner.get_time();
    status.time_precond_total += status.time_precond.get_time() + status.time_precond_inner.get_time();

    // Eigenvalues are sorted in ascending order.
    status.oldVal  = status.eigVal.topRows(nev);
    status.eigVal  = T_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
    status.absDiff = (status.eigVal - status.oldVal).cwiseAbs();
    status.relDiff = status.absDiff.array() / (RealScalar{0.5} * (status.eigVal + status.oldVal).array());

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
    RealScalar     relGap = status.gap * status.op_norm_estimate;
    // status.rNorm_below_rnormTol = (rNorms.array() < rnormTol(status.eigVal).array()).all(); // Residual norm condition
    status.rNorm_below_rnormTol = (rNorms.array() < rNormTols().array()).all(); // Residual norm condition
    status.rNorm_below_gap      = rNorms.maxCoeff() < beta * relGap;            // Gap condition for the currently selected operator (H1, H2, or H1/H2)

    if(status.rNorm_below_rnormTol and status.rNorm_below_gap) {
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
    } else if(status.saturation_count_eigVal > 0 and status.saturation_count_rNorm >= status.saturation_count_max * 2) {
        // Probably eigVal is stuck in some kind of cycle.
        status.stopMessage.emplace_back(fmt::format("saturation_count_rNorm {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                    status.saturation_count_rNorm, status.saturation_count_max, status.iter + 1, status.num_matvecs_total,
                                                    status.time_elapsed.get_time()));
        status.stopReason |= StopReason::saturated_rNorms;
    }
}

template<typename Scalar>
void solver_base<Scalar>::printStatus() {
    int printFreq = 1;
    if(eiglog->level() >= spdlog::level::info) return;
    if(eiglog->level() == spdlog::level::trace) printFreq = 1;
    if(eiglog->level() == spdlog::level::debug) printFreq = 5;
    if(status.iter + 1 % printFreq != 0) return;

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
    auto        H1ir            = H1.get_iterativeLinearSolverConfig();
    auto        H2ir            = H2.get_iterativeLinearSolverConfig();
    auto        H1H2ir          = H1H2.get_iterativeLinearSolverConfig();
    std::string innerMsg        = status.num_matvecs_inner == 0 ? std::string()
                                                                : fmt::format("[inner: ({}) mv {:5} err {:.2e} tol {:.2e} t {:.1e}s] ",                  //
                                                                              rCorrMsg,                                                                  //
                                                                              status.num_matvecs_inner,                                                  //
                                                                              fp(std::max({H1ir.result.error, H2ir.result.error, H1H2ir.result.error})), //
                                                                              fp(std::max({H1ir.tolerance, H2ir.tolerance, H1H2ir.tolerance})),          //
                                                                              fp(H1ir.result.time + H2ir.result.time + H1H2ir.result.time));
    bool        log_low_maxiter = max_iters < 10;
    bool        log_jacobi_prec = preconditioner_type == eig::Preconditioner::JACOBI and status.iter % 100 == 0;
    bool        log_solve_prec  = preconditioner_type == eig::Preconditioner::SOLVE;

    MatrixType  Gram      = use_h2_inner_product ? Q.adjoint() * H2Q : Q.adjoint() * Q;
    RealScalar  orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
    std::string evMsg;
    if(algo == OptAlgo::GDMRG) {
        VectorReal VH1V = (V.adjoint() * H1V).real();
        VectorReal VH2V = (V.adjoint() * H2V).real();
        evMsg           = fmt::format(" {::.16f} / {::.16f}", fv(VH1V), fv(VH2V));
    }

    if(log_low_maxiter or log_jacobi_prec or log_solve_prec)
        eiglog->debug("it {:3} mv {:3} pc {:3} t {:.1e}s {}"
                      "eigVal {::.16f}{} "
                      "oErr {:.3e} rNorms {::.8e} rNormTol {::.3e} tol {:.2e} "
                      "({:9.2e}/mv) sat {}:{}/{} col {:2} b {} ritz {} "
                      "op norm {:.2e} cond {:.2e}{}",
                      status.iter + 1,                   //
                      status.num_matvecs,                //
                      status.num_precond,                //
                      status.time_elapsed.restart_lap(), //
                      innerMsg,                          //
                      fv(status.eigVal),                 //
                      evMsg,                             //
                      fp(orthError),                     //
                      // fv(VectorReal(status.rNorms.topRows(nev))), //
                      fv(VectorReal(status.rNorms)),            //
                      fv(rNormTols()),                          //
                      fp(tol),                                  //
                      fp(get_rNorms_log10_change_per_matvec()), //
                      status.saturation_count_eigVal,           //
                      status.saturation_count_rNorm,            //
                      status.saturation_count_max,              //
                      Q.cols(),                                 //
                      b,                                        //
                      enum2sv(ritz),                            //
                      fp(status.op_norm_estimate),              //
                      fp(status.condition),                     //
                      msg_rnorm_gap);

    // status.time_elapsed.restart_lap();
}

template<typename Scalar>
void solver_base<Scalar>::set_maxPrevBlocks(Eigen::Index pb) {
    b  = std::min(std::max(nev, b), N / 2);
    pb = std::min<Eigen::Index>(pb, N / b);
    if(pb != maxPrevBlocks) eiglog->trace("gdplusk: maxPrevBlocks = {}", pb);
    maxPrevBlocks = pb;
}