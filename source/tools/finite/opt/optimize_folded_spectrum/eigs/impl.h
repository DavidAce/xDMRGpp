#pragma once

#include "../../../opt_meta.h"
#include "../../../opt_mps.h"
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/matvec/matvec_zero.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
// #include "tools/finite/measure.h"
// #include "tools/finite/multisite.h"
#include "math/tenx.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include <h5pp/h5pp.h>
#include <primme/primme.h>
using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

namespace folded_spectrum {
    template<typename CalcType>
    void preconditioner_jacobi(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
        if(x == nullptr) return;
        if(y == nullptr) return;
        if(primme == nullptr) return;
        const auto H_ptr      = static_cast<MatVecMPOS<CalcType> *>(primme->matrix);
        H_ptr->preconditioner = eig::Preconditioner::JACOBI;
        H_ptr->MultPc(x, ldx, y, ldy, blockSize, primme, ierr);
    }

    template<typename CalcType>
    void convTestFun([[maybe_unused]] double *eval, [[maybe_unused]] void *evec, double *rNorm, int *isconv, struct primme_params *primme, int *ierr) {
        if(rNorm == nullptr) return;
        if(primme == nullptr) return;

        double problemNorm;
        if(not primme->massMatrixMatvec) {
            problemNorm = primme->aNorm > 0.0 ? primme->aNorm : primme->stats.estimateLargestSVal;
        } else {
            problemNorm = primme->aNorm > 0.0 && primme->invBNorm > 0.0 ? primme->aNorm * primme->invBNorm : primme->stats.estimateLargestSVal;
        }
        double problemTol = problemNorm * primme->eps;
        double diff_rnorm = 0;
        double diff_eval  = 0;
        if(primme->monitor != nullptr and *rNorm != 0) {
            auto &solver = *static_cast<eig::solver *>(primme->monitor);
            auto &config = solver.config;
            auto &result = solver.result;
            result.meta.recent_evals.push_back(*eval);
            result.meta.recent_rnorms.push_back(*rNorm);
            while(result.meta.recent_evals.size() > 500) result.meta.recent_evals.pop_front();
            while(result.meta.recent_rnorms.size() > 500) result.meta.recent_rnorms.pop_front();
            double diff_evals_sum  = -1.0;
            double diff_rnorms_sum = -1.0;
            if(result.meta.recent_evals.size() >= 250) {
                auto diff      = num::diff(result.meta.recent_evals);
                diff_evals_sum = num::sum(diff, 1);
                // tools::log->info("recent evals: {::.3e}", result.meta.recent_evals);
                // tools::log->info("diff   evals: {::.3e}", diff);
            }
            if(result.meta.recent_rnorms.size() >= 250) {
                auto diff       = num::diff(result.meta.recent_rnorms);
                diff_rnorms_sum = num::sum(diff, 1);
                // tools::log->info("recent rnorm: {::.3e}", result.meta.recent_rnorms);
                // tools::log->info("diff   rnorm: {::.3e}", diff);
            }

            bool evals_saturated   = diff_evals_sum > -primme->eps;
            bool rnorms_saturated  = diff_rnorms_sum > -primme->eps;
            result.meta.last_eval  = *eval;
            result.meta.last_rnorm = *rNorm;

            *isconv = ((*rNorm < problemTol) and (evals_saturated or rnorms_saturated)) or (*rNorm < primme->eps);
            if(*isconv == 1) {
                tools::log->info("eval {:38.32f} ({:+8.3e}) | rnorm {:38.32f} ({:+8.3e}) | problemNorm {:.3e}", *eval, diff_evals_sum, *rNorm, diff_rnorms_sum,
                                 problemNorm);
            }
        } else {
            *isconv = 0;
        }

        *ierr = 0;
    }

    template<typename Scalar>
    struct opt_mps_init_t {
        Eigen::Tensor<Scalar, 3> mps = {};
        long                     idx = 0;
    };
    template<typename Scalar>
    struct opt_bond_init_t {
        Eigen::Tensor<Scalar, 2> bond = {};
        long                     idx  = 0;
    };
    template<typename CalcType, typename Scalar>
    Eigen::Tensor<Scalar, 3> get_initial_guess(const opt_mps<Scalar> &initial_mps, const std::vector<opt_mps<Scalar>> &results) {
        if(results.empty()) {
            return initial_mps.template get_tensor_as<CalcType>();
        } else {
            // Return whichever of initial_mps or results that has the lowest variance
            auto it = std::min_element(results.begin(), results.end(), internal::comparator::variance);
            if(it == results.end()) return get_initial_guess<CalcType>(initial_mps, {});

            if(it->get_variance() < initial_mps.get_variance()) {
                tools::log->debug("Previous result is a good initial guess: {} | var {:8.2e}", it->get_name(), it->get_variance());
                return get_initial_guess<CalcType>(*it, {});
            } else
                return get_initial_guess<CalcType>(initial_mps, {});
        }
    }

    template<typename CalcType, typename Scalar>
    std::vector<opt_mps_init_t<CalcType>> get_initial_guess_mps(const opt_mps<Scalar> &initial_mps, const std::vector<opt_mps<Scalar>> &results, long nev) {
        std::vector<opt_mps_init_t<CalcType>> init;
        if(results.empty()) {
            init.push_back({initial_mps.template get_tensor_as<CalcType>(), 0});
        } else {
            for(long n = 0; n < nev; n++) {
                // Take the latest result with idx == n

                // Start by collecting the results with the correct index
                std::vector<std::reference_wrapper<const opt_mps<Scalar>>> results_idx_n;
                for(const auto &r : results) {
                    if(r.get_eigs_idx() == n) results_idx_n.emplace_back(r);
                }
                if(not results_idx_n.empty()) { init.push_back({results_idx_n.back().get().template get_tensor_as<CalcType>(), n}); }
            }
        }
        if(init.size() > safe_cast<size_t>(nev)) throw except::logic_error("Found too many initial guesses");
        return init;
    }

    template<typename CalcType, typename Scalar>
    RealScalar<CalcType> get_largest_eigenvalue_hamiltonian_squared(const TensorsFinite<Scalar> &tensors) {
        auto env2                = tensors.edges->template get_multisite_env_var_blk_as<CalcType>();
        auto hamiltonian_squared = MatVecMPO<CalcType>(env2.L, env2.R, tensors.template get_multisite_mpo_squared<CalcType>());
        tools::log->trace("Finding largest-magnitude eigenvalue");
        eig::solver solver; // Define a solver just to find the maximum eigenvalue
        solver.config.tol             = settings::precision::eigs_tol_min;
        solver.config.maxIter         = 200;
        solver.config.maxNev          = 1;
        solver.config.maxNcv          = 16;
        solver.config.compute_eigvecs = eig::Vecs::OFF;
        solver.config.sigma           = std::nullopt;
        solver.config.ritz            = eig::Ritz::LM;
        solver.setLogLevel(2);
        solver.eigs(hamiltonian_squared);
        return eig::view::get_eigvals<RealScalar<CalcType>>(solver.result).cwiseAbs().maxCoeff();
    }

    // template<typename Scalar>
    // double max_gradient(const Eigen::Tensor<Scalar, 3> &mps, const Eigen::Tensor<Scalar, 4> &mpo1, const Eigen::Tensor<Scalar, 3> &en1L,
    //                     const Eigen::Tensor<Scalar, 3> &en1R, const Eigen::Tensor<Scalar, 4> &mpo2, const Eigen::Tensor<Scalar, 3> &en2L,
    //                     const Eigen::Tensor<Scalar, 3> &en2R) {
    //     using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    //     auto v           = Eigen::Map<const VectorType>(mps.data(), mps.size());
    //
    //     // auto H1t    = tools::common::contraction::matrix_vector_product(mps, mpo1, en1L, en1R);
    //     // auto Hv     = Eigen::Map<const VectorType>(H1t.data(), H1t.size());
    //     // auto vHv    = v.dot(Hv);
    //     auto H2t = tools::common::contraction::matrix_vector_product(mps, mpo2, en2L, en2R);
    //     auto H2v = Eigen::Map<const VectorType>(H2t.data(), H2t.size());
    //     // auto vH2v   = v.dot(H2v);
    //     // auto norm_1 = 1.0 / v.norm();
    //     auto pref = std::is_same_v<Scalar, fp64> ? 2.0 : 1.0; // Factor 2 for real
    //     // auto grad   = pref * norm_1 * (H2v - 2.0 * vHv * Hv - (vH2v - 2.0 * vHv * vHv) * v);
    //     auto grad = pref * H2v;
    //     return grad.template lpNorm<Eigen::Infinity>();
    // }
    //
    // double max_gradient(const Eigen::Tensor<cx64, 3> &mps, const TensorsFinite<Scalar> &tensors) {
    //     auto t_grad = tid::tic_token("grad");
    //
    //     if(tensors.is_real()) {
    //         Eigen::Tensor<fp64, 3> mpsr = mps.real();
    //         Eigen::Tensor<fp64, 3> en1L = tensors.get_multisite_env_ene_blk().L.real();
    //         Eigen::Tensor<fp64, 3> en1R = tensors.get_multisite_env_ene_blk().R.real();
    //         Eigen::Tensor<fp64, 3> en2L = tensors.get_multisite_env_var_blk().L.real();
    //         Eigen::Tensor<fp64, 3> en2R = tensors.get_multisite_env_var_blk().R.real();
    //         Eigen::Tensor<fp64, 4> mpo1 = tensors.get_multisite_mpo().real();
    //         Eigen::Tensor<fp64, 4> mpo2 = tensors.get_multisite_mpo_squared().real();
    //         return max_gradient(mpsr, mpo1, en1L, en1R, mpo2, en2L, en2R);
    //     } else {
    //         const auto &en1  = tensors.get_multisite_env_ene_blk();
    //         const auto &en2  = tensors.get_multisite_env_var_blk();
    //         const auto &mpo1 = tensors.get_multisite_mpo();
    //         const auto &mpo2 = tensors.get_multisite_mpo_squared();
    //         return max_gradient(mps, mpo1, en1.L, en1.R, mpo2, en2.L, en2.R);
    //     }

    //    auto v = Eigen::Map<const Eigen::VectorXcd>(mps.data(), mps.size());
    //
    //    auto en1 = tensors.get_multisite_env_ene_blk();
    //    auto H1t = tools::common::contraction::matrix_vector_product(mps, tensors.get_multisite_mpo(), en1.L, en1.R);
    //    auto Hv  = Eigen::Map<const Eigen::VectorXcd>(H1t.data(), H1t.size());
    //    auto vHv = v.dot(Hv);
    //
    //    auto en2  = tensors.get_multisite_env_var_blk();
    //    auto H2t  = tools::common::contraction::matrix_vector_product(mps, tensors.get_multisite_mpo_squared(), en2.L, en2.R);
    //    auto H2v  = Eigen::Map<const Eigen::VectorXcd>(H2t.data(), H2t.size());
    //    auto vH2v = v.dot(H2v);
    //
    //    double var    = std::abs(vH2v - vHv * vHv);
    //    auto   var_1  = 1.0 / var / std::log(10);
    //    auto   norm_1 = 1.0 / v.norm();
    //    auto   pref   = tensors.is_real() ? 1.0 : 2.0;
    //    auto   grad   = pref * var_1 * norm_1 * (H2v - 2.0 * vHv * Hv - (vH2v - 2.0 * vHv * vHv) * v); // Factor 2 for complex
    //    return grad.template lpNorm<Eigen::Infinity>();
    // }

}

template<typename MatVecType, typename Scalar>
void eigs_folded_spectrum_executor(eig::solver &solver, MatVecType &hamiltonian_squared, const TensorsFinite<Scalar> &tensors,
                                   const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results, const OptMeta &meta) {
    using CalcType                        = typename MatVecType::Scalar;
    solver.config.primme_effective_ham_sq = &hamiltonian_squared;
    hamiltonian_squared.reset();

    tools::log->trace("eigs_folded_spectrum_executor: Defining the Hamiltonian-squared matrix-vector product");
    if(!solver.config.lib.has_value()) throw except::runtime_error("lib has not been set");
    switch(solver.config.lib.value()) {
        case eig::Lib::ARPACK: {
            if(tensors.model->has_compressed_mpo_squared()) throw std::runtime_error("optimize_folded_spectrum_eigs with ARPACK requires non-compressed MPO²");
            if(not solver.config.ritz) solver.config.ritz = eig::Ritz::SM;
            if(not solver.config.sigma)
                solver.config.sigma =
                    folded_spectrum::get_largest_eigenvalue_hamiltonian_squared<CalcType>(tensors) + RealScalar<CalcType>(1); // Add one to shift enough
            break;
        }
        case eig::Lib::PRIMME: {
            if(not solver.config.ritz) solver.config.ritz = eig::Ritz::SM;
            if(solver.config.sigma and solver.config.sigma.value() != 0.0 and tensors.model->has_compressed_mpo_squared())
                throw except::logic_error("optimize_folded_spectrum_eigs with PRIMME with sigma requires non-compressed MPO²");
            break;
        }
        case eig::Lib::SPECTRA: {
            if(not solver.config.ritz) solver.config.ritz = eig::Ritz::SM;
            break;
        }
    }

    tools::log->debug("eigs_folded_spectrum_executor: Solving [H²x=λx] {} {} | sigma {} | shifts {} | maxIter {} | tol {:.2e} | nev {} ncv {} | size {} | "
                      "mps {} | jcb {}",
                      eig::LibToString(solver.config.lib), eig::RitzToString(solver.config.ritz), solver.config.sigma, solver.config.primme_targetShifts,
                      solver.config.maxIter.value(), solver.config.tol.value(), solver.config.maxNev.value(), solver.config.maxNcv.value(),
                      hamiltonian_squared.rows(), hamiltonian_squared.get_shape_mps(), solver.config.jcbMaxBlockSize);

    auto init = folded_spectrum::get_initial_guess_mps<CalcType>(initial_mps, results,
                                                                 solver.config.maxNev.value()); // Init holds the data in memory for this scope
    for(auto &i : init) solver.config.initial_guess.push_back({i.mps.data(), i.idx});
    solver.eigs(hamiltonian_squared);
    internal::extract_results<CalcType>(tensors, initial_mps, meta, solver, results, false);
}

template<typename CalcType, typename Scalar>
void eigs_manager_folded_spectrum(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results,
                                  const OptMeta &meta) {
    eig::solver solver;
    auto       &cfg           = solver.config;
    cfg.loglevel              = 2;
    cfg.compute_eigvecs       = eig::Vecs::ON;
    cfg.tol                   = meta.eigs_tol.value_or(settings::precision::eigs_tol_min); // 1e-12 is good. This Sets "eps" in primme, see link above.
    cfg.maxIter               = meta.eigs_iter_max.value_or(settings::precision::eigs_iter_max);
    cfg.maxNev                = meta.eigs_nev.value_or(1);
    cfg.maxNcv                = meta.eigs_ncv.value_or(settings::precision::eigs_ncv);
    cfg.maxTime               = meta.eigs_time_max.value_or(2 * 60 * 60); // Two hours default
    cfg.primme_minRestartSize = meta.primme_minRestartSize;
    cfg.primme_maxBlockSize   = meta.primme_maxBlockSize;
    cfg.primme_locking        = 0;

    cfg.lib           = eig::Lib::PRIMME;
    cfg.primme_method = eig::stringToMethod(meta.primme_method);
    cfg.tag += meta.label;
    switch(meta.optRitz) {
        case OptRitz::SR: cfg.ritz = eig::Ritz::primme_smallest; break;
        case OptRitz::LR: cfg.ritz = eig::Ritz::primme_largest; break;
        case OptRitz::LM: cfg.ritz = eig::Ritz::primme_largest_abs; break;
        case OptRitz::SM: {
            cfg.ritz                = eig::Ritz::primme_closest_abs; // H² is positive definite!
            cfg.primme_projection   = meta.primme_projection.value_or("primme_proj_default");
            cfg.primme_targetShifts = {meta.eigv_target.value_or(0.0)};
            break;
        }
        default: throw except::logic_error("undhandled ritz: {}", enum2sv(meta.optRitz));
    }

    cfg.primme_preconditioner = folded_spectrum::preconditioner_jacobi<CalcType>;
    cfg.jcbMaxBlockSize       = meta.eigs_jcbMaxBlockSize;

    const auto &mpos                  = tensors.get_model().get_mpo_active();
    const auto &envv                  = tensors.get_edges().get_var_active();
    auto        hamiltonian_squared   = MatVecMPOS<CalcType>(mpos, envv);
    hamiltonian_squared.factorization = eig::Factorization::LLT; // H² is positive definite so this should work for the preconditioner

    // if(tensors.active_problem_size() >= 4096 and tensors.state->get_direction() == 1) {
    //     auto matrix    = hamiltonian_squared.get_matrix();
    //     auto shape     = hamiltonian_squared.get_shape_mps();
    //     auto h5file    = h5pp::File("hamtemp.h5", h5pp::FileAccess::REPLACE);
    //     h5file.writeDataset(matrix, "matrix");
    //     h5file.writeAttribute(shape, "matrix", "shape");
    //
    //     auto   posL       = tensors.active_sites.front();
    //     auto   posR       = tensors.active_sites.back();
    //     auto   LL         = tensors.state->get_mps_site(posL).get_L();
    //     auto   RL         = posL == posR ? tensors.state->get_mps_site(posR).get_LC() :tensors.state->get_mps_site(posR).get_L() ;
    //     EnvVar envvL      = envv.L;
    //     EnvVar envvR      = envv.R;
    //     envvL.get_block() = envv.L.get_block()
    //                             .contract(tenx::asDiagonal(LL), tenx::idx({0}, {0}))
    //                             .contract(tenx::asDiagonal(LL), tenx::idx({0}, {0}))
    //                             .shuffle(tenx::array3{1, 2, 0});
    //     envvR.get_block() = envv.R.get_block()
    //                             .contract(tenx::asDiagonal(RL), tenx::idx({0}, {1}))
    //                             .contract(tenx::asDiagonal(RL), tenx::idx({0}, {1}))
    //                             .shuffle(tenx::array3{1, 2, 0});
    //     auto envv_temp = env_pair<const EnvVar &>(envvL, envvR);
    //     auto hamiltonian_temp = MatVecMPOS<Scalar>(mpos, envv_temp);
    //     auto matrix_temp      = hamiltonian_temp.get_matrix();
    //     auto shape_temp       = hamiltonian_temp.get_shape_mps();
    //     h5file.writeDataset(matrix_temp, "matrix_temp");
    //     h5file.writeAttribute(shape_temp, "matrix_temp", "shape");
    //     exit(0);
    // }

    eigs_folded_spectrum_executor(solver, hamiltonian_squared, tensors, initial_mps, results, meta);
}
template<typename Scalar>
opt_mps<Scalar> tools::finite::opt::internal::optimize_folded_spectrum(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                   [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<Scalar> &elog) {
    if(meta.optSolver == OptSolver::EIG) return optimize_folded_spectrum_eig(tensors, initial_mps, status, meta, elog);

    using namespace internal;
    using namespace settings::precision;
    initial_mps.validate_initial_mps();
    elog.eigs_add_entry(initial_mps, spdlog::level::debug);

    auto                         t_var = tid::tic_scope("eigs-xdmrg", tid::level::higher);
    std::vector<opt_mps<Scalar>> results;
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_folded_spectrum<fp32>(tensors, initial_mps, results, meta); break;
            case OptType::CX32: eigs_manager_folded_spectrum<cx32>(tensors, initial_mps, results, meta); break;
            case OptType::FP64: eigs_manager_folded_spectrum<fp64>(tensors, initial_mps, results, meta); break;
            case OptType::CX64: eigs_manager_folded_spectrum<cx64>(tensors, initial_mps, results, meta); break;
            case OptType::FP128: eigs_manager_folded_spectrum<fp128>(tensors, initial_mps, results, meta); break;
            case OptType::CX128: eigs_manager_folded_spectrum<cx128>(tensors, initial_mps, results, meta); break;
            default: throw except::logic_error("optimize_folded_spectrum: not implemented for type {}", enum2sv(meta.optType));
        }
    } else {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_folded_spectrum<fp32>(tensors, initial_mps, results, meta); break;
            case OptType::FP64: eigs_manager_folded_spectrum<fp64>(tensors, initial_mps, results, meta); break;
            case OptType::FP128: eigs_manager_folded_spectrum<fp128>(tensors, initial_mps, results, meta); break;
            case OptType::CX32: throw except::logic_error("Cannot run OptType::CX32 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
            default: throw except::logic_error("optimize_folded_spectrum: not implemented for type {}", enum2sv(meta.optType));
        }
    }

    auto t_post = tid::tic_scope("post");
    if(results.empty()) {
        meta.optExit = OptExit::FAIL_ERROR;
        return initial_mps; // The solver failed
    }

    // Sort results
    if(results.size() > 1) { std::sort(results.begin(), results.end(), internal::Comparator<Scalar>(meta)); }
    for(const auto &result : results) elog.eigs_add_entry(result, spdlog::level::debug);

    return results.front();
}
