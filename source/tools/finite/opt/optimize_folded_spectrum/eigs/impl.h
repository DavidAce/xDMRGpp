#pragma once

#include "../../../opt_meta.h"
#include "../../../opt_mps.h"
#include "../../launch_gdplusk.h"
#include "config/settings.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/matvec/matvec_zero.h"
#include "math/eig/solver_eigsmpo/solver_gdplusk.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
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
    namespace primme_functionals {
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
        void preconditioner_linearsolver(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
            if(x == nullptr) return;
            if(y == nullptr) return;
            if(primme == nullptr) return;
            using RealScalar      = typename MatVecMPOS<CalcType>::RealScalar;
            const auto H_ptr      = static_cast<MatVecMPOS<CalcType> *>(primme->matrix);
            H_ptr->preconditioner = eig::Preconditioner::SOLVE;
            H_ptr->factorization  = eig::Factorization::LLT;
            H_ptr->set_iterativeLinearSolverConfig(10000, RealScalar{0.1f}, MatDef::DEF);
            H_ptr->MultPc(x, ldx, y, ldy, blockSize, primme, ierr);
            // primme->stats.numMatvecs += H_ptr->get_iterativeLinearSolverConfig().result.matvecs;
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
                    tools::log->info("eval {:38.32f} ({:+8.3e}) | rnorm {:38.32f} ({:+8.3e}) | problemNorm {:.3e}", *eval, diff_evals_sum, *rNorm,
                                     diff_rnorms_sum, problemNorm);
                }
            } else {
                *isconv = 0;
            }

            *ierr = 0;
        }
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

}

// template<typename Scalar>
// [[nodiscard]] opt_mps<Scalar> tools::finite::opt::internal::optimize_lanczos_h1h2(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial,
//                                                                                   [[maybe_unused]] OptMeta &meta, reports::eigs_log<Scalar> &elog) {
//     using namespace internal;
//     using namespace settings::precision;
//     initial.validate_initial_mps();
//     elog.eigs_add_entry(initial, spdlog::level::debug);
//
//     auto token = tid::tic_scope(fmt::format("h1h2-{}", enum2sv(meta.optAlgo)), tid::level::higher);
//
//     std::string eigprob;
//     switch(meta.optAlgo) {
//         case OptAlgo::DMRG: eigprob = "Hx=λx"; break;
//         case OptAlgo::DMRGX: eigprob = "Hx=λx"; break;
//         case OptAlgo::HYBRID_DMRGX: eigprob = "Hx=λx"; break;
//         case OptAlgo::XDMRG: eigprob = "H²x=λx"; break;
//         case OptAlgo::GDMRG: eigprob = "Hx=λH²x"; break;
//     }
//
//     tools::log->debug("eigs_lanczos_h1h2_executor: Solving [{}] | ritz {} | maxIter {} | tol {:.2e} | init on | size {} | mps {}", eigprob,
//                       enum2sv(meta.optRitz), meta.eigs_iter_max, meta.eigs_tol, initial.get_tensor().size(), initial.get_tensor().dimensions());
//     if constexpr(sfinae::is_std_complex_v<Scalar>) {
//         switch(meta.optType) {
//             case OptType::FP32: return eigs_lanczos_h1h2<fp32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::FP64: return eigs_lanczos_h1h2<fp64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::FP128: return eigs_lanczos_h1h2<fp128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::CX32: return eigs_lanczos_h1h2<cx32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::CX64: return eigs_lanczos_h1h2<cx64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::CX128: return eigs_lanczos_h1h2<cx128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             default: throw std::runtime_error("unrecognized option type");
//         }
//     } else {
//         switch(meta.optType) {
//             case OptType::FP32: return eigs_lanczos_h1h2<fp32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::FP64: return eigs_lanczos_h1h2<fp64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::FP128: return eigs_lanczos_h1h2<fp128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
//             case OptType::CX32: throw except::logic_error("Cannot run OptType::CX32 with Scalar type {}", sfinae::type_name<Scalar>());
//             case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
//             case OptType::CX128: throw except::logic_error("Cannot run OptType::CX128 with Scalar type {}", sfinae::type_name<Scalar>());
//             default: throw std::runtime_error("unrecognized option type");
//         }
//     }
// }

template<typename MatVecType, typename Scalar>
void eigs_executor_folded_spectrum(eig::solver &solver, MatVecType &hamiltonian_squared, const TensorsFinite<Scalar> &tensors,
                                   const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results, const OptMeta &meta,
                                   reports::eigs_log<Scalar> &elog) {
    using CalcType = typename MatVecType::Scalar;
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
            solver.config.primme_effective_ham_sq = &hamiltonian_squared;
            if(not solver.config.ritz) solver.config.ritz = eig::Ritz::SM;
            if(solver.config.sigma and solver.config.sigma.value() != 0.0 and tensors.model->has_compressed_mpo_squared())
                throw except::logic_error("optimize_folded_spectrum_eigs with PRIMME with sigma requires non-compressed MPO²");
            break;
        }
        case eig::Lib::SPECTRA: {
            if(not solver.config.ritz) solver.config.ritz = eig::Ritz::SM;
            break;
        }
        case eig::Lib::EIGSMPO: {
            results = eigs_gdplusk<CalcType>(tensors, initial_mps, meta, elog);
            return;
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
                                  const OptMeta &meta, reports::eigs_log<Scalar> &elog) {
    eig::solver solver;
    auto       &cfg           = solver.config;
    cfg.loglevel              = 2;
    cfg.compute_eigvecs       = eig::Vecs::ON;
    cfg.tol                   = meta.eigs_tol.value_or(settings::precision::eigs_tol_min); // 1e-12 is good. This Sets "eps" in primme, see link above.
    cfg.maxIter               = meta.eigs_iter_max.value_or(settings::precision::eigs_iter_max);
    cfg.maxNev                = meta.eigs_nev.value_or(1);
    cfg.maxNcv                = meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    cfg.maxTime               = meta.eigs_time_max.value_or(2 * 60 * 60); // Two hours default
    cfg.primme_minRestartSize = meta.primme_minRestartSize;
    cfg.primme_maxBlockSize   = meta.primme_maxBlockSize;
    cfg.primme_locking        = 0;

    cfg.lib           = meta.eigs_lib.empty() ? eig::Lib::EIGSMPO : eig::StringToLib(meta.eigs_lib);
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
    if(meta.eigs_jcbMaxBlockSize.has_value() and meta.eigs_jcbMaxBlockSize.value() > 0) {
        cfg.primme_preconditioner = folded_spectrum::primme_functionals::preconditioner_linearsolver<CalcType>;
        cfg.jcbMaxBlockSize       = meta.eigs_jcbMaxBlockSize;
    }

    const auto &mpos                  = tensors.get_model().get_mpo_active();
    const auto &envv                  = tensors.get_edges().get_var_active();
    auto        hamiltonian_squared   = MatVecMPOS<CalcType>(mpos, envv);
    hamiltonian_squared.factorization = eig::Factorization::LLT; // H² is positive definite so this should work for the preconditioner

    eigs_executor_folded_spectrum(solver, hamiltonian_squared, tensors, initial_mps, results, meta, elog);
}
template<typename Scalar>
opt_mps<Scalar> tools::finite::opt::internal::optimize_folded_spectrum(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, OptMeta &meta,
                                                                       reports::eigs_log<Scalar> &elog) {
    if(meta.optSolver == OptSolver::EIG) return optimize_folded_spectrum_eig(tensors, initial_mps, meta, elog);

    using namespace internal;
    using namespace settings::precision;
    initial_mps.validate_initial_mps();
    elog.eigs_add_entry(initial_mps, spdlog::level::debug);

    auto                         t_var = tid::tic_scope("eigs-xdmrg", tid::level::higher);
    std::vector<opt_mps<Scalar>> results;
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_folded_spectrum<fp32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX32: eigs_manager_folded_spectrum<cx32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP64: eigs_manager_folded_spectrum<fp64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX64: eigs_manager_folded_spectrum<cx64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP128: eigs_manager_folded_spectrum<fp128>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX128: eigs_manager_folded_spectrum<cx128>(tensors, initial_mps, results, meta, elog); break;
            default: throw except::logic_error("optimize_folded_spectrum: not implemented for type {}", enum2sv(meta.optType));
        }
    } else {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_folded_spectrum<fp32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP64: eigs_manager_folded_spectrum<fp64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP128: eigs_manager_folded_spectrum<fp128>(tensors, initial_mps, results, meta, elog); break;
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
