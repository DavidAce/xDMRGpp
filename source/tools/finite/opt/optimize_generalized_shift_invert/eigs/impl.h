#pragma once

#include "../../../opt_meta.h"
#include "../../../opt_mps.h"
#include "../../launch_gdplusk.h"
#include "config/settings.h"
#include "general/sfinae.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include <primme/primme.h>

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;
namespace gsi {
    template<typename Scalar>
    void massMatrixMatvec(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
        if(x == nullptr) return;
        if(y == nullptr) return;
        if(primme == nullptr) return;
        const auto H_ptr = static_cast<MatVecMPOS<Scalar> *>(primme->matrix);
        H_ptr->MultBx(x, ldx, y, ldy, blockSize, primme, ierr);
    }
    template<typename Scalar>
    void preconditioner_jacobi(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
        if(x == nullptr) return;
        if(y == nullptr) return;
        if(primme == nullptr) return;
        const auto H_ptr      = static_cast<MatVecMPOS<Scalar> *>(primme->matrix);
        H_ptr->preconditioner = eig::Preconditioner::JACOBI;
        H_ptr->factorization  = eig::Factorization::LLT;
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
        H_ptr->set_iterativeLinearSolverConfig(10000, RealScalar{0.1}, MatDef::DEF);
        H_ptr->MultPc(x, ldx, y, ldy, blockSize, primme, ierr);
    }
    template<typename Scalar>
    struct opt_mps_init_t {
        Eigen::Tensor<Scalar, 3> mps = {};
        long                     idx = 0;
    };
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
    Eigen::Tensor<Scalar, 3> get_initial_guess(const opt_mps<Scalar> &initial_mps, const std::vector<opt_mps<Scalar>> &results) {
        if(results.empty()) {
            return initial_mps.template get_tensor_as<CalcType>();
        } else {
            // Return whichever of initial_mps or results that has the lowest variance
            auto it = std::min_element(results.begin(), results.end(), internal::comparator::variance);
            if(it == results.end()) return get_initial_guess<CalcType>(initial_mps, {});

            if(it->get_variance() < initial_mps.get_variance()) {
                tools::log->debug("Previous result is a good initial guess: {} | var {:8.2e}", it->get_name(), fp(it->get_variance()));
                return get_initial_guess<CalcType>(*it, {});
            } else
                return get_initial_guess<CalcType>(initial_mps, {});
        }
    }
}

template<typename MatVecType, typename Scalar>
void eigs_generalized_shift_invert_executor(eig::solver &solver, MatVecType &hamiltonian_squared, const TensorsFinite<Scalar> &tensors,
                                            const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results, const OptMeta &meta,
                                            reports::eigs_log<Scalar> &elog) {
    using CalcType = typename MatVecType::Scalar;
    if(std::is_same_v<CalcType, cx64> and meta.optType == OptType::FP64)
        throw except::logic_error("eigs_variance_executor error: Mixed Scalar:cx64 with OptType::FP64");
    if(std::is_same_v<CalcType, fp64> and meta.optType == OptType::CX64)
        throw except::logic_error("eigs_variance_executor error: Mixed Scalar:real with OptType::CX64");

    solver.config.primme_effective_ham_sq = &hamiltonian_squared;
    hamiltonian_squared.reset();

    tools::log->trace("eigs_variance_executor: Defining the Hamiltonian-squared matrix-vector product");
    switch(solver.config.lib.value_or(eig::Lib::EIGSMPO)) {
        case eig::Lib::ARPACK: throw except::logic_error("optimize_generalized_shift_invert_eigs: ARPACK is not supported");
        case eig::Lib::EIGSMPO: {
            results = eigs_gdplusk<CalcType>(tensors, initial_mps, meta, elog);
            return;
        }
        default: break;
    }

    if(solver.config.lib == eig::Lib::ARPACK) {
        throw except::logic_error("optimize_generalized_shift_invert_eigs: ARPACK is not supported");
    } else if(solver.config.lib == eig::Lib::PRIMME) {
        if(solver.config.sigma and solver.config.sigma.value() != 0.0 and tensors.model->has_compressed_mpo_squared())
            throw except::logic_error(
                "optimize_generalized_shift_invert_eigs: sigma shift is not supported: subtract the sigma/L at the global mpo level instead");
    }

    tools::log->debug(
        "eigs_generalized_shift_invert_executor: Solving [Hx=λH²x] {} {} | mmin {} mmax {} | shifts {} | maxIter {} | tol {:.2e} | size {} = {} | jcb {}",
        eig::RitzToString(solver.config.ritz), eig::MethodToString(solver.config.primme_method), solver.config.primme_minRestartSize,
        solver.config.primme_maxBlockSize, solver.config.primme_targetShifts, solver.config.maxIter.value(), solver.config.tol.value(),
        hamiltonian_squared.get_shape_mps(), hamiltonian_squared.rows(), solver.config.jcbMaxBlockSize);

    auto init = gsi::get_initial_guess_mps<CalcType>(initial_mps, results, solver.config.maxNev.value()); // Init holds the data in memory for this scope
    for(auto &i : init) solver.config.initial_guess.push_back({i.mps.data(), i.idx});
    solver.eigs(hamiltonian_squared);
    internal::extract_results<CalcType>(tensors, initial_mps, meta, solver, results, false);
}

template<typename CalcType, typename Scalar>
void eigs_manager_generalized_shift_invert(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results,
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
    cfg.lib                   = meta.eigs_lib.empty() ? eig::Lib::PRIMME : eig::StringToLib(meta.eigs_lib);
    cfg.primme_minRestartSize = meta.primme_minRestartSize;
    cfg.primme_maxBlockSize   = meta.primme_maxBlockSize;
    cfg.primme_locking        = 0;
    cfg.primme_method         = eig::stringToMethod(meta.primme_method);
    cfg.tag += meta.label;
    switch(meta.optRitz) {
        case OptRitz::SR: cfg.ritz = eig::Ritz::primme_smallest; break;
        case OptRitz::LR: cfg.ritz = eig::Ritz::primme_largest; break;
        case OptRitz::LM: cfg.ritz = eig::Ritz::primme_largest_abs; break;
        case OptRitz::SM: {
            cfg.ritz                = eig::Ritz::primme_closest_abs; // H² is positive definite!
            cfg.primme_targetShifts = {meta.eigv_target.value_or(0.0)};
            cfg.primme_projection   = meta.primme_projection.value_or("primme_proj_refined");
            break;
        }
        default: throw except::logic_error("undhandled ritz: {}", enum2sv(meta.optRitz));
    }

    cfg.primme_massMatrixMatvec = gsi::massMatrixMatvec<CalcType>;
    cfg.primme_projection       = meta.primme_projection.value_or("primme_proj_RR");
    // cfg.primme_projection       = meta.primme_projection.value_or("primme_proj_default");
    // #pragma message "revert primme method"
    // cfg.primme_method           = eig::PrimmeMethod::PRIMME_DYNAMIC;
    // cfg.primme_targetShifts.clear();
    cfg.primme_targetShifts   = {meta.eigv_target.value_or(0.0)};
    cfg.primme_preconditioner = gsi::preconditioner_jacobi<CalcType>;
    cfg.jcbMaxBlockSize       = meta.eigs_jcbMaxBlockSize;

    // Overrides from default
    const auto &mpos                  = tensors.get_model().get_mpo_active();
    const auto &enve                  = tensors.get_edges().get_ene_active();
    const auto &envv                  = tensors.get_edges().get_var_active();
    auto        hamiltonian_squared   = MatVecMPOS<CalcType>(mpos, enve, envv);
    hamiltonian_squared.factorization = eig::Factorization::LLT;
    eigs_generalized_shift_invert_executor(solver, hamiltonian_squared, tensors, initial_mps, results, meta, elog);
}

template<typename Scalar>
opt_mps<Scalar> internal::optimize_generalized_shift_invert(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                            [[maybe_unused]] OptMeta &meta, reports::eigs_log<Scalar> &elog) {
    if(meta.optSolver == OptSolver::EIG) return optimize_generalized_shift_invert_eig(tensors, initial_mps, meta, elog);

    using namespace internal;
    using namespace settings::precision;
    initial_mps.validate_initial_mps();
    elog.eigs_add_entry(initial_mps, spdlog::level::debug);

    auto                         t_gdmrg = tid::tic_scope("eigs-gdmrg", tid::level::higher);
    std::vector<opt_mps<Scalar>> results;
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_generalized_shift_invert<fp32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP64: eigs_manager_generalized_shift_invert<fp64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP128: eigs_manager_generalized_shift_invert<fp128>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX32: eigs_manager_generalized_shift_invert<cx32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX64: eigs_manager_generalized_shift_invert<cx64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX128: eigs_manager_generalized_shift_invert<cx128>(tensors, initial_mps, results, meta, elog); break;
            default: throw except::runtime_error("optimize_generalized_shift_invert(): not implemented for type {}", enum2sv(meta.optType));
        }
    } else {
        switch(meta.optType) {
            case OptType::FP32: eigs_manager_generalized_shift_invert<fp32>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP64: eigs_manager_generalized_shift_invert<fp64>(tensors, initial_mps, results, meta, elog); break;
            case OptType::FP128: eigs_manager_generalized_shift_invert<fp128>(tensors, initial_mps, results, meta, elog); break;
            case OptType::CX32: throw except::logic_error("Cannot run OptType::CX32 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX128: throw except::logic_error("Cannot run OptType::CX128 with Scalar type {}", sfinae::type_name<Scalar>());
            default: throw except::runtime_error("optimize_generalized_shift_invert(): not implemented for type {}", enum2sv(meta.optType));
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
