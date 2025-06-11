#pragma once
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
#include <primme/primme.h>

using namespace  tools::finite::opt;
template<typename Scalar>
void preconditioner(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
    if(x == nullptr) return;
    if(y == nullptr) return;
    if(primme == nullptr) return;
    const auto H_ptr = static_cast<MatVecMPOS<Scalar> *>(primme->matrix);
    H_ptr->FactorOP();
    H_ptr->MultOPv(x, ldx, y, ldy, blockSize, primme, ierr);
}

template<typename Scalar>
void preconditioner_jacobi(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
    if(x == nullptr) return;
    if(y == nullptr) return;
    if(primme == nullptr) return;
    auto       t_precond  = tid::tic_scope("prec_jcb");
    const auto H_ptr      = static_cast<MatVecMPOS<Scalar> *>(primme->matrix);
    H_ptr->preconditioner = eig::Preconditioner::JACOBI;
    H_ptr->MultPc(x, ldx, y, ldy, blockSize, primme, ierr);
}

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> optimize_energy_eigs_executor(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta) {
    if(meta.optAlgo != OptAlgo::DMRG)
        throw except::runtime_error("eigs_energy_executor: Expected OptCost [{}] | Got [{}]", enum2sv(OptAlgo::DMRG), enum2sv(meta.optAlgo));

    eig::Ritz ritz = eig::stringToRitz(enum2sv(meta.optRitz));
    tools::log->trace("Defining Hamiltonian matrix-vector product");
    tools::log->trace("Defining eigenvalue solver");
    eig::solver solver;

    const auto          &mpo = tensors.get_model().get_mpo_active();
    const auto          &env = tensors.get_edges().get_ene_active();
    MatVecMPOS<CalcType> hamiltonian(mpo, env);
    hamiltonian.factorization = eig::Factorization::LU; // H is not definite, so we can't use LLT or LDLT

    // https://www.cs.wm.edu/~andreas/software/doc/appendix.html#c.primme_params.eps
    solver.config.tol                   = meta.eigs_tol.value_or(settings::precision::eigs_tol_min);
    solver.config.compute_eigvecs       = eig::Vecs::ON;
    solver.config.lib                   = eig::Lib::PRIMME;
    solver.config.primme_method         = eig::stringToMethod(meta.primme_method);
    solver.config.maxIter               = meta.eigs_iter_max.value_or(settings::precision::eigs_iter_max);
    solver.config.maxTime               = 2 * 60 * 60;
    solver.config.maxNev                = meta.eigs_nev.value_or(1);
    solver.config.maxNcv                = meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min); // arpack needs ncv ~512. Primme seems happy with 4-32.
    solver.config.primme_minRestartSize = meta.primme_minRestartSize;
    solver.config.primme_maxBlockSize   = meta.primme_maxBlockSize;
    solver.config.primme_locking        = true;
    solver.config.loglevel              = 2;

    Eigen::Tensor<CalcType, 3> init;
    init = initial_mps.template get_tensor_as<CalcType>();
    if constexpr(std::is_same_v<CalcType, RealScalar<CalcType>>) {
        if(ritz == eig::Ritz::SR) ritz = eig::Ritz::SA;
        if(ritz == eig::Ritz::LR) ritz = eig::Ritz::LA;
    }
    solver.config.ritz = ritz;
    solver.config.initial_guess.push_back({init.data(), 0});
    tools::log->trace("Finding energy eigenstate {}", enum2sv(meta.optRitz));

    if(meta.optRitz == OptRitz::SM) {
        // This is used to find excited eigenstates of H
        if(meta.problem_size <= settings::precision::eigs_max_size_shift_invert) {
            hamiltonian.factorization           = eig::Factorization::LU;
            solver.config.ritz                  = eig::Ritz::primme_largest_abs;
            solver.config.sigma                 = cx64(meta.eigv_target.value_or(0.0), 0.0);
            solver.config.primme_projection     = "primme_proj_default";
            solver.config.primme_locking        = false;
            solver.config.primme_preconditioner = preconditioner_jacobi<Scalar>;
            solver.config.primme_preconditioner = preconditioner_jacobi<Scalar>;
            solver.config.jcbMaxBlockSize       = meta.problem_size;
        } else {
            tools::log->debug("eigs_energy_executor: ignoring OptAlgo::SHIFTINV: "
                              "problem size {} > settings::precision::eigs_max_size_shift_invert ({})",
                              meta.problem_size, settings::precision::eigs_max_size_shift_invert);
            // The problem size is too big. Just find the eigenstate closest to zero
            // Remember: since the Hamiltonian is expected to be shifted, the target energy is near zero.
            solver.config.ritz                  = eig::Ritz::primme_closest_abs;
            solver.config.primme_projection     = "primme_proj_refined";
            solver.config.primme_targetShifts   = {meta.eigv_target.value_or(0.0)};
            solver.config.primme_preconditioner = preconditioner_jacobi<Scalar>;
            solver.config.jcbMaxBlockSize       = meta.eigs_jcbMaxBlockSize.value_or(1);
        }
    }

    solver.eigs(hamiltonian);

    std::vector<opt_mps<Scalar>> results;
    internal::extract_results<CalcType>(tensors, initial_mps, meta, solver, results, false);
    return results;
}


template<typename Scalar>
opt_mps<Scalar> internal::optimize_energy(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const AlgorithmStatus &status,
                                          OptMeta &meta, reports::eigs_log<Scalar> &elog) {
    if(meta.optSolver == OptSolver::EIG) return optimize_energy_eig(tensors, initial_mps, status, meta, elog);
    if constexpr(tenx::sfinae::is_quadruple_prec_v<Scalar> or tenx::sfinae::is_single_prec_v<Scalar>) {
        throw except::runtime_error("optimize_energy_eigs(): not implemented for type {}", enum2sv(meta.optType));
    }
    tools::log->debug("optimize_energy_eigs: ritz {} | type {} | algo {}", enum2sv(meta.optRitz), enum2sv(meta.optType), enum2sv(meta.optAlgo));

    initial_mps.validate_initial_mps();
    // if(not tensors.model->is_shifted()) throw std::runtime_error("optimize_variance_eigs requires energy-shifted MPO²");
    elog.eigs_add_entry(initial_mps, spdlog::level::debug);

    auto                         t_eigs = tid::tic_scope("eigs-ene", tid::higher);
    std::vector<opt_mps<Scalar>> results;
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP64: results = optimize_energy_eigs_executor<fp64>(tensors, initial_mps, meta); break;
            case OptType::CX64: results = optimize_energy_eigs_executor<cx64>(tensors, initial_mps, meta); break;
            default: throw except::runtime_error("optimize_energy(): not implemented for type {}", enum2sv(meta.optType));
        }
    } else {
        switch(meta.optType) {
            case OptType::FP64: results = optimize_energy_eigs_executor<fp64>(tensors, initial_mps, meta); break;
            case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
            default: throw except::runtime_error("optimize_energy(): not implemented for type {}", enum2sv(meta.optType));
        }
    }

    auto t_post = tid::tic_scope("post");
    if(results.empty()) {
        meta.optExit = OptExit::FAIL_ERROR;
        return initial_mps; // The solver failed
    }

    auto comparator = [&meta](const opt_mps<Scalar> &lhs, const opt_mps<Scalar> &rhs) {
        // auto diff = std::abs(lhs.get_energy_shifted() - rhs.get_energy_shifted());
        // if(diff < settings::precision::eigs_tol_min) return lhs.get_overlap() > rhs.get_overlap();
        auto lhs_abs_energy = std::abs(lhs.get_energy()) + std::sqrt(std::abs(lhs.get_variance()));
        auto rhs_abs_energy = std::abs(rhs.get_energy()) + std::sqrt(std::abs(rhs.get_variance()));
        auto lhs_min_energy = lhs.get_energy() + std::sqrt(std::abs(lhs.get_variance()));
        auto rhs_min_energy = rhs.get_energy() + std::sqrt(std::abs(rhs.get_variance()));
        auto lhs_max_energy = lhs.get_energy() - std::sqrt(std::abs(lhs.get_variance()));
        auto rhs_max_energy = rhs.get_energy() - std::sqrt(std::abs(rhs.get_variance()));
        switch(meta.optRitz) {
            case OptRitz::SR: return lhs_min_energy < rhs_min_energy;
            case OptRitz::LR: return lhs_max_energy > rhs_max_energy;
            case OptRitz::SM: {
                // return std::abs(lhs.get_eigs_eigval()) < std::abs(rhs.get_eigs_eigval());
                // auto diff_energy_lhs = std::abs(lhs.get_energy() - initial_mps.get_energy());
                // auto diff_energy_rhs = std::abs(rhs.get_energy() - initial_mps.get_energy());
                return lhs_abs_energy < rhs_abs_energy;
            }
            default: throw except::runtime_error("Ground state optimization with ritz {} is not implemented", enum2sv(meta.optRitz));
        }
    };

    if(results.size() >= 2) std::sort(results.begin(), results.end(), comparator);
    for(const auto &mps : results) elog.eigs_add_entry(mps, spdlog::level::debug);
    return results.front();
}


