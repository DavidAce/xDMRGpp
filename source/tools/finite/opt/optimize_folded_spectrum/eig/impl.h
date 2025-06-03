#pragma once
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig.h"
#include "math/num.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

template<typename CalcType, typename Scalar>
void optimize_folded_spectrum_eig_executor(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results,
                                           OptMeta &meta) {
    // Solve the folded spectrum problem H²v=E²x (H² is positive definite)
    if(meta.optRitz == OptRitz::NONE) return;
    eig::solver solver;
    using R = decltype(std::real(std::declval<CalcType>()));
    auto        matrix = tensors.template get_effective_hamiltonian_squared<CalcType>();
    auto        nev    = std::min<int>(static_cast<int>(matrix.dimension(0)), meta.eigs_nev.value_or(1));
    auto        il     = 1;
    auto        iu     = nev;
    switch(meta.optRitz) {
        case OptRitz::LR: [[fallthrough]];
        case OptRitz::LM: {
            il = static_cast<int>(matrix.dimension(0) - (nev - 1));
            iu = static_cast<int>(matrix.dimension(0));
            break;
        }
        default: break;
    }
    solver.eig(matrix.data(), matrix.dimension(0), 'I', il, iu, R{0}, R{1});
    extract_results<CalcType>(tensors, initial_mps, meta, solver, results, true);

}

template<typename Scalar>
opt_mps<Scalar> tools::finite::opt::internal::optimize_folded_spectrum_eig(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                             [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<Scalar> &elog) {
    if constexpr(tenx::sfinae::is_quadruple_prec_v<Scalar> or tenx::sfinae::is_single_prec_v<Scalar>) {
        throw except::runtime_error("optimize_folded_spectrum_eig(): not implemented for type {}", enum2sv(meta.optType));
    }
    if(meta.optSolver == OptSolver::EIGS) return optimize_folded_spectrum(tensors, initial_mps, status, meta, elog);

    initial_mps.validate_initial_mps();

    const auto problem_size = tensors.active_problem_size();
    if(problem_size > settings::precision::eig_max_size)
        throw except::logic_error("optimize_folded_spectrum_eig: the problem size is too large for eig: {}", problem_size);

    tools::log->debug("optimize_folded_spectrum_eig: ritz {} | type {} | algo {}", enum2sv(meta.optRitz), enum2sv(meta.optType), enum2sv(meta.optAlgo));

    elog.eigs_add_entry(initial_mps, spdlog::level::debug);
    auto                         t_var = tid::tic_scope("eig-xdmrg", tid::level::higher);
    std::vector<opt_mps<Scalar>> results;
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP32: optimize_folded_spectrum_eig_executor<fp32>(tensors, initial_mps, results, meta); break;
            case OptType::FP64: optimize_folded_spectrum_eig_executor<fp64>(tensors, initial_mps, results, meta); break;
            case OptType::FP128: optimize_folded_spectrum_eig_executor<fp128>(tensors, initial_mps, results, meta); break;
            case OptType::CX32: optimize_folded_spectrum_eig_executor<cx32>(tensors, initial_mps, results, meta); break;
            case OptType::CX64: optimize_folded_spectrum_eig_executor<cx64>(tensors, initial_mps, results, meta); break;
            case OptType::CX128: optimize_folded_spectrum_eig_executor<cx128>(tensors, initial_mps, results, meta); break;
            default: throw except::runtime_error("optimize_folded_spectrum_eig(): not implemented for type {}", enum2sv(meta.optType));
        }
    } else {
        switch(meta.optType) {
            case OptType::FP32: optimize_folded_spectrum_eig_executor<fp32>(tensors, initial_mps, results, meta); break;
            case OptType::FP64: optimize_folded_spectrum_eig_executor<fp64>(tensors, initial_mps, results, meta); break;
            case OptType::FP128: optimize_folded_spectrum_eig_executor<fp128>(tensors, initial_mps, results, meta); break;
            case OptType::CX32:  throw except::logic_error("Cannot run OptType::CX32 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX64:  throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX128: throw except::logic_error("Cannot run OptType::CX128 with Scalar type {}", sfinae::type_name<Scalar>());
            default: throw except::runtime_error("optimize_folded_spectrum_eig(): not implemented for type {}", enum2sv(meta.optType));
        }
    }

    auto t_post = tid::tic_scope("post");
    if(results.empty()) {
        meta.optExit = OptExit::FAIL_ERROR;
        return initial_mps; // The solver failed
    }

    if(results.size() >= 2) {
        std::sort(results.begin(), results.end(), Comparator<Scalar>(meta)); // Smallest eigenvalue (i.e. variance) wins
    }

    for(const auto &mps : results) elog.eigs_add_entry(mps, spdlog::level::debug);
    return results.front();
}
