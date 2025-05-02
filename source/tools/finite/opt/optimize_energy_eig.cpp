#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/num.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"

namespace tools::finite::opt::internal {
    template<typename CalcType, typename Scalar>
    void optimize_energy_eig_executor(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, std::vector<opt_mps<Scalar>> &results,
                                      const OptMeta &meta) {
        eig::solver solver;
        auto        matrix = tensors.template get_effective_hamiltonian<CalcType>();
        int         SR_il  = 1; // min nev index (starts from 1)
        int         SR_iu  = 1; // max nev index
        int         LR_il  = safe_cast<int>(matrix.dimension(0));
        int         LR_iu  = safe_cast<int>(matrix.dimension(0));
        switch(meta.optRitz) {
            case OptRitz::NONE: throw std::logic_error("optimize_energy_eig_executor: Invalid: OptRitz::NONE");
            case OptRitz::SR: solver.eig(matrix.data(), matrix.dimension(0), 'I', SR_il, SR_iu, 0.0, 1.0); break;
            case OptRitz::LR: solver.eig(matrix.data(), matrix.dimension(0), 'I', LR_il, LR_iu, 0.0, 1.0); break;
            case OptRitz::LM: solver.eig(matrix.data(), matrix.dimension(0)); break; // Find all eigenvalues
            case OptRitz::SM:
            case OptRitz::TE:
            case OptRitz::IS: {
                // Find all eigenvalues within a thin energy band
                auto eigval = initial_mps.get_energy(); // The current energy
                auto eigvar = initial_mps.get_variance();
                auto eshift = initial_mps.get_eshift();                          // The energy shift is our target energy for excited states
                auto vl     = eshift - std::abs(eigval) - 2 * std::sqrt(eigvar); // Find energies at most two sigma away from the band
                auto vu     = eshift + std::abs(eigval) + 2 * std::sqrt(eigvar); // Find energies at most two sigma away from the band
                solver.eig(matrix.data(), matrix.dimension(0), 'V', 1, 1, vl, vu);
                // tools::log->info("optimize_energy_eig_executor: vl {:.3e} | vu {:.3e}", vl, vu);
                // Filter the results
                if(solver.result.meta.eigvals_found and solver.result.meta.eigvecsR_found) {
                    // tools::log->info("Found {} eigvals ({} converged)", solver.result.meta.nev, solver.result.meta.nev_converged);
                    auto eigvals = eig::view::get_eigvals<RealScalar<Scalar>>(solver.result, false);
                    auto indices = num::range<long>(0l, eigvals.size());
                    auto eigComp = EigIdxComparator(meta.optRitz, eigval, eigvals.data(), eigvals.size());
                    std::sort(indices.begin(), indices.end(), eigComp);               // Should sort them according to distance from eigval
                    indices.resize(safe_cast<size_t>(std::min(eigvals.size(), 10l))); // We only need the first few indices, say 4
                    // for(auto idx : indices) { tools::log->info(" -- idx {}: {:.16f}", idx, eigvals(idx)); }
                    extract_results<CalcType>(tensors, initial_mps, meta, solver, results, false, indices);
                }
                return;
            }
        }

        extract_results<CalcType>(tensors, initial_mps, meta, solver, results, false);
    }
    /* clang-format off */
    template void optimize_energy_eig_executor<fp64>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, std::vector<opt_mps<fp64>> &results, const OptMeta &meta);
    template void optimize_energy_eig_executor<fp64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, std::vector<opt_mps<cx64>> &results, const OptMeta &meta);
    template void optimize_energy_eig_executor<cx64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, std::vector<opt_mps<cx64>> &results, const OptMeta &meta);
    /* clang-format on */

    template<typename Scalar>
    opt_mps<Scalar> optimize_energy_eig(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                        [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<Scalar> &elog) {
        if constexpr(tenx::sfinae::is_quadruple_prec_v<Scalar> or tenx::sfinae::is_single_prec_v<Scalar>) {
            throw except::runtime_error("optimize_energy_eig(): not implemented for type {}", enum2sv(meta.optType));
        }
        if(meta.optAlgo != OptAlgo::DMRG)
            throw except::logic_error("optimize_energy_eig: Expected OptAlgo [{}] | Got [{}]", enum2sv(OptAlgo::DMRG), enum2sv(meta.optAlgo));

        const auto problem_size = tensors.active_problem_size();
        if(problem_size > settings::precision::eig_max_size)
            throw except::logic_error("optimize_energy_eig: the problem size is too large for eig: {} > {}(max)", problem_size,
                                      settings::precision::eig_max_size);

        tools::log->debug("optimize_energy_eig: ritz {} | type {} | algo {}", enum2sv(meta.optRitz), enum2sv(meta.optType), enum2sv(meta.optAlgo));

        initial_mps.validate_initial_mps();
        // if(not tensors.model->is_shifted()) throw std::runtime_error("optimize_variance_eigs requires energy-shifted MPO²");
        elog.eigs_add_entry(initial_mps, spdlog::level::debug);

        auto                         t_gs = tid::tic_scope("eig-ene", tid::level::higher);
        std::vector<opt_mps<Scalar>> results;
        if constexpr(sfinae::is_std_complex_v<Scalar>) {
            switch(meta.optType) {
                case OptType::FP64: optimize_energy_eig_executor<fp64>(tensors, initial_mps, results, meta); break;
                case OptType::CX64: optimize_energy_eig_executor<cx64>(tensors, initial_mps, results, meta); break;
                default: throw except::runtime_error("optimize_energy_eig(): not implemented for type {}", enum2sv(meta.optType));
            }
        } else {
            switch(meta.optType) {
                case OptType::FP64: optimize_energy_eig_executor<fp64>(tensors, initial_mps, results, meta); break;
                case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
                default: throw except::runtime_error("optimize_energy_eig(): not implemented for type {}", enum2sv(meta.optType));
            }
        }

        if(results.empty()) {
            meta.optExit = OptExit::FAIL_ERROR;
            return initial_mps; // The solver failed
        }
        // Smallest energy wins (because they are shifted!)
        tools::log->debug("Sorting eigenpairs | initial energy {}", fp(initial_mps.get_energy()));
        if(results.size() >= 2) std::sort(results.begin(), results.end(), Comparator<Scalar>(meta, initial_mps.get_energy()));
        for(const auto &mps : results) elog.eigs_add_entry(mps, spdlog::level::debug);
        return results.front();
    }
    /* clang-format off */
    template opt_mps<fp32>  internal::optimize_energy_eig(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp32> &elog);
    template opt_mps<fp64>  internal::optimize_energy_eig(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp64> &elog);
    template opt_mps<fp128> internal::optimize_energy_eig(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp128> &elog);
    template opt_mps<cx32>  internal::optimize_energy_eig(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx32> &elog);
    template opt_mps<cx64>  internal::optimize_energy_eig(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx64> &elog);
    template opt_mps<cx128> internal::optimize_energy_eig(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx128> &elog);
    /* clang-format on */
}
