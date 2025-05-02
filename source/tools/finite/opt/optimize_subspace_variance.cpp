#include "math/tenx.h"
// -- (textra first)
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

namespace settings {
    [[maybe_unused]] static constexpr bool debug_subspace = false;
}

/*
 * Subspace optimization
 *
 *
 * In subspace optimization we consider a local set of sites l of the L-site system,
 * usually this corresponds to l = 1 or up to l = 8 adjacent sites.
 *
 * A subspace in this context is a truncated basis, i.e. a small subset of eigenvectors
 * of the local "effective" Hamiltonian. The subspace is a set of k eigenvectors [x]
 * which have significant overlap with the current state |y⟩, i.e. we define k such that
 *
 *          ε > 1 - Σ_i^k |⟨x_i|y⟩|²,
 *
 * where ε is a small number that controls error of the truncation. A value ε ~1e-10 is
 * reasonable. Note that the truncation implies that k is smaller than the local
 * Hilbert space dimension.
 *
 * After having found a good subspace, the objective is to find a linear combination
 * of eigenvectors which minimizes the energy variance.
 *
 * It is worth noting some observations. Let {x} be the set of all eigenvectors
 * to the local effective Hamiltonian H_local.
 * Then, when the DMRG process is fully converged:
 *      - only one x_i has overlap ⟨x_i|y⟩ = 1
 *        Since the sum of all overlaps must add to 1, the rest have <x_j|y> = 0 when i != j.
 *      - This x is also the one that minimizes the energy variance.
 *
 * However, before the DMRG process has converged this is not true. Instead:
 *      - we have ⟨x_i|y⟩ > 0 for several i.
 *      - a linear combination of several x can have lower variance than any
 *        single x.
 *
 * Fully diagonalizing H_local yields all K eigenvectors {x}, but if H_local is too big this operation
 * becomes prohibitively expensive. Instead, we resort to finding a subset with k << K eigenvectors [x],
 * whose eigenvalues are the k energies closest to the current energy. Usually the eigenvectors
 * which have some overlap ⟨x_i|y⟩ > 0 are found in the subset [x] if k is large enough.
 *
 * Subspace optimization steps
 *
 * Step 1)  Find a subspace [x], i.e. take a set of k eigenvectors of the local effective Hamiltonian.
 *          Empirically, eigenvectors whose eigenvalues (energy) are closest to the current energy,
 *          tend to have nonzero overlap with the current vector |y⟩.
 *          On an iterative solver we keep increasing "nev" (number of requested eigenvectors) until
 *          the subspace error ε is small enough.
 *          If any eigenvectors have to removed, (e.g. due to memory/performance constraints),
 *          then sort the eigenvectors in order of decreasing overlap ⟨x_i|y⟩, and start deleting
 *          from the end.
 *
 * Step 2)  Project the squared effective K*K Hamiltonian, down to the k*k subspace, H².
 *          Using BFGS, find the linear combination |w⟩ of eigenvectors that minimizes the variance.
 *
 *              min_w Var H = ⟨H²⟩ - ⟨H⟩² = ⟨w|H²|w⟩ - ⟨E⟩²
 *
 *          where E are the energy eigenvalues from step 1.
 *
 */
template<typename Scalar>
opt_mps<Scalar> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps,
                                                                         [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta,
                                                                         reports::eigs_log<Scalar> &elog) {
    tools::log->trace("Optimizing subspace");
    auto t_sub = tid::tic_scope("subspace");
    initial_mps.validate_initial_mps();
    if(meta.optAlgo != OptAlgo::HYBRID_DMRGX)
        throw except::runtime_error("optimize_subspace_variance: Expected OptAlgo [{}]. Got [{}]", enum2sv(OptAlgo::HYBRID_DMRGX), enum2sv(meta.optAlgo));

    // Handy references
    const auto &model = tensors.get_model();
    const auto &edges = tensors.get_edges();

    /*
     *  Step 1) Find the subspace.
     *  The subspace is a set of eigenstates obtained from full or partial diagonalization
     */

    std::vector<opt_mps<Scalar>> subspace;
    auto                         slog = reports::subs_log<Scalar>();
    if constexpr(sfinae::is_any_v<Scalar, fp32, cx32>) {
        switch(meta.optType) {
            case OptType::FP32: subspace = internal::subspace::find_subspace<fp32>(tensors, meta, slog); break;
            case OptType::CX32: subspace = internal::subspace::find_subspace<cx32>(tensors, meta, slog); break;
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }
    if constexpr(sfinae::is_any_v<Scalar, fp64, cx64>) {
        switch(meta.optType) {
            case OptType::FP64: subspace = internal::subspace::find_subspace<fp64>(tensors, meta, slog); break;
            case OptType::CX64: subspace = internal::subspace::find_subspace<cx64>(tensors, meta, slog); break;
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }
    if constexpr(sfinae::is_any_v<Scalar, fp128, cx128>) {
        switch(meta.optType) {
            case OptType::FP128: subspace = internal::subspace::find_subspace<fp128>(tensors, meta, slog); break;
            case OptType::CX128: subspace = internal::subspace::find_subspace<cx128>(tensors, meta, slog); break;
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }

    tools::log->trace("Subspace found with {} eigenvectors", subspace.size());
    slog.print_subs_report();

    if(subspace.size() == 1) {
        auto &sub_mps      = subspace.front();
        auto  measurements = MeasurementsTensorsFinite<Scalar>();
        auto  sub_energy   = tools::finite::measure::energy(subspace.front().get_tensor(), model, edges, std::nullopt, &measurements);
        auto  sub_variance = tools::finite::measure::energy_variance(subspace.front().get_tensor(), model, edges, std::nullopt, &measurements);
        subspace.front().set_energy(sub_energy);
        subspace.front().set_variance(sub_variance);
        if(meta.optRitz == OptRitz::SM) {
            if(std::abs(sub_mps.get_energy()) < std::abs(initial_mps.get_energy())) {
                sub_mps.validate_result();
                return subspace.front();
            } else {
                // Use as initial guess
                meta.optRitz = OptRitz::SM;
                sub_mps.validate_initial_mps();
                return optimize_folded_spectrum(tensors, sub_mps, status, meta, elog);
            }
        }
    }

    /*
     * Filter the eigenvectors
     *
     */

    internal::subspace::filter_subspace(subspace, settings::precision::max_subspace_size);

    /*
     *
     * Step 2) Optimize variance in the subspace of k eigenvectors
     *
     */

    // Get the initial state expressed in the subspace
    auto subspace_vector = internal::subspace::get_vector_in_subspace<Scalar>(subspace, initial_mps.get_vector());

    tools::log->trace("Starting subspace optimization");
    // We need the eigenvalues in a convenient format as well
    auto eigvals = internal::subspace::get_eigvals<Scalar>(subspace);

    /*
     *
     *  Start the variance optimization process for the subspace.
     *
     */
    std::vector<opt_mps<Scalar>> results;
    eig::solver                  solver;

    auto t_eigs = tid::tic_scope("eigs");
    tools::log->trace("Optimizing subspace | {}", enum2sv(meta.optType));
    using Real = decltype(std::real(std::declval<Scalar>()));
    if constexpr(sfinae::is_any_v<Scalar, fp32, cx32>) {
        switch(meta.optType) {
            case OptType::FP32: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<fp32>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<fp32>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            case OptType::CX32: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<cx32>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<cx32>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }
    if constexpr(sfinae::is_any_v<Scalar, fp64, cx64>) {
        switch(meta.optType) {
            case OptType::FP64: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<fp64>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<fp64>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            case OptType::CX64: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<cx64>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<cx64>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }
    if constexpr(sfinae::is_any_v<Scalar, fp128, cx128>) {
        switch(meta.optType) {
            case OptType::FP128: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<fp128>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<fp128>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            case OptType::CX128: {
                auto H2_subspace = subspace::get_hamiltonian_squared_in_subspace<cx128>(model, edges, subspace);
                solver.eig<eig::Form::SYMM>(H2_subspace.data(), H2_subspace.rows(), 'I', 1, 1, Real{0}, Real{1});
                extract_results_subspace<cx128>(tensors, initial_mps, meta, solver, subspace, results);
                break;
            }
            default:
                throw except::runtime_error("optimize_subspace_variance<{}>(): not implemented for type {}", sfinae::type_name<Scalar>(),
                                            enum2sv(meta.optType));
        }
    }

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
/* clang-format off */
template opt_mps<fp32> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp32> &slog);
template opt_mps<fp64> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp64> &slog);
template opt_mps<fp128> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<fp128> &slog);
template opt_mps<cx32> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx32> &slog);
template opt_mps<cx64> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx64> &slog);
template opt_mps<cx128> tools::finite::opt::internal::optimize_subspace_variance(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta, reports::eigs_log<cx128> &slog);
/* clang-format on */
