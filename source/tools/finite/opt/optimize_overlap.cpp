#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

/*
 * Overlap optimization
 *
 * In overlap optimization we consider a local set of sites l of the L-site system,
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
 * After having found a good subspace, the objective is to find the eigenvector that has
 * the highest overlap with the current state.
 *
 * Some observations: Let {x} be the set of all eigenvectors
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
 * Overlap optimization steps
 *
 * Step 1)  Find a subspace [x], i.e. take a set of k eigenvectors of the local effective Hamiltonian.
 *          Empirically, eigenvectors whose eigenvalues (energy) are closest to the current energy,
 *          tend to have nonzero overlap with the current vector |y⟩.
 *          On an iterative solver we keep increasing "nev" (number of requested eigenvectors) until
 *          the subspace error ε is small enough, or an eigenstate is found with overlap > 0.5.
 *
 * Step 2) Return the eigenstate with the highest overlap to the current state.
 */
opt_mps tools::finite::opt::internal::optimize_overlap(const TensorsFinite &tensors, const opt_mps &initial_mps, [[maybe_unused]] const AlgorithmStatus &status,
                                                       OptMeta &meta) {
    if(meta.optAlgo != OptAlgo::DMRGX)
        throw except::runtime_error("optimize_overlap: Expected OptAlgo [{}] | Got [{}]", enum2sv(OptAlgo::DMRGX), enum2sv(meta.optAlgo));
    tools::log->trace("Optimizing in OVERLAP mode");
    auto t_olap = tid::tic_scope("overlap", tid::level::higher);
    initial_mps.validate_initial_mps();

    /*
     *  Step 1) Find the subspace.
     *  The subspace set of eigenstates obtained from full or partial diagonalization
     */

    std::vector<opt_mps> subspace;
    switch(meta.optType) {
        case OptType::FP64: subspace = internal::subspace::find_subspace<fp64>(tensors, meta); break;
        case OptType::CX64: subspace = internal::subspace::find_subspace<cx64>(tensors, meta); break;
        default: throw except::runtime_error("optimize_overlap(): not implemented for type {}", enum2sv(meta.optType));
    }

    tools::log->trace("optimize_overlap: subspace found with {} eigenvectors", subspace.size());

    /*
     *  Step 2) Return the eigenvector with the highest overlap, or the current one if none is found.
     */

    auto max_overlap_idx = internal::subspace::get_idx_to_eigvec_with_highest_overlap(subspace);
    if(max_overlap_idx) {
        auto &eigvec_max_overlap = *std::next(subspace.begin(), safe_cast<long>(max_overlap_idx.value()));
        eigvec_max_overlap.set_variance(tools::finite::measure::energy_variance(eigvec_max_overlap.get_tensor(), tensors, std::nullopt));
        if(tools::log->level() == spdlog::level::trace) {
            tools::log->trace("optimize_overlap: eigvec {:<2} has highest overlap {:.16f} | energy {:>20.16f} | variance {:>8.2e}", max_overlap_idx.value(),
                              eigvec_max_overlap.get_overlap(), eigvec_max_overlap.get_energy(), eigvec_max_overlap.get_variance());
        }
        if(eigvec_max_overlap.get_overlap() < 0.1) tools::log->debug("optimize_overlap: Overlap fell below < 0.1: {:20.16f}", eigvec_max_overlap.get_overlap());
        return eigvec_max_overlap;
    } else {
        tools::log->warn("optimize_overlap: No overlapping states in energy range. Max overlap has energy. Returning the state unchanged.");
        return initial_mps;
    }
}