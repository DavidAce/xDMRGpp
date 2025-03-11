#include "../opt_meta.h"
#include "../opt_mps.h"
#include "config/settings.h"
#include "math/eig.h"
#include "math/linalg/tensor.h"
#include "math/num.h"
#include "math/tenx.h"
#include "measure/MeasurementsTensorsFinite.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/opt/opt-internal.h"
#include <fmt/ranges.h>
#include <tensors/model/ModelFinite.h>

void tools::finite::opt::internal::extract_results(const TensorsFinite &tensors, const opt_mps &initial_mps, const OptMeta &meta, const eig::solver &solver,
                                                   std::vector<opt_mps> &results, bool converged_only, std::optional<std::vector<long>> indices) {
    auto t_ext    = tid::tic_scope("extract");
    auto dims_mps = initial_mps.get_tensor().dimensions();
    if(solver.result.meta.eigvals_found and solver.result.meta.eigvecsR_found) {
        auto eigvecs = eig::view::get_eigvecs<cx64>(solver.result, eig::Side::R, converged_only);
        auto eigvals = eig::view::get_eigvals<fp64>(solver.result, converged_only);

        if(eigvecs.cols() == eigvals.size()) /* Checks if eigenvectors converged for each eigenvalue */ {
            [[maybe_unused]] double overlap_sq_sum = 0;
            [[maybe_unused]] size_t num_solutions  = 0;
            // Eigenvalues are normally sorted small to large, so we reverse when looking for large.
            for(const auto &idx : indices.value_or(num::range<long>(0, eigvals.size()))) {
                if(idx >= eigvals.size()) throw except::logic_error("idx ({}) >= eigvals.size() ({})", idx, eigvals.size());
                auto udx = safe_cast<size_t>(idx);
                results.emplace_back(opt_mps());
                auto &mps           = results.back();
                mps.is_basis_vector = true;
                if constexpr(settings::debug) tools::log->trace("Extracting result: idx {} | eigval {:.16f}", idx, eigvals(idx));
                mps.set_name(fmt::format("eigenvector {} [{:^8}]", idx, solver.config.tag));
                mps.set_tensor(eigvecs.col(idx).normalized(), dims_mps); // eigvecs are not always well normalized when we get them from eig::solver
                mps.set_sites(initial_mps.get_sites());
                mps.set_eshift(initial_mps.get_eshift()); // Will set energy if also given the eigval
                mps.set_overlap(std::abs(initial_mps.get_vector().dot(mps.get_vector())));
                mps.set_length(initial_mps.get_length());
                mps.set_time(solver.result.meta.time_total);
                mps.set_time_mv(solver.result.meta.time_mv);
                mps.set_time_pc(solver.result.meta.time_pc);
                mps.set_op(safe_cast<size_t>(solver.result.meta.num_op));
                mps.set_mv(safe_cast<size_t>(solver.result.meta.num_mv));
                mps.set_pc(safe_cast<size_t>(solver.result.meta.num_pc));
                mps.set_iter(safe_cast<size_t>(solver.result.meta.iter));
                mps.set_eigs_idx(idx);
                mps.set_eigs_nev(solver.result.meta.nev_converged);
                mps.set_eigs_ncv(solver.result.meta.ncv);
                mps.set_eigs_tol(solver.result.meta.tol);
                mps.set_eigs_ritz(solver.result.meta.ritz);
                mps.set_eigs_shift(solver.result.meta.sigma);
                mps.set_optalgo(meta.optAlgo);
                mps.set_optsolver(meta.optSolver);
                if(solver.result.meta.residual_norms.size() > udx) mps.set_eigs_rnorm(solver.result.meta.residual_norms.at(udx));
                auto mpos    = tensors.get_model().get_mpo_active();
                auto enve    = tensors.get_edges().get_ene_active();
                auto envv    = tensors.get_edges().get_var_active();
                auto vh1v    = tools::finite::measure::expval_hamiltonian(mps.get_tensor(), mpos, enve);
                auto vh2v    = tools::finite::measure::expval_hamiltonian_squared(mps.get_tensor(), mpos, envv);
                auto rnormH  = tools::finite::measure::residual_norm(mps.get_tensor(), mpos, enve);
                auto rnormH2 = tools::finite::measure::residual_norm(mps.get_tensor(), mpos, envv);
                mps.set_rnorm_H1(rnormH);
                mps.set_rnorm_H2(rnormH2);

                // When using PRIMME_DYNAMIC with nev > 1, I have noticed that the eigenvalues are sometimes repeated,
                // so it looks like an exact degeneracy. This is probably a bug somewhere (maybe in PRIMME).
                mps.set_eigs_eigval(eigvals[idx]);

                double energy   = std::real(vh1v) + tensors.get_model().get_energy_shift_mpo();
                double variance = std::real(vh2v) - std::abs(vh1v * vh1v);

                mps.set_energy(energy);
                mps.set_energy_shifted(std::real(vh1v));
                mps.set_hsquared(std::real(vh2v));
                mps.set_variance(variance);

                mps.validate_basis_vector();

                // Sum up the contributions. Since full diag gives an orthonormal basis, this adds up to one. Normally only
                // a few eigenvectors contribute to most of the sum.
                overlap_sq_sum += mps.get_overlap() * mps.get_overlap();
                num_solutions++; // Count the number of solutions added
            }
        }
    }
}

void tools::finite::opt::internal::extract_results_subspace(const TensorsFinite &tensors, const opt_mps &initial_mps, const OptMeta &meta,
                                                            const eig::solver &solver, const std::vector<opt_mps> &subspace_mps,
                                                            std::vector<opt_mps> &results) {
    auto t_ext    = tid::tic_scope("extract");
    auto dims_mps = initial_mps.get_tensor().dimensions();
    if(solver.result.meta.eigvals_found and solver.result.meta.eigvecsR_found) {
        auto eigvecs = eig::view::get_eigvecs<cx64>(solver.result, eig::Side::R);
        auto eigvals = eig::view::get_eigvals<fp64>(solver.result);
        if(eigvecs.cols() == eigvals.size()) /* Checks if eigenvectors converged for each eigenvalue */ {
            auto indices = num::range<long>(0, eigvals.size());
            // Eigenvalues are normally sorted small to large, so we reverse when looking for large.
            if(meta.optRitz == OptRitz::LR) std::reverse(indices.begin(), indices.end());
            for(const auto &idx : indices) {
                results.emplace_back(opt_mps());
                auto  udx           = safe_cast<size_t>(idx);
                auto &mps           = results.back();
                mps.is_basis_vector = false;
                mps.set_name(fmt::format("eigenvector {} [{:^8}]", idx, solver.config.tag));
                // eigvecs are not always well normalized when we get them from eig::solver
                mps.set_tensor(subspace::get_vector_in_fullspace(subspace_mps, eigvecs.col(idx).normalized()), dims_mps);
                mps.set_sites(initial_mps.get_sites());
                mps.set_eshift(initial_mps.get_eshift()); // Will set energy if also given the eigval
                mps.set_overlap(std::abs(initial_mps.get_vector().dot(mps.get_vector())));
                mps.set_length(initial_mps.get_length());
                mps.set_time(solver.result.meta.time_total);
                mps.set_time_mv(solver.result.meta.time_mv);
                mps.set_time_pc(solver.result.meta.time_pc);
                mps.set_op(safe_cast<size_t>(solver.result.meta.num_op));
                mps.set_mv(safe_cast<size_t>(solver.result.meta.num_mv));
                mps.set_pc(safe_cast<size_t>(solver.result.meta.num_pc));
                mps.set_iter(safe_cast<size_t>(solver.result.meta.iter));
                mps.set_eigs_idx(idx);
                mps.set_eigs_nev(solver.result.meta.nev_converged);
                mps.set_eigs_ncv(solver.result.meta.ncv);
                mps.set_eigs_tol(solver.result.meta.tol);
                mps.set_eigs_eigval(eigvals[idx]);
                mps.set_eigs_ritz(solver.result.meta.ritz);
                mps.set_eigs_shift(solver.result.meta.sigma);
                mps.set_optalgo(meta.optAlgo);
                mps.set_optsolver(meta.optSolver);

                if(solver.result.meta.residual_norms.size() > udx) mps.set_eigs_rnorm(solver.result.meta.residual_norms.at(udx));
                auto mpos    = tensors.get_model().get_mpo_active();
                auto enve    = tensors.get_edges().get_ene_active();
                auto envv    = tensors.get_edges().get_var_active();
                auto vh1v    = tools::finite::measure::expval_hamiltonian(mps.get_tensor(), mpos, enve);
                auto vh2v    = tools::finite::measure::expval_hamiltonian_squared(mps.get_tensor(), mpos, envv);
                auto rnormH1 = tools::finite::measure::residual_norm(mps.get_tensor(), mpos, enve);
                auto rnormH2 = tools::finite::measure::residual_norm(mps.get_tensor(), mpos, envv);
                mps.set_rnorm_H1(rnormH1);
                mps.set_rnorm_H2(rnormH2);

                // When using PRIMME_DYNAMIC with nev > 1, I have noticed that the eigenvalues are sometimes repeated,
                // so it looks like an exact degeneracy. This is probably a bug somewhere (maybe in PRIMME).
                mps.set_eigs_eigval(eigvals[idx]);

                double energy   = std::real(vh1v) + tensors.get_model().get_energy_shift_mpo();
                double variance = std::real(vh2v) - std::abs(vh1v * vh1v);

                mps.set_energy(energy);
                mps.set_energy_shifted(std::real(vh1v));
                mps.set_hsquared(std::real(vh2v));
                mps.set_variance(variance);

                // tools::log->info("extract_results_subspace: set variance: {:.16f}", variance);
                //                mps.set_grad_max(grad_max);
                mps.validate_basis_vector();
            }
        }
    }
}
