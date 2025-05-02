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
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/opt/opt-internal.h"
#include <fmt/ranges.h>
#include <tensors/model/ModelFinite.h>

template<typename CalcType, typename Scalar>
void tools::finite::opt::internal::extract_results(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                                   const eig::solver &solver, std::vector<opt_mps<Scalar>> &results, bool converged_only,
                                                   std::optional<std::vector<long>> indices) {
    auto t_ext    = tid::tic_scope("extract");
    auto dims_mps = initial_mps.get_tensor().dimensions();
    if(solver.result.meta.eigvals_found and solver.result.meta.eigvecsR_found) {
        using Real     = decltype(std::real(std::declval<Scalar>()));
        using RealCalc = decltype(std::real(std::declval<CalcType>()));
        auto eigvals   = tenx::asScalarType<Real>(eig::view::get_eigvals<RealCalc>(solver.result, converged_only));
        auto eigvecs   = tenx::asScalarType<Scalar>(eig::view::get_eigvecs<CalcType>(solver.result, eig::Side::R, converged_only));

        if(eigvecs.cols() == eigvals.size()) /* Checks if eigenvectors converged for each eigenvalue */ {
            [[maybe_unused]] Real   overlap_sq_sum = 0;
            [[maybe_unused]] size_t num_solutions  = 0;
            // Eigenvalues are normally sorted small to large, so we reverse when looking for large.
            for(const auto &idx : indices.value_or(num::range<long>(0, eigvals.size()))) {
                if(idx >= eigvals.size()) throw except::logic_error("idx ({}) >= eigvals.size() ({})", idx, eigvals.size());
                auto udx = safe_cast<size_t>(idx);
                results.emplace_back(opt_mps<Scalar>());
                auto &res           = results.back();
                res.is_basis_vector = true;
                if constexpr(settings::debug) tools::log->trace("Extracting result: idx {} | eigval {:.16f}", idx, fp(eigvals(idx)));
                res.set_name(fmt::format("eigenvector {} [{:^8}]", idx, solver.config.tag));
                res.set_tensor(eigvecs.col(idx).normalized(), dims_mps); // eigvecs are not always well normalized when we get them from eig::solver
                res.set_sites(initial_mps.get_sites());
                res.set_eshift(initial_mps.get_eshift()); // Will set energy if also given the eigval
                res.set_overlap(std::abs(initial_mps.get_vector().dot(res.get_vector())));
                res.set_length(initial_mps.get_length());
                res.set_time(solver.result.meta.time_total);
                res.set_time_mv(solver.result.meta.time_mv);
                res.set_time_pc(solver.result.meta.time_pc);
                res.set_op(safe_cast<size_t>(solver.result.meta.num_op));
                res.set_mv(safe_cast<size_t>(solver.result.meta.num_mv));
                res.set_pc(safe_cast<size_t>(solver.result.meta.num_pc));
                res.set_iter(safe_cast<size_t>(solver.result.meta.iter));
                res.set_eigs_idx(idx);
                res.set_eigs_nev(solver.result.meta.nev_converged);
                res.set_eigs_ncv(solver.result.meta.ncv);
                res.set_eigs_tol(solver.result.meta.tol);
                res.set_eigs_ritz(solver.result.meta.ritz);
                res.set_eigs_shift(solver.result.meta.sigma);
                res.set_optalgo(meta.optAlgo);
                res.set_optsolver(meta.optSolver);
                if(solver.result.meta.residual_norms.size() > udx) res.set_eigs_rnorm(solver.result.meta.residual_norms.at(udx));
                auto mpos    = tensors.get_model().get_mpo_active();
                auto enve    = tensors.get_edges().get_ene_active();
                auto envv    = tensors.get_edges().get_var_active();
                auto vh1v    = tools::finite::measure::expval_hamiltonian(res.get_tensor(), mpos, enve);
                auto vh2v    = tools::finite::measure::expval_hamiltonian_squared(res.get_tensor(), mpos, envv);
                auto rnormH1 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, enve);
                auto rnormH2 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, envv);
                res.set_rnorm_H1(rnormH1);
                res.set_rnorm_H2(rnormH2);

                // When using PRIMME_DYNAMIC with nev > 1, I have noticed that the eigenvalues are sometimes repeated,
                // so it looks like an exact degeneracy. This is probably a bug somewhere (maybe in PRIMME).
                res.set_eigs_eigval(eigvals[idx]);

                Real energy   = std::real(vh1v + tensors.get_model().get_energy_shift_mpo());
                Real variance = std::real(vh2v) - std::abs(vh1v * vh1v);

                res.set_energy(energy);
                res.set_energy_shifted(std::real(vh1v));
                res.set_hsquared(std::real(vh2v));
                res.set_variance(variance);

                res.validate_basis_vector();

                // Sum up the contributions. Since full diag gives an orthonormal basis, this adds up to one. Normally only
                // a few eigenvectors contribute to most of the sum.
                overlap_sq_sum += res.get_overlap() * res.get_overlap();
                num_solutions++; // Count the number of solutions added
            }
        }
    }
}
/* clang-format off */
template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp32>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp64>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<fp128>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx32>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx64>> &results, bool converged_only, std::optional<std::vector<long>> indices);

template void tools::finite::opt::internal::extract_results<fp32>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp64>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<fp128>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx32>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx64>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
template void tools::finite::opt::internal::extract_results<cx128>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, std::vector<opt_mps<cx128>> &results, bool converged_only, std::optional<std::vector<long>> indices);
/* clang-format on */

template<typename CalcType, typename Scalar>
void tools::finite::opt::internal::extract_results_subspace(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                                            const eig::solver &solver, const std::vector<opt_mps<Scalar>> &subspace_mps,
                                                            std::vector<opt_mps<Scalar>> &results) {
    auto t_ext    = tid::tic_scope("extract");
    auto dims_mps = initial_mps.get_tensor().dimensions();
    if(solver.result.meta.eigvals_found and solver.result.meta.eigvecsR_found) {
        using Real     = decltype(std::real(std::declval<Scalar>()));
        using RealCalc = decltype(std::real(std::declval<CalcType>()));
        auto eigvals   = tenx::asScalarType<Real>(eig::view::get_eigvals<RealCalc>(solver.result));
        auto eigvecs   = tenx::asScalarType<Scalar>(eig::view::get_eigvecs<CalcType>(solver.result, eig::Side::R));

        if(eigvecs.cols() == eigvals.size()) /* Checks if eigenvectors converged for each eigenvalue */ {
            auto indices = num::range<long>(0, eigvals.size());
            // Eigenvalues are normally sorted small to large, so we reverse when looking for large.
            if(meta.optRitz == OptRitz::LR) std::reverse(indices.begin(), indices.end());
            for(const auto &idx : indices) {
                results.emplace_back(opt_mps<Scalar>());
                auto  udx           = safe_cast<size_t>(idx);
                auto &res           = results.back();
                res.is_basis_vector = false;
                res.set_name(fmt::format("eigenvector {} [{:^8}]", idx, solver.config.tag));
                // eigvecs are not always well normalized when we get them from eig::solver
                res.set_tensor(subspace::get_vector_in_fullspace(subspace_mps, eigvecs.col(idx).normalized()), dims_mps);
                res.set_sites(initial_mps.get_sites());
                res.set_eshift(initial_mps.get_eshift()); // Will set energy if also given the eigval
                res.set_overlap(std::abs(initial_mps.get_vector().dot(res.get_vector())));
                res.set_length(initial_mps.get_length());
                res.set_time(solver.result.meta.time_total);
                res.set_time_mv(solver.result.meta.time_mv);
                res.set_time_pc(solver.result.meta.time_pc);
                res.set_op(safe_cast<size_t>(solver.result.meta.num_op));
                res.set_mv(safe_cast<size_t>(solver.result.meta.num_mv));
                res.set_pc(safe_cast<size_t>(solver.result.meta.num_pc));
                res.set_iter(safe_cast<size_t>(solver.result.meta.iter));
                res.set_eigs_idx(idx);
                res.set_eigs_nev(solver.result.meta.nev_converged);
                res.set_eigs_ncv(solver.result.meta.ncv);
                res.set_eigs_tol(solver.result.meta.tol);
                res.set_eigs_eigval(eigvals[idx]);
                res.set_eigs_ritz(solver.result.meta.ritz);
                res.set_eigs_shift(solver.result.meta.sigma);
                res.set_optalgo(meta.optAlgo);
                res.set_optsolver(meta.optSolver);

                if(solver.result.meta.residual_norms.size() > udx) res.set_eigs_rnorm(solver.result.meta.residual_norms.at(udx));
                auto mpos    = tensors.get_model().get_mpo_active();
                auto enve    = tensors.get_edges().get_ene_active();
                auto envv    = tensors.get_edges().get_var_active();
                auto vh1v    = tools::finite::measure::expval_hamiltonian(res.get_tensor(), mpos, enve);
                auto vh2v    = tools::finite::measure::expval_hamiltonian_squared(res.get_tensor(), mpos, envv);
                auto rnormH1 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, enve);
                auto rnormH2 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, envv);
                res.set_rnorm_H1(rnormH1);
                res.set_rnorm_H2(rnormH2);

                // When using PRIMME_DYNAMIC with nev > 1, I have noticed that the eigenvalues are sometimes repeated,
                // so it looks like an exact degeneracy. This is probably a bug somewhere (maybe in PRIMME).
                res.set_eigs_eigval(eigvals[idx]);

                Real energy   = std::real(vh1v + tensors.get_model().get_energy_shift_mpo());
                Real variance = std::real(vh2v) - std::abs(vh1v * vh1v);

                res.set_energy(energy);
                res.set_energy_shifted(std::real(vh1v));
                res.set_hsquared(std::real(vh2v));
                res.set_variance(variance);

                // tools::log->info("extract_results_subspace: set variance: {:.16f}", variance);
                //                mps.set_grad_max(grad_max);
                res.validate_basis_vector();
            }
        }
    }
}
/* clang-format off */
template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<fp32> &tensors, const opt_mps<fp32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp32>> &subspace_mps, std::vector<opt_mps<fp32>> &results);

template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<fp64> &tensors, const opt_mps<fp64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp64>> &subspace_mps, std::vector<opt_mps<fp64>> &results);

template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<fp128> &tensors, const opt_mps<fp128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<fp128>> &subspace_mps, std::vector<opt_mps<fp128>> &results);

template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<cx32> &tensors, const opt_mps<cx32> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx32>> &subspace_mps, std::vector<opt_mps<cx32>> &results);

template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<cx64> &tensors, const opt_mps<cx64> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx64>> &subspace_mps, std::vector<opt_mps<cx64>> &results);

template void tools::finite::opt::internal::extract_results_subspace<fp32>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp64>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<fp128>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx32>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx64>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
template void tools::finite::opt::internal::extract_results_subspace<cx128>(const TensorsFinite<cx128> &tensors, const opt_mps<cx128> &initial_mps, const OptMeta &meta, const eig::solver &solver, const std::vector<opt_mps<cx128>> &subspace_mps, std::vector<opt_mps<cx128>> &results);
/* clang-format on */
