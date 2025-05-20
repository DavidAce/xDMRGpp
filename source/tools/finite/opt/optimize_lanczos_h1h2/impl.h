#pragma once
#include "../../opt_meta.h"
#include "../../opt_mps.h"
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "general/iter.h"
#include "general/sfinae.h"
// #include "GeneralizedLanczos.h"
#include "io/fmt_f128_t.h"
// #include "KrylovDualOp.h"
// #include "KrylovSingleOp.h"
#include "BlockLanczos.h"
#include "LOBPCG.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
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
#include "math/linalg.h"
#include "math/tenx.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include <Eigen/Eigenvalues>
#include <h5pp/h5pp.h>
#include <ranges>
using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

namespace lanczos_h1h2 {

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
    Eigen::Tensor<CalcType, 3> get_initial_guess(const opt_mps<Scalar> &initial_mps, const std::vector<opt_mps<Scalar>> &results) {
        if(results.empty()) {
            return initial_mps.template get_tensor_as<CalcType>();
        } else {
            // Return whichever of initial_mps or results that has the lowest variance
            auto it = std::min_element(results.begin(), results.end(), internal::comparator::variance);
            if(it == results.end()) return get_initial_guess<Scalar>(initial_mps, {});

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

}

template<typename MatrixT>
void do_dgks_sweep(MatrixT &Q, Eigen::Index i) {
    using Scalar = typename MatrixT::Scalar;
    // using RealScalar = decltype(std::real(std::declval<Scalar>()));
    // DGKS re-orthogonalization on Q.col(i)
    for(Eigen::Index j = 0; j < i; ++j) { Q.col(i).noalias() -= Q.col(j).dot(Q.col(i)) * Q.col(j); }
    // Second sweep: mop up the rounding residues
    for(Eigen::Index j = 0; j < i; ++j) { Q.col(i).noalias() -= Q.col(j).dot(Q.col(i)) * Q.col(j); }
    // Update the norm
    // RealScalar colNorm = Q.col(i).norm();
    // Q.col(i).noalias() /= colNorm; // Renormalize}
    // assert(colNorm != 0);
    // assert(Q.col(i).allFinite());
    // return colNorm;
}

template<typename MatVecT, typename MatrixT, typename VectorT, typename RealT>
void do_lanczos_orthonormalization(const MatVecT &H, MatrixT &V, VectorT &alpha, VectorT &beta, Eigen::Index &colIdx, Eigen::Index nCols, RealT normTol) {
    assert(colIdx + nCols <= V.cols());
    auto colMax = colIdx + nCols;
    assert(colMax <= V.cols());

    while(colIdx < colMax) {
        Eigen::Index i = colIdx;
        // Three-term Lanczos recurrence:
        V.col(i + 1) = H * V.col(i);
        alpha[i]     = V.col(i).dot(V.col(i + 1));
        if(i > 0) { V.col(i + 1).noalias() -= beta[i] * V.col(i - 1); }
        V.col(i + 1).noalias() -= alpha[i] * V.col(i);

        do_dgks_sweep(V, i + 1);

        beta[i + 1] = V.col(i + 1).norm();
        if(beta[i + 1] < normTol) {
            V.col(i + 1).setZero();
            break; // happy breakdown
        }
        V.col(i + 1) /= beta[i + 1];
        colIdx++;
    }
}

template<typename CalcType, typename Scalar>
opt_mps<Scalar> eigs_lanczos_h1h2(const opt_mps<Scalar>                      &initial,  //
                                  [[maybe_unused]] const StateFinite<Scalar> &state,    //
                                  const ModelFinite<Scalar>                  &model,    //
                                  const EdgesFinite<Scalar>                  &edges,    //
                                  OptMeta                                    &opt_meta, //
                                  reports::eigs_log<Scalar>                  &elog) {
    using RealScalar = tools::finite::opt::RealScalar<CalcType>;
    // using MatrixCT          = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorCR          = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto           t_mixblk = tid::tic_scope("eigs-h1h2");
    auto          &sites    = initial.get_sites();
    auto           mpos     = model.get_mpo(sites);
    auto           enve     = edges.get_multisite_env_ene(sites);
    auto           envv     = edges.get_multisite_env_var(sites);
    auto           size     = initial.get_tensor().size();
    constexpr auto eps      = std::numeric_limits<RealScalar>::epsilon();
    Eigen::Index   nev      = opt_meta.eigs_nev.value_or(1);
    Eigen::Index   ncv      = opt_meta.eigs_ncv.value_or(2 * nev);
    if(ncv <= 0) ncv = std::max<Eigen::Index>(2 * nev, safe_cast<Eigen::Index>(std::ceil(std::log2(size))));

    auto H1 = MatVecMPOS<CalcType>(mpos, enve);
    auto H2 = MatVecMPOS<CalcType>(mpos, envv);
    // BlockLanczos<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), mpos, enve, envv);
    LOBPCG<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2);
    solver.b              = 1;
    solver.status.initVal = initial.get_energy();
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_max);
    solver.tol            = opt_meta.eigs_tol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_tol.value()) //
                                                          : eps * 10000;
    solver.max_matvecs    = opt_meta.eigs_iter_max.value_or(500);

    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.set_jcbMaxBlockSize(1024);
    solver.set_chebyshevFilterDegree(1);
    solver.set_ResidualHistoryLength(4);
    solver.use_refined_rayleigh_ritz = false;
    solver.run();

    tools::log->debug("KrylovDualOp: status.exit = {}", solver.status.exitMsg);
    // Extract solution
    opt_mps<Scalar> res;
    res.is_basis_vector = false;
    res.set_name(fmt::format("eigenvector 0 [h1h2 lobpcg]"));
    res.set_tensor(Eigen::TensorMap<Eigen::Tensor<CalcType, 3>>(solver.V.col(0).data(), solver.mps_shape));
    res.set_overlap(std::abs(initial.get_vector().dot(res.get_vector())));
    res.set_sites(initial.get_sites());
    res.set_eshift(initial.get_eshift()); // Will set energy if also given the eigval
    res.set_eigs_idx(0);
    res.set_eigs_nev(1);
    res.set_eigs_ncv(ncv);
    res.set_eigs_tol(solver.tol);
    res.set_eigs_ritz(enum2sv(opt_meta.optRitz));
    res.set_optalgo(opt_meta.optAlgo);
    res.set_optsolver(opt_meta.optSolver);
    res.set_energy_shifted(initial.get_energy_shifted());

    res.set_length(initial.get_length());
    res.set_time(t_mixblk->get_last_interval());
    res.set_time_mv(solver.H1.t_multAx->get_time() + solver.H2.t_multAx->get_time());
    res.set_time_pc(solver.H1.t_multPc->get_time() + solver.H2.t_multPc->get_time());
    res.set_op(solver.H1.num_op + solver.H2.num_op);
    res.set_mv(solver.status.num_matvecs);
    res.set_pc(solver.status.num_precond);
    res.set_iter(solver.status.iter);
    res.set_eigs_rnorm(solver.status.rNorms(0));
    res.set_eigs_eigval(static_cast<fp64>(solver.status.optVal[0]));

    auto vh1v    = tools::finite::measure::expval_hamiltonian(res.get_tensor(), mpos, enve);
    auto vh2v    = tools::finite::measure::expval_hamiltonian_squared(res.get_tensor(), mpos, envv);
    auto rnormH1 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, enve);
    auto rnormH2 = tools::finite::measure::residual_norm(res.get_tensor(), mpos, envv);
    res.set_rnorm_H1(rnormH1);
    res.set_rnorm_H2(rnormH2);
    res.set_energy(std::real(vh1v + res.get_eshift()));
    res.set_variance(std::real(vh2v) - std::abs(vh1v * vh1v));
    res.set_energy_shifted(std::real(vh1v));
    res.set_hsquared(std::real(vh2v));

    // tools::log->info("lancsoz {}: {:.34f} [{}] | ⟨H⟩ {:.16f} | ⟨H²⟩ {:.16f} | ⟨H²⟩-⟨H⟩² {:.4e} | sites {} (size {}) | rnorm {:.3e} | ngs {} | iters "
    // "{} | {:.3e} s | {} | var {:.4e}",
    // sfinae::type_name<CalcType>(), fp(optVal), optIdx, fp(res.get_energy()), fp(res.get_hsquared()), fp(res.get_variance()), sites,
    // mps_size, fp(rnorm), ngs, iter, t_mixblk->get_last_interval(), exit_msg, fp(vh2v - vh1v * vh1v));
    elog.eigs_add_entry(res, spdlog::level::debug);
    return res;
}

template<typename Scalar>
[[nodiscard]] opt_mps<Scalar> tools::finite::opt::internal::optimize_lanczos_h1h2(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial,
                                                                                  [[maybe_unused]] const AlgorithmStatus &status, OptMeta &meta,
                                                                                  reports::eigs_log<Scalar> &elog) {
    using namespace internal;
    using namespace settings::precision;
    initial.validate_initial_mps();
    elog.eigs_add_entry(initial, spdlog::level::debug);

    auto token = tid::tic_scope(fmt::format("h1h2-{}", enum2sv(meta.optAlgo)), tid::level::higher);

    std::string eigprob;
    switch(meta.optAlgo) {
        case OptAlgo::DMRG: eigprob = "Hx=λx"; break;
        case OptAlgo::DMRGX: eigprob = "Hx=λx"; break;
        case OptAlgo::HYBRID_DMRGX: eigprob = "Hx=λx"; break;
        case OptAlgo::XDMRG: eigprob = "H²x=λx"; break;
        case OptAlgo::GDMRG: eigprob = "Hx=λH²x"; break;
    }

    tools::log->debug("eigs_lanczos_h1h2_executor: Solving [{}] | ritz {} | maxIter {} | tol {:.2e} | init on | size {} | mps {}", eigprob,
                      enum2sv(meta.optRitz), meta.eigs_iter_max, meta.eigs_tol, initial.get_tensor().size(), initial.get_tensor().dimensions());
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        switch(meta.optType) {
            case OptType::FP32: return eigs_lanczos_h1h2<fp32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::FP64: return eigs_lanczos_h1h2<fp64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::FP128: return eigs_lanczos_h1h2<fp128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::CX32: return eigs_lanczos_h1h2<cx32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::CX64: return eigs_lanczos_h1h2<cx64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::CX128: return eigs_lanczos_h1h2<cx128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            default: throw std::runtime_error("unrecognized option type");
        }
    } else {
        switch(meta.optType) {
            case OptType::FP32: return eigs_lanczos_h1h2<fp32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::FP64: return eigs_lanczos_h1h2<fp64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::FP128: return eigs_lanczos_h1h2<fp128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta, elog);
            case OptType::CX32: throw except::logic_error("Cannot run OptType::CX32 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX64: throw except::logic_error("Cannot run OptType::CX64 with Scalar type {}", sfinae::type_name<Scalar>());
            case OptType::CX128: throw except::logic_error("Cannot run OptType::CX128 with Scalar type {}", sfinae::type_name<Scalar>());
            default: throw std::runtime_error("unrecognized option type");
        }
    }
}

//
//
// template<typename Scalar>
// void do_lanczos_step(Krylov<Scalar> &lanczos) {
//     using MatrixType = typename Krylov<Scalar>::MatrixType;
//     using VectorType = typename Krylov<Scalar>::VectorType;
//     using VectorReal = typename Krylov<Scalar>::VectorReal;
//     using RealScalar = typename Krylov<Scalar>::RealScalar;
//     using VectorIdxT = typename Krylov<Scalar>::VectorIdxT;
//
//     auto &V            = lanczos.V;
//     auto &H1V          = lanczos.H1V;
//     auto &H2V          = lanczos.H2V;
//     auto &K1           = lanczos.K1;
//     auto &K2           = lanczos.K2;
//     auto &K1_on        = lanczos.K1_on;
//     auto &K2_on        = lanczos.K2_on;
//     auto &krylov_evals = lanczos.krylov_evals;
//     auto &krylov_evecs = lanczos.krylov_evecs;
//     auto &eps          = lanczos.eps;
//
//     const auto &absDiffTol = lanczos.absDiffTol;
//     const auto &relDiffTol = lanczos.relDiffTol;
//     const auto &max_iters  = lanczos.max_iters;
//     const auto &H1         = lanczos.H1;
//     const auto &H2         = lanczos.H2;
//     const auto &ncv        = lanczos.ncv;
//     const auto &algo       = lanczos.algo;
//     const auto &ritz       = lanczos.ritz;
//
//     auto &optVal      = lanczos.status.optVal;
//     auto &oldVal      = lanczos.status.oldVal;
//     auto &absDiff     = lanczos.status.absDiff;
//     auto &relDiff     = lanczos.status.relDiff;
//     auto &optIdx      = lanczos.status.optIdx;
//     auto &rnorm       = lanczos.status.rnorm;
//     auto &maxEval     = lanczos.status.maxEval;
//     auto &nonZeroCols = lanczos.status.nonZeroCols;
//     auto &iter        = lanczos.status.iter;
//     auto &numMGS      = lanczos.status.numMGS;
//     auto &exitMsg     = lanczos.status.exitMsg;
//     auto &exit        = lanczos.status.exit;
//
//     std::vector<long> mixedColOk;
//
//     // Define the krylov subspace
//     // After this, note that
//     //      V.col(1)     = H1*V.col(0),
//     //      V.col(ncv/2+1) = H2*V.col(0),
//     // Where V.col(0) holds the current estimate of the eigenvector (when iter == 0, V.col(0) = V0)
//     // These columns are useful for calculating the residual norms later.
//     auto t_dotprod = tid::tic_scope("dotprod");
//     V.col(0).normalize();
//     for(long i = 0; i + 1 < ncv; ++i) {
//         if(i < ncv / 2) {
//             H1.MultAx(V.col(i).data(), V.col(i + 1).data()); // V0, H1*V0, H1*H1*V0 ... ncv/2 times
//         } else if(i == ncv / 2) {
//             H2.MultAx(V.col(0).data(), V.col(i + 1).data()); // H2*V0,
//         } else {
//             H2.MultAx(V.col(i).data(), V.col(i + 1).data()); // H2*H2*V0 ... up to ncv
//         }
//         V.col(i + 1).normalize();
//     }
//     t_dotprod.toc();
//
//     // Calculate the current residual norms
//     if(!std::isnan(optVal)) {
//         auto v   = V.col(0);
//         auto h1v = V.col(1);
//         auto h2v = V.col(1 + ncv / 2);
//         if(algo == OptAlgo::DMRG)
//             rnorm = (h1v - optVal * v).template lpNorm<Eigen::Infinity>();
//         else if(algo == OptAlgo::GDMRG)
//             rnorm = (h1v - optVal * h2v).template lpNorm<Eigen::Infinity>();
//         else
//             rnorm = (h2v - optVal * v).template lpNorm<Eigen::Infinity>();
//     }
//
//     if(iter >= 1ul) {
//         if(absDiff < absDiffTol and iter >= 3) {
//             exitMsg.emplace_back(std::format("saturated: abs diff {:.3e} < tol {:.3e}", std::abs(oldVal - optVal), absDiffTol));
//             exit |= SolverExit::saturated_absDiffTol;
//         }
//         if(relDiff < relDiffTol and iter >= 3) {
//             exitMsg.emplace_back(std::format("saturated: rel diff {:.3e} < {:.3e}", relDiff, relDiffTol));
//             exit |= SolverExit::saturated_relDiffTol;
//         }
//
//         if(mixedColOk.size() == 1) {
//             exitMsg.emplace_back(fmt::format("saturated: only one valid eigenvector"));
//             exit |= SolverExit::one_valid_eigenvector;
//         }
//
//         if(mixedColOk.empty()) {
//             exitMsg.emplace_back(fmt::format("mixedColOk is empty"));
//             exit |= SolverExit::no_valid_eigenvector;
//         }
//     }
//     tools::log->info("ncv {} (nnz {}) | rnorm = {:.8e} | optVal = {:.8e} | optIdx = {} | absDiff = {:.8e} | relDiff = {:.8e} | iter = {} | exitMsg = {}",
//     ncv,
//                      nonZeroCols.size(), rnorm, optVal, optIdx, absDiff, relDiff, iter, exitMsg);
//
//     if(iter >= std::max<size_t>(1ul, max_iters)) {
//         exitMsg.emplace_back(fmt::format("iter ({}) >= maxiter ({})", iter, max_iters));
//         exit |= SolverExit::max_iterations;
//     }
//     if(rnorm < lanczos.rnormTol()) {
//         exitMsg.emplace_back(std::format("converged rnorm {:.3e} < tol {:.3e}", rnorm, lanczos.rnormTol()));
//         exit |= SolverExit::converged_rnormTol;
//     }
//     if(exit != SolverExit::ok) { return; }
//
//     // Orthonormalize with Modified Gram Schmidt
//     auto t_mgs = tid::tic_token("mgs");
//     // tools::log->debug("V = \n{}\n", linalg::matrix::to_string(V, 16));
//     for(size_t igs = 0; igs <= 5; ++igs) {
//         auto mgs    = linalg::matrix::modified_gram_schmidt_colpiv_dgks(V);
//         nonZeroCols = std::move(mgs.nonZeroCols);
//         V           = std::move(mgs.Q);
//         numMGS++;
//         // tools::log->debug("V = \n{}\n", linalg::matrix::to_string(V, 16));
//         // tools::log->debug("nonZeroCols = {}", nonZeroCols);
//         // tools::log->debug("Rdiag       = {}", mgs.Rdiag);
//         assert(V.allFinite());
//         if(nonZeroCols.size() == safe_cast<size_t>(mgs.nCols)) break;
//     }
//     t_mgs.toc();
//
//     // V should now have orthonormal vectors
//     t_dotprod.tic();
//     if(K1_on) {
//         for(auto &i : nonZeroCols) H1.MultAx(V.col(i).data(), H1V.col(i).data());
//         assert(H1V(Eigen::all, nonZeroCols).allFinite());
//     }
//     if(K2_on) {
//         for(auto &i : nonZeroCols) H2.MultAx(V.col(i).data(), H2V.col(i).data());
//         assert(H2V(Eigen::all, nonZeroCols).allFinite());
//     }
//
//     if(K1_on) {
//         K1.resize(nonZeroCols.size(), nonZeroCols.size());
//         for(auto [J, j] : iter::enumerate(nonZeroCols)) {
//             for(auto [I, i] : iter::enumerate(nonZeroCols)) {
//                 if(i < j) continue;
//                 K1(I, J) = V.col(i).dot(H1V.col(j));
//             }
//         }
//         K1 = K1.template selfadjointView<Eigen::Lower>();
//     }
//
//     if(K2_on) {
//         K2.resize(nonZeroCols.size(), nonZeroCols.size());
//         for(auto [J, j] : iter::enumerate(nonZeroCols)) {
//             for(auto [I, i] : iter::enumerate(nonZeroCols)) {
//                 if(i < j) continue;
//                 if(i == j)
//                     K2(I, J) = std::abs(V.col(i).dot(H2V.col(j))); // Use abs to avoid negative near-zero values
//                 else
//                     K2(I, J) = V.col(i).dot(H2V.col(j));
//             }
//         }
//         K2 = K2.template selfadjointView<Eigen::Lower>();
//     }
//
//     t_dotprod.toc();
//     auto t_eigsol      = tid::tic_scope("eigsol");
//     long numZeroRowsK1 = K1_on ? (K1.cwiseAbs().rowwise().maxCoeff().array() <= eps).count() : 0l;
//     long numZeroRowsK2 = K2_on ? (K2.cwiseAbs().rowwise().maxCoeff().array() <= eps).count() : 0l;
//     long numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
//     // VectorCR evals; // Eigen::VectorXd ::Zero();
//     // MatrixCT evecs; // Eigen::MatrixXcd::Zero();
//     OptRitz ritz_internal = ritz;
//     auto    solver        = eig::solver();
//     switch(algo) {
//         using enum OptAlgo;
//         case DMRG: {
//             solver.eig<eig::Form::SYMM>(K1.data(), K1.rows(), eig::Vecs::ON);
//
//             // solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K1, Eigen::ComputeEigenvectors);
//             // if(solver.info() == Eigen::ComputationInfo::Success) {
//             //     evals = solver.eigenvalues();
//             //     evecs = solver.eigenvectors();
//             // } else {
//             //     tools::log->info("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
//             //     tools::log->warn("Diagonalization of K1 exited with info {}", static_cast<int>(solver.info()));
//             // }
//             //
//             // if(evals.hasNaN()) throw except::runtime_error("found nan eigenvalues");
//             break;
//         }
//         case DMRGX: [[fallthrough]];
//         case HYBRID_DMRGX: {
//             MatrixType K = K2 - K1 * K1;
//             solver.eig<eig::Form::SYMM>(K.data(), K.rows(), eig::Vecs::ON);
//             // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
//             // evals       = solver.eigenvalues();
//             // evecs       = solver.eigenvectors();
//             break;
//         }
//         case XDMRG: {
//             solver.eig<eig::Form::SYMM>(K2.data(), K2.rows(), eig::Vecs::ON);
//             // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2, Eigen::ComputeEigenvectors);
//             // evals       = solver.eigenvalues();
//             // evecs       = solver.eigenvectors();
//             break;
//         }
//         case GDMRG: {
//             if(numZeroRows == 0) {
//                 solver.eig<eig::Form::SYMM>(K1.data(), K2.data(), K1.rows(), eig::Vecs::ON);
//
//                 // auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixCT>(
//                 // K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
//                 // evals = solver.eigenvalues().real();
//                 // evecs = solver.eigenvectors().colwise().normalized();
//             } else {
//                 MatrixType K = K2 - K1 * K1;
//                 tools::log->debug("K                      : \n{}\n", linalg::matrix::to_string(K1, 8));
//                 solver.eig<eig::Form::SYMM>(K.data(), K.rows(), eig::Vecs::ON);
//                 // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
//                 // evals       = solver.eigenvalues();
//                 // evecs       = solver.eigenvectors();
//                 if(ritz == OptRitz::LM) ritz_internal = OptRitz::SM;
//                 if(ritz == OptRitz::LR) ritz_internal = OptRitz::SM;
//                 if(ritz == OptRitz::SM) ritz_internal = OptRitz::LM;
//                 if(ritz == OptRitz::SR) ritz_internal = OptRitz::LR;
//             }
//             break;
//         }
//         default: throw except::runtime_error("unhandled algorithm: [{}]", enum2sv(algo));
//     }
//
//     krylov_evals = eig::view::get_eigvals<RealScalar>(solver.result);
//     krylov_evecs = eig::view::get_eigvecs<Scalar>(solver.result).colwise().normalized();
//
//     maxEval                    = static_cast<RealScalar>(krylov_evals.cwiseAbs().maxCoeff());
//     V(Eigen::all, nonZeroCols) = (V(Eigen::all, nonZeroCols) * krylov_evecs.real()).eval(); // Now V has columns mixed according to evecs
//
//     VectorReal mixedNorms = V.colwise().norm(); // New state norms after mixing cols of V according to cols of evecs
//     mixedColOk.clear();                         // New states with acceptable norm and eigenvalue
//     mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
//     auto mixedColNormTol = eps * 10000;
//     for(long i = 0; i < mixedNorms.size(); ++i) {
//         if(std::abs(mixedNorms(i) - RealScalar{1}) > mixedColNormTol) continue;
//         mixedColOk.emplace_back(i);
//     }
//     if(mixedColOk.size() <= 1) {
//         // tools::log->debug("K1 (modified by eig)   : \n{}\n", linalg::matrix::to_string(K1, 8));
//         // tools::log->debug("K2 (modified by eig)   : \n{}\n", linalg::matrix::to_string(K2, 8));
//         tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(krylov_evals, 8));
//         tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(krylov_evecs, 8));
//         // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
//         tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
//         tools::log->debug("mixedColOk             = {}", mixedColOk);
//         tools::log->debug("numZeroRowsK1          = {}", numZeroRowsK1);
//         tools::log->debug("numZeroRowsK2          = {}", numZeroRowsK2);
//         tools::log->debug("nonZeroCols            = {}", nonZeroCols);
//         tools::log->debug("ngramSchmidt           = {}", numMGS);
//     }
//
//     // Eigenvalues are sorted in ascending order.
//
//     long colIdx = 0;
//     switch(ritz_internal) {
//         case OptRitz::SR: krylov_evals(mixedColOk).minCoeff(&colIdx); break;
//         case OptRitz::LR: krylov_evals(mixedColOk).maxCoeff(&colIdx); break;
//         case OptRitz::SM: krylov_evals(mixedColOk).cwiseAbs().minCoeff(&colIdx); break;
//         case OptRitz::LM: krylov_evals(mixedColOk).cwiseAbs().maxCoeff(&colIdx); break;
//         case OptRitz::IS: [[fallthrough]];
//         case OptRitz::TE: [[fallthrough]];
//         case OptRitz::NONE: {
//             if(std::isnan(lanczos.status.initVal))
//                 throw except::runtime_error("Ritz [{}] does not work when lanczos.status.initVal is nan", enum2sv(ritz_internal));
//             (krylov_evals(mixedColOk).array() - lanczos.status.initVal).cwiseAbs().minCoeff(&colIdx);
//         }
//     }
//     optIdx = mixedColOk[colIdx];
//     if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others in the next step)
//
//     oldVal  = optVal;
//     optVal  = krylov_evals(optIdx);
//     absDiff = std::abs(oldVal - optVal);
//     relDiff = absDiff / (static_cast<RealScalar>(0.5) * (std::abs(optVal) + std::abs(oldVal)));
//     iter++;
// }