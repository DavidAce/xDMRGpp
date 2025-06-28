#pragma once
#include "../../opt_meta.h"
#include "../../opt_mps.h"
#include "BlockLanczos.h"
#include "config/settings.h"
#include "GD.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "io/fmt_f128_t.h"
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


template<typename CalcType, typename Scalar>
opt_mps<Scalar> eigs_lanczos_h1h2(const opt_mps<Scalar>                      &initial,                         //
                                  [[maybe_unused]] const StateFinite<Scalar> &state,                           //
                                  const ModelFinite<Scalar>                  &model,                           //
                                  const EdgesFinite<Scalar>                  &edges,                           //
                                  OptMeta                                    &opt_meta,                        //
                                  reports::eigs_log<Scalar>                  &elog,                            //
                                  Eigen::Index                                jcb,                             //
                                  eig::Preconditioner                         preconditioner_type,             //
                                  ResidualCorrectionType                      rct,                             //
                                  bool                                        use_coarse_inner_preconditioner, //
                                  std::string_view                            tag) {
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
    Eigen::Index   nev      = opt_meta.eigs_nev.value_or(settings::precision::eigs_nev_min);
    Eigen::Index   ncv      = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(ncv <= 0) {
        // Automatic selection
        Eigen::Index ncv_by_size = safe_cast<Eigen::Index>(std::ceil(std::log2(size)));
        Eigen::Index ncv_min     = std::max<Eigen::Index>(2 * nev, settings::precision::eigs_ncv_min);
        Eigen::Index ncv_max     = settings::precision::eigs_ncv_max <= 0 ? ncv_by_size : static_cast<Eigen::Index>(settings::precision::eigs_ncv_max);
        ncv                      = std::clamp(ncv_by_size, ncv_min, ncv_max);
        tools::log->trace("ncv automatic selection: {} (min {}, max {})", ncv, ncv_min, ncv_max);
    }

    auto H1 = MatVecMPOS<CalcType>(mpos, enve);
    auto H2 = MatVecMPOS<CalcType>(mpos, envv);
    // BlockLanczos<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), mpos, enve, envv);
    GD<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2);
    solver.tol = opt_meta.eigs_tol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_tol.value()) //
                                               : eps * 10000;
    eig::setLevel(spdlog::level::info);
    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.b              = opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    solver.status.initVal = static_cast<RealScalar>(initial.get_energy());
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.max_matvecs    = -1ul; // opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.set_jcbMaxBlockSize(jcb);
    solver.set_chebyshevFilterDegree(0);
    solver.set_chebyshevFilterLambdaCutBias(0.1f);
    solver.set_chebyshevFilterRelGapThreshold(1e-3f);
    solver.set_maxBasisBlocks(ncv / solver.b);
    solver.set_maxRetainBlocks(3 * ncv / solver.b / 8);
    solver.set_maxLanczosResidualHistory(0);
    solver.set_maxRitzResidualHistory(1);
    solver.set_maxExtraRitzHistory(1);
    solver.set_preconditioner_type(preconditioner_type);
    solver.use_refined_rayleigh_ritz       = true;
    solver.use_relative_rnorm_tolerance    = true;
    solver.use_adaptive_inner_tolerance    = true;
    solver.use_coarse_inner_preconditioner = use_coarse_inner_preconditioner;
    solver.residual_correction_type        = rct;
    solver.inject_randomness               = false;

    solver.run();

    tools::log->debug("GD+k: status.stopReason = {}", solver.status.stopMessage);
    // Extract solution
    opt_mps<Scalar> res;
    res.is_basis_vector = false;
    res.set_name(fmt::format("eigenvector 0 [{} H1H2 {}]", enum2sv(opt_meta.optAlgo), tag));
    res.set_tensor(Eigen::TensorMap<Eigen::Tensor<CalcType, 3>>(solver.V.col(0).data(), solver.mps_shape));
    res.set_overlap(std::abs(initial.get_vector().dot(res.get_vector())));
    res.set_sites(initial.get_sites());
    res.set_eshift(initial.get_eshift()); // Will set energy if also given the eigval
    res.set_eigs_idx(0);
    res.set_eigs_nev(1);
    res.set_eigs_ncv(ncv);
    res.set_eigs_tol(solver.tol);
    res.set_eigs_jcb(solver.get_jcbMaxBlockSize());
    res.set_eigs_ritz(enum2sv(opt_meta.optRitz));
    res.set_eigs_type(enum2sv(opt_meta.optType));
    res.set_optalgo(opt_meta.optAlgo);
    res.set_opttype(opt_meta.optType);
    res.set_optsolver(opt_meta.optSolver);
    res.set_energy_shifted(initial.get_energy_shifted());

    res.set_length(initial.get_length());
    res.set_time(t_mixblk->get_last_interval());
    res.set_time_mv(solver.status.time_matvecs_total.get_time());
    res.set_time_pc(solver.status.time_precond_total.get_time());
    res.set_op(solver.H1.num_op + solver.H2.num_op);
    res.set_mv(solver.status.num_matvecs_total);
    res.set_pc(solver.status.num_precond_total);
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

    elog.eigs_add_entry(res, spdlog::level::debug);
    return res;
}

template<typename CalcType, typename Scalar>
opt_mps<Scalar> eigs_lanczos_h1h2(const opt_mps<Scalar>                      &initial,  //
                                  [[maybe_unused]] const StateFinite<Scalar> &state,    //
                                  const ModelFinite<Scalar>                  &model,    //
                                  const EdgesFinite<Scalar>                  &edges,    //
                                  OptMeta                                    &opt_meta, //
                                  reports::eigs_log<Scalar>                  &elog) {
    auto jcb = opt_meta.eigs_jcbMaxBlockSize.value_or(settings::precision::eigs_jcb_blocksize_max);
    auto prt = eig::StringToPreconditioner(opt_meta.eigs_preconditioner_type.value_or("SOLVE"));
    auto rct = StringToResidualCorrection(opt_meta.eigs_residual_correction_type.value_or("JACOBI_DAVIDSON"));
    bool crs = opt_meta.eigs_use_coarse_inner_preconditioner.value_or(false);
    if(jcb <= 1) return eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, prt, rct, crs, "");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 0, ResidualCorrectionType::NONE, "NO 0");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, ResidualCorrectionType::NONE, "NO 1");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 256, ResidualCorrectionType::NONE, "NO 256");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 512, ResidualCorrectionType::NONE, "NO 512");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1024, ResidualCorrectionType::NONE, "NO 1024");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 0, ResidualCorrectionType::CHEAP_OLSEN, "CO 0");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, ResidualCorrectionType::CHEAP_OLSEN, "CO 1");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 256, ResidualCorrectionType::CHEAP_OLSEN, "CO 256");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 512, ResidualCorrectionType::CHEAP_OLSEN, "CO 512");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1024, ResidualCorrectionType::CHEAP_OLSEN, "CO 1024");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 0, ResidualCorrectionType::FULL_OLSEN, "FO 0");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, ResidualCorrectionType::FULL_OLSEN, "FO 1");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 256, ResidualCorrectionType::FULL_OLSEN, "FO 256");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 512, ResidualCorrectionType::FULL_OLSEN, "FO 512");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1024, ResidualCorrectionType::FULL_OLSEN, "FO 1024");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 0, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 0");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, eig::Preconditioner::SOLVE, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, eig::Preconditioner::JACOBI, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, eig::Preconditioner::SOLVE, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, eig::Preconditioner::JACOBI, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 256, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 256");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 512, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 512");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, 1, eig::Preconditioner::SOLVE, ResidualCorrectionType::JACOBI_DAVIDSON, "JD");
    // eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, false, "CO");
    eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, true, "CO (c)");
    return eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, prt, rct, crs, "JD");
    // return eigs_lanczos_h1h2<CalcType>(initial, state, model, edges, opt_meta, elog, jcb, prt, rct, true, "JD (c)");
}

template<typename Scalar>
[[nodiscard]] opt_mps<Scalar> tools::finite::opt::internal::optimize_lanczos_h1h2(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial,
                                                                                  [[maybe_unused]] OptMeta &meta, reports::eigs_log<Scalar> &elog) {
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

