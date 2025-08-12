#pragma once
#include "config/settings.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver_eigsmpo/solver_gdplusk.h"
#include "tools/finite/opt/opt-internal.h"

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> eigs_gdplusk(const opt_mps<Scalar>       &initial, //
                                          const TensorsFinite<Scalar> &tensors,
                                          const OptMeta               &opt_meta,                                //
                                          Eigen::Index                 jcb,                                     //
                                          eig::Preconditioner          preconditioner_type,                     //
                                          ResidualCorrectionType       rct,                                     //
                                          bool                         use_coarse_inner_preconditioner,         //
                                          bool                         use_rayleigh_quotients_instead_of_evals, //
                                          bool                         use_h2_inner_product,                    //
                                          bool                         use_h1h2_preconditioner,                 //
                                          bool                         skipjcb,                                 //
                                          bool                         dev_thick_jd_projector,                  //
                                          Eigen::Index                 block_size,
                                          Eigen::Index                 ncv, //
                                          std::string_view             tag, //
                                          reports::eigs_log<Scalar>   &elog) {
    using RealScalar = tools::finite::opt::RealScalar<CalcType>;
    // using MatrixCT          = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorCR          = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto           t_mixblk = tid::tic_scope("gdplusk");
    auto          &sites    = initial.get_sites();
    auto           mpos     = tensors.get_model().get_mpo(sites);
    auto           enve     = tensors.get_edges().get_multisite_env_ene(sites);
    auto           envv     = tensors.get_edges().get_multisite_env_var(sites);
    auto           size     = initial.get_tensor().size();
    constexpr auto eps      = std::numeric_limits<RealScalar>::epsilon();
    auto           nev      = opt_meta.eigs_nev.value_or(settings::precision::eigs_nev_min);
    // auto           ncv      = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(ncv <= 0) {
        // Automatic selection

        Eigen::Index ncv_by_size = safe_cast<Eigen::Index>(std::ceil(std::log2(size)));
        Eigen::Index ncv_min     = std::max<Eigen::Index>(2 * nev, settings::precision::eigs_ncv_min);
        Eigen::Index ncv_max     = settings::precision::eigs_ncv_max <= 0 ? ncv_by_size : static_cast<Eigen::Index>(settings::precision::eigs_ncv_max);
        ncv                      = std::clamp(ncv_by_size, ncv_min, ncv_max);
        tools::log->trace("ncv automatic selection: {} (min {}, max {})", ncv, ncv_min, ncv_max);
    }
    auto H1   = MatVecMPOS<CalcType>(mpos, enve);
    auto H2   = MatVecMPOS<CalcType>(mpos, envv);
    auto H1H2 = MatVecMPOS<CalcType>(mpos, enve, envv);
    // BlockLanczos<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), mpos, enve, envv);
    solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver.tol = opt_meta.eigs_tol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_tol.value()) //
                                               : eps * 10000;
    eig::setLevel(spdlog::level::info);
    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.setLogger(spdlog::level::debug, fmt::format("gd+k{}",tag));
    solver.b              = block_size; // opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    solver.status.initVal = static_cast<RealScalar>(initial.get_energy());
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.max_matvecs    = -1ul; // opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.set_jcbMaxBlockSize(jcb);
    solver.set_chebyshevFilterDegree(0);
    solver.set_chebyshevFilterLambdaCutBias(0.1f);
    solver.set_chebyshevFilterRelGapThreshold(1e-3f);
    solver.set_maxBasisBlocks(ncv / solver.b);
    solver.set_maxRetainBlocks(3 * ncv / solver.b / 8);
    // solver.set_maxLanczosResidualHistory(0);
    solver.set_maxRitzResidualHistory(1);
    solver.set_maxExtraRitzHistory(1);
    solver.set_preconditioner_type(preconditioner_type);
    solver.use_refined_rayleigh_ritz               = true;
    solver.use_relative_rnorm_tolerance            = true;
    solver.use_adaptive_inner_tolerance            = true;
    solver.use_coarse_inner_preconditioner         = use_coarse_inner_preconditioner;
    solver.use_rayleigh_quotients_instead_of_evals = use_rayleigh_quotients_instead_of_evals;
    solver.use_h2_inner_product                    = use_h2_inner_product;
    solver.use_h1h2_preconditioner                 = use_h1h2_preconditioner;
    solver.dev_skipjcb                             = skipjcb;
    solver.dev_thick_jd_projector                  = dev_thick_jd_projector;
    solver.residual_correction_type                = rct;
    solver.inject_randomness                       = false;
    solver.tag                                     = tag;

    solver.run();

    tools::log->debug("GD+k: status.stopReason = {}", solver.status.stopMessage);
    auto res = std::vector<opt_mps<Scalar>>();
    extract_results(tensors, initial, opt_meta, solver, res);

    elog.eigs_add_entry(res.front(), spdlog::level::debug);
    return res;
}

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> eigs_gdplusk(const TensorsFinite<Scalar> &tensors,  //
                                          const opt_mps<Scalar>       &initial,  //
                                          const OptMeta               &opt_meta, //
                                          reports::eigs_log<Scalar>   &elog) {
    auto         jcb        = opt_meta.eigs_jcbMaxBlockSize.value_or(settings::precision::eigs_jcb_blocksize_max);
    auto         prt        = eig::StringToPreconditioner(opt_meta.eigs_preconditioner_type.value_or("SOLVE"));
    auto         rct        = StringToResidualCorrection(opt_meta.eigs_residual_correction_type.value_or("JACOBI_DAVIDSON"));
    bool         crs        = opt_meta.eigs_use_coarse_inner_preconditioner.value_or(false);
    Eigen::Index block_size = opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    Eigen::Index ncv        = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(jcb == 1) return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, false, false, false, block_size, ncv, "", elog);
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 0, ResidualCorrectionType::NONE, "NO 0");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, ResidualCorrectionType::NONE, "NO 1");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 256, ResidualCorrectionType::NONE, "NO 256");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 512, ResidualCorrectionType::NONE, "NO 512");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1024, ResidualCorrectionType::NONE, "NO 1024");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 0, ResidualCorrectionType::CHEAP_OLSEN, "CO 0");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, ResidualCorrectionType::CHEAP_OLSEN, "CO 1");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 256, ResidualCorrectionType::CHEAP_OLSEN, "CO 256");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 512, ResidualCorrectionType::CHEAP_OLSEN, "CO 512");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1024, ResidualCorrectionType::CHEAP_OLSEN, "CO 1024");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 0, ResidualCorrectionType::FULL_OLSEN, "FO 0");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, ResidualCorrectionType::FULL_OLSEN, "FO 1");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 256, ResidualCorrectionType::FULL_OLSEN, "FO 256");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 512, ResidualCorrectionType::FULL_OLSEN, "FO 512");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1024, ResidualCorrectionType::FULL_OLSEN, "FO 1024");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 0, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 0");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, eig::Preconditioner::SOLVE, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, eig::Preconditioner::JACOBI, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, eig::Preconditioner::SOLVE, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, eig::Preconditioner::JACOBI, ResidualCorrectionType::CHEAP_OLSEN, "CO");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 256, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 256");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 512, ResidualCorrectionType::JACOBI_DAVIDSON, "JD 512");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, 1, eig::Preconditioner::SOLVE, ResidualCorrectionType::JACOBI_DAVIDSON, "JD");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, false, "CO");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, true, "CO (c)");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, eig::Preconditioner::SOLVE, ResidualCorrectionType::CHEAP_OLSEN, false, true, "");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, true, true, "JD rq b");
    // eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, true, false, "JD rq");
    //
    // | Case  | Inner product used | Jacobi block definition $K(j)$ |
    // | ----- | ------------------ | ------------------------------ |
    // | **A** | $L_2$              | $H_2(j)$                       |
    // | **B** | $L_2$              | $H_1(j) - \tau\,H_2(j)$        |
    // | **C** | $H_2$              | $H_2(j)$                       |
    // | **D** | $H_2$              | $H_1(j) - \tau\,H_2(j)$        |
    // | **E** | $L_2$              | Identity (no blocks)           |
    // | **F** | $H_2$              | Identity (no blocks)           |
    // ncv = 16;
    /* clang-format off */
    // auto result1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, false, false, false,false, 1, ncv  , "CO b1 L2 h2       ", elog);
    // auto result2 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, false, false, false,false, 4, ncv*4, "CO b4 L2 h2       ", elog);
    // auto result3 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, false, true, false, false, 1, ncv  , "CO b1 L2 h1-tau*h2", elog);
    // auto result4 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, false, true, false, false, 4, ncv*4, "CO b4 L2 h1-tau*h2", elog);
    // auto result5 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, true, false, false, false, 1, ncv  , "CO b1 H2 h2       ", elog);
    // auto result6 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, true, false, false, false, 4, ncv*4, "CO b4 H2 h2       ", elog);
    // auto result7 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, true, true, false,  false, 1, ncv  , "CO b1 H2 h1-tau*h2", elog);
    // auto result8 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, true, true, false,  false, 4, ncv*4, "CO b4 H2 h1-tau*h2", elog);
    // auto result5 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, false, false, true, false, "CO L2 Identity", elog);
    // auto result6 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, ResidualCorrectionType::CHEAP_OLSEN, crs, false, true, true,  true,  false, "CO H2 Identity", elog);

    auto resultA1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, false,  false, false, 1, ncv * 1  , "" , elog);
    // auto resultB1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, true,   false, false, 1, ncv * 1  , "B:JD b1 L2 h1-tau*h2"    , elog);
    // auto resultB2 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, true,   false, false, 4, ncv * 4  , "B:JD b4 L2 h1-tau*h2"    , elog);
    // auto resultC1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, true, false,   false, false, 1, ncv * 1  , "C:JD b1 H2 h2       "    , elog);
    // auto resultD1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, true, true,    false, false, 1, ncv * 1  , "D:JD b1 H2 h1-tau*h2"     , elog);
    // auto resultD4 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, true, true,    false, false, 4, ncv * 4  , "D:JD b4 H2 h1-tau*h2"     , elog);
    // auto resultE1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, false,  false, true, 1, ncv * 1   , "E:JD b1 L2 h2        th" , elog);
    // auto resultE4 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, false,  false, true, 4, ncv * 4   , "E:JD b4 L2 h2        th" , elog);
    // auto resultF1 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, true,   false, true, 1, ncv * 1   , "F:JD b1 L2 h1-tau*h2 th"  , elog);
    // auto resultF4 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, true,   false, true, 4, ncv * 4   , "F:JD b4 L2 h1-tau*h2 th"  , elog);
    /* clang-format on */

    return resultA1;
    // auto result5 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, true, false, "JD b", elog);
    // auto result6 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, true, true, "JD b h1h2", elog);
    // auto result7 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, true, "JD h1h2", elog);
    // return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, false, false, "JD", elog);
    // return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, crs, false, "JD");
}