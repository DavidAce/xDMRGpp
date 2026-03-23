#pragma once
#include "config/enums/OptAlgo.h"
#include "config/settings.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver_eigsmpo/solver_gdplusk.h"
#include "math/linalg/tensor/to_string.h"
#include "precond/generalized_basis_change.h"
#include "precond/standard_basis_change.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/TensorsFinite.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/opt/opt-internal.h"
using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

template<typename Scalar>
void analyze_spectrum(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat1, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat2,
                      std::string_view tag) {
    using MatrixCalc = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorCalc = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto es1 = Eigen::SelfAdjointEigenSolver<MatrixCalc>(mat1, Eigen::EigenvaluesOnly);
    auto es2 = Eigen::SelfAdjointEigenSolver<MatrixCalc>(mat2, Eigen::EigenvaluesOnly);
    auto ges = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixCalc>(mat1, mat2, Eigen::Ax_lBx | Eigen::EigenvaluesOnly);

    VectorCalc y1 = es1.eigenvalues().cwiseAbs();
    VectorCalc y2 = es2.eigenvalues().cwiseAbs();
    VectorCalc gy = ges.eigenvalues().cwiseAbs();

    auto max_y1 = y1.maxCoeff();
    auto min_y1 = y1.minCoeff();
    auto kappa1 = max_y1 / min_y1;

    auto max_y2 = y2.maxCoeff();
    auto min_y2 = y2.minCoeff();
    auto kappa2 = max_y2 / min_y2;

    auto max_gy = gy.maxCoeff();
    auto min_gy = gy.minCoeff();
    auto kappag = max_gy / min_gy;

    // tools::log->info("{}: y1: {::.4e}",tag, fv(y1));
    // tools::log->info("{}: y2: {::.4e}",tag, fv(y2));
    // tools::log->info("{}: gy: {::.4e}",tag, fv(gy));
    tools::log->info("{}: y1: [{:.16f}, {:.16f}] | kappa1 {:.3e}", tag, min_y1, max_y1, kappa1);
    tools::log->info("{}: y2: [{:.16f}, {:.16f}] | kappa2 {:.3e}", tag, min_y2, max_y2, kappa2);
    tools::log->info("{}: y3: [{:.16f}, {:.16f}] | kappag {:.3e}", tag, min_gy, max_gy, kappag);
}

template<typename Scalar>
void analyze_spectrum_std(const MatVecMPOS<Scalar> &H_orig, const MatVecMPOS<Scalar> &H_tilde, std::string_view tag,
                          const tools::finite::opt::precond::standard::BasisChange<Scalar> &bc) {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    auto mat_tilde = H_tilde.get_matrix();

    auto es_tilde = Eigen::SelfAdjointEigenSolver<MatrixType>(mat_tilde, Eigen::ComputeEigenvectors);

    VectorReal tilde_evals = es_tilde.eigenvalues();

    Eigen::Index min_idx, max_idx;

    RealScalar tilde_max_eval = tilde_evals.maxCoeff(&max_idx);
    RealScalar tilde_min_eval = tilde_evals.minCoeff(&min_idx);
    VectorType tilde_max_evec = es_tilde.eigenvectors().col(max_idx);
    VectorType tilde_min_evec = es_tilde.eigenvectors().col(min_idx);

    RealScalar tilde_kappa = std::abs(tilde_max_eval) / std::abs(tilde_min_eval);

    // Transform the tilde solution back to the original space
    VectorType tilde_max_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_max_evec, bc.shape_tilde, bc.TL, bc.TR);
    VectorType tilde_min_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_min_evec, bc.shape_tilde, bc.TL, bc.TR);

    VectorType tilde_max_Hv_orig = H_orig.MultAx(tilde_max_evec_orig);
    VectorType tilde_min_Hv_orig = H_orig.MultAx(tilde_min_evec_orig);

    RealScalar tilde_max_evec_orig_norm = std::real(tilde_max_evec_orig.dot(tilde_max_evec_orig));
    RealScalar tilde_min_evec_orig_norm = std::real(tilde_min_evec_orig.dot(tilde_min_evec_orig));

    RealScalar tilde_max_eval_orig = std::real(tilde_max_evec_orig.dot(tilde_max_Hv_orig)) / tilde_max_evec_orig_norm;
    RealScalar tilde_min_eval_orig = std::real(tilde_min_evec_orig.dot(tilde_min_Hv_orig)) / tilde_min_evec_orig_norm;

    // Compute the eigenpairs in the original space (not the round-trip ones) for comparison
    auto mat_orig = H_orig.get_matrix();
    auto es_orig  = Eigen::SelfAdjointEigenSolver<MatrixType>(mat_orig, Eigen::EigenvaluesOnly);

    VectorReal orig_evals = es_orig.eigenvalues();

    RealScalar orig_max_eval = orig_evals.maxCoeff();
    RealScalar orig_min_eval = orig_evals.minCoeff();

    RealScalar orig_kappa = std::abs(orig_max_eval) / std::abs(orig_min_eval);

    tools::log->info(
        "{}: orig [{:.16f} ... {:.16f}] | tilde [{:.16f} ... {:.16f}] | tilde (in orig basis) [{:.16f} ... {:.16f}] | kappa orig {:.3e} -> tilde {:.3e}", tag,
        orig_min_eval, orig_max_eval, tilde_min_eval, tilde_max_eval, tilde_min_eval_orig, tilde_max_eval_orig, orig_kappa, tilde_kappa);
}

template<typename Scalar>
void analyze_spectrum_gen(const MatVecMPOS<Scalar> &H_orig, const MatVecMPOS<Scalar> &H_tilde, const opt_mps<Scalar> &initial, std::string_view tag,
                          const tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar> &bc) {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    auto gap_rel = [](const VectorReal &evals, const RealScalar eval_target) -> RealScalar {
        constexpr static auto eps = std::numeric_limits<RealScalar>::epsilon();
        constexpr static auto inf = std::numeric_limits<RealScalar>::infinity();
        // Distances to target
        VectorReal d = (evals.array() - eval_target).cwiseAbs();

        // Mask out the target (or near-duplicates) with a relative tol
        const RealScalar reltol  = eps * (RealScalar(1) + abs(eval_target));
        const RealScalar mindist = (d.array() <= reltol).select(inf, d).minCoeff();
        const RealScalar denom   = RealScalar(1) + abs(eval_target);
        return mindist / denom;
    };

    auto P = [](const VectorType &v) {
        auto n = v.size();
        return MatrixType::Identity(n, n) - v * v.adjoint();
    };
    auto get_projected_kappa = [&P](const MatrixType &mat, const VectorType &v) -> RealScalar {
        auto       Pv  = P(v);
        MatrixType PHP = Pv * mat * Pv;
        auto       es  = Eigen::SelfAdjointEigenSolver<MatrixType>(PHP, Eigen::EigenvaluesOnly);
        return es.eigenvalues().cwiseAbs().maxCoeff() / es.eigenvalues().cwiseAbs().minCoeff();
    };
    auto mat_tilde = H_tilde.get_matrix();

    auto es_tilde = Eigen::SelfAdjointEigenSolver<MatrixType>(mat_tilde, Eigen::ComputeEigenvectors);

    VectorReal tilde_evals = es_tilde.eigenvalues();

    Eigen::Index min_idx, max_idx;

    RealScalar tilde_max_eval = tilde_evals.maxCoeff(&max_idx);
    RealScalar tilde_min_eval = tilde_evals.minCoeff(&min_idx);
    VectorType tilde_max_evec = es_tilde.eigenvectors().col(max_idx);
    VectorType tilde_min_evec = es_tilde.eigenvectors().col(min_idx);

    RealScalar tilde_kappa = tilde_evals.cwiseAbs().maxCoeff() / tilde_evals.cwiseAbs().minCoeff();

    // Transform the tilde solution back to the original space
    VectorType tilde_max_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_max_evec, bc.shape_tilde, bc.TL, bc.TR);
    VectorType tilde_min_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_min_evec, bc.shape_tilde, bc.TL, bc.TR);

    VectorType tilde_max_Hv_orig = H_orig.MultAx(tilde_max_evec_orig);
    VectorType tilde_min_Hv_orig = H_orig.MultAx(tilde_min_evec_orig);

    RealScalar tilde_max_evec_orig_norm = std::real(tilde_max_evec_orig.dot(tilde_max_evec_orig));
    RealScalar tilde_min_evec_orig_norm = std::real(tilde_min_evec_orig.dot(tilde_min_evec_orig));

    RealScalar tilde_max_eval_orig = std::real(tilde_max_evec_orig.dot(tilde_max_Hv_orig)) / tilde_max_evec_orig_norm;
    RealScalar tilde_min_eval_orig = std::real(tilde_min_evec_orig.dot(tilde_min_Hv_orig)) / tilde_min_evec_orig_norm;

    // Compute the eigenpairs in the original space (not the round-trip ones) for comparison
    auto mat_orig = H_orig.get_matrix();
    auto es_orig  = Eigen::SelfAdjointEigenSolver<MatrixType>(mat_orig, Eigen::EigenvaluesOnly);

    VectorReal orig_evals = es_orig.eigenvalues();

    RealScalar orig_max_eval = orig_evals.maxCoeff();
    RealScalar orig_min_eval = orig_evals.minCoeff();

    RealScalar orig_kappa = orig_evals.cwiseAbs().maxCoeff() / orig_evals.cwiseAbs().minCoeff();

    RealScalar orig_gap  = gap_rel(orig_evals, orig_evals.cwiseAbs().minCoeff());
    RealScalar tilde_gap = gap_rel(tilde_evals, tilde_evals.cwiseAbs().minCoeff());

    RealScalar kappa_proj_orig  = get_projected_kappa(mat_orig, initial.get_vector());
    RealScalar kappa_proj_tilde = get_projected_kappa(mat_tilde, bc.initial_guess.get_vector());

    tools::log->info("{}: orig [{:.16f} ... {:.16f}] | tilde [{:.16f} ... {:.16f}] | tilde (in orig basis) [{:.16f} ... {:.16f}]| kappa orig {:.3e} -> tilde "
                     "{:.3e} | rel gap orig {:.3e} -> tilde {:.3e} | kappa proj orig {:.3e} -> {:.3e}",
                     tag, orig_min_eval, orig_max_eval, tilde_min_eval, tilde_max_eval, tilde_min_eval_orig, tilde_max_eval_orig, orig_kappa, tilde_kappa,
                     orig_gap, tilde_gap, kappa_proj_orig, kappa_proj_tilde);
}
template<typename Scalar>
void analyze_spectrum_gen(const MatVecMPOS<Scalar> &H1_orig, const MatVecMPOS<Scalar> &H2_orig, const MatVecMPOS<Scalar> &H1_tilde,
                          const MatVecMPOS<Scalar> &H2_tilde, std::string_view tag,
                          const tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar> &bc) {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    auto gap_rel = [](const VectorReal &evals, const RealScalar eval_target) -> RealScalar {
        constexpr static auto eps = std::numeric_limits<RealScalar>::epsilon();
        constexpr static auto inf = std::numeric_limits<RealScalar>::infinity();
        // Distances to target
        VectorReal d = (evals.array() - eval_target).cwiseAbs();

        // Mask out the target (or near-duplicates) with a relative tol
        const RealScalar reltol  = eps * (RealScalar(1) + abs(eval_target));
        const RealScalar mindist = (d.array() <= reltol).select(inf, d).minCoeff();
        const RealScalar denom   = RealScalar(1) + abs(eval_target);
        return mindist / denom;
    };

    auto mat1_tilde = H1_tilde.get_matrix();
    auto mat2_tilde = H2_tilde.get_matrix();

    auto ges_tilde = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>(mat1_tilde, mat2_tilde, Eigen::Ax_lBx | Eigen::ComputeEigenvectors);

    VectorReal tilde_evals = ges_tilde.eigenvalues();

    Eigen::Index min_idx, max_idx;

    RealScalar tilde_max_eval = tilde_evals.maxCoeff(&max_idx);
    RealScalar tilde_min_eval = tilde_evals.minCoeff(&min_idx);
    VectorType tilde_max_evec = ges_tilde.eigenvectors().col(max_idx);
    VectorType tilde_min_evec = ges_tilde.eigenvectors().col(min_idx);

    RealScalar tilde_kappa = tilde_evals.cwiseAbs().maxCoeff() / tilde_evals.cwiseAbs().minCoeff();

    // Transform the tilde solution back to the original space
    VectorType tilde_max_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_max_evec, bc.shape_tilde, bc.TL, bc.TR);
    VectorType tilde_min_evec_orig = tools::finite::opt::precond::common::transform_vector(tilde_min_evec, bc.shape_tilde, bc.TL, bc.TR);

    // Compute the eigenpairs in the original space (not the round-trip ones) for comparison
    auto mat1_orig = H1_orig.get_matrix();
    auto mat2_orig = H2_orig.get_matrix();
    auto ges_orig  = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>(mat1_orig, mat2_orig, Eigen::Ax_lBx | Eigen::EigenvaluesOnly);

    VectorReal orig_evals = ges_orig.eigenvalues();

    RealScalar orig_max_eval = orig_evals.maxCoeff();
    RealScalar orig_min_eval = orig_evals.minCoeff();

    RealScalar orig_kappa = orig_evals.cwiseAbs().maxCoeff() / orig_evals.cwiseAbs().minCoeff();

    tools::log->info("{}: orig [{:.16f} ... {:.16f}] | tilde (in tilde basis) [{:.16f} ... {:.16f}]| kappa orig {:.3e} -> tilde {:.3e}", tag, orig_min_eval,
                     orig_max_eval, tilde_min_eval, tilde_max_eval, orig_kappa, tilde_kappa);
}

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> eigs_gdplusk_bc_std(const opt_mps<Scalar>       &initial, //
                                                 const TensorsFinite<Scalar> &tensors,
                                                 const OptMeta               &opt_meta,                        //
                                                 Eigen::Index                 jcb_max_block_size,              //
                                                 Eigen::Index                 jcb_overlap_size,                //
                                                 Eigen::Index                 jcb_num_passes,                  //
                                                 eig::Preconditioner          preconditioner_type,             //
                                                 ResidualCorrectionType       rct,                             //
                                                 bool                         use_coarse_inner_preconditioner, //
                                                 bool                         use_shifted_jd_eigenvalue,       //
                                                 bool                         use_h2_inner_product,            //
                                                 bool                         use_h1h2_preconditioner,         //
                                                 bool                         skipjcb,                         //
                                                 bool                         dev_thick_jd_projector,          //
                                                 bool                         use_jd_initial_guess,            //
                                                 BasisChangeScale             bcs,                             //
                                                 Eigen::Index                 block_size,
                                                 Eigen::Index                 ncv, //
                                                 std::string_view             tag, //
                                                 reports::eigs_log<Scalar>   &elog) {
    using CalcReal   = tools::finite::opt::RealScalar<CalcType>;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    // using MatrixCT          = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorCR          = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto           t_gdplusk  = tid::tic_scope("gdplusk");
    auto           t_preamble = tid::tic_scope("preamble");
    auto          &sites      = initial.get_sites();
    auto           mpos       = tensors.get_model().get_mpo(sites);
    auto           enve       = tensors.get_edges().get_multisite_env_ene(sites);
    auto           envv       = tensors.get_edges().get_multisite_env_var(sites);
    auto           size       = initial.get_tensor().size();
    constexpr auto eps        = std::numeric_limits<CalcReal>::epsilon();
    auto           nev        = opt_meta.eigs_nev.value_or(settings::precision::eigs_nev_min);
    // auto           ncv      = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(ncv <= 0) {
        // Automatic selection

        Eigen::Index ncv_by_size = safe_cast<Eigen::Index>(std::ceil(std::log2(size)));
        Eigen::Index ncv_min     = std::max<Eigen::Index>(2 * nev, settings::precision::eigs_ncv_min);
        Eigen::Index ncv_max     = settings::precision::eigs_ncv_max <= 0 ? ncv_by_size : static_cast<Eigen::Index>(settings::precision::eigs_ncv_max);
        ncv                      = std::clamp(ncv_by_size, ncv_min, ncv_max);
        tools::log->trace("ncv automatic selection: {} (min {}, max {})", ncv, ncv_min, ncv_max);
    }

    // auto H1   = MatVecMPOS<CalcType>(mpos, enve);
    // auto H2   = MatVecMPOS<CalcType>(mpos, envv);
    // auto H1H2 = MatVecMPOS<CalcType>(mpos, enve, envv);
    auto vh1v  = tools::finite::measure::expval_hamiltonian(initial.get_tensor(), mpos, enve);
    auto vh2v  = tools::finite::measure::expval_hamiltonian_squared(initial.get_tensor(), mpos, envv);
    auto scale = std::abs(vh1v / vh2v);
    auto alpha = RealScalar{0.5f};
    // auto bc    = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(initial, tensors, bcs, scale, alpha);
    auto bc = tools::finite::opt::precond::standard::BasisChange<Scalar>(initial, tensors, bcs, scale, alpha);

    auto H1   = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair());
    auto H2   = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());
    auto H1H2 = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair(), bc.get_envv_pair());

    if constexpr(std::is_same_v<CalcType, double> and std::is_same_v<CalcType, Scalar>)
        if(size <= 2048) {
            auto H1_orig = MatVecMPOS<CalcType>(mpos, enve);
            auto H2_orig = MatVecMPOS<CalcType>(mpos, envv);
            analyze_spectrum_std(H1_orig, H1, "H1", bc);
            analyze_spectrum_std(H2_orig, H2, "H2", bc);
        }

    // BlockLanczos<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), mpos, enve, envv);
    // solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, bc.initial_guess.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver.abstol = opt_meta.eigs_abstol.has_value() ? narrow_cast<CalcReal>(opt_meta.eigs_abstol.value()) : eps * 10000;
    solver.reltol = opt_meta.eigs_reltol.has_value() ? narrow_cast<CalcReal>(opt_meta.eigs_reltol.value()) : CalcReal{0};
    eig::setLevel(spdlog::level::info);
    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.setLogger(spdlog::level::info, fmt::format("gd+k {}", tag));
    solver.b              = block_size; // opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    solver.status.initVal = static_cast<CalcReal>(initial.get_energy());
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.max_matvecs    = -1ul; // opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.set_jcbMaxBlockSize(jcb_max_block_size);
    solver.set_jcbOverlapSize(jcb_overlap_size);
    solver.set_jcbNumPasses(jcb_num_passes);
    solver.set_chebyshevFilterDegree(0);
    solver.set_chebyshevFilterLambdaCutBias(0.1f);
    solver.set_chebyshevFilterRelGapThreshold(1e-3f);
    solver.set_maxBasisBlocks(ncv);
    solver.set_maxRetainBlocks(3 * ncv / 8);
    solver.set_maxPrevBlocks(1);
    // solver.set_maxLanczosResidualHistory(0);
    solver.set_maxRitzResidualHistory(1);
    solver.set_maxExtraRitzHistory(1);
    solver.set_preconditioner_type(preconditioner_type);
    solver.use_krylov_schur_gdplusk_restart        = true;
    solver.use_refined_rayleigh_ritz               = true;
    solver.use_relative_rnorm_tolerance            = true;
    solver.use_adaptive_inner_tolerance            = true;
    solver.use_coarse_inner_preconditioner         = use_coarse_inner_preconditioner;
    solver.use_rayleigh_quotients_instead_of_evals = true;
    solver.use_shifted_jd_eigenvalue               = use_shifted_jd_eigenvalue;
    solver.use_h2_inner_product                    = use_h2_inner_product;
    solver.use_jd_initial_guess                    = use_jd_initial_guess;
    solver.use_h1h2_jcb_preconditioner             = use_h1h2_preconditioner;
    solver.dev_skipjcb                             = skipjcb;
    solver.dev_thick_jd_projector                  = dev_thick_jd_projector;
    solver.residual_correction_type                = rct;
    solver.inject_randomness                       = false;
    solver.tag                                     = tag;
    t_preamble.toc();
    solver.run();

    if(!has_flag(solver.status.stopReason, StopReason::converged_rNorms) and solver.max_iters > 10)
        tools::log->info("GD+k: status.stopReason = {}", solver.status.stopMessage);

    // Sanity check
    // decltype(auto) TLc = tenx::asScalarType<CalcType>(bc.TL);
    // decltype(auto) TRc = tenx::asScalarType<CalcType>(bc.TR);
    // decltype(auto) SLc = tenx::asScalarType<CalcType>(bc.SL);
    // decltype(auto) SRc = tenx::asScalarType<CalcType>(bc.SR);
    //
    // {
    //     using VectorCalc     = Eigen::Matrix<CalcType, Eigen::Dynamic, 1>;
    //     VectorCalc tilde_psi = solver.V.col(0);
    //
    //     // Round trip check in tilde → orig → tilde using transform_vector
    //     auto x_tilde = tilde_psi; // or random unit vector
    //     auto x_orig  = tools::finite::opt::precond::common::transform_vector(x_tilde, bc.shape_tilde, TLc, TRc);
    //     auto x_rt    = tools::finite::opt::precond::common::transform_vector(x_orig, bc.shape_orig, SLc, SRc);
    //
    //     CalcReal rt_err = (x_rt - x_tilde).norm() / std::max<CalcReal>(1, x_tilde.norm());
    //     tools::log->info("round-trip vec (tilde→orig→tilde): {:.3e}", fp(rt_err));
    //
    //     // and the opposite direction
    //     auto     y_orig  = x_orig; // reuse
    //     auto     y_tilde = tools::finite::opt::precond::common::transform_vector(y_orig, bc.shape_orig, SLc, SRc);
    //     auto     y_rt    = tools::finite::opt::precond::common::transform_vector(y_tilde, bc.shape_tilde, TLc, TRc);
    //     CalcReal rt_err2 = (y_rt - y_orig).norm() / std::max<CalcReal>(1, y_orig.norm());
    //     tools::log->info("round-trip vec (orig→tilde→orig): {:.3e}", fp(rt_err2));
    //
    //     VectorCalc tilde_H1psi(tilde_psi.size());
    //     VectorCalc tilde_H2psi(tilde_psi.size());
    //     H1.MultAx(tilde_psi.data(), tilde_H1psi.data());
    //     H2.MultAx(tilde_psi.data(), tilde_H2psi.data());
    //
    //     VectorCalc tilde_H1psi_orig = tools::finite::opt::precond::common::transform_vector(tilde_H1psi, bc.shape_tilde, TLc, TRc);
    //     VectorCalc tilde_H2psi_orig = tools::finite::opt::precond::common::transform_vector(tilde_H2psi, bc.shape_tilde, TLc, TRc);
    //
    //     auto       orig_H1  = MatVecMPOS<CalcType>(mpos, enve);
    //     auto       orig_H2  = MatVecMPOS<CalcType>(mpos, envv);
    //     VectorCalc orig_psi = tools::finite::opt::precond::common::transform_vector(tilde_psi, bc.shape_tilde, TLc, TRc);
    //     VectorCalc orig_H1psi(orig_psi.size());
    //     VectorCalc orig_H2psi(orig_psi.size());
    //     orig_H1.MultAx(orig_psi.data(), orig_H1psi.data());
    //     orig_H2.MultAx(orig_psi.data(), orig_H2psi.data());
    //
    //     VectorCalc orig_H1psi_tilde = tools::finite::opt::precond::common::transform_vector(orig_H1psi, bc.shape_orig, SLc, SRc);
    //     VectorCalc orig_H2psi_tilde = tools::finite::opt::precond::common::transform_vector(orig_H2psi, bc.shape_orig, SLc, SRc);
    //
    //     CalcReal psi_h1error_in_orig  = (orig_H1psi - tilde_H1psi_orig).norm() / std::max(CalcReal{1}, orig_H1psi.norm());
    //     CalcReal psi_h2error_in_orig  = (orig_H2psi - tilde_H2psi_orig).norm() / std::max(CalcReal{1}, orig_H2psi.norm());
    //     CalcReal psi_h1error_in_tilde = (tilde_H1psi - orig_H1psi_tilde).norm() / std::max(CalcReal{1}, tilde_H1psi.norm());
    //     CalcReal psi_h2error_in_tilde = (tilde_H2psi - orig_H2psi_tilde).norm() / std::max(CalcReal{1}, tilde_H2psi.norm());
    //     tools::log->info("‖orig_H1psi‖  = {:.3e}, ‖tilde_H1psi_orig‖ = {:.3e}", fp(orig_H1psi.norm()), fp(tilde_H1psi_orig.norm()));
    //     tools::log->info("‖tilde_H1psi‖ = {:.3e}, ‖orig_H1psi_tilde‖ = {:.3e}", fp(tilde_H1psi.norm()), fp(orig_H1psi_tilde.norm()));
    //     tools::log->info("‖orig_H2psi‖  = {:.3e}, ‖tilde_H2psi_orig‖ = {:.3e}", fp(orig_H2psi.norm()), fp(tilde_H2psi_orig.norm()));
    //     tools::log->info("‖tilde_H2psi‖ = {:.3e}, ‖orig_H2psi_tilde‖ = {:.3e}", fp(tilde_H2psi.norm()), fp(orig_H2psi_tilde.norm()));
    //     tools::log->info("Psi error in orig : h1psi: {:.5e} h2psi: {:.5e}", fp(psi_h1error_in_orig), fp(psi_h2error_in_orig));
    //     tools::log->info("Psi error in tilde: h1psi: {:.5e} h2psi: {:.5e}", fp(psi_h1error_in_tilde), fp(psi_h2error_in_tilde));
    //
    //     CalcReal tilde_norm2 = std::real(tilde_psi.dot(tilde_psi));
    //     CalcReal tilde_E1    = std::real(tilde_psi.dot(tilde_H1psi)) / tilde_norm2;
    //     CalcReal tilde_E2    = std::real(tilde_psi.dot(tilde_H2psi)) / tilde_norm2;
    //     CalcReal tilde_Var   = tilde_E2 - tilde_E1 * tilde_E1;
    //
    //     CalcReal orig_norm2 = std::real(orig_psi.dot(orig_psi));
    //     CalcReal orig_E1    = std::real(orig_psi.dot(orig_H1psi)) / orig_norm2;
    //     CalcReal orig_E2    = std::real(orig_psi.dot(orig_H2psi)) / orig_norm2;
    //     CalcReal orig_Var   = orig_E2 - orig_E1 * orig_E1;
    //     tools::log->info("Psi var: orig: {:.5e} tilde: {:.5e}", fp(orig_Var), fp(tilde_Var));
    //
    //     auto rand_vec = [&]() {
    //         VectorCalc v(solver.mps_shape[0] * solver.mps_shape[1] * solver.mps_shape[2]);
    //         v.setRandom();
    //         v.normalize();
    //         return v;
    //     };
    //
    //     VectorCalc psi_orig_vec = rand_vec();
    //
    //     // Map to tilde
    //     VectorCalc psi_tilde_vec = tools::finite::opt::precond::common::transform_vector(psi_orig_vec, bc.shape_orig, SLc, SRc);
    //
    //     // Apply original operator
    //     VectorCalc y_orig_vec(psi_orig_vec.size());
    //     orig_H1.MultAx(psi_orig_vec.data(), y_orig_vec.data());
    //
    //     // Apply hat operator
    //     VectorCalc y_tilde_vec(psi_tilde_vec.size());
    //     H1.MultAx(psi_tilde_vec.data(), y_tilde_vec.data());
    //
    //     // Path A: tilde result back to original
    //     VectorCalc yA_orig = tools::finite::opt::precond::common::transform_vector(y_tilde_vec, bc.shape_tilde, TLc, TRc);
    //
    //     // Path B: original result over to tilde
    //     VectorCalc yB_tilde = tools::finite::opt::precond::common::transform_vector(y_orig_vec, bc.shape_orig, SLc, SRc);
    //
    //     // Compare in original basis
    //     auto err_orig = (yA_orig - y_orig_vec).norm() / std::max<decltype(yA_orig.norm())>(1, y_orig_vec.norm());
    //     // Compare in tilde basis
    //     auto err_tilde = (y_tilde_vec - yB_tilde).norm() / std::max<decltype(y_tilde_vec.norm())>(1, y_tilde_vec.norm());
    //
    //     tools::log->info("Operator equivalence: err_orig={:.3e}, err_tilde={:.3e}", fp(err_orig), fp(err_tilde));
    //
    //     // Print norms to see scale consistency
    //     tools::log->info("‖y_orig‖={:.3e}, ‖yA_orig‖={:.3e}; ‖y_tilde‖={:.3e}, ‖yB_tilde‖={:.3e}", fp(y_orig_vec.norm()), fp(yA_orig.norm()),
    //                      fp(y_tilde_vec.norm()), fp(yB_tilde.norm()));
    // }
    decltype(auto) TLc = tenx::asScalarType<CalcType>(bc.TL);
    decltype(auto) TRc = tenx::asScalarType<CalcType>(bc.TR);
    solver.V           = tools::finite::opt::precond::common::transform_matrix(solver.V, solver.mps_shape, TLc, TRc);
    solver.V.colwise().normalize();
    auto res = std::vector<opt_mps<Scalar>>();
    extract_results(tensors, initial, opt_meta, solver, res);

    elog.eigs_add_entry(res.front(), spdlog::level::debug);
    return res;
}

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> eigs_gdplusk_bc_gen(const opt_mps<Scalar>       &initial, //
                                                 const TensorsFinite<Scalar> &tensors,
                                                 const OptMeta               &opt_meta,                        //
                                                 Eigen::Index                 jcb_max_block_size,              //
                                                 Eigen::Index                 jcb_overlap_size,                //
                                                 Eigen::Index                 jcb_num_passes,                  //
                                                 eig::Preconditioner          preconditioner_type,             //
                                                 ResidualCorrectionType       rct,                             //
                                                 bool                         use_coarse_inner_preconditioner, //
                                                 bool                         use_shifted_jd_eigenvalue,       //
                                                 bool                         use_h2_inner_product,            //
                                                 bool                         use_h1h2_jcb_preconditioner,     //
                                                 bool                         skipjcb,                         //
                                                 bool                         dev_thick_jd_projector,          //
                                                 bool                         use_jd_initial_guess,            //
                                                 bool                         use_jd_h2_only,                  //
                                                 bool                         use_jd_def_solver,               //
                                                 Eigen::Index                 block_size,
                                                 Eigen::Index                 ncv,  //
                                                 BasisChangeConfig            bcfg, //
                                                 std::string_view             tag,  //
                                                 reports::eigs_log<Scalar>   &elog) {
    using CalcReal   = tools::finite::opt::RealScalar<CalcType>;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    // using MatrixCT          = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorCR          = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto log_name  = tools::log->name();
    auto log_level = tools::log->level();
    // tools::log = spdlog::get(std::string(tag));
    tools::log =
        spdlog::get(std::string(tag)) == nullptr ? spdlog::stdout_color_mt(std::string(tag), spdlog::color_mode::always) : spdlog::get(std::string(tag));
    tools::log->set_level(log_level);
    auto  t_mixblk   = tid::tic_token(tag);
    auto  t_gdplusk  = tid::tic_scope("gdplusk");
    auto  t_preamble = tid::tic_scope("preamble");
    auto &sites      = initial.get_sites();
    auto  mpos       = tensors.get_model().get_mpo(sites);
    // auto           enve       = tensors.get_edges().get_multisite_env_ene(sites);
    // auto           envv       = tensors.get_edges().get_multisite_env_var(sites);
    auto           size = initial.get_tensor().size();
    constexpr auto eps  = std::numeric_limits<CalcReal>::epsilon();
    auto           nev  = opt_meta.eigs_nev.value_or(settings::precision::eigs_nev_min);
    // auto           ncv      = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(ncv <= 0) {
        // Automatic selection

        Eigen::Index ncv_by_size = safe_cast<Eigen::Index>(std::ceil(std::log2(size)));
        Eigen::Index ncv_min     = std::max<Eigen::Index>(2 * nev, settings::precision::eigs_ncv_min);
        Eigen::Index ncv_max     = settings::precision::eigs_ncv_max <= 0 ? ncv_by_size : static_cast<Eigen::Index>(settings::precision::eigs_ncv_max);
        ncv                      = std::clamp(ncv_by_size, ncv_min, ncv_max);
        tools::log->trace("ncv automatic selection: {} (min {}, max {})", ncv, ncv_min, ncv_max);
    }
    // auto H1   = MatVecMPOS<CalcType>(mpos, enve);
    // auto H2   = MatVecMPOS<CalcType>(mpos, envv);
    // auto H1H2 = MatVecMPOS<CalcType>(mpos, enve, envv);
    // auto vh1v  = tools::finite::measure::expval_hamiltonian(initial.get_tensor(), mpos, enve);
    // auto vh2v  = tools::finite::measure::expval_hamiltonian_squared(initial.get_tensor(), mpos, envv);
    // auto scale = std::abs(vh1v / vh2v);
    // auto ewt   = EnvWeightType::NO_PSI_TRACE;
    // auto ewr   = EnvWeightRegularizer::NORM;
    auto bc = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(initial, tensors, bcfg);
    // for(Eigen::Index rep = 0; rep < bcfg.maxreps; ++rep) {
    //     if constexpr(std::is_same_v<CalcType, double> and std::is_same_v<CalcType, Scalar>) {
    //         if(size <= 2048) {
    //             auto H1      = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair());
    //             auto H2      = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());
    //             auto H1H2    = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair(), bc.get_envv_pair());
    //             auto H1_orig = MatVecMPOS<CalcType>(mpos, enve);
    //             auto H2_orig = MatVecMPOS<CalcType>(mpos, envv);
    //             tools::log->info("alpha = {:.2f} (rep:{})", bcfg.alpha, rep);
    //             analyze_spectrum_gen(H1_orig, H1, initial, " H1", bc);
    //             analyze_spectrum_gen(H2_orig, H2, initial, " H2", bc);
    //             analyze_spectrum_gen(H1_orig, H2_orig, H1, H2, " HH", bc);
    //         }
    //     }
    //     if(rep + 1 < bcfg.maxreps) {
    //         auto bc_next = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(bc, bcfg);
    //         bc           = bc_next;
    //     }
    // }

    // auto alpha = RealScalar{0.5f};
    // auto bc    = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(initial, tensors, bcs, scale, alpha);
    // auto bc1 = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(initial, tensors, bcs, scale, static_cast<RealScalar>(alphaf));
    // auto bc =  tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(bc1);
    auto H1   = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair());
    auto H2   = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());
    auto H1H2 = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair(), bc.get_envv_pair());

    // BlockLanczos<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), mpos, enve, envv);
    // solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, bc.initial_guess.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver.abstol = opt_meta.eigs_abstol.has_value() ? narrow_cast<CalcReal>(opt_meta.eigs_abstol.value()) : eps * 10000;
    solver.reltol = opt_meta.eigs_reltol.has_value() ? narrow_cast<CalcReal>(opt_meta.eigs_reltol.value()) : CalcReal{0};
    eig::setLevel(spdlog::level::info);
    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.setLogger(spdlog::level::trace, fmt::format("gen gd+k {}", tag));
    solver.b              = block_size; // opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    solver.status.initVal = static_cast<CalcReal>(initial.get_energy());
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.max_matvecs    = -1ul; // opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.set_jcbMaxBlockSize(jcb_max_block_size);
    solver.set_jcbOverlapSize(jcb_overlap_size);
    solver.set_jcbNumPasses(jcb_num_passes);
    solver.set_chebyshevFilterDegree(0);
    solver.set_chebyshevFilterLambdaCutBias(0.1f);
    solver.set_chebyshevFilterRelGapThreshold(1e-3f);
    solver.set_maxBasisBlocks(ncv);
    solver.set_maxRetainBlocks(3 * ncv / 8);
    solver.set_maxPrevBlocks(1);
    // solver.set_maxLanczosResidualHistory(0);
    solver.set_maxRitzResidualHistory(1);
    solver.set_maxExtraRitzHistory(1);
    solver.set_preconditioner_type(preconditioner_type);
    solver.use_krylov_schur_gdplusk_restart        = true;
    solver.use_refined_rayleigh_ritz               = true;
    solver.use_relative_rnorm_tolerance            = true;
    solver.use_adaptive_inner_tolerance            = true;
    solver.use_coarse_inner_preconditioner         = use_coarse_inner_preconditioner;
    solver.use_rayleigh_quotients_instead_of_evals = true;
    solver.use_shifted_jd_eigenvalue               = use_shifted_jd_eigenvalue;
    solver.use_h2_inner_product                    = use_h2_inner_product;
    solver.use_jd_initial_guess                    = use_jd_initial_guess;
    solver.use_h1h2_jcb_preconditioner             = use_h1h2_jcb_preconditioner;
    solver.use_jd_h2_only                          = use_jd_h2_only;
    solver.use_jd_def_solver                       = use_jd_def_solver;
    solver.dev_skipjcb                             = skipjcb;
    solver.dev_thick_jd_projector                  = dev_thick_jd_projector;
    solver.residual_correction_type                = rct;
    solver.inject_randomness                       = false;
    solver.tag                                     = tag;

    solver.debug_check_H2_symmetry(2);
    t_preamble.toc();

    solver.run();

    if(!has_flag(solver.status.stopReason, StopReason::converged_rNorms) and solver.max_iters > 10)
        tools::log->info("GD+k: status.stopReason = {}", solver.status.stopMessage);

    // Sanity check
    // decltype(auto) TLc = tenx::asScalarType<CalcType>(bc.TL);
    // decltype(auto) TRc = tenx::asScalarType<CalcType>(bc.TR);
    // decltype(auto) SLc = tenx::asScalarType<CalcType>(bc.SL);
    // decltype(auto) SRc = tenx::asScalarType<CalcType>(bc.SR);

    // {
    //     using VectorCalc     = Eigen::Matrix<CalcType, Eigen::Dynamic, 1>;
    //     VectorCalc tilde_psi = solver.V.col(0);
    //
    //     // Round trip check in tilde → orig → tilde using transform_vector
    //     auto x_tilde = tilde_psi; // or random unit vector
    //     auto x_orig  = tools::finite::opt::precond::common::transform_vector(x_tilde, bc.shape_tilde, TLc, TRc);
    //     auto x_rt    = tools::finite::opt::precond::common::transform_vector(x_orig, bc.shape_orig, SLc, SRc);
    //
    //     CalcReal rt_err = (x_rt - x_tilde).norm() / std::max<CalcReal>(1, x_tilde.norm());
    //     tools::log->info("round-trip vec (tilde→orig→tilde): {:.3e}", fp(rt_err));
    //
    //     // and the opposite direction
    //     auto     y_orig  = x_orig; // reuse
    //     auto     y_tilde = tools::finite::opt::precond::common::transform_vector(y_orig, bc.shape_orig, SLc, SRc);
    //     auto     y_rt    = tools::finite::opt::precond::common::transform_vector(y_tilde, bc.shape_tilde, TLc, TRc);
    //     CalcReal rt_err2 = (y_rt - y_orig).norm() / std::max<CalcReal>(1, y_orig.norm());
    //     tools::log->info("round-trip vec (orig→tilde→orig): {:.3e}", fp(rt_err2));
    //
    //     VectorCalc tilde_H1psi(tilde_psi.size());
    //     VectorCalc tilde_H2psi(tilde_psi.size());
    //     H1.MultAx(tilde_psi.data(), tilde_H1psi.data());
    //     H2.MultAx(tilde_psi.data(), tilde_H2psi.data());
    //
    //     VectorCalc tilde_H1psi_orig = tools::finite::opt::precond::common::transform_vector(tilde_H1psi, bc.shape_tilde, TLc, TRc);
    //     VectorCalc tilde_H2psi_orig = tools::finite::opt::precond::common::transform_vector(tilde_H2psi, bc.shape_tilde, TLc, TRc);
    //
    //     auto       orig_H1  = MatVecMPOS<CalcType>(mpos, enve);
    //     auto       orig_H2  = MatVecMPOS<CalcType>(mpos, envv);
    //     VectorCalc orig_psi = tools::finite::opt::precond::common::transform_vector(tilde_psi, bc.shape_tilde, TLc, TRc);
    //     VectorCalc orig_H1psi(orig_psi.size());
    //     VectorCalc orig_H2psi(orig_psi.size());
    //     orig_H1.MultAx(orig_psi.data(), orig_H1psi.data());
    //     orig_H2.MultAx(orig_psi.data(), orig_H2psi.data());
    //
    //     VectorCalc orig_H1psi_tilde = tools::finite::opt::precond::common::transform_vector(orig_H1psi, bc.shape_orig, SLc, SRc);
    //     VectorCalc orig_H2psi_tilde = tools::finite::opt::precond::common::transform_vector(orig_H2psi, bc.shape_orig, SLc, SRc);
    //
    //     CalcReal psi_h1error_in_orig  = (orig_H1psi - tilde_H1psi_orig).norm() / std::max(CalcReal{1}, orig_H1psi.norm());
    //     CalcReal psi_h2error_in_orig  = (orig_H2psi - tilde_H2psi_orig).norm() / std::max(CalcReal{1}, orig_H2psi.norm());
    //     CalcReal psi_h1error_in_tilde = (tilde_H1psi - orig_H1psi_tilde).norm() / std::max(CalcReal{1}, tilde_H1psi.norm());
    //     CalcReal psi_h2error_in_tilde = (tilde_H2psi - orig_H2psi_tilde).norm() / std::max(CalcReal{1}, tilde_H2psi.norm());
    //     tools::log->info("‖orig_H1psi‖  = {:.3e}, ‖tilde_H1psi_orig‖ = {:.3e}", fp(orig_H1psi.norm()), fp(tilde_H1psi_orig.norm()));
    //     tools::log->info("‖tilde_H1psi‖ = {:.3e}, ‖orig_H1psi_tilde‖ = {:.3e}", fp(tilde_H1psi.norm()), fp(orig_H1psi_tilde.norm()));
    //     tools::log->info("‖orig_H2psi‖  = {:.3e}, ‖tilde_H2psi_orig‖ = {:.3e}", fp(orig_H2psi.norm()), fp(tilde_H2psi_orig.norm()));
    //     tools::log->info("‖tilde_H2psi‖ = {:.3e}, ‖orig_H2psi_tilde‖ = {:.3e}", fp(tilde_H2psi.norm()), fp(orig_H2psi_tilde.norm()));
    //     tools::log->info("Psi error in orig : h1psi: {:.5e} h2psi: {:.5e}", fp(psi_h1error_in_orig), fp(psi_h2error_in_orig));
    //     tools::log->info("Psi error in tilde: h1psi: {:.5e} h2psi: {:.5e}", fp(psi_h1error_in_tilde), fp(psi_h2error_in_tilde));
    //
    //     CalcReal tilde_norm2 = std::real(tilde_psi.dot(tilde_psi));
    //     CalcReal tilde_E1    = std::real(tilde_psi.dot(tilde_H1psi)) / tilde_norm2;
    //     CalcReal tilde_E2    = std::real(tilde_psi.dot(tilde_H2psi)) / tilde_norm2;
    //     CalcReal tilde_Var   = tilde_E2 - tilde_E1 * tilde_E1;
    //
    //     CalcReal orig_norm2 = std::real(orig_psi.dot(orig_psi));
    //     CalcReal orig_E1    = std::real(orig_psi.dot(orig_H1psi)) / orig_norm2;
    //     CalcReal orig_E2    = std::real(orig_psi.dot(orig_H2psi)) / orig_norm2;
    //     CalcReal orig_Var   = orig_E2 - orig_E1 * orig_E1;
    //     tools::log->info("Psi var: orig: {:.5e} tilde: {:.5e}", fp(orig_Var), fp(tilde_Var));
    //
    //     auto rand_vec = [&]() {
    //         VectorCalc v(solver.mps_shape[0] * solver.mps_shape[1] * solver.mps_shape[2]);
    //         v.setRandom();
    //         v.normalize();
    //         return v;
    //     };
    //
    //     VectorCalc psi_orig_vec = rand_vec();
    //
    //     // Map to tilde
    //     VectorCalc psi_tilde_vec = tools::finite::opt::precond::common::transform_vector(psi_orig_vec, bc.shape_orig, SLc, SRc);
    //
    //     // Apply original operator
    //     VectorCalc y_orig_vec(psi_orig_vec.size());
    //     orig_H1.MultAx(psi_orig_vec.data(), y_orig_vec.data());
    //
    //     // Apply hat operator
    //     VectorCalc y_tilde_vec(psi_tilde_vec.size());
    //     H1.MultAx(psi_tilde_vec.data(), y_tilde_vec.data());
    //
    //     // Path A: tilde result back to original
    //     VectorCalc yA_orig = tools::finite::opt::precond::common::transform_vector(y_tilde_vec, bc.shape_tilde, TLc, TRc);
    //
    //     // Path B: original result over to tilde
    //     VectorCalc yB_tilde = tools::finite::opt::precond::common::transform_vector(y_orig_vec, bc.shape_orig, SLc, SRc);
    //
    //     // Compare in original basis
    //     auto err_orig = (yA_orig - y_orig_vec).norm() / std::max<decltype(yA_orig.norm())>(1, y_orig_vec.norm());
    //     // Compare in tilde basis
    //     auto err_tilde = (y_tilde_vec - yB_tilde).norm() / std::max<decltype(y_tilde_vec.norm())>(1, y_tilde_vec.norm());
    //
    //     tools::log->info("Operator equivalence: err_orig={:.3e}, err_tilde={:.3e}", fp(err_orig), fp(err_tilde));
    //
    //     // Print norms to see scale consistency
    //     tools::log->info("‖y_orig‖={:.3e}, ‖yA_orig‖={:.3e}; ‖y_tilde‖={:.3e}, ‖yB_tilde‖={:.3e}", fp(y_orig_vec.norm()), fp(yA_orig.norm()),
    //                      fp(y_tilde_vec.norm()), fp(yB_tilde.norm()));
    // }
    decltype(auto) TLc = tenx::asScalarType<CalcType>(bc.TL);
    decltype(auto) TRc = tenx::asScalarType<CalcType>(bc.TR);
    solver.V           = tools::finite::opt::precond::common::transform_matrix(solver.V, solver.mps_shape, TLc, TRc);
    solver.V.colwise().normalize();
    auto res = std::vector<opt_mps<Scalar>>();
    extract_results(tensors, initial, opt_meta, solver, res);

    elog.eigs_add_entry(res.front(), spdlog::level::debug);
    tools::log = spdlog::get(log_name) == nullptr ? spdlog::stdout_color_mt(log_name, spdlog::color_mode::always) : spdlog::get(log_name);
    tools::log->set_level(log_level);
    return res;
}

template<typename CalcType, typename Scalar>
std::vector<opt_mps<Scalar>> eigs_gdplusk(const opt_mps<Scalar>       &initial, //
                                          const TensorsFinite<Scalar> &tensors,
                                          const OptMeta               &opt_meta,                        //
                                          Eigen::Index                 jcb_max_block_size,              //
                                          Eigen::Index                 jcb_overlap_size,                //
                                          Eigen::Index                 jcb_num_passes,                  //
                                          eig::Preconditioner          preconditioner_type,             //
                                          ResidualCorrectionType       rct,                             //
                                          bool                         use_coarse_inner_preconditioner, //
                                          bool                         use_shifted_jd_eigenvalue,       //
                                          bool                         use_h2_inner_product,            //
                                          bool                         use_h1h2_preconditioner,         //
                                          bool                         skipjcb,                         //
                                          bool                         dev_thick_jd_projector,          //
                                          bool                         use_jd_initial_guess,            //
                                          Eigen::Index                 block_size,
                                          Eigen::Index                 ncv, //
                                          std::string_view             tag, //
                                          reports::eigs_log<Scalar>   &elog) {
    using RealScalar = tools::finite::opt::RealScalar<CalcType>;
    // using MatrixCT          = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorCR          = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto           t_tag      = tid::tic_token(tag);
    auto           t_gdplusk  = tid::tic_scope("gdplusk");
    auto           t_preamble = tid::tic_scope("preamble");
    auto          &sites      = initial.get_sites();
    auto           mpos       = tensors.get_model().get_mpo(sites);
    auto           enve       = tensors.get_edges().get_multisite_env_ene(sites);
    auto           envv       = tensors.get_edges().get_multisite_env_var(sites);
    auto           size       = initial.get_tensor().size();
    constexpr auto eps        = std::numeric_limits<RealScalar>::epsilon();
    auto           nev        = opt_meta.eigs_nev.value_or(settings::precision::eigs_nev_min);
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
    // solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver_gdplusk<CalcType> solver(nev, ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_tensor_as_matrix<CalcType>(), H1, H2, H1H2);
    solver.abstol = opt_meta.eigs_abstol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_abstol.value()) : eps * 10000;
    solver.reltol = opt_meta.eigs_reltol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_reltol.value()) : RealScalar{0};
    eig::setLevel(spdlog::level::info);
    if(opt_meta.eigs_jcbMaxBlockSize.has_value() and opt_meta.eigs_jcbMaxBlockSize.value() > 0) {
        solver.set_jcbMaxBlockSize(opt_meta.eigs_jcbMaxBlockSize.value_or(0));
    }
    solver.setLogger(spdlog::level::info, fmt::format("std gd+k {}", tag));
    solver.b              = block_size; // opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    solver.status.initVal = static_cast<RealScalar>(initial.get_energy());
    solver.max_iters      = opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.max_matvecs    = -1ul; // opt_meta.eigs_iter_max.value_or(settings::precision::eigs_iter_min);
    solver.set_jcbMaxBlockSize(jcb_max_block_size);
    solver.set_jcbOverlapSize(jcb_overlap_size);
    solver.set_jcbNumPasses(jcb_num_passes);
    solver.set_chebyshevFilterDegree(0);
    solver.set_chebyshevFilterLambdaCutBias(0.1f);
    solver.set_chebyshevFilterRelGapThreshold(1e-3f);
    solver.set_maxBasisBlocks(ncv);
    solver.set_maxRetainBlocks(3 * ncv / 8);
    solver.set_maxPrevBlocks(1);
    // solver.set_maxLanczosResidualHistory(0);
    solver.set_maxRitzResidualHistory(1);
    solver.set_maxExtraRitzHistory(1);
    solver.set_preconditioner_type(preconditioner_type);
    solver.use_krylov_schur_gdplusk_restart        = true;
    solver.use_refined_rayleigh_ritz               = true;
    solver.use_relative_rnorm_tolerance            = true;
    solver.use_adaptive_inner_tolerance            = true;
    solver.use_coarse_inner_preconditioner         = use_coarse_inner_preconditioner;
    solver.use_rayleigh_quotients_instead_of_evals = true;
    solver.use_shifted_jd_eigenvalue               = use_shifted_jd_eigenvalue;
    solver.use_h2_inner_product                    = use_h2_inner_product;
    solver.use_jd_initial_guess                    = use_jd_initial_guess;
    solver.use_h1h2_jcb_preconditioner             = use_h1h2_preconditioner;
    solver.dev_skipjcb                             = skipjcb;
    solver.dev_thick_jd_projector                  = dev_thick_jd_projector;
    solver.residual_correction_type                = rct;
    solver.inject_randomness                       = false;
    solver.tag                                     = tag;

    t_preamble.toc();
    solver.run();

    if(!has_flag(solver.status.stopReason, StopReason::converged_rNorms) and solver.max_iters > 10)
        tools::log->info("GD+k: status.stopReason = {}", solver.status.stopMessage);

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
    auto jcb_bs = opt_meta.eigs_jcbMaxBlockSize.value_or(settings::precision::eigs_jcb_blocksize_max);
    auto jcb_os = opt_meta.eigs_jcbOverlapSize.value_or(settings::precision::eigs_jcb_overlap_size);
    auto prt    = eig::StringToPreconditioner(opt_meta.eigs_preconditioner_type.value_or("SOLVE"));
    auto rct    = StringToResidualCorrection(opt_meta.eigs_residual_correction_type.value_or("JACOBI_DAVIDSON"));
    // bool         crs        = opt_meta.eigs_use_coarse_inner_preconditioner.value_or(false);
    Eigen::Index block_size = opt_meta.eigs_blk.value_or(settings::precision::eigs_blk_min);
    Eigen::Index ncv        = opt_meta.eigs_ncv.value_or(settings::precision::eigs_ncv_min);
    if(jcb_bs == 1)
        return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false, false, block_size, ncv,
                                      "jcb=1", elog);
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

    if(opt_meta.optAlgo == OptAlgo::XDMRG)
        return eigs_gdplusk_bc_std<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false, false,
                                             BasisChangeScale::NONE, 1, ncv * 1, "NONE    BC:JD b1 L2 h2", elog); // A:JD b1 L2 h2
    if(opt_meta.optAlgo == OptAlgo::GDMRG) {
        auto ewt_a = EnvWeightType::NO_PSI_TRACE;
        auto ewt_b = EnvWeightType::AB_TRACE;
        auto ewt_c = EnvWeightType::NO_PSI_SUM;
        auto ewt_d = EnvWeightType::WITH_PSI_TRACE;
        auto ewt_e = EnvWeightType::WITH_PSI_SUM;
        auto ewt_f = EnvWeightType::ONES;
        auto ewt_0 = EnvWeightType::OFF;

        auto ewr_a = EnvWeightRegularizer::NONE;
        auto ewr_b = EnvWeightRegularizer::NORM;
        auto ewr_c = EnvWeightRegularizer::MAX;
        auto ewr_d = EnvWeightRegularizer::SUM;
        auto ewr_e = EnvWeightRegularizer::MEAN;

        auto eat_a = EnvAggregateType::PLAIN;
        auto eat_b = EnvAggregateType::H2_zip;
        auto eat_c = EnvAggregateType::M1;
        auto eat_d = EnvAggregateType::M2;
        auto eat_e = EnvAggregateType::M2_inv;
        auto eat_f = EnvAggregateType::H2_inv;

        auto sym_a = SymmetrizeAggregates::OFF;
        auto sym_b = SymmetrizeAggregates::ON;

        auto tst_a = TransformSpectrumType::EnvAggregateSpectrum;
        auto tst_b = TransformSpectrumType::EnvProjectedDiagonal;

        static auto bcfg_a0 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_0, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a1 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a2 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a3 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a4 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a5 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a6 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a7 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};
        static auto bcfg_a8 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a, .sym = sym_a, .tst = tst_b};

        // static auto bcfg_a2 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a,.sym =sym_a,  .tst=tst_b};
        // static auto bcfg_a3 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_a, .ewr = ewr_a, .eat = eat_a,.sym = sym_a,  .tst=tst_b};
        // static auto bcfg_b1 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_b, .ewr = ewr_a, .eat = eat_a,.sym = sym_a,  .tst=tst_b};
        // static auto bcfg_b2 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_b, .ewr = ewr_a, .eat = eat_a,.sym = sym_a,  .tst=tst_b};
        // static auto bcfg_b3 = BasisChangeConfig{.alpha = 1.00, .ewt = ewt_b, .ewr = ewr_a, .eat = eat_a,.sym = sym_a,  .tst=tst_b};

        // auto resultA0 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false,
        // false, true, false, 1, ncv * 1, bcfg_a0, "A0 JD H2 h2", elog);
        auto resultA1 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, true, false, false, false, false,
                                                      true, false, 1, ncv * 1, bcfg_a1, "A1 JD H2 h2", elog);
        // auto resultA2 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false,
        // false, true, false, 1, ncv * 1, bcfg_a2, "A2 JD H2 h2", elog);

        // auto result_a1 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false, false, true, false, false, false, false,
        // true, 1, 2, bcfg_a1, "a1 JD H2 h2", elog); auto result_a2 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false,
        // false, true, false, false, false, false, true, 1, 3, bcfg_a2, "a2 JD H2 h2", elog); auto result_a3 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors,
        // opt_meta, 128, 16, 1, prt, rct, false, false, true, false, false, false, false, true, 1, 4, bcfg_a3, "a3 JD H2 h2", elog); auto result_a4 =
        // eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false, false, true, false, false, false, false, true, 1, 8, bcfg_a4,
        // "a4 JD H2 h2", elog); auto result_a5 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false, false, true, false,
        // false, false, false, true, 1, 16,bcfg_a5, "a5 JD H2 h2", elog);

        // auto result_a2 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, true, false, true, false, false, false, false, true,
        // 1, ncv * 1,  bcfg_a2, "a2 JD H2 h2", elog); auto result_a3 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false,
        // false, false, false, false, false, false, true, 1, ncv * 1,  bcfg_a4, "a3 JD L2 h2", elog); auto result_a4 = eigs_gdplusk_bc_gen<CalcType>(initial,
        // tensors, opt_meta, 128, 16, 1, prt, rct, true, false, true, false, false, false, false, false, 1, ncv * 1, bcfg_a3, "a4 JD H2 h1-h2", elog); auto
        // result_a5 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false, false, false, false, false, false, false, true, 1,
        // ncv * 1, bcfg_a5, "a5 JD L2 h1-h2", elog); auto result_a6 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, false,
        // false, false, false, false, false, false, false, 1, ncv * 1, bcfg_a6, "a6 JD b1 L2 h2", elog); auto result_a7 =
        // eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, true, false, false, false, false, false, false, true, 1, ncv * 1,
        // bcfg_a7, "a7 JD b1 L2 h2", elog); auto result_a8 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, 128, 16, 1, prt, rct, true, false,
        // false, false, false, false, false, false, 1, ncv * 1, bcfg_a8, "a8 JD b1 L2 h2", elog);

        // auto result_a3 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false,
        // false, 1, ncv * 1, bcfg_a3, "a3 BC:JD b1 L2 h2", elog); auto result_b1 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1,
        // prt, rct, false, false, false, false, false, false, false, 1, ncv * 1, bcfg_b1, "b1 BC:JD b1 L2 h2", elog); auto result_b2 =
        // eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false, false, 1, ncv * 1,
        // bcfg_b2, "b2 BC:JD b1 L2 h2", elog); auto result_b3 = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false,
        // false, false, false, false, false, false, 1, ncv * 1, bcfg_b3, "b3 BC:JD b1 L2 h2", elog);

        // if(initial.get_tensor().size() <= 2048) {
        //     auto resultAf = eigs_gdplusk_bc_gen<CalcType>(initial, tensors, opt_meta, jcb_bs, jcb_os, 1, prt, rct, false, false, false, false, false, false,
        //                                                   false, 1, ncv * 1, bcfg_a, "f BC:JD b1 L2 h2", elog); // A:JD b1 L2 h2
        // }
        //
        // static size_t A1_num_mv = 0;
        // static double A1_time_mv = 0;
        // A1_num_mv += resultA1.front().get_mv();
        // A1_time_mv += tid::get("A1 JD H2 h2").get_last_interval() ;

        // static size_t A2_num_mv = 0;
        // static double A2_time_mv = 0;
        // A2_num_mv += resultA2.front().get_mv();
        // A2_time_mv += tid::get("A2 JD H2 h2").get_last_interval() ;

        // bcfg_a1.num_mv += result_a1.front().get_mv();
        // bcfg_a2.num_mv += result_a2.front().get_mv();
        // bcfg_a3.num_mv += result_a3.front().get_mv();
        // bcfg_a4.num_mv += result_a4.front().get_mv();
        // bcfg_a5.num_mv += result_a5.front().get_mv();
        // bcfg_a6.num_mv += result_a6.front().get_mv();
        // bcfg_a7.num_mv += result_a7.front().get_mv();
        // bcfg_a8.num_mv += result_a8.front().get_mv();

        // bcfg_a1.time_mv += tid::get("a1 JD H2 h2") .get_last_interval() ;
        // bcfg_a2.time_mv += tid::get("a2 JD H2 h2") .get_last_interval() ;
        // bcfg_a3.time_mv += tid::get("a3 JD H2 h2") .get_last_interval() ;
        // bcfg_a4.time_mv += tid::get("a4 JD H2 h2").get_last_interval() ;
        // bcfg_a5.time_mv += tid::get("a5 JD H2 h2").get_last_interval() ;
        // bcfg_a6.time_mv += tid::get("a6 JD b1 L2 h2").get_last_interval() ;
        // bcfg_a7.time_mv += tid::get("a7 JD b1 L2 h2").get_last_interval() ;
        // bcfg_a8.time_mv += tid::get("a8 JD b1 L2 h2").get_last_interval() ;

        // tools::log->info("A1    : mv num = {}, time {:.3e} s", A1_num_mv, A1_time_mv);
        // tools::log->info("A2    : mv num = {}, time {:.3e} s", A2_num_mv, A2_time_mv);
        // tools::log->info("bcfg_a1: mv num = {}, time {:.3e} s", bcfg_a1.num_mv, bcfg_a1.time_mv);
        // tools::log->info("bcfg_a2: mv num = {}, time {:.3e} s", bcfg_a2.num_mv, bcfg_a2.time_mv);
        // tools::log->info("bcfg_a3: mv num = {}, time {:.3e} s", bcfg_a3.num_mv, bcfg_a3.time_mv);
        // tools::log->info("bcfg_a4: mv num = {}, time {:.3e} s", bcfg_a4.num_mv, bcfg_a4.time_mv);
        // tools::log->info("bcfg_a5: mv num = {}, time {:.3e} s", bcfg_a5.num_mv, bcfg_a5.time_mv);
        // tools::log->info("bcfg_a6: mv num = {}, time {:.3e} s", bcfg_a6.num_mv, bcfg_a6.time_mv);
        // tools::log->info("bcfg_a7: mv num = {}, time {:.3e} s", bcfg_a7.num_mv, bcfg_a7.time_mv);
        // tools::log->info("bcfg_a8: mv num = {}, time {:.3e} s", bcfg_a8.num_mv, bcfg_a8.time_mv);

        // return resultA0;
        return resultA1;
        // return resultA2;
    }

    // auto result5 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, true, false, "JD b", elog);
    // auto result6 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, true, true, "JD b h1h2", elog);
    // auto result7 = eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, false, true, "JD h1h2", elog);
    // return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, false, false, "JD", elog);
    // return eigs_gdplusk<CalcType>(initial, tensors, opt_meta, jcb, prt, rct, "JD");
    throw except::runtime_error("opt_meta.optAlgo not handled: {}", enum2sv(opt_meta.optAlgo));
}
