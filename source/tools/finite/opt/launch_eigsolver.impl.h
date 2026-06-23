#pragma once

#include "config/settings.h"
#include "launch_eigsolver.h"
#include "launch_gdplusk.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "precond/generalized_basis_change.h"
#include "precond/standard_basis_change.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include <algorithm>
#include <cmath>
#include <grit/enums.h>
#include <grit/grit.h>
#include <limits>
#include <string_view>

namespace tools::finite::opt::internal {

    template<typename CalcType>
    grit::Ritz get_grit_ritz(OptRitz ritz) {
        switch(ritz) {
            case OptRitz::SR: return grit::Ritz::SR;
            case OptRitz::LR: return grit::Ritz::LR;
            case OptRitz::LM: return grit::Ritz::LM;
            case OptRitz::SM: return grit::Ritz::SM;
            default: throw except::runtime_error("GRIT eigensolver does not support optRitz {}", enum2sv(ritz));
        }
    }

    inline grit::ResidualCorrectionType get_grit_residual_correction(std::string_view rct) {
        if(rct == "NONE") return grit::ResidualCorrectionType::NONE;
        if(rct == "CHEAP_OLSEN") return grit::ResidualCorrectionType::CHEAP_OLSEN;
        if(rct == "FULL_OLSEN") return grit::ResidualCorrectionType::FULL_OLSEN;
        if(rct == "JACOBI_DAVIDSON") return grit::ResidualCorrectionType::JACOBI_DAVIDSON;
        if(rct == "AUTO") return grit::ResidualCorrectionType::AUTO;
        return grit::ResidualCorrectionType::NONE;
    }

    inline bool grit_residual_correction_can_use_jd(grit::ResidualCorrectionType rct) {
        return rct == grit::ResidualCorrectionType::JACOBI_DAVIDSON || rct == grit::ResidualCorrectionType::AUTO;
    }

    inline void assert_valid_grit_preconditioner(grit::ResidualCorrectionType rct, eig::Preconditioner preconditioner) {
        if(grit_residual_correction_can_use_jd(rct) && preconditioner == eig::Preconditioner::SOLVE) {
            throw except::runtime_error("Invalid GRIT configuration: Jacobi-Davidson residual correction must use JACOBI preconditioner type, never SOLVE");
        }
    }

    inline Eigen::Index get_grit_valid_ncv(Eigen::Index size, Eigen::Index nev, Eigen::Index ncv, Eigen::Index &block_size) {
        if(size < 1) throw except::runtime_error("GRIT eigensolver got empty operator");
        nev        = std::clamp<Eigen::Index>(nev, 1, size);
        block_size = std::clamp<Eigen::Index>(block_size, 1, size);
        ncv        = std::clamp<Eigen::Index>(std::max({ncv, nev, block_size}), 1, size);

        if(ncv % block_size == 0) return ncv;

        auto ncv_floor = ncv - ncv % block_size;
        if(ncv_floor >= nev and ncv_floor >= block_size) return ncv_floor;

        auto ncv_ceil = ((std::max(nev, block_size) + block_size - 1) / block_size) * block_size;
        if(ncv_ceil <= size) return ncv_ceil;

        block_size = 1;
        return std::clamp<Eigen::Index>(std::max(ncv, nev), 1, size);
    }

    template<typename CalcType>
    void transform_grit_result_to_original_basis(grit::Result<CalcType> &result, const std::array<Eigen::Index, 3> &shape_tilde,
                                                 const Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic> &TL,
                                                 const Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic> &TR) {
        if(result.eigVecs().size() == 0) return;
        result.eigVecs() = tools::finite::opt::precond::common::transform_matrix(result.eigVecs(), shape_tilde, TL, TR);
        for(Eigen::Index idx = 0; idx < result.eigVecs().cols(); ++idx) result.eigVecs().col(idx).normalize();
    }

    template<typename CalcType, typename BasisChangeType>
    void transform_grit_result_to_original_basis(grit::Result<CalcType> &result, const BasisChangeType &bc) {
        auto TLc = tenx::asScalarType<CalcType>(bc.TL);
        auto TRc = tenx::asScalarType<CalcType>(bc.TR);
        transform_grit_result_to_original_basis(result, bc.shape_tilde, TLc, TRc);
    }

    template<typename CalcType, typename Scalar>
    void launch_grit_folded_spectrum(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                     [[maybe_unused]] reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results) {
        using RealScalar = tools::finite::opt::RealScalar<CalcType>;
        using GritSolver = grit::standard::gdplusk<CalcType>;

        auto t_grit = tid::tic_scope("grit-folded-spectrum");
        auto sites  = initial_mps.get_sites();
        auto mpos   = tensors.get_model().get_mpo(sites);
        auto enve   = tensors.get_edges().get_multisite_env_ene(sites);
        auto envv   = tensors.get_edges().get_multisite_env_var(sites);
        auto vh1v   = tools::finite::measure::expval_hamiltonian(initial_mps.get_tensor(), mpos, enve);
        auto vh2v   = tools::finite::measure::expval_hamiltonian_squared(initial_mps.get_tensor(), mpos, envv);
        auto scale  = std::abs(vh1v / vh2v);
        auto alpha  = RealScalar{0.5f};
        auto bc     = tools::finite::opt::precond::standard::BasisChange<Scalar>(initial_mps, tensors, BasisChangeScale::NONE, scale, alpha);

        auto hamiltonian_squared          = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());
        hamiltonian_squared.factorization = eig::Factorization::LLT;

        auto nev        = meta.eigs_nev.value_or(settings::solvers::eig::nev_min);
        auto block_size = meta.eigs_blk.value_or(settings::solvers::eig::blk_min);
        auto ncv        = meta.eigs_ncv.value_or(settings::solvers::eig::ncv_min);
        auto size       = safe_cast<Eigen::Index>(initial_mps.get_tensor().size());
        nev             = std::clamp<Eigen::Index>(nev, 1, size);
        ncv             = get_grit_valid_ncv(size, nev, ncv, block_size);
        auto tol        = meta.eigs_abstol.has_value() ? static_cast<RealScalar>(meta.eigs_abstol.value()) : std::numeric_limits<RealScalar>::epsilon() * 10000;
        auto ritz       = get_grit_ritz<CalcType>(meta.optRitz);
        auto rct        = get_grit_residual_correction(meta.eigs_residual_correction_type.value_or("AUTO"));
        auto jcb        = meta.eigs_jcbMaxBlockSize.value_or(0);

        if(jcb > 0) {
            auto preconditioner = eig::StringToPreconditioner(meta.eigs_preconditioner_type.value_or("JACOBI"));
            assert_valid_grit_preconditioner(rct, preconditioner);
            hamiltonian_squared.preconditioner = preconditioner;
            hamiltonian_squared.set_jcbMaxBlockSize(jcb);
            hamiltonian_squared.set_jcbOverlapSize(meta.eigs_jcbOverlapSize.value_or(settings::solvers::eig::jcb_overlap_size));
            hamiltonian_squared.set_jcbNumPasses(1);
        }

        auto op = grit::matvec<CalcType>(
            hamiltonian_squared.get_size(),
            [&hamiltonian_squared](const Eigen::Ref<const typename grit::Matvec<CalcType>::MatrixType> &X) { return hamiltonian_squared.MultAX(X); });

        if(jcb > 0) {
            op.set_preconditioner_update([&hamiltonian_squared](RealScalar) { hamiltonian_squared.CalcPc(); });
            op.set_preconditioner_apply([&hamiltonian_squared](const Eigen::Ref<const typename grit::Matvec<CalcType>::VectorType> &x,
                                                               Eigen::Ref<typename grit::Matvec<CalcType>::VectorType>              y,
                                                               RealScalar) { hamiltonian_squared.MultPc(x.data(), y.data()); });
        }

        GritSolver solver(op);
        solver.set_initial_guess(bc.initial_guess.template get_tensor_as_matrix<CalcType>());
        solver.config.nev                          = nev;
        solver.config.ncv                          = ncv;
        solver.config.block_size                   = block_size;
        solver.config.abstol                       = tol;
        solver.config.reltol                       = meta.eigs_reltol.has_value() ? static_cast<RealScalar>(meta.eigs_reltol.value()) : RealScalar{0};
        solver.config.max_iters                    = meta.eigs_iter_max.value_or(settings::solvers::eig::iter_min);
        solver.config.max_matvecs                  = -1;
        solver.config.ritz                         = ritz;
        solver.config.use_refined_rayleigh_ritz    = true;
        solver.config.use_rescaled_rnorm_tolerance = true;
        solver.config.use_adaptive_inner_tolerance = true;
        solver.config.use_rayleigh_quotients_instead_of_evals = true;
        solver.config.use_krylov_schur_gdplusk_restart        = true;
        solver.config.maxRetainBlocks                         = std::max<Eigen::Index>(1, 3 * ncv / (8 * block_size));
        solver.config.max_ritz_residual_history               = 1;
        solver.config.max_extra_ritz_history                  = 1;
        solver.config.residual_correction_type                = rct;
        solver.config.log_level                               = spdlog::level::info;
        solver.status.initVal                                 = static_cast<RealScalar>(initial_mps.get_energy());
        solver.tag                                            = "folded";

        tools::log->debug("launch_grit_folded_spectrum: Solving [H²x=λx] GRIT {} | maxIter {} | tol {:.2e} reltol {:.2e} | nev {} ncv {} blk {} | "
                          "size {} | mps {} | jcb {}",
                          grit::enum2sv(ritz), solver.config.max_iters, solver.config.abstol, solver.config.reltol, solver.config.nev, solver.config.ncv,
                          solver.config.block_size, hamiltonian_squared.rows(), hamiltonian_squared.get_shape_mps(), jcb);

        solver.run();

        auto result = solver.get_result();
        transform_grit_result_to_original_basis(result, bc);
        extract_results<CalcType>(tensors, initial_mps, meta, result, solver, hamiltonian_squared, results);
    }

    template<typename CalcType, typename Scalar>
    void launch_grit_generalized_shift_invert(const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                              [[maybe_unused]] reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results) {
        using RealScalar = tools::finite::opt::RealScalar<CalcType>;
        using GritSolver = grit::generalized::gdplusk<CalcType>;

        auto t_grit = tid::tic_scope("grit-gsi");
        auto sites  = initial_mps.get_sites();
        auto mpos   = tensors.get_model().get_mpo(sites);
        auto bcfg   = BasisChangeConfig{
              .alpha = 1.00,
              .ewt   = EnvWeightType::NO_PSI_TRACE,
              .ewr   = EnvWeightRegularizer::NONE,
              .eat   = EnvAggregateType::PLAIN,
              .sym   = SymmetrizeAggregates::OFF,
              .tst   = TransformSpectrumType::EnvProjectedDiagonal,
        };
        auto bc = tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>(initial_mps, tensors, bcfg);

        auto hamiltonian  = MatVecMPOS<CalcType>(mpos, bc.get_enve_pair());
        auto hamiltonian2 = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());
        auto precond      = MatVecMPOS<CalcType>(mpos, bc.get_envv_pair());

        hamiltonian.factorization  = eig::Factorization::LLT;
        hamiltonian2.factorization = eig::Factorization::LLT;
        precond.factorization      = eig::Factorization::LLT;

        auto nev        = meta.eigs_nev.value_or(settings::solvers::eig::nev_min);
        auto block_size = meta.eigs_blk.value_or(settings::solvers::eig::blk_min);
        auto ncv        = meta.eigs_ncv.value_or(settings::solvers::eig::ncv_min);
        auto size       = safe_cast<Eigen::Index>(initial_mps.get_tensor().size());
        nev             = std::clamp<Eigen::Index>(nev, 1, size);
        ncv             = get_grit_valid_ncv(size, nev, ncv, block_size);
        auto tol        = meta.eigs_abstol.has_value() ? static_cast<RealScalar>(meta.eigs_abstol.value()) : std::numeric_limits<RealScalar>::epsilon() * 10000;
        auto ritz       = get_grit_ritz<CalcType>(meta.optRitz);
        auto rct        = get_grit_residual_correction(meta.eigs_residual_correction_type.value_or("AUTO"));
        auto jcb        = meta.eigs_jcbMaxBlockSize.value_or(0);

        if(jcb > 0) {
            auto preconditioner = eig::StringToPreconditioner(meta.eigs_preconditioner_type.value_or("JACOBI"));
            assert_valid_grit_preconditioner(rct, preconditioner);
            precond.preconditioner = preconditioner;
            precond.set_jcbMaxBlockSize(jcb);
            precond.set_jcbOverlapSize(meta.eigs_jcbOverlapSize.value_or(settings::solvers::eig::jcb_overlap_size));
            precond.set_jcbNumPasses(1);
        }

        auto A = grit::matvec<CalcType>(
            hamiltonian.get_size(), [&hamiltonian](const Eigen::Ref<const typename grit::Matvec<CalcType>::MatrixType> &X) { return hamiltonian.MultAX(X); });
        auto B = grit::matvec<CalcType>(hamiltonian2.get_size(), [&hamiltonian2](const Eigen::Ref<const typename grit::Matvec<CalcType>::MatrixType> &X) {
            return hamiltonian2.MultAX(X);
        });

        if(jcb > 0) {
            A.set_preconditioner_update([&precond](RealScalar) { precond.CalcPc(); });
            A.set_preconditioner_apply([&precond](const Eigen::Ref<const typename grit::Matvec<CalcType>::VectorType> &x,
                                                  Eigen::Ref<typename grit::Matvec<CalcType>::VectorType>              y,
                                                  RealScalar) { precond.MultPc(x.data(), y.data()); });
        }

        GritSolver solver(A, B);
        solver.set_initial_guess(bc.initial_guess.template get_tensor_as_matrix<CalcType>());
        solver.config.nev                          = nev;
        solver.config.ncv                          = ncv;
        solver.config.block_size                   = block_size;
        solver.config.abstol                       = tol;
        solver.config.reltol                       = meta.eigs_reltol.has_value() ? static_cast<RealScalar>(meta.eigs_reltol.value()) : RealScalar{0};
        solver.config.max_iters                    = meta.eigs_iter_max.value_or(settings::solvers::eig::iter_min);
        solver.config.max_matvecs                  = -1;
        solver.config.ritz                         = ritz;
        solver.config.use_b_inner_product          = true;
        solver.config.use_jd_b_only                = true;
        solver.config.use_refined_rayleigh_ritz    = true;
        solver.config.use_rescaled_rnorm_tolerance = true;
        solver.config.use_adaptive_inner_tolerance = true;
        solver.config.use_rayleigh_quotients_instead_of_evals = true;
        solver.config.use_krylov_schur_gdplusk_restart        = true;
        solver.config.maxRetainBlocks                         = std::max<Eigen::Index>(1, 3 * ncv / (8 * block_size));
        solver.config.max_ritz_residual_history               = 1;
        solver.config.max_extra_ritz_history                  = 1;
        solver.config.residual_correction_type                = rct;
        solver.config.log_level                               = spdlog::level::trace;
        solver.status.initVal                                 = static_cast<RealScalar>(initial_mps.get_energy());
        solver.tag                                            = "gsi";

        // --ritz=LM
        // --reps=1
        // --seed=0
        // --max-iters=-1
        // --inner-max-iters=2000
        // --tol=1e-12
        // --ncv=[12]
        // --block-size=[1]
        // --residual-correction=[auto]
        // --refined-rayleigh-ritz=[true]
        // --log-level=debug
        // --save-results=results-bcs.h5
        // --auto-sat-eigval-threshold=1e-3
        // --auto-sat-rnorm-threshold=1e-2
        // --auto-cheap-probe-interval=3
        // --auto-jd-start-rnorm-threshold=1e-5
        // --use-relative-rnorm-tolerance=true
        // --use-adaptive-inner-tolerance=true

        tools::log->debug("launch_grit_generalized_shift_invert: Solving [Hx=λH²x] GRIT {} | maxIter {} | tol {:.2e} reltol {:.2e} | nev {} ncv "
                          "{} blk {} | size {} | mps {} | jcb {} | b-inner {} | jd-b-only {}",
                          grit::enum2sv(ritz), solver.config.max_iters, solver.config.abstol, solver.config.reltol, solver.config.nev, solver.config.ncv,
                          solver.config.block_size, hamiltonian.rows(), hamiltonian.get_shape_mps(), jcb, solver.config.use_b_inner_product,
                          solver.config.use_jd_b_only);

        solver.run();

        auto result = solver.get_result();
        transform_grit_result_to_original_basis(result, bc);
        extract_results<CalcType>(tensors, initial_mps, meta, result, solver, hamiltonian, hamiltonian2, precond, results);
    }

    template<typename CalcType, typename Scalar>
    bool launch_eigsolver_folded_spectrum(eig::Lib lib, const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                          reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results) {
        switch(lib) {
            case eig::Lib::EIGSMPO: results = eigs_gdplusk<CalcType>(tensors, initial_mps, meta, elog); return true;
            case eig::Lib::GRIT: launch_grit_folded_spectrum<CalcType>(tensors, initial_mps, meta, elog, results); return true;
            case eig::Lib::ARPACK:
            case eig::Lib::PRIMME:
            case eig::Lib::SPECTRA: return false;
        }
        throw except::runtime_error("Unhandled eigensolver library: {}", eig::LibToString(lib));
    }

    template<typename CalcType, typename Scalar>
    bool launch_eigsolver_generalized_shift_invert(eig::Lib lib, const TensorsFinite<Scalar> &tensors, const opt_mps<Scalar> &initial_mps, const OptMeta &meta,
                                                   reports::eigs_log<Scalar> &elog, std::vector<opt_mps<Scalar>> &results) {
        switch(lib) {
            case eig::Lib::EIGSMPO: results = eigs_gdplusk<CalcType>(tensors, initial_mps, meta, elog); return true;
            case eig::Lib::GRIT: launch_grit_generalized_shift_invert<CalcType>(tensors, initial_mps, meta, elog, results); return true;
            case eig::Lib::ARPACK:
            case eig::Lib::PRIMME:
            case eig::Lib::SPECTRA: return false;
        }
        throw except::runtime_error("Unhandled eigensolver library: {}", eig::LibToString(lib));
    }

}
