#include "math/tenx.h"
// -- (textra first)
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/eig.h"
#include "math/eig/enums.h"
#include "math/eig/matvec/matvec_dense.h"
#include "math/eig/matvec/matvec_mpo.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include "tools/finite/opt_meta.h"
#include "tools/finite/opt_mps.h"
//
#include <Eigen/QR>
#include <general/sfinae.h>
#include <primme/primme.h>

namespace tools::finite::opt::internal::subspace {
    template<typename Scalar>
    void preconditioner_jacobi(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
        if(x == nullptr) return;
        if(y == nullptr) return;
        if(primme == nullptr) return;
        const auto H_ptr      = static_cast<MatVecMPOS<Scalar> *>(primme->matrix);
        H_ptr->preconditioner = eig::Preconditioner::JACOBI;
        H_ptr->MultPc(x, ldx, y, ldy, blockSize, primme, ierr);
    }
}

using namespace tools::finite::opt;
using namespace tools::finite::opt::internal;

std::vector<int> subspace::generate_nev_list(int rows) {
    std::vector<int> nev_list = {4, 8};
    if(32 < rows and rows <= 64) nev_list = {16, 32};
    if(64 < rows and rows <= 128) nev_list = {32, 64};
    if(128 < rows and rows <= 512) nev_list = {64, 256};
    if(512 < rows and rows <= 1024) nev_list = {64, 256, 512};
    if(1024 < rows and rows <= 2048) nev_list = {16, 256, 512};
    if(2048 < rows and rows <= 3072) nev_list = {16, 256};
    if(3072 < rows and rows <= 4096) nev_list = {16};
    if(4096 < rows) nev_list = {4};

    while(nev_list.size() > 1 and (nev_list.back() * 2 > rows or safe_cast<size_t>(nev_list.back()) > settings::precision::max_subspace_size))
        nev_list.pop_back();
    if(nev_list.empty()) throw except::logic_error("nev_list is empty");
    return nev_list;
}

template<typename Scalar>
std::vector<opt_mps> subspace::find_subspace(const TensorsFinite &tensors, const OptMeta &meta) {
    const auto &state = *tensors.state;
    const auto &model = *tensors.model;
    tools::log->trace("find_subspace ...");
    auto t_find = tid::tic_scope("find");

    Eigen::MatrixXcd eigvecs;
    Eigen::VectorXd  eigvals;
    // Determine the eigval target (shift)
    double eigval_target = 0;
    if(meta.eigv_target.has_value()) {
        eigval_target = meta.eigv_target.value();
    } else {
        switch(meta.optAlgo) {
            case OptAlgo::HYBRID_DMRGX: {
                // We are trying to find a subspace close to the current energy, on which to optimize the variance.
                switch(meta.optRitz) {
                    case OptRitz::NONE: throw std::logic_error("find_subspace: invalid OptRitz::NONE");
                    case OptRitz::SR: [[fallthrough]];
                    case OptRitz::LM: [[fallthrough]];
                    case OptRitz::LR: {
                        if(model.has_energy_shifted_mpo()) {
                            eigval_target = tools::finite::measure::energy_minus_energy_shift(tensors); // E-Eshf Should be close to 0
                        } else {
                            eigval_target = tools::finite::measure::energy(tensors);
                        }
                        break;
                    }

                    // Subspace of energy eigenpairs close to eigval == 0.
                    // Note that mpos may have an in-built energy shift already.
                    case OptRitz::IS: [[fallthrough]];
                    case OptRitz::TE: {
                        if(model.has_energy_shifted_mpo()) {
                            eigval_target = 0.0; // E-Eshf Should be close to 0
                        } else {
                            eigval_target = tools::finite::measure::energy(tensors);
                        }
                        break;
                    }
                    case OptRitz::SM: {
                        eigval_target = 0.0;
                        break;
                    }
                }
                break;
            }
            // case OptAlgo::DMRG:
            // case OptAlgo::DMRGX: {
            //     // We are trying
            //     break;
            // }
            default: break;
        }
    }

    // If the mps is small enough you can afford full diag.
    if(tensors.state->active_problem_size() <= settings::precision::eig_max_size) {
        std::tie(eigvecs, eigvals) = find_subspace_lapack<Scalar>(tensors);
    } else {
        // {
        // auto [eigvecs_test, eigvals_test] = find_subspace_lapack<Scalar>(tensors);
        // fmt::print("primme eigvals\n");
        // for(auto &&[idx, eigv] : iter::enumerate(eigvals_test)) { fmt::print("idx {:3} | {:.16f}\n", idx, eigv); }
        // }
        std::tie(eigvecs, eigvals) = find_subspace_primme<Scalar>(tensors, eigval_target, meta);

        // fmt::print("primme eigvals\n");
        // for(auto &&[idx, eigv] : iter::enumerate(eigvals)) { fmt::print("idx {:3} | {:.16f}\n", idx, eigv); }
    }
    /* clang-format off */
    tools::log->trace("Eigval range         : {:.16f} --> {:.16f}", eigvals.minCoeff(), eigvals.maxCoeff());
    tools::log->trace("Energy range         : {:.16f} --> {:.16f}", eigvals.minCoeff() + model.get_energy_shift_mpo(), eigvals.maxCoeff() + model.get_energy_shift_mpo());
    /* clang-format on */
    reports::print_subs_report();

    if constexpr(std::is_same<Scalar, double>::value) {
        tenx::subtract_phase(eigvecs);
        double trunc = eigvecs.imag().cwiseAbs().sum();
        if(trunc > 1e-12) tools::log->warn("truncating imag of eigvecs, sum: {}", trunc);
        eigvecs = eigvecs.real();
    }
    const auto     &multisite_mps = state.get_multisite_mps<cx64>();
    const auto      multisite_vec = Eigen::Map<const Eigen::VectorXcd>(multisite_mps.data(), multisite_mps.size());
    auto            energy_shift  = model.get_energy_shift_mpo();
    Eigen::VectorXd overlaps      = (multisite_vec.adjoint() * eigvecs).cwiseAbs().real();

    double eigvec_time = 0;
    for(const auto &item : reports::subs_log) { eigvec_time += item.ham_time + item.lu_time + item.eig_time; }

    std::vector<opt_mps> subspace;
    subspace.reserve(safe_cast<size_t>(eigvals.size()));
    for(long idx = 0; idx < eigvals.size(); idx++) {
        // Important to normalize the eigenvectors that we get from the solver: they are not always well normalized when we get them!
        auto eigvec_i = tenx::TensorCast(eigvecs.col(idx).normalized(), multisite_mps.dimensions());
        subspace.emplace_back(fmt::format("eigenvector {}", idx), eigvec_i, tensors.active_sites, energy_shift, eigvals(idx), std::nullopt, overlaps(idx),
                              tensors.get_length());
        subspace.back().is_basis_vector = true;
        subspace.back().set_time(eigvec_time);
        subspace.back().set_mv(reports::subs_log.size());
        subspace.back().set_iter(reports::subs_log.size());
        subspace.back().set_eigs_idx(idx);
        subspace.back().set_eigs_eigval(eigvals(idx));
        subspace.back().set_eigs_ritz(enum2sv(meta.optRitz));
        subspace.back().set_optsolver(meta.optSolver);
        subspace.back().set_optalgo(meta.optAlgo);
        subspace.back().validate_basis_vector();
    }
    return subspace;
}

template std::vector<opt_mps> subspace::find_subspace<cx64>(const TensorsFinite &tensors, const OptMeta &meta);

template std::vector<opt_mps> subspace::find_subspace<fp64>(const TensorsFinite &tensors, const OptMeta &meta);

template<typename Scalar>
std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_part(const TensorsFinite &tensors, double energy_target, const OptMeta &meta) {
    tools::log->trace("Finding subspace -- partial");
    auto  t_iter = tid::tic_scope("part");
    auto &t_lu   = tid::get("lu_decomp");

    // Initial mps and a vector map
    const auto                        &multisite_mps = tensors.state->get_multisite_mps<cx64>();
    Eigen::Map<const Eigen::VectorXcd> multisite_vector(multisite_mps.data(), multisite_mps.size());
    const auto                         problem_size = multisite_mps.size();

    // Mutable initial mps vector used for initial guess in arpack
    Eigen::Tensor<Scalar, 3> init = tensors.state->get_multisite_mps<Scalar>();

    // Get the local effective Hamiltonian as a matrix
    const auto &effective_hamiltonian = tensors.get_effective_hamiltonian<Scalar>();
    double      time_ham              = tid::get("ham").get_last_interval();

    // Create the dense matrix object for the eigenvalue solver
    MatVecDense<Scalar> hamiltonian(effective_hamiltonian.data(), effective_hamiltonian.dimension(0), false, eig::Form::SYMM, eig::Side::R);

    // Create a reusable config for multiple nev trials
    eig::settings config;
    config.tol             = settings::precision::eigs_tol_min;
    config.sigma           = cx64(energy_target, 0.0);
    config.shift_invert    = eig::Shinv::ON;
    config.compute_eigvecs = eig::Vecs::ON;
    config.ritz            = eig::Ritz::LM;
    config.initial_guess.push_back({init.data(), 0});
    std::string reason = "exhausted";

    // Initialize eigvals/eigvecs containers that store the results
    Eigen::VectorXd  eigvals;
    Eigen::MatrixXcd eigvecs;
    for(auto nev : generate_nev_list(safe_cast<int>(problem_size))) {
        eig::solver solver;
        solver.config        = config;
        solver.config.maxNev = nev;
        // Set the new initial guess if we are doing one more round
        if(eigvecs.cols() != 0) {
            solver.config.initial_guess.clear();
            for(long n = 0; n < eigvecs.cols(); n++) { solver.config.initial_guess.push_back({eigvecs.col(n).data(), n}); }
        }

        solver.eigs(hamiltonian);
        t_lu += *hamiltonian.t_factorOP;

        eigvals = eig::view::get_eigvals<fp64>(solver.result);
        eigvecs = eig::view::get_eigvecs<cx64>(solver.result, eig::Side::R);

        // Check the quality of the subspace
        Eigen::VectorXd overlaps       = (multisite_vector.adjoint() * eigvecs).cwiseAbs().real();
        double          max_overlap    = overlaps.maxCoeff();
        double          min_overlap    = overlaps.minCoeff();
        double          sq_sum_overlap = overlaps.cwiseAbs2().sum();
        double          subspace_error = 1.0 - sq_sum_overlap;
        reports::subs_add_entry(nev, max_overlap, min_overlap, subspace_error, solver.result.meta.time_total, time_ham, t_lu.get_last_interval(),
                                solver.result.meta.iter, solver.result.meta.num_mv, solver.result.meta.num_pc);
        time_ham = 0;
        if(max_overlap > 1.0 + 1e-6) throw except::runtime_error("max_overlap larger than one: {:.16f}", max_overlap);
        if(sq_sum_overlap > 1.0 + 1e-6) throw except::runtime_error("eps larger than one: {:.16f}", sq_sum_overlap);
        if(min_overlap < 0.0) throw except::runtime_error("min_overlap smaller than zero: {:.16f}", min_overlap);
        if(subspace_error < meta.subspace_tol.value_or(1e-10)) {
            reason = fmt::format("subspace error is low enough: {:.3e} < tolerance {:.3e}", subspace_error, meta.subspace_tol.value_or(1e-10));
            break;
        }
        if(meta.optAlgo == OptAlgo::DMRGX and sq_sum_overlap >= 1.0 / std::sqrt(2.0)) {
            reason = fmt::format("Overlap is sufficient:  {:.16f} >= threshold {:.16f}", max_overlap, 1.0 / std::sqrt(2.0));
            break;
        }
    }
    tools::log->debug("Finished iterative eigensolver -- reason: {}", reason);
    return {eigvecs, eigvals};
}

template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_part<cx64>(const TensorsFinite &tensors, double energy_target,
                                                                                         const OptMeta &meta);
template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_part<fp64>(const TensorsFinite &tensors, double energy_target,
                                                                                         const OptMeta &meta);

// template<typename Scalar>
// void preconditioner(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *ierr) {
//     if(x == nullptr) return;
//     if(y == nullptr) return;
//     if(primme == nullptr) return;
//     const auto H_ptr = static_cast<MatVecMPO<Scalar> *>(primme->matrix);
//     H_ptr->FactorOP();
//     H_ptr->MultOPv(x, ldx, y, ldy, blockSize, primme, ierr);
// }

template<typename Scalar>
void convTestFun([[maybe_unused]] double *eval, [[maybe_unused]] void *evec, double *rNorm, int *isconv, struct primme_params *primme, int *ierr) {
    if(rNorm == nullptr) return;
    if(primme == nullptr) return;

    double problemNorm;
    if(not primme->massMatrixMatvec) {
        problemNorm = primme->aNorm > 0.0 ? primme->aNorm : primme->stats.estimateLargestSVal;
    } else {
        problemNorm = primme->aNorm > 0.0 && primme->invBNorm > 0.0 ? primme->aNorm * primme->invBNorm : primme->stats.estimateLargestSVal;
    }
    double problemTol = problemNorm * primme->eps;
    if(primme->monitor != nullptr and *rNorm != 0) {
        auto &solver                 = *static_cast<eig::solver *>(primme->monitor);
        auto &config                 = solver.config;
        auto &result                 = solver.result;
        result.meta.problemNorm      = problemNorm;
        result.meta.last_eval        = *eval;
        result.meta.last_rnorm       = *rNorm;
        long   iter_since_last_check = std::abs(result.meta.last_conv_iter - primme->stats.numOuterIterations);
        long   mvec_since_last_check = std::abs(result.meta.last_conv_mvec - primme->stats.numMatvecs);
        double time_since_last_check = std::abs(result.meta.last_conv_time - primme->stats.elapsedTime);
        bool   check_subspace        = config.subspace_tol.has_value() and !std::isnan(config.subspace_tol.value()) and result.meta.subspace_ok == false and
                              (iter_since_last_check > 1000 or mvec_since_last_check > 10000) and time_since_last_check > 30;
        if(check_subspace) {
            using VectorType       = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            using MatrixType       = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
            auto      &eigvecs_vec = result.get_eigvecs<Scalar, eig::Form::SYMM>();
            auto       rows        = static_cast<long>(primme->n);
            auto       cols        = static_cast<long>(primme->numEvals);
            MatrixType eigvecs     = Eigen::Map<MatrixType>(eigvecs_vec.data(), rows, cols);
            eigvecs                = linalg::matrix::modified_gram_schmidt(eigvecs); // Orthogonalize
            std::vector<long> valid_cols;
            for(long col = 0; col < eigvecs.cols(); ++col) {
                if(result.meta.residual_norms.at(static_cast<size_t>(col)) > problemTol) continue;
                if(eigvecs.col(col).isZero()) continue;
                valid_cols.emplace_back(col);
            }

            if(valid_cols.size() >= 2) {
                eigvecs = eigvecs(Eigen::all, valid_cols).colwise().normalized().eval();

                // Check the quality of the subspace
                const auto initial_vec = Eigen::Map<const VectorType>(static_cast<Scalar *>(config.initial_guess.front().ptr), eigvecs.rows());
                // const auto      initial_vec = Eigen::Map<const Eigen::VectorXcd>(static_cast<cx64 *>(config.initial_guess.front().ptr), eigvecs.rows());
                Eigen::VectorXd overlaps = (initial_vec.adjoint() * eigvecs).cwiseAbs().real();

                double max_overlap    = overlaps.maxCoeff();
                double min_overlap    = overlaps.minCoeff();
                double sq_sum_overlap = overlaps.cwiseAbs2().sum();
                double subspace_error = 1.0 - sq_sum_overlap;
                if(max_overlap > 1.0 + 1e-6) tools::log->debug("max_overlap larger than one: {:.16f}", max_overlap);
                if(sq_sum_overlap > 1.0 + 1e-6) tools::log->debug("eps larger than one: {:.16f}", sq_sum_overlap);
                if(min_overlap < 0.0) tools::log->debug("min_overlap smaller than zero: {:.16f}", min_overlap);
                tools::log->debug("subspace_error: {:.16f} | eigvecs: {} x {} (of {})", subspace_error, eigvecs.rows(), eigvecs.cols(), cols);
                result.meta.subspace_ok = subspace_error < config.subspace_tol.value();
            }
            result.meta.last_conv_iter = primme->stats.numOuterIterations;
            result.meta.last_conv_mvec = primme->stats.numMatvecs;
            result.meta.last_conv_time = primme->stats.elapsedTime;
        }

        *isconv = *rNorm < problemTol;

    } else {
        *isconv = 0;
    }

    *ierr = 0;
}

template<typename Scalar>
std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_primme(const TensorsFinite &tensors, double eigval_shift, const OptMeta &meta) {
    tools::log->trace("find_subspace_primme: ritz {} | target eigval {:.16f} | subspace tolerance {:.3e} | type {}", enum2sv(meta.optRitz), eigval_shift,
                      meta.subspace_tol.value_or(1e-10), sfinae::type_name<Scalar>());
    auto t_iter      = tid::tic_scope("part");
    using VectorType = Eigen::Matrix<fp64, Eigen::Dynamic, 1>;                // We always have real eigenvalues
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; // We may have real or complex eigenvectors

    // Initial mps and a vector map
    const auto &initial_mps = tensors.state->get_multisite_mps<cx64>();
    const auto  initial_vec = Eigen::Map<const Eigen::VectorXcd>(initial_mps.data(), initial_mps.size());

    // Mutable initial mps vector used for initial guess
    Eigen::Tensor<Scalar, 3> init = tensors.state->get_multisite_mps<Scalar>();

    // Get the local effective Hamiltonian in mps/mpo form
    const auto &mpos          = tensors.get_model().get_mpo_active();
    const auto &enve          = tensors.get_edges().get_ene_active();
    auto        hamiltonian   = MatVecMPOS<Scalar>(mpos, enve);
    hamiltonian.factorization = eig::Factorization::LU;

    // Create a reusable config for multiple nev trials
    // https://www.cs.wm.edu/~andreas/software/doc/appendix.html#c.primme_params.eps
    eig::settings config;
    config.lib                 = eig::Lib::PRIMME;
    config.tol                 = meta.eigs_tol; // 1e-12 is good. This Sets "eps" in primme, see link above.
    config.maxNev              = meta.eigs_nev;
    config.maxNcv              = meta.eigs_ncv;
    config.shift_invert        = eig::Shinv::OFF;
    config.maxIter             = settings::precision::eigs_iter_max; // We need them to actually converge;
    config.ritz                = eig::Ritz::primme_closest_abs;
    config.primme_targetShifts = {eigval_shift};
    // config.primme_projection    = "primme_proj_refined";
    config.compute_eigvecs    = eig::Vecs::ON;
    config.primme_locking     = 1;
    config.loglevel           = 1;
    config.subspace_tol       = meta.optAlgo == OptAlgo::DMRGX ? 1.0 - 1.0 / std::sqrt(2.0) : meta.subspace_tol;
    config.primme_convTestFun = convTestFun<Scalar>;
    config.primme_method      = eig::PrimmeMethod::PRIMME_DEFAULT_MIN_TIME;
    // if(initial_mps.size() <= settings::precision::eigs_max_size_shift_invert) {
    // Instead of doing shift invert, we can simply set the jacobi block size equal to the matrix size
    // config.jcbMaxBlockSize = settings::precision::eigs_max_size_shift_invert;
    // }

    if(initial_mps.size() <= settings::precision::eigs_max_size_shift_invert) {
        hamiltonian.factorization  = eig::Factorization::LU;
        config.shift_invert        = eig::Shinv::ON;
        config.ritz                = eig::Ritz::primme_largest_abs;
        config.sigma               = cx64(eigval_shift, 0.0);
        config.primme_projection   = "primme_proj_default";
        config.primme_locking      = true;
        config.primme_targetShifts = {};
    }
    // else {
    // config.primme_preconditioner = subspace::preconditioner_jacobi<Scalar>;
    // config.jcbMaxBlockSize       = 1; // meta.eigs_jcbMaxBlockSize;
    // }

    std::string reason = "exhausted";

    // Initialize eigvals/eigvecs containers that store the results
    VectorType eigvals;
    MatrixType eigvecs;
    // tools::log->info("initial: \n{}\n", linalg::matrix::to_string(initial_vec.transpose().real(), 8));
    auto nev_list = std::vector<int>{static_cast<int>(settings::precision::max_subspace_size)};
    // for(int nev = 2; nev <= static_cast<int>(settings::precision::max_subspace_size); nev *= 4) { nev_list.emplace_back(nev); }

    for(auto nev : nev_list) {
        eig::solver solver;
        solver.config         = config;
        solver.config.maxNev  = nev;
        solver.config.maxNcv  = std::max(16, std::max(meta.eigs_ncv.value_or(nev * 2), nev * 2));
        solver.config.maxIter = nev * safe_cast<int>(settings::precision::eigs_iter_max);

        // Set the new initial guess if we are doing one more round
        if(eigvecs.size() == 0) {
            solver.config.initial_guess.push_back({init.data(), 0});
            config.primme_targetShifts = {eigval_shift};
        } else {
            for(long n = 0; n < eigvecs.cols(); ++n) { solver.config.initial_guess.push_back({eigvecs.col(n).data(), n}); }
            // if(solver.config.shift_invert == eig::Shinv::OFF) {
            //     config.primme_target_shifts.clear();
            //     for(long n = 0; n < eigvals.size(); ++n) { config.primme_target_shifts.push_back({eigvals[n]}); }
            // }
        }
        tools::log->trace("Running eigensolver | nev {} | ncv {} | factorization {} | initial guesses {}", solver.config.maxNev.value(),
                          solver.config.maxNcv.value(), eig::FactorizationToString(hamiltonian.factorization), solver.config.initial_guess.size());
        solver.eigs(hamiltonian);
        auto &eigvals_vec = solver.result.get_eigvals<fp64>();
        auto &eigvecs_vec = solver.result.get_eigvecs<Scalar, eig::Form::SYMM>();
        auto  rows        = static_cast<long>(hamiltonian.rows());
        auto  cols        = static_cast<long>(solver.config.maxNev.value());
        eigvals           = Eigen::Map<VectorType>(eigvals_vec.data(), cols);
        eigvecs           = Eigen::Map<MatrixType>(eigvecs_vec.data(), rows, cols);

        // eigvals = eig::view::get_eigvals<fp64>(solver.result, false);
        // eigvecs = eig::view::get_eigvecs<Scalar>(solver.result, eig::Side::R, false);
        // Normalize the columns (they are not normalized if primme hasn't converged!)
        // Normalize the columns (they are not necessarily normalized if primme hasn't converged!)
        if(solver.result.meta.nev_converged != nev) {
            eigvecs = linalg::matrix::modified_gram_schmidt(eigvecs); // Orthogonalize
            std::vector<long> valid_cols;
            for(long col = 0; col < eigvecs.cols(); ++col) {
                if(solver.result.meta.residual_norms.at(static_cast<size_t>(col)) > solver.result.meta.problemNorm * solver.config.tol.value()) continue;
                if(eigvecs.col(col).isZero()) continue;

                valid_cols.emplace_back(col);
            }
            if(valid_cols.empty()) continue;
            eigvals = eigvals(valid_cols).eval();
            eigvecs = eigvecs(Eigen::all, valid_cols).colwise().normalized().eval();
        }
        // else {
        // eigvals = eigvals_map;
        // eigvecs = eigvecs_map;
        // }

        // Check the quality of the subspace
        Eigen::VectorXd overlaps = (initial_vec.adjoint() * eigvecs).cwiseAbs().real();
        // tools::log->info("eigvecs: \n{}\n", linalg::matrix::to_string(eigvecs.real(), 8));
        // tools::log->info("norms  : \n{}\n", linalg::matrix::to_string(eigvecs.colwise().norm(), 8));
        // tools::log->info("eigvals: \n{}\n", linalg::matrix::to_string(eigvals.real().transpose(), 8));
        // tools::log->info("overlaps: \n{}\n", linalg::matrix::to_string(overlaps.real().transpose(), 8));

        double max_overlap    = overlaps.maxCoeff();
        double min_overlap    = overlaps.minCoeff();
        double sq_sum_overlap = overlaps.cwiseAbs2().sum();
        double subspace_error = 1.0 - sq_sum_overlap;
        double lu_time        = hamiltonian.t_factorOP.get()->get_last_interval();
        double ham_time       = hamiltonian.t_genMat.get()->get_last_interval();
        reports::subs_add_entry(nev, max_overlap, min_overlap, subspace_error, solver.result.meta.time_total, ham_time, lu_time, solver.result.meta.iter,
                                solver.result.meta.num_mv, solver.result.meta.num_pc);
        // if(max_overlap > 1.0 + 1e-6) throw except::runtime_error("max_overlap larger than one: {:.16f}", max_overlap);
        // if(sq_sum_overlap > 1.0 + 1e-6) throw except::runtime_error("eps larger than one: {:.16f}", sq_sum_overlap);
        // if(min_overlap < 0.0) throw except::runtime_error("min_overlap smaller than zero: {:.16f}", min_overlap);
        if(max_overlap > 1.0 + 1e-6) tools::log->debug("max_overlap larger than one: {:.16f}", max_overlap);
        if(sq_sum_overlap > 1.0 + 1e-6) tools::log->debug("eps larger than one: {:.16f}", sq_sum_overlap);
        if(min_overlap < 0.0) tools::log->debug("min_overlap smaller than zero: {:.16f}", min_overlap);

        tools::log->debug("Found {} eigenpairs | nev {} converged {} | subspace error {:.3e} | iters {}", eigvals.size(), nev, eigvecs.cols(), subspace_error,
                          solver.result.meta.iter);

        if(subspace_error < meta.subspace_tol.value_or(1e-10)) {
            reason = fmt::format("subspace error is low enough: {:.3e} < threshold {:.3e}", subspace_error, meta.subspace_tol.value_or(1e-10));
            break;
        }
        if(meta.optAlgo == OptAlgo::DMRGX and sq_sum_overlap >= 1.0 / std::sqrt(2.0)) {
            reason = fmt::format("Overlap is sufficient:  {:.16f} >= threshold {:.16f}", max_overlap, 1.0 / std::sqrt(2.0));
            break;
        }
    }
    tools::log->debug("Finished iterative eigensolver -- reason: {}", reason);
    if(eigvals.size() == 0 or eigvecs.size() == 0) {
        eigvecs = Eigen::Map<MatrixType>(init.data(), init.size(), 1);
        eigvals.resize(1);
        eigvals(0) = eigval_shift;
        return std::make_pair(eigvecs, eigvals);
    }

    if(config.shift_invert == eig::Shinv::ON)
        for(auto &e : eigvals) e = 1.0 / e + eigval_shift;

    return std::make_pair(eigvecs, eigvals);
}

template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_primme<cx64>(const TensorsFinite &tensors, double eigval_target,
                                                                                           const OptMeta &meta);
template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_primme<fp64>(const TensorsFinite &tensors, double eigval_target,
                                                                                           const OptMeta &meta);

template<typename Scalar>
std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_lapack(const TensorsFinite &tensors) {
    tools::log->trace("find_subspace_lapack");
    auto t_full = tid::tic_scope("full");
    // Generate the Hamiltonian matrix
    auto effective_hamiltonian = tensors.get_effective_hamiltonian<Scalar>();

    // Create a solver and diagonalize the local effective Hamiltonian
    eig::solver solver;
    solver.eig<eig::Form::SYMM>(effective_hamiltonian.data(), effective_hamiltonian.dimension(0), eig::Vecs::ON, eig::Dephase::OFF);
    //    solver.eig(effective_hamiltonian.data(), effective_hamiltonian.dimension(0), 'I', 1, 1, 0.0, 1.0, 1, eig::Vecs::ON, eig::Dephase::OFF);

    tools::log->debug("Finished eigensolver -- reason: Full diagonalization");

    const auto &multisite_mps = tensors.state->get_multisite_mps<cx64>();
    const auto  multisite_vec = Eigen::Map<const Eigen::VectorXcd>(multisite_mps.data(), multisite_mps.size());

    auto            eigvals  = eig::view::get_eigvals<double>(solver.result);
    auto            eigvecs  = eig::view::get_eigvecs<Scalar>(solver.result);
    Eigen::VectorXd overlaps = (multisite_vec.adjoint() * eigvecs).cwiseAbs().real();
    int             idx;
    double          max_overlap    = overlaps.maxCoeff(&idx);
    double          min_overlap    = overlaps.minCoeff();
    double          sq_sum_overlap = overlaps.cwiseAbs2().sum();
    double          subspace_error = 1.0 - sq_sum_overlap;
    long            nev            = eigvecs.cols();
    auto            time_eig       = tid::get("eig").get_last_interval();
    auto            time_ham       = tid::get("ham").get_last_interval();
    reports::subs_add_entry(nev, max_overlap, min_overlap, subspace_error, time_eig, time_ham, 0, 1, 0, 0);
    return {eigvecs, eigvals};
}

template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_lapack<cx64>(const TensorsFinite &tensors);
template std::pair<Eigen::MatrixXcd, Eigen::VectorXd> subspace::find_subspace_lapack<fp64>(const TensorsFinite &tensors);

template<typename T>
MatrixType<T> subspace::get_hamiltonian_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs) {
    // First, make sure every candidate is actually a basis vector, otherwise this computation would turn difficult if we have to skip rows and columns
    auto t_ham = tid::tic_scope("ham_sub");
    for(const auto &eigvec : eigvecs)
        if(not eigvec.is_basis_vector)
            throw std::runtime_error("One eigvec is not a basis vector. When constructing a hamiltonian subspace matrix, make sure the candidates are all "
                                     "eigenvectors/basis vectors");

    const auto &env1 = edges.get_multisite_env_ene_blk_as<T>();
    const auto &mpo1 = model.get_multisite_mpo<T>();

    tools::log->trace("Generating H² in a subspace of {} eigenvectors of H", eigvecs.size());
    long dim0   = mpo1.dimension(2);
    long dim1   = env1.L.dimension(0);
    long dim2   = env1.R.dimension(0);
    long eignum = safe_cast<long>(eigvecs.size()); // Number of eigenvectors

    Eigen::Tensor<T, 0> H1_ij;
    Eigen::Tensor<T, 3> H1_mps(dim0, dim1, dim2); // The local hamiltonian multiplied by mps at column j.
    MatrixType<T>       H1_sub(eignum, eignum);   // The local hamiltonian projected to the subspace (spanned by eigvecs)
    for(auto col = 0; col < eignum; col++) {
        const auto mps_j = std::next(eigvecs.begin(), col)->get_tensor_as<T>();
        tools::common::contraction::matrix_vector_product(H1_mps, mps_j, mpo1, env1.L, env1.R);
        auto &threads = tenx::threads::get();
        for(auto row = col; row < eignum; row++) {
            const auto mps_i            = std::next(eigvecs.begin(), row)->get_tensor_as<T>();
            H1_ij.device(*threads->dev) = mps_i.conjugate().contract(H1_mps, tenx::idx({0, 1, 2}, {0, 1, 2}));
            H1_sub(row, col)            = H1_ij(0);
            if constexpr(sfinae::is_std_complex_v<T>) {
                H1_sub(col, row) = std::conj(H1_ij(0));
            } else {
                H1_sub(col, row) = H1_ij(0);
            }
        }
    }
    return H1_sub;
}

// Explicit instantiations
template MatrixType<fp32> subspace::get_hamiltonian_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs);
template MatrixType<fp64> subspace::get_hamiltonian_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs);
template MatrixType<cx32> subspace::get_hamiltonian_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs);
template MatrixType<cx64> subspace::get_hamiltonian_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs);

template<typename T>
MatrixType<T> subspace::get_hamiltonian_squared_in_subspace(const ModelFinite &model, const EdgesFinite &edges, const std::vector<opt_mps> &eigvecs) {
    // First, make sure every candidate is actually a basis vector, otherwise this computation would turn difficult if we have to skip rows and columns
    auto t_ham = tid::tic_scope("ham²_sub");
    for(const auto &eigvec : eigvecs)
        if(not eigvec.is_basis_vector)
            throw std::runtime_error("One eigvec is not a basis vector. When constructing a hamiltonian subspace matrix, make sure the candidates are all "
                                     "eigenvectors/basis vectors");

    const auto &env2 = edges.get_multisite_env_var_blk_as<T>();
    const auto &mpo2 = model.get_multisite_mpo_squared<T>();

    tools::log->trace("Generating H² in a subspace of {} eigenvectors of H", eigvecs.size());
    long dim0   = mpo2.dimension(2);
    long dim1   = env2.L.dimension(0);
    long dim2   = env2.R.dimension(0);
    long eignum = safe_cast<long>(eigvecs.size()); // Number of eigenvectors

    Eigen::Tensor<T, 0> H2_ij;
    Eigen::Tensor<T, 3> H2_mps(dim0, dim1, dim2); // The local hamiltonian multiplied by mps at column j.
    MatrixType<T>       H2_sub(eignum, eignum);   // The local hamiltonian projected to the subspace (spanned by eigvecs)
    for(auto col = 0; col < eignum; col++) {
        const auto mps_j = std::next(eigvecs.begin(), col)->get_tensor_as<T>();
        tools::common::contraction::matrix_vector_product(H2_mps, mps_j, mpo2, env2.L, env2.R);
        auto &threads = tenx::threads::get();
        for(auto row = col; row < eignum; row++) {
            const auto mps_i            = std::next(eigvecs.begin(), row)->get_tensor_as<T>();
            H2_ij.device(*threads->dev) = mps_i.conjugate().contract(H2_mps, tenx::idx({0, 1, 2}, {0, 1, 2}));
            H2_sub(row, col)            = H2_ij(0);
            if constexpr(sfinae::is_std_complex_v<T>) {
                H2_sub(col, row) = std::conj(H2_ij(0));
            } else {
                H2_sub(col, row) = H2_ij(0);
            }
        }
    }
    return H2_sub;
}

// Explicit instantiations
template MatrixType<fp32> subspace::get_hamiltonian_squared_in_subspace(const ModelFinite &model, const EdgesFinite &edges,
                                                                        const std::vector<opt_mps> &eigvecs);
template MatrixType<fp64> subspace::get_hamiltonian_squared_in_subspace(const ModelFinite &model, const EdgesFinite &edges,
                                                                        const std::vector<opt_mps> &eigvecs);
template MatrixType<cx32> subspace::get_hamiltonian_squared_in_subspace(const ModelFinite &model, const EdgesFinite &edges,
                                                                        const std::vector<opt_mps> &eigvecs);
template MatrixType<cx64> subspace::get_hamiltonian_squared_in_subspace(const ModelFinite &model, const EdgesFinite &edges,
                                                                        const std::vector<opt_mps> &eigvecs);