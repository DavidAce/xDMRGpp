#include "../opt_meta.h"
#include "../opt_mps.h"
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "io/fmt_f128_t.h"
#include "math/eig.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/float.h"
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
#include "tools/finite/measure/residual.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include <Eigen/Eigenvalues>
#include <h5pp/h5pp.h>
#include <primme/primme.h>

namespace tools::finite::opt {
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
        template<typename Scalar>
        Eigen::Tensor<Scalar, 3> get_initial_guess(const opt_mps &initial_mps, const std::vector<opt_mps> &results) {
            if(results.empty()) {
                if constexpr(std::is_same_v<Scalar, fp64>)
                    return initial_mps.get_tensor().real();
                else
                    return initial_mps.get_tensor();
            } else {
                // Return whichever of initial_mps or results that has the lowest variance
                auto it = std::min_element(results.begin(), results.end(), internal::comparator::variance);
                if(it == results.end()) return get_initial_guess<Scalar>(initial_mps, {});

                if(it->get_variance() < initial_mps.get_variance()) {
                    tools::log->debug("Previous result is a good initial guess: {} | var {:8.2e}", it->get_name(), it->get_variance());
                    return get_initial_guess<Scalar>(*it, {});
                } else
                    return get_initial_guess<Scalar>(initial_mps, {});
            }
        }

        template<typename Scalar>
        std::vector<opt_mps_init_t<Scalar>> get_initial_guess_mps(const opt_mps &initial_mps, const std::vector<opt_mps> &results, long nev) {
            std::vector<opt_mps_init_t<Scalar>> init;
            if(results.empty()) {
                if constexpr(std::is_same_v<Scalar, fp64>)
                    init.push_back({initial_mps.get_tensor().real(), 0});
                else
                    init.push_back({initial_mps.get_tensor(), 0});
            } else {
                for(long n = 0; n < nev; n++) {
                    // Take the latest result with idx == n

                    // Start by collecting the results with the correct index
                    std::vector<std::reference_wrapper<const opt_mps>> results_idx_n;
                    for(const auto &r : results) {
                        if(r.get_eigs_idx() == n) results_idx_n.emplace_back(r);
                    }
                    if(not results_idx_n.empty()) {
                        if constexpr(std::is_same_v<Scalar, fp64>) {
                            init.push_back({results_idx_n.back().get().get_tensor().real(), n});
                        } else {
                            init.push_back({results_idx_n.back().get().get_tensor(), n});
                        }
                    }
                }
            }
            if(init.size() > safe_cast<size_t>(nev)) throw except::logic_error("Found too many initial guesses");
            return init;
        }

    }

    template<typename Scalar>
    std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, std::vector<long>>
        modified_gram_schmidt(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q) {
        auto t_gramSchmidt = tid::tic_scope("gramschmidt");

        // Orthonormalize with Modified Gram Schmidt
        using RealScalar            = typename Eigen::NumTraits<Scalar>::Real;
        using MatrixType            = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        auto           nonOrthoCols = std::vector<long>();
        auto           validCols    = std::vector<long>();
        constexpr auto normTol      = std::numeric_limits<RealScalar>::epsilon() * static_cast<RealScalar>(1e2);
        constexpr auto orthTol      = std::numeric_limits<RealScalar>::epsilon() * static_cast<RealScalar>(1e1);
        constexpr auto one          = static_cast<RealScalar>(1.0);
        auto           idenTol      = static_cast<RealScalar>(settings::precision::max_norm_error);
        validCols.reserve(Q.cols());
        nonOrthoCols.reserve(Q.cols());

        for(long i = 0; i < Q.cols(); ++i) {
            auto norm = Q.col(i).norm();
            if(std::abs(norm) < normTol) { continue; }
            Q.col(i) /= norm;
            for(long j = i + 1; j < Q.cols(); ++j) { Q.col(j) -= Q.col(i).dot(Q.col(j)) * Q.col(i); }
        }
        MatrixType Qid = MatrixType::Zero(Q.cols(), Q.cols());
        for(long j = 0; j < Qid.cols(); ++j) {
            for(long i = 0; i <= j; ++i) {
                Qid(i, j) = Q.col(i).dot(Q.col(j));
                Qid(j, i) = Qid(i, j);
            }
            if(j == 0 and std::abs(Qid(j, j) - one) > orthTol) {
                nonOrthoCols.emplace_back(j);
            } else if(j > 0 and std::abs(Qid(j, j) - one + Qid.col(j).topRows(j).cwiseAbs().sum()) > orthTol) {
                nonOrthoCols.emplace_back(j);
            }

            if(j == 0 and std::abs(Qid(j, j) - one) <= orthTol) {
                validCols.emplace_back(j);
            } else if(j > 0 and std::abs(Qid(j, j) - one + Qid.col(j).topRows(j).cwiseAbs().sum()) <= orthTol) {
                validCols.emplace_back(j);
            }
        }

        if(!Qid(validCols, validCols).isIdentity(idenTol)) {
            tools::log->info("Qid \n{}\n", linalg::matrix::to_string(Qid, 8));
            tools::log->info("vc  {}", validCols);
            tools::log->info("noc {}", nonOrthoCols);
            throw except::runtime_error("Q has non orthonormal columns: \n{}\n"
                                        " validCols   : {}\n"
                                        " nonOrthoCols: {}",
                                        linalg::matrix::to_string(Qid, 8), validCols, nonOrthoCols);
        }

        // Q(Eigen::all, nonOrthoCols).setZero();
        Q.colwise().normalize();
        for(long j = 0; j < Q.cols(); ++j) {
            if(!Q.col(j).allFinite()) Q.col(j).setZero();
        }
        return {Q, nonOrthoCols};
    }

    template<typename T>
    opt_mps eigs_lanczos_h1h2(const opt_mps &initial, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges, OptMeta &opt_meta) {
        auto          t_mixblk = tid::tic_scope("mixblk");
        auto          K1_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG);
        auto          K2_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG);
        auto          sites    = initial.get_sites();
        auto          mpos     = model.get_mpo(sites);
        auto          enve     = edges.get_multisite_env_ene(sites);
        auto          envv     = edges.get_multisite_env_var(sites);
        MatVecMPOS<T> H1       = MatVecMPOS<T>(mpos, enve);
        MatVecMPOS<T> H2       = MatVecMPOS<T>(mpos, envv);
        using RealScalar       = typename Eigen::NumTraits<T>::Real;
        using CplxScalar       = std::complex<RealScalar>;
        using MatrixType       = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using MatrixCplx       = Eigen::Matrix<CplxScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using MatrixReal       = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorCplx       = Eigen::Matrix<CplxScalar, Eigen::Dynamic, 1>;
        using VectorReal       = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        auto nonOrthoCols      = std::vector<long>();

        auto mps_size  = H1.get_size();
        auto mps_shape = H1.get_shape_mps();
        long ncv       = opt_meta.eigs_ncv.value_or(3);
        if(ncv <= 0) { ncv = safe_cast<int>(std::ceil(std::log2(initial.get_tensor().size()))); }

        auto       H1V = MatrixType();
        auto       H2V = MatrixType();
        MatrixType K1  = MatrixType::Zero(ncv, ncv);
        MatrixType K2  = MatrixType::Zero(ncv, ncv);

        if(K1_on) H1V.resize(mps_size, ncv);
        if(K2_on) H2V.resize(mps_size, ncv);

        // Default solution
        opt_mps result;
        result.set_tensor(initial.get_tensor());

        result.is_basis_vector = false;
        result.set_name(fmt::format("eigenvector 0 [lanczos h1h2]"));
        result.set_sites(initial.get_sites());
        result.set_eshift(initial.get_eshift()); // Will set energy if also given the eigval
        result.set_eigs_idx(0);
        result.set_eigs_nev(1);
        result.set_eigs_ncv(ncv);
        result.set_eigs_tol(opt_meta.eigs_tol.value_or(1e-12));
        result.set_eigs_ritz(enum2sv(opt_meta.optRitz));
        result.set_optalgo(opt_meta.optAlgo);
        result.set_optsolver(opt_meta.optSolver);

        result.set_energy(initial.get_energy());
        result.set_energy_shifted(initial.get_energy_shifted());
        result.set_hsquared(initial.get_hsquared());
        result.set_variance(initial.get_variance());

        // res.alpha_mps = 1.0;
        // res.alpha_h1v = 0.0;
        // res.alpha_h2v = 0.0;

        // Initialize Krylov vector 0
        MatrixType V(mps_size, ncv);
        V.col(0) = initial.get_vector_as<T>();

        auto                    mixedColOk = std::vector<long>(); // New states with acceptable norm and eigenvalue
        constexpr auto          eps        = std::numeric_limits<RealScalar>::epsilon();
        RealScalar              optVal     = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar              oldVal     = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar              relVal     = std::numeric_limits<RealScalar>::quiet_NaN();
        long                    optIdx     = 0;
        RealScalar              tol        = static_cast<RealScalar>(opt_meta.eigs_tol.value_or(settings::precision::eigs_tol_max));
        RealScalar              absTol     = eps * static_cast<RealScalar>(1e2);
        RealScalar              relTol     = std::sqrt(eps); // 1e-4
        RealScalar              rnorm      = 1.0;
        [[maybe_unused]] double snorm      = 1.0; // Estimate the matrix norm from the largest singular value/eigenvalue. Converged if  rnorm  < snorm * tol
        size_t                  iter       = 0;
        size_t                  ngs        = 0;
        std::string             exit_msg;
        while(true) {
            // Define the krylov subspace
            for(long i = 0; i + 1 < ncv; ++i) {
                if(i < ncv / 2) {
                    H1.MultAx(V.col(i).data(), V.col(i + 1).data());
                } else if(i == ncv / 2) {
                    H2.MultAx(V.col(0).data(), V.col(i + 1).data());
                } else {
                    H2.MultAx(V.col(i).data(), V.col(i + 1).data());
                }
            }

            // Orthonormalize with Modified Gram Schmidt
            for(size_t igs = 0; igs <= 5; ++igs) {
                std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
                ngs++;
                if(nonOrthoCols.empty()) break;
            }

            // V should now have orthonormal vectors
            if(K1_on) {
                for(long i = 0; i < ncv; ++i) H1.MultAx(V.col(i).data(), H1V.col(i).data());
            }
            if(K2_on) {
                for(long i = 0; i < ncv; ++i) H2.MultAx(V.col(i).data(), H2V.col(i).data());
            }
            if(!std::isnan(optVal)) {
                if(opt_meta.optAlgo == OptAlgo::DMRG)
                    rnorm = (H1V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();
                else if(opt_meta.optAlgo == OptAlgo::GDMRG)
                    rnorm = (H1V.col(0) - optVal * H2V.col(0)).template lpNorm<Eigen::Infinity>();
                else
                    rnorm = (H2V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();
            }

            if(iter >= 1ul) {
                if(rnorm < tol /* * snorm */) {
                    exit_msg = std::format("converged rnorm {:.3e} < tol {:.3e}", rnorm, tol);
                    break;
                }

                if(std::abs(oldVal - optVal) < absTol) {
                    exit_msg = std::format("saturated: abs change {:.3e} < {:.3e}", std::abs(oldVal - optVal), absTol);
                    break;
                }
                if(relVal < relTol) {
                    exit_msg = std::format("saturated: rel change ({:.3e}) < {:.3e}", relVal, relTol);
                    break;
                }
                if(iter >= std::max<size_t>(1ul, opt_meta.eigs_iter_max.value_or(1))) {
                    exit_msg = fmt::format("iter ({}) >= maxiter ({})", iter, opt_meta.eigs_iter_max.value_or(1));
                    break;
                }
                if(mixedColOk.size() == 1) {
                    exit_msg = fmt::format("saturated: only one valid eigenvector");
                    break;
                }

                if(mixedColOk.empty()) {
                    exit_msg = fmt::format("mixedColOk is empty");
                    break;
                }
            }

            auto t_dotprod = tid::tic_scope("dotprod");

            if(K1_on) {
                for(long j = 0; j < ncv; ++j) {
                    for(long i = j; i < ncv; ++i) { K1(i, j) = V.col(i).dot(H1V.col(j)); }
                }
                K1 = K1.template selfadjointView<Eigen::Lower>();
            }

            if(K2_on) {
                // Use abs to avoid negative near-zero values
                for(long j = 0; j < ncv; ++j) {
                    for(long i = j; i < ncv; ++i) {
                        if(i == j)
                            K2(i, j) = std::abs(V.col(i).dot(H2V.col(j)));
                        else
                            K2(i, j) = V.col(i).dot(H2V.col(j));
                    }
                }
                K2 = K2.template selfadjointView<Eigen::Lower>();
            }

            t_dotprod.toc();
            auto       t_eigsol      = tid::tic_scope("eigsol");
            long       numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
            long       numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
            long       numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
            VectorReal evals; // Eigen::VectorXd ::Zero();
            MatrixCplx evecs; // Eigen::MatrixXcd::Zero();
            OptRitz    ritz_internal = opt_meta.optRitz;
            switch(opt_meta.optAlgo) {
                using enum OptAlgo;
                case DMRG: {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K1, Eigen::ComputeEigenvectors);
                    if(solver.info() == Eigen::ComputationInfo::Success) {
                        evals = solver.eigenvalues();
                        evecs = solver.eigenvectors();
                    } else {
                        tools::log->info("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
                        tools::log->warn("Diagonalization of K1 exited with info {}", static_cast<int>(solver.info()));
                    }

                    if(evals.hasNaN()) throw except::runtime_error("found nan eigenvalues");
                    break;
                }
                case DMRGX: [[fallthrough]];
                case HYBRID_DMRGX: {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                    evals       = solver.eigenvalues();
                    evecs       = solver.eigenvectors();
                    break;
                }
                case XDMRG: {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K2, Eigen::ComputeEigenvectors);
                    evals       = solver.eigenvalues();
                    evecs       = solver.eigenvectors();
                    break;
                }
                case GDMRG: {
                    if(nonOrthoCols.empty() and numZeroRows == 0) {
                        auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>(K1.template selfadjointView<Eigen::Lower>(),
                                                                                           K2.template selfadjointView<Eigen::Lower>(),
                                                                                           Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                        evals       = solver.eigenvalues().real();
                        evecs       = solver.eigenvectors().colwise().normalized();
                    } else {
                        auto solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                        evals       = solver.eigenvalues();
                        evecs       = solver.eigenvectors();
                        if(opt_meta.optRitz == OptRitz::LM) ritz_internal = OptRitz::SM;
                        if(opt_meta.optRitz == OptRitz::LR) ritz_internal = OptRitz::SM;
                        if(opt_meta.optRitz == OptRitz::SM) ritz_internal = OptRitz::LM;
                        if(opt_meta.optRitz == OptRitz::SR) ritz_internal = OptRitz::LR;
                    }

                    break;
                }
            }
            auto t_checks         = tid::tic_scope("checks");
            snorm                 = static_cast<double>(evals.cwiseAbs().maxCoeff());
            V                     = (V * evecs.real()).eval(); // Now V has ncv columns mixed according to evecs
            VectorReal mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
            mixedColOk.clear();                                // New states with acceptable norm and eigenvalue
            mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
            for(long i = 0; i < mixedNorms.size(); ++i) {
                if(std::abs(mixedNorms(i) - static_cast<RealScalar>(1.0)) > static_cast<RealScalar>(settings::precision::max_norm_error)) continue;
                // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
                // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues
                // of GDMRG are not
                mixedColOk.emplace_back(i);
            }
            if constexpr(!tenx::sfinae::is_quadruple_prec_v<T>) {
                if(mixedColOk.size() <= 1) {
                    tools::log->debug("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
                    tools::log->debug("K2                     : \n{}\n", linalg::matrix::to_string(K2, 8));
                    tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
                    // tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(evecs, 8));
                    // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
                    tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
                    tools::log->debug("mixedColOk             = {}", mixedColOk);
                    tools::log->debug("numZeroRowsK1          = {}", numZeroRowsK1);
                    tools::log->debug("numZeroRowsK2          = {}", numZeroRowsK2);
                    tools::log->debug("nonOrthoCols           = {}", nonOrthoCols);
                    tools::log->debug("ngramSchmidt           = {}", ngs);
                    if(opt_meta.optAlgo == OptAlgo::GDMRG) {
                        H2.MultAx(V.col(0).data(), H2V.col(0).data());
                        H2.MultAx(V.col(1).data(), H2V.col(1).data());
                        H2.MultAx(V.col(2).data(), H2V.col(2).data());
                        tools::log->debug("V.col(0).dot(H2*V.col(1)) = {:.16f}", V.col(0).dot(H2V.col(1)));
                        tools::log->debug("V.col(0).dot(H2*V.col(2)) = {:.16f}", V.col(0).dot(H2V.col(2)));
                        tools::log->debug("V.col(1).dot(H2*V.col(2)) = {:.16f}", V.col(1).dot(H2V.col(2)));
                    } else {
                        tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
                        tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
                        tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
                    }
                }
            }
            // tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
            // Eigenvalues are sorted in ascending order.
            long colIdx = 0;
            switch(ritz_internal) {
                case OptRitz::SR: {
                    evals(mixedColOk).minCoeff(&colIdx);
                    break;
                }
                case OptRitz::LR: {
                    evals(mixedColOk).maxCoeff(&colIdx);
                    break;
                }
                case OptRitz::SM: {
                    evals(mixedColOk).cwiseAbs().minCoeff(&colIdx);
                    break;
                }
                case OptRitz::LM: {
                    evals(mixedColOk).cwiseAbs().maxCoeff(&colIdx);
                    break;
                }
                case OptRitz::IS: [[fallthrough]];
                case OptRitz::TE: [[fallthrough]];
                case OptRitz::NONE: {
                    (evals(mixedColOk).array() - static_cast<RealScalar>(initial.get_energy())).cwiseAbs().minCoeff(&colIdx);
                }
            }
            optIdx = mixedColOk[colIdx];

            oldVal = optVal;
            optVal = evals(optIdx);
            relVal = std::abs((oldVal - optVal) / (static_cast<RealScalar>(0.5) * (optVal + oldVal)));

            // Check convergence

            // If we make it here: update the solution
            result.set_tensor(Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(optIdx).data(), mps_shape).template cast<cx64>());
            VectorReal col = evecs.col(optIdx).real();
            // res.alpha_mps       = col.coeff(0);
            // res.alpha_h1v       = col.coeff(1);
            // res.alpha_h2v       = col.coeff(ncv / 2 + 1);

            if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others)

            if(iter + 1 < opt_meta.eigs_iter_max)
                tools::log->trace("lanczos: {:.34f} [{}] | sites {} (size {}) | rnorm {:.3e} | ngs {} | iters {} | "
                                 "{:.3e} it/s |  {:.3e} s",
                                 fp(optVal), optIdx, sites, mps_size, fp(rnorm), ngs, iter, iter / t_mixblk->get_last_interval(),
                                 t_mixblk->get_last_interval());

            iter++;
        }

        result.set_overlap(std::abs(initial.get_vector().dot(result.get_vector())));
        result.set_length(initial.get_length());
        result.set_time(t_mixblk->get_last_interval());
        result.set_time_mv(H1.t_multAx->get_time() + H2.t_multAx->get_time());
        result.set_time_pc(H1.t_multPc->get_time() + H2.t_multPc->get_time());
        result.set_op(H1.num_op + H2.num_op);
        result.set_mv(H1.num_mv + H2.num_mv);
        result.set_pc(H1.num_pc + H2.num_pc);
        result.set_iter(iter);
        result.set_eigs_rnorm(static_cast<fp64>(rnorm));
        result.set_rnorm_H1(static_cast<fp64>((H1V.col(0) - optVal * V.col(0)).norm()));
        result.set_rnorm_H2(static_cast<fp64>((H2V.col(0) - optVal * V.col(0)).norm()));
        result.set_eigs_eigval(static_cast<fp64>(optVal));
        RealScalar vh1v = std::real(V.col(0).dot(H1V.col(0)));
        RealScalar vh2v = std::real(V.col(0).dot(H2V.col(0)));
        result.set_energy(static_cast<fp64>(vh1v) + result.get_eshift());
        result.set_hsquared(static_cast<fp64>(vh2v));
        if(K1_on) { result.set_variance(static_cast<fp64>(vh2v - vh1v * vh1v)); }

        tools::log->info("lancsoz {}: {:.34f} [{}] | ⟨H⟩ {:.16f} | ⟨H²⟩ {:.16f} | ⟨H²⟩-⟨H⟩² {:.4e} | sites {} (size {}) | rnorm {:.3e} | ngs {} | iters "
                         "{} | {:.3e} s | {} | var {:.4e}",
                         sfinae::type_name<RealScalar>(), fp(optVal), optIdx, result.get_energy(), result.get_hsquared(), result.get_variance(), sites,
                         mps_size, fp(rnorm), ngs, iter, t_mixblk->get_last_interval(), exit_msg, fp(vh2v - vh1v * vh1v));
        reports::eigs_add_entry(result, spdlog::level::debug);
        return result;
    }

    template opt_mps eigs_lanczos_h1h2<fp32>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                             OptMeta &opt_meta);
    template opt_mps eigs_lanczos_h1h2<fp64>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                             OptMeta &opt_meta);
    template opt_mps eigs_lanczos_h1h2<fp128>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                              OptMeta &opt_meta);
    template opt_mps eigs_lanczos_h1h2<cx32>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                             OptMeta &opt_meta);
    template opt_mps eigs_lanczos_h1h2<cx64>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                             OptMeta &opt_meta);
    template opt_mps eigs_lanczos_h1h2<cx128>(const opt_mps &initial_mps, const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges,
                                              OptMeta &opt_meta);

    [[nodiscard]] opt_mps internal::optimize_lanczos_h1h2(const TensorsFinite &tensors, const opt_mps &initial, [[maybe_unused]] const AlgorithmStatus &status,
                                                          OptMeta &meta) {
        using namespace internal;
        using namespace settings::precision;
        initial.validate_initial_mps();
        reports::eigs_add_entry(initial, spdlog::level::debug);

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

        switch(meta.optType) {
            case OptType::FP32: return eigs_lanczos_h1h2<fp32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            case OptType::FP64: return eigs_lanczos_h1h2<fp64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            case OptType::FP128: return eigs_lanczos_h1h2<fp128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            case OptType::CX32: return eigs_lanczos_h1h2<cx32>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            case OptType::CX64: return eigs_lanczos_h1h2<cx64>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            case OptType::CX128: return eigs_lanczos_h1h2<cx128>(initial, tensors.get_state(), tensors.get_model(), tensors.get_edges(), meta);
            default: throw std::runtime_error("unrecognized option type");
        }
    }
}