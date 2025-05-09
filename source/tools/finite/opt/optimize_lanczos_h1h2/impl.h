#pragma once
#include "../../opt_meta.h"
#include "../../opt_mps.h"
#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "general/iter.h"
#include "general/sfinae.h"
#include "io/fmt_f128_t.h"
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
#include "tools/finite/measure/hamiltonian.impl.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/opt/opt-internal.h"
#include "tools/finite/opt/report.h"
#include <Eigen/Eigenvalues>
#include <h5pp/h5pp.h>

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

enum class LanczosExit : int {
    ok                    = 0,  /*! No issues on this Lanczos step: can keep running */
    converged_rnormTol    = 1,  /*! The residual norm fell below a threshold. */
    saturated_relDiffTol  = 2,  /*! The relative value change fell below a threshold.  */
    saturated_absDiffTol  = 4,  /*! The absolute value change fell below a threshold.  */
    max_iterations        = 8,  /*! The maximum number of iterations was reached. */
    one_valid_eigenvector = 16, /*! Only one valid eigenvector was found. */
    no_valid_eigenvector  = 32, /*! No valid eigenvector was found. */
    allow_bitops
};

template<typename Scalar>
struct Lanczos {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    private:
    struct Status {
        RealScalar               optVal  = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               oldVal  = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               absDiff = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               relDiff = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               initVal = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               rnorm   = RealScalar{1};
        RealScalar               maxEval = RealScalar{1};
        long                     optIdx  = 0;
        size_t                   iter    = 0;
        long                     numMGS  = 0;
        std::vector<long>        nonZeroCols; // Nonzero Gram Schmidt columns
        std::vector<long>        mixedColOk;  // New states with acceptable norm and eigenvalue
        std::vector<std::string> exitMsg = {};
        LanczosExit              exit    = LanczosExit::ok;
    };

    public:
    Status                      status = {};
    Eigen::Index                mps_size;
    std::array<Eigen::Index, 3> mps_shape;
    const Eigen::Index ncv       = 3; // Krylov dimension, i.e. {V, H1V..., H2V...} ( minimum 2, recommend 3 or more)
    OptAlgo                     algo;
    OptRitz                     ritz;
    MatVecMPOS<Scalar>          H1, H2;
    MatrixType                  K1, K2;
    MatrixType                  V, H1V, H2V;
    VectorReal                  krylov_evals;
    MatrixType                  krylov_evecs;
    bool                        K1_on = false;
    bool                        K2_on = false;

    const RealScalar   eps       = std::numeric_limits<RealScalar>::epsilon();
    RealScalar         tol       = std::numeric_limits<RealScalar>::epsilon() * 10000;
    Eigen::Index       max_iters = 1000;

    Lanczos(Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const VectorType &V0, const auto &mpos, const auto &enve, const auto &envv)
        : ncv(ncv),                                                                                         //
          algo(algo),                                                                                       //
          ritz(ritz),                                                                                       //
          H1(mpos, enve),                                                                                   //
          H2(mpos, envv),                                                                                   //
          K1_on(has_any_flags(algo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG)), //
          K2_on(has_any_flags(algo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG)) {
        assert(ncv >= 2);
        mps_size  = H1.get_size();
        mps_shape = H1.get_shape_mps();
        if(K1_on) { assert(mps_size = H1.rows()); }
        if(K2_on) { assert(mps_size = H2.rows()); }
        V.resize(H1.rows(), ncv);
        V.col(0) = V0;
    }

    RealScalar absDiffTol = std::numeric_limits<RealScalar>::epsilon();
    RealScalar relDiffTol = RealScalar{1e-5f};
    RealScalar rnormTol() const { return tol * status.maxEval; }
};

template<typename Scalar>
void do_lanczos_step(Lanczos<Scalar> &lanczos) {
    using MatrixType = typename Lanczos<Scalar>::MatrixType;
    using VectorType = typename Lanczos<Scalar>::VectorType;
    using VectorReal = typename Lanczos<Scalar>::VectorReal;
    using RealScalar = typename Lanczos<Scalar>::RealScalar;
    using VectorIdxT = typename Lanczos<Scalar>::VectorIdxT;

    auto &V            = lanczos.V;
    auto &H1V          = lanczos.H1V;
    auto &H2V          = lanczos.H2V;
    auto &K1           = lanczos.K1;
    auto &K2           = lanczos.K2;
    auto &K1_on        = lanczos.K1_on;
    auto &K2_on        = lanczos.K2_on;
    auto &krylov_evals = lanczos.krylov_evals;
    auto &krylov_evecs = lanczos.krylov_evecs;
    auto &eps          = lanczos.eps;

    const auto &absDiffTol = lanczos.absDiffTol;
    const auto &relDiffTol = lanczos.relDiffTol;
    const auto &max_iters  = lanczos.max_iters;
    const auto &H1         = lanczos.H1;
    const auto &H2         = lanczos.H2;
    const auto &ncv        = lanczos.ncv;
    const auto &algo       = lanczos.algo;
    const auto &ritz       = lanczos.ritz;

    auto &optVal      = lanczos.status.optVal;
    auto &oldVal      = lanczos.status.oldVal;
    auto &absDiff     = lanczos.status.absDiff;
    auto &relDiff     = lanczos.status.relDiff;
    auto &optIdx      = lanczos.status.optIdx;
    auto &rnorm       = lanczos.status.rnorm;
    auto &maxEval     = lanczos.status.maxEval;
    auto &nonZeroCols = lanczos.status.nonZeroCols;
    auto &iter        = lanczos.status.iter;
    auto &numMGS      = lanczos.status.numMGS;
    auto &exitMsg     = lanczos.status.exitMsg;
    auto &exit        = lanczos.status.exit;

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
        auto t_mgs  = tid::tic_token("mgs");
        auto mgs    = linalg::matrix::modified_gram_schmidt(V);
        nonZeroCols = std::move(mgs.nonZeroCols);
        V           = mgs.Q(Eigen::all, nonZeroCols);
        numMGS++;
        if(nonZeroCols.size() == safe_cast<size_t>(mgs.Q.cols())) break;
    }

    // V should now have orthonormal vectors
    if(K1_on) {
        H1V.resize(H1.rows(), V.cols());
        for(long i = 0; i < V.cols(); ++i) H1.MultAx(V.col(i).data(), H1V.col(i).data());
    }
    if(K2_on) {
        H2V.resize(H1.rows(), V.cols());
        for(long i = 0; i < V.cols(); ++i) H2.MultAx(V.col(i).data(), H2V.col(i).data());
    }

    if(!std::isnan(optVal)) {
        if(algo == OptAlgo::DMRG)
            rnorm = (H1V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();
        else if(algo == OptAlgo::GDMRG)
            rnorm = (H1V.col(0) - optVal * H2V.col(0)).template lpNorm<Eigen::Infinity>();
        else
            rnorm = (H2V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();
    }

    if(iter >= 1ul) {}
    if(rnorm < lanczos.rnormTol()) {
        exitMsg.emplace_back(std::format("converged rnorm {:.3e} < tol {:.3e}", rnorm, lanczos.rnormTol()));
        exit |= LanczosExit::converged_rnormTol;
        return;
    }

    auto t_dotprod = tid::tic_scope("dotprod");

    if(K1_on) {
        assert(H1V.cols() == V.cols());
        K1.resize(V.cols(), V.cols());
        for(long j = 0; j < V.cols(); ++j) {
            for(long i = j; i < V.cols(); ++i) { K1(i, j) = V.col(i).dot(H1V.col(j)); }
        }
        K1 = K1.template selfadjointView<Eigen::Lower>();
    }

    if(K2_on) {
        assert(H2V.cols() == V.cols());
        K2.resize(V.cols(), V.cols());
        // Use abs to avoid negative near-zero values
        for(long j = 0; j < V.cols(); ++j) {
            for(long i = j; i < V.cols(); ++i) {
                if(i == j)
                    K2(i, j) = std::abs(V.col(i).dot(H2V.col(j)));
                else
                    K2(i, j) = V.col(i).dot(H2V.col(j));
            }
        }
        K2 = K2.template selfadjointView<Eigen::Lower>();
    }

    t_dotprod.toc();
    auto t_eigsol      = tid::tic_scope("eigsol");
    long numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
    long numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
    long numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
    // VectorCR evals; // Eigen::VectorXd ::Zero();
    // MatrixCT evecs; // Eigen::MatrixXcd::Zero();
    OptRitz ritz_internal = ritz;
    auto    solver        = eig::solver();
    switch(algo) {
        using enum OptAlgo;
        case DMRG: {
            solver.eig<eig::Form::SYMM>(K1.data(), K1.rows(), eig::Vecs::ON);

            // solver = Eigen::SelfAdjointEigenSolver<MatrixType>(K1, Eigen::ComputeEigenvectors);
            // if(solver.info() == Eigen::ComputationInfo::Success) {
            //     evals = solver.eigenvalues();
            //     evecs = solver.eigenvectors();
            // } else {
            //     tools::log->info("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
            //     tools::log->warn("Diagonalization of K1 exited with info {}", static_cast<int>(solver.info()));
            // }
            //
            // if(evals.hasNaN()) throw except::runtime_error("found nan eigenvalues");
            break;
        }
        case DMRGX: [[fallthrough]];
        case HYBRID_DMRGX: {
            MatrixType K = K2 - K1 * K1;
            solver.eig<eig::Form::SYMM>(K.data(), K.rows(), eig::Vecs::ON);
            // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
            // evals       = solver.eigenvalues();
            // evecs       = solver.eigenvectors();
            break;
        }
        case XDMRG: {
            solver.eig<eig::Form::SYMM>(K2.data(), K2.rows(), eig::Vecs::ON);
            // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2, Eigen::ComputeEigenvectors);
            // evals       = solver.eigenvalues();
            // evecs       = solver.eigenvectors();
            break;
        }
        case GDMRG: {
            if(numZeroRows == 0) {
                solver.eig<eig::Form::SYMM>(K1.data(), K2.data(), K1.rows(), eig::Vecs::ON);

                // auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixCT>(
                // K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                // evals = solver.eigenvalues().real();
                // evecs = solver.eigenvectors().colwise().normalized();
            } else {
                MatrixType K = K2 - K1 * K1;
                solver.eig<eig::Form::SYMM>(K.data(), K.rows(), eig::Vecs::ON);
                // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                // evals       = solver.eigenvalues();
                // evecs       = solver.eigenvectors();
                if(ritz == OptRitz::LM) ritz_internal = OptRitz::SM;
                if(ritz == OptRitz::LR) ritz_internal = OptRitz::SM;
                if(ritz == OptRitz::SM) ritz_internal = OptRitz::LM;
                if(ritz == OptRitz::SR) ritz_internal = OptRitz::LR;
            }
            break;
        }
        default: throw except::runtime_error("unhandled algorithm: [{}]", enum2sv(algo));
    }

    krylov_evals = eig::view::get_eigvals<RealScalar>(solver.result);
    krylov_evecs = eig::view::get_eigvecs<Scalar>(solver.result);

    auto t_checks                = tid::tic_scope("checks");
    maxEval                      = static_cast<RealScalar>(krylov_evals.cwiseAbs().maxCoeff());
    V                            = (V * krylov_evecs.real()).eval(); // Now V has columns mixed according to evecs
    VectorReal        mixedNorms = V.colwise().norm();               // New state norms after mixing cols of V according to cols of evecs
    std::vector<long> mixedColOk;
    mixedColOk.clear(); // New states with acceptable norm and eigenvalue
    mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
    for(long i = 0; i < mixedNorms.size(); ++i) {
        if(std::abs(mixedNorms(i) - RealScalar{1}) > static_cast<RealScalar>(settings::precision::max_norm_error)) continue;
        // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
        // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues
        // of GDMRG are not
        mixedColOk.emplace_back(i);
    }
    // if constexpr(!tenx::sfinae::is_quadruple_prec_v<CalcType>) {
    if(mixedColOk.size() <= 1) {
        tools::log->debug("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
        tools::log->debug("K2                     : \n{}\n", linalg::matrix::to_string(K2, 8));
        tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(krylov_evals, 8));
        // tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(evecs, 8));
        // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
        tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
        tools::log->debug("mixedColOk             = {}", mixedColOk);
        tools::log->debug("numZeroRowsK1          = {}", numZeroRowsK1);
        tools::log->debug("numZeroRowsK2          = {}", numZeroRowsK2);
        tools::log->debug("nonZeroCols            = {}", nonZeroCols);
        tools::log->debug("ngramSchmidt           = {}", numMGS);
    }
    // }
    // tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
    // Eigenvalues are sorted in ascending order.
    long colIdx = 0;
    switch(ritz_internal) {
        case OptRitz::SR: krylov_evals(mixedColOk).minCoeff(&colIdx); break;
        case OptRitz::LR: krylov_evals(mixedColOk).maxCoeff(&colIdx); break;
        case OptRitz::SM: krylov_evals(mixedColOk).cwiseAbs().minCoeff(&colIdx); break;
        case OptRitz::LM: krylov_evals(mixedColOk).cwiseAbs().maxCoeff(&colIdx); break;
        case OptRitz::IS: [[fallthrough]];
        case OptRitz::TE: [[fallthrough]];
        case OptRitz::NONE: {
            if(std::isnan(lanczos.status.initVal))
                throw except::runtime_error("Ritz [{}] does not work when lanczos.status.initVal is nan", enum2sv(ritz_internal));
            (krylov_evals(mixedColOk).array() - lanczos.status.initVal).cwiseAbs().minCoeff(&colIdx);
        }
    }
    optIdx = mixedColOk[colIdx];
    if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others in the next step)

    oldVal  = optVal;
    optVal  = krylov_evals(optIdx);
    absDiff = std::abs(oldVal - optVal);
    relDiff = absDiff / (static_cast<RealScalar>(0.5) * std::abs(optVal + oldVal));
    iter++;

    if(absDiff < absDiffTol) {
        exitMsg.emplace_back(std::format("saturated: abs diff {:.3e} < tol {:.3e}", std::abs(oldVal - optVal), absDiffTol));
        exit |= LanczosExit::saturated_absDiffTol;
    }
    if(relDiff < relDiffTol) {
        exitMsg.emplace_back(std::format("saturated: rel diff {:.3e} < {:.3e}", relDiff, relDiffTol));
        exit |= LanczosExit::saturated_relDiffTol;
    }
    if(iter >= std::max<size_t>(1ul, max_iters)) {
        exitMsg.emplace_back(fmt::format("iter ({}) >= maxiter ({})", iter, max_iters));
        exit |= LanczosExit::max_iterations;
    }

    if(mixedColOk.size() == 1) {
        exitMsg.emplace_back(fmt::format("saturated: only one valid eigenvector"));
        exit |= LanczosExit::one_valid_eigenvector;
    }

    if(mixedColOk.empty()) {
        exitMsg.emplace_back(fmt::format("mixedColOk is empty"));
        exit |= LanczosExit::no_valid_eigenvector;
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
    auto  t_mixblk   = tid::tic_scope("eigs-h1h2");
    auto &sites      = initial.get_sites();
    auto  mpos       = model.get_mpo(sites);
    auto  enve       = edges.get_multisite_env_ene(sites);
    auto  envv       = edges.get_multisite_env_var(sites);

    long ncv = opt_meta.eigs_ncv.value_or(3);
    if(ncv <= 0) { ncv = safe_cast<int>(std::ceil(std::log2(initial.get_tensor().size()))); }

    Lanczos<CalcType> lanczos(ncv, opt_meta.optAlgo, opt_meta.optRitz, initial.template get_vector_as<CalcType>(), mpos, enve, envv);
    lanczos.status.initVal = initial.get_energy();
    lanczos.max_iters      = opt_meta.eigs_iter_max.value_or(1000);
    lanczos.tol            = opt_meta.eigs_tol.has_value() ? static_cast<RealScalar>(opt_meta.eigs_tol.value()) //
                                                           : std::numeric_limits<RealScalar>::epsilon() * 10000;

    while(true) {
        do_lanczos_step(lanczos);
        if(lanczos.status.exit != LanczosExit::ok) break;
    }

    // Extract solution
    opt_mps<Scalar> res;
    res.is_basis_vector = false;
    res.set_name(fmt::format("eigenvector 0 [lanczos h1h2]"));
    res.set_tensor(Eigen::TensorMap<Eigen::Tensor<CalcType, 3>>(lanczos.V.col(0).data(), lanczos.mps_shape));
    res.set_overlap(std::abs(initial.get_vector().dot(res.get_vector())));
    res.set_sites(initial.get_sites());
    res.set_eshift(initial.get_eshift()); // Will set energy if also given the eigval
    res.set_eigs_idx(0);
    res.set_eigs_nev(1);
    res.set_eigs_ncv(ncv);
    res.set_eigs_tol(lanczos.tol);
    res.set_eigs_ritz(enum2sv(opt_meta.optRitz));
    res.set_optalgo(opt_meta.optAlgo);
    res.set_optsolver(opt_meta.optSolver);
    res.set_energy_shifted(initial.get_energy_shifted());

    res.set_length(initial.get_length());
    res.set_time(t_mixblk->get_last_interval());
    res.set_time_mv(lanczos.H1.t_multAx->get_time() + lanczos.H2.t_multAx->get_time());
    res.set_time_pc(lanczos.H1.t_multPc->get_time() + lanczos.H2.t_multPc->get_time());
    res.set_op(lanczos.H1.num_op + lanczos.H2.num_op);
    res.set_mv(lanczos.H1.num_mv + lanczos.H2.num_mv);
    res.set_pc(lanczos.H1.num_pc + lanczos.H2.num_pc);
    res.set_iter(lanczos.status.iter);
    res.set_eigs_rnorm(lanczos.status.rnorm);
    res.set_eigs_eigval(static_cast<fp64>(lanczos.status.optVal));

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

template<typename CalcType, typename Scalar>
opt_mps<Scalar> eigs_lanczos_h1h2_old(const opt_mps<Scalar>                      &initial,  //
                                      [[maybe_unused]] const StateFinite<Scalar> &state,    //
                                      const ModelFinite<Scalar>                  &model,    //
                                      const EdgesFinite<Scalar>                  &edges,    //
                                      OptMeta                                    &opt_meta, //
                                      reports::eigs_log<Scalar>                  &elog) {
    auto t_mixblk = tid::tic_scope("eigs-h1h2");
    auto K1_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG);
    auto K2_on    = has_any_flags(opt_meta.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG);
    auto sites    = initial.get_sites();
    auto mpos     = model.get_mpo(sites);
    auto enve     = edges.get_multisite_env_ene(sites);
    auto envv     = edges.get_multisite_env_var(sites);
    auto H1       = MatVecMPOS<CalcType>(mpos, enve);
    auto H2       = MatVecMPOS<CalcType>(mpos, envv);
    using tools::finite::opt::MatrixType;
    using tools::finite::opt::RealScalar;
    using tools::finite::opt::VectorReal;

    using CalcReal = RealScalar<CalcType>;
    using VectorCR = VectorReal<CalcType>;
    using MatrixCT = MatrixType<CalcType>;

    // using RealScalar  = decltype(std::real(std::declval<CalcType>()));
    // using CplxScalar  = std::complex<RealScalar>;
    // using MatrixType  = Eigen::Matrix<CalcType, Eigen::Dynamic, Eigen::Dynamic>;
    // using MatrixCplx  = Eigen::Matrix<CplxScalar, Eigen::Dynamic, Eigen::Dynamic>;
    // using VectorReal  = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    auto nonZeroCols = std::vector<long>();

    auto mps_size  = H1.get_size();
    auto mps_shape = H1.get_shape_mps();
    long ncv       = opt_meta.eigs_ncv.value_or(3);
    if(ncv <= 0) { ncv = safe_cast<int>(std::ceil(std::log2(initial.get_tensor().size()))); }

    auto     H1V = MatrixCT();
    auto     H2V = MatrixCT();
    MatrixCT K1  = MatrixCT::Zero(ncv, ncv);
    MatrixCT K2  = MatrixCT::Zero(ncv, ncv);

    if(K1_on) H1V.resize(mps_size, ncv);
    if(K2_on) H2V.resize(mps_size, ncv);

    // Default solution
    opt_mps<Scalar> result;
    result.set_tensor(initial.template get_tensor_as<Scalar>());

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
    MatrixCT V(mps_size, ncv);
    V.col(0) = initial.template get_vector_as<CalcType>();

    auto                      mixedColOk = std::vector<long>(); // New states with acceptable norm and eigenvalue
    constexpr auto            eps        = std::numeric_limits<CalcReal>::epsilon();
    CalcReal                  optVal     = std::numeric_limits<CalcReal>::quiet_NaN();
    CalcReal                  oldVal     = std::numeric_limits<CalcReal>::quiet_NaN();
    CalcReal                  relVal     = std::numeric_limits<CalcReal>::quiet_NaN();
    long                      optIdx     = 0;
    CalcReal                  tol        = static_cast<CalcReal>(opt_meta.eigs_tol.value_or(settings::precision::eigs_tol_max));
    CalcReal                  absTol     = eps * static_cast<CalcReal>(1e2);
    CalcReal                  relTol     = std::sqrt(eps); // 1e-4
    CalcReal                  rnorm      = 1.0;
    [[maybe_unused]] CalcReal snorm      = 1.0;
    size_t                    iter       = 0;
    size_t                    ngs        = 0;
    std::string               exit_msg;
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
            auto t_mgs  = tid::tic_token("mgs");
            auto mgs    = linalg::matrix::modified_gram_schmidt(V);
            V           = std::move(mgs.Q);
            nonZeroCols = std::move(mgs.nonZeroCols);
            ngs++;
            if(nonZeroCols.size() == mgs.Q.cols()) break;
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
        auto     t_eigsol      = tid::tic_scope("eigsol");
        long     numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
        long     numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
        long     numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
        VectorCR evals; // Eigen::VectorXd ::Zero();
        MatrixCT evecs; // Eigen::MatrixXcd::Zero();
        OptRitz  ritz_internal = opt_meta.optRitz;
        switch(opt_meta.optAlgo) {
            using enum OptAlgo;
            case DMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K1, Eigen::ComputeEigenvectors);
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
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case XDMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case GDMRG: {
                if(nonZeroCols.empty() and numZeroRows == 0) {
                    auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixCT>(
                        K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                    evals = solver.eigenvalues().real();
                    evecs = solver.eigenvectors().colwise().normalized();
                } else {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
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
        auto t_checks       = tid::tic_scope("checks");
        snorm               = static_cast<CalcReal>(evals.cwiseAbs().maxCoeff());
        V                   = (V * evecs.real()).eval(); // Now V has ncv columns mixed according to evecs
        VectorCR mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
        mixedColOk.clear();                              // New states with acceptable norm and eigenvalue
        mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
        for(long i = 0; i < mixedNorms.size(); ++i) {
            if(std::abs(mixedNorms(i) - static_cast<CalcReal>(1.0)) > static_cast<CalcReal>(settings::precision::max_norm_error)) continue;
            // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
            // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues
            // of GDMRG are not
            mixedColOk.emplace_back(i);
        }
        if constexpr(!tenx::sfinae::is_quadruple_prec_v<CalcType>) {
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
                tools::log->debug("nonOrthoCols           = {}", nonZeroCols);
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
                (evals(mixedColOk).array() - static_cast<CalcReal>(initial.get_energy())).cwiseAbs().minCoeff(&colIdx);
            }
        }
        optIdx = mixedColOk[colIdx];

        oldVal = optVal;
        optVal = evals(optIdx);
        relVal = std::abs((oldVal - optVal) / (static_cast<CalcReal>(0.5) * (optVal + oldVal)));

        // Check convergence

        // If we make it here: update the solution
        result.set_tensor(Eigen::TensorMap<Eigen::Tensor<CalcType, 3>>(V.col(optIdx).data(), mps_shape));
        VectorCR col = evecs.col(optIdx).real();
        // res.alpha_mps       = col.coeff(0);
        // res.alpha_h1v       = col.coeff(1);
        // res.alpha_h2v       = col.coeff(ncv / 2 + 1);

        if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others)

        if(iter + 1 < opt_meta.eigs_iter_max)
            tools::log->trace("lanczos: {:.34f} [{}] | sites {} (size {}) | rnorm {:.3e} | ngs {} | iters {} | "
                              "{:.3e} it/s |  {:.3e} s",
                              fp(optVal), optIdx, sites, mps_size, fp(rnorm), ngs, iter, iter / t_mixblk->get_last_interval(), t_mixblk->get_last_interval());

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
    result.set_eigs_rnorm(rnorm);
    result.set_rnorm_H1((H1V.col(0) - optVal * V.col(0)).norm());
    result.set_rnorm_H2((H2V.col(0) - optVal * V.col(0)).norm());
    result.set_eigs_eigval(static_cast<fp64>(optVal));
    auto vh1v = RealScalar<Scalar>(std::real(V.col(0).dot(H1V.col(0))));
    auto vh2v = RealScalar<Scalar>(std::real(V.col(0).dot(H2V.col(0))));
    result.set_energy(vh1v + result.get_eshift());
    result.set_hsquared(vh2v);
    if(K1_on) { result.set_variance(vh2v - vh1v * vh1v); }

    tools::log->info("lancsoz {}: {:.34f} [{}] | ⟨H⟩ {:.16f} | ⟨H²⟩ {:.16f} | ⟨H²⟩-⟨H⟩² {:.4e} | sites {} (size {}) | rnorm {:.3e} | ngs {} | iters "
                     "{} | {:.3e} s | {} | var {:.4e}",
                     sfinae::type_name<CalcType>(), fp(optVal), optIdx, fp(result.get_energy()), fp(result.get_hsquared()), fp(result.get_variance()), sites,
                     mps_size, fp(rnorm), ngs, iter, t_mixblk->get_last_interval(), exit_msg, fp(vh2v - vh1v * vh1v));
    elog.eigs_add_entry(result, spdlog::level::debug);
    return result;
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
