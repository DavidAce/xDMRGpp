#include "../../env.h"
#include "../assertions.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "../expansion_terms.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/num.h"
#include "math/svd.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/mps.h"
#include <Eigen/Eigenvalues>

template<typename T, typename Scalar>
void get_optimally_mixed_block(const std::vector<size_t>   &sites, //
                               const StateFinite<Scalar>   &state, //
                               const ModelFinite<Scalar>   &model, //
                               const EdgesFinite<Scalar>   &edges, //
                               BondExpansionConfig          bcfg,  //
                               BondExpansionResult<Scalar> &res) {
    if constexpr(sfinae::is_std_complex_v<T>) {
        using RealT = decltype(std::real(std::declval<T>()));
        if(state.is_real() and model.is_real() and edges.is_real()) { return get_optimally_mixed_block<RealT>(sites, state, model, edges, bcfg, res); }
    }

    auto t_mixblk = tid::tic_scope("mixblk");
    auto K1_on    = has_any_flags(bcfg.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG);
    auto K2_on    = has_any_flags(bcfg.optAlgo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG);

    MatVecMPOS<T> H1 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    MatVecMPOS<T> H2 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    using R          = decltype(std::real(std::declval<T>()));
    using MatrixT    = typename MatVecMPOS<T>::MatrixType;
    // using MatrixR     = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorR    = Eigen::Matrix<R, Eigen::Dynamic, 1>;
    auto nonZeroCols = std::vector<long>();

    auto mps_size  = H1.get_size();
    auto mps_shape = H1.get_shape_mps();
    long ncv       = std::clamp(bcfg.nkrylov, 3ul, 256ul);

    auto H1V = MatrixT();
    auto H2V = MatrixT();
    if(K1_on) H1V.resize(mps_size, ncv);
    if(K2_on) H2V.resize(mps_size, ncv);

    // Default solution
    res.mixed_blk = state.template get_multisite_mps<Scalar>(sites);

    // Initialize Krylov vector 0
    auto V   = MatrixT(mps_size, ncv);
    V.col(0) = tenx::asScalarType<T>(tenx::VectorCast(res.mixed_blk));

    R                  optVal = std::numeric_limits<R>::quiet_NaN();
    long               optIdx = 0;
    R                  tol    = static_cast<R>(settings::precision::eigs_tol_max);
    R                  absTol = std::numeric_limits<R>::epsilon() * 100;
    R                  relTol = R{1e-4f};
    R                  rnorm  = R{1};
    [[maybe_unused]] R snorm  = R{1}; // Estimate the matrix norm from the largest singular value/eigenvalue. Converged if  rnorm  < snorm * tol
    size_t             iter   = 0;
    size_t             numMGS = 0;
    std::string        msg;
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
            nonZeroCols = std::move(mgs.nonZeroCols);
            V           = mgs.Q(Eigen::all, nonZeroCols);
            numMGS++;
            if(nonZeroCols.size() == static_cast<size_t>(mgs.Q.cols())) break;
        }

        // V should now have orthonormal vectors
        if(K1_on) {
            for(long i = 0; i < ncv; ++i) H1.MultAx(V.col(i).data(), H1V.col(i).data());
        }
        if(K2_on) {
            for(long i = 0; i < ncv; ++i) H2.MultAx(V.col(i).data(), H2V.col(i).data());
        }
        if(!std::isnan(optVal)) {
            if(bcfg.optAlgo == OptAlgo::DMRG)
                rnorm = (H1V.col(0) - optVal * V.col(0)).cwiseAbs().maxCoeff();
            else if(bcfg.optAlgo == OptAlgo::GDMRG)
                rnorm = (H1V.col(0) - optVal * H2V.col(0)).cwiseAbs().maxCoeff();
            else
                rnorm = (H2V.col(0) - optVal * V.col(0)).cwiseAbs().maxCoeff();
        }

        if(iter >= 1ul and rnorm < tol /* * snorm */) {
            msg = fmt::format("converged rnorm {:.3e} < tol {:.3e}", fp(rnorm), fp(tol));
            break;
        }
        auto t_dotprod = tid::tic_scope("dotprod");

        MatrixT K1 = MatrixT::Zero(ncv, ncv);
        if(K1_on) {
            for(long j = 0; j < ncv; ++j) {
                for(long i = j; i < ncv; ++i) { K1(i, j) = V.col(i).dot(H1V.col(j)); }
            }
            K1 = K1.template selfadjointView<Eigen::Lower>();
        }

        MatrixT K2 = MatrixT::Zero(ncv, ncv);
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
        auto t_eigsol = tid::tic_scope("eigsol");
        // auto    eps           = std::numeric_limits<R>::epsilon();
        long    numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() < std::numeric_limits<double>::epsilon()).count();
        long    numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() < std::numeric_limits<double>::epsilon()).count();
        long    numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
        VectorR evals; // Eigen::VectorXd ::Zero();
        MatrixT evecs; // Eigen::MatrixXcd::Zero();
        OptRitz ritz_internal = bcfg.optRitz;
        switch(bcfg.optAlgo) {
            using enum OptAlgo;
            case DMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K1, Eigen::ComputeEigenvectors);
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
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case XDMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case GDMRG: {
                if(numZeroRows == 0) {
                    auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixT>(
                        K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                    evals = solver.eigenvalues().real();
                    evecs = solver.eigenvectors().colwise().normalized();
                } else {
                    auto solver = Eigen::SelfAdjointEigenSolver<MatrixT>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                    evals       = solver.eigenvalues();
                    evecs       = solver.eigenvectors();
                    if(bcfg.optRitz == OptRitz::LM) ritz_internal = OptRitz::SM;
                    if(bcfg.optRitz == OptRitz::LR) ritz_internal = OptRitz::SM;
                    if(bcfg.optRitz == OptRitz::SM) ritz_internal = OptRitz::LM;
                    if(bcfg.optRitz == OptRitz::SR) ritz_internal = OptRitz::LR;
                }

                break;
            }
        }
        auto t_checks      = tid::tic_scope("checks");
        snorm              = static_cast<R>(evals.cwiseAbs().maxCoeff());
        V                  = (V * evecs.real()).eval(); // Now V has ncv columns mixed according to evecs
        VectorR mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
        auto    mixedColOk = std::vector<long>();       // New states with acceptable norm and eigenvalue
        mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
        auto normTol = std::numeric_limits<R>::epsilon() * settings::precision::max_norm_slack;
        for(long i = 0; i < mixedNorms.size(); ++i) {
            if(std::abs(mixedNorms(i) - R{1}) > normTol) continue;
            // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
            // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues of
            // GDMRG are not
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
                tools::log->debug("ngramSchmidt           = {}", numMGS);
                if(bcfg.optAlgo == OptAlgo::GDMRG) {
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
        if(mixedColOk.empty()) {
            msg = fmt::format("mixedColOk is empty");
            break;
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
                (evals(mixedColOk).array() - static_cast<R>(res.ene_old)).cwiseAbs().minCoeff(&colIdx);
            }
        }
        optIdx = mixedColOk[colIdx];

        auto oldVal = optVal;
        optVal      = evals(optIdx);
        auto relval = std::abs((oldVal - optVal) / (R{0.5} * (optVal + oldVal)));

        // Check convergence
        if(std::abs(oldVal - optVal) < absTol) {
            msg = fmt::format("saturated: abs change {:.3e} < 1e-14", fp(std::abs(oldVal - optVal)));
            break;
        }
        if(relval < relTol) {
            msg = fmt::format("saturated: rel change ({:.3e}) < 1e-4", fp(relval));
            break;
        }

        // If we make it here: update the solution
        res.mixed_blk = tenx::asScalarType<Scalar>(tenx::TensorCast(V.col(optIdx), mps_shape));
        VectorR col   = evecs.col(optIdx).real();

        if(mixedColOk.size() == 1) {
            msg = fmt::format("saturated: only one valid eigenvector");
            break;
        }

        if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others)
        if(iter + 1 < bcfg.maxiter)
            tools::log->debug("bond expansion result: {:.16f} [{}] | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters {} | "
                              "{:.3e} it/s |  {:.3e} s",
                              fp(optVal), optIdx, sites, mps_size, fp(V.col(0).norm()), fp(rnorm), numMGS, iter, iter / t_mixblk->get_last_interval(),
                              t_mixblk->get_last_interval());

        iter++;
        if(iter >= std::max(1ul, bcfg.maxiter)) {
            msg = fmt::format("iter ({}) >= maxiter ({})", iter, bcfg.maxiter);
            break;
        }
    }

    tools::log->debug("mixed state result: {:.16f} [{}] | ncv {} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters "
                      "{} | {:.3e} s | {}",
                      fp(optVal), optIdx, ncv, sites, mps_size, fp(V.col(0).norm()), fp(rnorm), numMGS, iter, t_mixblk->get_last_interval(), msg);
}

template<typename Scalar>
BondExpansionResult<Scalar> get_mixing_factors_preopt_krylov(const std::vector<size_t> &sites, const StateFinite<Scalar> &state,
                                                             const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg) {
    // using R = decltype(std::real(std::declval<Scalar>()));
    tools::finite::env::assert_edges_ene(state, model, edges);
    tools::finite::env::assert_edges_var(state, model, edges);
    auto res         = BondExpansionResult<Scalar>();
    res.sites        = sites;
    res.dims_old     = state.get_mps_dims(sites);
    res.bond_old     = state.get_bond_dims(sites);
    res.posL         = safe_cast<long>(sites.front());
    res.posR         = safe_cast<long>(sites.back());
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    res.dimL_old     = mpsL.dimensions();
    res.dimR_old     = mpsR.dimensions();

    res.ene_old = tools::finite::measure::energy(state, model, edges);
    res.var_old = tools::finite::measure::energy_variance(state, model, edges);
    /*! For PREOPT_NSITE_REAR and PREOPT_NSITE_FORE:
        - the expansion occurs just before the main DMRG optimization step.
        - the expansion involves [active sites] plus sites behind or ahead.
        - at least two sites are used, the upper limit depends on dmrg_blocksize
        - on these sites, we find α that minimizes f(Ψ'), where Ψ' = (α₀ + α₁ H¹ + α₂ H²)Ψ, and f is
          the relevant objective function (energy, variance or <H>/<H²>).
        - note that no zero-padding is used here.
       In multisie DMRG we can afford to estimate mixing factors using multiple sites.
        This technique should use at least two sites. Note that it is possible to use this technique
        with a single site in principle, but it underestimates the mixing factors by several orders of magnitude,
        leading to poor convergence.
     */
    // auto bcfg2 = bcfg;
    // bcfg2.optRitz = OptRitz::SM;
    // bcfg2.optAlgo = OptAlgo::XDMRG;
    get_optimally_mixed_block<Scalar>(sites, state, model, edges, bcfg, res);
    return res;
}

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::expand_bond_preopt_nsite(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                         EdgesFinite<Scalar> &edges, BondExpansionConfig bcfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_forward_nsite: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_bond_forward_nsite: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for bond expansion");

    bool nopreopt = !has_any_flags(bcfg.policy, BondExpansionPolicy::PREOPT_NSITE_REAR, BondExpansionPolicy::PREOPT_NSITE_FORE);
    if(nopreopt) throw except::logic_error("expand_bond_ssite_preopt: BondExpansionPolicy::PREOPT_NSITE_REAR|PREOPT_NSITE_FORE was not set in bcfg.policy");

    // Determine which bond to expand
    // We need at least 1 extra site, apart from the active site(s), to expand the environment for the upcoming optimization.
    size_t blocksize = std::max(bcfg.blocksize, state.active_sites.size() + 1);
    size_t posL      = state.active_sites.front();
    size_t posR      = state.active_sites.back();
    size_t length    = state.template get_length<size_t>();

    // Grow the posL and posR boundary until they cover the block size
    long poslL      = safe_cast<long>(posL);
    long poslR      = safe_cast<long>(posR);
    long lengthl    = safe_cast<long>(length);
    long blocksizel = safe_cast<long>(blocksize);
    if(has_flag(bcfg.policy, BondExpansionPolicy::PREOPT_NSITE_FORE)) {
        if(state.get_direction() > 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
        if(state.get_direction() < 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
    } else if(has_flag(bcfg.policy, BondExpansionPolicy::PREOPT_NSITE_REAR)) {
        if(state.get_direction() > 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
        if(state.get_direction() < 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
    }

    posL = safe_cast<size_t>(poslL);
    posR = safe_cast<size_t>(poslR);
    if(posR - posL + 1 > blocksize) throw except::logic_error("error in block size selection | posL {} to posR {} != blocksize {}", posL, posR, blocksize);

    auto pos_active_and_expanded = num::range<size_t>(posL, posR + 1);

    if(pos_active_and_expanded.size() < 2ul) { return BondExpansionResult<Scalar>(); }

    // Define the left and right mps that will get modified
    auto        res  = get_mixing_factors_preopt_krylov(pos_active_and_expanded, state, model, edges, bcfg);
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    assert(mpsL.get_chiR() == mpsR.get_chiL());
    // assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    res.ene_old = tools::finite::measure::energy(state, model, edges);
    res.var_old = tools::finite::measure::energy_variance(state, model, edges);

    tools::log->debug("Expanding {}({}) - {}({})", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(), mpsR.get_position());

    auto svd_cfg = svd::config(bcfg.bond_limit, bcfg.trnc_limit);

    mps::merge_multisite_mps(state, res.mixed_blk, pos_active_and_expanded, state.template get_position<long>(), MergeEvent::EXP, svd_cfg);

    res.dims_new = state.get_mps_dims(pos_active_and_expanded);
    res.bond_new = state.get_bond_dims(pos_active_and_expanded);

    tools::log->debug("Bond expansion pos {} | {} {} | svd_ε {:.2e} | χlim {} | χ {} -> {}", pos_active_and_expanded,
                      enum2sv(bcfg.optAlgo), enum2sv(bcfg.optRitz), bcfg.trnc_limit, bcfg.bond_limit, res.bond_old, res.bond_new);
    state.clear_cache();
    state.clear_measurements();
    for(const auto &mps : state.get_mps(pos_active_and_expanded)) mps.get().assert_normalized();
    env::rebuild_edges(state, model, edges);

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();

    assert(num::prod(res.dimL_new) > 0);
    assert(num::prod(res.dimR_new) > 0);
    assert(mpsL.get_chiR() == mpsR.get_chiL());

    res.ene_new = tools::finite::measure::energy(state, model, edges);
    res.var_new = tools::finite::measure::energy_variance(state, model, edges);
    if(std::isnan(res.ene_new)) throw except::runtime_error("res.ene_new is nan");
    if(std::isnan(res.var_new)) throw except::runtime_error("res.var_new is nan");
    res.ok = true;
    return res;
}
