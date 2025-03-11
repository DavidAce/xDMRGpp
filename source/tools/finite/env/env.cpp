#include "../env.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "EnvExpansionResult.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/matrix.h"
#include "math/num.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include "tools/finite/measure.h"
#include "tools/finite/mps.h"
#include <Eigen/Eigenvalues>

namespace settings {
    static constexpr bool debug_edges     = false;
    static constexpr bool debug_expansion = false;
}

std::tuple<long, double, bool> get_optimal_eigenvalue(long numZeroEigenvalues, Eigen::Vector3d evals, double oldVal, OptRitz ritz) {
    long   optIdx    = 0;
    double optVal    = 0;
    bool   saturated = false;
    auto   eps       = std::max(1.0, evals.cwiseAbs().maxCoeff()) * std::numeric_limits<double>::epsilon();
    switch(ritz) {
        case OptRitz::SR: {
            if(numZeroEigenvalues > 0) { evals = (evals.cwiseAbs().array() < eps).select(evals.maxCoeff() + 1, evals).eval(); }
            [[maybe_unused]] auto tmp = evals.minCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = oldVal <= optVal;
            break;
        }
        case OptRitz::SM: {
            optIdx    = numZeroEigenvalues;
            optVal    = evals.coeff(optIdx);
            saturated = std::abs(oldVal) <= std::abs(optVal);
            break;
        }
        case OptRitz::LR: {
            if(numZeroEigenvalues > 0) evals = (evals.cwiseAbs().array() < eps).select(evals.minCoeff() - 1, evals).eval();
            [[maybe_unused]] auto tmp = evals.maxCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = oldVal >= optVal;
            break;
        }
        case OptRitz::LM: {
            [[maybe_unused]] auto tmp = evals.cwiseAbs().maxCoeff(&optIdx);
            optVal                    = evals.coeff(optIdx);
            saturated                 = std::abs(oldVal) >= std::abs(optVal);
            break;
        }
        default: {
            // Take the closest to the old value
            optVal    = (evals.array() - oldVal).cwiseAbs().minCoeff(&optIdx);
            optVal    = evals.coeff(optIdx);
            saturated = std::abs(oldVal - optVal) < std::numeric_limits<fp64>::epsilon();

            break;
        }
    }
    return {optIdx, optVal, saturated};
}

template<typename Scalar>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, std::vector<long>>
    modified_gram_schmidt(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q) {
    auto t_eigsol = tid::tic_scope("gramschmidt");

    // Orthonormalize with Modified Gram Schmidt
    using MatrixType  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    auto nonOrthoCols = std::vector<long>();
    auto validCols    = std::vector<long>();
    validCols.reserve(Q.cols());
    nonOrthoCols.reserve(Q.cols());

    for(long i = 0; i < Q.cols(); ++i) {
        auto norm = Q.col(i).norm();
        if(std::abs(norm) < 1e2 * std::numeric_limits<fp64>::epsilon()) { continue; }
        Q.col(i) /= norm;
        for(long j = i + 1; j < Q.cols(); ++j) { Q.col(j) -= Q.col(i).dot(Q.col(j)) * Q.col(i); }
    }
    MatrixType     Qid             = MatrixType::Zero(Q.cols(), Q.cols());
    constexpr auto max_ortho_error = 1e-14;
    for(long j = 0; j < Qid.cols(); ++j) {
        for(long i = 0; i <= j; ++i) {
            Qid(i, j) = Q.col(i).dot(Q.col(j));
            Qid(j, i) = Qid(i, j);
        }
        if(j == 0 and std::abs(Qid(j, j) - 1.0) > max_ortho_error) {
            nonOrthoCols.emplace_back(j);
        } else if(j > 0 and std::abs(Qid(j, j) - 1.0 + Qid.col(j).topRows(j).cwiseAbs().sum()) > max_ortho_error) {
            nonOrthoCols.emplace_back(j);
        }

        if(j == 0 and std::abs(Qid(j, j) - 1.0) <= max_ortho_error) {
            validCols.emplace_back(j);
        } else if(j > 0 and std::abs(Qid(j, j) - 1.0 + Qid.col(j).topRows(j).cwiseAbs().sum()) <= max_ortho_error) {
            validCols.emplace_back(j);
        }
    }

    if(!Qid(validCols, validCols).isIdentity(settings::precision::max_norm_error)) {
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
EnvExpansionResult tools::finite::env::get_optimally_mixed_block_1site(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                       const EdgesFinite &edges, EnvExpandMode envExpandMode) {
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
    auto res         = EnvExpansionResult();
    res.sites        = sites;
    res.dims_old     = state.get_mps_dims(sites);
    res.bond_old     = state.get_bond_dims(sites);
    res.posL         = safe_cast<long>(sites.front());
    res.posR         = safe_cast<long>(sites.back());
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    res.dimL_old     = mpsL.dimensions();
    res.dimR_old     = mpsR.dimensions();
    res.alpha_mps    = 1.0; // Default value
    res.alpha_h1v    = 0.0; // Default value
    res.alpha_h2v    = 0.0; // Default value
    auto H1          = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    auto H2          = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));

    using MatrixType = typename MatVecMPOS<T>::MatrixType;
    std::vector<long> nonOrthoCols;
    auto              size = H1.get_size();
    auto              V    = MatrixType(size, 3);
    auto              H1V  = MatrixType(size, 3);
    auto              H2V  = MatrixType(size, 3);
    V.col(0)               = tenx::VectorCast(state.get_multisite_mps<T>(sites));

    // Determine whether we use a 2 or 3-dimensional krylov subspace
    // If H1_on we use {1, H¹ }|\psi>
    // If H2_on we use {1, H² }|\psi>
    // If H1_on and H2_on we use {1,  H¹, H² }|\psi>
    bool H1_on = has_flag(envExpandMode, EnvExpandMode::H1);
    bool H2_on = has_flag(envExpandMode, EnvExpandMode::H2);

    // Define the krylov subspace v0,  H1v0, H2v0
    H1.MultAx(V.col(0).data(), V.col(1).data());
    H2.MultAx(V.col(0).data(), V.col(2).data());

    // Orthonormalize with Modified Gram Schmidt
    std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
    // V should now have orthonormal columns

    H1.MultAx(V.col(0).data(), H1V.col(0).data());
    H1.MultAx(V.col(1).data(), H1V.col(1).data());
    H1.MultAx(V.col(2).data(), H1V.col(2).data());

    H2.MultAx(V.col(0).data(), H2V.col(0).data());
    H2.MultAx(V.col(1).data(), H2V.col(1).data());
    H2.MultAx(V.col(2).data(), H2V.col(2).data());

    Eigen::Matrix3cd K1;
    K1(0, 0) = V.col(0).dot(H1V.col(0));
    K1(1, 0) = V.col(1).dot(H1V.col(0));
    K1(2, 0) = V.col(2).dot(H1V.col(0));
    K1(1, 1) = V.col(1).dot(H1V.col(1));
    K1(2, 1) = V.col(2).dot(H1V.col(1));
    K1(2, 2) = V.col(2).dot(H1V.col(2));
    K1       = K1.template selfadjointView<Eigen::Lower>();

    Eigen::Matrix3cd K2;
    K2(0, 0) = V.col(0).dot(H2V.col(0));
    K2(1, 0) = V.col(1).dot(H2V.col(0));
    K2(2, 0) = V.col(2).dot(H2V.col(0));
    K2(1, 1) = V.col(1).dot(H2V.col(1));
    K2(2, 1) = V.col(2).dot(H2V.col(1));
    K2(2, 2) = V.col(2).dot(H2V.col(2));
    K2       = K2.template selfadjointView<Eigen::Lower>();

    auto K3 = Eigen::Matrix3cd(K2 - K1 * K1);

    if(!H1_on and K3(2, 2) != 0.0) {
        K3(1, 0) = 0.0;
        K3(1, 1) = 0.0;
        K3(2, 1) = 0.0;
    }
    if(!H2_on and K3(1, 1) != 0.0) {
        K3(2, 0) = 0.0;
        K3(2, 1) = 0.0;
        K3(2, 2) = 0.0;
    }

    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K3.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors);
    auto evals  = solver.eigenvalues();
    auto evecs  = solver.eigenvectors();

    // Calculate the new variances
    // H1.MultAx(V.col(0).data(), H1V.col(0).data());
    // H1.MultAx(V.col(1).data(), H1V.col(1).data());
    // H1.MultAx(V.col(2).data(), H1V.col(2).data());
    // H2.MultAx(V.col(0).data(), H2V.col(0).data());
    // H2.MultAx(V.col(1).data(), H2V.col(1).data());
    // H2.MultAx(V.col(2).data(), H2V.col(2).data());
    // VectorType Vars(3);
    // Vars(0) = V.col(0).dot(H2V.col(0)) - std::pow(V.col(0).dot(H1V.col(0)), 2.0);
    // Vars(1) = V.col(1).dot(H2V.col(1)) - std::pow(V.col(1).dot(H1V.col(1)), 2.0);
    // Vars(2) = V.col(2).dot(H2V.col(2)) - std::pow(V.col(2).dot(H1V.col(2)), 2.0);
    // tools::log->debug("Vars: {}", linalg::matrix::to_string(Vars.real().transpose(), 8));

    // Eigenvalues are sorted in ascending order.
    auto tol           = std::clamp(std::pow(K3(0, 0).real(), 2.0), std::numeric_limits<fp64>::epsilon(), 1e-10);
    long numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
    long numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
    long numZeroRowsK3 = (K3.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
    long numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2, numZeroRowsK3});
    long optIdx        = 0;
    auto optVal        = evals.coeff(optIdx);
    if(numZeroRows > 0) {
        for(long i = 0; i < 3; ++i) { tools::log->debug("V.col({}).norm() = {:.16f}", i, V.col(i).norm()); }
        tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
        tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
        tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
        tools::log->debug("K1: \n{}\n", linalg::matrix::to_string(K1, 8));
        tools::log->debug("K2: \n{}\n", linalg::matrix::to_string(K2, 8));
        tools::log->debug("K3: \n{}\n", linalg::matrix::to_string(K3, 8));
        tools::log->debug("evals: \n{}\n", linalg::matrix::to_string(evals, 8));
        tools::log->debug("evecs: \n{}\n", linalg::matrix::to_string(evecs, 8));
        tools::log->debug("numZeroRowsK3  {}", numZeroRows);
        long                  maxEvecRow0Idx = 0;
        [[maybe_unused]] auto maxEvecRow0Val =
            evecs.row(0).cwiseAbs().maxCoeff(&maxEvecRow0Idx); // Select the column that modifies the current state the least.
        tools::log->debug("optIdx {} -> {} ", optIdx, maxEvecRow0Idx);
        optIdx = maxEvecRow0Idx;
        optVal = evals.coeff(optIdx);
    }

    Eigen::Vector3d col = evecs.col(optIdx).real();
    res.alpha_mps       = col.coeff(0);
    res.alpha_h1v       = col.coeff(1);
    res.alpha_h2v       = col.coeff(2);

    // Define new vectors based on the eigenvalues
    V.col(0) = (V * col).eval();

    tools::log->debug("mixed state result:  <H²>-<H>²= {:.16f} | sites {} (size {})", optVal, sites, size);
    res.mixed_blk = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(0).data(), H1.get_shape_mps()).template cast<cx64>();
    return res;
}

template EnvExpansionResult tools::finite::env::get_optimally_mixed_block_1site<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                      const ModelFinite &model, const EdgesFinite &edges,
                                                                                      EnvExpandMode envExpandMode);
template EnvExpansionResult tools::finite::env::get_optimally_mixed_block_1site<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                      const ModelFinite &model, const EdgesFinite &edges,
                                                                                      EnvExpandMode envExpandMode);
template<typename T>
void tools::finite::env::internal::get_optimally_mixed_block_H1(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                const EdgesFinite &edges, OptRitz ritz, size_t maxiter, EnvExpansionResult &res) {
    if constexpr(std::is_same_v<T, cx64>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block_H1<fp64>(sites, state, model, edges, ritz, maxiter, res);
        }
    }

    auto t_multax = tid::ur("multax");
    auto t_totals = tid::ur("totals");
    t_totals.tic();

    auto H1 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    auto H2 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));

    using MatrixType  = typename MatVecMPOS<T>::MatrixType;
    auto nonOrthoCols = std::vector<long>();
    auto size         = H1.get_size();
    auto V            = MatrixType(size, 3);
    auto H1V          = MatrixType(size, 3);
    auto H2V          = MatrixType(size, 3);
    V.col(0)          = tenx::VectorCast(state.get_multisite_mps<T>(sites));

    double optVal = std::numeric_limits<double>::quiet_NaN();
    double rnorm  = 1.0;
    size_t iter   = 0;

    for(iter = 0; iter < maxiter; ++iter) {
        t_multax.tic();
        // Define the krylov subspace v0,  H1v0, H2v0
        H1.MultAx(V.col(0).data(), V.col(1).data());
        H2.MultAx(V.col(0).data(), V.col(2).data());
        t_multax.toc();

        // Orthonormalize with Modified Gram Schmidt
        std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);

        // V should now have orthonormal vectors
        t_multax.tic();
        H1.MultAx(V.col(0).data(), H1V.col(0).data());
        H1.MultAx(V.col(1).data(), H1V.col(1).data());
        H1.MultAx(V.col(2).data(), H1V.col(2).data());
        t_multax.toc();

        Eigen::Matrix3cd K1;
        K1(0, 0) = V.col(0).dot(H1V.col(0));
        K1(1, 0) = V.col(1).dot(H1V.col(0));
        K1(2, 0) = V.col(2).dot(H1V.col(0));
        K1(1, 1) = V.col(1).dot(H1V.col(1));
        K1(2, 1) = V.col(2).dot(H1V.col(1));
        K1(2, 2) = V.col(2).dot(H1V.col(2));
        K1       = K1.template selfadjointView<Eigen::Lower>();

        auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K1.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors);
        auto evals  = solver.eigenvalues();
        auto evecs  = solver.eigenvectors();

        auto tol                            = std::numeric_limits<fp64>::epsilon();
        long numZeroRowsK1                  = (K1.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long optIdx                         = 0;
        bool saturated                      = false;
        std::tie(optIdx, optVal, saturated) = get_optimal_eigenvalue(numZeroRowsK1, evals, optVal, ritz);

        Eigen::Vector3d col = evecs.col(optIdx).normalized().real();
        res.alpha_mps       = col.coeff(0);
        res.alpha_h1v       = col.coeff(1);
        res.alpha_h2v       = col.coeff(2);

        // Define the new optimal solution
        V.col(0) = (V * col).eval();

        // Check convergence
        auto oldVal = optVal;
        H1.MultAx(V.col(0).data(), H1V.col(0).data());
        optVal = std::real(V.col(0).dot(H1V.col(0)));
        if(std::abs(oldVal - optVal) < 1e-14) break;
        rnorm = (H1V.col(0) - optVal * V.col(0)).norm();
        if(rnorm < settings::precision::eigs_tol_min) break;
        if(numZeroRowsK1 > 0) { break; }
    }

    t_totals.toc();
    tools::log->debug("mixed state result: <H> = {:.16f} | sites {} (size {}) | rnorm {:.3e} | iters {} | t_multax {:.3e} s |  t_totals {:.3e} s", optVal,
                      sites, size, rnorm, iter, t_multax.get_time(), t_totals.get_time());
    res.mixed_blk = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(0).data(), H1.get_shape_mps()).template cast<cx64>();
}

template void tools::finite::env::internal::get_optimally_mixed_block_H1<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                               const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz, size_t maxiter,
                                                                               EnvExpansionResult &res);
template void tools::finite::env::internal::get_optimally_mixed_block_H1<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                               const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz, size_t maxiter,
                                                                               EnvExpansionResult &res);

template<typename T>
void tools::finite::env::internal::get_optimally_mixed_block_H2(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                const EdgesFinite &edges, [[maybe_unused]] OptRitz ritz, size_t maxiter,
                                                                EnvExpansionResult &res) {
    if constexpr(std::is_same_v<T, cx64>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block_H2<fp64>(sites, state, model, edges, ritz, maxiter, res);
        }
    }

    auto t_totals = tid::tic_scope("totals");
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);

    auto H1           = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    auto H2           = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    using MatrixType  = typename MatVecMPOS<T>::MatrixType;
    auto nonOrthoCols = std::vector<long>();
    auto size         = H1.get_size();
    auto V            = MatrixType(size, 3);
    auto H2V          = MatrixType(size, 3);
    V.col(0)          = tenx::VectorCast(state.get_multisite_mps<T>(sites));

    double optVal = std::numeric_limits<double>::quiet_NaN();
    double rnorm  = 1.0;
    size_t iter   = 0;

    for(iter = 0; iter < maxiter; ++iter) {
        auto t_multax = tid::tic_scope("multax");
        // Define the krylov subspace v0,  H1v0, H2v0
        H1.MultAx(V.col(0).data(), V.col(1).data());
        H2.MultAx(V.col(0).data(), V.col(2).data());
        t_multax.toc();

        // Orthonormalize with Modified Gram Schmidt
        std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
        // V should now have orthonormal vectors
        t_multax.tic();
        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        H2.MultAx(V.col(1).data(), H2V.col(1).data());
        H2.MultAx(V.col(2).data(), H2V.col(2).data());
        t_multax.toc();
        auto t_dotprod = tid::tic_scope("dotprod");

        Eigen::Matrix3cd K2;
        K2(0, 0) = V.col(0).dot(H2V.col(0));
        K2(1, 0) = V.col(1).dot(H2V.col(0));
        K2(2, 0) = V.col(2).dot(H2V.col(0));
        K2(1, 1) = V.col(1).dot(H2V.col(1));
        K2(2, 1) = V.col(2).dot(H2V.col(1));
        K2(2, 2) = V.col(2).dot(H2V.col(2));
        K2       = K2.template selfadjointView<Eigen::Lower>();
        t_dotprod.toc();
        auto t_eigsol = tid::tic_scope("eigsol");

        auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K2, Eigen::ComputeEigenvectors);
        auto evals  = solver.eigenvalues();
        auto evecs  = solver.eigenvectors();
        t_eigsol.toc();
        auto t_checks = tid::tic_scope("checks");

        // Eigenvalues are sorted in ascending order.
        long optIdx                = 0;
        V                          = (V * evecs.real()).eval(); // Now V has three columns mixed according o evecs
        Eigen::VectorXd mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
        auto            mixedColOk = std::vector<long>();       // New states with acceptable norm and eigenvalue
        mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
        for(long i = 0; i < mixedNorms.size(); ++i) {
            if(std::abs(mixedNorms(i) - 1.0) > settings::precision::max_norm_error) continue;
            if(evals(i) <= 0) continue;
            mixedColOk.emplace_back(i);
        }

        if(mixedColOk.empty()) {
            tools::log->debug("K2                     : \n{}\n", linalg::matrix::to_string(K2, 8));
            tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
            tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(evecs, 8));
            // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
            tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
            tools::log->debug("mixedColOk             = {}", mixedColOk);
            // tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
            // tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
            // tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
            tools::log->error("mixedColOk is empty");
            break;
        }

        optIdx              = mixedColOk.front(); // The eigenvectors are sorted in increasing eigenvalues, so we just need to take the first.
        Eigen::Vector3d col = evecs.col(optIdx).real();
        res.alpha_mps       = col.coeff(0);
        res.alpha_h1v       = col.coeff(1);
        res.alpha_h2v       = col.coeff(2);

        // Put the best column first (we discard the others)
        if(optIdx != 0) V.col(0) = V.col(optIdx);

        // Check convergence
        auto oldVal = optVal;
        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        optVal      = std::real(V.col(0).dot(H2V.col(0)));
        auto relval = std::abs((oldVal - optVal) / (0.5 * (optVal + oldVal)));

        if(relval < 1e-4) {
            tools::log->debug("saturated: no rel change {:.16f} -> {:.16f} | rel change {:.3e}", oldVal, optVal, relval);
            break;
        }
        rnorm = (H2V.col(0) - optVal * V.col(0)).template lpNorm<1>();
        if(rnorm < settings::precision::eigs_tol_min) {
            tools::log->debug("saturated: rnorm {:.3e} < tol {:.3e}", rnorm, settings::precision::eigs_tol_min);
            break;
        }
        if(mixedColOk.size() == 1) {
            tools::log->debug("saturated: only one valid eigenvector");
            break;
        }
    }

    tools::log->debug("mixed state result: <H²> = {:.16f} | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | iters {} | t_multax "
                      "{:.3e} s |  t_totals {:.3e} s",
                      optVal, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, size, V.col(0).norm(), rnorm, iter, tid::get("t_multax").get_time(),
                      t_totals->get_time());
    res.mixed_blk = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(0).data(), H1.get_shape_mps()).template cast<cx64>();
    t_totals.toc();
}

template void tools::finite::env::internal::get_optimally_mixed_block_H2<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                               const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz, size_t maxiter,
                                                                               EnvExpansionResult &res);
template void tools::finite::env::internal::get_optimally_mixed_block_H2<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                               const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz, size_t maxiter,
                                                                               EnvExpansionResult &res);
template<typename T>
void tools::finite::env::internal::get_optimally_mixed_block_VarH(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                  const EdgesFinite &edges, [[maybe_unused]] OptRitz ritz, size_t maxiter,
                                                                  EnvExpansionResult &res) {
    if constexpr(std::is_same_v<T, cx64>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block_VarH<fp64>(sites, state, model, edges, ritz, maxiter, res);
        }
    }
    auto t_multax = tid::ur("multax");
    auto t_totals = tid::ur("totals");
    t_totals.tic();

    auto H1           = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    auto H2           = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    using MatrixType  = typename MatVecMPOS<T>::MatrixType;
    auto nonOrthoCols = std::vector<long>();
    auto size         = H1.get_size();
    auto V            = MatrixType(size, 3);
    auto H1V          = MatrixType(size, 3);
    auto H2V          = MatrixType(size, 3);
    V.col(0) = tenx::VectorCast(state.get_multisite_mps<T>(sites));

    double optVal = std::numeric_limits<double>::quiet_NaN();
    double rnorm  = 1.0;
    size_t iter   = 0;

    for(iter = 0; iter < maxiter; ++iter) {
        // Define the krylov subspace
        t_multax.tic();
        H1.MultAx(V.col(0).data(), V.col(1).data());
        H2.MultAx(V.col(0).data(), V.col(2).data());
        t_multax.toc();

        // Orthonormalize with Modified Gram Schmidt
        std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);

        // V should now have orthonormal columns
        H1.MultAx(V.col(0).data(), H1V.col(0).data());
        H1.MultAx(V.col(1).data(), H1V.col(1).data());
        H1.MultAx(V.col(2).data(), H1V.col(2).data());

        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        H2.MultAx(V.col(1).data(), H2V.col(1).data());
        H2.MultAx(V.col(2).data(), H2V.col(2).data());

        Eigen::Matrix3cd K1;
        K1(0, 0) = V.col(0).dot(H1V.col(0));
        K1(1, 0) = V.col(1).dot(H1V.col(0));
        K1(2, 0) = V.col(2).dot(H1V.col(0));
        K1(1, 1) = V.col(1).dot(H1V.col(1));
        K1(2, 1) = V.col(2).dot(H1V.col(1));
        K1(2, 2) = V.col(2).dot(H1V.col(2));
        K1       = K1.template selfadjointView<Eigen::Lower>();

        Eigen::Matrix3cd K2;
        K2(0, 0) = V.col(0).dot(H2V.col(0));
        K2(1, 0) = V.col(1).dot(H2V.col(0));
        K2(2, 0) = V.col(2).dot(H2V.col(0));
        K2(1, 1) = V.col(1).dot(H2V.col(1));
        K2(2, 1) = V.col(2).dot(H2V.col(1));
        K2(2, 2) = V.col(2).dot(H2V.col(2));
        K2       = K2.template selfadjointView<Eigen::Lower>();

        auto K3     = Eigen::Matrix3cd(K2 - K1 * K1);
        auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K3, Eigen::ComputeEigenvectors);
        auto evals  = solver.eigenvalues();
        auto evecs  = solver.eigenvectors();

        // Calculate the new variances
        // H1.MultAx(V.col(0).data(), H1V.col(0).data());
        // H1.MultAx(V.col(1).data(), H1V.col(1).data());
        // H1.MultAx(V.col(2).data(), H1V.col(2).data());
        // H2.MultAx(V.col(0).data(), H2V.col(0).data());
        // H2.MultAx(V.col(1).data(), H2V.col(1).data());
        // H2.MultAx(V.col(2).data(), H2V.col(2).data());
        // VectorType Vars(3);
        // Vars(0) = V.col(0).dot(H2V.col(0)) - std::pow(V.col(0).dot(H1V.col(0)), 2.0);
        // Vars(1) = V.col(1).dot(H2V.col(1)) - std::pow(V.col(1).dot(H1V.col(1)), 2.0);
        // Vars(2) = V.col(2).dot(H2V.col(2)) - std::pow(V.col(2).dot(H1V.col(2)), 2.0);
        // tools::log->debug("Vars: {}", linalg::matrix::to_string(Vars.real().transpose(), 8));

        // Eigenvalues are sorted in ascending order.
        auto tol           = std::clamp(std::pow(K3(0, 0).real(), 2.0), std::numeric_limits<fp64>::epsilon(), 1e-10);
        long numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long numZeroRowsK3 = (K3.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2, numZeroRowsK3});
        long optIdx        = 0;
        if(numZeroRows > 0) {
            for(long i = 0; i < 3; ++i) { tools::log->debug("V.col({}).norm() = {:.16f}", i, V.col(i).norm()); }
            tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
            tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
            tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
            tools::log->debug("K1: \n{}\n", linalg::matrix::to_string(K1, 8));
            tools::log->debug("K2: \n{}\n", linalg::matrix::to_string(K2, 8));
            tools::log->debug("K3: \n{}\n", linalg::matrix::to_string(K3, 8));
            tools::log->debug("evals: \n{}\n", linalg::matrix::to_string(evals, 8));
            tools::log->debug("evecs: \n{}\n", linalg::matrix::to_string(evecs, 8));
            tools::log->debug("numZeroRowsK3  {}", numZeroRows);
            long                  maxEvecRow0Idx = 0;
            [[maybe_unused]] auto maxEvecRow0Val =
                evecs.row(0).cwiseAbs().maxCoeff(&maxEvecRow0Idx); // Select the column that modifies the current state the least.
            tools::log->debug("optIdx {} -> {} ", optIdx, maxEvecRow0Idx);
            optIdx = maxEvecRow0Idx;
        }
        Eigen::Vector3d col = evecs.col(optIdx).real();
        res.alpha_mps       = col.coeff(0);
        res.alpha_h1v       = col.coeff(1);
        res.alpha_h2v       = col.coeff(2);

        // Define new vectors based on the best eigencolumn
        V.col(0) = (V * col).eval();

        // Check convergence
        auto oldVal = optVal;
        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        optVal = std::real(V.col(0).dot(H2V.col(0)));
        if(std::abs(oldVal - optVal) < 1e-14) break;
        rnorm = (H2V.col(0) - optVal * V.col(0)).norm();
        if(rnorm < settings::precision::eigs_tol_min) break;
        if(numZeroRows > 0) { break; }
    }

    t_totals.toc();
    tools::log->debug("mixed state result:  <H²>-<H>²= {:.16f} | sites {} (size {}) | rnorm {:.3e} | iters {} | t_multax {:.3e} s |  t_totals {:.3e} s", optVal,
                      sites, size, rnorm, iter, t_multax.get_time(), t_totals.get_time());
    res.mixed_blk = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(0).data(), H1.get_shape_mps()).template cast<cx64>();
}

template void tools::finite::env::internal::get_optimally_mixed_block_VarH<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                 const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz,
                                                                                 size_t maxiter, EnvExpansionResult &res);
template void tools::finite::env::internal::get_optimally_mixed_block_VarH<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                 const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz,
                                                                                 size_t maxiter, EnvExpansionResult &res);

template<typename T>
void tools::finite::env::internal::get_optimally_mixed_block_GsiH(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                  const EdgesFinite &edges, OptRitz ritz, size_t maxiter, EnvExpansionResult &res) {
    if constexpr(std::is_same_v<T, cx64>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block_GsiH<fp64>(sites, state, model, edges, ritz, maxiter, res);
        }
    }

    auto t_multax = tid::ur("multax");
    auto t_totals = tid::ur("totals");
    t_totals.tic();

    auto H1 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    auto H2 = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    // using VectorType = typename MatVecMPOS<T>::VectorType;
    using MatrixType  = typename MatVecMPOS<T>::MatrixType;
    auto nonOrthoCols = std::vector<long>();
    auto size         = H1.get_size();
    auto V            = MatrixType(size, 3);
    auto H1V          = MatrixType(size, 3);
    auto H2V          = MatrixType(size, 3);
    V.col(0) = tenx::VectorCast(state.get_multisite_mps<T>(sites));

    double optVal = std::numeric_limits<double>::quiet_NaN();
    double rnorm  = 1.0;
    size_t iter   = 0;

    for(iter = 0; iter < maxiter; ++iter) {
        t_multax.tic();
        // Define the krylov subspace v0,  H1v0, H2v0
        H1.MultAx(V.col(0).data(), V.col(1).data());
        H2.MultAx(V.col(0).data(), V.col(2).data());
        t_multax.toc();

        // Orthonormalize with Modified Gram Schmidt
        std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
        // V should now have orthonormal vectors
        t_multax.tic();
        H1.MultAx(V.col(0).data(), H1V.col(0).data());
        H1.MultAx(V.col(1).data(), H1V.col(1).data());
        H1.MultAx(V.col(2).data(), H1V.col(2).data());

        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        H2.MultAx(V.col(1).data(), H2V.col(1).data());
        H2.MultAx(V.col(2).data(), H2V.col(2).data());
        t_multax.toc();

        Eigen::Matrix3cd K1;
        K1(0, 0) = V.col(0).dot(H1V.col(0));
        K1(1, 0) = V.col(1).dot(H1V.col(0));
        K1(2, 0) = V.col(2).dot(H1V.col(0));
        K1(1, 1) = V.col(1).dot(H1V.col(1));
        K1(2, 1) = V.col(2).dot(H1V.col(1));
        K1(2, 2) = V.col(2).dot(H1V.col(2));
        K1       = K1.template selfadjointView<Eigen::Lower>();

        Eigen::Matrix3cd K2;
        K2(0, 0) = V.col(0).dot(H2V.col(0));
        K2(1, 0) = V.col(1).dot(H2V.col(0));
        K2(2, 0) = V.col(2).dot(H2V.col(0));
        K2(1, 1) = V.col(1).dot(H2V.col(1));
        K2(2, 1) = V.col(2).dot(H2V.col(1));
        K2(2, 2) = V.col(2).dot(H2V.col(2));
        K2       = K2.template selfadjointView<Eigen::Lower>();

        auto tol           = std::numeric_limits<fp64>::epsilon();
        long numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});

        Eigen::Vector3d evals;
        Eigen::Matrix3d evecs;
        long            optIdx    = 0;
        bool            saturated = 0;
        if(numZeroRows == 0) {
            auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3cd>(
                K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
            evals                               = solver.eigenvalues().real();
            evecs                               = solver.eigenvectors().real();
            std::tie(optIdx, optVal, saturated) = get_optimal_eigenvalue(numZeroRows, evals, optVal, ritz);
            if(optIdx != 0) tools::log->debug("optIdx 0 -> {}", optIdx);
        } else {
            auto K3     = Eigen::Matrix3cd(K2 - K1 * K1);
            auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K3, Eigen::ComputeEigenvectors);
            evals       = solver.eigenvalues().real();
            evecs       = solver.eigenvectors().real();
            for(long i = 0; i < 3; ++i) { tools::log->debug("V.col({}).norm() = {:.16f}", i, V.col(i).norm()); }
            tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
            tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
            tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
            tools::log->debug("K1: \n{}\n", linalg::matrix::to_string(K1, 8));
            tools::log->debug("K2: \n{}\n", linalg::matrix::to_string(K2, 8));
            tools::log->debug("evals: \n{}\n", linalg::matrix::to_string(evals, 8));
            tools::log->debug("evecs: \n{}\n", linalg::matrix::to_string(evecs, 8));
            tools::log->debug("numZeroRowsK3  {}", numZeroRows);
            long                  maxEvecRow0Idx = 0;
            [[maybe_unused]] auto maxEvecRow0Val =
                evecs.row(0).cwiseAbs().maxCoeff(&maxEvecRow0Idx); // Select the column that modifies the current state the least.
            tools::log->debug("optIdx {} -> {} ", optIdx, maxEvecRow0Idx);
            optIdx = maxEvecRow0Idx;
        }

        // Calculate the new variances
        // H1.MultAx(V.col(0).data(), H1V.col(0).data());
        // H1.MultAx(V.col(1).data(), H1V.col(1).data());
        // H1.MultAx(V.col(2).data(), H1V.col(2).data());
        // H2.MultAx(V.col(0).data(), H2V.col(0).data());
        // H2.MultAx(V.col(1).data(), H2V.col(1).data());
        // H2.MultAx(V.col(2).data(), H2V.col(2).data());
        // VectorType Vars(3);
        // Vars(0) = V.col(0).dot(H2V.col(0)) - std::pow(V.col(0).dot(H1V.col(0)), 2.0);
        // Vars(1) = V.col(1).dot(H2V.col(1)) - std::pow(V.col(1).dot(H1V.col(1)), 2.0);
        // Vars(2) = V.col(2).dot(H2V.col(2)) - std::pow(V.col(2).dot(H1V.col(2)), 2.0);
        // tools::log->debug("Vars: {}", linalg::matrix::to_string(Vars.real().transpose(), 8));

        Eigen::Vector3d col = evecs.col(optIdx).normalized().real();
        res.alpha_mps       = col.coeff(0);
        res.alpha_h1v       = col.coeff(1);
        res.alpha_h2v       = col.coeff(2);

        // Define the new optimal solution
        V.col(0) = (V * col).eval();

        // Check convergence
        auto oldVal = optVal;
        H1.MultAx(V.col(0).data(), H1V.col(0).data());
        H2.MultAx(V.col(0).data(), H2V.col(0).data());
        optVal = std::real(V.col(0).dot(H1V.col(0))) / std::real(V.col(0).dot(H2V.col(0)));
        if(std::abs(oldVal - optVal) < 1e-14) {
            tools::log->debug("saturated: no change {:.16f} -> {:.16f}", oldVal, optVal);
            break;
        }
        rnorm = (H1V.col(0) - optVal * H2V.col(0)).norm();
        if(rnorm < settings::precision::eigs_tol_min) {
            tools::log->debug("saturated: rnorm {:.3e} < tol {:.3e}", rnorm, settings::precision::eigs_tol_min);
            break;
        }
        if(numZeroRows > 0) {
            tools::log->debug("saturated: V has zeroed rows");
            break;
        }
    }
    t_totals.toc();
    tools::log->debug("mixed state result: <H>/<H²> = {:.16f} | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | rnorm {:.3e} | iters {} | t_multax {:.3e} s |  "
                      "t_totals {:.3e} s",
                      optVal, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, size, rnorm, iter, t_multax.get_time(), t_totals.get_time());

    res.mixed_blk = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(0).data(), H1.get_shape_mps()).template cast<cx64>();
}

template void tools::finite::env::internal::get_optimally_mixed_block_GsiH<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                 const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz,
                                                                                 size_t maxiter, EnvExpansionResult &res);
template void tools::finite::env::internal::get_optimally_mixed_block_GsiH<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                 const ModelFinite &model, const EdgesFinite &edges, OptRitz ritz,
                                                                                 size_t maxiter, EnvExpansionResult &res);

template<typename T>
void tools::finite::env::internal::get_optimally_mixed_block_any(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                 const EdgesFinite &edges, OptAlgo algo, [[maybe_unused]] OptRitz ritz, size_t maxiter,
                                                                 EnvExpansionResult &res) {
    if constexpr(std::is_same_v<T, cx64>) {
        if(state.is_real() and model.is_real() and edges.is_real()) {
            return tools::finite::env::internal::get_optimally_mixed_block_any<fp64>(sites, state, model, edges, algo, ritz, maxiter, res);
        }
    }

    auto t_totals = tid::tic_scope("totals");
    auto t_multax = tid::tic_scope("multax");
    t_multax.toc();
    auto K1_on = has_any_flags(algo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::DMRG);
    auto K2_on = has_any_flags(algo, OptAlgo::DMRGX, OptAlgo::HYBRID_DMRGX, OptAlgo::GDMRG, OptAlgo::XDMRG);

    MatVecMPOS<T> H1  = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_ene(sites));
    MatVecMPOS<T> H2  = MatVecMPOS<T>(model.get_mpo(sites), edges.get_multisite_env_var(sites));
    using MatrixType  = typename MatVecMPOS<T>::MatrixType;
    auto nonOrthoCols = std::vector<long>();

    auto mps_size  = H1.get_size();
    auto mps_shape = H1.get_shape_mps();

    auto H1V = MatrixType();
    auto H2V = MatrixType();
    if(K1_on) H1V.resize(mps_size, 3);
    if(K2_on) H2V.resize(mps_size, 3);

    // Default solution
    res.mixed_blk = state.get_multisite_mps<cx64>(sites);
    res.alpha_mps = 1.0;
    res.alpha_h1v = 0.0;
    res.alpha_h2v = 0.0;

    // Initialize Krylov vector 0
    auto V = MatrixType(mps_size, 3);
    if constexpr(std::is_same_v<T, fp64>) {
        V.col(0) = tenx::VectorCast(res.mixed_blk).real();
    } else {
        V.col(0) = tenx::VectorCast(res.mixed_blk);
    }

    double optVal = std::numeric_limits<double>::quiet_NaN();
    long   optIdx = 0;
    double rnorm  = 1.0;
    size_t iter   = 0;
    size_t ngs    = 0;
    for(iter = 0; iter < maxiter; ++iter) {
        // Check convergence (residual norm)
        // H2.MultAx(V.col(0).data(), H2V.col(0).data());
        // optVal = std::real();

        // Define the krylov subspace v0,  H1v0, H2v0
        t_multax.tic();
        H1.MultAx(V.col(0).data(), V.col(1).data());
        H2.MultAx(V.col(0).data(), V.col(2).data());
        t_multax.toc();
        // tools::log->debug("V                      : \n{}\n", linalg::matrix::to_string(V, 8));

        // Orthonormalize with Modified Gram Schmidt
        for(size_t igs = 0; igs <= 2; ++igs) {
            std::tie(V, nonOrthoCols) = modified_gram_schmidt(V);
            ngs++;
            // tools::log->debug("V                      : \n{}\n", linalg::matrix::to_string(V, 8));
            // tools::log->debug("nonOrthoCols           = {}", nonOrthoCols);
            // tools::log->debug("V.col(0).dot(V.col(1)) = {:.16f}", V.col(0).dot(V.col(1)));
            // tools::log->debug("V.col(0).dot(V.col(2)) = {:.16f}", V.col(0).dot(V.col(2)));
            // tools::log->debug("V.col(1).dot(V.col(2)) = {:.16f}", V.col(1).dot(V.col(2)));
            if(nonOrthoCols.empty()) break;
        }

        // V should now have orthonormal vectors
        t_multax.tic();
        if(K1_on) {
            H1.MultAx(V.col(0).data(), H1V.col(0).data());
            H1.MultAx(V.col(1).data(), H1V.col(1).data());
            H1.MultAx(V.col(2).data(), H1V.col(2).data());
        }
        if(K2_on) {
            H2.MultAx(V.col(0).data(), H2V.col(0).data());
            H2.MultAx(V.col(1).data(), H2V.col(1).data());
            H2.MultAx(V.col(2).data(), H2V.col(2).data());
        }

        if(algo == OptAlgo::DMRG)
            rnorm = (H1V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();
        else if(algo == OptAlgo::GDMRG)
            rnorm = (H1V.col(0) - optVal * H2V.col(0)).template lpNorm<Eigen::Infinity>();
        else
            rnorm = (H2V.col(0) - optVal * V.col(0)).template lpNorm<Eigen::Infinity>();

        if(rnorm < settings::precision::eigs_tol_min) {
            tools::log->debug("saturated: rnorm {:.3e} < tol {:.3e}", rnorm, settings::precision::eigs_tol_min);
            break;
        }

        t_multax.toc();
        auto t_dotprod = tid::tic_scope("dotprod");

        Eigen::Matrix3cd K1 = Eigen::Matrix3cd::Zero();
        if(K1_on) {
            K1(0, 0) = V.col(0).dot(H1V.col(0));
            K1(1, 0) = V.col(1).dot(H1V.col(0));
            K1(2, 0) = V.col(2).dot(H1V.col(0));
            K1(1, 1) = V.col(1).dot(H1V.col(1));
            K1(2, 1) = V.col(2).dot(H1V.col(1));
            K1(2, 2) = V.col(2).dot(H1V.col(2));
            K1       = K1.template selfadjointView<Eigen::Lower>();
        }

        Eigen::Matrix3cd K2 = Eigen::Matrix3cd::Zero();
        if(K2_on) {
            // Use abs to avoid negative near-zero values
            K2(0, 0) = std::abs(V.col(0).dot(H2V.col(0)));
            K2(1, 0) = V.col(1).dot(H2V.col(0));
            K2(2, 0) = V.col(2).dot(H2V.col(0));
            K2(1, 1) = std::abs(V.col(1).dot(H2V.col(1)));
            K2(2, 1) = V.col(2).dot(H2V.col(1));
            K2(2, 2) = std::abs(V.col(2).dot(H2V.col(2)));
            K2       = K2.template selfadjointView<Eigen::Lower>();
        }

        t_dotprod.toc();
        auto             t_eigsol      = tid::tic_scope("eigsol");
        auto             tol           = std::numeric_limits<fp64>::epsilon();
        long             numZeroRowsK1 = (K1.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long             numZeroRowsK2 = (K2.cwiseAbs().rowwise().maxCoeff().array() <= tol).count();
        long             numZeroRows   = std::max({numZeroRowsK1, numZeroRowsK2});
        Eigen::Vector3d  evals         = Eigen::Vector3d ::Zero();
        Eigen::Matrix3cd evecs         = Eigen::Matrix3cd::Zero();
        OptRitz          ritz_internal = ritz;
        switch(algo) {
            using enum OptAlgo;
            case DMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K1, Eigen::ComputeEigenvectors);
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
                auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case XDMRG: {
                auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K2, Eigen::ComputeEigenvectors);
                evals       = solver.eigenvalues();
                evecs       = solver.eigenvectors();
                break;
            }
            case GDMRG: {
                if(nonOrthoCols.empty() and numZeroRows == 0) {
                    auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3cd>(
                        K1.template selfadjointView<Eigen::Lower>(), K2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                    evals = solver.eigenvalues().real();
                    evecs = solver.eigenvectors().colwise().normalized();
                } else {
                    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd>(K2 - K1 * K1, Eigen::ComputeEigenvectors);
                    evals       = solver.eigenvalues();
                    evecs       = solver.eigenvectors();
                    if(ritz == OptRitz::LM) ritz_internal = OptRitz::SM;
                    if(ritz == OptRitz::LR) ritz_internal = OptRitz::SM;
                    if(ritz == OptRitz::SM) ritz_internal = OptRitz::LM;
                    if(ritz == OptRitz::SR) ritz_internal = OptRitz::LR;
                }

                break;
            }
        }
        auto t_checks = tid::tic_scope("checks");

        V                          = (V * evecs.real()).eval(); // Now V has three columns mixed according o evecs
        Eigen::VectorXd mixedNorms = V.colwise().norm();        // New state norms after mixing cols of V according to cols of evecs
        auto            mixedColOk = std::vector<long>();       // New states with acceptable norm and eigenvalue
        mixedColOk.reserve(static_cast<size_t>(mixedNorms.size()));
        for(long i = 0; i < mixedNorms.size(); ++i) {
            if(std::abs(mixedNorms(i) - 1.0) > settings::precision::max_norm_error) continue;
            // if(algo != OptAlgo::GDMRG and evals(i) <= 0) continue; // H2 and variance are positive definite, but the eigenvalues of GDMRG are not
            // if(algo != OptAlgo::GDMRG and (evals(i) < -1e-15 or evals(i) == 0)) continue; // H2 and variance are positive definite, but the eigenvalues of
            // GDMRG are not
            mixedColOk.emplace_back(i);
        }
        if(mixedColOk.size() <= 1) {
            tools::log->debug("K1                     : \n{}\n", linalg::matrix::to_string(K1, 8));
            tools::log->debug("K2                     : \n{}\n", linalg::matrix::to_string(K2, 8));
            tools::log->debug("evals                  : \n{}\n", linalg::matrix::to_string(evals, 8));
            tools::log->debug("evecs                  : \n{}\n", linalg::matrix::to_string(evecs, 8));
            // tools::log->debug("Vnorms                 = {}", linalg::matrix::to_string(V.colwise().norm().transpose(), 16));
            tools::log->debug("mixedNorms             = {}", linalg::matrix::to_string(mixedNorms.transpose(), 16));
            tools::log->debug("mixedColOk             = {}", mixedColOk);
            tools::log->debug("numZeroRowsK1          = {}", numZeroRowsK1);
            tools::log->debug("numZeroRowsK2          = {}", numZeroRowsK2);
            tools::log->debug("nonOrthoCols           = {}", nonOrthoCols);
            tools::log->debug("ngramSchmidt           = {}", ngs);
            if(algo == OptAlgo::GDMRG) {
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

            if(mixedColOk.empty()) {
                tools::log->debug("mixedColOk is empty");
                break;
            }
        }
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
                (evals(mixedColOk).array() - res.ene_old).cwiseAbs().minCoeff(&colIdx);
            }
        }
        optIdx = mixedColOk[colIdx];

        auto oldVal = optVal;
        optVal      = evals(optIdx);
        auto relval = std::abs((oldVal - optVal) / (0.5 * (optVal + oldVal)));

        // Check convergence
        if(std::abs(oldVal - optVal) < 1e-14) {
            tools::log->debug("saturated: no change {:.16f} -> {:.16f}", oldVal, optVal);
            break;
        }
        if(relval < 1e-12) {
            tools::log->debug("saturated: no rel change {:.16f} -> {:.16f} | rel change {:.3e}", oldVal, optVal, relval);
            break;
        }

        // If we make it here: update the solution
        res.mixed_blk       = Eigen::TensorMap<Eigen::Tensor<T, 3>>(V.col(optIdx).data(), mps_shape).template cast<cx64>();
        Eigen::Vector3d col = evecs.col(optIdx).real();
        res.alpha_mps       = col.coeff(0);
        res.alpha_h1v       = col.coeff(1);
        res.alpha_h2v       = col.coeff(2);

        if(mixedColOk.size() == 1) {
            tools::log->debug("saturated: only one valid eigenvector");
            break;
        }

        if(optIdx != 0) V.col(0) = V.col(optIdx); // Put the best column first (we discard the others)
        if(iter + 1 < maxiter)
            tools::log->debug(
                "mixed state result: {:.16f} [{}] | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters {} | "
                "{:.3e} it/s |  t_totals {:.3e} s",
                optVal, optIdx, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, mps_size, V.col(0).norm(), rnorm, ngs, iter,
                t_multax->get_time() / t_totals->get_time(), t_totals->get_time());
    }

    tools::log->debug(
        "mixed state result: {:.16f} [{}] | α: {:.3e} {:.3e} {:.3e} | sites {} (size {}) | norm {:.16f} | rnorm {:.3e} | ngs {} | iters {} | t_multax "
        "{:.3e} s |  t_totals {:.3e} s",
        optVal, optIdx, res.alpha_mps, res.alpha_h1v, res.alpha_h2v, sites, mps_size, V.col(0).norm(), rnorm, ngs, iter, t_multax->get_time(),
        t_totals->get_time());
    t_totals.toc();
}

template void tools::finite::env::internal::get_optimally_mixed_block_any<fp64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                const ModelFinite &model, const EdgesFinite &edges, OptAlgo algo, OptRitz ritz,
                                                                                size_t maxiter, EnvExpansionResult &res);
template void tools::finite::env::internal::get_optimally_mixed_block_any<cx64>(const std::vector<size_t> &sites, const StateFinite &state,
                                                                                const ModelFinite &model, const EdgesFinite &edges, OptAlgo algo, OptRitz ritz,
                                                                                size_t maxiter, EnvExpansionResult &res);

EnvExpansionResult tools::finite::env::get_optimally_mixed_block(const std::vector<size_t> &sites, const StateFinite &state, const ModelFinite &model,
                                                                 const EdgesFinite &edges, OptAlgo algo, OptRitz ritz, size_t maxiter) {
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
    auto res         = EnvExpansionResult();
    res.sites        = sites;
    res.dims_old     = state.get_mps_dims(sites);
    res.bond_old     = state.get_bond_dims(sites);
    res.posL         = safe_cast<long>(sites.front());
    res.posR         = safe_cast<long>(sites.back());
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    res.dimL_old     = mpsL.dimensions();
    res.dimR_old     = mpsR.dimensions();

    // switch(algo) {
    //     using namespace internal;
    //     case OptAlgo::DMRG: {
    //         get_optimally_mixed_block_H1<cx64>(sites, state, model, edges, ritz, maxiter, res);
    //         break;
    //     }
    //     case OptAlgo::DMRGX: {
    //         get_optimally_mixed_block_VarH<cx64>(sites, state, model, edges, ritz, maxiter, res);
    //         break;
    //     }
    //     case OptAlgo::HYBRID_DMRGX: {
    //         get_optimally_mixed_block_VarH<cx64>(sites, state, model, edges, ritz, maxiter, res);
    //         break;
    //     }
    //     case OptAlgo::XDMRG: {
    //         get_optimally_mixed_block_H2<cx64>(sites, state, model, edges, ritz, maxiter, res);
    //         break;
    //     }
    //     case OptAlgo::GDMRG: {
    //         get_optimally_mixed_block_GsiH<cx64>(sites, state, model, edges, ritz, maxiter, res);
    //         break;
    //     }
    //     default: throw except::logic_error("get_optimally_mixed_block: algorithm not implemented: {}", enum2sv(algo));
    // }
    internal::get_optimally_mixed_block_any<cx64>(sites, state, model, edges, algo, ritz, maxiter, res);
    // internal::get_optimally_mixed_block_any<cx64>(sites, state, model, edges, OptAlgo::XDMRG, OptRitz::SM, maxiter, res);
    return res;
}

void merge_expansion_term_PR(const StateFinite &state, MpsSite &mpsL, const Eigen::Tensor<cx64, 3> &ML_PL, MpsSite &mpsR, const Eigen::Tensor<cx64, 3> &MR_PR,
                             const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR
    // We zero-pad mpsL and enrich mpsR.
    // During forward expansion -->
    //      * mpsL is AC(i), or B(i) during multisite dmrg
    //      * mpsR is B(i+1)
    // During backward expansion <--
    //      * mpsL is A(i-1) always
    //      * mpsR is AC(i) always
    // Thus, the possible situations are [AC,B] or [B,B] or [A,AC]
    // After USV = SVD(MR_PR):
    //      If [AC,B]:
    //           * ML_PL is [AC(i), PL]            <--- Note that we use full AC(i)! Not bare A(i)
    //           * MR_PR is [B(i+1), PR]^T
    //           * mpsL:  AC(i)   = ML_PL * U     <---- lost left normalization, but it is not needed during optimization
    //           * mpsL:  C(i)   = S
    //           * mpsR:  B(i+1) = V
    //      If [B,B]:
    //           * ML_PL is [B(i), PL]
    //           * MR_PR is [B(i+1), PR]^T
    //           * mpsL:  B(i)   = ML_PL * U * S  <-- lost right normalization, but it is not needed during optimization
    //           * mpsL:  Λ(i)   = S
    //           * mpsR:  B(i+1) = V
    //      If [A,AC]:
    //           * ML_PL is [A(i-1), PL]
    //           * MR_PR is [A(i), PR]^T             <--- Note that we use bare A(i)! Not AC(i)
    //           * mpsL:  A(i-1) = ML_PL * U         <--- lost left normalization, but it will not be needed during the optimization later
    //           * mpsR:  Λ(i) = S, (will become a center later)
    //           * mpsR:  A(i) = S * V (note that we replace bare A(i), not AC(i)!)
    //           * The old C(i) living in mpsR is outdated, but it incorporated into AC(i) when building MR_PR.
    //           * We can simply replace C(i) with an identity matrix. It will not be needed after we transform AC(i) into a B(i) later.
    //

    svd::solver svd;
    auto        posL = mpsL.get_position();
    auto        labL = mpsL.get_label();
    auto        labR = mpsR.get_label();
    auto [U, S, V]   = svd.schmidt_into_right_normalized(MR_PR, mpsR.spin_dim(), svd_cfg);

    if(labL == "AC" and labR == "B") {
        mpsR.set_M(V);
        mpsR.stash_U(U, posL);
        mpsR.stash_C(S, -1.0, posL); // Set a negative truncation error to ignore it.
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR); // normalization of mpsL is lost here.
    } else if(labL == "B" and labR == "B") {
        auto US = Eigen::Tensor<cx64, 3>(U.contract(tenx::asDiagonal(S), tenx::idx({2}, {0})));
        mpsR.set_M(V);
        mpsR.stash_U(US, posL);
        mpsR.stash_S(S, -1.0, posL); // Set a negative truncation error to ignore it.
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR); // normalization of mpsL is lost here.
    } else if(labL == "A" and labR == "AC") {
        auto SV = Eigen::Tensor<cx64, 3>(tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2}));
        auto C1 = Eigen::Tensor<cx64, 1>(V.dimension(2));
        C1.setConstant(cx64(1.0, 0.0)); // Replaces C with an identity
        mpsR.set_M(SV);
        mpsR.set_L(S);
        mpsR.set_LC(C1);
        mpsR.stash_U(U, posL);
        mpsL.set_M(ML_PL);
        mpsL.take_stash(mpsR);
    } else {
        throw except::runtime_error("merge_expansion_term_PR: could not match case: [{},{}]", labL, labR);
    }

    {
        // Make mpsL normalized so that later checks can succeed
        auto                   multisite_mpsL = state.get_multisite_mps<cx64>({posL});
        cx64                   norm_old       = tools::common::contraction::contract_mps_norm(multisite_mpsL);
        Eigen::Tensor<cx64, 3> M_tmp          = mpsL.get_M_bare() * mpsL.get_M_bare().constant(std::pow(norm_old, -0.5)); // Rescale
        mpsL.set_M(M_tmp);
        if constexpr(settings::debug_expansion) {
            auto mpsL_final = state.get_multisite_mps<cx64>({mpsL.get_position()});
            cx64 norm_new   = tools::common::contraction::contract_mps_norm(mpsL_final);
            tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsL.get_label(), mpsL.get_position(), std::abs(norm_old),
                              std::abs(norm_new));
        }
    }
}

void merge_expansion_term_PL(const StateFinite &state, MpsSite &mpsL, const Eigen::Tensor<cx64, 3> &ML_PL, MpsSite &mpsR, const Eigen::Tensor<cx64, 3> &MR_PR,
                             const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR.
    // During forward expansion <--
    //      * mpsL is A(i-1) always
    //      * mpsR is AC(i), or A(i) during multisite dmrg
    // During backward expansion -->
    //      * mpsL is AC(i) always
    //      * mpsR is B(i+1) always
    // Thus, the possible situations are  [A, AC], [A,A] or [AC,B]
    // After USV = SVD(ML_PL):
    //      If [A, AC]:
    //           * ML_PL is [A(i-1), PL]
    //           * MR_PR is [A(i), PR]^T             <--- Note that we use bare A(i)! Not AC(i)
    //           * mpsL:  A(i-1) = U
    //           * mpsR:  Λ(i)   = S                <--- takes stash S
    //           * mpsR:  A(i)   = S * V * MR_PR    <--- takes stash S,V and loses left-right normalization
    //           * mpsR:  C(i)                      <--- does not change
    //      If [A,A]:
    //           * ML_PL is [A(i-1), PL]^T
    //           * MR_PR is [A(i), 0]^T             <--- Note that we use bare A(i)! Not AC(i)
    //           * mpsL:  A(i-1) = U
    //           * mpsR:  Λ(i)   = S (takes stash S)
    //           * mpsR:  A(i) = S * V * MR_PR (takes stash S,SV and loses left normalization)
    //      If [AC,B]:
    //           * ML_PL is [AC(i), PL]^T
    //           * MR_PR is [B(i+1), 0]^T
    //           * mpsL:  A(i) = U
    //           * mpsL:  C(i) = S
    //           * mpsR:  B(i+1) = V * MR_PR (loses right normalization, but that is not be needed during the next optimization)
    //

    svd::solver svd;
    auto        posR = mpsR.get_position();
    auto        labL = mpsL.get_label();
    auto        labR = mpsR.get_label();
    auto [U, S, V]   = svd.schmidt_into_left_normalized(ML_PL, mpsL.spin_dim(), svd_cfg);

    if(labL == "A" and labR == "AC") {
        auto SV = Eigen::Tensor<cx64, 3>(tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2}));
        mpsL.set_M(U);
        mpsL.stash_S(S, -1.0, posR); // Set a negative truncation error to ignore it.
        mpsL.stash_V(SV, posR);
        mpsR.set_M(MR_PR);
        mpsR.take_stash(mpsL); // normalization of mpsR is lost here
    } else if(labL == "A" and labR == "A") {
        auto SV = Eigen::Tensor<cx64, 3>(tenx::asDiagonal(S).contract(V, tenx::idx({1}, {1})).shuffle(tenx::array3{1, 0, 2}));
        mpsL.set_M(U);
        mpsL.stash_S(S, -1.0, posR); // Set a negative truncation error to ignore it.
        mpsL.stash_V(SV, posR);
        mpsR.set_M(MR_PR);
        mpsR.take_stash(mpsL); // normalization of mpsR is lost here
    } else if(labL == "AC" and labR == "B") {
        mpsL.set_M(U);
        mpsL.set_LC(S);
        mpsL.stash_V(V, posR);
        mpsR.set_M(MR_PR);
        mpsR.take_stash(mpsL);
    } else {
        throw except::runtime_error("merge_expansion_term_PL: could not match case: [{},{}]", labL, labR);
    }

    {
        // Make mpsR normalized so that later checks can succeed
        auto                   multisite_mpsR = state.get_multisite_mps<cx64>({posR});
        cx64                   norm_old       = tools::common::contraction::contract_mps_norm(multisite_mpsR);
        Eigen::Tensor<cx64, 3> M_tmp          = mpsR.get_M_bare() * mpsR.get_M_bare().constant(std::pow(norm_old, -0.5)); // Rescale by the norm
        mpsR.set_M(M_tmp);
        if constexpr(settings::debug_expansion) {
            auto mpsR_final = state.get_multisite_mps<cx64>({mpsR.get_position()});
            cx64 norm_new   = tools::common::contraction::contract_mps_norm(mpsR_final);
            tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsR.get_label(), mpsR.get_position(), std::abs(norm_old),
                              std::abs(norm_new));
        }
    }
}

/*!
    This is a modified (aka subspace) expansion technique, compared to the one explained in https://link.aps.org/doi/10.1103/PhysRevB.91.155115
    In this convention we expand/enrich the forward bond during a DMRG sweep, i.e. the bond one step ahead in the sweep direction.

    In 1-site DMRG going left-to-right with active site == i, we update:

        - AC(i), LC(i) -->      [AC(i)     ,  0 ]  * U, S(i)      : AC(i) is bare, without LC(i) (which is used below). Loses left-normalization due to U.
        - LC*B(i+1)    --> SVD( [LC*B(i+1) , PR(i+1) ]^T ) = U*S*V: update B(i+1) = V, LC(i) = S and U is contracted onto AC(i) above

    where PR(i+1) = alpha * ENVR(i+1) * B(i+1) * MPO(i+1) (dimensions d(i-1), B(i+1).dimension(1), B(i+1).dimension(2) * MPO(i+1).dimension(0))
    Then, immediately after calling this function, we should optimize the current site AC(i) with a DMRG step.

    Similarly, in 1-site DMRG going right-to-left with active site == i, we update:

         - A(i-1)       --> SVD( [A(i-1), PL(i-1)] ) = U*S*V : update A(i-1) = U, L(i) = S and S*V is contracted onto A(i) below.
         - L(i), A(i)   --> S, S*V*[ A(i)    ,  0      ]^T     : A(i) (which is AC(i) bare), loses left-normalization due to SV.

    where PL(i-1) = alpha * ENVL(i-1) * A(i-1) * MPO(i-1) (dimensions d(i-1), A(i-1).dimension(2), A(i-1).dimension(0)*MPO(i-1).dimension(1))
    Then, immediately after calling this function, we should optimize the current site AC(i) with a DMRG step.


    Thus, one step of the DMRG algorithm proceeds as

    1. expand between sites i, i+1
    2. optimize $\Psi^{i} = A_C^{i} \Lambda_C^{i}$.
    3. split $\Psi^{i} \stackrel{\text{SVD}}{\rightarrow}A_C^{i}  \Lambda_C^{i} V^\dagger B^{i+1}$ normally and update the MPS on sites $i,i+1$.
    4. move: $A_C^i \Lambda_C^i \rightarrow A^i$ and $\Lambda_C^i B^{i+1} \rightarrow \Lambda^i A_C^{i+1} \Lambda_C^{i+1}$, and repeat from step 1.
    5. update $\alpha$: decrease by $0.01/L$ if **if the variance decreased** in the last step, else increase by $10/L$, bounded by $\alpha \in [\alpha_{\min},
   \alpha_{\max}]$, where:

      - $\alpha_{\min} = \text{Var}(H) \cdot 10^{-3}$,
      - $\alpha_{\max}= \min{(10^{-4}, \text{Var}(H))}$.

    The main differences compared with the original:
    - The SVD in the expansion can be done with high precision and bond dimension: the real truncation happens after the optimization in step 3.
    - Therefore, the optimization step sees a much richer environment, which speeds up convergence.
    - Backwards expansion pushes a non-optimal/degraded site-tensor into the trailing environment. Forward expansion optimizes it first.

    Note that this expansion works similarly for multisite dmrg, with the expanded site chosen accordingly.


*/
EnvExpansionResult tools::finite::env::expand_environment_1site(StateFinite &state, const ModelFinite &model, EdgesFinite &edges, EnvExpandMode envExpandMode,
                                                                [[maybe_unused]] OptAlgo algo, [[maybe_unused]] OptRitz ritz, svd::config svd_cfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_environment_1site: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_environment_1site: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for environment expansion");

    if(!has_flag(envExpandMode, EnvExpandMode::FORE) and !has_flag(envExpandMode, EnvExpandMode::REAR)) envExpandMode |= EnvExpandMode::FORE;
    if(!has_flag(envExpandMode, EnvExpandMode::H1) and !has_flag(envExpandMode, EnvExpandMode::H2)) envExpandMode |= EnvExpandMode::H1 | EnvExpandMode::H2;

    // Determine which bond to expand (FORE has preference if both are specified)
    std::vector<size_t> pos_expanded;
    if(has_flag(envExpandMode, EnvExpandMode::FORE)) {
        if(state.get_direction() > 0) {
            auto pos = state.active_sites.back();
            if(pos == std::clamp<size_t>(pos, 0, state.get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};
        }
        if(state.get_direction() < 0) {
            auto pos = state.active_sites.front();
            if(pos == std::clamp<size_t>(pos, 1, state.get_length<size_t>() - 1)) pos_expanded = {pos - 1, pos};
        }
    } else if(has_flag(envExpandMode, EnvExpandMode::REAR)) {
        auto pos = state.get_position<size_t>();
        if(state.get_direction() > 0 and pos == std::clamp<size_t>(pos, 1, state.get_length<size_t>() - 1)) pos_expanded = {pos - 1, pos};
        if(state.get_direction() < 0 and pos == std::clamp<size_t>(pos, 0, state.get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};
    }
    if(pos_expanded.empty()) {
        tools::log->trace("No positions to expand: active sites {} | mode {}", state.active_sites, flag2str(envExpandMode));
        return {}; // No update
    }

    auto res = EnvExpansionResult();
    if(state.is_real() and model.is_real() and edges.is_real()) {
        res = get_optimally_mixed_block_1site<fp64>(pos_expanded, state, model, edges, envExpandMode);
    } else {
        res = get_optimally_mixed_block_1site<cx64>(pos_expanded, state, model, edges, envExpandMode);
    }
    auto &mpsL     = state.get_mps_site(res.posL);
    auto &mpsR     = state.get_mps_site(res.posR);
    auto  dimL_old = mpsL.dimensions();
    auto  dimR_old = mpsR.dimensions();
    res.ene_old    = tools::finite::measure::energy(state, model, edges);
    res.var_old    = tools::finite::measure::energy_variance(state, model, edges);

    if(res.alpha_h1v == 0 and res.alpha_h2v == 0) {
        tools::log->debug("Expansion canceled: {}({}) - {}({}) | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(),
                          mpsR.get_position(), res.alpha_mps, res.alpha_h1v, res.alpha_h2v);
        return res;
    }

    tools::log->debug("Expanding {}({}) - {}({}) | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(), mpsR.get_position(),
                      res.alpha_mps, res.alpha_h1v, res.alpha_h2v);

    // Set up the SVD
    // Bond dimension can't grow faster than x spin_dim, but we can generate a highly enriched environment here for optimization,
    // and let the proper truncation happen after optimization instead.
    auto bond_lim            = std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR());
    svd_cfg.truncation_limit = svd_cfg.truncation_limit.value_or(settings::precision::svd_truncation_min);
    svd_cfg.rank_max         = std::min(bond_lim, svd_cfg.rank_max.value_or(bond_lim));
    svd::solver svd;

    if(has_flag(envExpandMode, EnvExpandMode::FORE)) {
        if(state.get_direction() > 0) {
            // The expanded bond sits between mpsL and mpsR. When direction is left-to-right:
            // We have
            //      * mpsL is AC(i), or B(i) during multisite dmrg
            //      * mpsR is B(i+1) and belongs in envR later in the optimization step
            // auto &mpoL = model.get_mpo(res.posL);
            auto &mpoR = model.get_mpo(safe_cast<size_t>(res.posR));
            // Eigen::Tensor<cx64, 3> ML_PL = mpsL.get_M();
            Eigen::Tensor<cx64, 3> MR_PR =
                mpsR.get_M() * mpsR.get_M().constant(cx64(res.alpha_mps, 0.0)); // mpsR is going into the environment, enriched with PR.
            long PRdim1 = 0;
            if(res.alpha_h1v != 0) {
                // auto                  &envL    = edges.get_env_eneL(res.posL);
                auto &envR    = edges.get_env_eneR(safe_cast<size_t>(res.posR));
                long  chi_max = 2 * bond_lim / (1 + mpoR.MPO().dimension(0));
                // Eigen::Tensor<cx64, 3> PL      = envL.get_expansion_term(mpsL, mpoL, 1.0, -1);
                Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h1v, chi_max);
                // Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h1v, -1);
                // Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                Eigen::Tensor<cx64, 3> MR_PR_tmp = MR_PR.concatenate(PR, 1);
                // ML_PL                            = std::move(ML_PL_tmp);
                MR_PR = std::move(MR_PR_tmp);
                PRdim1 += PR.dimension(1);
                res.env_ene = envR;
            }
            if(res.alpha_h2v != 0) {
                // auto                  &envL    = edges.get_env_varL(res.posL);
                auto &envR    = edges.get_env_varR(safe_cast<size_t>(res.posR));
                long  chi_max = 2 * bond_lim / (1 + mpoR.MPO2().dimension(0));
                // Eigen::Tensor<cx64, 3> PL      = envL.get_expansion_term(mpsL, mpoL, 1.0, -1);
                Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h2v, chi_max);
                // Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h2v, -1);
                // Eigen::Tensor<cx64, 3> PR        = envR.get_expansion_term(mpsR, mpoR, res.alpha_h2v, -1);
                // Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                Eigen::Tensor<cx64, 3> MR_PR_tmp = MR_PR.concatenate(PR, 1);
                // ML_PL                            = std::move(ML_PL_tmp);
                MR_PR = std::move(MR_PR_tmp);
                PRdim1 += PR.dimension(1);
                res.env_var = envR;
            }
            Eigen::Tensor<cx64, 3> P0    = tenx::TensorConstant<cx64>(cx64(0.0, 0.0), mpsL.spin_dim(), mpsL.get_chiL(), PRdim1);
            Eigen::Tensor<cx64, 3> ML_P0 = mpsL.get_M().concatenate(P0, 2); // mpsL is going to be optimized, zero-padded with P0
            merge_expansion_term_PR(state, mpsL, ML_P0, mpsR, MR_PR, svd_cfg);
            // merge_expansion_term_PR(state, mpsL, ML_PL, mpsR, MR_PR, svd_cfg);
            state.clear_measurements();
            state.clear_cache();

            tools::log->debug("Environment expansion forward pos {} | {} | αₑ:{:.2e} αᵥ:{:.2e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {}", pos_expanded,
                              flag2str(envExpandMode), res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimL_old[2],
                              MR_PR.dimension(1), mpsL.get_chiR());
        }
        if(state.get_direction() < 0) {
            // The expanded bond sits between mpsL and mpsR. When direction is right-to-left:
            //      * mpsL is A(i-1) and belongs in envL later in the optimization step
            //      * mpsR is A(i) (or AC(i)) and is the active site
            auto                  &mpoL   = model.get_mpo(safe_cast<size_t>(res.posL));
            Eigen::Tensor<cx64, 3> ML_PL  = mpsL.get_M() * mpsL.get_M().constant(cx64(res.alpha_mps, 0.0));
            long                   PLdim2 = 0;
            if(res.alpha_h1v != 0) {
                auto                  &envL    = edges.get_env_eneL(safe_cast<size_t>(res.posL));
                long                   chi_max = 2 * bond_lim / (1 + mpoL.MPO().dimension(1));
                Eigen::Tensor<cx64, 3> PL      = envL.get_expansion_term(mpsL, mpoL, res.alpha_h1v, chi_max);
                // Eigen::Tensor<cx64, 3> PL        = envL.get_expansion_term(mpsL, mpoL, res.alpha_h1v, -1);
                Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                ML_PL                            = std::move(ML_PL_tmp);
                PLdim2 += PL.dimension(2);
                res.env_ene = envL;
            }
            if(res.alpha_h2v != 0) {
                auto                  &envL    = edges.get_env_varL(safe_cast<size_t>(res.posL));
                long                   chi_max = 2 * bond_lim / (1 + mpoL.MPO2().dimension(1));
                Eigen::Tensor<cx64, 3> PL      = envL.get_expansion_term(mpsL, mpoL, res.alpha_h2v, chi_max);
                // Eigen::Tensor<cx64, 3> PL        = envL.get_expansion_term(mpsL, mpoL, res.alpha_h2v, -1);
                Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                ML_PL                            = std::move(ML_PL_tmp);
                PLdim2 += PL.dimension(2);
                res.env_var = envL;
            }
            if(PLdim2 == 0) { tools::log->error("expand_environment_forward_1site: PRdim1 == 0: this is likely an error: mpsL was needlessly modified! "); }

            Eigen::Tensor<cx64, 3> P0    = tenx::TensorConstant<cx64>(cx64(0.0, 0.0), mpsR.spin_dim(), PLdim2, mpsR.get_chiR());
            Eigen::Tensor<cx64, 3> MR_P0 = mpsR.get_M_bare().concatenate(P0, 1);
            merge_expansion_term_PL(state, mpsL, ML_PL, mpsR, MR_P0, svd_cfg);

            state.clear_cache();
            state.clear_measurements();

            tools::log->debug("Environment expansion forward pos {} | {} | αₑ:{:.2e} αᵥ:{:.2e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {}", pos_expanded,
                              flag2str(envExpandMode), res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimR_old[1],
                              ML_PL.dimension(2), mpsR.get_chiL());
        }
    } else if(has_flag(envExpandMode, EnvExpandMode::REAR)) {
        if(state.get_direction() > 0 and res.posL > 0) {
            // The expanded bond sits between mpsL and mpsR. When direction is left-to-right:
            //      * mpsL is "AC(i)" and will get moved into envL later
            //      * mpsR is "B(i+1)" and will become the active site after moving center position
            auto                  &mpoL   = model.get_mpo(safe_cast<size_t>(res.posL));
            Eigen::Tensor<cx64, 3> ML_PL  = mpsL.get_M() * mpsL.get_M().constant(cx64(res.alpha_mps, 0.0));
            long                   PLdim2 = 0;
            if(res.alpha_h1v != 0) {
                auto                  &envL      = edges.get_env_eneL(safe_cast<size_t>(res.posL));
                long                   chi_max   = 2 * bond_lim / (1 + mpoL.MPO().dimension(1));
                Eigen::Tensor<cx64, 3> PL        = envL.get_expansion_term(mpsL, mpoL, res.alpha_h1v, chi_max);
                Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                ML_PL                            = std::move(ML_PL_tmp);
                PLdim2 += PL.dimension(2);
                res.env_ene = envL; // Saves a reference to the expanded environment
            }
            if(res.alpha_h2v != 0) {
                auto                  &envL      = edges.get_env_varL(safe_cast<size_t>(res.posL));
                long                   chi_max   = 2 * bond_lim / (1 + mpoL.MPO2().dimension(1));
                Eigen::Tensor<cx64, 3> PL        = envL.get_expansion_term(mpsL, mpoL, res.alpha_h2v, chi_max);
                Eigen::Tensor<cx64, 3> ML_PL_tmp = ML_PL.concatenate(PL, 2);
                ML_PL                            = std::move(ML_PL_tmp);
                PLdim2 += PL.dimension(2);
                res.env_var = envL; // Saves a reference to the expanded environment
            }
            if(PLdim2 == 0) { tools::log->error("expand_environment_backward: PLdim2 == 0: this is likely an error"); }
            Eigen::Tensor<cx64, 3> P0    = tenx::TensorConstant<cx64>(cx64(0.0, 0.0), mpsR.spin_dim(), PLdim2, mpsR.get_chiR());
            Eigen::Tensor<cx64, 3> MR_P0 = mpsR.get_M().concatenate(P0, 1); // mpsR is going into the environment, padded with zeros
            merge_expansion_term_PL(state, mpsL, ML_PL, mpsR, MR_P0, svd_cfg);
            tools::log->debug("Environment expansion backward pos {} | {} | αₑ:{:.2e} αᵥ:{:.2e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {}", pos_expanded,
                              flag2str(envExpandMode), res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimL_old[2],
                              ML_PL.dimension(2), mpsL.get_chiR());
        }
        if(state.get_direction() < 0) {
            // The expanded bond sits between mpsL and mpsR. In 1-site DMRG, when the direction is left-to-right:
            //      * mpsL is "A(i-1)" and will become the active site after moving center position
            //      * mpsR is "AC(i)" and is the active site that will get moved into envR later
            // auto                  &mpoL   = model.get_mpo(posL);
            auto                  &mpoR   = model.get_mpo(safe_cast<size_t>(res.posR));
            Eigen::Tensor<cx64, 3> MR_PR  = mpsR.get_M() * mpsR.get_M().constant(cx64(res.alpha_mps, 0.0)); // [ AC  P ]^T  including LC
            long                   PRdim1 = 0;
            if(res.alpha_h1v != 0) {
                // auto                  &envL       = edges.get_env_eneL(posL);
                auto &envR    = edges.get_env_eneR(safe_cast<size_t>(res.posR));
                long  chi_max = 2 * bond_lim / (1 + mpoR.MPO().dimension(0));
                // Eigen::Tensor<cx64, 3> PL         = envL.get_expansion_term(mpsL, mpoL, 0.0, chi_max);
                Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h1v, chi_max);
                // Eigen::Tensor<cx64, 3> ML_PL_temp = ML_PL.concatenate(PL, 2);
                Eigen::Tensor<cx64, 3> MR_PR_temp = MR_PR.concatenate(PR, 1);
                // ML_PL                             = std::move(ML_PL_temp);
                MR_PR = std::move(MR_PR_temp);
                PRdim1 += PR.dimension(1);
                res.env_ene = envR;
            }
            if(res.alpha_h2v != 0) {
                // auto                  &envL       = edges.get_env_varL(posL);
                auto &envR    = edges.get_env_varR(safe_cast<size_t>(safe_cast<size_t>(res.posR)));
                long  chi_max = 2 * bond_lim / (1 + mpoR.MPO2().dimension(0));
                // Eigen::Tensor<cx64, 3> PL         = envL.get_expansion_term(mpsL, mpoL, 0.0, chi_max);
                Eigen::Tensor<cx64, 3> PR = envR.get_expansion_term(mpsR, mpoR, res.alpha_h2v, chi_max);
                // Eigen::Tensor<cx64, 3> ML_PL_temp = ML_PL.concatenate(PL, 2);
                Eigen::Tensor<cx64, 3> MR_PR_temp = MR_PR.concatenate(PR, 1);
                // ML_PL                             = std::move(ML_PL_temp);
                MR_PR = std::move(MR_PR_temp);
                PRdim1 += PR.dimension(1);
                res.env_var = envR;
            }
            if(PRdim1 == 0) { tools::log->error("expand_environment (backward): PRdim1 == 0: this is likely an error"); }
            Eigen::Tensor<cx64, 3> P0    = tenx::TensorConstant<cx64>(cx64(0.0, 0.0), mpsL.spin_dim(), mpsL.get_chiL(), PRdim1);
            Eigen::Tensor<cx64, 3> ML_P0 = mpsL.get_M().concatenate(P0, 2); // [ A   0 ]
            merge_expansion_term_PR(state, mpsL, ML_P0, mpsR, MR_PR, svd_cfg);

            tools::log->debug("Environment expansion backward pos {} | {} | αₑ:{:.2e} αᵥ:{:.3e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {}", pos_expanded,
                              flag2str(envExpandMode), res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimL_old[2],
                              MR_PR.dimension(1), mpsL.get_chiR());
        }
    }

    if(dimL_old[1] != mpsL.get_chiL()) throw except::runtime_error("mpsL changed chiL during environment expansion: {} -> {}", dimL_old, mpsL.dimensions());
    if(dimR_old[2] != mpsR.get_chiR()) throw except::runtime_error("mpsR changed chiR during environment expansion: {} -> {}", dimR_old, mpsR.dimensions());
    if constexpr(settings::debug_expansion) mpsL.assert_normalized();
    if constexpr(settings::debug_expansion) mpsR.assert_normalized();
    state.clear_cache();
    state.clear_measurements();
    env::rebuild_edges(state, model, edges);
    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}

EnvExpansionResult tools::finite::env::expand_environment_nsite(StateFinite &state, const ModelFinite &model, EdgesFinite &edges, EnvExpandMode envExpandMode,
                                                                OptAlgo algo, OptRitz ritz, size_t blocksize, svd::config svd_cfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_environment_forward_nsite: All lengths not equal: state {} | model {} | edges {}", state.get_length(),
                                    model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_environment_forward_nsite: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for environment expansion");

    // Determine which bond to expand
    // We need at least 1 extra site, apart from the active site(s), to expand the environment for the upcoming optimization.
    blocksize     = std::max(blocksize, state.active_sites.size() + 1);
    size_t posL   = state.active_sites.front();
    size_t posR   = state.active_sites.back();
    size_t length = state.get_length<size_t>();

    if(!has_flag(envExpandMode, EnvExpandMode::FORE) and !has_flag(envExpandMode, EnvExpandMode::REAR)) envExpandMode |= EnvExpandMode::FORE;

    // Grow the posL and posR boundary until they cover the block size
    long poslL      = safe_cast<long>(posL);
    long poslR      = safe_cast<long>(posR);
    long lengthl    = safe_cast<long>(length);
    long blocksizel = safe_cast<long>(blocksize);
    if(has_flag(envExpandMode, EnvExpandMode::REAR | EnvExpandMode::FORE)) {
        // Center the expansion block around the active sites
        if(state.get_direction() > 0) {
            long poslC      = (posL + posR + 1l) / 2l;
            long blockhalfl = (blocksizel) / 2l;
            poslR           = std::clamp<long>(poslC + blockhalfl, poslR, lengthl - 1l);
            poslL           = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslL);
        }
        if(state.get_direction() < 0) {
            long poslC      = (posL + posR) / 2l;
            long blockhalfl = (blocksizel) / 2l;
            poslL           = std::clamp<long>(poslC - blockhalfl, 0l, poslL);
            poslR           = std::clamp<long>(poslL + (blocksizel - 1l), poslR, lengthl - 1l);
        }
    } else if(has_flag(envExpandMode, EnvExpandMode::FORE)) {
        if(state.get_direction() > 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
        if(state.get_direction() < 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
    } else if(has_flag(envExpandMode, EnvExpandMode::REAR)) {
        if(state.get_direction() > 0) poslL = std::clamp<long>(poslR - (blocksizel - 1l), 0l, poslR);
        if(state.get_direction() < 0) poslR = std::clamp<long>(poslL + (blocksizel - 1l), poslL, lengthl - 1l);
    }

    posL = safe_cast<size_t>(poslL);
    posR = safe_cast<size_t>(poslR);
    if(posR - posL + 1 > blocksize) throw except::logic_error("error in block size selection | posL {} to posR {} != blocksize {}", posL, posR, blocksize);

    // while(posR - posL + 1 < blocksize) {
    //     size_t posR_old = posR;
    //     size_t posL_old = posL;
    //     if(state.get_direction() > 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //     if(state.get_direction() < 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //
    //     if(state.get_direction() > 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //     if(state.get_direction() < 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //     if(posR_old == posR and posL_old == posL) break; // No change
    // }

    // while(posR - posL + 1 < blocksize) {
    //     size_t posR_old = posR;
    //     size_t posL_old = posL;
    //     if(state.get_direction() > 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //     if(state.get_direction() < 0) {
    //         if(has_flag(envExpandMode, EnvExpandMode::REAR) and posR - posL + 1 < blocksize and posR + 1 < length) { posR++; }
    //         if(has_flag(envExpandMode, EnvExpandMode::FORE) and posR - posL + 1 < blocksize and posL > 0) { posL--; }
    //     }
    //     if(posR_old == posR and posL_old == posL) break; // No change
    // }

    // if(state.get_direction() > 0) posR = std::clamp(posL + (blocksize - 1), posR, state.get_length<size_t>() - 1);
    // if(state.get_direction() < 0) posL = std::clamp(posR - std::min((blocksize - 1), posR), 0ul, posL);

    auto pos_active_and_expanded = num::range<size_t>(posL, posR + 1);

    if(pos_active_and_expanded.size() < 2ul) {
        // tools::log->warn("Expansion: need at least two sites, but pos_active_and_expanded = {}", pos_active_and_expanded);
        return EnvExpansionResult();
    }

    // Define the left and right mps that will get modified
    auto        res  = get_optimally_mixed_block(pos_active_and_expanded, state, model, edges, algo, ritz, settings::strategy::dmrg_env_expand_iter);
    const auto &mpsL = state.get_mps_site(res.posL);
    const auto &mpsR = state.get_mps_site(res.posR);
    res.ene_old      = tools::finite::measure::energy(state, model, edges);
    res.var_old      = tools::finite::measure::energy_variance(state, model, edges);

    if(res.alpha_h1v == 0 and res.alpha_h2v == 0) {
        tools::log->debug("Expansion canceled: {}({}) - {}({}) | αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(),
                          mpsR.get_position(), res.alpha_h1v, res.alpha_h2v);
        return res;
    }

    tools::log->debug("Expanding {}({}) - {}({}) | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_label(), mpsL.get_position(), mpsR.get_label(), mpsR.get_position(),
                      res.alpha_mps, res.alpha_h1v, res.alpha_h2v);

    mps::merge_multisite_mps(state, res.mixed_blk, pos_active_and_expanded, state.get_position<long>(), MergeEvent::EXP, svd_cfg);

    res.dims_new = state.get_mps_dims(pos_active_and_expanded);
    res.bond_new = state.get_bond_dims(pos_active_and_expanded);

    tools::log->debug("Environment expansion pos {} | {} {} | αₑ:{:.2e} αᵥ:{:.2e} | svd_ε {:.2e} | χlim {} | χ {} -> {}", pos_active_and_expanded,
                      enum2sv(algo), enum2sv(ritz), res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), res.bond_old,
                      res.bond_new);
    state.clear_cache();
    state.clear_measurements();
    for(const auto &mps : state.get_mps(pos_active_and_expanded)) mps.get().assert_normalized();
    env::rebuild_edges(state, model, edges);

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    if(num::prod(res.dimL_new) == 0) {
        tools::log->info("mpsL({}): {}", mpsL.get_position(), tenx::VectorMap(mpsL.get_M_bare()));
        tools::log->info("mpsR({}): {}", mpsR.get_position(), tenx::VectorMap(mpsR.get_M_bare()));
        tools::log->info("res     : {}", tenx::VectorMap(res.mixed_blk));
        throw except::runtime_error("New dimL has a zero: {}", res.dimL_new);
    }
    if(num::prod(res.dimR_new) == 0) {
        tools::log->info("mpsL({}): {}", mpsL.get_position(), tenx::VectorMap(mpsL.get_M_bare()));
        tools::log->info("mpsR({}): {}", mpsR.get_position(), tenx::VectorMap(mpsR.get_M_bare()));
        throw except::runtime_error("New dimR has a zero: {}", res.dimR_new);
    }

    res.ene_new = tools::finite::measure::energy(state, model, edges);
    res.var_new = tools::finite::measure::energy_variance(state, model, edges);
    if(std::isnan(res.ene_new)) throw except::runtime_error("res.ene_new is nan");
    if(std::isnan(res.var_new)) throw except::runtime_error("res.var_new is nan");
    res.ok = true;
    return res;
}

void tools::finite::env::assert_edges_ene(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert energy edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.get_position<long>();

    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);

    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &ene = edges.get_env_eneL(pos);
        if(pos == 0 and not ene.has_block()) throw except::runtime_error("ene L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_next = edges.get_env_eneL(pos + 1);
        ene_next.assert_unique_id(ene, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_ene: pos {} | dir {} | "
                          "asserting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);

    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &ene = edges.get_env_eneR(pos);
        if(pos == state.get_length() - 1 and not ene.has_block()) throw except::runtime_error("ene R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &ene_prev = edges.get_env_eneR(pos - 1);
        ene_prev.assert_unique_id(ene, mps, mpo);
    }
}

void tools::finite::env::assert_edges_var(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("assert_edges_var: fLBIT algorithm should never assert variance edges!");
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites we shouldn't be asserting edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge assertions with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("assert_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();

    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &var = edges.get_env_varL(pos);
        if(pos == 0 and not var.has_block()) throw except::runtime_error("var L at pos {} does not have a block", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_next = edges.get_env_varL(pos + 1);
        var_next.assert_unique_id(var, mps, mpo);
    }
    if constexpr(settings::debug_edges)
        tools::log->trace("assert_edges_var: pos {} | dir {} | "
                          "asserting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &var = edges.get_env_varR(pos);
        if(pos == state.get_length() - 1 and not var.has_block()) throw except::runtime_error("var R at pos {} does not have a block", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &mps      = state.get_mps_site(pos);
        auto &mpo      = model.get_mpo(pos);
        auto &var_prev = edges.get_env_varR(pos - 1);
        var_prev.assert_unique_id(var, mps, mpo);
    }
}

void tools::finite::env::assert_edges(const StateFinite &state, const ModelFinite &model, const EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
}

void tools::finite::env::rebuild_edges_ene(const StateFinite &state, const ModelFinite &model, EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_ene: fLBIT algorithm should never rebuild energy edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(), edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto   t_reb   = tid::tic_scope("rebuild_edges_ene", tid::higher);
    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites then we can build up until current position
    /*
     * LOG:
     * - 2021-10-14:
     *      Just had a terribly annoying bug:
     *      Moving the center position clears active_sites, which caused problems when turning back from the right edge.
     *          1)  active_sites [A(L-1), AC(L)] are updated, left edge exist for A(L-1), right edge exists for AC(L)
     *          2)  move dir -1, clear active sites
     *          3)  assert_edges checks up to AC(L-1), but this site has a stale right edge.
     *      Therefore, one would have to rebuild edges between steps 2) and 3) to solve this issue
     *
     *      One solution would be to always rebuild edges up to the current position from both sides, but that would be
     *      wasteful. Instead, we could just accept that some edges are stale after moving the center-point,
     *      as long as we rebuild those when sites get activated again.
     *
     * - 2024-07-23
     *      Just found a way to calculate the optimal mixing factor for subspace expansion.
     *      In forward expansion we need H_eff including one site beyond active_sites.
     *      Therefore, we need to build more environments than we have needed previously.
     *      Examples:
     *          - Forward, direction == 1, active_sites = [5,6,7,8].
     *            Then we expand bond [8,9], and so we need envL[8] and envR[9].
     *          - Forward, direction == -1, active_sites = [5,6,7,8].
     *            Then we expand bond [4,5], and so we need envL[4] and envR[5].
     *      These environments weren't built before.
     *      Therefore, we must now rebuild
     *          - envL 0 to active_sites.back()
     *          - envR L to active_sites.front()
     */

    // If there are no active sites we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_ene: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this rebuild after\n"
                                    " activating sites.");

    long current_position = state.get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    std::vector<size_t> env_pos_log;
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &env_here = edges.get_env_eneL(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == 0) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_ene: No eneL block detected at pos {}", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &env_rght = edges.get_env_eneL(pos + 1);
        auto  id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneL edges: {}", env_pos_log);

    env_pos_log.clear();
    if constexpr(settings::debug_edges)
        tools::log->trace("rebuild_edges_ene: pos {} | dir {} | "
                          "inspecting edges eneR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &env_here = edges.get_env_eneR(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == state.get_length() - 1) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_ene: No eneR block detected at pos {}", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &env_left = edges.get_env_eneR(pos - 1);
        auto  id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_ene: rebuilt eneR edges: {}", env_pos_log);
    if(not edges.get_env_eneL(posL_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneL has undefined block");
    if(not edges.get_env_eneR(posR_active).has_block()) throw except::logic_error("rebuild_edges_ene: active env eneR has undefined block");
}

void tools::finite::env::rebuild_edges_var(const StateFinite &state, const ModelFinite &model, EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) throw except::logic_error("rebuild_edges_var: fLBIT algorithm should never rebuild variance edges!");
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("rebuild_edges_var: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("rebuild_edges_var: All active sites are not equal: state {} | model {} | edges {}", state.active_sites, model.active_sites,
                                    edges.active_sites);
    auto t_reb = tid::tic_scope("rebuild_edges_var", tid::level::higher);

    size_t min_pos = 0;
    size_t max_pos = state.get_length() - 1;

    // If there are no active sites we shouldn't be rebuilding edges.
    // For instance, the active sites are cleared after a move of center site.
    // We could always keep all edges refreshed but that would be wasteful, since the next iteration
    // may activate other sites and not end up needing those edges.
    // Instead, we force the hand of the algorithm, to only allow edge rebuilds with active sites defined.
    // Ideally, then, this should be done directly after activating new sites in a new iteration.
    if(edges.active_sites.empty())
        throw except::runtime_error("rebuild_edges_var: no active sites.\n"
                                    "Hint:\n"
                                    " One could in principle keep edges refreshed always, but\n"
                                    " that would imply rebuilding many edges that end up not\n"
                                    " being used. Make sure to only run this assertion after\n"
                                    " activating sites.");

    long current_position = state.get_position<long>();
    // size_t posL_active      = edges.active_sites.front();
    // size_t posR_active      = edges.active_sites.back();

    // These back and front positions will seem reversed: we need extra edges for optimal subspace expansion: see the Log from 2024-07-23
    size_t posL_active = edges.active_sites.back();
    size_t posR_active = edges.active_sites.front();
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varL from [{} to {}]",
                          current_position, state.get_direction(), min_pos, posL_active);
    }

    std::vector<size_t> env_pos_log;
    for(size_t pos = min_pos; pos <= posL_active; pos++) {
        auto &env_here = edges.get_env_varL(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == 0) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_var: No varL block detected at pos {}", pos);
        if(pos >= std::min(posL_active, state.get_length() - 1)) continue;
        auto &env_rght = edges.get_env_varL(pos + 1);
        auto  id_rght  = env_rght.get_unique_id();
        env_rght.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_rght != env_rght.get_unique_id()) env_pos_log.emplace_back(env_rght.get_position());
    }

    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varL edges: {}", env_pos_log);
    env_pos_log.clear();
    if constexpr(settings::debug_edges) {
        tools::log->trace("rebuild_edges_var: pos {} | dir {} | "
                          "inspecting edges varR from [{} to {}]",
                          current_position, state.get_direction(), posR_active, max_pos);
    }

    for(size_t pos = max_pos; pos >= posR_active and pos < state.get_length(); --pos) {
        auto &env_here = edges.get_env_varR(pos);
        auto  id_here  = env_here.get_unique_id();
        if(pos == state.get_length() - 1) env_here.set_edge_dims(state.get_mps_site(pos), model.get_mpo(pos));
        if(not env_here.has_block()) throw except::runtime_error("rebuild_edges_var: No varR block detected at pos {}", pos);
        if(pos <= std::max(posR_active, 0ul)) continue;
        auto &env_left = edges.get_env_varR(pos - 1);
        auto  id_left  = env_left.get_unique_id();
        env_left.refresh(env_here, state.get_mps_site(pos), model.get_mpo(pos));
        if(id_here != env_here.get_unique_id()) env_pos_log.emplace_back(env_here.get_position());
        if(id_left != env_left.get_unique_id()) env_pos_log.emplace_back(env_left.get_position());
    }
    std::reverse(env_pos_log.begin(), env_pos_log.end());
    if(not env_pos_log.empty()) tools::log->trace("rebuild_edges_var: rebuilt varR edges: {}", env_pos_log);
    if(not edges.get_env_varL(posL_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varL has undefined block");
    if(not edges.get_env_varR(posR_active).has_block()) throw except::logic_error("rebuild_edges_var: active env varR has undefined block");
}

void tools::finite::env::rebuild_edges(const StateFinite &state, const ModelFinite &model, EdgesFinite &edges) {
    if(state.get_algorithm() == AlgorithmType::fLBIT) return;
    rebuild_edges_ene(state, model, edges);
    rebuild_edges_var(state, model, edges);
}
