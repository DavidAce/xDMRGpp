#pragma once
#include "math/float_eigen.h"
//
#include "../common.h"
#include "../standard_basis_change.h"
#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/tensor/to_string.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/TensorsFinite.h"

namespace settings {
#if defined(NDEBUG)
    inline constexpr bool debug_standard_basis_change = false;
#else
    inline constexpr bool debug_standard_basis_change = false;
#endif
}

using namespace tools::finite::opt::precond::standard;
using namespace tools::finite::opt::precond::common;

template<typename Scalar> env_pair<const EnvEne<Scalar> &> tools::finite::opt::precond::standard::BasisChange<Scalar>::get_enve_pair() const {
    return {bc_enveL, bc_enveR};
}

template<typename Scalar> env_pair<const EnvVar<Scalar> &> tools::finite::opt::precond::standard::BasisChange<Scalar>::get_envv_pair() const {
    return {bc_envvL, bc_envvR};
}

template<typename Scalar>
tools::finite::opt::precond::standard::BasisChange<Scalar>::BasisChange(const opt_mps<Scalar> &initial, const TensorsFinite<Scalar> &tensors,
                                                                        BasisChangeScale bcs, RealScalar scale_, RealScalar alpha_)
    : scale(scale_), alpha(alpha_) {
    if(scale <= 0) throw except::runtime_error("Scale must be positive");
    if(!(alpha > RealScalar{0} && alpha < RealScalar{1})) throw except::runtime_error("Expected 0 < alpha < 1: got {:.3e}", fp(alpha));
    std::vector<size_t>  sites = initial.get_sites();
    constexpr RealScalar eps   = std::numeric_limits<RealScalar>::epsilon();

    auto get_aggregate_envs = [](const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR,
                                 const Eigen::Tensor<Scalar, 4> &mpo) -> std::pair<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 2>> {
        // Trace physical indices off the mpo
        Eigen::Tensor<Scalar, 2> mpo_traced = mpo.trace(std::array<Eigen::Index, 2>{2, 3});
        // Trace bond indices off the environments
        Eigen::Tensor<Scalar, 1> envL_traced = envL.trace(std::array<Eigen::Index, 2>{0, 1});
        Eigen::Tensor<Scalar, 1> envR_traced = envR.trace(std::array<Eigen::Index, 2>{0, 1});

        // Contract the traced mpo with the traced environments to get the weights
        Eigen::Tensor<Scalar, 1> wL = envR_traced.contract(mpo_traced, tenx::idx({0}, {1}));
        Eigen::Tensor<Scalar, 1> wR = envL_traced.contract(mpo_traced, tenx::idx({0}, {0}));

        // Form the aggregate environments
        Eigen::Tensor<Scalar, 2> envL_agg = envL.contract(wL, tenx::idx({2}, {0}));
        Eigen::Tensor<Scalar, 2> envR_agg = envR.contract(wR, tenx::idx({2}, {0}));
        if constexpr(settings::debug_standard_basis_change) {
            Eigen::Tensor<RealScalar, 0> min_wL = wL.real().minimum(); // or iterate
            Eigen::Tensor<RealScalar, 0> min_wR = wR.real().minimum();
            tools::log->info("min wL {:.3e}, min wR {:.3e}", fp(min_wL(0)), fp(min_wR(0)));

            tools::log->info("envL_agg dims: {}", envL_agg.dimensions());
            tools::log->info("envR_agg dims: {}", envR_agg.dimensions());

            auto is_hermitian = [&](Eigen::Tensor<Scalar, 2> &A) {
                auto       Am  = MapMatType(A.data(), A.dimension(0), A.dimension(1));
                const auto num = (Am.adjoint() - Am).norm();
                const auto den = std::max(RealScalar{1}, Am.norm());
                return num <= RealScalar{1e-12f} * den; // relative tolerance
            };
            if(not is_hermitian(envL_agg)) throw except::runtime_error("envL_agg is not hermitian: \n{}\n", linalg::tensor::to_string(envL_agg, 8));
            if(not is_hermitian(envR_agg)) throw except::runtime_error("envR_agg is not hermitian: \n{}\n", linalg::tensor::to_string(envR_agg, 8));
        }

        return {envL_agg, envR_agg};
    };

    Eigen::Tensor<Scalar, 4> mpo1 = tensors.get_model().template get_multisite_mpo<Scalar>(sites);         // Multisite mpo for H1
    Eigen::Tensor<Scalar, 4> mpo2 = tensors.get_model().template get_multisite_mpo_squared<Scalar>(sites); // Multisite mpo for H2
    // Get the environment blocks
    auto env1 = tensors.get_edges().get_multisite_env_ene(sites); // The H1 environment for these sites
    auto env2 = tensors.get_edges().get_multisite_env_var(sites); // The H2 environment for these sites

    const Eigen::Tensor<Scalar, 3> &env2L = env2.L.get_block();
    const Eigen::Tensor<Scalar, 3> &env2R = env2.R.get_block();

    auto [env2L_agg, env2R_agg] = get_aggregate_envs(env2L, env2R, mpo2);

    auto get_standard_transforms = [&](const Eigen::Tensor<Scalar, 2> &env2_agg) -> std::tuple<MatrixType, MatrixType, RealScalar> {
        auto agg2 = MapConstMatType(env2_agg.data(), env2_agg.dimension(0), env2_agg.dimension(1));
        auto es   = Eigen::SelfAdjointEigenSolver<MatrixType>(agg2, Eigen::ComputeEigenvectors);
        if(es.info() != Eigen::Success) throw except::runtime_error("Standard eigen solve failed: info={}", int(es.info()));

        auto U = es.eigenvectors();
        auto Y = es.eigenvalues();
        tools::log->info("Y eigvals: {::.5e}", fv(Y));
        if constexpr(settings::debug_standard_basis_change) { tools::log->info("Y min: {:.5e} max {:.5e}", fp(Y.minCoeff()), fp(Y.maxCoeff())); }

        const RealScalar   factor          = RealScalar{1e-3f};
        const Eigen::Index n               = Y.size();
        const RealScalar   ymax            = Y.maxCoeff();
        const RealScalar   min_gap         = n >= 2 ? (Y.tail(n - 1) - Y.head(n - 1)).minCoeff() : ymax;
        const RealScalar   min_gap_guarded = std::max(min_gap, eps * ymax);
        const RealScalar   tau             = std::max(eps * ymax, factor * min_gap_guarded);

        // Clip spectrum at tau and temper with ±alpha/2
        VectorReal tauY = Y.cwiseAbs().cwiseMax(tau);

        VectorReal invPowY = tauY.array().pow(-alpha / 2).matrix();
        VectorReal absPowY = tauY.array().pow(+alpha / 2).matrix();

        MatrixType T = U * invPowY.asDiagonal() * U.adjoint(); // == U |Y|^{-α/2} U^H
        MatrixType S = U * absPowY.asDiagonal() * U.adjoint(); // == U |Y|^{+α/2} U^H

        // MatrixType W = U * invPowY.asDiagonal() * U.adjoint();
        // // Unitary polar factor: W = Q * H, to make T unitary
        // auto svd = Eigen::BDCSVD<MatrixType>();
        // svd.setSwitchSize(256);
        // svd.compute(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        //
        // MatrixType T = svd.matrixU() * svd.matrixV().adjoint();
        // MatrixType S = T.adjoint();

        auto get_Y_normalization = [&]() -> RealScalar {
            switch(bcs) {
                case BasisChangeScale::NONE: return RealScalar{1};
                case BasisChangeScale::MIN: return tauY.minCoeff();
                case BasisChangeScale::AVG: return tauY.mean();
                case BasisChangeScale::MAX: return tauY.maxCoeff();
                case BasisChangeScale::SQRTMIN: return tauY.cwiseSqrt().minCoeff();
                case BasisChangeScale::SQRTAVG: return tauY.cwiseSqrt().mean();
                case BasisChangeScale::SCALE: return scale;
                default: throw except::runtime_error("Unknown BasisChangeScale");
            }
        };
        auto mY = get_Y_normalization();

        if constexpr(settings::debug_standard_basis_change) {
            // Helpers
            auto I = [&](Eigen::Index m) { return MatrixType::Identity(m, m); };

            auto rel_err = [&](const MatrixType &X, const MatrixType &G, RealScalar ref_norm) -> RealScalar {
                RealScalar num = (X - G).norm();
                RealScalar den = std::max<RealScalar>(RealScalar{1}, ref_norm);
                return num / den;
            };

            auto herm_resid = [&](const MatrixType &X) -> RealScalar {
                RealScalar num = (X.adjoint() - X).norm();
                RealScalar den = std::max<RealScalar>(RealScalar{1}, X.norm());
                return num / den;
            };

            auto sym = [&](const MatrixReal &R) -> MatrixReal { return RealScalar{0.5} * (R + R.adjoint()); };

            auto eig_range = [&](const MatrixType &X, std::string_view lbl) {
                Eigen::SelfAdjointEigenSolver<MatrixReal> es(sym(X.real()));
                if(es.info() == Eigen::Success) {
                    auto ev = es.eigenvalues();
                    tools::log->info("  eig({}): min={:.6e}, max={:.6e}", lbl, fp(ev.minCoeff()), fp(ev.maxCoeff()));
                } else {
                    tools::log->warn("  eig({}): decomposition failed", lbl);
                }
            };
            auto check_inverse = [&](const MatrixType &TT, const MatrixType &SS) {
                const auto Id = MatrixType::Identity(TT.rows(), TT.cols());

                RealScalar err_ST = (SS * TT - Id).norm() / std::max<RealScalar>(RealScalar{1}, Id.norm());
                RealScalar err_TS = (TT * SS - Id).norm() / std::max<RealScalar>(RealScalar{1}, Id.norm());

                tools::log->info("S*T inverse error {:.2e}", fp(err_ST));
                tools::log->info("T*S inverse error {:.2e}", fp(err_TS));
            };
            auto check_projector = [&](const MatrixType &TT, const MatrixType &SS) {
                MatrixType P    = SS * TT; // should be ~projector
                RealScalar symm = herm_resid(P);
                RealScalar idem = (P * P - P).norm() / std::max<RealScalar>(RealScalar{1}, P.norm());
                tools::log->info("P symmetry {:.2e}, idempotence {:.2e}", fp(symm), fp(idem));
                RealScalar inv_rt = (SS * (TT * SS) - SS).norm() / std::max<RealScalar>(RealScalar{1}, SS.norm());
                tools::log->info("round-trip inverse: {:.2e}", fp(inv_rt));
            };

            auto check_congruence = [&](const MatrixType &X, const MatrixType &TT, const MatrixType &G, std::string_view lbl) {
                MatrixType W   = TT.adjoint() * X * TT;
                RealScalar rel = rel_err(W, G, X.norm());
                RealScalar hrm = herm_resid(W);
                tools::log->info("{}: ||T^H X T - G||/max(1,||X||) = {:.3e}", lbl, fp(rel));
                tools::log->info("{}: Herm residual (rel) = {:.3e}", lbl, fp(hrm));
                eig_range(W, std::string(lbl) + "_sym");
            };

            // Targets for magnitude-only tempered scheme:
            //   T = U |Y|^{-alpha/2} U^H  ⇒  T^H A T = U [sgn(Y)|Y|^{1-alpha}] U^H,
            //                                T^H B T = U [|Y|^{-alpha}]        U^H.
            VectorReal sgnY    = Y.array().sign().matrix();
            const auto n       = env2_agg.dimension(0);
            MatrixType target1 = U * (sgnY.array() * tauY.array().pow(RealScalar{1} - alpha)).matrix().asDiagonal() * U.adjoint();
            MatrixType target2 = U * tauY.array().pow(RealScalar{1} - alpha).matrix().asDiagonal() * U.adjoint();
            // Logs
            check_congruence(agg2, T, target2, "[agg2]");
            check_inverse(T, S);
        }

        return {T, S, std::sqrt(mY)};
    };

    std::tie(TL, SL, kappaL) = get_standard_transforms(env2L_agg);
    std::tie(TR, SR, kappaR) = get_standard_transforms(env2R_agg);

    auto init_dims = initial.get_tensor().dimensions();
    shape_orig     = {init_dims[0], TL.cols(), TR.cols()};
    shape_tilde    = {init_dims[0], TL.rows(), TR.rows()};

    // Transform the environments
    bc_enveL = env1.L;
    bc_enveR = env1.R;
    bc_envvL = env2.L;
    bc_envvR = env2.R;

    bc_enveL.set_block_raw(transform_env(env1.L.get_block(), TL, kappaL));
    bc_enveR.set_block_raw(transform_env(env1.R.get_block(), TR, kappaR));
    bc_envvL.set_block_raw(transform_env(env2.L.get_block(), TL, kappaL * kappaL));
    bc_envvR.set_block_raw(transform_env(env2.R.get_block(), TR, kappaR * kappaR));
    // auto enveL_norm      = tenx::VectorMap(bc_enveL.get_block()).norm();
    // auto enveR_norm      = tenx::VectorMap(bc_enveR.get_block()).norm();
    // auto envvL_norm      = tenx::VectorMap(bc_envvL.get_block()).norm();
    // auto envvR_norm      = tenx::VectorMap(bc_envvR.get_block()).norm();
    //
    // auto gamma = RealScalar{1} / std::max(eps, envvL_norm * envvR_norm);
    // // Apply the SAME gamma to both operators (all env blocks)
    // bc_enveL.get_block() *= bc_enveL.get_block().constant(gamma);
    // bc_enveR.get_block() *= bc_enveR.get_block().constant(gamma);
    // bc_envvL.get_block() *= bc_envvL.get_block().constant(gamma);
    // bc_envvR.get_block() *= bc_envvR.get_block().constant(gamma);
    // tools::log->info("|enveL| * gamma = {:.3e} * {:.3e} = {:.3e}", fp(enveL_norm), fp(gamma), fp(gamma * enveL_norm));
    // tools::log->info("|enveR| * gamma = {:.3e} * {:.3e} = {:.3e}", fp(enveR_norm), fp(gamma), fp(gamma * enveR_norm));
    // tools::log->info("|envvL| * gamma = {:.3e} * {:.3e} = {:.3e}", fp(envvL_norm), fp(gamma), fp(gamma * envvL_norm));
    // tools::log->info("|envvR| * gamma = {:.3e} * {:.3e} = {:.3e}", fp(envvR_norm), fp(gamma), fp(gamma * envvR_norm));

    // Transform the initial guess
    initial_guess = initial;
    initial_guess.set_tensor(transform_tensor(initial.get_tensor(), SL, SR));
}
//
// template<typename Scalar>
// tools::finite::opt::precond::standard::BasisChange<Scalar>::BasisChange(const opt_mps<Scalar> &initial, const TensorsFinite<Scalar> &tensors,
//                                                                         BasisChangeScale bcs, RealScalar scale_, RealScalar alpha_)
//     : scale(scale_), alpha(alpha_) {
//     if(scale <= 0) throw except::runtime_error("Scale must be positive");
//     std::vector<size_t>  sites = initial.get_sites();
//     constexpr RealScalar eps   = std::numeric_limits<RealScalar>::epsilon();
//
//     // Get the H2 environment blocks
//     const auto  enve = tensors.get_edges().get_multisite_env_ene(sites); // The H1 environment for these sites
//     const auto  envv = tensors.get_edges().get_multisite_env_var(sites); // The H2 environment for these sites
//     const auto &envL = envv.L.get_block();
//     const auto &envR = envv.R.get_block();
//
//     // Trace physical indices off the H2 mpo at the current sites
//     Eigen::Tensor<Scalar, 2> mpo2_traced = tensors.get_model().template get_multisite_mpo_squared<Scalar>(sites).trace(std::array<Eigen::Index, 2>{2, 3});
//     // Trace bond indices off the environments
//     Eigen::Tensor<Scalar, 1> envL_traced = envL.trace(std::array<Eigen::Index, 2>{0, 1});
//     Eigen::Tensor<Scalar, 1> envR_traced = envR.trace(std::array<Eigen::Index, 2>{0, 1});
//
//     // Contract the traced mpo with the traced environments to get the weights
//     Eigen::Tensor<Scalar, 1>     wL     = envR_traced.contract(mpo2_traced, tenx::idx({0}, {1}));
//     Eigen::Tensor<Scalar, 1>     wR     = envL_traced.contract(mpo2_traced, tenx::idx({0}, {0}));
//     Eigen::Tensor<RealScalar, 0> min_wL = wL.real().minimum(); // or iterate
//     Eigen::Tensor<RealScalar, 0> min_wR = wR.real().minimum();
//     tools::log->info("min wL {:.3e}, min wR {:.3e}", fp(min_wL(0)), fp(min_wR(0)));
//
//     // Form the aggregate environments
//     Eigen::Tensor<Scalar, 2> aggL = envL.contract(wL, tenx::idx({2}, {0}));
//     Eigen::Tensor<Scalar, 2> aggR = envR.contract(wR, tenx::idx({2}, {0}));
//
//     tools::log->info("aggL dims: {}", aggL.dimensions());
//     tools::log->info("aggR dims: {}", aggR.dimensions());
//
//     auto is_hermitian = [&](Eigen::Tensor<Scalar, 2> &A) {
//         auto       Am  = MapMatType(A.data(), A.dimension(0), A.dimension(1));
//         const auto num = (Am.adjoint() - Am).norm();
//         const auto den = std::max(RealScalar{1}, Am.norm());
//         return num <= RealScalar{1e-12f} * den; // relative tolerance
//     };
//     if(not is_hermitian(aggL)) throw except::runtime_error("aggL is not hermitian: \n{}\n", linalg::tensor::to_string(aggL, 8));
//     if(not is_hermitian(aggR)) throw except::runtime_error("aggR is not hermitian: \n{}\n", linalg::tensor::to_string(aggR, 8));
//
//     // Diagonalize envL_aggregate and envR_aggregate
//     auto aggL_map = MapMatType(aggL.data(), aggL.dimension(0), aggL.dimension(1));
//     auto aggR_map = MapMatType(aggR.data(), aggR.dimension(0), aggR.dimension(1));
//     auto esL      = Eigen::SelfAdjointEigenSolver<MatrixType>(aggL_map, Eigen::ComputeEigenvectors);
//     auto esR      = Eigen::SelfAdjointEigenSolver<MatrixType>(aggR_map, Eigen::ComputeEigenvectors);
//
//     auto UL = esL.eigenvectors();
//     auto YL = esL.eigenvalues();
//     auto UR = esR.eigenvectors();
//     auto YR = esR.eigenvalues();
//
//     if constexpr(settings::debug_standard_basis_change) {
//         tools::log->info("YL min: {:.5e} max {:.5e}", fp(YL.minCoeff()), fp(YL.maxCoeff()));
//         tools::log->info("YR min: {:.5e} max {:.5e}", fp(YR.minCoeff()), fp(YR.maxCoeff()));
//     }
//
//     auto get_Y_normalization = [&](const VectorReal &Y) -> RealScalar {
//         switch(bcs) {
//             case BasisChangeScale::NONE: return RealScalar{1};
//             case BasisChangeScale::MIN: return Y.minCoeff();
//             case BasisChangeScale::AVG: return Y.mean();
//             case BasisChangeScale::MAX: return Y.maxCoeff();
//             case BasisChangeScale::SQRTMIN: return Y.cwiseSqrt().minCoeff();
//             case BasisChangeScale::SQRTAVG: return Y.cwiseSqrt().mean();
//             case BasisChangeScale::SCALE: return scale;
//             default: throw except::runtime_error("Unknown BasisChangeScale");
//         }
//     };
//     RealScalar mYL    = get_Y_normalization(YL);
//     RealScalar mYR    = get_Y_normalization(YR);
//     RealScalar kappaL = std::sqrt(mYL);
//     RealScalar kappaR = std::sqrt(mYR);
//     // VectorReal YL_n = YL;
//     // VectorReal YR_n = YR;
//
//     VectorReal absEpsYL = (YL.cwiseAbs().array() + eps).matrix();
//     VectorReal absEpsYR = (YR.cwiseAbs().array() + eps).matrix();
//
//     VectorReal invPowYL = absEpsYL.array().pow(-alpha / 2).matrix();
//     VectorReal invPowYR = absEpsYR.array().pow(-alpha / 2).matrix();
//
//     VectorReal absPowYL = absEpsYL.array().pow(+alpha / 2).matrix();
//     VectorReal absPowYR = absEpsYR.array().pow(+alpha / 2).matrix();
//
//     TL = UL * invPowYL.asDiagonal() * UL.adjoint();
//     TR = UR * invPowYR.asDiagonal() * UR.adjoint();
//
//     SL = UL * absPowYL.asDiagonal() * UL.adjoint();
//     SR = UR * absPowYR.asDiagonal() * UR.adjoint();
//
//     auto init_dims = initial.get_tensor().dimensions();
//     shape_orig     = {init_dims[0], TL.cols(), TR.cols()};
//     shape_tilde    = {init_dims[0], TL.rows(), TR.rows()};
//
//     if constexpr(settings::debug_standard_basis_change) {
//         const auto nL = aggL.dimension(0);
//         const auto nR = aggR.dimension(0);
//
//         MapMatType Lm(aggL.data(), nL, nL);
//         MapMatType Rm(aggR.data(), nR, nR);
//
//         // Whitened aggregates: TL^H * agg * TL and TR^H * agg * TR
//         MatrixType WL = TL.adjoint() * Lm * TL;
//         MatrixType WR = TR.adjoint() * Rm * TR;
//
//         // Identity references
//         MatrixType IL = MatrixType::Identity(nL, nL);
//         MatrixType IR = MatrixType::Identity(nR, nR);
//
//         auto rel = [](const MatrixType &A, const MatrixType &B, RealScalar ref_norm) {
//             RealScalar num = (A - B).norm();
//             RealScalar den = std::max<RealScalar>(RealScalar{1}, ref_norm);
//             return num / den;
//         };
//
//         // Relative Frobenius errors (normalize by ||agg||, as discussed)
//         RealScalar relL = rel(WL, IL, Lm.norm());
//         RealScalar relR = rel(WR, IR, Rm.norm());
//
//         // Hermiticity residuals (relative)
//         auto relHerm = [](const MatrixType &A) {
//             RealScalar num = (A.adjoint() - A).norm();
//             RealScalar den = std::max<RealScalar>(RealScalar{1}, A.norm());
//             return num / den;
//         };
//         RealScalar hermWL = relHerm(WL);
//         RealScalar hermWR = relHerm(WR);
//
//         tools::log->info("Whitening check:");
//         tools::log->info("  ||TL^H * aggL * TL - I|| / max(1,||aggL||) = {:.3e}", fp(relL));
//         tools::log->info("  ||TR^H * aggR * TR - I|| / max(1,||aggR||) = {:.3e}", fp(relR));
//         tools::log->info("  Herm( WL ) residual (rel) = {:.3e}", fp(hermWL));
//         tools::log->info("  Herm( WR ) residual (rel) = {:.3e}", fp(hermWR));
//
//         // OPTIONAL: eigenvalue ranges of symmetrized WL/WR
//         // (useful to see they’re ~1 up to round-off)
//         {
//             auto                                      sym = [](const MatrixReal &A) -> MatrixReal { return (A + A.adjoint()) * RealScalar{0.5f}; };
//             Eigen::SelfAdjointEigenSolver<MatrixReal> es_L(sym(WL.real()));
//             Eigen::SelfAdjointEigenSolver<MatrixReal> es_R(sym(WR.real()));
//             if(es_L.info() == Eigen::Success && es_R.info() == Eigen::Success) {
//                 RealScalar minWL = es_L.eigenvalues().minCoeff();
//                 RealScalar maxWL = es_L.eigenvalues().maxCoeff();
//                 RealScalar minWR = es_R.eigenvalues().minCoeff();
//                 RealScalar maxWR = es_R.eigenvalues().maxCoeff();
//                 tools::log->info("  eig(WL_sym): min={:.6e}, max={:.6e}", fp(minWL), fp(maxWL));
//                 tools::log->info("  eig(WR_sym): min={:.6e}, max={:.6e}", fp(minWR), fp(maxWR));
//             } else {
//                 tools::log->warn("  Eigen decomposition of WL/WR (sym) failed");
//             }
//         }
//
//         auto check = [](auto T, auto S, std::string_view tag) {
//             auto       P   = S * T;                                                        // should be ≈ projector on kept subspace
//             RealScalar err = (P - P.adjoint()).norm() / std::max(RealScalar{1}, P.norm()); // symmetry
//             RealScalar inv = (P * P - P).norm() / std::max(RealScalar{1}, P.norm());       // idempotence
//             tools::log->info("P{} symmetry {:.2e}, idempotence {:.2e}", tag, fp(err), fp(inv));
//
//             RealScalar invRoundtrip = (S * (T * S) - S).norm() / std::max(RealScalar{1}, S.norm());
//             tools::log->info("round-trip inverse ({}) {:.2e}", tag, fp(invRoundtrip));
//         };
//
//         check(TL, SL, "L");
//         check(TR, SR, "R");
//     }
//
//     // Transform the environments
//     bc_enveL             = enve.L;
//     bc_enveR             = enve.R;
//     bc_envvL             = envv.L;
//     bc_envvR             = envv.R;
//     bc_enveL.get_block() = transform_env(enve.L.get_block(), TL, kappaL);
//     bc_enveR.get_block() = transform_env(enve.R.get_block(), TR, kappaR);
//     bc_envvL.get_block() = transform_env(envv.L.get_block(), TL, kappaL * kappaL);
//     bc_envvR.get_block() = transform_env(envv.R.get_block(), TR, kappaR * kappaR);
//
//     // Transform the initial guess
//     initial_guess = initial;
//     initial_guess.set_tensor(transform_tensor(initial.get_tensor(), SL, SR));
// }
