#pragma once
#include "math/float_eigen.h"
//
#include "../generalized_basis_change.h"
#include "config/settings.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/linalg/matrix/to_string.h"
#include "math/linalg/tensor/to_string.h"
#include "math/svd.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/contraction.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

namespace settings {
#if defined(NDEBUG)
    inline constexpr bool debug_generalized_basis_change = false;
#else
    inline constexpr bool debug_generalized_basis_change = false;
#endif
}

using namespace tools::finite::opt::precond::generalized;
using namespace tools::finite::opt::precond::common;

template<typename Scalar> env_pair<const EnvEne<Scalar> &> tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>::get_enve_pair() const {
    return {bc_enveL, bc_enveR};
}

template<typename Scalar> env_pair<const EnvVar<Scalar> &> tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>::get_envv_pair() const {
    return {bc_envvL, bc_envvR};
}

template<typename Scalar>
bool GeneralizedBasisChange<Scalar>::is_hermitian_tensor(const Eigen::TensorRef<Eigen::Tensor<Scalar, 2>> &A) {
    auto       Am  = MapConstMatType(A.data(), A.dimension(0), A.dimension(1));
    const auto num = (Am.adjoint() - Am).norm();
    const auto den = Am.norm();
    return num <= RealScalar{1e-10f} * den; // relative tolerance
};

template<typename Scalar>
bool GeneralizedBasisChange<Scalar>::is_hermitian_matrix(const MatrixType &Am) {
    const auto num = (Am.adjoint() - Am).norm();
    const auto den = Am.norm();
    return num <= RealScalar{1e-10f} * den; // relative tolerance
};
template<typename Scalar>
bool GeneralizedBasisChange<Scalar>::is_anti_hermitian_matrix(const MatrixType &Am) {
    const auto num = (Am.adjoint() + Am).norm();
    const auto den = Am.norm();
    return num <= RealScalar{1e-10f} * den; // relative tolerance
}

template<typename Scalar>
void GeneralizedBasisChange<Scalar>::print_stats(const Eigen::Tensor<Scalar, 1> &w, std::string_view tag) {
    auto       wmap  = MapConstVecType(w.data(), w.size());
    RealScalar wmean = wmap.cwiseAbs().mean();
    RealScalar wnorm = tenx::norm(w);
    RealScalar wmax  = wmap.cwiseAbs().maxCoeff();
    tools::log->info("{} : {::.3e} | norm {:.3e} | mean {:.3e} | max {:.3e}", tag, fv(tenx::VectorCast(w.real())), fp(wnorm), fp(wmean), fp(wmax));
};
template<typename Scalar>
void GeneralizedBasisChange<Scalar>::symmetrize(Eigen::Tensor<Scalar, 2> &E) {
    auto Em = MapMatType(E.data(), E.dimension(0), E.dimension(1));
    auto er = (Em - Em.adjoint()).norm() / Em.norm();
    if(er >= RealScalar{1e-12f}) { tools::log->warn("hermiticty error: {:.5e}", fp(er)); }
    Em = (RealScalar{0.5} * (Em + Em.adjoint())).eval();
}

template<typename Scalar>
auto GeneralizedBasisChange<Scalar>::matrix_norm(const MatrixType &A) -> MatrixType {
    // |A| = (A^H A)^{1/2}
    MatrixType                                AH_A = A.adjoint() * A;
    Eigen::SelfAdjointEigenSolver<MatrixType> es(AH_A);
    if(es.info() != Eigen::Success)
        throw except::runtime_error("matrix_sqrt: Eigen::SelfAdjointEigenSolver<MatrixType> es(AH_A) failed: info {}", static_cast<int>(es.info()));
    auto U   = es.eigenvectors();
    auto sig = es.eigenvalues().cwiseMax(RealScalar(0));
    return U * sig.cwiseSqrt().asDiagonal() * U.adjoint();
};

template<typename Scalar>
void GeneralizedBasisChange<Scalar>::regularize(Eigen::Tensor<Scalar, 1> &w, const EnvWeightRegularizer ewr, [[maybe_unused]] std::string_view tag) {
    // print_stats(w, tag);
    Eigen::Map<VectorType> wmap = tenx::VectorMap(w);

    auto safe_divide = [&](RealScalar d) {
        if(d > RealScalar{0}) wmap.array() /= d;
    };
    switch(ewr) {
        case EnvWeightRegularizer::NONE: break;
        case EnvWeightRegularizer::NORM: safe_divide(wmap.norm()); break;
        case EnvWeightRegularizer::MAX: safe_divide(wmap.cwiseAbs().maxCoeff()); break;
        case EnvWeightRegularizer::MEAN: safe_divide(std::abs(wmap.mean())); break;
        case EnvWeightRegularizer::SUM: safe_divide(std::abs(wmap.sum())); break;
        default: throw except::runtime_error("EnvWeightRegularizer not implemented");
    }
    // print_stats(w, tag);
};

template<typename Scalar>
auto GeneralizedBasisChange<Scalar>::get_env_weights(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 3> &envL,
                                                     const Eigen::Tensor<Scalar, 3> &envR, const Eigen::Tensor<Scalar, 4> &mpo, EnvWeightType ewt,
                                                     EnvWeightRegularizer ewr) -> std::pair<Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 1>> {
    Eigen::Tensor<Scalar, 1> wL(envL.dimension(2));
    Eigen::Tensor<Scalar, 1> wR(envR.dimension(2));
    switch(ewt) {
        case EnvWeightType::ONES: {
            wL.setConstant(RealScalar{1});
            wR.setConstant(RealScalar{1});
            break;
        }
        case EnvWeightType::NO_PSI_TRACE: {
            // Trace physical indices off the mpo
            Eigen::Tensor<Scalar, 2> mpo_traced = mpo.trace(std::array<Eigen::Index, 2>{2, 3});
            // Trace bond indices off the environments
            Eigen::Tensor<Scalar, 1> envL_traced = envL.trace(std::array<Eigen::Index, 2>{0, 1});
            Eigen::Tensor<Scalar, 1> envR_traced = envR.trace(std::array<Eigen::Index, 2>{0, 1});

            // Contract the traced mpo with the traced environments to get the weights
            wL = envR_traced.contract(mpo_traced, tenx::idx({0}, {1}));
            wR = envL_traced.contract(mpo_traced, tenx::idx({0}, {0}));

            break;
        }
        case EnvWeightType::NO_PSI_SUM: {
            // Trace physical indices off the mpo
            Eigen::Tensor<Scalar, 2> mpo_traced = mpo.trace(std::array<Eigen::Index, 2>{2, 3});
            // Contract the traced mpo with the environments to get the weights
            wL = envR.contract(mpo_traced, tenx::idx({2}, {1})).sum(std::array<Eigen::Index, 2>{0, 1});
            wR = envL.contract(mpo_traced, tenx::idx({2}, {0})).sum(std::array<Eigen::Index, 2>{0, 1});

            break;
        }
        case EnvWeightType::WITH_PSI_TRACE: {
            wL = envR.contract(psi.conjugate(), tenx::idx({1}, {2}))
                     .contract(mpo, tenx::idx({1, 2}, {1, 3}))
                     .contract(psi, tenx::idx({0, 3}, {2, 0}))
                     .trace(std::array<Eigen::Index, 2>{0, 2});
            wR = envL.contract(psi.conjugate(), tenx::idx({1}, {1}))
                     .contract(mpo, tenx::idx({1, 2}, {0, 3}))
                     .contract(psi, tenx::idx({0, 3}, {1, 0}))
                     .trace(std::array<Eigen::Index, 2>{0, 2});

            break;
        }
        case EnvWeightType::WITH_PSI_SUM: {
            Eigen::Tensor<Scalar, 3> x3L = envR.contract(psi.conjugate(), tenx::idx({1}, {2}))
                                               .contract(mpo, tenx::idx({1, 2}, {1, 3}))
                                               .contract(psi, tenx::idx({0, 3}, {2, 0}))
                                               .shuffle(std::array<Eigen::Index, 3>{2, 0, 1});
            Eigen::Tensor<Scalar, 3> x3R = envL.contract(psi.conjugate(), tenx::idx({1}, {1}))
                                               .contract(mpo, tenx::idx({1, 2}, {0, 3}))
                                               .contract(psi, tenx::idx({0, 3}, {1, 0}))
                                               .shuffle(std::array<Eigen::Index, 3>{2, 0, 1});

            for(Eigen::Index b = 0; b < wL.size(); ++b) { wL(b) = Eigen::Tensor<Scalar, 0>(x3L.chip(b, 2).sum()).coeff(0); }
            for(Eigen::Index b = 0; b < wR.size(); ++b) { wR(b) = Eigen::Tensor<Scalar, 0>(x3R.chip(b, 2).sum()).coeff(0); }

            break;
        }
        case EnvWeightType::AB_TRACE: {
            auto get_A = [](const Eigen::Tensor<Scalar, 3> &psi) -> Eigen::Tensor<Scalar, 3> {
                auto rank_max   = psi.dimension(1);
                auto cfg        = svd::config(rank_max, 1e-15);
                auto sv         = svd::solver(cfg);
                auto [U, S, VT] = sv.schmidt_into_left_normalized(psi, psi.dimension(0));
                return U;
            };
            auto get_B = [](const Eigen::Tensor<Scalar, 3> &psi) -> Eigen::Tensor<Scalar, 3> {
                auto rank_max   = psi.dimension(2);
                auto cfg        = svd::config(rank_max, 1e-15);
                auto sv         = svd::solver(cfg);
                auto [U, S, VT] = sv.schmidt_into_right_normalized(psi, psi.dimension(0));
                return VT;
            };
            auto A = get_A(psi);
            auto B = get_B(psi);
            wL     = envR.contract(B.conjugate(), tenx::idx({1}, {2}))
                     .contract(mpo, tenx::idx({1, 2}, {1, 3}))
                     .contract(B, tenx::idx({0, 3}, {2, 0}))
                     .trace(std::array<Eigen::Index, 2>{0, 2});
            wR = envL.contract(A.conjugate(), tenx::idx({1}, {1}))
                     .contract(mpo, tenx::idx({1, 2}, {0, 3}))
                     .contract(A, tenx::idx({0, 3}, {1, 0}))
                     .trace(std::array<Eigen::Index, 2>{0, 2});

            break;
        }
        default: throw except::runtime_error("EnvWeightType not implemented");
    }

    regularize(wL, ewr, "wL");
    regularize(wR, ewr, "wR");
    return {wL, wR};
}

template<typename Scalar>
auto GeneralizedBasisChange<Scalar>::get_aggregate_envs(const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR,
                                                        const Eigen::Tensor<Scalar, 4> &mpo)
    -> std::tuple<Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 2>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 1>, Eigen::Tensor<Scalar, 2>,
                  Eigen::Tensor<Scalar, 2>> {
    Eigen::Tensor<Scalar, 2> envL_agg(envL.dimension(0), envL.dimension(1));
    Eigen::Tensor<Scalar, 2> envR_agg(envR.dimension(0), envR.dimension(1));
    Eigen::Tensor<Scalar, 2> PL;
    Eigen::Tensor<Scalar, 2> PR;

    auto [wL, wR] = get_env_weights(initial_guess.get_tensor(), envL, envR, mpo, bcfg.ewt, bcfg.ewr);
    switch(bcfg.eat) {
        case EnvAggregateType::PLAIN: {
            envL_agg = envL.contract(wL, tenx::idx({2}, {0}));
            envR_agg = envR.contract(wR, tenx::idx({2}, {0}));
            break;
        }
        case EnvAggregateType::M1: {
            auto gen_M1 = [&](const Eigen::Tensor<Scalar, 3> &env, const Eigen::Tensor<Scalar, 1> &w) -> Eigen::Tensor<Scalar, 2> {
                Eigen::Tensor<Scalar, 2> M1(env.dimension(0), env.dimension(1));
                M1.setZero();
                auto               M1_map = MapMatType(M1.data(), env.dimension(0), env.dimension(1));
                const Eigen::Index n      = env.dimension(0);
                const Eigen::Index stride = n * n;
                for(Eigen::Index b = 0; b < env.dimension(2); ++b) {
                    const Scalar   *ptr = env.data() + b * stride;
                    MapConstMatType A_b(ptr, n, n);
                    RealScalar      tr = std::abs(A_b.trace()); // pre-scale with trace
                    if(tr < RealScalar{1e-10f}) continue;
                    if(std::abs(w(b)) <= RealScalar(1e-10f)) continue;
                    if(!is_hermitian_matrix(A_b)) continue;
                    A_b /= tr;
                    M1_map.noalias() += w(b) * (A_b);
                }
                return M1;
            };
            envL_agg = gen_M1(envL, wL);
            envR_agg = gen_M1(envR, wR);
            break;
        }
        case EnvAggregateType::M2: {
            auto gen_M2 = [&](const Eigen::Tensor<Scalar, 3> &env, const Eigen::Tensor<Scalar, 1> &w) -> Eigen::Tensor<Scalar, 2> {
                Eigen::Tensor<Scalar, 2> M2(env.dimension(0), env.dimension(1));
                M2.setZero();
                auto               M2_map = MapMatType(M2.data(), env.dimension(0), env.dimension(1));
                const Eigen::Index n      = env.dimension(0);
                const Eigen::Index stride = n * n;
                for(Eigen::Index b = 0; b < env.dimension(2); ++b) {
                    const Scalar   *ptr = env.data() + b * stride;
                    MapConstMatType A_b(ptr, n, n);
                    RealScalar      tr = std::abs(A_b.trace()); // pre-scale with trace
                    if(tr < RealScalar{1e-10f}) continue;
                    if(std::abs(w(b)) <= RealScalar(1e-10f)) continue;
                    if(!is_hermitian_matrix(A_b)) continue;
                    A_b /= tr;
                    M2_map.noalias() += w(b) * (A_b.adjoint() * A_b);
                }
                return M2;
            };
            envL_agg = gen_M2(envL, wL);
            envR_agg = gen_M2(envR, wR);
            break;
        }
        case EnvAggregateType::H2_inv: {
            wL.setConstant(RealScalar{1});
            wR.setConstant(RealScalar{1});

            // Build H
            Eigen::Tensor<Scalar, 6> H =
                envL.contract(mpo, tenx::idx({2}, {0})).contract(envR, tenx::idx({2}, {2})).shuffle(std::array<Eigen::Index, 6>{2, 0, 4, 3, 1, 5});
            auto rows = H.dimension(0) * H.dimension(1) * H.dimension(2);
            auto cols = H.dimension(3) * H.dimension(4) * H.dimension(5);
            auto Hmap = MapMatType(H.data(), rows, cols);

            auto es = Eigen::SelfAdjointEigenSolver<MatrixType>(Hmap, Eigen::ComputeEigenvectors);

            if(es.info() != Eigen::Success)
                throw except::runtime_error("matrix_sqrt: Eigen::SelfAdjointEigenSolver<MatrixType> es(Hmap) failed: info {}", static_cast<int>(es.info()));
            if(es.info() == Eigen::Success) {
                auto Hinv = Eigen::Tensor<Scalar, 6>(H.dimensions());
                Hinv.setZero();

                MapMatType Hinvmap(Hinv.data(), rows, cols);
                {
                    auto U  = es.eigenvectors();
                    auto Y  = es.eigenvalues();
                    Hinvmap = U * Y.cwiseMax(eps).cwiseInverse().asDiagonal() * U.adjoint();
                }

                envL_agg = Hinv.trace(std::array<Eigen::Index, 2>{0, 3}).trace(std::array<Eigen::Index, 2>{1, 3});
                envR_agg = Hinv.trace(std::array<Eigen::Index, 2>{0, 3}).trace(std::array<Eigen::Index, 2>{0, 2});
            }
            break;
        }
        case EnvAggregateType::H2_zip: {
            throw except::runtime_error("For H2_zip you must use the custom function");
        }
        case EnvAggregateType::M2_inv: {
            auto inv = [this](const Eigen::Ref<const MatrixType> &mat) -> MatrixType {
                auto es = Eigen::SelfAdjointEigenSolver<MatrixType>(mat, Eigen::ComputeEigenvectors);
                if(es.info() != Eigen::Success) throw except::runtime_error("M2_inv: eigensolver failed: info={}", int(es.info()));

                const auto U = es.eigenvectors();
                const auto Y = es.eigenvalues().cwiseAbs().cwiseMax(eps);
                return U * Y.cwiseInverse().asDiagonal() * U.adjoint();
            };
            auto gen_M2 = [&](const Eigen::Tensor<Scalar, 3> &env, const Eigen::Tensor<Scalar, 1> &w) -> Eigen::Tensor<Scalar, 2> {
                Eigen::Tensor<Scalar, 2> M2(env.dimension(0), env.dimension(1));
                M2.setZero();
                auto               M2_map = MapMatType(M2.data(), env.dimension(0), env.dimension(1));
                const Eigen::Index n      = env.dimension(0);
                const Eigen::Index stride = n * n;
                for(Eigen::Index b = 0; b < env.dimension(2); ++b) {
                    const Scalar   *ptr = env.data() + b * stride;
                    MapConstMatType A_b(ptr, n, n);
                    // MatrixType               A             = RealScalar(0.5f) * (matrix_chip_b + matrix_chip_b.adjoint()); // Hermitian
                    RealScalar tr = std::abs(A_b.trace()); // pre-scale with trace
                    if(tr < RealScalar{1e-10f}) continue;
                    if(std::abs(w(b)) <= RealScalar(1e-10f)) continue;
                    if(!is_hermitian_matrix(A_b)) continue;
                    // A_b /= tr;
                    MatrixType A_b_inv = inv(A_b);
                    M2_map.noalias() += w(b) * (A_b_inv.adjoint() * A_b_inv);
                }
                return M2;
            };
            envL_agg = gen_M2(envL, wL);
            envR_agg = gen_M2(envR, wR);
            break;
        }

        default: throw except::runtime_error("EnvAggregateType not implemented");
    }

    if(bcfg.sym == SymmetrizeAggregates::ON) {
        symmetrize(envL_agg);
        symmetrize(envR_agg);
    }

    if constexpr(settings::debug_generalized_basis_change) {
        if(not is_hermitian_tensor(envL_agg)) throw except::runtime_error("envL_agg is not hermitian: \n{}\n", linalg::tensor::to_string(envL_agg, 8));
        if(not is_hermitian_tensor(envR_agg)) throw except::runtime_error("envR_agg is not hermitian: \n{}\n", linalg::tensor::to_string(envR_agg, 8));
    }

    return {envL_agg, envR_agg, wL, wR, PL, PR};
};

template<typename Scalar>
struct EtaStats {
    Eigen::Matrix<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1> eta; // per-μ
    typename Eigen::NumTraits<Scalar>::Real                                   median{0}, p90{0}, max{0};
};

template<typename Scalar>
EtaStats<Scalar> compute_eta(const Eigen::Tensor<Scalar, 3>                              &env, // shape [nα, nα, nμ]
                             const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U) {
    using Real   = typename Eigen::NumTraits<Scalar>::Real;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index n  = env.dimension(0);
    const Eigen::Index n2 = env.dimension(1);
    const Eigen::Index m  = env.dimension(2);
    if(n != n2) throw std::runtime_error("env2L is not square on (α,α′)");
    if(U.rows() != n || U.cols() != n) throw std::runtime_error("U_L has wrong shape");

    Eigen::Matrix<Real, Eigen::Dynamic, 1> eta(m);
    eta.setZero();

    for(Eigen::Index mu = 0; mu < m; ++mu) {
        // Extract A = env2L(:,:,μ)
        // Use your helper if available; else map via chip:
        // A = tenx::MatrixCast(env2L.chip(mu, 2), n, n);  // if you have this
        // Portable fallback:
        Eigen::Tensor<Scalar, 2> slice = env.chip(mu, 2); // rank-2 tensor view [n,n]
        auto                     A     = Eigen::Map<const Matrix>(slice.data(), n, n);

        // Transform: B = U_L^H * A * U_L
        Matrix B = U.adjoint() * A * U;

        // squared Frobenius norms
        Real normB2 = B.squaredNorm();
        if(normB2 < Real(1e-300)) {
            eta(mu) = Real(0);
            continue;
        }

        Real diag2 = B.diagonal().squaredNorm();

        // η^2 = 1 - ||diag(B)||_F^2 / ||B||_F^2  ∈ [0,1]
        Real eta2 = Real(1) - std::min<Real>(Real(1), diag2 / normB2);
        eta(mu)   = std::sqrt(std::max<Real>(Real(0), eta2));

        // Diagonal matrix from B.diagonal()
        // Matrix Bdiag = B.diagonal().asDiagonal();

        // Frobenius norms
        // Real denom = std::max<Real>(Real(1e-30), B.norm());
        // Real numer = (B - Bdiag).norm();
        // eta(mu)    = numer / denom;
    }

    // Summaries
    EtaStats<Scalar> out;
    out.eta = eta;

    // Median, p90, max
    Eigen::Matrix<Real, Eigen::Dynamic, 1> tmp = eta;
    std::vector<Real>                      v(tmp.data(), tmp.data() + tmp.size());
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    out.median = v[v.size() / 2];
    std::sort(v.begin(), v.end());
    out.p90 = v[static_cast<size_t>(0.9 * (v.size() - 1))];
    out.max = v.back();

    return out;
}

template<typename Scalar>
auto GeneralizedBasisChange<Scalar>::get_generalized_transforms_H2_zip(const Eigen::Tensor<Scalar, 3> &env2L, const Eigen::Tensor<Scalar, 3> &env2R,
                                                                       const Eigen::Tensor<Scalar, 4> &mpo2) -> Transform_H2_zip {
    assert(bcfg.eat == EnvAggregateType::H2_zip);
    auto  tf               = Transform_H2_zip();
    auto &mpo              = mpo2;
    auto &envL             = env2L;
    auto &envR             = env2R;
    auto &wL               = tf.w2L;
    auto &wR               = tf.w2R;
    auto &PL               = tf.P2L;
    auto &PR               = tf.P2R;
    auto &TL               = tf.T2L;
    auto &TR               = tf.T2R;
    auto &SL               = tf.S2L;
    auto &SR               = tf.S2R;
    auto &UL               = tf.U2L;
    auto &UR               = tf.U2R;
    auto &envL_agg_zip_inv = tf.env2L_agg;
    auto &envR_agg_zip_inv = tf.env2R_agg;
    auto &envL_zip         = tf.env2L_zip;
    auto &envR_zip         = tf.env2R_zip;

    // Compress the environments using SVD

    // auto get_P_from_env = [](const Eigen::Tensor<Scalar, 3> &env, Eigen::Index rank_max) -> MatrixType {
    //     auto cfg        = svd::config(rank_max, 1e-12);
    //     auto sv         = svd::solver(cfg);
    //     auto envmap     = MapConstMatType(env.data(), env.dimension(0), env.dimension(1) * env.dimension(2));
    //     auto [U, S, VT] = sv.do_svd(envmap, cfg);
    //     return U;
    // };

    auto get_PL_from_psi = [](const Eigen::Tensor<Scalar, 3> &psi, Eigen::Index rank_max) -> MatrixType {
        const auto               shf3   = std::array<Eigen::Index, 3>{1, 0, 2};
        const auto               shp2   = std::array<Eigen::Index, 2>{psi.dimension(1), psi.dimension(0) * psi.dimension(2)};
        Eigen::Tensor<Scalar, 2> psiL   = psi.shuffle(shf3).reshape(shp2);
        auto                     cfg    = svd::config(rank_max, 1e-12);
        auto                     sv     = svd::solver(cfg);
        auto                     psimap = MapConstMatType(psiL.data(), psiL.dimension(0), psiL.dimension(1));
        auto [U, S, VT]                 = sv.do_svd(psimap, cfg);
        return U;
    };
    auto get_PR_from_psi = [](const Eigen::Tensor<Scalar, 3> &psi, Eigen::Index rank_max) -> MatrixType {
        auto cfg        = svd::config(rank_max, 1e-14);
        auto sv         = svd::solver(cfg);
        auto psimap     = MapConstMatType(psi.data(), psi.dimension(0) * psi.dimension(1), psi.dimension(2));
        auto [U, S, VT] = sv.do_svd(psimap, cfg);
        return VT.transpose();
    };

    // auto get_env_zip = [&get_P_from_env](const Eigen::Tensor<Scalar, 3> &env, Eigen::Index rank_max) -> std::pair<Eigen::Tensor<Scalar, 3>, MatrixType> {
    //     auto P       = get_P_from_env(env, rank_max); // The P matrix is a projector to a space of smaller bond dimension
    //     auto Pmap    = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(P.data(), P.rows(), P.cols());
    //     auto env_zip = env.contract(Pmap, tenx::idx({0}, {0})).contract(Pmap.conjugate(), tenx::idx({0}, {0})).shuffle(std::array<Eigen::Index, 3>{1, 2, 0});
    //     return {env_zip, P};
    // };

    auto get_envL_zip_from_psi = [&get_PL_from_psi](const Eigen::Tensor<Scalar, 3> &env, const Eigen::Tensor<Scalar, 3> &psi,
                                                    Eigen::Index rank_max) -> std::pair<Eigen::Tensor<Scalar, 3>, MatrixType> {
        auto P       = get_PL_from_psi(psi, rank_max); // The P matrix is a projector to a space of smaller bond dimension
        auto Pmap    = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(P.data(), P.rows(), P.cols());
        auto env_zip = env.contract(Pmap, tenx::idx({0}, {0})).contract(Pmap.conjugate(), tenx::idx({0}, {0})).shuffle(std::array<Eigen::Index, 3>{1, 2, 0});
        return {env_zip, P};
    };
    auto get_envR_zip_from_psi = [&get_PR_from_psi](const Eigen::Tensor<Scalar, 3> &env, const Eigen::Tensor<Scalar, 3> &psi,
                                                    Eigen::Index rank_max) -> std::pair<Eigen::Tensor<Scalar, 3>, MatrixType> {
        auto P       = get_PR_from_psi(psi, rank_max); // The P matrix is a projector to a space of smaller bond dimension
        auto Pmap    = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(P.data(), P.rows(), P.cols());
        auto env_zip = env.contract(Pmap, tenx::idx({0}, {0})).contract(Pmap.conjugate(), tenx::idx({0}, {0})).shuffle(std::array<Eigen::Index, 3>{1, 2, 0});
        return {env_zip, P};
    };
    Eigen::Index max_size = settings::precision::eig_max_size;
    Eigen::Index max_bond = std::max<Eigen::Index>(40, static_cast<Eigen::Index>(std::sqrt(max_size / mpo.dimension(2))));
    Eigen::Index max_chiL = std::min(envL.dimension(0), max_bond);
    Eigen::Index max_chiR = std::min(envR.dimension(0), max_bond);

    std::tie(envL_zip, PL) = get_envL_zip_from_psi(envL, initial_guess.get_tensor(), max_chiL);
    std::tie(envR_zip, PR) = get_envR_zip_from_psi(envR, initial_guess.get_tensor(), max_chiR);
    tools::log->info("psi: {} | PL: [{}, {}] | PR: [{}, {}] | max_chiL: {} | max_chiR: {}", initial_guess.get_tensor().dimensions(), PL.rows(), PL.cols(),
                     PR.rows(), PR.cols(), max_chiL, max_chiR);
    // Build H
    Eigen::Tensor<Scalar, 6> H =
        envL_zip.contract(mpo, tenx::idx({2}, {0})).contract(envR_zip, tenx::idx({2}, {2})).shuffle(std::array<Eigen::Index, 6>{2, 0, 4, 3, 1, 5});
    auto rows = H.dimension(0) * H.dimension(1) * H.dimension(2);
    auto cols = H.dimension(3) * H.dimension(4) * H.dimension(5);

    auto es = eig::solver();
    es.eig<eig::Form::SYMM>(H.data(), rows, eig::Vecs::ON);
    if(!es.result.meta.eigvecsR_found) throw except::runtime_error("matrix_sqrt: es(Hmap) failed}");

    if(es.result.meta.eigvecsR_found) {
        auto Hinv = Eigen::Tensor<Scalar, 6>(H.dimensions());
        Hinv.setZero();

        MapMatType Hinvmap(Hinv.data(), rows, cols);
        {
            // auto U  = es.eigenvectors();
            // auto Y  = es.eigenvalues();
            auto U  = eig::view::get_eigvecs<Scalar>(es.result);
            auto Y  = eig::view::get_eigvals<RealScalar>(es.result);
            Hinvmap = U * Y.cwiseMax(eps).cwiseInverse().asDiagonal() * U.adjoint();
        }

        envL_agg_zip_inv = Hinv.trace(std::array<Eigen::Index, 2>{0, 3}).trace(std::array<Eigen::Index, 2>{1, 3});
        envR_agg_zip_inv = Hinv.trace(std::array<Eigen::Index, 2>{0, 3}).trace(std::array<Eigen::Index, 2>{0, 2});
        if(bcfg.sym == SymmetrizeAggregates::ON) {
            symmetrize(envL_agg_zip_inv);
            symmetrize(envR_agg_zip_inv);
        }
    }

    // Step 2, produce transformers TL, TR, SL, SR
    auto get_transform_H2_zip = [this](const Eigen::Tensor<Scalar, 1> &w, const MatrixType &P, const MatrixType &agg,
                                       const Eigen::Tensor<Scalar, 3> &env_zip) -> std::tuple<MatrixType, MatrixType, MatrixType> {
        assert(agg.rows() == env_zip.dimension(0));
        assert(agg.cols() == env_zip.dimension(1));
        assert(w.size() == env_zip.dimension(2));
        auto es = Eigen::SelfAdjointEigenSolver<MatrixType>(agg, Eigen::ComputeEigenvectors);
        if(es.info() != Eigen::Success) throw except::runtime_error("Generalized eigen solve failed: info={}", int(es.info()));

        auto       U = es.eigenvectors();
        auto       Y = es.eigenvalues();
        VectorReal D;
        switch(bcfg.tst) {
            case TransformSpectrumType::EnvAggregateSpectrum: {
                // VectorReal absEpsY = Y.cwiseAbs().cwiseMax(eps).normalized();
                D = Y.cwiseAbs().cwiseMax(eps);
                break;
            }
            case TransformSpectrumType::EnvProjectedDiagonal: {
                assert(w.size() == env_zip.dimension(2));
                D.setZero(env_zip.dimension(0));
                for(Eigen::Index b = 0; b < env_zip.dimension(2); ++b) {
                    auto env2_b     = Eigen::Tensor<Scalar, 2>(env_zip.chip(b, 2));
                    auto env2_b_map = MapMatType(env2_b.data(), env2_b.dimension(0), env2_b.dimension(1));
                    D += (w(b) * (U.adjoint() * env2_b_map * U).diagonal()).real();
                }
                break;
            }
        }
        D = D.cwiseAbs();
        D /= D.mean();
        assert(!D.isZero());
        assert(D.allFinite());

        // VectorReal absEpsY = Y.cwiseAbs().cwiseMax(eps);
        // absEpsY /= absEpsY.mean();
        // tools::log->info("Y : {::.5e}", fv(Y));
        // tools::log->info("D : {::.5e}", fv(D));
        VectorReal invPowD = D.array().pow(-RealScalar{bcfg.alpha} / 2);
        VectorReal absPowD = D.array().pow(+RealScalar{bcfg.alpha} / 2);
        MatrixType T_zip   = U * absPowD.asDiagonal() * U.adjoint();
        MatrixType S_zip   = U * invPowD.asDiagonal() * U.adjoint();

        // We expect to have PL and PR also
        assert(P.size() != 0 and P.allFinite());
        MatrixType P_ortho = MatrixType::Identity(P.rows(), P.rows()) - P * P.adjoint();
        MatrixType T       = P * T_zip * P.adjoint() + P_ortho;
        MatrixType S       = P * S_zip * P.adjoint() + P_ortho;

        return {T, S, U};
    };

    auto envL_agg_inv_map = MapConstMatType(envL_agg_zip_inv.data(), envL_agg_zip_inv.dimension(0), envL_agg_zip_inv.dimension(1));
    auto envR_agg_inv_map = MapConstMatType(envR_agg_zip_inv.data(), envR_agg_zip_inv.dimension(0), envR_agg_zip_inv.dimension(1));

    // Get the weights
    std::tie(wL, wR) = get_env_weights(initial_guess.get_tensor(), envL, envR, mpo, bcfg.ewt, bcfg.ewr);

    std::tie(TL, SL, UL) = get_transform_H2_zip(wL, PL, envL_agg_inv_map, envL_zip);
    std::tie(TR, SR, UR) = get_transform_H2_zip(wR, PR, envR_agg_inv_map, envR_zip);

    return tf;
}

template<typename Scalar>
auto GeneralizedBasisChange<Scalar>::get_generalized_transforms([[maybe_unused]] const Eigen::Tensor<Scalar, 3> &env1,     //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 3> &env2,     //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 2> &env1_agg, //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 2> &env2_agg, //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 1> &w1,       //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 1> &w2,       //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 2> &P1,       //
                                                                [[maybe_unused]] const Eigen::Tensor<Scalar, 2> &P2)
    -> std::tuple<MatrixType, MatrixType, MatrixType, RealScalar> {
    if(env1_agg.dimension(0) != env2_agg.dimension(0))
        throw except::runtime_error("env1_agg/env2_agg dimension mismatch: {} vs {}", env1_agg.dimension(0), env2_agg.dimension(0));

    // auto agg1 = MapConstMatType(env1_agg.data(), env1_agg.dimension(0), env1_agg.dimension(1));
    auto agg2 = MapConstMatType(env2_agg.data(), env2_agg.dimension(0), env2_agg.dimension(1));
    auto ges  = Eigen::SelfAdjointEigenSolver<MatrixType>(agg2, Eigen::ComputeEigenvectors);
    // auto ges  = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>(agg1, agg2, Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
    if(ges.info() != Eigen::Success) throw except::runtime_error("Generalized eigen solve failed: info={}", int(ges.info()));

    auto U = ges.eigenvectors();
    auto Y = ges.eigenvalues();

    MatrixType T;
    MatrixType S;
    VectorReal D;
    switch(bcfg.tst) {
        case TransformSpectrumType::EnvAggregateSpectrum: {
            D = Y.cwiseAbs().cwiseMax(eps);
            break;
        }
        case TransformSpectrumType::EnvProjectedDiagonal: {
            assert(w2.size() == env2.dimension(2));
            D.setZero(env2.dimension(0));
            for(Eigen::Index b = 0; b < env2.dimension(2); ++b) {
                auto env2_b     = Eigen::Tensor<Scalar, 2>(env2.chip(b, 2));
                auto env2_b_map = MapMatType(env2_b.data(), env2_b.dimension(0), env2_b.dimension(1));
                // D += (w2(b) * (U.adjoint() * matrix_norm(A_b) * U).diagonal()).real();
                // if(!is_hermitian_matrix(A_b)) continue;
                // RealScalar tr = std::abs(A_b.trace());
                // if(tr < RealScalar{1e-10f}) continue;
                // if(std::abs(w2(b)) <= RealScalar(1e-10f)) continue;
                // D += (w2(b) * (U.adjoint() * matrix_norm(A_b) * U).diagonal()).real();
                D += (w2(b) * (U.adjoint() * env2_b_map * U).diagonal()).real();
            }
            break;
        }
    }
    D = D.cwiseAbs();
    D /= D.mean();
    // tools::log->info("Y : {::.5e}", fv(Y));
    // tools::log->info("D : {::.5e}", fv(D));
    assert(!D.isZero());
    assert(D.allFinite());

    switch(bcfg.eat) {
        case EnvAggregateType::H2_zip: throw std::runtime_error("For H2_zip, call the custom function: get_generalized_transforms_H2_zip(...)");
        case EnvAggregateType::H2_inv: [[fallthrough]];
        case EnvAggregateType::M2_inv: {
            VectorReal absEpsY = Y.cwiseAbs().cwiseMax(eps);
            VectorReal invPowD = D.array().pow(-RealScalar{bcfg.alpha} / 2);
            VectorReal absPowD = D.array().pow(+RealScalar{bcfg.alpha} / 2);
            T                  = U * absPowD.asDiagonal() * U.adjoint(); // == U |Y|^{+α/2} U^H
            S                  = U * invPowD.asDiagonal() * U.adjoint(); // == U |Y|^{-α/2} U^H
            break;
        }
        default: {
            VectorReal invPowD = D.array().pow(-RealScalar{bcfg.alpha} / 2);
            VectorReal absPowD = D.array().pow(+RealScalar{bcfg.alpha} / 2);
            T                  = U * invPowD.asDiagonal() * U.adjoint(); // == U |Y|^{-α/2} U^H
            S                  = U * absPowD.asDiagonal() * U.adjoint(); // == U |Y|^{+α/2} U^H
        }
    }

    auto get_Y_normalization = [&]() -> RealScalar {
        switch(bcfg.bcs) {
            case BasisChangeScale::NONE: return RealScalar{1};
            case BasisChangeScale::MIN: return D.minCoeff();
            case BasisChangeScale::AVG: return D.mean();
            case BasisChangeScale::MAX: return D.maxCoeff();
            case BasisChangeScale::SQRTMIN: return D.cwiseSqrt().minCoeff();
            case BasisChangeScale::SQRTAVG: return D.cwiseSqrt().mean();
            case BasisChangeScale::SCALE: return bcfg.scale;
            default: throw except::runtime_error("Unknown BasisChangeScale");
        }
    };
    auto mY = get_Y_normalization();

    if constexpr(settings::debug_generalized_basis_change) {
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

        auto check_projector = [&](const MatrixType &TT, const MatrixType &SS) {
            MatrixType P    = SS * TT; // should be ~projector
            RealScalar symm = herm_resid(P);
            RealScalar idem = (P * P - P).norm() / std::max<RealScalar>(RealScalar{1}, P.norm());
            tools::log->info("P symmetry {:.2e}, idempotence {:.2e}", fp(symm), fp(idem));
            RealScalar inv_rt = (SS * (TT * SS) - SS).norm() / std::max<RealScalar>(RealScalar{1}, SS.norm());
            tools::log->info("round-trip inverse: {:.2e}", fp(inv_rt));
        };
        auto check_inverse = [&](const MatrixType &TT, const MatrixType &SS) {
            const auto Id = MatrixType::Identity(TT.rows(), TT.cols());

            RealScalar err_ST = (SS * TT - Id).norm() / std::max<RealScalar>(RealScalar{1}, Id.norm());
            RealScalar err_TS = (TT * SS - Id).norm() / std::max<RealScalar>(RealScalar{1}, Id.norm());

            tools::log->info("S*T inverse error {:.2e}", fp(err_ST));
            tools::log->info("T*S inverse error {:.2e}", fp(err_TS));
        };
        auto check_congruence = [&](const MatrixType &X, const MatrixType &TT, const MatrixType &G, std::string_view lbl) {
            MatrixType W   = TT.adjoint() * X * TT;
            RealScalar rel = rel_err(W, G, X.norm());
            RealScalar hrm = herm_resid(W);
            tools::log->info("{}: ||T^H X T - G||/max(1,||X||) = {:.3e}", lbl, fp(rel));
            tools::log->info("{}: Herm residual (rel) = {:.3e}", lbl, fp(hrm));
            eig_range(W, std::string(lbl) + "_sym");
        };
        if(bcfg.eat != EnvAggregateType::H2_zip) {
            // In H2Zip we have unequal sizes for agg1 agg2 and T

            // Targets for magnitude-only tempered scheme:
            //   T = U |Y|^{-alpha/2} U^H  ⇒  T^H A T = U [sgn(Y)|Y|^{1-alpha}] U^H,
            //                                T^H B T = U [|Y|^{-alpha}]        U^H.
            VectorReal sgnY = Y.array().sign().matrix();
            // const auto n       = env1_agg.dimension(0);
            MatrixType target1 = U * (sgnY.array() * D.array().pow(RealScalar{1} - bcfg.get_alpha<RealScalar>())).matrix().asDiagonal() * U.adjoint();
            MatrixType target2 = U * D.array().pow(-bcfg.get_alpha<RealScalar>()).matrix().asDiagonal() * U.adjoint();

            // Logs
            // check_congruence(agg1, T, target1, "[agg1]");
            check_congruence(agg2, T, target2, "[agg2]");
            check_inverse(T, S);
        }
    }

    return {U, T, S, std::sqrt(mY)};
};


template<typename Scalar>
tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>::GeneralizedBasisChange(
    const opt_mps<Scalar>                  &initial, /*!< Initial guess */
    const Eigen::Tensor<Scalar, 4>         &mpo1_,   /*!< Multisite mpo for H1 */
    const Eigen::Tensor<Scalar, 4>         &mpo2_,   /*!< Multisite mpo for H2 */
    const env_pair<const EnvEne<Scalar> &> &env1,    /*!< Multisite env for H1 */
    const env_pair<const EnvVar<Scalar> &> &env2,    /*!< Multisite env for H2 */
    BasisChangeConfig                       bcfg_)
    : sites(initial.get_sites()), initial_guess(initial), mpo1(mpo1_), mpo2(mpo2_), bcfg(bcfg_) {
    if(bcfg.scale <= 0) throw except::runtime_error("Scale must be positive");

    if(bcfg.ewt == EnvWeightType::OFF) {
        bc_enveL = env1.L;
        bc_enveR = env1.R;
        bc_envvL = env2.L;
        bc_envvR = env2.R;

        initial_guess.set_tensor(initial.get_tensor());

        const auto dims = initial.get_tensor().dimensions();
        shape_orig  = {dims[0], dims[1], dims[2]};
        shape_tilde = shape_orig;

        TL = MatrixType::Identity(dims[1], dims[1]);
        TR = MatrixType::Identity(dims[2], dims[2]);
        SL = MatrixType::Identity(dims[1], dims[1]);
        SR = MatrixType::Identity(dims[2], dims[2]);

        UL = MatrixType::Identity(dims[1], dims[1]);
        UR = MatrixType::Identity(dims[2], dims[2]);

        kappaL = RealScalar{1};
        kappaR = RealScalar{1};
        pass   = 0;
        return;
    }

    const Eigen::Tensor<Scalar, 3> &env1L = env1.L.get_block();
    const Eigen::Tensor<Scalar, 3> &env1R = env1.R.get_block();
    const Eigen::Tensor<Scalar, 3> &env2L = env2.L.get_block();
    const Eigen::Tensor<Scalar, 3> &env2R = env2.R.get_block();

    if(bcfg.eat == EnvAggregateType::H2_zip) {
        auto tf = get_generalized_transforms_H2_zip(env2L, env2R, mpo2);

        TL     = tf.T2L;
        TR     = tf.T2R;
        SL     = tf.S2L;
        SR     = tf.S2R;
        UL     = tf.U2L;
        UR     = tf.U2R;
        kappaL = RealScalar{1};
        kappaR = RealScalar{1};

    } else {
        auto [env1L_agg, env1R_agg, w1L, w1R, P1L, P1R] = get_aggregate_envs(env1L, env1R, mpo1);
        auto [env2L_agg, env2R_agg, w2L, w2R, P2L, P2R] = get_aggregate_envs(env2L, env2R, mpo2);

        std::tie(UL, TL, SL, kappaL) = get_generalized_transforms(env1L, env2L, env1L_agg, env2L_agg, w1L, w2L, P1L, P2L);
        std::tie(UR, TR, SR, kappaR) = get_generalized_transforms(env1R, env2R, env1R_agg, env2R_agg, w1R, w2R, P1R, P2R);
    }

    // auto etaL_stats = compute_eta(env2L, UL);
    // auto etaR_stats = compute_eta(env2R, UR);

    // tools::log->info("etaL=max={:.5e}, median={:.5e}, p90={:.5e}", fp(etaL_stats.max), fp(etaL_stats.median), fp(etaL_stats.p90));
    // tools::log->info("etaL: \n{}", linalg::matrix::to_string(etaL_stats.eta, 8));
    // tools::log->info("etaR=max={:.5e}, median={:.5e}, p90={:.5e}", fp(etaR_stats.max), fp(etaR_stats.median), fp(etaR_stats.p90));
    // tools::log->info("etaR: \n{}", linalg::matrix::to_string(etaR_stats.eta, 8));

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

    // auto gamma = RealScalar{1}; // std::sqrt(RealScalar{1} / std::max(eps, envvL_norm * envvR_norm));
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
    initial_guess.set_tensor(transform_tensor(initial.get_tensor(), SL, SR));

    // Rescale the environments so that <H2> = 1 in the new basis
    // auto get_expval_H2 = [this]() -> RealScalar {
    //     const auto &psi   = initial_guess.get_tensor();
    //     const auto &env2L = bc_envvL.get_block();
    //     const auto &env2R = bc_envvR.get_block();
    //     auto        vv    = tools::common::contraction::contract_mps_overlap(psi, psi);
    //     auto        vh2v  = tools::common::contraction::expectation_value(psi, mpo2, env2L, env2R);
    //     return std::abs(vh2v / vv);
    // };
    // RealScalar eh2   = get_expval_H2();
    // RealScalar gamma = RealScalar{1} / std::max(std::sqrt(eh2), RealScalar{1e-30f});
    // bc_enveL.get_block() *= bc_enveL.get_block().constant(gamma);
    // bc_enveR.get_block() *= bc_enveR.get_block().constant(gamma);
    // bc_envvL.get_block() *= bc_envvL.get_block().constant(gamma);
    // bc_envvR.get_block() *= bc_envvR.get_block().constant(gamma);
}

template<typename Scalar>
tools::finite::opt::precond::generalized::GeneralizedBasisChange<Scalar>::GeneralizedBasisChange(const opt_mps<Scalar>       &initial,
                                                                                                 const TensorsFinite<Scalar> &tensors, BasisChangeConfig bcfg_)
    : GeneralizedBasisChange(initial, tensors.get_model().template get_multisite_mpo<Scalar>(initial.get_sites()),
                             tensors.get_model().template get_multisite_mpo_squared<Scalar>(initial.get_sites()),
                             tensors.get_edges().get_multisite_env_ene(initial.get_sites()), tensors.get_edges().get_multisite_env_var(initial.get_sites()),
                             bcfg_) {}

template<typename Scalar>
GeneralizedBasisChange<Scalar>::GeneralizedBasisChange(const GeneralizedBasisChange<Scalar> &bc)
    : GeneralizedBasisChange(bc.initial_guess, bc.mpo1, bc.mpo2, bc.get_enve_pair(), bc.get_envv_pair(), bc.bcfg) {
    if(bcfg.ewt == EnvWeightType::OFF) return;
    // bc now has the old basis, and "this" object now has the new transformed basis.

    pass = bc.pass + 1;
    // We need to update the transforms so that we can undo the transformation.
    TL = (bc.TL * TL).eval();
    TR = (bc.TR * TR).eval();
    SL = (SL * bc.SL).eval();
    SR = (SR * bc.SR).eval();
}

template<typename Scalar>
GeneralizedBasisChange<Scalar>::GeneralizedBasisChange(const GeneralizedBasisChange<Scalar> &bc, BasisChangeConfig bcfg_)
    : GeneralizedBasisChange(bc.initial_guess, bc.mpo1, bc.mpo2, bc.get_enve_pair(), bc.get_envv_pair(), bcfg_) {
    if(this->bcfg.ewt == EnvWeightType::OFF) return;
    // bc now has the old basis, and "this" object now has the new transformed basis.
    pass = bc.pass + 1;
    // We need to update the transforms so that we can undo the transformation.
    TL = (bc.TL * TL).eval();
    TR = (bc.TR * TR).eval();
    SL = (SL * bc.SL).eval();
    SR = (SR * bc.SR).eval();
}