#include "../../env.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "../mixing_terms.h"
#include "config/settings.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include <Eigen/Core>
#include <Eigen/QR>

template<typename T>
void tools::finite::env::internal::merge_mixing_terms_MP_N0(const StateFinite<T>      &state, //
                                                            MpsSite<T>                &mpsL,  //
                                                            const Eigen::Tensor<T, 3> &MP,    //
                                                            MpsSite<T>                &mpsR,  //
                                                            const Eigen::Tensor<T, 3> &N0,    //
                                                            const svd::config         &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR.
    //           * MP is [AC(i), PL]^T
    //           * N0  is [B(i+1), 0]^T
    //           * mpsL:  A(i) = U
    //           * mpsL:  C(i) = S
    //           * mpsR:  B(i+1) = V * MR_PR (loses right normalization, but that is not be needed during the next optimization)
    //

    tools::log->trace("merge_mixing_terms_MP_N0: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());
    svd::solver svd;
    // using Real = decltype(std::real(std::declval<T>()));
    auto posR  = mpsR.get_position();
    assert(mpsL.get_label() == "AC");
    assert(mpsR.get_label() == "B");
    auto [U, S, V] = svd.schmidt_into_left_normalized(MP, mpsL.spin_dim(), svd_cfg);
    mpsL.set_M(U);
    mpsL.set_LC(S);
    mpsL.stash_V(V, posR);
    mpsR.set_M(N0);
    mpsR.take_stash(mpsL);
    state.clear_cache();
    state.clear_measurements();
    // {
    //     // Make mpsR normalized so that later checks can succeed
    //     auto           multisite_mpsR = state.template get_multisite_mps<T>({posR});
    //     auto           norm_old       = tenx::norm(multisite_mpsR);
    //     constexpr auto eps            = std::numeric_limits<Real>::epsilon();
    //     if(std::abs(norm_old) < eps) { throw except::runtime_error("merge_expansion_term_PL: norm_old {:.5e} < eps {:.5e}", fp(norm_old), fp(eps)); }
    //     Eigen::Tensor<T, 3> M_tmp = mpsR.get_M_bare() * mpsR.get_M_bare().constant(std::pow(norm_old, -Real(0.5))); // Rescale by the norm
    //     mpsR.set_M(M_tmp);
    //     state.clear_cache();
    //     state.clear_measurements();
    //     // if constexpr(settings::debug_expansion) {
    //     auto mpsR_final = state.template get_multisite_mps<T>({mpsR.get_position()});
    //     auto norm_new   = tenx::norm(mpsR_final);
    //     tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsR.get_label(), mpsR.get_position(), fp(std::abs(norm_old)),
    //                       fp(std::abs(norm_new)));
    //     // }
    // }
}

template<typename T>
void tools::finite::env::internal::merge_mixing_terms_N0_MP(const StateFinite<T>      &state, //
                                                            MpsSite<T>                &mpsL,  //
                                                            const Eigen::Tensor<T, 3> &N0,    //
                                                            MpsSite<T>                &mpsR,  //
                                                            const Eigen::Tensor<T, 3> &MP,    //
                                                            const svd::config         &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR

    // After USV = SVD(MP):
    //           * N0 is [A, 0]
    //           * MP is [CB, P]^T            <--- Note that we use full AC(i)! Not bare A(i)

    tools::log->trace("merge_mixing_terms_N0_MP: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());
    using Real = decltype(std::real(std::declval<T>()));
    svd::solver svd;
    auto        posL = mpsL.get_position();
    auto        labL = mpsL.get_label();
    auto        labR = mpsR.get_label();
    assert(labL == "AC");
    assert(labR == "B");
    auto [U, S, V] = svd.schmidt_into_right_normalized(MP, mpsR.spin_dim(), svd_cfg);
    mpsR.set_M(V);
    mpsR.stash_U(U, posL);
    mpsR.stash_C(S, -1.0, posL); // Set a negative truncation error to ignore it.
    mpsL.set_M(N0);
    mpsL.take_stash(mpsR); // normalization of mpsL is lost here.

    state.clear_cache();
    state.clear_measurements();
    // {
    //     // Make mpsL normalized so that later checks can succeed
    //     auto           multisite_mpsL = state.template get_multisite_mps<T>({posL});
    //     auto           norm_old       = tenx::norm(multisite_mpsL);
    //     constexpr auto eps            = std::numeric_limits<Real>::epsilon();
    //     if(std::abs(norm_old) < eps) { throw except::runtime_error("merge_expansion_term_PR: norm_old {:.5e} < eps {:.5e}", fp(norm_old), fp(eps)); }
    //     Eigen::Tensor<T, 3> M_tmp = mpsL.get_M_bare() * mpsL.get_M_bare().constant(std::pow(norm_old, Real{-0.5})); // Rescale
    //     mpsL.set_M(M_tmp);
    //     state.clear_cache();
    //     state.clear_measurements();
    //     // if constexpr(settings::debug_expansion) {
    //     auto mpsL_final = state.template get_multisite_mps<T>({posL});
    //     auto norm_new   = tenx::norm(mpsL_final);
    //     tools::log->debug("Normalized expanded mps {}({}): {:.16f} -> {:.16f}", mpsL.get_label(), mpsL.get_position(), fp(std::abs(norm_old)),
    //                       fp(std::abs(norm_new)));
    //     // }
    // }
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> tools::finite::env::internal::get_mixing_terms_MP_N0(const Eigen::Tensor<T, 3> &M,  //
                                                                                                         const Eigen::Tensor<T, 3> &N,  //
                                                                                                         const Eigen::Tensor<T, 3> &P1, //
                                                                                                         const Eigen::Tensor<T, 3> &P2, //
                                                                                                         const BondExpansionConfig &cfg) {
    /*
        We form M_P = [α₀M | α₁P1 | α₂P2] by concatenating along dimension 2.
        The scaling factors α control the amount of perturbation provided by P1 and P2.
        In matrix-language, the rescaled columns of P1 and P2 are added as new columns to M.

        We then apply a column-pivoting QR factorization of M_P.
        The factorization sorts the columns in M_P such that they contribute decreasingly to the basis formed by previous columns.
        We want to preserve the column order of M, but use this sorting mechanism on the columns from P1 and P2.
        Since M arises from an SVD (typically U * S), its columns are already orthogonal (although not orthonormal due to S).


     */
    assert(M.dimension(2) == N.dimension(1));
    assert(P1.size() == 0 or P1.dimension(1) == M.dimension(1));
    assert(P1.size() == 0 or P1.dimension(0) == M.dimension(0));
    assert(P2.size() == 0 or P2.dimension(1) == M.dimension(1));
    assert(P2.size() == 0 or P2.dimension(0) == M.dimension(0));
    using MatrixType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = typename Eigen::Tensor<T, 3>::RealScalar;

    auto offM  = Eigen::DSizes<long, 3>{0, 0, 0};
    auto offP1 = Eigen::DSizes<long, 3>{0, 0, M.dimension(2)};
    auto offP2 = Eigen::DSizes<long, 3>{0, 0, M.dimension(2) + P1.dimension(2)};

    // Rescale P1 and P2 so that their column norms equal 1, then scale by mixing factor

    auto P1_matrix     = tenx::MatrixMap(P1, P1.dimension(0) * P1.dimension(1), P1.dimension(2));
    auto P2_matrix     = tenx::MatrixMap(P2, P2.dimension(0) * P2.dimension(1), P2.dimension(2));
    auto P1_maxcolnorm = RealScalar{1};
    auto P2_maxcolnorm = RealScalar{1};
    if(P1_matrix.cols() > 0) P1_maxcolnorm = P1_matrix.colwise().norm().maxCoeff();
    if(P1_matrix.cols() > 0) P2_maxcolnorm = P2_matrix.colwise().norm().maxCoeff();

    auto M_P = Eigen::Tensor<T, 3>(M.dimension(0), M.dimension(1), M.dimension(2) + P1.dimension(2) + P2.dimension(2));
    M_P.setZero();

    auto M_M  = M_P.slice(offM, M.dimensions());
    auto M_P1 = M_P.slice(offP1, P1.dimensions());
    auto M_P2 = M_P.slice(offP2, P2.dimensions());

    M_M  = M;
    M_P1 = P1 * P1.constant(static_cast<RealScalar>(cfg.mixing_factor) / P1_maxcolnorm);
    M_P2 = P2 * P2.constant(static_cast<RealScalar>(cfg.mixing_factor) / P2_maxcolnorm);

    auto N_0 = Eigen::Tensor<T, 3>(N.dimension(0), M_P.dimension(2), N.dimension(2));
    N_0.setZero();
    auto extN_0                              = std::array<long, 3>{N.dimension(0), std::min(N.dimension(1), M_P.dimension(2)), N.dimension(2)};
    N_0.slice(tenx::array3{0, 0, 0}, extN_0) = N.slice(tenx::array3{0, 0, 0}, extN_0); // Copy N into N_0
    return {M_P, N_0};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> tools::finite::env::internal::get_mixing_terms_N0_MP(const Eigen::Tensor<T, 3> &N,  // Gets padded
                                                                                                         const Eigen::Tensor<T, 3> &M,  // Gets expanded
                                                                                                         const Eigen::Tensor<T, 3> &P1, //
                                                                                                         const Eigen::Tensor<T, 3> &P2, //
                                                                                                         const BondExpansionConfig &cfg) {
    constexpr auto shf = std::array<long, 3>{0, 2, 1};
    assert(N.dimension(2) == M.dimension(1));
    // assert(std::min(M.dimension(0) * M.dimension(2), N.dimension(0) * N.dimension(1)) >= M.dimension(1));

    auto N_       = Eigen::Tensor<T, 3>(N.shuffle(shf));
    auto M_       = Eigen::Tensor<T, 3>(M.shuffle(shf));
    auto P1_      = Eigen::Tensor<T, 3>(P1.shuffle(shf));
    auto P2_      = Eigen::Tensor<T, 3>(P2.shuffle(shf));
    auto [MP, N0] = get_mixing_terms_MP_N0(M_, N_, P1_, P2_, cfg);
    return {N0.shuffle(shf), MP.shuffle(shf)};
}
