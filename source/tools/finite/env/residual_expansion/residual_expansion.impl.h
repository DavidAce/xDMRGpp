#include "../../env.h"
#include "../BondExpansionResult.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/linalg/tensor/to_string.h"
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
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/residual.h"
#include "tools/finite/mps.h"
#include "tools/finite/opt_meta.h"
#include <Eigen/Eigenvalues>
#include <source_location>

namespace settings {
    inline constexpr bool debug_rexpansion = false;
}

template<typename Derived>
void assert_allfinite(const Eigen::MatrixBase<Derived> &X, const std::source_location &location = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        bool allFinite = X.allFinite();
        if(!allFinite) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X is not all finite: \n{}\n", linalg::matrix::to_string(X, 8));
            throw except::runtime_error("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name());
        }
    }
}

template<typename Derived>
void assert_orthonormal_cols(const Eigen::MatrixBase<Derived> &X,
                             typename Derived::RealScalar      threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon() * 10000,
                             const std::source_location       &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        using Scalar         = typename Derived::Scalar;
        using MatrixType     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        MatrixType XX        = X.adjoint() * X;
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            tools::log->warn("X orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix is not orthonormal: error = {:.5e} > threshold = {:.5e}", location.file_name(), location.line(),
                                        location.function_name(), orthError, threshold);
        }
    }
}

template<typename Derived>
void assert_orthonormal_rows(const Eigen::MatrixBase<Derived> &X,
                             typename Derived::RealScalar      threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon() * 10000,
                             const std::source_location       &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        if(X.cols() == 0) return;
        using Scalar         = typename Derived::Scalar;
        using MatrixType     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        MatrixType XX        = X * X.adjoint();
        auto       orthError = (XX - MatrixType::Identity(XX.cols(), XX.rows())).norm();
        if(orthError > threshold) {
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(X, 8));
            tools::log->warn("X.adjoint()*X: \n{}\n", linalg::matrix::to_string(XX, 8));
            tools::log->warn("X orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix has non-orthonormal rows: error = {:.5e} > threshold = {:.5e}", location.file_name(),
                                        location.line(), location.function_name(), orthError, threshold);
        }
    }
}

template<Eigen::Index axis, typename Derived>
void assert_orthonormal(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors> &expr,
                        typename Derived::RealScalar threshold = std::numeric_limits<typename Derived::RealScalar>::epsilon() * 10000,
                        const std::source_location  &location  = std::source_location::current()) {
    if constexpr(settings::debug) {
        static_assert(axis == 1 or axis == 2);
        auto tensor         = tenx::asEval(expr);
        using DimType       = typename decltype(tensor)::Dimensions;
        constexpr auto rank = Eigen::internal::array_size<DimType>::value;
        static_assert(rank == 3 and "assert_orthonormal: expression must be a tensor of rank 3");

        using Scalar     = typename decltype(tensor)::Scalar;
        using TensorType = Eigen::Tensor<Scalar, rank>;
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using MapType    = Eigen::Map<const MatrixType>;

        auto X = Eigen::TensorMap<const TensorType>(tensor.data(), tensor.dimensions());
        if(X.size() == 0) return;

        Eigen::Tensor<Scalar, 2> G;
        Eigen::Index             oxis    = axis == 1l ? oxis = 2l : 1l; // other axis
        auto                     idxpair = tenx::idx({0l, oxis}, {0l, oxis});
        G                                = X.conjugate().contract(X, idxpair); // Gram matrix
        auto Gm                          = MapType(G.data(), G.dimension(0), G.dimension(1));
        auto orthError                   = (Gm - MatrixType::Identity(Gm.cols(), Gm.rows())).norm();
        if(orthError > threshold) {
            auto Xm = MapType(X.data(), X.dimension(0) * X.dimension(oxis), X.dimension(axis));
            tools::log->warn("X: \n{}\n", linalg::matrix::to_string(Xm, 8));
            tools::log->warn("G: \n{}\n", linalg::matrix::to_string(Gm, 8));
            tools::log->warn("orthError: {:.5e}", orthError);
            throw except::runtime_error("{}:{}: {}: matrix is non-orthonormal along axis [{}]: error = {:.5e} > threshold = {:.5e}", location.file_name(),
                                        location.line(), location.function_name(), axis, orthError, threshold);
        }
    }
}

template<typename MatrixTypeX, typename MatrixTypeY>
void dgks(const MatrixTypeX &X, MatrixTypeY &Y) {
    assert(X.rows() == Y.rows());
    if(X.cols() == 0 || Y.cols() == 0) return;
    static_assert(std::is_same_v<typename MatrixTypeX::Scalar, typename MatrixTypeY::Scalar>, "MatrixTypeX and MatrixTypeY must have the same scalar type");
    using RealScalar = typename MatrixTypeX::RealScalar;
    assert_allfinite(X);
    assert_allfinite(Y);
    // DGKS clean Y against X and orthonormalize Y
    // We do not assume that X or Y are normalized!
    for(int rep = 0; rep < 2; ++rep) {
        for(Eigen::Index col_y = 0; col_y < Y.cols(); ++col_y) {
            auto Ycol = Y.col(col_y);
            for(Eigen::Index col_x = 0; col_x < X.cols(); ++col_x) {
                auto Xcol = X.col(col_x);
                auto Xsqn = Xcol.squaredNorm();
                if(Xsqn < RealScalar{5e-8f}) continue;
                Ycol.noalias() -= Xcol * (Xcol.adjoint() * Ycol).eval() / Xcol.squaredNorm();
            }
            for(Eigen::Index col_prev_y = 0; col_prev_y < col_y; ++col_prev_y) {
                auto Ycol_prev = Y.col(col_prev_y);
                auto Ysqn_prev = Ycol_prev.squaredNorm();
                if(Ysqn_prev < RealScalar{5e-8f}) continue;
                Ycol.noalias() -= Ycol_prev * (Ycol_prev.adjoint() * Ycol).eval() / Ycol_prev.squaredNorm();
            }
        }
    }
    assert_allfinite(X);
    assert_allfinite(Y);
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_rexpansion_terms_l2r(const Eigen::Tensor<T, 3>                     &M,   // Gets expanded
                                                                             const Eigen::Tensor<T, 3>                     &N,   // Gets padded
                                                                             const Eigen::Tensor<T, 3>                     &P1,  //
                                                                             const Eigen::Tensor<T, 3>                     &P2,  //
                                                                             [[maybe_unused]] const BondExpansionResult<T> &res, //
                                                                             [[maybe_unused]] const svd::config            &svd_cfg) {
    /*
        We form M_P = [M | P] by concatenating along dimension 2.
        In matrix-language, the columns of P are added as new columns to M.
        The columns of M and P are normalized (M is bare).

        We form P by taking P1=H1*M and P2=H2*M, where the effective operators act only to the left of the M.
        We then subtract from P1 and P2 their projections against M and each other. After stacking P = [P1 | P2],
        we sort the columns of P with a column-pivoting householder QR.
      */
    assert(M.dimension(2) == N.dimension(1));
    assert(P1.size() == 0 or P1.dimension(1) == M.dimension(1));
    assert(P1.size() == 0 or P1.dimension(0) == M.dimension(0));
    assert(P2.size() == 0 or P2.dimension(1) == M.dimension(1));
    assert(P2.size() == 0 or P2.dimension(0) == M.dimension(0));
    using R       = decltype(std::real(std::declval<T>()));
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    assert_orthonormal<2>(M); // "
    assert_orthonormal<1>(N);

    auto max_cols_keep = std::min<Eigen::Index>({M.dimension(0) * M.dimension(1), N.dimension(0) * N.dimension(2)});
    if(svd_cfg.rank_max.has_value()) { max_cols_keep = std::min(max_cols_keep, svd_cfg.rank_max.value()); }
    if(max_cols_keep <= M.dimension(2)) {
        // We can't add more columns beyond the ones that are already fixed
        return {M, N};
    }

    // Now let's start calculating residuals.
    const auto M_matrix = tenx::MatrixMap(M, M.dimension(0) * M.dimension(1), M.dimension(2));

    MatrixT P_matrix;
    if(P1.size() > 0) {
        P_matrix = tenx::MatrixMap(P1, P1.dimension(0) * P1.dimension(1), P1.dimension(2));
        dgks(M_matrix, P_matrix);
    }
    if(P2.size() > 0) {
        P_matrix.conservativeResize(M_matrix.rows(), P_matrix.cols() + P2.dimension(2));
        auto P1_matrix = P_matrix.leftCols(P1.dimension(2));
        auto P2_matrix = P_matrix.rightCols(P2.dimension(2));
        P2_matrix      = tenx::MatrixMap(P2, P2.dimension(0) * P2.dimension(1), P2.dimension(2));
        // tools::log->info("P before dgks(M,P2): \n{}\n", linalg::matrix::to_string(P_matrix, 8));
        dgks(M_matrix, P2_matrix);
        dgks(P1_matrix, P2_matrix);
    }
    // tools::log->info("P before QR: \n{}\n", linalg::matrix::to_string(P_matrix, 8));
    Eigen::ColPivHouseholderQR<MatrixT> cpqr(P_matrix);
    cpqr.setThreshold(std::numeric_limits<R>::epsilon() * 10);
    auto max_cols_keep_P = std::min<Eigen::Index>(cpqr.rank(), std::max<Eigen::Index>(0, max_cols_keep - M.dimension(2)));
    P_matrix             = cpqr.householderQ().setLength(cpqr.rank()) * MatrixT::Identity(P_matrix.rows(), max_cols_keep_P);
    // tools::log->info("P after QR (rank = {}): \n{}\n", cpqr.rank(), linalg::matrix::to_string(P_matrix, 8));

    Eigen::Index dim0 = M.dimension(0);
    Eigen::Index dim1 = M.dimension(1);
    Eigen::Index dim2 = M_matrix.cols() + P_matrix.cols();

    auto extM = std::array<long, 3>{M.dimension(0), M.dimension(1), M_matrix.cols()};
    auto extP = std::array<long, 3>{M.dimension(0), M.dimension(1), P_matrix.cols()};
    auto offM = std::array<long, 3>{0, 0, 0};
    auto offP = std::array<long, 3>{0, 0, M_matrix.cols()};

    auto M_P = Eigen::Tensor<T, 3>(dim0, dim1, dim2);

    if(extM[2] > 0) M_P.slice(offM, extM) = M;
    if(extP[2] > 0) M_P.slice(offP, extP) = tenx::TensorMap(P_matrix, extP);

    auto N_0 = Eigen::Tensor<T, 3>(N.dimension(0), M_P.dimension(2), N.dimension(2));
    N_0.setZero();
    auto extN_0                              = std::array<long, 3>{N.dimension(0), std::min(N.dimension(1), M_P.dimension(2)), N.dimension(2)};
    N_0.slice(tenx::array3{0, 0, 0}, extN_0) = N.slice(tenx::array3{0, 0, 0}, extN_0); // Copy N into N_0

    // Sanity checks
    auto M_P_matrix = tenx::MatrixMap(M_P, M_P.dimension(0) * M_P.dimension(1), M_P.dimension(2));
    // tools::log->info("M: \n{}\n", linalg::matrix::to_string(M_matrix, 8));
    // tools::log->info("M_P: \n{}\n", linalg::matrix::to_string(M_P_matrix, 8));
    assert_allfinite(M_P_matrix);
    assert_orthonormal<2>(M_P);

    return {M_P, N_0};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_rexpansion_terms_r2l(const Eigen::Tensor<T, 3> &N, // Gets padded
                                                                             const Eigen::Tensor<T, 3> &M, // Gets expanded
                                                                             const Eigen::Tensor<T, 3> &P1, const Eigen::Tensor<T, 3> &P2,
                                                                             const BondExpansionResult<T> &res, const svd::config &svd_cfg) {
    constexpr auto shf = std::array<long, 3>{0, 2, 1};
    assert(N.dimension(2) == M.dimension(1));
    using R = decltype(std::real(std::declval<T>()));
    assert_orthonormal<2>(N); // N is an "A"
    assert_orthonormal<1>(M); // M is a "B"

    auto N_           = Eigen::Tensor<T, 3>(N.shuffle(shf));
    auto M_           = Eigen::Tensor<T, 3>(M.shuffle(shf));
    auto P1_          = Eigen::Tensor<T, 3>(P1.shuffle(shf));
    auto P2_          = Eigen::Tensor<T, 3>(P2.shuffle(shf));
    auto [M_P_, N_0_] = get_rexpansion_terms_l2r(M_, N_, P1_, P2_, res, svd_cfg);
    return {N_0_.shuffle(shf), M_P_.shuffle(shf)};
}

template<typename Scalar>
void merge_rexpansion_terms_r2l(const StateFinite<Scalar> &state, MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &N_0, MpsSite<Scalar> &mpsR,
                                const Eigen::Tensor<Scalar, 3> &M_P, const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR.
    // During postopt expansion <--
    //      * mpsL is A(i-1) transformed into A(i-1)Λc
    //      * mpsR is A(i), Λc transformed into B(i)
    //      * N_0 is [A(i-1), 0]
    //      * M_P is [B(i), P]^T

    tools::log->trace("merge_rexpansion_terms_r2l: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());

    // Create a padded L

    Eigen::Tensor<Scalar, 1> LC_pad(N_0.dimension(2));
    LC_pad.setConstant(Scalar{1});

    auto LC_off                  = tenx::array1{0};
    auto LC_ext                  = tenx::array1{mpsL.get_LC().size()};
    LC_pad.slice(LC_off, LC_ext) = mpsL.get_LC();

    mpsL.set_M(N_0);
    mpsL.set_LC(LC_pad);

    mpsR.set_M(M_P);

    state.clear_cache();
    state.clear_measurements();
}

template<typename Scalar>
void merge_rexpansion_terms_l2r(const StateFinite<Scalar> &state, MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &M_P, MpsSite<Scalar> &mpsR,
                                const Eigen::Tensor<Scalar, 3> &N_0, const svd::config &svd_cfg) {
    // The expanded bond sits between mpsL and mpsR.
    // During postopt expansion -->
    //      * mpsL is A(i)Λc
    //      * mpsR is B(i+1)
    //      * M_P is [A(i), P]
    //      * N_0 is [B, 0]^T

    tools::log->trace("merge_rexpansion_terms_l2r: ({}{},{}{}) {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(), svd_cfg.to_string());
    svd::solver svd;

    // Create a longer padded LC
    Eigen::Tensor<Scalar, 1> LC_pad(M_P.dimension(2));
    LC_pad.setConstant(Scalar{1});
    auto LC_off                  = tenx::array1{0};
    auto LC_ext                  = tenx::array1{mpsL.get_LC().size()};
    LC_pad.slice(LC_off, LC_ext) = mpsL.get_LC();
    mpsL.set_M(M_P);
    mpsL.set_LC(LC_pad, -1.0);

    mpsR.set_M(N_0);

    state.clear_cache();
    state.clear_measurements();
}

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::rexpand_bond_postopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model,
                                                                           EdgesFinite<Scalar> &edges, const OptMeta &opt_meta) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_postopt_1site: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(!has_flag(opt_meta.bondexp_policy, BondExpansionPolicy::POSTOPT_1SITE))
        throw except::logic_error("expand_bond_postopt_1site: opt_meta.bondexp_policy must have BondExpansionPolicy::POSTOPT set");
    if(opt_meta.optExit == OptExit::NONE) throw except::logic_error("expand_bond_postopt_1site: requires opt_meta.optExit != OptExit::NONE");

    // POSTOPT enriches the current site and zero-pads the upcoming site.
    // Case list
    // (a)     [ML, P] [MR 0]^T : postopt_rear (AC,B) -->
    // (b)     [ML, 0] [MR P]^T : postopt_rear (A,AC) <--
    using R = decltype(std::real(std::declval<Scalar>()));

    std::vector<size_t> pos_expanded;
    auto                pos = state.template get_position<size_t>();
    if(state.get_direction() > 0 and pos == std::clamp<size_t>(pos, 0, state.template get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};
    if(state.get_direction() < 0 and pos == std::clamp<size_t>(pos, 1, state.template get_length<size_t>() - 1)) pos_expanded = {pos - 1, pos};

    if(pos_expanded.empty()) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("No positions to expand: mode {}", flag2str(opt_meta.bondexp_policy));
        return res; // No update
    }

    size_t posL = pos_expanded.front();
    size_t posR = pos_expanded.back();
    auto  &mpsL = state.get_mps_site(posL);
    auto  &mpsR = state.get_mps_site(posR);
    if(state.num_bonds_at_maximum(pos_expanded) == 1) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("The bond upper limit has been reached for site pair [{}-{}] | mode {}", mpsL.get_tag(), mpsR.get_tag(),
                               flag2str(opt_meta.bondexp_policy));
        return res; // No update
    }

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    // assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    size_t posP = pos;
    size_t pos0 = state.get_direction() > 0 ? posR : posL;
    auto  &mpsP = state.get_mps_site(posP);
    auto  &mps0 = state.get_mps_site(pos0);

    auto dimL_old = mpsL.dimensions();
    auto dimR_old = mpsR.dimensions();
    auto dimP_old = mpsP.dimensions();

    auto res = get_mixing_factors_postopt_rnorm(pos_expanded, state, model, edges, opt_meta);
    internal::set_mixing_factors_to_stdv_H<Scalar>(pos_expanded, state, model, edges, opt_meta, res);
    if(res.alpha_h1v == 0 and res.alpha_h2v == 0) {
        res.msg = fmt::format("Expansion canceled: {}{} - {}{} | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                              mpsR.dimensions(), res.alpha_mps, res.alpha_h1v, res.alpha_h2v);
        return res;
    }

    tools::log->debug("Expanding {}{} - {}{} | α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.2e} | factor {:.1e}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                      mpsR.dimensions(), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, opt_meta.bondexp_factor);

    // Set up the SVD
    // Bond dimension can't grow faster than x spin_dim.
    auto svd_cfg             = opt_meta.svd_cfg.value();
    auto bond_max            = std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR());
    svd_cfg.truncation_limit = svd_cfg.truncation_limit.value_or(settings::precision::svd_truncation_min);
    svd_cfg.rank_max         = std::min(bond_max, svd_cfg.rank_max.value_or(bond_max));

    // We operate on mps tensors [A, B].
    // When entering this function, the matrices may be in a different form:
    // In left-to-right (l2r) expansion:
    //        [A(i)Λc, B(i+1)] (no movement)
    //        Expand A(i), pad B(i+1)
    // In right-to-left (r2l) expansion:
    //        [A(i-1), A(i)Λc] --move--> [A(i-1)Λc, B(i)]
    //        Pad A(i-1), expand B(i)

    if(state.get_direction() < 0) {
        // Move center site
        auto        M  = mpsP.template get_M_as<Scalar>();  // Include LC by taking non-bare
        auto        LC = mpsP.template get_LC_as<Scalar>(); // Sits to the right of A(i)
        svd::solver svd;
        auto [U, S, V] = svd.schmidt_into_right_normalized(M, M.dimension(0), svd_cfg);
        mpsP.unset_LC();
        mpsP.set_mps(V, LC, -1.0, "B");
        mpsP.stash_C(S, -1.0, pos0);
        mpsP.stash_U(U, pos0);
        mps0.take_stash(mpsP);
        tools::log->debug("Moved MPS {}{} - {}{}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(pos0));
        state.active_sites = {pos0};
        model.active_sites = {pos0};
        edges.active_sites = {pos0};
    }

    const auto    &mpoP  = model.get_mpo(posP);
    const auto    &envP1 = state.get_direction() > 0 ? edges.get_env_eneL(posP) : edges.get_env_eneR(posP);
    const auto    &envP2 = state.get_direction() > 0 ? edges.get_env_varL(posP) : edges.get_env_varR(posP);
    const auto     P1    = res.alpha_h1v == 0 ? Eigen::Tensor<Scalar, 3>() : envP1.template get_expansion_term<Scalar>(mpsP, mpoP);
    const auto     P2    = res.alpha_h2v == 0 ? Eigen::Tensor<Scalar, 3>() : envP2.template get_expansion_term<Scalar>(mpsP, mpoP);
    decltype(auto) M     = mpsP.template get_M_bare_as<Scalar>();
    decltype(auto) N     = mps0.template get_M_bare_as<Scalar>();
    if(state.get_direction() > 0) {
        // [M Λc, N] are [A(i)Λc, B(i+1)]
        using R = decltype(std::real(std::declval<Scalar>()));
        assert_orthonormal<2>(M); // A should be left-orthonormal
        assert_orthonormal<1>(N); // B should be right-orthonormal

        auto [M_P, N_0] = get_rexpansion_terms_l2r(M, N, P1, P2, res, svd_cfg);
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        merge_rexpansion_terms_l2r(state, mpsP, M_P, mps0, N_0, svd_cfg);
        tools::log->debug("Bond expansion l2r {} | {} α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.3e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded,
                          flag2str(opt_meta.bondexp_policy), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(),
                          svd_cfg.rank_max.value(), dimP_old, M.dimensions(), M_P.dimensions(), mpsP.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(posP));

    } else {
        // [N Λc, M] are now [A(i-1)Λc, B(i)]
        assert_orthonormal<2>(N); // A(i-1) should be left-orthonormal
        assert_orthonormal<1>(M); // B(i) should be right-orthonormal

        auto [N_0, M_P] = get_rexpansion_terms_r2l(N, M, P1, P2, res, svd_cfg);
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        merge_rexpansion_terms_r2l(state, mps0, N_0, mpsP, M_P, svd_cfg);
        tools::log->debug("Bond expansion r2l {} | {} α₀:{:.2e} αₑ:{:.2e} αᵥ:{:.3e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded,
                          flag2str(opt_meta.bondexp_policy), res.alpha_mps, res.alpha_h1v, res.alpha_h2v, svd_cfg.truncation_limit.value(),
                          svd_cfg.rank_max.value(), dimP_old, M.dimensions(), M_P.dimensions(), mpsP.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(pos0));
    }

    if(mpsP.dimensions()[0] * std::min(mpsP.dimensions()[1], mpsP.dimensions()[2]) < std::max(mpsP.dimensions()[1], mpsP.dimensions()[2])) {
        tools::log->warn("Bond expansion failed: {} -> {}", dimP_old, mpsP.dimensions());
    }

    if(dimL_old[1] != mpsL.get_chiL()) throw except::runtime_error("mpsL changed chiL during bond expansion: {} -> {}", dimL_old, mpsL.dimensions());
    if(dimR_old[2] != mpsR.get_chiR()) throw except::runtime_error("mpsR changed chiR during bond expansion: {} -> {}", dimR_old, mpsR.dimensions());
    if constexpr(settings::debug_rexpansion) mpsL.assert_normalized();
    if constexpr(settings::debug_rexpansion) mpsR.assert_normalized();
    state.clear_cache();
    state.clear_measurements();

    env::rebuild_edges(state, model, edges);

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}