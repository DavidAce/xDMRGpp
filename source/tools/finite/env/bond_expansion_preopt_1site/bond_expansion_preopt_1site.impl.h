#include "../../env.h"
#include "../assertions.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "../orthonormalize_dgks.h"
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

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_rexpansion_terms_preopt_r2l(const Eigen::Tensor<T, 3>                     &M,   // Gets expanded
                                                                                    const Eigen::Tensor<T, 3>                     &N,   // Gets padded
                                                                                    const Eigen::Tensor<T, 3>                     &P1,  //
                                                                                    const Eigen::Tensor<T, 3>                     &P2,  //
                                                                                    [[maybe_unused]] const BondExpansionResult<T> &res, //
                                                                                    [[maybe_unused]] const Eigen::Index            bond_max) {
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
    using MatrixT    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = typename MatrixT::RealScalar;
    using VectorR    = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    assert_orthonormal<2>(M); // "
    assert_orthonormal<1>(N);

    auto max_cols_keep = std::min<Eigen::Index>({M.dimension(0) * M.dimension(1), N.dimension(0) * N.dimension(2)});
    max_cols_keep      = std::min(max_cols_keep, bond_max);
    if(max_cols_keep <= M.dimension(2)) {
        // We can't add more columns beyond the ones that are already fixed
        return {M, N};
    }

    // Now let's start calculating residuals.
    const auto M_matrix = tenx::MatrixMap(M, M.dimension(0) * M.dimension(1), M.dimension(2));
    MatrixT    P_matrix;
    if(P1.size() > 0) {
        P_matrix              = tenx::MatrixMap(P1, P1.dimension(0) * P1.dimension(1), P1.dimension(2));
        RealScalar maxcolnorm = P_matrix.colwise().norm().maxCoeff();
        ;
        P_matrix /= maxcolnorm;
    }
    if(P2.size() > 0) {
        P_matrix.conservativeResize(M_matrix.rows(), P_matrix.cols() + P2.dimension(2));
        P_matrix.rightCols(P2.dimension(2)) = tenx::MatrixMap(P2, P2.dimension(0) * P2.dimension(1), P2.dimension(2));
        RealScalar maxcolnorm               = P_matrix.rightCols(P2.dimension(2)).colwise().norm().maxCoeff();
        P_matrix.rightCols(P2.dimension(2)) /= maxcolnorm;
    }
    VectorR P_norms = P_matrix.colwise().norm();
    // tools::log->info("P norms: {::.3e}", fv(P_norms));
    orthonormalize_dgks(M_matrix, P_matrix);
    RealScalar maxPnorm = std::max<RealScalar>(RealScalar{1}, P_matrix.colwise().norm().maxCoeff());

    Eigen::ColPivHouseholderQR<MatrixT> cpqr(P_matrix);
    auto                                max_cols_keep_P = std::min<Eigen::Index>(cpqr.rank(), std::max<Eigen::Index>(0, max_cols_keep - M.dimension(2)));
    P_matrix                                            = cpqr.householderQ().setLength(cpqr.rank()) * MatrixT::Identity(P_matrix.rows(), max_cols_keep_P);

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

    assert_allfinite(M_P_matrix);
    assert_orthonormal<2>(M_P, std::numeric_limits<RealScalar>::epsilon() * maxPnorm);

    return {M_P, N_0};
}

template<typename T>
std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_rexpansion_terms_preopt_l2r(const Eigen::Tensor<T, 3> &N, // Gets padded
                                                                                    const Eigen::Tensor<T, 3> &M, // Gets expanded
                                                                                    const Eigen::Tensor<T, 3> &P1, const Eigen::Tensor<T, 3> &P2,
                                                                                    const BondExpansionResult<T> &res, const Eigen::Index bond_max) {
    constexpr auto shf = std::array<long, 3>{0, 2, 1};
    assert(N.dimension(2) == M.dimension(1));
    assert_orthonormal<2>(N); // N is an "A"
    assert_orthonormal<1>(M); // M is a "B"

    auto N_           = Eigen::Tensor<T, 3>(N.shuffle(shf));
    auto M_           = Eigen::Tensor<T, 3>(M.shuffle(shf));
    auto P1_          = Eigen::Tensor<T, 3>(P1.shuffle(shf));
    auto P2_          = Eigen::Tensor<T, 3>(P2.shuffle(shf));
    auto [M_P_, N_0_] = get_rexpansion_terms_preopt_r2l(M_, N_, P1_, P2_, res, bond_max);
    return {N_0_.shuffle(shf), M_P_.shuffle(shf)};
}

template<typename Scalar>
void merge_rexpansion_terms_preopt_l2r(MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &N_0, MpsSite<Scalar> &mpsR, const Eigen::Tensor<Scalar, 3> &M_P) {
    // The expanded bond sits between mpsL and mpsR.
    // During preopt expansion -->
    //      * mpsL is A(i)Λc
    //      * mpsR is B(i+1)
    //      * N_0 is [A(i), 0]
    //      * M_P is [B(i+1), P]^T

    tools::log->trace("merge_rexpansion_terms_preopt_l2r: ({}{},{}{})", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions());

    // Create a padded LC

    Eigen::Tensor<Scalar, 1> LC_pad(N_0.dimension(2));
    LC_pad.setConstant(Scalar{0});

    auto LC_off                  = tenx::array1{0};
    auto LC_ext                  = tenx::array1{mpsL.get_LC().size()};
    LC_pad.slice(LC_off, LC_ext) = mpsL.get_LC();

    mpsL.set_M(N_0);
    mpsL.set_LC(LC_pad);

    mpsR.set_M(M_P);
}

template<typename Scalar>
void merge_rexpansion_terms_preopt_r2l(MpsSite<Scalar> &mpsL, const Eigen::Tensor<Scalar, 3> &M_P, MpsSite<Scalar> &mpsR, const Eigen::Tensor<Scalar, 3> &N_0) {
    // The expanded bond sits between mpsL and mpsR.
    // During preopt expansion <--
    //      * mpsL is A(i)Λc
    //      * mpsR is B(i+1)
    //      * M_P is [A(i), P]
    //      * N_0 is [B, 0]^T

    tools::log->trace("merge_rexpansion_terms_preopt_r2l: ({}{},{}{})", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions());
    svd::solver svd;

    // Create a longer padded LC
    Eigen::Tensor<Scalar, 1> LC_pad(N_0.dimension(1));
    LC_pad.setConstant(Scalar{0});
    auto LC_off                  = tenx::array1{0};
    auto LC_ext                  = tenx::array1{mpsL.get_LC().size()};
    LC_pad.slice(LC_off, LC_ext) = mpsL.get_LC();
    mpsL.set_M(M_P);
    mpsL.set_LC(LC_pad, -1.0);

    mpsR.set_M(N_0);
}

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::rexpand_bond_preopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges,
                                                                          BondExpansionConfig bcfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_postopt_1site: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(!has_flag(bcfg.policy, BondExpansionPolicy::PREOPT_1SITE))
        throw except::logic_error("expand_bond_postopt_1site: bcfg.policy must have BondExpansionPolicy::PREOPT_1SITE set");

    // PREOPT enriches the forward site and zero-pads the current site.
    // Case list
    // (a)     --> : [AC,B]  becomes  [AC, 0] [B P]^T
    // (b)     <-- : [AC,B]  becomes  [AC, P] [B 0]^T
    // where C gets zero-padded
    std::vector<size_t> pos_expanded;
    auto                pos = state.template get_position<size_t>();
    // if(state.get_direction() > 0 and pos == std::clamp<size_t>(pos, 0, state.template get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};
    // if(state.get_direction() < 0 and pos == std::clamp<size_t>(pos, 1, state.template get_length<size_t>() - 1)) pos_expanded = {pos - 1, pos};
    if(pos == std::clamp<size_t>(pos, 0, state.template get_length<size_t>() - 2)) pos_expanded = {pos, pos + 1};

    if(pos_expanded.empty()) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("No positions to expand: mode {}", flag2str(bcfg.policy));
        return res; // No update
    }

    size_t posL = pos_expanded.front();
    size_t posR = pos_expanded.back();
    auto  &mpsL = state.get_mps_site(posL);
    auto  &mpsR = state.get_mps_site(posR);
    if(state.num_bonds_at_maximum(pos_expanded) == 1) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("The bond upper limit has been reached for site pair [{}-{}] | mode {}", mpsL.get_tag(), mpsR.get_tag(), flag2str(bcfg.policy));
        return res; // No update
    }

    assert(mpsL.get_chiR() == mpsR.get_chiL());

    size_t posP = state.get_direction() > 0 ? posR : posL;
    size_t pos0 = state.get_direction() > 0 ? posL : posR;
    auto  &mpsP = state.get_mps_site(posP);
    auto  &mps0 = state.get_mps_site(pos0);

    // Check that the enriched site is active, otherwise measurements will be off
    if(state.get_direction() > 0) assert(pos0 == state.active_sites.front());
    if(state.get_direction() < 0) assert(pos0 == state.active_sites.back());

    auto dimL_old = mpsL.dimensions();
    auto dimR_old = mpsR.dimensions();
    auto dimP_old = mpsP.dimensions();

    assert_edges_ene(state, model, edges);
    assert_edges_var(state, model, edges);
    auto res      = BondExpansionResult<Scalar>();
    res.direction = state.get_direction();
    res.sites     = pos_expanded;
    res.dims_old  = state.get_mps_dims(pos_expanded);
    res.bond_old  = state.get_bond_dims(pos_expanded);
    res.posL      = safe_cast<long>(pos_expanded.front());
    res.posR      = safe_cast<long>(pos_expanded.back());
    res.dimL_old  = mpsL.dimensions();
    res.dimR_old  = mpsR.dimensions();
    res.ene_old   = tools::finite::measure::energy(state, model, edges);
    res.var_old   = tools::finite::measure::energy_variance(state, model, edges);

    // Set up the SVD
    // Bond dimension can't grow faster than x spin_dim.
    if(bcfg.bondlim < 1) throw except::logic_error("Invalid bondexp_bondlim: {} < 1", bcfg.bondlim);
    auto bondL_max = mpsL.spin_dim() * mpsL.get_chiL();
    auto bondR_max = mpsR.spin_dim() * mpsR.get_chiR();
    auto bond_max  = std::min<Eigen::Index>({bondL_max, bondR_max, bcfg.bondlim});

    tools::log->debug("Expanding {}{} - {}{} | ene {:.8f} var {:.5e} | χmax {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(),
                      res.ene_old, res.var_old, bond_max);

    const auto    &mpoP  = model.get_mpo(posP);
    const auto    &envP1 = state.get_direction() > 0 ? edges.get_env_eneR(posP) : edges.get_env_eneL(posP);
    const auto    &envP2 = state.get_direction() > 0 ? edges.get_env_varR(posP) : edges.get_env_varL(posP);
    const auto     P1    = res.alpha_h1v == 0 ? Eigen::Tensor<Scalar, 3>() : envP1.template get_expansion_term<Scalar>(mpsP, mpoP);
    const auto     P2    = res.alpha_h2v == 0 ? Eigen::Tensor<Scalar, 3>() : envP2.template get_expansion_term<Scalar>(mpsP, mpoP);
    decltype(auto) M     = mpsP.template get_M_bare_as<Scalar>();
    decltype(auto) N     = mps0.template get_M_bare_as<Scalar>();
    if(state.get_direction() > 0) {
        // [N Λc, M] are [A(i)Λc, B(i+1)]
        assert_orthonormal<2>(N); // A(i) should be left-orthonormal
        assert_orthonormal<1>(M); // B(i+1) should be right-orthonormal

        auto [N_0, M_P] = get_rexpansion_terms_preopt_l2r(N, M, P1, P2, res, bond_max);
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        merge_rexpansion_terms_preopt_l2r(mps0, N_0, mpsP, M_P);
        tools::log->debug("Bond expansion preopt l2r {} | {} | χmax {} | χ {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy), bond_max, dimP_old,
                          M.dimensions(), M_P.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(pos0));

    } else {
        // [M Λc, N] are now [A(i)Λc, B(i+1)]
        assert_orthonormal<2>(M); // A(i) should be left-orthonormal
        assert_orthonormal<1>(N); // B(i+1) should be right-orthonormal

        auto [M_P, N_0] = get_rexpansion_terms_preopt_r2l(M, N, P1, P2, res, bond_max);
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        merge_rexpansion_terms_preopt_r2l(mpsP, M_P, mps0, N_0);
        tools::log->debug("Bond expansion preopt r2l {} | {} | χmax {} | χ {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy), bond_max, dimP_old,
                          M.dimensions(), M_P.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(posP));
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
    if(mpsL.get_chiR() > bond_max) {
        throw except::logic_error("rexpand_bond_postopt_1site: {}{} - {}{} | bond {} > max bond{}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                                  mpsR.dimensions(), mpsL.get_chiR(), bond_max);
    }
    if(mpsR.get_chiL() > bond_max) {
        throw except::logic_error("rexpand_bond_postopt_1site: {}{} - {}{} | bond {} > max bond{}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                                  mpsR.dimensions(), mpsL.get_chiR(), bond_max);
    }
    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}