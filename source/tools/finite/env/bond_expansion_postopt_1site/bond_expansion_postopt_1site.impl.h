#pragma once

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
#include <Eigen/Eigenvalues>
#include <source_location>

namespace settings {
    inline constexpr bool debug_rexpansion_postopt = false;
}

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::rexpand_bond_postopt_1site(StateFinite<Scalar> &state, ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges,
                                                                           BondExpansionConfig bcfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_postopt_1site: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(!has_flag(bcfg.policy, BondExpansionPolicy::POSTOPT_1SITE))
        throw except::logic_error("expand_bond_postopt_1site: bcfg.policy must have BondExpansionPolicy::POSTOPT set");

    // POSTOPT enriches the current site and zero-pads the upcoming site.
    // Case list
    // (a)     [ML, P] [MR 0]^T : postopt_rear (AC,B) -->
    // (b)     [ML, 0] [MR P]^T : postopt_rear (AC,B) <--
    // using R = decltype(std::real(std::declval<Scalar>()));

    std::vector<size_t> pos_expanded;
    auto                pos = state.template get_position<size_t>();
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
        res.msg =
            fmt::format("The bond [{}-{}] has reached its upper limit {} | mode {}", mpsL.get_tag(), mpsR.get_tag(), mpsL.get_chiR(), flag2str(bcfg.policy));
        return res; // No update
    }

    // Determine the maximum bond size
    // Bond dimension can't grow faster than x spin_dim.
    if(bcfg.bond_lim < 1) throw except::logic_error("Invalid bondexp_bondlim: {} < 1", bcfg.bond_lim);
    auto bondL_max = mpsL.spin_dim() * mpsL.get_chiL();
    auto bondR_max = mpsR.spin_dim() * mpsR.get_chiR();
    auto bond_lim  = std::min<Eigen::Index>({bondL_max, bondR_max, bcfg.bond_lim});
    if(bond_lim == mpsL.get_chiR()) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("The bond [{}-{}] has reached bond_max {} | mode {}", mpsL.get_tag(), mpsR.get_tag(), bond_lim, flag2str(bcfg.policy));
        return res; // No update
    }

    assert(mpsL.get_chiR() == mpsR.get_chiL());

    size_t posP = state.get_direction() > 0 ? posL : posR;
    size_t pos0 = state.get_direction() > 0 ? posR : posL;
    auto  &mpsP = state.get_mps_site(posP);
    auto  &mps0 = state.get_mps_site(pos0);

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

    tools::log->debug("Expanding {}{} - {}{} | ene {:.8f} var {:.5e} | χmax {}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(), mpsR.dimensions(),
                      fp(res.ene_old), fp(res.var_old), bond_lim);

    bool use_P1 = has_flag(bcfg.policy, BondExpansionPolicy::H1);
    bool use_P2 = has_flag(bcfg.policy, BondExpansionPolicy::H2);

    const auto    &mpoP  = model.get_mpo(posP);
    const auto    &envP1 = state.get_direction() > 0 ? edges.get_env_eneL(posP) : edges.get_env_eneR(posP);
    const auto    &envP2 = state.get_direction() > 0 ? edges.get_env_varL(posP) : edges.get_env_varR(posP);
    const auto     P1    = use_P1 ? envP1.template get_expansion_term<Scalar>(mpsP, mpoP) : Eigen::Tensor<Scalar, 3>();
    const auto     P2    = use_P2 ? envP2.template get_expansion_term<Scalar>(mpsP, mpoP) : Eigen::Tensor<Scalar, 3>();
    decltype(auto) M     = mpsP.template get_M_bare_as<Scalar>();
    decltype(auto) N     = mps0.template get_M_bare_as<Scalar>();
    if(state.get_direction() > 0) {
        // [M Λc, N] are [A(i)Λc, B(i+1)]
        assert_orthonormal<2>(M); // A(i) should be left-orthonormal
        assert_orthonormal<1>(N); // B(i+1) should be right-orthonormal

        auto [M_P, N_0] = internal::get_expansion_terms_MP_N0(M, N, P1, P2, res, bond_lim, Scalar{0});
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        internal::merge_rexpansion_terms_MP_N0(mpsP, M_P, mps0, N_0, bond_lim, Scalar{1e-4f});
        tools::log->debug("Bond expansion l2r {} | {} | χmax {} | χ {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy), bond_lim, dimP_old, M.dimensions(),
                          M_P.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(posP));

    } else {
        // [N Λc, M] are now [A(i)Λc, B(i+1)]
        assert_orthonormal<2>(N); // A(i) should be left-orthonormal
        assert_orthonormal<1>(M); // B(i+1) should be right-orthonormal

        auto [N_0, M_P] = internal::get_expansion_terms_N0_MP(N, M, P1, P2, res, bond_lim, Scalar{0});
        res.dimMP       = M_P.dimensions();
        res.dimN0       = N_0.dimensions();
        internal::merge_rexpansion_terms_N0_MP(mps0, N_0, mpsP, M_P, bond_lim, Scalar{1e-4f});
        tools::log->debug("Bond expansion r2l {} | {} | χmax {} | χ {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy), bond_lim, dimP_old, M.dimensions(),
                          M_P.dimensions());
        assert(state.template get_position<long>() == static_cast<long>(pos0));
    }

    if(mpsP.dimensions()[0] * std::min(mpsP.dimensions()[1], mpsP.dimensions()[2]) < std::max(mpsP.dimensions()[1], mpsP.dimensions()[2])) {
        tools::log->warn("Bond expansion failed: {} -> {}", dimP_old, mpsP.dimensions());
    }

    if(dimL_old[1] != mpsL.get_chiL()) throw except::runtime_error("mpsL changed chiL during bond expansion: {} -> {}", dimL_old, mpsL.dimensions());
    if(dimR_old[2] != mpsR.get_chiR()) throw except::runtime_error("mpsR changed chiR during bond expansion: {} -> {}", dimR_old, mpsR.dimensions());
    if constexpr(settings::debug_rexpansion_postopt) mpsL.assert_normalized();
    if constexpr(settings::debug_rexpansion_postopt) mpsR.assert_normalized();
    state.clear_cache();
    state.clear_measurements();

    env::rebuild_edges(state, model, edges);

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    if(mpsL.get_chiR() > bond_lim) {
        throw except::logic_error("rexpand_bond_postopt_1site: {}{} - {}{} | bond {} > max bond{}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                                  mpsR.dimensions(), mpsL.get_chiR(), bond_lim);
    }
    if(mpsR.get_chiL() > bond_lim) {
        throw except::logic_error("rexpand_bond_postopt_1site: {}{} - {}{} | bond {} > max bond{}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                                  mpsR.dimensions(), mpsL.get_chiR(), bond_lim);
    }
    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.dimMP    = mpsP.dimensions();
    res.dimN0    = mps0.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}