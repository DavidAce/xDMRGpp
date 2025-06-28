#include "../../env.h"
#include "../assertions.h"
#include "../BondExpansionConfig.h"
#include "../BondExpansionResult.h"
#include "../mixing_terms.h"
#include "config/debug.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "math/num.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/residual.h"
namespace settings {
    inline constexpr bool debug_dmrg3s = false;
}

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::env::expand_bond_dmrg3s(StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, EdgesFinite<Scalar> &edges,
                                                                   BondExpansionConfig bcfg) {
    if(not num::all_equal(state.get_length(), model.get_length(), edges.get_length()))
        throw except::runtime_error("expand_bond_dmrg3s: All lengths not equal: state {} | model {} | edges {}", state.get_length(), model.get_length(),
                                    edges.get_length());
    if(not num::all_equal(state.active_sites, model.active_sites, edges.active_sites))
        throw except::runtime_error("expand_bond_dmrg3s: All active sites are not equal: state {} | model {} | edges {}", state.active_sites,
                                    model.active_sites, edges.active_sites);
    if(state.active_sites.empty()) throw except::logic_error("No active sites for bond expansion");

    if(!has_flag(bcfg.policy, BondExpansionPolicy::DMRG3S))
        throw except::logic_error("expand_bond_dmrg3s: bcfg.policy must have BondExpansionPolicy::DMRG3S set");

    if(bcfg.mixing_factor == 0) {
        auto res = BondExpansionResult<Scalar>();
        res.msg  = fmt::format("Expansion canceled because α = {:.2e} ", fp(bcfg.mixing_factor));
        return res;
    }
    tools::finite::env::assert_edges_ene(state, model, edges);
    tools::finite::env::assert_edges_var(state, model, edges);

    // DMRG3S enriches the current site and zero-pads the upcoming site after optimization, before moving.
    // This method adds noise to the bond when expanding. Therefore, we rarely benefit from this method once
    // the bond dimension has already grown to its theoretical maximum: If bonds are numbered l=0,1,2...L
    // the maximum bond is d^min(l,L-l), where d is the spin dimension
    // Case list
    // (a)  --> (AC,B) becomes [M, P] [N 0]^T
    // (b)  <-- (AC,B) becomes [N, 0] [M P]^T

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
        res.msg  = fmt::format("The bond upper limit has been reached for site pair [{}-{}] | mode {}", mpsL.get_tag(), mpsR.get_tag(), flag2str(bcfg.policy));
        return res; // No update
    }

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    // assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    size_t posP = state.get_direction() > 0 ? posL : posR;
    size_t pos0 = state.get_direction() > 0 ? posR : posL;
    auto  &mpsP = state.get_mps_site(posP);
    auto  &mps0 = state.get_mps_site(pos0);

    auto dimL_old = mpsL.dimensions();
    auto dimR_old = mpsR.dimensions();
    auto dimP_old = mpsP.dimensions();

    // auto res = get_mixing_factors_postopt_rnorm(pos_expanded, state, model, edges, bcfg);
    // using R = decltype(std::real(std::declval<Scalar>()));

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

    tools::log->debug("Expanding {}{} - {}{} | α = {:.5e} | factor {:.1e} | ene {:.16f} var {:.5e}", mpsL.get_tag(), mpsL.dimensions(), mpsR.get_tag(),
                      mpsR.dimensions(), fp(bcfg.mixing_factor), bcfg.bond_factor, fp(res.ene_old), fp(res.var_old));

    // Set up the SVD
    // Bond dimension can't grow faster than x spin_dim.
    auto bondL_max = mpsL.spin_dim() * mpsL.get_chiL();
    auto bondR_max = mpsR.spin_dim() * mpsR.get_chiR();
    auto bond_max  = std::min<Eigen::Index>({bondL_max, bondR_max, bcfg.bond_lim});

    auto svd_cfg = svd::config(bond_max, bcfg.trnc_lim);

    bool use_P1 = has_flag(bcfg.policy, BondExpansionPolicy::H1);
    bool use_P2 = has_flag(bcfg.policy, BondExpansionPolicy::H2);

    decltype(auto) M     = state.template get_multisite_mps<Scalar>({posP});
    decltype(auto) N     = mps0.template get_M_bare_as<Scalar>();
    const auto    &mpoP  = model.get_mpo(posP);
    const auto    &envP1 = state.get_direction() > 0 ? edges.get_env_eneL(posP) : edges.get_env_eneR(posP);
    const auto    &envP2 = state.get_direction() > 0 ? edges.get_env_varL(posP) : edges.get_env_varR(posP);
    const auto     P1    = use_P1 ? envP1.template get_expansion_term<Scalar>(M, mpoP) : Eigen::Tensor<Scalar, 3>();
    const auto     P2    = use_P2 ? envP2.template get_expansion_term<Scalar>(M, mpoP) : Eigen::Tensor<Scalar, 3>();

    if(state.get_direction() > 0) {
        auto [M_P_del, N_0] = internal::get_mixing_terms_MP_N0(M, N, P1, P2, bcfg);
        internal::merge_mixing_terms_MP_N0(state, mpsP, M_P_del, mps0, N_0, svd_cfg);
        tools::log->debug("Bond expansion l2r {} | {} α {:.4e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy),
                          fp(bcfg.mixing_factor), svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimP_old, M.dimensions(), res.dimMP,
                          mpsP.dimensions());
    } else {
        auto [N_0, M_P_del] = internal::get_mixing_terms_N0_MP(N, M, P1, P2, bcfg);
        internal::merge_mixing_terms_N0_MP(state, mps0, N_0, mpsP, M_P_del, svd_cfg);
        tools::log->debug("Bond expansion r2l {} | {} α {:.4e} | svd_ε {:.2e} | χlim {} | χ {} -> {} -> {} -> {}", pos_expanded, flag2str(bcfg.policy),
                          fp(bcfg.mixing_factor), svd_cfg.truncation_limit.value(), svd_cfg.rank_max.value(), dimP_old, M.dimensions(), M_P_del.dimensions(),
                          mpsP.dimensions());
    }

    if(mpsP.dimensions()[0] * std::min(mpsP.dimensions()[1], mpsP.dimensions()[2]) < std::max(mpsP.dimensions()[1], mpsP.dimensions()[2])) {
        tools::log->warn("Bond expansion failed: {} -> {}", dimP_old, mpsP.dimensions());
    }

    if(dimL_old[1] != mpsL.get_chiL()) throw except::runtime_error("mpsL changed chiL during bond expansion: {} -> {}", dimL_old, mpsL.dimensions());
    if(dimR_old[2] != mpsR.get_chiR()) throw except::runtime_error("mpsR changed chiR during bond expansion: {} -> {}", dimR_old, mpsR.dimensions());
    if constexpr(settings::debug_dmrg3s) mpsL.assert_normalized();
    if constexpr(settings::debug_dmrg3s) mpsR.assert_normalized();
    state.clear_cache();
    state.clear_measurements();
    env::rebuild_edges(state, model, edges);

    assert(mpsL.get_chiR() == mpsR.get_chiL());
    assert(std::min(mpsL.spin_dim() * mpsL.get_chiL(), mpsR.spin_dim() * mpsR.get_chiR()) >= mpsL.get_chiR());

    res.dimL_new = mpsL.dimensions();
    res.dimR_new = mpsR.dimensions();
    res.dimMP    = mpsP.dimensions();
    res.dimN0    = mps0.dimensions();
    res.ene_new  = tools::finite::measure::energy(state, model, edges);
    res.var_new  = tools::finite::measure::energy_variance(state, model, edges);
    res.ok       = true;
    return res;
}
