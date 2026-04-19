#pragma once

#include "config/debug.h"
#include "config/enums/BondExpansionPolicy.h"
#include "tensors/TensorsFinite.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/finite/bex.h"
#include "tools/finite/bex/BondExpansionConfig.h"
#include "tools/finite/bex/BondExpansionResult.h"
#include "tools/finite/pos.h"

template<typename Scalar>
BondExpansionResult<Scalar> tools::finite::bex::expand_bonds(TensorsFinite<Scalar> &tensors, BondExpansionConfig bcfg) {
    auto res = BondExpansionResult<Scalar>();
    if(tensors.template get_position<long>() < 0) {
        res.msg = "Negative position";
        return res;
    }
    auto t_exp = tid::tic_scope("bondexp");

    if(tensors.active_sites.empty()) tools::finite::pos::activate_sites(tensors, {tensors.template get_position<size_t>()});
    tensors.rebuild_edges(); // Use fresh edges
    if constexpr(settings::debug) tensors.assert_validity();
    if(has_flag(bcfg.policy, BondExpansionPolicy::DMRG3S) and bcfg.order == BondExpansionOrder::POSTOPT) {
        res = tools::finite::bex::expand_bond_dmrg3s(tensors.get_state(), tensors.get_model(), tensors.get_edges(), bcfg);
    } else if(has_flag(bcfg.policy, BondExpansionPolicy::POSTOPT_1SITE) and bcfg.order == BondExpansionOrder::POSTOPT) {
        res = tools::finite::bex::rexpand_bond_postopt_1site(tensors.get_state(), tensors.get_model(), tensors.get_edges(), bcfg);
    } else if(has_flag(bcfg.policy, BondExpansionPolicy::PREOPT_1SITE) and bcfg.order == BondExpansionOrder::PREOPT) {
        res = tools::finite::bex::rexpand_bond_preopt_1site(tensors.get_state(), tensors.get_model(), tensors.get_edges(), bcfg);
    } else if(has_flag(bcfg.policy, BondExpansionPolicy::POSTOPT_RDMP_1SITE) and bcfg.order == BondExpansionOrder::POSTOPT) {
        res = tools::finite::bex::density_matrix_perturbation_postopt_1site(tensors, bcfg);
    } else if(has_flag(bcfg.policy, BondExpansionPolicy::PREOPT_RDMP_1SITE) and bcfg.order == BondExpansionOrder::PREOPT) {
        res = tools::finite::bex::density_matrix_perturbation_preopt_1site(tensors, bcfg);
    } else if(has_any_flags(bcfg.policy, BondExpansionPolicy::PREOPT_NSITE_REAR, BondExpansionPolicy::PREOPT_NSITE_FORE) and
              bcfg.order == BondExpansionOrder::PREOPT) {
        res = tools::finite::bex::expand_bond_preopt_nsite(tensors.get_state(), tensors.get_model(), tensors.get_edges(), bcfg);
    }

    if(res.ok) {
        tools::log->debug("Expanded environment {} block [{}-{}] | var {:.3e} -> {:.3e} | ene {:.16f} -> {:.16f} | hsq {:.16f} -> {:.16f}",
                          flag2str(bcfg.policy), res.posL, res.posR, res.var_old, res.var_new, res.ene_old, res.ene_new, res.hsq_old, res.hsq_new);
        tensors.clear_measurements();
    } else {
        tools::log->debug("Expansion canceled: {}", res.msg);
    }
    if constexpr(settings::debug) tensors.assert_validity();
    return res;
}
