#pragma once
#include "config/debug.h"
#include "config/enums/GateMove.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/num.h"
#include "tensors/TensorsFinite.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include "tools/finite/mpo.h"
#include "tools/finite/mps.h"
#include "tools/finite/multisite.h"
#include "tools/finite/pos.h"
#include <algorithm>
#include <cmath>

template<typename Scalar>
bool tools::finite::pos::has_center_point(const TensorsFinite<Scalar> &tensors) {
    return tensors.get_state().has_center_point();
}

template<typename Scalar>
bool tools::finite::pos::position_is_the_middle(const TensorsFinite<Scalar> &tensors) {
    return tensors.get_state().position_is_the_middle();
}

template<typename Scalar>
bool tools::finite::pos::position_is_the_middle_any_direction(const TensorsFinite<Scalar> &tensors) {
    return tensors.get_state().position_is_the_middle_any_direction();
}

template<typename Scalar>
bool tools::finite::pos::position_is_outward_edge_left(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_outward_edge_left(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_outward_edge_right(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_outward_edge_right(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_outward_edge(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_outward_edge(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_inward_edge_left(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_inward_edge_left(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_inward_edge_right(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_inward_edge_right(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_inward_edge(const TensorsFinite<Scalar> &tensors, size_t nsite) {
    return tensors.get_state().position_is_inward_edge(nsite);
}

template<typename Scalar>
bool tools::finite::pos::position_is_at(const TensorsFinite<Scalar> &tensors, long pos) {
    return tensors.get_state().position_is_at(pos);
}

template<typename Scalar>
bool tools::finite::pos::position_is_at(const TensorsFinite<Scalar> &tensors, long pos, int dir) {
    return tensors.get_state().position_is_at(pos, dir);
}

template<typename Scalar>
bool tools::finite::pos::position_is_at(const TensorsFinite<Scalar> &tensors, long pos, int dir, bool isCenter) {
    return tensors.get_state().position_is_at(pos, dir, isCenter);
}

template<typename Scalar>
void tools::finite::pos::sync_active_sites(TensorsFinite<Scalar> &tensors) {
    if(num::all_equal(tensors.active_sites, tensors.state->active_sites, tensors.model->active_sites, tensors.edges->active_sites)) return;
    if(not tensors.active_sites.empty())
        activate_sites(tensors, tensors.active_sites);
    else if(not tensors.state->active_sites.empty())
        activate_sites(tensors, tensors.state->active_sites);
    else if(not tensors.model->active_sites.empty())
        activate_sites(tensors, tensors.model->active_sites);
    else if(not tensors.edges->active_sites.empty())
        activate_sites(tensors, tensors.edges->active_sites);
    else
        clear_active_sites(tensors);
}

template<typename Scalar>
void tools::finite::pos::clear_active_sites(TensorsFinite<Scalar> &tensors) {
    if constexpr(settings::debug) tools::log->trace("Clearing active sites {}", tensors.active_sites);
    tensors.active_sites.clear();
    tensors.state->active_sites.clear();
    tensors.model->active_sites.clear();
    tensors.edges->active_sites.clear();
}

template<typename Scalar>
void tools::finite::pos::activate_sites(TensorsFinite<Scalar> &tensors, const std::vector<size_t> &sites) {
    tools::log->trace("Activating sites: {}", sites);
    if(num::all_equal(sites, tensors.active_sites, tensors.state->active_sites, tensors.model->active_sites, tensors.edges->active_sites)) return;
    tensors.active_sites        = sites;
    tensors.state->active_sites = tensors.active_sites;
    tensors.model->active_sites = tensors.active_sites;
    tensors.edges->active_sites = tensors.active_sites;
    tensors.clear_cache();
    tensors.clear_measurements();
}

template<typename Scalar>
void tools::finite::pos::activate_sites(TensorsFinite<Scalar> &tensors) {
    sync_active_sites(tensors);
    if(tensors.active_sites.empty()) {
        if(position_is_at(tensors, -1)) throw except::logic_error("activate_sites: cannot activate a default site when pos == -1");
        activate_sites(tensors, {tensors.template get_position<size_t>()});
    }
}

template<typename Scalar>
void tools::finite::pos::activate_sites(TensorsFinite<Scalar> &tensors, long threshold, size_t max_sites, size_t min_sites) {
    activate_sites(tensors, tools::finite::multisite::generate_site_list(tensors.get_state(), threshold, max_sites, min_sites));
}

template<typename Scalar>
std::array<long, 3> tools::finite::pos::active_problem_dims(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::multisite::get_dimensions(tensors.get_state(), tensors.active_sites);
}

template<typename Scalar>
long tools::finite::pos::active_problem_size(const TensorsFinite<Scalar> &tensors) {
    return tools::finite::multisite::get_problem_size(tensors.get_state(), tensors.active_sites);
}

template<typename Scalar>
size_t tools::finite::pos::move_center_point(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_single_site(tensors.get_state(), svd_cfg);
    if(moves != 0) clear_active_sites(tensors);
    return moves;
}

template<typename Scalar>
size_t tools::finite::pos::move_center_point_to_pos(TensorsFinite<Scalar> &tensors, long pos, std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_pos(tensors.get_state(), pos, svd_cfg);
    if(moves != 0) clear_active_sites(tensors);
    return moves;
}

template<typename Scalar>
size_t tools::finite::pos::move_center_point_to_inward_edge(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_inward_edge(tensors.get_state(), svd_cfg);
    if(moves != 0) clear_active_sites(tensors);
    return moves;
}

template<typename Scalar>
size_t tools::finite::pos::move_center_point_to_middle(TensorsFinite<Scalar> &tensors, std::optional<svd::config> svd_cfg) {
    auto moves = tools::finite::mps::move_center_point_to_middle(tensors.get_state(), svd_cfg);
    if(moves != 0) clear_active_sites(tensors);
    return moves;
}

template<typename Scalar>
void tools::finite::pos::move_site_mps(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mps,
                                       std::optional<long> new_pos) {
    if(sites_mps.size() != tensors.template get_length<size_t>()) {
        sites_mps.clear();
        for(const auto &mps : tensors.state->mps_sites) sites_mps.emplace_back(mps->template get_position<size_t>());
    }
    long dir = steps < 0l ? -1l : 1l;

    for(long step = 0; std::abs(step) < std::abs(steps); step += dir) {
        long posL = std::min(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        long posR = std::max(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        if(posL == posR) break;
        if(posL < 0 or posL >= tensors.template get_length<long>()) break;
        if(posR < 0 or posR >= tensors.template get_length<long>()) break;
        tools::log->debug("swapping mps sites {} <--> {}", posL, posR);
        tools::finite::mps::swap_sites(tensors.get_state(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mps, GateMove::OFF);
    }
    if(new_pos) {
        if(new_pos.value() != std::clamp(new_pos.value(), 0l, tensors.template get_length<long>()))
            throw except::runtime_error("move_site: expected new_pos in range [0,{}]. Got {}", tensors.template get_length<long>(), new_pos.value());
        move_center_point_to_pos(tensors, new_pos.value());
        activate_sites(tensors, std::vector<size_t>{safe_cast<size_t>(new_pos.value())});
    }

    tools::log->debug("Sites mps: {}", sites_mps);
    tensors.clear_cache();
    tensors.clear_measurements();
}

template<typename Scalar>
void tools::finite::pos::move_site_mpo(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mpo) {
    if(sites_mpo.size() != tensors.template get_length<size_t>()) {
        sites_mpo.clear();
        for(const auto &mpo : tensors.model->MPO) sites_mpo.emplace_back(mpo->get_position());
    }
    long dir = steps < 0l ? -1l : 1l;

    for(long step = 0; std::abs(step) < std::abs(steps); step += dir) {
        long posL = std::min(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        long posR = std::max(safe_cast<long>(site) + step, safe_cast<long>(site) + step + dir);
        if(posL == posR) break;
        if(posL < 0 or posL >= tensors.template get_length<long>()) break;
        if(posR < 0 or posR >= tensors.template get_length<long>()) break;
        tools::log->debug("swapping mpo sites {} <--> {}", posL, posR);
        tools::finite::mpo::swap_sites(tensors.get_model(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mpo);
    }
    tools::log->debug("Sites mpo: {}", sites_mpo);
    tensors.clear_cache();
    tensors.clear_measurements();
}

template<typename Scalar>
void tools::finite::pos::move_site_mps_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos, std::vector<size_t> &sites_mps,
                                              std::optional<long> new_pos) {
    if(sites_mps.size() != tensors.template get_length<size_t>()) {
        sites_mps.clear();
        for(const auto &mps : tensors.state->mps_sites) sites_mps.emplace_back(mps->template get_position<size_t>());
    }
    while(true) {
        auto src_itr = std::find(sites_mps.begin(), sites_mps.end(), site);
        if(src_itr == sites_mps.end()) throw except::logic_error("site {} was not found in sites_mps: {}", site, sites_mps);
        auto src_pos = std::distance(sites_mps.begin(), src_itr);
        if(src_pos == tgt_pos) break;
        long step = tgt_pos < src_pos ? -1l : 1l;
        long posL = src_pos + (step < 0 ? -1l : 0);
        long posR = src_pos + (step < 0 ? 0 : 1l);
        if(posL == posR) break;
        if(posL < 0 or posL >= tensors.template get_length<long>()) break;
        if(posR < 0 or posR >= tensors.template get_length<long>()) break;
        tools::log->debug("swapping mps sites {} <--> {}", posL, posR);
        tools::finite::mps::swap_sites(tensors.get_state(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mps, GateMove::OFF);
    }
    if(new_pos) {
        if(new_pos.value() != std::clamp(new_pos.value(), 0l, tensors.template get_length<long>()))
            throw except::runtime_error("move_site: expected new_pos in range [0,{}]. Got {}", tensors.template get_length<long>(), new_pos.value());
        move_center_point_to_pos(tensors, new_pos.value());
        activate_sites(tensors, std::vector<size_t>{safe_cast<size_t>(new_pos.value())});
    }

    tools::log->debug("Sites mps: {}", sites_mps);
    tensors.clear_cache();
    tensors.clear_measurements();
}

template<typename Scalar>
void tools::finite::pos::move_site_mpo_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos, std::vector<size_t> &sites_mpo) {
    if(sites_mpo.size() != tensors.template get_length<size_t>()) {
        sites_mpo.clear();
        for(const auto &mpo : tensors.model->MPO) sites_mpo.emplace_back(mpo->get_position());
    }

    while(true) {
        auto src_itr = std::find(sites_mpo.begin(), sites_mpo.end(), site);
        if(src_itr == sites_mpo.end()) throw except::logic_error("site {} was not found in sites_mpo: {}", site, sites_mpo);
        auto src_pos = std::distance(sites_mpo.begin(), src_itr);
        if(src_pos == tgt_pos) break;
        long step = tgt_pos < src_pos ? -1l : 1l;
        long posL = src_pos + (step < 0 ? -1l : 0);
        long posR = src_pos + (step < 0 ? 0 : 1l);
        if(posL == posR) break;
        if(posL < 0 or posL >= tensors.template get_length<long>()) break;
        if(posR < 0 or posR >= tensors.template get_length<long>()) break;
        tools::log->debug("swapping mpo sites {} <--> {}", posL, posR);
        tools::finite::mpo::swap_sites(tensors.get_model(), safe_cast<size_t>(posL), safe_cast<size_t>(posR), sites_mpo);
    }
    tools::log->debug("Sites mpo: {}", sites_mpo);
    tensors.clear_cache();
    tensors.clear_measurements();
}

template<typename Scalar>
void tools::finite::pos::move_site(TensorsFinite<Scalar> &tensors, const size_t site, const long steps, std::vector<size_t> &sites_mps,
                                   std::vector<size_t> &sites_mpo, std::optional<long> new_pos) {
    move_site_mps(tensors, site, steps, sites_mps, new_pos);
    move_site_mpo(tensors, site, steps, sites_mpo);
}

template<typename Scalar>
void tools::finite::pos::move_site_to_pos(TensorsFinite<Scalar> &tensors, const size_t site, const long tgt_pos,
                                          std::optional<std::vector<size_t>> &sites_mps, std::optional<std::vector<size_t>> &sites_mpo,
                                          std::optional<long> new_pos) {
    if(not sites_mps) sites_mps = std::vector<size_t>{};
    if(not sites_mpo) sites_mpo = std::vector<size_t>{};
    move_site_mps_to_pos(tensors, site, tgt_pos, sites_mps.value(), new_pos);
    move_site_mpo_to_pos(tensors, site, tgt_pos, sites_mpo.value());
    if(sites_mps != sites_mpo) throw except::logic_error("sites mismatch \n sites_mps {}\n sites_mpo {}", sites_mps.value(), sites_mpo.value());
}
