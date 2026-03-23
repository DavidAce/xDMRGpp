#pragma once
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include <stdexcept>

template<typename Scalar>
template<typename Value>
const MeasurementFiniteCached<Value> *
MeasurementsTensorsFinite<Scalar>::get_cached_impl(const std::optional<MeasurementFiniteCached<Value>> &cache, const MeasurementFiniteIds &ids) {
    if(cache and cache->ids == ids) return &cache.value();
    return nullptr;
}

template<typename Scalar>
template<typename Value>
void MeasurementsTensorsFinite<Scalar>::set_cached_impl(std::optional<MeasurementFiniteCached<Value>> &cache, Value value, MeasurementFiniteIds ids) {
    cache = MeasurementFiniteCached<Value>{.data = std::move(value), .ids = std::move(ids)};
}

template<typename Scalar>
template<typename Site>
std::vector<size_t> MeasurementsTensorsFinite<Scalar>::get_sites(const std::vector<std::reference_wrapper<const Site>> &refs) {
    std::vector<size_t> sites;
    sites.reserve(refs.size());
    for(const auto &ref : refs) sites.emplace_back(ref.get().get_position());
    return sites;
}

template<typename Scalar>
template<typename EnvRef>
std::vector<size_t> MeasurementsTensorsFinite<Scalar>::get_env_ids(const env_pair<EnvRef> &envs) {
    return {envs.L.get_unique_id(), envs.R.get_unique_id()};
}

template<typename Scalar>
std::vector<size_t> MeasurementsTensorsFinite<Scalar>::get_mps_ids(const MpsRefs &mps) {
    std::vector<size_t> ids;
    ids.reserve(mps.size());
    for(const auto &ref : mps) ids.emplace_back(ref.get().get_unique_id());
    return ids;
}

template<typename Scalar>
std::vector<size_t> MeasurementsTensorsFinite<Scalar>::get_mpo_ids(const MpoRefs &mpo) {
    std::vector<size_t> ids;
    ids.reserve(mpo.size());
    for(const auto &ref : mpo) ids.emplace_back(ref.get().get_unique_id());
    return ids;
}

template<typename Scalar>
std::vector<size_t> MeasurementsTensorsFinite<Scalar>::get_mpo_sq_ids(const MpoRefs &mpo) {
    std::vector<size_t> ids;
    ids.reserve(mpo.size());
    for(const auto &ref : mpo) ids.emplace_back(ref.get().get_unique_id_sq());
    return ids;
}

template<typename Scalar>
typename MeasurementsTensorsFinite<Scalar>::MpsRefs MeasurementsTensorsFinite<Scalar>::get_all_mps_refs(const StateFinite<Scalar> &state) {
    MpsRefs refs;
    refs.reserve(state.mps_sites.size());
    for(const auto &mps : state.mps_sites) refs.emplace_back(std::cref(*mps));
    return refs;
}

template<typename Scalar>
typename MeasurementsTensorsFinite<Scalar>::MpoRefs MeasurementsTensorsFinite<Scalar>::get_all_mpo_refs(const ModelFinite<Scalar> &model) {
    MpoRefs refs;
    refs.reserve(model.MPO.size());
    for(const auto &mpo : model.MPO) refs.emplace_back(std::cref(*mpo));
    return refs;
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::assert_matching_sites(const std::vector<size_t> &sites, const std::vector<size_t> &other_sites) {
    if(sites.empty()) throw std::logic_error("Measurement cache requires non-empty sites");
    if(sites != other_sites) throw std::logic_error("Measurement cache site mismatch");
}

template<typename Scalar>
template<typename EnvRef>
void MeasurementsTensorsFinite<Scalar>::assert_matching_env_sites(const std::vector<size_t> &sites, const env_pair<EnvRef> &envs) {
    if(sites.empty()) throw std::logic_error("Measurement cache requires non-empty sites");
    if(envs.L.get_position() != sites.front() or envs.R.get_position() != sites.back()) throw std::logic_error("Measurement cache env mismatch");
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_energy_like(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) {
    const auto sites_mps = get_sites(mps);
    const auto sites_mpo = get_sites(mpo);
    assert_matching_sites(sites_mps, sites_mpo);
    assert_matching_env_sites(sites_mps, envs);
    return MeasurementFiniteIds{
        .sites      = sites_mps,
        .mps_ids    = get_mps_ids(mps),
        .mpo_ids    = get_mpo_ids(mpo),
        .env_ene_ids = get_env_ids(envs),
    };
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_h2_like(const MpsRefs &mps, const MpoRefs &mpo, const VarEnvs &envs) {
    const auto sites_mps = get_sites(mps);
    const auto sites_mpo = get_sites(mpo);
    assert_matching_sites(sites_mps, sites_mpo);
    assert_matching_env_sites(sites_mps, envs);
    return MeasurementFiniteIds{
        .sites       = sites_mps,
        .mps_ids     = get_mps_ids(mps),
        .mpo_sq_ids  = get_mpo_sq_ids(mpo),
        .env_var_ids = get_env_ids(envs),
    };
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_variance_like(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene,
                                                                               const VarEnvs &envs_var) {
    const auto sites_mps = get_sites(mps);
    const auto sites_mpo = get_sites(mpo);
    assert_matching_sites(sites_mps, sites_mpo);
    assert_matching_env_sites(sites_mps, envs_ene);
    assert_matching_env_sites(sites_mps, envs_var);
    return MeasurementFiniteIds{
        .sites       = sites_mps,
        .mps_ids     = get_mps_ids(mps),
        .mpo_ids     = get_mpo_ids(mpo),
        .mpo_sq_ids  = get_mpo_sq_ids(mpo),
        .env_ene_ids = get_env_ids(envs_ene),
        .env_var_ids = get_env_ids(envs_var),
    };
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_global_h1_like(const MpoRefs &mpo, const EneEnvs &envs) {
    const auto sites_mpo = get_sites(mpo);
    assert_matching_env_sites(sites_mpo, envs);
    return MeasurementFiniteIds{.sites = sites_mpo, .mpo_ids = get_mpo_ids(mpo), .env_ene_ids = get_env_ids(envs)};
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_global_h2_like(const MpoRefs &mpo, const VarEnvs &envs) {
    const auto sites_mpo = get_sites(mpo);
    assert_matching_env_sites(sites_mpo, envs);
    return MeasurementFiniteIds{.sites = sites_mpo, .mpo_sq_ids = get_mpo_sq_ids(mpo), .env_var_ids = get_env_ids(envs)};
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_state_model_like(const MpsRefs &mps, const MpoRefs &mpo) {
    const auto sites_mps = get_sites(mps);
    const auto sites_mpo = get_sites(mpo);
    assert_matching_sites(sites_mps, sites_mpo);
    return MeasurementFiniteIds{.sites = sites_mps, .mps_ids = get_mps_ids(mps), .mpo_ids = get_mpo_ids(mpo)};
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_local_norm_h1_like(const MpoRefs &mpo, const EneEnvs &envs, long long maxiter,
                                                                                    double reltol) {
    auto ids   = get_ids_global_h1_like(mpo, envs);
    ids.maxiter = maxiter;
    ids.reltol  = reltol;
    return ids;
}

template<typename Scalar>
MeasurementFiniteIds MeasurementsTensorsFinite<Scalar>::get_ids_local_norm_h2_like(const MpoRefs &mpo, const VarEnvs &envs, long long maxiter,
                                                                                    double reltol) {
    auto ids   = get_ids_global_h2_like(mpo, envs);
    ids.maxiter = maxiter;
    ids.reltol  = reltol;
    return ids;
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) const {
    return get_cached_impl(energy, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_minus_energy_shift(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) const {
    return get_cached_impl(energy_minus_energy_shift, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_variance(const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene,
                                                              const VarEnvs &envs_var) const {
    return get_cached_impl(energy_variance, get_ids_variance_like(mps, mpo, envs_ene, envs_var));
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian(const MpsRefs &mps, const MpoRefs &mpo,
                                                                                                         const EneEnvs &envs) const {
    return get_cached_impl(expval_hamiltonian, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian_squared(const MpsRefs &mps, const MpoRefs &mpo,
                                                                                                                 const VarEnvs &envs) const {
    return get_cached_impl(expval_hamiltonian_squared, get_ids_h2_like(mps, mpo, envs));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) {
    set_cached_impl(energy, value, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_minus_energy_shift(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo,
                                                                             const EneEnvs &envs) {
    set_cached_impl(energy_minus_energy_shift, value, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_variance(RealScalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs_ene,
                                                                   const VarEnvs &envs_var) {
    set_cached_impl(energy_variance, value, get_ids_variance_like(mps, mpo, envs_ene, envs_var));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian(Scalar value, const MpsRefs &mps, const MpoRefs &mpo, const EneEnvs &envs) {
    set_cached_impl(expval_hamiltonian, value, get_ids_energy_like(mps, mpo, envs));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian_squared(Scalar value, const MpsRefs &mps, const MpoRefs &mpo,
                                                                               const VarEnvs &envs) {
    set_cached_impl(expval_hamiltonian_squared, value, get_ids_h2_like(mps, mpo, envs));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) const {
    return get_cached_energy(state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_minus_energy_shift(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                        const EdgesFinite<Scalar> &edges) const {
    return get_cached_energy_minus_energy_shift(state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_variance(const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                              const EdgesFinite<Scalar> &edges) const {
    return get_cached_energy_variance(state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active(), edges.get_var_active());
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian(const StateFinite<Scalar> &state,
                                                                                                         const ModelFinite<Scalar> &model,
                                                                                                         const EdgesFinite<Scalar> &edges) const {
    return get_cached_expval_hamiltonian(state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian_squared(const StateFinite<Scalar> &state,
                                                                                                                 const ModelFinite<Scalar> &model,
                                                                                                                 const EdgesFinite<Scalar> &edges) const {
    return get_cached_expval_hamiltonian_squared(state.get_mps_active(), model.get_mpo_active(), edges.get_var_active());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy(RealScalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                          const EdgesFinite<Scalar> &edges) {
    set_cached_energy(value, state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_minus_energy_shift(RealScalar value, const StateFinite<Scalar> &state,
                                                                             const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    set_cached_energy_minus_energy_shift(value, state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_variance(RealScalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                   const EdgesFinite<Scalar> &edges) {
    set_cached_energy_variance(value, state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active(), edges.get_var_active());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian(Scalar value, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                                                      const EdgesFinite<Scalar> &edges) {
    set_cached_expval_hamiltonian(value, state.get_mps_active(), model.get_mpo_active(), edges.get_ene_active());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian_squared(Scalar value, const StateFinite<Scalar> &state,
                                                                              const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges) {
    set_cached_expval_hamiltonian_squared(value, state.get_mps_active(), model.get_mpo_active(), edges.get_var_active());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_energy(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_minus_energy_shift(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_energy_minus_energy_shift(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_energy_variance(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_energy_variance(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_expval_hamiltonian(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
const MeasurementFiniteCached<Scalar> *MeasurementsTensorsFinite<Scalar>::get_cached_expval_hamiltonian_squared(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_expval_hamiltonian_squared(tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_local_hamiltonian_norm(const TensorsFinite<Scalar> &tensors, long long maxiter, double reltol) const {
    return get_cached_impl(local_hamiltonian_norm, get_ids_local_norm_h1_like(tensors.get_model().get_mpo_active(), tensors.get_edges().get_ene_active(), maxiter, reltol));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_local_hamiltonian_squared_norm(const TensorsFinite<Scalar> &tensors, long long maxiter, double reltol) const {
    return get_cached_impl(local_hamiltonian_squared_norm,
                           get_ids_local_norm_h2_like(tensors.get_model().get_mpo_active(), tensors.get_edges().get_var_active(), maxiter, reltol));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_global_hamiltonian_trace(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_impl(global_hamiltonian_trace, get_ids_global_h1_like(get_all_mpo_refs(tensors.get_model()),
                                                                            tensors.get_edges().get_env_ene(0, tensors.get_state().template get_length<size_t>() - 1)));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_global_hamiltonian_squared_trace(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_impl(global_hamiltonian_squared_trace, get_ids_global_h2_like(get_all_mpo_refs(tensors.get_model()),
                                                                                    tensors.get_edges().get_env_var(0, tensors.get_state().template get_length<size_t>() - 1)));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_residual_norm_h1(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_impl(residual_norm_h1, get_ids_energy_like(tensors.get_state().get_mps_active(), tensors.get_model().get_mpo_active(),
                                                                 tensors.get_edges().get_ene_active()));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_residual_norm_h2(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_impl(residual_norm_h2, get_ids_h2_like(tensors.get_state().get_mps_active(), tensors.get_model().get_mpo_active(), tensors.get_edges().get_var_active()));
}

template<typename Scalar>
const MeasurementFiniteCached<typename MeasurementsTensorsFinite<Scalar>::RealScalar> *
MeasurementsTensorsFinite<Scalar>::get_cached_residual_norm_full(const TensorsFinite<Scalar> &tensors) const {
    return get_cached_impl(residual_norm_full, get_ids_state_model_like(get_all_mps_refs(tensors.get_state()), get_all_mpo_refs(tensors.get_model())));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_energy(value, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_minus_energy_shift(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_energy_minus_energy_shift(value, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_energy_variance(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_energy_variance(value, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian(Scalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_expval_hamiltonian(value, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_expval_hamiltonian_squared(Scalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_expval_hamiltonian_squared(value, tensors.get_state(), tensors.get_model(), tensors.get_edges());
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_local_hamiltonian_norm(RealScalar value, const TensorsFinite<Scalar> &tensors, long long maxiter, double reltol) {
    set_cached_impl(local_hamiltonian_norm, value,
                    get_ids_local_norm_h1_like(tensors.get_model().get_mpo_active(), tensors.get_edges().get_ene_active(), maxiter, reltol));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_local_hamiltonian_squared_norm(RealScalar value, const TensorsFinite<Scalar> &tensors, long long maxiter,
                                                                                   double reltol) {
    set_cached_impl(local_hamiltonian_squared_norm, value,
                    get_ids_local_norm_h2_like(tensors.get_model().get_mpo_active(), tensors.get_edges().get_var_active(), maxiter, reltol));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_global_hamiltonian_trace(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_impl(global_hamiltonian_trace, value,
                    get_ids_global_h1_like(get_all_mpo_refs(tensors.get_model()),
                                           tensors.get_edges().get_env_ene(0, tensors.get_state().template get_length<size_t>() - 1)));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_global_hamiltonian_squared_trace(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_impl(global_hamiltonian_squared_trace, value,
                    get_ids_global_h2_like(get_all_mpo_refs(tensors.get_model()),
                                           tensors.get_edges().get_env_var(0, tensors.get_state().template get_length<size_t>() - 1)));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_residual_norm_h1(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_impl(residual_norm_h1, value,
                    get_ids_energy_like(tensors.get_state().get_mps_active(), tensors.get_model().get_mpo_active(), tensors.get_edges().get_ene_active()));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_residual_norm_h2(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_impl(residual_norm_h2, value,
                    get_ids_h2_like(tensors.get_state().get_mps_active(), tensors.get_model().get_mpo_active(), tensors.get_edges().get_var_active()));
}

template<typename Scalar>
void MeasurementsTensorsFinite<Scalar>::set_cached_residual_norm_full(RealScalar value, const TensorsFinite<Scalar> &tensors) {
    set_cached_impl(residual_norm_full, value, get_ids_state_model_like(get_all_mps_refs(tensors.get_state()), get_all_mpo_refs(tensors.get_model())));
}
