#pragma once
#include "tensors/site/mps/MpsSite.h"

template<typename Scalar>
template<typename Value>
const MeasurementStateFiniteCached<Value> *MeasurementsStateFinite<Scalar>::get_cached_impl(
    const std::optional<MeasurementStateFiniteCached<Value>> &cache, const MeasurementStateFiniteIds &ids, bool allow_info_policy_compatibility) {
    if(cache and ids_match(ids, cache->ids, allow_info_policy_compatibility)) return &cache.value();
    return nullptr;
}

template<typename Scalar>
template<typename Value>
void MeasurementsStateFinite<Scalar>::set_cached_impl(std::optional<MeasurementStateFiniteCached<Value>> &cache, Value value, MeasurementStateFiniteIds ids) {
    cache = MeasurementStateFiniteCached<Value>{.data = std::move(value), .ids = std::move(ids)};
}

template<typename Scalar>
bool MeasurementsStateFinite<Scalar>::ids_match(const MeasurementStateFiniteIds &requested, const MeasurementStateFiniteIds &cached,
                                                bool allow_info_policy_compatibility) {
    if(requested.mps_ids != cached.mps_ids) return false;
    if(requested.position != cached.position) return false;
    if(requested.model_type != cached.model_type) return false;
    if(requested.algorithm != cached.algorithm) return false;
    if(requested.popcount != cached.popcount) return false;
    if(not allow_info_policy_compatibility) return requested.info_policy == cached.info_policy;
    if(requested.info_policy.has_value()) return requested.info_policy->is_compatible(cached.info_policy);
    return not cached.info_policy.has_value();
}

template<typename Scalar>
std::vector<size_t> MeasurementsStateFinite<Scalar>::get_mps_ids(const StateFinite<Scalar> &state) {
    std::vector<size_t> ids;
    ids.reserve(state.mps_sites.size());
    for(const auto &mps : state.mps_sites) ids.emplace_back(mps->get_unique_id());
    return ids;
}

template<typename Scalar>
MeasurementStateFiniteIds MeasurementsStateFinite<Scalar>::get_ids_state(const StateFinite<Scalar> &state) {
    return MeasurementStateFiniteIds{.mps_ids = get_mps_ids(state)};
}

template<typename Scalar>
MeasurementStateFiniteIds MeasurementsStateFinite<Scalar>::get_ids_opdm(const StateFinite<Scalar> &state, ModelType model_type) {
    auto ids        = get_ids_state(state);
    ids.model_type  = model_type;
    return ids;
}

template<typename Scalar>
MeasurementStateFiniteIds MeasurementsStateFinite<Scalar>::get_ids_number_entropies(const StateFinite<Scalar> &state) {
    auto ids       = get_ids_state(state);
    ids.algorithm  = state.get_algorithm();
    ids.popcount   = state.popcount;
    return ids;
}

template<typename Scalar>
MeasurementStateFiniteIds MeasurementsStateFinite<Scalar>::get_ids_info(const StateFinite<Scalar> &state, const InfoPolicy &ip) {
    auto ids        = get_ids_state(state);
    ids.info_policy = ip;
    return ids;
}

template<typename Scalar>
const MeasurementStateFiniteCached<Eigen::Tensor<Scalar, 2>> *
MeasurementsStateFinite<Scalar>::get_cached_opdm(const StateFinite<Scalar> &state, ModelType model_type) const {
    return get_cached_impl(cached_opdm, get_ids_opdm(state, model_type));
}

template<typename Scalar>
const MeasurementStateFiniteCached<Eigen::Tensor<typename MeasurementsStateFinite<Scalar>::RealScalar, 1>> *
MeasurementsStateFinite<Scalar>::get_cached_opdm_spectrum(const StateFinite<Scalar> &state, ModelType model_type) const {
    return get_cached_impl(cached_opdm_spectrum, get_ids_opdm(state, model_type));
}

template<typename Scalar>
const MeasurementStateFiniteCached<std::vector<typename MeasurementsStateFinite<Scalar>::RealScalar>> *
MeasurementsStateFinite<Scalar>::get_cached_number_entropies(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_number_entropies, get_ids_number_entropies(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealArrayXX> *
MeasurementsStateFinite<Scalar>::get_cached_subsystem_entanglement_entropies(const StateFinite<Scalar> &state, const InfoPolicy &ip) const {
    return get_cached_impl(cached_subsystem_entanglement_entropies, get_ids_info(state, ip), true);
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealArrayXX> *
MeasurementsStateFinite<Scalar>::get_cached_information_lattice(const StateFinite<Scalar> &state, const InfoPolicy &ip) const {
    return get_cached_impl(cached_information_lattice, get_ids_info(state, ip), true);
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealArrayX> *
MeasurementsStateFinite<Scalar>::get_cached_information_per_scale(const StateFinite<Scalar> &state, const InfoPolicy &ip) const {
    return get_cached_impl(cached_information_per_scale, get_ids_info(state, ip), true);
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealScalar> *
MeasurementsStateFinite<Scalar>::get_cached_information_center_of_mass(const StateFinite<Scalar> &state, const InfoPolicy &ip) const {
    return get_cached_impl(cached_information_center_of_mass, get_ids_info(state, ip), true);
}

template<typename Scalar>
const MeasurementStateFiniteCached<Eigen::Tensor<typename MeasurementsStateFinite<Scalar>::RealScalar, 2>> *
MeasurementsStateFinite<Scalar>::get_cached_correlation_matrix_sx(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_correlation_matrix_sx, get_ids_state(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<Eigen::Tensor<typename MeasurementsStateFinite<Scalar>::RealScalar, 2>> *
MeasurementsStateFinite<Scalar>::get_cached_correlation_matrix_sy(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_correlation_matrix_sy, get_ids_state(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<Eigen::Tensor<typename MeasurementsStateFinite<Scalar>::RealScalar, 2>> *
MeasurementsStateFinite<Scalar>::get_cached_correlation_matrix_sz(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_correlation_matrix_sz, get_ids_state(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealScalar> *
MeasurementsStateFinite<Scalar>::get_cached_structure_factor_x(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_structure_factor_x, get_ids_state(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealScalar> *
MeasurementsStateFinite<Scalar>::get_cached_structure_factor_y(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_structure_factor_y, get_ids_state(state));
}

template<typename Scalar>
const MeasurementStateFiniteCached<typename MeasurementsStateFinite<Scalar>::RealScalar> *
MeasurementsStateFinite<Scalar>::get_cached_structure_factor_z(const StateFinite<Scalar> &state) const {
    return get_cached_impl(cached_structure_factor_z, get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_opdm(Eigen::Tensor<Scalar, 2> value, const StateFinite<Scalar> &state, ModelType model_type) {
    opdm = value;
    set_cached_impl(cached_opdm, std::move(value), get_ids_opdm(state, model_type));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_opdm_spectrum(Eigen::Tensor<RealScalar, 1> value, const StateFinite<Scalar> &state, ModelType model_type) {
    opdm_spectrum = value;
    set_cached_impl(cached_opdm_spectrum, std::move(value), get_ids_opdm(state, model_type));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_number_entropies(std::vector<RealScalar> value, Eigen::Tensor<RealScalar, 2> probabilities,
                                                                  const StateFinite<Scalar> &state) {
    number_entropies        = value;
    number_probabilities    = probabilities;
    number_entropy_midchain = value.at(state.template get_length<size_t>() / 2);
    number_entropy_current  = state.has_center_point() ? std::optional<RealScalar>(value.at(state.template get_position<size_t>() + 1))
                                                       : std::optional<RealScalar>(RealScalar{0});
    set_cached_impl(cached_number_entropies, std::move(value), get_ids_number_entropies(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_subsystem_entanglement_entropies(RealArrayXX value, const StateFinite<Scalar> &state, const InfoPolicy &ip) {
    subsystem_entanglement_entropies = value;
    info_policy                      = ip;
    set_cached_impl(cached_subsystem_entanglement_entropies, std::move(value), get_ids_info(state, ip));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_information_lattice(RealArrayXX value, const StateFinite<Scalar> &state, const InfoPolicy &ip) {
    information_lattice = value;
    info_policy         = ip;
    set_cached_impl(cached_information_lattice, std::move(value), get_ids_info(state, ip));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_information_per_scale(RealArrayX value, const StateFinite<Scalar> &state, const InfoPolicy &ip) {
    information_per_scale = value;
    info_policy           = ip;
    set_cached_impl(cached_information_per_scale, std::move(value), get_ids_info(state, ip));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_information_center_of_mass(RealScalar value, const StateFinite<Scalar> &state, const InfoPolicy &ip) {
    information_center_of_mass = value;
    info_policy                = ip;
    set_cached_impl(cached_information_center_of_mass, value, get_ids_info(state, ip));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_correlation_matrix_sx(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state) {
    correlation_matrix_sx = value;
    set_cached_impl(cached_correlation_matrix_sx, std::move(value), get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_correlation_matrix_sy(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state) {
    correlation_matrix_sy = value;
    set_cached_impl(cached_correlation_matrix_sy, std::move(value), get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_correlation_matrix_sz(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state) {
    correlation_matrix_sz = value;
    set_cached_impl(cached_correlation_matrix_sz, std::move(value), get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_structure_factor_x(RealScalar value, const StateFinite<Scalar> &state) {
    structure_factor_x = value;
    set_cached_impl(cached_structure_factor_x, value, get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_structure_factor_y(RealScalar value, const StateFinite<Scalar> &state) {
    structure_factor_y = value;
    set_cached_impl(cached_structure_factor_y, value, get_ids_state(state));
}

template<typename Scalar>
void MeasurementsStateFinite<Scalar>::set_cached_structure_factor_z(RealScalar value, const StateFinite<Scalar> &state) {
    structure_factor_z = value;
    set_cached_impl(cached_structure_factor_z, value, get_ids_state(state));
}
