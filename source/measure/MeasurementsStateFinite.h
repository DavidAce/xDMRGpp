#pragma once
#include "../config/enums/AlgorithmType.h"
#include "../config/enums/ModelType.h"
#include "../tools/finite/measure/infopolicy.h"
#include <array>
#include <complex>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

template<typename Scalar>
class StateFinite;

struct MeasurementStateFiniteIds {
    std::vector<size_t>        mps_ids{};
    std::optional<size_t>      position   = std::nullopt;
    std::optional<ModelType>   model_type = std::nullopt;
    std::optional<AlgorithmType> algorithm = std::nullopt;
    std::optional<size_t>      popcount   = std::nullopt;
    std::optional<InfoPolicy>  info_policy = std::nullopt;

    [[nodiscard]] auto operator==(const MeasurementStateFiniteIds &) const -> bool = default;
};

template<typename Value>
struct MeasurementStateFiniteCached {
    Value                     data{};
    MeasurementStateFiniteIds ids{};

    [[nodiscard]] const Value &value() const { return data; }
    [[nodiscard]] Value       &value() { return data; }
};

template<typename Scalar>
struct MeasurementsStateFinite {
    using RealScalar                                                             = decltype(std::real(std::declval<Scalar>()));
    using RealArrayX                                                             = Eigen::Array<RealScalar, Eigen::Dynamic, 1>;
    using RealArrayXX                                                            = Eigen::Array<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    std::optional<size_t>                       length                           = std::nullopt;
    std::optional<long>                         bond_mid                         = std::nullopt;
    std::optional<long>                         bond_dim                         = std::nullopt;
    std::optional<std::vector<long>>            bond_dimensions                  = std::nullopt;
    std::optional<RealScalar>                   norm                             = std::nullopt;
    std::optional<std::array<RealScalar, 3>>    spin_components                  = std::nullopt;
    std::optional<std::vector<double>>          truncation_errors                = std::nullopt;
    std::optional<RealScalar>                   entanglement_entropy_midchain    = std::nullopt;
    std::optional<RealScalar>                   entanglement_entropy_current     = std::nullopt;
    std::optional<std::vector<RealScalar>>      entanglement_entropies           = std::nullopt;
    std::optional<RealScalar>                   number_entropy_midchain          = std::nullopt;
    std::optional<RealScalar>                   number_entropy_current           = std::nullopt;
    std::optional<std::vector<RealScalar>>      number_entropies                 = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 2>> number_probabilities             = std::nullopt;
    std::optional<std::vector<RealScalar>>      renyi_2                          = std::nullopt;
    std::optional<std::vector<RealScalar>>      renyi_3                          = std::nullopt;
    std::optional<std::vector<RealScalar>>      renyi_4                          = std::nullopt;
    std::optional<std::vector<RealScalar>>      renyi_inf                        = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 1>> expectation_values_sx            = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 1>> expectation_values_sy            = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 1>> expectation_values_sz            = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 2>> correlation_matrix_sx            = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 2>> correlation_matrix_sy            = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 2>> correlation_matrix_sz            = std::nullopt;
    std::optional<RealScalar>                   structure_factor_x               = std::nullopt;
    std::optional<RealScalar>                   structure_factor_y               = std::nullopt;
    std::optional<RealScalar>                   structure_factor_z               = std::nullopt;
    std::optional<Eigen::Tensor<RealScalar, 1>> opdm_spectrum                    = std::nullopt;
    std::optional<Eigen::Tensor<Scalar, 2>>     opdm                             = std::nullopt;
    std::optional<InfoPolicy>                   info_policy                      = std::nullopt;
    std::optional<RealArrayXX>                  subsystem_entanglement_entropies = std::nullopt;
    std::optional<RealArrayXX>                  information_lattice              = std::nullopt;
    std::optional<RealArrayX>                   information_per_scale            = std::nullopt;
    std::optional<RealScalar>                   information_center_of_mass       = std::nullopt;
    std::optional<double>                       see_time = std::nullopt; /*! The time it took to calculate the last subsystem_entanglement_entropies */

    std::optional<MeasurementStateFiniteCached<Eigen::Tensor<Scalar, 2>>>     cached_opdm                             = std::nullopt;
    std::optional<MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 1>>> cached_opdm_spectrum                    = std::nullopt;
    std::optional<MeasurementStateFiniteCached<std::vector<RealScalar>>>      cached_number_entropies                 = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealArrayXX>>                  cached_subsystem_entanglement_entropies = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealArrayXX>>                  cached_information_lattice              = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealArrayX>>                   cached_information_per_scale            = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealScalar>>                   cached_information_center_of_mass       = std::nullopt;
    std::optional<MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>>> cached_correlation_matrix_sx            = std::nullopt;
    std::optional<MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>>> cached_correlation_matrix_sy            = std::nullopt;
    std::optional<MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>>> cached_correlation_matrix_sz            = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealScalar>>                   cached_structure_factor_x               = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealScalar>>                   cached_structure_factor_y               = std::nullopt;
    std::optional<MeasurementStateFiniteCached<RealScalar>>                   cached_structure_factor_z               = std::nullopt;

    [[nodiscard]] const MeasurementStateFiniteCached<Eigen::Tensor<Scalar, 2>>     *get_cached_opdm(const StateFinite<Scalar> &state,
                                                                                                       ModelType model_type) const;
    [[nodiscard]] const MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 1>> *get_cached_opdm_spectrum(const StateFinite<Scalar> &state,
                                                                                                                ModelType model_type) const;
    [[nodiscard]] const MeasurementStateFiniteCached<std::vector<RealScalar>>      *get_cached_number_entropies(const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealArrayXX>                  *get_cached_subsystem_entanglement_entropies(
        const StateFinite<Scalar> &state, const InfoPolicy &ip) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealArrayXX> *get_cached_information_lattice(const StateFinite<Scalar> &state,
                                                                                                   const InfoPolicy &ip) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealArrayX>  *get_cached_information_per_scale(const StateFinite<Scalar> &state,
                                                                                                     const InfoPolicy &ip) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealScalar>  *get_cached_information_center_of_mass(const StateFinite<Scalar> &state,
                                                                                                          const InfoPolicy &ip) const;
    [[nodiscard]] const MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>> *get_cached_correlation_matrix_sx(
        const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>> *get_cached_correlation_matrix_sy(
        const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<Eigen::Tensor<RealScalar, 2>> *get_cached_correlation_matrix_sz(
        const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealScalar> *get_cached_structure_factor_x(const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealScalar> *get_cached_structure_factor_y(const StateFinite<Scalar> &state) const;
    [[nodiscard]] const MeasurementStateFiniteCached<RealScalar> *get_cached_structure_factor_z(const StateFinite<Scalar> &state) const;

    void set_cached_opdm(Eigen::Tensor<Scalar, 2> value, const StateFinite<Scalar> &state, ModelType model_type);
    void set_cached_opdm_spectrum(Eigen::Tensor<RealScalar, 1> value, const StateFinite<Scalar> &state, ModelType model_type);
    void set_cached_number_entropies(std::vector<RealScalar> value, Eigen::Tensor<RealScalar, 2> probabilities, const StateFinite<Scalar> &state);
    void set_cached_subsystem_entanglement_entropies(RealArrayXX value, const StateFinite<Scalar> &state, const InfoPolicy &ip);
    void set_cached_information_lattice(RealArrayXX value, const StateFinite<Scalar> &state, const InfoPolicy &ip);
    void set_cached_information_per_scale(RealArrayX value, const StateFinite<Scalar> &state, const InfoPolicy &ip);
    void set_cached_information_center_of_mass(RealScalar value, const StateFinite<Scalar> &state, const InfoPolicy &ip);
    void set_cached_correlation_matrix_sx(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state);
    void set_cached_correlation_matrix_sy(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state);
    void set_cached_correlation_matrix_sz(Eigen::Tensor<RealScalar, 2> value, const StateFinite<Scalar> &state);
    void set_cached_structure_factor_x(RealScalar value, const StateFinite<Scalar> &state);
    void set_cached_structure_factor_y(RealScalar value, const StateFinite<Scalar> &state);
    void set_cached_structure_factor_z(RealScalar value, const StateFinite<Scalar> &state);

    private:
    template<typename Value>
    [[nodiscard]] static const MeasurementStateFiniteCached<Value> *get_cached_impl(const std::optional<MeasurementStateFiniteCached<Value>> &cache,
                                                                                     const MeasurementStateFiniteIds &ids,
                                                                                     bool allow_info_policy_compatibility = false);
    template<typename Value>
    static void set_cached_impl(std::optional<MeasurementStateFiniteCached<Value>> &cache, Value value, MeasurementStateFiniteIds ids);
    [[nodiscard]] static bool ids_match(const MeasurementStateFiniteIds &requested, const MeasurementStateFiniteIds &cached,
                                        bool allow_info_policy_compatibility);
    [[nodiscard]] static std::vector<size_t> get_mps_ids(const StateFinite<Scalar> &state);
    [[nodiscard]] static MeasurementStateFiniteIds get_ids_state(const StateFinite<Scalar> &state);
    [[nodiscard]] static MeasurementStateFiniteIds get_ids_opdm(const StateFinite<Scalar> &state, ModelType model_type);
    [[nodiscard]] static MeasurementStateFiniteIds get_ids_number_entropies(const StateFinite<Scalar> &state);
    [[nodiscard]] static MeasurementStateFiniteIds get_ids_info(const StateFinite<Scalar> &state, const InfoPolicy &ip);
};

#include "MeasurementsStateFinite.tmpl.h"
