#pragma once
#include "../tools/finite/measure/infopolicy.h"
#include <array>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

template<typename Scalar>
struct MeasurementsStateFinite {
    using RealScalar                                                             = typename Eigen::NumTraits<Scalar>::Real;
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
    std::optional<double>                       information_center_of_mass       = std::nullopt;
    std::optional<double>                       see_time = std::nullopt; /*! The time it took to calculate the last subsystem_entanglement_entropies */
};
