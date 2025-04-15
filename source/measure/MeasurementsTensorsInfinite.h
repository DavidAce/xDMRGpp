#pragma once
#include <array>
#include <optional>
#include <vector>
#include <Eigen/src/Core/NumTraits.h>

template<typename Scalar>
struct MeasurementsTensorsInfinite {
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    std::optional<size_t> length                       = std::nullopt;
    std::optional<RealScalar> energy_mpo                   = std::nullopt;
    std::optional<RealScalar> energy_per_site_mpo          = std::nullopt;
    std::optional<RealScalar> energy_variance_mpo          = std::nullopt;
    std::optional<RealScalar> energy_per_site_ham          = std::nullopt;
    std::optional<RealScalar> energy_per_site_mom          = std::nullopt;
    std::optional<RealScalar> energy_variance_per_site_mpo = std::nullopt;
    std::optional<RealScalar> energy_variance_per_site_ham = std::nullopt;
    std::optional<RealScalar> energy_variance_per_site_mom = std::nullopt;
};
