#pragma once
#include <array>
#include <optional>
#include <vector>

template<typename Scalar>
struct MeasurementsTensorsInfinite {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
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
