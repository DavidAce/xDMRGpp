#pragma once
#include <array>
#include <optional>
#include <vector>
template<typename Scalar>
struct MeasurementsTensorsFinite {
    using RealScalar                                    = decltype(std::real(std::declval<Scalar>()));
    std::optional<size_t>     length                    = std::nullopt;
    std::optional<RealScalar> energy                    = std::nullopt;
    std::optional<RealScalar> energy_variance           = std::nullopt;
    std::optional<RealScalar> energy_shift              = std::nullopt;
    std::optional<RealScalar> energy_minus_energy_shift = std::nullopt;
};
