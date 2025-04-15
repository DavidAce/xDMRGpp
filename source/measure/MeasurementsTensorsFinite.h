#pragma once
#include <array>
#include <Eigen/src/Core/NumTraits.h>
#include <optional>
#include <vector>
template<typename Scalar>
struct MeasurementsTensorsFinite {
    using RealScalar                                    = typename Eigen::NumTraits<Scalar>::Real;
    std::optional<size_t>     length                    = std::nullopt;
    std::optional<RealScalar> energy                    = std::nullopt;
    std::optional<RealScalar> energy_variance           = std::nullopt;
    std::optional<RealScalar> energy_shift              = std::nullopt;
    std::optional<RealScalar> energy_minus_energy_shift = std::nullopt;
};
