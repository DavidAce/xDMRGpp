#pragma once

#include <complex>
#include <optional>
#include <utility>

template<typename Scalar>
struct MeasurementsOverride {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    // Optional HDF5 measurement-table values supplied by the algorithm when the nominal measurement is not the value we want to report.
    std::optional<RealScalar> energy                 = std::nullopt;
    std::optional<RealScalar> energy_variance        = std::nullopt;
    std::optional<RealScalar> energy_variance_lowest = std::nullopt;
};
