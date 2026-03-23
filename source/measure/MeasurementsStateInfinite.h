#pragma once
#include <complex>
#include <optional>
#include <utility>
template<typename Scalar>
struct MeasurementsStateInfinite {
    using RealScalar                               = decltype(std::real(std::declval<Scalar>()));
    std::optional<RealScalar> norm                 = std::nullopt;
    std::optional<long>       bond_dim             = std::nullopt;
    std::optional<RealScalar> entanglement_entropy = std::nullopt;
    std::optional<double>     truncation_error     = std::nullopt;
};
