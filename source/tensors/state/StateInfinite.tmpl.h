#pragma once
#include "math/float.h"
#include "StateInfinite.h"
#include "tensors/site/mps/MpsSite.h"
template<typename Scalar>
template<typename T>
StateInfinite<Scalar>::StateInfinite(const StateInfinite<T> &other) noexcept {
    MPS_A                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteA());
    MPS_B                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteB());
    swapped                  = other.swapped;
    name                     = other.name;
    algo                     = other.algo;
    lowest_recorded_variance = other.lowest_recorded_variance;
    if constexpr(std::is_same_v<Scalar, T>) {
        cache        = other.cache;
        measurements = other.measurements;
    }
}

template<typename Scalar>
template<typename T>
StateInfinite<Scalar> &StateInfinite<Scalar>::operator=(const StateInfinite<T> &other) noexcept {
    if constexpr(std::is_same_v<Scalar, T>) {
        if(this == &other) return *this; // check for self-assignment
    }
    MPS_A                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteA());
    MPS_B                    = std::make_unique<MpsSite<Scalar>>(other.get_mps_siteB());
    swapped                  = other.swapped;
    name                     = other.name;
    algo                     = other.algo;
    lowest_recorded_variance = other.lowest_recorded_variance;
    if constexpr(std::is_same_v<Scalar, T>) {
        cache        = other.cache;
        measurements = other.measurements;
    }
    return *this;
}
