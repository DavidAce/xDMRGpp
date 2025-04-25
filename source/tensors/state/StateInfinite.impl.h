#pragma once
#include "StateInfinite.h"
#include "math/float.h"
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
// template StateInfinite<fp32>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<fp32>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<fp32>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp32>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp32>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp32>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<fp64>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
// template StateInfinite<fp64>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<fp64>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp64>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp64>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp64>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<fp128>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<fp128>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
// template StateInfinite<fp128>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp128>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp128>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp128>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx32>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx32>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx32>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
// template StateInfinite<cx32>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<cx32>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<cx32>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx64>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx64>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx64>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<cx64>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
// template StateInfinite<cx64>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<cx64>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx128>::StateInfinite(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx128>::StateInfinite(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx128>::StateInfinite(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<cx128>::StateInfinite(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<cx128>::StateInfinite(const StateInfinite<cx64> &other) noexcept;
// template StateInfinite<cx128>::StateInfinite(const StateInfinite<cx128> &other) noexcept;

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
// template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp32> &StateInfinite<fp32>::operator=(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<fp32> &other) noexcept;
// template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp64> &StateInfinite<fp64>::operator=(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<fp64> &other) noexcept;
// template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<fp128> &StateInfinite<fp128>::operator=(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<fp128> &other) noexcept;
// template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<cx32> &StateInfinite<cx32>::operator=(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<cx32> &other) noexcept;
// template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<cx64> &other) noexcept;
template StateInfinite<cx64> &StateInfinite<cx64>::operator=(const StateInfinite<cx128> &other) noexcept;

template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<fp32> &other) noexcept;
template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<fp64> &other) noexcept;
template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<fp128> &other) noexcept;
template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<cx32> &other) noexcept;
template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<cx64> &other) noexcept;
// template StateInfinite<cx128> &StateInfinite<cx128>::operator=(const StateInfinite<cx128> &other) noexcept;
