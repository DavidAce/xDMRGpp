#pragma once
#include "config/enum_utils.h"

/*! Selects the top-level algorithm family */
enum class AlgorithmType : int {
    iDMRG, /*!< Infinite-system density matrix renormalization group */
    fDMRG, /*!< Finite-system density matrix renormalization group */
    xDMRG, /*!< Excited-state density matrix renormalization group */
    iTEBD, /*!< Infinite time-evolving block decimation */
    fLBIT, /*!< Finite-system l-bit evolution */
    ANY    /*!< Wildcard that matches any algorithm type */
};

template<> std::string_view enum2sv(AlgorithmType item) noexcept;
template<> AlgorithmType    sv2enum<AlgorithmType>(std::string_view item);
