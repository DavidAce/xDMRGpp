#pragma once

#include "config/enum_utils.h"

/*! How Pauli terms are arranged when building randomized MPOs */
enum class RandomizerMode {
    SHUFFLE, /*!< Shuffle the available terms independently on each site */
    SELECT1, /*!< Choose one term independently on each site */
    ASIS     /*!< Keep the input order of the terms */
};

template<> std::string_view enum2sv(RandomizerMode item) noexcept;
template<> RandomizerMode   sv2enum<RandomizerMode>(std::string_view item);
