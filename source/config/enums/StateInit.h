#pragma once

#include "config/enum_utils.h"

/*! Strategies for constructing the initial MPS state */
enum class StateInit {
    RANDOM_PRODUCT_STATE,          /*!< Random product state */
    RANDOM_ENTANGLED_STATE,        /*!< Random entangled state */
    RANDOMIZE_PREVIOUS_STATE,      /*!< Randomize the previously loaded state */
    MIDCHAIN_SINGLET_NEEL_STATE,   /*!< Neel state with a singlet at the chain center */
    PRODUCT_STATE_ALIGNED,         /*!< Product state aligned along a chosen axis */
    PRODUCT_STATE_DOMAIN_WALL,     /*!< Product state with a domain wall */
    PRODUCT_STATE_NEEL,            /*!< Neel product state */
    PRODUCT_STATE_NEEL_SHUFFLED,   /*!< Shuffled Neel product state */
    PRODUCT_STATE_NEEL_DISLOCATED, /*!< Dislocated Neel product state */
    PRODUCT_STATE_PATTERN,         /*!< Product state following the configured spin pattern */
    SUM_OF_RANDOM_PRODUCT_STATES   /*!< Normalized sum of random product states */
};

template<> std::string_view enum2sv(StateInit item) noexcept;
template<> StateInit        sv2enum<StateInit>(std::string_view item);
