#pragma once
#include "config/enum_utils.h"

/*! Whether cached edge environments are current */
enum class EdgeStatus {
    STALE, /*!< The cached edge data needs to be rebuilt */
    FRESH  /*!< The cached edge data is current */
};

template<> std::string_view enum2sv(EdgeStatus item) noexcept;
template<> EdgeStatus       sv2enum<EdgeStatus>(std::string_view item);
