#pragma once
#include "config/enum_utils.h"

/*! Bitflags controlling cache access for reusable data */
enum class CachePolicy {
    NONE  = 0,   /*!< Do not use cache */
    READ  = 1,   /*!< Read only */
    WRITE = 2,   /*!< Write only */
    allow_bitops /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(CachePolicy item) noexcept;
template<> CachePolicy      sv2enum<CachePolicy>(std::string_view item);
template<> std::string      flag2str(const CachePolicy &item) noexcept;
