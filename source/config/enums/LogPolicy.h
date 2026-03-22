#pragma once
#include "config/enum_utils.h"

/*! How aggressively optional information should be logged */
enum class LogPolicy {
    SILENT, /*!< Never log */
    DEBUG,  /*!< Log on debug runs */
    VERBOSE /*!< Always log */
};
template<> std::string_view enum2sv(LogPolicy item) noexcept;
template<> LogPolicy        sv2enum<LogPolicy>(std::string_view item);
