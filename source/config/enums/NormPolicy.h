#pragma once
#include "config/enum_utils.h"

/*! When to renormalize an MPS state */
enum class NormPolicy {
    ALWAYS,  /*!< Always renormalize after the operation */
    IFNEEDED /*!< Renormalize only if the norm has drifted */
};
template<> std::string_view enum2sv(NormPolicy item) noexcept;
template<> NormPolicy       sv2enum<NormPolicy>(std::string_view item);
