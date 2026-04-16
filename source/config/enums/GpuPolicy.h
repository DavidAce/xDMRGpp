#pragma once
#include "config/enum_utils.h"

/*! Policy for optional GPU-backed tensor contractions */
enum class GpuPolicy {
    ON,  /*!< Require a usable GPU and fail during startup otherwise */
    OFF, /*!< Disable GPU contractions even if a usable GPU is present */
    TRY  /*!< Use a GPU when one is usable, otherwise continue without it */
};
template<> std::string_view enum2sv(GpuPolicy item) noexcept;
template<> GpuPolicy        sv2enum<GpuPolicy>(std::string_view item);
