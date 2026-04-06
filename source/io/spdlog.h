#pragma once
#include "fmt_custom.h"

#if !defined(SPDLOG_FMT_EXTERNAL)
    #define SPDLOG_FMT_EXTERNAL
#endif
#if !defined(SPDLOG_COMPILED_LIB)
    #if !defined(SPDLOG_HEADER_ONLY)
        #define SPDLOG_HEADER_ONLY
    #endif
    #pragma message "SPDLOG_COMPILED_LIB is not defined. Define it and link libspdlog to speed up compilation"
#endif

#include <spdlog/common.h>
#include <spdlog/logger.h>
