#pragma once

#include "config/enums/GpuPolicy.h"
#include <cstddef>
#include <string>

namespace config::cuda {
    struct MemoryStatus {
        std::size_t free_bytes     = 0;
        std::size_t total_bytes    = 0;
        std::size_t usable_bytes   = 0;
        std::size_t required_bytes = 0;
        bool        fits           = false;
    };

    bool               compiled() noexcept;
    void               initialize();
    bool               available();
    MemoryStatus       query_memory(std::size_t required_bytes = 0);
    GpuPolicy          gpu_policy() noexcept;
    int                requested_gpu_id() noexcept;
    int                active_gpu_id();
    const std::string &description();
}
