#pragma once

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
    int                requested_device() noexcept;
    int                active_device();
    const std::string &description();
}
