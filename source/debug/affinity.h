#pragma once

#include <optional>
#include <string>
#include <vector>

namespace debug::affinity {
    struct Status {
        std::vector<int> allowed_cpus;
        unsigned int     host_threads   = 0;
        unsigned int     omp_threads    = 1;
        bool             oversubscribed = false;
        bool             restricted     = false;
    };

    std::optional<Status>      query_status();
    std::string                format_status(const Status &status);
    std::vector<std::string>   describe_pathologies(const Status &status);
    std::optional<std::string> format_openmp_placement();
    void                       log_sanity();
}
