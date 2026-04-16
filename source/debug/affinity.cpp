#include "affinity.h"
#include "tools/common/log.h"

#include <algorithm>
#include <fmt/core.h>
#include <thread>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#if defined(__linux__)
    #include <sched.h>
#endif

namespace debug::affinity {
    std::optional<Status> query_status() {
#if defined(__linux__)
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if(sched_getaffinity(0, sizeof(mask), &mask) != 0) return std::nullopt;

        Status status;
        status.host_threads = std::thread::hardware_concurrency();
#if defined(_OPENMP)
        status.omp_threads = static_cast<unsigned int>(std::max(1, omp_get_max_threads()));
#endif
        for(int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if(CPU_ISSET(cpu, &mask)) status.allowed_cpus.push_back(cpu);

        status.restricted     = status.host_threads > 0 and status.allowed_cpus.size() < status.host_threads;
        status.oversubscribed = status.omp_threads > status.allowed_cpus.size();
        return status;
#else
        return std::nullopt;
#endif
    }

    std::string format_status(const Status &status) {
        auto out = fmt::format("affinity | allowed cpus {} / host threads {} | omp_max_threads {} |", status.allowed_cpus.size(), status.host_threads,
                               status.omp_threads);
        for(const auto cpu : status.allowed_cpus) out += fmt::format(" {}", cpu);
        return out;
    }

    std::vector<std::string> describe_pathologies(const Status &status) {
        std::vector<std::string> messages;
        if(status.restricted)
            messages.emplace_back(fmt::format("CPU affinity is restricted: this process sees {} logical CPUs out of {}. Benchmark results may not reflect "
                                              "full-machine scaling.",
                                              status.allowed_cpus.size(), status.host_threads));
        if(status.oversubscribed)
            messages.emplace_back(fmt::format("OpenMP oversubscription detected: omp_max_threads = {} but the current affinity mask contains only {} logical "
                                              "CPUs.",
                                              status.omp_threads, status.allowed_cpus.size()));
        return messages;
    }

    std::optional<std::string> format_openmp_placement() {
#if defined(_OPENMP) && defined(__linux__)
        std::vector<int> cpus(static_cast<std::size_t>(omp_get_max_threads()), -1);
#pragma omp parallel
        {
            cpus[static_cast<std::size_t>(omp_get_thread_num())] = sched_getcpu();
        }

        std::vector<int> unique_cpus = cpus;
        std::ranges::sort(unique_cpus);
        unique_cpus.erase(std::ranges::unique(unique_cpus).begin(), unique_cpus.end());

        auto out = fmt::format("openmp placement | threads {} | unique cpus {} |", cpus.size(), unique_cpus.size());
        for(std::size_t idx = 0; idx < cpus.size(); ++idx) out += fmt::format(" {}:{}", idx, cpus[idx]);
        return out;
#else
        return std::nullopt;
#endif
    }

    void log_sanity() {
        auto status = query_status();
        if(not status) return;

        auto pathologies = describe_pathologies(*status);
        if(pathologies.empty()) {
            tools::log->debug(format_status(*status));
            return;
        }

        tools::log->warn(format_status(*status));
        for(const auto &message : pathologies) tools::log->warn(message);
    }
}
