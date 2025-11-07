#include "info.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <ctime>
#include <fstream>
#include <unistd.h>

std::string debug::datetime() {
    char        buf[32] = {0};
    std::time_t now     = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

std::string debug::hostname() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    return hostname;
}

std::string debug::cpu_info(std::string_view info) {
    std::ifstream    filestream("/proc/cpuinfo");
    std::string      line;
    std::string_view sep = ": ";
    while(std::getline(filestream, line)) {
        if(line.rfind(info, 0) == 0) { // Starts with info
            auto pos_colon = line.find(sep);
            return line.substr(pos_colon + sep.size());
        }
    }
    return {};
}

double debug::mem_usage_in_mb(std::string_view info) {
    std::ifstream    filestream("/proc/self/status");
    std::string      line;
    std::string_view sep = ": ";
    while(std::getline(filestream, line)) {
        if(line.rfind(info, 0) == 0) { // Starts with info
            auto pos_colon = line.find(sep);
            auto value_str = line.substr(pos_colon + sep.size());
            value_str.erase(std::remove_if(value_str.begin(), value_str.end(), [](auto const &c) -> bool { return not std::isdigit(c); }), value_str.end());
            // Extract the number
            long long value = -1;
            try {
                value = std::stoll(value_str);
            } catch(const std::exception &ex) {
                std::fprintf(stderr, "Could not read mem usage from /proc/self/status: Failed to parse string [%s]: %s", value_str.c_str(), ex.what());
            }
            // Now we have the value in kb
            return static_cast<double>(value) / 1024.0;
        }
    }
    return -1.0;
}

double debug::mem_rss_in_mb() { return mem_usage_in_mb("VmRSS"); }
double debug::mem_hwm_in_mb() { return mem_usage_in_mb("VmHWM"); }
double debug::mem_vm_in_mb() { return mem_usage_in_mb("VmPeak"); }

void debug::print_mem_usage() {
    std::printf("%-30s%10.1f MB\n", "Memory RSS", mem_rss_in_mb());
    std::printf("%-30s%10.1f MB\n", "Memory Peak", mem_hwm_in_mb());
    std::printf("%-30s%10.1f MB\n", "Memory Vm", mem_vm_in_mb());
}
