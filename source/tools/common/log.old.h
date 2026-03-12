#pragma once
#include "io/spdlog.h"
#include <memory>
#include <optional>
#include <stdfloat>
namespace tools {
    inline std::shared_ptr<spdlog::logger> log;

    namespace Logger {
        extern void   enableTimestamp(const std::shared_ptr<spdlog::logger> &log);
        extern void   disableTimestamp(const std::shared_ptr<spdlog::logger> &log);
        extern size_t getLogLevel(const std::shared_ptr<spdlog::logger> &log);
        template<typename levelType>
        extern void setLogLevel(const std::shared_ptr<spdlog::logger> &log, levelType levelZeroToSix);
        extern void setLogger(std::shared_ptr<spdlog::logger> &log, const std::string &name, std::optional<size_t> levelZeroToSix = std::nullopt,
                              std::optional<bool> timestamp = std::nullopt);
        extern std::shared_ptr<spdlog::logger> setLogger(const std::string &name, std::optional<size_t> levelZeroToSix = std::nullopt,
                                                         std::optional<bool> timestamp = std::nullopt);
        extern std::shared_ptr<spdlog::logger> getLogger(const std::string &name);
    }

}