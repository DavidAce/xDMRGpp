#include "log.h"
#include "debug/exceptions.h"
#include "io/spdlog.h"

void tools::Logger::enableTimestamp(const std::shared_ptr<spdlog::logger> &other_log) {
    if(other_log != nullptr) {
        //        other_log->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%n]%^[%=8l]%$ %v");
        other_log->set_pattern("[%Y-%m-%d %H:%M:%S][%n]%^[%=8l]%$ %v");
        other_log->trace("Enabled timestamp");
    }
}

void tools::Logger::disableTimestamp(const std::shared_ptr<spdlog::logger> &other_log) {
    if(other_log != nullptr) {
        other_log->set_pattern("[%n]%^[%=8l]%$ %v");
        other_log->trace("Disabled timestamp");
    }
}

template<typename levelType>
void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, levelType levelZeroToSix) {
    if constexpr(std::is_same_v<levelType, spdlog::level::level_enum>)
        other_log->set_level(levelZeroToSix);
    else if constexpr(std::is_integral_v<levelType>) {
        if(levelZeroToSix > 5) { throw except::runtime_error("Expected verbosity level integer in [0-5]. Got: {}", levelZeroToSix); }
        return tools::Logger::setLogLevel(other_log, static_cast<spdlog::level::level_enum>(levelZeroToSix));
    } else if constexpr(std::is_same_v<levelType, std::optional<size_t>>) {
        if(levelZeroToSix)
            return setLogLevel(other_log, levelZeroToSix.value());
        else
            return;
    } else if constexpr(std::is_same_v<levelType, std::optional<spdlog::level::level_enum>>) {
        if(levelZeroToSix)
            return setLogLevel(other_log, levelZeroToSix.value());
        else
            return;
    } else {
        throw except::runtime_error("Given wrong type for spdlog verbosity level");
    }
    //    other_log->info("Log verbosity level: {}   | trace:0 | debug:1 | info:2 | warn:3 | error:4 | critical:5 |", static_cast<int>(other_log->level()));
    //    other_log->debug("Log verbosity level: {}", static_cast<int>(other_log->level()));
}

template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, int levelZeroToSix);
template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, long levelZeroToSix);
template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, size_t levelZeroToSix);
template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, std::optional<size_t> levelZeroToSix);
template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, std::optional<spdlog::level::level_enum> levelZeroToSix);
template void tools::Logger::setLogLevel(const std::shared_ptr<spdlog::logger> &other_log, spdlog::level::level_enum levelZeroToSix);

size_t tools::Logger::getLogLevel(const std::shared_ptr<spdlog::logger> &other_log) {
    if(other_log != nullptr)
        return static_cast<size_t>(other_log->level());
    else
        return 2;
}

void tools::Logger::setLogger(std::shared_ptr<spdlog::logger> &other_log, const std::string &name, std::optional<size_t> levelZeroToSix,
                              std::optional<bool> timestamp) {
    if(spdlog::get(name) == nullptr)
        other_log = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
    else
        other_log = spdlog::get(name);
    other_log->set_pattern("[%n]%^[%=8l]%$ %v"); // Disabled timestamp is the default
    setLogLevel(other_log, levelZeroToSix);
    if(timestamp and timestamp.value()) enableTimestamp(other_log);
}

std::shared_ptr<spdlog::logger> tools::Logger::setLogger(const std::string &name, std::optional<size_t> levelZeroToSix, std::optional<bool> timestamp) {
    std::shared_ptr<spdlog::logger> other_log;

    if(spdlog::get(name) == nullptr) {
        other_log = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
    } else {
        other_log = spdlog::get(name);
    }
    tools::Logger::setLogLevel(other_log, levelZeroToSix);
    if(not timestamp or (timestamp and timestamp.value()))
        tools::Logger::enableTimestamp(other_log);
    else if(timestamp and not timestamp.value())
        tools::Logger::disableTimestamp(other_log);
    return other_log;
}

std::shared_ptr<spdlog::logger> tools::Logger::getLogger(const std::string &name) {
    auto log_found = spdlog::get(name);
    if(log_found == nullptr)
        throw except::runtime_error("spdlog: logger does not exist: {}", name);
    else
        return log_found;
}
