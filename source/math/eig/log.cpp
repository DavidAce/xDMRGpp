#include "log.h"

void eig::setLevel(spdlog::level::level_enum level) { log->set_level(level); }
void eig::setLevel(size_t level) { log->set_level(static_cast<spdlog::level::level_enum>(level)); }
void eig::setTimeStamp(std::string_view stamp) { log->set_pattern(std::string(stamp)); }
