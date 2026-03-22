#pragma once
#include "config/enum_utils.h"

/*! How to handle finished_all when deciding whether to resume from an existing file */
enum class FileResumePolicy {
    FULL, /*!< Keep scanning the file and configuration before deciding whether to exit */
    FAST  /*!< Exit immediately when finished_all is detected */
};

template<> std::string_view enum2sv(FileResumePolicy item) noexcept;
template<> FileResumePolicy sv2enum<FileResumePolicy>(std::string_view item);
