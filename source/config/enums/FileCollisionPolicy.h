#pragma once
#include "config/enum_utils.h"

/*! What to do when the output file already exists */
enum class FileCollisionPolicy {
    RESUME, /*!< If finished -> exit, else resume simulation from the latest "FULL" storage state. Throw if none is found. */
    BACKUP, /*!< Backup the existing file by appending .bak, then start with a new file. */
    RENAME, /*!< Rename the current file by appending .# to avoid collision with existing. */
    REVIVE, /*!< Try RESUME, but do REPLACE on error instead of throwing */
    REPLACE /*!< Just erase/truncate the existing file and start from the beginning. */
};

template<> std::string_view    enum2sv(FileCollisionPolicy item) noexcept;
template<> FileCollisionPolicy sv2enum<FileCollisionPolicy>(std::string_view item);
