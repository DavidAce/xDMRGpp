#pragma once

#include "io/fmt_custom.h"
#include <stdexcept>

namespace except {
    class runtime_error : public std::runtime_error {
        public:
        using std::runtime_error::runtime_error;
        template<typename... Args>
        runtime_error(fmt::format_string<fmw::adapted_arg_t<Args>...> fs, Args &&...args)
            : std::runtime_error(fmt::format(fs, fmw::wrap(std::forward<Args>(args))...)) {}
    };

    class logic_error : public std::logic_error {
        public:
        using std::logic_error::logic_error;
        template<typename... Args>
        logic_error(fmt::format_string<fmw::adapted_arg_t<Args>...> fs, Args &&...args)
            : std::logic_error(fmt::format(fs, fmw::wrap(std::forward<Args>(args))...)) {}
    };

    class range_error : public std::range_error {
        public:
        using std::range_error::range_error;
        template<typename... Args>
        range_error(fmt::format_string<fmw::adapted_arg_t<Args>...> fs, Args &&...args)
            : std::range_error(fmt::format(fs, fmw::wrap(std::forward<Args>(args))...)) {}
    };

    class state_error : public except::runtime_error {
        // Used for signaling that no resumable state was found
        using except::runtime_error::runtime_error;
    };

    class file_error : public except::runtime_error {
        // Used for signaling that the existing file is corrupted
        using except::runtime_error::runtime_error;
    };

    class load_error : public except::runtime_error {
        // Used for signaling an error when loading an existing file
        using except::runtime_error::runtime_error;
    };

    class resume_error : public except::runtime_error {
        // Used to signal that an error ocurred when trying to resume a simulation
        using except::runtime_error::runtime_error;
    };

}
