#pragma once
#include "io/spdlog.h"
#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#if defined(DMRG_USE_FLOAT128)
    #include <stdfloat>
#endif

namespace tools::logfmt {
    template<typename T>
    concept message_like = std::convertible_to<T, std::string_view>;

    template<typename T>
    using adapted_arg_t = fmw::adapted_arg_t<T>;

    template<typename T>
    constexpr decltype(auto) adapt_arg(T &&value) {
        return fmw::wrap(std::forward<T>(value));
    }

    static_assert(std::same_as<adapted_arg_t<double>, double>, "Native floating-point types should be forwarded directly to fmt.");
    static_assert(std::same_as<adapted_arg_t<long double>, long double>, "Native floating-point types should be forwarded directly to fmt.");
    static_assert(std::same_as<adapted_arg_t<std::complex<double>>, fp<std::complex<double>>>,
                  "Native complex floating-point types should be wrapped in fp before reaching fmt.");
    static_assert(std::same_as<adapted_arg_t<std::filesystem::path>, fmtwrap::path_view>,
                  "filesystem::path should be adapted through the lightweight path formatter.");
    static_assert(std::same_as<adapted_arg_t<std::optional<double>>, fmtwrap::optional_view<double>>,
                  "std::optional should be adapted through the lightweight optional formatter.");
    static_assert(std::same_as<adapted_arg_t<std::vector<size_t>>, fmtwrap::listed_view<std::vector<size_t>>>,
                  "Iterable ranges should be adapted through the lightweight range formatter.");
    static_assert(!fmtwrap::contiguous_floating_range<std::string>, "Text types must not be treated as numeric ranges.");
#if defined(DMRG_USE_FLOAT128) && defined(__STDCPP_FLOAT128_T__)
    static_assert(std::same_as<adapted_arg_t<std::float128_t>, fp<std::float128_t>>,
                  "Extended floating-point types must be wrapped in fp before reaching fmt.");
    static_assert(std::same_as<adapted_arg_t<std::complex<std::float128_t>>, fp<std::complex<std::float128_t>>>,
                  "Complex extended floating-point types must be wrapped in fp before reaching fmt.");
#endif
}

namespace tools {
    // LoggerHandle preserves the existing tools::log->info(...) call style while
    // giving us a place to normalize problematic arguments before spdlog sees them.
    class LoggerHandle {
        private:
        std::shared_ptr<spdlog::logger> impl_;

        [[nodiscard]] spdlog::logger &require_logger() const {
            if(!impl_) throw std::runtime_error("tools::log is not initialized");
            return *impl_;
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void message_impl(spdlog::level::level_enum lvl, Msg &&msg) const {
            require_logger().log(lvl, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void format_impl(spdlog::level::level_enum lvl, fmt::string_view fmt_sv, Args &&...args) const {
            auto rendered = fmt::format(fmt::runtime(fmt_sv), logfmt::adapt_arg(std::forward<Args>(args))...);
            require_logger().log(lvl, rendered);
        }

        public:
        LoggerHandle() = default;
        LoggerHandle(std::nullptr_t) noexcept {}
        LoggerHandle(std::shared_ptr<spdlog::logger> impl) noexcept : impl_(std::move(impl)) {}

        LoggerHandle &operator=(std::shared_ptr<spdlog::logger> impl) noexcept {
            impl_ = std::move(impl);
            return *this;
        }

        LoggerHandle &operator=(std::nullptr_t) noexcept {
            impl_.reset();
            return *this;
        }

        // Return this wrapper rather than the underlying spdlog::logger so existing
        // call sites can keep using tools::log->info(...).
        LoggerHandle       *operator->() noexcept { return this; }
        const LoggerHandle *operator->() const noexcept { return this; }

        [[nodiscard]] bool     operator==(std::nullptr_t) const noexcept { return impl_ == nullptr; }
        [[nodiscard]] bool     operator!=(std::nullptr_t) const noexcept { return impl_ != nullptr; }
        [[nodiscard]] explicit operator bool() const noexcept { return impl_ != nullptr; }

        [[nodiscard]] spdlog::logger                        *get() const noexcept { return impl_.get(); }
        [[nodiscard]] const std::shared_ptr<spdlog::logger> &shared() const noexcept { return impl_; }

        [[nodiscard]] const std::string        &name() const { return require_logger().name(); }
        [[nodiscard]] spdlog::level::level_enum level() const { return require_logger().level(); }
        void                                    set_level(spdlog::level::level_enum lvl) const { require_logger().set_level(lvl); }
        void                                    set_pattern(std::string pattern) const { require_logger().set_pattern(std::move(pattern)); }

        void enable_backtrace(size_t n_messages = 32) const { require_logger().enable_backtrace(n_messages); }
        void disable_backtrace() const { require_logger().disable_backtrace(); }
        void dump_backtrace() const { require_logger().dump_backtrace(); }

        template<typename Handler>
        requires std::invocable<Handler, const std::string &>
        void set_error_handler(Handler &&handler) const {
            require_logger().set_error_handler(std::forward<Handler>(handler));
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void trace(Msg &&msg) const {
            message_impl(spdlog::level::trace, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void trace(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::trace, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void debug(Msg &&msg) const {
            message_impl(spdlog::level::debug, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void debug(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::debug, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void info(Msg &&msg) const {
            message_impl(spdlog::level::info, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void info(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::info, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void warn(Msg &&msg) const {
            message_impl(spdlog::level::warn, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void warn(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::warn, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void error(Msg &&msg) const {
            message_impl(spdlog::level::err, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void error(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::err, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void critical(Msg &&msg) const {
            message_impl(spdlog::level::critical, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void critical(fmt::string_view fmt, Args &&...args) const {
            format_impl(spdlog::level::critical, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void log(spdlog::level::level_enum lvl, Msg &&msg) const {
            message_impl(lvl, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void log(spdlog::level::level_enum lvl, fmt::string_view fmt, Args &&...args) const {
            format_impl(lvl, fmt, std::forward<Args>(args)...);
        }
    };

    inline LoggerHandle log;

    namespace Logger {
        extern void   enableTimestamp(const std::shared_ptr<spdlog::logger> &log);
        extern void   disableTimestamp(const std::shared_ptr<spdlog::logger> &log);
        extern size_t getLogLevel(const std::shared_ptr<spdlog::logger> &log);
        template<typename levelType>
        extern void setLogLevel(const std::shared_ptr<spdlog::logger> &log, levelType levelZeroToSix);
        extern void setLogger(std::shared_ptr<spdlog::logger> &log, const std::string &name, std::optional<size_t> levelZeroToSix = std::nullopt,
                              std::optional<bool> timestamp = std::nullopt);
        extern void setLogger(LoggerHandle &log, const std::string &name, std::optional<size_t> levelZeroToSix = std::nullopt,
                              std::optional<bool> timestamp = std::nullopt);
        extern std::shared_ptr<spdlog::logger> setLogger(const std::string &name, std::optional<size_t> levelZeroToSix = std::nullopt,
                                                         std::optional<bool> timestamp = std::nullopt);
        extern std::shared_ptr<spdlog::logger> getLogger(const std::string &name);
    }

}
