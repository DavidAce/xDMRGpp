#pragma once
#include "io/spdlog.h"
#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stdfloat>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tools::logfmt {
    template<typename T>
    using bare_t = std::decay_t<T>;

    template<typename T>
    struct complex_scalar_type {
        static constexpr bool value = false;
        using type                  = void;
    };

    template<typename T>
    struct complex_scalar_type<std::complex<T>> {
        static constexpr bool value = true;
        using type                  = T;
    };

    template<typename T>
    concept native_fp_scalar = std::same_as<bare_t<T>, float> || std::same_as<bare_t<T>, double> || std::same_as<bare_t<T>, long double>;

    template<typename T>
    concept complex_floating_scalar = complex_scalar_type<bare_t<T>>::value && std::floating_point<typename complex_scalar_type<bare_t<T>>::type>;

    template<typename T>
    concept extended_fp_scalar = std::floating_point<bare_t<T>> && !native_fp_scalar<T>;

    template<typename T>
    concept message_like = std::convertible_to<T, std::string_view>;

    template<typename T>
    concept contiguous_storage = requires(const bare_t<T> &value) {
        { value.data() };
        requires std::is_pointer_v<decltype(value.data())>;
        { value.size() } -> std::convertible_to<std::size_t>;
    };

    template<typename T>
    requires complex_floating_scalar<T>
    using complex_value_t = typename bare_t<T>::value_type;

    template<typename T>
    requires contiguous_storage<T>
    using range_value_t = std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const bare_t<T> &>().data())>>;

    template<typename T>
    concept contiguous_floating_range = contiguous_storage<T> && !sfinae::is_text_v<bare_t<T>> && std::floating_point<range_value_t<T>>;

    template<typename T>
    inline constexpr bool needs_fp_wrapper_v = [] {
        if constexpr(extended_fp_scalar<T>) {
            return true;
        } else if constexpr(complex_floating_scalar<T>) {
            return extended_fp_scalar<complex_value_t<T>>;
        } else {
            return false;
        }
    }();

    template<typename T>
    inline constexpr bool needs_fv_wrapper_v = [] {
        if constexpr(contiguous_floating_range<T>) {
            return true;
        } else {
            return false;
        }
    }();

    // The logger only auto-wraps the known problematic numeric domain:
    // extended floating-point scalars and contiguous floating-point ranges.
    // Ranges always go through fv so call sites can keep using fv-style
    // element formatting such as "{::+9.6f}" without explicit wrappers.
    // Everything else is forwarded unchanged and must already be formattable by fmt/spdlog.
    template<typename T>
    struct adapted_arg {
        using type = T;
    };

    template<typename T>
    requires needs_fp_wrapper_v<T>
    struct adapted_arg<T> {
        using type = fp<bare_t<T>>;
    };

    template<typename T>
    requires(!needs_fp_wrapper_v<T> && needs_fv_wrapper_v<T>)
    struct adapted_arg<T> {
        using type = fv<range_value_t<T>>;
    };

    template<typename T>
    using adapted_arg_t = typename adapted_arg<T>::type;

    // Keep the wrapper construction in one place so the formatting policy is easy to audit.
    template<typename T>
    requires needs_fp_wrapper_v<T>
    constexpr auto adapt_arg(T &&value) {
        static_assert(extended_fp_scalar<T> || complex_floating_scalar<T>,
                      "fp(...) adaptation is reserved for floating-point scalars and std::complex<floating-point>.");
        return fp<bare_t<T>>(std::forward<T>(value));
    }

    template<typename T>
    requires(!needs_fp_wrapper_v<T> && needs_fv_wrapper_v<T>)
    constexpr auto adapt_arg(T &&value) {
        static_assert(contiguous_floating_range<T>, "fv(...) adaptation requires a non-text contiguous range with data() and size().");
        return fv<range_value_t<T>>(value);
    }

    template<typename T>
    requires(!needs_fp_wrapper_v<T> && !needs_fv_wrapper_v<T>)
    constexpr decltype(auto) adapt_arg(T &&value) {
        return std::forward<T>(value);
    }

    static_assert(std::same_as<adapted_arg_t<double>, double>, "Native floating-point types should be forwarded directly to fmt.");
    static_assert(std::same_as<adapted_arg_t<long double>, long double>, "Native floating-point types should be forwarded directly to fmt.");
    static_assert(std::same_as<adapted_arg_t<std::complex<double>>, std::complex<double>>,
                  "Native complex floating-point types should be forwarded directly to fmt.");
    static_assert(!contiguous_floating_range<std::string>, "Text types must not be treated as numeric ranges.");
#if defined(__STDCPP_FLOAT128_T__)
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
        void format_impl(spdlog::level::level_enum lvl, spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            require_logger().log(lvl, fmt, logfmt::adapt_arg(std::forward<Args>(args))...);
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
        void trace(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::trace, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void debug(Msg &&msg) const {
            message_impl(spdlog::level::debug, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void debug(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::debug, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void info(Msg &&msg) const {
            message_impl(spdlog::level::info, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void info(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::info, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void warn(Msg &&msg) const {
            message_impl(spdlog::level::warn, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void warn(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::warn, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void error(Msg &&msg) const {
            message_impl(spdlog::level::err, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void error(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::err, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void critical(Msg &&msg) const {
            message_impl(spdlog::level::critical, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void critical(spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
            format_impl(spdlog::level::critical, fmt, std::forward<Args>(args)...);
        }

        template<typename Msg>
        requires logfmt::message_like<Msg>
        void log(spdlog::level::level_enum lvl, Msg &&msg) const {
            message_impl(lvl, std::forward<Msg>(msg));
        }

        template<typename... Args>
        requires(sizeof...(Args) > 0)
        void log(spdlog::level::level_enum lvl, spdlog::format_string_t<logfmt::adapted_arg_t<Args>...> fmt, Args &&...args) const {
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
