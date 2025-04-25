#pragma once

#include "cast.h"
#include "float.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <iterator>
#include <numeric>
#include <vector>

#if defined(DMRG_USE_QUADMATH)
    #include <quadmath.h>
#endif

/*!
 *  \namespace num
 *  \brief Small convenience-type num functions, like modulo
 *  \tableofcontents
 */

namespace num {
#if defined(NDEBUG)
    static constexpr bool ndebug = true;
#else
    static constexpr bool ndebug = false;
#endif
    namespace internal {
        template<typename T>
        concept has_value_type_v = requires { typename T::value_type; };

        template<typename T>
        struct is_reference_wrapper : std::false_type {};

        template<typename T>
        struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

        template<typename T>
        concept is_float128 =
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
            std::same_as<T, fp128>;
#else
            false;
#endif

        template<typename T>
        concept floating_point = is_float128<T> || std::floating_point<T>;

        template<typename T>
        concept is_arithmetic_v = num::internal::floating_point<T> || std::integral<T>;

        template<typename T>
        auto pow(T b, T e) {
            using std::pow;
            return pow(b, e);
        }
        template<typename T>
        auto log(T v) {
            using std::log;
            return log(v);
        }
        template<typename T>
        auto log10(T v) {
            using std::log10;
            return log10(v);
        }
        template<typename T>
        auto isnan(T v) {
            using std::isnan;
            return isnan(v);
        }
        template<typename T>
        auto isinf(T v) {
            using std::isinf;
            return isinf(v);
        }

        template<typename T>
        auto floor(T v) {
            using std::floor;
            return floor(v);
        }
        template<typename T>
        auto ceil(T v) {
            using std::ceil;
            return ceil(v);
        }
    }

    template<std::floating_point A, std::floating_point B>
    constexpr bool lt(A a, B b) {
        using CT = std::common_type_t<A, B>;
        return CT{a} < CT{b}; // CT{…} is value-initialization + conversion
    }
    template<std::floating_point A, std::floating_point B>
    constexpr bool leq(A a, B b) {
        using CT = std::common_type_t<A, B>;
        return CT{a} <= CT{b};
    }
    template<std::floating_point A, std::floating_point B>
    constexpr bool gt(A a, B b) {
        using CT = std::common_type_t<A, B>;
        return CT{a} > CT{b};
    }
    template<std::floating_point A, std::floating_point B>
    constexpr bool geq(A a, B b) {
        using CT = std::common_type_t<A, B>;
        return CT{a} > CT{b};
    }

    template<typename out_t, typename ContainerT, typename Func>
    std::vector<out_t> cast(const ContainerT &x, Func &&f) {
        std::vector<out_t> y;
        std::transform(x.data(), x.data() + x.size(), std::back_inserter(y), std::forward<Func>(f));
        return y;
    }

    /*! \brief MatLab-style modulo operator
     *   \param x first number
     *   \param y second number
     *   \return modulo of x and y. Example, <code> mod(7,2)  = 1 </code> but <code> mod(-0.5,10)  = 9.5 </code>, instead of <code> -0.5 </code>  as given by
     * x%y.
     */
    template<typename T>
    [[nodiscard]] inline T mod(const T x, const T y) {
        if constexpr(!ndebug)
            if(y == 0) throw std::logic_error("num::mod(x,y): divisor y == 0");
        if constexpr(std::is_integral_v<T>) {
            if constexpr(std::is_unsigned_v<T>)
                return x >= y ? x % y : x;
            else { return x >= y ? x % y : (x < 0 ? (x % y + y) % y : x); }

        }
        //            return (x % y + y) % y;
        else
            return std::fmod((std::fmod(x, y) + y), y);
    }

    /*! \brief Similar to mod but faster for use with periodic boundary condition
     *   \param x first number
     *   \param y second number
     *   \return modulo of x and y. Example, <code> mod(7,2)  = 1 </code> but <code> mod(-0.5,10)  = 9.5 </code>, instead of <code> -0.5 </code>  as given by
     * x%y.
     */
    template<typename T>
    [[nodiscard]] inline T pbc(const T x, const T y) {
        if constexpr(!ndebug)
            if(y == 0) throw std::logic_error("num::pbc(x,y): divisor y == 0");
        if constexpr(std::is_signed_v<T>) {
            if(x >= 0 and x < y) return x;
            if(x < 0 and x >= -2 * y) return x + y;
        } else {
            if(x < y) return x;
        }
        if(x >= y and x < 2 * y) return x - y;
        return num::mod(x, y);
    }

    template<typename T>
    [[nodiscard]] int sign(const T val) noexcept {
        if(val > 0) return +1;
        if(val < 0) return -1;
        return 0;
    }

    template<typename T>
    [[nodiscard]] bool between(const T &value, const T &low, const T &high) noexcept {
        return value >= low and value <= high;
    }

    /*! \brief Python-style range generator, i.e. not-including "last"
     *   \return Range of T's. Example, <code> range(0,8,2) </code> gives a std::vector<int>: <code> [0,2,4,6] </code>
     */
    namespace internal {
        template<typename TA, typename TB>
        using int_or_dbl = typename std::conditional<std::is_floating_point_v<TA> or std::is_floating_point_v<TB>, double, int>::type;
    }

    template<typename T = int, typename T1, typename T2, typename T3 = internal::int_or_dbl<T1, T2>>
    [[nodiscard]] std::vector<T> range(T1 first, T2 last, T3 step = static_cast<T3>(1)) noexcept {
        if(step == 0) return {static_cast<T>(first)};
        using TC = std::common_type_t<T, T1, T2, T3>;
        auto num_steps =
            safe_cast<size_t>(std::abs((static_cast<double>(last) - static_cast<double>(first) + static_cast<double>(step)) / static_cast<double>(step)));
        std::vector<T> vec;
        vec.reserve(num_steps);
        auto val = static_cast<TC>(first);
        if(static_cast<TC>(first) < static_cast<TC>(last)) {
            while(static_cast<TC>(val) < static_cast<TC>(last)) {
                vec.emplace_back(val);
                val += std::cmp_less(step, 0) ? -static_cast<TC>(step) : static_cast<TC>(step);
            }
        } else {
            while(static_cast<TC>(val) > static_cast<TC>(last)) {
                vec.emplace_back(val);
                val -= static_cast<TC>(step) < static_cast<TC>(0) ? -static_cast<TC>(step) : static_cast<TC>(step);
            }
        }
        if constexpr(std::is_signed_v<T3>) {
            if(step < 0) return {vec.rbegin(), vec.rend()};
        }
        return vec;
    }

    /*! \brief MatLab-style linearly spaced array
     *   \param num number of linearly spaced values
     *   \param a first value in range
     *   \param b last value in range
     *   \return std::vector<T2>. Example,  <code> Linspaced(5,1,5) </code> gives a std::vector<int>: <code> [1,2,3,4,5] </code>
     */

    template<typename TA, typename TB>
    [[nodiscard]] auto LinSpaced(std::size_t N, TA a, TB b) -> std::vector<std::common_type_t<double, TA, TB>> {
        using T            = std::common_type_t<double, TA, TB>;
        T              h   = (static_cast<T>(b) - static_cast<T>(a)) / static_cast<T>(N - 1);
        T              val = static_cast<T>(a);
        std::vector<T> xs(N);
        for(auto &x : xs) {
            x = val;
            val += h;
        }
        xs.front() = static_cast<T>(a);
        xs.back()  = static_cast<T>(b);
        return xs;
    }

    template<typename TA, typename TB>
    [[nodiscard]] auto LogSpaced(std::size_t N, TA a, TB b, int base = 10, int keep_digits = -1) -> std::vector<std::common_type_t<double, TA, TB>> {
        if(a <= 0) throw std::range_error("a must be positive");
        if(b <= 0) throw std::range_error("b must be positive");
        using T               = std::common_type_t<double, TA, TB>;
        T              baseT  = static_cast<T>(base);
        T              loga   = internal::log(a) / internal::log(baseT);
        T              logb   = internal::log(b) / internal::log(baseT);
        T              h      = (logb - loga) / static_cast<T>(N - 1);
        T              factor = internal::pow(baseT, h);
        T              val    = internal::pow(baseT, loga);
        std::vector<T> xs(N);
        for(auto &x : xs) {
            x   = val;
            T s = internal::ceil(internal::log10(x)); // Base 10 exponent of x
            if(keep_digits > 0 and s > keep_digits) {
                T t = internal::pow(static_cast<T>(10), s - keep_digits); // Order of magnitude to truncate
                x   = internal::floor(x / t) * t;
            }
            if(internal::isnan(x) or internal::isinf(x)) throw std::logic_error("Invalid value in logspaced");
            val *= factor;
        }
        xs.front() = static_cast<T>(a);
        xs.back()  = static_cast<T>(b);
        return xs;
    }

    /*! \brief MAX operator that ignores NaN for containers such as vector
     */
    template<typename Input>
    [[nodiscard]] auto nanmax(const Input &in, size_t from = 0, size_t num = -1ul) {
        if(num == -1ul) num = in.size();
        num                = std::min<size_t>(num, static_cast<size_t>(in.size()) - from);
        using Scalar       = typename Input::value_type;
        constexpr auto nan = std::numeric_limits<Scalar>::quiet_NaN();
        bool           allnan =
            std::all_of(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num), [](auto val) -> bool { return std::isnan(val); });
        if(allnan) return nan;

        auto val = nan;

        for(size_t idx = from; idx < from + num; ++idx) {
            if(std::isnan(in[idx])) continue;
            if(std::isnan(val))
                val = in[idx];
            else
                val = std::max(val, in[idx]);
        }
        return val;
    }
    /*! \brief MAX operator that ignores NaN for containers such as vector
     */
    template<typename Input>
    [[nodiscard]] auto nanmin(const Input &in, size_t from = 0, size_t num = -1ul) {
        if(num == -1ul) num = in.size();
        num                = std::min<size_t>(num, static_cast<size_t>(in.size()) - from);
        using Scalar       = typename Input::value_type;
        constexpr auto nan = std::numeric_limits<Scalar>::quiet_NaN();
        bool           allnan =
            std::all_of(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num), [](auto val) -> bool { return std::isnan(val); });
        if(allnan) return nan;

        auto val = nan;

        for(size_t idx = from; idx < from + num; ++idx) {
            if(std::isnan(in[idx])) continue;
            if(std::isnan(val))
                val = in[idx];
            else
                val = std::min(val, in[idx]);
        }
        return val;
    }

    /*! \brief Sum operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to add (default == 0)
     *   \param to last element to add (default == -1: from - size)
     *   \return sum of of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> sum(v,0,3) = 10 </code>.
     */
    template<typename Input>
    [[nodiscard]] auto sum(const Input &in, size_t from = 0, size_t num = -1ul) {
        if(num == -1ul) num = in.size();
        num = std::min<size_t>(num, static_cast<size_t>(in.size()) - from);
        return std::accumulate(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num),
                               static_cast<typename Input::value_type>(0));
    }

    /*! \brief Product operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to multiply (default == 0)
     *   \param to last element to multiply (default == -1: from - size)
     *   \return product of of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> prod(v,0,3) = 24 </code>.
     */
    template<typename Input>
    [[nodiscard]] auto prod(const Input &in, size_t from = 0, size_t num = -1ul) {
        from = std::clamp<size_t>(from, 0, in.size());
        num  = std::clamp<size_t>(num, 0, in.size() - from);
        return std::accumulate(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num), 1, std::multiplies<>());
    }
    template<typename Input>
    [[nodiscard]] auto diff(const Input &in, size_t from = 0, size_t num = -1ul) {
        from = std::clamp<size_t>(from, 0, in.size());
        num  = std::clamp<size_t>(num, 0, in.size() - from);
        if constexpr(std::is_default_constructible_v<Input>) {
            Input res;
            std::adjacent_difference(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num), std::back_inserter(res));
            return res;
        } else {
            using value_type = typename Input::value_type;
            std::vector<value_type> res;
            res.reserve(num);
            std::adjacent_difference(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num), std::back_inserter(res));
            return res;
        }
    }

    template<typename Input>
    [[nodiscard]] auto mean(const Input &in, size_t from = 0, size_t num = -1ul) {
        from = std::clamp<size_t>(from, 0, in.size());
        num  = std::clamp<size_t>(num, 0, in.size() - from);
        return std::reduce(std::begin(in) + safe_cast<long>(from), std::begin(in) + safe_cast<long>(from + num)) / static_cast<double>(num);
    }

    /*! \brief Cumulative operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to add (default == 0)
     *   \param to last element to add (default == -1: from - size)
     *   \return binary op of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> cumop(v,std::plus<>(),0,3) = {1,3,6,10} </code>.
     */
    template<typename Input, typename BinaryOp>
    [[nodiscard]] Input cumop(const Input &in, BinaryOp op, size_t from = 0, size_t num = -1ul) {
        Input res;
        from = std::clamp<size_t>(from, 0, in.size());
        num  = std::clamp<size_t>(num, 0, in.size());
        std::partial_sum(in.begin() + safe_cast<long>(from), in.begin() + safe_cast<long>(from + num), std::back_inserter(res), op);
        return res;
    }

    template<typename Input>
    [[nodiscard]] Input cumsum(const Input &in, size_t from = 0, size_t num = -1ul) {
        return cumop(in, std::plus<typename Input::value_type>(), from, num);
    }
    template<typename Input>
    [[nodiscard]] Input cumsub(const Input &in, size_t from = 0, size_t num = -1ul) {
        return cumop(in, std::minus<typename Input::value_type>(), from, num);
    }
    template<typename Input>
    [[nodiscard]] Input cumprod(const Input &in, size_t from = 0, size_t num = -1ul) {
        return cumop(in, std::multiplies<typename Input::value_type>(), from, num);
    }
    template<typename Input>
    [[nodiscard]] Input cummin(const Input &in, size_t from = 0, size_t num = -1ul) {
        return cumop(in, [](auto a, auto b) { return std::min(a, b); }, from, num);
    }
    template<typename Input>
    [[nodiscard]] Input cummax(const Input &in, size_t from = 0, size_t num = -1ul) {
        return cumop(in, [](auto a, auto b) { return std::max(a, b); }, from, num);
    }

    namespace internal {
        template<typename ContainerType, typename Scalar>
        requires floating_point<Scalar>
        size_t nearestNeighbourIndex(const ContainerType &x, Scalar value) {
            Scalar dist    = std::numeric_limits<Scalar>::max();
            Scalar newDist = dist;
            size_t idx     = 0;
            for(size_t i = 0; i < x.size(); ++i) {
                newDist = std::abs(value - x[i]);
                if(newDist <= dist) {
                    dist = newDist;
                    idx  = i;
                }
            }
            return idx;
        }
        template<typename ContainerType, typename Scalar>
        requires floating_point<Scalar>
        size_t previousNeighbourIndex(const ContainerType &x, Scalar value) {
            size_t idx = 0;
            for(size_t i = 0; i < x.size(); ++i) {
                if(x[i] >= value) { return idx; }
                idx = i;
            }
            return idx;
        }
    }

    template<typename ContainerTypeX, typename ContainerTypeY, typename ContainerTypeXnew>
    ContainerTypeY interp1(const ContainerTypeX &x, const ContainerTypeY &y, const ContainerTypeXnew &x_new) {
        using ScalarY = std::conditional_t<internal::has_value_type_v<ContainerTypeY>, typename ContainerTypeY::value_type, typename ContainerTypeY::Scalar>;
        static_assert(internal::floating_point<ScalarY>);
        ContainerTypeY y_new(x_new.size());
        ScalarY        dx, dy, m, b;

        for(size_t i = 0; i < x_new.size(); ++i) {
            size_t idx = internal::nearestNeighbourIndex(x, x_new[i]);
            if(x[idx] > static_cast<ScalarY>(x_new[i])) {
                dx = idx > 0 ? (x[idx] - x[idx - 1]) : (x[idx + 1] - x[idx]);
                dy = idx > 0 ? (y[idx] - y[idx - 1]) : (y[idx + 1] - y[idx]);
            } else {
                dx = idx + 1 < x.size() ? (x[idx + 1] - x[idx]) : (x[idx] - x[idx - 1]);
                dy = idx + 1 < x.size() ? (y[idx + 1] - y[idx]) : (y[idx] - y[idx - 1]);
            }
            m        = dy / dx;
            b        = y[idx] - x[idx] * m;
            y_new[i] = static_cast<ScalarY>(x_new[i]) * m + b;
        }
        return y_new;
    }

    template<typename ContainerTypeX, typename ContainerTypeY, typename ContainerTypeXnew>
    ContainerTypeY interp1_prev(const ContainerTypeX &x, const ContainerTypeY &y, const ContainerTypeXnew &x_new) {
        // Interpolates using the previous value
        using ScalarY = std::conditional_t<internal::has_value_type_v<ContainerTypeY>, typename ContainerTypeY::value_type, typename ContainerTypeY::Scalar>;
        static_assert(internal::floating_point<ScalarY>);
        ContainerTypeY y_new(x_new.size());
        for(size_t i = 0; i < x_new.size(); ++i) {
            size_t idx = internal::previousNeighbourIndex(x, x_new[i]);
            y_new[i]   = y[idx];
        }
        return y_new;
    }

    /*! \brief Trapezoidal rule for numerical integration
     *   \param X a vector-like container
     *   \param Y a vector-like container
     *   \param from first element to multiply (default == 0)
     *   \param to last element to multiply (default == -1: from - size)
     *   \return product of of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> prod(v,0,3) = 24 </code>.
     */
    template<typename ContainerTypeX, typename ContainerTypeY>
    auto trapz(const ContainerTypeX &X, const ContainerTypeY &Y, long from = 0, long num = -1) {
        using ScalarY = std::conditional_t<internal::has_value_type_v<ContainerTypeY>, typename ContainerTypeY::value_type, typename ContainerTypeY::Scalar>;
        static_assert(internal::floating_point<ScalarY>);
        auto xfrm = std::clamp<size_t>(from, 0, static_cast<size_t>(X.size()));
        auto yfrm = std::clamp<size_t>(from, 0, static_cast<size_t>(Y.size()));
        auto xnum = std::clamp<size_t>(num, xfrm, static_cast<size_t>(X.size()));
        auto ynum = std::clamp<size_t>(num, yfrm, static_cast<size_t>(Y.size()));
        auto x_it = X.begin() + from;
        auto y_it = Y.begin() + from;
        auto x_en = X.begin() + xnum;
        auto y_en = Y.begin() + ynum;
        auto res  = ScalarY(0.0);
        while(x_it != x_en and y_it != y_en) {
            auto x_nx = std::next(x_it);
            auto y_nx = std::next(y_it);
            if(x_nx == x_en) break;
            if(y_nx == y_en) break;
            res += static_cast<ScalarY>(*x_nx - *x_it) * 0.5 * (*y_nx + *y_it);
            x_it++;
            y_it++;
        }
        return res;
    }

    /*! \brief Cumulative sum of the trapezoidal numerical integration
     *   \param X a vector-like container
     *   \param Y a vector-like container
     *   \param from first element to multiply (default == 0)
     *   \param to last element to multiply (default == -1: from - size)
     *   \return product of of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> prod(v,0,3) = 24 </code>.
     */
    template<typename ContainerTypeX, typename ContainerTypeY>
    ContainerTypeY cumtrapz(const ContainerTypeX &X, const ContainerTypeY &Y, long from = 0, long num = -1) {
        using ScalarY = std::conditional_t<internal::has_value_type_v<ContainerTypeY>, typename ContainerTypeY::value_type, typename ContainerTypeY::Scalar>;
        static_assert(internal::floating_point<ScalarY>);
        auto R    = ContainerTypeY(Y.size());
        auto xfrm = std::clamp<size_t>(from, 0, static_cast<size_t>(X.size()));
        auto yfrm = std::clamp<size_t>(from, 0, static_cast<size_t>(Y.size()));
        auto rfrm = std::clamp<size_t>(from, 0, static_cast<size_t>(R.size()));
        auto xnum = std::clamp<size_t>(num, xfrm, static_cast<size_t>(X.size()));
        auto ynum = std::clamp<size_t>(num, yfrm, static_cast<size_t>(Y.size()));
        auto rnum = std::clamp<size_t>(num, rfrm, static_cast<size_t>(R.size()));
        auto x_it = X.begin() + from;
        auto y_it = Y.begin() + from;
        auto r_it = R.begin() + from;
        auto x_en = X.begin() + xnum;
        auto y_en = Y.begin() + ynum;
        auto r_en = R.begin() + rnum;

        auto sum = ScalarY(0.0);
        while(x_it != x_en and y_it != y_en and r_it != r_en) {
            auto x_nx = std::next(x_it);
            auto y_nx = std::next(y_it);
            if(x_nx == x_en) break;
            if(y_nx == y_en) break;
            *r_it = sum;
            sum += static_cast<ScalarY>(*x_nx - *x_it) * 0.5 * (*y_nx + *y_it);
            x_it++;
            y_it++;
            r_it++;
        }
        *r_it = sum;
        return R;
    }

    /*! \brief Checks if multiple values are equal to each other
     *   \param args any number of values
     *   \return bool, true if all args are equal
     */
    template<typename First, typename... T>
    [[nodiscard]] bool all_equal(First &&first, T &&...t) noexcept {
        return ((first == t) && ...);
    }

    template<typename R, typename T>
    [[nodiscard]] R next_power_of_two(T val) {
        return static_cast<R>(std::pow<long>(2, safe_cast<long>(std::ceil(std::log2(std::real(val))))));
    }
    template<typename R, typename T>
    [[nodiscard]] R prev_power_of_two(T val) {
        return static_cast<R>(std::pow<long>(2, safe_cast<long>(std::floor(std::log2(std::real(val - 1))))));
    }
    template<typename R, typename T>
    [[nodiscard]] R nearest_power_of_two(T val) {
        auto next2 = next_power_of_two<R>(val);
        auto prev2 = prev_power_of_two<R>(val);
        if(std::abs(val - next2) < std::abs(val - prev2))
            return next2;
        else
            return prev2;
    }

    template<typename R, typename T>
    [[nodiscard]] inline R next_multiple(const T num, const T mult) {
        if(mult == 0) return num;
        return (num + mult) - mod(num, mult);
    }
    template<typename R, typename T>
    [[nodiscard]] inline R prev_multiple(const T num, const T mult) {
        if(mult == 0) return num;
        auto m = mod(num, mult);
        if(m == 0) return prev_multiple<R>(num - 1, mult);
        return num - m;
    }

    template<typename T>
    [[nodiscard]] inline T round_to_multiple_of(const T number, const T multiple) {
        T result = number + multiple / 2;
        result -= num::mod(result, multiple);
        return result;
    }

    template<typename T>
    [[nodiscard]] inline T round_up_to_multiple_of(const T number, const T multiple) {
        if(multiple == 0) return number;
        auto remainder = num::mod(std::abs(number), multiple);
        if(remainder == 0) return number;
        if(number < 0)
            return -(std::abs(number) - remainder);
        else
            return number + multiple - remainder;
    }
}
