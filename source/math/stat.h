#pragma once

#include "cast.h"
#include "num.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

/*!
 *  \namespace stat
 *  \brief Small convenience-type statistical functions, mean and linearFit
 *  \tableofcontents
 */

namespace stat {
    template<typename T, typename = std::void_t<>>
    struct is_iterable : public std::false_type {};
    template<typename T>
    struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>> : public std::true_type {};
    template<typename T>
    inline constexpr bool is_iterable_v = is_iterable<T>::value;

    template<typename ContainerType>
    void check_bounds(ContainerType &X, std::optional<size_t> start_point = std::nullopt, std::optional<size_t> end_point = std::nullopt) {
        if(start_point.has_value() and std::cmp_greater_equal(start_point.value(), std::size(X))) throw std::range_error("Start point is out of range");
        if(end_point.has_value() and (std::cmp_greater(end_point.value(), std::size(X)) or end_point.value() < start_point))
            throw std::range_error("End point is out of range");
    }

    template<typename ContainerType>
    [[nodiscard]] auto get_start_end_iterators(ContainerType &X, std::optional<size_t> start_point = std::nullopt,
                                               std::optional<size_t> end_point = std::nullopt) {
        if(end_point.has_value() and end_point.value() == -1ul) end_point = X.size();
        try {
            check_bounds(X, start_point, end_point);
        } catch(std::exception &err) { throw std::range_error("check_bounds failed: " + std::string(err.what())); }
        if(not start_point.has_value()) start_point = 0;
        if(not end_point.has_value()) end_point = X.size();
        return std::make_pair(X.begin() + static_cast<std::ptrdiff_t>(start_point.value()), X.begin() + static_cast<std::ptrdiff_t>(end_point.value()));
    }

    template<typename ContainerType>
    [[nodiscard]] auto min(ContainerType &X, std::optional<size_t> start_point = std::nullopt, std::optional<size_t> end_point = std::nullopt) {
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        return *std::min_element(x_it, x_en);
    }
    template<typename ContainerType>
    [[nodiscard]] auto max(ContainerType &X, std::optional<size_t> start_point = std::nullopt, std::optional<size_t> end_point = std::nullopt) {
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        return *std::max_element(x_it, x_en);
    }

    template<typename ContainerType>
    [[nodiscard]] typename ContainerType::value_type mean(ContainerType &X, std::optional<size_t> start_point = std::nullopt,
                                                          std::optional<size_t> end_point = std::nullopt) {
        using value_type  = typename ContainerType::value_type;
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        auto n            = static_cast<value_type>(std::distance(x_it, x_en));
        return std::accumulate(x_it, x_en, static_cast<typename ContainerType::value_type>(0.0)) / n;
    }

    template<typename ContainerType>
    [[nodiscard]] typename ContainerType::value_type median(ContainerType X, std::optional<size_t> offset = std::nullopt,
                                                            std::optional<size_t> extent = std::nullopt) {
        using value_type = typename ContainerType::value_type;
        if(X.empty()) return std::numeric_limits<value_type>::quiet_NaN();
        offset = offset.value_or(0);
        offset = std::clamp<size_t>(offset.value(), 0ul, X.size() - 1);
        extent = extent.value_or(X.size() - offset.value());
        extent = std::clamp<size_t>(extent.value(), 0ul, X.size() - offset.value());
        auto x = std::span<value_type>(X.begin() + static_cast<std::ptrdiff_t>(offset.value()), extent.value());
        auto n = static_cast<size_t>(x.size());
        if(n == 0) return std::numeric_limits<value_type>::quiet_NaN();
        std::sort(x.begin(), x.end());
        // Find the starting point
        auto x_it_mid = x.begin();
        // ... and then the mid-point between start to end
        std::advance(x_it_mid, safe_cast<long>(n / 2));
        if(n % 2ul == 0) {
            // Even number of elements, take mean between middle elements
            auto a = *x_it_mid;
            std::advance(x_it_mid, -1);
            auto b = *x_it_mid;
            return static_cast<value_type>(0.5) * (a + b);
        } else {
            // odd number of elements
            return *x_it_mid;
        }
    }

    template<typename ContainerType>
    [[nodiscard]] auto typical(const ContainerType &X, std::optional<size_t> start_point = std::nullopt, std::optional<size_t> end_point = std::nullopt) {
        auto xlog = num::cast<double>(X, [](const auto v) { return std::log(std::abs(v)); });
        return std::exp(stat::mean(xlog, start_point, end_point));
    }
    template<typename ContainerType>
    [[nodiscard]] typename ContainerType::value_type variance(const ContainerType &X, std::optional<size_t> start_point = std::nullopt,
                                                              std::optional<size_t> end_point = std::nullopt) {
        using value_type  = typename ContainerType::value_type;
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        auto n            = static_cast<value_type>(std::distance(x_it, x_en));
        if(n == 0) return 0.0;
        auto X_mean = stat::mean(X, start_point, end_point);
        auto sum    = std::accumulate(x_it, x_en, static_cast<value_type>(0.0), [&X_mean](auto x1, auto x2) { return x1 + (x2 - X_mean) * (x2 - X_mean); });
        return sum / n;
    }
    template<typename ContainerType>
    [[nodiscard]] typename ContainerType::value_type stdev(const ContainerType &X, std::optional<size_t> start_point = std::nullopt,
                                                           std::optional<size_t> end_point = std::nullopt) {
        return std::sqrt(variance(X, start_point, end_point));
    }

    template<typename ContainerType>
    [[nodiscard]] typename ContainerType::value_type sterr(const ContainerType &X, std::optional<size_t> start_point = std::nullopt,
                                                           std::optional<size_t> end_point = std::nullopt) {
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        if(x_it == x_en) return 0.0;
        auto n = static_cast<double>(std::distance(x_it, x_en));
        return stdev(X, start_point, end_point) / std::sqrt(n);
    }
    template<typename ContainerType>
    [[nodiscard]] std::vector<typename ContainerType::value_type> stdev_moving(const ContainerType &X, typename ContainerType::value_type width = 0.1,
                                                                               std::optional<size_t> start_point = std::nullopt,
                                                                               std::optional<size_t> end_point   = std::nullopt) {
        using value_type = typename ContainerType::value_type;
        static_assert(std::is_floating_point_v<value_type>);
        std::vector<value_type> res;
        width             = std::clamp<value_type>(width, 0, 1);
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        long idx0         = std::distance(X.begin(), x_it);
        long idxN         = std::distance(X.begin(), x_en);
        long xlen         = std::distance(x_it, x_en);                                                 // Number of points to consider in X
        long wlen         = std::max<long>(2, safe_cast<long>(width * static_cast<value_type>(xlen))); // Number of points in the moving window
        while(x_it != x_en) {
            long idx_beg = std::distance(X.begin(), x_it); // Center around x_it
            long idx_end = std::distance(X.begin(), x_it); // Center around x_it
            while(idx_end - idx_beg < wlen) {
                idx_end = std::clamp(idx_end + 1, idx_end, idxN); // Move forward 1 step if possible.
                if(idx_end - idx_beg < wlen) {
                    idx_beg = std::clamp(idx_beg - 1, idx0, idx_beg); // Move back 1 step if possible.
                }
                if(idx_beg == idx0 and idx_end == idxN) break;
            }
            if(idx_beg < 0 or safe_cast<size_t>(idx_end) > X.size()) break;
            res.emplace_back(stdev(X, idx_beg, idx_end));
            x_it++;
        }
        return res;
    }

    template<typename ContainerType>
    [[nodiscard]] std::vector<double> sterr_moving(const ContainerType &X, double width = 0.1, std::optional<size_t> start_point = std::nullopt,
                                                   std::optional<size_t> end_point = std::nullopt) {
        std::vector<double> res;
        width             = std::clamp<double>(width, 0.0, 1.0);
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        long idx0         = std::distance(X.begin(), x_it);
        long idxN         = std::distance(X.begin(), x_en);
        long xlen         = std::distance(x_it, x_en);                                             // Number of points to consider in X
        long wlen         = std::max<long>(2, safe_cast<long>(width * static_cast<double>(xlen))); // Number of points in the moving window
        while(x_it != x_en) {
            long idx_beg = std::distance(X.begin(), x_it); // Center around x_it
            long idx_end = std::distance(X.begin(), x_it); // Center around x_it
            while(idx_end - idx_beg < wlen) {
                idx_end = std::clamp(idx_end + 1, idx_end, idxN); // Move forward 1 step if possible.
                if(idx_end - idx_beg < wlen) {
                    idx_beg = std::clamp(idx_beg - 1, idx0, idx_beg); // Move back 1 step if possible.
                }
                if(idx_beg == idx0 and idx_end == idxN) break;
            }
            if(idx_beg < 0 or static_cast<size_t>(idx_end) > X.size()) break;
            res.emplace_back(sterr(X, idx_beg, idx_end));
            x_it++;
        }
        return res;
    }
    template<typename ContainerType>
    [[nodiscard]] std::vector<double> rel_sterr_moving(const ContainerType &X, double width = 0.1, std::optional<size_t> start_point = std::nullopt,
                                                       std::optional<size_t> end_point = std::nullopt) {
        std::vector<double> res;
        width             = std::clamp<double>(width, 0.0, 1.0);
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        long idx0         = std::distance(X.begin(), x_it);
        long idxN         = std::distance(X.begin(), x_en);
        long xlen         = std::distance(x_it, x_en);                                             // Number of points to consider in X
        long wlen         = std::max<long>(2, safe_cast<long>(width * static_cast<double>(xlen))); // Number of points in the moving window
        while(x_it != x_en) {
            long idx_beg = std::distance(X.begin(), x_it); // Center around x_it
            long idx_end = std::distance(X.begin(), x_it); // Center around x_it
            while(idx_end - idx_beg < wlen) {
                idx_end = std::clamp(idx_end + 1, idx_end, idxN); // Move forward 1 step if possible.
                if(idx_end - idx_beg < wlen) {
                    idx_beg = std::clamp(idx_beg - 1, idx0, idx_beg); // Move back 1 step if possible.
                }
                if(idx_beg == idx0 and idx_end == idxN) break;
            }
            if(idx_beg < 0 or static_cast<size_t>(idx_end) > X.size()) break;
            // Take relative X
            auto Xavg = mean(X, idx_beg, idx_end);
            auto Xrel = X;
            if(std::abs(Xavg) > 0) {
                for(auto &x : Xrel) x = x / Xavg;
            }
            res.emplace_back(sterr(Xrel, idx_beg, idx_end));
            x_it++;
        }
        return res;
    }

    template<typename ContainerType1, typename ContainerType2, typename = std::enable_if_t<is_iterable_v<ContainerType1> and is_iterable_v<ContainerType2>>>
    [[nodiscard]] std::vector<double> cumtrapz(const ContainerType1 &X, const ContainerType2 &Y, std::optional<size_t> start_point = std::nullopt,
                                               std::optional<size_t> end_point = std::nullopt) {
        auto [x_it, x_en]       = get_start_end_iterators(X, start_point, end_point);
        auto [y_it, y_en]       = get_start_end_iterators(Y, start_point, end_point);
        auto                num = std::min(std::distance(x_it, x_en), std::distance(y_it, y_en));
        std::vector<double> res;
        res.reserve(num);
        for(long i = 0; i < num; i++) { res.emplace_back(num::trapz(X, Y, i, num - i)); }
        return res;
    }
    template<typename ContainerType1, typename ContainerType2, typename = std::enable_if_t<is_iterable_v<ContainerType1> and is_iterable_v<ContainerType2>>>
    [[nodiscard]] std::vector<double> cumtrapz_avg(const ContainerType1 &X, const ContainerType2 &Y, std::optional<size_t> start_point = std::nullopt,
                                                   std::optional<size_t> end_point = std::nullopt) {
        auto [x_it, x_en]           = get_start_end_iterators(X, start_point, end_point);
        auto [y_it, y_en]           = get_start_end_iterators(Y, start_point, end_point);
        auto                num     = std::min(std::distance(x_it, x_en), std::distance(y_it, y_en));
        auto                idx_end = std::min(std::distance(X.begin(), x_en), std::distance(Y.begin(), y_en));
        std::vector<double> res;
        res.reserve(num);
        while(x_it != x_en and y_it != y_en) {
            auto idx_start = std::min(std::distance(X.begin(), x_it), std::distance(Y.begin(), y_it));
            auto div       = static_cast<double>(X[idx_end] - X[idx_start]);
            num            = std::min(std::distance(x_it, x_en), std::distance(y_it, y_en));
            res.emplace_back(num::trapz(X, Y, idx_start, num) / div);
            x_it++;
            y_it++;
        }
        return res;
    }

    /*! \brief Contains the result from fitting (X,Y) data to Y = M + KX */
    template<typename T>
    requires std::is_floating_point_v<T>
    struct LinearFit {
        T isect = std::numeric_limits<T>::quiet_NaN(); /*!< m: intersection with x=0 */
        T slope = std::numeric_limits<T>::quiet_NaN(); /*!< k: slope */
        T rms   = std::numeric_limits<T>::quiet_NaN(); /*!< Root mean squared deviation */
        T rsq   = std::numeric_limits<T>::quiet_NaN(); /*!< R² or coefficient of determination */
    };
    template<typename ContainerType1, typename ContainerType2, typename = std::enable_if_t<is_iterable_v<ContainerType1> and is_iterable_v<ContainerType2>>>
    [[nodiscard]] auto linearFit(const ContainerType1 &X, const ContainerType2 &Y, std::optional<size_t> start_point = std::nullopt,
                                 std::optional<size_t> end_point = std::nullopt) {
        if(X.size() != Y.size())
            throw std::range_error("linearFit: size mismatch in arrays: X.size() == " + std::to_string(X.size()) +
                                   " | Y.size() == " + std::to_string(Y.size()));
        auto [x_it, x_en] = get_start_end_iterators(X, start_point, end_point);
        auto [y_it, y_en] = get_start_end_iterators(Y, start_point, end_point);
        using T           = std::common_type_t<decltype(*x_it), decltype(*y_it)>;
        static_assert(std::is_floating_point_v<T>);
        auto n = static_cast<T>(std::distance(x_it, x_en));
        if(n <= 1) return LinearFit<T>(); // Need at least 2 points
        auto avgX    = std::accumulate(x_it, x_en, T{0}) / n;
        auto avgY    = std::accumulate(y_it, y_en, T{0}) / n;
        auto xxsqerr = [&](auto a, auto b) { return (a - avgX) * (b - avgX); };
        auto yysqerr = [&](auto a, auto b) { return (a - avgY) * (b - avgY); };
        auto xysqerr = [&](auto a, auto b) { return (a - avgX) * (b - avgY); };
        auto sxx     = std::inner_product(x_it, x_en, x_it, T{0}, std::plus<>(), xxsqerr);
        auto syy     = std::inner_product(y_it, y_en, y_it, T{0}, std::plus<>(), yysqerr); // aka total sum of squares TSS
        auto sxy     = std::inner_product(x_it, x_en, y_it, T{0}, std::plus<>(), xysqerr);
        auto slope   = sxy / sxx;
        auto isect   = avgY - slope * avgX;

        auto yfiterr = [&](const auto &x, const auto &y) { return std::pow(y - (isect + slope * x), 2); };
        auto rss     = std::inner_product(x_it, x_en, y_it, T{0}, std::plus<>(), yfiterr);

        auto rms = std::sqrt(rss / n);
        auto rsq = 1 - rss / syy;

        auto result  = LinearFit<T>();
        result.isect = isect;
        result.slope = slope;
        result.rms   = rms;
        result.rsq   = rsq;
        return result;
    }

    template<typename ContainerType>
    [[nodiscard]] std::pair<double, double> slope(const ContainerType &Y, std::optional<size_t> start_point = std::nullopt,
                                                  std::optional<size_t> end_point = std::nullopt) {
        ContainerType X(Y.size());
        std::iota(X.begin(), X.end(), 0);
        return linearFit(X, Y, start_point, end_point);
    }

    template<typename ContainerType1, typename ContainerType2>
    [[nodiscard]] auto slope_at(const ContainerType1 &X, const ContainerType2 &Y, size_t at, size_t width = 1) {
        auto min_idx = static_cast<size_t>(std::max(safe_cast<long>(at) - safe_cast<long>(width), 0l));
        auto max_idx = static_cast<size_t>(std::min(at + width, Y.size()));
        return linearFit(X, Y, min_idx, max_idx).first;
    }

    template<typename ContainerType>
    [[nodiscard]] ContainerType smooth(const ContainerType &X, long width = 2) {
        if(X.size() <= 2) return X;
        width = std::min<long>(4, safe_cast<long>(X.size()) / 2);
        ContainerType S;
        S.reserve(X.size());
        for(size_t i = 0; i < X.size(); i++) {
            auto min_idx = std::clamp<long>(safe_cast<long>(i) - width, 0l, safe_cast<long>(i));
            auto max_idx = std::clamp<long>(safe_cast<long>(i) + width, safe_cast<long>(i), safe_cast<long>(X.size()) - 1l);
            S.push_back(stat::mean(X, min_idx, max_idx));
        }
        return S;
    }

    template<typename ContainerType>
    [[nodiscard]] size_t find_last_valid_point(const ContainerType &Y, std::optional<size_t> start_point = std::nullopt,
                                               std::optional<size_t> end_point = std::nullopt) {
        auto [y_it, y_en] = get_start_end_iterators(Y, start_point, end_point);
        auto n            = static_cast<double>(std::distance(y_it, y_en));
        if(n == 0) return start_point.has_value() ? start_point.value() : 0;
        auto y_invalid_it = std::find_if(y_it, y_en, [](const auto &val) {
            return std::isinf(std::real(val)) or std::isinf(std::imag(val)) or std::isnan(std::real(val)) or std::isnan(std::imag(val));
        });
        if(y_invalid_it != y_en and y_invalid_it != y_it) {
            // Found an invalid point! Back-track once
            std::advance(y_invalid_it, -1);
        }
        end_point = std::distance(Y.begin(), y_invalid_it);
        return end_point.value();
    }

    template<typename ContainerType>
    [[nodiscard]] size_t find_saturation_point(const ContainerType &Y, double slope_tolerance = 1, double std_tolerance = 1,
                                               std::optional<size_t> start_point = std::nullopt, std::optional<size_t> end_point = std::nullopt) {
        auto [y_it, y_en] = get_start_end_iterators(Y, start_point, end_point);
        auto n            = static_cast<double>(std::distance(y_it, y_en));
        auto idx          = start_point.has_value() ? start_point.value() : 0;
        if(n == 0) return idx;

        // Sometimes we check saturation using logarithms. It is important that the start-end point range doesn't have nan or infs
        end_point = find_last_valid_point(Y, start_point, end_point);

        // Consider Y vs X: a noisy signal decaying in the shape of a hockey-club, say.
        // We want to identify the point at which the signal stabilizes. We use the fact that the
        // standard deviation is high if it includes parts of the non-stable signal, and low if
        // it includes only the stable part.
        // Here we monitor the standard deviation of the signal between [start_point, end_point],
        // and move "start_point" towards the end. If the standard deviation goes below a certain
        // threshold, i.e. threshold < max_std, then we have found the stabilization point.

        auto X = num::range<size_t>(0, Y.size());
        while(idx < end_point.value()) {
            auto std        = stat::stdev(Y, idx, end_point.value());
            auto [slp, res] = stat::linearFit(X, Y, idx, end_point.value());
            printf("std: %g | slp: %g\n", std, slp);
            if(std::abs(slp) < slope_tolerance or std < std_tolerance) {
                if(idx > 0) {
                    printf("found idx %ld\n", idx - 1);
                    return idx - 1; // Backtrack
                } else {
                    printf("found idx %ld\n", idx);
                    return idx;
                }
            }
            idx++;
        }
        return end_point.value() - 1;
    }
}