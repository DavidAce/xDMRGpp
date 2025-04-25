#pragma once
#include <complex>
#include <optional>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>
#include <vector>

namespace rnd {

    enum class dist { uniform, normal, lognormal };
    constexpr std::string_view enum2sv(const dist &d);
    constexpr dist             sv2enum(std::string_view d);

    namespace internal {
        // The random number engine
        inline pcg64                rng64;
        inline pcg128_once_insecure rng128;

    }
    // Random functions
    void                        seed(std::optional<long> n = std::nullopt);
    extern int                  uniform_integer_01();
    extern double               uniform_double_01();
    extern double               uniform_double_box(double min, double max);
    extern double               uniform_double_box(double halfwidth);
    extern std::complex<double> uniform_complex_in_unit_circle();
    extern std::complex<double> uniform_complex_on_unit_circle();
    extern std::complex<double> uniform_complex_box(double real_min, double real_max, double imag_min, double imag_max);
    extern std::complex<double> uniform_complex_slice(double radius_max, double angle_min, double angle_max);
    extern std::vector<int>     random_with_replacement(const std::vector<int> &indata);
    extern std::vector<int>     random_with_replacement(const std::vector<int> &indata, size_t num_choose);
    extern double               gaussian_truncated(double lowerLimit, double upperLimit, double mean, double std);
    template<typename T>
    T uniform_integer_box(T min, T max);

    template<typename out_t>
    out_t uniform(out_t min, out_t max);
    template<typename out_t>
    out_t normal(out_t mean, out_t std);
    template<typename out_t>
    out_t log_normal(out_t mean, out_t std);

    template<typename real_t = double>
    auto uniform_complex_in_circle(real_t radius = real_t{1}) {
        return std::polar(uniform(real_t{0}, radius), uniform(real_t{0}, 2 * std::numbers::pi_v<real_t>));
    }
    template<typename real_t = double>
    auto uniform_complex_on_circle(real_t radius = real_t{1}) {
        return std::polar(radius, uniform(real_t{0}, 2 * std::numbers::pi_v<real_t>));
    }
    template<typename real_t = double>
    auto uniform_complex_in_square(real_t real_min, real_t real_max, real_t imag_min, real_t imag_max) {
        return std::complex(uniform(real_min, real_max), uniform(imag_min, imag_max));
    }

    template<typename T>
    std::vector<T> uniform_unit_n_sphere(size_t n);

    template<typename T>
    void shuffle(T &list);

    template<typename out_t = double>
    out_t random(dist d, out_t mean, out_t width);
    template<typename out_t = double>
    out_t random(std::string_view distribution, out_t mean, out_t width);
    template<typename out_t = double>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, size_t num);

    template<typename out_t = double>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, const std::vector<out_t> &weights);

}
