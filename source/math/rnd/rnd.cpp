#include "../rnd.h"
#include "math/float.h"
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <omp.h>
#include <random>
#include <stdexcept>

namespace rnd {

    constexpr std::string_view enum2sv(const dist &d) {
        switch(d) {
            case dist::uniform: return "uniform";
            case dist::normal: return "normal";
            case dist::lognormal: return "lognormal";
        }
    }
    constexpr dist sv2enum(std::string_view d) {
        if(d == "uniform") return dist::uniform;
        if(d == "normal") return dist::normal;
        if(d == "lognormal") return dist::lognormal;
        throw std::runtime_error("rnd: unrecognized distribution: " + std::string(d));
    }

    namespace internal {
        // Make a random number engine
        //        inline pcg64 rng;
        inline std::mutex                          rng64_mutex;
        inline std::mutex                          rng128_mutex;
        // Commonly used distributions
        inline std::uniform_int_distribution<int>     rand_int_01(0, 1);
        inline std::uniform_real_distribution<double> rand_double_01(0.0, 1.0);
        inline std::uniform_real_distribution<double> rand_double_0_2pi(0, 2.0 * M_PI);
        inline std::normal_distribution<double>       normal_double_01(0.0, 1.0);

        template<typename Func>
        decltype(auto) with_rng64_lock(Func &&func) {
            std::scoped_lock lock(rng64_mutex);
            return std::forward<Func>(func)(rng64);
        }

        template<typename Func>
        decltype(auto) with_rng128_lock(Func &&func) {
            std::scoped_lock lock(rng128_mutex);
            return std::forward<Func>(func)(rng128);
        }
    }

    void seed(std::optional<long> n) {
        if(n.has_value() and n.value() >= 0) {
            auto given_seed = (unsigned long) n.value();
            std::printf("pcg-rng seed: %ld\n", given_seed);
            pcg_extras::seed_seq_from<pcg64> seq64(given_seed);
            {
                std::scoped_lock lock(internal::rng64_mutex, internal::rng128_mutex);
                internal::rng64.seed(seq64);
                pcg_extras::seed_seq_from<pcg128_once_insecure> seq128(given_seed);
                internal::rng128.seed(seq128);
            }
        } else {
            std::printf("pcg-rng seed: std::random_device\n");
            pcg_extras::seed_seq_from<std::random_device> seed_source;
            std::scoped_lock                            lock(internal::rng64_mutex, internal::rng128_mutex);
            internal::rng64.seed(seed_source);
            internal::rng128.seed(seed_source);
        }
        std::srand(internal::with_rng64_lock([](auto &rng) { return static_cast<unsigned>(rng()); }));
    }

    int uniform_integer_01() { return internal::with_rng64_lock([](auto &rng) { return internal::rand_int_01(rng); }); }

    double uniform_double_01() { return internal::with_rng64_lock([](auto &rng) { return internal::rand_double_01(rng); }); }

    template<typename T>
    T uniform_integer_box(T min, T max) {
        std::uniform_int_distribution<T> rand_int(std::min(min, max), std::max(min, max));
        return internal::with_rng64_lock([&](auto &rng) { return rand_int(rng); });
    }
    template int      uniform_integer_box(int min, int max);
    template unsigned uniform_integer_box(unsigned min, unsigned max);
    template long     uniform_integer_box(long min, long max);
    template size_t   uniform_integer_box(size_t min, size_t max);

    double uniform_double_box(double min, double max) {
        std::uniform_real_distribution<> rand_real(std::min(min, max), std::max(min, max));
        return internal::with_rng64_lock([&](auto &rng) { return rand_real(rng); });
    }
    double uniform_double_box(double halfwidth) {
        std::uniform_real_distribution<> rand_real(-halfwidth, halfwidth);
        return internal::with_rng64_lock([&](auto &rng) { return rand_real(rng); });
    }

    std::complex<double> uniform_complex_in_unit_circle() {
        return internal::with_rng64_lock([](auto &rng) { return std::polar(std::sqrt(internal::rand_double_01(rng)), internal::rand_double_0_2pi(rng)); });
    }

    std::complex<double> uniform_complex_on_unit_circle() {
        return internal::with_rng64_lock([](auto &rng) { return std::polar(1.0, internal::rand_double_0_2pi(rng)); });
    }

    std::complex<double> uniform_complex_box(double real_min, double real_max, double imag_min, double imag_max) {
        return {uniform_double_box(real_min, real_max), uniform_double_box(imag_min, imag_max)};
    }

    template<typename T>
    std::vector<T> uniform_unit_n_sphere(size_t n) {
        std::vector<T> arr;
        arr.reserve(n);
        double         norm = 0.0;
        internal::with_rng64_lock([&](auto &rng) {
            for(size_t i = 0; i < n; i++) {
                if constexpr(std::is_same<T, std::complex<double>>::value) {
                    double re   = internal::normal_double_01(rng);
                    double im   = internal::normal_double_01(rng);
                    T      cx64 = T(1.0, 0.0) * re + T(0.0, 1.0) * im;
                    arr.push_back(cx64);
                    norm += re * re + im * im;
                } else {
                    arr.push_back(internal::normal_double_01(rng));
                    norm += std::abs(arr[i] * arr[i]);
                }
            }
        });

        norm = std::sqrt(norm);
        for(size_t i = 0; i < n; i++) { arr[i] /= norm; }
        return arr;
    }
    template std::vector<fp64> uniform_unit_n_sphere(size_t n);
    template std::vector<cx64> uniform_unit_n_sphere(size_t n);

    std::complex<double> uniform_complex_slice(double radius_max, double angle_min, double angle_max) {
        return std::polar(uniform_double_box(0, radius_max), uniform_double_box(angle_min, angle_max));
    }
    template<typename out_t>
    out_t uniform(out_t a, out_t b) {
        if constexpr(std::is_arithmetic_v<out_t>) {
            std::uniform_real_distribution<out_t> distribution(a, b);
            return internal::with_rng64_lock([&](auto &rng) { return distribution(rng); });
        } else if constexpr(std::is_same_v<out_t, fp128>) {
            __extension__ typedef __uint128_t __uint128;
            auto rndval = internal::with_rng128_lock(
                [](auto &rng) { return static_cast<fp128>(rng()) / static_cast<fp128>(std::numeric_limits<__uint128>::max()); }
            );
            return (rndval * (b - a)) + a;
        }
    }
    template float  uniform(float mean, float std);
    template double uniform(double mean, double std);
    template fp128  uniform(fp128 mean, fp128 std);

    template<typename out_t>
    out_t normal_box_muller(out_t mu, out_t sigma) { // Box-Muller from Wiki
        constexpr out_t two_pi = std::numbers::pi_v<out_t> * 2;
        out_t           u1 = 0, u2 = uniform<out_t>(0, 1);
        while(u1 == 0) u1 = uniform<out_t>(0, 1);
        return sigma * sqrt<out_t>(-2 * log<out_t>(u1)) * cos<out_t>(two_pi * u2) + mu;
    }

    template<typename out_t>
    out_t normal(out_t mean, out_t std) {
        if constexpr(std::is_arithmetic_v<out_t>) {
            std::normal_distribution<out_t> distribution(mean, std);
            return internal::with_rng64_lock([&](auto &rng) { return distribution(rng); });
        } else if constexpr(std::is_same_v<out_t, fp128>) {
            return normal_box_muller(mean, std);
        }
    }
    template fp32  normal(fp32 mean, fp32 std);
    template fp64  normal(fp64 mean, fp64 std);
    template fp128 normal(fp128 mean, fp128 std);

    template<typename out_t>
    out_t log_normal(out_t mean, out_t std) {
        if constexpr(std::is_arithmetic_v<out_t>) {
            std::lognormal_distribution<out_t> distribution(mean, std);
            return internal::with_rng64_lock([&](auto &rng) { return distribution(rng); });
        } else if constexpr(std::is_same_v<out_t, fp128>) {
            auto n = std * normal_box_muller<out_t>(0, 1) + mean;
            return std::exp(n);
        }
    }
    template float   log_normal(float mean, float std);
    template double  log_normal(double mean, double std);
    template fp128   log_normal(fp128 mean, fp128 std);
    std::vector<int> random_with_replacement(const std::vector<int> &in) {
        std::vector<int> boot;
        boot.reserve(in.size());
        for(size_t i = 0; i < in.size(); i++) boot.emplace_back(in[uniform_integer_box(0ul, in.size() - 1)]);
        return boot;
    }
    std::vector<int> random_with_replacement(const std::vector<int> &in, const size_t n) {
        if(n > in.size()) throw std::logic_error("random_with_replacement: n too large");
        std::vector<int> boot;
        boot.reserve(n);
        for(size_t i = 0; i < n; i++) { boot.emplace_back(in[uniform_integer_box(0ul, in.size() - 1)]); }
        return boot;
    }

    double gaussian_truncated(const double lowerLimit, const double upperLimit, const double mean, const double std) {
        std::normal_distribution<double> distribution(mean, std);
        double                           ul = fmax(lowerLimit, upperLimit);
        double                           ll = fmin(lowerLimit, upperLimit);
        double                           number;
        while(true) {
            number = distribution(internal::rng64);
            if(number >= ll && number <= ul) { return number; }
        }
    }

    template<typename T>
    void shuffle(T &list) {
        std::shuffle(std::begin(list), std::end(list), internal::rng64);
    }
    template void shuffle(std::vector<int> &list);
    template void shuffle(std::vector<unsigned> &list);
    template void shuffle(std::vector<long> &list);
    template void shuffle(std::vector<size_t> &list);
    template void shuffle(std::vector<double> &list);
    template void shuffle(std::vector<fp128> &list);
    template void shuffle(std::string &list);

    template<typename out_t, typename Distribution>
    std::vector<out_t> random(Distribution &&d, size_t num) {
        auto rndvec = std::vector<out_t>(num);
        internal::with_rng64_lock([&](auto &rng) {
            for(size_t i = 0; i < num; ++i) rndvec[i] = d(rng);
        });
        return rndvec;
    }
    //    template std::vetor<fp128>  random<fp128>(Distribution &&d, size_t num);

    template<typename out_t>
    out_t random(dist d, out_t mean, out_t width) {
        switch(d) {
            case dist::uniform: return uniform<out_t>(mean - width / static_cast<out_t>(2), mean + width / static_cast<out_t>(2));
            case dist::normal: return normal<out_t>(mean, width);
            case dist::lognormal: return log_normal<out_t>(mean, width);
            default: throw std::runtime_error("Invalid distribution");
        }
    }
    template float  random<float>(dist d, float mean, float width);
    template double random<double>(dist d, double mean, double width);
    template fp128  random<fp128>(dist d, fp128 mean, fp128 width);
    template<typename out_t>
    out_t random(std::string_view distribution, out_t mean, out_t width) {
        return random<out_t>(sv2enum(distribution), mean, width);
    }
    template float  random<float>(std::string_view d, float mean, float width);
    template double random<double>(std::string_view d, double mean, double width);
    template fp128  random<fp128>(std::string_view d, fp128 mean, fp128 width);

    template<typename out_t>
    std::vector<out_t> random(dist d, out_t mean, out_t width, size_t num) {
        if constexpr(std::is_arithmetic_v<out_t>) {
            switch(d) {
                case dist::uniform: return random<out_t>(std::uniform_real_distribution<out_t>(mean - width / 2, mean + width / 2), num);
                case dist::normal: return random<out_t>(std::normal_distribution<out_t>(mean, width), num);
                case dist::lognormal: return random<out_t>(std::lognormal_distribution<out_t>(mean, width), num);
                default: throw std::runtime_error("Invalid distribution");
            }
        } else if(std::is_same_v<out_t, fp128>) {
            std::vector<fp128> rndvec(num);
            for(auto &r : rndvec) r = random<out_t>(d, mean, width);
            return rndvec;
        } else
            throw std::runtime_error("rnd::random: unrecognized type");
    }
    template std::vector<float>  random<float>(dist d, float mean, float width, size_t num);
    template std::vector<double> random<double>(dist d, double mean, double width, size_t num);
    template std::vector<fp128>  random<fp128>(dist d, fp128 mean, fp128 width, size_t num);

    template<typename out_t>
    std::vector<out_t> random(dist d, out_t mean, out_t width, const std::vector<out_t> &weights) {
        auto rndvec = random<out_t>(d, mean, width, weights.size());
        for(size_t i = 0; i < weights.size(); ++i) rndvec[i] *= weights[i];
        return rndvec;
    }
    template std::vector<float>  random<float>(dist d, float mean, float width, const std::vector<float> &weights);
    template std::vector<double> random<double>(dist d, double mean, double width, const std::vector<double> &weights);
    template std::vector<fp128>  random<fp128>(dist d, fp128 mean, fp128 width, const std::vector<fp128> &weights);

    template<typename out_t>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, size_t num) {
        return random<out_t>(sv2enum(distribution), mean, width, num);
    }
    template std::vector<float>  random<float>(std::string_view d, float mean, float width, size_t num);
    template std::vector<double> random<double>(std::string_view d, double mean, double width, size_t num);
    template std::vector<fp128>  random<fp128>(std::string_view d, fp128 mean, fp128 width, size_t num);

    template<typename out_t>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, const std::vector<out_t> &weights) {
        return random<out_t>(sv2enum(distribution), mean, width, weights);
    }
    template std::vector<float>  random<float>(std::string_view d, float mean, float width, const std::vector<float> &weights);
    template std::vector<double> random<double>(std::string_view d, double mean, double width, const std::vector<double> &weights);
    template std::vector<fp128>  random<fp128>(std::string_view d, fp128 mean, fp128 width, const std::vector<fp128> &weights);

}
