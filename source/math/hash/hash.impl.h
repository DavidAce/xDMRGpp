#include "../hash.h"
#include "math/float.h"
#include <bit>
#include <complex>
#include <cstdint>
#include <type_traits>

namespace hash {

    template<typename T>
    struct is_std_complex : std::false_type {};
    template<typename T>
    struct is_std_complex<std::complex<T>> : std::true_type {};
    template<typename T>
    inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

    // Map float/double to an unsigned "bits" type
    template<typename F> struct float_bits_uint;
    template<> struct float_bits_uint<float> {
        using type = std::uint32_t;
    };
    template<> struct float_bits_uint<double> {
        using type = std::uint64_t;
    };

    // Bit-cast a float/double to its integer bits
    template<typename F>
    static inline auto bits_of_fp(F x) {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
        using U = typename float_bits_uint<F>::type;
        return std::bit_cast<U>(x);
    }

    inline void hash_combine([[maybe_unused]] std::size_t &seed) {}

    template<typename T, typename... Rest>
    inline void hash_combine(std::size_t &seed, const T &v, Rest... rest) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        hash_combine(seed, rest...);
    }

    template<typename T>
    std::size_t hash_buffer(const T *v, unsigned long size, std::size_t seed) {
        std::size_t h = seed;

        // 1) fp128: treat as raw 128 bits split into two u64 (your current approach)
        if constexpr(std::is_same_v<T, fp128>) {
            for(unsigned long i = 0; i < size; ++i) {
                __uint128_t   bits = std::bit_cast<__uint128_t>(v[i]);
                std::uint64_t lo   = static_cast<std::uint64_t>(bits);
                std::uint64_t hi   = static_cast<std::uint64_t>(bits >> 64);
                hash_combine(h, lo, hi);
            }
            return h;
        }

        // 2) cx128: hash real/imag raw 128 bits each (your current approach)
        else if constexpr(std::is_same_v<T, cx128>) {
            for(unsigned long i = 0; i < size; ++i) {
                __uint128_t br = std::bit_cast<__uint128_t>(std::real(v[i]));
                __uint128_t bi = std::bit_cast<__uint128_t>(std::imag(v[i]));

                std::uint64_t rlo = static_cast<std::uint64_t>(br);
                std::uint64_t rhi = static_cast<std::uint64_t>(br >> 64);
                std::uint64_t ilo = static_cast<std::uint64_t>(bi);
                std::uint64_t ihi = static_cast<std::uint64_t>(bi >> 64);

                hash_combine(h, rlo, rhi, ilo, ihi);
            }
            return h;
        }

        // 3) std::complex<float/double>: hash bit patterns of real/imag
        else if constexpr(is_std_complex_v<T>) {
            using R = typename T::value_type;
            static_assert(std::is_same_v<R, float> || std::is_same_v<R, double>);

            for(unsigned long i = 0; i < size; ++i) {
                auto rb = bits_of_fp<R>(v[i].real());
                auto ib = bits_of_fp<R>(v[i].imag());
                hash_combine(h, rb, ib);
            }
            return h;
        }

        // 4) float/double: hash bit patterns
        else if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>) {
            for(unsigned long i = 0; i < size; ++i) {
                auto b = bits_of_fp<T>(v[i]);
                hash_combine(h, b);
            }
            return h;
        }

        // 5) Fallback: hash values (integers, enums, etc.)
        else {
            for(unsigned long i = 0; i < size; ++i) { hash_combine(h, v[i]); }
            return h;
        }
    }

}