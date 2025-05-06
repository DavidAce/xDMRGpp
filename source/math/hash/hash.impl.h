#include "../hash.h"
#include "math/float.h"
#include <complex>
#include <functional>
namespace hash {
    template<typename T>
    struct is_std_complex : public std::false_type {};
    template<typename T>
    struct is_std_complex<std::complex<T>> : public std::true_type {};
    template<typename T>
    inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

    inline void hash_combine([[maybe_unused]] std::size_t &seed) {}

    template<typename T, typename... Rest>
    inline void hash_combine(std::size_t &seed, const T &v, Rest... rest) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        hash_combine(seed, rest...);
    }

    template<typename T>
    std::size_t hash_buffer(const T *v, unsigned long size, std::size_t seed) {
        std::size_t h = seed;
        if constexpr(std::is_same_v<T, fp128>) {
            for(unsigned long idx = 0; idx < size; idx++) {
                __uint128_t bits = std::bit_cast<__uint128_t>(v[idx]);
                uint64_t    lo   = uint64_t(bits);
                uint64_t    hi   = uint64_t(bits >> 64);
                hash_combine(h, lo, hi);
            }
            return h;
        } else if constexpr(std::is_same_v<T, cx128>) {
            for(unsigned long idx = 0; idx < size; idx++) {
                __uint128_t bits_r = std::bit_cast<__uint128_t>(std::real(v[idx]));
                __uint128_t bits_i = std::bit_cast<__uint128_t>(std::imag(v[idx]));
                uint64_t    lo_r   = uint64_t(bits_r);
                uint64_t    hi_r   = uint64_t(bits_r >> 64);
                uint64_t    lo_i   = uint64_t(bits_i);
                uint64_t    hi_i   = uint64_t(bits_i >> 64);
                hash_combine(h, lo_r, hi_r);
                hash_combine(h, lo_i, hi_i);
            }
            return h;

        } else {
            if constexpr(is_std_complex_v<T>) {
                for(unsigned long idx = 0; idx < size; idx++) hash_combine(h, v[idx].real(), v[idx].imag());
                return h;
            } else {
                for(unsigned long idx = 0; idx < size; idx++) hash_combine(h, v[idx]);
                return h;
            }
        }
    }

}
