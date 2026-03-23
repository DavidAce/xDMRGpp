#pragma once
#include "float.h"
#include <Eigen/Core>
namespace Eigen {
#if defined(DMRG_USE_QUADMATH) || defined(DMRG_USE_FLOAT128)
    template<>
    struct NumTraits<fp128> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef fp128 Real;
        typedef fp128 NonInteger;
        typedef fp128 Nested;

        enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 3, MulCost = 3 };
    };

    template<>
    struct NumTraits<cx128> : NumTraits<cx64> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef fp128 Real;
        typedef fp128 NonInteger;
        typedef fp128 Nested;
        enum { IsComplex = 1, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 6, MulCost = 6 };
    };

    namespace numext {
        template<size_t Size>
        struct get_integer_by_size;

        template<>
        struct get_integer_by_size<16> {
            typedef __int128_t  signed_type;
            typedef __uint128_t unsigned_type;
        };
    }
    namespace internal {
        template<typename T>
        struct is_arithmetic;
        template<>
        struct is_arithmetic<fp128> {
            enum { value = true };
        };

        template<typename Scalar>
        struct random_bits_impl;
        template<>
        struct random_bits_impl<__uint128_t> {
            using Scalar                    = __uint128_t;
            using RandomDevice              = eigen_random_device;
            using RandomReturnType          = typename RandomDevice::ReturnType;
            static constexpr int kEntropy   = RandomDevice::Entropy;
            static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
            // return a Scalar filled with numRandomBits beginning from the least significant bit
            static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
                eigen_assert((numRandomBits >= 0) && (numRandomBits <= kTotalBits));
                const Scalar mask       = Scalar(-1) >> ((kTotalBits - numRandomBits) & (kTotalBits - 1));
                Scalar       randomBits = 0;
                for(int shift = 0; shift < numRandomBits; shift += kEntropy) {
                    RandomReturnType r  = RandomDevice::run();
                    randomBits         |= static_cast<Scalar>(r) << shift;
                }
                // clear the excess bits
                randomBits &= mask;
                return randomBits;
            }
        };

    }
#endif

}
