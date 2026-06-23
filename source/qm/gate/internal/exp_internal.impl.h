#pragma once

#include "debug/exceptions.h"
#include "general/sfinae.h"
#include "math/float.h"
#include "math/tenx.h"
#include "qm/gate.h"
#include "tools/common/log.h"
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>

template<typename scalar_t, typename alpha_t>
requires tenx::sfinae::is_std_complex_v<scalar_t>
Eigen::Tensor<scalar_t, 2> qm::Gate::exp_internal(const Eigen::Tensor<scalar_t, 2> &op_, alpha_t alpha) const {
    auto           op_map                       = tenx::MatrixMap(op_);
    constexpr bool scalar_t_is_float128         = std::is_same_v<scalar_t, __float128>;
    constexpr bool scalar_t_is_complex_float128 = sfinae::is_std_complex_v<scalar_t> and std::is_same_v<typename scalar_t::value_type, __float128>;
    bool           exp_diagonal                 = tenx::isDiagonal(op);

    if(exp_diagonal and op_map.imag().isZero() and std::real(alpha) == 0) {
        using namespace std::complex_literals;
        auto diag = op_map.diagonal()
                        .unaryExpr([&alpha](const scalar_t &h) -> scalar_t {
                            scalar_t exp_ialpha_t;
#if defined(DMRG_USE_QUADMATH)
                            {
                                fp128 two_pi_128       = acosq(-1.0) * fp128(2.0);
                                fp128 alpha_h_128      = fp128(-alpha.imag()) * fp128(h.real());
                                fp128 fmod_alpha_h_128 = fmodq(alpha_h_128, two_pi_128);
                                exp_ialpha_t           = std::exp(-1.0i * static_cast<fp64>(fmod_alpha_h_128));
                            }
#else
                            {
                                fp128 two_pi_ld       = std::acos(fp128(-1.0)) * fp128(2.0);
                                fp128 alpha_h_ld      = static_cast<fp128>(std::imag(-alpha)) * static_cast<fp128>(std::real(h));
                                fp128 fmod_alpha_h_ld = std::fmod(alpha_h_ld, two_pi_ld);
                                exp_ialpha_t          = std::exp(-1.0i * static_cast<fp64>(fmod_alpha_h_ld));
                                if(std::isnan(fmod_alpha_h_ld)) { throw except::runtime_error("fmod gave nan"); }
                            }
#endif
                            return exp_ialpha_t;
                        })
                        .asDiagonal();

        return tenx::TensorMap(diag.toDenseMatrix());
    } else {
        tools::log->error("The given matrix is not diagonal!");
        if constexpr(scalar_t_is_float128 or scalar_t_is_complex_float128) {
            throw except::runtime_error("Non-diagonal Matrix exponential is undefined for type {}", sfinae::type_name<scalar_t>());
        }
        if constexpr(std::is_arithmetic_v<scalar_t>) {
            return tenx::TensorMap((static_cast<scalar_t>(alpha) * tenx::MatrixMap(op_)).exp().eval());
        } else if constexpr(sfinae::is_std_complex_v<scalar_t>) {
            if constexpr(std::is_arithmetic_v<typename scalar_t::value_type>) {
                using value_t   = typename scalar_t::value_type;
                auto alpha_cast = std::complex<value_t>(static_cast<value_t>(std::real(alpha)), static_cast<value_t>(std::imag(alpha)));
                return tenx::TensorMap((alpha_cast * tenx::MatrixMap(op_)).exp().eval());
            }
        }
        throw except::runtime_error("Matrix exponential is undefined for type {}", sfinae::type_name<scalar_t>());
    }
}
