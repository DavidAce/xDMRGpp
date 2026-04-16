#pragma once

#include "math/float.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tools::common::contraction::internal {
    template<typename Scalar>
    inline constexpr bool cutensor_supported_v = std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64> or std::is_same_v<Scalar, cx32> or
                                                 std::is_same_v<Scalar, cx64>;

    template<typename Index, int Rank>
    [[nodiscard]] constexpr std::array<long, static_cast<std::size_t>(Rank)> to_std_array(const Eigen::DSizes<Index, Rank> &dims) {
        std::array<long, static_cast<std::size_t>(Rank)> out {};
        for(int i = 0; i < Rank; ++i) out[static_cast<std::size_t>(i)] = static_cast<long>(dims[i]);
        return out;
    }

    template<typename Scalar>
    std::size_t get_cutensor_operation_bytes(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims,
                                             std::array<long, 3> envR_dims);

    template<typename Scalar>
    bool cutensor_can_fit(std::array<long, 3> mps_dims, std::array<long, 4> mpo_dims, std::array<long, 3> envL_dims, std::array<long, 3> envR_dims);

    template<typename Scalar>
    void contract_with_cutensor(Scalar *res_ptr, std::array<long, 3> res_dims, const Scalar *mps_ptr, std::array<long, 3> mps_dims, const Scalar *mpo_ptr,
                                std::array<long, 4> mpo_dims, const Scalar *envL_ptr, std::array<long, 3> envL_dims, const Scalar *envR_ptr,
                                std::array<long, 3> envR_dims);

    template<typename Scalar, typename MpsTensor, typename MpoTensor, typename EnvTensor>
    std::size_t get_cutensor_operation_bytes(const MpsTensor &mps, const MpoTensor &mpo, const EnvTensor &envL, const EnvTensor &envR) {
        return get_cutensor_operation_bytes<Scalar>(to_std_array(mps.dimensions()), to_std_array(mpo.dimensions()), to_std_array(envL.dimensions()),
                                                    to_std_array(envR.dimensions()));
    }

    template<typename Scalar, typename MpsTensor, typename MpoTensor, typename EnvTensor>
    bool cutensor_can_fit(const MpsTensor &mps, const MpoTensor &mpo, const EnvTensor &envL, const EnvTensor &envR) {
        return cutensor_can_fit<Scalar>(to_std_array(mps.dimensions()), to_std_array(mpo.dimensions()), to_std_array(envL.dimensions()),
                                        to_std_array(envR.dimensions()));
    }

    template<typename Scalar, typename ResTensor, typename MpsTensor, typename MpoTensor, typename EnvTensor>
    void contract_with_cutensor(ResTensor &res, const MpsTensor &mps, const MpoTensor &mpo, const EnvTensor &envL, const EnvTensor &envR) {
        return contract_with_cutensor<Scalar>(res.data(), to_std_array(res.dimensions()), mps.data(), to_std_array(mps.dimensions()), mpo.data(),
                                              to_std_array(mpo.dimensions()), envL.data(), to_std_array(envL.dimensions()), envR.data(),
                                              to_std_array(envR.dimensions()));
    }
}
