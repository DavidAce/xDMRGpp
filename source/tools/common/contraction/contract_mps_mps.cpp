#include "../contraction.h"
#include "debug/exceptions.h"
#include "math/tenx.h"
// #include "tid/tid.h"
#if defined(DMRG_ENABLE_TBLIS)
    #include <tblis/tblis_config.h>
    #include <tblis/util/configs.h>
#endif

/* clang-format off */
using namespace tools::common::contraction;


template<typename Scalar>
void tools::common::contraction::contract_mps_mps(      Scalar * res_ptr       , std::array<long,3> res_dims,
                                                  const Scalar * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                  const Scalar * const mpsR_ptr, std::array<long,3> mpsR_dims){
//    auto t_con = tid::tic_token("contract_mps_mps", tid::level::highest);
    auto res  = Eigen::TensorMap<Eigen::Tensor<Scalar,3>>(res_ptr,res_dims);
    auto mpsL = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mpsL_ptr, mpsL_dims);
    auto mpsR = Eigen::TensorMap<const Eigen::Tensor<Scalar,3>>(mpsR_ptr, mpsR_dims);
    constexpr auto shuffle_idx  = std::array<long,4>{0, 2, 1, 3};
    constexpr auto contract_idx = tenx::idx({2}, {1});
    [[maybe_unused]] auto check_dims = std::array<long,3>{mpsL.dimension(0) * mpsR.dimension(0), mpsL.dimension(1), mpsR.dimension(2)};
    assert(res_dims == check_dims);
    assert(mpsL.dimension(2) == mpsR.dimension(1));
    auto &threads = tenx::threads::get();
    if constexpr(std::is_same_v<Scalar, fp32> or std::is_same_v<Scalar, fp64>){
        auto tmp = Eigen::Tensor<Scalar,4>(mpsL_dims[0], mpsL_dims[1], mpsR_dims[0], mpsR_dims[2]);
        #if defined(DMRG_ENABLE_TBLIS)
        auto arch =  get_tblis_arch();
        const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config(arch.data());
        // contract_tblis(mpsL, mpsR, tmp, "abe", "ced", "abcd", tblis_config);
        contract_tblis(mpsL.data(), mpsL.dimensions(),  //
                       mpsR.data(), mpsR.dimensions(),  //
                       tmp.data(), tmp.dimensions(),    //
                       "abe", "ced", "abcd", tblis_config);
        res.device(*threads->dev)  = tmp.shuffle(shuffle_idx).reshape(res_dims);
        #else
        res.device(*threads->dev) = mpsL.contract(mpsR, contract_idx).shuffle(shuffle_idx).reshape(res_dims);
        #endif
    }else{
        res.device(*threads->dev) = mpsL.contract(mpsR, contract_idx).shuffle(shuffle_idx).reshape(res_dims);
    }
}
template void tools::common::contraction::contract_mps_mps(      fp32 * res_ptr       , std::array<long,3> res_dims,
                                                           const fp32 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const fp32 * const mpsR_ptr, std::array<long,3> mpsR_dims);
template void tools::common::contraction::contract_mps_mps(      fp64 * res_ptr       , std::array<long,3> res_dims,
                                                           const fp64 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const fp64 * const mpsR_ptr, std::array<long,3> mpsR_dims);
template void tools::common::contraction::contract_mps_mps(      fp128 * res_ptr       , std::array<long,3> res_dims,
                                                           const fp128 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const fp128 * const mpsR_ptr, std::array<long,3> mpsR_dims);
template void tools::common::contraction::contract_mps_mps(      cx32 * res_ptr       , std::array<long,3> res_dims,
                                                           const cx32 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const cx32 * const mpsR_ptr, std::array<long,3> mpsR_dims);
template void tools::common::contraction::contract_mps_mps(      cx64 * res_ptr       , std::array<long,3> res_dims,
                                                           const cx64 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const cx64 * const mpsR_ptr, std::array<long,3> mpsR_dims);
template void tools::common::contraction::contract_mps_mps(      cx128 * res_ptr       , std::array<long,3> res_dims,
                                                           const cx128 * const mpsL_ptr, std::array<long,3> mpsL_dims,
                                                           const cx128 * const mpsR_ptr, std::array<long,3> mpsR_dims);

/* clang-format on */
