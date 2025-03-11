#include "../contraction.h"
#include "math/tenx.h"

using namespace tools::common::contraction;

/* clang-format off */
template<typename Scalar>
void tools::common::contraction::contract_mps_mpo_env(      Scalar *       res_ptr, std::array<long, 2> res_dims,
                                                      const Scalar * const env_ptr, std::array<long, 2> env_dims,
                                                      const Scalar * const mps_ptr, std::array<long, 3> mps_dims,
                                                      const Scalar * const mpo_ptr, std::array<long, 2> mpo_dims) {
    auto res = Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>(res_ptr, res_dims);
    auto env = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(env_ptr, env_dims);
    auto mps = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto mpo = Eigen::TensorMap<const Eigen::Tensor<Scalar, 2>>(mpo_ptr, mpo_dims);
    auto &threads = tenx::threads::get();
    res.device(*threads->dev) = env.contract(mps,             tenx::idx({0}, {2}))
                                   .contract(mpo,             tenx::idx({1}, {0}))
                                   .contract(mps.conjugate(), tenx::idx({0, 2}, {2, 0}));
}
template void tools::common::contraction::contract_mps_mpo_env(      fp32 *       res_ptr , std::array<long,2> res_dims,
                                                               const fp32 * const env_ptr , std::array<long,2> env_dims,
                                                               const fp32 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const fp32 * const mpo_ptr , std::array<long,2> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      fp64 *       res_ptr , std::array<long,2> res_dims,
                                                               const fp64 * const env_ptr , std::array<long,2> env_dims,
                                                               const fp64 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const fp64 * const mpo_ptr , std::array<long,2> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      cx32 *       res_ptr , std::array<long,2> res_dims,
                                                               const cx32 * const env_ptr , std::array<long,2> env_dims,
                                                               const cx32 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const cx32 * const mpo_ptr , std::array<long,2> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      cx64 *       res_ptr , std::array<long,2> res_dims,
                                                               const cx64 * const env_ptr , std::array<long,2> env_dims,
                                                               const cx64 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const cx64 * const mpo_ptr , std::array<long,2> mpo_dims);

template<typename Scalar>
void tools::common::contraction::contract_mps_mpo_env(Scalar       *       res_ptr, std::array<long, 3> res_dims,
                                                      const Scalar * const env_ptr, std::array<long, 3> env_dims,
                                                      const Scalar * const mps_ptr, std::array<long, 3> mps_dims,
                                                      const Scalar * const mpo_ptr, std::array<long, 4> mpo_dims) {
    auto res                           = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(res_ptr, res_dims);
    auto env                           = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(env_ptr, env_dims);
    auto mps                           = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
    auto mpo                           = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
    auto &threads = tenx::threads::get();
    res.device(*threads->dev) = env.contract(mps,            tenx::idx({0}, {2}))
                                  .contract(mpo,             tenx::idx({1, 2}, {1, 2}))
                                  .contract(mps.conjugate(), tenx::idx({0, 3}, {2, 0}))
                                  .shuffle(                  tenx::array3{0, 2, 1});
}

template void tools::common::contraction::contract_mps_mpo_env(      fp32 *       res_ptr , std::array<long,3> res_dims,
                                                               const fp32 * const env_ptr , std::array<long,3> env_dims,
                                                               const fp32 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const fp32 * const mpo_ptr , std::array<long,4> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      fp64 *       res_ptr , std::array<long,3> res_dims,
                                                               const fp64 * const env_ptr , std::array<long,3> env_dims,
                                                               const fp64 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const fp64 * const mpo_ptr , std::array<long,4> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      cx32 *       res_ptr , std::array<long,3> res_dims,
                                                               const cx32 * const env_ptr , std::array<long,3> env_dims,
                                                               const cx32 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const cx32 * const mpo_ptr , std::array<long,4> mpo_dims);
template void tools::common::contraction::contract_mps_mpo_env(      cx64 *       res_ptr , std::array<long,3> res_dims,
                                                               const cx64 * const env_ptr , std::array<long,3> env_dims,
                                                               const cx64 * const mps_ptr , std::array<long,3> mps_dims,
                                                               const cx64 * const mpo_ptr , std::array<long,4> mpo_dims);

/* clang-format on */
