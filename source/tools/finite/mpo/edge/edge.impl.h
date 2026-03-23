#pragma once
#include "../../mpo.h"
#include "math/tenx.h"

template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_mpos_with_edges(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos,
                                                                              const Eigen::Tensor<Scalar, 1> &Ledge, const Eigen::Tensor<Scalar, 1> &Redge) {
    auto  mpos_with_edge = mpos;
    auto &threads        = tenx::threads::get();

    /* We can prepend edgeL to the first mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *                        2               2
     *                        |               |
     *    0---[L]---(1)(0)---[M]---1 =  0---[LM]---1
     *                        |               |
     *                        3               3
     */
    const auto &mpoL_src = mpos.front();
    auto       &mpoL_tgt = mpos_with_edge.front();
    mpoL_tgt.resize(tenx::array4{1, mpoL_src.dimension(1), mpoL_src.dimension(2), mpoL_src.dimension(3)});
    mpoL_tgt.device(*threads->dev) = Ledge.reshape(tenx::array2{1, Ledge.size()}).contract(mpoL_src, tenx::idx({1}, {0}));

    /* We can append edgeR to the last mpo to reduce the size of subsequent operations.
     * Start by converting the edge from a rank1 to a rank2 with a dummy index of size 1:
     *         2                              1                       2
     *         |                              |                       |
     *    0---[M]---1  0---[R]---1   =  0---[MR]---3  [shuffle]  0---[MR]---1
     *         |                              |                       |
     *         3                              2                       3
     */
    const auto &mpoR_src = mpos.back();
    auto       &mpoR_tgt = mpos_with_edge.back();
    mpoR_tgt.resize(tenx::array4{mpoR_src.dimension(0), 1, mpoR_src.dimension(2), mpoR_src.dimension(3)});
    mpoR_tgt.device(*threads->dev) = mpoR_src.contract(Redge.reshape(tenx::array2{Redge.size(), 1}), tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
    return mpos_with_edge;
}
