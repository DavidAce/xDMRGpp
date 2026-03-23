//
#include "../../mpo.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include "math/svd.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"

using tools::finite::mpo::Real;
template<typename Scalar>
std::pair<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 4>> tools::finite::mpo::swap_mpo(const Eigen::Tensor<Scalar, 4> &mpoL,
                                                                                           const Eigen::Tensor<Scalar, 4> &mpoR) {
    /* The swap operation takes two neighboring sites
     *
     *            (2)dL                           (2)dR
     *              |                               |
     *   (0)mL---[ mpsL ]---mC(1)       mC(0)---[ mpsR ]---(1)mR
     *              |                               |
     *           (3)dL                            (3)dR
     *
     *
     * and joins them into
     *
     *          (1)dL (4)dR
     *             |    |
     *   (0)mL ---[tensor]--- (3)mR
     *             |    |
     *          (2)dL (5)dR
     *
     *
     * Then we swap the physical indices
     *
     *
     *         (4)dR     (1)dL
     *            |        |
     *             \      /
     *               \  /
     *                /\      <---- Swap
     *              /    \
     *            /        \
     *            |        |
     *   (0)mL---[  tensor  ]---(3)mR
     *            |        |
     *             \      /
     *               \  /
     *                /\      <---- Swap
     *              /    \
     *            /        \
     *            |        |
     *         (5)dR     (2)dL
     *
     *
     * This is accomplished by the shuffle:
     *       shuffle(0, 4, 5, 3, 1, 2);
     *
     *
     * We can now split the tensor back into two MPO's, by first reshaping the tensor into a matrix with merged indices (012) x (345)
     *
     *         (4)dR     (1)dL                             (1)dR                                      (1)dL
     *            |        |                                 |                                          |
     *   (0)mL---[  tensor  ]---(3)mR     --->    (0)mL---[ mpoL ]---mC(3)  (0)---[S]---(1)  mC(0)---[ mpoR ]---(1)mR
     *            |        |                                 |                                          |
     *         (5)dR     (2)dL                            (2)dR                                       (2)dL
     *
     *
     * The square root of S can then be multiplied into both left and right MPO's, on the mC index.
     * The left mpo can be shuffled back to standard form with
     *   mpoL: shuffle(0,3,1,2)
     *
     */

    Eigen::Tensor<Scalar, 6> swapped_mpo = mpoL.contract(mpoR, tenx::idx({1}, {0})).shuffle(tenx::array6{0, 4, 5, 3, 1, 2}); // swap
    auto                     svd_cfg     = svd::config();
    svd_cfg.truncation_limit             = 1e-16;
    svd_cfg.svd_lib                      = svd::lib::lapacke;
    svd_cfg.svd_rtn                      = svd::rtn::gejsv;

    auto svd = svd::solver(svd_cfg);
    return svd.split_mpo_pair(swapped_mpo, svd_cfg);
}

template<typename Scalar>
void tools::finite::mpo::swap_sites(ModelFinite<Scalar> &model, size_t posL, size_t posR, std::vector<size_t> &sites) {
    auto t_swap = tid::tic_scope("swap");
    if(posR != posL + 1) throw except::logic_error("Expected posR == posL+1. Got posL {} and posR {}", posL, posR);
    if(posR != std::clamp(posR, 0ul, model.get_length() - 1ul)) throw except::logic_error("Expected posR in [0,{}]. Got {}", model.get_length() - 1, posR);
    if(posL != std::clamp(posL, 0ul, model.get_length() - 1ul)) throw except::logic_error("Expected posL in [0,{}]. Got {}", model.get_length() - 1, posL);

    auto &mpo_posL = model.get_mpo(posL);
    auto &mpo_posR = model.get_mpo(posR);

    auto [mpoL, mpoR]   = swap_mpo(mpo_posL.MPO(), mpo_posR.MPO());
    auto [mpoL2, mpoR2] = swap_mpo(mpo_posL.MPO2(), mpo_posR.MPO2());
    tools::log->debug("mpo({}) : {} -> {}", mpo_posL.get_position(), mpo_posL.MPO().dimensions(), mpoL.dimensions());
    tools::log->debug("mpo({}) : {} -> {}", mpo_posR.get_position(), mpo_posR.MPO().dimensions(), mpoR.dimensions());
    tools::log->debug("mpo²({}): {} -> {}", mpo_posL.get_position(), mpo_posL.MPO2().dimensions(), mpoL2.dimensions());
    tools::log->debug("mpo²({}): {} -> {}", mpo_posR.get_position(), mpo_posR.MPO2().dimensions(), mpoR2.dimensions());
    mpo_posL.set_mpo(mpoL);
    mpo_posR.set_mpo(mpoR);
    mpo_posL.set_mpo_squared(mpoL2);
    mpo_posR.set_mpo_squared(mpoR2);

    std::swap(sites[posL], sites[posR]);
}
