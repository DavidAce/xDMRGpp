#include "../lbit.h"
#include "../spin.h"
#include "config/debug.h"
#include "config/enums.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/cast.h"
#include "math/linalg.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/stat.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "tensors/site/mps/MpsSite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include <algorithm>
#include <fmt/ranges.h>
#include <unordered_set>
#include <vector>

namespace settings {
    inline constexpr bool debug_cls = false;
}

/*! \brief Merge two MPO layers into a single one using SVD.
 *
 * Step 1:
 * @verbatim
 *                                        2
 *                                        |
 *            2                   0--[ mpo_up_L ]--1
 *            |                           |
 *   0--[ mpo_updn_L ]--1   =             3
 *            |                           2
 *            3                           |
 *                               0--[ mpo_dn_L ]--1
 *                                        |
 *                                        3
 *
 *                                        2
 *                                        |
 *            2                   0--[ mpo_up_R ]--1
 *            |                           |
 *   0--[ mpo_updn_R ]--1   =             3
 *            |                           2
 *            3                           |
 *                               0--[ mpo_dn_R ]--1
 *                                        |
 *                                        3
 *
 * @endverbatim
 *
 * Step 2:
 * @verbatim
 *              1                            2                            2
 *              |             SVD            |                            |
 *     0--[ mpo_updn_L ]--3    =      0--[ mpo(i) ]--1    S*VT* 0--[ mpo_updn_R ]--1
 *              |                            |                            |
 *              2                            3                            3
 * @endverbatim
 *
 *
 * @endverbatim
 */
template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up, bool adj_dn) {
    if(mpos_dn.size() != mpos_up.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_up.size());
    if constexpr(settings::debug_cls) tools::log->debug("Merging mpos dn and up");
    auto t_merge         = tid::tic_scope("merge2");
    auto mpos            = std::vector<Eigen::Tensor<Scalar, 4>>(mpos_dn.size());
    auto cfg             = svd::config();
    cfg.rank_max         = settings::flbit::cls::mpo_circuit_svd_bondlim;
    cfg.truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    cfg.svd_lib          = svd::lib::lapacke;
    cfg.svd_rtn          = svd::rtn::geauto;
    auto  svd            = svd::solver(cfg);
    auto &threads        = tenx::threads::get();

    {
        // Initialize a dummy SV to start contracting from the left
        auto mpo_du = Eigen::Tensor<Scalar, 4>();
        auto SV     = Eigen::Tensor<Scalar, 2>();
        SV.resize(std::array<long, 2>{1, mpos_dn.front().dimension(0) * mpos_up.front().dimension(0)});
        SV.setConstant(1.0);
        for(size_t idx = 0; idx < mpos.size(); ++idx) {
            {
                auto           t_svmpos = tid::tic_scope("svmpos");
                auto           dd       = mpos_dn[idx].dimensions();
                auto           du       = mpos_up[idx].dimensions();
                constexpr auto shf5     = std::array<long, 5>{0, 1, 3, 4, 2};
                auto           rsh_svl3 = std::array<long, 3>{SV.dimension(0), dd[0], du[0]};                              // Dims of SV from the left side
                auto rsh_mpo4      = std::array<long, 4>{SV.dimension(0), dd[1] * du[1], du[2], (adj_dn ? dd[2] : dd[3])}; // Dims of the new mpo_dmu to split
                auto idx_contract1 = tenx::idx({1}, {0});
                auto idx_contract2 = adj_dn ? tenx::idx({1, 4}, {0, 3}) : tenx::idx({1, 3}, {0, 3});
                auto mpos_dn_idx_  = adj_dn ? Eigen::Tensor<Scalar, 4>(mpos_dn[idx].conjugate()) : mpos_dn[idx];
                mpo_du.resize(rsh_mpo4);
                mpo_du.device(*threads->dev) =
                    SV.reshape(rsh_svl3).contract(mpos_dn_idx_, idx_contract1).contract(mpos_up[idx], idx_contract2).shuffle(shf5).reshape(rsh_mpo4);
            }
            if(idx + 1 < mpos.size()) {
                auto t_split            = tid::tic_scope("split");
                std::tie(mpos[idx], SV) = svd.split_mpo_l2r(mpo_du, cfg);
            } else {
                mpos[idx] = mpo_du;
                SV.resize(std::array<long, 2>{mpo_du.dimension(1), 1}); // So that the log message is nice
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> {} + SV {} | trunc {:.4e}", idx, mpo_du.dimensions(), mpos[idx].dimensions(), SV.dimensions(),
                                  svd.get_truncation_error());
        }
    }

    // Now compress once backwards
    {
        auto t_back = tid::tic_scope("back");
        auto mpoUS  = Eigen::Tensor<Scalar, 4>();
        auto US     = Eigen::Tensor<Scalar, 2>();
        US.resize(std::array<long, 2>{mpos.back().dimension(1), 1});
        US.setConstant(1.0);
        for(size_t idx = mpos.size() - 1; idx < mpos.size(); --idx) {
            auto dmpo  = mpos[idx].dimensions();
            auto rshUS = std::array<long, 4>{dmpo[0], US.dimension(1), dmpo[2], dmpo[3]};
            mpoUS.resize(rshUS);
            mpoUS.device(*threads->dev) = mpos[idx].contract(US, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx > 0) {
                std::tie(US, mpos[idx]) = svd.split_mpo_r2l(mpoUS, cfg);
            } else {
                mpos[idx] = mpoUS;
                US.resize(std::array<long, 2>{1, mpoUS.dimension(0)}); // So that the log message is nice
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> US {} + mpo {} | trunc {:.4e}", idx, dmpo, US.dimensions(), mpos[idx].dimensions(),
                                  svd.get_truncation_error());
        }
    }
    if constexpr(settings::debug_cls)
        for(const auto &[idx, mpo] : iter::enumerate(mpos)) tools::log->debug("mpo {:2}: {}", idx, mpo.dimensions());

    return mpos;
}
template std::vector<Eigen::Tensor<cx32, 4>>  qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx32, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx32, 4>> &mpos_up, bool adj_dn);
template std::vector<Eigen::Tensor<cx64, 4>>  qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx64, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx64, 4>> &mpos_up, bool adj_dn);
template std::vector<Eigen::Tensor<cx128, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx128, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx128, 4>> &mpos_up, bool adj_dn);
/*! \brief Merge two MPO layers into a single one using SVD.
 *
 * Step 1:
 * @verbatim
 *
 *
 *
 *                                        2
 *                                        |
 *                                0--[ mpo_up_L ]--1
 *                                        |
 *                                        3
 *            2                           2
 *            |                           |
 *   0--[ mpo_dnmdup_L ]--1   =  0--[ mpo_md_L ]--1
 *            |                           |
 *            3                           3 ---------|
 *                                        2          |
 *                                        |          |
 *                               0--[ mpo_dn_L*]--1  |  (we take the adjoint of mpo_dn)
 *                                        |          |
 *                                        3 ----------

 *
 * @endverbatim
 *
 * Step 2:
 * @verbatim
 *              1                            2                            2
 *              |             SVD            |                            |
 *     0--[ mpo_updn_L ]--3    =      0--[ mpo(i) ]--1    S*VT* 0--[ mpo_updn_R ]--1
 *              |                            |                            |
 *              2                            3                            3
 * @endverbatim
 *
 *
 * @endverbatim
 */
template<typename Scalar>
std::vector<Eigen::Tensor<Scalar, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_md,
                                                                         const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up) {
    if(mpos_md.empty()) return merge_unitary_mpo_layers(mpos_dn, mpos_up, true);
    if(mpos_dn.size() != mpos_up.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_up.size());
    if(mpos_dn.size() != mpos_md.size()) throw except::logic_error("size mismatch: {} != {}", mpos_dn.size(), mpos_md.size());
    if constexpr(settings::debug_cls) tools::log->debug("Merging mpos dn md up");
    auto t_merge         = tid::tic_scope("merge3");
    auto mpos            = std::vector<Eigen::Tensor<Scalar, 4>>(mpos_dn.size());
    auto cfg             = svd::config();
    cfg.rank_max         = settings::flbit::cls::mpo_circuit_svd_bondlim;
    cfg.truncation_limit = settings::flbit::cls::mpo_circuit_svd_trnclim;
    cfg.switchsize_gesdd = settings::precision::svd_switchsize_bdc;
    cfg.svd_lib          = svd::lib::lapacke;
    cfg.svd_rtn          = svd::rtn::geauto;
    auto  svd            = svd::solver(cfg);
    auto &threads        = tenx::threads::get();
    {
        // Initialize a dummy SV to start contracting from the left
        auto mpo_dmu = Eigen::Tensor<Scalar, 4>();
        auto SV      = Eigen::Tensor<Scalar, 2>();
        SV.resize(std::array<long, 2>{1, mpos_dn.front().dimension(0) * mpos_md.front().dimension(0) * mpos_up.front().dimension(0)});
        SV.setConstant(1.0);
        for(size_t idx = 0; idx < mpos.size(); ++idx) {
            {
                auto t_svmpos = tid::tic_scope("svmpos");
                auto dd       = mpos_dn[idx].dimensions();
                auto dm       = mpos_md[idx].dimensions();
                auto du       = mpos_up[idx].dimensions();

                constexpr auto shf6     = std::array<long, 6>{0, 1, 3, 4, 5, 2};
                auto           rsh_svl4 = std::array<long, 4>{SV.dimension(0), dd[0], dm[0], du[0]};                 // Dims of SV from the left side
                auto           rsh_mpo4 = std::array<long, 4>{SV.dimension(0), dd[1] * dm[1] * du[1], du[2], dd[2]}; // Dims of the new mpo_dmu to split
                mpo_dmu.resize(rsh_mpo4);
                mpo_dmu.device(*threads->dev) = SV.reshape(rsh_svl4)
                                                    .contract(mpos_dn[idx].conjugate(), tenx::idx({1}, {0}))
                                                    .contract(mpos_md[idx], tenx::idx({1, 5}, {0, 3}))
                                                    .contract(mpos_up[idx], tenx::idx({1, 5}, {0, 3}))
                                                    .shuffle(shf6)
                                                    .reshape(rsh_mpo4);
            }
            if(idx + 1 < mpos.size()) {
                auto t_split            = tid::tic_scope("split");
                std::tie(mpos[idx], SV) = svd.split_mpo_l2r(mpo_dmu, cfg);
            } else {
                mpos[idx] = mpo_dmu;
                SV.resize(std::array<long, 2>{mpo_dmu.dimension(1), 1}); // So that the log message is nice
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> {} + SV {} | trunc {:.4e}", idx, mpo_dmu.dimensions(), mpos[idx].dimensions(), SV.dimensions(),
                                  svd.get_truncation_error());
        }
    }

    // Now compress once backwards
    {
        auto t_back = tid::tic_scope("back");
        auto mpoUS  = Eigen::Tensor<Scalar, 4>();
        auto US     = Eigen::Tensor<Scalar, 2>();
        US.resize(std::array<long, 2>{mpos.back().dimension(1), 1});
        US.setConstant(1.0);
        for(size_t idx = mpos.size() - 1; idx < mpos.size(); --idx) {
            auto dmpo  = mpos[idx].dimensions();
            auto rshUS = std::array<long, 4>{dmpo[0], US.dimension(1), dmpo[2], dmpo[3]};
            mpoUS.resize(rshUS);
            mpoUS.device(*threads->dev) = mpos[idx].contract(US, tenx::idx({1}, {0})).shuffle(tenx::array4{0, 3, 1, 2});
            if(idx > 0) {
                std::tie(US, mpos[idx]) = svd.split_mpo_r2l(mpoUS, cfg);
            } else {
                mpos[idx] = mpoUS;
                US.resize(std::array<long, 2>{1, mpoUS.dimension(0)}); // So that the log message is nice
            }
            if constexpr(settings::debug_cls)
                tools::log->debug("split svd mpo {}: {} --> US {} + mpo {} | trunc {:.4e}", idx, dmpo, US.dimensions(), mpos[idx].dimensions(),
                                  svd.get_truncation_error());
        }
    }
    if constexpr(settings::debug_cls)
        for(const auto &[idx, mpo] : iter::enumerate(mpos)) tools::log->debug("mpo {:2}: {}", idx, mpo.dimensions());

    return mpos;
}
template std::vector<Eigen::Tensor<cx32, 4>>  qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx32, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx32, 4>> &mpos_md,
                                                                                 const std::vector<Eigen::Tensor<cx32, 4>> &mpos_up);
template std::vector<Eigen::Tensor<cx64, 4>>  qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx64, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx64, 4>> &mpos_md,
                                                                                 const std::vector<Eigen::Tensor<cx64, 4>> &mpos_up);
template std::vector<Eigen::Tensor<cx128, 4>> qm::lbit::merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx128, 4>> &mpos_dn,
                                                                                 const std::vector<Eigen::Tensor<cx128, 4>> &mpos_md,
                                                                                 const std::vector<Eigen::Tensor<cx128, 4>> &mpos_up);

Eigen::Tensor<cx64, 1> qm::lbit::get_lbit_2point_correlator5(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers, const Eigen::Matrix2cd &szi,
                                                             size_t pos_szi, const Eigen::Matrix2cd &szj) {
    if(mpo_layers.empty()) throw except::logic_error("mpo layers is empty");
    for(const auto &mpo_layer : mpo_layers) {
        if(mpo_layer.empty()) throw except::logic_error("mpo layer is empty");
        if(mpo_layer.size() != mpo_layers.front().size()) throw except::logic_error("mpo layer size mismatch");
    }
    auto mpo_layer_md = std::vector<Eigen::Tensor<cx64, 4>>(); // The growing center layer of the mpo mpo^adj contraction

    auto opi = tenx::TensorCast(szi);
    auto opj = tenx::TensorCast(szj);

    for(const auto &[idx_layer, mpo_layer] : iter::enumerate(mpo_layers)) {
        auto mpo_layer_up = mpo_layer;
        auto mpo_layer_dn = mpo_layer;
        // Contract the Pauli operators to get the sandwhich szj mpo[n] ... mpo[0] szi mpo[0]^adj ... mpo[n]^adj, where the indices in [] are the layers
        if(idx_layer == 0) { mpo_layer_up[pos_szi] = Eigen::Tensor<cx64, 4>(mpo_layer_up[pos_szi].contract(opi, tenx::idx({3}, {0}))); }
        mpo_layer_md = merge_unitary_mpo_layers(mpo_layer_dn, mpo_layer_md, mpo_layer_up);
    }

    // Initialize results to one
    auto one = Eigen::Tensor<cx64, 2>(1, 1);
    one.setConstant(1.0);
    auto           results = std::vector<Eigen::Tensor<cx64, 2>>(mpo_layer_md.size(), one);
    auto           temp4   = Eigen::Tensor<cx64, 4>();
    auto           temp2   = Eigen::Tensor<cx64, 2>();
    constexpr auto trc2    = std::array<long, 2>{2, 3};
    constexpr auto idx1    = tenx::idx({1}, {0});
    auto          &threads = tenx::threads::get();
    // Apply szj to each position one by one
    for(auto &&[pos_res, result] : iter::enumerate(results)) {
        // If pos_szi is too far way from pos_szj, then the result should be zero exactly, no need to compute it.
        long pos_diff = std::abs(static_cast<long>(pos_res) - safe_cast<long>(pos_szi));
        long max_diff = 2 * safe_cast<long>(mpo_layers.size());
        if(pos_diff > max_diff) {
            result.setZero();
            continue;
        }
        // Contract and trace the mid-layer
        auto t_trace = tid::tic_scope("trace");
        for(size_t pos_szj = 0; pos_szj < mpo_layer_md.size(); ++pos_szj) {
            auto mpo_id = mpo_layer_md[pos_szj];
            auto mpo_op = pos_szj == pos_res
                              ? Eigen::Tensor<cx64, 4>(mpo_layer_md[pos_szj].contract(opj, tenx::idx({2}, {1})).shuffle(std::array<long, 4>{0, 1, 3, 2}))
                              : mpo_id;
            temp4.resize(result.dimension(0), mpo_op.dimension(1), mpo_op.dimension(2), mpo_op.dimension(3));
            temp4.device(*threads->dev) = result.contract(mpo_op, idx1);
            temp2                       = temp4.trace(trc2);
            result                      = temp2 / temp2.constant(2.0);
        }
    }

    // Collect the results into a 1d tensor
    auto corrvec = Eigen::Tensor<cx64, 1>(safe_cast<long>(results.size()));
    for(const auto &[pos_res, result] : iter::enumerate(results)) {
        if(result.size() != 1) { throw except::logic_error("result should be a 1x1 matrix"); }
        corrvec(static_cast<long>(pos_res)) = result.coeff(0);
    }
    return corrvec;
}
Eigen::Tensor<cx64, 1> qm::lbit::get_lbit_2point_correlator6(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers, const Eigen::Matrix2cd &szi,
                                                             size_t pos_szi, const Eigen::Matrix2cd &szj) {
    // This function is the same as get_lbit_2point_correlator5, except we calculate <tau_i tau_j> instead of <tau_i sigma_j> (for each choice of j)
    if(mpo_layers.empty()) throw except::logic_error("mpo layers is empty");
    for(const auto &mpo_layer : mpo_layers) {
        if(mpo_layer.empty()) throw except::logic_error("mpo layer is empty");
        if(mpo_layer.size() != mpo_layers.front().size()) throw except::logic_error("mpo layer size mismatch");
    }
    auto mpo_layer_md = std::vector<Eigen::Tensor<cx64, 4>>(); // The growing center layer of the mpo mpo^adj contraction

    auto opi = tenx::TensorCast(szi);
    auto opj = tenx::TensorCast(szj);
    // Initialize results to one
    auto one = Eigen::Tensor<cx64, 2>(1, 1);
    one.setConstant(1.0);
    auto           results = std::vector<Eigen::Tensor<cx64, 2>>(mpo_layers.front().size(), one);
    auto           temp4   = Eigen::Tensor<cx64, 4>();
    auto           temp2   = Eigen::Tensor<cx64, 2>();
    constexpr auto trc2    = std::array<long, 2>{2, 3};
    constexpr auto idx1    = tenx::idx({1}, {0});
    auto          &threads = tenx::threads::get();

    // Apply szj to each position one by one
    for(auto &&[pos_szj, result] : iter::enumerate(results)) {
        // Initialize the middle layer
        for(const auto &[idx_layer, mpo_layer] : iter::enumerate(mpo_layers)) {
            auto mpo_layer_up = mpo_layer;
            auto mpo_layer_dn = mpo_layer;
            // Contract the Pauli operators to get the sandwhich szj mpo[n] ... mpo[0] szi mpo[0]^adj ... mpo[n]^adj, where the indices in [] are the layers
            if(idx_layer == 0) {
                mpo_layer_up[pos_szi] = Eigen::Tensor<cx64, 4>(mpo_layer_up[pos_szi].contract(opi, tenx::idx({3}, {0})));
                mpo_layer_up[pos_szj] = Eigen::Tensor<cx64, 4>(mpo_layer_up[pos_szj].contract(opj, tenx::idx({3}, {0})));
            }
            mpo_layer_md = merge_unitary_mpo_layers(mpo_layer_dn, mpo_layer_md, mpo_layer_up);
        }

        // Trace the middle layer
        auto t_trace = tid::tic_scope("trace");
        for(size_t pos_res = 0; pos_res < mpo_layer_md.size(); ++pos_res) {
            const auto &mpo_op = mpo_layer_md[pos_res];
            temp4.resize(result.dimension(0), mpo_op.dimension(1), mpo_op.dimension(2), mpo_op.dimension(3));
            temp4.device(*threads->dev) = result.contract(mpo_op, idx1);
            temp2                       = temp4.trace(trc2);
            result                      = temp2 / temp2.constant(2.0);
        }
    }

    // Collect the results into a 1d tensor
    auto corrvec = Eigen::Tensor<cx64, 1>(safe_cast<long>(results.size()));
    for(const auto &[pos_res, result] : iter::enumerate(results)) {
        if(result.size() != 1) { throw except::logic_error("result should be a 1x1 matrix"); }
        corrvec(static_cast<long>(pos_res)) = result.coeff(0);
    }
    return corrvec;
}