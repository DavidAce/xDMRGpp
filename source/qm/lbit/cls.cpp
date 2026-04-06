#include "../lbit.h"
#include "../spin.h"
#include "config/debug.h"
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
#include <unordered_set>
#include <vector>

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
