#pragma once
#include "../common.h"
#include <array>
#include <algorithm>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>

namespace linalg::tensor {
    template<typename Scalar, int rank, auto npair>
    Eigen::Tensor<Scalar, rank - 2 * npair> trace(const Eigen::Tensor<Scalar, rank>                       &tensor,
                                                  const std::array<Eigen::IndexPair<Eigen::Index>, npair> &idx_pair) {
        /*
         * Returns the partial trace of a tensor
         * Note that the tensor given here may be mirrored!
         */
        static_assert(rank >= 2 * npair, "Rank must be large enough");
        static_assert(npair <= 3, "npair > 3 is not implemented");
        if constexpr(npair == 0) {
            return tensor;
        } else if constexpr(npair == 1) {
            // Collect indices and dimensions traced
            return tensor.trace(std::array<Eigen::Index, 2>{idx_pair[0].first, idx_pair[0].second});
        } else if constexpr(npair == 2) {
            std::array<long, 2> pair1{idx_pair[1].first, idx_pair[1].second};
            std::array<long, 2> pair0{idx_pair[0].first, idx_pair[0].second};
            pair0[0] -= std::count_if(pair1.begin(), pair1.end(), [&pair0](auto i) { return i < pair0[0]; });
            pair0[1] -= std::count_if(pair1.begin(), pair1.end(), [&pair0](auto i) { return i < pair0[1]; });
            return tensor.trace(pair1).trace(pair0);
        } else if constexpr(npair == 3) {
            std::array<long, 2> pair2{idx_pair[2].first, idx_pair[2].second};
            std::array<long, 2> pair1{idx_pair[1].first, idx_pair[1].second};
            std::array<long, 2> pair0{idx_pair[0].first, idx_pair[0].second};
            pair1[0] -= std::count_if(pair2.begin(), pair2.end(), [&pair1](auto i) { return i < pair1[0]; });
            pair1[1] -= std::count_if(pair2.begin(), pair2.end(), [&pair1](auto i) { return i < pair1[1]; });
            pair0[0] -= std::count_if(pair2.begin(), pair2.end(), [&pair0](auto i) { return i < pair0[0]; });
            pair0[1] -= std::count_if(pair2.begin(), pair2.end(), [&pair0](auto i) { return i < pair0[1]; });
            return tensor.trace(pair2).trace(pair1).trace(pair0);
        }
    }

}