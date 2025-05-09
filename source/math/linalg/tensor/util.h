#pragma once
#include "../common.h"
#include <array>
#include <iterator>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

namespace linalg::tensor {

    // Shorthand for the list of index pairs.
    template<auto N>
    using idxlistpair = std::array<Eigen::IndexPair<Eigen::Index>, N>;

    inline constexpr idxlistpair<0> idx() { return {}; }

    template<std::size_t N, typename idxType>
    constexpr idxlistpair<N> idx(const idxType (&list1)[N], const idxType (&list2)[N]) {
        // Use numpy-style indexing for contraction. Each list contains a list of indices to be contracted for the respective
        // tensors. This function zips them together into pairs as used in Eigen::Tensor module. This does not sort the indices in decreasing order.
        idxlistpair<N> pairlistOut;
        for(size_t i = 0; i < N; i++) {
            pairlistOut[i] = Eigen::IndexPair<Eigen::Index>{static_cast<Eigen::Index>(list1[i]), static_cast<Eigen::Index>(list2[i])};
        }
        return pairlistOut;
    }

    template<typename Scalar = double>
    Eigen::Tensor<Scalar, 2> identity(const Eigen::Index &dim) {
        Eigen::Tensor<double, 1> tensor(dim);
        tensor.setConstant(1);
        return tensor.inflate(std::array<long, 1>{tensor.size() + 1}).reshape(std::array<long, 2>{tensor.size(), tensor.size()}).template cast<Scalar>();
    }

    template<typename Scalar, int rank>
    Eigen::Tensor<Scalar, rank> mirror(const Eigen::Tensor<Scalar, rank> &tensor) {
        /*
         Returns a mirrored tensor

         Example: Starting with A
                0 1 2 3
                | | | |
               [  A   ]
               | | | |
               4 5 6 7

         returns

                3 2 1 0
                | | | |
               [  A   ]
               | | | |
               7 6 5 4

         This is useful for compaitibility with kronecker products which gives results indexed right to left.

        */
        if constexpr(rank <= 2)
            return tensor;
        else {
            std::array<Eigen::Index, rank> shf_idx{};
            for(size_t i = 0; i < static_cast<size_t>(rank); i++) { shf_idx[i] = static_cast<Eigen::Index>(i); }
            std::reverse(shf_idx.begin(), shf_idx.begin() + rank / 2);
            std::reverse(shf_idx.begin() + rank / 2, shf_idx.end());
            return tensor.shuffle(shf_idx);
        }
    }

    template<int rank>
    Eigen::IndexPair<Eigen::Index> mirror_idx_pair(const Eigen::IndexPair<Eigen::Index> &idx_pair) {
        if constexpr(rank <= 2)
            return idx_pair;
        else {
            std::array<Eigen::Index, rank> shf_idx{};
            for(size_t i = 0; i < static_cast<size_t>(rank); i++) { shf_idx[i] = static_cast<Eigen::Index>(i); }
            std::reverse(shf_idx.begin(), shf_idx.begin() + rank / 2);
            std::reverse(shf_idx.begin() + rank / 2, shf_idx.end());
            Eigen::IndexPair<Eigen::Index> idx_pair_mirrored{};
            idx_pair_mirrored.first  = std::distance(shf_idx.begin(), std::find(shf_idx.begin(), shf_idx.end(), idx_pair.first));
            idx_pair_mirrored.second = std::distance(shf_idx.begin(), std::find(shf_idx.begin(), shf_idx.end(), idx_pair.second));
            return idx_pair_mirrored;
        }
    }
}