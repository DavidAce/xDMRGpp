#pragma once
#include "../common.h"
#include "math/cast.h"
#include <array>
#include <fmt/core.h>
#include <io/fmt_custom.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace linalg::tensor {

    template<typename Scalar, int rank1, int rank2>
    Eigen::Tensor<Scalar, rank1 + rank2> outer(const Eigen::Tensor<Scalar, rank1> &tensor1, const Eigen::Tensor<Scalar, rank2> &tensor2) {
        std::array<Eigen::IndexPair<Eigen::Index>, 0> idx{};
        return tensor1.contract(tensor2, idx);
    }
    template<typename Scalar>
    Eigen::Tensor<Scalar, 4> outer(const EigenMatrix<Scalar> &matrix1, const EigenMatrix<Scalar> &matrix2) {
        std::array<Eigen::IndexPair<Eigen::Index>, 0> idx{};
        auto tensor1 = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(matrix1.data(), matrix1.rows(), matrix1.cols());
        auto tensor2 = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(matrix2.data(), matrix2.rows(), matrix2.cols());
        return tensor1.contract(tensor2, idx);
    }

    template<typename Scalar, int rankA, int rankB>
    Eigen::Tensor<Scalar, rankA + rankB> kronecker(const Eigen::Tensor<Scalar, rankA> &tensorA, const Eigen::Tensor<Scalar, rankB> &tensorB) {
        /*
         Returns the equivalent kronecker product for a tensor following left-to-right index order
            \verbatim
                0  1       0 1 2         0 1 4 5 6           0 1 2 3 4
                |  |       | | |         | | | | |           | | | | |
                [ A ]  ⊗  [  B  ]   =   [   AB    ]  ===>   [   AB    ] (shuffle 0,1,4,5,6,2,3,7,8,9)
                |  |       | | |         | | | | |           | | | | |
                2  3       3 4 5         2 3 7 8 9           5 6 7 8 9
            \endverbatim
         */
        constexpr Eigen::Index                  topA  = static_cast<Eigen::Index>(rankA) / 2;
        constexpr Eigen::Index                  topB  = static_cast<Eigen::Index>(rankB) / 2;
        constexpr Eigen::Index                  topAB = topA + topB;
        std::array<Eigen::Index, rankA + rankB> shf{};
        for(size_t i = 0; i < shf.size(); i++) {
            if(i < topAB) {
                if(i < topA)
                    shf[i] = static_cast<Eigen::Index>(i);
                else
                    shf[i] = static_cast<Eigen::Index>(i) + (rankA - topA);
            } else {
                if(i - topAB < topA)
                    shf[i] = static_cast<Eigen::Index>(i) - topB;
                else
                    shf[i] = shf[i] = static_cast<Eigen::Index>(i);
            }
        }
        return linalg::tensor::outer(tensorA, tensorB).shuffle(shf);
    }

    template<typename Scalar>
    Eigen::Tensor<Scalar, 4> kronecker(const EigenMatrix<Scalar> &A, const EigenMatrix<Scalar> &B) {
        Eigen::Tensor<Scalar, 2> tensorA = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(A.data(), A.rows(), A.cols());
        Eigen::Tensor<Scalar, 2> tensorB = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(B.data(), B.rows(), B.cols());
        return linalg::tensor::kronecker(tensorA, tensorB);
    }
    template<typename Scalar, int rankA>
    Eigen::Tensor<Scalar, rankA + 2> kronecker(const Eigen::Tensor<Scalar, rankA> &tensorA, const EigenMatrix<Scalar> &B) {
        Eigen::Tensor<Scalar, 2> tensorB = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(B.data(), B.rows(), B.cols());
        return linalg::tensor::kronecker(tensorA, tensorB);
    }
    template<typename Scalar, int rankB>
    Eigen::Tensor<Scalar, rankB + 2> kronecker(const EigenMatrix<Scalar> &A, const Eigen::Tensor<Scalar, rankB> &tensorB) {
        Eigen::Tensor<Scalar, 2> tensorA = Eigen::TensorMap<const Eigen::Tensor<const Scalar, 2>>(A.data(), A.rows(), A.cols());
        return linalg::tensor::kronecker(tensorA, tensorB);
    }

}