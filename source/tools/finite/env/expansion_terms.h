
#pragma once
#include "math/float.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

template<typename T> struct BondExpansionResult;

namespace tools::finite::env::internal {
    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_expansion_terms_MP_N0(const Eigen::Tensor<T, 3>    &M,   // Gets expanded
                                                                                  const Eigen::Tensor<T, 3>    &N,   // Gets padded
                                                                                  const Eigen::Tensor<T, 3>    &P1,  //
                                                                                  const Eigen::Tensor<T, 3>    &P2,  //
                                                                                  const BondExpansionResult<T> &res, //
                                                                                  const Eigen::Index            bond_max);

    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_expansion_terms_N0_MP(const Eigen::Tensor<T, 3>    &N,   // Gets padded
                                                                                  const Eigen::Tensor<T, 3>    &M,   // Gets expanded
                                                                                  const Eigen::Tensor<T, 3>    &P1,  //
                                                                                  const Eigen::Tensor<T, 3>    &P2,  //
                                                                                  const BondExpansionResult<T> &res, //
                                                                                  const Eigen::Index            bond_max);

}