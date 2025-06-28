
#pragma once
#include "math/float.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

template<typename T> struct BondExpansionResult;
template<typename T> class MpsSite;

namespace tools::finite::env::internal {

    template<typename T>
    void merge_rexpansion_terms_MP_N0(MpsSite<T> &mpsL, const Eigen::Tensor<T, 3> &M_P, MpsSite<T> &mpsR, const Eigen::Tensor<T, 3> &N_0, Eigen::Index bond_max,
                                      T schmidt_pad_value);

    template<typename T>
    void merge_rexpansion_terms_N0_MP(MpsSite<T> &mpsL, const Eigen::Tensor<T, 3> &N_0, MpsSite<T> &mpsR, const Eigen::Tensor<T, 3> &M_P, Eigen::Index bond_max,
                                      T schmid_pad_value);

    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_expansion_terms_MP_N0(const Eigen::Tensor<T, 3>    &M,        // Gets expanded
                                                                                  const Eigen::Tensor<T, 3>    &N,        // Gets padded
                                                                                  const Eigen::Tensor<T, 3>    &P1,       //
                                                                                  const Eigen::Tensor<T, 3>    &P2,       //
                                                                                  const BondExpansionResult<T> &res,      //
                                                                                  Eigen::Index                  bond_max, //
                                                                                  T                             pad_value);

    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_expansion_terms_N0_MP(const Eigen::Tensor<T, 3>    &N,        // Gets padded
                                                                                  const Eigen::Tensor<T, 3>    &M,        // Gets expanded
                                                                                  const Eigen::Tensor<T, 3>    &P1,       //
                                                                                  const Eigen::Tensor<T, 3>    &P2,       //
                                                                                  const BondExpansionResult<T> &res,      //
                                                                                  Eigen::Index                  bond_max, //
                                                                                  T                             pad_value);
}