#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

template<typename T> struct BondExpansionResult;
template<typename T> struct MpsSite;

namespace tools::finite::env::internal {

    template<typename T>
    void merge_mixing_terms_MP_N0(const StateFinite<T>      &state, //
                                  MpsSite<T>                &mpsL,  //
                                  const Eigen::Tensor<T, 3> &MP,    //
                                  MpsSite<T>                &mpsR,  //
                                  const Eigen::Tensor<T, 3> &N0,    //
                                  const svd::config         &svd_cfg);

    template<typename T>
    void merge_mixing_terms_N0_MP(const StateFinite<T>      &state, //
                                  MpsSite<T>                &mpsL,  //
                                  const Eigen::Tensor<T, 3> &N0,    //
                                  MpsSite<T>                &mpsR,  //
                                  const Eigen::Tensor<T, 3> &MP,    //
                                  const svd::config         &svd_cfg);

    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_mixing_terms_MP_N0(const Eigen::Tensor<T, 3> &M,  // Gets expanded
                                                                               const Eigen::Tensor<T, 3> &N,  // Gets padded
                                                                               const Eigen::Tensor<T, 3> &P1, //
                                                                               const Eigen::Tensor<T, 3> &P2, //
                                                                               const BondExpansionConfig &cfg);

    template<typename T>
    std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> get_mixing_terms_N0_MP(const Eigen::Tensor<T, 3> &N,  // Gets padded
                                                                               const Eigen::Tensor<T, 3> &M,  // Gets expanded
                                                                               const Eigen::Tensor<T, 3> &P1, //
                                                                               const Eigen::Tensor<T, 3> &P2, //
                                                                               const BondExpansionConfig &cfg);

}