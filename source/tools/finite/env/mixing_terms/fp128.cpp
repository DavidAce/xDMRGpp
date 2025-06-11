#include "../mixing_terms/mixing_terms.impl.h"

using T = fp128;

template void tools::finite::env::internal::merge_mixing_terms_MP_N0(const StateFinite<T>      &state, //
                                                                     MpsSite<T>                &mpsL,  //
                                                                     const Eigen::Tensor<T, 3> &MP,    //
                                                                     MpsSite<T>                &mpsR,  //
                                                                     const Eigen::Tensor<T, 3> &N0,    //
                                                                     const svd::config         &svd_cfg);

template void tools::finite::env::internal::merge_mixing_terms_N0_MP(const StateFinite<T>      &state, //
                                                                     MpsSite<T>                &mpsL,  //
                                                                     const Eigen::Tensor<T, 3> &N0,    //
                                                                     MpsSite<T>                &mpsR,  //
                                                                     const Eigen::Tensor<T, 3> &MP,    //
                                                                     const svd::config         &svd_cfg);

template std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> tools::finite::env::internal::get_mixing_terms_MP_N0(const Eigen::Tensor<T, 3> &M, // Gets expanded
                                                                                                                  const Eigen::Tensor<T, 3> &N, // Gets padded
                                                                                                                  const Eigen::Tensor<T, 3> &P1, //
                                                                                                                  const Eigen::Tensor<T, 3> &P2, //
                                                                                                                  const BondExpansionConfig &cfg);

template std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>> tools::finite::env::internal::get_mixing_terms_N0_MP(const Eigen::Tensor<T, 3> &N, // Gets padded
                                                                                                                  const Eigen::Tensor<T, 3> &M, // Gets expanded
                                                                                                                  const Eigen::Tensor<T, 3> &P1, //
                                                                                                                  const Eigen::Tensor<T, 3> &P2, //
                                                                                                                  const BondExpansionConfig &cfg);