#include "expansion_terms.impl.h"

using T = cx64;

template std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>>
    tools::finite::env::internal::get_expansion_terms_MP_N0(const Eigen::Tensor<T, 3>                     &M,   // Gets expanded
                                                            const Eigen::Tensor<T, 3>                     &N,   // Gets padded
                                                            const Eigen::Tensor<T, 3>                     &P1,  //
                                                            const Eigen::Tensor<T, 3>                     &P2,  //
                                                            [[maybe_unused]] const BondExpansionResult<T> &res, //
                                                            [[maybe_unused]] const Eigen::Index            bond_max);

template std::pair<Eigen::Tensor<T, 3>, Eigen::Tensor<T, 3>>
    tools::finite::env::internal::get_expansion_terms_N0_MP(const Eigen::Tensor<T, 3>                     &N,   // Gets padded
                                                            const Eigen::Tensor<T, 3>                     &M,   // Gets expanded
                                                            const Eigen::Tensor<T, 3>                     &P1,  //
                                                            const Eigen::Tensor<T, 3>                     &P2,  //
                                                            [[maybe_unused]] const BondExpansionResult<T> &res, //
                                                            [[maybe_unused]] const Eigen::Index            bond_max);
