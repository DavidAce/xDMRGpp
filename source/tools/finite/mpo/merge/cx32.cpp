#include "impl.h"

using Scalar = cx32;

/* clang-format off */

template std::vector<Eigen::Tensor<Scalar, 4>> tools::finite::mpo::get_merged_mpos(
                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_dn,
                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_md,
                                                                          const std::vector<Eigen::Tensor<Scalar, 4>> &mpos_up,
                                                                          const svd::config &cfg);