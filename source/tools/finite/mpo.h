#pragma once

#include "config/enums.h"
#include "math/tenx/fwd_decl.h"
#include <complex>
#include <math/float.h>
#include <vector>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
namespace svd {
    struct config;
}
/* clang-format off */
namespace tools::finite::mpo {
    template<typename Scalar> extern std::pair<Eigen::Tensor<Scalar, 4>,Eigen::Tensor<Scalar, 4>>
                                          swap_mpo    (const Eigen::Tensor<Scalar, 4> & mpoL, const Eigen::Tensor<Scalar, 4> & mpoR);
    template<typename Scalar> extern void swap_sites  (ModelFinite<Scalar> & model, size_t posL, size_t posR, std::vector<size_t> & sites);

    template<typename Scalar>
    extern std::vector<Eigen::Tensor<Scalar,4>> get_mpos_with_edges (const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge);
    // extern std::vector<Eigen::Tensor<cx128,4>>  get_mpos_with_edges_t (const std::vector<Eigen::Tensor<cx128,4>> & mpos, const Eigen::Tensor<cx128,1> & Ledge, const Eigen::Tensor<cx128,1> & Redge);

    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_compressed_mpos (std::vector<Eigen::Tensor<Scalar, 4>> mpos, MpoCompress mpoCompress);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_compressed_mpos (const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge, MpoCompress mpoCompress);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_svdcompressed_mpos (std::vector<Eigen::Tensor<Scalar, 4>> mpos);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_svdcompressed_mpos (const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge);
    template <typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_inverted_mpos (const std::vector<Eigen::Tensor<Scalar,4>> & mpos);
    template<typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_deparallelized_mpos (std::vector<Eigen::Tensor<Scalar, 4>> mpos);
    template<typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_deparallelized_mpos (const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge);
    template<typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_deprojected_mpos (const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model);
    template<typename Scalar> extern std::vector<Eigen::Tensor<Scalar,4>> get_merged_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_dn,
                                                              const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_md,
                                                              const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_up,
                                                              const svd::config &svd_cfg);


}

/* clang-format on */
