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
namespace tools::finite::mpo {
    template<typename Scalar>
    using Real = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar>
    using Cplx = std::complex<Real<Scalar>>;
    /* clang-format off */
    template <typename Scalar>  std::pair<Eigen::Tensor<Scalar, 4>,Eigen::Tensor<Scalar, 4>>
                                          swap_mpo    (const Eigen::Tensor<Scalar, 4> & mpoL, const Eigen::Tensor<Scalar, 4> & mpoR);
    template <typename Scalar> void swap_sites  (ModelFinite<Scalar> & model, size_t posL, size_t posR, std::vector<size_t> & sites);

    template <typename Scalar>  std::vector<Eigen::Tensor<Scalar,4>> get_mpos_with_edges (const std::vector<Eigen::Tensor<Scalar,4>> & mpos, const Eigen::Tensor<Scalar,1> & Ledge, const Eigen::Tensor<Scalar,1> & Redge);
    template <typename Scalar>  std::vector<Eigen::Tensor<Scalar,4>> get_svdcompressed_mpos (std::vector<Eigen::Tensor<Scalar, 4>> mpos);
    template <typename Scalar>  std::vector<Eigen::Tensor<Scalar,4>> get_inverted_mpos (const std::vector<Eigen::Tensor<Scalar,4>> & mpos);
    template <typename Scalar>  std::vector<Eigen::Tensor<Scalar,4>> get_deparallelized_mpos (std::vector<Eigen::Tensor<Scalar, 4>> mpos);
    template <typename Scalar>  std::vector<Eigen::Tensor<Scalar,4>> get_merged_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_dn,
                                                              const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_md,
                                                              const std::vector<Eigen::Tensor<Scalar, 4>> & mpos_up,
                                                              const svd::config &svd_cfg);
    /* clang-format on */

    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 4>> get_compressed_mpos(std::vector<Eigen::Tensor<Scalar, 4>> mpos, MpoCompress mpoComp) {
        switch(mpoComp) {
            case MpoCompress::NONE: return mpos;
            case MpoCompress::SVD: return get_svdcompressed_mpos(mpos);
            case MpoCompress::DPL: return get_deparallelized_mpos(mpos);
            default: return mpos;
        }
    }
    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 4>> get_svdcompressed_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge,
                                                                 const Eigen::Tensor<Scalar, 1> &Redge) {
        return get_svdcompressed_mpos(get_mpos_with_edges(mpos, Ledge, Redge));
    }
    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 4>> get_deparallelized_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge,
                                                                  const Eigen::Tensor<Scalar, 1> &Redge) {
        return get_deparallelized_mpos(get_mpos_with_edges(mpos, Ledge, Redge));
    }
    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 4>> get_compressed_mpos(const std::vector<Eigen::Tensor<Scalar, 4>> &mpos, const Eigen::Tensor<Scalar, 1> &Ledge,
                                                              const Eigen::Tensor<Scalar, 1> &Redge, MpoCompress mpoComp) {
        switch(mpoComp) {
            case MpoCompress::NONE: return mpos;
            case MpoCompress::SVD: return get_svdcompressed_mpos(mpos, Ledge, Redge);
            case MpoCompress::DPL: return get_deparallelized_mpos(mpos, Ledge, Redge);
            default: return mpos;
        }
    }
}
