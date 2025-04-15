#pragma once
#include "math/float.h"
#include "math/svd/config.h"
#include "math/tenx/fwd_decl.h"
#include <complex>
#include <deque>
#include <optional>
#include <tuple>
#include <vector>

template<typename Scalar>
class MpsSite;
namespace tools::common::split {
    /* clang-format off */
    template<typename Scalar, typename T>
    extern std::vector<MpsSite<Scalar>> split_mps (const Eigen::Tensor<T,3> & multisite_mps,
                                           const std::vector<long>         & spin_dims,
                                           const std::vector<size_t>       & positions,
                                           long                              center_position,
                                           std::optional<svd::config>        svd_cfg);


    namespace internal{
        template<typename Scalar, typename T>
        extern std::vector<MpsSite<Scalar>>
                    split_mps_into_As(const Eigen::Tensor<T,3> & multisite_mps,
                                      const std::vector<long>       & spin_dims,
                                      const std::vector<size_t>     & positions,
                                      long                            center_position,
                                      svd::config                   & svd_cfg);

        template<typename Scalar, typename T>
        extern std::deque<MpsSite<Scalar>>
                    split_mps_into_Bs(const Eigen::Tensor<T,3> & multisite_mps,
                                      const std::vector<long>       & spin_dims,
                                      const std::vector<size_t>     & positions,
                                      long                            center_position,
                                      svd::config                   & svd_cfg);
    }

    /* clang-format on */
}