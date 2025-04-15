#pragma once
#include "math/float.h"
#include "math/svd/config.h"
#include "math/tenx.h"
#include <functional>
#include <optional>
#include <vector>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class MpoSite;
template<typename Scalar>
class MpsSite;
template<typename T>
struct env_pair;

namespace tools::finite::measure {
    template<typename T>
    struct LocalObservableOp {
        Eigen::Tensor<T, 2> op;
        long                pos;
        mutable bool        used = false;
    };
    template<typename T>
    struct LocalObservableMpo {
        Eigen::Tensor<T, 4> mpo;
        long                pos;
        mutable bool        used = false;
    };

    /* clang-format off */
    template<typename CalcType, typename Scalar,typename OpType>
    [[nodiscard]] extern Scalar                   expectation_value      (const StateFinite<Scalar> & state, const std::vector<LocalObservableOp<OpType>> & ops);
    template<typename CalcType, typename Scalar, typename MpoType>
    [[nodiscard]] extern Scalar                   expectation_value      (const StateFinite<Scalar> & state, const std::vector<LocalObservableMpo<MpoType>> & mpos);
    template<typename CalcType, typename Scalar, typename MpoType>
    [[nodiscard]] extern Scalar                   expectation_value      (const StateFinite<Scalar> & state1, const StateFinite<Scalar> & state2, const std::vector<Eigen::Tensor<MpoType,4>> & mpos);

    template<typename CalcType, typename Scalar, typename MpoType>
    [[nodiscard]] extern Scalar                        expectation_value    (const StateFinite<Scalar> & state1, const StateFinite<Scalar> & state2,
                                                                             const std::vector<Eigen::Tensor<MpoType,4>> & mpos,
                                                                             const Eigen::Tensor<MpoType,1> & ledge,
                                                                             const Eigen::Tensor<MpoType,1> & redge);
    template<typename CalcType, typename Scalar, typename EnvType>
    Scalar                                        expectation_value    (const Eigen::Tensor<Scalar, 3> &mpsBra,
                                                                        const Eigen::Tensor<Scalar, 3> &mpsKet,
                                                                        const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos,
                                                                        const env_pair<EnvType> &envs);

    template<typename CalcType, typename Scalar, typename EnvType>
    [[nodiscard]] extern Scalar                   expectation_value      (const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> & mpsBra,
                                                                          const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> & mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> & mpos,
                                                                          const env_pair<EnvType> & envs);

    template<typename CalcType, typename Scalar, typename EnvType>
    [[nodiscard]] extern Scalar                     expectation_value    (const Eigen::Tensor<Scalar, 3> &mpsBra,
                                                                          const Eigen::Tensor<Scalar, 3> &mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos,
                                                                          const env_pair<EnvType> &envs,
                                                                          std::optional<svd::config> svd_cfg);
    template<typename CalcType, typename Scalar, typename EnvType>
    [[nodiscard]] extern Scalar                     expectation_value    (const Eigen::Tensor<Scalar, 3> &multisite_mps,
                                                                          const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos,
                                                                          const env_pair<EnvType> &envs,
                                                                          std::optional<svd::config> svd_cfg);

    template<typename CalcType, typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar, 1>   expectation_values     (const StateFinite<Scalar> & state, const Eigen::Tensor<cx64,2> &op);
    template<typename CalcType, typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar, 1>   expectation_values     (const StateFinite<Scalar> & state, const Eigen::Tensor<cx64,4> &mpo);
    template<typename CalcType, typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar, 1>   expectation_values     (const StateFinite<Scalar> & state, const Eigen::Matrix2cd &op);
    /* clang-format on */
}