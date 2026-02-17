#pragma once
#include "tensors/site/env/EnvBase.h"

/*! \brief Environment class with energy MPOs for environment blocks och type Left or Right corresponding to a single site.
 */

template<typename Scalar_>
class EnvEne : public EnvBase<Scalar_> {
    public:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using EnvBase<Scalar>::enlarge;
    using EnvBase<Scalar>::EnvBase;
    using EnvBase<Scalar>::set_edge_dims;
    using EnvBase<Scalar>::tag;
    using EnvBase<Scalar>::side;
    using EnvBase<Scalar>::get_position;
    using EnvBase<Scalar>::blkx2;
    using EnvBase<Scalar>::has_block;
    using EnvBase<Scalar>::build_blkx2;
    using EnvBase<Scalar>::get_unique_id;
    using EnvBase<Scalar>::unique_id;
    using EnvBase<Scalar>::unique_id_mpo;
    using EnvBase<Scalar>::unique_id_env;
    using EnvBase<Scalar>::unique_id_mps;
    using EnvBase<Scalar>::assert_unique_id;

    explicit EnvEne(std::string side_, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo);
    [[nodiscard]] EnvEne<Scalar> enlarge(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const;
    void                         refresh(const EnvEne &env, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo);
    void                         set_block(const Eigen::Tensor<Scalar, 3> &blk, const EnvEne &env, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo);
    void                         set_block_raw(const Eigen::Tensor<Scalar, 3> &blk);
    void                         set_edge_dims(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) final;

    template<typename T>
    std::unique_ptr<EnvEne<T>> cast() const {
        return this->template cast_typed<T, EnvEne>("ene");
    }
};
