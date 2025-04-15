#pragma once
#include "tensors/site/env/EnvBase.h"

/*! \brief Environment class with energy MPOs for environment blocks och type Left or Right corresponding to a single site.
 */

template<typename Scalar = cx64>
class EnvEne : public EnvBase<Scalar> {
    public:
    using EnvBase<Scalar>::enlarge;
    using EnvBase<Scalar>::EnvBase;
    using EnvBase<Scalar>::set_edge_dims;
    using EnvBase<Scalar>::tag;
    using EnvBase<Scalar>::side;
    using EnvBase<Scalar>::get_position;
    using EnvBase<Scalar>::block;
    using EnvBase<Scalar>::has_block;
    using EnvBase<Scalar>::build_block;
    using EnvBase<Scalar>::get_unique_id;
    using EnvBase<Scalar>::unique_id;
    using EnvBase<Scalar>::unique_id_mpo;
    using EnvBase<Scalar>::unique_id_env;
    using EnvBase<Scalar>::unique_id_mps;

    explicit EnvEne(std::string side_, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo);
    [[nodiscard]] EnvEne<Scalar> enlarge(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) const;
    void                         refresh(const EnvEne &env, const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo);
    void                         set_edge_dims(const MpsSite<Scalar> &mps, const MpoSite<Scalar> &mpo) final;
};
