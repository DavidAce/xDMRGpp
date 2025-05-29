#include "matvec_mpos.impl.h"


using Scalar = cx128;
template class MatVecMPOS<Scalar>;


template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp32>>>  &mpos_, const env_pair<const EnvEne<fp32> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp64>>>  &mpos_, const env_pair<const EnvEne<fp64> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp128>>> &mpos_, const env_pair<const EnvEne<fp128> &> &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx32>>>  &mpos_, const env_pair<const EnvEne<cx32> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx64>>>  &mpos_, const env_pair<const EnvEne<cx64> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos_, const env_pair<const EnvEne<cx128> &> &envs_);

template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp32>>> &mpos_, const env_pair<const EnvVar<fp32> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp64>>> &mpos_, const env_pair<const EnvVar<fp64> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp128>>>&mpos_, const env_pair<const EnvVar<fp128> &> &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx32>>> &mpos_, const env_pair<const EnvVar<cx32> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx64>>> &mpos_, const env_pair<const EnvVar<cx64> &>  &envs_);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx128>>>&mpos_, const env_pair<const EnvVar<cx128> &> &envs_);

template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp32>>>  &mpos_, const env_pair<const EnvVar<fp32> &>  &enva_, const env_pair<const EnvEne<fp32> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp64>>>  &mpos_, const env_pair<const EnvVar<fp64> &>  &enva_, const env_pair<const EnvEne<fp64> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp128>>> &mpos_, const env_pair<const EnvVar<fp128> &> &enva_, const env_pair<const EnvEne<fp128> &> &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx32>>>  &mpos_, const env_pair<const EnvVar<cx32> &>  &enva_, const env_pair<const EnvEne<cx32> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx64>>>  &mpos_, const env_pair<const EnvVar<cx64> &>  &enva_, const env_pair<const EnvEne<cx64> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos_, const env_pair<const EnvVar<cx128> &> &enva_, const env_pair<const EnvEne<cx128> &> &envb);

template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp32>>>  &mpos_, const env_pair<const EnvEne<fp32> &>  &enva_, const env_pair<const EnvVar<fp32> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp64>>>  &mpos_, const env_pair<const EnvEne<fp64> &>  &enva_, const env_pair<const EnvVar<fp64> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<fp128>>> &mpos_, const env_pair<const EnvEne<fp128> &> &enva_, const env_pair<const EnvVar<fp128> &> &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx32>>>  &mpos_, const env_pair<const EnvEne<cx32> &>  &enva_, const env_pair<const EnvVar<cx32> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx64>>>  &mpos_, const env_pair<const EnvEne<cx64> &>  &enva_, const env_pair<const EnvVar<cx64> &>  &envb);
template MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<cx128>>> &mpos_, const env_pair<const EnvEne<cx128> &> &enva_, const env_pair<const EnvVar<cx128> &> &envb);
