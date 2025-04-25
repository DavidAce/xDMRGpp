#pragma once
#include "matvec_mpos.h"
#include "tensors/edges/EdgesFinite.h"
#include "tensors/site/env/EnvEne.h"
#include "tensors/site/env/EnvVar.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tid/tid.h"

template<typename Scalar>
template<typename T, typename EnvType>
MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<T>>> &mpos_, /*!< The Hamiltonian MPO's  */
                               const env_pair<const EnvType &>                             &envs_  /*!< The left and right environments.  */
) {
    static_assert(sfinae::is_any_v<EnvType, EnvEne<T>, EnvVar<T>>);
    mpos_A.reserve(mpos_.size());
    fullsystem = envs_.L.get_sites() == 0 and envs_.R.get_sites() == 0; //  mpos.size() == settings::model::model_size;

    if constexpr(std::is_same_v<EnvType, EnvEne<T>>) {
        for(const auto &mpo_ : mpos_) mpos_A.emplace_back(mpo_.get().template MPO_as<Scalar>());
    }
    if constexpr(std::is_same_v<EnvType, EnvVar<T>>) {
        for(const auto &mpo_ : mpos_) mpos_A.emplace_back(mpo_.get().template MPO2_as<Scalar>());
    }
    envL_A = envs_.L.template get_block_as<Scalar>();
    envR_A = envs_.R.template get_block_as<Scalar>();

    long spin_dim = 1;
    for(const auto &mpo : mpos_A) spin_dim *= mpo.dimension(2);
    spindims.reserve(mpos_A.size());
    for(const auto &mpo : mpos_A) spindims.emplace_back(mpo.dimension(2));

    shape_mps = {spin_dim, envL_A.dimension(0), envR_A.dimension(0)};
    size_mps  = spin_dim * envL_A.dimension(0) * envR_A.dimension(0);

    // if(mpos.size() == settings::model::model_size) {
    //     auto t_spm = tid::ur("t_spm");
    //     t_spm.tic();
    //     eig::log->info("making sparse matrix ... ", t_spm.get_last_interval());
    //     sparseMatrix = get_sparse_matrix();
    //     t_spm.toc();
    //     eig::log->info("making sparse matrix ... {:.3e} s | nnz {} / {} = {:.16f}", t_spm.get_last_interval(), sparseMatrix.nonZeros(), sparseMatrix.size(),
    //                    static_cast<double>(sparseMatrix.nonZeros()) / static_cast<double>(sparseMatrix.size()));
    // }

    // If we have 5 or fewer mpos, it is faster to just merge them once and apply them in one contraction.
    if(mpos_A.size() <= 5) {
        constexpr auto contract_idx    = tenx::idx({1}, {0});
        constexpr auto shuffle_idx     = tenx::array6{0, 3, 1, 4, 2, 5};
        auto          &threads         = tenx::threads::get();
        auto           contracted_mpos = mpos_A.front();
        for(size_t idx = 0; idx + 1 < mpos_A.size(); ++idx) {
            const auto &mpoL = idx == 0 ? mpos_A[idx] : contracted_mpos;
            const auto &mpoR = mpos_A[idx + 1];
            auto new_dims    = std::array{mpoL.dimension(0), mpoR.dimension(1), mpoL.dimension(2) * mpoR.dimension(2), mpoL.dimension(3) * mpoR.dimension(3)};
            auto temp        = Eigen::Tensor<Scalar, 4>(new_dims);
            temp.device(*threads->dev) = mpoL.contract(mpoR, contract_idx).shuffle(shuffle_idx).reshape(new_dims);
            contracted_mpos            = std::move(temp);
        }
        mpos_A   = {contracted_mpos}; // Replace by a single pre-contracted mpo
        spindims = {mpos_A.front().dimension(2)};
    } else {
        // We pre-shuffle each mpo to speed up the sequential contraction
        for(const auto &mpo : mpos_A) mpos_A_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }

    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}

template<typename Scalar>
template<typename T, typename EnvTypeA, typename EnvTypeB>
MatVecMPOS<Scalar>::MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<T>>> &mpos_, /*!< The Hamiltonian MPO's  */
                               const env_pair<const EnvTypeA &>                            &enva_, /*!< The left and right environments.  */
                               const env_pair<const EnvTypeB &>                            &envb_)
    : MatVecMPOS(mpos_, enva_) {
    // static_assert(sfinae::is_any_v<EnvTypeA, EnvVar>);
    // static_assert(sfinae::is_any_v<EnvTypeB, EnvEne>);
    if constexpr(std::is_same_v<EnvTypeB, EnvEne<T>>) {
        for(const auto &mpo_ : mpos_) mpos_B.emplace_back(mpo_.get().template MPO_as<Scalar>());
    }
    if constexpr(std::is_same_v<EnvTypeB, EnvVar<T>>) {
        for(const auto &mpo_ : mpos_) mpos_B.emplace_back(mpo_.get().template MPO2_as<Scalar>());
    }
    envL_B = envb_.L.template get_block_as<Scalar>();
    envR_B = envb_.R.template get_block_as<Scalar>();

    if(mpos_B.size() <= 5) {
        constexpr auto contract_idx    = tenx::idx({1}, {0});
        constexpr auto shuffle_idx     = tenx::array6{0, 3, 1, 4, 2, 5};
        auto          &threads         = tenx::threads::get();
        auto           contracted_mpos = mpos_B.front();
        for(size_t idx = 0; idx + 1 < mpos_B.size(); ++idx) {
            const auto &mpoL = idx == 0 ? mpos_B[idx] : contracted_mpos;
            const auto &mpoR = mpos_B[idx + 1];
            auto new_dims    = std::array{mpoL.dimension(0), mpoR.dimension(1), mpoL.dimension(2) * mpoR.dimension(2), mpoL.dimension(3) * mpoR.dimension(3)};
            auto temp        = Eigen::Tensor<Scalar, 4>(new_dims);
            temp.device(*threads->dev) = mpoL.contract(mpoR, contract_idx).shuffle(shuffle_idx).reshape(new_dims);
            contracted_mpos            = std::move(temp);
        }
        mpos_B = {contracted_mpos}; // Replace by a single pre-contracted mpo
    } else {
        // We pre-shuffle each mpo to speed up the sequential contraction
        for(const auto &mpo : mpos_B) mpos_B_shf.emplace_back(mpo.shuffle(tenx::array4{2, 3, 0, 1}));
    }
}
