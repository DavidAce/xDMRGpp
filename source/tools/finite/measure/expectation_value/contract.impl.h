#pragma once
#include "contract.h" 

template<typename Scalar>
void contract_op_M_1_0(TN<Scalar, 3> &temp, const TN<Scalar, 2> &op, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    temp.resize(op.dimension(0), M.dimension(1), M.dimension(2));
    temp.device(*threads->dev) = op.contract(M, tenx::idx({1}, {0}));
}

template<typename Scalar>
TN<Scalar, 2> contract_chain_M_Mconj_0_1_01_10(const TN<Scalar, 2> &chain, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    TN<Scalar, 2> temp;
    temp.resize(M.dimension(2), M.dimension(2));
    temp.device(*threads->dev) = chain.contract(M, tenx::idx({0}, {1})).contract(M.conjugate(), tenx::idx({0, 1}, {1, 0}));
    return temp;
}

template<typename Scalar>
void contract_M_Ledge3_mpo_Mconj_0_1_0_1_013_023(TN<Scalar, 3> &temp, const TN<Scalar, 3> &M, const TN<Scalar, 3> &Ledge3, const TN<Scalar, 4> &mpo,
                                                        const ThreadPtr &threads) {
    temp.resize(M.dimension(2), M.dimension(2), mpo.dimension(1));
    temp.device(*threads->dev) = M.contract(Ledge3, tenx::idx({0}, {1})) //
                                     .contract(mpo, tenx::idx({0}, {1})) //
                                     .contract(M.conjugate(), tenx::idx({0, 1, 3}, {0, 2, 3}));
}

template<typename Scalar>
TN<Scalar, 0> contract_Ledge3_Redge3_012_012(const TN<Scalar, 3> &Ledge3, const TN<Scalar, 3> &Redge3, const ThreadPtr &threads) {
    TN<Scalar, 0> res;
    res.device(*threads->dev) = Ledge3.contract(Redge3, tenx::idx({0, 1, 2}, {0, 1, 2}));
    return res;
}


template<typename Scalar>
void contract_mps1_mpo_mps2_0_2_4_0(TN<Scalar, 4> &result, const TN<Scalar, 3> &mps1, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &mps2,
                                           const ThreadPtr &threads) {
    auto           dim4 = tenx::array4{mpo.dimension(0) * mps1.dimension(1) * mps2.dimension(1), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
    constexpr auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};
    result.resize(dim4);
    result.device(*threads->dev) = mps1.contract(mpo, tenx::idx({0}, {2})).contract(mps2, tenx::idx({4}, {0})).shuffle(shf6).reshape(dim4);
}

template<typename Scalar>
void contract_res_mps1conj_mpo_mps2_1_1_13_02_14_10(TN<Scalar, 4> &tmp, const TN<Scalar, 4> &result, const TN<Scalar, 3> &mps1, const TN<Scalar, 4> &mpo,
                                                           const TN<Scalar, 3> &mps2, const ThreadPtr &threads) {
    auto dim4 = tenx::array4{result.dimension(0), mps1.dimension(2), mpo.dimension(1), mps2.dimension(2)};
    tmp.resize(dim4);
    tmp.device(*threads->dev) = result
                                    .contract(mps1.conjugate(), tenx::idx({1}, {1})) //
                                    .contract(mpo, tenx::idx({1, 3}, {0, 2}))        //
                                    .contract(mps2, tenx::idx({1, 4}, {1, 0}));      //
}

template<typename Scalar>
void contract_resL_ket_mpo_braconj_0_1_02_02_03_10(TN<Scalar, 3> &tmp, const TN<Scalar, 3> &resL, const TN<Scalar, 3> &ket, const TN<Scalar, 4> &mpo,
                                                          const TN<Scalar, 3> &bra, const ThreadPtr &threads) {
    auto dim3 = tenx::array3{ket.dimension(2), mpo.dimension(1), bra.dimension(2)};
    tmp.resize(dim3);
    tmp.device(*threads->dev) = resL.contract(ket, tenx::idx({0}, {1}))       //
                                    .contract(mpo, tenx::idx({0, 2}, {0, 2})) //
                                    .contract(bra.conjugate(), tenx::idx({0, 3}, {1, 0}));
}


template<typename Scalar>
TN<Scalar, 0> contract_resL_envR_012_021(const TN<Scalar, 3> &resL, const TN<Scalar, 3> &envR, const ThreadPtr &threads) {
    TN<Scalar, 0> res;
    res.device(*threads->dev) = resL.contract(envR, tenx::idx({0, 1, 2}, {0, 2, 1}));
    return res;
}