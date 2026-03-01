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
//
// template<typename Scalar>
// void contract_mps1_mpo_mps2_0_2_4_0(TN<Scalar, 4> &result, const TN<Scalar, 3> &bra, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &ket,
//                                     const ThreadPtr &threads) {
//     auto           dim4 = tenx::array4{bra.dimension(1) * mpo.dimension(0) * ket.dimension(1), bra.dimension(2), mpo.dimension(1), ket.dimension(2)};
//     constexpr auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};
//     result.resize(dim4);
//     result.device(*threads->dev) = bra.conjugate().contract(mpo, tenx::idx({0}, {3})).contract(ket, tenx::idx({4}, {0})).shuffle(shf6).reshape(dim4);
// }

template<typename Scalar>
void contract_mps1_mpo_mps2_0_2_4_0(TN<Scalar, 4> &result, const TN<Scalar, 3> &bra, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &ket,
                                    const ThreadPtr &threads) {
    [[maybe_unused]] const long bd = bra.dimension(0);
    [[maybe_unused]] const long bL = bra.dimension(1);
    [[maybe_unused]] const long bR = bra.dimension(2);

    [[maybe_unused]] const long kd = ket.dimension(0);
    [[maybe_unused]] const long kL = ket.dimension(1);
    [[maybe_unused]] const long kR = ket.dimension(2);

    [[maybe_unused]] const long wL  = mpo.dimension(0);
    [[maybe_unused]] const long wR  = mpo.dimension(1);
    [[maybe_unused]] const long wdi = mpo.dimension(2);
    [[maybe_unused]] const long wdo = mpo.dimension(3);

    assert(bd == wdo);
    assert(kd == wdi);

    // Shuffled order after building the full 6-index tensor:
    constexpr auto shf6 = tenx::array6{0, 2, 4, 1, 3, 5};

    TN<Scalar, 5> t1(tenx::array5{bL, bR, wL, wR, wdi});
    t1.device(*threads->dev) = bra.conjugate().contract(mpo, tenx::idx({0}, {3})); // bra phys with dout

    TN<Scalar, 6> t2(tenx::array6{bL, bR, wL, wR, kL, kR});
    t2.device(*threads->dev) = t1.contract(ket, tenx::idx({4}, {0})); // wdi with ket phys

    TN<Scalar, 6> t3(tenx::array6{bL, wL, kL, bR, wR, kR});
    t3.device(*threads->dev) = t2.shuffle(shf6);

    // Final result dims: (bL*wL*kL, bR, wR, kR)
    const auto out4 = tenx::array4{bL * wL * kL, bR, wR, kR};
    result.resize(out4);
    result.device(*threads->dev) = t3.reshape(out4);
}

// template<typename Scalar>
// void contract_res_mps1conj_mpo_mps2_1_1_13_02_14_10(TN<Scalar, 4> &tmp, const TN<Scalar, 4> &result, const TN<Scalar, 3> &bra, const TN<Scalar, 4> &mpo,
//                                                     const TN<Scalar, 3> &ket, const ThreadPtr &threads) {
//     auto dim4 = tenx::array4{result.dimension(0), bra.dimension(2), mpo.dimension(1), ket.dimension(2)};
//     tmp.resize(dim4);
//     tmp.device(*threads->dev) = result
//                                     .contract(bra.conjugate(), tenx::idx({1}, {1})) //
//                                     .contract(mpo, tenx::idx({1, 3}, {0, 3}))       //
//                                     .contract(ket, tenx::idx({1, 4}, {1, 0}));      //
// }

template<typename Scalar>
void contract_res_mps1conj_mpo_mps2_1_1_13_02_14_10(TN<Scalar, 4> &tmp, const TN<Scalar, 4> &result, const TN<Scalar, 3> &bra, const TN<Scalar, 4> &mpo,
                                                    const TN<Scalar, 3> &ket, const ThreadPtr &threads) {
    [[maybe_unused]] const long bd = bra.dimension(0);
    [[maybe_unused]] const long bL = bra.dimension(1);
    [[maybe_unused]] const long bR = bra.dimension(2);

    [[maybe_unused]] const long kd = ket.dimension(0);
    [[maybe_unused]] const long kL = ket.dimension(1);
    [[maybe_unused]] const long kR = ket.dimension(2);

    [[maybe_unused]] const long wL  = mpo.dimension(0);
    [[maybe_unused]] const long wR  = mpo.dimension(1);
    [[maybe_unused]] const long wdi = mpo.dimension(2);
    [[maybe_unused]] const long wdo = mpo.dimension(3);

    // result dims are whatever your sweep builds; name them explicitly
    [[maybe_unused]] const long r0 = result.dimension(0); // aggregated left index (includes earlier bL*wL*kL)
    [[maybe_unused]] const long r1 = result.dimension(1); // bra bond to be contracted with bra bL
    [[maybe_unused]] const long r2 = result.dimension(2); // mpo left bond to be contracted with wL
    [[maybe_unused]] const long r3 = result.dimension(3); // ket bond carried along

    assert(r1 == bL);
    assert(r2 == wL);
    assert(bd == wdo);
    assert(kd == wdi);

    // t1 = result ⋅ bra*  over (r1 == bL)
    TN<Scalar, 5> t1(tenx::array5{r0, r2, r3, bd, bR});
    t1.device(*threads->dev) = result.contract(bra.conjugate(), tenx::idx({1}, {1}));

    // t2 = t1 ⋅ mpo over (r2 == wL) and (bra phys == dout)
    // Remaining mpo phys is wdi, remaining mpo bond is wR
    TN<Scalar, 5> t2(tenx::array5{r0, r3, bR, wR, wdi});
    t2.device(*threads->dev) = t1.contract(mpo, tenx::idx({1, 3}, {0, 3}));

    // tmp = t2 ⋅ ket over (r3 == kL) and (wdi == ket phys)
    tmp.resize(tenx::array4{r0, bR, wR, kR});
    tmp.device(*threads->dev) = t2.contract(ket, tenx::idx({1, 4}, {1, 0}));
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