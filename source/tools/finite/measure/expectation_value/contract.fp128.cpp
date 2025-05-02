#include "contract.impl.h"

using Scalar = fp128;

/* clang-format off */
template void contract_op_M_1_0(TN<Scalar, 3> &temp, const TN<Scalar, 2> &op, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template TN<Scalar, 2> contract_chain_M_Mconj_0_1_01_10(const TN<Scalar, 2> &chain, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template void contract_M_Ledge3_mpo_Mconj_0_1_0_1_013_023(TN<Scalar, 3> &temp, const TN<Scalar, 3> &M, const TN<Scalar, 3> &Ledge3, const TN<Scalar, 4> &mpo, const ThreadPtr &threads);

template TN<Scalar, 0> contract_Ledge3_Redge3_012_012(const TN<Scalar, 3> &Ledge3, const TN<Scalar, 3> &Redge3, const ThreadPtr &threads);

template void contract_mps1_mpo_mps2_0_2_4_0(TN<Scalar, 4> &result, const TN<Scalar, 3> &mps1, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &mps2, const ThreadPtr &threads);

template void contract_res_mps1conj_mpo_mps2_1_1_13_02_14_10(TN<Scalar, 4> &tmp, const TN<Scalar, 4> &result, const TN<Scalar, 3> &mps1, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &mps2, const ThreadPtr &threads);

template void contract_resL_ket_mpo_braconj_0_1_02_02_03_10(TN<Scalar, 3> &tmp, const TN<Scalar, 3> &resL, const TN<Scalar, 3> &ket, const TN<Scalar, 4> &mpo, const TN<Scalar, 3> &bra, const ThreadPtr &threads);

template TN<Scalar, 0> contract_resL_envR_012_021(const TN<Scalar, 3> &resL, const TN<Scalar, 3> &envR, const ThreadPtr &threads) ;
/* clang-format on */
