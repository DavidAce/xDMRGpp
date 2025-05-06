#pragma once
#include "math/tenx.h"

template<typename Scalar, long rank>
using TN = Eigen::Tensor<Scalar, rank>;

using ThreadPtr = std::unique_ptr<tenx::threads::internal::ThreadPoolWrapper>;

template<typename Scalar>
void contract_Mconj_M_1_1(TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_Mconj_M_2_2(TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_rho_M_Mconj_2_1_23_10(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_rho_Mconj_M_2_2_23_20(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_rho_Mconj_M_2_1_2_1(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_rho_Mconj_M_2_2_2_2(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_Mconj_M_0_0(TN<Scalar, 2> &res, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_Mconj_M_0_0(TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_trf_Mconj_M_2_1_23_10(TN<Scalar, 4> &trf_temp4, const TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads);

template<typename Scalar>
void contract_M_trf_2_1(TN<Scalar, 5> &trf_tmp5, const TN<Scalar, 3> &M, const TN<Scalar, 4> &trf_temp, const ThreadPtr &threads);

template<typename Scalar>
void contract_Mconj_trf5_02_02(TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const TN<Scalar, 5> &trf_tmp5, const ThreadPtr &threads);
