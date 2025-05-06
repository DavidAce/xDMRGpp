#include "impl.h"

using Scalar = cx64;
using T = cx64;
/* clang-format off */

template std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);

template std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);

template std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
