#include "contract.h"

template<typename Scalar, long rank>
using TN = Eigen::Tensor<Scalar, rank>;

template<typename Scalar>
void contract_Mconj_M_1_1(TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto dim = M.dimensions();
    rho_temp.resize(std::array{dim[0], dim[0], dim[2], dim[2]});
    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({1}, {1})).shuffle(std::array{0, 2, 1, 3});
}

template<typename Scalar>
void contract_Mconj_M_2_2(TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto dim = M.dimensions();
    rho_temp.resize(std::array{dim[0], dim[0], dim[1], dim[1]});
    rho_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({2}, {2})).shuffle(std::array{0, 2, 1, 3});
}

template<typename Scalar>
void contract_rho_M_Mconj_2_1_23_10(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto rho_dim = rho_temp.dimensions();
    auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[2], mps_dim[2]};
    rho_temp2.resize(new_dim);
    rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
}

template<typename Scalar>
void contract_rho_Mconj_M_2_2_23_20(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto rho_dim = rho_temp.dimensions();
    auto new_dim = std::array{rho_dim[0], rho_dim[1], mps_dim[1], mps_dim[1]};
    rho_temp2.resize(new_dim);
    rho_temp2.device(*threads->dev) = rho_temp.contract(M.conjugate(), tenx::idx({2}, {2})).contract(M, tenx::idx({2, 3}, {2, 0}));
}

template<typename Scalar>
void contract_rho_Mconj_M_2_1_2_1(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto rho_dim = rho_temp.dimensions();
    auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[2], mps_dim[2]};
    rho_temp2.resize(new_dim);
    rho_temp2.device(*threads->dev) =
        rho_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2}, {1})).shuffle(std::array{0, 2, 1, 4, 3, 5}).reshape(new_dim);
}

template<typename Scalar>
void contract_rho_Mconj_M_2_2_2_2(TN<Scalar, 4> &rho_temp2, const TN<Scalar, 4> &rho_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto rho_dim = rho_temp.dimensions();
    auto new_dim = std::array{rho_dim[0] * mps_dim[0], rho_dim[1] * mps_dim[0], mps_dim[1], mps_dim[1]};
    rho_temp2.resize(new_dim);
    rho_temp2.device(*threads->dev) =
        rho_temp.contract(M.conjugate(), tenx::idx({2}, {2})).contract(M, tenx::idx({2}, {2})).shuffle(std::array{2, 0, 4, 1, 3, 5}).reshape(new_dim);
}

template<typename Scalar>
void contract_Mconj_M_0_0(TN<Scalar, 2> &res, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto dim = std::array{M.dimension(1) * M.dimension(2), M.dimension(1) * M.dimension(2)};
    res.resize(dim);
    res.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).reshape(dim);
}

template<typename Scalar>
void contract_Mconj_M_0_0(TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto dim = M.dimensions();
    trf_temp.resize(std::array{dim[1], dim[1], dim[2], dim[2]});
    trf_temp.device(*threads->dev) = M.conjugate().contract(M, tenx::idx({0}, {0})).shuffle(std::array{0, 2, 1, 3});
}

template<typename Scalar>
void contract_trf_Mconj_M_2_1_23_10(TN<Scalar, 4> &trf_tmp4, const TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto trf_dim = trf_temp.dimensions();
    auto new_dim = std::array{trf_dim[0], trf_dim[1], mps_dim[2], mps_dim[2]};
    trf_tmp4.resize(new_dim);
    trf_tmp4.device(*threads->dev) = trf_temp.contract(M.conjugate(), tenx::idx({2}, {1})).contract(M, tenx::idx({2, 3}, {1, 0}));
}

template<typename Scalar>
void contract_M_trf_2_1(TN<Scalar, 5> &trf_tmp5, const TN<Scalar, 3> &M, const TN<Scalar, 4> &trf_temp, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto trf_dim = trf_temp.dimensions();
    auto tm5_dim = std::array{mps_dim[0], mps_dim[1], trf_dim[0], trf_dim[2], trf_dim[3]};
    trf_tmp5.resize(tm5_dim);
    trf_tmp5.device(*threads->dev) = M.contract(trf_temp, tenx::idx({2}, {1}));
}

template<typename Scalar>
void contract_Mconj_trf5_02_02(TN<Scalar, 4> &trf_temp, const TN<Scalar, 3> &M, const TN<Scalar, 5> &trf_tmp5, const ThreadPtr &threads) {
    auto mps_dim = M.dimensions();
    auto trf_dim = trf_temp.dimensions();
    auto new_dim = std::array{mps_dim[1], mps_dim[1], trf_dim[2], trf_dim[3]};
    trf_temp.resize(new_dim);
    trf_temp.device(*threads->dev) = M.conjugate().contract(trf_tmp5, tenx::idx({0, 2}, {0, 2}));
}
