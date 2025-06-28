#include "mpo.impl.h"

using Scalar = cx128;

/* clang-format off */

template std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>    qm::mpo::pauli_mpo(const Eigen::MatrixXcd &paulimatrix);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>,  Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>  qm::mpo::parity_projector_mpos<Scalar>(const Eigen::MatrixXcd &paulimatrix, size_t sites, int sign);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>,  Eigen::Tensor<Scalar, 3>,  Eigen::Tensor<Scalar, 3>>  qm::mpo::random_pauli_mpos<Scalar>(const Eigen::MatrixXcd &paulimatrix, size_t sites);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> qm::mpo::random_pauli_mpos_x2(const Eigen::MatrixXcd &paulimatrix1, const Eigen::MatrixXcd &pauliMatrixX, const size_t sites);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>    qm::mpo::random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>     qm::mpo::sum_of_pauli_mpo(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites, RandomizerMode mode);

template std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> qm::mpo::random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, const std::vector<double> &uniform_dist_widths, size_t sites);
