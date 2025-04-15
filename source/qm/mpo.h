#pragma once

#include "../config/enums.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <vector>

namespace qm::mpo {
    template<typename Scalar> extern std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> pauli_mpo(const Eigen::MatrixXcd &paulimatrix);
    template<typename Scalar> extern std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> prod_pauli_mpo(std::string_view axis);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        parity_projector_mpos(const Eigen::MatrixXcd &paulimatrix, size_t sites, int sector = 1);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos(const Eigen::MatrixXcd &paulimatrix, size_t sites);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos_x2(const Eigen::MatrixXcd &paulimatrix1, const Eigen::MatrixXcd &paulimatrix2, size_t sites);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        sum_of_pauli_mpo(const std::vector<Eigen::MatrixXcd> &paulimatrices, size_t sites, RandomizerMode mode);

    extern std::tuple<std::vector<Eigen::Tensor<cx64, 4>>, Eigen::Tensor<cx64, 3>, Eigen::Tensor<cx64, 3>>
        random_pauli_mpos(const std::vector<Eigen::MatrixXcd> &paulimatrices, const std::vector<double> &uniform_dist_widths, size_t sites);
}