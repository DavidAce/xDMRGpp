#pragma once

#include "../config/enums.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <vector>

namespace qm::mpo {
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> using CplxScalar = std::complex<RealScalar<Scalar>>;
    template<typename Scalar>
    extern std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> pauli_mpo(const Eigen::Matrix2cd &paulimatrix);

    template<typename Scalar>
    extern std::tuple<Eigen::Tensor<Scalar, 4>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>> prod_pauli_mpo(std::string_view axis);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        parity_projector_mpos(const Eigen::Matrix2cd &paulimatrix, size_t sites, int sector = 1);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        random_pauli_mpos(const Eigen::Matrix2cd &paulimatrix, size_t sites);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        random_pauli_mpos_x2(const Eigen::Matrix2cd &paulimatrix1, const Eigen::Matrix2cd &paulimatrix2, size_t sites);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        random_pauli_mpos(const std::vector<Eigen::Matrix2cd> &paulimatrices, size_t sites);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        sum_of_pauli_mpo(const std::vector<Eigen::Matrix2cd> &paulimatrices, size_t sites, RandomizerMode mode);

    template<typename Scalar>
    extern std::tuple<std::vector<Eigen::Tensor<Scalar, 4>>, Eigen::Tensor<Scalar, 3>, Eigen::Tensor<Scalar, 3>>
        random_pauli_mpos(const std::vector<Eigen::Matrix2cd> &paulimatrices, const std::vector<double> &uniform_dist_widths, size_t sites);
}