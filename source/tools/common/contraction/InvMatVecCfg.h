#pragma once

#include <complex>
// #include <Eigen/Core>
// #include <utility>
#include <Eigen/Cholesky>
#include <Eigen/LU>

enum class MatDef { IND, SEMI, DEF };
template<typename Scalar>
struct InvMatVecCfg {
    using Real              = decltype(std::real(std::declval<Scalar>()));
    using MatrixType        = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using LLTType           = Eigen::LLT<MatrixType, Eigen::Lower>;
    using LDLTType          = Eigen::LDLT<MatrixType, Eigen::Lower>;
    using LUType            = Eigen::PartialPivLU<MatrixType>;
    using LLTJcbBlocksType  = std::vector<std::tuple<long, std::unique_ptr<LLTType>>>;
    using LDLTJcbBlocksType = std::vector<std::tuple<long, std::unique_ptr<LDLTType>>>;
    using LUJcbBlocksType   = std::vector<std::tuple<long, std::unique_ptr<LUType>>>;

    // using VectorType        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    long                     maxiters      = 1000;
    Real                     tolerance     = Real{1e-5f};
    const Scalar            *invdiag       = nullptr;
    const LLTJcbBlocksType  *lltJcbBlocks  = nullptr;
    const LDLTJcbBlocksType *ldltJcbBlocks = nullptr;
    const LUJcbBlocksType   *luJcbBlocks   = nullptr;
    MatDef                   matdef        = MatDef::DEF; /*! Whether the matrix is indefinite or (semi) definite*/
};