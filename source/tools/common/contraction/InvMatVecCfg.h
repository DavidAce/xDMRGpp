#pragma once

#include <complex>
// #include <Eigen/Core>
// #include <utility>
enum class MatDef { IND, SEMI, DEF };
template<typename Scalar>
struct InvMatVecCfg {
    using Real = decltype(std::real(std::declval<Scalar>()));
    // using VectorType        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    long    maxiters  = 100;
    Real    tolerance = Real{1e-3f};
    Scalar *invdiag   = nullptr;
    MatDef  matdef    = MatDef::DEF; /*! Whether the matrix is indefinite or (semi) definite*/
};