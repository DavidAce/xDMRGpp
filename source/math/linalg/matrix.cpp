
#include "matrix.h"
#include <unsupported/Eigen/KroneckerProduct>

template<typename DerivedA, typename DerivedB>
linalg::matrix::KroneckerResultType<DerivedA, DerivedB> linalg::matrix::kronecker(const Eigen::PlainObjectBase<DerivedA> &A,
                                                                                  const Eigen::PlainObjectBase<DerivedB> &B, bool mirror) {
    if(mirror)
        return kroneckerProduct(B.derived(), A.derived());
    else
        return kroneckerProduct(A.derived(), B.derived());
}

template linalg::matrix::KroneckerResultType<Eigen::MatrixXcd, Eigen::MatrixXcd>
    linalg::matrix::kronecker(const Eigen::PlainObjectBase<Eigen::MatrixXcd> &A, const Eigen::PlainObjectBase<Eigen::MatrixXcd> &B, bool mirror);

template linalg::matrix::KroneckerResultType<Eigen::MatrixXd, Eigen::MatrixXd>
    linalg::matrix::kronecker(const Eigen::PlainObjectBase<Eigen::MatrixXd> &A, const Eigen::PlainObjectBase<Eigen::MatrixXd> &B, bool mirror);

template linalg::matrix::KroneckerResultType<Eigen::Matrix2cd, Eigen::Matrix2cd>
    linalg::matrix::kronecker(const Eigen::PlainObjectBase<Eigen::Matrix2cd> &A, const Eigen::PlainObjectBase<Eigen::Matrix2cd> &B, bool mirror);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Vconst) {
    // Orthonormalize with Modified Gram Schmidt
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Real       = decltype(std::real(std::declval<T>()));
    MatrixType V     = Vconst;
    MatrixType Q     = MatrixType::Zero(V.rows(), V.cols());
    MatrixType R     = MatrixType::Zero(V.cols(), V.cols());
    for(long i = 0; i < V.cols(); ++i) {
        Q.col(i) = V.col(i);
        R(i, i)  = Q.col(i).norm();
        if(std::abs(R(i, i)) < std::numeric_limits<Real>::epsilon()) {
            // tools::log->error("Q.col({}) is a zero vector:\n Q: \n{}\n", i, linalg::matrix::to_string(Q.real(), 8));
            continue;
        }
        Q.col(i) /= R(i, i);
        for(long j = i + 1; j < V.cols(); ++j) {
            R(i, j) = Q.col(i).dot(V.col(j));
            V.col(j) -= Q.col(i) * R(i, j);
        }
    }
    return Q;
}
/* clang-format off */
template Eigen::Matrix<fp32, Eigen::Dynamic, Eigen::Dynamic>  linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<fp32, Eigen::Dynamic, Eigen::Dynamic> &V);
template Eigen::Matrix<fp64, Eigen::Dynamic, Eigen::Dynamic>  linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<fp64, Eigen::Dynamic, Eigen::Dynamic> &V);
template Eigen::Matrix<fp128, Eigen::Dynamic, Eigen::Dynamic> linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<fp128, Eigen::Dynamic, Eigen::Dynamic> &V);
template Eigen::Matrix<cx32, Eigen::Dynamic, Eigen::Dynamic>  linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<cx32, Eigen::Dynamic, Eigen::Dynamic> &V);
template Eigen::Matrix<cx64, Eigen::Dynamic, Eigen::Dynamic>  linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<cx64, Eigen::Dynamic, Eigen::Dynamic> &V);
template Eigen::Matrix<cx128, Eigen::Dynamic, Eigen::Dynamic> linalg::matrix::modified_gram_schmidt(const Eigen::Matrix<cx128, Eigen::Dynamic, Eigen::Dynamic> &V);
