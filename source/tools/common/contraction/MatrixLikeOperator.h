#include <chrono>
#include <Eigen/Core>
template<typename Scalar_>
class MatrixLikeOperator;
template<typename T>
using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace Eigen::internal {
    // MatrixLikeOperator looks-like a Dense Matrix, so let's inherits its traits:
    template<typename Scalar_>
    struct traits<MatrixLikeOperator<Scalar_>> : public Eigen::internal::traits<DenseMatrix<Scalar_>> {};

}

template<typename Scalar_>
class MatrixLikeOperator : public Eigen::EigenBase<MatrixLikeOperator<Scalar_>> {
    public:
    // Required typedefs, constants, and method:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    typedef int StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    mutable Eigen::Index m_opcounter = 0;
    mutable double       m_optimer   = 0.0;
    Eigen::Index         size;

    void _check_template_params() {};
    // Custom API:
    MatrixLikeOperator() = default;
    MatrixLikeOperator(Eigen::Index size_, std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp_) : size(size_), MatrixOp(MatrixOp_) {}
    std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp;

    template<typename Rhs>
    Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto result  = Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
        auto t_end   = std::chrono::high_resolution_clock::now();
        m_optimer += std::chrono::duration<double>(t_end - t_start).count();
        m_opcounter++;
        return result;
    }

    template<typename Rhs>
    Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct> gemm(const Eigen::MatrixBase<Rhs> &x) const {
        return operator*(x);
    }

    [[nodiscard]] Eigen::Index rows() const { return safe_cast<Eigen::Index>(size); };
    [[nodiscard]] Eigen::Index cols() const { return safe_cast<Eigen::Index>(size); };
    Eigen::Index               iterations() const { return m_opcounter; }
    double                     elapsed_time() const { return m_optimer; }
};

namespace Eigen::internal {

    template<typename Rhs, typename ReplScalar>
    struct generic_product_impl<MatrixLikeOperator<ReplScalar>, Rhs, DenseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<MatrixLikeOperator<ReplScalar>, Rhs, generic_product_impl<MatrixLikeOperator<ReplScalar>, Rhs>> {
        typedef typename Product<MatrixLikeOperator<ReplScalar>, Rhs>::Scalar Scalar;

        template<typename Dest>
        static void scaleAndAddTo(Dest &dst, const MatrixLikeOperator<ReplScalar> &mat, const Rhs &rhs, const Scalar &alpha) {
            // This method should implement "dst += alpha * lhs * rhs" inplace; however, for iterative solvers, alpha is always equal to 1, so let's not worry
            // about it.
            assert(alpha == Scalar(1) && "scaling is not implemented");
            assert(rhs.size() == mat.rows());
            assert(dst.size() == mat.rows());
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            dst.noalias() += mat.MatrixOp(rhs);
        }
    };

}