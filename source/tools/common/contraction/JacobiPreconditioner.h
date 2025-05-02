#pragma once
#include <Eigen/Core>

template<typename Scalar_>
class JacobiPreconditioner {
    private:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    protected:
    VectorType m_invdiag;
    bool       m_isInitialized = false;

    public:
    using StorageIndex = typename VectorType::StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    JacobiPreconditioner() = default;

    template<typename MatType>
    explicit JacobiPreconditioner(const MatType &) {}

    EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return m_invdiag.size(); }
    EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return m_invdiag.size(); }

    template<typename MatType>
    JacobiPreconditioner &analyzePattern(const MatType &) {
        return *this;
    }

    template<typename MatType>
    JacobiPreconditioner &factorize(const MatType &) {
        return *this;
    }

    template<typename VecType>
    void set_invdiag(const VecType &vec) {
        m_invdiag       = vec;
        m_isInitialized = true;
    }

    template<typename MatType>
    JacobiPreconditioner &compute(const MatType &) {
        return *this;
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs &b, Dest &x) const {
        x = m_invdiag.array().cwiseProduct(b.array());
    }

    template<typename Rhs>
    inline const Eigen::Solve<JacobiPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "JacobiPreconditioner is not initialized.");
        eigen_assert(m_invdiag.size() == b.rows() && "JacobiPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
        return Eigen::Solve<JacobiPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};
