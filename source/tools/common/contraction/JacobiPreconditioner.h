#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

template<typename Scalar_>
class JacobiPreconditioner {
    private:
    using Scalar            = Scalar_;
    using RealScalar        = decltype(std::real(std::declval<Scalar>()));
    using VectorType        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType        = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using LLTType           = Eigen::LLT<MatrixType, Eigen::Lower>;
    using LDLTType          = Eigen::LDLT<MatrixType, Eigen::Lower>;
    using LUType            = Eigen::PartialPivLU<MatrixType>;
    using LLTJcbBlocksType  = std::vector<std::tuple<long, std::unique_ptr<LLTType>>>;
    using LDLTJcbBlocksType = std::vector<std::tuple<long, std::unique_ptr<LDLTType>>>;
    using LUJcbBlocksType   = std::vector<std::tuple<long, std::unique_ptr<LUType>>>;

    protected:
    Eigen::Index             m_rows       = 0;
    Eigen::Index             m_cols       = 0;
    mutable Eigen::Index     m_iterations = 0;
    const Scalar            *m_invdiag    = nullptr;
    const LLTJcbBlocksType  *m_lltJcbBlocks;
    const LDLTJcbBlocksType *m_ldltJcbBlocks;
    const LUJcbBlocksType   *m_luJcbBlocks;
    bool                     m_isInitialized = false;

    public:
    using StorageIndex = typename VectorType::StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    JacobiPreconditioner() = default;

    template<typename MatType>
    explicit JacobiPreconditioner(const MatType &) {}
    Eigen::Index    iterations() { return m_iterations; }
    EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return m_rows; }
    EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return m_cols; }

    template<typename MatType>
    JacobiPreconditioner &analyzePattern(const MatType &) {
        return *this;
    }

    template<typename MatType>
    JacobiPreconditioner &factorize(const MatType &) {
        return *this;
    }

    void set_size(Eigen::Index linearSize) {
        m_rows          = linearSize;
        m_cols          = linearSize;
        m_isInitialized = true;
    }

    void set_invdiag(const Scalar *invdiag) { m_invdiag = invdiag; }
    void set_lltJcbBlocks(const LLTJcbBlocksType *lltJcbBlocks) { m_lltJcbBlocks = lltJcbBlocks; }
    void set_ldltJcbBlocks(const LDLTJcbBlocksType *ldltJcbBlocks) { m_ldltJcbBlocks = ldltJcbBlocks; }
    void set_luJcbBlocks(const LUJcbBlocksType *luJcbBlocks) { m_luJcbBlocks = luJcbBlocks; }

    template<typename MatType>
    JacobiPreconditioner &compute(const MatType &) {
        return *this;
    }



    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs &b, Dest &x) const {
        if(m_invdiag != nullptr) {
            auto invdiag = Eigen::Map<const VectorType>(m_invdiag, m_rows);
            x.noalias()  = invdiag.array().cwiseProduct(b.array()).matrix();
        }

        if(m_lltJcbBlocks != nullptr) {
#pragma omp parallel for
            for(size_t idx = 0; idx < m_lltJcbBlocks->size(); ++idx) {
                const auto &[offset, solver] = m_lltJcbBlocks->at(idx);
                long extent                  = solver->rows();
                auto x_segment               = Eigen::Map<VectorType>(x.data() + offset, extent);
                auto b_segment               = Eigen::Map<const VectorType>(b.data() + offset, extent);
                x_segment.noalias()          = solver->solve(b_segment);
            }
        }
        if(m_ldltJcbBlocks != nullptr) {
#pragma omp parallel for
            for(size_t idx = 0; idx < m_ldltJcbBlocks->size(); ++idx) {
                const auto &[offset, solver] = m_ldltJcbBlocks->at(idx);
                long extent                  = solver->rows();
                auto x_segment               = Eigen::Map<VectorType>(x.data() + offset, extent);
                auto b_segment               = Eigen::Map<const VectorType>(b.data() + offset, extent);
                x_segment.noalias()          = solver->solve(b_segment);
            }
        }
        if(m_luJcbBlocks != nullptr) {
#pragma omp parallel for
            for(size_t idx = 0; idx < m_luJcbBlocks->size(); ++idx) {
                const auto &[offset, solver] = m_luJcbBlocks->at(idx);
                long extent                  = solver->rows();
                auto x_segment               = Eigen::Map<VectorType>(x.data() + offset, extent);
                auto b_segment               = Eigen::Map<const VectorType>(b.data() + offset, extent);
                x_segment.noalias()          = solver->solve(b_segment);
            }
        }
        m_iterations++;
    }

    template<typename Rhs>
    inline const Eigen::Solve<JacobiPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "JacobiPreconditioner is not initialized.");
        eigen_assert(b.rows() == rows() && "Size mismatchs");
        return Eigen::Solve<JacobiPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};
