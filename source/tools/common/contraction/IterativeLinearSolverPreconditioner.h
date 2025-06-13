#pragma once
#include "IterativeLinearSolverConfig.h"
#include <chrono>
#include <Eigen/Core>

template<typename MatrixLikeType>
class IterativeLinearSolverPreconditioner {
    private:
    using Scalar     = typename MatrixLikeType::Scalar;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    protected:
    mutable Eigen::Index m_iterations    = 0;
    mutable double       m_time          = 0.0;
    bool                 m_isInitialized = false;

    const MatrixLikeType                      *matrix = nullptr;
    const IterativeLinearSolverConfig<Scalar> *config = nullptr;

    // Vectors used in the chebyshev preconditioner
    mutable VectorType y_old;
    mutable VectorType y_new;
    mutable VectorType y_next;
    mutable VectorType z_res;

    public:
    using StorageIndex = typename VectorType::StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    IterativeLinearSolverPreconditioner() = default;

    void attach(MatrixLikeType *matrix_, IterativeLinearSolverConfig<Scalar> *config_) {
        matrix = matrix_;
        config = config_;
        if(matrix != nullptr and config != nullptr) { m_isInitialized = true; }
    }

    template<typename MatType>
    explicit IterativeLinearSolverPreconditioner(const MatType &) {}
    Eigen::Index    iterations() { return m_iterations; }
    double          elapsed_time() { return m_time; }
    EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return matrix->rows(); }
    EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return matrix->cols(); }

    template<typename MatType>
    IterativeLinearSolverPreconditioner &analyzePattern(const MatType &) {
        return *this;
    }

    template<typename MatType>
    IterativeLinearSolverPreconditioner &factorize(const MatType &) {
        return *this;
    }

    template<typename MatType>
    IterativeLinearSolverPreconditioner &compute(const MatType &) {
        return *this;
    }

    template<typename Rhs, typename Dest>
    void solve_chebyshev(const Rhs &b, Dest &x) const {
        // b: input vector (e.g., the residual)
        // x: output vector, result of applying the preconditioner (approximates A^{-1} b)
        const auto degree     = config->chebyshev.degree;
        const auto lambda_min = config->chebyshev.lambda_min;
        const auto lambda_max = config->chebyshev.lambda_max;

        if(degree <= 0 || std::isnan(lambda_min) || std::isnan(lambda_max) || lambda_max <= lambda_min) {
            x = b;
            return;
        }

        // Chebyshev parameters
        RealScalar c     = (lambda_max - lambda_min) / RealScalar{2};
        RealScalar d     = (lambda_max + lambda_min) / RealScalar{2};
        RealScalar alpha = RealScalar{1} / d; // First recurrence coefficient

        // Vectors for the recurrence
        y_old.setZero(b.size());
        y_new = alpha * b;
        y_next.resize(b.size());
        z_res.resize(b.size());

        if(degree == 1) {
            x = y_new;
            m_iterations++;
            return;
        }

        RealScalar delta = c / d; // Used in recurrence
        RealScalar beta  = RealScalar{0};

        // Now build the recurrence for higher degree
        for(int k = 2; k <= degree; ++k) {
            beta             = (c / (RealScalar{2} * d)) * beta; // Compute recurrence coefficients (see Saad Alg 12.1 or PETSc)
            z_res.noalias()  = b - matrix->operator*(y_new);     // Residual: z = b - A * y_new
            y_next.noalias() = alpha * z_res + RealScalar{2} * delta * y_new - beta * y_old;
            // Rotate vectors for next step
            y_old = std::move(y_new);
            y_new = std::move(y_next);
        }
        x = y_new;
        m_iterations++;
    }
    template<typename Rhs, typename Dest, typename BlocksPtrType>
    void apply_jacobi_blocks(const Rhs &b, Dest &x, const BlocksPtrType *blocks) const {
        if(blocks == nullptr) return;
        if(blocks->empty()) return;
#pragma omp parallel for
        for(size_t idx = 0; idx < blocks->size(); ++idx) {
            const auto &[offset, solver] = blocks->at(idx);
            long extent                  = solver->rows();
            auto x_segment               = Eigen::Map<VectorType>(x.data() + offset, extent);
            auto b_segment               = Eigen::Map<const VectorType>(b.data() + offset, extent);
            if constexpr(std::is_same_v<decltype(solver), Eigen::LLT<MatrixType, Eigen::Lower>>) {
                // Symmetric solve (split L and L^T)
                VectorType temp     = solver->matrixL().solve(b_segment);
                x_segment.noalias() = solver->matrixU().solve(temp);
            } else {
                x_segment.noalias() = solver->solve(b_segment);
            }
        }
        m_iterations++;
    }

    template<typename Rhs, typename Dest>
    void solve_jacobi(const Rhs &b, Dest &x) const {
        auto old_iterations = m_iterations;
        if(config->jacobi.invdiag != nullptr) {
            auto invdiag = Eigen::Map<const VectorType>(config->jacobi.invdiag, rows());
            x.noalias()  = invdiag.array().cwiseProduct(b.array()).matrix();
            m_iterations++;
        }
        apply_jacobi_blocks(b, x, config->jacobi.lltJcbBlocks);
        apply_jacobi_blocks(b, x, config->jacobi.ldltJcbBlocks);
        apply_jacobi_blocks(b, x, config->jacobi.luJcbBlocks);
        if(m_iterations == old_iterations) {
            // No blocks given (all are nullptr or size 0)
            // Then this should act like an identity preconditioner
            x = b;
        }
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs &b, Dest &x) const {
        auto t_start = std::chrono::high_resolution_clock::now();
        if(config->precondType == PreconditionerType::CHEBYSHEV) { solve_chebyshev(b, x); }
        if(config->precondType == PreconditionerType::JACOBI) { solve_jacobi(b, x); }
        auto t_end = std::chrono::high_resolution_clock::now();
        m_time += std::chrono::duration<double>(t_end - t_start).count();
    }

    template<typename Rhs>
    inline const Eigen::Solve<IterativeLinearSolverPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "IterativeLinearSolverPreconditioner is not initialized.");
        eigen_assert(b.rows() == rows() && "Size mismatchs");
        return Eigen::Solve<IterativeLinearSolverPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};
