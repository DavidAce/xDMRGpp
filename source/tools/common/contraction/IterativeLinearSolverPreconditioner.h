#pragma once
#include "IterativeLinearSolverConfig.h"
#include <chrono>
#include <Eigen/Core>
namespace settings {
    static constexpr bool debug_jcb = false;
}

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
    mutable VectorType temp;
    mutable MatrixType C;

    public:
    using StorageIndex = typename VectorType::StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    IterativeLinearSolverPreconditioner() = default;

    void attach(MatrixLikeType *matrix_, const IterativeLinearSolverConfig<Scalar> *config_) {
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

        if(degree <= 0 || std::isnan(lambda_min) || std::isnan(lambda_max) || lambda_max <= lambda_min || lambda_min <= 0) {
            x = b;
            return;
        }

        RealScalar   lmin_eff = std::max<RealScalar>(lambda_min, lambda_max * RealScalar{1e-3f});
        RealScalar   rho      = (lambda_max - lmin_eff) / (lambda_max + lmin_eff);
        RealScalar   gamma    = RealScalar{0.1f}; // we want a ~10× reduction per precond application
        Eigen::Index m_auto   = static_cast<Eigen::Index>(std::ceil(std::log(RealScalar{2} / gamma) / (RealScalar{2} * std::acosh(RealScalar{1} / rho))));
        Eigen::Index m        = std::clamp<Eigen::Index>(m_auto, 1, 20);
        if(m != degree and m_iterations == 0) { std::printf("m %ld -> %ld\n", degree, m); }

        // Chebyshev parameters
        RealScalar c     = (lambda_max - lmin_eff) / RealScalar{2};
        RealScalar d     = (lambda_max + lmin_eff) / RealScalar{2};
        RealScalar sigma = c / (RealScalar{2} * d); // σ
        RealScalar beta  = sigma * sigma;           // β = σ²
        RealScalar inv_d = RealScalar{1} / d;       // First recurrence coefficient

        // Vectors for the recurrence
        y_old.setZero(b.size());
        y_new = inv_d * b;
        y_next.resize(b.size());
        z_res.resize(b.size());

        if(m == 1) {
            x = y_new;
            m_iterations++;
            return;
        }

        // Now build the recurrence for higher degree
        for(int k = 2; k <= m; ++k) {
            z_res.noalias()  = b - matrix->MatrixOp(y_new); // Residual: z = b - A * y_new
            y_next.noalias() = inv_d * z_res + RealScalar{2} * sigma * y_new - beta * y_old;
            assert(z_res.allFinite());
            assert(y_next.allFinite());
            // Rotate vectors for next step
            y_old.swap(y_new);
            y_new.swap(y_next);
        }
        x = y_new;
        m_iterations++;
    }

    template<typename Rhs, typename Dest, typename SolverType>
    void apply_jacobi_blocks(const Rhs &b, Dest &x, const std::vector<std::tuple<long, int, std::unique_ptr<SolverType>>> *blocks) const {
        if(blocks == nullptr) return;
        if(blocks->empty()) return;
#pragma omp parallel for
        for(size_t idx = 0; idx < blocks->size(); ++idx) {
            const auto &[offset, sign, solver] = blocks->at(idx);
            long extent                         = solver->rows();
            auto x_segment                      = Eigen::Map<VectorType>(x.data() + offset, extent);
            auto b_segment                      = Eigen::Map<const VectorType>(b.data() + offset, extent);
            x_segment.noalias() = solver->solve(b_segment*static_cast<RealScalar>(sign));
        }
        m_iterations++;
    }

    template<typename Rhs, typename Dest>
    void solve_jacobi(const Rhs &b, Dest &x) const {
        if(config->jacobi.skipjcb) {
            x = b;
            return;
        }
        VectorType y = b;
        if constexpr(matrix->has_projector_op) {
            // Project out an operator if present here
            y = matrix->ProjectOpR(b);
        }

        auto old_iterations = m_iterations;
        if(config->jacobi.invdiag != nullptr) {
            // None of the block jacobi preconditioners were applied.
            auto invdiag = Eigen::Map<const VectorType>(config->jacobi.invdiag, rows());
            assert(invdiag.allFinite());
            x.noalias()  = invdiag.array().cwiseProduct(y.array()).matrix();
            m_iterations++;
        }
        apply_jacobi_blocks(y, x, config->jacobi.lltJcbBlocks);
        apply_jacobi_blocks(y, x, config->jacobi.ldltJcbBlocks);
        apply_jacobi_blocks(y, x, config->jacobi.luJcbBlocks);
        apply_jacobi_blocks(y, x, config->jacobi.qrJcbBlocks);

        if(m_iterations == old_iterations) {
            // No blocks given (all are nullptr or size 0)
            // Then this should act like an identity preconditioner
            throw std::runtime_error("no blocks applied");
            // x = b;
        }

        if constexpr(matrix->has_projector_op) {
            // Project out an operator if present here
            x = matrix->ProjectOpL(x);
        }
    }
    template<typename Rhs, typename Dest>
    void solve_deflated_jacobi(const Rhs &b, Dest &x) const {
        bool has_defl_eigvecs = config->jacobi.deflationEigVecs.rows() == b.rows();
        bool has_defl_eigvals = config->jacobi.deflationEigInvs.size() > 0;
        if(!has_defl_eigvecs or !has_defl_eigvals) {
            solve_jacobi(b, x);
            return;
        }
        if(m_iterations == 0) std::printf("deflating");
        const MatrixType &Z = config->jacobi.deflationEigVecs;
        const VectorType &Y = config->jacobi.deflationEigInvs;
        // Step 1: Remove projections on the smallest eigenvectors explicitly
        VectorType alpha        = Z.adjoint() * matrix->gemm(b); // α = Zᵀ B r
        VectorType lambda_alpha = alpha.cwiseQuotient(Y);        // Λ α  because Y = λ⁻¹
        VectorType b_deflated   = b - Z * lambda_alpha;          // b_deflated = (I - B Z Zᵀ) r

        // Step 2: Solve the block-Jacobi system on the deflated residual
        solve_jacobi(b_deflated, x); // y = M_BJ⁻¹ b_deflated

        // Step 3: Reincorporate explicitly solved smallest eigenvectors
        VectorType alpha_scaled = Y.cwiseProduct(alpha); // Λ⁻¹ α
        x.noalias() += Z * alpha_scaled;                 // y = y + Z Λ⁻¹ Zᵀ r
    }

    template<typename Rhs, typename Dest>
    void solve_coarse_jacobi(const Rhs &b, Dest &x) const {
        // x = b;
        // return;
        // Step 1: Apply the jacobi preconditioner
        solve_jacobi(b, x); // x = M_BJ⁻¹ * b
        const MatrixType &Z  = config->jacobi.coarseZ;
        const MatrixType &HZ = config->jacobi.coarseHZ;
        if(Z.cols() == 0 || HZ.cols() == 0) return; // no coarse space

        // Step 2: Apply the coarse solve
        if(m_iterations == 1) std::printf("coarse solve\n");
        if(C.size() == 0) { C = (Z.adjoint() * HZ).ldlt().solve(MatrixType::Identity(Z.cols(), Z.cols())); }
        VectorType alpha = C * (HZ.adjoint() * b); //  p×1
        x.noalias() += Z * alpha;
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs &b, Dest &x) const {
        auto t_start = std::chrono::high_resolution_clock::now();
        if(config->precondType == PreconditionerType::CHEBYSHEV) { solve_chebyshev(b, x); }
        if(config->precondType == PreconditionerType::JACOBI) { solve_coarse_jacobi(b, x); }
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
