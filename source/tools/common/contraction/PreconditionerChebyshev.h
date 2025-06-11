#pragma once
#include "tools/common/contraction.h"
#include <Eigen/Core>
#include <tools/finite/mpo.h>

template<typename Scalar_>
class PreconditionerChebyshev {
    private:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    protected:
    Eigen::Index         m_rows          = 0;
    Eigen::Index         m_cols          = 0;
    mutable Eigen::Index m_iterations    = 0;
    bool                 m_isInitialized = false;

    const Scalar_              *envL = nullptr;
    const Scalar_              *envR = nullptr;
    const Scalar_              *mpo  = nullptr;
    std::array<Eigen::Index, 3> shape_mps;
    std::array<Eigen::Index, 4> shape_mpo;
    std::array<Eigen::Index, 3> shape_envL;
    std::array<Eigen::Index, 3> shape_envR;
    Eigen::Index                mps_size;

    RealScalar   lambda_min;
    RealScalar   lambda_max;
    Eigen::Index degree = 1;

    public:
    using StorageIndex = typename VectorType::StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    PreconditionerChebyshev() = default;

    void attach(const Scalar_ *const envL_,      /*!< The left block tensor.  */
                const Scalar_ *const envR_,      /*!< The right block tensor.  */
                const Scalar_ *const mpo_,       /*!< The Hamiltonian MPO's  */
                std::array<long, 3>  shape_mps_, /*!< An array containing the shapes of the mps  */
                std::array<long, 4>  shape_mpo_, /*!< An array containing the shapes of the mpo  */
                RealScalar lambda_min_, RealScalar lambda_max_, Eigen::Index degree_) {
        envL       = envL_;
        envR       = envR_;
        mpo        = mpo_;
        shape_mps  = shape_mps_;
        shape_mpo  = shape_mpo_;
        shape_envL = {shape_mps_[1], shape_mps_[1], shape_mpo_[0]};
        shape_envR = {shape_mps_[2], shape_mps_[2], shape_mpo_[1]};
        if(envL == nullptr) throw std::runtime_error("Lblock is a nullptr!");
        if(envR == nullptr) throw std::runtime_error("Rblock is a nullptr!");
        if(mpo == nullptr) throw std::runtime_error("mpo is a nullptr!");
        mps_size   = shape_mps[0] * shape_mps[1] * shape_mps[2];
        lambda_min = lambda_min_;
        lambda_max = lambda_max_;
        degree     = degree_;
    }

    template<typename MatType>
    explicit PreconditionerChebyshev(const MatType &) {}
    Eigen::Index    iterations() { return m_iterations; }
    EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return m_rows; }
    EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return m_cols; }

    template<typename MatType>
    PreconditionerChebyshev &analyzePattern(const MatType &) {
        return *this;
    }

    template<typename MatType>
    PreconditionerChebyshev &factorize(const MatType &) {
        return *this;
    }

    void set_size(Eigen::Index linearSize) {
        m_rows          = linearSize;
        m_cols          = linearSize;
        m_isInitialized = true;
    }

    template<typename MatType>
    PreconditionerChebyshev &compute(const MatType &) {
        return *this;
    }

    void matvec(const VectorType &in, VectorType &out) const {
        tools::common::contraction::matrix_vector_product(out.data(), in.data(), shape_mps, mpo, shape_mpo, envL, shape_envL, envR, shape_envR);
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs &b, Dest &x) const {
        using Scalar   = typename VectorType::Scalar;
        const Scalar d = (lambda_max + lambda_min) / RealScalar{2};
        const Scalar c = (lambda_max - lambda_min) / RealScalar{2};

        VectorType r0 = b;
        VectorType r1(b.size());
        VectorType temp(b.size());

        // Initialize the preconditioned result
        x.setZero();

        // Chebyshev coefficients
        Scalar alpha0 = RealScalar{1} / d;
        Scalar beta   = RealScalar{0}; // Initial beta

        // First step
        matvec(r0, temp); // temp = A * r0
        r1 = (temp - d * r0) / c;
        x  = alpha0 * r0;

        if(degree == 1) {
            x += (RealScalar{1} / c) * r1;
            return;
        }

        Scalar alpha1 = RealScalar{2} / c;

        // Recurrence for steps 2 to m
        for(int k = 2; k <= degree; ++k) {
            matvec(r1, temp); // temp = A * r1
            VectorType r2 = (RealScalar{2} / c) * (temp - d * r1) - r0;
            x += alpha1 * r1;

            // Prepare for the next iteration
            r0 = std::move(r1);
            r1 = std::move(r2);
        }
        m_iterations++;
    }

    template<typename Rhs>
    inline const Eigen::Solve<PreconditionerChebyshev, Rhs> solve(const Eigen::MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "PreconditionerChebyshev is not initialized.");
        eigen_assert(b.rows() == rows() && "Size mismatchs");
        return Eigen::Solve<PreconditionerChebyshev, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};
