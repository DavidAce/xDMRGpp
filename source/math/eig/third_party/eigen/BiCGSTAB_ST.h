// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// Modifications:
//   Copyright (C) 2025 David Aceituno Chavez <aceituno@kth.se>
//   - Renamed the class BiCGSTAB -> BiCGSTAB_ST to avoid collision .
//   - Added Stall Termination (ST) logic in the BiCGSTAB loop.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BICGSTAB_ST_H
#define EIGEN_BICGSTAB_ST_H
#include <deque>
#include <Eigen/IterativeLinearSolvers>

namespace Eigen {

    namespace internal {

        /** \internal Low-level bi conjugate gradient stabilized algorithm
         * \param mat The matrix A
         * \param rhs The right hand side vector b
         * \param x On input and initial solution, on output the computed solution.
         * \param precond A preconditioner being able to efficiently solve for an
         *                approximation of Ax=b (regardless of b)
         * \param iters On input the max number of iteration, on output the number of performed iterations.
         * \param tol_error On input the tolerance error, on output an estimation of the relative error.
         * \return false in the case of numerical issue, for example a break down of BiCGSTAB_ST.
         */
        template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
        bool bicgstab_st(const MatrixType &mat, const Rhs &rhs, Dest &x, const Preconditioner &precond, Index &iters, typename Dest::RealScalar &tol_error) {
            typedef typename Dest::RealScalar  RealScalar;
            typedef typename Dest::Scalar      Scalar;
            typedef Matrix<Scalar, Dynamic, 1> VectorType;
            RealScalar                         tol      = tol_error;
            Index                              maxIters = iters;

            Index      n  = mat.cols();
            VectorType r  = rhs - mat * x;
            VectorType r0 = r;

            RealScalar r0_norm  = r0.stableNorm();
            RealScalar r_norm   = r0_norm;
            RealScalar rhs_norm = rhs.stableNorm();
            if(rhs_norm == 0) {
                x.setZero();
                return true;
            }
            Scalar rho(1);
            Scalar alpha(0);
            Scalar w(1);

            VectorType v = VectorType::Zero(n), p = VectorType::Zero(n);
            VectorType y(n), z(n);
            VectorType kt(n), ks(n);

            VectorType s(n), t(n);

            RealScalar eps      = NumTraits<Scalar>::epsilon();
            Index      i        = 0;
            Index      restarts = 0;

            // BEGIN BiCGSTAB -> BiCGSTAB_ST
            auto get_stats = [&](const std::deque<RealScalar> &a, bool sample = true) -> std::tuple<RealScalar, RealScalar, RealScalar, RealScalar> {
                const std::size_t     n   = a.size();
                static constexpr auto eps = std::numeric_limits<RealScalar>::epsilon();
                static constexpr auto nan = std::numeric_limits<RealScalar>::quiet_NaN();
                if(n == 0) return {1, 1, 1, nan};
                if(sample && n < 2) return {1, 1, 1, nan};

                // Pass 1: average
                RealScalar sum = 0;
                for(const auto &elem : a) sum += elem;
                const RealScalar avg = sum / n;

                // Pass 2: sum of squared deviations
                RealScalar ss = 0;
                for(const auto &elem : a) {
                    const RealScalar d  = elem - avg;
                    ss                 += d * d;
                }

                RealScalar var = sample ? (ss / static_cast<RealScalar>(n - 1)) : (ss / static_cast<RealScalar>(n));
                if(var < 0) var = 0; // tiny negative from roundoff, just in case
                auto sdv = std::sqrt(var);

                RealScalar slope = nan;
                if(n >= 2) {
                    RealScalar Sx = 0, Sxx = 0, Sy = 0, Sxy = 0;
                    for(size_t j = 0; j < n; ++j) {
                        RealScalar x  = j;    // iteration index as RealScalar
                        RealScalar y  = a[j]; // eta_k value
                        Sx           += x;
                        Sxx          += x * x;
                        Sy           += y;
                        Sxy          += x * y;
                    }
                    RealScalar den = n * Sxx - Sx * Sx;
                    if(den != 0) { slope = (n * Sxy - Sx * Sy) / den; }
                }

                return {avg, sdv, sdv / std::max(RealScalar{eps}, std::abs(avg)), slope};
            };
            // Monotone energy (exact arithmetic): phi = 0.5 x^H A x - b^H x
            // Use phi = -0.5 * Re( x^H b + x^H r ), with r = b - A x
            RealScalar             phi_i = RealScalar(-0.5) * numext::real(x.dot(rhs) + x.dot(r));
            RealScalar             error = phi_i;
            std::deque<RealScalar> error_history;
            error_history.push_back(error);
            // END BiCGSTAB -> BiCGSTAB_ST

            while(r_norm > tol && i < maxIters) {
                Scalar rho_old = rho;
                rho            = r0.dot(r);
                if(Eigen::numext::abs(rho) / Eigen::numext::maxi(r0_norm, r_norm) < eps * Eigen::numext::mini(r0_norm, r_norm)) {
                    // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
                    // Let's restart with a new r0:
                    r       = rhs - mat * x;
                    r0      = r;
                    rho     = r.squaredNorm();
                    r0_norm = r.stableNorm();
                    alpha   = Scalar(0);
                    w       = Scalar(1);
                    if(restarts++ == 0) i = 0;
                }
                Scalar beta = (rho / rho_old) * (alpha / w);
                p           = r + beta * (p - w * v);

                y = precond.solve(p);

                v.noalias()  = mat * y;
                Scalar theta = r0.dot(v);
                // For small angles ∠(r0, v) < eps, random restart.
                RealScalar v_norm = v.stableNorm();
                if(Eigen::numext::abs(theta) / Eigen::numext::maxi(r0_norm, v_norm) < eps * Eigen::numext::mini(r0_norm, v_norm)) {
                    r       = rhs - mat * x;
                    r0      = Eigen::VectorXf::Random(r0.size()).template cast<Scalar>();
                    r0_norm = r0.stableNorm();
                    rho     = Scalar(1);
                    alpha   = Scalar(0);
                    w       = Scalar(1);
                    if(restarts++ == 0) i = 0;
                    continue;
                }
                alpha = rho / theta;
                s     = r - alpha * v;

                z           = precond.solve(s);
                t.noalias() = mat * z;

                RealScalar tmp = t.squaredNorm();
                if(tmp > RealScalar(0)) {
                    w = t.dot(s) / tmp;
                } else {
                    w = Scalar(0);
                }
                x      += alpha * y + w * z;
                r       = s - w * t;
                r_norm  = r.stableNorm();

                // BEGIN BiCGSTAB -> BiCGSTAB_ST
                RealScalar rrnorm = r_norm / rhs_norm; // Relative residual norm
                error_history.push_back(-std::log10(rrnorm));
                bool rrnorm_has_converged     = rrnorm < tol_error;
                bool rrnorm_has_made_progress = rrnorm < std::max(tol_error, RealScalar{0.1f});
                bool bicgstab_has_converged   = rrnorm_has_converged;

                if(i > 20 and i % 20 == 0) {
                    // Check for stall every 10 iterations
                    while(error_history.size() >= 20) error_history.pop_front();
                    auto [avg, sdv, rel, slope] = get_stats(error_history);
                    bool rrnorm_has_saturated   = rel < RealScalar{1e-2f};
                    bool bicgstab_has_saturated = rrnorm_has_saturated and rrnorm_has_made_progress;
                    // auto error_prev = error_history[error_history.size()-2];
                    // auto relnow = std::abs(error - error_prev)/error;
                    // std::printf("iter %4ld tol_error: %.5e rnorm: %.5e rhsNorm: %.5e error: %.5e (avg: %.5e  sdv: %.5e  rel: %.5e relnow: %.5e slope:
                    // %.5e)\n",
                    //  i, tol_error, std::sqrt(residualNorm2), std::sqrt(rhsNorm2), error_history.back(),  avg, sdv, rel, relnow, slope);

                    if(bicgstab_has_converged) {
                        // std::printf("bicgstab converged\n");
                        break;
                    }
                    if(bicgstab_has_saturated) {
                        // std::printf("bicgstab saturated\n");
                        break;
                    }
                }
                // END BiCGSTAB -> BiCGSTAB_ST

                ++i;
            }

            tol_error = r_norm / rhs_norm;
            iters     = i;
            return true;
        }

    } // namespace internal

    template<typename MatrixType_, typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar> >
    class BiCGSTAB_ST;

    namespace internal {

        template<typename MatrixType_, typename Preconditioner_>
        struct traits<BiCGSTAB_ST<MatrixType_, Preconditioner_> > {
            typedef MatrixType_     MatrixType;
            typedef Preconditioner_ Preconditioner;
        };

    } // namespace internal

    /** \ingroup IterativeLinearSolvers_Module
     * \brief A bi conjugate gradient stabilized solver for sparse square problems
     *
     * This class allows to solve for A.x = b sparse linear problems using a bi conjugate gradient
     * stabilized algorithm. The vectors x and b can be either dense or sparse.
     *
     * \tparam MatrixType_ the type of the sparse matrix A, can be a dense or a sparse matrix.
     * \tparam Preconditioner_ the type of the preconditioner. Default is DiagonalPreconditioner
     *
     * \implsparsesolverconcept
     *
     * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
     * and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations
     * and NumTraits<Scalar>::epsilon() for the tolerance.
     *
     * The tolerance corresponds to the relative residual error: |Ax-b|/|b|
     *
     * \b Performance: when using sparse matrices, best performance is achied for a row-major sparse matrix format.
     * Moreover, in this case multi-threading can be exploited if the user code is compiled with OpenMP enabled.
     * See \ref TopicMultiThreading for details.
     *
     * This class can be used as the direct solver classes. Here is a typical usage example:
     * \include BiCGSTAB_simple.cpp
     *
     * By default the iterations start with x=0 as an initial guess of the solution.
     * One can control the start using the solveWithGuess() method.
     *
     * BiCGSTAB_ST can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
     *
     * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
     */
    template<typename MatrixType_, typename Preconditioner_>
    class BiCGSTAB_ST : public IterativeSolverBase<BiCGSTAB_ST<MatrixType_, Preconditioner_> > {
        typedef IterativeSolverBase<BiCGSTAB_ST> Base;
        using Base::m_error;
        using Base::m_info;
        using Base::m_isInitialized;
        using Base::m_iterations;
        using Base::matrix;

        public:
        typedef MatrixType_                     MatrixType;
        typedef typename MatrixType::Scalar     Scalar;
        typedef typename MatrixType::RealScalar RealScalar;
        typedef Preconditioner_                 Preconditioner;

        public:
        /** Default constructor. */
        BiCGSTAB_ST() : Base() {}

        /** Initialize the solver with matrix \a A for further \c Ax=b solving.
         *
         * This constructor is a shortcut for the default constructor followed
         * by a call to compute().
         *
         * \warning this class stores a reference to the matrix A as well as some
         * precomputed values that depend on it. Therefore, if \a A is changed
         * this class becomes invalid. Call compute() to update it with the new
         * matrix A, or modify a copy of A.
         */
        template<typename MatrixDerived>
        explicit BiCGSTAB_ST(const EigenBase<MatrixDerived> &A) : Base(A.derived()) {}

        ~BiCGSTAB_ST() {}

        /** \internal */
        template<typename Rhs, typename Dest>
        void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const {
            m_iterations = Base::maxIterations();
            m_error      = Base::m_tolerance;

            bool ret = internal::bicgstab_st(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error);

            m_info = (!ret) ? NumericalIssue : m_error <= Base::m_tolerance ? Success : NoConvergence;
        }

        protected:
    };

} // end namespace Eigen

#endif // EIGEN_BICGSTAB_H
