// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Giacomo Po <gpo@ucla.edu>
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2018 David Hyde <dabh@stanford.edu>
//
// Modifications:
//   Copyright (C) 2025 David Aceituno Chavez <aceituno@kth.se>
//   - Renamed the class MINRES -> MINRES_ST to avoid collision with the original.
//   - Added optional stall termination logic based on history statistics.
//   - Added diagnostics (rrnorm/deltax) and configurable correction checks.
//   - Adjusted the stopping metric to use a consistent preconditioned residual norm (when enabled).
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MINRES_ST_H
#define EIGEN_MINRES_ST_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <type_traits>
#include <unsupported/Eigen/IterativeSolvers>

namespace Eigen {

    namespace internal {
        /** \internal Low-level MINRES algorithm
         * \param mat The matrix A
         * \param rhs The right hand side vector b
         * \param x On input and initial solution, on output the computed solution.
         * \param precond A right preconditioner being able to efficiently solve for an
         *                approximation of Ax=b (regardless of b)
         * \param iters On input the max number of iteration, on output the number of performed iterations.
         * \param tol_error On input the tolerance error, on output an estimation of the relative error.
         */
        template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
        EIGEN_DONT_INLINE void minres_st(const MatrixType &mat, const Rhs &rhs, Dest &x, const Preconditioner &precond, Index &iters,
                                         typename Dest::RealScalar &tol_error) {
            using std::sqrt;
            typedef typename Dest::RealScalar  RealScalar;
            typedef typename Dest::Scalar      Scalar;
            typedef Matrix<Scalar, Dynamic, 1> VectorType;

            // Check for zero rhs
            const RealScalar rhsNorm2_euclidean(rhs.squaredNorm());
            if(rhsNorm2_euclidean == 0) {
                x.setZero();
                iters     = 0;
                tol_error = 0;
                return;
            }

            // initialize
            const Index maxIters(iters); // initialize maxIters to iters
            const Index N(mat.cols());   // the size of the matrix

            struct Stats {
                RealScalar  avg       = RealScalar{1};
                RealScalar  sdv       = RealScalar{1};
                RealScalar  rel_sdv   = RealScalar{1}; // sdv / |avg|, keep if you still want it
                RealScalar  slope_avg = RealScalar{0}; // mean (y_i - y_{i-1})/dt
                RealScalar  slope_sdv = RealScalar{0}; // stddev of those slopes
                std::size_t n         = 0;
            };

            auto get_stats = [&](const std::deque<RealScalar> &y, RealScalar dt = RealScalar{1}, bool sample = true) -> Stats {
                Stats s;
                s.n = y.size();

                static constexpr RealScalar eps = std::numeric_limits<RealScalar>::epsilon();

                if(s.n == 0) return s;
                if(sample && s.n < 2) return s;

                // Pass 1: average
                RealScalar sum = 0;
                for(const auto &yi : y) { sum += yi; }
                s.avg = sum / static_cast<RealScalar>(s.n);

                // Pass 2: standard deviation
                RealScalar ss = 0;
                for(const auto &yi : y) {
                    const RealScalar d  = yi - s.avg;
                    ss                 += d * d;
                }
                RealScalar var = sample ? (ss / static_cast<RealScalar>(s.n - 1)) : (ss / static_cast<RealScalar>(s.n));

                if(var < 0) var = 0;
                s.sdv     = std::sqrt(var);
                s.rel_sdv = s.sdv / std::max(eps, std::abs(s.avg));

                // Averaged slope over the window: mean of finite differences.
                // d_i = (y_i - y_{i-1}) / dt, for i=1..n-1
                dt = std::max(dt, RealScalar{1}); // avoid divide-by-zero or silly dt

                const std::size_t m    = s.n - 1;
                RealScalar        dsum = 0;
                for(std::size_t i = 1; i < s.n; ++i) { dsum += (y[i] - y[i - 1]) / dt; }
                s.slope_avg = dsum / static_cast<RealScalar>(m);

                // Stddev of slopes (optional but helpful to see “spiky vs smooth”)
                if(m >= 2) {
                    RealScalar dss = 0;
                    for(std::size_t i = 1; i < s.n; ++i) {
                        RealScalar di  = (y[i] - y[i - 1]) / dt;
                        RealScalar dd  = di - s.slope_avg;
                        dss           += dd * dd;
                    }
                    RealScalar dvar = sample ? (dss / static_cast<RealScalar>(m - 1)) : (dss / static_cast<RealScalar>(m));
                    if(dvar < 0) dvar = 0;
                    s.slope_sdv = std::sqrt(dvar);
                } else {
                    s.slope_sdv = RealScalar{0};
                }

                return s;
            };

            // Initialize preconditioned Lanczos
            VectorType v_old(N);               // will be initialized inside loop
            VectorType v(VectorType::Zero(N)); // initialize v
            VectorType v_new(rhs - mat * x);   // initialize v_new

            // If the initial guess already solves the system, exit
            if(v_new.squaredNorm() == RealScalar{0}) {
                iters     = 0;
                tol_error = 0;
                return;
            }

            VectorType w(N);                         // will be initialized inside loop
            VectorType w_new = precond.solve(v_new); // initialize w_new

            RealScalar beta_new2 = numext::real(v_new.dot(w_new));
            eigen_assert(beta_new2 > RealScalar{0} && "PRECONDITIONER IS NOT POSITIVE DEFINITE");
            if(beta_new2 == RealScalar{0}) {
                iters     = 0;
                tol_error = 0;
                return;
            }

            RealScalar beta_new = std::sqrt(beta_new2);
            eigen_assert(beta_new > RealScalar{0});
            const RealScalar beta_one = beta_new;

            // Interpretation:
            // rhsNorm2 := ||r0||_{M^{-1}}^2 = r0^T M^{-1} r0
            // residualNorm2 := estimated ||rk||_{M^{-1}}^2 tracked by MINRES scalar recursion
            RealScalar rhsNorm2      = beta_new2;
            RealScalar residualNorm2 = beta_new2;

            // Relative residual in the preonditioned norm
            RealScalar       rrnorm0    = std::sqrt(residualNorm2 / rhsNorm2);
            const RealScalar threshold2 = (tol_error * tol_error) * rhsNorm2; // convergence threshold (compared to residualNorm2)

            // Initialize other variables
            RealScalar c     = 1.0; // the cosine of the Givens rotation
            RealScalar c_old = 1.0;
            RealScalar s     = 0.0;                 // the sine of the Givens rotation
            RealScalar s_old = 0.0;                 // the sine of the Givens rotation
            VectorType p_oold(N);                   // will be initialized in loop
            VectorType p_old = VectorType::Zero(N); // initialize p_old=0
            VectorType p     = p_old;               // initialize p=0
            RealScalar eta   = 1.0;

            iters                    = 0; // reset iters
            int minres_has_converged = 0;

            constexpr int rate_rrnorm = 1;  // How often to save into history
            constexpr int rate_deltax = 10; // How often to save into history
            // constexpr int          rate_resid2      = 100; // How often to correct the residual norm
            constexpr int          rate_checks      = 10;   // How often to check convergence/saturation
            constexpr int          rate_printf      = 1000; // How often to print results
            constexpr size_t       max_history_size = 50;
            std::deque<RealScalar> rrnorm_history;
            std::deque<RealScalar> deltax_history;

            while(iters < maxIters) {
                // Preconditioned Lanczos
                /* Note that there are 4 variants on the Lanczos algorithm. These are
                 * described in Paige, C. C. (1972). Computational variants of
                 * the Lanczos method for the eigenproblem. IMA Journal of Applied
                 * Mathematics, 10(3), 373-381. The current implementation corresponds
                 * to the case A(2,7) in the paper. It also corresponds to
                 * algorithm 6.14 in Y. Saad, Iterative Methods for Sparse Linear
                 * Systems, 2003 p.173. For the preconditioned version see
                 * A. Greenbaum, Iterative Methods for Solving Linear Systems, SIAM (1987).
                 */
                if(beta_new == RealScalar{0}) {
                    // happy breakdown: Krylov process terminated
                    break;
                }
                const RealScalar beta   = beta_new;
                v_old                   = v;                      // update: at first time step, this makes v_old = 0 so value of beta doesn't matter
                v_new                  /= beta_new;               // overwrite v_new for next iteration
                w_new                  /= beta_new;               // overwrite w_new for next iteration
                v                       = v_new;                  // update
                w                       = w_new;                  // update
                v_new.noalias()         = mat * w - beta * v_old; // compute v_new
                const RealScalar alpha  = numext::real(v_new.dot(w));
                v_new                  -= alpha * v;                      // overwrite v_new
                w_new                   = precond.solve(v_new);           // overwrite w_new
                beta_new2               = numext::real(v_new.dot(w_new)); // compute beta_new
                eigen_assert(beta_new2 >= RealScalar{0} && "PRECONDITIONER IS NOT POSITIVE DEFINITE");
                beta_new = std::sqrt(beta_new2); // compute beta_new

                // Givens rotation
                const RealScalar r2     = s * alpha + c * c_old * beta; // s, s_old, c and c_old are still from previous iteration
                const RealScalar r3     = s_old * beta;                 // s, s_old, c and c_old are still from previous iteration
                const RealScalar r1_hat = c * alpha - c_old * s * beta;
                const RealScalar r1     = std::hypot(r1_hat, beta_new); // sqrt( std::pow(r1_hat,2) + std::pow(beta_new,2) );
                if(r1 == RealScalar{0}) {
                    std::printf(" -- minres: r1 == 0 ... quitting");
                    break;
                }
                c_old = c;             // store for next iteration
                s_old = s;             // store for next iteration
                c     = r1_hat / r1;   // new cosine
                s     = beta_new / r1; // new sine

                // Update solution
                p_oold                 = p_old;
                p_old                  = p;
                p.noalias()            = (w - r2 * p_old - r3 * p_oold) / r1;
                const RealScalar step  = beta_one * c * eta;
                x                     += step * p;

                // Estimated update of the (preconditioned) residual norm squared.
                residualNorm2             *= s * s;
                bool rrnorm_has_converged  = residualNorm2 < threshold2;

                static constexpr RealScalar rrnorm_floor = std::numeric_limits<RealScalar>::denorm_min();
                RealScalar                  rrnorm       = std::sqrt(residualNorm2 / rhsNorm2); // Relative residual norm

                rrnorm            = std::max(rrnorm, rrnorm_floor);
                RealScalar deltax = RealScalar{1};

                if(!std::isfinite(rrnorm)) {
                    std::printf(" -- minres: rrnorm is not finite ... quitting\n");
                    break;
                }

                // if(((iters > 0) && (iters % rate_resid2 == 0)) || rrnorm_has_converged) {
                //     VectorType r_true             = rhs - mat * x;
                //     VectorType z_true             = precond.solve(r_true);
                //     RealScalar residualNorm2_true = numext::real(r_true.dot(z_true));
                //     eigen_assert(residualNorm2_true >= RealScalar{0} && "PRECONDITIONER IS NOT POSITIVE DEFINITE");
                //
                //     if constexpr(std::is_same_v<RealScalar, double>) {
                //         RealScalar drift = residualNorm2_true / std::max(residualNorm2, RealScalar{0});
                //         std::printf("corrected residualnorm2 %.5e -> %.5e, drift=%.3e\n", residualNorm2, residualNorm2_true, drift);
                //     }
                // }

                static constexpr RealScalar tiny = std::numeric_limits<RealScalar>::denorm_min();
                if(iters >= rate_rrnorm and iters % rate_rrnorm == 0) {
                    rrnorm_history.push_back(-std::log10(std::max(rrnorm, tiny)));
                    while(rrnorm_history.size() > max_history_size) rrnorm_history.pop_front();
                }
                auto refresh_deltax = [&x, &deltax, &step, &p]() {
                    constexpr RealScalar eps    = std::numeric_limits<RealScalar>::epsilon();
                    const RealScalar     xNorm  = x.norm();
                    const RealScalar     dxNorm = std::abs(step) * p.norm();
                    deltax                      = dxNorm / std::max(xNorm, eps); // ||dx|| / ||x|| where dx = step * p
                };

                if((iters >= rate_deltax) and (iters % rate_deltax == 0)) {
                    refresh_deltax();
                    deltax_history.push_back(-std::log10(std::max(deltax, tiny)));
                    while(deltax_history.size() > max_history_size) deltax_history.pop_front();
                }

                if((iters >= rate_checks and iters % rate_checks == 0) or rrnorm_has_converged) {
                    // Check for stall
                    if(iters % rate_deltax != 0) refresh_deltax();
                    auto rr_stats = get_stats(rrnorm_history, RealScalar{rate_rrnorm});
                    auto dx_stats = get_stats(deltax_history, RealScalar{rate_deltax});

                    bool rrnorm_has_made_progress = rrnorm < rrnorm0 * std::max(tol_error, RealScalar{0.63f});
                    bool rrnorm_large_history     = rrnorm_history.size() >= max_history_size / 2;
                    bool rrnorm_slope_issmall     = rrnorm_large_history and std::abs(rr_stats.slope_avg) < RealScalar{1e-4f};
                    bool rrnorm_slope_isclean     = rrnorm_large_history and std::abs(rr_stats.slope_sdv) < RealScalar{1e-3f};
                    bool rrnorm_has_saturated     = rrnorm_slope_issmall and rrnorm_slope_isclean;

                    bool                  deltax_large_history = deltax_history.size() >= max_history_size / 2;
                    bool                  deltax_slope_issmall = deltax_large_history and std::abs(dx_stats.slope_avg) < RealScalar{1e-4f};
                    bool                  deltax_slope_isclean = deltax_large_history and std::abs(dx_stats.slope_sdv) < RealScalar{1e-3f};
                    [[maybe_unused]] bool deltax_has_saturated = deltax_slope_issmall and deltax_slope_isclean;

                    bool minres_has_saturated = rrnorm_has_saturated and rrnorm_has_made_progress;
                    minres_has_converged      = rrnorm_has_converged ? (minres_has_converged + 1) : 0;

                    if constexpr(std::is_same_v<RealScalar, double>) {
                        if((iters >= rate_printf && iters % rate_printf == 0) or minres_has_converged or minres_has_saturated) {
                            bool has_printed = false;
                            if(iters >= rate_printf && iters % rate_printf == 0) {
                                std::printf("MINRES k: %4ld |rk|=%.3e |r0|=%.3e |rk|/|r0|=%.3e δ=%.3e log10 "
                                            "| rr avg: %.3e  std: %.3e  rel: %.3e sla: %.3e sls: %.3e "
                                            "| dx avg: %.3e  std: %.3e  rel: %.3e sla: %.3e sls: %.3e",
                                            iters, std::sqrt(residualNorm2), std::sqrt(rhsNorm2), rrnorm, deltax,                 //
                                            rr_stats.avg, rr_stats.sdv, rr_stats.rel_sdv, rr_stats.slope_avg, rr_stats.slope_sdv, //
                                            dx_stats.avg, dx_stats.sdv, dx_stats.rel_sdv, dx_stats.slope_avg, dx_stats.slope_sdv  //
                                );
                                has_printed = true;
                            }
                            if(minres_has_saturated) {
                                // std::printf(" -- minres saturated (rrnorm %d, deltax %d)\n", rrnorm_has_saturated, deltax_has_saturated);
                                // break;
                            } else if(minres_has_converged >= 1) {
                                // std::printf(" -- minres converged for %d iters\n", minres_has_converged);
                                break;
                            } else if(has_printed)
                                std::printf("\n");
                        }
                    }
                }

                eta = -s * eta; // update eta
                iters++;        // increment iteration number (for output purposes)
            }

            /* Compute error. Note that this is the estimated error. The real
             error |Ax-b|/|b| may be slightly larger */
            tol_error = std::sqrt(residualNorm2 / rhsNorm2);
        }

    }

    template<typename _MatrixType, int _UpLo = Lower, typename _Preconditioner = IdentityPreconditioner>
    class MINRES_ST;

    namespace internal {

        template<typename _MatrixType, int _UpLo, typename _Preconditioner>
        struct traits<MINRES_ST<_MatrixType, _UpLo, _Preconditioner> > {
            typedef _MatrixType     MatrixType;
            typedef _Preconditioner Preconditioner;
        };

    }

    /** \ingroup IterativeLinearSolvers_Module
     * \brief A minimal residual solver for sparse symmetric problems
     *
     * This class allows to solve for A.x = b sparse linear problems using the MINRES_ST algorithm
     * of Paige and Saunders (1975). The sparse matrix A must be symmetric (possibly indefinite).
     * The vectors x and b can be either dense or sparse.
     *
     * \tparam _MatrixType the type of the sparse matrix A, can be a dense or a sparse matrix.
     * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower,
     *               Upper, or Lower|Upper in which the full matrix entries will be considered. Default is Lower.
     * \tparam _Preconditioner the type of the preconditioner. Default is DiagonalPreconditioner
     *
     * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
     * and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations
     * and NumTraits<Scalar>::epsilon() for the tolerance.
     *
     * This class can be used as the direct solver classes. Here is a typical usage example:
     * \code
     * int n = 10000;
     * VectorXd x(n), b(n);
     * SparseMatrix<double> A(n,n);
     * // fill A and b
     * MINRES_ST<SparseMatrix<double> > mr;
     * mr.compute(A);
     * x = mr.solve(b);
     * std::cout << "#iterations:     " << mr.iterations() << std::endl;
     * std::cout << "estimated error: " << mr.error()      << std::endl;
     * // update b, and solve again
     * x = mr.solve(b);
     * \endcode
     *
     * By default the iterations start with x=0 as an initial guess of the solution.
     * One can control the start using the solveWithGuess() method.
     *
     * MINRES_ST can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
     *
     * \sa class ConjugateGradient, BiCGSTAB, SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
     */
    template<typename _MatrixType, int _UpLo, typename _Preconditioner>
    class MINRES_ST : public IterativeSolverBase<MINRES_ST<_MatrixType, _UpLo, _Preconditioner> > {
        typedef IterativeSolverBase<MINRES_ST> Base;
        using Base::m_error;
        using Base::m_info;
        using Base::m_isInitialized;
        using Base::m_iterations;
        using Base::matrix;

        public:
        using Base::_solve_impl;
        typedef _MatrixType                     MatrixType;
        typedef typename MatrixType::Scalar     Scalar;
        typedef typename MatrixType::RealScalar RealScalar;
        typedef _Preconditioner                 Preconditioner;

        enum { UpLo = _UpLo };

        public:
        /** Default constructor. */
        MINRES_ST() : Base() {}

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
        explicit MINRES_ST(const EigenBase<MatrixDerived> &A) : Base(A.derived()) {}

        /** Destructor. */
        ~MINRES_ST() {}

        /** \internal */
        template<typename Rhs, typename Dest>
        void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const {
            typedef typename Base::MatrixWrapper    MatrixWrapper;
            typedef typename Base::ActualMatrixType ActualMatrixType;
            enum { TransposeInput = (!MatrixWrapper::MatrixFree) && (UpLo == (Lower | Upper)) && (!MatrixType::IsRowMajor) && (!NumTraits<Scalar>::IsComplex) };
            typedef typename internal::conditional<TransposeInput, Transpose<const ActualMatrixType>, ActualMatrixType const &>::type RowMajorWrapper;
            EIGEN_STATIC_ASSERT(internal::check_implication(MatrixWrapper::MatrixFree, UpLo == (Lower | Upper)),
                                MATRIX_FREE_CONJUGATE_GRADIENT_IS_COMPATIBLE_WITH_UPPER_UNION_LOWER_MODE_ONLY);
            typedef typename internal::conditional<UpLo == (Lower | Upper), RowMajorWrapper,
                                                   typename MatrixWrapper::template ConstSelfAdjointViewReturnType<UpLo>::Type>::type SelfAdjointWrapper;

            m_iterations = Base::maxIterations();
            m_error      = Base::m_tolerance;
            RowMajorWrapper row_mat(matrix());
            internal::minres_st(SelfAdjointWrapper(row_mat), b, x, Base::m_preconditioner, m_iterations, m_error);
            m_info = m_error <= Base::m_tolerance ? Success : NoConvergence;
        }

        protected:
    };

} // end namespace Eigen

#endif // EIGEN_MINRES_ST_H
