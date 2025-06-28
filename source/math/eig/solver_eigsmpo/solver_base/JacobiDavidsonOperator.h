#pragma once
#include "math/tenx/fwd_decl.h"
// Eigen goes first
#include "../solver_base.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/tenx.h"
#include "tid/tid.h"
#include "tools/common/contraction/IterativeLinearSolverConfig.h"
#include "tools/common/contraction/IterativeLinearSolverPreconditioner.h"
#include "tools/common/log.h"
#include <complex>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <variant>

template<typename Scalar_>
class JacobiDavidsonOperator;
template<typename T>
using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace settings {
    static constexpr bool debug_jdop = false;
}

namespace Eigen::internal {
    // JacobiDavidsonOperator looks-like a Dense Matrix, so let's inherits its traits:
    template<typename Scalar_>
    struct traits<JacobiDavidsonOperator<Scalar_>> : public Eigen::internal::traits<DenseMatrix<Scalar_>> {};

}

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::Matrix.
template<typename Scalar_>
class JacobiDavidsonOperator : public Eigen::EigenBase<JacobiDavidsonOperator<Scalar_>> {
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

    const VectorType &v;
    const VectorType &Bv;
    const VectorType &s;

    RealScalar            vH_Bv = RealScalar{1};
    static constexpr auto eps   = std::numeric_limits<RealScalar>::epsilon();
    mutable VectorType    x_tmp; // Scratch memory
    mutable VectorType    y_tmp; // Scratch memory

    // Timers
    public:
    std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> ResidualOp;
    std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp;

    void _check_template_params() {};
    // Custom API:
    JacobiDavidsonOperator() = default;
    JacobiDavidsonOperator(const VectorType                                               &v_,          //
                           const VectorType                                               &Bv_,         //
                           const VectorType                                               &s_,          //
                           std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> ResidualOp_, //
                           std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp_)
        :

          v(v_), Bv(Bv_), s(s_), ResidualOp(ResidualOp_), MatrixOp(MatrixOp_) {
        vH_Bv = std::real(v.dot(Bv));
        x_tmp.resizeLike(v);
        y_tmp.resizeLike(v);
        assert(vH_Bv > eps); // B ≻ 0 ⇒ v†Bv strictly positive
    }

    template<typename Rhs>
    Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto result  = Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
        auto t_end   = std::chrono::high_resolution_clock::now();
        m_optimer += std::chrono::duration<double>(t_end - t_start).count();
        m_opcounter++;
        return result;
    }
    template<typename Rhs>
    Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct> gemm(const Eigen::MatrixBase<Rhs> &x) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto result  = MatrixOp(x);
        auto t_end   = std::chrono::high_resolution_clock::now();
        m_optimer += std::chrono::duration<double>(t_end - t_start).count();
        m_opcounter++;
        return result;
    }
    [[nodiscard]] Eigen::Index rows() const { return safe_cast<Eigen::Index>(v.size()); };
    [[nodiscard]] Eigen::Index cols() const { return safe_cast<Eigen::Index>(v.size()); };
    Eigen::Index               iterations() const { return m_opcounter; }
    double                     elapsed_time() const { return m_optimer; }
};

namespace Eigen::internal {

    template<typename Rhs, typename ReplScalar>
    struct generic_product_impl<JacobiDavidsonOperator<ReplScalar>, Rhs, DenseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<JacobiDavidsonOperator<ReplScalar>, Rhs, generic_product_impl<JacobiDavidsonOperator<ReplScalar>, Rhs>> {
        typedef typename Product<JacobiDavidsonOperator<ReplScalar>, Rhs>::Scalar Scalar;

        template<typename Dest>
        static void scaleAndAddTo(Dest &dst, const JacobiDavidsonOperator<ReplScalar> &mat, const Rhs &rhs, const Scalar &alpha) {
            // This method should implement "dst += alpha * lhs * rhs" inplace; however, for iterative solvers, alpha is always equal to 1, so let's not worry
            // about it.
            assert(alpha == Scalar(1) && "scaling is not implemented");
            assert(rhs.size() == mat.rows());
            assert(dst.size() == mat.rows());
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);

            {
                using VectorType                        = JacobiDavidsonOperator<ReplScalar>::VectorType;
                [[maybe_unused]] const auto &v          = mat.v;                     // Ritz vector
                const auto                  &Bv         = mat.Bv;                    // B * v    (cached)
                const auto                  &ResidualOp = mat.ResidualOp;            // (A - θB)
                VectorType                  &x_tmp      = mat.x_tmp;                 // Scratch memory
                [[maybe_unused]] VectorType &y_tmp      = mat.y_tmp;                 // Scratch memory
                const ReplScalar             inv_vHvB   = ReplScalar(1) / mat.vH_Bv; // 1 / (v† B v)

                // B-orthogonal projector
                auto project_B = [&](const VectorType &z) -> VectorType {
                    Scalar delta = inv_vHvB * Bv.dot(z); // (v† z)/(v† B v)
                    return z - Bv * delta;               // P_B z
                };

                // ---- apply JD operator ----

                // standard JD-op
                x_tmp.noalias() = project_B(rhs);
                y_tmp.noalias() = ResidualOp(x_tmp);
                dst.noalias()   = project_B(y_tmp);

                // harmonic JD-op
                // x_tmp.noalias() = project_B(rhs);
                // dst.noalias()   = ResidualOp(x_tmp);
            }
        }
    };

}

template<typename Scalar>
typename solver_base<Scalar>::VectorType solver_base<Scalar>::JacobiDavidsonSolver(JacobiDavidsonOperator<Scalar>      &matRepl, //
                                                                                 const VectorType                    &rhs,     //
                                                                                 IterativeLinearSolverConfig<Scalar> &cfg) {
    using PreconditionerType = IterativeLinearSolverPreconditioner<JacobiDavidsonOperator<Scalar>>;
    using DefSolverType      = Eigen::ConjugateGradient<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>;
    using IndSolverType      = std::conditional_t<sfinae::is_std_complex_v<Scalar>,                                    //
                                                  Eigen::BiCGSTAB<JacobiDavidsonOperator<Scalar>, PreconditionerType>, //
                                                  Eigen::MINRES<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>
                                                  // Eigen::GMRES<JacobiDavidsonOperator<Scalar>, PreconditionerType>
                                                  >;
    std::variant<IndSolverType, DefSolverType> solverVariant;
    if(cfg.matdef == MatDef::IND)
        solverVariant.template emplace<0>();
    else
        solverVariant.template emplace<1>();

    auto get_solver_name = [&]() -> std::string {
        if(std::holds_alternative<DefSolverType>(solverVariant)) return "Eigen::ConjugateGradient";
        if(std::holds_alternative<IndSolverType>(solverVariant)) {
            if constexpr(sfinae::is_std_complex_v<Scalar>) {
                return "Eigen::BiCGSTAB";
            } else {
                return "Eigen::MINRES";
            }
        }
        return "Unknown solver";
    };
    VectorType res;
    auto       run = [&](auto &solver) {
        auto t_jdop = tid::tic_token("jdop", tid::level::higher);
        solver.setMaxIterations(cfg.maxiters);
        solver.setTolerance(cfg.tolerance);
        solver.preconditioner().attach(&matRepl, &cfg);
        solver.compute(matRepl);
        if(cfg.initialGuess.size() == rhs.size()) {
            // eiglog->info("solving with guess");
            res = solver.solveWithGuess(rhs, cfg.initialGuess);
        } else {
            // eiglog->info("solving without guess");
            res = solver.solve(rhs);
        }
        t_jdop.toc();
        if constexpr(settings::debug_jdop)
            tools::log->trace("{}: size {} | info {} | tol {:8.5e} | err {:8.5e} | iter {} | mat iter {} | time {:.2e}", get_solver_name(), matRepl.rows(),
                                    static_cast<int>(solver.info()), fp(solver.tolerance()), fp(solver.error()), solver.iterations(), matRepl.iterations(),
                                    t_jdop->get_last_interval());
        cfg.result.iters += solver.iterations();
        cfg.result.matvecs += matRepl.iterations();
        cfg.result.precond += solver.preconditioner().iterations();
        cfg.result.time += t_jdop->get_last_interval();
        cfg.result.time_matvecs += matRepl.elapsed_time();
        cfg.result.time_precond += solver.preconditioner().elapsed_time();

        // cfg.result.total_iters += solver.iterations();
        // cfg.result.total_matvecs += matRepl.iterations();
        // cfg.result.total_precond += solver.preconditioner().iterations();
        // cfg.result.total_time += t_jdop->get_last_interval();
        // cfg.result.total_time_matvecs += matRepl.elapsed_time();
        // cfg.result.total_time_precond += solver.preconditioner().elapsed_time();

        cfg.result.error = solver.error();
        cfg.result.info  = solver.info();
        if(std::isnan(solver.error())) throw except::runtime_error("NaN in solver");
        return res;
    };

    try {
        std::visit(run, solverVariant);
    } catch(const std::exception &e) {
        if(std::holds_alternative<DefSolverType>(solverVariant)) {
            tools::log->error("Exception in {}: {}\n Trying another solver...", get_solver_name(), e.what());
            if(std::holds_alternative<IndSolverType>(solverVariant))
                solverVariant.template emplace<1>();
            else if(std::holds_alternative<DefSolverType>(solverVariant))
                solverVariant.template emplace<0>();
            std::visit(run, solverVariant);
        } else {
            tools::log->error("Exception in {}: {}", get_solver_name(), e.what());
            throw;
        }
    }

    return res;
}
