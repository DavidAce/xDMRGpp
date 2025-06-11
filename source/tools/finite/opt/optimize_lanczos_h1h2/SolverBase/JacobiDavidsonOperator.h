#pragma once
#include "math/tenx/fwd_decl.h"
// Eigen goes first
#include "../SolverBase.h"
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

    const VectorType                                               &v;
    const VectorType                                               &s;
    std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> ResidualOp;
    RealScalar                                                      vH_v = 1;

    // Timers
    mutable Eigen::Index        counter = 0;
    mutable std::vector<Scalar> tmp;

    [[nodiscard]] Eigen::Index rows() const { return safe_cast<Eigen::Index>(v.size()); };
    [[nodiscard]] Eigen::Index cols() const { return safe_cast<Eigen::Index>(v.size()); };

    template<typename Rhs>
    Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }
    void _check_template_params() {};
    // Custom API:
    JacobiDavidsonOperator() = default;

    JacobiDavidsonOperator(const VectorType                                               &v_, //
                           const VectorType                                               &s_, //
                           std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> ResidualOp_)
        :

          v(v_), s(s_), ResidualOp(ResidualOp_) {
        vH_v = std::real(v.adjoint().dot(v));
    }
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
                auto &v          = mat.v;
                auto &vH_v       = mat.vH_v;
                auto &ResidualOp = mat.ResidualOp;
                using VectorType = JacobiDavidsonOperator<ReplScalar>::VectorType;
                // Project x to orthogonal complement of v
                Scalar delta = vH_v > 0 ? (v.adjoint().dot(rhs)) / vH_v : Scalar(0);
                assert(std::isfinite(std::abs(delta)));
                VectorType x_perp = rhs - v * delta;
                VectorType y      = ResidualOp(x_perp);

                assert(std::isfinite(y.norm()));
                assert(y.size() == v.size());
                // Project output again (for numerical stability)
                Scalar beta = vH_v > 0 ? (v.adjoint().dot(y)) / vH_v : Scalar(0);
                assert(std::isfinite(std::abs(beta)));
                dst.noalias() = y - v * beta;
            }
            mat.counter++;
        }
    };

}

template<typename Scalar>
typename SolverBase<Scalar>::VectorType SolverBase<Scalar>::JacobiDavidsonSolver(JacobiDavidsonOperator<Scalar>      &matRepl, //
                                                                                 const VectorType                    &rhs,     //
                                                                                 IterativeLinearSolverConfig<Scalar> &cfg) {
    using PreconditionerType = IterativeLinearSolverPreconditioner<JacobiDavidsonOperator<Scalar>>;
    using DefSolverType      = Eigen::ConjugateGradient<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>;
    using IndSolverType      = std::conditional_t<sfinae::is_std_complex_v<Scalar>,                                    //
                                                  Eigen::BiCGSTAB<JacobiDavidsonOperator<Scalar>, PreconditionerType>, //
                                                  Eigen::MINRES<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>>;
    std::variant<IndSolverType, DefSolverType> solverVariant;
    if(cfg.matdef == MatDef::IND)
        solverVariant.template emplace<0>();
    else
        solverVariant.template emplace<1>();

    auto t_jdop = tid::tic_token("jdop", tid::level::higher);

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
        solver.setMaxIterations(cfg.maxiters);
        solver.setTolerance(cfg.tolerance);
        solver.preconditioner().attach(&matRepl, &cfg);
        solver.compute(matRepl);
        res = solver.solve(rhs);
        if constexpr(settings::debug_jdop)
            tools::log->trace("{}: size {} | info {} | tol {:8.5e} | err {:8.5e} | iter {} | counter {} | time {:.2e}", get_solver_name(), matRepl.rows(),
                                    static_cast<int>(solver.info()), fp(solver.tolerance()), fp(solver.error()), solver.iterations(), matRepl.counter,
                                    t_jdop->get_last_interval());
        cfg.result.iters   = solver.iterations();
        cfg.result.matvecs = matRepl.counter;
        cfg.result.precond = solver.preconditioner().iterations();
        cfg.result.time    = t_jdop->get_last_interval();

        cfg.result.total_iters += solver.iterations();
        cfg.result.total_matvecs += matRepl.counter;
        cfg.result.total_precond += solver.preconditioner().iterations();
        cfg.result.total_time += t_jdop->get_last_interval();

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
