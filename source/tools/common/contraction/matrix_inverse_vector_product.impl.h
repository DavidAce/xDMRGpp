#pragma once
#include "math/tenx/fwd_decl.h"
// Eigen goes first
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/tenx.h"
#include "MatrixLikeOperator.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/contraction/IterativeLinearSolverConfig.h"
#include "tools/common/contraction/IterativeLinearSolverPreconditioner.h"
#include "tools/common/log.h"
#include <complex>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <variant>

namespace settings {
    static constexpr bool debug_jdop = false;
}

template<typename Scalar>
tools::common::contraction::VectorType<Scalar> tools::common::contraction::matrix_inverse_vector_product(MatrixLikeOperator<Scalar>          &MatrixOp, //
                                                                                                         const Scalar                        *rhs_ptr,  //
                                                                                                         const IterativeLinearSolverConfig<Scalar> &cfg) {
    using PreconditionerType = IterativeLinearSolverPreconditioner<MatrixLikeOperator<Scalar>>;
    using DefSolverType      = Eigen::ConjugateGradient<MatrixLikeOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>;
    using IndSolverType      = std::conditional_t<sfinae::is_std_complex_v<Scalar>,                                //
                                                  Eigen::BiCGSTAB<MatrixLikeOperator<Scalar>, PreconditionerType>, //
                                                  Eigen::MINRES<MatrixLikeOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>
                                                  // Eigen::GMRES<MatrixLikeOperator<Scalar>, PreconditionerType>
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
    auto               rhs = Eigen::Map<const VectorType<Scalar>>(rhs_ptr, MatrixOp.rows());
    VectorType<Scalar> res;
    auto               run = [&](auto &solver) {
        auto t_jdop = tid::tic_token("jdop", tid::level::higher);
        solver.setMaxIterations(cfg.maxiters);
        solver.setTolerance(cfg.tolerance);
        solver.preconditioner().attach(&MatrixOp, &cfg);
        solver.compute(MatrixOp);
        if(cfg.initialGuess.size() == rhs.size()) {
            // eiglog->info("solving with guess");
            res = solver.solveWithGuess(rhs, cfg.initialGuess);
        } else {
            // eiglog->info("solving without guess");
            res = solver.solve(rhs);
        }
        t_jdop.toc();
        if constexpr(settings::debug_jdop)
            tools::log->trace("{}: size {} | info {} | tol {:8.5e} | err {:8.5e} | iter {} | mat iter {} | time {:.2e}", get_solver_name(), MatrixOp.rows(),
                                            static_cast<int>(solver.info()), fp(solver.tolerance()), fp(solver.error()), solver.iterations(), MatrixOp.iterations(),
                                            t_jdop->get_last_interval());
        cfg.result.iters += solver.iterations();
        cfg.result.matvecs += MatrixOp.iterations();
        cfg.result.precond += solver.preconditioner().iterations();
        cfg.result.time += t_jdop->get_last_interval();
        cfg.result.time_matvecs += MatrixOp.elapsed_time();
        cfg.result.time_precond += solver.preconditioner().elapsed_time();
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
