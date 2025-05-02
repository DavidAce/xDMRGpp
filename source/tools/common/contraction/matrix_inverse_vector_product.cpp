#include "math/tenx/fwd_decl.h"
// Eigen goes first
#include "debug/exceptions.h"
#include "InvMatVecCfg.h"
#include "JacobiPreconditioner.h"
#include "math/cast.h"
#include "math/tenx.h"
#include "tid/tid.h"
#include "tools/common/contraction.h"
#include "tools/common/log.h"
#include <complex>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <variant>

template<typename Scalar_>
class MatrixReplacement;
template<typename T>
using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace Eigen::internal {
    // MatrixReplacement looks-like a Dense Matrix, so let's inherits its traits:
    template<typename Scalar_>
    struct traits<MatrixReplacement<Scalar_>> : public Eigen::internal::traits<DenseMatrix<Scalar_>> {};

}

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::Matrix.
template<typename Scalar_>
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement<Scalar_>> {
    public:
    // Required typedefs, constants, and method:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    typedef int StorageIndex;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    const Scalar_      *envL = nullptr;
    const Scalar_      *envR = nullptr;
    const Scalar_      *mpo  = nullptr;
    std::array<long, 3> shape_mps;
    std::array<long, 4> shape_mpo;
    std::array<long, 3> shape_envL;
    std::array<long, 3> shape_envR;
    std::vector<Scalar> shift_mpo;
    long                mps_size;
    // Timers
    mutable int                 counter = 0;
    mutable std::vector<Scalar> tmp;

    [[nodiscard]] Eigen::Index rows() const { return safe_cast<int>(mps_size); }; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */
    [[nodiscard]] Eigen::Index cols() const { return safe_cast<int>(mps_size); }; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */

    template<typename Rhs>
    Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }
    void _check_template_params() {};
    // Custom API:
    MatrixReplacement() = default;

    void attachTensors(const Scalar_ *const envL_,      /*!< The left block tensor.  */
                       const Scalar_ *const envR_,      /*!< The right block tensor.  */
                       const Scalar_ *const mpo_,       /*!< The Hamiltonian MPO's  */
                       std::array<long, 3>  shape_mps_, /*!< An array containing the shapes of the mps  */
                       std::array<long, 4>  shape_mpo_  /*!< An array containing the shapes of the mpo  */
    ) {
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
        mps_size = shape_mps[0] * shape_mps[1] * shape_mps[2];
        //        t_multAx = std::make_unique<class_tic_toc>(true, 5, "Time MultAx");
    }
};

// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of init::generic_product_impl:
namespace Eigen::internal {

    template<typename Rhs, typename ReplScalar>
    struct generic_product_impl<MatrixReplacement<ReplScalar>, Rhs, DenseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<MatrixReplacement<ReplScalar>, Rhs, generic_product_impl<MatrixReplacement<ReplScalar>, Rhs>> {
        typedef typename Product<MatrixReplacement<ReplScalar>, Rhs>::Scalar Scalar;

        template<typename Dest>
        static void scaleAndAddTo(Dest &dst, const MatrixReplacement<ReplScalar> &mat, const Rhs &rhs, const Scalar &alpha) {
            // This method should implement "dst += alpha * lhs * rhs" inplace; however, for iterative solvers, alpha is always equal to 1, so let's not worry
            // about it.
            assert(alpha == Scalar(1) && "scaling is not implemented");
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            mat.tmp.resize(static_cast<size_t>(dst.size()));
            Eigen::Map<Dest> tmp_map(mat.tmp.data(), dst.size());
            tools::common::contraction::matrix_vector_product(tmp_map.data(), rhs.data(), mat.shape_mps, mat.mpo, mat.shape_mpo, mat.envL, mat.shape_envL,
                                                              mat.envR, mat.shape_envR);

            dst.noalias() += tmp_map;
            mat.counter++;
        }
    };

}

template<typename Scalar>
void tools::common::contraction::matrix_inverse_vector_product(Scalar             *res_ptr,                                 //
                                                               const Scalar *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                               const Scalar *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                               const Scalar *const envL_ptr, std::array<long, 3> envL_dims, //
                                                               const Scalar *const envR_ptr, std::array<long, 3> envR_dims, //
                                                               InvMatVecCfg<Scalar> cfg) {
    // Here we return x <-- A^-1 * b
    // Where A^-1 * b is obtained by solving
    //       A*x = b
    // using an iterative matrix-free solver.

    // We have previously tried using bfgs for unconstrained minimization of f = |Aφ - ψ|², where
    //      φ = res
    //      ψ = mps
    //      A = (H-E) = effective hamiltonian (from mpo and env)
    //
    // The gradient is ∇f = (H-E)²φ - (H-E)ψ (note that the (H-E)² is just (H-E) applied twice, not the second moment).
    // After minimization we have φ ~ (H-E)⁻¹ψ = A⁻¹ * x
    // The result was not better than using BiCGSTAB or MINRES

    {
        auto mps  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(mps_ptr, mps_dims);
        auto mpo  = Eigen::TensorMap<const Eigen::Tensor<Scalar, 4>>(mpo_ptr, mpo_dims);
        auto envL = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envL_ptr, envL_dims);
        auto envR = Eigen::TensorMap<const Eigen::Tensor<Scalar, 3>>(envR_ptr, envR_dims);

        if(mps.dimension(1) != envL.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envL {}", mps.dimensions(), envL.dimensions());
        if(mps.dimension(2) != envR.dimension(0)) throw except::runtime_error("Dimension mismatch mps {} and envR {}", mps.dimensions(), envR.dimensions());
        if(mps.dimension(0) != mpo.dimension(2)) throw except::runtime_error("Dimension mismatch mps {} and mpo {}", mps.dimensions(), mpo.dimensions());
        if(envL.dimension(2) != mpo.dimension(0)) throw except::runtime_error("Dimension mismatch envL {} and mpo {}", envL.dimensions(), mpo.dimensions());
        if(envR.dimension(2) != mpo.dimension(1)) throw except::runtime_error("Dimension mismatch envR {} and mpo {}", envR.dimensions(), mpo.dimensions());
    }

    // Define the "matrix-free" matrix replacement.
    MatrixReplacement<Scalar> matRepl;
    matRepl.attachTensors(envL_ptr, envR_ptr, mpo_ptr, mps_dims, mpo_dims);
    auto res = Eigen::Map<tenx::VectorType<Scalar>>(res_ptr, matRepl.rows());
    auto mps = Eigen::Map<const tenx::VectorType<Scalar>>(mps_ptr, matRepl.rows());

    using DefSolverType = Eigen::ConjugateGradient<MatrixReplacement<Scalar>, Eigen::Upper | Eigen::Lower, JacobiPreconditioner<Scalar>>;
    using IndSolverType = std::conditional_t<sfinae::is_std_complex_v<Scalar>,                                         //
                                             Eigen::BiCGSTAB<MatrixReplacement<Scalar>, JacobiPreconditioner<Scalar>>, //
                                             Eigen::MINRES<MatrixReplacement<Scalar>, Eigen::Upper | Eigen::Lower, JacobiPreconditioner<Scalar>>>;
    std::variant<IndSolverType, DefSolverType> solverVariant;
    if(cfg.matdef == MatDef::IND)
        solverVariant.template emplace<0>();
    else
        solverVariant.template emplace<1>();

    auto                            t_mativec = tid::tic_token("matrix_inverse_vector_product", tid::level::higher);
    static tenx::VectorType<Scalar> guess;
    if(guess.size() != res.size()) guess = res;

    auto get_solver_name = [&]() -> std::string {
        if(std::holds_alternative<DefSolverType>(solverVariant)) return "Eigen::ConjugateGradient";
        if(std::holds_alternative<IndSolverType>(solverVariant))
            if constexpr(sfinae::is_std_complex_v<Scalar>)
                return "Eigen::BiCGSTAB";
            else
                return "Eigen::MINRES";
        return "Unknown solver";
    };

    auto run = [&](auto &solver) {
        solver.setMaxIterations(cfg.maxiters);
        solver.setTolerance(cfg.tolerance);
        if(cfg.invdiag != nullptr) {
            auto invdiag = Eigen::Map<const tenx::VectorType<Scalar>>(cfg.invdiag, matRepl.rows());
            if(invdiag.allFinite()) { solver.preconditioner().set_invdiag(invdiag); }
        }
        solver.compute(matRepl);
        res = solver.solveWithGuess(mps, guess);
        if(std::isnan(solver.error())) throw except::runtime_error("NaN in solver");
        tools::log->info("{}: size {} | info {} | tol {:8.5e} | err {:8.5e} | iter {} | counter {} | time {:.2e}", get_solver_name, mps.size(),
                         static_cast<int>(solver.info()), fp(solver.tolerance()), fp(solver.error()), solver.iterations(), matRepl.counter,
                         t_mativec->get_last_interval());
    };

    try {
        std::visit(run, solverVariant);
    } catch(const std::exception &e) {
        if(std::holds_alternative<DefSolverType>(solverVariant)) {
            tools::log->error("Exception in {}: {}\n Trying another solver...", get_solver_name(), e.what());
            solverVariant.template emplace<1>();
            std::visit(run, solverVariant);
        } else {
            tools::log->error("Exception in {}: {}", get_solver_name(), e.what());
            throw;
        }
    }

    guess = res;
}
template void tools::common::contraction::matrix_inverse_vector_product(fp32             *res_ptr,                                 //
                                                                        const fp32 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const fp32 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const fp32 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const fp32 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<fp32> cfg);
template void tools::common::contraction::matrix_inverse_vector_product(fp64             *res_ptr,                                 //
                                                                        const fp64 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const fp64 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const fp64 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const fp64 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<fp64> cfg);
template void tools::common::contraction::matrix_inverse_vector_product(fp128             *res_ptr,                                 //
                                                                        const fp128 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const fp128 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const fp128 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const fp128 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<fp128> cfg);
template void tools::common::contraction::matrix_inverse_vector_product(cx32             *res_ptr,                                 //
                                                                        const cx32 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const cx32 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const cx32 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const cx32 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<cx32> cfg);
template void tools::common::contraction::matrix_inverse_vector_product(cx64             *res_ptr,                                 //
                                                                        const cx64 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const cx64 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const cx64 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const cx64 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<cx64> cfg);
template void tools::common::contraction::matrix_inverse_vector_product(cx128             *res_ptr,                                 //
                                                                        const cx128 *const mps_ptr, std::array<long, 3> mps_dims,   //
                                                                        const cx128 *const mpo_ptr, std::array<long, 4> mpo_dims,   //
                                                                        const cx128 *const envL_ptr, std::array<long, 3> envL_dims, //
                                                                        const cx128 *const envR_ptr, std::array<long, 3> envR_dims, //
                                                                        InvMatVecCfg<cx128> cfg);