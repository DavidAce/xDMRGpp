#pragma once

#include <complex>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <memory>

enum class MatDef { IND, SEMI, DEF };
enum class PreconditionerType { JACOBI, CHEBYSHEV };

template<typename Scalar>
struct IterativeLinearSolverConfig {
    using Real       = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    private:
    struct JacobiPreconditionerConfig {
        using LLTType                          = Eigen::LLT<MatrixType, Eigen::Lower>;
        using LDLTType                         = Eigen::LDLT<MatrixType, Eigen::Lower>;
        using LUType                           = Eigen::PartialPivLU<MatrixType>;
        using LLTJcbBlocksType                 = std::vector<std::tuple<long, std::unique_ptr<LLTType>>>;
        using LDLTJcbBlocksType                = std::vector<std::tuple<long, std::unique_ptr<LDLTType>>>;
        using LUJcbBlocksType                  = std::vector<std::tuple<long, std::unique_ptr<LUType>>>;
        const Scalar            *invdiag       = nullptr;
        const LLTJcbBlocksType  *lltJcbBlocks  = nullptr;
        const LDLTJcbBlocksType *ldltJcbBlocks = nullptr;
        const LUJcbBlocksType   *luJcbBlocks   = nullptr;
    };

    struct ChebyshevPreconditionerConfig {
        Real         lambda_min = std::numeric_limits<Real>::quiet_NaN();
        Real         lambda_max = std::numeric_limits<Real>::quiet_NaN();
        Eigen::Index degree     = 0;
    };
    struct Result {
        Eigen::Index           iters              = 0; /*! For the last run */
        Eigen::Index           matvecs            = 0; /*! For the last run */
        Eigen::Index           precond            = 0; /*! For the last run */
        double                 time               = 0; /*! For the last run */
        double                 time_matvecs       = 0; /*! For the last run */
        double                 time_precond       = 0; /*! For the last run */
        Eigen::Index           total_iters        = 0; /*! For all runs */
        Eigen::Index           total_matvecs      = 0; /*! For all runs */
        Eigen::Index           total_precond      = 0; /*! For all runs */
        double                 total_time         = 0; /*! For all runs */
        double                 total_time_matvecs = 0; /*! For all runs */
        double                 total_time_precond = 0; /*! For all runs */
        Real                   error              = 0;
        Eigen::ComputationInfo info               = Eigen::ComputationInfo::NoConvergence;

        Result &operator+=(const Result &other) {
            this->iters += other.iters;
            this->matvecs += other.matvecs;
            this->precond += other.precond;
            this->time += other.time;
            this->error = std::max(this->error, other.error);
            this->info  = std::max(this->info, other.info);
            this->total_iters += other.iters;
            this->total_matvecs += other.matvecs;
            this->total_precond += other.precond;
            this->total_time += other.time;
            this->total_time_matvecs += other.time_matvecs;
            this->total_time_precond += other.time_precond;

            return *this;
        }
        void add_latest(const Result &other) {
            this->iters += other.iters;
            this->matvecs += other.matvecs;
            this->precond += other.precond;
            this->time += other.time;
            this->error = std::max(this->error, other.error);
            this->info  = std::max(this->info, other.info);
        }
        void copy_latest(const Result &other) {
            this->iters   = other.iters;
            this->matvecs = other.matvecs;
            this->precond = other.precond;
            this->time    = other.time;
            this->error   = std::max(this->error, other.error);
            this->info    = std::max(this->info, other.info);
            this->total_iters += other.iters;
            this->total_matvecs += other.matvecs;
            this->total_precond += other.precond;
            this->total_time += other.time;
            this->total_time_matvecs += other.time_matvecs;
            this->total_time_precond += other.time_precond;
        }
    };

    public:
    long                          maxiters    = 10000;
    Real                          tolerance   = Real{1e-1f};
    MatDef                        matdef      = MatDef::DEF; /*! Whether the matrix is indefinite or (semi) definite*/
    PreconditionerType            precondType = PreconditionerType::JACOBI;
    JacobiPreconditionerConfig    jacobi      = {};
    ChebyshevPreconditionerConfig chebyshev   = {};
    Result                        result      = {};
};