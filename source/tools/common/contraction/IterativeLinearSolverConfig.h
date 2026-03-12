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
    using VectorReal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    private:
    struct JacobiPreconditionerConfig {
        using LLTType                                   = Eigen::LLT<MatrixType, Eigen::Lower>;
        using LDLTType                                  = Eigen::LDLT<MatrixType, Eigen::Lower>;
        using LUType                                    = Eigen::PartialPivLU<MatrixType>;
        using QRType                                    = Eigen::ColPivHouseholderQR<MatrixType>;
        using LLTJcbBlocksType                          = std::vector<std::tuple<long, int, std::unique_ptr<LLTType>>>;
        using LDLTJcbBlocksType                         = std::vector<std::tuple<long, int, std::unique_ptr<LDLTType>>>;
        using LUJcbBlocksType                           = std::vector<std::tuple<long, int, std::unique_ptr<LUType>>>;
        using QRJcbBlocksType                           = std::vector<std::tuple<long, int, std::unique_ptr<QRType>>>;
        const Scalar            *invdiag                = nullptr;
        const LLTJcbBlocksType  *lltJcbBlocks           = nullptr;
        const LDLTJcbBlocksType *ldltJcbBlocks          = nullptr;
        const LUJcbBlocksType   *luJcbBlocks            = nullptr;
        const QRJcbBlocksType   *qrJcbBlocks            = nullptr;
        const VectorReal        *jcbInvSqrtMultiplicity = nullptr;
        Eigen::Index             jcbMaxMultiplicity     = 1l;
        Eigen::Index             jcbNumPasses           = 1l;
        Real                     cond                   = std::numeric_limits<Real>::quiet_NaN();
        MatrixType               deflationEigVecs       = {};
        VectorType               deflationEigInvs       = {};
        MatrixType               coarseZ                = {};
        MatrixType               coarseBZ               = {};
        bool                     skipjcb                = false;
    };

    struct ChebyshevPreconditionerConfig {
        Real         lambda_min = std::numeric_limits<Real>::quiet_NaN();
        Real         lambda_max = std::numeric_limits<Real>::quiet_NaN();
        Eigen::Index degree     = 0;
    };

    public:
    struct Result {
        Eigen::Index           iters          = 0; /*! For the last run */
        Eigen::Index           matvecs        = 0; /*! For the last run */
        Eigen::Index           precond        = 0; /*! For the last run */
        double                 time           = 0; /*! For the last run */
        double                 time_matvecs   = 0; /*! For the last run */
        double                 time_precond   = 0; /*! For the last run */
        double                 time_jacobi    = 0;
        double                 time_chebyshev = 0;
        Real                   error          = 0;
        Eigen::ComputationInfo info           = Eigen::ComputationInfo::NoConvergence;
        void                   add_latest(const Result &other) { *this += other; }
        Result                &operator+=(const Result &other) {
            this->iters          += other.iters;
            this->matvecs        += other.matvecs;
            this->precond        += other.precond;
            this->time           += other.time;
            this->time_matvecs   += other.time_matvecs;
            this->time_precond   += other.time_precond;
            this->time_jacobi    += other.time_jacobi;
            this->time_chebyshev += other.time_chebyshev;
            this->error           = other.error;
            this->info            = other.info;
            return *this;
        }

        void copy_latest(const Result &other) {
            this->iters          = other.iters;
            this->matvecs        = other.matvecs;
            this->precond        = other.precond;
            this->time           = other.time;
            this->time_matvecs   = other.time_matvecs;
            this->time_precond   = other.time_precond;
            this->time_jacobi    = other.time_jacobi;
            this->time_chebyshev = other.time_chebyshev;
            this->error          = other.error;
            this->info           = other.info;
        }

        void reset() {
            this->iters          = {};
            this->matvecs        = {};
            this->precond        = {};
            this->time           = {};
            this->time_matvecs   = {};
            this->time_precond   = {};
            this->time_jacobi    = {};
            this->time_chebyshev = {};
            this->error          = {};
            this->info           = {};
        }
    };

    long                          maxiters     = 1000;
    Real                          tolerance    = Real{0.1f};
    MatDef                        matdef       = MatDef::IND; /*! Whether the matrix is indefinite or (semi) definite*/
    PreconditionerType            precondType  = PreconditionerType::JACOBI;
    MatrixType                    initialGuess = {};
    JacobiPreconditionerConfig    jacobi       = {};
    ChebyshevPreconditionerConfig chebyshev    = {};
    mutable Result                result       = {};
};