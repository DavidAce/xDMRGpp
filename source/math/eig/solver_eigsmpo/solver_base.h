#pragma once
#include "config/enums.h"
#include "io/fmt_custom.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include "StopReason.h"
#include <spdlog/spdlog.h>
// #include "tensors/site/env/EnvEne.h"
// #include "tensors/site/env/EnvPair.h"
// #include "tensors/site/env/EnvVar.h"
#include "tid/tid.h"
#include <Eigen/Eigenvalues>
#include <source_location>

template<typename Scalar> class JacobiDavidsonOperator;
enum class ResidualCorrectionType { NONE, CHEAP_OLSEN, FULL_OLSEN, JACOBI_DAVIDSON, AUTO };

inline std::string_view ResidualCorrectionToString(ResidualCorrectionType rct) {
    switch(rct) {
        case ResidualCorrectionType::NONE: return "NONE";
        case ResidualCorrectionType::CHEAP_OLSEN: return "CHEAP_OLSEN";
        case ResidualCorrectionType::FULL_OLSEN: return "FULL_OLSEN";
        case ResidualCorrectionType::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
        case ResidualCorrectionType::AUTO: return "AUTO";
    }
}

inline ResidualCorrectionType StringToResidualCorrection(std::string_view rct) {
    if(rct == "NONE") return ResidualCorrectionType::NONE;
    if(rct == "CHEAP_OLSEN") return ResidualCorrectionType::CHEAP_OLSEN;
    if(rct == "FULL_OLSEN") return ResidualCorrectionType::FULL_OLSEN;
    if(rct == "JACOBI_DAVIDSON") return ResidualCorrectionType::JACOBI_DAVIDSON;
    if(rct == "AUTO") return ResidualCorrectionType::AUTO;
    return ResidualCorrectionType::NONE;
}

template<typename Scalar>
class solver_base {
    public:
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    void setLogger(spdlog::level::level_enum logLevel, const std::string &name = "");

    solver_base(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1, MatVecMPOS<Scalar> &H2,
                spdlog::level::level_enum logLevel_ = spdlog::level::warn);

    private:
    struct Status {
        VectorReal optVal;
        VectorReal oldVal;
        VectorReal absDiff;
        VectorReal relDiff;
        RealScalar initVal = std::numeric_limits<RealScalar>::quiet_NaN();
        // RealScalar                max_eval = RealScalar{1};
        private:
        std::deque<RealScalar> min_eval_history;
        std::deque<RealScalar> max_eval_history;

        public:
        void                      commit_evals(RealScalar min_eval, RealScalar max_eval);
        RealScalar                op_norm_estimate(OptAlgo algo) const;
        RealScalar                max_eval_estimate() const;
        RealScalar                min_eval_estimate() const;
        RealScalar                condition   = RealScalar{1};
        RealScalar                H1_max_eval = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                H2_max_eval = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                H1_min_eval = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                H2_min_eval = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                gap         = std::numeric_limits<RealScalar>::infinity();
        RealScalar                gap_H1      = std::numeric_limits<RealScalar>::infinity();
        RealScalar                gap_H2      = std::numeric_limits<RealScalar>::infinity();
        std::vector<Eigen::Index> optIdx;
        Eigen::Index              iter                                          = 0;
        Eigen::Index              iter_last_restart                             = 0;
        Eigen::Index              iter_last_preconditioner_tolerance_adjustment = 0;
        Eigen::Index              iter_last_preconditioner_H1_limit_adjustment  = 0;
        Eigen::Index              iter_last_preconditioner_H2_limit_adjustment  = 0;
        Eigen::Index              num_matvecs                                   = 0; /*!< Number of matvecs this iteration */
        Eigen::Index              num_matvecs_inner                             = 0; /*!< Number of matvecs over all iterations */
        Eigen::Index              num_matvecs_total                             = 0; /*!< Number of matvecs over all iterations */
        Eigen::Index              num_precond                                   = 0;
        Eigen::Index              num_precond_inner                             = 0;
        Eigen::Index              num_precond_total                             = 0;
        Eigen::Index              numMGS                                        = 0;
        tid::ur                   time_elapsed;
        tid::ur                   time_matvecs;
        tid::ur                   time_precond;
        tid::ur                   time_matvecs_inner;
        tid::ur                   time_precond_inner;
        tid::ur                   time_matvecs_total;
        tid::ur                   time_precond_total;

        bool rNorm_below_rnormTol = false;
        bool rNorm_below_gap      = false;
        bool rNorm_below_gapH1    = false;
        bool rNorm_below_gapH2    = false;

        VectorReal                rNorms, rNorms_H1, rNorms_H2;
        std::deque<VectorReal>    rNorms_history;
        std::deque<VectorReal>    optVals_history;
        std::deque<Eigen::Index>  matvecs_history;
        size_t                    max_history_size = 5;
        std::vector<Eigen::Index> nonZeroCols; // Nonzero Gram Schmidt columns
        Eigen::Index              numZeroRows   = 0;
        std::vector<std::string>  stopMessage   = {};
        StopReason                stopReason    = StopReason::none;
        OptRitz                   ritz_internal = OptRitz::NONE;
    };

    private:
    Eigen::Index      i_HQ     = -1;
    Eigen::Index      i_HQ_cur = -1;
    RealScalar        get_rNorms_log10_change_per_iteration();
    RealScalar        get_rNorms_log10_change_per_matvec();
    RealScalar        get_rNorms_log10_standard_deviation();
    static RealScalar get_max_standard_deviation(const std::deque<VectorReal> &v, bool apply_log10);
    bool              rNorm_has_saturated();
    bool              optVal_has_saturated(RealScalar threshold = 0);
    void              adjust_preconditioner_tolerance(const Eigen::Ref<const MatrixType> &S);
    void              adjust_preconditioner_H1_limits();
    void              adjust_preconditioner_H2_limits();
    void              adjust_residual_correction_type();

    protected:
    spdlog::level::level_enum       logLevel = spdlog::level::warn;
    std::shared_ptr<spdlog::logger> eiglog;

    using fMultHX_t = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;
    using fMultPX_t = std::function<MatrixType(const Eigen::Ref<const MatrixType> &, std::optional<const Eigen::Ref<const MatrixType>>)>;

    Eigen::Index qBlocks = 0;

    MatrixType        get_wBlock(fMultPX_t MultPX);
    MatrixType        get_mBlock();
    MatrixType        get_sBlock(const MatrixType &S_in, fMultPX_t MultPX);
    MatrixType        get_rBlock();
    const MatrixType &get_HQ();
    const MatrixType &get_HQ_cur();
    void              unset_HQ();
    void              unset_HQ_cur();

    [[nodiscard]] MatrixType cheap_Olsen_correction(const MatrixType &V, const MatrixType &S);
    [[nodiscard]] MatrixType full_Olsen_correction(const MatrixType &V, const MatrixType &S);
    [[nodiscard]] MatrixType jacobi_davidson_correction(const MatrixType &V, const MatrixType &S);

    VectorType JacobiDavidsonSolver(JacobiDavidsonOperator<Scalar>      &matRepl, //
                                    const VectorType                    &rhs,     //
                                    IterativeLinearSolverConfig<Scalar> &cfg);

    [[nodiscard]] MatrixType chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       /*!< input Q (orthonormal) */
                                             RealScalar                          lambda_min, /*!< estimated smallest eigenvalue */
                                             RealScalar                          lambda_max, /*!< estimated largest eigenvalue */
                                             RealScalar                          lambda_cut, /*!< cut-off (e.g. λmin for low-end) */
                                             int                                 degree      /*!< polynomial degree k */
    );
    [[nodiscard]] MatrixType qr_and_chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref /*!< input Q (may be non-orthonormal) */);
    void                     orthogonalize(const Eigen::Ref<const MatrixType> &X,       // (N, xcols)
                                           Eigen::Ref<MatrixType>              Y,       // (N, ycols)
                                           RealScalar                          normTol, // The largest allowed norm error
                                           Eigen::Ref<VectorIdxT>              mask     // block norm mask, size = n_blocks = ycols / blockWidth
                        );
    void                     orthonormalize(const Eigen::Ref<const MatrixType> &X,       // (N, xcols)
                                            Eigen::Ref<MatrixType>              Y,       // (N, ycols)
                                            RealScalar                          normTol, // The largest allowed norm error
                                            Eigen::Ref<VectorIdxT>              mask     // block norm mask, size = n_blocks = ycols / blockWidth
                        );
    void                     compress_cols(MatrixType &X, const VectorIdxT &mask);
    void                     compress_rows_and_cols(MatrixType &X, const VectorIdxT &mask);

    void assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location = std::source_location::current());
    void assert_orthonormal(const Eigen::Ref<const MatrixType> &X, RealScalar threshold,
                            const std::source_location &location = std::source_location::current());
    void assert_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, RealScalar threshold,
                           const std::source_location &location = std::source_location::current());

    bool       use_preconditioner                      = false;
    bool       use_initial_guess                       = false;
    int        chebyshev_filter_degree                 = 0;                 /*!< The chebyshev polynomial degree to use when filtering */
    RealScalar chebyshev_filter_relative_gap_threshold = RealScalar{1e-2f}; /*!< Enable chebyshev filtering if the relative spectral gap is smaller than this */
    RealScalar chebyshev_filter_lambda_cut_bias =
        RealScalar{1e-1f}; /*!< A percentage between 0 and 1. Value 0.1 puts lambda_cut 10% towards evals(1) from evals(0) */
    bool use_extra_ritz_vectors_in_the_next_basis            = false; /*!< Add the b next-best ritz vector block M to form the next basis (LOBPCG only) */
    ResidualCorrectionType residual_correction_type_internal = ResidualCorrectionType::NONE;
    eig::Preconditioner    preconditioner_type               = eig::Preconditioner::NONE;

    public:
    Status                      status = {};
    Eigen::Index                N;                                         /*!< The size of the underlying state tensor */
    Eigen::Index                mps_size;                                  /*!< The size of the underlying state tensor in mps representation (equal to N!) */
    std::array<Eigen::Index, 3> mps_shape;                                 /*!< The shape of the underlying state tensor in mps representation */
    Eigen::Index                nev                               = 1;     /*!< Number of eigenvalues to find */
    Eigen::Index                ncv                               = 8;     /*!< Krylov dimension, i.e. {V, H1V..., H2V...} ( minimum 2, recommend 3 or more) */
    Eigen::Index                b                                 = 2;     /*!< The block size */
    bool                        use_refined_rayleigh_ritz         = false; /*!< Refined ritz extraction uses 1 matvec per nev */
    bool                        use_relative_rnorm_tolerance      = true;
    bool                        use_adaptive_inner_tolerance      = true;
    bool                        use_deflated_inner_preconditioner = false;
    bool                        use_coarse_inner_preconditioner   = false;
    bool                        use_rayleigh_quotients_instead_of_evals      = false;
    bool                        use_b_orthonormal_jd_projection              = false;
    bool                        dev_orthogonalization_before_preconditioning = false;
    bool                        dev_cheap_olsen_as_jd_initial_guess          = false;
    bool                        dev_append_extra_blocks_to_basis             = false;

    ResidualCorrectionType residual_correction_type = ResidualCorrectionType::NONE;
    OptAlgo                algo;    /*!< Selects the current DMRG algorithm */
    OptRitz                ritz;    /*!< Selects the target eigenvalues */
    MatVecMPOS<Scalar>    &H1, &H2; /*!< The Hamiltonian and Hamiltonian squared operators */
    MatrixType             T;       /*!< The projections of H1 H2 to the tridiagonal Lanczos basis */
    MatrixType             A, B, W, Q;
    MatrixType             HQ;                             /*!< Save H*Q when preconditioning */
    MatrixType             HQ_cur;                         /*!< Save H*Q_cur when preconditioning */
    MatrixType             H1Q, H2Q;                       /*!< H1 or H2 times the basis blocks Q used for GDMRG */
    MatrixType             V;                              /*!< Holds the current top ritz eigenvectors. Use this to pass initial guesses */
    MatrixType             HV;                             /*!< Holds the current top ritz eigenvectors multiplied by H. */
    MatrixType             H1V;                            /*!< Holds the current top ritz eigenvectors multiplied by H1 (for GDMRG). */
    MatrixType             H2V;                            /*!< Holds the current top ritz eigenvectors multiplied by H2 (for GDMRG). */
    MatrixType             V_prev;                         /*!< Holds the previous top ritz eigenvectors */
    MatrixType             S, S1, S2;                      /*!< The residual vectors for the top b ritz vectors, also for H1 and H2 (for GDMRG) */
    MatrixType             V_prec, S_prec, D_prec, W_prec; /*!< The results directly after preconditioning, to use as initial guess for the next iteration */
    MatrixType             M, HM, H1M, H2M;                /*!< The b next best residual vectors M, and with the applied operators */
    VectorReal             T_evals;
    MatrixType             T_evecs;

    Eigen::HouseholderQR<MatrixType> hhqr;

    const RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
    RealScalar       tol = std::numeric_limits<RealScalar>::epsilon() * 10000;
    /* clang-format off */
    RealScalar       normTolQ   = std::numeric_limits<RealScalar>::epsilon() * 100; /*!< Normalization tolerance for columns in Q. */
    RealScalar       orthTolQ   = std::numeric_limits<RealScalar>::epsilon() * 10000; /*!< Orthonormality tolerance between columns in Q. Orthonormality can be improved with extra DGKS passes */
    RealScalar       quotTolB   = RealScalar{1e-10f};                                 /*!< Quotient tolerance for |B|/|A|. Triggers the Lanczos recurrence breakdown. */
    /* clang-format on */

    /*! Convergence tolerance of ritz-vector residuals.
     * Converged if rnorm < tol * opNorm. */
    RealScalar rnormTol() const {
        if(use_relative_rnorm_tolerance)
            return tol * status.op_norm_estimate(algo);
        else
            return tol;
    }

    /*! Norm tolerance of B-matrices.
     * Triggers the Lanczos recurrence breakdown. */
    [[nodiscard]] RealScalar bNormTol(const RealScalar B_norm) const noexcept {
        auto scale = std::max({status.max_eval_estimate(), B_norm, RealScalar{1}}); //  H_norm tracks A norms already
        return N * eps * scale;
    }
    /*! Norm tolerance of B-matrices.
     * Triggers the Lanczos recurrence breakdown. */
    RealScalar bNormTol(const MatrixType &B) const noexcept { return bNormTol(B.norm()); }

    [[nodiscard]] bool bNormIsOK(const MatrixType &B) const noexcept {
        auto B_norm = B.norm();
        return B_norm >= bNormTol(B_norm);
    }
    [[nodiscard]] bool bNormIsOK(const RealScalar &B_norm) const noexcept { return B_norm >= bNormTol(B_norm); }
    [[nodiscard]] bool bQuotIsOK(const MatrixType &B, const MatrixType &A) const noexcept {
        auto quotBA = B.norm() / A.norm();
        return quotBA < quotTolB;
    }
    [[nodiscard]] bool bQuotIsOK(RealScalar B_norm, RealScalar A_norm) const noexcept {
        auto quotBA = B_norm / A_norm;
        return quotBA >= quotTolB;
    }
    [[nodiscard]] bool bIsOK(const MatrixType &B, const MatrixType &A) const noexcept {
        auto B_norm = B.norm();
        auto A_norm = A.norm();
        return bNormIsOK(B_norm) and bQuotIsOK(B_norm, A_norm);
    }

    Eigen::Index max_iters   = 100;
    Eigen::Index max_matvecs = -1ul;

    RealScalar rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon();
    RealScalar absDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;
    RealScalar relDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;

    Eigen::Index get_jcbMaxBlockSize() const;
    void         set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize);
    void         set_preconditioner_type(eig::Preconditioner preconditioner_type_);
    void         set_preconditioner_params(Eigen::Index maxiters = 20000, RealScalar initialTol = RealScalar{1e-1f}, Eigen::Index jcbMaxBlockSize = -1ul);
    void         set_chebyshevFilterRelGapThreshold(RealScalar threshold);
    void         set_chebyshevFilterLambdaCutBias(RealScalar bias);
    void         set_chebyshevFilterDegree(Eigen::Index degree);

    MatrixType MultHX(const Eigen::Ref<const MatrixType> &X);
    MatrixType MultH1X(const Eigen::Ref<const MatrixType> &X);
    MatrixType MultH2X(const Eigen::Ref<const MatrixType> &X);

    MatrixType MultPX(const Eigen::Ref<const MatrixType> &X, std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);
    MatrixType MultP1X(const Eigen::Ref<const MatrixType> &X, std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);
    MatrixType MultP2X(const Eigen::Ref<const MatrixType> &X, std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);

    std::vector<Eigen::Index> get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals);

    void init();

    virtual void build() = 0;
    virtual void diagonalizeT();
    virtual void diagonalizeT1T2(); /*!< For GDMRG (generalized problem) */

    template<typename Comp>
    std::vector<Eigen::Index> getIndices(const VectorType &v, const Eigen::Index offset, const Eigen::Index num, Comp comp) {
        std::vector<Eigen::Index> idx(static_cast<size_t>(v.size()));
        Eigen::Index              numSort = offset + num;
        std::iota(idx.begin(), idx.end(), 0); // 1) build an index array [0, 1, 2, …, N-1]
        std::partial_sort(idx.begin(), idx.begin() + numSort, idx.end(),
                          [&](auto i, auto j) { return comp(std::real(v(i)), std::real(v(j))); }); // Sort the first offset+num elements
        return std::vector(idx.begin() + offset, idx.begin() + offset + num);                      // now idx[offset...offset+num) are the num sorted indices
    }

    void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S, VectorReal &rNorms);
    void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms);
    void extractRitzVectors();

    void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &H1V, MatrixType &H2V, MatrixType &S, VectorReal &rNorms);
    void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &HV, MatrixType &S, VectorReal &rNorms);
    void refinedRitzVectors();

    void preamble();
    void updateStatus();
    void printStatus();

    void step() {
        preamble();
        build();
        diagonalizeT();
        extractRitzVectors();
        updateStatus();
        printStatus();
        status.iter++;
    }

    void run() {
        auto token_elapsed = status.time_elapsed.tic_token();
        init();
        printStatus();
        while(true) {
            step();
            if(status.stopReason != StopReason::none) break;
        }
    }
};
