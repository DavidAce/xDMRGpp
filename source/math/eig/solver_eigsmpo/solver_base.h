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
enum class MaskPolicy { COMPRESS, RANDOMIZE };

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
    using fMultH_t   = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;
    using fMultP_t   = std::function<MatrixType(const Eigen::Ref<const MatrixType> &, const Eigen::Ref<const VectorReal> &,
                                                std::optional<const Eigen::Ref<const MatrixType>>)>;

    void setLogger(spdlog::level::level_enum logLevel, const std::string &name = "");

    solver_base(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1, MatVecMPOS<Scalar> &H2,
                MatVecMPOS<Scalar> &H1H2, spdlog::level::level_enum logLevel_ = spdlog::level::warn);

    struct OrthMeta {
        // By convention, H2Y is the matrix that we modify, while H2X is const.
        // private:
        // MatrixType cacheH2X;
        // MatrixType cacheH2Y;
        // fMultH_t   fMultH2;

        public:
        // const MatrixType                   &get_H2X() const { return cacheH2X; }
        // const Eigen::Ref<const MatrixType> &get_H2X(const Eigen::Ref<const MatrixType> &X) const;
        // MatrixType                         &get_H2Y() { return cacheH2X; }
        // Eigen::Ref<MatrixType>             &get_H2Y(Eigen::Ref<MatrixType> Y);
        // void                                set_cache_H2X(const Eigen::Ref<const MatrixType> &H2X) { cacheH2X = H2X; }
        // void                                set_cache_H2Y(Eigen::Ref<MatrixType> H2Y) { cacheH2Y = H2Y; }
        // void                                has_cache_H2X(Eigen::Index rows, Eigen::Index cols) { return cacheH2X.rows() == rows && cacheH2X.cols() == cols;
        // } void                                has_cache_H2Y(Eigen::Index rows, Eigen::Index cols) { return cacheH2Y.rows() == rows && cacheH2Y.cols() ==
        // cols; } void                                set_MultH2(fMultH_t fMultH2_) { fMultH2 = fMultH2_; }
        MatrixType Gram;
        VectorReal Rdiag;
        RealScalar maskTol   = 10 * std::numeric_limits<RealScalar>::epsilon();
        RealScalar orthTol   = 100 * std::numeric_limits<RealScalar>::epsilon();
        RealScalar orthError = std::numeric_limits<RealScalar>::quiet_NaN();
        VectorReal proj_sum_h;
        VectorReal proj_sum_h1;
        VectorReal proj_sum_h2;
        VectorReal scale_log;
        VectorIdxT mask;
        // bool       compress_cols       = true;
        // bool       randomize_tiny_cols = true;
        bool       force_refresh_h = false;
        MaskPolicy maskPolicy      = MaskPolicy::RANDOMIZE;
    };

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
        RealScalar                max_eval_estimate() const;
        RealScalar                min_eval_estimate() const;
        RealScalar                condition        = RealScalar{1};
        RealScalar                op_norm_estimate = RealScalar{1};
        RealScalar                T1_max_eval      = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                T2_max_eval      = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                T1_min_eval      = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                T2_min_eval      = std::numeric_limits<RealScalar>::quiet_NaN(); // Used for preconditioning
        RealScalar                gap              = std::numeric_limits<RealScalar>::infinity();
        RealScalar                gap_H1           = std::numeric_limits<RealScalar>::infinity();
        RealScalar                gap_H2           = std::numeric_limits<RealScalar>::infinity();
        VectorReal                T1_evals;
        VectorReal                T2_evals;
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

        VectorReal                rNorms;
        std::deque<VectorReal>    rNorms_history;
        std::deque<VectorReal>    optVals_history;
        std::deque<Eigen::Index>  matvecs_history;
        size_t                    max_history_size        = 5;
        size_t                    saturation_count_optVal = 0;
        size_t                    saturation_count_rNorm  = 0;
        size_t                    saturation_count_max    = 20;
        std::vector<Eigen::Index> nonZeroCols; // Nonzero Gram Schmidt columns
        Eigen::Index              numZeroRows   = 0;
        std::vector<std::string>  stopMessage   = {};
        StopReason                stopReason    = StopReason::none;
        OptRitz                   ritz_internal = OptRitz::NONE;
    };

    private:
    Eigen::Index i_HQ     = -1;
    Eigen::Index i_HQ_cur = -1;
    RealScalar   get_rNorms_log10_change_per_iteration();
    RealScalar   get_rNorms_log10_change_per_matvec();
    RealScalar   get_rNorms_log10_standard_deviation();
    RealScalar   get_op_norm_estimate(std::optional<RealScalar> eigval = std::nullopt) const;
    VectorReal   get_op_norm_estimates(Eigen::Ref<VectorReal> eigvals) const;

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

    Eigen::Index qBlocks = 0;

    MatrixType        get_wBlock(fMultP_t MultP);
    MatrixType        get_mBlock();
    MatrixType        get_sBlock(const MatrixType &S_in, fMultP_t MultP);
    MatrixType        get_rBlock();
    const MatrixType &get_HQ();
    const MatrixType &get_HQ_cur();
    void              unset_HQ();
    void              unset_HQ_cur();

    auto colMask2ColIndex(const VectorIdxT &mask) const {
        std::vector<Eigen::Index> index;
        for(Eigen::Index j = 0; j < mask.size(); ++j) {
            if(mask(j) == 1) { index.push_back(j); }
        }
        return index;
    }
    auto blockMask2ColIndex(const VectorIdxT &mask, Eigen::Index b) const {
        std::vector<Eigen::Index> index;
        Eigen::Index              nblocks = mask.size();
        for(Eigen::Index j = 0; j < nblocks; ++j) {
            if(mask(j) == 1) {
                for(Eigen::Index i = 0; i < b; ++i) index.push_back(j * b + i);
            }
        }
        assert(index.size() == static_cast<size_t>(nblocks * b));
        return index;
    }

    auto get_masked_blocks(const Eigen::Ref<const MatrixType> &Y, const VectorIdxT &mask) const {
        assert(mask.size() == Y.cols() / b);
        return Y(Eigen::all, blockMask2ColIndex(mask, b));
    }

    auto get_masked_cols(const Eigen::Ref<const MatrixType> &Y, const VectorIdxT &mask) const {
        assert(mask.size() == Y.cols());
        return Y(Eigen::all, colMask2ColIndex(mask));
    }

    [[nodiscard]] MatrixType cheap_Olsen_correction(const MatrixType &V, const MatrixType &S);
    [[nodiscard]] MatrixType full_Olsen_correction(const MatrixType &V, const MatrixType &S);
    [[nodiscard]] MatrixType jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S, const VectorReal &evals);
    [[nodiscard]] MatrixType jacobi_davidson_h2_correction(const MatrixType &V, const MatrixType &H2V, const MatrixType &S, const VectorReal &evals);

    VectorType JacobiDavidsonSolver(JacobiDavidsonOperator<Scalar>      &matRepl, //
                                    const VectorType                    &rhs,     //
                                    IterativeLinearSolverConfig<Scalar> &cfg);

    [[nodiscard]] MatrixType chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       /*!< linput Q (orthonormal) */
                                             RealScalar                          lambda_min, /*!< estimated smallest eigenvalue */
                                             RealScalar                          lambda_max, /*!< estimated largest eigenvalue */
                                             RealScalar                          lambda_cut, /*!< cut-off (e.g. λmin for low-end) */
                                             int                                 degree      /*!< polynomial degree k */
    );
    [[nodiscard]] MatrixType qr_and_chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref /*!< input Q (may be non-orthonormal) */);

    void mask_col_blocks(Eigen::Ref<MatrixType> Y, OrthMeta &m);
    void mask_cols(Eigen::Ref<MatrixType> Y, OrthMeta &m);

    void compress_col_blocks(MatrixType &X, const VectorIdxT &mask);
    void compress_cols(MatrixType &X, const VectorIdxT &mask);
    void compress_row_blocks(VectorReal &X, const VectorIdxT &mask);
    void compress_rows(VectorReal &X, const VectorIdxT &mask);

    void compress_rows_and_cols(MatrixType &X, const VectorIdxT &mask);

    void assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location = std::source_location::current());

    void block_l2_orthonormalize(MatrixType &Y, MatrixType &HY, OrthMeta &m);
    void block_l2_orthonormalize(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m);
    void block_l2_orthogonalize(const MatrixType &X, const MatrixType &HX, MatrixType &Y, MatrixType &HY, OrthMeta &m);
    void block_l2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y,
                                OrthMeta &m);

    void block_h2_orthonormalize_dgks(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m);
    void block_h2_orthonormalize_llt(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m);
    void block_h2_orthonormalize_eig(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m);
    // void block_h2_orthonormalize_old(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, OrthMeta &m);
    void block_h2_orthogonalize(const MatrixType &X, const MatrixType &H1X, const MatrixType &H2X, MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y,
                                OrthMeta &m);

    void assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m = {},
                               const std::source_location &location = std::source_location::current());
    void assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m = {},
                              const std::source_location &location = std::source_location::current());

    void assert_h2_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &H2X, const OrthMeta &m = {},
                               const std::source_location &location = std::source_location::current());
    void assert_h2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &H2Y, const OrthMeta &m = {},
                              const std::source_location &location = std::source_location::current());

    void pad_and_orthonormalize(MatrixType &Y, MatrixType &HY, Eigen::Index nBlocks, OrthMeta &m);
    void pad_and_orthonormalize(MatrixType &Y, MatrixType &H1Y, MatrixType &H2Y, Eigen::Index nBlocks, OrthMeta &m);

    bool       use_preconditioner                      = false;
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
    bool                        use_h2_inner_product                         = false;
    bool                        use_krylov_schur_gdplusk_restart             = false;
    bool                        use_h1h2_preconditioner                      = false;
    bool                        dev_thick_jd_projector                       = false;
    bool                        dev_orthogonalization_before_preconditioning = false;
    bool                        dev_cheap_olsen_as_jd_initial_guess          = false;
    bool                        dev_append_extra_blocks_to_basis             = false;
    bool                        dev_skipjcb                                  = false;
    std::string                 tag;

    ResidualCorrectionType residual_correction_type = ResidualCorrectionType::NONE;
    OptAlgo                algo;           /*!< Selects the current DMRG algorithm */
    OptRitz                ritz;           /*!< Selects the target eigenvalues */
    MatVecMPOS<Scalar>    &H1, &H2, &H1H2; /*!< The Hamiltonian and Hamiltonian squared operators */
    MatrixType             T;              /*!< The projections of H1 H2 to the tridiagonal Lanczos basis */
    MatrixType             A, B, W, Q;
    MatrixType             HQ;              /*!< Save H*Q when preconditioning */
    MatrixType             HQ_cur;          /*!< Save H*Q_cur when preconditioning */
    MatrixType             H1Q, H2Q;        /*!< H1 or H2 times the basis blocks Q used for GDMRG */
    MatrixType             V;               /*!< Holds the current top ritz eigenvectors. Use this to pass initial guesses */
    MatrixType             HV;              /*!< Holds the current top ritz eigenvectors multiplied by H. */
    MatrixType             H1V;             /*!< Holds the current top ritz eigenvectors multiplied by H1 (for GDMRG). */
    MatrixType             H2V;             /*!< Holds the current top ritz eigenvectors multiplied by H2 (for GDMRG). */
    MatrixType             V_prev;          /*!< Holds the previous top ritz eigenvectors */
    MatrixType             S, S1, S2;       /*!< The residual vectors for the top b ritz vectors, also for H1 and H2 (for GDMRG) */
    MatrixType             M, HM, H1M, H2M; /*!< The b next best residual vectors M, and with the applied operators */
    VectorReal             T_evals;
    MatrixType             T1, T2, T_evecs;

    Eigen::HouseholderQR<MatrixType> hhqr;

    const RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
    RealScalar       tol = std::numeric_limits<RealScalar>::epsilon() * 10000;
    /* clang-format off */
    RealScalar       normTol   = std::numeric_limits<RealScalar>::epsilon() * 10;   /*!< Normalization tolerance for columns in Q. */
    RealScalar       orthTol   = std::numeric_limits<RealScalar>::epsilon() * 100; /*!< Orthonormality tolerance between columns in Q. Orthonormality can be improved with extra DGKS passes */
    RealScalar       quotTolB  = RealScalar{1e-10f};                                 /*!< Quotient tolerance for |B|/|A|. Triggers the Lanczos recurrence breakdown. */
    /* clang-format on */

    /*! Convergence tolerance of ritz-vector residuals.
     * Converged if rnorm < tol * opNorm. */
    // VectorReal rnormTol(Eigen::Ref<VectorReal> evals) const;
    RealScalar rNormTol(Eigen::Index n) const;
    VectorReal rNormTols() const;

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

    MatrixType MultH(const Eigen::Ref<const MatrixType> &X);
    MatrixType MultH1(const Eigen::Ref<const MatrixType> &X);
    MatrixType MultH2(const Eigen::Ref<const MatrixType> &X);

    MatrixType MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                     std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);
    MatrixType MultP1(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                      std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);
    MatrixType MultP2(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                      std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);
    MatrixType MultP1P2(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                        std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);

    std::vector<Eigen::Index> get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const;

    void init();

    virtual void build() = 0;
    virtual void diagonalizeT();
    virtual void diagonalizeT1T2(); /*!< For GDMRG (generalized problem) */

    template<typename Comp>
    std::vector<Eigen::Index> getIndices(const VectorType &v, const Eigen::Index offset, const Eigen::Index num, Comp comp) const {
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

    std::pair<MatrixType, MatrixType> get_h2_normalizer_for_the_projected_pencil(const MatrixType &T2);
    MatrixType get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1, const MatrixType &T2);

    MatrixType get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                 const MatrixType &HQ);
    MatrixType get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &H1Q,
                                                 const MatrixType &H2Q);

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
