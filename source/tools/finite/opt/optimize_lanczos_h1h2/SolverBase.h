#pragma once
#include "config/enums.h"
#include "io/fmt_custom.h"
#include "math/eig/log.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include "SolverExit.h"
// #include "tensors/site/env/EnvEne.h"
// #include "tensors/site/env/EnvPair.h"
// #include "tensors/site/env/EnvVar.h"
#include <Eigen/Eigenvalues>

template<typename Scalar>
class SolverBase {
    public:
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    // using mpos_t = std::vector<std::reference_wrapper<const MpoSite<Scalar>>>;
    // using enve_t = env_pair<const EnvEne<Scalar> &>;
    // using envv_t = env_pair<const EnvVar<Scalar> &>;
    SolverBase(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, MatVecMPOS<Scalar> &H1, MatVecMPOS<Scalar> &H2);

    private:
    struct Status {
        VectorReal optVal;
        VectorReal oldVal;
        VectorReal absDiff;
        VectorReal relDiff;
        RealScalar initVal       = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar H_norm_approx = RealScalar{1};

        std::vector<Eigen::Index> optIdx;
        Eigen::Index              iter        = 0;
        Eigen::Index              num_matvecs = 0;
        Eigen::Index              num_precond = 0;
        Eigen::Index              numMGS      = 0;
        VectorReal                rNorms;
        std::vector<Eigen::Index> nonZeroCols; // Nonzero Gram Schmidt columns
        Eigen::Index              numZeroRows   = 0;
        std::vector<std::string>  exitMsg       = {};
        SolverExit                exit          = SolverExit::ok;
        OptRitz                   ritz_internal = OptRitz::NONE;
    };

    protected:
    bool use_preconditioner                         = false;
    int  chebyshev_filter_degree                    = 0;
    bool use_chebyshev_basis_during_ritz_extraction = false;

    public:
    Status                      status = {};
    Eigen::Index                N;                        /*!< The size of the underlying state tensor */
    Eigen::Index                mps_size;                 /*!< The size of the underlying state tensor in mps representation (equal to N!) */
    std::array<Eigen::Index, 3> mps_shape;                /*!< The shape of the underlying state tensor in mps representation */
    Eigen::Index                nev              = 1;     /*!< Number of eigenvalues to find */
    Eigen::Index                ncv              = 8;     /*!< Krylov dimension, i.e. {V, H1V..., H2V...} ( minimum 2, recommend 3 or more) */
    Eigen::Index                b                = 2;     /*!< The block size */
    bool                        use_refined_rayleigh_ritz = false; /*!< Refined ritz extraction uses 1 matvec per nev */

    OptAlgo                          algo;    /*!< Selects the current DMRG algorithm */
    OptRitz                          ritz;    /*!< Selects the target eigenvalues */
    MatVecMPOS<Scalar>              &H1, &H2; /*!< The Hamiltonian and Hamiltonian squared operators */
    MatrixType                       T;       /*!< The projections of H1 H2 to the tridiagonal Lanczos basis */
    MatrixType                       A, B, W, Q, X;
    MatrixType                       HQ, HX; /*! Save H*Q when preconditioning */
    MatrixType                       V;      /*! Holds the current ritz eigenvectors. Use this to pass initial guesses */
    VectorReal                       T_evals;
    MatrixType                       T_evecs;
    Eigen::HouseholderQR<MatrixType> hhqr;

    const RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
    RealScalar       tol = std::numeric_limits<RealScalar>::epsilon() * 10000;
    /* clang-format off */
    RealScalar       normTolQ   = std::numeric_limits<RealScalar>::epsilon() * 100; /*!< Normalization tolerance for columns in Q. */
    RealScalar       orthTolQ   = std::numeric_limits<RealScalar>::epsilon() * 10000; /*!< Orthonormality tolerance between columns in Q. Orthonormality can be improved with extra DGKS passes */
    RealScalar       quotTolB   = RealScalar{1e-10f};                                 /*!< Quotient tolerance for |B|/|A|. Triggers the Lanczos recurrence breakdown. */
    /* clang-format on */

    /*! Norm tolerance of ritz-vector residuals.
     * Lanczos converges if rnorm < normTolR * H_norm. */
    RealScalar rnormTol() const { return tol * status.H_norm_approx; }

    /*! Norm tolerance of B-matrices.
     * Triggers the Lanczos recurrence breakdown. */
    [[nodiscard]] RealScalar bNormTol(const RealScalar B_norm) const noexcept {
        auto scale = std::max({status.H_norm_approx, B_norm, RealScalar{1}}); //  H_norm tracks A norms already
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
    Eigen::Index max_matvecs = 1000;

    RealScalar rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon();
    RealScalar absDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;
    RealScalar relDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;

    void set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize);
    void set_chebyshevFilterDegree(Eigen::Index degree);

    MatrixType MultHX(const Eigen::Ref<const MatrixType> &X);

    MatrixType MultPX(const Eigen::Ref<const MatrixType> &X);

    std::vector<Eigen::Index> get_ritz_indices(OptRitz ritz, Eigen::Index num, const VectorReal &evals);

    void init();

    virtual void build() = 0;

    void diagonalizeT();

    template<typename Comp>
    std::vector<Eigen::Index> getIndices(const VectorType &v, const Eigen::Index k, Comp comp) {
        std::vector<Eigen::Index> idx(static_cast<size_t>(v.size()));
        std::iota(idx.begin(), idx.end(), 0);                             // 1) build an index array [0, 1, 2, …, N-1]
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), comp); // Sort k elements
        return std::vector(idx.begin(), idx.begin() + k);                 // now idx[0..k) are the k sorted indices
    }

    void extractRitzVectors();
    void updateStatus();

    virtual void extractResidualNorms() = 0;

    void step() {
        build();
        diagonalizeT();
        extractRitzVectors();
        extractResidualNorms();
        updateStatus();
        status.iter++;
    }

    void run() {
        init();
        while(true) {
            step();
            if(status.exit != SolverExit::ok) break;
        }
    }
};
