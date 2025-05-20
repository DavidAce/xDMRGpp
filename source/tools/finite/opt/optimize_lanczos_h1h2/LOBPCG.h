#pragma once
#include "io/fmt_custom.h"
#include "math/eig/log.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/gramSchmidt.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include "SolverBase.h"
#include "SolverExit.h"
#include <Eigen/Eigenvalues>

template<typename Scalar>
class LOBPCG : public SolverBase<Scalar> {
    public:
    using SolverBase<Scalar>::SolverBase;

    using RealScalar = typename SolverBase<Scalar>::RealScalar;
    using MatrixType = typename SolverBase<Scalar>::MatrixType;
    using VectorType = typename SolverBase<Scalar>::VectorType;
    using VectorReal = typename SolverBase<Scalar>::VectorReal;
    using VectorIdxT = typename SolverBase<Scalar>::VectorIdxT;

    using SolverBase<Scalar>::use_preconditioner;
    using SolverBase<Scalar>::use_refined_rayleigh_ritz;
    using SolverBase<Scalar>::chebyshev_filter_degree;
    using SolverBase<Scalar>::use_chebyshev_basis_during_ritz_extraction;
    using SolverBase<Scalar>::status;
    using SolverBase<Scalar>::N;
    using SolverBase<Scalar>::mps_size;
    using SolverBase<Scalar>::mps_shape;
    using SolverBase<Scalar>::nev;
    using SolverBase<Scalar>::ncv;
    using SolverBase<Scalar>::b;
    using SolverBase<Scalar>::algo;
    using SolverBase<Scalar>::ritz;
    using SolverBase<Scalar>::H1;
    using SolverBase<Scalar>::H2;
    using SolverBase<Scalar>::T;
    using SolverBase<Scalar>::A;
    using SolverBase<Scalar>::B;
    using SolverBase<Scalar>::W;
    using SolverBase<Scalar>::Q;
    using SolverBase<Scalar>::X;
    using SolverBase<Scalar>::HQ;
    using SolverBase<Scalar>::HX;
    using SolverBase<Scalar>::V;
    using SolverBase<Scalar>::T_evals;
    using SolverBase<Scalar>::T_evecs;
    using SolverBase<Scalar>::hhqr;
    using SolverBase<Scalar>::eps;
    using SolverBase<Scalar>::tol;
    using SolverBase<Scalar>::normTolQ;
    using SolverBase<Scalar>::orthTolQ;
    using SolverBase<Scalar>::quotTolB;
    using SolverBase<Scalar>::max_iters;
    using SolverBase<Scalar>::max_matvecs;
    using SolverBase<Scalar>::rnormRelDiffTol;
    using SolverBase<Scalar>::absDiffTol;
    using SolverBase<Scalar>::relDiffTol;

    using SolverBase<Scalar>::bIsOK;
    using SolverBase<Scalar>::get_ritz_indices;
    using SolverBase<Scalar>::MultHX;
    using SolverBase<Scalar>::MultPX;

    private:
    void                     write_Q3b_LOBPCG(Eigen::Index i);
    Eigen::Index             max_wBlocks = 16;
    [[nodiscard]] MatrixType chebyshevFilter(const Eigen::Ref<const MatrixType> &Qref,       /*!< input Q (orthonormal) */
                                             const Eigen::Ref<const MatrixType> &HQref,      /*!< input H * Q*/
                                             RealScalar                          lambda_min, /*!< estimated smallest eigenvalue */
                                             RealScalar                          lambda_max, /*!< estimated largest eigenvalue */
                                             RealScalar                          lambda_cut, /*!< cut-off (e.g. λmin for low-end) */
                                             int                                 degree      /*!< polynomial degree k */
    );

    public:
    void build() final;
    void extractResidualNorms() final;
    void set_ResidualHistoryLength(Eigen::Index k);
};
