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
#include "solver_base.h"
#include "StopReason.h"
#include <Eigen/Eigenvalues>

template<typename Scalar>
class solver_gdplusk : public solver_base<Scalar> {
    public:
    using solver_base<Scalar>::solver_base;

    using RealScalar = typename solver_base<Scalar>::RealScalar;
    using MatrixType = typename solver_base<Scalar>::MatrixType;
    using VectorType = typename solver_base<Scalar>::VectorType;
    using VectorReal = typename solver_base<Scalar>::VectorReal;
    using VectorIdxT = typename solver_base<Scalar>::VectorIdxT;
    using fMultHX_t  = typename solver_base<Scalar>::fMultHX_t;
    using fMultPX_t  = typename solver_base<Scalar>::fMultPX_t;

    using solver_base<Scalar>::eiglog;
    using solver_base<Scalar>::use_preconditioner;
    using solver_base<Scalar>::use_refined_rayleigh_ritz;
    using solver_base<Scalar>::dev_orthogonalization_before_preconditioning;
    using solver_base<Scalar>::dev_append_extra_blocks_to_basis;
    using solver_base<Scalar>::residual_correction_type_internal;
    using solver_base<Scalar>::chebyshev_filter_degree;
    using solver_base<Scalar>::status;
    using solver_base<Scalar>::N;
    using solver_base<Scalar>::mps_size;
    using solver_base<Scalar>::mps_shape;
    using solver_base<Scalar>::nev;
    using solver_base<Scalar>::ncv;
    using solver_base<Scalar>::b;
    using solver_base<Scalar>::qBlocks;
    using solver_base<Scalar>::algo;
    using solver_base<Scalar>::ritz;
    using solver_base<Scalar>::H1;
    using solver_base<Scalar>::H2;
    using solver_base<Scalar>::T;
    using solver_base<Scalar>::A;
    using solver_base<Scalar>::B;
    using solver_base<Scalar>::W;
    using solver_base<Scalar>::Q;
    using solver_base<Scalar>::M;
    using solver_base<Scalar>::HM;
    using solver_base<Scalar>::H1M;
    using solver_base<Scalar>::H2M;
    using solver_base<Scalar>::HQ;
    using solver_base<Scalar>::H1Q;
    using solver_base<Scalar>::H2Q;

    using solver_base<Scalar>::get_wBlock;
    using solver_base<Scalar>::get_mBlock;
    using solver_base<Scalar>::get_sBlock;
    using solver_base<Scalar>::get_rBlock;

    using solver_base<Scalar>::V;
    using solver_base<Scalar>::HV;
    using solver_base<Scalar>::H1V;
    using solver_base<Scalar>::H2V;
    using solver_base<Scalar>::S;
    using solver_base<Scalar>::S1;
    using solver_base<Scalar>::S2;
    using solver_base<Scalar>::V_prev;
    using solver_base<Scalar>::T_evals;
    using solver_base<Scalar>::T_evecs;
    using solver_base<Scalar>::hhqr;
    using solver_base<Scalar>::eps;
    using solver_base<Scalar>::tol;
    using solver_base<Scalar>::rnormTol;
    using solver_base<Scalar>::normTolQ;
    using solver_base<Scalar>::orthTolQ;
    using solver_base<Scalar>::quotTolB;
    using solver_base<Scalar>::max_iters;
    using solver_base<Scalar>::max_matvecs;
    using solver_base<Scalar>::rnormRelDiffTol;
    using solver_base<Scalar>::absDiffTol;
    using solver_base<Scalar>::relDiffTol;

    using solver_base<Scalar>::bIsOK;
    using solver_base<Scalar>::get_ritz_indices;
    using solver_base<Scalar>::extractRitzVectors;
    using solver_base<Scalar>::MultHX;
    using solver_base<Scalar>::MultH1X;
    using solver_base<Scalar>::MultH2X;
    using solver_base<Scalar>::MultPX;
    using solver_base<Scalar>::MultP1X;
    using solver_base<Scalar>::MultP2X;
    using solver_base<Scalar>::assert_allFinite;
    using solver_base<Scalar>::assert_orthonormal;
    using solver_base<Scalar>::assert_orthogonal;
    using solver_base<Scalar>::chebyshevFilter;
    using solver_base<Scalar>::qr_and_chebyshevFilter;
    using solver_base<Scalar>::orthogonalize;
    using solver_base<Scalar>::orthonormalize;
    using solver_base<Scalar>::compress_cols;
    using solver_base<Scalar>::compress_rows_and_cols;

    private:
    using solver_base<Scalar>::use_extra_ritz_vectors_in_the_next_basis;

    void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
    void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);

    Eigen::Index max_wBlocks     = 1;
    Eigen::Index max_mBlocks     = 1;
    Eigen::Index max_sBlocks     = 1;
    Eigen::Index max_s1Blocks    = 0;
    Eigen::Index max_s2Blocks    = 0;
    Eigen::Index vBlocks         = 0;
    Eigen::Index wBlocks         = 0;
    Eigen::Index mBlocks         = 0;
    Eigen::Index rBlocks         = 0;
    Eigen::Index sBlocks         = 0;
    Eigen::Index s1Blocks        = 0; // Last resort blocks if GDMRG fails to add sBlocks
    Eigen::Index s2Blocks        = 0; // Last resort blocks if GDMRG fails to add sBlocks
    Eigen::Index kBlocks         = 0;
    Eigen::Index maxBasisBlocks  = 8;
    Eigen::Index maxRetainBlocks = 1;
    MatrixType   G;
    // Store all Qenr_i
    void delete_blocks_from_left_until_orthogonal(const Eigen::Ref<const MatrixType> X,         // (N, xcols)
                                                  MatrixType                        &Y,         // (N, ycols)
                                                  MatrixType                        &HY,        // (N, ycols)
                                                  Eigen::Index                       maxBlocks, // Keep this many blocks at most
                                                  RealScalar                         threshold  // Allow this much overlap between V and Q_enr
    );
    void selective_orthonormalize(const Eigen::Ref<const MatrixType> X,            // (N, xcols)
                                  Eigen::Ref<MatrixType>             Y,            // (N, ycols)
                                  RealScalar                         breakdownTol, // The smallest allowed norm
                                  VectorIdxT                        &mask          // block norm mask, size = n_blocks = ycols / blockWidth
    );

    MatrixType get_Q_res(fMultPX_t MultPX);

    public:
    bool inject_randomness = false;
    void build() final;
    void build(MatrixType &Q_res, MatrixType &Q, MatrixType &HQ, fMultHX_t MultHX);
    void build(MatrixType &Q1_res, MatrixType &Q2_res, MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q);
    void build(MatrixType &Q_res, MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q);
    void set_maxLanczosResidualHistory(Eigen::Index k);
    void set_maxExtraRitzHistory(Eigen::Index m);
    void set_maxRitzResidualHistory(Eigen::Index s);
    void set_maxBasisBlocks(Eigen::Index bb);
    void set_maxRetainBlocks(Eigen::Index rb);
};
