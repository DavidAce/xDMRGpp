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
    using fMultH_t   = typename solver_base<Scalar>::fMultH_t;
    using fMultP_t   = typename solver_base<Scalar>::fMultP_t;
    using OrthMeta   = typename solver_base<Scalar>::OrthMeta;
    // using MaskPolicy = typename solver_base<Scalar>::MaskPolicy;

    using solver_base<Scalar>::eiglog;
    using solver_base<Scalar>::use_preconditioner;
    using solver_base<Scalar>::use_refined_rayleigh_ritz;
    using solver_base<Scalar>::use_h2_inner_product;
    using solver_base<Scalar>::use_h1h2_preconditioner;
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
    using solver_base<Scalar>::normTol;
    using solver_base<Scalar>::orthTol;
    using solver_base<Scalar>::quotTolB;
    using solver_base<Scalar>::max_iters;
    using solver_base<Scalar>::max_matvecs;
    using solver_base<Scalar>::rnormRelDiffTol;
    using solver_base<Scalar>::absDiffTol;
    using solver_base<Scalar>::relDiffTol;

    using solver_base<Scalar>::bIsOK;
    using solver_base<Scalar>::get_ritz_indices;
    using solver_base<Scalar>::extractRitzVectors;
    using solver_base<Scalar>::MultH;
    using solver_base<Scalar>::MultH1;
    using solver_base<Scalar>::MultH2;
    using solver_base<Scalar>::MultP;
    using solver_base<Scalar>::MultP1;
    using solver_base<Scalar>::MultP2;

    using solver_base<Scalar>::block_l2_orthogonalize;
    using solver_base<Scalar>::block_l2_orthonormalize;
    using solver_base<Scalar>::block_h2_orthonormalize_dgks;
    using solver_base<Scalar>::block_h2_orthonormalize_llt;
    using solver_base<Scalar>::block_h2_orthogonalize;
    using solver_base<Scalar>::assert_l2_orthonormal;
    using solver_base<Scalar>::assert_l2_orthogonal;
    using solver_base<Scalar>::assert_h2_orthonormal;
    using solver_base<Scalar>::assert_h2_orthogonal;
    using solver_base<Scalar>::assert_allFinite;

    using solver_base<Scalar>::chebyshevFilter;
    using solver_base<Scalar>::qr_and_chebyshevFilter;
    using solver_base<Scalar>::compress_col_blocks;
    using solver_base<Scalar>::compress_rows_and_cols;

    private:
    using solver_base<Scalar>::use_extra_ritz_vectors_in_the_next_basis;

    void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
    void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);

    Eigen::Index max_mBlocks     = 1;
    Eigen::Index max_sBlocks     = 1;
    Eigen::Index vBlocks         = 0;
    Eigen::Index mBlocks         = 0;
    Eigen::Index sBlocks         = 0;
    Eigen::Index kBlocks         = 0;
    Eigen::Index maxBasisBlocks  = 8;
    Eigen::Index maxRetainBlocks = 1;
    MatrixType   Q_new, HQ_new, H1Q_new, H2Q_new;
    MatrixType   G;

    void selective_orthonormalize(const Eigen::Ref<const MatrixType> X,            // (N, xcols)
                                  Eigen::Ref<MatrixType>             Y,            // (N, ycols)
                                  RealScalar                         breakdownTol, // The smallest allowed norm
                                  VectorIdxT                        &mask          // block norm mask, size = n_blocks = ycols / blockWidth
    );

    void make_new_Q_block(fMultP_t fMultP);

    public:
    bool inject_randomness = false;
    void build() final;
    void build(MatrixType &Q, MatrixType &HQ, const MatrixType &Q_new, const MatrixType &HQ_new);
    void build(MatrixType &Q, MatrixType &H1Q, MatrixType &H2Q, const MatrixType &Q_new, const MatrixType &H1Q_new, const MatrixType &H2Q_new);
    void set_maxExtraRitzHistory(Eigen::Index m);
    void set_maxRitzResidualHistory(Eigen::Index s);
    void set_maxBasisBlocks(Eigen::Index bb);
    void set_maxRetainBlocks(Eigen::Index rb);
};
