#pragma once
#include "io/fmt_custom.h"
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
class solver_lobpcg : public solver_base<Scalar> {
    public:
    using solver_base<Scalar>::solver_base;

    using RealScalar = typename solver_base<Scalar>::RealScalar;
    using MatrixType = typename solver_base<Scalar>::MatrixType;
    using VectorType = typename solver_base<Scalar>::VectorType;
    using VectorReal = typename solver_base<Scalar>::VectorReal;
    using VectorIdxT = typename solver_base<Scalar>::VectorIdxT;
    using OrthMeta   = typename solver_base<Scalar>::OrthMeta;

    using solver_base<Scalar>::eiglog;
    using solver_base<Scalar>::use_preconditioner;
    using solver_base<Scalar>::use_refined_rayleigh_ritz;
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
    using solver_base<Scalar>::S;
    using solver_base<Scalar>::HQ;
    using solver_base<Scalar>::get_wBlock;
    using solver_base<Scalar>::get_mBlock;
    using solver_base<Scalar>::get_sBlock;
    using solver_base<Scalar>::get_rBlock;
    // using solver_base<Scalar>::HQ_cur;
    // using solver_base<Scalar>::get_HQ;
    // using solver_base<Scalar>::get_HQ_cur;
    // using solver_base<Scalar>::unset_HQ;
    // using solver_base<Scalar>::unset_HQ_cur;
    using solver_base<Scalar>::V;
    using solver_base<Scalar>::HV;
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
    using solver_base<Scalar>::MultH;
    using solver_base<Scalar>::MultP;
    using solver_base<Scalar>::chebyshevFilter;
    using solver_base<Scalar>::qr_and_chebyshevFilter;

    using solver_base<Scalar>::block_l2_orthogonalize;
    using solver_base<Scalar>::block_l2_orthonormalize;
    using solver_base<Scalar>::assert_l2_orthonormal;
    using solver_base<Scalar>::assert_l2_orthogonal;
    using solver_base<Scalar>::assert_allFinite;

    using solver_base<Scalar>::compress_col_blocks;
    using solver_base<Scalar>::compress_rows_and_cols;

    private:
    void write_Q3b_solver_lobpcg(Eigen::Index i);
    using solver_base<Scalar>::use_extra_ritz_vectors_in_the_next_basis;

    void                              shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
    void                              roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);
    std::pair<VectorIdxT, VectorIdxT> selective_orthonormalize();

    Eigen::Index max_wBlocks = 1;
    Eigen::Index max_mBlocks = 1;
    Eigen::Index max_sBlocks = 1;
    Eigen::Index wBlocks     = 0;
    Eigen::Index mBlocks     = 0;
    Eigen::Index rBlocks     = 0;
    Eigen::Index sBlocks     = 0;
    MatrixType   G;

    public:
    bool inject_randomness = false;
    void build() final;
    void set_maxLanczosResidualHistory(Eigen::Index k);
    void set_maxExtraRitzHistory(Eigen::Index m);
    void set_maxRitzResidualHistory(Eigen::Index s);
};
