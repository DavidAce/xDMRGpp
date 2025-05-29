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
#include "StopReason.h"
#include <Eigen/Eigenvalues>

template<typename Scalar>
class GD : public SolverBase<Scalar> {
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
    using SolverBase<Scalar>::status;
    using SolverBase<Scalar>::N;
    using SolverBase<Scalar>::mps_size;
    using SolverBase<Scalar>::mps_shape;
    using SolverBase<Scalar>::nev;
    using SolverBase<Scalar>::ncv;
    using SolverBase<Scalar>::b;
    using SolverBase<Scalar>::qBlocks;
    using SolverBase<Scalar>::algo;
    using SolverBase<Scalar>::ritz;
    using SolverBase<Scalar>::H1;
    using SolverBase<Scalar>::H2;
    using SolverBase<Scalar>::T;
    using SolverBase<Scalar>::A;
    using SolverBase<Scalar>::B;
    using SolverBase<Scalar>::W;
    using SolverBase<Scalar>::Q;
    using SolverBase<Scalar>::M;
    using SolverBase<Scalar>::HQ;
    // using SolverBase<Scalar>::HQ_cur;
    // using SolverBase<Scalar>::get_HQ;
    // using SolverBase<Scalar>::get_HQ_cur;
    // using SolverBase<Scalar>::unset_HQ;
    // using SolverBase<Scalar>::unset_HQ_cur;
    using SolverBase<Scalar>::V;
    using SolverBase<Scalar>::HV;
    using SolverBase<Scalar>::V_prev;
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
    using SolverBase<Scalar>::assert_allfinite;
    using SolverBase<Scalar>::assert_orthonormal;
    using SolverBase<Scalar>::assert_orthogonal;
    using SolverBase<Scalar>::chebyshevFilter;
    using SolverBase<Scalar>::qr_and_chebyshevFilter;
    using SolverBase<Scalar>::orthonormalize;
    using SolverBase<Scalar>::compress_cols;
    using SolverBase<Scalar>::compress_rows_and_cols;

    private:
    using SolverBase<Scalar>::use_extra_ritz_vectors_in_the_next_basis;

    void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
    void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);

    Eigen::Index max_wBlocks     = 1;
    Eigen::Index max_mBlocks     = 1;
    Eigen::Index max_sBlocks     = 1;
    Eigen::Index vBlocks         = 0;
    Eigen::Index wBlocks         = 0;
    Eigen::Index mBlocks         = 0;
    Eigen::Index rBlocks         = 0;
    Eigen::Index sBlocks         = 0;
    Eigen::Index kBlocks         = 0;
    Eigen::Index maxBasisBlocks  = 8;
    Eigen::Index maxRetainBlocks = 1;
    MatrixType   G;
    MatrixType   Q_res_i; // Enrichment created on iteration i

    MatrixType Q_enr;                                                                                 // Store all Qenr_i
    MatrixType HQ_enr;                                                                                // Store all Qenr_i
    void       delete_blocks_from_left_until_orthogonal(const Eigen::Ref<const MatrixType> X,         // (N, xcols)
                                                        MatrixType                        &Y,         // (N, ycols)
                                                        MatrixType                        &HY,        // (N, ycols)
                                                        Eigen::Index                       maxBlocks, // Keep this many blocks at most
                                                        RealScalar                         threshold  // Allow this much overlap between V and Q_enr
          );
    void       selective_orthonormalize(const Eigen::Ref<const MatrixType> X,            // (N, xcols)
                                        Eigen::Ref<MatrixType>             Y,            // (N, ycols)
                                        RealScalar                         breakdownTol, // The smallest allowed norm
                                        VectorIdxT                        &mask          // block norm mask, size = n_blocks = ycols / blockWidth
          );

    void build_Q_res_i();

    MatrixType get_wBlock();
    MatrixType get_mBlock();
    MatrixType get_sBlock();
    MatrixType get_rBlock();

    public:
    bool inject_randomness = false;
    void build() final;
    void diagonalizeT() final;
    void extractRitzVectors() final;
    void extractResidualNorms() final;
    void set_maxLanczosResidualHistory(Eigen::Index k);
    void set_maxExtraRitzHistory(Eigen::Index m);
    void set_maxRitzResidualHistory(Eigen::Index s);
    void set_maxBasisBlocks(Eigen::Index bb);
    void set_maxRetainBlocks(Eigen::Index rb);
};
