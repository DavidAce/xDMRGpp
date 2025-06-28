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
class solver_lanczos : solver_base<Scalar> {
    using solver_base<Scalar>::solver_base;

    using RealScalar = typename solver_base<Scalar>::RealScalar;
    using MatrixType = typename solver_base<Scalar>::MatrixType;
    using VectorType = typename solver_base<Scalar>::VectorType;
    using VectorReal = typename solver_base<Scalar>::VectorReal;
    using VectorIdxT = typename solver_base<Scalar>::VectorIdxT;

    using solver_base<Scalar>::eiglog;
    using solver_base<Scalar>::use_preconditioner;
    using solver_base<Scalar>::status;
    using solver_base<Scalar>::N;
    using solver_base<Scalar>::mps_size;
    using solver_base<Scalar>::mps_shape;
    using solver_base<Scalar>::nev;
    using solver_base<Scalar>::ncv;
    using solver_base<Scalar>::b;
    using solver_base<Scalar>::algo;
    using solver_base<Scalar>::ritz;
    using solver_base<Scalar>::H1;
    using solver_base<Scalar>::H2;
    using solver_base<Scalar>::T;
    using solver_base<Scalar>::A;
    using solver_base<Scalar>::B;
    using solver_base<Scalar>::W;
    using solver_base<Scalar>::Q;
    using solver_base<Scalar>::HQ;
    using solver_base<Scalar>::V;
    using solver_base<Scalar>::T_evals;
    using solver_base<Scalar>::T_evecs;
    using solver_base<Scalar>::hhqr;
    using solver_base<Scalar>::eps;
    using solver_base<Scalar>::tol;
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
    using solver_base<Scalar>::MultHX;
    using solver_base<Scalar>::MultPX;

    void write_Q_next_B_DGKS(Eigen::Index i);

    void build() final;
};
