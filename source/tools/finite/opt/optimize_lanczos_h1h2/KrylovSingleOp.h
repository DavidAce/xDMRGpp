template<typename Scalar>
struct KrylovSingleOp {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    private:
    struct Status {
        RealScalar               optVal  = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               oldVal  = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               absDiff = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               relDiff = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               initVal = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar               rnorm   = RealScalar{1};
        RealScalar               maxEval = RealScalar{1};
        long                     optIdx  = 0;
        size_t                   iter    = 0;
        long                     numMGS  = 0;
        std::vector<long>        nonZeroCols; // Nonzero Gram Schmidt columns
        std::vector<long>        mixedColOk;  // New states with acceptable norm and eigenvalue
        std::vector<std::string> exitMsg = {};
        SolverExit              exit    = SolverExit::ok;
    };

    public:
    Status                      status = {};
    Eigen::Index                mps_size;
    std::array<Eigen::Index, 3> mps_shape;
    const Eigen::Index          nev = 1; // Number of eigenvalues to find
    const Eigen::Index          ncv = 3; // Krylov dimension, i.e. {V, H1V..., H2V...} ( minimum 2, recommend 3 or more)
    OptAlgo                     algo;
    OptRitz                     ritz;
    MatVecMPOS<Scalar>          H;
    MatrixType                  T;
    MatrixType                  V0;
    MatrixType                  HV;
    VectorType                  alpha, beta;
    VectorReal                  ritz_evals;
    MatrixType                  ritz_evecs;

    const RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();
    RealScalar       tol       = std::numeric_limits<RealScalar>::epsilon() * 10000;
    Eigen::Index     max_iters = 1000;

    KrylovSingleOp(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const VectorType &V0, const auto &mpos, const auto &env)
        : nev(nev),   //
          ncv(ncv),   //
          algo(algo), //
          ritz(ritz), //
          H(mpos, env) {
        assert(ncv >= 2);
        mps_size  = H.get_size();
        mps_shape = H.get_shape_mps();
        assert(mps_size = H.rows());
        HV.resize(H.rows(), ncv);
        V0.resize(H.rows(), ncv);
        V0.col(0) = V0.normalized();
        alpha     = VectorType::Zero(ncv);
        beta      = VectorType::Zero(ncv);
    }

    RealScalar rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon();
    RealScalar absDiffTol      = std::numeric_limits<RealScalar>::epsilon();
    RealScalar relDiffTol      = std::numeric_limits<RealScalar>::epsilon();
    RealScalar rnormTol() const { return tol * status.maxEval; }
};
