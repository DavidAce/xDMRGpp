#pragma once
#include "io/fmt_custom.h"
#include "StopReason.h"
#include "math/eig/log.h"
#include "math/eig/matvec/matvec_mpos.h"
#include "math/eig/solver.h"
#include "math/eig/view.h"
#include "math/float.h"
#include "math/linalg/matrix/to_string.h"
#include "math/tenx.h"
#include <Eigen/Eigenvalues>
template<typename Scalar>
struct KrylovDualOp {
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    private:
    struct Status {
        VectorReal optVal;
        VectorReal oldVal;
        VectorReal absDiff;
        VectorReal relDiff;
        RealScalar initVal = std::numeric_limits<RealScalar>::quiet_NaN();
        RealScalar maxEval = RealScalar{1};

        std::vector<Eigen::Index> optIdx;
        size_t                    iter   = 0;
        long                      numMGS = 0;
        VectorReal                rNorms;
        std::vector<Eigen::Index> nonZeroCols; // Nonzero Gram Schmidt columns
        Eigen::Index              numZeroRows   = 0;
        std::vector<std::string>  stopMessage       = {};
        StopReason               stopReason          = StopReason::ok;
        OptRitz                   ritz_internal = OptRitz::NONE;
    };

    public:
    Status                      status = {};
    Eigen::Index                N;         /*! The size of the underlying state tensor */
    Eigen::Index                mps_size;  /*! The size of the underlying state tensor in mps representation (equal to N!) */
    std::array<Eigen::Index, 3> mps_shape; /*! The shape of the underlying state tensor in mps representation */
    Eigen::Index                nev = 1;   /*! Number of eigenvalues to find */
    Eigen::Index                ncv = 8;   /*! Krylov dimension, i.e. {V, H1V..., H2V...} ( minimum 2, recommend 3 or more) */
    Eigen::Index                b   = 2;   /*! The block size */
    OptAlgo                     algo;      /*! Selects the current DMRG algorithm */
    OptRitz                     ritz;      /*! Selects the target eigenvalues */
    MatVecMPOS<Scalar>          H1, H2;    /*! The Hamiltonian and Hamiltonian squared operators */
    MatrixType                  T1, T2;    /*! The projections of H1 H2 to the tridiagonal Lanczos basis */
    MatrixType                  T1_debug, T2_debug;    /*! The projections of H1 H2 to the tridiagonal Lanczos basis */
    MatrixType                  A, B, W, Q, Q_prev;
    MatrixType                  V; /*! Holds the current eigenvector approximations. Use this to pass initial guesses */
    VectorReal                  ritz_evals;
    MatrixType                  ritz_evecs;

    const RealScalar eps       = std::numeric_limits<RealScalar>::epsilon();
    RealScalar       tol       = std::numeric_limits<RealScalar>::epsilon() * 10000;
    RealScalar       normTol   = std::numeric_limits<RealScalar>::epsilon() * 10000;
    Eigen::Index     max_iters = 1000;

    KrylovDualOp(Eigen::Index nev, Eigen::Index ncv, OptAlgo algo, OptRitz ritz, const MatrixType &V, const auto &mpos, const auto &enve, const auto &envv)
        : nev(nev),       //
          ncv(ncv),       //
          algo(algo),     //
          ritz(ritz),     //
          H1(mpos, enve), //
          H2(mpos, envv), //
          V(V) {
        ncv       = std::max(nev, ncv);
        b         = std::max(nev, b);
        N         = H1.get_size();
        mps_size  = H1.get_size();
        mps_shape = H1.get_shape_mps();
        status.rNorms.setOnes(nev);
        status.optVal.setOnes(nev);
        status.oldVal.setOnes(nev);
        status.absDiff.setOnes(nev);
        status.relDiff.setOnes(nev);

        assert(mps_size == H1.rows());
        assert(mps_size == H2.rows());
    }

    RealScalar rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon();
    RealScalar absDiffTol      = std::numeric_limits<RealScalar>::epsilon();
    RealScalar relDiffTol      = std::numeric_limits<RealScalar>::epsilon();
    RealScalar rnormTol() const { return tol * status.maxEval; }

    void buildLanczosBlocks() {
        const Eigen::Index N = H1.rows();
        assert(H1.rows() == H1.cols() && "H1 must be square");
        assert(H2.rows() == H2.cols() && "H2 must be square");
        assert(N == H1.rows() && "H1 and H2 must have same dimension");
        assert(N == H2.rows() && "H1 and H2 must have same dimension");
        Eigen::Index m = ncv;

        // Step 0: Construct and orthonormalize initial block V until reaching full rank (s cols)
        assert(V.size() == 0 or N == V.rows());
        Eigen::Index r0 = V.cols();
        if(r0 < b) {
            V.conservativeResize(N, b);
            V.rightCols(b - r0).setRandom();
        }
        Eigen::HouseholderQR<MatrixType> QR0(V);
        r0 = b ; //QR0.rank();
        assert(r0 >= b);
        // Now extract r0 orthonormal vectors. Hopefully r0 >= s,
        // but it is OK if r0 == 1 (highly unlikely if we padded with random columns though).
        Q.setZero(N, b * (m + 1));
        Q.leftCols(b) = QR0.householderQ().setLength(V.cols()) * MatrixType::Identity(N, b); // Extract the b left-most columns

        assert(Q.leftCols(b).allFinite());
        for(Eigen::Index i = 0; i < b; ++i) {
            assert( std::abs(Q.col(i).norm() - RealScalar{1}) < normTol);
        }

        // placeholders for previous block and other matrices
        Q_prev.setZero(N, b);
        T1.setZero(b * m, b * m);
        T2.setZero(b * m, b * m);
        MatrixType Id_Nb = MatrixType::Identity(N, b);


        // Main block-Lanczos loop
        // We enforce the block width b on every iteration
        for(Eigen::Index i = 0; i < m; ++i) {
            const auto Q_cur = Q.middleCols(i * b, b);

            // 1) Apply both operators and form W = [H1*Q_cur, H2*Q_cur]
            W.resize(N, 2 * b);
            assert(W.rows() == N);
            assert(Q_cur.allFinite());
            W.leftCols(b)  = H1.MultAX(Q_cur);
            W.rightCols(b) = H2.MultAX(Q_cur);
            // W.leftCols(b)  = H2.MultAX(Q_cur);
            // W.rightCols(b) = H1.MultAX(Q_cur);
            assert(W.allFinite());
            // eig::log->info("Q_cur (after step 1): \n{}\n", linalg::matrix::to_string(Q_cur, 8));
            // eig::log->info("W (after step 1): \n{}\n", linalg::matrix::to_string(W, 8));

            // 2) Compute projections onto Q_cur and Q_prev
            A = Q_cur.adjoint() * W;            // b×2b
            if(i > 0) B = Q_prev.adjoint() * W; // b×2b

            // 3) Subtract projections
            W.noalias() -= Q_cur * A;
            if(i > 0) W.noalias() -= Q_prev * B;
            // eig::log->info("W (after step 3): \n{}\n", linalg::matrix::to_string(W, 8));

            // 4a) Block DGKS: reorthogonalize W against all previous blocks (first pass)
            for(Eigen::Index j = 0; j <= i; ++j) {
                auto Qj = Q.middleCols(j*b, b);
                W -= Qj * Qj.adjoint() * W;
            }
            // 4b) Block DGKS: second pass to clean up rounding errors
            for(Eigen::Index j = 0; j <= i; ++j) {
                auto Qj = Q.middleCols(j*b, b);
                W -= Qj * Qj.adjoint() * W;
            }


            // 5) Orthonormalize W -> Q_next via ColPivHouseholderQR
            Eigen::HouseholderQR<MatrixType> QR1(W);
            // Eigen::Index                           r1 = 0; //QR1.rank();
            VectorReal Rdiag = QR1.matrixQR().template diagonal().cwiseAbs();
            RealScalar colNormTol   =  eps * std::max(RealScalar{1}, W.norm());
            Eigen::Index r1=0;
            for(Eigen::Index j=0;j<Rdiag.size();++j) {
                if(Rdiag(j) > colNormTol) ++r1;
                else break;
            }
            // Step 5, add the orthonormal columns from Q_next = QR1 to Q, but make sure to project out Q from Q_next to using DGKS
            // In addition, we should make sure that the column norms of Q_next do not drop below a threshold
            auto       Q_next       = Q.middleCols((i + 1) * b, b);
            VectorReal Q_next_norms = VectorReal::Zero(b);
            eig::log->info("colNormTol {:.3e} | r1 = {}, Rdiag() = {::.5e}", colNormTol, r1, fv(Rdiag));

            if(r1 >= b) {
                Q_next = QR1.householderQ().setLength(W.cols()) * Id_Nb; // Extract the left-most b columns
                Q_next_norms = Q_next.colwise().norm();
                eig::log->info("Q_next_norms = {::.5e}", Q_next_norms);

                // 5a) Block DGKS Reorthogonalize Q_next against all previous blocks
                for(Eigen::Index j = 0; j <= i; ++j) {
                    auto Qj = Q.middleCols(j*b, b);
                    Q_next -= Qj * Qj.adjoint() * Q_next;
                }
                // 5b) Block DGKS: second pass to clean up rounding errors
                for(Eigen::Index j = 0; j <= i; ++j) {
                    auto Qj = Q.middleCols(j*b, b);
                    Q_next -= Qj * Qj.adjoint() * Q_next;
                }
                // Normalize
                for(Eigen::Index k = 0; k < b; ++k) {
                    auto colNorm         = Q_next.col(k).norm();
                    Q_next_norms(k) = colNorm;
                    if(colNorm > colNormTol) Q_next.col(k) /= colNorm;
                }


                // Eigen::Index k0 = (i + 1) * b; // Points to the first column in Q_next
                //for(Eigen::Index k = k0; k < k0 + b; ++k) {
                    // 5a) first DGKS pass
                    // for(Eigen::Index j = 0; j < k; ++j) { Q.col(k) -= Q.col(j).dot(Q.col(k)) * Q.col(j); }
                    // 5b) second DGKS pass (to mop up rounding errors)
                    // for(Eigen::Index j = 0; j < k; ++j) { Q.col(k) -= Q.col(j).dot(Q.col(k)) * Q.col(j); }


                //}
            }
            assert(Q_next.allFinite());
            if(r1 < b or Q_next_norms.minCoeff() <= colNormTol) {
                // Happy breakdown, reached the invariant subspace:
                //      Q_next has at least one column whose norm is close to zero
                Eigen::Index doneCols = i * b;
                // shrink Q down to the vectors we actually have
                Q.conservativeResize(N, doneCols);
                // shrink T1/T2 to i×i blocks of size s
                T1.conservativeResize(i * b, i * b);
                T2.conservativeResize(i * b, i * b);

                eig::log->debug("saturated basis");
                status.stopReason |= StopReason::saturated_basis;
                status.stopMessage.emplace_back("saturated basis: exhausted subspace search");
                break;
            }

            // 6) Fill block-tridiagonals: diagonals from A = [A1, A2]; and off-diagonals from B = [B1, B2]
            // A = [A1, A2]
            // We will end up with the following form on each of T1 and T2:
            //
            // T[1|2] =
            //  [ A₀     B₁ᴴ     0       0       ...        0     ]
            //  [ B₁     A₁      B₂ᴴ     0       ...        0     ]
            //  [ 0      B₂      A₂      B₃ᴴ     ...        0     ]
            //  [ 0       0      B₃      A₃      ...        0     ]
            //  [ ...   ...     ...     ...      ...       ...    ]
            //  [ 0       0      0       0       Bₘ         Aₘ    ]
            //
            auto A1                      = A.block(0, 0, b, b);
            auto A2                      = A.block(0, b, b, b);
            // auto A2                      = A.block(0, 0, b, b);
            // auto A1                      = A.block(0, b, b, b);
            T1.block(i * b, i * b, b, b) = A1;
            T2.block(i * b, i * b, b, b) = A2;
            if(i > 0) {
                // B = [B1, B2]
                auto B1                            = B.block(0, 0, b, b);
                auto B2                            = B.block(0, b, b, b);
                // auto B2                            = B.block(0, 0, b, b);
                // auto B1                            = B.block(0, b, b, b);
                T1.block((i - 1) * b, i * b, b, b) = B1;
                T1.block(i * b, (i - 1) * b, b, b) = B1.adjoint();
                T2.block((i - 1) * b, i * b, b, b) = B2;
                T2.block(i * b, (i - 1) * b, b, b) = B2.adjoint();
            }

            // Prepare for the next iteration
            Q_prev = Q_cur;


            // eig::log->info("T1: \n{}\n", linalg::matrix::to_string(T1, 8));
            // eig::log->info("T2: \n{}\n", linalg::matrix::to_string(T2, 8));
        }

        T1_debug = T1;
        T2_debug = T2;

        // eig::log->info("||T1-T1.adjoint()|| = {:.16f}", (T1-T1.adjoint()).norm());
        // eig::log->info("||T2-T2.adjoint()|| = {:.16f}", (T2-T2.adjoint()).norm());

    }

    void diagonalizeLanczosBlocks() {
        // We now have T1 and T2 to do whatever with
        if(T1.rows() == 0 or T2.rows() == 0) return;
        if(status.stopReason != none) return;
        auto numZeroRowsT1 = (T1.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
        auto numZeroRowsT2 = (T2.cwiseAbs().rowwise().maxCoeff().array() <= eps).count();
        status.numZeroRows = std::max({numZeroRowsT1, numZeroRowsT2});

        status.ritz_internal = ritz;
        auto solver          = eig::solver();
        switch(algo) {
            using enum OptAlgo;
            case DMRG: {
                solver.eig<eig::Form::SYMM>(T1.data(), T1.rows(), eig::Vecs::ON);

                // solver = Eigen::SelfAdjointEigenSolver<MatrixType>(T1, Eigen::ComputeEigenvectors);
                // if(solver.info() == Eigen::ComputationInfo::Success) {
                //     ritz_evals = solver.eigenvalues();
                //     ritz_evecs = solver.eigenvectors();
                // } else {
                //     eig::log->info("T1                     : \n{}\n", linalg::matrix::to_string(T1, 8));
                //     eig::log->warn("Diagonalization of T1 exited with info {}", static_cast<int>(solver.info()));
                // }
                //
                // if(ritz_evals.hasNaN()) throw except::runtime_error("found nan eigenvalues");
                break;
            }
            case DMRGX: [[fallthrough]];
            case HYBRID_DMRGX: {
                MatrixType T = T2 - T1 * T1;
                solver.eig<eig::Form::SYMM>(T.data(), T.rows(), eig::Vecs::ON);
                // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(T2 - T1 * T1, Eigen::ComputeEigenvectors);
                // ritz_evals       = solver.eigenvalues();
                // ritz_evecs       = solver.eigenvectors();
                break;
            }
            case XDMRG: {
                // solver.eig<eig::Form::SYMM>(T2.data(), T2.rows(), eig::Vecs::ON);
                auto eigsol = Eigen::SelfAdjointEigenSolver<MatrixType>(T2, Eigen::ComputeEigenvectors);
                ritz_evals       = eigsol.eigenvalues();
                ritz_evecs       = eigsol.eigenvectors();
                break;
            }
            case GDMRG: {
                if(status.numZeroRows == 0) {
                    solver.eig<eig::Form::SYMM>(T1.data(), T2.data(), T1.rows(), eig::Vecs::ON);

                    // auto solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixCT>(
                    // T1.template selfadjointView<Eigen::Lower>(), T2.template selfadjointView<Eigen::Lower>(), Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
                    // ritz_evals = solver.eigenvalues().real();
                    // ritz_evecs = solver.eigenvectors().colwise().normalized();
                } else {
                    MatrixType K = T2 - T1 * T1;
                    eig::log->debug("K                      : \n{}\n", linalg::matrix::to_string(T1, 8));
                    solver.eig<eig::Form::SYMM>(K.data(), K.rows(), eig::Vecs::ON);
                    // auto solver = Eigen::SelfAdjointEigenSolver<MatrixCT>(T2 - T1 * T1, Eigen::ComputeEigenvectors);
                    // ritz_evals       = solver.eigenvalues();
                    // ritz_evecs       = solver.eigenvectors();
                    if(ritz == OptRitz::LM) status.ritz_internal = OptRitz::SM;
                    if(ritz == OptRitz::LR) status.ritz_internal = OptRitz::SM;
                    if(ritz == OptRitz::SM) status.ritz_internal = OptRitz::LM;
                    if(ritz == OptRitz::SR) status.ritz_internal = OptRitz::LR;
                }
                break;
            }
            default: throw except::runtime_error("unhandled algorithm: [{}]", enum2sv(algo));
        }

        // ritz_evals = eig::view::get_eigvals<RealScalar>(solver.result);
        // ritz_evecs = eig::view::get_eigvecs<Scalar>(solver.result); //.colwise().normalized();
    }

    template<typename Comp>
    std::vector<Eigen::Index> getIndices(const VectorType &v, const Eigen::Index k, Comp comp) {
        std::vector<Eigen::Index> idx(static_cast<size_t>(v.size()));
        std::iota(idx.begin(), idx.end(), 0);                             // 1) build an index array [0, 1, 2, …, N-1]
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), comp); // Sort k elements
        return std::vector(idx.begin(), idx.begin() + k);                 // now idx[0..k) are the k sorted indices
    }

    void extractLanczosSolution() {
        if(status.stopReason != StopReason::none) return;
        eig::log->info("ritz_evals             = {}", linalg::matrix::to_string(ritz_evals.transpose(), 16));

            // Check if H2 is PD
        if(ritz_evals.minCoeff() < 0) {

            {
                Eigen::Index tcols = T1_debug.rows();
                MatrixType T1_direct = Q.adjoint() * H1.MultAX(Q);
                // 3) Compare with your assembled T2:
                MatrixType diff = T1_direct.topLeftCorner(tcols,tcols) - T1_debug;
                eig::log->info("T1: \n{}\n", linalg::matrix::to_string(T1_debug, 8));
                eig::log->info("T1_direct: \n{}\n", linalg::matrix::to_string(T1_direct, 8));
                eig::log->info("‖T1_direct – T1‖ = {} ",diff.norm());
            }

            {
                Eigen::Index tcols = T2_debug.rows();
                MatrixType T2_direct = Q.adjoint() * H2.MultAX(Q);
                // 3) Compare with your assembled T2:
                MatrixType diff = T2_direct.topLeftCorner(tcols,tcols) - T2_debug;
                eig::log->info("T2: \n{}\n", linalg::matrix::to_string(T2_debug, 8));
                eig::log->info("T2_direct: \n{}\n", linalg::matrix::to_string(T2_direct, 8));
                eig::log->info("‖T2_direct – T2‖ = {} ",diff.norm());
                {
                    auto eigsol = Eigen::SelfAdjointEigenSolver<MatrixType>(T2_direct.topLeftCorner(tcols,tcols), Eigen::ComputeEigenvectors);
                    eig::log->info("T2_direct evals = {}", linalg::matrix::to_string(eigsol.eigenvalues().transpose(), 16));
                    eig::log->info("T2        evals = {}", linalg::matrix::to_string(ritz_evals.transpose(), 16));
                }
            }

            throw except::runtime_error("Found a negative eigenvalue coming from T2!");

        }
        status.maxEval = static_cast<RealScalar>(ritz_evals.cwiseAbs().maxCoeff());

        // Select eigenvalues
        switch(status.ritz_internal) {
            case OptRitz::SR: status.optIdx = getIndices(ritz_evals, b, std::less<RealScalar>()); break;
            case OptRitz::LR: status.optIdx = getIndices(ritz_evals, b, std::greater<RealScalar>()); break;
            case OptRitz::SM: status.optIdx = getIndices(ritz_evals.cwiseAbs(), b, std::less<RealScalar>()); break;
            case OptRitz::LM: status.optIdx = getIndices(ritz_evals.cwiseAbs(), b, std::greater<RealScalar>()); break;
            case OptRitz::IS: [[fallthrough]];
            case OptRitz::TE: [[fallthrough]];
            case OptRitz::NONE: {
                if(std::isnan(status.initVal))
                    throw except::runtime_error("Ritz [{} ({})] does not work when lanczos.status.initVal is nan", enum2sv(status.ritz_internal),
                                                enum2sv(status.ritz_internal));
                status.optIdx = getIndices((ritz_evals.array() - status.initVal).cwiseAbs(), b, std::less<RealScalar>());
                break;
            }
            default: throw except::runtime_error("unhandled ritz: [{} ({})]", enum2sv(ritz), enum2sv(status.ritz_internal));
        }

        Eigen::Index qcols = std::min(Q.cols(), ritz_evecs.rows());

        // MatrixType G = Q.leftCols(qcols).adjoint() * Q.leftCols(qcols);
        // eig::log->info("Orthogonality check: {} \n G = \n{}\n", (G - MatrixType::Identity(qcols, qcols)).norm(), linalg::matrix::to_string(G, 8));
        V = Q.leftCols(qcols) * ritz_evecs(Eigen::all, status.optIdx); // Now V has b columns mixed according to the selected columns in ritz_evecs

        // Eigenvalues are sorted in ascending order.
        status.oldVal  = status.optVal;
        status.optVal  = ritz_evals(status.optIdx).topRows(nev); // Make sure we only take nev values here. In general, nev <= b
        status.absDiff = (status.optVal - status.oldVal).cwiseAbs();
        status.relDiff = status.absDiff.array() / (RealScalar{0.5} * (status.optVal + status.oldVal).array());
        eig::log->info("optIdx {} : {::.16f} | ritz {} ({})", status.optIdx, fv(status.optVal), enum2sv(ritz), enum2sv(status.ritz_internal));

        // Calculate residual norms
        for(Eigen::Index i = 0; i < nev; ++i) { // Only consider nev rnorms
            if(algo == OptAlgo::DMRG)
                status.rNorms(i) = (H1.MultAX(V.col(i)) - status.optVal(i) * V.col(i)).template lpNorm<Eigen::Infinity>();
            else if(algo == OptAlgo::GDMRG)
                status.rNorms(i) = (H1.MultAX(V.col(i)) - status.optVal(i) * H2.MultAX(V.col(i))).template lpNorm<Eigen::Infinity>();
            else
                status.rNorms(i) = (H2.MultAX(V.col(i)) - status.optVal(i) * V.col(i)).template lpNorm<Eigen::Infinity>();
        }

        if(status.iter >= 1ul) {
            if(status.absDiff.maxCoeff() < absDiffTol and status.iter >= 3) {
                status.stopMessage.emplace_back(fmt::format("saturated: abs diff {::.3e} < tol {:.3e}", fv(status.absDiff), absDiffTol));
                status.stopReason |= StopReason::saturated_absDiffTol;
            }
            if(status.relDiff.maxCoeff() < relDiffTol and status.iter >= 3) {
                status.stopMessage.emplace_back(fmt::format("saturated: rel diff {::.3e} < {:.3e}", fv(status.relDiff), relDiffTol));
                status.stopReason |= StopReason::saturated_relDiffTol;
            }

            // if(status.VColOK.size() == 1) {
            //     status.stopMessage.emplace_back(fmt::format("saturated: only one valid eigenvector"));
            //     status.stopReason |= StopReason::one_valid_eigenvector;
            // }
            //
            // if(status.VColOK.empty()) {
            //     status.stopMessage.emplace_back(fmt::format("mixedColOk is empty"));
            //     status.stopReason |= StopReason::no_valid_eigenvector;
            // }
        }

        if(status.rNorms.maxCoeff() < rnormTol()) {
            status.stopMessage.emplace_back(fmt::format("converged rNorms {::.3e} < tol {:.3e}", fv(status.rNorms), rnormTol()));
            status.stopReason |= StopReason::converged_rnormTol;
        }
        if(status.iter >= std::max<size_t>(1ul, max_iters)) {
            status.stopMessage.emplace_back(fmt::format("iter ({}) >= maxiter ({})", status.iter, max_iters));
            status.stopReason |= StopReason::max_iterations;
        }
        eig::log->info("ncv {} (nnz {}) | rnorms = {::.8e} | optVal = {::.8e} | optIdx = {} | absDiff = {::.8e} | relDiff = {::.8e} | iter = {} | stopMessage = {}",
                       ncv, status.nonZeroCols.size(), fv(status.rNorms), fv(status.optVal), status.optIdx, fv(status.absDiff), fv(status.relDiff), status.iter,
                       status.stopMessage);
        status.iter++;
    }
};
