#pragma once
#include "../enums.h"
#include "math/float.h"
#include <array>
#include <complex>
#include <Eigen/Cholesky>
// #include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <Eigen/Sparse>
// #include <Eigen/SparseCholesky>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace tid {
    class ur;
}
template<typename Scalar> class MpoSite;
template<typename Scalar> struct env_pair;
template<typename Scalar> struct InvMatVecCfg;
struct primme_params;

template<typename Scalar_>
class MatVecMPOS {
    // static_assert(std::is_same_v<Scalar, fp64> or std::is_same_v<Scalar, cx64>);

    public:
    using Scalar     = Scalar_;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    using CplxScalar = std::complex<RealScalar>;
    using T32        = std::conditional_t<std::is_same_v<Scalar, fp64>, float, std::complex<float>>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using BlockType  = Eigen::Block<MatrixType, Eigen::Dynamic, Eigen::Dynamic, true>; // contiguous block
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SparseType = Eigen::SparseMatrix<Scalar>;
    using MatrixRowM = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using SparseRowM = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using MatrixT32  = Eigen::Matrix<T32, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr static bool         can_shift_invert = true;
    constexpr static bool         can_shift        = true;
    constexpr static bool         can_precondition = true;
    constexpr static eig::Storage storage          = eig::Storage::MPS;
    eig::Factorization            factorization    = eig::Factorization::NONE;
    eig::Preconditioner           preconditioner   = eig::Preconditioner::NONE;

    private:
    bool fullsystem = false;

    std::vector<Eigen::Tensor<Scalar, 4>> mpos_A, mpos_B, mpos_A_shf, mpos_B_shf;
    Eigen::Tensor<Scalar, 3>              envL_A, envR_A, envL_B, envR_B;
    void                                  init_mpos_A();
    void                                  init_mpos_B();

    std::array<long, 3>   shape_mps;
    long                  size_mps;
    std::vector<long>     spindims;
    eig::Form             form            = eig::Form::SYMM;
    eig::Side             side            = eig::Side::R;
    std::optional<Scalar> jcbShift        = std::nullopt;
    long                  jcbMaxBlockSize = 1l;             // Maximum Jacobi block size. The default is 1, which defaults to the diagonal preconditioner
    VectorType            jcbDiagA, jcbDiagB;               // The diagonals of matrices A and B for block jacobi preconditioning (for jcbMaxBlockSize == 1)
    VectorType            invJcbDiagonal;                   // The inverted diagonals used when jcBMaxBlockSize == 1
    mutable VectorType    invJcbDiagB;                      // Userd with spectra
    std::vector<std::pair<long, SparseType>> sInvJcbBlocks; // inverted blocks for the block Jacobi preconditioner stored as sparse matrices
    std::vector<std::pair<long, MatrixType>> dInvJcbBlocks; // inverted blocks for the block Jacobi preconditioner stored as dense matrices
    // std::vector<std::pair<long, MatrixType>> dJcbBlocksA;   // the blocks for the Jacobi preconditioner stored as dense matrices
    // std::vector<std::pair<long, MatrixType>> dJcbBlocksB;   // the blocks for the Jacobi preconditioner stored as dense matrices

    using LLTType = Eigen::LLT<MatrixType, Eigen::Lower>;
    std::vector<std::tuple<long, std::unique_ptr<LLTType>>> lltJcbBlocks; // Solvers for the block Jacobi preconditioner

    using LUType = Eigen::PartialPivLU<MatrixType>;
    std::vector<std::tuple<long, std::unique_ptr<LUType>>> luJcbBlocks; // Solvers for the block Jacobi preconditioner

    using LDLTType = Eigen::LDLT<MatrixType, Eigen::Lower>;
    std::vector<std::tuple<long, std::unique_ptr<LDLTType>>> ldltJcbBlocks; // Solvers for the block Jacobi preconditioner

    // using BICGType = Eigen::BiCGSTAB<SparseRowM, Eigen::IncompleteLUT<Scalar>>;
    // std::vector<std::tuple<long, std::unique_ptr<SparseRowM>, std::unique_ptr<BICGType>>> bicgstabJcbBlocks; // Solvers for the block Jacobi preconditioner

    // using CGType = Eigen::ConjugateGradient<SparseType, Eigen::Lower | Eigen::Upper, Eigen::SimplicialLLT<SparseType>>;
    // using CGType = Eigen::ConjugateGradient<SparseType, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<Scalar, Eigen::Lower | Eigen::Upper>>;
    // std::vector<std::tuple<long, std::unique_ptr<SparseType>, std::unique_ptr<CGType>>> cgJcbBlocks; // Solvers for the block Jacobi preconditioner

    Eigen::LLT<MatrixType>          llt;  // Stores the llt matrix factorization on shift-invert
    Eigen::PartialPivLU<MatrixType> lu;   // Stores the lu matrix factorization on shift-invert
    Eigen::LDLT<MatrixType>         ldlt; // Stores the ldlt matrix factorization on shift-invert

    SparseType                      sparseMatrix;
    VectorType                      solverGuess;
    [[nodiscard]] std::vector<long> get_k_smallest(const VectorType &vec, size_t k) const;
    [[nodiscard]] std::vector<long> get_k_largest(const VectorType &vec, size_t k) const;

    [[nodiscard]] std::vector<long> get_offset(long x, long rank, long size) const;

    // template<size_t rank>
    // constexpr std::array<long, rank> get_offset(long flatindex, const std::array<long, rank> &dimensions);

    [[nodiscard]] std::vector<long> get_offset(long flatindex, size_t rank, const std::vector<long> &dimensions) const;
    template<size_t rank>
    [[nodiscard]] constexpr std::array<long, rank> get_offset(long flatindex, const std::array<long, rank> &dimensions) const;

    // template<typename TI>
    // TI HammingDist(TI x, TI y);

    [[nodiscard]] long round_dn(long num, long multiple) const;
    [[nodiscard]] long round_up(long num, long multiple) const;

    template<auto rank>
    [[nodiscard]] constexpr long ravel_multi_index(const std::array<long, rank> &multi_index, const std::array<long, rank> &dimensions,
                                                   char order = 'F') const noexcept;

    template<auto rank>
    [[nodiscard]] constexpr std::array<long, rank> get_extent(long N, const std::array<long, rank> &dimensions,
                                                              const std::array<long, rank> &offsets = {0}) const;

    template<auto rank>
    [[nodiscard]] constexpr std::array<long, rank> get_extent(const std::array<long, rank> &I0, const std::array<long, rank> &IN,
                                                              const std::array<long, rank> &dimensions) const;

    Scalar     get_matrix_element(long I, long J, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                                  const Eigen::Tensor<Scalar, 3> &ENVR) const;
    VectorType get_diagonal_new(long offset, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                                const Eigen::Tensor<Scalar, 3> &ENVR) const;
    MatrixType get_diagonal_block(long offset, long extent, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                                  const Eigen::Tensor<Scalar, 3> &ENVR) const;
    MatrixType get_diagonal_block(long offset, long extent, Scalar shift, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_A,
                                  const Eigen::Tensor<Scalar, 3> &ENVL_A, const Eigen::Tensor<Scalar, 3> &ENVR_A,
                                  const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS_B, const Eigen::Tensor<Scalar, 3> &ENVL_B,
                                  const Eigen::Tensor<Scalar, 3> &ENVR_B) const;

    // VectorType get_diagonal_old(long offset) const;
    VectorType get_row(long row_idx, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                       const Eigen::Tensor<Scalar, 3> &ENVR) const;
    VectorType get_col(long col_idx, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                       const Eigen::Tensor<Scalar, 3> &ENVR) const;
    VectorType get_diagonal(long offset, const std::vector<Eigen::Tensor<Scalar, 4>> &MPOS, const Eigen::Tensor<Scalar, 3> &ENVL,
                            const Eigen::Tensor<Scalar, 3> &ENVR) const;
    void       thomas(long rows, Scalar *x, Scalar *const dl, Scalar *const dm, Scalar *const du);
    void       thomas2(long rows, Scalar *x, Scalar *const dl, Scalar *const dm, Scalar *const du);
    // void                             thomas(const long rows, const VectorType &x, const VectorType &dl, const VectorType &dm, const VectorType &du);

    // Shift stuff
    CplxScalar   sigma         = CplxScalar(0.0, 0.0); // The shift
    bool         readyShift    = false;                // Flag to make sure the shift has occurred
    bool         readyFactorOp = false;                // Flag to check if factorization has occurred
    mutable bool readyCalcPc   = false;
    mutable bool lockCalcPc    = false;

    public:
    MatVecMPOS() = default;
    template<typename A, typename EnvType>
    MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<A>>> &mpos, /*!< The Hamiltonian MPO's  */
               const env_pair<const EnvType &>                             &envs  /*!< The left and right environments.  */
    );
    template<typename A, typename EnvTypeA, typename EnvTypeB>
    MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite<A>>> &mpos, /*!< The Hamiltonian MPO's  */
               const env_pair<const EnvTypeA &>                            &enva, /*!< The left and right environments.  */
               const env_pair<const EnvTypeB &>                            &envb  /*!< The left and right environments.  */
    );

    // Functions used in Arpack++ solver
    [[nodiscard]] int rows() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */
    [[nodiscard]] int cols() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */

    void FactorOP();                                //  Factorizes (A-sigma*I) (or finds its diagonal elements)
    void MultOPv(Scalar *mps_in_, Scalar *mps_out); //  Applies the preconditioner as the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err); //  Applies the preconditioner
    void MultAx(Scalar *mps_in_, Scalar *mps_out_);             //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultAx(const Scalar *mps_in_, Scalar *mps_out_) const; //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err) const;
    void MultBx(Scalar *mps_in_, Scalar *mps_out_) const; //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultBx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err) const;
    void MultBx(const Scalar *mps_in_, Scalar *mps_out_) const; //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultInvBx(const Scalar *mps_in_, Scalar *mps_out_, long maxiters = 200,
                   RealScalar tolerance = 2e-3f) const;             //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void perform_op(const Scalar *mps_in_, Scalar *mps_out_) const; //  Computes the matrix-vector multiplication x_out <- A*x_in.

    Eigen::Tensor<Scalar, 3> operator*(const Eigen::Tensor<Scalar, 3> &x) const;
    Eigen::Tensor<Scalar, 1> operator*(const Eigen::Tensor<Scalar, 1> &x) const;
    VectorType               operator*(const VectorType &x) const;
    MatrixType               MultAX(const Eigen::Ref<const MatrixType> &X) const;
    VectorType               MultAx(const Eigen::Ref<const VectorType> &x) const;

    void       CalcPc(Scalar shift = 0.0);                                         // Calculates the diagonal or tridiagonal part of A
    void       MultPc(const Scalar *mps_in_, Scalar *mps_out, Scalar shift = 0.0); // Applies the preconditioner
    void       MultPc(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err); // Applies the preconditioner
    MatrixType MultPX(const Eigen::Ref<const MatrixType> &X); //  Applies the preconditioner onto many columns in X

    // Various utility functions
    mutable long num_mv = 0;
    mutable long num_op = 0;
    mutable long num_pc = 0;
    void         print() const;
    void         reset();
    void         set_shift(CplxScalar shift);
    void         set_mode(eig::Form form_);
    void         set_side(eig::Side side_);
    void         set_jcbMaxBlockSize(std::optional<long> jcbSize); // the llt preconditioner bandwidth (default 8) (tridiagonal has bandwidth == 1)

    [[nodiscard]] Scalar                                       get_shift() const;
    [[nodiscard]] eig::Form                                    get_form() const;
    [[nodiscard]] eig::Side                                    get_side() const;
    [[nodiscard]] static constexpr eig::Type                   get_type() { return eig::ScalarToType<Scalar>(); }
    [[nodiscard]] const std::vector<Eigen::Tensor<Scalar, 4>> &get_mpos() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>              &get_envL() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>              &get_envR() const;
    [[nodiscard]] long                                         get_size() const;
    [[nodiscard]] std::array<long, 3>                          get_shape_mps() const;
    [[nodiscard]] std::vector<std::array<long, 4>>             get_shape_mpo() const;
    [[nodiscard]] std::array<long, 3>                          get_shape_envL() const;
    [[nodiscard]] std::array<long, 3>                          get_shape_envR() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                     get_tensor() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                     get_tensor_shf() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                     get_tensor_ene() const;
    [[nodiscard]] MatrixType                                   get_matrix() const;
    [[nodiscard]] MatrixType                                   get_matrix_shf() const;
    [[nodiscard]] MatrixType                                   get_matrix_ene() const;
    [[nodiscard]] SparseType                                   get_sparse_matrix() const;
    [[nodiscard]] double                                       get_sparsity() const;
    [[nodiscard]] long                                         get_non_zeros() const;
    [[nodiscard]] long                                         get_jcbMaxBlockSize() const;
    [[nodiscard]] bool                                         isReadyShift() const;
    [[nodiscard]] bool                                         isReadyFactorOp() const;

    // Timers
    std::unique_ptr<tid::ur> t_factorOP; // Factorization time
    std::unique_ptr<tid::ur> t_genMat;
    std::unique_ptr<tid::ur> t_multOPv;
    std::unique_ptr<tid::ur> t_multPc; // Preconditioner time
    std::unique_ptr<tid::ur> t_multAx; // Matvec time
};
