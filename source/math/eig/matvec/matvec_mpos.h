#pragma once
#include "../enums.h"
#include "math/float.h"
#include <array>
#include <complex>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace tid {
    class ur;
}
class MpoSite;

template<typename T>
struct env_pair;

struct primme_params;

template<typename T>
class MatVecMPOS {
    static_assert(std::is_same_v<T, fp64> or std::is_same_v<T, cx64>);

    public:
    using Scalar     = T;
    using T32        = std::conditional_t<std::is_same_v<T, fp64>, float, std::complex<float>>;
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SparseType = Eigen::SparseMatrix<T>;
    using MatrixRowM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using SparseRowM = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using MatrixT32  = Eigen::Matrix<T32, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr static bool         can_shift_invert = true;
    constexpr static bool         can_shift        = true;
    constexpr static bool         can_precondition = true;
    constexpr static eig::Storage storage          = eig::Storage::MPS;
    eig::Factorization            factorization    = eig::Factorization::NONE;
    eig::Preconditioner           preconditioner   = eig::Preconditioner::NONE;

    private:
    bool fullsystem = false;

    std::vector<Eigen::Tensor<T, 4>> mpos_A, mpos_B, mpos_A_shf, mpos_B_shf;
    Eigen::Tensor<T, 3>              envL_A, envR_A, envL_B, envR_B;
    std::array<long, 3>              shape_mps;
    long                             size_mps;
    std::vector<long>                spindims;
    eig::Form                        form = eig::Form::SYMM;
    eig::Side                        side = eig::Side::R;
    VectorType                       jcbDiagA, jcbDiagB;      // The diagonals of matrices A and B for block jacobi preconditioning (for jcbMaxBlockSize == 1)
    VectorType                       denseJcbDiagonal;        // The inverted diagonals used when jcBMaxBlockSize == 1
    std::vector<std::pair<long, SparseType>> sparseJcbBlocks; // inverted blocks for the block Jacobi preconditioner stored as sparse matrices
    std::vector<std::pair<long, MatrixType>> denseJcbBlocks;  // inverted blocks for the block Jacobi preconditioner stored as dense matrices

    using BICGType = Eigen::BiCGSTAB<SparseRowM, Eigen::IncompleteLUT<Scalar>>;
    std::vector<std::tuple<long, std::unique_ptr<SparseRowM>, std::unique_ptr<BICGType>>> bicgstabJcbBlocks; // Solvers for the block Jacobi preconditioner

    // using CGType = Eigen::ConjugateGradient<SparseType, Eigen::Lower | Eigen::Upper, Eigen::SimplicialLLT<SparseType>>;
    using CGType = Eigen::ConjugateGradient<SparseType, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<Scalar, Eigen::Lower | Eigen::Upper>>;
    std::vector<std::tuple<long, std::unique_ptr<SparseType>, std::unique_ptr<CGType>>> cgJcbBlocks; // Solvers for the block Jacobi preconditioner

    using LLTType = Eigen::LLT<MatrixType, Eigen::Lower>;
    std::vector<std::tuple<long, std::unique_ptr<LLTType>>> lltJcbBlocks; // Solvers for the block Jacobi preconditioner

    using LDLTType = Eigen::LDLT<MatrixType, Eigen::Lower>;
    std::vector<std::tuple<long, std::unique_ptr<LDLTType>>> ldltJcbBlocks; // Solvers for the block Jacobi preconditioner

    using LUType = Eigen::PartialPivLU<MatrixType>;
    std::vector<std::tuple<long, std::unique_ptr<LUType>>> luJcbBlocks; // Solvers for the block Jacobi preconditioner

    Eigen::LDLT<MatrixType>         ldlt; // Stores the ldlt matrix factorization on shift-invert
    Eigen::LLT<MatrixType>          llt;  // Stores the llt matrix factorization on shift-invert
    Eigen::PartialPivLU<MatrixType> lu;   // Stores the lu matrix factorization on shift-invert

    SparseType        sparseMatrix;
    VectorType        solverGuess;
    std::vector<long> get_k_smallest(const VectorType &vec, size_t k) const;
    std::vector<long> get_k_largest(const VectorType &vec, size_t k) const;

    T get_matrix_element(long I, long J, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL, const Eigen::Tensor<T, 3> &ENVR) const;
    VectorType get_diagonal_new(long offset, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL,
                                const Eigen::Tensor<T, 3> &ENVR) const;
    MatrixType get_diagonal_block_old(long offset, long extent, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL,
                                      const Eigen::Tensor<T, 3> &ENVR) const;
    MatrixType get_diagonal_block(long offset, long extent, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL,
                                  const Eigen::Tensor<T, 3> &ENVR) const;
    MatrixType get_diagonal_block(long offset, long extent, T shift, const std::vector<Eigen::Tensor<T, 4>> &MPOS_A, const Eigen::Tensor<T, 3> &ENVL_A,
                                  const Eigen::Tensor<T, 3> &ENVR_A, const std::vector<Eigen::Tensor<T, 4>> &MPOS_B, const Eigen::Tensor<T, 3> &ENVL_B,
                                  const Eigen::Tensor<T, 3> &ENVR_B) const;
    MatrixType get_diagonal_block_old(long offset, long extent, T shift, const std::vector<Eigen::Tensor<T, 4>> &MPOS_A, const Eigen::Tensor<T, 3> &ENVL_A,
                                      const Eigen::Tensor<T, 3> &ENVR_A, const std::vector<Eigen::Tensor<T, 4>> &MPOS_B, const Eigen::Tensor<T, 3> &ENVL_B,
                                      const Eigen::Tensor<T, 3> &ENVR_B) const;

    // VectorType get_diagonal_old(long offset) const;
    VectorType get_row(long row_idx, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL, const Eigen::Tensor<T, 3> &ENVR) const;
    VectorType get_col(long col_idx, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL, const Eigen::Tensor<T, 3> &ENVR) const;
    VectorType get_diagonal(long offset, const std::vector<Eigen::Tensor<T, 4>> &MPOS, const Eigen::Tensor<T, 3> &ENVL, const Eigen::Tensor<T, 3> &ENVR) const;
    void       thomas(long rows, T *x, T *const dl, T *const dm, T *const du);
    void       thomas2(long rows, T *x, T *const dl, T *const dm, T *const du);
    // void                             thomas(const long rows, const VectorType &x, const VectorType &dl, const VectorType &dm, const VectorType &du);

    // Shift stuff
    std::complex<double> sigma           = cx64(0.0, 0.0); // The shift
    bool                 readyShift      = false;          // Flag to make sure the shift has occurred
    bool                 readyFactorOp   = false;          // Flag to check if factorization has occurred
    bool                 readyCalcPc     = false;
    long                 jcbMaxBlockSize = 1l; // Maximum Jacobi block size. The default is 1, which defaults to the diagonal preconditioner
    public:
    MatVecMPOS() = default;
    template<typename EnvType>
    MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite>> &mpos, /*!< The Hamiltonian MPO's  */
               const env_pair<const EnvType &>                          &envs  /*!< The left and right environments.  */
    );
    template<typename EnvTypeA, typename EnvTypeB>
    MatVecMPOS(const std::vector<std::reference_wrapper<const MpoSite>> &mpos, /*!< The Hamiltonian MPO's  */
               const env_pair<const EnvTypeA &>                         &enva, /*!< The left and right environments.  */
               const env_pair<const EnvTypeB &>                         &envb  /*!< The left and right environments.  */
    );

    // Functions used in Arpack++ solver
    [[nodiscard]] int rows() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */
    [[nodiscard]] int cols() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */

    void FactorOP();                      //  Factorizes (A-sigma*I) (or finds its diagonal elements)
    void MultOPv(T *mps_in_, T *mps_out); //  Applies the preconditioner as the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err); //  Applies the preconditioner
    void MultAx(T *mps_in_, T *mps_out_); //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);
    void MultBx(T *mps_in_, T *mps_out_); //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultBx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);

    void CalcPc(T shift = 0.0);                         //  Calculates the diagonal or tridiagonal part of A
    void MultPc(T *mps_in_, T *mps_out, T shift = 0.0); //  Applies the preconditioner as the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void MultPc(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err); //  Applies the preconditioner

    // Various utility functions
    long num_mv = 0;
    long num_op = 0;
    long num_pc = 0;
    void print() const;
    void reset();
    void set_shift(std::complex<double> shift);
    void set_mode(eig::Form form_);
    void set_side(eig::Side side_);
    void set_jcbMaxBlockSize(std::optional<long> jcbSize); // the llt preconditioner bandwidth (default 8) (tridiagonal has bandwidth == 1)

    [[nodiscard]] T                                       get_shift() const;
    [[nodiscard]] eig::Form                               get_form() const;
    [[nodiscard]] eig::Side                               get_side() const;
    [[nodiscard]] eig::Type                               get_type() const;
    [[nodiscard]] const std::vector<Eigen::Tensor<T, 4>> &get_mpos() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>         &get_envL() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>         &get_envR() const;
    [[nodiscard]] long                                    get_size() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_mps() const;
    [[nodiscard]] std::vector<std::array<long, 4>>        get_shape_mpo() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_envL() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_envR() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                get_tensor() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                get_tensor_shf() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 6>                get_tensor_ene() const;
    [[nodiscard]] MatrixType                              get_matrix() const;
    [[nodiscard]] MatrixType                              get_matrix_shf() const;
    [[nodiscard]] MatrixType                              get_matrix_ene() const;
    [[nodiscard]] SparseType                              get_sparse_matrix() const;
    [[nodiscard]] double                                  get_sparsity() const;
    [[nodiscard]] long                                    get_non_zeros() const;
    [[nodiscard]] long                                    get_jcbMaxBlockSize() const;
    [[nodiscard]] bool                                    isReadyShift() const;
    [[nodiscard]] bool                                    isReadyFactorOp() const;

    // Timers
    std::unique_ptr<tid::ur> t_factorOP; // Factorization time
    std::unique_ptr<tid::ur> t_genMat;
    std::unique_ptr<tid::ur> t_multOPv;
    std::unique_ptr<tid::ur> t_multPc; // Preconditioner time
    std::unique_ptr<tid::ur> t_multAx; // Matvec time
};
