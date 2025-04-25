#pragma once
#include "../enums.h"
#include "math/cast.h"
#include <complex>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <memory>
#include <vector>

namespace tid {
    class ur;
}

struct primme_params;

template<typename T, bool sparseLU = false>
class MatVecSparse {
    public:
    using Scalar                                   = T;
    using Real                                     = typename Eigen::NumTraits<Scalar>::Real;
    using Cplx                                     = std::complex<Real>;
    using MatrixType                               = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType                               = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorTypeT                              = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using SparseMatrixType                         = Eigen::SparseMatrix<Scalar>;
    using MatrixLUType                             = std::conditional_t<sparseLU, SparseMatrixType, MatrixType>;
    using LUType                                   = std::conditional_t<sparseLU, Eigen::SparseLU<SparseMatrixType>, Eigen::PartialPivLU<MatrixType>>;
    constexpr static bool         can_shift_invert = true;
    constexpr static bool         can_shift        = true;
    constexpr static bool         can_precondition = false;
    constexpr static eig::Storage storage          = eig::Storage::SPARSE;

    private:
    std::vector<T> A_stl; // The actual matrix. Given matrices will be copied into this one if desired.
    const T       *A_ptr; // A pointer to the matrix to allow optional copying of the matrix.
    const long     L;     // The linear matrix dimension
    eig::Form      form;  // Chooses SYMMETRIC / NONSYMMETRIC mode
    eig::Side      side;  // Chooses whether to find (R)ight or (L)eft eigenvectors

    // Shift and shift-invert mode stuff
    Cplx sigma         = {};    // A possibly complex-valued shift
    bool readyFactorOp = false; // Flag to make sure LU factorization has occurred
    bool readyShift    = false; // Flag to make sure the shift has occurred
    void init_timers();

    Eigen::SparseMatrix<Scalar> A_sparse;
    LUType                      lu;

    public:
    // Pointer to data constructor, copies the matrix into an init Eigen matrix.
    MatVecSparse(const T *A_, long L_, bool copy_data, eig::Form form_ = eig::Form::SYMM, eig::Side side_ = eig::Side::R);

    [[nodiscard]] int rows() const { return safe_cast<int>(L); };
    [[nodiscard]] int cols() const { return safe_cast<int>(L); };
    void              FactorOP();                         //  Factors (A-sigma*I) into PLU
    void              MultOPv(T *x_in_ptr, T *x_out_ptr); //   Computes the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void              MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);
    void              MultAx(T *x_in_ptr, T *x_out_ptr); //   Computes the matrix-vector multiplication x_out <- A*x_in.
    void              MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);
    void              perform_op(const Scalar *mps_in_, Scalar *mps_out_) const; //  Computes the matrix-vector multiplication x_out <- A*x_in.

    // Various utility functions
    mutable long                             num_mv = 0;
    mutable long                             num_op = 0;
    void                                     print() const;
    void                                     set_shift(Cplx sigma_);
    void                                     set_mode(const eig::Form form_);
    void                                     set_side(const eig::Side side_);
    [[nodiscard]] eig::Form                  get_form() const;
    [[nodiscard]] eig::Side                  get_side() const;
    [[nodiscard]] static constexpr eig::Type get_type() { return eig::ScalarToType<Scalar>(); }

    [[nodiscard]] bool isReadyFactorOp() const { return readyFactorOp; }
    [[nodiscard]] bool isReadyShift() const { return readyShift; }

    // Timers
    std::unique_ptr<tid::ur> t_factorOP; // Factorization time
    std::unique_ptr<tid::ur> t_genMat;
    std::unique_ptr<tid::ur> t_multOPv;
    std::unique_ptr<tid::ur> t_multPc; // Preconditioner time
    std::unique_ptr<tid::ur> t_multAx; // Matvec time
};
