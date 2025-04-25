#pragma once

#include "../enums.h"
#include "math/cast.h"
#include <complex>
#include <Eigen/LU>
#include <memory>
#include <vector>
namespace tid {
    class ur;
}
struct primme_params;

template<typename Scalar_>
class MatVecDense {
    public:
    using Scalar      = Scalar_;
    using Real        = typename Eigen::NumTraits<Scalar>::Real;
    using Cplx        = std::complex<Real>;
    using MatrixType  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType  = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorTypeT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    constexpr static bool         can_shift_invert = true;
    constexpr static bool         can_shift        = true;
    constexpr static bool         can_precondition = false;
    constexpr static eig::Storage storage          = eig::Storage::DENSE;

    private:
    std::vector<Scalar> A_stl; // The actual matrix. Given matrices will be copied into this one if desired
    const Scalar       *A_ptr; // A pointer to the matrix, to allow optional copying of the matrix. Note that PartialPivLU stores LU in A.
    const long          L;     // The linear matrix dimension
    eig::Form           form;  // Chooses SYMMETRIC / NONSYMMETRIC mode
    eig::Side           side;  // Chooses whether to find (R)ight or (L)eft eigenvectors
    // Shift and shift-invert mode stuff
    Cplx                            sigma         = {};    // A possibly complex-valued shift
    bool                            readyFactorOp = false; // Flag to make sure LU factorization has occurred
    bool                            readyShift    = false; // Flag to make sure the shift has occurred
    void                            init_timers();
    MatrixType                      A_dense;
    Eigen::PartialPivLU<MatrixType> lu_dense;

    public:
    // Pointer to data constructor, copies the matrix into an init Eigen matrix.
    MatVecDense(const Scalar *const A_, const long L_, const bool copy_data, const eig::Form form_ = eig::Form::SYMM, const eig::Side side_ = eig::Side::R);

    // Functions used in the Arpack++ solver
    [[nodiscard]] int rows() const { return safe_cast<int>(L); };
    [[nodiscard]] int cols() const { return safe_cast<int>(L); };
    void              FactorOP();                                   //  Factors (A-sigma*I) into PLU
    void              MultOPv(Scalar *x_in_ptr, Scalar *x_out_ptr); //   Computes the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void              MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);
    void              MultAx(Scalar *x_in_ptr, Scalar *x_out_ptr); //   Computes the matrix-vector multiplication x_out <- A*x_in.
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
