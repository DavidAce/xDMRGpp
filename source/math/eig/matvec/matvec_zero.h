#pragma once
#include "../enums.h"
#include "math/float.h"
#include <array>
#include <complex>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace tid {
    class ur;
}
template<typename Scalar> class MpoSite;
template<typename Scalar> class MpsSite;

template<typename T>
struct env_pair;

struct primme_params;

template<typename Scalar_>
class MatVecZero {

    public:
    using T      = Scalar_;
    using Scalar = Scalar_;
    static_assert(std::is_same_v<T, fp64> or std::is_same_v<T, cx64>);

    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    constexpr static bool               can_shift_invert = false;
    constexpr static bool               can_shift        = false;
    constexpr static bool               can_precondition = false;
    constexpr static eig::Storage       storage          = eig::Storage::MPS;
    constexpr static eig::Factorization factorization    = eig::Factorization::NONE;

    private:
    std::vector<Eigen::Tensor<T, 4>> mpos, mpos_shf;
    Eigen::Tensor<T, 3>              envL;
    Eigen::Tensor<T, 3>              envR;
    std::array<long, 3>              shape_mps;
    std::array<long, 2>              shape_bond;
    long                             size_mps, size_bond;
    eig::Form                        form = eig::Form::SYMM;
    eig::Side                        side = eig::Side::R;

    // Shift stuff
    std::complex<double>  sigma         = cx64(0.0, 0.0); // The shift
    bool                  readyShift    = false;          // Flag to make sure the shift has occurred
    constexpr static bool readyFactorOp = false;          // Flag to check if factorization has occurred

    public:
    MatVecZero() = default;
    template<typename EnvType>
    MatVecZero(const std::vector<std::reference_wrapper<const MpsSite<Scalar>>> &mpss_, /*!< The MPS sites  */
               const std::vector<std::reference_wrapper<const MpoSite<Scalar>>> &mpos,  /*!< The Hamiltonian MPO's  */
               const env_pair<const EnvType &>                          &envs   /*!< The left and right environments.  */
    );
    // Functions used in Arpack++ solver
    [[nodiscard]] int rows() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */
    [[nodiscard]] int cols() const; /*!< Linear size\f$d^2 \times \chi_L \times \chi_R \f$  */

    void FactorOP();                      //  Would normally factor (A-sigma*I) into PLU --> here it does nothing
    void MultOPv(T *mps_in_, T *mps_out); //  Computes the matrix-vector product x_out <- inv(A-sigma*I)*x_in.
    void MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);
    void MultAx(T *bond_in_, T *bond_out_); //  Computes the matrix-vector multiplication x_out <- A*x_in.
    void MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, primme_params *primme, int *err);

    // Various utility functions
    long num_mv = 0;
    long num_op = 0;
    void print() const;
    void reset();
    void set_shift(std::complex<double> shift);
    void set_mode(eig::Form form_);
    void set_side(eig::Side side_);

    [[nodiscard]] T                                       get_shift() const;
    [[nodiscard]] eig::Form                               get_form() const;
    [[nodiscard]] eig::Side                               get_side() const;
    [[nodiscard]] eig::Type                               get_type() const;
    [[nodiscard]] const std::vector<Eigen::Tensor<T, 4>> &get_mpos() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>         &get_envL() const;
    [[nodiscard]] const Eigen::Tensor<Scalar, 3>         &get_envR() const;
    [[nodiscard]] long                                    get_size_mps() const;
    [[nodiscard]] long                                    get_size_bond() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_mps() const;
    [[nodiscard]] std::array<long, 2>                     get_shape_bond() const;
    [[nodiscard]] std::vector<std::array<long, 4>>        get_shape_mpo() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_envL() const;
    [[nodiscard]] std::array<long, 3>                     get_shape_envR() const;
    [[nodiscard]] Eigen::Tensor<Scalar, 4>                get_tensor() const;
    [[nodiscard]] MatrixType                              get_matrix() const;

    [[nodiscard]] bool isReadyShift() const;
    [[nodiscard]] bool isReadyFactorOp() const;

    // Timers
    std::unique_ptr<tid::ur> t_factorOP; // Factorization time
    std::unique_ptr<tid::ur> t_genMat;
    std::unique_ptr<tid::ur> t_multOPv;
    std::unique_ptr<tid::ur> t_multPc; // Preconditioner time
    std::unique_ptr<tid::ur> t_multAx; // Matvec time
};
