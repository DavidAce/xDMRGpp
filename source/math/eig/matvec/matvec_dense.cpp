#include "matvec_dense.h"
#include "math/cast.h"
#include "math/eig/log.h"
#include "math/float.h"
#include "tid/tid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#define profile_matrix_product_dense 1

// Function definitions
template<typename Scalar>
using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar>
using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using VectorTypeT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

namespace dense_lu {
    std::optional<Eigen::PartialPivLU<MatrixType<fp64>>>               lu_real;
    std::optional<Eigen::PartialPivLU<MatrixType<cx64>>> lu_cplx;
    void                                                                 reset() {
        lu_real.reset();
        lu_cplx.reset();
    }
    template<typename Scalar>
    void init() {
        if constexpr(std::is_same_v<Scalar, fp64>) { dense_lu::lu_real = Eigen::PartialPivLU<MatrixType<Scalar>>(); }
        if constexpr(std::is_same_v<Scalar, cx64>) dense_lu::lu_cplx = Eigen::PartialPivLU<MatrixType<Scalar>>();
    }

}

template<typename Scalar>
MatVecDense<Scalar>::~MatVecDense() {
    dense_lu::reset();
}

// Pointer to data constructor, copies the matrix into an init Eigen matrix.
template<typename Scalar>
MatVecDense<Scalar>::MatVecDense(const Scalar *const A_, const long L_, const bool copy_data, const eig::Form form_, const eig::Side side_)
    : A_ptr(A_), L(L_), form(form_), side(side_) {
    if(copy_data) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
    dense_lu::init<Scalar>();
    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}

template<typename Scalar>
void MatVecDense<Scalar>::FactorOP()

/*  Partial pivot LU decomposition
 *  Factors P(A-sigma*I) = LU
 */
{
    auto token = t_factorOP->tic_token();
    if(readyFactorOp) return; // happens only once
    if(not readyShift) throw std::runtime_error("Cannot FactorOP: Shift value sigma has not been set.");
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
    if constexpr(std::is_same_v<Scalar, fp64>) { dense_lu::lu_real.value().compute(A_matrix); }
    if constexpr(std::is_same_v<Scalar, cx64>) { dense_lu::lu_cplx.value().compute(A_matrix); }
    readyFactorOp = true;
}

template<typename Scalar>
void MatVecDense<Scalar>::MultOPv(Scalar *x_in_ptr, Scalar *x_out_ptr) {
    assert(readyFactorOp and "FactorOp() has not been run yet.");
    auto t_token = t_multOPv->tic_token();
    switch(side) {
        case eig::Side::R: {
            Eigen::Map<VectorType<Scalar>> x_in(x_in_ptr, L);
            Eigen::Map<VectorType<Scalar>> x_out(x_out_ptr, L);
            if constexpr(std::is_same_v<Scalar, fp64>) x_out.noalias() = dense_lu::lu_real.value().solve(x_in);
            if constexpr(std::is_same_v<Scalar, cx64>) x_out.noalias() = dense_lu::lu_cplx.value().solve(x_in);
            break;
        }
        case eig::Side::L: {
            Eigen::Map<VectorTypeT<Scalar>> x_in(x_in_ptr, L);
            Eigen::Map<VectorTypeT<Scalar>> x_out(x_out_ptr, L);
            if constexpr(std::is_same_v<Scalar, fp64>) x_out.noalias() = x_in * dense_lu::lu_real.value().inverse();
            if constexpr(std::is_same_v<Scalar, cx64>) x_out.noalias() = x_in * dense_lu::lu_cplx.value().inverse();
            break;
        }
        case eig::Side::LR: {
            throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
        }
    }
    num_op++;
}

template<typename T>
void MatVecDense<T>::MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *err) {
    if(not readyFactorOp) throw std::logic_error("FactorOp() has not been run yet.");
    auto t_token = t_multOPv->tic_token();
    switch(side) {
        case eig::Side::R: {
            for(int i = 0; i < *blockSize; i++) {
                T                             *x_in_ptr  = static_cast<T *>(x) + *ldx * i;
                T                             *x_out_ptr = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType<Scalar>> x_in(x_in_ptr, L);
                Eigen::Map<VectorType<Scalar>> x_out(x_out_ptr, L);
                if constexpr(std::is_same_v<Scalar, fp64>) x_out.noalias() = dense_lu::lu_real.value().solve(x_in);
                if constexpr(std::is_same_v<Scalar, cx64>) x_out.noalias() = dense_lu::lu_cplx.value().solve(x_in);
                num_op++;
            }

            break;
        }
        case eig::Side::L: {
            for(int i = 0; i < *blockSize; i++) {
                T                             *x_in_ptr  = static_cast<T *>(x) + *ldx * i;
                T                             *x_out_ptr = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType<Scalar>> x_in(x_in_ptr, L);
                Eigen::Map<VectorType<Scalar>> x_out(x_out_ptr, L);
                if constexpr(std::is_same_v<Scalar, fp64>) x_out.noalias() = x_in * dense_lu::lu_real.value().inverse();
                if constexpr(std::is_same_v<Scalar, cx64>) x_out.noalias() = x_in * dense_lu::lu_cplx.value().inverse();
                num_op++;
            }
            break;
        }
        case eig::Side::LR: {
            throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
        }
    }
    *err = 0;
}

template<typename Scalar>
void MatVecDense<Scalar>::MultAx(Scalar *x_in, Scalar *x_out) {
    auto                                 t_token = t_multAx->tic_token();
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    Eigen::Map<VectorType<Scalar>> x_vec_in(x_in, L);
                    Eigen::Map<VectorType<Scalar>> x_vec_out(x_out, L);
                    x_vec_out.noalias() = A_matrix * x_vec_in;
                    break;
                }
                case eig::Side::L: {
                    Eigen::Map<VectorTypeT<Scalar>> x_vec_in(x_in, L);
                    Eigen::Map<VectorTypeT<Scalar>> x_vec_out(x_out, L);
                    x_vec_out.noalias() = x_vec_in * A_matrix;
                    break;
                }
                case eig::Side::LR: {
                    throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
                }
            }
            break;
        case eig::Form::SYMM: {
            Eigen::Map<VectorType<Scalar>> x_vec_in(x_in, L);
            Eigen::Map<VectorType<Scalar>> x_vec_out(x_out, L);
            x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Lower>() * x_vec_in;
            break;
        }
    }
    num_mv++;
}

template<typename T>
void MatVecDense<T>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *err) {
    auto                                 t_token = t_multAx->tic_token();
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    for(int i = 0; i < *blockSize; i++) {
                        T                             *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                             *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorType<Scalar>> x_vec_in(x_in, L);
                        Eigen::Map<VectorType<Scalar>> x_vec_out(x_out, L);
                        x_vec_out.noalias() = A_matrix * x_vec_in;
                        num_mv++;
                    }
                    break;
                }
                case eig::Side::L: {
                    for(int i = 0; i < *blockSize; i++) {
                        T                             *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                             *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorType<Scalar>> x_vec_in(x_in, L);
                        Eigen::Map<VectorType<Scalar>> x_vec_out(x_out, L);
                        x_vec_out.noalias() = x_vec_in * A_matrix;
                        num_mv++;
                    }
                    break;
                }
                case eig::Side::LR: {
                    throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
                }
            }
            break;
        case eig::Form::SYMM: {
            for(int i = 0; i < *blockSize; i++) {
                T                             *x_in  = static_cast<T *>(x) + *ldx * i;
                T                             *x_out = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType<Scalar>> x_vec_in(x_in, L);
                Eigen::Map<VectorType<Scalar>> x_vec_out(x_out, L);
                x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Lower>() * x_vec_in;
                num_mv++;
            }
            break;
        }
    }
    *err = 0;
}

template<typename Scalar>
void MatVecDense<Scalar>::print() const {
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
}

template<typename Scalar>
void MatVecDense<Scalar>::set_shift(std::complex<double> sigma_) {
    if(readyShift) return;
    if(sigma == sigma_) return;
    eig::log->trace("Setting shift = {:.16f} + i{:.16f}", std::real(sigma), std::imag(sigma));
    sigma = sigma_;
    if(A_stl.empty()) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
    Eigen::Map<MatrixType<Scalar>> A_matrix(A_stl.data(), L, L);
    if constexpr(std::is_same_v<Scalar, fp64>) A_matrix -= Eigen::MatrixXd::Identity(L, L) * std::real(sigma);
    if constexpr(std::is_same_v<Scalar, cx64>) A_matrix -= Eigen::MatrixXd::Identity(L, L) * sigma;
    readyShift = true;
}

template<typename Scalar>
void MatVecDense<Scalar>::set_mode(const eig::Form form_) {
    form = form_;
}
template<typename Scalar>
void MatVecDense<Scalar>::set_side(const eig::Side side_) {
    side = side_;
}
template<typename Scalar>
eig::Form MatVecDense<Scalar>::get_form() const {
    return form;
}
template<typename Scalar>
eig::Side MatVecDense<Scalar>::get_side() const {
    return side;
}
template<typename Scalar>
eig::Type MatVecDense<Scalar>::get_type() const {
    if constexpr(std::is_same_v<Scalar, fp64>)
        return eig::Type::FP64;
    else if constexpr(std::is_same_v<Scalar, cx64>)
        return eig::Type::CX64;
    else
        throw std::runtime_error("Unsupported type");
}

// Explicit instantiations

template class MatVecDense<fp64>;
template class MatVecDense<cx64>;
