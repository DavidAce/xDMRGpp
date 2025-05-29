#pragma once
#include "../matvec_dense.h"
#include "io/fmt_custom.h"
#include "math/cast.h"
#include "math/eig/log.h"
#include "math/float.h"
#include "tid/tid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <general/sfinae.h>
#define profile_matrix_product_dense 1

// Explicit instantiations


template<typename Scalar>
using MatrixType = typename MatVecDense<Scalar>::MatrixType;
template<typename Scalar>
using VectorType = typename MatVecDense<Scalar>::VectorType;
template<typename Scalar>
using VectorTypeT = typename MatVecDense<Scalar>::VectorTypeT;

// Pointer to data constructor, copies the matrix into an init Eigen matrix.
template<typename Scalar>
MatVecDense<Scalar>::MatVecDense(const Scalar *const A_, const long L_, const bool copy_data, const eig::Form form_, const eig::Side side_)
    : A_ptr(A_), L(L_), form(form_), side(side_) {
    if(copy_data) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
    init_timers();
}
template<typename Scalar>
void MatVecDense<Scalar>::init_timers() {
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
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
    lu_dense.compute(A_matrix);
    readyFactorOp = true;
}

template<typename Scalar>
void MatVecDense<Scalar>::MultOPv(Scalar *x_in_ptr, Scalar *x_out_ptr) {
    assert(readyFactorOp and "FactorOp() has not been run yet.");
    auto t_token = t_multOPv->tic_token();
    switch(side) {
        case eig::Side::R: {
            Eigen::Map<VectorType> x_in(x_in_ptr, L);
            Eigen::Map<VectorType> x_out(x_out_ptr, L);
            x_out.noalias() = lu_dense.solve(x_in);
            break;
        }
        case eig::Side::L: {
            Eigen::Map<VectorTypeT> x_in(x_in_ptr, L);
            Eigen::Map<VectorTypeT> x_out(x_out_ptr, L);
            x_out.noalias() = x_in * lu_dense.inverse();
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
                T                     *x_in_ptr  = static_cast<T *>(x) + *ldx * i;
                T                     *x_out_ptr = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_in(x_in_ptr, L);
                Eigen::Map<VectorType> x_out(x_out_ptr, L);
                x_out.noalias() = lu_dense.solve(x_in);
                num_op++;
            }

            break;
        }
        case eig::Side::L: {
            for(int i = 0; i < *blockSize; i++) {
                T                     *x_in_ptr  = static_cast<T *>(x) + *ldx * i;
                T                     *x_out_ptr = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_in(x_in_ptr, L);
                Eigen::Map<VectorType> x_out(x_out_ptr, L);
                x_out.noalias() = x_in * lu_dense.inverse();
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
void MatVecDense<Scalar>::perform_op(const Scalar *x_in, Scalar *x_out) const {
    auto                         t_token = t_multAx->tic_token();
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    Eigen::Map<const VectorType> x_vec_in(x_in, L);
                    Eigen::Map<VectorType>       x_vec_out(x_out, L);
                    x_vec_out.noalias() = A_matrix * x_vec_in;
                    break;
                }
                case eig::Side::L: {
                    Eigen::Map<const VectorTypeT> x_vec_in(x_in, L);
                    Eigen::Map<VectorTypeT>       x_vec_out(x_out, L);
                    x_vec_out.noalias() = x_vec_in * A_matrix;
                    break;
                }
                case eig::Side::LR: {
                    throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
                }
            }
            break;
        case eig::Form::SYMM: {
            Eigen::Map<const VectorType> x_vec_in(x_in, L);
            Eigen::Map<VectorType>       x_vec_out(x_out, L);
            x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Lower>() * x_vec_in;
            break;
        }
    }
    num_mv++;
}

template<typename Scalar>
void MatVecDense<Scalar>::MultAx(Scalar *x_in, Scalar *x_out) {
    perform_op(x_in, x_out);
}

template<typename T>
void MatVecDense<T>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *err) {
    auto                         t_token = t_multAx->tic_token();
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    for(int i = 0; i < *blockSize; i++) {
                        T                     *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                     *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorType> x_vec_in(x_in, L);
                        Eigen::Map<VectorType> x_vec_out(x_out, L);
                        x_vec_out.noalias() = A_matrix * x_vec_in;
                        num_mv++;
                    }
                    break;
                }
                case eig::Side::L: {
                    for(int i = 0; i < *blockSize; i++) {
                        T                     *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                     *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorType> x_vec_in(x_in, L);
                        Eigen::Map<VectorType> x_vec_out(x_out, L);
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
                T                     *x_in  = static_cast<T *>(x) + *ldx * i;
                T                     *x_out = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_vec_in(x_in, L);
                Eigen::Map<VectorType> x_vec_out(x_out, L);
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
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
}

template<typename Scalar>
void MatVecDense<Scalar>::set_shift(Cplx sigma_) {
    if(readyShift) return;
    if(sigma == sigma_) return;
    eig::log->trace("Setting shift = {:.16f}", fp(sigma));
    sigma = sigma_;
    if(A_stl.empty()) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
    Eigen::Map<MatrixType> A_matrix(A_stl.data(), L, L);
    if constexpr(sfinae::is_std_complex_v<Scalar>) {
        A_matrix -= MatrixType::Identity(L, L) * sigma;
    } else {
        A_matrix -= MatrixType::Identity(L, L) * std::real(sigma);
    }
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
