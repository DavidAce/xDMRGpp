
#include "matvec_sparse.h"
#include "../log.h"
#include "io/fmt_custom.h"
#include "math/float.h"
#include "tid/tid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <general/sfinae.h>
#define profile_matrix_product_sparse 1

// Explicit instantiations
template class MatVecSparse<fp32, false>;
template class MatVecSparse<fp64, false>;
template class MatVecSparse<fp128, false>;
template class MatVecSparse<cx32, false>;
template class MatVecSparse<cx64, false>;
template class MatVecSparse<cx128, false>;
template class MatVecSparse<fp32, true>;
template class MatVecSparse<fp64, true>;
template class MatVecSparse<fp128, true>;
template class MatVecSparse<cx32, true>;
template class MatVecSparse<cx64, true>;
template class MatVecSparse<cx128, true>;

template<typename Scalar, bool sparseLU>
MatVecSparse<Scalar, sparseLU>::MatVecSparse(const Scalar *A_, long L_, bool copy_data, eig::Form form_, eig::Side side_)
    : A_ptr(A_), L(L_), form(form_), side(side_) {
    if(copy_data) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::init_timers() {
    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}

/*  Sparse decomposition
 *  Factors P(A-sigma*I) = LU
 */

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::FactorOP() {
    if(readyFactorOp) { return; }
    if(not readyShift) throw std::runtime_error("Cannot FactorOP: Shift value sigma has not been set.");
    auto                         t_token = t_factorOP->tic_token();
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);

    if constexpr(sparseLU) {
        A_sparse = A_matrix.sparseView();
        A_sparse.makeCompressed();
        lu.compute(A_sparse);
    } else {
        lu.compute(A_matrix);
    }
    readyFactorOp = true;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::MultOPv(Scalar *x_in_ptr, Scalar *x_out_ptr) {
    if(not readyFactorOp) throw std::logic_error("FactorOp() has not been run yet.");
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Eigen::Map<VectorType> x_in(x_in_ptr, L);
    Eigen::Map<VectorType> x_out(x_out_ptr, L);

    switch(side) {
        case eig::Side::R: {
            x_out.noalias() = lu.solve(x_in);
            break;
        }
        case eig::Side::L: {
            if constexpr(sparseLU)
                throw std::runtime_error("Left sided sparse MultOPv has not been implemented");
            else
                x_out.noalias() = x_in * lu.inverse();
            break;
        }
        case eig::Side::LR: {
            throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
        }
    }
    num_op++;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::MultOPv(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme, int *err) {
    if(not readyFactorOp) throw std::logic_error("FactorOp() has not been run yet.");
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    switch(side) {
        case eig::Side::R: {
            for(int i = 0; i < *blockSize; i++) {
                Scalar                *x_in_ptr  = static_cast<Scalar *>(x) + *ldx * i;
                Scalar                *x_out_ptr = static_cast<Scalar *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_in(x_in_ptr, L);
                Eigen::Map<VectorType> x_out(x_out_ptr, L);
                x_out.noalias() = lu.solve(x_in);
                num_op++;
            }
            break;
        }
        case eig::Side::L: {
            for(int i = 0; i < *blockSize; i++) {
                Scalar                *x_in_ptr  = static_cast<Scalar *>(x) + *ldx * i;
                Scalar                *x_out_ptr = static_cast<Scalar *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_in(x_in_ptr, L);
                Eigen::Map<VectorType> x_out(x_out_ptr, L);
                if constexpr(sparseLU)
                    throw std::runtime_error("Left sided sparse MultOPv has not been implemented");
                else
                    x_out.noalias() = x_in * lu.inverse();
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

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::perform_op(const Scalar *x_in, Scalar *x_out) const {
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
                    Eigen::Map<const VectorType> x_vec_in(x_in, L);
                    Eigen::Map<VectorType>       x_vec_out(x_out, L);
                    if constexpr(sparseLU) {
                        x_vec_out.noalias() = A_sparse * x_vec_in;
                    } else {
                        x_vec_out.noalias() = A_matrix * x_vec_in;
                    }
                    break;
                }
                case eig::Side::L: {
                    using VectorTypeT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
                    Eigen::Map<const VectorTypeT> x_vec_in(x_in, L);
                    Eigen::Map<VectorTypeT>       x_vec_out(x_out, L);
                    if constexpr(sparseLU) {
                        x_vec_out.noalias() = x_vec_in * A_sparse;
                    } else {
                        x_vec_out.noalias() = x_vec_in * A_matrix;
                    }
                    break;
                }
                case eig::Side::LR: {
                    throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
                }
            }
            break;
        case eig::Form::SYMM: {
            using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            Eigen::Map<const VectorType> x_vec_in(x_in, L);
            Eigen::Map<VectorType>       x_vec_out(x_out, L);
            if constexpr(sparseLU) {
                x_vec_out.noalias() = A_sparse * x_vec_in;
            } else {
                x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Upper>() * x_vec_in;
            }
            break;
        }
    }
    num_mv++;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::MultAx(Scalar *x_in, Scalar *x_out) {
    perform_op(x_in, x_out);
}

template<typename T, bool sparseLU>
void MatVecSparse<T, sparseLU>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme,
                                       [[maybe_unused]] int *err) {
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
                    for(int i = 0; i < *blockSize; i++) {
                        T                     *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                     *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorType> x_vec_in(x_in, L);
                        Eigen::Map<VectorType> x_vec_out(x_out, L);
                        if constexpr(sparseLU) {
                            x_vec_out.noalias() = A_sparse * x_vec_in;
                        } else {
                            x_vec_out.noalias() = A_matrix * x_vec_in;
                        }
                        num_mv++;
                    }
                    break;
                }
                case eig::Side::L: {
                    using VectorTypeT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
                    for(int i = 0; i < *blockSize; i++) {
                        T                      *x_in  = static_cast<T *>(x) + *ldx * i;
                        T                      *x_out = static_cast<T *>(y) + *ldy * i;
                        Eigen::Map<VectorTypeT> x_vec_in(x_in, L);
                        Eigen::Map<VectorTypeT> x_vec_out(x_out, L);
                        if constexpr(sparseLU) {
                            x_vec_out.noalias() = x_vec_in * A_sparse;
                        } else {
                            x_vec_out.noalias() = x_vec_in * A_matrix;
                        }
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
            using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
            for(int i = 0; i < *blockSize; i++) {
                T                     *x_in  = static_cast<T *>(x) + *ldx * i;
                T                     *x_out = static_cast<T *>(y) + *ldy * i;
                Eigen::Map<VectorType> x_vec_in(x_in, L);
                Eigen::Map<VectorType> x_vec_out(x_out, L);
                if constexpr(sparseLU) {
                    x_vec_out.noalias() = A_sparse * x_vec_in;
                } else {
                    x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Upper>() * x_vec_in;
                }
                num_mv++;
            }
            break;
        }
    }
    *err = 0;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::print() const {
    Eigen::Map<const MatrixType> A_matrix(A_ptr, L, L);
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::set_shift(Cplx shift) {
    if(readyShift) return;
    if(sigma == shift) return;
    eig::log->trace("Setting shift = {:.16f}", fp(shift));

    sigma = shift;
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

    if constexpr(sparseLU) {
        A_sparse = A_matrix.sparseView();
        A_sparse.makeCompressed();
    }

    readyShift = true;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::set_mode(const eig::Form form_) {
    form = form_;
}
template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::set_side(const eig::Side side_) {
    side = side_;
}
template<typename Scalar, bool sparseLU>
eig::Form MatVecSparse<Scalar, sparseLU>::get_form() const {
    return form;
}
template<typename Scalar, bool sparseLU>
eig::Side MatVecSparse<Scalar, sparseLU>::get_side() const {
    return side;
}
