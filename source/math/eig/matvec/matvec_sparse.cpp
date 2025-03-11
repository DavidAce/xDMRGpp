
#include "matvec_sparse.h"
#include "../log.h"
#include "math/float.h"
#include "tid/tid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#define profile_matrix_product_sparse 1

template<typename T>
using SparseMatrixType = Eigen::SparseMatrix<T>;
template<typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template<typename T>
using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
template<typename T>
using VectorTypeT = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

namespace sparse_lu {
    std::optional<Eigen::SparseMatrix<fp64>>                             A_real_sparse  = std::nullopt;
    std::optional<Eigen::SparseMatrix<cx64>>               A_cplx_sparse  = std::nullopt;
    std::optional<Eigen::SparseLU<SparseMatrixType<fp64>>>               lu_real_sparse = {};
    std::optional<Eigen::SparseLU<SparseMatrixType<cx64>>> lu_cplx_sparse = {};
    std::optional<Eigen::PartialPivLU<MatrixType<fp64>>>                 lu_real_dense  = std::nullopt;
    std::optional<Eigen::PartialPivLU<MatrixType<cx64>>>   lu_cplx_dense  = std::nullopt;

    void reset() {
        A_real_sparse.reset();
        A_cplx_sparse.reset();
        lu_real_sparse.reset();
        lu_cplx_sparse.reset();
        lu_real_dense.reset();
        lu_cplx_dense.reset();
    }
}

template<typename Scalar, bool sparseLU>
MatVecSparse<Scalar, sparseLU>::~MatVecSparse() {
    sparse_lu::reset();
}

template<typename Scalar, bool sparseLU>
MatVecSparse<Scalar, sparseLU>::MatVecSparse(const Scalar *A_, long L_, bool copy_data, eig::Form form_, eig::Side side_)
    : A_ptr(A_), L(L_), form(form_), side(side_) {
    if(copy_data) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }

    if constexpr(sparseLU) {
        Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
        if constexpr(std::is_same_v<Scalar, fp64>) {
            sparse_lu::A_real_sparse = A_matrix.sparseView();
            sparse_lu::A_real_sparse.value().makeCompressed();
        }
        if constexpr(std::is_same_v<Scalar, cx64>) {
            sparse_lu::A_cplx_sparse = A_matrix.sparseView();
            sparse_lu::A_cplx_sparse.value().makeCompressed();
        }
    }
    t_factorOP = std::make_unique<tid::ur>("Time FactorOp");
    t_genMat   = std::make_unique<tid::ur>("Time genMat");
    t_multOPv  = std::make_unique<tid::ur>("Time MultOpv");
    t_multAx   = std::make_unique<tid::ur>("Time MultAx");
    t_multPc   = std::make_unique<tid::ur>("Time MultPc");
}

// Function definitions

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::FactorOP()

/*  Sparse decomposition
 *  Factors P(A-sigma*I) = LU
 */

{
    if(readyFactorOp) { return; }
    if(not readyShift) throw std::runtime_error("Cannot FactorOP: Shift value sigma has not been set.");
    auto                                 t_token = t_factorOP->tic_token();
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);

    // Real
    if constexpr(std::is_same_v<Scalar, fp64> and not sparseLU) {
        sparse_lu::lu_real_dense = Eigen::PartialPivLU<MatrixType<Scalar>>();
        sparse_lu::lu_real_dense.value().compute(A_matrix);
    }
    if constexpr(std::is_same_v<Scalar, fp64> and sparseLU) { sparse_lu::lu_real_sparse.value().compute(sparse_lu::A_real_sparse.value()); }
    // Complex
    if constexpr(std::is_same_v<Scalar, cx64> and not sparseLU) {
        sparse_lu::lu_cplx_dense = Eigen::PartialPivLU<MatrixType<Scalar>>();
        sparse_lu::lu_cplx_dense.value().compute(A_matrix);
    }
    if constexpr(std::is_same_v<Scalar, cx64> and sparseLU) { sparse_lu::lu_cplx_sparse.value().compute(sparse_lu::A_cplx_sparse.value()); }

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
            if constexpr(std::is_same_v<Scalar, fp64> and not sparseLU)
                x_out.noalias() = sparse_lu::lu_real_dense.value().solve(x_in);
            else if constexpr(std::is_same_v<Scalar, cx64> and not sparseLU)
                x_out.noalias() = sparse_lu::lu_cplx_dense.value().solve(x_in);
            else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                x_out.noalias() = sparse_lu::lu_real_sparse.value().solve(x_in);
            else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                x_out.noalias() = sparse_lu::lu_cplx_sparse.value().solve(x_in);
            break;
        }
        case eig::Side::L: {
            if constexpr(std::is_same_v<Scalar, fp64> and not sparseLU)
                x_out.noalias() = x_in * sparse_lu::lu_real_dense.value().inverse();
            else if constexpr(std::is_same_v<Scalar, cx64> and not sparseLU)
                x_out.noalias() = x_in * sparse_lu::lu_cplx_dense.value().inverse();
            else { throw std::runtime_error("Left sided sparse shift invert hasn't been implemented yet"); }
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
                if constexpr(std::is_same_v<Scalar, fp64> and not sparseLU)
                    x_out.noalias() = sparse_lu::lu_real_dense.value().solve(x_in);
                else if constexpr(std::is_same_v<Scalar, cx64> and not sparseLU)
                    x_out.noalias() = sparse_lu::lu_cplx_dense.value().solve(x_in);
                else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                    x_out.noalias() = sparse_lu::lu_real_sparse.value().solve(x_in);
                else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                    x_out.noalias() = sparse_lu::lu_cplx_sparse.value().solve(x_in);
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
                if constexpr(std::is_same_v<Scalar, fp64> and not sparseLU)
                    x_out.noalias() = x_in * sparse_lu::lu_real_dense.value().inverse();
                else if constexpr(std::is_same_v<Scalar, cx64> and not sparseLU)
                    x_out.noalias() = x_in * sparse_lu::lu_cplx_dense.value().inverse();
                else { throw std::runtime_error("Left sided sparse shift invert hasn't been implemented yet"); }
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
void MatVecSparse<Scalar, sparseLU>::MultAx(Scalar *x_in, Scalar *x_out) {
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
    switch(form) {
        case eig::Form::NSYM:
            switch(side) {
                case eig::Side::R: {
                    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
                    Eigen::Map<VectorType> x_vec_in(x_in, L);
                    Eigen::Map<VectorType> x_vec_out(x_out, L);
                    if constexpr(not sparseLU)
                        x_vec_out.noalias() = A_matrix * x_vec_in;
                    else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                        x_vec_out.noalias() = sparse_lu::A_real_sparse.value() * x_vec_in;
                    else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                        x_vec_out.noalias() = sparse_lu::A_cplx_sparse.value() * x_vec_in;
                    break;
                }
                case eig::Side::L: {
                    using VectorTypeT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
                    Eigen::Map<VectorTypeT> x_vec_in(x_in, L);
                    Eigen::Map<VectorTypeT> x_vec_out(x_out, L);
                    if constexpr(not sparseLU)
                        x_vec_out.noalias() = x_vec_in * A_matrix;
                    else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                        x_vec_out.noalias() = x_vec_in * sparse_lu::A_real_sparse.value();
                    else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                        x_vec_out.noalias() = x_vec_in * sparse_lu::A_cplx_sparse.value();
                    break;
                }
                case eig::Side::LR: {
                    throw std::runtime_error("eigs cannot handle sides L and R simultaneously");
                }
            }
            break;
        case eig::Form::SYMM: {
            using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            Eigen::Map<VectorType> x_vec_in(x_in, L);
            Eigen::Map<VectorType> x_vec_out(x_out, L);
            if constexpr(not sparseLU) x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Upper>() * x_vec_in;
            if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                x_vec_out.noalias() = sparse_lu::A_real_sparse.value().template selfadjointView<Eigen::Upper>() * x_vec_in;
            if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                x_vec_out.noalias() = sparse_lu::A_cplx_sparse.value().template selfadjointView<Eigen::Upper>() * x_vec_in;
            break;
        }
    }
    num_mv++;
}

template<typename T, bool sparseLU>
void MatVecSparse<T, sparseLU>::MultAx(void *x, int *ldx, void *y, int *ldy, int *blockSize, [[maybe_unused]] primme_params *primme,
                                       [[maybe_unused]] int *err) {
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
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
                        if constexpr(not sparseLU)
                            x_vec_out.noalias() = A_matrix * x_vec_in;
                        else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                            x_vec_out.noalias() = sparse_lu::A_real_sparse.value() * x_vec_in;
                        else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                            x_vec_out.noalias() = sparse_lu::A_cplx_sparse.value() * x_vec_in;
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
                        if constexpr(not sparseLU)
                            x_vec_out.noalias() = x_vec_in * A_matrix;
                        else if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                            x_vec_out.noalias() = x_vec_in * sparse_lu::A_real_sparse.value();
                        else if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                            x_vec_out.noalias() = x_vec_in * sparse_lu::A_cplx_sparse.value();
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
                if constexpr(not sparseLU) x_vec_out.noalias() = A_matrix.template selfadjointView<Eigen::Upper>() * x_vec_in;
                if constexpr(std::is_same_v<Scalar, fp64> and sparseLU)
                    x_vec_out.noalias() = sparse_lu::A_real_sparse.value().template selfadjointView<Eigen::Upper>() * x_vec_in;
                if constexpr(std::is_same_v<Scalar, cx64> and sparseLU)
                    x_vec_out.noalias() = sparse_lu::A_cplx_sparse.value().template selfadjointView<Eigen::Upper>() * x_vec_in;
                num_mv++;
            }
            break;
        }
    }
    *err = 0;
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::print() const {
    Eigen::Map<const MatrixType<Scalar>> A_matrix(A_ptr, L, L);
}

template<typename Scalar, bool sparseLU>
void MatVecSparse<Scalar, sparseLU>::set_shift(std::complex<double> shift) {
    if(readyShift) return;
    if(sigma == shift) return;
    eig::log->trace("Setting shift = {:.16f} + i{:.16f}", std::real(shift), std::imag(shift));

    sigma = shift;
    if(A_stl.empty()) {
        A_stl.resize(safe_cast<size_t>(L * L));
        std::copy(A_ptr, A_ptr + safe_cast<size_t>(L * L), A_stl.begin());
        A_ptr = A_stl.data();
    }
    Eigen::Map<MatrixType<Scalar>> A_matrix(A_stl.data(), L, L);
    if constexpr(std::is_same_v<Scalar, fp64>) A_matrix -= Eigen::MatrixXd::Identity(L, L) * std::real(sigma);
    if constexpr(std::is_same_v<Scalar, cx64>) A_matrix -= Eigen::MatrixXd::Identity(L, L) * sigma;

    if constexpr(sparseLU) {
        if constexpr(std::is_same_v<Scalar, double>) {
            sparse_lu::A_real_sparse = A_matrix.sparseView();
            sparse_lu::A_real_sparse.value().makeCompressed();
        }
        if constexpr(std::is_same_v<Scalar, std::complex<double>>) {
            sparse_lu::A_cplx_sparse = A_matrix.sparseView();
            sparse_lu::A_cplx_sparse.value().makeCompressed();
        }
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
template<typename Scalar, bool sparseLU>
eig::Type MatVecSparse<Scalar, sparseLU>::get_type() const {
    if constexpr(std::is_same_v<Scalar, fp64>)
        return eig::Type::FP64;
    else if constexpr(std::is_same_v<Scalar, cx64>)
        return eig::Type::CX64;
    else
        throw std::runtime_error("Unsupported type");
}

// Explicit instantiations
template class MatVecSparse<fp64, true>;
template class MatVecSparse<fp64, false>;
template class MatVecSparse<cx64, true>;
template class MatVecSparse<cx64, false>;
