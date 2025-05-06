#pragma once
#include "common.h"
#include "math/float.h"
#include "unsupported/Eigen/KroneckerProduct"
#include <Eigen/Core>

namespace linalg::matrix {
    template<auto N, auto M>
    constexpr auto multiply() {
        if constexpr(N == -1 or M == -1)
            return -1;
        else
            return N * M;
    }

    template<typename DA, typename DB>
    using KroneckerResultType = Eigen::Matrix<cplx_or_real<typename DA::Scalar, typename DB::Scalar>, multiply<DA::RowsAtCompileTime, DB::RowsAtCompileTime>(),
                                              multiply<DA::ColsAtCompileTime, DB::ColsAtCompileTime>(), Eigen::ColMajor>;

    template<typename DerivedA, typename DerivedB>
    KroneckerResultType<DerivedA, DerivedB> kronecker(const Eigen::PlainObjectBase<DerivedA> &A, const Eigen::PlainObjectBase<DerivedB> &B, bool mirror) {
        if(mirror)
            return Eigen::kroneckerProduct(B.derived(), A.derived());
        else
            return Eigen::kroneckerProduct(A.derived(), B.derived());
    }

    template<typename DerivedA, typename DerivedB>
    auto kronecker(const Eigen::EigenBase<DerivedA> &A, const Eigen::EigenBase<DerivedB> &B, bool mirror = false) {
        if constexpr(is_PlainObject<DerivedA>::value and is_PlainObject<DerivedB>::value)
            return kronecker(A, B, mirror);
        else { return kronecker(A.derived().eval(), B.derived().eval(), mirror); }
    }

    template<typename T, typename IOFormat_t, typename = std::enable_if_t<std::is_same_v<IOFormat_t, Eigen::IOFormat>>>
    std::string to_string(const Eigen::EigenBase<T> &m, const IOFormat_t &f) {
        using Scalar = typename T::Scalar;
        if constexpr(sfinae::is_native_float<Scalar>) {
            std::stringstream ss;
            ss << m.derived().format(f);
            return ss.str();
        } else if constexpr(is_std_complex_v<Scalar>) {
            using Inner = typename Scalar::value_type;
            if constexpr(sfinae::is_native_float<Inner>) {
                std::stringstream ss;
                ss << m.derived().format(f);
                return ss.str();
            }
#if defined(DMRG_USE_QUADMATH)
            else if constexpr(std::is_same_v<Inner, fp128>) {
                std::string s;
                auto        probablesize = 4 + 2 * (m.derived().size() + 1) * std::max<size_t>(64, static_cast<size_t>(f.precision) + 34);
                s.reserve(probablesize);
                std::string fmt = "%." + std::to_string(f.precision) + "Qg";
                s += f.matPrefix;
                for(long i = 0; i < m.derived().rows(); ++i) {
                    s += f.rowPrefix;
                    for(long j = 0; j < m.derived().cols(); ++j) {
                        int rextent = quadmath_snprintf(nullptr, 0, fmt.data(), m.derived()(i, j).real());
                        int iextent = quadmath_snprintf(nullptr, 0, fmt.data(), m.derived()(i, j).imag());
                        if(rextent < 0) throw std::runtime_error("quadmath_snprintf (real) returned < 0");
                        if(iextent < 0) throw std::runtime_error("quadmath_snprintf (imag) returned < 0");
                        s += '(';
                        auto roffset = s.size();
                        s.resize(roffset + static_cast<size_t>(rextent));
                        quadmath_snprintf(s.data() + roffset, static_cast<size_t>(rextent + 1), fmt.data(), m.derived()(i, j).real());
                        s += ',';
                        auto ioffset = s.size();
                        s.resize(ioffset + static_cast<size_t>(iextent));
                        quadmath_snprintf(s.data() + ioffset, static_cast<size_t>(iextent + 1), fmt.data(), m.derived()(i, j).imag());
                        s += ')';
                        if(j + 1 != m.derived().cols()) s += f.coeffSeparator;
                    }
                    s += f.rowSuffix;
                    if(i + 1 != m.derived().rows()) s += f.rowSeparator;
                }
                s += f.matSuffix;
                return s;
            }
#endif
            else {
                throw std::runtime_error("to_string: this complex type is not implemented");
            }

        }
#if defined(DMRG_USE_QUADMATH)
        else if constexpr(std::is_same_v<Scalar, fp128>) {
            std::string s;
            std::string fmt          = "%." + std::to_string(f.precision) + "Qf";
            auto        probablesize = m.derived().size() * std::max<size_t>(64, static_cast<size_t>(f.precision) + 34);
            s.reserve(probablesize);
            s += f.matPrefix;
            for(long i = 0; i < m.derived().rows(); ++i) {
                s += f.rowPrefix;
                for(long j = 0; j < m.derived().cols(); ++j) {
                    int extent = quadmath_snprintf(nullptr, 0, fmt.data(), m.derived()(i, j));
                    if(extent < 0) throw std::runtime_error("quadmath_snprintf returned < 0");
                    auto offset = s.size();
                    s.resize(s.size() + static_cast<size_t>(extent));
                    quadmath_snprintf(s.data() + offset, static_cast<size_t>(extent + 1), fmt.data(), m.derived()(i, j));
                    if(j + 1 != m.derived().cols()) s += f.coeffSeparator;
                }
                s += f.rowSuffix;
                if(i + 1 != m.derived().rows()) s += f.rowSeparator;
            }
            s += f.matSuffix;
            s.shrink_to_fit();
            return s;
        }
#endif
        else
            throw std::runtime_error("to_string: type is not implemented");
    }

    template<typename T>
    std::string to_string(const Eigen::EigenBase<T> &m, int prec, int flags = 0, const std::string &sep = ", ") {
        Eigen::IOFormat f(prec, flags, sep, "\n", "[", "]", "[", "]");
        return to_string(m, f);
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> modified_gram_schmidt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Vconst) {
        // Orthonormalize with Modified Gram Schmidt
        using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using Real       = decltype(std::real(std::declval<T>()));
        MatrixType V     = Vconst;
        MatrixType Q     = MatrixType::Zero(V.rows(), V.cols());
        MatrixType R     = MatrixType::Zero(V.cols(), V.cols());
        for(long i = 0; i < V.cols(); ++i) {
            Q.col(i) = V.col(i);
            R(i, i)  = Q.col(i).norm();
            if(std::abs(R(i, i)) < std::numeric_limits<Real>::epsilon()) {
                // tools::log->error("Q.col({}) is a zero vector:\n Q: \n{}\n", i, linalg::matrix::to_string(Q.real(), 8));
                continue;
            }
            Q.col(i) /= R(i, i);
            for(long j = i + 1; j < V.cols(); ++j) {
                R(i, j) = Q.col(i).dot(V.col(j));
                V.col(j) -= Q.col(i) * R(i, j);
            }
        }
        return Q;
    }

}
