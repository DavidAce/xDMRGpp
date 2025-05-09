#pragma once
#include "../common.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include <Eigen/Core>

namespace linalg::matrix {

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
}
