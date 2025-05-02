#pragma once
#include "math/float.h"
#include "sfinae.h"
#include <complex>
#include <optional>

namespace eig {
    using size_type = long;

    // Enums
    enum class Lib { ARPACK, SPECTRA, PRIMME };               // Choose the underlying library
    enum class Form { SYMM, NSYM };                           // Symmetric or non-symmetric problems (complex symmetric are assumed Hermitian)
    enum class Type { FP32, FP64, FP128, CX32, CX64, CX128 }; // Real or complex, 32, 64 or 128 bit
    enum class Side { L, R, LR };                             // Left, right or both eigenvectors (for nsym problems)
    enum class Prob { STD, GEN };                             // Standard or generalized eigenvalue problem
    enum class Ritz {
        LA,
        SA,
        LM,
        SM,
        LR,
        SR,
        LI,
        SI,
        BE,
        primme_smallest,
        primme_largest,
        primme_closest_geq,
        primme_closest_leq,
        primme_closest_abs,
        primme_largest_abs
    }; // Choice of eigenvalue. LA is largest algebraic, and so on.
    enum class Factorization { NONE, LLT, LDLT, LU, QR, ILUT, ILDLT };
    enum class Preconditioner { NONE, JACOBI };
    enum class Storage { DENSE, SPARSE, MPS }; // Eigen Dense or Sparse, or std::vector for container
    enum class Shinv { ON, OFF };
    enum class Dephase { ON, OFF };
    enum class Vecs { ON, OFF };

    enum class PrimmeMethod {
        PRIMME_DEFAULT_METHOD,
        PRIMME_DYNAMIC,
        PRIMME_DEFAULT_MIN_TIME,
        PRIMME_DEFAULT_MIN_MATVECS,
        PRIMME_Arnoldi,
        PRIMME_GD,
        PRIMME_GD_plusK,
        PRIMME_GD_Olsen_plusK,
        PRIMME_JD_Olsen_plusK,
        PRIMME_RQI,
        PRIMME_JDQR,
        PRIMME_JDQMR,
        PRIMME_JDQMR_ETol,
        PRIMME_STEEPEST_DESCENT,
        PRIMME_LOBPCG_OrthoBasis,
        PRIMME_LOBPCG_OrthoBasis_Window,
    };

    inline Ritz stringToRitz(std::string_view ritzstring) {
        if(ritzstring == "LA") return Ritz::LA;
        if(ritzstring == "SA") return Ritz::SA;
        if(ritzstring == "LM") return Ritz::LM;
        if(ritzstring == "SM") return Ritz::SM;
        if(ritzstring == "LR") return Ritz::LR;
        if(ritzstring == "SR") return Ritz::SR;
        if(ritzstring == "LI") return Ritz::LI;
        if(ritzstring == "SI") return Ritz::SI;
        if(ritzstring == "BE") return Ritz::BE;
        if(ritzstring == "primme_smallest") return Ritz::primme_smallest;
        if(ritzstring == "primme_largest") return Ritz::primme_largest;
        if(ritzstring == "primme_closest_geq") return Ritz::primme_closest_geq;
        if(ritzstring == "primme_closest_leq") return Ritz::primme_closest_leq;
        if(ritzstring == "primme_closest_abs") return Ritz::primme_closest_abs;
        if(ritzstring == "primme_largest_abs") return Ritz::primme_largest_abs;
        throw std::runtime_error("Wrong ritz string: " + std::string(ritzstring));
    }

    inline std::string_view RitzToString(Ritz ritz) {
        switch(ritz) {
            case Ritz::LA: return "LA";
            case Ritz::SA: return "SA";
            case Ritz::LM: return "LM";
            case Ritz::SM: return "SM";
            case Ritz::LR: return "LR";
            case Ritz::SR: return "SR";
            case Ritz::LI: return "LI";
            case Ritz::SI: return "SI";
            case Ritz::BE: return "BE";
            case Ritz::primme_smallest: return "primme_smallest";
            case Ritz::primme_largest: return "primme_largest";
            case Ritz::primme_closest_geq: return "primme_closest_geq";
            case Ritz::primme_closest_leq: return "primme_closest_leq";
            case Ritz::primme_closest_abs: return "primme_closest_abs";
            case Ritz::primme_largest_abs: return "primme_largest_abs";
            default: throw std::logic_error("No valid eig::Ritz given");
        }
    }

    inline std::string_view FactorizationToString(Factorization fact) {
        switch(fact) {
            case Factorization::NONE: return "NONE";
            case Factorization::LLT: return "LLT";
            case Factorization::LDLT: return "LDLT";
            case Factorization::LU: return "LU";
            case Factorization::QR: return "QR";
            case Factorization::ILUT: return "ILUT";
            case Factorization::ILDLT: return "ILDLT";
            default: throw std::logic_error("No valid eig::Factorization given");
        }
    }
    inline std::string_view PreconditionerToString(Preconditioner prec) {
        switch(prec) {
            case Preconditioner::NONE: return "NONE";
            case Preconditioner::JACOBI: return "JACOBI";
            default: throw std::logic_error("No valid eig::Preconditioner given");
        }
    }
    inline std::string_view RitzToString(std::optional<Ritz> ritz) { return ritz ? RitzToString(ritz.value()) : "Ritz:NONE"; }
    inline std::string_view LibToString(Lib lib) {
        switch(lib) {
            case Lib::ARPACK: return "ARPACK";
            case Lib::SPECTRA: return "SPECTRA";
            case Lib::PRIMME: return "PRIMME";
            default: throw std::logic_error("No valid eig::Lib given");
        }
    }
    inline std::string_view LibToString(std::optional<Lib> lib) { return lib ? LibToString(lib.value()) : "Lib:NONE"; }
    inline eig::Lib         StringToLib(std::string_view libstr) {
        if(libstr == "ARPACK") return Lib::ARPACK;
        if(libstr == "PRIMME") return Lib::PRIMME;
        if(libstr == "SPECTRA") return Lib::SPECTRA;
        throw std::logic_error("The string does not match any eig::Lib");
    }

    constexpr std::string_view TypeToString(Type type) {
        switch(type) {
            case Type::FP32: return "FP32";
            case Type::FP64: return "FP64";
            case Type::FP128: return "FP128";
            case Type::CX32: return "CX32";
            case Type::CX64: return "CX64";
            case Type::CX128: return "CX128";
            default: throw std::logic_error("Not a valid eig::Type");
        }
    }

    template<typename Scalar>
    constexpr eig::Type ScalarToType() {
        if constexpr(std::is_same_v<Scalar, fp32>) return eig::Type::FP32;
        if constexpr(std::is_same_v<Scalar, fp64>) return eig::Type::FP64;
        if constexpr(std::is_same_v<Scalar, fp128>) return eig::Type::FP128;
        if constexpr(std::is_same_v<Scalar, cx32>) return eig::Type::CX32;
        if constexpr(std::is_same_v<Scalar, cx64>) return eig::Type::CX64;
        if constexpr(std::is_same_v<Scalar, cx128>) return eig::Type::CX128;
        throw std::logic_error("Not a valid Scalar for eig::Type");
    }

    constexpr std::string_view TypeToString(std::optional<Type> type) { return type ? TypeToString(type.value()) : "Type:UNKNOWN"; }

    constexpr std::string_view ProbToString(Prob prob) {
        switch(prob) {
            case Prob::STD: return "STD";
            case Prob::GEN: return "GEN";
            default: throw std::logic_error("Not a valid eig::Prob");
        }
    }

    inline PrimmeMethod stringToMethod(std::optional<std::string> methodstring) {
        if(not methodstring.has_value()) return PrimmeMethod::PRIMME_DYNAMIC;
        if(methodstring.value() == "PRIMME_DEFAULT_METHOD") return PrimmeMethod::PRIMME_DEFAULT_METHOD;
        if(methodstring.value() == "PRIMME_DYNAMIC") return PrimmeMethod::PRIMME_DYNAMIC;
        if(methodstring.value() == "PRIMME_DEFAULT_MIN_TIME") return PrimmeMethod::PRIMME_DEFAULT_MIN_TIME;
        if(methodstring.value() == "PRIMME_DEFAULT_MIN_MATVECS") return PrimmeMethod::PRIMME_DEFAULT_MIN_MATVECS;
        if(methodstring.value() == "PRIMME_Arnoldi") return PrimmeMethod::PRIMME_Arnoldi;
        if(methodstring.value() == "PRIMME_GD") return PrimmeMethod::PRIMME_GD;
        if(methodstring.value() == "PRIMME_GD_plusK") return PrimmeMethod::PRIMME_GD_plusK;
        if(methodstring.value() == "PRIMME_GD_Olsen_plusK") return PrimmeMethod::PRIMME_GD_Olsen_plusK;
        if(methodstring.value() == "PRIMME_JD_Olsen_plusK") return PrimmeMethod::PRIMME_JD_Olsen_plusK;
        if(methodstring.value() == "PRIMME_RQI") return PrimmeMethod::PRIMME_RQI;
        if(methodstring.value() == "PRIMME_JDQR") return PrimmeMethod::PRIMME_JDQR;
        if(methodstring.value() == "PRIMME_JDQMR") return PrimmeMethod::PRIMME_JDQMR;
        if(methodstring.value() == "PRIMME_JDQMR_ETol") return PrimmeMethod::PRIMME_JDQMR_ETol;
        if(methodstring.value() == "PRIMME_STEEPEST_DESCENT") return PrimmeMethod::PRIMME_STEEPEST_DESCENT;
        if(methodstring.value() == "PRIMME_LOBPCG_OrthoBasis") return PrimmeMethod::PRIMME_LOBPCG_OrthoBasis;
        if(methodstring.value() == "PRIMME_LOBPCG_OrthoBasis_Window") return PrimmeMethod::PRIMME_LOBPCG_OrthoBasis_Window;
        throw std::runtime_error("Wrong method string: " + methodstring.value());
    }
    inline std::string_view MethodToString(std::optional<PrimmeMethod> method) {
        if(not method.has_value()) return "PRIMME_DYNAMIC";
        switch(method.value()) {
            case PrimmeMethod::PRIMME_DEFAULT_METHOD: return "PRIMME_DEFAULT_METHOD";
            case PrimmeMethod::PRIMME_DYNAMIC: return "PRIMME_DYNAMIC";
            case PrimmeMethod::PRIMME_DEFAULT_MIN_TIME: return "PRIMME_DEFAULT_MIN_TIME";
            case PrimmeMethod::PRIMME_DEFAULT_MIN_MATVECS: return "PRIMME_DEFAULT_MIN_MATVECS";
            case PrimmeMethod::PRIMME_Arnoldi: return "PRIMME_Arnoldi";
            case PrimmeMethod::PRIMME_GD: return "PRIMME_GD";
            case PrimmeMethod::PRIMME_GD_plusK: return "PRIMME_GD_plusK";
            case PrimmeMethod::PRIMME_GD_Olsen_plusK: return "PRIMME_GD_Olsen_plusK";
            case PrimmeMethod::PRIMME_JD_Olsen_plusK: return "PRIMME_JD_Olsen_plusK";
            case PrimmeMethod::PRIMME_RQI: return "PRIMME_RQI";
            case PrimmeMethod::PRIMME_JDQR: return "PRIMME_JDQR";
            case PrimmeMethod::PRIMME_JDQMR: return "PRIMME_JDQMR";
            case PrimmeMethod::PRIMME_JDQMR_ETol: return "PRIMME_JDQMR_ETol";
            case PrimmeMethod::PRIMME_STEEPEST_DESCENT: return "PRIMME_STEEPEST_DESCENT";
            case PrimmeMethod::PRIMME_LOBPCG_OrthoBasis: return "PRIMME_LOBPCG_OrthoBasis";
            case PrimmeMethod::PRIMME_LOBPCG_OrthoBasis_Window: return "PRIMME_LOBPCG_OrthoBasis_Window";
            default: throw std::logic_error("No valid eig::PrimmeMethod given");
        }
    }
}