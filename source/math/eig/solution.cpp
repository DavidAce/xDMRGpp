#include "solution.h"
namespace eig {

    template<typename Scalar, Side side>
    std::vector<Scalar> &solution::get_eigvecs() const {
        static_assert(side != Side::LR and "Cannot get both L/R eigvecs simultaneusly");
        if constexpr(std::is_same_v<Scalar, fp32>) {
            build_eigvecs_fp32();
            if constexpr(side == Side::R) return eigvecsR_real_fp32;
            if constexpr(side == Side::L) return eigvecsL_real_fp32;
        } else if constexpr(std::is_same_v<Scalar, fp64>) {
            build_eigvecs_fp64();
            if constexpr(side == Side::R) return eigvecsR_real_fp64;
            if constexpr(side == Side::L) return eigvecsL_real_fp64;
        } else if constexpr(std::is_same_v<Scalar, cx32>) {
            build_eigvecs_cx32();
            if constexpr(side == Side::R) return eigvecsR_cx32;
            if constexpr(side == Side::L) return eigvecsL_cx32;
        } else if constexpr(std::is_same_v<Scalar, cx64>) {
            build_eigvecs_cx64();
            if constexpr(side == Side::R) return eigvecsR_cx64;
            if constexpr(side == Side::L) return eigvecsL_cx64;
        }
    }
    template std::vector<fp32> &solution::get_eigvecs<fp32, Side::L>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Side::L>() const;
    template std::vector<fp32> &solution::get_eigvecs<fp32, Side::R>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Side::R>() const;
    template std::vector<fp64> &solution::get_eigvecs<fp64, Side::L>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Side::L>() const;
    template std::vector<fp64> &solution::get_eigvecs<fp64, Side::R>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Side::R>() const;

    template<typename Scalar>
    std::vector<Scalar> &solution::get_eigvecs(Side side) const {
        if(side == Side::R) return get_eigvecs<Scalar, Side::R>();
        if(side == Side::L) return get_eigvecs<Scalar, Side::L>();
        throw std::runtime_error("Cannot return both L and R eigenvectors");
    }
    template std::vector<fp64> &solution::get_eigvecs<fp64>(Side side) const;
    template std::vector<cx64> &solution::get_eigvecs<cx64>(Side side) const;

    template<typename Scalar, Form form, Side side>
    std::vector<Scalar> &solution::get_eigvecs() const {
        if constexpr(std::is_same<fp32, Scalar>::value) return get_eigvecs<form, Type::FP32, side>();
        if constexpr(std::is_same<fp64, Scalar>::value) return get_eigvecs<form, Type::FP64, side>();
        if constexpr(std::is_same<cx32, Scalar>::value) return get_eigvecs<form, Type::CX32, side>();
        if constexpr(std::is_same<cx64, Scalar>::value) return get_eigvecs<form, Type::CX64, side>();
    }

    template std::vector<fp32> &solution::get_eigvecs<fp32, Form::SYMM, Side::L>() const;
    template std::vector<fp32> &solution::get_eigvecs<fp32, Form::SYMM, Side::R>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Form::SYMM, Side::L>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Form::SYMM, Side::R>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Form::NSYM, Side::L>() const;
    template std::vector<cx32> &solution::get_eigvecs<cx32, Form::NSYM, Side::R>() const;
    template std::vector<fp64> &solution::get_eigvecs<fp64, Form::SYMM, Side::L>() const;
    template std::vector<fp64> &solution::get_eigvecs<fp64, Form::SYMM, Side::R>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Form::SYMM, Side::L>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Form::SYMM, Side::R>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Form::NSYM, Side::L>() const;
    template std::vector<cx64> &solution::get_eigvecs<cx64, Form::NSYM, Side::R>() const;

    template<typename Scalar>
    std::vector<Scalar> &solution::get_eigvals() const {
        if constexpr(std::is_same_v<Scalar, fp32>) {
            build_eigvals_fp32();
            return eigvals_real_fp32;
        } else if constexpr(std::is_same_v<Scalar, fp64>) {
            build_eigvals_fp64();
            return eigvals_real_fp64;
        } else if constexpr(std::is_same_v<Scalar, cx32>) {
            build_eigvals_cx32();
            return eigvals_cx32;
        } else if constexpr(std::is_same_v<Scalar, cx64>) {
            build_eigvals_cx64();
            return eigvals_cx64;
        }
    }

    template std::vector<fp32> &solution::get_eigvals<fp32>() const;
    template std::vector<fp64> &solution::get_eigvals<fp64>() const;
    template std::vector<cx32> &solution::get_eigvals<cx32>() const;
    template std::vector<cx64> &solution::get_eigvals<cx64>() const;

    const std::vector<double> &solution::get_resnorms() const { return meta.residual_norms; }

    void solution::reset() {
        eigvals_real_fp32.clear();
        eigvals_imag_fp32.clear();
        eigvals_cx32.clear();
        eigvecsR_real_fp32.clear();
        eigvecsR_imag_fp32.clear();
        eigvecsL_real_fp32.clear();
        eigvecsL_imag_fp32.clear();
        eigvecsR_cx32.clear();
        eigvecsL_cx32.clear();

        eigvals_real_fp64.clear();
        eigvals_imag_fp64.clear();
        eigvals_cx64.clear();
        eigvecsR_real_fp64.clear();
        eigvecsR_imag_fp64.clear();
        eigvecsL_real_fp64.clear();
        eigvecsL_imag_fp64.clear();
        eigvecsR_cx64.clear();
        eigvecsL_cx64.clear();
        meta = Meta();
    }

    bool solution::eigvecs_are_real() const { return meta.form == Form::SYMM and (meta.type == Type::FP32 or meta.type == Type::FP64); }

    bool solution::eigvals_are_real() const { return meta.form == Form::SYMM; }

    // std::type_index solution::get_eigvecs_type() const {
    //     if(eigvecs_are_real())
    //         return typeid(fp64);
    //     else
    //         return typeid(cx64);
    // }

    void solution::build_eigvecs_cx32() const {
        bool build_eigvecsR_cplx = eigvecsR_cx32.empty() and (not eigvecsR_real_fp32.empty() or not eigvecsR_imag_fp32.empty());
        bool build_eigvecsL_cplx = eigvecsL_cx32.empty() and (not eigvecsL_real_fp32.empty() or not eigvecsL_imag_fp32.empty());

        if(build_eigvecsR_cplx) {
            eigvecsR_cx32.resize(std::max(eigvecsR_real_fp32.size(), eigvecsR_imag_fp32.size()));
            for(size_t i = 0; i < eigvecsR_cx32.size(); i++) {
                if(not eigvecsR_real_fp32.empty() and not eigvecsR_imag_fp32.empty() and i < eigvecsR_real_fp32.size() and i < eigvecsR_imag_fp32.size())
                    eigvecsR_cx32[i] = cx32(eigvecsR_real_fp32[i], eigvecsR_imag_fp32[i]);
                else if(not eigvecsR_real_fp32.empty() and i < eigvecsR_real_fp32.size())
                    eigvecsR_cx32[i] = cx32(eigvecsR_real_fp32[i], 0.0);
                else if(not eigvecsR_imag_fp32.empty() and i < eigvecsR_imag_fp32.size())
                    eigvecsR_cx32[i] = cx32(0.0, eigvecsR_imag_fp32[i]);
            }
            eigvecsR_real_fp32.clear();
            eigvecsR_imag_fp32.clear();
        }
        if(build_eigvecsL_cplx) {
            eigvecsL_cx32.resize(std::max(eigvecsL_real_fp32.size(), eigvecsL_imag_fp32.size()));
            for(size_t i = 0; i < eigvecsL_cx32.size(); i++) {
                if(not eigvecsL_real_fp32.empty() and not eigvecsL_imag_fp32.empty() and i < eigvecsL_real_fp32.size() and i < eigvecsL_imag_fp32.size())
                    eigvecsL_cx32[i] = cx32(eigvecsL_real_fp32[i], eigvecsL_imag_fp32[i]);
                else if(not eigvecsL_real_fp32.empty() and i < eigvecsL_real_fp32.size())
                    eigvecsL_cx32[i] = cx32(eigvecsL_real_fp32[i], 0.0);
                else if(not eigvecsL_imag_fp32.empty() and i < eigvecsL_imag_fp32.size())
                    eigvecsL_cx32[i] = cx32(0.0, eigvecsL_imag_fp32[i]);
            }
            eigvecsL_real_fp32.clear();
            eigvecsL_imag_fp32.clear();
        }
    }

    void solution::build_eigvecs_fp32() const {
        bool build_eigvecsR_real = eigvecsR_real_fp32.empty() and not eigvecsR_cx32.empty();
        bool build_eigvecsL_real = eigvecsL_real_fp32.empty() and not eigvecsL_cx32.empty();

        if(build_eigvecsR_real) {
            eigvecsR_real_fp32.resize(eigvecsR_cx32.size());
            for(size_t i = 0; i < eigvecsR_real_fp32.size(); i++) {
                if(std::imag(eigvecsR_cx32[i]) > 1e-12f) throw std::runtime_error("Error building real eigvecR: Nonzero imaginary part");
                eigvecsR_real_fp32[i] = std::real(eigvecsR_cx32[i]);
            }
            eigvecsR_cx32.clear();
        }
        if(build_eigvecsL_real) {
            eigvecsL_real_fp32.resize(eigvecsL_cx32.size());
            for(size_t i = 0; i < eigvecsL_real_fp32.size(); i++) {
                if(std::imag(eigvecsL_cx32[i]) > 1e-12f) throw std::runtime_error("Error building real eigvecL: Nonzero imaginary part");
                eigvecsL_real_fp32[i] = std::real(eigvecsL_cx32[i]);
            }
            eigvecsL_cx32.clear();
        }
    }

    void solution::build_eigvals_cx32() const {
        bool build_cplx = eigvals_cx32.empty() and not eigvals_imag_fp32.empty() and eigvals_real_fp32.size() == eigvals_imag_fp32.size();
        if(build_cplx) {
            eigvals_cx32.resize(eigvals_real_fp32.size());
            for(size_t i = 0; i < eigvals_real_fp32.size(); i++) eigvals_cx32[i] = std::complex<double>(eigvals_real_fp32[i], eigvals_imag_fp32[i]);
            eigvals_real_fp32.clear();
            eigvals_imag_fp32.clear();
        }
    }

    void solution::build_eigvals_fp32() const {
        bool build_real = (eigvals_real_fp32.empty() or eigvals_imag_fp32.empty()) and not eigvals_cx32.empty();
        if(build_real) {
            eigvals_real_fp32.resize(eigvals_cx32.size());
            eigvals_imag_fp32.resize(eigvals_cx32.size());
            for(size_t i = 0; i < eigvals_cx32.size(); i++) {
                eigvals_real_fp32[i] = std::real(eigvals_cx32[i]);
                eigvals_imag_fp32[i] = std::imag(eigvals_cx32[i]);
            }
            eigvals_cx32.clear();
        }
    }

    void solution::build_eigvecs_cx64() const {
        bool build_eigvecsR_cplx = eigvecsR_cx64.empty() and (not eigvecsR_real_fp64.empty() or not eigvecsR_imag_fp64.empty());
        bool build_eigvecsL_cplx = eigvecsL_cx64.empty() and (not eigvecsL_real_fp64.empty() or not eigvecsL_imag_fp64.empty());

        if(build_eigvecsR_cplx) {
            eigvecsR_cx64.resize(std::max(eigvecsR_real_fp64.size(), eigvecsR_imag_fp64.size()));
            for(size_t i = 0; i < eigvecsR_cx64.size(); i++) {
                if(not eigvecsR_real_fp64.empty() and not eigvecsR_imag_fp64.empty() and i < eigvecsR_real_fp64.size() and i < eigvecsR_imag_fp64.size())
                    eigvecsR_cx64[i] = cx64(eigvecsR_real_fp64[i], eigvecsR_imag_fp64[i]);
                else if(not eigvecsR_real_fp64.empty() and i < eigvecsR_real_fp64.size())
                    eigvecsR_cx64[i] = cx64(eigvecsR_real_fp64[i], 0.0);
                else if(not eigvecsR_imag_fp64.empty() and i < eigvecsR_imag_fp64.size())
                    eigvecsR_cx64[i] = cx64(0.0, eigvecsR_imag_fp64[i]);
            }
            eigvecsR_real_fp64.clear();
            eigvecsR_imag_fp64.clear();
        }
        if(build_eigvecsL_cplx) {
            eigvecsL_cx64.resize(std::max(eigvecsL_real_fp64.size(), eigvecsL_imag_fp64.size()));
            for(size_t i = 0; i < eigvecsL_cx64.size(); i++) {
                if(not eigvecsL_real_fp64.empty() and not eigvecsL_imag_fp64.empty() and i < eigvecsL_real_fp64.size() and i < eigvecsL_imag_fp64.size())
                    eigvecsL_cx64[i] = cx64(eigvecsL_real_fp64[i], eigvecsL_imag_fp64[i]);
                else if(not eigvecsL_real_fp64.empty() and i < eigvecsL_real_fp64.size())
                    eigvecsL_cx64[i] = cx64(eigvecsL_real_fp64[i], 0.0);
                else if(not eigvecsL_imag_fp64.empty() and i < eigvecsL_imag_fp64.size())
                    eigvecsL_cx64[i] = cx64(0.0, eigvecsL_imag_fp64[i]);
            }
            eigvecsL_real_fp64.clear();
            eigvecsL_imag_fp64.clear();
        }
    }

    void solution::build_eigvecs_fp64() const {
        bool build_eigvecsR_real = eigvecsR_real_fp64.empty() and not eigvecsR_cx64.empty();
        bool build_eigvecsL_real = eigvecsL_real_fp64.empty() and not eigvecsL_cx64.empty();

        if(build_eigvecsR_real) {
            eigvecsR_real_fp64.resize(eigvecsR_cx64.size());
            for(size_t i = 0; i < eigvecsR_real_fp64.size(); i++) {
                if(std::imag(eigvecsR_cx64[i]) > 1e-12) throw std::runtime_error("Error building real eigvecR: Nonzero imaginary part");
                eigvecsR_real_fp64[i] = std::real(eigvecsR_cx64[i]);
            }
            eigvecsR_cx64.clear();
        }
        if(build_eigvecsL_real) {
            eigvecsL_real_fp64.resize(eigvecsL_cx64.size());
            for(size_t i = 0; i < eigvecsL_real_fp64.size(); i++) {
                if(std::imag(eigvecsL_cx64[i]) > 1e-12) throw std::runtime_error("Error building real eigvecL: Nonzero imaginary part");
                eigvecsL_real_fp64[i] = std::real(eigvecsL_cx64[i]);
            }
            eigvecsL_cx64.clear();
        }
    }

    void solution::build_eigvals_cx64() const {
        bool build_cplx = eigvals_cx64.empty() and not eigvals_imag_fp64.empty() and eigvals_real_fp64.size() == eigvals_imag_fp64.size();
        if(build_cplx) {
            eigvals_cx64.resize(eigvals_real_fp64.size());
            for(size_t i = 0; i < eigvals_real_fp64.size(); i++) eigvals_cx64[i] = cx64(eigvals_real_fp64[i], eigvals_imag_fp64[i]);
            eigvals_real_fp64.clear();
            eigvals_imag_fp64.clear();
        }
    }

    void solution::build_eigvals_fp64() const {
        bool build_real = (eigvals_real_fp64.empty() or eigvals_imag_fp64.empty()) and not eigvals_cx64.empty();
        if(build_real) {
            eigvals_real_fp64.resize(eigvals_cx64.size());
            eigvals_imag_fp64.resize(eigvals_cx64.size());
            for(size_t i = 0; i < eigvals_cx64.size(); i++) {
                eigvals_real_fp64[i] = std::real(eigvals_cx64[i]);
                eigvals_imag_fp64[i] = std::imag(eigvals_cx64[i]);
            }
            eigvals_cx64.clear();
        }
    }
}