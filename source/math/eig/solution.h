#pragma once
#include "enums.h"
#include "math/float.h"
#include <deque>
#include <typeindex>
#include <vector>

namespace eig {
    class solver;
    template<typename MatrixType>
    class solver_arpack;

    class solution {
        public:
        friend class solver;
        template<typename MatrixType>
        friend class solver_arpack;

        private:
        mutable std::vector<fp32> eigvals_real_fp32;
        mutable std::vector<fp32> eigvals_imag_fp32;
        mutable std::vector<cx32> eigvals_cx32;

        mutable std::vector<fp64> eigvals_real_fp64;
        mutable std::vector<fp64> eigvals_imag_fp64;
        mutable std::vector<cx64> eigvals_cx64;

        mutable std::vector<fp128> eigvals_real_fp128;
        mutable std::vector<fp128> eigvals_imag_fp128;
        mutable std::vector<cx128> eigvals_cx128;

        mutable std::vector<fp32> eigvecsR_real_fp32;
        mutable std::vector<fp32> eigvecsR_imag_fp32;
        mutable std::vector<fp32> eigvecsL_real_fp32;
        mutable std::vector<fp32> eigvecsL_imag_fp32;
        mutable std::vector<cx32> eigvecsR_cx32;
        mutable std::vector<cx32> eigvecsL_cx32;

        mutable std::vector<fp64> eigvecsR_real_fp64;
        mutable std::vector<fp64> eigvecsR_imag_fp64;
        mutable std::vector<fp64> eigvecsL_real_fp64;
        mutable std::vector<fp64> eigvecsL_imag_fp64;
        mutable std::vector<cx64> eigvecsR_cx64;
        mutable std::vector<cx64> eigvecsL_cx64;

        mutable std::vector<fp128> eigvecsR_real_fp128;
        mutable std::vector<fp128> eigvecsR_imag_fp128;
        mutable std::vector<fp128> eigvecsL_real_fp128;
        mutable std::vector<fp128> eigvecsL_imag_fp128;
        mutable std::vector<cx128> eigvecsR_cx128;
        mutable std::vector<cx128> eigvecsL_cx128;

        void build_eigvecs_cx32() const;
        void build_eigvecs_fp32() const;
        void build_eigvals_cx32() const;
        void build_eigvals_fp32() const;

        void build_eigvecs_cx64() const;
        void build_eigvecs_fp64() const;
        void build_eigvals_cx64() const;
        void build_eigvals_fp64() const;

        void build_eigvecs_cx128() const;
        void build_eigvecs_fp128() const;
        void build_eigvals_cx128() const;
        void build_eigvals_fp128() const;

        struct Meta {
            eig::size_type rows           = 0;
            eig::size_type cols           = 0;
            int            iter           = 0; // Number of iterations reported by solver
            int            nev            = 0; // Requested eigenvectors. aka cols
            int            nev_converged  = 0; // Converged eigenvectors
            eig::size_type n              = 0; // Linear dimension of the input matrix to diagonalize, aka rows.
            int            ncv            = 0;
            double         tol            = 0;
            long           num_mv         = 0; // Number of matrix-vector multiplications counted
            long           num_pc         = 0; // Number of preconditioner operations y = A⁻¹x counted
            long           num_op         = 0; // Number of operations, for instance in primme if maxblocksize > 1
            double         time_total     = 0;
            double         time_mv        = 0;
            double         time_op        = 0;
            double         time_pc        = 0;
            double         time_prep      = 0;
            bool           eigvals_found  = false;
            bool           eigvecsR_found = false;
            bool           eigvecsL_found = false;
            bool           arnoldi_found  = false;

            std::vector<double>  residual_norms = {}; /*!< For each eigenpair, stores the norm of its residual_norm vector (Ax - λx) */
            double               last_eval      = std::numeric_limits<double>::quiet_NaN();
            double               last_rnorm     = std::numeric_limits<double>::quiet_NaN();
            double               last_grad_max  = std::numeric_limits<double>::quiet_NaN();
            long                 last_grad_iter = 0;
            double               last_grad_time = 0;
            long                 last_conv_iter = 0;
            long                 last_conv_mvec = 0;
            double               last_conv_time = 0;
            long                 last_log_iter  = 0;
            double               last_log_time  = 0;
            std::deque<double>   recent_evals;
            std::deque<double>   recent_rnorms;
            bool                 subspace_ok = false; // Used for subspace convergence check in hybrid xdmrg++-x
            double               problemNorm = std::numeric_limits<double>::quiet_NaN();
            Form                 form;
            Type                 type;
            std::string          tag;
            std::string          ritz;
            std::complex<double> sigma = std::complex<double>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        };

        public:
        Meta meta;

        template<typename Scalar, Side side = Side::R>
        std::vector<Scalar> &get_eigvecs() const;

        template<typename Scalar>
        std::vector<Scalar> &get_eigvecs(Side side = Side::R) const;

        template<Form form, Type type, Side side = Side::R>
        auto &get_eigvecs() const {
            if constexpr(type == Type::FP32) {
                if constexpr(form == Form::SYMM) return get_eigvecs<fp32, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx32, side>();
            }
            if constexpr(type == Type::FP64) {
                if constexpr(form == Form::SYMM) return get_eigvecs<fp64, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx64, side>();
            }
            if constexpr(type == Type::FP128) {
                if constexpr(form == Form::SYMM) return get_eigvecs<fp128, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx128, side>();
            }
            if constexpr(type == Type::CX32) {
                if constexpr(form == Form::SYMM) return get_eigvecs<cx32, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx32, side>();
            }
            if constexpr(type == Type::CX64) {
                if constexpr(form == Form::SYMM) return get_eigvecs<cx64, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx64, side>();
            }
            if constexpr(type == Type::CX128) {
                if constexpr(form == Form::SYMM) return get_eigvecs<cx128, side>();
                if constexpr(form == Form::NSYM) return get_eigvecs<cx128, side>();
            }
        }

        template<Type type, Form form, Side side = Side::R>
        auto &get_eigvecs() const {
            return get_eigvecs<form, type, side>();
        }

        template<typename Scalar, Form form, Side side = Side::R>
        std::vector<Scalar> &get_eigvecs() const;

        template<typename Scalar>
        std::vector<Scalar> &get_eigvals() const;

        template<Form form, Type type>
        auto &get_eigvals() const {
            if constexpr(form == Form::SYMM) {
                if constexpr(type == Type::FP32 or type == Type::CX32) return get_eigvals<fp32>();
                if constexpr(type == Type::FP64 or type == Type::CX64) return get_eigvals<fp64>();
                if constexpr(type == Type::FP128 or type == Type::CX128) return get_eigvals<fp128>();
            }
            if constexpr(form == Form::NSYM) {
                if constexpr(type == Type::FP32 or type == Type::CX32) return get_eigvals<cx32>();
                if constexpr(type == Type::FP64 or type == Type::CX64) return get_eigvals<cx64>();
                if constexpr(type == Type::FP128 or type == Type::CX128) return get_eigvals<cx128>();
            }
        }

        const std::vector<double> &get_resnorms() const;

        void reset();

        bool eigvecs_are_real() const;

        bool eigvals_are_real() const;

        std::type_index get_eigvecs_type() const;
    };
}
