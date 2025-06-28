#pragma once
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <array>
#include <complex>
#include <vector>

namespace Eigen {
    template<typename std::ptrdiff_t... Indices>
    struct Sizes;
}

namespace qm::spin {
    extern Eigen::MatrixXcd              gen_embedded_spin_operator(const Eigen::MatrixXcd &s, size_t at, size_t sites, bool reverse = false);
    extern std::vector<Eigen::MatrixXcd> gen_manybody_spins(const Eigen::MatrixXcd &s, size_t sites, bool reverse = false);

    namespace half {
        template<typename T>
        T get_sx() {
            T mat(2, 2);
            mat.setZero();
            mat(0, 1) = 1.0;
            mat(1, 0) = 1.0;
            return mat;
        }
        template<typename T>
        T get_sy() {
            T mat(2, 2);
            static_assert(std::is_same_v<typename T::Scalar, std::complex<double>>);
            mat.setZero();
            mat(0, 1) = -1.0i;
            mat(1, 0) = +1.0i;
            return mat;
        }
        template<typename T>
        T get_sz() {
            T mat(2, 2);
            mat.setZero();
            mat(0, 0) = 1.0;
            mat(1, 1) = -1.0;
            return mat;
        }
        template<typename T>
        T get_sp() {
            T mat(2, 2);
            mat.setZero();
            mat(0, 1) = 1.0;
            return mat;
        }
        template<typename T>
        T get_sm() {
            T mat(2, 2);
            mat.setZero();
            mat(1, 0) = 1.0;
            return mat;
        }
        template<typename T>
        T get_nu() {
            T mat(2, 2);
            mat.setZero();
            mat(0, 0) = 1.0;
            return mat;
        }
        template<typename T>
        T get_nd() {
            T mat(2, 2);
            mat.setZero();
            mat(1, 1) = 1.0;
            return mat;
        }
        template<typename T>
        T get_id() {
            T mat(2, 2);
            mat.setZero();
            mat(0, 0) = 1.0;
            mat(1, 1) = 1.0;
            return mat;
        }
        template<typename T>
        std::array<T, 2> get_sx_spinors() {
            std::array<T, 2> vecs{T(2), T(2)};
            vecs[0](0) = 1.0 / std::sqrt(2);
            vecs[0](1) = 1.0 / std::sqrt(2);
            vecs[1](0) = 1.0 / std::sqrt(2);
            vecs[1](1) = -1.0 / std::sqrt(2);
            return vecs;
        }
        template<typename T>
        std::array<T, 2> get_sy_spinors() {
            std::array<T, 2> vecs{T(2), T(2)};
            vecs[0](0) = 1.0 / std::sqrt(2);
            vecs[0](1) = 1.0i / std::sqrt(2);
            vecs[1](0) = 1.0 / std::sqrt(2);
            vecs[1](1) = -1.0i / std::sqrt(2);
            return vecs;
        }
        template<typename T>
        std::array<T, 2> get_sz_spinors() {
            std::array<T, 2> vecs{T(2), T(2)};
            vecs[0](0) = 1.0;
            vecs[0](1) = 0.0;
            vecs[1](0) = 0.0;
            vecs[1](1) = 1.0;
            return vecs;
        }
        namespace matrix {
            extern const Eigen::MatrixXcd                sx;
            extern const Eigen::MatrixXcd                sy;
            extern const Eigen::MatrixXcd                sz;
            extern const Eigen::MatrixXcd                sp;
            extern const Eigen::MatrixXcd                sm;
            extern const Eigen::MatrixXcd                nu;
            extern const Eigen::MatrixXcd                nd;
            extern const Eigen::MatrixXcd                id;
            extern const std::array<Eigen::VectorXcd, 2> sx_spinors;
            extern const std::array<Eigen::VectorXcd, 2> sy_spinors;
            extern const std::array<Eigen::VectorXcd, 2> sz_spinors;
        }
        namespace tensor {
            extern const Eigen::Tensor<cx64, 2>                sx;
            extern const Eigen::Tensor<cx64, 2>                sy;
            extern const Eigen::Tensor<cx64, 2>                sz;
            extern const Eigen::Tensor<cx64, 2>                sp;
            extern const Eigen::Tensor<cx64, 2>                sm;
            extern const Eigen::Tensor<cx64, 2>                nu;
            extern const Eigen::Tensor<cx64, 2>                nd;
            extern const Eigen::Tensor<cx64, 2>                id;
            extern const std::array<Eigen::Tensor<cx64, 1>, 2> sx_spinors;
            extern const std::array<Eigen::Tensor<cx64, 1>, 2> sy_spinors;
            extern const std::array<Eigen::Tensor<cx64, 1>, 2> sz_spinors;
            extern Eigen::Tensor<cx64, 1>                      get_spinor(std::string_view axis, int sign);
            extern Eigen::Tensor<cx64, 1>                      get_spinor(std::string_view axis);
            extern Eigen::Tensor<cx64, 2>                      get_pauli(std::string_view axis);
        }
        extern const Eigen::MatrixXcd          sx;
        extern const Eigen::MatrixXcd          sy;
        extern const Eigen::MatrixXcd          sz;
        extern const Eigen::MatrixXcd          sp;
        extern const Eigen::MatrixXcd          sm;
        extern const Eigen::MatrixXcd          nu;
        extern const Eigen::MatrixXcd          nd;
        extern const Eigen::MatrixXcd          id;
        extern std::array<Eigen::VectorXcd, 2> sx_spinors;
        extern std::array<Eigen::VectorXcd, 2> sy_spinors;
        extern std::array<Eigen::VectorXcd, 2> sz_spinors;
        extern std::vector<Eigen::MatrixXcd>   SX;
        extern std::vector<Eigen::MatrixXcd>   SY;
        extern std::vector<Eigen::MatrixXcd>   SZ;
        extern std::vector<Eigen::MatrixXcd>   II;
        extern Eigen::MatrixXcd                gen_embedded_spin_half_operator(const Eigen::MatrixXcd &s, size_t at, size_t sites, bool swap = false);
        extern std::vector<Eigen::Matrix4cd>   gen_twobody_spins(const Eigen::MatrixXcd &s, bool swap = false);

        inline static constexpr std::array valid_axis_str = {"x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z", "i", "id"};
        extern bool                        is_valid_axis(std::string_view axis);
        extern int                         get_sign(std::string_view axis);
        extern std::string_view            get_axis_unsigned(std::string_view axis);
        extern Eigen::VectorXcd            get_spinor(std::string_view axis, int sign);
        extern Eigen::VectorXcd            get_spinor(std::string_view axis);
        extern Eigen::MatrixXcd            get_pauli(std::string_view axis);
    }

    namespace one {
        extern Eigen::Matrix3cd              sx;
        extern Eigen::Matrix3cd              sy;
        extern Eigen::Matrix3cd              sz;
        extern Eigen::Matrix3cd              id;
        extern std::vector<Eigen::MatrixXcd> SX;
        extern std::vector<Eigen::MatrixXcd> SY;
        extern std::vector<Eigen::MatrixXcd> SZ;
        extern std::vector<Eigen::MatrixXcd> II;
        extern std::vector<Eigen::MatrixXcd> gen_twobody_spins(const Eigen::Matrix3cd &s, bool swap = false);
    }

}