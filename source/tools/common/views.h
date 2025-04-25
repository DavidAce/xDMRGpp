#pragma once
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
template<typename Scalar>
class StateInfinite;
template<typename Scalar>
class StateFinite;

namespace tools::common {
    template<typename Scalar>
    struct views {
        inline static Eigen::Tensor<Scalar, 4> theta                = {};
        inline static Eigen::Tensor<Scalar, 4> theta_evn_normalized = {};
        inline static Eigen::Tensor<Scalar, 4> theta_odd_normalized = {};
        inline static Eigen::Tensor<Scalar, 4> theta_sw             = {};
        inline static Eigen::Tensor<Scalar, 3> LAGA                 = {};
        inline static Eigen::Tensor<Scalar, 3> LCGB                 = {};
        inline static Eigen::Tensor<Scalar, 2> l_evn                = {};
        inline static Eigen::Tensor<Scalar, 2> r_evn                = {};
        inline static Eigen::Tensor<Scalar, 2> l_odd                = {};
        inline static Eigen::Tensor<Scalar, 2> r_odd                = {};
        inline static Eigen::Tensor<Scalar, 4> transfer_matrix_LAGA = {};
        inline static Eigen::Tensor<Scalar, 4> transfer_matrix_LCGB = {};
        inline static Eigen::Tensor<Scalar, 4> transfer_matrix_evn  = {};
        inline static Eigen::Tensor<Scalar, 4> transfer_matrix_odd  = {};
        inline static bool                     components_computed  = false;

        static void                     compute_mps_components(const StateInfinite<Scalar> &state);
        static Eigen::Tensor<Scalar, 4> get_theta(const StateFinite<Scalar> &state, Scalar norm = 1.0); /*!< Returns rank 4 tensor \f$\Theta\f$.*/
        static Eigen::Tensor<Scalar, 4> get_theta(const StateInfinite<Scalar> &state, Scalar norm = 1.0); /*!< Returns rank 4 tensor \f$\Theta\f$.*/
        static Eigen::Tensor<Scalar, 4> get_theta_swapped(const StateInfinite<Scalar> &state, Scalar norm = 1.0); /*!< Returns rank 4 tensor \f$\Theta\f$, with A and B swapped.*/
        static Eigen::Tensor<Scalar, 4> get_theta_evn(const StateInfinite<Scalar> &state, Scalar norm = 1.0); /*!< Returns rank 4 tensor \f$\Theta\f$.*/
        static Eigen::Tensor<Scalar, 4> get_theta_odd(const StateInfinite<Scalar> &state, Scalar                       norm = 1.0); /*!< Returns rank 4 tensor \f$\Theta\f$, with A and B swapped.*/
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_zero(const StateInfinite<Scalar> &state);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_LBGA(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_GALC(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_GBLB(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_LCGB(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_theta_evn(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_theta_odd(const StateInfinite<Scalar> &state, Scalar norm = 1.0);
        static Eigen::Tensor<Scalar, 4> get_transfer_matrix_AB(const StateInfinite<Scalar> &state, int p);
    };

}