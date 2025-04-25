#pragma once
#include "gate.h"
#include "math/float.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <complex>
#include <vector>

namespace qm::time {
    /* clang-format off */
    template <typename T, typename Scalar>
    requires sfinae::is_std_complex_v<T>
     extern std::vector<Eigen::Tensor<T, 2>> Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename T, typename Scalar>
    requires sfinae::is_std_complex_v<T>
    extern std::vector<Eigen::Tensor<T, 2>> Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);
    template <typename T, typename Scalar>
    requires sfinae::is_std_complex_v<T>
    extern std::vector<Eigen::Tensor<T, 2>> Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd);


    extern std::pair<std::vector<qm::Gate>, std::vector<qm::Gate>> get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite);
    /* clang-format on */

    /*! Returns a set of 2-site unitary gates, using Suzuki Trotter decomposition to order 1, 2 or 3.
     * These gates need to be applied to the MPS one at a time with a swap in between.
     */
    template<typename T, typename Scalar>
    requires sfinae::is_std_complex_v<T>
    std::vector<Eigen::Tensor<T, 2>> get_twosite_time_evolution_operators(cx128 delta_t, size_t susuki_trotter_order, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                          const Eigen::Tensor<Scalar, 2> &h_odd) {
        switch(susuki_trotter_order) {
            case 1: return Suzuki_Trotter_1st_order<T>(delta_t, h_evn, h_odd);
            case 2: return Suzuki_Trotter_2nd_order<T>(delta_t, h_evn, h_odd);
            case 4: return Suzuki_Trotter_4th_order<T>(delta_t, h_evn, h_odd);
            default: return Suzuki_Trotter_2nd_order<T>(delta_t, h_evn, h_odd);
        }
    }
    /*! Returns the moment generating function, or characteristic function (if a is imaginary) for the Hamiltonian as a rank 2 tensor.
     *  The legs contain two physical spin indices each
    *   G := exp(iaM) or exp(aM), where a is a small parameter and M is an MPO.
    *   Note that G(-a) = G(a) if exp(iaM) !
    *
    @verbatim
                     0
                     |
                [ exp(aH) ]
                     |
                     1
    @endverbatim
    */
    template<typename T, typename Scalar>
    requires sfinae::is_std_complex_v<T>
    std::vector<Eigen::Tensor<T, 2>> compute_G(const cx128 a, size_t susuki_trotter_order, const Eigen::Tensor<Scalar, 2> &h_evn,
                                               const Eigen::Tensor<Scalar, 2> &h_odd)

    {
        // tools::log->warn("compute_G(...): Convention has changed: delta_t, or a, are now multiplied by [-i] in exponentials."
        //                  " This function may not have been adjusted to the new convention");
        return get_twosite_time_evolution_operators<T>(a, susuki_trotter_order, h_evn, h_odd);
    }

}