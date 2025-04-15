#include "qm/time.h"
#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "math/linalg.h"
#include "math/tenx.h"
#include "tools/common/log.h"
#include <fmt/ranges.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace qm::time {
    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                   const Eigen::Tensor<Scalar, 2> &h_odd) {
        auto h_evn_matrix = tenx::MatrixMap(h_evn);
        auto h_odd_matrix = tenx::MatrixMap(h_odd);
        auto dt           = cx64(static_cast<fp64>(delta_t.real()), static_cast<fp64>(delta_t.imag()));
        return {
            tenx::TensorCast((static_cast<cx64>(-1.0i * dt) * h_evn_matrix).exp()), // exp(-i dt H)
            tenx::TensorCast((static_cast<cx64>(-1.0i * dt) * h_odd_matrix).exp())  // exp(-i dt H)
        };
    }
    template std::vector<Eigen::Tensor<cx64, 2>> Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<cx64, 2> &h_evn,
                                                                          const Eigen::Tensor<cx64, 2> &h_odd);

    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                   const Eigen::Tensor<Scalar, 2> &h_odd) {
        auto h_evn_matrix = tenx::MatrixMap(h_evn);
        auto h_odd_matrix = tenx::MatrixMap(h_odd);
        auto dt           = cx64(static_cast<fp64>(delta_t.real()), static_cast<fp64>(delta_t.imag()));

        return {tenx::TensorCast((static_cast<cx64>(-1.0i * dt) * h_evn_matrix / 2.0).exp()),
                tenx::TensorCast((static_cast<cx64>(-1.0i * dt) * h_odd_matrix).exp()),
                tenx::TensorCast((static_cast<cx64>(-1.0i * dt) * h_evn_matrix / 2.0).exp())};
    }
    template std::vector<Eigen::Tensor<cx64, 2>> Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<cx64, 2> &h_evn,
                                                                          const Eigen::Tensor<cx64, 2> &h_odd);

    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 2>> Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd)
    /*!
     * Implementation based on
     * Janke, W., & Sauer, T. (1992).
     * Properties of higher-order Trotter formulas.
     * Physics Letters A, 165(3), 199–205.
     * https://doi.org/10.1016/0375-9601(92)90035-K
     *
     */
    {
        auto   h_evn_matrix = tenx::MatrixMap(h_evn);
        auto   h_odd_matrix = tenx::MatrixMap(h_odd);
        double cbrt2        = pow(2.0, 1.0 / 3.0);
        double beta1        = 1.0 / (2.0 - cbrt2);
        double beta2        = -cbrt2 * beta1;
        double alph1        = 0.5 * beta1;
        double alph2        = (1.0 - cbrt2) / 2.0 * beta1;
        auto   dt           = cx64(static_cast<fp64>(delta_t.real()), static_cast<fp64>(delta_t.imag()));

        std::vector<Eigen::Tensor<cx64, 2>> temp;
        temp.emplace_back(tenx::TensorCast((alph1 * static_cast<Scalar>(-1.0i * dt) * h_evn_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((beta1 * static_cast<Scalar>(-1.0i * dt) * h_odd_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((alph2 * static_cast<Scalar>(-1.0i * dt) * h_evn_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((beta2 * static_cast<Scalar>(-1.0i * dt) * h_odd_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((alph2 * static_cast<Scalar>(-1.0i * dt) * h_evn_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((beta1 * static_cast<Scalar>(-1.0i * dt) * h_odd_matrix).exp()));
        temp.emplace_back(tenx::TensorCast((alph1 * static_cast<Scalar>(-1.0i * dt) * h_evn_matrix).exp()));
        return temp;
    }
    template std::vector<Eigen::Tensor<cx64, 2>> Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<cx64, 2> &h_evn,
                                                                          const Eigen::Tensor<cx64, 2> &h_odd);

    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 2>> get_twosite_time_evolution_operators(cx128 delta_t, size_t susuki_trotter_order,
                                                                               const Eigen::Tensor<Scalar, 2> &h_evn, const Eigen::Tensor<Scalar, 2> &h_odd)
    /*! Returns a set of 2-site unitary gates, using Suzuki Trotter decomposition to order 1, 2 or 3.
     * These gates need to be applied to the MPS one at a time with a swap in between.
     */
    {
        switch(susuki_trotter_order) {
            case 1: return Suzuki_Trotter_1st_order(delta_t, h_evn, h_odd);
            case 2: return Suzuki_Trotter_2nd_order(delta_t, h_evn, h_odd);
            case 4: return Suzuki_Trotter_4th_order(delta_t, h_evn, h_odd);
            default: return Suzuki_Trotter_2nd_order(delta_t, h_evn, h_odd);
        }
    }
    template<typename Scalar>
    std::vector<Eigen::Tensor<Scalar, 2>> compute_G(const cx128 a, size_t susuki_trotter_order, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                    const Eigen::Tensor<Scalar, 2> &h_odd)
    /*! Returns the moment generating function, or characteristic function (if a is imaginary) for the Hamiltonian as a rank 2 tensor.
     *  The legs contain two physical spin indices each
    *   G := exp(iaM) or exp(aM), where a is a small parameter and M is an MPO.
    *   Note that G(-a) = G(a)* if  exp(iaM) !
    *
    @verbatim
                     0
                     |
                [ exp(aH) ]
                     |
                     1
    @endverbatim
    */
    {
        tools::log->warn("compute_G(...): Convention has changed: delta_t, or a, are now multiplied by [-i] in exponentials."
                         " This function may not have been adjusted to the new convention");
        return get_twosite_time_evolution_operators(a, susuki_trotter_order, h_evn, h_odd);
    }
    template std::vector<Eigen::Tensor<cx64, 2>> compute_G(const cx128 a, size_t susuki_trotter_order, const Eigen::Tensor<cx64, 2> &h_evn,
                                                           const Eigen::Tensor<cx64, 2> &h_odd);

    std::pair<std::vector<qm::Gate>, std::vector<qm::Gate>> get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite) {
        /* Here we do a second-order Suzuki-Trotter decomposition which holds for n-site hamiltonians as described
         * here https://tensornetwork.org/mps/algorithms/timeevo/tebd.html
         * For instance,
         *      H = Sum_a^n H_a
         * where each H_a is a sum of n-site terms.
         *
         * The second-order Suzuki-Trotter decomposition them becomes
         *
         * U2(d) = Prod_{a=1}^n exp(-i[d/2]H_a) Prod_{a=n}^1 exp(-i[d/2]H_a)
         *
         * So this is just the layers applied in reversed order!
         * We return these as a pair of gate layers, and both need to be applied normally for the time evolution
         * to take place
         *
         */

        std::vector<Gate> time_evolution_gates_forward;
        std::vector<Gate> time_evolution_gates_reverse;
        time_evolution_gates_forward.reserve(hams_nsite.size());
        time_evolution_gates_reverse.reserve(hams_nsite.size());
        auto dt = cx64(static_cast<fp64>(delta_t.real()), static_cast<fp64>(delta_t.imag()));

        // Generate first forward layer
        for(auto &h : hams_nsite) {
            time_evolution_gates_forward.emplace_back(h.exp(-1.0i * dt * 0.5)); // exp(-i * delta_t * h)
        }
        // Generate second reversed layer
        for(auto &h : iter::reverse(hams_nsite)) {
            time_evolution_gates_reverse.emplace_back(h.exp(-1.0i * dt * 0.5)); // exp(-i * delta_t * h)
        }

        if constexpr(settings::debug) {
            // Sanity checks
            if(std::imag(delta_t) == 0) {
                for(auto &t : time_evolution_gates_forward)
                    if(not t.isUnitary(Eigen::NumTraits<double>::dummy_precision() * static_cast<double>(t.op.dimension(0)))) {
                        throw except::runtime_error("Time evolution operator at pos {} is not unitary:\n{}", t.pos, linalg::tensor::to_string(t.op));
                    }
                for(auto &t : time_evolution_gates_reverse)
                    if(not t.isUnitary(Eigen::NumTraits<double>::dummy_precision() * static_cast<double>(t.op.dimension(0)))) {
                        throw except::runtime_error("Time evolution operator at pos {} is not unitary:\n{}", t.pos, linalg::tensor::to_string(t.op));
                    }
            }
        }

        return {time_evolution_gates_forward, time_evolution_gates_reverse};
    }

}
