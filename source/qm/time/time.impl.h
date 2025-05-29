#include "config/debug.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/fmt_custom.h"
#include "math/linalg/tensor/to_string.h"
#include "math/tenx.h"
#include "qm/time.h"
#include "tools/common/log.h"
#include <fmt/ranges.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

template<typename T, typename Scalar>
requires sfinae::is_std_complex_v<T>
std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_1st_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                    const Eigen::Tensor<Scalar, 2> &h_odd) {
    using RealT       = decltype(std::real(std::declval<T>()));
    auto h_evn_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_evn));
    auto h_odd_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_odd));
    T    idt          = static_cast<T>(-1.0i) * T(static_cast<RealT>(std::real(delta_t)), static_cast<RealT>(std::imag(delta_t)));

    std::vector<Eigen::Tensor<T, 2>> temp;
    temp.emplace_back(tenx::TensorCast((idt * h_evn_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((idt * h_odd_matrix).exp()));
    return temp;
}

template<typename T, typename Scalar>
requires sfinae::is_std_complex_v<T>
std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_2nd_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                    const Eigen::Tensor<Scalar, 2> &h_odd) {
    using RealT       = decltype(std::real(std::declval<T>()));
    auto h_evn_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_evn));
    auto h_odd_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_odd));
    T    idt          = static_cast<T>(-1.0i) * T(static_cast<RealT>(std::real(delta_t)), static_cast<RealT>(std::imag(delta_t)));

    return {tenx::TensorCast(idt * h_evn_matrix / RealT{2}).exp(), //
            tenx::TensorCast(idt * h_odd_matrix).exp(),            //
            tenx::TensorCast(idt * h_evn_matrix / RealT{2}).exp()};
}

/*!
 * Implementation based on
 * Janke, W., & Sauer, T. (1992).
 * Properties of higher-order Trotter formulas.
 * Physics Letters A, 165(3), 199–205.
 * https://doi.org/10.1016/0375-9601(92)90035-K
 *
 */
template<typename T, typename Scalar>
requires sfinae::is_std_complex_v<T>
std::vector<Eigen::Tensor<T, 2>> qm::time::Suzuki_Trotter_4th_order(cx128 delta_t, const Eigen::Tensor<Scalar, 2> &h_evn,
                                                                    const Eigen::Tensor<Scalar, 2> &h_odd) {
    using RealT = decltype(std::real(std::declval<T>()));

    auto  h_evn_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_evn));
    auto  h_odd_matrix = tenx::asScalarType<T>(tenx::MatrixMap(h_odd));
    RealT cbrt2        = pow<RealT>(RealT{2}, RealT{1} / RealT{3});
    RealT beta1        = static_cast<RealT>(RealT{1} / (RealT{2} - cbrt2));
    RealT beta2        = static_cast<RealT>(-cbrt2 * beta1);
    RealT alph1        = static_cast<RealT>(RealT{0.5} * beta1);
    RealT alph2        = static_cast<RealT>((RealT{1} - cbrt2) / RealT{2} * beta1);
    T     idt          = static_cast<T>(-1.0i) * T(static_cast<RealT>(std::real(delta_t)), static_cast<RealT>(std::imag(delta_t)));

    std::vector<Eigen::Tensor<T, 2>> temp;
    temp.emplace_back(tenx::TensorCast((alph1 * idt * h_evn_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((beta1 * idt * h_odd_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((alph2 * idt * h_evn_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((beta2 * idt * h_odd_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((alph2 * idt * h_evn_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((beta1 * idt * h_odd_matrix).exp()));
    temp.emplace_back(tenx::TensorCast((alph1 * idt * h_evn_matrix).exp()));
    return temp;
}

inline std::pair<std::vector<qm::Gate>, std::vector<qm::Gate>> qm::time::get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite) {
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
