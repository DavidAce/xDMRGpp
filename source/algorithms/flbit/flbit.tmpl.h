#pragma once
#include "../AlgorithmFinite.h"
#include "qm/gate.h"
#include "qm/lbit.h"
#include <deque>

/*!
// * \brief Class that runs the finite LBIT algorithm.
 */

template<typename Scalar>
class StateFinite;

namespace flbit_tmpl {
    template<typename Scalar>
    std::pair<StateFinite<Scalar>, AlgorithmStatus> update_state(const size_t time_index, cx128 time_point, const StateFinite<Scalar> &state_lbit_init,
                                                                 const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                                                 const std::vector<std::vector<qm::Gate>> &unitary_circuit, const AlgorithmStatus &status_init);

    // extern std::vector<std::vector<qm::SwapGate>> get_time_evolution_gates(const cx128                                  &time_point,
    // const std::vector<std::vector<qm::SwapGate>> &ham_swap_gates);

    inline std::vector<std::vector<qm::SwapGate>> get_time_evolution_gates(const cx128                                  &time_point,
                                                                                       const std::vector<std::vector<qm::SwapGate>> &ham_swap_gates) {
        auto t_upd = tid::tic_scope("gen_swap_gates", tid::level::normal);
        tools::log->debug("Updating time evolution swap gates to t = {:.2e}", fp(time_point));
        auto time_swap_gates = std::vector<std::vector<qm::SwapGate>>();
        for(const auto &hams : ham_swap_gates) { // ham_swap_gates contain 1body, 2body and 3body hamiltonian terms (each as a layer of swap gates)
            time_swap_gates.emplace_back(qm::lbit::get_time_evolution_swap_gates(time_point, hams));
        }
        return time_swap_gates;
    }

    template<typename Scalar>
    StateFinite<Scalar> time_evolve_lbit_state(const StateFinite<Scalar> &state_lbit_init, const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                               const AlgorithmStatus &status);

    template<typename Scalar>
    StateFinite<Scalar> transform_to_real_basis(const StateFinite<Scalar> &state_lbit, const std::vector<std::vector<qm::Gate>> &unitary_circuit,
                                                const AlgorithmStatus &status);

    extern AlgorithmStatus check_convergence(const AlgorithmStatus &status);

    template<typename Scalar>
    void print_status(const StateFinite<Scalar> &state_real, const AlgorithmStatus &status);

}