#pragma once
#include "AlgorithmFinite.h"
#include "qm/gate.h"
#include "qm/lbit.h"
#include <deque>

/*!
// * \brief Class that runs the finite LBIT algorithm.
 */

template<typename Scalar>
class StateFinite;

namespace flbit_impl {
    template<typename Scalar>
    std::pair<StateFinite<Scalar>, AlgorithmStatus> update_state(const size_t time_index, cx128 time_point, const StateFinite<Scalar> &state_lbit_init,
                                                                 const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                                                 const std::vector<std::vector<qm::Gate>> &unitary_circuit, const AlgorithmStatus &status_init);
    std::vector<std::vector<qm::SwapGate>> get_time_evolution_gates(const cx128 &time_point, const std::vector<std::vector<qm::SwapGate>> &ham_swap_gates);
    template<typename Scalar>
    StateFinite<Scalar> time_evolve_lbit_state(const StateFinite<Scalar> &state_lbit_init, const std::vector<std::vector<qm::SwapGate>> &gates_tevo,
                                               const AlgorithmStatus &status);
    template<typename Scalar>
    StateFinite<Scalar> transform_to_real_basis(const StateFinite<Scalar> &state_lbit, const std::vector<std::vector<qm::Gate>> &unitary_circuit,
                                                const AlgorithmStatus &status);
    AlgorithmStatus     check_convergence(const AlgorithmStatus &status);
    template<typename Scalar>
    void print_status(const StateFinite<Scalar> &state_real, const AlgorithmStatus &status);

}