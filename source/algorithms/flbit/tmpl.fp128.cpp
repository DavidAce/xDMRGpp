#include "../flbit.tmpl.impl.h"

using Scalar = fp128;

// template std::pair<StateFinite<Scalar>, AlgorithmStatus>
// flbit_tmpl::update_state(const size_t time_index, cx128 time_point,                      //
//                          const StateFinite<Scalar>                    &state_lbit_init,  //
//                          const std::vector<std::vector<qm::SwapGate>> &gates_tevo,       //
//                          const std::vector<std::vector<qm::Gate>>     &unitary_circuit,  //
//                          const AlgorithmStatus                        &status_init);
//
// template StateFinite<Scalar>
// flbit_tmpl::time_evolve_lbit_state(const StateFinite<Scalar>                      &state_lbit_init,  //
//                                    const std::vector<std::vector<qm::SwapGate>> &gates_tevo,         //
//                                    const AlgorithmStatus &status);
//
//
// template StateFinite<cx64>
// flbit_tmpl::transform_to_real_basis(const StateFinite<cx64> &state_lbit,                           //
//                                     const std::vector<std::vector<qm::Gate>> &unitary_circuit,     //
//                                     const AlgorithmStatus &status);
//
// template void flbit_tmpl::print_status(const StateFinite<Scalar> &state_real, const AlgorithmStatus &status);