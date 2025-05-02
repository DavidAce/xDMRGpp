#pragma once

#include "config/enums.h"
#include "math/float.h"
#include "math/svd/config.h"
#include "math/tenx/fwd_decl.h"
#include <complex>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tensors/state/StateFinite.h>
#include <vector>
namespace qm {
    class Gate;
    class SwapGate;
}

/* clang-format off */
template<typename Scalar> class StateFinite;
template<typename Scalar> class MpsSite;
enum class BitOrder {AsIs, Reverse};
namespace tools{
    extern std::string get_bitfield(size_t nbits, const std::string &pattern, BitOrder bitOrder = BitOrder::AsIs);
    extern std::string get_bitfield(size_t nbits, size_t pattern, BitOrder bitOrder = BitOrder::AsIs);
}


namespace tools::finite::mps {
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));

    /* clang-format off */

    template<typename CalcType, typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar,1> mps2tensor   (const std::vector<std::unique_ptr<MpsSite<Scalar>>> & mps_sites, std::string_view name);
    template<typename CalcType, typename Scalar> [[nodiscard]] extern Eigen::Tensor<Scalar,1> mps2tensor   (const StateFinite<Scalar> & state);

    template<typename Scalar> extern size_t move_center_point_single_site      (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t move_center_point                  (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t move_center_point_to_pos           (StateFinite<Scalar>& state, long pos, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t move_center_point_to_pos_dir       (StateFinite<Scalar>& state, long pos, int dir, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t move_center_point_to_inward_edge   (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t move_center_point_to_middle        (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern size_t merge_multisite_mps                (StateFinite<Scalar>& state, const Eigen::Tensor<Scalar,3> & multisite_mps, const std::vector<size_t> & sites, long center_position, MergeEvent mevent, std::optional<svd::config> svd_cfg = std::nullopt, std::optional<LogPolicy> logPolicy = std::nullopt);
    template<typename Scalar> extern bool normalize_state                      (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt, NormPolicy norm_policy = NormPolicy::IFNEEDED);
    template<typename Scalar> extern void initialize_state                     (StateFinite<Scalar>& state, StateInit state_type, StateInitType type, std::string_view sector, bool use_eigenspinors, long bond_lim, std::string & pattern);
    template<typename Scalar> extern StateFinite<Scalar> add_states            (const StateFinite<Scalar>& stateA, const StateFinite<Scalar>& stateB);

    template<typename Scalar> extern void apply_random_paulis                  (StateFinite<Scalar>& state, const std::vector<Eigen::Matrix2cd> & paulimatrices);
    template<typename Scalar> extern void apply_random_paulis                  (StateFinite<Scalar>& state, const std::vector<std::string> & paulistrings);
    template<typename Scalar> extern void truncate_all_sites                   (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern void truncate_active_sites                (StateFinite<Scalar>& state, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar> extern void truncate_next_sites                  (StateFinite<Scalar>& state, size_t num_sites = 4, std::optional<svd::config> svd_cfg = std::nullopt);

    template<typename GateType>
    extern std::vector<size_t>                              generate_gate_sequence               (long state_length, long state_position, const std::vector<GateType> &gates, CircuitOp cop, bool range_long_to_short = false);
    template<typename Scalar>   extern void                 apply_gate                           (StateFinite<Scalar>& state, const qm::Gate & gate, GateOp gop, GateMove gmov, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_gates                          (StateFinite<Scalar>& state, const std::vector<Eigen::Tensor<Scalar,2>> & nsite_tensors, size_t gate_size, CircuitOp cop, bool moveback = true, GateMove gm = GateMove::AUTO, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_gates                          (StateFinite<Scalar>& state, const std::vector<qm::Gate> & gates,  CircuitOp cop, bool moveback = true, GateMove gm = GateMove::AUTO, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_gates_old                      (StateFinite<Scalar>& state, const std::vector<qm::Gate> & gates,  CircuitOp cop, bool moveback = true,  std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_circuit                        (StateFinite<Scalar>& state, const std::vector<std::vector<qm::Gate>> & gates, CircuitOp gop, bool moveback = true, GateMove gm = GateMove::AUTO, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 swap_sites                           (StateFinite<Scalar>& state, size_t posL, size_t posR, std::vector<size_t> & sites, GateMove gm, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_swap_gate                      (StateFinite<Scalar>& state, const qm::SwapGate & gate, GateOp gop, std::vector<size_t> & sites, GateMove gm, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_swap_gates                     (StateFinite<Scalar>& state, std::vector<qm::SwapGate> & gates, CircuitOp cop, GateMove gm = GateMove::AUTO, std::optional<svd::config> svd_cfg = std::nullopt);
    template<typename Scalar>   extern void                 apply_swap_gates                     (StateFinite<Scalar>& state, const std::vector<qm::SwapGate> & gates, CircuitOp cop, GateMove gm = GateMove::AUTO, std::optional<svd::config> svd_cfg = std::nullopt);


    namespace init{
        inline std::set<size_t> used_bitfields;
        extern bool bitfield_is_valid (size_t bitfield);
        extern std::vector<long> get_valid_bond_dimensions(size_t sizeplusone, long spin_dim, long bond_lim);

        template<typename Scalar> extern void random_product_state        (StateFinite<Scalar>& state, StateInitType type, std::string_view axis, bool use_eigenspinors, std::string &pattern);
        template<typename Scalar> extern void random_entangled_state      (StateFinite<Scalar>& state, StateInitType type, std::string_view axis, bool use_eigenspinors, long bond_lim);

        // Product states
        template<typename Scalar> extern void set_random_product_state_with_random_spinors(StateFinite<Scalar>& state, StateInitType type, std::string &pattern);
        template<typename Scalar> extern void set_random_product_state_on_axis_using_eigenspinors(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string &pattern);
        template<typename Scalar> extern void set_random_product_state_on_axis(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string &pattern);
        template<typename Scalar> extern void set_product_state_domain_wall(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string  & pattern);
        template<typename Scalar> extern void set_product_state_aligned(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string &pattern);
        template<typename Scalar> extern void set_product_state_neel(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string  &pattern);
        template<typename Scalar> extern void set_product_state_neel_shuffled (StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string & pattern);
        template<typename Scalar> extern void set_product_state_neel_dislocated (StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string & pattern);
        template<typename Scalar> extern void set_product_state_on_axis_using_pattern(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string & pattern);
        template<typename Scalar> extern void set_sum_of_random_product_states(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string & pattern);

        // Entangled states
        template<typename Scalar> extern void randomize_given_state (StateFinite<Scalar>& state, StateInitType type);
        template<typename Scalar> extern void set_midchain_singlet_neel_state(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, std::string &pattern);
        template<typename Scalar> extern void set_random_entangled_state_on_axes_using_eigenspinors(StateFinite<Scalar>& state, StateInitType type, const std::vector<std::string> & axes, long bond_lim);
        template<typename Scalar> extern void set_random_entangled_state_on_axis_using_eigenspinors(StateFinite<Scalar>& state, StateInitType type, std::string_view axis, long bond_lim);
        template<typename Scalar> extern void set_random_entangled_state_with_random_spinors(StateFinite<Scalar>& state, StateInitType type, long bond_lim);
        template<typename Scalar> extern void set_random_entangled_state_haar(StateFinite<Scalar>& state, StateInitType type, long bond_lim);
    }

    /* clang-format on */

    template<typename Scalar>
    void initialize_state(StateFinite<Scalar> &state, StateInit init, StateInitType type, std::string_view axis, bool use_eigenspinors, long bond_lim,
                          std::string &pattern) {
        switch(init) {
            case StateInit::RANDOM_PRODUCT_STATE: return init::random_product_state(state, type, axis, use_eigenspinors, pattern);
            case StateInit::RANDOM_ENTANGLED_STATE: return init::random_entangled_state(state, type, axis, bond_lim, use_eigenspinors);
            case StateInit::RANDOMIZE_PREVIOUS_STATE: return init::randomize_given_state(state, type);
            case StateInit::MIDCHAIN_SINGLET_NEEL_STATE: return init::set_midchain_singlet_neel_state(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_DOMAIN_WALL: return init::set_product_state_domain_wall(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_ALIGNED: return init::set_product_state_aligned(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_NEEL: return init::set_product_state_neel(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_NEEL_SHUFFLED: return init::set_product_state_neel_shuffled(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_NEEL_DISLOCATED: return init::set_product_state_neel_dislocated(state, type, axis, pattern);
            case StateInit::PRODUCT_STATE_PATTERN: return init::set_product_state_on_axis_using_pattern(state, type, axis, pattern);
            case StateInit::SUM_OF_RANDOM_PRODUCT_STATES: return init::set_sum_of_random_product_states(state, type, axis, pattern);
        }
    }

}
