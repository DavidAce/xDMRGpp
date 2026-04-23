#pragma once
#include "config/enum_utils.h"

/*! Choose the algorithm for the optimization.
 *
 * In the descriptions below, H and |v> are the local effective Hamiltonians and states.
 * In xdmrg, a Hamiltonian shift H -> H - E_tgt is controlled by the selected target energy,
 * for example `settings::xdmrg::energy_spectrum_shift` when `settings::xdmrg::ritz == OptRitz::SM`.
 */

enum class OptAlgo {
    DMRG,         /*!< Plain DMRG that solves the eigenvalue problem H|v>=E|v> */
    DMRG_X,        /*!< Find an eigenvector of H that maximizes the overlap:  |v_new> = max_k <v_old | v_k> */
    DMRG_X_HYBRID, /*!< Minimize the variance of |v_new> = sum_{k in K} λ_k|v_k>..., where K is a basis of H_eff eigenvectors satisfying <v_old | v_k> != 0 */
    DMRG_FOLDED,  /*!< Solve the folded eigenvalue problem H²|v> = λ|v> */
    DMRG_GSI      /*!< Solve the generalized shift-invert problem H|v> = λH²|v> */
};
template<> std::string_view enum2sv(OptAlgo item) noexcept;
template<> OptAlgo          sv2enum<OptAlgo>(std::string_view item);
