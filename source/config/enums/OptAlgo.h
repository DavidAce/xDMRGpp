#pragma once
#include "config/enum_utils.h"

/*! Choose the algorithm for the optimization.
 *
 * In the descriptions below, H and |v> are the local effective Hamiltonians and states.
 */

enum class OptAlgo {
    DMRG,         /*!< Plain DMRG that solves the eigenvalue problem H|v>=E|v> for some given */
    DMRGX,        /*!< Find an eigenvector of H that maximizes the overlap:  |v_new> = max_k <v_old | v_k> */
    HYBRID_DMRGX, /*!< Minimize the variance of |v_new> = sum_{k in K} λ_k|v_k>..., where K is a basis of H_eff eigenvectors satisfying <v_old | v_k> != 0 */
    XDMRG,        /*!< Solve the folded eigenvalue problem (H-Eshift)²|v> */
    GDMRG         /*!< Solve the generalized shift-invert problem (H-Eshift)|v> = (1/E)(H-Eshift)²|v> */
};
template<> std::string_view enum2sv(OptAlgo item) noexcept;
template<> OptAlgo          sv2enum<OptAlgo>(std::string_view item);
