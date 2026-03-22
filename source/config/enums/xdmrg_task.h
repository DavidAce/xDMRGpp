#pragma once

#include "config/enum_utils.h"

/*! Task identifiers for the excited-state DMRG workflow */
enum class xdmrg_task {
    INIT_RANDOMIZE_MODEL,                /*!< Randomize model parameters */
    INIT_RANDOMIZE_INTO_PRODUCT_STATE,   /*!< Initialize a random product state */
    INIT_RANDOMIZE_INTO_ENTANGLED_STATE, /*!< Initialize a random entangled state */
    INIT_RANDOMIZE_FROM_CURRENT_STATE,   /*!< Randomize the currently loaded state */
    INIT_BOND_LIMITS,                    /*!< Initialize bond-dimension limits */
    INIT_TRNC_LIMITS,                    /*!< Initialize truncation-error limits */
    INIT_ENERGY_TARGET,                  /*!< Initialize the target energy density or shift */
    INIT_WRITE_MODEL,                    /*!< Write the model to file */
    INIT_CLEAR_STATUS,                   /*!< Clear the algorithm status */
    INIT_CLEAR_CONVERGENCE,              /*!< Clear convergence tracking */
    INIT_DEFAULT,                        /*!< Run the default initialization task list */
    FIND_ENERGY_RANGE,                   /*!< Find the extremal energies used to set the target window */
    FIND_EXCITED_STATE,                  /*!< Find the targeted excited state */
    POST_WRITE_RESULT,                   /*!< Write the final result to file */
    POST_PRINT_RESULT,                   /*!< Print the final result */
    POST_PRINT_TIMERS,                   /*!< Print timing information */
    POST_RBDS_ANALYSIS,                  /*!< Run reverse bond-dimension scaling analysis */
    POST_RTES_ANALYSIS,                  /*!< Run reverse truncation-error scaling analysis */
    POST_DEFAULT,                        /*!< Run the default post-processing task list */
    TIMER_RESET                          /*!< Reset accumulated timers */
};

template<> std::string_view enum2sv(xdmrg_task item) noexcept;
template<> xdmrg_task       sv2enum<xdmrg_task>(std::string_view item);
