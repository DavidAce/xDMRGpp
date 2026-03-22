#pragma once
#include "config/enum_utils.h"

/*! \brief Policy to determine the number of sites (the block size) involved in either
 *  a dmrg update (optimization) step, and/or the bond expansion step.
 *  We can either use a fixed number or use one of the information lattice metrics, and enable the increased
 *  block size conditionally if the algorithm has entered a special status.
 */

enum class BlockSizePolicy {
    MIN         = 0,       /*!< Block size = settings::strategy::dmrg_min_blocksize */
    MAX         = 1 << 0,  /*!< Block size = settings::strategy::dmrg_max_blocksize */
    INFO        = 1 << 1,  /*!< Block size = ceil of "information_center_of_mass" */
    INFOPLUS1   = 1 << 2,  /*!< Block size = ceil of "information_center_of_mass + 1.00" */
    INFO150     = 1 << 3,  /*!< Block size = ceil of "information_center_of_mass * 1.50" */
    INFO200     = 1 << 4,  /*!< Block size = ceil of "information_center_of_mass * 2.00" */
    BIT_ONE     = 1 << 5,  /*!< Block size = scale to find at least one bit in the information lattice. */
    BIT_TWO     = 1 << 6,  /*!< Block size = scale to find at least two bits in the information lattice. */
    BIT_MID     = 1 << 7,  /*!< Block size = scale to find L/2 bits in the information lattice. */
    BIT_PEN     = 1 << 8,  /*!< Block size = scale to find L-1 (up to PENultimate) bits in the information lattice. */
    BIT_ALL     = 1 << 9,  /*!< Block size = scale to find L (all) bits in the information lattice. */
    IF_SAT_ENTR = 1 << 10, /*!< Set the block size if the entanglement entropy has saturated */
    IF_SAT_INFO = 1 << 11, /*!< Set the block size if the information center of mass has saturated */
    IF_SAT_EVAR = 1 << 12, /*!< Set the block size if the energy variance has saturated */
    IF_SAT_ALGO = 1 << 13, /*!< Set the block size if the algorithm status == saturated (implies all other saturations) */
    IF_STK_ALGO = 1 << 14, /*!< Set the block size if the algorithm status == stuck */
    IF_FIN_BOND = 1 << 15, /*!< Require that the bond dimension has reached its final (maximum) value before increasing the block size */
    IF_FIN_TRNC = 1 << 16, /*!< Require that the truncation error has reached its final (minimum) value before increasing the block size */
    ON_BONDEXP  = 1 << 17, /*!< Enable the selected blocksize during bond expansion (otherwise default to dmrg_min_blocksize) */
    ON_UPDATE   = 1 << 18, /*!< Enable the selected blocksize during the dmrg update step (otherwise default to dmrg_min_blocksize) */
    DEFAULT     = MIN,     /*!< The default choice usually works well */
    allow_bitops           /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(BlockSizePolicy item) noexcept;
template<> BlockSizePolicy  sv2enum<BlockSizePolicy>(std::string_view item);
template<> std::string      flag2str(const BlockSizePolicy &item) noexcept;
