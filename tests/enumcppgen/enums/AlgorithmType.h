#pragma once

namespace test::enumcppgen_demo {

/*! \brief Selects which top-level algorithm family to run. */
enum class AlgorithmType : int {
    iDMRG, /*!< Infinite DMRG */
    fDMRG, /*!< Finite DMRG */
    xDMRG, /*!< Excited-state DMRG */
    iTEBD, /*!< Infinite TEBD */
    fLBIT, /*!< Finite-system l-bit evolution */
    ANY,   /*!< Match any algorithm */
};

}
