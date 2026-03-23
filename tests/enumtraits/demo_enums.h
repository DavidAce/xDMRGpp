#pragma once

#include "enum_support.h"
#include <array>
#include <string_view>

namespace test::enumtraits_demo {

/*! \brief Selects which top-level algorithm family to run. */
enum class AlgorithmType : int {
    iDMRG, /*!< Infinite DMRG */
    fDMRG, /*!< Finite DMRG */
    xDMRG, /*!< Excited-state DMRG */
    iTEBD, /*!< Infinite TEBD */
    fLBIT, /*!< Finite-system l-bit evolution */
    ANY,   /*!< Match any algorithm */
};

/*! \brief Controls how MPOs are compressed. */
enum class MpoCompress {
    NONE, /*!< Do not compress */
    SVD,  /*!< Use SVD on each mpo */
    DPL,  /*!< Deparallelization: removes parallel columns/rows from each mpo */
    AUTO, /*!< Select based on global setting */
};

/*! \brief When to do projections to a symmetry sector. */
enum class ProjectionPolicy : int {
    NEVER     = 0,                        /*!< Never project */
    INIT      = 1,                        /*!< Project after initializing the state  */
    WARMUP    = 2,                        /*!< Project during warmup  */
    STUCK     = 4,                        /*!< Project when the algorithm is stuck */
    ITER      = 8,                        /*!< Project every iteration */
    CONVERGED = 16,                       /*!< Project on converged iterations */
    FINISHED  = 32,                       /*!< Project the finished state during postprocessing */
    FORCE     = 64,                       /*!< Project even if not needed */
    DEFAULT   = INIT | STUCK | CONVERGED, /*!< Default policy for multisite dmrg steps */
};

} // namespace test::enumtraits_demo

namespace test::enum_support {

template<>
struct enum_traits<test::enumtraits_demo::AlgorithmType> {
    static constexpr bool             is_bitflag = false;
    static constexpr std::string_view doc        = "Selects which top-level algorithm family to run.";
    static constexpr std::array       entries    = {
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::iDMRG, "iDMRG", "Infinite DMRG"},
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::fDMRG, "fDMRG", "Finite DMRG"},
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::xDMRG, "xDMRG", "Excited-state DMRG"},
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::iTEBD, "iTEBD", "Infinite TEBD"},
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::fLBIT, "fLBIT", "Finite-system l-bit evolution"},
        enum_entry<test::enumtraits_demo::AlgorithmType>{test::enumtraits_demo::AlgorithmType::ANY, "ANY", "Match any algorithm"},
    };
};

template<>
struct enum_traits<test::enumtraits_demo::MpoCompress> {
    static constexpr bool             is_bitflag = false;
    static constexpr std::string_view doc        = "Controls how MPOs are compressed.";
    static constexpr std::array       entries    = {
        enum_entry<test::enumtraits_demo::MpoCompress>{test::enumtraits_demo::MpoCompress::NONE, "NONE", "Do not compress"},
        enum_entry<test::enumtraits_demo::MpoCompress>{test::enumtraits_demo::MpoCompress::SVD, "SVD", "Use SVD on each mpo"},
        enum_entry<test::enumtraits_demo::MpoCompress>{test::enumtraits_demo::MpoCompress::DPL, "DPL",
                                                       "Deparallelization: removes parallel columns/rows from each mpo"},
        enum_entry<test::enumtraits_demo::MpoCompress>{test::enumtraits_demo::MpoCompress::AUTO, "AUTO", "Select based on global setting"},
    };
};

template<>
struct enum_traits<test::enumtraits_demo::ProjectionPolicy> {
    static constexpr bool             is_bitflag = true;
    static constexpr std::string_view doc        = "When to do projections to a symmetry sector.";
    static constexpr std::array       entries    = {
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::NEVER, "NEVER", "Never project"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::INIT, "INIT",
                                                            "Project after initializing the state"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::WARMUP, "WARMUP", "Project during warmup"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::STUCK, "STUCK",
                                                            "Project when the algorithm is stuck"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::ITER, "ITER", "Project every iteration"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::CONVERGED, "CONVERGED",
                                                            "Project on converged iterations"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::FINISHED, "FINISHED",
                                                            "Project the finished state during postprocessing"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::FORCE, "FORCE",
                                                            "Project even if not needed"},
        enum_entry<test::enumtraits_demo::ProjectionPolicy>{test::enumtraits_demo::ProjectionPolicy::DEFAULT, "DEFAULT",
                                                            "Default policy for multisite dmrg steps", false},
    };
};

} // namespace test::enum_support
