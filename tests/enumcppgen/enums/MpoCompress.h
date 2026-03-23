#pragma once

namespace test::enumcppgen_demo {

/*! \brief Controls how MPOs are compressed. */
enum class MpoCompress {
    NONE, /*!< Do not compress */
    SVD,  /*!< Use SVD on each mpo */
    DPL,  /*!< Deparallelization: removes parallel columns/rows from each mpo */
    AUTO, /*!< Select based on global setting */
};

}
