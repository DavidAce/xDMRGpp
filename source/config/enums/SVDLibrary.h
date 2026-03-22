#pragma once

#include "config/enum_utils.h"

/*! Backend used for singular value decompositions */
enum class SVDLibrary {
    EIGEN,   /*!< Use Eigen's SVD implementation */
    LAPACKE, /*!< Use the LAPACKE backend */
    RSVD     /*!< Use the randomized SVD implementation */
};

template<> std::string_view enum2sv(SVDLibrary item) noexcept;
template<> SVDLibrary       sv2enum<SVDLibrary>(std::string_view item);
