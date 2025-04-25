#pragma once
#include <array>
#include <complex>
#include <vector>

template<typename Scalar> class MpoSite;
template<typename Scalar> class TensorsFinite;
template<typename Scalar>
class ModelLocal {
    public:
    std::vector<std::unique_ptr<MpoSite<Scalar>>> mpos; /*!< A subset of mpos */

    std::vector<size_t>              get_positions() const;
    std::vector<std::array<long, 4>> get_dimensions() const;

    //    operator * ()
};