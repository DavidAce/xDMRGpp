#pragma once
#include <vector>
template<typename Scalar> class StateFinite;
namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> [[nodiscard]] extern std::vector<double> truncation_errors                          (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::vector<double> truncation_errors_active                   (const StateFinite<Scalar> & state);
    /* clang-format on */

}