#pragma once
#include <vector>
#include <utility>
template<typename Scalar>  class StateFinite;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> [[nodiscard]] extern long                 bond_dimension_current    (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern long                 bond_dimension_midchain   (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::vector<long>    bond_dimensions_active    (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::pair<long,long> bond_dimensions           (const StateFinite<Scalar> & state, size_t pos);
    template<typename Scalar> [[nodiscard]] extern std::vector<long>    bond_dimensions           (const StateFinite<Scalar> & state);
    template<typename Scalar> [[nodiscard]] extern std::vector<long>    spin_dimensions           (const StateFinite<Scalar> & state);
    /* clang-format off */
}