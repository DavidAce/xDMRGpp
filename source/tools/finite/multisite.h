#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;

namespace tools::finite::multisite {
    /* clang-format off */
    template<typename Scalar> extern long                get_problem_size(const StateFinite<Scalar> &state, std::optional<std::vector<size_t>> sites = std::nullopt);
    template<typename Scalar> extern std::array<long, 3> get_dimensions(const StateFinite<Scalar> &state, std::optional<std::vector<size_t>> sites = std::nullopt);
    template<typename Scalar> extern std::array<long, 4> get_dimensions(const ModelFinite<Scalar> &model, std::optional<std::vector<size_t>> sites = std::nullopt);
    template<typename Scalar> extern std::array<long, 4> get_dimensions_squared(const ModelFinite<Scalar> &model, std::optional<std::vector<size_t>> sites = std::nullopt);
    template<typename Scalar> extern std::vector<size_t> generate_site_list(StateFinite<Scalar> &state, long threshold, size_t max_sites, size_t min_sites = 1, const std::string &tag = "");
    // template<typename Scalar> extern std::vector<size_t> generate_site_list_old(StateFinite<Scalar> &state, long threshold, size_t max_sites, size_t min_sites = 1, const std::string &tag = "");
    template<typename Scalar> extern std::vector<size_t> generate_truncated_site_list(StateFinite<Scalar> &state, long threshold, long bond_lim, size_t max_sites, size_t min_sites = 1);
                              extern std::vector<size_t> activate_sites(long threshold, size_t max_sites, size_t min_sites = 1);
                              extern std::vector<size_t> activate_truncated_sites(long threshold, long bond_lim, size_t max_sites, size_t min_sites = 1);
                              extern std::array<long, 3> active_dimensions();
                              extern long                active_problem_size();
    /* clang-format on */

}
