#include "impl.h"

using Scalar = cx64;

/* clang-format off */

template std::array<long, 3> tools::finite::multisite::get_dimensions(const StateFinite<Scalar> &state, std::optional<std::vector<size_t>> sites);

template std::array<long, 4> tools::finite::multisite::get_dimensions(const ModelFinite<Scalar> &model, std::optional<std::vector<size_t>> sites);

template std::array<long, 4> tools::finite::multisite::get_dimensions_squared(const ModelFinite<Scalar> &model, std::optional<std::vector<size_t>> sites);

template long tools::finite::multisite::get_problem_size(const StateFinite<Scalar> &state, std::optional<std::vector<size_t>> sites);

template std::vector<size_t> tools::finite::multisite::generate_site_list(StateFinite<Scalar> &state, long threshold, size_t max_sites, size_t min_sites, const std::string &tag);

