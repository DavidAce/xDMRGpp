#include "../../StateFinite.tmpl.h"

using Scalar = cx32;
using T      = cx64;

template StateFinite<Scalar>::StateFinite(const StateFinite<T> &other) noexcept;

template Eigen::Tensor<T, 3>  StateFinite<Scalar>::get_multisite_mps<T>(const std::vector<size_t> &sites, bool use_cache) const;

template Eigen::Tensor<T, 2>  StateFinite<Scalar>::get_reduced_density_matrix<T>(const std::vector<size_t> &sites) const;

template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<T>(const std::vector<size_t> &sites) const;

template Eigen::Tensor<T, 2>  StateFinite<Scalar>::get_transfer_matrix<T>(const std::vector<size_t> &sites, std::string_view side) const;

template double StateFinite<Scalar>::get_transfer_matrix_cost<T>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<T>> &trf_cache) const;

template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<T>(const std::vector<size_t> &sites, std::string_view side) const;
