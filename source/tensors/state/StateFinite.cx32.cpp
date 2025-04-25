#include "StateFinite.impl.h"

using Scalar = cx32;

template StateFinite<Scalar>::StateFinite(const StateFinite<fp32> &other) noexcept;
template StateFinite<Scalar>::StateFinite(const StateFinite<fp64> &other) noexcept;
template StateFinite<Scalar>::StateFinite(const StateFinite<fp128> &other) noexcept;
// template StateFinite<Scalar>::StateFinite(const StateFinite<cx32> &other) noexcept;
template StateFinite<Scalar>::StateFinite(const StateFinite<cx64> &other) noexcept;
template StateFinite<Scalar>::StateFinite(const StateFinite<cx128> &other) noexcept;

template Eigen::Tensor<fp32, 3>  StateFinite<Scalar>::get_multisite_mps<fp32>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<fp64, 3>  StateFinite<Scalar>::get_multisite_mps<fp64>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<fp128, 3> StateFinite<Scalar>::get_multisite_mps<fp128>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<cx32, 3>  StateFinite<Scalar>::get_multisite_mps<cx32>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<cx64, 3>  StateFinite<Scalar>::get_multisite_mps<cx64>(const std::vector<size_t> &sites, bool use_cache) const;
template Eigen::Tensor<cx128, 3> StateFinite<Scalar>::get_multisite_mps<cx128>(const std::vector<size_t> &sites, bool use_cache) const;

template Eigen::Tensor<fp32, 2>  StateFinite<Scalar>::get_reduced_density_matrix<fp32>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<fp64, 2>  StateFinite<Scalar>::get_reduced_density_matrix<fp64>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<fp128, 2> StateFinite<Scalar>::get_reduced_density_matrix<fp128>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<cx32, 2>  StateFinite<Scalar>::get_reduced_density_matrix<cx32>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<cx64, 2>  StateFinite<Scalar>::get_reduced_density_matrix<cx64>(const std::vector<size_t> &sites) const;
template Eigen::Tensor<cx128, 2> StateFinite<Scalar>::get_reduced_density_matrix<cx128>(const std::vector<size_t> &sites) const;

template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<fp32>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<fp64>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<fp128>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<cx32>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<cx64>(const std::vector<size_t> &sites) const;
template std::array<double, 3> StateFinite<Scalar>::get_reduced_density_matrix_cost<cx128>(const std::vector<size_t> &sites) const;

template Eigen::Tensor<fp32, 2>  StateFinite<Scalar>::get_transfer_matrix<fp32>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<fp64, 2>  StateFinite<Scalar>::get_transfer_matrix<fp64>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<fp128, 2> StateFinite<Scalar>::get_transfer_matrix<fp128>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<cx32, 2>  StateFinite<Scalar>::get_transfer_matrix<cx32>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<cx64, 2>  StateFinite<Scalar>::get_transfer_matrix<cx64>(const std::vector<size_t> &sites, std::string_view side) const;
template Eigen::Tensor<cx128, 2> StateFinite<Scalar>::get_transfer_matrix<cx128>(const std::vector<size_t> &sites, std::string_view side) const;

/* clang-format off */
template double StateFinite<Scalar>::get_transfer_matrix_cost<fp32>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<fp32>> &trf_cache) const;
template double StateFinite<Scalar>::get_transfer_matrix_cost<fp64>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<fp64>> &trf_cache) const;
template double StateFinite<Scalar>::get_transfer_matrix_cost<fp128>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<fp128>> &trf_cache) const;
template double StateFinite<Scalar>::get_transfer_matrix_cost<cx32>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<cx32>> &trf_cache) const;
template double StateFinite<Scalar>::get_transfer_matrix_cost<cx64>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<cx64>> &trf_cache) const;
template double StateFinite<Scalar>::get_transfer_matrix_cost<cx128>(const std::vector<size_t> &sites, std::string_view side, const std::optional<TrfCacheEntry<cx128>> &trf_cache) const;
/* clang-format on */

template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<fp32>(const std::vector<size_t> &sites, std::string_view side) const;
template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<fp64>(const std::vector<size_t> &sites, std::string_view side) const;
template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<fp128>(const std::vector<size_t> &sites, std::string_view side) const;
template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<cx32>(const std::vector<size_t> &sites, std::string_view side) const;
template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<cx64>(const std::vector<size_t> &sites, std::string_view side) const;
template std::array<double, 2> StateFinite<Scalar>::get_transfer_matrix_costs<cx128>(const std::vector<size_t> &sites, std::string_view side) const;
