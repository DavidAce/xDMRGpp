#include "../../ModelFinite.tmpl.h"

using Scalar = cx32;
using T = cx128;

/* clang-format off */
template Eigen::Tensor<T, 4>  ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;

template const Eigen::Tensor<T, 4>  &ModelFinite<Scalar>::get_multisite_mpo() const;

template Eigen::Tensor<T, 4>  ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;

template const Eigen::Tensor<T, 4>  &ModelFinite<Scalar>::get_multisite_mpo_squared() const;

template Eigen::Tensor<T, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;

template Eigen::Tensor<T, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;

template const Eigen::Tensor<T, 2>  &ModelFinite<Scalar>::get_multisite_ham() const;

template Eigen::Tensor<T, 2>  ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;

template const Eigen::Tensor<T, 2>  &ModelFinite<Scalar>::get_multisite_ham_squared() const;

template Eigen::Tensor<T, 2>  ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
