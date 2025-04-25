#include "ModelFinite.impl.h"

using Scalar = cx64;

/* clang-format off */
template Eigen::Tensor<fp32, 4>  ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
template Eigen::Tensor<fp64, 4>  ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
template Eigen::Tensor<fp128, 4> ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
template Eigen::Tensor<cx32, 4>  ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
template Eigen::Tensor<cx64, 4>  ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
template Eigen::Tensor<cx128, 4> ModelFinite<Scalar>::get_multisite_mpo(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody, bool with_edgeL, bool with_edgeR) const;
/* clang-format on */

template const Eigen::Tensor<fp32, 4>  &ModelFinite<Scalar>::get_multisite_mpo() const;
template const Eigen::Tensor<fp64, 4>  &ModelFinite<Scalar>::get_multisite_mpo() const;
template const Eigen::Tensor<fp128, 4> &ModelFinite<Scalar>::get_multisite_mpo() const;
template const Eigen::Tensor<cx32, 4>  &ModelFinite<Scalar>::get_multisite_mpo() const;
template const Eigen::Tensor<cx64, 4>  &ModelFinite<Scalar>::get_multisite_mpo() const;
template const Eigen::Tensor<cx128, 4> &ModelFinite<Scalar>::get_multisite_mpo() const;

template Eigen::Tensor<fp32, 4>  ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<fp64, 4>  ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<fp128, 4> ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx32, 4>  ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx64, 4>  ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx128, 4> ModelFinite<Scalar>::get_multisite_mpo_shifted_view(Scalar energy_per_site) const;

template const Eigen::Tensor<fp32, 4>  &ModelFinite<Scalar>::get_multisite_mpo_squared() const;
template const Eigen::Tensor<fp64, 4>  &ModelFinite<Scalar>::get_multisite_mpo_squared() const;
template const Eigen::Tensor<fp128, 4> &ModelFinite<Scalar>::get_multisite_mpo_squared() const;
template const Eigen::Tensor<cx32, 4>  &ModelFinite<Scalar>::get_multisite_mpo_squared() const;
template const Eigen::Tensor<cx64, 4>  &ModelFinite<Scalar>::get_multisite_mpo_squared() const;
template const Eigen::Tensor<cx128, 4> &ModelFinite<Scalar>::get_multisite_mpo_squared() const;

/* clang-format off */
template Eigen::Tensor<fp32, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp64, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp128, 4> ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx32, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx64, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx128, 4> ModelFinite<Scalar>::get_multisite_mpo_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
/* clang-format on */

template Eigen::Tensor<fp32, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<fp64, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<fp128, 4> ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx32, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx64, 4>  ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;
template Eigen::Tensor<cx128, 4> ModelFinite<Scalar>::get_multisite_mpo_squared_shifted_view(Scalar energy_per_site) const;

template const Eigen::Tensor<fp32, 2>  &ModelFinite<Scalar>::get_multisite_ham() const;
template const Eigen::Tensor<fp64, 2>  &ModelFinite<Scalar>::get_multisite_ham() const;
template const Eigen::Tensor<fp128, 2> &ModelFinite<Scalar>::get_multisite_ham() const;
template const Eigen::Tensor<cx32, 2>  &ModelFinite<Scalar>::get_multisite_ham() const;
template const Eigen::Tensor<cx64, 2>  &ModelFinite<Scalar>::get_multisite_ham() const;
template const Eigen::Tensor<cx128, 2> &ModelFinite<Scalar>::get_multisite_ham() const;

template Eigen::Tensor<fp32, 2>  ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp64, 2>  ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp128, 2> ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx32, 2>  ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx64, 2>  ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx128, 2> ModelFinite<Scalar>::get_multisite_ham(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;

template const Eigen::Tensor<fp32, 2>  &ModelFinite<Scalar>::get_multisite_ham_squared() const;
template const Eigen::Tensor<fp64, 2>  &ModelFinite<Scalar>::get_multisite_ham_squared() const;
template const Eigen::Tensor<fp128, 2> &ModelFinite<Scalar>::get_multisite_ham_squared() const;
template const Eigen::Tensor<cx32, 2>  &ModelFinite<Scalar>::get_multisite_ham_squared() const;
template const Eigen::Tensor<cx64, 2>  &ModelFinite<Scalar>::get_multisite_ham_squared() const;
template const Eigen::Tensor<cx128, 2> &ModelFinite<Scalar>::get_multisite_ham_squared() const;

/* clang-format off */
template Eigen::Tensor<fp32, 2>  ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp64, 2>  ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<fp128, 2> ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx32, 2>  ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx64, 2>  ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
template Eigen::Tensor<cx128, 2> ModelFinite<Scalar>::get_multisite_ham_squared(const std::vector<size_t> &sites, std::optional<std::vector<size_t>> nbody) const;
/* clang-format on */
