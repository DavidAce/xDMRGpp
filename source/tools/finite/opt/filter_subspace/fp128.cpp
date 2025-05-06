#include "impl.h"

using Scalar = fp128;

/* clang-format off */


template void tools::finite::opt::internal::subspace::filter_subspace(std::vector<opt_mps<Scalar>> &subspace, size_t max_accept);

template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> &eigvecs);

template std::optional<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_lowest_variance(const std::vector<opt_mps<Scalar>> &eigvecs);

template std::vector<size_t> tools::finite::opt::internal::subspace::get_idx_to_eigvec_with_highest_overlap(const std::vector<opt_mps<Scalar>> &eigvecs, size_t max_eigvecs);

template MatrixType<Scalar>  tools::finite::opt::internal::subspace::get_eigvecs(const std::vector<opt_mps<Scalar>> &eigvecs);

template VectorReal<Scalar>  tools::finite::opt::internal::subspace::get_eigvals(const std::vector<opt_mps<Scalar>> &eigvecs);

template VectorReal<Scalar>  tools::finite::opt::internal::subspace::get_energies(const std::vector<opt_mps<Scalar>> &eigvecs);

template RealScalar<Scalar>  tools::finite::opt::internal::subspace::get_subspace_error(const std::vector<opt_mps<Scalar>> &eigvecs, std::optional<size_t> max_eigvecs);

template std::vector<RealScalar<Scalar>>  tools::finite::opt::internal::subspace::get_subspace_errors(const std::vector<opt_mps<Scalar>> &eigvecs);

template RealScalar<Scalar>  tools::finite::opt::internal::subspace::get_subspace_error<Scalar>(const std::vector<RealScalar<Scalar>> &overlaps);

template VectorReal<Scalar>  tools::finite::opt::internal::subspace::get_overlaps(const std::vector<opt_mps<Scalar>> &eigvecs);

template VectorType<Scalar>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<Scalar>> &eigvecs, size_t idx);

template VectorType<Scalar>  tools::finite::opt::internal::subspace::get_vector_in_subspace(const std::vector<opt_mps<Scalar>> &eigvecs, const VectorType<Scalar> &fullspace_vector);

template VectorType<Scalar> tools::finite::opt::internal::subspace::get_vector_in_fullspace(const std::vector<opt_mps<Scalar>> &eigvecs, const VectorType<Scalar>   &subspace_vector);

template Eigen::Tensor<Scalar, 3>  tools::finite::opt::internal::subspace::get_tensor_in_fullspace(const std::vector<opt_mps<Scalar>> &eigvecs, const VectorType<Scalar> &subspace_vector, const std::array<Eigen::Index, 3>  &dims);
