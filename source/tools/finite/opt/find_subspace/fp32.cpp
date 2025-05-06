#include "impl.h"

using Scalar = fp32;
using T = fp32;

/* clang-format off */

template std::vector<opt_mps<Scalar>>            subspace::find_subspace<T>(const TensorsFinite<Scalar> &tensors, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_part<T>(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> energy_target, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_primme<T>(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> eigval_shift, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_lapack<T>(const TensorsFinite<Scalar> &tensors, reports::subs_log<Scalar> &slog);
template MatrixType<T>                           subspace::get_hamiltonian_in_subspace<T>(const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const std::vector<opt_mps<Scalar>> &eigvecs);
template MatrixType<T>                           subspace::get_hamiltonian_squared_in_subspace<T>(const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const std::vector<opt_mps<Scalar>> &eigvecs);

