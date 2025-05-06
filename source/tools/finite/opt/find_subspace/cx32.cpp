#include "impl.h"

using Scalar = cx32;
using T = cx32;
using R = fp32;


/* clang-format off */

template std::vector<opt_mps<Scalar>>            subspace::find_subspace<T>(const TensorsFinite<Scalar> &tensors, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_part<T>(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> energy_target, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_primme<T>(const TensorsFinite<Scalar> &tensors, RealScalar<Scalar> eigval_shift, const OptMeta &meta, reports::subs_log<Scalar> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_lapack<T>(const TensorsFinite<Scalar> &tensors, reports::subs_log<Scalar> &slog);
template MatrixType<T>                           subspace::get_hamiltonian_in_subspace<T>(const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const std::vector<opt_mps<Scalar>> &eigvecs);
template MatrixType<T>                           subspace::get_hamiltonian_squared_in_subspace<T>(const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const std::vector<opt_mps<Scalar>> &eigvecs);

template std::vector<opt_mps<R>>                 subspace::find_subspace<T>(const TensorsFinite<R> &tensors, const OptMeta &meta, reports::subs_log<R> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_part<T>(const TensorsFinite<R> &tensors, RealScalar<R> energy_target, const OptMeta &meta, reports::subs_log<R> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_primme<T>(const TensorsFinite<R> &tensors, RealScalar<R> eigval_shift, const OptMeta &meta, reports::subs_log<R> &slog);
template std::pair<MatrixType<T>, VectorReal<T>> subspace::find_subspace_lapack<T>(const TensorsFinite<R> &tensors, reports::subs_log<R> &slog);
template MatrixType<T>                           subspace::get_hamiltonian_in_subspace<T>(const ModelFinite<R> &model, const EdgesFinite<R> &edges, const std::vector<opt_mps<R>> &eigvecs);
template MatrixType<T>                           subspace::get_hamiltonian_squared_in_subspace<T>(const ModelFinite<R> &model, const EdgesFinite<R> &edges, const std::vector<opt_mps<R>> &eigvecs);

template std::vector<opt_mps<T>>                 subspace::find_subspace<R>(const TensorsFinite<T> &tensors, const OptMeta &meta, reports::subs_log<T> &slog);
template std::pair<MatrixType<R>, VectorReal<R>> subspace::find_subspace_part<R>(const TensorsFinite<T> &tensors, RealScalar<T> energy_target, const OptMeta &meta, reports::subs_log<T> &slog);
template std::pair<MatrixType<R>, VectorReal<R>> subspace::find_subspace_primme<R>(const TensorsFinite<T> &tensors, RealScalar<T> eigval_shift, const OptMeta &meta, reports::subs_log<T> &slog);
template std::pair<MatrixType<R>, VectorReal<R>> subspace::find_subspace_lapack<R>(const TensorsFinite<T> &tensors, reports::subs_log<T> &slog);
template MatrixType<R>                           subspace::get_hamiltonian_in_subspace<R>(const ModelFinite<T> &model, const EdgesFinite<T> &edges, const std::vector<opt_mps<T>> &eigvecs);
template MatrixType<R>                           subspace::get_hamiltonian_squared_in_subspace<R>(const ModelFinite<T> &model, const EdgesFinite<T> &edges, const std::vector<opt_mps<T>> &eigvecs);

