#include "impl.h"

using Scalar = cx128;

/* clang-format off */

template RealScalar<Scalar>  tools::infinite::measure::energy_minus_energy_shift(const StateInfinite<Scalar> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_minus_energy_shift(const  Eigen::Tensor<Scalar, 3> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_mpo(const StateInfinite<Scalar> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_mpo(const  Eigen::Tensor<Scalar, 3> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_per_site_mpo(const StateInfinite<Scalar> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_per_site_mpo(const  Eigen::Tensor<Scalar, 3> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_mpo(const StateInfinite<Scalar> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_mpo(const  Eigen::Tensor<Scalar, 3> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_per_site_mpo(const StateInfinite<Scalar> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_per_site_mpo(const  Eigen::Tensor<Scalar, 3> &, const ModelInfinite<Scalar> &model, const EdgesInfinite<Scalar> &edges);

template RealScalar<Scalar>  tools::infinite::measure::energy_mpo(const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_per_site_mpo(const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_mpo(const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_per_site_mpo(const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_mpo(const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_per_site_mpo(const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_mpo(const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> &);

template RealScalar<Scalar>  tools::infinite::measure::energy_variance_per_site_mpo(const Eigen::Tensor<Scalar,3> &mps, const TensorsInfinite<Scalar> &);
