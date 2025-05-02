#include "../information.impl.h"

using Scalar = fp32;

template RealScalar<Scalar>  tools::finite::measure::subsystem_entanglement_entropy_log2(const StateFinite<Scalar> &state, const std::vector<size_t> &sites, Precision prec, size_t eig_max_size, std::string_view side);

template RealArrayXX<Scalar>  tools::finite::measure::subsystem_entanglement_entropies_log2(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealArrayXX<Scalar>  tools::finite::measure::information_lattice<Scalar>(const RealArrayXX<Scalar> &SEE);

template RealArrayXX<Scalar>  tools::finite::measure::information_lattice(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealArrayX<Scalar>  tools::finite::measure::information_per_scale<Scalar>(const RealArrayXX<Scalar> &information_lattice);

template RealArrayX<Scalar>  tools::finite::measure::information_per_scale(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealScalar<Scalar>  tools::finite::measure::information_bit_scale<Scalar>(const RealArrayX<Scalar> &information_per_scale, double bit);

template RealScalar<Scalar>  tools::finite::measure::information_center_of_mass<Scalar>(const RealArrayXX<Scalar> &information_lattice);

template RealScalar<Scalar>  tools::finite::measure::information_center_of_mass<Scalar>(const RealArrayX<Scalar> &information_per_scale);

template RealScalar<Scalar>  tools::finite::measure::information_center_of_mass(const StateFinite<Scalar> &state, InfoPolicy ip);

template InfoAnalysis<Scalar>  tools::finite::measure::information_lattice_analysis(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealScalar<Scalar>  tools::finite::measure::information_xi_from_geometric_dist(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealScalar<Scalar>  tools::finite::measure::information_xi_from_avg_log_slope(const StateFinite<Scalar> &state, InfoPolicy ip);

template RealScalar<Scalar>  tools::finite::measure::information_xi_from_exp_fit(const StateFinite<Scalar> &state, InfoPolicy ip);

