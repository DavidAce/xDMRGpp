#pragma once

#include "infoanalysis.h"
#include "infopolicy.h"
#include <Eigen/Core>
#include <vector>
template<typename Scalar>
class StateFinite;
enum class Precision;

namespace tools::finite::measure {
    /* clang-format off */
    template<typename Scalar> using RealScalar = decltype(std::real(std::declval<Scalar>()));
    template<typename Scalar> using RealArrayX  = Eigen::Array<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    template<typename Scalar> using RealArrayXX  = Eigen::Array<RealScalar<Scalar>, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   subsystem_entanglement_entropy_log2   (const StateFinite<Scalar> & state, const std::vector<size_t> & sites, Precision prec, size_t eig_max_size, std::string_view side);
    template<typename Scalar> [[nodiscard]] extern RealArrayXX<Scalar>  subsystem_entanglement_entropies_log2 (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealArrayXX<Scalar>  information_lattice                   (const RealArrayXX<Scalar> & SEE);
    template<typename Scalar> [[nodiscard]] extern RealArrayXX<Scalar>  information_lattice                   (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealArrayX<Scalar>   information_per_scale                 (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealArrayX<Scalar>   information_per_scale                 (const RealArrayXX<Scalar> & information_lattice);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_bit_scale                 (const RealArrayX<Scalar> & information_per_scale, double bit);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_center_of_mass            (const RealArrayXX<Scalar> & information_lattice);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_center_of_mass            (const RealArrayX<Scalar> & information_per_scale);
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_center_of_mass            (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern InfoAnalysis<Scalar> information_lattice_analysis          (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_xi_from_geometric_dist    (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_xi_from_avg_log_slope     (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    template<typename Scalar> [[nodiscard]] extern RealScalar<Scalar>   information_xi_from_exp_fit           (const StateFinite<Scalar> & state, InfoPolicy ip = {});
    /* clang-format on */
}