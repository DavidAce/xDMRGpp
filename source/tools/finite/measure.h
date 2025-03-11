#pragma once
#include "config/enums.h"
#include "math/float.h"
#include "math/svd/config.h"
#include <complex>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

class StateFinite;
class ModelFinite;
class EdgesFinite;
class TensorsFinite;
class MpoSite;
class MpsSite;
class AlgorithmStatus;
struct MeasurementsTensorsFinite;
template<typename T>
struct env_pair;
class EnvEne;
class EnvVar;
enum class RDM;

struct InfoPolicy {
    std::optional<double>      bits_max_error = std::nullopt; /*!< Positive for relative error = 1-bits_found/L, negative for absolute error = L-bits_found */
    std::optional<size_t>      eig_max_size   = std::nullopt; /*!< Maximum matrix size to diagonalize (skip if larger). Recommend <= 8192 */
    std::optional<double>      svd_max_size   = std::nullopt; /*!< Maximum matrix size for svd during swaps (skip if larger) Recommend <= 4096 */
    std::optional<double>      svd_trnc_lim   = std::nullopt; /*!< Maximum discarded weight in the svd during swaps. Recommend <= 1e-6 */
    std::optional<CachePolicy> cachePolicy    = std::nullopt; /*!< Read and/or write intermediate results into the state cache */
    std::optional<Precision>   precision      = std::nullopt;
};

struct InfoAnalysis {
    double          bits_found    = 0.0;              /*!< The number of bits found. If all goes well this is equal to L. */
    double          icom          = 0.0;              /*!< information center of mass, aka "expected correlation length" or "xi", "<I^\ell>"  */
    double          scale_bit_one = 0.0;              /*!< The minimum length scale to measure 1 bit (the first). */
    double          scale_bit_two = 0.0;              /*!< The minimum length scale to measure 2 bits (the first two).  */
    double          scale_bit_mid = 0.0;              /*!< The minimum length scale to measure L/2 bits. */
    double          scale_bit_pen = 0.0;              /*!< The minimum length scale to measure L-1 bits (all but the last). */
    double          scale_bit_all = 0.0;              /*!< The minimum length scale to measure L bits. */
    InfoPolicy      ip            = {};               /*!< Settings used to calculate this info lattice */
    Eigen::ArrayXd  info_per_scale;                   /*!< The information per scale "I^\ell" */
    Eigen::ArrayXXd info_lattice;                     /*!< The information lattice "i^\ell_n" */
    Eigen::ArrayXXd subsystem_entanglement_entropies; /*!< All the bipartite entanglement entropies (log2) */
};

namespace tools::finite::measure {
    struct LocalObservableOp {
        Eigen::Tensor<cx64, 2> op;
        long                   pos;
        mutable bool           used = false;
    };

    struct LocalObservableMpo {
        Eigen::Tensor<cx64, 4> mpo;
        long                   pos;
        mutable bool           used = false;
    };

    /* clang-format off */


    [[nodiscard]] extern size_t length                                      (const TensorsFinite & tensors);
    [[nodiscard]] extern size_t length                                      (const StateFinite & state);
    [[nodiscard]] extern size_t length                                      (const ModelFinite & model);
    [[nodiscard]] extern size_t length                                      (const EdgesFinite & edges);
    [[nodiscard]] extern long   bond_dimension_current                      (const StateFinite & state);
    [[nodiscard]] extern long   bond_dimension_midchain                     (const StateFinite & state);
    [[nodiscard]] extern std::vector<long>    bond_dimensions_active        (const StateFinite & state);
    [[nodiscard]] extern std::pair<long,long> bond_dimensions               (const StateFinite & state, size_t pos);
    [[nodiscard]] extern std::vector<long>    bond_dimensions               (const StateFinite & state);
    [[nodiscard]] extern double norm                                        (const StateFinite & state, bool full = false);

//  [[nodiscard]]  extern double norm_fast                                   (const StateFinite & state);


    [[nodiscard]] extern double spin_component                              (const StateFinite & state, const Eigen::Matrix2cd &paulimatrix);
    [[nodiscard]] extern double spin_component                              (const StateFinite & state, std::string_view axis);
    [[nodiscard]] extern double spin_alignment                              (const StateFinite & state, std::string_view axis);
    [[nodiscard]] extern int    spin_sign                                   (const StateFinite & state, std::string_view axis);
    template<typename Scalar = cx64>
    [[nodiscard]] extern Eigen::Tensor<cx64,1> mps2tensor                               (const std::vector<std::unique_ptr<MpsSite>> & mps_sites, std::string_view name);
    [[nodiscard]] extern Eigen::Tensor<cx64,1> mps2tensor                               (const StateFinite & state);
    [[nodiscard]] extern double entanglement_entropy                                    (const Eigen::Tensor<cx64,1> & bond);
    [[nodiscard]] extern double entanglement_entropy_current                            (const StateFinite & state);
    [[nodiscard]] extern double entanglement_entropy_midchain                           (const StateFinite & state);
    [[nodiscard]] extern std::vector<double> entanglement_entropies                     (const StateFinite & state);
    [[nodiscard]] extern double              entanglement_entropy_log2                  (const StateFinite & state, size_t nsites);
    [[nodiscard]] extern std::vector<double> entanglement_entropies_log2                (const StateFinite & state);
    [[nodiscard]] extern double              subsystem_entanglement_entropy_log2        (const StateFinite & state, const std::vector<size_t> & sites, Precision prec, size_t eig_max_size, std::string_view side);
    [[nodiscard]] extern Eigen::ArrayXXd     subsystem_entanglement_entropies_log2      (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern Eigen::ArrayXXd     information_lattice                        (const Eigen::ArrayXXd & SEE);
    [[nodiscard]] extern Eigen::ArrayXXd     information_lattice                        (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern Eigen::ArrayXd      information_per_scale                      (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern Eigen::ArrayXd      information_per_scale                      (const Eigen::ArrayXXd & information_lattice);
    [[nodiscard]] extern double              information_bit_scale                      (const Eigen::ArrayXd & information_per_scale, double bit);
    [[nodiscard]] extern double              information_center_of_mass                 (const Eigen::ArrayXXd & information_lattice);
    [[nodiscard]] extern double              information_center_of_mass                 (const Eigen::ArrayXd & information_per_scale);
    [[nodiscard]] extern double              information_center_of_mass                 (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern InfoAnalysis        information_lattice_analysis               (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern double              information_xi_from_geometric_dist         (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern double              information_xi_from_avg_log_slope          (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern double              information_xi_from_exp_fit                (const StateFinite & state, InfoPolicy ip = {});
    [[nodiscard]] extern double              renyi_entropy_midchain                     (const StateFinite & state, double q);
    [[nodiscard]] extern std::vector<double> renyi_entropies                            (const StateFinite & state, double q);
    [[nodiscard]] extern double              number_entropy_current                     (const StateFinite & state);
    [[nodiscard]] extern double              number_entropy_midchain                    (const StateFinite & state);
    [[nodiscard]] extern std::vector<double> number_entropies                           (const StateFinite & state);
    [[nodiscard]] extern std::array<double,3> spin_components                           (const StateFinite & state);
    [[nodiscard]] extern std::vector<double> truncation_errors                          (const StateFinite & state);
    [[nodiscard]] extern std::vector<double> truncation_errors_active                   (const StateFinite & state);

    [[nodiscard]] cx64 expval_hamiltonian                        (const TensorsFinite & tensors);
    [[nodiscard]] cx64 expval_hamiltonian                        (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges);
    [[nodiscard]] cx64 expval_hamiltonian                        (const Eigen::Tensor<cx64, 3> &mps, const ModelFinite & model, const EdgesFinite & edges);
    [[nodiscard]] cx64 expval_hamiltonian                        (const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvEne &> &envs);
    [[nodiscard]] cx64 expval_hamiltonian_squared                (const TensorsFinite & tensors);
    [[nodiscard]] cx64 expval_hamiltonian_squared                (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges);
    [[nodiscard]] cx64 expval_hamiltonian_squared                (const Eigen::Tensor<cx64, 3> &mps, const ModelFinite & model, const EdgesFinite & edges);
    [[nodiscard]] cx64 expval_hamiltonian_squared                (const Eigen::Tensor<cx64, 3> &mps, const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs, const env_pair<const EnvVar &> &envs);

    [[nodiscard]] double energy_minus_energy_shift               (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy                                  (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy_variance                         (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy_normalized                       (const StateFinite & state, const ModelFinite & model, const EdgesFinite & edges, double energy_min, double energy_max, MeasurementsTensorsFinite * measurements = nullptr);

    [[nodiscard]] extern double energy_shift                    (const TensorsFinite & tensors);
    [[nodiscard]] extern double energy_minus_energy_shift       (const TensorsFinite & tensors);
    [[nodiscard]] extern double energy                          (const TensorsFinite & tensors);
    [[nodiscard]] extern double energy_variance                 (const TensorsFinite & tensors);
    [[nodiscard]] extern double energy_normalized               (const TensorsFinite & tensors, double energy_minimum, double energy_maximum);

    [[nodiscard]] extern double energy_minus_energy_shift       (const StateFinite & state, const TensorsFinite & tensors, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy                          (const StateFinite & state, const TensorsFinite & tensors, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy_variance                 (const StateFinite & state, const TensorsFinite & tensors, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy_normalized               (const StateFinite & state, const TensorsFinite & tensors, double energy_minimum, double energy_maximum, MeasurementsTensorsFinite * measurements = nullptr);


    [[nodiscard]] double energy_minus_energy_shift               (const Eigen::Tensor<cx64,3> & multisite_mps, const ModelFinite & model, const EdgesFinite & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy                                  (const Eigen::Tensor<cx64,3> & multisite_mps, const ModelFinite & model, const EdgesFinite & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy_variance                         (const Eigen::Tensor<cx64,3> & multisite_mps, const ModelFinite & model, const EdgesFinite & edges, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] double energy_normalized                       (const Eigen::Tensor<cx64,3> & multisite_mps, const ModelFinite & model, const EdgesFinite & edges, double energy_min, double energy_max, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);


    [[nodiscard]] extern double energy_minus_energy_shift   (const Eigen::Tensor<cx64,3> &mps, const TensorsFinite & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy                      (const Eigen::Tensor<cx64,3> &mps, const TensorsFinite & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy_variance             (const Eigen::Tensor<cx64,3> &mps, const TensorsFinite & tensors, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double energy_normalized           (const Eigen::Tensor<cx64,3> &mps, const TensorsFinite & tensors, double energy_minimum, double energy_maximum, std::optional<svd::config> svd_cfg, MeasurementsTensorsFinite * measurements = nullptr);
    [[nodiscard]] extern double residual_norm               (const Eigen::Tensor<cx64, 3> &mps,
                                                             const Eigen::Tensor<cx64, 4> &mpo,
                                                             const Eigen::Tensor<cx64, 3> &envL,
                                                             const Eigen::Tensor<cx64, 3> &envR);
    [[nodiscard]] extern double residual_norm               (const Eigen::Tensor<cx64, 3> &mps,
                                                             const std::vector<Eigen::Tensor<cx64, 4>> &mpos,
                                                             const Eigen::Tensor<cx64, 3> &envL,
                                                             const Eigen::Tensor<cx64, 3> &envR);
    [[nodiscard]] extern double residual_norm                 (const Eigen::Tensor<cx64, 3> &mps,
                                                             const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                             const env_pair<const EnvEne &> &envs);
    [[nodiscard]] extern double residual_norm               (const Eigen::Tensor<cx64, 3> &mps,
                                                             const std::vector<std::reference_wrapper<const MpoSite>> &mpo_refs,
                                                             const env_pair<const EnvVar &> &envs);
    [[nodiscard]] extern double residual_norm_H1             (const TensorsFinite & tensors);
    [[nodiscard]] extern double residual_norm_H2             (const TensorsFinite & tensors);

    [[nodiscard]] extern double residual_norm_full           (const StateFinite &state, const ModelFinite &model, const EdgesFinite & edges);


    [[nodiscard]] extern cx64                     expectation_value      (const StateFinite & state, const std::vector<LocalObservableOp> & ops);
    [[nodiscard]] extern cx64                     expectation_value      (const StateFinite & state, const std::vector<LocalObservableMpo> & mpos);
    [[nodiscard]] extern cx64                     expectation_value      (const StateFinite & state1, const StateFinite & state2,
                                                                          const std::vector<Eigen::Tensor<cx64,4>> & mpos);
    [[nodiscard]] extern cx128                   expectation_value      (const StateFinite & state1, const StateFinite & state2,
                                                                          const std::vector<Eigen::Tensor<cx128,4>> & mpos_t);
    [[nodiscard]] extern cx64                     expectation_value      (const StateFinite & state1, const StateFinite & state2,
                                                                          const std::vector<Eigen::Tensor<cx64,4>> & mpos,
                                                                          const Eigen::Tensor<cx64,1> & ledge,
                                                                          const Eigen::Tensor<cx64,1> & redge);
    template<typename EnvType>
    fp64                                          expectation_value      (const Eigen::Tensor<fp64, 3> &mpsBra,
                                                                          const Eigen::Tensor<fp64, 3> &mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite>> &mpos,
                                                                          const env_pair<EnvType> &envs);

    template<typename EnvType>
    cx64                                          expectation_value      (const Eigen::Tensor<cx64, 3> &mpsBra,
                                                                          const Eigen::Tensor<cx64, 3> &mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite>> &mpos,
                                                                          const env_pair<EnvType> &envs);
    template<typename EnvType>
    [[nodiscard]] extern cx64                     expectation_value      (const std::vector<std::reference_wrapper<const MpsSite>> & mpsBra,
                                                                          const std::vector<std::reference_wrapper<const MpsSite>> & mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite>> & mpos,
                                                                          const env_pair<EnvType> & envs);
    template<typename EnvType>
    [[nodiscard]] extern cx64                     expectation_value      (const Eigen::Tensor<cx64, 3> &mpsBra,
                                                                          const Eigen::Tensor<cx64, 3> &mpsKet,
                                                                          const std::vector<std::reference_wrapper<const MpoSite>> &mpos,
                                                                          const env_pair<EnvType> &envs,
                                                                          std::optional<svd::config> svd_cfg);
    template<typename EnvType>
    [[nodiscard]] extern cx64                     expectation_value      (const Eigen::Tensor<cx64, 3> &multisite_mps,
                                                                          const std::vector<std::reference_wrapper<const MpoSite>> &mpos,
                                                                          const env_pair<EnvType> &envs,
                                                                          std::optional<svd::config> svd_cfg);
    [[nodiscard]] extern Eigen::Tensor<cx64, 1>   expectation_values     (const StateFinite & state, const Eigen::Tensor<cx64,2> &op);
    [[nodiscard]] extern Eigen::Tensor<cx64, 1>   expectation_values     (const StateFinite & state, const Eigen::Tensor<cx64,4> &mpo);
    [[nodiscard]] extern Eigen::Tensor<cx64, 1>   expectation_values     (const StateFinite & state, const Eigen::Matrix2cd &op);
    [[nodiscard]] extern cx64                     correlation            (const StateFinite & state, const Eigen::Tensor<cx64,2> &op1, const Eigen::Tensor<cx64,2> &op2, long pos1, long pos2);
    [[nodiscard]] extern Eigen::Tensor<cx64, 2>   correlation_matrix     (const StateFinite & state, const Eigen::Tensor<cx64,2> &op1, const Eigen::Tensor<cx64,2> &op2);
    [[nodiscard]] extern Eigen::Tensor<cx64, 2>   opdm                   (const StateFinite & state);
    [[nodiscard]] extern Eigen::Tensor<double, 1> opdm_spectrum          (const StateFinite & state);
    [[nodiscard]] extern std::array<Eigen::Tensor<double, 1>, 3>
                                                  expectation_values_xyz (const StateFinite & state);
    [[nodiscard]] extern std::array<double, 3>    expectation_value_xyz  (const StateFinite & state);
    [[nodiscard]] extern std::array<Eigen::Tensor<double,2>, 3>
                                                  correlation_matrix_xyz (const StateFinite & state);
    [[nodiscard]] extern double                   structure_factor       (const StateFinite & state, const Eigen::Tensor<cx64, 2> &correlation_matrix);
    [[nodiscard]] extern std::array<double, 3>    structure_factor_xyz   (const StateFinite & state);
                  extern void                     parity_components(const StateFinite &state, const Eigen::Matrix2cd &paulimatrix);


}

/* clang-format on */
