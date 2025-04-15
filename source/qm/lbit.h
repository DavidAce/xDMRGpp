#pragma once
#include "gate.h"
#include "math/svd/config.h"
#include "math/tenx/fwd_decl.h"
#include "qm.h"
#include <complex>
#include <optional>
#include <string_view>
#include <vector>

enum class LbitCircuitGateWeightKind;
enum class LbitCircuitGateMatrixKind;
enum class MeanType;
template<typename Scalar>
class StateFinite;
namespace h5pp {
    class File;
    namespace hid {
        class h5t;
    }
};

namespace qm::lbit {
    template<typename Scalar>
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    /*! The values that we need to recreate a unitary gate exactly */
    struct UnitaryGateParameters {
        size_t                layer; /*!< Layer number of this gate */
        std::array<size_t, 2> sites; /*!< The two sites which this site connects */
        double                f;     /*!< Mixing factor  */
        double                w;     /*!< Gate weight value = exp(-2|h_i - h_i+1|) */
        std::array<double, 4> theta; /*!< Random real valued thetas (for the mixing term)  */
        double l; /*!< lambda parameter used in the Hermitian matrix g8mx for the unitary circuit gates: controls the size of the sz_i*sz_j terms */
        std::complex<double>      c;     /*!< Random complex valued c (exchange term)  */
        LbitCircuitGateWeightKind wkind; /*!< Weights w_i in the unitary 2-site gates. Choose [IDENTITY, EXPDECAY] for 1 or exp(-2|h[i] - h[i+1]|), h are onsite
                                            fields in the Hamiltonian */
        LbitCircuitGateMatrixKind mkind; /*!< MATRIX_(V1|V2|V3) controls the kind o Hermitian matrix used in the unitary circuit gates */

        [[nodiscard]] static const h5pp::hid::h5t                      get_h5_type();
        [[nodiscard]] constexpr static std::array<std::string_view, 9> get_parameter_names() noexcept;
        void                                                           print_parameter_names() const noexcept;
        void                                                           print_parameter_values() const noexcept;
        [[nodiscard]] std::string                                      fmt_value(std::string_view p) const;

        static const h5pp::hid::h5t get_h5t_enum_umkind();
        static const h5pp::hid::h5t get_h5t_enum_uwkind();
    };

    struct UnitaryGateProperties {
        size_t                    sites;  /*!< Width of the circuit, i.e. system length/number of sites */
        size_t                    depth;  /*!< Number of layers in the unitary circuit (1 layer connects all neighbors once) */
        double                    fmix;   /*!< The mixing factor f in exp(-ifwM) */
        double                    lambda; /*!< The mixing factor f in exp(-ifwM) */
        LbitCircuitGateWeightKind wkind; /*!< Weights w_i in the unitary 2-site gates. Choose [IDENTITY, EXPDECAY] for 1 or exp(-2|h[i] - h[i+1]|), h are onsite
                                            fields in the Hamiltonian */
        LbitCircuitGateMatrixKind                  mkind;   /*!< MATRIX_(V1|V2|V3) controls the kind of Hermitian matrix used in the unitary circuit gates */
        double                                     hmean;   /*!< mean of random onsite fields */
        double                                     hwdth;   /*!< width of random onsite fields (st.dev. if normal) */
        std::string_view                           hdist;   /*!< distribution of onsite fields */
        mutable std::vector<double>                hvals;   /*!< onsite fields of the l-bit hamiltonian, needed for type == UnitaryGateWeight::EXPDECAY */
        mutable std::vector<std::vector<qm::Gate>> ulayers; /*!< The generated unitary circuit of two-site gates */
        UnitaryGateProperties() = default;
        UnitaryGateProperties(const std::vector<double> &h);
        std::string    string() const;
        void           randomize_hvals() const;
        bool           keep_circuit = false;
        mutable size_t layer_count  = 0;

        mutable std::vector<UnitaryGateParameters> circuit;
    };

    void write_unitary_circuit_parameters(h5pp::File &file, std::string_view table_path, const std::vector<UnitaryGateParameters> &circuit);
    std::vector<UnitaryGateParameters> read_unitary_circuit_parameters(const h5pp::File &file, std::string_view table_path);
    std::vector<std::vector<qm::Gate>> read_unitary_2site_gate_layers(const h5pp::File &file, std::string_view table_path);
    std::vector<std::vector<qm::Gate>> get_unitary_2site_gate_layers(const std::vector<UnitaryGateParameters> &circuit);

    struct lbitSupportAnalysis {
        Eigen::Tensor<fp64, 5> cls_avg_fit; // Characteristic length-scale of lbits from linear regression of log data
        Eigen::Tensor<fp64, 5> cls_avg_rms; // Root mean squared deviation
        Eigen::Tensor<fp64, 5> cls_avg_rsq; // R-squared or coefficient of determination
        Eigen::Tensor<fp64, 5> cls_typ_fit; // Characteristic length-scale of lbits from linear regression of log data
        Eigen::Tensor<fp64, 5> cls_typ_rms; // Root mean squared deviation
        Eigen::Tensor<fp64, 5> cls_typ_rsq; // R-squared or coefficient of determination
        Eigen::Tensor<fp64, 6> corravg;     // The lbit correlation matrix of l-bits averaged over site and disorder
        Eigen::Tensor<fp64, 6> corrtyp;     // The lbit correlation matrix of l-bits geometrically averaged over site and disorder
        Eigen::Tensor<fp64, 6> correrr;     // The sterr of l-bits: permuted and averaged over site and disorder
        Eigen::Tensor<fp64, 8> corrmat;     // The raw data from l-bit correlation matrices or the trace O(i,j) for each realization
        Eigen::Tensor<fp64, 8> corroff;     // The offset lbit correlation matrices O(i, |i-j|) for each realization.
        lbitSupportAnalysis() {
            cls_avg_fit.setZero();
            cls_avg_rms.setZero();
            cls_avg_rsq.setZero();
            cls_typ_fit.setZero();
            cls_typ_rms.setZero();
            cls_typ_rsq.setZero();
            corravg.setZero();
            corrtyp.setZero();
            correrr.setZero();
            corrmat.setZero();
            corroff.setZero();
        }
        lbitSupportAnalysis(size_t ndpth, size_t nfmix, size_t nlambda, size_t nweight, size_t nmatrix, size_t nreps, size_t nsize) : lbitSupportAnalysis() {
            auto idpth   = static_cast<Eigen::Index>(ndpth);
            auto ifmix   = static_cast<Eigen::Index>(nfmix);
            auto ilambda = static_cast<Eigen::Index>(nlambda);
            auto iweight = static_cast<Eigen::Index>(nweight);
            auto imatrix = static_cast<Eigen::Index>(nmatrix);
            auto ireps   = static_cast<Eigen::Index>(nreps);
            auto isize   = static_cast<Eigen::Index>(nsize);
            cls_avg_fit.resize(idpth, ifmix, ilambda, iweight, imatrix);
            cls_avg_rms.resize(idpth, ifmix, ilambda, iweight, imatrix);
            cls_avg_rsq.resize(idpth, ifmix, ilambda, iweight, imatrix);
            cls_typ_fit.resize(idpth, ifmix, ilambda, iweight, imatrix);
            cls_typ_rms.resize(idpth, ifmix, ilambda, iweight, imatrix);
            cls_typ_rsq.resize(idpth, ifmix, ilambda, iweight, imatrix);
            corravg.resize(idpth, ifmix, ilambda, iweight, imatrix, isize);
            corrtyp.resize(idpth, ifmix, ilambda, iweight, imatrix, isize);
            correrr.resize(idpth, ifmix, ilambda, iweight, imatrix, isize);
            corrmat.resize(idpth, ifmix, ilambda, iweight, imatrix, ireps, isize, isize);
            corroff.resize(idpth, ifmix, ilambda, iweight, imatrix, ireps, isize, isize);
        }
    };

    /* clang-format off */
    extern Eigen::Tensor<cx64, 2>               get_unitary_layer_as_tensor(const std::vector<qm::Gate> &unitary_layer);
    extern Eigen::Tensor<cx64, 2>               get_unitary_circuit_as_tensor(const std::vector<std::vector<qm::Gate>> &unitary_circuit);
    extern Eigen::Tensor<cx64, 2>               get_time_evolution_operator(cx128 delta_t, const Eigen::Tensor<cx64, 2> &hamiltonian);
    extern std::vector<qm::Gate>                get_time_evolution_gates(cx128 delta_t, const std::vector<qm::Gate> &hams_nsite);
    extern std::vector<qm::SwapGate>            get_time_evolution_swap_gates(cx128 delta_t, const std::vector<qm::SwapGate> &hams_nsite);
//    extern std::vector<qm::Gate>                get_unitary_2gate_layer(size_t sites, double fmix);
    extern qm::Gate                             get_unitary_2site_gate(const UnitaryGateParameters &u);
    extern std::vector<qm::Gate>                create_unitary_2site_gate_layer(const qm::lbit::UnitaryGateProperties &u);
    extern std::vector<Eigen::Tensor<cx64, 4>>  get_unitary_mpo_layer(const std::vector<qm::Gate> & ulayer, std::optional<svd::config> cfg = std::nullopt);
    extern std::vector<Eigen::Tensor<cx64, 4>>  get_unitary_mpo_layer(const UnitaryGateProperties & u);
    extern std::vector<Eigen::Tensor<cx64, 4>>  merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx64, 4>> & mpos_dn, const std::vector<Eigen::Tensor<cx64, 4>> & mpos_up, bool adj_dn = false);
    extern std::vector<Eigen::Tensor<cx64, 4>>  merge_unitary_mpo_layers(const std::vector<Eigen::Tensor<cx64, 4>> & mpos_dn,
                                                                         const std::vector<Eigen::Tensor<cx64, 4>> & mpos_md,
                                                                         const std::vector<Eigen::Tensor<cx64, 4>> & mpos_up);
    extern std::vector<Eigen::Tensor<cx64, 4>>  merge_unitary_mpo_layers(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> & mpos);
    extern std::vector<Eigen::Tensor<cx64, 2>>  get_time_evolution_operators_2site(size_t sites, cx128 delta_t, const std::vector<Eigen::Tensor<cx64, 2>> &twosite_hams);
    extern std::vector<Eigen::Tensor<cx64, 2>>  get_time_evolution_operators_3site(size_t sites, cx128 delta_t, const std::vector<Eigen::Tensor<cx64, 2>> &hams_3site);
    extern std::vector<Eigen::Tensor<cx64, 4>>  get_time_evolution_mpos(cx128 delta_t, const std::vector<Eigen::Tensor<cx64, 4>> &mpos);
    extern cx64                                 get_lbit_2point_correlator1(const std::vector<std::vector<qm::Gate>> &unitary_circuit, const Eigen::Matrix2cd &rho, size_t pos_rho, const Eigen::Matrix2cd &sig, size_t pos_sig);
    extern cx64                                 get_lbit_2point_correlator2(const std::vector<std::vector<qm::Gate>> &unitary_circuit, const Eigen::Matrix2cd &szi, size_t pos_szi, const Eigen::Matrix2cd &szj, size_t pos_szj, long len);
    extern cx64                                 get_lbit_2point_correlator3(const std::vector<std::vector<qm::Gate>> &unitary_circuit, const Eigen::Matrix2cd &szi, size_t pos_szi, const Eigen::Matrix2cd &szj, size_t pos_szj, long len);
    extern cx64                                 get_lbit_2point_correlator4(const std::vector<Eigen::Tensor<cx64, 4>> &mpo_layer, const Eigen::Matrix2cd &szi, size_t pos_szi, const Eigen::Matrix2cd &szj, size_t pos_szj);
    extern Eigen::Tensor<cx64, 1>               get_lbit_2point_correlator5(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers, const Eigen::Matrix2cd &szi, size_t pos_szi, const Eigen::Matrix2cd &szj);
    extern Eigen::Tensor<cx64, 1>               get_lbit_2point_correlator6(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers, const Eigen::Matrix2cd &szi, size_t pos_szi, const Eigen::Matrix2cd &szj);
    extern Eigen::Tensor<fp64, 2>               get_lbit_correlation_matrix(const std::vector<Eigen::Tensor<cx64, 4>> &mpo_layer);
    extern Eigen::Tensor<fp64, 2>               get_lbit_correlation_matrix(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers);
    extern Eigen::Tensor<fp64, 2>               get_lbit_correlation_matrix2(const std::vector<std::vector<Eigen::Tensor<cx64, 4>>> &mpo_layers);
    extern Eigen::Tensor<fp64, 2>               get_lbit_correlation_matrix(const std::vector<std::vector<qm::Gate>> &unitary_circuit, size_t sites);
    extern Eigen::Tensor<fp64, 2>               get_lbit_correlation_matrix(const std::vector<std::vector<qm::Gate>> &unitary_circuit, size_t sites, size_t max_num_states, double tol);
    extern std::vector<Eigen::Tensor<fp64, 2>>  get_lbit_correlation_matrices(const UnitaryGateProperties &uprop, size_t reps, bool randomize_fields, bool use_mpo);
    extern std::vector<Eigen::Tensor<fp64, 2>>  get_lbit_correlation_matrices2(const UnitaryGateProperties &uprop, size_t reps, bool randomize_fields, bool use_mpo);
    extern std::tuple<Eigen::Tensor<fp64, 2>, Eigen::Tensor<fp64, 2>, Eigen::Tensor<fp64, 2>>
                                                get_lbit_correlation_statistics(const std::vector<Eigen::Tensor<fp64, 2>> &lbit_corrmats);
    extern std::tuple<double,
                      double,
                      double,
                      std::vector<double>,
                      size_t>                   get_characteristic_length_scale(const Eigen::Tensor<fp64, 2> &lbit_corrmat_disorder_mean, MeanType);
//    extern lbitSupportAnalysis                  get_lbit_support_analysis(const std::vector<size_t> &udepth_vec, const std::vector<double> &fmix_vec,
//                                                                          size_t reps, const UnitaryGateProperties & uprop, bool randomize_fields = false);
//    extern lbitSupportAnalysis                  get_lbit_support_analysis(const std::vector<UnitaryGateProperties> &uprops, bool randomize_fields = false);

    extern lbitSupportAnalysis                  get_lbit_support_analysis(
                                                                  const UnitaryGateProperties      & u_defaults,
                                                                  std::vector<size_t>                  udpths = {},
                                                                  std::vector<double>                  ufmixs = {},
                                                                  std::vector<double>                  ulambdas = {},
                                                                  std::vector<LbitCircuitGateWeightKind>   uweights = {},
                                                                  std::vector<LbitCircuitGateMatrixKind>   umatrixs = {}
                                                                );
    template<typename Scalar>
    StateFinite<Scalar> transform_to_real_basis(const StateFinite<Scalar> &lbit_state,
                                        const std::vector<std::vector<qm::Gate>> & unitary_gates_2site_layers,
                                        svd::config svd_cfg);
    template<typename Scalar>
    StateFinite<Scalar> transform_to_real_basis(const StateFinite<Scalar> &lbit_state,
                                        const std::vector<std::vector<Eigen::Tensor<Scalar, 4>>> &unitary_gates_mpo_layers,
                                        const Eigen::Tensor<std::complex<double>, 1> & ledge,
                                        const Eigen::Tensor<std::complex<double>, 1> & redge,
                                        svd::config svd_cfg);
    template<typename Scalar>
    StateFinite<Scalar> transform_to_lbit_basis(const StateFinite<Scalar> &real_state,
                                        const std::vector<std::vector<qm::Gate>> & unitary_gates_2site_layers,
                                        svd::config svd_cfg);
    template<typename Scalar>
    StateFinite<Scalar> transform_to_lbit_basis(const StateFinite<Scalar> &real_state,
                                        const std::vector<std::vector<Eigen::Tensor<Scalar, 4>>> &unitary_gates_mpo_layers,
                                        const Eigen::Tensor<std::complex<double>, 1> & ledge,
                                        const Eigen::Tensor<std::complex<double>, 1> & redge,
                                        svd::config svd_cfg);
    /* clang-format on */
}