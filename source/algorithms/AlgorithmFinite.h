#pragma once

#include "algorithms/AlgorithmBase.h"
#include "math/svd/config.h"
#include "measure/MeasurementsStateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/h5/storage_info.h"
#include "tools/finite/h5.h"
#include "tools/finite/opt_meta.h"
#include <general/sfinae.h>

struct BondExpansionConfig;
enum class BondExpansionOrder;
template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
namespace tools::finite::opt {
    template<typename Scalar> class opt_mps;
    struct OptMeta;
}

// class h5pp_table_measurements_finite;
template<typename Scalar>
class AlgorithmFinite : public AlgorithmBase {
    protected:
    using OptMeta    = tools::finite::opt::OptMeta;
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    private:
    size_t                             dmrg_blocksize        = 1; // Number of sites in a DMRG step. This is updated by the information per scale mass center
    double                             dmrg_eigs_tol         = 1e-12; // Tolerance for the iterative eigenvalue solver
    double                             eigval_upper_bound    = 1;
    size_t                             iter_last_bond_reduce = 0;
    std::optional<std::vector<size_t>> sites_mps, sites_mpo; // Used when moving sites
    protected:
    int get_eigs_iter_max() const;

    public:
    // Inherit the constructor of class_algorithm_base
    using AlgorithmBase::AlgorithmBase;
    explicit AlgorithmFinite(OptRitz opt_ritz_, AlgorithmType algo_type);
    explicit AlgorithmFinite(std::shared_ptr<h5pp::File> h5ppFile_, OptRitz opt_ritz_, AlgorithmType algo_type);
    TensorsFinite<Scalar> tensors; // State, model and edges

    // using TensorsType =
    // std::variant<TensorsFinite<fp32>, TensorsFinite<fp64>, TensorsFinite<fp128>, TensorsFinite<cx32>, TensorsFinite<cx64>, TensorsFinite<cx128>>;
    // TensorsType tensors;
    // std::unique_ptr<TensorsFiniteI> tensors;
    // template<typename T>
    // TensorsFinite<T> get_tensors() {
    //     if(std::holds_alternative<TensorsFinite<T>>(tensors)) return std::get<TensorsFinite<T>>(tensors);
    //     std::visit(
    //         [&](const auto &tensor) {
    //             throw except::runtime_error("Tensors holds type {}, but {} was requested.", sfinae::type_name<decltype(tensor)>(),
    //                                         sfinae::type_name<TensorsFinite<T>>());
    //         },
    //         tensors);
    //     throw;
    // }
    // static_assert(std::is_same_v<decltype(get_tensors<cx32>), TensorsFinite<cx32>>);

    size_t                   projected_iter = 0; /*!< The last iteration when projection was tried */
    std::optional<OptAlgo>   last_optalgo   = std::nullopt;
    std::optional<OptSolver> last_optsolver = std::nullopt;

    public:
    virtual void                resume()                = 0;
    virtual void                run_default_task_list() = 0;
    void                        try_projection(std::optional<std::string> target_axis = std::nullopt);
    void                        set_parity_shift_mpo(std::optional<std::string> target_axis = std::nullopt);
    void                        set_parity_shift_mpo_squared(std::optional<std::string> target_axis = std::nullopt);
    void                        try_moving_sites();
    BondExpansionResult<Scalar> expand_bonds(BondExpansionOrder order);
    void                        move_center_point(std::optional<long> num_moves = std::nullopt);
    virtual void                set_energy_shift_mpo(); // We override this in xdmrg
    void                        rebuild_tensors();
    void                        update_precision_limit(std::optional<double> energy_upper_bound = std::nullopt) final;
    void                        update_bond_dimension_limit() final;
    void                        reduce_bond_dimension_limit(double rate, UpdatePolicy when, StorageEvent storage_event);
    void                        update_truncation_error_limit() final;
    void                        update_mixing_factor();
    void                        update_dmrg_blocksize();
    void                        update_eigs_tolerance();
    void                        initialize_model();
    void                        run() final;
    void                        run_rbds_analysis();
    void                        run_rtes_analysis();
    void                        run_postprocessing() override;
    [[nodiscard]] OptMeta       get_opt_meta();
    void                        clear_convergence_status() override;
    void                        initialize_state(ResetReason reason, StateInit state_init, std::optional<StateInitType> state_type = std::nullopt,
                                                 std::optional<std::string> axis = std::nullopt, std::optional<bool> use_eigenspinors = std::nullopt,
                                                 std::optional<std::string> pattern = std::nullopt, std::optional<long> bond_lim = std::nullopt,
                                                 std::optional<double> trnc_lim = std::nullopt);

    void write_to_file(StorageEvent storage_event = StorageEvent::ITERATION, CopyPolicy copy_policy = CopyPolicy::TRY) override;
    void print_status() override;
    void print_status_full() final;
    void check_convergence() override;
    void check_convergence_variance(std::optional<RealScalar> threshold = std::nullopt, std::optional<RealScalar> saturation_sensitivity = std::nullopt);
    void check_convergence_locinfoscale(std::optional<RealScalar> saturation_sensitivity = std::nullopt);
    void check_convergence_entanglement(std::optional<RealScalar> saturation_sensitivity = std::nullopt);
    void check_convergence_spin_parity_sector(std::string_view target_axis, double threshold = 1e-8);
    void check_convergence_truncation_error();
    // template<typename T>
    // void write_to_file(const StateFinite<T> &state, const ModelFinite<T> &model, const EdgesFinite<T> &edges, StorageEvent storage_event,
    // CopyPolicy copy_policy = CopyPolicy::TRY);
    // template<typename T>
    // void write_to_file(const T &data, std::string_view name, StorageEvent storage_event, CopyPolicy copy_policy = CopyPolicy::TRY);

    template<typename T>
    void write_to_file(const StateFinite<T> &state, const ModelFinite<T> &model, const EdgesFinite<T> &edges, StorageEvent storage_event,
                       CopyPolicy copy_policy = CopyPolicy::TRY) {
        if(not h5file) return;
        status.event = storage_event;
        tools::finite::h5::save::simulation(*h5file, state, model, edges, status, copy_policy);
        status.event = StorageEvent::NONE;
    }

    template<typename T>
    void write_tensor_to_file(const T &data, std::string_view name, StorageEvent storage_event, CopyPolicy copy_policy = CopyPolicy::TRY) {
        if(not h5file) return;
        status.event      = storage_event;
        auto        sinfo = StorageInfo(status, tensors.get_state().get_name(), tensors.get_model().model_type);
        std::string prefix;
        switch(sinfo.storage_event) {
            case StorageEvent::MODEL: {
                prefix = fmt::format("{}/model", sinfo.algo_name);
                break;
            }
            default: {
                prefix = sinfo.get_state_prefix();
                break;
            }
        }
        auto dims = std::vector<long>{data.dimensions().begin(), data.dimensions().end()};
        tools::finite::h5::save::data(*h5file, sinfo, data.data(), dims, name, prefix, copy_policy);
        status.event = StorageEvent::NONE;
    }

    struct log_entry {
        AlgorithmStatus         status;
        RealScalar              energy;
        RealScalar              variance;
        RealScalar              locinfoscale;
        double                  time;
        std::vector<RealScalar> entanglement_entropies;
        log_entry(const AlgorithmStatus &s, const TensorsFinite<Scalar> &t);
    };
    std::vector<log_entry> algorithm_history;
    RealScalar             ene_latest    = 0.0;
    RealScalar             var_latest    = 1.0;
    RealScalar             ene_bondex    = 1.0;
    RealScalar             var_bondex    = 1.0;
    RealScalar             ene_delta     = 0.0;
    RealScalar             var_delta     = 0.0;
    RealScalar             ene_delta_opt = 0.0;
    RealScalar             ene_delta_svd = 0.0;
    RealScalar             var_delta_opt = 0.0; // Variance change from optimization
    RealScalar             var_delta_svd = 0.0; // Variance change from truncation
    RealScalar             std_delta_opt = 0.0; // Standard deviation of energy change from optimization
    RealScalar             std_delta_svd = 0.0; // Standard deviation of energy change from truncation

    std::deque<double> qexp_history;
    // std::vector<double> alphas;
};
