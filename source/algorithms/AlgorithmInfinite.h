#pragma once
#include "algorithms/AlgorithmBase.h"
#include "tensors/TensorsInfinite.h"
template<typename Scalar> class StateInfinite;
template<typename Scalar> class ModelInfinite;
template<typename Scalar> class EdgesInfinite;

template<typename Scalar>
class AlgorithmInfinite : public AlgorithmBase {
    protected:
    using RealScalar = decltype(std::real(std::declval<Scalar>()));

    public:
    // Inherit the constructor of class_algorithm_base
    using AlgorithmBase::AlgorithmBase;
    explicit AlgorithmInfinite(std::shared_ptr<h5pp::File> h5ppFile_, OptRitz opt_ritz_, AlgorithmType algo_type);
    TensorsInfinite<Scalar> tensors;

    /* clang-format off */
    void run()                                                                                             final;
    void run_preprocessing()                                                                               override;
    void run_postprocessing()                                                                              override;
    void clear_convergence_status()                                                                        override;
    void update_precision_limit(std::optional<double> energy_upper_bound = std::nullopt)                   final;
    void update_bond_dimension_limit()                                                                     final;
    void update_truncation_error_limit()                                                                   final;

    void initialize_model();
    void initialize_state(ResetReason reason,
                         std::optional<std::string> sector = std::nullopt,
                         std::optional<bool> use_eigenspinors = std::nullopt, std::optional<std::string> pattern = std::nullopt);


    void write_to_file(StorageEvent storage_event = StorageEvent::ITERATION, CopyPolicy copy_policy = CopyPolicy::TRY) final;
    void print_status()                                                                                                 final;
    void print_status_full()                                                                                            final;
    /* clang-format on */

    void check_convergence_variance_mpo(std::optional<RealScalar> threshold = std::nullopt, std::optional<RealScalar> sensitivity = std::nullopt);
    void check_convergence_variance_ham(std::optional<RealScalar> threshold = std::nullopt, std::optional<RealScalar> sensitivity = std::nullopt);
    void check_convergence_variance_mom(std::optional<RealScalar> threshold = std::nullopt, std::optional<RealScalar> sensitivity = std::nullopt);
    void check_convergence_entg_entropy(std::optional<RealScalar> sensitivity = std::nullopt);

    std::vector<RealScalar> var_mpo_iter; // History of energy variances (from mpo) at each iteration
    std::vector<RealScalar> var_ham_iter;
    std::vector<RealScalar> var_mom_iter;
    std::vector<RealScalar> entropy_iter;
};
