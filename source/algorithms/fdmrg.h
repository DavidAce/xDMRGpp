#pragma once
#include "AlgorithmFinite.h"
#include <deque>
class class_h5table_measurements_finite;
namespace tools::finite::opt {
    class opt_mps;
    struct OptMeta;
}
/*!
// * \brief Class that runs the finite DMRG algorithm.
 */

template<typename Scalar> class StateFinite;
class fdmrg : public AlgorithmFinite {
    using OptMeta = tools::finite::opt::OptMeta;

    private:
    std::optional<double> variance_before_step = std::nullopt;
    std::string_view      get_state_name() const;

    public:
    // Inherit the constructor of class_algorithm_base
    using AlgorithmFinite::AlgorithmFinite;
             fdmrg();
    explicit fdmrg(std::shared_ptr<h5pp::File> h5file_);
    void     resume() final;
    void     run_task_list(std::deque<fdmrg_task> &task_list);
    void     run_default_task_list() final;
    void     run_preprocessing() final;
    void     run_algorithm() final;
    void     update_state() final;
};
