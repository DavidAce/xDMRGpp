#pragma once
#include "general/sfinae.h"
#include <string>

template<typename Scalar>
class StateFinite;
template<typename Scalar>
class ModelFinite;
template<typename Scalar>
class EdgesFinite;
template<typename Scalar>
class TensorsFinite;
class AlgorithmStatus;
enum class StorageLevel;
enum class StorageEvent;
enum class StoragePolicy;
enum class CopyPolicy;
enum class AlgorithmType;
struct StorageInfo;
namespace h5pp {
    class File;
    namespace hid {
        class h5t;
    }
    struct DimsType;
}
namespace tools::common::h5 {
    namespace save {
        bool should_save(const StorageInfo &sinfo, StoragePolicy policy);
    }
    struct MpsInfo;
}

namespace tools::finite::h5 {
    template<typename Scalar>
    using RealScalar = decltype(std::real(std::declval<Scalar>()));
    /* clang-format off */
    namespace save {
//        void bootstrap_save_log(std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> &save_log, const h5pp::File &h5file, const std::vector<std::string_view> &links);
//        void bootstrap_save_log(std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> &save_log, const h5pp::File &h5file, std::string_view link);
        template<typename T>
        void data_as_table(h5pp::File &h5file, const StorageInfo & sinfo, const T * const data, size_t size, std::string_view table_name, std::string_view table_title, std::string_view fieldname);
        template<typename T>
        void data_as_table(h5pp::File &h5file, const StorageInfo & sinfo, const T & data, std::string_view table_name, std::string_view table_title, std::string_view fieldname){
            if constexpr(sfinae::is_std_optional_v<T>){
                if(data.has_value()) data_as_table(h5file, sinfo, data.value(), table_name, table_title, fieldname);
            }
            else if constexpr (sfinae::has_data_v<T> and sfinae::has_size_v<T>) data_as_table(h5file, sinfo, data.data(), static_cast<size_t>(data.size()), table_name, table_title, fieldname);
            else if constexpr (std::is_arithmetic_v<T>) data_as_table(h5file, sinfo, &data, 1, table_name, table_title, fieldname);
            else static_assert(sfinae::invalid_type_v<T> and "Datatype must have .data() and .size() (or be std::optional of such)");
        }
        template<typename T>
        void data_as_table_vla(h5pp::File &h5file, const  StorageInfo & sinfo, const std::vector<T> & data, const h5pp::hid::h5t & h5elem_t, std::string_view table_name, std::string_view table_title, std::string_view fieldname);
        extern int decide_layout(std::string_view prefix_path);
        // template<typename T>
        // extern void data      (h5pp::File & h5file, const StorageInfo & sinfo,  const T &data, std::string_view data_name, CopyPolicy copy_policy);
        template<typename T>
        void data      (h5pp::File & h5file, const StorageInfo & sinfo,  const T *data, const std::vector<long> & dims, std::string_view data_name, std::string_view prefix, CopyPolicy copy_policy);
        // template<typename T>
        // extern void data      (h5pp::File & h5file, const StorageInfo & sinfo, const T &data, std::string_view data_name, std::string_view prefix);
        template<typename Scalar> void bonds     (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar> void state     (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void model     (h5pp::File & h5file, const StorageInfo & sinfo, const ModelFinite<Scalar> & model);
        template<typename Scalar>  void mpo       (h5pp::File & h5file, const StorageInfo & sinfo, const ModelFinite<Scalar> & model);

        template<typename Scalar>  void correlations (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);

        template<typename Scalar>  void measurements                    (h5pp::File & h5file, const StorageInfo & sinfo, const TensorsFinite<Scalar> & tensors, const AlgorithmStatus & status);
        template<typename Scalar>  void measurements                    (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state, const ModelFinite<Scalar> & model, const EdgesFinite<Scalar> & edges, const AlgorithmStatus & status);
        template<typename Scalar>  void bond_dimensions                 (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void schmidt_values                  (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void truncation_errors               (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void entanglement_entropies          (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void subsystem_entanglement_entropies(h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void information_lattice             (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void information_per_scale           (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void information_center_of_mass      (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void renyi_entropies                 (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void number_entropies                (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void expectations                    (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void structure_factors               (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void opdm                            (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void opdm_spectrum                   (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);
        template<typename Scalar>  void number_probabilities            (h5pp::File & h5file, const StorageInfo & sinfo, const StateFinite<Scalar> & state);

        template<typename Scalar> [[nodiscard]] StorageInfo get_storage_info(const StateFinite<Scalar> & state, const AlgorithmStatus &status);

        template<typename Scalar>
        void simulation(h5pp::File &h5file,
                               const TensorsFinite<Scalar> & tensors,
                               const AlgorithmStatus &status,
                               CopyPolicy copy_policy);
        template<typename Scalar>
        void simulation(h5pp::File &h5file,
                               const StateFinite<Scalar> & state,
                               const ModelFinite<Scalar> & model,
                               const EdgesFinite<Scalar> & edges,
                               const AlgorithmStatus &status,
                               CopyPolicy copy_policy);

    }
    namespace find{
        extern std::string find_unitary_circuit(const h5pp::File &h5file, AlgorithmType algo_type, std::string_view name,
                                                                  size_t iter);
    }
    namespace load {
        using MpsInfo = tools::common::h5::MpsInfo;
        template<typename Scalar>  void simulation (const h5pp::File & h5file, std::string_view  state_prefix, TensorsFinite<Scalar> & tensors, AlgorithmStatus & status, AlgorithmType algo_type);
        template<typename Scalar>  void state   (const h5pp::File & h5file, std::string_view  state_prefix, StateFinite<Scalar> & state, MpsInfo & mpsinfo);
        template<typename Scalar>  void model   (const h5pp::File & h5file, AlgorithmType algo_type, ModelFinite<Scalar> & model);
        template<typename Scalar>  void validate (const h5pp::File & h5file, std::string_view  state_prefix, TensorsFinite<Scalar> & tensors, AlgorithmStatus & status, AlgorithmType algo_type);
    }

    namespace tmp{
        extern void copy(const AlgorithmStatus &status, const h5pp::File &h5file, StorageEvent storage_reason, std::optional<CopyPolicy> copy_policy);

    }

//    namespace load{

//        extern std::vector<SimulationTask>
//            getTaskList(const h5pp::File &h5file, std::string_view sim_name, std::string_view state_prefix, const TensorsFinite & tensors, const class_algorithm_status & status);
//    }
    /* clang-format on */
}
