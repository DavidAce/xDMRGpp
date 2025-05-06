#pragma once

#include <string>
#include <vector>
template<typename Scalar>
class TensorsInfinite;
template<typename Scalar>
class StateInfinite;
template<typename Scalar>
class ModelInfinite;
template<typename Scalar>
class EdgesInfinite;
class AlgorithmStatus;
enum class StorageLevel;
enum class SimulationTask;
namespace h5pp {
    class File;
}

namespace tools::infinite::h5 {
    /* clang-format off */
    namespace save{
        extern int decide_layout(std::string_view prefix_path);
        template<typename Scalar>  void bonds            (h5pp::File & h5file, const StorageInfo & sinfo, const StateInfinite<Scalar> & state);
        template<typename Scalar>  void state            (h5pp::File & h5file, const StorageInfo & sinfo, const StateInfinite<Scalar> & state);
        template<typename Scalar>  void edges            (h5pp::File & h5file, const StorageInfo & sinfo, const EdgesInfinite<Scalar> & edges);
        template<typename Scalar>  void model            (h5pp::File & h5file, const StorageInfo & sinfo, const ModelInfinite<Scalar> & model);
        template<typename Scalar>  void mpo              (h5pp::File & h5file, const StorageInfo & sinfo, const ModelInfinite<Scalar> & model);
        template<typename Scalar>  void measurements     (h5pp::File & h5file, const StorageInfo & sinfo, const TensorsInfinite<Scalar> & tensors, const AlgorithmStatus & status);
    }

    namespace load{
        template<typename Scalar>  void tensors (const h5pp::File & h5file, std::string_view  state_prefix, TensorsInfinite<Scalar> & state, AlgorithmStatus & status);
        template<typename Scalar>  void state   (const h5pp::File & h5file, std::string_view  state_prefix, StateInfinite<Scalar> & state, const AlgorithmStatus & status);
        template<typename Scalar>  void model   (const h5pp::File & h5file, std::string_view  state_prefix, ModelInfinite<Scalar> & state, const AlgorithmStatus & status);
        template<typename Scalar>  void validate (const h5pp::File & h5file, std::string_view state_prefix, const TensorsInfinite<Scalar> & state, const AlgorithmStatus & status);
        template<typename Scalar>  std::vector<SimulationTask>
            getTaskList(const h5pp::File &h5file, std::string_view state_prefix, const StateInfinite<Scalar> & state, const AlgorithmStatus & status);

    }
    /* clang-format on */
}
