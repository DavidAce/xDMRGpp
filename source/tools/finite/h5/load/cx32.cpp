#include "../load.impl.h"

using Scalar = cx32;


/* clang-format off */
template void tools::finite::h5::load::simulation(const h5pp::File &h5file, std::string_view state_prefix, TensorsFinite<Scalar> &tensors, AlgorithmStatus &status,AlgorithmType algo_type);

template void tools::finite::h5::load::state(const h5pp::File &h5file, std::string_view state_prefix, StateFinite<Scalar> &state, MpsInfo &info);

template void tools::finite::h5::load::model(const h5pp::File &h5file, AlgorithmType algo_type, ModelFinite<Scalar> &model) ;

template void tools::finite::h5::load::validate(const h5pp::File &h5file, std::string_view state_prefix, TensorsFinite<Scalar> &tensors, AlgorithmStatus &status, AlgorithmType algo_type);

/* clang-format on */
