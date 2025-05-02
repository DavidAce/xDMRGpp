#include "../save.impl.h"

using Scalar = cx32;

/* clang-format off */
template void tools::finite::h5::save::data(h5pp::File &h5file, const StorageInfo &sinfo, const Scalar * data, const std::vector<long> & dims, std::string_view data_name, std::string_view prefix, CopyPolicy copy_policy);

template void tools::finite::h5::save::measurements(h5pp::File &, const StorageInfo &, const TensorsFinite<Scalar> &, const AlgorithmStatus &);

template void tools::finite::h5::save::measurements(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &, const ModelFinite<Scalar> &, const EdgesFinite<Scalar> &, const AlgorithmStatus &);

template void tools::finite::h5::save::correlations(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::expectations(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::opdm(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::opdm_spectrum(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::bond_dimensions(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::truncation_errors(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::entanglement_entropies(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::subsystem_entanglement_entropies(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::information_lattice(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::information_per_scale(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::information_center_of_mass(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::number_probabilities(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::renyi_entropies(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::number_entropies(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::bonds(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::state(h5pp::File &, const StorageInfo &, const StateFinite<Scalar> &);

template void tools::finite::h5::save::model(h5pp::File &, const StorageInfo &, const ModelFinite<Scalar> &);

template void tools::finite::h5::save::mpo(h5pp::File &, const StorageInfo &, const ModelFinite<Scalar> &);

template void tools::finite::h5::save::simulation(h5pp::File &h5file, const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status, CopyPolicy copy_policy);

template void tools::finite::h5::save::simulation(h5pp::File &h5file, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model, const EdgesFinite<Scalar> &edges, const AlgorithmStatus &status, CopyPolicy copy_policy);

/* clang-format on */