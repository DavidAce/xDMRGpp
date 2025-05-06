#include "impl.h"

using Scalar = cx32;

/* clang-format off */

template void tools::infinite::h5::save::bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<Scalar> &state);
template void tools::infinite::h5::save::state(h5pp::File &h5file, const StorageInfo &sinfo, const StateInfinite<Scalar> &state);
template void tools::infinite::h5::save::edges(h5pp::File &h5file, const StorageInfo &sinfo, const EdgesInfinite<Scalar> &state);
template void tools::infinite::h5::save::model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<Scalar> &state);
template void tools::infinite::h5::save::mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelInfinite<Scalar> &state);
template void tools::infinite::h5::save::measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsInfinite<Scalar> &tensors, const AlgorithmStatus &status);
