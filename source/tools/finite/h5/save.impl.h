#pragma once

#include "algorithms/AlgorithmStatus.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "io/hdf5_types.h"
#include "math/float.h"
#include "math/num.h"
#include "qm/spin.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/dimensions.h"
#include "tools/finite/measure/entanglement_entropy.h"
#include "tools/finite/measure/expectation_value.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/measure/information.h"
#include "tools/finite/measure/norm.h"
#include "tools/finite/measure/number_entropy.h"
#include "tools/finite/measure/opdm.h"
#include "tools/finite/measure/spin.h"
#include "tools/finite/measure/truncation.h"
#include "tools/finite/ops.h"
#include <complex>
#include <h5pp/h5pp.h>

using tools::common::h5::save::should_save;

template<typename T>
void tools::finite::h5::save::data(h5pp::File &h5file, const StorageInfo &sinfo, const T *data, const std::vector<long> &dims, std::string_view data_name,
                                   std::string_view prefix, CopyPolicy copy_policy) {
    auto t_data    = tid::tic_scope("data");
    auto data_path = fmt::format("{}/{}", prefix, data_name);
    // Check if the current entry has already been saved
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, data_path);
    if(attrs == sinfo) return;
    auto         size   = Eigen::internal::array_prod(dims);
    H5D_layout_t layout = size > 10 ? H5D_CHUNKED : H5D_CONTIGUOUS;
    tools::log->trace("Writing to dataset: {} | event {}", data_path, enum2sv(sinfo.storage_event));
    h5file.writeDataset(data, data_path, dims, layout);
    tools::common::h5::save::set_save_attrs(h5file, data_path, sinfo);
    tools::common::h5::tmp::copy_from_tmp(h5file, sinfo.iter, sinfo.step, sinfo.storage_event, copy_policy);
}

template<typename T>
void save_data(h5pp::File &h5file, const StorageInfo &sinfo, const T &data, std::string_view data_name, std::string_view prefix) {
    auto t_data    = tid::tic_scope("data");
    auto data_path = fmt::format("{}/{}", prefix, data_name);
    // Check if the current entry has already been saved
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, data_path);
    if(attrs == sinfo) return;
    H5D_layout_t layout;
    if(std::is_scalar_v<T>)
        layout = H5D_layout_t::H5D_COMPACT;
    else
        layout = H5D_layout_t::H5D_CHUNKED;
    tools::log->trace("Writing to dataset: {} | event {}", data_path, enum2sv(sinfo.storage_event));
    h5file.writeDataset(data, data_path, layout);
    tools::common::h5::save::set_save_attrs(h5file, data_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::measurements(h5pp::File &h5file, const StorageInfo &sinfo, const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status) {
    tools::finite::h5::save::measurements(h5file, sinfo, tensors.get_state(), tensors.get_model(), tensors.get_edges(), status);
}

template<typename Scalar>
void tools::finite::h5::save::measurements(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                           const EdgesFinite<Scalar> &edges, const AlgorithmStatus &status) {
    if(not should_save(sinfo, settings::storage::table::measurements::policy)) return;
    auto t_hdf = tid::tic_scope("measurements", tid::level::highest);
    tools::log->trace("Saving measurements with type {} ({} bits)", sfinae::type_name<Scalar>(), sizeof(Scalar));
    sinfo.assert_well_defined();
    // Define the table
    std::string table_path = fmt::format("{}/measurements", sinfo.get_state_prefix());

    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, table_path);
    if(not attrs.link_exists) h5file.createTable(h5pp_table_measurements_finite<Scalar>::get_h5t(), table_path, "measurements");
    if(attrs == sinfo) return;
    auto offset = tools::common::h5::save::get_table_offset(h5file, table_path, sinfo, attrs);

    // Define the table entry
    typename h5pp_table_measurements_finite<Scalar>::table measurement_entry{};
    measurement_entry.iter     = safe_cast<uint64_t>(sinfo.iter);
    measurement_entry.step     = safe_cast<uint64_t>(sinfo.step);
    measurement_entry.position = safe_cast<long>(sinfo.position);
    measurement_entry.event    = sinfo.storage_event;
    measurement_entry.length   = state.template get_length<uint64_t>();
    if(status.algo_type != AlgorithmType::fLBIT) {
        measurement_entry.energy                 = tools::finite::measure::energy(state, model, edges);
        measurement_entry.energy_variance        = tools::finite::measure::energy_variance(state, model, edges);
        measurement_entry.energy_variance_lowest = static_cast<RealScalar<Scalar>>(status.energy_variance_lowest);
    }
    measurement_entry.norm             = tools::finite::measure::norm(state);
    measurement_entry.truncation_error = state.get_truncation_error_midchain();
    measurement_entry.bond_lim         = sinfo.bond_lim;
    measurement_entry.bond_max         = sinfo.bond_max;
    measurement_entry.bond_mid         = safe_cast<long>(tools::finite::measure::bond_dimension_midchain(state));

    measurement_entry.entanglement_entropy = tools::finite::measure::entanglement_entropy_midchain(state);
    measurement_entry.renyi_entropy_2      = tools::finite::measure::renyi_entropy_midchain(state, 2);
    measurement_entry.renyi_entropy_3      = tools::finite::measure::renyi_entropy_midchain(state, 3);
    measurement_entry.renyi_entropy_4      = tools::finite::measure::renyi_entropy_midchain(state, 4);
    measurement_entry.renyi_entropy_inf    = tools::finite::measure::renyi_entropy_midchain(state, std::numeric_limits<double>::infinity());
    if(status.algo_type == AlgorithmType::fLBIT) { measurement_entry.number_entropy = tools::finite::measure::number_entropy_midchain(state); }
    measurement_entry.spin_global = tools::finite::measure::spin_components(state);
    measurement_entry.spin_local  = tools::finite::measure::spin_expectation_value_xyz(state);
    if(status.algo_type == AlgorithmType::xDMRG and sinfo.storage_event == StorageEvent::FINISHED) {
        measurement_entry.structure_factors = tools::finite::measure::spin_structure_factor_xyz(state);
    }
    measurement_entry.total_time     = status.wall_time;
    measurement_entry.algorithm_time = status.algo_time;
    measurement_entry.physical_time  = status.phys_time;

    tools::log->trace("Writing to table: {} | event {} | offset {} | policy {}", table_path, enum2sv(sinfo.storage_event), offset,
                      flag2str(settings::storage::table::measurements::policy));
    h5file.writeTableRecords(measurement_entry, table_path, offset);
    tools::common::h5::save::set_save_attrs(h5file, table_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::correlations(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::dataset::correlation_matrix_spin_xyz::policy)) return;
    auto correlation_matrix_xyz = tools::finite::measure::spin_correlation_matrix_xyz(state);
    /* clang-format off */
        save_data(h5file, sinfo, correlation_matrix_xyz[0], "correlation_matrix_sx", sinfo.get_state_prefix());
        save_data(h5file, sinfo, correlation_matrix_xyz[1], "correlation_matrix_sy", sinfo.get_state_prefix());
        save_data(h5file, sinfo, correlation_matrix_xyz[2], "correlation_matrix_sz", sinfo.get_state_prefix());
    /* clang-format on */
}

template<typename Scalar>
void tools::finite::h5::save::expectations(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(sinfo.algo_type == AlgorithmType::xDMRG) {
        if(not should_save(sinfo, settings::storage::table::expectation_values_spin_xyz::policy)) return;
        auto t_hdf = tid::tic_scope("spin_xyz", tid::level::higher);
        tools::log->trace("Saving spin expectation values to {}", sinfo.get_state_prefix());
        auto expectation_values_xyz = tools::finite::measure::spin_expectation_values_xyz(state);
        tools::finite::h5::save::data_as_table(h5file, sinfo, expectation_values_xyz[0], "expectation_values_sx", "<sigma x>", "L_");
        tools::finite::h5::save::data_as_table(h5file, sinfo, expectation_values_xyz[1], "expectation_values_sy", "<sigma y>", "L_");
        tools::finite::h5::save::data_as_table(h5file, sinfo, expectation_values_xyz[2], "expectation_values_sz", "<sigma z>", "L_");
    }
    if(sinfo.algo_type == AlgorithmType::fLBIT) {
        if(not should_save(sinfo, settings::storage::dataset::expectation_values_spin_xyz::policy)) return;
        auto t_hdf = tid::tic_scope("spin_local", tid::level::higher);
        //            tools::log->trace("Saving spin expectation values to {}", sinfo.get_state_prefix());
        if(not state.measurements.expectation_values_sz.has_value())
            state.measurements.expectation_values_sz = measure::expectation_values<Scalar>(state, qm::spin::half::sz).real();
        //            tools::finite::h5::save::data_as_table(h5file, sinfo, state.measurements.expectation_values_sz, "expectation_values_sz", "<sigma z>",
        //            "L_");
        auto dset_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "expectation_values_sz");
        // Check if the current entry has already been appended
        auto attrs = tools::common::h5::save::get_save_attrs(h5file, dset_path);
        if(attrs == sinfo) return;
        if(not attrs.link_exists) {
            auto                 rows = safe_cast<hsize_t>(state.measurements.expectation_values_sz->dimension(0));
            std::vector<hsize_t> dims = {rows, 0};
            std::vector<hsize_t> chnk = {rows, settings::storage::dataset::expectation_values_spin_xyz::chunksize};
            using Real = typename MeasurementsStateFinite<Scalar>::RealScalar;
            h5file.createDataset(dset_path, h5pp::type::getH5Type<Real>(), H5D_CHUNKED, dims, chnk);
        }
        tools::log->trace("Writing to dataset: {} | event {} | policy {}", dset_path, enum2sv(sinfo.storage_event),
                          flag2str(settings::storage::dataset::expectation_values_spin_xyz::policy));
        h5file.appendToDataset(state.measurements.expectation_values_sz.value(), dset_path, 1);
        tools::common::h5::save::set_save_attrs(h5file, dset_path, sinfo);
    }
}

template<typename Scalar>
void tools::finite::h5::save::opdm(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::dataset::opdm::policy)) return;
    auto dset_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "opdm");

    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, dset_path);
    if(attrs == sinfo) return;
    const auto &opdm = tools::finite::measure::opdm(state, sinfo.model_type);
    if(opdm.size() == 0) return;
    if(not attrs.link_exists) {
        auto                 rows = safe_cast<hsize_t>(opdm.dimension(0));
        auto                 cols = safe_cast<hsize_t>(opdm.dimension(1));
        std::vector<hsize_t> dims = {rows, cols, 0};
        std::vector<hsize_t> chnk = {rows, cols, settings::storage::dataset::opdm::chunksize};
        using opdm_type           = typename std::remove_cvref_t<decltype(opdm)>::Scalar;
        h5file.createDataset(dset_path, h5pp::type::getH5Type<opdm_type>(), H5D_CHUNKED, dims, chnk, std::nullopt, 2);
        h5file.writeAttribute("extent-1,offset,iter", dset_path, "index");
        h5file.writeAttribute("One-particle density matrix", dset_path, "description");
    }
    tools::log->trace("Writing to dataset: {} | event {} | policy {}", dset_path, enum2sv(sinfo.storage_event),
                      flag2str(sinfo.get_dataset_storage_policy(dset_path)));
    h5file.appendToDataset(opdm, dset_path, 2);
    tools::common::h5::save::set_save_attrs(h5file, dset_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::opdm_spectrum(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::opdm_spectrum::policy)) return;
    auto opdm_spectrum = tools::finite::measure::opdm_spectrum(state, sinfo.model_type);
    tools::finite::h5::save::data_as_table(h5file, sinfo, opdm_spectrum, "opdm_spectrum", "One-particle Density Matrix spectrum", "eigval");
}

template<typename T>
void tools::finite::h5::save::data_as_table(h5pp::File &h5file, const StorageInfo &sinfo, const T *const data, size_t size, std::string_view table_name,
                                            std::string_view table_title, std::string_view fieldname) {
    auto t_hdf = tid::tic_scope(table_name, tid::level::highest);
    sinfo.assert_well_defined();

    // Define the table
    auto table_path = fmt::format("{}/{}", sinfo.get_state_prefix(), table_name);
    auto h5_type    = h5pp_table_data<T>::register_table_type(h5pp::type::getH5Type<T>(), size, fieldname);

    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, table_path);
    if(not attrs.link_exists) h5file.createTable(h5_type, table_path, table_title);
    if(attrs == sinfo) return;
    auto offset = tools::common::h5::save::get_table_offset(h5file, table_path, sinfo, attrs);

    tools::log->trace("Writing to table: {} | event {} | offset {} | policy {}", table_path, enum2sv(sinfo.storage_event), offset,
                      flag2str(sinfo.get_table_storage_policy(table_name)));
    // Copy the data into an std::vector<std::byte> stream, which will act as a struct for our table entry
    auto entry = h5pp_table_data<T>::make_entry(sinfo.iter, sinfo.step, sinfo.position, sinfo.storage_event, sinfo.bond_lim, data, size);
    h5file.writeTableRecords(entry, table_path, offset);
    tools::common::h5::save::set_save_attrs(h5file, table_path, sinfo);
}

template<typename T>
void tools::finite::h5::save::data_as_table_vla(h5pp::File &h5file, const StorageInfo &sinfo, const std::vector<T> &data, const h5pp::hid::h5t &h5elem_t,
                                                std::string_view table_name, std::string_view table_title, std::string_view fieldname) {
    auto t_hdf = tid::tic_scope(table_name);
    sinfo.assert_well_defined();

    // Define the table
    auto table_path = fmt::format("{}/{}", sinfo.get_state_prefix(), table_name);
    auto h5_type    = h5pp_table_data<T>::register_table_type(h5elem_t, data.size(), fieldname);

    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, table_path);
    if(not attrs.link_exists) h5file.createTable(h5_type, table_path, table_title);
    if(attrs == sinfo) return;
    auto offset = tools::common::h5::save::get_table_offset(h5file, table_path, sinfo, attrs);

    // Copy the data into an std::vector<std::byte> stream, which will act as a struct for our table entry
    tools::log->trace("Writing to table: {} | event {} | offset {} | policy {}", table_path, enum2sv(sinfo.storage_event), offset,
                      flag2str(sinfo.get_table_storage_policy(table_name)));
    auto entry = h5pp_table_data<T>::make_entry(sinfo.iter, sinfo.step, sinfo.position, sinfo.storage_event, sinfo.bond_lim, data.data(), data.size());
    h5file.writeTableRecords(entry, table_path, offset);
    tools::common::h5::save::set_save_attrs(h5file, table_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::bond_dimensions(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::bond_dimensions::policy)) return;
    auto t_hdf = tid::tic_scope("bond_dimensions", tid::level::higher);
    data_as_table(h5file, sinfo, tools::finite::measure::bond_dimensions(state), "bond_dimensions", "Bond Dimensions", "L_");
}

template<typename Scalar>
void tools::finite::h5::save::truncation_errors(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::truncation_errors::policy)) return;
    auto t_hdf = tid::tic_scope("truncation_errors", tid::level::higher);
    data_as_table(h5file, sinfo, tools::finite::measure::truncation_errors(state), "truncation_errors", "Truncation errors", "L_");
}

template<typename Scalar>
void tools::finite::h5::save::entanglement_entropies(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::entanglement_entropies::policy)) return;
    auto t_hdf = tid::tic_scope("entropies", tid::level::higher);
    data_as_table(h5file, sinfo, tools::finite::measure::entanglement_entropies(state), "entanglement_entropies", "Entanglement Entropies", "L_");
}

template<typename Scalar>
void tools::finite::h5::save::subsystem_entanglement_entropies(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::dataset::subsystem_entanglement_entropies::policy)) return;
    auto dset_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "subsystem_entanglement_entropies");
    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, dset_path);
    if(attrs == sinfo) return;
    if(not state.measurements.subsystem_entanglement_entropies)
        state.measurements.subsystem_entanglement_entropies = tools::finite::measure::subsystem_entanglement_entropies_log2(state);
    auto t_hdf = tid::tic_scope("subsystem_entanglement_entropies", tid::level::higher);

    auto rows = safe_cast<hsize_t>(state.measurements.subsystem_entanglement_entropies->rows());
    auto cols = safe_cast<hsize_t>(state.measurements.subsystem_entanglement_entropies->cols());
    if(not attrs.link_exists) {
        std::vector<hsize_t> dims = {rows, cols, 0};
        std::vector<hsize_t> chnk = {rows, cols, settings::storage::dataset::subsystem_entanglement_entropies::chunksize};
        using Real = typename MeasurementsStateFinite<Scalar>::RealScalar;
        h5file.createDataset(dset_path, h5pp::type::getH5Type<Real>(), H5D_CHUNKED, dims, chnk, std::nullopt, 2);
        h5file.writeAttribute("extent-1,offset,iter", dset_path, "index");
        h5file.writeAttribute("Entanglement entropies for subsystems", dset_path, "description");
    }

    auto offset = tools::common::h5::save::get_table_offset(h5file, dset_path, sinfo, attrs);
    auto dims   = h5file.getDatasetDimensions(dset_path);
    if(offset < dims[2]) { // Found an existing record
        tools::log->trace("Writing to dataset: {} | offset {} | event {} | policy {}", dset_path, offset, enum2sv(sinfo.storage_event),
                          flag2str(sinfo.get_dataset_storage_policy(dset_path)));
        auto opts = h5pp::Options();
        auto slab = h5pp::Hyperslab({0, 0, offset}, {rows, cols, 1l});
        h5file.writeHyperslab(state.measurements.subsystem_entanglement_entropies.value(), dset_path, slab);
    } else {
        tools::log->trace("Writing to dataset: {} | event {} | policy {}", dset_path, offset, enum2sv(sinfo.storage_event),
                          flag2str(sinfo.get_dataset_storage_policy(dset_path)));
        h5file.appendToDataset(state.measurements.subsystem_entanglement_entropies.value(), dset_path, 2);
    }
    tools::common::h5::save::set_save_attrs(h5file, dset_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::information_lattice(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::dataset::information_lattice::policy)) return;
    auto dset_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "information_lattice");
    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, dset_path);
    if(attrs == sinfo) return;
    if(not state.measurements.information_lattice) state.measurements.information_lattice = tools::finite::measure::information_lattice(state);
    auto t_hdf = tid::tic_scope("information_lattice", tid::level::higher);
    auto rows  = safe_cast<hsize_t>(state.measurements.information_lattice->rows());
    auto cols  = safe_cast<hsize_t>(state.measurements.information_lattice->cols());
    if(not attrs.link_exists) {
        std::vector<hsize_t> dims = {rows, cols, 0};
        std::vector<hsize_t> chnk = {rows, cols, settings::storage::dataset::information_lattice::chunksize};
        using Real = typename MeasurementsStateFinite<Scalar>::RealScalar;
        h5file.createDataset(dset_path, h5pp::type::getH5Type<Real>(), H5D_CHUNKED, dims, chnk, std::nullopt, 2);
        h5file.writeAttribute("extent-1,offset,iter", dset_path, "index");
        h5file.writeAttribute("Information lattice", dset_path, "description");
    }
    auto offset = tools::common::h5::save::get_table_offset(h5file, dset_path, sinfo, attrs);
    auto dims   = h5file.getDatasetDimensions(dset_path);
    if(offset < dims[2]) { // Found an existing record
        tools::log->trace("Writing to dataset: {} | offset {} | event {} | policy {}", dset_path, offset, enum2sv(sinfo.storage_event),
                          flag2str(sinfo.get_dataset_storage_policy(dset_path)));
        auto opts = h5pp::Options();
        auto slab = h5pp::Hyperslab({0, 0, offset}, {rows, cols, 1l});
        h5file.writeHyperslab(state.measurements.information_lattice.value(), dset_path, slab);
    } else {
        tools::log->trace("Writing to dataset: {} | event {} | policy {}", dset_path, enum2sv(sinfo.storage_event),
                          flag2str(sinfo.get_dataset_storage_policy(dset_path)));
        h5file.appendToDataset(state.measurements.information_lattice.value(), dset_path, 2);
    }
    tools::common::h5::save::set_save_attrs(h5file, dset_path, sinfo);
}

template<typename Scalar>
void tools::finite::h5::save::information_per_scale(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::information_per_scale::policy)) return;
    auto table_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "information_per_scale");
    auto attrs      = tools::common::h5::save::get_save_attrs(h5file, table_path);
    if(attrs == sinfo) return;
    auto information_per_scale = tools::finite::measure::information_per_scale(state);
    tools::finite::h5::save::data_as_table(h5file, sinfo, information_per_scale, "information_per_scale", "Information per length scale", "scale");
    auto information_center_of_mass = tools::finite::measure::information_center_of_mass(state);
    h5file.writeAttribute(information_center_of_mass, table_path, "information_center_of_mass");
}

template<typename Scalar>
void tools::finite::h5::save::information_center_of_mass(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::information_center_of_mass::policy)) return;
    auto table_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "information_center_of_mass");
    auto attrs      = tools::common::h5::save::get_save_attrs(h5file, table_path);
    if(attrs == sinfo) return;
    auto information_center_of_mass = tools::finite::measure::information_center_of_mass(state);
    tools::log->info("information_center_of_mass: {:.16f} | t = {:.3e}", fp(information_center_of_mass),
                     state.measurements.see_time.has_value() ? state.measurements.see_time.value() : std::numeric_limits<double>::quiet_NaN());
    tools::finite::h5::save::data_as_table(h5file, sinfo, information_center_of_mass, "information_center_of_mass", "Information center of mass", "scale");
    if(state.measurements.see_time.has_value()) h5file.writeAttribute(state.measurements.see_time.value(), table_path, "time_see");
}

template<typename Scalar>
void tools::finite::h5::save::number_probabilities(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::dataset::number_probabilities::policy)) return;
    if(state.get_algorithm() != AlgorithmType::fLBIT) {
        tools::log->warn("Called tools::finite::h5::save::number_probabilities from algorithm [{}]: This is currently only valid with the [fLBIT] algorithm",
                         enum2sv(state.get_algorithm()));
        return;
    }
    if(not state.measurements.number_probabilities)
        throw except::logic_error("Number probabilities have not been computed yet.\n"
                                  "Make sure to compute the number entropies before saving the number probabilities");
    auto t_hdf     = tid::tic_scope("number_probabilities", tid::level::higher);
    auto dset_path = fmt::format("{}/{}", sinfo.get_state_prefix(), "number_probabilities");
    // Check if the current entry has already been appended
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, dset_path);
    if(attrs == sinfo) return;
    if(not attrs.link_exists) {
        auto                 rows = static_cast<hsize_t>(state.measurements.number_probabilities->dimension(0));
        auto                 cols = static_cast<hsize_t>(state.measurements.number_probabilities->dimension(1));
        std::vector<hsize_t> dims = {rows, cols, 0};
        std::vector<hsize_t> chnk = {rows, cols, settings::storage::dataset::number_probabilities::chunksize};
        using Real = typename MeasurementsStateFinite<Scalar>::RealScalar;
        h5file.createDataset(dset_path, h5pp::type::getH5Type<Real>(), H5D_CHUNKED, dims, chnk, std::nullopt, 6);
        h5file.writeAttribute("n_count, site, time", dset_path, "index");
        h5file.writeAttribute("Probability of finding n_count particles to the left of a site at a time index", dset_path, "description");
    }
    tools::log->trace("Writing to dataset: {} | event {} | policy {}", dset_path, enum2sv(sinfo.storage_event),
                      flag2str(sinfo.get_dataset_storage_policy(dset_path)));
    h5file.appendToDataset(state.measurements.number_probabilities.value(), dset_path, 2);
    tools::common::h5::save::set_save_attrs(h5file, dset_path, sinfo);
    // Do not append if the iteration number is smaller than the dataset iter dimension
    // auto num_entries = h5file.getDatasetDimensions(dset_path).back();
    // if(sinfo.iter >= num_entries * settings::storage::storage_interval) {
    // }
}

template<typename Scalar>
void tools::finite::h5::save::renyi_entropies(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::renyi_entropies::policy)) return;
    auto t_hdf = tid::tic_scope("entropies", tid::level::higher);
    auto inf   = std::numeric_limits<double>::infinity();
    data_as_table(h5file, sinfo, tools::finite::measure::renyi_entropies(state, 2), "renyi_entropies_2", "Midchain Renyi Entropy 2", "L_");
    data_as_table(h5file, sinfo, tools::finite::measure::renyi_entropies(state, 3), "renyi_entropies_3", "Midchain Renyi Entropy 3", "L_");
    data_as_table(h5file, sinfo, tools::finite::measure::renyi_entropies(state, 4), "renyi_entropies_4", "Midchain Renyi Entropy 4", "L_");
    data_as_table(h5file, sinfo, tools::finite::measure::renyi_entropies(state, inf), "renyi_entropies_inf", "Midchain Renyi Entropy inf", "L_");
}

template<typename Scalar>
void tools::finite::h5::save::number_entropies(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::number_entropies::policy)) return;
    if(state.get_algorithm() != AlgorithmType::fLBIT) {
        tools::log->warn("Called tools::finite::h5::save::number_entropies from algorithm [{}]: This is currently only valid with the [fLBIT] algorithm",
                         enum2sv(state.get_algorithm()));
        return;
    }
    auto t_hdf = tid::tic_scope("entropies", tid::level::higher);
    data_as_table(h5file, sinfo, tools::finite::measure::number_entropies(state), "number_entropies", "Number entropies", "L_");
}

template<typename Scalar>
void tools::finite::h5::save::bonds(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, settings::storage::table::bonds::policy)) return;
    auto t_hdf = tid::tic_scope("bonds", tid::level::higher);
    // Transform from cx64 to real to save space
    auto        h5real      = h5pp::type::getH5Type<RealScalar<Scalar>>();
    auto        bonds_real  = std::vector<h5pp::varr_t<RealScalar<Scalar>>>();
    std::string column_name = "L_";
    std::string table_title = "Bonds (singular values)";
    bonds_real.reserve(state.template get_length<size_t>() + 1);
    for(const auto &mps : state.mps_sites) {
        bonds_real.emplace_back(Eigen::Tensor<RealScalar<Scalar>, 1>(mps->get_L().real()));
        if(mps->isCenter()) { bonds_real.emplace_back(Eigen::Tensor<RealScalar<Scalar>, 1>(mps->get_LC().real())); }
    }
    data_as_table_vla(h5file, sinfo, bonds_real, h5real, "bonds", table_title, column_name);
}

template<typename Scalar>
void tools::finite::h5::save::state(h5pp::File &h5file, const StorageInfo &sinfo, const StateFinite<Scalar> &state) {
    if(not should_save(sinfo, sinfo.get_state_storage_policy())) return;
    auto t_hdf      = tid::tic_scope("state", tid::level::higher);
    auto mps_prefix = sinfo.get_mps_prefix();

    // Check if the current entry has already been saved
    auto attrs = tools::common::h5::save::get_save_attrs(h5file, mps_prefix);
    if(attrs == sinfo) return;
    auto mpsIsReal = state.is_real();
    tools::log->trace("Writing mps: {} | event {} | policy {}", mps_prefix, enum2sv(sinfo.storage_event), flag2str(sinfo.get_state_storage_policy()));
    for(const auto &mps : state.mps_sites) {
        auto dsetName = fmt::format("{}/M_{}", mps_prefix, mps->template get_position<long>());
        // Delete preexisting arrays. We can keep scalars since they are just overwritten
        h5file.deleteAttribute(dsetName, "L");
        h5file.deleteAttribute(dsetName, "LC");
        h5file.deleteAttribute(dsetName, "truncation_error_LC");

        if(mpsIsReal) {
            h5file.writeDataset(Eigen::Tensor<RealScalar<Scalar>, 3>(mps->get_M_bare().real()), dsetName,
                                H5D_CHUNKED); // Important to write bare matrices!!
        } else {
            h5file.writeDataset(mps->get_M_bare(), dsetName, H5D_CHUNKED); // Important to write bare matrices!!
        }

        // TODO: what happens if L is shorter than an existing L? Does it overwrite?
        h5file.writeAttribute(Eigen::Tensor<RealScalar<Scalar>, 1>(mps->get_L().real()), dsetName, "L");
        h5file.writeAttribute(mps->template get_position<long>(), dsetName, "position");
        h5file.writeAttribute(mps->get_M_bare().dimensions(), dsetName, "dimensions");
        h5file.writeAttribute(mps->get_label(), dsetName, "label");
        h5file.writeAttribute(mps->get_truncation_error(), dsetName, "truncation_error");
        h5file.writeAttribute(mps->get_unique_id(), dsetName, "unique_id");
        h5file.writeAttribute(mps->isCenter(), dsetName, "isCenter");
        if(mps->isCenter()) {
            h5file.writeAttribute(Eigen::Tensor<RealScalar<Scalar>, 1>(mps->get_LC().real()), dsetName, "LC");
            h5file.writeAttribute(mps->get_truncation_error_LC(), dsetName, "truncation_error_LC");
        }
    }
    h5file.writeAttribute(state.template get_length<size_t>(), mps_prefix, "model_size");
    h5file.writeAttribute(state.template get_position<long>(), mps_prefix, "position");
    h5file.writeAttribute(sinfo.algo_type, mps_prefix, "algo_type", std::nullopt, h5_enum_algo_type::get_h5t());
    h5file.writeAttribute(mps_prefix, sinfo.get_state_prefix(), "mps_is_saved");
    tools::common::h5::save::set_save_attrs(h5file, mps_prefix, sinfo);
}

/*! Write down the Hamiltonian model type and site info as attributes */
template<typename Scalar>
void tools::finite::h5::save::model(h5pp::File &h5file, const StorageInfo &sinfo, const ModelFinite<Scalar> &model) {
    if(not should_save(sinfo, settings::storage::table::model::policy)) return;
    std::string table_path = fmt::format("{}/model/hamiltonian", sinfo.algo_name);
    if(h5file.linkExists(table_path)) return tools::log->debug("The hamiltonian has already been written to [{}]", table_path);

    tools::log->trace("Writing table: {} | event {} | policy {}", table_path, enum2sv(sinfo.storage_event), flag2str(settings::storage::table::model::policy));
    auto t_hdf = tid::tic_scope("model", tid::level::higher);
    for(auto site = 0ul; site < model.get_length(); site++) model.get_mpo(site).save_hamiltonian(h5file, table_path);
    h5file.writeAttribute(settings::model::model_size, table_path, "model_size");
    h5file.writeAttribute(settings::model::model_type, table_path, "model_type");
    h5file.writeAttribute(enum2sv(settings::model::model_type), table_path, "model_name");
    h5file.writeAttribute(table_path, sinfo.get_state_prefix(), "model_is_saved");
}

/*! Write all the MPO's with site info in attributes */
template<typename Scalar>
void tools::finite::h5::save::mpo(h5pp::File &h5file, const StorageInfo &sinfo, const ModelFinite<Scalar> &model) {
    if(not should_save(sinfo, settings::storage::mpo::model::policy)) return;
    std::string mpo_prefix = fmt::format("{}/model/mpo", sinfo.get_state_prefix());
    // We do not expect the MPO's to change. Therefore, if they exist, there is nothing else to do here
    if(h5file.linkExists(mpo_prefix)) return tools::log->trace("The MPO's have already been written to [{}]", mpo_prefix);
    auto t_hdf = tid::tic_scope("mpo", tid::level::higher);
    tools::log->trace("Writing mpos: {} | event {} | policy {}", mpo_prefix, enum2sv(sinfo.storage_event), flag2str(sinfo.get_mpo_storage_policy(mpo_prefix)));
    for(size_t pos = 0; pos < model.get_length(); pos++) { model.get_mpo(pos).save_mpo(h5file, mpo_prefix); }
    h5file.writeAttribute(settings::model::model_size, mpo_prefix, "model_size");
    h5file.writeAttribute(enum2sv(settings::model::model_type), mpo_prefix, "model_type");
}

template<typename Scalar>
void tools::finite::h5::save::simulation(h5pp::File &h5file, const TensorsFinite<Scalar> &tensors, const AlgorithmStatus &status, CopyPolicy copy_policy) {
    tools::finite::h5::save::simulation(h5file, tensors.get_state(), tensors.get_model(), tensors.get_edges(), status, copy_policy);
}

template<typename Scalar>
void tools::finite::h5::save::simulation(h5pp::File &h5file, const StateFinite<Scalar> &state, const ModelFinite<Scalar> &model,
                                         const EdgesFinite<Scalar> &edges, const AlgorithmStatus &status, CopyPolicy copy_policy) {
    if(not state.position_is_inward_edge()) return;
    auto sinfo   = StorageInfo(status, state.get_name(), model.model_type);
    auto t_h5    = tid::tic_scope("h5");
    auto t_event = tid::tic_scope(enum2sv(status.event), tid::level::highest);

    tools::log->debug("Writing to file: Event [{}] | state prefix [{}]", enum2sv(sinfo.storage_event), sinfo.get_state_prefix());

    // The file can be kept open during writes
    h5file.setKeepFileOpened();

    // The main results have now been written. Next we append data to tables
    tools::common::h5::save::initial_state_attrs(h5file, sinfo); // Save the initial state type and pattern (rather than the MPS itself)
    tools::finite::h5::save::model(h5file, sinfo, model);
    tools::finite::h5::save::state(h5file, sinfo, state);
    tools::common::h5::save::status(h5file, sinfo, status);
    tools::common::h5::save::memory(h5file, sinfo);
    tools::common::h5::save::timer(h5file, sinfo);
    tools::finite::h5::save::measurements(h5file, sinfo, state, model, edges, status);
    tools::finite::h5::save::bond_dimensions(h5file, sinfo, state);
    tools::finite::h5::save::bonds(h5file, sinfo, state);
    tools::finite::h5::save::truncation_errors(h5file, sinfo, state);
    tools::finite::h5::save::entanglement_entropies(h5file, sinfo, state);
    tools::finite::h5::save::subsystem_entanglement_entropies(h5file, sinfo, state);
    tools::finite::h5::save::information_lattice(h5file, sinfo, state);
    tools::finite::h5::save::information_per_scale(h5file, sinfo, state);
    tools::finite::h5::save::information_center_of_mass(h5file, sinfo, state);
    tools::finite::h5::save::renyi_entropies(h5file, sinfo, state);
    tools::finite::h5::save::number_entropies(h5file, sinfo, state);
    tools::finite::h5::save::number_probabilities(h5file, sinfo, state);
    tools::finite::h5::save::expectations(h5file, sinfo, state);
    tools::finite::h5::save::correlations(h5file, sinfo, state);
    tools::finite::h5::save::opdm(h5file, sinfo, state);
    tools::finite::h5::save::opdm_spectrum(h5file, sinfo, state);
    tools::common::h5::save::resume_attrs(h5file, sinfo); // Save attributes relevant for resuming

    // The file can now be closed
    h5file.setKeepFileClosed();

    // Copy from temporary location to destination depending on given policy
    tools::common::h5::tmp::copy_from_tmp(h5file, sinfo.iter, sinfo.step, sinfo.storage_event, copy_policy);
}
