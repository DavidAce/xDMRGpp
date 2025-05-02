#pragma once
#include "config/settings.h"
#include "debug/exceptions.h"
#include "io/hdf5_types.h"
#include "math/float.h"
#include "math/num.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/log.h"
#include "tools/finite/env.h"
#include "tools/finite/h5.h"
#include "tools/finite/measure/hamiltonian.h"
#include "tools/finite/print.h"
#include <algorithms/AlgorithmStatus.h>
#include <complex>
#include <h5pp/h5pp.h>
#include <typeindex>

// Load model, state and simulation status from HDF5
template<typename Scalar>
void tools::finite::h5::load::simulation(const h5pp::File &h5file, std::string_view state_prefix, TensorsFinite<Scalar> &tensors, AlgorithmStatus &status,
                                         AlgorithmType algo_type) {
    try {
        if(algo_type == AlgorithmType::fLBIT) {
            // To successfully load a simulation there has to be a clearly defined initial state, either a pattern or an initial state selection
            tensors = TensorsFinite<Scalar>(algo_type, settings::model::model_type, settings::model::model_size, 0);
            tools::common::h5::load::initial_state_attrs(h5file, state_prefix, settings::strategy::initial_pattern);
            tensors.initialize_state(ResetReason::INIT, settings::strategy::initial_state, settings::strategy::initial_type, settings::strategy::initial_axis,
                                     settings::strategy::use_eigenspinors, status.bond_lim, settings::strategy::initial_pattern);
            tools::common::h5::load::status(h5file, state_prefix, status);
            tools::finite::h5::load::model(h5file, algo_type, *tensors.model);
            tools::common::h5::load::timer(h5file, state_prefix, status);
            tools::finite::h5::load::validate(h5file, state_prefix, tensors, status, algo_type);
        } else {
            // To successfully load a simulation there has to be an MPS
            tensors = TensorsFinite<Scalar>(algo_type, settings::model::model_type, settings::model::model_size, 0);
            MpsInfo info;
            tools::finite::h5::load::state(h5file, state_prefix, *tensors.state, info);
            tools::common::h5::load::status(h5file, state_prefix, status, info);
            tools::finite::h5::load::model(h5file, algo_type, *tensors.model);
            tools::common::h5::load::timer(h5file, state_prefix, status);
            tools::finite::h5::load::validate(h5file, state_prefix, tensors, status, algo_type);
        }
    } catch(const std::exception &ex) { throw except::load_error("error loading from state prefix [{}]: {}", state_prefix, ex.what()); }
}

template<typename Scalar>
void tools::finite::h5::load::state(const h5pp::File &h5file, std::string_view state_prefix, StateFinite<Scalar> &state, MpsInfo &info) {
    using cx64 = std::complex<double>;
    try {
        // To successfully load a state there has to be an MPS with StorageLevel::FULL
        auto mps_info = tools::common::h5::resume::find_fully_stored_mps(h5file, state_prefix);
        if(mps_info.empty()) throw except::load_error("No mps with StorageLevel::FULL are stored under [{}]", state_prefix);
        info            = mps_info.back();
        auto model_size = h5file.readAttribute<size_t>(info.pfx, "model_size");
        auto algo_type  = h5file.readAttribute<AlgorithmType>(info.pfx, "algo_type");

        state.mps_sites.clear();
        state.set_algorithm(algo_type);
        for(long pos = 0; pos < safe_cast<long>(model_size); ++pos) {
            auto dset_M_name = fmt::format("{}/M_{}", info.pfx, pos);
            auto position    = h5file.readAttribute<long>(dset_M_name, "position");
            assert(position == pos);
            auto typeInfo = h5file.getTypeInfoDataset(dset_M_name);

            if(typeInfo.cppTypeIndex.has_value() and typeInfo.cppTypeIndex.value() == typeid(double)) {
                auto M     = h5file.readDataset<Eigen::Tensor<double, 3>>(dset_M_name);
                auto L     = h5file.readAttribute<Eigen::Tensor<double, 1>>(dset_M_name, "L");
                auto error = h5file.readAttribute<double>(dset_M_name, "truncation_error");
                auto label = h5file.readAttribute<std::string>(dset_M_name, "label");
                state.mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(M, L, position, error, label));
            } else {
                auto M     = h5file.readDataset<Eigen::Tensor<cx64, 3>>(dset_M_name);
                auto L     = h5file.readAttribute<Eigen::Tensor<double, 1>>(dset_M_name, "L");
                auto error = h5file.readAttribute<double>(dset_M_name, "truncation_error");
                auto label = h5file.readAttribute<std::string>(dset_M_name, "label");
                state.mps_sites.emplace_back(std::make_unique<MpsSite<Scalar>>(M, L, position, error, label));
            }

            auto isCenter = h5file.readAttribute<bool>(dset_M_name, "isCenter");
            if(isCenter) {
                auto LC       = h5file.readAttribute<Eigen::Tensor<double, 1>>(dset_M_name, "LC");
                auto error_LC = h5file.readAttribute<double>(dset_M_name, "truncation_error_LC");
                state.mps_sites.back()->set_LC(LC, error_LC);
            }
            tools::log->trace("Loaded mps: {}({})", state.mps_sites.back()->get_label(), state.mps_sites.back()->get_position());
        }
        state.set_positions();
    } catch(const std::exception &ex) { throw except::load_error("load state error: {}", ex.what()); }
}
template<typename Scalar>
void tools::finite::h5::load::model(const h5pp::File &h5file, AlgorithmType algo_type, ModelFinite<Scalar> &model) {
    auto model_path = fmt::format("{}/model", enum2sv(algo_type));
    try {
        // Find the path to the model
        auto table_path = fmt::format("{}/hamiltonian", model_path);
        tools::log->info("Loading hamiltonian from table: [{}]", table_path);
        auto model_type = h5file.readAttribute<ModelType>(table_path, "model_type");
        auto model_size = h5file.readAttribute<size_t>(table_path, "model_size");
        if(model_type != settings::model::model_type)
            throw std::runtime_error(
                fmt::format("model_type [{}] != settings::model::model_type [{}]", enum2sv(model_type), enum2sv(settings::model::model_type)));
        if(model_size != settings::model::model_size)
            throw except::runtime_error("model_size [{}] != settings::model::model_size [{}]", model_size, settings::model::model_size);
        for(const auto &mpo : model.MPO) mpo->load_hamiltonian(h5file, model_path);
        for(const auto &mpo : model.MPO) tools::log->trace("Loaded mpo: {}({})", enum2sv(mpo->model_type), mpo->get_position());
        tools::log->debug("Finished loading model");
        tools::finite::print::model(model);
    } catch(const std::exception &ex) { throw except::load_error("failed to load model [{}]: {}", model_path, ex.what()); }
}

template<typename T1, typename T2, typename T3>
requires std::is_floating_point_v<T1> and std::is_floating_point_v<T2>
void compare(T1 val1, T2 val2, T3 tol, std::string_view tag) {
    if(val2 == 0) {
        tools::log->warn("Value mismatch after resume: {} = {:.16f} | expected {:.16f} | diff = {:.16f} | tol = {:.16f}", tag, fp(val1), fp(val2),
                         fp(std::abs(val1 - val2)), fp(tol));
        tools::log->warn("Value 2 is zero: It may not have been initialized in the previous simulation");
        return;
    }
    if(num::gt(std::abs(val1 - val2), tol))
        throw except::runtime_error("Value mismatch after resume: {} = {:.16f} | expected {:.16f} | diff = {:.16f} | tol = {:.16f}", tag, fp(val1), fp(val2),
                                    fp(std::abs(val1 - val2)), fp(tol));
    else
        tools::log->debug("{} matches measurement on file: {:.16f} == {:.16f} | diff = {:.16f} | tol = {:.16f}", tag, fp(val1), fp(val2),
                          fp(std::abs(val1 - val2)), fp(tol));
}
template<typename Scalar>
void tools::finite::h5::load::validate(const h5pp::File &h5file, std::string_view state_prefix, TensorsFinite<Scalar> &tensors, AlgorithmStatus &status,
                                       AlgorithmType algo_type) {
    auto t_val = tid::tic_scope("validate");
    tools::log->info("Validating state: [{}]", state_prefix);
    tensors.rebuild_mpo();
    tensors.rebuild_mpo_squared();
    tensors.activate_sites({tensors.template get_position<size_t>()});
    tensors.rebuild_edges();
    tools::log->debug("State labels: {}", tensors.state->get_labels());
    tensors.state->clear_cache();
    tensors.state->clear_measurements();

    if(algo_type == AlgorithmType::fLBIT) {
        // Check that the time limits are the same
        tools::log->info("Validating time series: [{}]", state_prefix);
        auto time_scale          = h5file.readAttribute<std::optional<std::string>>(state_prefix, "time_scale");
        auto time_start          = h5file.readAttribute<std::optional<cx64>>(state_prefix, "time_start");
        auto time_final          = h5file.readAttribute<std::optional<cx64>>(state_prefix, "time_final");
        auto time_num_steps      = h5file.readAttribute<std::optional<uint64_t>>(state_prefix, "time_num_steps");
        auto expected_time_start = cx64(settings::flbit::time_start_real, settings::flbit::time_start_imag);
        auto expected_time_final = cx64(settings::flbit::time_final_real, settings::flbit::time_final_imag);
        if(time_scale.has_value() and time_scale.value() != enum2sv(settings::flbit::time_scale)) {
            throw except::load_error("Mismatching time scale: file {} != config {}", time_scale, enum2sv(settings::flbit::time_scale));
        }
        if(time_start.has_value() and time_start->real() != expected_time_start.real()) {
            throw except::load_error("Mismatching time start real: file {} != config {}", time_start->real(), expected_time_start.real());
        }
        if(time_start.has_value() and time_start->imag() != expected_time_start.imag()) {
            throw except::load_error("Mismatching time start imag: file {} != config {}", time_start->imag(), expected_time_start.imag());
        }
        if(time_final.has_value() and time_final->real() != expected_time_final.real()) {
            throw except::load_error("Mismatching time final real: file {} != config {}", time_final->real(), expected_time_final.real());
        }
        if(time_final.has_value() and time_final->imag() != expected_time_final.imag()) {
            throw except::load_error("Mismatching time final imag: file {} != config {}", time_final->imag(), expected_time_final.imag());
        }
        if(time_num_steps.has_value() and time_num_steps.value() != settings::flbit::time_num_steps) {
            throw except::load_error("Mismatching time num steps: file {} != config {}", time_num_steps.value(), settings::flbit::time_num_steps);
        }
        if(status.algo_time != std::clamp<double>(status.algo_time, std::abs(expected_time_start), std::abs(expected_time_final))) {
            throw except::load_error("Current status.algo_time == [{}] is outside the expected time interval [{} -> {}]", status.algo_time, expected_time_start,
                                     expected_time_final);
        }
    }

    if(algo_type != AlgorithmType::fLBIT) {
        auto measurements_path = fmt::format("{}/measurements", state_prefix);
        auto expected_measurements =
            h5file.readTableRecords<std::optional<typename h5pp_table_measurements_finite<Scalar>::table>>(measurements_path, h5pp::TableSelection::LAST);
        if(expected_measurements and expected_measurements->iter == status.iter) {
            // In this case we have loaded state_real from file.
            // In the fLBIT case, the MPO's belong to state_lbit, so measuring the energy on state_real w.r.t the lbit-hamiltonian makes no sense.
            tools::log->debug("Validating resumed state energy (without energy reduction): [{}]", state_prefix);
            tensors.clear_measurements();
            compare(tools::finite::measure::energy(tensors), expected_measurements->energy, 1e-5, "Energy");
            compare(tools::finite::measure::energy_variance(tensors), expected_measurements->energy_variance, 1e-5, "Energy variance");

            if(settings::precision::use_energy_shifted_mpo) {
                auto energy_shift = tools::finite::measure::energy(tensors);
                tensors.set_energy_shift_mpo(energy_shift);
                tensors.rebuild_mpo();
                tensors.rebuild_mpo_squared();
                tensors.compress_mpo_squared();
                tensors.rebuild_edges();
                tools::log->debug("Validating resumed state (after energy reduction): [{}]", state_prefix);
                tensors.clear_measurements();
                compare(tools::finite::measure::energy(tensors), expected_measurements->energy, 1e-5, "Energy");
                compare(tools::finite::measure::energy_variance(tensors), expected_measurements->energy_variance, 1e-5, "Energy variance");
            }
        }
    }
}
