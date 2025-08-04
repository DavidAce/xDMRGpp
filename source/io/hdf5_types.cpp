#include "hdf5_types.h"
#include "algorithms/AlgorithmStatus.h"
#include "debug/exceptions.h"
#include "io/fmt_custom.h"
#include "tid/enums.h"
#include <array>
#include <cstdio>
#include <h5pp/h5pp.h>
#include <memory>
#include <unordered_map>
#include <vector>

template class h5pp_table_measurements_finite<fp32>;
template class h5pp_table_measurements_finite<fp64>;
template class h5pp_table_measurements_finite<fp128>;
template class h5pp_table_measurements_finite<cx32>;
template class h5pp_table_measurements_finite<cx64>;
template class h5pp_table_measurements_finite<cx128>;

template class h5pp_table_measurements_infinite<fp32>;
template class h5pp_table_measurements_infinite<fp64>;
template class h5pp_table_measurements_infinite<fp128>;
template class h5pp_table_measurements_infinite<cx32>;
template class h5pp_table_measurements_infinite<cx64>;
template class h5pp_table_measurements_infinite<cx128>;

h5pp::hid::h5t &h5_enum_storage_event::get_h5t() {
    create();
    return h5_storage_event;
}

void h5_enum_storage_event::create() {
    if(h5_storage_event.valid()) return;
    //    h5_storage_event = H5Tcreate(H5T_ENUM, sizeof(StorageEvent));
    h5_storage_event = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(h5_storage_event, "NONE", (val = static_cast<int>(StorageEvent::NONE), &val));
    H5Tenum_insert(h5_storage_event, "MODEL", (val = static_cast<int>(StorageEvent::MODEL), &val));
    H5Tenum_insert(h5_storage_event, "INIT", (val = static_cast<int>(StorageEvent::INIT), &val));
    H5Tenum_insert(h5_storage_event, "EMIN", (val = static_cast<int>(StorageEvent::EMIN), &val));
    H5Tenum_insert(h5_storage_event, "EMAX", (val = static_cast<int>(StorageEvent::EMAX), &val));
    H5Tenum_insert(h5_storage_event, "PROJECTION", (val = static_cast<int>(StorageEvent::PROJECTION), &val));
    H5Tenum_insert(h5_storage_event, "BOND_UPDATE", (val = static_cast<int>(StorageEvent::BOND_UPDATE), &val));
    H5Tenum_insert(h5_storage_event, "TRNC_UPDATE", (val = static_cast<int>(StorageEvent::TRNC_UPDATE), &val));
    H5Tenum_insert(h5_storage_event, "RBDS_STEP", (val = static_cast<int>(StorageEvent::RBDS_STEP), &val));
    H5Tenum_insert(h5_storage_event, "RTES_STEP", (val = static_cast<int>(StorageEvent::RTES_STEP), &val));
    H5Tenum_insert(h5_storage_event, "ITERATION", (val = static_cast<int>(StorageEvent::ITERATION), &val));
    H5Tenum_insert(h5_storage_event, "FINISHED", (val = static_cast<int>(StorageEvent::FINISHED), &val));
}

void h5_enum_storage_event::commit(const h5pp::hid::h5f &file_id) {
    if(H5Tcommitted(get_h5t()) > 0) return;
    herr_t err = H5Tcommit(file_id, "StorageEvent", get_h5t(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(err < 0) throw except::runtime_error("Failed to commit StorageEvent to file");
}

h5pp::hid::h5t &h5_enum_opt_ritz::get_h5t() {
    create();
    return h5_opt_ritz;
}

void h5_enum_opt_ritz::create() {
    if(h5_opt_ritz.valid()) return;
    h5_opt_ritz = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(h5_opt_ritz, "NONE", (val = static_cast<int>(OptRitz::NONE), &val));
    H5Tenum_insert(h5_opt_ritz, "LR", (val = static_cast<int>(OptRitz::LR), &val));
    H5Tenum_insert(h5_opt_ritz, "SR", (val = static_cast<int>(OptRitz::SR), &val));
    H5Tenum_insert(h5_opt_ritz, "SM", (val = static_cast<int>(OptRitz::SM), &val));
    H5Tenum_insert(h5_opt_ritz, "IS", (val = static_cast<int>(OptRitz::IS), &val));
    H5Tenum_insert(h5_opt_ritz, "TE", (val = static_cast<int>(OptRitz::TE), &val));
}

void h5_enum_opt_ritz::commit(const h5pp::hid::h5f &file_id) {
    if(H5Tcommitted(get_h5t()) > 0) return;
    herr_t err = H5Tcommit(file_id, "OptRitz", get_h5t(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(err < 0) throw except::runtime_error("Failed to commit OptRitz to file");
}

h5pp::hid::h5t &h5_enum_algo_type::get_h5t() {
    create();
    return h5_algo_type;
}

void h5_enum_algo_type::create() {
    if(h5_algo_type.valid()) return;
    h5_algo_type = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(h5_algo_type, "iDMRG", (val = static_cast<int>(AlgorithmType::iDMRG), &val));
    H5Tenum_insert(h5_algo_type, "fDMRG", (val = static_cast<int>(AlgorithmType::fDMRG), &val));
    H5Tenum_insert(h5_algo_type, "xDMRG", (val = static_cast<int>(AlgorithmType::xDMRG), &val));
    H5Tenum_insert(h5_algo_type, "iTEBD", (val = static_cast<int>(AlgorithmType::iTEBD), &val));
    H5Tenum_insert(h5_algo_type, "fLBIT", (val = static_cast<int>(AlgorithmType::fLBIT), &val));
    H5Tenum_insert(h5_algo_type, "ANY", (val = static_cast<int>(AlgorithmType::ANY), &val));
}

void h5_enum_algo_type::commit(const h5pp::hid::h5f &file_id) {
    if(H5Tcommitted(get_h5t()) > 0) return;
    herr_t err = H5Tcommit(file_id, "AlgorithmType", get_h5t(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(err < 0) throw except::runtime_error("Failed to commit AlgorithmType to file");
}

h5pp::hid::h5t &h5_enum_algo_stop::get_h5t() {
    create();
    return h5_algo_stop;
}

void h5_enum_algo_stop::create() {
    if(h5_algo_stop.valid()) return;
    h5_algo_stop = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(h5_algo_stop, "SUCCESS", (val = static_cast<int>(AlgorithmStop::SUCCESS), &val));
    H5Tenum_insert(h5_algo_stop, "SATURATED", (val = static_cast<int>(AlgorithmStop::SATURATED), &val));
    H5Tenum_insert(h5_algo_stop, "MAX_ITERS", (val = static_cast<int>(AlgorithmStop::MAX_ITERS), &val));
    H5Tenum_insert(h5_algo_stop, "NONE", (val = static_cast<int>(AlgorithmStop::NONE), &val));
}

void h5_enum_algo_stop::commit(const h5pp::hid::h5f &file_id) {
    if(H5Tcommitted(get_h5t()) > 0) return;
    herr_t err = H5Tcommit(file_id, "AlgorithmStop", get_h5t(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(err < 0) throw except::runtime_error("Failed to commit AlgorithmStop to file");
}

template<typename Scalar>
h5pp::hid::h5t h5pp_table_measurements_finite<Scalar>::get_h5t() {
    register_table_type();
    return h5_type;
}
template<typename Scalar>
void h5pp_table_measurements_finite<Scalar>::register_table_type() {
    if(h5_type.valid()) return;

    auto H5T_REAL_SCALAR = h5pp::type::getH5Type<RealScalar>();
    h5_enum_storage_event::create();
    std::array<hsize_t, 1> array3_dims    = {3};
    h5pp::hid::h5t         H5_ARRAY3_TYPE = H5Tarray_create(H5T_REAL_SCALAR, array3_dims.size(), array3_dims.data());

    /* clang-format off */
        h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
        H5Tinsert(h5_type, "iter",                          HOFFSET(table, iter),                          H5T_NATIVE_UINT64);
        H5Tinsert(h5_type, "step",                          HOFFSET(table, step),                          H5T_NATIVE_UINT64);
        H5Tinsert(h5_type, "position",                      HOFFSET(table, position),                      H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "event",                         HOFFSET(table, event),                         h5_enum_storage_event::get_h5t());
        H5Tinsert(h5_type, "length",                        HOFFSET(table, length),                        H5T_NATIVE_UINT64);
        H5Tinsert(h5_type, "energy",                        HOFFSET(table, energy),                        H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "energy_variance",               HOFFSET(table, energy_variance),               H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "energy_variance_lowest",        HOFFSET(table, energy_variance_lowest),        H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "norm",                          HOFFSET(table, norm),                          H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "truncation_error",              HOFFSET(table, truncation_error),              H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "bond_mid",                      HOFFSET(table, bond_mid),                      H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "bond_lim",                      HOFFSET(table, bond_lim),                      H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "bond_max",                      HOFFSET(table, bond_max),                      H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "entanglement_entropy",          HOFFSET(table, entanglement_entropy),          H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "renyi_entropy_2",               HOFFSET(table, renyi_entropy_2),               H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "renyi_entropy_3",               HOFFSET(table, renyi_entropy_3),               H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "renyi_entropy_4",               HOFFSET(table, renyi_entropy_4),               H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "renyi_entropy_inf",             HOFFSET(table, renyi_entropy_inf),             H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "number_entropy",                HOFFSET(table, number_entropy),                H5T_REAL_SCALAR);
        H5Tinsert(h5_type, "spin_global",                   HOFFSET(table, spin_global),                   H5_ARRAY3_TYPE);
        H5Tinsert(h5_type, "spin_local",                    HOFFSET(table, spin_local),                    H5_ARRAY3_TYPE);
        H5Tinsert(h5_type, "structure_factors",             HOFFSET(table, structure_factors),             H5_ARRAY3_TYPE);
        H5Tinsert(h5_type, "total_time",                    HOFFSET(table, total_time),                    H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "algorithm_time",                HOFFSET(table, algorithm_time),                H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "physical_time",                 HOFFSET(table, physical_time),                 decltype(table::physical_time)::get_h5type());
    /* clang-format on */
}

template<typename Scalar>
h5pp::hid::h5t h5pp_table_measurements_infinite<Scalar>::get_h5t() {
    register_table_type();
    return h5_type;
}
template<typename Scalar>
void h5pp_table_measurements_infinite<Scalar>::register_table_type() {
    if(h5_type.valid()) return;
    h5_enum_storage_event::create();
    auto H5T_REAL_SCALAR = h5pp::type::getH5Type<RealScalar>();

    /* clang-format off */
    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "iter",                         HOFFSET(table, iter),                         H5T_NATIVE_UINT64);
    H5Tinsert(h5_type, "step",                         HOFFSET(table, step),                         H5T_NATIVE_UINT64);
    H5Tinsert(h5_type, "position",                     HOFFSET(table, position),                     H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "event",                        HOFFSET(table, event),                        h5_enum_storage_event::get_h5t());
    H5Tinsert(h5_type, "length",                       HOFFSET(table, length),                       H5T_NATIVE_UINT64);
    H5Tinsert(h5_type, "bond_dim",                     HOFFSET(table, bond_dim),                     H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "bond_lim",                     HOFFSET(table, bond_lim),                     H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "bond_max",                     HOFFSET(table, bond_max),                     H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "entanglement_entropy",         HOFFSET(table, entanglement_entropy),         H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "norm",                         HOFFSET(table, norm),                         H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_mpo",                   HOFFSET(table, energy_mpo),                   H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_per_site_mpo",          HOFFSET(table, energy_per_site_mpo),          H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_per_site_ham",          HOFFSET(table, energy_per_site_ham),          H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_per_site_mom",          HOFFSET(table, energy_per_site_mom),          H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_variance_mpo",          HOFFSET(table, energy_variance_mpo),          H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_variance_per_site_mpo", HOFFSET(table, energy_variance_per_site_mpo), H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_variance_per_site_ham", HOFFSET(table, energy_variance_per_site_ham), H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "energy_variance_per_site_mom", HOFFSET(table, energy_variance_per_site_mom), H5T_REAL_SCALAR);
    H5Tinsert(h5_type, "truncation_error",             HOFFSET(table, truncation_error),             H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "wall_time",                    HOFFSET(table, wall_time),                    H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "phys_time",                    HOFFSET(table, phys_time),                    decltype(table::phys_time)::get_h5type());
    H5Tinsert(h5_type, "time_step",                    HOFFSET(table, time_step),                    decltype(table::time_step)::get_h5type());
    /* clang-format on */
}

h5pp::hid::h5t h5pp_table_algorithm_status::get_h5t() {
    register_table_type();
    return h5_type;
}

void h5pp_table_algorithm_status::register_table_type() {
    if(h5_type.valid()) return;
    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));

    /* clang-format off */
    H5Tinsert(h5_type, "iter",                        HOFFSET(table, iter),                       H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "step",                        HOFFSET(table, step),                       H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "position",                    HOFFSET(table, position),                   H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "direction",                   HOFFSET(table, direction),                  H5T_NATIVE_INT);
    H5Tinsert(h5_type, "event",                       HOFFSET(table, event),                      h5_enum_storage_event::get_h5t());
    H5Tinsert(h5_type, "opt_ritz",                    HOFFSET(table, opt_ritz),                   h5_enum_opt_ritz::get_h5t());
    H5Tinsert(h5_type, "algo_type",                   HOFFSET(table, algo_type),                  h5_enum_algo_type::get_h5t());
    H5Tinsert(h5_type, "algo_stop",                   HOFFSET(table, algo_stop),                  h5_enum_algo_stop::get_h5t());
    H5Tinsert(h5_type, "min_iters",                   HOFFSET(table, min_iters),                  H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "bond_lim",                    HOFFSET(table, bond_lim),                   H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "bond_max",                    HOFFSET(table, bond_max),                   H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "bond_min",                    HOFFSET(table, bond_min),                   H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "trnc_lim",                    HOFFSET(table, trnc_lim),                   H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "trnc_min",                    HOFFSET(table, trnc_min),                   H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "trnc_max",                    HOFFSET(table, trnc_max),                   H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_min",                  HOFFSET(table, energy_min),                 H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_max",                  HOFFSET(table, energy_max),                 H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_tgt",                  HOFFSET(table, energy_tgt),                 H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_dens",                 HOFFSET(table, energy_dens),                H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_dens_target",          HOFFSET(table, energy_dens_target),         H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_variance_lowest",      HOFFSET(table, energy_variance_lowest),     H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_variance_max_digits",  HOFFSET(table, energy_variance_max_digits), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "energy_variance_prec_limit",  HOFFSET(table, energy_variance_prec_limit), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "mixing_factor",               HOFFSET(table, mixing_factor),              H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "phys_time",                   HOFFSET(table, phys_time),                  decltype(table::phys_time)::get_h5type());
    H5Tinsert(h5_type, "wall_time",                   HOFFSET(table, wall_time),                  H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "algo_time",                   HOFFSET(table, algo_time),                  H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "delta_t",                     HOFFSET(table, delta_t),                    decltype(table::delta_t)::get_h5type());
    H5Tinsert(h5_type, "algorithm_has_finished",      HOFFSET(table, algorithm_has_finished),     H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "algorithm_has_succeeded",     HOFFSET(table, algorithm_has_succeeded),    H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "algorithm_has_to_stop",       HOFFSET(table, algorithm_has_to_stop),      H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "algorithm_has_stuck_for",     HOFFSET(table, algorithm_has_stuck_for),    H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "algorithm_saturated_for",     HOFFSET(table, algorithm_saturated_for),    H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "algorithm_converged_for",     HOFFSET(table, algorithm_converged_for),    H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "entanglement_converged_for",  HOFFSET(table, entanglement_converged_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "entanglement_saturated_for",  HOFFSET(table, entanglement_saturated_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "locinfoscale_saturated_for",  HOFFSET(table, locinfoscale_saturated_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_mpo_converged_for",  HOFFSET(table, variance_mpo_converged_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_mpo_saturated_for",  HOFFSET(table, variance_mpo_saturated_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_ham_converged_for",  HOFFSET(table, variance_ham_converged_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_ham_saturated_for",  HOFFSET(table, variance_ham_saturated_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_mom_converged_for",  HOFFSET(table, variance_mom_converged_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "variance_mom_saturated_for",  HOFFSET(table, variance_mom_saturated_for), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "bond_limit_has_reached_max",  HOFFSET(table, bond_limit_has_reached_max), H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "trnc_limit_has_reached_min",  HOFFSET(table, trnc_limit_has_reached_min), H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "spin_parity_has_converged",   HOFFSET(table, spin_parity_has_converged),  H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "trnc_error_has_converged",    HOFFSET(table, trnc_error_has_converged),   H5T_NATIVE_UINT8);
    H5Tinsert(h5_type, "time_step_has_converged",     HOFFSET(table, time_step_has_converged),    H5T_NATIVE_UINT8);
    /* clang-format on */
}

h5pp::hid::h5t h5pp_table_memory_usage::get_h5t() {
    register_table_type();
    return h5_type;
}

void h5pp_table_memory_usage::register_table_type() {
    if(h5_type.valid()) return;
    h5_enum_storage_event::create();
    /* clang-format off */
        h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
        H5Tinsert(h5_type, "iter",     HOFFSET(table, iter),    H5T_NATIVE_UINT64);
        H5Tinsert(h5_type, "step",     HOFFSET(table, step),    H5T_NATIVE_UINT64);
        H5Tinsert(h5_type, "position", HOFFSET(table, position),H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "event",    HOFFSET(table, event),   h5_enum_storage_event::get_h5t());
        H5Tinsert(h5_type, "bond_lim", HOFFSET(table, bond_lim),H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "rss",      HOFFSET(table, rss),     H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "hwm",      HOFFSET(table, hwm),     H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "vm",       HOFFSET(table, vm),      H5T_NATIVE_DOUBLE);
    /* clang-format on */
}

h5pp_ur::item::item(const item &it) : name(it.name), time(it.time), sum(it.sum), pcnt(it.pcnt), avg(it.avg), level(it.level), count(it.count) {}
h5pp_ur::item::item(std::string_view name_, double time, double sum, double pcnt, double avg, int level, size_t count)
    : name(name_), time(time), sum(sum), pcnt(pcnt), avg(avg), level(level), count(count) {}

h5pp::hid::h5t h5pp_ur::get_h5t() {
    register_table_type();
    return h5_type;
}
void h5pp_ur::register_table_type() {
    if(h5_type.valid()) return;
    if(not h5_level_type.valid()) {
        h5_level_type = H5Tcreate(H5T_ENUM, sizeof(tid::level));
        int val;
        H5Tenum_insert(h5_level_type, "disabled", (val = tid::level::disabled, &val));
        H5Tenum_insert(h5_level_type, "parent", (val = tid::level::parent, &val));
        H5Tenum_insert(h5_level_type, "normal", (val = tid::level::normal, &val));
        H5Tenum_insert(h5_level_type, "higher", (val = tid::level::higher, &val));
        H5Tenum_insert(h5_level_type, "highest", (val = tid::level::highest, &val));
    }

    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(item));

    // Create a type for the char array from the template H5T_C_S1
    // The template describes a string with a single char.
    // Set the size with H5Tset_size, or h5pp::hdf5::setStringSize(...)
    h5pp::hid::h5t h5t_custom_string = H5Tcopy(H5T_C_S1);
    H5Tset_size(h5t_custom_string, H5T_VARIABLE);

    // Optionally set the null terminator or pad with '\0'
    H5Tset_strpad(h5t_custom_string, H5T_STR_NULLTERM);
    H5Tset_cset(h5t_custom_string, H5T_CSET_UTF8);

    H5Tinsert(h5_type, "name", HOFFSET(item, name), h5t_custom_string);
    H5Tinsert(h5_type, "time", HOFFSET(item, time), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "sum", HOFFSET(item, sum), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "pcnt", HOFFSET(item, pcnt), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "avg", HOFFSET(item, avg), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "level", HOFFSET(item, level), h5_level_type);
    H5Tinsert(h5_type, "count", HOFFSET(item, count), H5T_NATIVE_UINT64);
}