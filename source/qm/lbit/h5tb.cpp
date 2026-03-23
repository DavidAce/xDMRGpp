#include "../lbit.h"
#include "config/enums/LbitCircuitGateMatrixKind.h"
#include "config/enums/LbitCircuitGateWeightKind.h"
#include "debug/exceptions.h"
#include "tools/common/log.h"
#include <h5pp/details/h5ppType.h>
#include <hdf5.h>

const h5pp::hid::h5t qm::lbit::UnitaryGateParameters::get_h5_type() {
    static h5pp::hid::h5t h5_type;
    if(h5_type.valid()) return h5_type;
    h5_type                               = H5Tcreate(H5T_COMPOUND, sizeof(qm::lbit::UnitaryGateParameters));
    std::array<hsize_t, 1> array2_dims    = {2};
    std::array<hsize_t, 1> array4_dims    = {4};
    h5pp::hid::h5t         H5_ARRAY2_TYPE = H5Tarray_create(H5T_NATIVE_ULONG, array2_dims.size(), array2_dims.data());
    h5pp::hid::h5t         H5_ARRAY4_TYPE = H5Tarray_create(H5T_NATIVE_DOUBLE, array4_dims.size(), array4_dims.data());

    H5Tinsert(h5_type, "layer", HOFFSET(qm::lbit::UnitaryGateParameters, layer), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "sites", HOFFSET(qm::lbit::UnitaryGateParameters, sites), H5_ARRAY2_TYPE);
    H5Tinsert(h5_type, "f", HOFFSET(qm::lbit::UnitaryGateParameters, f), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "w", HOFFSET(qm::lbit::UnitaryGateParameters, w), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "theta", HOFFSET(qm::lbit::UnitaryGateParameters, theta), H5_ARRAY4_TYPE);
    H5Tinsert(h5_type, "l", HOFFSET(qm::lbit::UnitaryGateParameters, l), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "c", HOFFSET(qm::lbit::UnitaryGateParameters, c), h5pp::type::getH5Type<decltype(qm::lbit::UnitaryGateParameters::c)>());
    H5Tinsert(h5_type, "wkind", HOFFSET(qm::lbit::UnitaryGateParameters, wkind), get_h5t_enum_uwkind());
    H5Tinsert(h5_type, "mkind", HOFFSET(qm::lbit::UnitaryGateParameters, mkind), get_h5t_enum_umkind());
    return h5_type;
}

void qm::lbit::UnitaryGateParameters::print_parameter_names() const noexcept {
    std::string name_line;
    for(const auto &name : get_parameter_names()) name_line.append(fmt::format("{:<{}} ", name, fmt_value(name).size()));
    tools::log->info(name_line);
}

void qm::lbit::UnitaryGateParameters::print_parameter_values() const noexcept {
    std::string value_line;
    for(const auto &name : get_parameter_names()) value_line.append(fmt::format("{} ", fmt_value(name)));
    tools::log->info(value_line);
}

const h5pp::hid::h5t qm::lbit::UnitaryGateParameters::get_h5t_enum_umkind() {
    static h5pp::hid::h5t enum_umkind;
    if(enum_umkind.valid()) return enum_umkind;
    enum_umkind = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(enum_umkind, "MATRIX_V1", (val = static_cast<int>(LbitCircuitGateMatrixKind::MATRIX_V1), &val));
    H5Tenum_insert(enum_umkind, "MATRIX_V2", (val = static_cast<int>(LbitCircuitGateMatrixKind::MATRIX_V2), &val));
    H5Tenum_insert(enum_umkind, "MATRIX_V3", (val = static_cast<int>(LbitCircuitGateMatrixKind::MATRIX_V3), &val));
    return enum_umkind;
}

const h5pp::hid::h5t qm::lbit::UnitaryGateParameters::get_h5t_enum_uwkind() {
    static h5pp::hid::h5t enum_uwkind;
    if(enum_uwkind.valid()) return enum_uwkind;
    enum_uwkind = H5Tenum_create(H5T_NATIVE_INT);
    int val;
    H5Tenum_insert(enum_uwkind, "IDENTITY", (val = static_cast<int>(LbitCircuitGateWeightKind::IDENTITY), &val));
    H5Tenum_insert(enum_uwkind, "EXPDECAY", (val = static_cast<int>(LbitCircuitGateWeightKind::EXPDECAY), &val));
    return enum_uwkind;
}

std::string qm::lbit::UnitaryGateParameters::fmt_value(std::string_view p) const {
    /* clang-format off */
        if(p == "layer")  return fmt::format("{:>7}"     , layer);
        if(p == "sites")  return fmt::format("{::4}"     , sites);
        if(p == "f")      return fmt::format("{:<7.4f}"  , f);
        if(p == "w")      return fmt::format("{:<9.2e}"  , w);
        if(p == "theta")  return fmt::format("{::<+9.2e}", theta);
        if(p == "c")      return fmt::format("{}"        , c);
        if(p == "lambda") return fmt::format("{:<7.4f}"  , l);
        if(p == "wkind") return fmt::format("{}"         , enum2sv(wkind));
        if(p == "mkind") return fmt::format("{}"         , enum2sv(mkind));
    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 9> qm::lbit::UnitaryGateParameters::get_parameter_names() noexcept {
    return {"layer", "sites", "f", "w", "theta", "c", "lambda", "wkind", "mkind"};
}
