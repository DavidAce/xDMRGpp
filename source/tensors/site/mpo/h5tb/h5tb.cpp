#include "h5tb.h"
#include "debug/exceptions.h"
#include "math/cast.h"
#include "math/float.h"
#include "tools/common/log.h"
#include <hdf5.h>

template<typename h5tb_derived_t>
const h5pp::hid::h5t &h5tb_base<h5tb_derived_t>::get_h5_type() const {
    register_table_type();
    return h5_type;
}

template<typename h5tb_derived_t>
void h5tb_base<h5tb_derived_t>::print_parameter_names() const noexcept {
    std::string name_line;
    for(const auto &name : h5tb_derived_t::get_parameter_names()) name_line.append(fmt::format("{:<{}} ", name, fmt_value(name).size()));
    tools::log->info(name_line);
}

template<typename h5tb_derived_t>
void h5tb_base<h5tb_derived_t>::print_parameter_values() const noexcept {
    std::string value_line;
    for(const auto &name : h5tb_derived_t::get_parameter_names()) value_line.append(fmt::format("{} ", fmt_value(name)));
    tools::log->info(value_line);
}

template class h5tb_base<h5tb_ising_tf_rf>;
template class h5tb_base<h5tb_ising_majorana>;
template class h5tb_base<h5tb_ising_selfdual>;
template class h5tb_base<h5tb_lbit>;
template class h5tb_base<h5tb_xxz>;

/*
 *
 *
 *  ISING SELFDUAL
 *
 *
 */

void h5tb_ising_selfdual::register_table_type() const {
    if(h5_type.valid()) return;

    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "J_mean", HOFFSET(table, J_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J_wdth", HOFFSET(table, J_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J_rand", HOFFSET(table, J_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_mean", HOFFSET(table, h_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_wdth", HOFFSET(table, h_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_rand", HOFFSET(table, h_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "lambda", HOFFSET(table, lambda), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "delta", HOFFSET(table, delta), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "spin_dim", HOFFSET(table, spin_dim), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "distribution", HOFFSET(table, distribution), decltype(table::distribution)::get_h5type());
}

std::string h5tb_ising_selfdual::fmt_value(std::string_view p) const {
    /* clang-format off */
        if(p == "J_mean")           return fmt::format("{:<+9.2e}", param.J_mean);
        if(p == "J_wdth")           return fmt::format("{:<9.2e}" , param.J_wdth);
        if(p == "J_rand")           return fmt::format("{:<+9.2e}", param.J_rand);
        if(p == "h_mean")           return fmt::format("{:<+9.2e}", param.h_mean);
        if(p == "h_wdth")           return fmt::format("{:<9.2e}", param.h_wdth);
        if(p == "h_rand")           return fmt::format("{:<+9.2e}", param.h_rand);
        if(p == "lambda")           return fmt::format("{:<7.4f}", param.lambda);
        if(p == "delta")            return fmt::format("{:<+7.4f}", param.delta);
        if(p == "spin_dim")         return fmt::format("{:>8}", param.spin_dim);
        if(p == "distribution")     return fmt::format("{:<12}", static_cast<std::string_view>(param.distribution));
    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 10> h5tb_ising_selfdual::get_parameter_names() noexcept {
    return {"J_mean", "J_wdth", "J_rand", "h_mean", "h_wdth", "h_rand", "lambda", "delta", "spin_dim", "distribution"};
}

/*
 *
 *
 *  ISING MAJORANA
 *
 *
 */

void h5tb_ising_majorana::register_table_type() const {
    if(h5_type.valid()) return;

    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "g", HOFFSET(table, g), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "delta", HOFFSET(table, delta), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J_rand", HOFFSET(table, J_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_rand", HOFFSET(table, h_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "spin_dim", HOFFSET(table, spin_dim), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "distribution", HOFFSET(table, distribution), decltype(table::distribution)::get_h5type());
}

std::string h5tb_ising_majorana::fmt_value(std::string_view p) const {
    /* clang-format off */
        if(p == "g")                return fmt::format("{:<7.4f}" , param.g);
        if(p == "delta")            return fmt::format("{:<+7.4f}", param.delta);
        if(p == "J_rand")           return fmt::format("{:<8.2e}" , param.J_rand);
        if(p == "h_rand")           return fmt::format("{:<8.2e}" , param.h_rand);
        if(p == "spin_dim")         return fmt::format("{:>8}"    , param.spin_dim);
        if(p == "distribution")     return fmt::format("{:<12}"   , static_cast<std::string_view>(param.distribution));
    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 6> h5tb_ising_majorana::get_parameter_names() noexcept {
    return {"g", "delta", "J_rand", "h_rand", "spin_dim", "distribution"};
}

/*
 *
 *
 *  XXZ
 *
 *
 */

void h5tb_xxz::register_table_type() const {
    if(h5_type.valid()) return;

    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "delta", HOFFSET(table, delta), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_rand", HOFFSET(table, h_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "spin_dim", HOFFSET(table, spin_dim), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "distribution", HOFFSET(table, distribution), decltype(table::distribution)::get_h5type());
}

std::string h5tb_xxz::fmt_value(std::string_view p) const {
    /* clang-format off */
    if(p == "delta")            return fmt::format("{:<+7.4f}", param.delta);
    if(p == "h_rand")           return fmt::format("{:<8.2e}" , param.h_rand);
    if(p == "spin_dim")         return fmt::format("{:>8}"    , param.spin_dim);
    if(p == "distribution")     return fmt::format("{:<12}", static_cast<std::string_view>(param.distribution));

    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 6> h5tb_xxz::get_parameter_names() noexcept { return {"delta", "h_rand", "spin_dim", "distribution"}; }

/*
 *
 *
 *  ISING TRANSVERSE RANDOM FIELD
 *
 *
 */

void h5tb_ising_tf_rf::register_table_type() const {
    if(h5_type.valid()) return;

    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "J1", HOFFSET(table, J1), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J2", HOFFSET(table, J2), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_tran", HOFFSET(table, h_tran), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_mean", HOFFSET(table, h_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_wdth", HOFFSET(table, h_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "h_rand", HOFFSET(table, h_rand), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "spin_dim", HOFFSET(table, spin_dim), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "distribution", HOFFSET(table, distribution), decltype(table::distribution)::get_h5type());
}

std::string h5tb_ising_tf_rf::fmt_value(std::string_view p) const {
    /* clang-format off */
        if(p == "J1")               return fmt::format("{:<+9.2e}", param.J1);
        if(p == "J2")               return fmt::format("{:<+9.2e}", param.J2);
        if(p == "h_tran")           return fmt::format("{:<+9.2e}", param.h_tran);
        if(p == "h_mean")           return fmt::format("{:<+9.2e}", param.h_mean);
        if(p == "h_wdth")           return fmt::format("{:<+9.2e}", param.h_wdth);
        if(p == "h_rand")           return fmt::format("{:<+9.2e}", param.h_rand);
        if(p == "spin_dim")         return fmt::format("{:>8}"    , param.spin_dim);
        if(p == "distribution")     return fmt::format("{:<12}", static_cast<std::string_view>(param.distribution));

    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 8> h5tb_ising_tf_rf::get_parameter_names() noexcept {
    return {"J1", "J2", "h_tran", "h_mean", "h_wdth", "h_rand", "spin_dim", "distribution"};
}

/*
 *
 *
 *  LBIT
 *
 *
 */

void h5tb_lbit::register_table_type() const {
    if(h5_type.valid()) return;
    h5_type                       = H5Tcreate(H5T_COMPOUND, sizeof(table));
    h5pp::hid::h5t h5_varr_vstr_t = h5pp::varr_t<h5pp::vstr_t>::get_h5type();

    H5Tinsert(h5_type, "J1_rand", HOFFSET(table, J1_rand), decltype(table::J1_rand)::get_h5type());
    H5Tinsert(h5_type, "J2_rand", HOFFSET(table, J2_rand), decltype(table::J2_rand)::get_h5type());
    H5Tinsert(h5_type, "J3_rand", HOFFSET(table, J3_rand), decltype(table::J3_rand)::get_h5type());
    H5Tinsert(h5_type, "J1_mean", HOFFSET(table, J1_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J2_mean", HOFFSET(table, J2_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J3_mean", HOFFSET(table, J3_mean), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J1_wdth", HOFFSET(table, J1_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J2_wdth", HOFFSET(table, J2_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J3_wdth", HOFFSET(table, J3_wdth), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "J2_span", HOFFSET(table, J2_span), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "J2_ctof", HOFFSET(table, J2_ctof), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "xi_Jcls", HOFFSET(table, xi_Jcls), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "spin_dim", HOFFSET(table, spin_dim), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "distribution", HOFFSET(table, distribution), decltype(table::distribution)::get_h5type());
}

std::string h5tb_lbit::fmt_value(std::string_view p) const {
    // J2_rand is special since it varies in length for each mpo. Let's just pad with nan to make it pretty
    if(p == "J2_rand") {
        std::vector<fp64> J2_rand;
        for(size_t i = 0; i < param.J2_ctof + 1; ++i) {
            if(i < param.J2_rand.size())
                J2_rand.emplace_back(param.J2_rand[i].to_floating_point<fp64>());
            else
                J2_rand.emplace_back(std::numeric_limits<fp64>::quiet_NaN());
        }
        return fmt::format("{::<+9.2e}", J2_rand);
    }

    /* clang-format off */
        if(p == "J1_rand")     return fmt::format("{:<+9.2e}", param.J1_rand.to_floating_point<fp64>());
        if(p == "J3_rand")     return fmt::format("{:<+9.2e}", param.J3_rand.to_floating_point<fp64>());
        if(p == "J1_mean")     return fmt::format("{:<+9.2e}", param.J1_mean);
        if(p == "J2_mean")     return fmt::format("{:<+9.2e}", param.J2_mean);
        if(p == "J3_mean")     return fmt::format("{:<+9.2e}", param.J3_mean);
        if(p == "J1_wdth")     return fmt::format("{:<7.4f}",  param.J1_wdth);
        if(p == "J2_wdth")     return fmt::format("{:<7.4f}",  param.J2_wdth);
        if(p == "J3_wdth")     return fmt::format("{:<7.4f}",  param.J3_wdth);
        if(p == "xi_Jcls")     return fmt::format("{:<7.4f}",  param.xi_Jcls);
        if(p == "J2_span")     return fmt::format("{:>7}",     param.J2_span == -1ul ? -1l : safe_cast<long>(param.J2_span));
        if(p == "J2_ctof")     return fmt::format("{:>7}",     param.J2_ctof);
        if(p == "spin_dim")    return fmt::format("{:>8}",     param.spin_dim);
        if(p == "distribution")return fmt::format("{:<12}", static_cast<std::string_view>(param.distribution));
    /* clang-format on */
    throw except::runtime_error("Unrecognized parameter: {}", p);
}

constexpr std::array<std::string_view, 14> h5tb_lbit::get_parameter_names() noexcept {
    return {"J1_rand", "J2_rand", "J3_rand", "J1_mean", "J2_mean", "J3_mean",  "J1_wdth",
            "J2_wdth", "J3_wdth", "J2_span", "J2_ctof", "xi_Jcls", "spin_dim", "distribution"};
}
