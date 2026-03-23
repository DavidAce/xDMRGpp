#pragma once
#include "math/float.h"
#include <h5pp/details/h5ppFstr.h>
#include <h5pp/details/h5ppHid.h>
#include <h5pp/details/h5ppVarr.h>
#include <h5pp/details/h5ppVstr.h>
#include <string>
#include <string_view>
#include <vector>

template<typename h5tb_derived_t>
class h5tb_base {
    protected:
    inline static h5pp::hid::h5t h5_type;

    public:
    virtual void                        register_table_type() const = 0;
    [[nodiscard]] const h5pp::hid::h5t &get_h5_type() const;
    [[nodiscard]] virtual std::string   fmt_value(std::string_view p) const = 0;
    void                                print_parameter_names() const noexcept;
    void                                print_parameter_values() const noexcept;
};

class h5tb_ising_selfdual : public h5tb_base<h5tb_ising_selfdual> {
    public:
    struct table {
        double           J_mean   = 0; /*!< Mean for the distrbution of J_rand */
        double           J_wdth   = 0; /*!< Width for the distrbution of J_rand */
        double           J_rand   = 0; /*!< Randomly distributed nearest neighbour coupling */
        double           h_mean   = 0; /*!< Mean for the distrbution of h_rand */
        double           h_wdth   = 0; /*!< Width for the distrbution of h_rand */
        double           h_rand   = 0; /*!< Randomly distributed on-site field */
        double           lambda   = 0; /*!< Factor involved in next-nearest neighbor interaction */
        double           delta    = 0; /*!< Difference log(J_mean) - log(h_mean) */
        long             spin_dim = 2; /*!< Spin dimension */
        h5pp::fstr_t<16> distribution; /*!< The random distribution of J_rnd and h_rnd. Choose between lognormal, normal or uniform */
    };

    table                                                           param;
    void                                                            register_table_type() const;
    [[nodiscard]] std::string                                       fmt_value(std::string_view p) const;
    [[nodiscard]] static constexpr std::array<std::string_view, 10> get_parameter_names() noexcept;
};

class h5tb_ising_majorana : public h5tb_base<h5tb_ising_majorana> {
    public:
    struct table {
        double           g      = 0; /*!< Interaction parameter for nearest ZZ and next-nearest XX neighbor coupling */
        double           delta  = 0; /*!< Delta defined as log(J_mean) - log(h_mean). We get J_mean and h_mean by fixing delta = 2lnW, W = J_wdth = 1/h_wdth */
        double           J_rand = 0; /*!< Randomly distributed nearest neighbour coupling */
        double           h_rand = 0; /*!< Randomly distributed on-site field */
        long             spin_dim = 2; /*!< Spin dimension */
        h5pp::fstr_t<16> distribution; /*!< The random distribution of J_rand and h_rand. Choose between lognormal, normal or uniform */
    };
    table                                                          param;
    void                                                           register_table_type() const;
    [[nodiscard]] std::string                                      fmt_value(std::string_view p) const;
    [[nodiscard]] static constexpr std::array<std::string_view, 6> get_parameter_names() noexcept;
};

class h5tb_xxz : public h5tb_base<h5tb_xxz> {
    public:
    struct table {
        double           delta  = 0; /*!< Delta defined as log(J_mean) - log(h_mean). We get J_mean and h_mean by fixing delta = 2lnW, W = J_wdth = 1/h_wdth */
        double           h_rand = 0; /*!< Random on-site fields */
        long             spin_dim = 2; /*!< Spin dimension */
        h5pp::fstr_t<16> distribution; /*!< The random distribution of J_rand and h_rand. Choose between lognormal, normal or uniform */
    };
    table                                                          param;
    void                                                           register_table_type() const;
    [[nodiscard]] std::string                                      fmt_value(std::string_view p) const;
    [[nodiscard]] static constexpr std::array<std::string_view, 6> get_parameter_names() noexcept;
};

class h5tb_ising_tf_rf : public h5tb_base<h5tb_ising_tf_rf> {
    public:
    struct table {
        double           J1       = 0; /*!< Nearest neighbor coupling */
        double           J2       = 0; /*!< Next-nearest neighbor coupling */
        double           h_tran   = 0; /*!< Transverse field strength */
        double           h_mean   = 0; /*!< Random field mean of distribution */
        double           h_wdth   = 0; /*!< Random field width of distribution. */
        double           h_rand   = 0; /*!< Random field value */
        long             spin_dim = 2; /*!< Spin dimension */
        h5pp::fstr_t<16> distribution; /*!< The random distribution of h_rand. Choose between lognormal, normal or uniform */
    };
    table                                                          param;
    void                                                           register_table_type() const;
    [[nodiscard]] std::string                                      fmt_value(std::string_view p) const;
    [[nodiscard]] static constexpr std::array<std::string_view, 8> get_parameter_names() noexcept;
};

class h5tb_lbit : public h5tb_base<h5tb_lbit> {
    public:
    struct table {
        using vlen_type                         = h5pp::varr_t<h5pp::fstr_t<64>>;
        h5pp::fstr_t<64>               J1_rand  = {}; /*!< On-site interaction */
        h5pp::varr_t<h5pp::fstr_t<64>> J2_rand  = {}; /*!< Two-body interaction */
        h5pp::fstr_t<64>               J3_rand  = {}; /*!< Three-body interaction */
        double                         J1_mean  = 0;  /*!< Constant offset for on-site */
        double                         J2_mean  = 0;  /*!< Constant offset for two-body interaction */
        double                         J3_mean  = 0;  /*!< Constant offset for three-body interaction */
        double                         J1_wdth  = 0;  /*!< Width of the distribution J1 */
        double                         J2_wdth  = 0;  /*!< Width of the distribution J2 */
        double                         J3_wdth  = 0;  /*!< Width of the distribution J3 */
        size_t                         J2_span  = 0;  /*!< Maximum range for pairwise interactions, |i-j| <= J2_span. */
        size_t                         J2_ctof  = 0;  /*!< Effective range for pairwise interactions, |i-j| <= std::min(J2_span, model_size-1). */
        double                         xi_Jcls  = 0;  /*!< Exp. decay rate of two-body interactions: exp(-|i-j|/xi_Jcls) * J2_rand */
        long                           spin_dim = 2;  /*!< Spin dimension */
        h5pp::fstr_t<16>               distribution;  /*!< The random number distribution. Choose between lognormal, normal or uniform */
    };

    table                                                           param;
    void                                                            register_table_type() const;
    [[nodiscard]] std::string                                       fmt_value(std::string_view p) const;
    [[nodiscard]] static constexpr std::array<std::string_view, 14> get_parameter_names() noexcept;
};
