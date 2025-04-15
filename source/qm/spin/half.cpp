#include "../spin.h"
#include "debug/exceptions.h"
#include "math/float.h"
#include <array>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
namespace qm::spin::half {

    namespace matrix {
        const Eigen::MatrixXcd                sx(get_sx<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                sy(get_sy<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                sz(get_sz<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                sp(get_sp<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                sm(get_sm<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                nu(get_nu<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                nd(get_nd<Eigen::MatrixXcd>());
        const Eigen::MatrixXcd                id(get_id<Eigen::MatrixXcd>());
        const std::array<Eigen::VectorXcd, 2> sx_spinors(get_sx_spinors<Eigen::VectorXcd>());
        const std::array<Eigen::VectorXcd, 2> sy_spinors(get_sy_spinors<Eigen::VectorXcd>());
        const std::array<Eigen::VectorXcd, 2> sz_spinors(get_sz_spinors<Eigen::VectorXcd>());
        Eigen::VectorXcd                      get_spinor(std::string_view axis, int sign) {
            auto axus = get_axis_unsigned(axis);
            if(axus == "x" and sign >= 0) return sx_spinors[0];
            if(axus == "x" and sign < 0) return sx_spinors[1];
            if(axus == "y" and sign >= 0) return sy_spinors[0];
            if(axus == "y" and sign < 0) return sy_spinors[1];
            if(axus == "z" and sign >= 0) return sz_spinors[0]; // Spin up is |0> = (1,0)
            if(axus == "z" and sign < 0) return sz_spinors[1];  // Spin down is |1> = (0,1)
            throw except::runtime_error("get_spinor given invalid axis: {}", axis);
        }
        Eigen::VectorXcd get_spinor(std::string_view axis) { return get_spinor(axis, get_sign(axis)); }

    }
    namespace tensor {
        const Eigen::Tensor<cx64, 2> sx(get_sx<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> sy(get_sy<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> sz(get_sz<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> sp(get_sp<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> sm(get_sm<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> nu(get_nu<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> nd(get_nd<Eigen::Tensor<cx64, 2>>());
        const Eigen::Tensor<cx64, 2> id(get_id<Eigen::Tensor<cx64, 2>>());

        const std::array<Eigen::Tensor<cx64, 1>, 2> sx_spinors(get_sx_spinors<Eigen::Tensor<cx64, 1>>());
        const std::array<Eigen::Tensor<cx64, 1>, 2> sy_spinors(get_sy_spinors<Eigen::Tensor<cx64, 1>>());
        const std::array<Eigen::Tensor<cx64, 1>, 2> sz_spinors(get_sz_spinors<Eigen::Tensor<cx64, 1>>());
        Eigen::Tensor<cx64, 1>                      get_spinor(std::string_view axis, int sign) {
            auto axus = get_axis_unsigned(axis);
            if(axus == "x" and sign >= 0) return sx_spinors[0];
            if(axus == "x" and sign < 0) return sx_spinors[1];
            if(axus == "y" and sign >= 0) return sy_spinors[0];
            if(axus == "y" and sign < 0) return sy_spinors[1];
            if(axus == "z" and sign >= 0) return sz_spinors[0]; // Spin up is |0> = (1,0)
            if(axus == "z" and sign < 0) return sz_spinors[1];  // Spin down is |1> = (0,1)
            throw except::runtime_error("get_spinor given invalid axis: {}", axis);
        }
        Eigen::Tensor<cx64, 1> get_spinor(std::string_view axis) { return get_spinor(axis, get_sign(axis)); }

        Eigen::Tensor<cx64, 2> get_pauli(std::string_view axis) {
            auto axus = get_axis_unsigned(axis);
            if(axus == "x") return sx;
            if(axus == "y") return sy;
            if(axus == "z") return sz;
            if(axus == "i") return id;
            throw except::runtime_error("get_pauli: could not match axis string: {}", axis);
        }

    }

    const Eigen::Matrix2cd sx(get_sx<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd sy(get_sy<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd sz(get_sz<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd sp(get_sp<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd sm(get_sm<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd nu(get_nu<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd nd(get_nd<Eigen::Matrix2cd>());
    const Eigen::Matrix2cd id(get_id<Eigen::Matrix2cd>());

    // // auto sx2 = Eigen::Matrix2cd({{0.0 + 0.0i, 1.0 + 0.0i},   //
    // // {1.0 + 0.0i, 0.0 + 0.0i}}); //
    // /* clang-format off */
    // Eigen::Matrix2cd sx = (Eigen::Matrix2cd() <<
    //     0.0, 1.0,
    //     1.0, 0.0).finished();
    // Eigen::Matrix2cd sy = (Eigen::Matrix2cd() <<
    //     0.0+0.0i, 0.0-1.0i,
    //     0.0+1.0i, 0.0+0.0i).finished();
    // Eigen::Matrix2cd sz = (Eigen::Matrix2cd() <<
    //     1.0 + 0.0i,  0.0 + 0.0i,
    //     0.0 + 0.0i, -1.0 + 0.0i).finished();
    // Eigen::Matrix2cd sp = (Eigen::Matrix2cd() <<
    //     0.0, 1.0,
    //     0.0, 0.0).finished();
    // Eigen::Matrix2cd sm = (Eigen::Matrix2cd() <<
    //     0.0, 0.0,
    //     1.0, 0.0).finished();
    // Eigen::Matrix2cd nu  = (Eigen::Matrix2cd() <<
    //     1.0, 0.0,
    //     0.0, 0.0).finished();
    // Eigen::Matrix2cd nd  = (Eigen::Matrix2cd() <<
    //     0.0, 0.0,
    //     0.0, 1.0).finished();
    // Eigen::Matrix2cd id  = (Eigen::Matrix2cd() <<
    //     1.0, 0.0,
    //     0.0, 1.0).finished();

    std::array<Eigen::Vector2cd, 2> sx_spinors{(Eigen::Vector2cd() << 1.0 + 0.0i, 1.0 + 0.0i).finished() / std::sqrt(2),
                                               (Eigen::Vector2cd() << 1.0 + 0.0i, -1.0 + 0.0i).finished() / std::sqrt(2)};

    std::array<Eigen::Vector2cd, 2> sy_spinors{(Eigen::Vector2cd() << 1.0 + 0.0i, 0.0 + 1.0i).finished() / std::sqrt(2),
                                               (Eigen::Vector2cd() << 1.0 + 0.0i, 0.0 - 1.0i).finished() / std::sqrt(2)};

    std::array<Eigen::Vector2cd, 2> sz_spinors{(Eigen::Vector2cd() << 1.0 + 0.0i, 0.0 + 0.0i).finished() / std::sqrt(2),
                                               (Eigen::Vector2cd() << 0.0 + 0.0i, 1.0 + 0.0i).finished() / std::sqrt(2)};

    std::vector<Eigen::MatrixXcd> SX;
    std::vector<Eigen::MatrixXcd> SY;
    std::vector<Eigen::MatrixXcd> SZ;
    std::vector<Eigen::MatrixXcd> II;
    /* clang-format on */

    Eigen::MatrixXcd gen_embedded_spin_half_operator(const Eigen::Matrix2cd &s, size_t at, size_t sites, bool swap) {
        // Don't forget to set "swap = true" if you intend to use the result as a tensor.
        return gen_embedded_spin_operator(s, at, sites, swap);
    }

    std::vector<Eigen::Matrix4cd> gen_twobody_spins(const Eigen::Matrix2cd &s, bool swap)
    // Returns a pair of two-body 4x4 spin operators for embedded in a two-site Hilbert space:
    //        (σ ⊗ i, i ⊗ σ)
    // where σ is a 2x2 (pauli) matrix and i is the 2x2 identity matrix.
    // Don't forget to set "swap = true" if you intend to use the result as a tensor.
    {
        std::vector<Eigen::Matrix4cd> S;
        for(size_t site = 0; site < 2; site++) S.emplace_back(gen_embedded_spin_operator(s, site, 2, swap));
        return S;
    }

    bool is_valid_axis(std::string_view axis) { return std::find(valid_axis_str.begin(), valid_axis_str.end(), axis) != valid_axis_str.end(); }

    int get_sign(std::string_view axis) {
        if(axis.empty()) throw std::logic_error("get_sign: empty axis string");
        if(axis[0] == '+')
            return 1;
        else if(axis[0] == '-')
            return -1;
        else
            return 0;
    }

    std::string_view get_axis_unsigned(std::string_view axis) {
        if(axis.empty()) throw std::logic_error("get_axis_unsigned: empty axis string");
        if(not is_valid_axis(axis)) throw except::runtime_error("get_axis_unsigned: invalid axis string [{}]. Choose one of (+-) x,y,z or i,id.", axis);
        int sign = get_sign(axis);
        if(sign == 0) {
            return axis.substr(0, 1);
        } else {
            return axis.substr(1, 1);
        }
    }

    Eigen::Vector2cd get_spinor(std::string_view axis, int sign) {
        auto axus = get_axis_unsigned(axis);
        if(axus == "x" and sign >= 0) return sx_spinors[0];
        if(axus == "x" and sign < 0) return sx_spinors[1];
        if(axus == "y" and sign >= 0) return sy_spinors[0];
        if(axus == "y" and sign < 0) return sy_spinors[1];
        if(axus == "z" and sign >= 0) return sz_spinors[0]; // Spin up is |0> = (1,0)
        if(axus == "z" and sign < 0) return sz_spinors[1];  // Spin down is |1> = (0,1)
        throw except::runtime_error("get_spinor given invalid axis: {}", axis);
    }

    Eigen::Vector2cd get_spinor(std::string_view axis) { return get_spinor(axis, get_sign(axis)); }

    Eigen::Matrix2cd get_pauli(std::string_view axis) {
        auto axus = get_axis_unsigned(axis);
        if(axus == "x") return sx;
        if(axus == "y") return sy;
        if(axus == "z") return sz;
        if(axus == "i") return id;
        throw except::runtime_error("get_pauli: could not match axis string: {}", axis);
    }

}