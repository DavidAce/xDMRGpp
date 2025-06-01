#pragma once
#include "../../mps.h"
#include "config/settings.h"
#include "debug/exceptions.h"
#include "general/iter.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"
#include "tools/finite/measure/spin.h"
#include <bitset>
#include <Eigen/QR>
#include <fmt/ranges.h>

using tools::finite::mps::RealScalar;

inline std::vector<long> tools::finite::mps::init::get_valid_bond_dimensions(size_t sizeplusone, long spin_dim, long bond_lim) {
    // Construct a valid set of bond dimensions
    std::vector<long> bond_dimensions(sizeplusone, 1);
    bond_dimensions.front() = 1;
    bond_dimensions.back()  = 1;
    for(size_t i = 1; i < bond_dimensions.size() / 2; i++) { bond_dimensions.at(i) = std::min(spin_dim * bond_dimensions.at(i - 1), bond_lim); }
    for(size_t i = bond_dimensions.size() - 2; i >= bond_dimensions.size() / 2; i--) {
        bond_dimensions.at(i) = std::min(spin_dim * bond_dimensions.at(i + 1), bond_lim);
    }
    return bond_dimensions;
}

template<typename Scalar>
Eigen::Tensor<Scalar, 2> get_random_unitary_matrix(long rows, long cols) {
    using mat_t   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using vec_t   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    auto gaussian = []() -> Scalar {
        using Real = decltype(std::real(std::declval<Scalar>()));
        if constexpr(sfinae::is_std_complex_v<Scalar>)
            return Scalar(rnd::normal(Real(0.), Real(1.)), rnd::normal(Real(0.), Real(1.))) / std::sqrt<Real>(2.0);
        else
            return rnd::normal(Real(0.0), Real(1.));
    };
    mat_t M = mat_t::NullaryExpr(rows, cols, gaussian);

    auto qr = Eigen::ColPivHouseholderQR<mat_t>();
    qr.compute(M);
    mat_t Q = qr.householderQ().setLength(M.cols()) * mat_t::Identity(M.rows(), qr.nonzeroPivots());
    if(rows != cols) { return tenx::TensorCast(Q.leftCols(std::min(Q.cols(), cols))); }
    vec_t R = qr.matrixR().diagonal().topRows(qr.nonzeroPivots());
    Q *= R.cwiseProduct(R.cwiseAbs().cwiseInverse()).asDiagonal();
    return tenx::TensorMap(Q);
}

template<typename Scalar>
void tools::finite::mps::init::set_midchain_singlet_neel_state(StateFinite<Scalar> &state, StateInitType type, std::string_view axis, std::string &pattern) {
    tools::log->debug("Setting Néel state of type {} on axis {} with a midchain singlet {}", enum2sv(type), axis,
                      pattern.empty() ? "" : fmt::format(" | from pattern: {}", pattern));
    move_center_point_to_middle(state);
    Eigen::Tensor<cx64, 1> L(1);
    L.setConstant(cx64{1.0, 0.0});
    Eigen::Tensor<cx64, 1> LC(2); // L for the singlet
    LC.setConstant(RealScalar<Scalar>(1.0 / std::sqrt(2)));
    auto axus = qm::spin::half::get_axis_unsigned(axis);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state on axis [y] which impliex CPLX");
    using namespace qm::spin::half::tensor;
    std::array<Eigen::Tensor<cx64, 3>, 2> spinors = {get_spinor(axus, +1).reshape(tenx::array3{2, 1, 1}), //
                                                     get_spinor(axus, -1).reshape(tenx::array3{2, 1, 1})};
    Eigen::Tensor<cx64, 3>                singlet_a(2, 1, 2);
    Eigen::Tensor<cx64, 3>                singlet_b(2, 2, 1);
    singlet_a.slice(std::array<long, 3>{0, 0, 0}, std::array<long, 3>{2, 1, 1}) = spinors[0];  // a_up
    singlet_a.slice(std::array<long, 3>{0, 0, 1}, std::array<long, 3>{2, 1, 1}) = spinors[1];  // a_down
    singlet_b.slice(std::array<long, 3>{0, 0, 0}, std::array<long, 3>{2, 1, 1}) = spinors[1];  // b_down
    singlet_b.slice(std::array<long, 3>{0, 1, 0}, std::array<long, 3>{2, 1, 1}) = -spinors[0]; // b_up // Minus sign for singlet instead of triplet state
    auto bitfield                                                               = tools::get_bitfield(state.get_length(), pattern);
    if(bitfield.empty() or bitfield.size() != state.get_length()) {
        bitfield.resize(state.get_length(), 0);
        for(auto &&[i, p] : iter::enumerate(bitfield)) {
            if(i == state.get_length() / 2 - 1) {
                p = 'a';
            } else if(i == state.get_length() / 2) {
                p = 'b';
            } else {
                p = num::mod<size_t>(i, 2) == 0 ? '0' : '1'; // Set Neel pattern 0101010}
            }
        }
        if(rnd::uniform_integer_box<size_t>(0, 1) == 1) {
            // Reverse with 50% probability
            std::reverse(bitfield.begin(), bitfield.end());
            bitfield.at(state.get_length() / 2 - 1) = 'a';
            bitfield.at(state.get_length() / 2)     = 'b';
        }
    }

    std::string label = "A";
    for(const auto &[pos, mps_ptr] : iter::enumerate(state.mps_sites)) {
        auto &&mps = *mps_ptr;
        auto   bit = bitfield.at(pos);
        if(bit == '0') {
            mps.set_mps(spinors.at(0), L, 0, label);
        } else if(bit == '1') {
            mps.set_mps(spinors.at(1), L, 0, label);
        } else if(bit == 'a') {
            mps.set_mps(singlet_a, L, 0, label); // This is an "AC" site, so LC goes on the right of this matrix
        } else if(bit == 'b') {
            mps.set_mps(singlet_b, L, 0, label); // This is a "B" site
        }
        if(mps.isCenter()) {
            label = "B";
            mps.set_LC(LC);
        }
    }
    pattern        = fmt::format("b{}", bitfield);
    state.popcount = 1 + safe_cast<size_t>(std::count(bitfield.begin(), bitfield.end(), '1')); // Add one because of the singlet
    move_center_point_to_pos_dir(state, 0, 1);
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
    tools::log->debug("Initial state: {}", bitfield);
}

template<typename Scalar>
void tools::finite::mps::init::set_random_entangled_state_haar(StateFinite<Scalar> &state, StateInitType type, long bond_lim) {
    tools::log->info("Setting random entangled state with Haar distribution | bond_lim {}", bond_lim);
    const auto  spin_dim        = state.template get_mps_site<size_t>(0).spin_dim();
    auto        bond_dimensions = init::get_valid_bond_dimensions(state.get_length() + 1, spin_dim, bond_lim);
    bool        pastCenter      = false;
    std::string label           = "A";
    using VectorReal            = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    for(auto &mps_ptr : state.mps_sites) {
        auto                    &mps  = *mps_ptr;
        auto                     chiL = bond_dimensions[mps.get_position()];
        auto                     chiR = bond_dimensions[mps.get_position() + 1];
        Eigen::Tensor<Scalar, 2> U;
        Eigen::Tensor<Scalar, 3> G(spin_dim, chiL, chiR);
        if(label == "A") {
            if(type == StateInitType::REAL) U = get_random_unitary_matrix<RealScalar<Scalar>>(spin_dim * chiL, chiR).template cast<Scalar>();
            if(type == StateInitType::CPLX) U = get_random_unitary_matrix<Scalar>(spin_dim * chiL, chiR);
            chiR     = U.dimension(1);
            auto rsh = std::array<long, 3>{spin_dim, chiL, chiR};
            G        = U.reshape(rsh);
        }
        if(label == "B") {
            if(type == StateInitType::REAL) U = get_random_unitary_matrix<RealScalar<Scalar>>(spin_dim * chiR, chiL).template cast<Scalar>();
            if(type == StateInitType::CPLX) U = get_random_unitary_matrix<Scalar>(spin_dim * chiR, chiL);
            chiL     = U.dimension(1);
            auto rsh = std::array<long, 3>{spin_dim, chiR, chiL};
            auto shf = std::array<long, 3>{0, 2, 1};
            G        = U.reshape(rsh).shuffle(shf);
        }
        auto smdt = pastCenter ? chiR : chiL;
        auto L    = tenx::TensorCast(VectorReal::Ones(smdt).normalized());

        mps.set_mps(G, L, 0, label);
        if(mps.isCenter()) {
            auto LC = tenx::TensorCast(VectorReal::Ones(chiR).normalized());
            mps.set_LC(LC);
            pastCenter = true;
            label      = "B";
        }
    }
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}

template<typename Scalar>
void tools::finite::mps::init::random_entangled_state(StateFinite<Scalar> &state, StateInitType type, [[maybe_unused]] std::string_view axis,
                                                      [[maybe_unused]] bool use_eigenspinors, long bond_lim) {
    // TODO: Make version with/without eigenspinors
    set_random_entangled_state_with_random_spinors(state, type, bond_lim);
}

template<typename Scalar>
void tools::finite::mps::init::set_random_entangled_state_with_random_spinors(StateFinite<Scalar> &state, StateInitType type, long bond_lim) {
    tools::log->info("Setting random entangled state with random unit spinors");
    // static_assert(sfinae::is_std_complex_v<Scalar>);
    const auto  spin_dim        = state.template get_mps_site<size_t>(0).spin_dim();
    auto        bond_dimensions = init::get_valid_bond_dimensions(state.get_length() + 1, spin_dim, bond_lim);
    bool        pastCenter      = false;
    std::string label           = "A";
    using VectorReal            = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    using VectorType            = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    for(auto &mps_ptr : state.mps_sites) {
        auto      &mps  = *mps_ptr;
        auto       chiL = bond_dimensions[mps.get_position()];
        auto       chiR = bond_dimensions[mps.get_position() + 1];
        auto       size = spin_dim * chiL * chiR;
        auto       smdt = pastCenter ? chiR : chiL;
        VectorReal Ltmp = VectorReal(smdt).unaryExpr([]([[maybe_unused]] auto dummy) { return rnd::uniform<RealScalar<Scalar>>(0, 1); });
        std::sort(Ltmp.data(), Ltmp.data() + Ltmp.size(), std::greater<RealScalar<Scalar>>());
        Eigen::Tensor<Scalar, 1> L = tenx::TensorCast(Ltmp.normalized()).template cast<Scalar>();
        Eigen::Tensor<Scalar, 3> G(spin_dim, chiL, chiR);
        if(type == StateInitType::REAL) {
            VectorReal Gtmp = VectorReal(size).unaryExpr([]([[maybe_unused]] auto dummy) { return rnd::uniform<RealScalar<Scalar>>(-1, 1); });
            G               = tenx::TensorCast(Gtmp.normalized(), spin_dim, chiL, chiR).template cast<Scalar>();
        } else if(type == StateInitType::CPLX) {
            if constexpr(sfinae::is_std_complex_v<Scalar>) {
                VectorType Gtmp = VectorType(size).unaryExpr([]([[maybe_unused]] auto dummy) { return rnd::uniform_complex_in_circle<RealScalar<Scalar>>(1); });
                G               = tenx::TensorCast(Gtmp.normalized(), spin_dim, chiL, chiR).template cast<Scalar>();
            } else {
                tools::log->warn("set_random_entangled_state_with_random_spinors: Detected StateInitType::CPLX, but the state has real scalar type [{}]. \n"
                                 "Reverting the distribution uniform_complex_in_circle(1) -> uniform(-1,1)",
                                 sfinae::type_name<Scalar>());
                VectorType Gtmp = VectorType(size).unaryExpr([]([[maybe_unused]] auto dummy) { return rnd::uniform<RealScalar<Scalar>>(-1, 1); });
                G               = tenx::TensorCast(Gtmp.normalized(), spin_dim, chiL, chiR).template cast<Scalar>();
            }
        }

        mps.set_mps(G, L, 0, label);
        if(mps.isCenter()) {
            Ltmp = VectorReal(chiR).unaryExpr([]([[maybe_unused]] auto dummy) { return rnd::uniform<RealScalar<Scalar>>(0, 1); });
            std::sort(Ltmp.data(), Ltmp.data() + Ltmp.size(), std::greater<RealScalar<Scalar>>());
            Eigen::Tensor<Scalar, 1> LC = tenx::TensorCast(Ltmp.normalized()).template cast<Scalar>();
            mps.set_LC(LC);
            pastCenter = true;
            label      = "B";
        }
    }
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3> get_random_spinor_tensor(const std::array<long, 3> &dims, std::string_view label, const std::vector<Eigen::VectorXcd> &eigenspinors) {
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = typename MatrixType::RealScalar;
    if(label == "A") {
        auto G = Eigen::Tensor<Scalar, 3>(dims);
        G.setZero();
        auto   Gmap       = Eigen::Map<MatrixType>(G.data(), dims[0] * dims[1], dims[2]);
        auto   ext        = std::array<long, 3>{2, 1, 1};
        size_t rounds     = 0;
        bool   isIdentity = false;
        while(true) {
            for(long j = 0; j < dims[2]; ++j) {
                for(long i = 0; i < dims[1]; ++i) {
                    auto idx     = rnd::uniform_integer_box(0ul, eigenspinors.size() - 1);
                    auto off     = std::array<long, 3>{0, i, j};
                    auto eigspin = tenx::TensorCast(eigenspinors.at(idx));
                    if constexpr(!sfinae::is_std_complex_v<Scalar>) {
                        if(!tenx::isReal(eigspin)) throw except::runtime_error("get_random_spinor_tensor(): Detected non-real eigenspinor.");
                    }
                    G.slice(off, ext) = tenx::asScalarType<Scalar>(eigspin).reshape(ext);
                }
                Gmap.colwise().normalize();
                auto GtG   = Gmap.adjoint() * Gmap;
                isIdentity = GtG.isIdentity(RealScalar{1e-12f});
                if(isIdentity) break;
            }
            rounds++;
            if(isIdentity) break;
        }
        return G;
    } else if(label == "B") {
        return get_random_spinor_tensor<Scalar>(std::array<long, 3>{dims[0], dims[2], dims[1]}, "A", eigenspinors).shuffle(std::array<long, 3>{0, 2, 1});
    }
    throw except::logic_error("Unexpected label: {}", label);
}

template<typename Scalar>
void tools::finite::mps::init::set_random_entangled_state_on_axes_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type,
                                                                                     const std::vector<std::string> &axes, long bond_lim) {
    auto spin_dim        = state.template get_mps_site<size_t>(0).spin_dim();
    auto bond_dimensions = init::get_valid_bond_dimensions(state.get_length() + 1, spin_dim, bond_lim);
    auto eigenspinors    = std::vector<Eigen::VectorXcd>();
    tools::log->info("Setting random entangled state on axes {}: bond dims {}", axes, bond_dimensions);
    using VectorReal = Eigen::Matrix<RealScalar<Scalar>, Eigen::Dynamic, 1>;
    for(const auto &axis : axes) {
        auto axus = qm::spin::half::get_axis_unsigned(axis);
        if(type == StateInitType::REAL and axus == "y") throw except::logic_error("axis y is incompatible with StateInitType::REAL");
        eigenspinors.emplace_back(qm::spin::half::get_spinor(axus, 1));
        eigenspinors.emplace_back(qm::spin::half::get_spinor(axus, -1));
    }
    bool        past_center = false;
    std::string label       = "A";
    for(auto &mps_ptr : state.mps_sites) {
        auto &mps  = *mps_ptr;
        auto  chiL = bond_dimensions[mps.get_position()];
        auto  chiR = bond_dimensions[mps.get_position() + 1];
        auto  dims = std::array<long, 3>{spin_dim, chiL, chiR};
        auto  G    = get_random_spinor_tensor<Scalar>(dims, label, eigenspinors);
        auto  L    = Eigen::Tensor<RealScalar<Scalar>, 1>(tenx::TensorCast(VectorReal::Ones((past_center ? chiR : chiL)).normalized()));

        mps.set_mps(G, L, 0, label);
        if(mps.isCenter()) {
            Eigen::Tensor<RealScalar<Scalar>, 1> LC = tenx::TensorCast(VectorReal::Ones(chiR).normalized());
            mps.set_LC(LC);
            past_center = true;
            label       = "B";
        }
    }
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}

template<typename Scalar>
void tools::finite::mps::init::set_random_entangled_state_on_axis_using_eigenspinors(StateFinite<Scalar> &state, StateInitType type, std::string_view axis,
                                                                                     long bond_lim) {
    const auto spin_dim        = state.template get_mps_site<size_t>(0).spin_dim();
    auto       bond_dimensions = init::get_valid_bond_dimensions(state.get_length() + 1, spin_dim, bond_lim);
    auto       axus            = qm::spin::half::get_axis_unsigned(axis);
    auto       sign            = qm::spin::half::get_sign(axis);
    tools::log->info("Setting random entangled state on axis {} using eigenspinors of the pauli matrix σ{}: bond dims {}", axis, axus, bond_dimensions);
    if(type == StateInitType::REAL and axus == "y") throw std::runtime_error("StateInitType REAL incompatible with state in axis [y] which impliex CPLX");
    bool        past_center = false;
    std::string label       = "A";
    using MatrixType        = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    for(auto &mps_ptr : state.mps_sites) {
        auto                    &mps  = *mps_ptr;
        auto                     chiL = bond_dimensions[mps.get_position()];
        auto                     chiR = bond_dimensions[mps.get_position() + 1];
        auto                     rows = past_center ? chiL : chiL * spin_dim;
        auto                     cols = past_center ? chiR * spin_dim : chiR;
        Eigen::Tensor<Scalar, 1> L    = tenx::TensorCast(Eigen::VectorXd::Ones(chiL).normalized()).template cast<Scalar>();
        Eigen::Tensor<Scalar, 3> G(spin_dim, chiL, chiR);
        Eigen::Map<MatrixType>   G_mat(G.data(), rows, cols);
        // Here we construct the set of spinors for each site using randomly selected eigenspinors
        std::array<long, 3> offset3;
        std::array<long, 3> extent3{spin_dim, 1, 1};
        std::array<long, 1> extent1{spin_dim};
        for(long col = 0; col < chiR; col++) {
            for(long row = 0; row < chiL; row++) {
                auto rnd_sign                              = 2 * rnd::uniform_integer_01() - 1;
                auto spinor                                = qm::spin::half::tensor::get_spinor(axus, rnd_sign);
                offset3                                    = {0, row, col};
                G.slice(offset3, extent3).reshape(extent1) = tenx::asScalarType<Scalar>(spinor);
            }
        }
        G_mat.colwise().normalize();

        //        tenx::normalize(G);
        //        G = tenx::TensorMap(tenx::VectorCast(G).normalized(), spin_dim, chiL, chiR);
        mps.set_mps(G, L, 0, label);
        if(mps.isCenter()) {
            Eigen::Tensor<Scalar, 1> LC = tenx::TensorCast(VectorType::Ones(chiR).normalized());
            mps.set_LC(LC);
            past_center = true;
            label       = "B";
        }
    }

    // Here we can make sure the state ends up in the correct spin parity sector by flipping the last
    // spin if necessary
    auto spin_component = tools::finite::measure::spin_component(state, axus);
    if(spin_component * sign < 0) {
        auto                                                             &mps  = *state.mps_sites.back();
        auto                                                              chiL = bond_dimensions[mps.get_position()];
        auto                                                              chiR = bond_dimensions[mps.get_position() + 1];
        auto                                                              rows = past_center ? chiL : chiL * spin_dim;
        auto                                                              cols = past_center ? chiR * spin_dim : chiR;
        std::array<long, 3>                                               offset3;
        std::array<long, 3>                                               extent3{spin_dim, 1, 1};
        std::array<long, 1>                                               extent1{spin_dim};
        Eigen::Tensor<Scalar, 3>                                          G(spin_dim, chiL, chiR);
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> G_mat(G.data(), rows, cols);
        MatrixType                                                        pauli_matrix = tenx::asScalarType<Scalar>(qm::spin::half::get_pauli(axus));
        for(long col = 0; col < chiR; col++) {
            for(long row = 0; row < chiL; row++) {
                offset3                                              = {0, row, col};
                Eigen::Tensor<Scalar, 1> old_spinor_tensor           = mps.get_M().slice(offset3, extent3).reshape(extent1);
                VectorType               old_spinor_vector           = tenx::VectorMap(old_spinor_tensor);
                VectorType               pauli_axus_times_old_spinor = pauli_matrix * old_spinor_vector;
                Scalar                   old_spinor_expval           = old_spinor_vector.dot(pauli_axus_times_old_spinor);
                int                      old_spinor_sign             = std::real(old_spinor_expval) > 0 ? 1 : -1;
                auto                     new_spinor_vector           = qm::spin::half::tensor::get_spinor(axus, -old_spinor_sign);
                G.slice(offset3, extent3).reshape(extent1)           = tenx::asScalarType<Scalar>(new_spinor_vector);
            }
        }
        G_mat.colwise().normalize();
        mps.set_M(G);
    }
    if(spin_component * sign < 0) throw except::logic_error("Could not initialize_state in the correct sector");
    state.clear_measurements();
    state.clear_cache();
    state.tag_all_sites_normalized(false); // This operation denormalizes all sites
}

template<typename Scalar>
void tools::finite::mps::init::randomize_given_state(StateFinite<Scalar> &state, StateInitType type) {
    using namespace qm::spin::half::matrix;
    switch(type) {
        case StateInitType::REAL: tools::finite::mps::apply_random_paulis(state, std::vector<Eigen::Matrix2cd>{id, 1.0 / std::sqrt(2.0) * (sx + sz)}); break;
        case StateInitType::CPLX:
            tools::finite::mps::apply_random_paulis(state, std::vector<Eigen::Matrix2cd>{id, 1.0 / std::sqrt(3.0) * (sx + sy + sz)});
            break;
    }
}
