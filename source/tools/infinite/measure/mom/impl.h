#pragma once
#include "math/cast.h"
#include "math/eig.h"
#include "math/tenx.h"
#include "qm/time.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/common/views.h"
#include "tools/infinite/measure.h"
using tools::infinite::measure::RealScalar;

template<typename Scalar, typename OpType>
// requires sfinae::is_std_complex_v<Scalar>
auto moment_generating_function(const StateInfinite<Scalar> &state_original, std::vector<Eigen::Tensor<OpType, 2>> &Op_vec) {
    using CplxScalar                        = std::complex<RealScalar<Scalar>>;
    StateInfinite<CplxScalar> state_evolved = state_original;

    long bond_lim = 5 * state_evolved.chiC();
    for(auto &Op : Op_vec) {
        // Evolve
        Eigen::Tensor<CplxScalar, 3> mps_evo = Op.contract(state_evolved.get_2site_mps(), tenx::idx({0}, {0}));
        state_evolved.set_mps(mps_evo, MergeEvent::GATE, svd::config(bond_lim));
        if(&Op != &Op_vec.back()) { state_evolved.swap_AB(); }
    }

    long sizeLB = state_evolved.chiB() * state_evolved.chiB();
    // Normalize
    Eigen::Tensor<CplxScalar, 2> transfer_matrix_theta_evn =
        tools::common::views<CplxScalar>::get_transfer_matrix_theta_evn(state_evolved).reshape(tenx::array2{sizeLB, sizeLB});
    eig::solver solver;
    auto        nev = 1;
    auto        ncv = settings::precision::eigs_ncv_min;

    solver.eigs(transfer_matrix_theta_evn.data(), sizeLB, nev, ncv, eig::Ritz::LM, eig::Form::NSYM, eig::Side::R, std::nullopt, eig::Shinv::OFF, eig::Vecs::OFF,
                eig::Dephase::OFF);
    auto eigval                   = eig::view::get_eigval<CplxScalar>(solver.result, 0);
    auto new_theta_evn_normalized = tools::common::views<CplxScalar>::get_theta_evn(state_evolved, sqrt(eigval));
    auto old_theta_evn_normalized = tools::common::views<CplxScalar>::get_theta_evn(state_original);
    long sizeL                    = new_theta_evn_normalized.dimension(1) * state_original.chiA();
    long sizeR                    = new_theta_evn_normalized.dimension(3) * state_original.chiB();

    Eigen::Tensor<CplxScalar, 2> transfer_matrix_G = new_theta_evn_normalized.contract(old_theta_evn_normalized.conjugate(), tenx::idx({0, 2}, {0, 2}))
                                                         .shuffle(tenx::array4{0, 2, 1, 3})
                                                         .reshape(tenx::array2{sizeL, sizeR});
    // Compute the characteristic function G(a).
    solver.eigs(transfer_matrix_G.data(), transfer_matrix_G.dimension(0), nev, ncv, eig::Ritz::LM, eig::Form::NSYM, eig::Side::R, std::nullopt, eig::Shinv::OFF,
                eig::Vecs::OFF, eig::Dephase::OFF);
    //    solver.eig(transfer_matrix_G.data(),(int)transfer_matrix_G.dimension(0), 1, eigs_default_ncv, Ritz::LM, Side::R, false);
    auto lambdaG = eig::view::get_eigval<CplxScalar>(solver.result, 0);
    return lambdaG;
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_mom(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_per_site_mom) return tensors.measurements.energy_per_site_mom.value();
    using CplxScalar  = std::complex<RealScalar<Scalar>>;
    const auto &state = tensors.get_state();
    const auto &model = tensors.get_model();
    if(state.chiC() <= 2) {
        tensors.measurements.energy_per_site_mom          = std::numeric_limits<RealScalar<Scalar>>::quiet_NaN();
        tensors.measurements.energy_variance_per_site_mom = std::numeric_limits<RealScalar<Scalar>>::quiet_NaN();
        return tensors.measurements.energy_per_site_mom.value();
    }
    tools::log->trace("Measuring energy mom");
    auto t_ene_mom = tid::tic_scope("ene_mom");

    auto  a      = CplxScalar(static_cast<RealScalar<Scalar>>(0.0), static_cast<RealScalar<Scalar>>(5e-3));
    auto &h_evn  = model.get_2site_ham_AB();
    auto &h_odd  = model.get_2site_ham_BA();
    auto  Op_vec = qm::time::compute_G<CplxScalar>(a, 4, h_evn, h_odd);

    // The following only works if state.MPS has been normalized! I.e, you have to have run MPS->compute_mps_components() prior.
    constexpr auto two                                = RealScalar<Scalar>{2};
    auto           l                                  = two; // Number of sites in a unit cell
    auto           lambdaG                            = moment_generating_function(state, Op_vec);
    auto           G                                  = pow(lambdaG, RealScalar<Scalar>{1} / l);
    auto           logG                               = std::log(lambdaG) / l;
    auto           logGc                              = std::log(conj(lambdaG)) / l;
    auto           O                                  = (logG - logGc) / (two * a);
    auto           VarO                               = two * std::log(abs(G)) / (a * a);
    tensors.measurements.energy_per_site_mom          = std::real(O);
    tensors.measurements.energy_variance_per_site_mom = std::real(VarO);
    return tensors.measurements.energy_per_site_mom.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_mom(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_per_site_mom) return tensors.measurements.energy_variance_per_site_mom.value();
    tensors.measurements.energy_per_site_mom = tools::infinite::measure::energy_per_site_mom(tensors);
    return tensors.measurements.energy_variance_per_site_mom.value();
}

