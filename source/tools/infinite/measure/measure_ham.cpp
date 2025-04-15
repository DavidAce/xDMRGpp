#include "config/settings.h"
#include "math/svd.h"
#include "math/tenx.h"
#include "qm/qm.h"
#include "tensors/model/ModelInfinite.h"
#include "tensors/site/mpo/MpoSite.h"
#include "tensors/state/StateInfinite.h"
#include "tensors/TensorsInfinite.h"
#include "tid/tid.h"
#include "tools/common/log.h"
#include "tools/common/views.h"
#include "tools/infinite/measure.h"
using tools::infinite::measure::RealScalar;

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_per_site_ham(const TensorsInfinite<Scalar> &tensors) {
    auto        t_ene_ham = tid::tic_scope("ene_ham");
    const auto &state     = *tensors.state;
    const auto &model     = *tensors.model;
    if(tensors.measurements.energy_per_site_ham) return tensors.measurements.energy_per_site_ham.value();
    if(state.measurements.bond_dim <= 2) return std::numeric_limits<double>::quiet_NaN();
    if(state.chiA() != state.chiB()) return std::numeric_limits<double>::quiet_NaN();
    if(state.chiA() != state.chiC()) return std::numeric_limits<double>::quiet_NaN();
    if(state.chiB() != state.chiC()) return std::numeric_limits<double>::quiet_NaN();
    tools::log->trace("Measuring energy ham");
    auto &h_evn = model.get_2site_ham_AB();
    auto &h_odd = model.get_2site_ham_BA();
    tools::common::views::compute_mps_components(state);
    using namespace tools::common::views;

    Eigen::Tensor<Scalar, 0> E_evn = theta_evn_normalized.contract(h_evn.reshape(tenx::array4{2, 2, 2, 2}), tenx::idx({0, 2}, {0, 1}))
                                         .contract(theta_evn_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                         .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                         .contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E_odd = theta_odd_normalized.contract(h_odd.reshape(tenx::array4{2, 2, 2, 2}), tenx::idx({0, 2}, {0, 1}))
                                         .contract(theta_odd_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                         .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                         .contract(r_odd, tenx::idx({0, 1}, {0, 1}));
    assert(abs(imag(E_evn(0) + E_odd(0))) < 1e-10 and "Energy has an imaginary part!!!");
    tensors.measurements.energy_per_site_ham = 0.5 * std::real(E_evn(0) + E_odd(0));
    return tensors.measurements.energy_per_site_ham.value();
}

template<typename Scalar>
RealScalar<Scalar> tools::infinite::measure::energy_variance_per_site_ham(const TensorsInfinite<Scalar> &tensors) {
    if(tensors.measurements.energy_variance_per_site_ham) return tensors.measurements.energy_variance_per_site_ham.value();
    //    if(tensors.MPS->chiA() != tensors.MPS->chiB()) return std::numeric_limits<double>::quiet_NaN();
    //    if(tensors.MPS->chiA() != tensors.MPS->chiC()) return std::numeric_limits<double>::quiet_NaN();
    //    if(tensors.MPS->chiB() != tensors.MPS->chiC()) return std::numeric_limits<double>::quiet_NaN();
    if(tensors.state->chiC() <= 2) return std::numeric_limits<RealScalar<Scalar>>::quiet_NaN();

    const auto &state = *tensors.state;
    const auto &model = *tensors.model;

    tools::log->trace("Measuring energy variance ham from tensors");

    auto t_var_ham = tid::tic_scope("var_ham");
    using namespace tools::common::views;

    auto &h_evn = model.get_2site_ham_AB();
    auto &h_odd = model.get_2site_ham_BA();
    tools::common::views::compute_mps_components(state);

    Eigen::Tensor<Scalar, 0> E_evn = theta_evn_normalized.contract(h_evn.reshape(tenx::array4{2, 2, 2, 2}), tenx::idx({0, 2}, {0, 1}))
                                         .contract(theta_evn_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                         .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                         .contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E_odd = theta_odd_normalized.contract(h_odd.reshape(tenx::array4{2, 2, 2, 2}), tenx::idx({0, 2}, {0, 1}))
                                         .contract(theta_odd_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                         .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                         .contract(r_odd, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 4> h0 = tenx::TensorCast((tenx::MatrixMap(h_evn) - E_evn(0) * tenx::MatrixType<Scalar>::Identity(4, 4)).eval(), 2, 2, 2, 2);
    Eigen::Tensor<Scalar, 4> h1 = tenx::TensorCast((tenx::MatrixMap(h_odd) - E_odd(0) * tenx::MatrixType<Scalar>::Identity(4, 4)).eval(), 2, 2, 2, 2);

    Eigen::Tensor<Scalar, 0> E2AB = theta_evn_normalized.contract(h0, tenx::idx({0, 2}, {0, 1}))
                                        .contract(h0, tenx::idx({2, 3}, {0, 1}))
                                        .contract(theta_evn_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                        .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                        .contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E2BA = theta_odd_normalized.contract(h1, tenx::idx({0, 2}, {0, 1}))
                                        .contract(h1, tenx::idx({2, 3}, {0, 1}))
                                        .contract(theta_odd_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                        .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                        .contract(r_odd, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 5> thetaABA = theta_evn_normalized.contract(LAGA, tenx::idx({3}, {1}));
    Eigen::Tensor<Scalar, 5> thetaBAB = theta_odd_normalized.contract(LCGB, tenx::idx({3}, {1}));

    Eigen::Tensor<Scalar, 0> E2ABA_1 = thetaABA.contract(h1, tenx::idx({2, 3}, {0, 1}))
                                           .contract(h0, tenx::idx({0, 3}, {0, 1}))
                                           .contract(thetaABA.conjugate(), tenx::idx({3, 4, 2}, {0, 2, 3}))
                                           .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                           .contract(r_odd, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E2BAB_1 = thetaBAB.contract(h1, tenx::idx({0, 2}, {0, 1}))
                                           .contract(h0, tenx::idx({4, 1}, {0, 1}))
                                           .contract(thetaBAB.conjugate(), tenx::idx({2, 3, 4}, {0, 2, 3}))
                                           .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                           .contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E2ABA_2 = thetaABA.contract(h0, tenx::idx({0, 2}, {0, 1}))
                                           .contract(h1, tenx::idx({4, 1}, {0, 1}))
                                           .contract(thetaABA.conjugate(), tenx::idx({2, 3, 4}, {0, 2, 3}))
                                           .contract(l_evn, tenx::idx({0, 2}, {0, 1}))
                                           .contract(r_odd, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 0> E2BAB_2 = thetaBAB.contract(h0, tenx::idx({2, 3}, {0, 1}))
                                           .contract(h1, tenx::idx({0, 3}, {0, 1}))
                                           .contract(thetaBAB.conjugate(), tenx::idx({3, 4, 2}, {0, 2, 3}))
                                           .contract(l_odd, tenx::idx({0, 2}, {0, 1}))
                                           .contract(r_evn, tenx::idx({0, 1}, {0, 1}));

    Eigen::Tensor<Scalar, 2> E2d_L_evn = theta_evn_normalized.contract(h0, tenx::idx({0, 2}, {0, 1}))
                                             .contract(theta_evn_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                             .contract(l_evn, tenx::idx({0, 2}, {0, 1}));

    Eigen::Tensor<Scalar, 2> E2d_R_evn = theta_evn_normalized.contract(h0, tenx::idx({0, 2}, {0, 1}))
                                             .contract(theta_evn_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                             .contract(r_evn, tenx::idx({1, 3}, {0, 1}));

    Eigen::Tensor<Scalar, 2> E2d_L_odd = theta_odd_normalized.contract(h1, tenx::idx({0, 2}, {0, 1}))
                                             .contract(theta_odd_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                             .contract(l_odd, tenx::idx({0, 2}, {0, 1}));

    Eigen::Tensor<Scalar, 2> E2d_R_odd = theta_odd_normalized.contract(h1, tenx::idx({0, 2}, {0, 1}))
                                             .contract(theta_odd_normalized.conjugate(), tenx::idx({2, 3}, {0, 2}))
                                             .contract(r_odd, tenx::idx({1, 3}, {0, 1}));

    std::array<Eigen::IndexPair<long>, 0> pair         = {};
    Eigen::Tensor<Scalar, 4>              fixpoint_evn = r_evn.contract(l_evn, pair);
    Eigen::Tensor<Scalar, 4>              fixpoint_odd = r_odd.contract(l_odd, pair);

    long                     sizeLA                        = state.chiC();
    long                     sizeLB                        = state.chiB();
    Eigen::Tensor<Scalar, 2> one_minus_transfer_matrix_evn = tenx::TensorCast(tenx::MatrixType<Scalar>::Identity(sizeLB * sizeLB, sizeLA * sizeLA).eval()) -
                                                             (transfer_matrix_evn - fixpoint_evn).reshape(tenx::array2{sizeLB * sizeLB, sizeLA * sizeLA});
    Eigen::Tensor<Scalar, 2> one_minus_transfer_matrix_odd = tenx::TensorCast(tenx::MatrixType<Scalar>::Identity(sizeLA * sizeLA, sizeLB * sizeLB).eval()) -
                                                             (transfer_matrix_odd - fixpoint_odd).reshape(tenx::array2{sizeLA * sizeLA, sizeLB * sizeLB});

    svd::solver svd;

    Eigen::Tensor<Scalar, 4> E_evn_pinv = svd.pseudo_inverse(one_minus_transfer_matrix_evn).reshape(tenx::array4{sizeLB, sizeLB, sizeLA, sizeLA});
    Eigen::Tensor<Scalar, 4> E_odd_pinv = svd.pseudo_inverse(one_minus_transfer_matrix_odd).reshape(tenx::array4{sizeLA, sizeLA, sizeLB, sizeLB});
    Eigen::Tensor<Scalar, 0> E2LRP_ABAB = E2d_L_evn.contract(E_evn_pinv, tenx::idx({0, 1}, {0, 1})).contract(E2d_R_evn, tenx::idx({0, 1}, {0, 1}));
    Eigen::Tensor<Scalar, 0> E2LRP_ABBA = E2d_L_evn.contract(transfer_matrix_LAGA, tenx::idx({0, 1}, {0, 1}))
                                              .contract(E_odd_pinv, tenx::idx({0, 1}, {0, 1}))
                                              .contract(E2d_R_odd, tenx::idx({0, 1}, {0, 1}));
    Eigen::Tensor<Scalar, 0> E2LRP_BABA = E2d_L_odd.contract(E_odd_pinv, tenx::idx({0, 1}, {0, 1})).contract(E2d_R_odd, tenx::idx({0, 1}, {0, 1}));
    Eigen::Tensor<Scalar, 0> E2LRP_BAAB = E2d_L_odd.contract(transfer_matrix_LCGB, tenx::idx({0, 1}, {0, 1}))
                                              .contract(E_evn_pinv, tenx::idx({0, 1}, {0, 1}))
                                              .contract(E2d_R_evn, tenx::idx({0, 1}, {0, 1}));

    Scalar e2ab      = E2AB(0);
    Scalar e2ba      = E2BA(0);
    Scalar e2aba_1   = E2ABA_1(0);
    Scalar e2bab_1   = E2BAB_1(0);
    Scalar e2aba_2   = E2ABA_2(0);
    Scalar e2bab_2   = E2BAB_2(0);
    Scalar e2lrpabab = E2LRP_ABAB(0);
    Scalar e2lrpabba = E2LRP_ABBA(0);
    Scalar e2lrpbaba = E2LRP_BABA(0);
    Scalar e2lrpbaab = E2LRP_BAAB(0);
    tensors.measurements.energy_variance_per_site_ham =
        std::real(0.5 * (e2ab + e2ba) + 0.5 * (e2aba_1 + e2bab_1 + e2aba_2 + e2bab_2) + e2lrpabab + e2lrpabba + e2lrpbaba + e2lrpbaab);
    return tensors.measurements.energy_variance_per_site_ham.value();
}