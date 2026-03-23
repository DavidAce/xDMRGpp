#include "algorithms/flbit.h"
#include "config/parse.h"
#include "config/settings.h"
#include "config/threading.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "debug/stacktrace.h"
#include "env/environment.h"
#include "io/fmt_f128_t.h"
#include "math/float.h"
#include "math/linalg.h"
#include "math/num.h"
#include "math/rnd.h"
#include "math/tenx.h"
#include "qm/lbit.h"
#include "qm/spin.h"
#include "tensors/model/ModelFinite.h"
#include "tensors/site/mps/MpsSite.h"
#include "tensors/state/StateFinite.h"
#include "tensors/TensorsFinite.h"
#include "tools/common/log.h"
#include "tools/finite/mps.h"
#include "tools/finite/ops.h"
#include <fmt/format.h>
#include <h5pp/h5pp.h>

auto get_hamiltonian(const flbit &f) -> Eigen::Matrix<cx128, Eigen::Dynamic, Eigen::Dynamic> {
    //    for(const auto &field : f.tensors.model->get_parameter("J1_rand"))
    using matrix_t = Eigen::Matrix<cx128, Eigen::Dynamic, Eigen::Dynamic>;
    auto L         = f.tensors.get_length<size_t>();
    auto N         = static_cast<long>(std::pow(2.0, L));
    auto H         = matrix_t(N, N);
    auto SZ        = qm::spin::gen_manybody_spins(qm::spin::half::sz, L, true);

    auto J1 = f.tensors.model->get_parameter("J1_rand");
    auto J2 = f.tensors.model->get_parameter("J2_rand");
    auto J3 = f.tensors.model->get_parameter("J3_rand");

    H.setZero();
    for(size_t i = 0; i < L - 0; ++i) H += std::any_cast<fp128>(J1[i]) * SZ[i].cast<cx128>();
    for(size_t i = 0; i < L - 1; ++i) {
        auto J2i = std::any_cast<h5pp::varr_t<fp128>>(J2[i]);
        for(size_t j = i + 1; j < L; ++j) {
            auto r  = j - i;
            H      += J2i[r] * SZ[i].cast<cx128>() * SZ[j].cast<cx128>();
        }
    }
    for(size_t i = 0; i < L - 2; ++i) H += std::any_cast<fp128>(J3[i]) * SZ[i + 0].cast<cx128>() * SZ[i + 1].cast<cx128>() * SZ[i + 2].cast<cx128>();

    return H;
}

size_t assert_lbit_evolution(const flbit &f) {
    // Test time evolution as  |psi(t+dt)> = U exp(-iH'dt) U^-1 |psi(t)>, starting from t = 0
    // We get a unitary tensor whose index 1 connects onto index 0 on an MPS, meaning,
    /*
     * @verbatim
     *        |psi>
     *         |
     *         0
     *         1
     *         |
     *         U†
     *         |
     *         0
     * @endverbatim
     *
     *  When we use matrix operations, the multiplication works as
     *
     *  0---U†---1 0---|psi>
     *
     *
     *  The unitary is expressed as a circuit of 2-site gates, composed of many layers U[i].
     *  When we apply the inverse from the left, the whole circuit needs to be reversed:
     *
     *  (...U[3]U[2]U[1]U[0])† |psi>  = (U[0]†U[1]†U[2]†U[3]†...) |psi>
     *
     *
     *
     */
    for(const auto &mps : f.state_lbit->mps_sites) mps->assert_normalized();
    f.state_lbit->assert_validity();
    bool has_time_swap_gates = not f.time_swap_gates_Lbody.empty();
    bool has_time_slow_gates = not f.time_gates_Lbody.empty();
    bool has_unitary_gates   = not f.unitary_gates_2site_layers.empty();

    if(not has_time_swap_gates and not has_time_slow_gates) return 0;
    if(not has_unitary_gates) return 0;

    auto H_exact = get_hamiltonian(f);
    auto H_Lbody = not f.ham_gates_Lbody.empty() ? tenx::MatrixMap(f.ham_gates_Lbody.front().op_t) : tenx::MatrixMap(f.ham_swap_gates_Lbody.front().op_t);
#if defined(DMRG_USE_QUADMATH)
    auto digits10_t = FLT128_DIG;
//    auto epsilon_t  = 1.926e-34; // FLT128_EPSILON;
#else
    auto digits10_t = std::numeric_limits<fp128>::digits10;
//    auto   epsilon_t      = std::numeric_limits<fp128>::epsilon();
#endif
    auto digits10_d  = std::numeric_limits<fp64>::digits10;
    auto epsilon_d   = std::numeric_limits<fp64>::epsilon();
    auto precision_d = std::max({
        epsilon_d,
        std::pow(f.status.trnc_lim, 2.0),
    });

#if defined(DMRG_USE_QUADMATH)
    double max_tevo_error = 100 * precision_d *
                            std::max({
                                1.0,
                                static_cast<double>(fabsq(f.status.delta_t.to_floating_point<cx128>().real())),
                                static_cast<double>(fabsq(f.status.delta_t.to_floating_point<cx128>().imag())),
                            });

#else
    double max_tevo_error = 100 * precision_d *
                            std::max({
                                1.0,
                                static_cast<double>(std::abs(f.status.delta_t.to_floating_point<cx128>().real())),
                                static_cast<double>(std::abs(f.status.delta_t.to_floating_point<cx128>().imag())),
                            });
#endif
    double max_norm_error = settings::precision::max_norm_error;
    double max_untary_err = 10 * epsilon_d;

    tools::log->info("H_exact: {}", linalg::matrix::to_string(H_exact.diagonal().transpose().real(), digits10_t));
    tools::log->info("H_Lbody: {}", linalg::matrix::to_string(H_Lbody.diagonal().transpose().real(), digits10_t));
    if(not H_exact.isApprox(H_Lbody, epsilon_d))
        throw except::runtime_error("Hamiltonians do not match: \nH_exact: {}\nH_Lbody: {}",
                                    linalg::matrix::to_string(H_exact.diagonal().transpose().real(), digits10_t),
                                    linalg::matrix::to_string(H_Lbody.diagonal().transpose().real(), digits10_t));

    auto T_Lgate = has_time_swap_gates ? f.time_swap_gates_Lbody.front() : f.time_gates_Lbody.front();
    auto T_Lbody = tenx::MatrixMap(T_Lgate.op_t);
    auto T_Hgate = qm::lbit::get_time_evolution_gates(f.status.delta_t.to_floating_point<cx128>(), {qm::Gate(H_exact, T_Lgate.pos, T_Lgate.dim)});
    auto T_exact = tenx::MatrixMap(T_Hgate.front().op_t);

    tools::log->info("T_Lbody: {}", linalg::matrix::to_string(T_Lbody.diagonal().transpose(), digits10_d));
    tools::log->info("T_exact: {}", linalg::matrix::to_string(T_exact.diagonal().transpose(), digits10_d));

    if(not T_Lbody.isApprox(T_exact, max_tevo_error))
        throw except::runtime_error("Time evolution operators do not match: \nT_Lbody: {}\nT_exact: {}",
                                    linalg::matrix::to_string(T_Lbody.diagonal().transpose(), digits10_d),
                                    linalg::matrix::to_string(T_exact.diagonal().transpose(), digits10_d));

    auto unitarytensor = qm::lbit::get_unitary_circuit_as_tensor(f.unitary_gates_2site_layers);
    auto unitarymatrix = tenx::MatrixMap(unitarytensor);
    auto exp_lbit_iHdt = has_time_swap_gates ? tenx::MatrixMap(f.time_swap_gates_Lbody.front().op) : tenx::MatrixMap(f.time_gates_Lbody.front().op);

    tools::log->info("exp_lbit_iHdt: {}", linalg::matrix::to_string(exp_lbit_iHdt.diagonal().transpose(), digits10_d));

    // Check that the matrices are unitary
    if(not unitarymatrix.isUnitary()) throw except::logic_error("unitarymatrix is not unitary");
    if(not exp_lbit_iHdt.isUnitary()) throw except::logic_error("exp_lbit_iHdt is not unitary");

    auto             psi_real_init = tools::finite::measure::mps2tensor(*f.state_real_init);
    auto             psi_lbit_init = tools::finite::measure::mps2tensor(*f.state_lbit_init);
    auto             psi_fp128ebd  = tools::finite::measure::mps2tensor(*f.tensors.state);
    auto             psi_lbit_tebd = tools::finite::measure::mps2tensor(*f.state_lbit);
    Eigen::VectorXcd vec_real_init = tenx::VectorMap(psi_real_init);
    Eigen::VectorXcd vec_lbit_init = tenx::VectorMap(psi_lbit_init);
    Eigen::VectorXcd vec_fp128ebd  = tenx::VectorMap(psi_fp128ebd);
    Eigen::VectorXcd vec_lbit_tebd = tenx::VectorMap(psi_lbit_tebd);
    Eigen::VectorXcd vec_lbit_levo = exp_lbit_iHdt * vec_lbit_init;                           // From lbit
    Eigen::VectorXcd vec_lbit_revo = exp_lbit_iHdt * unitarymatrix.adjoint() * vec_real_init; // From real
    Eigen::VectorXcd vec_lbit_devo = exp_lbit_iHdt.conjugate() * vec_lbit_tebd;               // From real
    Eigen::VectorXcd vec_real_levo = unitarymatrix * exp_lbit_iHdt * vec_lbit_init;
    Eigen::VectorXcd vec_real_revo = unitarymatrix * exp_lbit_iHdt * unitarymatrix.adjoint() * vec_real_init;
    Eigen::VectorXcd vec_real_back = unitarymatrix * vec_lbit_init; // Returns the initial lbit state to real basis

    auto overlap_lbit_levo_tebd = vec_lbit_levo.dot(vec_lbit_tebd); // One if time evo works independently of U
    auto overlap_lbit_revo_tebd = vec_lbit_revo.dot(vec_lbit_tebd); // One if time evo and U both work
    auto overlap_lbit_devo_init = vec_lbit_devo.dot(vec_lbit_init); // One if time evo works independently of U
    auto overlap_real_levo_tebd = vec_real_levo.dot(vec_fp128ebd);  // One if time evo and U both work
    auto overlap_real_revo_tebd = vec_real_revo.dot(vec_fp128ebd);  // One if time evo and U both work
    auto overlap_real_back_init = vec_real_back.dot(vec_real_init); // One if U works independently of time evo

    auto real_error_lbit_levo_tebd = std::abs(std::real(overlap_lbit_levo_tebd) - 1.0);
    auto real_error_lbit_revo_tebd = std::abs(std::real(overlap_lbit_revo_tebd) - 1.0);
    auto real_error_lbit_devo_init = std::abs(std::real(overlap_lbit_devo_init) - 1.0);
    auto real_error_real_levo_tebd = std::abs(std::real(overlap_real_levo_tebd) - 1.0);
    auto real_error_real_revo_tebd = std::abs(std::real(overlap_real_revo_tebd) - 1.0);
    auto real_error_real_back_init = std::abs(std::real(overlap_real_back_init) - 1.0);
    auto imag_error_lbit_levo_tebd = std::abs(std::imag(overlap_lbit_levo_tebd));
    auto imag_error_lbit_revo_tebd = std::abs(std::imag(overlap_lbit_revo_tebd));
    auto imag_error_lbit_devo_init = std::abs(std::imag(overlap_lbit_devo_init));
    auto imag_error_real_levo_tebd = std::abs(std::imag(overlap_real_levo_tebd));
    auto imag_error_real_revo_tebd = std::abs(std::imag(overlap_real_revo_tebd));
    auto imag_error_real_back_init = std::abs(std::imag(overlap_real_back_init));
    auto norm_error_lbit_levo_tebd = std::abs(std::abs(overlap_lbit_levo_tebd) - 1.0);
    auto norm_error_lbit_revo_tebd = std::abs(std::abs(overlap_lbit_revo_tebd) - 1.0);
    auto norm_error_lbit_devo_init = std::abs(std::abs(overlap_lbit_devo_init) - 1.0);
    auto norm_error_real_levo_tebd = std::abs(std::abs(overlap_real_levo_tebd) - 1.0);
    auto norm_error_real_revo_tebd = std::abs(std::abs(overlap_real_revo_tebd) - 1.0);
    auto norm_error_real_back_init = std::abs(std::abs(overlap_real_back_init) - 1.0);

    tools::log->info("vec_real_init.norm()   : {:.16f}", vec_real_init.norm());
    tools::log->info("vec_lbit_init.norm()   : {:.16f}", vec_lbit_init.norm());
    tools::log->info("vec_fp128ebd.norm()   : {:.16f}", vec_fp128ebd.norm());
    tools::log->info("vec_lbit_tebd.norm()   : {:.16f}", vec_lbit_tebd.norm());
    tools::log->info("vec_lbit_levo.norm()   : {:.16f}", vec_lbit_levo.norm());
    tools::log->info("vec_lbit_revo.norm()   : {:.16f}", vec_lbit_revo.norm());
    tools::log->info("vec_lbit_devo.norm()   : {:.16f}", vec_lbit_devo.norm());
    tools::log->info("vec_real_levo.norm()   : {:.16f}", vec_real_levo.norm());
    tools::log->info("vec_real_revo.norm()   : {:.16f}", vec_real_revo.norm());
    tools::log->info("vec_real_back.norm()   : {:.16f}", vec_real_back.norm());
    /* clang-format off */
    tools::log->info("overlap_lbit_levo_tebd : {:.16f} | {:.16f} | One if time evo works independently of U", overlap_lbit_levo_tebd, std::abs(overlap_lbit_levo_tebd));
    tools::log->info("overlap_lbit_revo_tebd : {:.16f} | {:.16f} | One if time evo and U both work"         , overlap_lbit_revo_tebd, std::abs(overlap_lbit_revo_tebd));
    tools::log->info("overlap_lbit_devo_init : {:.16f} | {:.16f} | One if time evo works independently of U", overlap_lbit_devo_init, std::abs(overlap_lbit_devo_init));
    tools::log->info("overlap_real_levo_tebd : {:.16f} | {:.16f} | One if time evo and U both work"         , overlap_real_levo_tebd, std::abs(overlap_real_levo_tebd));
    tools::log->info("overlap_real_revo_tebd : {:.16f} | {:.16f} | One if time evo and U both work"         , overlap_real_revo_tebd, std::abs(overlap_real_revo_tebd));
    tools::log->info("overlap_real_lbit_init : {:.16f} | {:.16f} | One if U works independently of time evo", overlap_real_back_init, std::abs(overlap_real_back_init));

    if(std::abs(vec_real_init.norm()-1.0) > max_norm_error) throw except::logic_error("vec_real_init.norm() {0:.{2}f} > {1:.{2}f}", vec_real_init.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_lbit_init.norm()-1.0) > max_norm_error) throw except::logic_error("vec_lbit_init.norm() {0:.{2}f} > {1:.{2}f}", vec_lbit_init.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_fp128ebd.norm()-1.0) > max_norm_error) throw except::logic_error("vec_fp128ebd.norm() {0:.{2}f} > {1:.{2}f}", vec_fp128ebd.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_lbit_tebd.norm()-1.0) > max_norm_error) throw except::logic_error("vec_lbit_tebd.norm() {0:.{2}f} > {1:.{2}f}", vec_lbit_tebd.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_lbit_levo.norm()-1.0) > max_norm_error) throw except::logic_error("vec_lbit_levo.norm() {0:.{2}f} > {1:.{2}f}", vec_lbit_levo.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_lbit_revo.norm()-1.0) > max_norm_error) throw except::logic_error("vec_lbit_revo.norm() {0:.{2}f} > {1:.{2}f}", vec_lbit_revo.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_lbit_devo.norm()-1.0) > max_norm_error) throw except::logic_error("vec_lbit_devo.norm() {0:.{2}f} > {1:.{2}f}", vec_lbit_devo.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_real_levo.norm()-1.0) > max_norm_error) throw except::logic_error("vec_real_levo.norm() {0:.{2}f} > {1:.{2}f}", vec_real_levo.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_real_revo.norm()-1.0) > max_norm_error) throw except::logic_error("vec_real_revo.norm() {0:.{2}f} > {1:.{2}f}", vec_real_revo.norm(), max_norm_error, digits10_d);
    if(std::abs(vec_real_back.norm()-1.0) > max_norm_error) throw except::logic_error("vec_real_back.norm() {0:.{2}f} > {1:.{2}f}", vec_real_back.norm(), max_norm_error, digits10_d);

    if(real_error_lbit_levo_tebd > max_tevo_error) throw except::logic_error("real_error_lbit_levo_tebd {0:.{2}f} > {1:.{2}f}", real_error_lbit_levo_tebd, max_tevo_error, digits10_d);
    if(real_error_lbit_revo_tebd > max_tevo_error) throw except::logic_error("real_error_lbit_revo_tebd {0:.{2}f} > {1:.{2}f}", real_error_lbit_revo_tebd, max_tevo_error, digits10_d);
    if(real_error_lbit_devo_init > max_tevo_error) throw except::logic_error("real_error_lbit_devo_init {0:.{2}f} > {1:.{2}f}", real_error_lbit_devo_init, max_tevo_error, digits10_d);
    if(real_error_real_levo_tebd > max_tevo_error) throw except::logic_error("real_error_real_levo_tebd {0:.{2}f} > {1:.{2}f}", real_error_real_levo_tebd, max_tevo_error, digits10_d);
    if(real_error_real_revo_tebd > max_tevo_error) throw except::logic_error("real_error_real_revo_tebd {0:.{2}f} > {1:.{2}f}", real_error_real_revo_tebd, max_tevo_error, digits10_d);
    if(real_error_real_back_init > max_tevo_error) throw except::logic_error("real_error_real_back_init {0:.{2}f} > {1:.{2}f}", real_error_real_back_init, max_tevo_error, digits10_d);
    if(imag_error_lbit_levo_tebd > max_tevo_error) throw except::logic_error("imag_error_lbit_levo_tebd {0:.{2}f} > {1:.{2}f}", imag_error_lbit_levo_tebd, max_tevo_error, digits10_d);
    if(imag_error_lbit_revo_tebd > max_tevo_error) throw except::logic_error("imag_error_lbit_revo_tebd {0:.{2}f} > {1:.{2}f}", imag_error_lbit_revo_tebd, max_tevo_error, digits10_d);
    if(imag_error_lbit_devo_init > max_tevo_error) throw except::logic_error("imag_error_lbit_devo_init {0:.{2}f} > {1:.{2}f}", imag_error_lbit_devo_init, max_tevo_error, digits10_d);
    if(imag_error_real_levo_tebd > max_tevo_error) throw except::logic_error("imag_error_real_levo_tebd {0:.{2}f} > {1:.{2}f}", imag_error_real_levo_tebd, max_tevo_error, digits10_d);
    if(imag_error_real_revo_tebd > max_tevo_error) throw except::logic_error("imag_error_real_revo_tebd {0:.{2}f} > {1:.{2}f}", imag_error_real_revo_tebd, max_tevo_error, digits10_d);
    if(imag_error_real_back_init > max_tevo_error) throw except::logic_error("imag_error_real_back_init {0:.{2}f} > {1:.{2}f}", imag_error_real_back_init, max_tevo_error, digits10_d);
    if(norm_error_lbit_levo_tebd > max_tevo_error) throw except::logic_error("norm_error_lbit_levo_tebd {0:.{2}f} > {1:.{2}f}", norm_error_lbit_levo_tebd, max_tevo_error, digits10_d);
    if(norm_error_lbit_revo_tebd > max_tevo_error) throw except::logic_error("norm_error_lbit_revo_tebd {0:.{2}f} > {1:.{2}f}", norm_error_lbit_revo_tebd, max_tevo_error, digits10_d);
    if(norm_error_lbit_devo_init > max_tevo_error) throw except::logic_error("norm_error_lbit_devo_init {0:.{2}f} > {1:.{2}f}", norm_error_lbit_devo_init, max_tevo_error, digits10_d);
    if(norm_error_real_levo_tebd > max_tevo_error) throw except::logic_error("norm_error_real_levo_tebd {0:.{2}f} > {1:.{2}f}", norm_error_real_levo_tebd, max_tevo_error, digits10_d);
    if(norm_error_real_revo_tebd > max_tevo_error) throw except::logic_error("norm_error_real_revo_tebd {0:.{2}f} > {1:.{2}f}", norm_error_real_revo_tebd, max_tevo_error, digits10_d);
    if(norm_error_real_back_init > max_untary_err) throw except::logic_error("norm_error_real_back_init {0:.{2}f} > {1:.{2}f}", norm_error_real_back_init, max_untary_err, digits10_d);
    /* clang-format on */

    return 1;
}

size_t assert_lbit_evolution(const flbit &f1, const flbit &f2) {
    auto overlap_real_init = tools::finite::ops::overlap(*f1.state_real_init, *f2.state_real_init);
    auto overlap_lbit_init = tools::finite::ops::overlap(*f1.state_lbit_init, *f2.state_lbit_init);
    auto overlap_fp128evo  = tools::finite::ops::overlap(*f1.tensors.state, *f2.tensors.state);
    auto overlap_lbit_tevo = tools::finite::ops::overlap(*f1.state_lbit, *f2.state_lbit);
    tools::log->info("overlap_real_init : {:.16f} | {:.16f} |", overlap_real_init, std::abs(overlap_real_init));
    tools::log->info("overlap_lbit_init : {:.16f} | {:.16f} |", overlap_lbit_init, std::abs(overlap_lbit_init));
    tools::log->info("overlap_fp128evo : {:.16f} | {:.16f} |", overlap_fp128evo, std::abs(overlap_fp128evo));
    tools::log->info("overlap_lbit_tevo : {:.16f} | {:.16f} |", overlap_lbit_tevo, std::abs(overlap_lbit_tevo));
    auto real_error_real_init = std::abs(std::real(overlap_real_init) - 1.0);
    auto real_error_lbit_init = std::abs(std::real(overlap_lbit_init) - 1.0);
    auto real_error_fp128evo  = std::abs(std::real(overlap_fp128evo) - 1.0);
    auto real_error_lbit_tevo = std::abs(std::real(overlap_lbit_tevo) - 1.0);
    auto imag_error_real_init = std::abs(std::imag(overlap_real_init) - 0.0);
    auto imag_error_lbit_init = std::abs(std::imag(overlap_lbit_init) - 0.0);
    auto imag_error_fp128evo  = std::abs(std::imag(overlap_fp128evo) - 0.0);
    auto imag_error_lbit_tevo = std::abs(std::imag(overlap_lbit_tevo) - 0.0);

    auto   digits10_d     = std::numeric_limits<fp64>::digits10;
    auto   epsilon_d      = std::numeric_limits<fp64>::epsilon();
    auto   precision_d    = std::max({
        epsilon_d,
        std::pow(f1.status.trnc_lim, 2.0),
        std::pow(f2.status.trnc_lim, 2.0),
    });
    double max_tevo_error = 100 * precision_d *
                            std::max({
                                1.0,
                                static_cast<fp64>(abs(f1.status.delta_t.to_floating_point<cx128>())),
                                static_cast<fp64>(abs(f2.status.delta_t.to_floating_point<cx128>())),
                            });
    /* clang-format off */
    if(real_error_real_init > max_tevo_error) throw except::logic_error("real_error_real_init {0:.{2}f} > {1:.{2}f}", real_error_real_init, max_tevo_error,digits10_d);
    if(real_error_lbit_init > max_tevo_error) throw except::logic_error("real_error_lbit_init {0:.{2}f} > {1:.{2}f}", real_error_lbit_init, max_tevo_error,digits10_d);
    if(real_error_fp128evo > max_tevo_error) throw except::logic_error("real_error_fp128evo {0:.{2}f} > {1:.{2}f}", real_error_fp128evo, max_tevo_error,digits10_d);
    if(real_error_lbit_tevo > max_tevo_error) throw except::logic_error("real_error_lbit_tevo {0:.{2}f} > {1:.{2}f}", real_error_lbit_tevo, max_tevo_error,digits10_d);
    if(imag_error_real_init > max_tevo_error) throw except::logic_error("imag_error_real_init {0:.{2}f} > {1:.{2}f}", imag_error_real_init, max_tevo_error,digits10_d);
    if(imag_error_lbit_init > max_tevo_error) throw except::logic_error("imag_error_lbit_init {0:.{2}f} > {1:.{2}f}", imag_error_lbit_init, max_tevo_error,digits10_d);
    if(imag_error_fp128evo > max_tevo_error) throw except::logic_error("imag_error_fp128evo {0:.{2}f} > {1:.{2}f}", imag_error_fp128evo, max_tevo_error,digits10_d);
    if(imag_error_lbit_tevo > max_tevo_error) throw except::logic_error("imag_error_lbit_tevo {0:.{2}f} > {1:.{2}f}", imag_error_lbit_tevo, max_tevo_error,digits10_d);
    /* clang-format on */

    tools::log->info("f1: Sₑ(L/2) {0:.{2}f} | Sₙ {1:.{2}f}", tools::finite::measure::entanglement_entropy_midchain(*f1.tensors.state),
                     tools::finite::measure::number_entropy_midchain(*f1.tensors.state), digits10_d);
    tools::log->info("f2: Sₑ(L/2) {0:.{2}f} | Sₙ {1:.{2}f}", tools::finite::measure::entanglement_entropy_midchain(*f2.tensors.state),
                     tools::finite::measure::number_entropy_midchain(*f2.tensors.state), digits10_d);
    return 1;
}

void set_log(const std::string &name) { tools::log = tools::Logger::setLogger(name, settings::console::loglevel, settings::console::timestamp); }

int main(int argc, char *argv[]) {
    settings::parse(argc, argv);

    tools::log = tools::Logger::setLogger("lbit-evoution-test", settings::console::loglevel, settings::console::timestamp);

    // Set up the number of openmp and std threads for Eigen Tensor
    settings::configure_threads();

    // Seed with random::device initially (This also takes care of srand used by Eigen)
    // This is to make reproducible simulations
    rnd::seed(settings::input::seed);

    // Register termination codes and what to do in those cases
    debug::register_callbacks();
    tools::log->info("cplx            : {:.4f}", std::complex<double>(1.0, -1.0));
    tools::log->info("fp64     epsilon: {:.4e}", std::numeric_limits<double>::epsilon());
    tools::log->info("fp64     max_dig: {}", std::numeric_limits<double>::max_digits10);
    tools::log->info("ldouble  max_dig: {}", std::numeric_limits<long double>::max_digits10);
#if defined(DMRG_USE_QUADMATH)
    tools::log->info("fp128 epsilon: {}", f128_t(1.926e-34)); // The true def of FLT128_EPSILON has a literal extension "Q" which gives a warning
    tools::log->info("fp128 max_dig: {}", FLT128_DIG);
#elif defined(DMRG_USE_FLOAT128)
    tools::log->info("fp128 epsilon: {:.4e}", std::numeric_limits<fp128>::epsilon());
    tools::log->info(" __float128 max_dig: {}", std::numeric_limits<fp128>::max_digits10());
#endif
    // Initialize the flbit algorithm
    auto flbit_swap = flbit(nullptr); // Will evolve with swap gates on
    auto flbit_slow = flbit(nullptr); // Will evolve with swap gates off

    settings::flbit::use_swap_gates = true;
    set_log("swap");
    flbit_swap.run_preprocessing();

    settings::flbit::use_swap_gates = false;
    set_log("slow");
    flbit_slow.run_preprocessing();

    // Copy the model, unitaries and initial states
    flbit_slow.tensors                    = flbit_swap.tensors;
    *flbit_slow.state_real_init           = *flbit_swap.state_real_init;
    flbit_slow.unitary_gates_2site_layers = flbit_swap.unitary_gates_2site_layers;
    flbit_slow.create_hamiltonian_gates();
    flbit_slow.update_time_evolution_gates();
    flbit_slow.transform_to_lbit_basis();

    size_t test_swap_count = 0;
    size_t test_slow_count = 0;
    size_t test_both_count = 0;

    while(flbit_swap.status.algo_stop == AlgorithmStop::NONE and flbit_slow.status.algo_stop == AlgorithmStop::NONE) {
        set_log("swap");
        flbit_swap.update_state();
        flbit_swap.print_status();
        test_swap_count += assert_lbit_evolution(flbit_swap);
        flbit_swap.update_time_evolution_gates();

        set_log("slow");
        flbit_slow.update_state();
        flbit_slow.print_status();
        test_slow_count += assert_lbit_evolution(flbit_slow);
        flbit_slow.update_time_evolution_gates();

        set_log("both");
        test_both_count += assert_lbit_evolution(flbit_slow, flbit_swap);
    }
    if(test_swap_count == 0) { throw except::logic_error("No swap tests ran. Count: {}", test_swap_count); }
    if(test_slow_count == 0) { throw except::logic_error("No slow tests ran. Count: {}", test_slow_count); }
    if(test_both_count == 0) { throw except::logic_error("No both tests ran. Count: {}", test_both_count); }
    if(test_swap_count != test_slow_count) { throw except::logic_error("Unequal number of tests ran: {} != {}", test_swap_count, test_slow_count); }
    if(test_swap_count != test_both_count) { throw except::logic_error("Unequal number of tests ran: {} != {}", test_swap_count, test_both_count); }
    tools::log->info("Success!");
    return 0;
}
