#include "opdm.h"
#include "expectation_value.h"
#include "math/eig.h"
#include "math/float.h"
#include "math/num.h"
#include "math/tenx.h"
#include "qm/spin.h"
#include "tensors/state/StateFinite.h"
#include "tools/common/log.h"

Eigen::Tensor<cx64, 2> tools::finite::measure::opdm(const StateFinite &state) {
    /* We create a matrix of the form
     *
     *
     * R(i,j) =  | r++ r+- |
     *           | r-+ r-- |
     *
     * where
     *      r++ = r++(i,j) = ⟨sp(i) sz(i)sz(i+1)...sz(j-1) sm(j)⟩
     *      r+- = r+-(i,j) = ⟨sp(i) sz(i)sz(i+1)...sz(j-1) sp(j)⟩
     *      r-+ = r-+(i,j) = ⟨sm(i) sz(i)sz(i+1)...sz(j-1) sm(j)⟩
     *      r-- = r+-(i,j) = ⟨sm(i) sz(i)sz(i+1)...sz(j-1) sp(j)⟩
     *
     * and sp, sm, sz are the plus, minus and z pauli matrices at some site i.
     * Note that each r is an LxL matrix, so R must be a 2Lx2L matrix.
     * Note also that the signs are "crossed" intentionally: r++ has sp and sm, and so on.
     *
     */
    if(state.measurements.opdm) return state.measurements.opdm.value();
    tools::log->trace("Measuring the one-particle density matrix (OPDM)");
    long L   = state.get_length<long>();
    auto R   = Eigen::MatrixXcd(2 * L, 2 * L); // Allocate the full rho matrix "R"
    auto rpp = R.topLeftCorner(L, L);          // One quadrant of R: rho++
    auto rpm = R.topRightCorner(L, L);         // One quadrant of R: rho+-
    auto rmp = R.bottomLeftCorner(L, L);       // One quadrant of R: rho-+
    auto rmm = R.bottomRightCorner(L, L);      // One quadrant of R: rho--

    using namespace qm::spin::half::tensor;

    // Shorthand types
    using op_t       = LocalObservableOp<cx64>;
    using opstring_t = std::vector<op_t>;
    for(long pos_i = 0; pos_i < L; ++pos_i) {
        for(long pos_j = pos_i; pos_j < L; ++pos_j) {
            // Create an operator string from pos_i to pos_j, where
            //      pos_i has sp (or sm)
            //      pos_j has sm (or sp)
            // Then insert sz from pos_i (including) to pos_j (excluding).
            auto opp = opstring_t{op_t{sp, pos_i}}; // adds s+_i
            auto opm = opstring_t{op_t{sp, pos_i}}; // adds s+_i
            auto omp = opstring_t{op_t{sm, pos_i}}; // adds s-_i
            auto omm = opstring_t{op_t{sm, pos_i}}; // adds s-_i
            if(pos_i < pos_j) {
                for(auto pos_x : num::range(pos_i, pos_j)) {
                    opp.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    opm.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    omp.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                    omm.emplace_back(op_t{sz, pos_x}); // adds sz_i sz_{i} ... sz_{j-1}
                }
            }
            opp.emplace_back(op_t{sm, pos_j}); // adds  s-_j
            opm.emplace_back(op_t{sp, pos_j}); // adds  s+_j
            omp.emplace_back(op_t{sm, pos_j}); // adds  s-_j
            omm.emplace_back(op_t{sp, pos_j}); // adds  s+_j

            // Calculate the expectation value of the operator string
            rpp(pos_i, pos_j) = expectation_value<cx64>(state, opp);
            rpm(pos_i, pos_j) = expectation_value<cx64>(state, opm);
            rmp(pos_i, pos_j) = expectation_value<cx64>(state, omp);
            rmm(pos_i, pos_j) = expectation_value<cx64>(state, omm);

            // Set the Hermitian conjugates on the opposite side
            if(pos_i != pos_j) {
                rpp(pos_j, pos_i) = std::conj(rpp(pos_i, pos_j));
                rpm(pos_j, pos_i) = std::conj(rmp(pos_i, pos_j)); // Mix pm <-> mp
                rmp(pos_j, pos_i) = std::conj(rpm(pos_i, pos_j)); // Mix mp <-> pm
                rmm(pos_j, pos_i) = std::conj(rmm(pos_i, pos_j));
            }
        }
    }
    //    tools::log->info("rho++: trace {:.16f}\n{}", rpp.trace(), linalg::matrix::to_string(rpp, 8));
    //    tools::log->info("rho+-: trace {:.16f}\n{}", rpm.trace(), linalg::matrix::to_string(rpm, 8));
    //    tools::log->info("rho-+: trace {:.16f}\n{}", rmp.trace(), linalg::matrix::to_string(rmp, 8));
    //    tools::log->info("rho--: trace {:.16f}\n{}", rmm.trace(), linalg::matrix::to_string(rmm, 8));
    // tools::log->debug("R    : trace {:.16f}", R.trace());
    if(not R.isApprox(R.conjugate().transpose())) throw except::logic_error("R is not hermitian");
    if(std::abs(R.trace() - static_cast<double>(L)) > 1e-8) throw std::runtime_error("R.trace() != L");
    state.measurements.opdm = tenx::TensorMap(R);
    return state.measurements.opdm.value();
}

Eigen::Tensor<double, 1> tools::finite::measure::opdm_spectrum(const StateFinite &state) {
    if(not state.measurements.opdm) state.measurements.opdm = opdm(state);
    if(not state.measurements.opdm_spectrum) {
        auto &opdm   = state.measurements.opdm.value();
        auto  solver = eig::solver();
        solver.eig<eig::Form::SYMM>(opdm.data(), opdm.dimension(0), eig::Vecs::OFF);
        state.measurements.opdm_spectrum = tenx::TensorCast(eig::view::get_eigvals<double>(solver.result));
        // tools::log->debug("OPDM spectrum: {::+9.4e}", tenx::span(state.measurements.opdm_spectrum.value()));
    }
    return state.measurements.opdm_spectrum.value();
}
