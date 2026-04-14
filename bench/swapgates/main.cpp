#define ANKERL_NANOBENCH_IMPLEMENT
#include "algorithms/AlgorithmStatus.h"
#include "config/parse.h"
#include "config/settings.h"
#include "debug/info.h"
#include "env/environment.h"
#include "math/tenx/threads.h"
#include "nanobench.h"
#include "qm/gate.h"
#include "tensors/state/StateFinite.h"
#include "tid/tid.h"
#include "tools/common/h5.h"
#include "tools/common/prof.h"
#include "tools/finite/h5.h"
#include "tools/finite/mps.h"
#include <cstdlib>
#include <fmt/core.h>
#include <h5pp/h5pp.h>
#include <string_view>
#include <unsupported/Eigen/CXX11/Tensor>
int main(int argc, char *argv[]) {
    auto filepath = fmt::format("{}/swapgates.h5", BENCH_DATA_DIR);
    if(not h5pp::fs::exists(filepath)) {
        fmt::print("File does not exist: {}", filepath);
        exit(0);
    }
    settings::parse(argc, argv);
    std::atexit(debug::print_mem_usage);
    std::atexit(tools::common::timer::print_timers);
    std::at_quick_exit(debug::print_mem_usage);
    std::at_quick_exit(tools::common::timer::print_timers);
    settings::timer::level = tid::highest;
    auto t_bench           = tid::tic_scope("fLBIT");

    using cplx = std::complex<double>;
    using op_t = Eigen::Tensor<cplx, 2>;
    fmt::print("Compiler flags {}\n", env::build::compiler_flags);

    auto                      h5svd = h5pp::File(filepath, h5pp::FileAccess::READONLY);
    StateFinite<cx64>         state;
    AlgorithmStatus           status;
    std::vector<qm::SwapGate> gates;

    for(const auto &iter_prefix : h5svd.findGroups("iter_10", "fLBIT")) {
        auto state_prefix = fmt::format("fLBIT/{}", iter_prefix);
        tools::finite::h5::load::MpsInfo info;
        tools::common::h5::load::status(h5svd, state_prefix, status);
        tools::finite::h5::load::state(h5svd, state_prefix, state, info);
        auto gate_indices = h5svd.findGroups("gate_", state_prefix);
        gates.resize(gate_indices.size());
        for(const auto &gate_index : gate_indices) {
            auto  gate_prefix = fmt::format("{}/{}", state_prefix, gate_index);
            auto  idx         = h5svd.readAttribute<size_t>(gate_prefix, "idx");
            auto &gate        = gates[idx];
            gate.op           = h5svd.readDataset<op_t>(gate_prefix + "/op");
            auto swaps        = h5svd.readDataset<std::vector<qm::Swap>>(gate_prefix + "/swaps");
            auto rwaps        = h5svd.readDataset<std::vector<qm::Rwap>>(gate_prefix + "/rwaps");
            gate.swaps        = std::deque<qm::Swap>(swaps.begin(), swaps.end());
            gate.rwaps        = std::deque<qm::Rwap>(rwaps.begin(), rwaps.end());
            gate.pos          = h5svd.readAttribute<std::vector<size_t>>(gate_prefix, "pos");
            gate.dim          = h5svd.readAttribute<std::vector<long>>(gate_prefix, "dim");
        }
        auto svdset      = svd::config();
        svdset.benchmark = true;
        auto bench       = ankerl::nanobench::Bench().title(fmt::format("Swap Gates | {}", state_prefix)).relative(true).minEpochIterations(1);
        auto switchsizes = std::vector<size_t>{16};
        auto truncations = std::vector<double>{1e-8};
        auto num_threads = std::vector<int>{1};
        for(const auto &num_thread : num_threads) {
            for(const auto &switchsize : switchsizes) {
                for(const auto &truncation : truncations) {
                    svdset.switchsize_gesdd = switchsize;
                    svdset.truncation_limit = truncation;
                    omp_set_num_threads(num_thread);
                    tenx::threads::setNumThreads(num_thread);
                    auto name =
                        fmt::format("threads {} | switchsize {} | truncation {:.1e} | bond_lim {}", num_thread, switchsize, truncation, status.bond_lim);
                    bench.name(name).run([&]() {
                        auto state_tmp = state;
                        auto gates_tmp = gates;
                        tools::finite::mps::apply_swap_gates(state_tmp, gates_tmp, CircuitOp::NONE, GateMove::ON, svdset);
                        ankerl::nanobench::doNotOptimizeAway(state_tmp);
                        ankerl::nanobench::doNotOptimizeAway(gates_tmp);
                    });
                }
            }
        }
    }
}
