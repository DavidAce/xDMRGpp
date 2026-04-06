#include <config/settings.h>
#include <config/threading.h>
#include <h5pp/h5pp.h>
#include <math/tenx/threads.h>
#include <omp.h>
#include <tensors/model/ModelFinite.h>
#include <tensors/site/mpo/MpoSite.h>
#include <tensors/state/StateFinite.h>
#include <tools/common/h5.h>
#include <tools/common/log.h>
#include <tools/finite/h5.h>
#include <tools/finite/measure/expectation_value.h>

struct PertData {
    long   seed               = 0;
    long   model_size         = 0;
    double delta              = 0.0;
    double delta_avg          = 0.0;
    double g                  = 0;
    double g_pert             = 0;
    double energy             = 0;
    double energy_pert        = 0;
    double energy_upper_bound = 0;
    double variance           = 0;
    double variance_pert      = 0;
};
struct spins {
    double x, y, z;
};

int main() {
    using scalar_t = cx64;
    tools::log                       = tools::Logger::setLogger("pert", 2, true);
    settings::threading::num_threads = 8; // omp_get_max_threads();
    settings::configure_threads();
    settings::precision::use_compressed_mpo_squared = MpoCompress::DPL;
    // std::vector    g_perts                          = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};
    std::vector    g_perts = {1.00000000e+00, 7.94328235e-01, 6.30957344e-01, 5.01187234e-01, 3.98107171e-01, 3.16227766e-01, 2.51188643e-01,
                              1.99526231e-01, 1.58489319e-01, 1.25892541e-01, 1.00000000e-01, 7.94328235e-02, 6.30957344e-02, 5.01187234e-02,
                              3.98107171e-02, 3.16227766e-02, 2.51188643e-02, 1.99526231e-02, 1.58489319e-02, 1.25892541e-02, 1.00000000e-02,
                              7.94328235e-03, 6.30957344e-03, 5.01187234e-03, 3.98107171e-03, 3.16227766e-03, 2.51188643e-03, 1.99526231e-03,
                              1.58489319e-03, 1.25892541e-03, 1.00000000e-03, 7.94328235e-04, 6.30957344e-04, 5.01187234e-04, 3.98107171e-04,
                              3.16227766e-04, 2.51188643e-04, 1.99526231e-04, 1.58489319e-04, 1.25892541e-04, 1.00000000e-04};
    h5pp::fs::path h5dir   = "/mnt/WDB-AN1500/mbl_transition/xdmrg3-letsgo/output/L48/g0.000";
    auto           h5tgt   = h5pp::File("../../../../output/pert-L48-new.h5", h5pp::FileAccess::REPLACE);
    h5pp::hid::h5t h5_type = H5Tcreate(H5T_COMPOUND, sizeof(PertData));
    H5Tinsert(h5_type, "seed", HOFFSET(PertData, seed), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "model_size", HOFFSET(PertData, model_size), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "delta", HOFFSET(PertData, delta), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "delta_avg", HOFFSET(PertData, delta_avg), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "g", HOFFSET(PertData, g), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "g_pert", HOFFSET(PertData, g_pert), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy", HOFFSET(PertData, energy), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_pert", HOFFSET(PertData, energy_pert), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "energy_upper_bound", HOFFSET(PertData, energy_upper_bound), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "variance", HOFFSET(PertData, variance), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "variance_pert", HOFFSET(PertData, variance_pert), H5T_NATIVE_DOUBLE);
    // Collect and sort all the files in h5dir
    using h5iter = h5pp::fs::recursive_directory_iterator;
    // long counter = 0;
    h5tgt.setKeepFileOpened();

    for(const auto &file : h5iter(h5dir)) {
        if(not file.is_regular_file()) continue;
        if(file.path().extension() != ".h5") continue;

        // tools::log->info("file: {}\n", file.path().string());
        auto h5src = h5pp::File(file, h5pp::FileAccess::READONLY);
        try {
            auto success = h5src.readAttribute<bool>("xDMRG/state_emid", "algorithm_has_succeeded") == 1;
            if(not success) { continue; }
        } catch(const std::exception &err) {
            tools::log->warn("Could not read file: {}: {}", file.path().string(), err.what());
            continue;
        }
        h5src.setKeepFileOpened();
        auto ene_old = h5src.readTableField<double>("xDMRG/state_emid/measurements", "energy", h5pp::TableSelection::LAST);
        auto var_old = h5src.readTableField<double>("xDMRG/state_emid/measurements", "energy_variance", h5pp::TableSelection::LAST);
        // if(var_old > 1e-12) continue;

        auto spinZ = h5src.readTableField<spins>("xDMRG/state_emid/measurements", "spin_global", h5pp::TableSelection::LAST).z;

        auto seed            = h5src.readDataset<long>("common/seed");
        auto model_size      = h5src.readAttribute<size_t>("xDMRG/model/hamiltonian", "model_size");
        auto model_type      = h5src.readAttribute<ModelType>("xDMRG/model/hamiltonian", "model_type");
        auto position        = h5src.readAttribute<long>("xDMRG/state_emid/mps", "position");
        auto model_g         = h5src.readTableField<double>("xDMRG/model/hamiltonian", "g", h5pp::TableSelection::FIRST);
        auto model_delta     = h5src.readTableField<double>("xDMRG/model/hamiltonian", "delta", h5pp::TableSelection::FIRST);
        auto model_hrand     = h5src.readTableField<Eigen::VectorXd>("xDMRG/model/hamiltonian", "h_rand", 0, model_size);
        auto model_Jrand     = h5src.readTableField<Eigen::VectorXd>("xDMRG/model/hamiltonian", "J_rand", 0, model_size - 1);
        auto model_delta_avg = std::log(model_Jrand.mean()) - std::log(model_hrand.mean());

        settings::model::model_size            = model_size;
        settings::model::model_type            = model_type;
        settings::model::ising_majorana::g     = model_g;
        settings::model::ising_majorana::delta = model_delta;

        tools::log->set_level(spdlog::level::warn);
        auto model = ModelFinite<scalar_t>(ModelType::ising_majorana, model_size);
        tools::finite::h5::load::model(h5src, AlgorithmType::xDMRG, model);
        auto ene_upbd = model.get_energy_upper_bound();

        auto mpsinfo = tools::finite::h5::load::MpsInfo();
        auto state   = StateFinite<scalar_t>(AlgorithmType::xDMRG, model_size, position);
        tools::finite::h5::load::state(h5src, "xDMRG/state_emid", state, mpsinfo);
        tools::log->set_level(spdlog::level::info);

        // auto expH_old    = tools::finite::measure::expectation_value(state, state, mposEne_old);

        // auto mposEne_old = model.get_all_mpo_tensors(MposWithEdges::ON);
        // auto mposVar_old = model.get_compressed_mpos_squared(MposWithEdges::ON);

        // auto expH2_old   = tools::finite::measure::expectation_value(state, state, mposVar_old);
        // auto var_old     = std::abs(std::abs(expH2_old) - std::abs(expH_old * expH_old));
        // auto var_file    = h5src.readTableField<double>("xDMRG/state_emid/measurements", "energy_variance", h5pp::TableSelection::LAST);

        // tools::log->info("Old H {:<20.16f} H² {:.3e} | var {:.3e}", std::real(ene_old), std::real(var_old), std::real(var_old) - std::real(ene_old *
        // ene_old));
        #pragma omp parallel for shared(h5src) schedule(dynamic)
        for(size_t gidx = 0; gidx < g_perts.size(); ++gidx) {
            auto g_perturb = g_perts[gidx];
            auto model_pert = model;
            auto state_pert = state;
            for(auto &mpo : model_pert.MPO) { mpo->set_parameter("g", g_perturb); }
            model_pert.build_mpo();
            auto mposEne_pert = model_pert.get_mpo_tensors(0.0, MposWithEdges::ON);
            auto ene_pert     = tools::finite::measure::expectation_value<scalar_t>(state_pert, state_pert, mposEne_pert);

            for(const auto &mpo : model_pert.MPO) mpo->set_energy_shift_mpo(cx64(std::real(ene_pert) / static_cast<double>(model_size), 0.0));
            for(auto &mpo : model_pert.MPO) { mpo->set_parity_shift_mpo_squared(spinZ > 0 ? 1 : -1, "z"); }

            model_pert.clear_cache();
            model_pert.build_mpo();
            model_pert.build_mpo_squared();

            auto mposVar_pert = model_pert.get_mpo2_tensors(0.0, MposWithEdges::ON);
            auto var_pert =
                std::abs(tools::finite::measure::expectation_value<scalar_t>(state_pert, state_pert, mposVar_pert)); // Has reduced energy so this is immediately the variance

            tools::log->info("omp {:2} seed {} | delta {:.3e} -> {:.3e} | energy {:.3e} -> {:.3e} | variance {:.3e} -> {:.3e} | bound {:.3f}",omp_get_thread_num(),  seed, model_delta,
                             model_delta_avg, ene_old, ene_pert, var_old, var_pert, ene_upbd);

            auto tablename = fmt::format("L{}-d{:.2f}-gpert{:.1e}", model_size, model_delta, g_perturb);

            #pragma omp critical
            {
                auto linkExists = h5tgt.linkExists(tablename);
                if(not linkExists) {
                    h5tgt.createTable(h5_type, tablename, "PertData", 100, 3);
                }
            }

            auto pert               = PertData();
            pert.seed               = seed;
            pert.model_size         = model_size;
            pert.delta              = model_delta;
            pert.delta_avg          = model_delta_avg;
            pert.g                  = model_g;   // The perturbation
            pert.g_pert             = g_perturb; // The perturbation
            pert.energy             = std::real(ene_old);
            pert.energy_pert        = std::real(ene_pert);
            pert.energy_upper_bound = ene_upbd;
            pert.variance           = std::real(var_old);
            pert.variance_pert      = std::real(var_pert);
            #pragma omp critical
            {
                h5tgt.appendTableRecords(pert, tablename);
            }
            //

        }
        h5src.setKeepFileClosed();
        // counter++;
        // if(counter > 100) exit(0);
    }
    h5tgt.setKeepFileClosed();
    return 0;
}
