#include "settings.h"
#include "debug/exceptions.h"
#include "loader.h"

bool settings::algorithm_is_on(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::on;
        case AlgorithmType::fDMRG: return settings::fdmrg::on;
        case AlgorithmType::xDMRG: return settings::xdmrg::on;
        case AlgorithmType::iTEBD: return settings::itebd::on;
        case AlgorithmType::fLBIT: return settings::flbit::on;
        default: return false;
    }
}
long settings::get_bond_min(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::bond_min;
        case AlgorithmType::fDMRG: return settings::fdmrg::bond_min;
        case AlgorithmType::xDMRG: return settings::xdmrg::bond_min;
        case AlgorithmType::iTEBD: return settings::itebd::bond_min;
        case AlgorithmType::fLBIT: return settings::flbit::bond_min;
        default: return 8;
    }
}
long settings::get_bond_max(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::bond_max;
        case AlgorithmType::fDMRG: return settings::fdmrg::bond_max;
        case AlgorithmType::xDMRG: return settings::xdmrg::bond_max;
        case AlgorithmType::iTEBD: return settings::itebd::bond_max;
        case AlgorithmType::fLBIT: return settings::flbit::bond_max;
        default: return 64;
    }
}

OptRitz settings::get_ritz(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::fDMRG: return settings::fdmrg::ritz;
        case AlgorithmType::xDMRG: return settings::xdmrg::ritz;
        default: throw except::logic_error("Ritz not defined for algorithm [{}]", enum2sv(algo_type));
    }
}

size_t settings::print_freq(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::print_freq;
        case AlgorithmType::fDMRG: return settings::fdmrg::print_freq;
        case AlgorithmType::xDMRG: return settings::xdmrg::print_freq;
        case AlgorithmType::iTEBD: return settings::itebd::print_freq;
        case AlgorithmType::fLBIT: return settings::flbit::print_freq;
        default: return 1;
    }
}

bool settings::store_wave_function(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return false;
        case AlgorithmType::fDMRG: return settings::fdmrg::store_wavefn;
        case AlgorithmType::xDMRG: return settings::xdmrg::store_wavefn;
        case AlgorithmType::iTEBD: return false;
        case AlgorithmType::fLBIT: return settings::flbit::store_wavefn;
        default: return false;
    }
}

extern size_t settings::get_iter_min(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::iter_min;
        case AlgorithmType::fDMRG: return settings::fdmrg::iter_min;
        case AlgorithmType::xDMRG: return settings::xdmrg::iter_min;
        case AlgorithmType::iTEBD: return settings::itebd::iter_min;
        case AlgorithmType::fLBIT: return settings::flbit::iter_min;
        default: return 0;
    }
}

extern size_t settings::get_iter_max(AlgorithmType algo_type) {
    switch(algo_type) {
        case AlgorithmType::iDMRG: return settings::idmrg::iter_max;
        case AlgorithmType::fDMRG: return settings::fdmrg::iter_max;
        case AlgorithmType::xDMRG: return settings::xdmrg::iter_max;
        case AlgorithmType::iTEBD: return settings::itebd::iter_max;
        case AlgorithmType::fLBIT: return settings::flbit::iter_max;
        default: return -1ul;
    }
}

/*
 * The section below attempts to find and read the parameters from the given inputfile.
 *
 */

void settings::load(Loader &dmrg_config) {
    if(not dmrg_config.file_exists) throw except::runtime_error("Could not load config [{}]: File does not exist", dmrg_config.file_path.string());
    dmrg_config.load();
    input::config_filename      = dmrg_config.file_path.string();
    input::config_file_contents = dmrg_config.get_config_file_as_string();
    /* clang-format off */
    dmrg_config.load_parameter("threading::num_threads"                       , threading::num_threads);

    dmrg_config.load_parameter("input::seed"                                  , input::seed);

    dmrg_config.load_parameter("storage::output_filepath"                                       , storage::output_filepath);
    dmrg_config.load_parameter("storage::output_append_seed"                                    , storage::output_append_seed);
    dmrg_config.load_parameter("storage::storage_interval"                                      , storage::storage_interval);
    dmrg_config.load_parameter("storage::use_temp_dir"                                          , storage::use_temp_dir);
    dmrg_config.load_parameter("storage::copy_from_temp_freq"                                   , storage::copy_from_temp_freq);
    dmrg_config.load_parameter("storage::temp_dir"                                              , storage::temp_dir);
    dmrg_config.load_parameter("storage::compression_level"                                     , storage::compression_level);
    dmrg_config.load_parameter("storage::resume_policy"                                         , storage::resume_policy);
    dmrg_config.load_parameter("storage::file_collision_policy"                                 , storage::file_collision_policy);
    dmrg_config.load_parameter("storage::file_resume_policy"                                    , storage::file_resume_policy);
    dmrg_config.load_parameter("storage::file_resume_name"                                      , storage::file_resume_name);
    dmrg_config.load_parameter("storage::file_resume_iter"                                      , storage::file_resume_iter);
    dmrg_config.load_parameter("storage::mps::state_emid::policy"                               , storage::mps::state_emid::policy);
    dmrg_config.load_parameter("storage::mps::state_emin::policy"                               , storage::mps::state_emin::policy);
    dmrg_config.load_parameter("storage::mps::state_emax::policy"                               , storage::mps::state_emax::policy);
    dmrg_config.load_parameter("storage::mps::state_real::policy"                               , storage::mps::state_real::policy);
    dmrg_config.load_parameter("storage::mps::state_lbit::policy"                               , storage::mps::state_lbit::policy);
    dmrg_config.load_parameter("storage::mpo::model::policy"                                    , storage::mpo::model::policy);
    dmrg_config.load_parameter("storage::table::bonds::policy"                                  , storage::table::bonds::policy);
    dmrg_config.load_parameter("storage::table::model::policy"                                  , storage::table::model::policy);
    dmrg_config.load_parameter("storage::table::measurements::policy"                           , storage::table::measurements::policy);
    dmrg_config.load_parameter("storage::table::status::policy"                                 , storage::table::status::policy);
    dmrg_config.load_parameter("storage::table::memory::policy"                                 , storage::table::memory::policy);
    dmrg_config.load_parameter("storage::table::timers::level"                                  , storage::table::timers::level);
    dmrg_config.load_parameter("storage::table::timers::policy"                                 , storage::table::timers::policy);
    dmrg_config.load_parameter("storage::table::entanglement_entropies::policy"                 , storage::table::entanglement_entropies::policy);
    dmrg_config.load_parameter("storage::table::truncation_errors::policy"                      , storage::table::truncation_errors::policy);
    dmrg_config.load_parameter("storage::table::bond_dimensions::policy"                        , storage::table::bond_dimensions::policy);
    dmrg_config.load_parameter("storage::table::number_entropies::policy"                       , storage::table::number_entropies::policy);
    dmrg_config.load_parameter("storage::table::renyi_entropies::policy"                        , storage::table::renyi_entropies::policy);
    dmrg_config.load_parameter("storage::table::opdm_spectrum::policy"                          , storage::table::opdm_spectrum::policy);
    dmrg_config.load_parameter("storage::table::expectation_values_spin_xyz::policy"            , storage::table::expectation_values_spin_xyz::policy);
    dmrg_config.load_parameter("storage::table::random_unitary_circuit::policy"                 , storage::table::random_unitary_circuit::policy);
    dmrg_config.load_parameter("storage::table::information_per_scale::policy"                  , storage::table::information_per_scale::policy);
    dmrg_config.load_parameter("storage::table::information_center_of_mass::policy"             , storage::table::information_center_of_mass::policy);
    dmrg_config.load_parameter("storage::dataset::opdm::policy"                                 , storage::dataset::opdm::policy);
    dmrg_config.load_parameter("storage::dataset::opdm::chunksize"                              , storage::dataset::opdm::chunksize);
    dmrg_config.load_parameter("storage::dataset::information_lattice::policy"                  , storage::dataset::information_lattice::policy);
    dmrg_config.load_parameter("storage::dataset::information_lattice::chunksize"               , storage::dataset::information_lattice::chunksize);
    dmrg_config.load_parameter("storage::dataset::lbit_analysis::policy"                        , storage::dataset::lbit_analysis::policy);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::policy"     , storage::dataset::subsystem_entanglement_entropies::policy);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::chunksize"  , storage::dataset::subsystem_entanglement_entropies::chunksize);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::bits_err"   , storage::dataset::subsystem_entanglement_entropies::bits_err);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::eig_size"   , storage::dataset::subsystem_entanglement_entropies::eig_size);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::bond_lim"   , storage::dataset::subsystem_entanglement_entropies::bond_lim);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::trnc_lim"   , storage::dataset::subsystem_entanglement_entropies::trnc_lim);
    dmrg_config.load_parameter("storage::dataset::subsystem_entanglement_entropies::precision"  , storage::dataset::subsystem_entanglement_entropies::precision);
    dmrg_config.load_parameter("storage::dataset::number_probabilities::policy"                 , storage::dataset::number_probabilities::policy);
    dmrg_config.load_parameter("storage::dataset::number_probabilities::chunksize"              , storage::dataset::number_probabilities::chunksize);
    dmrg_config.load_parameter("storage::dataset::expectation_values_spin_xyz::policy"          , storage::dataset::expectation_values_spin_xyz::policy);
    dmrg_config.load_parameter("storage::dataset::expectation_values_spin_xyz::chunksize"       , storage::dataset::expectation_values_spin_xyz::chunksize);
    dmrg_config.load_parameter("storage::dataset::correlation_matrix_spin_xyz::policy"          , storage::dataset::correlation_matrix_spin_xyz::policy);
    dmrg_config.load_parameter("storage::dataset::correlation_matrix_spin_xyz::chunksize"       , storage::dataset::correlation_matrix_spin_xyz::chunksize);


    dmrg_config.load_parameter("model::model_type"                            , model::model_type);
    dmrg_config.load_parameter("model::model_size"                            , model::model_size);
    dmrg_config.load_parameter("model::ising_tf_rf::J1"                       , model::ising_tf_rf::J1);
    dmrg_config.load_parameter("model::ising_tf_rf::J2"                       , model::ising_tf_rf::J2);
    dmrg_config.load_parameter("model::ising_tf_rf::h_tran"                   , model::ising_tf_rf::h_tran);
    dmrg_config.load_parameter("model::ising_tf_rf::h_mean"                   , model::ising_tf_rf::h_mean);
    dmrg_config.load_parameter("model::ising_tf_rf::h_wdth"                   , model::ising_tf_rf::h_wdth);
    dmrg_config.load_parameter("model::ising_tf_rf::spin_dim"                 , model::ising_tf_rf::spin_dim);
    dmrg_config.load_parameter("model::ising_tf_rf::distribution"             , model::ising_tf_rf::distribution);
    dmrg_config.load_parameter("model::ising_sdual::delta"                    , model::ising_sdual::delta);
    dmrg_config.load_parameter("model::ising_sdual::lambda"                   , model::ising_sdual::lambda);
    dmrg_config.load_parameter("model::ising_sdual::distribution"             , model::ising_sdual::distribution);
    dmrg_config.load_parameter("model::ising_sdual::spin_dim"                 , model::ising_sdual::spin_dim);
    dmrg_config.load_parameter("model::ising_majorana::delta"                 , model::ising_majorana::delta);
    dmrg_config.load_parameter("model::ising_majorana::g"                     , model::ising_majorana::g);
    dmrg_config.load_parameter("model::ising_majorana::distribution"          , model::ising_majorana::distribution);
    dmrg_config.load_parameter("model::ising_majorana::spin_dim"              , model::ising_majorana::spin_dim);
    dmrg_config.load_parameter("model::lbit::J1_mean"                         , model::lbit::J1_mean);
    dmrg_config.load_parameter("model::lbit::J2_mean"                         , model::lbit::J2_mean);
    dmrg_config.load_parameter("model::lbit::J3_mean"                         , model::lbit::J3_mean);
    dmrg_config.load_parameter("model::lbit::J1_wdth"                         , model::lbit::J1_wdth);
    dmrg_config.load_parameter("model::lbit::J2_wdth"                         , model::lbit::J2_wdth);
    dmrg_config.load_parameter("model::lbit::J3_wdth"                         , model::lbit::J3_wdth);
    dmrg_config.load_parameter("model::lbit::J2_span"                         , model::lbit::J2_span);
    dmrg_config.load_parameter("model::lbit::xi_Jcls"                         , model::lbit::xi_Jcls);
    dmrg_config.load_parameter("model::lbit::u_depth"                         , model::lbit::u_depth);
    dmrg_config.load_parameter("model::lbit::u_fmix"                          , model::lbit::u_fmix);
    dmrg_config.load_parameter("model::lbit::u_lambda"                        , model::lbit::u_lambda);
    dmrg_config.load_parameter("model::lbit::u_wkind"                         , model::lbit::u_wkind);
    dmrg_config.load_parameter("model::lbit::u_mkind"                         , model::lbit::u_mkind);
    dmrg_config.load_parameter("model::lbit::spin_dim"                        , model::lbit::spin_dim);
    dmrg_config.load_parameter("model::lbit::distribution"                    , model::lbit::distribution);

    dmrg_config.load_parameter("strategy::move_sites_when_stuck"                  , strategy::move_sites_when_stuck);
    dmrg_config.load_parameter("strategy::projection_policy"                      , strategy::projection_policy);
    dmrg_config.load_parameter("strategy::use_eigenspinors"                       , strategy::use_eigenspinors);
    dmrg_config.load_parameter("strategy::iter_max_warmup"                        , strategy::iter_max_warmup);
    dmrg_config.load_parameter("strategy::iter_max_stuck"                         , strategy::iter_max_stuck);
    dmrg_config.load_parameter("strategy::iter_max_saturated"                     , strategy::iter_max_saturated);
    dmrg_config.load_parameter("strategy::iter_min_converged"                     , strategy::iter_min_converged);
    dmrg_config.load_parameter("strategy::dmrg_blocksize_policy"                  , strategy::dmrg_blocksize_policy);
    dmrg_config.load_parameter("strategy::dmrg_min_blocksize"                     , strategy::dmrg_min_blocksize);
    dmrg_config.load_parameter("strategy::dmrg_max_blocksize"                     , strategy::dmrg_max_blocksize);
    dmrg_config.load_parameter("strategy::dmrg_max_prob_size"                     , strategy::dmrg_max_prob_size);
    dmrg_config.load_parameter("strategy::dmrg_bond_expansion_policy"             , strategy::dmrg_bond_expansion_policy);
    dmrg_config.load_parameter("strategy::dmrg_bond_expansion::postopt::maxalpha" , strategy::dmrg_bond_expansion::postopt::maxalpha);
    dmrg_config.load_parameter("strategy::dmrg_bond_expansion::postopt::minalpha" , strategy::dmrg_bond_expansion::postopt::minalpha);
    dmrg_config.load_parameter("strategy::dmrg_bond_expansion::preopt::maxiter"   , strategy::dmrg_bond_expansion::preopt::maxiter);
    dmrg_config.load_parameter("strategy::dmrg_bond_expansion::preopt::nkrylov"   , strategy::dmrg_bond_expansion::preopt::nkrylov);
    dmrg_config.load_parameter("strategy::target_axis"                            , strategy::target_axis);
    dmrg_config.load_parameter("strategy::initial_axis"                           , strategy::initial_axis);
    dmrg_config.load_parameter("strategy::initial_type"                           , strategy::initial_type);
    dmrg_config.load_parameter("strategy::initial_state"                          , strategy::initial_state);
    dmrg_config.load_parameter("strategy::initial_pattern"                        , strategy::initial_pattern);
    dmrg_config.load_parameter("strategy::rbds_rate"                              , strategy::rbds_rate);
    dmrg_config.load_parameter("strategy::bond_increase_when"                     , strategy::bond_increase_when);
    dmrg_config.load_parameter("strategy::bond_increase_rate"                     , strategy::bond_increase_rate);
    dmrg_config.load_parameter("strategy::trnc_decrease_when"                     , strategy::trnc_decrease_when);
    dmrg_config.load_parameter("strategy::trnc_decrease_rate"                     , strategy::trnc_decrease_rate);
    dmrg_config.load_parameter("strategy::etol_decrease_when"                     , strategy::etol_decrease_when);
    dmrg_config.load_parameter("strategy::etol_decrease_rate"                     , strategy::etol_decrease_rate);

    dmrg_config.load_parameter("precision::algoScalar"                        , precision::algoScalar);
    dmrg_config.load_parameter("precision::optScalar"                         , precision::optScalar);
    dmrg_config.load_parameter("precision::eig_max_size"                      , precision::eig_max_size);
    dmrg_config.load_parameter("precision::eigs_iter_min"                     , precision::eigs_iter_min);
    dmrg_config.load_parameter("precision::eigs_iter_max"                     , precision::eigs_iter_max);
    dmrg_config.load_parameter("precision::eigs_iter_gain"                    , precision::eigs_iter_gain);
    dmrg_config.load_parameter("precision::eigs_iter_gain_policy"             , precision::eigs_iter_gain_policy);
    dmrg_config.load_parameter("precision::eigs_tol_min"                      , precision::eigs_tol_min);
    dmrg_config.load_parameter("precision::eigs_tol_max"                      , precision::eigs_tol_max);
    dmrg_config.load_parameter("precision::eigs_ncv"                          , precision::eigs_ncv);
    dmrg_config.load_parameter("precision::eigs_nev_min"                      , precision::eigs_nev_min);
    dmrg_config.load_parameter("precision::eigs_nev_max"                      , precision::eigs_nev_max);
    dmrg_config.load_parameter("precision::eigs_max_size_shift_invert"        , precision::eigs_max_size_shift_invert);
    dmrg_config.load_parameter("precision::eigs_jcb_min_blocksize"            , precision::eigs_jcb_min_blocksize);
    dmrg_config.load_parameter("precision::eigs_jcb_max_blocksize"            , precision::eigs_jcb_max_blocksize);
    dmrg_config.load_parameter("precision::svd_truncation_min"                , precision::svd_truncation_min);
    dmrg_config.load_parameter("precision::svd_truncation_max"                , precision::svd_truncation_max);
    dmrg_config.load_parameter("precision::svd_switchsize_bdc"                , precision::svd_switchsize_bdc);
    dmrg_config.load_parameter("precision::svd_save_fail"                     , precision::svd_save_fail);
    dmrg_config.load_parameter("precision::use_compressed_mpo"                , precision::use_compressed_mpo);
    dmrg_config.load_parameter("precision::use_compressed_mpo_squared"        , precision::use_compressed_mpo_squared);
    dmrg_config.load_parameter("precision::use_energy_shifted_mpo"            , precision::use_energy_shifted_mpo);
    dmrg_config.load_parameter("precision::use_parity_shifted_mpo"            , precision::use_parity_shifted_mpo);
    dmrg_config.load_parameter("precision::use_parity_shifted_mpo_squared"    , precision::use_parity_shifted_mpo_squared);
    dmrg_config.load_parameter("precision::variance_convergence_threshold"    , precision::variance_convergence_threshold);
    dmrg_config.load_parameter("precision::variance_saturation_sensitivity"   , precision::variance_saturation_sensitivity);
    dmrg_config.load_parameter("precision::entropy_saturation_sensitivity"    , precision::entropy_saturation_sensitivity);
    dmrg_config.load_parameter("precision::infocom_saturation_sensitivity"    , precision::infocom_saturation_sensitivity);
    dmrg_config.load_parameter("precision::target_subspace_error"             , precision::target_subspace_error);
    dmrg_config.load_parameter("precision::max_subspace_size"                 , precision::max_subspace_size);
    dmrg_config.load_parameter("precision::max_norm_error"                    , precision::max_norm_error);
    dmrg_config.load_parameter("precision::max_cache_gbts"                    , precision::max_cache_gbts);



    //Parameters controlling infinite-DMRG
    dmrg_config.load_parameter("idmrg::on"                                    , idmrg::on);
    dmrg_config.load_parameter("idmrg::iter_max"                              , idmrg::iter_max);
    dmrg_config.load_parameter("idmrg::bond_max"                              , idmrg::bond_max);
    dmrg_config.load_parameter("idmrg::bond_min"                              , idmrg::bond_min);
    dmrg_config.load_parameter("idmrg::print_freq"                            , idmrg::print_freq);


    //Parameters controlling imaginary TEBD (Zero temperature)
    dmrg_config.load_parameter("itebd::on"                                    , itebd::on       );
    dmrg_config.load_parameter("itebd::iter_max"                              , itebd::iter_max);
    dmrg_config.load_parameter("itebd::time_step_init_real"                   , itebd::time_step_init_real  );
    dmrg_config.load_parameter("itebd::time_step_init_imag"                   , itebd::time_step_init_imag  );
    dmrg_config.load_parameter("itebd::time_step_min"                         , itebd::time_step_min);
    dmrg_config.load_parameter("itebd::suzuki_order"                          , itebd::suzuki_order);
    dmrg_config.load_parameter("itebd::bond_max"                              , itebd::bond_max  );
    dmrg_config.load_parameter("itebd::bond_min"                              , itebd::bond_min);
    dmrg_config.load_parameter("itebd::print_freq"                            , itebd::print_freq);


    //Parameters controlling finite-DMRG
    dmrg_config.load_parameter("fdmrg::on"                                    , fdmrg::on);
    dmrg_config.load_parameter("fdmrg::ritz"                                  , fdmrg::ritz);
    dmrg_config.load_parameter("fdmrg::iter_max"                              , fdmrg::iter_max);
    dmrg_config.load_parameter("fdmrg::iter_min"                              , fdmrg::iter_min);
    dmrg_config.load_parameter("fdmrg::bond_max"                              , fdmrg::bond_max);
    dmrg_config.load_parameter("fdmrg::bond_min"                              , fdmrg::bond_min);
    dmrg_config.load_parameter("fdmrg::print_freq "                           , fdmrg::print_freq);
    dmrg_config.load_parameter("fdmrg::store_wavefn"                          , fdmrg::store_wavefn);

    //Parameters controlling finite-LBIT
    dmrg_config.load_parameter("flbit::on"                                    , flbit::on);
    dmrg_config.load_parameter("flbit::run_iter_in_parallel"                  , flbit::run_iter_in_parallel);
    dmrg_config.load_parameter("flbit::run_effective_model"                   , flbit::run_effective_model);
    dmrg_config.load_parameter("flbit::iter_max"                              , flbit::iter_max);
    dmrg_config.load_parameter("flbit::iter_min"                              , flbit::iter_min);
    dmrg_config.load_parameter("flbit::use_swap_gates"                        , flbit::use_swap_gates);
    dmrg_config.load_parameter("flbit::use_mpo_circuit"                       , flbit::use_mpo_circuit);
    dmrg_config.load_parameter("flbit::bond_max"                              , flbit::bond_max);
    dmrg_config.load_parameter("flbit::bond_min"                              , flbit::bond_min);
    dmrg_config.load_parameter("flbit::time_scale"                            , flbit::time_scale);
    dmrg_config.load_parameter("flbit::time_start_real"                       , flbit::time_start_real);
    dmrg_config.load_parameter("flbit::time_start_imag"                       , flbit::time_start_imag);
    dmrg_config.load_parameter("flbit::time_final_real"                       , flbit::time_final_real);
    dmrg_config.load_parameter("flbit::time_final_imag"                       , flbit::time_final_imag);
    dmrg_config.load_parameter("flbit::time_num_steps"                        , flbit::time_num_steps);
    dmrg_config.load_parameter("flbit::print_freq"                            , flbit::print_freq);
    dmrg_config.load_parameter("flbit::store_wavefn"                          , flbit::store_wavefn);
    dmrg_config.load_parameter("flbit::cls::num_rnd_circuits"                 , flbit::cls::num_rnd_circuits);
    dmrg_config.load_parameter("flbit::cls::exit_when_done"                   , flbit::cls::exit_when_done);
    dmrg_config.load_parameter("flbit::cls::randomize_hfields"                , flbit::cls::randomize_hfields);
    dmrg_config.load_parameter("flbit::cls::mpo_circuit_switchdepth"          , flbit::cls::mpo_circuit_switchdepth);
    dmrg_config.load_parameter("flbit::cls::mpo_circuit_svd_bondlim"          , flbit::cls::mpo_circuit_svd_bondlim);
    dmrg_config.load_parameter("flbit::cls::mpo_circuit_svd_trnclim"          , flbit::cls::mpo_circuit_svd_trnclim);
    dmrg_config.load_parameter("flbit::opdm::num_rps"                         , flbit::opdm::num_rps);
    dmrg_config.load_parameter("flbit::opdm::exit_when_done"                  , flbit::opdm::exit_when_done);



    //Parameters controlling excited state DMRG
    dmrg_config.load_parameter("xdmrg::on"                                    , xdmrg::on);
    dmrg_config.load_parameter("xdmrg::algo"                                  , xdmrg::algo);
    dmrg_config.load_parameter("xdmrg::ritz"                                  , xdmrg::ritz);
    dmrg_config.load_parameter("xdmrg::algo_warmup"                           , xdmrg::algo_warmup);
    dmrg_config.load_parameter("xdmrg::ritz_warmup"                           , xdmrg::ritz_warmup);
    dmrg_config.load_parameter("xdmrg::algo_stuck"                            , xdmrg::algo_stuck);
    dmrg_config.load_parameter("xdmrg::ritz_stuck"                            , xdmrg::ritz_stuck);
    dmrg_config.load_parameter("xdmrg::energy_spectrum_shift"                 , xdmrg::energy_spectrum_shift);
    dmrg_config.load_parameter("xdmrg::energy_density_target"                 , xdmrg::energy_density_target);
    dmrg_config.load_parameter("xdmrg::iter_max"                              , xdmrg::iter_max);
    dmrg_config.load_parameter("xdmrg::iter_min"                              , xdmrg::iter_min);
    dmrg_config.load_parameter("xdmrg::bond_max"                              , xdmrg::bond_max);
    dmrg_config.load_parameter("xdmrg::bond_min"                              , xdmrg::bond_min);
    dmrg_config.load_parameter("xdmrg::print_freq "                           , xdmrg::print_freq);
    dmrg_config.load_parameter("xdmrg::store_wavefn"                          , xdmrg::store_wavefn);
    dmrg_config.load_parameter("xdmrg::max_states"                            , xdmrg::max_states);

    //Timers
    dmrg_config.load_parameter("timer::level"      , timer::level     );
    //Console settings
    dmrg_config.load_parameter("console::loglevel"    , console::loglevel);
    dmrg_config.load_parameter("console::logh5pp"     , console::logh5pp);
    dmrg_config.load_parameter("console::timestamp"   , console::timestamp);
    /* clang-format off */
}



void settings::load(std::string_view  config_filename){
    Loader indata(config_filename);
    settings::load(indata);
}
