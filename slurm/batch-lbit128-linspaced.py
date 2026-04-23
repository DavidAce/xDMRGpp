from utils.generators import get_config_product, write_config_file, write_batch_files, move_directories
from utils.flbit import get_max_time, get_max_steps, get_output_filepath, get_config_filename, update_batch_status
from batch_setup import get_batch_setup
import os
import platform

config_paths = {
    'config_template'   : 'template_configs/flbit-v2.cfg',
    'output_prfx'       : "/mnt/S990PRO/mbl_transition",
    'output_stem'       : 'mbl',
    'config_dir'        : "config-lbit128",
    'output_dir'        : "output-lbit128",
    'status_dir'        : "status-lbit128",
    'temp_dir'          : "/scratch/local" if "lith" in platform.node() else (os.environ.get('PDC_TMP') if "PDC_TMP" in os.environ else "/tmp")
}

config_ranges = {
    "filename" : [''],
    "storage::output_filepath": [get_output_filepath],
    "storage::temp_dir": [config_paths['temp_dir']],
    "storage::copy_from_temp_freq": ['100'],
    "storage::table::entanglement_entropies::policy": ['NONE'],
    "storage::table::number_entropies::policy" : ['NONE'],
    "storage::table::renyi_entropies::policy": ['NONE'],
    "storage::table::truncation_errors::policy": ['NONE'],
    "storage::table::bond_dimensions::policy": ['NONE'],
    "storage::dataset::number_probabilities::policy": ['NONE'],
    "console::loglevel": ['2'],
    "solvers::svd::truncation_min": ['1e-5'],
    "solvers::svd::truncation_max": ['1e-5'],
    "solver::svd_switchsize_bdc": ['16'],
    "solver::svd_save_fail": ['false'],
    "state::init::initial_state": ["PRODUCT_STATE_NEEL"],
    "model::model_type": ['lbit'],
    "model::model_size": ['12','16', '20', '24', '28', '32'],
    "model::lbit::J1_mean": ['+0.00'],
    "model::lbit::J2_mean": ['+0.00'],
    "model::lbit::J3_mean": ['+0.00'],
    "model::lbit::J1_wdth": ['1.00'],
    "model::lbit::J2_wdth": ['1.00'],
    "model::lbit::J3_wdth": ['1.00'],
    "model::lbit::J2_span": ['-1'],
    "model::lbit::xi_Jcls": ['1.00'],
    "model::lbit::u_depth": ['16'],
    "model::lbit::u_fmix": ['0.20'],
    "model::lbit::u_lambda": ['1.0'],
    "model::lbit::u_wkind": ['EXPDECAY'],
    "model::lbit::u_mkind": ['MATRIX_V2'],
    "flbit::run_effective_model": ['false'],
    "flbit::run_iter_in_parallel": ['true'],
    "flbit::bond_max": ['8192'],
    "flbit::max_iters": ['5000'],
    "flbit::time_scale": ['LINSPACED'],
    "flbit::time_start_real": ['10000000000'],  # 1e10
    "flbit::time_start_imag": ['0.0'],
    "flbit::time_final_real": ['10000000050'],  # 1e10+50,
    "flbit::time_final_imag": ['0.0'],
    "flbit::time_num_steps": ['201'],
    "flbit::cls::num_rnd_circuits": ['0'],
    "flbit::cls::mpo_circuit_svd_bondlim": ['20'],
    "flbit::cls::mpo_circuit_switchdepth": ['5'],
    "flbit::opdm::num_rps": ['0'],
}

configs = get_config_product(config_ranges, config_paths)
for config in configs:
    # Set up the config file
    config['filename'] = get_config_filename(config, config_ranges, config_paths)
    config['template'] = config_paths['config_template']

batch_setup = get_batch_setup('lbit128-linspaced')
write_batch_files(batch_setup=batch_setup, configs=configs, config_paths=config_paths)
update_batch_status(config_paths=config_paths)
move_directories(batch_setup=batch_setup, config_paths=config_paths)