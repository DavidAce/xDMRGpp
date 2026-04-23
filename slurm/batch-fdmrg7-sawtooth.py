from utils.generators import get_config_product, write_config_file, write_batch_files, move_directories
from utils.fdmrg import get_output_filepath, get_config_filename, update_batch_status
from batches_fdmrg import get_fdmrg_batch_setup
import os
import platform

config_paths = {
    'config_template'   : 'template_configs/fdmrg-ising-majorana.cfg',
    'output_prfx'       : "/mnt/WDB-AN1500/mbl_transition",
    'output_stem'       : 'mbl',
    'config_dir'        : "config",
    'output_dir'        : "output",
    'status_dir'        : "status",
    'temp_dir'          : "/scratch/local" if "lith" in platform.node() else (os.environ.get('PDC_TMP') if "PDC_TMP" in os.environ else "/tmp")
}

config_ranges = {
    "filename" : [''],
    "storage::output_filepath": [get_output_filepath],
    "storage::resume_policy": ['IF_UNSUCCESSFUL'],
    "storage::file_collision_policy": ['REVIVE'],
    "storage::temp_dir": [config_paths['temp_dir']],
    "storage::mps::state_emin::policy": ["NONE"],
    "storage::table::opdm::policy": ["FINISH|RBDS|RTES"],
    "storage::table::opdm_spectrum::policy": ["FINISH|RBDS|RTES"],
    "storage::dataset::subsystem_entanglement_entropies::bond_lim": ["2048"],
    "storage::dataset::subsystem_entanglement_entropies::trnc_lim": ["1e-6"],
    "console::loglevel": ['2'],
    "solvers::svd::truncation_min": ['1e-9'],
    "solvers::svd::truncation_max": ['1e-9'],
    "solvers::svd::switchsize_bdc": ['16'],
    "convergence::variance_threshold": ['1e-12'],
    "state::init::initial_state": ["PRODUCT_STATE_NEEL"],
    "schedule::dmrg::blocksize_policy": ["MAX|IF_STK_ALGO|ON_UPDATE"],
    "schedule::dmrg::min_blocksize": ["1"],
    "schedule::dmrg::max_blocksize": ["8"],
    "model::use_parity_shifted_mpo": ["true"],
    "model::use_parity_shifted_mpo_squared": ["true"],
    "model::model_type": ['ising_majorana'],
    "model::model_size": ['15'],
    "model::ising_majorana::g": ['0.500'],
    "model::ising_majorana::delta": ['+9.00'],
    "fdmrg::ritz": ['SR'],
    "fdmrg::iter_max": ['30'],
    "fdmrg::iter_min": ['1'],
    "fdmrg::warmup_iters": ['2'],
    "fdmrg::bond_max": ['2048'],
    "fdmrg::bond_init": ['16'],
    "fdmrg::print_freq": ['1'],
    "storage::dataset::statevector::policy": ["NONE"],
}

configs = get_config_product(config_ranges, config_paths)
for config in configs:
    # Set up the config file
    config['filename'] = get_config_filename(config, config_ranges, config_paths)
    config['template'] = config_paths['config_template']

batch_setup = get_fdmrg_batch_setup('fdmrg7-sawtooth')
write_batch_files(batch_setup=batch_setup, configs=configs, config_paths=config_paths)
update_batch_status(config_paths=config_paths)
move_directories(batch_setup=batch_setup, config_paths=config_paths)