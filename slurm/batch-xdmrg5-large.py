from utils.generators import get_config_product, write_config_file, write_batch_files, move_directories
from utils.xdmrg import get_output_filepath, get_config_filename, update_batch_status
from batches_xdmrg import get_xdmrg_batch_setup
import os
import platform

config_paths = {
    'config_template'   : 'template_configs/xdmrg-ising-majorana.cfg',
    'output_prfx'       : "/mnt/WDB-AN1500/mbl_transition",
    'output_stem'       : 'mbl',
    'config_dir'        : "config",
    'output_dir'        : "output",
    'status_dir'        : "status",
    'temp_dir'          : "/scratch/local" if "lith" in platform.node() else (os.environ.get('PDC_TMP') if "PDC_TMP" in os.environ else "/tmp")
}

config_ranges = {
    "filename" : [''],
    "console::loglevel": ['2'],
    "storage::output_filepath": [get_output_filepath],
    "storage::resume_policy": ['IF_UNSUCCESSFUL'],
    "storage::file_collision_policy": ['REVIVE'],
    "storage::temp_dir": [config_paths['temp_dir']],
    "storage::mps::state_emid::policy": ["ITER|FINISH|REPLACE"],
    "storage::table::opdm::policy": ["FINISH"],
    "storage::table::opdm_spectrum::policy": ["FINISH"],
    "storage::dataset::subsystem_entanglement_entropies::bits_err": ["1e-6"],
    "storage::dataset::subsystem_entanglement_entropies::eig_size": ["8192"],
    "storage::dataset::subsystem_entanglement_entropies::bond_lim": ["4096"],
    "storage::dataset::subsystem_entanglement_entropies::trnc_lim": ["1e-6"],
    "schedule::opt::iter_max_warmup": ['8'],
    "schedule::opt::iter_max_stuck": ['20'],
    "schedule::dmrg::blocksize_policy": ["ICOMPLUS1|EXP"],
    "schedule::dmrg::min_blocksize": ["1"],
    "schedule::dmrg::max_blocksize": ["8"],
    "schedule::dmrg::bond_expansion_policy": ["PREOPT_NSITE_FORE|H1|H2"],
    "state::init::initial_state": ["PRODUCT_STATE_NEEL"],
    "schedule::opt::bond_increase_when": ["SATURATED"],
    "schedule::opt::bond_increase_rate": ["2.0"],
    "schedule::opt::trnc_decrease_when": ["NEVER"],
    "schedule::opt::trnc_decrease_rate": ["0.25"],
    "solvers::eig::iter_min": ["1000"],
    "solvers::eig::iter_max": ["10000"],
    "solvers::eig::iter_gain": ["2.0"],
    "solvers::eig::iter_gain_policy": ["SAT_VAR"],
    "solvers::eig::jcb_blocksize_min": ["128"],
    "solvers::eig::jcb_blocksize_max": ["1024"],
    "solvers::svd::truncation_min": ['1e-8'],
    "solvers::svd::truncation_max": ['1e-8'],
    "solvers::svd::switchsize_bdc": ['16'],
    "convergence::variance_threshold": ['1e-13'],
    "model::use_parity_shifted_mpo": ["false"],
    "model::use_parity_shifted_mpo_squared": ["true"],
    "model::model_type": ['ising_majorana'],
    "model::model_size": ['18'],
    "model::ising_majorana::g": ['0.000', '0.005', '0.010','0.015', '0.020','0.025', '0.030'],
    "model::ising_majorana::delta": ['-6.00', '-5.00', '-4.00', '-3.00', '-2.00', '-1.00', '-0.75', '-0.50' , '-0.25', '+0.00', '+0.25',  '+0.50' ,  '+0.75', '+1.00', '+2.00', '+3.00', '+4.00', '+5.00', '+6.00'],
    "xdmrg::energy_spectrum_shift": ['1e-6'],
    "xdmrg::algo": ['DMRG_GSI'],
    "xdmrg::ritz": ['LM'],
    "xdmrg::algo_warmup": ['DMRG_FOLDED'],
    "xdmrg::ritz_warmup": ['SM'],
    "xdmrg::iter_min": ['10'],
    "xdmrg::iter_max": ['2000'],
    "xdmrg::bond_max": ['8192'],
    "xdmrg::bond_min": ['48'],
}

configs = get_config_product(config_ranges, config_paths)
for config in configs:
    # Set up the config file
    config['filename'] = get_config_filename(config, config_ranges, config_paths)
    config['template'] = config_paths['config_template']

batch_setup = get_xdmrg_batch_setup('xdmrg3-letsgo')
write_batch_files(batch_setup=batch_setup, configs=configs, config_paths=config_paths)
update_batch_status(config_paths=config_paths)
move_directories(batch_setup=batch_setup, config_paths=config_paths)
