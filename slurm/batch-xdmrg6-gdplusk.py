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
    "strategy::initial_state": ["PRODUCT_STATE_NEEL_SHUFFLED"],
    "model::model_type": ['ising_majorana'],
    "model::model_size": ['12', '14', '16'],
    "model::ising_majorana::g": ['0.500'],
    "model::ising_majorana::delta": ['-8.00', '-7.50' '-7.00', '-6.50', '-6.00', '-5.50', '-5.00', '-4.50', '-4.00', '-3.50', '-3.00', '-2.50', '-2.00', '-1.50', '-1.00', '-0.50', '+0.00',
                                     '+0.50', '+1.00', '+1.50', '+2.00', '+2.50', '+3.00', '+3.50', '+4.00', '+4.50', '+5.00', '+5.50', '+6.00', '+6.50', '+7.00', '+7.50','+8.00'],
    "xdmrg::energy_spectrum_shift": ['0.0'],
    "xdmrg::iter_max": ['600'],
}

configs = get_config_product(config_ranges, config_paths)
for config in configs:
    # Set up the config file
    config['filename'] = get_config_filename(config, config_ranges, config_paths)
    config['template'] = config_paths['config_template']

batch_setup = get_xdmrg_batch_setup('xdmrg6-gdplusk')
write_batch_files(batch_setup=batch_setup, configs=configs, config_paths=config_paths)
update_batch_status(config_paths=config_paths)
move_directories(batch_setup=batch_setup, config_paths=config_paths)