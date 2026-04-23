from generate_inputs.src.generate_inputs import *
import numpy as np
from fractions import Fraction
from itertools import product
import platform

# Using the input_template.cfg, make num_copies new files enumerated as 0....num_copies,
# while replacing the fields stated in find_replace.

template_filename = 'input_template_xdmrg.cfg'
basename    = 'mbl'
location    = "input"


sites               = np.array([10])
lambdas             = [0.5]
deltas              = np.arange(-8,8+0.5,0.5)
initial_state       = ["RANDOM_PRODUCT_STATE"]
initial_sector      = "z"
target_sector       = "z"
multisite_mps_size_def  = [1]
multisite_mps_size_max  = [2]
output_prefix       = "output"

tmp_storage = "/tmp"
if "lith" in  platform.node():
    tmp_storage = "/scratch/local"

num_total = 0
settings = []
input_filenames = []

def delta(J,h):
    return np.log(J) - np.log(h)

def undelta(delta):
    J = np.min(1.0, np.exp(delta))
    h = np.min(1.0, np.exp(-delta))
    return J,h

print("Generating", len(sites) * len(lambdas) * len(deltas) * len(multisite_mps_size_def) * len(initial_state), "input files")

for val_L,val_l, val_d, init, multi in  product(sites,lambdas,deltas,initial_state,multisite_mps_size_max):
    val_j,val_h = undelta(val_d)
    str_L = str(val_L)
    str_d = "{:+.4f}".format(val_d)
    str_l = "{:.4f}".format(val_l)
    str_j = "{:.6f}".format(val_j)
    str_h = "{:.6f}".format(val_h)

    extra_prefix       = ""
    if len(initial_state) > 1:
        if init == "RANDOM_PRODUCT_STATE":
            extra_prefix = extra_prefix + "_rps"
        else:
            extra_prefix = extra_prefix + "_res"

    if len(multisite_mps_size_def) > 1:
            extra_prefix = extra_prefix + "_multi" + str(multi)

    input_filename = "{}/{}_L{}_l{}_d{}.cfg".format(location+extra_prefix,basename,str_L,str_l,str_d)
    settings = {
        "storage::output_filepath"            : "{}/L_{}/l_{}/d_{}/{}.h5".format(output_prefix+extra_prefix,str_L,str_l,str_d, basename),
        "storage::temp_dir"                  : tmp_storage,
        "threading::stl_threads"             : "2",
        "threading::omp_threads"             : "2",
        "console::loglevel"                  : "2",
        "model::model_size"                  : str_L,
        "model::ising_sdual::delta"          : str_d,
        "model::ising_sdual::lambda"         : str_l,
        "xdmrg::bond_max"                    : "64",
        "xdmrg::max_states"                  : "1",
        "schedule::dmrg::min_blocksize"   : str(multisite_mps_size_def[0]),
        "schedule::dmrg::max_blocksize"   : str(multi),
        "state::init::initial_state"            : str(init),
        "state::init::initial_axis"   : str(initial_sector),
        "state::sector::target_axis"  : str(target_sector),
    }
    os.makedirs(location + extra_prefix, exist_ok=True)
    num_total = num_total + 1
    print(input_filename, "L:", str_L,"l:",str_l, "d:", str_d,"j:", str_j, "h:",str_h)
    generate_input_file(settings, input_filename, template_filename)
