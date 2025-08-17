import os
import traceback
import h5py
import numpy as np
from .common import *


def get_unique_config_string(d: dict, dl: dict, delim: str):
    str_L = str(d['model::model_size'])
    if d['model::model_type'] == "ising_majorana":
        str_g = str(d['model::ising_majorana::g'])
        str_d = str(d['model::ising_majorana::delta'])
        return f"L{str_L}{delim}g{str_g}{delim}d{str_d}"


def get_config_filename(d: dict, dl: dict, p: dict):
    unique_path = get_unique_config_string(d, dl, '_')
    return f"{p['config_dir']}/{p['output_stem']}_{unique_path}.cfg"


def get_output_filepath(d: dict, dl: dict, p: dict):
    unique_path = get_unique_config_string(d, dl, '/')
    return f"{p['output_dir']}/{unique_path}/{p['output_stem']}.h5"



def get_num_entries(dsets, expected_num, kind=256):  # 256 is the enum value for ITER_STATE
    indices = {}
    skip = ['/common/finished_all',  #
            '/fLBIT/state_real/initial_pattern',  #
            '/fLBIT/state_real/number_probabilities'
            ]
    for dset in dsets:
        # Check if there is an 'iter' field
        if dset is not None:
            if dset.name in skip:
                indices[dset.name] = None
            elif dset.dtype.fields is not None and 'event' in dset.dtype.fields.keys():
                # idx = get_entry_indices(dset)
                # It can happen that some entries are repeated (due to the old way of resuming simulations)
                indices[dset.name] = len(get_entry_indices(dset, expected_num, kind))
            else:
                indices[dset.name] = np.shape(dset)[-1]
    return indices


def get_h5_status(filename, batch):
    if os.path.isfile(filename):
        config = f'{batch["config_file"]}'
        model_type = batch["model_type"]
        if model_type != 'ising_majorana':
            raise AssertionError(f"Expected model_type==ising_majorana. Got: {model_type}")

        expected_dset_paths = [
            '/common/finished_all',
            '/xDMRG/model/hamiltonian',
            '/xDMRG/state_emid/measurements',
            '/xDMRG/state_emid/status',
            '/xDMRG/state_emid/subsystem_entanglement_entropies',

        ]
        expected_link_attrs = {
            'initial_pattern': '/xDMRG/state_emid',
            'initial_state':   '/xDMRG/state_emid',
        }

        try:
            with h5py.File(filename, 'r') as h5file:
                if h5file['/common/finished_all'][()] == 0:
                    return f"FAILED|simulation has not finished"
                expected_dsets = [h5file.get(path) for path in expected_dset_paths]
                expected_attrs = [h5file.get(link).attrs.get(attr) for attr,link in expected_link_attrs.items() if link in h5file]
                missing_dsets = [path for dset,path in zip(expected_dsets,expected_dset_paths) if dset is None]
                missing_attrs = [path for link,path in zip(expected_attrs,expected_link_attrs) if link is None]
                if len(missing_dsets) > 0:
                    return f"FAILED|missing datasets:{missing_dsets}"
                if len(missing_dsets) > 0:
                    return f"FAILED|missing attributes:{missing_attrs}"
                enum_event = h5py.check_enum_dtype(h5file['xDMRG/state_emid/status'].dtype['event'])  # key value pairs defining the enum
                enum_algo_stop = h5py.check_enum_dtype(h5file['xDMRG/state_emid/status'].dtype['algo_stop'])  # key value pairs defining the enum

                # We also check if the simulation saturated
                if algorithm_stop := h5file['/xDMRG/state_emid'].attrs.get('algorithm_stop'): # This is a string!
                    if algorithm_stop != 'SUCCESS':

                        for evar, (int_event, prec_limit, int_algo_stop) in zip(
                                h5file['xDMRG/state_emid/measurements']['energy_variance'][::-1],
                                h5file['xDMRG/state_emid/status']['event', 'energy_variance_prec_limit', 'algo_stop'][
                                ::-1]):
                            if int_event != enum_event['FINISHED']:
                                continue
                            if evar < 100*prec_limit:
                                return f"FINISHED"
                            else:
                                return f"FAILED|algorithm_stop:{algorithm_stop}:variance:{evar:.5e}[limit:{prec_limit:.5e}]"
                else:
                    # We need to check the status table manually
                    # Find a finished event
                    for evar, (int_event, prec_limit, int_algo_stop) in zip(h5file['xDMRG/state_emid/measurements']['energy_variance'][::-1],
                                                                h5file['xDMRG/state_emid/status']['event', 'energy_variance_prec_limit', 'algo_stop'][::-1]):
                        if int_event != enum_event['FINISHED']:
                            continue
                        if evar <= prec_limit:
                            return "FINISHED"
                        if int_algo_stop == enum_algo_stop['SUCCESS']:
                            # Some realizations that "succeeded" actually have poor variance because they exited too soon.
                            # We need to check the variances
                            if evar > 10*prec_limit:
                                return f"FAILED|variance is too high:{evar:.5e}[limit:{prec_limit:.5e}]"

                        if int_algo_stop != enum_algo_stop['SUCCESS']:
                            str_algo_stop = list(enum_algo_stop.keys())[list(enum_algo_stop.values()).index(int_algo_stop)]
                            return f"FAILED|algorithm_stop:{str_algo_stop}"

                return "FINISHED"
        except Exception as e:
            print(traceback.format_exc())
            return f"FAILED|{e}"

    else:
        return "MISSING"


def write_batch_status(batch_filename):
    parser = argparse.ArgumentParser(description='SLURM batch generator')
    parser.add_argument('--update-status', action='store_true', help='Update status files', default=False)
    args = parser.parse_args()
    with open(batch_filename, 'r') as fp:
        batch = json.load(fp)
        output_prfx = f'{batch["output_prfx"]}'
        config_file = f'{batch["config_file"]}'
        status_file = f'{batch["status_dir"]}/{Path(config_file).stem}.status'
        seed_status = copy(batch.get('seed_status'))  # Old one
        batch['seed_status'] = []
        if platform.node() != "neumann" and args.update_status:
            raise AssertionError("--update-status is only valid on neumann")

        if platform.node() == "neumann" and args.update_status:
            Path(status_file).parent.mkdir(parents=True, exist_ok=True)
            output_path = f'{output_prfx}/{batch["projectname"]}/{batch["output_path"]}'
            print(f"Updating status: {status_file}")
            with open(status_file, 'w') as sf:
                status_count = 0
                # First we check if there are any stray h5 files that have finished simulations
                # This could happen for example if we reduced the batch size after we already ran some on the cluster.
                # In that case we get these seeds for free.
                num_seeds = 0
                for h5file in sorted(Path(output_path).rglob('*.h5')):
                    if 'save' in h5file.name:
                        continue
                    seed = int(re.split('_|\.', h5file.name)[1])
                    # Collect if it has finished
                    filename = f'{output_path}/{batch["output_stem"]}_{seed}.h5'
                    h5status = get_h5_status(filename=filename, batch=batch)
                    if "FINISHED" in h5status:
                        print(f'{config_file}: {seed}|{h5status}')
                        sf.write(f'{seed}|{h5status}\n')
                        num_seeds += 1
                # Now add missing and failed seeds
                for sidx, (offset, extent) in enumerate(zip(batch['seed_offset'], batch['seed_extent'])):
                    extent_size = len(batch['seed_extent'])
                    offset_size = len(batch['seed_offset'])
                    if offset_size != extent_size:
                        raise ValueError(
                            f"offset:{offset_size} and extent:{extent_size} are not equal lengths")

                    ## Start checking the h5 files
                    is_finished = True
                    for seed in range(offset, offset + extent):
                        if num_seeds >= extent:
                            break
                        h5status = None
                        if seed_status is not None:
                            if seed_status[sidx] == "FINISHED":
                                h5status = seed_status[sidx]  # Short circuit
                        if h5status is None:
                            filename = f'{output_path}/{batch["output_stem"]}_{seed}.h5'
                            h5status = get_h5_status(filename=filename, batch=batch)
                        is_finished = "FINISHED" in h5status
                        if is_finished:
                            # We already included this
                            continue
                        num_seeds += 1
                        print(f'{config_file}: {seed}|{h5status}')
                        sf.write(f'{seed}|{h5status}\n')


                    if is_finished:
                        batch['seed_status'].append("FINISHED")
                    else:
                        batch['seed_status'].append("PENDING")
            # We can now check if there are stray simulations
            strays_file = f'strays/{Path(config_file).stem}.strays'
            Path(strays_file).parent.mkdir(parents=True, exist_ok=True)
            with open(status_file, 'r') as sf, open(strays_file, 'w') as st:
                for h5file in sorted(Path(output_path).rglob('*.h5')):
                    if 'save' in h5file.name:
                        continue
                    seed_found = int(re.split('_|\.', h5file.name)[1])
                    seed_stray = True
                    for offset, extent in zip(batch['seed_offset'], batch['seed_extent']):
                        if offset <= seed_found < offset + extent:
                            seed_stray = False
                    if seed_stray is True:
                        print(f"Stray seed found for {batch['config_file']}: {seed_found}")
                        st.write(f'{h5file}\n')
        else:
            if platform.node() == "neumann":
                status_file = "{}/{}/{}".format(batch['output_prfx'], batch['projectname'], status_file)
            else:
                if not os.path.isfile(config_file):
                    print(f'Config file not found: {config_file}')
                    return batch
                rclone.copyto(f'neumann:{batch["output_prfx"]}/{batch["projectname"]}/{status_file}',
                              f'{status_file}',
                              args=['-L', '--update'])
            if not os.path.isfile(status_file):
                raise FileNotFoundError(f"{status_file}\n Hint: Perhaps you need to add --update on the first run")

            status_count = 0
            for offset, extent in zip(batch['seed_offset'], batch['seed_extent']):
                extent_size = len(batch['seed_extent'])
                offset_size = len(batch['seed_offset'])
                if offset_size != extent_size:
                    raise ValueError(
                        f"offset:{offset_size} and extent:{extent_size} are not equal lengths")
                is_finished = True
                for idx, seed in enumerate(range(offset, offset + extent)):
                    sfline = linecache.getline(status_file, idx + status_count + 1).rstrip()
                    # print(idx, seed, sfline)
                    sfseed, sfstatus = sfline.split('|', maxsplit=1)
                    if seed != int(sfseed):
                        print(f'WARNING: seed mismatch [{seed=}] != [{sfseed=}]')
                    if sfstatus != "FINISHED":
                        is_finished = False
                        break
                    # print(linecache.getline(status_file, idx+status_count))
                status_count += extent
                if is_finished:
                    batch['seed_status'].append('FINISHED')
                else:
                    batch['seed_status'].append('PENDING')

        return batch


def update_batch_status(config_paths):
    batch_filenames = sorted(Path(config_paths['config_dir']).rglob('*.json'))
    with Pool() as pool:
        batch_jsons = pool.map(write_batch_status, batch_filenames)
        for batch_filename, batch_json in zip(batch_filenames, batch_jsons):
            with open(batch_filename, 'w') as fp:
                json.dump(batch_json, fp, sort_keys=True, indent=4)
