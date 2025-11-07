from glob import glob
import os
import sys
here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, '..'))
sys.path.insert(0, os.path.join(here, '../..'))
sys.path.insert(0, os.path.join(here, ''))
from dmrg_plot.common.io.h5ops import *
from dmrg_plot.common.io.parse import parse
from stats.write_statistics import *

def lbit_avg(args):
    projdir = args.basedir
    batchdirs = []
    for batch in args.batches:
        batchdirs.extend(glob('{}/{}*'.format(projdir, batch)))

    for batchdir in batchdirs:
        if 'test' in batchdir:
            print(f'Skipping: {batchdir}')
            continue
        print(f'Averaging: {batchdir}')
        analysisdir = batchdir + '/analysis'
        plotdir = analysisdir + '/plots'
        datadir = analysisdir + '/data'

        src = datadir + '/merged.h5'
        tgt = datadir + '/averaged-hartley-1e-8.h5'
        if os.path.isfile(tgt):
            if args.clear:
                print("Removing file: {}".format(tgt))
                os.remove(tgt)
            else:
                continue
        h5_tgt = h5open(tgt, 'a')
        h5close(h5_tgt)

        data_props = {
            'dsets': {  # For time independent data (or at the last time step)
                'schmidt_midchain': {},
                'model/hamiltonian': {'copy': True, },
                'model/opdm-eigv': {'copy': True, },
                'model/unitary_circuit': {'copy': True, },
                'model/model_size': {'copy': True, },
                'model/lbits/cls_avg_fit': {'axis': 0, },
                'model/lbits/cls_avg_rms': {'axis': 0, },
                'model/lbits/cls_avg_rsq': {'axis': 0, },
                'model/lbits/cls_typ_fit': {'axis': 0, },
                'model/lbits/cls_typ_rms': {'axis': 0, },
                'model/lbits/cls_typ_rsq': {'axis': 0, },
                'model/lbits/corravg': {'axis': 0, },
                'model/lbits/corrtyp': {'axis': 0, },
                'model/lbits/correrr': {'axis': 0, },
                'model/lbits/decay_avg': {'axis': 0, },
                'model/lbits/decay_err': {'axis': 0, },
                'model/lbits/corrmat': {
                    'copy': False,
                },
                'model/lbits/corroff': {
                    'copy': False,
                },
                'model/lbits/data': {
                    'copy': False,
                },
                'model/lbits/data_shifted': {
                    'copy': False,
                },
                'number_probabilities': {
                    'maxrealizations': None, # None is infinite
                    'maxbatchsize': 5000, # None is infinite
                    'copy': False,  # Copy the dataset as is, without averaging
                    'qin_probability' : False, # Compute (and store) q_i(n), the nth particle probability at each site
                    'pos_expvalue' : False, # Compute the position expectation value of each particle
                    'pos_expvalue_davg' : False, # Compute also the disorder average of pos_expvals
                    'pos_variance' : False, # Compute the position standard deviation of each particle
                    'pos_variance_davg' : False, # Compute also the disorder average of pos_stddevs
                    'hartley': True,  # Compute the hartley number entropy
                    'renyi2': False, # Compute the second renyi entropy of the number probabilities
                    'pinfty': False, # Compute the shannon entropy of the saturation value of the number probabilities
                },
            },

            'tables': {  # For data at the last time step
                'status': ['iter',
                           'bond_lim', 'bond_max',
                           'phys_time', 'algo_time', 'delta_t'],
                # 'mem_usage': 'ALL',
                'memory': 'ALL',
                'bond_dimensions': 'ALL',
                'truncation_errors': 'ALL',
                # If the following tables exist, save it under <tablename>/data
                '__save_data__': ['mem_usage'],
            },

            'cronos':  # For data at each time step
                {
                    'measurements': ['iter',
                                     'entanglement_entropy',
                                     'number_entropy',
                                     'renyi_entropy_2',
                                     'bond_mid', 'bond_lim',
                                     'truncation_error',
                                     'algorithm_time',
                                     'physical_time'],
                    # 'bond_dims': 'ALL',
                    # 'bond_dimensions': 'ALL',
                    'entanglement_entropies': 'ALL',
                    'number_entropies': 'ALL',
                    # 'truncation_errors': 'ALL',
                    # 'number_probabilities': 'ALL',
                    # If a table with this name exists, save its midchain column to a new dataset with the same name
                    '__save_mid__': ['entanglement_entropies', 'number_entropies'],
                    # If a table has this column, save all columns to a new dataset with the same name
                    '__save_col__': ['entanglement_entropy', 'number_entropy', 'bond_mid', 'algorithm_time'],
                },

        }

        write_statistics(src, tgt, data_props)


if __name__ == '__main__':
    # args = parse('fLBIT', ['lbit113'],)# basedir='/mnt/wdpool/backup/lbit')
    # args = parse('fLBIT', ['lbit114'],)# basedir='/mnt/wdpool/backup/lbit')
    args = parse('fLBIT', ['lbit93'])
    # args = parse('fLBIT', ['lbit115'])
    # args = parse('fLBIT', ['lbit117'])
    # args = parse('fLBIT', ['lbit118'])
    # args = parse('fLBIT', ['lbit119'])
    # args = parse('fLBIT', ['lbit120'])
    # args = parse('fLBIT', ['lbit121'], basedir='/mnt/S990PRO/mbl_transition')
    # args = parse('fLBIT', ['lbit122'])
    # args = parse('fLBIT', ['lbit123'])
    # args = parse('fLBIT', ['lbit124'])
    # args = parse('fLBIT', ['lbit125'])
    # args = parse('fLBIT', ['lbit126'])
    # args = parse('fLBIT', ['lbit127'], basedir='/mnt/S990PRO/mbl_transition')
    # args = parse('fLBIT', ['lbit128'], basedir='/mnt/S990PRO/mbl_transition')

    # args = parse('fLBIT', ['lbit106'], )#basedir='/mnt/wdpool/backup/lbit')
    # args = parse('fLBIT', ['lbit106'], basedir='/mnt/wdpool/backup/lbit')
    args.clear = True
    lbit_avg(args)
