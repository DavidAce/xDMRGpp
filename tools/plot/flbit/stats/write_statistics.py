from itertools import islice
import numpy as np
import scipy.stats as sc
from plotting.tools import get_timepoints
from .write_statistics_table import *
from numba import njit,prange, float64,int64
# from line_profiler_pycharm import profile
from cProfile import Profile
from pstats import SortKey, Stats

def get_value(key_candidates, node_candidates, optional = True, default=None):
    if isinstance(key_candidates, str):
        keys = [key_candidates]
    elif isinstance(key_candidates, list):
        keys = key_candidates
    else:
        raise TypeError(f'key_candidates must be str or list: got {type(key_candidates)}: {key_candidates}')

    for key in keys:
        for node in node_candidates:
            if node is None:
                continue
            if key in node.attrs:
                return node.attrs[key][()]
            if isinstance(node, h5py.Dataset):
                if key in node.dtype.fields:
                    return node[key][0] # Just the first element
            if isinstance(node, h5py.Group):
                if key in node:
                    val = node[key][()]
                    if isinstance(val, np.ndarray):
                        return val[0]
                    return val
    if not optional:
        raise LookupError(f'could not find keys {keys} among node candidates: [{node_candidates}]')
    return default


def write_stats_to_node(data, tgt_node, axis):
    print(f'writing stat to node: {tgt_node.name}')
    std = np.nanstd(data, axis=axis)
    num = np.shape(data)[axis]
    dama = np.ma.masked_invalid(np.ma.masked_equal(np.abs(data), 0))
    dtyp = np.exp(np.nanmean(np.log(np.abs(data)), axis=axis))

    if '_typ' in tgt_node.name:
        if np.any(np.isnan(dama)):
            print('dama has nans: \n{}', dama)
        avg=tgt_node.create_dataset(name='avg', data=dtyp)
        typ=tgt_node.create_dataset(name='typ', data=dtyp)
    else:
        tgt_node.create_dataset(name='avg', data=np.nanmean(data, axis=axis))
        tgt_node.create_dataset(name='typ', data=dtyp)

    std_dset = tgt_node.create_dataset(name='std', data=std)
    ste_dset = tgt_node.create_dataset(name='ste', data=std / np.sqrt(num))
    med_dset = tgt_node.create_dataset(name='med', data=np.nanmedian(data, axis=axis))
    max_dset = tgt_node.create_dataset(name='max', data=np.nanmax(data, axis=axis))
    min_dset = tgt_node.create_dataset(name='min', data=np.nanmin(data, axis=axis))
    num_dset = tgt_node.create_dataset(name='num', data=num)
    data_dset = tgt_node.create_dataset(name='data', data=data, compression="gzip", compression_opts=9 )


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch
@njit(cache=True, parallel=True)
def get_renyi_r(pn, alpha, pn_cutoff=0.0):
    d0,d2,d3 = np.shape(pn) # n, time, realization (pn for the midchain only)
    if pn_cutoff > 0:
        pn = np.where(pn < pn_cutoff , 0, pn)  # Cutoff
        pn = pn / np.sum(pn, axis=0).reshape(1, d2, d3)  # Normalize
    renyi = 1.0 / (1 - alpha) * np.log(np.sum(pn ** alpha, axis=0))
    return np.where(np.isinf(renyi), 0, renyi)  # Cutoff

def get_renyi(pn, alpha, pn_cutoff=0.0):
    d0, d1, d2, d3 = np.shape(pn)  # n, site, time, a batch of realizations
    L = d0-1
    Lhalf = int(L//2)
    popcount = int(L//2) # Half filling
    return get_renyi_r(pn=pn[:popcount+1,Lhalf,:,:][()], alpha=alpha, pn_cutoff=pn_cutoff)


def get_tdata(crononode):
    tsize = len(crononode.keys())
    if tsize in [101, 102]:
        tsize = 100
    if tsize in [201, 202]:
        tsize = 200
    tdata = np.zeros(shape=(tsize))
    # Instead of reading through all the datasets, we can calculate the time points!
    tmin = crononode[f'iter_1/measurements']['physical_time'][0].astype(float)
    tmax = crononode[f'iter_{tsize}/measurements']['physical_time'][0].astype(float)
    tchk = crononode[f'iter_{tsize-1}/measurements']['physical_time'][0].astype(float)
    logspace_times = np.logspace(np.log10(tmin), np.log10(tmax), tsize, endpoint=True)
    if abs(logspace_times[-2] - tchk) < 1e-1:
        return logspace_times
    linspace_times = np.linspace(tmin, tmax, tsize, endpoint=True)
    if abs(linspace_times[-2] - tchk) < 1e-1:
        return linspace_times
    print("WARNING: The time points do not seem to be logspaced or linspaced")
    print(f"{logspace_times=}")
    print(f"{linspace_times=}")
    # if not np.allclose(times, expected_times, atol=1):

    for tidx in range(tsize):
        tdata[tidx] = crononode[f'iter_{tidx+1}/measurements']['physical_time'][0].astype(float)
    tdata = np.sort(np.asarray(tdata))
    print(f"actual times: {tdata=}")
    return tdata

def get_pinfty(numprobs, dsetnode):
    print('get_pinfty()')
    modelnode = dsetnode.parent.parent['model']
    L = modelnode['hamiltonian'].attrs['model_size'][()]
    Lhalf = int(L//2)
    popcount = int(L//2) # Half filling
    hamiltonian = modelnode['hamiltonian']
    db = {'vals': {}}
    db['vals']['L'] = hamiltonian.attrs['model_size'][()]
    db['vals']['r'] = hamiltonian['J2_span'][()]
    db['vals']['x'] = hamiltonian['xi_Jcls'][()]
    db['vals']['w'] = (hamiltonian['J1_wdth'][()], hamiltonian['J2_wdth'][()], hamiltonian['J3_wdth'][()])
    print('get_pinfty() timepoints ...')
    tdata = get_tdata(dsetnode.parent['cronos'])
    # idx_num, idx_ent = find_saturation_idx3(tdata, db)
    t = get_timepoints(tdata, db)
    idx_num = t.idx_num_saturated
    print('get_pinfty() time averaging ...')
    ptavg = np.mean(numprobs[:popcount+1, Lhalf, idx_num:, :], axis=1) # infinite time average: time average for saturated time segment
    with np.errstate(invalid='ignore', divide='ignore'):
        print('get_pinfty() number entropy ...')
        return -np.nansum(ptavg * np.log(ptavg), axis=0), tdata # number entropy for the infinite time average

def get_pinfty_r(pn, idx_num_sat):
    # pn should have 3 dimensions: n, time, realization
    ptavg = np.mean(pn[:, idx_num_sat:, :], axis=1) # infinite time average: time average for the saturated time segment
    # with np.errstate(invalid='ignore', divide='ignore'):
    return -np.nansum(ptavg * np.log(ptavg), axis=0) # number entropy for the infinite time average


def get_matching_prop(props, dsetpath):
    for prop in props.keys():
        if dsetpath.endswith(prop):
            return prop
    return None


@njit([float64[:,:,:,:](float64[:,:,:,:], int64)], cache=True, parallel=True)
def get_pslice_new_convention(pslice,popcount):
    # In the old convention, the number probabilities were calculated in mixed left-right convention, such that
    # it got a triangular-looking shape. For sites 0...L/2, the convention was that p[:, :L/2,...] meant the probability
    # of finding a particle to the left, and p[:,L/2:,...] was the probability of finding a particle to the right.
    # In the new convention, p[:, :,...] all mean the probability of finding a particle to the left!
    # We can detect the old convention by checking for triangular form: particle 1 from the right is either at site L or L-1
    is_oldconvention = pslice[0,-2, 0, :] + pslice[1,-2, 0, :] > 0.5
    if not np.any(is_oldconvention):
        # print(f'   detected the new convention')
        return pslice
    # Some are old convention!
    d0,d1,d2,d3 = np.shape(pslice) # pslice has dimensions: n, site, time, realization
    pslice_new = np.empty(shape=(popcount+1,d1,d2,d3))
    print(f'   detected the old convention (all:{np.all(is_oldconvention)})' )
    for r in prange(d3):
        if is_oldconvention[r]:
            for row in range(popcount + 1):
                for col in range(d1):
                    if col <= int(d1 // 2):
                        pslice_new[row, col, :, r] = pslice[row, col, :, r]
                    else:
                        pslice_new[row, col, :, r] = pslice[popcount - row, col, :, r]
        else:
            pslice_new[:, :, :, r] = pslice[:popcount+1, :, :, r]  # Simply copy
    return pslice_new


@njit(cache=True,parallel=True)
def get_qin_probability_slice(pslice,popcount):
    pslice = get_pslice_new_convention(pslice, popcount)
    pslice[popcount, -1, :, :] = 1.0  # Fix for right edge lacking a 1 for the last particle for one of the neel states
    d0,d1,d2,d3 = np.shape(pslice) # pslice has dimensions: n, site, time, realization(slice)
    pdiff = pslice[:, 1:, :, :] - pslice[:, 0:-1, :, :]
    q = np.zeros(shape=(popcount, d1-1, d2, d3))
    for n in prange(1,popcount+1):
        q[n-1, :, :, :] = np.sum(pdiff[n:, :, :, :], axis=0)
    q = np.where(q<1e-14, 0, q) # Cutoff
    return q / np.sum(q, axis=1).reshape(popcount, 1, d2, d3) # Normalize



#@profile
def write_statistics_dset(meta, props, h5_tgt):
    print(f'writing dset stats: {meta[0]}')
    # Props contains the names of the datasets
    dsetname = meta[0]
    dsetpath = meta[1]
    dsetnode = meta[2]
    dsetprop = get_matching_prop(props, dsetpath)
    dsetaxis = props.get(dsetprop).get('axis')
    dsetcopy = props.get(dsetprop).get('copy')
    if not dsetaxis:
        dsetaxis = 0
    if dsetname == 'schmidt_midchain':
        dsetdata = np.array(dsetnode.view(dtype=np.complex128).real)
    if dsetname == 'number_probabilities':
        if dsetcopy:
            print(f'deep copying dset: {dsetname} {np.shape(dsetnode)}')
            tgt_node = h5_tgt.require_group(dsetnode.parent.name)
            tgt_node.copy(dsetnode, tgt_node, dsetname)

        # Since this object is huge we need to work in batches
        d0, d1, d2, d3 = np.shape(dsetnode)  # n, site, time, realization
        maxr = props.get(dsetprop).get('maxrealizations')
        maxbatchsize = props.get(dsetprop).get('maxbatchsize')
        if maxr is None:
            maxr = d3
        if maxbatchsize is None:
            maxr = d3
        modelnode = dsetnode.parent.parent['model']
        hamiltonian = modelnode['hamiltonian']
        L = get_value('model_size', [modelnode, hamiltonian], optional=False)
        if L+1 != d1:  # We have length L + 1 on the d0 and d1 dimensions
            raise AssertionError(f"System size mismatch: {L=} != {d1-1=}")
        # Site indexing
        # With 8 sites, the half-chain boundary is between site index 3 and 4 (indexing from 0), i.e. at L//2-1
        # For number probabilities, we do not really use site indices but bond indices, starting to the left of site 0.
        # Therefore, for 8 sites we get 9 indices:
        #       .o.o.o.o,o.o.o.o.
        # where the "o" are sites and "." are bonds. The central bond is marked with a ",", and it lies at index 4, i.e. L//2.


        Lhalf = int(L//2) # E.g. for 8 sites, the midchain is to the right of index 3
        popcount = int(L // 2)  # E.g. for 8 sites, we have 4 particles at half-filling

        # Get the time points
        print('get timepoints ...')
        db = {'vals': {}}
        db['vals']['L'] = L
        db['vals']['r'] = get_value('J2_span', [hamiltonian], optional=False)
        db['vals']['x'] = get_value(['xi_Jcls', 'J2_xcls'], [hamiltonian], optional=False)
        db['vals']['w'] = tuple([get_value(key, [hamiltonian], optional=False) for key in ['J1_wdth', 'J2_wdth', 'J3_wdth']])
        tdata = get_tdata(dsetnode.parent['cronos'])
        # idx_num, idx_ent = find_saturation_idx3(tdata, db)
        t = get_timepoints(tdata, db)

        # Initialize data containers
        hartley_number_entropy_data = None  # time, realization (midchain only)
        renyi2_number_entropy_data  = None  # time, realization (midchain only)
        pinfty_number_entropy_data  = None  # n, realization

        # Initialize group paths
        hartley_number_entropy_path = dsetnode.parent.name + '/hartley_number_entropies'
        renyi2_number_entropy_path = dsetnode.parent.name + '/renyi2_number_entropies'
        pinfty_number_entropy_path = dsetnode.parent.name + '/pinfty_number_entropies'
        nth_particle_position_path = dsetnode.parent.name + '/nth_particle_position'
        do_nth_particle = any([props.get(dsetprop).get(x) is True for x in ['qin_probability', 'pos_expvalue', 'pos_variance']])
        do_other_entropies = any([props.get(dsetprop).get(x) is True for x in ['pinfty', 'hartley', 'renyi2']])

        # Initialize dsets
        qin_probability_dset = None
        pos_expvalue_neel0_dset = None
        pos_expvalue_neel1_dset = None
        pos_variance_neel0_dset = None
        pos_variance_neel1_dset = None
        neel_type_dset = None

        chunksize = np.min([d3, maxbatchsize, 1000])
        if do_nth_particle:
            tgt_node = h5_tgt.require_group(nth_particle_position_path)
            tgt_node.attrs.create(name="description",
                                  data="This group contains q_i(n) [qin_probability] and the moments of position\n"
                                       "for the nth particle on the chain [pos_expvalue] and [pos_variance].\n"
                                       "The order of dimensions is [particle n, site i, time, realization]\n"
                                       "For datasets other than qin_probability, dimension [site] has been averaged\n"
                                       "For _davg_ and _dste_ datasets, dimension [realization] has been averaged\n"
                                       "davg: disorder-average\n"
                                       "dste: standard error of the disorder-average\n"
                                       "neel0: 010101|010101 for L=12: state starts empty at site 0\n"
                                       "neel1: 101010|101010 for L=12: state starts filled at site 0\n"
                                       "if neel0: particle #n L/4-1=2 at i=5 is closest to the midchain boundary 5.5\n"
                                       "if neel1: particle #n L/4  =3 at i=6 is closest to the midchain boundary 5.5"
                                  )
        if props.get(dsetprop).get('qin_probability'):
            tgt_node = h5_tgt.require_group(nth_particle_position_path)
            qin_probability_dset = tgt_node.create_dataset(name="qin_probability", data=None, shape=(popcount, d1-1, d2, maxr),
                                                   dtype=np.float64, compression="gzip", compression_opts=1,
                                                   chunks=(popcount, d1-1, d2, chunksize), )
        if props.get(dsetprop).get('pos_expvalue'):
            tgt_node = h5_tgt.require_group(nth_particle_position_path)
            pos_expvalue_neel0_dset = tgt_node.create_dataset(name="pos_expvalue_neel0", data=None,
                                                              shape=(popcount, d2, 0),
                                                              maxshape=(popcount,d2, None),
                                                              dtype=np.float64, compression="gzip", compression_opts=2,
                                                              chunks=(popcount, d2, chunksize), )
            pos_expvalue_neel1_dset = tgt_node.create_dataset(name="pos_expvalue_neel1", data=None,
                                                              shape=(popcount, d2, 0),
                                                              maxshape=(popcount, d2, None),
                                                              dtype=np.float64, compression="gzip", compression_opts=2,
                                                              chunks=(popcount, d2, chunksize), )
            neel_type_dset = tgt_node.create_dataset(name="neel_type", data=None, shape=(maxr),
                                                     dtype=np.int8, compression="gzip",
                                                     compression_opts=0,
                                                     chunks=(chunksize,))
            neel_type_dset.attrs.create(name="comment",
                                        data="neel_type=0: empty at site 0 | neel_type=1: filled at site 0")
        if props.get(dsetprop).get('pos_variance'):
            tgt_node = h5_tgt.require_group(nth_particle_position_path)
            pos_variance_neel0_dset = tgt_node.create_dataset(name="pos_variance_neel0", data=None,
                                                        shape=(popcount, d2, 0),
                                                        maxshape=(popcount, d2, None),
                                                        dtype=np.float64,
                                                        compression="gzip", compression_opts=2,
                                                        chunks=(popcount, d2, chunksize), )
            pos_variance_neel1_dset = tgt_node.create_dataset(name="pos_variance_neel1", data=None,
                                                        shape=(popcount, d2, 0),
                                                        maxshape=(popcount, d2, None),
                                                        dtype=np.float64,
                                                        compression="gzip", compression_opts=2,
                                                        chunks=(popcount, d2, chunksize), )

        neel0_off, neel0_ext = 0,0
        neel1_off, neel1_ext = 0,0
        for r in batched(range(maxr), maxbatchsize):
            if not do_other_entropies and not do_nth_particle:
                continue
            batchsize = len(r)
            # with Profile() as profile:
            print(f'reading number probabilities {r[0]} to {r[-1]} ({maxr} total) ...')
            numprobs = dsetnode[:,:,:,r][()] # Read a slice

            if props.get(dsetprop).get('hartley'):
                if hartley_number_entropy_data is None:
                    hartley_number_entropy_data = np.zeros(shape=(d2, maxr))
                print(f'-- calculating hartley_number_entropies')
                hartley_number_entropy_data[:, r] = get_renyi_r(numprobs[:popcount + 1, Lhalf, :, :], alpha=1e-3, pn_cutoff=1e-8)
            if props.get(dsetprop).get('renyi2'):
                if renyi2_number_entropy_data is None:
                    renyi2_number_entropy_data = np.zeros(shape=(d2, maxr))
                print(f'-- calculating renyi2_number_entropies')
                renyi2_number_entropy_data[:, r] = get_renyi_r(pn=numprobs[:popcount + 1, Lhalf, :, :], alpha=2, pn_cutoff=0)
            if props.get(dsetprop).get('pinfty'):
                if pinfty_number_entropy_data is None:
                    pinfty_number_entropy_data = np.zeros(shape=(d2, maxr))
                print(f'-- calculating pinfty_number_entropies')
                pinfty_number_entropy_data[:,r] = get_pinfty_r(numprobs[:popcount+1, Lhalf, :, :], idx_num_sat=t.idx_num_saturated)
            if do_nth_particle:
                print(f'-- calculating qin_probability')
                qin_probability = get_qin_probability_slice(numprobs, popcount=popcount)  # n site time realization
                if props.get(dsetprop).get('qin_probability'):
                    print(f'-- writing qin_probability')
                    qin_probability_dset[:, :, :, r] = qin_probability  # Write to file, it may not fit in memory
                if props.get(dsetprop).get('pos_expvalue'):
                    print(f'  -- calculating pos_expvalue')
                    pos_expvalue = np.sum(qin_probability * np.arange(0, L)[np.newaxis, :, np.newaxis, np.newaxis],axis=1)  # n, time, realization
                    neel_type = pos_expvalue[0, 0, :] < 0.5
                    neel0, = np.where(neel_type == 0)  # Starts empty on site 0
                    neel1, = np.where(neel_type != 0)  # Starts filled on site 1
                    neel0_ext, neel1_ext = len(neel0), len(neel1)
                    range0 = range(neel0_off, neel0_off+neel0_ext)
                    range1 = range(neel1_off, neel1_off+neel1_ext)

                    neel_type_dset[r] = neel_type  # Write to file
                    print(f'  -- writing pos_expvalue neel0:{len(neel0)} neel1:{len(neel1)}')
                    pos_expvalue_neel0_dset.resize(neel0_off+neel0_ext, axis=2)
                    pos_expvalue_neel1_dset.resize(neel1_off+neel1_ext, axis=2)
                    pos_expvalue_neel0_dset[:, :, range0] = pos_expvalue[:, :, neel0]  # Write to file
                    pos_expvalue_neel1_dset[:, :, range1] = pos_expvalue[:, :, neel1]  # Write to file
                    # If the position expectation value of the first particle is close to <x>=0, then site 0 is filled.
                    # If site 0 is filled, it means we start filled with a |1> on site 0, so neel_type is 1.
                    if props.get(dsetprop).get('pos_variance'):
                        print(f'  -- calculating pos_variance')
                        pos_diff_sq = (np.broadcast_to(pos_expvalue, (L, popcount, d2, batchsize)).transpose(1, 0, 2, 3)  #
                                       - np.arange(0, L)[np.newaxis, :, np.newaxis, np.newaxis]  #
                                      ) ** 2  # n, time, realization
                        pos_variance = np.sum(qin_probability * pos_diff_sq, axis=1)
                        print(f'  -- writing pos_variance neel0:{len(neel0)} neel1:{len(neel1)}')
                        pos_variance_neel0_dset.resize(neel0_off+neel0_ext, axis=2)
                        pos_variance_neel1_dset.resize(neel1_off+neel1_ext, axis=2)
                        pos_variance_neel0_dset[:, :, range0] = pos_variance[:,:,neel0]  # Write to file
                        pos_variance_neel1_dset[:, :, range1] = pos_variance[:,:,neel1]  # Write to file
                    neel0_off, neel1_off = neel0_off+neel0_ext, neel1_off+neel1_ext

            # Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

        # Now write the calculated data
        # with Profile() as profile:
        if props.get(dsetprop).get('hartley'):
            print(f' -- writing hartley_number_entropies')
            tgt_node = h5_tgt.require_group(hartley_number_entropy_path)
            write_stats_to_node(data=hartley_number_entropy_data, tgt_node=tgt_node, axis=1)
        if props.get(dsetprop).get('renyi2'):
            print(f' -- writing renyi2_number_entropies')
            tgt_node = h5_tgt.require_group(renyi2_number_entropy_path)
            write_stats_to_node(data=hartley_number_entropy_data, tgt_node=tgt_node, axis=1)
        if props.get(dsetprop).get('pinfty'):
            print(f' -- writing pinfty_number_entropies')
            tgt_node = h5_tgt.require_group(pinfty_number_entropy_path)
            write_stats_to_node(data=pinfty_number_entropy_data, tgt_node=tgt_node, axis=0)
            tgt_node.attrs.create(name="physical_time", data=tdata)

        if neel_type_dset is not None:
            neel_type = neel_type_dset[:maxr]
            neel0, = np.where(neel_type == 0)  # Starts empty on site 0
            neel1, = np.where(neel_type != 0)  # Starts filled on site 1
            tgt_node = h5_tgt.require_group(nth_particle_position_path)
            # neel0: 0101|0101 empty at site 1, so index L//4-1 = is nearest the midchain boundary
            # neel1: 1010|1010 index L//4 = 2 is nearest the midchain boundary
            if pos_expvalue_neel0_dset is not None and props.get(dsetprop).get('pos_expvalue_davg'):
                print('writing pos_expvalue_davg_mid')
                pos_expvalue_neel0 = pos_expvalue_neel0_dset[:, :, :maxr]
                pos_expvalue_neel1 = pos_expvalue_neel1_dset[:, :, :maxr]
                print(f'shape: {np.shape(pos_expvalue)=}')
                pos_expvalue_davg_neel0 = np.mean(pos_expvalue_neel0, axis=2)
                pos_expvalue_davg_neel1 = np.mean(pos_expvalue_neel1, axis=2)
                pos_expvalue_dste_neel0 = np.std(pos_expvalue_neel0, axis=2)/np.sqrt(len(neel0))
                pos_expvalue_dste_neel1 = np.std(pos_expvalue_neel1, axis=2)/np.sqrt(len(neel1))
                tgt_node.create_dataset(name="pos_expvalue_davg_neel0", data=pos_expvalue_davg_neel0, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_expvalue_davg_neel1", data=pos_expvalue_davg_neel1, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_expvalue_dste_neel0", data=pos_expvalue_dste_neel0, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_expvalue_dste_neel1", data=pos_expvalue_dste_neel1, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
            if pos_variance_neel0_dset is not None and props.get(dsetprop).get('pos_variance_davg'):
                print('writing pos_variance_davg_mid')
                pos_variance_neel0 = pos_variance_neel0_dset[:, :, :maxr]
                pos_variance_neel1 = pos_variance_neel1_dset[:, :, :maxr]
                print(f'shape: {np.shape(pos_variance_neel0)}')
                pos_variance_davg_neel0 = np.mean(pos_variance_neel0, axis=2)
                pos_variance_davg_neel1 = np.mean(pos_variance_neel1, axis=2)
                pos_variance_dste_neel0 =  np.std(pos_variance_neel0, axis=2)/np.sqrt(len(neel0))
                pos_variance_dste_neel1 =  np.std(pos_variance_neel1, axis=2)/np.sqrt(len(neel1))
                tgt_node.create_dataset(name="pos_variance_davg_neel0", data=pos_variance_davg_neel0, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_variance_davg_neel1", data=pos_variance_davg_neel1, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_variance_dste_neel0", data=pos_variance_dste_neel0, compression="gzip", compression_opts=2, chunks=(popcount, d2), )
                tgt_node.create_dataset(name="pos_variance_dste_neel1", data=pos_variance_dste_neel1, compression="gzip", compression_opts=2, chunks=(popcount, d2), )

            # Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()

    else:
        if dsetcopy:
            print(f'copying dset: {dsetname} {np.shape(dsetnode)}')
            tgt_node = h5_tgt.require_group(dsetnode.parent.name)
            tgt_node.copy(source=dsetnode, dest=tgt_node)
        else:
            tgt_node = h5_tgt.require_group(dsetpath)
            # print('writing dset "{}" along axis {}'.format(dsetname, dsetaxis))
            write_stats_to_node(data=dsetnode[()], tgt_node=tgt_node, axis=dsetaxis)


def write_statistics_ed(h5_ed_src, tgt_node, L):
    measurements_node = tgt_node.require_group(tgt_node.name + "/measurements")
    parity_candidates = ["/", "/0/", "/1/"]
    if "energies" in h5_ed_src:
        data = None
        for par in parity_candidates:
            path = "energies" + par + str(L)
            if path in h5_ed_src:
                if np.shape(h5_ed_src[path]) is None:
                    print("Shape", path, "is None")
                    continue
                if data is None:
                    data = np.array(h5_ed_src[path])
                else:
                    data = np.concatenate([data, np.array(h5_ed_src[path])], axis=0)
        energy_node = measurements_node.require_group("energy")
        write_stats_to_node(data=data, tgt_node=energy_node, axis=0)
        data = data / L
        energy_per_site_node = measurements_node.require_group("energy_per_site")
        write_stats_to_node(data=data, tgt_node=energy_per_site_node, axis=0)

    if "relativeEnergies" in h5_ed_src:
        data = None
        for par in parity_candidates:
            path = "relativeEnergies" + par + str(L)
            if path in h5_ed_src:
                if np.shape(h5_ed_src[path]) is None:
                    print("Shape", path, "is None")
                    continue
                if data is None:
                    data = np.array(h5_ed_src[path])
                else:
                    data = np.concatenate([data, np.array(h5_ed_src[path])], axis=0)
        energy_dens_node = measurements_node.require_group("energy_dens")
        write_stats_to_node(data=data, tgt_node=energy_dens_node, axis=0)

    if "entropies" in h5_ed_src:
        data = None
        for par in parity_candidates:
            path = "entropies" + par + str(L)
            if path in h5_ed_src:
                if np.shape(h5_ed_src[path]) is None:
                    print("Shape", path, "is None")
                    continue
                if data is None:
                    data = np.array(h5_ed_src[path])
                else:
                    data = np.concatenate([data, np.array(h5_ed_src[path])], axis=1)

        # Pad the data
        zero_row = np.zeros([1, data.shape[1]])
        data = np.append(zero_row, data, axis=0)
        data = np.append(data, zero_row, axis=0)
        entropy_node = measurements_node.require_group("entanglement_entropies")
        write_stats_to_node(data=data, tgt_node=entropy_node, axis=1)
        width = np.shape(data)[0]
        middle = int(width / 2)
        data = data[middle, :]
        entropy_node = measurements_node.require_group("entanglement_entropy")
        write_stats_to_node(data=data, tgt_node=entropy_node, axis=0)


def write_statistics(src, tgt, reqs):
    # h5_src = h5open(src, 'r')
    write_statistics_crono4.number_probabilities_path = None
    write_statistics_crono4.hartley_number_entropy_data = None

    with h5py.File(src, 'r', libver='latest', swmr=True, rdcc_nbytes=1 * 1024 ** 3, rdcc_nslots=521, driver='sec2') as h5_src:
        with h5py.File(tgt, 'w') as h5_tgt:
            print('Averaging dsets')
            for dsetname, dsetpath, dsetnode in h5py_node_iterator(node=h5_src, keypattern=reqs['dsets'], dep=20, excludeKeys=['.db', 'cronos', 'iter_'],
                                                                   nodeType=h5py.Dataset):
                print('Found dset: {}'.format(dsetpath))
                if 'r8' not in dsetpath:
                    continue
                write_statistics_dset((dsetname, dsetpath, dsetnode), reqs['dsets'], h5_tgt)

            print('Averaging tables')
            for tablename, tablepath, tablenode in h5py_node_iterator(node=h5_src, keypattern=reqs['tables'], dep=20,
                                                                      excludeKeys=['.db', 'cronos', 'dsets', 'iter_'],
                                                                      nodeType=h5py.Dataset):
                if 'r8' not in tablepath:
                    continue
                write_statistics_table2((tablename, tablepath, tablenode), reqs['tables'], tgt)


        # with tb.File(tgt, 'a') as h5f:
        #     print('Averaging cronos v4')
        #     node_cache = {}
        #     done_crono = {}
        #     for crononame, cronopath, crononode in h5py_node_iterator(node=h5_src, keypattern='cronos', dep=20,
        #                                                               excludeKeys=['.db', 'model', 'tables',
        #                                                                            'dsets'],
        #                                                               nodeType=h5py.Group, godeeper=False):
        #         print('found cronos:', cronopath)
        #         if done := done_crono.get(cronopath):
        #             continue
        #         else:
        #             for iternode in h5py_node_iterator(node=crononode, keypattern='iter_', dep=1,
        #                                                nodeType=h5py.Group,
        #                                                godeeper=False):
        #                 done_crono[cronopath] = write_statistics_crono4(iternode, reqs['cronos'], h5f, node_cache)
        #                 # print('found iter:', iternode[1])
        #                 if done := done_crono.get(cronopath):
        #                     print('{} is done'.format(cronopath))
        #                     break


    with h5py.File(tgt, 'a') as h5_tgt:
        # Find ED data
        print('Finding ED data')
        for dirName, subdirList, fileList in os.walk("ed_data"):
            subdirList.sort()
            fileList.sort()
            for src_filename in fileList:
                try:
                    h5_ed_src = h5py.File(dirName + '/' + src_filename, 'r', swmr=True)
                except:
                    continue

                if "ed-l_0.0000-d_[-0.5...0.5]-e[0.00-1.00].h5" in src_filename:
                    for key, path, node in h5py_node_finder(h5_tgt, keypattern="d_", dep=3, num=0):
                        # node on the target file refers to a point in the phase diagram, encoded in its path
                        # to find the same phase diagram point in ED, just compare the path!
                        if path in h5_ed_src:
                            for algo_key, algo_node in h5_ed_src[path].items():
                                node.copy(source=algo_node, dest=node)
                    break

                # We now have some ED data but we must infer what its parameters are
                # The numbers in paths such as L_16/l_0/J_0/h_0 simply enumerate
                # the values in batch simulations. The numbers are NOT the values
                # of these parameters, except in the case of L

                # Start by finding out the values in the current ED data
                # ed_lambda = h5_ed_src["lambda"][()]
                # ed_delta  = h5_ed_src["delta"][()]
                # ed_J_mean = h5_ed_src["J_mean"][()]
                # ed_h_mean = h5_ed_src["h_mean"][()]
                # ed_efmt   = h5_ed_src["efmt"][()]
                #
                # for key, path, node in h5py_node_finder(h5_tgt, filter="d_", dep=3, num=0):
                #     tgt_lambda = node.attrs["lambda"]
                #     tgt_delta  = node.attrs["delta"]
                #     tgt_J_mean = node.attrs["J_mean"]
                #     tgt_h_mean = node.attrs["h_mean"]
                #     tgt_L      = node.attrs["model_size"]
                #     if(ed_lambda != tgt_lambda):
                #         continue
                #     if(ed_delta != tgt_delta):
                #         continue
                #     if(ed_J_mean != tgt_J_mean):
                #         continue
                #     if(ed_h_mean != tgt_h_mean):
                #         continue
                #     # Now we know he ED and target data refer to the same point on the
                #     # phase diagram. Now we just have to match L data
                #     for ed_L in h5_ed_src["Ls"]:
                #         if ed_L == tgt_L:
                #             tgt_node = h5_tgt.require_group(path + "/ed-e" + ed_efmt + "/states")
                #             write_statistics_ed(h5_ed_src=h5_ed_src, tgt_node=tgt_node,L=ed_L)
                #             tgt_node.attrs["efmt"] = h5_ed_src["efmt"]
                #             tgt_node.attrs["emin"] = h5_ed_src["emin"]
                #             tgt_node.attrs["emax"] = h5_ed_src["emax"]


