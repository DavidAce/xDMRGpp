import os
import warnings
from timeit import default_timer as timer

import numba as nb
import tables as tb
import tables.parameters
from numba import njit
from tables import NaturalNameWarning
from decimal import Decimal
from common.io.h5ops import *

warnings.filterwarnings('ignore', category=NaturalNameWarning)

tb.parameters.COND_CACHE_SLOTS = 1024  # Maximum number of conditions for table queries to be kept in memory.
tb.parameters.METADATA_CACHE_SIZE = 10485760  # Size (in bytes) of the HDF5 metadata cache.
#    NODE_CACHE_SLOTS: Maximum number of nodes to be kept in the metadata cache.
#    It is the number of nodes to be kept in the metadata cache. Least recently used nodes are unloaded from memory when this number of loaded nodes is reached. To load a node again, simply access it as usual. Nodes referenced by user variables and, in general, all nodes that are still open are registered in the node manager and can be quickly accessed even if they are not in the cache.
#    Negative value means that all the touched nodes will be kept in an internal dictionary. This is the faster way to load/retrieve nodes. However, and in order to avoid a large memory comsumption, the user will be warned when the number of loaded nodes will reach the -NODE_CACHE_SLOTS value.
#    Finally, a value of zero means that any cache mechanism is disabled.
tables.parameters.NODE_CACHE_SLOTS = 4096
tables.parameters.CHUNK_CACHE_NELMTS = 521  # Number of elements for HDF5 chunk cache.
tables.parameters.CHUNK_CACHE_PREEMPT = 0.75  # Chunk preemption policy. This value should be between 0 and 1 inclusive and indicates how much chunks that have been fully read are favored for preemption. A value of zero means fully read chunks are treated no differently than other chunks (the preemption is strictly LRU) while a value of one means fully read chunks are always preempted before other chunks.
tables.parameters.CHUNK_CACHE_SIZE = 65536000  # Size (in bytes) for HDF5 chunk cache.

nb.set_num_threads(8)


def write_statistics_table(meta, props, h5_tgt):
    # Props contains the names of the tables
    # We should generate the statistics for each column in the table
    tablename = meta[0]
    tablepath = meta[1]
    tabledset = meta[2]
    if not isinstance(tabledset, h5py.Dataset):
        raise TypeError("write_statistics_table: meta should point to a h5py.Dataset. Got: ", meta)

    table_col_names = [col for col in tabledset if col in props[tablename]]
    for table_col_name in table_col_names:
        data = tabledset[table_col_name]
        num = np.shape(data)[0]
        std = np.nanstd(data, axis=0)
        tgt_path = tablepath + "/" + table_col_name
        tgt_node = h5_tgt.require_group(tgt_path)
        tgt_node.create_dataset(name='avg', data=np.nanmean(data, axis=0))
        tgt_node.create_dataset(name='std', data=std)
        tgt_node.create_dataset(name='ste', data=std / np.sqrt(np.shape(data)[0]))
        tgt_node.create_dataset(name='med', data=np.nanmedian(data, axis=0))
        tgt_node.create_dataset(name='max', data=np.nanmax(data, axis=0))
        tgt_node.create_dataset(name='min', data=np.nanmin(data, axis=0))
        tgt_node.create_dataset(name='q25', data=np.nanpercentile(data, 0.25, axis=0))
        tgt_node.create_dataset(name='q75', data=np.nanpercentile(data, 0.75, axis=0))
        tgt_node.create_dataset(name='num', data=num)
        tgt_node.create_dataset(name='data', data=data)
        # Generate some nice attributes for later plotting
    # model_path = tablepath[0: tablepath.index("LBIT")] + "LBIT/model"
    # if model_path in tabledset.file:
    #     model_node = tabledset.file[model_path]
    #     attrs_path = model_node.parent.parent.name
    #     attrs_node = h5_tgt.require_group(attrs_path)
    #     for key, dset in model_node.items():
    #         if '.db' in key:
    #             continue
    #         if key in attrs_node.attrs:
    #             break
    #         # print(key,dset,model_node)
    #         attrs_node.attrs.create(name=key, data=dset)
    # else:
    #     raise LookupError("Could not find [" + model_path + "] in ", tabledset.file)


def write_statistics_crono(meta, props, h5_tgt):
    # Props contains the names of the tables
    # We should generate the statistics for each column in the table
    tablename = meta[0]
    tablepath = meta[1]
    tablenode = meta[2]
    if not isinstance(tablenode, h5py.Group):
        raise TypeError("write_statistics_crono: meta should point to a h5py.Group. Got: ", meta)

    for iterkey, iterpath, iternode in h5py_dataset_iterator(node=tablenode, dep=1):
        for table_col_name in props[tablename]:
            data = iternode[table_col_name]
            num = np.shape(data)[0]
            std = np.nanstd(data, axis=0)
            tgt_path = "{}/{}/{}".format(tablepath, table_col_name, iterkey)
            tgt_node = h5_tgt.require_group(tgt_path)
            tgt_node.create_dataset(name='avg', data=np.nanmean(data, axis=0))
            tgt_node.create_dataset(name='std', data=std)
            tgt_node.create_dataset(name='ste', data=std / np.sqrt(np.shape(data)[0]))
            tgt_node.create_dataset(name='med', data=np.nanmedian(data, axis=0))
            tgt_node.create_dataset(name='max', data=np.nanmax(data, axis=0))
            tgt_node.create_dataset(name='min', data=np.nanmin(data, axis=0))
            tgt_node.create_dataset(name='q25', data=np.nanpercentile(data, 0.25, axis=0))
            tgt_node.create_dataset(name='q75', data=np.nanpercentile(data, 0.75, axis=0))
            tgt_node.create_dataset(name='num', data=num)
            tgt_node.create_dataset(name='data', data=data)
            # Generate some nice attributes for later plotting
        model_path = tablepath[0: tablepath.index("LBIT")] + "LBIT/model"
        if model_path in iternode.file:
            model_node = iternode.file[model_path]
            attrs_path = model_node.parent.parent.name
            attrs_node = h5_tgt.require_group(attrs_path)
            for key, dset in model_node.items():
                if '.db' in key:
                    continue
                if key in attrs_node.attrs:
                    break
                # print(key,dset,model_node)
                attrs_node.attrs.create(name=key, data=dset)
        else:
            raise LookupError("Could not find [" + model_path + "] in ", iternode.file)


def write_statistics_crono2(meta, props, h5_tgt):
    # Props contains the names of the tables
    # We should generate the statistics for each column in the table
    # itername = meta[0]
    iterpath = meta[1]
    iternode = meta[2]
    if not isinstance(iternode, h5py.Group):
        raise TypeError("write_statistics_crono2: meta should point to a h5py.Group. Got: ", meta)
    point_path = iternode.parent.name
    print('cronos:', point_path)
    for tablekey, tablepath, tablenode in h5py_node_iterator(node=iternode, keypattern=props, dep=1,
                                                             nodeType=h5py.Dataset):
        for table_col_name in props[tablekey]:
            iter = tablenode['iter'][0]
            tgt_path = "{}/{}/{}".format(point_path, table_col_name, iter)
            if tgt_path in h5_tgt:
                continue
            tgt_node = h5_tgt.require_group(tgt_path)
            data = tablenode[table_col_name]
            num = np.shape(data)[0]
            std = np.nanstd(data, axis=0)
            tgt_node.create_dataset(name='avg', data=np.nanmean(data, axis=0))
            tgt_node.create_dataset(name='std', data=std)
            tgt_node.create_dataset(name='ste', data=std / np.sqrt(np.shape(data)[0]))
            tgt_node.create_dataset(name='med', data=np.nanmedian(data, axis=0))
            tgt_node.create_dataset(name='max', data=np.nanmax(data, axis=0))
            tgt_node.create_dataset(name='min', data=np.nanmin(data, axis=0))
            # tgt_node.create_dataset(name='q25', data=np.nanpercentile(data, 0.25, axis=0))
            # tgt_node.create_dataset(name='q75', data=np.nanpercentile(data, 0.75, axis=0))
            tgt_node.create_dataset(name='num', data=num)
            tgt_node.create_dataset(name='data', data=data)
            # Generate some nice attributes for later plotting
    model_path = iterpath[0: iterpath.index("LBIT")] + "LBIT/model"
    if model_path in iternode.file:
        model_node = iternode.file[model_path]
        attrs_path = model_node.parent.parent.name
        attrs_node = h5_tgt.require_group(attrs_path)
        for key, dset in model_node.items():
            if '.db' in key:
                continue
            if key in attrs_node.attrs:
                break
            # print(key,dset,model_node)
            attrs_node.attrs.create(name=key, data=dset)
    else:
        raise LookupError("Could not find [" + model_path + "] in ", iternode.file)


stattables = {
    'avg': (lambda data, std, num: np.nanmean(data, axis=0)),
    'std': (lambda data, std, num: std),
    'ste': (lambda data, std, num: std / np.sqrt(num)),
    'med': (lambda data, std, num: np.nanmedian(data, axis=0)),
    'max': (lambda data, std, num: np.nanmax(data, axis=0)),
    'min': (lambda data, std, num: np.nanmin(data, axis=0)),
}

constant_cols = [
    'iter', 'step', 'position', 'num', 'num_iters', 'max_iters', 'bond_lim', 'event',
    'bond_dimension_current', 'bond_dimension_max',
    'phys_time', 'algo_type', 'algo_stop']


@nb.njit(nogil=False, parallel=True, cache=True)
def get_stat(data, colname, statkey):
    result = 0.0
    if colname in constant_cols:
        result = np.max(data)
    else:
        if statkey == 'avg':
            result = np.mean(data)
        elif statkey == 'std':
            result = np.std(data)
        elif statkey == 'ste':
            num = np.shape(data)[0]
            result = np.std(data) / np.sqrt(num)
        elif statkey == 'med':
            result = np.median(data)
        elif statkey == 'max':
            result = np.max(data)
        elif statkey == 'min':
            result = np.min(data)
    return result


# @nb.jit('Tuple((float64[:], float64[:,:]))(float64[:], float64[:,:])',nopython=True)
# @njit(nogil=False, parallel=True, cache=True)
# def get_stats_complex128(data):
#     num = np.shape(data)[0]
#     print("shape:",np.shape(data))
#     std = np.std(data)
#     return np.mean(data), std, std / np.sqrt(num), np.median(data), np.max(data), np.min(data)

@nb.njit(parallel=True, cache=True)
def get_stats(data):
    num = np.shape(data)[0]
    std = np.std(data)
    avg = np.mean(data)
    med = np.median(data)
    min = np.min(data)
    max = np.max(data)
    return avg, std, std / np.sqrt(num), med, max, min


# @overload(get_stats, jit_options={'nogil':False, 'parallel':True, 'cache':True})
# def jit_get_stats(data):
#     # print(data, type(data), type(data.dtype))
#
#     nb.set_num_threads(4)
#
#     if isinstance(data.dtype, np.float64):
#         def get_stats_float64(data):
#             print(type(data[0]))
#             if np.all(data == data[0]):
#                 return data[0], 0., 0., 0., data[0], data[0]
#             num = np.shape(data)[0]
#             std = np.std(data)
#             avg = np.mean(data)
#             med = 0#np.median(data)
#             min = np.min(data)
#             max = np.max(data)
#             return avg, std, std / np.sqrt(num), med, max, min
#         return get_stats_float64
#     else :
#         def get_stats_other(data):
#             # print(type(data[0]))
#             num = np.shape(data)[0]
#             std = np.std(data)
#             avg = np.mean(data)
#             med = np.median(data)
#             min = np.min(data)
#             max = np.max(data)
#             return avg, std, std / np.sqrt(num), med, max, min
#         return get_stats_other
#
# @nb.njit
# def get_stats_safe(data):
#     return get_stats(data)

# @nb.njit(nogil=False, parallel=True, cache=True)
# def get_stats(data):
#     print(data.dtype)
#     if np.all(data == data[0]):
#         return data[0], 0., 0., 0., data[0], data[0]
#
#     num = np.shape(data)[0]
#     std = np.std(data)
#     avg = np.mean(data)
#     med = np.median(data)
#     min = np.min(data)
#     max = np.max(data)
#     return avg, std, std / np.sqrt(num), med, max, min


def get_dtype(tablenode, req_columns, num=True):
    names = []
    formats = []
    if num:
        names = ['num']
        formats = ['i8']
    for col, (dtype, offset) in tablenode.dtype.fields.items():
        if req_columns == 'ALL' or col in req_columns:
            names.append(col)
            if col in constant_cols or np.issubdtype(dtype, np.uint8):  # np.uint8 is a bool!
                formats.append(('{}={}{}'.format(dtype.shape, dtype.base.kind, dtype.alignment)))
            elif dtype.base.kind == 'V' and dtype.alignment == 1 and dtype.name == 'void128':  # Check if complex. These show up as "V1"
                formats.append('complex128')
            elif dtype.base.kind == 'S' and dtype.alignment == 1:
                formats.append(f'{dtype.kind}{dtype.itemsize}')
            else:
                formats.append(('{}=f{}'.format(dtype.shape, dtype.alignment)))
            if num and col == 'number_entropy' and 'hartley_number_entropy' in req_columns:
                names.append('hartley_number_entropy')
                formats.append(('<f8'))
    return np.dtype({"names": names, "formats": formats})


def write_statistics_table2(nodemeta, tablereqs, tgt):
    # Props contains the names of the tables as keys, and each value contains either "ALL" or a list of desired columns
    # We should generate the statistics for each column in the table, averaging each time point over all realizations
    t_tot = timer()
    t_stat = 0
    t_gets = 0
    t_crt = 0
    t_pre = 0
    t_itr = 0
    t_app = 0
    tablename = nodemeta[0]  # The name of the table
    tablepath = nodemeta[1]  # The path of the table e.g. '...u_3/r_8/fLBIT/state_real/tables/iter_1'
    tablenode = nodemeta[2]  # The node of the table

    if not isinstance(tablenode, h5py.Dataset):
        raise TypeError("write_statistics_table2: meta should point to a h5py.Dataset. Got: ", nodemeta)
    point_node = tablenode.parent
    point_path = point_node.name
    statgroup = '{}/{}'.format(point_path, tablename)

    if '__save_data__' in tablereqs and tablename in tablereqs['__save_data__']:
        with h5py.File(tgt,'a') as h5f:
            chunks = np.min([len(tablenode[()]), 100])
            h5f.create_dataset(f'{statgroup}/data', data=tablenode,
                               chunks=(chunks,), compression='gzip', compression_opts=9)

    # open h5 file for appending
    with tb.File(tgt, 'a') as h5f:
        t_pre_start = timer()
        # We now have a source table. Let's get the final dtype
        dt = get_dtype(tablenode=tablenode, req_columns=tablereqs[tablename])
        t_pre = t_pre + timer() - t_pre_start

        statrows = {}
        num = len(tablenode)

        for statkey in stattables.keys():
            t_crt_start = timer()
            statpath = '{}/{}/{}'.format(point_path, tablename, statkey)
            stattitle = '{} {}'.format(tablename, statkey)

            if not statpath in h5f:
                a = h5f.create_table(statgroup, name=statkey, title=stattitle,
                                     description=dt.newbyteorder('<'),
                                     createparents=True,
                                     expectedrows=num,
                                     filters=tb.Filters(3),
                                     track_times=False)
            else:
                a = h5f.get_node(statgroup, name=statkey, classname="Table")
            t_crt = t_crt + (timer() - t_crt_start)
            t_itr_start = timer()
            # create dataset row iterator
            a_row = a.row
            for r in a_row:
                r.append()
            statrows[statkey] = {'it': a_row, 'tb': a}

            t_itr = t_itr + timer() - t_itr_start

        t_stat_start = timer()

        for col, (dtype, offset) in dt.fields.items():
            # print("getting stats for table {} | col {} | dtype {}".format(tablepath, col,dtype))
            if col == 'num':
                stats = np.full(6, num)
            elif col in constant_cols:
                stats = np.full(6, tablenode.fields(col)[0])
            elif col == 'delta_t' and np.issubdtype(np.complex128, dtype):  # For constant complex values
                stats = np.full(6, tablenode.fields(col)[0].view(dtype=np.complex128))
            elif col == 'delta_t' and np.issubdtype('S128', dtype):  # For constant complex values as str
                stats = np.full(6, tablenode.fields(col)[0].astype(str))
            elif col == 'physical_time' and np.issubdtype('S64', dtype):
                t_gets_start = timer()
                vals = tablenode.fields(col)[()]
                stats = get_stats(data=vals.astype(np.float64))
                t_gets = t_gets + (timer() - t_gets_start)
            else:
                t_gets_start = timer()
                stats = get_stats(data=tablenode.fields(col)[()])
                t_gets = t_gets + (timer() - t_gets_start)

            for stat_tgt, stat_src in zip(statrows.values(), stats):
                stat_tgt['it'][col] = stat_src


        t_stat = t_stat + (timer() - t_stat_start)
        t_app_start = timer()
        for key, stat in statrows.items():
            stat['it'].append()
            stat['tb'].flush()
        t_app = t_app + timer() - t_app_start
    with h5py.File(tgt, 'a') as h5f:
        # Add model parameters as attributes to the fLBIT path
        flbit_path = tablepath[0: tablepath.index("fLBIT")] + "fLBIT"
        model_path = flbit_path + "/model"
        if model_path in tablenode.file:
            model_node = tablenode.file[model_path]
            flbit_node = h5f.require_group(flbit_path)
            for key, dset in model_node.items():
                if isinstance(dset, h5py.Group):
                    continue
                if '.db' in key or dset.dtype.type is np.void:  # Tables are np.void
                    continue
                if 'opdm-eigv' in key:
                    continue
                if key in flbit_node.attrs:
                    break
                # print(key,dset,model_node)
                flbit_node.attrs.create(name=key, data=dset)
        else:
            raise LookupError("Could not find [" + model_path + "] in ", tablenode.file)

    t_tot = timer() - t_tot
    print('table2: {} | '
          'tot {:8.3e} | '
          'pre {:8.3e} {:.1f}% | '
          'crt {:8.3e} {:.1f}% | '
          'itr {:8.3e} {:.1f}% | '
          'stat {:8.3e} {:.1f}% | '
          'gets {:8.3e} {:.1f}% | '
          'app {:8.3e} {:.1f}% | '
          .format(tablepath,
                  t_tot,
                  t_pre,
                  t_pre / t_tot * 100,
                  t_crt,
                  t_crt / t_tot * 100,
                  t_itr,
                  t_itr / t_tot * 100,
                  t_stat,
                  t_stat / t_tot * 100,
                  t_gets,
                  t_gets / t_tot * 100,
                  t_app,
                  t_app / t_tot * 100
                  ))


# @profile
def getmaxiter(iternode):
    numkeys = len(iternode.parent.keys())
    if 100 < numkeys < 105:
        numkeys=100
    if 200 < numkeys < 205:
        numkeys=200
    return numkeys
    # statuspath = '{}/tables/status'.format(iternode.name.split('cronos/', 1)[0])
    # statusnode = iternode.file[statuspath]
    # return statusnode['iter'][0]


@njit(cache=True)
def get_count(array1d, cutoff=1e-6):
    count = 0
    for val in array1d:
        if val >= cutoff:
            count += 1
    return count


@njit(parallel=False, cache=True)
def get_sum_power(array1d, alpha=1e-3, cutoff=1e-6):
    sum = 0
    for val in array1d:
        if val >= cutoff:
            sum += val ** alpha
    return sum


@njit(parallel=True, cache=True)
def get_renyi(data, alpha=1e-3, cutoff=1e-6):
    # Assume indices are n, site , time, realization

    sh = np.shape(data)
    re = np.zeros((sh[1], sh[2], sh[3]))
    for s in range(sh[1]):  # Site index
        for t in range(sh[2]):  # Time index
            for r in nb.prange(sh[3]):  # Realization index
                re[s, t, r] = 1.0 / (1 - alpha) * np.log(get_sum_power(data[:, s, t, r], alpha, cutoff))

    return re


@njit(parallel=True, cache=True)
def get_hartley(data, cutoff=1e-6):
    # Assume indices are n, site , time, realization
    sh = np.shape(data)
    ha = np.zeros((sh[1], sh[2], sh[3]))
    for s in range(sh[1]):  # Site index
        for t in range(sh[2]):  # Time index
            for r in nb.prange(sh[3]):  # Realization index
                ha[s, t, r] = np.log(get_count(data[:, s, t, r], cutoff=cutoff))
    return ha


def get_earray(h5f,
               where,
               name,
               dtype,
               shape,
               chunkshape=None,
               expectedrows=None):
    statpath = "{}/{}".format(where, name)
    if statpath in h5f:
        return h5f.get_node(where=where, name=name, classname="EArray")
    else:
        return h5f.create_earray(where=where,
                                 name=name,
                                 atom=tb.Atom.from_dtype(dtype),
                                 shape=shape,
                                 title="Midchain data",
                                 chunkshape=chunkshape,
                                 createparents=True,
                                 expectedrows=expectedrows,
                                 filters=tb.Filters(3),
                                 track_times=False)


def get_table(h5f,
              where,
              name,
              description,
              title=None,
              chunkshape=None,
              expectedrows=None):
    statpath = "{}/{}".format(where, name)
    if statpath in h5f:
        return h5f.get_node(where=where, name=name, classname="Table")
    else:
        return h5f.create_table(where=where,
                                name=name,
                                title=title,
                                description=description,
                                createparents=True,
                                chunkshape=None,
                                expectedrows=expectedrows,
                                filters=tb.Filters(3),
                                track_times=False)


# @profile
def write_statistics_crono4(nodemeta, crono_tables, h5f: tb.File, nodecache):
    # Props contains the names of the tables as keys, and each value contains either "ALL" or a list of desired columns
    # We should generate the statistics for each column in the table, averaging each time point over all realizations
    t_tot = timer()
    t_stat = 0
    t_gets = 0
    t_data = 0
    t_cols = 0
    t_read = 0
    t_crt = 0
    t_pre = 0
    t_itr = 0
    t_app = 0
    itername = nodemeta[0]  # The name of the iteration group, e.g. 'iter_1'
    iterpath = nodemeta[1]  # The path of the iteration group, e.g. '...u_3/r_8/fLBIT/state_real/tables/iter_1'
    iternode = nodemeta[2]  # The node of the iteration group
    iterval = int(itername.split('iter_', 1)[1])
    itermax = getmaxiter(iternode)

    if not isinstance(iternode, h5py.Group):
        raise TypeError("write_statistics_crono3: meta should point to a h5py.Group. Got: ", nodemeta)
    point_node = iternode.parent
    point_path = point_node.name
    if 'number_probabilities' in crono_tables:
        pn_node = point_node.parent['number_probabilities']
        if write_statistics_crono4.hartley_number_entropy_data is None or \
                write_statistics_crono4.number_probabilities_path is None or \
                write_statistics_crono4.number_probabilities_path != pn_node.name:
            pn_shape = np.shape(pn_node)
            print("Found number_probabilities: {}: {}".format(pn_shape, pn_node.name))
            pn_sites = [int(pn_shape[0] / 2)]
            print("Calculating renyi on shape {}, sites {}".format(np.shape(pn_node), pn_sites))
            write_statistics_crono4.hartley_number_entropy_data = np.squeeze(get_renyi(pn_node[:, pn_sites, :, :], alpha=1e-3, cutoff=1e-6))
            write_statistics_crono4.number_probabilities_path = pn_node.name

    fastforward = True
    tableiter = None

    for tablename, tablepath, tablenode in h5py_node_iterator(node=iternode, keypattern=crono_tables, dep=1,
                                                              nodeType=h5py.Dataset):
        t_pre_start = timer()
        # srciter = tablenode['iter'][0] # We can check if this source iteration is already in the target table (in which case skip)
        # We now have a source table
        table_columns = crono_tables[tablename]
        dt_num = get_dtype(tablenode=tablenode, req_columns=table_columns, num=True)
        t_pre = t_pre + timer() - t_pre_start
        statgroup = '{}/{}'.format(point_path, tablename)
        statrows = {}
        nbins = 60
        statelen = sum('L_' in s for s in dt_num.fields.keys())
        statemid = int(statelen / 2)

        # Create the tables/datasets
        for statkey in ['avg', 'std', 'ste', 'med', 'max', 'min']:
            t_crt_start = timer()
            statpath = '{}/{}/{}'.format(point_path, tablename, statkey)
            stattitle = '{} {}'.format(tablename, statkey)
            if not statgroup in nodecache:
                nodecache[statgroup] = {}
            nodecache[statgroup][statkey] = get_table(h5f=h5f,
                                                      where=statgroup,
                                                      name=statkey,
                                                      title=stattitle,
                                                      description=dt_num.newbyteorder('<'),
                                                      expectedrows=itermax)
            t_crt = t_crt + timer() - t_crt_start
            t_itr_start = timer()
            # create table row iterator
            # print("Writing to table: {}/{} | iter {}/{}".format(statgroup, statkey,iterval,itermax))
            a_row = nodecache[statgroup][statkey].row
            for r in a_row:
                r.append()
            statrows[statkey] = {'it': a_row, 'tb': nodecache[statgroup][statkey]}

            t_itr = t_itr + timer() - t_itr_start
        if fastforward and all(stat['tb'].nrows > 0 and stat['tb'][-1]['iter'] >= iterval for stat in statrows.values()):
            #print('crono4: table entry already exists: {}/{} | table rows {} | tableiter {} | {}'.format(iterval, itermax, nodecache[statgroup][statkey].nrows,tableiter, tablename))
            tableiter = nodecache[statgroup][statkey].nrows
            continue
        else:
            fastforward = False

        dt = get_dtype(tablenode=tablenode, req_columns=table_columns, num=False)
        t_read_start = timer()
        tabledata = tablenode.fields(dt.fields.keys())[()]
        t_read = t_read + (timer() - t_read_start)

        if len(tabledata) == 0:
            print("table is empty: {} | shape {}".format(tablepath, np.shape(tablepath)))
            continue
        t_stat_start = timer()
        for col, (dtype, offset) in dt_num.fields.items():
            if col == 'iter' and not tableiter:
                tableiter = tabledata[col][0]
            elif col == 'num':
                stats = np.full(6, len(tabledata))
            elif col == 'hartley_number_entropy':
                if not write_statistics_crono4.hartley_number_entropy_data:
                    continue
                pn_itr = statrows[statkey]['tb'].nrows
                stats = get_stats(data=write_statistics_crono4.hartley_number_entropy_data[pn_itr])
            elif col in constant_cols:
                stats = np.full(6, tabledata[col][0])
            elif col == 'delta_t' and np.issubdtype(np.complex128, dtype):  # For constant complex values
                stats = np.full(6, tabledata[col][0].view(dtype=np.complex128))
            elif col == 'delta_t' and  np.issubdtype('S128', dtype):
                vals = tablenode.fields(col)[()]
                vals = np.array(['{}+{}j'.format(*cplx.lstrip('(').rstrip(')').split(',')) for cplx in arr]).astype(
                    np.complex128)
            elif col == 'physical_time' and np.issubdtype('S64', dtype):
                t_gets_start = timer()
                vals = tablenode.fields(col)[0].astype(np.float64)
                stats = np.full(6, vals)
                # if not np.all(np.isclose(vals, vals)):
                #     raise ValueError(f"Mismatching time values:\n {vals}")
                # stats = get_stats(data=vals)
                t_gets += (timer() - t_gets_start)
            else:
                t_gets_start = timer()
                stats = get_stats(data=tabledata[col])
                t_gets += (timer() - t_gets_start)
                # Save midchain data for histograms
                if '__save_mid__' in crono_tables and tablename in crono_tables['__save_mid__'] and f'L_{statemid}' in col:
                    # print('saving column L_{} from table {}...'.format(statemid, tablename))
                    t_data_start = timer()
                    dataname = 'data'
                    datasize = len(tabledata[col])
                    nodecache[statgroup][dataname] = get_earray(
                        h5f=h5f,
                        where=statgroup,
                        name=dataname,
                        dtype=dtype,
                        shape=(0, datasize),
                        chunkshape=(25, datasize),
                        expectedrows=itermax)
                    nodecache[statgroup][dataname].append(np.asmatrix(tabledata[col]))
                    t_data += (timer() - t_data_start)
                    # print('saving column L_{} from table {}... done'.format(statemid, tablename))
                elif '__save_col__' in crono_tables and col in crono_tables['__save_col__']:
                    t_cols_start = timer()
                    dataname = col
                    datasize = len(tabledata[col])
                    # print(f'{dataname=} | {datasize=}')
                    # print('saving column {} from table {} to table {}...'.format(col, tablename, dataname))
                    nodecache[statgroup][dataname] = get_earray(
                        h5f=h5f,
                        where=statgroup,
                        name=dataname,
                        dtype=dtype,
                        shape=(0, datasize),
                        chunkshape=(10, datasize),
                        expectedrows=itermax)
                    if(datasize != np.shape(nodecache[statgroup][dataname])[1]):
                        raise AssertionError("Unexpected data size when saving column for\n"
                                             "statgroup: {}\n"
                                             "dataname: {}\n"
                                             ": {} != {}".format(statgroup, dataname, datasize, np.shape(nodecache[statgroup][dataname])[1]))
                    nodecache[statgroup][dataname].append(np.asmatrix(tabledata[col]))
                    # print('saving column {} from table {} to table {}... done'.format(col, tablename, dataname))
                    t_cols += (timer() - t_cols_start)
            for stat_tgt, stat_src in zip(statrows.values(), stats):
                stat_tgt['it'][col] = stat_src

        t_stat = t_stat + (timer() - t_stat_start)

        t_app_start = timer()
        for key, stat in statrows.items():
            stat['it'].append()
            # stat['tb'].flush()  ## not needed? becomes slower
            if not tableiter:
                tableiter = stat['it']['iter']
        t_app = t_app + timer() - t_app_start

    t_model_start = timer()
    model_path = iterpath[0: iterpath.index("LBIT")] + "LBIT/model"
    if model_path in iternode.file:
        model_node = iternode.file[model_path]
        group_path = model_node.parent.parent.name  # Path to the group that will get the attributes
        group_name = os.path.basename(group_path)  # Name of the group getting the attributes
        group_prnt = model_node.parent.parent.parent.name  # Path to the parent of group_path
        if not group_path in h5f:
            h5f.create_group(where=group_prnt, name=group_name, createparents=True)
        attrs_node = h5f.get_node(group_path, classname="Group")

        for key, dset in model_node.items():
            if isinstance(dset, h5py.Group):
                continue
            if any(x in key for x in ['.db', 'hamiltonian', 'unitary_circuit', 'opdm-eigv']):
                continue
            if key in attrs_node._v_attrs:
                break
            attrs_node._f_setattr(key, dset[()])
    else:
        raise LookupError("Could not find [" + model_path + "] in ", iternode.file)
    t_mod = timer() - t_model_start
    t_tot = timer() - t_tot
    print(f'crono4: {iterpath} | '
          f'tot {t_tot:8.3e} | '
          f'pre {t_pre:8.3e} {t_pre / t_tot * 100:.1f}% | '
          f'crt {t_crt:8.3e} {t_crt / t_tot * 100:.1f}% | '
          f'itr {t_itr:8.3e} {t_itr / t_tot * 100:.1f}% | '
          f'read {t_read:8.3e} {t_read / t_tot * 100:.1f}% | '
          f'stat {t_stat:8.3e} {t_stat / t_tot * 100:.1f}% | '
          f'gets {t_gets:8.3e} {t_gets / t_tot * 100:.1f}% | '
          f'data {t_data:8.3e} {t_data / t_tot * 100:.1f}% | '
          f'cols {t_cols:8.3e} {t_cols / t_tot * 100:.1f}% | '
          f'app {t_app:8.3e} {t_app / t_tot * 100:.1f}% | '
          f'mod {t_mod:8.3e} {t_mod / t_tot * 100:.1f}% | ')
    return itermax == tableiter
