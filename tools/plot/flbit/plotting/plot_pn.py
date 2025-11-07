import os.path
from itertools import product
from pathlib import Path

import h5py
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.optimize import curve_fit

from .tools import *
from scipy.stats import gmean


# def floglog(x, a, b,c):
#     return a + b*np.log(np.log(x + c))
def floglog_v2(x, a, b, c):
    with np.errstate(invalid='ignore'):
        return a + b * np.log(np.log(x - c))


def flinear(x, a, b):
    with np.errstate(invalid='ignore'):
        return a + b * x


def fpower(x, a, b):
    with np.errstate(invalid='ignore'):
        return a * x ** b


def plot_pn_v3_fig_sub_line(db, meta, figspec, subspec, linspec, algo_filter=None, state_filter=None, point_filter=None, figs=None, palette_name=None):
    if 'mplstyle' in meta:
        plt.style.use(meta['mplstyle'])
    if 'mplstyle' in meta and 'slack' in meta['mplstyle']:
        # palette_name = "Spectral"
        if not palette_name:
            palette_name = "colorblind"
        path_effects = [pe.SimpleLineShadow(offset=(0.5, -0.5), alpha=0.3), pe.Normal()]
        # path_effects = None
    else:
        if not palette_name:
            palette_name = "colorblind"
        # path_effects = None
        path_effects = [pe.SimpleLineShadow(offset=(0.5, -0.5), alpha=0.3), pe.Normal()]

    prb_style = 'prb' in meta['mplstyle'] if 'mplstyle' in meta else False

    # legend_col_keys = list(itertools.chain(l1, [col for col in meta['legendcols'] if 'legendcols' in meta]))
    legend_col_keys = []
    if legendcols := meta['legendcols']:
        for col in legendcols:
            if not col in [l.split(':')[0] for l in subspec]:
                legend_col_keys.append(col)

    figprod = list(product(*get_vals(db=db, keyfmt=figspec, filter=meta.get('filter'))))  # All combinations of figspecs values
    subprod = list(product(*get_vals(db=db, keyfmt=subspec, filter=meta.get('filter'))))  # All combinations of subspecs values
    linprod = list(product(*get_vals(db=db, keyfmt=linspec, filter=meta.get('filter'))))  # All combinations of linspecs values
    numfigs = len(figprod)
    numsubs = len(subprod)
    if figs is None:
        figs = [get_fig_meta(numsubs, meta=meta) for _ in range(numfigs)]

    for figvals, f in zip(figprod, figs):
        logger.debug('- plotting figs: {}'.format(figvals))
        dbval = None
        for idx, (subvals, ax) in enumerate(zip(subprod, f['ax'])):
            popt = None
            pcov = None
            dbval = None
            logger.debug('-- plotting subs: {}'.format(subvals))
            for linvals in linprod:
                logger.debug('--- plotting lins: {}'.format(linvals))
                datanodes = match_datanodes(db=db, meta=meta, specs=figspec + subspec + linspec,
                                            vals=figvals + subvals + linvals)

                if len(datanodes) != 1:
                    logger.warning(f"match: \n"
                                   f"\tspec:{[figspec + subspec + linspec]}\n"
                                   f"\tvals:{[figvals + subvals + linvals]}")
                    logger.warning(f"found {len(datanodes)} datanodes: {datanodes=}")
                    continue
                for datanode in datanodes:
                    dbval = db['dsets'][datanode.name]
                    ndata = np.shape(datanode)[3]
                    L     = np.shape(datanode)[1] - 1
                    # if L != 28:
                    #     continue
                    mid   = np.shape(datanode)[1] // 2
                    mmntnode = datanode.parent['cronos']['measurements']
                    physical_time = mmntnode['avg']['physical_time'][()].astype(float)
                    # print(physical_time)
                    # exit(0)
                    # modelnode = datanode.parent.parent['model']
                    if tfracs := meta.get('tfracs', None):
                        tidx_max = np.shape(datanode)[2]
                        tidx = [int(tf * (tidx_max-1)) for tf in tfracs]
                    else:
                        tidx = meta['tidx']
                    tidx=range(0, len(physical_time), 2)
                    print(f'{tidx=}')
                    print(f'{physical_time[tidx]=}')
                    cachepath = f'{db['cachedir']}/pn-dist.h5'
                    pndata = None
                    if os.path.exists(cachepath):
                        try:
                            with h5py.File(cachepath, 'r') as cf:
                                if datanode.name in cf:
                                    print(f'Loading cache: {cachepath} | {datanode.name}')
                                    pndata = cf[datanode.name][()]
                                else:
                                    print(f'Dataset not in cache: {cachepath} | {datanode.name}')
                                    pndata = None
                        except Exception as e:
                            print(f'Could not load from cache: {cachepath} | {datanode.name}\n Exception: {e}')
                            pndata = None


                    if pndata is None:
                        pndata = datanode[:, mid, :, :]
                        with h5py.File(cachepath, 'a') as cf:
                            groupname = os.path.dirname(datanode.name)
                            dsetname = os.path.basename(datanode.name)
                            print(f'{groupname=}')
                            print(f'{dsetname=}')
                            tgt_node = cf.require_group(f'{groupname}')
                            if not dsetname in tgt_node:
                                tgt_node.create_dataset(name=dsetname, data=pndata, dtype=np.float64, compression="gzip", compression_opts=1 )
                    pndata = pndata[:, tidx, :]
                    neel_type = datanode.parent['neel_type']

                    indices_of_ones = [index for index, value in enumerate(neel_type) if value == 1]
                    indices_of_zero = [index for index, value in enumerate(neel_type) if value == 0]
                    print(f'{np.shape(pndata)=}')
                    print(f'{np.shape(tidx)=}')
                    print(f'{np.shape(indices_of_ones)=}')
                    # pn0 = np.mean(pndata[:, :, indices_of_zero], axis=2)
                    # pn1 = np.mean(pndata[:, :, indices_of_ones], axis=2)
                    # pnn = np.mean(pndata[:, :, :], axis=2)
                    print(f'{np.shape(pndata[:, :, indices_of_zero])=}')

                    # pn0 = np.mean(1e-15+pndata[:, :, indices_of_zero], axis=2)
                    # pn1 = np.mean(1e-15+pndata[:, :, indices_of_ones], axis=2)
                    pnn = gmean(1e-17+pndata[:, :, :], axis=2)
                    times = physical_time[tidx]
                    legendrow = get_legend_row(db=db, datanode=datanode, legend_col_keys=legend_col_keys)
                    palette = sns.color_palette(palette='viridis_r', n_colors=len(times))
                    for ti, color in zip(range(np.shape(pndata)[1]), palette):
                        # line, = ax.plot(pn0[:, ti], label=None, color=color, path_effects=path_effects, linestyle=lstyle)
                        # line, = ax.plot(pn1[:, ti], label=None, color=color, path_effects=path_effects, linestyle=lstyle)
                        line, = ax.plot(pnn[:, ti], label=None, color=color, path_effects=path_effects, linestyle=None)

                        for icol, (col, key) in enumerate(zip(legendrow, legend_col_keys)):
                            key, fmt = key.split(':') if ':' in key else [key, '']
                            f['legends'][idx][icol]['handle'].append(line)
                            f['legends'][idx][icol]['title'] = db['tex'][key]
                            f['legends'][idx][icol]['label'].append(col)

                        f['legends'][idx][icol + 1]['handle'].append(line)
                        f['legends'][idx][icol + 1]['title'] = '$t$'
                        f['legends'][idx][icol + 1]['label'].append(f'${times[ti]:.1e}$')
                    # hist, edges = np.histogram(ydata, bins=meta.get('bins'), density=meta.get('density'))
                    # bincentres = [(edges[j] + edges[j + 1]) / 2. for j in range(len(edges) - 1)]
                    # line, = ax.step(x=bincentres, y=hist, where='mid', label=None,
                    #                 color=color, path_effects=path_effects, linestyle=lstyle)
                    textstr = (f'Geometric mean of {np.shape(pndata)[2]} samples\n'
                               f'Mixing factor: $f=0.4$\n'
                               f'Initial state: Neel (both types)\n'
                               f'Time regimes: \n'
                               f'   Yellow $S_N \sim t$\n'
                               f'   Green  $S_N \sim \\ln\\ln t$\n'
                               f'   Blue   $S_N \sim $ constant'
                               # f'Note: Noisy tails due to arithmetic mean'
                               )
                    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)


                    if not idx in f['axes_used']:
                        f['axes_used'].append(idx)

            if axtitle := get_default(meta, 'axtitle'):
                if dbval and isinstance(axtitle, bool):
                    axtitle = get_title(dbval, subspec)
                ax.set_title(axtitle,horizontalalignment='left', x=0.05,fontstretch="ultra-condensed")

        if figspec_title := get_figspec_title(meta, dbval, figspec):
            f['fig'].suptitle(figspec_title)

        # prettify_plot4(fmeta=f, lgnd_meta=axes_legends)
        if not f['filename']:
            suffix = ''
            f['filename'] = "{}/{}_pn_fig({})_sub({}){}".format(meta['plotdir'], meta['plotprefix'],
                                                                  '-'.join(map(str, figvals)),
                                                                  '-'.join(map(str, get_keys(db, subspec))),
                                                                  suffix)

    return figs


def plot_pn_fig_sub_line(db, meta, figspec, subspec, linspec, algo_filter=None, state_filter=None, point_filter=None,
                           figs=None, palette_name=None):
    if db['version'] == 3:
        return plot_pn_v3_fig_sub_line(db=db, meta=meta, figspec=figspec, subspec=subspec, linspec=linspec,
                                         algo_filter=algo_filter, state_filter=state_filter, point_filter=point_filter,
                                         figs=figs, palette_name=palette_name)
    else:
        raise NotImplementedError('database version not implemented:' + db['version'])
