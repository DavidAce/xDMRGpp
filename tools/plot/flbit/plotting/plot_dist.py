from itertools import product
from pathlib import Path

import h5py
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.optimize import curve_fit

from .tools import *


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


def plot_dist_v3_fig_sub_line(db, meta, figspec, subspec, linspec, algo_filter=None, state_filter=None, point_filter=None, figs=None, palette_name=None):
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
            palette, lstyles = get_colored_lstyles(db, linspec, palette_name)
            for linvals, color, lstyle in zip(linprod, palette, lstyles):
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

                    mmntnode = datanode.parent
                    ndata = mmntnode['avg']['num'][()]
                    if meta.get('dsetname') and meta.get('colname'):
                        ydata = mmntnode[meta['dsetname']][meta['colname']]
                    elif meta.get('dsetname') and meta.get('tidx'):
                        ydata = mmntnode[meta['dsetname']][meta['tidx']]
                    else:
                        raise LookupError('Unhandled dataset type')
                    if meta.get('normpage'):
                        ydata /= page_entropy(dbval['L'])
                    if normalize := meta.get('normalize'):
                        ydata /= normalize


                    hist, edges = np.histogram(ydata, bins=meta.get('bins'), density=meta.get('density'))
                    bincentres = [(edges[j] + edges[j + 1]) / 2. for j in range(len(edges) - 1)]
                    line, = ax.step(x=bincentres, y=hist, where='mid', label=None,
                                    color=color, path_effects=path_effects, linestyle=lstyle)
                    if 'number_' in meta['dsetname']:
                        ax.axvline(x=np.log(2), color='grey')
                        ax.axvline(x=np.log(3), color='darkseagreen')
                        # ax.set_xticks(list(ax.get_xticks()) + [np.log(2), np.log(3)])
                        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(np.log(2), 0.8, '$\ln 2$', fontsize='small', color='grey', ha='right', va='center', rotation='vertical',
                                transform=trans)
                        ax.text(np.log(3), 0.8, '$\ln 3$', fontsize='small', color='darkseagreen', ha='right', va='center', rotation='vertical',
                                transform=trans)
                    if 'algorithm_time' in meta['dsetname']:
                        t_mean = np.mean(ydata)
                        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        ax.axvline(x=t_mean, color=color)
                        ax.text(t_mean, 0.8, 'avg', fontsize='small', color='grey', ha='right', va='center', rotation='vertical',
                                transform=trans)
                    legendrow = get_legend_row(db=db, datanode=datanode, legend_col_keys=legend_col_keys)
                    for icol, (col, key) in enumerate(zip(legendrow, legend_col_keys)):
                        key, fmt = key.split(':') if ':' in key else [key, '']
                        f['legends'][idx][icol]['handle'].append(line)
                        f['legends'][idx][icol]['title'] = db['tex'][key]
                        f['legends'][idx][icol]['label'].append(col)


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
            suffix = suffix + '_normpage' if 'normpage' in meta and meta['normpage'] else suffix
            suffix = suffix + '_loglog' if meta.get('timeselection') == 'lnlnt' else suffix
            f['filename'] = "{}/{}_dist_fig({})_sub({}){}".format(meta['plotdir'], meta['plotprefix'],
                                                                  '-'.join(map(str, figvals)),
                                                                  '-'.join(map(str, get_keys(db, subspec))),
                                                                  suffix)

    return figs


def plot_dist_fig_sub_line(db, meta, figspec, subspec, linspec, algo_filter=None, state_filter=None, point_filter=None,
                           figs=None, palette_name=None):
    if db['version'] == 2:
        specs = figspec + subspec + linspec
        nonv3spec = lambda x: x not in ['cstd', 'tstd', 'tgw8', 'cgw8']
        specs = list(filter(nonv3spec, specs))
        fig3 = specs[:3]
        sub3 = specs[3:6]
        lin1 = specs[6:7]
        return plot_dist_v2_fig3_sub3_line1(db=db, meta=meta, figspec=fig3, subspec=sub3, linspec=lin1,
                                            algo_filter=algo_filter, state_filter=state_filter,
                                            point_filter=point_filter, figs=figs, palette_name=palette_name)
    elif db['version'] == 3:
        return plot_dist_v3_fig_sub_line(db=db, meta=meta, figspec=figspec, subspec=subspec, linspec=linspec,
                                         algo_filter=algo_filter, state_filter=state_filter, point_filter=point_filter,
                                         figs=figs, palette_name=palette_name)
    else:
        raise NotImplementedError('database version not implemented:' + db['version'])
