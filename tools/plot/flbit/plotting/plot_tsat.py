import os.path
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,MaxNLocator
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats.mstats import gmean
import h5py
from .tools import *
from itertools import islice
from scipy.signal import savgol_filter,find_peaks, find_peaks_cwt
from scipy import interpolate
import json
from dataclasses import asdict


def format_err(val, err, min_decimals=None):
    err_exponent =  np.floor(np.log10(err))
    if not np.isinf(err_exponent):
        num_decimals =  int(np.abs(err_exponent))
    else:
        num_decimals = 0
    if min_decimals is not None:
        num_decimals = min_decimals
    return f'{val:.{num_decimals}f}({err*10**num_decimals:1.0f})'

# def floglog(x, a, b,c):
#     return a + b*np.log(np.log(x + c))
def floglog_v2(x, a, b, c):
    with np.errstate(invalid='ignore'):
        return a + b * np.log(np.log(x - c))


def flinear(x, a, b):
    with np.errstate(invalid='ignore'):
        return a + b * x

def fexplin(x, a, b):
    with np.errstate(invalid='ignore'):
        return np.log(a) + 1.0/b * x

def fexp(x, a, b):
    return a * np.exp(x/b)


def fpower(x, a, b):
    with np.errstate(invalid='ignore'):
        return a * x ** b

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def plot_v3_tsat_fig_sub_line(db, meta, figspec, subspec, linspec, xaxspec, algo_filter=None, state_filter=None, point_filter=None, figs=None, palette_name=None,dbidx=0,dbnum=1):
    path_effects = [pe.SimpleLineShadow(offset=(0.35, -0.35), alpha=0.4), pe.Normal()]
    if 'mplstyle' in meta:
        plt.style.use(meta['mplstyle'])
        if 'slack' in meta['mplstyle']:
            path_effects = [pe.SimpleLineShadow(offset=(0.5, -0.5), alpha=0.3), pe.Normal()]
    mark_effects = [pe.SimpleLineShadow(offset=meta.get('shadowoffset', (0.35, -0.35)), alpha=0.4), pe.Normal()]

    # legend_col_keys = list(itertools.chain(l1, [col for col in meta['legendcols'] if 'legendcols' in meta]))
    legend_col_keys = linspec.copy()
    if legendcols := meta['legendcols']:
        for col in legendcols:
            c = col.split(':')[0] # Put a ? in the fmt string to skip
            if not c in [l.split(':')[0] for l in figspec + subspec + linspec + xaxspec]:
                legend_col_keys.append(col)

    plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
    figprod = list(product(*get_vals(db=db, keyfmt=figspec, filter=meta.get('filter'))))  # All combinations of figspecs values
    subprod = list(product(*get_vals(db=db, keyfmt=subspec, filter=meta.get('filter'))))  # All combinations of subspecs values
    linprod = list(product(*get_vals(db=db, keyfmt=linspec, filter=meta.get('filter'))))  # All combinations of linspecs values
    xaxprod = list(product(*get_vals(db=db, keyfmt=xaxspec, filter=meta.get('filter'))))  # All combinations of linspecs values
    numfigs = len(figprod)
    numsubs = len(subprod)

    numsubs *= dbnum
    subprod *= dbnum

    if figs is None:
        figs = [get_fig_meta(numsubs, meta=meta) for _ in range(numfigs)]

    for figvals, f in zip(figprod, figs):
        logger.debug('- plotting figs: {}'.format(figvals))
        dbval = None
        luitz = []
        for idx, (subvals, ax) in enumerate(zip(subprod, f['ax'])):
            idx_palette = idx % len(palette_name)
            if dbidx != int(idx / dbnum):
                continue
            popt = None
            pcov = None
            t_isect, t_slope = None, None

            dbval = None
            logger.debug('-- plotting subs: {}'.format(subvals))
            for lidx, linvals in enumerate(linprod):
                logger.debug('--- plotting lins: {}'.format(linvals))
                xvals, tvals_phys, tvals_boot, evals_boot = [], [], [], []
                legendrow = None
                for xaxvals in xaxprod:
                    logger.debug('--- plotting xaxs: {}'.format(xaxvals))
                    datanodes = match_datanodes(db=db, meta=meta, specs=figspec + subspec + linspec + xaxspec,
                                                vals=figvals + subvals + linvals + xaxvals)
                    if len(datanodes) != 1:
                        logger.warning(f"match: \n"
                                       f"\tspec:{[figspec + subspec + linspec]}\n"
                                       f"\tvals:{[figvals + subvals + linvals]}")
                        logger.warning(f"found {len(datanodes)} datanodes: {datanodes=}")
                    for datanode in datanodes:
                        # print(f'found datanode: {datanode}')
                        dbval = db['dsets'][datanode.name]
                        L = dbval['vals']['L']
                        # print(datanode)
                        # print(datanode.parent)
                        mmntnode = datanode.parent
                        # Get this datapoint
                        t = mmntnode['avg']['physical_time'][()].astype(float)
                        # s = datanode[meta['dsetname']][()] # Should be an entropy
                        s = mmntnode[meta['dsetname']][()] # Should be an entropy
                        n = mmntnode['avg']['num'][()][0]
                        if meta.get('normpage'):
                            s /= midchain_page_entropy(dbval['vals']['L'])
                        elif meta.get('normsize'):
                            s /= dbval['vals']['L']
                        elif normalize := meta.get('normalize'):
                            s /= normalize
                        # Calculate the infinite time average (1/T) integral_0^T y(t) dt
                        # reals = np.shape(y)[1]
                        # off = int(reals / 2)
                        # ext = int(reals / 5)
                        # Plot the saturation time
                        if 'saturated' in meta['timepoints']:
                            filenamejson = "{}/tsat_{}|{}_L[{}]_x[{}]_w[{}]_r[{}]_f[{}]_l[{}].json".format(
                                dbval['vals']['cachedir'],
                                dbval['vals']['filename'].replace('/', '|'),
                                meta['dsetname'],
                                dbval['vals']['L'],
                                dbval['vals']['x'],
                                dbval['vals']['w'],
                                dbval['vals']['r'],
                                dbval['vals']['f'],
                                dbval['vals']['l'])

                            if meta.get('loadjson') and os.path.isfile(filenamejson):
                                with open(filenamejson, 'r') as fp:
                                    tsb_json = json.load(fp)
                                    tsb = entropy_saturation_bootstrap(**tsb_json)
                            else:
                                # tboot_idx_avg, tboot_idx_err, sboot_avg, sboot_err = get_entropy_saturation_from_bootstrap(ydata=s, nbs=100)
                                tsb = find_entropy_inftime_saturation_value_from_bootstrap(sdata=s, tdata=t, nbs=meta.get('num-bootstraps', 100))
                                if meta.get('savejson'):
                                    if not os.path.exists(dbval['vals']['cachedir']):
                                        os.makedirs(dbval['vals']['cachedir'])
                                    with open(filenamejson, 'w') as fp:
                                        json.dump(asdict(tsb), fp, indent=4)
                            tboot = tsb.tsat_boot_avg
                            eboot = tsb.tsat_boot_err

                        t = get_timepoints(t,dbval)
                        tphys = t.time_num_saturated if 'number' in meta['dsetname'] else t.time_ent_saturated

                        tvals_phys.append(tphys)
                        tvals_boot.append(tboot)
                        evals_boot.append(eboot)
                        xvals.append(get_vals(dbval, xaxspec))
                        # nvals.append(n)

                        idx_num, idx_ent = t.idx_num_saturated, t.idx_ent_saturated
                        idx_sat = idx_num if 'number' in meta['dsetname'] else idx_ent
                        # idx_sat = idx_ent
                        idx_len = len(t.time)
                        print(f'Saturation for {L=}: at {idx_num=} {idx_ent=}  ({idx_len}) total): time averaging {idx_len-idx_sat} points')


                xvals = np.ravel(xvals)
                evals_boot = np.atleast_2d(evals_boot)
                # tvals_phys = np.ravel(tvals_phys)
                # tvals_boot = np.ravel(tvals_boot)
                # evals_boot = np.ravel(evals_boot)
                # nvals = np.ravel(nvals)
                axn = ax
                if meta.get('ytwinx'):
                    if lidx == 0:
                        axn.annotate('', xy=(0.35, 0.85), xytext=(0.50, 0.85), xycoords='axes fraction',
                                     arrowprops=dict(color=color, arrowstyle='->'))
                    else :
                        axn = ax.twinx()
                        axn.set_box_aspect(1)
                        # axn.set_visible(True)
                        # axn.axis('off')
                        axn.set_frame_on(False)
                        axn.yaxis.set_visible(True)
                        axn.annotate('', xy=(0.65, 0.85), xytext=(0.50,0.85), xycoords='axes fraction',
                                arrowprops=dict(color=color, arrowstyle='->'))

                        # l = ax.get_ylim()
                        # ln = yvals.min(), yvals.max()
                        # fun = lambda x: ln[0] + (x - l[0]) / (l[1] - l[0]) * (ln[1] - ln[0])
                        # ticks = fun(ax.get_yticks())
                        axn.yaxis.set_major_locator(MaxNLocator(nbins=3))



                line2 = None
                palette, lstyles = get_colored_lstyles(db, xaxspec, palette_name, filter=None, idx=idx_palette)
                # color = palette[-1]
                # lstyle = lstyles[-1]
                print('xvals:', xvals)
                print('tboot:',tvals_boot)
                # print('eboot:',evals_boot)
                # line, = ax.plot(xvals, tvals_phys, marker=None, color='black', path_effects=None)
                # for xval, tval, color in zip(xvals, tvals_phys, palette):
                #     axn.plot(xval, tval, color=color, markersize=5, marker='o', markerfacecolor='none', markeredgewidth=0.5)
                line, = ax.plot(xvals, tvals_boot, marker=None,linestyle='none', color='black', path_effects=None)
                marker='o' if 'entanglement_entropy' in meta['dsetname'] else 's'
                for xval, tval, eval, color in zip(xvals, tvals_boot, evals_boot, palette):
                    axn.plot([xval], [tval], color=color, marker=marker, markersize=4.0,
                             linestyle='None',
                             markeredgecolor=color,
                             markerfacecolor='none',
                             markeredgewidth=0.6,
                             path_effects=mark_effects,
                             zorder=10)
                    axn.errorbar(x=xval, y=tval, yerr=np.atleast_2d(eval).T, color=color, linestyle='none', capsize=5.0,
                                 marker=None,
                                 path_effects=path_effects, zorder=9)

                if meta.get('fit-tsat') and len(xvals) > 1:

                    tlog = np.log(tvals_boot)
                    print(evals_boot)
                    elog = np.log([np.mean([e1,e2]) for e1,e2 in evals_boot])
                    print(elog)
                    popt, pcov = curve_fit(f=fexplin, xdata=xvals, ydata=tlog, sigma=elog)
                    print(f'{popt=}')
                    print(f'{np.sqrt(np.diag(pcov))=}')

                    tfit = np.exp(fexplin(xvals, *popt))
                    line, = ax.plot(xvals, tfit, marker=None, linewidth=0.4,
                            linestyle='--', label='fit', color='black',
                            path_effects=path_effects, zorder=1)

                    t_isect, t_slope = popt[0], popt[1]
                    t_isect_err , t_slope_err = np.sqrt(np.diag(pcov))
                    print(t_isect_err, t_slope_err)
                    # Annotate
                    xmid = 22 #np.mean(xvals)
                    ymid = np.exp(fexplin(np.asarray([xmid]), *popt))
                    xtxt_fr = xmid * 1.00
                    ytxt_fr = np.exp(fexplin(np.asarray([xtxt_fr]), *popt))
                    xtxt_to = xmid * 1.15
                    ytxt_to = ymid * 0.01
                    # entropy_symbol = '$t_{\mathrm{sat}}(\overline S_\mathrm{N})$' if 'number' in meta['dsetname'] else '$t_{\mathrm{sat}}(\overline S_\mathrm{E})$'
                    entropy_symbol = '$\overline S_\mathrm{N}$' if 'number' in meta['dsetname'] else '$\overline S_\mathrm{E}$'
                    ax.annotate(entropy_symbol,
                                xy=(xtxt_fr, ytxt_fr),
                                xytext=(xtxt_to, ytxt_to),
                                arrowprops=dict(arrowstyle="->", color='black'),
                                )


                if not datanode:
                    continue
                legendrow = get_legend_row(db=db, datanode=datanode, legend_col_keys=legend_col_keys)
                icol = 0
                for col, key in zip(legendrow, legend_col_keys):
                    key, fmt = key.split(':') if ':' in key else [key, '']
                    if '?' in fmt:
                        continue
                    f['legends'][idx][icol]['handle'].append(line)
                    f['legends'][idx][icol]['title'] = db['tex'][key]
                    f['legends'][idx][icol]['label'].append(col)
                    icol += 1
                if t_isect is not None and t_slope is not None:
                    # entropy_symbol = '\overline S_\mathrm{N}' if 'number' in meta['dsetname'] else '\overline S_\mathrm{E}'
                    # entropy_symbol = 't(\overline S_\mathrm{N})' if 'number' in meta['dsetname'] else 't(\overline S_\mathrm{E})'
                    entropy_symbol = '\overline{S}_\mathrm{N}' if 'number' in meta['dsetname'] else '\overline{S}_\mathrm{E}'
                    # entropy_symbol = 't_{\square}(\overline{S}_\mathrm{N})' if 'number' in meta['dsetname'] else 't_{\mbox{\large$\circ$}}(\overline{S}_\mathrm{E})'
                    f['legends'][idx][icol]['handle'].append(line)
                    f['legends'][idx][icol]['title'] = 'Fit $t(\overline S_{\mathrm{E|N}}) = c e^{L/\\xi}$'
                    # f['legends'][idx][icol]['label'].append(f'${entropy_symbol} = {t_isect:.3f}  e^{{(L/{t_slope:.3f})}}$')
                    # f['legends'][idx][icol]['label'].append(f'${entropy_symbol} : c={t_isect:.3f}\pm {t_isect_err:.3f},  \\xi={t_slope:.3f}\pm {t_slope_err:.3f}$')
                    f['legends'][idx][icol]['label'].append(f'${entropy_symbol} : c={format_err(t_isect, t_isect_err)},  \\xi={format_err(t_slope, t_slope_err)}$')

                    # f['legends'][idx][icol]['handle'].append(line)
                    # f['legends'][idx][icol]['title'] = 'RPS' if dbidx==0 else 'Néel'
                    # f['legends'][idx][icol]['label'].append('$\overline S_\mathrm{N}^\infty$')
                    #
                    # f['legends'][idx][icol]['handle'].append(line2)
                    # f['legends'][idx][icol]['title'] = 'RPS' if dbidx==0 else 'Néel'
                    # f['legends'][idx][icol]['label'].append('$\overline S(p_\mathrm{N}^\infty)$')



                if not idx in f['axes_used']:
                    f['axes_used'].append(idx)

            if axtitle := get_default(meta, 'axtitle'):
                if dbidx == 0:
                    if isinstance(axtitle, bool):
                        axtitle = get_title(dbval, subspec)
                    ax.set_title(axtitle,horizontalalignment='center', x=0.5,fontstretch="ultra-condensed")
            # if dbval:
            #     ax.set_title(get_title(dbval, subspec),
            #                  horizontalalignment='left', x=0.05,
            #                  fontstretch="ultra-condensed",
            #                  # bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.6)
            #                  )


        if figspec_title := get_figspec_title(meta, dbval, figspec):
            f['fig'].suptitle(figspec_title)
        # prettify_plot4(fmeta=f, lgnd_meta=axes_legends)
        if filename := meta.get('filename'):
            f['filename'] = f"{meta['plotdir']}/{filename}"
        else:
            if not f['filename']:
                suffix = ''
                suffix = suffix + '_normpage' if 'normpage' in meta and meta['normpage'] else suffix
                f['filename'] = "{}/{}_tsat_fig({})_sub({}){}".format(meta['plotdir'], meta['plotprefix'],
                                                                      '-'.join(map(str, figvals)),
                                                                      '-'.join(map(str, get_keys(db, subspec))),
                                                                      suffix)
            suffix = ''
            suffix = suffix + '_normpage' if 'normpage' in meta and meta['normpage'] else suffix
            f['filename'] = "{}/{}_fig({})_sub({}){}".format(meta['plotdir'], meta['plotprefix'],
                                                           get_specvals(db, figspec, figvals),
                                                           get_specvals(db, subspec), suffix)
    return figs



def plot_tsat_fig_sub_line(db, meta, figspec, subspec, linspec, xaxspec, algo_filter=None, state_filter=None,
                           point_filter=None, figs=None, palette_name=None,dbidx=0,dbnum=1):
    if db['version'] == 2:
        raise ValueError("version 2 not supported")
    elif db['version'] == 3:
        return plot_v3_tsat_fig_sub_line(db=db, meta=meta, figspec=figspec, subspec=subspec, linspec=linspec,
                                         xaxspec=xaxspec,
                                         algo_filter=algo_filter, state_filter=state_filter, point_filter=point_filter,
                                         figs=figs, palette_name=palette_name,dbidx=dbidx,dbnum=dbnum)
    else:
        raise NotImplementedError('database version not implemented:' + db['version'])
