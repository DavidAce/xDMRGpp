import os
import matplotlib.pyplot
import numba as nb
import numpy as np
from numba import njit, prange, float64,int64,boolean,optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams, legend
from copy import deepcopy
import itertools
from packaging import version
import logging
import matplotlib.gridspec as gs
from git import Repo
import seaborn as sns
from itertools import product
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import find_peaks
from collections.abc import Iterable
from dataclasses import dataclass, field
from random import choices
import json

logger = logging.getLogger('tools')
from pathlib import Path

# matplotlib_version = pkg_resources.get_distribution("matplotlib").version
# if version.parse(matplotlib_version) == version.parse("3.5"):
#     logger.warning("Matplotlib version {} may not work correctly with gridspec".format(matplotlib_version))


def get_markerlist():
    return itertools.cycle(('^', 'v', '<', '>', 'o'))


def get_linestyles():
    return itertools.cycle(("-", "--", "-.", ":"))


def get_uniform_palette_names(num):
    if num > 6:
        raise ValueError("num == {} must be smaller than 7".format(num))
    palettes = [
        'Blues',  # sequential from matplotlib
        'Greens',  # sequential from matplotlib
        'Oranges',  # sequential from matplotlib
        'Purples',  # sequential from matplotlib
        'Reds',  # sequential from matplotlib
        'crest',  # perceptually uniform from seaborn
        'flare',  # perceptually uniform from seaborn
        'magma_r',  # perceptually uniform from matplotlib (reversed)
        'viridis_r',  # perceptually uniform from matplotlib (reversed)
        'dark:salmon_r',
    ]

    return palettes[0:num]


def get_colored_lstyles(db, specs, default_palette, filter=None, idx = None):
    print(f"{default_palette=}")
    linprod = list(product(*get_vals(db, specs, filter)))  # All combinations of linspecs (names of parameters that iterate lines)
    lstyles = [None] * len(linprod)
    if idx is None:
        idx = 0
    if isinstance(default_palette, list):
        slist = [get_vals(db=db, keyfmt=spec, filter=filter) for spec in specs]
        slens = [len(llist) for llist in slist]
        if len(specs) == 2 and all([sl > 1 for sl in slens]):
            # The number of palettes == size of the first linspec.
            # The number of colors in each palette == the size of the second linspec
            palette_prod = []
            for pidx in range(slens[0]):
                if any([x in default_palette[pidx] for x in ['viridis', 'autumn','cool', 'Dark2', 'tab10'] ]):
                    # Colors at the edge of the palette are sometimes missing, so here we force them to appear
                    cmap = plt.colormaps[default_palette[pidx]]
                    palette = [cmap(i) for i in np.linspace(0, cmap.N, slens[1], endpoint=True, dtype=int)]
                    palette_prod.extend(palette)
                else:
                    # Skip the first color that is usually too dark or bright
                    palette_prod.extend(sns.color_palette(palette=default_palette[pidx], n_colors=slens[1]+1)[1:] )
            return palette_prod, lstyles
        else:
            default_palette = default_palette[idx]

    if any([x in default_palette for x in ['viridis', 'autumn', 'cool', 'Dark2', 'tab10']]):
        # Colors at the edge of the palette are sometimes missing, so here we force them to appear
        cmap = plt.colormaps[default_palette]
        palette = [cmap(i) for i in np.linspace(0, cmap.N, len(linprod), endpoint=True, dtype=int)]
        # palette = sns.color_palette(palette=default_palette, n_colors=len(linprod))
    else:
        # Skip the first color that is usually too dark or bright
        palette = sns.color_palette(palette=default_palette, n_colors=len(linprod)+1)[1:]
    # palette = sns.color_palette(palette=default_palette, n_colors=len(linprod))
    # print(linprod)
    # palette = sns.color_palette(palette=default_palette, n_colors=(10*len(linprod)+1))[::10]
    lstyles = [None] * len(linprod)
    if len(specs) == 2:
        linkey0 = get_keys(db, specs[0])  # Sets number of colors
        linkey1 = get_keys(db, specs[1])  # Sets number of linestyles
        if len(linkey0) == 2:
            palette = reversed(sns.color_palette(palette='tab20', n_colors=len(linprod)))
        if len(linkey0) == 3:
            palette = reversed(sns.color_palette(palette='tab20c', n_colors=len(linprod)))
            # del palette[4 - 1::4]
        if len(linkey0) == 4:
            palette = reversed(sns.color_palette(palette='tab20c', n_colors=len(linprod)))

    return palette, lstyles

def get_default(d, key, defkey='default'):
    val = d.get(key, d.get(defkey).get(key))
    return None if val is False else val

def get_subspec_title(meta, dbval, subspec):
    if subspec_title := get_default(meta, 'subspec_title'):
        if isinstance(subspec_title, bool) and dbval:
            return get_title(dbval, subspec)
        else:
            return subspec_title
    return None
def get_figspec_title(meta, dbval, figspec):
    titlename = get_default(meta,'titlename')
    titlespec = get_default(meta, 'figspec_title')
    if titlespec is True and dbval is not None:
        titlespec = get_title(dbval, figspec)
    title = '{}{}{}'.format(
        titlename if isinstance(titlename, str) else '',
        '\n' if isinstance(titlename, str) and isinstance(titlespec, str) else '',
        titlespec if isinstance(titlespec, str) else ''
        )
    return title if title != '' else None

def match_datanodes(db, meta, specs, vals):
    nodes = set()
    gname = meta.get('groupname')
    dname = meta.get('dsetname')
    for dsetpath, dset in db['dsets'].items():
        # print('-- checking', dsetpath)
        if gname is not None:
            if isinstance(gname, Iterable) and not any([g in dsetpath for g in gname]):
                continue
            elif not gname in dsetpath:
                continue
        if dname is not None:
            if isinstance(dname, Iterable) and not any([d in dsetpath for d in dname]):
                continue
            # print(f'{dname=}')
            # print(f'{dset=}')
            # if not dname in dset['node']['data'].name and not dname in dset['node']['data']:
            if not dname in dset['node']['data'].name:
                    continue
        equal = True
        for s, v in zip(specs, vals):
            s_noformat = s.split(':')[0]
            d = dset['vals'][s_noformat]
            s_iterable = isinstance(s, Iterable) and not isinstance(s, str)
            v_iterable = isinstance(v, Iterable) and not isinstance(v, str)
            d_iterable = isinstance(d, Iterable) and not isinstance(d, str)
            if d_iterable and v_iterable:
                equal = all([x == y for x, y in zip(d, v)])
            else:
                equal = d == v
            if not equal:
                break
        if not equal:
            continue
        nodes.add(dset['node']['data'])
    return nodes


def write_attributes(*args, **kwargs):
    for a in args:
        print(a)
    for k, v in kwargs.items():
        print("%s = %s" % (k, v))


def get_optimal_subplot_num(numplots):
    if numplots == 3:
        return 1, 3
    r = np.sqrt(numplots)
    cols = int(np.ceil(r))
    rows = int(np.floor(r))
    while cols * rows < numplots:
        if (cols <= rows):
            cols = cols + 1
        else:
            rows = rows + 1
    return rows, cols


def get_empty_figs_w_subplots(num_figs, num_subfigs, figsize=3.5):
    figures = []
    axes = []
    rows, cols = get_optimal_subplot_num(num_subfigs)
    for i in range(num_figs):
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize * cols, figsize * rows), num=i)
        figures.append(fig)
        axes.append(ax)
    return figures, axes


def remove_empty_subplots(fig, axes, axes_used):
    for idx, ax in enumerate(np.ravel(axes)):  # Last one is for the legend
        if not idx in axes_used:
            fig.delaxes(ax)




class stretchedFit:
    def __init__(self):
        pass
    pos : int = -1
    # @njit(parallel=True, cache=True)
    def stretched_exp(self, x, C, xi, beta):
        return C * np.exp(-(np.abs(x-self.pos) / xi) ** beta)

    # @njit(parallel=True, cache=True)
    def stretched_log(self, x, C, xi, beta):
        return np.log(C) - (np.abs(x-self.pos) / xi) ** beta

class unstretchedFit:
    def __init__(self):
        pass
    pos : int = -1
    # @njit(parallel=True, cache=True)
    def unstretched_exp(self, x, C, xi):
        return C * np.exp(-(np.abs(x-self.pos) / xi))

    # @njit(parallel=True, cache=True)
    def unstretched_log(self, x, C, xi):
        return np.log(C) - (np.abs(x-self.pos) / xi)



@njit(parallel=True, cache=True)
def stretched_exp(x, C, xi, beta, pos):
    return C * np.exp(-(np.abs(x-pos) / xi) ** beta)


@njit(parallel=True, cache=True)
def stretched_log(x, C, xi, beta, pos):
    return np.log(C) - (np.abs(x-pos) / xi) ** beta


@njit(parallel=True, cache=True)
def regular_exp(x, C, xi):
    return C * np.exp(-(x / xi))


@njit(parallel=True, cache=True)
def linear_log(x, C, xi):
    return np.log(C) - (x / xi)


@njit(parallel=True, cache=True)
def linear_fit(x, m, k):
    return m + k * x


# def floglog(x, a, b,c):
#     return a + b*np.log(np.log(x + c))
@njit(parallel=True, cache=True)
def floglog_v2(x, a, b, c):
    with np.errstate(invalid='ignore'):
        return a + b * np.log(np.log(x - c))


@njit(parallel=True, cache=True)
def fpower(x, a, b):
    with np.errstate(invalid='ignore'):
        return a * x ** b


# @njit(parallel=True, cache=True)
def flinear(x, a, b):
    with np.errstate(invalid='ignore'):
        return a + b * x

@njit(cache=True, parallel=False)
def find_peaks_fast(signal, height=None, peakwidth=1):
    size = len(signal)
    peaks = np.zeros((size),dtype=np.float64) * np.nan
    for i in prange(peakwidth,size-peakwidth):
        if signal[i] <= np.max(signal[i-peakwidth:i]) or signal[i] <= np.max(signal[i + 1:i + peakwidth + 1]):
            continue
        if height is not None and signal[i] <= height[i]:
            continue
        peaks[i] = signal[i]
    peaki = np.where(np.isfinite(peaks))[0]
    peaks = peaks[peaki]

    return peaki, peaks


@njit(cache=True, parallel=True)
def get_every_realization_minmax(sn, times, offset=5):
    tdim = len(times) - offset
    rdim = np.shape(sn)[1]
    sn_min = np.zeros((tdim, np.shape(sn)[1]))
    sn_max = np.zeros((tdim, np.shape(sn)[1]))
    for t in prange(tdim):
        for r in prange(rdim):
            sn_min[t, r] = np.min(sn[t:t + offset, r])
            sn_max[t, r] = np.max(sn[t:t + offset, r])
        t_win[t] = times[t]
    return sn_min, sn_max, times[:tdim]

@njit(cache=True, parallel=True)
def get_every_realization_peakavg(sn):
    ntstep,nreals = np.shape(sn)
    sn_min = np.zeros((nreals))
    sn_max = np.zeros((nreals))

    for r in prange(nreals):
        # avg = np.ones((ntstep)) * np.nanmean(sn[:,r])
        max_idx, max_val = find_peaks_fast(sn[:,r], height=None, peakwidth=1)
        min_idx, min_val = find_peaks_fast(-sn[:,r], height=None, peakwidth=1)
        # print(max_val, max_idx)
        # print(-min_val, min_idx)
        # plt.figure()
        # plt.plot(sn[:,r])
        # plt.scatter(max_idx, sn[max_idx,r])
        # plt.scatter(min_idx, sn[min_idx,r])
        # plt.show()

        if len(max_idx) > 0:
            sn_max[r] = max_val[0] # np.nanmean(max_val) # max_val[0]
        else:
            sn_max[r] = np.nanmean(sn[:, r])
        if len(min_idx) > 0:
            sn_min[r] =  -min_val[0] # np.nanmean(-min_val) # -min_val[0]
        else:
            sn_min[r] = np.nanmean(sn[:, r])
    return sn_min, sn_max



@njit(cache=True, parallel=False)
def get_disorder_averaged_peaks(sn, times, offset=5):
    tdim = len(times) - offset
    rdim = np.shape(sn)[1]
    sn_min_davg = np.zeros(tdim)
    sn_max_davg = np.zeros(tdim)
    sn_min_dste = np.zeros(tdim)
    sn_max_dste = np.zeros(tdim)
    sn_avg_davg = np.zeros(tdim)
    sn_avg_dste = np.zeros(tdim)
    for t in prange(tdim):
        sn_min, sn_max = get_every_realization_peakavg(sn[t:t+offset, :])
        nmin = np.count_nonzero(~np.isnan(sn_min))
        nmax = np.count_nonzero(~np.isnan(sn_max))
        # print(np.shape(sn_max), nmax)
        # print(np.shape(sn_min), nmin)
        if nmin == 0:
            print('nmin==0')
            nmin = rdim
            sn_min = np.sum(sn[t:t+offset,:], axis=0)/offset
        if nmax == 0:
            print('nmax==0')
            nmax = rdim
            sn_max = np.sum(sn[t:t+offset,:], axis=0)/offset

        sn_min_davg[t] = np.nanmean(sn_min)
        sn_min_dste[t] = np.nanstd(sn_min) / np.sqrt(nmin)
        sn_max_davg[t] = np.nanmean(sn_max)
        sn_max_dste[t] = np.nanstd(sn_max) / np.sqrt(nmax)
        sn_avg_davg[t] = 0.5 * (sn_min_davg[t] + sn_max_davg[t])
        sn_avg_dste[t] = np.sqrt(nmin * sn_min_dste[t] ** 2 + nmax * sn_max_dste[t] ** 2) / np.sqrt(nmin + nmax)

    return (sn_min_davg,
            sn_min_dste,
            sn_max_davg,
            sn_max_dste,
            sn_avg_davg,
            sn_avg_dste,
            times[:tdim])


def get_disorder_averaged_minmax(sn,times, offset=5):
    tindices = np.arange(0, len(times) - offset, 1)
    nindices = len(tindices)
    sn_min_mean = np.zeros(nindices)
    sn_max_mean = np.zeros(nindices)
    sn_min = np.zeros(np.shape(sn)[1])
    sn_max = np.zeros(np.shape(sn)[1])
    t_win = np.zeros(nindices)
    for t in tindices:
        for r in prange(np.shape(sn)[1]):
            sn_min[r] = np.min(sn[t:t + offset, r])
            sn_max[r] = np.max(sn[t:t + offset, r])
        sn_min_mean[t] = np.mean(sn_min)
        sn_max_mean[t] = np.mean(sn_max)
        t_win[t] = times[t]
    return sn_min_mean, sn_max_mean, t_win





@dataclass
class lbit_fit:
    C: np.float64 = np.nan
    xi: np.float64 = np.nan
    beta: np.float64 = np.nan
    pos : int = -1
    yfit: np.ndarray = field(default_factory=np.ndarray)
    xierr: np.float64 = np.nan
    betaerr: np.float64 = np.nan
    idxN: int = -1


def get_lbit_fit(x, y, beta=None, ymin=None, skip=None):
    if beta is None:
        beta = False
    if ymin is None:
        ymin = 1e-4
    if skip is None:
        skip = 2

    if np.size(y) <= skip + 2:
        print('get_lbit_fit: y is too short (skip={}):{}'.format(skip, y))
        return lbit_fit()
    ydata = np.ndarray.flatten(y)
    xdata = np.ndarray.flatten(x)
    idx0 = skip  # Skip first tries, which usually do not obey exp decay
    idxN = 0
    for idx, val in enumerate(ydata):
        if abs(val) < np.max(np.abs(ydata[idx:])):
            break
        if idx >= 2 and abs(val) > 0.5 * (abs(ydata[idx - 1]) + abs(ydata[idx - 2])):
            break
        if abs(val) < ymin:
            break
        if val == 0:
            break

        idxN = idx

    ydata = ydata[idx0:idxN]
    xdata = xdata[idx0:idxN]
    try:
        if idxN >= 0 and idxN <= idx0:
            raise IndexError("Invalid index order: idx0 {} | idxN {}".format(idx0, idxN))
        if idxN >= 0 and idxN < idx0 + 2:
            raise IndexError("Too few datapoints for a fit: idx0 {} | idxN {}".format(idx0, idxN))

        with np.errstate(invalid='ignore'):
            ylogs = np.log(np.abs(ydata))
            if beta:
                p0 = 0.5, 1.0, 1.0
                sfit = stretchedFit()
                sfit.b = np.argmax(ylogs)
                popt, pcov = curve_fit(stretched_log, xdata=xdata, ydata=ylogs, p0=p0)
                pstd = np.sqrt(np.diag(pcov))  # Gives 3 columns with len(xdata) rows
                xierr = pstd[1] / np.sqrt(np.size(pstd[1]))
                return lbit_fit(popt[0], popt[1], popt[2], stretched_exp(x, *popt), xierr, idxN)
            else:
                result = linregress(x=xdata, y=ylogs, alternative='less')
                C = np.exp(result.intercept)
                xi = 1.0 / abs(result.slope)
                yfit = C * np.exp(-x / xi)
                return lbit_fit(C, xi, np.nan, yfit, result.stderr, idxN)
    except IndexError as e:
        print("Index error:", e)
        pass
    except ValueError as e:
        print("Fit failed:", e)

    return lbit_fit()



# Returns C, xi, beta, yfit, LinregressResult instance or pstd
def get_lbit_fit_data(x, y, e=None, ymin=None, beta=None):
    if ymin is None:
        if e is None:
            ymin = 1e-8
        else:
            ymin = 0.0
    if beta is None:
        beta = False
    try:
        with np.errstate(invalid='ignore'):
            ymask = np.ma.masked_where(np.abs(y) <= ymin, np.abs(y))
            emask = None
            elogs = None
            if e is not None:
                emask = np.ma.masked_where(np.ma.getmask(ymask), e)
                elogs = np.ndarray.flatten(np.abs(np.log(emask.compressed())))
            ylogs = np.ndarray.flatten(np.log(ymask.compressed()))
            xtile = np.atleast_2d(x).T if np.size(y) == np.shape(y)[0] else np.tile(x, (np.shape(y)[0], 1))
            xflat = np.ndarray.flatten(np.ma.masked_where(np.ma.getmask(ymask), xtile).compressed())
            if beta:
                pos = np.argmax(y)
                p0 = 0.5, 1.0, 1.0
                bs = ([0, 0, 0], [np.inf, 5, 5])
                # print('x {} \n{}'.format(np.shape(xflat),xflat))
                # print('y {} \n{}'.format(np.shape(ylogs),ylogs))
                # print('e {} \n{}'.format(np.shape(elogs),elogs))
                sfit  = stretchedFit()
                sfit.pos = pos
                popt, pcov = curve_fit(sfit.stretched_log, xdata=xflat, ydata=ylogs, sigma=elogs, p0=p0, bounds=bs)
                perr = np.sqrt(np.diag(pcov))  # Gives 3 columns with len(xdata) rows
                print(f'{perr=}')
                xierr = perr[1] / np.sqrt(np.size(perr[1]))
                betaerr = perr[2] / np.sqrt(np.size(perr[2]))
                return lbit_fit(popt[0], popt[1], popt[2], pos, sfit.stretched_exp(x, *popt), xierr, betaerr, -1)
            else:
                pos = np.argmax(y)
                p0 = 0.5, 1.0, 1.0
                bs = ([0, 0, 1.0-1e-10], [np.inf, 5, 1.0+1e-10])
                # print('x {} \n{}'.format(np.shape(xflat),xflat))
                # print('y {} \n{}'.format(np.shape(ylogs),ylogs))
                # print('e {} \n{}'.format(np.shape(elogs),elogs))
                sfit  = stretchedFit()
                sfit.pos = pos
                popt, pcov = curve_fit(sfit.stretched_log, xdata=xflat, ydata=ylogs, sigma=elogs, p0=p0, bounds=bs)
                perr = np.sqrt(np.diag(pcov))  # Gives 3 columns with len(xdata) rows
                print(f'{perr=}')
                xierr = perr[1] / np.sqrt(np.size(perr[1]))
                betaerr = perr[2] / np.sqrt(np.size(perr[2]))
                return lbit_fit(popt[0], popt[1], popt[2], pos, sfit.stretched_exp(x, *popt), xierr,betaerr, -1)
    except IndexError as e:
        print("Index error:", e)
        raise
    except ValueError as e:
        print("Fit failed:", e)
        raise

    return lbit_fit()


@njit(parallel=True, cache=True)
def nb_mean_cmat(a, mean=None):
    if mean is None:
        mean = 'arithmetic'
    if not mean in ['arithmetic', 'geometric']:
        raise ValueError("invalid mean")
    shp = np.shape(a)
    avg = np.empty(shape=(shp[1], shp[2]))
    std = np.empty(shape=(shp[1], shp[2]))
    for i in range(shp[1]):
        for j in range(shp[2]):
            if mean == 'arithmetic':
                avg[i, j] = np.nanmean(a[:, i, j])
                std[i, j] = np.nanstd(a[:, i, j])
            elif mean == 'geometric':
                avg[i, j] = np.exp(np.nanmean(np.log(np.abs(a[:, i, j]))))
                std[i, j] = np.nanstd(a[:, i, j])
    return np.abs(avg), std


@njit(parallel=True, cache=True)
def nb_nnz_mean_axis0(a):
    shp = np.shape(a)
    res = np.zeros(shape=(shp[1], 1))

    for j in prange(shp[1]):
        sum = 0.0
        num = 0.0
        for i in prange(shp[0]):
            if a[i, j] > 0.0 and np.isfinite(a[i, j]):
                sum += a[i, j]
                num += 1.0
        if num > 0.0:
            res[j, 0] += sum / num
    return res


@dataclass
class lbit_corr:
    mean: np.ndarray
    full: np.ndarray
    stdv: np.ndarray
    fold: np.ndarray
    fodv: np.ndarray
    csup: np.ndarray
    csrr: np.ndarray
    sprd: np.ndarray
    sprr: np.ndarray



@njit(parallel=True, cache=True)
def get_folded_matrix(mat, rms=False, mean=None):
    if mean is None:
        mean = 'arithmetic'
    fold = np.empty(shape=np.shape(mat))
    rows = np.shape(mat)[0]
    for i in range(rows):
        yl = np.flip(mat[i, :i])
        yr = mat[i, i + 1:]
        yn = max(len(yl), len(yr))
        yz = np.zeros(shape=(yn), dtype=np.float64)
        for k in range(yn):
            i1 = 1.0 if k < len(yl) else 0.0
            i2 = 1.0 if k < len(yr) else 0.0
            y1 = yl[k] if k < len(yl) else 0.0
            y2 = yr[k] if k < len(yr) else 0.0
            if rms:
                yz[k] = (y1 ** 2 + y2 ** 2) ** 0.5
            else:
                if mean == 'arithmetic':
                    yz[k] = (y1 + y2) / (i1 + i2)
                elif mean == 'geometric':
                    tmp = (np.log(np.abs(y1)) + np.log(np.abs(y2))) / (i1 + i2)
                    yz[k] = np.exp(tmp)
        fold[i, :] = 0.0
        fold[i, 0] = mat[i, i]
        fold[i, 1:len(yz) + 1] = yz
    return fold


# @njit(parallel=True, cache=True)
def get_lbit_avg(corrmat, site=None, mean=None):
    if site is None:
        site = 0
    if mean is None:
        mean = 'arithmetic'
    print(f'corrmat shape {np.shape(corrmat)}')
    full, stdv = nb_mean_cmat(corrmat, mean=mean)
    fold = get_folded_matrix(full, rms=False, mean=mean)
    fodv = get_folded_matrix(stdv, rms=True, mean=mean)

    size = np.shape(full)[0]
    midr = int(np.shape(full)[0] / 2)
    num = np.shape(corrmat)[0]
    csup = np.sum([np.sum(full[midr:-1, 0:midr]), np.sum(full[0:midr, midr:-1])])
    csrr = np.sum([np.sum(stdv[midr:-1, 0:midr]/np.sqrt(num)), np.sum(stdv[0:midr, midr:-1]/np.sqrt(num))])
    sprd = np.sum(np.abs((full - np.identity(n=size))))/size
    sprr = np.sum(stdv/np.sqrt(num))/size
    # with np.printoptions(edgeitems=30, linewidth=100000, formatter={'float': '{: 0.3e}'.format}):
    #     print(full)
    if site is not None:
        if isinstance(site, list):
            slice = []
            for s in site:
                if isinstance(s, str) and s == "mid":
                    slice.append(int(np.shape(full)[0] / 2))
                elif isinstance(s, str) and s == "last":
                    slice.append(np.shape(full)[0]-1)
                elif isinstance(s, int):
                    slice.append(s)
                else:
                    raise TypeError("Unsupported site selection: {}".format(s))
            full = np.atleast_2d(full[slice, :]).T
            stdv = np.atleast_2d(stdv[slice, :]).T
            fold = np.atleast_2d(fold[slice, :]).T
            fodv = np.atleast_2d(fodv[slice, :]).T

        elif isinstance(site, str) and site == "mid":
            row = int(np.shape(full)[0] / 2)
            full = np.atleast_2d(full[row, :]).T
            stdv = np.atleast_2d(stdv[row, :]).T
            fold = np.atleast_2d(fold[row, :]).T
            fodv = np.atleast_2d(fodv[row, :]).T
        elif isinstance(site, int):
            full = np.atleast_2d(full[site, :]).T
            stdv = np.atleast_2d(stdv[site, :]).T
            fodv = np.atleast_2d(fodv[site, :]).T
        else:
            raise TypeError("Unsupported site selection: {}".format(site))
    mean = nb_nnz_mean_axis0(fold)
    return lbit_corr(mean, full, stdv, fold, fodv, csup, csrr,sprd, sprr)

#
# def find_saturation_idx(ydata, std_threshold):
#     sdata = []
#     for i in range(len(ydata)):
#         sdata.append(np.std(ydata[i:]))
#     sdiff = -np.log(np.abs(np.diff(sdata)))
#     return np.argmax(sdiff)


@njit(parallel=True, cache=True)
def find_saturation_idx(ydata):
    # Consider Y vs X: a noisy signal decaying in the shape of a hockey-club, say.
    # We want to identify the point at which the signal stabilizes. We use the fact that the
    # standard deviation is high if it includes parts of the non-stationary signal, and low if
    # it includes only the stationary part.
    # Here we monitor the standard deviation of the signal between [start_point, end_point],
    # and move "start_point" towards the end. If the standard deviation goes below a certain
    # threshold, i.e. threshold < max_std, then we have found the stabilization point.
    sdata = np.empty_like(ydata)
    for i in range(len(ydata)):
        sdata[i] = np.std(ydata[i:])
    sdiff = -np.log(np.abs(np.diff(sdata)))
    return np.argmax(sdiff)


def find_saturation_idx2(ydata, threshold=1e-2):
    if len(np.shape(ydata)) != 2:
        raise "sdata must be 2d matrix (time, realization), eg (200 x 80000)"
    ylog = -np.log10(ydata)
    ylog = ylog / ylog[-1]
    w = 2
    sdata = []
    for i, yl in enumerate(ylog):
        min_idx = np.min([len(ylog) - w, i])
        min_idx = np.max([min_idx, 0])
        s = np.std(ylog[min_idx:])
        sdata.append(s)
    fig,ax = plt.subplots()
    ax.plot(sdata)
    plt.show()
    idx = np.argwhere(np.asarray(sdata) < threshold)[0, 0]
    return idx

@njit(float64[:,:](float64[:,:]), cache=True, parallel=True)
def running_avg(ydata):
    tdim ,rdim = np.shape(ydata)
    ymean = np.empty_like(ydata)
    for i in prange(tdim): # Iterate rows
        idx = int(np.min(np.asarray([i, tdim-1])))
        for j in prange(rdim):
            ymean[i, j] = np.mean(ydata[idx:,j])
    return ymean
@njit(float64[:,:](float64[:,:]), cache=True, parallel=True)
def running_std(ydata):
    tdim ,rdim = np.shape(ydata)
    ystds = np.empty_like(ydata)
    for i in prange(tdim): # Iterate rows
        idx = int(np.min(np.asarray([i, tdim-1])))
        for j in prange(rdim):
            ystds[i, j] = np.std(ydata[idx:, j])
    return ystds

@njit(cache=True, parallel=True)
def running_stats(ydata):
    tdim ,rdim = np.shape(ydata)
    ystds = np.empty_like(ydata)
    ymean = np.empty_like(ydata)
    ymaxs = np.empty_like(ydata)
    for i in prange(tdim): # Iterate rows
        idx = int(np.min(np.asarray([i, tdim-1])))
        for j in prange(rdim):
            ystds[i, j] = np.std(ydata[idx:, j])
            ymean[i, j] = np.mean(ydata[idx:,j])
            ymaxs[i, j] = np.max(ydata[idx:,j])
            # ycmsm[i, j] = np.sum(ydata[:i, j]) / (i)
    return ystds,ymean,ymaxs

# @njit(numba.types.Tuple((int64[:],float64[:]))
#           (float64[:,:],float64, int64,boolean, int64), cache=True,parallel=True)
# def get_saturation_from_diff_old(ydata, wsize=0.05, count=1, require_beyond=False, setsign = 0):
#     tdim,rdim = np.shape(ydata)
#     sign = np.ones(shape=(rdim), dtype=np.int64) * setsign
#     tidx = np.ones(shape=rdim, dtype=np.int64) * (tdim-1)
#     yval = np.zeros(shape=rdim, dtype=np.float64) * np.nan
#     nhit = np.zeros(shape=rdim, dtype=np.int64) # Counts the number of times the signal switches directio
#     whalf = np.max(np.asarray([1, int(tdim*wsize/2)]))
#     for j in prange(rdim):
#         for i in prange(whalf , tdim - whalf):
#             ychnk = np.ascontiguousarray(ydata[i-whalf:i+whalf, j])
#             diffsum = np.sum(np.diff(ychnk))
#             if sign[j] == 0 and diffsum != 0:
#                 sign[j] = 1 if diffsum > 0 else -1
#                 continue
#             # A hit means that the signal is changing direction!
#             hit = (sign[j] == 1 and diffsum < 0) or (sign[j] == -1 and diffsum > 0)
#             # We also require that this point is beyond the mean of the remaning time series
#
#             beyond_ymean = False
#             if require_beyond:
#                 ymean = np.mean(np.ascontiguousarray(ydata[i:, j]))
#                 if sign[j] == 1 and ydata[i,j] >= ymean:
#                     beyond_ymean = True
#                 if sign[j] == -1 and ydata[i,j] <= ymean:
#                     beyond_ymean = False
#             else:
#                 beyond_ymean = True
#             if hit and beyond_ymean:
#                 nhit[j] += 1
#                 if nhit[j] >= count:
#                     tidx[j] = i
#                     yval[j] = ydata[i, j]
#                     break
#         # if np.isnan(yval[j]):
#         #     yval[j] = ydata[-1, j]
#     return tidx, yval
@njit(nb.types.Tuple((int64[:],float64[:]))
          (float64[:,:], float64, int64), cache=True,parallel=True)
def get_saturation_from_diff(ydata, wsize=0.01, setsign = 0):
    tdim,rdim = np.shape(ydata)
    sign = np.ones(shape=(rdim), dtype=np.int64) * setsign
    tidx = np.ones(shape=rdim, dtype=np.int64) * (tdim-1)
    yval = np.zeros(shape=rdim, dtype=np.float64) * np.nan
    # nhit = np.zeros(shape=rdim, dtype=np.int64) # Counts the number of times the signal switches directio
    # whalf = np.max(np.asarray([1, int(tdim*wsize/2)]))
    wstep = np.max(np.asarray([2, int(tdim*wsize)]))
    for j in range(rdim):
        for i in range(0 , tdim - wstep):
            ychnk = np.ascontiguousarray(ydata[i:i+wstep, j])
            diffsum = np.sum(np.diff(ychnk))
            if sign[j] == 0 and diffsum != 0:
                sign[j] = 1 if diffsum > 0 else -1
                continue
            # A hit means that the signal is changing direction!
            hit = (sign[j] == 1 and diffsum < 0) or (sign[j] == -1 and diffsum > 0)
            # We also require that this point is beyond the mean of the remaning time series

            if hit:
                tidx[j] = i
                yval[j] = ydata[i, j]
                break
        # if np.isnan(yval[j]):
        #     yval[j] = ydata[-1, j]
    return tidx, yval


@dataclass
class entropy_saturation_time_bootstrap:
    tsat_full_avg : np.float64 = np.nan
    tsat_full_err : np.float64 = np.nan
    tsat_full_avg_idx : np.float64 = np.nan
    tsat_full_err_idx : np.float64 = np.nan
    ysat_full_avg : np.float64 = np.nan
    tsat_boot_avg : np.float64 = np.nan
    tsat_boot_err : np.float64 = np.nan
    tsat_boot_avg_idx : np.float64 = np.nan
    tsat_boot_err_idx : np.float64 = np.nan
    ysat_boot_avg : np.float64 = np.nan

@dataclass
class entropy_saturation_bootstrap:
    # sdavg : np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))# Disorder average
    # sravg : np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))# Running disorder average

    sinf_full_avg : np.float64 = np.nan
    sinf_full_err : np.float64 = np.nan

    sinf_boot_avg : np.float64 = np.nan
    sinf_boot_med : np.float64 = np.nan
    sinf_boot_err : np.float64 = np.nan

    ssat_full_avg: np.float64 = np.nan
    ssat_full_err: np.float64 = np.nan

    ssat_boot_avg : np.float64 = np.nan
    ssat_boot_med : np.float64 = np.nan
    ssat_boot_max : np.float64 = np.nan
    ssat_boot_err : np.float64 = np.nan

    tsat_full_avg : np.float64 = np.nan
    tsat_full_err : np.float64 = np.nan
    tsat_full_avg_idx : np.float64 = np.nan
    tsat_full_err_idx : np.float64 = np.nan

    tsat_boot_avg : np.float64 = np.nan
    tsat_boot_med : np.float64 = np.nan
    tsat_boot_max : np.float64 = np.nan
    tsat_boot_err : tuple[np.float64,np.float64] = field(default_factory=tuple)
    tsat_boot_avg_idx : np.float64 = np.nan
    tsat_boot_med_idx : np.float64 = np.nan
    tsat_boot_max_idx : np.float64 = np.nan
    tsat_boot_err_idx : np.float64 = np.nan



# @njit(numba.types.Tuple((float64,float64,float64,float64[:],float64,float64)) (float64[:,:], float64[:], int64), cache=True,parallel=True)
def get_entropy_saturation_from_bootstrap(sdata,
                                          tdata,
                        nbs = 100 # Number of bootstraps
                        ):
    if len(np.shape(sdata)) != 2:
        raise AssertionError("sdata must be 2d matrix (time, realization), eg (200 x 80000)")
    tdim,rdim = np.shape(sdata)
    # ytran = np.ascontiguousarray(ydata.T)
    ridx = np.arange(rdim) # realization indices
    tidx = np.empty(shape=nbs, dtype=np.float64) # Bootstrapped saturation time index
    tsat = np.empty(shape=nbs, dtype=np.float64) # Bootstrapped saturation time value
    ssat = np.empty(shape=nbs, dtype=np.float64) # Bootstrapped saturation entropy value
    sinf = np.empty(shape=nbs, dtype=np.float64) # Bootstrapped saturation entropy value
    # sboots = np.empty(shape=(tdim, nbs))
    for n in range(nbs):
        # rrnd = choices(ridx, k=rdim) # Randomly selected realizations
        rrnd = np.sort(np.random.choice(ridx, size=rdim, replace=True))
        sboot = np.mean(sdata[:, rrnd], keepdims=True, axis=1)
        # sboots[:, n] = sboot[:,0]
        sbravg = running_avg(sboot)
        sbidx, sbsat = get_saturation_from_diff(sbravg, wsize=0.01, setsign=1)
        if not np.any(np.isnan(sbsat)):
            tidx[n] = np.mean(sbidx)
            ssat[n] = np.interp(tidx[n], range(tdim), sboot[:,0])
            tsat[n] = np.interp(tidx[n], range(tdim), tdata)
            sinf[n] = np.interp(tidx[n], range(tdim), sbravg[:,0])
        else:
            # We did not detect a convergence point, try detecting it from the standard deviation instead
            sbrstd = running_std(sboot)
            sbidx, _ = get_saturation_from_diff(sbrstd, wsize=0.01, setsign=-1)
            print("used stds to get", sbidx)
            tidx[n] = np.mean(sbidx)
            ssat[n] = np.interp(tidx[n], range(tdim), sboot[:,0])
            tsat[n] = np.interp(tidx[n], range(tdim), tdata)
            sinf[n] = np.interp(tidx[n], range(tdim), sbravg[:,0])
            # ax[0].scatter(x=[tidx[n]], y=[yval[n]], s=20, color='red')
            # ax[1].scatter(x=[tidx[n]], y=[np.mean(stds[round(tidx[n])])], s=20, color='red')

            if np.isnan(ssat[n]):
                raise ValueError('yval is nan')
        # print(f'n: {n} | t {tval[n]:.3e}')

    # We now have nbs estimations of the saturation time index and corresponding values.
    # The standard error is given by the standard deviation of these mean estimations.

    esb = entropy_saturation_bootstrap()

    # print(tidx)
    # print(tsat)
    esb.sinf_boot_avg = np.mean(sinf)
    esb.sinf_boot_med = np.median(sinf)
    esb.sinf_boot_err = np.std(sinf, ddof=1)

    esb.tsat_boot_avg_idx = np.mean(tidx) # This is a floating point index such as 83.45. Time is interpolated between 83 and 84.
    esb.tsat_boot_med_idx = np.median(tidx) # This is a floating point index such as 83.45. Time is interpolated between 83 and 84.
    esb.tsat_boot_max_idx = np.max(tidx) # This is a floating point index such as 83.45. Time is interpolated between 83 and 84.
    esb.tsat_boot_err_idx = np.std(tidx, ddof=1)
    # Interpolate the time index average
    esb.tsat_boot_avg = np.interp(esb.tsat_boot_avg_idx, range(tdim), tdata)
    esb.tsat_boot_med = np.interp(esb.tsat_boot_med_idx, range(tdim), tdata)
    esb.tsat_boot_max = np.interp(esb.tsat_boot_max_idx, range(tdim), tdata)
    # We need to get the time error asymmetrically down and up
    esb.tsat_boot_err = (np.abs(esb.tsat_boot_avg - np.interp(esb.tsat_boot_avg_idx-esb.tsat_boot_err_idx, range(tdim), tdata)),
                         np.abs(esb.tsat_boot_avg - np.interp(esb.tsat_boot_avg_idx+esb.tsat_boot_err_idx, range(tdim), tdata)))


    esb.ssat_boot_avg = np.mean(ssat)
    esb.ssat_boot_err = np.std(ssat, ddof=1)
    esb.ssat_boot_med = np.median(ssat)
    esb.ssat_boot_max = np.max(ssat)
    # esb.ssat_boot_avg = np.mean(sboots[round(esb.tsat_boot_avg_idx), :])
    # esb.ssat_boot_err = np.std(sboots[round(esb.tsat_boot_avg_idx), :])
    # esb.ssat_boot_med = np.mean(sboots[round(esb.tsat_boot_med_idx), :])
    # esb.ssat_boot_max = np.mean(sboots[round(esb.tsat_boot_max_idx), :])

    # Calculate the full (non-bootstrapped) infinite-time values,
    # assuming that they all saturate at the same time index  np.max(tsat)
    sinfs = np.mean(sdata[round(esb.tsat_boot_max_idx):, :], keepdims=True, axis=0) # Time point of certain convergence
    esb.sinf_full_avg = np.mean(sinfs)
    esb.sinf_full_med = np.median(sinfs)
    esb.sinf_full_err = np.std(sinfs)/np.sqrt(rdim)

    sdavg = np.mean(sdata, keepdims=True, axis=1)
    sravg = running_avg(sdavg)
    tsat_full_idx, ssat_full_avg = get_saturation_from_diff(sravg, wsize=0.01, setsign=1)
    tsat_full_idx = round(np.mean(tsat_full_idx))
    ssat_full_avg = np.mean(ssat_full_avg)

    esb.ssat_full_avg = ssat_full_avg
    esb.ssat_full_med = np.median(sdata[tsat_full_idx, :])
    esb.ssat_full_err = np.std(sdata[tsat_full_idx, :]) / np.sqrt(rdim)

    esb.tsat_full_avg_idx = tsat_full_idx
    esb.tsat_full_err_idx = np.nan
    esb.tsat_full_avg = tdata[tsat_full_idx]
    esb.tsat_full_med = tdata[tsat_full_idx]
    esb.tsat_full_err = np.nan

    # print('tsat_boot_avg_idx:',tsat_boot_avg_idx)
    # print('tsat_boot_err_idx:',tsat_boot_err_idx)
    # print('tsat_boot_avg    :',tsat_boot_avg)
    # print('tsat_boot_err    :',tsat_boot_err)
    return esb


def get_entropy_inftime_saturation_value_from_bootstrap(ydata,
                                          tdata,
                        nbs = 100 # Number of bootstraps
                        ):
    if len(np.shape(ydata)) != 2:
        raise AssertionError("sdata must be 2d matrix (time, realization), eg (200 x 80000)")
    tdim,rdim = np.shape(ydata)
    ridx = np.arange(rdim) # realization indices
    boot_yval = np.empty(shape=nbs, dtype=np.float64) # Bootstrapped saturation entropy value

    yavgs = np.mean(ydata, keepdims=True, axis=1)
    yravg = running_avg(yavgs)
    full_tidx, full_yval = get_saturation_from_diff(yravg, wsize=0.01, setsign=1)


    for n in range(nbs):
        rrnd = np.sort(np.random.choice(ridx, size=rdim, replace=True))
        boot = np.mean(ydata[:, rrnd], keepdims=True, axis=1)
        bravg = running_avg(boot)
        bsatidx, _ = get_saturation_from_diff(bravg, wsize=0.01, setsign=1)
        # Now we have the average saturation time for this bootstrapped set.
        # We calculate the infinite-time from the mean of saturated values
        if not np.any(np.isnan(boot_yval)):
            boot_yval[n] = bravg[bsatidx]
        else:
            # We did not detect a convergence point, try detecting it from the standard deviation instead
            brstd = running_std(boot)
            stds_tidx, _ = get_saturation_from_diff(brstd, wsize=0.01, setsign=-1)

            boot_yval[n] = bravg[stds_tidx]
            print("used stds to get", stds_tidx, boot_yval[n])
            # ax[0].scatter(x=[tidx[n]], y=[yval[n]], s=20, color='red')
            # ax[1].scatter(x=[tidx[n]], y=[np.mean(stds[round(tidx[n])])], s=20, color='red')

            if np.isnan(boot_yval[n]):
                raise ValueError('boot_yval is nan')
        # print(f'n: {n} | t {tval[n]:.3e}')
    # The standard error is given by the standard deviation of the mean estimations.

    # Interpolate the time index average
    ysat_full_avg = np.mean(full_yval)
    ysat_full_med = np.median(full_yval)
    ysat_full_err = np.std(full_yval, ddof=1)

    ysat_boot_avg = np.mean(boot_yval)
    ysat_boot_med = np.median(boot_yval)
    ysat_boot_err = np.std(boot_yval, ddof=1)

    print(f'{ysat_full_avg=}')
    print(f'{ysat_full_med=}')
    print(f'{ysat_full_err=}')
    print(f'{ysat_boot_avg=}')
    print(f'{ysat_boot_med=}')
    print(f'{ysat_boot_err=}')
    return ysat_boot_avg,ysat_boot_med, ysat_boot_err


def find_davg_entropy_saturation_time_from_bootstrap(ydata, tp, nbs=100, dsetname='entanglement_entropy'):
    if len(np.shape(ydata)) != 2:
        raise "sdata must be 2d matrix (time, realization), eg (200 x 80000)"
    tsb = get_entropy_saturation_from_bootstrap(ydata, tp.time, nbs=nbs)
    # yavgs = np.mean(ydata, axis=1, keepdims=True)
    # yravg = running_avg(yavgs)
    # yrstd = running_std(yavgs)
    # tsats_ravg_avg_idx, ysats_ravg_avg = get_saturation_from_diff(yravg, wsize=0.01, setsign=1)
    # tsats_rstd_avg_idx, ysats_rstd_avg = get_saturation_from_diff(yrstd, wsize=0.01, setsign=-1)
    # # Take the average of these temporary values
    # tsb.tsat_full_avg_idx = np.mean(tsats_ravg_avg_idx)
    # tsb.tsat_full_avg = tp.time[round(tsb.tsat_full_avg_idx)]
    # tsb.ysat_full_avg = np.mean(ysats_ravg_avg)


    ridx = range(150,160)
    fig, ax = plt.subplots()
    # ax.plot(ydata[:,ridx])
    # ax.plot(yavgs, linewidth=1.0, color='red', linestyle=':')
    # ax.plot(yravg, linewidth=1.0, color='black', linestyle='--')
    # ax.scatter(x=tsb.tsat_full_avg_idx, y=tsb.ysat_full_avg, marker='o',s=40, linestyle='None', color='red', label='ravg')
    # ax.scatter(x=tsats_rstd_avg_idx, y=yravg[tsats_rstd_avg_idx], marker='o',s=30, linestyle='None', color='orange', label='rstd')
    # tsat_phys_idx = tp.idx_ent_saturated if 'entanglement' in dsetname else tp.idx_num_saturated
    # ax.scatter(x=tsat_phys_idx, y=yavgs[tsat_phys_idx,0], marker='o',s=40, linestyle='None', color='green', label='phys')
    # ax.errorbar(x=[tsb.tsat_boot_avg_idx], y=[tsb.ysat_boot_avg], xerr=[tsb.tsat_boot_err_idx], yerr=[tsb.ysat_boot_err], color='blue', label='boot')
    # ax.legend()
    # plt.show()
    return tsb

def find_entropy_inftime_saturation_value_from_bootstrap(sdata, tdata, nbs=100):
    # For each bootstrap set of realizations from ydata, this calculates the saturation time
    # and saturation value of individual realizations before averaging.
    if len(np.shape(sdata)) != 2:
        raise Exception("sdata must be 2d matrix (time, realization), eg (200 x 80000)")
    print(f'bootstrapping {nbs=} ...')
    esb = get_entropy_saturation_from_bootstrap(sdata=sdata, tdata=tdata, nbs=nbs)
    print(f'bootstrapping {nbs=} ... done:')

    # tsats_rstd_avg_idx = np.mean(tsats_rstd_avg_idx)
    return esb


    fig, ax = plt.subplots()
    ridx = range(150,160)
    # ax.plot(esb.sdavg, linewidth=1.0, color='red', linestyle=':')
    with sns.color_palette(palette='tab10', n_colors=len(ridx)):
        ax.plot(sdata[:,ridx])
        # ax.plot(esb.sravg, linewidth=1.0, linestyle='--')
    # tsat_phys_idx = tp.idx_ent_saturated if 'entanglement' in dsetname else tp.idx_num_saturated
    # ax.scatter(x=tsat_phys_idx, y=esb.sdavg[tsat_phys_idx,0], marker='o',s=40, linestyle='None', color='green', label='phys')
    ax.scatter(x=esb.tsat_full_avg_idx, y=esb.ssat_full_avg, marker='o', s=40, linestyle='None', color='red', label='ravg')
    ax.scatter(x=esb.tsat_boot_med_idx, y=esb.ssat_boot_med, marker='o', s=40, linestyle='None', color='blue', label='boot-median')
    ax.scatter(x=esb.tsat_boot_max_idx, y=esb.ssat_boot_max, marker='o', s=40, linestyle='None', color='purple', label='boot-max')
    ax.errorbar(x=[esb.tsat_boot_avg_idx], y=[esb.ssat_boot_avg], xerr=[esb.tsat_boot_err_idx], yerr=[esb.ssat_boot_err], color='blue', label='boot-avg')
    ax.legend()


    # plt.show()
    return esb



def find_saturation_idx4(ydata,idx_sat):
    # sdata has to be a numpy array of dimension 2
    # the result will be a 1d array with the saturation index of each column
    if len(np.shape(ydata)) != 2:
        raise "sdata must be 2d matrix (time, realization), eg (200 x 80000)"
    idx = []
    avgs = np.mean(ydata[idx_sat:,:], axis=0)
    for col, avg in zip(ydata.T, avgs):
        # for i in range(len(col)):
        #     if np.mean(col[i:]) >= avg:
        #         idx.append(i)
        #         break
        idx.append(np.argwhere(np.asarray(col) > avg)[0, 0])


    return idx, avgs



def find_loglog_window(tdata, ydata, J2_width, J2_ctof, threshold1=0.6, threshold2=1e-2):
    if len(ydata) <= 2:
        return
    ylog = -np.log10(ydata)
    ylog = ylog / ylog[-1]
    sdata = []
    w = 2
    for i, yl in enumerate(ylog):
        min_idx = np.min([len(ylog) - w, i])
        min_idx = np.max([min_idx, 0])
        s = np.std(ylog[min_idx:])
        sdata.append(s)
    tdx = np.argwhere(np.asarray(tdata) >= 1.0)
    idx1 = np.argwhere(np.asarray(sdata) < threshold1)
    idx2 = np.argwhere(np.asarray(sdata) < threshold2)

    tdx = tdx[0, 0] if len(tdx) > 0 else 0
    idx1 = idx1[0, 0] if len(idx1) > 0 else 0
    idx2 = idx2[0, 0] if len(idx2) > 0 else 0

    idx1 = np.max([tdx, idx1])
    return idx1, idx2


def find_loglog_window2(tdata, ydata, db, threshold2=1e-2):
    if len(tdata) == 1:
        return 0, 0
    L = db['vals']['L']
    r = db['vals']['r']
    x = db['vals']['x']
    wn = db['vals']['w']

    w1 = wn[0]  # The width of distribution for on-site field.
    w2 = wn[
        1]  # The width of distribution for pairwise interactions. The distribution is either U(J2_mean-w,J2_mean+w) or N(J2_mean,w)
    w3 = wn[2]  # The width of distribution for three-body interactions.

    if r == np.iinfo(np.uint64).max or r == 'L':
        r = L

    # We multiply by np.sqrt(2 / np.pi) because we are interested in <|N(0,w)|>
    w1 *= np.sqrt(1 - 2 / np.pi)
    w2 *= np.sqrt(1 - 2 / np.pi)
    w3 *= np.sqrt(1 - 2 / np.pi)

    J1min = w1 * np.exp(0 / x)
    J1max = w1 * np.exp(0 / x)
    tmin1 = 1.0 / J1max
    tmax1 = 1.0 / J1min

    r2max = np.float64(np.min([r, L]))  # Maximum interaction range, max(|i-j|)
    Jmin2 = w2 * np.exp(- r2max / (4.0 * x)) * np.sqrt(2.0 / np.pi)  # Size of the smallest 2-body terms (furthest neighbor, up to L/2)
    Jmax2 = w2 * np.exp(- 1.0 / x)           * np.sqrt(2.0 / np.pi)  # Size of the largest 2-body terms (nearest neighbor)
    tmax2 = 1.0 / Jmin2  # (0.5 to improve fits) Time that it takes for the most remote site to interact with the middle
    tmin2 = 1.0 / Jmax2  # Time that it takes for neighbors to interact

    J3min = w3 * np.exp(-2 / x)
    J3max = w3 * np.exp(-2 / x)
    tmin3 = 1.0 / J3max
    tmax3 = 1.0 / J3min

    tmin = np.max([tmin1, tmin2, tmin3])
    tmax = np.max([tmax1, tmax2, tmax3])
    # Make sure the time points are ordered
    tmin = np.min([tmin, tmax])
    tmax = np.max([tmin, tmax])

    idx1 = np.argmax(tdata.astype(float) >= tmin) - 1
    idx2 = np.argmax(tdata.astype(float) >= tmax) - 1

    # idx1 = np.where(tdata >= tmin)[0][0]
    # idx2 = np.where(tdata <= tmax)[0][-1]
    # print(tmin1, tmin2, tmin3)
    # print(tmax1, tmax2, tmax3)
    # print(tdata[idx1], tdata[idx2])
    if idx2 == len(tdata) - 1:
        idx2 = np.max([idx1, idx2 - 1])

    return idx1, idx2

# def find_saturation_idx3(tdata, db):
#     if len(tdata) == 1:
#         return 0, 0
#     L = db['vals']['L']
#     r = db['vals']['r']
#     x = db['vals']['x']
#     wn = db['vals']['w']
#
#     w1 = wn[0]  # The width of distribution for on-site field.
#     w2 = wn[1]  # The width of distribution for pairwise interactions. The distribution is either U(J2_mean-w,J2_mean+w) or N(J2_mean,w)
#     w3 = wn[2]  # The width of distribution for three-body interactions.
#
#     if r == np.iinfo(np.uint64).max or r == 'L':
#         r = L
#
#     r2max = np.float64(np.min([r, L]))  # Number of sites from the center site to the edge site, max(|i-j|)/2
#     N = np.sqrt(2.0 / np.pi) # Factor coming from folded normal distribution
#
#
#
#     # We can subtract one here because max (|i-j|) = L-1
#     Jmin2_ent = np.exp(- (r2max) / x) * w2 * N  # order of magnitude of
#     # We should not subtract 1 from r2max here, because we are counting sites from the edge to one passed the central bond
#     Jmax2_ent = np.exp(-1.0 / x) * w2  # Order of magnitude of the largest 2-body terms (nearest neighbor)
#
#     tmin1_ent = 1.0 / w1
#     tmax1_ent = 1.0 / w1
#     tmin2_ent = 1.0 / Jmax2_ent
#     tmax2_ent = 1.0 / Jmin2_ent  # (0.5 to improve fits) Time that it takes for the most remote site to interact with the middle
#     tmin3_ent = 1.0 / w3
#     tmax3_ent = 1.0 / w3
#
#     tmin_ent = np.min([tmin1_ent, tmin2_ent, tmin3_ent])
#     tmax_ent = np.max([tmax1_ent, tmax2_ent, tmax3_ent])
#     idx_ent_sat = np.where(tdata <= tmax_ent)[0][-1]
#     idx_ent_sat = np.max([idx_ent_sat, 0])  # Make sure its non-negative
#
#
#     # Now for the number entropy
#     tmin1_num = 1.0 / w1
#     tmax1_num = 1.0 / w1
#     tmax2_num = 1.0/(np.exp(-(r2max/2.0) / x) * w2 * N) # Time before an exchange happens between the edge and one past the center from the central bond hits the edge
#     tmid2_num = 1.0/(np.exp(-(r2max/4.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
#     tmin2_num = 1.0/(np.exp(-(1.0/2.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
#     tmin3_num = 1.0 / w3
#     tmax3_num = 1.0 / w3
#
#     tmin_num = np.min([tmin1_num, tmin2_num, tmin3_num])
#     tmid_num = tmid2_num
#     tmax_num = np.max([tmax1_num, tmax2_num, tmax3_num])
#     idx_num_sat = np.where(tdata <= tmax_num)[0][-1]
#     idx_num_sat = np.max([idx_num_sat, 0])  # Make sure its non-negative
#
#     return idx_num_sat,idx_ent_sat

@dataclass
class timepoints:
    time: np.ndarray = field(default_factory=np.ndarray)
    time_ent_lnt_begin: np.float64 = np.nan
    time_ent_lnt_cease: np.float64 = np.nan
    time_ent_saturated: np.float64 = np.nan
    time_num_lnlnt_begin: np.float64 = np.nan
    time_num_lnlnt_cease: np.float64 = np.nan
    time_num_saturated: np.float64 = np.nan
    idx_ent_lnt_begin: int = 0
    idx_ent_lnt_cease: int = 0
    idx_ent_saturated: int = 0
    idx_num_lnlnt_begin: int = 0
    idx_num_lnlnt_cease: int = 0
    idx_num_saturated: int = 0


def get_timepoints_old (tdata, db):
    if len(tdata) == 1:
        return 0, 0
    L = db['vals']['L']
    # r = np.max(db['vals']['r'])
    x = db['vals']['x']
    wn = db['vals']['w']

    w1 = wn[0]  # The width of distribution for on-site field.
    w2 = wn[1]  # The width of distribution for pairwise interactions. The distribution is either U(J2_mean-w,J2_mean+w) or N(J2_mean,w)
    w3 = wn[2]  # The width of distribution for three-body interactions.

    # if r == np.iinfo(np.uint64).max or r == 'L':
    #     r = L
    r = L # We get the correct times if we use r == L,
    r2max = np.float64(np.min([r, L]))  # Number of sites from the center site to the edge site, max(|i-j|)/2
    N = np.sqrt(2.0 / np.pi) # Factor coming from folded normal distribution

    t = timepoints(time=tdata)
    # t.time = tdata
    w1 = 1.0 if w1 == 0.0 else w1
    w2 = 1.0 if w2 == 0.0 else w2
    w3 = 1.0 if w3 == 0.0 else w3

    tmin1_ent = 1.0 / (w1*np.exp(0 / x))
    tmax1_ent = 1.0 / (w1*np.exp(0 / x))
    tmin2_ent = 1.0 / (w2*np.exp(-1.0 / x))
    tmid2_ent = 1.0 / (w2*np.exp(- (r2max / 2)  / x) * N)
    tmax2_ent = 1.0 / (w2*np.exp(- (r2max - 1 ) / x) * N)
    # print('RETURN THE SHIFT-VALUE OF TMAX2_ENT')
    tmin3_ent = 1.0 / (w3*np.exp(-2.0 / x))
    tmax3_ent = 1.0 / (w3*np.exp(-2.0 / x))

    t.time_ent_lnt_begin = np.max([tmin1_ent, tmin2_ent, tmin3_ent])
    t.time_ent_lnt_cease = tmid2_ent
    t.time_ent_saturated = np.max([tmax1_ent, tmax2_ent, tmax3_ent])

    t.idx_ent_lnt_begin = np.argmax(tdata.astype(float) >= t.time_ent_lnt_begin) -1
    t.idx_ent_lnt_cease = np.argmax(tdata.astype(float) >= t.time_ent_lnt_cease) -1
    t.idx_ent_saturated = np.argmax(tdata.astype(float) >= t.time_ent_saturated) -1

    t.idx_ent_lnt_begin = np.max([t.idx_ent_lnt_begin, 0])  # Make sure it is non-negative
    t.idx_ent_lnt_cease = np.max([t.idx_ent_lnt_cease, 0])  # Make sure it is non-negative
    t.idx_ent_saturated = np.max([t.idx_ent_saturated, 0])  # Make sure it is non-negative

    # Now for the number entropy
    tmin1_num = 1.0 / (w1*np.exp(0 / x))
    tmax1_num = 1.0 / (w1*np.exp(0 / x))
    tmin2_num = 1.0 / (w2*np.exp(- (1.0   / 2) / x) * N)
    tmid2_num = 1.0 / (w2 * np.exp(- ((r2max / 4)) / x) * N)
    # Subtract two:
    #   In 010101|010101 the particle.
    #   (OLD) The half-chain particle is a distance |i-j| = |(L/2-1) - 1| away from the furthest particle on the same side.
    #   Excluding the particle at the edge, the most remote pair of particles are at |i-j| = |(L-1) - 2 - 1| = |L-4|
    #   away from the furthest particle on the same side.
    #   We exclude the particle at the edge because it is too restricted.
    # tmax2_num = 1.0 / (w2 * np.exp(-((r2max / 2)-2.5) / x) * N)
    tmax2_num = 1.0 / (w2 * np.exp(-((r2max-4) / 2) / x) * N)
    # tmid2_num = tmid2_ent ** 0.5
    # tmax2_num = tmax2_ent ** 0.5


    # tmid2_num = 1.0 / (w2*np.exp(- ((r2max / 4)) / x) * N)
    # Subtract two: In 01010|10101 excluding the edge particle, the most distant "1"'s are L/2-2=4 sites way
    # tmax2_num = 1.0 / (w2*np.exp(- ((r2max / 2)-2) / x) * N)
    tmin3_num = 1.0 / (w3*np.exp(- (2.0   / 1) / x))
    tmax3_num = 1.0 / (w3*np.exp(- (2.0   / 1) / x))

    # tmin1_num = 1.0 / w1
    # tmax1_num = 1.0 / w1
    # tmax2_num = 1.0/(np.exp(-(r2max/2.0) / x) * w2 * N) # Time before an exchange happens between the edge and one past the center from the central bond hits the edge
    # tmid2_num = 1.0/(np.exp(-(r2max/4.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
    # tmin2_num = 1.0/(np.exp(-(1.0  /2.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
    # tmin3_num = 1.0 / w3
    # tmax3_num = 1.0 / w3

    t.time_num_lnlnt_begin = np.max([tmin1_num, tmin2_num, tmin3_num])
    t.time_num_lnlnt_cease = tmid2_num
    t.time_num_saturated = np.max([tmax1_num, tmax2_num, tmax3_num])
    t.idx_num_lnlnt_begin = np.argmax(tdata.astype(float) >= t.time_num_lnlnt_begin) - 1
    t.idx_num_lnlnt_cease = np.argmax(tdata.astype(float) >= t.time_num_lnlnt_cease) - 1
    t.idx_num_saturated = np.argmax(tdata.astype(float) >= t.time_num_saturated) - 1
    # t.idx_num_lnlnt_begin = np.max([t.idx_num_lnlnt_begin, 0])  # Make sure its non-negative
    # t.idx_num_lnlnt_cease = np.max([t.idx_num_lnlnt_cease, 0])  # Make sure its non-negative
    # t.idx_num_saturated   = np.max([t.idx_num_saturated, 0])  # Make sure its non-negative

    return t

def get_timepoints(tdata, db):

    if len(tdata) == 1:
        return 0, 0
    enttsb = None
    numtsb = None
    try:
        if not 'cachedir' in db['vals']:
            raise LookupError("Could not find 'cachedir' in db['vals']")
        entfile = "{}/tsat_{}|{}_L[{}]_x[{}]_w[{}]_r[{}]_f[{}]_l[{}].json".format(db['vals']['cachedir'],
                                                                                  db['vals']['filename'].replace('/', '|'),
                                                                               'entanglement_entropy',
                                                                               db['vals']['L'],
                                                                               db['vals']['x'],
                                                                               db['vals']['w'],
                                                                               db['vals']['r'],
                                                                               db['vals']['f'],
                                                                               db['vals']['l'])
        numfile = "{}/tsat_{}|{}_L[{}]_x[{}]_w[{}]_r[{}]_f[{}]_l[{}].json".format(db['vals']['cachedir'],
                                                                               db['vals']['filename'].replace('/', '|'),
                                                                               'number_entropy',
                                                                               db['vals']['L'],
                                                                               db['vals']['x'],
                                                                               db['vals']['w'],
                                                                               db['vals']['r'],
                                                                               db['vals']['f'],
                                                                               db['vals']['l'])

        with open(entfile, 'r') as fp:
            entjson = json.load(fp)
            enttsb = entropy_saturation_bootstrap(**entjson)
        with open(numfile, 'r') as fp:
            numjson = json.load(fp)
            numtsb = entropy_saturation_bootstrap(**numjson)
    except Exception as ex:
        print(f"Error: {ex}")
        print(f"Failed to load saturation times: try running plot_tsat.py first!")
        pass
        # raise FileNotFoundError(f"{ex}\n"
        #                         f"Failed to load saturation times: try running plot_tsat.py first!")



    L = db['vals']['L']
    # r = np.max(db['vals']['r'])
    x = db['vals']['x']
    wn = db['vals']['w']


    w1 = wn[0]  # The width of distribution for on-site field.
    w2 = wn[1]  # The width of distribution for pairwise interactions. The distribution is either U(J2_mean-w,J2_mean+w) or N(J2_mean,w)
    w3 = wn[2]  # The width of distribution for three-body interactions.

    # if r == np.iinfo(np.uint64).max or r == 'L':
    #     r = L
    r = L # We get the correct times if we use r == L,
    r2max = np.float64(np.min([r, L]))  # Number of sites from the center site to the edge site, max(|i-j|)/2
    N = np.sqrt(2.0 / np.pi) # Factor coming from folded normal distribution

    t = timepoints(time=tdata)
    # t.time = tdata
    w1 = 1.0 if w1 == 0.0 else w1
    w2 = 1.0 if w2 == 0.0 else w2
    w3 = 1.0 if w3 == 0.0 else w3


    tmin1_ent = 1.0 / (w1*np.exp(0 / x))
    tmax1_ent = 1.0 / (w1*np.exp(0 / x))
    tmin2_ent = 1.0 / (w2*np.exp(-1.0 / x))
    tmid2_ent = 1.0 / (w2*np.exp(- (r2max / 2)  / x) * N)
    # tmax2_ent = 1.0 / (w2*np.exp(- (r2max - 1 ) / x) * N)
    tmax2_ent = 0.027 * np.exp(L/0.99)
    # print('RETURN THE SHIFT-VALUE OF TMAX2_ENT')
    tmin3_ent = 1.0 / (w3*np.exp(-2.0 / x))
    tmax3_ent = 1.0 / (w3*np.exp(-2.0 / x))

    t.time_ent_lnt_begin = np.max([tmin1_ent, tmin2_ent, tmin3_ent])
    t.time_ent_lnt_cease = tmid2_ent
    t.idx_ent_lnt_begin = np.argmax(tdata.astype(float) >= t.time_ent_lnt_begin) - 1
    t.idx_ent_lnt_cease = np.argmax(tdata.astype(float) >= t.time_ent_lnt_cease) - 1
    if enttsb is None:
        t.time_ent_saturated = np.max([tmax1_ent, tmax2_ent, tmax3_ent])
        t.idx_ent_saturated = np.argmax(tdata.astype(float) >= t.time_ent_saturated) -1
    else:
        t.time_ent_saturated = enttsb.tsat_boot_avg  # np.max([tmax1_ent, tmax2_ent, tmax3_ent])
        t.idx_ent_saturated = round(enttsb.tsat_boot_avg_idx) #np.argmax(tdata.astype(float) >= t.time_ent_saturated) -1

        t.time_ent_lnt_cease = np.sqrt(enttsb.tsat_boot_avg)
        # t.time_ent_lnt_cease = numtsb.tsat_boot_avg
        t.idx_ent_lnt_cease = np.argmax(tdata.astype(float) >= t.time_ent_lnt_cease) - 1




    t.idx_ent_lnt_begin = np.max([t.idx_ent_lnt_begin, 0])  # Make sure it is non-negative
    t.idx_ent_lnt_cease = np.max([t.idx_ent_lnt_cease, 0])  # Make sure it is non-negative
    t.idx_ent_saturated = np.max([t.idx_ent_saturated, 0])  # Make sure it is non-negative

    # Now for the number entropy
    cfit_num =  0.08
    xfit_num = 2.0
    tmin1_num = 1.0 / (w1*np.exp(0 / x))
    tmax1_num = 1.0 / (w1*np.exp(0 / x))
    tmin2_num = 1.0 / (w2*np.exp(- (1.0   / 2) / x) * N)
    tmid2_num = 1.0 / (w2 * np.exp(- ((r2max / 4)) / x) * N)
    # Subtract two:
    #   In 010101|010101 the particle.
    #   (OLD) The half-chain particle is a distance |i-j| = |(L/2-1) - 1| away from the furthest particle on the same side.
    #   Excluding the particle at the edge, the most remote pair of particles are at |i-j| = |(L-1) - 2 - 1| = |L-4|
    #   away from the furthest particle on the same side.
    #   We exclude the particle at the edge because it is too restricted.
    # tmax2_num = 1.0 / (w2 * np.exp(-((r2max / 2)-2.5) / x) * N)
    # tmax2_num = 1.0 / (w2 * np.exp(-((r2max-4) / 2) / x) * N)
    tmax2_num = 0.008 * np.exp(L/1.98)

    # tmid2_num = tmid2_ent ** 0.5
    # tmax2_num = tmax2_ent ** 0.5


    # tmid2_num = 1.0 / (w2*np.exp(- ((r2max / 4)) / x) * N)
    # Subtract two: In 01010|10101 excluding the edge particle, the most distant "1"'s are L/2-2=4 sites way
    # tmax2_num = 1.0 / (w2*np.exp(- ((r2max / 2)-2) / x) * N)
    tmin3_num = 1.0 / (w3*np.exp(- (2.0   / 1) / x))
    tmax3_num = 1.0 / (w3*np.exp(- (2.0   / 1) / x))

    # tmin1_num = 1.0 / w1
    # tmax1_num = 1.0 / w1
    # tmax2_num = 1.0/(np.exp(-(r2max/2.0) / x) * w2 * N) # Time before an exchange happens between the edge and one past the center from the central bond hits the edge
    # tmid2_num = 1.0/(np.exp(-(r2max/4.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
    # tmin2_num = 1.0/(np.exp(-(1.0  /2.0) / x) * w2 * N) # Time before entanglement from the central bond hits the edge
    # tmin3_num = 1.0 / w3
    # tmax3_num = 1.0 / w3

    t.time_num_lnlnt_begin = np.max([tmin1_num, tmin2_num, tmin3_num])
    t.time_num_lnlnt_cease = tmid2_num
    t.idx_num_lnlnt_begin = np.argmax(tdata.astype(float) >= t.time_num_lnlnt_begin) - 1
    t.idx_num_lnlnt_cease = np.argmax(tdata.astype(float) >= t.time_num_lnlnt_cease) - 1
    if numtsb is None:
        t.time_num_saturated =  np.max([tmax1_num, tmax2_num, tmax3_num])
        t.idx_num_saturated = np.argmax(tdata.astype(float) >= t.time_num_saturated) - 1
    else:
        t.time_num_saturated =  numtsb.tsat_boot_avg #np.max([tmax1_num, tmax2_num, tmax3_num])
        t.idx_num_saturated = round(numtsb.tsat_boot_avg_idx)

    return t


def midchain_page_entropy(L):
    return L/2 * np.log(2) - 0.5

# @njit(parallel=True, cache=True)
def page_entropy(L):
    return L / 2 * np.log(2) - 0.5
#
#    n = int(2 ** (L / 2))
#    S = - float(n - 1) / (2 * n)
#    for k in prange(1 + n, 1 + n ** 2):  # Include last
#        S = S + 1.0 / k
#    return S


def get_filtered_list(key, vals, filter):
    if vals is None:
        return None
    if filter is None:
        return vals
    elif isinstance(filter, dict):
        for fkey, fvals in filter.items():
            if fkey in key:
                return [v for v in fvals if v in vals]
    return vals


def get_prop(db, keyfmt, prop, filter=None):
    if isinstance(keyfmt, list) or isinstance(keyfmt, set) or isinstance(keyfmt, tuple):
        keys = []
        fmts = []
        for kf in keyfmt:
            if ':' in kf:
                k, f = kf.split(':')
            else:
                k, f = kf.split(':')[0], ''
            if prop == 'keys' and db.get('version') >= 3:
                keys.append(k)
                fmts.append(f)
            elif prop in db:
                v = db[prop].get(k) if k in db[prop] else db[prop].get('keys').get(k)
                v = get_filtered_list(k, v, filter)
                if v is not None:
                    keys.append(v)
                    fmts.append(f)
        return keys, fmts
    else:
        if prop in db:
            if ':' in keyfmt:
                k, f = keyfmt.split(':')
            else:
                k, f = keyfmt.split(':')[0], ''
            v = db[prop].get(k) if k in db[prop] else db[prop].get('keys').get(k)
            v = get_filtered_list(k, v, filter)
            if v is not None:
                return v, f
    return [], []


def get_vals(db, keyfmt, filter=None):
    vals, _ = get_prop(db, keyfmt, 'vals', filter)
    return vals


def get_keys(db, keyfmt):
    keys, _ = get_prop(db, keyfmt, 'keys')
    return keys


def get_tex(db, keyfmt):
    keys, _ = get_prop(db, keyfmt, 'tex')
    if isinstance(keys, list):
        keys = '$' + ', '.join([k.strip('$') for k in keys]) + '$'
    return keys


def get_specvals(db, keyfmt, vals=None, filter=None):
    if vals is not None and len(keyfmt) != len(vals):
        raise AssertionError("unequal lengths: keyfmt {} != vals {} ".format(len(keyfmt), len(vals)))
    keys, fmts = get_prop(db, keyfmt, 'keys')
    kv = []
    if vals is None:
        for key in keys:
            kv.append("{}".format(key))
        return ','.join(map(str, kv))
    else:
        for key, fmt, val in zip(keys, fmts, vals):
            kv.append("{0}[{2:{1}}]".format(key, fmt, val))
        return ','.join(map(str, kv))


def get_title(db, keys, width=20):
    fmtvals = []
    newline = 0
    for s in keys:
        key, fmt = s.split(':') if ':' in s else [s, '']
        if key in db['vals'] and key in db['tex']['keys']:
            if isinstance(db['vals'][key], list):
                fmtvals.append('${}=[{}]$'.format(
                    db['tex']['keys'][key].strip('$'),
                    ','.join('{0:{1}}'.format(val, fmt) for val in db['vals'][key])))
            else:
                if key == 'r' and db['vals'][key] == np.iinfo(np.uint64).max:
                    fmtvals.append(db['tex']['eqs'][key])
                else:
                    fmtvals.append(
                        '${}={}$'.format(db['tex']['keys'][key].strip('$'), '{0:{1}}'.format(db['vals'][key], fmt)))
        elif key in db['tex']['eqs']:
            fmtvals.append(db['tex']['eqs'][key])
        if len(fmtvals) - newline >= width:
            newline = len(fmtvals)
            fmtvals.append('\n')
    return ', '.join(fmtvals)


def get_safe_table_field_name(node, keys):
    # Returns all keys that exist in node
    match = []
    if isinstance(keys, list):
        match = [field for field in keys if field in node.dtype.fields.keys()]
    else:
        if keys in node.dtype.fields.keys():
            match.append(keys)
    if not match:
        raise LookupError('Field not found in table:\n'
                          'Table   : {}\n'
                          'Expected: {}\n'
                          'Existing: {}'.format(node.name, keys, node.dtype.fields.keys()))
    return match


def get_table_data(node, keys=None, dtype='f8', transpose=None):
    if keys == None:
        return node[()].ravel().view((dtype, (1,)))[()], np.array([None], ndmin=2)

    elif node.dtype.fields == None:
        # This is a scalar put inside an ndarray, so that we have
        # a consistent return type
        return np.array(node[()]), np.array([None], ndmin=2)
    else:
        # Find the column names that match keys
        cols = get_safe_table_field_name(node, keys)
        return node.fields(cols)[()].view(dtype=(dtype, (len(cols),)))[()], np.array(cols)


def get_legend_row(db, datanode, legend_col_keys):
    legendrow = [None] * len(legend_col_keys)
    dbval = db['dsets'][datanode.name]
    for idx, key in enumerate(legend_col_keys):
        key, fmt = key.split(':') if ':' in key else [key, '']
        sfx = 'm' if 'time' in key or 'tsim' in key else ''
        if '?' in fmt:
            continue
        if key in dbval['vals'] and fmt:
            legendrow[idx] = '{0:{1}}{2}'.format(dbval['vals'][key], fmt, sfx)
        elif key in dbval['tex']['vals']:
            legendrow[idx] = dbval['tex']['vals'][key]

        if legendrow[idx] is None:
            print(legend_col_keys)
            print(legendrow)
            print(dbval['tex'])
            raise Exception("Failed to get legend row")
    legendrow = [key for key in legendrow if key is not None]
    return legendrow


def get_fig_meta(numplots: int, meta: dict):
    f = {
        'fig': None,
        'filename': meta.get('filename'),
        'lstyles': get_linestyles(),
        'constrained_layout': get_default(meta,'constrained_layout'),
        'numplots': numplots,
        'nrows': None,
        'ncols': None,
        'font.size': get_default(meta,'font.size'),
        'figsize':  get_default(meta,'figsize'),
        'figcount': 0, # How many times we have used this fig to plot something
        'frameon' : meta.get('frameon'),
        'box_aspect': get_default(meta,'box_aspect'),
        'crows': 1,  # Common row put at the bottom of all subplots
        'ccols': 1,  # Common col put at the right of all subplots
        'irows': 2,  # We make a 2x2 grid where 0,0 is the plot, and 0,1 and 1,0 are legends
        'icols': 2,  # We make a 2x2 grid where 0,0 is the plot, and 0,1 and 1,0 are legends
        'owr': None,
        'ohr': None,
        'iwr': [10000, 1],  # Width ratio between plot and right legend
        'ihr': [10000, 1],  # Height ratio between plot and bottom legend
        'owr_pad': meta.get('owr_pad') if 'owr_pad' in meta else 1.00,#1.00,
        'ohr_pad': meta.get('ohr_pad') if 'ohr_pad' in meta else 1.00,#1.00,

        'go': None,  # Outer gridspec
        'gi': [],  # Inner gridspec
        'rc': [],  # List of coordinates "(row,col)" for each ax
        'ax': [],  # List of subplots with plots (i.e. [0,0] in gsi)
        'ix': [],  # List of insets, one for each subplot axis
        'lr': [],  # List of subplots with legend right (i.e. [0,1] in gsi)
        'lb': [],  # List of subplots with legend below (i.e. [1,0] in gsi)
        'lc': [],  # List of subplots with legend common to all subplots
        'suptitle': meta.get('suptitle'),
        'ymax': meta.get('ymax'),
        'ymin': meta.get('ymin'),
        'xmax': meta.get('xmax'),
        'xmin': meta.get('xmin'),
        'sharex': meta.get('sharex') if 'sharex' in meta else 'all',
        'sharey': meta.get('sharey') if 'sharey' in meta else 'all',
        'xscale': meta.get('xscale') if 'xscale' in meta else 'linear',
        'yscale': meta.get('yscale') if 'yscale' in meta else 'linear',
        'xnopos': meta.get('xnopos'),
        'ynopos': meta.get('ynopos'),
        'xmaloc': meta.get('xmaloc'),
        'ymaloc': meta.get('ymaloc'),
        'xmiloc': meta.get('xmiloc'),
        'ymiloc': meta.get('ymiloc'),
        'xmafmt': meta.get('xmafmt'),
        'ymafmt': meta.get('ymafmt'),
        'xmifmt': meta.get('xmifmt'),
        'ymifmt': meta.get('ymifmt'),
        'xformat': meta.get('xformat'),
        'yformat': meta.get('yformat'),
        'xticks': meta.get('xticks'),
        'yticks': meta.get('yticks'),
        'xticklabels': meta.get('xticklabels'),
        'yticklabels': meta.get('yticklabels'),
        'xlabel': meta.get('xlabel'),
        'ylabel': meta.get('ylabel'),
        'xlabelpad': meta.get('xlabelpad'),
        'ylabelpad': meta.get('ylabelpad'),
        'xcoords': meta.get('xcoords'),
        'ycoords': meta.get('ycoords'),
        'ylabel_inner_visible' : meta.get('ylabel_inner_visible'),
        'ymarkoffset': [],
        'yticklength': meta.get('yticklength'),
        'xticklength': meta.get('xticklength'),
        'ytickparams': meta.get('ytickparams'),
        'xtickparams': meta.get('xtickparams'),
        'axes_used': [],
        'legends': [],
        'legendshow': meta.get('legendshow'),
        'legendshow2': meta.get('legendshow2'),
        'legendoutside': get_default(meta, 'legendoutside'),
        'legendcollect': get_default(meta, 'legendcollect'),
        'legendlocation': meta.get('legendlocation'),
        'legend2location': meta.get('legend2location'),
        'legendtitle': meta.get('legendtitle'),
        'legendtitle2': meta.get('legendtitle2'),
        'bbox_to_anchor': meta.get('bbox_to_anchor'),
        'bbox_to_anchor2': meta.get('bbox_to_anchor2'),
        'figlegend': meta.get('figlegend'),
    }
    logger.info('Generated f dict: \n {}'.format(f))
    # Initialize a figure. The size is taken from the stylesheet
    # constrained_layout will make sure to add objects to fill the area
    f['fig'] = plt.figure(figsize=f.get('figsize'), constrained_layout=f.get('constrained_layout'),
                          #tight_layout = {'pad': 0}
                          )
    if fontsize := f.get('font.size'):
        plt.rcParams.update({'font.size': fontsize})

    if s := get_default(meta,'subplots'):
        plt.subplots_adjust(
            left=s.get('left'),
            bottom=s.get('bottom'),
            right=s.get('right'),
            top=s.get('top'),
            wspace=s.get('wspace'),
            hspace=s.get('hspace')
        )
    # Set padding between subplots
    # w_pad: Width padding in inches.
    #        This is the pad around Axes and is meant to make sure there is enough room for fonts to look good.
    #        Defaults to 3 pts = 0.04167 inches
    # h_pad: Height padding in inches. Defaults to 3 pts.
    # wspace: Width padding between subplots, expressed as a fraction of the subplot width.
    #         The total padding ends up being w_pad + wspace.
    # hspace: Height padding between subplots, expressed as a fraction of the subplot width.
    #         The total padding ends up being h_pad + hspace.
    # , sharex = True, sharey = True
    # f['fig'].set_constrained_layout_pads(w_pad=1 / 72, h_pad=1 / 72, hspace=0, wspace=0)
    f['fig'].set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)

    # Get the outer grid rows and columns
    f['nrows'], f['ncols'] = get_optimal_subplot_num(numplots=numplots)

    # Set width ratios for outer and inner grid rows
    f['owr'] = [1.0] * (f['ncols'] + f['ccols'])
    f['ohr'] = [1.0] * (f['nrows'] + f['crows'])
    f['owr'][0] *= f['owr_pad']  # The left-most subplot needs more space for the ylabel (NEEDS FINE TUNING)
    f['ohr'][-2] *= f['ohr_pad']  # The bottom subplot needs more space for the xlabel (NEEDS FINE TUNING)
    f['owr'][-1] = 0.0001  # The right common area can be small to begin with
    f['ohr'][-1] = 0.0001  # The bottom common area can be small to begin with

    # Create the outer subplot grid  ([g]rid[o]uter)
    f['go'] = f['fig'].add_gridspec(nrows=f['nrows'] + f['crows'], ncols=f['ncols'] + f['ccols'], width_ratios=f['owr'],
                                    height_ratios=f['ohr'], wspace=0.00, hspace=0.00)

    # Generate the outer grid common areas
    f['lc'].append(f['fig'].add_subplot(f['go'][:, -1]))  # The right common area
    f['lc'].append(f['fig'].add_subplot(f['go'][-1, :]))  # The bottom common area
    # Turn off their axes
    for lc in f['lc']:
        lc.axis('off')

    # Keep track of the axes rows/cols
    ax_matrix = np.empty(shape=(f['nrows'], f['ncols']), dtype=plt.Axes)

    # Generate the inner grids
    for ir, ic in np.ndindex(f['nrows'], f['ncols']):
        go = f['go'][ir, ic]
        logger.info('Setting up outer gridspec: row {} col {}'.format(ir, ic))
        f['gi'].append(go.subgridspec(nrows=f['irows'], ncols=f['icols'], width_ratios=f['iwr'], height_ratios=f['ihr'],
                                      wspace=0.00, hspace=0.00))

        gi = f['gi'][-1]
        # Setup axis sharing
        sharex = None
        sharey = None
        if shx := f.get('sharex'):
            if shx == 'col':
                sharex = ax_matrix[0, ic]
            if shx == 'all' or shx is True:
                sharex = ax_matrix[0, 0]
        if shy := f.get('sharey'):
            if shy == 'row':
                sharey = ax_matrix[ir, 0]
            if shy == 'all' or shy is True:
                sharey = ax_matrix[0, 0]

        f['ax'].append(f['fig'].add_subplot(gi[0, 0], sharex=sharex, sharey=sharey))
        f['ix'].append(None)
        f['lr'].append(f['fig'].add_subplot(gi[0, 1]))
        f['lb'].append(f['fig'].add_subplot(gi[1, 0]))
        ax_matrix[ir, ic] = f['ax'][-1]

        # Turn off axis elements on the legend box
        f['lr'][-1].axis('off')
        f['lb'][-1].axis('off')

        if box_aspect := f['box_aspect']:
            f['ax'][-1].set_box_aspect(box_aspect)

        if xscale := f.get('xscale'):
            if xnopos := f.get('xnopos'):
                f['ax'][-1].set_xscale(xscale, nonpositive=xnopos)
            else:
                f['ax'][-1].set_xscale(xscale)
        if yscale := f.get('yscale'):
            if ynopos := f.get('ynopos'):
                f['ax'][-1].set_yscale(yscale, nonpositive=ynopos)
            else:
                f['ax'][-1].set_yscale(yscale)

        if f['ymax'] is not None:
            f['ax'][-1].set_ylim(ymax=f['ymax'])
        if f['ymin'] is not None:
            f['ax'][-1].set_ylim(ymin=f['ymin'])
        if f['xmax'] is not None:
            f['ax'][-1].set_xlim(xmax=f['xmax'])
        if f['xmin'] is not None:
            f['ax'][-1].set_xlim(xmin=f['xmin'])

        if ymaloc := f.get('ymaloc'):
            f['ax'][-1].yaxis.set_major_locator(ymaloc)
        if xmaloc := f.get('xmaloc'):
            f['ax'][-1].xaxis.set_major_locator(xmaloc)
        if ymiloc := f.get('ymiloc'):
            f['ax'][-1].yaxis.set_minor_locator(ymiloc)
        if xmiloc := f.get('xmiloc'):
            f['ax'][-1].xaxis.set_minor_locator(xmiloc)

        if f.get('xticks') is not None:
            f['ax'][-1].set_xticks(f.get('xticks'))
        if f.get('xticklabels') is not None:
            f['ax'][-1].set_xticklabels(f.get('xticklabels'))
        if f.get('yticks') is not None:
            f['ax'][-1].set_yticks(f.get('yticks'))
        if f.get('yticklabels') is not None:
            f['ax'][-1].set_yticklabels(f.get('yticklabels'))
        if f.get('xlabel') is not None:
            f['ax'][-1].set_xlabel(f.get('xlabel'), labelpad=f.get('xlabelpad'))
        if f.get('ylabel') is not None:
            f['ax'][-1].set_ylabel(f.get('ylabel'), labelpad=f.get('ylabelpad'))
        if yformat := f.get('yformat'):
            f['ax'][-1].yaxis.set_major_formatter(FormatStrFormatter(yformat))
        if xformat := f.get('xformat'):
            f['ax'][-1].yaxis.set_major_formatter(FormatStrFormatter(xformat))
        if ymafmt := f.get('ymafmt'):
            f['ax'][-1].yaxis.set_major_formatter(ymafmt)
        if xmafmt := f.get('xmafmt'):
            f['ax'][-1].xaxis.set_major_formatter(xmafmt)
        if ymifmt := f.get('ymifmt'):
            f['ax'][-1].yaxis.set_minor_formatter(ymifmt)
        if xmifmt := f.get('xmifmt'):
            f['ax'][-1].xaxis.set_minor_formatter(xmifmt)
        if ycoords := f.get('ycoords'):
            f['ax'][-1].yaxis.set_label_coords(x=ycoords[0], y=ycoords[1])
        if xcoords := f.get('xcoords'):
            f['ax'][-1].xaxis.set_label_coords(x=xcoords[0], y=xcoords[1])
        if f.get('xticklength') is not None:
            f['ax'][-1].tick_params(axis='x', which='both', length=f.get('xticklength'))
        if f.get('yticklength') is not None:
            f['ax'][-1].tick_params(axis='y', which='both', length=f.get('yticklength'))

        is_last_row = ir + 1 == f['nrows']
        is_first_col = ic == 0
        if not is_last_row and f.get('sharex') in ['col', 'all', True]:
            f['ax'][-1].tick_params(axis='x', which='both', labelbottom=False, zorder=0)
            f['ax'][-1].xaxis.label.set_visible(False)

        if not is_first_col and f.get('sharey') in ['row', 'all', True]:
            f['ax'][-1].tick_params(axis='y', which='both', labelleft=False, zorder=0)
            f['ax'][-1].yaxis.label.set_visible(False)
        if not is_first_col and f.get('ylabel_inner_visible') is False:
            f['ax'][-1].yaxis.label.set_visible(False)


        if ytp := f.get('ytickparams'):
            if not is_first_col:
                f['ax'][-1].set_ylabel('')
                f['ax'][-1].tick_params(axis='y',direction=ytp.get('direction'), pad=ytp.get('pad'))
                if ytp.get('tick_right'):
                    f['ax'][-1].yaxis.tick_right()
                    f['ax'][-1].yaxis.set_ticks_position('both')
        if xtp := f.get('xtickparams'):
            if not is_last_row:
                f['ax'][-1].set_xlabel('')
                f['ax'][-1].tick_params(axis='x',direction=xtp.get('direction'), pad=xtp.get('pad'))


    # Set title
    if title := f.get('suptitle'):
        f['fig'].suptitle(title)

    legend = {'handle': [], 'label': [], 'title': None, 'header': None}  # One such per legend column

    f['legends'] = {}
    for i, ax in enumerate(f['ax']):
        # We should probably never need more than 10 legend columns per ax
        f['legends'][i] = {}
        for j in range(10):
            f['legends'][i][j] = deepcopy(legend)  # i  for ax, j for legend column
    f['legends2'] = {}
    for i, ax in enumerate(f['ax']):
        # We should probably never need more than 10 legend columns per ax
        f['legends2'][i] = {}
        for j in range(10):
            f['legends2'][i][j] = deepcopy(legend)  # i  for ax, j for legend column


    # m1_legend: {'handle': [], 'label': [], 'ncol': 1, 'unique': True, 'loc': 'upper left', 'insubfig': False,
    #           'title': ["$t_{}$".format('{\ln\ln}')]},

    return f


def get_formatted_columns(columns):
    # columns is a list of legend columns like

    #     [[L, 8, 12, 16 ...],
    #      [n, 2400, 2400, 2400 ...]]
    #
    # The first entry in each legend column is its title
    # This function then returns a list of formatted rows like this
    #
    #       [['L  n   '],
    #        ['8  2400'],
    #        ['12 2400'],
    #        ['16 2400'],
    #
    # Where each column has been formatted to uniform width.

    # First, we check that each list has the same length
    num_cols = len(columns)
    if num_cols == 0:
        return []
    num_rows = len(columns[0])  # Number of labels in each column
    if not all(len(x) == num_rows for x in columns):
        print("columns shape: ", np.shape(columns))
        for i, col in enumerate(columns):
            print("-- col {}: shape {}: {}".format(i, np.shape(col), col))
        raise ValueError("All columns must be of the same length:\n" + str(columns))

    # Now we get the maximum string length of each column
    column_longest = [0] * num_cols
    column_phantom = [''] * num_cols  # The longest string in each column, not including the title
    for icol, col in enumerate(columns):  # Iterate columns
        maxlenstr = max([str(c) for c in col[1:]], key=len)  # Get the longest string in the column
        column_longest[icol] = len(maxlenstr)
        column_phantom[icol] = maxlenstr
    # Now we know how wide each column needs to be, So we can now format each row
    fmt_rows = []
    for irow in range(num_rows):  # Iterate the rows in each column
        fmt_label = ''
        for icol in range(num_cols):  # Iterate columns
            c = columns[icol][irow]  # An entry at irow,icol
            # Let n == len(str(c)) and m == phantom_length[i] be the number of characters in the longest column word
            # Then we need to print c padded with a phantom word of length m-n
            zerolen_string = '\\makebox[0pt][l]{{{}}}'.format(
                str(c))  # Put in text in left-aligned box that does not consume width
            phantom_string = '\\hphantom{{{}}}'.format(
                column_phantom[icol])  # Let the phantom string consume width instead
            colsep_string = ' ' if icol + 1 < num_cols else ''
            fmt_label += '{}{}{}'.format(zerolen_string, phantom_string, colsep_string)
        fmt_rows.append(fmt_label)
    return fmt_rows


def px_to_in(w, h):
    return 27. / 2560 * w, 27. / 1440 * h


def px_to_cm(w, h):
    w, h = px_to_in(w, h)
    return 2.54 * w, 2.54 * h


def labels_are_equal(lgnd_meta):
    # Gather all the labels for the different kinds of legend keys (e.g. "l", "m" type legends)
    labels = {}
    for idx, lgnd in enumerate(lgnd_meta):
        for lkey in lgnd['legend_keys']:
            label = lgnd[lkey]  # The legend description dict for this type of legend
            if not lkey in labels:
                labels[lkey] = []
            labels[lkey].append(label['label'])
    labels_equal = {}
    for idx, (lkey, label_list) in enumerate(labels.items()):
        l0 = label_list[0]
        labels_equal[lkey] = all(l == l0 for l in label_list)
    return labels_equal


def columns_are_equal(fmeta):
    # Gather all the labels for the different kinds of legend keys (e.g. "l", "m" type legends)
    # This will contain True, True, False, True... etc for each column,
    # where True/False tells whether this column is equal on all axes
    numaxes = len(fmeta['legends'].keys())
    numcols = len(fmeta['legends'][0].keys())
    columns_equal = []
    for icol in range(numcols):
        l0 = fmeta['legends'][0][icol]['label']
        columns_equal.append(
            all(fmeta['legends'][0][icol]['label'] == fmeta['legends'][iax][icol]['label'] for iax in range(numaxes)))

    return columns_equal
def columns_are_equal2(fmeta):
    # Gather all the labels for the different kinds of legend keys (e.g. "l", "m" type legends)
    # This will contain True, True, False, True... etc for each column,
    # where True/False tells whether this column is equal on all axes
    numaxes = len(fmeta['legends2'].keys())
    numcols = len(fmeta['legends2'][0].keys())
    columns_equal = []
    for icol in range(numcols):
        l0 = fmeta['legends2'][0][icol]['label']
        columns_equal.append(
            all(fmeta['legends2'][0][icol]['label'] == fmeta['legends2'][iax][icol]['label'] for iax in range(numaxes)))

    return columns_equal



def add_legend5(fmeta):
    if fmeta.get('legendshow') is False and fmeta.get('legendshow2') is False:
        return fmeta
    # meta should be a dict with keys 'ax' with the corresponding axes and 'legends' which names the extra legends
    # for (oidx,gso),(nrow,ncol) in zip(enumerate(fmeta['gso']), np.ndindex((fmeta['nrows'], fmeta['ncols']))):
    n = len(fmeta['ax'])
    if not all(len(x) == n for x in [fmeta['ax'], fmeta['lr'], fmeta['lb'], fmeta['legends']]):
        raise AssertionError("Not equal lengths")
    numaxes = len(fmeta['legends'].keys())
    numcols = len(fmeta['legends'][0].keys())
    columns_equal = columns_are_equal(fmeta)
    columns_equal2 = columns_are_equal(fmeta)
    eqidx = [i for i, x in enumerate(columns_equal) if x is True]  # The indices of columns that are equal
    nqidx = [i for i, x in enumerate(columns_equal) if x is False]  # The indices of columns that are not equal
    eqidx2 = [i for i, x in enumerate(columns_equal2) if x is True]  # The indices of columns that are equal
    nqidx2 = [i for i, x in enumerate(columns_equal2) if x is False]  # The indices of columns that are not equal

    # Let's first treat the common columns that can be factored out

    outside = fmeta.get('legendoutside')  # Put legend inside or outside each subplot
    collect = fmeta.get('legendcollect')  # Collect equal columns into a single legend outside
    location = fmeta.get('legendlocation')
    location2 = fmeta.get('legend2location')
    figlegend = fmeta.get('figlegend')
    bbox_to_anchor=fmeta.get('bbox_to_anchor')
    bbox_to_anchor2=fmeta.get('bbox_to_anchor2')
    legend_ax = 'lr' if outside else 'ax'
    legend_eq = 'lc' if collect else legend_ax
    legend_nq = legend_ax
    loc_nq = 'center left' if outside else (location if location is not None else 'best')
    loc2_nq = 'center left' if outside else (location2 if location2 is not None else 'best')
    loc_eq = 'center left' if collect else loc_nq
    loc2_eq = 'center left' if collect else loc2_nq
    anchor = (0.0,0.5) if outside else bbox_to_anchor
    anchor2 = (0.0,0.5) if outside else bbox_to_anchor2

    if legend_eq == legend_nq:
        # If legends all go together anyway, we may as well put them together again
        eqidx.extend(nqidx)
        eqidx2.extend(nqidx2)
        nqidx = []
        nqidx2 = []
        eqidx.sort()
        eqidx2.sort()

    iax_tgt = None
    if collect:
        # There may be an unused axis where we can put the common legend instead of lc
        for iax, ax in enumerate(fmeta['ax']):
            if not iax in fmeta['axes_used']:
                legend_eq = 'ax'
                iax_tgt = iax  # Put the common legend on the unused axis instead
                fmeta['axes_used'].append(iax)
                # Remove axis elements
                ax.set(yticklabels=[], xticklabels=[], ylabel=None, xlabel=None, xticks=[], yticks=[])
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.axis('off')
                break

    lg = None
    if not fmeta.get('legendshow') is False:
        for iax in range(numaxes):
            # Collect the columns
            columns = []
            handles = []
            for icol in eqidx:
                title = fmeta['legends'][iax][icol]['title']
                label = fmeta['legends'][iax][icol]['label']
                handle = fmeta['legends'][iax][icol]['handle']
                if not title or not label or not handle:
                    continue
                columns.append([title] + label)
                handles.extend(handle)

                # Decide where to put it. ax, lr, lg or lc[0]/lc[1]
            if not columns or not handles:
                continue

            # pick a point in the middle, make it invisible with alpha = 0, and move to back with zorder so that
            # the legend entry appears in the beginning for instance with tikzplotlib
            xmin, xmax = fmeta['ax'][iax].get_xlim()
            ymin, ymax = fmeta['ax'][iax].get_ylim()
            titlepatch, = fmeta['ax'][iax].plot([0.5 * (xmin + xmax)], [0.5 * (ymin + ymax)], label=None, alpha=0.0,
                                                zorder=0
                                                )
            formatted_labels = get_formatted_columns(columns)
            legendtitle = fmeta.get('legendtitle')
            if legendtitle is None:
                legendtitle = fmeta['legends'][iax][0]['header']
            if iax < len(loc_eq):
                loc_eq_iax = loc_eq[iax] if isinstance(loc_eq, list) else loc_eq
                anchor_iax = anchor[iax] if isinstance(anchor, list) else anchor
            else:
                loc_eq_iax = loc_eq[-1] if isinstance(loc_eq, list) else loc_eq
                anchor_iax = anchor[-1] if isinstance(anchor, list) else anchor

            if iax_tgt is not None:
                iax = iax_tgt  # Put legends on common axis
            if figlegend:
                lg = fmeta['fig'].legend(handles=[titlepatch] + handles, labels=formatted_labels,
                                                  title=legendtitle,  # title=fmeta['legends'][iax][0]['header'],
                                                  loc=loc_eq_iax,
                                                  frameon=fmeta.get('frameon'),
                                                  prop=dict(stretch="ultra-condensed"))
            else:
                lg = fmeta[legend_eq][iax].legend(handles=[titlepatch] + handles, labels=formatted_labels,
                                                  title=legendtitle,  # title=fmeta['legends'][iax][0]['header'],
                                                  loc=loc_eq_iax,
                                                  frameon=fmeta.get('frameon'),
                                                  bbox_to_anchor=anchor_iax,
                                                  prop=dict(stretch="ultra-condensed"))
                numlegend2 = len(fmeta['legends2'][iax][0]['handle'])
                if numlegend2 > 0:
                    fmeta[legend_eq][iax].add_artist(lg)
            lg.set_zorder(100)
            lg._legend_box.align = "left"  # Align the legend title right (i.e. the title row)
            if collect:
                break

    # Legend2
    if not fmeta.get('legendshow2') is False:
        for iax in range(numaxes):
            # Collect the columns
            columns = []
            handles = []
            for icol in eqidx2:
                title = fmeta['legends2'][iax][icol]['title']
                label = fmeta['legends2'][iax][icol]['label']
                handle = fmeta['legends2'][iax][icol]['handle']
                if not title or not label or not handle:
                    continue
                columns.append([title] + label)
                handles.extend(handle)

                # Decide where to put it. ax, lr, lg or lc[0]/lc[1]
            if not columns or not handles:
                continue

            # pick a point in the middle, make it invisible with alpha = 0, and move to back with zorder so that
            # the legend entry appears in the beginning for instance with tikzplotlib
            xmin, xmax = fmeta['ax'][iax].get_xlim()
            ymin, ymax = fmeta['ax'][iax].get_ylim()
            titlepatch, = fmeta['ax'][iax].plot([0.5 * (xmin + xmax)], [0.5 * (ymin + ymax)], label=None, alpha=0.0,
                                                zorder=0
                                                )
            formatted_labels = get_formatted_columns(columns)
            legendtitle = fmeta.get('legend2title')
            if legendtitle is None:
                legendtitle = fmeta['legends2'][iax][0]['header']
            loc_eq_iax = loc2_eq[iax] if isinstance(loc2_eq, list) else loc2_eq
            anchor_iax = anchor2[iax] if isinstance(anchor2, list) else anchor2

            if iax_tgt is not None:
                iax = iax_tgt  # Put legends on common axis
            if figlegend:
                lg = fmeta['fig'].legend(handles=[titlepatch] + handles, labels=formatted_labels,
                                                  title=legendtitle,  # title=fmeta['legends'][iax][0]['header'],
                                                  loc=loc_eq_iax,
                                                  frameon=fmeta.get('frameon'),
                                                  prop=dict(stretch="ultra-condensed"))
            else:
                lg = fmeta[legend_eq][iax].legend(handles=[titlepatch] + handles, labels=formatted_labels,
                                                  title=legendtitle,  # title=fmeta['legends'][iax][0]['header'],
                                                  loc=loc_eq_iax,
                                                  frameon=fmeta.get('frameon'),
                                                  bbox_to_anchor=anchor_iax,
                                                  prop=dict(stretch="ultra-condensed"))

                # fmeta[legend_eq][iax].add_artist(lg2)
            lg._legend_box.align = "left"  # Align the legend title right (i.e. the title row)
            if collect:
                break


    # Now treat the unequal columns. These should go into each subplot
    for iax in range(numaxes):
        # Collect the columns
        columns = []
        handles = []
        for icol in nqidx:
            title = fmeta['legends'][iax][icol]['title']
            label = fmeta['legends'][iax][icol]['label']
            handle = fmeta['legends'][iax][icol]['handle']
            if not title or not label or not handle:
                continue
            columns.append([title] + label)
            handles.extend(handle)
        if not columns or not handles:
            continue
        # pick a point in the middle, make it invisible with alpha = 0, and move to back with zorder so that
        # the legend entry appears in the beginning for instance with tikzplotlib
        xmin, xmax = fmeta['ax'][iax].get_xlim()
        ymin, ymax = fmeta['ax'][iax].get_ylim()
        titlepatch, = fmeta['ax'][iax].plot([0.5 * (xmin + xmax)], [0.5 * (ymin + ymax)], label=None, alpha=0.0,
                                            zorder=0)
        formatted_labels = get_formatted_columns(columns)
        loc_nq_iax = loc_nq[iax] if isinstance(loc_nq, list) else loc_nq
        lg = fmeta[legend_nq][iax].legend(handles=[titlepatch] + handles, labels=formatted_labels,
                                          title=''.join(formatted_labels), loc=loc_nq_iax,
                                          prop=dict(stretch="ultra-condensed"))
        lg._legend_box.align = "left"  # Align the legend title right (i.e. the title row)

    return fmeta


def prettify_plot5(fmeta):
    for idx, (ax, lr, lb) in enumerate(zip(fmeta['ax'], fmeta['lr'], fmeta['lb'])):
        if not idx in fmeta['axes_used']:
            print('Removing axes {}  (used {})'.format(idx, fmeta['axes_used']))
            fmeta['fig'].delaxes(ax)
            fmeta['fig'].delaxes(lr)
            fmeta['fig'].delaxes(lb)
            logger.info('Deleting unused subplot idx {}'.format(idx))
    # Add legends
    add_legend5(fmeta=fmeta)

    # Remove legend boxes that are not used
    for idx, (lr, lb) in enumerate(zip(fmeta['lr'], fmeta['lb'])):
        # Get all legend objects from lr
        objs_lr = [c for c in lr.get_children() if isinstance(c, legend.Legend)]
        objs_lb = [c for c in lb.get_children() if isinstance(c, legend.Legend)]
        if len(objs_lr) == 0 and lr in fmeta['fig'].axes:
            fmeta['fig'].delaxes(lr)
            logger.info('Deleting unused lr idx {}'.format(idx))

        if len(objs_lb) == 0 and lb in fmeta['fig'].axes:
            fmeta['fig'].delaxes(lb)
            logger.info('Deleting unused lb idx {}'.format(idx))

    for idx, lc in enumerate(fmeta['lc']):
        objs_lc = [c for c in lc.get_children() if isinstance(c, legend.Legend)]
        if len(objs_lc) == 0 and lc in fmeta['fig'].axes:
            fmeta['fig'].delaxes(lc)
            logger.info('Deleting unused lc idx {}'.format(idx))

    for ax in fmeta['ax']:
        if yformat := fmeta.get('yformat'):
            ax.yaxis.set_major_formatter(FormatStrFormatter(yformat))
        if xformat := fmeta.get('xformat'):
            ax.yaxis.set_major_formatter(FormatStrFormatter(xformat))
        if ymafmt := fmeta.get('ymafmt'):
            ax.yaxis.set_major_formatter(ymafmt)
        if xmafmt := fmeta.get('xmafmt'):
            ax.xaxis.set_major_formatter(xmafmt)
        if ymifmt := fmeta.get('ymifmt'):
            ax.yaxis.set_minor_formatter(ymifmt)
        if xmifmt := fmeta.get('xmifmt'):
            ax.xaxis.set_minor_formatter(xmifmt)
        if ymaloc := fmeta.get('ymaloc'):
            ax.yaxis.set_major_locator(ymaloc)
        if xmaloc := fmeta.get('xmaloc'):
            ax.xaxis.set_major_locator(xmaloc)
        if ymiloc := fmeta.get('ymiloc'):
            ax.yaxis.set_minor_locator(ymiloc)
        if xmiloc := fmeta.get('xmiloc'):
            ax.xaxis.set_minor_locator(xmiloc)
        # f['ax'][-1].ticklabel_format(axis='x', scilimits=[-3, 20])
        # f['ax'][-1].xaxis.grid(True, which='minor')
        # ax.locator_params(tight=True)
        # ax.autoscale()
        # print("xticks:", ax.get_xticks())


def save_figure(figs):
    if figs is None:
        return
    if isinstance(figs, list):
        for f in figs:
            if not isinstance(f, dict):
                raise TypeError("Expected type(f): dict")
            prettify_plot5(f)
            Path(f['filename']).parent.mkdir(parents=True, exist_ok=True)
            print('Saving figure: {}'.format(f['filename']))
            f['fig'].savefig('{}.pdf'.format(f['filename']), format='pdf',backend='pgf')
            f['fig'].savefig('{}.png'.format(f['filename']), format='png', transparent=False, pad_inches=0)
            # f['fig'].savefig('{}.svg'.format(f['filename']), format='svg')
            # f['fig'].savefig('{}.pgf'.format(f['filename']), format='pgf')

            with open('{}.git'.format(f['filename']), 'w') as gitfile:
                repo = Repo(search_parent_directories=True)
                commit_hash = repo.git.rev_parse("HEAD")
                gitfile.write(commit_hash)

            # tikzplotlib.save('{}.tex'.format(f['filename']), figure=f['fig'])
    else:
        raise TypeError("Unexpected type: ", type(figs))


def plt_scatter(x, y, xlabel, ylabel, ax, markerlist, label='', xscale='linear', yscale='linear', markersize=20):
    ax.scatter(x, y, marker=next(markerlist), label=label, alpha=.90, s=markersize)
    if xlabel != '':  ax.set_xlabel(xlabel)
    if ylabel != '':  ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if label != '': ax.legend()
