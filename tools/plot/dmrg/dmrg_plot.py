import glob
import os.path

from batches import get_batches
from src.database import *
# from meta_fdmrg4 import *
# from meta_fdmrg6 import *
# from meta_xdmrg3 import *
from meta_xdmrg6 import *
from src.plots.multiplot import *


def dmrg_plot(args):
    projdir = args.basedir
    algo_filter = args.algos
    state_filter = args.states
    model_filter = args.models
    point_filter = args.points
    batches = args.batches
    h5avgs = []
    dbs = []
    metas = []
    for batch in batches:
        batchglob = glob.glob(f'{projdir}/{batch}*')
        batchnum = int(''.join(i for i in batch if i.isdigit()))
        version2 = batchnum <= 59
        version3 = batchnum >= 60
        batchglob = [dir for dir in batchglob]
        if batchglob:
            print(f"globbed: {batchglob}")
            batchdir = batchglob[0]
            # avgfile = f'{batchdir}/analysis/data-epstest/averaged-epstest.h5'
            plotdir = f'{batchdir}/analysis/plots'
            cachedir = f'{batchdir}/analysis/cache'
            for avgfile in [
                            f'{batchdir}/analysis/data/averaged.h5',
            ]:
                if not os.path.exists(plotdir):
                    os.makedirs(plotdir)
                print(f'found {avgfile=}')
                h5avgs.append(h5py.File(avgfile, 'r'))
                print(f'loading database for states: {state_filter}')
                for state in state_filter:
                    metas.append(get_meta(plotdir, cachedir))
                    dbs.append(load_isingmajorana_database(h5avgs[-1], metas[-1], algo_filter=algo_filter, model_filter=model_filter,
                                                   state_filter=[state], debug=True))



    # palettes = [  # Palette group for up to 4 categories
        # ["Dark2"]
        # ["gnuplot2"]
        # ["viridis_r"]
        # ["autumn_r", "autumn_r", "winter_r", "spring_r"],
        # ["winter_r", "autumn_r", "winter_r", "spring_r"],
        # ["viridis_r"]
        # ["winter_r",  "summer_r", "autumn_r", "spring_r"]
        # ["Greens", "Oranges", "Reds", "Purples"],
        # ["Greys", "Blues", "Purples", "Purples"],
        # ["Greens", "Greens", "Oranges", "Purples"],
        # ["winter_r", "autumn_r"],
        # ["Blues", "Oranges"],
        # ["Greens", "Reds"],
        # ["Blues", "Oranges", "Purples", "Reds"],
        # ["Blues", "Oranges", "Purples"],
        # ["Reds", "Purples", "Blues" ] # for epstest
        # ["Oranges"]
    # ]


    figspec = ['L']
    subspec = ['d']
    linspec = ['g']
    # xaxspec = ['f']


    # logging.basicConfig(level=logging.DEBUG)

    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_infomasscenter_fig_sub_line(db=db, meta=meta['infomasscenter-dLg'], figs=f)
    # save_figure(f)

    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_infomasscenter_fig_sub_line(db=db, meta=meta['infomasscenter-gLd'], figs=f)
    save_figure(f)

    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_infomasscenter_fig_sub_line(db=db, meta=meta['infomasscenter-d'], figs=f)
    # save_figure(f)
    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_infotopobit_fig_sub_line(db=db, meta=meta['infotopobit-d'], figs=f)
    # save_figure(f)


    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_opdm_fig_sub_line(db=db, meta=meta['opdm-spectrum-g'], figs=f)
    # save_figure(f)

    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_opdm_fig_sub_line(db=db, meta=meta['opdm-spectrum-d'], figs=f)
    # save_figure(f)

    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_opdm_gap_fig_sub_line(db=db, meta=meta['opdm-gapsize-d-L'], figs=f)
    # save_figure(f)

    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_dist_fig_sub_line(db=db, meta=meta['var-dist-g'], figs=f)
    # save_figure(f)

    plt.show()
    exit(0)

    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_infoperscale_fig_sub_line(db=db, meta=meta['infoperscale-Lgd'], figs=f)
    save_figure(f)



    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_infoperscale_fig_sub_line(db=db, meta=meta['infoperscale-gdL'], figs=f)
    save_figure(f)


    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_opdm_gap_fig_sub_line(db=db, meta=meta['opdm-gapsize-d-g'], figs=f)
    save_figure(f)

    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_opdm_gap_fig_sub_line(db=db, meta=meta['opdm-gapsize-g'], figs=f)
    save_figure(f)




    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_opdm_fig_sub_line(db=db, meta=meta['opdm-spectrum-d'], figs=f)
    save_figure(f)
    plt.show()
    exit(0)
    # f = None
    # for idx, (db, meta) in enumerate(zip(dbs, metas)):
    #     f = plot_opdm_gap_fig_sub_line(db=db, meta=meta['opdm-gapsize-d-L'], figs=f)
    # save_figure(f)

    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_dist_fig_sub_line(db=db, meta=meta['var-dist-d'], figs=f)
    save_figure(f)

    f = None
    for idx, (db, meta) in enumerate(zip(dbs, metas)):
        f = plot_dist_fig_sub_line(db=db, meta=meta['var-dist-g'], figs=f)
    save_figure(f)

    # plt.show()
    # exit(0)
    plt.show()
    exit(0)








if __name__ == '__main__':
    # batch = get_batches('fDMRG', ['fdmrg-see-test3'], states=['state_emin'], basedir='/mnt/WDB-AN1500/mbl_transition')
    # batch = get_batches('fDMRG', ['fdmrg6-see'], states=['state_emin'], basedir='/mnt/WDB-AN1500/mbl_transition')
    # batch = get_batches('xDMRG', ['xdmrg3-letsgo'], states=['state_emid'], basedir='/mnt/WDB-AN1500/mbl_transition')
    batch = get_batches('xDMRG', ['xdmrg6-gdplusk'], states=['state_emid'], basedir='/mnt/WDB-AN1500/mbl_transition')
    dmrg_plot(batch)


