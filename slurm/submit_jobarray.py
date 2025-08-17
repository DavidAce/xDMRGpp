import argparse
import os
import errno
import subprocess
from pathlib import Path
from itertools import zip_longest
from itertools import chain, islice
from datetime import datetime
import socket
import json
from random import shuffle
import numpy as np
import linecache

def chunks(iterable, n):
   "chunks(ABCDE,2) => AB CD E"
   iterable = iter(iterable)
   while True:
       # store one line in memory,
       # chain it to an iterator on the rest of the chunk
       try:
           yield chain([next(iterable)], islice(iterable, n-1))
       except StopIteration:
           return

def get_line_number(fname, text):
    with open(fname) as f:
        for i, line in enumerate(f):
            if str(text) in line:
                return i

def get_lines(fname, off, ext):
    line_off = get_line_number(fname, off)
    lines = []
    for idx in range(line_off, line_off + ext):
        lines.append(linecache.getline(fname, idx + 1).rstrip())
    return lines


def parse(project_name):
    parser = argparse.ArgumentParser(description='SLURM batch submission for {}'.format(project_name))
    parser.add_argument('-b', '--build-type', type=str, help='Build type', default='Release')
    parser.add_argument('-M', '--clusters', type=str, help='Comma separated list of Slurm clusters', default=None)
    parser.add_argument('-w', '--nodelist', type=str, help='Comma separated list of node names', default=None)
    parser.add_argument('--account', type=str, help='Name of account to run under', default=None)
    parser.add_argument('--reservation', type=str, help='Name of reservation to run under', default=None)
    parser.add_argument('--config', type=str, help='Path to simulation config files (suffixed .cfg)', default='config')
    parser.add_argument('--status', type=str, help='Path to simulation status files (suffixed .status)', default='status')
    parser.add_argument('--seedpath', type=str, help='Path to simulation seed files (suffixed .json). Default to --config.', default=None)
    parser.add_argument('--pattern', type=str, help='Only consider simulation config files containing this substring', default=None)
    parser.add_argument('--omp-num-threads', type=int, help='Number of openmp threads', default=None)
    parser.add_argument('--omp-dynamic', action='store_true', help='Sets OMP_DYNAMIC=true|false.', default=None)
    parser.add_argument('--omp-max-active-levels', type=int, help='Sets OMP_MAX_ACTIVE_LEVELS=n', default=None)
    parser.add_argument('--omp-places', type=str, help='Sets OMP_PLACES', choices=['threads', 'cores', 'sockets'],default=None )
    parser.add_argument('--omp-proc-bind', type=str, help='Sets OMP_PROC_BIND', choices=['true', 'false', 'close', 'spread','master'],default=None)
    parser.add_argument('--cpus-per-task', type=int, help='Number of cores per task', default=1)
    parser.add_argument('--nodes', type=int, help='Number of nodes to allocate for each job', default=1)
    parser.add_argument('--ntasks', type=int, help='Number of tasks per job (e.g. mpi threads or gnu parallel threads)', default=1)
    parser.add_argument('--ntasks-per-core', type=int, help='Number of tasks (sims) on each core', default=1)
    parser.add_argument('--ntasks-per-node', type=int, help='Number of tasks per node', default=None)
    parser.add_argument('--threads-per-core', type=int, help='Number of threads to allocate on each core (1 disables hyperthreading)', default=1)
    parser.add_argument('--openblas-coretype', type=str, help='Sets OPENBLAS_CORETYPE', default=None)
    parser.add_argument('--dryrun', action='store_true', help='Dry run')
    parser.add_argument('--debug', action='store_true', help='Debug this script')
    parser.add_argument('--exclusive', action='store_true', help='Reserve whole node')
    parser.add_argument('--execname', type=str, help='Name of executable', default='xDMRG++')
    parser.add_argument('--hint', type=str, default=None, help='Slurm parallelization hint. Cannot be used with --ntasks-per-core', choices=['multithread', 'nomultithread', 'compute_bound', 'memory_bound'])
    # parser.add_argument('-j', '--job-dir', type=str, help='Directory with existing .job files. Use for resuming failed runs')
    parser.add_argument('-J', '--job-name', type=str, help='Slurm job name', default='DMRG')
    parser.add_argument('-m', '--mem-per-cpu', type=str, help='Memory per core, e.g 2000, 2000M or 2G', default='1G')
    #parser.add_argument('-N', '--sims-per-cfg', type=int, help='Number of simulations per config file. Can be split up into chunks with -n', default=10)
    parser.add_argument('-n', '--sims-per-array', type=int, help='Number of simulations in each job-array', default=1000)
    parser.add_argument('--sims-per-task', type=int, help='Number of simulations per job-array task. This is equivalent to the step, or stride in the array', default=10)
    parser.add_argument('-o', '--other', type=str, help='Other options for sbatch (verbatim)', default=None)
    parser.add_argument('--open-mode', type=str, help='Access mode for logs', default='append', choices=['append','truncate'])
    parser.add_argument('-p','--partition', type=str, help='Partition name', default=None)
    parser.add_argument('-q','--qos', type=str, help='Quality of service', default=None)
    parser.add_argument('--requeue', action='store_true', help='Requeue job in case of failure')
    # parser.add_argument('--start-seed', type=int, help='Starting seed for random number generator', default=0)
    # parser.add_argument('--shuffle', action='store_true', help='Shuffle all seeeds and config files')
    parser.add_argument('--parallel', action='store_true', help='Use GNU parallel to run the job array step in parallel')
    parser.add_argument('-t', '--time', type=str, help='Time limit for each job', default='0-01:00:00' )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose sbatch')
    parser.add_argument('--default-kraken', action='store_true', help='Set defaults for kraken cluster')
    parser.add_argument('--default-tetralith', action='store_true', help='Set defaults for tetralith cluster')
    parser.add_argument('--rclone-prefix', type=str, help='Use rclone to copy results to this remote directory', default=None)
    parser.add_argument('--rclone-remove', action='store_true', help='Remove local file after rclone', default=None)
    parser.add_argument('--minseed', type=int, help='Minimum seed value to consider',default=None)
    parser.add_argument('--maxseed', type=int, help='Maximum seed value to consider',default=None)
    parser.add_argument('--force-run', action='store_true', help='Force run of seeds with status failed|timeout|missing')
    parser.add_argument('--replace', action='store_true', help='Set --replace')
    parser.add_argument('--ignore-seed-order', action='store_true', help='Do not require that the seeds in the .status file be ordered')

    args = parser.parse_args()
    if args.seedpath is None:
        args.seedpath = args.config

    if args.default_kraken:
        parser.set_defaults(partition='dedicated',qos='lowprio')
        args = parser.parse_args()
    if args.default_tetralith:
        parser.set_defaults(partition='tetralith')
        args = parser.parse_args()

    return args



def run(cmd,env,args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT):
    if len(cmd) == 0:
        return
    if isinstance(cmd,str):
        cmd = cmd.split()
    if args.debug or args.dryrun:
        print("Run command:", ' '.join(cmd))

    if args.dryrun:
        return

    with subprocess.Popen(cmd,bufsize=1, shell=shell, stdout=stdout, stderr=stderr, encoding='utf-8',env=env) as p:
        if p.stdout:
            for line in p.stdout:
                print(line, end='')  # process line here
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, ' '.join(p.args))

def split_range(extent, offset, chunksize):
    extents = [min(extent,chunksize)]
    offsets = [offset]
    while True:
        sumextent = np.sum(extents)    # Sum
        remextent = extent - sumextent # Remaining
        if remextent == 0:
            break
        else:
            offsets.append(offsets[-1] + extents[-1])
            extents.append(min(chunksize, remextent))

    return extents, offsets

def generate_sbatch_commands(project_name, args):
    sbatch_cmd = []
    sbatch_arg = []
    sbatch_env = os.environ.copy()  # Add environment variables here

    if args.omp_num_threads:
        sbatch_env['OMP_NUM_THREADS'] = str(args.omp_num_threads)
    if args.omp_dynamic:
        sbatch_env['OMP_DYNAMIC'] = str(args.omp_dynamic)
    if args.omp_max_active_levels:
        sbatch_env['OMP_MAX_ACTIVE_LEVELS'] = str(args.omp_max_active_levels)
    if args.omp_places:
        sbatch_env['OMP_PLACES'] = str(args.omp_places)
    if args.omp_proc_bind:
        sbatch_env['OMP_PROC_BIND'] = str(args.omp_proc_bind)
    if args.openblas_coretype:
        sbatch_env['OPENBLAS_CORETYPE'] = str(args.openblas_coretype)

    # Find executable
    if args.build_type == 'None':
        exec = args.execname
    else:
        exec = '../build/{}/{}'.format(args.build_type, args.execname)
    if(os.access(exec, os.X_OK)):
        print('Found executable:', exec)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), exec)


    # Make sure the executable would find its dynamically loaded libraries
    ldd_out = subprocess.check_output('ldd {}'.format(exec), shell=True,text=True)
    print(ldd_out)
    if 'not found' in ldd_out:
      raise FileNotFoundError(errno.ENOENT, "Some dynamic libraries were not found. Perhaps a module needs to be loaded.", exec)

    if args.clusters:
        sbatch_arg.extend([f'--clusters={args.clusters}'])
    if args.nodelist:
        sbatch_arg.extend([f'--nodelist={args.nodelist}'])
    if args.account:
       sbatch_arg.extend([f'--account={args.account}'])
    if args.reservation:
       sbatch_arg.extend([f'--reservation={args.reservation}'])
    if args.cpus_per_task:
        sbatch_arg.extend(['--cpus-per-task={}'.format(args.cpus_per_task)])
    if args.nodes:
        sbatch_arg.extend(['--nodes={}'.format(args.nodes)])
    if args.ntasks_per_core:
        sbatch_arg.extend(['--ntasks-per-core={}'.format(args.ntasks_per_core)])
    if args.ntasks_per_node:
        sbatch_arg.extend(['--ntasks-per-node={}'.format(args.ntasks_per_node)])
    if args.job_name:
        sbatch_arg.extend(['--job-name={}'.format(args.job_name)])
    if args.mem_per_cpu:
        sbatch_arg.extend(['--mem-per-cpu={}'.format(args.mem_per_cpu)])
    if args.ntasks:
        sbatch_arg.extend(['--ntasks={}'.format(args.ntasks)])
    if args.open_mode:
        sbatch_arg.extend(['--open-mode={}'.format(args.open_mode)])
    if args.partition:
        sbatch_arg.extend(['--partition={}'.format(args.partition)])
    if args.qos:
        sbatch_arg.extend(['--qos={}'.format(args.qos)])
    if args.requeue:
        sbatch_arg.extend(['--requeue'])
    if args.exclusive:
        sbatch_arg.extend(['--exclusive'])
    if args.hint:
        sbatch_arg.extend(['--hint={}'.format(args.hint)])
    if args.time:
        sbatch_arg.extend(['--time={}'.format(args.time)])
    if args.verbose:
        sbatch_arg.extend(['-v'])
    rclone_prefix = f' -p {args.rclone_prefix}' if args.rclone_prefix else ''
    rclone_remove = ' -r' if args.rclone_remove else ''
    force_run = ' -F' if args.force_run else ''
    replace = ' -R' if args.replace else ''
    parallel = ' -P' if args.parallel else ''

    # Load the seed configurations
    if args.pattern:
        cfgs = sorted(list(Path(args.config).glob('*{}*.cfg'.format(args.pattern))))
    else:
        cfgs = sorted(list(Path(args.config).glob('*.cfg')))
    if not cfgs:
        raise FileNotFoundError(errno.ENOENT, f'{os.strerror(errno.ENOENT)}: no .cfg files found in {args.cfgspath}')

    for cfg in cfgs:
        seedfile = '{}/{}.json'.format(Path(args.seedpath), Path(cfg).stem)
        statfile = '{}/{}.status'.format(Path(args.status), Path(cfg).stem)
        donekeys = ['FINISHED', "SKIP"]
        if not args.force_run:
            donekeys.append('FAILED')
        with open(seedfile, 'r') as fp:
            seedjson = json.load(fp)
            for extent, offset, status in zip(seedjson['seed_extent'],seedjson['seed_offset'], seedjson['seed_status']):
                if status == "FINISHED":
                    continue
                extents, offsets = split_range(extent,offset,args.sims_per_array)
                for ext,off in zip(extents,offsets):
                    step = min(ext, args.sims_per_task)
                    # We can now check in the statusfile if this sub-portion has actually finished
                    all_done = all([l.split('|')[1] in donekeys for l in get_lines(statfile, off, ext) ])
                    if all_done:
                        continue
                    off_final = off
                    ext_final = ext

                    if args.minseed:
                        if off+ext <= args.minseed:
                            continue
                        if off < args.minseed and off+ext >= args.minseed:
                            off_final = args.minseed
                            ext_final = ext - (args.minseed - off)
                    if args.maxseed:
                        if off_final >= args.maxseed:
                            continue
                        if off_final < args.maxseed and off_final+ext_final >= args.maxseed:
                            ext_final = args.maxseed-off_final

                    sbatch_cmd.append('sbatch {} --array=0-{}:{} run_jobarray.sh -e {} -c {} -s {} -o {}{}{}{}{}{}'
                                      .format(' '.join(sbatch_arg), ext_final-1, step, exec, cfg, args.status, off_final,
                                              parallel, rclone_prefix, rclone_remove, force_run, replace))
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("jobs").mkdir(parents=True, exist_ok=True)

    # Save the sbatch command
    with open("jobs/sbatch-{}-{}.txt".format(socket.gethostname(), datetime.now().strftime("%Y-%m-%dT%H.%M.%S")),
              mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(sbatch_cmd))

    if args.debug:
        print('sbatch args:', ' '.join(sbatch_arg))
        print('environment:', ' '.join(sbatch_env))
    return sbatch_cmd, sbatch_env




def main():
    project_name = 'DMRG'
    args = parse(project_name=project_name)
    sbatch_cmd, sbatch_env = generate_sbatch_commands(project_name=project_name,args=args)

    if not args.dryrun:
        with open('job-report-{}.txt'.format(socket.gethostname()),'a') as f:
            f.write('\n------------------------------------\n')
            f.write('Running sbatch - {}\n'.format(datetime.now().strftime("%Y-%m-%dT%H.%M.%S")))
            f.flush()
            run('git log -1', sbatch_env, args, shell=False, stdout=f, stderr=f)

    for cmd in sbatch_cmd:
        run(cmd=cmd, env=sbatch_env, args=args)

if __name__ == "__main__":
    main()
