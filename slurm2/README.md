# slurm2

Python-first Slurm workflow for the `xdmrg6-gdplusk*` sweeps.

## Terms

- `experiment`: one experiment directory, usually `/mnt/WDB-AN1500/mbl_transition/xdmrg6-gdplusk`
- `point`: one exact parameter point such as `L20_g0.500_d+2.50`
- `chunk`: one subset of seeds handled by one Slurm array task

## Layout

Inside an experiment:

- `experiment.json`: top-level metadata
- `configs/`: generated `.cfg` files
- `points/`: one JSON file per point
- `status/`: scanned per-point status JSON
- `events/`: per-seed event logs written by workers
- `logs/`: per-seed text logs and Slurm logs
- `output/`: local HDF5 staging area
- `submissions/`: saved `sbatch` records

## Typical flow

On `neumann`:

```bash
python -m slurm2 generate --name xdmrg6-gdplusk

python -m slurm2 scan \
  --name xdmrg6-gdplusk \
  --pattern L20 \
  --jobs 16
```

On the cluster:

```bash
python slurm2/submit.py \
  --name xdmrg6-gdplusk \
  --rclone-prefix xdmrg6-gdplusk \
  --pattern L20 \
  --job-name xdmrg6-L20 \
  --clusters=draken \
  --mem-per-cpu 3000M \
  --time 4-00:00:00 \
  --sims-per-array=1000 \
  --cpus-per-task=4 \
  --threads-per-core=1 \
  --ntasks-per-core=1 \
  --omp-num-threads=4 \
  --ntasks=1 \
  --requeue \
  --sims-per-task=10 \
  --build-type=Release \
  --partition=dedicated \
  --qos=lowprio \
  --force-run
```

For GPU runs, request a GPU from Slurm and separately choose the `xDMRG++` GPU policy:

```bash
python slurm2/submit.py \
  --name xdmrg6-gdplusk \
  --rclone-prefix xdmrg6-gdplusk \
  --pattern L20 \
  --gpus-per-task=1 \
  --gpu-policy=TRY \
  --gpu-id=auto
```

The Slurm allocation flags `--gpus`, `--gpus-per-node`, `--gpus-per-task`, and `--gres` are passed to `sbatch`. The runtime flags `--gpu-policy`, `--gpu-id`, `--gpu-switchsize`, and `--gpu-max-alloc-fraction` are forwarded to `xDMRG++` by each worker. Leave the runtime flags unset to use the values already written in the generated cfg files. Some clusters allocate GPUs through generic resources; on those systems, replace `--gpus-per-task=1` with `--gres=gpu:1`.

`--name` is the one canonical experiment identifier. It selects the TOML file `slurm2/specs/<name>.toml`, the experiment metadata name, and the default subdirectory name under the base directory. When the experiment lives under the default base directory `/mnt/WDB-AN1500/mbl_transition`, `--name xdmrg6-gdplusk` is enough. Use `--base-dir /some/other/root` to override that base, or `--experiment-dir /full/path/xdmrg6-gdplusk` when you want to bypass name-based resolution entirely.

`submit` has a preflight sync stage before it plans jobs. It tries to pull `experiment.json`, `configs/`, `points/`, `status/`, and `events/` from `neumann`, and it also tries to push any already-produced local `output/`, `logs/`, and `events/` back to the remote experiment before new submissions.

Each worker then does a second, more fine-grained sync stage before every seed. Right before deciding whether to run seed `N`, the worker refreshes that seed's remote `events/`, `logs/`, and `output/` files. This preserves the legacy behavior where simultaneous clusters can see each other's latest per-seed state and avoid colliding on work.

## Experiment Spec Format

Experiment specs live in [slurm2/specs](/mnt/S990PRO/GitProjects/xDMRG++/slurm2/specs). They use:

- exact point keys such as `"d+6.00|g0.500"`
- full `settings.*` namespaces for values that map into `settings.h`
- a `token_map` that maps key fragments such as `d`, `g`, and `L` onto `settings.*` paths

For example:

```toml
[experiment.token_map]
L = "settings.model.model_size"
d = "settings.model.ising_majorana.delta"
g = "settings.model.ising_majorana.g"

[defaults]
settings.storage.resume_policy = "IF_UNSUCCESSFUL"
settings.storage.file_collision_policy = "REVIVE"

[groups.L20]
settings.xdmrg.iter_max = 200

[groups.L20.batch]
"d+5.00|g0.500" = { seed_extent = [300], seed_offset = [2_500_0500_000000] }
```

Templates can also use Python format specs, for example `point_id_template = "L{L}_g{g:.3f}_d{d:+.2f}"`.

The raw token text is preserved exactly when no format spec is given, so the same decimals appear again in point ids, output paths, and generated configs. Format specs are only applied when you request them explicitly in the template.

## Notes

- Without `--rclone-remove`, workers keep local `.h5` and `.txt` files after upload.
- With `--rclone-remove`, workers use safe copy-then-remove semantics: they first upload with `rclone copyto`, and only if that succeeds do they delete the local `.h5` or `.txt`. Event logs are never removed automatically. This is the flag to use when you want the old cluster-disk-saving behavior.
- If `neumann` is offline or a transfer fails, local results stay on the cluster. A later `submit`, a later worker run for the same seed, or a manual `rclone` can push them back.
- `scan` is the HDF5 ground truth. `submit` also consults local `events/` so already-finished seeds can be skipped before the next scan.
- `MISSING` and `TIMEOUT` seeds are already considered runnable. `--force-run` additionally makes `FAILED` seeds runnable again and ignores remote `RUNNING` locks for the explicitly scheduled seeds. It does not rerun seeds already marked `FINISHED` or `SKIP` in the scanned status.
- `L22` and `L24` only generate the subset of deltas that are active in the legacy batch scripts.
- For detailed flag behavior, read `python -m slurm2 generate --help`, `python -m slurm2 scan --help`, `python -m slurm2 submit --help`, and `python -m slurm2 worker --help`.
