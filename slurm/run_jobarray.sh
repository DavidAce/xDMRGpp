#!/bin/bash

#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=logs/%x-%A_%a.txt

PROGNAME=$0
usage() {
  cat << EOF >&2

Usage                               : $PROGNAME [-options] with the following options:
-h                                  : Help. Shows this text.
-d                                  : Dry run
-c <config file>                    : Path to a config file (.cfg)
-e <executable>                     : Path to executable (default = "")
-o <seed offset>                    : Start seed count from this offset
-p <remote prefix>                  : Rclone copy to this remote dir prefix (default "")
-r                                  : Remove the file after rclone
-P                                  : Run seeds in parallel
-F                                  : Forced run of failed/missing seeds
-R                                  : Set --replace instead of --revive
-s                                  : Status file directory
EOF
  exit 1
}
export rclone_remote="neumann:/mnt/WDB-AN1500/mbl_transition"
export rclone_remove="false"
export parallel="false"
export seed_offset=0
export status_dir="status"
export force_run="false"
export replace="false"
while getopts c:hde:Ff:m:o:p:PrRs: o; do
    case $o in
        (h) usage ;;
        (d) export dryrun="ON";;
        (c) printf -v config_path $OPTARG;; # Use printf to expand unicode characters
        (e) export exec=$OPTARG;;
        (o) export seed_offset=$OPTARG;;
        (p) export rclone_prefix=$OPTARG;;
        (r) export rclone_remove="true";;
        (P) export parallel="true";;
        (F) export force_run="true";;
        (R) export replace="true";;
        (s) export status_dir=$OPTARG;;
        (:) echo "Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
        (*) usage ;;
  esac
done
export config_path=$config_path

echondate(){
    printf "%(%Y-%m-%dT%H:%M:%S)T:$*" -1
}

echodate(){
    printf "%(%Y-%m-%dT%H:%M:%S)T:$*\n" -1
}

log(){
  printf -v statusline "%(%Y-%m-%dT%H:%M:%S)T|$1"
  echo "$statusline" >> $2
}

rclone_files_to_remote () {
  if [ -z "$rclone_prefix" ]; then
    return 0
  fi
  case "$1" in
  auto|copy|move)
    ;;
  *)
    echo "Error: invalid rclone operation: [$1]" >&2
    exit 1
    ;;
  esac

  # Generate a file list
  mkdir -p "$tempdir/xDMRG++.$USER/rclone"
  filesfromtxt="$tempdir/xDMRG++.$USER/rclone/filesfrom.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
  transferlist=()
  for file in "${@:2}"; do
    if [ -f "$file" ]; then
      echo "$file" >> "$filesfromtxt"
      transferlist+=("$(basename $file)")
    fi
  done

  if [ ${#transferlist[@]} -eq 0 ]; then
    return 0 # empty array
  fi

  rclone_operation="$1"
  if [[ "$rclone_operation" == "auto" ]]; then
    if [[ "$rclone_remove" == "true" ]]; then
      rclone_operation="move"
    else
      rclone_operation="copy"
    fi
  fi
  echondate "RCLONE LOCAL->REMOTE     : rclone $rclone_operation "
  for file in "${transferlist[@]}"; do
     printf "$file "
  done
  rclone $rclone_operation --files-from="$filesfromtxt" . "$rclone_remote/$rclone_prefix" -L --update --no-traverse --multi-thread-streams 1
  if [ "$?" == "0" ]; then
      printf " -- SUCCESS\n"
      rm -rf "$filesfromtxt"
      return 0
  else
      printf " -- FAILED\n"
      cat $filesfromtxt
      rm -rf "$filesfromtxt"
      return 1
  fi
}

rclone_files_from_remote () {
  if [ -z "$rclone_prefix" ]; then
    return
  fi
  case "$1" in
  auto|copy|move)
    ;;
  *)
    echo "Error: invalid rclone operation: [$1]" >&2
    exit 1
    ;;
  esac

  # Generate a file list
  mkdir -p "$tempdir/xDMRG++.$USER/rclone"
  filesfromtxt="$tempdir/xDMRG++.$USER/rclone/filesfrom.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
  transferlist=()
  for file in "${@:2}"; do
    echo "$file" >> "$filesfromtxt"
    transferlist+=("$(basename $file)")
  done
  rclone_operation="$1"
  if [[ "$rclone_operation" == "auto" ]]; then
    if [[ "$rclone_remove" == "true" ]]; then
      rclone_operation="move"
    else
      rclone_operation="copy"
    fi
  fi

  echondate "RCLONE REMOTE->LOCAL     : rclone $rclone_operation "
  for file in "${transferlist[@]}"; do
     printf "$file "
  done
  rclone $rclone_operation --files-from="$filesfromtxt" "$rclone_remote/$rclone_prefix" . -L --update --no-traverse --multi-thread-streams 1
  if [ "$?" == "0" ]; then
      printf " -- SUCCESS\n"
  else
      printf " -- FAILED\n"
      cat $filesfromtxt
  fi
  rm -rf "$filesfromtxt"
  return 0 # It's fine if this function fails
}

run_sim_id() {
  id=$1 # Shorthand
  array_task_plus_step_id=$1
  model_seed="$(( seed_offset + array_task_plus_step_id ))" # zero-indexed id's

  # The plan is to
  # 1) check if a status file exists in $status_path
  #   -- it should have a line with "$model_seed | <some status>"
  #   -- if it $status == FINISHED, return 0
  # 2) now check if the results exist in the remote: retrieve logs/.../<model_seed>.info
  #   -- This step allows us to run simulations on multiple clusters simultaneously
  #   -- The info file has meta-data lines, one entry per event. The last field in each line is the current "infostatus".
  #   -- If the infostatus == FINISHED, return 0
  # 3) If we make it this far, fetch the .h5 and .txt files from remote if they is newer than the local ones.
  #   -- We only need to do this if either status or infostatus is TIMEOUT|FAILED|MISSING
  # 4) Set the resume policy:
  #   -- Set nothing by default, else replace if -R was given
  #   -- If -F (force_run) was passed: ignore the "FINISHED" status in the loginfo and run anyway (because the $status_path says FAILED/TIMEOUT/MISSING)
  # 5) Prepare to launch the simulation
  #   -- Append a "RUNNING" line to the info file, and send it to remote.
  #   -- Launch the simulation

  config_file="${config_path##*/}"
  config_base="${config_file%.*}"
  config_dir="${config_path%/*}"
  outdir="${output_path%/*}"
  outfile="$outdir/mbl_$model_seed.h5"
  logdir="logs/$config_dir/$config_base"
  logtext="$logdir/$model_seed.txt"
  loginfo="$logdir/$model_seed.info"
  infoline="SLURM_CLUSTER_NAME:$SLURM_CLUSTER_NAME|HOSTNAME:$HOSTNAME|SEED:$model_seed|SLURM_ARRAY_JOB_ID:$SLURM_ARRAY_JOB_ID|SLURM_ARRAY_TASK_ID:$SLURM_ARRAY_TASK_ID|SLURM_ARRAY_TASK_STEP:$SLURM_ARRAY_TASK_STEP"

  # Step 1)
  # Check if there is a status file. Return if this seed has finished
  if [ -f "$status_path" ]; then
    status="$(fgrep -e "$model_seed" "$status_path" | cut -d '|' -f2)" # Should get one of TIMEOUT,FAILED,MISSING,FINISHED
    if [ -z "$status" ]; then
          echodate "STATUS                   : $model_seed $id NULL"
          return 0
    fi
    echodate "STATUS                   : $model_seed $id $status"
    if [ "$status" =~ FINISHED|SKIP  ]; then
      # Copy results back to remote
      # We do this in case there are remnant files on disk that need to be moved.
      # The rclone command has --update, so only newer files get moved.
      rclone_files_to_remote auto "$logtext" "$loginfo" "$outfile"
       # We do not add an RCLONED line anymore.
      return 0
    fi
  fi


  mkdir -p $logdir

  # Step  2)
  # Next, check if the results already exist in the remote
  # If they do, use copy the remote file to local
  # This command will only copy if the remote file is newer.
  rclone_files_from_remote copy "$loginfo"
    if [[ -f $loginfo ]]; then
    echodate "LOGINFO                  : $(tail -n 1 $loginfo)"
    infostatus=$(tail -n 2 $loginfo | awk -F'|' '{print $NF}') # Should be one of RUNNING, FINISHED, RCLONED or FAILED. Add -n 2 to read two lines, in case there is a trailing newline
    echodate "STATUS (loginfo)         : $model_seed $id $infostatus"

    if [[ "$infostatus" =~ FINISHED|SKIP ]] && [[ "$force_run" == "false" ]]; then
      # Copy results back to remote
      # We do this in case there are remnant files on disk that need to be moved.
      # The rclone command has --update, so only newer files get moved.
      rclone_files_to_remote auto "$logtext" "$outfile" "$loginfo"
      return 0
    elif [[ "$infostatus" =~ RUNNING ]] ; then
      # This could be a simulation that terminated abruptly, or it is actually running right now.
      # We can find out because we can check if the slurm job id is still running using sacct
      cluster="$(tail -n 1 $loginfo  | xargs -d '|'  -n1 | grep SLURM_CLUSTER_NAME | awk -F ':' '{print $2}')"
      if [[ "$cluster" == "$SLURM_CLUSTER_NAME" ]];then
        old_array_job_id="$(tail -n 1 $loginfo  | xargs -d '|'  -n1 | grep SLURM_ARRAY_JOB_ID | awk -F ':' '{print $2}')"
        old_array_task_id="$(tail -n 1 $loginfo  | xargs -d '|'  -n1 | grep SLURM_ARRAY_TASK_ID | awk -F ':' '{print $2}')"
        old_job_id=${old_array_job_id}_${old_array_task_id}
        slurm_state=$(sacct -X --jobs $old_job_id --format=state --parsable2 --noheader)
        echodate "STATUS                   : jobid [$old_array_job_id] task [$old_array_task_id] step $id with seed $model_seed  has state [$slurm_state] on this cluster ($cluster)"
        if [ "$slurm_state" == "RUNNING" ] ; then
          echodate "STATUS                   : already running on this cluster, with jobid [$old_job_id], aborting"
          return 0 # Go to next id
        fi
      elif [ ! -z "$cluster" ]; then
        if [[ "$status" =~ TIMEOUT|FAILED ]]; then
          # We should assume that it timed out on the other cluster
          echodate "STATUS                   : $model_seed $id may have timed out on another cluster: $cluster, running job"
        elif [[ "$force_run" == "false" ]]; then
          echodate "STATUS                   : $model_seed $id RUNNING detected on another cluster: $cluster, aborting"
          return 0 # Go to next id because this job is handled by another cluster
        fi
      elif [[ "$force_run" == "false" ]]; then
        echodate "STATUS                   : $model_seed $id RUNNING on unknown cluster, aborting"
        return 0 # Go to next id because this job is handled somehow...
      fi
    fi
    # In all other cases (e.g. FAILED) we go ahead and run the simulation
  fi


  # Step 3)
  # Get the latest data to continue from.
  rclone_files_from_remote copy "$logtext" "$outfile"


  # Step 4)
  # Set the resume policy
  extra_args=""
  if [ "$replace" == "true" ]; then
      extra_args="--replace"
  fi

  # Step 5) Prepare to launch
  if [ -z  "$dryrun" ]; then
    # Add a RUNNING line to loginfo and copy it to remote, to make sure other clusters can see this seed is taken
    log "$infoline|RUNNING" "$loginfo"
    rclone_files_to_remote copy "$loginfo"
    # Add a TIMEOUT line to loginfo if we are force-closed at any time from now on
    trap 'log "$infoline|TIMEOUT" "$loginfo"' SIGINT SIGTERM
    echodate "EXEC LINE                : $exec --config=$config_path --outfile=$outfile --seed=$model_seed --threads=$SLURM_CPUS_PER_TASK $extra_args"
    echodate "LOGFILE                  : $logtext"

    # Run the simulation
    $exec --config=$config_path --outfile=$outfile --seed=$model_seed --threads=$SLURM_CPUS_PER_TASK $extra_args &>> $logtext
    exit_code=$?
    echodate "EXIT CODE                : $exit_code"
    if [ "$exit_code" != "0" ]; then
      log "$infoline|FAILED" "$loginfo"
    else
      log "$infoline|FINISHED" "$loginfo"
      rclone_files_to_remote auto "$logtext" "$loginfo" "$outfile"
      # We do not add an RCLONED line anymore.
    fi
    return $exit_code
  fi
}


if [ ! -f $config_path ]; then
    echodate "config file is not valid: $config_path"
    exit 1
fi

echodate "HOSTNAME                 : $HOSTNAME"
echodate "USER                     : $USER"
echodate "CONFIG PATH              : $config_path"
echodate "SLURM_CLUSTER_NAME       : $SLURM_CLUSTER_NAME"
echodate "SLURM_NTASKS             : $SLURM_NTASKS"
echodate "SLURM_CPUS_ON_NODE       : $SLURM_CPUS_ON_NODE"
echodate "SLURM_JOB_CPUS_PER_NODE  : $SLURM_JOB_CPUS_PER_NODE"
echodate "SLURM_CPUS_PER_TASK      : $SLURM_CPUS_PER_TASK" # Task is the same as simulation
echodate "SLURM_MEM_PER_CPU        : $SLURM_MEM_PER_CPU"
echodate "SLURM_MEM_PER_NODE       : $SLURM_MEM_PER_NODE"
echodate "SLURM_JOBID              : $SLURM_JOBID"
echodate "SLURM_ARRAY_JOB_ID       : $SLURM_ARRAY_JOB_ID"
echodate "SLURM_ARRAY_TASK_ID      : $SLURM_ARRAY_TASK_ID"
echodate "SLURM_ARRAY_TASK_STEP    : $SLURM_ARRAY_TASK_STEP"
echodate "SLURM_ARRAY_TASK_MIN     : $SLURM_ARRAY_TASK_MIN"
echodate "SLURM_ARRAY_TASK_MAX     : $SLURM_ARRAY_TASK_MAX"
echodate "OMP_NUM_THREADS          : $OMP_NUM_THREADS"

if [ -z "$SLURM_CLUSTER_NAME" ]; then
  echodate "This is not a valid slurm job environment. Use this script with sbatch or srun"
  exit 1
fi

ssh-agent -k
eval "$(ssh-agent -s)"

# Zero-indexed  id's
export start_id=$SLURM_ARRAY_TASK_ID
export end_id=$(( SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_STEP - 1))
exit_code_save=0
ulimit -c unlimited


# Determine the output path from the config file
output_path="$(fgrep -e "storage::output_filepath" "$config_path" | awk '{sub(/.*=/,""); sub(/ \/!*<.*/,""); print $1;}')"
export output_path="$output_path"

# Find and copy the status file to tmp
config_base="$(basename -s ".cfg" "$config_path")"
export status_path="$status_dir/$config_base.status"
if [ -f "$status_path" ]; then
    tempdir="/tmp"
    if [ -d "/scratch/local" ];then
      tempdir="/scratch/local"
    elif [ -n "$PDC_TMP" ]; then
       tempdir="$PDC_TMP"
    fi
    status_temp="$tempdir/xDMRG++.$USER/status/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    status_name="$config_base.status"
    if [ ! -f "$status_temp/$status_name" ]; then
      mkdir -p $status_temp
      cp $status_path $status_temp/
    fi
    export status_path=$status_temp/$status_name
    trap 'rm -rf "$status_temp"' EXIT
fi

echodate "TASK ID SEQUENCE         : $(seq -s ' ' $start_id $end_id)"
if [ "$parallel" == "true" ]; then
  # Load GNU Parallel from modules
  if [[ "$SLURM_CLUSTER_NAME" =~ tetralith ]] ; then
    module load parallel/20181122-nsc1
  else
    module load parallel
  fi
  if [ "$?" != "0" ] ; then
    echo "Failed to module load parallel"
    exit 1
  fi

  export -f echodate
  export -f log
  export -f run_sim_id
  export -f rclone_files_to_remote
  export -f rclone_files_from_remote
  joblog="logs/parallel-$(( seed_offset + start_id ))_$(( seed_offset + end_id )).log" # zero-indexed id's
  echodate "parallel --memfree=$SLURM_MEM_PER_CPU --jobs=$SLURM_NTASKS --ungroup --delay=.2s --joblog=$joblog --colsep=' ' run_sim_id ::: seq $start_id $end_id"
  parallel --memfree=$SLURM_MEM_PER_CPU \
           --jobs=$SLURM_NTASKS \
           --ungroup --delay=.2s --resume \
           --joblog=$joblog \
           --colsep=' ' run_sim_id \
           ::: $(seq $start_id $end_id)
  exit_code_save=$?
else
    for id in $(seq $start_id $end_id); do
      run_sim_id $id
      exit_code=$?
      if [ "$exit_code" != "0" ]; then
        exit_code_save=$exit_code
      fi
    done
fi

exit $exit_code_save


