#!/usr/bin/env bash

#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=logs/%x-%A_%a.txt

PROGNAME=$0
echodate(){
    printf "%(%Y-%m-%dT%H:%M:%S)T:$*\n" -1
}
usage() {
  cat << EOF >&2

Usage              : $PROGNAME -<option1>:arg1 -<option2>:arg2 ...
-h                 : Help. Shows this text.
-d                 : Performs a dry run.
-f                 : rclone --files-from=
-o <rclone op>     : Choose rclone operation (move|copy)
-p <target prefix> : Prefix at destination
-r <address>       : Name of remote
-s <source dir>    : Source relative to current dir
-t <target dir>    : Target directory
-u <user>          : User at target machine (default = david)
-L                 : Follow symbolic links
EOF
  exit 1
}

while getopts "df:ho:p:r:s:t:u:L" o; do
      case $o in
        (d) dry_run="--dry-run";;
        (f) filesfrom=$OPTARG;;
        (h) usage ;;
        (o) operation=$OPTARG;;
        (p) prefix=$OPTARG;;
        (r) remote=$OPTARG;;
        (s) source=$OPTARG;;
        (t) target=$OPTARG;;
        (u) user=$OPTARG;;
        (L) follow_links="-L";;
        (:) echo "Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
        (?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
        (*) usage ;;
      esac
done
if [ -z "$remote" ]; then
        echo 'Missing -r:remote' >&2
        exit 1
fi
if [ -z "$prefix" ]; then
        echo 'Missing -p:prefix' >&2
        exit 1
fi
if [ -z "$source" ]; then
        echo 'Missing -s:source' >&2
        exit 1
fi
if [ -z "$target" ]; then
        echo 'Missing -t:target' >&2
        exit 1
fi
if [ -z "$filesfrom" ]; then
        echo 'Missing -f:filesfrom' >&2
        exit 1
fi
if [ -z "$operation" ]; then
        echo 'Missing -o:operation' >&2
        exit 1
fi


ssh-agent -k
eval "$(ssh-agent -s)"

rclone "$operation" --files-from="$filesfrom" "$source" "$remote:${prefix}/${target}" $dry_run $follow_links --update --multi-thread-streams=1 --transfers=1 --no-traverse
