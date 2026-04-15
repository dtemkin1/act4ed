#!/usr/bin/env bash
set -euo pipefail

LOGIN_ALIAS="${ORCD_LOGIN_ALIAS:-orcd-login}"
PROJECT_DIR="${ORCD_PROJECT_DIR:-}"
SBATCH_SCRIPT="${ORCD_SBATCH_SCRIPT:-run_jupyter_server.sh}"
PROFILE_FILE="${ORCD_JLAB_PROFILE:-}"
JOB_NAME="${ORCD_JLAB_JOBNAME:-jlab}"
WAIT_READY="${ORCD_WAIT_READY:-900}"
REMOTE_STATE_DIR="${ORCD_REMOTE_STATE_DIR:-}"

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/orcd-jlab"
mkdir -p "$CACHE_DIR"

MODE=""
PYTHON_VERSION=""
CONTAINER=""
USE_NV=0
ACTION="submit"
JOBID=""

SBATCH_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  orcd-jlab.sh --project REMOTE_DIR [options]
  orcd-jlab.sh --project REMOTE_DIR --reconnect [JOBID]
  orcd-jlab.sh --project REMOTE_DIR --stop [JOBID]
  orcd-jlab.sh --project REMOTE_DIR --cancel [JOBID]
  orcd-jlab.sh [JOBID]

Core options:
  --project DIR          Remote project directory on ORCD.
  --script PATH          Remote batch script path. Default: run_jupyter_server.sh
  --profile PATH         Remote profile file to source before launch.
  --container PATH       Remote .sif container image.
  --mode MODE            uv | venv | custom
  --python VERSION       Python version for 'uv venv --python'.
  --login-alias HOST     SSH host alias from ~/.ssh/config. Default: orcd-login

Slurm overrides:
  --partition NAME
  --gres SPEC            Example: gpu:1
  --cpus N
  --mem SIZE             Example: 32G
  --time HH:MM:SS
  --account NAME
  --constraint EXPR
  --qos NAME
  --job-name NAME
  --use-nv               Force Apptainer --nv inside the batch job.

Actions:
  --reconnect [JOBID]    Recreate the tunnel for an existing job.
  --stop [JOBID]         Stop the SSH tunnel for a job.
  --cancel [JOBID]       Cancel the Slurm job and stop the tunnel.
  --list                 List active jobs with this job name.
  --help                 Show this help.

Examples:
  orcd-jlab.sh --project /path/to/my-project \
    --container /path/to/my-jupyter.sif \
    --partition mit_normal_gpu --gres gpu:1 --cpus 8 --mem 32G --time 06:00:00

  orcd-jlab.sh --project /path/to/my-project --reconnect
  orcd-jlab.sh --stop 1234567
  orcd-jlab.sh --cancel 1234567
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

need_value() {
  local opt="$1"
  local remaining="$2"
  (( remaining >= 2 )) || die "Missing value for ${opt}"
}

q() {
  printf '%q' "$1"
}

remote() {
  ssh "$LOGIN_ALIAS" "$@"
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    return 1
  fi
}

port_is_free() {
  local port="$1"
  local py

  py="$(pick_python || true)"
  [[ -n "$py" ]] || return 0

  "$py" - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

pick_free_port() {
  local py
  py="$(pick_python || true)"
  [[ -n "$py" ]] || return 1

  "$py" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

remote_meta_expr() {
  local jobid="$1"
  if [[ -n "$REMOTE_STATE_DIR" ]]; then
    printf '%q' "${REMOTE_STATE_DIR%/}/jlab-${jobid}.env"
  else
    printf '%s' '"$HOME/.local/state/orcd-jlab/jlab-'
    printf '%s' "$jobid"
    printf '%s' '.env"'
  fi
}

job_exists() {
  local jobid="$1"
  local out
  out="$(remote "squeue -j $(q "$jobid") -h -o %i" 2>/dev/null || true)"
  [[ -n "$out" ]]
}

submit_job() {
  [[ -n "$PROJECT_DIR" ]] || die "--project is required when submitting a new job"

  local export_items=("ALL" "JLAB_PROJECT_DIR=$PROJECT_DIR")
  [[ -n "$PROFILE_FILE" ]] && export_items+=("JLAB_PROFILE_FILE=$PROFILE_FILE")
  [[ -n "$CONTAINER" ]] && export_items+=("JLAB_CONTAINER=$CONTAINER")
  [[ -n "$MODE" ]] && export_items+=("JLAB_MODE=$MODE")
  [[ -n "$PYTHON_VERSION" ]] && export_items+=("JLAB_PYTHON=$PYTHON_VERSION")
  [[ "$USE_NV" == "1" ]] && export_items+=("JLAB_USE_NV=1")
  [[ -n "$REMOTE_STATE_DIR" ]] && export_items+=("JLAB_STATE_DIR=$REMOTE_STATE_DIR")

  local export_string
  export_string="$(IFS=,; echo "${export_items[*]}")"

  local remote_script="$SBATCH_SCRIPT"
  if [[ "$remote_script" != /* ]]; then
    remote_script="./$remote_script"
  fi

  local cmd="cd $(q "$PROJECT_DIR") && sbatch"
  local arg
  for arg in "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}"; do
    cmd+=" $(q "$arg")"
  done
  cmd+=" --export=$(q "$export_string") $(q "$remote_script")"

  remote "$cmd" | awk '{print $NF}'
}

project_jobids() {
  [[ -n "$PROJECT_DIR" ]] || return 0

  local cmd
  cmd="cd $(q "$PROJECT_DIR") >/dev/null 2>&1 || exit 1; \
for id in \
\$(squeue -u \$USER -h -n $(q "$JOB_NAME") -t R,PD -o %i | sort -n); do \
  if [[ -f logs/${JOB_NAME}-\$id.out || -f logs/${JOB_NAME}-\$id.err ]]; then \
    echo \$id; \
  fi; \
done"

  remote "$cmd" 2>/dev/null || true
}

latest_active_jobid() {
  local ids=""

  if [[ -n "$PROJECT_DIR" ]]; then
    ids="$(project_jobids)"
    if [[ -n "$ids" ]]; then
      awk 'NF{print}' <<<"$ids" | sort -n | tail -n 1
      return 0
    fi
  fi

  remote "squeue -u \$USER -h -n $(q "$JOB_NAME") -t R,PD -o %i | sort -n | tail -n 1" 2>/dev/null || true
}

fetch_job_diagnostics() {
  local jobid="$1"
  local cmd
  cmd="job_line=\$(scontrol show job -o $(q "$jobid") 2>/dev/null || true); \
if [[ -z \$job_line ]]; then exit 0; fi; \
out=\$(tr ' ' '\n' <<<\"\$job_line\" | sed -n 's/^StdOut=//p'); \
err=\$(tr ' ' '\n' <<<\"\$job_line\" | sed -n 's/^StdErr=//p'); \
for f in \"\$out\" \"\$err\"; do \
  if [[ -n \$f && -r \$f ]]; then \
    echo \"===== \$f =====\"; \
    tail -n 80 \"\$f\"; \
  fi; \
done"

  remote "$cmd" 2>/dev/null || true
}

wait_for_meta() {
  local jobid="$1"
  local meta_expr
  meta_expr="$(remote_meta_expr "$jobid")"

  local i meta_text=""
  for ((i = 1; i <= WAIT_READY; i++)); do
    meta_text="$(remote "if [[ -r $meta_expr ]]; then cat $meta_expr; fi" 2>/dev/null || true)"
    if [[ -n "$meta_text" ]]; then
      printf '%s\n' "$meta_text"
      return 0
    fi

    if (( i % 5 == 0 )); then
      if ! job_exists "$jobid"; then
        echo "Job $jobid exited before it became ready." >&2
        fetch_job_diagnostics "$jobid" >&2 || true
        return 1
      fi
    fi

    sleep 1
  done

  echo "Timed out waiting for job $jobid to become ready." >&2
  fetch_job_diagnostics "$jobid" >&2 || true
  return 1
}

parse_meta() {
  local meta_text="$1"
  META_NODE=""
  META_PORT=""
  META_TOKEN=""
  META_PROJECT_DIR=""

  while IFS='=' read -r key value; do
    case "$key" in
      JLAB_NODE) META_NODE="$value" ;;
      JLAB_PORT) META_PORT="$value" ;;
      JLAB_TOKEN) META_TOKEN="$value" ;;
      JLAB_PROJECT_DIR) META_PROJECT_DIR="$value" ;;
    esac
  done <<< "$meta_text"

  [[ -n "$META_NODE" && -n "$META_PORT" && -n "$META_TOKEN" ]] || die "Could not parse node/port/token from metadata"
}

start_or_restart_tunnel() {
  local jobid="$1"
  local node="$2"
  local remote_port="$3"
  local force_restart="$4"

  local sock="$CACHE_DIR/j-${jobid}.sock"
  local port_file="$CACHE_DIR/${jobid}.local_port"
  local local_port=""

  if [[ "$force_restart" == "1" ]]; then
    ssh -S "$sock" -O exit "$LOGIN_ALIAS" >/dev/null 2>&1 || true
  fi

  if ssh -S "$sock" -O check "$LOGIN_ALIAS" >/dev/null 2>&1; then
    if [[ -f "$port_file" ]]; then
      cat "$port_file"
      return 0
    fi
    ssh -S "$sock" -O exit "$LOGIN_ALIAS" >/dev/null 2>&1 || true
  fi

  if [[ -f "$port_file" ]]; then
    local_port="$(cat "$port_file" 2>/dev/null || true)"
  fi

  if [[ -z "$local_port" ]]; then
    local_port="$remote_port"
  fi

  if ! port_is_free "$local_port"; then
    local_port="$(pick_free_port)" || die "Could not find a free local port"
  fi

  ssh -f -N \
    -o ExitOnForwardFailure=yes \
    -o ControlMaster=yes \
    -o ControlPersist=yes \
    -o ControlPath="$sock" \
    -L "${local_port}:${node}:${remote_port}" \
    "$LOGIN_ALIAS"

  printf '%s\n' "$local_port" > "$port_file"
  printf '%s\n' "$local_port"
}

stop_tunnel() {
  local jobid="$1"
  local sock="$CACHE_DIR/j-${jobid}.sock"

  ssh -S "$sock" -O exit "$LOGIN_ALIAS" >/dev/null 2>&1 || die "No tunnel control socket found for job $jobid"
  rm -f "$CACHE_DIR/${jobid}.local_port"
  echo "Tunnel stopped for job $jobid." >&2
}

list_jobs() {
  remote "squeue -u \$USER -h -n $(q "$JOB_NAME") -o '%i %T %M %N %j'" || true
}

update_ssh_config_hostname() {
  local alias="$1"
  local node="$2"
  local cfg="${HOME}/.ssh/config"
  [[ -f "$cfg" ]] || return 0
  # Replace the HostName line that belongs to the given Host block.
  perl -i -0pe "s/((?:^|\n)Host[^\n]*\b${alias}\b[^\n]*\n(?:[^\n]*\n)*?[ \t]*HostName[ \t]+)\S+/\${1}${node}/" "$cfg" 2>/dev/null || true
}

connect_and_print_url() {
  local jobid="$1"
  local force_restart="$2"

  job_exists "$jobid" || die "Job $jobid is not active"

  local meta_text
  meta_text="$(wait_for_meta "$jobid")" || exit 1
  parse_meta "$meta_text"

  update_ssh_config_hostname "orcd-compute" "$META_NODE"

  local local_port
  local_port="$(start_or_restart_tunnel "$jobid" "$META_NODE" "$META_PORT" "$force_restart")"

  echo "http://localhost:${local_port}/lab?token=${META_TOKEN}"
  echo "JobID: ${jobid}  Node: ${META_NODE}  Project: ${META_PROJECT_DIR:-unknown}  RemotePort: ${META_PORT}  LocalPort: ${local_port}" >&2
  echo "Stop tunnel: orcd-jlab.sh --stop ${jobid}" >&2
  echo "Cancel job:  orcd-jlab.sh --cancel ${jobid}" >&2
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help)
        usage
        exit 0
        ;;
      --project)
        need_value "$1" "$#"
        PROJECT_DIR="$2"
        shift 2
        ;;
      --script)
        need_value "$1" "$#"
        SBATCH_SCRIPT="$2"
        shift 2
        ;;
      --profile)
        need_value "$1" "$#"
        PROFILE_FILE="$2"
        shift 2
        ;;
      --container)
        need_value "$1" "$#"
        CONTAINER="$2"
        shift 2
        ;;
      --mode)
        need_value "$1" "$#"
        MODE="$2"
        shift 2
        ;;
      --python)
        need_value "$1" "$#"
        PYTHON_VERSION="$2"
        shift 2
        ;;
      --login-alias)
        need_value "$1" "$#"
        LOGIN_ALIAS="$2"
        shift 2
        ;;
      --partition)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--partition=$2")
        shift 2
        ;;
      --gres)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--gres=$2")
        [[ "$2" == *gpu* ]] && USE_NV=1
        shift 2
        ;;
      --cpus)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--cpus-per-task=$2")
        shift 2
        ;;
      --mem)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--mem=$2")
        shift 2
        ;;
      --time)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--time=$2")
        shift 2
        ;;
      --account)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--account=$2")
        shift 2
        ;;
      --constraint)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--constraint=$2")
        shift 2
        ;;
      --qos)
        need_value "$1" "$#"
        SBATCH_ARGS+=("--qos=$2")
        shift 2
        ;;
      --job-name)
        need_value "$1" "$#"
        JOB_NAME="$2"
        SBATCH_ARGS+=("--job-name=$2")
        shift 2
        ;;
      --use-nv)
        USE_NV=1
        shift
        ;;
      --reconnect)
        ACTION="reconnect"
        shift
        if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
          JOBID="$1"
          shift
        fi
        ;;
      --stop)
        ACTION="stop"
        shift
        if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
          JOBID="$1"
          shift
        fi
        ;;
      --cancel|--kill)
        ACTION="cancel"
        shift
        if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
          JOBID="$1"
          shift
        fi
        ;;
      --list)
        ACTION="list"
        shift
        ;;
      --)
        shift
        break
        ;;
      -*)
        die "Unknown option: $1"
        ;;
      *)
        if [[ "$1" =~ ^[0-9]+$ ]]; then
          JOBID="$1"
          ACTION="attach"
          shift
        else
          die "Unexpected argument: $1"
        fi
        ;;
    esac
  done
}

main() {
  parse_args "$@"
  command -v ssh >/dev/null 2>&1 || die "ssh is not installed"

  if [[ -z "$PROJECT_DIR" && -z "$JOBID" && "$ACTION" != "list" ]]; then
    die "--project is required (or supply a job id directly)"
  fi

  case "$ACTION" in
    list)
      list_jobs
      ;;

    stop)
      if [[ -z "$JOBID" ]]; then
        JOBID="$(latest_active_jobid)"
      fi
      [[ -n "$JOBID" ]] || die "No active job found to stop"
      stop_tunnel "$JOBID"
      ;;

    cancel)
      if [[ -z "$JOBID" ]]; then
        JOBID="$(latest_active_jobid)"
      fi
      [[ -n "$JOBID" ]] || die "No active job found to cancel"
      remote "scancel $(q "$JOBID")" && echo "Cancelled job $JOBID" >&2 || die "Failed to cancel job $JOBID"
      stop_tunnel "$JOBID" 2>/dev/null || true
      ;;

    reconnect)
      if [[ -z "$JOBID" ]]; then
        JOBID="$(latest_active_jobid)"
      fi
      [[ -n "$JOBID" ]] || die "No active job found to reconnect to"
      connect_and_print_url "$JOBID" "1"
      ;;

    attach)
      [[ -n "$JOBID" ]] || die "No job id supplied"
      connect_and_print_url "$JOBID" "1"
      ;;

    submit)
      JOBID="$(submit_job)"
      [[ "$JOBID" =~ ^[0-9]+$ ]] || die "Could not parse job id from sbatch output"
      echo "Submitted job $JOBID" >&2
      connect_and_print_url "$JOBID" "0"
      ;;

    *)
      die "Unhandled action: $ACTION"
      ;;
  esac
}

main "$@"