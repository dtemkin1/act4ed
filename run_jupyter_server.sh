#!/usr/bin/env bash
#SBATCH --job-name=jlab
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

load_profile_if_present() {
  local project_dir="$1"
  local profile_file="${JLAB_PROFILE_FILE:-}"

  if [[ -n "$profile_file" && "$profile_file" != /* ]]; then
    profile_file="${project_dir%/}/$profile_file"
  fi

  if [[ -n "$profile_file" && -r "$profile_file" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "$profile_file"
    set -u
  elif [[ -r "${project_dir%/}/.orcd-jlab.env" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${project_dir%/}/.orcd-jlab.env"
    set -u
  fi
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

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  if [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  fi

  command -v module >/dev/null 2>&1 || return 1
}

ensure_apptainer() {
  if command -v apptainer >/dev/null 2>&1; then
    return 0
  fi

  ensure_module_cmd || true
  module load apptainer >/dev/null 2>&1 || module load apptainer/1.4.2 >/dev/null 2>&1 || true

  command -v apptainer >/dev/null 2>&1 || {
    echo "Apptainer was requested but the apptainer command is not available." >&2
    exit 1
  }
}

main() {
  local initial_project_dir="${JLAB_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
  load_profile_if_present "$initial_project_dir"

  local project_dir="${JLAB_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
  local mode="${JLAB_MODE:-uv}"                    # uv | venv | custom
  local python_for_uv="${JLAB_PYTHON:-}"
  local venv_path="${JLAB_VENV_PATH:-.venv}"
  local container="${JLAB_CONTAINER:-}"
  local use_nv="${JLAB_USE_NV:-0}"
  local apptainer_args="${JLAB_APPTAINER_ARGS:-}"
  local jupyter_bin="${JLAB_JUPYTER_BIN:-jupyter}"
  local jupyter_args="${JLAB_JUPYTER_ARGS:-}"
  local pre_cmd="${JLAB_PRE_CMD:-}"
  local uv_frozen="${JLAB_UV_FROZEN:-1}"
  local log_dir_name="${JLAB_LOG_DIR:-logs}"
  local state_dir="${JLAB_STATE_DIR:-$HOME/.local/state/orcd-jlab}"
  local start_timeout="${JLAB_START_TIMEOUT:-1800}"

  [[ -d "$project_dir" ]] || {
    echo "Project directory does not exist: $project_dir" >&2
    exit 1
  }

  if [[ -n "$container" && "$container" != /* ]]; then
    container="${project_dir%/}/$container"
  fi

  local log_dir="${project_dir%/}/$log_dir_name"
  mkdir -p "$log_dir" "$state_dir"

  local host_python
  host_python="$(pick_python)" || {
    echo "Could not find python3 or python on the compute node." >&2
    exit 1
  }

  local node
  node="${SLURMD_NODENAME:-$(hostname -s)}"

  local port
  port="$($host_python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"

  local token
  token="$($host_python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"

  local meta_file="${state_dir%/}/jlab-${SLURM_JOB_ID}.env"
  local launcher="${state_dir%/}/jlab-${SLURM_JOB_ID}.launch.sh"
  local launcher_effective_project_dir="$project_dir"

  cat > "$launcher" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    return 1
  fi
}

ensure_uv() {
  export PATH="$HOME/.local/bin:$PATH"

  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  local py
  py="$(pick_python)" || {
    echo "uv was requested, but python3/python is not available to install it." >&2
    exit 1
  }

  "$py" -m pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"

  command -v uv >/dev/null 2>&1 || {
    echo "uv installation finished, but uv is still not on PATH." >&2
    exit 1
  }
}

resolve_venv_dir() {
  if [[ "$JLAB_VENV_PATH" = /* ]]; then
    printf '%s\n' "$JLAB_VENV_PATH"
  else
    printf '%s\n' "${JLAB_EFFECTIVE_PROJECT_DIR%/}/$JLAB_VENV_PATH"
  fi
}

cd "$JLAB_EFFECTIVE_PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export JUPYTER_RUNTIME_DIR="${JLAB_STATE_DIR%/}/runtime/$SLURM_JOB_ID"
mkdir -p "$JUPYTER_RUNTIME_DIR"

if [[ -n "${JLAB_PRE_CMD:-}" ]]; then
  eval "$JLAB_PRE_CMD"
fi

case "$JLAB_MODE" in
  uv)
    ensure_uv

    if [[ ! -d .venv ]]; then
      if [[ -n "${JLAB_PYTHON:-}" ]]; then
        uv venv --python "$JLAB_PYTHON"
      else
        uv venv
      fi
    fi

    if [[ -f pyproject.toml ]]; then
      if [[ "${JLAB_UV_FROZEN:-1}" == "1" && -f uv.lock ]]; then
        uv sync --frozen
      else
        uv sync
      fi
    elif [[ -f requirements.txt ]]; then
      uv pip install -r requirements.txt
    fi

    jlab_cmd=(uv run jupyter lab)
    ;;

  venv)
    venv_dir="$(resolve_venv_dir)"
    [[ -x "$venv_dir/bin/jupyter" ]] || {
      echo "Could not find jupyter in $venv_dir/bin/jupyter" >&2
      exit 1
    }
    jlab_cmd=("$venv_dir/bin/jupyter" lab)
    ;;

  custom)
    jlab_cmd=("${JLAB_JUPYTER_BIN:-jupyter}" lab)
    ;;

  *)
    echo "Unknown JLAB_MODE=$JLAB_MODE. Use uv, venv, or custom." >&2
    exit 1
    ;;
esac

# shellcheck disable=SC2086
"${jlab_cmd[@]}" \
  --ServerApp.ip=0.0.0.0 \
  --ServerApp.port="$JLAB_PORT" \
  --ServerApp.port_retries=0 \
  --ServerApp.open_browser=False \
  --ServerApp.allow_remote_access=True \
  --ServerApp.terminals_enabled=True \
  --ServerApp.root_dir="$JLAB_EFFECTIVE_PROJECT_DIR" \
  --ServerApp.custom_display_url="http://localhost:$JLAB_PORT" \
  --ServerApp.token="$JLAB_TOKEN" \
  ${JLAB_JUPYTER_ARGS:-}
LAUNCHER
  chmod 700 "$launcher"
  trap 'rm -f "$launcher" "$meta_file"' EXIT

  export JLAB_MODE="$mode"
  export JLAB_PYTHON="$python_for_uv"
  export JLAB_VENV_PATH="$venv_path"
  export JLAB_USE_NV="$use_nv"
  export JLAB_JUPYTER_BIN="$jupyter_bin"
  export JLAB_JUPYTER_ARGS="$jupyter_args"
  export JLAB_PRE_CMD="$pre_cmd"
  export JLAB_UV_FROZEN="$uv_frozen"
  export JLAB_STATE_DIR="$state_dir"
  export JLAB_PORT="$port"
  export JLAB_TOKEN="$token"

  if [[ -n "$container" ]]; then
    ensure_apptainer
    [[ -r "$container" ]] || {
      echo "Container image does not exist or is not readable: $container" >&2
      exit 1
    }
    launcher_effective_project_dir="/work"
  fi

  export JLAB_EFFECTIVE_PROJECT_DIR="$launcher_effective_project_dir"

  echo "============================================================"
  echo "JLAB_JOBID=$SLURM_JOB_ID"
  echo "JLAB_NODE=$node"
  echo "JLAB_PROJECT_DIR=$project_dir"
  echo "JLAB_META_FILE=$meta_file"
  echo "============================================================"

  if [[ -n "$container" ]]; then
    export JLAB_CONTAINER="$container"

    local nv_flag=()
    if [[ "$use_nv" == "1" ]]; then
      nv_flag=(--nv)
    fi

    # shellcheck disable=SC2086
    apptainer exec \
      "${nv_flag[@]}" \
      $apptainer_args \
      --bind "$project_dir:/work" \
      --pwd /work \
      "$container" \
      bash "$launcher" &
  else
    bash "$launcher" &
  fi

  local jlab_pid=$!
  local ready=0

  for _ in $(seq 1 "$start_timeout"); do
    if "$host_python" - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1.0)
try:
    ok = (s.connect_ex(("127.0.0.1", port)) == 0)
finally:
    s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      ready=1
      break
    fi

    if ! kill -0 "$jlab_pid" 2>/dev/null; then
      wait "$jlab_pid"
      exit 1
    fi

    sleep 1
  done

  if [[ "$ready" != "1" ]]; then
    echo "Jupyter did not become ready before the timeout (${start_timeout}s)." >&2
    kill "$jlab_pid" 2>/dev/null || true
    wait "$jlab_pid" 2>/dev/null || true
    exit 1
  fi

  umask 077
  cat > "$meta_file" <<EOF_META
JLAB_JOBID=$SLURM_JOB_ID
JLAB_NODE=$node
JLAB_PORT=$port
JLAB_TOKEN=$token
JLAB_PROJECT_DIR=$project_dir
JLAB_MODE=$mode
JLAB_CONTAINER=$container
EOF_META

  echo "JLAB_READY=1"
  echo "JLAB_NODE=$node"
  echo "JLAB_PORT=$port"
  echo "Use the local helper script to create the tunnel and print the final localhost URL."

  wait "$jlab_pid"
}

main "$@"
