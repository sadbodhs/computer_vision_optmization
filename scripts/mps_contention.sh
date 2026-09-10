#!/bin/bash
# CUDA MPS A/B for the contention study: N independent A2 pipelines on one GPU,
# measured with the MPS daemon off and on, everything else identical.
#
# Why this exists: docs/contention.md concluded "Triton shares the GPU more
# gracefully than raw CUDA contexts" from A2 +153% vs B2 +76% latency growth at
# N=3. That was measured with MPS OFF -- and inter-process context-switch
# serialization is exactly what MPS removes. This script tests that.
#
# Usage: scripts/mps_contention.sh [duration] [N]
#
# IMPORTANT -- two constraints this script works around:
#  1. MPS clients must run as the same uid as the MPS daemon. The daemon here
#     runs as the invoking user, so the benchmark container runs --user <uid>.
#  2. While an MPS daemon is active, CUDA clients that CANNOT reach it fail with
#     "MPS client failed to connect (error 805)". The root-owned triton-server
#     container is one of those, so it is stopped for the MPS half and restarted
#     afterwards. A2 is in-process TensorRT and needs no server, so this costs
#     nothing scientifically.
set -euo pipefail
DURATION=${1:-8}
N=${2:-3}
REPEATS=${REPEATS:-3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R=$ROOT/results/v3
mkdir -p "$R"
OUTF=$R/mps_contention_N${N}.tsv

STAGE=${STAGE:-/home/$USER/mps_test}
IMAGE=${IMAGE:-triton-bench:v3}
CONTAINER=${CONTAINER:-triton-server}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
UID_GID="$(id -u):$(id -g)"

# --- stage engine + binary + frames.bin somewhere the uid can read -----------
mkdir -p "$STAGE" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ ! -f "$STAGE/frames.bin" ]; then
  docker cp $CONTAINER:/work/cpp/build/frames.bin "$STAGE/frames.bin"
fi
[ -f "$STAGE/model.plan" ] || cp "$ROOT/triton/models/yolov8s/1/model.plan" "$STAGE/model.plan"
[ -f "$STAGE/trt_pipeline_cuda" ] || docker cp $CONTAINER:/work/cpp/build/trt_pipeline_cuda "$STAGE/trt_pipeline_cuda"

run_batch() {  # run_batch <label> <extra docker args...>
  local LABEL=$1; shift
  docker run --rm --gpus all --user "$UID_GID" "$@" -v "$STAGE:/data" "$IMAGE" bash -c "
    cd /data
    for i in \$(seq $N); do
      ./trt_pipeline_cuda --engine model.plan --mode file --file frames.bin \
        --streams 1 --duration $DURATION > /tmp/r\$i.json 2>/dev/null &
    done
    wait
    for i in \$(seq $N); do tail -1 /tmp/r\$i.json; done
  " 2>/dev/null | while read -r line; do
    # the container prints a startup banner on stdout: keep only JSON result lines
    case "$line" in
      \{*) echo -e "${LABEL}\t${line}" | tee -a "$OUTF" ;;
    esac
  done
}

echo -e "condition\tjson" > "$OUTF"

echo "=== MPS OFF (N=$N, ${REPEATS} repeats) ==="
for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_batch "mps_off"; done

echo "=== starting MPS daemon (stopping $CONTAINER: it cannot reach the daemon) ==="
docker stop $CONTAINER > /dev/null 2>&1 || true
nvidia-cuda-mps-control -d
sleep 2

echo "=== MPS ON (N=$N, ${REPEATS} repeats) ==="
for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_batch "mps_on" --ipc=host \
  -e CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
  -e CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
  -v "$CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY" \
  -v "$CUDA_MPS_LOG_DIRECTORY:$CUDA_MPS_LOG_DIRECTORY"; done

echo "=== stopping MPS daemon and restoring $CONTAINER ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2
docker start $CONTAINER > /dev/null 2>&1 || true

echo "=== saved to $OUTF ==="
python3 - "$OUTF" <<'PY'
import json, sys, statistics, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    cond, _, js = line.strip().partition("\t")
    if js.startswith("{"):
        d = json.loads(js); rows[cond].append((d["fps"], d["lat_ms_p50"]))
n = None
for cond in ("mps_off", "mps_on"):
    if cond not in rows: continue
    v = rows[cond]
    print(f"{cond}: instances={len(v)}  total_fps~{sum(f for f,_ in v)/ (len(v)//3 or 1):.1f}  "
          f"lat_p50 median={statistics.median(l for _,l in v):.3f} ms")
PY
