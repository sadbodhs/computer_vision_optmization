#!/bin/bash
# CUDA MPS A/B for the contention study: N independent A2 pipelines on one GPU,
# measured with the MPS daemon off and on, everything else identical.
#
# Why this exists: docs/contention.md concluded "Triton shares the GPU more
# gracefully than raw CUDA contexts" from A2 +153% vs B2 +76% latency growth at
# N=3. That was measured with MPS OFF -- and inter-process context-switch
# serialization is exactly what MPS removes. This script tests that.
#
# Covers BOTH flows:
#   A2 - N independent in-process TensorRT pipelines (N CUDA contexts)
#   B2 - N gRPC clients against ONE Triton server (server holds the CUDA context)
# The contrast is the whole point: MPS multiplexes separate processes, and Triton
# already avoids having separate processes to multiplex.
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

cleanup() {
  echo "--- cleanup: removing test server, stopping MPS, restoring $CONTAINER ---" >&2
  docker rm -f triton-mps-test > /dev/null 2>&1 || true
  # a leftover MPS daemon makes every root CUDA container fail with error 805
  if pgrep -f nvidia-cuda-mps-control > /dev/null 2>&1; then
    echo quit | nvidia-cuda-mps-control > /dev/null 2>&1 || true
    sleep 2
  fi
  docker start $CONTAINER > /dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

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

# --- B2: N gRPC clients against one non-root Triton server ------------------
# The server must run --user <uid> too: an MPS client must share the daemon's uid,
# and a root server cannot reach a user-owned daemon (error 805).
start_nonroot_server() {  # start_nonroot_server <extra docker args...>
  docker rm -f triton-mps-test > /dev/null 2>&1 || true
  docker run -d --name triton-mps-test --gpus all --shm-size=2g --network host \
    --user "$UID_GID" "$@" -v "$ROOT/triton/models:/models" -v "$STAGE:/data" \
    "$IMAGE" tritonserver --model-repository=/models > /dev/null
  for i in $(seq 1 45); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready 2>/dev/null)" = "200" ] && return 0
    sleep 2
  done
  return 1
}

run_b2() {  # run_b2 <label>
  local LABEL=$1
  docker exec triton-mps-test bash -c "
    cd /work/cpp/build
    for i in \$(seq $N); do
      ./trt_grpc_cuda --mode file --file /data/frames.bin --model yolov8s \
        --streams 1 --duration $DURATION > /tmp/b\$i.json 2>/dev/null &
    done
    wait
    for i in \$(seq $N); do tail -1 /tmp/b\$i.json; done
  " 2>/dev/null | while read -r line; do
    case "$line" in
      \{*) echo -e "${LABEL}\t${line}" | tee -a "$OUTF" ;;
    esac
  done
}

echo -e "condition\tjson" > "$OUTF"

echo "=== MPS OFF (N=$N, ${REPEATS} repeats) ==="
for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_batch "a2_mps_off"; done

echo "=== B2 MPS OFF (non-root server, N=$N) ==="
docker stop $CONTAINER > /dev/null 2>&1 || true
if start_nonroot_server; then
  for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_b2 "b2_mps_off"; done
else echo "b2 server failed to start (MPS off)" >&2; fi
docker rm -f triton-mps-test > /dev/null 2>&1 || true

echo "=== starting MPS daemon ($CONTAINER stays down: it cannot reach the daemon) ==="
nvidia-cuda-mps-control -d
sleep 2

echo "=== B2 MPS ON (non-root server, N=$N) ==="
MPS_ARGS=(--ipc=host
  -e CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY"
  -e CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY"
  -v "$CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY"
  -v "$CUDA_MPS_LOG_DIRECTORY:$CUDA_MPS_LOG_DIRECTORY")
if start_nonroot_server "${MPS_ARGS[@]}"; then
  for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_b2 "b2_mps_on"; done
else echo "b2 server failed to start (MPS on)" >&2; fi
docker rm -f triton-mps-test > /dev/null 2>&1 || true

echo "=== MPS ON (N=$N, ${REPEATS} repeats) ==="
for r in $(seq $REPEATS); do echo "-- repeat $r --"; run_batch "a2_mps_on" --ipc=host \
  -e CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
  -e CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
  -v "$CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY" \
  -v "$CUDA_MPS_LOG_DIRECTORY:$CUDA_MPS_LOG_DIRECTORY"; done

echo "=== stopping MPS daemon and restoring $CONTAINER ==="
docker rm -f triton-mps-test > /dev/null 2>&1 || true
echo quit | nvidia-cuda-mps-control || true
sleep 2
docker start $CONTAINER > /dev/null 2>&1 || true

echo "=== saved to $OUTF ==="
python3 - "$OUTF" <<'PYEOF'
import json, sys, statistics, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    cond, _, js = line.strip().partition("\t")
    if js.startswith("{"):
        d = json.loads(js)
        rows[cond].append((d["fps"], d["lat_ms_p50"]))
hdr = "%-13s %9s %11s %12s" % ("condition", "instances", "total_fps", "lat_p50_ms")
print(hdr)
for cond in ("a2_mps_off", "a2_mps_on", "b2_mps_off", "b2_mps_on"):
    v = rows.get(cond)
    if not v:
        print("%-13s %9s" % (cond, "(no data)"))
        continue
    totals = [sum(f for f, _ in v[i:i + 3]) for i in range(0, len(v), 3)]
    print("%-13s %9d %11.1f %12.3f" % (
        cond, len(v), statistics.median(totals),
        statistics.median(l for _, l in v)))
PYEOF
