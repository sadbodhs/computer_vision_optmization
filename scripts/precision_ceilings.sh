#!/bin/bash
# Precision / sparsity SPEED ceilings on the batch-1 engines.
#
# ############################ READ THIS ############################
# These are THROUGHPUT NUMBERS ONLY. No accuracy is measured, and the INT8
# engines here are built WITHOUT a calibration set -- TensorRT falls back to
# generated scales, so their detections are meaningless. The point is to bound
# how much speed INT8 / 2:4 sparsity could buy on this GPU, so you can decide
# whether the accuracy work (a real calibration set + mAP harness) is worth
# doing. Do NOT quote these as "INT8 results".
# ###################################################################
#
# Engines are built into /tmp INSIDE the container; the model repo is never
# touched.
#
# Usage: scripts/precision_ceilings.sh [models...]   (default: yolov8s yolov8n)
set -euo pipefail
MODELS=${@:-"yolov8s yolov8n"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R=$ROOT/results/v3
mkdir -p "$R"
OUTF=$R/precision_ceilings.tsv
CONTAINER=${CONTAINER:-triton-server}
TRTEXEC=/usr/src/tensorrt/bin/trtexec

echo -e "model\tvariant\tthroughput_qps\tlatency_median_ms\tnote" > "$OUTF"

for m in $MODELS; do
  for variant in fp16 fp16_sparse int8_nocalib best_nocalib; do
    case $variant in
      fp16)          FLAGS="--fp16" ;;
      fp16_sparse)   FLAGS="--fp16 --sparsity=force" ;;
      int8_nocalib)  FLAGS="--int8" ;;
      best_nocalib)  FLAGS="--best" ;;
    esac
    echo "--- building $m / $variant ---" >&2
    out=$(docker exec $CONTAINER bash -c "
      $TRTEXEC --onnx=/models/$m/1/model.onnx $FLAGS \
        --saveEngine=/tmp/${m}_${variant}.plan --warmUp=500 --duration=5 2>&1" || true)
    tp=$(echo "$out" | grep "Throughput:" | head -1 \
         | awk '{for(i=1;i<=NF;i++) if($i=="Throughput:"){print $(i+1); exit}}')
    lat=$(echo "$out" | grep -E "\[I\] Latency:" | head -1 \
          | grep -oE "median = [0-9.]+" | awk '{print $3}')
    note="speed only"
    [[ "$variant" == *nocalib* ]] && note="speed only - NO calibration, accuracy invalid"
    [[ "$variant" == fp16_sparse ]] && note="speed ceiling - dense weights forced sparse, accuracy invalid"
    echo -e "$m\t$variant\t${tp:-BUILD_FAILED}\t${lat:-}\t$note" | tee -a "$OUTF"
    docker exec $CONTAINER rm -f /tmp/${m}_${variant}.plan 2>/dev/null || true
  done
done

echo "=== saved to $OUTF ==="
