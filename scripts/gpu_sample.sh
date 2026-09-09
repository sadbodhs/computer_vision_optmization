#!/bin/bash
# Sample GPU utilization + memory during a benchmark run
# Usage: gpu_sample.sh <output_file> <duration_seconds>
OUT=$1
DUR=$2
echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib" > $OUT
END=$((SECONDS + DUR))
while [ $SECONDS -lt $END ]; do
  read -r util mem total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr ',' ' ')
  echo "$(date +%s),$util,$mem,$total" >> $OUT
  sleep 0.5
done
