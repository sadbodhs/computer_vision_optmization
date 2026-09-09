#!/bin/bash
# Staggered parallel launch for client_v2 instances (avoids shm registration race)
DURATION=$1; N=$2
cd /work
for i in $(seq $N); do
  python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 1 --duration $DURATION > /tmp/par_C2_$i.json 2>/dev/null &
  sleep 3
done
wait