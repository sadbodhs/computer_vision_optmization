#!/bin/bash
# ultralytics <name> -> ONNX at <imgsz>, into /models/_exp/.
# ultralytics writes <name>.onnx next to the .pt, so move it afterwards.
set -e
m="$1"; sz="${2:-640}"
cd /models
yolo export model=/models/$m.pt format=onnx imgsz=$sz dynamic=false simplify=True >/tmp/${m}_exp.log 2>&1 \
  || { echo "$m EXPORT_FAILED: $(grep -iE 'error|not supported' /tmp/${m}_exp.log | tail -1)"; exit 0; }
mkdir -p /models/_exp
mv -f /models/$m.onnx /models/_exp/$m.onnx
echo "$m exported at $sz"
