#!/bin/bash
# Compile ds_bench (Flow E) against DeepStream + GStreamer. Run inside the
# ds-build image (nvcr.io/nvidia/deepstream:7.1-*), where the DeepStream SDK is at
# /opt/nvidia/deepstream/deepstream. Produces /work/cpp/build/ds_bench.
set -euo pipefail
DS=/opt/nvidia/deepstream/deepstream
OUT=${1:-/work/cpp/build/ds_bench}
mkdir -p "$(dirname "$OUT")"

g++ -std=c++17 -O2 -o "$OUT" /work/cpp/src/ds_bench.cpp \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0) \
    -I"$DS/sources/includes" \
    $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0) \
    -L"$DS/lib" -lnvdsgst_meta -lnvds_meta \
    -Wl,-rpath,"$DS/lib"

echo "built $OUT"
