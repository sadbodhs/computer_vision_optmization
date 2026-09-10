#!/bin/bash
# Rebuild both benchmark images from the repo (run from repo root).
#   docker/build.sh            # builds both
#   docker/build.sh triton     # just the Triton arm
#   docker/build.sh deepstream # just the DeepStream arm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WHAT=${1:-all}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "triton" ]; then
  echo "=== building triton-bench:v3 ==="
  docker build -f docker/Dockerfile.triton -t triton-bench:v3 .
fi
if [ "$WHAT" = "all" ] || [ "$WHAT" = "deepstream" ]; then
  echo "=== building ds-build:latest ==="
  docker build -f docker/Dockerfile.deepstream -t ds-build:latest .
fi
echo "done."
