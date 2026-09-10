#!/bin/bash
# Rebuild both benchmark images from the repo (run from repo root).
#   docker/build.sh            # builds both
#   docker/build.sh triton     # just the Triton arm
#   docker/build.sh deepstream # just the DeepStream arm
#
# Tags default to the names the scripts and docs expect. Override to build
# alongside a known-good image instead of replacing it:
#   TRITON_TAG=triton-bench:rebuild docker/build.sh triton
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WHAT=${1:-all}
TRITON_TAG=${TRITON_TAG:-triton-bench:v3}
DS_TAG=${DS_TAG:-ds-build:latest}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "triton" ]; then
  echo "=== building $TRITON_TAG ==="
  docker build -f docker/Dockerfile.triton -t "$TRITON_TAG" .
fi
if [ "$WHAT" = "all" ] || [ "$WHAT" = "deepstream" ]; then
  echo "=== building $DS_TAG ==="
  docker build -f docker/Dockerfile.deepstream -t "$DS_TAG" .
fi
echo "done."
