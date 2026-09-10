#!/bin/bash
# Generate frames.bin inside the triton-server container (everything runs in Docker).
# Usage: scripts/make_frames.sh [source_video] [n_frames]
#   source_video : path RELATIVE TO THE REPO ROOT (default videos/real.mp4)
#   n_frames     : distinct frames to write (default 500)
#
# The repo is mounted at /work in triton-server, so the same relative paths work
# inside the container. Output lands at cpp/build/frames.bin (where the binaries
# and client_v2.py expect it).
set -euo pipefail
VIDEO=${1:-videos/real.mp4}
NFRAMES=${2:-500}

docker exec triton-server bash -c "cd /work && python3 scripts/make_frames.py --video '$VIDEO' --out cpp/build/frames.bin --frames $NFRAMES"
