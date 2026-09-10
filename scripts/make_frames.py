#!/usr/bin/env python3
"""Generate frames.bin — the preprocessed-tensor replay file used by every
capacity-mode benchmark (arms A/B/C/D all read it via --file frames.bin).

Format (headerless, must stay in lockstep with every consumer):
    N frames, each frame = [1,3,640,640] float32, C-contiguous (CHW).
    frame_bytes = 3*640*640*4 = 4,915,200.  n_frames = filesize / frame_bytes.
Consumers: cpp/src/*.cu|*.cpp read `in_size` chunks; triton/client_v2.py does
    np.memmap(...).reshape(-1, 1, 3, 640, 640).

Preprocessing is reused verbatim from triton/client_v2.py:letterbox_nv12_np
(centered letterbox, pad 114, BGR->RGB, /255) so the tensors — and therefore the
detection counts — are byte-identical to the C2 numpy path and to whatever the
original frames.bin contained.

Runs inside the Triton container (has numpy + ffmpeg). See scripts/make_frames.sh
for the Docker wrapper.

Example (inside triton-server):
    python3 scripts/make_frames.py --video videos/real.mp4 --out cpp/build/frames.bin --frames 500
"""
import argparse
import os
import subprocess
import sys

import numpy as np

# Reuse the exact preprocessing the C2 client uses, so we never drift from it.
# client_v2.py lives at triton/client_v2.py in the repo, but at /work/client_v2.py
# in the baked container image — search both layouts (+ the script's own dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "triton"), _HERE, "/work", os.getcwd()):
    if os.path.exists(os.path.join(_p, "client_v2.py")):
        sys.path.insert(0, _p)
        break
from client_v2 import IMG, letterbox_nv12_np  # noqa: E402

FRAME_BYTES = 3 * IMG * IMG * 4


def probe_resolution(src: str):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", src]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def decode_nv12(src: str, src_w: int, src_h: int):
    """Yield raw NV12 frames (uint8) decoded from a file or RTSP URL.

    CPU decode on purpose: the letterboxed tensor is identical regardless of the
    decoder, and CPU decode has no cuvid/driver dependency, so frames.bin is
    reproducible anywhere ffmpeg runs.
    """
    cmd = ["ffmpeg", "-loglevel", "error", "-i", src,
           "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs = src_w * src_h * 3 // 2
    try:
        while True:
            b = proc.stdout.read(fs)
            if len(b) < fs:
                break
            yield np.frombuffer(b, dtype=np.uint8)
    finally:
        proc.stdout.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a source video file (e.g. videos/real.mp4)")
    src.add_argument("--url", help="RTSP/HTTP source URL")
    ap.add_argument("--out", default="cpp/build/frames.bin", help="output path for frames.bin")
    ap.add_argument("--frames", type=int, default=500,
                    help="number of distinct frames to write (binaries loop over them)")
    args = ap.parse_args()

    source = args.video or args.url
    if args.video and not os.path.exists(args.video):
        ap.error(f"video not found: {args.video}")

    src_w, src_h = probe_resolution(source)
    print(f"source {source}  {src_w}x{src_h}  ->  {args.frames} x [1,3,{IMG},{IMG}] f32", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = 0
    with open(args.out, "wb") as fout:
        for nv12 in decode_nv12(source, src_w, src_h):
            tensor = letterbox_nv12_np(nv12, src_w, src_h)  # (1,3,640,640) f32
            assert tensor.dtype == np.float32 and tensor.nbytes == FRAME_BYTES, tensor.shape
            fout.write(np.ascontiguousarray(tensor).tobytes())
            written += 1
            if written >= args.frames:
                break

    if written == 0:
        print("ERROR: no frames decoded — check the source", file=sys.stderr)
        sys.exit(1)
    size = os.path.getsize(args.out)
    print(f"wrote {written} frames -> {args.out} ({size/1e6:.1f} MB, "
          f"{size // FRAME_BYTES} frames on disk)", file=sys.stderr)


if __name__ == "__main__":
    main()
