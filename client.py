#!/usr/bin/env python3
"""Triton pipeline: NVDEC decode -> GPU preprocess -> gRPC (CUDA shm) -> Triton TRT -> GPU postprocess."""
import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from tritonclient import utils as triton_utils

IMG = 640
NUM_CLASSES = 80
NUM_ANCHORS = 8400


def letterbox_nv12_to_tensor(nv12: np.ndarray, src_w: int, src_h: int, device: str) -> torch.Tensor:
    """NV12 (H*1.5 x W) -> RGB CHW float32 0-1, letterboxed to 640x640, on GPU."""
    y = nv12[:src_h, :src_w]
    uv = nv12[src_h:src_h + src_h // 2, :src_w].reshape(src_h // 2, src_w // 2, 2)
    u = uv[:, :, 0].repeat(2, axis=0).repeat(2, axis=1)
    v = uv[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)
    # BT.601 limited-range YUV -> RGB
    yf = y.astype(np.float32) - 16.0
    uf = u.astype(np.float32) - 128.0
    vf = v.astype(np.float32) - 128.0
    r = 1.164 * yf + 1.596 * vf
    g = 1.164 * yf - 0.392 * uf - 0.813 * vf
    b = 1.164 * yf + 2.017 * uf
    rgb = np.stack([r, g, b], axis=0)  # 3,H,W
    t = torch.from_numpy(rgb).to(device).float() / 255.0
    # letterbox: resize keeping aspect, pad to 640x640
    scale = min(IMG / src_w, IMG / src_h)
    nw, nh = int(src_w * scale), int(src_h * scale)
    t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]
    pad_h = IMG - nh
    pad_w = IMG - nw
    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0.0)
    return t.unsqueeze(0)  # 1,3,640,640


def postprocess_gpu(out: torch.Tensor, conf_thr=0.25, iou_thr=0.45):
    """out: (1,84,8400) -> list of boxes. GPU decode + CPU NMS."""
    out = out[0]  # 84,8400
    boxes = out[:4, :]  # cx,cy,w,h
    scores = out[4:, :]  # 80,8400
    max_scores, max_cls = scores.max(dim=0)
    mask = max_scores > conf_thr
    if mask.sum() == 0:
        return 0
    boxes = boxes[:, mask].t()  # N,4
    scores = max_scores[mask]
    cls = max_cls[mask]
    # cxcywh -> xyxy
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    xyxy = torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()
    scores = scores.cpu().numpy()
    cls = cls.cpu().numpy()
    # simple NMS
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xyxy[i, 0], xyxy[order[1:], 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[order[1:], 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[order[1:], 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        uni = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]) + \
              (xyxy[order[1:], 2] - xyxy[order[1:], 0]) * (xyxy[order[1:], 3] - xyxy[order[1:], 1]) - inter
        iou = inter / np.maximum(uni, 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return len(keep)


def decode_stream(url: str, src_w: int, src_h: int):
    """NVDEC decode via ffmpeg h264_cuvid, yield raw NV12 frames."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-hwaccel", "cuda", "-c:v", "h264_cuvid",
        "-i", url, "-f", "rawvideo", "-pix_fmt", "nv12", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = src_w * src_h * 3 // 2
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(src_h * 3 // 2, src_w)


def run_stream(url, model, client, device, duration, results, idx):
    src_w, src_h = 1280, 720
    frames = 0
    dets = 0
    t0 = time.perf_counter()
    for nv12 in decode_stream(url, src_w, src_h):
        if time.perf_counter() - t0 > duration:
            break
        inp = letterbox_nv12_to_tensor(nv12, src_w, src_h, device)
        inputs = [grpcclient.InferInput("images", list(inp.shape), "FP32")]
        inputs[0].set_data_from_numpy(inp.cpu().numpy())
        outputs = [grpcclient.InferRequestedOutput("output0")]
        res = client.infer(model, inputs, outputs=outputs)
        out = torch.from_numpy(res.as_numpy("output0")).to(device)
        dets += postprocess_gpu(out)
        frames += 1
    dt = time.perf_counter() - t0
    results[idx] = {"stream": idx, "frames": frames, "detections": dets, "fps": frames / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtsp://localhost:8554/cam1")
    ap.add_argument("--model", default="yolov8s")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--server", default="localhost:8001")
    args = ap.parse_args()

    client = grpcclient.InferenceServerClient(url=args.server)
    if not client.is_server_live():
        print("Triton not live"); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    urls = [args.url.replace("cam1", f"cam{i+1}") for i in range(args.streams)]
    results = [None] * args.streams
    threads = [threading.Thread(target=run_stream, args=(u, args.model, client, device, args.duration, results, i))
               for i, u in enumerate(urls)]
    for t in threads: t.start()
    for t in threads: t.join()

    total_frames = sum(r["frames"] for r in results)
    total_dets = sum(r["detections"] for r in results)
    print(json.dumps({"pipeline": "triton", "streams": args.streams, "frames": total_frames,
                      "detections": total_dets, "fps": total_frames / args.duration,
                      "per_stream_fps": [r["fps"] for r in results]}))


if __name__ == "__main__":
    main()
