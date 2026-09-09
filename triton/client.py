#!/usr/bin/env python3
"""Triton Python client pipeline.
Modes:
  rtsp : live RTSP -> NVDEC decode -> letterbox -> gRPC -> postprocess
  file : replay preprocessed frames.bin -> gRPC -> postprocess (capacity mode)
Fair-preprocessing parity with the C++ arms: centered letterbox, pad=114, /255, RGB.
Reports per-frame inference latency percentiles (gRPC round-trip).
"""
import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import tritonclient.grpc as grpcclient

IMG = 640
NUM_CLASSES = 80
NUM_ANCHORS = 8400
IN_BYTES = 1 * 3 * IMG * IMG * 4
OUT_BYTES = 1 * 84 * NUM_ANCHORS * 4


def letterbox_nv12_to_tensor(nv12: np.ndarray, src_w: int, src_h: int, device: str) -> torch.Tensor:
    """NV12 -> RGB CHW float32 0-1, centered letterbox to 640x640 with pad=114 (matches C++ arms)."""
    y = nv12[:src_h, :src_w]
    uv = nv12[src_h:src_h + src_h // 2, :src_w].reshape(src_h // 2, src_w // 2, 2)
    u = uv[:, :, 0].repeat(2, axis=0).repeat(2, axis=1)
    v = uv[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)
    yf = y.astype(np.float32) - 16.0
    uf = u.astype(np.float32) - 128.0
    vf = v.astype(np.float32) - 128.0
    r = 1.164 * yf + 1.596 * vf
    g = 1.164 * yf - 0.392 * uf - 0.813 * vf
    b = 1.164 * yf + 2.017 * uf
    rgb = np.stack([r, g, b], axis=0)  # 3,H,W
    t = torch.from_numpy(rgb).to(device).float() / 255.0
    scale = min(IMG / src_w, IMG / src_h)
    nw, nh = int(src_w * scale), int(src_h * scale)
    t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]
    pad_y = (IMG - nh) // 2
    pad_x = (IMG - nw) // 2
    # centered pad with 114/255
    canvas = torch.full((3, IMG, IMG), 114.0 / 255.0, device=t.device)
    canvas[:, pad_y:pad_y + nh, pad_x:pad_x + nw] = t
    return canvas.unsqueeze(0)  # 1,3,640,640


def postprocess_count(out: torch.Tensor, conf_thr=0.25, iou_thr=0.45) -> int:
    """out: (1,84,8400) -> number of kept boxes after class-aware NMS (matches C++ arms)."""
    out = out[0]  # 84,8400
    boxes = out[:4, :]
    scores = out[4:, :]
    max_scores, max_cls = scores.max(dim=0)
    mask = max_scores > conf_thr
    if mask.sum() == 0:
        return 0
    boxes = boxes[:, mask].t()
    scores = max_scores[mask].cpu().numpy()
    cls = max_cls[mask].cpu().numpy()
    x1 = (boxes[:, 0] - boxes[:, 2] / 2).cpu().numpy()
    y1 = (boxes[:, 1] - boxes[:, 3] / 2).cpu().numpy()
    x2 = (boxes[:, 0] + boxes[:, 2] / 2).cpu().numpy()
    y2 = (boxes[:, 1] + boxes[:, 3] / 2).cpu().numpy()
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    order = scores.argsort()[::-1]
    keep = 0
    removed = np.zeros(len(order), dtype=bool)
    for ii in range(len(order)):
        if removed[ii]:
            continue
        keep += 1
        i = order[ii]
        for jj in range(ii + 1, len(order)):
            if removed[jj]:
                continue
            j = order[jj]
            if cls[i] != cls[j]:
                continue
            xx1 = max(xyxy[i, 0], xyxy[j, 0]); yy1 = max(xyxy[i, 1], xyxy[j, 1])
            xx2 = min(xyxy[i, 2], xyxy[j, 2]); yy2 = min(xyxy[i, 3], xyxy[j, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            uni = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]) + \
                  (xyxy[j, 2] - xyxy[j, 0]) * (xyxy[j, 3] - xyxy[j, 1]) - inter
            if inter / max(uni, 1e-6) > iou_thr:
                removed[jj] = True
    return keep


def decode_stream(url: str, src_w: int, src_h: int):
    cmd = ["ffmpeg", "-loglevel", "error", "-hwaccel", "cuda", "-c:v", "h264_cuvid",
           "-i", url, "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = src_w * src_h * 3 // 2
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(src_h * 3 // 2, src_w)


def get_resolution(url: str):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", url]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def run_stream(args, url, model, client, device, results, idx, frames_np, lock):
    latencies = []
    frames = 0
    dets = 0
    t0 = time.perf_counter()

    use_shm = getattr(args, "cuda_shm", False) and args.mode == "file"
    shm_handles = []
    if use_shm:
        import tritonclient.utils.cuda_shared_memory as csm
        try:
            in_region = csm.create_shared_memory_region(f"in_{idx}", IN_BYTES, 0)
            out_region = csm.create_shared_memory_region(f"out_{idx}", OUT_BYTES, 0)
            client.register_cuda_shared_memory(in_region._triton_shm_name, csm.get_raw_handle(in_region), 0, IN_BYTES)
            client.register_cuda_shared_memory(out_region._triton_shm_name, csm.get_raw_handle(out_region), 0, OUT_BYTES)
            shm_handles = [in_region, out_region]
            in_shm = grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")
            in_shm.set_shared_memory(in_region._triton_shm_name, 0, IN_BYTES)
            out_shm = grpcclient.InferRequestedOutput("output0")
            out_shm.set_shared_memory(out_region._triton_shm_name, 0, OUT_BYTES)
            inputs, outputs = [in_shm], [out_shm]
        except Exception as e:
            print(f"CUDA shm not available ({e}); falling back to raw gRPC", file=sys.stderr)
            use_shm = False

    if args.mode == "file":
        n_frames = frames_np.shape[0]
        fi = 0
        while time.perf_counter() - t0 < args.duration:
            ts = time.perf_counter()
            if use_shm:
                # write frame directly into the registered CUDA shm input region
                import tritonclient.utils.cuda_shared_memory as csm
                csm.set_shared_memory_region(shm_handles[0], frames_np[fi].reshape(-1))
                res = client.infer(model, inputs, outputs=outputs)
                out = res.as_numpy("output0")
            else:
                inp = frames_np[fi].copy()
                inputs = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
                inputs[0].set_data_from_numpy(inp)
                outputs = [grpcclient.InferRequestedOutput("output0")]
                res = client.infer(model, inputs, outputs=outputs)
                out = res.as_numpy("output0")
            latencies.append((time.perf_counter() - ts) * 1000)
            with lock:
                dets_local = postprocess_count(torch.from_numpy(out))
            dets += dets_local
            frames += 1
            fi = (fi + 1) % n_frames
    else:
        src_w, src_h = get_resolution(url)
        for nv12 in decode_stream(url, src_w, src_h):
            if time.perf_counter() - t0 > args.duration:
                break
            inp = letterbox_nv12_to_tensor(nv12, src_w, src_h, device)
            ts = time.perf_counter()
            inputs = [grpcclient.InferInput("images", list(inp.shape), "FP32")]
            inputs[0].set_data_from_numpy(inp.cpu().numpy())
            outputs = [grpcclient.InferRequestedOutput("output0")]
            res = client.infer(model, inputs, outputs=outputs)
            out = res.as_numpy("output0")
            latencies.append((time.perf_counter() - ts) * 1000)
            with lock:
                dets_local = postprocess_count(torch.from_numpy(out))
            dets += dets_local
            frames += 1

    dt = time.perf_counter() - t0
    lat = np.array(latencies)
    results[idx] = {
        "stream": idx, "frames": frames, "detections": dets,
        "fps": frames / dt,
        "lat_p50": float(np.percentile(lat, 50)) if len(lat) else 0,
        "lat_p95": float(np.percentile(lat, 95)) if len(lat) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtsp://localhost:8554/cam1")
    ap.add_argument("--model", default="yolov8s")
    ap.add_argument("--models", default=None, help="comma-separated models, one per stream")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--server", default="localhost:8001")
    ap.add_argument("--mode", default="rtsp", choices=["rtsp", "file"])
    ap.add_argument("--file", default="frames.bin")
    ap.add_argument("--cuda-shm", action="store_true", help="use CUDA shared memory (file mode)")
    args = ap.parse_args()

    client = grpcclient.InferenceServerClient(url=args.server)
    if not client.is_server_live():
        print("Triton not live"); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    urls = [args.url.replace("cam1", f"cam{i+1}") for i in range(args.streams)]
    models = args.models.split(",") if args.models else [args.model] * args.streams
    results = [None] * args.streams
    lock = threading.Lock()

    frames_np = None
    if args.mode == "file":
        frames_np = np.memmap(args.file, dtype=np.float32, mode="r")
        frames_np = frames_np.reshape(-1, 1, 3, IMG, IMG)

    threads = [threading.Thread(target=run_stream,
                                args=(args, urls[i], models[i], client, device, results, i, frames_np, lock))
               for i in range(args.streams)]
    for t in threads: t.start()
    for t in threads: t.join()

    all_lat_p50 = [r["lat_p50"] for r in results]
    all_lat_p95 = [r["lat_p95"] for r in results]
    total_frames = sum(r["frames"] for r in results)
    total_dets = sum(r["detections"] for r in results)
    print(json.dumps({
        "pipeline": "triton_python", "mode": args.mode, "streams": args.streams,
        "frames": total_frames, "detections": total_dets,
        "fps": total_frames / args.duration,
        "per_stream_fps": [round(r["fps"], 1) for r in results],
        "lat_ms_p50": round(max(all_lat_p50), 2),
        "lat_ms_p95": round(max(all_lat_p95), 2),
    }))


if __name__ == "__main__":
    main()