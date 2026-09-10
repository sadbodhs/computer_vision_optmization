#!/usr/bin/env python3
"""Flow C2: pure-numpy preprocess (no torch) + transfer variants.
--transfer raw | sys | cuda
  raw : gRPC payload (baseline)
  sys : system shared memory (client memcpy into shm; server reads directly)
  cuda: CUDA shared memory (client-side H2D into IPC region; server reads GPU)
Postprocess: pure numpy NMS (no torch).
Use --processes to split streams across processes (bypass GIL).
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time

import numpy as np

IMG = 640
NUM_ANCHORS = 8400
IN_BYTES = 3 * IMG * IMG * 4
OUT_BYTES = 84 * NUM_ANCHORS * 4


def letterbox_nv12_np(buf: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """NV12 (h*3/2, w) uint8 -> [1,3,640,640] float32, centered letterbox pad 114, /255. Pure numpy."""
    y = buf[:src_w * src_h].reshape(src_h, src_w).astype(np.float32)
    uv = buf[src_w * src_h:].reshape(src_h // 2, src_w // 2, 2)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1).astype(np.float32)
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1).astype(np.float32)
    yf, uf, vf = y - 16.0, u - 128.0, v - 128.0
    r = np.clip(1.164 * yf + 1.596 * vf, 0, 255)
    g = np.clip(1.164 * yf - 0.392 * uf - 0.813 * vf, 0, 255)
    b = np.clip(1.164 * yf + 2.017 * uf, 0, 255)
    rgb = np.empty((3, src_h, src_w), dtype=np.float32)
    rgb[0], rgb[1], rgb[2] = r, g, b
    rgb /= 255.0
    scale = min(IMG / src_w, IMG / src_h)
    nw, nh = int(src_w * scale), int(src_h * scale)
    if (nw, nh) != (src_w, src_h):  # nearest resize only if needed
        rows = (np.arange(nh) * src_h / nh).astype(int)
        cols = (np.arange(nw) * src_w / nw).astype(int)
        rgb = rgb[:, rows][:, :, cols]
    canvas = np.full((3, IMG, IMG), 114.0 / 255.0, dtype=np.float32)
    py, px = (IMG - nh) // 2, (IMG - nw) // 2
    canvas[:, py:py + nh, px:px + nw] = rgb
    return canvas[None]  # 1,3,640,640


def postprocess_np(out: np.ndarray, conf_thr=0.25, iou_thr=0.45) -> int:
    o = out[0]  # 84,8400
    boxes = o[:4, :]
    scores = o[4:, :]
    cls = scores.argmax(axis=0)
    mx = scores.max(axis=0)
    mask = mx > conf_thr
    if not mask.any():
        return 0
    idx = np.nonzero(mask)[0]
    cx, cy, w, h = boxes[0, idx], boxes[1, idx], boxes[2, idx], boxes[3, idx]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    sc = mx[idx].astype(np.float64)
    cl = cls[idx]
    order = sc.argsort()[::-1]
    keep = 0
    removed = np.zeros(len(order), dtype=bool)
    xa = x1[order]; ya = y1[order]; xb = x2[order]; yb = y2[order]
    ca = cl[order]
    for ii in range(len(order)):
        if removed[ii]:
            continue
        keep += 1
        ix1 = np.maximum(xa[ii], xa[ii + 1:]); iy1 = np.maximum(ya[ii], ya[ii + 1:])
        ix2 = np.minimum(xb[ii], xb[ii + 1:]); iy2 = np.minimum(yb[ii], yb[ii + 1:])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        uni = (xb[ii] - xa[ii]) * (yb[ii] - ya[ii]) + (xb[ii + 1:] - xa[ii + 1:]) * (yb[ii + 1:] - ya[ii + 1:]) - inter
        iou = inter / np.maximum(uni, 1e-6)
        same = ca[ii + 1:] == ca[ii]
        removed[ii + 1:][(iou > iou_thr) & same] = True
    return keep


def get_resolution(url):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", url]
    w, h = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().split(",")
    return int(w), int(h)


def decode_stream(url, src_w, src_h):
    cmd = ["ffmpeg", "-loglevel", "error", "-hwaccel", "cuda", "-c:v", "h264_cuvid",
           "-i", url, "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs = src_w * src_h * 3 // 2
    while True:
        b = proc.stdout.read(fs)
        if len(b) < fs:
            break
        yield b


def worker(rank, args, urls, models, q):
    import os
    import tritonclient.grpc as grpcclient
    client = grpcclient.InferenceServerClient(url=args.server)
    results = [None] * len(urls)
    frames_np = None
    if args.mode == "file":
        frames_np = np.memmap(args.file, dtype=np.float32, mode="r").reshape(-1, 1, 3, IMG, IMG)

    # unique region names per process instance (parallel instances share the server)
    uid = f"{rank}_{os.getpid()}"
    run_tag = str(int(time.time() * 1000) % 100000000)  # unique per run; avoids stale-registry collisions
    shm_in_h = shm_out_h = None
    shm_mod = None
    for k, (url, model) in enumerate(zip(urls, models)):
        idx = rank * len(urls) + k
        lat, frames, dets = [], 0, 0
        t0 = time.perf_counter()

        if args.transfer == "sys":
            import tritonclient.utils.shared_memory as shm
            try:  # clear stale regions from previous runs (server keeps registry)
                client.unregister_system_shared_memory(f"pin_{run_tag}_{uid}")
                client.unregister_system_shared_memory(f"pout_{run_tag}_{uid}")
            except Exception:
                pass
            shm_in_h = shm.create_shared_memory_region(f"pin_{run_tag}_{uid}", f"/pin_{idx}", IN_BYTES)
            shm_out_h = shm.create_shared_memory_region(f"pout_{run_tag}_{uid}", f"/pout_{idx}", OUT_BYTES)
            client.register_system_shared_memory(f"pin_{run_tag}_{uid}", f"/pin_{idx}", IN_BYTES)
            client.register_system_shared_memory(f"pout_{run_tag}_{uid}", f"/pout_{idx}", OUT_BYTES)

        if args.mode == "file":
            n = frames_np.shape[0]
            fi = 0
            while time.perf_counter() - t0 < args.duration:
                ts = time.perf_counter()
                if args.transfer == "raw":
                    inp = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
                    inp[0].set_data_from_numpy(frames_np[fi].copy())
                    out_spec = [grpcclient.InferRequestedOutput("output0")]
                    res = client.infer(model, inp, outputs=out_spec)
                elif args.transfer == "sys":
                    shm.set_shared_memory_region(shm_in_h, [frames_np[fi]])
                    inp = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
                    inp[0].set_shared_memory(f"pin_{run_tag}_{uid}", IN_BYTES, 0)
                    out_spec = [grpcclient.InferRequestedOutput("output0")]
                    out_spec[0].set_shared_memory(f"pout_{run_tag}_{uid}", OUT_BYTES, 0)
                    res = client.infer(model, inp, outputs=out_spec)
                    out = shm.get_contents_as_numpy(shm_out_h, np.float32, [1, 84, NUM_ANCHORS])
                elif args.transfer == "cuda":
                    import tritonclient.utils.cuda_shared_memory as csm
                    if shm_in_h is None:
                        try:  # clear stale regions from previous runs
                            client.unregister_cuda_shared_memory(f"cin_{run_tag}_{uid}")
                            client.unregister_cuda_shared_memory(f"cout_{run_tag}_{uid}")
                        except Exception:
                            pass
                        shm_in_h = csm.create_shared_memory_region(f"cin_{run_tag}_{uid}", IN_BYTES, 0)
                        shm_out_h = csm.create_shared_memory_region(f"cout_{run_tag}_{uid}", OUT_BYTES, 0)
                        client.register_cuda_shared_memory(f"cin_{run_tag}_{uid}", csm.get_raw_handle(shm_in_h), 0, IN_BYTES)
                        client.register_cuda_shared_memory(f"cout_{run_tag}_{uid}", csm.get_raw_handle(shm_out_h), 0, OUT_BYTES)
                    csm.set_shared_memory_region(shm_in_h, [frames_np[fi]])
                    inp = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
                    inp[0].set_shared_memory(f"cin_{run_tag}_{uid}", IN_BYTES, 0)
                    out_spec = [grpcclient.InferRequestedOutput("output0")]
                    out_spec[0].set_shared_memory(f"cout_{run_tag}_{uid}", OUT_BYTES, 0)
                    res = client.infer(model, inp, outputs=out_spec)
                    out = csm.get_contents_as_numpy(shm_out_h, np.float32, [1, 84, NUM_ANCHORS])
                else:  # raw
                    out = res.as_numpy("output0")
                lat.append((time.perf_counter() - ts) * 1000)
                dets += postprocess_np(out)
                frames += 1
                fi = (fi + 1) % n
    else:
        src_w, src_h = get_resolution(url)
        dec_times, pre_times = [], []
        prev_t = 0.0
        for nv12 in decode_stream(url, src_w, src_h):
            if time.perf_counter() - t0 > args.duration:
                break
            t_dec_end = time.perf_counter()
            tensor = letterbox_nv12_np(np.frombuffer(nv12, dtype=np.uint8), src_w, src_h)
            t_pre_end = time.perf_counter()
            ts = time.perf_counter()
            inp = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
            inp[0].set_data_from_numpy(tensor)
            out_spec = [grpcclient.InferRequestedOutput("output0")]
            res = client.infer(model, inp, outputs=out_spec)
            out = res.as_numpy("output0")
            t_inf_end = time.perf_counter()
            lat.append((t_inf_end - ts) * 1000)
            dec_times.append((t_dec_end - prev_t) * 1000 if prev_t else 0)
            pre_times.append((t_pre_end - t_dec_end) * 1000)
            prev_t = t_dec_end
            dets_local = postprocess_np(out)
            dets += dets_local
            frames += 1
        dt = time.perf_counter() - t0
        la = np.array(lat)
        dta = np.array(dec_times); pta = np.array(pre_times)
        results[k] = {"frames": frames, "detections": dets, "fps": frames / dt,
                        "lat_p50": float(np.percentile(la, 50)) if len(la) else 0,
                        "lat_p95": float(np.percentile(la, 95)) if len(la) else 0,
                        "decode_ms": float(dta.mean()) if len(dta) else 0,
                        "preprocess_ms": float(pta.mean()) if len(pta) else 0}
    q.put(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtsp://localhost:8554/cam1")
    ap.add_argument("--model", default="yolov8s")
    ap.add_argument("--models", default=None)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--server", default="localhost:8001")
    ap.add_argument("--mode", default="rtsp", choices=["rtsp", "file"])
    ap.add_argument("--file", default="frames.bin")
    ap.add_argument("--transfer", default="raw", choices=["raw", "sys", "cuda"])
    ap.add_argument("--processes", type=int, default=1)
    args = ap.parse_args()

    urls = [args.url.replace("cam1", f"cam{i+1}") for i in range(args.streams)]
    models = args.models.split(",") if args.models else [args.model] * args.streams
    P = max(1, min(args.processes, args.streams))
    per = [urls[i::P] for i in range(P)]
    pmodels = [models[i::P] for i in range(P)]

    q = mp.Queue()
    procs = []
    for r in range(P):
        p = mp.Process(target=worker, args=(r, args, per[r], pmodels[r], q))
        p.start()
        procs.append(p)
    all_results = []
    for _ in range(P):
        all_results.extend(q.get())
    for p in procs:
        p.join()

    n = sum(r["frames"] for r in all_results)
    d = sum(r["detections"] for r in all_results)
    lat50 = max(r["lat_p50"] for r in all_results)
    lat95 = max(r["lat_p95"] for r in all_results)
    print(json.dumps({
        "pipeline": "python_numpy", "mode": args.mode, "transfer": args.transfer,
        "streams": args.streams, "processes": P,
        "frames": n, "detections": d, "fps": n / args.duration,
        "per_stream_fps": [round(r["fps"], 1) for r in all_results],
        "lat_ms_p50": round(lat50, 2), "lat_ms_p95": round(lat95, 2),
        "stages_ms": {
            "decode_pipe": round(max(r.get("decode_ms", 0) for r in all_results), 2),
            "preprocess_np": round(max(r.get("preprocess_ms", 0) for r in all_results), 2),
            "infer_grpc": round(lat50, 2),
        },
    }))


if __name__ == "__main__":
    main()