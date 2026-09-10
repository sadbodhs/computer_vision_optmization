#!/usr/bin/env python3
"""Measure what the OUTPUT tensor costs on the wire.

Same input, same weights, same server - the only difference is how much comes
back. The stock model returns [1,84,8400] FP32 (2.82 MB per frame); the in-graph
NMS build returns [1,300,6] (7.2 KB), 392x smaller.

Deliberately does no postprocessing: this measures round-trip only, so the
difference is transport plus whatever the in-graph NMS costs on the GPU.

Usage: python3 probe_transport.py --models yolov8s,yolov8s_nms --duration 6
"""
import argparse
import json
import statistics
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yolov8s,yolov8s_nms")
    ap.add_argument("--server", default="localhost:8001")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    import tritonclient.grpc as grpcclient
    client = grpcclient.InferenceServerClient(url=args.server)
    frame = np.random.rand(1, 3, 640, 640).astype(np.float32)

    for model in args.models.split(","):
        meta = client.get_model_metadata(model, as_json=True)
        out_name = meta["outputs"][0]["name"]
        out_shape = [int(d) for d in meta["outputs"][0]["shape"]]
        out_bytes = int(np.prod(out_shape)) * 4

        inp = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")]
        inp[0].set_data_from_numpy(frame)
        outs = [grpcclient.InferRequestedOutput(out_name)]

        for _ in range(args.warmup):
            client.infer(model, inp, outputs=outs)

        lat, n, t0 = [], 0, time.perf_counter()
        while time.perf_counter() - t0 < args.duration:
            ts = time.perf_counter()
            client.infer(model, inp, outputs=outs)
            lat.append((time.perf_counter() - ts) * 1000)
            n += 1
        dt = time.perf_counter() - t0

        print(json.dumps({
            "model": model,
            "output_shape": out_shape,
            "output_bytes": out_bytes,
            "fps": round(n / dt, 1),
            "lat_p50_ms": round(statistics.median(lat), 3),
            "lat_p95_ms": round(sorted(lat)[int(len(lat) * 0.95)], 3),
            "requests": n,
        }))


if __name__ == "__main__":
    main()
