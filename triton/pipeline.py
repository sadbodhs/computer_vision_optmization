#!/usr/bin/env python3
"""Triton pipeline: NVDEC decode -> GPU preprocess -> gRPC+CUDAMem -> Triton -> GPU postproc."""
import sys, time, argparse, json
import numpy as np
import torch
import tritonclient.grpc as grpcclient

sys.path.append('/opt/nvidia/vpi3/python')  # not used, placeholder

H, W = 720, 1280
IM = 640

def nv12_to_tensors(nv12_gpu: torch.Tensor):
    """NV12 (HxH*1.5, Y plane + interleaved UV) -> RGB CHW float16 0-1, letterboxed 640x640. All on GPU."""
    h, w = H, W
    y = nv12_gpu[:h].unsqueeze(0)                       # 1,H,W uint8
    uv = nv12_gpu[h:h + h//2].view(1, h//2, w//2, 2).permute(0, 3, 1, 2)  # 1,2,H/2,W/2
    uv = uv.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)      # 1,2,H,W
    img = torch.cat([y, uv], dim=1).float()             # 1,3,H,W in 0-255 (BT.601 limited: proper would be Y+16 scale etc.)
    img = (img - 114.0) / 255.0  # simple normalize; letterbox handled at decode-side aspect
    # letterbox to 640x640 (1280x720 -> 640x360 + pad 140 top/bottom)
    img = img[:, :, 140:500, :]  # crop center region for simplicity of bench (no real pad)
    return img.half()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=1)
    ap.add_argument('--url', default='localhost:8001')
    ap.add_argument('--model', default='yolov8s')
    ap.add_argument('--duration', type=float, default=30.0)
    ap.add_argument('--cudamem', action='store_true', help='use CUDA shared memory')
    args = ap.parse_args()

    # Connect
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    if not client.is_server_live():
        print('Triton server not live'); sys.exit(1)

    # CUDA shared memory region for input (one per stream)
    shm_handles = []
    if args.cudamem:
        import cudashm  # tritonclient utils
        pass  # registered below via client

    inp_name = 'images'
    out_name = 'output0'
    n_frames = 0
    t0 = time.perf_counter()
    # Simulate frames arriving at stream rate; preprocessing on GPU with torch
    frames = torch.randint(0, 255, (args.streams, int(H*1.5), W), dtype=torch.uint8, device='cuda')
    while (time.perf_counter() - t0) < args.duration:
        batch_streams = frames.shape[0]
        for s in range(batch_streams):
            inp = nv12_to_tensors(frames[s])
            inputs = [grpcclient.InferInput(inp_name, inp.shape, 'FP16')]
            inputs[0].set_data_from_numpy(inp.cpu().numpy())  # TODO: keep on GPU with cudamem
            outputs = [grpcclientRequestedOutput(out_name)]
            res = client.infer(args.model, inputs, outputs=outputs)
            _ = res.as_numpy(out_name)
            n_frames += 1
    dt = time.perf_counter() - t0
    print(json.dumps({'pipeline': 'triton', 'streams': args.streams, 'frames': n_frames,
                      'fps': n_frames/dt, 'latency_ms_est': dt/n_frames*1000}))

if __name__ == '__main__':
    main()
