// grpc_async_client.cu — Flow D: async in-flight C++ gRPC client -> Triton dyn-batch model
// Each stream keeps `--depth` requests in flight; batches form in the server.
// Mode: file (capacity replay) only — async is about request rate, not decode pacing.
#include <cuda_runtime_api.h>
#define TRITON_ENABLE_GPU 1
#include <grpc_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tc = triton::client;

#define CUDA_CHECK(x)                                                          \
  do { cudaError_t e = (x); if (e != cudaSuccess) {                             \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " @" << __LINE__ << std::endl; exit(1); } } while (0)
#define CHECK_OK(x)                                                            \
  do { tc::Error err = (x); if (!err.IsOk()) {                                  \
    std::cerr << "client error: " << err << std::endl; exit(1); } } while (0)

struct GPUDet { float x1, y1, x2, y2, score; int cls; };

static int nms_count(const GPUDet* dets, int n, float iou_thr) {
  std::vector<GPUDet> v(dets, dets + n);
  std::sort(v.begin(), v.end(), [](const GPUDet& a, const GPUDet& b) { return a.score > b.score; });
  std::vector<bool> removed(v.size(), false);
  int kept = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (removed[i]) continue;
    kept++;
    for (size_t j = i + 1; j < v.size(); ++j) {
      if (removed[j] || v[i].cls != v[j].cls) continue;
      float ix1 = std::max(v[i].x1, v[j].x1), iy1 = std::max(v[i].y1, v[j].y1);
      float ix2 = std::min(v[i].x2, v[j].x2), iy2 = std::min(v[i].y2, v[j].y2);
      float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
      float uni = (v[i].x2 - v[i].x1) * (v[i].y2 - v[i].y1) +
                  (v[j].x2 - v[j].x1) * (v[j].y2 - v[j].y1) - inter;
      if (inter / std::max(uni, 1e-6f) > iou_thr) removed[j] = true;
    }
  }
  return kept;
}

// per-request context handed to the completion callback
struct ReqCtx {
  tc::InferResult* result = nullptr;
  tc::InferInput* input = nullptr;
  tc::InferRequestedOutput* output = nullptr;
  std::chrono::steady_clock::time_point send_ts;
  double lat_ms = 0;
  int det_count = 0;
  bool done = false;
};

struct Stats {
  std::atomic<long> frames{0}, dets{0};
  std::vector<double> latencies;
  std::mutex mtx;
};

static void run_stream(int stream_id, const std::string& model, const std::string& file_path,
                       double duration, int depth, Stats* stats) {
  const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400;
  const size_t IN_BYTES = (size_t)IMG * IMG * 3 * sizeof(float);
  const size_t OUT_BYTES = (size_t)84 * NUM_ANCHORS * sizeof(float);

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  CHECK_OK(tc::InferenceServerGrpcClient::Create(&client, "localhost:8001", false));

  // CUDA shm region per stream: input written via cudaMemcpy, output read after completion
  float* shm_in = nullptr; float* shm_out = nullptr;
  CUDA_CHECK(cudaMalloc(&shm_in, IN_BYTES));
  CUDA_CHECK(cudaMalloc(&shm_out, OUT_BYTES));
  cudaIpcMemHandle_t in_handle, out_handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&in_handle, shm_in));
  CUDA_CHECK(cudaIpcGetMemHandle(&out_handle, shm_out));
  std::string in_region = "ain_" + std::to_string(stream_id) + "_" + std::to_string(getpid());
  std::string out_region = "aout_" + std::to_string(stream_id) + "_" + std::to_string(getpid());
  CHECK_OK(client->RegisterCudaSharedMemory(in_region, in_handle, 0, IN_BYTES));
  CHECK_OK(client->RegisterCudaSharedMemory(out_region, out_handle, 0, OUT_BYTES));

  std::vector<int64_t> in_shape = {1, 3, IMG, IMG};
  tc::InferInput* inp = nullptr;
  CHECK_OK(tc::InferInput::Create(&inp, "images", in_shape, "FP32"));
  CHECK_OK(inp->SetSharedMemory(in_region, IN_BYTES, 0));
  tc::InferRequestedOutput* out = nullptr;
  CHECK_OK(tc::InferRequestedOutput::Create(&out, "output0"));
  CHECK_OK(out->SetSharedMemory(out_region, OUT_BYTES, 0));
  std::vector<tc::InferInput*> inputs = {inp};
  std::vector<const tc::InferRequestedOutput*> outputs = {out};

  // load frames
  std::ifstream f(file_path, std::ios::binary);
  f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
  size_t n_frames = sz / IN_BYTES;
  std::vector<char> buf(sz);
  f.read(buf.data(), sz);

  tc::InferOptions options(model);

  // in-flight bookkeeping
  std::mutex inflight_mtx;
  std::condition_variable inflight_cv;
  int inflight = 0;
  const int DEPTH = 8;

  std::atomic<long> sent{0};
  auto t0 = std::chrono::steady_clock::now();
  size_t fi = 0;
  long completed = 0;
  GPUDet* h_dets; int* h_count; int* d_count; GPUDet* d_dets;
  CUDA_CHECK(cudaMallocHost((void**)&h_dets, 4096 * sizeof(GPUDet)));
  CUDA_CHECK(cudaMallocHost((void**)&h_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dets, 4096 * sizeof(GPUDet)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  while (true) {
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (el > duration) break;

    // wait until we have in-flight budget
    std::unique_lock<std::mutex> lk(inflight_mtx);
    inflight_cv.wait(lk, [&] { return inflight < DEPTH; });
    lk.unlock();

    // write next frame into shm, then fire async request
    CUDA_CHECK(cudaMemcpyAsync(shm_in, buf.data() + fi * IN_BYTES, IN_BYTES, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fi = (fi + 1) % n_frames;

    ReqCtx* rc = new ReqCtx();
    rc->send_ts = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk2(inflight_mtx);
      inflight++;
    }
    CHECK_OK(client->AsyncInfer(
        [&, rc, stream, h_dets, h_count, d_count, d_dets](tc::InferResult* result) {
          auto ts = std::chrono::steady_clock::now();
          double lat = std::chrono::duration<double, std::milli>(ts - rc->send_ts).count();
          // compact candidates from shm output region
          cudaMemsetAsync(d_count, 0, sizeof(int), stream);
          compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
              shm_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, 4096);
          cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          int n = std::min(*h_count, 4096);
          cudaMemcpy(h_dets, d_dets, n * sizeof(GPUDet), cudaMemcpyDeviceToHost);
          {
            std::lock_guard<std::mutex> lk3(stats->mtx);
            stats->latencies.push_back(lat);
            stats->dets += nms_count(h_dets, n, 0.45f);
            stats->frames++;
          }
          delete rc;
          std::lock_guard<std::mutex> lk2(inflight_mtx);
          inflight--;
          inflight_cv.notify_one();
        },
        options, inputs, outputs));
    sent++;
  }

  // drain: wait for all in-flight to complete
  while (true) {
    std::unique_lock<std::mutex> lk(inflight_mtx);
    if (inflight == 0) break;
    inflight_cv.wait_for(lk, std::chrono::milliseconds(100));
  }
}

// compact kernel (same as other flows)
__global__ void compact_candidates_kernel(
    const float* __restrict__ out, int num_classes, int num_anchors, float conf_thr,
    GPUDet* __restrict__ dets, int* __restrict__ d_count, int max_dets) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  float best = 0.f; int best_c = -1;
  for (int c = 0; c < num_classes; ++c) {
    float s = out[(4 + c) * num_anchors + a];
    if (s > best) { best = s; best_c = c; }
  }
  if (best < conf_thr) return;
  float cx = out[0 * num_anchors + a], cy = out[1 * num_anchors + a];
  float w = out[2 * num_anchors + a], h = out[3 * num_anchors + a];
  int slot = atomicAdd(d_count, 1);
  if (slot < max_dets) dets[slot] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, best_c};
}

int main(int argc, char** argv) {
  std::string model = "yolov8s_dyn", file_path = "frames.bin";
  int streams = 1;
  double duration = 10.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
  }
  Stats stats;
  std::vector<std::thread> threads;
  for (int i = 0; i < streams; ++i)
    threads.emplace_back(run_stream, i, model, file_path, duration, &stats);
  for (auto& t : threads) t.join();

  long n = stats.frames.load();
  std::sort(stats.latencies.begin(), stats.latencies.end());
  auto pct = [&](double p) -> double {
    if (stats.latencies.empty()) return 0;
    return stats.latencies[std::min((size_t)(p * stats.latencies.size()), stats.latencies.size() - 1)];
  };
  printf("{\"pipeline\":\"cpp_grpc_async_dyn\",\"model\":\"%s\",\"streams\":%d,"
         "\"frames\":%ld,\"detections\":%ld,\"fps\":%.2f,"
         "\"lat_ms_p50\":%.3f,\"lat_ms_p95\":%.3f,\"lat_ms_p99\":%.3f}\n",
         model.c_str(), streams, n, stats.dets.load(), n / duration,
         pct(0.50), pct(0.95), pct(0.99));
  return 0;
}