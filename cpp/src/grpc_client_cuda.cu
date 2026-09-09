// grpc_client_cuda.cu — Flow B2: C++ gRPC -> Triton with CUDA shared memory
//
// Memory chain (zero-copy):
//   NVDEC AV_PIX_FMT_CUDA frames (GPU)
//   -> fused kernel writes letterboxed tensor DIRECTLY into a cudaIpc-registered
//      region -> Triton reads it (server-side H2D eliminated)
//   -> server writes result into cudaIpc output region
//   -> compact kernel reads output on GPU, only candidate boxes cross to host
//
// Modes: rtsp (live) | file (capacity replay of preprocessed frames)
#include <cuda_runtime_api.h>
#define TRITON_ENABLE_GPU 1
#include <grpc_client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
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
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " @" << __LINE__ << std::endl; exit(1); } \
  } while (0)
#define CHECK_OK(x)                                                            \
  do { tc::Error err = (x); if (!err.IsOk()) {                                  \
    std::cerr << "client error: " << err << std::endl; exit(1); } } while (0)

// ---------------- kernels (shared with flow A2) ----------------
__global__ void nv12_letterbox_kernel(
    const unsigned char* __restrict__ src_y,
    const unsigned char* __restrict__ src_uv,
    int y_pitch, int uv_pitch, int src_w, int src_h,
    float* __restrict__ dst, int nw, int nh, int pad_x, int pad_y, int IMG) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= IMG || y >= IMG) return;
  int sx = x - pad_x, sy = y - pad_y;
  bool inside = (sx >= 0 && sx < nw && sy >= 0 && sy < nh);
  float r, g, b;
  if (!inside) { r = g = b = 114.0f / 255.0f; }
  else {
    int u = sx * src_w / nw, v = sy * src_h / nh;
    unsigned char Y = src_y[(size_t)v * y_pitch + u];
    unsigned char U = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1)];
    unsigned char V = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1) + 1];
    float yf = (float)Y - 16.0f, uf = (float)U - 128.0f, vf = (float)V - 128.0f;
    r = fmaxf(0.f, fminf(1.164f * yf + 1.596f * vf, 255.f)) / 255.0f;
    g = fmaxf(0.f, fminf(1.164f * yf - 0.392f * uf - 0.813f * vf, 255.f)) / 255.0f;
    b = fmaxf(0.f, fminf(1.164f * yf + 2.017f * uf, 255.f)) / 255.0f;
  }
  size_t plane = (size_t)IMG * IMG;
  dst[0 * plane + y * IMG + x] = r;
  dst[1 * plane + y * IMG + x] = g;
  dst[2 * plane + y * IMG + x] = b;
}

struct GPUDet { float x1, y1, x2, y2, score; int cls; };

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

struct StreamCtx {
  std::string url, model, mode, file_path;
  int stream_id, streams;
  std::atomic<long>* frames;
  std::atomic<long>* dets;
  std::vector<double>* latencies;
  std::mutex* lat_mtx;
  double duration;
};

static void run_stream(StreamCtx* ctx) {
  const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400, MAX_DETS = 4096;
  const size_t IN_BYTES = (size_t)IMG * IMG * 3 * sizeof(float);
  const size_t OUT_BYTES = (size_t)84 * NUM_ANCHORS * sizeof(float);

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  CHECK_OK(tc::InferenceServerGrpcClient::Create(&client, "localhost:8001", false));

  // ---- register CUDA shm regions (input written by our kernel, output by server) ----
  float* shm_in = nullptr; float* shm_out = nullptr;
  CUDA_CHECK(cudaMalloc(&shm_in, IN_BYTES));
  CUDA_CHECK(cudaMalloc(&shm_out, OUT_BYTES));
  cudaIpcMemHandle_t in_handle, out_handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&in_handle, shm_in));
  CUDA_CHECK(cudaIpcGetMemHandle(&out_handle, shm_out));
  std::string in_region = "in_" + std::to_string(ctx->stream_id) + "_" + std::to_string(getpid());
  std::string out_region = "out_" + std::to_string(ctx->stream_id) + "_" + std::to_string(getpid());
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

  // output-side compact scratch
  GPUDet* d_dets; int* d_count;
  CUDA_CHECK(cudaMalloc(&d_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  GPUDet* h_dets; int* h_count;
  CUDA_CHECK(cudaMallocHost((void**)&h_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMallocHost((void**)&h_count, sizeof(int)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  tc::InferOptions options(ctx->model);
  tc::InferResult* res = nullptr;

  auto do_infer_and_post = [&](long& fi_count) -> double {
    auto ts = std::chrono::steady_clock::now();
    CHECK_OK(client->Infer(&res, options, inputs, outputs));
    // compact candidates on GPU reading the shm output region, then tiny D2H
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
        shm_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int n = std::min(*h_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpy(h_dets, d_dets, n * sizeof(GPUDet), cudaMemcpyDeviceToHost));
    double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
    ctx->dets->fetch_add(nms_count(h_dets, n, 0.45f));
    ctx->frames->fetch_add(1);
    fi_count++;
    delete res; res = nullptr;
    return lat;
  };

  if (ctx->mode == "file") {
    std::ifstream f(ctx->file_path, std::ios::binary);
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    size_t n_frames = sz / IN_BYTES;
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    long fi = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      CUDA_CHECK(cudaMemcpyAsync(shm_in, buf.data() + fi * IN_BYTES, IN_BYTES, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      double lat = do_infer_and_post(fi);
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      fi = (fi + 1) % n_frames;
    }
  } else {
    // ---- RTSP: NVDEC zero-copy -> kernel writes into shm_in ----
    AVBufferRef* hw_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
      std::cerr << "cannot create CUDA hw device ctx" << std::endl; return;
    }
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, ctx->url.c_str(), nullptr, nullptr) < 0) {
      std::cerr << "stream " << ctx->stream_id << ": open failed" << std::endl; return;
    }
    avformat_find_stream_info(fmt, nullptr);
    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const AVCodec* dec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* cctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(cctx, fmt->streams[vs]->codecpar);
    cctx->hw_device_ctx = av_buffer_ref(hw_ctx);
    cctx->get_format = [](AVCodecContext*, const enum AVPixelFormat* fmts) -> enum AVPixelFormat {
      for (const enum AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == AV_PIX_FMT_CUDA) return AV_PIX_FMT_CUDA;
      return fmts[0];
    };
    if (avcodec_open2(cctx, dec, nullptr) < 0) {
      std::cerr << "stream " << ctx->stream_id << ": codec open failed" << std::endl; return;
    }
    int src_w = cctx->width, src_h = cctx->height;
    float scale = std::min((float)IMG / src_w, (float)IMG / src_h);
    int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
    int pad_x = (IMG - nw) / 2, pad_y = (IMG - nh) / 2;

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    dim3 blk(32, 16);
    dim3 grd((IMG + blk.x - 1) / blk.x, (IMG + blk.y - 1) / blk.y);

    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      if (av_read_frame(fmt, pkt) < 0) { av_seek_frame(fmt, -1, 0, AVSEEK_FLAG_BACKWARD); continue; }
      if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
      if (avcodec_send_packet(cctx, pkt) < 0) { av_packet_unref(pkt); continue; }
      av_packet_unref(pkt);
      if (avcodec_receive_frame(cctx, frame) < 0) continue;
      if (frame->format != AV_PIX_FMT_CUDA) { av_frame_unref(frame); continue; }

      unsigned char* src_y = (unsigned char*)frame->data[0];
      unsigned char* src_uv = (unsigned char*)frame->data[1];
      int y_pitch = frame->linesize[0], uv_pitch = frame->linesize[1];
      if (!src_y || !src_uv) { av_frame_unref(frame); continue; }

      nv12_letterbox_kernel<<<grd, blk, 0, stream>>>(
          src_y, src_uv, y_pitch, uv_pitch, src_w, src_h, shm_in, nw, nh, pad_x, pad_y, IMG);
      CUDA_CHECK(cudaStreamSynchronize(stream));  // server reads shm_in only after kernel done
      long fi = 0;
      double lat = do_infer_and_post(fi);
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      av_frame_unref(frame);
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fmt);
    if (hw_ctx) av_buffer_unref(&hw_ctx);
  }

  // keep regions registered for process lifetime (server may hold IPC mapping)
  cudaStreamDestroy(stream);
  cudaFreeHost(h_dets); cudaFreeHost(h_count);
  cudaFree(d_dets); cudaFree(d_count);
}

int main(int argc, char** argv) {
  std::string url = "rtsp://localhost:8554/cam1", model = "yolov8s";
  std::string mode = "rtsp", file_path = "frames.bin";
  int streams = 1;
  double duration = 15.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--url" && i + 1 < argc) url = argv[++i];
    else if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
  }

  std::atomic<long> frames{0}, dets{0};
  std::vector<double> latencies; std::mutex lat_mtx;
  std::vector<std::thread> threads;
  std::vector<StreamCtx> ctxs(streams);
  for (int i = 0; i < streams; ++i) {
    ctxs[i] = {url, model, mode, file_path, i, streams, &frames, &dets, &latencies, &lat_mtx, duration};
    threads.emplace_back(run_stream, &ctxs[i]);
  }
  for (auto& t : threads) t.join();

  long n = frames.load();
  std::sort(latencies.begin(), latencies.end());
  auto pct = [&](double p) -> double {
    if (latencies.empty()) return 0;
    return latencies[std::min((size_t)(p * latencies.size()), latencies.size() - 1)];
  };
  printf("{\"pipeline\":\"cpp_grpc_cuda_shm\",\"mode\":\"%s\",\"model\":\"%s\",\"streams\":%d,"
         "\"frames\":%ld,\"detections\":%ld,\"fps\":%.2f,"
         "\"lat_ms_p50\":%.3f,\"lat_ms_p95\":%.3f,\"lat_ms_p99\":%.3f}\n",
         mode.c_str(), model.c_str(), streams, n, dets.load(), n / duration,
         pct(0.50), pct(0.95), pct(0.99));
  return 0;
}