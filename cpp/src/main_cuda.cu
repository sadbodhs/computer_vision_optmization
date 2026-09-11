// trt_pipeline_cuda: fully-GPU pipeline (flow A-opt)
// NVDEC zero-copy (AV_PIX_FMT_CUDA frames stay on GPU)
// -> fused CUDA kernel: NV12 -> RGB letterbox(pad114)/255 -> CHW FP32
// -> TRT enqueue on same stream (input buffer written in-place, zero extra copies)
// -> GPU candidate compaction kernel -> CPU NMS on ~few hundred candidates
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
  }
} gLogger;

#define CUDA_CHECK(x)                                                          \
  do { cudaError_t e = (x); if (e != cudaSuccess) {                             \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __LINE__ << std::endl; exit(1); } \
  } while (0)

// ---------------------------------------------------------------------------
// Fused preprocess kernel: NV12 (GPU) -> letterboxed RGB CHW FP32 (GPU)
// src: Y plane (src_h x src_w) followed by interleaved UV (src_h/2 x src_w/2)
// dst: [1,3,640,640] FP32, centered letterbox, pad 114/255
// ---------------------------------------------------------------------------
__global__ void nv12_letterbox_kernel(
    const unsigned char* __restrict__ src_y,
    const unsigned char* __restrict__ src_uv,
    int y_pitch, int uv_pitch, int src_w, int src_h,
    float* __restrict__ dst,   // 3*640*640, CHW
    int nw, int nh, int pad_x, int pad_y, int IMG) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // dst x
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // dst y
  if (x >= IMG || y >= IMG) return;

  // centered letterbox
  int sx = x - pad_x, sy = y - pad_y;
  bool inside = (sx >= 0 && sx < nw && sy >= 0 && sy < nh);
  float r, g, b;
  if (!inside) {
    r = g = b = 114.0f / 255.0f;
  } else {
    // Bilinear luma, nearest chroma. Center-aligned sampling, matching
    // cv2 INTER_LINEAR: src = (dst + 0.5) * scale - 0.5. Nearest-neighbour here
    // measured -1.25% mAP50-95 against the reference (see docs/accuracy.md);
    // chroma stays nearest because NV12 already subsamples it 2x.
    float fx = ((float)sx + 0.5f) * (float)src_w / (float)nw - 0.5f;
    float fy = ((float)sy + 0.5f) * (float)src_h / (float)nh - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - (float)x0, ay = fy - (float)y0;
    int x0c = min(max(x0, 0), src_w - 1), x1c = min(max(x0 + 1, 0), src_w - 1);
    int y0c = min(max(y0, 0), src_h - 1), y1c = min(max(y0 + 1, 0), src_h - 1);
    float Y00 = (float)src_y[(size_t)y0c * y_pitch + x0c];
    float Y01 = (float)src_y[(size_t)y0c * y_pitch + x1c];
    float Y10 = (float)src_y[(size_t)y1c * y_pitch + x0c];
    float Y11 = (float)src_y[(size_t)y1c * y_pitch + x1c];
    float Yb = (Y00 * (1.0f - ax) + Y01 * ax) * (1.0f - ay)
             + (Y10 * (1.0f - ax) + Y11 * ax) * ay;
    int u = min(max((int)(fx + 0.5f), 0), src_w - 1);
    int v = min(max((int)(fy + 0.5f), 0), src_h - 1);
    unsigned char U = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1)];
    unsigned char V = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1) + 1];
    float yf = Yb - 16.0f, uf = (float)U - 128.0f, vf = (float)V - 128.0f;
    r = fmaxf(0.f, fminf(1.164f * yf + 1.596f * vf, 255.f)) / 255.0f;
    g = fmaxf(0.f, fminf(1.164f * yf - 0.392f * uf - 0.813f * vf, 255.f)) / 255.0f;
    b = fmaxf(0.f, fminf(1.164f * yf + 2.017f * uf, 255.f)) / 255.0f;
  }
  size_t plane = (size_t)IMG * IMG;
  dst[0 * plane + y * IMG + x] = r;
  dst[1 * plane + y * IMG + x] = g;
  dst[2 * plane + y * IMG + x] = b;
}

// ---------------------------------------------------------------------------
// GPU candidate compaction: out is [84,8400] -> produce compact list of
// (cx,cy,w,h,score,cls) for anchors passing conf; count written to d_count.
// Removes the 2.8MB D2H copy + 8400*80 CPU scan from the critical path.
// ---------------------------------------------------------------------------
struct GPUDet { float x1, y1, x2, y2, score; int cls; };

__global__ void compact_candidates_kernel(
    const float* __restrict__ out,   // [84,8400] channel-major
    int num_classes, int num_anchors, float conf_thr,
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
  if (slot < max_dets) {
    dets[slot] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, best_c};
  }
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

static ICudaEngine* loadEngine(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) { std::cerr << "cannot open engine " << path << std::endl; return nullptr; }
  f.seekg(0, std::ios::end); size_t size = f.tellg(); f.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  f.read(data.data(), size);
  IRuntime* runtime = createInferRuntime(gLogger);
  return runtime->deserializeCudaEngine(data.data(), size);
}

struct StageTimes {  // milliseconds, per-thread accumulation
  double h2d = 0, infer = 0, compact = 0, nms = 0;
  double decode = 0, preprocess = 0;
  long n = 0;
  std::mutex mtx;
  void add(const StageTimes& o) {
    h2d += o.h2d; infer += o.infer; compact += o.compact; nms += o.nms;
    decode += o.decode; preprocess += o.preprocess; n += o.n;
  }
};

static StageTimes g_stages;

struct StreamCtx {
  std::string url;
  ICudaEngine* engine;
  int stream_id;
  std::atomic<long>* frame_count;
  std::atomic<long>* det_count;
  std::atomic<bool>* running;
  double duration;
  std::string mode = "rtsp";
  std::string file_path;
  std::vector<double>* latencies;
  std::mutex* lat_mtx;
  bool use_graph = false;
};

// Reported in the JSON so a run can never be mislabelled: capture is allowed to
// fail at runtime (TensorRT does not guarantee every engine is capturable), and
// this records what actually executed, not what was asked for.
static std::atomic<bool> g_graph_active{false};

static void runStream(StreamCtx* ctx) {
  const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400;
  const int MAX_DETS = 4096;

  IExecutionContext* exec = ctx->engine->createExecutionContext();
  const char* in_name = ctx->engine->getIOTensorName(0);
  const char* out_name = ctx->engine->getIOTensorName(1);
  auto in_dims = ctx->engine->getTensorShape(in_name);
  auto out_dims = ctx->engine->getTensorShape(out_name);
  size_t in_size = 1, out_size = 1;
  for (int i = 0; i < in_dims.nbDims; ++i) in_size *= in_dims.d[i];
  for (int i = 0; i < out_dims.nbDims; ++i) out_size *= out_dims.d[i];
  in_size *= sizeof(float); out_size *= sizeof(float);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, in_size));
  CUDA_CHECK(cudaMalloc(&d_out, out_size));
  GPUDet* d_dets; int* d_count;
  CUDA_CHECK(cudaMalloc(&d_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  exec->setTensorAddress(in_name, d_in);
  exec->setTensorAddress(out_name, d_out);
  GPUDet* h_dets; int* h_count;
  CUDA_CHECK(cudaMallocHost((void**)&h_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMallocHost((void**)&h_count, sizeof(int)));

  if (ctx->mode == "file") {
    std::ifstream f(ctx->file_path, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot open " << ctx->file_path << std::endl; return; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    size_t n_frames = sz / in_size;
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    long fi = 0;
    bool use_graph = ctx->use_graph;
    bool graph_ready = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      auto ts = std::chrono::steady_clock::now();
      StageTimes st;
      CUDA_CHECK(cudaMemcpyAsync(d_in, buf.data() + fi * in_size, in_size, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      st.h2d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      auto t1 = std::chrono::steady_clock::now();
      // The captured region is exactly the st.infer window, so the graph's
      // effect lands in a stage the report already breaks out. The input H2D
      // stays outside it: a graph freezes every argument at capture time and
      // that copy's source pointer walks through the frame buffer.
      if (use_graph) {
        if (!graph_ready) {
          // TensorRT does lazy per-context setup on the first enqueue; doing it
          // inside the capture would record that one-off work into every replay.
          exec->enqueueV3(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          cudaError_t cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
          if (cap == cudaSuccess) {
            exec->enqueueV3(stream);
            cudaMemsetAsync(d_count, 0, sizeof(int), stream);
            compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
                d_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
            cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cap = cudaStreamEndCapture(stream, &graph);
          }
          if (cap == cudaSuccess) {
            cap = cudaGraphInstantiateWithFlags(&graph_exec, graph, 0);
          }
          if (cap == cudaSuccess) {
            graph_ready = true;
            g_graph_active.store(true);
          } else {
            // Not fatal, and not silently ignored either: fall back to the
            // normal path and say so, so a fallback run is never read as a
            // graph run.
            std::cerr << "cuda graph capture failed: " << cudaGetErrorString(cap)
                      << " - falling back to plain launches" << std::endl;
            cudaGetLastError();
            use_graph = false;
          }
        }
        if (graph_ready) {
          CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      if (!use_graph) {
        exec->enqueueV3(stream);
        // GPU compact + copy only compacted dets (tiny) to host
        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
        compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
            d_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
        CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      st.infer = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
      auto t2 = std::chrono::steady_clock::now();
      int n = std::min(*h_count, MAX_DETS);
      CUDA_CHECK(cudaMemcpy(h_dets, d_dets, n * sizeof(GPUDet), cudaMemcpyDeviceToHost));
      int kept = nms_count(h_dets, n, 0.45f);
      st.compact = 0;  // compact ran on GPU inside t1 window
      st.nms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
      st.n = 1;
      double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      { std::lock_guard<std::mutex> lk(g_stages.mtx); g_stages.add(st); }
      ctx->det_count->fetch_add(kept);
      ctx->frame_count->fetch_add(1);
      fi = (fi + 1) % n_frames;
    }
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
  } else {
    // ---- RTSP with NVDEC zero-copy (frames stay on GPU) ----
    // 1) hw device ctx for decoder
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
    // generic h264 decoder + CUDA hwaccel => AV_PIX_FMT_CUDA frames (zero-copy GPU)
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
    int src_w = cctx->coded_width > 0 ? cctx->coded_width : 640;
    int src_h = cctx->coded_height > 0 ? cctx->coded_height : 360;
    if (cctx->width > 0) { src_w = cctx->width; src_h = cctx->height; }
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
      auto td = std::chrono::steady_clock::now();
      if (av_read_frame(fmt, pkt) < 0) { av_seek_frame(fmt, -1, 0, AVSEEK_FLAG_BACKWARD); continue; }
      if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
      if (avcodec_send_packet(cctx, pkt) < 0) { av_packet_unref(pkt); continue; }
      av_packet_unref(pkt);
      if (avcodec_receive_frame(cctx, frame) < 0) continue;
      double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
      if (frame->format != AV_PIX_FMT_CUDA) {
        // decoder gave us CPU frames (hwaccel not active) — skip
        av_frame_unref(frame); continue;
      }

      // zero-copy: CUDA hw frame => data[0]=Y plane (GPU), data[1]=UV plane (GPU)
      unsigned char* src_y = (unsigned char*)frame->data[0];
      unsigned char* src_uv = (unsigned char*)frame->data[1];
      int y_pitch = frame->linesize[0], uv_pitch = frame->linesize[1];
      if (!src_y || !src_uv) { av_frame_unref(frame); continue; }

      auto ts = std::chrono::steady_clock::now();
      StageTimes st;
      // fused preprocess writes straight into TRT input buffer
      nv12_letterbox_kernel<<<grd, blk, 0, stream>>>(
          src_y, src_uv, y_pitch, uv_pitch, src_w, src_h, d_in, nw, nh, pad_x, pad_y, IMG);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      st.decode = dec_ms;
      st.preprocess = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      auto t1 = std::chrono::steady_clock::now();
      exec->enqueueV3(stream);
      CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
      compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
          d_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
      CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      int n = std::min(*h_count, MAX_DETS);
      CUDA_CHECK(cudaMemcpy(h_dets, d_dets, n * sizeof(GPUDet), cudaMemcpyDeviceToHost));
      st.infer = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
      auto t2 = std::chrono::steady_clock::now();
      int kept = nms_count(h_dets, n, 0.45f);
      st.nms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
      st.n = 1;
      double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      { std::lock_guard<std::mutex> lk(g_stages.mtx); g_stages.add(st); }
      ctx->det_count->fetch_add(kept);
      ctx->frame_count->fetch_add(1);
      av_frame_unref(frame);
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fmt);
    if (hw_ctx) av_buffer_unref(&hw_ctx);
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(h_dets); cudaFreeHost(h_count);
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_dets); cudaFree(d_count);
}

int main(int argc, char** argv) {
  std::string engine_path = "model.plan", url = "rtsp://localhost:8554/cam1";
  std::string mode = "rtsp", file_path = "frames.bin";
  int streams = 1;
  double duration = 30.0;
  bool use_graph = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
    else if (a == "--url" && i + 1 < argc) url = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
    else if (a == "--cuda-graph") use_graph = true;
  }
  ICudaEngine* engine = loadEngine(engine_path);
  if (!engine) return 1;
  std::atomic<long> frames{0}, dets{0};
  std::atomic<bool> running{true};
  std::vector<double> latencies; std::mutex lat_mtx;
  std::vector<std::thread> threads;
  std::vector<StreamCtx> ctxs(streams);
  for (int i = 0; i < streams; ++i) {
    ctxs[i] = {url, engine, i, &frames, &dets, &running, duration, mode, file_path, &latencies, &lat_mtx, use_graph};
    threads.emplace_back(runStream, &ctxs[i]);
  }
  for (auto& t : threads) t.join();
  double dt = duration;
  long n = frames.load();
  std::sort(latencies.begin(), latencies.end());
  auto pct = [&](double p) -> double {
    if (latencies.empty()) return 0;
    return latencies[std::min((size_t)(p * latencies.size()), latencies.size() - 1)];
  };
  double sn = g_stages.n ? g_stages.n : 1;
  std::cout << "{\"pipeline\":\"cpp_trt_cuda\",\"mode\":\"" << mode << "\",\"streams\":" << streams
            << ",\"frames\":" << n << ",\"detections\":" << dets.load() << ",\"fps\":" << (n / dt)
            << ",\"lat_ms_p50\":" << pct(0.50) << ",\"lat_ms_p95\":" << pct(0.95)
            << ",\"lat_ms_p99\":" << pct(0.99)
            << ",\"cuda_graph\":" << (g_graph_active.load() ? "true" : "false")
            << ",\"stages_ms\":{\"decode\":" << g_stages.decode / sn
            << ",\"preprocess\":" << g_stages.preprocess / sn
            << ",\"h2d\":" << g_stages.h2d / sn
            << ",\"infer_incl_compact\":" << g_stages.infer / sn
            << ",\"nms_cpu\":" << g_stages.nms / sn << "}}" << std::endl;
  return 0;
}