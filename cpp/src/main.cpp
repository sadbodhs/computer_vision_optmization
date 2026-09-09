#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace nvinfer1;

// ---- TensorRT logger ----
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
  }
} gLogger;

// ---- YOLO postprocess ----
struct Box { float x1, y1, x2, y2, score; int cls; };

static float iou(const Box& a, const Box& b) {
  float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
  float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
  float inter = iw * ih;
  float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0 ? inter / uni : 0.f;
}

static std::vector<Box> postprocess(const float* out, int num_classes, int num_anchors,
                                    float conf_thr, float iou_thr, int img_w, int img_h) {
  std::vector<Box> cand;
  for (int a = 0; a < num_anchors; ++a) {
    // output is [1, 4+num_classes, num_anchors] channel-major
    float cx = out[0 * num_anchors + a];
    float cy = out[1 * num_anchors + a];
    float w  = out[2 * num_anchors + a];
    float h  = out[3 * num_anchors + a];
    float best = 0; int best_c = -1;
    for (int c = 0; c < num_classes; ++c) {
      float s = out[(4 + c) * num_anchors + a];  // already sigmoid'd by export
      if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thr) continue;
    Box b;
    b.x1 = cx - w / 2; b.y1 = cy - h / 2; b.x2 = cx + w / 2; b.y2 = cy + h / 2;
    b.score = best; b.cls = best_c;
    cand.push_back(b);
  }
  std::sort(cand.begin(), cand.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> kept;
  std::vector<bool> removed(cand.size(), false);
  for (size_t i = 0; i < cand.size(); ++i) {
    if (removed[i]) continue;
    kept.push_back(cand[i]);
    for (size_t j = i + 1; j < cand.size(); ++j) {
      if (!removed[j] && cand[i].cls == cand[j].cls && iou(cand[i], cand[j]) > iou_thr)
        removed[j] = true;
    }
  }
  return kept;
}

// ---- Engine load ----
static ICudaEngine* loadEngine(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) { std::cerr << "cannot open engine " << path << std::endl; return nullptr; }
  f.seekg(0, std::ios::end); size_t size = f.tellg(); f.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  f.read(data.data(), size);
  IRuntime* runtime = createInferRuntime(gLogger);
  return runtime->deserializeCudaEngine(data.data(), size);
}

// ---- Per-stream worker ----
struct StreamCtx {
  std::string url;
  ICudaEngine* engine;
  int stream_id;
  std::atomic<long>* frame_count;
  std::atomic<long>* det_count;
  std::atomic<bool>* running;
  double duration;
  std::string mode = "rtsp";       // "rtsp" | "file"
  std::string file_path;           // binary preprocessed frames for file mode
  std::vector<double>* latencies;  // per-frame inference latency (ms)
  std::mutex* lat_mtx;
};

static void runStream(StreamCtx* ctx) {
  const int IMG = 640;
  const int NUM_CLASSES = 80;
  const int NUM_ANCHORS = 8400;

  // TRT context + buffers
  IExecutionContext* exec = ctx->engine->createExecutionContext();
  const char* in_name = ctx->engine->getIOTensorName(0);
  const char* out_name = ctx->engine->getIOTensorName(1);
  auto in_dims = ctx->engine->getTensorShape(in_name);
  auto out_dims = ctx->engine->getTensorShape(out_name);
  size_t in_size = 1, out_size = 1;
  for (int i = 0; i < in_dims.nbDims; ++i) in_size *= in_dims.d[i];
  for (int i = 0; i < out_dims.nbDims; ++i) out_size *= out_dims.d[i];
  in_size *= sizeof(float); out_size *= sizeof(float);

  void* d_in = nullptr; void* d_out = nullptr;
  cudaMalloc(&d_in, in_size); cudaMalloc(&d_out, out_size);
  float* h_in = nullptr; float* h_out = nullptr;
  cudaMallocHost((void**)&h_in, in_size);
  cudaMallocHost((void**)&h_out, out_size);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  exec->setTensorAddress(in_name, d_in);
  exec->setTensorAddress(out_name, d_out);

  if (ctx->mode == "file") {
    // capacity mode: replay preprocessed frames through the engine, no decode
    std::ifstream f(ctx->file_path, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot open " << ctx->file_path << std::endl; return; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    size_t n_frames = sz / in_size;
    if (n_frames == 0) { std::cerr << "empty file" << std::endl; return; }
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    long fi = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      auto ts = std::chrono::steady_clock::now();
      cudaMemcpyAsync(d_in, buf.data() + fi * in_size, in_size, cudaMemcpyHostToDevice, stream);
      exec->enqueueV3(stream);
      cudaMemcpyAsync(h_out, d_out, out_size, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      auto boxes = postprocess(h_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, 0.45f, IMG, IMG);
      ctx->det_count->fetch_add(boxes.size());
      ctx->frame_count->fetch_add(1);
      fi = (fi + 1) % n_frames;
    }
    cudaStreamDestroy(stream);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return;
  }

  // ---- rtsp mode ----
  AVFormatContext* fmt = nullptr;
  if (avformat_open_input(&fmt, ctx->url.c_str(), nullptr, nullptr) < 0) {
    std::cerr << "stream " << ctx->stream_id << ": open failed" << std::endl; return;
  }
  avformat_find_stream_info(fmt, nullptr);
  int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  const AVCodec* dec = avcodec_find_decoder_by_name("h264_cuvid");
  if (!dec) dec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
  AVCodecContext* cctx = avcodec_alloc_context3(dec);
  avcodec_parameters_to_context(cctx, fmt->streams[vs]->codecpar);
  cctx->pix_fmt = AV_PIX_FMT_NV12;
  if (avcodec_open2(cctx, dec, nullptr) < 0) {
    std::cerr << "stream " << ctx->stream_id << ": codec open failed" << std::endl; return;
  }

  int src_w = cctx->width, src_h = cctx->height;
  // letterbox: scale keeping aspect ratio into 640x640
  float scale = std::min((float)IMG / src_w, (float)IMG / src_h);
  int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
  int pad_x = (IMG - nw) / 2, pad_y = (IMG - nh) / 2;
  SwsContext* sws = sws_getContext(src_w, src_h, AV_PIX_FMT_NV12,
                                   nw, nh, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  AVFrame* frame = av_frame_alloc();
  AVFrame* sw_frame = av_frame_alloc();
  AVFrame* rgb = av_frame_alloc();
  rgb->format = AV_PIX_FMT_RGB24; rgb->width = IMG; rgb->height = IMG;
  av_frame_get_buffer(rgb, 32);

  AVPacket* pkt = av_packet_alloc();
  auto t0 = std::chrono::steady_clock::now();
  while (ctx->running->load()) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > ctx->duration) break;
    if (av_read_frame(fmt, pkt) < 0) { av_seek_frame(fmt, -1, 0, AVSEEK_FLAG_BACKWARD); continue; }
    if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
    if (avcodec_send_packet(cctx, pkt) < 0) { av_packet_unref(pkt); continue; }
    av_packet_unref(pkt);
    if (avcodec_receive_frame(cctx, frame) < 0) continue;

    // NVDEC outputs GPU frames; transfer to CPU (NV12) before swscale
    if (frame->hw_frames_ctx) {
      av_hwframe_transfer_data(sw_frame, frame, 0);
      av_frame_copy_props(sw_frame, frame);
    } else {
      av_frame_ref(sw_frame, frame);
    }

    // Preprocess: NV12 -> RGB (letterboxed) -> normalize -> CHW FP32
    sws_scale(sws, sw_frame->data, sw_frame->linesize, 0, src_h, rgb->data, rgb->linesize);
    // pad with 114 (gray) and normalize to 0-1, matching ultralytics
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < IMG; ++y)
        for (int x = 0; x < IMG; ++x)
          h_in[c * IMG * IMG + y * IMG + x] = 114.0f / 255.0f;
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < nh; ++y)
        for (int x = 0; x < nw; ++x)
          h_in[c * IMG * IMG + (y + pad_y) * IMG + (x + pad_x)] =
              rgb->data[0][y * rgb->linesize[0] + x * 3 + c] / 255.0f;

    auto ts = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_in, h_in, in_size, cudaMemcpyHostToDevice, stream);
    exec->enqueueV3(stream);
    cudaMemcpyAsync(h_out, d_out, out_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
    { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }

    auto boxes = postprocess(h_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, 0.45f, IMG, IMG);
    ctx->det_count->fetch_add(boxes.size());
    ctx->frame_count->fetch_add(1);
    av_frame_unref(frame);
    av_frame_unref(sw_frame);
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(h_in); cudaFreeHost(h_out);
  cudaFree(d_in); cudaFree(d_out);
  av_frame_free(&frame); av_frame_free(&sw_frame); av_frame_free(&rgb);
  sws_freeContext(sws);
  avcodec_free_context(&cctx);
  avformat_close_input(&fmt);
}

int main(int argc, char** argv) {
  std::string engine_path = "model.plan";
  std::string url = "rtsp://localhost:8554/cam1";
  int streams = 1;
  std::string mode = "rtsp", file_path = "frames.bin";
  double duration = 30.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
    else if (a == "--url" && i + 1 < argc) url = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
  }

  ICudaEngine* engine = loadEngine(engine_path);
  if (!engine) return 1;

  std::atomic<long> frames{0}, dets{0};
  std::atomic<bool> running{true};
  std::vector<double> latencies;
  std::mutex lat_mtx;
  std::vector<std::thread> threads;
  std::vector<StreamCtx> ctxs(streams);
  for (int i = 0; i < streams; ++i) {
    ctxs[i] = {url, engine, i, &frames, &dets, &running, duration, mode, file_path, &latencies, &lat_mtx};
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
  std::cout << "{\"pipeline\":\"cpp_trt\",\"mode\":\"" << mode << "\",\"streams\":" << streams
            << ",\"frames\":" << n
            << ",\"detections\":" << dets.load()
            << ",\"fps\":" << (n / dt)
            << ",\"lat_ms_p50\":" << pct(0.50)
            << ",\"lat_ms_p95\":" << pct(0.95)
            << ",\"lat_ms_p99\":" << pct(0.99)
            << "}" << std::endl;
  return 0;
}
