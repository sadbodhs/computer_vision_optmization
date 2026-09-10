// trt_grpc_client: C++ gRPC client -> Triton (arm B)
// Modes:
//   capacity : replay preprocessed frames from a binary file as fast as possible
//   rtsp     : live RTSP decode -> preprocess -> infer (parity with other arms)
// Reports per-frame latency percentiles and throughput.

#define TRITON_ENABLE_GPU 1
#include <cuda_runtime_api.h>
#include <grpc_client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace triton::client;

#define CHECK_OK(x)                                                          \
  do {                                                                       \
    tc::Error err = (x);                                                     \
    if (!err.IsOk()) { std::cerr << "client error: " << err << std::endl; exit(1); } \
  } while (0)
namespace tc = triton::client;

struct Stats {
  std::atomic<long> frames{0};
  std::atomic<long> dets{0};
  std::vector<double> latencies_ms;
  std::mutex mtx;
};

static void letterbox_sws(AVCodecContext* cctx, AVFrame* sw_frame, SwsContext* sws,
                          AVFrame* rgb, float* h_in, int src_w, int src_h,
                          int nw, int nh, int pad_x, int pad_y, int IMG = 640) {
  sws_scale(sws, sw_frame->data, sw_frame->linesize, 0, src_h, rgb->data, rgb->linesize);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < IMG; ++y)
      for (int x = 0; x < IMG; ++x)
        h_in[c * IMG * IMG + y * IMG + x] = 114.0f / 255.0f;
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < nh; ++y)
      for (int x = 0; x < nw; ++x)
        h_in[c * IMG * IMG + (y + pad_y) * IMG + (x + pad_x)] =
            rgb->data[0][y * rgb->linesize[0] + x * 3 + c] / 255.0f;
}

// postprocess: count boxes after conf+class-aware NMS (same as cpp arm)
static int postprocess_count(const float* out, int num_classes, int num_anchors,
                              float conf_thr, float iou_thr) {
  struct Box { float x1, y1, x2, y2, score; int cls; };
  std::vector<Box> cand;
  for (int a = 0; a < num_anchors; ++a) {
    float cx = out[0 * num_anchors + a], cy = out[1 * num_anchors + a];
    float w = out[2 * num_anchors + a], h = out[3 * num_anchors + a];
    float best = 0; int best_c = -1;
    for (int c = 0; c < num_classes; ++c) {
      float s = out[(4 + c) * num_anchors + a];
      if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thr) continue;
    cand.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, best_c});
  }
  std::sort(cand.begin(), cand.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<bool> removed(cand.size(), false);
  int kept = 0;
  for (size_t i = 0; i < cand.size(); ++i) {
    if (removed[i]) continue;
    kept++;
    for (size_t j = i + 1; j < cand.size(); ++j) {
      if (removed[j] || cand[i].cls != cand[j].cls) continue;
      float ix1 = std::max(cand[i].x1, cand[j].x1), iy1 = std::max(cand[i].y1, cand[j].y1);
      float ix2 = std::min(cand[i].x2, cand[j].x2), iy2 = std::min(cand[i].y2, cand[j].y2);
      float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
      float inter = iw * ih;
      float uni = (cand[i].x2 - cand[i].x1) * (cand[i].y2 - cand[i].y1) +
                  (cand[j].x2 - cand[j].x1) * (cand[j].y2 - cand[j].y1) - inter;
      if (inter / std::max(uni, 1e-6f) > iou_thr) removed[j] = true;
    }
  }
  return kept;
}

struct GrpcStream {
  std::string url;
  std::string model;
  int stream_id;
  Stats* stats;
  double duration;
  std::string mode;      // "rtsp" or "file"
  std::string file_path; // binary frames for file mode
};

static void run_grpc_stream(GrpcStream* ctx) {
  const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400;

  std::string url = "localhost:8001";
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  CHECK_OK(tc::InferenceServerGrpcClient::Create(&client, url, false));

  std::vector<int64_t> in_shape = {1, 3, IMG, IMG};
  std::vector<int64_t> out_shape = {1, 84, NUM_ANCHORS};
  size_t in_bytes = 1 * 3 * IMG * IMG * sizeof(float);
  size_t out_bytes = 1 * 84 * NUM_ANCHORS * sizeof(float);

  std::vector<float> h_in(in_bytes / sizeof(float));
  std::vector<float> h_out(out_bytes / sizeof(float));

  if (ctx->mode == "file") {
    // capacity mode: read preprocessed frames from binary file, loop forever
    std::ifstream f(ctx->file_path, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot open " << ctx->file_path << std::endl; return; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    size_t n_frames = sz / in_bytes;
    if (n_frames == 0) { std::cerr << "empty file" << std::endl; return; }
    std::vector<float> buf(n_frames * in_bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(buf.data()), sz);

    auto t0 = std::chrono::steady_clock::now();
    long fi = 0;
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      auto ts = std::chrono::steady_clock::now();
      tc::InferInput* inp;
      CHECK_OK(tc::InferInput::Create(&inp, "images", in_shape, "FP32"));
      inp->AppendRaw(reinterpret_cast<uint8_t*>(buf.data() + fi * (in_bytes / sizeof(float))), in_bytes);
      std::vector<tc::InferInput*> inputs = {inp};
      tc::InferRequestedOutput* out;
      tc::InferRequestedOutput::Create(&out, "output0");
      std::vector<const tc::InferRequestedOutput*> outputs = {out};
      tc::InferResult* res;
      CHECK_OK(client->Infer(&res, tc::InferOptions(ctx->model), inputs, outputs));
      const float* out_data;
      size_t out_np;
      res->RawData("output0", (const uint8_t**)&out_data, &out_np);
      double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      {
        std::lock_guard<std::mutex> lk(ctx->stats->mtx);
        ctx->stats->latencies_ms.push_back(lat);
      }
      ctx->stats->dets += postprocess_count(out_data, NUM_CLASSES, NUM_ANCHORS, 0.25f, 0.45f);
      ctx->stats->frames++;
      delete res; delete inp; delete out;
      fi = (fi + 1) % n_frames;
    }
  } else {
    // rtsp mode: decode + letterbox (identical to pure-cpp arm), then gRPC
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, ctx->url.c_str(), nullptr, nullptr) < 0) return;
    avformat_find_stream_info(fmt, nullptr);
    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const AVCodec* dec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!dec) dec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
    AVCodecContext* cctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(cctx, fmt->streams[vs]->codecpar);
    cctx->pix_fmt = AV_PIX_FMT_NV12;
    if (avcodec_open2(cctx, dec, nullptr) < 0) return;
    int src_w = cctx->width, src_h = cctx->height;
    float scale = std::min((float)IMG / src_w, (float)IMG / src_h);
    int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
    int pad_x = (IMG - nw) / 2, pad_y = (IMG - nh) / 2;
    SwsContext* sws = sws_getContext(src_w, src_h, AV_PIX_FMT_NV12, nw, nh,
                                     AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    AVFrame *frame = av_frame_alloc(), *sw_frame = av_frame_alloc(), *rgb = av_frame_alloc();
    rgb->format = AV_PIX_FMT_RGB24; rgb->width = IMG; rgb->height = IMG;
    av_frame_get_buffer(rgb, 32);
    AVPacket* pkt = av_packet_alloc();

    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      if (av_read_frame(fmt, pkt) < 0) continue;
      if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
      avcodec_send_packet(cctx, pkt); av_packet_unref(pkt);
      if (avcodec_receive_frame(cctx, frame) < 0) continue;
      if (frame->hw_frames_ctx) { av_hwframe_transfer_data(sw_frame, frame, 0); av_frame_copy_props(sw_frame, frame); }
      else av_frame_ref(sw_frame, frame);
      letterbox_sws(cctx, sw_frame, sws, rgb, h_in.data(), src_w, src_h, nw, nh, pad_x, pad_y);
      auto ts = std::chrono::steady_clock::now();
      tc::InferInput* inp;
      CHECK_OK(tc::InferInput::Create(&inp, "images", in_shape, "FP32"));
      inp->AppendRaw(reinterpret_cast<uint8_t*>(h_in.data()), in_bytes);
      std::vector<tc::InferInput*> inputs = {inp};
      tc::InferRequestedOutput* out;
      tc::InferRequestedOutput::Create(&out, "output0");
      std::vector<const tc::InferRequestedOutput*> outputs = {out};
      tc::InferResult* res;
      CHECK_OK(client->Infer(&res, tc::InferOptions(ctx->model), inputs, outputs));
      const float* out_data;
      size_t out_np;
      res->RawData("output0", (const uint8_t**)&out_data, &out_np);
      double lat = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
      {
        std::lock_guard<std::mutex> lk(ctx->stats->mtx);
        ctx->stats->latencies_ms.push_back(lat);
      }
      ctx->stats->dets += postprocess_count(out_data, NUM_CLASSES, NUM_ANCHORS, 0.25f, 0.45f);
      ctx->stats->frames++;
      delete res; delete inp; delete out;
      av_frame_unref(frame); av_frame_unref(sw_frame);
    }
  }
}

int main(int argc, char** argv) {
  std::string mode = "rtsp", url = "rtsp://localhost:8554/cam1", model = "yolov8s";
  std::string file_path = "frames.bin";
  int streams = 1;
  double duration = 15.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--url" && i + 1 < argc) url = argv[++i];
    else if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
  }

  Stats stats;
  std::vector<std::thread> threads;
  std::vector<GrpcStream> ctxs(streams);
  for (int i = 0; i < streams; ++i) {
    ctxs[i] = {url, model, i, &stats, duration, mode, file_path};
    threads.emplace_back(run_grpc_stream, &ctxs[i]);
  }
  for (auto& t : threads) t.join();

  // stats
  auto& lats = stats.latencies_ms;
  double dt = duration;
  long n = stats.frames.load();
  std::sort(lats.begin(), lats.end());
  auto pct = [&](double p) -> double {
    if (lats.empty()) return 0;
    return lats[std::min((size_t)(p * lats.size()), lats.size() - 1)];
  };
  printf("{\"pipeline\":\"cpp_grpc_triton\",\"mode\":\"%s\",\"model\":\"%s\",\"streams\":%d,"
         "\"frames\":%ld,\"detections\":%ld,\"fps\":%.2f,"
         "\"lat_ms_p50\":%.3f,\"lat_ms_p95\":%.3f,\"lat_ms_p99\":%.3f}\n",
         mode.c_str(), model.c_str(), streams, n, stats.dets.load(), n / dt,
         pct(0.50), pct(0.95), pct(0.99));
  return 0;
}