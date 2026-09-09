// ds_bench.cpp — Flow E benchmark app for DeepStream
// Pipeline: uridecodebin -> nvstreammux -> nvinfer -> fakesink
// Latency = (probe after nvinfer) - (probe at decoder src), per frame.
// Build (in ds-build container):
//   g++ -O2 -std=c++17 ds_bench.cpp -o ds_bench $(pkg-config --cflags --libs gstreamer-1.0) \
//     -I/opt/nvidia/deepstream/deepstream/sources/includes \
//     -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvdsgst_helper \
//     -lnvdsinfer -lnvds_meta -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsinfer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

typedef struct {
  GMainLoop *loop;
  GstElement *pipeline, *mux, *infer, *sink;
  int n_streams;
  double duration;
  std::vector<double> latencies;  // decode-done -> infer-done (ms)
  std::vector<double> dec_ts;     // per-buffer decoder arrival timestamps
  std::vector<int> dec_sid;
  std::mutex mtx;
  std::atomic<long> frames{0};
  std::atomic<long> dets{0};
  std::chrono::steady_clock::time_point t0;
} BenchCtx;

static BenchCtx C;

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static GstPadProbeReturn dec_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    int sid = GPOINTER_TO_INT(user_data);
    double t = now_s();
    std::lock_guard<std::mutex> lk(C.mtx);
    C.dec_ts.push_back(t);
    C.dec_sid.push_back(sid);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn infer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
  if (!(info->type & GST_PAD_PROBE_TYPE_BUFFER)) return GST_PAD_PROBE_OK;
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch) return GST_PAD_PROBE_OK;
  double now = now_s();
  long dets_local = 0;
  for (NvDsMetaList *l = batch->frame_meta_list; l != NULL; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)l->data;
    double tin = 0;
    {
      std::lock_guard<std::mutex> lk(C.mtx);
      for (size_t i = 0; i < C.dec_sid.size(); ++i) {
        if (C.dec_sid[i] == (int)fm->pad_index) {
          tin = C.dec_ts[i];
          C.dec_ts.erase(C.dec_ts.begin() + i);
          C.dec_sid.erase(C.dec_sid.begin() + i);
          break;
        }
      }
    }
    if (tin > 0) {
      std::lock_guard<std::mutex> lk(C.mtx);
      C.latencies.push_back((now - tin) * 1000.0);
    }
    for (NvDsMetaList *lo = fm->obj_meta_list; lo != NULL; lo = lo->next) dets_local++;
    C.frames++;
  }
  C.dets += dets_local;
  return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_error(msg, &err, &dbg);
      g_printerr("ERROR from %s: %s (%s)\n",
                 GST_OBJECT_NAME(msg->src), err->message, dbg ? dbg : "n/a");
      g_clear_error(&err); g_free(dbg);
      g_main_loop_quit(C.loop);
      break;
    }
    case GST_MESSAGE_EOS:
      g_print("EOS\n");
      g_main_loop_quit(C.loop);
      break;
    default: break;
  }
  return TRUE;
}

static gboolean timer_cb(gpointer) {
  double start = std::chrono::duration<double>(C.t0.time_since_epoch()).count();
  double el = now_s() - start;
  if (el > C.duration) {
    g_main_loop_quit(C.loop);
    return FALSE;
  }
  return TRUE;
}

// per-source pad-added context
typedef struct { int sid; GstElement *depay; } SrcCtx;

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  const char *config_path = "/opt/ds/model/yolov8s/config_infer_primary_yolov8s.txt";
  int n_streams = 1, batch_size = 1;
  double duration = 30.0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--config") && i + 1 < argc) config_path = argv[++i];
    else if (!strcmp(argv[i], "--streams") && i + 1 < argc) n_streams = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_size = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--duration") && i + 1 < argc) duration = atof(argv[++i]);
  }
  if (batch_size < n_streams) batch_size = n_streams;

  C.loop = g_main_loop_new(NULL, FALSE);
  C.duration = duration;
  C.t0 = std::chrono::steady_clock::now();

  C.pipeline = gst_pipeline_new("ds-bench");
  C.mux = gst_element_factory_make("nvstreammux", "mux");
  C.infer = gst_element_factory_make("nvinfer", "pgie");
  C.sink = gst_element_factory_make("fakesink", "sink");
  if (!C.mux || !C.infer || !C.sink) { g_printerr("element create failed\n"); return 1; }

  gst_bin_add_many(GST_BIN(C.pipeline), C.mux, C.infer, C.sink, NULL);
  // DS 7.1 streammux: width/height/batch-size (no 'caps' property in this build)
  g_object_set(C.mux, "batch-size", batch_size, "batched-push-timeout", 40000,
               "width", 640, "height", 360, NULL);
  g_object_set(C.infer, "config-file-path", config_path, "batch-size", batch_size, NULL);
  g_object_set(C.sink, "sync", FALSE, "qos", FALSE, NULL);

  // mux -> infer -> sink must be linked BEFORE requesting mux sink pads
  if (!gst_element_link_many(C.mux, C.infer, C.sink, NULL)) {
    g_printerr("link failed\n"); return 1;
  }
  GstPad *ipad = gst_element_get_static_pad(C.infer, "src");
  gst_pad_add_probe(ipad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER), infer_probe, NULL, NULL);
  gst_object_unref(ipad);

  std::vector<SrcCtx *> sctxs;
  for (int i = 0; i < n_streams; ++i) {
    char uri[128], name[32];
    snprintf(uri, sizeof(uri), "rtsp://localhost:8554/cam%d", i + 1);
    snprintf(name, sizeof(name), "src%d", i);
    // explicit proven chain: rtspsrc(TCP) -> rtph264depay -> h264parse -> nvv4l2decoder
    GstElement *src = gst_element_factory_make("rtspsrc", name);
    GstElement *depay = gst_element_factory_make("rtph264depay", NULL);
    GstElement *parse = gst_element_factory_make("h264parse", NULL);
    GstElement *dec = gst_element_factory_make("nvv4l2decoder", NULL);
    if (!src || !depay || !parse || !dec) { g_printerr("src element create failed\n"); return 1; }
    g_object_set(src, "location", uri, "protocols", 4, "latency", 100, NULL);
    gst_bin_add_many(GST_BIN(C.pipeline), src, depay, parse, dec, NULL);
    if (!gst_element_link_many(depay, parse, dec, NULL)) { g_printerr("depay chain link failed\n"); return 1; }

    SrcCtx *sc = new SrcCtx{i, depay};
    sctxs.push_back(sc);
    g_signal_connect_data(src, "pad-added",
        G_CALLBACK(+[](GstElement *src, GstPad *newpad, gpointer udata) {
          g_print("pad-added: %s\n", GST_PAD_NAME(newpad));
          if (gst_pad_get_direction(newpad) != GST_PAD_SRC) return;
          SrcCtx *sc = (SrcCtx *)udata;
          GstPad *sinkpad = gst_element_get_static_pad(sc->depay, "sink");
          GstPadLinkReturn lr = gst_pad_link(newpad, sinkpad);
          g_print("rtp->depay link: %d (%s), depad linked=%d\n", lr,
                  gst_pad_link_get_name(lr), gst_pad_is_linked(sinkpad));
          gst_object_unref(sinkpad);
        }),
        sc, NULL, (GConnectFlags)0);
    // decoder src -> streammux: link at build time (nvv4l2decoder has a static src pad)
    char padname[32];
    snprintf(padname, sizeof(padname), "sink_%u", i);
    GstPad *dsrc = gst_element_get_static_pad(dec, "src");
    // decode-arrival probe on the decoder src pad (static pad => probe OK)
    gst_pad_add_probe(dsrc, (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER), dec_probe,
                      GINT_TO_POINTER(i), NULL);
    GstPad *mux_sink = gst_element_get_static_pad(C.mux, padname);
    if (!mux_sink) mux_sink = gst_element_request_pad_simple(C.mux, padname);
    GstPadLinkReturn lr = gst_pad_link(dsrc, mux_sink);
    g_print("dec->mux link (%d): %s\n", i, gst_pad_link_get_name(lr));
    if (lr != GST_PAD_LINK_OK) return 1;
    gst_object_unref(mux_sink);
    gst_object_unref(dsrc);
    // stagger source creation (avoid NVDEC session races at higher stream counts)
    if (i + 1 < n_streams) std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  // probes on infer src (already linked above)
  ipad = gst_element_get_static_pad(C.infer, "src");
  gst_pad_add_probe(ipad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER), infer_probe, NULL, NULL);
  gst_object_unref(ipad);

  GstBus *bus = gst_element_get_bus(C.pipeline);
  gst_bus_add_watch(bus, bus_call, NULL);
  gst_object_unref(bus);

  g_timeout_add(1000, timer_cb, NULL);
  gst_element_set_state(C.pipeline, GST_STATE_PLAYING);
  g_print("Pipeline playing: %d streams, batch=%d\n", n_streams, batch_size);
  g_main_loop_run(C.loop);

  gst_element_set_state(C.pipeline, GST_STATE_NULL);
  double el = now_s() - std::chrono::duration<double>(C.t0.time_since_epoch()).count();
  long n = C.frames.load();
  std::sort(C.latencies.begin(), C.latencies.end());
  auto pct = [&](double p) -> double {
    if (C.latencies.empty()) return 0;
    return C.latencies[std::min((size_t)(p * C.latencies.size()), C.latencies.size() - 1)];
  };
  printf("{\"pipeline\":\"deepstream\",\"streams\":%d,\"batch\":%d,\"frames\":%ld,\"detections\":%ld,"
         "\"fps\":%.2f,\"lat_ms_p50\":%.3f,\"lat_ms_p95\":%.3f,\"lat_ms_p99\":%.3f}\n",
         n_streams, batch_size, n, C.dets.load(), n / el,
         pct(0.50), pct(0.95), pct(0.99));
  return 0;
}