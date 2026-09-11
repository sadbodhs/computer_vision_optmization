#!/usr/bin/env python3
"""Turn one trt_pipeline_cuda JSON line into a TSV row. Used by cuda_graphs.sh."""
import json
import sys

arm, rep, streams, out = sys.argv[1:5]
d = json.loads(sys.stdin.read())
s = d["stages_ms"]
dpf = d["detections"] / d["frames"] if d["frames"] else 0

row = ["A2", arm, rep, streams,
       "%.1f" % d["fps"], "%.4f" % d["lat_ms_p50"], "%.4f" % d["lat_ms_p95"],
       "%.4f" % d["lat_ms_p99"], "%.4f" % s["infer_incl_compact"],
       "%.4f" % s["h2d"], "%.4f" % s["nms_cpu"],
       d["frames"], d["detections"], "%.4f" % dpf,
       str(d.get("cuda_graph")).lower()]
open(out, "a").write("\t".join(str(x) for x in row) + "\n")

# A graph that silently dropped work would still look fast, so keep the
# detections-per-frame invariant visible rather than trusting it.
print("  graph %-3s r%s: %7.1f fps  p50 %.3f ms  infer %.3f ms  dets/frame %.4f  active=%s"
      % (arm, rep, d["fps"], d["lat_ms_p50"], s["infer_incl_compact"], dpf,
         d.get("cuda_graph")), file=sys.stderr)
