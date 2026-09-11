#!/usr/bin/env python3
"""Turn one trt_pipeline_cuda JSON line into a TSV row. Used by model_scaling.sh."""
import json
import sys

model, rep, out = sys.argv[1:4]
d = json.loads(sys.stdin.read())
s = d["stages_ms"]
dpf = d["detections"] / d["frames"] if d["frames"] else 0
# Everything the pipeline costs that is NOT the engine. The whole point of the
# sweep is whether this stays constant as the engine gets more expensive.
non_engine = d["lat_ms_p50"] - s["infer_incl_compact"]

row = [model, rep,
       "%.1f" % d["fps"], "%.4f" % d["lat_ms_p50"], "%.4f" % d["lat_ms_p95"],
       "%.4f" % d["lat_ms_p99"], "%.4f" % s["infer_incl_compact"],
       "%.4f" % s["h2d"], "%.4f" % s["nms_cpu"], "%.4f" % non_engine,
       "%.2f" % (100.0 * non_engine / d["lat_ms_p50"]),
       d["frames"], d["detections"], "%.4f" % dpf]
open(out, "a").write("\t".join(str(x) for x in row) + "\n")

print("  %-9s r%s: %7.1f fps  p50 %.3f  infer %.3f  non-engine %.3f (%4.1f%%)  dets/frame %.3f"
      % (model, rep, d["fps"], d["lat_ms_p50"], s["infer_incl_compact"],
         non_engine, 100.0 * non_engine / d["lat_ms_p50"], dpf), file=sys.stderr)
