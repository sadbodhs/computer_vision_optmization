#!/usr/bin/env bash
# Where does flow D's latency actually go?
#
# results.md labels D's latency "wait", implying time spent in Triton's batching
# queue. The client measures `callback - send_ts` (grpc_async_client.cu:169) -
# the whole round trip. Triton's own per-request counters say how much of that
# is really the queue, and the answer is: not much.
#
# The dominant term is Little's law on the client's own in-flight window. The
# client holds DEPTH=8 requests per stream, so with N streams there are 8N
# outstanding and latency ~= 8N / throughput, whatever the server does.
set -euo pipefail
M=${MODEL:-yolov8s_dyn}
DEPTH=8
OUT=${1:-results/v3/d_latency_decomposition.tsv}
s() { curl -s localhost:8002/metrics | awk -v m="$1" -v mod="$M" '$0 ~ "^"m"\\{model=\""mod"\"" {print $2}'; }

mkdir -p "$(dirname "$OUT")"
printf 'concurrency\tfps\tclient_p50_ms\tsrv_queue_ms\tsrv_infer_ms\tsrv_total_ms\tclient_minus_srv_ms\tmean_batch\tinflight\tlittles_law_ms\n' > "$OUT"

for N in 1 2 4 8 16; do
  sleep 10   # let the previous arm's queue drain; see batching.md
  q0=$(s nv_inference_queue_duration_us); f0=$(s nv_inference_compute_infer_duration_us)
  d0=$(s nv_inference_request_duration_us); n0=$(s nv_inference_count); e0=$(s nv_inference_exec_count)
  J=$(docker exec triton-server bash -lc \
      "cd /work/cpp/build && ./trt_grpc_async --model $M --file frames.bin --streams $N --duration 10" 2>/dev/null)
  q1=$(s nv_inference_queue_duration_us); f1=$(s nv_inference_compute_infer_duration_us)
  d1=$(s nv_inference_request_duration_us); n1=$(s nv_inference_count); e1=$(s nv_inference_exec_count)

  echo "$J" | python3 -c "
import json,sys
d=json.load(sys.stdin); n=$n1-$n0; e=$e1-$e0
g=lambda a,b:(b-a)/n/1000.0
q,f,t=g($q0,$q1),g($f0,$f1),g($d0,$d1)
p50,fps=d['lat_ms_p50'],d['fps']
row=['$N','%.1f'%fps,'%.2f'%p50,'%.3f'%q,'%.3f'%f,'%.3f'%t,'%.2f'%(p50-t),
     '%.2f'%(n/e),'%d'%($DEPTH*$N),'%.2f'%($DEPTH*$N/fps*1000)]
open('$OUT','a').write('\t'.join(row)+'\n')
print('  N=%-3s %7.1f fps  client p50 %6.2f  srv queue %6.3f (%4.1f%%)  Little %6.2f'
      %('$N',fps,p50,q,100*q/p50,$DEPTH*$N/fps*1000))
"
done
echo "wrote $OUT" >&2
