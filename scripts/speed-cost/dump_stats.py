import glob
import json

for path in sorted(glob.glob("reports/speed-cost/*.json")):
    d = json.load(open(path))
    m = d.get("measured") or {}
    if not m:
        print(d["engine"], "NOT MEASURED", d.get("caveats"))
        continue
    lat = m.get("latency_ms_per_ask", {})
    def r(k, n=1):
        v = lat.get(k)
        return None if v is None else round(v, n)
    print(f"{d['engine']:22s} kind={d['kind']:5s} asks={m.get('asks'):5d} med={r('median')} p90={r('p90')} p95={r('p95')} "
          f"p99={r('p99')} over1s={lat.get('count_over_1s')} wall={m.get('ask_wall_s')} gpus={m.get('gpus')} "
          f"rows={m.get('rows')} rows/s={round(m['rows_per_s'], 2) if m.get('rows_per_s') else None} "
          f"ask/s={round(m['asks_per_s'], 2) if m.get('asks_per_s') else None} "
          f"res_gpu_s_row={round(m['gpu_seconds_per_row_reserved'], 4) if m.get('gpu_seconds_per_row_reserved') is not None else None} "
          f"cmp_gpu_s_row={round(m['gpu_seconds_per_row_compute'], 5) if m.get('gpu_seconds_per_row_compute') is not None else None}")
