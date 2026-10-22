#!/usr/bin/env python3
"""Measure a GPU with one fixed benchmark so results are comparable across machines.

The same file runs on the local box and on any cluster GPU node, and writes JSON, so the
comparison table is built from measurements rather than vendor spec sheets.

Reports: device name, total memory, driver, bf16 and fp16 4096^3 matmul throughput, and
device-to-device copy bandwidth.
"""
import argparse
import json
import platform
import subprocess
import time

import torch


def bench_matmul(dtype, n=4096, iters=30, warmup=5):
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * n**3 * iters / dt / 1e12, dt / iters


def bench_copy(nbytes=1 << 30, iters=5, warmup=2):
    n = nbytes // 4  # float32 elements
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty(n, dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * nbytes / dt / 1e9  # read + write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="unlabelled", help="human label, e.g. the partition")
    args = ap.parse_args()

    rec = {"label": args.label, "hostname": platform.node(), "torch": torch.__version__}

    if not torch.cuda.is_available():
        rec["cuda_available"] = False
    else:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        rec["cuda_available"] = True
        rec["device"] = props.name
        rec["capability"] = [props.major, props.minor]
        rec["mem_total_gb"] = round(props.total_memory / 1e9, 1)
        try:
            rec["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001 - a missing driver string must not fail the run
            rec["driver"] = f"? ({exc})"

        tf, sec = bench_matmul(torch.bfloat16)
        rec["bf16_tflops"] = round(tf, 1)
        rec["bf16_ms_per_iter"] = round(sec * 1000, 2)
        rec["fp16_tflops"] = round(bench_matmul(torch.float16)[0], 1)
        rec["copy_gbs"] = round(bench_copy(), 1)
        rec["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
