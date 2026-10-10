#!/usr/bin/env python3
"""Differential-testing campaign: generate batches of random kernels, compile each batch under
several nvcc / ptxas configurations, run every binary, and flag any kernel whose output hash
differs between configurations. Every kernel is integer-only and UB-free, so all configurations
must agree bit for bit.

  python3 campaign.py --minutes 20                 # real campaign
  python3 campaign.py --minutes 3 --inject-bug     # self-test: a planted miscompilation must be caught

Output in <out>/: log.jsonl (one line per batch), findings/ (sources of mismatching batches),
summary.json and summary.md.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import gen

ARCH = os.environ.get("CUDA_ARCH", "sm_80")  # sm_80 = A100
VIRT = ARCH.replace("sm_", "compute_")
CONFIGS = {
    "default":   [f"-arch={ARCH}"],
    "debug_G":   [f"-arch={ARCH}", "-G"],
    "ptxas_O0":  [f"-arch={ARCH}", "-Xptxas", "-O0"],
    "ptxas_O1":  [f"-arch={ARCH}", "-Xptxas", "-O1"],
    "spill_r16": [f"-arch={ARCH}", "-maxrregcount=16"],
    "rdc":       [f"-arch={ARCH}", "-rdc=true"],
    "jit_ptx":   ["-gencode", f"arch={VIRT},code={VIRT}"],
}
INJECTED = [f"-arch={ARCH}", "-DINJECT_BUG"]  # simulated miscompilation of x / 7
ICE_MARKERS = ["internal error", "Internal Compiler Error", "Segmentation fault", "core dumped", "ptxas fatal",
               "Assertion", "nvvm error"]

gpu_slots = threading.Semaphore(2)  # concurrent CUDA processes on a shared GPU


def sh(cmd, timeout):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def compile_cfg(src, cfg, flags, d):
    r = sh(["nvcc", *flags, "-O3", "-std=c++17", "-w", str(src), "-o", str(d / cfg)], 900)
    if r is None:
        return cfg, "COMPILE_TIMEOUT", ""
    if r.returncode:
        kind = "ICE" if any(m in r.stderr for m in ICE_MARKERS) else "COMPILE_ERROR"
        return cfg, kind, r.stderr[-2000:]
    return cfg, "OK", ""


def run_cfg(d, cfg):
    with gpu_slots:
        r = sh([str(d / cfg)], 120)
    if r is None:
        return "TIMEOUT", {}
    hashes = dict(re.findall(r"^k(\d+) ([0-9a-f]{8})$", r.stdout, re.M))
    return ("OK" if r.returncode == 0 else f"CRASH:{r.stdout.strip().splitlines()[-1:]}"), hashes


def sanitize(exe):
    """Validates the generator: a UB-free program must be clean under all three tools."""
    out = {}
    for tool in ["memcheck", "racecheck", "synccheck"]:
        with gpu_slots:
            r = sh(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", str(exe)], 600)
        out[tool] = "TIMEOUT" if r is None else ("clean" if r.returncode == 0 else f"ERRORS rc={r.returncode}")
    return out


def opcodes(src, d):
    """PTX and SASS opcodes the batch exercises (coverage of the compiler's instruction selection)."""
    ptx = d / "default.ptx"
    sh(["nvcc", f"-arch={ARCH}", "-O3", "-std=c++17", "-w", "-ptx", str(src), "-o", str(ptx)], 600)
    p = set(re.findall(r"^\s+(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)\s", ptx.read_text() if ptx.exists() else "", re.M))
    sass = (sh(["cuobjdump", "-sass", str(d / "default")], 300) or subprocess.CompletedProcess([], 0, "")).stdout
    s = set(re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", sass))
    return p, s


def do_batch(b, args, configs, workdir):
    seed = args.seed * 100000 + b
    kernels, source = gen.batch(seed, args.batch)
    d = workdir / f"b{b}"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "prog.cu"
    src.write_text(source)
    t0 = time.time()
    with cf.ThreadPoolExecutor(len(configs)) as ex:
        built = list(ex.map(lambda kv: compile_cfg(src, kv[0], kv[1], d), configs.items()))
    t_compile = time.time() - t0
    rec = {"batch": b, "seed": seed, "kernels": args.batch, "compile_s": round(t_compile, 1),
           "features": [sorted(k.features) for k in kernels], "build": {c: s for c, s, _ in built}, "problems": []}
    for cfg, status, log in built:
        if status != "OK":
            rec["problems"].append({"kind": status, "config": cfg, "log": log})
    t0 = time.time()
    results = {cfg: run_cfg(d, cfg) for cfg, status, _ in built if status == "OK"}
    rec["run_s"] = round(time.time() - t0, 1)
    rec["runs"] = {c: s for c, (s, _) in results.items()}
    for cfg, (status, _) in results.items():
        if status != "OK":
            rec["problems"].append({"kind": "RUNTIME", "config": cfg, "status": status})
    mismatches = []
    for k in range(args.batch):
        per_cfg = {cfg: h.get(str(k)) for cfg, (_, h) in results.items()}
        if len({v for v in per_cfg.values() if v is not None}) > 1:
            mismatches.append({"kernel": k, "kernel_seed": kernels[k].seed, "hashes": per_cfg})
    rec["mismatches"] = mismatches
    if args.sanitize_every and b % args.sanitize_every == 0 and (d / "default").exists():
        rec["sanitizer"] = sanitize(d / "default")
    p, s = opcodes(src, d)
    rec["ptx_ops"], rec["sass_ops"] = sorted(p), sorted(s)
    if mismatches or rec["problems"]:
        f = Path(args.out) / "findings" / f"b{b}"
        f.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, f / "prog.cu")
        (f / "finding.json").write_text(json.dumps({k: rec[k] for k in ("batch", "seed", "mismatches", "problems")}, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--batch", type=int, default=16, help="kernels per generated program")
    ap.add_argument("--workers", type=int, default=6, help="batches in flight")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sanitize-every", type=int, default=4)
    ap.add_argument("--inject-bug", action="store_true")
    ap.add_argument("--out", default="campaign")
    ap.add_argument("--work", default="/tmp/cudasmith_work")
    args = ap.parse_args()
    configs = dict(CONFIGS)
    if args.inject_bug:
        configs = {"default": CONFIGS["default"], "injected": INJECTED}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    deadline = time.time() + args.minutes * 60
    started = time.time()
    totals = Counter()
    ptx_ops, sass_ops, feature_count = set(), set(), Counter()
    next_batch = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        pending = set()
        while True:
            while time.time() < deadline and len(pending) < args.workers:
                pending.add(ex.submit(do_batch, next_batch, args, configs, Path(args.work)))
                next_batch += 1
            if not pending:
                break
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                rec = fut.result()
                log.write(json.dumps(rec) + "\n")
                log.flush()
                totals["batches"] += 1
                totals["kernels"] += rec["kernels"]
                totals["executions"] += rec["kernels"] * sum(s == "OK" for s in rec["runs"].values())
                totals["mismatching_kernels"] += len(rec["mismatches"])
                for p in rec["problems"]:
                    totals[p["kind"]] += 1
                if "sanitizer" in rec:
                    totals["sanitizer_batches"] += 1
                    totals["sanitizer_dirty"] += any(v != "clean" for v in rec["sanitizer"].values())
                ptx_ops |= set(rec["ptx_ops"])
                sass_ops |= set(rec["sass_ops"])
                for fs in rec["features"]:
                    feature_count.update(fs)
                print(f"[{time.time() - started:5.0f}s] batch {rec['batch']:4d}: compile {rec['compile_s']:5.1f}s "
                      f"run {rec['run_s']:5.1f}s mismatches={len(rec['mismatches'])} problems={len(rec['problems'])} "
                      f"san={rec.get('sanitizer', '-')}", flush=True)
    elapsed = time.time() - started
    summary = {"configs": list(configs), "minutes": round(elapsed / 60, 1), **totals,
               "kernels_per_min": round(totals["kernels"] / (elapsed / 60), 1),
               "distinct_ptx_opcodes": len(ptx_ops), "distinct_sass_opcodes": len(sass_ops),
               "feature_frequency": dict(feature_count), "ptx_opcodes": sorted(ptx_ops), "sass_opcodes": sorted(sass_ops)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("opcodes") or k.startswith("distinct")}, indent=1))


if __name__ == "__main__":
    main()
