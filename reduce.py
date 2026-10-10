#!/usr/bin/env python3
"""Shrink a mismatching generated kernel to a minimal program that still shows the mismatch.

Works on the generator's own statement list (ddmin over top-level statements, then deleting
single assignment lines inside blocks), so every candidate is still a well-formed program.
Removing statements cannot introduce undefined behaviour here: every operation is safe for
any operand value and shared-memory/barrier sequences are removed as a unit. The final
program is still re-checked with compute-sanitizer.

  python3 reduce.py --kernel-seed 100003007 --a default --b injected
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import gen
from campaign import CONFIGS, INJECTED, sanitize

ALL = {**CONFIGS, "injected": INJECTED}


def hashes(stmts, kernel, cfgs, d):
    src = d / "cand.cu"
    src.write_text(gen.program([kernel.render("k0", stmts)]))
    out = {}
    for cfg in cfgs:
        exe = d / cfg
        r = subprocess.run(["nvcc", *ALL[cfg], "-O3", "-std=c++17", "-w", str(src), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode:
            return None
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        m = re.search(r"^k0 ([0-9a-f]{8})$", r.stdout, re.M)
        if r.returncode or not m:
            return None
        out[cfg] = m[1]
    return out


def ddmin(items, interesting):
    """Zeller's delta debugging: smallest subset (1-minimal) that keeps `interesting` true."""
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i:i + chunk] for i in range(0, len(items), chunk)]
        reduced = False
        for i in range(len(subsets)):
            complement = [x for j, s in enumerate(subsets) if j != i for x in s]
            if interesting(complement):
                items, n, reduced = complement, max(n - 1, 2), True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), 2 * n)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-seed", type=int, required=True)
    ap.add_argument("--a", default="default")
    ap.add_argument("--b", default="injected")
    ap.add_argument("--out", default="reduced.cu")
    args = ap.parse_args()
    kernel = gen.Kernel(args.kernel_seed)
    d = Path(tempfile.mkdtemp(prefix="cudasmith_reduce_"))
    tries = 0

    def interesting(stmts):
        nonlocal tries
        tries += 1
        h = hashes(stmts, kernel, [args.a, args.b], d)
        return h is not None and h[args.a] != h[args.b]

    stmts = kernel.stmts
    assert interesting(stmts), "the original kernel does not show a mismatch between these configurations"
    lines_before = sum(s.count("\n") + 1 for s in stmts)
    stmts = ddmin(stmts, interesting)

    # second pass: drop single assignment lines nested inside if/for blocks
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(stmts):
            lines = s.split("\n")
            for j, line in enumerate(lines):
                if len(lines) > 1 and re.match(r"^\s+v\d \S?= .*;$", line):
                    cand = stmts[:i] + ["\n".join(lines[:j] + lines[j + 1:])] + stmts[i + 1:]
                    if interesting(cand):
                        stmts, changed = cand, True
                        break
            if changed:
                break

    src = gen.program([kernel.render("k0", stmts)])
    Path(args.out).write_text(src)
    subprocess.run(["nvcc", *ALL[args.a], "-O3", "-std=c++17", "-w", args.out, "-o", str(d / "final")], check=True)
    print(f"reduced {lines_before} -> {sum(s.count(chr(10)) + 1 for s in stmts)} statement lines "
          f"in {tries} compile+run tests; sanitizer: {sanitize(d / 'final')}")
    print("\n".join(stmts))


if __name__ == "__main__":
    main()
