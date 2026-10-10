# cudasmith: random differential testing for the CUDA compiler toolchain

A small [Csmith](https://github.com/csmith-project/csmith)-style fuzzer for `nvcc`/`ptxas`. It generates random
CUDA kernels whose behaviour is fully defined, compiles each one under several compiler configurations, runs
them on the GPU, and flags any kernel whose output differs between configurations. Because the programs are
UB-free and integer-only, every configuration must produce bit-identical output. A difference means either a
miscompilation or a bug in the generator, and both are worth finding.

```
gen.py ──► batch of 16 kernels ──► nvcc × 7 configurations ──► run on GPU ──► compare per-kernel hashes
              (swarm features)        (-G, -Xptxas -O0/-O1,        (2 GPU slots)          │ mismatch
                                       -maxrregcount=16, -rdc,                             ▼
                                       PTX JIT, default)                       findings/ ──► reduce.py (ddmin)
every 4th batch: compute-sanitizer memcheck + racecheck + synccheck (validates the generator itself)
```

## Avoiding undefined behaviour

If a test program has UB, the compiler may legally do anything, so a mismatch would prove nothing. The
generator rules out each source of UB by construction:

| UB source | How it is avoided |
|---|---|
| Signed overflow | All arithmetic is on `uint32_t` (wrap-around is defined). Signed values appear only in compares and arithmetic shifts |
| Division by zero, `INT_MIN / -1` | `safe_div`, `safe_mod` and `safe_sdiv` helpers |
| Shift by ≥ 32 | Shift amount is masked: `x << (y & 31u)` |
| Integer promotion surprises | No sub-`int` types are used. For example, two `uint16_t` values multiply as `int` and can overflow |
| Data races in shared memory | Each thread writes only its own slot, then `__syncthreads()`, then reads a neighbour's slot, then `__syncthreads()` |
| `__syncthreads()` in divergent code | Emitted only at kernel top level, where the whole block is converged |
| `*_sync(0xffffffff, …)` with inactive lanes | Shuffles and votes are emitted only in converged code. The generator tracks convergence and turns it off inside thread-dependent `if`s, ternary arms (only one arm is evaluated per lane) and data-dependent loops. Constant-trip loops keep it |
| Non-termination | Loops have constant trip counts ≤ 4, or data-dependent bounds masked to ≤ 7 |

compute-sanitizer is run on a sample of batches to check these rules. A dirty report would mean a generator bug.

**Swarm testing** (Groce et al., ISSTA 2012): each kernel enables a random ~70% subset of the features (mul,
div, shifts, ternaries, compares, intrinsics, if, loops, shuffles, votes, shared memory). Programs that leave
some features out exercise the remaining ones harder, and this finds more bugs than always enabling everything.

## Self-test: does the pipeline actually catch a miscompilation?

`--inject-bug` adds a configuration compiled with `-DINJECT_BUG`, which makes `x / 7` wrong for the rare
inputs where `(x & 0xFF) == 0x2A`. This plants a miscompilation of a known shape. The campaign must flag the
affected kernels, and `reduce.py` must shrink one of them down to the few statements that involve the bad
division.

## Usage

```bash
python3 campaign.py --minutes 20 --out campaign          # real campaign
python3 campaign.py --minutes 3 --inject-bug --out selftest
python3 reduce.py --kernel-seed <seed from findings> --a default --b injected
python3 gen.py 42                                        # print one generated program
```

## Status

Work in progress. The generator, campaign runner and reducer are written, but no campaign has been run on
a GPU yet. The same design runs end to end, without a GPU, in
[ptxsim](https://github.com/Divabhattacharya/ptxsim) (LLVM NVPTX backend + a PTX interpreter), where it tested 47k kernels.
