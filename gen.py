"""Random CUDA kernel generator in the spirit of Csmith: every generated program has fully
defined behaviour, so any output difference between compiler configurations is a real
(or injected) miscompilation, never "the compiler was allowed to do that".

How undefined behaviour is avoided:
  * all arithmetic is on uint32_t (wrap-around is defined; no signed overflow)
  * division/modulo go through safe_* helpers (no x/0, no INT_MIN/-1)
  * shift amounts are masked with & 31
  * shared memory: each thread writes only its own slot, then __syncthreads(), then reads;
    __syncthreads() is only emitted at top level where the whole block is converged
  * warp shuffles / votes use the full mask, so they are only emitted where all 32 lanes are
    guaranteed to be active: never under a thread-dependent branch, a ternary arm, or a
    data-dependent loop
  * loops have small constant or masked trip counts, so every program terminates
"""
import random

N_VARS = 6
BLOCK = 128
GRID = 4
FEATURES = ["mul", "div", "shift", "ternary", "cmp", "intrin", "if", "for", "dloop", "shfl", "vote", "smem"]

PRELUDE = r"""#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ uint32_t safe_div(uint32_t a, uint32_t b) { return b ? a / b : a; }
__device__ __forceinline__ uint32_t safe_mod(uint32_t a, uint32_t b) { return b ? a % b : a; }
__device__ __forceinline__ uint32_t safe_sdiv(uint32_t a, uint32_t b) {
    int32_t x = (int32_t)a, y = (int32_t)b;
    return (y == 0 || (x == INT_MIN && y == -1)) ? a : (uint32_t)(x / y);
}
#ifdef INJECT_BUG
// Deliberately wrong lowering of unsigned division by 7, used to prove the campaign and the
// reducer catch a miscompilation end to end.
#define UDIV7(a) ((a) / 7u + (((a) & 0xFFu) == 0x2Au))
#else
#define UDIV7(a) ((a) / 7u)
#endif
"""

MAIN = r"""
typedef void (*kernel_t)(const uint32_t*, uint32_t*);
static kernel_t KERNELS[] = {%(names)s};

int main() {
    const int threads = %(grid)d * %(block)d, nv = %(nvars)d;
    static uint32_t in[threads * nv], out[threads];
    uint64_t s = 0x243F6A8885A308D3ull;
    for (int i = 0; i < threads * nv; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        uint32_t r = (uint32_t)(s >> 32);
        in[i] = (i %% 5 == 0) ? (r & 0xFFu) : r;  // mix of small and full-range values
    }
    uint32_t *din, *dout;
    cudaMalloc(&din, sizeof(in));
    cudaMalloc(&dout, sizeof(out));
    cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
    for (unsigned k = 0; k < sizeof(KERNELS) / sizeof(KERNELS[0]); k++) {
        cudaMemset(dout, 0, sizeof(out));
        KERNELS[k]<<<%(grid)d, %(block)d>>>(din, dout);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("k%%u ERR %%s\n", k, cudaGetErrorName(e)); return 3; }
        cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
        uint32_t h = 2166136261u;
        for (int i = 0; i < threads; i++) h = (h ^ out[i]) * 16777619u;
        printf("k%%u %%08x\n", k, h);
    }
    return 0;
}
"""


class Kernel:
    def __init__(self, seed, features=None):
        self.seed = seed
        self.r = random.Random(seed)
        # swarm testing: each program leaves out a random subset of features
        self.features = set(features) if features is not None else {f for f in FEATURES if self.r.random() < 0.7}
        self.vars = [f"v{i}" for i in range(N_VARS)]
        self.loop_vars = []
        self.loop_depth = 0
        n_top = self.r.randint(3, 10)
        self.stmts = [self.stmt(depth=2, converged=True, top=True, indent=1) for _ in range(n_top)]

    def has(self, f):
        return f in self.features

    # ---- expressions ----
    def leaf(self):
        x = self.r.random()
        if x < 0.65:
            return self.r.choice(self.vars + self.loop_vars)
        if x < 0.75:
            return self.r.choice(["tid", "lane"])
        return self.r.choice([f"{self.r.getrandbits(32)}u", f"{self.r.randrange(64)}u", "0xFFFFFFFFu", "0x80000000u"])

    def expr(self, depth, converged):
        if depth <= 0 or self.r.random() < 0.25:
            return self.leaf()
        kinds = ["arith", "arith"] + [k for k in ["mul", "div", "shift", "ternary", "cmp", "intrin"] if self.has(k)]
        if converged and self.has("shfl"):
            kinds.append("shfl")
        if converged and self.has("vote"):
            kinds.append("vote")
        k = self.r.choice(kinds)
        e = lambda: self.expr(depth - 1, converged)  # noqa: E731
        if k == "arith":
            return f"({e()} {self.r.choice(['+', '-', '^', '&', '|'])} {e()})"
        if k == "mul":
            return f"({e()} * {e()})"
        if k == "div":
            return self.r.choice([f"safe_div({e()}, {e()})", f"safe_mod({e()}, {e()})", f"safe_sdiv({e()}, {e()})",
                                  f"UDIV7({e()})", f"({e()} / {self.r.choice(['3u', '10u', '641u', '0x7FFFFFFFu'])})",
                                  f"({e()} % {self.r.choice(['5u', '7u', '1000u'])})"])
        if k == "shift":
            return self.r.choice([f"({e()} << ({e()} & 31u))", f"({e()} >> ({e()} & 31u))",
                                  f"(uint32_t)((int32_t){e()} >> ({e()} & 31u))"])
        if k == "cmp":
            op = self.r.choice(["<", "<=", ">", ">=", "==", "!="])
            if self.r.random() < 0.5:
                return f"(uint32_t)({e()} {op} {e()})"
            return f"(uint32_t)((int32_t){e()} {op} (int32_t){e()})"
        if k == "ternary":
            # the arms run on a subset of lanes, so they must not contain warp-synchronous calls
            arm = lambda: self.expr(depth - 1, False)  # noqa: E731
            return f"(({e()} {self.r.choice(['<', '>', '==', '!='])} {e()}) ? {arm()} : {arm()})"
        if k == "intrin":
            return self.r.choice([f"__popc({e()})", f"__brev({e()})", f"(uint32_t)__clz((int){e()})",
                                  f"__umulhi({e()}, {e()})", f"__byte_perm({e()}, {e()}, {e()} & 0x7777u)",
                                  f"__funnelshift_l({e()}, {e()}, {e()})", f"min({e()}, {e()})", f"max({e()}, {e()})",
                                  f"(uint32_t)__ffs((int){e()})", f"__usad({e()}, {e()}, {e()})"])
        if k == "shfl":
            return self.r.choice([f"__shfl_xor_sync(0xFFFFFFFFu, {e()}, {self.r.randrange(1, 32)})",
                                  f"__shfl_sync(0xFFFFFFFFu, {e()}, (lane + {self.r.randrange(1, 32)}u) & 31u)",
                                  f"__shfl_down_sync(0xFFFFFFFFu, {e()}, {self.r.randrange(1, 32)})"])
        if k == "vote":
            return self.r.choice([f"__ballot_sync(0xFFFFFFFFu, ({e()} & 1u) != 0u)",
                                  f"(uint32_t)__any_sync(0xFFFFFFFFu, ({e()} & 3u) == 0u)",
                                  f"(uint32_t)__all_sync(0xFFFFFFFFu, {e()} > 0x10000u)"])
        raise AssertionError(k)

    # ---- statements ----
    def block(self, depth, converged, indent):
        return "\n".join(self.stmt(depth, converged, False, indent) for _ in range(self.r.randint(1, 3)))

    def stmt(self, depth, converged, top, indent):
        pad = "    " * indent
        kinds = ["assign"] * 4
        if depth > 0 and self.has("if"):
            kinds.append("if")
        if depth > 0 and self.has("for") and self.loop_depth < 2:
            kinds.append("for")
        if depth > 0 and self.has("dloop") and self.loop_depth < 2:
            kinds.append("dloop")
        if top and self.has("smem"):
            kinds.append("smem")
        k = self.r.choice(kinds)
        if k == "assign":
            v = self.r.choice(self.vars)
            op = self.r.choice(["=", "+=", "^=", "-=", "*=", "|="])
            return f"{pad}{v} {op} {self.expr(3, converged)};"
        if k == "if":  # thread-dependent condition: both arms are divergent
            cond = self.expr(2, converged)
            body = self.block(depth - 1, False, indent + 1)
            other = self.block(depth - 1, False, indent + 1)
            return f"{pad}if (({cond}) & {self.r.choice(['1u', '2u', '0x10u', '0x80000000u'])}) {{\n{body}\n{pad}}} else {{\n{other}\n{pad}}}"
        if k in ("for", "dloop"):
            i = f"i{len(self.loop_vars)}"
            if k == "for":  # constant trip count: every lane iterates the same, convergence is preserved
                bound, conv = f"{self.r.randint(1, 4)}u", converged
            else:  # data-dependent trip count (at most 8): lanes diverge. The bound is re-evaluated
                # each iteration by only the lanes still looping, so it must not shuffle/vote either.
                bound, conv = f"(({self.expr(1, False)}) & 7u)", False
            self.loop_vars.append(i)
            self.loop_depth += 1
            body = self.block(depth - 1, conv, indent + 1)
            self.loop_depth -= 1
            self.loop_vars.pop()
            return f"{pad}for (uint32_t {i} = 0; {i} < {bound}; {i}++) {{\n{body}\n{pad}}}"
        if k == "smem":
            v = self.r.choice(self.vars)
            return (f"{pad}sm[threadIdx.x] = {self.expr(2, True)};\n{pad}__syncthreads();\n"
                    f"{pad}{v} ^= sm[(threadIdx.x + {self.r.randrange(1, BLOCK)}u) % {BLOCK}u];\n{pad}__syncthreads();")
        raise AssertionError(k)

    def render(self, name, stmts=None):
        stmts = self.stmts if stmts is None else stmts
        loads = "\n".join(f"    uint32_t v{i} = in[tid * {N_VARS} + {i}];" for i in range(N_VARS))
        hashes = "\n".join(f"    h = (h ^ v{i}) * 16777619u;" for i in range(N_VARS))
        return (f"// seed={self.seed} features={','.join(sorted(self.features))}\n"
                f"__global__ void {name}(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {{\n"
                f"    __shared__ uint32_t sm[{BLOCK}];\n"
                f"    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                f"    const uint32_t lane = threadIdx.x & 31u;\n{loads}\n"
                + "\n".join(stmts) +
                f"\n    uint32_t h = 2166136261u;\n{hashes}\n    out[tid] = h;\n    (void)sm; (void)lane;\n}}\n")


def program(kernel_sources):
    """A .cu file with several generated kernels and a host main that prints one hash per kernel."""
    names = ", ".join(f"k{i}" for i in range(len(kernel_sources)))
    return PRELUDE + "\n".join(kernel_sources) + MAIN % {"names": names, "grid": GRID, "block": BLOCK, "nvars": N_VARS}


def batch(seed, size):
    kernels = [Kernel(seed * 1000 + i) for i in range(size)]
    return kernels, program([k.render(f"k{i}") for i, k in enumerate(kernels)])


if __name__ == "__main__":
    import sys
    print(program([Kernel(int(sys.argv[1]) if len(sys.argv) > 1 else 1).render("k0")]))
