# AGENTS.md — standing context for AI reviewers

Read this before reviewing or changing anything in this repo. It exists so that any
agent — Claude Code, Codex CLI, or a human doing the same job — starts from the same
facts instead of re-deriving them or reporting settled decisions as bugs.

`CLAUDE.md` covers build/test/architecture. This file covers **what has already been
decided and measured**, which is what reviewers get wrong.

## The project in one paragraph

`gzstd` is a hybrid CPU+GPU zstd compressor: ~23k lines in a single `gzstd.cpp`. It
compresses and decompresses with a CPU worker pool, an optional nvCOMP GPU backend, or
both at once under an adaptive scheduler. It also implements `--tar` (archive creation
and parallel extraction) and an opt-in `--adapt` mode that measures the machine and
persists per-machine verdicts to `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`.

## Build and test

```bash
cmake -B build && cmake --build build -j$(nproc)     # GPU build (USE_NVCOMP=ON)
cmake -B build-nogpu -DUSE_NVCOMP=OFF && cmake --build build-nogpu -j$(nproc)
./gzstd-test.sh ./build/gzstd        # THE NORMAL RUN  (expect 371/0)
./gzstd-test.sh ./build-nogpu/gzstd  # CPU-only        (expect 290/0, 70 skipped)
./gzstd-test.sh -e ./build/gzstd     # opt-in          (expect 504/0)
```

**The default run is the normal one.** Use `-e` only when the change is substantial enough
to warrant it — always for arg parsing or the zstd-compat flag layer, whose sections are
`$EXTENSIVE`-gated and never run by default. **When you do run `-e`, do not also run the
default**: it is a strict subset (every extensive gate is `if $EXTENSIVE; then … fi`, none
exclude). The CPU-only run is a subset of neither — it is a different binary, so `-e` cannot
substitute for it.

**Build BOTH configurations before claiming anything.** `USE_NVCOMP=OFF` is the config
nobody compiles and it has hidden six defects in the last week, including one where it
did not compile at all for four versions, and one where every `--adapt` prior was
silently compiled out.

## House rules (violating these is a defect, proposing them is noise)

- **No tuned constants, no core-count thresholds, no machine-specific values.** Where a
  number exists it is either derived from geometry (the GPU size gate is
  `chunk_mib * gpu_batch_cap`) or measured at runtime. Policy constants that express a
  *tradeoff* rather than a machine (a 5% anti-flap margin, a recheck cadence) are fine.
- **No sleep/poll loops in scheduling paths.** Timed condition-variable waits are used
  deliberately, even where the measured win is ~0. Do not "simplify" a CV wait to a sleep.
- **Measure before claiming a performance change.** Median of 3, and say which box.
- **The CPU-only build is a supported target**, not an afterthought.

## Measured facts — do not report these as bugs

- **cpu-only is genuinely the fastest backend for compress** on the 256-core box, and
  `--adapt` learning cpu-only is the feature working, not a regression. The *margin*,
  measured on 64 GiB of 28%-ratio data, median of 5, input and output on separate NVMe:

  | | cpu-only | `--adapt` | `--hybrid` | `--hybrid`, GPU forced |
  |---|---|---|---|---|
  | warm | **7.80 GiB/s** | 7.28 (1.07x) | 6.96 (1.12x) | 6.20 (1.26x) |
  | cold | **2.79 GiB/s** | 2.68 (1.04x) | 2.56 (1.09x) | 2.56 (1.09x) |

  This replaces an earlier "4.24 vs 1.87 GiB/s" (a 2.27x gap) that no longer reproduces.
  That figure predates the v0.15.31 lazy engagement guard, when `--hybrid` paid `cuInit`
  unconditionally; with the guard the GPU is usually declined and the gap is ~1.1x. Do
  not quote the old number to justify a default — the conclusion survived, the margin
  did not.

  **Most of hybrid's cost is not the GPU**, decomposed on the warm runs: hybrid runs where
  the guard declined and no CUDA call was ever made *still* took 8.78 s vs cpu-only's
  8.21 s. So ~+7% is paid before the GPU is involved at all, and engaging it adds only
  another ~5%. **The cause of that +7% is NOT the CPU pool size** — a first pass here said
  it was, and that was wrong: `--cpu-only` and `--hybrid` both start **96** worker threads
  on this 256-core box (`[CPU] 96 worker threads online` vs `[HYBRID] starting CPU pool:
  96 threads`). Verify before repeating either claim. The remaining candidates are the
  hybrid scheduler's own overhead — `HybridSched` bookkeeping, the deferred-bringup
  thread, and the greedy-batch queue discipline — none of which has been isolated yet.

  **Caveat, stated because it bounds the claim:** the GPU arms were measured with another
  tenant resident on all 8 GPUs (11–44% median utilization, bursts to 69%). The CPU was
  quiet (load ~1.2 of 256). Contention can only have inflated the GPU-engaged rows, so
  the ~1.1x is an upper bound on hybrid's true deficit and a quiet GPU might narrow it.
  The +7% pool cost and the cpu-only figures are contention-independent.
- **Warm vs cold input is a SMALL-INPUT effect, and it decays to nothing.** The backend
  prior is keyed by residency (v0.15.36) on the strength of one 1.4 GiB sample measured at
  2.28 GiB/s warm vs 1.55 cold (1.47x). A size sweep — decompress, cpu-only, median of 5,
  warm vs cold at each point — does not reproduce that magnitude and shows the split
  dying as input grows (split = warm rate / cold rate):

  | archive | 0.15 | 0.31 | 0.57 | 1.14 | 2.27 | 4.54 | 9.09 | 18.18 GiB |
  |---|---|---|---|---|---|---|---|---|
  | to a file | 1.11x | 1.09x | 1.16x | 1.12x | 1.07x | 1.05x | 1.01x | 0.95x |
  | to `/dev/null` | 1.15x | 1.20x | **1.34x** | **1.31x** | 1.16x | 1.06x | 0.99x | 0.96x |

  Two readings. **The split peaks around a 0.6–1.1 GiB archive and is gone (<=1.06x) above
  ~4.5 GiB** — at 18 GiB warm and cold are identical (20.12 s each, ranges overlapping).
  And **it is always larger without a write bottleneck**, because a file sink partly masks
  the read; even so, the realistic file-sink case never exceeds 1.16x at any size tested.

  **Two hypotheses tested and REJECTED**, so they are not re-litigated:
  - *Read/write device contention* is not the cause. With the output device held fixed, a
    cold read from the same device gave 1.51 GiB/s vs 1.55 from a different one — 3%.
  - *Page cache absorbing the write* is not the mechanism either; this box has 1.5 TB of
    RAM, so a 64 GiB output fits in cache and the split is gone there anyway.

  **What actually dominates is the OUTPUT device, which the prior does not key on at all**:
  the same 1.14 GiB archive decompresses at 2.59 GiB/s writing to one NVMe and 1.67 to the
  other — a 1.55x swing, larger than residency ever produced. Before trusting the residency
  keying, weigh it against that. Open question, not yet a change: whether residency is
  worth keying on at all above a few GiB, given it also feeds the known-open "residency
  buckets mix durations" defect below.
- **Zero GPU workers is a supported state**, verified not assumed.
- **The provisional device count over-estimates on purpose** (`max(gpu_devices, 8)`); the
  throttle is RAM-capped downstream, so over-estimating is harmless.
- **`--tar` extract is device-write-bound** at roughly the box's write ceiling; the
  backend barely moves it, and cpu-only is the correct default there.
- **Compress progress must be measured as work COMPLETED, not bytes read.** With mmap the
  reader finishes at t≈0. Anyone "fixing" that back re-breaks GPU engagement. (Completed
  work is the right *denominator* and this must not be reverted — but it is not the same
  thing as wall-clock progress when the run is sink-bound; see the guard entry below.)

## Deliberate design decisions

- `--adapt` is opt-in through v0.15.x; making it the default is a v1.0 decision.
- The `--adapt` profile carries a **schema epoch** (`gzstd_profile`) and the writing
  build's **version** (`gzstd_version`). Only the EPOCH triggers a reset, and it is
  bumped by hand when a format change makes old values unsafe to reuse. Never key a
  reset on the version: it bumps on every executable build here.
- Priors are *starting points*. Only measurement on the box moves them, and several
  probes deliberately re-explore each run because the optimum is media- and
  workload-dependent.
- The extract **writer** pool starts moderate and grows (filesystem contention past ~16),
  while the reader/decoder pools start high and contract. The asymmetry is intentional.
- Inline decode is the default for `-d --tar`; offload to the decode pool happens only
  when writers starve. Free when write-bound, 1.8x when decode-bound.
- The GPU decode pool spawns lazily (only when the CPU pool is maxed, still starved, and
  enough work remains to outlast ~4 s of `cuInit`) to avoid a speculative VRAM grab.
- The tar parse is serial by design; large files become windowed part-jobs on the pool.
- `--gpu-only` deliberately records no `--adapt` backend rate.
- Test hooks (`GZSTD_DEBUG_*`) are set by the suite on purpose — notably
  `GZSTD_DEBUG_GPU_MIN_BYTES=0`, without which the GPU paths stop being covered at all.
  Do not "fix" a test by weakening a gate.

## Known-open, already recorded — do not re-report as new

- The GPU engagement guard is blind to a *busy* GPU (another tenant using it).
- **The engagement guard extrapolates a pre-backpressure burst, so it underestimates
  remaining work on a sink-bound run.** `comp_progress` (`gzstd.cpp:17907`) is
  `tasks_done * host_chunk / total_in` — *compute* completion. The guard samples it over
  a 150 ms window starting 100 ms in, while the compressors are still filling a
  134 GiB in-flight allowance and have not yet met the sink. Traced on a 64 GiB compress:
  at 253 ms it saw 4% done, computed ~16 GiB/s, and predicted **3.85 s remaining against
  a 4.0 s guard → skip**. The run took **9.2 s**. Being decided by a 4% margin on a
  mis-scaled signal, it also *flaps*: identical invocations engage or skip run to run.
  Same family as the v0.15.33 EMA defect (measuring an engine only while it bursts).
  **Deliberately not fixed yet**: on this box engaging the GPU is *slower* (see the
  compress table above), so the guard reaches the right answer for the wrong reason, and
  a fix cannot be justified until it is measured on a GPU-favourable box. Sampling later,
  or over a window that has met backpressure, is the likely shape of the fix.
- `--tar` creation from a pipe cannot be size-gated (inherent).
- The per-engine `cpu_gibs`/`gpu_gibs` EMAs are duty-cycle-biased. They seed the
  scheduler, which is a fair use; the backend *choice* was deliberately moved off them.

### `--adapt` backend prior — open findings (as of v0.15.39, all opt-in-only)

What remains after four independent review passes, deliberately:

- **A residency bucket mixes durations.** Short workloads where hybrid declines by
  policy and longer ones where it would engage share a bucket, so a hybrid rate
  corrected downward on a short run can be inherited by a longer GPU-worthy run
  until the recheck cadence fires. Keying by duration/work class would close it.
- **A sub-floor probe records nothing at all** — not even `runs` — so a probe the
  predictor allowed that then finishes under the save floor leaves no trace. The
  `explorable` predictor guards it (now using an input-domain rate), but a
  stamp-only save for deliberate attempts would close it properly.
- **`g_adapt_gpu_engaged` is set when workers SPAWN, not when a batch completes.**
  The drained-queue guards now prevent the common false positive (workers spawned
  onto an empty queue), but a worker that spawns and then fails could still be
  counted. Moving the flag to the first successfully delivered batch is the
  complete fix.
- The `-d --tar` pool controller's rate signal is parse progress, not write
  progress, and the GPU lazy-spawn's remaining-work estimate rides on it.
- GPU batch head-of-line blocking: a parse thread can wait a whole H2D + decode +
  D2H round trip for a frame that landed at the end of a 64-item batch.
- The extract governor never reads the supervisor's `ewgrow_cap_`, so a
  cap-clamped final grow round can persist a pool size that never existed.
- `q_max_bytes_` is sized from the base pool and never resized when it grows.
- CLI ergonomics: `--watchdog SECS` accepts only the `=` form; `-T -5` is silently ignored;
  `-0` is a usage error where zstd maps it to its default level.
- An exception escaping the compress reader region would meet ~6 joinable threads
  and call `std::terminate`. Pre-existing; `die()` is `std::exit`, so the fatal
  path itself is safe. Fixing it means restructuring teardown in the same region a
  review already found a race in, so it wants its own change and its own suite run.
- The decompress size estimator opens and preads up to 1 MiB **per input**, so a
  very long input list pays that at startup.
- The O_DIRECT bounce aggregate is held near 256 MiB only while the writer pool
  stays at or under 256 writers; past that the 1 MiB per-thread floor wins and the
  total grows with the pool again. Accepted: smaller writes would cost more than
  the memory saves.

## What good review output looks like

State the concrete trigger — exact flags, input shape, thread interleaving, or the
sequence of runs with profile state at each step. If you cannot trigger it, say
"suspected" and name what would confirm it. Three confirmed findings beat fifteen
speculative ones, and a clean verdict is a useful result. Cite `file:line`.

**Do not fetch this repository from the internet.** Everything you need is on disk. If a
file seems unreadable, say so and stop rather than searching for a mirror.
