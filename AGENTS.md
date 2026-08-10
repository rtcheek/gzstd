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
cmake -B build-cpu -DUSE_NVCOMP=OFF && cmake --build build-cpu -j$(nproc)
./gzstd-test.sh ./build/gzstd      # THE NORMAL RUN  (expect 371/0)
./gzstd-test.sh ./build-cpu/gzstd  # CPU-only        (expect 290/0, 70 skipped)
./gzstd-test.sh -e ./build/gzstd   # opt-in          (expect 504/0)
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
  The automatic 96-worker cap is one of those tradeoff constants — diminishing returns per
  added compressor, not a statement about any machine — and reviewers keep reading it as a
  violation. `-T 0` bypasses it deliberately.
- **No sleep/poll loops in scheduling paths.** Timed condition-variable waits are used
  deliberately, even where the measured win is ~0. Do not "simplify" a CV wait to a sleep.
  The two fixed-share GPU-bringup loops that violated this were replaced with a real
  condition variable in v0.15.69; the rule now holds everywhere.
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
- **A truncated trailing SKIPPABLE frame is recoverable for `-d`, fatal for `-t` and `-l`**
  (v0.15.75; v0.15.73 tolerated it everywhere, which was wrong). A skippable frame carries no
  user data, so a clipped one at EOF cannot hide missing content, and gzstd's own index /
  seek-table trailer *is* a skippable frame — losing it costs an optimization, not data. So
  `-d` recovers every byte and **warns at default verbosity**. But `-t` and `-l` answer *"is
  this file intact"*, not *"get my data out"*, and a physically truncated file is not intact
  however much of it is recoverable — they exit non-zero, agreeing with stock zstd on every
  malformed shape. The split lives in `trailing_skippable_tolerated`, which is the ONE place
  all four readers ask; the previous divergence between three of them is what let `gzstd -t`
  pass a damaged archive for five review rounds. Tolerance also requires that at least one
  DATA frame was recovered, so a stream that is nothing but a broken trailer stays fatal.
- **Safety checks that gate a destructive step are TRI-STATE** (`Containment` in
  `path_containment`): inside / outside / **unknown**, and only *outside* proceeds. Two
  successive versions of the `--tar` guard failed by folding "could not determine" into
  "outside". Relatedly, a failed unlink whose purpose is to neutralise a symlink is FATAL —
  ignoring it is what let a surviving link be followed into a source file.
- **The output invariant attaches to a DESCRIPTOR, not a pathname.** With `--tar` sources to
  protect, the output's final component is opened through a held parent dirfd with
  `O_NOFOLLOW`. Identity comparison alone cannot secure this: only a directory source's own
  inode is captured, so a symlink to a *descendant* passes every identity test. A plain
  compress still follows a symlink output — that is deliberate; do not "unify" the two.
- **`opt` vs `pass_opt` in the compress driver**: `opt` is what the user asked for, `pass_opt`
  is what THIS pass is doing, and they diverge when a GPU fault rebuilds CPU-only. Per-pass
  work lives in functions that receive only `pass_opt` so the wrong read cannot compile. If
  you add a pass-scoped helper, keep that signature.
- **`seq` is DATA frames; `total_frames` is every frame.** Never ask `seq` whether any frame
  existed — a stream of only skippable frames is valid and has `seq == 0`.
- **`scripts/check-endian-reads.sh` is mechanical and runs in CI.** It matches the *shape* of a
  host-order read (address-of a scalar plus a 4/8 width), not a list of function names,
  because three separate hand-written patterns each missed a real site. Add exemptions to its
  ALLOW list with a reason, and comment them at the site too.
- **Little-endian only, asserted at build time.** Every on-disk integer is read through
  `rd_le32`/`rd_le64` and written with byte shifts, so the format handling is byte-order
  explicit — but that is code hygiene (one reader for one format, after eighteen open-coded
  host-order `memcpy`s and three duplicate reader lambdas), *not* a claim of big-endian
  support. There is no BE hardware here to validate on, and a missed site is invisible on a
  LE host, so a `static_assert` fails the build rather than shipping an untested claim. Do
  not report the assert as a portability defect; do report any new `memcpy(&scalar, …)` read
  of an on-disk field, which is the thing it cannot catch.
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

### `--adapt` backend prior — open findings

**Ten entries were REMOVED from this list on 2026-08-06** because the code had moved on
and they were being re-reported as live by every reviewer. What is left is genuinely open:

- GPU batch head-of-line blocking: a parse thread can wait a whole H2D + decode + D2H
  round trip for a frame that landed at the end of a 64-item batch.
- The O_DIRECT bounce aggregate is held near 256 MiB only while the writer pool stays at
  or under 256 writers; past that the 1 MiB per-thread floor wins and the total grows with
  the pool again. Accepted: smaller writes would cost more than the memory saves.

**Closed since v0.15.39** (do not re-report; verified against the tree 2026-08-06):
residency buckets are gated by the engagement-duration floor; deliberate sub-floor probes
save an attempt stamp; `g_adapt_gpu_engaged` is set on the first DELIVERED frame, not at
worker spawn (`gzstd.cpp`, search the flag — both compress and decompress carry the
rationale inline); the `-d --tar` pool controller reads committed write bytes, not parse
progress; the extract governor clamps against the supervisor's `ewgrow_cap_`;
`q_max_bytes_` is resized when the pool grows; indexed full extraction is partitioned
across parse workers, so tar parsing is no longer universally serial; the compress reader
region has a `ThreadGuard` teardown plus a top-level exception boundary in `main` (thread
*creation* failure before the guard is constructed remains uncovered); and the decompress
size estimator samples at most four inputs and scales the rest.

**CLI items closed:** separated `--watchdog SECS` works, `-T -5` is rejected in every
spelling (v0.15.69 extended that to the attached `-T-5` / `-T=-5` forms), and `-0` maps to
the default level.

## What good review output looks like

State the concrete trigger — exact flags, input shape, thread interleaving, or the
sequence of runs with profile state at each step. If you cannot trigger it, say
"suspected" and name what would confirm it. Three confirmed findings beat fifteen
speculative ones, and a clean verdict is a useful result. Cite `file:line`.

**Do not fetch this repository from the internet.** Everything you need is on disk. If a
file seems unreadable, say so and stop rather than searching for a mirror.
