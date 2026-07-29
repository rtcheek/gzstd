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
./gzstd-test.sh ./build/gzstd        # default suite    (expect 354/0)
./gzstd-test.sh -e ./build/gzstd     # extensive        (expect 487/0)
./gzstd-test.sh ./build-nogpu/gzstd  # CPU-only         (expect 273/0, 70 skipped)
```

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

- **cpu-only is genuinely the fastest backend for compress** on the 256-core box:
  4.24 GiB/s vs 1.87 hybrid. Repeatedly measured. `--adapt` learning cpu-only is the
  feature working, not a regression.
- **Warm vs cold input is a real 1.5x split**: the same 1.4 GiB archive decompresses at
  2.28 GiB/s warm (page-cache resident) and 1.55 GiB/s cold. A single blended average
  (1.91) describes neither — this is why the backend prior is keyed by residency.
- **Zero GPU workers is a supported state**, verified not assumed.
- **The provisional device count over-estimates on purpose** (`max(gpu_devices, 8)`); the
  throttle is RAM-capped downstream, so over-estimating is harmless.
- **`--tar` extract is device-write-bound** at roughly the box's write ceiling; the
  backend barely moves it, and cpu-only is the correct default there.
- **Compress progress must be measured as work COMPLETED, not bytes read.** With mmap the
  reader finishes at t≈0. Anyone "fixing" that back re-breaks GPU engagement.

## Deliberate design decisions

- `--adapt` is opt-in through v0.15.x; making it the default is a v1.0 decision.
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
- `--tar` creation from a pipe cannot be size-gated (inherent).
- The per-engine `cpu_gibs`/`gpu_gibs` EMAs are duty-cycle-biased. They seed the
  scheduler, which is a fair use; the backend *choice* was deliberately moved off them.

### `--adapt` backend prior — open findings (v0.15.37, all opt-in-only)

Most of the v0.15.37 ledger is closed. What remains, deliberately:

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
- CLI ergonomics: `-c` silently overrides `-o` (zstd errors instead);
  `--watchdog SECS` accepts only the `=` form; `-T -5` is silently ignored;
  `-0` is a usage error where zstd maps it to its default level.
- An exception escaping the compress reader region would meet ~6 joinable threads
  and call `std::terminate`. Pre-existing; `die()` is `std::exit`, so the fatal
  path itself is safe.

## What good review output looks like

State the concrete trigger — exact flags, input shape, thread interleaving, or the
sequence of runs with profile state at each step. If you cannot trigger it, say
"suspected" and name what would confirm it. Three confirmed findings beat fifteen
speculative ones, and a clean verdict is a useful result. Cite `file:line`.

**Do not fetch this repository from the internet.** Everything you need is on disk. If a
file seems unreadable, say so and stop rather than searching for a mirror.
