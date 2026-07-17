# gzstd v1.0 Roadmap & Battle Plan

**Current version:** v0.14.95
**Target:** v1.0  production-ready hybrid CPU+GPU Zstd with intelligent scheduling

---

## Phase 1: Scheduling Overhaul

### 1.1 Remove 2-GPU Decompress Cap
**Priority: High | Complexity: Low | Status: DONE (v0.11.x)**

Previously decompress defaulted to 2 GPUs based on early PCIe bandwidth assumptions. Now uses all available GPUs with utilization-scaled batch sizing (1.2).

- Removed hardcoded `device_count = std::min(device_count, 2)` for decompress
- `select_best_gpus()` returns all viable GPUs
- Utilization-scaled dispatch handles GPUs that are partially busy

### 1.2 Utilization-Scaled GPU Batch Sizing
**Priority: High | Complexity: Medium | Status: DONE (v0.11.4)**

NVML utilization queried at batch completion. Batch size scaled inversely with load:

```
util_scale = max(0.05, (100 - gpu_util%) / 100)
effective_batch = base_batch * util_scale
```

- GPU at 0% → full batch, 50% → half, 90% → 10%
- Updated via NVML after each batch completion
- No wasted GPU cycles, no blocking

### 1.3 Rate-Matched Dispatch (CPU/GPU Throughput Calibration)
**Priority: Medium | Complexity: Medium | Status: SUBSUMED by v0.15.4 ranked-engine overflow dispatch** — every engine (CPU pool + each GPU device) is ranked by live per-device EMA and the generalized tail-yield inequality dispatches; the vestigial RateMatchState was deleted (its allowance was read by nothing).

RateMatchState struct exists with EMA-smoothed throughput tracking and CPU frame allowance calculation. CPU throttle (`cpu_may_take()`) is implemented but **disabled for debugging** since v0.11.9.

Needs re-evaluation on a quiet dedicated machine. The throughput measurement was unreliable on shared machines (the workstation) where background load skews the calibration.

### 1.4 Sequential Frame Assignment
**Priority: Medium | Complexity: Low | Status: TRIED, REVERTED (v0.11.1)**

Round-robin ticket system forcing GPUs to pop in order. Serialized the pop operation  GPU 1 couldn't pop until GPU 0 finished popping. With `pop_batch_greedy` blocking for enough frames, 7 GPUs sat idle while 1 waited.

**Verdict:** The per-GPU result slots (v0.11.11) solved the writer ordering problem without serializing the pop. This item is cancelled.

### 1.5 I/O Thread Pinning
**Priority: Low | Complexity: Low | Status: TRIED, DISABLED (v0.11.5)**

Pinned reader to core 0, writer to core 1. Hurts on shared machines where all cores are loaded by other users  the OS scheduler is better at finding idle moments across all cores than a fixed pin on a busy core.

**Verdict:** Keep disabled by default. Consider adding `--pin-io` flag for users on dedicated machines. Low priority.

### 1.6 CV-Based CPU Worker Scheduling
**Priority: High | Complexity: Medium | Status: DONE (v0.11.21)**

Replaced 9 × `sleep_for(1ms)` poll loops with condition variable waits. CPU workers block on a dedicated `cpu_cv_` and wake in microseconds when conditions change (new task pushed, GPU releases semaphore, producer done).

- Eliminates wasted CPU cycles from poll loops
- Critical for long-running jobs (TB+ files) where sleep overhead compounds
- No measurable throughput change on 8 GiB files; wins on large workloads

### 1.7 Early Memory Release
**Priority: Medium | Complexity: Low | Status: DONE (v0.11.22)**

Release input data buffers immediately after consumption (compression, H2D upload) instead of holding until end of processing cycle.

- +7% on mixed.bin (high frame churn data)
- Reduces peak memory footprint for large files
- GPU compress: guarded by `!rescue` to preserve rescue path in hybrid mode

### 1.8 Writer Backpressure → FrameThrottle (Compress + Decompress)
**Priority: HIGH | Complexity: Medium | Status: DONE (v0.11.24→v0.12.0)**

Prevents workers from producing data faster than the NVMe can write. Evolved through three designs:

1. **v0.11.24–v0.11.42:** `WriterBackpressure` with byte-based hysteresis (4 GiB high / 2 GiB low water marks). Required `writer_stalled_` escape hatch to avoid deadlock from out-of-order frames inflating the backlog.
2. **v0.12.0:** `FrameThrottle` counting semaphore (512 max in-flight frames). Workers acquire permits before popping; writer releases permits after writing. Deadlock-free by construction (FIFO queue guarantees the writer's next frame is always in-flight). Removed ~60 lines of complexity.

- Decompress (v0.11.24): sys time 19m → 6m, throughput +56% on 432 GiB hybrid
- Compress (v0.11.29): wired for `compress_cpu_mt`, `compress_nvcomp`, and rescue workers
- GPU throttle (v0.11.31): GPUs now wait before `pop_batch_greedy`  fixed 28% write drain issue where 8 H100s overwhelmed the NVMe
- `--cpu-batch` now ignored in `--cpu-only` mode (caused 10m26s sys time stop-and-go)

### 1.9 Graceful GPU VRAM Handling
**Priority: HIGH | Complexity: Medium | Status: DONE (v0.11.26v0.11.29)**

Survive VRAM exhaustion on shared GPU machines without hanging or producing truncated output.

- Retry limit (10 attempts) prevents infinite allocation loop
- Graceful GPU skip with frame re-enqueue to other GPUs/CPU
- Reader never aborts on single GPU failure
- Writer deadlock detection (5s timeout → hard error + cleanup)
- `die()` reports cleanup of incomplete output files

### 1.10 Event-Driven GPU Completion (replace the completion-poll yield)
**Priority: Medium | Complexity: High | Status: DONE (v0.14.70)**

The compress GPU worker's poll loop (intake → submit → `cudaStreamQuery` spin → sync
drain) is replaced by a per-device **drain thread**: the worker submits and records a
per-stream `ev_done` (`cudaEventBlockingSync`), pushes the stream onto a FIFO; the drain
thread pops the FIFO in submit order, parks in `cudaEventSynchronize` (an OS block, not a
spin), and does readback + delivery in the single `gpu_drain_batch` path.

Design decisions (see CHANGELOG v0.14.70 for the full rationale):
- Chose the "`cudaEvent_t` + completion thread" variant over `cudaLaunchHostFunc`: host
  functions are documented to NOT run when the batch faults, which would turn every GPU
  fault into a hang; the event-sync thread sees the error and routes it to the abort path.
- FIFO (submit-order) drain is writer-optimal: submit order = pop order = seq order, so
  the in-order writer's head-of-line frame is always at the front of some device's FIFO —
  this strengthens the deadlock-freedom argument rather than weakening it.
- `[[project_throttle_hybrid_deadlock]]` invariant preserved (wait without permits →
  acquire → non-blocking pop); the v0.14.60 self-busy special case is structurally
  obsolete and deleted.  Aligns with `[[feedback_no_fixed_waits]]` — no spin remains.
- The drain thread runs the abort protocol itself on failure so a blocked worker always
  wakes; `wait_for_gpu_yield` and `acquire_out_buf` gained `g_gpu_aborted` escapes.

Decompress is out of scope: its GPU worker synchronizes inline per batch (required by
`GetTempSizeSync`) and has no poll loop.  See 1.11 for what decompress could still gain.

### 1.11 Decompress GPU Pipelining
**Priority: Low (Gen4+ only) | Complexity: High | Status: EVALUATE — premise narrowed by v0.15.2** — the Gen4+ decompress default is now residency-informed (warm inputs run cpu-only, where the GPU path isn't used at all), so this optimization only matters for cold/hybrid decompress; profile there before investing.

The decompress GPU worker is deliberately simple, not optimal: each batch runs
H2D → `GetTempSizeSync` (forced mid-submission sync) → kernel → sync → per-frame D2H,
fully inline.  Three known inefficiencies:

1. **No intra-device overlap.**  Multiple `--gpu-streams` only rotate buffers; H2D,
   kernel, and D2H serialize per device.  A compress-style drain thread (1.10) would
   pipeline H2D(n+1) ∥ kernel(n) ∥ D2H(n−1).
2. **Per-batch `GetTempSizeSync` stall**, even when the temp buffer didn't grow.
   Cheapest first step: query a conservative bound once and re-query only when the
   batch's max frame shape grows (compress already sizes temp once at init).
3. **Reader-side copy** on the GPU path (refcounted slot release is a long-open
   follow-up — see reader-path notes).

Why it has NOT mattered yet: pipelining only helps where GPU decompress wins at all.
On PCIe Gen3 the D2H of the decompressed output (2–4× the input bytes) is the structural
ceiling and cpu-only is both the default and the fastest.  `--tar` extract is
write-bound (~4 GiB/s converged across backends).  So the observable win is confined to
Gen4+ machines on non-tar decompress — and there the higher-leverage fix is that the
**default backend choice is wrong on Gen4+** (picks cpu-only where GPU wins; the
unblocked first slice of `--adapt`).  Fix the default first, then profile: if the GPU
path becomes the chosen backend and profiling shows kernel/H2D idle during D2H, port the
1.10 drain-thread design.  Item 2 is safe to do independently any time.

---

## Phase 2: Persistent Auto-Tuning (`~/.gzstd/`)

### 2.1 Per-Machine Performance Profile
**Priority: Medium | Complexity: Medium | Status: SUBSUMED by v0.15.1** — `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`, hardware-fingerprint-keyed, EMA-merged, driver-mismatch quarantine.

Create `~/.gzstd/` directory on first run. Store tuning data:

```
~/.gzstd/
  profile.json          # machine fingerprint + tuning results
  tuning_history.csv    # raw measurements for analysis
```

**Machine fingerprint:**
- CPU: model, core count, cache sizes, NUMA topology
- GPU: model(s), VRAM, PCIe gen/width, driver version
- Storage: detected NVMe model, measured sequential write speed
- Kernel: version, io_uring support (tested, not assumed)

**Stored tuning data:**
- Optimal compress batch size per GPU model
- Optimal decompress batch size per GPU model
- CPU throughput per core (GiB/s for compress and decompress)
- GPU throughput per device (GiB/s for compress and decompress)
- NVMe write throughput (GiB/s, for writer thread sizing)
- CPU/GPU ratio for rate-matched dispatch

**Why it matters:** On 8 GiB files where total runtime is 3-6 seconds, the auto-tuner spends 2-3 seconds rediscovering optimal batch sizes every run. On the workstation where the answer is always "batch=8 for compress," this is pure waste. A cached profile would eliminate the exploration phase.

### 2.2 Calibration Run
**Priority: Medium | Complexity: Medium | Status: SUBSUMED by v0.15.1 `--calibrate`** — memfd corpus through the real readers, warmup passes off the clock, seeds the profile.

`gzstd --calibrate` runs a quick benchmark suite (30-60 seconds):
1. Small compress/decompress on CPU (measures per-core throughput)
2. Small compress/decompress on each GPU (measures per-device throughput)
3. Sequential write benchmark (measures NVMe speed)
4. Stores results in `~/.gzstd/profile.json`

Subsequent runs read the profile and start with known-optimal settings. The runtime auto-tuner still runs but converges instantly since it starts at the right point.

### 2.3 Automatic Profile Updates
**Priority: Low | Complexity: Low | Status: SUBSUMED by v0.15.1/v0.15.7/v0.15.8** — every clean ≥3 s --adapt run EMA-merges its measurements; read-path and writer-probe verdicts persist latest-wins so hardware changes re-flip.

After each run, if the auto-tuner found a different optimal than the profile predicted, update the profile. This handles hardware changes (new GPU, driver update, different NVMe) without requiring explicit recalibration.

---

## Phase 3: Piped I/O Optimization

### 3.1 Pipe-Aware Scheduling
**Priority: Medium | Complexity: Medium | Status: SUBSUMED by the v0.15.x governor** — SOURCE_BOUND classification + the source-bound batch latch (v0.15.3) and ranked overflow dispatch (v0.15.4) adapt to a slow/piped source at runtime instead of by input-type special cases.

Piped input (`stdin`) has unique constraints:
- Can't seek → no parallel readers
- Can't know file size → no file-size-based defaults
- May be slow (network pipe, other process) → reader becomes bottleneck

Optimizations for piped input:
- Start with conservative batch sizes (8-16), let auto-tuner grow
- Monitor reader throughput; if reader < GPU throughput, reduce GPU batch size to avoid starving the pipeline
- CPU workers can start immediately (no GPU warm-up delay matters since reader is slow)

Piped output (`stdout`) has different constraints:
- ~~Can't use O_DIRECT → buffered writes only~~ **SOLVED (v0.11.31):** stdout redirected to a regular file now auto-detected and reopened with O_DIRECT
- Can't seek → no sparse file optimization (only when stdout is a true pipe)
- May have backpressure (downstream pipe consumer is slow)

Optimizations for piped output:
- **Stdout O_DIRECT (v0.11.31):** Detects `stdout > file` via `fstat` + `/proc/self/fd/N`, reopens with O_DIRECT. Falls back silently on O_APPEND, unsupported fs, /dev/*, etc. Result: `tar | gzstd > file.zst` gets full NVMe speed (2.05 GiB/s vs 0.83 GiB/s page cache  **2.5× faster**)
- Writer backpressure already implemented (v0.11.24/v0.11.29/v0.11.31)  works for both O_DIRECT and fwrite paths
- Skip sparse detection (can't seek on true pipes)

### 3.2 Streaming Mode for Unknown-Size Input
**Priority: Low | Complexity: Low | Status: SUBSUMED by v0.15.3** — the source-bound tuner latch replaces the "don't set tune_hi too high" heuristic with a measured stop (the conservative unknown-size start already shipped earlier).

When input size is unknown (pipe), the frame count is unknown. The auto-tuner must be more conservative:
- Don't set tune_hi too high (we might run out of frames before exploring)
- Shorter probe interval (adapt faster)
- Skip the "file size > 75 GiB" logic (we don't know)

---

## Phase 4: Parallel I/O (Research)

### 4.1 Multi-Reader for NVMe
**Priority: Low | Complexity: High | Status: DONE (v0.13.44–v0.13.51) — multi-reader is NEGATIVE on real NVMe; the win is a single O_DIRECT stream + zero-copy, shipped as `--direct-read`. See RESULT below.**

**The idea:** Open the input file N times, each reader seeks to offset `i * filesize / N`, reads its chunk in parallel. NVMe drives have deep internal queues and can serve multiple read streams simultaneously.

**Why it might work:**
- NVMe SSDs have 64-128 internal command queues
- A single `read()` thread can only keep 1 queue busy (queue depth 1)
- Multiple threads doing `pread()` at different offsets can saturate the device
- Measured NVMe sequential read: ~3-5 GiB/s single-thread, ~6-7 GiB/s theoretical max

**Why it might NOT work:**
- Linux readahead is already very good for sequential access
- Multiple readers cause random-ish access patterns from the NVMe's perspective (seeking between N positions)
- Page cache thrashing with N large read streams
- For compression: frames must still be processed in order (reader produces frames sequentially for the compressor)
- For decompression: the zstd frame boundaries must be found before parallel reading is possible (frames are variable-length in the compressed file)

**Decompression-specific challenge:** The compressed file has variable-length frames. You can't just split at byte offsets  you need to find frame headers. A pre-scan of the frame index (skippable frames or magic number search) could identify split points, but adds latency.

**Verdict (pre-research hypothesis):** Likely small gain for compression (reader is rarely the bottleneck  3+ GiB/s single-thread is usually enough). For decompression, the complexity of frame boundary detection likely outweighs the benefit. Worth benchmarking with a simple 2-reader prototype before committing.

**RESULT (v0.13.44–v0.13.51): researched and resolved — multi-reader is the WRONG approach on real NVMe; the win is a *single* O_DIRECT stream that does nothing but read.** Built as the `--direct-read` flag (O_DIRECT input — bypasses the page cache, so it's honest-cold every run with no eviction). Findings on the 256-core Gen4 box against a real 432 GiB file:

- **Concurrent readers CONTEND, they do not scale.** `dd` O_DIRECT: 1 stream = 4.5 GB/s; **4 independent streams = ~3.0 GB/s *aggregate*** (0.77 each) — slower combined than one. The "N parallel preads saturate the deep queue" premise is false here, so the v0.13.46/47 multi-threaded readers were reverted to a single stream (v0.13.49).
- **A single stream already saturates the drive (4.5 GB/s); the only job is to not stall it.** Levers that mattered, in order:
  1. **Zero-copy** (v0.13.49) — `pread` straight into a pooled aligned buffer handed to the worker as a `Task` view, eliminating the per-chunk 16 MiB `memcpy` that competed with the compressors for memory bandwidth.
  2. **mallopt mmap threshold** (v0.13.48) — frame buffers above glibc's 32 MiB ceiling were `munmap`'d per free, firing a TLB-shootdown IPI to every core (dominated sys time on 256 cores).
  3. **One large contiguous pool region** (v0.13.50) — many small `posix_memalign`s came from the fragmented heap, so O_DIRECT hit `max_segments=127` and shattered each 16 MiB read into ~340 KiB device requests; one big `mmap` faults as contiguous runs → ~1230 KiB requests (the device max). Net: stall → **4.08 GiB/s read-isolated**.
- **Read/write contention is physical, not a code bug.** Reads + writes on one drive share the NAND/controller (~1.9 R + 0.8 W = 2.7 GB/s mixed); the page cache only hides this for reads (RAM-resident), never for a sustained write. Go fast by reading and writing on **separate drives** (3.8 GiB/s) — confirms Phase 4.2's NVMe-write-ceiling verdict.
- **The decompression frame-boundary concern above was a non-issue:** O_DIRECT reads aligned blocks into a bounce buffer and the existing frame parser consumes from it unchanged (`stream_frames_to_queue`, v0.13.51) — works for both decompress paths.

**Net:** multi-reader shelved (negative result); the single-stream O_DIRECT zero-copy reader shipped as `--direct-read` for honest-cold benchmarking and one-pass reads that don't pollute the cache. On a big-RAM box the buffered/page-cache path is still the throughput king (reads served from RAM). The benchmark methodology was rebuilt around this: `gzstd-benchmark.sh` now reads cold via `--direct-read` and writes `/dev/null`; `gzstd-gendata.sh` builds a matching `.bin.zst` per profile. See CHANGELOG v0.13.44–v0.13.51.

### 4.2 Multi-Writer with pwrite()
**Priority: Low | Complexity: Medium | Status: RE-OPENED per-machine by v0.15.8** — the --adapt writer-parallelism probe tries +1 positional-pwrite drain thread on SINK_BOUND O_DIRECT runs, keeps on ≥10% measured gain, and persists the per-fingerprint verdict (buffered multi-writer remains negative by design; the probe is O_DIRECT-only).

**Already tested in v0.10.29:** 4 pwrite threads through page cache was 2.5× slower due to page cache thrashing (38 minutes sys time vs 12 minutes with O_DIRECT).

**Untested variant:** Multiple pwrite threads with O_DIRECT. Each thread opens its own fd with O_DIRECT, writes to non-overlapping aligned regions. This avoids page cache entirely. Requires knowing output frame sizes in advance (possible for decompression, not for compression).

**Risk:** O_DIRECT pwrite per-frame was catastrophic in v0.9.72 (27k individual pwrite calls). But with larger writes (batch of frames concatenated into one pwrite per thread), the overhead might be acceptable.

**Verdict:** Low priority. The NVMe write ceiling (~2-3 GiB/s) is the physical limit. Multiple O_DIRECT writers might get 10-20% more by keeping the NVMe queue deeper, but the complexity is high.

---

## Phase 5: Smart Defaults & Asymmetric Mode

### 5.1 Asymmetric Mode (GPU Compress + CPU Decompress)
**Priority: HIGH | Complexity: Low | Status: DONE (v0.13.0)**

Benchmark data from the workstation (v0.11.20) conclusively shows:
- **Compress:** GPU/Hybrid wins on 4/5 data types (up to 2.14 GiB/s vs 1.50 CPU)
- **Decompress:** CPU wins on ALL 5 data types (up to 4.88 GiB/s vs 3.50 hybrid)

On consumer GPUs with PCIe Gen3, the D2H transfer cost makes GPU decompression slower than CPU for every data type tested. The optimal strategy is:
- **Compress:** Use hybrid (GPU + CPU)
- **Decompress:** Use CPU-only

Implemented in v0.13.0: PCIe generation queried via NVML (with sysfs fallback). On Gen<4, decompress and `-t` default to `--cpu-only`. On Gen4+, default to `--hybrid`. User can override with `--gpu-only` or `--hybrid`.

Visible at `-v` as `[ASYMMETRIC] PCIe Gen3 detected; defaulting decompress to --cpu-only`.

### 5.2 PCIe Generation Detection
**Priority: High | Complexity: Low | Status: DONE (v0.13.0)**

Implemented as part of 5.1. Uses `nvmlDeviceGetMaxPcieLinkGeneration()` (Max, not Curr — idle GPUs drop their link to Gen1 for power management, which would mislead the heuristic). Fallback parses `/sys/bus/pci/devices/*/max_link_speed` when NVML isn't built in.

Map to decompress default:
- Gen<4: CPU-only decompress
- Gen4+: Hybrid decompress
- Detection unavailable: Hybrid (degrades gracefully)

### 5.3 Auto `--direct` (O_DIRECT output) for Gen4+ Compress & Decompress
**Priority: HIGH | Complexity: Low | Status: DONE (decompress v0.13.25, compress v0.13.26)**

O_DIRECT output is a large win on fast-fabric / high-core (PCIe Gen4+) boxes
where frame production outruns buffered writeback, scaling with output volume,
and a regression on Gen<4 (which stay buffered). It applies to both modes:
- **Decompress** is ~95% write-bound on disk — Gen4 compute ceiling ~14 GiB/s
  cpu-only (`-c >/dev/null`) vs ~0.68 GiB/s buffered (see Phase 7). O_DIRECT
  takes mixed `-d` ~0.68 → ~2.0 GiB/s (up to +130–230%).
- **Compress** benefits the same way (Gen4 server `--direct` data): cpu-only
  low +103% / mixed +50% / medium +15%, gpu-only +71% / +29% / +12%, hybrid
  +70% / +24% / +21%; tiny-output (high, zeros) neutral. No Gen4 regression.

`apply_backend_defaults()` auto-enables `--direct` on Gen4+ for **both** compress
and decompress (same `detect_min_pcie_gen()` probe, lifted above the compress
branch), unless the user passed `--direct`/`--no-direct`. Backend-independent
(the win is the output write path), so it covers cpu-only/hybrid/gpu-only alike.
Test mode writes nothing, so it's skipped. Visible at `-v` as
`[ASYMMETRIC] PCIe Gen4 detected; defaulting output to --direct`.

**Caveats:** compress output size is unknown, so the O_DIRECT path preallocates
`input_size` as an upper bound and `ftruncate`s down at finalize (handled;
`--no-preallocate` opts out). O_DIRECT can raise tail-latency variance (NVMe GC /
journal commits); medians favor it on Gen4.

**Note for benchmarking:** a Gen4 standard sweep now uses O_DIRECT for both
compress and decompress by default — pass `--no-direct` for the buffered
comparison. Tracks the read/writeback asymmetry recorded in the CHANGELOG and
memory.

**Input-side counterpart — `--direct-read` (O_DIRECT *input*, opt-in):** the read
analog of `--direct`, added v0.13.44–v0.13.51 (see Phase 4.1). Bypasses the page
cache on input → honest-cold every run + one-pass reads that don't pollute/evict the
cache. NOT auto-enabled (the buffered/page-cache read path wins on big-RAM boxes
where the input is resident); it's opt-in for cold benchmarking and the
`gzstd-benchmark.sh` methodology. Works for compress and decompress.

**Gen4+ regressions fixed (v0.13.31):** the default `--direct` exposed two issues
that only manifest where it auto-engages. (1) DirectWriter preallocate
(`fallocate`) defeated sparse output — fixed with a **punch-hole hybrid**
(`seek_forward` punches skipped zero runs back to holes; `write_sparse` coalesces
runs so it's one punch per run). Keeps preallocate's dense-write perf AND
sparseness. (2) The `--direct` auto-default log reused the `[ASYMMETRIC]` tag and
ran before the backend-user-set return, tripping the asymmetric tests under
explicit backends — retagged `[O_DIRECT]`.

---

## Phase 6: Testing & Hardening (v0.11.26v0.11.30, ongoing)

### 6.1 Comprehensive Test Suite
**Priority: HIGH | Complexity: Medium | Status: DONE (v0.11.26v0.11.30)**

`gzstd-test.sh`: ~170+ tests, live progress bar, per-test timing, auto GPU detection. Covers all CLI options, error handling, edge cases, VRAM pressure, and data integrity.

### 6.2 Structured Exit Codes
**Priority: Medium | Complexity: Low | Status: DONE (v0.11.26)**

0=OK, 1=runtime, 2=usage, 3=I/O, 4=data, 5=GPU_FAIL. Enables scripting and CI integration.

### 6.3 RAM Budget Check
**Priority: Medium | Complexity: Low | Status: DONE (v0.11.29)**

Auto-reduces chunk size to fit 75% of available RAM. Prevents OOM on memory-constrained machines.

### 6.4 Argument Hardening
**Priority: Medium | Complexity: Low | Status: DONE (v0.11.26v0.11.30)**

Unknown flags rejected, `--` end-of-options, `--threads=N` form, argument order independence, `.zst` double-compression warning, `--cpu-batch` ignored in `--cpu-only`.

---

## Phase 7: Code Review Findings (v0.13.23+)

Deep-read review of the full CPU/shared pipeline (TaskQueue, FrameThrottle,
writer thread + AsyncWritePool, HybridSched, both CPU workers, the frame
splitter, `main`, arg parsing, backend defaults, DirectWriter) plus a sampling
pass over the nvCOMP compress/decompress bodies. Each item below is
independently testable. **Validate every item on both a low-core PCIe Gen3 box
and a high-core PCIe Gen4 box** — the read/writeback balance differs enough
between them that a win on one can be a wash or regression on the other (same
asymmetry that governs `--mmap`/`--direct` defaults).

### 7.1 `AsyncWritePool::flush()` returned before the final write completed
**Priority: HIGH | Complexity: Low | Status: DONE (v0.13.23)**

`flush()` waited only on `pending_.empty()`, but the worker empties `pending_`
by *moving* the batch out before it writes it. So `flush()` could return while
the last batch was still being written; a write failure there (disk full, EIO,
broken O_DIRECT tail) set `error_` only after the single `had_error()` check in
`writer_thread`, so the run reported success (exit 0) and the atomic `rename`
proceeded over truncated/corrupt output. Mid-stream errors were caught one batch
late by the `had_error()` check inside `submit()`; only the *final* batch was
exposed — i.e. exactly the disk-full-at-the-end case.

**Fix:** added a `writing_` flag (guarded by the pool mutex) set true when the
worker dequeues a batch and cleared after the batch is physically written (and
on the error-return path, with a notify). `flush()` now blocks on
`pending_.empty() && !writing_`, making the post-`flush()` `had_error()` check
reliable.

**Test:** point output at a tiny tmpfs / quota'd directory sized so the *last*
batch write is the one that fails; confirm a non-zero `EXIT_IO` (3) and that no
successful `rename` occurs. Round-trip integrity across all five data profiles
must be unchanged; compress/decompress throughput should be unaffected (no
hot-path change).

### 7.2 GPU result buffers allocate fresh per frame (no recycled pool)
**Priority: HIGH | Complexity: Medium | Status: DONE (decompress v0.13.24, compress v0.13.33)**

The CPU workers recycle a bounded `FrameBuf` pool (the v0.13.7/v0.13.8 fix for
the per-iteration alloc + page-fault storm). The GPU completion paths do not —
each finished frame does a fresh `make_shared<std::vector<char>>(size)`
(`compress_nvcomp` async-poll + sync-drain paths; `decompress_nvcomp` D2H push).
On the **Gen4 hybrid-decompress path** `size` is the full decompressed frame
(~16 MiB), so every frame is a fresh ~16 MiB allocation + fault — the same storm
the CPU pools were built to eliminate, never ported to the GPU side. This is the
highest-value perf lead and it sits on the fast-fabric path that actually runs
hybrid decompress.

**Done (v0.13.24, decompress):** `DecompStreamCtx` now owns a recycled `out_pool`
(`acquire_out_buf()`, `use_count()==1` reclaim, lazy growth to two batches, waits
on the writer's drain signal past the cap). Deadlock-free by the same FIFO
argument as the throttle. Gen3 proxy (`--gpu-only -d`, 2 GiB mixed): minor-faults
636k→538k (−15%), peak RSS 2.57→2.26 GiB (−12%); 253/253 tests pass; round-trip
verified on `--gpu-only`/`--hybrid`.

**Still to test (Gen4):** this is the hybrid-decompress *default* on Gen4+, where
batches are larger and frames cycle faster, so the win should be larger than the
Gen3 proxy. Benchmark `--hybrid -d` on `mixed`/`low` vs v0.13.23 (throughput +
`/usr/bin/time -v` faults/RSS). `perf stat` needs `perf_event_paranoid` lowered;
`/usr/bin/time -v` works unprivileged and is what the Gen3 numbers above used.

**Done (compress, v0.13.33):** `StreamCtx` got the same recycled `out_pool`; both
`gpu_worker` completion paths use it. Lower-value (compressed output is small) but
removes the per-frame alloc churn; round-trips verified, 213/213. 7.2 fully closed.

### 7.3 Throttle budget computed from the unresolved chunk size (compress)
**Priority: Medium | Complexity: Low | Status: DONE (v0.13.28)**

`compress_cpu_mt` now sizes the `FrameThrottle` from the resolved `host_chunk`
(= `chosen_mib`) instead of `opt.chunk_mib`. Verified at `-v`: `--ultra -22 -T4`
reports a 4.00 GiB in-flight cap (32 × 128 MiB) vs the old 512 MiB. GPU compress
already used `chosen_mib`; decompress paths stay heuristic (frame size unknown
until the stream is parsed). 259/259 tests pass.

`compress_cpu_mt` builds the `FrameThrottle` from `opt.chunk_mib * ONE_MIB`, but
the frame size actually used is `host_chunk` (= `chosen_mib`), which can be
auto-bumped for `--ultra` or shrunk by `check_ram_budget`. For the default path
they're equal; on ultra / low-RAM runs the in-flight RAM cap is computed against
a stale 16 MiB and can over- or under-shoot. Pass the resolved `chosen_mib`.
(`decompress_cpu_mt` has the same stale-`opt.chunk_mib` argument, but there the
true frame size isn't known until after streaming, so it stays a heuristic.)

**Test:** `--ultra -22 --chunk-size`-unset compress; verify peak RSS stays within
the intended in-flight cap (watch `[THROTTLE]` at `-v`/`-vv`) and that throughput
is unchanged at default settings.

### 7.4 Redundant memcpy of every compressed frame (CPU compress)
**Priority: Medium | Complexity: Medium | Status: DONE (v0.13.32) — kept; throughput within noise**

Conditional swap: when `csz >= in_size/2`, `cpu_worker` swaps the scratch buffer
into the pooled FrameBuf (zero-copy) instead of memcpy; small output keeps the
memcpy (avoids inflating pool slots to compressBound). **Gen5 benchmark verdict:**
throughput change is within run noise — cpu-only `low` compress (the only profile
that crosses the threshold) +4%, same as the memcpy-path controls and the
untouchable paths (gpu-only compress, all decompress) which also swing ±6%. The
eliminated memcpy (~14 MiB, ~1 ms) is ~1–2% of per-frame compress time at level 3,
below the noise floor. Kept anyway: the old memcpy was pure data-shuffle overhead,
so the swap does strictly less work, is correct, and the RSS overhang is negligible
(only `low` swaps; its csz ≈ 14.4 MiB is near the 16 MiB compressBound, ~1.6 MiB/
slot). Leaner path, no regression.

`cpu_worker` compresses into a per-thread `scratch`, then copies `csz` bytes into
a pooled `FrameBuf`. For low-compressibility data (`mixed`/`low`) `csz ≈ chunk`,
so it's a full-frame memcpy per frame on exactly the profiles where compress is
slowest. Compressing straight into the pooled buffer eliminates the copy **but**
inflates every pool slot to `compressBound` capacity, which on a high-core box
with a deep throttle is a large RAM regression for *compressible* data — so the
current scratch+copy is a deliberate trade. A targeted version (e.g. `swap`
scratch into the slot only when `csz` exceeds a fraction of the chunk) could
capture the incompressible-data win without the memory hit.

**Test:** prototype the conditional swap; benchmark compress on all five profiles
on both box classes, watching both throughput *and* peak RSS. Only land if
`mixed`/`low` improve with no RSS regression on `high`/`zeros`. Do not change
blindly.

### 7.5 `--sync-output` is a no-op under `--direct`
**Priority: Low | Complexity: Low | Status: DONE (v0.13.30)**

`main` now fsyncs the DirectWriter's own fd when `sync_output` is set (the
`FILE* out` is closed/nulled under `--direct`, so the buffered fsync path never
ran). Confirmed via strace: `--direct --sync-output` issues one fsync, `--direct`
alone issues none.

When the O_DIRECT writer owns output, the `FILE* out` was closed and nulled, so
the `if (out) fsync(out)` branch in `main` is skipped. O_DIRECT data is durable
but the `ftruncate`-set size metadata isn't fsync'd. If a user pairs
`--direct --sync-output` expecting durability they don't get the fsync. Add an
`fdatasync(dw->fd())` in `DirectWriter::finalize()` (or before close) when
`sync_output` is set.

**Test:** `--direct --sync-output`, then inspect with `strace`/`fsync` that the
sync actually fires; functional round-trip unchanged.

### 7.6 `is_all_zero` does an unaligned `size_t` load
**Priority: Low | Complexity: Low | Status: DONE (v0.13.30)**

Replaced the unaligned `reinterpret_cast<const size_t*>` with a constant-size
`memcpy` into a `size_t` (same wide load on x86, portable on strict-alignment
targets).

`reinterpret_cast<const size_t*>(p)` where `p` is `vector<char>::data()` — fine
on x86, UB on strict-alignment targets. Cosmetic for current hardware; use
`memcpy` into a `size_t` (compiles to the same load on x86) to keep it portable.

### 7.7 `SequentialDispatcher` appears unused
**Priority: Low | Complexity: Low | Status: DONE (v0.13.30)**

Verified dead (type + methods appeared only in its own definition; superseded by
the per-GPU result slots in v0.11.11) and removed — ~46 lines.

Defined but no caller was found in the reviewed paths (GPU workers use
`pop_batch_greedy(min_n=1)` directly). If a full-file grep confirms it's dead,
delete it (~40 lines of concurrency surface removed). Verify before removing.

### 7.8 Decompress reader has no queue-depth backpressure (slow-consumer RSS blowup)
**Priority: Medium | Complexity: Medium | Status: DONE (v0.13.29); the original hybrid-fault hypothesis was disproven (see Update below)**

Fixed in v0.13.29: `TaskQueue` gained an optional `max_depth_` (0 = unbounded);
`push()` blocks on `space_cv_` when full, woken by the pop paths a bounded queue
uses + `set_done()` (`re_enqueue`/push_front bypasses it). Both decompress paths
cap the queue at `max(THROTTLE_MIN_FRAMES, parallelism * slack)`, so queued RAM is
O(pipeline) not O(input) — skipped under `--no-throttle`. Cap is ≥ the auto-tuner's
batch needs (no throughput risk). Verified: no deadlock across cpu/gpu/hybrid
round-trips; slow-consumer gpu-only `-d` of a 3 GB / ~2861-frame input held
1.79 GiB RSS (capped, not whole-input). Compress queues stay unbounded (mmap input
is zero-copy). 259/259 tests pass.

Surfaced by the v0.13.24 Gen4 validation. On a 256-core Gen4 box decompressing a
9.75 GiB → 19.53 GiB mixed file (buffered output, no `--direct`), the three
backends fault very differently:

| backend | minor-faults | sys time | vol ctx-switches |
|---|---|---|---|
| cpu-only | 5.03M (≈ output pages) | 38.9s | 25k |
| gpu-only | 6.04M (+0.9M) | 46.6s | 38k |
| hybrid   | 8.46M (+3.4M) | 57.0s | 199k |

~5.12M of those faults are just the 19.53 GiB output landing in the page cache —
cpu-only sits right at that floor, confirming the v0.13.8 + 7.2 buffer pools
leave ~zero excess allocation faults. But hybrid carries +3.4M faults and 8× the
voluntary context switches over that floor: excess allocation/blocking specific
to the hybrid path. It's currently masked in throughput (hybrid decompress is
~flat vs baseline), but it's a real efficiency drain that will bite on
larger/faster runs.

Likely suspects (profile to confirm):
- **Unpooled input `Task.data`:** `stream_frames_to_queue` reads each compressed
  frame into a fresh `std::vector<char>` (~9.75 GiB of allocation total), shared
  by all backends but interacting worst with hybrid's two consumer pools.
- **Scheduler / trivial-frame re-enqueue churn** (frames bouncing GPU→CPU).
- The 199k voluntary context switches point at CV/lock blocking (throttle drain,
  `cpu_cv_`, `gpu_got_data` wakeups), not just faults.

Caveat: most of *this* run's faults and sys-time are the buffered-write storm
that `--direct` eliminates (the read/writeback asymmetry tracked in the CHANGELOG
and memory). Isolate hybrid's excess by re-running to `/dev/null` or with
`--direct` so the write path doesn't dominate; pursue only if the isolated
numbers still show a hybrid-specific gap.

**Update — isolated with `-c >/dev/null` (Gen4, write path removed):** the
hybrid-fault hypothesis is **disproven**. Hybrid faults collapse 8.46M → 699k,
essentially identical to cpu-only (686k) — the +3.4M was the buffered-output
page-cache storm, not allocation or scheduling. Headline: **decompress is ~95%
write-bound.** Removing the write took throughput cpu-only 0.68 → 13.94, hybrid
0.67 → 8.13, gpu-only 0.58 → 2.64 GiB/s (compute ceilings). The dominant
decompress lever is the write path / `--direct`, not compute — so 7.2-style
buffer pooling is correct but second-order on disk-backed runs.

The isolation *did* expose a real issue: **gpu-only decompress holds 11.3 GiB
RSS and 4.6M faults** (vs ~1.9 GiB / ~690k for cpu-only and hybrid). The GPU
consumer (2.64 GiB/s, D2H-bound) is far slower than the reader, and
`stream_frames_to_queue` buffers the entire ~9.75 GiB compressed input into
queued `Task.data` vectors. The FrameThrottle bounds *popped-but-unwritten*
frames; nothing bounds queue depth *ahead* of the workers, so a slow consumer
lets the queue grow to the whole input — bounded here (fits RAM) but a latent
OOM on 100s-of-GiB gpu-only decompress.

**Reframed fix:** cap reader/queue depth (block the reader above N queued
frames) so RSS is bounded by pipeline depth, not input size. Compress already
avoids this via zero-copy mmap (`view_ptr`); decompress copies frames into
`Task.data`, so it needs an explicit cap. Low urgency — gpu-only decompress is
opt-in and not the Gen4 default (asymmetric routes Gen4 decompress to hybrid,
which is unaffected) — but worth it before any TB-scale gpu-only decompress.

### 7.9 Support bundled short flags (`-dc`, `-dk`, …) for zstd/gzip compat
**Priority: Medium | Complexity: Medium | Status: DONE (v0.13.27)**

Implemented via a pre-pass in `parse_args` that expands a bundled group into
individual flags before the match loop. Conservative scope: a group expands only
when every char after a single leading `-` is a no-arg operation flag
`{d,t,k,f,c}`; everything else (value flags `-o`/`-T`/`-M`/`-B`/`-D`, numeric
levels, attached-value `-T4`/`-M512`/`-b3`, repeat flags `-vv`/`-vvv`/`-qq`, long
options, `--`/`-`) passes through unchanged. `v`/`q` excluded so repeat semantics
survive — bundle verbosity separately (`-d -vv`). The original analysis below
documents the edge cases this handles.

`parse_args` exact-matches each argv token (`a == "-d"`, `a == "-c"`, …), so a
bundled short-flag group like `-dc` is rejected with `unknown option: -dc`
(exit 2). But `zstd -dc` and `gzip -dc` both accept it, and gzstd advertises
itself as a "drop-in-compatible replacement for the zstd CLI" — so this is a real
compatibility wart. It bites in the common idioms users carry over from
zstd/gzip (e.g. `gzstd -dc archive | tar -xf -`, `-dk`, `-df`).

**Fix:** before the match loop, expand a leading `-<chars>` token (where every
char is a known no-arg short flag) into individual flags. Edge cases to handle —
these are why it's not a one-liner:
- **Value-consuming flags** stop the bundle: `-o FILE`, `-T N`, `-M N`. Pick a
  rule (e.g. stop expanding at the first value-flag, treat the rest of the token
  as its attached value, like `zstd -T4`).
- **Numeric levels** (`-19`, `-1`): already a separate branch; a bundle must not
  swallow digits as flags (`-9` is a level, not flag `9`).
- `-T0` (all threads) and repeat-count flags (`-qq`, `-vv`, `-vvv`) must survive.

Discovered while debugging a `gzstd -dc` invocation (2026-06; the bundled form
silently failed in a test pipeline). Low risk once the value-flag rule is fixed,
high compat value.

### 7.10 Not-yet-audited areas — next deep-dive targets
**Priority: Medium | Complexity: High | Status: DONE — all three targets audited (auto-tuner + failure rescue v0.13.34, HybridSched v0.13.35)**

**Audit results (3 of 3 targets done):**
- **GPU auto-tuner — audited + dead code removed.** The live tuner is the
  cross-GPU `SharedTuneState` (BASELINE→HALVE/DOUBLE→REFINE→SETTLED); no races or
  convergence bugs found in it on this pass.  The per-stream
  EXPLORE/REFINE/SETTLE hill-climb (`TuneState` + `tune_*`/`refine_*` fields in
  both the compress `StreamCtx` and the decompress per-stream struct, plus the
  save/restore across buffer reallocation) was **dead code** — superseded by
  `SharedTuneState`, never read by any decision path — and is now removed (~56
  lines), same class as the 7.7 `SequentialDispatcher` removal.
- **VRAM-exhaustion + GPU-failure rescue — audited, two bugs fixed.**  (1) Both
  catch blocks leaked one FrameThrottle permit per rescued/re-enqueued frame
  (handed frames off without releasing; the receiver re-acquires) — up to
  `streams × per_stream_batch` per device failure, enough to deadlock the
  survivors.  (2) The compress catch guarded on `C.busy`, which isn't set until
  after the throwing H2D/compress-launch calls, so a launch failure stranded the
  just-popped batch and hung the writer.  Both fixed; see CHANGELOG v0.13.34.
  **Superseded (v0.14.43):** the COMPRESS rescue was deleted entirely — a faulting
  GPU's output is untrusted, so the pass now aborts cleanly and the driver rebuilds
  CPU-only (no mid-run salvage to leak permits from).  The DECOMPRESS rescue stays
  (a faulted GPU there is finished on CPU and the output is kept and correct).

- **`HybridSched` corner cases — audited (v0.13.35), clean + one robustness fix.**
  No deadlock (fixed-mode `should_cpu_take`/`should_gpu_take` can't both be
  false), no missed-wakeup (`push` wakes one CPU/task, exit paths `notify_all`),
  floor enforced atomically in `may_take` and correctly skipped in fixed mode,
  `gpus_waiting_` balanced.  Fixed: in fixed `--cpu-share` mode with no active GPU
  (still initializing, or all failed), the share cap stalled the main CPU workers
  for the whole production phase; `should_cpu_take` now short-circuits to `true`
  when `active_gpu_streams_ == 0`.  Remaining cosmetic note: the ±0.02 hysteresis
  leaks ~2% to the opposite engine at `--cpu-share` 0.0/1.0 (not worth fixing).

Background: the Phase 7 review (v0.13.23–v0.13.33) had read the shared/CPU
machinery line-by-line — `TaskQueue`, `FrameThrottle`, `AsyncWritePool`/
`writer_thread`, `ResultStore`, the CPU compress/decompress workers,
`stream_frames_to_queue`, `DirectWriter`, `main`, `parse_args`,
`apply_backend_defaults` — and only **sampled** the ~3,200-line nvCOMP bodies.
The three areas left un-audited then (GPU auto-tuner, VRAM/GPU-failure rescue,
`HybridSched`) are the ones resolved in the audit-results block above.

Thinner ice still worth a second look: punch-hole + O_DIRECT was only validated on the
test boxes' filesystems (ext4-class) — confirm on others (xfs/btrfs/zfs) before
assuming.  **(Still open — no xfs/btrfs host available to test; needs a loopback
image or a CI matrix.)**

~~the 7.8 reader queue-cap is deliberately conservative (`parallelism * slack`)~~
**DONE (v0.13.40):** the 7.8 frame-count cap held RAM proportional to
compressibility (incompressible ~4× compressible).  Added a parallel byte cap to
`TaskQueue` (`max_bytes_`/`queued_bytes_`, centralized in `take_front_locked`):
reader blocks on `frames>=floor OR bytes>=budget`, budget = `floor*8 MiB`, with a
`!q_.empty()` deadlock guard and mmap-view-aware accounting (`data.size()`).
Measured gpu-only decompress RSS −8…−11% (145–225 MiB on 4 GiB), throughput
flat.  Tunable via `--throttle-factor`; flagged for validation on the Gen4
8×H100 server (reduced
big-frame buffering on a faster reader/consumer ratio).  See CHANGELOG v0.13.40.

**Extended to the compress producer (v0.13.41):** compress had the same exposure
on **pipe/stdin** input (fread → heap; the compress queue was uncapped — a
producer outrunning the workers could buffer the whole input → OOM).  Regular
files were always safe (mmap = zero-copy views, no heap, so a 1 TB file streams in
bounded RAM).  Same byte cap now set on both compress queues; no-op for mmap.
Demonstrated −75% peak RSS (2232→568 MiB) on a slow-worker piped incompressible
run, throughput unchanged.  See CHANGELOG v0.13.41.

### GPU compress D2H readback `resize()` zeroing — CHECKED negligible, then fixed cleanly (closed)

v0.13.36 removed the per-frame output-buffer zero-fill on the **CPU** compress
path (`compress_one_cpu_frame`'s `resize(bound)`→`resize(csz)` shrink-regrow
cycle).  The **GPU** compress D2H readback (`gpu_worker`, both the async-poll and
sync-drain completion paths) does the analogous `h_out->resize(csz)` before the
`cudaMemcpy` D2H.  Reasoned negligible — unlike the CPU path it has no forced
shrink-regrow cycle, so the pooled `out_pool` buffers self-stabilize at the
steady compressed size and `resize(csz)` becomes a no-op after warm-up (zeroing
only the upward `csz` *variation*).

**Measured on the Gen4 8×H100 server, `gpu-only`, 8 GiB mixed data (worst case for csz
variation), `perf --call-graph dwarf`:** `__memset` via
`_M_default_append`←`resize`←`gpu_worker` = **0.59%** of host CPU self-time —
below the 1% threshold, and on a path whose host CPU isn't the bottleneck anyway
(dominated by the NVIDIA driver spinlock, `_raw_spin_lock_irqsave` 15.9%, waiting
on the GPU; `gpu_worker` self-time 0.01%).  Throughput-irrelevant, and the
invasive `FrameBuf` default-init-allocator change was NOT justified for it.

**Update (v0.13.37): fixed cleanly anyway, without the allocator.** The pinned
D2H path already stages the bytes through a host slot (`pin_slot`), so
`h_out->assign(pin_slot, pin_slot+csz)` replaces `resize(csz)+memcpy` — same
copy, but `assign` copy-constructs from the source instead of value-initializing
then overwriting, so the zero-fill is gone.  Applied to the CPU memcpy branch and
both GPU pinned completion paths (async-poll + sync-drain).  The GPU **non-pinned**
direct-D2H fallback keeps `resize()` (dst must be pre-sized before `cudaMemcpy`;
`assign` can't source from device memory) — slow fallback.

**Update (v0.13.39): the default-init allocator was adopted after all** — not for
this 0.59% GPU residual, but because the same audit found the **CPU decompress**
resize-zero (`ZSTD_decompressDCtx` writes direct, so `assign` can't help) at **~16%
of instructions** (large buffer pool → most frames grow a fresh full-frame buffer).
`FrameBuf` now uses `default_init_allocator<char>`, which also mops up the remaining
direct-write resize-zeros (CPU decompress, both non-pinned D2H paths).  Note: it's
**throughput-neutral** (the memset was parallel/overlapped, not the wall-clock
bottleneck) — kept as resource-waste elimination (fewer cycles + less memory-write
traffic), not a speedup.  See CHANGELOG v0.13.39.

<details><summary>Reproduction runbook (perf on the Gen4 server)</summary>

```bash
# 1. perf needs paranoid <= 2 for unprivileged sampling; check then (if needed) lower:
cat /proc/sys/kernel/perf_event_paranoid           # if > 2:
sudo sysctl kernel.perf_event_paranoid=1           # (run via `! sudo …` in-session)

# 2. mixed data MAXIMIZES csz frame-to-frame variation = worst case for the resize-grow.
#    Output to /dev/null so disk I/O doesn't dominate the profile.
dd if=gzstd-testdata/mixed.bin of=/tmp/gpin bs=1M count=8192 status=none

# 3. record with DWARF call graphs (needed: hot frames are in libc/libzstd;
#    --call-graph dwarf uses .eh_frame CFI, works on the -O3 release build).
perf record -g --call-graph dwarf -o /tmp/gpu.perf \
    ./build/gzstd --gpu-only -c -f /tmp/gpin > /dev/null

# 4. find __memset_avx2 and WHO calls it:
perf report --stdio -g -i /tmp/gpu.perf | grep -B2 -A25 memset_avx2 | head -60
```
Decision rule: if `__memset` is < ~1% **or** its callers are all nvcomp/cuda/
libzstd (not `std::vector::_M_default_append` / `acquire_out_buf` / `gpu_worker`),
the readback zeroing is confirmed negligible — close this and do nothing.  If
`_M_default_append` under the GPU readback shows meaningful self-time, justify the
`FrameBuf` default-init allocator change (it would also make the CPU path's
one-time first-frame zeroing free).  Optional confirmation: A/B `gpu-only`
throughput (`-c >/dev/null`, best-of-5) before/after the allocator change.

</details>

---

## Remaining Work for v1.0

| Item | Phase | Priority | Status |
|------|-------|----------|--------|
| Streaming decompression output | — | HIGH | DONE (v0.12.24) |
| Asymmetric mode (PCIe Gen3 detection) | 5.1, 5.2 | HIGH | DONE (v0.13.0) |
| Auto --direct for Gen4+ compress & decompress | 5.3 | HIGH | DONE (decompress v0.13.25, compress v0.13.26) |
| Persistent auto-tuning (`~/.gzstd/`) | 2.12.3 | Medium | Not started |
| Rate-matched dispatch (re-enable) | 1.3 | Medium | Disabled, needs eval |
| Pipe-aware scheduling | 3.1 | Medium | Not started |
| Streaming mode for unknown-size input | 3.2 | Low | Not started |
| Multi-reader NVMe | 4.1 | Low | Research |
| Multi-writer O_DIRECT pwrite | 4.2 | Low | Tested negative for buffered |
| AsyncWritePool flush() final-batch error | 7.1 | HIGH | DONE (v0.13.23) |
| GPU result buffer pool (compress + decompress) | 7.2 | HIGH | DONE (decompress v0.13.24, compress v0.13.33) |
| Throttle budget uses resolved chunk size | 7.3 | Medium | DONE (v0.13.28) |
| CPU-compress redundant memcpy | 7.4 | Medium | DONE (v0.13.32) — kept; throughput within noise |
| --sync-output under --direct | 7.5 | Low | DONE (v0.13.30) |
| is_all_zero unaligned load | 7.6 | Low | DONE (v0.13.30) |
| Remove dead SequentialDispatcher | 7.7 | Low | DONE (v0.13.30) |
| Decompress reader queue-depth cap (gpu-only RSS blowup) | 7.8 | Medium | DONE (v0.13.29) |
| Bundled short flags (-dc, -dk) for zstd/gzip compat | 7.9 | Medium | DONE (v0.13.27) |
| `--verify` — background decompress-verify on compress (untrusted GPU) | — | Medium | DONE (v0.14.39–40) |
| `--keep-going` — recover a damaged archive on decompress | — | Medium | DONE (v0.14.41–42) |
| Delete compress CPU-rescue → clean GPU-fault abort | — | Medium | DONE (v0.14.43) |
| Checkpoint/resume on fault (resume from last good frame vs. rebuild from zero) | — | Low | Not started |
| `--tar` input ergonomics (`--exclude-from`/`-X`, `--files-from`, `-P`, `--exclude-vcs`) | tar | Low | DONE (v0.14.90) |
| `--selinux` context storage (third leg of xattrs/ACLs) | tar | Low | DONE (v0.14.91) — spot-check a labeled-host round-trip if one appears |
| Restore xattrs/contexts on symlinks & special files (extract side) | tar | Low | Open — stored on create but silently not reapplied (apply_ext is fd-based; needs lsetxattr via the secure parent-fd walk); inherited gap affecting --xattrs AND --selinux, GNU tar restores these; documented in --help since v0.14.91 |
| Hoist the pax record-grammar walk into a shared for_each_pax_record | — | Low | DONE (v0.14.92) — grammar walk single-sourced; per-caller key dispatch deliberately kept local (premature-abstraction verdict) |
| Seek table on PLAIN (non-tar) compress output | — | Low | DONE (v0.14.92) — all compress paths emit it (cpu/gpu/hybrid/serial/stdin verified; --sliding-window and --no-index excluded); self-validating geometry so a wrong table can never be emitted |
| Warm/cold-adaptive `-l` fallback walk (mincore dispatch → buffered pread walk) | — | Low | DONE (v0.14.93) — warm 65 GiB single frame 3.0 s → 1.29 s (zstd-class; faults 560k → 4.4k); cold keeps the mmap+SEQUENTIAL walk (still beats zstd's QD1 strided reads 38.7 s vs 42.4 s); buffered walk is strictly validated, bails to the mmap walk on anything unmodeled |
| Cold `-l` walk: batched posix_fadvise(WILLNEED) header prefetch | — | Low | Open — the remaining half: prefetch upcoming header offsets at deep queue so the cold walk reads ~2 GiB of header pages instead of streaming 65 GiB (should beat both tools cold); build on v0.14.93's buffered_frame_walk; validate with the cold umount methodology and scripts/drop_cache |
| Parallelize the `--tar` layout walk | tar | Medium | DONE (v0.14.9) — the lstat storm (Pass B) runs parallel; serial Pass A enumerate is the residual, unmeasured |
| Cache-bypass member reads on `--tar` create (FADV_DONTNEED vs O_DIRECT) | tar | Low | Investigate + measure |
| O_DIRECT extraction writes for large files (Gen4+) | tar | Low | Investigate + measure |
| Punch-hole + O_DIRECT on xfs/btrfs/zfs (validated ext4-class only) | 7.10 | Low | Open — needs loopback image or CI matrix |

### Versioning plan (as of v0.14.95, 2026-07)

- **0.14.x — the `--tar` chapter is CLOSED as of v0.14.91.** Parallel
  create/extract including the parallel-lstat layout walk, member index +
  instant `-l`, seek-based selective extraction, zstd seekable-format interop
  (including header-hop `-l` for foreign archives), parallel-dispatch full
  extraction, sparse files, xattrs/ACLs/SELinux, and the v0.14.90
  input-ergonomics flags. Remaining tar-adjacent work is opportunistic only:
  the two measure-first O_DIRECT/cache probes, the xfs/btrfs/zfs punch-hole
  validation, and an SELinux labeled-host round-trip spot-check.
- **The 0.14.x line itself is closed as of v0.14.95.** v0.14.94 fixed the
  disk-full field report (the `--direct` permit-starvation hang AND the worse
  buffered silent-success data loss); v0.14.95 was the deliberate close-out: a
  three-angle sequential review (concurrency hangs → data correctness →
  help/parse accuracy) that fixed two pooled-reader deadlocks, the
  malformed-tar SIGABRT, extract exit-code fidelity (corrupt archive → exit 4),
  the rename-fallback silent truncation, and a full help/parse audit. See
  CHANGELOG v0.14.94–95.
- **0.15.0** opens the next big change — likely `--adapt` (AI/heuristic runtime
  self-tuning; regime-signal instrumentation already built). The unblocked first
  slice is independent of the full design: **the decompress default backend is
  wrong on Gen4+** (picks cpu-only where GPU wins — see 1.11). Note Phase 2
  (persistent auto-tuning), 1.3 (rate-matched dispatch), and 3.1/3.2 (pipe-aware
  scheduling) are exactly what `--adapt` would subsume — decide there, not
  piecemeal.
- **v1.0** when the chosen 0.15 scope is polished and proven. Whether `--adapt`
  is *in* v1.0 or lands after is deliberately still open.

### Data integrity & recovery (v0.14.39–43)
**Status: DONE (verify, keep-going, rescue removal); checkpoint/resume NOT started**

A faulting GPU is an untrusted producer (see CHANGELOG v0.14.38).  `--verify`
(compress) decompress-verifies every frame in the background and rebuilds CPU-only
on any mismatch; `--keep-going` (decompress) recovers what it can from a damaged
archive and reports the affected files / byte ranges; and the now-pointless
compress rescue machinery was deleted in favor of a clean abort-and-rebuild.  See
CHANGELOG for each.

Remaining: on a fault, gzstd rebuilds the whole archive from frame zero.  A
checkpoint/resume would decompress-verify the written prefix, find the last good
frame, and resume CPU-only from there — a large win for a fault near the end of a
multi-TB archive.  Fixed-size chunking makes the resume offset trivial; `--tar`
re-walks its layout.  Not started.

### Streaming Decompression Output
**Priority: HIGH | Complexity: Medium | Status: DONE (v0.12.24)**

gzstd decompresses frame-at-a-time: each worker allocates a full-frame output buffer, decompresses into it, then hands it to the writer. For oversized single-frame files (from `zstd -T0` or `--sliding-window`), this meant allocating the entire decompressed size (e.g., 125 GiB) as one buffer — no progress, no backpressure, massive memory spike.

**Fix (v0.12.24):** For frames > 64 MiB, `cpu_decomp_worker` uses `ZSTD_decompressStream` with 16 MiB output chunks. Each chunk gets its own ResultStore sequence number (`total_tasks` adjusted upward). The writer starts writing as soon as the first chunk arrives, so progress bar tracks smoothly and memory stays at ~16 MiB working set.

Normal multi-frame files (16 MiB frames from gzstd's default path) are unaffected — they use the existing `ZSTD_decompressDCtx` fast path.

**Note:** The broader proposal of streaming ALL frames (including normal 16 MiB ones) through a small ring buffer remains a potential future optimization for overwrite workloads, but the acute problem (single-frame files) is solved.

---

## Future Ideas (v2.0+)

### Speculative CPU/GPU Racing
Submit the same frame to both CPU and GPU. Take whichever finishes first, discard the other. Skip GPU D2H transfer when CPU wins.

**Pros:** Optimal for mixed-compressibility data. Minimal cost on 256-core machines (2 speculative CPU threads = <1% overhead).

**Cons:** Fights the auto-tuner (GPU sees stolen frames as lower throughput). Memory pressure (frame exists twice). GPU slot wasted even when CPU wins (H2D already happened). Significant architectural complexity.

**Verdict:** Rate-matched dispatch (Phase 1.3) gets 90% of the benefit with 10% of the complexity. Revisit if benchmarks show specific data patterns where prediction fails.

### Compression-Aware Frame Routing
Sample first few KB of each frame for entropy estimation. Route high-entropy (incompressible) frames to CPU (avoid PCIe overhead). Route low-entropy (highly compressible) frames to GPU (kernel is fast, D2H is small).

**Status:** Partially implemented  trivial frame detection (ratio < 2%) routes to CPU for decompress. Could extend with entropy sampling for compress.

### Network-Distributed Decompression
For truly massive files (TB+), distribute frames across multiple machines. Each machine decompresses its assigned frames and writes to a shared filesystem or sends results back. gzstd could act as coordinator.

**Verdict:** Out of scope for v1.0. Would require significant architectural changes (network protocol, fault tolerance, frame assignment).

---

## Native tar archiving (`--tar`, v0.14.0+)

`--tar` builds a GNU-format `.tar.zst` directly, reading members in parallel and
feeding the existing CPU/GPU pipeline (replaces the single-threaded `tar -cf - |
gzstd` bottleneck). Creation **and** extraction (`-d --tar [-C DIR]`) shipped in
v0.14.0 — extraction overlaps parallel decompression with a file-writer pool and
is secured with `openat`/`O_NOFOLLOW` against path-traversal and symlink-escape.
Since v0.14.19/20 both `-t --tar` (verify) and `-d --tar` (extract) feed the
in-order decompressed frames to the tar parser **in memory** (`FrameSink`), not
through a kernel pipe — verify's `skip()` over file data became pointer
arithmetic (decompress-bound now), and extract dropped the per-byte pipe copy +
syscalls. v0.14.23 made the O_DIRECT large-file writer **persistent across files**
(was spawned/joined per file), so a file's tail writes overlap the next file's
parse/read instead of stalling the parse thread on a per-file join — a structural
win that grows with file count. The tar **parse itself is still serial** on both
paths (one thread walks the stream in order). Open follow-ups:

- **Parallel-dispatch `-d --tar` extraction** — *Priority: Medium | Complexity:
  High | Status: DONE (v0.14.86) — full no-selection extraction now parses +
  dispatches in parallel; serial walk kept as the always-correct fallback.* See
  the resolution note directly below; the design record is retained for context.

  **DONE (v0.14.86):** `Extractor::run_parallel` splits the entries (from our
  index or a foreign header-hop scan) into N contiguous partitions; each worker
  `pread`s + decompresses only its frames (shared `decode_seek_frame`) and runs
  the standard `parse`/`handle_entry` over its slice via a new `StreamReader`
  producer source with a partial-slice limit (clean EOF at the partition's final
  entry boundary), dispatching to the one shared writer pool. The hard parts
  resolved as: **dir-creation ordering** — no upfront pass needed; `mkdirat` is
  atomic and `open_parent(create)` treats `EEXIST` as success, so concurrent
  implicit/explicit dir creation is race-free and true dir mode/ext is reapplied
  once in `finish_deferred`; **hardlink/deferred ordering** — collected into
  per-worker `ParCtx` lists, merged in partition (archive) order after join, so
  hardlinks are created after their targets exist and dir metadata is applied in
  reverse order; **security** — the `openat`/`O_NOFOLLOW` walk is stateless and
  runs per worker unchanged (`map_owner` cache mutex-guarded). Engaged by default
  with automatic fallback to the serial walk (no selection · seekable · not
  `--keep-going` · valid contiguous table · no duplicate names · N≥2 after
  capping by CPU/16/entries/**frames**). Both our indexed and foreign
  zstd-seekable archives supported. ThreadSanitizer-clean; byte-identical to the
  serial walk and to source across mixed trees; corruption → exit 4. The
  current-hardware caveat below still holds (write-bound; the win is
  future-proofing), which is why it stayed Medium and the serial path remains.

  *Original design record (for context):* Today one
  thread walks the decompressed tar stream in order (header → size → next header)
  and dispatches files to a writer pool; the walk is serial because tar has no
  index and each header's position depends on the previous member's size. To
  parallelize the *parse + dispatch*, give workers independent entry-range
  start points. Two routes:
  1. **In-memory entry table (no format change):** a first pass walks headers to
     build an offset→entry map, then N workers each extract a disjoint entry
     range from the already-in-RAM (or re-decompressed) frames. Costs a parse
     pass; only the dispatch parallelizes.
  2. **Skippable-frame parse index (our archives only):** plant a header-boundary
     index in a zstd skippable frame at create time (file stays a standard
     `.tar.zst` — see the parse-index design note); readers seek straight to
     entry boundaries, no pre-walk. Foreign archives fall back to route 1 or the
     serial walk. **Shipped in v0.14.80 for create + `-l` (instant listing;
     records carry hdr_off/data_off/entry_end per member), and in v0.14.82 for
     selective extraction:** the index now also carries a frame table (chunk
     size + per-frame compressed sizes → seek offsets by prefix sum), and
     `-d --tar ARCHIVE MEMBER...` reads/decompresses only the frames the
     selection touches (one file out of a 9.76 GiB archive: 0.58 s / 0.1% read
     vs 6.7 s for the walk). What remains of this item is the NO-selection
     case: parallel parse+dispatch of a full extraction using the same frame
     table (workers start at entry boundaries), which is exactly where the
     hard parts below (dir-creation ordering, write contention, hardlink/
     deferred ordering) live — the selective path sidesteps them by keeping
     the single serial Extractor and only shrinking its input.
  Hard parts independent of which route: **directory-creation ordering** (a
  worker extracting `a/b/f` must not race the creation of `a/`, `a/b/` — needs an
  upfront dir-tree pass or per-prefix coordination), **write-stream contention**
  (see below), **metadata/hardlink/`finish_deferred` ordering**, and **security
  invariant preservation** (`openat`/`O_NOFOLLOW` per worker).

  **Current-hardware caveat (measured, NOT a reason to drop this):** on a Gen5
  NVMe server the array is the wall, not the software — a *single* O_DIRECT write
  stream already reaches the device ceiling (`dd` probe: 1 stream 3.0 GiB/s, 4
  concurrent streams 3.6 GiB/s aggregate ≈ no scaling), and `-d --tar` extract of
  a 130 GiB / large-file archive already runs at ~3.0 GiB/s wall vs a ~3.5 GiB/s
  single-stream write rate. So on *today's* drives the parallel-dispatch win is
  bounded to the ~15% the serial parse/pipeline loses below the device rate — most
  of which the cheap single-writer pipelining cleanup (persistent writer + deeper
  buffer queue, v0.14.22/23) already reclaims, no parallel dispatch needed.
  **The design stays on the roadmap deliberately**: it is correct and future-
  proofs gzstd for arrays whose write fabric *does* scale with concurrent streams
  (multi-controller / CXL / next-gen NVMe), where parallel writers become the
  dominant lever. Re-probe concurrent-O_DIRECT scaling on new hardware; build when
  a target array shows >1.5× aggregate from N streams.

- **Zstd-ecosystem seek interop** — *Priority: Medium | Complexity: Medium |
  Status: (1) and (2) shipped v0.14.83 for seekable-format archives; pzstd
  inline-tag reading deferred (legacy tool, chunks may lack the content-size
  header the map needs, no pzstd available to test against — revisit only if
  real pzstd archives show up).*
  1. **Emit the standard zstd seekable format** (contrib/seekable_format — what
     t2sz produces and indexed_zstd/ratarmount-class readers consume): indexed
     archives end with a spec-conformant seek table (u32 csize/dsize per data
     frame, footer magic at EOF); the GZIDX member index sits immediately
     before it and dropped its private GZFT frame table — the standard table
     now serves outside readers AND our own seek-extract, so foreign tools get
     random access to gzstd archives for free. Omitted only when a frame
     exceeds the format's u32 fields (huge --chunk-size); the index then sits
     at EOF as before and seek-extract falls back to the walk. pzstd's inline
     tags are NOT emitted — legacy format, benefits only pzstd's own
     decompressor (zstd -T superseded it).
  2. **Read foreign seek metadata for selective extraction** (shipped
     v0.14.83 for seekable-format archives): t2sz/seekable archives map
     compressed↔uncompressed offsets. With member selection, header-hop
     (`build_foreign_seek_plan`): walk tar headers decompressing only
     header-bearing frames, skip file data by arithmetic, feed matched ranges
     to the existing seek_feed → Extractor pipeline. Big win on large-file
     archives; degrades toward the full walk when small files put headers in
     every frame. The scan bails to the walk on GNU sparse, pax globals, bad
     checksums, or mid-stream skippables; a scan miss can only produce "Not
     found", never corruption (the Extractor re-parses the sliced stream for
     real). Their formats carry no entry metadata, but a **header-hop -l
     shipped in v0.14.91**: the same scanner also collects the tar-tvf listing
     fields from the hopped headers and feeds the index route's `list_entries`,
     so foreign seekable archives list near-instantly (byte-identical to
     `tar -tvf`, walk fallback on any scan bail). Only pzstd inline-tag reading
     remains deferred (see status above).
- **GNU sparse files** — *Status: DONE (opt-in `--sparse`; PAX GNU.sparse.1.0 default
  since v0.14.89).* `--sparse` on create detects holes via `SEEK_DATA`/`SEEK_HOLE`
  during the parallel stat pass (`probe_sparse`, reads no data) and stores holey
  files compactly as **PAX GNU.sparse.1.0** by default (graceful degradation for
  sparse-unaware readers), or OLDGNU `'S'` via `--format=gnu`/`oldgnu`. Verified:
  a 10 GiB-logical file (data past the 8 GiB base-256 boundary) round-trips
  byte-identical with holes preserved through gzstd, GNU tar, bsdtar, and Python,
  no `--sparse` flag on extract. Left OPT-IN (matching GNU tar); the compressed-size
  win is small (zstd already crushes zero runs) — the real benefit is a smaller
  uncompressed stream and not reading/writing the holes (a large win for VM/DB
  images). Auto-enable on create was considered and is CLOSED as won't-do:
  gzstd matches GNU tar's opt-in `--sparse` (2026-07 decision — not a pending
  item, do not revisit without a user-visible reason).
- **More tar input ergonomics** — *Priority: Low | Complexity: Low | Status:
  DONE (v0.14.90).* `--exclude-from FILE`/`-X` (also `-` = stdin),
  `--exclude-vcs` (GNU tar's version-control table, listing-parity verified),
  `--files-from FILE` (long-only — GNU tar's `-T` short form is taken by
  threads for zstd-CLI compat; lines are literal paths, never options), and
  `-P`/`--absolute-names` (create-only; extraction always strips leading `/`
  and stays contained, so `-P` on extract is refused, not ignored).
- **xattrs / ACLs / SELinux** — *Priority: Low | Complexity: Medium | Status:
  DONE (`--xattrs`/`--acls` v0.14.3; `--selinux` v0.14.91).* Opt-in flags
  store/restore PAX `SCHILY.xattr.*`, `SCHILY.acl.access`/`SCHILY.acl.default`,
  and `RHT.security.selinux` records, GNU-tar interoperable in both directions
  (xattr/ACL round-trips verified gzstd↔GNU tar including directory default
  ACLs; SELinux record handling verified against crafted PAX archives).
  `--selinux` reads/restores through the `security.selinux` xattr directly — no
  libselinux dependency — and restore is best-effort like `--xattrs`. Caveat:
  create-side emission with a REAL context is verifiable only on an
  SELinux-labeled host; spot-check a full round-trip if one becomes available.
- **Parallelize the walk** — *Status: DONE (v0.14.9) — this item was stale.*
  `build_layout` is a three-pass design: Pass A enumerates serially via readdir
  `d_type` (no leaf lstat), Pass B runs every `lstat`/`readlink` (+ `--sparse`
  hole probe) in parallel — the cold-inode storm that measured ~10.7 s / ≈20% of
  cold wall on a 1M-file tree — and Pass C finalizes serially so archives stay
  byte-identical to the old walk. `-v [TIMING]` reports the `enum`/`stat` split;
  if the serial Pass A enumerate ever shows up as the residual bottleneck on a
  cold many-small-file tree, that is the remaining (unmeasured) lever.
- **Cache-bypass / O_DIRECT member reads on `--tar` create** — *Priority: Low |
  Complexity: Medium | Status: investigate + MEASURE, two separate questions.*
  The `--tar` member reader (`assemble()`) opens every member `O_RDONLY` buffered
  and `--direct-read` is a no-op there (only wired to the single-file input path,
  which warns). Motivating use case: a backup-then-delete (deleteuser) reads the
  whole home once and never reuses it, so populating the page cache is pure
  waste — doubly so right before the data is deleted.
  1. **`POSIX_FADV_DONTNEED` after each member read (the likely-right answer).**
     Keep the buffered parallel reader (and its kernel readahead), but drop each
     member's pages from cache once read — fast reads AND no cache pollution.
     Cheapest packaging: make `--direct-read` on `--tar` DO this instead of
     warning, so the obvious flag just works. Gate it (default off): a normal
     user may want the cache warm; a backup does not.
  2. **Actually benchmark a real O_DIRECT multi-reader for `--tar`, cold.** This
     was NEVER tested for the many-small-file case — the measured O_DIRECT loss
     (single-file: buffered readahead ~9.6 GiB/s vs O_DIRECT ~4.5 GiB/s, see the
     `--direct-read` notes) is for ONE large sequential stream, not a parallel
     small-file walk. The a-priori expectation is that O_DIRECT still loses here
     (no cross-file readahead, alignment overhead on small files, and the
     concurrent-O_DIRECT contention already documented), but that's a guess —
     wire O_DIRECT into `assemble()`'s member opens behind a flag and measure
     cold (umount/remount or drop_caches) on Gen4/Gen5 vs the buffered reader
     across a realistic home-dir mix. If it genuinely wins, keep it; if not, (1)
     delivers the actual goal (cache hygiene) without the throughput hit. Either
     way, settle it with numbers rather than the current warning.
- **O_DIRECT extraction writes for large files (Gen4+)** — *Priority: Low |
  Complexity: Medium | Status: investigate, measure first.* Extraction currently
  writes every member buffered (page cache) — the 4 MiB `SMALL_FILE_MAX` cut only
  splits "buffer whole file → writer pool" from "stream inline", both buffered;
  there is no O_DIRECT on the extract side. On fast NVMe (PCIe Gen4/Gen5) and
  memory-bandwidth-bound boxes, writing *large* members (new, higher threshold —
  ~≥32–64 MiB, NOT the 4 MiB cut) via O_DIRECT could sustain device bandwidth and
  avoid evicting the whole page cache during a big restore. Caveats that make it
  measure-first, not obvious: buffered `write()` is already async (writeback
  overlaps decompression) while O_DIRECT is synchronous, so big-file writes must
  move off the parser thread with double-buffered aligned chunks or they stall the
  pipe; alignment + unaligned-tail handling needed (reuse `DirectWriter`);
  concurrent O_DIRECT streams contend (keep to one writer); only helps
  large-file-heavy archives, never the many-small-files case. Gate behind
  `--direct` (+ Gen4 auto, mirroring compress). Prototype + cold-cache benchmark
  on the Gen4+ box before shipping.

---

## Version History Summary

| Version Range | Key Changes |
|--------------|-------------|
| v0.9.50-v0.9.59 | Initial GPU support, scheduler tuning, failed pinned memory & mmap & CUDA warm-up |
| v0.9.60-v0.9.73 | Performance instrumentation, GPU selection, O_DIRECT writer, async write pool, sparse files |
| v0.9.74-v0.9.99 | Semaphore scheduler, VRAM-aware batching, trivial frame detection, per-GPU auto-tuner |
| v0.10.0-v0.10.8 | Binary-search auto-tuner, shared tuning across GPUs, REFINE phase |
| v0.10.9-v0.10.21 | Shared auto-tuner wired for compress+decompress, continuous probing, writer drain diagnostics |
| v0.10.22-v0.10.29 | io_uring (failed), pwrite pool (failed), reverted to O_DIRECT |
| v0.10.30-v0.10.34 | Removed fsync, --sync-output flag, file-size-based decompress batch start |
| v0.11.0-v0.11.19 | Per-GPU result slots, batch-completion notifications, proportional GPU scaling, rate-match (disabled) |
| v0.11.20 | Removed dead liburing references (cleanup) |
| v0.11.21 | CV-based CPU worker scheduling (replaced 9 sleep loops with condition variable waits) |
| v0.11.22 | Early memory release (+7% on mixed data), rescue-safe GPU buffer management |
| v0.11.23 | Write drain progress bar, verbose output cleanup, wrote_bytes tracks physical I/O |
| v0.11.24 | Writer backpressure (+56% hybrid decompress on 432 GiB file, sys time -66%) |
| v0.11.25 | Test mode fixes (wrote_bytes double-counting, backpressure stall, progress label) |
| v0.11.26 | Graceful GPU VRAM skip, structured exit codes, argument hardening, `--threads=N`, `--` support |
| v0.11.27 | Writer deadlock detection (5s timeout → hard error), `die()` cleanup reporting, atomic temp cleanup |
| v0.11.28-29 | Compress backpressure (all paths), RAM budget check, `--cpu-batch` ignored in `--cpu-only`, VRAM retry limit |
| v0.11.30 | Default chunk 16 MiB everywhere, dual-rate progress bar, removed auto-chunk scaling, comprehensive test suite |
| v0.11.31 | Stdout O_DIRECT detection (3× faster piped decompress), GPU backpressure on pop, -t defaults to 2 streams |
| v0.11.38–v0.11.44 | backpressure set_done ordering, fallocate preallocation, hybrid deadlock fixes, thundering herd fix, writer_stalled_ signal |
| v0.12.0 | FrameThrottle: counting semaphore replaces byte-based backpressure (-57 lines, deadlock-free by construction) |
| v0.12.14–20 | Pipeline-depth throttle budget, throttle diagnostics/tunables, thundering herd fix, default buffered I/O, hybrid deadlock fixes, re_enqueue FIFO fix |
| v0.12.21 | mmap zero-copy compression input (3.1s vs 9.9s), benchmark accuracy fix, failed mmap output experiment documented |
| v0.13.0–v0.13.16 | Asymmetric mode + PCIe-gen detection, streaming single-frame decompress, bounded per-worker buffer pools (page-fault storm fix), CV-wait pool drain, skip-serial-GPU-probe, CUDA-init/reader overlap |
| v0.13.17–v0.13.22 | mmap fault-storm investigation: producer prefault and kernel-gated mmap/fread both tried and reverted (pre-6.4-kernel mmap_lock artifact); `--cold` flag for honest cold-cache benchmarking; mmap restored as default everywhere |
| v0.13.23 | AsyncWritePool flush() waits for physical write completion (final-batch I/O errors no longer slip past had_error()) |
| v0.13.24 | Recycled GPU decompress output-buffer pool (DecompStreamCtx::out_pool) — kills per-frame D2H alloc churn; Gen3 proxy −15% faults / −12% RSS on gpu-only -d |
