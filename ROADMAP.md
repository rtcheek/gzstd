# gzstd v1.0 Roadmap & Battle Plan

**Current version:** v0.13.31
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
**Priority: Medium | Complexity: Medium | Status: PARTIALLY DONE**

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

---

## Phase 2: Persistent Auto-Tuning (`~/.gzstd/`)

### 2.1 Per-Machine Performance Profile
**Priority: Medium | Complexity: Medium | Status: NOT STARTED**

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
**Priority: Medium | Complexity: Medium | Status: NOT STARTED**

`gzstd --calibrate` runs a quick benchmark suite (30-60 seconds):
1. Small compress/decompress on CPU (measures per-core throughput)
2. Small compress/decompress on each GPU (measures per-device throughput)
3. Sequential write benchmark (measures NVMe speed)
4. Stores results in `~/.gzstd/profile.json`

Subsequent runs read the profile and start with known-optimal settings. The runtime auto-tuner still runs but converges instantly since it starts at the right point.

### 2.3 Automatic Profile Updates
**Priority: Low | Complexity: Low | Status: NOT STARTED**

After each run, if the auto-tuner found a different optimal than the profile predicted, update the profile. This handles hardware changes (new GPU, driver update, different NVMe) without requiring explicit recalibration.

---

## Phase 3: Piped I/O Optimization

### 3.1 Pipe-Aware Scheduling
**Priority: Medium | Complexity: Medium | Status: NOT STARTED**

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
**Priority: Low | Complexity: Low | Status: NOT STARTED**

When input size is unknown (pipe), the frame count is unknown. The auto-tuner must be more conservative:
- Don't set tune_hi too high (we might run out of frames before exploring)
- Shorter probe interval (adapt faster)
- Skip the "file size > 75 GiB" logic (we don't know)

---

## Phase 4: Parallel I/O (Research)

### 4.1 Multi-Reader for NVMe
**Priority: Low | Complexity: High | Status: Research needed**

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

**Verdict:** Likely small gain for compression (reader is rarely the bottleneck  3+ GiB/s single-thread is usually enough). For decompression, the complexity of frame boundary detection likely outweighs the benefit. Worth benchmarking with a simple 2-reader prototype before committing.

### 4.2 Multi-Writer with pwrite()
**Priority: Low | Complexity: Medium | Status: Tested, NEGATIVE for buffered I/O**

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
- **Compress** benefits the same way (knuth `--direct` data): cpu-only
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
**Priority: HIGH | Complexity: Medium | Status: DONE for decompress (v0.13.24); compress deferred**

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

**Deferred (compress):** `compress_nvcomp` completion paths also allocate per
frame, but hold only the *compressed* output (small), so fault pressure is far
lower. Same pool pattern applies; do it only if a Gen4 compress profile shows it
matters.

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
**Priority: Medium | Complexity: Medium | Status: NEEDS BENCHMARK**

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
| GPU result buffer pool (Gen4 hybrid decompress) | 7.2 | HIGH | Decompress DONE (v0.13.24); compress deferred |
| Throttle budget uses resolved chunk size | 7.3 | Medium | DONE (v0.13.28) |
| CPU-compress redundant memcpy | 7.4 | Medium | Needs benchmark |
| --sync-output under --direct | 7.5 | Low | DONE (v0.13.30) |
| is_all_zero unaligned load | 7.6 | Low | DONE (v0.13.30) |
| Remove dead SequentialDispatcher | 7.7 | Low | DONE (v0.13.30) |
| Decompress reader queue-depth cap (gpu-only RSS blowup) | 7.8 | Medium | DONE (v0.13.29) |
| Bundled short flags (-dc, -dk) for zstd/gzip compat | 7.9 | Medium | DONE (v0.13.27) |

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
