# gzstd v1.0 Roadmap & Battle Plan

**Current version:** v0.13.0
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

## Remaining Work for v1.0

| Item | Phase | Priority | Status |
|------|-------|----------|--------|
| Streaming decompression output | — | HIGH | DONE (v0.12.24) |
| Asymmetric mode (PCIe Gen3 detection) | 5.1, 5.2 | HIGH | DONE (v0.13.0) |
| Persistent auto-tuning (`~/.gzstd/`) | 2.12.3 | Medium | Not started |
| Rate-matched dispatch (re-enable) | 1.3 | Medium | Disabled, needs eval |
| Pipe-aware scheduling | 3.1 | Medium | Not started |
| Streaming mode for unknown-size input | 3.2 | Low | Not started |
| Multi-reader NVMe | 4.1 | Low | Research |
| Multi-writer O_DIRECT pwrite | 4.2 | Low | Tested negative for buffered |

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
