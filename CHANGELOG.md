# gzstd Optimization Changelog

**Covers:** v0.9.50 → v0.11.24  
**Test machines:**
- **Knuth:** 256-core CPU, 8× NVIDIA H100 (95 GiB VRAM each), NVMe ~3 GiB/s write
- **Lovelace:** 24-core CPU, 2× NVIDIA RTX 2080 Ti (10 GiB VRAM each), NVMe ~1.8 GiB/s write

---

## Write Path Optimizations

### O_DIRECT Writer (v0.9.71)  POSITIVE
Bypasses page cache for sequential writes. Uses 16 MiB aligned buffer, flushes in aligned chunks.
- **Knuth:** Writer I/O improved 1.1 → 2.72 GiB/s on 432 GiB file
- **Why it works:** Avoids double-buffering through page cache for large sequential writes
- **Caveat:** Unaligned tail requires dropping O_DIRECT via fcntl for final write

### pwrite for Out-of-Order Decompress (v0.9.72)  NEGATIVE (reverted)
Tried using pwrite() to write decompressed frames directly to their final offset without waiting for in-order delivery.
- **Knuth:** 0.93 GiB/s (worse than sequential 2.72 GiB/s)
- **Why it failed:** 27k individual O_DIRECT pwrite calls = massive kernel DMA setup overhead. sys time: 12m45s.
- **Lesson:** O_DIRECT pwrite per-frame is catastrophically expensive. Sequential batch drain is better.

### Async Double-Buffered Write Pool (v0.9.73)  POSITIVE
Background write thread with one pending slot. Writer collects batch → submits to pool (non-blocking) → collects next batch while pool writes previous.
- **Knuth:** Improved overlap between GPU D2H and disk writes
- **Why it works:** Writer thread doesn't block on disk I/O; can collect next batch while previous is being written

### Sparse File Support (v0.9.73)  POSITIVE (for zero-heavy data)
Scans 4K blocks for zeros, lseek past them instead of writing. Integrated with both O_DIRECT (DirectWriter::seek_forward) and fwrite paths.
- **Knuth:** zeros.bin decompress: sparse=5.2s vs no-sparse=6.9s (~25% faster)
- **Why it works:** Avoids physical writes for zero-filled regions
- **Caveat:** O_DIRECT seek_forward must flush internal buffer before seeking. Added --[no-]sparse flag matching zstd syntax.

### io_uring Writer  NOT YET TRIED
Proposed: Replace O_DIRECT write() with io_uring for less syscall overhead per write.
- **Expected:** 10-20% improvement on NVMe drives where per-syscall overhead is significant
- **Rationale:** NVMe drives have deep internal queues; io_uring can submit multiple writes without syscalls

### mmap + memcpy Writer  NOT YET TRIED
Proposed: mmap output file at target size, memcpy frames directly. Kernel handles writeback.
- **Expected:** Good for sparse data (unmapped pages stay as holes), possibly worse for dense data
- **Risk:** mmap as INPUT was already tried and was negative (v0.9.53-54)

### Multiple pwrite Threads  NOT YET TRIED
Proposed: Open output file multiple times, pwrite from multiple threads at known offsets.
- **Expected:** Could double NVMe throughput by increasing queue depth
- **Risk:** O_DIRECT pwrite per-frame was catastrophic (v0.9.72); would need large contiguous writes

### Page-Cache Path for Trivial Data  NOT YET TRIED
Proposed: When >90% of blocks are zero, drop O_DIRECT and use fwrite + ftruncate. This is what zstd does  the page cache handles sparse much more efficiently.
- **Expected:** Match zstd's 2-3s on zeros.bin (currently 4-5s)
- **Rationale:** zstd achieves 0.3s sys time on zeros vs our 5-8s with O_DIRECT sparse

---

## Read Path Optimizations

### mmap Input (v0.9.53-54)  NEGATIVE (reverted)
Replaced fread with mmap for zero-copy reading.
- **Why it failed:** mmap with t.data.assign() still copies from mapped pages (not zero-copy). Worse than fread for sequential I/O due to page fault overhead and TLB pressure.
- **Lesson:** mmap only wins with true zero-copy (string_view/span) or random access patterns. Sequential fread is hard to beat.

### Offset-Based Buffer (v0.9.50)  NEGLIGIBLE (kept)
Replaced buf.erase(0,N) O(n) memmove with offset cursor.
- Correct optimization, prevents pathological quadratic behavior, but invisible at 128-frame scale.

---

## GPU Memory & Transfer Optimizations

### Pinned (Page-Locked) Memory for GPU Decompress (v0.9.53)  NEGATIVE (catastrophic, reverted)
cudaHostAlloc for H2D/D2H staging buffers to enable true async DMA.
- **Knuth:** GPU decompress nearly doubled: 13.4s → 25.6s
- **Why it failed:** Massive pinned allocations (512 MiB per stream) starved system memory, caused page faults in other threads, and the copy-to-pinned + DMA was slower than direct pageable transfer for our access pattern.
- **Lesson:** Pinned memory requires small rotating pools, not batch-sized allocations. The extra memcpy to/from pinned staging negated any DMA benefit.

### Frame-Level Pinned Buffer Pool  NOT YET TRIED (proposed v0.9.55)
Small rotating pool of frame-sized pinned buffers (2-4 × 16 MiB) shared across streams. True async overlap.
- **Key difference from failed #7:** Small pool vs massive per-stream allocation. Would enable cudaMemcpyAsync to actually overlap with kernel.

### Pre-Allocated GPU Decompress Buffers (v0.9.51)  NEGLIGIBLE (kept)
ensure_buffers() allocates once, reuses across batches. Saves ~150-300ms of cudaMalloc/cudaFree per file.
- Invisible at 8 GiB scale but correct for repeated small files.

### VRAM-Aware Batch Sizing (v0.9.96-98)  POSITIVE
Binary search for largest compress batch that fits in VRAM. Includes nvCOMP temp workspace in estimate.
- **Lovelace (10 GiB VRAM):** Finds batch=104 instead of hanging on batch=256
- **Why it matters:** cudaMalloc can hang on some drivers if request exceeds VRAM. Pre-check avoids this.
- Fixed partial allocation leak on retry (free_stream_buffers_only before halving).

---

## CUDA Context & Init Optimizations

### CUDA Context Warm-Up (v0.9.58-59)  NEGATIVE (reverted)
Pre-initialize CUDA contexts on all devices before GPU workers start.
- **Both sync and async versions added ~3s overhead**
- **Why it failed:** CUDA contexts are per-thread. Warming up from a temporary thread creates a throwaway context; the actual GPU worker creates its own anyway.
- **Lesson:** CUDA per-thread context model makes warm-up ineffective. Would need cuDevicePrimaryCtxRetain (driver API) to share contexts.

---

## Scheduling & Routing Optimizations

### Hybrid Scheduler: 256 Threads (v0.9.52)  NEGATIVE (reverted)
Full hardware_concurrency() threads at 80% CPU start share.
- **Regression:** 0.75x across all configs
- **Why it failed:** 256 worker threads starved the reader/writer I/O threads. Even GPU-only mode degraded because I/O pipeline couldn't keep up.
- **Lesson:** I/O pipeline (reader + writer) is the critical path. Cap CPU threads below full hardware count.

### Adaptive CPU/GPU Share via EMA (v0.9.52-78)  MIXED
Various attempts at throughput-based adaptive scheduling.
- 50/50 start: CPU ate everything before GPU initialized
- 10/90 GPU-favored start: CPU still drained queue during GPU init
- Throughput measurement: CPU always appeared faster because GPU was starved for data
- **Final solution:** Semaphore-based scheduling (v0.9.83)

### GPU-Priority Semaphore Scheduler (v0.9.83)  POSITIVE
`gpus_waiting` atomic counter. GPU increments before pop, decrements after. CPU yields when counter > 0.
- **Why it works:** Direct, instant priority signaling. No measurement delay. GPU always gets fed first.
- CPU runs wild during GPU init, then yields once GPU signals ready
- CPU helps when all GPUs are busy processing (counter = 0)

### Trivially-Compressed Frame Detection (v0.9.93)  POSITIVE
Decompress: peek at front frame's ratio. If < 2%, CPU takes it regardless of GPU priority.
- **Why it works:** Frames decompressing to mostly zeros are faster on CPU (no PCIe D2H overhead). CPU + sparse writes = near-instant.
- **Knuth:** zeros.bin: CPU path 1.4s vs GPU path 4.4s

### Auto CPU Thread Cap at 96 (v0.9.80)  POSITIVE
Default auto: min(hw-1, 96). -T0 = all threads (matches zstd).
- **Why:** Diminishing returns beyond 96 threads on large-core machines. Leaves headroom for I/O threads.

### --cpu-batch as Queue Depth Threshold (v0.9.92-94)  POSITIVE
Minimum queue depth before CPU workers activate. Each CPU takes 1 frame (no CPU batching benefit).
- **Why:** Keeps queue stocked for GPUs. CPU only helps when there's overflow.

---

## Batch Size Auto-Tuning

### Decompress Greedy Batch Pop (v0.9.69)  POSITIVE (massive)
pop_batch_greedy waits for full batch before GPU processes. DEFAULT_GPU_DECOMP_BATCH_CAP = 256.
- **Knuth:** medium_compress kernel dropped 24.7s → 1.27s (55× speedup!)
- **Why:** Default batch=8 caused 64 kernel launches × 385ms each. Batch=256 = 3 launches × 424ms.

### Continuous Binary-Search Auto-Tuner (v0.10.0-0.10.6)  POSITIVE
Runtime throughput-aware batch sizing for compress. Explores both directions from default.
1. Record baseline throughput at starting batch size
2. Try halving  if better, continue halving
3. Try doubling  if better, continue doubling
4. Settle at best when throughput drops
5. Periodically probe to detect data character changes
- **Lovelace:** Correctly finds batch=8 optimal for compress, settles in 2 steps
- **Fixed bugs:** free_stream_buffers_only wiped tune state (v0.10.4), tune ceiling was default not VRAM limit (v0.10.2), baseline never recorded (v0.10.3)

---

## GPU Selection & Topology

### NVML/NUMA-Aware GPU Selection (v0.9.63-68)  POSITIVE
Queries GPU utilization and NUMA topology. Penalizes GPUs on busy NUMA nodes.
- **Why:** Prevents selecting GPU 6 when GPU 4 (same NUMA node) is busy at 29%.

### --gpu-devices N (v0.9.62)  POSITIVE
Decompress default: 2 GPUs (PCIe bandwidth optimal for 1-2 GPUs).
Compress default: all GPUs.

---

## Performance Instrumentation

### -vvv Breakdown (v0.9.61)  ESSENTIAL
PerfCounters struct with atomic accumulators for every pipeline phase.
- **Bug found (v0.9.89):** Compress GPU worker had TWO completion paths (async poll + sync drain). Only async poll recorded to g_perf. Sync drain path handled majority of completions for small batches.
- **Lesson:** Comment both paths with "MUST record to g_perf  see also other path"

---

## Key Architectural Lessons

1. **PCIe is the wall for GPU decompress.** GPU kernel is fast; moving 8 GiB D2H at 1.5-3.5 GiB/s dominates. 1 GPU often beats 8 GPUs due to PCIe contention.

2. **Writer I/O is the wall for CPU decompress.** CPU decompresses at 5-20 GiB/s aggregate but NVMe writes at 1.8-3.0 GiB/s.

3. **Never starve the I/O pipeline.** Reader and writer are serial bottlenecks. Too many CPU threads, too-high I/O priority, or GPU-induced memory pressure all cause regression.

4. **CUDA contexts are per-thread.** Warm-up on temporary threads is useless. Pinned memory from wrong context causes slowdown. Always design around the thread that will actually use the GPU.

5. **Measure before optimizing.** The -vvv breakdown has been the single most valuable tool. Every successful optimization was guided by perf data. Every failed one was based on hypothesis alone.

6. **Small GPUs need different tuning than large GPUs.** H100 (95 GiB): batch=256 decompress, 8 GPUs. RTX 2080 Ti (10 GiB): batch=8 compress, batch=16 decompress, 2 GPUs. The auto-tuner handles this automatically.

---

## Benchmark Snapshots

### Knuth (H100 × 8)  v0.9.74 vs zstd -T0, 8 GiB files, decompress
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| zeros | 4.85s | 4.40s | 1.10× |
| high_compress | 9.52s | 7.07s | **1.35×** |
| medium_compress | 15.39s | 9.69s | **1.59×** |
| mixed | 9.12s | 6.55s | **1.39×** |
| low_compress | 9.25s | 7.14s | **1.30×** |
| **Total** | **48.13s** | **34.85s** | **1.38×** |

### Lovelace (RTX 2080 Ti × 2)  v0.10.6 vs zstd -T0, 8 GiB files
**Decompress:** gzstd wins 2/5 (medium_compress 1.22×, low_compress 1.06×). Loses on trivial data where zstd's page-cache sparse dominates.
**Compress:** gzstd wins 4/5 (high 1.83×, low 1.54×, medium 1.11×, mixed 1.26×). Only loses zeros.

---

### io_uring Writer (v0.10.22-0.10.28)  NEGATIVE (reverted)
Replaced DirectWriter + AsyncWritePool with Linux io_uring for async writes.
- **v0.10.22-26:** O_DIRECT + io_uring. Writes submitted but never completed  `io_uring_wait_cqe` hung forever. Likely kernel/NVMe driver incompatibility with O_DIRECT + io_uring on Knuth.
- **v0.10.27:** Tried `io_uring_submit_and_wait()`  still hung.
- **v0.10.28:** Dropped O_DIRECT, tried buffered io_uring  still hung.
- **Root cause:** Unknown kernel-level issue. io_uring write completions never arrived despite successful submission. Possibly a kernel config, seccomp policy, or filesystem limitation.
- **Decision:** Reverted to DirectWriter + AsyncWritePool.

### Multi-threaded pwrite Pool (v0.10.29)  NEGATIVE (reverted)
4 threads doing pwrite() at known offsets through the page cache.
- **Knuth:** 10m30s (vs 4m with DirectWriter). `sys: 38m40s` (vs 12m).
- **Why it failed:** Without O_DIRECT, 432 GiB went through the page cache. The pwrite() calls returned fast (page cache absorb), but kernel writeback stalled massively. The page cache backlog created 9.5 minutes of post-completion flush.
- **Key lesson:** You cannot beat the NVMe's physical write speed (~2-3 GiB/s on Knuth). O_DIRECT + single-threaded sequential write is already optimal for this workload. The 220s writer drain IS the hardware limit  not a software bottleneck.
- **Decision:** Reverted to DirectWriter + AsyncWritePool (v0.10.30).

---

### Removed fsync on output (v0.10.31-33)  POSITIVE
Removed fsync() call before closing output file. Like zstd, the OS handles writeback in the background after close(). With O_DIRECT, data is already on physical media  only the tiny unaligned tail goes through the page cache.
- Added `--sync-output` flag for users who need guaranteed persistence before exit.
- Renamed misleading "flushing to disk" messages to "draining write queue" / "writing..."
- **Decision:** Default off. Matches zstd behavior.

### File-size-based decompress batch start (v0.10.34)  POSITIVE
Starting batch size for decompress auto-tuner now scales with input file size:
- >75 GiB: start at 256 (was 16, wasted minutes exploring upward on large files)
- >10 GiB: start at 64
- ≤10 GiB: start at 16
Auto-tuner still refines from the starting point. On 217 GiB file, converges to 512 in 3 steps.

---

## Benchmark Snapshots (Updated)

### Knuth (H100 × 2 GPUs)  v0.10.34, 432 GiB file (rpfrancis.tar)

**Decompress test mode (-t, no disk I/O):**
- 432.58 GiB decompressed in **53.5 seconds** = **8.13 GiB/s**
- Auto-tuned to batch=512
- 96 CPU threads + 2 GPU devices

**Decompress to disk (O_DIRECT):**
- 432.58 GiB in ~3m37s-5m22s = **1.3-2.0 GiB/s** (varies with NVMe contention)
- Writer drain: ~220s (NVMe write bandwidth ceiling)
- Compute pipeline runs at 4+ GiB/s, storage is the bottleneck

**Compress (4 GPUs, v0.10.11):**
- 432.58 GiB → 217 GiB in **3m21s** = **2.16 GiB/s**
- Auto-tuned to batch=48→816 over the run

### Lovelace (RTX 2080 Ti × 2)  v0.10.6, 8 GiB files

**Compress: gzstd wins 4/5 vs zstd -T0**
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| high_compress | 7.23s | 3.96s | **1.83×** |
| low_compress | 5.85s | 3.80s | **1.54×** |
| medium_compress | 4.33s | 3.91s | **1.11×** |
| mixed | 4.28s | 3.40s | **1.26×** |
| zeros | 2.47s | 3.88s | 0.64× |

**Decompress: gzstd wins 2/5 (storage-limited on consumer NVMe)**

### Lovelace (RTX 2080 Ti × 2)  v0.11.20, 8 GiB files, 3 iterations

**Compress (GiB/s):**
| File | CPU | GPU-only | Hybrid | Best config |
|------|-----|----------|--------|-------------|
| high_compress | 1.46 | **2.04** | 2.02 | GPU ≈ Hybrid |
| medium_compress | 1.36 | **2.10** | 2.06 | GPU ≈ Hybrid |
| mixed | 1.50 | 1.73 | **2.14** | Hybrid |
| low_compress | **1.50** | 1.22 | 1.51 | CPU ≈ Hybrid |
| zeros | 1.49 | **2.06** | 2.02 | GPU ≈ Hybrid |

Hybrid compress matches or beats the best single backend on every data type.
Mixed data (2.14 GiB/s) beats both CPU (1.50) and GPU-only (1.73)  the scheduler
correctly splits work between CPU and GPU based on observed throughput.

**Decompress (GiB/s):**
| File | CPU | GPU-only | Hybrid | Best config |
|------|-----|----------|--------|-------------|
| zeros | **4.88** | 3.49 | 3.50 | CPU |
| medium_compress | **2.80** | 1.93 | 2.11 | CPU |
| mixed | **2.05** | 1.95 | 1.85 | CPU |
| high_compress | **1.52** | 1.44 | 1.40 | CPU |
| low_compress | 1.44 | 1.45 | 1.43 | ~Tied |

CPU wins decompress across the board on Lovelace. PCIe Gen3 bandwidth makes D2H
the bottleneck  the GPU can't transfer decompressed data back fast enough to
justify the round-trip. Trivial frame detection helps (zeros at 4.88 GiB/s).
Confirms that asymmetric mode (GPU compress + CPU decompress) would be the
ideal default for consumer GPUs with PCIe Gen3.

### Lovelace (RTX 2080 Ti × 2)  v0.11.22, 8 GiB files, 3 iterations

Machine was under student load (~12% lower baseline than v0.11.20 run).
Back-to-back v0.11.21 → v0.11.22 comparison is valid (same load).

**Hybrid Compress (GiB/s):**
| File | v0.11.21 | v0.11.22 | Delta |
|------|----------|----------|-------|
| mixed | 1.850 | **1.989** | **+7.5%** |
| high_compress | 1.844 | 1.845 |  |
| medium_compress | 1.845 | 1.845 |  |
| low_compress | 1.438 | 1.439 |  |
| zeros | 1.844 | 1.844 |  |

**Hybrid Decompress (GiB/s):**
| File | v0.11.21 | v0.11.22 | Delta |
|------|----------|----------|-------|
| mixed | 1.719 | **1.841** | **+7.1%** |
| medium_compress | 1.981 | 1.985 |  |
| zeros | 3.218 | 3.205 |  |
| high_compress | 1.288 | 1.289 |  |
| low_compress | 1.357 | 1.357 |  |

Early memory release (v0.11.22) improved mixed.bin by ~7% in both directions.
Mixed data has high frame churn (alternating compressible/random blocks) where
freeing input buffers sooner reduces page allocation contention. Other data
types are flat  bottlenecked by PCIe or NVMe, not memory lifecycle.

## Key Lessons Learned (Updated)

7. **io_uring may not work on all kernels.** Knuth's kernel accepted io_uring submissions but never completed writes. Possibly a seccomp policy, kernel config, or NVMe driver limitation. Always have a fallback.

8. **Page cache is not free.** Multi-threaded pwrite through page cache caused 38 minutes of sys time (vs 12 min with O_DIRECT). The page cache absorbed writes instantly but kernel writeback created a massive backlog. O_DIRECT + sequential single-thread is optimal for large sequential output.

9. **Don't fsync unless asked.** zstd doesn't fsync, cp doesn't fsync. O_DIRECT data is already on disk. Removing fsync saves seconds and matches user expectations. Provide `--sync-output` for paranoid users.

10. **The disk is the ceiling.** At 8.13 GiB/s compute vs 1.5-2.0 GiB/s NVMe write, the decompression pipeline is 4-5× faster than storage. No software optimization can fix this. Faster NVMe (Gen5, RAID) is the only path forward.

---

### Batched H2D Transfer (v0.11.6-0.11.8)  NEGATIVE (reverted)
Packed all compressed frames into contiguous host buffer, one cudaMemcpyAsync.
- **Why it failed:** `alloc_comp` per frame is max size (16 MiB) but actual compressed data is smaller. Packing copies 4 GiB of mostly padding. Per-frame async only copies actual bytes. CUDA driver already coalesces async transfers internally.
- H2D went from ~2 GiB/s to 0.22 GiB/s.

### Batched D2H Transfer (v0.11.6)  NEGATIVE (reverted)
Single cudaMemcpy for entire decompressed batch, deliver all frames at once.
- **Why it failed:** Blocked writer thread for entire 4 GiB transfer. Writer could no longer pipeline disk writes with GPU D2H. Per-frame D2H feeds writer continuously.
- D2H: 0.14 GiB/s. Result lock contention: 451 seconds (8 GPUs fighting one mutex).

### Thread Pinning (v0.11.5)  NEGATIVE (disabled)
Pinned reader to core 0, writer to core 1.
- **Why it failed on Knuth:** Students had ALL cores at 97-99%. Pinning forced I/O threads onto busy cores instead of letting the OS scheduler find idle moments on any core.
- **When it would help:** Dedicated machine with no competing workloads.

### GPU Utilization Backoff (v0.11.3)  REPLACED by proportional scaling
Paused GPU workers when utilization >50%, resumed at ≤30%.
- **Why it was wrong:** Blocking wastes a GPU that could still contribute at reduced capacity.
- **Replaced by:** `util_scale` factor (v0.11.4)  GPU at 50% gets half the batch size, still contributes.

### Proportional GPU Utilization Scaling (v0.11.4)  POSITIVE
`util_scale = max(0.05, (100 - gpu_util%) / 100)` applied to batch size.
- Updated via NVML after each batch completion.
- GPU at 0% → full batch, 50% → half, 90% → 10%.
- No wasted GPU cycles, no blocking.

### Sequential Frame Dispatcher (v0.11.1)  NEGATIVE (reverted)
Round-robin ticket system forcing GPUs to pop in order.
- **Why it failed:** Serialized the pop operation  GPU 1 couldn't pop until GPU 0 finished popping. With `pop_batch_greedy` blocking for enough frames, 7 GPUs sat idle while 1 waited.

## Key Lessons Learned (Updated)

11. **Don't batch what CUDA already batches.** `cudaMemcpyAsync` in a stream is already coalesced by the driver. Manual packing adds host-side memcpy overhead and padding waste.

12. **Writer parallelism > transfer efficiency.** Per-frame D2H is "inefficient" per-transfer but keeps the writer pipeline full. Batched D2H is "efficient" but starves the writer. Pipeline throughput wins.

13. **Thread pinning hurts on shared machines.** The OS scheduler is better at finding idle moments across all cores than a fixed pin on a busy core.

14. **Proportional > binary.** Don't block a resource (GPU, core)  scale its allocation proportionally. A 50%-loaded GPU with half the batch is better than an idle GPU.

---

### Per-GPU Result Slots (v0.11.11)  POSITIVE (major)
Each GPU pushes decompressed frames to its own slot (own mutex). Writer drains all slots periodically. Eliminates cross-GPU mutex contention.
- **Result lock: 451s → 0.06s** (7,500× improvement)
- Why: 8 GPUs doing per-frame lock/unlock on one shared mutex = massive contention. Per-GPU slots = zero contention (one producer per slot).

### Batch-Completion Writer Notification (v0.11.14-15)  POSITIVE
Only notify writer after full D2H batch completes (not per-frame). CPU fallback path still notifies per-frame (low volume).
- **Writer wakeups: 23,185 → 254** (91× reduction)
- Each wakeup now drains 200+ frames instead of checking and sleeping.

### Pinned D2H Buffer (v0.11.17)  NEGATIVE (reverted, 3rd attempt)
Pinned host buffer per stream for D2H, then memcpy to frame vector.
- 9% slower than pageable. Two copies (DMA→pinned→vector) worse than CUDA's internal staging (DMA→internal_pinned→vector, optimized by driver).
- **Three failed pinned attempts documented.** CUDA's pageable transfer is highly optimized internally. Don't try to outsmart it unless you can eliminate ALL copies.

### Rate-Match CPU Throttle (v0.11.0, disabled v0.11.9)  MIXED
`cpu_may_take()` throttled CPU workers to match GPU batch timing.
- Correctly reduced CPU usage on loaded machines (user time dropped from 8m to 2m)
- Disabled for debugging; needs re-evaluation on quiet machine.

### Thread Pinning (v0.11.5, disabled v0.11.9)  NEGATIVE on shared machines
Reader pinned to core 0, writer to core 1. Hurts when cores are loaded by other users.
- Would help on a dedicated machine. Keep disabled by default, consider `--pin-io` flag.

### Remove dead liburing references (v0.11.20)  CLEANUP
Removed `#include <liburing.h>` and stale io_uring comment left over from the reverted io_uring writer (v0.10.22-0.10.28).
- **Why:** The include created an unnecessary build dependency on liburing despite io_uring code being fully reverted in v0.10.28.
- No functional change.

### CV-Based CPU Worker Scheduling (v0.11.21)  POSITIVE (correctness + scalability)
Replaced 9 × `sleep_for(1ms)` poll loops in CPU compress and decompress workers with proper condition variable waits. CPU workers now block on a dedicated `cpu_cv_` and wake in microseconds when conditions change.
- **TaskQueue:** Added `cpu_cv_` (dedicated CV for CPU workers), `wait_for_cpu(predicate)`, and `pop_one_cpu(task, predicate)`. Predicates receive a `QueueState` snapshot to avoid recursive lock deadlocks. `push()` uses `cv_.notify_all()` to ensure all waiting GPUs see new frames.
- **HybridSched:** `gpu_got_data()` and `set_gpu_ready()` now call `notify_cpu_waiters()` so CPU threads wake instantly when scheduling state changes instead of sleeping up to 1ms.
- **Lovelace (8 GiB files):** No measurable throughput change on small workloads (127 frames  sleep overhead was ~2% of runtime). The win is on large files with thousands of frames where 22 threads × 1ms × thousands of iterations compounds to minutes of waste.
- **Bug fixed during development:** Initial implementation deadlocked because predicate lambdas called `tq->peek_front_ratio()` / `tq->size()` / `tq->drained()` while `pop_one_cpu` held `m_` (non-recursive mutex). Fixed by passing a `QueueState` snapshot to predicates instead.
- **Bug fixed during development:** `push()` with `cv_.notify_one()` could deliver notifications to a GPU that was busy processing (not waiting), starving the other GPU. Changed to `cv_.notify_all()`.

### Early Memory Release (v0.11.22)  POSITIVE
Release input data buffers immediately after they're consumed instead of holding them until end of processing cycle. Reduces peak memory by freeing frames as soon as they're no longer needed.
- **CPU compress worker:** `t.data` (up to 32 MiB) released via swap immediately after `compress_one_cpu_frame`. Previously held through logging, stats, and result delivery.
- **GPU compress worker:** Batch input data (up to 16 × 16 MiB = 256 MiB) released after H2D upload. Guarded by `!rescue`  in hybrid mode data stays alive for potential CPU rescue on GPU failure; in gpu-only mode released immediately.
- **GPU decompress worker:** Batch compressed data released before kernel launch (after re-upload path). Saved `batch_seqs[]` and `batch_comp_sizes[]` for completion paths.
- **CPU decompress worker:** Already had early release (swap at line 2280)  no change needed.
- **Lovelace (8 GiB files):** +7.1% decompress and +7.5% compress on mixed.bin. Other data types flat (bottlenecked elsewhere). Mixed data benefits most because alternating compressible/random blocks cause high frame churn  freeing memory sooner reduces page allocation contention.

### Write Drain Progress Bar (v0.11.23)  IMPROVEMENT
Progress bar now shows write drain percentage when the compute pipeline finishes but disk I/O is still in progress. Previously showed a static "writing..." message or sat at 100% while the NVMe caught up.
- **Meter:** Added `total_out` atomic tracking expected total output bytes. Set by `stream_frames_to_queue` (decompress, from frame headers) or accumulated by writer thread (compress, as frames complete).
- **AsyncWritePool:** `wrote_bytes` now updated by the AIO worker after physical write completes (not on submit), so progress reflects actual disk I/O.
- **Progress format:** `[85.3%] writing: 6.75 GiB / 7.91 GiB @ 2.01 GiB/s` during drain phase.
- **Verbose output cleanup:** `vlog()` now checks `g_progress_active` flag and clears the progress line (`\r\033[K`) before printing, so `-v`/`-vv`/`-vvv` messages don't overlap the progress bar.
- **GPU ready message:** Now shows device ID: `GPU 7 ready  semaphore scheduling active`.
- **Decompress progress lifetime:** Progress bar stays alive through the entire DirectWriter finalize + file close, replacing the old one-shot "writing..." message.

### Writer Backpressure (v0.11.24)  POSITIVE (major)
Prevents CPU decompression workers from producing data faster than the NVMe can write, which was causing massive kernel writeback pressure in hybrid mode.
- **WriterBackpressure class:** Tracks produced vs physically-written bytes with hysteresis (4 GiB high-water / 2 GiB low-water). CPU workers block on a CV when backlog exceeds high-water, wake instantly when it drops below low-water. GPUs are never throttled (batches in-flight).
- **AIO worker** calls `mark_written()` after each physical write, waking blocked CPU workers.
- **CPU decomp workers** call `wait_if_backlogged()` before popping a new task, `mark_produced()` after delivering a decompressed frame.
- **GPU decomp workers** call `mark_produced()` for accurate backlog accounting but are never blocked.
- No artificial sleeps  all coordination via condition variables with instant wakeup.

**Knuth (H100 × 8, 432 GiB file) hybrid decompress:**

| Metric | v0.11.22 (before) | v0.11.24 (after) | Change |
|--------|-------------------|------------------|--------|
| Wall clock | 6m07s | **3m56s** | **-36%** |
| user time | 4m36s | 2m55s | -37% |
| sys time | **19m11s** | **6m27s** | **-66%** |
| Throughput | 1.18 GiB/s | **1.84 GiB/s** | **+56%** |

Hybrid went from worst of all three modes (6m07s) to best of all three (3m56s), beating both CPU-only (4m42s, 1.53 GiB/s) and GPU-only tuned (4m13s, 1.72 GiB/s). The sys time drop from 19m to 6m confirms the root cause: 96 CPU threads were flooding the kernel with write syscalls. Backpressure keeps CPU workers productive during GPU batch gaps while preventing writer saturation.

## Key Lessons Learned (Updated)

15. **Writer saturation is the hidden hybrid killer.** When CPU + GPU both produce decompressed data at full speed, the writer can't keep up and the kernel spends more time on writeback (19 min sys) than actual work. Backpressure with hysteresis (block at 4 GiB, wake at 2 GiB) keeps the pipeline full without flooding it. The 2 GiB low-water gives the writer ~1 second of runway at NVMe speed  enough to never starve.

---

## Early Benchmark History (Knuth, v0.9.50v0.9.59)

Baseline: v0.9.51 CPU-default avg compress 7.7s (1.06 GiB/s), decompress 11.2s (0.72 GiB/s).
All times are averages across 5 data types (8 GiB each).

| Version | Key Changes | CPU Compress | CPU Decompress | GPU Decompress | Hybrid Compress | Hybrid Decompress |
|---------|------------|-------------|----------------|----------------|-----------------|-------------------|
| v0.9.50 | Pre-optimization baseline | 10.65s | 11.02s | -- | -- | -- |
| v0.9.50-opt | DCtx, offset buf, early release | 10.74s | 11.29s | -- | -- | -- |
| v0.9.51 | Batch 8, streams 1, pre-alloc | 7.69s | 11.19s | 13.37s | 7.90s | 11.66s |
| v0.9.52 | 256 threads, 80% start | 8.26s | 11.10s | 16.54s | 10.41s | 15.53s |
| v0.9.53 | +mmap, +pinned decomp | 9.25s | 12.55s | 25.55s | 9.90s | 17.24s |
| v0.9.54 | -pinned (mmap only) | 8.14s | 11.32s | 15.04s | 9.81s | 13.80s |
| v0.9.55 | -mmap (scheduler+IO prio only) | 7.65s | 10.84s | 19.70s | 7.43s | 13.39s |
| v0.9.56 | nice(-5) skip in GPU-only | 8.14s* | -- | 17.69s | -- | -- |
| v0.9.57 | Decomp pinned fully removed | 8.01s* | -- | 12.75s | -- | -- |
| v0.9.58 | CUDA sync warm-up | 11.27s* | -- | 14.72s* | -- | -- |
| v0.9.59 | CUDA async warm-up | 11.35s* | -- | 14.94s* | -- | -- |

*GPU-only compress only (targeted benchmark). -- = not tested in isolation.
v0.9.56 and v0.9.57 were targeted GPU-only benchmarks (1 iteration). v0.9.55 was a full 3-iteration sweep.

**Key early observations:**

- **Scale matters.** DCtx reuse, offset buffer, and early release showed negligible impact at 8 GiB / 128 frames. They would matter at 500+ GiB / 10,000+ frames where per-frame overhead accumulates. This was confirmed by the CV scheduling work (v0.11.21) which targets the same scaling issue.

- **Hybrid mode works when conditions are right.** v0.9.51 hybrid beat both CPU and GPU for high-compressibility compression (5.86s vs 7.23s CPU vs 6.63s GPU). The adaptive scheduler can find the optimal balance  but only if the I/O pipeline is not starved and the thread count is sensible.

- **mmap needs true zero-copy to win.** mmap with `t.data.assign()` (which copies from mapped pages) is worse than fread for sequential I/O. The benefit only materializes with zero-copy access (`string_view`/`span` into mapped region) or random-access patterns.

- **Pinned memory requires architectural fit.** Pinned memory is a powerful optimization when used correctly (async overlap, modest sizes). Naively bolting it onto a synchronous pipeline with massive allocations makes things worse. Three separate attempts (v0.9.53, v0.11.17, and a third) all failed. CUDA's pageable transfer path is internally optimized with its own pinned staging.

---

*Note: This file supersedes the former PERFORMANCE_LOG.md, which covered v0.9.50v0.9.59 in detail. All content from that file has been integrated here.*
