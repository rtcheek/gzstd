# gzstd Performance Optimization Log

**Project:** gzstd -- Hybrid CPU+GPU Zstd compression/decompression  
**Hardware:** 256-core CPU, 8x NVIDIA GPUs, NVMe storage  
**Test data:** 8 GiB files (high_compress, low_compress, medium_compress, mixed, zeros)  
**Baseline:** v0.9.51 CPU-default avg compress 7.7s (1.06 GiB/s), decompress 11.2s (0.72 GiB/s)  
**Current best:** v0.9.55 = v0.9.51 + scheduler tuning + I/O priority boost

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| Positive | Measurable improvement observed |
| Negative | Performance regression observed |
| Negligible | Within measurement noise (plus or minus 5%) |
| Untested | Implemented but not yet benchmarked |
| Proposed | Discussed but not yet implemented |

---

## Tested Optimizations

### 1. Thread-Local Decompression Context (DCtx Reuse)

**Version:** v0.9.50  
**Status:** Negligible

**Hypothesis:** Creating and destroying a ZSTD_DCtx per frame adds overhead. Reusing a thread-local DCtx avoids repeated allocation/initialization.

**Implementation:** Changed from ZSTD_decompress() (creates/destroys context per call) to a thread-local ZSTD_DCtx created once per worker thread and reused across all frames.

**Observed:** 0.98x (within noise). Baseline avg decompress 11.02s vs optimized 11.29s.

**Explanation:** With 128 frames across 256 threads, each thread processes roughly one frame. DCtx alloc/free takes microseconds versus ~80ms to decompress a 64 MiB frame. Savings invisible at this scale.

**When it would help:** Thousands of small frames (e.g., 1 MiB chunks on a 100+ GiB file = 100,000+ frames) where per-frame overhead accumulates.

**Decision:** Kept. Zero cost, correct optimization for small-chunk workloads.

---

### 2. Offset-Based Buffer (Eliminating buf.erase memmove)

**Version:** v0.9.50  
**Status:** Negligible

**Hypothesis:** The frame reader called buf.erase(0, N) after each frame parse -- an O(n) memmove of the entire remaining buffer. Replacing with an offset cursor and compacting only when necessary should speed up reading.

**Implementation:** Replaced buf.erase() with buf_off cursor. Only memmove when the unconsumed tail is too small to hold a new read chunk.

**Observed:** 0.99x (within noise).

**Explanation:** The reader thread is not the bottleneck at 128 frames. Total memmove savings (~512 MiB across all frames) is tiny compared to worker time.

**When it would help:** Very high frame counts where the reader becomes the bottleneck; also cleaner code that avoids quadratic behavior.

**Decision:** Kept. Cleaner code, zero cost, prevents pathological behavior at scale.

---

### 3. Early Data Release (Compressed Buffer Lifetime)

**Version:** v0.9.50  
**Status:** Negligible

**Hypothesis:** After GPU compression, the original uncompressed Task.data stays alive until task scope ends. Releasing immediately after H2D copy reduces peak memory pressure.

**Implementation:** std::move / shrink_to_fit on Task.data after H2D copy.

**Observed:** No measurable difference.

**Explanation:** 128 x 64 MiB = 8 GiB of frame data; system has enough RAM that holding buffers a bit longer does not cause allocation stalls.

**When it would help:** Files larger than available RAM, or systems with memory pressure from other processes.

**Decision:** Kept. Good hygiene, helps with larger files.

---

### 4. GPU Default Tuning: Batch Size and Stream Count

**Version:** v0.9.51  
**Status:** Positive (compression), Negligible (decompression)

**Hypothesis:** Default gpu_batch_cap=16 and gpu_streams=3 are suboptimal. 512 MiB benchmarks showed batch-8 and streams-1 as optimal.

**Implementation:** Changed DEFAULT_GPU_BATCH_CAP from 16 to 8, DEFAULT_GPU_STREAMS from 3 to 1.

**Observed:** GPU compress improved modestly on high-compressibility data. GPU decompress unchanged at a fixed ~13.4s floor regardless of tuning.

**Explanation:** Smaller batches launch sooner; single stream avoids context-switching. But the dominant GPU cost is PCIe transfer, not kernel scheduling.

**Decision:** Kept. Correct direction, minor gains.

---

### 5. Pre-Allocated GPU Decompression Buffers

**Version:** v0.9.51  
**Status:** Negligible

**Hypothesis:** GPU decompress called cudaMalloc 9 times and cudaFree 9 times per batch (150-300ms overhead per file). Pre-allocating and reusing eliminates this.

**Implementation:** New DecompStreamCtx with ensure_buffers() -- allocates once on first batch, reuses across all subsequent batches.

**Observed:** GPU-only decompress remained ~13.4s. Allocation overhead was real but dwarfed by PCIe transfer.

**When it would help:** Many small files processed sequentially (amortized init cost matters), or GPUs with faster PCIe where malloc overhead becomes proportionally larger.

**Decision:** Kept. Eliminates real overhead, matters more with different hardware.

---

### 6. Hybrid Scheduler: 256 Threads + 80% CPU Start

**Version:** v0.9.52  
**Status:** Negative (severe regression)

**Hypothesis:** Starting at 25% CPU share with only hw/3 (capped at 32) CPU threads left 75% of CPU capacity idle during 2-5s GPU warm-up. Using all 256 threads at 80% share should maximize utilization.

**Implementation:** CPU threads: hw/3 capped at 32 changed to full hardware_concurrency() (256). Initial share: 25% changed to 80%. Tick: 800ms changed to 300ms. EMA alpha: 0.2 changed to 0.5.

**Observed:** Massive regression across ALL configs:

| Config | v0.9.51 | v0.9.52 | Ratio |
|--------|---------|---------|-------|
| Hybrid compress | 7.90s | 10.41s | 0.76x |
| Hybrid decompress | 11.66s | 15.53s | 0.75x |
| CPU-only compress | 7.69s | 8.26s | 0.93x |
| GPU-only decompress | 13.37s | 16.54s | 0.81x |

**Root cause:** 256 worker threads overwhelmed the OS scheduler. The reader thread (single-threaded fread) and writer thread (single-threaded fwrite) could not get CPU time to keep the pipeline fed. Even GPU-only mode depends on CPU for I/O, so everything degraded.

**Key lesson:** The I/O pipeline (reader + writer) is the critical path. Thread count and I/O thread starvation matter more than scheduler share percentage. You cannot simply throw more threads at the problem.

**Decision:** Reverted thread count. Kept faster tick rate and convergence.

---

### 7. Pinned (Page-Locked) Memory for GPU Decompression

**Version:** v0.9.53  
**Status:** Negative (catastrophic for GPU decompress)

**Hypothesis:** GPU decompress uses pageable memory for H2D and D2H transfers. With pageable memory, cudaMemcpyAsync is actually synchronous -- the CUDA driver copies to an internal pinned staging buffer first. Using cudaHostAlloc (page-locked) memory enables true async DMA without CPU involvement.

**Implementation:** Added h2d_pinned and d2h_pinned buffers to DecompStreamCtx via cudaHostAlloc. H2D: memcpy to pinned then cudaMemcpyAsync (DMA). D2H: cudaMemcpy to pinned (DMA) then memcpy to output vector.

**Observed:** GPU-only decompress nearly doubled in time:

| File | v0.9.51 | v0.9.53 (pinned) | v0.9.54 (no pinned) |
|------|---------|-------------------|---------------------|
| high_compress | 13.76s | 23.28s | 14.54s |
| low_compress | 13.34s | 27.98s | 14.98s |
| medium_compress | 13.46s | 24.99s | 15.67s |
| mixed | 13.17s | 27.00s | 15.30s |
| zeros | 13.12s | 24.49s | 14.70s |
| **Average** | **13.37s** | **25.55s (0.52x)** | **15.04s** |

Removing pinned memory (v0.9.54) recovered most of the loss, confirming it was the culprit.

**Root cause analysis:**
1. Pinned memory allocation cost: cudaHostAlloc for batch_n x alloc_decomp (8 x 64 MiB = 512 MiB) per stream context is expensive -- the kernel must pin pages and set up DMA mappings.
2. System pinned memory limits: 8 GPUs each requesting 512+ MiB of pinned memory may exceed system limits, causing fallback or contention.
3. Double-copy overhead: D2H path now does GPU to pinned buffer to memcpy to output vector, adding a redundant memcpy that did not exist before.
4. The theoretical benefit assumes async overlap -- but the code uses synchronous cudaMemcpy for D2H readback anyway, so there is no overlap to exploit.

**When it would help:** If the decompress pipeline were redesigned to be truly asynchronous (streaming D2H while the next batch computes), with modest pinned buffer sizes (one frame at a time, not the entire batch). Also more effective with fewer GPUs competing for pinned memory.

**Decision:** Reverted. Needs architectural redesign to benefit.

---

### 8. Memory-Mapped I/O (mmap)

**Version:** v0.9.53-v0.9.54  
**Status:** Negative (compress regression, mixed decompress)

**Hypothesis:** The single-threaded fread reader is a bottleneck feeding 100+ worker threads. mmap maps the entire file into the address space, letting the kernel handle readahead. Workers reference mapped pages directly without a centralized reader.

**Implementation:** MmapFile class wrapping mmap/munmap with MADV_SEQUENTIAL. All producer paths (CPU compress, GPU compress, decompress frame reader) try mmap first, fall back to fread for stdin/pipes.

**Observed (v0.9.54 = mmap only, no pinned, vs v0.9.51 = no mmap):**

| Config | Mode | v0.9.51 | v0.9.54 | Ratio |
|--------|------|---------|---------|-------|
| CPU-default | compress | 7.69s | 8.14s | 0.94x |
| CPU-default | decompress | 11.19s | 11.32s | 0.99x |
| GPU-only | compress | 8.71s | 10.87s | 0.80x |
| GPU-only | decompress | 13.37s | 15.04s | 0.89x |
| Hybrid | compress | 7.90s | 9.81s | 0.81x |
| Hybrid | decompress | 11.66s | 13.80s | 0.84x |

CPU-only decompress was roughly flat (some individual files faster, some slower), but compress was consistently 5-20% slower across all backends.

**Root cause analysis:**
1. Not truly zero-copy: Despite mmap, t.data.assign() still copies data from mapped pages into the Task's vector. The copy happens on page faults instead of fread, but it is still a copy.
2. Page fault serialization: On a 256-core machine, multiple threads faulting into nearby pages serialize through the kernel's page fault handler. fread pre-fills userspace buffers without this contention.
3. fread is already optimal for sequential I/O: The kernel's readahead with buffered I/O is well-optimized for sequential access patterns. mmap's advantage is random access and true zero-copy, neither of which we exploit.
4. GPU paths hurt more because the GPU compress producer slices data into subchunks (two nested loops), causing more scattered page faults.

**When it would help:** If tasks held string_view or span pointers into the mapped region (true zero-copy, no assign/copy), which requires careful lifetime management. Also useful for random-access patterns (e.g., seeking to specific frames) or when the file is already in page cache.

**Decision:** Reverted. fread is better for our sequential workload.

---

### 9. Scheduler Tuning: hw/2 Threads + 50% Start + I/O Priority

**Version:** v0.9.55-v0.9.57 (current)  
**Status:** Positive

**Hypothesis:** Fix the v0.9.52 regression with a balanced approach: enough threads to utilize CPU (hw/2 = 128 on 256-core), neutral starting share (50%), faster adaptation (300ms ticks), and nice(-5) on reader/writer to prevent I/O starvation.

**Implementation:**
- CPU threads in hybrid: hw/2 (min 4) -- 128 threads on 256-core
- Initial share: 50% (neutral start)
- Tick: 300ms, sleep: 100ms (roughly 3x more adaptation than v0.9.51)
- Early convergence: alpha=0.5 for first 5 ticks, then 0.3
- Upper bound: 95% (was 80%)
- nice(-5) on reader, writer, and producer threads (skipped in GPU-only mode)

**Observed (v0.9.55, 3-iteration sweep vs v0.9.51):**

| Config | Mode | v0.9.51 | v0.9.55 | Ratio |
|--------|------|---------|---------|-------|
| CPU-default | compress | 7.69s | 7.65s | 1.00x |
| CPU-default | decompress | 11.19s | 10.84s | 1.03x |
| Hybrid | compress | 7.90s | 7.43s | 1.06x |
| GPU-only | compress | 8.71s | 8.01s* | 1.09x |
| GPU-only | decompress | 13.37s | 12.75s* | 1.05x |

*Measured in v0.9.57 after fixing decomp pinned memory and nice(-5) in GPU-only mode.

**Key finding:** nice(-5) hurt GPU-only decompress (19.70s with nice vs 17.69s without). Removed from GPU-only mode in v0.9.56. Decomp pinned memory that was still present was the main regression source -- fully removed in v0.9.57 recovered to 12.75s.

**Decision:** Kept. Genuine improvements for hybrid compress and GPU modes.

---

## Proposed Optimizations (Not Yet Tried)

### 10. GPU-Direct Storage (GDS)

**Status:** Proposed  
**Expected impact:** High (eliminates H2D PCIe transfer)

NVMe to GPU VRAM transfers directly, bypassing CPU and system memory entirely. Eliminates the H2D leg: NVMe to PCIe switch to GPU instead of NVMe to CPU RAM to PCIe to GPU.

**Requirements:** NVIDIA GPUs with GDS support, compatible NVMe controller, libcufile / nvidia-fs kernel module.

**Best for:** Compression (all input data must go H2D). Less impactful for decompression of high-ratio files where compressed data is small.

**Risks:** Hardware-specific, complex setup, not universally available.

---

### 11. Compression-Aware Scheduling

**Status:** Proposed  
**Expected impact:** Medium (better resource utilization for mixed content)

Route chunks based on compressibility. Highly compressible goes to GPU (small D2H return). Low compressibility goes to CPU (avoids large D2H transfer). Sample first few KB per chunk for entropy estimation.

**Rationale:** Benchmarks show GPU excels at high_compress (6.6s vs 7.2s CPU) but loses badly on low_compress (13.0s vs 9.7s CPU). Smart routing could get the best of both.

**Risks:** Sampling overhead, prediction accuracy, added complexity.

---

### 12. Pipeline Overlap (Double-Buffering)

**Status:** Proposed  
**Expected impact:** Medium-High (overlaps H2D/compute/D2H)

Use 2 CUDA streams per device to overlap phases: while stream A runs the kernel, stream B does H2D for the next batch. Requires CUDA events for synchronization.

**Key insight from pinned memory failure:** The current D2H is synchronous (cudaMemcpy, not async). True pipelining requires async D2H with pinned memory, but done correctly -- small pinned buffers per frame, not per batch, with actual overlap between stages.

**Risks:** Higher GPU memory (two batches in flight), complex synchronization. PCIe bandwidth is still the ceiling.

---

### 13. GPU Staggering Across Devices

**Status:** Proposed  
**Expected impact:** Medium (reduces PCIe contention)

With 8 GPUs sharing PCIe lanes, simultaneous transfers saturate the bus. Stagger operations: GPUs 0-3 compute while GPUs 4-7 transfer, then swap.

**Risks:** Complex coordination, may hurt individual GPU latency.

---

### 14. Parallel Output Writing (pwritev)

**Status:** Proposed  
**Expected impact:** Low-Medium (writer is not always the bottleneck)

For decompression, output frames have predictable sizes and offsets. Use pwrite/pwritev to write multiple frames simultaneously from different threads.

**Risks:** Less useful for compression (output sizes unknown until compressed). Writer thread does not appear to be the primary bottleneck in current benchmarks.

---

### 15. CUDA Context Warm-Up

**Version:** v0.9.58-v0.9.59  
**Status:** Negative (reverted)

**Hypothesis:** The first CUDA call on each device takes 2-5 seconds for driver init and context creation. Pre-initializing all devices in parallel before GPU workers start should hide this latency.

**Implementation:** v0.9.58: synchronous cuda_warmup() calling cudaSetDevice + cudaFree(nullptr) on all devices, joined before proceeding. v0.9.59: async version that launches warmup threads and joins just before GPU workers start, overlapping with queue/scheduler/CPU pool setup.

**Observed:**

| Version | Compress avg | Decompress avg |
|---------|-------------|----------------|
| v0.9.57 (no warmup) | 8.01s | 12.74s |
| v0.9.58 (sync warmup) | 11.27s (+3.3s) | 14.72s (+2.0s) |
| v0.9.59 (async warmup) | 11.35s (+3.3s) | 14.94s (+2.2s) |

Both versions added ~3s to compress and ~2s to decompress.

**Root cause:** CUDA contexts are per-thread. Warming up device D from a temporary thread creates a context for that thread, but when the actual GPU worker calls cudaSetDevice(D) from its own thread, it still creates a new context. The warm-up work is completely wasted -- it only adds overhead from creating and destroying throwaway contexts.

**When it would help:** If the GPU workers reused the warm-up threads instead of creating new ones. Or if using CUDA's primary context sharing (cuDevicePrimaryCtxRetain), which shares a single context across threads for the same device. However, this requires switching from the CUDA runtime API to the driver API.

**Decision:** Reverted. CUDA per-thread context model makes this approach ineffective.

---

### 16. Async D2H with Frame-Level Pinned Buffers

**Status:** Proposed (refined from failed optimization #7)  
**Expected impact:** Medium-High

Instead of batch-level pinned buffers (512 MiB per stream -- what failed in v0.9.53), use a small rotating pool of frame-sized pinned buffers (e.g., 2-4 x 64 MiB = 128-256 MiB total). Stream decompressed frames back to host with true cudaMemcpyAsync while the next frame decompresses.

**Key difference from #7:** Small pinned pool shared across streams vs. massive per-stream allocation. Async overlap vs. synchronous copy. Frame-level granularity vs. batch-level.

**Risks:** Complex lifetime management, but directly addresses the "GPU decompress is slow because D2H is synchronous" root cause.

---

### 17. Compress on GPU, Decompress on CPU (Asymmetric Mode)

**Status:** Proposed  
**Expected impact:** Medium

Benchmarks consistently show GPU wins at compression (especially high-ratio data) while CPU wins at decompression (no D2H transfer needed). A mode that always uses GPU for compress and CPU for decompress would match each operation to its best processor.

**Risks:** Simple to implement as a default recommendation. Does not help when the user explicitly needs GPU decompression.

---

## Key Observations

### The PCIe Wall
The 8 GPUs share PCIe bandwidth. GPU compute is fast, but H2D and D2H transfers saturate the bus. This is the single biggest bottleneck for GPU performance. Any optimization that reduces PCIe traffic (GDS, compression-aware routing, asymmetric mode) will have the most impact.

### CPU I/O Pipeline is Critical
The reader and writer threads are serial bottlenecks. Even GPU-only mode depends on the CPU to feed data in and write results out. Starving these threads (too many workers, no priority boost) causes system-wide regression. This was the v0.9.52 lesson.

### GPU Decompress is Fundamentally Disadvantaged
GPU decompression must transfer 8 GiB of decompressed data D2H. CPU decompression keeps data in system memory. The ~13-15s GPU floor is dominated by PCIe transfer, not compute. This makes GPU decompression only viable when CPU cores are busy with something else.

### Pinned Memory Requires Architectural Fit
Pinned memory is a powerful optimization when used correctly (async overlap, modest sizes). Naively bolting it onto a synchronous pipeline with massive allocations makes things worse. The failed experiment (#7) and the proposed fix (#16) illustrate this.

### mmap Needs True Zero-Copy to Win
mmap with t.data.assign() (which copies from mapped pages) is worse than fread for sequential I/O. The benefit only materializes with zero-copy access (string_view/span into mapped region) or random-access patterns.

### Scale Matters
Optimizations #1-3 (DCtx reuse, offset buffer, early release) showed negligible impact at 8 GiB / 128 frames. They would likely matter at 500+ GiB / 10,000+ frames where per-frame overhead accumulates.

### Hybrid Mode Works When Conditions Are Right
v0.9.51 hybrid beat both CPU and GPU for high-compressibility compression (5.86s vs 7.23s CPU vs 6.63s GPU). The adaptive scheduler can find the optimal balance -- but only if the I/O pipeline is not starved and the thread count is sensible.

---

## Benchmark History

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
| v0.9.60 | Revert warm-up (= v0.9.57) | pending | pending | pending | pending | pending |

*GPU-only compress only (targeted benchmark). -- = not tested in isolation.

Note: v0.9.56 and v0.9.57 were targeted GPU-only benchmarks (1 iteration). v0.9.55 was a full 3-iteration sweep.

All times are averages across 5 data types (8 GiB each).

---

## Recommended Next Steps (Priority Order)

1. **Benchmark v0.9.60** -- confirm clean revert matches v0.9.57 performance
2. **Asymmetric mode (#17)** -- GPU compress + CPU decompress as smart default
3. **Compression-aware scheduling (#11)** -- route by compressibility for hybrid mode
4. **Async D2H with small pinned pool (#16)** -- proper pipeline overlap for GPU decompress
5. **GPU staggering (#13)** -- reduce PCIe contention across 8 GPUs
