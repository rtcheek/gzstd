# gzstd Optimization Changelog

**Covers:** v0.9.50 → v0.10.6  
**Test machines:**
- **Knuth:** 256-core CPU, 8× NVIDIA H100 (95 GiB VRAM each), NVMe ~3 GiB/s write
- **Lovelace:** 24-core CPU, 2× NVIDIA RTX 2080 Ti (10 GiB VRAM each), NVMe ~1.8 GiB/s write

---

## Write Path Optimizations

### O_DIRECT Writer (v0.9.71) — POSITIVE
Bypasses page cache for sequential writes. Uses 16 MiB aligned buffer, flushes in aligned chunks.
- **Knuth:** Writer I/O improved 1.1 → 2.72 GiB/s on 432 GiB file
- **Why it works:** Avoids double-buffering through page cache for large sequential writes
- **Caveat:** Unaligned tail requires dropping O_DIRECT via fcntl for final write

### pwrite for Out-of-Order Decompress (v0.9.72) — NEGATIVE (reverted)
Tried using pwrite() to write decompressed frames directly to their final offset without waiting for in-order delivery.
- **Knuth:** 0.93 GiB/s (worse than sequential 2.72 GiB/s)
- **Why it failed:** 27k individual O_DIRECT pwrite calls = massive kernel DMA setup overhead. sys time: 12m45s.
- **Lesson:** O_DIRECT pwrite per-frame is catastrophically expensive. Sequential batch drain is better.

### Async Double-Buffered Write Pool (v0.9.73) — POSITIVE
Background write thread with one pending slot. Writer collects batch → submits to pool (non-blocking) → collects next batch while pool writes previous.
- **Knuth:** Improved overlap between GPU D2H and disk writes
- **Why it works:** Writer thread doesn't block on disk I/O; can collect next batch while previous is being written

### Sparse File Support (v0.9.73) — POSITIVE (for zero-heavy data)
Scans 4K blocks for zeros, lseek past them instead of writing. Integrated with both O_DIRECT (DirectWriter::seek_forward) and fwrite paths.
- **Knuth:** zeros.bin decompress: sparse=5.2s vs no-sparse=6.9s (~25% faster)
- **Why it works:** Avoids physical writes for zero-filled regions
- **Caveat:** O_DIRECT seek_forward must flush internal buffer before seeking. Added --[no-]sparse flag matching zstd syntax.

### io_uring Writer — NOT YET TRIED
Proposed: Replace O_DIRECT write() with io_uring for less syscall overhead per write.
- **Expected:** 10-20% improvement on NVMe drives where per-syscall overhead is significant
- **Rationale:** NVMe drives have deep internal queues; io_uring can submit multiple writes without syscalls

### mmap + memcpy Writer — NOT YET TRIED
Proposed: mmap output file at target size, memcpy frames directly. Kernel handles writeback.
- **Expected:** Good for sparse data (unmapped pages stay as holes), possibly worse for dense data
- **Risk:** mmap as INPUT was already tried and was negative (v0.9.53-54)

### Multiple pwrite Threads — NOT YET TRIED
Proposed: Open output file multiple times, pwrite from multiple threads at known offsets.
- **Expected:** Could double NVMe throughput by increasing queue depth
- **Risk:** O_DIRECT pwrite per-frame was catastrophic (v0.9.72); would need large contiguous writes

### Page-Cache Path for Trivial Data — NOT YET TRIED
Proposed: When >90% of blocks are zero, drop O_DIRECT and use fwrite + ftruncate. This is what zstd does — the page cache handles sparse much more efficiently.
- **Expected:** Match zstd's 2-3s on zeros.bin (currently 4-5s)
- **Rationale:** zstd achieves 0.3s sys time on zeros vs our 5-8s with O_DIRECT sparse

---

## Read Path Optimizations

### mmap Input (v0.9.53-54) — NEGATIVE (reverted)
Replaced fread with mmap for zero-copy reading.
- **Why it failed:** mmap with t.data.assign() still copies from mapped pages (not zero-copy). Worse than fread for sequential I/O due to page fault overhead and TLB pressure.
- **Lesson:** mmap only wins with true zero-copy (string_view/span) or random access patterns. Sequential fread is hard to beat.

### Offset-Based Buffer (v0.9.50) — NEGLIGIBLE (kept)
Replaced buf.erase(0,N) O(n) memmove with offset cursor.
- Correct optimization, prevents pathological quadratic behavior, but invisible at 128-frame scale.

---

## GPU Memory & Transfer Optimizations

### Pinned (Page-Locked) Memory for GPU Decompress (v0.9.53) — NEGATIVE (catastrophic, reverted)
cudaHostAlloc for H2D/D2H staging buffers to enable true async DMA.
- **Knuth:** GPU decompress nearly doubled: 13.4s → 25.6s
- **Why it failed:** Massive pinned allocations (512 MiB per stream) starved system memory, caused page faults in other threads, and the copy-to-pinned + DMA was slower than direct pageable transfer for our access pattern.
- **Lesson:** Pinned memory requires small rotating pools, not batch-sized allocations. The extra memcpy to/from pinned staging negated any DMA benefit.

### Frame-Level Pinned Buffer Pool — NOT YET TRIED (proposed v0.9.55)
Small rotating pool of frame-sized pinned buffers (2-4 × 16 MiB) shared across streams. True async overlap.
- **Key difference from failed #7:** Small pool vs massive per-stream allocation. Would enable cudaMemcpyAsync to actually overlap with kernel.

### Pre-Allocated GPU Decompress Buffers (v0.9.51) — NEGLIGIBLE (kept)
ensure_buffers() allocates once, reuses across batches. Saves ~150-300ms of cudaMalloc/cudaFree per file.
- Invisible at 8 GiB scale but correct for repeated small files.

### VRAM-Aware Batch Sizing (v0.9.96-98) — POSITIVE
Binary search for largest compress batch that fits in VRAM. Includes nvCOMP temp workspace in estimate.
- **Lovelace (10 GiB VRAM):** Finds batch=104 instead of hanging on batch=256
- **Why it matters:** cudaMalloc can hang on some drivers if request exceeds VRAM. Pre-check avoids this.
- Fixed partial allocation leak on retry (free_stream_buffers_only before halving).

---

## CUDA Context & Init Optimizations

### CUDA Context Warm-Up (v0.9.58-59) — NEGATIVE (reverted)
Pre-initialize CUDA contexts on all devices before GPU workers start.
- **Both sync and async versions added ~3s overhead**
- **Why it failed:** CUDA contexts are per-thread. Warming up from a temporary thread creates a throwaway context; the actual GPU worker creates its own anyway.
- **Lesson:** CUDA per-thread context model makes warm-up ineffective. Would need cuDevicePrimaryCtxRetain (driver API) to share contexts.

---

## Scheduling & Routing Optimizations

### Hybrid Scheduler: 256 Threads (v0.9.52) — NEGATIVE (reverted)
Full hardware_concurrency() threads at 80% CPU start share.
- **Regression:** 0.75x across all configs
- **Why it failed:** 256 worker threads starved the reader/writer I/O threads. Even GPU-only mode degraded because I/O pipeline couldn't keep up.
- **Lesson:** I/O pipeline (reader + writer) is the critical path. Cap CPU threads below full hardware count.

### Adaptive CPU/GPU Share via EMA (v0.9.52-78) — MIXED
Various attempts at throughput-based adaptive scheduling.
- 50/50 start: CPU ate everything before GPU initialized
- 10/90 GPU-favored start: CPU still drained queue during GPU init
- Throughput measurement: CPU always appeared faster because GPU was starved for data
- **Final solution:** Semaphore-based scheduling (v0.9.83)

### GPU-Priority Semaphore Scheduler (v0.9.83) — POSITIVE
`gpus_waiting` atomic counter. GPU increments before pop, decrements after. CPU yields when counter > 0.
- **Why it works:** Direct, instant priority signaling. No measurement delay. GPU always gets fed first.
- CPU runs wild during GPU init, then yields once GPU signals ready
- CPU helps when all GPUs are busy processing (counter = 0)

### Trivially-Compressed Frame Detection (v0.9.93) — POSITIVE
Decompress: peek at front frame's ratio. If < 2%, CPU takes it regardless of GPU priority.
- **Why it works:** Frames decompressing to mostly zeros are faster on CPU (no PCIe D2H overhead). CPU + sparse writes = near-instant.
- **Knuth:** zeros.bin: CPU path 1.4s vs GPU path 4.4s

### Auto CPU Thread Cap at 96 (v0.9.80) — POSITIVE
Default auto: min(hw-1, 96). -T0 = all threads (matches zstd).
- **Why:** Diminishing returns beyond 96 threads on large-core machines. Leaves headroom for I/O threads.

### --cpu-batch as Queue Depth Threshold (v0.9.92-94) — POSITIVE
Minimum queue depth before CPU workers activate. Each CPU takes 1 frame (no CPU batching benefit).
- **Why:** Keeps queue stocked for GPUs. CPU only helps when there's overflow.

---

## Batch Size Auto-Tuning

### Decompress Greedy Batch Pop (v0.9.69) — POSITIVE (massive)
pop_batch_greedy waits for full batch before GPU processes. DEFAULT_GPU_DECOMP_BATCH_CAP = 256.
- **Knuth:** medium_compress kernel dropped 24.7s → 1.27s (55× speedup!)
- **Why:** Default batch=8 caused 64 kernel launches × 385ms each. Batch=256 = 3 launches × 424ms.

### Continuous Binary-Search Auto-Tuner (v0.10.0-0.10.6) — POSITIVE
Runtime throughput-aware batch sizing for compress. Explores both directions from default.
1. Record baseline throughput at starting batch size
2. Try halving — if better, continue halving
3. Try doubling — if better, continue doubling
4. Settle at best when throughput drops
5. Periodically probe to detect data character changes
- **Lovelace:** Correctly finds batch=8 optimal for compress, settles in 2 steps
- **Fixed bugs:** free_stream_buffers_only wiped tune state (v0.10.4), tune ceiling was default not VRAM limit (v0.10.2), baseline never recorded (v0.10.3)

---

## GPU Selection & Topology

### NVML/NUMA-Aware GPU Selection (v0.9.63-68) — POSITIVE
Queries GPU utilization and NUMA topology. Penalizes GPUs on busy NUMA nodes.
- **Why:** Prevents selecting GPU 6 when GPU 4 (same NUMA node) is busy at 29%.

### --gpu-devices N (v0.9.62) — POSITIVE
Decompress default: 2 GPUs (PCIe bandwidth optimal for 1-2 GPUs).
Compress default: all GPUs.

---

## Performance Instrumentation

### -vvv Breakdown (v0.9.61) — ESSENTIAL
PerfCounters struct with atomic accumulators for every pipeline phase.
- **Bug found (v0.9.89):** Compress GPU worker had TWO completion paths (async poll + sync drain). Only async poll recorded to g_perf. Sync drain path handled majority of completions for small batches.
- **Lesson:** Comment both paths with "MUST record to g_perf — see also other path"

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

### Knuth (H100 × 8) — v0.9.74 vs zstd -T0, 8 GiB files, decompress
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| zeros | 4.85s | 4.40s | 1.10× |
| high_compress | 9.52s | 7.07s | **1.35×** |
| medium_compress | 15.39s | 9.69s | **1.59×** |
| mixed | 9.12s | 6.55s | **1.39×** |
| low_compress | 9.25s | 7.14s | **1.30×** |
| **Total** | **48.13s** | **34.85s** | **1.38×** |

### Lovelace (RTX 2080 Ti × 2) — v0.10.6 vs zstd -T0, 8 GiB files
**Decompress:** gzstd wins 2/5 (medium_compress 1.22×, low_compress 1.06×). Loses on trivial data where zstd's page-cache sparse dominates.
**Compress:** gzstd wins 4/5 (high 1.83×, low 1.54×, medium 1.11×, mixed 1.26×). Only loses zeros.
