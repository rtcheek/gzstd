# gzstd Optimization Changelog

**Covers:** v0.9.50 → v0.12.24  
**Test machines:**
- **Knuth:** 256-core CPU, 8× NVIDIA H100 (95 GiB VRAM each), NVMe ~3 GiB/s write
- **Lovelace:** 256 GiB RAM, 24-core CPU, 2× NVIDIA RTX 2080 Ti (10 GiB VRAM each), NVMe ~1.8 GiB/s write

---

## v0.12.24 — Streaming decompression for oversized frames

**Symptom.** Decompressing single-frame .zst files (e.g., from `zstd -T0` or `gzstd --sliding-window`) showed the `out:` progress bar stuck at 0% for the entire decompression, then jumping to 99.9% at the end. Memory usage spiked to the full decompressed size (e.g., 125 GiB) because the worker allocated one giant buffer for the entire frame.

**Root cause.** `cpu_decomp_worker` called `ZSTD_decompressDCtx` with a single output buffer sized to the full decompressed frame. Nothing was written to disk until the entire frame was decompressed, so the writer (and its progress tracking) had no work to do until the very end.

**Fix.** For frames larger than 64 MiB (`STREAM_THRESHOLD`), the worker now uses `ZSTD_decompressStream` with 16 MiB output chunks. Each chunk is pushed to `ResultStore` with its own sequence number, and `total_tasks` is adjusted upward to account for the sub-chunks. This lets the writer start writing (and updating `out:` progress) as soon as the first 16 MiB is decompressed, rather than waiting for the entire frame.

**Key details:**
- `n_chunks_est = ceil(decomp_size / 16 MiB)` — pre-calculated from the frame header
- `total_tasks` adjusted atomically before streaming begins; corrected after if actual chunk count differs
- Only triggers for frames > 64 MiB; normal multi-frame files (16 MiB frames) take the existing fast path
- FrameThrottle naturally releases permits per sub-chunk, providing backpressure
- Memory usage drops from full-frame to ~16 MiB working set per worker

---

## v0.12.23 — Progressive writeback fix + --sliding-window compression

**Symptom (decompression overwrite stall).** Decompressing low_compress.bin.zst (19.53 GiB output) with `-f` (overwrite) showed a 46-second stall after all decompression work completed, reported as "atomic rename: 46155 ms" in verbose output.

**Root cause.** Buffered `fwrite` of 19.53 GiB accumulated dirty pages in the page cache faster than the kernel's background writeback could drain them. When `rename()` was called on the `.gzstd.tmp` file, ext4's `data=ordered` journaling required flushing ALL dirty pages to disk before committing the metadata transaction — a synchronous 46s wait. Compression didn't suffer because: (1) it's CPU-intensive, giving writeback time to drain; (2) compressed output is smaller.

**Fix.** Added `sync_file_range(fd, offset, len, SYNC_FILE_RANGE_WRITE)` in `AsyncWritePool::worker_fn` after each batch of writes. This non-blocking call tells the kernel to start writing dirty pages to disk immediately rather than letting them accumulate. By the time `rename()` executes, most pages are already on disk.

**Result on Lovelace** (low_compress.bin.zst → existing file with `-f`):
- Atomic rename: 46,155 ms → ~700 ms (**66× faster**)
- No regression to compression or fresh-file decompression paths

---

## v0.12.22 — `--sliding-window` single-frame compression mode

**Motivation.** For highly repetitive data (e.g., random word lists repeated across 125 GiB), gzstd's multi-frame architecture (8000 independent 16 MiB frames) achieved 0.29% ratio while `zstd -T0` achieved 0.01% (31× better). The difference: zstd produces a single frame with a 2 MiB sliding window that maintains context across the entire file, while gzstd's frames each start with a cold window.

**Feature.** New `--sliding-window` flag delegates compression to zstd's built-in multi-threaded mode (`ZSTD_c_nbWorkers`), producing a single standard zstd frame. Trade-offs:
- Ratio matches `zstd -T0` exactly (shared sliding window context)
- Output is a standard .zst file — `zstd -d` can decompress it
- Decompression is single-threaded (one frame = one unit of work for any decompressor)
- Implies `--cpu-only` (GPU/nvCOMP has no sliding window API)

**Validation:**
- `--sliding-window --gpu-only` and `--sliding-window --hybrid` rejected with clear error
- `--sliding-window -d` rejected (compression-only)
- `--sliding-window` without `--cpu-only` auto-enables it with a warning
- Round-trip verified; `zstd --list` confirms single frame; `zstd -d` interop confirmed

**GPU fallback for oversized frames.** When decompressing a single-frame file (from `zstd -T0` or `--sliding-window`), the frame's decompressed size can be hundreds of GiB — far exceeding nvCOMP's 16 MiB per-slot VRAM allocation. gzstd now detects oversized frames during the pre-scan and automatically falls back to CPU-only decompression with a clear warning. `--gpu-only` is gracefully overridden rather than crashing.

**Progress bar fix (mmap compression).** The mmap zero-copy reader enqueued all tasks instantly (pointer arithmetic, no I/O), causing the `in:` progress to jump to 100% immediately. Fixed by deferring `read_bytes` updates to when workers actually pick up each task, so the progress bar reflects real processing throughput.

**Test coverage.** Full `./gzstd-test.sh` suite passes (200/200).

---

## v0.12.21 — mmap zero-copy compression input + benchmark accuracy fix

**Symptom (compression).** CPU-only compression of mixed.bin (19.5 GiB) took 9.9s vs zstd's 6.1s. Profiling showed the single-threaded `fread` producer was the bottleneck — 22 worker threads were starved, achieving only ~1.5 effective cores of utilization.

**Fix.** Memory-map input files for both CPU (`compress_cpu_mt`) and GPU (`compress_nvcomp`) compression paths. Workers read directly from the mapped pages via `view_ptr`/`view_len` on `Task`, eliminating the `fread` + `memcpy` bottleneck. Pipes and stdin fall back to the existing `fread` path. Key changes:

- Added `MmapRegion` RAII class (read-only mmap with `MADV_SEQUENTIAL`)
- Extended `Task` struct with `view_ptr`/`view_len` for borrowed (mmap) data vs owned `std::vector<char> data`, plus `ptr()`, `len()`, `release_input()` helpers
- Updated all consumer touch points: `t.data.data()` → `t.ptr()`, `t.data.size()` → `t.len()`, `std::vector<char>().swap(t.data)` → `t.release_input()`

**Result on Lovelace** (24-core, mixed.bin CPU-only compress):
- Before: 9.9s (1.97 GiB/s)
- After: 3.1s (6.3 GiB/s) — **3.2× faster**, now 1.9× faster than zstd

**Symptom (benchmark).** `gzstd-benchmark.sh` reported compression at ~3.4s / 5.7 GiB/s, roughly 2× faster than manual `time` measurements (~7.2s). Results appeared suspiciously fast.

**Root cause.** `rm -f "$comp_out"` before each iteration deleted the output file. This forced the kernel to create a fresh file (fast new-block allocation) instead of exercising the atomic overwrite path (`write .gzstd.tmp` → `rename`), which contends with the old file's dirty-page writeback. The `sync` in `run_timed` was defeated because `rm` discards dirty pages before sync runs.

**Fix.** Removed `rm -f` calls before both compress and decompress iterations so iterations 2+ exercise the realistic overwrite path. Also changed hardcoded version string to generic "gzstd benchmark suite".

**Failed experiments (reverted, documented here for future reference):**
- **mmap output for decompression**: MAP_SHARED writes from multiple threads caused 9+ min of kernel sys time for 20 GiB due to page-fault contention. Buffered fwrite through the sequential writer thread is faster.
- **MADV_HUGEPAGE on write mmap**: THP defragmentation overhead made everything 2× worse.
- **32K write chunking**: Increased syscall count without improving I/O scheduling.
- **Removing atomic overwrite**: The overwrite penalty comes from kernel dirty-page throttling on gzstd's 16 MiB write chunks (one per decompressed frame), not from the .tmp + rename mechanism. zstd avoids this via streaming decompression with 32 KB writes. This is an architectural difference, not a quick fix.

**Decompression status.** On fresh files, gzstd matches zstd (6.5s vs 5.1s on mixed.bin). The overwrite penalty (2–3×) remains an open issue tied to frame-at-a-time vs streaming decompression architecture.

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.20 — Fix re_enqueue FIFO violation causing throttle deadlock on disk

**Symptom.** Hybrid decompression stalled at ~39.7% on medium_compress.bin when writing to a real file, but completed reliably to `/dev/null`. The v0.12.19 `may_take` fix resolved the GPU-waits-on-CPU deadlock, but a second deadlock remained on disk-backed output.

**Root cause.** `TaskQueue::re_enqueue()` used `push_back`, sending GPU-skipped trivial frames (low sequence numbers) to the *back* of the queue. CPU workers then processed higher-sequence frames first, consuming all `FrameThrottle` permits. The writer needed the low-sequence frames to release permits (sequential ordering), but those frames were stuck behind the high-sequence work. Classic circular wait: workers need permits to produce frames, writer needs low-seq frames to release permits, low-seq frames are queued behind work that needs permits.

This only manifested with real disk I/O because `/dev/null` releases permits instantly (no write latency), so the writer could always drain fast enough to recycle permits before exhaustion.

**Fix.** Changed `re_enqueue` from `push_back` to `push_front` with reverse iteration:
```cpp
for (auto it = batch.rbegin(); it != batch.rend(); ++it)
    q_.push_front(std::move(*it));
```
Reverse iteration preserves original sequence order at the front of the queue. This restores the FIFO invariant that `FrameThrottle` depends on for deadlock freedom: "the frame the writer needs next is always among the oldest in-flight frames."

**Result on Lovelace** (24-core, 2× RTX 2080 Ti, medium_compress.bin.zst → real file):
- Before: stalled at ~39.7% (4/5 runs to disk, 0/5 to `/dev/null` after v0.12.19)
- After: 5/5 real-file completions at 2.96–3.15 GiB/s, 15/15 `/dev/null` at 6.68–7.54 GiB/s

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.19 — Fix hybrid compress/decompress deadlock

**Symptom.** Hybrid decompression hung on medium_compress.bin (~4 out of 5 runs). Diagnostic showed `cpu_taken=0, gpu_taken=0, queue_floor=29, gpus_waiting=0` repeating indefinitely — no worker was making progress.

**Root cause (primary).** The `may_take` predicate in both CPU compress and decompress workers called `sched->should_cpu_take()` unconditionally. `should_cpu_take()` returns false when `gpus_waiting_ > 0`. But there was no `done` bypass: once the producer finished (`queue.set_done()`), CPU workers still deferred to the GPU scheduler. If the GPU was stuck in `bp->acquire` (throttle exhaustion), a CUDA operation, or `pop_batch_greedy` while holding `gpus_waiting_=1`, CPU workers would never take work — permanent deadlock.

**Root cause (secondary).** GPU workers never deregistered their streams from `HybridSched` on exit, so `gpu_queue_floor_` persisted at 29 after GPU exit, blocking CPU workers when `depth <= floor` and `done=false`.

**Fix.**
1. **`may_take` predicate**: Added `if (qs.done) return true;` before the `should_cpu_take()` check in both compress and decompress CPU workers. Once the producer is done, CPU workers drain the queue regardless of GPU state. The redundant `&& !qs.done` guards on floor/cpu_queue_min checks were removed (now handled by the early return).
2. **GPU stream deregistration**: Added `HybridSched::unregister_gpu_stream()` — decrements `active_gpu_streams_`, recalculates `gpu_queue_floor_` to 0, and calls `notify_cpu_waiters()`. Both GPU compress and decompress workers call it on all exit paths.
3. **Defensive wake**: After GPU worker threads are joined, `notify_cpu_waiters()` is called as a safety net.

---

## v0.12.18 — Default to buffered I/O; O_DIRECT now opt-in via --direct

**Motivation.** Despite fixing CPU-side contention in v0.12.17 (thundering herd via CV mismatch), wall-time variance on disk-backed runs remained severe: 2–10× on the same file between consecutive runs. Reference tool zstd showed 1.2× variance on the same workload. The difference: zstd uses buffered `fwrite` (OS page cache absorbs write latency); gzstd used `O_DIRECT` by default, bypassing the cache and exposing every write to NVMe-internal GC, ext4 journal commits, and writeback contention from prior runs' dirty pages.

**Root cause.** `O_DIRECT` writes are synchronous to the device: if the NVMe controller is busy (garbage collection, NAND erase, journal commit from a prior buffered write), each 4 KiB–4 MiB `write()` stalls until the device is ready. With buffered I/O, the kernel coalesces writes in the page cache and flushes to the device at its own pace — the application sees consistent ~1.5–1.9 GiB/s regardless of device state.

**Fix.** O_DIRECT is now off by default. `--direct` opts in; `--no-direct` is accepted for explicitness (already the default). Both the explicit-output-file path and the stdout-redirect-to-file path are gated on `opt.direct_io`.

**Result on Lovelace** (24-core, ext4/NVMe, `mixed.bin` 19.5 GiB, 5-run median):

| Mode | Before (O_DIRECT) | After (buffered) |
|------|-------------------|-------------------|
| CPU compress | 5.7–48.2 s | 5.81–6.67 s (1.15×) |
| CPU decompress | 5.4–31.1 s | 5.37–5.83 s (1.09×) |
| Hybrid compress | 6.6–105.8 s | 6.56–6.93 s (1.06×) |
| Hybrid decompress | stalled at 55% | 6.44–6.75 s (1.05×) |

The "stalled" hybrid decompress was caused by O_DIRECT writes contending with NVMe GC, stalling the aio thread, exhausting throttle permits, and blocking all workers.

**v0.12.16 push_to_slot change reverted.** The earlier "only notify writer when seq == next_to_write" optimization was reverted — not safe in hybrid mode where GPU batch-completion and CPU per-frame notifications interact. Per-CPU-push `notify_one` is cheap (single writer waiter, not a herd).

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.17 — Kill the CPU-side thundering herd (wait_for_work / notify fixes)

**Motivation.** v0.12.10–0.12.15 fixed several pipeline-depth and throttle issues but left a ~1.7× run-to-run variance on CPU-only decompress at high thread counts (22 workers on Lovelace): fast runs ~4.0 s to `/dev/null`, slow runs ~7.0 s on the same cached input. Reducing `-T` from 22 to 4 collapsed both the variance and the absolute time (3.1–3.6 s). Variance scaled with worker count — a contention signature, not a hardware one.

**Root cause 1 — `TaskQueue::wait_for_work()` was waiting on the wrong CV.** `TaskQueue` exposes two condition variables by design: `cv_` for GPU batch waiters (woken by `notify_all` because batch predicates need every waiter to re-check) and `cpu_cv_` for CPU workers (woken by `notify_one` in the push path). Non-hybrid CPU workers called `wait_for_work()`, which — incorrectly — waited on `cv_`. So `push()`'s targeted `cpu_cv_.notify_one()` hit nothing, and the `cv_.notify_all()` it fires for GPU waiters woke **every** CPU worker in the pool on every frame push. 22 threads × ~8000 frames = ~176k spurious wakeups per run, all contending on the same queue mutex as they raced to pop, 21 of them losing each race and going back to sleep.

Fix: `wait_for_work()` and `pop_one()` now wait on `cpu_cv_`. One CPU worker wakes per pushed frame, matching the actual work. `set_done()` still notifies both CVs so shutdown wakes everyone.

Result on `-T 22 --cpu-only -c mixed.bin.zst > /dev/null` (5-run sample): 4.1–7.0 s → 3.25–3.84 s. Variance collapsed from 1.7× to 1.18×, and the *floor* improved — i.e. the "good" runs got faster too, confirming the herd was costing work even in the best case.

**Root cause 2 — `FrameThrottle::release()` used `notify_all`.** With 22+ workers saturated against the throttle, `notify_all` on every single-permit release fan-out-woke the whole pool per aio write. One woke-up worker won the mutex, took the single new permit, 21 went back to sleep after redundantly contending. Replaced with a `notify_one` loop over `n`: wake exactly the number of waiters that the capacity change can satisfy. (v0.12.16 change kept here for clarity.)

**Root cause 3 — `ResultStore::push_to_slot()` CPU path notified on every push.** CPU workers finish out-of-order and push to the shared results map; the writer only cares when its `next_to_write` seq arrives. Waking the writer for every out-of-order push churned it through wake/recheck/sleep cycles. Now only notifies when `seq == next_to_write`; out-of-order frames get drained on the next natural writer cycle. (v0.12.16 change kept here for clarity.)

**Residual variance.** CPU-side variance is resolved (≤1.18×). Writes to `/tmp` (ext4) still show a bimodal pattern (11.5 s fast / ~28 s slow) that does *not* track worker count — this is OS page-cache flush / NVMe writeback behavior between consecutive large writes, not algorithmic. Left for a future pass; out of scope here.

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.15 — Throttle diagnostics, tunables, and coverage

**Deadlock guardrail (late add).** Initial smoke-testing of the new knobs caught a real bug: `--throttle-frames=1` with a GPU path active would hang indefinitely. GPU workers greedy-acquire up to `gpu_batch_cap` permits per batch in one call — with a 1-permit budget, the first stream takes the single permit and blocks waiting for the rest, which only the writer can release, which it can never do because no frame has been pushed yet. Classic producer-consumer circular wait.

`compute_throttle_budget` now takes a `gpu_batch_floor` argument (= `devices × streams × gpu_batch_cap`) and clamps user-provided `--throttle-frames` up to it when GPU is active, with a warning at `-q` or above. The default-formula path sets source to `user+gpu-floor` in the startup log when clamping fires. CPU-only paths pass `gpu_batch_floor=0` and are unaffected.

Regression test added to section 36: hybrid `--throttle-frames=1` round-trip, gated on `has_gpu`, must warn-and-complete in under 30 s.



Follow-up to v0.12.14: instrument the throttle so its behaviour is observable, expose the knobs needed to sweep it, and lock the new surface in with tests.

**FrameThrottle instrumentation.** New atomic counters on every throttle: `block_count` (how many `acquire()` calls actually had to wait), `block_nanos` (cumulative blocked time), `peak_in_flight` (high-water mark vs budget). `acquire()` times the wait from the first `cv_.wait` until the permit is taken; exits without a wait skip the timing hot path entirely. A `Stats` snapshot struct + `stats()` method expose them to the end-of-run logger.

**New CLI knobs.**
- `--throttle-factor=N` — override the default `SLACK_FACTOR=4` slack multiplier. Lets you sweep pipeline tightness without changing source.
- `--throttle-frames=N` — explicit in-flight frame cap; bypasses the parallelism formula entirely. Useful for repro cases and deadlock stress tests.

Both validated (must be >= 1) and reflected in help output. The startup summary labels where the budget came from: `source=user` when `--throttle-frames` is set, `source=pipeline` when the parallelism formula wins, `source=ram` when the RAM cap binds, `source=floor` when the 32-frame minimum binds.

**Verbose output.**
- `-v` (V_VERBOSE): one-line startup summary — `throttle: N frames (X GiB in-flight max, source=..., parallelism=..., slack=...)`.
- `-vv` (V_DEBUG): adds `throttle detail: pipeline_cap=..., ram_cap=..., floor=..., avail_ram=...` at startup and an end-of-run `throttle stats [phase]: peak=P/M (S%), block_count=N, block_time=Xms` line tagged by path (`compress-cpu`, `decompress-hybrid`, `compress-gpu`, etc.).

Saturation (`peak/max`) plus `block_count` tells you at a glance whether the throttle is the bottleneck: low saturation means something upstream (reader, GPU, CPU) is the limiter; high saturation with low `block_time` means workers fill the pipeline but never stall; high saturation with significant `block_time` means the writer is the limiter and backpressure is engaged as designed.

**Test coverage** (section 36 of `gzstd-test.sh`, 8 new tests → 192 total):
- Round-trips at `--throttle-frames=1`, `=32`, and `--throttle-factor=1`, `=16` (deadlock guards at both extremes).
- `-v` output contains the throttle startup line.
- `-vv` output contains the end-of-run stats line.
- `--throttle-frames=0` and `--throttle-factor=0` are rejected with `EXIT_USAGE` (2).

**Benchmark sweep.** `gzstd-benchmark.sh` gains `--sweep-throttle`, which sweeps `--throttle-factor` over `{1,2,4,8,16}` (or `{1,4,16}` with `--quick`) across all enabled paths (cpu-only, hybrid, gpu-only). Rolled into `--sweep-all`.

---

## v0.12.14 — Pipeline-depth throttle budget (principled scaling)

**Motivation:** v0.12.13 fixed the lost-backpressure bug by capping the throttle budget at a hard 8 GiB. That number worked on the two test systems but was arbitrary: too restrictive for a 256-core / 8×H100 server (Knuth) whose pipeline can legitimately hold hundreds of GiB in flight, too generous for a 16 GiB VM where 8 GiB is half of physical RAM. The budget needs to track the machine, not a magic constant.

**Fix:** Replace the fixed byte cap with a formula rooted in observable hardware parallelism:

```cpp
pipeline_frames = (cpu_threads + gpu_count * streams * batch_cap) * SLACK_FACTOR
ram_cap_frames  = (avail_ram / 2) / frame_bytes
frames = max(min(pipeline_frames, ram_cap_frames), 32)
```

`SLACK_FACTOR = 4` gives each active producer ~4 frames of headroom — enough to ride out writer jitter, not so much that we queue hundreds of frames per producer with no throughput payoff. The RAM cap stays as a safety net; the 32-frame floor guards against pathological low-parallelism or huge-chunk configs.

Expected budgets:

| System                         | Parallelism             | Frames   | In-flight |
|--------------------------------|-------------------------|----------|-----------|
| Laptop (8 CPU, no GPU)         | 8                       | 32 (floor) | 512 MiB  |
| 16 GiB VM (4 CPU, no GPU)      | 4                       | 32 (floor) | 512 MiB  |
| Lovelace (24 CPU, 2×1×16 GPU)  | 24 + 32 = 56            | 224      | ~3.5 GiB  |
| Knuth (256 CPU, 8×2×64 GPU)    | 256 + 1024 = 1280       | 5120     | ~80 GiB   |

On Lovelace this is ~2× tighter than the old 8 GiB cap but well above the ~320 MiB the writer actually drains before the next producer wakeup, so no throughput regression is expected. On Knuth it unlocks the pipeline the hardware can actually sustain.

The `-vvv` throttle debug line now shows all inputs: `parallelism=`, `pipeline=` (pre-clamp), `ram_cap=`, plus the chosen frame count and in-flight byte equivalent.

---

## v0.12.13 — Throttle budget byte cap (restore writer backpressure)

**Bug:** On Lovelace (256 GiB RAM) decompression appeared to lose all writer backpressure. Reader, GPUs, and CPUs finished in seconds; the writer then ground through a massive in-RAM backlog at ~88 MiB/s with no throttling of producers. The throttle budget formula `avail_ram / (2 × frame_bytes)` gave ~7,800 frames on a 246 GiB-available box — 123 GiB of permitted in-flight data. Files under that size (mixed.bin.zst at ~20 GiB decompressed = 1,220 frames) fit entirely within the budget, so workers never blocked, decompressed everything immediately, and the writer drained alone.

**Fix:** Cap the budget at an absolute byte ceiling in addition to the RAM-relative calculation:

```cpp
budget_bytes = min(avail_ram / 2, THROTTLE_MAX_BYTES);   // 8 GiB cap
frames = budget_bytes / frame_bytes;
frames = max(frames, 32);                                // min pipeline depth
```

On Lovelace: budget = 512 frames (8 GiB in-flight) instead of ~7,800. Workers fill the pipeline, then block on `acquire(1)` — writer releases permits as frames are written, producers resume. Lockstep backpressure restored.

Fast-I/O systems are unaffected: when the writer can release permits faster than workers acquire them, nobody blocks. Only slow-I/O systems (relative to producer throughput) feel the cap — which is exactly where it's needed.

Prior floor of 1024 was also wrong on low-RAM systems (forced a minimum 16 GiB in-flight on 24 GiB boxes, would have caused swap once the reader's buffer was resident). Floor is now 32 frames.

**Debug log change:** `-vvv` throttle message now reports both the frame count and the byte in-flight max, and the available-RAM figure:
```
throttle: 512 frame budget (8.00 GiB in-flight max, 246.03 GiB avail RAM)
```

---

## v0.12.12 — EMA-scaled hybrid queue floor + tuning knobs

**Bug:** The fixed queue floor introduced in v0.12.9 — `active_gpu_streams × gpu_batch_size` — assumed GPU was strictly faster than CPU per frame. For compression that holds (GPU batches pay off), but for decompression nvCOMP throughput ≈ CPU zstd throughput, so reserving frames for the GPU just idles CPUs. Lovelace benchmarks (v0.12.11) showed hybrid decompression 2–8% slower than the best pure path on every file, and 18% slower than either pure config on `zeros.bin` (3.675 vs 4.482 GiB/s).

**Fix:** `HybridSched` now scales the nominal floor by observed GPU advantage. Each tick (every ≥0.5s) feeds per-side EMA throughput (`cpu_rate_ema_`, `gpu_rate_ema_`, α=0.3) from the already-tracked `cpu_bytes_` / `gpu_bytes_` counters. The floor factor is `clamp((gpu_per_stream − cpu_per_thread) / gpu_per_stream, 0, 1)`:
- Compression: GPU ≫ CPU → factor ≈ 1.0 (nominal reservation, preserves v0.12.9 gains).
- Decompression: GPU ≈ CPU → factor → 0 (CPUs compete freely).

During warm-up (<2 EMA samples on either side), factor defaults to 1.0, matching v0.12.9. Convergence is typical after ~2s (≈6 ticks at α=0.3). The constructor now consumes the `cpu_threads` argument it previously ignored.

**New CLI flags** (gated; defaults preserve v0.12.12 AUTO behaviour):
- `--hybrid-floor=auto|nominal|off`
  - `auto` (default): EMA-scaled as above.
  - `nominal`: v0.12.9 behaviour (`streams × batch`).
  - `off`: no reservation — CPUs compete freely, relying on the `gpus_waiting_` semaphore for GPU priority.
- `--hybrid-floor-factor=X` — manual override in `[0.0, 1.0]`; bypasses mode selection.

`-vvv` tick output adds `floor_factor=` alongside `queue_floor=` for diagnosis.

---

## v0.12.11 — Progress bar: freeze input rate after read completes

**Bug:** After the reader finished (in:100.0%), the displayed input rate (GiB/s) kept declining because it was computed as `read_bytes / total_elapsed_time` — a cumulative average where the denominator grows but the numerator is fixed.

**Fix:** The progress loop snapshots the elapsed time the first time it sees `read_bytes >= total_in` and reuses that frozen duration for subsequent input rate calculations. Added `read_elapsed_ms` (mutable atomic) to `Meter`. The output rate continues using wall-clock time since writes are still in progress.

---

## v0.12.10 — Progress bar: ratio-estimated output percentage

**Bug:** The `out:` progress percentage was misleading during the read phase for both compression and decompression. During decompression, `out:` showed ~56% at only 10% input read because `wrote_bytes / total_out` used a partial (still-growing) denominator — workers complete frames faster than the reader parses new ones, so the ratio is inflated and then drops as the denominator catches up. Compression had the same class of issue (showed `---` until the reader finished).

**Fix:** While the reader is still running (`total_frames` not yet set), the progress bar estimates total output from the current input/output ratio: `estimated_total = total_in × (total_out_so_far / read_bytes_so_far)`. This works for both compression (ratio < 1) and decompression (ratio > 1), converges as more data is read, and only increases monotonically for files with uniform compressibility. Once the reader finishes: decompression switches to exact `wrote_bytes / total_out` (byte-level); compression uses `tasks_done / total_frames` (frame-level) then `wrote_bytes / total_out` during AIO drain. Added `total_out_final` flag to `Meter` to distinguish finalized vs partial `total_out`.

---

## v0.12.9 — GPU queue depth reservation (hybrid scheduler)

**Bug:** On Knuth, hybrid mode was ~10% slower than `--gpu-only`. Scheduler stats showed CPUs took 18,344 tasks vs GPUs 9,341 — despite CPUs being ~6× slower per task (0.21 vs 1.19 GiB/s). The `should_cpu_take()` gate only blocked CPUs when `gpus_waiting > 0`, but GPUs cycle through wants→got in microseconds. During the much longer GPU processing phase (milliseconds), `gpus_waiting == 0` and all 96 CPUs flooded the queue, leaving it empty when GPUs came back for their next batch.

**Fix:** `HybridSched` now tracks total active GPU streams (`register_gpu_stream()`) and current batch size (`set_gpu_batch_size()`). A dynamic queue floor = `active_streams × batch_size` reserves enough tasks for every GPU stream to fill one full batch. The `may_take` lambda in both compress and decompress CPU workers checks `qs.depth <= floor` and yields if so. The floor updates automatically as the auto-tuner adjusts batch size. When the queue is draining (`qs.done`), the floor is bypassed so CPUs can process remaining tasks. The floor is logged in `-vvv` tick output for diagnosis.

---

## v0.12.8 — FrameThrottle ordering deadlock + memory-based throttle + thundering herd

**Three fixes targeting hybrid mode reliability and performance:**

### 1. FrameThrottle ordering deadlock (acquire-before-pop)

**Bug:** Hybrid compress stalled with `cpu_rate=0 gpu_rate=0` — everything frozen. The v0.12.7 fix (acquire after pop) prevented hoarding but introduced a new circular deadlock: a worker pops the frame the writer needs, then blocks on `acquire()` because all permits are consumed by ResultStore. Writer waits for the frame, nobody releases permits.

**Fix:** CPU workers now use a three-step pattern: (1) wait for predicate with no permit held (no hoarding), (2) acquire a permit (may block, but no task is held, so writer can progress), (3) non-blocking `try_pop_one_cpu` (if pop fails because state changed, release permit and retry). Workers only hold both a task AND a permit simultaneously — never one without the other. Added `try_pop_one_cpu()` to `TaskQueue` for the non-blocking pop step. Rescue workers use simple acquire-before-pop (hoarding is acceptable — rescue queue only fills on GPU failure).

### 2. Memory-based throttle sizing

**Bug:** The throttle was sized from a worker-count formula (`max(512, 2 × (cpu + gpu×batch + rescue))`). On a 256-core + 8×GPU system the ceiling of 512 was routinely exhausted by ResultStore write lag, triggering the ordering deadlock above.

**Fix:** Throttle budget is now `(available_ram / 2) / chunk_size` with a floor of 1024. A 256 GiB system gets ~8192 permits; a 16 GiB system gets ~512; a 2 TiB cluster gets ~65536. Scales naturally from embedded to thousand-GPU without any per-configuration tuning. Uses `get_available_ram_bytes()` (/proc/meminfo on Linux, 8 GiB fallback elsewhere). Applied to all four throttle sites (GPU compress, GPU decompress, CPU compress, CPU decompress).

### 3. Thundering herd reduction

**Bug:** `gpu_got_data()` called `notify_cpu_waiters()` (notify_all) every time the last waiting GPU got data. With 8 GPUs × 24 streams, this woke all 96 CPU threads — each grabbed the queue mutex, found the predicate false, and went back to sleep. The resulting `futex` churn increased `sys` time by ~24%, making hybrid 10% slower than gpu-only.

**Fix:** `gpu_got_data()` now calls `notify_cpu_one()` (new method). GPU workers call `notify_cpu_waiters()` (all) on exit so CPU stragglers always get woken for the drain path — this was the specific edge case that killed the previous `notify_one` attempt in v0.11.41.

---

## v0.12.7 — FrameThrottle deadlock + VRAM-starved stream auto-decrement

**Two separate deadlocks fixed:**

### 1. Stream count auto-decrement on VRAM starvation

**Bug:** `./build/gzstd -f -k --gpu-only --gpu-batch=1 --gpu-streams=64 …` hung with:
```
[GPU1] insufficient VRAM for even batch=1  skipping device
[GPU0] insufficient VRAM for even batch=1  skipping device
in:100.0% 19.53 GiB ... out:0.0% 0.00 B
```

**Root cause:** When per_stream_batch reached 1 and the allocator still failed, the GPU worker skipped the *entire device* — even if earlier streams in the loop had already initialized successfully. With `--gpu-only` and every GPU skipped, the producer kept reading into a queue no one would ever consume.

**Fix (compress + decompress):**
- If stream `s` can't fit at batch=1 but streams `[0..s)` initialized fine, cleanly destroy the failed stream, `ctxs.resize(s)`, and continue running with `s` streams instead of the requested count. Emits `WARNING: [GPU#] VRAM insufficient for N streams at batch=1; auto-reducing to M stream(s)` at `V_DEFAULT` (suppressed under `-q`).
- Only skip the GPU entirely when `s == 0` (zero usable streams).
- Added `std::atomic<int> gpu_init_failures` shared with the producer. In `--gpu-only`, the compress producer now bails out of the read loop once `gpu_init_failures == gpu_count`, instead of buffering the entire input into RAM before the post-join `die(EXIT_GPU_FAIL)` fires.

### 2. FrameThrottle permit starvation on multi-GPU with large batches

**Bug:** On 8× H100 with `--gpu-batch=64 --gpu-streams=4`, hybrid compress hung indefinitely. `hybrid: tick` showed `cpu_rate=0 gpu_rate=0 gpus_waiting=0 cpu_taken=0 gpu_taken=0` — everything frozen before any task moved.

**Root cause:** `FrameThrottle` had a hard default of 512 permits, and idle worker threads were hoarding permits:
- `cpu_worker`, `cpu_worker_rescue`, and `cpu_decomp_worker` called `bp->acquire(1)` at the top of their loop — *before* popping a task. An idle rescue thread blocked on an empty rescue queue held 1 permit forever.
- Rescue pool = `hw_concurrency/2` (128 on a 256-core box) → 128 permits locked idle.
- Plus CPU pool (96) + GPU workers needing `stream_count × per_stream_cap × gpu_count` = 4 × 16 × 8 = 512 → total demand 736 against 512 supply. Classic permit-starvation deadlock, made worse by `acquire()` hoarding partial grabs while waiting for the rest.

**Fix:**
1. Moved `bp->acquire(1)` to *after* a successful pop in all three CPU-path workers. Idle workers no longer hold permits. Matching `release(1)` calls on exit paths removed (no permit to release).
2. `compress_nvcomp` and `decompress_nvcomp` now size the throttle to `max(512, 2 × (cpu_threads + gpu_batch_cap × gpu_count + rescue_threads))`. Small runs unchanged; large fleets get proportional headroom with a 2× pipeline factor.

---

## v0.12.6 — Multi-GPU parallelism fix + verbose output improvements

- **Multi-GPU compress starvation fix**: `pop_batch_greedy` was called with `min_n = max_n`, causing all GPU workers to block until a full batch was available. Only one GPU could ever win the batch race; the other remained idle. Fixed by using `min_n=1` (same fix already applied to decompress), allowing GPUs to interleave and take partial batches
- **Decompress GPU verbose output**: `take` and `done` log lines now match compress format — `seq=[lo..hi]` added to take, and done now shows `in=`, `h2d=`, `comp=`, `d2h=`, `tot=`, `thr=` breakdown with CPU-side timing
- **CPU worker take log**: added `[CPU/T#] take seq=N in=X` log at `-vv` so early frames grabbed during GPU init are visible (previously only completion was logged)
- **Number colorizer**: changed `!isalpha` to `!isalnum` predecessor check — digits embedded in identifiers like `h2d` and `d2h` are no longer colorized

---

## v0.12.5 — Progress bar UI polish + decompression % fix

- **Only `XX.X%` values are bold/bright**; labels, sizes, and rates use dim cyan/green
- **Dark grey `|` separator** (`\033[90m`) between in and out sections
- **Completion summary colorized**: `OK` bold green, input size cyan, output size/rate green, ratio bold
- **Decompression `out%` fix**: was stuck while size display changed because frame-completion counter (`tasks_done/total_frames`) updated before AIO finished writing. Now uses `wrote_bytes/total_out` (byte-level, matches the size display) throughout decompression
- **Compression early `out%` fix**: spurious 99.9% at startup caused by `wrote_bytes/total_out` where `total_out` was only a partial running sum. Now shows `---` until frame tracking (`total_frames`) is established
- **Benchmark script**: falls back to `in:XX.X%` when `out%` is not yet available; status suffix shortened (`gzstd ` dropped, `ETA ` → `~`) to stay under 80 columns

---

## v0.12.4 — Colorized progress bar

Progress bar now uses ANSI colors for readability (already required ANSI for cursor control):

- **`in:` label** — cyan; **`in%` value** — bold bright cyan
- **`out:` label** — green; **`out%` value** — bold bright green (bold yellow when unknown)
- **rates and separator** — dim

Test mode (`--test`) colorizes `in%` and `verified:` bytes consistently.

---

## v0.12.3 — Dual-percentage progress bar (in% and out% shown independently)

The v0.12.2 frame-based progress caused a visible jump: read-based % reached 100% quickly, then switched to frame-based % near 0% and climbed again. Confusing.

**Fix:** Show two independent percentages side by side — no single number ever jumps backwards.

```
in:34.2% 2.10 GiB  out:12.7% 780 MiB | in:1.20 GiB/s out:450 MiB/s
```

- **in%** — `read_bytes / total_input`: how much input has been consumed; climbs fast on fast NVMe
- **out%** — `tasks_done / total_frames` while compressing/decompressing, then switches to `wrote_bytes / total_out` during the AIO flush phase; reflects actual CPU/GPU work
- Shows `---` when a metric is unknown (pipe input, single-thread stream path)
- Test mode updated to match new format

---

## v0.12.2 — Progress bar tracks frame completion instead of reader

Previously the progress percentage was based on `read_bytes / total_input_size`. The reader finishes quickly (it's just I/O), so at ultra compression levels with large chunks the bar jumps to 100% while workers are still compressing — useless as a progress indicator.

**Fix:** Added `total_frames` to `Meter` (set by the producer after enqueuing all work) and moved `tasks_done` tracking to `writer_thread` (incremented per frame batch handed off for writing). Progress is now `tasks_done / total_frames`, which reflects actual compression work.

- Falls back to read-based % before `total_frames` is known (single-thread stream path, or brief start-of-run window)
- Write drain phase (all frames done, AIO still flushing): shows `[X.X%] writing: A / B @ C/s`
- Also fixed: removed premature `[done]` flash from inside the progress loop (race where `wrote_bytes` transiently caught up to the incrementally-accumulated `total_out`)

---

## v0.12.1 — Ultra compression window fix

### Ultra levels (--ultra -20/-21/-22) now set ZSTD_c_windowLog — POSITIVE (correctness fix)

Previously, `--ultra` enabled level 20–22 but never set `ZSTD_c_windowLog`, causing zstd to silently clamp the window to its default (~8 MiB). The result: ultra levels incurred the CPU cost of the extended search strategy with none of the compression benefit.

**Root cause:** `compress_one_cpu_frame()` and `compress_cpu_stream()` only called `ZSTD_CCtx_setParameter(ZSTD_c_compressionLevel)`. Without an explicit `ZSTD_c_windowLog`, the library ignores the intended 32–128 MiB window.

**Fix:**
- Added `ultra_window_log()`, `ultra_min_chunk_mib()`, and `apply_ultra_cctx()` helpers
- `compress_one_cpu_frame()` now takes a `bool ultra` parameter and calls `apply_ultra_cctx()` after setting the level
- `compress_cpu_stream()` sets `ZSTD_c_windowLog` on its direct `CCtx` and logs it at `-v`
- All three compress paths (stream, MT, nvCOMP) auto-increase chunk size to match window size (32/64/128 MiB for levels 20/21/22) when `--chunk-size` was not explicitly set; warns if user-specified chunk is too small
- `check_ram_budget()` now accounts for ~8× window size per thread for CCtx hash/chain tables (~256 MiB/thread at -20, ~512 MiB at -21, ~1 GiB at -22); auto-reduces thread count rather than OOMing

**Window sizes:**
- `-20 --ultra`: windowLog=25 (32 MiB window, min chunk 32 MiB)
- `-21 --ultra`: windowLog=26 (64 MiB window, min chunk 64 MiB)
- `-22 --ultra`: windowLog=27 (128 MiB window, min chunk 128 MiB)

**Result:** `gzstd --ultra -22` now produces compression ratios comparable to `zstd --ultra -22 -T0` and runs at similar speed (both doing the intended work). Output is fully interoperable with `zstd -d`.

---

## v0.12.0 — FrameThrottle (counting semaphore replaces byte-based backpressure)

### FrameThrottle Refactor — POSITIVE (simplification)
Replaced `WriterBackpressure` (byte-based high/low water marks + `writer_stalled_` escape hatch) with `FrameThrottle`, a counting semaphore that bounds the number of in-flight frames (popped from queue but not yet written to disk).

**How it works:**
- Workers call `acquire(N)` before popping (1 for CPU, `pop_n` for GPU batches)
- Writer calls `release(1)` per frame after physical disk write (via AsyncWritePool)
- GPU batches release excess permits if fewer frames are returned than requested
- Default: 512 permits (max in-flight frames)

**What was removed (-57 net lines):**
- `mark_produced()` — 6 call sites across CPU/GPU compress/decompress workers
- `mark_written()` — byte-level tracking in AsyncWritePool
- `writer_stalled_` flag + `set_writer_stalled()` — the deadlock escape hatch
- High/low water mark hysteresis (4 GiB / 2 GiB byte thresholds)
- `produced_` / `written_` atomic counters

**Why it's deadlock-free by construction:** The task queue is FIFO. If all 512 permits are consumed, the frame the writer needs (the oldest) was the first one popped and is guaranteed to be in-flight. The writer never waits for a frame that hasn't been popped yet while all permits are consumed. No `writer_stalled_` escape hatch needed.

**Why the old design was fragile:** `WriterBackpressure` counted total `produced - written` bytes, which included out-of-order frames sitting in ResultStore that the writer couldn't drain yet. This inflated the apparent backlog, triggering backpressure even when the writer had capacity. The resulting deadlock (all workers blocked on backpressure while the writer waited for frame N still in the queue) required progressively more complex fixes: first a 100ms timeout (v0.11.43), then the `writer_stalled_` signal (v0.11.44). The counting semaphore eliminates the root cause.

### v0.11.38–v0.11.44 (intermediate fixes, subsumed by v0.12.0)
- **v0.11.38:** Fixed backpressure disabled prematurely — `set_done()` moved after `pool.join()` in all 4 teardown paths
- **v0.11.39:** Added `fallocate()` preallocation for all write paths (compress + decompress, all modes). Avoids per-write extent allocation on NVMe.
- **v0.11.40:** Fixed hybrid decompress deadlock at 55.5% — `gpu_wants_data()` called before backpressure check blocked GPUs and CPUs simultaneously. Fixed by swapping order.
- **v0.11.41:** Fixed thundering herd — `gpu_got_data()` only notifies CPUs when last GPU is satisfied (`gpus_waiting_` drops from 1→0), not on every GPU completion.
- **v0.11.42:** Fixed CPU hang at end — reverted `notify_one` back to `notify_all` for `set_done()` path. Removed `-D` suffix from CPU decompress labels.
- **v0.11.43:** Timeout-based fix for out-of-order ResultStore deadlock (`cv_.wait_for(100ms)`). Replaced by proper solution in v0.11.44.
- **v0.11.44:** Replaced timeout with `writer_stalled_` signal approach. Subsumed by FrameThrottle in v0.12.0.

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

### Test Mode & Progress Fixes (v0.11.25)  BUGFIX
- **`wrote_bytes` double-counting:** Worker-side updates removed; writer thread is now the sole source of truth for output bytes.
- **Test mode backpressure stall:** Backpressure pointer set to `nullptr` in test mode (`-t`) since there's no AIO, so `mark_written()` is never called. Without this, CPU workers would block forever waiting for writes that never happen.
- **Progress bar in test mode:** Shows `verified:` label instead of `out:`.

### Graceful GPU VRAM Handling (v0.11.26)  POSITIVE (robustness)
GPU workers now survive VRAM exhaustion instead of crashing the process. Critical for shared GPU environments where other users consume VRAM mid-run.
- **Graceful skip:** GPU workers return early when batch=1 allocation fails. `TaskQueue::re_enqueue()` returns in-flight frames to the queue (without incrementing `total_tasks_`) so other GPUs or CPU workers process them.
- **VRAM reserve:** Each GPU holds a half-batch-sized reserve. On allocation failure, the reserve is freed for retry before giving up.
- **VRAM retry limit:** 10 attempts max in both compress and decompress allocation loops, preventing infinite retry when VRAM fluctuates near the threshold.
- **Reader no longer aborts early:** Removed `abort_on_failure && any_gpu_failed` check from compress reader loop. A single GPU VRAM failure no longer truncates the output  surviving GPUs handle the work.
- **Post-join GPU failure:** Only calls `die()` if ALL GPUs failed (count-based), not on any single failure.

### Structured Exit Codes & Argument Hardening (v0.11.26)  IMPROVEMENT
- **Exit codes:** 0=OK, 1=runtime, 2=usage, 3=I/O, 4=data, 5=GPU_FAIL. All `die()` calls categorized via `die_io()`, `die_data()`, `die_usage()`. Help text documents exit codes.
- **Unknown option rejection:** Flags starting with `-` that aren't recognized exit with code 2 (EXIT_USAGE) instead of being treated as filenames.
- **`--` end-of-options:** Everything after `--` is treated as a filename, matching POSIX convention.
- **`--threads=N` form:** Now recognized alongside `-T N`, `-T2`, `--threads N`.
- **Argument order independence:** `-22 --ultra` works (deferred ultra check to post-parse validation).
- **Truncated stream detection:** Decompress checks `ret > 0` after loop and dies on >8 trailing bytes at EOF.
- **`.zst` double-compression warning:** Warns when compressing a file that already has `.zst` extension.

### Writer Deadlock Detection & Cleanup Safety (v0.11.27)  IMPROVEMENT
- **Writer deadlock detection:** 5-second timed wait when `workers_done` is set. If the next expected frame never arrives, calls `die()` with diagnostic instead of hanging forever or silently producing truncated output.
- **`die()` reports cleanup:** Shows `gzstd: removing incomplete output: path` on fatal error.
- **Atomic temp file cleanup:** When using `-f` to overwrite, the `.tmp` file is registered for cleanup on failure/signal. Original file is preserved untouched.
- **Consistent log format:** `[GPU-D...]` → `[GPU...]` everywhere. `[GPU2] ready, semaphore scheduling active` format.

### Compress Backpressure (v0.11.29)  POSITIVE (major)
Same backpressure mechanism from decompress (v0.11.24) now applied to all compress paths. Prevents CPU workers from producing compressed data faster than the NVMe can write.
- **`compress_cpu_mt`:** `WriterBackpressure` created and passed to writer + all CPU workers.
- **`compress_nvcomp`:** Backpressure passed to writer, GPU workers, CPU hybrid workers, and rescue workers.
- **CPU workers** call `wait_if_backlogged()` before popping, `mark_produced(csz)` after delivering compressed frame.
- **GPU workers** call `mark_produced(out_sum)` on both async and sync D2H paths. Never throttled.
- **`--cpu-batch` in `--cpu-only` mode:** Now ignored with a note, since the stop-and-go pattern caused massive sys overhead (10m26s sys on 432 GiB file).

### Default Chunk Size = 16 MiB (v0.11.30)  POSITIVE
All paths (CPU-only, hybrid, GPU-only) now default to 16 MiB chunk size. Removed auto-chunk scaling that previously used 32512 MiB based on file size and device count.
- **Why 16 MiB wins:** More tasks for better load balancing (27,685 vs 3,461 tasks for 432 GiB), lower per-thread memory (3 GiB vs 24 GiB for 96 threads), and matches `GPU_SUBCHUNK_MAX` so no splitting is needed in hybrid mode.
- **Removed:** `auto_chunk_mib_cpu`, `auto_chunk_mib_gpu`, `is_regular_file_stream`, `AUTO_HOST_CHUNK_*` constants.
- **`--chunk-size=N`** still available for manual override.

### RAM Budget Check (v0.11.29)  IMPROVEMENT
Pre-flight check reads `/proc/meminfo` `MemAvailable` and estimates memory needed for N threads × chunk_mib. If estimated usage exceeds 75% of available RAM, auto-reduces chunk size (halving until it fits) with a warning instead of OOMing.

### Progress Bar Improvements (v0.11.30)  IMPROVEMENT
- **Dual rate display:** Progress bar now shows both `in:` and `out:` rates: `[45.2%] in:195.6 GiB out:97.3 GiB | in:1.84 GiB/s out:918 MiB/s`
- **Write drain states:** Three states  AIO writing (capped 99.9%), flushing to disk (elapsed timer), finalizing message.
- **Finalize message:** Shows `[done] finalizing 217.3 GiB ...` for large files during file close/rename.

### Comprehensive Test Suite (v0.11.26v0.11.30)  NEW
`gzstd-test.sh`: ~170+ tests across 35 sections with live progress bar, per-test timing, CTRL-C handling, and auto GPU detection. Key coverage: round-trip, compression levels with ratios, integrity, pipes, tar, file management, threading forms, chunk sizes, verbosity validation, stats JSON, exit codes, zstd interop, VRAM pressure, wildcards, `--` end-of-options, output redirection, space-separated options, GPU options, error handling, cross-level decompress, argument order, completion summary format.

### Stdout O_DIRECT Detection (v0.11.31)  POSITIVE (major)
When stdout is redirected to a regular file (`gzstd -d < file.zst > output.tar`), gzstd now detects this via `fstat(fileno(stdout))` + `/proc/self/fd/N` and reopens with O_DIRECT, bypassing the page cache.
- **Safety checks:** Skips O_APPEND (undefined with O_DIRECT), `/dev/*`, deleted files, non-regular files. Falls back silently to buffered fwrite on any failure.
- **Result:** `tar | gzstd > file.zst` gets full NVMe speed without the user needing to know about `-o`.

**Knuth (432 GiB decompress via stdin redirect):**

| Method | Wall | sys | GiB/s |
|--------|------|-----|-------|
| gzstd stdout → page cache | 8m59s | 27m27s | 0.83 |
| zstd stdout → page cache | 10m46s | 11m01s | 0.67 |
| **gzstd stdout → O_DIRECT** | **3m33s** | **19m53s** | **2.05** |

### GPU Backpressure on Pop (v0.11.31)  BUGFIX (major)
GPUs now call `wait_if_backlogged()` before `pop_batch_greedy` in both compress and decompress workers. Previously GPUs were exempt from backpressure ("never throttled"), but 8 H100s decompressing at full speed overwhelmed the NVMe writer  decompression finished with only 28% of data written to disk.
- GPU workers block before grabbing new work, not mid-kernel
- One batch per GPU remains in-flight beyond the high-water mark at most
- Writer drain after decompress: was 72% remaining, now <1 second

### Test Mode Defaults to 2 GPU Streams (v0.11.31)  POSITIVE
`-t` verify mode now defaults to `--gpu-streams=2` instead of 1. No write bottleneck in verify mode, so stream overlap helps.
- **Knuth (432 GiB verify):** 1 stream: 4.09 GiB/s (1m47s) → 2 streams: 6.39 GiB/s (1m09s)  **56% faster**
- Compress/decompress stays at 1 stream (NVMe is the bottleneck, larger batches win)
- Fixed help text: was incorrectly showing default as 3

## Key Lessons Learned (Updated)

19. **Stdout O_DIRECT is a free 2.5× win.** Users don't think about O_DIRECT  they just write `> file`. Detecting stdout-to-file and auto-enabling O_DIRECT gives them NVMe speed without any knowledge of I/O internals. The page cache adds 17 GiB of dirty pages and halves throughput on 432 GiB files.

20. **GPUs need backpressure too.** The original design exempted GPUs ("batches in-flight, can't throttle"). But on H100 × 8, GPU decompression throughput vastly exceeds NVMe write speed. The result: 300+ GiB buffered in RAM, massive kernel writeback, and a frozen "28% writing" progress bar after decompression finished. Throttling GPUs before their next `pop_batch_greedy`  not mid-kernel  keeps the pipeline balanced with <1s drain.

21. **Bound frames, not bytes.** Byte-based backpressure conflated two concerns: memory pressure (bytes in RAM) and frame ordering (which frames the writer can drain). Out-of-order frames in ResultStore inflated the byte count, triggering false backpressure. A counting semaphore on frames separates these concerns: frame ordering lives in ResultStore, flow control lives in the semaphore. The FIFO queue guarantees the writer's next-needed frame is always in-flight, making the design deadlock-free without escape hatches.

18. **GPU VRAM is a shared resource  design for it.** On a multi-user machine (Knuth, 8× H100), any GPU can lose VRAM at any moment. Infinite retry loops, early reader aborts on single GPU failure, and missing frame deadlocks all surfaced under real student workloads. The fix: retry limits, graceful skip with re-enqueue, deadlock detection with hard error, and never abort the reader on partial failure.

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
