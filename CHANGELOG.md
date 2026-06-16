# gzstd Optimization Changelog

**Covers:** v0.9.50 → v0.13.77  
**Test machines:**
- **Server:** 256-core CPU, 8× NVIDIA H100 (95 GiB VRAM each), NVMe ~3 GiB/s write
- **Workstation:** 256 GiB RAM, 24-core CPU, 2× NVIDIA RTX 2080 Ti (10 GiB VRAM each), NVMe ~1.8 GiB/s write

---

## v0.13.77 — decompress integrity guard + graceful parallel-reader fallback

Three decompress-path fixes from a correctness review.

1. **GPU decompress now verifies the produced size against the frame header.**
   `gpu_decomp_worker` checked nvCOMP's per-chunk status but then trusted the
   reported output size (`h_actual[i]`) verbatim.  The CPU path gets this check
   for free — `ZSTD_decompressDCtx` rejects a frame whose output differs from
   its declared content size — so the GPU path could silently write a short or
   wrong-length frame and exit 0 on corrupt/malformed input where the CPU path
   and stock `zstd -d` both error.  It now compares `actual` to the header's
   `decomp_size` and throws on mismatch; the existing catch re-enqueues the
   undelivered tail to the main queue, where a CPU worker re-decodes it and
   either succeeds (a transient device glitch — the run completes correctly) or
   dies cleanly with a zstd data error.  No effect on valid archives, where the
   two sizes always agree; verified the guard does not false-positive on a real
   GPU round-trip.

2. **The parallel-prefetch decompress reader falls back instead of dying.**
   A mid-stream frame with no content-size header (e.g. a gzstd archive
   concatenated with a zstd-streamed segment) made the MT reader `die_data`,
   while the single-threaded reader handled the same input by streaming the
   remainder.  The MT reader now records the offending frame's file offset,
   reads `[offset, EOF)` into the caller's buffer, and signals the same
   fallback the single reader uses — the parsed frames are written, then the
   tail goes through the CPU streaming decoder.  Both readers now emit one
   shared `warning:` line (previously the only message here was a `note:`).
   Confirmed byte-identical output on cpu-only, `--hybrid`, and `--gpu-only`.

3. **Bounded the MT reader's frame-spanning re-parse cost.**
   Completing a frame that straddles a 64 MiB block re-runs
   `ZSTD_findFrameCompressedSize()` on the growing carry (zstd has no resumable
   size parser), and growing the carry by a fixed 4 MiB step made that
   re-walk quadratic for frames larger than a block (`--ultra` / huge
   `--chunk-size`: ~16 re-parses per block, each rescanning the whole carry).
   The step now grows geometrically (capped at one block), so a
   straddle-by-a-little still resolves on the first small step while a
   multi-block frame needs only ~log2(block/step) parses per block.  Overshoot
   is bounded at 2× and re-parsed in place, so output is unchanged.

Adds a regression test (`MT reader streaming-fallback`) covering the §2 path:
a > 128 MiB gzstd-archive + piped-zstd tail must round-trip, warn, and match
the single reader.

## v0.13.76 — progress bar for a redirected stdin: auto-show + real percentages

Two fixes for `gzstd -dc < big.zst`, both flowing from "we now know the
input size via fstat":

1. **Auto-show the bar.** The default-verbosity suppression treated any
   non-TTY stdin as a pipe and hid the bar, so a `< file` redirect needed
   `--progress`.  It now suppresses only a TRUE pipe — keyed on whether the
   input size is known (`total_in == 0`).  A redirect from a real file /
   block device has a known size, so the bar shows automatically; genuine
   pipes (`tar -I gzstd`, `cat | gzstd`, a terminal) stay quiet.
2. **Real percentages instead of `---`.**  The progress denominator came
   from `fs::file_size` of a named path, so stdin showed `in:--- out:---`.
   New `known_input_size()` returns the size for a named regular file OR a
   seekable stdin redirect (`fstat` S_ISREG/S_ISBLK), so `< file` now shows
   `in:NN% out:NN%`.  All five total_in sites (4 compress + decompress/test)
   route through it; named-file and true-pipe behavior is unchanged.

A side benefit: the single-threaded streaming compress paths now learn the
pledged source size on a `gzstd < big.bin` redirect too (ZSTD_CCtx_set
PledgedSrcSize), so the frame header carries the content size.

## v0.13.75 — parallel reads from a redirected stdin and block devices (fstat the fd)

The parallel readers (compress pooled + decompress prefetch) gated on a
NAMED regular-file argument, so `gzstd -d < big.zst` — stdin redirected
from a real file — fell to the single-threaded reader even though fd 0 is
fully seekable.  New `probe_preadable_input()` `fstat`s the fd instead of
trusting the path: S_ISREG or S_ISBLK + seekable → preadable, whatever the
fd's origin (named file, `< file` redirect, `< /dev/sdX`).  Pipes/FIFOs/
sockets/ttys (the `tar -I gzstd` stream, process substitution, a terminal)
are correctly rejected and stay on the sequential reader.

Both readers now take an optional borrowed fd: for a redirect they pread
the inherited stdin fd directly (positional, so it coexists with the
buffered peek and never disturbs the FILE* position) and don't close it.
Block devices are sized via SEEK_END (st_size is 0 for them).  --direct-read
is excluded (it owns its single O_DIRECT stream).

So `gzstd -d < big.zst` and `gzstd < big.bin` now get the same multi-reader
speedup as the named-file form.  What still can't: `tar -I gzstd -cf
out.tar.zst /dir` — tar generates the stream into a pipe, so the bytes
never exist as a seekable file; fstat sees the FIFO and we stay sequential
(correctly).  Validated byte-identical across named / `< redirect` / pipe /
process-substitution for cpu/gpu/hybrid, both directions.

## v0.13.74 — help: separate --direct-read and --read-threads (docs only)

The two flags were crammed together as if related.  In the short help a
`--direct-read` note ("one-pass speedup…") was stranded under
`--read-threads`; in the long help they shared one header and the
paragraph described only `--direct-read`, leaving `--read-threads`
undocumented.  They are unrelated beyond being mutually exclusive
(`--direct-read` is always a single O_DIRECT stream; `--read-threads` is
parallelism for the buffered path).  Now each has its own entry:
`--direct-read` keeps its standalone write-up (and notes it's single-stream
/ benchmarking-oriented), and `--read-threads` gets a real description
(parallel readers for the buffered input path, compress pooled reader +
decompress prefetch reader, auto `clamp(threads/8,3,12)`, 1 = single).

## v0.13.73 — progress bar: clamp out% monotonic (stop it going backwards)

The output percentage could move backwards.  Root cause: there is no
file-level "total decompressed size" or "frame count" in a zstd stream —
each FRAME header carries only its own content size, so the total is known
only after the reader has parsed every frame.  `out% = wrote_bytes /
total_out` therefore divides by a denominator that GROWS during reading;
when a burst of highly-compressible frames is discovered, total_out jumps
and the percentage dips.  (The parallel reader made it more visible by
discovering frames in fast bursts.)  Counting frames at the writer doesn't
help — `total_frames` grows during reading too.

Fix: clamp the displayed out% to be monotonically non-decreasing (track a
floor across progress samples).  Worst case is a brief forward stall if the
running estimate overshot, never a backward step; with the fast parallel
reader the estimate phase is short anyway.  Considered but rejected: an
exact compressed-domain out% (comp-bytes-written / file_size, monotonic by
construction) — it would essentially duplicate the in% bar and costs
per-frame memory proportional to frame count.  Cosmetic; affects only the
live -v/--progress bar, not the final summary.

## v0.13.72 — fix: parallel decompress reader double-counted input bytes

v0.13.71's prefetch threads added each block to `m->read_bytes`, but the
decompress workers ALSO count input per frame (the single-threaded reader
relies on them and adds nothing itself).  So `-d` reported 2× the input
("423.78 GiB =>" for a 211.89 GiB file) and the progress bar hit 100% at
the halfway point.  (`-t` was unaffected — its summary derives the input
size differently.)  Fix: the prefetch threads keep only the reader-state
timing (and the -vvv g_perf totals); the workers remain the sole counter
of `read_bytes`.  Display-only — decompressed output was always correct.

Server speedup confirmed on the 432 GiB archive: gpu-only decompress
7.37 → **14.13 GiB/s** (≈2×), cpu-only 15.12 → **18.48** (+22%), -t 18.80.

## v0.13.71 — parallel-prefetch decompress reader (multi-reader)

The v0.13.70 accounting confirmed the decompress reader is the cap on the
cpu-only path: on the server (432 GiB / 27685 frames) the single reader ran
**92% saturated** (io 52.5%, parse 2.5%, task-copy 37.0%, blocked 0.3%) and
the writer was 50% starved waiting for it, holding cpu-only decompress to
15.1 GiB/s.  (gpu-only is 7.4 — decompress, like compress, is a cpu-only
regime on that box, so the cpu-only reader is the lever.)

Compression's N-independent-reader trick can't be copied: decompress frame
boundaries are found by sequential parsing and the zstd magic can appear in
payload, so there is no safe blind resync without a frame index.  Instead,
`stream_frames_to_queue_mt`: K prefetch threads `pread` fixed 64 MiB BLOCKs
of the file (claimed in order via a shared counter) into a bounded ring, and
one consumer parses frames in-place and pushes them — a carry buffer bridges
a frame (or skippable frame) that straddles a block boundary.  This
parallelizes the dominant read I/O; the per-frame copy stays on the consumer
(a later zero-copy pass can remove it).  Reader count scales with the box
(`clamp(threads/8, 3, 12)`, `--read-threads N` overrides), matching compress.

Conservatively gated: engaged only for a seekable regular file > 128 MiB,
not `--direct-read` (O_DIRECT stays single-stream), whose first frame is a
normal known-size zstd frame.  Every other case (stdin, O_DIRECT,
unknown-size single-frame, foreign archives) stays on the single-threaded
`stream_frames_to_queue`, which still owns all the fallback paths;
mid-stream anomalies are a hard data error (as the single reader treats
truncation), never a mid-stream handoff.  Validated byte-identical against
the single reader across chunk sizes 1/4/16/64/128 MiB (frames smaller than,
equal to, and larger than a block — exercising heavy boundary-spanning and
the multi-block carry path), all consumers (cpu/gpu/hybrid), `--read-threads`
1–16, and highly-compressible input.  Server speedup to be measured.

---

## v0.13.70 — decompress reader-state accounting (-v); groundwork for parallelizing the decompress reader

Toward a "multi-reader" decompress path: first, make the decompress reader
measurable, because its architecture differs fundamentally from compress.
The compress reader slices the input at fixed byte offsets, so N threads
grab independent chunk indices trivially.  The decompress reader
(`stream_frames_to_queue`) must PARSE variable-length frame boundaries
sequentially — frame N+1's start is unknown until frame N's header is
walked (`ZSTD_findFrameCompressedSize`) — so N independent parsers aren't
safe without a frame index (the zstd magic can appear inside payload).

`stream_frames_to_queue` now feeds the same Meter reader-state counters the
compress path uses, with a new `parse` bucket: `[READER] io | parse |
task-copy | blocked-downstream` at -v.  io = read syscall; parse =
frame-boundary walk; task-copy = the per-frame `assign` out of the parse
buffer; blocked-downstream = `queue.push()` stalling on the bounded queue
(which means the GPU/CPU consumers, not the reader, are the faucet).  The
verdict distinguishes a saturated reader (and which sub-component to attack
— copy → zero-copy frame reader, parse → needs an index, io → faster
source) from a reader that's merely blocked because the consumers can't
keep up.  The `[WRITER]` three-state report (v0.13.59) already covered
decompress — it just hadn't been exercised on a decompress run.

Diagnostic-only (no behavior change).  Early local signal (tiny runs, not
representative): parse is ~0.1% — the serial spine is NOT the bottleneck,
so parallelizing it isn't the lever; copy slightly exceeds io.  The actual
optimization (likely a zero-copy frame reader if knuth shows the reader is
copy-bound, or pipelined raw-read I/O if io-bound) is to be chosen from a
real-workload -v run on the server.

---

## v0.13.69 — deadlock: gpu-only DECOMPRESS with locked --gpu-batch × many streams

The v0.13.68 fix covered compress; probing its decompress analog found a
real but DIFFERENT deadlock.  Confirmed on the server: `-d --gpu-only
--gpu-batch=64 --gpu-streams=16` on a many-frame archive hangs at ~74%
(must ^C); the same flags with `--hybrid` complete cleanly.

Mechanism — classic permit / head-of-line, not the v0.13.68
pool-exhaustion: each of 128 streams (8 dev × 16) does `bp->acquire(pop_n)`
upfront, then blocks waiting for a FULL locked batch.  The throttle budget
is sized FROM `device × streams × batch`, so GPU demand can consume the
entire permit pool; the in-order writer then wedges behind a head-of-line
frame that no stream can pop.  Hybrid survives because its CPU workers are
a fine-grained relief valve (they pop one low-seq frame at a time);
gpu-only has none (rescue threads fire only on GPU failure).

Fix: in gpu-only decompress, a locked `--gpu-batch` becomes a CAP, not a
hard floor — reuse the unlocked soft minimum (`min(pop_n, 4)`), so streams
take whatever is queued and release the excess permits (the worker already
does this) instead of sequestering a full batch.  Hybrid keeps the honored
full-batch wait.  One-line predicate change (`locked_batch && !opt.gpu_only`).

Could NOT be reproduced on the 2-GPU workstation (VRAM-fit shrinks
per-stream demand on 10 GiB cards; 2 devices don't open deep enough
head-of-line gaps), so correctness was verified locally (round-trip + full
suite) and the deadlock cure is to be confirmed on the server.  Adds a
gpu-only decompress wedge canary to the bounded-queue test section.

---

## v0.13.68 — deadlock: locked --gpu-batch × many streams wedges under a bounded queue

Reproduced on the server: `--gpu-streams=16 --gpu-batch=64` hung at 45%
(^C needed).  Mechanism: a user-pinned batch makes streams wait for FULL
batches in pop_batch_greedy, and each stream acquires its FrameThrottle
permits BEFORE blocking (that order prevents a different deadlock).  128
streams × VRAM-fit ~20 frames ≈ 2,560 permits sequestered by sleeping
streams; the bounded queue (pool 1,504) can rarely present 16–25
consecutive frames past 96 CPU workers, in-flight output accumulates
until the 8,576-permit throttle exhausts, CPUs can't pop the frames the
writer needs, the writer can't release permits — circular wait.  Same
disease as the v0.13.67 floor lockout: a blocking reservation sized
against a queue that can no longer back it.

Guard: when aggregate locked demand (active streams × pop_n) exceeds
half the queue's depth ceiling, full-batch waits relax to min_n=1 with a
one-time warning — the user's batch remains the pop CAP.  Auto-tuned
(unlocked) runs already use min_n=1 and were never exposed.  Note: the
decompress path's bounded queue + locked batches has the same
theoretical hazard (its bound is TaskQueue max_depth, not the pool);
not yet plumbed — locked decompress batches there remain unguarded.

## v0.13.67 — bounded producer zeroes the AUTO queue floor; the server's compress verdict is in

The v0.13.66 cap/4 clamp (376 frames) was not enough: the GPUs' combined
appetite (16 streams × ~200 batch) exceeds the reader's ~1000 frames/s
supply, so queue depth never builds past ANY substantial floor, and the
AUTO mode is a latch under starvation (CPU <5% share ⇒ factor=4 ⇒ CPU
stays at <5%).  Server floor sweep on the 432 GiB tar:

| floor          | GiB/s | GPU share of frames |
|----------------|-------|---------------------|
| auto           | 15.74 | ~92%                |
| factor 0.5     | 15.69 | ~90%                |
| **off**        | 18.17 | ~37%                |
| (cpu-only ref) | 18.93 | 0%                  |
| (gpu-only ref) | 15.59 | 100%                |

Under a continuously-refilling bounded queue the reservation is obsolete
(GPUs pop min_n=1; the auto-tuner adapts batch size), so a bounded
producer now zeroes the AUTO floor; explicit --hybrid-floor=nominal /
--hybrid-floor-factor are honored, clamped to cap/4.

**Compress verdict for the dual-socket 8-GPU server**: CPU pool alone
18.9 GiB/s, H100 pool alone 15.6, both together 18.2 — the shared memory
fabric, not either engine, is the ceiling.  GPUs add ~nothing to compress
on this box (and with the floor fixed, no longer subtract).  The week's
totals on this workload: 5.73 → ~18–19 GiB/s (75.5 s → ~23 s), all of it
from the input path; engine mix was never the lever.

## v0.13.66 — queue floor clamped to the pooled queue's depth ceiling; gpu-only reader count fixed

The first fed-pipeline server runs falsified the "hybrid loses to
bandwidth" reading and exposed two interacting bugs:

- **Hybrid (11.63 GiB/s vs cpu-only's 18.93) wasn't contention — the CPU
  pool was locked out.**  Ratio arithmetic (gpu-only 50.23%, cpu-only
  48.70%, hybrid 50.19%) shows the GPU took ~97% of frames.  The GPU
  queue floor (streams × batch ≈ 900 frames) predates the pooled reader:
  with mmap the whole file sat in the queue and the floor was harmless,
  but the pool bounds depth at ~1500, the GPUs' batch gulps held depth
  below the floor near-permanently, and `may_take` refused the CPU pool
  forever.  Side effect: the starved CPU pool never accumulated EMA
  samples, so the tail-yield never armed.  Fix: the pooled producer
  declares its depth ceiling (`HybridSched::set_queue_depth_cap`) and
  `update_queue_floor` clamps the floor to a quarter of it.
- **gpu-only was reader-capped, not GPU-capped.**  Its 14.17 GiB/s run
  used 3 readers: the auto count divided `cpu_threads`, which is 0 in
  gpu-only mode.  Now scales from `resolve_cpu_threads` (machine
  parallelism) regardless of mode.  The H100 pool's true ceiling is
  ABOVE 14 GiB/s and still unmeasured — the auto-tune GiB/s figures are
  per-stream, not pool (a misreading that fed the earlier wrong verdict).

Corrected picture on the server: CPU pool ~19 GiB/s, GPU pool ≥14 — a
correctly-scheduled hybrid should beat both.  To be measured.

## v0.13.65 — GPU/hybrid compress gets the multi-reader pooled path (was still on fread+copy)

The first "fed pipeline" hybrid run on the server exposed the scoping gap:
the v0.13.63 multi-reader only served cpu-only compress; hybrid still ran
the single fread+assign reader and capped at 6.11 GiB/s while cpu-only did
~17 ([READER] said it directly: 1 reader, task-copy 33.8%).

Wiring it needed two things the CPU path didn't:

- **One pool buffer per Task, no refcount.**  The host-chunk→gpu_chunk
  subchunk split (one read backing many tasks) was an fread-efficiency
  artifact; the pooled reader simply preads at gpu_chunk granularity, so
  the existing single-owner slot release (`direct_buf`) works unchanged
  and seq == chunk idx stays dense.
- **Slot recycling on every GPU input lifecycle.**  Hybrid keeps batch
  inputs alive for rescue and previously never released them on success
  (owned vectors freed via destructor; pool slots would leak → reader
  starvation → hang).  Releases added at: batch fully delivered (both
  completion paths), the delivered-prefix of a mid-delivery throw (the
  rescue handoff erases those frames), and the rescue worker after
  recompression.  gpu-only keeps its existing release-after-H2D.

Pool sizing for the GPU path: cpu_threads + 32/reader + 1024 (GPU batches
hold slots from pop to delivery), clamped to file size and a quarter of
MemAvailable; the plain-pages prefault is capped at 4 GiB (beyond that,
first-touch faults amortize — MADV_NOHUGEPAGE already prevents the toxic
THP attempts).  Verified on the 2-GPU workstation: hybrid and gpu-only
--no-mmap round-trip clean through the pool (1250 frames through a ~1150-
slot pool — a single leaked slot would hang it).

## v0.13.64 — reader count scales with the worker pool; pool sized for the readers

Server sweep of `--read-threads` on the 432 GiB tar: 3 → 7.46 GiB/s,
6 → 15.61, 8 → 16.28, 12 → **18.74** (23.1 s; the same file took 75.5 s
at v0.13.59).  Two saturation signals emerged in the [READER] line: per-
thread io fell (96.5 → 75.4%) while blocked-on-pool climbed (0 → 15.3%) —
the readers were starving for pool buffers, not hitting the device.
Caveat recorded: by the 12-reader run much of the file sat in the 1.5 TB
page cache, so absolute numbers are cache-flattered; the scaling shape
and the blocked-on-pool growth are the trustworthy signals.

Auto reader count is now `clamp(threads/8, 3, 12)` — 3 on the 24-thread
workstation (measured optimal), 12 on the 96-worker server (best
measured) — and the pool gains 32 buffers per reader (512 MiB each step;
the threads+128 sizing predates multi-reader).  `--read-threads N` still
overrides.

## v0.13.63 — parallel buffered readers: fan the kernel copy out, keep the device stream sequential

The v0.13.62 analysis left one lever: the buffered pooled reader's wall is
the per-thread cold-destination copy_to_user (~3.5 GB/s node-local), not
the device (~10 GB/s buffered).  Probe on the server confirmed page-cache
reads parallelize where O_DIRECT contends: two simultaneous buffered dd
streams = 17.4 GB/s aggregate (vs 9.9 single; O_DIRECT measured 1 stream
4.5 / 4 streams 3.0 — that rule does NOT carry over).

`pooled_read_chunks` now runs N reader threads (default 3, `--read-threads
N` overrides; O_DIRECT and the scratch path stay at 1) pulling chunk
indices from a shared atomic counter against ONE fd.  Interleaved indices
— deliberately NOT partitioned file regions — keep the offset stream
near-sequential (one readahead context keeps working) and bound the
queue's seq skew to ~N.  A partitioned design would flood the ResultStore
with distant-seq frames, exhaust FrameThrottle permits, then the pool, and
starve the region the writer needs — the re_enqueue FIFO-invariant
deadlock.  EOF and abort propagate via a shared done flag; each index is
preaded by exactly one thread, so the emit set is dense and seq-exact.

With the per-thread copy wall broken, the pre-6.4 kernel gate from
v0.13.62 is removed — old-kernel large files use the multi-reader pooled
path instead of fread+assign (which remains for stdin/pipes).  Measured
(workstation, --no-mmap, 20 GiB warm, /dev/null, median of 3): 1 reader
3.11 s → 3 readers **1.67 s** — within ~15% of the mmap zero-copy path.
The `[READER]` report now prints per-thread percentages with the thread
count.  Server expectation: ~8–10 GiB/s vs the 5.7 fread floor (and 2.14
single-pooled); to be validated on the 432 GiB tar.

## v0.13.62 — buffered pooled reader gated to ≥6.4 kernels; cold-destination copy was the real culprit

v0.13.61's THP hypothesis was falsified on-box: identical 2.14 GiB/s with
MADV_NOHUGEPAGE and THP=madvise.  The controlled experiments named the
true mechanism:

- `dd` (same file, same 16 MiB buffered reads): **9.9 GB/s** — one hot
  reused buffer; the kernel copy lands in cache-warm lines.
- gzstd's pool: **2.14** — 224×16 MiB of cycling buffers means every
  pread's destination is cache-cold and recently touched by a remote
  worker core; copy_to_user has no NT stores, so cold destinations pay
  RFO + writeback (~3× the memory traffic).
- `numactl --cpunodebind=0 --membind=0`: **3.46** — removes the
  cross-socket hop (node distance 32 vs 10), recovering the NUMA share.
  `--interleave=all`: no change (spreading coldness isn't warmth).
- Old fread+assign's 5.7 decomposes cleanly: fread's kernel copy at 9.6
  (hot staging buffer, the dd pattern) + assign at ~14 (glibc switches to
  non-temporal stores for L3-sized memcpys).  Two copies, each in a fast
  regime, beat one copy in the slowest regime.

Resolution: the pooled buffered reader keeps its win where measured
(+72%, ≥6.4-kernel workstation, --no-mmap) and is gated off on pre-6.4
kernels (same per-VMA-locks proxy as the mmap gate), restoring
fread+assign's 5.7 floor on the server.  Open lever for that box:
parallel buffered readers — the O_DIRECT single-stream rule (1 stream
4.5, 4 streams 3.0 GB/s) does NOT obviously apply to page-cache reads,
and N readers would break the single-thread cold-copy wall (~3.5 GB/s
local) against the device's ~10.  Needs a dual-dd probe before building.

## v0.13.61 — pooled-reader pool takes plain pages on pre-6.4 kernels (v0.13.60 regressed 2.7× on the server)

v0.13.60's buffered pooled reader, +72% on the workstation, measured 2.14
GiB/s on the server — 2.7× WORSE than the fread path it replaced, with the
reader 99.7% saturated inside pread.  Working hypothesis (consistent with
the box's documented pre-6.4 pathologies, pending on-box validation): the
pool's MADV_HUGEPAGE + sparse one-byte-per-2 MiB prefault was built for
O_DIRECT DMA-segment merging; under buffered reads the kernel copies into
the buffer instead, and on that kernel THP never engages — so the sparse
prefault leaves 1 of 512 pages mapped and every copy page-faults into a
THP-eligible VMA, with compaction attempts on a fragmented box.

The same machinery is a WIN on modern kernels (huge-page-backed
copy_to_user: 3.10 s vs 3.85 s plain, workstation --no-mmap A/B), so it is
gated, not removed: `DirectReadPool::init(want_thp)` keeps the THP
prefault for O_DIRECT (DMA always needs it) and for buffered mode on
kernels ≥ 6.4 (per-VMA-locks check as the vintage proxy); older kernels
get MADV_NOHUGEPAGE explicitly (system THP=always must not re-enable the
pathology) plus a full memset prefault — deterministic, zero faults during
reads.  Workstation verified at parity (3.08–3.13 s); server expectation
is fread-class or better (≥ 5.7, target ~9 GiB/s), to be validated.

## v0.13.60 — buffered zero-copy reader: the fread fallback's hidden copy halved intake

Diagnosed live on the server with the v0.13.59 writer accounting plus the
-vvv breakdown: compressing a 432 GiB tar, the run reported upstream-bound
with 96 workers averaging only ~17.5 busy, and the arithmetic exposed the
reader thread as ~99% saturated — 45 s inside fread (9.6 GiB/s) plus ~30 s
of UNTIMED `t.data.assign` copying every byte a second time.  Effective
intake 1/(1/9.6 + 1/~15) ≈ 5.7 GiB/s — exactly the observed throughput.
The fread fallback is the default on that box because the pre-6.4-kernel
mmap gate declines large files.  `--direct-read` was not the answer there:
its O_DIRECT ceiling (~4.1–4.5 GB/s, matching the old dd measurement)
loses to the page cache's buffered 9.6 GiB/s with readahead.

Fix: `odirect_read_chunks` generalized to `pooled_read_chunks(o_direct)` —
the same single-stream pooled zero-copy reader, with O_DIRECT now just an
open flag.  When mmap is declined (kernel gate, --no-mmap, open failure)
and the input is a regular file, the cpu-only compress reader now preads
buffered (readahead intact, POSIX_FADV_SEQUENTIAL) straight into pooled
buffers and emits view tasks: one kernel→buffer copy instead of two, and
the pool acquire doubles as producer backpressure.  fread+assign remains
only for stdin/pipes.  Measured (workstation, --no-mmap, 20 GiB warm
input, /dev/null, median of 3 alternating): 5.34 s → **3.10 s (+72%
throughput, 3.66 → 6.30 GiB/s)**.  Path ranking per box: mmap (zero
copies) where the kernel allows; buffered-pooled (one copy) otherwise;
O_DIRECT only when the device's raw rate beats its buffered rate.

Not yet covered: the GPU/hybrid reader keeps its copy fallback — its
host-chunk-to-subchunk split means one pooled buffer would back many view
tasks, and slot recycling is single-owner (`direct_buf`); needs a
refcounted release before the pool can serve it.  Decompress readers
untouched (compressed input is 3–30× smaller; far less reader-bound).

Also: reader-state accounting (input mirror of v0.13.59's writer states,
compress readers only): `[READER] io | task-copy | blocked-on-pool` at -v
with its own verdict; task-copy > 0 IS the double-copy diagnosis.  The
nvcomp fread fallback is instrumented too, so hybrid runs on the server
will now show the copy share directly.

## v0.13.59 — writer-state accounting: every run reports whether it pegged the writer

Motivation: runs that grind below the output device's capability are
maddening to triage because the candidate causes — sink saturated, frame
stragglers stalling the in-order writer, or upstream compute/read too slow —
all look identical from the outside (disk not pegged, machine "busy").
Observed concretely on the server benchmarks: cpu-only compress pegs the
~3 GiB/s NVMe at 3.3–3.5 GiB/s while hybrid grinds at ~2.0 with strictly
more hardware in play.

The output side is now modeled as three mutually exclusive states, measured
always (two timestamps per wait segment / write call — negligible):

- **write-path busy** — inside physical write/seek calls (AsyncWritePool
  worker).  ≥85% of wall time means the sink is the bottleneck; mission
  accomplished, nothing upstream can help.
- **head-of-line** — writer idle waiting for the next in-sequence frame
  while LATER frames sit buffered: a straggler (slow GPU batch, unlucky
  frame) is capping output, the pipeline's fault.
- **starved** — writer idle with nothing buffered at all: compute/read
  simply hasn't produced; the engines are the bottleneck.

The buckets accrue on different threads (busy on the AIO worker, the waits
on the writer thread), so percentages overlap and need not sum to 100; each
is independently meaningful against run wall time.  At `-v` each run prints
the three percentages plus a one-line interpreted verdict
(`writer_verdict()`), e.g. `stragglers — writer idled waiting for the next
in-order frame while later frames sat buffered`.  These three signals are
the regime detectors a future --adapt mode would switch on (io-bound /
pipeline-bound / compute-bound).

## v0.13.58 — hybrid compress: slow GPU no longer sets the makespan (tail-aware intake)

On the workstation (slow-GPU box: GPU pool ~1.1 GiB/s vs CPU pool ~15),
hybrid compress ran 26–45% behind cpu-only.  Root cause: in adaptive mode the
GPU popped batches unconditionally, including from the near-empty queue at
the end of the run — a 1 GiB batch takes the slow GPU pool ~2 s while the
entire CPU pool sits idle waiting for it.  The damage was almost entirely a
*tail* effect: mid-run greedy intake is harmless (work conserves, both pools
stay busy), but whoever holds frames when the queue runs dry decides when the
run ends.

**Failed approach (v0.13.56, never released):** cap GPU intake at its
EMA-measured fair share of throughput per ~0.5 s scheduler window, lifting
the cap when the producer finishes so the tail drains greedily.  Falsified by
A/B on a 100 GiB page-cached input: identical to v0.13.55 within noise.  Two
design errors: (1) with mmap input (the default) the reader enqueues view
tasks for the whole file in milliseconds, so "producer done" fires at t≈0 and
the cap was disabled for effectively the entire run; (2) lifting the cap for
the tail re-creates the exact failure being fixed — the tail *is* where
greedy intake hurts.

**Shipped approach:** `should_gpu_take()` (adaptive, compress only) yields
only at the tail — the GPU starts a new batch only if the queue holds more
than ~1.3 GPU-batch-times of CPU work: `(depth − batch)/cpu_ema ≥
1.3 · batch · streams/gpu_ema` (frames are uniform, so frame counts over EMA
byte-rates compare directly).  The check arms once the producer is done
(t≈0 for mmap, so it is live the whole run; a streaming producer keeps the
queue shallow and would otherwise starve the GPU).  The first yield latches
`tail_yield_`, which zeroes `cpu_queue_floor()` and wakes sleeping CPU
workers — otherwise CPUs would refuse the very frames the floor had reserved
for the GPU that just declined them.  The GPU remains the drain of last
resort below `--cpu-queue-min` (CPUs refuse those depths; mutual yield would
hang).  A yielded GPU worker (all streams idle) parks on the queue CV via
`wait_for_gpu_yield()` — event-driven like the CPU side's `wait_for_cpu`, no
polling, no fixed sleeps — and wakes exactly when a decision input changes:
any pop (`take_front_locked` is the centralized dequeue point and notifies
when a waiter is parked; a free integer check otherwise), the queue
draining, or a scheduler tick moving the EMAs (the one input with no queue
event; `notify_gpu_yield_waiters()` covers it).  The wait predicate
evaluates `should_gpu_take_at(depth)` from the QueueState snapshot — like
wait_for_cpu's predicate, it must not call back into TaskQueue.
Fixed-share mode is unchanged (its share check oscillates per-batch by
design and must keep spinning).

Measured (workstation, 100 GiB medium-compressibility input, page-cached,
output to /dev/null, median of 3 alternating runs): hybrid 8.57 s →
**6.31 s** (−26%), variance collapsed (7.8–9.4 s → 6.2–6.3 s); cpu-only
reference 5.94 s.  Hybrid now lands within ~6% of cpu-only on this box
(≈ CUDA init cost) instead of 30–45% behind.  A fast GPU is unaffected:
its fair share is large, so the yield condition only trims the last batch.

## v0.13.55 — portable binary failed to start on machines without an NVIDIA driver

The released portable binary aborted at load time on any machine without the
NVIDIA driver:

    gzstd: error while loading shared libraries: libnvidia-ml.so.1:
    cannot open shared object file: No such file or directory

Root cause: NVML ships with the *driver* (no static archive exists), and the
build linked the CUDA toolkit's stub — which still writes a DT_NEEDED entry
for `libnvidia-ml.so.1` into the binary.  The ELF loader resolves DT_NEEDED at
startup, before `main()`, so the binary could not start at all on driver-less
machines — even though the CPU-only paths never call NVML.  The release
workflow had been masking this in its own smoke test by putting the stub on
`LD_LIBRARY_PATH`.

Fix: NVML is now loaded at **runtime** via `dlopen` (loader shim at the top of
gzstd.cpp; same function names, so call sites are unchanged).  No link-time
dependency remains.  With the driver present, behaviour is identical
(verified: NVML device ranking and PCIe-gen detection still work).  Without
it, the wrappers report failure and the existing fallbacks take over —
free-VRAM device ranking, sysfs PCIe probe — and CUDA detection already
handled the missing driver gracefully through cudart.  Verified by blocking
the driver libraries from dlopen: `--version` and full CPU round-trips work.

Build changes: CMake no longer searches for/links `nvidia-ml` (HAVE_NVML is
on whenever the GPU backend builds on UNIX; `dl` is linked explicitly).  The
release workflow's smoke test now runs with **no** stub on a driver-less
runner — the real user environment — and additionally fails the release if
`ldd` ever shows a hard `libnvidia-ml` dependency again.  BUILD.md updated:
the NVIDIA driver is now optional at runtime.

## v0.13.54 — full-file code review: 3 reproduced critical bugs + GPU-failure CPU fallback

A line-by-line review of gzstd.cpp found three serious decompress-path bugs
(all reproduced on the workstation before fixing) plus a batch of smaller
correctness and robustness issues.  Common thread: error/fallback paths written
in one era weren't revisited when later features (bounded queues v0.13.29/41,
async GPU bringup v0.13.13/15) changed the invariants they relied on.

**1. Silent data loss on concatenated zstd streams (reproduced).** When a
multi-frame input contained a frame with no content-size header *after* at
least one parseable frame (e.g. `cat a.zst b.zst` where b came from
`... | zstd` on a pipe — valid zstd), `stream_frames_to_queue` buffered the
rest of the input in `raw_data` and returned, but both decompress paths only
consumed `raw_data` when *zero* frames had parsed.  Result: the tail was
silently dropped — truncated output, exit 0.  Repro: 2 MiB concatenated input
→ 1 MiB output.  Fix: after the writer drains the parsed frames, append the
tail via the CPU streaming decoder (`decompress_from_buffer`), with a note.

**2. SIGABRT on hybrid/gpu-only decompress of streamed-zstd files
(reproduced).** The `fallback && n_frames == 0` early-return in
`decompress_nvcomp` never joined the deferred GPU-bringup thread: the joinable
`std::thread`'s destructor called `std::terminate` — an abort (exit 134) on
*every* hybrid or gpu-only decompress of a file whose first frame lacks a
content size.  It also iterated `gpu_workers` while the bringup thread could
still be appending to it (data race).  Fix: join the bringup thread first
(matches the normal path's ordering comment).  The fallback message is now a
visible warning explaining that the mode fallback is for data safety.

**3. Deadlock decompressing >64 MiB frames at low thread counts
(reproduced).** `cpu_decomp_worker`'s single-frame-file detection (v0.13.1)
blocked on `producer_done` for every frame over 64 MiB.  Once the input queue
became bounded (v0.13.29), the producer could be blocked in `push()` while
every worker sat in that wait — circular wait, hard hang.  Repro: 1 GiB of
`--chunk-size 128` frames, `-d --cpu-only -T 2` → infinite hang (now 1.2 s).
Fix (also a small speedup): only frame seq 0 can be "the single frame", and a
second pushed task disproves single-frame instantly — oversize frames in
multi-frame files now skip the wait entirely; the genuine wait is a timed
re-checking loop that can't deadlock against a blocked producer.

**4. All-GPU failure in --gpu-only now falls back to CPU instead of dying
(or hanging).** Previously: all GPUs failing at init aborted the reader and
died with EXIT_GPU_FAIL — but mid-run failures weren't counted by that check,
so with pipe input the producer blocked forever on the bounded queue (no
consumer), and with mmap input the writer's watchdog killed the run with a
misleading "internal error: writer stuck" (exit 1).  Now the last terminally
failing GPU worker runs `gpu_only_cpu_fallback`: a full CPU pool drains the
queue (maximum remaining throughput), a warning explains the fallback is for
data safety, and the run completes with exit 0.  Exit code 5 is now reserved
(documented in --help); the reader-side abort checks are gone — the queue
always has a consumer.

**5. Missing nvCOMP per-chunk status check in the compress sync-drain path
(silent corruption risk).** The async-poll completion path validated
`h_stats[i]`; the sync drain did not — a failed chunk's garbage comp_size
would have been delivered as output.  Both paths now check.

**6. GPU-decompress rescue re-enqueued empty tasks.** Inputs were released
("it's on the GPU now") *before* the kernel ran, so any failure after that
point re-enqueued zero-byte tasks — the retry was dead on arrival.  Inputs are
now released per-frame after successful delivery.

**7. Partial-batch failure accounting (compress + decompress).** A throw
mid-delivery rescued/re-enqueued the *whole* batch including frames already
pushed to the ResultStore: duplicate-seq work and permit drift.  Streams now
track `delivered` and the failure paths handle only the undelivered tail, with
exact permit release.

**Smaller fixes:**
- `--fast=abc`, `-M abc`, `--memlimit=`/`--memory` garbage, and overflowing
  `-NNN…` levels crashed with an uncaught std::stoi/stoull exception; now
  clean usage errors (exit 2).  Malformed level flags like `-5x` were silently
  swallowed (and compressed at the default level!); now "unknown option".
- Corrupt frame headers claiming absurd content sizes aborted via uncaught
  `std::bad_alloc`; now a clean data error (exit 4).
- `tasks_done` was double-counted on CPU decompress (writer + worker), skewing
  the progress bar's frame-level percentage; the writer is the sole counter.
- O_DIRECT + sparse: a sparse seek from a non-4 KiB-aligned position left the
  fd offset unaligned, making the next O_DIRECT flush fail with EINVAL
  (reported as "disk full?").  Sparse skips through the DirectWriter now only
  happen from aligned positions; unaligned zero runs are written instead
  (correct, merely less sparse — unreachable with gzstd's own 16 MiB frames).
- `DirectWriter::write_all`/`pwrite_all` looped forever on `write() == 0`;
  `robust_fwrite` could loop on a stale `EINTR` in errno.  Both bounded.
- gpu_worker counted `--direct-read` view bytes into the progress meter that
  the O_DIRECT reader had already counted (mirrors cpu_worker's guard).
- Removed the write-only `Options::remove_input` field (`--rm` works via
  `keep = false`).

## v0.13.53 — reconcile --help / -h with actual operation (docs only)

Help text had drifted from the code after a lot of churn. Audited every flag in
both screens against the parser and the runtime defaults; the flag names all
matched, but three stated defaults/details were stale. Docs follow code (no
behaviour change):

- **`--gpu-batch` default.** Both screens said `default: 16`. Actual: 8 for
  compress (`DEFAULT_GPU_BATCH_CAP`), 16 for decompress auto-scaled up by input
  size (64 above 10 GiB, 256 above 75 GiB). Updated both screens to state the
  mode-dependent default.
- **`--gpu-devices` auto.** Long help and the `Options::gpu_devices` comment said
  auto = "all GPUs for compress, 1 for decompress". Both decompress paths
  (synchronous and deferred) actually use all available GPUs, same as compress.
  Corrected the wording and the struct comment.
- **`--cold`.** Documented in the short help but only mentioned in passing in the
  long help; gave it its own entry in the long-help I/O section.

## v0.13.52 — fix GPU-compress hybrid rescue dropping mmap/view tasks (silent data loss)

Two correctness fixes in the failure/edge paths, found by code review.

**1. Hybrid GPU-compress rescue lost zero-copy (mmap) frames.** When a GPU failed
mid-batch in hybrid mode, the worker's `catch` re-routed the in-flight chunks to the
CPU rescue queue by reconstructing each task as `Task{ seq, data }`. But the default
compress reader for a regular file is the zero-copy **mmap** reader, whose tasks carry
their bytes in `view_ptr`/`view_len` with an **empty `data` vector**. Rebuilding from
`.data` alone dropped the view, so the rescue worker compressed **0 bytes** and emitted
an empty zstd frame for that sequence number — the output decompressed cleanly but was
**silently missing those chunks' bytes**. This corrupted output on the exact path the
rescue mechanism exists to handle gracefully (VRAM exhaustion / driver error), whenever
the input used the mmap reader (the default). Fix: `std::move` the whole `Task` into the
rescue queue instead of reconstructing it — the mmap region outlives the rescue join so
the view stays valid, and the move also preserves `direct_buf` ownership and avoids
copying owning data. The sibling paths were already correct (GPU-only failure and the
decompress failure path both `re_enqueue` the intact tasks); only this hybrid
`rescue->push` reconstructed the task.

**2. Throttle permit over-release in single-frame streaming decompress.** The CPU
decompress worker's streaming branch (single giant frame, e.g. `--ultra` / `zstd`
output) acquired exactly one `FrameThrottle` permit before the pop but pushed N result
chunks, and the writer releases one permit per frame written — so the writer
over-released by `(actual_chunks - 1)`, drifting `permits_` above its cap (and making
`in_flight()` read negative). Harmless in practice (only fires on a single-frame file at
end of work, and in-flight memory is independently bounded by the per-thread decomp
pool), but a real acquire/release asymmetry. Fix: acquire one additional permit per
streamed chunk beyond the first, so acquires match releases. Deadlock-free — chunks
ascend from the lowest seq, so the writer always drains the oldest first and frees a
permit.

## v0.13.51 — `--direct-read` for decompress (was compress-only)

`--direct-read` only honored the compress reader; decompress silently fell back to
buffered `fread`, so a benchmark or cold run would read the compressed input warm.
Wired O_DIRECT into `stream_frames_to_queue` (the shared reader behind both
`decompress_cpu_mt` and `decompress_nvcomp`): when `--direct-read` is set on a
regular-file input it opens its own O_DIRECT fd on `opt.input` and reads 4 KiB-aligned
`READ_CHUNK` (4 MiB) blocks into an aligned bounce buffer, copying into the existing
frame-parse buffer (frame boundaries don't align to reads, so a bounce copy is
required — but it's the same copy `fread` did internally). The FILE* `in` is at offset
0 here (`peek_first_frame_decomp_size` rewinds) and is simply unused for reading while
O_DIRECT is active; the streaming-fallback path (unknown content size) reads through
the same helper. Falls back to `fread` if O_DIRECT can't be set up. Byte-identical
decompressed output vs the buffered reader on cpu-only and gpu-only across multi-frame
+ unaligned inputs; round-trip clean; 290/290 extensive. (Known minor gap: the
single-giant-frame streaming path, `decompress_stream_from_file` for inputs above
`SINGLE_FRAME_STREAM_MIN` — i.e. `--sliding-window` / `zstd` outputs — still reads
buffered; gzstd's own chunked output is multi-frame and uses the wired path.)

## v0.13.50 — `--direct-read`: one contiguous pool region (fix the ~340 KiB request split)

After v0.13.49, `--direct-read` was still only 1.55 GiB/s vs the page-cache path's
4.46. `-vvv` showed the reader spends 99% of wall *inside* `pread` (279 s of a 283 s
run) yet moves only 1.55 GiB/s — not starvation (cores were 97% idle waiting on it),
the `pread`s themselves are slow. `iostat` found why and `dd` confirmed it:

| | rareq-sz | aqu-sz | throughput |
|---|---|---|---|
| dd (16 MiB O_DIRECT) | **638 KiB** | 8.5 | 3.8 GB/s |
| our reader | **340 KiB** | 15 | 1.8 GB/s |

Same requests/sec; dd's DMA requests are ~2× larger — the whole gap. Our 150 pool
buffers were separate `posix_memalign(16 MiB)` calls, and because 16 MiB is below
v0.13.48's `M_MMAP_THRESHOLD` they came from the **fragmented heap**: physically
scattered 4 KiB pages, so O_DIRECT's scatter-gather list hits the driver's
`max_segments=127` and each 16 MiB read shatters into ~340 KiB requests.

Fix: allocate the **whole pool as one large region** (> the mmap threshold ⇒ a fresh
dedicated `mmap`) and slice it. On an unfragmented box its pages fault in as long
physically-contiguous runs that merge into a few big DMA segments, so a `pread`
reaches the device's `max_sectors_kb`. Measured on the server: **rareq-sz 340 →
~1230 KiB** (≈ the 1280 KiB max), throughput **1.55 → 1.96 GiB/s**, run 4m43 → 3m41.
The region is 2 MiB-aligned + `MADV_HUGEPAGE` + lightly pre-faulted as
belt-and-suspenders for THP where it's healthy, but **THP did not engage on the 5.15
server** (`AnonHugePages=0`) — the win is the contiguous allocation, not huge pages.
Byte-identical cpu+gpu, round-trip clean, 290/290 extensive.

Remaining limiter (not a code issue): with `--direct --direct-read` on one drive,
O_DIRECT reads (~1.9 GB/s) and the O_DIRECT output writes (~0.8 GB/s) **contend for
the same device queue** — iostat shows ~2.7 GB/s mixed at 80% util. The page-cache
path avoids this only because its reads are free (served from RAM), leaving the whole
drive for writes. Reading and writing on separate drives removes the contention; on a
single big-RAM box the buffered path remains the throughput king and `--direct-read`'s
value stays honest-cold benchmarking + not evicting other users' cache.

## v0.13.49 — `--direct-read`: single-stream + zero-copy reader

Two `dd` facts settled the design. (1) On this NVMe a *single* O_DIRECT stream does
**4.5 GB/s** (4.9 at 128 MiB), while **4 independent streams collapse to ~3.0 GB/s
aggregate** (0.77 each) — concurrent O_DIRECT *contends*, it does not scale. So the
v0.13.46/47 multi-threaded reader was wrong for this hardware. (2) A single `dd`
stream already saturates the drive, yet our pipeline extracted only ~1.5 GB/s of
that 4.5 — because every chunk did a 16 MiB `memcpy` from the O_DIRECT buffer into
the Task, and on the 256-core box that copy competes for memory bandwidth with the
compressors and stalls the read stream between requests.

Rewritten as **one stream, zero copies**:
- **Single reader.** Dropped the work-stealing/multi-thread machinery; one
  uninterrupted O_DIRECT stream is fastest here.
- **Zero-copy (CPU path).** `pread` lands straight in a pooled 4 KiB-aligned buffer
  (`DirectReadPool`); the Task aliases it as a `view_ptr` (like the mmap path) and
  the worker recycles the slot on `release_input()`. No per-chunk copy — the stream
  reads continuously. `pool->acquire()` blocks when all buffers are in flight, so
  the pool *is* the producer backpressure (the queue byte-cap is a no-op for
  zero-byte view tasks). Pool sized to keep every worker fed plus a read-ahead
  backlog (`threads + 128`, capped by file size and 1024); a buffer is held only
  from pread until compression finishes (not during write), so peak RSS stays
  bounded. Read-byte metering moved fully to the reader for these views (workers
  skip `direct_buf >= 0` tasks) — no double-count.
- **GPU path unchanged.** It splits each host chunk into gpu subchunks, so one
  owning buffer per read doesn't map to one Task; it keeps the copy (pool == null,
  single scratch buffer), where PCIe dominates anyway.

Expected to close most of the 1.5 → ~4.5 GB/s gap on the 256-core box (capped by
the compressor at ~3.5); the win is memory-bandwidth-bound so it doesn't show on a
low-core workstation (where the copy was never the bottleneck) — local runs confirm
byte-identical output (cpu + gpu, multi-chunk + unaligned tail), clean round-trip,
correct read metering, and bounded RSS. 290/290 extensive. (v0.13.48's mallopt stays
— it still helps the GPU/fread/decompress and output-buffer paths.)

## v0.13.48 — Recycle frame buffers (pin mmap threshold) — kill the munmap TLB-shootdown storm

After v0.13.47 fixed the reader's access pattern, `--direct-read` still ran at only
~1.35 GiB/s on the 432 GiB compress while `dd` showed the same NVMe doing **4.5
GB/s single-stream O_DIRECT** (4.9 at 128 MiB blocks) — so ~70% of the drive was
left on the table, and the limiter was ours, not the disk. Root cause: our
per-frame buffers are 16 MiB (32 MiB ultra), above glibc's 32 MiB dynamic
`mmap` ceiling, so under the 4-producer/N-consumer hand-off the allocator `mmap`s
each chunk and **`munmap`s it on free**. The munmap — not the page faults — is the
killer: tearing down a 16 MiB mapping triggers a TLB shootdown (an IPI to every
other core), whose cost scales with core count, so on the 256-core server it
dominated (the run's enormous `sys` time).

Fix: at startup, `mallopt(M_MMAP_THRESHOLD, 128 MiB)` + `mallopt(M_TRIM_THRESHOLD,
256 MiB)` so frame buffers come from the heap and freed chunks are reused from the
arena bins — no munmap, no shootdown, no re-grow. Local A/B on a 4 GiB direct-read
(no 256-core shootdown tax to begin with): wall **2.97 s → 1.58 s** (~1.9×, 1.35 →
2.53 GB/s); minor faults essentially unchanged (255k → 243k), confirming the win is
the syscall/shootdown churn, not the faults. Expected to help *more* on the
256-core box where the shootdown cost is highest. Benefits every path that churns
large per-frame buffers (compress, decompress, GPU host staging), not just
`--direct-read`. Peak RSS stays bounded by the in-flight cap. 290/290 extensive.

## v0.13.47 — Fix the v0.13.46 parallel O_DIRECT reader (strided → work-stealing)

v0.13.46's parallel reader was catastrophically slow in practice — on a 432 GiB
real-data run it crawled at ~95 MiB/s (vs 1.47 GiB/s for the v0.13.44 single
thread it replaced, and 3.51 GiB/s for the page-cache path), starving the whole
pipeline so it looked like the writer had stalled. Two flaws, now fixed:

- **Strided assignment was the killer.** Each thread took `idx += ODIRECT_READERS`
  (a 64 MiB stride). That is only sequential while the threads stay in lockstep;
  the first copy/push stall desynchronises them and the in-flight reads scatter up
  to `N*cap` apart into a random-looking pattern. With O_DIRECT (no kernel
  readahead) that destroys NVMe locality. Replaced with **work-stealing**: a shared
  `atomic<size_t> next_idx` hands the next chunk to whichever reader is free, so the
  N outstanding reads are always on *consecutive* chunks — a contiguous window that
  slides forward at queue depth N. Near-sequential access, still deep-queued.
- **Shared fd.** All readers `pread` one fd; each now opens its own O_DIRECT fd to
  avoid serialising on the shared file struct.

seq is still the chunk index (file position), so output stays ordered and the
ordered writer keeps RAM bounded regardless of completion order. Output
byte-identical to the normal reader on cpu-only and gpu-only across multi-chunk +
unaligned-tail files; round-trip clean; 290/290 extensive.

## v0.13.46 — Parallel O_DIRECT reader (--direct-read was QD1-bound)

The v0.13.44 --direct-read used a single synchronous pread loop, which runs the
NVMe at queue-depth 1: O_DIRECT has no kernel readahead, so the drive sits idle
between reads and the reader starves the workers (measured ~1.47 GiB/s on a 432
GiB cold read vs ~4.5 the drive can do). Now ODIRECT_READERS=4 reader threads
each pread strided, 4 KiB-aligned host chunks, keeping multiple requests in
flight (deep queue) to saturate the NVMe. seq is assigned deterministically from
the chunk index (file position), not a shared counter, so frames stay correctly
ordered/contiguous despite out-of-order completion across threads (completion is
tracked by push count). The per-chunk copy lives in a noinline enqueue helper so
vector::assign is analysed in a clean context (avoids a -Wnonnull false positive
on the alloc when inlined into the threaded reader). Output byte-identical to the
normal reader on cpu-only and gpu-only across multi-chunk + unaligned-size files;
290/290.

## v0.13.45 — Document the mmap kernel-gate + `--direct-read` in `--help`

Help text only.  The extended (`--help`) and short (`-h`) entries for
`--mmap`/`--no-mmap` now describe the v0.13.43 auto-gate (kernel <6.4 + input
>4 GiB → fread; mmap on for 6.4+), and a new extended `--direct-read` entry
explains the O_DIRECT page-cache bypass, its one-pass-speedup vs honest-cold-
benchmark uses, that it implies fread and is independent of `--direct` (output).

## v0.13.44 — `--direct-read`: O_DIRECT input reader (page-cache bypass)

A first-class O_DIRECT input reader for the compress path (`compress_cpu_mt` and
`compress_nvcomp`).  O_DIRECT transfers straight disk→buffer, **bypassing the page
cache entirely** — it neither reads from nor populates it.  Two payoffs:

1. **One-pass speedup (real feature):** compressing a backup touches every input
   byte exactly once and never re-reads it, so the page cache provides zero reuse
   benefit — it's pure populate + writeback-pressure overhead.  Reading around it
   skips that.  (Caveat: O_DIRECT loses kernel readahead, so it needs large reads
   to keep the disk saturated — gzstd's 16 MiB chunks + pipelining cover that.)
2. **Honest cold benchmarking with zero system impact:** because nothing is cached
   or evicted, every run reads cold from disk deterministically — no warm-cache
   skew, and critically **no `kcompactd` storm**.  (The old `--cold` =
   `fadvise(DONTNEED)` *populates then drops* the cache; dropping a huge file
   fragments free memory and wakes kernel compaction, stalling the whole box.
   O_DIRECT sidesteps the drop entirely — see project_mmap_kernel_storm.)

Shared helper `odirect_read_chunks` (4 KiB-aligned `pread` into a `posix_memalign`
bounce buffer; EOF handled via O_DIRECT's short read; falls back to fread if
O_DIRECT can't be set up, e.g. tmpfs).  Takes precedence over mmap (mmap *is* the
page cache, so it can't bypass it).  Output is byte-identical to the normal reader
on both cpu-only and gpu-only (verified); round-trips incl. unaligned-size files;
290/290.  `--direct` (O_DIRECT *output*) and `--direct-read` (O_DIRECT *input*)
are independent — combine for a fully cache-bypassing run.

## v0.13.43 — Auto-fall-back to fread for large inputs on pre-6.4 kernels (mmap_lock storm)

On a 256-core box at kernel **5.15** (pre-6.4, no per-VMA locks), compressing a
432 GiB file with the default mmap reader cost **13–41%** vs `--no-mmap`: the
single `mmap_lock` rwsem serialises ~108M page faults across 256 cores (the file
fits in 1.5 TiB RAM, so it's pure lock contention, not eviction).  The v0.13.22
"mmap on everywhere" decision was calibrated on 20 GiB test files where the storm
was a tolerable regression; it scales with fault count = file size, so real
backup-scale files hit it ~20× harder.

Fix: gate mmap on **kernel version + input size** — on kernels `< 6.4`
(`kernel_has_per_vma_locks()` via `uname`), fall back to fread for inputs
`> 4 GiB`.  6.4+ kernels and small files are unchanged (mmap stays on, where its
zero-copy wins and few faults don't storm), so this is *not* the kernel-only gate
that regressed cold small files in v0.13.20–22, and *not* a core-count gate.
`--mmap`/`--no-mmap` hard-override it; the gate auto-retires when the box reaches
≥6.4.  Verified gate-off on a 6.17 box; 290/290.

---

## v0.13.42 — Fix `-T` with no numeric value crashing (uncaught `std::stoi`)

`-T`/`--threads` parsed their separate value with a raw `std::stoi(argv[++i])`,
so a bare `-T` followed by a non-numeric token — `gzstd -T --cpu-only file`, or
`-T file.zst`, or a trailing `-T` — threw an uncaught `std::invalid_argument`
and `abort()`ed (core dump).  The attached `-T4` form had the same unguarded
`stoi` (`-Tx` would crash too).

Fix: the separate form now only consumes the next token as the thread count when
it actually looks like an integer (new `looks_like_int` helper); otherwise the
token is left for normal parsing and `-T` falls back to the default (auto) thread
count — no crash, no error.  Both the attached (`-T4`, `--threads=N`) and
consumed-value paths now go through `parse_int_value`, which reports a clean
usage error (exit 2) on a genuinely malformed attached value (`-Tx`) instead of
aborting.  `-T0` (= all cores), `-T N`, `-T4`, `--threads=N`, `--threads N` all
unchanged.  Verified across the extensive suite's `-T` cases.

---

## v0.13.41 — Extend the byte cap to the compress producer (pipe/stdin RAM safety)

(Committed together with v0.13.40.)  The 7.8/v0.13.40 queue cap was decompress-only,
but **compress had the same exposure**: for a regular file the producer mmaps the
input (zero-copy views, no heap — a 1 TB file streams in bounded RAM regardless of
size), but for a **pipe/stdin** input it falls back to `fread`, reading frames onto
the heap, and the compress queue had *no* cap.  A producer that outruns the workers
(or a writer/disk bottleneck that blocks them on throttle permits, so they stop
popping) could then buffer the entire input in RAM → OOM.

Fix: call `queue.set_max_bytes(floor * host_chunk/2)` on both compress queues
(`compress_cpu_mt`, `compress_nvcomp`).  Bytes only, no frame cap — mmap views are
`data.size()==0` so it's a **no-op for the common regular-file path** and bounds only
fread.  Reuses the same `TaskQueue` machinery (`max_bytes_`/`queued_bytes_`/
`take_front_locked`, `!q_.empty()` deadlock guard) added in v0.13.40.

Demonstrated (Gen3, 2 GiB incompressible piped via `cat | gzstd --cpu-only -T2 -19`,
slow workers so the warm-cache pipe outruns them; max-RSS):

| build                  | maxRSS   | time    |
|------------------------|----------|---------|
| before (no compress cap) | 2232 MiB | 193.2 s |
| after  (byte cap)        |  568 MiB | 191.5 s |

−75% peak RSS, throughput unchanged.  At default level the workers keep up so the
queue never grows and the cap doesn't engage (both builds ~153 MiB) — it's a safety
net for the slow-worker / slow-output pathological case, where without it a large
pipe input OOMs.  Pipe + mmap round-trips verified across cpu/gpu/hybrid; 213/213.

## v0.13.40 — Byte-aware decompress reader queue cap (ROADMAP 7.8 follow-up): bound queue RAM

The 7.8 reader queue cap (`set_max_depth`) bounds the queue by **frame count**
(`parallelism * slack`), so the RAM it holds scales with compressibility — an
incompressible file (near-full-size compressed frames) buffers ~4× the RAM a
compressible one does for the same frame count.  Added a parallel **byte** cap to
`TaskQueue`: the reader now blocks when `frames >= max_depth_` **OR**
`queued_bytes_ >= max_bytes_`, whichever binds first.  `queued_bytes_` tracks
owned heap (`Task::data.size()`) so zero-copy mmap views (size 0) are correctly
ignored; a `!q_.empty()` guard on the byte cap guarantees progress even when a
single frame exceeds the whole budget (no deadlock).  Byte accounting is
centralized in one `take_front_locked()` helper so it can't drift across the pop
sites.  Budget = `floor * 8 MiB` (~half a standard 16 MiB frame per slot), set at
both decompress readers; tunable via `--throttle-factor`.

Measured (Gen3 2×2080Ti, `gpu-only` decompress, 4 GiB, max-RSS / best-of-3):

| profile         | RSS before | RSS after  | Δ      | time |
|-----------------|------------|------------|--------|------|
| low_compress    | 2127 MiB   | 1902 MiB   | −11%   | flat |
| medium_compress | 1748 MiB   | 1603 MiB   | −8%    | flat |

Throughput-neutral, RAM down 145–225 MiB — biggest on incompressible input, as
intended.  The reduced buffering for big frames *could* matter on a much faster
reader/consumer ratio (knuth, 8×H100) — flagged in-code and tunable; validate
there.  Round-trips verified cpu/gpu × incompressible + sparse `zeros` (deadlock-
checked with timeouts); 213/213 tests pass.  CPU decompress RSS is unaffected (its
RAM is the output-buffer throttle budget, not the input queue).

---

## v0.13.39 — Default-init allocator on `FrameBuf`: eliminate the resize() zero-fill on direct-write paths

The `assign()` fixes (v0.13.37/38) only cover handoffs where bytes pass through
host memory first.  The **direct-write** paths — CPU decompress (`ZSTD_decompressDCtx`
writes straight into the buffer), GPU non-pinned D2H — still did
`out_buf->resize(decomp_size)` before the write, value-initializing (zeroing) the
grown region that the decompressor then fully overwrites.  Profiling CPU decompress
(callgrind) pinned this at **`_M_default_append`→memset = ~16% of instructions**:
the decomp buffer *pool* is large (`throttle_budget/n_workers`), so for any file up
to ~pool-size frames almost every frame grows a never-yet-used buffer and pays the
full ~16 MiB zero (it amortizes only on very large files).

Fix: `FrameBuf` now uses a `default_init_allocator<char>` (a `std::allocator`
subclass whose no-arg `construct()` default-initializes instead of value-initializes),
so `resize()`-grow leaves the new region uninitialized rather than zeroed.  Safe:
every producer fully fills `[0,size())` before the buffer is read (ZSTD / cudaMemcpy /
assign / memcpy), and no writer reads past `size()` — `DirectWriter` copies exactly
`size()` bytes into its own aligned buffer, so a buffer's `[size(),capacity())` tail
never reaches disk.  Ripple was contained: the typedef, 5 `make_shared` sites, and
the per-thread compress `scratch` (now `FrameVec` so it still `swap`s with the pooled
output).

Confirmed: `__memset` 16.2% → 1.9%, `_M_default_append` zeroing gone.  **Throughput-
neutral** in wall-clock, though — those cycles were parallel across workers and
overlapped with the memory-bandwidth/writer bottleneck, so this removes provably-
wasted CPU cycles + ~0.9× output-size of memory-write traffic per decompress rather
than adding speed.  Kept deliberately as resource-waste elimination.  All 30
compress×decompress combinations (5 profiles × cpu/gpu/hybrid × cpu/gpu decompress)
bit-identical incl. the sparse `zeros` path; 213/213 tests pass.

## v0.13.38 — `assign()` for the GPU decompress pinned readback

Extends v0.13.37's `assign()` change to the GPU decompress D2H readback pinned path
(`gpu_decomp_worker`, both completion paths): `h_out->resize(actual)+memcpy(pin_slot)`
→ `h_out->assign(pin_slot, pin_slot+actual)`.  Here `actual` is a full decompressed
frame (~16 MiB), so the avoided zero-fill is far from tiny when the pooled buffer
grows.  Non-pinned direct-D2H keeps `resize()` (dst must be pre-sized; superseded by
the v0.13.39 allocator anyway).  Round-trips verified; 213/213 tests pass.

---

## v0.13.37 — Use `assign()` for buffer handoffs to drop the residual resize() zero-fill

Follow-up to v0.13.36.  Three handoff sites took a recycled pooled `FrameBuf`,
`resize(csz)`'d it, then `memcpy`'d `csz` bytes over it — value-initializing
(zeroing) the grown region only to immediately overwrite it.  Replacing
`resize(csz)+memcpy` with `vector::assign(src, src+csz)` does the identical copy
but copy-constructs straight from the source, so it never zeroes — same bytes,
no wasted memset.  Sites:
- CPU compress worker, well-compressible (memcpy) branch.
- GPU compress D2H readback, **pinned** path (async-poll + sync-drain) — the bytes
  already pass through the pinned host slot, so `assign` from that slot applies.

This is the clean, local alternative to the `FrameBuf` default-init allocator
considered (and rejected as too invasive) for the GPU readback — no type change,
no extra copy, strictly less work than what it replaces.  The GPU **non-pinned**
direct-D2H fallback still needs `resize()` (the dst must be pre-sized before
`cudaMemcpy` writes into it; `assign` can't source from device memory) — that's
the slow fallback path, left as-is.

Throughput-neutral by design — the eliminated zeroing was the 0.59%-class
residual measured on a non-bottleneck host thread (see the closed GPU-readback
ROADMAP check) — but it stops doing provably useless work.  Round-trips verified
on cpu-only + gpu-only across all four profiles; 213/213 tests pass.

---

## v0.13.36 — CPU compress: stop zero-filling the output buffer every frame

Profiled the cpu-only compress path (callgrind, no root: `valgrind --tool=callgrind`
on the real `-T>1` per-frame path — note `-T1` takes the separate
`compress_cpu_stream` streaming path).  After zstd's own entropy kernels (~67%,
all BMI2-SIMD and not ours to touch), the single largest *gzstd-attributable*
cost was `std::vector<char>::_M_default_append → memset`, ~7% of total
instructions: `compress_one_cpu_frame` did `out.resize(bound)` before every
`ZSTD_compress2`, value-initializing (zeroing) up to ~16 MiB per frame — pure
waste, since ZSTD only writes `[0,csz)` and never reads the dst buffer, and the
`[csz,bound)` tail is undefined padding anyway.  Because the buffer was then
shrunk to `csz`, the next frame re-grew and re-zeroed `(bound − csz)` bytes —
worst on compressible data, where `csz` is tiny so nearly the whole buffer was
re-zeroed each frame.

Fix: `compress_one_cpu_frame` now grows the reusable per-thread buffer to
`compressBound` **once** (grow-only, never shrunk) and returns `csz` explicitly
instead of resizing the buffer down.  `resize()` then zeroes once on a thread's
first frame and no-ops thereafter.  The poorly-compressible "zero-copy swap"
branch (ROADMAP 7.4) resizes the swapped-in buffer down to `csz` after the swap
(a shrink — no zeroing).  Output bytes are identical (no ratio change).

Measured on the 24-core workstation, cpu-only `-T8`, 4 GiB inputs, compute-bound
(output to /dev/null), best of 5:

| profile          | before     | after      | gain  |
|------------------|------------|------------|-------|
| high_compress    | 17.1 GiB/s | 19.5 GiB/s | +14%  |
| medium_compress  |  8.9 GiB/s |  9.8 GiB/s | +9.5% |
| low_compress     | 11.1 GiB/s | 11.4 GiB/s | +2.8% |
| mixed            | 12.0 GiB/s | 12.4 GiB/s | +3.0% |

Round-trips verified on all four profiles; 213/213 tests pass.  (The win amortizes
over frames-per-thread, so it needs frames ≫ threads to show — a 4-frame/4-thread
microbenchmark gives each thread only its one-time first-frame zeroing and hides
it.)

---

## v0.13.35 — HybridSched: don't cap CPU when no GPU is active (ROADMAP 7.10)

Completes the 7.10 deep-dive by auditing the third target, `HybridSched`.  No
deadlock, missed-wakeup, or `gpus_waiting_` accounting bugs found: in fixed-share
mode `should_cpu_take`/`should_gpu_take` can never both be false (so one engine
always takes), the queue floor is enforced atomically in `may_take` and correctly
skipped in fixed mode, `push()` wakes one CPU per task while the exit paths
`notify_all`, and `gpu_got_data()` always fires before any throwing CUDA op.

One robustness gap fixed: in fixed `--cpu-share` mode, if the GPU(s) never
register (still initializing) or all fail/exit mid-run, nothing advances
`gpu_taken_`, so the share cap (`cpu/total < share+0.02`) stalls the main CPU
workers until the producer finishes — wasting the whole production phase before
the drain fast-path recovers.  `should_cpu_take` now short-circuits to `true`
when `active_gpu_streams_ == 0`, letting CPU run unrestricted whenever no GPU is
present to consume its share.  Adaptive mode already handled this via the
`gpus_waiting_`/floor path; healthy fixed-share runs keep `active_gpu_streams_ >
0` throughout, so this is a no-op for them (verified: `--cpu-share 0.5` hybrid
round-trip unchanged, GPU active throughout).

213/213 tests pass.  This closes the 7.10 audit — all three targets (auto-tuner,
failure rescue, HybridSched) are now done.

---

## v0.13.34 — GPU-failure rescue: fix permit leak + stranded-batch hang (ROADMAP 7.10)

Deep-dive audit of the VRAM-exhaustion / GPU-failure rescue path (one of the
7.10 targets) found two correctness bugs in the worker catch blocks, both of
which only fire when a device fails mid-run — exactly when graceful rescue
matters most.

1. **FrameThrottle permit leak (compress + decompress).**  The throttle invariant
   is "the popper acquires one permit per frame; the writer releases it after the
   frame is written."  The success paths honour this (the GPU bulk-acquires
   `pop_n` permits and the writer releases them one-by-one).  But both catch
   blocks handed in-flight frames to the rescue queue / re-enqueued them
   **without releasing the GPU's permits** — and the receiver (rescue worker, or
   the next popper after re_enqueue) re-acquires a fresh permit per frame.  That
   leaked one permit per rescued frame, up to `streams × per_stream_batch` per
   device failure.  Since the auto budget floor is `devices × streams ×
   per_stream_batch`, losing a device's permits could starve the surviving
   rescue/CPU workers into deadlock.  Both catches now release the held permits
   on hand-off (matching the in-loop decompress re_enqueue, which already did).

2. **Stranded batch on submit-time failure (compress only).**  The compress
   catch guarded rescue on `C.busy && !C.batch.empty()`, but `C.busy` isn't set
   until *after* the H2D copies and the `nvcompBatchedZstdCompressAsync` launch —
   any of which can throw.  A launch failure therefore left the just-popped batch
   un-rescued: those frames never reached the sequence-ordered `ResultStore`, so
   in hybrid mode (no abort) the writer blocked forever on the missing seq.  The
   guard is now `!C.batch.empty()` (matching the decompress side); both success
   paths clear `C.batch`, so a non-empty batch is always popped-but-undelivered
   and safe to rescue without duplicate-output risk.

Also removed dead code surfaced by the same audit: the per-stream
EXPLORE/REFINE/SETTLE batch-size tuner (the `TuneState` enum + `tune_*` /
`refine_*` fields in the compress `StreamCtx`, its ~26-line save/restore across
buffer reallocation, and the identical unreferenced block in the decompress
worker's per-stream struct).  It was fully superseded by the cross-GPU
`SharedTuneState` hill-climb that all streams/devices share — same class as the
7.7 `SequentialDispatcher` removal.  No behavioural change; batch size already
came solely from `shared_tune`.

213/213 tests pass.

---

## v0.13.33 — Recycle GPU compress output buffers; CLAUDE.md line count

Finishes the compress side of ROADMAP 7.2 (the decompress side landed in
v0.13.24).  Both GPU-compress completion paths (`gpu_worker` async-poll and sync
drain) did a fresh `make_shared<std::vector<char>>(csz)` per frame for the D2H
readback buffer; `StreamCtx` now owns the same recycled `out_pool` as the
decompress path (`acquire_out_buf`, `use_count()==1` reclaim, lazy growth to two
batches, drain-wait past the cap).  Lower-value than the decompress pool —
compress output is the *compressed* bytes (small for compressible data) — but it
removes the per-frame allocation churn on the GPU/hybrid compress paths, most
relevant for incompressible input where csz approaches the chunk size.

Round-trips verified on gpu-only and hybrid compress (incompressible + mixed);
213/213 tests pass.  This closes 7.2 — GPU result buffers are now pooled on both
compress and decompress.

Also: corrected the stale `~6,400 lines` figure in CLAUDE.md (the file is ~9,900).

---

## v0.13.32 — CPU compress: zero-copy swap for poorly-compressible frames (ROADMAP 7.4)

`cpu_worker` compressed into a per-thread `scratch` buffer, then `memcpy`'d the
result into a pooled `FrameBuf` for the writer.  For poorly-compressible data
(`mixed` ~50%, `low` ~90%) `csz` is a large fraction of the chunk, so that was a
near-full-chunk memcpy per frame on exactly the profiles where compress is
slowest.

Now, when `csz >= in_size / 2`, the worker `std::swap`s the scratch buffer
straight into the pooled `FrameBuf` (zero-copy) and takes the pool slot's old
buffer as the next scratch.  Well-compressible output (`csz` small) keeps the
memcpy path — the copy is cheap there, and swapping would leave every pool slot
carrying scratch's full `compressBound` capacity (a memory regression for tiny
output).  The threshold confines the capacity overhang to slots that already
hold large frames.

Correctness verified (round-trips on incompressible/swap path, zeros/memcpy path,
and mixed/both, multi-threaded); 213/213 tests pass.

**Benchmark verdict (Gen5, 256-core):** the throughput change is **within run
noise** — cpu-only `low` compress (the only profile that crosses the threshold)
moved +4%, identical to the memcpy-path controls (`high` +4%, `medium` +3%) and
no bigger than the swings on paths 7.4 can't touch (gpu-only compress, all
decompress, ±6%).  That's expected: the eliminated memcpy (~14 MiB at memory
bandwidth, ~1 ms) is only ~1–2% of per-frame compress time at level 3, below the
measurement floor.  **Kept anyway** — the old memcpy was pure data-shuffle
overhead (scratch → pool), so the swap path does strictly less work, is correct,
and carries a negligible RSS overhang (only `low` swaps, and its `csz` ≈ 14.4 MiB
is close to the 16 MiB `compressBound`, so ~1.6 MiB/slot; `mixed` at 49.9% stays
on memcpy).  No throughput regression, leaner code path.  ROADMAP 7.4 closed.

---

## v0.13.31 — Fix two Gen4+ regressions from the --direct default (sparse + log tag)

Two test failures surfaced on a Gen5 box, both fallout from making `--direct` the
default on Gen4+ (v0.13.25/26) — they only manifest where `--direct` auto-engages,
so Gen<4 never saw them.

- **Sparse output defeated by preallocate.**  Default `--direct` decompress uses
  the DirectWriter, which preallocates the output (`fallocate`); the sparse path
  then `lseek`s over zero regions whose blocks are *already allocated*, so the
  file came out fully allocated instead of sparse (a real disk-bloat regression
  for zero-heavy data, not just a test artifact).  Fix — **punch-hole hybrid**:
  `seek_forward` now `fallocate(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE)`s the
  skipped region when the file was preallocated, deallocating those blocks back
  to a hole.  Keeps preallocate's extent-stall-free dense writes AND restores
  sparseness.  `write_sparse` now coalesces consecutive zero blocks into one
  skip, so it's one punch per zero run, not one per 4 KiB.  Best-effort:
  filesystems without punch support degrade to non-sparse (never incorrect).
  Verified (forced `--direct` on a Gen3 box): 64 MiB zeros decompress → 16 blocks
  sparse vs 131080 dense, both byte-correct; random-data `--direct` round-trip
  unaffected.

- **`[ASYMMETRIC]` log tag collision.**  The v0.13.25 `--direct` auto-default
  reused the `[ASYMMETRIC]` tag and (correctly) runs before the
  `backend_user_set` return, so on Gen4+ it logged `[ASYMMETRIC] … --direct` even
  with an explicit `--cpu-only`/`--hybrid` — tripping the asymmetric tests that
  assert explicit backends silence `[ASYMMETRIC]`.  Retagged that line
  `[O_DIRECT]` (it's an I/O decision, not backend selection).  No behavior change.

259/259 tests pass.  Validate the punch-hole + `[O_DIRECT]` retag on the Gen4+
box where the failures originally appeared.

---

## v0.13.30 — Review cleanups: --sync-output under --direct, unaligned load, dead code

Three small fixes from the Phase 7 review (ROADMAP 7.5–7.7):

- **7.5 — `--sync-output` was a no-op under `--direct`.** When the O_DIRECT writer
  owns the output, the `FILE* out` is closed and nulled, so the `fsync_file(out)`
  path in `main` never ran — `--direct --sync-output` returned without ever
  flushing.  Now `main` fsyncs the DirectWriter's own fd (device write cache +
  the size metadata from finalize's ftruncate) when `sync_output` is set.
  Confirmed via strace: `--direct --sync-output` issues one fsync, `--direct`
  alone issues none.

- **7.6 — `is_all_zero` did an unaligned `size_t` load.**
  `reinterpret_cast<const size_t*>(p)` on a `vector<char>::data()` pointer is UB
  on strict-alignment targets.  Replaced with a constant-size `memcpy` into a
  `size_t` (same wide load on x86, portable elsewhere).

- **7.7 — removed the dead `SequentialDispatcher` class.**  Superseded by the
  per-GPU result slots (v0.11.11); no callers remained (verified — the type and
  its methods appeared only in its own definition).  ~46 lines of concurrency
  surface gone.

259/259 tests pass; sparse and `--direct --sync-output` round-trips verified.

---

## v0.13.29 — Bound the decompress reader: queue-depth backpressure (slow-consumer RSS)

The FrameThrottle bounds *popped-but-unwritten* frames, but nothing bounded the
TaskQueue *ahead* of the workers.  On decompress, `stream_frames_to_queue` reads
each compressed frame into a fresh `Task.data` vector and pushes it; when the
consumer is slower than the reader — classically `--gpu-only` decompress, which
is D2H-bound — the reader races ahead and buffers the *entire* compressed input
in RAM.  The v0.13.24 Gen4 isolation caught this: gpu-only `-d` of a 9.75 GiB
input held 11.3 GiB RSS (vs ~1.9 GiB for cpu-only/hybrid), and it's a latent OOM
on inputs larger than RAM.  (ROADMAP 7.8; the original "hybrid excess faults"
hypothesis there was disproven — that was the buffered-write storm.)

`TaskQueue` gains an optional `max_depth_` (0 = unbounded, the default): `push()`
blocks on a new `space_cv_` once the queue is full, and the pop paths a bounded
queue uses (`try_pop_one`, `pop_batch_greedy`, `try_pop_one_cpu`) plus `set_done()`
wake it.  `re_enqueue` (push_front) bypasses the cap so it never blocks.  Both
decompress paths set the cap to a pipeline-depth multiple
(`max(THROTTLE_MIN_FRAMES, parallelism * slack)`), so queued RAM is O(pipeline),
not O(input) — skipped under `--no-throttle`.  The cap is deliberately ≥ the
auto-tuner's batch needs, so it bounds RAM without constraining GPU batch growth
(no throughput risk).  Compress queues are never bounded; with mmap input they're
zero-copy views anyway.

Verified: cpu-only/gpu-only/hybrid decompress round-trip with no deadlock (the
producer blocks and resumes correctly); a slow-consumer gpu-only `-d` of a 3 GB /
~2861-frame incompressible input holds 1.79 GiB RSS (queue capped, not the whole
input buffered); 259/259 tests pass.

---

## v0.13.28 — Size the compress throttle from the resolved chunk, not opt.chunk_mib

`compress_cpu_mt` built its `FrameThrottle` from `opt.chunk_mib`, but the frame
size actually used is `host_chunk` (= the resolved `chosen_mib`), which can be
auto-bumped for `--ultra` (e.g. level 22 forces a 128 MiB chunk) or shrunk by
`check_ram_budget`.  `compute_throttle_budget` divides `avail/2` by the frame
size for its RAM-cap term, so a stale 16 MiB there mis-sizes the in-flight cap:
on `--ultra` it under-counts in-flight RAM (thinks frames are 16 MiB when they're
128 MiB — 8× under), which on a RAM-constrained box could admit more frames than
memory holds.  Pass the resolved `host_chunk` instead.

Default (non-ultra) runs are unchanged — `opt.chunk_mib == chosen_mib == 16`
there.  Demonstrated at `-v`: a `--ultra -22 -T4` run now reports
`[THROTTLE] … 4.00 GiB in-flight max` (32 × 128 MiB) instead of the old
512 MiB (32 × 16 MiB).  The GPU compress path already used `chosen_mib`; the two
decompress paths keep `opt.chunk_mib` as a heuristic (the true decompressed frame
size isn't known until the stream is parsed, after the throttle is built).
259/259 tests pass.  ROADMAP Phase 7.3.

---

## v0.13.27 — Accept bundled short flags (`-dc`, `-dkf`, …) for zstd/gzip compat

`parse_args` exact-matched each argv token, so a bundled short-flag group like
`-dc` was rejected with `unknown option: -dc` — even though `zstd -dc` and
`gzip -dc` both accept it, and gzstd bills itself as a drop-in zstd replacement.
This bit common idioms carried over from zstd/gzip (`gzstd -dc archive | tar -xf -`,
`-dk`, `-df`, `-dcf`).

Fix: a pre-pass at the top of `parse_args` expands a bundled group into
individual flags *before* the match loop, so the loop and all its value-flag
(`argv[++i]`) handling are untouched.  A group is expanded only when every
character after a single leading `-` is a no-arg operation flag — `{d,t,k,f,c}`.
Anything else passes through unchanged: value flags (`-o`/`-T`/`-M`/`-B`/`-D`),
numeric levels (`-19`), attached-value short flags (`-T4`, `-M512`, `-b3`), the
repeat flags (`-vv`/`-vvv`/`-qq`), long options, and `--`/`-`.  `v`/`q` are
deliberately excluded so their repeat semantics survive — bundle verbosity flags
separately (`-d -vv`, not `-dvv`).  Unknown bundles (e.g. `-dz`) still error as
before.

Verified: `-dc`/`-dcf` decompress to stdout and round-trip; `-vv` still maps to
debug, `-19`/`-T4`/`-M512` parse, `-dz` rejected; 253/253 tests pass.  ROADMAP
Phase 7.9.

---

## v0.13.26 — Extend the Gen4+ `--direct` default to compress (was decompress-only)

v0.13.25 auto-enabled `--direct` for decompress on Gen4+; this extends the same
gate to **compress**.  The knuth (Gen4) `--direct` data shows compress benefits
the same way decompress does — the win scales with output volume and is
backend-independent: cpu-only low_compress +103% / mixed +50% / medium +15%,
gpu-only +71% / +29% / +12%, hybrid +70% / +24% / +21%; tiny-output profiles
(high, zeros) are neutral.  No measured regression on Gen4 for any backend.

The decompress-only scoping in v0.13.25 existed solely to avoid the Gen<4
compress regression (a low-core box never saturates buffered writeback, so
O_DIRECT just adds alignment overhead) — but the `gen >= 4` gate already excludes
that case, so the extra `mode == DECOMPRESS` guard was redundant.

`apply_backend_defaults()` now enables `--direct` on Gen4+ for both compress and
decompress (test mode skipped — it writes nothing), unless the user passed
`--direct`/`--no-direct`.  The PCIe-gen probe moved above the compress branch so
it runs for all modes.  Verbose line generalized to
`[ASYMMETRIC] PCIe Gen4 detected; defaulting output to --direct`.

Caveats unchanged from v0.13.25: compress output size is unknown, so the O_DIRECT
path preallocates `input_size` as an upper bound and `ftruncate`s down at
finalize (already handled; `--no-preallocate` opts out).  O_DIRECT can raise
tail-latency variance (NVMe GC / journal commits); medians favor it on Gen4.
`--no-direct` forces the buffered baseline.

Verified: Gen3 compress stays buffered (no auto-enable), explicit
`--direct`/`--no-direct` honored, round-trip clean, 253/253 tests pass.  ROADMAP
Phase 5.3.

---

## v0.13.25 — Default `--direct` (O_DIRECT output) for decompress on PCIe Gen4+

Decompress on the Gen4 reference box is ~95% write-bound: with the write path
removed (`-c >/dev/null`) cpu-only decompress hits ~14 GiB/s, but buffered output
to disk runs at ~0.68 GiB/s — page-cache population + writeback throttling is the
whole cost.  O_DIRECT output bypasses that, and on fast-fabric / high-core boxes
where frame production outruns buffered writeback it is a large decompress win
(up to +130–230% on the Gen4 reference; mixed `-d` ~0.68 → ~2.0 GiB/s).  On
smaller Gen<4 boxes O_DIRECT regresses — the producer never saturates buffered
writeback, so it only adds alignment overhead — so they stay buffered.

`apply_backend_defaults()` now auto-enables `--direct` for **decompress** on PCIe
Gen4+ (reusing the `detect_min_pcie_gen()` probe that already drives the
cpu-only/hybrid decompress default), unless the user passed `--direct` or
`--no-direct`.  It is backend-independent — the win is in the output write path —
so it applies to cpu-only, hybrid, and gpu-only decompress alike.  Compress never
auto-enables it (O_DIRECT regresses compress on smaller boxes; strictly opt-in).
Test mode writes nothing, so it is skipped.  Visible at `-v` as
`[ASYMMETRIC] PCIe Gen4 detected; defaulting decompress output to --direct`.

Behavior notes:
- Override with `--no-direct` (e.g. to benchmark the buffered baseline on Gen4).
- The standard benchmark's plain decompress runs on a Gen4 box now use O_DIRECT
  by default; pass `--no-direct` for the buffered comparison.
- Gen<4 and detection-unavailable paths are unchanged (buffered).

Verified: Gen3 stays buffered (no auto-enable), explicit `--direct`/`--no-direct`
honored, round-trip clean, 253/253 tests pass.  ROADMAP Phase 5.3.

---

## v0.13.24 — Recycle GPU decompress output buffers (kill the per-frame D2H alloc churn)

The CPU workers recycle a bounded `FrameBuf` pool (the v0.13.7/v0.13.8 page-fault
storm fix), but the GPU decompress completion path never got the same treatment:
every readback frame did a fresh `make_shared<vector<char>>(actual)` — a full
decompressed-frame (~16 MiB) allocation that faults every page, on every frame
of every batch.  On the Gen4+ hybrid-decompress default this is the hot path and
the allocation cycles fast.

`DecompStreamCtx` now owns a recycled `out_pool`: `acquire_out_buf()` reuses a
slot whose `use_count()==1` (writer has drained it), grows the pool lazily up to
two batches' worth, and past that waits on the writer's drain signal rather than
allocating.  Recycled slots keep their resident pages, so after warm-up the path
stops faulting.  Deadlock-free by the same FIFO argument as the throttle: a
stream pushes frames in ascending seq, so the writer always has the oldest
in-flight frame to write and frees a slot when it drops the ref.

Measured on a consumer Gen3 box (2 GiB→2 GiB mixed, `--gpu-only -d`, isolating
the path): minor page-faults 636k → 538k (−15%), peak RSS 2.57 → 2.26 GiB
(−12%).  The Gen3 default is `--cpu-only` decompress so this only shows under a
forced GPU/hybrid run; the real target is the Gen4+ hybrid-decompress default,
where batches are larger and frames cycle faster — validate there.  Round-trip
verified on `--gpu-only` and `--hybrid`; 253/253 tests pass.

Compress-side GPU output buffers (`compress_nvcomp`) still allocate per frame but
hold only the *compressed* output (small), so the fault pressure is far lower;
left as a follow-up.  ROADMAP Phase 7.2.

---

## v0.13.23 — Fix: AsyncWritePool flush() waits for the physical write, not just the dequeue

Correctness fix found in a full-pipeline review.  `AsyncWritePool::flush()` waited
only on `pending_.empty()`, but the background write thread empties `pending_` by
*moving* the batch out before it writes it to disk.  So `flush()` could return
while the last batch was still in flight.  A write error on that final batch
(disk full, EIO, broken O_DIRECT tail) sets `error_` only *after* the single
`had_error()` check in `writer_thread`, so the run reported success (exit 0) and
the atomic `rename` proceeded over truncated/corrupt output.  Mid-stream errors
were already caught one batch late by the `had_error()` check inside `submit()`;
only the final batch escaped — i.e. precisely the disk-full-at-the-end case.

Fix: a `writing_` flag (guarded by the pool mutex) is set true when the worker
dequeues a batch and cleared once the batch is physically written — including on
the error-return path, which now also notifies so a blocked `flush()` wakes.
`flush()` now waits on `pending_.empty() && !writing_`, making the post-`flush()`
`had_error()` check reliable.  No hot-path change: the flag is touched twice per
batch under a mutex that was already taken on those transitions.

Roadmap Phase 7.1.  The remaining review items (GPU result-buffer pooling on the
Gen4 hybrid-decompress path, throttle-budget chunk size, CPU-compress memcpy,
and minor nits) are tracked in ROADMAP.md Phase 7.

---

## v0.13.22 — Revert v0.13.20's kernel gate; add `--cold` for honest benchmarking

Plain mmap is restored as the default for compress on every kernel.  v0.13.20
auto-switched to fread on pre-6.4 kernels (no per-VMA locks), but a follow-up
benchmark exposed the gate as a net regression: cpu-only `zeros` and
`high_compress` were **2.6-3× slower** than the prior mmap path, and other
configs were no better than the noise floor (hybrid `mixed` "won" by ~1%, well
inside benchmark jitter).

**Mechanism (why mmap beats fread even on a pre-6.4 high-core box).**  For
I/O-bound workloads the compressor finishes a chunk in microseconds and asks
for the next.  mmap lets all 256 worker threads fault their own pages directly
into their address space and read from the page cache *in parallel* — zero
copy, aggregate bandwidth scales with cores.  fread instead does a *single
producer thread* `fread` + `memcpy` into a `Task.data` buffer; that one thread
caps the whole pipeline at single-thread memcpy bandwidth (~4-5 GiB/s),
regardless of how many workers are downstream.  The `mmap_lock` cacheline
storm is real (28% `down_read_trylock` in the v0.13.20 profile) but doesn't
move wall time outside the run-to-run noise the kernel-gate ostensibly fixed.

**Removed:** `kernel_has_per_vma_locks()`, the gate in `apply_backend_defaults`,
the `mmap_user_set` flag, and `#include <sys/utsname.h>`.  `--mmap`/`--no-mmap`
still work as explicit overrides; default is mmap everywhere.

**Kept (added in v0.13.21, folded into this entry):** a `--cold` flag that
calls `posix_fadvise(POSIX_FADV_DONTNEED)` on the input fd right after open.
Without it, `gzstd-benchmark.sh`'s median-of-3 against a 20 GiB file on a
600+ GiB-RAM box was measuring memory-to-memory throughput — iteration 1
warmed the cache and iterations 2-3 served from RAM.  `--cold` makes every
iteration a real cold-disk read as an ordinary user (no root, no
`drop_caches`).  Documented in `--help` as benchmarking-only.
`gzstd-benchmark.sh` now passes `--cold` for both compress and decompress
invocations and no longer writes to `/proc/sys/vm/drop_caches` (system-wide
cache wipe under sudo, bad citizen on a shared host).

---

## v0.13.20 — Kernel-gated mmap/fread for compress (fixes the high-core mmap storm)

> **Reverted in v0.13.22 — net regression.**  Cold-cache benchmarks on the
> 256-core box (5.15 kernel) showed cpu-only `zeros`/`high_compress` 2.6-3×
> slower than plain mmap; the "fix" only helped warm-cache `mixed` by ~1%,
> within noise.  fread serializes the read on a single producer thread, which
> caps aggregate throughput at one core's memcpy bandwidth — a worse problem
> than the `mmap_lock` cacheline contention the gate was trying to avoid.
> See v0.13.22.

Resolve the high-core mmap compress slowdown for real, and **revert the failed
v0.13.17 prefault** (which made it worse). The root cause turned out to be the
kernel, not gzstd.

**Investigation.** On a 256-core box, mmap compress burned ~94s system time
(28% of perf samples in `down_read_trylock`) vs ~16s for `--no-mmap`. A toggle
matrix (v0.13.18, `GZSTD_MMAP_ADVISE` × `GZSTD_MMAP_PREFAULT`) showed the
v0.13.17 producer-side `MADV_POPULATE_READ` prefault made it *worse* in every
advise mode (~88-90s sys regardless), because the populated pages were reclaimed
before the workers reached them (minflt barely changed) — pure added faulting
work, not a fix. A binary-search sweep then showed `fread` beats mmap at *every*
thread count on that box (7.4 vs 7.8s at T=2, up to 7.8 vs 11s at T=256) — so
there's no thread-count crossover at all.

**Root cause.** Pre-6.4 kernels have no per-VMA locks, so every page fault takes
the single global `mm->mmap_lock` rwsem; many cores doing atomic ops on that one
counter cacheline dominates system time. Linux **6.4** (per-VMA locks, faults via
RCU + a fine-grained per-VMA lock) removes it. The slow box was on 5.15; the
workstation is already on a 6.x kernel, which is why mmap was always fine there.

**Fix.** `apply_backend_defaults` now checks the kernel (`uname`, parsed once):
on **< 6.4**, compress falls back to `fread`; on **6.4+**, it keeps mmap (the
faster path). Skipped for `--mmap`/`--no-mmap` (explicit override via a new
`mmap_user_set` flag), for `--gpu-only` (only a few H2D faulters, not the worker
storm), and for decompress (never used the mmap reader). It's a clean per-kernel
switch — no thread-count threshold, since fread wins at all thread counts on
pre-6.4. Distro backports (e.g. RHEL per-VMA locks on a `5.14.x-*.el9` string)
get a harmless false negative (fread ≈ mmap there); `--mmap` overrides.

Removed: the v0.13.17 `mmap_prefault` and the v0.13.18 `GZSTD_MMAP_*` diagnostic
toggles. `MmapRegion` is back to plain `MADV_SEQUENTIAL`. The slow box upgrades
to a 6.x kernel later this year, after which mmap will be the default there too.

---

## v0.13.17 — Pre-fault mmap input on the producer (kill the fault storm)

> **Reverted in v0.13.20 — this did NOT work.** The producer-side
> `MADV_POPULATE_READ` populated pages that were reclaimed before the workers
> used them, so it *added* system time instead of removing the storm. The real
> cause was a pre-6.4-kernel `mmap_lock` limitation; see v0.13.20.

Fix mmap compression being *slower* than `--no-mmap` on high-core machines,
despite being the "zero-copy" default.

**Cause.**  The compress producer hands workers `Task`s whose `view_ptr`
points into the mmap'd input; each worker faults its own pages in on first
touch.  With hundreds of workers hammering one mapping, concurrent faulting
storms the kernel's `mmap_lock` and per-page fault path.  On a 256-core
machine this showed as ~4× system time (~66s vs ~16s) and ~15% slower wall
than `--no-mmap` — the same `mmap_lock` storm already designed out of the
*decompress* path (which reads via `fread`).  Compress still defaulted to
mmap, so it ate the penalty.

**Fix.**  The producer now bulk-pre-faults each chunk with
`MADV_POPULATE_READ` *before* pushing its task (`mmap_prefault`, new helper
next to `MmapRegion`).  The faulting happens once, in bulk, on the single
producer thread — no concurrent storm — and because population precedes the
push, a worker can never touch an unpopulated page (no startup race, no need
for a separate prefetch thread).  Zero-copy reads and read/compress overlap
are preserved.  The producer paces itself to stay ≤ ~1 GiB ahead of
consumption (`m->read_bytes`, which workers bump per chunk), so the whole
file is never read up front.  Applied to both `compress_cpu_mt` and
`compress_nvcomp`.

**Portability.**  `MADV_POPULATE_READ` (Linux 5.14+) is `#define`d to its
stable UAPI value (22) when missing from build headers (the ubuntu-20.04
portable-build container, glibc 2.31), so the shipped binary still uses it on
a 5.14+ runtime kernel; on older kernels `madvise` returns `EINVAL` and we
fall back to lazy per-access faulting.

**Verified (24-core workstation):** 4 GiB compress round-trip byte-identical;
mmap+prefault now ~1.7s vs `--no-mmap` ~2.3s with low sys time (no storm);
full suite passes.  The decisive win is expected on a 256-core machine where
the storm was severe — re-measure `--direct` vs `--direct --no-mmap` there:
the goal is mmap matching or beating `--no-mmap` with the ~4× sys gap gone.

---

## v0.13.16 — Stream large single-frame files directly from the file

Fix a long-standing slow path: decompressing a single-frame `.zst`
(stock `zstd`, `--sliding-window`) was far slower than `zstd -d`,
spiked memory, and showed a frozen progress bar — in every mode,
including `--cpu-only`.

**Cause.**  A single zstd frame can't be split across CPU threads (nor
GPU subchunks).  The fallback routed the lone frame through the normal
queue: `stream_frames_to_queue` had to read and buffer the *entire*
compressed frame (growing its read buffer with realloc churn), `memcpy`
it into one Task, and only then could a CPU worker decompress it — and
even then the worker's streaming branch waits on the `producer_done`
gate (a v0.13.1 seq-collision guard) which can't fire until the reader
has consumed the whole file.  Net: read and decompress ran *serially*,
peak memory was input + frame-copy + output (~30 GiB on a 20 GiB file),
and neither meter moved until decompression started.

**Fix.**  In `main`'s decompress dispatch, peek the first frame; if it
decompresses to more than `SINGLE_FRAME_STREAM_MIN` (256 MiB) the input
is effectively a single-frame file, so hand it to a new
`decompress_stream_from_file` regardless of mode: a plain
`ZSTD_decompressStream` loop reading 4 MiB at a time straight from the
`FILE*`, writing each output chunk through the existing DirectWriter /
fwrite path.  Read, decompress, and write now overlap; peak RSS drops to
a couple of I/O buffers; the progress bar moves (we set `total_out` /
`total_out_final` from the peeked size up front); and for GPU modes no
CUDA is touched (no bringup thread, no cuInit).  Single-threaded by
nature — one zstd frame can't be split — which is inherent, not a
regression.

**Why a 256 MiB threshold, not 16 MiB.**  The threshold sits well above
gzstd's largest practical chunk (`--ultra` auto-bumps to 128 MiB) and the
v0.13.1 regression test's 100 MiB chunks, so genuinely *multi-frame*
inputs — even with large per-frame sizes — keep the parallel queue path.
`decompress_nvcomp` therefore retains its `gpu_disabled_by_peek` CPU
fallback for the 16–256 MiB multi-frame-oversize case (GPU can't subchunk
those frames, but the CPU pool still decompresses them in parallel).
Streaming a multi-frame file would needlessly serialise it.

Applies to all modes and both build configs (`decompress_stream_from_file`
is not GPU-gated).  Seekable input only — stdin (peek returns -1) keeps
the old path.  `ZSTD_decompressStream` also decodes the rare
trailing-frames-after-a-large-first-frame case correctly.

**Verified (24-core workstation):** 2 GiB single-frame `--sliding-window` round-trip
byte-identical via both `--cpu-only` and `--gpu-only` (peak RSS 24 MB, was
~2-3 GiB; no GPU bringup logged); a 4×100 MiB multi-frame file correctly
stays on the parallel queue path; full suite passes including the
`--sliding-window` round-trip and the v0.13.1 multi-frame-oversized guard.

---

## v0.13.15 — Overlap CUDA init with the reader (gpu-only decompress)

Extend the v0.13.13 bringup overlap to `--gpu-only` decompress, killing the
3-4s startup stall (the gap between `[O_DIRECT]` and `[INIT]` at `-v`) on
high-GPU-count boxes.

**Cause.**  v0.13.13 deferred the `cudaGetDeviceCount` cuInit (~2-3s on an
8-GPU box) to a background bringup thread, but *only* for adaptive hybrid —
the rationale was "no CPU pool to overlap with" in gpu-only.  That overlooked
the **reader**: `stream_frames_to_queue` reads and frame-parses the entire
compressed input, which on a multi-GiB file takes about as long as cuInit and
has to happen regardless.  In gpu-only the reader ran *after* the synchronous
cuInit + inline bringup, so the GPU sat idle through init and the reader sat
idle through nothing useful — pure serial cost.

**Fix.**  Generalize the deferral predicate from `hybrid_overlap` to
`defer_detect = hybrid_overlap || opt.gpu_only`.  In gpu-only the bringup
thread now does the deferred `cudaGetDeviceCount` + `select_best_gpus` +
worker spawn while the main thread goes straight to the reader, filling the
`TaskQueue`.  GPU workers consume a warm queue the instant their contexts are
ready instead of starting cold.  The `[INIT]` banner reports "GPUs detecting
in background" for gpu-only too.

**gpu-only edge cases** (handled synchronously before, now need explicit care
because detection is deferred):

- *No CUDA device.*  The synchronous path errored instantly at
  `cudaGetDeviceCount`.  Deferred, the bringup thread sets a
  `gpu_only_no_device` atomic; `stream_frames_to_queue` takes a new optional
  `abort` pointer and returns early when it's set, so main errors with the
  same `EXIT_USAGE` message instead of buffering a consumer-less queue to EOF.
- *Oversize first frame* (`--sliding-window` / `zstd` single frame).  The
  peek sets `gpu_disabled_by_peek`; the CPU-pool spawn condition now also
  fires on that flag (gpu-only has no `sched`-driven pool), so the file
  decompresses on CPU.  The deferred bringup still pays a (hidden, discarded)
  cuInit on the background thread for this case — acceptable for a rare path.

**Verified (2-GPU workstation):** 253/253 suite; gpu-only round-trip
byte-identical; masked-GPU run errors cleanly with exit 2 and removes the
partial output; sliding-window file falls back to CPU and round-trips.  The
win scales with GPU count, so the 8-GPU machine's gap should drop from ~3-4s
to ~0 — re-measure the `[O_DIRECT]`→`[INIT]` interval there.

---

## v0.13.14 — Fixed-share: wait for GPU registration before streaming

Fix a `--cpu-share` regression (introduced by the v0.13.11 device-probe
short-circuit) where the requested CPU/GPU split collapsed to all-CPU
on high-GPU-count machines.

**Cause.**  Before v0.13.11, `select_best_gpus` did a serial CUDA probe
that pre-created GPU contexts, so GPU workers registered almost
instantly.  v0.13.11 removed that probe; `warm_gpu_contexts` was meant
to compensate but only creates the CUDA *contexts* — the GPU worker
still does VRAM probe + cudaMalloc + `register_gpu_stream` afterward.
On an 8-GPU box with many fast CPU workers, the reader + CPU pool drain
a small input via the drain-phase fast path (`qs.done &&
!any_gpu_active()`) before any GPU registers, so `--cpu-share 0.0`
(all-GPU) produced 128c/0g — every frame went to CPU.  Surfaced by the
suite's `--cpu-share split responds to value` test on the 8-GPU
system; the 512 MiB test input gave enough lead time on 2-GPU hardware
but not on 8.

**Fix.**  In fixed-share mode only, wait for at least one GPU stream to
register (`any_gpu_active()`) — or for all GPUs to fail init
(`gpu_init_failures >= gpu_count`) — before starting the reader.  This
guarantees the drain-phase fast path can't fire before the GPU is in
the rotation, so the split is honored.  Applied to both compress and
decompress.  Adaptive mode (the default) skips the barrier: it promises
no exact split and wants the fastest possible start.

**Test suite changes:**
- New section "Hybrid GPU-bringup overlap (decompress)" guarding the
  v0.13.13 restructure: adaptive round-trip, fixed-share round-trip,
  stdout output, and a repeated-run teardown-stability check.
- Added a `--extensive` flag.  Lower-value / cosmetic sections are now
  gated behind it so the default run is leaner (253 tests vs 284 with
  `--extensive`): Stress tests, Help/version, Space-separated option
  values, and Completion summary format.  GPU correctness and
  regression-guard sections stay in the default run.  Gate further
  groups with `if $EXTENSIVE; then ... fi`.

---

## v0.13.13 — Overlap CUDA init with CPU decompression (hybrid)

Eliminate the startup stall before the progress meter moves in hybrid
decompress.  v0.13.12's timing showed the residual delay was almost
entirely `cudaSetDevice` context creation; this release also addresses
the *other* half — `cuInit`.

**Cause.**  `cudaGetDeviceCount` (decompress_nvcomp, the first CUDA call
in the process) triggers the one-time CUDA driver init `cuInit` —
~2s on an 8-GPU box.  It ran on the main thread *before* the CPU
decompression pool was spawned, so in hybrid mode the CPUs couldn't
start decompressing until cuInit finished.  The user's exact
observation: "the CPUs should be decompressing while the GPU detection
is going on" — and they were blocked from doing so.

**Fix.**  In hybrid mode, GPU detection + selection + worker spawn now
run on a background "bringup" thread:

- Main thread spawns the CPU pool and starts the frame reader
  immediately, using a *provisional* device count for throttle sizing
  (RAM-capped, so over-estimating is safe).
- CPU workers decompress from t≈0 — the scheduler already runs CPUs
  "wild" while `gpu_ready_` is false (no new scheduling logic needed).
- The bringup thread does `cudaGetDeviceCount` (the deferred cuInit),
  `select_best_gpus`, `init_slots`, and spawns GPU workers.  They
  register with the scheduler when ready and it rebalances.

**Concurrency safeguards:**
- `init_slots` (resizes `ResultStore::slots`) is called under
  `results.m`, which the writer's `drain_slots_locked` also holds — no
  resize/iterate race.
- Teardown joins the bringup thread before iterating `gpu_workers`
  (the bringup thread populates that vector).
- If the CPU pool drains the whole file during cuInit, the bringup
  thread skips GPU spawn entirely (no wasted context creation; process
  exits as soon as CPU+writer finish).  Late-spawned GPU workers that
  hit a done+empty queue exit cleanly via the existing
  `producer_done_seen` path.

gpu-only and cpu-only paths are unchanged (gpu-only has no CPU pool to
overlap with; detection stays inline).

**Result** (consumer Gen3, hybrid decompress): TTFB dropped from
0.256s (v0.13.12) to 0.035s.  The CPU pool starts before cuInit
instead of after it.  On high-GPU-count systems where cuInit is ~2s,
the win is correspondingly larger.  Tiny-file edge cases (CPU finishes
before GPU init) verified for correct output and clean teardown.
280/280 tests pass.

---

## v0.13.12 — Per-phase GPU init timing at -vv

Diagnostic only, no behavior change.  v0.13.11 removed the 5s serial
device-selection probe; a residual GPU-init delay remained (~4s on an
8-GPU box, ~2.5s hybrid).  To locate it, the GPU compress worker now
logs an init-phase breakdown at -vv:

  [GPU<d>] init phases: ctx=Nms probe=Nms malloc=Nms total=Nms

- **ctx**: `cudaSetDevice` forcing CUDA primary-context creation.
- **probe**: per-stream VRAM binary search (nvCOMP temp-size queries).
- **malloc**: `allocate_stream_buffers` (`cudaMalloc` of device buffers).

Measured on consumer Gen3 (2× 2080 Ti): ctx≈230ms, probe 1-11ms,
malloc≈1ms — context creation is ~99% of init.  The VRAM probe and
device allocation are negligible, so they're not worth optimizing.
The remaining startup cost is CUDA context creation, which on
multi-GPU systems appears to serialize on the driver's init lock
(8 × ~500ms ≈ the observed 4s).  This release just makes that
visible; reducing it (parallel context creation, or overlapping it
behind useful work) is follow-up.

---

## v0.13.11 — Skip serial GPU probe when using all devices

Fix a ~5s startup stall before the progress meter moves in `--gpu-only`
and `--hybrid` modes (absent in `--cpu-only`).

**Cause.**  `select_best_gpus()` ranks GPUs by free VRAM / utilization
so it can pick the best N when the user wants a subset.  The subset
path uses NVML (no CUDA context creation — fast).  But when using ALL
devices (the default), the NVML guard `if (want < total_devices)` is
false and the function fell through to an all-devices loop that calls
`cudaSetDevice(d)` + `cudaMemGetInfo()` on every device.  Those force
serial CUDA context creation on the main thread — ~0.6-1s per
datacenter GPU, ~5s for 8 — and this runs before the reader, progress
thread, and worker pool start.  The pipeline waits the whole time.

The probe was pointless in this case: when N == all devices there's
nothing to rank.  We paid 5s gathering ranking data we then ignored.

**Fix.**  Short-circuit when `want >= total_devices`: return the
trivial `[0..N)` device list without probing.  The GPU worker threads
create their CUDA contexts in parallel at startup (one per device on
its own thread) instead of serially on the main thread.  In hybrid
mode the reader and CPU pool also start immediately and overlap with
GPU context warm-up.  Expected: ~5s → ~1s (one parallel context init)
in gpu-only, near-zero perceived delay in hybrid.

**Fixed-share exception.**  Deferring context creation to the worker
threads broke `--cpu-share`: on a small input the CPU pool drains every
frame (via the `qs.done && !any_gpu_active()` path) before the GPU
finishes booting and registers, so the explicit split was silently
ignored — `--cpu-share 0.0` gave 100% CPU.  Fix: when `--cpu-share` is
set, `warm_gpu_contexts()` creates the primary contexts in parallel
(one thread per device, ~1s for 8 vs ~5s serial) before the pipeline
starts, so the GPU is ready to take its share.  Adaptive mode (the
default) still defers for fastest startup — there the GPU naturally
catches up on any non-trivial input, and a small input being
CPU-drained is the correct fast path.

**Telemetry.**  `[GPU] device selection: N ms` logged at `-v`, so this
startup cost is visible going forward.

**On the subset case** (`--gpu-devices N` with N < total): already
fast — it uses NVML to read utilization and free memory without
creating CUDA contexts (`cudaGetDeviceProperties` reads cached device
attributes, no context).  The only remaining slow path is NVML being
unavailable AND selecting a subset, where free-memory ranking requires
`cudaMemGetInfo` (hence a context).  That's unavoidable without NVML
and rare on NVIDIA systems.

---

## v0.13.10 — Condition-variable wait for the bounded pool

v0.13.9's bounded pool architecture is correct, but its acquire-when-
full path used `std::this_thread::yield()` to wait — which is a
`sched_yield` syscall on Linux.  With 96 workers each yielding hundreds
of thousands of times per run, sys time on cpu-only decompress mixed
jumped from 11.87s (v0.13.8) to 51.41s (v0.13.9): same throughput, 4×
more kernel cycles burned in `sched_yield`.

**Fix.**  Wait on a condition variable instead.  Added `drain_cv_` +
`drain_m_` to `FrameThrottle` with two methods:

- `notify_drain()` — called by `AsyncWritePool::worker_fn` after each
  `buf.reset()` (the moment a frame's `shared_ptr` ref drops from 2 to
  1, freeing a worker pool slot).  `notify_all()` because the writer
  doesn't know which worker owns the freed slot.
- `wait_for_drain(predicate)` — workers call this when their pool is
  full.  Standard CV `wait_for` with predicate, 10ms timeout as a
  safety net for any missed notify.

Both `cpu_worker` and `cpu_decomp_worker` now use `wait_for_drain`
instead of `yield`.  Predicate scans the per-worker pool for a slot
with `use_count() == 1`.

**Why a separate CV from the existing permit-acquire CV (`cv_`):**
sharing would force pool-waiters and permit-waiters onto the same
mutex, blocking `release()` while broadcasting.  `drain_m_` is
dedicated and never held by `release()`, so permit-acquire stays fast.

**Notify granularity.**  Per-frame `notify_all`, not per-batch.
Trades more wakeups for lower latency: workers wake immediately when
their slot frees rather than waiting for the writer's whole batch.
Wake cost is bounded — only workers currently in `wait_for_drain` are
woken, and the predicate check is a few atomic loads.

---

## v0.13.9 — Bounded per-worker buffer pool: route page-faults through backpressure

v0.13.8 introduced a per-worker output-buffer pool to eliminate the
per-iteration allocation storm.  Profiling on a 256-core / 8-GPU
system showed it didn't actually fix decompress at high thread counts
— the pool was UNBOUNDED and grew faster than the writer could drain.
This release makes the pool bounded so it participates in the existing
backpressure chain instead of bypassing it.

**The diagnosis (perf record, cpu-only decompress, zeros.bin):**

| Metric (T96 vs T16) | T96 | T16 |
|---|---|---|
| Wall time | 4.14s | 3.27s |
| Sys time | 29.2s | 3.11s |
| Sys/real ratio | **9.4×** | 1.0× |
| Page faults | 2.35M | 510k |
| IPC | 0.29 | 1.00 |

Hot path at T96: 82% of cycles in `std::vector::resize` → memset →
`asm_exc_page_fault` → `down_read_trylock` (the per-process mmap_lock
rwsem).  Same shape as the v0.13.7 compress diagnosis.

Hot path at T16: 68% in `AsyncWritePool::write_sparse` → `fseek` →
`lseek` syscall.  Writer was the bottleneck; 16 workers were enough.

**Root cause: the v0.13.8 pool bypassed the throttle's backpressure.**
The FrameThrottle bounds total in-flight frames (default 512).  With
96 workers, that's ~5 frames/worker on average.  But v0.13.8's
`acquire_decomp_buf()` grew the pool whenever `use_count() > 1` on all
existing slots — and since the writer was the bottleneck (~5 GiB/s
ceiling on sparse zeros), slots stayed in flight long enough that
workers grew their pools to 5+ entries.  Each new entry was a fresh
~64 MiB allocation → page-fault storm.

**Fix: pool is now bounded at startup and yields on full.**

```cpp
const int pool_size = std::max(2, throttle_budget / N_workers);
std::vector<FrameBuf> pool(pool_size);
for (auto & b : pool) b = std::make_shared<std::vector<char>>();

auto acquire = [&]() -> FrameBuf {
  while (true) {
    for (auto & b : pool) if (b.use_count() == 1) return b;
    std::this_thread::yield();  // backpressure: wait for writer
  }
};
```

The min-of-2 guarantees pipelining (one frame in flight + one being
worked on).  Above that, the throttle's global cap is divided across
workers.  This makes the chain work end-to-end:

```
writer slow → result store fills → pool slots stay in-flight
            → pool acquire yields → worker waits → no new alloc
            → frame production rate = writer drain rate

writer fast → result store drains → slots free fast
            → acquire returns immediately → worker proceeds full speed
            → throttle is the only cap (intended design)
```

**No thread-count cap.**  An arbitrary "decompress shouldn't exceed N
workers" rule was considered and rejected — it sidesteps the
architectural issue without fixing it, and gets the wrong answer on
hardware we haven't measured.  The bounded pool + existing throttle
lets workers scale to actual hardware while routing back-pressure
correctly.

**Applied to both `cpu_worker` (compress) and `cpu_decomp_worker`.**
GPU `h_out` allocations not changed — no evidence they're hitting the
same issue at current concurrency, but the pattern would transfer if
needed.

**Telemetry at -vv.**  Per-worker summary now includes `pool=N waits=K`
showing pool size and yield count.  Non-zero waits indicate the worker
was blocked waiting for the writer — useful for confirming the
backpressure is actually engaging.

---

## v0.13.8 — Result-store buffer pool: decompress page-fault fix

Apply the v0.13.7 page-fault diagnosis to the decompress path via a
proper buffer pool — the simple "per-thread scratch + copy" pattern
that worked for compress can't work for decompress because the output
is the same size as the buffer, so the copy would page-fault as many
bytes as the original allocation.  Instead, recycle whole buffers
through the writer.

**Mechanism.**  `ResultStore` and `AsyncWritePool` now carry
`std::shared_ptr<std::vector<char>>` (alias `FrameBuf`) end-to-end
instead of bare `std::vector<char>`.  Workers maintain a per-thread
pool of FrameBufs and reuse a slot once `use_count() == 1` (writer has
dropped its reference after writing to disk).  Buffers stay resident
across iterations — `resize()` only memsets resident memory, no kernel
page-fault path.

**cpu_decomp_worker now uses the pool.**  The single-frame path
(`acquire_decomp_buf()` → `resize(decomp_size)` → ZSTD writes → push)
and the streaming-frame path (16 MiB chunks in a loop) both pull from
the same per-thread pool.  Pool grows on demand and is bounded
implicitly by FrameThrottle's in-flight cap.

**Compress workers wrap their output in `make_shared` at the push
site.**  The v0.13.7 fix (per-thread scratch buffer, copy `csz` bytes
into a sized vector) is unchanged; that sized vector now becomes the
backing storage for a shared_ptr.  No reuse on the compress side
because `csz` is small for compressible data and the per-iteration
alloc is already cheap.

**GPU workers** (compress D2H, decompress D2H) also wrap `h_out` in
`make_shared` — uniform interface, negligible overhead.

**Expected impact** (needs re-benchmarking on high-core-count
multi-GPU systems):
- `cpu-only` decompress at the default high-thread cap: should
  approach the per-thread sweet-spot ceiling the same way v0.13.7
  lifted compress.  v0.13.7 cpu-only decompress on the worst-affected
  file was 3.71 GiB/s; the hand-tuned low-thread ceiling for the same
  file was 5.25 GiB/s.  Target: 5+ GiB/s.
- `hybrid` decompress should also benefit (same per-thread overhead).
- Compress behaviour unchanged from v0.13.7.

**Allocator overhead.**  shared_ptr's atomic refcount adds two atomic
ops per frame (one push, one writer drop) — measured at ~10-30ns each
on modern hardware.  At even 1000 frames/sec, that's 60µs/sec total.
Negligible relative to the page-fault savings.

---

## v0.13.7 — Hoist per-iteration output buffer in CPU workers

Fix the actual root cause of the hybrid-vs-gpu-only compress gap that
v0.13.6 only partially closed.  The 14-17% slowdown on high-core-count
multi-GPU systems wasn't scheduler overhead from idle threads — it was
**page-fault contention on the per-process mmap_lock from 96 worker
threads simultaneously allocating fresh 16+ MiB output buffers in their
hot loops**.

Discovered via `perf record` on a 256-core / 8-GPU server (zeros.bin
compress, 4-second run).  Counters told the story:

|                    | hybrid (T96) | gpu-only | ratio |
|--------------------|---|---|---|
| sys time           | 45.86s | 4.90s  | **9.35×** |
| context switches   | 107,784 | 1,728 | 62× |
| page faults        | 2.34M | 348k | 6.7× |
| IPC                | 0.43 | 1.55 | 0.28× |

The flame graph for hybrid showed **64% of all cycles in
`asm_exc_page_fault`**, with the call path:

```
compress_one_cpu_frame
  std::vector<char>::resize
    memset_avx512_unaligned_erms
      asm_exc_page_fault
        do_user_addr_fault
          down_read_trylock   ← 28% of fault time on mmap_lock
```

**Root cause.** Both `cpu_worker` and `cpu_worker_rescue` allocated a
fresh `std::vector<char> out_frame` inside the per-iteration block,
which `compress_one_cpu_frame` then grew to `ZSTD_compressBound(src)`
(~16 MiB for the default GPU subchunk size).  Vector growth value-
initializes new elements — for `char`, that's `memset(0)` across every
page, triggering one minor page fault per 4 KiB.  At 96 workers all
hitting this path during the "CPU runs wild while GPU initializes"
phase, the kernel serialized all 96 threads on the single per-process
`mmap_lock` rwsem.  Same pattern in `cpu_decomp_worker` (allocating
`out_buf(t.decomp_size)` per frame).

**Fix (compress only).**  Hoist the output buffer to per-thread
(lifetime = worker thread) in `cpu_worker` and `cpu_worker_rescue`.
On iteration 1, `resize()` pays the page-fault cost once.  On
iterations 2+, the pages are already resident — `resize` just memsets
resident memory (fast, no kernel involvement).  Then copy `csz` bytes
into a sized vector for the result store, preserving `scratch`'s
capacity for the next iteration.

For highly compressible data (csz ≈ 0) the copy is essentially free
and the fix recovers almost all lost throughput.  For poorly
compressible data (csz ≈ src_size) the copy still costs a memcpy of
the compressed output, but the worst-case page-fault storm is gone.

**Decompress is NOT patched** — and we tried, then reverted.  For
decompress, `actual ≈ decomp_size` (output is the decompressed
payload), so the scratch+copy pattern would page-fault as many bytes
on the copy as the original allocation, then ADD a memcpy.  Net cost:
original + memcpy = worse.  The decompress allocation pattern stays
as-is; the writer owns each `out_buf` after `std::move` and frees it
as it drains.  Fixing decompress requires a true buffer pool with
writer-side return — out of scope here.

**This also lifts `cpu-only` compress at the default high-thread cap**
by the same mechanism.  Measurements on a 256-core / 8-GPU system:
zeros.bin compress went from 7.91 to 12.62 GiB/s (+59%),
high_compress from 9.05 to 15.82 (+75%) — both now ABOVE the
hand-tuned low-thread sweet spot from the v0.13.2 baseline (10.5
GiB/s).  The empirical "lower thread count is best" finding is
retired for compress on this hardware class.

Validation on the same hardware:
- `--hybrid` and `--gpu-only` compress now converge to identical
  numbers (3.99 GiB/s on the highly-compressible files) — both
  bottlenecked downstream (writer/NVMe), CPU contribution can no
  longer go net-negative.
- `--cpu-only` (default high-thread cap) is now the clear winner on
  this hardware class: 12.6 GiB/s on zeros vs 3.99 for any GPU-using
  mode.
- Lower-core consumer hardware unchanged (single-thread page-fault
  storm doesn't exist on 24-ish cores).

---

## v0.13.6 — Hybrid mode: proactive batch reservation

Fix hybrid mode regressing below `--gpu-only` on high-core-count
multi-GPU systems.

A 930-measurement sweep at iterations=3 on a 256-core / 8-GPU server
showed hybrid compress at 2.45 GiB/s on zeros.bin vs gpu-only at 3.05
GiB/s — a 20% regression below the GPU-alone baseline.  Decompress was 25-60% slower on every file.
Hybrid mode is supposed to add CPU contribution on top of GPU; on this
hardware class it was instead displacing GPU work.

**Root cause: AUTO floor factor too small.**  `compute_auto_factor_()`
in HybridSched (added v0.12.12) computed the queue floor as
`(gpu_per_worker - cpu_per_worker) / gpu_per_worker`, clamped to [0, 1].
On systems where the per-worker GPU and CPU rates are similar (~0.13
GiB/s each), this produced factor ~0.15.  The effective floor was
`0.15 * streams * batch` — too shallow to actually reserve a GPU round.
96 CPU workers drained the queue during the millisecond-long GPU
processing window, so when a stream returned for its next batch via
`pop_batch_greedy(min_n=1)`, it got a tiny batch.  Small nvCOMP batches
don't amortize per-call kernel-launch overhead, and the throughput loss
from shrunken GPU batches exceeded CPU's contribution.

**Fix: "GPU first, CPU as surplus" policy.**  New AUTO formula keys off
CPU's measured share of aggregate throughput, not per-worker rates:

  cpu_share < 5%   : factor = 4.0  (CPU not contributing → heavy lockout,
                                     hybrid converges to gpu-only)
  cpu_share > 20%  : factor = 1.5  (CPU helping → reserve 1.5 batches,
                                     but let CPU work the surplus)
  in between       : linear interpolation
  warm-up          : factor = 2.0  (proactive default)

Floor is now always >= 1 full GPU round; a CPU pop can never leave the
next GPU batch short.  Cap on `--hybrid-floor-factor` raised from 1.0
to 4.0 so users can lock CPUs out further if needed.

**Second bug: drain-phase short-circuit ignored the floor.**  The
may_take predicate in cpu_worker and cpu_decomp_worker had an early
return when `qs.done && !is_fixed_mode()` — for AUTO mode, once the
producer finished, CPU took regardless of floor.  On a 20 GiB file with
mmap and warm page cache, the reader finishes in ~1s but the GPU drain
takes 10+s, during which CPU floods the queue and shrinks GPU batches.
Symptom: hybrid still ~10% slower than gpu-only after the AUTO formula
fix, on the same hardware class where the formula fix should have
sufficed.  Fix: drain-phase short-circuit only fires when no GPU is
active.  While any GPU stream is registered (working through its share
or about to), the floor applies in both fill and drain phases.

**Third bug: fixed-share mode deadlocked under the new floor.**  The
old code bypassed the floor entirely in fixed-share mode via the
drain-phase short-circuit; with that short-circuit removed,
`--cpu-share 0.5` could deadlock: `should_cpu_take` returns true when
CPU's share is below target, `should_gpu_take` returns true when CPU's
share is at-or-above target minus 2%.  If the floor blocks CPU, CPU's
share stays at 0, GPU's predicate also fails, both sides wait forever.
Fix: skip the floor check entirely in fixed-share mode — the user's
explicit share is the constraint, not GPU batch preservation.  Floor
only applies in adaptive AUTO/NOMINAL mode.

The diagnosis path: gpu-only's per-config sweep showed a flat 3.05
GiB/s ceiling across all batch×stream combinations (the signature of a
downstream bottleneck — writer/NVMe/ResultStore).  On mixed.bin where
nothing was saturated, gpu-only showed a clean "bigger batch wins"
gradient (0.85 → 0.93 GiB/s); hybrid's gradient was flat at ~0.88,
evidence that CPU was destroying the GPU's batch-size tuning by
draining the queue.

Consumer Gen3 hardware (24-core, 2× RTX 2080 Ti) validates the fix
preserves hybrid winning on its target tier: medium_compress.bin
compress shows hybrid 3.30-3.71 GiB/s vs gpu-only 1.87-2.12 GiB/s
(~70% faster).

No behavior change for explicit `--cpu-only`, `--gpu-only`, or users
who set `--hybrid-floor=nominal` or `--hybrid-floor-factor=X` manually.

---

## v0.13.4 — CLI arg-parser hardening + auto-tune log fix

Polish pass on the CLI surface plus one cosmetic bug in the GPU
decompress auto-tuner.  No behavior change on valid inputs; bad inputs
that previously crashed or silently truncated now produce a usage hint.

- **Argument-parser error handling.**  parse_num_arg / parse_int_arg /
  parse_double_arg called std::stoull / std::stoi / std::stod directly,
  with two failure modes:
    - `--gpu-streams=12abc` silently parsed as 12 — stoull stops at the
      first non-digit and never tells the caller.
    - `--gpu-streams=foo` let std::invalid_argument escape to main,
      printing a terminate-style backtrace instead of a usage hint.

  Added parse_u64_value / parse_int_value / parse_double_value helpers
  that catch invalid_argument and out_of_range, verify the full string
  was consumed, and call die_usage on failure.  All three parsing
  wrappers route through them.

- **--gpu-mem-frac validation.**  Hard-rejects values outside (0.0, 1.0)
  and warn-clamps anything outside [0.10, 0.95] so existing scripts that
  pass slightly aggressive values still run but the user learns why
  they did not get what they asked for.

- **--pinned auto|on|off.**  The old code combined parse_str_arg with an
  rfind prefix check and a manual `=` split; the space-separated form
  bypassed validation.  Replaced with a single parse_str_arg call into
  a scratch buffer.

- **Asymmetric default visibility.**  Promoted the PCIe Gen3 →
  --cpu-only notice from V_VERBOSE to V_DEFAULT.  Users on
  workstation-class hardware otherwise saw zero GPU activity during
  decompress and had no signal the runtime had switched backends on
  them.  Prefix changed from [ASYMMETRIC] to gzstd: to match the other
  default-verbosity notices.

- **GPU decomp auto-tune log fired twice per settle.**
  gpu_decomp_worker printed `[AUTO-TUNE] settled at batch=N` on every
  tune-step completion regardless of whether the next phase was REFINE
  or SETTLED, because the verbose-log sat outside the if/else.  Split
  into `refining [lo..hi] trying mid` vs `settled at N`.

---

## v0.13.2 — Build-system fixes for portable-build workflow

Two issues surfaced when first running scripts/build-portable.sh under
the new GitHub Actions release workflow:

- **NVCOMP_ROOT cache variable was ignored.**  CMakeLists.txt's
  find_path and find_library calls only checked `$ENV{NVCOMP_ROOT}`,
  so passing `-DNVCOMP_ROOT=/nvcomp` at the cmake command line had no
  effect — the build silently fell back to CPU-only.  Now the HINTS
  list includes both forms (`${NVCOMP_ROOT} $ENV{NVCOMP_ROOT}`).

- **CPU-only build broken.**  When HAVE_NVCOMP is undefined,
  try_reserve_pinned and release_pinned referenced PinMode and
  opt.pin_mode, which both live inside #ifdef HAVE_NVCOMP in the
  Options struct.  The whole pinned-budget infrastructure is now
  wrapped in #ifdef HAVE_NVCOMP since it's only ever called from GPU
  paths anyway.  CPU-only builds compile cleanly again.

No runtime behavior change for users on the GPU path.

---

## v0.13.1 — Multi-frame oversized decompress no longer corrupts output

The CPU streaming-decompress path added in v0.12.24 (for frames whose
decompressed size exceeds 64 MiB) was only safe for **single-frame**
inputs.  When fed multi-frame inputs with per-frame decomp_size > 64 MiB,
streaming chunks reused sequence numbers that collided with adjacent
frames' natural seqs in the ResultStore — chunks overwrote each other
and the writer either produced truncated output or got stuck waiting
for a frame that had been clobbered.

Surfaced on server during a benchmark sweep at `--ultra -22`: ultra
auto-bumps chunk size to 128 MiB (the windowLog 27 minimum), and the
RAM budget on server's 256 GiB allowed the full 128 MiB to survive,
so every frame qualified for the streaming path.  Two failure modes
observed:
- `cpu-ultra22 / mixed.bin` decompress: produced 2.7 GiB of output for a
  19.5 GiB input (clean exit, truncated data).
- `cpu-ultra22 / zeros.bin` decompress: writer-deadlock detector fired
  with `frame 163 of 577 missing (have 161 buffered)`.

Fix: in `cpu_decomp_worker`, before entering the streaming branch, wait
for the reader to set `producer_done` and only stream when
`results.total_tasks == 1`.  Multi-frame oversized inputs fall through
to the normal `ZSTD_decompressDCtx` path with a per-frame `decomp_size`
allocation — uses more peak RAM but is correct and parallelizable.
The original v0.12.24 motivation (single-frame `zstd` /
`--sliding-window` outputs) is preserved.

Regression test added in `gzstd-test.sh` (`--chunk-size 100` on a
200 MiB input forces 2 frames of 100 MiB each, which trips the bug
without needing the multi-minute `--ultra -22` workload).

---

## v0.13.0 — Asymmetric mode + Apache 2.0 relicense

**License: GPL v3 → Apache 2.0.**  Required for distributable binaries
that link nvCOMP: NVIDIA's nvCOMP license (§2.6) prohibits using the
SDK in a way that would subject it to a copyleft open-source license.
Apache 2.0 keeps gzstd's source permanently free and public, preserves
copyright, and adds an explicit patent grant + retaliation clause that
GPL doesn't have.  See LICENSE (root) and the SPDX header in gzstd.cpp.
Same license used by TensorFlow, PyTorch, RAPIDS, and every other
CUDA-using project that ships binaries.

### Asymmetric mode: smart, hardware-aware backend defaults

GPU compress wins consistently across hardware tiers, but on PCIe Gen3
(consumer cards: RTX 20-series, 30-series, etc.) the D2H transfer cost
makes hybrid *decompress* slower than CPU MT for every data type
measured on Workstation (2× RTX 2080 Ti):

| Data type | CPU-only | Hybrid | Asymmetric default wins by |
|-----------|----------|--------|----------------------------|
| zeros     | 4.88     | 3.50   | +39%                       |
| trivial   | 4.65     | 3.42   | +36%                       |
| medium    | 2.80     | 2.45   | +14%                       |
| mixed     | 1.40     | 1.31   | +7%                        |
| random    | 1.40     | 1.32   | +6%                        |

(GiB/s decompress on Workstation; raw v0.11.20 benchmark numbers.)

gzstd now picks the backend based on hardware *and* operation:
- **Compress (any GPU):** hybrid — GPU compress consistently wins.
- **Decompress / test, PCIe Gen<4:** cpu-only — D2H eats GPU benefit.
- **Decompress / test, PCIe Gen4+:** hybrid — D2H is cheap (Server's H100s).
- **Detection unavailable / no GPU:** hybrid (degrades gracefully).

PCIe gen detection uses `nvmlDeviceGetMaxPcieLinkGeneration()` (the
hardware ceiling, not `Curr` — idle GPUs drop their link to Gen1 for
power management and would otherwise mislead the heuristic).  Fallback
parses `/sys/bus/pci/devices/*/max_link_speed` when NVML isn't built in.

Visible at `-v` as `[ASYMMETRIC] PCIe Gen3 detected; defaulting
decompress to --cpu-only`.  Override with `--hybrid` or `--gpu-only`
when you specifically want to measure or use GPU decompress.

Implementation: `Options::backend_user_set` tracks whether the user
explicitly chose a backend (parsing `--cpu-only`/`--gpu-only`/`--hybrid`
or being implied by `--sliding-window`); the new `apply_backend_defaults()`
runs after `parse_args` and only fills in defaults when no explicit
choice was made.

**Tuning-flag promotion.** Asymmetric mode would silently route around
GPU-tuning flags on Gen3 — `gzstd -d --gpu-batch=64 file.zst` would auto-
flip to cpu-only and the user's tuning hint would do nothing.
`Options::gpu_hybrid_tuning_seen` now tracks any flag that only makes
sense in hybrid/GPU mode (`--gpu-batch`, `--gpu-streams`, `--gpu-devices`,
`--gpu-mem-frac`, `--pinned`/`--no-pinned`, `--cpu-share`, `--cpu-batch`,
`--cpu-backlog`, `--hybrid-floor`, `--hybrid-floor-factor`).
`apply_backend_defaults` promotes these to an implicit `--hybrid` when
no explicit backend flag was given — same precedent as `--sliding-window`
implying `--cpu-only`.  Explicit `--cpu-only` always wins over the
promotion (unchanged precedence).

---

## v0.12.51 — `--cpu-share` actually enforces the requested split

`--cpu-share X` was effectively a no-op: every value from 0.0 to 1.0
landed at ~85% CPU work on Workstation, because the `may_take` predicate
short-circuited on `qs.done` (`if (qs.done) return true;`) — so the
moment the reader called `set_done()`, CPUs drained everything
regardless of the user-set share.  The GPU side never consulted the
share at all, so even after fixing the drain bypass, high shares
(0.9, 1.0) capped at ~86% CPU because GPU kept stealing work.

Three coordinated fixes in `HybridSched` and the worker loops:
- The `qs.done` bypass now only triggers in fixed-share mode if every
  GPU stream has unregistered (real GPU exit, not just stuck in CUDA).
  Otherwise the share is honored through drain.
- New `should_gpu_take()` is the symmetric counterpart of
  `should_cpu_take()`: in fixed-share mode GPU yields when the
  cumulative CPU ratio is below `target − 0.02`, same hysteresis band
  the CPU side uses, so the ratio oscillates around the target instead
  of one side starving.  Adaptive mode is unchanged (always returns
  true; the EMA path drives sharing through `gpus_waiting_` and the
  queue floor).
- GPU workers propagate `producer_done_seen` when the share-yield path
  observes `queue->drained()`, otherwise a perpetually-yielding GPU
  would never exit its loop and the run would hang at high shares.

Measured on Workstation, 19.5 GiB medium-compressibility input, 22 CPU
threads + 2× RTX 2080 Ti.  Before: 0.0 → 0.82, 0.5 → 0.86, 0.9 → 0.84
(all within noise of each other).  After: 0.0 → 0.02, 0.1 → 0.12,
0.25 → 0.27, 0.5 → 0.51, 0.75 → 0.76, 0.9 → 0.87, 1.0 → 0.98.  The
slight undershoot at 0.9/1.0 is end-of-run drainage where GPU sweeps
the tail after CPU threads exit.

Adaptive mode (no `--cpu-share`) is untouched and still hits 5.6 GiB/s
compress on the same input.

---

## v0.12.50 — `--preallocate` / `--no-preallocate` toggle for fallocate

Adds the same on/off control over the `fallocate` upfront-preallocate
that `--mmap`, `--pinned`, and `--direct` already have.  Default stays
ON (matches the prior unconditional behaviour); `--no-preallocate`
skips fallocate so users can A/B test whether it actually helps on
their filesystem.

Touches all four call sites where `g_direct_writer->preallocate(...)`
fires today (compress/decompress, CPU-only and nvCOMP paths).
Preallocation only runs when:
- `--direct` (O_DIRECT) is in effect (fallocate is on `DirectWriter`)
- The expected size is known (input file size for compress, sum of
  frame_decomp sizes for decompress)
- `--preallocate` is on (new — was previously unconditional)

`--no-preallocate` is documented as useful on filesystems that
handle inline extent allocation efficiently (XFS, ZFS), or for
benchmarking the allocation cost.

Tests: 263/263.

---

## v0.12.49 — `[STARTUP]` banner + uniform `[TAG]` verbose-output style

Two related changes addressing the "no visual feedback for several
seconds after pressing enter" complaint on loaded servers:

**1. `[STARTUP]` banner before any heavy init.**  At -v+, the very
first line printed is `[STARTUP] gzstd vX.Y.Z MODE (backend)`.
Printed in main() right after `parse_args` returns, BEFORE `cudaGet
DeviceCount`, file open, output preallocate, or any other potentially
slow step.  Output is `fflush`'d immediately to bypass stderr line
buffering.  Examples:

```
[STARTUP] gzstd 0.12.49 COMPRESS (cpu-only)
[STARTUP] gzstd 0.12.49 COMPRESS (hybrid, CPU share adaptive)
[STARTUP] gzstd 0.12.49 DECOMPRESS (gpu-only)
[STARTUP] gzstd 0.12.49 TEST (auto-select backend)
```

**2. Uniform `[TAG]` style across all verbose output.**  The codebase
had drifted to a mix of `lowercase: prefix:`, `[lowercase]`, and
`[UPPERCASE]` formats.  Standardised every -v / -vv / -vvv message to
`[UPPERCASE_TAG] sentence-case body` with a single space between tag
and body.  Tags now used:

- `[STARTUP]`, `[INIT]`
- `[CPU]`, `[CPU/T#]`
- `[GPU]`, `[GPU#]`, `[GPU#/S#]`
- `[HYBRID]`, `[RESCUE]`, `[WRITER]`, `[READER]`, `[SPLIT]`
- `[THROTTLE]`, `[PINNED]`, `[AUTO-TUNE]`
- `[MMAP]`, `[O_DIRECT]`, `[FALLOCATE]`, `[FSYNC]`, `[RENAME]`
- `[ULTRA]`, `[SLIDING-WINDOW]`

Lines previously emitted as `throttle: ...`, `hybrid: ...`,
`writer: ...`, `using mmap...`, `using O_DIRECT...`, `preallocated
...`, `streamed N frames`, `[pinned] ...`, `[auto-tune] ...`,
`atomic rename: ...`, `fsync: ...`, `GPUs: ...` etc. are now all
prefixed with the appropriate uppercase tag.

The duplicate "Using hybrid mode: CPU share X%" line that used to
fire late inside `compress_nvcomp` is replaced by the early
`[STARTUP]` banner; a `[HYBRID]` confirmation line at -vv announces
when the scheduler actually starts.

Test grep patterns updated for the new format (5 tests).

---

## v0.12.48 — `--throttle-frames=0` / `--no-throttle` to fully disable throttling

For benchmarking the no-throttle baseline.  `FrameThrottle` now
recognizes a non-positive `max_in_flight` as "disabled":
`acquire`/`release`/`set_done` become no-op early returns (no lock
taken, no permit accounting, no peak/block stats).

`--throttle-frames` parsing extended:
- `N >= 1`  : explicit cap (existing behaviour, `source=user`).
- `N == 0`  : DISABLE throttle entirely.  -v shows `throttle: DISABLED`.
- `N == -1` : auto / formula (NEW DEFAULT — was 0 before; semantic
              shift on the sentinel only, default behaviour unchanged
              for users who never passed the flag).
- `N <= -2` : rejected with exit 2 (usage error).

`--no-throttle` is a convenience alias for `--throttle-frames=0`.

Both `compress_*` and `decompress_*` paths construct `FrameThrottle(0)`
when disabled; the existing acquire/release calls scattered through
the workers and the writer just no-op.  No control-flow changes
required outside the throttle class.

`log_throttle_stats` skips the stats line on disabled throttles and
just prints `DISABLED` instead.

**Quick A/B on Workstation** (24-core, 2 GiB mixed input):

| mode | throttle=auto | throttle=0 |
|---|---|---|
| compress | ~0.53 s | ~0.55 s |
| decompress | ~0.62 s | ~0.68 s |

Throttle off is *slightly slower* on this workload — without backpressure
the result store can grow large enough that L3 / RAM cache effects
hurt.  As suspected, the throttle is a guardrail not an optimization,
but you can now measure that directly.

Tests: 263/263 (+2 new — disabled-mode verification, `--no-throttle`
alias verification, replaced the old "must reject 0" rejection test).

---

## v0.12.47 — `--sweep-matrix` benchmark option for backend × mmap × pinned

Adds a small structured sweep that produces 10 configs (cpu-only × 2
mmap states + hybrid × 4 + gpu-only × 4, with pinned skipped on cpu-only
since it's a GPU-side knob).  Captures the same "which tricks actually
help on this system" analysis we did manually in v0.12.45/46 but as a
re-runnable harness, so the result interpretation can be reproduced on
any new machine without hand-rolling shell loops.

Smoke-tested against the 20 GiB `medium_compress.bin` profile:

| config | GiB/s |
|---|---|
| mtx-cpu-mmap | **7.05** (best) |
| mtx-hyb-mmap-pin0 | 5.23 |
| mtx-hyb-mmap-pin1 | 4.47 |
| mtx-cpu-nommap | 2.42 |
| mtx-gpu-mmap-pin0 | 2.33 |
| mtx-gpu-nommap-pin1 | 2.33 |
| ... | |

Confirms the v0.12.45/46 conclusions at 20 GiB scale on Workstation
(2× 2080 Ti): CPU-only crushes, mmap wins, pinned hurts.

`--sweep-all` now also enables `--sweep-matrix`.  Add `--sweep-matrix`
on its own for a focused 10-config run.

---

## v0.12.46 — `--mmap=on/off` toggle (default: on)

The mmap zero-copy reader has been the default for regular-file inputs
since early in the project but had no escape hatch — useful for
benchmarking against a stack of "tricks that don't help on this
system" (O_DIRECT, pinned RAM, atomic rename).  This pulls mmap up to
the same level: on by default, but `--no-mmap` lets you A/B against
fread to verify it's actually winning on your hardware.

**Local validation** (Workstation, 4 GiB mixed input, page cache warm):
| mode | mmap (default) | --no-mmap |
|---|---|---|
| `--cpu-only -T18` compress | ~1.4 s | ~2.2 s (~50% slower) |
| `--gpu-only` compress | ~3.9 s | ~3.9 s (wash) |

mmap wins clearly on CPU-only paths because workers read directly from
the page cache — no producer-side fread + memcpy through a userspace
buffer to serialise the input read.  GPU paths are a wash: the input
gets H2D-copied either way, and the page cache makes pageable cudaMemcpy
near-DMA-speed regardless of source.

`--mmap=on` / `--mmap=off` / `--no-mmap` accepted.  Pipes and stdin
always fall back to fread (mmap requires a regular file).

---

## v0.12.45 — `--pinned` default flipped to `off` (pinned was measured slower)

The plumbed-in pinned-host-memory infrastructure (v0.12.43, v0.12.44)
turned out to be slower than pageable on every workload tested.
On Workstation (2× 2080 Ti), 4 GiB mixed-compressibility input:

| mode       | --pinned=off | --pinned=on    |
|------------|--------------|----------------|
| compress   | ~3.6 s       | ~4.2 s (-15%)  |
| decompress | ~1.9 s       | ~4.4 s (-2.4×) |

Decompression was particularly bad — pinned cudaMemcpy + extra copy
into the result vector was 2-3× slower than direct device→pageable.

Likely causes:
- Input pages are usually already in the OS page cache (mmap'd file
  on a fast NVMe), so cudaMemcpy from pageable is near-DMA-speed
  anyway — the locked-page DMA path doesn't win.
- Locking pages out of the page cache hurts other parts of the
  pipeline (reader fread-ahead, writer fwrite cache).
- The mandatory pinned -> pageable memcpy for the result vector adds
  pure overhead with no offsetting gain.

**Fix.** `Options::pin_mode` default changed from `AUTO` to `OFF`.
The infrastructure is plumbed and exposed; users can opt in with
`--pinned=on` or `--pinned=auto` if their hardware/workload differs.
Help text updated to explain the trade-off.

Existing pinned tests (which verify flag acceptance) still pass.
`-v` no longer prints `[pinned]` lines unless explicitly enabled.

---

## v0.12.44 — Compress reuses one pinned buffer for both H2D and D2H

v0.12.43 only pinned H2D for compress (output went direct device →
pageable vector).  But the H2D pinned slot is unused after the upload
finishes — the GPU has the data on-device, the host slot sits idle for
the rest of the batch.  v0.12.44 reuses that same slot for the D2H
output readback:

1. H2D phase: input chunk is memcpy'd into `pinned[i]`, then
   `cudaMemcpyAsync` host → device.
2. Compute phase: GPU has the data; pinned slot is idle.
3. D2H phase: compressed output `cudaMemcpy`'d device → `pinned[i]`,
   then `memcpy` from pinned slot into the output `std::vector<char>`.

Each slot is sized to `max(gpu_chunk, max_out_chunk)` (≈ 16 MiB +
~3 KiB) so either direction's data fits.  Per-stream batches are
already serialised (`C.busy` gates re-pop until D2H + result delivery
complete), so there's no buffer-conflict race.

**Net effect:** compress now gets pinned D2H **for free** — same RAM
allocation as before, just slightly larger slot stride.  Pinned
cudaMemcpy uses a faster DMA path than pageable, so D2H finishes
sooner and the tot_ms / GPU-throughput numbers at -vv reflect that.

The `[pinned]` log line at -v changed:
```
[pinned] H2D 1.62 GiB reserved              # before v0.12.44
[pinned] H2D+D2H 1.63 GiB reserved (shared per slot)  # v0.12.44+
```

Decompress is unchanged (separate H2D-pageable / D2H-pinned scheme
from v0.12.43).  Adding pinned H2D there would mean an extra
mmap → pinned memcpy, which usually doesn't pay off because the
input pages are already cached.

---

## v0.12.43 — `--pinned auto` rations to ≤50% RAM + adds D2H pinning on decompress

### `--pinned auto` is now actually a heuristic

Before: `auto` and `on` were treated identically — both unconditionally
called `cudaHostAlloc`.  Misleading naming.

Now: `auto` rations pinned host memory to ≤50% of available system RAM,
summed across ALL gpu-worker threads (compress H2D + decompress D2H).
Streams that fit get pinned.  Streams that don't ("unlucky" ones) fall
back to pageable memory silently.  Same fallback if `cudaHostAlloc`
fails for any reason.

`--pinned on` keeps the prior behaviour (unconditional reserve, ignores
the budget).  `--pinned off` (and `--no-pinned`) skip pinning entirely.

Implementation: a global `g_pinned_bytes_reserved` atomic + `try_reserve_pinned` /
`release_pinned` helpers.  AUTO uses CAS to reserve from the global
budget; ON / OFF short-circuit.  The `[pinned]` log line at -v shows
each reservation and any skipped streams with the reason.

### Pinned D2H buffer added to decompress

Before: `DecompStreamCtx` had no pinned host memory at all — every D2H
copied straight from device to a freshly-allocated `std::vector<char>`
in pageable memory.

Now: each decompress stream allocates a pinned host staging buffer of
`alloc_batch * alloc_decomp` bytes (typically a few GiB per stream).
The D2H loop copies device → pinned slot, then `memcpy` from pinned
slot into the output `std::vector`.  Pinned cudaMemcpy uses a faster
DMA path; the pinned-to-pageable memcpy is a plain `memcpy` (which the
kernel optimises well).

Allocation honours the same `--pinned auto` budget; on `--pinned off`
the decompress path falls back to the previous direct-to-pageable
behaviour.  The pinned buffer is reused across batches and grown only
when `alloc_batch` or `alloc_decomp` increase, so per-batch overhead
is zero.

(Compress D2H still uses direct exact-size copies — output sizes are
variable, so a fixed-size pinned slot would either waste 2× memory or
need a per-chunk pinned allocator.  Could be added later if measured
to matter.)

---

## v0.12.42 — `--help`: throttle flags moved to CPU/GPU TUNING (apply to all modes)

`--throttle-factor` and `--throttle-frames` were listed only under
HYBRID SCHEDULER, which made them look hybrid-only.  They actually
affect every multi-threaded path (CPU-only compress with `-T ≥ 2`,
CPU-only decompress, GPU-only, hybrid).

Moved the canonical description into CPU TUNING with `[all modes]`
markers and concrete tuning guidance ("bump to 8 or 16 if you see
`source=pipeline` and the writer is bursty").  Added a cross-reference
in GPU TUNING with a GPU-specific note about permit starvation when
N_GPUs * streams * batch exceeds the default budget.  Removed the
duplicate entries from HYBRID SCHEDULER.

No behaviour change — purely documentation.

---

## v0.12.41 — `--overwrite`: unlink-then-create instead of truncate-on-fopen

**Symptom (Server, 432 GiB output).**
```
time ./build/gzstd -d --cpu-only -T18 --direct --overwrite -v ...
using O_DIRECT for output (--direct)
```
The `using O_DIRECT` line appeared 10–30 seconds after the command was
launched.  No output at all during that window.

**Cause.**  `--overwrite` opened the existing target with `fopen(path,
"wb")`, which truncates the file in place.  On ext4, `truncate(0)` on
a 432 GiB file has to free every extent the inode references — that's
O(file_size), and ext4's journal makes the freeing synchronous before
`fopen` returns.  All subsequent setup (the `using O_DIRECT` log,
throttle config, worker spawn) sat behind that truncate.

**Fix.**  `--overwrite` now `unlink()`s the existing target first, then
`fopen("wb")` creates a fresh empty inode in O(1).  The original
inode is unreferenced immediately; ext4 frees its extents in the
background.  No user-visible blocking.

Verified locally on a 4 GiB stand-in: time-to-first-output went from
visible delay to ~0.2 s end-to-end including round-trip.

`--sync-output` semantics are unchanged.  `-f` (atomic, with rename)
already wrote to a fresh `.gzstd.tmp` file and didn't have this
problem.

---

## v0.12.40 — Parameter-honor verification tests + `--overwrite` no progressive sync

### Tests

The existing test suite did round-trip checks for `--gpu-batch=N` etc.
("compress with the flag, decompress with the flag, output matches"),
but never verified the flag was actually applied at runtime.  The
v0.12.39 `--gpu-batch` regression slipped through because the tests
couldn't see that batches were popped at size 4 instead of N.

New `Parameter honor verification` section (+20 tests, 261 total) parses
verbose output to check runtime behaviour matches CLI input:

- **`--gpu-batch=N` honored at -vv**: parses every `[GPU/S] take batch=N`
  line, asserts all non-final batches equal N exactly.
- **`--gpu-streams=N` honored**: counts unique `[GPU#/S0..N-1]` indices
  in `pre-alloc` lines.
- **`--chunk-size=N` produces `ceil(file_size / N MiB)` frames**: counts
  `[CPU/T#] take seq=` lines emitted at -vv.
- **`-T N` spawns N workers**: greps for the worker-online line at -v
  (or single-thread streaming path for `-T 1`).
- **Verbosity escalates correctly**: `-v` has no `[CPU/T#] take seq=`
  (V_DEBUG content); `-vv` does, but no `[SPLIT]` (V_TRACE content);
  `-vvv` includes `[SPLIT]`.  Unique-line count strictly increases with
  verbosity level.
- **`-M N` round-trip**: re-verifies the v0.12.30 fix end-to-end.
- **`--throttle-frames=N` visible at -v**: greps for `source=user` or
  the explicit count.
- **`--no-sparse` vs default sparse**: compares `stat -c '%b'` block
  count on an all-zeros decompressed file.
- **`--ultra` is required for level 20+** and `--ultra -20` produces
  valid output.

### `--overwrite` skip progressive writeback

In v0.12.25 we enabled `sync_file_range(SYNC_FILE_RANGE_WRITE)` for all
decompress runs to fix the multi-second rename stall on ext4
`data=ordered`.  But `--overwrite` skips the tmp+rename dance entirely —
the rename stall doesn't apply, and the writeback hint just steals
bandwidth from `fwrite`.  Now disabled when `unsafe_overwrite` is set.

`--sync-output` is still opt-in (default off): the only thing that
forces an explicit `fsync()` on the output is `--sync-output`.  Plain
`fclose()` flushes user buffers to the kernel; the OS handles writeback
on its own schedule.

---

## v0.12.39 — Honour `--gpu-batch=N` exactly (full batches, not soft-min)

**Symptom (Workstation).**  `gzstd -d --gpu-only --gpu-batch=64 -vv` showed
`pre-alloc batch=64` (buffers correct) but actual pops were small:
```
[GPU0/S0] take batch=4 seq=[0..3] in=22.05 KiB
[GPU0/S0] take batch=8 seq=[8..15] in=128.00 MiB
[GPU1/S0] take batch=7 seq=[16..22] in=112.00 MiB
```

**Cause.**  Both `compress_nvcomp` and `decompress_nvcomp` pop with a
hardcoded soft minimum:
- decompress: `pop_batch_greedy(pop_n, ..., min_n=min(pop_n, 4))`
- compress:   `pop_batch_greedy(pop_n, ..., min_n=1)`

So when the queue had only 4 frames available, the GPU returned with 4
even though the user pinned `--gpu-batch=64`.  The soft minimum is
sensible during auto-tuning (multi-GPU shouldn't serialize behind a
single producer) but contradicts the user's explicit pin.

**Fix.**  When `shared_tune->locked` is set (user pinned
`--gpu-batch`), `min_n = pop_n` — wait for the full batch.  When
unlocked (auto-tuner active), the previous soft minimums apply.
`pop_batch_greedy` still returns early at end-of-queue regardless, so
no deadlock — but during steady-state operation the GPU now sees the
batch size the user asked for.

Applied to both compress (gzstd.cpp:5462) and decompress
(gzstd.cpp:6927).

---

## v0.12.38 — Restore concurrent worker spawn / parser (v0.12.21 architecture)

**Regression introduced in v0.12.22.**  When `--sliding-window` shipped
in v0.12.22, `decompress_nvcomp` was restructured to call
`stream_frames_to_queue` BEFORE spawning workers — so the producer's
`max_frame_decomp` could be checked against `GPU_SUBCHUNK_MAX` to
short-circuit GPU init for oversized single-frame files.

**Side effect.**  On large inputs the parse phase blocks for tens of
seconds.  All worker init (`throttle: …`, `[GPU] N device(s) online`,
`[GPU#/S#] pre-alloc batch=`, `hybrid decompress: N CPU threads`,
`hybrid: tick …`) was silent during that window — users saw nothing
but the producer's `[SPLIT] frame N` lines until parsing finished.

v0.12.21 had it the right way around: workers spawn first, parser runs
afterwards while workers are already consuming.  Init lines appeared
immediately at -v/-vv/-vvv.

**Fix.** Restored the v0.12.21 ordering in `decompress_nvcomp`:
1. Detect GPU device count (existing).
2. **NEW:** `peek_first_frame_decomp_size(in)` — read just the frame
   header bytes, get frame 0's decomp size, then `fseek` back to 0.
   If size > 16 MiB (single-frame oversize), set `device_count=0` and
   fall back to CPU.  Cheap because it touches only ~64 bytes.
3. Set up throttle, writer thread, hybrid scheduler.
4. Spawn CPU pool and GPU workers (init lines fire here).
5. Run `stream_frames_to_queue` (workers consume concurrently as the
   parser pushes).

The peek-only check covers the typical "oversize" case (zstd /
--sliding-window single-frame files where frame 0 IS the whole file).
For pathological multi-frame files where only a non-first frame is
oversize, the GPU runtime fallback path handles it as before.

User-visible result: at `-vvv` the throttle config, GPU device list,
`[GPU] N device(s) online`, and per-stream `[GPU#/S#] pre-alloc`
output all appear immediately when the command is run, instead of
after a 30-second delay on a 432 GiB input.

---

## v0.12.37 — CPU decompress worker verbose output parity with compress

**Symptom.** Compression's `cpu_worker` emits per-task and per-thread
verbose output:
- `[CPU/T#] take seq=N in=X` before each frame (`-vv`)
- `[CPU/T#] seq=N in=X out=Y ms=… thr=…` after each frame (`-vv`)
- `[CPU/T#] total tasks=… in=… out=… time=…ms thr=…` per-thread summary (`-vv`)
- `[CPU/T#] idle (0 tasks)` for unused workers (`-vvv`)

Decompression's `cpu_decomp_worker` only emitted the post-frame
`seq=…` line.  No "take" line, no per-thread summary, no idle reporting
— so `--cpu-only -d -vv` and `--hybrid -d -vv` looked drastically more
sparse than the equivalent compress runs.

**Fix.** Added the missing logs to `cpu_decomp_worker` so output now
matches the compress pattern:
- Per-task `[CPU/T#] take seq=N comp=X decomp=Y` before processing (V_DEBUG)
- Per-thread `[CPU/T#] total tasks=… comp=… decomp=… time=…ms thr=…`
  summary at exit (V_DEBUG)
- `[CPU/T#] idle (0 tasks)` for unused workers (V_TRACE)

Trace-mode users now see the same level of detail on the decompress
side that they've always had on compress.

---

## v0.12.36 — Visible init output during decompress pre-scan

**Symptom (Server, large `-d` runs).** With a 432 GiB `.zst` file the user
saw a long stretch of nothing but `[SPLIT] frame N` lines and asked
"where's the init output?".  No `[GPU]` device-online lines, no
`[GPU/S] pre-alloc batch=`, no throttle line — until the parse phase
finished tens of seconds later.

**Cause.** `decompress_nvcomp` does a full pre-scan of the input
(`stream_frames_to_queue`) *before* spawning GPU workers.  The pre-scan
is needed to detect oversized frames (sliding-window / `zstd`) and
fall back to CPU before allocating GPU buffers it can't fill.  But on
large inputs the pre-scan is the bulk of wall time, and during it only
the producer's `[SPLIT]` lines emit — the user-visible init lines
(throttle config, `[GPU] N device(s) online`, `[GPU#/S#] pre-alloc`,
etc.) all queue up behind the pre-scan.

**Fix.** Added three `[INIT]` log lines that fire BEFORE the pre-scan:
- `[INIT] decompress: N GPU(s) detected, mode=gpu-only|cpu-only|hybrid|auto`
- `[INIT] pre-scanning input frames (workers spawn after pre-scan)`
- (after pre-scan) `[INIT] pre-scan complete: N frames, max_decomp=X (Ts)`

Visible at `-v`/`-vv`/`-vvv`.  This doesn't change the architectural
ordering — workers still spawn after pre-scan — but the user now sees
that gzstd is alive and what phase it's in.  A future change can move
parsing into a thread that runs concurrently with worker spawn.

---

## v0.12.35 — Per-chunk `-vvv` output for GPU compress/decompress

**Symptom (Server, `--gpu-only -d -vvv`).** The trace output looked
sparse — mostly just the producer's `[SPLIT] frame N` lines every 1000
frames, with little visible GPU activity.

**Cause.** Side effect of v0.12.32: per-stream batches are now allowed
to grow up to 256 chunks (vs the previous 8-cap).  The existing
`[GPU#/S#] take batch=` and `[GPU#/S#] done batch=` lines fire once per
batch — at V_DEBUG (`-vv`).  After v0.12.32 a 16k-frame run produces
~63 batches instead of ~2000, so those lines show up ~30× less often.
At `-vvv` the user expects flood-of-detail, not "less than `-vv` used
to give."

**Fix.** Added per-chunk emission at V_TRACE (`-vvv`) in three places:
- GPU compress async-poll completion path
- GPU compress sync-drain completion path
- GPU decompress completion path

Each chunk in a completed batch now prints
`[GPU#/S#] chunk seq=N in=X out=Y` at -vvv.  V_DEBUG output is
unchanged.

---

## v0.12.34 — Test-count display in `gzstd-test.sh`

The runner's progress bar showed `N of 192` while the actual run
ended at 241 tests, finishing past 100% completion.  The `count_tests`
function had a hand-maintained per-section breakdown that drifted as
new sections were added.

Replaced with a single `EXPECTED_TESTS=241` constant at the top of the
file — bump it when you add/remove tests.  Two safety nets prevent
display breakage if the constant is forgotten:

- `progress_bar` clamps `pct` to 100 and auto-expands `TOTAL_TESTS` if
  the running count exceeds the planned count, so the bar never shows
  more than 100%.
- A drift-check line at the end of the run prints
  `note: EXPECTED_TESTS=N at top of script but M ran — please update.`
  whenever the actual ran count diverges from the constant.

Simpler than chasing a perfect static count or maintaining a cache file.

---

## v0.12.33 — Throttle starvation in hybrid mode (GPUs blocked on permits)

**Symptom (Server, hybrid compress).** Per-batch GPU subchunk count grew
fine after v0.12.32, but `nvtop` showed the H100s mostly idle.  CPUs
were doing the bulk of the work while GPUs sat blocked.  `-vvv` reported
`gpus_waiting=0`, which is technically correct (the wants/got window is
microseconds long) but obscured the real cause.

**Cause.** The frame-throttle budget in both `compress_nvcomp` and
`decompress_nvcomp` was sized off `opt.gpu_batch_cap` (default 8):

```cpp
const int comp_gpu_batch_floor = gpu_count * gpu_streams * opt.gpu_batch_cap;
const int comp_parallelism     = cpu_threads + comp_gpu_batch_floor;
FrameThrottle throttle(compute_throttle_budget(..., comp_parallelism, ...));
```

After v0.12.32 the auto-tuner can grow per-stream batches up to
`AUTO_TUNE_BATCH_CEILING` (256), but the throttle was sized for 8.  On
Server (8 GPUs × 1 stream + 96 CPU workers), every CPU that had popped a
frame was holding one permit (held until the writer drains it), so when
a GPU stream tried to `bp->acquire(pop_n)` for, say, 64 permits, it
blocked waiting for CPUs to drain.  Effectively the GPU pipeline was
serialised through CPU writeout speed.

**Fix.** Both `compress_nvcomp` (line 6125) and `decompress_nvcomp`
(line 7333) now compute the throttle floor using the *effective* per-
stream max:

```cpp
per_stream_budget = opt.gpu_batch_user_set
    ? opt.gpu_batch_cap
    : std::max(opt.gpu_batch_cap, AUTO_TUNE_BATCH_CEILING);
gpu_batch_floor   = gpu_count * gpu_streams * per_stream_budget;
```

When `--gpu-batch=N` is set, the budget honours that value exactly (no
auto-grow either, so no headroom needed).  Otherwise it provisions
enough permits for the auto-tuner's full growth path.

**Server example.** Before: floor = 8×1×8 = 64; throttle ≈ 640 frames
total → 8 streams × 64 = 512 GPU permits + 96 CPUs ≈ over budget.
After: floor = 8×1×256 = 2048; throttle ≈ 8192 frames (RAM-capped) →
2048 GPU + 96 CPU = 2144, well under budget.

---

## v0.12.32 — Fix GPU batch frozen by allocation (auto-tuner had no headroom)

**Symptom (Server, `--gpu-only` compress).** The per-batch GPU subchunk
count was stuck at 8 across the entire run regardless of throughput.  The
shared auto-tuner appeared to do nothing.  Hybrid compression had the same
problem (same code path).  Decompression was partially affected for files
under ~10 GiB.

**Cause.** Two interacting pieces, present in both compress and decompress:
1. The GPU init path allocates per-stream buffers based on
   `per_stream_cap = std::min(opt.gpu_batch_cap, HARD_BATCH_CAP)` — for
   compress that defaults to `min(8, 1024) = 8`; for decompress on small
   files it's `min(16, 1024) = 16`.  A VRAM-fit search lowers this further
   if needed but never raises it.
2. The pop site clamps the per-batch size: `pop_n = std::min(pop_n,
   C.per_stream_batch)`.  So even when `SharedTuneState::batch_size` grew
   to 16, 32, etc., the actual pop was still 8 (compress) or 16
   (small-file decompress) because the buffers were only big enough for
   that many subchunks.

The auto-tuner's growth path was therefore silently dead on those paths.
Long-standing: the clamp was introduced in v0.10.34 alongside
`SharedTuneState`.

**Fix.** Both `compress_nvcomp` and `decompress_nvcomp` now size per-stream
buffers up to `AUTO_TUNE_BATCH_CEILING` (256) when `--gpu-batch` is not
user-pinned, giving the shared tuner real room to grow.  The VRAM-fit
halve loop still shrinks this if the GPU can't hold it.

**Compress.**  Pure win — buffer was previously capped at 8.

**Decompress.**  Already-large files (>75 GiB → cap=256, >10 GiB →
cap=64) are unchanged because `max(cap, 256) == cap` (or close to it).
Small files (<10 GiB) now allocate up to 256 per stream instead of 16.

**VRAM impact.**  Each subchunk needs `gpu_chunk + max_out_chunk +
temp/N` in device memory.  With 16 MiB chunks that's ~33 MiB per slot
plus nvCOMP scratch.  256 slots per stream is ~8.5 GiB plus scratch —
fits comfortably in H100 VRAM under the default `--gpu-mem-frac=0.60`.
On smaller GPUs the binary-search VRAM-fit loop already halves the
allocation when needed.

**User override.** Pass `--gpu-batch=N` to pin a specific size and skip
auto-tuning; that path is unchanged on both compress and decompress.

---

## v0.12.31 — Fix `out:%` jumping to ~90% immediately on `--cpu-only` compress

**Symptom (Server, 432 GiB tar via `--overwrite --cpu-only --direct`):**
```
in:12.8% 55.34 GiB 4.56 GiB/s | out:91.5% 14.95 GiB 1.23 GiB/s
```
The `out:%` jumped to ~90% almost immediately and stayed there for the
duration of the run, while `in:%` ticked up normally.

**Cause.** `compress_cpu_mt` set `meter.total_out_final = true` inside the
producer-done block.  That flag was designed for decompression — where
`total_out` is summed from `frame_header.decomp_size` during the pre-scan
and IS a known final total.  For compress, `total_out` is a *running
accumulator* incremented by the writer-collector at line 3016 as compressed
batches arrive.  Setting `total_out_final = true` makes the progress code
take the "decompress, reader done" branch
(`wrote_bytes / total_out_so_far`), which is just the writer's catch-up
ratio — typically 80–95%.

The GPU compress path (`compress_nvcomp`) correctly leaves
`total_out_final` unset.  Now `compress_cpu_mt` matches that.

**Result.** With the flag unset, the percentage logic falls through to the
frame-level branch (`tasks_done / total_frames`), giving a percentage that
tracks `in:%` instead of jumping to 90% right away.

---

## v0.12.30 — `-M` / `--memlimit` / `--memory` now real flags

Promoted from v0.12.29's warn-no-op set to actual implementations.

**Accepted forms:** `-M N`, `-M N` (joined or separated), `--memlimit N`,
`--memlimit=N`, `--memory N`, `--memory=N` — value is in MiB to match zstd.

**Decompression.** The value is pushed to every `ZSTD_DCtx` via
`ZSTD_d_windowLogMax` with `wlog = floor(log2(N * 1 MiB))`, clamped to the
`[10, 31]` range zstd accepts.  Streams whose frames require a larger window
are rejected with zstd's `Frame requires too much memory for decoding`
error (exit 4 = data error) rather than being allowed to allocate unbounded
memory.  This matches zstd's own semantics for `-M`.

**Compression.** zstd itself ignores `-M` for compress; gzstd uses the
value to tighten the in-flight frame-throttle budget in
`compute_throttle_budget`.  Without `-M`, the RAM cap is
`min(pipeline_parallelism * slack, RAM/2 / frame_bytes)`.  With `-M N`
the cap is lowered to `max(1, N * 1 MiB / frame_bytes)` if that's
smaller, and the throttle source in `-vv` output shows `source=ram`
whenever the user's limit is the binding constraint.

**Not applied to nvCOMP decompression.**  The GPU path allocates its
VRAM buffers through nvCOMP, which has its own memory accounting via
`--gpu-mem-frac` — the host-side `-M` cap doesn't directly apply there.
Frames that fall back to CPU rescue respect the limit through the
worker thread's `tl_dctx`.

---

## v0.12.29 — zstd-compat flag layer

gzstd now accepts the full zstd CLI flag set so it can truly serve as a
drop-in replacement.  Flags fall into four buckets:

**Real aliases** (map to existing gzstd semantics):
`--decompress`, `--uncompress`, `--force`, `--keep`, `--test`, `--verbose`,
`--stdout`, `--to-stdout`, `-H` (long help), `--single-thread` (≡ `-T 1`),
`--fast=#`.

**Silent no-ops** (zstd defaults that gzstd already matches — accepted without
comment): `--asyncio`, `--no-asyncio`, `--check`, `--no-check`,
`--format=zstd`, `--no-dictID`, `--compress-literals`,
`--no-compress-literals`, `--row-match-finder`, `--no-row-match-finder`,
`--mmap-dict`, `--no-mmap-dict`, `--stream-size=…`, `--size-hint=…`,
`--target-compressed-block-size=…`, `--auto-threads=…`.

**Warn no-ops** (zstd features gzstd does not implement — accepted with a
`gzstd: warning: <flag> accepted for zstd compatibility but ignored` line):
`--adapt`, `--long[=#]`, `--patch-from[=REF]`, `--rsyncable`,
`--exclude-compressed`, `--format=gzip|xz|lzma|lz4`, `--pass-through`,
`--no-pass-through`, `-r`/`--recursive`, `-l`/`--list`, `--filelist`,
`--output-dir-flat`, `--output-dir-mirror`, `--trace`, `-D`/`--dict`/
`--dictionary`, `--train`/`--train-*`, `--maxdict`, `--dictID=#`, `-B#`,
zstd benchmark flags (`-b#`/`-e#`/`-i#`/`-S`/`--priority=rt`).

(`-M#` / `--memlimit` / `--memory` started here as warn-no-ops and were
promoted to real flags in v0.12.30 — see that entry.)

The warn stream respects verbosity: `-q` / `-qq` / `--quiet` / `--silent`
suppress the compat warnings.  A pre-scan of `argv` sets the suppression
threshold so the quieting flag can appear in any position.

---

## v0.12.28 — Help split: concise `-h` / `-?`, detailed `--help` with examples

`-h` and the new `-?` alias print a short, grouped option list (Operation /
Output / Compression / Backend / Tuning / I/O / Logging / Misc) intended to
fit on a single terminal screen.  `--help` now prints a long reference with
per-flag descriptions, flag interactions, exit codes, and a block of runnable
examples covering the common workflows (compress, decompress with
`--overwrite`, piped tar, CPU-only baseline, GPU-only tuning,
`--sliding-window`, integrity check, forced progress, stats JSON).

---

## v0.12.27 — `--gpu-batch` is now per-stream on compress (BEHAVIOR CHANGE)

**Symptom.** `--gpu-only --gpu-batch=512 --gpu-streams=4` on the compression path was allocating only **128** subchunks per stream, not 512. The user expected "each stream gets batches of 512."

**Cause.** The compression producer-side batch cap was computed as `ceil(gpu_batch_cap / stream_count)` — treating `--gpu-batch` as a *per-device* total and dividing across streams. The decompression path treated the same flag as *per-stream* (see comment at `decompress_nvcomp`: "kernel launch overhead dominates, so each stream needs large batches"). The help text ("Max GPU subchunks per device") agreed with compress, disagreed with decompress.

**Fix.** Compress now uses `--gpu-batch` as a per-stream cap, matching decompress. With `--gpu-batch=512 --gpu-streams=4`, each of the 4 streams now aims for 512 subchunks. VRAM safety is preserved: `gpu_mem_fraction` is still divided across streams, and the per-stream binary search clamps down when the requested batch doesn't fit.

**Compatibility.** Runs that relied on the old semantics (compress dividing the flag) will see more subchunks in flight and higher VRAM usage. If VRAM is tight the binary search will report "VRAM-fit: batch=N (requested M)" at `-v`. To restore the previous effective per-stream batch, divide the old value by `--gpu-streams` (e.g., old `--gpu-batch=512 --gpu-streams=4` → new `--gpu-batch=128 --gpu-streams=4`).

**Help text updated:** `Max GPU subchunks per CUDA stream (default: 16)`.

---

## v0.12.26 — `--overwrite` (non-atomic) + perf-breakdown reader stats fix

### 1. New `--overwrite` flag

**Symptom (Workstation).** Running `gzstd -d -f big.zst` against a pre-existing output file stalled for tens of seconds at the final rename, while deleting the target first and letting gzstd create a fresh file was fast. v0.12.23 already reduced this stall with `sync_file_range`, but on ext4 with large outputs a substantial rename cost remained.

**Cause.** `-f` always used the `.gzstd.tmp` + `rename()` atomic-overwrite dance, which on ext4 `data=ordered` ties rename commit to flushing dirty pages. For workloads where atomicity isn't worth that cost, users want to opt out.

**Fix.** New `--overwrite` flag (implies `-f`) bypasses the atomic dance: gzstd calls `fopen(target, "wb")` directly, truncating the target in place. No tmp file, no rename. Trade-off: if gzstd is interrupted, the target is partial/corrupt.

- Default `-f` behaviour (atomic) is unchanged.
- Regular-file check still applies (FIFOs, devices, stdout are unaffected).
- The target is still registered with the cleanup handler so `Ctrl-C` removes the half-written file.

### 2. Reader stats showing all zeros under `-vvv`

**Symptom.** The `PERFORMANCE BREAKDOWN` table printed `Reader: 0.000 s (0.00 GiB, 0.00 GiB/s)` for any run that used the default mmap zero-copy reader.

**Cause.** `compress_cpu_mt` and `compress_nvcomp` had two producer paths: a `fread` path (which recorded `read_ns` / `read_bytes_total`) and an `mmap` path (which didn't). For regular files on Linux, gzstd always takes the mmap path, so `PerfCounters` never saw any bytes.

**Fix.** Both mmap producer loops now record `read_bytes_total` (= mapped file size) and `read_ns` (time spent enqueuing view tasks). The timing is small — pointer arithmetic, not I/O — but the bytes column now reflects reality.

---

## v0.12.25 — Compression I/O fixes + GPU D2H timing correction

**Three independent fixes surfaced while investigating why gzstd was 6.5× slower than `zstd -T0` on barely-compressible data and inconsistent across runs.**

### 1. Output `setvbuf` — multi-MiB buffer instead of glibc default

**Symptom.** Compression of large barely-compressible data was dominated by `write()` syscall overhead. A 14.4 MiB `fwrite` was being split into 1800–3600 individual `write()` syscalls by the glibc default ~4–8 KiB FILE buffer.

**Fix.** Set a 1 MiB `_IOFBF` buffer on every output `FILE *` opened by gzstd (both `open_output_atomic` and the two `fopen` call sites in `main`). This collapses the syscall count by ~128–256×.

**Long-standing issue.** This has been present for the entire history of the tool — not a v0.12.x regression. Affected both compression and decompression output paths.

### 2. `sync_file_range` gated behind `progressive_sync_` flag

**Symptom.** v0.12.23 added unconditional `sync_file_range(..., SYNC_FILE_RANGE_WRITE)` in `AsyncWritePool::worker_fn` to fix a 46s `rename()` stall on decompression. This caused a measurable regression on compression: the non-blocking writeback hint created I/O contention with subsequent `fwrite` calls, triggering `balance_dirty_pages` throttling inside the writer thread.

**Root cause.** Compression produces much smaller output than decompression (e.g., 19 GiB in → 2 MiB out for trivially-compressible data), so the writeback hint buys nothing but steals bandwidth from the next `fwrite`.

**Fix.** `AsyncWritePool` now takes a `progressive_sync` bool (default `false`). Only enabled for decompression, where the ordered-journal rename stall is the real concern.

### 3. GPU D2H timing always reporting `0.00 ms` at `-vvv`

**Symptom.** `--gpu-only` compression with `-vvv` consistently reported `d2h=0.00ms` per batch, even though real D2H copies were happening.

**Root cause.** `cudaEventRecord(C.ev_comp_end)` and `cudaEventRecord(C.ev_d2h_end)` were recorded back-to-back in the CUDA stream with no D2H operation between them. The actual D2H copies are synchronous host-side `cudaMemcpy` per chunk, which happen *after* stream completion — the CUDA events couldn't see them.

**Fix.** Replaced CUDA event timing with wall-clock `now_ns()` timing bracketed around the host-side D2H memcpy loop in both the async poll path and the sync drain path. Total time is now correctly computed as `h2d_ms + comp_ms + d2h_ms`. The `ev_d2h_end` event is still created/destroyed for ABI simplicity but is unused.

---

## v0.12.24 — Streaming decompression for oversized frames

**Symptom.** Decompressing single-frame .zst files (e.g., from `zstd` or `gzstd --sliding-window`) showed the `out:` progress bar stuck at 0% for the entire decompression, then jumping to 99.9% at the end. Memory usage spiked to the full decompressed size (e.g., 125 GiB) because the worker allocated one giant buffer for the entire frame.

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

**Result on Workstation** (low_compress.bin.zst → existing file with `-f`):
- Atomic rename: 46,155 ms → ~700 ms (**66× faster**)
- No regression to compression or fresh-file decompression paths

---

## v0.12.22 — `--sliding-window` single-frame compression mode

**Motivation.** For highly repetitive data (e.g., random word lists repeated across 125 GiB), gzstd's multi-frame architecture (8000 independent 16 MiB frames) achieved 0.29% ratio while `zstd` achieved 0.01% (31× better). The difference: zstd produces a single frame with a 2 MiB sliding window that maintains context across the entire file, while gzstd's frames each start with a cold window.

**Feature.** New `--sliding-window` flag delegates compression to zstd's built-in multi-threaded mode (`ZSTD_c_nbWorkers`), producing a single standard zstd frame. Trade-offs:
- Ratio matches `zstd` exactly (shared sliding window context)
- Output is a standard .zst file — `zstd -d` can decompress it
- Decompression is single-threaded (one frame = one unit of work for any decompressor)
- Implies `--cpu-only` (GPU/nvCOMP has no sliding window API)

**Validation:**
- `--sliding-window --gpu-only` and `--sliding-window --hybrid` rejected with clear error
- `--sliding-window -d` rejected (compression-only)
- `--sliding-window` without `--cpu-only` auto-enables it with a warning
- Round-trip verified; `zstd --list` confirms single frame; `zstd -d` interop confirmed

**GPU fallback for oversized frames.** When decompressing a single-frame file (from `zstd` or `--sliding-window`), the frame's decompressed size can be hundreds of GiB — far exceeding nvCOMP's 16 MiB per-slot VRAM allocation. gzstd now detects oversized frames during the pre-scan and automatically falls back to CPU-only decompression with a clear warning. `--gpu-only` is gracefully overridden rather than crashing.

**Progress bar fix (mmap compression).** The mmap zero-copy reader enqueued all tasks instantly (pointer arithmetic, no I/O), causing the `in:` progress to jump to 100% immediately. Fixed by deferring `read_bytes` updates to when workers actually pick up each task, so the progress bar reflects real processing throughput.

**Test coverage.** Full `./gzstd-test.sh` suite passes (200/200).

---

## v0.12.21 — mmap zero-copy compression input + benchmark accuracy fix

**Symptom (compression).** CPU-only compression of mixed.bin (19.5 GiB) took 9.9s vs zstd's 6.1s. Profiling showed the single-threaded `fread` producer was the bottleneck — 22 worker threads were starved, achieving only ~1.5 effective cores of utilization.

**Fix.** Memory-map input files for both CPU (`compress_cpu_mt`) and GPU (`compress_nvcomp`) compression paths. Workers read directly from the mapped pages via `view_ptr`/`view_len` on `Task`, eliminating the `fread` + `memcpy` bottleneck. Pipes and stdin fall back to the existing `fread` path. Key changes:

- Added `MmapRegion` RAII class (read-only mmap with `MADV_SEQUENTIAL`)
- Extended `Task` struct with `view_ptr`/`view_len` for borrowed (mmap) data vs owned `std::vector<char> data`, plus `ptr()`, `len()`, `release_input()` helpers
- Updated all consumer touch points: `t.data.data()` → `t.ptr()`, `t.data.size()` → `t.len()`, `std::vector<char>().swap(t.data)` → `t.release_input()`

**Result on Workstation** (24-core, mixed.bin CPU-only compress):
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

**Result on Workstation** (24-core, 2× RTX 2080 Ti, medium_compress.bin.zst → real file):
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

**Result on Workstation** (24-core, ext4/NVMe, `mixed.bin` 19.5 GiB, 5-run median):

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

**Motivation.** v0.12.10–0.12.15 fixed several pipeline-depth and throttle issues but left a ~1.7× run-to-run variance on CPU-only decompress at high thread counts (22 workers on Workstation): fast runs ~4.0 s to `/dev/null`, slow runs ~7.0 s on the same cached input. Reducing `-T` from 22 to 4 collapsed both the variance and the absolute time (3.1–3.6 s). Variance scaled with worker count — a contention signature, not a hardware one.

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

**Motivation:** v0.12.13 fixed the lost-backpressure bug by capping the throttle budget at a hard 8 GiB. That number worked on the two test systems but was arbitrary: too restrictive for a 256-core / 8×H100 server whose pipeline can legitimately hold hundreds of GiB in flight, too generous for a 16 GiB VM where 8 GiB is half of physical RAM. The budget needs to track the machine, not a magic constant.

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
| Workstation (24 CPU, 2×1×16 GPU)  | 24 + 32 = 56            | 224      | ~3.5 GiB  |
| Server (256 CPU, 8×2×64 GPU)    | 256 + 1024 = 1280       | 5120     | ~80 GiB   |

On Workstation this is ~2× tighter than the old 8 GiB cap but well above the ~320 MiB the writer actually drains before the next producer wakeup, so no throughput regression is expected. On Server it unlocks the pipeline the hardware can actually sustain.

The `-vvv` throttle debug line now shows all inputs: `parallelism=`, `pipeline=` (pre-clamp), `ram_cap=`, plus the chosen frame count and in-flight byte equivalent.

---

## v0.12.13 — Throttle budget byte cap (restore writer backpressure)

**Bug:** On Workstation (256 GiB RAM) decompression appeared to lose all writer backpressure. Reader, GPUs, and CPUs finished in seconds; the writer then ground through a massive in-RAM backlog at ~88 MiB/s with no throttling of producers. The throttle budget formula `avail_ram / (2 × frame_bytes)` gave ~7,800 frames on a 246 GiB-available box — 123 GiB of permitted in-flight data. Files under that size (mixed.bin.zst at ~20 GiB decompressed = 1,220 frames) fit entirely within the budget, so workers never blocked, decompressed everything immediately, and the writer drained alone.

**Fix:** Cap the budget at an absolute byte ceiling in addition to the RAM-relative calculation:

```cpp
budget_bytes = min(avail_ram / 2, THROTTLE_MAX_BYTES);   // 8 GiB cap
frames = budget_bytes / frame_bytes;
frames = max(frames, 32);                                // min pipeline depth
```

On Workstation: budget = 512 frames (8 GiB in-flight) instead of ~7,800. Workers fill the pipeline, then block on `acquire(1)` — writer releases permits as frames are written, producers resume. Lockstep backpressure restored.

Fast-I/O systems are unaffected: when the writer can release permits faster than workers acquire them, nobody blocks. Only slow-I/O systems (relative to producer throughput) feel the cap — which is exactly where it's needed.

Prior floor of 1024 was also wrong on low-RAM systems (forced a minimum 16 GiB in-flight on 24 GiB boxes, would have caused swap once the reader's buffer was resident). Floor is now 32 frames.

**Debug log change:** `-vvv` throttle message now reports both the frame count and the byte in-flight max, and the available-RAM figure:
```
throttle: 512 frame budget (8.00 GiB in-flight max, 246.03 GiB avail RAM)
```

---

## v0.12.12 — EMA-scaled hybrid queue floor + tuning knobs

**Bug:** The fixed queue floor introduced in v0.12.9 — `active_gpu_streams × gpu_batch_size` — assumed GPU was strictly faster than CPU per frame. For compression that holds (GPU batches pay off), but for decompression nvCOMP throughput ≈ CPU zstd throughput, so reserving frames for the GPU just idles CPUs. Workstation benchmarks (v0.12.11) showed hybrid decompression 2–8% slower than the best pure path on every file, and 18% slower than either pure config on `zeros.bin` (3.675 vs 4.482 GiB/s).

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

**Bug:** On Server, hybrid mode was ~10% slower than `--gpu-only`. Scheduler stats showed CPUs took 18,344 tasks vs GPUs 9,341 — despite CPUs being ~6× slower per task (0.21 vs 1.19 GiB/s). The `should_cpu_take()` gate only blocked CPUs when `gpus_waiting > 0`, but GPUs cycle through wants→got in microseconds. During the much longer GPU processing phase (milliseconds), `gpus_waiting == 0` and all 96 CPUs flooded the queue, leaving it empty when GPUs came back for their next batch.

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
- **Server:** Writer I/O improved 1.1 → 2.72 GiB/s on 432 GiB file
- **Why it works:** Avoids double-buffering through page cache for large sequential writes
- **Caveat:** Unaligned tail requires dropping O_DIRECT via fcntl for final write

### pwrite for Out-of-Order Decompress (v0.9.72)  NEGATIVE (reverted)
Tried using pwrite() to write decompressed frames directly to their final offset without waiting for in-order delivery.
- **Server:** 0.93 GiB/s (worse than sequential 2.72 GiB/s)
- **Why it failed:** 27k individual O_DIRECT pwrite calls = massive kernel DMA setup overhead. sys time: 12m45s.
- **Lesson:** O_DIRECT pwrite per-frame is catastrophically expensive. Sequential batch drain is better.

### Async Double-Buffered Write Pool (v0.9.73)  POSITIVE
Background write thread with one pending slot. Writer collects batch → submits to pool (non-blocking) → collects next batch while pool writes previous.
- **Server:** Improved overlap between GPU D2H and disk writes
- **Why it works:** Writer thread doesn't block on disk I/O; can collect next batch while previous is being written

### Sparse File Support (v0.9.73)  POSITIVE (for zero-heavy data)
Scans 4K blocks for zeros, lseek past them instead of writing. Integrated with both O_DIRECT (DirectWriter::seek_forward) and fwrite paths.
- **Server:** zeros.bin decompress: sparse=5.2s vs no-sparse=6.9s (~25% faster)
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
- **Server:** GPU decompress nearly doubled: 13.4s → 25.6s
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
- **Workstation (10 GiB VRAM):** Finds batch=104 instead of hanging on batch=256
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
- **Server:** zeros.bin: CPU path 1.4s vs GPU path 4.4s

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
- **Server:** medium_compress kernel dropped 24.7s → 1.27s (55× speedup!)
- **Why:** Default batch=8 caused 64 kernel launches × 385ms each. Batch=256 = 3 launches × 424ms.

### Continuous Binary-Search Auto-Tuner (v0.10.0-0.10.6)  POSITIVE
Runtime throughput-aware batch sizing for compress. Explores both directions from default.
1. Record baseline throughput at starting batch size
2. Try halving  if better, continue halving
3. Try doubling  if better, continue doubling
4. Settle at best when throughput drops
5. Periodically probe to detect data character changes
- **Workstation:** Correctly finds batch=8 optimal for compress, settles in 2 steps
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

### Server (H100 × 8)  v0.9.74 vs zstd -T0, 8 GiB files, decompress
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| zeros | 4.85s | 4.40s | 1.10× |
| high_compress | 9.52s | 7.07s | **1.35×** |
| medium_compress | 15.39s | 9.69s | **1.59×** |
| mixed | 9.12s | 6.55s | **1.39×** |
| low_compress | 9.25s | 7.14s | **1.30×** |
| **Total** | **48.13s** | **34.85s** | **1.38×** |

### Workstation (RTX 2080 Ti × 2)  v0.10.6 vs zstd -T0, 8 GiB files
**Decompress:** gzstd wins 2/5 (medium_compress 1.22×, low_compress 1.06×). Loses on trivial data where zstd's page-cache sparse dominates.
**Compress:** gzstd wins 4/5 (high 1.83×, low 1.54×, medium 1.11×, mixed 1.26×). Only loses zeros.

---

### io_uring Writer (v0.10.22-0.10.28)  NEGATIVE (reverted)
Replaced DirectWriter + AsyncWritePool with Linux io_uring for async writes.
- **v0.10.22-26:** O_DIRECT + io_uring. Writes submitted but never completed  `io_uring_wait_cqe` hung forever. Likely kernel/NVMe driver incompatibility with O_DIRECT + io_uring on Server.
- **v0.10.27:** Tried `io_uring_submit_and_wait()`  still hung.
- **v0.10.28:** Dropped O_DIRECT, tried buffered io_uring  still hung.
- **Root cause:** Unknown kernel-level issue. io_uring write completions never arrived despite successful submission. Possibly a kernel config, seccomp policy, or filesystem limitation.
- **Decision:** Reverted to DirectWriter + AsyncWritePool.

### Multi-threaded pwrite Pool (v0.10.29)  NEGATIVE (reverted)
4 threads doing pwrite() at known offsets through the page cache.
- **Server:** 10m30s (vs 4m with DirectWriter). `sys: 38m40s` (vs 12m).
- **Why it failed:** Without O_DIRECT, 432 GiB went through the page cache. The pwrite() calls returned fast (page cache absorb), but kernel writeback stalled massively. The page cache backlog created 9.5 minutes of post-completion flush.
- **Key lesson:** You cannot beat the NVMe's physical write speed (~2-3 GiB/s on Server). O_DIRECT + single-threaded sequential write is already optimal for this workload. The 220s writer drain IS the hardware limit  not a software bottleneck.
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

### Server (H100 × 2 GPUs)  v0.10.34, 432 GiB file (rpfrancis.tar)

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

### Workstation (RTX 2080 Ti × 2)  v0.10.6, 8 GiB files

**Compress: gzstd wins 4/5 vs zstd -T0**
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| high_compress | 7.23s | 3.96s | **1.83×** |
| low_compress | 5.85s | 3.80s | **1.54×** |
| medium_compress | 4.33s | 3.91s | **1.11×** |
| mixed | 4.28s | 3.40s | **1.26×** |
| zeros | 2.47s | 3.88s | 0.64× |

**Decompress: gzstd wins 2/5 (storage-limited on consumer NVMe)**

### Workstation (RTX 2080 Ti × 2)  v0.11.20, 8 GiB files, 3 iterations

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

CPU wins decompress across the board on Workstation. PCIe Gen3 bandwidth makes D2H
the bottleneck  the GPU can't transfer decompressed data back fast enough to
justify the round-trip. Trivial frame detection helps (zeros at 4.88 GiB/s).
Confirms that asymmetric mode (GPU compress + CPU decompress) would be the
ideal default for consumer GPUs with PCIe Gen3.

### Workstation (RTX 2080 Ti × 2)  v0.11.22, 8 GiB files, 3 iterations

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

7. **io_uring may not work on all kernels.** Server's kernel accepted io_uring submissions but never completed writes. Possibly a seccomp policy, kernel config, or NVMe driver limitation. Always have a fallback.

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
- **Why it failed on Server:** Students had ALL cores at 97-99%. Pinning forced I/O threads onto busy cores instead of letting the OS scheduler find idle moments on any core.
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
- **Workstation (8 GiB files):** No measurable throughput change on small workloads (127 frames  sleep overhead was ~2% of runtime). The win is on large files with thousands of frames where 22 threads × 1ms × thousands of iterations compounds to minutes of waste.
- **Bug fixed during development:** Initial implementation deadlocked because predicate lambdas called `tq->peek_front_ratio()` / `tq->size()` / `tq->drained()` while `pop_one_cpu` held `m_` (non-recursive mutex). Fixed by passing a `QueueState` snapshot to predicates instead.
- **Bug fixed during development:** `push()` with `cv_.notify_one()` could deliver notifications to a GPU that was busy processing (not waiting), starving the other GPU. Changed to `cv_.notify_all()`.

### Early Memory Release (v0.11.22)  POSITIVE
Release input data buffers immediately after they're consumed instead of holding them until end of processing cycle. Reduces peak memory by freeing frames as soon as they're no longer needed.
- **CPU compress worker:** `t.data` (up to 32 MiB) released via swap immediately after `compress_one_cpu_frame`. Previously held through logging, stats, and result delivery.
- **GPU compress worker:** Batch input data (up to 16 × 16 MiB = 256 MiB) released after H2D upload. Guarded by `!rescue`  in hybrid mode data stays alive for potential CPU rescue on GPU failure; in gpu-only mode released immediately.
- **GPU decompress worker:** Batch compressed data released before kernel launch (after re-upload path). Saved `batch_seqs[]` and `batch_comp_sizes[]` for completion paths.
- **CPU decompress worker:** Already had early release (swap at line 2280)  no change needed.
- **Workstation (8 GiB files):** +7.1% decompress and +7.5% compress on mixed.bin. Other data types flat (bottlenecked elsewhere). Mixed data benefits most because alternating compressible/random blocks cause high frame churn  freeing memory sooner reduces page allocation contention.

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

**Server (H100 × 8, 432 GiB file) hybrid decompress:**

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

**Server (432 GiB decompress via stdin redirect):**

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
- **Server (432 GiB verify):** 1 stream: 4.09 GiB/s (1m47s) → 2 streams: 6.39 GiB/s (1m09s)  **56% faster**
- Compress/decompress stays at 1 stream (NVMe is the bottleneck, larger batches win)
- Fixed help text: was incorrectly showing default as 3

## Key Lessons Learned (Updated)

19. **Stdout O_DIRECT is a free 2.5× win.** Users don't think about O_DIRECT  they just write `> file`. Detecting stdout-to-file and auto-enabling O_DIRECT gives them NVMe speed without any knowledge of I/O internals. The page cache adds 17 GiB of dirty pages and halves throughput on 432 GiB files.

20. **GPUs need backpressure too.** The original design exempted GPUs ("batches in-flight, can't throttle"). But on H100 × 8, GPU decompression throughput vastly exceeds NVMe write speed. The result: 300+ GiB buffered in RAM, massive kernel writeback, and a frozen "28% writing" progress bar after decompression finished. Throttling GPUs before their next `pop_batch_greedy`  not mid-kernel  keeps the pipeline balanced with <1s drain.

21. **Bound frames, not bytes.** Byte-based backpressure conflated two concerns: memory pressure (bytes in RAM) and frame ordering (which frames the writer can drain). Out-of-order frames in ResultStore inflated the byte count, triggering false backpressure. A counting semaphore on frames separates these concerns: frame ordering lives in ResultStore, flow control lives in the semaphore. The FIFO queue guarantees the writer's next-needed frame is always in-flight, making the design deadlock-free without escape hatches.

18. **GPU VRAM is a shared resource  design for it.** On a multi-user machine (Server, 8× H100), any GPU can lose VRAM at any moment. Infinite retry loops, early reader aborts on single GPU failure, and missing frame deadlocks all surfaced under real student workloads. The fix: retry limits, graceful skip with re-enqueue, deadlock detection with hard error, and never abort the reader on partial failure.

---

## Early Benchmark History (Server, v0.9.50v0.9.59)

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
