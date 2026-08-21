# gzstd v1.0 Roadmap & Battle Plan

**Current version:** v0.17.2
**Target:** v1.0  production-ready hybrid CPU+GPU Zstd with intelligent scheduling

---

## Guiding goal: a faithful drop-in replacement for the zstd CLI

**gzstd strives to be a drop-in replacement for the `zstd` CLI. This is a standing,
ongoing commitment, not a finished state** — where gzstd and zstd disagree, gzstd is
the one that is wrong, and closing the gap is legitimate work at any time.

What that commits us to, in priority order:

1. **Read anything zstd writes.** A file zstd produces must decompress correctly, or fail
   loudly. This is the one rule with no acceptable exceptions.
2. **Accept zstd's spelling.** Every option form — short, long, bundled, `=`-attached,
   space-separated — should parse the way zstd parses it.
3. **Mean the same thing.** An option that is accepted must either do what zstd does or
   say plainly that it did not. Silent divergence is the worst outcome and is treated as a
   defect, not a wart.
4. **Match the observable contract.** Exit codes, stdout/stderr split, and `-q` behaviour
   should be indistinguishable in scripts.

Where full fidelity is not yet implemented, the help must say so — see the
COMPATIBILITY OPTIONS section of `--help`, which classifies every accepted option as
mapped, partially mapped, silently ignored, or warn-and-ignored. **An unqualified
"drop-in-compatible replacement" claim in the help was itself a defect** (v0.15.69): it
was true of the common path and false of roughly thirty options.

### Measured state (2026-08-06, against zstd 1.5.7)

- **Parser:** of zstd's 35 advertised long options, 5 are rejected in their bare form and
  all 5 work as `--opt=VALUE`. 143 option spellings accepted overall. Bundled short flags,
  gzip aliases, `-0`, `--single-thread` and `-M` all work.
- **Decompression fidelity:** verified byte-identical round-trips of zstd-produced output
  for `-19`, `--long=27`, `--no-check` and `--rsyncable`.
- **The one hard incompatibility — dictionaries.** `gzstd -d -D dict file.zst` exits 4 on a
  stream zstd reads fine, and `-D` on compress warns and compresses without the dictionary.
  Anything using `-D`, `--train*`, `--patch-from` or `--maxdict*` is not portable to gzstd.
  **This is the highest-value compatibility work outstanding.**
- **Accepted but divergent** (each warns): `-r`/`--recursive` compresses nothing (exit 3
  since v0.15.69 — it used to exit 0 having done nothing), `--format=gzip` emits zstd,
  `--output-dir-flat`/`--output-dir-mirror` write nothing, `--no-check` still writes the
  checksum.
- **Accepted silently, no effect:** `--[no-]asyncio`, `--[no-]check`, `--no-dictID`,
  `--[no-]compress-literals`, `--[no-]row-match-finder`, `--[no-]mmap-dict`,
  `--stream-size=`, `--size-hint=`, `--target-compressed-block-size=`, `--auto-threads=`.

### Ordered gap list

| Gap | Size | Note |
|---|---|---|
| Dictionary support (`-D`, `--train*`, `--patch-from`, `--maxdict*`) | Large | The only correctness-class gap: gzstd cannot READ dict-compressed streams |
| `-r` / `--output-dir-flat` / `--output-dir-mirror` | Small | Self-contained file-walking and output-path mapping |
| `--long`, `--rsyncable`, `--no-check` | Small | Thin wrappers over zstd parameters gzstd already sets |
| `--format=gzip\|xz\|lzma\|lz4` | High | Needs zlib/liblzma/liblz4; see Phase 8 |
| `-b#`/`-e#`/`-i#` benchmark mode | Medium | Overlaps `--calibrate`; decide whether to map or keep separate |

**Re-measure this table before each tag** — it is the only part of the roadmap that
describes a moving target we do not control.

---

## Future security enhancement: content-identity `--rm` (deferred, not a defect)

`--rm` and `--tar --rm` refuse to remove a source whose **size or mtime** changed after its
bytes were compressed. That is a *stat-visible* stability guarantee, and it has a known blind
spot: a same-size rewrite that leaves the reported mtime unchanged — a deliberate restore, or a
filesystem whose timestamp granularity is too coarse to notice — passes the check, so bytes the
archive does not contain can be deleted at exit 0.

**This is deferred rather than fixed, because it is GNU tar's contract for this flag and we measured it.**
Against tar 1.35, a 1.5 GB source modified in place during the read:

| case | GNU tar 1.35 | gzstd |
|---|---|---|
| modified mid-read, mtime advances | `file changed as we read it`, exit 1, source kept | refused, exit 3, source kept, entry named |
| same-size rewrite, mtime restored | **exit 0, source removed, no warning** | same |

tar uses the same size+mtime test and has the same blind spot. Closing it in gzstd alone would
make `--rm` *diverge* from the reference behaviour GNU tar defines for this flag, and an ordinary concurrent writer
advances mtime and is already caught — the residual needs a write that deliberately hides.

**If it is ever taken, the shape is known** (proposed by the independent reviewer, round 24, and
worth recording because it is the right design):

1. **One `ArchivedSnapshot` abstraction** consumed by all four consumers — layout, sparse
   probing, reader/descriptor reuse, and removal. The recurring defect in this area has been one
   representation omitted from a rule the others share (ordinary, empty, PAX sparse, **wholly
   sparse OLDGNU**, hardlink). A single abstraction makes omission a compile-time question
   instead of a review-time one.
2. **A fingerprint of the exact archived bytes**, taken during the compress read (which already
   touches every byte, so it is one hash pass, not an extra read), and re-verified at removal.
   **Cost: a full re-read of every source at removal time** — this is the real price, and it
   roughly doubles read I/O for `--tar --rm` over a large tree. It should therefore be **opt-in**
   (`--rm-verify` or similar), not the default, precisely because the default must stay tar's.
3. **Mutation-tested per representation through deterministic gates**, not through races — every
   one of the five representations above, each proven to fail when its own check is removed.

Until then the bound is documented in `--help` and in the code, and the promise is stated as
stat-visible stability rather than content identity.

---

## ACTIVE: GPU staging — the drain thread is the per-device serializer

**The GPU hardware is ~3.7× the CPU and the pipeline delivers 4.2% of it.** Kernels are
**71.8 GiB/s aggregate** against **19.34** for all 256 cores; `--gpu-only` delivers **3.0**.
Nothing is wrong with nvCOMP — everything is wrong with the host staging around it.

Per-worker phase accounting (`-vvv`, v0.16.1) localised it in one line: workers block
**75–82% on `wait_idle_stream`**, with **zero** time waiting for input. One `drainer` thread per
device holds the `StreamCtx` ~100 ms/batch against ~25 ms of submit, so extra streams cannot help
(`--gpu-streams=2` changes nothing — they funnel through the same drain).

**Next change: async D2H directly into the frame buffers** — `cudaHostRegister` over the
`out_pool`, so the readback is page-locked *and* copy-free.

- **Do not use a staging slab.** Tried at v0.16.2, **26% slower** (3.25 → 2.40), reverted; see the
  CHANGELOG. It adds a ~160 MiB host memcpy per batch, and *any design that adds a host-side copy
  loses here* — the transfer was never the expensive part.
- **The hard part is ownership, not CUDA.** `FrameVec` buffers are `shared_ptr`s held by the
  writer for arbitrary time, and any `resize()` past capacity reallocates and **silently
  invalidates the registration**. Every pooled buffer must be reserved to `max_out_chunk + 4`
  once and never grown — enforced structurally, not by a comment saying "don't grow this".
- **Benchmark against RAM sources/sinks on this host.** Its NVMe reads at 4.56 GiB/s and
  `--cpu-only` already hits 98.5% of that, so a storage-to-storage run here cannot *measure* the
  win. That is a limit of the box, not of the work.

Also still unbound: `--acls/--xattrs` metadata gathering reopens the path, and the H2D content
checksum is now parallelised but still host-side (a GPU-side XXH64 is awkward — the algorithm is
sequential across stripes, so only ~4 lanes × chunks of parallelism).

## Tooling gap: `gzstd-benchmark.sh` has NO `--tar` coverage

The benchmark sweeps plain compress/decompress across batch sizes, streams, threads and levels.
It does not invoke `--tar` **at all** — zero occurrences in the script. So the entire tar path is
unbenchmarked: archive creation, the assembly reader, the parallel extract path, and seek-extract.

**Why this matters more than it looks.** Most of the correctness work in v0.14.x–v0.15.x landed in
exactly that path, including per-member identity validation added to the assembly reader's hot
loop in v0.15.95–97. When performance was finally checked after that arc, the benchmark could not
answer the question — it exercises none of the changed code — and the check had to be done with a
hand-built A/B against a `git worktree` of the previous release. That is not repeatable and it is
not recorded anywhere the next person will look.

What it should cover, in rough priority:

1. **`--tar` create** on two corpora, because they stress different things and the first one
   hides regressions in the second: a few large members (throughput-bound, ~12 GiB) and *many
   small* members (per-member-bound, ~40k files). The v0.15.97 check found no regression on
   either, but only the second could have detected a per-member cost at all.
2. **`-d --tar` extract**, both the parallel and the serial fallback paths.
3. **Seek-extract** (`-d --tar ARCHIVE MEMBER...`) against an indexed archive — the one path with
   a completely different cost model, since it preads only the frames a selection touches.
4. Compare against `tar --zstd` for the same reason the `-l` output is compared against
   `tar -tvf`: parity claims need a measured baseline, not an assertion.

Until this exists, **any statement about tar performance is unmeasured** — the suite proves
correctness there and nothing proves speed.

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

> **⚠️ That parenthesis was false for four years, and this is the sentence that hid it.**
> "FIFO queue guarantees the writer's next frame is always in-flight" was true only while a
> *single* reader existed to make it true for free — it was asserted here, in
> `TaskQueue::re_enqueue`, and at two output pools, and enforced nowhere. The multi-reader
> pooled reader broke it three separate ways, all of which wedged permanently
> (v0.15.66–v0.15.67; see CHANGELOG). It now holds because it is *enforced*: `TaskQueue::push`
> inserts sorted by seq, the four per-worker output pools overdraft instead of blocking, and
> `FrameThrottle::acquire_or_overdraft` lets the writer's next-needed frame past an exhausted
> budget — because a reader can claim a chunk index before the frame reaches the queue at all,
> which no amount of queue ordering can fix. **Do not restate the guarantee without saying
> which mechanism enforces it.**

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
  **⚠️ "pop order = seq order" is another statement of the premise annotated at the
  `FrameThrottle` entry above, and it needs the same caveat**: it holds only *within* one
  device's FIFO. The writer needs the global minimum unwritten seq, which may sit in another
  stream, in a per-worker pool, or — as v0.15.67 found — not yet in the queue at all.
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

### 2.4 Chunk-Size and GPU-Stream Priors (`--adapt` levers to explore)
**Priority: Medium | Complexity: Medium | Status: NOT STARTED (proposed 2026-07-17)**

The two most consequential knobs the v0.15.x governor does NOT touch:

- **Chunk size** (`--chunk-size`, static 16 MiB): feeds the throttle math,
  per-frame overhead, GPU batch geometry, and the parallelism/latency
  trade — the right value plausibly differs between a 24-core Gen3 box and
  a 256-core Gen4+ one. Candidate design: measure per-chunk overhead vs
  throughput in `--calibrate` (compress a fixed corpus at 2-3 chunk sizes),
  persist a `chunk_mib` prior per direction, seed the default from it
  (user-set `--chunk-size` always wins). Caveat to design around: chunk
  size changes the OUTPUT frame geometry, so unlike pure scheduling levers
  it is observable in the artifact — the prior must never override
  compatibility expectations (e.g. keep the default for piped/unknown-size
  runs, document that archives from the same box may differ across
  calibrations).
- **GPU streams** (`--gpu-streams`, static per-device): the natural
  companion to the persisted `settled_batch` — same pattern: let a probe
  (or `--calibrate`) measure 1 vs 2 streams per device, persist
  `settled_streams`, seed with tuner freedom. Interacts with the
  v0.14.58 permit-hoarding fix and per-stream VRAM footprint (the
  `--verify` VRAM doubling), so the probe must respect the existing
  batch-floor deadlock guardrail.

Both slot into the existing profile grammar (EMA or latest-wins discrete,
driver-quarantine only for the GPU one) and the established review gates.

### 2.5 `--adapt` × `--tar` Integration (proposed 2026-07-17)
**Priority: Medium | Complexity: Medium | Status: IN PROGRESS — #1 signal DONE v0.15.11, #1 ACTUATOR DONE v0.15.12; the rest are the governor's remaining blind spots**

**Architecture note — UPDATED v0.15.24 (the decouple changed this).** The
original finding was that only the writer POOL was a clean in-run probe target and
`run_parallel`'s decode/read were fused and fixed-at-start. That is no longer true:
v0.15.20–22 made decode a live-growable pool (the shared decode queue), and v0.15.24
split reading from decoding into two independently-scalable stages (reader pool →
`ddq` → decoders + GPU). So the extract now has THREE in-run levers — reader `R`,
decoder `D`, writer `W` — all shared job queues with spawn/reap. `run_sink` still
rides the streaming `decompress_cpu_mt` pipeline (already read/decode-decoupled) and
its parse is serial by design; the FrameSink budget stays tar-exempt.

**Bottleneck-aware unified controller (Phase 3, part 2 — DONE v0.15.25; regression found + fixed v0.15.26).**
Landed as START-HIGH / CONTRACT rather than grow-from-zero: reader + decoder pools come
up fully provisioned at t=0 and the controller RETIRES workers of any stage whose input
queue sits empty (over-provisioned), converging to just enough per stage. Rationale:
grow can't adapt in time for short jobs, and the reader (first in the pipe) starves
everything downstream if slow to ramp; idle pool workers block on their input queue
(≈ free), so over-provisioning is cheap. Asymmetric by physics — R/D start high
(no contention cliff), W stays moderate + grow-probe (fs journal/dentry contention past
~16), GPU stays lazy (fixed cuInit/VRAM). No explicit `N+R+D+W ≤ cores` math: each stage
blocks on its input queue so only the bottleneck stage is CPU-runnable — the budget is
EMERGENT. Also structurally removed the bootstrap deadlock a reactive grow had (a
decoupled pipe needs ≥1 reader AND ≥1 decoder to flow). Validated byte-identical + suite
346 (sink-bound: started 96+96 → settled 1+1). **v0.15.26 found the gap the hard way:**
the "keep the bottleneck stage high" case IS reachable here — an incompressible archive
extracted across two NVMes is write-limited, and v0.15.25's queue-depth contraction
collapsed R+D to 1+1 and starved the writers (a 28% regression vs inline). Fixed by (1)
routing trivial-decode archives to fused inline (the decouple only adds handoff cost when
decode is free), and (2) making the pool's contraction keep-or-revert on measured
end-to-end RATE (the writer-starved ratio is blind to R/D — it pins high whenever writers
outnumber a fast sink; and rate must be integrated over ~0.8 s because the sink drains in
bursts). **STILL OWED:** a genuinely CPU-poor box for the few-huge-file incompressible
case (Np≈2), where a reader-count lever would beat inline — the one scenario the ratio
gate leaves on the table.

The v0.15.x governor runs on `--tar` operations but several of its senses
and levers don't reach the tar-specific machinery (each was an explicit,
documented deferral during M4):

- **Extract sink is invisible (the big one): DONE v0.15.11 — signal only.**
  `-d --tar`'s parallel file-writer pool now feeds the governor its busy/
  starved time via separate Meter fields (`extract_busy_ns`/`extract_starved_ns`
  /`writer_pool_threads`), flushed live in coarse ~50 ms deltas (the pool's own
  atomics still flush once-at-exit for the `-v` line), so `SINK_BOUND` finally
  classifies on extract. Two subtleties the original framing missed: the
  counters were `-v`-gated (now `verbose || adapt`) and flushed once-at-exit
  (wrong shape for a 100 ms delta tick). The classifier takes
  `max(disk-term, extract-term/writer_pool_threads)`. The writer-probe (action
  5b) had to be **gated off extract** (`is_extract_`): its actuator is the plain
  DirectWriter's second drain thread, absent on extract, so it would misreport a
  phantom action and persist a bogus verdict.
  **ACTUATOR — DONE v0.15.12 (action 5c).** SINK_BOUND extract now GROWS the
  writer pool: a supervisor thread (armed only under `--adapt`) spawns/reaps
  extra writers on the shared job queue as the governor moves
  `g_adapt_ewgrow_target` (woken via a global CV — the only governor→pool
  channel, no pointer). Steps by half the base pool, EMA-baselined keep/revert,
  caps at 2 rounds / doubled pool, persists `tar_write_threads` (+ `converged`
  latch) to seed the next run. `--write-threads` (user pin) always wins. Found a
  real +26% (16→24) optimum live on the 256-core box.
  **BIDIRECTIONAL + SHAPE-KEYED — DONE v0.15.27/v0.15.28.** Grow-only turned out
  to be half a controller: a prior learned on one archive shape mis-sized the
  other (60 writers = 92.8% busy on 390 K small files, 11.2% busy / 84.0%
  starved and ~9% slower on 13 huge ones), and it could not be walked back
  because the prior was baked into the base pool, which has no retire path — so
  `--adapt` extract ran ~7% SLOWER than the plain default on trivial-decode
  archives. Now: the prior seeds as retirable EXTRAS above the auto base; 5c
  contracts on the writers' own busy/starved split (NOT gated on `SINK_BOUND` —
  an over-provisioned extract classifies `COMPUTE_BOUND`, because the surplus
  drags the per-thread busy average under the sink threshold); every step is
  keep-or-revert on rate integrated over ~0.8 s; and the settled size persists
  per (machine, archive SHAPE), bucketed by mean bytes per entry, so alternating
  shapes no longer re-teach one shared number. The converged latch was already
  removed in v0.15.24, so the earlier "periodic re-probe" follow-up is moot.
  `--write-threads` now suppresses the sizing supervisor entirely (it previously
  won only for the base pool, letting the probe override the pin).
  Follow-up: positive-perf validation on the Gen<4 workstation; the 3-bucket
  shape split (64 KiB / 4 MiB) is measured only at its extremes.
- **DONE v0.15.31 — the lazy-engagement guard on COMPRESS.** `compress_nvcomp`
  called `cudaGetDeviceCount` first thing, charging ~5.4 s of cuInit before any
  work: 512 MB compress took 5.51 s against 0.26 s cpu-only. Detection now runs on
  a background bringup thread that owns detect + select + per-device sizing +
  worker spawn, so the main thread proceeds to the reader. Adaptive hybrid only —
  `--cpu-share` keeps the synchronous bringup (v0.13.11) and `--gpu-only` keeps a
  clean no-device error. **512 MB 5.51 -> 0.25 s, 5.3 GB 5.79 -> 1.52 s.**
  THE TRAP, if this is ever revisited: compress cannot use read progress. With an
  mmap'd input the reader finishes at t~0 ("producer_done fires at t~0 with
  mmap"), so read progress reads ~100% while all the compression is still to do,
  AND the main thread reaches teardown while the pool still has seconds of work —
  so a teardown-tied stop cuts the sample short instantly. The first working build
  skipped the GPU on every compress for exactly this reason. It measures work
  COMPLETED (frames handed to the writer, scaled to input bytes) instead.
- **GPU decoders in the extract decode pool: Phase 1 DONE v0.15.22, Phase 2 DONE
  v0.15.23.** The v0.15.20 decode pool was CPU-only; GPU-stream decoders now
  batch-drain the same shared frame queue and scatter into the same reorder
  buffers alongside the CPU decoders (oversize/failed frames and stream faults
  fall back to `decode_seek_frame`; CPU decoders are the guaranteed backstop, so
  correctness never depends on the GPU). **Phase 1** = the machinery + `GZSTD_POOL_GPU`
  eager spawn (correctness). **Phase 2** = they engage AUTOMATICALLY under `--adapt`:
  the controller spawns them lazily once the CPU pool is maxed AND still
  decode-starved, but only if the estimated remaining extract time outlasts cuInit
  (~4 s) — so a fast/short or CPU-rich run never pays a speculative VRAM grab, with
  no machine-specific threshold. The frame budget also gained a VRAM dimension:
  when GPU engages it grows from the CPU-sized `2·D` to also cover `ndev·gpu_batch`
  frames in flight (still capped by the 4 GiB host ceiling), so the streams get
  real batches on the CPU-poor box. Byte-identical everywhere. **Deferred:** Phase 3
  = D2H-cost-aware routing (keep trivially-compressed/small frames on CPU per the
  existing <2% rule, decode-heavy frames on GPU); and positive-perf validation on a
  genuine CPU-poor/GPU-rich box (a CPU-rich box's pool clears the starvation before
  the GPU is worth engaging, so the win isn't demonstrable there).
- **Tar-create member-reader scale-up:** the v0.15.5 dormant-reader
  mechanism covers only the plain-decompress prefetch pool; the tar-create
  member readers (`--read-threads`, device-bound per the v0.14.x
  measurements) are the same shape — dormant threads + the source-bound
  wake — but a separate pool. Same design, second consumer.
- **Read-path priors skip tar entirely:** v0.15.7 deliberately excludes
  tar runs from `path_<p>_gibs` recording (write-bound payload rates
  aren't comparable with plain-decompress reads). Tar could get its own
  profile keys (e.g. `tar_extract_gibs` per path) rather than silence.
- **Throttle grow is tar-exempt by design:** extract keeps its deliberate
  16 GiB in-flight cap and the FrameSink's own v0.14.74 grow — revisit
  only if extract measurements show the cap binding on big-RAM boxes.
- **(General, unblocks tar-create too):** the mmap queue-starvation
  classifier fallback — mmap leaves all four reader counters at zero, so
  SOURCE_BOUND is invisible on mmap-fed runs; needs queue-depth taps.

These are also listed as carried follow-ups in the v0.15.10 CHANGELOG
entry; this item is their tracked home.

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
uses + `set_done()` (`re_enqueue` bypasses it — and since v0.15.69 it inserts by
seq rather than `push_front`, because pushing a whole batch to the head inverted
sequence order as soon as two GPU workers requeued concurrently). Both decompress paths
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

## Phase 8: Multi-Format Codec Compatibility (zstd `--format=` parity)

### 8.1 `--format=zstd|gzip|xz|lzma|lz4` (proposed 2026-07-17)
**Priority: Medium | Complexity: High | Status: NOT STARTED**

Implement as much as practical of zstd's multi-format support, in the same
manner zstd does it: `--format=<codec>` selects the compression format for
both directions (zstd's own build does gzip/xz/lzma/lz4 via zlib/liblzma/
liblz4 when compiled in; decompression can also sniff magic bytes).

**The flag collision, and its resolution (decided up front):** gzstd's
existing `--format` selects the TAR archive format (`--format=gnu` vs the
PAX-sparse default). Rather than renaming either flag, `--format` becomes
domain-classified: every value after `=` is classified into its domain —
codec ({zstd, gzip, xz, lzma, lz4}) or tar format ({gnu, ...}) — with BOTH
spellings accepted and equivalent:

    --format=gzip,gnu              # commas within one flag
    --format=gzip --format=gnu     # or repeated flags

Two values landing in the SAME domain is a usage error like any other
invalid syntax (exit 2): `--format=gzip --format=zstd` errors, exactly as
`--format=gnu,pax` would. Unknown values error with the full vocabulary
listed. This keeps zstd-CLI compatibility (their `--format=gzip` works
verbatim) without breaking existing gzstd scripts using `--format=gnu`.

**Implementation landscape (why Complexity: High):**
- gzstd's whole pipeline is frame-parallel on the zstd framing. lz4 has a
  comparable frame format; gzip/xz/lzma do not map 1:1 — parallel COMPRESS
  can emit independent members/blocks (pigz-style concatenated gzip
  members, xz multi-block), but parallel DECOMPRESS of foreign
  single-member files degrades to streaming (same fallback discipline as
  today's unknown-content-size zstd path).
- GPU: nvCOMP ships batched Deflate and LZ4 codecs, so gzip and lz4 could
  keep GPU acceleration on our own frame-parallel output; xz/lzma are
  CPU-only.
- Deps: zlib, liblzma, liblz4 (all optional at build time, feature-gated
  like nvCOMP; static-link implications for BUILD_STATIC).
- Suite: each codec needs round-trip + foreign-file interop sections
  (compress with gzstd, decompress with the reference tool, and vice
  versa), added to the extensive `-e` compat runs.
- Interactions to spec before starting: `--tar` (codec wraps the tar
  stream — natural), seek-table/index (zstd-specific; skip for foreign
  codecs), `--adapt` profile keys per codec, exit-code fidelity.

---

## Phase 9: GPUDirect Storage — NVMe-to-VRAM P2P DMA (research, proposed 2026-08-06)

**Priority: Medium | Complexity: High | Status: NOT STARTED — and still NEVER ACTUALLY RUN**

> **2026-08-13 note.** This was once written off on the arithmetic that the drive (4.56 GiB/s)
> has 5× less bandwidth than the H2D link (25+ GiB/s), so removing the host bounce optimises a
> link with margin. That reasoning is sound *for this host* and is not a verdict on the feature:
> gzstd has to be fast on machines we do not own, and a box with a fast array or NVMe-oF inverts
> the premise. **It has never been measured, only argued about, and it should be tried.**
> The blocker here is environmental, not architectural: `nvidia-fs-dkms` is absent, so `cuFile`
> falls back to compat mode — literally "POSIX read into a host bounce buffer, then cudaMemcpy",
> strictly worse than today — and the IOMMU is in DMA-FQ mode, needing a GRUB change plus a
> reboot of an 8×H100 box. Also note the GPU→SSD *write* direction has never been examined at
> all; the original analysis only covered SSD→GPU.
> **Before spending the reboot, finish the staging work above** — the pipeline currently delivers
> 3.0 GiB/s against 71.8 GiB/s of kernel capability, so the host path, not the storage link, is
> what is throwing performance away.

Read compressed frames from NVMe **directly into GPU memory**, and write decompressed
output **directly from GPU memory to NVMe**, via NVIDIA GPUDirect Storage (the `cuFile`
API).  Today every GPU byte makes a round trip through host DRAM:

    now:  NVMe -> PCIe -> host DRAM -> PCIe -> VRAM     (two traversals + DRAM bandwidth + CPU)
    GDS:  NVMe -> PCIe -> VRAM                          (one traversal, DMA, no CPU)

### Why gzstd is unusually well placed for it
- **The O_DIRECT plumbing already exists.** `cuFile` needs O_DIRECT-style alignment, and
  gzstd already has `--direct-read`, the cold-input read probe, and 4 KiB alignment handling.
- **nvCOMP already operates on device buffers.** The change is narrow at the core: replace
  `pread` into a host pool buffer + `cudaMemcpyAsync` H2D with `cuFileRead` straight into
  the device buffer that nvCOMP was going to consume anyway.
- **Frame-parallel access maps 1:1.** `cuFileRead(fd, devptr, size, file_offset, 0)` is
  exactly the pattern `pooled_read_chunks` already generates.

### THE CASE FOR IT (rtcheek, 2026-08-06) — and why the earlier "it will lose here" was wrong
The first read of this said GDS would lose on this box because the page cache serves warm
reads at ~14 GiB/s and GDS deliberately bypasses it.  **That reasoning only applies to WARM
input, and real workloads here are COLD** — archives far larger than RAM, read once.  Cold
is precisely where the page cache contributes nothing and where the host bounce is pure
overhead.  The standing "cpu-only beats hybrid for compress" verdict was measured largely
warm, so **GDS is a credible route to inverting it** and should be measured, not assumed.

### Hardware reality on this box (measured, do not repeat the 128 GiB/s figure)
- **GPUs: PCIe Gen5 x16** (~63 GB/s per direction).  That is NOT the constraint.
- **NVMe: Gen4 x4, 16.0 GT/s** on all three devices (`0000:22:00.0`, `0000:23:00.0`,
  `0000:82:00.0`, Samsung) — about **7 GB/s per drive**, ~21 GB/s aggregate across three.
  **Storage is the narrow end of the link, not the bus.**
- Cold single-stream O_DIRECT reads measure **3.57 GiB/s** today, roughly half the Gen4 x4
  ceiling — so there IS headroom for a path that removes the host bounce.
- **Topology is favourable**: NVMe `22/23` sit adjacent to GPU1 (`0000:21:00.0`) and NVMe
  `82` next to GPU4 (`0000:81:00.0`), same root complex, matching NUMA node.  P2P is
  topologically possible here without crossing the inter-socket link.
- Filesystems are ext4 on both `/` and `/backup`, which GDS supports.

### PREREQUISITE, AND THE TRAP
`libcufile.so.1.13.1` is present (CUDA 12.8 ships it), but **`nvidia_fs` is NOT loaded and
no GDS packages are installed**.  Without that kernel module `cuFile` silently falls back to
**compatibility mode** — an ordinary POSIX read plus `cudaMemcpy`.  You get the whole API,
none of the DMA, and a benchmark that shows nothing.  **It is entirely possible to "add GDS",
measure neutral, and wrongly conclude the idea does not work.**  Verify the module is loaded
and `cuFileDriverOpen` reports GDS (not compat) mode BEFORE trusting any number.

### What to measure, here, before writing the integration
1. Baseline the ceiling: cold `--gpu-only` decompress and compress of a large archive, with
   `-vvv` to attribute Reader / H2D / Kernel / D2H / Writer time.  **If H2D+D2H is not a top
   cost, stop — there is nothing for GDS to win.**
2. Raw-path check with `gdscheck`/a `cuFile` microbenchmark: NVMe -> VRAM against
   NVMe -> DRAM -> VRAM on the same cold file, same device.
3. Only then decide.  Decompress is the better first target: output volume exceeds input, so
   the `cuFileWrite` side moves more bytes than the read side.

### 9.1 Topology-aware GPU selection (stands alone; prerequisite for GDS)

**Priority: Medium | Complexity: Low | Status: NOT STARTED**

Device selection today is `min(gpu_devices, device_count)` (`gzstd.cpp:19356`) — the FIRST N
devices, with no awareness of where the data is coming from. On this box that is measurably
the wrong choice:

    NVMe 0000:22:00.0, 0000:23:00.0  ->  adjacent to GPU1 (0000:21:00.0), NUMA 0
    NVMe 0000:82:00.0                ->  adjacent to GPU4 (0000:81:00.0), NUMA 1
    GPU0-3 = NUMA 0 | GPU4-7 = NUMA 1 | `SYS` between halves (crosses the inter-socket link)

So `--gpu-devices 2` takes GPU0+GPU1 regardless of which NVMe holds the input, and reading
from `/backup` (nvme0) to GPUs 0-3 crosses sockets for no reason. Select by PCIe/NUMA
proximity to the *source* instead. Worth measuring on its own for H2D bandwidth, and it
becomes load-bearing under GDS, where true P2P requires the NVMe and GPU to share a root
complex.

### 9.2 NVLink — what it does and does not buy us

Measured here: these are **H100 PCIe cards with 2-way NVLink bridges**, not SXM boards on an
NVSwitch. Bridged pairs are GPU2<->GPU3 and GPU6<->GPU7 (`NV12`); GPU1 reports `NV12` toward
GPU0 while **GPU0 reports all links inactive**, which is an asymmetry worth investigating
before relying on that pair. Everything else is `NODE` or `SYS`.

**NVLink gives peer ACCESS, not a memory POOL.** `cudaDeviceEnablePeerAccess` plus UVA lets
one GPU dereference a pointer into a bridged peer's memory at bridge speed, but a single
`cudaMalloc` still cannot exceed one device's VRAM. There is no 8x95 GiB allocation.
`cudaMallocManaged` can oversubscribe, but that is driver page migration with fault overhead,
not a fast pool.

**And gzstd does not need one.** The GPU work is embarrassingly parallel across frames: each
device does H2D batch -> nvCOMP -> D2H independently and **GPUs never exchange data**. Nothing
approaches 95 GiB — at the default 16 MiB chunk, ONE H100 already holds ~5,900 frames. VRAM
pressure exists only in batch sizing, which the auto-tuner already clamps.

**Specifically, "load a whole large file into pooled VRAM, then compress" would be a
regression, not a win.** It serialises what is currently a pipeline: gzstd overlaps read,
compress and write, so time-to-first-output is one frame, peak memory is bounded by the
throttle, and a 400 GiB input works on a machine with far less RAM. Reading everything before
computing anything gives up all three. The useful version of that instinct is **deeper
in-VRAM prefetch depth** so the GPU never waits on PCIe — a buffering-depth knob on one
device, not a pooling problem.

**Where NVLink could genuinely earn its place** is as a second hop under GDS: DMA
NVMe -> topologically adjacent GPU, then NVLink-forward to its bridged partner, avoiding a
second PCIe traversal for the far GPU. That only helps actually-bridged pairs, and it is an
optimisation layered on an unbuilt feature — sequence it after 9.1 and the Phase 9 baseline
measurement, not before.

### Known design conflicts to resolve before any integration
- **It fights `HybridSched`.** The scheduler picks CPU or GPU *per frame*, after the frame is
  read.  A frame landed straight in VRAM cannot go to a CPU worker without a D2H, so the
  backend decision would have to move BEFORE the read — inverting the current design.
  `--gpu-only` has no such conflict and is the natural first target.
- Trivially-compressed frames are routed to CPU by policy (ratio < 2%, to avoid D2H cost);
  those need host data.
- CPU-side `--verify` needs host data.
- `--tar` create assembles chunks from many small files in host memory; GDS applies to the
  single-large-file paths first.

---

## Remaining Work for v1.0

| Item | Phase | Priority | Status |
|------|-------|----------|--------|
| Streaming decompression output | — | HIGH | DONE (v0.12.24) |
| Asymmetric mode (PCIe Gen3 detection) | 5.1, 5.2 | HIGH | DONE (v0.13.0) |
| Auto --direct for Gen4+ compress & decompress | 5.3 | HIGH | DONE (decompress v0.13.25, compress v0.13.26) |
| Persistent auto-tuning (per-machine profile) | 2.1–2.3 | Medium | DONE (`--adapt`, v0.15.0–40) — `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`, not `~/.gzstd/`; carries a schema epoch that self-resets on a format change (v0.15.40). Opt-in through v0.15.x; default-flip is a v1.0 decision |
| Rate-matched dispatch (re-enable) | 1.3 | Medium | Disabled, needs eval |
| Pipe-aware scheduling | 3.1 | Medium | Not started |
| Streaming mode for unknown-size input | 3.2 | Low | Not started |
| Multi-reader NVMe | 4.1 | Low | DONE for decompress (v0.13.71/75, incl. redirected-stdin and block-device pread); compress deliberately stays single-reader — concurrent O_DIRECT contends |
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
| `-d --tar` writer pool measures PAGE CACHE, not the device | tar | Medium | Open (found 2026-08-04) — extract never waits for durability (zero `fdatasync` call sites; O_DIRECT exists for the leaf open but is not the default), so it writes into page cache and returns. Measured: a 10.36 GiB extract to a 0.81 GiB/s drive **returned in 1.63 s** and the kernel then spent **13.19 s** flushing after the process exited. The writer pool's busy/starved split is therefore `memcpy` bandwidth, and the sink-bound grow lever (v0.15.27, +26% on the server) cannot calibrate against real storage on any box whose RAM absorbs the output — it only engages where write volume outruns page cache or trips dirty-page throttling, which is a fragile precondition rather than a general property. Not obviously a defect (GNU tar does not fsync either, and forcing it would slow every extract and change semantics), but it means (a) `--adapt`'s extract writer verdicts are not measuring what they appear to, and (b) gzstd's reported extract time is not durable time. Wants a decision: measure sink pressure some other way, or expose durability explicitly, or document the limit |
| `--adapt` does nothing for operations shorter than `RAMP_SEC` | — | Medium | Open (found 2026-08-04) — the governor classifies nothing for the first 3.0 s, so any run finishing inside ~4 s never leaves warmup and gets no adaptation at all. Measured: extract from a healthy drive completed its pipeline in 2.59 s and produced no `[ADAPT] regime shares` line whatsoever. **This gets WORSE as hardware gets faster** — more operations finish inside the ramp — so it will bite the 8xH100/256-core box harder than this one. Interacts with the v0.15.55 evidence gate (2 s / 25% classified) which correctly refuses to persist a verdict from such a run, so fast runs also teach the profile nothing. **DECIDED 2026-08-04 (rtcheek): leave the ramp as is.** Recorded as a known, accepted limit rather than a pending fix — short runs get no adaptation, and that is the deliberate trade for not classifying on startup transients. If it is ever revisited, derive the ramp from observed signal stability rather than shrinking the constant, which would trade one arbitrary number for another |
| Reader-pool controller ratcheted to its CEILING when the reader was not the bottleneck | — | Medium | DONE (v0.15.57) — the controller maximises READER throughput, which is only a proxy for run time when the reader is the wall. Warm on a 24-thread Gen3 box (compute-bound, ~6.97 GiB/s) it walked 3→4→6→9, pinned and persisted 9, while a fixed sweep put the optimum at 3 (median-of-7: 5.22 s at 3 vs 5.35 s at 9, **non-overlapping**). A confirm-window was tried first and did NOT help — proof it was not noise but a real gain in the wrong quantity: extra readers do raise reader throughput, by taking CPU from the workers. It now gates on the published regime like every other `--adapt` acting site (it was the only one that did not), stepping only while SOURCE_BOUND and reverting an in-flight probe if the regime leaves it. Verified: warm now holds 3 across 5 runs and persists 3 |
| `--adapt` governor classified clearly I/O-bound runs as COMPUTE_BOUND | — | Medium | DONE (v0.15.58-60) — three separate defects in the ONE signal the governor uses (`reader_io_ns` summed across readers / `reader_threads`): the divisor was written only at teardown so every window used the pool's STARTING count (wrong in both directions — shrink understates, grow overstates); `--tar` create never fed any of the four reader counters, so `rbusy` was identically 0 and it could never classify source-bound; and `-d --tar` extract neither timed its `pread_seek_frame` nor published the decode pool's reader count, leaving the divisor at its default of 1 against ~15-22 live readers. All three now measured. Verified: cold read 3%→91% source-bound, tar create compute→source-bound 47%, extract compute→source-bound 46%, and `[READER]` io went from an impossible 105.7% to 91.5% |
| Converged reader controller paid exploration cost every run | — | Low | DONE (v0.15.61) — the step size was answering only the COLD question. Proportional `max(1, cur/2)` is right when searching (it must cross the range in a bounded budget; `±1` settled at 14 where 24 was 16% better) but wrong when VERIFYING a seeded answer: from a settled 4 it probes 2 and 6, and 2 measured 7.31 s against the optimum's 4.63 s. Now `±1` when seeded, proportional when not — cold-start behaviour on a big box is untouched, and a seeded controller still travels one step per run so a moved workload is still followed. Converged steady state 4.93 s → 4.73 s against a 4.63 s forced optimum, i.e. the tax falls from ~6.5% to ~2.2% |
| `--adapt` reported `GPU driver changed` on a BRAND-NEW profile | — | Low | DONE (v0.15.62) — the guard must also fire on a MISSING key (a stored "" once matched a probe failure and skipped the clear), so the branch is right; only the wording was wrong. Announcement now requires a previous driver to have existed. Verified silent on a virgin profile and still reporting a forced 999.999 -> 570.207 (found v0.15.55)
| `--adapt` summary said `actions: none` while a probe printed | — | Low | DONE (v0.15.62) — not a contradiction: the governor sets a probe REQUEST, the summary reported whether a writer thread ENGAGED, and nothing did. Asked / engaged / kept are now three distinguishable states, so a request that never took effect says so (found v0.15.55)
| Reader-count objective is NOT unimodal — hill-climbing cannot reach the optimum | — | Medium | Open (found 2026-08-04, v0.15.64) — the controller takes a step, measures it, and keeps it only if it pays, which assumes a single hill. Measured cold buffered on a 24-thread Gen3 box, 32 GiB, median-of-3: **1 reader 15.15 s, 2 readers 18.93 s, 3 readers 16.14 s**, then 6 → 16.45, 9 → 16.58, 12 → 16.86. The global optimum (1) sits behind a **valley at 2** that is 17% worse than either neighbour, so a ±1 climb from the default 3 correctly rejects both directions and settles on a true LOCAL optimum 6.1% off the best. Verified with `GZSTD_DEBUG_RD_CTL=1`: it probed 3→4 (2.061→1.999 GiB/s, reverted) and 3→2 (2.061→1.811, reverted) and stopped — the implementation is behaving exactly as designed, and the design is what is wrong. Plausible mechanism (unconfirmed): 2 readers interleaving chunk indices on one fd defeat sequential readahead, while 3 restore enough parallelism to compensate and 1 is purely sequential. Impact here is nil in practice — the default path takes O_DIRECT (11.27 s), far better than any buffered count — so this only bites where buffered is the chosen path. Options: evaluate the FLOOR explicitly on a cold start (one extra window), or coarse-scan 3 points before climbing. **Wants a look on the big box first**: its range is 6–32 starting at 12, where a valley would be more consequential and the cost of a wrong settle is larger **SERVER SAYS IT DOES NOT GENERALISE (2026-08-05):** the same sweep there, including 1 and 2 which the v0.15.44 sweep never tested, is 1→1.28, 2→2.26, 3→2.63, 4→2.76, 6→2.81, 12→2.66, 16→2.70, 24→2.67, 32→2.60 GiB/s — no valley, no optimum at 1, and above ~4 it is FLAT INSIDE THE NOISE. So the valley is a property of that machine, not of the objective, and a general fix cannot be justified from it. STILL OPEN: a ±step search cannot cross a valley where one exists. Candidates unchanged (evaluate the FLOOR explicitly on a cold start, or coarse-scan 3 points before climbing) — but the v0.15.65 noise gate must be re-validated there FIRST, since it may already suppress the steps this row wants taken. |
| O_DIRECT compress read failed on any input whose size is not 4096-aligned | — | **High** | DONE (v0.15.63) — the last block of such a file is partial and an O_DIRECT `pread` returns exactly that; the v0.15.41 short-read retry guard (added against silent mid-file truncation) could not tell that from an unresumable short read and failed the job with exit 3 and no output. Hit `--direct-read` always and the DEFAULT path whenever the cold probe adopted O_DIRECT — i.e. 4095 of every 4096 real inputs. Hidden for four versions because every generated corpus is MiB-sized and therefore aligned; the guard now compares against the input size, and the regression test uses deliberately unaligned sizes |
| `--direct-read` took a memcpy path the probe did not (GPU/hybrid producer) | — | Medium | DONE (v0.15.63) — the flag had its own branch reading at host-chunk granularity into one scratch buffer, copying every subchunk into a Task; the pooled reader had since made that copy unnecessary and `compress_cpu_mt` was converted while this producer — the one a plain `gzstd FILE` uses — was not. Cold 32 GiB: 17.65 s by flag vs 11.77 s for the probe adopting the same physical path, `task-copy` 23.1% → 0.0%. `--adapt` sets `direct_read` from its measured prior, so a machine that learned O_DIRECT wins re-applied it through the slow copy every run. Same mirrored-path class as the four v0.15.35 defects whose fix was already written in the sibling |
| Is the O_DIRECT READ regression a LOW-CORE property? | 4.1 | Medium | ANSWERED NO (2026-08-04) — provenance first: the 20–40% figure cited as the `--direct-read` justification matches the v0.13.22 dataset for `--direct` (O_DIRECT **output**, a different and independent flag); no equivalent `--direct-read` compress dataset for this box is in the log, and the output-side row is NOT affected by anything below (every run wrote to `/dev/null`). The read-side claim was in any case measured on a box whose source NVMe was linked at PCIe **Gen1** (866 MB/s). Refitted to Gen3 x4 (2.9 GB/s) and with the copy above removed, cold 32 GiB median-of-3 gives O_DIRECT **11.27 s** vs buffered 16.06 s vs mmap 18.67 s — O_DIRECT **wins 1.42x** on the one machine that was evidence for the opposite rule. Core count predicts nothing here. The probe stands (it is what caught the rule being wrong) and its verdict is now correct on both machines on record |
| Compress `ThreadGuard` teardown does not signal ABANDONMENT to the writer | — | Low | DONE (v0.15.53–54) — `g_run_abandoned` + `run_abandoned()` generalise what `g_gpu_aborted` already meant to the writer: missing frames are EXPECTED and the output is discarded. Set first in the guard so `workers_done` is never visible without it. Fixing the misreport then exposed a deadlock the old `die()` had masked — the guard's joins had never executed, and workers parked in `acquire_out_buf` waiting for output-pool slots the departed writer could no longer return; those escapes now test `run_abandoned()` too |
| `--tar` input ergonomics (`--exclude-from`/`-X`, `--files-from`, `-P`, `--exclude-vcs`) | tar | Low | DONE (v0.14.90) |
| `--selinux` context storage (third leg of xattrs/ACLs) | tar | Low | DONE (v0.14.91) — spot-check a labeled-host round-trip if one appears |
| Restore xattrs/contexts on symlinks & special files (extract side) | tar | Low | DONE (v0.15.48), UNVERIFIED HERE — `apply_ext_path` uses O_PATH\|O_NOFOLLOW + lsetxattr through /proc/self/fd, so the LINK is labelled, not its target. Cannot be demonstrated on this box: Linux forbids `user.*` xattrs on symlinks/FIFOs entirely, so only `security.*` (SELinux) and `trusted.*` exercise it, both privileged. Wants an SELinux host or a root-capable CI job |
| Hoist the pax record-grammar walk into a shared for_each_pax_record | — | Low | DONE (v0.14.92) — grammar walk single-sourced; per-caller key dispatch deliberately kept local (premature-abstraction verdict) |
| Seek table on PLAIN (non-tar) compress output | — | Low | DONE (v0.14.92) — all compress paths emit it (cpu/gpu/hybrid/serial/stdin verified; --sliding-window and --no-index excluded); self-validating geometry so a wrong table can never be emitted |
| Warm/cold-adaptive `-l` fallback walk (mincore dispatch → buffered pread walk) | — | Low | DONE (v0.14.93) — warm 65 GiB single frame 3.0 s → 1.29 s (zstd-class; faults 560k → 4.4k); cold keeps the mmap+SEQUENTIAL walk (still beats zstd's QD1 strided reads 38.7 s vs 42.4 s); buffered walk is strictly validated, bails to the mmap walk on anything unmodeled |
| Cold `-l` walk: batched posix_fadvise(WILLNEED) header prefetch | — | Low | Open — the remaining half: prefetch upcoming header offsets at deep queue so the cold walk reads ~2 GiB of header pages instead of streaming 65 GiB (should beat both tools cold); build on v0.14.93's buffered_frame_walk; validate with the cold umount methodology and scripts/drop_cache |
| `--adapt`: sub-floor probe records nothing (not even `runs`) | — | Low | DONE (v0.15.48) — a deliberate exploration is admitted under the floor and records its attempt stamp only, no rate |
| `--adapt`: `g_adapt_gpu_engaged` set at worker SPAWN | — | Low | DONE (v0.15.48) — moved to the first DELIVERED frame on both GPU paths; note the shared `writer_thread` is the wrong site (a CPU batch would set it) |
| `--adapt`: a residency bucket mixes durations | — | Medium | DONE (v0.15.48) — runs under the engagement guard no longer contribute to the backend pair (they still contribute runs/regime/read-path/reader count); a gate, not a fourth key dimension, because the sub-guard side can never form a pair |
| `-d --tar` pool controller's rate signal is PARSE progress, not write progress | tar | Medium | DONE (v0.15.48) — parallel extract routes the progress meter elsewhere and counted tar bytes PARSED; the writer pool now steers on a dedicated write-completion counter, so it sizes against the sink rather than against its own supply |
| GPU batch head-of-line blocking in the decode pool (a parse thread can wait a whole H2D+decode+D2H for a frame at the end of a 64-item batch) | tar | Low | Open |
| Extract governor never reads the supervisor's `ewgrow_cap_` | tar | Low | DONE (v0.15.48) — the pool publishes its real ceiling and the governor clamps to it before persisting |
| `q_max_bytes_` sized from the base writer pool, never resized when it grows | tar | Low | DONE (v0.15.48) — resized on every grow round; a grown pool was starving on a queue budget for the base width |
| Decompress size estimator opens + preads 1 MiB PER INPUT | — | Low | DONE (v0.15.48) — samples the first 4 inputs and scales the rest by the ratio they establish; the consumer is a coarse size gate |
| O_DIRECT bounce aggregate only held near 256 MiB while the writer pool stays ≤256 writers; past that the 1 MiB floor wins | tar | Low | Accepted (v0.15.39) — smaller writes would cost more than the memory saves |
| `std::terminate` if an exception escapes the compress reader region | — | Low | DONE (v0.15.48) — RAII guard unblocks (queue done, throttle released, results marked) then joins on unwind, mirroring the success path exactly |
| Reader controller graded NOISE: a fixed 5% keep-margin below the measured window-to-window spread | — | Medium | DONE (v0.15.65) — on the server, consecutive BASELINE windows at an unchanged reader count measured 3.561 then 4.121 GiB/s (**16% apart, nothing changed**), so `KEEP 12 -> 11 rate 3.683 (was 3.248)` recorded a 13% "win" between counts a median-of-5 calls identical; five identical runs settled 12,12,6,12,12 and the same settled 12 produced 3.14 and 2.57 GiB/s. The gate now keeps the last 4 baseline windows and requires `max(5%, observed max/min spread)`. `GZSTD_DEBUG_RD_CTL=1` prints the floor |
| Does the v0.15.65 noise gate OVER-SUPPRESS on a low-core box? | — | Medium | **OPEN, and the server CANNOT answer it** — only the suppression half is verified there (floor 1.16-1.27x, controller correctly holds). The one server regime with a large real effect (copy-bound warm: 3→5.72 vs 16→16.23 GiB/s) never reaches the controller, because the v0.15.61 regime gate holds there correctly (at 12 that run is compute-bound). The low-core box has real differences of 6-17% with TIGHT distributions — if its measured floor exceeds those, the gate suppresses a real win and the margin rule is wrong. See `project_workstation_validation` |
| Compress buffer pool RSS: the `--adapt` reader controller now spawns up to 32 reader threads (parked ones hold no buffers, but the pool is sized for the ceiling) | — | Low | Open (v0.15.44) — pool is RAM-capped since v0.15.43; the thread count itself is unmeasured on a small box |
| Reader controller explores from the static start every run; the settled count is not persisted to the profile | — | Medium | DONE (v0.15.45) — persisted per (machine, direction, residency, sink class); no flat fallback, because seeding from a different regime measurably starts a copy-bound run at the cold bucket's number. A fast regime's runs fall under the save floor and still re-climb |
| `--adapt` reader controller not wired into `--tar` create | tar | Medium | DONE (v0.15.47) — the cause was a MISSING NOTIFY, not the claim/park interaction: the final `fetch_add` satisfies the parked readers' predicate but notifies nobody, and `stop()` runs after the join. Found by in-process watchdog (gdb cannot attach; yama ptrace_scope) |
| Decompress prefetch reader now uses the shared measured controller | — | — | DONE (v0.15.46) — replaced the one-shot 2x latch, which was inert at >=96 threads; ceiling RAM-clamped because its look-ahead ring is 64 MiB per slot |
| CLI ergonomics: `--watchdog SECS` `=`-form only; `-T -5` silently ignored; `-0` a usage error | — | Low | DONE (v0.15.48) — all three; `-0` parity verified against real zstd v1.5.7, and the suite test that asserted the old behaviour was corrected |
| Quiet-box A/B: `--adapt` vs `--cpu-only` vs `--hybrid`, both directions, median of 3 | — | Medium | DONE (v0.15.41, 2026-07-30) — 80 runs, median of 5, per-run GPU contention recorded because the box was NOT quiet. cpu-only wins all four cells; `--adapt` lands within 1–7%. Corrected two `AGENTS.md` claims: the compress margin is ~1.1x not 2.27x, and the warm/cold decompress split is a small-input effect that is gone above ~4.5 GiB |
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
| v0.13.25–v0.14.95 | See `CHANGELOG.md` — the `--direct` defaults, `--tar` archiving and parallel extract, `--verify`, `--keep-going`, xattr/ACL/SELinux storage, the zero-copy extractor and its windowed part-jobs |
| v0.15.0–v0.15.20 | `--adapt`: regime classifier + actuators, per-machine profile priors, the decoupled extract pipeline and its shared CPU/GPU decode pool, the unified start-high/contract bottleneck controller |
| v0.15.21–v0.15.34 | GPU engagement guards (stop paying `cuInit` on inputs too small to use a GPU), the workload-keyed extract writer prior, the end-to-end backend prior, and the CPU-only-build arg-parsing repair |
| v0.15.35 | **Six-angle code review, 17 defects fixed** — a `ResultStore` resize racing the live writer, every `--adapt` prior compiled out of the CPU-only build, a budget-permit leak that hung on a corrupt tar, a `--cpu-batch` deadlock, and the parallel extract path silently memcpy'ing every byte (peak RSS −27% once fixed) |
| v0.15.36 | Backend prior **keyed by input residency** (warm 2.28 vs cold 1.55 GiB/s on the same file; the blended 1.91 described neither) + the four state-machine defects keying exposed |
| v0.15.37–v0.15.39 | **Four independent review passes by a different model** (Codex CLI / GPT-5.6-sol). It found a bug v0.15.37 shipped, three of the fixes that were only half-fixes, and two order-dependence bugs in multi-input handling. Method: ask it to *confirm or reject its own prior findings*, and run it BEFORE the suite — it twice caught what a green suite hid |
| v0.15.73 | **Round 5: `gzstd -t` passed a damaged archive.** Appending the 4-byte zstd magic to a valid file made `-t` report OK and exit 0 where stock zstd says "premature end", and `-d --rm` then deleted it; an empty `.zst` was accepted too. Took THREE decompress paths, not the two reported. One deliberate divergence kept: a truncated trailing SKIPPABLE frame (gzstd's own index trailer) warns and falls back, because it carries no user data — a truncated DATA frame stays fatal. **Round 4's `--tar` fix was a no-op**: `-C` resolution was half the reason, the other half was `--overwrite` unlinking the output BEFORE the identity check ran. Plus the sliding-window verifier accepting truncated frames, the new empty frame skipping `--verify`, `-l` accepting malformed input, the profile lock racing on failure, and a host-endian XXH64 |
| v0.15.72 | **Round 4 found a feature attached to one of four write paths.** `--verify` checked ZERO frames on `-T1` and `--sliding-window` — both bypass the ordered writer the verifier taps — so an integrity request silently did nothing and `--rm` then deleted the source. Sliding-window gets a streaming verifier (constant memory, rolling XXH64) rather than being rejected. The SAME divergence produced a zero-byte output for empty input on 3 of 4 paths, fixed once at `run_one_pass`. Also: gzstd never owned an XXH64 — `extern ZSTD_XXH64` is not exported by stock libzstd, so dynamic GPU builds failed at LOAD; now implemented in-tree and validated 57/57 against libzstd. Plus the `--tar` absent-output race, stdout tails on `/dev/shm`/unlinked fds, the buffered rebuild ignoring `ftruncate`, calibration TOCTOU, global PAX extended metadata, and two `--adapt` profile defects |
| v0.15.71 | **A third review round, deliberately left OPEN** — adjudicate the five, then look anywhere. Nine findings, **one of them a regression v0.15.70 had shipped**: adopting fd 1 for redirected stdout reproduced the data path of the `open(...,O_TRUNC)` it replaced but not the TRUNCATION, so a longer existing target came back as a corrupt concatenation at exit 0. Also `--tar --overwrite` deleting a source before archiving it, the `--calibrate -o` symlink race, `pread` failure as clean EOF in the unknown-frame fallback, xattrs landing on the procfs magic link, global PAX applying to one member, and `--keep-going --tar` reporting 6 over a truncated tree. **Two findings pushed back on**: the xattr trigger is not constructible (Linux refuses `user.*` on symlinks/FIFOs) though the mechanism was real, and the global-PAX fix was reverted in the index builder, which rejects `'g'` outright |
| v0.15.70 | **The re-review rejected five of the eleven v0.15.69 fixes**, each with a concrete trigger, and found no new regressions. The identity check had a TOCTOU window for a not-yet-existing output (now: open without O_TRUNC, `fstat` the FD against the input's, truncate only after); `--direct` discarded the `O_EXCL` guarantee by reopening the temp BY NAME (now `DirectWriter::adopt_fd`, three call sites incl. the rebuild path); two more read-error sites turned `EIO` into a clean EOF; `EINVAL` was forgiven from errno alone, which a FUSE regular file can return, so `--sync-output` could report durability it never obtained; and the compress GPU catch discarded the failure counter, hanging the fixed-share barrier forever when a GPU threw before registration. **Also corrected the record on the v0.15.69 tar-hang call: the finding stood, my rebuttal missed that the pusher waits for the first UNCLAIMED index** |
| v0.15.69 | **An independent whole-codebase review before tagging.** 25,918 lines and the help text, reviewed by a different model primed with `AGENTS.md`: verdict NOT SAFE TO TAG, with **seven CRITICAL findings all of the form "destroys or corrupts data and exits 0"**. Two reproduced verbatim first: `--rm -f -o data data` deleted BOTH the source and the archive after printing a success line, and the fixed `<output>.gzstd.tmp` name was symlink-followed, letting a pre-created symlink redirect the write over any writable file. Also: `fread` errors read as clean EOF at five sites (compress a prefix, install it, exit 0, then `--rm` the source), ignored `ftruncate`/`fsync` failures, device nodes silently skipped on `EPERM`, and a discarded `--rm` failure. Plus three tar write loops that spun on `write()==0`, and `re_enqueue` reintroducing the v0.15.66 seq inversion. **One reported hang did NOT reproduce** (seven fault-injection timings) and is recorded as hardening, not a fix. Help audit: 143 parser spellings vs 79/89 documented, 33 discrepancies, 12 HIGH — including an unqualified "drop-in-compatible" claim over ~30 silently-ignored zstd options |
| v0.15.68 | The last unsorted queue, `RescueQueue`, turned out to be **unreachable** — no producer, no consumer, only `set_done()` — so it was removed (73 lines) rather than guarded. Sorting an unreachable push would have documented a hazard that cannot occur. ROADMAP's own "deadlock-free by construction (FIFO queue guarantees...)" claim annotated with what actually enforces it |
| v0.15.67 | **v0.15.66 fixed two of three cycles.** Validation on the 256-thread box reproduced the deadlock 5/5 pre-fix and **still hung 2/30 after** — rising to 34/40 at 48 readers. The survivor was the cycle the original root-cause note described first: `pooled_read_chunks` claims its chunk index BEFORE acquiring a buffer, so the writer's next frame can be absent from the queue entirely, and sorted insert cannot order a frame that is not there. Fixed with a **bounded (max 1) head-of-line permit overdraft**; the bound matters because `-M` makes the budget a user-visible memory cap (unbounded measured a 2.6x overshoot). Every failing cell to zero, no perf cost |
| v0.15.66 | **Multiple readers had quietly broken the invariant everything else assumed.** `--cpu-only` compress of a WARM source with >=9 pooled readers deadlocked permanently, ~20-25% of runs. One false premise stated in four places and enforced in none: "frames are pushed in seq order". A single reader made it true for free. Fixed by sorted insert in `TaskQueue::push` and by making the four per-worker output pools overdraft instead of block |
| v0.15.65 | The reader controller **was grading noise**: baseline windows 16% apart against a fixed 5% margin. The gate now MEASURES the floor (last 4 baseline windows, `max(5%, spread)`) and holds instead of wandering. Suppression half verified on the server; the "still finds real structure" half is owed on the low-core box |
| v0.15.63–v0.15.64 | **O_DIRECT refused almost every real file** — the v0.15.41 short-read guard could not tell EOF from a mid-file stall, so any input whose size is not 4096-aligned failed with exit 3 and no output, **on the default path**. It survived four versions because every generated test corpus is MiB-sized, hence aligned. Also: `--direct-read` took a memcpy branch the probe did not (1.50x slower on the same input), and the reader search was shown to be non-unimodal on a low-core box |
| v0.15.54–v0.15.62 | **First execution on a PCIe Gen3, low-core box, and it found what only that box could.** The v0.15.48 unwind guard never ran (four defects hid behind it); `--adapt` persisted regime verdicts it never measured; the reader-pool controller climbed the wrong quantity in the 1-9 band the server never enters; the governor could not see three of its own reader paths (all three inputs to its source-busy fraction were wrong, in three different places — every fix corrected a MEASUREMENT, no constant was retuned); and telemetry claimed things it could not know |
| v0.15.48 | **The open-items ledger, closed**: eight carried items (three `--adapt` correctness, two extract-pool sizing, the O(inputs) size estimator, `std::terminate` in the compress reader region, and three CLI ergonomics) plus the `--tar` reader bucket that was never written, plus first validation of `--gpu-only --tar` create |
| v0.15.47 | `--tar` create joins the reader controller, and **four liveness bugs** go with it: the missing notify above, a pre-existing queue-floor deadlock when an explicit hybrid request finds no GPU, a member turning into a FIFO mid-assembly wedging the pusher, and abort not cancelling tar assembly. Plus a controller that could persist an unvalidated probe. Three timeout-guarded regression tests |
| v0.15.46 | The sizing controller becomes shared code and the **decompress prefetch reader** adopts it, replacing a one-shot latch that was inert on any box with >=96 threads. `--tar` create was tried and reverted: parking a reader strands the chunk its pusher is waiting on, and it deadlocked |
| v0.15.45 | The settled reader count **persists, keyed by the regime it was measured in** (residency x sink class) — one number per machine would be re-taught on every workload alternation, and a flat fallback measurably seeded a copy-bound run with the cold bucket's value |
| v0.15.44 | **Adaptive reader-pool sizing** under `--adapt`: a bounded hill-climb that takes a step, measures it, and keeps it only if it pays — because the *direction* is not derivable from the regime label (`SOURCE_BOUND` covers both the copy-bound case, which wants more readers, and the device-saturated case, which wants fewer). +12.8% copy-bound, +11% device-bound, neutral when sink-bound |
| v0.15.41–v0.15.43 | **A cold compress input is read with O_DIRECT**, chosen by timing both read paths on the real data — because O_DIRECT is residency-independent while mmap collapses cold, so the winner *inverts* with residency and neither PCIe generation nor NVMe-ness separates the machines where it wins from the one where it regresses. Default cold 64 GiB compress 25.12 s → 14.06 s. An independent review (Codex CLI) returned NOT SAFE TO TEST on the first cut and was right: the probe never ran on the default path at all, a short `pread` was treated as EOF (pre-existing, silent truncation), and each pass reopened the input by name |
| v0.15.40 | The `--adapt` profile records its **schema epoch** and the writing build's version, and discards itself when the epoch changes — closing the "do the hosts need their cache cleared?" question permanently. The version is never the trigger: it bumps every build |
