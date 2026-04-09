# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
cmake -B build
cmake --build build -j$(nproc)
./build/gzstd --version
```

Key CMake options:
- `USE_NVCOMP` (default: ON) — GPU backend via NVCOMP/CUDA; auto-disables if unavailable
- `BUILD_STATIC` (default: OFF) — static linking for portable binaries

For deployment with bundled nvCOMP:
```bash
cmake --install build --prefix ./dist
```

See `BUILD.md` for conda setup and static linking details.

## Tests

```bash
./gzstd-test.sh                # run all tests (~170+)
./gzstd-test.sh ./build/gzstd  # explicit binary path
```

GPU tests are automatically skipped if no GPU is detected. Run before every commit.

## Benchmarks

```bash
./gzstd-gendata.sh 512         # generate 2.5 GiB test data (5 compressibility profiles)
./gzstd-benchmark.sh           # full sweep (batch size, streams, threads, levels)
```

Results written to `benchmark-results.json`.

## Architecture

**Single-file monolith**: All ~6,400 lines live in `gzstd.cpp`. Read the architecture comment at the top (lines 1–36) first.

### Data flow

```
INPUT → [Reader] → [TaskQueue] → [CPU workers / GPU workers] → [ResultStore] → [Writer] → OUTPUT
```

- **TaskQueue**: Thread-safe deque; CPU workers call `pop_one`, GPU workers call `pop_batch_greedy`
- **ResultStore**: Out-of-order accumulator keyed by sequence number — ensures frames are written in order despite parallel completion
- **FrameThrottle**: Counting semaphore that blocks workers when buffered output exceeds 4 GiB (resumes at 2 GiB) — prevents OOM on fast CPUs + slow disks
- **DirectWriter**: O_DIRECT file writer (4 KiB-aligned, sparse-aware); also reopens stdout with O_DIRECT if redirected to a regular file
- **HybridSched**: Adaptive CPU/GPU work-sharing scheduler based on observed EMA throughput; GPU sets `gpus_waiting_` flag so CPU yields rather than racing

### Compression paths

| Mode | Entry point | Workers |
|------|-------------|---------|
| CPU-only | `compress_cpu_mt` | N CPU threads, `ZSTD_compress2` per frame |
| GPU | `compress_nvcomp` | batched nvCOMP + optional CPU hybrid |
| Hybrid | same, `HybridSched` decides | adaptive CPU/GPU split per frame |

### Decompression paths

| Mode | Entry point | Workers |
|------|-------------|---------|
| CPU-only | `decompress_cpu_mt` | N CPU threads, `ZSTD_decompress` per frame |
| GPU | `decompress_nvcomp` | batched nvCOMP + CPU rescue for failed frames |

Frames with compression ratio < 2% (trivially compressed) are always routed to CPU to avoid PCIe D2H cost.

### Key classes

| Class | Role |
|-------|------|
| `Options` | All CLI config; passed by value through every path |
| `Meter` | Atomic counters for progress (bytes read/written, task counts) |
| `TaskQueue` | Thread-safe work queue with CV wakeups |
| `ResultStore` | Sequence-ordered output accumulator |
| `FrameThrottle` | High/low-water backpressure semaphore |
| `AsyncWritePool` | Background write thread; decouples workers from NVMe latency |
| `HybridSched` | CPU/GPU throughput tracker and dispatch policy |
| `StreamCtx` | Per-GPU-stream state (batches, CUDA buffers) |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error |
| 2 | Bad usage |
| 3 | I/O error (disk full, permissions) |
| 4 | Data error (corrupt input, integrity failure) |
| 5 | All GPUs failed (VRAM exhaustion, driver error) |

### Adding a new CLI flag

1. Add field to `Options` struct
2. Parse in `parse_args`
3. Thread through all relevant compress/decompress paths (CPU and GPU)

### Performance notes

- Documented optimization attempts (successful and failed) are in `CHANGELOG.md` — read before trying anything that sounds clever
- `ROADMAP.md` lists planned work and open design questions
- Verbosity levels (`-v`, `-vv`, `-vvv`) expose Meter counters and timing for profiling
