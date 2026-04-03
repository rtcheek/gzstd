# gzstd Benchmark Suite

## Overview

`gzstd-benchmark.sh` is an automated benchmark harness for `gzstd`. It can benchmark CPU-only, GPU-only, and hybrid modes, sweep multiple tuning parameters, measure both compression and decompression throughput, verify decompression correctness, print a live progress view, and emit machine-readable JSON results.

The script auto-discovers `gzstd` in this order:

1. `$GZSTD` environment variable
2. `./gzstd`
3. `./build/gzstd`

Test data defaults to `./gzstd-testdata`.

## Quick Start

```bash
# 1. Generate test data (512 MiB per file, ~2.5 GiB total — output dir defaults to ./gzstd-testdata)
./gzstd-gendata.sh 512

# 1a. Or specify a custom output directory as the second argument
./gzstd-gendata.sh 512 ./gzstd-testdata

# 2. Run a quick sweep (reduced matrix, 1 iteration)
./gzstd-benchmark.sh --quick --sweep-all

# 3. Run a full sweep (default: 3 iterations per config)
./gzstd-benchmark.sh --sweep-all
```

## What the Script Benchmarks

The harness can test:

- CPU-only mode
- GPU-only mode
- Hybrid CPU+GPU mode
- GPU batch sizes
- GPU stream counts
- CPU thread counts
- CPU compression levels
- Optional decompression performance

## Test Data

`gzstd-gendata.sh` creates five files with different compressibility profiles. All five are
generated in parallel (one per CPU core, up to 5), with a live per-file progress display
showing percentage complete and GiB written.

| File | Description | Typical Ratio |
|---|---|---:|
| `high_compress.bin` | Repeated log lines | ~0.1% |
| `medium_compress.bin` | Pseudo-text + random printable ASCII | ~30-40% |
| `low_compress.bin` | 90% random, 10% repeated pattern | ~85-95% |
| `mixed.bin` | Alternating JSON records + random | ~50-60% |
| `zeros.bin` | All zeros | ~0.01% |

### Requirements

| Dependency | Required | Notes |
|---|---|---|
| `bash` | Yes | |
| `python3` | Yes | All file generation runs in Python for speed |
| `dd` | Yes | Used for zero-fill fallback |
| `numfmt` / `stat` | Yes | Part of GNU coreutils |
| `numpy` | No | Install for ~5x speed boost on `medium_compress.bin` — the script detects and uses it automatically if present |

### Performance notes

- All five files are written simultaneously in parallel background processes.
- Each generator uses 64 MiB write chunks, reducing I/O syscall count ~1000x versus the previous 64 KiB chunks.
- `medium_compress.bin` uses NumPy (if available) for SIMD-vectorised random-printable-ASCII generation, falling back to `array.array` otherwise.
- On a fast NVMe with NumPy installed, a 20 GiB run completes in roughly 2–5 minutes.

## Actual Sweep Values Used by `gzstd-benchmark.sh`

### Batch sizes

- Quick mode: `8, 64, 256`
- Full mode: `4, 8, 16, 32, 64, 128, 256, 512, 1024`

### Stream counts

- Quick mode: `1, 3, 6`
- Full mode: `1, 2, 3, 4, 6, 8`

### Thread counts

- Quick mode: `1, N/2, N-1` (clamped to valid values)
- Full mode: `1, 2, 4, 8, 16, N-1` (where applicable)

Thread sweeps now add both:

- CPU-only thread-count variants such as `cpu-T8`
- Hybrid baseline thread-count variants such as `hyb-T8`

### Compression levels

- Quick mode: `1, 6, 19`
- Full mode: `1, 3, 6, 9, 12, 15, 19`

Compression level sweeps remain CPU-only.

### Combined GPU sweeps

When both `--sweep-batches` and `--sweep-streams` are enabled in full mode, the script also tests these combined GPU tuning points:

- batches: `16, 32, 64`
- streams: `2, 4, 6`

for both GPU-only (`gpu-b32s4`) and hybrid (`hyb-b32s4`) labels when those modes are enabled.

## Usage

```bash
./gzstd-benchmark.sh [options]
```

### Options

```text
--gzstd PATH          Path to gzstd binary (default: auto-discover)
--data-dir DIR        Directory with test files (default: ./gzstd-testdata)
--output FILE         JSON output file (default: benchmark-results.json)
--iterations N        Runs per config, reports median (default: 3)
--files PATTERN       Only test files matching glob (default: *)
--quick               Reduced sweep and 1 iteration
--gpu-only            Only test GPU-only configs
--cpu-only            Only test CPU-only configs
--hybrid-only         Only test hybrid configs
--sweep-batches       Sweep GPU batch sizes
--sweep-streams       Sweep GPU stream counts
--sweep-threads       Sweep CPU thread counts for CPU-only and hybrid baseline configs
--sweep-levels        Sweep CPU compression levels
--sweep-all           Enable all sweeps
--no-decompress       Skip decompression benchmarks
--help, -h            Show this help
```

## Common Benchmark Workflows

### Find the best GPU batch size

```bash
./gzstd-benchmark.sh --sweep-batches
```

### Find the best GPU stream count

```bash
./gzstd-benchmark.sh --sweep-streams
```

### Explore combined batch x stream configurations

```bash
./gzstd-benchmark.sh --sweep-batches --sweep-streams
```

### CPU thread scaling only

```bash
./gzstd-benchmark.sh --cpu-only --sweep-threads
```

### Hybrid thread scaling only

```bash
./gzstd-benchmark.sh --hybrid-only --sweep-threads
```

### CPU compression-level sweep

```bash
./gzstd-benchmark.sh --cpu-only --sweep-levels
```

### Compression only (skip decompression)

```bash
./gzstd-benchmark.sh --sweep-all --no-decompress
```

### Single file only

```bash
./gzstd-benchmark.sh --sweep-all --files "medium_compress.bin"
```

### Larger data set for more stable measurements

```bash
./gzstd-gendata.sh 2048 ./gzstd-testdata-2g
./gzstd-benchmark.sh --data-dir ./gzstd-testdata-2g --sweep-all --iterations 5
```

### Compare two `gzstd` builds

```bash
./gzstd-benchmark.sh --gzstd ./build-old/gzstd --output old-results.json
./gzstd-benchmark.sh --gzstd ./build/gzstd --output new-results.json
```

## Output

The script prints:

- a startup summary,
- the discovered test files,
- a live progress line with ETA,
- one completed-result line per test,
- a compression summary table,
- an optional decompression summary table,
- best-throughput and best-ratio summaries,
- and JSON output written to the file named by `--output`.

### Example result shape

```text
-- COMPRESSION --
Config                 File                       Size    Time(s)    GiB/s   Ratio%
------                 ----                       ----    -------    -----   ------
cpu-default            medium_compress.bin      512MiB     1.2340    0.406    34.2%
gpu-only               medium_compress.bin      512MiB     0.5670    0.884    34.2%
gpu-batch32            medium_compress.bin      512MiB     0.4980    1.006    34.2%
...
```

### JSON structure

The JSON payload contains:

- `gzstd_version`
- `system.cpus`
- `system.has_gpu`
- `iterations`
- `options.*`
- `results[]`

Each result row includes:

- `config`
- `file`
- `mode`
- `file_bytes`
- `comp_bytes`
- `median_secs`
- `throughput_gibs`
- `ratio_pct`

## Notes and Benchmarking Tips

- Larger files produce more stable results because startup overhead matters less.
- More iterations (for example `5`) usually give more reliable medians.
- The script attempts to drop Linux page cache when permitted, but it continues if that is unavailable.
- The first GPU run can be slower due to device/runtime initialization.
- On shared systems, back-to-back A/B comparisons are usually more meaningful than absolute numbers.
- Decompression correctness is verified with `diff` on the final iteration for each config/file pair.

## Revision Notes for This Benchmark Script

This section replaces a standalone benchmark-script changelog for this update.

### What changed in gzstd-gendata.sh (v0.11.37)

- **Argument order changed**: `size_mib` is now the first argument, `output_dir` is now the
  second. Passing only one argument sets the file size; the output directory defaults to
  `./gzstd-testdata`.
- **Parallel generation**: all five files are now generated simultaneously in background
  processes (up to 5 parallel jobs, auto-detected via `nproc`/`sysctl`).
- **Python3 generators**: all file generation moved from bash loops to inline `python3 -c`
  scripts, eliminating per-chunk fork overhead and dramatically reducing wall time.
- **64 MiB write chunks**: up from 64 KiB, reducing loop iterations ~1000x for large files.
- **NumPy fast path**: `medium_compress.bin` now uses NumPy's SIMD-vectorised array operations
  for the random-printable-ASCII pass when NumPy is available, falling back to `array.array`
  from the stdlib otherwise. The startup banner reports which path is active and the detected
  NumPy version.
- **Live progress display**: a multi-line ANSI progress view redraws in-place at 250 ms
  intervals, showing per-file percentage, GiB written, and a ✔/✘ completion marker. The
  terminal cursor is hidden during the run and restored on exit or Ctrl-C.
- **Version variable**: version `0.11.37` is declared as `VERSION` at the top of the script
  and displayed in the startup banner.

### What changed in this revision

- Synchronized the documented sweep values with the actual script behavior.
- Clarified that compression-level sweeps are CPU-only.
- Expanded `--sweep-threads` so it can benchmark hybrid baseline thread counts in addition to CPU-only thread counts.
- Replaced the old `--help` extraction approach with an explicit help block so usage text stays accurate.
- Made GPU capability detection more tolerant of version-banner wording by checking for `nvCOMP`, `CUDA`, or `GPU` tokens.
- Reworked progress parsing so it does not depend on `grep -P`.
- Improved median calculation so even-sized sample sets are averaged correctly.
- Added more explicit validation for incompatible mode flags and invalid iteration counts.
- Expanded JSON metadata to record the selected sweep options.

### Why these changes were made

The previous benchmark script and benchmark documentation had drifted apart in a few places, especially around batch-size and compression-level sweep values. The earlier script also limited thread sweeps to CPU-only mode, which made it harder to compare hybrid thread-scaling behavior. This revision aligns the user-facing documentation with the implementation and broadens the benchmark matrix in a controlled way.
