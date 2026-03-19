# gzstd Benchmark Suite

## Quick Start

```bash
# 1. Generate test data (512 MiB per file, ~2.5 GiB total)
./gzstd-gendata.sh ./testdata 512

# 2. Run quick benchmark (reduced sweep, 1 iteration)
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --quick --sweep-all

# 3. Run full benchmark (3 iterations per config)
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-all
```

## Test Data

`gzstd-gendata.sh` creates 5 files with different compressibility:

| File | Description | Typical Ratio |
|------|-------------|---------------|
| `high_compress.bin` | Repeated log lines | ~0.1% |
| `medium_compress.bin` | English-like text | ~30-40% |
| `low_compress.bin` | Structured random | ~85-95% |
| `mixed.bin` | Alternating text/binary | ~50-60% |
| `zeros.bin` | All zeros | ~0.01% |

## Benchmark Sweeps

| Flag | What it tests | Default values |
|------|--------------|----------------|
| `--sweep-batches` | GPU batch sizes | 4, 8, 16, 32, 64, 128 |
| `--sweep-streams` | GPU streams per device | 1, 2, 3, 4, 6, 8 |
| `--sweep-threads` | CPU thread counts | 1, 2, 4, 8, 16, N-1 |
| `--sweep-levels` | Compression levels | 1, 3, 6, 9, 12, 15, 19 |
| `--sweep-all` | All of the above |  |

## Targeted Testing

```bash
# Find optimal GPU batch size for your hardware
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-batches

# Find optimal GPU stream count
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-streams

# Test combined batch × stream configurations
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-batches --sweep-streams

# CPU thread scaling only
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-threads --cpu-only

# Compression only (skip decompress, faster)
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-all --no-decompress

# Single file only
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata --sweep-all --files "medium_compress.bin"

# Larger test data for more stable measurements
./gzstd-gendata.sh ./testdata-2g 2048
./gzstd-benchmark.sh --gzstd ./build/gzstd --data-dir ./testdata-2g --sweep-all --iterations 5
```

## Output

Results are printed as a table and saved to `benchmark-results.json`:

```
--- COMPRESSION ---
Config                 File                       Size    Time(s)   GiB/s  Ratio%
------                 ----                       ----    -------   -----  ------
cpu-default            medium_compress.bin       512MiB      1.234   0.406   34.2%
gpu-default            medium_compress.bin       512MiB      0.567   0.884   34.2%
gpu-batch32            medium_compress.bin       512MiB      0.498   1.006   34.2%
...

--- OPTIMAL CONFIGURATIONS ---
Best compression throughput:
  medium_compress.bin  -> gpu-batch32               1.006 GiB/s
Best decompression throughput:
  medium_compress.bin  -> gpu-default               2.341 GiB/s
```

## Tips

- **Larger files** give more stable measurements (less noise from startup overhead)
- **More iterations** (5+) give more reliable medians
- **Drop caches**  the script attempts this automatically (needs root)
- **Warm GPU**  first run may be slower due to GPU initialization; the median handles this
- The script verifies decompression correctness via `diff` on the last iteration
