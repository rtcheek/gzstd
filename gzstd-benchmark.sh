#!/usr/bin/env bash
#======================================================================
# gzstd-benchmark.sh  Automated benchmark suite for gzstd
#
# Sweeps across GPU batch sizes, stream counts, CPU thread counts,
# and compression levels.  Measures compression AND decompression
# throughput for each configuration.  Reports results as a table
# and optionally as JSON.
#
# Usage:
#   ./gzstd-benchmark.sh [options]
#
# Options:
#   --gzstd PATH        Path to gzstd binary (default: ./gzstd)
#   --data-dir DIR      Directory with test files (default: ./gzstd-testdata)
#   --output FILE       JSON output file (default: benchmark-results.json)
#   --iterations N      Runs per config, reports median (default: 3)
#   --files PATTERN     Only test files matching glob (default: all)
#   --quick             Reduced sweep (fewer configs, 1 iteration)
#   --gpu-only          Only test GPU configs (skip CPU-only and hybrid)
#   --cpu-only          Only test CPU configs (skip GPU)
#   --hybrid-only       Only test hybrid configs (skip CPU-only and GPU-only)
#   --sweep-batches     Sweep GPU batch sizes (4,8,16,32,64,128)
#   --sweep-streams     Sweep GPU stream counts (1,2,3,4,6,8)
#   --sweep-threads     Sweep CPU thread counts (1,2,4,8,16,N-1)
#   --sweep-levels      Sweep compression levels (1,3,6,9,15,19)
#   --sweep-all         Enable all sweeps
#   --no-decompress     Skip decompression benchmarks
#   --help              Show this help
#
# The script generates a temporary compressed file for each config,
# measures wall-clock time, and computes throughput in GiB/s.
#======================================================================
set -euo pipefail

#----------------------------------------------------------------------
# Defaults
#----------------------------------------------------------------------
GZSTD="${GZSTD:-./gzstd}"
DATA_DIR="./gzstd-testdata"
OUTPUT="benchmark-results.json"
ITERATIONS=3
FILE_PATTERN="*"
QUICK=false
GPU_ONLY=false
CPU_ONLY=false
HYBRID_ONLY=false
SWEEP_BATCHES=false
SWEEP_STREAMS=false
SWEEP_THREADS=false
SWEEP_LEVELS=false
DO_DECOMPRESS=true

#----------------------------------------------------------------------
# Parse arguments
#----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gzstd)       GZSTD="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --output)      OUTPUT="$2"; shift 2 ;;
    --iterations)  ITERATIONS="$2"; shift 2 ;;
    --files)       FILE_PATTERN="$2"; shift 2 ;;
    --quick)       QUICK=true; ITERATIONS=1; shift ;;
    --gpu-only)    GPU_ONLY=true; shift ;;
    --cpu-only)    CPU_ONLY=true; shift ;;
    --hybrid-only) HYBRID_ONLY=true; shift ;;
    --sweep-batches) SWEEP_BATCHES=true; shift ;;
    --sweep-streams) SWEEP_STREAMS=true; shift ;;
    --sweep-threads) SWEEP_THREADS=true; shift ;;
    --sweep-levels)  SWEEP_LEVELS=true; shift ;;
    --sweep-all)   SWEEP_BATCHES=true; SWEEP_STREAMS=true
                   SWEEP_THREADS=true; SWEEP_LEVELS=true; shift ;;
    --no-decompress) DO_DECOMPRESS=false; shift ;;
    --help|-h)
      head -35 "$0" | tail -30
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

#----------------------------------------------------------------------
# Validation
#----------------------------------------------------------------------
if [[ ! -x "$GZSTD" ]]; then
  echo "ERROR: gzstd binary not found at '$GZSTD'"
  echo "       Use --gzstd PATH or build first."
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Test data directory '$DATA_DIR' not found."
  echo "       Run gzstd-gendata.sh first to generate test data."
  exit 1
fi

# Detect hardware
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
HAS_GPU=false
if "$GZSTD" --version 2>&1 | grep -q "nvCOMP"; then
  HAS_GPU=true
fi

if $GPU_ONLY && ! $HAS_GPU; then
  echo "ERROR: --gpu-only requested but gzstd was built without nvCOMP"
  exit 1
fi

echo "============================================================"
echo " gzstd benchmark suite"
echo "============================================================"
echo " Binary     : $GZSTD"
echo " Version    : $($GZSTD --version 2>&1 | head -1)"
echo " Data dir   : $DATA_DIR"
echo " Iterations : $ITERATIONS"
echo " CPUs       : $NCPU"
echo " GPU        : $HAS_GPU"
echo " Output     : $OUTPUT"
echo "============================================================"
echo ""

#----------------------------------------------------------------------
# Build configuration matrix
#----------------------------------------------------------------------
# Each config is a string: "label|comp_flags|decomp_flags"

CONFIGS=()

# Baseline: defaults
if ! $GPU_ONLY && ! $HYBRID_ONLY; then
  CONFIGS+=("cpu-default|--cpu-only|--cpu-only")
fi

if $HAS_GPU && ! $CPU_ONLY; then
  if ! $HYBRID_ONLY; then
    CONFIGS+=("gpu-only|--gpu-only|--gpu-only")
  fi
  if ! $GPU_ONLY; then
    CONFIGS+=("hybrid|--hybrid|--hybrid")
  fi
fi

# Sweep GPU batch sizes
if $SWEEP_BATCHES && $HAS_GPU && ! $CPU_ONLY; then
  if $QUICK; then
    BATCH_SIZES="8 32 128"
  else
    BATCH_SIZES="4 8 16 32 64 128"
  fi
  for b in $BATCH_SIZES; do
    if ! $HYBRID_ONLY; then
      CONFIGS+=("gpu-batch${b}|--gpu-only --gpu-batch $b|--gpu-only")
    fi
    if ! $GPU_ONLY; then
      CONFIGS+=("hyb-batch${b}|--hybrid --gpu-batch $b|--hybrid")
    fi
  done
fi

# Sweep GPU stream counts
if $SWEEP_STREAMS && $HAS_GPU && ! $CPU_ONLY; then
  if $QUICK; then
    STREAM_COUNTS="1 3 6"
  else
    STREAM_COUNTS="1 2 3 4 6 8"
  fi
  for s in $STREAM_COUNTS; do
    if ! $HYBRID_ONLY; then
      CONFIGS+=("gpu-streams${s}|--gpu-only --gpu-streams $s|--gpu-only")
    fi
    if ! $GPU_ONLY; then
      CONFIGS+=("hyb-streams${s}|--hybrid --gpu-streams $s|--hybrid")
    fi
  done
fi

# Sweep CPU thread counts
if $SWEEP_THREADS && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then
    THREAD_COUNTS="1 $((NCPU/2)) $((NCPU-1))"
  else
    THREAD_COUNTS="1 2 4 8"
    if [[ $NCPU -gt 8 ]]; then
      THREAD_COUNTS="$THREAD_COUNTS 16"
    fi
    if [[ $NCPU -gt 16 ]]; then
      THREAD_COUNTS="$THREAD_COUNTS $((NCPU-1))"
    fi
  fi
  for t in $THREAD_COUNTS; do
    [[ $t -lt 1 ]] && continue
    [[ $t -ge $NCPU ]] && continue
    CONFIGS+=("cpu-T${t}|--cpu-only -T $t|--cpu-only -T $t")
  done
fi

# Sweep compression levels
if $SWEEP_LEVELS && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then
    LEVELS="1 6 19"
  else
    LEVELS="1 3 6 9 12 15 19"
  fi
  for l in $LEVELS; do
    CONFIGS+=("cpu-level${l}|--cpu-only -$l|--cpu-only")
  done
fi

# Combined GPU sweeps: batch x streams (if both enabled)
if $SWEEP_BATCHES && $SWEEP_STREAMS && $HAS_GPU && ! $CPU_ONLY && ! $QUICK; then
  for b in 16 32 64; do
    for s in 2 4 6; do
      if ! $HYBRID_ONLY; then
        CONFIGS+=("gpu-b${b}s${s}|--gpu-only --gpu-batch $b --gpu-streams $s|--gpu-only")
      fi
      if ! $GPU_ONLY; then
        CONFIGS+=("hyb-b${b}s${s}|--hybrid --gpu-batch $b --gpu-streams $s|--hybrid")
      fi
    done
  done
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configurations to test. Use --sweep-all or check flags."
  exit 1
fi

echo "Configurations to test: ${#CONFIGS[@]}"
echo ""

#----------------------------------------------------------------------
# Collect test files
#----------------------------------------------------------------------
TEST_FILES=()
for f in "$DATA_DIR"/$FILE_PATTERN; do
  [[ -f "$f" ]] || continue
  # Skip .zst files (compressed outputs from previous runs)
  [[ "$f" == *.zst ]] && continue
  TEST_FILES+=("$f")
done

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No test files found in $DATA_DIR matching '$FILE_PATTERN'"
  exit 1
fi

echo "Test files:"
for f in "${TEST_FILES[@]}"; do
  echo "  $(basename "$f")  $(numfmt --to=iec-i --suffix=B $(stat -c%s "$f"))"
done
echo ""

#----------------------------------------------------------------------
# Helper: run a single timed test, return elapsed seconds
#----------------------------------------------------------------------
run_timed() {
  local start end elapsed rc
  # Sync and drop caches if possible (Linux)
  sync 2>/dev/null || true
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  fi

  start=$(date +%s.%N)
  # Run the command; suppress stderr (progress/summary) but log to file for debugging
  "$@" 2>"$TMPDIR/last_stderr.log" || rc=$?
  end=$(date +%s.%N)

  # If command failed, show the full command and stderr for debugging
  if [[ ${rc:-0} -ne 0 ]]; then
    echo "" >&2
    echo "  [DEBUG] FAILED (rc=$rc): $*" >&2
    if [[ -s "$TMPDIR/last_stderr.log" ]]; then
      echo "  [DEBUG] stderr: $(head -3 "$TMPDIR/last_stderr.log")" >&2
    fi
  fi

  # Float subtraction  try python3, then bc, then awk
  elapsed=$(python3 -c "print(f'{$end - $start:.4f}')" 2>/dev/null) \
    || elapsed=$(echo "$end - $start" | bc -l 2>/dev/null) \
    || elapsed=$(awk "BEGIN{printf \"%.4f\", $end - $start}" 2>/dev/null) \
    || elapsed="0.001"
  echo "$elapsed"
}

#----------------------------------------------------------------------
# Helper: compute median of N values
#----------------------------------------------------------------------
median() {
  local sorted
  sorted=$(echo "$@" | tr ' ' '\n' | sort -g)
  local count=$(echo "$sorted" | wc -l)
  local mid=$(( (count + 1) / 2 ))
  local val
  val=$(echo "$sorted" | sed -n "${mid}p")
  # Truncate to 4 decimal places for display consistency
  printf "%.4f" "$val"
}

#----------------------------------------------------------------------
# Helper: format throughput
#----------------------------------------------------------------------
throughput_gibs() {
  local bytes="$1" secs="$2"
  python3 -c "print(f'{$bytes / 1073741824 / $secs:.3f}')" 2>/dev/null \
    || echo "scale=3; $bytes / 1073741824 / $secs" | bc -l 2>/dev/null \
    || echo "0"
}

#----------------------------------------------------------------------
# Run benchmarks
#----------------------------------------------------------------------
TMPDIR=$(mktemp -d /tmp/gzstd-bench.XXXXXX)
trap "rm -rf $TMPDIR" EXIT

# Results accumulator (tab-separated for easy processing)
RESULTS_FILE="$TMPDIR/results.tsv"
echo -e "config\tfile\tmode\tfile_bytes\tcomp_bytes\tmedian_secs\tthroughput_gibs\tratio_pct" > "$RESULTS_FILE"

total_configs=${#CONFIGS[@]}
total_files=${#TEST_FILES[@]}
decomp_mult=1
if $DO_DECOMPRESS; then decomp_mult=2; fi
total_tests=$(( total_configs * total_files * decomp_mult ))
test_num=0
BENCH_START=$(date +%s.%N)

#----------------------------------------------------------------------
# ANSI helpers
#----------------------------------------------------------------------
BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
WHITE=$'\033[37m'
RESET=$'\033[0m'
CLEAR_LINE=$'\033[2K'
COLS=$(tput cols 2>/dev/null || echo 80)

draw_bar() {
  # draw_bar <fraction> <width> [<fill_char>] [<empty_char>]
  local frac="$1" width="$2"
  local fill="${3:-}" empty="${4:-}"
  local filled=$(python3 -c "print(int($frac * $width + 0.5))" 2>/dev/null || echo 0)
  local remaining=$(( width - filled ))
  local bar=""
  for ((i=0; i<filled; i++)); do bar+="$fill"; done
  for ((i=0; i<remaining; i++)); do bar+="$empty"; done
  echo -n "$bar"
}

format_eta() {
  local secs="$1"
  if (( secs < 0 )); then secs=0; fi
  if (( secs < 60 )); then
    printf "%ds" "$secs"
  elif (( secs < 3600 )); then
    printf "%dm%02ds" "$((secs/60))" "$((secs%60))"
  else
    printf "%dh%02dm" "$((secs/3600))" "$(( (secs%3600)/60 ))"
  fi
}

print_status() {
  local num="$1" total="$2" label="$3" file="$4" mode="$5"
  local last_time="${6:-}" last_thr="${7:-}"

  local pct=0
  if (( total > 0 )); then
    pct=$(python3 -c "print(f'{$num/$total*100:.0f}')" 2>/dev/null || echo "0")
  fi
  local frac=$(python3 -c "print(f'{$num/$total:.4f}')" 2>/dev/null || echo "0")

  # Calculate ETA
  local now elapsed eta_s eta_str
  now=$(date +%s.%N)
  elapsed=$(python3 -c "print(f'{$now - $BENCH_START:.1f}')" 2>/dev/null || echo "0")
  if [[ "$frac" != "0" && "$frac" != "0.0000" ]]; then
    eta_s=$(python3 -c "
e=$elapsed; f=$frac
if f > 0: print(int(e/f - e + 0.5))
else: print(0)
" 2>/dev/null || echo "0")
  else
    eta_s=0
  fi
  eta_str=$(format_eta "$eta_s")

  # Overall progress bar
  local bar_width=30
  local bar=$(draw_bar "$frac" "$bar_width")

  # Build the display
  printf "\r${CLEAR_LINE}"
  printf "${BOLD}${CYAN}[%d/%d]${RESET} " "$num" "$total"
  printf "${GREEN}%s${RESET} " "$bar"
  printf "${BOLD}%s%%${RESET} " "$pct"

  # Current test info
  printf "${DIM}│${RESET} "
  printf "${YELLOW}%-14s${RESET} " "$label"
  printf "%-18s " "$file"
  if [[ "$mode" == "compress" ]]; then
    printf "${CYAN}⟫ compress${RESET}"
  else
    printf "${GREEN}⟪ decompress${RESET}"
  fi

  # Last result
  if [[ -n "$last_time" ]]; then
    printf "  ${DIM}│${RESET} ${WHITE}${last_time}s${RESET} ${DIM}(${last_thr} GiB/s)${RESET}"
  fi

  # ETA
  printf "  ${DIM}ETA %s${RESET}" "$eta_str"
}

# Print a completed result line (scrolls up)
print_result() {
  local label="$1" file="$2" mode="$3" time="$4" thr="$5" ratio="$6"
  local icon
  if [[ "$mode" == "compress" ]]; then
    icon="${CYAN}⟫${RESET}"
  else
    icon="${GREEN}⟪${RESET}"
  fi

  printf "\r${CLEAR_LINE}"
  printf "  ${DIM}✓${RESET} %-16s %-20s %s %-10s " "$label" "$file" "$icon" "$mode"
  printf "${BOLD}%8ss${RESET}  %s GiB/s" "$time" "$thr"
  if [[ "$mode" == "compress" && -n "$ratio" && "$ratio" != "?" ]]; then
    printf "  ${DIM}(%s%%)${RESET}" "$ratio"
  fi
  echo ""
}

echo ""
echo "${BOLD}Starting benchmark: ${total_tests} tests across ${total_configs} configs × ${total_files} files${RESET}"
echo "${DIM}$(printf '─%.0s' $(seq 1 $COLS))${RESET}"

last_time=""
last_thr=""

for config_str in "${CONFIGS[@]}"; do
  IFS='|' read -r label comp_flags decomp_flags <<< "$config_str"

  for test_file in "${TEST_FILES[@]}"; do
    file_base=$(basename "$test_file")
    file_bytes=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null)
    comp_out="$TMPDIR/compressed.zst"

    #--- Compression benchmark ---
    test_num=$((test_num + 1))
    print_status "$test_num" "$total_tests" "$label" "$file_base" "compress" "$last_time" "$last_thr"

    times_c=()
    for ((i=1; i<=ITERATIONS; i++)); do
      rm -f "$comp_out"
      # shellcheck disable=SC2086
      elapsed=$(run_timed "$GZSTD" $comp_flags -q -f --output="$comp_out" "$test_file")
      times_c+=("$elapsed")
    done

    median_c=$(median "${times_c[@]}")

    # Verify compressed output exists (debug first file)
    if [[ ! -f "$comp_out" ]]; then
      echo "" >&2
      echo "  [DEBUG] compressed output not found at: $comp_out" >&2
      echo "  [DEBUG] TMPDIR contents: $(ls -la "$TMPDIR"/ 2>&1 | head -5)" >&2
      echo "  [DEBUG] Last command stderr:" >&2
      cat "$TMPDIR/last_stderr.log" >&2 2>/dev/null
    fi

    comp_bytes=$(stat -c%s "$comp_out" 2>/dev/null || stat -f%z "$comp_out" 2>/dev/null || echo "0")
    thr_c=$(throughput_gibs "$file_bytes" "$median_c")
    ratio=$(python3 -c "print(f'{$comp_bytes * 100 / $file_bytes:.1f}')" 2>/dev/null || echo "?")

    last_time="$median_c"
    last_thr="$thr_c"

    echo -e "${label}\t${file_base}\tcompress\t${file_bytes}\t${comp_bytes}\t${median_c}\t${thr_c}\t${ratio}" >> "$RESULTS_FILE"
    print_result "$label" "$file_base" "compress" "$median_c" "$thr_c" "$ratio"

    #--- Decompression benchmark ---
    if $DO_DECOMPRESS; then
      test_num=$((test_num + 1))
      print_status "$test_num" "$total_tests" "$label" "$file_base" "decompress" "$last_time" "$last_thr"

      decomp_out="$TMPDIR/decompressed.bin"
      times_d=()
      for ((i=1; i<=ITERATIONS; i++)); do
        rm -f "$decomp_out"
        # Use decomp_flags if set, otherwise just -d
        local_flags="-d"
        if [[ -n "$decomp_flags" ]]; then
          local_flags="$decomp_flags -d"
        fi
        # shellcheck disable=SC2086
        elapsed=$(run_timed "$GZSTD" $local_flags -q -f --output="$decomp_out" "$comp_out")
        times_d+=("$elapsed")
      done

      median_d=$(median "${times_d[@]}")
      thr_d=$(throughput_gibs "$file_bytes" "$median_d")

      last_time="$median_d"
      last_thr="$thr_d"

      echo -e "${label}\t${file_base}\tdecompress\t${file_bytes}\t${comp_bytes}\t${median_d}\t${thr_d}\t${ratio}" >> "$RESULTS_FILE"
      print_result "$label" "$file_base" "decompress" "$median_d" "$thr_d" "$ratio"

      # Verify correctness on last iteration
      if [[ ! -f "$decomp_out" ]]; then
        echo ""
        echo "  WARNING: decompression output missing for $label / $file_base!"
      elif ! diff -q "$test_file" "$decomp_out" > /dev/null 2>&1; then
        orig_sz=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null || echo "?")
        dec_sz=$(stat -c%s "$decomp_out" 2>/dev/null || stat -f%z "$decomp_out" 2>/dev/null || echo "?")
        echo ""
        echo "  WARNING: decompression mismatch for $label / $file_base (orig=$orig_sz dec=$dec_sz)"
      fi
      rm -f "$decomp_out"
    fi

    rm -f "$comp_out"
  done
done

# Final summary line
total_elapsed=$(python3 -c "
import time; print(f'{time.time() - $BENCH_START:.1f}')
" 2>/dev/null || echo "?")
printf "\r${CLEAR_LINE}"
echo "${DIM}$(printf '─%.0s' $(seq 1 $COLS))${RESET}"
echo "${BOLD}${GREEN}✓ Benchmark complete${RESET}  ${total_tests} tests in ${total_elapsed}s"
echo ""

#----------------------------------------------------------------------
# Print results table
#----------------------------------------------------------------------
echo "${BOLD}============================================================${RESET}"
echo "${BOLD} BENCHMARK RESULTS${RESET}"
echo "${BOLD}============================================================${RESET}"
echo ""

# Compression results
echo "${BOLD}${CYAN}── COMPRESSION ──${RESET}"
printf "${DIM}%-22s %-20s %10s %10s %8s %8s${RESET}\n" \
       "Config" "File" "Size" "Time(s)" "GiB/s" "Ratio%"
printf "${DIM}%-22s %-20s %10s %10s %8s %8s${RESET}\n" \
       "──────" "────" "────" "───────" "─────" "──────"
grep "compress[^_]" "$RESULTS_FILE" | tail -n +1 | while IFS=$'\t' read -r cfg file mode fbytes cbytes secs thr ratio; do
  [[ "$mode" == "compress" ]] || continue
  fsize_h=$(numfmt --to=iec-i --suffix=B "$fbytes" 2>/dev/null || echo "${fbytes}B")
  printf "%-22s %-20s %10s %8.4f %8.3f %7s%%\n" \
         "$cfg" "$file" "$fsize_h" "$secs" "$thr" "$ratio"
done

echo ""

# Decompression results
if $DO_DECOMPRESS; then
  echo "${BOLD}${GREEN}── DECOMPRESSION ──${RESET}"
  printf "${DIM}%-22s %-20s %10s %10s %8s${RESET}\n" \
         "Config" "File" "Size" "Time(s)" "GiB/s"
  printf "%-22s %-20s %10s %10s %8s\n" \
         "------" "----" "----" "-------" "-----"
  grep "decompress" "$RESULTS_FILE" | while IFS=$'\t' read -r cfg file mode fbytes cbytes secs thr ratio; do
    fsize_h=$(numfmt --to=iec-i --suffix=B "$fbytes" 2>/dev/null || echo "${fbytes}B")
    printf "%-22s %-20s %10s %8.4f %8.3f\n" \
           "$cfg" "$file" "$fsize_h" "$secs" "$thr"
  done
  echo ""
fi

#----------------------------------------------------------------------
# Find optimal configurations
#----------------------------------------------------------------------
echo "${BOLD}${YELLOW}── OPTIMAL CONFIGURATIONS ──${RESET}"
echo ""

# Best compression throughput per file
echo "Best compression throughput:"
for test_file in "${TEST_FILES[@]}"; do
  file_base=$(basename "$test_file")
  best=$(grep "compress[^_]" "$RESULTS_FILE" | grep "$file_base" | grep "compress" |
         sort -t$'\t' -k7 -rg | head -1)
  if [[ -n "$best" ]]; then
    cfg=$(echo "$best" | cut -f1)
    thr=$(echo "$best" | cut -f7)
    printf "  %-20s -> %-22s  %.3f GiB/s\n" "$file_base" "$cfg" "$thr"
  fi
done

echo ""

if $DO_DECOMPRESS; then
  echo "Best decompression throughput:"
  for test_file in "${TEST_FILES[@]}"; do
    file_base=$(basename "$test_file")
    best=$(grep "decompress" "$RESULTS_FILE" | grep "$file_base" |
           sort -t$'\t' -k7 -rg | head -1)
    if [[ -n "$best" ]]; then
      cfg=$(echo "$best" | cut -f1)
      thr=$(echo "$best" | cut -f7)
      printf "  %-20s -> %-22s  %.3f GiB/s\n" "$file_base" "$cfg" "$thr"
    fi
  done
  echo ""
fi

# Best compression ratio per file
echo "Best compression ratio:"
for test_file in "${TEST_FILES[@]}"; do
  file_base=$(basename "$test_file")
  best=$(grep "compress[^_]" "$RESULTS_FILE" | grep "$file_base" | grep "compress" |
         sort -t$'\t' -k8 -g | head -1)
  if [[ -n "$best" ]]; then
    cfg=$(echo "$best" | cut -f1)
    ratio=$(echo "$best" | cut -f8)
    printf "  %-20s -> %-22s  %s%%\n" "$file_base" "$cfg" "$ratio"
  fi
done

echo ""

#----------------------------------------------------------------------
# Write JSON output
#----------------------------------------------------------------------
echo "Writing JSON results to $OUTPUT..."
GZSTD_VER=$("$GZSTD" --version 2>&1 | head -1)
if $HAS_GPU; then GPU_JSON="True"; else GPU_JSON="False"; fi
python3 -c "
import json, sys

results = []
with open('$RESULTS_FILE') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        # Convert numeric fields
        for k in ['file_bytes', 'comp_bytes']:
            try: row[k] = int(row[k])
            except: pass
        for k in ['median_secs', 'throughput_gibs']:
            try: row[k] = float(row[k])
            except: pass
        results.append(row)

output = {
    'gzstd_version': '$GZSTD_VER',
    'system': {
        'cpus': $NCPU,
        'has_gpu': $GPU_JSON,
    },
    'iterations': $ITERATIONS,
    'results': results,
}

with open('$OUTPUT', 'w') as f:
    json.dump(output, f, indent=2)

print(f'  {len(results)} results written.')
" 2>&1 || echo "  (JSON output failed)"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo "============================================================"
