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
#   --sweep-ultra       Sweep ultra compression levels (20,21,22 with --ultra)
#   --sweep-all         Enable all sweeps (including --sweep-ultra)
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
# Auto-discover gzstd binary: GZSTD env var > ./gzstd > ./build/gzstd
if [[ -n "${GZSTD:-}" ]]; then
  :  # user set GZSTD env var, use it
elif [[ -x "./gzstd" ]]; then
  GZSTD="./gzstd"
elif [[ -x "./build/gzstd" ]]; then
  GZSTD="./build/gzstd"
else
  GZSTD="./gzstd"  # fall through to validation error below
fi
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
SWEEP_ULTRA=false
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
    --sweep-ultra)   SWEEP_ULTRA=true; shift ;;
    --sweep-all)   SWEEP_BATCHES=true; SWEEP_STREAMS=true
                   SWEEP_THREADS=true; SWEEP_LEVELS=true; SWEEP_ULTRA=true; shift ;;
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
  echo "ERROR: gzstd binary not found (tried ./gzstd and ./build/gzstd)"
  echo "       Use --gzstd PATH, set GZSTD env var, or build first."
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Test data directory '$DATA_DIR' not found."
  echo "       Run gzstd-gendata.sh first to generate test data."
  exit 1
fi

# Detect hardware
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
HAS_GPU_BUILD=false
if "$GZSTD" --version 2>&1 | grep -q "nvCOMP"; then
  HAS_GPU_BUILD=true
fi

if $GPU_ONLY && ! $HAS_GPU_BUILD; then
  echo "ERROR: --gpu-only requested but gzstd was built without nvCOMP"
  exit 1
fi

GPU_SUMMARY="none detected"
GPU_DRIVER=""
GPU_DETAILS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_DETAILS < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
  if [[ ${#GPU_DETAILS[@]} -gt 0 ]]; then
    GPU_SUMMARY="${#GPU_DETAILS[@]} detected"
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  fi
elif command -v lspci >/dev/null 2>&1; then
  mapfile -t GPU_DETAILS < <(lspci | grep -Ei 'vga|3d|display' | grep -Ei 'nvidia|amd|intel' || true)
  if [[ ${#GPU_DETAILS[@]} -gt 0 ]]; then
    GPU_SUMMARY="${#GPU_DETAILS[@]} detected"
  fi
fi

#----------------------------------------------------------------------
# ANSI / display helpers
#----------------------------------------------------------------------
BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
WHITE=$'\033[37m'
ORANGE=$'\033[38;5;208m'
RESET=$'\033[0m'
CLEAR_LINE=$'\033[2K'
COLS=$(tput cols 2>/dev/null || echo 80)
STATUS_LINE=""
CURRENT_GZSTD_PCT=""
CURRENT_GZSTD_LABEL=""

repeat_char() {
  local char="$1" width="$2"
  printf '%*s' "$width" '' | tr ' ' "$char"
}

print_separator() {
  echo "${DIM}$(repeat_char '-' "$COLS")${RESET}"
}

render_status_line() {
  local suffix=""
  if [[ -n "${CURRENT_GZSTD_PCT:-}" ]]; then
    suffix="  ${DIM}|${RESET} ${ORANGE}${CURRENT_GZSTD_LABEL}:${CURRENT_GZSTD_PCT}%${RESET}"
  fi
  printf "\r${CLEAR_LINE}%s%s" "$STATUS_LINE" "$suffix" >&2
}

clear_status_line() {
  printf "\r${CLEAR_LINE}" >&2
}

print_separator
echo "${BOLD}${WHITE} gzstd benchmark suite v0.12.4${RESET}"
print_separator
echo " ${WHITE}Binary${RESET}     : ${YELLOW}$GZSTD${RESET}"
echo " ${WHITE}Version${RESET}    : ${CYAN}$($GZSTD --version 2>&1 | head -1)${RESET}"
echo " ${WHITE}Data dir${RESET}   : ${YELLOW}$DATA_DIR${RESET}"
echo " ${WHITE}Iterations${RESET} : ${GREEN}$ITERATIONS${RESET}"
echo " ${WHITE}CPUs${RESET}       : ${GREEN}$NCPU${RESET}"
echo " ${WHITE}GPU build${RESET}  : ${ORANGE}$HAS_GPU_BUILD${RESET}"
echo " ${WHITE}GPUs${RESET}       : ${YELLOW}$GPU_SUMMARY${RESET}"
if [[ -n "$GPU_DRIVER" ]]; then
  echo " ${WHITE}Driver${RESET}     : ${CYAN}$GPU_DRIVER${RESET}"
fi
for gpu in "${GPU_DETAILS[@]}"; do
  echo " ${WHITE}GPU detail${RESET} : ${YELLOW}$gpu${RESET}"
done
echo " ${WHITE}Output${RESET}     : ${YELLOW}$OUTPUT${RESET}"
print_separator
echo ""

#----------------------------------------------------------------------
# Build configuration matrix
#----------------------------------------------------------------------
# Each config is a string: "label|comp_flags|decomp_flags"

CONFIGS=()

# Baseline: defaults
if ! $GPU_ONLY && ! $HYBRID_ONLY; then
  CONFIGS+=("cpu-only|--cpu-only|--cpu-only")
fi

if $HAS_GPU_BUILD && ! $CPU_ONLY; then
  if ! $HYBRID_ONLY; then
    CONFIGS+=("gpu-only|--gpu-only|--gpu-only")
  fi
  if ! $GPU_ONLY; then
    CONFIGS+=("hybrid|--hybrid|--hybrid")
  fi
fi

# Sweep GPU batch sizes
if $SWEEP_BATCHES && $HAS_GPU_BUILD && ! $CPU_ONLY; then
  if $QUICK; then
    BATCH_SIZES="8 64 256"
  else
    BATCH_SIZES="4 8 16 32 64 128 256 512 1024"
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
if $SWEEP_STREAMS && $HAS_GPU_BUILD && ! $CPU_ONLY; then
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

# Sweep ultra compression levels (20-22).
# gzstd auto-adjusts chunk size and guards against OOM via check_ram_budget,
# so these are safe to run but expect significantly longer wall-clock times.
if $SWEEP_ULTRA && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then
    ULTRA_LEVELS="22"
  else
    ULTRA_LEVELS="20 21 22"
  fi
  for l in $ULTRA_LEVELS; do
    CONFIGS+=("cpu-ultra${l}|--cpu-only --ultra -$l|--cpu-only")
  done
fi

# Combined GPU sweeps: batch x streams (if both enabled)
if $SWEEP_BATCHES && $SWEEP_STREAMS && $HAS_GPU_BUILD && ! $CPU_ONLY && ! $QUICK; then
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

echo "${BOLD}${WHITE}Configurations to test:${RESET} ${CYAN}${#CONFIGS[@]}${RESET}"
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

echo "${BOLD}${WHITE}Test files:${RESET}"
for f in "${TEST_FILES[@]}"; do
  echo "  ${YELLOW}$(basename "$f")${RESET}  ${GREEN}$(numfmt --to=iec-i --suffix=B $(stat -c%s "$f"))${RESET}"
done
echo ""

#----------------------------------------------------------------------
# Helper: run a single timed test, return elapsed seconds
#----------------------------------------------------------------------
run_timed() {
  local start end elapsed rc
  sync 2>/dev/null || true
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  fi

  start=$(date +%s.%N)
  "$@" --progress 2>"$TMPDIR/last_stderr.log" &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    sleep 0.3
    if [[ -s "$TMPDIR/last_stderr.log" ]]; then
      local gzstd_pct gzstd_label stripped
      # Strip ANSI escapes once, reuse for both patterns.
      stripped=$(tail -c 256 "$TMPDIR/last_stderr.log" 2>/dev/null \
                 | sed 's/\x1b\[[0-9;]*m//g')
      # Prefer out: percentage; fall back to in: when out: is not yet available.
      gzstd_pct=$(echo "$stripped" | grep -oP '(?<=out:)[0-9]+\.[0-9]+(?=%)' 2>/dev/null | tail -1)
      if [[ -n "$gzstd_pct" ]]; then
        gzstd_label="out"
      else
        gzstd_pct=$(echo "$stripped" | grep -oP '(?<=in:)[0-9]+\.[0-9]+(?=%)' 2>/dev/null | tail -1)
        gzstd_label="in"
      fi
      if [[ -n "$gzstd_pct" ]]; then
        CURRENT_GZSTD_PCT="$gzstd_pct"
        CURRENT_GZSTD_LABEL="$gzstd_label"
        render_status_line
      fi
    fi
  done
  wait "$pid" 2>/dev/null || rc=$?
  CURRENT_GZSTD_PCT=""
  CURRENT_GZSTD_LABEL=""
  render_status_line

  end=$(date +%s.%N)

  if [[ ${rc:-0} -ne 0 ]]; then
    echo "" >&2
    echo "  [DEBUG] FAILED (rc=$rc): $*" >&2
    if [[ -s "$TMPDIR/last_stderr.log" ]]; then
      echo "  [DEBUG] stderr: $(head -3 "$TMPDIR/last_stderr.log")" >&2
    fi
  fi

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

  local pct=0
  if (( total > 0 )); then
    pct=$(python3 -c "print(f'{$num/$total*100:.0f}')" 2>/dev/null || echo "0")
  fi
  local frac=$(python3 -c "print(f'{$num/$total:.4f}')" 2>/dev/null || echo "0")

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

  local line=""
  printf -v line "%s%s[%d/%d]%s %s%s%%%s "     "$BOLD" "$CYAN" "$num" "$total" "$RESET"     "$BOLD" "$pct" "$RESET"
  line+="${DIM}|${RESET} "
  printf -v line "%s%s%-14s%s %-18s "     "$line" "$YELLOW" "$label" "$RESET" "$file"
  if [[ "$mode" == "compress" ]]; then
    line+="${CYAN}» compress${RESET}"
  else
    line+="${GREEN}« decompress${RESET}"
  fi
  line+=" ${DIM}~${eta_str}${RESET}"

  STATUS_LINE="$line"
  CURRENT_GZSTD_PCT=""
  CURRENT_GZSTD_LABEL=""
  render_status_line
}

# Print a completed result line (scrolls up)
print_result() {
  local label="$1" file="$2" mode="$3" time="$4" thr="$5" ratio="$6"
  local icon

  clear_status_line

  if [[ "$mode" == "compress" ]]; then
    icon="${CYAN}»${RESET}"
  else
    icon="${GREEN}«${RESET}"
  fi

  printf "${CLEAR_LINE}"
  printf "  ${BOLD}${GREEN}✓${RESET} %-16s %-20s %s %-10s " "$label" "$file" "$icon" "$mode"
  printf "${BOLD}%8ss${RESET}  %s GiB/s" "$time" "$thr"
  if [[ "$mode" == "compress" && -n "$ratio" && "$ratio" != "?" ]]; then
    printf "  ${ORANGE}(%s%%)${RESET}" "$ratio"
  fi
  echo ""
}


echo ""
echo "${BOLD}Starting benchmark: ${total_tests} tests across ${total_configs} configs × ${total_files} files${RESET}"
print_separator

for config_str in "${CONFIGS[@]}"; do
  IFS='|' read -r label comp_flags decomp_flags <<< "$config_str"

  for test_file in "${TEST_FILES[@]}"; do
    file_base=$(basename "$test_file")
    file_bytes=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null)
    comp_out="$TMPDIR/compressed.zst"

    #--- Compression benchmark ---
    test_num=$((test_num + 1))
    print_status "$test_num" "$total_tests" "$label" "$file_base" "compress"

    times_c=()
    for ((i=1; i<=ITERATIONS; i++)); do
      rm -f "$comp_out"
      # shellcheck disable=SC2086
      elapsed=$(run_timed "$GZSTD" $comp_flags -f --output="$comp_out" "$test_file")
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


    echo -e "${label}\t${file_base}\tcompress\t${file_bytes}\t${comp_bytes}\t${median_c}\t${thr_c}\t${ratio}" >> "$RESULTS_FILE"
    print_result "$label" "$file_base" "compress" "$median_c" "$thr_c" "$ratio"

    #--- Decompression benchmark ---
    if $DO_DECOMPRESS; then
      test_num=$((test_num + 1))
      print_status "$test_num" "$total_tests" "$label" "$file_base" "decompress"

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
        elapsed=$(run_timed "$GZSTD" $local_flags -f --output="$decomp_out" "$comp_out")
        times_d+=("$elapsed")
      done

      median_d=$(median "${times_d[@]}")
      thr_d=$(throughput_gibs "$file_bytes" "$median_d")


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
print_separator
echo "${BOLD}${GREEN}✓ Benchmark complete${RESET}  ${total_tests} tests in ${total_elapsed}s"
echo ""

#----------------------------------------------------------------------
# Print results table
#----------------------------------------------------------------------
print_separator
echo "${BOLD}${WHITE} BENCHMARK RESULTS${RESET}"

cfg_rule=$(repeat_char '-' 22)
file_rule=$(repeat_char '-' 20)
size_rule=$(repeat_char '-' 10)
time_rule=$(repeat_char '-' 10)
thr_rule=$(repeat_char '-' 8)
ratio_rule=$(repeat_char '-' 8)

echo ""
echo "${BOLD}${CYAN} COMPRESSION${RESET}"
printf "${DIM}${WHITE}%-22s %-20s %10s %10s %8s %8s${RESET}\n" \
       "Config" "File" "Size" "Time(s)" "GiB/s" "Ratio%"
printf "${DIM}%-22s %-20s %10s %10s %8s %8s${RESET}\n" \
       "$cfg_rule" "$file_rule" "$size_rule" "$time_rule" "$thr_rule" "$ratio_rule"
awk -F '\t' 'NR > 1 && $3 == "compress" { print }' "$RESULTS_FILE" | while IFS=$'\t' read -r cfg file mode fbytes cbytes secs thr ratio; do
  fsize_h=$(numfmt --to=iec-i --suffix=B "$fbytes" 2>/dev/null || echo "${fbytes}B")
  printf "${YELLOW}%-22s${RESET} ${WHITE}%-20s${RESET} ${DIM}%10s${RESET} ${CYAN}%10.4f${RESET} ${GREEN}%8.3f${RESET} ${ORANGE}%7s%%${RESET}\n" \
         "$cfg" "$file" "$fsize_h" "$secs" "$thr" "$ratio"
done

echo ""

if $DO_DECOMPRESS; then
  echo "${BOLD}${GREEN} DECOMPRESSION${RESET}"
  printf "${DIM}${WHITE}%-22s %-20s %10s %10s %8s${RESET}\n" \
         "Config" "File" "Size" "Time(s)" "GiB/s"
  printf "${DIM}%-22s %-20s %10s %10s %8s${RESET}\n" \
         "$cfg_rule" "$file_rule" "$size_rule" "$time_rule" "$thr_rule"
  awk -F '\t' 'NR > 1 && $3 == "decompress" { print }' "$RESULTS_FILE" | while IFS=$'\t' read -r cfg file mode fbytes cbytes secs thr ratio; do
    fsize_h=$(numfmt --to=iec-i --suffix=B "$fbytes" 2>/dev/null || echo "${fbytes}B")
    printf "${YELLOW}%-22s${RESET} ${WHITE}%-20s${RESET} ${DIM}%10s${RESET} ${CYAN}%10.4f${RESET} ${GREEN}%8.3f${RESET}\n" \
           "$cfg" "$file" "$fsize_h" "$secs" "$thr"
  done
  echo ""
fi

#----------------------------------------------------------------------
# Find optimal configurations
#----------------------------------------------------------------------
print_separator
echo "${BOLD}${YELLOW} OPTIMAL CONFIGURATIONS${RESET}"
echo ""

echo "${BOLD}${CYAN}Best compression throughput:${RESET}"
for test_file in "${TEST_FILES[@]}"; do
  file_base=$(basename "$test_file")
  best=$(awk -F'\t' -v f="$file_base" 'NR>1 && $3=="compress" && $2==f' "$RESULTS_FILE" |
         sort -t$'\t' -k7 -rg | head -1)
  if [[ -n "$best" ]]; then
    cfg=$(echo "$best" | cut -f1)
    thr=$(echo "$best" | cut -f7)
    printf "  ${WHITE}%-20s${RESET} ${DIM}->${RESET} ${YELLOW}%-22s${RESET} ${GREEN}%.3f GiB/s${RESET}\n" "$file_base" "$cfg" "$thr"
  fi
done

echo ""
if $DO_DECOMPRESS; then
  echo "${BOLD}${GREEN}Best decompression throughput:${RESET}"
  for test_file in "${TEST_FILES[@]}"; do
    file_base=$(basename "$test_file")
    best=$(awk -F'\t' -v f="$file_base" 'NR>1 && $3=="decompress" && $2==f' "$RESULTS_FILE" |
           sort -t$'\t' -k7 -rg | head -1)
    if [[ -n "$best" ]]; then
      cfg=$(echo "$best" | cut -f1)
      thr=$(echo "$best" | cut -f7)
      printf "  ${WHITE}%-20s${RESET} ${DIM}->${RESET} ${YELLOW}%-22s${RESET} ${GREEN}%.3f GiB/s${RESET}\n" "$file_base" "$cfg" "$thr"
    fi
  done
  echo ""
fi

echo "${BOLD}${ORANGE}Best compression ratio:${RESET}"
for test_file in "${TEST_FILES[@]}"; do
  file_base=$(basename "$test_file")
  best=$(awk -F'\t' -v f="$file_base" 'NR>1 && $3=="compress" && $2==f' "$RESULTS_FILE" |
         sort -t$'\t' -k8 -g | head -1)
  if [[ -n "$best" ]]; then
    cfg=$(echo "$best" | cut -f1)
    ratio=$(echo "$best" | cut -f8)
    printf "  ${WHITE}%-20s${RESET} ${DIM}->${RESET} ${YELLOW}%-22s${RESET} ${ORANGE}%s%%${RESET}\n" "$file_base" "$cfg" "$ratio"
  fi
done

echo ""
#----------------------------------------------------------------------
# Write JSON output
#----------------------------------------------------------------------
echo "Writing JSON results to $OUTPUT..."
GZSTD_VER=$("$GZSTD" --version 2>&1 | head -1)
if $HAS_GPU_BUILD; then GPU_JSON="True"; else GPU_JSON="False"; fi
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
print_separator
echo "${BOLD}${WHITE} Benchmark complete!${RESET}"
print_separator
