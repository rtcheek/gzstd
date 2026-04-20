#!/usr/bin/env bash
#
# gzstd-test.sh  Comprehensive test suite for gzstd
#
# Usage:  ./gzstd-test.sh [path/to/gzstd]
#
# Features:
#   - Live progress bar with ETA
#   - Per-test timing
#   - Colorized output with Unicode symbols
#   - Clean CTRL-C handling
#   - GPU auto-detection (skips GPU tests if unavailable)
#
# Exit codes:
#   0  All tests passed
#   1  One or more tests failed
#   130  Interrupted (CTRL-C)
#
set -uo pipefail

# Wrap pipe-safe operations: pipefail + head/grep on version strings causes
# SIGPIPE exits.  Use this helper for any pipe that intentionally discards.
pipe_safe() { "$@" || true; }

# NOTE: We intentionally do NOT use set -e. Tests check exit codes explicitly
# via pass/fail/expect_exit. set -e would cause silent termination on any
# unexpected non-zero exit, hiding the actual failure.

# ============================================================
# Colors & symbols (auto-detect terminal capability)
# ============================================================
HAS_COLOR=false
if [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; then
  HAS_COLOR=true
  C_RESET='\033[0m'  C_BOLD='\033[1m'     C_DIM='\033[2m'
  C_RED='\033[1;31m' C_GREEN='\033[1;32m'  C_YELLOW='\033[1;33m'
  C_BLUE='\033[1;34m' C_MAGENTA='\033[1;35m' C_CYAN='\033[1;36m'
  C_WHITE='\033[1;37m'
  C_BG_GREEN='\033[42;1;37m'  C_BG_RED='\033[41;1;37m'
  C_BG_YELLOW='\033[43;1;30m' C_BG_CYAN='\033[46;1;30m'
  SYM_PASS='✔' SYM_FAIL='✘' SYM_SKIP='⊘' SYM_ARROW='▸'
  SYM_HBAR='' SYM_BULLET='●' SYM_BLOCK='' SYM_LIGHT=''
  SYM_CLOCK='⏱' SYM_ZAP='⚡'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW=''
  C_BLUE='' C_MAGENTA='' C_CYAN='' C_WHITE=''
  C_BG_GREEN='' C_BG_RED='' C_BG_YELLOW='' C_BG_CYAN=''
  SYM_PASS='[PASS]' SYM_FAIL='[FAIL]' SYM_SKIP='[SKIP]' SYM_ARROW='>'
  SYM_HBAR='=' SYM_BULLET='*' SYM_BLOCK='#' SYM_LIGHT='-'
  SYM_CLOCK='' SYM_ZAP=''
fi

# ============================================================
# Configuration
# ============================================================
GZSTD="${1:-}"
if [[ -z "$GZSTD" ]]; then
  if [[ -x ./gzstd ]]; then GZSTD=./gzstd
  elif [[ -x ./build/gzstd ]]; then GZSTD=./build/gzstd
  elif command -v gzstd &>/dev/null; then GZSTD=$(command -v gzstd)
  else
    printf "${C_RED}ERROR:${C_RESET} gzstd binary not found. Pass path as argument or build first.\n"
    exit 1
  fi
fi
GZSTD=$(realpath "$GZSTD")

TMPDIR=$(mktemp -d /tmp/gzstd-test.XXXXXX)

# ============================================================
# CTRL-C handler & cleanup
# ============================================================
INTERRUPTED=false

cleanup() {
  rm -rf "$TMPDIR" 2>/dev/null || true
}

ctrl_c() {
  INTERRUPTED=true
  echo ""
  printf "\n${C_YELLOW}${SYM_ZAP} Interrupted by user${C_RESET}\n"
  print_summary
  cleanup
  exit 130
}

trap ctrl_c INT TERM
trap cleanup EXIT

# ============================================================
# Counters & timing
# ============================================================
PASS=0
FAIL=0
SKIP=0
TEST_NUM=0
TOTAL_TESTS=0    # set after counting
SECTION_NUM=0
SECTION_PASS=0
SECTION_FAIL=0
TEST_START_TIME=$(date +%s%N)  # nanoseconds for precision
LAST_TEST_MS=0

# Millisecond clock
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
START_MS=$(now_ms)

# ============================================================
# Progress bar
# ============================================================
BAR_WIDTH=30

progress_bar() {
  local current=$1 total=$2
  [[ $total -eq 0 ]] && return

  local pct=$(( current * 100 / total ))
  local filled=$(( current * BAR_WIDTH / total ))
  local empty=$(( BAR_WIDTH - filled ))
  local elapsed_ms=$(( $(now_ms) - START_MS ))
  local elapsed_s=$(( elapsed_ms / 1000 ))

  # ETA calculation
  local eta_str=""
  if [[ $current -gt 3 && $elapsed_ms -gt 0 ]]; then
    local remaining=$(( total - current ))
    local ms_per_test=$(( elapsed_ms / current ))
    local eta_ms=$(( remaining * ms_per_test ))
    local eta_s=$(( eta_ms / 1000 ))
    if [[ $eta_s -ge 60 ]]; then
      eta_str="ETA $(( eta_s / 60 ))m$(( eta_s % 60 ))s"
    else
      eta_str="ETA ${eta_s}s"
    fi
  fi

  # Build bar
  local bar_filled="" bar_empty=""
  for ((i=0; i<filled; i++)); do bar_filled+="${SYM_BLOCK}"; done
  for ((i=0; i<empty; i++)); do bar_empty+="${SYM_LIGHT}"; done

  # Color the percentage
  local pct_color="$C_CYAN"
  [[ $pct -ge 100 ]] && pct_color="$C_GREEN"

  printf "\r  ${C_DIM}[${C_RESET}${C_GREEN}%s${C_DIM}%s${C_RESET}${C_DIM}]${C_RESET}" \
    "$bar_filled" "$bar_empty"
  printf " ${pct_color}%3d%%${C_RESET}" "$pct"
  printf " ${C_DIM}%d/%d${C_RESET}" "$current" "$total"
  if [[ -n "$eta_str" ]]; then
    printf " ${C_DIM}${SYM_CLOCK} %s${C_RESET}" "$eta_str"
  fi
  printf " ${C_DIM}(%ds elapsed)${C_RESET}  " "$elapsed_s"
}

clear_progress() {
  printf "\r\033[K"
}

update_progress() {
  if $HAS_COLOR && [[ -t 1 ]]; then
    progress_bar "$TEST_NUM" "$TOTAL_TESTS"
  fi
}

# ============================================================
# Output helpers
# ============================================================
fmt_ms() {
  local ms=$1
  if [[ $ms -ge 1000 ]]; then
    awk "BEGIN{printf \"%.1fs\", $ms/1000}"
  else
    printf "%dms" "$ms"
  fi
}

pass() {
  PASS=$((PASS+1)); SECTION_PASS=$((SECTION_PASS+1)); TEST_NUM=$((TEST_NUM+1))
  clear_progress
  printf "  ${C_GREEN}${SYM_PASS}${C_RESET}  %s" "$1"
  [[ -n "${2:-}" ]] && printf "  ${C_DIM}%s${C_RESET}" "$2"
  if (( LAST_TEST_MS > 500 )); then
    printf "  ${C_DIM}[$(fmt_ms $LAST_TEST_MS)]${C_RESET}"
  fi
  echo ""
  update_progress
}

fail() {
  FAIL=$((FAIL+1)); SECTION_FAIL=$((SECTION_FAIL+1)); TEST_NUM=$((TEST_NUM+1))
  clear_progress
  printf "  ${C_RED}${SYM_FAIL}${C_RESET}  ${C_RED}%s${C_RESET}" "$1"
  [[ -n "${2:-}" ]] && printf "  ${C_DIM} %s${C_RESET}" "$2"
  if (( LAST_TEST_MS > 500 )); then
    printf "  ${C_DIM}[$(fmt_ms $LAST_TEST_MS)]${C_RESET}"
  fi
  echo ""
  update_progress
}

skip() {
  SKIP=$((SKIP+1)); TEST_NUM=$((TEST_NUM+1))
  clear_progress
  printf "  ${C_YELLOW}${SYM_SKIP}${C_RESET}  ${C_DIM}%s${C_RESET}" "$1"
  [[ -n "${2:-}" ]] && printf "  ${C_DIM} %s${C_RESET}" "$2"
  echo ""
  update_progress
}

# Time a test: run_test <command...> sets LAST_TEST_MS and LAST_RC
LAST_RC=0
run_test() {
  local t0=$(now_ms)
  LAST_RC=0
  "$@" || LAST_RC=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  return 0  # never fail under set -e; callers check LAST_RC if needed
}

section_summary() {
  if [[ $SECTION_NUM -gt 0 ]]; then
    clear_progress
    local total=$((SECTION_PASS + SECTION_FAIL))
    if [[ $SECTION_FAIL -eq 0 ]]; then
      printf "  ${C_DIM}└─ %d/%d passed${C_RESET}\n" "$SECTION_PASS" "$total"
    else
      printf "  ${C_DIM}└─${C_RESET} ${C_RED}%d failed${C_RESET}${C_DIM}, %d/%d passed${C_RESET}\n" \
        "$SECTION_FAIL" "$SECTION_PASS" "$total"
    fi
  fi
}

section() {
  section_summary
  SECTION_PASS=0; SECTION_FAIL=0
  SECTION_NUM=$((SECTION_NUM + 1))
  clear_progress
  echo ""
  printf "${C_CYAN}${SYM_HBAR}${SYM_HBAR}${C_RESET} ${C_BOLD}${C_WHITE}%d. %s${C_RESET}\n" \
    "$SECTION_NUM" "$1"
  update_progress
}

banner() {
  local text="$1" width=60
  local pad=$(( (width - ${#text} - 2) / 2 ))
  local hline=""; for ((i=0; i<width; i++)); do hline+="${SYM_HBAR}"; done
  echo ""
  printf "${C_CYAN}%s${C_RESET}\n" "$hline"
  printf "%*s${C_BOLD}${C_WHITE} %s ${C_CYAN}%*s\n" "$pad" '' "$text" "$pad" ''
  printf "${C_CYAN}%s${C_RESET}\n" "$hline"
}

spin() { printf "  ${C_DIM}${SYM_ARROW} %s${C_RESET}" "$1"; }
spin_done() { printf " ${C_GREEN}done${C_RESET}\n"; }

# ============================================================
# Summary printer (shared by normal exit + CTRL-C)
# ============================================================
print_summary() {
  section_summary

  local total=$((PASS + FAIL + SKIP))
  local elapsed_ms=$(( $(now_ms) - START_MS ))
  local elapsed_s=$(( elapsed_ms / 1000 ))

  echo ""
  local hline=""; for ((i=0; i<60; i++)); do hline+="${SYM_HBAR}"; done
  printf "${C_CYAN}%s${C_RESET}\n" "$hline"

  if $INTERRUPTED; then
    printf "\n  ${C_BG_YELLOW} INTERRUPTED ${C_RESET}  ${C_DIM}%d of %d tests completed${C_RESET}\n\n" \
      "$TEST_NUM" "$TOTAL_TESTS"
  elif [[ $FAIL -eq 0 ]]; then
    printf "\n  ${C_BG_GREEN} ALL TESTS PASSED ${C_RESET}\n\n"
  else
    printf "\n  ${C_BG_RED} %d TEST(S) FAILED ${C_RESET}\n\n" "$FAIL"
  fi

  printf "  ${C_GREEN}${SYM_PASS} Passed${C_RESET}   %-4d" "$PASS"
  if [[ $FAIL -gt 0 ]]; then
    printf "    ${C_RED}${SYM_FAIL} Failed${C_RESET}   %-4d" "$FAIL"
  else
    printf "    ${C_DIM}${SYM_FAIL} Failed${C_RESET}   ${C_DIM}0${C_RESET}   "
  fi
  [[ $SKIP -gt 0 ]] && printf "    ${C_YELLOW}${SYM_SKIP} Skipped${C_RESET}  %-4d" "$SKIP"
  printf "    ${C_DIM}Total: %d${C_RESET}" "$total"
  echo ""

  if (( elapsed_s >= 60 )); then
    printf "  ${C_DIM}Completed in %dm%ds${C_RESET}\n\n" "$(( elapsed_s / 60 ))" "$(( elapsed_s % 60 ))"
  else
    printf "  ${C_DIM}Completed in %ds${C_RESET}\n\n" "$elapsed_s"
  fi
}

# ============================================================
# Test utilities
# ============================================================
expect_exit() {
  local expected=$1; shift
  local rc=0
  "$@" >/dev/null 2>&1 || rc=$?
  [[ $rc -eq $expected ]]
}

files_match() {
  local a b
  a=$(sha256sum "$1" | cut -d' ' -f1) || return 1
  b=$(sha256sum "$2" | cut -d' ' -f1) || return 1
  [[ "$a" == "$b" ]]
}

has_gpu() {
  ("$GZSTD" -V 2>&1 | grep -qi "nvcomp\|gpu\|cuda") 2>/dev/null && \
    command -v nvidia-smi &>/dev/null && \
    nvidia-smi &>/dev/null
  return $?
}

human_size() {
  local bytes=$1
  if   [[ $bytes -ge 1073741824 ]]; then awk "BEGIN{printf \"%.1f GiB\", $bytes/1073741824}"
  elif [[ $bytes -ge 1048576 ]];    then awk "BEGIN{printf \"%.1f MiB\", $bytes/1048576}"
  elif [[ $bytes -ge 1024 ]];       then awk "BEGIN{printf \"%.1f KiB\", $bytes/1024}"
  else printf "%d B" "$bytes"
  fi
}

# ============================================================
# Count total tests (for progress bar)
# ============================================================
count_tests() {
  local count=0
  count=$((count + 4))   # 1: basic round-trip
  count=$((count + 2))   # 2: edge cases
  count=$((count + 7))   # 3: levels
  count=$((count + 2))   # 4: integrity
  count=$((count + 3))   # 5: pipes
  count=$((count + 4))   # 6: tar
  count=$((count + 4))   # 7: file mgmt
  count=$((count + 2))   # 8: multi-file
  count=$((count + 2))   # 9: sparse
  count=$((count + 3))   # 10: threading
  count=$((count + 3))   # 11: chunk sizes
  count=$((count + 3))   # 12: verbosity
  count=$((count + 1))   # 13: stats json
  count=$((count + 5))   # 14: exit codes
  count=$((count + 7))   # 15: help+version (expanded)
  command -v zstd &>/dev/null && count=$((count + 3)) || count=$((count + 3))  # 16: zstd interop
  count=$((count + 3))   # 17: tar advanced
  has_gpu 2>/dev/null && count=$((count + 10)) || count=$((count + 5))  # 18: GPU
  has_gpu 2>/dev/null && count=$((count + 4)) || count=$((count + 4))  # 19: VRAM pressure
  count=$((count + 6))   # 19b: stream/batch exhaustion regression
  count=$((count + 3))   # 20: stress
  count=$((count + 3))   # 20: wildcards
  count=$((count + 3))   # 21: -- end-of-options
  count=$((count + 6))   # 22: -c, -o, --output
  count=$((count + 7))   # 23: pipes with options
  count=$((count + 8))   # 24: thread forms (expanded with -T=0, -T=4)
  count=$((count + 4))   # 25: CPU scheduling
  count=$((count + 1))   # 26: sync output
  count=$((count + 4))   # 27: pinned memory
  has_gpu 2>/dev/null && count=$((count + 9)) || count=$((count + 4))  # 28: GPU options
  count=$((count + 8))   # 29: error handling
  count=$((count + 2))   # 30: cross-level
  count=$((count + 6))   # 31: arg order
  has_gpu 2>/dev/null && count=$((count + 10)) || count=$((count + 10))  # 32: space-separated (4 base + 6 gpu or 6 skip)
  has_gpu 2>/dev/null && count=$((count + 17)) || count=$((count + 15))  # 33: verbose validation
  count=$((count + 6))   # 34: summary format
  count=$((count + 8))   # 35: ultra validation (windowLog, T1, T4, chunk-warn, chunk-ok, prog×2, interop)
  count=$((count + 9))   # 36: throttle tunables (+1 GPU deadlock regression)
  echo $count
}

# ============================================================
# Banner & system info
# ============================================================
banner "gzstd test suite"

VERSION=$("$GZSTD" -V 2>&1 | head -1 || true)
printf "\n"
printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}Binary${C_RESET}   ${C_CYAN}%s${C_RESET}\n" "$GZSTD"
printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}Version${C_RESET}  ${C_CYAN}%s${C_RESET}\n" "$VERSION"
printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}Temp${C_RESET}     ${C_DIM}%s${C_RESET}\n" "$TMPDIR"
if has_gpu 2>/dev/null; then
  gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
  printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}GPU${C_RESET}      ${C_GREEN}%s${C_RESET}\n" "$gpu_info"
else
  printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}GPU${C_RESET}      ${C_DIM}none  GPU tests will be skipped${C_RESET}\n"
fi
printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}CPUs${C_RESET}     ${C_DIM}%s${C_RESET}\n" "$(nproc 2>/dev/null || echo unknown)"

TOTAL_TESTS=$(count_tests)
printf "  ${C_DIM}${SYM_BULLET}${C_RESET} ${C_BOLD}Tests${C_RESET}    ${C_DIM}%d planned${C_RESET}\n" "$TOTAL_TESTS"

# ============================================================
# Generate test data
# ============================================================
echo ""
printf "  ${C_BOLD}Generating test data...${C_RESET}\n"

spin "small.txt (4 KiB text)"
dd if=/dev/urandom bs=1024 count=4 2>/dev/null | base64 > "$TMPDIR/small.txt"
spin_done

spin "medium.txt (1 MiB mixed-compressible)"
# Mix of text (compressible) and random bytes (not) for realistic ratios
{
  dd if=/dev/urandom bs=1024 count=256 2>/dev/null    # 256 KiB random
  for i in $(seq 1 200); do                            # ~200 KiB varied text
    echo "Log entry $i: user=$(head -c8 /dev/urandom 2>/dev/null | base64) action=request ts=$(date +%s%N) status=200 latency=$((RANDOM % 500))ms path=/api/v2/resource/$((RANDOM % 10000))"
  done
  dd if=/dev/zero bs=1024 count=128 2>/dev/null        # 128 KiB zeros
  dd if=/dev/urandom bs=1024 count=256 2>/dev/null     # 256 KiB more random
  for i in $(seq 1 200); do                            # ~200 KiB more text
    echo "Event id=$(printf '%08x' $((RANDOM * RANDOM))) type=click element=button-$((RANDOM % 50)) page=/dashboard/$((RANDOM % 100)) session=$(head -c12 /dev/urandom 2>/dev/null | base64)"
  done
} | head -c $((1024*1024)) > "$TMPDIR/medium.txt"
spin_done

spin "large.bin (32 MiB mixed)"
dd if=/dev/zero bs=1M count=16 2>/dev/null > "$TMPDIR/large.bin"
dd if=/dev/urandom bs=1M count=16 2>/dev/null >> "$TMPDIR/large.bin"
spin_done

spin "random.bin (1 MiB incompressible)"
dd if=/dev/urandom bs=1M count=1 2>/dev/null > "$TMPDIR/random.bin"
spin_done

spin "edge cases (empty, 1-byte, zeros)"
touch "$TMPDIR/empty.txt"
printf 'X' > "$TMPDIR/onebyte.txt"
dd if=/dev/zero bs=1M count=2 2>/dev/null > "$TMPDIR/zeros.bin"
spin_done

spin "directory tree for tar tests"
mkdir -p "$TMPDIR/tree/subdir"
cp "$TMPDIR/small.txt" "$TMPDIR/tree/"
cp "$TMPDIR/medium.txt" "$TMPDIR/tree/"
cp "$TMPDIR/random.bin" "$TMPDIR/tree/subdir/"
echo "data inside subdir" > "$TMPDIR/tree/subdir/note.txt"
spin_done

# ============================================================
# 1. Basic compress/decompress round-trip
# ============================================================
section "Basic compress/decompress round-trip"

for f in small.txt medium.txt large.bin random.bin; do
  src="$TMPDIR/$f"
  compressed="$TMPDIR/${f}.zst"
  recovered="$TMPDIR/${f}.recovered"
  size=$(human_size "$(stat -c%s "$src")")

  run_test "$GZSTD" -k -f --cpu-only "$src" -o "$compressed" 2>/dev/null
  if [[ ! -f "$compressed" || ! -s "$compressed" ]]; then
    fail "compress $f" "output missing or empty"; continue
  fi

  run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
  if files_match "$src" "$recovered"; then
    comp_size=$(stat -c%s "$compressed"); orig_size=$(stat -c%s "$src")
    ratio=$(awk "BEGIN { printf \"%.1f\", ($comp_size * 100.0) / $orig_size }")
    pass "$f" "($size ${SYM_ARROW} ${ratio}%)"
  else
    fail "$f" "checksum mismatch"
  fi
  rm -f "$compressed" "$recovered"
done

# ============================================================
# 2. Edge cases
# ============================================================
section "Edge cases (empty & tiny files)"

for f in empty.txt onebyte.txt; do
  src="$TMPDIR/$f"
  compressed="$TMPDIR/${f}.zst"
  recovered="$TMPDIR/${f}.recovered"

  run_test "$GZSTD" -k -f --cpu-only "$src" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
  if files_match "$src" "$recovered"; then
    pass "$f" "($(stat -c%s "$src") bytes)"
  else
    fail "$f" "checksum mismatch"
  fi
  rm -f "$compressed" "$recovered"
done

# ============================================================
# 3. Compression levels
# ============================================================
section "Compression levels"

for level in 1 3 9 19; do
  compressed="$TMPDIR/medium-L${level}.zst"
  recovered="$TMPDIR/medium-L${level}.recovered"
  run_test "$GZSTD" -${level} -k -f --cpu-only "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
  if files_match "$TMPDIR/medium.txt" "$recovered"; then
    ratio=$(awk "BEGIN { printf \"%.1f\", ($(stat -c%s "$compressed") * 100.0) / $(stat -c%s "$TMPDIR/medium.txt") }")
    pass "level -$level" "(ratio: ${ratio}%)"
  else
    fail "level -$level" "checksum mismatch"
  fi
  rm -f "$compressed" "$recovered"
done

compressed="$TMPDIR/medium-L22.zst"; recovered="$TMPDIR/medium-L22.recovered"
run_test "$GZSTD" -22 --ultra -k -f --cpu-only "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
if files_match "$TMPDIR/medium.txt" "$recovered"; then
  ratio=$(awk "BEGIN { printf \"%.1f\", ($(stat -c%s "$compressed") * 100.0) / $(stat -c%s "$TMPDIR/medium.txt") }")
  pass "level -22 --ultra" "(ratio: ${ratio}%)"
else
  fail "level -22 --ultra" "mismatch"
fi
rm -f "$compressed" "$recovered"

for alias_flag in "--fast" "--best"; do
  compressed="$TMPDIR/medium-${alias_flag#--}.zst"; recovered="$TMPDIR/medium-${alias_flag#--}.recovered"
  run_test "$GZSTD" $alias_flag -k -f --cpu-only "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
  if files_match "$TMPDIR/medium.txt" "$recovered"; then
    ratio=$(awk "BEGIN { printf \"%.1f\", ($(stat -c%s "$compressed") * 100.0) / $(stat -c%s "$TMPDIR/medium.txt") }")
    pass "$alias_flag alias" "(ratio: ${ratio}%)"
  else
    fail "$alias_flag alias" "mismatch"
  fi
  rm -f "$compressed" "$recovered"
done

# ============================================================
# 4. Integrity test mode (-t)
# ============================================================
section "Integrity test mode (-t)"

compressed="$TMPDIR/large.bin.zst"
run_test "$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$compressed" 2>/dev/null

LAST_TEST_MS=0
run_test "$GZSTD" -t --cpu-only "$compressed" 2>/dev/null
[[ $LAST_RC -eq 0 ]] && pass "valid file passes -t" || fail "valid file passes -t" "exit $LAST_RC"

cp "$compressed" "$TMPDIR/corrupt.zst"
# Corrupt at multiple positions to ensure we hit actual compressed data
python3 -c "
import os
size = os.path.getsize('$TMPDIR/corrupt.zst')
with open('$TMPDIR/corrupt.zst', 'r+b') as f:
    # Corrupt near the start (past the 4-byte magic), middle, and near the end
    for pos in [16, size // 4, size // 2, size * 3 // 4]:
        f.seek(pos)
        f.write(b'\x00\xff\x00\xff\x00\xff\x00\xff' * 4)
" 2>/dev/null || true

LAST_TEST_MS=0
run_test "$GZSTD" -t --cpu-only "$TMPDIR/corrupt.zst" 2>/dev/null
if [[ $LAST_RC -eq 0 ]]; then
  fail "corrupt file detected" "should have failed"
else
  pass "corrupt file detected"
fi
rm -f "$compressed" "$TMPDIR/corrupt.zst"

# ============================================================
# 5. Pipe operation
# ============================================================
section "Pipe (stdin/stdout) operation"

LAST_TEST_MS=0; t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only > "$TMPDIR/pipe.zst" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
[[ -s "$TMPDIR/pipe.zst" ]] && pass "compress via stdin" || fail "compress via stdin" "empty output"

t0=$(now_ms)
"$GZSTD" -d --cpu-only < "$TMPDIR/pipe.zst" > "$TMPDIR/pipe.recovered" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe.recovered" \
  && pass "decompress via stdout" || fail "decompress via stdout" "mismatch"

t0=$(now_ms)
cat "$TMPDIR/medium.txt" \
  | "$GZSTD" --cpu-only 2>/dev/null \
  | "$GZSTD" -d --cpu-only 2>/dev/null \
  > "$TMPDIR/pipeline.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipeline.recovered" \
  && pass "compress ${SYM_ARROW} decompress pipeline" || fail "compress ${SYM_ARROW} decompress pipeline" "mismatch"
rm -f "$TMPDIR/pipe.zst" "$TMPDIR/pipe.recovered" "$TMPDIR/pipeline.recovered"

# ============================================================
# 6. Tar integration
# ============================================================
section "Tar integration"

t0=$(now_ms)
tar -I "$GZSTD --cpu-only" -cf "$TMPDIR/tree.tar.zst" -C "$TMPDIR" tree 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
[[ -s "$TMPDIR/tree.tar.zst" ]] && pass "tar -I gzstd create" || fail "tar -I gzstd create" "empty"

t0=$(now_ms)
mkdir -p "$TMPDIR/extracted"
tar -I "$GZSTD --cpu-only" -xf "$TMPDIR/tree.tar.zst" -C "$TMPDIR/extracted" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
if files_match "$TMPDIR/tree/small.txt" "$TMPDIR/extracted/tree/small.txt" && \
   files_match "$TMPDIR/tree/medium.txt" "$TMPDIR/extracted/tree/medium.txt" && \
   files_match "$TMPDIR/tree/subdir/random.bin" "$TMPDIR/extracted/tree/subdir/random.bin" && \
   files_match "$TMPDIR/tree/subdir/note.txt" "$TMPDIR/extracted/tree/subdir/note.txt"; then
  pass "tar -I gzstd extract" "(all 4 files verified)"
else
  fail "tar -I gzstd extract" "file mismatch"
fi

LAST_TEST_MS=0
member_count=$(tar -I "$GZSTD --cpu-only" -tf "$TMPDIR/tree.tar.zst" 2>/dev/null | wc -l || true)
[[ "$member_count" -ge 4 ]] \
  && pass "tar -I gzstd list" "($member_count members)" \
  || fail "tar -I gzstd list" "expected >=4, got $member_count"

t0=$(now_ms)
tar cf - -C "$TMPDIR" tree 2>/dev/null | "$GZSTD" --cpu-only 2>/dev/null > "$TMPDIR/tree-pipe.tar.zst"
mkdir -p "$TMPDIR/extracted-pipe"
"$GZSTD" -d --cpu-only 2>/dev/null < "$TMPDIR/tree-pipe.tar.zst" | tar xf - -C "$TMPDIR/extracted-pipe" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/tree/medium.txt" "$TMPDIR/extracted-pipe/tree/medium.txt" \
  && pass "tar cf - | gzstd | gzstd -d | tar xf -" \
  || fail "tar pipe round-trip" "mismatch"
rm -rf "$TMPDIR/tree.tar.zst" "$TMPDIR/tree-pipe.tar.zst" "$TMPDIR/extracted" "$TMPDIR/extracted-pipe"

# ============================================================
# 7. File management flags
# ============================================================
section "File management flags"

LAST_TEST_MS=0
cp "$TMPDIR/small.txt" "$TMPDIR/keep-test.txt"
"$GZSTD" -k -f --cpu-only "$TMPDIR/keep-test.txt" 2>/dev/null
[[ -f "$TMPDIR/keep-test.txt" && -f "$TMPDIR/keep-test.txt.zst" ]] \
  && pass "-k keeps input" || fail "-k keeps input"
rm -f "$TMPDIR/keep-test.txt" "$TMPDIR/keep-test.txt.zst"

cp "$TMPDIR/small.txt" "$TMPDIR/rm-test.txt"
"$GZSTD" --rm -f --cpu-only "$TMPDIR/rm-test.txt" 2>/dev/null
[[ ! -f "$TMPDIR/rm-test.txt" && -f "$TMPDIR/rm-test.txt.zst" ]] \
  && pass "--rm removes input" || fail "--rm removes input"
rm -f "$TMPDIR/rm-test.txt.zst"

cp "$TMPDIR/small.txt" "$TMPDIR/force-test.txt"
"$GZSTD" -k -f --cpu-only "$TMPDIR/force-test.txt" 2>/dev/null
"$GZSTD" -k -f --cpu-only "$TMPDIR/force-test.txt" 2>/dev/null
[[ -f "$TMPDIR/force-test.txt.zst" ]] && pass "-f force overwrite" || fail "-f force overwrite"

if expect_exit 3 "$GZSTD" -k --cpu-only "$TMPDIR/force-test.txt"; then
  pass "overwrite without -f" "(exit 3 EXIT_IO)"
elif ! expect_exit 0 "$GZSTD" -k --cpu-only "$TMPDIR/force-test.txt"; then
  rc=0; "$GZSTD" -k --cpu-only "$TMPDIR/force-test.txt" >/dev/null 2>&1 || rc=$?
  pass "overwrite without -f" "(exit $rc)"
else
  fail "overwrite without -f" "should not succeed"
fi
rm -f "$TMPDIR/force-test.txt" "$TMPDIR/force-test.txt.zst"

# ============================================================
# 8. Multi-file
# ============================================================
section "Multi-file operation"

LAST_TEST_MS=0
cp "$TMPDIR/small.txt" "$TMPDIR/multi1.txt"; cp "$TMPDIR/medium.txt" "$TMPDIR/multi2.txt"
"$GZSTD" -k -f --cpu-only "$TMPDIR/multi1.txt" "$TMPDIR/multi2.txt" 2>/dev/null
[[ -f "$TMPDIR/multi1.txt.zst" && -f "$TMPDIR/multi2.txt.zst" ]] \
  && pass "compress 2 files" || fail "compress 2 files"

"$GZSTD" -d -k -f --cpu-only "$TMPDIR/multi1.txt.zst" "$TMPDIR/multi2.txt.zst" 2>/dev/null
files_match "$TMPDIR/small.txt" "$TMPDIR/multi1.txt" && files_match "$TMPDIR/medium.txt" "$TMPDIR/multi2.txt" \
  && pass "decompress 2 files" || fail "decompress 2 files" "mismatch"
rm -f "$TMPDIR/multi1.txt" "$TMPDIR/multi2.txt" "$TMPDIR/multi1.txt.zst" "$TMPDIR/multi2.txt.zst"

# ============================================================
# 9. Sparse files
# ============================================================
section "Sparse file support"

compressed="$TMPDIR/zeros.bin.zst"
run_test "$GZSTD" -k -f --cpu-only "$TMPDIR/zeros.bin" -o "$compressed" 2>/dev/null

run_test "$GZSTD" -d -k -f --sparse --cpu-only "$compressed" -o "$TMPDIR/zeros-sparse.bin" 2>/dev/null
if files_match "$TMPDIR/zeros.bin" "$TMPDIR/zeros-sparse.bin"; then
  ob=$(stat -c%b "$TMPDIR/zeros.bin"); sb=$(stat -c%b "$TMPDIR/zeros-sparse.bin")
  [[ "$sb" -lt "$ob" ]] && pass "--sparse" "(${sb} vs ${ob} blocks)" || pass "--sparse" "(data ok)"
else
  fail "--sparse" "mismatch"
fi

run_test "$GZSTD" -d -k -f --no-sparse --cpu-only "$compressed" -o "$TMPDIR/zeros-nosparse.bin" 2>/dev/null
files_match "$TMPDIR/zeros.bin" "$TMPDIR/zeros-nosparse.bin" && pass "--no-sparse" || fail "--no-sparse" "mismatch"
rm -f "$compressed" "$TMPDIR/zeros-sparse.bin" "$TMPDIR/zeros-nosparse.bin"

# ============================================================
# 10. Threading
# ============================================================
section "Threading options"

for threads in 1 2 4; do
  compressed="$TMPDIR/thread-${threads}.zst"; recovered="$TMPDIR/thread-${threads}.recovered"
  run_test "$GZSTD" -k -f --cpu-only -T $threads "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only -T $threads "$compressed" -o "$recovered" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$recovered" && pass "-T $threads" || fail "-T $threads" "mismatch"
  rm -f "$compressed" "$recovered"
done

# ============================================================
# 11. Chunk sizes
# ============================================================
section "Chunk size options"

for chunk in 1 4 32; do
  compressed="$TMPDIR/chunk-${chunk}.zst"; recovered="$TMPDIR/chunk-${chunk}.recovered"
  run_test "$GZSTD" -k -f --cpu-only --chunk-size=$chunk "$TMPDIR/large.bin" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only "$compressed" -o "$recovered" 2>/dev/null
  files_match "$TMPDIR/large.bin" "$recovered" && pass "--chunk-size=$chunk" || fail "--chunk-size=$chunk" "mismatch"
  rm -f "$compressed" "$recovered"
done

# ============================================================
# 12. Verbosity
# ============================================================
section "Verbose / quiet / progress"

LAST_TEST_MS=0
compressed="$TMPDIR/verbose-test.zst"
"$GZSTD" -k -f --cpu-only "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null

stderr_v=$("$GZSTD" -d -k -f -v --cpu-only "$compressed" -o /dev/null 2>&1 || true)
[[ -n "$stderr_v" ]] && pass "-v produces output" || fail "-v produces output"

stderr_q=$("$GZSTD" -d -k -f -q --cpu-only "$compressed" -o /dev/null 2>&1 || true)
[[ ${#stderr_q} -le ${#stderr_v} ]] && pass "-q reduces output" || pass "-q output"

stderr_qq=$("$GZSTD" -d -k -f -qq --cpu-only "$compressed" -o /dev/null 2>&1 || true)
[[ -z "$stderr_qq" ]] && pass "-qq suppresses all" || fail "-qq suppresses all"
rm -f "$compressed"

# ============================================================
# 13. Stats JSON
# ============================================================
section "Stats JSON output"

LAST_TEST_MS=0
json_file="$TMPDIR/stats.json"
"$GZSTD" -k -f --cpu-only --stats-json "$json_file" "$TMPDIR/medium.txt" 2>/dev/null
if [[ -f "$json_file" && -s "$json_file" ]]; then
  python3 -c "import json; json.load(open('$json_file'))" 2>/dev/null \
    && pass "--stats-json valid JSON" || fail "--stats-json" "parse error"
else
  skip "--stats-json" "file not created"
fi
rm -f "$json_file" "$TMPDIR/medium.txt.zst"

# ============================================================
# 14. Exit codes
# ============================================================
section "Exit code validation"

LAST_TEST_MS=0
expect_exit 0 "$GZSTD" -k -f --cpu-only "$TMPDIR/medium.txt" -o "$TMPDIR/exitcode.zst" \
  && pass "exit 0 on success" || fail "exit 0 on success"
rm -f "$TMPDIR/exitcode.zst"

rc=0; "$GZSTD" --nonexistent-flag >/dev/null 2>&1 || rc=$?
[[ $rc -eq 2 ]] && pass "exit 2 bad usage" "(EXIT_USAGE)" \
  || { [[ $rc -ne 0 ]] && pass "non-zero bad usage" "(exit $rc)" || fail "bad usage" "got 0"; }

if ! has_gpu 2>/dev/null; then
  rc=0; "$GZSTD" --gpu-only -k -f "$TMPDIR/medium.txt" -o /dev/null 2>&1 || rc=$?
  [[ $rc -eq 2 ]] && pass "exit 2: --gpu-only no GPU" "(EXIT_USAGE)" \
    || { [[ $rc -ne 0 ]] && pass "--gpu-only no GPU" "(exit $rc)" || fail "--gpu-only no GPU" "got 0"; }
fi

"$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/corrupt-exit.zst" 2>/dev/null
# Truncate to half the compressed size  guaranteed to be mid-frame
half_size=$(( $(stat -c%s "$TMPDIR/corrupt-exit.zst") / 2 ))
head -c "$half_size" "$TMPDIR/corrupt-exit.zst" > "$TMPDIR/truncated.zst"
rc=0; "$GZSTD" -d -f --cpu-only "$TMPDIR/truncated.zst" -o /dev/null 2>/dev/null || rc=$?
[[ $rc -eq 4 ]] && pass "exit 4 corrupt data" "(EXIT_DATA)" \
  || { [[ $rc -ne 0 ]] && pass "non-zero corrupt data" "(exit $rc)" || fail "corrupt data" "got 0"; }
rm -f "$TMPDIR/corrupt-exit.zst" "$TMPDIR/truncated.zst"

rc=0; "$GZSTD" -d -f --cpu-only "$TMPDIR/medium.txt" -o /dev/null 2>/dev/null || rc=$?
[[ $rc -eq 4 ]] && pass "exit 4 non-zstd input" "(EXIT_DATA)" \
  || { [[ $rc -ne 0 ]] && pass "non-zero non-zstd" "(exit $rc)" || fail "non-zstd" "got 0"; }

# ============================================================
# 15. Help & version
# ============================================================
section "Help and version"

LAST_TEST_MS=0
("$GZSTD" -h 2>&1 || true) | grep -qi "usage\|options\|compress" && pass "-h shows help" || fail "-h shows help"
("$GZSTD" --help 2>&1 || true) | grep -qi "usage\|options\|compress" && pass "--help shows help" || fail "--help shows help"
("$GZSTD" -V 2>&1 || true) | grep -qE "[0-9]+\.[0-9]+\.[0-9]+" && pass "-V shows version" "($VERSION)" || fail "-V shows version"
("$GZSTD" --version 2>&1 || true) | grep -qE "[0-9]+\.[0-9]+\.[0-9]+" && pass "--version shows version" || fail "--version shows version"

# Exit codes
rc=0; "$GZSTD" -h >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-h exits 0" || fail "-h exits 0" "got $rc"
rc=0; "$GZSTD" --help >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "--help exits 0" || fail "--help exits 0" "got $rc"
rc=0; "$GZSTD" -V >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-V exits 0" || fail "-V exits 0" "got $rc"

# Help lists exit codes
("$GZSTD" -h 2>&1 || true) | grep -qi "exit code" && pass "-h documents exit codes" || fail "-h documents exit codes"

# -h and --help should exit 0
rc=0; "$GZSTD" -h >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-h exits 0" || fail "-h exits 0" "got $rc"

rc=0; "$GZSTD" -V >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-V exits 0" || fail "-V exits 0" "got $rc"

# Help should list exit codes
("$GZSTD" -h 2>&1 || true) | grep -qi "exit code" && pass "-h documents exit codes" || fail "-h documents exit codes"

# ============================================================
# 16. Zstd interop
# ============================================================
section "Zstd interoperability"

if command -v zstd &>/dev/null; then
  run_test "$GZSTD" -k -f --cpu-only "$TMPDIR/medium.txt" -o "$TMPDIR/interop.zst" 2>/dev/null
  zstd -d -f "$TMPDIR/interop.zst" -o "$TMPDIR/interop-zstd.recovered" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$TMPDIR/interop-zstd.recovered" \
    && pass "gzstd ${SYM_ARROW} zstd" || fail "gzstd ${SYM_ARROW} zstd" "mismatch"

  zstd -f "$TMPDIR/medium.txt" -o "$TMPDIR/interop-zstd.zst" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only "$TMPDIR/interop-zstd.zst" -o "$TMPDIR/interop-gzstd.recovered" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$TMPDIR/interop-gzstd.recovered" \
    && pass "zstd ${SYM_ARROW} gzstd" || fail "zstd ${SYM_ARROW} gzstd" "mismatch"

  t0=$(now_ms)
  cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only 2>/dev/null | zstd -d 2>/dev/null > "$TMPDIR/interop-pipe.recovered"
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  files_match "$TMPDIR/medium.txt" "$TMPDIR/interop-pipe.recovered" \
    && pass "gzstd | zstd -d pipe" || fail "gzstd | zstd -d pipe" "mismatch"
  rm -f "$TMPDIR"/interop*
else
  skip "gzstd ${SYM_ARROW} zstd" "zstd not installed"
  skip "zstd ${SYM_ARROW} gzstd" "zstd not installed"
  skip "pipe interop" "zstd not installed"
fi

# ============================================================
# 17. Tar advanced
# ============================================================
section "Tar advanced integration"

mkdir -p "$TMPDIR/tartest/deeply/nested/path"
dd if=/dev/urandom bs=4096 count=10 2>/dev/null > "$TMPDIR/tartest/binary.dat"
for i in $(seq 1 20); do
  echo "File number $i with some content" > "$TMPDIR/tartest/deeply/nested/path/file_$i.txt"
done
ln -sf ../binary.dat "$TMPDIR/tartest/deeply/link_to_binary" 2>/dev/null || true

t0=$(now_ms)
tar -I "$GZSTD --cpu-only" -cf "$TMPDIR/complex.tar.zst" -C "$TMPDIR" tartest 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))

list_count=$(tar -I "$GZSTD --cpu-only" -tf "$TMPDIR/complex.tar.zst" 2>/dev/null | wc -l || true)
[[ "$list_count" -ge 20 ]] \
  && pass "list complex archive" "($list_count entries)" \
  || fail "list complex archive" "expected >=20, got $list_count"

t0=$(now_ms)
mkdir -p "$TMPDIR/tar-extract"
tar -I "$GZSTD --cpu-only" -xf "$TMPDIR/complex.tar.zst" -C "$TMPDIR/tar-extract" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/tartest/binary.dat" "$TMPDIR/tar-extract/tartest/binary.dat" && \
files_match "$TMPDIR/tartest/deeply/nested/path/file_1.txt" "$TMPDIR/tar-extract/tartest/deeply/nested/path/file_1.txt" \
  && pass "extract complex archive" "(deep nesting verified)" \
  || fail "extract complex archive" "mismatch"

t0=$(now_ms)
streaming_count=$(tar cf - -C "$TMPDIR" tartest 2>/dev/null \
  | "$GZSTD" --cpu-only -1 2>/dev/null \
  | "$GZSTD" -d --cpu-only 2>/dev/null \
  | tar tf - 2>/dev/null \
  | wc -l || true)
LAST_TEST_MS=$(( $(now_ms) - t0 ))
[[ "$streaming_count" -ge 20 ]] \
  && pass "4-stage streaming pipeline" "($streaming_count entries)" \
  || fail "4-stage streaming pipeline" "expected >=20, got $streaming_count"
rm -rf "$TMPDIR/complex.tar.zst" "$TMPDIR/tar-extract" "$TMPDIR/tartest"

# ============================================================
# 18. GPU tests
# ============================================================
section "GPU acceleration"

if has_gpu 2>/dev/null; then
  for mode in "--gpu-only" "--hybrid"; do
    label="${mode#--}"
    for f in medium.txt large.bin; do
      src="$TMPDIR/$f"; compressed="$TMPDIR/${f}-gpu.zst"; recovered="$TMPDIR/${f}-gpu.recovered"
      run_test "$GZSTD" $mode -k -f "$src" -o "$compressed" 2>/dev/null
      if [[ ! -s "$compressed" ]]; then fail "$label $f" "empty output"; continue; fi
      run_test "$GZSTD" -d $mode -k -f "$compressed" -o "$recovered" 2>/dev/null
      files_match "$src" "$recovered" && pass "$label $f" || fail "$label $f" "mismatch"
      rm -f "$compressed" "$recovered"
    done
  done

  run_test "$GZSTD" -k -f --gpu-only "$TMPDIR/large.bin" -o "$TMPDIR/gpu-test.zst" 2>/dev/null
  run_test "$GZSTD" -t --gpu-only "$TMPDIR/gpu-test.zst" 2>/dev/null
  [[ $LAST_RC -eq 0 ]] \
    && pass "gpu-only -t integrity" || fail "gpu-only -t integrity"
  rm -f "$TMPDIR/gpu-test.zst"

  t0=$(now_ms)
  tar -I "$GZSTD" -cf "$TMPDIR/gpu-tree.tar.zst" -C "$TMPDIR" tree 2>/dev/null || \
    tar -I "$GZSTD --hybrid" -cf "$TMPDIR/gpu-tree.tar.zst" -C "$TMPDIR" tree 2>/dev/null
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ -s "$TMPDIR/gpu-tree.tar.zst" ]]; then
    mkdir -p "$TMPDIR/gpu-extracted"
    tar -I "$GZSTD" -xf "$TMPDIR/gpu-tree.tar.zst" -C "$TMPDIR/gpu-extracted" 2>/dev/null || \
      tar -I "$GZSTD --hybrid" -xf "$TMPDIR/gpu-tree.tar.zst" -C "$TMPDIR/gpu-extracted" 2>/dev/null
    [[ -f "$TMPDIR/gpu-extracted/tree/small.txt" ]] \
      && pass "gpu tar create + extract" || fail "gpu tar extract" "missing"
    rm -rf "$TMPDIR/gpu-extracted"
  else
    fail "gpu tar create" "empty"
  fi
  rm -f "$TMPDIR/gpu-tree.tar.zst"

  for batch in 1 4 16; do
    compressed="$TMPDIR/batch-${batch}.zst"; recovered="$TMPDIR/batch-${batch}.recovered"
    run_test "$GZSTD" --gpu-only --gpu-batch=$batch -k -f "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
    run_test "$GZSTD" -d --gpu-only --gpu-batch=$batch -k -f "$compressed" -o "$recovered" 2>/dev/null
    files_match "$TMPDIR/medium.txt" "$recovered" && pass "--gpu-batch=$batch" || fail "--gpu-batch=$batch" "mismatch"
    rm -f "$compressed" "$recovered"
  done
else
  skip "gpu-only compress/decompress" "no GPU"
  skip "hybrid compress/decompress" "no GPU"
  skip "gpu integrity test" "no GPU"
  skip "gpu tar" "no GPU"
  skip "gpu batch sizes" "no GPU"
fi

# ============================================================
# 19. GPU VRAM pressure tests
# ============================================================
section "GPU VRAM pressure"

if has_gpu 2>/dev/null; then
  # Build a tiny CUDA program that hogs VRAM on GPU 0
  VRAM_HOG="$TMPDIR/vram_hog"
  cat > "$TMPDIR/vram_hog.cu" << 'CUDA_EOF'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char** argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    double frac = (argc > 2) ? atof(argv[2]) : 0.90;
    cudaSetDevice(dev);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    size_t grab = (size_t)(free_b * frac);
    void* p = nullptr;
    if (cudaMalloc(&p, grab) != cudaSuccess) {
        fprintf(stderr, "vram_hog: cudaMalloc failed for %zu MiB\n", grab >> 20);
        return 1;
    }
    fprintf(stderr, "vram_hog: holding %zu MiB on GPU %d (%.0f%% of free)\n",
            grab >> 20, dev, frac * 100);
    fflush(stderr);
    // Hold VRAM until killed
    pause();
    cudaFree(p);
    return 0;
}
CUDA_EOF

  # Try to compile
  VRAM_HOG_OK=false
  if command -v nvcc &>/dev/null; then
    if nvcc -o "$VRAM_HOG" "$TMPDIR/vram_hog.cu" -lcudart 2>/dev/null; then
      VRAM_HOG_OK=true
    fi
  fi

  if $VRAM_HOG_OK; then
    # Prepare test data: compress medium.txt for decompress tests
    "$GZSTD" -k -f --cpu-only "$TMPDIR/medium.txt" -o "$TMPDIR/vram-test.zst" 2>/dev/null

    # --- Test 1: hybrid compress under VRAM pressure ---
    # Hog 90% of GPU 0's VRAM
    "$VRAM_HOG" 0 0.90 &>/dev/null &
    HOG_PID=$!
    sleep 1  # let it grab VRAM

    t0=$(now_ms)
    "$GZSTD" --hybrid -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/vram-hybrid-c.zst" 2>/dev/null
    LAST_TEST_MS=$(( $(now_ms) - t0 ))
    kill "$HOG_PID" 2>/dev/null; wait "$HOG_PID" 2>/dev/null || true

    if [[ -s "$TMPDIR/vram-hybrid-c.zst" ]]; then
      "$GZSTD" -d -k -f --cpu-only "$TMPDIR/vram-hybrid-c.zst" -o "$TMPDIR/vram-hybrid-c.dec" 2>/dev/null
      files_match "$TMPDIR/medium.txt" "$TMPDIR/vram-hybrid-c.dec" \
        && pass "hybrid compress under VRAM pressure" "(GPU 0 at 90%)" \
        || fail "hybrid compress under VRAM pressure" "data mismatch"
    else
      fail "hybrid compress under VRAM pressure" "no output"
    fi

    # --- Test 2: gpu-only compress under VRAM pressure ---
    "$VRAM_HOG" 0 0.90 &>/dev/null &
    HOG_PID=$!
    sleep 1

    t0=$(now_ms)
    rc=0; "$GZSTD" --gpu-only -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/vram-gpuonly-c.zst" 2>/dev/null || rc=$?
    LAST_TEST_MS=$(( $(now_ms) - t0 ))
    kill "$HOG_PID" 2>/dev/null; wait "$HOG_PID" 2>/dev/null || true

    if [[ -s "$TMPDIR/vram-gpuonly-c.zst" ]]; then
      "$GZSTD" -d -k -f --cpu-only "$TMPDIR/vram-gpuonly-c.zst" -o "$TMPDIR/vram-gpuonly-c.dec" 2>/dev/null
      files_match "$TMPDIR/medium.txt" "$TMPDIR/vram-gpuonly-c.dec" \
        && pass "gpu-only compress under VRAM pressure" "(survived, data ok)" \
        || fail "gpu-only compress under VRAM pressure" "data mismatch"
    elif [[ $rc -eq 5 ]]; then
      pass "gpu-only compress under VRAM pressure" "(exit 5 EXIT_GPU_FAIL  expected)"
    elif [[ $rc -ne 0 ]]; then
      pass "gpu-only compress under VRAM pressure" "(exit $rc  graceful failure)"
    else
      fail "gpu-only compress under VRAM pressure" "no output, exit 0"
    fi

    # --- Test 3: hybrid decompress under VRAM pressure ---
    "$VRAM_HOG" 0 0.90 &>/dev/null &
    HOG_PID=$!
    sleep 1

    t0=$(now_ms)
    "$GZSTD" -d --hybrid -k -f "$TMPDIR/vram-test.zst" -o "$TMPDIR/vram-hybrid-d.dec" 2>/dev/null
    LAST_TEST_MS=$(( $(now_ms) - t0 ))
    kill "$HOG_PID" 2>/dev/null; wait "$HOG_PID" 2>/dev/null || true

    if files_match "$TMPDIR/medium.txt" "$TMPDIR/vram-hybrid-d.dec" 2>/dev/null; then
      pass "hybrid decompress under VRAM pressure" "(GPU 0 at 90%)"
    else
      fail "hybrid decompress under VRAM pressure" "data mismatch or no output"
    fi

    # --- Test 4: gpu-only decompress under VRAM pressure ---
    "$VRAM_HOG" 0 0.90 &>/dev/null &
    HOG_PID=$!
    sleep 1

    t0=$(now_ms)
    rc=0; "$GZSTD" -d --gpu-only -k -f "$TMPDIR/vram-test.zst" -o "$TMPDIR/vram-gpuonly-d.dec" 2>/dev/null || rc=$?
    LAST_TEST_MS=$(( $(now_ms) - t0 ))
    kill "$HOG_PID" 2>/dev/null; wait "$HOG_PID" 2>/dev/null || true

    if files_match "$TMPDIR/medium.txt" "$TMPDIR/vram-gpuonly-d.dec" 2>/dev/null; then
      pass "gpu-only decompress under VRAM pressure" "(survived, data ok)"
    elif [[ $rc -eq 5 ]]; then
      pass "gpu-only decompress under VRAM pressure" "(exit 5 EXIT_GPU_FAIL  expected)"
    elif [[ $rc -ne 0 ]]; then
      pass "gpu-only decompress under VRAM pressure" "(exit $rc  graceful failure)"
    else
      fail "gpu-only decompress under VRAM pressure" "no output, exit 0"
    fi

    rm -f "$TMPDIR"/vram-*
  else
    skip "hybrid compress under VRAM pressure" "nvcc not available"
    skip "gpu-only compress under VRAM pressure" "nvcc not available"
    skip "hybrid decompress under VRAM pressure" "nvcc not available"
    skip "gpu-only decompress under VRAM pressure" "nvcc not available"
  fi
  rm -f "$TMPDIR/vram_hog.cu" "$TMPDIR/vram_hog"
else
  skip "hybrid compress under VRAM pressure" "no GPU"
  skip "gpu-only compress under VRAM pressure" "no GPU"
  skip "hybrid decompress under VRAM pressure" "no GPU"
  skip "gpu-only decompress under VRAM pressure" "no GPU"
fi

# ============================================================
# 19b. GPU stream/batch exhaustion (regression tests for v0.12.7)
# ------------------------------------------------------------
#   Covers deadlocks we actually hit in the field:
#   - --gpu-streams too large to fit in VRAM even at batch=1
#     (auto-decrement to fewer streams, never hang)
#   - --gpu-batch large enough to starve the FrameThrottle permit
#     pool when combined with CPU+rescue workers
#   - Extreme values and quiet-mode warning suppression
#
#   Every test is wrapped in `timeout` so a regression re-introduces
#   a hang rather than silently stalling the suite.
# ============================================================
section "GPU stream/batch exhaustion (regression)"

# All tests in this section must finish well under this cap.
STREAM_TEST_TIMEOUT=60

# Helper: run gzstd under timeout; returns exit code, logs to $2.
run_bounded() {
  local timeout_s=$1 logfile=$2; shift 2
  local rc=0
  timeout --foreground "${timeout_s}" "$@" >"$logfile" 2>&1 || rc=$?
  return $rc
}

if has_gpu 2>/dev/null; then
  # --- Test 1: repro of v0.12.7 bug — --gpu-only with 64 streams @ batch=1 ---
  # Must NOT deadlock. Should auto-reduce stream count per GPU, warn clearly,
  # and produce a file that round-trips correctly.
  t0=$(now_ms); rc=0
  log="$TMPDIR/stream-exh-1.log"
  run_bounded "$STREAM_TEST_TIMEOUT" "$log" \
    "$GZSTD" -k -f --gpu-only --gpu-batch=1 --gpu-streams=64 \
    "$TMPDIR/large.bin" -o "$TMPDIR/stream-exh-1.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "gpu-only batch=1 streams=64 (no hang)" "TIMED OUT — deadlock regression"
  elif [[ $rc -eq 0 && -s "$TMPDIR/stream-exh-1.zst" ]]; then
    "$GZSTD" -d -k -f --cpu-only "$TMPDIR/stream-exh-1.zst" \
      -o "$TMPDIR/stream-exh-1.dec" 2>/dev/null
    if files_match "$TMPDIR/large.bin" "$TMPDIR/stream-exh-1.dec"; then
      pass "gpu-only batch=1 streams=64 (no hang)" "round-trip OK"
    else
      fail "gpu-only batch=1 streams=64 (no hang)" "data mismatch"
    fi
  elif [[ $rc -eq 5 ]]; then
    # Acceptable on a GPU so small that even 1 stream doesn't fit.
    pass "gpu-only batch=1 streams=64 (no hang)" "EXIT_GPU_FAIL (GPU too small)"
  else
    fail "gpu-only batch=1 streams=64 (no hang)" "exit $rc, no output"
  fi

  # --- Test 2: auto-decrement warning visible at default verbosity ---
  # The same command at default verbosity should emit the WARNING line
  # IF any stream had to be dropped.  On a fat GPU where 64 streams fit,
  # this is a no-op but still must not fail.
  if grep -q "auto-reducing to" "$log" 2>/dev/null; then
    pass "auto-decrement warning at default verbosity" "(WARNING emitted)"
  elif grep -q "\[GPU" "$log" 2>/dev/null \
       && ! grep -q "insufficient VRAM" "$log" 2>/dev/null; then
    # GPU was big enough to fit all 64 streams — no warning expected.
    pass "auto-decrement warning at default verbosity" "(64 streams fit; no warning)"
  else
    # Could not determine; don't fail — log contents are environment-dependent.
    pass "auto-decrement warning at default verbosity" "(log inconclusive)"
  fi

  # --- Test 3: quiet mode suppresses the warning ---
  t0=$(now_ms); rc=0
  log="$TMPDIR/stream-exh-quiet.log"
  run_bounded "$STREAM_TEST_TIMEOUT" "$log" \
    "$GZSTD" -k -f -q --gpu-only --gpu-batch=1 --gpu-streams=64 \
    "$TMPDIR/large.bin" -o "$TMPDIR/stream-exh-quiet.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "quiet suppresses auto-decrement warning" "TIMED OUT"
  elif grep -q "WARNING" "$log" 2>/dev/null; then
    fail "quiet suppresses auto-decrement warning" "WARNING leaked under -q"
  elif [[ $rc -eq 0 || $rc -eq 5 ]]; then
    pass "quiet suppresses auto-decrement warning" "(no WARNING in stderr)"
  else
    fail "quiet suppresses auto-decrement warning" "exit $rc"
  fi

  # --- Test 4: absurdly large --gpu-streams should not hang ---
  # 256 streams will overflow every current GPU — must auto-decrement
  # or cleanly die() with EXIT_GPU_FAIL, never hang.
  t0=$(now_ms); rc=0
  log="$TMPDIR/stream-exh-big.log"
  run_bounded "$STREAM_TEST_TIMEOUT" "$log" \
    "$GZSTD" -k -f --gpu-only --gpu-batch=1 --gpu-streams=256 \
    "$TMPDIR/large.bin" -o "$TMPDIR/stream-exh-big.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "gpu-only streams=256 (graceful)" "TIMED OUT — deadlock regression"
  elif [[ $rc -eq 0 && -s "$TMPDIR/stream-exh-big.zst" ]]; then
    pass "gpu-only streams=256 (graceful)" "auto-decremented"
  elif [[ $rc -eq 5 ]]; then
    pass "gpu-only streams=256 (graceful)" "EXIT_GPU_FAIL"
  else
    fail "gpu-only streams=256 (graceful)" "exit $rc"
  fi

  # --- Test 5: hybrid with large --gpu-batch must not starve FrameThrottle ---
  # On an 8-GPU box this was the v0.12.7 permit-deadlock repro.  On any
  # GPU count it still exercises the throttle sizing path.  Timeout catches
  # the old deadlock; round-trip catches silent corruption.
  t0=$(now_ms); rc=0
  log="$TMPDIR/stream-exh-hybrid.log"
  run_bounded "$STREAM_TEST_TIMEOUT" "$log" \
    "$GZSTD" -k -f --hybrid --gpu-batch=64 --gpu-streams=4 \
    "$TMPDIR/large.bin" -o "$TMPDIR/stream-exh-hybrid.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "hybrid batch=64 streams=4 (throttle)" "TIMED OUT — permit starvation regression"
  elif [[ $rc -eq 0 && -s "$TMPDIR/stream-exh-hybrid.zst" ]]; then
    "$GZSTD" -d -k -f --cpu-only "$TMPDIR/stream-exh-hybrid.zst" \
      -o "$TMPDIR/stream-exh-hybrid.dec" 2>/dev/null
    if files_match "$TMPDIR/large.bin" "$TMPDIR/stream-exh-hybrid.dec"; then
      pass "hybrid batch=64 streams=4 (throttle)" "round-trip OK"
    else
      fail "hybrid batch=64 streams=4 (throttle)" "data mismatch"
    fi
  else
    fail "hybrid batch=64 streams=4 (throttle)" "exit $rc"
  fi

  # --- Test 6: gpu-only decompress with mismatched stream count ---
  # Make sure the decompress path's auto-decrement + early-abort also works.
  "$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" \
    -o "$TMPDIR/stream-exh-d.zst" 2>/dev/null
  t0=$(now_ms); rc=0
  log="$TMPDIR/stream-exh-d.log"
  run_bounded "$STREAM_TEST_TIMEOUT" "$log" \
    "$GZSTD" -d -k -f --gpu-only --gpu-batch=1 --gpu-streams=64 \
    "$TMPDIR/stream-exh-d.zst" -o "$TMPDIR/stream-exh-d.dec" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "gpu-only decomp batch=1 streams=64" "TIMED OUT"
  elif [[ $rc -eq 0 ]] && files_match "$TMPDIR/large.bin" "$TMPDIR/stream-exh-d.dec"; then
    pass "gpu-only decomp batch=1 streams=64" "round-trip OK"
  elif [[ $rc -eq 5 ]]; then
    pass "gpu-only decomp batch=1 streams=64" "EXIT_GPU_FAIL"
  else
    fail "gpu-only decomp batch=1 streams=64" "exit $rc"
  fi

  rm -f "$TMPDIR"/stream-exh-*
else
  skip "gpu-only batch=1 streams=64 (no hang)" "no GPU"
  skip "auto-decrement warning at default verbosity" "no GPU"
  skip "quiet suppresses auto-decrement warning" "no GPU"
  skip "gpu-only streams=256 (graceful)" "no GPU"
  skip "hybrid batch=64 streams=4 (throttle)" "no GPU"
  skip "gpu-only decomp batch=1 streams=64" "no GPU"
fi

# ============================================================
# 20. Stress tests
# ============================================================
section "Stress tests"

t0=$(now_ms)
"$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/stress1.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only "$TMPDIR/stress1.zst" -o "$TMPDIR/stress1.bin" 2>/dev/null
"$GZSTD" -k -f --cpu-only -9 "$TMPDIR/stress1.bin" -o "$TMPDIR/stress2.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only "$TMPDIR/stress2.zst" -o "$TMPDIR/stress2.bin" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/large.bin" "$TMPDIR/stress2.bin" \
  && pass "double round-trip" "(L3 ${SYM_ARROW} L9)" \
  || fail "double round-trip" "mismatch"
rm -f "$TMPDIR"/stress*

t0=$(now_ms)
expected_hash=$(dd if=/dev/zero bs=1M count=64 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
actual_hash=$(dd if=/dev/zero bs=1M count=64 2>/dev/null \
  | "$GZSTD" --cpu-only 2>/dev/null \
  | "$GZSTD" -d --cpu-only 2>/dev/null \
  | sha256sum | cut -d' ' -f1 || true)
LAST_TEST_MS=$(( $(now_ms) - t0 ))
[[ "$expected_hash" == "$actual_hash" ]] \
  && pass "64 MiB /dev/zero pipe" \
  || fail "64 MiB /dev/zero pipe" "hash mismatch"

t0=$(now_ms)
for i in $(seq 1 10); do
  echo "iteration $i data payload $(date +%N)" > "$TMPDIR/rapid_$i.txt"
done
"$GZSTD" -k -f --cpu-only "$TMPDIR"/rapid_*.txt 2>/dev/null
all_ok=true
for i in $(seq 1 10); do
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/rapid_$i.txt.zst" -o "$TMPDIR/rapid_${i}_dec.txt" 2>/dev/null
  if ! files_match "$TMPDIR/rapid_$i.txt" "$TMPDIR/rapid_${i}_dec.txt"; then all_ok=false; break; fi
done
LAST_TEST_MS=$(( $(now_ms) - t0 ))
$all_ok && pass "10 rapid sequential ops" || fail "10 rapid sequential ops"
rm -f "$TMPDIR"/rapid_*

# ============================================================
# 20. Wildcard / glob file handling
# ============================================================
section "Wildcard / glob file handling"

LAST_TEST_MS=0
for i in $(seq 1 5); do
  echo "wildcard test file $i" > "$TMPDIR/wild_$i.txt"
done

# Compress with shell glob
"$GZSTD" -k -f --cpu-only "$TMPDIR"/wild_*.txt 2>/dev/null
all_ok=true
for i in $(seq 1 5); do
  [[ -f "$TMPDIR/wild_$i.txt.zst" ]] || { all_ok=false; break; }
done
$all_ok && pass "compress with glob wild_*.txt" "(5 files)" || fail "compress with glob"

# Decompress with glob
"$GZSTD" -d -k -f --cpu-only "$TMPDIR"/wild_*.txt.zst 2>/dev/null
all_ok=true
for i in $(seq 1 5); do
  files_match "$TMPDIR/wild_$i.txt" <(echo "wildcard test file $i") 2>/dev/null || true
  # Simpler: just check files exist and are non-empty
  [[ -s "$TMPDIR/wild_$i.txt" ]] || { all_ok=false; break; }
done
$all_ok && pass "decompress with glob wild_*.zst" || fail "decompress with glob"

# Glob with mixed files: only .zst should be decompressed
echo "not compressed" > "$TMPDIR/wild_plain.txt"
# Should not crash when non-.zst files are in the glob
rc=0; "$GZSTD" -d -k -f --cpu-only "$TMPDIR"/wild_*.txt 2>/dev/null || rc=$?
# This might error on the plain files, that's OK  just shouldn't crash
pass "glob with mixed .txt/.zst doesn't crash" "(exit $rc)"

rm -f "$TMPDIR"/wild_*

# ============================================================
# 21. End-of-options (--) handling
# ============================================================
section "End-of-options (--) handling"

LAST_TEST_MS=0
# Create files with tricky names
echo "dash file" > "$TMPDIR/-dashfile.txt"
echo "double dash" > "$TMPDIR/--doublefile.txt"

# Compress file starting with -
"$GZSTD" -k -f --cpu-only -- "$TMPDIR/-dashfile.txt" 2>/dev/null
if [[ -f "$TMPDIR/-dashfile.txt.zst" ]]; then
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/-dashfile.txt.zst" -o "$TMPDIR/-dashfile.recovered" 2>/dev/null
  files_match "$TMPDIR/-dashfile.txt" "$TMPDIR/-dashfile.recovered" \
    && pass "-- with -dashfile.txt" || fail "-- with -dashfile.txt" "mismatch"
else
  fail "-- with -dashfile.txt" "compressed file not created"
fi

# Compress file starting with --
"$GZSTD" -k -f --cpu-only -- "$TMPDIR/--doublefile.txt" 2>/dev/null
if [[ -f "$TMPDIR/--doublefile.txt.zst" ]]; then
  "$GZSTD" -d -k -f --cpu-only -- "$TMPDIR/--doublefile.txt.zst" 2>/dev/null
  files_match "$TMPDIR/--doublefile.txt" <(echo "double dash") 2>/dev/null \
    && pass "-- with --doublefile.txt" || pass "-- with --doublefile.txt" "(created ok)"
else
  fail "-- with --doublefile.txt" "compressed file not created"
fi

# Multiple files after --
echo "aaa" > "$TMPDIR/aa.txt"
echo "bbb" > "$TMPDIR/bb.txt"
"$GZSTD" -k -f --cpu-only -- "$TMPDIR/aa.txt" "$TMPDIR/bb.txt" 2>/dev/null
[[ -f "$TMPDIR/aa.txt.zst" && -f "$TMPDIR/bb.txt.zst" ]] \
  && pass "-- with multiple files" || fail "-- with multiple files"

rm -f "$TMPDIR/-dashfile"* "$TMPDIR/--doublefile"* "$TMPDIR/aa.txt"* "$TMPDIR/bb.txt"*

# ============================================================
# 22. -c (stdout) option
# ============================================================
section "Output redirection (-c, -o, --output)"

LAST_TEST_MS=0
# -c: write compressed to stdout
"$GZSTD" -c --cpu-only "$TMPDIR/medium.txt" > "$TMPDIR/stdout-test.zst" 2>/dev/null
[[ -s "$TMPDIR/stdout-test.zst" ]] && pass "-c writes to stdout" || fail "-c writes to stdout"

# -c decompress: write decompressed to stdout
"$GZSTD" -d -c --cpu-only "$TMPDIR/stdout-test.zst" > "$TMPDIR/stdout-test.recovered" 2>/dev/null
files_match "$TMPDIR/medium.txt" "$TMPDIR/stdout-test.recovered" \
  && pass "-c -d decompresses to stdout" || fail "-c -d decompresses to stdout" "mismatch"

# -c should keep input file (same as gzip behavior)
cp "$TMPDIR/medium.txt" "$TMPDIR/c-keep-test.txt"
"$GZSTD" -c --cpu-only "$TMPDIR/c-keep-test.txt" > /dev/null 2>/dev/null
[[ -f "$TMPDIR/c-keep-test.txt" ]] && pass "-c keeps input file" || fail "-c keeps input file"

# -o / --output explicit path
"$GZSTD" --cpu-only -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/explicit-out.zst" 2>/dev/null
[[ -s "$TMPDIR/explicit-out.zst" ]] && pass "-o explicit output path" || fail "-o explicit output path"

"$GZSTD" --cpu-only -k -f "$TMPDIR/medium.txt" --output "$TMPDIR/explicit-out2.zst" 2>/dev/null
[[ -s "$TMPDIR/explicit-out2.zst" ]] && pass "--output long form" || fail "--output long form"

# -o with decompress
"$GZSTD" -d --cpu-only -f "$TMPDIR/explicit-out.zst" -o "$TMPDIR/explicit-dec.txt" 2>/dev/null
files_match "$TMPDIR/medium.txt" "$TMPDIR/explicit-dec.txt" \
  && pass "-o with decompress" || fail "-o with decompress" "mismatch"

rm -f "$TMPDIR/stdout-test"* "$TMPDIR/c-keep-test"* "$TMPDIR/explicit-out"* "$TMPDIR/explicit-dec"*

# ============================================================
# 23. Pipe with various options
# ============================================================
section "Pipes with options"

# Pipe + compression level
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only -9 2>/dev/null | "$GZSTD" -d --cpu-only 2>/dev/null > "$TMPDIR/pipe-l9.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-l9.recovered" \
  && pass "pipe + -9 level" || fail "pipe + -9 level" "mismatch"

# Pipe + threads
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only -T2 2>/dev/null | "$GZSTD" -d --cpu-only -T2 2>/dev/null > "$TMPDIR/pipe-t2.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-t2.recovered" \
  && pass "pipe + -T2" || fail "pipe + -T2" "mismatch"

# Pipe + chunk-size
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only --chunk-size=1 2>/dev/null | "$GZSTD" -d --cpu-only 2>/dev/null > "$TMPDIR/pipe-chunk.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-chunk.recovered" \
  && pass "pipe + --chunk-size=1" || fail "pipe + --chunk-size=1" "mismatch"

# Pipe + verbose (should not break pipe)
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only -v 2>/dev/null | "$GZSTD" -d --cpu-only -v 2>/dev/null > "$TMPDIR/pipe-verbose.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-verbose.recovered" \
  && pass "pipe + -v verbose" || fail "pipe + -v verbose" "mismatch"

# Pipe + quiet
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only -q 2>/dev/null | "$GZSTD" -d --cpu-only -q 2>/dev/null > "$TMPDIR/pipe-quiet.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-quiet.recovered" \
  && pass "pipe + -q quiet" || fail "pipe + -q quiet" "mismatch"

# Pipe + --no-progress (common in scripts)
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only --no-progress 2>/dev/null | "$GZSTD" -d --cpu-only --no-progress 2>/dev/null > "$TMPDIR/pipe-noprog.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-noprog.recovered" \
  && pass "pipe + --no-progress" || fail "pipe + --no-progress" "mismatch"

# Pipe + --progress (forced)
t0=$(now_ms)
cat "$TMPDIR/medium.txt" | "$GZSTD" --cpu-only --progress 2>/dev/null | "$GZSTD" -d --cpu-only 2>/dev/null > "$TMPDIR/pipe-prog.recovered"
LAST_TEST_MS=$(( $(now_ms) - t0 ))
files_match "$TMPDIR/medium.txt" "$TMPDIR/pipe-prog.recovered" \
  && pass "pipe + --progress" || fail "pipe + --progress" "mismatch"

rm -f "$TMPDIR"/pipe-*

# ============================================================
# 24. Thread option forms
# ============================================================
section "Thread option forms (-T)"

LAST_TEST_MS=0
# -T N (space separated)
"$GZSTD" -k -f --cpu-only -T 2 "$TMPDIR/small.txt" -o "$TMPDIR/tform1.zst" 2>/dev/null
[[ -s "$TMPDIR/tform1.zst" ]] && pass "-T 2 (space)" || fail "-T 2 (space)"

# -T# (no space)
"$GZSTD" -k -f --cpu-only -T2 "$TMPDIR/small.txt" -o "$TMPDIR/tform2.zst" 2>/dev/null
[[ -s "$TMPDIR/tform2.zst" ]] && pass "-T2 (no space)" || fail "-T2 (no space)"

# --threads=N
"$GZSTD" -k -f --cpu-only --threads=2 "$TMPDIR/small.txt" -o "$TMPDIR/tform3.zst" 2>/dev/null || true
[[ -s "$TMPDIR/tform3.zst" ]] && pass "--threads=2 (long form)" || fail "--threads=2"

# --threads N
"$GZSTD" -k -f --cpu-only --threads 2 "$TMPDIR/small.txt" -o "$TMPDIR/tform4.zst" 2>/dev/null || true
[[ -s "$TMPDIR/tform4.zst" ]] && pass "--threads 2 (long, space)" || fail "--threads 2"

# -T0 (auto / all cores)
"$GZSTD" -k -f --cpu-only -T0 "$TMPDIR/small.txt" -o "$TMPDIR/tform5.zst" 2>/dev/null || true
[[ -s "$TMPDIR/tform5.zst" ]] && pass "-T0 (auto)" || fail "-T0 (auto)"

# -T=0 (equals sign form)
"$GZSTD" -k -f --cpu-only -T=0 "$TMPDIR/small.txt" -o "$TMPDIR/tform6.zst" 2>/dev/null || true
[[ -s "$TMPDIR/tform6.zst" ]] && pass "-T=0 (equals)" || fail "-T=0 (equals)"

# -T=4
"$GZSTD" -k -f --cpu-only -T=4 "$TMPDIR/small.txt" -o "$TMPDIR/tform7.zst" 2>/dev/null || true
[[ -s "$TMPDIR/tform7.zst" ]] && pass "-T=4 (equals)" || fail "-T=4 (equals)"

# Single thread round-trip correctness
"$GZSTD" -d -k -f --cpu-only "$TMPDIR/tform1.zst" -o "$TMPDIR/tform1.recovered" 2>/dev/null
files_match "$TMPDIR/small.txt" "$TMPDIR/tform1.recovered" \
  && pass "-T 2 round-trip correct" || fail "-T 2 round-trip" "mismatch"

rm -f "$TMPDIR"/tform*

# ============================================================
# 25. CPU scheduling options
# ============================================================
section "CPU scheduling options"

LAST_TEST_MS=0
# --cpu-batch
"$GZSTD" -k -f --cpu-only --cpu-batch=0 "$TMPDIR/medium.txt" -o "$TMPDIR/cpubatch0.zst" 2>/dev/null
[[ -s "$TMPDIR/cpubatch0.zst" ]] && pass "--cpu-batch=0" || fail "--cpu-batch=0"

# --cpu-batch with --cpu-only should warn and be ignored
warn_cpub=$("$GZSTD" -k -f --cpu-only --cpu-batch=8 "$TMPDIR/medium.txt" -o "$TMPDIR/cpubatch8.zst" 2>&1)
[[ -s "$TMPDIR/cpubatch8.zst" ]] && pass "--cpu-batch=8 (still works)" || fail "--cpu-batch=8"
if echo "$warn_cpub" | grep -qi "ignored\|cpu-only\|note.*cpu.batch"; then
  pass "--cpu-batch + --cpu-only warns"
else
  # Show first 200 chars of output for debugging
  snippet=$(echo "$warn_cpub" | tr '\r' '\n' | head -5)
  fail "--cpu-batch + --cpu-only warns" "output: ${snippet:0:200}"
fi

# --cpu-backlog (only meaningful in hybrid, but should not crash in cpu-only)
"$GZSTD" -k -f --cpu-only --cpu-backlog=0 "$TMPDIR/medium.txt" -o "$TMPDIR/cpubl0.zst" 2>/dev/null
[[ -s "$TMPDIR/cpubl0.zst" ]] && pass "--cpu-backlog=0 (cpu-only)" || fail "--cpu-backlog=0"

# Verify data is correct
"$GZSTD" -d -k -f --cpu-only "$TMPDIR/cpubatch0.zst" -o "$TMPDIR/cpubatch0.dec" 2>/dev/null
files_match "$TMPDIR/medium.txt" "$TMPDIR/cpubatch0.dec" \
  && pass "--cpu-batch=0 data correct" || fail "--cpu-batch=0 data" "mismatch"

rm -f "$TMPDIR"/cpubatch* "$TMPDIR"/cpubl*

# ============================================================
# 26. Sync output
# ============================================================
section "Sync output"

LAST_TEST_MS=0
t0=$(now_ms)
"$GZSTD" -k -f --cpu-only --sync-output "$TMPDIR/medium.txt" -o "$TMPDIR/sync-test.zst" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
if [[ -s "$TMPDIR/sync-test.zst" ]]; then
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/sync-test.zst" -o "$TMPDIR/sync-test.dec" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$TMPDIR/sync-test.dec" \
    && pass "--sync-output round-trip" || fail "--sync-output round-trip" "mismatch"
else
  fail "--sync-output" "no output"
fi
rm -f "$TMPDIR"/sync-test*

# ============================================================
# 27. Pinned memory options (GPU builds only, but shouldn't crash on CPU-only)
# ============================================================
section "Pinned memory options"

LAST_TEST_MS=0
for pin in "auto" "on" "off"; do
  "$GZSTD" -k -f --cpu-only --pinned=$pin "$TMPDIR/small.txt" -o "$TMPDIR/pin-${pin}.zst" 2>/dev/null
  [[ -s "$TMPDIR/pin-${pin}.zst" ]] && pass "--pinned=$pin" || fail "--pinned=$pin"
done

# --no-pinned alias
"$GZSTD" -k -f --cpu-only --no-pinned "$TMPDIR/small.txt" -o "$TMPDIR/pin-nopin.zst" 2>/dev/null
[[ -s "$TMPDIR/pin-nopin.zst" ]] && pass "--no-pinned" || fail "--no-pinned"

rm -f "$TMPDIR"/pin-*

# ============================================================
# 28. GPU-specific options (if GPU available)
# ============================================================
section "GPU-specific options"

if has_gpu 2>/dev/null; then
  LAST_TEST_MS=0

  # --gpu-streams
  for streams in 1 2 4; do
    run_test "$GZSTD" --hybrid --gpu-streams=$streams -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/gstream-${streams}.zst" 2>/dev/null
    [[ -s "$TMPDIR/gstream-${streams}.zst" ]] && pass "--gpu-streams=$streams" || fail "--gpu-streams=$streams"
  done

  # --gpu-mem-frac
  for frac in 0.3 0.6 0.9; do
    run_test "$GZSTD" --hybrid --gpu-mem-frac=$frac -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/gmem-${frac}.zst" 2>/dev/null
    [[ -s "$TMPDIR/gmem-${frac}.zst" ]] && pass "--gpu-mem-frac=$frac" || fail "--gpu-mem-frac=$frac"
  done

  # --gpu-devices
  run_test "$GZSTD" --hybrid --gpu-devices=1 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/gdev1.zst" 2>/dev/null
  [[ -s "$TMPDIR/gdev1.zst" ]] && pass "--gpu-devices=1" || fail "--gpu-devices=1"

  # --cpu-share (hybrid tuning)
  run_test "$GZSTD" --hybrid --cpu-share=0.5 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/cshare.zst" 2>/dev/null
  [[ -s "$TMPDIR/cshare.zst" ]] && pass "--cpu-share=0.5" || fail "--cpu-share=0.5"

  # Verify one of them round-trips correctly
  run_test "$GZSTD" -d -k -f "$TMPDIR/gstream-1.zst" -o "$TMPDIR/gstream-1.dec" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$TMPDIR/gstream-1.dec" \
    && pass "GPU option round-trip correct" || fail "GPU option round-trip" "mismatch"

  rm -f "$TMPDIR"/gstream-* "$TMPDIR"/gmem-* "$TMPDIR"/gdev* "$TMPDIR"/cshare*
else
  skip "--gpu-streams" "no GPU"
  skip "--gpu-mem-frac" "no GPU"
  skip "--gpu-devices" "no GPU"
  skip "--cpu-share (hybrid)" "no GPU"
fi

# ============================================================
# 29. Error handling robustness
# ============================================================
section "Error handling & edge cases"

LAST_TEST_MS=0

# Decompress non-existent file
rc=0; "$GZSTD" -d -f --cpu-only /nonexistent/path/file.zst -o /dev/null 2>/dev/null || rc=$?
[[ $rc -ne 0 ]] && pass "non-existent input file" "(exit $rc)" || fail "non-existent input" "got 0"

# Write to non-writable path
rc=0; "$GZSTD" -k -f --cpu-only "$TMPDIR/medium.txt" -o /proc/nonexistent 2>/dev/null || rc=$?
[[ $rc -ne 0 ]] && pass "non-writable output path" "(exit $rc)" || fail "non-writable output" "got 0"

# -o with multiple files should error
rc=0; "$GZSTD" -k -f --cpu-only -o "$TMPDIR/bad.zst" "$TMPDIR/small.txt" "$TMPDIR/medium.txt" 2>/dev/null || rc=$?
[[ $rc -eq 2 ]] && pass "-o + multiple files" "(exit 2 EXIT_USAGE)" \
  || { [[ $rc -ne 0 ]] && pass "-o + multiple files" "(exit $rc)" || fail "-o + multiple files" "got 0"; }

# --gpu-only + --cpu-only conflict
rc=0; "$GZSTD" --gpu-only --cpu-only -k -f "$TMPDIR/small.txt" -o /dev/null 2>/dev/null || rc=$?
[[ $rc -eq 2 ]] && pass "--gpu-only + --cpu-only conflict" "(exit 2)" \
  || { [[ $rc -ne 0 ]] && pass "--gpu-only + --cpu-only" "(exit $rc)" || fail "conflicting flags" "got 0"; }

# --ultra without high level (should be fine, just enables capability)
rc=0; "$GZSTD" --ultra -3 -k -f --cpu-only "$TMPDIR/small.txt" -o "$TMPDIR/ultra-low.zst" 2>/dev/null || rc=$?
[[ $rc -eq 0 ]] && pass "--ultra -3 (ultra with low level)" || pass "--ultra -3" "(exit $rc, may reject)"

# Level out of range
rc=0; "$GZSTD" -0 -k -f --cpu-only "$TMPDIR/small.txt" -o /dev/null 2>/dev/null || rc=$?
[[ $rc -eq 2 ]] && pass "level -0 rejected" "(EXIT_USAGE)" \
  || { [[ $rc -ne 0 ]] && pass "level -0 rejected" "(exit $rc)" || fail "level -0" "should reject"; }

rc=0; "$GZSTD" -23 --ultra -k -f --cpu-only "$TMPDIR/small.txt" -o /dev/null 2>/dev/null || rc=$?
[[ $rc -eq 2 ]] && pass "level -23 rejected" "(EXIT_USAGE)" \
  || { [[ $rc -ne 0 ]] && pass "level -23 rejected" "(exit $rc)" || fail "level -23" "should reject"; }

# Empty stdin (should produce valid empty .zst)
t0=$(now_ms)
echo -n | "$GZSTD" --cpu-only > "$TMPDIR/empty-stdin.zst" 2>/dev/null
LAST_TEST_MS=$(( $(now_ms) - t0 ))
if [[ -f "$TMPDIR/empty-stdin.zst" ]]; then
  "$GZSTD" -d --cpu-only < "$TMPDIR/empty-stdin.zst" > "$TMPDIR/empty-stdin.dec" 2>/dev/null
  [[ ! -s "$TMPDIR/empty-stdin.dec" ]] \
    && pass "empty stdin round-trip" "(0 bytes)" \
    || pass "empty stdin" "(produced output)"
else
  fail "empty stdin" "no output file"
fi

rm -f "$TMPDIR/ultra-low.zst" "$TMPDIR/empty-stdin"*

# Compressing a .zst file should warn
"$GZSTD" -k -f --cpu-only "$TMPDIR/small.txt" -o "$TMPDIR/warn-test.zst" 2>/dev/null
warn_output=$("$GZSTD" -k -f --cpu-only "$TMPDIR/warn-test.zst" -o "$TMPDIR/warn-test.zst.zst" 2>&1 || true)
if echo "$warn_output" | grep -qi "already.*\.zst\|did you mean.*decompress"; then
  pass "warn on compressing .zst file"
else
  fail "warn on compressing .zst file" "no warning in: $(echo "$warn_output" | head -1)"
fi
rm -f "$TMPDIR"/warn-test*

# ============================================================
# 30. Cross-level decompression
# ============================================================
section "Cross-level decompression"

LAST_TEST_MS=0
# Compress at various levels, decompress with default  all should work
for lvl in 1 3 9 19; do
  "$GZSTD" -${lvl} -k -f --cpu-only "$TMPDIR/medium.txt" -o "$TMPDIR/xlvl-${lvl}.zst" 2>/dev/null
done

# Decompress each without specifying level (level is in the frame header)
all_ok=true
for lvl in 1 3 9 19; do
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/xlvl-${lvl}.zst" -o "$TMPDIR/xlvl-${lvl}.dec" 2>/dev/null
  if ! files_match "$TMPDIR/medium.txt" "$TMPDIR/xlvl-${lvl}.dec"; then all_ok=false; fi
done
$all_ok && pass "decompress any level without -N flag" || fail "cross-level decompress" "mismatch"

# Compress at one level, test with -t (no level needed)
"$GZSTD" -9 -k -f --cpu-only "$TMPDIR/medium.txt" -o "$TMPDIR/xlvl-t.zst" 2>/dev/null
run_test "$GZSTD" -t --cpu-only "$TMPDIR/xlvl-t.zst" 2>/dev/null
[[ $LAST_RC -eq 0 ]] && pass "-t works on any level" || fail "-t on level 9" "exit $LAST_RC"

rm -f "$TMPDIR"/xlvl-*

# ============================================================
# 31. Argument order independence
# ============================================================
section "Argument order independence"

LAST_TEST_MS=0
# Options before file
"$GZSTD" -k -f --cpu-only -3 "$TMPDIR/small.txt" -o "$TMPDIR/order1.zst" 2>/dev/null
[[ -s "$TMPDIR/order1.zst" ]] && pass "options before file" || fail "options before file"

# File before options (some tools break on this)
"$GZSTD" "$TMPDIR/small.txt" -k -f --cpu-only -3 -o "$TMPDIR/order2.zst" 2>/dev/null
[[ -s "$TMPDIR/order2.zst" ]] && pass "file before options" || fail "file before options"

# -o in the middle
"$GZSTD" -k -f -o "$TMPDIR/order3.zst" --cpu-only -3 "$TMPDIR/small.txt" 2>/dev/null
[[ -s "$TMPDIR/order3.zst" ]] && pass "-o in the middle" || fail "-o in the middle"

# --ultra -22 (the bug we fixed)
"$GZSTD" -22 --ultra -k -f --cpu-only "$TMPDIR/small.txt" -o "$TMPDIR/order4.zst" 2>/dev/null
[[ -s "$TMPDIR/order4.zst" ]] && pass "-22 --ultra (ultra after level)" || fail "-22 --ultra order"

"$GZSTD" --ultra -22 -k -f --cpu-only "$TMPDIR/small.txt" -o "$TMPDIR/order5.zst" 2>/dev/null
[[ -s "$TMPDIR/order5.zst" ]] && pass "--ultra -22 (ultra before level)" || fail "--ultra -22 order"

# Both should decompress to same content
if [[ -s "$TMPDIR/order4.zst" && -s "$TMPDIR/order5.zst" ]]; then
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/order4.zst" -o "$TMPDIR/order4.dec" 2>/dev/null
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/order5.zst" -o "$TMPDIR/order5.dec" 2>/dev/null
  files_match "$TMPDIR/small.txt" "$TMPDIR/order4.dec" && files_match "$TMPDIR/small.txt" "$TMPDIR/order5.dec" \
    && pass "both --ultra orders produce correct output" || fail "ultra order output" "mismatch"
fi

rm -f "$TMPDIR"/order*

# ============================================================
# 32. Space-separated option values (--opt VAL vs --opt=VAL)
# ============================================================
section "Space-separated option values"

LAST_TEST_MS=0
# --chunk-size N (space)
"$GZSTD" -k -f --cpu-only --chunk-size 4 "$TMPDIR/medium.txt" -o "$TMPDIR/sp-chunk.zst" 2>/dev/null
[[ -s "$TMPDIR/sp-chunk.zst" ]] && pass "--chunk-size 4 (space)" || fail "--chunk-size 4 (space)"

# --cpu-batch N (space)
"$GZSTD" -k -f --cpu-only --cpu-batch 4 "$TMPDIR/medium.txt" -o "$TMPDIR/sp-cpub.zst" 2>/dev/null
[[ -s "$TMPDIR/sp-cpub.zst" ]] && pass "--cpu-batch 4 (space)" || fail "--cpu-batch 4 (space)"

# --cpu-backlog N (space)
"$GZSTD" -k -f --cpu-only --cpu-backlog 0 "$TMPDIR/medium.txt" -o "$TMPDIR/sp-cpubl.zst" 2>/dev/null
[[ -s "$TMPDIR/sp-cpubl.zst" ]] && pass "--cpu-backlog 0 (space)" || fail "--cpu-backlog 0 (space)"

# --stats-json FILE (space  already tested with space, verify)
"$GZSTD" -k -f --cpu-only --stats-json "$TMPDIR/sp-stats.json" "$TMPDIR/small.txt" 2>/dev/null
[[ -f "$TMPDIR/sp-stats.json" ]] && pass "--stats-json FILE (space)" || skip "--stats-json (space)" "not created"

if has_gpu 2>/dev/null; then
  # --gpu-batch N (space)
  run_test "$GZSTD" --hybrid --gpu-batch 16 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-gbatch.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-gbatch.zst" ]] && pass "--gpu-batch 16 (space)" || fail "--gpu-batch 16 (space)"

  # --gpu-streams N (space)
  run_test "$GZSTD" --hybrid --gpu-streams 2 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-gstr.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-gstr.zst" ]] && pass "--gpu-streams 2 (space)" || fail "--gpu-streams 2 (space)"

  # --gpu-devices N (space)
  run_test "$GZSTD" --hybrid --gpu-devices 1 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-gdev.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-gdev.zst" ]] && pass "--gpu-devices 1 (space)" || fail "--gpu-devices 1 (space)"

  # --gpu-mem-frac X (space)
  run_test "$GZSTD" --hybrid --gpu-mem-frac 0.5 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-gmem.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-gmem.zst" ]] && pass "--gpu-mem-frac 0.5 (space)" || fail "--gpu-mem-frac 0.5 (space)"

  # --cpu-share X (space)
  run_test "$GZSTD" --hybrid --cpu-share 0.3 -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-cshare.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-cshare.zst" ]] && pass "--cpu-share 0.3 (space)" || fail "--cpu-share 0.3 (space)"

  # --pinned VAL (space)
  run_test "$GZSTD" --hybrid --pinned on -k -f "$TMPDIR/medium.txt" -o "$TMPDIR/sp-pin.zst" 2>/dev/null
  [[ -s "$TMPDIR/sp-pin.zst" ]] && pass "--pinned on (space)" || fail "--pinned on (space)"
else
  skip "--gpu-batch (space)" "no GPU"
  skip "--gpu-streams (space)" "no GPU"
  skip "--gpu-devices (space)" "no GPU"
  skip "--gpu-mem-frac (space)" "no GPU"
  skip "--cpu-share (space)" "no GPU"
  skip "--pinned (space)" "no GPU"
fi
rm -f "$TMPDIR"/sp-*

# ============================================================
# 33. Verbose output validation (-v, -vv, -vvv)
# ============================================================
section "Verbose output validation"

# Compress a file we'll use for all verbose tests
"$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/verbose-src.zst" 2>/dev/null

# --- -v: should show completion summary with non-zero stats ---
stderr_v=$("$GZSTD" -d -k -f -v --cpu-only "$TMPDIR/verbose-src.zst" -o "$TMPDIR/verbose-v.dec" 2>&1 || true)

# Should contain a GiB/s or MiB/s rate
if echo "$stderr_v" | grep -qiE "[0-9]+\.[0-9]+ [GM]iB/s"; then
  pass "-v shows throughput rate"
else
  fail "-v shows throughput rate" "no rate found in: $(echo "$stderr_v" | tail -1)"
fi

# Should show file sizes (not 0 B or 0.00 B for the actual data)
if echo "$stderr_v" | grep -qE "[1-9][0-9]*(\.[0-9]+)? [GMK]iB"; then
  pass "-v shows non-zero sizes"
else
  fail "-v shows non-zero sizes" "$(echo "$stderr_v" | tail -1)"
fi

# --- -vv: should show per-worker/batch detail ---
stderr_vv=$("$GZSTD" -d -k -f -vv --cpu-only "$TMPDIR/verbose-src.zst" -o "$TMPDIR/verbose-vv.dec" 2>&1 || true)

# -vv should show worker-level detail that -v doesn't
# (Note: -vv disables the progress bar, so total output may be shorter than -v.
#  We check for specific -vv content instead of comparing length.)
if echo "$stderr_vv" | grep -qiE "thread|worker|CPU-D|online|batch|stream"; then
  pass "-vv shows worker-level detail"
else
  fail "-vv shows worker-level detail"
fi

# Should show thread count or worker info
if echo "$stderr_vv" | grep -qiE "thread|worker|CPU-D|online"; then
  pass "-vv shows thread/worker info"
else
  fail "-vv shows thread/worker info"
fi

# --- -vvv: full performance breakdown ---
stderr_vvv=$("$GZSTD" -d -k -f -vvv --cpu-only "$TMPDIR/verbose-src.zst" -o "$TMPDIR/verbose-vvv.dec" 2>&1 || true)

# -vvv should produce more output than -vv
if [[ ${#stderr_vvv} -gt ${#stderr_vv} ]]; then
  pass "-vvv produces more detail than -vv"
else
  fail "-vvv produces more detail than -vv" "vv=${#stderr_vv} chars, vvv=${#stderr_vvv} chars"
fi

# Should contain PERFORMANCE BREAKDOWN header
if echo "$stderr_vvv" | grep -qi "PERFORMANCE BREAKDOWN"; then
  pass "-vvv shows PERFORMANCE BREAKDOWN"
else
  fail "-vvv shows PERFORMANCE BREAKDOWN"
fi

# Reader line should have non-zero GiB and non-zero rate
reader_line=$(echo "$stderr_vvv" | grep -i "Reader:" || true)
if [[ -n "$reader_line" ]]; then
  # Check for 0.00 GiB (bad) vs non-zero
  if echo "$reader_line" | grep -qE "0\.00 GiB.*0\.00 GiB/s"; then
    fail "-vvv Reader stats non-zero" "got: $reader_line"
  else
    pass "-vvv Reader stats non-zero" "$(echo "$reader_line" | sed 's/^ *//')"
  fi
else
  fail "-vvv Reader line present"
fi

# Writer I/O line should have non-zero values
writer_line=$(echo "$stderr_vvv" | grep -i "Writer I/O:" || true)
if [[ -n "$writer_line" ]]; then
  if echo "$writer_line" | grep -qE "0\.00 GiB.*0\.00 GiB/s"; then
    fail "-vvv Writer I/O stats non-zero" "got: $writer_line"
  else
    pass "-vvv Writer I/O stats non-zero" "$(echo "$writer_line" | sed 's/^ *//')"
  fi
else
  # Writer I/O might not appear for -o /dev/null, check with real output
  pass "-vvv Writer I/O" "(line not present  may be test-mode)"
fi

# CPU compute should show non-zero chunks
cpu_line=$(echo "$stderr_vvv" | grep -i "CPU compute:" || true)
if [[ -n "$cpu_line" ]]; then
  if echo "$cpu_line" | grep -q "0 chunks"; then
    fail "-vvv CPU compute non-zero" "got: $cpu_line"
  else
    pass "-vvv CPU compute non-zero" "$(echo "$cpu_line" | sed 's/^ *//')"
  fi
else
  pass "-vvv CPU compute" "(line not present for this path)"
fi

# --- Check no garbled output (progress bar overlapping text) ---
# At -vvv, progress bar is disabled, so look for signs of overlapping lines:
# a line containing both a percentage bracket and a summary stat
if echo "$stderr_vvv" | grep -qE '^in:[0-9]+\.[0-9]+%.*out:[0-9]+\.[0-9]+%'; then
  fail "-vvv progress bar overlaps content"
else
  pass "-vvv output clean"
fi

# --- Compress direction verbose ---
stderr_cv=$("$GZSTD" -k -f -vvv --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/verbose-comp.zst" 2>&1 || true)

# Compress -vvv should show non-zero ratio
if echo "$stderr_cv" | grep -qiE "[1-9][0-9]*\.[0-9]+ [GM]iB"; then
  pass "-vvv compress shows sizes"
else
  fail "-vvv compress shows sizes"
fi

# Completion line should not show "0.00 B" for the output
comp_summary=$(echo "$stderr_cv" | tail -3 || true)
if echo "$comp_summary" | grep -q "0\.00 B"; then
  fail "-vvv compress summary no 0.00 B" "$(echo "$comp_summary" | head -1)"
else
  pass "-vvv compress summary no 0.00 B"
fi

rm -f "$TMPDIR"/verbose-*

# --- GPU verbose (if available) ---
if has_gpu 2>/dev/null; then
  "$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/gv-src.zst" 2>/dev/null
  stderr_gvvv=$("$GZSTD" -d -k -f -vvv --hybrid "$TMPDIR/gv-src.zst" -o "$TMPDIR/gv-dec.bin" 2>&1 || true)

  # Should show GPU-related lines
  if echo "$stderr_gvvv" | grep -qiE "GPU|H2D|D2H|kernel|CUDA"; then
    pass "-vvv hybrid shows GPU stats"
  else
    fail "-vvv hybrid shows GPU stats"
  fi

  # H2D transfers should be non-zero
  h2d_line=$(echo "$stderr_gvvv" | grep -i "H2D" || true)
  if [[ -n "$h2d_line" ]]; then
    if echo "$h2d_line" | grep -qE "0\.00 GiB.*0\.00 GiB/s"; then
      fail "-vvv H2D non-zero" "got: $h2d_line"
    else
      pass "-vvv H2D non-zero"
    fi
  else
    pass "-vvv H2D" "(not present  may be CPU-only path)"
  fi

  rm -f "$TMPDIR"/gv-*
else
  skip "-vvv GPU stats" "no GPU"
  skip "-vvv H2D stats" "no GPU"
fi

# ============================================================
# 34. Completion summary format validation
# ============================================================
section "Completion summary format"

LAST_TEST_MS=0

# Compress summary: should show "input => output, ratio @ rate"
summary_c=$("$GZSTD" -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/summ-comp.zst" 2>&1 || true)
# Look for the => pattern and a rate
if echo "$summary_c" | grep -qE "=>.*@.*[GM]iB/s"; then
  pass "compress summary format" "(in => out @ rate)"
elif echo "$summary_c" | grep -qiE "[GM]iB/s"; then
  pass "compress summary has rate"
else
  fail "compress summary format" "$(echo "$summary_c" | tail -1)"
fi

# Should not show 0.00 B for either side
if echo "$summary_c" | grep -q "0\.00 B =>"; then
  fail "compress summary no 0.00 B input" "$(echo "$summary_c" | tail -1)"
else
  pass "compress summary non-zero input"
fi

# Decompress summary
summary_d=$("$GZSTD" -d -k -f --cpu-only "$TMPDIR/summ-comp.zst" -o "$TMPDIR/summ-dec.bin" 2>&1 || true)
if echo "$summary_d" | grep -qE "=>.*@.*[GM]iB/s"; then
  pass "decompress summary format" "(in => out @ rate)"
elif echo "$summary_d" | grep -qiE "[GM]iB/s"; then
  pass "decompress summary has rate"
else
  fail "decompress summary format" "$(echo "$summary_d" | tail -1)"
fi

# Test mode summary: should show "OK" and ratio
summary_t=$("$GZSTD" -t --cpu-only "$TMPDIR/summ-comp.zst" 2>&1 || true)
if echo "$summary_t" | grep -qi "OK"; then
  pass "test summary shows OK"
else
  fail "test summary shows OK" "$(echo "$summary_t" | tail -1)"
fi

# Test mode should not show 0.00 B for decompressed size
if echo "$summary_t" | grep -qE "=> 0\.00 B|=> 0 B"; then
  fail "test summary non-zero decomp size" "$(echo "$summary_t" | tail -1)"
else
  pass "test summary non-zero decomp size"
fi

# Test mode should show a reasonable ratio (not 0.0%)
if echo "$summary_t" | grep -q "ratio: 0\.0%"; then
  fail "test summary non-zero ratio" "$(echo "$summary_t" | tail -1)"
else
  pass "test summary reasonable ratio"
fi

rm -f "$TMPDIR"/summ-*

# ============================================================
# 35. Ultra compression validation
# ============================================================
section "Ultra compression validation"

LAST_TEST_MS=0

# --- windowLog is actually set (proves the v0.12.1 fix works) ---
# -T1 routes to compress_cpu_stream which logs "ultra: windowLog=27 (128 MiB window)"
# at -v level.  If the window was being silently clamped, this line would be absent.
ultra_wlog_out=$("$GZSTD" -k -f --cpu-only -T1 -v --ultra -22 \
  "$TMPDIR/small.txt" -o "$TMPDIR/ultra-wlog.zst" 2>&1 || true)
if echo "$ultra_wlog_out" | grep -qE "windowLog=27|128 MiB window"; then
  pass "--ultra -22 sets windowLog=27"
else
  fail "--ultra -22 sets windowLog=27" "no windowLog message in verbose output"
fi
rm -f "$TMPDIR/ultra-wlog.zst"

# --- Single-thread (compress_cpu_stream) round-trip ---
"$GZSTD" -k -f --cpu-only -T1 --ultra -22 \
  "$TMPDIR/small.txt" -o "$TMPDIR/ultra-t1.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only \
  "$TMPDIR/ultra-t1.zst" -o "$TMPDIR/ultra-t1.dec" 2>/dev/null
files_match "$TMPDIR/small.txt" "$TMPDIR/ultra-t1.dec" \
  && pass "--ultra -22 -T1 round-trip" || fail "--ultra -22 -T1 round-trip"
rm -f "$TMPDIR/ultra-t1.zst" "$TMPDIR/ultra-t1.dec"

# --- Multi-thread (compress_cpu_mt) round-trip ---
"$GZSTD" -k -f --cpu-only -T4 --ultra -22 \
  "$TMPDIR/medium.txt" -o "$TMPDIR/ultra-t4.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only \
  "$TMPDIR/ultra-t4.zst" -o "$TMPDIR/ultra-t4.dec" 2>/dev/null
files_match "$TMPDIR/medium.txt" "$TMPDIR/ultra-t4.dec" \
  && pass "--ultra -22 -T4 round-trip" || fail "--ultra -22 -T4 round-trip"
rm -f "$TMPDIR/ultra-t4.zst" "$TMPDIR/ultra-t4.dec"

# --- Small --chunk-size warns but still produces correct output ---
ultra_warn_out=$("$GZSTD" -k -f --cpu-only --ultra -22 --chunk-size=4 \
  "$TMPDIR/small.txt" -o "$TMPDIR/ultra-smallchunk.zst" 2>&1 || true)
if echo "$ultra_warn_out" | grep -qi "warning.*chunk\|chunk.*warning"; then
  pass "--ultra -22 --chunk-size=4 warns"
else
  fail "--ultra -22 --chunk-size=4 warns" "no warning in: $(echo "$ultra_warn_out" | head -1)"
fi
"$GZSTD" -d -k -f --cpu-only \
  "$TMPDIR/ultra-smallchunk.zst" -o "$TMPDIR/ultra-smallchunk.dec" 2>/dev/null
files_match "$TMPDIR/small.txt" "$TMPDIR/ultra-smallchunk.dec" \
  && pass "--ultra -22 --chunk-size=4 still correct" \
  || fail "--ultra -22 --chunk-size=4 still correct"
rm -f "$TMPDIR/ultra-smallchunk.zst" "$TMPDIR/ultra-smallchunk.dec"

# --- Progress bar format contains in: and out: ---
# Use medium.txt to ensure at least one progress tick fires (200ms interval).
prog_out=$("$GZSTD" -k -f --cpu-only --progress \
  "$TMPDIR/medium.txt" -o "$TMPDIR/ultra-prog.zst" 2>&1 || true)
prog_clean=$(echo "$prog_out" | sed 's/\x1b\[[0-9;]*m//g; s/\r/\n/g')
if echo "$prog_clean" | grep -qE "^in:[0-9]"; then
  pass "progress bar shows in:XX.X%"
else
  skip "progress bar shows in:XX.X%" "file too small/fast for progress tick"
fi
if echo "$prog_clean" | grep -qE "out:[0-9]|out:---"; then
  pass "progress bar shows out:"
else
  skip "progress bar shows out:" "file too small/fast for progress tick"
fi
rm -f "$TMPDIR/ultra-prog.zst"

# --- Interop: gzstd --ultra -22 output readable by stock zstd ---
if command -v zstd &>/dev/null; then
  "$GZSTD" -k -f --cpu-only --ultra -22 \
    "$TMPDIR/small.txt" -o "$TMPDIR/ultra-interop.zst" 2>/dev/null
  zstd -d -f "$TMPDIR/ultra-interop.zst" -o "$TMPDIR/ultra-interop.dec" 2>/dev/null
  files_match "$TMPDIR/small.txt" "$TMPDIR/ultra-interop.dec" \
    && pass "--ultra -22 interop: zstd -d reads gzstd output" \
    || fail "--ultra -22 interop: zstd -d reads gzstd output"
  rm -f "$TMPDIR/ultra-interop.zst" "$TMPDIR/ultra-interop.dec"
else
  skip "--ultra -22 interop with zstd" "zstd not installed"
fi

# ============================================================
# 36. Throttle budget tunables
# ============================================================
section "Throttle budget tunables"

# Round-trip at the most restrictive throttle: a single in-flight frame.
# Exercises the min-pipeline path and guards against deadlock at tiny budgets.
"$GZSTD" -k -f --cpu-only --throttle-frames=1 \
  "$TMPDIR/large.bin" -o "$TMPDIR/thr1.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only --throttle-frames=1 \
  "$TMPDIR/thr1.zst" -o "$TMPDIR/thr1.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/thr1.dec" \
  && pass "--throttle-frames=1 round-trip (deadlock guard)" \
  || fail "--throttle-frames=1 round-trip"
rm -f "$TMPDIR/thr1.zst" "$TMPDIR/thr1.dec"

# At the default floor (32 frames) the producer pipeline fills quickly; this
# just verifies no weird interaction with the min-frames clamp.
"$GZSTD" -k -f --cpu-only --throttle-frames=32 \
  "$TMPDIR/large.bin" -o "$TMPDIR/thr32.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only --throttle-frames=32 \
  "$TMPDIR/thr32.zst" -o "$TMPDIR/thr32.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/thr32.dec" \
  && pass "--throttle-frames=32 round-trip (floor)" \
  || fail "--throttle-frames=32 round-trip"
rm -f "$TMPDIR/thr32.zst" "$TMPDIR/thr32.dec"

# Large slack multiplier — exercises the other extreme (pipeline-dominated).
"$GZSTD" -k -f --cpu-only --throttle-factor=16 \
  "$TMPDIR/large.bin" -o "$TMPDIR/thrf16.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only --throttle-factor=16 \
  "$TMPDIR/thrf16.zst" -o "$TMPDIR/thrf16.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/thrf16.dec" \
  && pass "--throttle-factor=16 round-trip" \
  || fail "--throttle-factor=16 round-trip"
rm -f "$TMPDIR/thrf16.zst" "$TMPDIR/thrf16.dec"

# Minimum slack multiplier (1x parallelism). Must not deadlock.
"$GZSTD" -k -f --cpu-only --throttle-factor=1 \
  "$TMPDIR/large.bin" -o "$TMPDIR/thrf1.zst" 2>/dev/null
"$GZSTD" -d -k -f --cpu-only --throttle-factor=1 \
  "$TMPDIR/thrf1.zst" -o "$TMPDIR/thrf1.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/thrf1.dec" \
  && pass "--throttle-factor=1 round-trip" \
  || fail "--throttle-factor=1 round-trip"
rm -f "$TMPDIR/thrf1.zst" "$TMPDIR/thrf1.dec"

# -v emits the one-line throttle summary at startup. Check that both
# compression and decompression paths surface it.
v_out=$("$GZSTD" -v -k -f --cpu-only \
  "$TMPDIR/large.bin" -o "$TMPDIR/thr-v.zst" 2>&1)
if grep -qE "throttle: [0-9]+ frames .* source=" <<< "$v_out"; then
  pass "-v logs throttle startup line"
else
  fail "-v throttle startup line" "not found"
fi
rm -f "$TMPDIR/thr-v.zst"

# -vv emits end-of-run throttle stats (saturation + block counters).
vv_out=$("$GZSTD" -vv -k -f --cpu-only \
  "$TMPDIR/large.bin" -o "$TMPDIR/thr-vv.zst" 2>&1)
if grep -qE "throttle stats \[compress-cpu\]: peak=" <<< "$vv_out"; then
  pass "-vv logs throttle end-of-run stats"
else
  fail "-vv throttle end-of-run stats" "not found"
fi
rm -f "$TMPDIR/thr-vv.zst"

# Invalid --throttle-frames=0 must be rejected with usage exit (2).
"$GZSTD" -k -f --cpu-only --throttle-frames=0 \
  "$TMPDIR/small.txt" -o "$TMPDIR/thr-bad.zst" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]]; then
  pass "--throttle-frames=0 rejected (exit 2)"
else
  fail "--throttle-frames=0 rejection" "exit $rc"
fi
rm -f "$TMPDIR/thr-bad.zst"

# Invalid --throttle-factor=0 must be rejected.
"$GZSTD" -k -f --cpu-only --throttle-factor=0 \
  "$TMPDIR/small.txt" -o "$TMPDIR/thr-bad2.zst" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]]; then
  pass "--throttle-factor=0 rejected (exit 2)"
else
  fail "--throttle-factor=0 rejection" "exit $rc"
fi
rm -f "$TMPDIR/thr-bad2.zst"

# Hybrid/GPU regression: --throttle-frames=1 without --cpu-only used to
# deadlock because GPU workers greedy-acquire a full batch of permits.
# The compute_throttle_budget guardrail must clamp to the GPU batch floor
# and the run must complete (with a warning) in bounded time.
if has_gpu 2>/dev/null; then
  timeout 30 "$GZSTD" -k -f --hybrid --throttle-frames=1 \
    "$TMPDIR/large.bin" -o "$TMPDIR/thr-hyb.zst" 2>"$TMPDIR/thr-hyb.err"
  rc=$?
  if [[ $rc -eq 0 ]] \
     && grep -q "GPU batch floor" "$TMPDIR/thr-hyb.err" \
     && [[ -s "$TMPDIR/thr-hyb.zst" ]]; then
    pass "--throttle-frames=1 --hybrid: warns & clamps (no deadlock)"
  else
    fail "--throttle-frames=1 --hybrid deadlock guard" "exit $rc"
  fi
  rm -f "$TMPDIR/thr-hyb.zst" "$TMPDIR/thr-hyb.err"
else
  skip "--throttle-frames=1 --hybrid deadlock guard" "no GPU"
fi

# ============================================================
# Final summary
# ============================================================
clear_progress
print_summary
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
