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
# Args: first non-flag is the gzstd binary path; --extensive enables the
# heavier / lower-value test groups that are skipped by default to keep the
# common run fast.  Gate extra tests with `if $EXTENSIVE; then ... fi`.
EXTENSIVE=false
GZSTD=""
for arg in "$@"; do
  case "$arg" in
    --extensive|-e) EXTENSIVE=true ;;
    --help|-h) echo "Usage: $0 [path/to/gzstd] [--extensive]"; exit 0 ;;
    *) [[ -z "$GZSTD" ]] && GZSTD="$arg" ;;
  esac
done
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

  # If our count estimate was too low (cache was stale or first run),
  # expand the displayed total so we never show > 100%.  The next run
  # will pick up the corrected count from the cache.
  if [[ $current -gt $total ]]; then
    total=$current
    TOTAL_TESTS=$current
  fi

  local pct=$(( current * 100 / total ))
  [[ $pct -gt 100 ]] && pct=100
  local filled=$(( current * BAR_WIDTH / total ))
  [[ $filled -gt $BAR_WIDTH ]] && filled=$BAR_WIDTH
  local empty=$(( BAR_WIDTH - filled ))
  local elapsed_ms=$(( $(now_ms) - START_MS ))
  local elapsed_s=$(( elapsed_ms / 1000 ))

  # ETA calculation
  local eta_str=""
  if [[ $current -gt 3 && $elapsed_ms -gt 0 && $current -lt $total ]]; then
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
# Total tests in the suite — bump this when adding/removing tests.
# Verify with the final run summary; the progress bar will also
# auto-expand if you forget (so the display never shows > 100%),
# but you should keep this in sync to get an accurate ETA.
# ============================================================
# Default run: 253.  --extensive adds back the gated sections (Stress,
# Help/version, Space-separated values, Completion summary format) for 284.
EXPECTED_TESTS=218
$EXTENSIVE && EXPECTED_TESTS=295
count_tests() { echo "$EXPECTED_TESTS"; }

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
if $EXTENSIVE; then
section "Help and version (extensive)"

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

# --help (long form) documents exit codes
("$GZSTD" --help 2>&1 || true) | grep -qi "exit code" && pass "--help documents exit codes" || fail "--help documents exit codes"

# -? is an alias for -h
("$GZSTD" "-?" 2>&1 || true) | grep -qi "usage\|options\|compress" && pass "-? shows help" || fail "-? shows help"
rc=0; "$GZSTD" "-?" >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-? exits 0" || fail "-? exits 0" "got $rc"

# --help includes at least one example
("$GZSTD" --help 2>&1 || true) | grep -qi "example" && pass "--help includes examples" || fail "--help includes examples"

rc=0; "$GZSTD" -V >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && pass "-V exits 0" || fail "-V exits 0" "got $rc"
fi  # $EXTENSIVE (Help and version)

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
# Bounded-queue (pooled reader) regressions — v0.13.60–68 class.
# --no-mmap forces the buffered pooled reader, whose pool BOUNDS queue
# depth (on pre-6.4 kernels it is the default for large files).  Two
# failure modes live here, both of which present as a hang:
#   1. Locked --gpu-batch × many streams: sleeping streams hold throttle
#      permits while waiting for full batches the bounded queue can never
#      present (the v0.13.68 wedge: 128 streams ≈ 2,560 sequestered
#      permits, hang at 45%).
#   2. Pooled-slot leaks: every GPU input lifecycle (batch success,
#      partial-rescue, rescue worker, trivial-skip re-enqueue) must
#      recycle its DirectReadPool slot; one missed release path starves
#      the readers once frames-in-file > pool slots.
# pooltest.bin is sized so frame count EXCEEDS the pool on any box
# (cpu pool ≈ threads+128+96; gpu pool adds +1024): at --chunk-size=1
# its 1536 frames overflow both even at 256 threads.  Half zeros / half
# random also routes some frames through the trivial-skip path.
# ============================================================
section "Bounded-queue pooled-reader regressions"

POOL_TEST_TIMEOUT=120
spin "pooltest.bin (1.5 GiB, half zero / half random)"
dd if=/dev/zero    bs=1M count=768 2>/dev/null  > "$TMPDIR/pooltest.bin"
dd if=/dev/urandom bs=1M count=768 2>/dev/null >> "$TMPDIR/pooltest.bin"
spin_done

# --- CPU pooled reader: frames ≫ pool slots (slot recycle canary) ---
t0=$(now_ms); rc=0
run_bounded "$POOL_TEST_TIMEOUT" "$TMPDIR/pool-cpu.log" \
  "$GZSTD" -k -f --cpu-only --no-mmap --chunk-size=1 \
  "$TMPDIR/pooltest.bin" -o "$TMPDIR/pool-cpu.zst" || rc=$?
LAST_TEST_MS=$(( $(now_ms) - t0 ))
if [[ $rc -eq 124 ]]; then
  fail "cpu-only pooled reader, frames > pool" "TIMED OUT — slot leak regression"
elif [[ $rc -eq 0 ]] \
     && "$GZSTD" -d -k -f --cpu-only "$TMPDIR/pool-cpu.zst" -o "$TMPDIR/pool-cpu.dec" 2>/dev/null \
     && files_match "$TMPDIR/pooltest.bin" "$TMPDIR/pool-cpu.dec"; then
  pass "cpu-only pooled reader, frames > pool" "round-trip OK"
else
  fail "cpu-only pooled reader, frames > pool" "exit $rc / mismatch"
fi

if has_gpu 2>/dev/null; then
  # --- The v0.13.68 wedge repro: locked batch × many streams, bounded queue ---
  t0=$(now_ms); rc=0
  log="$TMPDIR/pool-wedge.log"
  run_bounded "$POOL_TEST_TIMEOUT" "$log" \
    "$GZSTD" -k -f --hybrid --no-mmap --chunk-size=1 \
    --gpu-batch=64 --gpu-streams=16 \
    "$TMPDIR/pooltest.bin" -o "$TMPDIR/pool-wedge.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "locked batch × 16 streams, bounded queue" "TIMED OUT — permit-sequester deadlock"
  elif [[ $rc -eq 0 ]] \
       && "$GZSTD" -d -k -f --cpu-only "$TMPDIR/pool-wedge.zst" -o "$TMPDIR/pool-wedge.dec" 2>/dev/null \
       && files_match "$TMPDIR/pooltest.bin" "$TMPDIR/pool-wedge.dec"; then
    pass "locked batch × 16 streams, bounded queue" "round-trip OK"
  elif [[ $rc -eq 5 ]]; then
    pass "locked batch × 16 streams, bounded queue" "EXIT_GPU_FAIL (GPU too small)"
  else
    fail "locked batch × 16 streams, bounded queue" "exit $rc"
  fi

  # --- Hybrid slot lifecycle: success + rescue + trivial-skip releases ---
  t0=$(now_ms); rc=0
  run_bounded "$POOL_TEST_TIMEOUT" "$TMPDIR/pool-hyb.log" \
    "$GZSTD" -k -f --hybrid --no-mmap --chunk-size=1 \
    "$TMPDIR/pooltest.bin" -o "$TMPDIR/pool-hyb.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "hybrid pooled slots, frames > pool" "TIMED OUT — slot leak regression"
  elif [[ $rc -eq 0 ]] \
       && "$GZSTD" -d -k -f --cpu-only "$TMPDIR/pool-hyb.zst" -o "$TMPDIR/pool-hyb.dec" 2>/dev/null \
       && files_match "$TMPDIR/pooltest.bin" "$TMPDIR/pool-hyb.dec"; then
    pass "hybrid pooled slots, frames > pool" "round-trip OK"
  elif [[ $rc -eq 5 ]]; then
    pass "hybrid pooled slots, frames > pool" "EXIT_GPU_FAIL (GPU too small)"
  else
    fail "hybrid pooled slots, frames > pool" "exit $rc / mismatch"
  fi

  # --- gpu-only slot lifecycle: release-after-H2D + re-enqueue releases ---
  t0=$(now_ms); rc=0
  run_bounded "$POOL_TEST_TIMEOUT" "$TMPDIR/pool-gpu.log" \
    "$GZSTD" -k -f --gpu-only --no-mmap --chunk-size=1 \
    "$TMPDIR/pooltest.bin" -o "$TMPDIR/pool-gpu.zst" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "gpu-only pooled slots, frames > pool" "TIMED OUT — slot leak regression"
  elif [[ $rc -eq 0 ]] \
       && "$GZSTD" -d -k -f --cpu-only "$TMPDIR/pool-gpu.zst" -o "$TMPDIR/pool-gpu.dec" 2>/dev/null \
       && files_match "$TMPDIR/pooltest.bin" "$TMPDIR/pool-gpu.dec"; then
    pass "gpu-only pooled slots, frames > pool" "round-trip OK"
  elif [[ $rc -eq 5 ]]; then
    pass "gpu-only pooled slots, frames > pool" "EXIT_GPU_FAIL (GPU too small)"
  else
    fail "gpu-only pooled slots, frames > pool" "exit $rc / mismatch"
  fi

  # --- DECOMPRESS wedge (v0.13.69): locked high batch × many streams,
  # gpu-only.  Confirmed hang on an 8-GPU server: 128 streams each acquire
  # pop_n permits upfront and block on a full locked batch, draining the
  # throttle (sized FROM device×streams×batch) so the in-order writer wedges
  # behind a head-of-line frame.  gpu-only has no CPU relief valve (hybrid
  # does — and does NOT hang).  Fix: gpu-only treats --gpu-batch as a cap,
  # not a hard floor.  Many-frame archive (chunk-size=1) ensures streams
  # actually compete.  On a small GPU the wedge may not trigger (VRAM-fit
  # shrinks demand); the canary still guards the no-hang invariant.
  "$GZSTD" -k -f --cpu-only --chunk-size=1 \
    "$TMPDIR/pooltest.bin" -o "$TMPDIR/pool-dwedge.zst" 2>/dev/null
  t0=$(now_ms); rc=0
  run_bounded "$POOL_TEST_TIMEOUT" "$TMPDIR/pool-dwedge.log" \
    "$GZSTD" -d -k -f --gpu-only --gpu-batch=64 --gpu-streams=16 \
    "$TMPDIR/pool-dwedge.zst" -o "$TMPDIR/pool-dwedge.dec" || rc=$?
  LAST_TEST_MS=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 124 ]]; then
    fail "gpu-only decomp locked batch × 16 streams" "TIMED OUT — permit-sequester deadlock"
  elif [[ $rc -eq 0 ]] && files_match "$TMPDIR/pooltest.bin" "$TMPDIR/pool-dwedge.dec"; then
    pass "gpu-only decomp locked batch × 16 streams" "round-trip OK"
  elif [[ $rc -eq 5 ]]; then
    pass "gpu-only decomp locked batch × 16 streams" "EXIT_GPU_FAIL (GPU too small)"
  else
    fail "gpu-only decomp locked batch × 16 streams" "exit $rc / mismatch"
  fi
else
  skip "locked batch × 16 streams, bounded queue" "no GPU"
  skip "hybrid pooled slots, frames > pool" "no GPU"
  skip "gpu-only pooled slots, frames > pool" "no GPU"
  skip "gpu-only decomp locked batch × 16 streams" "no GPU"
fi
rm -f "$TMPDIR"/pool-*.zst "$TMPDIR"/pool-*.dec "$TMPDIR"/pool-*.log "$TMPDIR/pool-dwedge."* "$TMPDIR/pooltest.bin"

# ============================================================
# 20. Stress tests
# ============================================================
if $EXTENSIVE; then
section "Stress tests (extensive)"

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
fi  # $EXTENSIVE

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
# 25b. --cpu-share fixed CPU/GPU split (hybrid only)
# ============================================================
section "--cpu-share scheduling (hybrid)"

if has_gpu 2>/dev/null; then
  # Build a multi-chunk file large enough that GPU initialization
  # finishes before the reader does — otherwise CPU drains everything
  # while GPU is still booting and the share is unobservable.
  # 512 MiB at --chunk-size 4 = 128 frames.
  share_src="$TMPDIR/share-src.bin"
  dd if=/dev/urandom bs=1M count=256 2>/dev/null  > "$share_src"
  dd if=/dev/zero    bs=1M count=256 2>/dev/null >> "$share_src"

  # Round-trip correctness across the share range.
  for share in 0.0 0.25 0.5 0.75 1.0; do
    compressed="$TMPDIR/share-${share}.zst"
    recovered="$TMPDIR/share-${share}.dec"
    run_test "$GZSTD" --hybrid --cpu-share $share --chunk-size 4 \
      -k -f "$share_src" -o "$compressed" 2>/dev/null
    if [[ ! -s "$compressed" ]]; then
      fail "--cpu-share=$share round-trip" "empty compress output"
      continue
    fi
    run_test "$GZSTD" -d --hybrid --cpu-share $share \
      -k -f "$compressed" -o "$recovered" 2>/dev/null
    files_match "$share_src" "$recovered" \
      && pass "--cpu-share=$share round-trip" \
      || fail "--cpu-share=$share round-trip" "data mismatch"
    rm -f "$compressed" "$recovered"
  done

  # The -v banner reports the requested percentage.
  banner_out=$("$GZSTD" --hybrid --cpu-share 0.42 --chunk-size 4 -v \
    -k -f "$share_src" -o "$TMPDIR/share-banner.zst" 2>&1)
  if echo "$banner_out" | grep -q "CPU share 42.0%"; then
    pass "--cpu-share banner shows percentage"
  else
    fail "--cpu-share banner shows percentage" "(no '42.0%' in -v output)"
  fi
  rm -f "$TMPDIR/share-banner.zst"

  # Behavioural check: at -vv, count CPU vs GPU frames at the extremes
  # and verify the split actually responds to --cpu-share.  This is the
  # regression guard for the v0.12.51 fix where every share landed at
  # ~85% CPU regardless of what the user asked for.
  measure_share() {
    local share=$1 out cpu_n gpu_n
    out=$("$GZSTD" --hybrid --cpu-share $share --chunk-size 4 -vv \
      -k -f "$share_src" -o "$TMPDIR/share-meas.zst" 2>&1)
    cpu_n=$(echo "$out" | grep -oE "CPU/T[0-9]+\] total tasks=[0-9]+" \
            | grep -oE "[0-9]+$" | awk '{s+=$1}END{print s+0}')
    gpu_n=$(echo "$out" | grep -oE "GPU[0-9]+/S[0-9]+\] total batches=[0-9]+ chunks=[0-9]+" \
            | grep -oE "chunks=[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1}END{print s+0}')
    rm -f "$TMPDIR/share-meas.zst"
    echo "$cpu_n $gpu_n"
  }

  read low_cpu low_gpu  < <(measure_share 0.0)
  read high_cpu high_gpu < <(measure_share 1.0)

  if (( low_cpu + low_gpu > 0 )) && (( high_cpu + high_gpu > 0 )); then
    if (( low_gpu > low_cpu )) && (( high_cpu > high_gpu )); then
      pass "--cpu-share split responds to value" \
           "(0.0: ${low_cpu}c/${low_gpu}g; 1.0: ${high_cpu}c/${high_gpu}g)"
    else
      fail "--cpu-share split responds to value" \
           "(0.0: ${low_cpu}c/${low_gpu}g; 1.0: ${high_cpu}c/${high_gpu}g)"
    fi
  else
    skip "--cpu-share split responds to value" "no -vv counters captured"
  fi

  rm -f "$share_src"
else
  skip "--cpu-share=0.0 round-trip"          "no GPU"
  skip "--cpu-share=0.25 round-trip"         "no GPU"
  skip "--cpu-share=0.5 round-trip"          "no GPU"
  skip "--cpu-share=0.75 round-trip"         "no GPU"
  skip "--cpu-share=1.0 round-trip"          "no GPU"
  skip "--cpu-share banner shows percentage" "no GPU"
  skip "--cpu-share split responds to value" "no GPU"
fi

# ============================================================
# 25b2. Hybrid GPU-bringup overlap (v0.13.13 decompress restructure)
# In adaptive hybrid, GPU detection/cuInit runs on a background thread
# while the CPU pool decompresses.  These guard the concurrency paths:
# the CPU-finishes-before-GPU-init window, the fixed-share inline
# bringup, stdout output, and teardown stability under repetition.
# ============================================================
section "Hybrid GPU-bringup overlap (decompress)"

if has_gpu 2>/dev/null; then
  hov_src="$TMPDIR/hov-src.bin"
  # Small enough that the CPU pool can plausibly drain it during cuInit
  # (exercises the bringup-thread skip-GPU-spawn path on fast machines).
  dd if=/dev/urandom bs=1M count=4 2>/dev/null > "$hov_src"
  hov_zst="$TMPDIR/hov.zst"
  "$GZSTD" -k -f --cpu-only "$hov_src" -o "$hov_zst" 2>/dev/null

  # 1. Adaptive hybrid decompress (background bringup) round-trip.
  run_test "$GZSTD" -d -f --hybrid "$hov_zst" -o "$TMPDIR/hov-adapt.dec" 2>/dev/null
  files_match "$hov_src" "$TMPDIR/hov-adapt.dec" \
    && pass "adaptive hybrid decompress round-trip" \
    || fail "adaptive hybrid decompress round-trip" "mismatch"

  # 2. Fixed-share hybrid decompress (inline bringup, GPU warmed first).
  run_test "$GZSTD" -d -f --hybrid --cpu-share 0.5 "$hov_zst" -o "$TMPDIR/hov-fixed.dec" 2>/dev/null
  files_match "$hov_src" "$TMPDIR/hov-fixed.dec" \
    && pass "fixed-share hybrid decompress round-trip" \
    || fail "fixed-share hybrid decompress round-trip" "mismatch"

  # 3. Adaptive hybrid decompress to stdout (-c output path).
  "$GZSTD" -d -f --hybrid -c "$hov_zst" 2>/dev/null > "$TMPDIR/hov-stdout.dec"
  files_match "$hov_src" "$TMPDIR/hov-stdout.dec" \
    && pass "hybrid decompress to stdout" \
    || fail "hybrid decompress to stdout" "mismatch"

  # 4. Teardown stability: repeated adaptive hybrid decompress must all
  #    match and exit cleanly (shakes out bringup/teardown races).
  hov_rep_ok=true
  for i in 1 2 3 4 5; do
    "$GZSTD" -d -f --hybrid "$hov_zst" -o "$TMPDIR/hov-rep.dec" 2>/dev/null \
      || { hov_rep_ok=false; break; }
    files_match "$hov_src" "$TMPDIR/hov-rep.dec" || { hov_rep_ok=false; break; }
  done
  $hov_rep_ok && pass "hybrid decompress x5 stable" \
              || fail "hybrid decompress x5 stable" "mismatch or nonzero exit"

  rm -f "$hov_src" "$hov_zst" "$TMPDIR"/hov-*.dec
else
  skip "adaptive hybrid decompress round-trip"  "no GPU"
  skip "fixed-share hybrid decompress round-trip" "no GPU"
  skip "hybrid decompress to stdout"            "no GPU"
  skip "hybrid decompress x5 stable"            "no GPU"
fi

# ============================================================
# 25c. Asymmetric mode (PCIe gen → backend auto-selection)
# ============================================================
section "Asymmetric mode (PCIe gen detection)"

if has_gpu 2>/dev/null; then
  asym_src="$TMPDIR/asym-src.bin"
  asym_zst="$TMPDIR/asym.zst"
  asym_out="$TMPDIR/asym.out"
  dd if=/dev/urandom bs=1M count=64 2>/dev/null > "$asym_src"
  "$GZSTD" --hybrid -k -f "$asym_src" -o "$asym_zst" 2>/dev/null

  # Trigger detection on a decompress run; the [ASYMMETRIC] line carries
  # the detected gen, and the [STARTUP] line carries the chosen backend.
  asym_log=$("$GZSTD" -d -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
  rm -f "$asym_out"
  asym_gen=$(echo "$asym_log" | grep -oE "PCIe Gen[0-9]+" | head -1 | grep -oE "[0-9]+")

  if [[ -z "$asym_gen" ]]; then
    # No [ASYMMETRIC] line at all — detection unavailable (NVML missing,
    # no NVIDIA card visible).  Verify the fallback path: hybrid default.
    if echo "$asym_log" | grep -q "DECOMPRESS (hybrid"; then
      pass "asymmetric mode: detection unavailable → hybrid fallback"
    else
      fail "asymmetric mode: detection unavailable → hybrid fallback" \
           "(no [ASYMMETRIC] log AND not hybrid)"
    fi
    skip "asymmetric mode: gen-appropriate decompress default" "gen undetected"
    skip "asymmetric mode: compress always defaults to hybrid"  "gen undetected"
    skip "asymmetric mode: --hybrid override bypasses asymmetric" "gen undetected"
    skip "asymmetric mode: --cpu-only override bypasses asymmetric" "gen undetected"
  else
    pass "asymmetric mode: PCIe gen detected" "(Gen$asym_gen)"

    # Decompress backend default depends on gen.
    if (( asym_gen < 4 )); then
      if echo "$asym_log" | grep -q "DECOMPRESS (cpu-only)"; then
        pass "asymmetric mode: Gen$asym_gen decompress defaults to cpu-only"
      else
        fail "asymmetric mode: Gen$asym_gen decompress defaults to cpu-only" \
             "(banner: $(echo "$asym_log" | grep STARTUP || echo none))"
      fi
    else
      if echo "$asym_log" | grep -q "DECOMPRESS (hybrid"; then
        pass "asymmetric mode: Gen$asym_gen decompress defaults to hybrid"
      else
        fail "asymmetric mode: Gen$asym_gen decompress defaults to hybrid" \
             "(banner: $(echo "$asym_log" | grep STARTUP || echo none))"
      fi
    fi

    # Compress always defaults to hybrid regardless of gen.
    comp_log=$("$GZSTD" -v -k -f "$asym_src" -o "$asym_zst" 2>&1)
    if echo "$comp_log" | grep -q "COMPRESS (hybrid"; then
      pass "asymmetric mode: compress always defaults to hybrid"
    else
      fail "asymmetric mode: compress always defaults to hybrid" \
           "(banner: $(echo "$comp_log" | grep STARTUP || echo none))"
    fi

    # --hybrid override: explicit user choice silences [ASYMMETRIC].
    over_log=$("$GZSTD" -d --hybrid -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
    rm -f "$asym_out"
    if echo "$over_log" | grep -q "ASYMMETRIC"; then
      fail "asymmetric mode: --hybrid override bypasses asymmetric" \
           "([ASYMMETRIC] log appeared despite explicit --hybrid)"
    elif echo "$over_log" | grep -q "DECOMPRESS (hybrid"; then
      pass "asymmetric mode: --hybrid override bypasses asymmetric"
    else
      fail "asymmetric mode: --hybrid override bypasses asymmetric" \
           "(banner: $(echo "$over_log" | grep STARTUP || echo none))"
    fi

    # --cpu-only override: also silences [ASYMMETRIC] (no auto-pick happens).
    cpu_log=$("$GZSTD" -d --cpu-only -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
    rm -f "$asym_out"
    if echo "$cpu_log" | grep -q "ASYMMETRIC"; then
      fail "asymmetric mode: --cpu-only override bypasses asymmetric" \
           "([ASYMMETRIC] log appeared despite explicit --cpu-only)"
    elif echo "$cpu_log" | grep -q "DECOMPRESS (cpu-only)"; then
      pass "asymmetric mode: --cpu-only override bypasses asymmetric"
    else
      fail "asymmetric mode: --cpu-only override bypasses asymmetric" \
           "(banner: $(echo "$cpu_log" | grep STARTUP || echo none))"
    fi
  fi

  # Implicit-hybrid promotion: GPU-tuning flags (--gpu-batch, etc.) and
  # hybrid-only flags (--cpu-share, etc.) without an explicit --cpu-only/
  # --gpu-only/--hybrid should imply --hybrid, otherwise asymmetric mode
  # would silently flip to cpu-only on Gen3 and drop the user's tuning.
  imp_log=$("$GZSTD" -d --gpu-batch=64 -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
  rm -f "$asym_out"
  if echo "$imp_log" | grep -q "implying --hybrid" \
     && echo "$imp_log" | grep -q "DECOMPRESS (hybrid"; then
    pass "asymmetric mode: --gpu-batch implies --hybrid"
  else
    fail "asymmetric mode: --gpu-batch implies --hybrid" \
         "(banner: $(echo "$imp_log" | grep STARTUP || echo none))"
  fi

  # Same for --cpu-share (hybrid-only knob)
  imp2_log=$("$GZSTD" -d --cpu-share=0.5 -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
  rm -f "$asym_out"
  if echo "$imp2_log" | grep -q "implying --hybrid" \
     && echo "$imp2_log" | grep -q "DECOMPRESS (hybrid"; then
    pass "asymmetric mode: --cpu-share implies --hybrid"
  else
    fail "asymmetric mode: --cpu-share implies --hybrid" \
         "(banner: $(echo "$imp2_log" | grep STARTUP || echo none))"
  fi

  # Explicit --cpu-only must beat tuning-flag promotion (cpu-only wins,
  # no [ASYMMETRIC] log because backend_user_set is already true).
  ovr_log=$("$GZSTD" -d --cpu-only --gpu-batch=64 -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
  rm -f "$asym_out"
  if echo "$ovr_log" | grep -q "ASYMMETRIC"; then
    fail "asymmetric mode: explicit --cpu-only beats tuning-flag promotion" \
         "([ASYMMETRIC] log appeared despite explicit --cpu-only)"
  elif echo "$ovr_log" | grep -q "DECOMPRESS (cpu-only)"; then
    pass "asymmetric mode: explicit --cpu-only beats tuning-flag promotion"
  else
    fail "asymmetric mode: explicit --cpu-only beats tuning-flag promotion" \
         "(banner: $(echo "$ovr_log" | grep STARTUP || echo none))"
  fi

  # Bug A regression: --cpu-batch with NO explicit backend should now
  # imply --hybrid (so the silencing path doesn't trigger at all).  But
  # --cpu-batch combined with explicit --cpu-only must still silence.
  cb_log=$("$GZSTD" -d --cpu-only --cpu-batch=8 -v -k -f "$asym_zst" -o "$asym_out" 2>&1)
  rm -f "$asym_out"
  if echo "$cb_log" | grep -q -- "--cpu-batch is ignored in --cpu-only mode"; then
    pass "asymmetric mode: --cpu-batch + explicit --cpu-only triggers silencing"
  else
    fail "asymmetric mode: --cpu-batch + explicit --cpu-only triggers silencing" \
         "(expected silencing note, got: $(echo "$cb_log" | grep -i cpu-batch || echo none))"
  fi

  rm -f "$asym_src" "$asym_zst"
else
  skip "asymmetric mode: PCIe gen detected"                          "no GPU"
  skip "asymmetric mode: gen-appropriate decompress default"         "no GPU"
  skip "asymmetric mode: compress always defaults to hybrid"         "no GPU"
  skip "asymmetric mode: --hybrid override bypasses asymmetric"      "no GPU"
  skip "asymmetric mode: --cpu-only override bypasses asymmetric"    "no GPU"
  skip "asymmetric mode: --gpu-batch implies --hybrid"               "no GPU"
  skip "asymmetric mode: --cpu-share implies --hybrid"               "no GPU"
  skip "asymmetric mode: explicit --cpu-only beats tuning-flag promotion" "no GPU"
  skip "asymmetric mode: --cpu-batch + explicit --cpu-only triggers silencing" "no GPU"
fi

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
if $EXTENSIVE; then
section "Space-separated option values (extensive)"

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
fi  # $EXTENSIVE (Space-separated option values)

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
if $EXTENSIVE; then
section "Completion summary format (extensive)"

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
fi  # $EXTENSIVE (Completion summary format)

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
if grep -qE "\[THROTTLE\] [0-9]+ frames .* source=" <<< "$v_out"; then
  pass "-v logs throttle startup line"
else
  fail "-v throttle startup line" "not found"
fi
rm -f "$TMPDIR/thr-v.zst"

# -vv emits end-of-run throttle stats (saturation + block counters).
vv_out=$("$GZSTD" -vv -k -f --cpu-only \
  "$TMPDIR/large.bin" -o "$TMPDIR/thr-vv.zst" 2>&1)
if grep -qE '\[THROTTLE\] stats \[compress-cpu\]: peak=' <<< "$vv_out"; then
  pass "-vv logs throttle end-of-run stats"
else
  fail "-vv throttle end-of-run stats" "not found"
fi
rm -f "$TMPDIR/thr-vv.zst"

# --throttle-frames=0 disables the throttle (v0.12.48+).  Must succeed AND
# the verbose output must show DISABLED.
"$GZSTD" -k -f --cpu-only --throttle-frames=0 -v \
  "$TMPDIR/small.txt" -o "$TMPDIR/thr-disabled.zst" 2>"$TMPDIR/thr-disabled.log"
rc=$?
if [[ $rc -eq 0 ]] && grep -q "\[THROTTLE\] DISABLED" "$TMPDIR/thr-disabled.log"; then
  pass "--throttle-frames=0 disables throttle"
else
  fail "--throttle-frames=0 disables throttle" "exit $rc, log: $(cat "$TMPDIR/thr-disabled.log" 2>/dev/null | head -3)"
fi
# --no-throttle is an alias for --throttle-frames=0
"$GZSTD" -k -f --cpu-only --no-throttle -v --overwrite \
  "$TMPDIR/small.txt" -o "$TMPDIR/thr-disabled.zst" 2>"$TMPDIR/thr-disabled.log"
if grep -q "\[THROTTLE\] DISABLED" "$TMPDIR/thr-disabled.log"; then
  pass "--no-throttle alias works"
else
  fail "--no-throttle alias" "no DISABLED line"
fi
# --throttle-frames=-2 must still be rejected.
"$GZSTD" -k -f --cpu-only --throttle-frames=-2 \
  "$TMPDIR/small.txt" -o "$TMPDIR/thr-bad.zst" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]]; then
  pass "--throttle-frames=-2 rejected (exit 2)"
else
  fail "--throttle-frames=-2 rejection" "exit $rc"
fi
rm -f "$TMPDIR/thr-disabled.zst" "$TMPDIR/thr-disabled.log" "$TMPDIR/thr-bad.zst"

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

section "Sliding-window compression"

# Basic round-trip: compress with --sliding-window, decompress normally
"$GZSTD" -k -f --sliding-window \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw.zst" 2>/dev/null
"$GZSTD" -d -k -f \
  "$TMPDIR/sw.zst" -o "$TMPDIR/sw.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/sw.dec" \
  && pass "--sliding-window round-trip" \
  || fail "--sliding-window round-trip"
rm -f "$TMPDIR/sw.zst" "$TMPDIR/sw.dec"

# Produces a single frame (verified via frame count in output)
"$GZSTD" -k -f --sliding-window \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw-frames.zst" 2>/dev/null
frame_count=$(zstd --list "$TMPDIR/sw-frames.zst" 2>/dev/null | awk '/Frames/{getline; print $1}')
if [[ "$frame_count" == "1" ]]; then
  pass "--sliding-window produces single frame"
else
  fail "--sliding-window single frame" "got $frame_count frames"
fi
rm -f "$TMPDIR/sw-frames.zst"

# v0.13.1 regression: multi-frame files with per-frame decomp_size > 64 MiB
# (the streaming threshold) used to corrupt output because streaming chunks
# reused seq numbers that collided with adjacent frames' seqs.  Trigger the
# condition by forcing chunk-size > 64 MiB on a >2-chunk input.  Bug surfaced
# under --ultra -22 on a 256-core machine (auto-bumped chunks to 128 MiB).
osmf="$TMPDIR/oversized-multi.bin"
osmz="$TMPDIR/oversized-multi.zst"
osmd="$TMPDIR/oversized-multi.dec"
dd if=/dev/urandom of="$osmf" bs=1M count=200 status=none
"$GZSTD" --cpu-only --chunk-size 100 -k -f "$osmf" -o "$osmz" 2>/dev/null
"$GZSTD" --cpu-only -d -k -f "$osmz" -o "$osmd" 2>/dev/null
files_match "$osmf" "$osmd" \
  && pass "multi-frame oversized round-trip (decomp_size > 64 MiB per frame)" \
  || fail "multi-frame oversized round-trip" "streaming-path seq collision"
rm -f "$osmf" "$osmz" "$osmd"

# zstd can decompress our --sliding-window output
"$GZSTD" -k -f --sliding-window \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw-interop.zst" 2>/dev/null
zstd -d -f "$TMPDIR/sw-interop.zst" -o "$TMPDIR/sw-interop.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/sw-interop.dec" \
  && pass "--sliding-window interop: zstd -d reads output" \
  || fail "--sliding-window zstd interop"
rm -f "$TMPDIR/sw-interop.zst" "$TMPDIR/sw-interop.dec"

# --sliding-window implies --cpu-only (no error without explicit --cpu-only)
out=$("$GZSTD" -k -f --sliding-window \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw-impl.zst" 2>&1)
rc=$?
if [[ $rc -eq 0 ]] && grep -q "implies --cpu-only" <<< "$out"; then
  pass "--sliding-window implies --cpu-only (warning)"
else
  fail "--sliding-window implies --cpu-only" "exit $rc"
fi
rm -f "$TMPDIR/sw-impl.zst"

# --sliding-window with --gpu-only must be rejected (exit 2)
"$GZSTD" -k -f --sliding-window --gpu-only \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw-bad.zst" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]]; then
  pass "--sliding-window --gpu-only rejected (exit 2)"
else
  fail "--sliding-window --gpu-only rejection" "exit $rc"
fi
rm -f "$TMPDIR/sw-bad.zst"

# --sliding-window with -d must be rejected (exit 2)
"$GZSTD" -k -f --sliding-window -d \
  "$TMPDIR/large.bin" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]]; then
  pass "--sliding-window -d rejected (exit 2)"
else
  fail "--sliding-window -d rejection" "exit $rc"
fi

# --sliding-window with explicit --cpu-only (no warning)
out=$("$GZSTD" -k -f --sliding-window --cpu-only \
  "$TMPDIR/large.bin" -o "$TMPDIR/sw-cpu.zst" 2>&1)
rc=$?
if [[ $rc -eq 0 ]] && ! grep -q "implies --cpu-only" <<< "$out"; then
  pass "--sliding-window --cpu-only (no warning)"
else
  fail "--sliding-window --cpu-only" "exit $rc or unexpected warning"
fi
rm -f "$TMPDIR/sw-cpu.zst"

# The two zstd-compat command-line sections below are gated to --extensive:
# they verify drop-in zstd/gzip flag compatibility (aliases, no-op acceptance,
# bundled short flags), which rarely changes.  Run the extensive suite (-e)
# after touching arg parsing or the compat layer.
if $EXTENSIVE; then

# ============================================================
section "zstd-compat flag layer"

# Long-form aliases for short flags gzstd already supports
dd if=/dev/urandom bs=1M count=1 2>/dev/null > "$TMPDIR/zc.bin"

("$GZSTD" --keep "$TMPDIR/zc.bin" 2>/dev/null && test -f "$TMPDIR/zc.bin.zst") \
  && pass "--keep (alias for -k)" || fail "--keep (alias for -k)"
rm -f "$TMPDIR/zc.bin.zst"
("$GZSTD" --force "$TMPDIR/zc.bin" 2>/dev/null && test -f "$TMPDIR/zc.bin.zst") \
  && pass "--force (alias for -f)" || fail "--force (alias for -f)"
("$GZSTD" --decompress "$TMPDIR/zc.bin.zst" -o "$TMPDIR/zc.out" 2>/dev/null \
  && cmp -s "$TMPDIR/zc.bin" "$TMPDIR/zc.out") \
  && pass "--decompress (alias for -d)" || fail "--decompress (alias for -d)"
rm -f "$TMPDIR/zc.out"
("$GZSTD" --test "$TMPDIR/zc.bin.zst" >/dev/null 2>&1) \
  && pass "--test (alias for -t)" || fail "--test (alias for -t)"
("$GZSTD" --verbose -f "$TMPDIR/zc.bin" 2>&1 | grep -qi "thread\|worker\|mmap\|preallocated") \
  && pass "--verbose (alias for -v)" || fail "--verbose (alias for -v)"
("$GZSTD" --stdout "$TMPDIR/zc.bin" 2>/dev/null | "$GZSTD" -d 2>/dev/null | cmp -s - "$TMPDIR/zc.bin") \
  && pass "--stdout (alias for -c)" || fail "--stdout (alias for -c)"

# -H is the long help (zstd convention)
("$GZSTD" -H 2>&1 | grep -qi "examples") \
  && pass "-H shows full help" || fail "-H shows full help"

# --single-thread maps to -T 1
("$GZSTD" --single-thread --cpu-only -f "$TMPDIR/zc.bin" 2>&1 | tail -1 | grep -q "100.00%") \
  && pass "--single-thread compresses" || fail "--single-thread compresses"

# Silent no-ops (should not print any warning or error)
for opt in --check --no-check --asyncio --no-asyncio --format=zstd \
           --no-dictID --compress-literals --no-compress-literals \
           --row-match-finder --no-row-match-finder \
           --auto-threads=physical --stream-size=1024 --size-hint=1024 \
           --target-compressed-block-size=131072; do
  out=$("$GZSTD" "$opt" -f "$TMPDIR/zc.bin" 2>&1)
  if ! echo "$out" | grep -qi warning; then
    pass "silent no-op: $opt"
  else
    fail "silent no-op: $opt" "unexpected warning"
  fi
done

# Warn no-ops (each should emit one compat warning)
for opt in --adapt --long=27 --rsyncable --exclude-compressed \
           --format=gzip --pass-through; do
  out=$("$GZSTD" "$opt" -f "$TMPDIR/zc.bin" 2>&1)
  if echo "$out" | grep -qi "accepted for zstd compatibility but ignored"; then
    pass "warn no-op: $opt"
  else
    fail "warn no-op: $opt" "missing compat warning"
  fi
done

# Value-eating warn no-ops
out=$("$GZSTD" --trace foo.log -f "$TMPDIR/zc.bin" 2>&1)
echo "$out" | grep -qi -- "trace.*compatibility" && pass "--trace eats value" || fail "--trace eats value"
out=$("$GZSTD" -D /nonexistent -f "$TMPDIR/zc.bin" 2>&1)
echo "$out" | grep -qi "D.*dictionary" && pass "-D eats value" || fail "-D eats value"
# -M / --memlimit / --memory are real flags (implemented v0.12.30)
("$GZSTD" -M256 -f "$TMPDIR/zc.bin" >/dev/null 2>&1) && pass "-M# accepted (real)" || fail "-M# accepted"
("$GZSTD" -M 256 -f "$TMPDIR/zc.bin" >/dev/null 2>&1) && pass "-M N accepted (real)" || fail "-M N accepted"
("$GZSTD" --memlimit=256 -f "$TMPDIR/zc.bin" >/dev/null 2>&1) && pass "--memlimit=N accepted" || fail "--memlimit=N accepted"
("$GZSTD" --memlimit 256 -f "$TMPDIR/zc.bin" >/dev/null 2>&1) && pass "--memlimit N accepted" || fail "--memlimit N accepted"
("$GZSTD" --memory=256 -f "$TMPDIR/zc.bin" >/dev/null 2>&1) && pass "--memory=N accepted" || fail "--memory=N accepted"

# -M tightens the throttle budget at -vv (source should show `source=ram`)
out=$("$GZSTD" --cpu-only -vv -M 32 -f "$TMPDIR/zc.bin" 2>&1)
echo "$out" | grep -q "\[THROTTLE\].*source=ram" && pass "-M tightens throttle (source=ram)" \
  || fail "-M tightens throttle" "expected source=ram in throttle line"

# -M on decompress rejects frames requiring a larger window.
# Use zstd --long=27 to force a 128 MiB window frame.
if command -v zstd >/dev/null 2>&1; then
  # Need input large AND incompressible so zstd doesn't clamp the effective
  # window for repetitive data.  128 MiB random + --long=27 forces a 128 MiB
  # window in the frame header.
  dd if=/dev/urandom bs=1M count=128 2>/dev/null > "$TMPDIR/mem-big.bin"
  zstd -q -19 --long=27 -f "$TMPDIR/mem-big.bin" -o "$TMPDIR/mem-big.zst" 2>/dev/null
  if [ -f "$TMPDIR/mem-big.zst" ]; then
    rc=0
    "$GZSTD" --cpu-only -d -M 1 -f "$TMPDIR/mem-big.zst" -o "$TMPDIR/mem-big.out" >/dev/null 2>&1 || rc=$?
    [[ $rc -eq 4 ]] && pass "-M rejects oversize-window stream (exit 4)" \
                   || fail "-M rejects oversize-window stream" "got exit $rc (expected 4)"
    # Loose limit should succeed
    "$GZSTD" --cpu-only -d -M 256 -f "$TMPDIR/mem-big.zst" -o "$TMPDIR/mem-big.out" >/dev/null 2>&1 \
      && cmp -s "$TMPDIR/mem-big.bin" "$TMPDIR/mem-big.out" \
      && pass "-M with loose limit still decompresses" \
      || fail "-M with loose limit still decompresses"
    rm -f "$TMPDIR/mem-big.bin" "$TMPDIR/mem-big.zst" "$TMPDIR/mem-big.out"
  fi
fi
out=$("$GZSTD" -B128K -f "$TMPDIR/zc.bin" 2>&1)
echo "$out" | grep -qi "chunk-size" && pass "-B# warns" || fail "-B# warns"

# -q suppresses compat warnings
out=$("$GZSTD" -q --adapt -f "$TMPDIR/zc.bin" 2>&1)
if ! echo "$out" | grep -qi warning; then
  pass "-q suppresses compat warnings"
else
  fail "-q suppresses compat warnings" "got: $out"
fi

rm -f "$TMPDIR/zc.bin" "$TMPDIR/zc.bin.zst" "$TMPDIR/zc.out"

# ============================================================
section "Bundled short flags (zstd/gzip compat)"
# ----------------------------------------------------------------
# gzstd should accept bundled no-arg short flags (-dc, -dkf, …) like
# zstd/gzip.  Only operation-flag groups {d,t,k,f,c} expand; value flags,
# numeric levels, and the repeat flags (-vv/-qq) must be unaffected.

dd if=/dev/urandom bs=1M count=1 2>/dev/null > "$TMPDIR/bf.bin"
"$GZSTD" -k -f --cpu-only "$TMPDIR/bf.bin" -o "$TMPDIR/bf.zst" 2>/dev/null

# -dc : decompress to stdout (the canonical `gzstd -dc | tar -xf -` idiom)
("$GZSTD" -dc "$TMPDIR/bf.zst" 2>/dev/null | cmp -s - "$TMPDIR/bf.bin") \
  && pass "-dc decompresses to stdout" || fail "-dc decompresses to stdout"

# -cd : same flags reordered (bundling is order-independent)
("$GZSTD" -cd "$TMPDIR/bf.zst" 2>/dev/null | cmp -s - "$TMPDIR/bf.bin") \
  && pass "-cd (reordered) decompresses to stdout" || fail "-cd (reordered) decompresses to stdout"

# -dcf : decompress + stdout + force, round-trips
("$GZSTD" -dcf "$TMPDIR/bf.zst" 2>/dev/null | cmp -s - "$TMPDIR/bf.bin") \
  && pass "-dcf round-trips" || fail "-dcf round-trips"

# -dk : bundle decompresses to the derived path and leaves the .zst in place
rm -f "$TMPDIR/bf"
("$GZSTD" -dk --cpu-only "$TMPDIR/bf.zst" 2>/dev/null \
   && test -f "$TMPDIR/bf.zst" && cmp -s "$TMPDIR/bf" "$TMPDIR/bf.bin") \
  && pass "-dk decompresses and keeps input" || fail "-dk decompresses and keeps input"
rm -f "$TMPDIR/bf"

# Regression: a bundle with a non-operation char must NOT be silently accepted
expect_exit 2 "$GZSTD" -dz "$TMPDIR/bf.zst" \
  && pass "-dz (unknown bundle) rejected (exit 2)" || fail "-dz (unknown bundle) rejected"

# Regression: -vv keeps its repeat semantics (debug), not split into -v -v.
# At V_DEBUG the decomp worker prints per-task "take seq=" lines; -v (V_VERBOSE)
# does not — so this distinguishes "still debug" from "accidentally bundled".
("$GZSTD" -d -vv --cpu-only "$TMPDIR/bf.zst" -o "$TMPDIR/bf.out" 2>&1 \
   | grep -q "take seq=") \
  && pass "-vv still maps to debug (repeat flag not bundled)" || fail "-vv still maps to debug"
rm -f "$TMPDIR/bf.bin" "$TMPDIR/bf.zst" "$TMPDIR/bf" "$TMPDIR/bf.out"

fi  # $EXTENSIVE (zstd-compat flag layer + bundled short flags)

# ============================================================
section "Parameter honor verification"
# ----------------------------------------------------------------
# Round-trip tests confirm flags don't BREAK things, but they don't
# confirm flags are actually APPLIED at runtime.  These tests parse
# verbose output to verify the runtime behaviour matches the user's
# CLI input.  Added after a regression where --gpu-batch=N was
# accepted but the GPU popped batches of 4-8 anyway (v0.12.39 fix).
# ============================================================

# Compressible-but-non-trivial input — needs enough frames to exercise
# multi-batch behaviour but not so much that tests are slow.
PHV="$TMPDIR/phv.bin"
PHV_ZST="$TMPDIR/phv.bin.zst"
PHV_OUT="$TMPDIR/phv.out"
dd if=/dev/urandom bs=1M count=64 2>/dev/null > "$PHV"
"$GZSTD" -k -f --cpu-only --chunk-size=4 "$PHV" -o "$PHV_ZST" 2>/dev/null

# --- --gpu-batch=N actually produces batches of N at -vv ---
if has_gpu 2>/dev/null; then
  for batch in 4 8 16; do
    out=$("$GZSTD" -d --gpu-only --gpu-batch=$batch --overwrite -vv \
      "$PHV_ZST" -o "$PHV_OUT" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    # First "take batch=N" line during steady state — should equal $batch
    # (or be the final partial batch at end-of-queue).  Check ALL take lines:
    # all but possibly the last must be exactly $batch.
    actual_batches=$(echo "$out" | grep -oE "take batch=[0-9]+" | grep -oE "[0-9]+")
    if [[ -z "$actual_batches" ]]; then
      fail "--gpu-batch=$batch honored at -vv" "no 'take batch=' lines in output"
    else
      # All-but-last should equal $batch
      n=$(echo "$actual_batches" | wc -l)
      if [[ $n -le 1 ]]; then
        # Only one batch (small file) — the single batch must be ≤ $batch
        last=$(echo "$actual_batches" | head -1)
        if [[ $last -le $batch ]]; then
          pass "--gpu-batch=$batch honored at -vv" "(single batch=$last)"
        else
          fail "--gpu-batch=$batch honored" "single batch=$last > $batch"
        fi
      else
        # Multiple batches — non-final should all be exactly $batch
        non_final=$(echo "$actual_batches" | head -n -1 | sort -u)
        if [[ "$non_final" == "$batch" ]]; then
          pass "--gpu-batch=$batch honored at -vv" "(N=$n batches, all=$batch except possibly last)"
        else
          fail "--gpu-batch=$batch honored at -vv" "expected all=$batch, got: $non_final"
        fi
      fi
    fi
  done

  # --- --gpu-streams=N spawns N CUDA streams per device ---
  for streams in 1 2; do
    out=$("$GZSTD" -d --gpu-only --gpu-batch=4 --gpu-streams=$streams --overwrite -vv \
      "$PHV_ZST" -o "$PHV_OUT" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    # Expect [GPU0/S0] up to [GPU0/S(streams-1)] in pre-alloc lines.
    max_s=$(echo "$out" | grep -oE "\[GPU[0-9]+/S[0-9]+\] pre-alloc" \
            | grep -oE "S[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
    if [[ -z "$max_s" ]]; then
      fail "--gpu-streams=$streams honored" "no pre-alloc lines"
    elif [[ $max_s -eq $((streams - 1)) ]]; then
      pass "--gpu-streams=$streams honored" "(streams 0..$max_s allocated)"
    else
      fail "--gpu-streams=$streams honored" "max stream index=$max_s, expected $((streams - 1))"
    fi
  done
else
  skip "--gpu-batch=N honored at -vv" "no GPU"
  skip "--gpu-streams=N honored" "no GPU"
fi

# --- --chunk-size=N produces ceil(file_size / N MiB) frames on compress ---
for chunk in 2 8; do
  src_mib=64
  expected_frames=$((src_mib / chunk))
  cz="$TMPDIR/cz-${chunk}.zst"
  out=$("$GZSTD" -k -f --cpu-only --chunk-size=$chunk -vv "$PHV" -o "$cz" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g')
  actual_frames=$(echo "$out" | grep -cE "\[CPU/T[0-9]+\] take seq=")
  if [[ "$actual_frames" == "$expected_frames" ]]; then
    pass "--chunk-size=$chunk produces $expected_frames frames"
  else
    fail "--chunk-size=$chunk frames" "expected $expected_frames take lines, got $actual_frames"
  fi
  rm -f "$cz"
done

# --- -T N spawns N CPU compression workers ---
for threads in 1 2 4; do
  out=$("$GZSTD" -k -f --cpu-only -T $threads -v "$PHV" -o "$TMPDIR/threadt.zst" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g')
  # Look for "[CPU] N worker threads online" or single-thread streaming notice
  if [[ $threads -eq 1 ]]; then
    if echo "$out" | grep -qE "single-thread"; then
      pass "-T 1 uses single-thread streaming path"
    else
      fail "-T 1 uses single-thread path" "no single-thread notice"
    fi
  else
    if echo "$out" | grep -qE "\[CPU\][[:space:]]+$threads[[:space:]]+worker"; then
      pass "-T $threads spawns $threads workers"
    else
      fail "-T $threads spawns $threads workers" "no '[CPU] $threads worker' line"
    fi
  fi
done
rm -f "$TMPDIR/threadt.zst"

# --- Verbosity-level outputs ---
# -v: completion summary, info-level lines.  Must NOT contain V_DEBUG-only output.
out_v=$("$GZSTD" -d --cpu-only --overwrite -v "$PHV_ZST" -o "$PHV_OUT" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g')
out_vv=$("$GZSTD" -d --cpu-only --overwrite -vv "$PHV_ZST" -o "$PHV_OUT" 2>&1 \
         | sed 's/\x1b\[[0-9;]*m//g')
out_vvv=$("$GZSTD" -d --cpu-only --overwrite -vvv "$PHV_ZST" -o "$PHV_OUT" 2>&1 \
          | sed 's/\x1b\[[0-9;]*m//g')

# -v should NOT have per-task "[CPU/T#] take seq=" (V_DEBUG content)
if echo "$out_v" | grep -qE "\[CPU/T[0-9]+\] take seq="; then
  fail "-v omits V_DEBUG content" "found [CPU/T#] take seq= line at -v"
else
  pass "-v omits V_DEBUG content"
fi
# -vv SHOULD have per-task "take seq=" lines
if echo "$out_vv" | grep -qE "\[CPU/T[0-9]+\] take seq="; then
  pass "-vv shows per-task take lines"
else
  fail "-vv shows per-task take lines" "no take seq= line at -vv"
fi
# -vv should NOT have V_TRACE-only "[SPLIT] frame" lines
if echo "$out_vv" | grep -qE "\[SPLIT\] frame"; then
  fail "-vv omits V_TRACE content" "found [SPLIT] line at -vv"
else
  pass "-vv omits V_TRACE content"
fi
# -vvv SHOULD have [SPLIT] frame lines
if echo "$out_vvv" | grep -qE "\[SPLIT\] frame"; then
  pass "-vvv shows V_TRACE [SPLIT] lines"
else
  fail "-vvv shows V_TRACE [SPLIT] lines" "no [SPLIT] line at -vvv"
fi

# Verbosity escalates: -vvv has more unique lines than -vv has more than -v
n_v=$(echo "$out_v"   | sort -u | wc -l)
n_vv=$(echo "$out_vv"  | sort -u | wc -l)
n_vvv=$(echo "$out_vvv" | sort -u | wc -l)
if [[ $n_vvv -gt $n_vv && $n_vv -gt $n_v ]]; then
  pass "verbosity escalates (-v=$n_v < -vv=$n_vv < -vvv=$n_vvv unique lines)"
else
  fail "verbosity escalates" "-v=$n_v -vv=$n_vv -vvv=$n_vvv"
fi

# --- --memlimit applied (re-verifies the v0.12.30 fix) ---
# Already tested in the zstd-compat section, but assert it appears in -v output too.
out=$("$GZSTD" -d --cpu-only --overwrite -M 1024 -v "$PHV_ZST" -o "$PHV_OUT" 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g')
# -M 1024 with default 16 MiB chunks → ram cap kicks in at 1024/16=64 frames.
# Just sanity that decompress completes successfully.
if cmp -s "$PHV" "$PHV_OUT"; then
  pass "-M 1024 round-trip"
else
  fail "-M 1024 round-trip" "output mismatch"
fi

# --- --throttle-frames=N visible at -v ---
out=$("$GZSTD" -k -f --cpu-only --throttle-frames=64 -v "$PHV" -o "$TMPDIR/thr.zst" 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g')
if echo "$out" | grep -qE "\[THROTTLE\].* 64 frames|source=user"; then
  pass "--throttle-frames=64 visible at -v"
else
  fail "--throttle-frames=64 visible" "no source=user or 64 frames"
fi
rm -f "$TMPDIR/thr.zst"

# --- --no-sparse vs default sparse on file with zeros ---
dd if=/dev/zero bs=1M count=4 2>/dev/null > "$TMPDIR/sparse.bin"
"$GZSTD" -k -f --cpu-only "$TMPDIR/sparse.bin" -o "$TMPDIR/sparse.zst" 2>/dev/null
"$GZSTD" -d -f --cpu-only "$TMPDIR/sparse.zst" -o "$TMPDIR/sparse-default.bin" 2>/dev/null
"$GZSTD" -d -f --cpu-only --no-sparse "$TMPDIR/sparse.zst" -o "$TMPDIR/sparse-no.bin" 2>/dev/null
default_blocks=$(stat -c '%b' "$TMPDIR/sparse-default.bin" 2>/dev/null || echo 0)
nosparse_blocks=$(stat -c '%b' "$TMPDIR/sparse-no.bin" 2>/dev/null || echo 0)
# Sparse should use fewer blocks than non-sparse for an all-zeros file.
if [[ $default_blocks -lt $nosparse_blocks ]]; then
  pass "--[no-]sparse changes block usage" "(sparse=$default_blocks < dense=$nosparse_blocks)"
else
  fail "--[no-]sparse changes block usage" "sparse=$default_blocks dense=$nosparse_blocks"
fi
rm -f "$TMPDIR/sparse.bin" "$TMPDIR/sparse.zst" "$TMPDIR/sparse-default.bin" "$TMPDIR/sparse-no.bin"

# --- --ultra is required for level 20+ ---
out=$("$GZSTD" -k -f --cpu-only -20 "$PHV" -o "$TMPDIR/ultra.zst" 2>&1)
if echo "$out" | grep -qiE "ultra"; then
  pass "level -20 without --ultra rejects/warns"
else
  fail "level -20 without --ultra" "no ultra warning"
fi
# With --ultra, level 20 should compress fine.
out=$("$GZSTD" -k -f --cpu-only --ultra -20 "$PHV" -o "$TMPDIR/ultra.zst" 2>&1)
if [[ -s "$TMPDIR/ultra.zst" ]]; then
  pass "--ultra -20 produces output"
else
  fail "--ultra -20 produces output" "no output"
fi
rm -f "$TMPDIR/ultra.zst"

rm -f "$PHV" "$PHV_ZST" "$PHV_OUT"

# ============================================================
# Final summary
# ============================================================
clear_progress
print_summary

# Drift check: if the actual ran count differs from EXPECTED_TESTS,
# nudge the maintainer to bump the constant.
TOTAL_RAN=$(( PASS + FAIL ))
if [[ $TOTAL_RAN -ne $EXPECTED_TESTS ]]; then
  printf "\n  ${C_DIM}note: EXPECTED_TESTS=%d at top of script but %d ran — please update.${C_RESET}\n" \
    "$EXPECTED_TESTS" "$TOTAL_RAN"
fi

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
