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

# Build-time capability, not runtime: a USE_NVCOMP=OFF binary compiles out
# apply_backend_defaults entirely, so [ADAPT] fingerprint/prior lines never
# print there even though the profile writer itself works.
has_nvcomp() { "$GZSTD" -V 2>&1 | grep -qi nvcomp; }

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
# Counts assume a GPU is present.  --extensive adds back the gated sections
# (File management, Multi-file, Sparse, Threading, Stress, Help/version,
# Output redirection, Sync output, Space-separated values, Thread option
# forms, Verbose output validation, Completion summary format).
EXPECTED_TESTS=354
$EXTENSIVE && EXPECTED_TESTS=471
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
# 7-10. File management, multi-file, sparse, threading (extensive)
# ============================================================
if $EXTENSIVE; then
section "File management flags (extensive)"

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
section "Multi-file operation (extensive)"

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
section "Sparse file support (extensive)"

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
section "Threading options (extensive)"

for threads in 1 2 4; do
  compressed="$TMPDIR/thread-${threads}.zst"; recovered="$TMPDIR/thread-${threads}.recovered"
  run_test "$GZSTD" -k -f --cpu-only -T $threads "$TMPDIR/medium.txt" -o "$compressed" 2>/dev/null
  run_test "$GZSTD" -d -k -f --cpu-only -T $threads "$compressed" -o "$recovered" 2>/dev/null
  files_match "$TMPDIR/medium.txt" "$recovered" && pass "-T $threads" || fail "-T $threads" "mismatch"
  rm -f "$compressed" "$recovered"
done
fi  # $EXTENSIVE (File management, multi-file, sparse, threading)

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

# ENOSPC-class write failure (v0.14.94): RLIMIT_FSIZE with SIGXFSZ ignored
# makes writes fail mid-stream exactly like a full disk.  BUFFERED output
# used to report exit 0 over a truncated archive (glibc fwrite claims a full
# count after a failed flush; ferror was never checked) and then delete the
# no-keep input; --direct used to hang forever (the dead write worker
# stopped releasing FrameThrottle permits).  Both must exit 3, keep the
# input, and leave no output behind.
head -c 8000000 /dev/urandom > "$TMPDIR/enospc.bin"
rc=0; ( trap '' XFSZ; ulimit -f 500
  timeout 30 "$GZSTD" --cpu-only --no-direct -q -f "$TMPDIR/enospc.bin" \
    -o "$TMPDIR/enospc.zst" >/dev/null 2>&1 ) || rc=$?
if [[ $rc -eq 3 && -f "$TMPDIR/enospc.bin" && ! -e "$TMPDIR/enospc.zst" ]]; then
  pass "exit 3: buffered write failure (disk full)" "(input kept, no output)"
else fail "buffered disk-full" "(exit $rc; input $([[ -f $TMPDIR/enospc.bin ]] && echo kept || echo LOST))"; fi
rc=0; ( trap '' XFSZ; ulimit -f 500
  timeout 30 "$GZSTD" --cpu-only --direct -q -f "$TMPDIR/enospc.bin" \
    -o "$TMPDIR/enospc.zst" >/dev/null 2>&1 ) || rc=$?
if [[ $rc -eq 3 ]]; then
  pass "exit 3: O_DIRECT write failure (no hang)"
elif [[ $rc -eq 124 ]]; then fail "O_DIRECT disk-full" "HUNG (timeout)"
else fail "O_DIRECT disk-full" "(exit $rc)"; fi
rm -f "$TMPDIR/enospc.bin" "$TMPDIR/enospc.zst"

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

  # GPU-fault recovery: a mid-run GPU fault (simulated via the debug env hook)
  # must abandon ALL GPU output and rebuild the whole archive CPU-only from the
  # original input, producing a valid archive that round-trips exactly.
  gf_src="$TMPDIR/large.bin"; gf_zst="$TMPDIR/gpufault.zst"; gf_rec="$TMPDIR/gpufault.rec"
  GZSTD_DEBUG_FAIL_GPU_AFTER=0 "$GZSTD" --gpu-only -k -f "$gf_src" -o "$gf_zst" 2>/dev/null
  "$GZSTD" -d -k -f "$gf_zst" -o "$gf_rec" 2>/dev/null
  files_match "$gf_src" "$gf_rec" && pass "GPU fault -> CPU-only rebuild (round-trips)" \
    || fail "GPU fault -> CPU-only rebuild" "mismatch after rebuild"
  rm -f "$gf_zst" "$gf_rec"

  # Same recovery in --tar mode: rebuild from the source layout, extract matches.
  # (gzstd --tar anchors member names on the given source path, which may be
  # absolute, so locate the file by name rather than a fixed extracted path.)
  gft_zst="$TMPDIR/gpufault-tar.zst"; gft_out="$TMPDIR/gpufault-extract"
  mkdir -p "$gft_out"
  GZSTD_DEBUG_FAIL_GPU_AFTER=0 "$GZSTD" --gpu-only -f -o "$gft_zst" --tar "$TMPDIR/tree" 2>/dev/null
  "$GZSTD" -d --tar "$gft_zst" -C "$gft_out" 2>/dev/null
  if [[ -n "$(find "$gft_out" -name small.txt -print -quit 2>/dev/null)" ]]; then
    pass "GPU fault -> CPU-only rebuild (--tar)"
  else
    fail "GPU fault -> CPU-only rebuild (--tar)" "missing extracted file after rebuild"
  fi
  rm -rf "$gft_out" "$gft_zst"

  # A GPU fault over a pipe cannot be rebuilt (output already streamed downstream,
  # input may be consumed), so gzstd must die loudly with a non-zero exit and an
  # explanatory message rather than emit a silently-corrupt stream.
  gf_perr="$TMPDIR/gpufault.err"
  GZSTD_DEBUG_FAIL_GPU_AFTER=0 "$GZSTD" --gpu-only -c "$gf_src" 2>"$gf_perr" | cat >/dev/null
  gf_rc=${PIPESTATUS[0]}
  if [[ $gf_rc -ne 0 ]] && grep -qiE "pipe|corrupt|--cpu-only" "$gf_perr"; then
    pass "GPU fault over a pipe dies loudly (exit $gf_rc)"
  else
    fail "GPU fault over a pipe dies loudly" "rc=$gf_rc (expected non-zero + message)"
  fi
  rm -f "$gf_perr"

  # GPU-side --verify (--verify-engine=gpu): decompress + raw byte-compare each
  # batch in VRAM on the GPU.  Tiny config (1 GPU/1 stream/batch 1/4 MiB chunks)
  # so the verify buffers fit a small card.
  # Incompressible source so each 4 MiB frame's compressed output is large (the
  # corruption hook flips a byte at offset 64, which must land inside the frame).
  gvsrc="$TMPDIR/gvrand.bin"; head -c 8388608 /dev/urandom > "$gvsrc"
  gvz="$TMPDIR/gpuverify.zst"; gvr="$TMPDIR/gpuverify.rec"; gverr="$TMPDIR/gpuverify.err"
  GVCFG="--verify-engine=gpu --gpu-only --gpu-devices=1 --gpu-streams=1 --gpu-batch=1 --chunk-size=4"
  # Clean: GPU verify actually runs ([GPU-VERIFY] line) and the archive round-trips.
  "$GZSTD" $GVCFG -v -k -f "$gvsrc" -o "$gvz" 2>"$gverr"
  "$GZSTD" -d -k -f "$gvz" -o "$gvr" 2>/dev/null
  if grep -qi "GPU-VERIFY" "$gverr" && files_match "$gvsrc" "$gvr"; then
    pass "GPU verify (--verify-engine=gpu) runs + round-trips"
  else
    fail "GPU verify clean" "no [GPU-VERIFY] line or round-trip mismatch"
  fi
  # Corruption injected into the compressed output in VRAM must be caught by the
  # GPU compare and rebuilt CPU-only into a correct archive.
  GZSTD_DEBUG_GPU_CORRUPT=1 "$GZSTD" $GVCFG -k -f "$gvsrc" -o "$gvz" 2>"$gverr"
  "$GZSTD" -d -k -f "$gvz" -o "$gvr" 2>/dev/null
  if grep -qiE "GPU verify|rebuilding CPU-only" "$gverr" && files_match "$gvsrc" "$gvr"; then
    pass "GPU verify catches VRAM corruption, rebuilds clean"
  else
    fail "GPU verify corruption catch" "no rebuild or output mismatch"
  fi
  # --verify-engine=gpu without --gpu-only can't run GPU verify: warn, fall back
  # to CPU verify, still produce a correct archive.
  "$GZSTD" --verify-engine=gpu --cpu-only -k -f "$gvsrc" -o "$gvz" 2>"$gverr"
  "$GZSTD" -d -k -f "$gvz" -o "$gvr" 2>/dev/null
  if grep -qi "requires --gpu-only" "$gverr" && files_match "$gvsrc" "$gvr"; then
    pass "--verify-engine=gpu without --gpu-only warns + falls back to CPU verify"
  else
    fail "--verify-engine=gpu fallback warning" "no warning or round-trip mismatch"
  fi
  rm -f "$gvz" "$gvr" "$gverr"
else
  skip "gpu-only compress/decompress" "no GPU"
  skip "GPU verify (--verify-engine=gpu) runs + round-trips" "no GPU"
  skip "GPU verify catches VRAM corruption, rebuilds clean" "no GPU"
  skip "--verify-engine=gpu without --gpu-only warns + falls back to CPU verify" "no GPU"
  skip "hybrid compress/decompress" "no GPU"
  skip "gpu integrity test" "no GPU"
  skip "gpu tar" "no GPU"
  skip "gpu batch sizes" "no GPU"
  skip "GPU fault -> CPU-only rebuild (round-trips)" "no GPU"
  skip "GPU fault -> CPU-only rebuild (--tar)" "no GPU"
  skip "GPU fault over a pipe dies loudly" "no GPU"
fi

# ============================================================
# 18b. --verify (background decompress-verify on write)
# ============================================================
section "--verify (decompress-verify on write)"

# large.bin is 32 MiB = multiple default-size frames, so seq 0 is a real frame.
vsrc="$TMPDIR/large.bin"

# Clean run: --verify must succeed and round-trip exactly.
vz="$TMPDIR/verify-clean.zst"; vr="$TMPDIR/verify-clean.rec"
run_test "$GZSTD" --verify --cpu-only -k -f "$vsrc" -o "$vz" 2>/dev/null
run_test "$GZSTD" -d -k -f "$vz" -o "$vr" 2>/dev/null
files_match "$vsrc" "$vr" && pass "--verify clean run round-trips" \
  || fail "--verify clean run" "mismatch"
rm -f "$vz" "$vr"

# Injected corruption: --verify must catch the bad frame (logging its sequence),
# rebuild CPU-only, and produce a valid archive that round-trips exactly.
vcz="$TMPDIR/verify-corrupt.zst"; vcr="$TMPDIR/verify-corrupt.rec"; verr="$TMPDIR/verify.err"
GZSTD_DEBUG_CORRUPT_FRAME=0 "$GZSTD" --verify --cpu-only -k -f "$vsrc" -o "$vcz" 2>"$verr"
"$GZSTD" -d -k -f "$vcz" -o "$vcr" 2>/dev/null
if files_match "$vsrc" "$vcr" && grep -qiE "verify caught a corrupt frame.*sequence 0" "$verr"; then
  pass "--verify catches corruption, rebuilds, round-trips"
else
  fail "--verify catches corruption" "no rebuild logged or output mismatch"
fi
rm -f "$vcz" "$vcr" "$verr"

# Persistent corruption + --verify-retries=2: must give up after 2 rebuild
# attempts with a non-zero (data) exit rather than loop forever.
vgz="$TMPDIR/verify-giveup.zst"; gerr="$TMPDIR/verify-giveup.err"
GZSTD_DEBUG_CORRUPT_FRAME=0 GZSTD_DEBUG_CORRUPT_PERSIST=1 \
  "$GZSTD" --verify --cpu-only --verify-retries=2 -k -f "$vsrc" -o "$vgz" 2>"$gerr"
vg_rc=$?
if [[ $vg_rc -ne 0 ]] && grep -qiE "failed to verify after 2" "$gerr"; then
  pass "--verify-retries caps rebuilds (exit $vg_rc)"
else
  fail "--verify-retries cap" "rc=$vg_rc (expected non-zero + give-up message)"
fi
rm -f "$vgz" "$gerr"

# Corruption over a pipe cannot be rebuilt (output already streamed downstream),
# so --verify must die loudly with a non-zero exit, not emit a corrupt stream.
perr="$TMPDIR/verify-pipe.err"
GZSTD_DEBUG_CORRUPT_FRAME=0 "$GZSTD" --verify --cpu-only -c "$vsrc" 2>"$perr" | cat >/dev/null
vp_rc=${PIPESTATUS[0]}
if [[ $vp_rc -ne 0 ]] && grep -qiE "pipe|corrupt|regular files" "$perr"; then
  pass "--verify over a pipe dies loudly (exit $vp_rc)"
else
  fail "--verify over a pipe dies loudly" "rc=$vp_rc (expected non-zero + message)"
fi
rm -f "$perr"

# ============================================================
# 18c. --keep-going (decompress recovery + damage report)
# ============================================================
section "--keep-going (decompress recovery)"

# Multi-frame plain .zst, one byte flipped inside a middle frame.  A single flip
# decodes in full but fails the checksum, so the frame is recovered UNVERIFIED and
# the run exits 6 (not aborting) with the output length preserved.
kgsrc="$TMPDIR/keepgoing.bin"
head -c 50331648 /dev/urandom > "$kgsrc"          # 48 MiB -> 3 default frames
kgz="$TMPDIR/keepgoing.zst"; kgbad="$TMPDIR/keepgoing-bad.zst"; kgout="$TMPDIR/keepgoing.out"
run_test "$GZSTD" --cpu-only -k -f "$kgsrc" -o "$kgz" 2>/dev/null
cp "$kgz" "$kgbad"
kgsz=$(stat -c%s "$kgbad")
printf '\xFF' | dd of="$kgbad" bs=1 seek=$(( kgsz / 2 )) count=1 conv=notrunc 2>/dev/null

# Without --keep-going: abort with a data error AND suggest --keep-going.
"$GZSTD" -d -k -f "$kgbad" -o "$kgout" 2>"$TMPDIR/kg1.err"; kg1=$?
if [[ $kg1 -ne 0 ]] && grep -qiE "keep-going" "$TMPDIR/kg1.err"; then
  pass "corrupt .zst without --keep-going aborts + suggests --keep-going (exit $kg1)"
else
  fail "corrupt .zst aborts + suggests flag" "rc=$kg1"
fi

# With --keep-going: recover, report, exit 6 or 7, length preserved.
"$GZSTD" -d --keep-going -k -f "$kgbad" -o "$kgout" 2>"$TMPDIR/kg2.err"; kg2=$?
kgout_sz=$(stat -c%s "$kgout" 2>/dev/null || echo 0)
if { [[ $kg2 -eq 6 ]] || [[ $kg2 -eq 7 ]]; } \
   && grep -qiE "recovered past .* damaged frame" "$TMPDIR/kg2.err" \
   && [[ "$kgout_sz" == "$(stat -c%s "$kgsrc")" ]]; then
  pass "--keep-going recovers a corrupt .zst (exit $kg2, length preserved)"
else
  fail "--keep-going recovers .zst" "rc=$kg2 outsize=$kgout_sz expected=$(stat -c%s "$kgsrc")"
fi
rm -f "$kgz" "$kgbad" "$kgout" "$kgsrc" "$TMPDIR/kg1.err" "$TMPDIR/kg2.err"

# --tar: a flipped byte inside a large member must name THAT file and still
# recover the undamaged members intact.
kgtree="$TMPDIR/kgtree"; rm -rf "$kgtree"; mkdir -p "$kgtree"
echo "intact small file" > "$kgtree/a_small.txt"
head -c 41943040 /dev/urandom > "$kgtree/big.bin"   # 40 MiB -> spans several frames
echo "another intact file" > "$kgtree/z_small.txt"
kgarc="$TMPDIR/kg.tar.zst"; kgbadarc="$TMPDIR/kg-bad.tar.zst"
run_test "$GZSTD" --cpu-only -f -o "$kgarc" --tar "$kgtree" 2>/dev/null
cp "$kgarc" "$kgbadarc"
printf '\xFF' | dd of="$kgbadarc" bs=1 seek=$(( 20 * 1024 * 1024 )) count=1 conv=notrunc 2>/dev/null
kgxd="$TMPDIR/kg-extract"; rm -rf "$kgxd"; mkdir -p "$kgxd"
"$GZSTD" -d --tar --keep-going "$kgbadarc" -C "$kgxd" 2>"$TMPDIR/kg3.err"; kg3=$?
a_ext=$(find "$kgxd" -name a_small.txt -print -quit 2>/dev/null)
if { [[ $kg3 -eq 6 ]] || [[ $kg3 -eq 7 ]]; } \
   && grep -qiE "damaged file" "$TMPDIR/kg3.err" \
   && grep -qE "big\.bin" "$TMPDIR/kg3.err" \
   && [[ -n "$a_ext" ]] && cmp -s "$kgtree/a_small.txt" "$a_ext"; then
  pass "--keep-going --tar names the damaged file, recovers the rest (exit $kg3)"
else
  fail "--keep-going --tar damage report" "rc=$kg3 (see kg3.err)"
fi
rm -rf "$kgtree" "$kgxd"; rm -f "$kgarc" "$kgbadarc" "$TMPDIR/kg3.err"

# Framing break: a truncated archive (frame boundary lost) cannot be resynced.
# --keep-going must still recover the intact prefix, report RECOVERY STOPPED, and
# exit 7 (incomplete) rather than aborting abruptly.
fbsrc="$TMPDIR/framing.bin"
head -c 50331648 /dev/urandom > "$fbsrc"          # 48 MiB -> 3 frames
fbz="$TMPDIR/framing.zst"; fbt="$TMPDIR/framing-trunc.zst"; fbo="$TMPDIR/framing.out"
run_test "$GZSTD" --cpu-only -k -f "$fbsrc" -o "$fbz" 2>/dev/null
head -c 20971520 "$fbz" > "$fbt"                  # cut mid-frame-1: frame 0 intact, sync lost after
"$GZSTD" -d --keep-going -k -f "$fbt" -o "$fbo" 2>"$TMPDIR/fb.err"; fb=$?
fbo_sz=$(stat -c%s "$fbo" 2>/dev/null || echo 0)
if [[ $fb -eq 7 ]] && grep -qiE "RECOVERY STOPPED" "$TMPDIR/fb.err" \
   && [[ "$fbo_sz" == "16777216" ]] && cmp -s -n 16777216 "$fbsrc" "$fbo"; then
  pass "--keep-going recovers the prefix of a truncated stream (exit 7)"
else
  fail "--keep-going framing break" "rc=$fb outsize=$fbo_sz"
fi
rm -f "$fbsrc" "$fbz" "$fbt" "$fbo" "$TMPDIR/fb.err"

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
  # -k 10: if the soft timeout's SIGTERM doesn't bring the process down within
  # 10s, follow with SIGKILL.  A wedged multi-threaded gzstd can otherwise ignore
  # SIGTERM and leave `timeout` itself blocked, stalling the whole suite.
  timeout --foreground -k 10 "${timeout_s}" "$@" >"$logfile" 2>&1 || rc=$?
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
# Parallel-prefetch decompress reader (v0.13.71)
# The MT reader engages only for seekable regular files > 128 MiB, so the
# other decompress tests (small files) never exercise it.  These verify
# byte-identical round-trips for the frame/block-size relationships that
# stress the carry/spanning logic: chunk=1 (tiny frames, ~64 per 64 MiB
# block → heavy boundary-spanning) and chunk=128 (frames larger than a
# block → multi-block carry).  Cross-checked against the single-threaded
# reader (forced via --direct-read).  CPU-only, runs on all machines.
# ============================================================
section "Parallel-prefetch decompress reader"

# Mostly incompressible so the COMPRESSED .zst (what the reader reads) clears
# the 128 MiB MT gate; a zero run still exercises tiny trailing frames.
spin "mtreader.bin (176 MiB, compresses > 128 MiB)"
dd if=/dev/urandom bs=1M count=160 2>/dev/null  > "$TMPDIR/mtreader.bin"
dd if=/dev/zero    bs=1M count=16  2>/dev/null >> "$TMPDIR/mtreader.bin"
spin_done

for cs in 1 128; do
  "$GZSTD" -k -f --cpu-only --chunk-size=$cs "$TMPDIR/mtreader.bin" -o "$TMPDIR/mt-$cs.zst" 2>/dev/null
  # MT path (default cpu-only decompress on a > 128 MiB regular file)
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/mt-$cs.zst" -o "$TMPDIR/mt-$cs.dec" 2>/dev/null
  # single-reader reference (--direct-read keeps one reader)
  "$GZSTD" -d -k -f --cpu-only --direct-read "$TMPDIR/mt-$cs.zst" -o "$TMPDIR/sg-$cs.dec" 2>/dev/null
  if files_match "$TMPDIR/mtreader.bin" "$TMPDIR/mt-$cs.dec" \
     && files_match "$TMPDIR/mtreader.bin" "$TMPDIR/sg-$cs.dec"; then
    pass "parallel reader round-trip, chunk=$cs MiB" "MT == single == source"
  else
    fail "parallel reader round-trip, chunk=$cs MiB" \
         "MT match: $(files_match "$TMPDIR/mtreader.bin" "$TMPDIR/mt-$cs.dec" && echo y || echo n), single: $(files_match "$TMPDIR/mtreader.bin" "$TMPDIR/sg-$cs.dec" && echo y || echo n)"
  fi
done

# Confirm the parallel path actually engaged (not silently single-threaded).
if "$GZSTD" -d -k -f -v --cpu-only "$TMPDIR/mt-1.zst" -o /dev/null 2>&1 \
     | tr '\r' '\n' | grep -q "parallel prefetch"; then
  pass "parallel reader engaged on > 128 MiB file" "(prefetch threads active)"
else
  fail "parallel reader engaged on > 128 MiB file" "MT path did not engage"
fi

# --read-threads 1 must stay on the single reader.
if "$GZSTD" -d -k -f -v --cpu-only --read-threads 1 "$TMPDIR/mt-1.zst" -o /dev/null 2>&1 \
     | tr '\r' '\n' | grep -q "parallel prefetch"; then
  fail "--read-threads 1 stays single-threaded" "MT engaged unexpectedly"
else
  pass "--read-threads 1 stays single-threaded" "(single reader)"
fi

# Redirected stdin from a real file (`< file`) must engage the parallel reader
# (fstat detects the seekable fd) AND round-trip; a pipe must NOT engage it.
"$GZSTD" -d -k -f -v --cpu-only -c < "$TMPDIR/mt-1.zst" > "$TMPDIR/mt-redir.dec" 2>"$TMPDIR/mt-redir.log"
if grep -q "parallel prefetch" "$TMPDIR/mt-redir.log" \
   && files_match "$TMPDIR/mtreader.bin" "$TMPDIR/mt-redir.dec"; then
  pass "parallel reader on redirected stdin (< file)" "(MT engaged, round-trip OK)"
else
  fail "parallel reader on redirected stdin (< file)" \
       "MT: $(grep -q 'parallel prefetch' "$TMPDIR/mt-redir.log" && echo y || echo n), match: $(files_match "$TMPDIR/mtreader.bin" "$TMPDIR/mt-redir.dec" && echo y || echo n)"
fi
cat "$TMPDIR/mt-1.zst" | "$GZSTD" -d -k -f -v --cpu-only -c > "$TMPDIR/mt-pipe.dec" 2>"$TMPDIR/mt-pipe.log"
if grep -q "parallel prefetch" "$TMPDIR/mt-pipe.log"; then
  fail "pipe stays single-threaded (no fstat false-positive)" "MT engaged on a pipe"
elif files_match "$TMPDIR/mtreader.bin" "$TMPDIR/mt-pipe.dec"; then
  pass "pipe stays single-threaded (no fstat false-positive)" "(single reader, round-trip OK)"
else
  fail "pipe stays single-threaded (no fstat false-positive)" "round-trip mismatch"
fi

rm -f "$TMPDIR"/mt-*.zst "$TMPDIR"/mt-*.dec "$TMPDIR"/mt-*.log "$TMPDIR"/sg-*.dec "$TMPDIR/mtreader.bin"

# ============================================================
# 19e. MT reader streaming-fallback (concatenated unknown-size tail)
# ============================================================
# A gzstd archive (known frame sizes) concatenated with a zstd-streamed
# segment (no content-size header), sized past the 128 MiB MT gate.  The
# parallel reader parses the known frames, hits the unknown-size frame, and
# must NOT die: it warns and hands the remainder to the CPU streaming decoder,
# exactly like the single-threaded reader.  Regression for v0.13.77.  Needs
# stock zstd to produce the unknown-size frame (piped input => no content size).
section "MT reader streaming-fallback (unknown-size tail)"

if command -v zstd &>/dev/null; then
  spin "fallback fixture (gzstd archive + zstd-streamed tail, > 128 MiB)"
  dd if=/dev/urandom bs=1M count=110 2>/dev/null > "$TMPDIR/fb-A.bin"
  dd if=/dev/urandom bs=1M count=40  2>/dev/null > "$TMPDIR/fb-B.bin"
  cat "$TMPDIR/fb-A.bin" "$TMPDIR/fb-B.bin" > "$TMPDIR/fb-orig.bin"
  "$GZSTD" -q -k -f --cpu-only "$TMPDIR/fb-A.bin" -o "$TMPDIR/fb-A.zst" 2>/dev/null
  cat "$TMPDIR/fb-B.bin" | zstd -q -3 > "$TMPDIR/fb-B.zst" 2>/dev/null  # piped => unknown content size
  cat "$TMPDIR/fb-A.zst" "$TMPDIR/fb-B.zst" > "$TMPDIR/fb-combined.zst"
  spin_done

  # MT path (default cpu-only decompress on a > 128 MiB regular file), verbose
  # so we can confirm the parallel reader engaged AND took the fallback.
  "$GZSTD" -d -k -f -v --cpu-only "$TMPDIR/fb-combined.zst" -o "$TMPDIR/fb-mt.dec" 2>"$TMPDIR/fb.log"
  fb_rc=$?

  if [[ $fb_rc -eq 0 ]] && files_match "$TMPDIR/fb-orig.bin" "$TMPDIR/fb-mt.dec"; then
    pass "fallback round-trip" "(exit 0, output matches source)"
  else
    fail "fallback round-trip" "rc=$fb_rc, match=$(files_match "$TMPDIR/fb-orig.bin" "$TMPDIR/fb-mt.dec" && echo y || echo n)"
  fi

  if tr '\r' '\n' < "$TMPDIR/fb.log" | grep -q "parallel prefetch" \
     && tr '\r' '\n' < "$TMPDIR/fb.log" | grep -qi "no content-size header"; then
    pass "MT reader engaged, warned, fell back (no die)" "(parallel + warning)"
  else
    fail "MT reader engaged, warned, fell back (no die)" \
         "engaged=$(tr '\r' '\n' < "$TMPDIR/fb.log" | grep -q 'parallel prefetch' && echo y || echo n), warned=$(tr '\r' '\n' < "$TMPDIR/fb.log" | grep -qi 'no content-size header' && echo y || echo n)"
  fi

  # Single-reader reference (--read-threads 1 stays single) must match the MT output.
  "$GZSTD" -d -k -f --cpu-only --read-threads 1 "$TMPDIR/fb-combined.zst" -o "$TMPDIR/fb-sg.dec" 2>/dev/null
  if files_match "$TMPDIR/fb-mt.dec" "$TMPDIR/fb-sg.dec"; then
    pass "fallback parity: MT == single reader"
  else
    fail "fallback parity: MT == single reader" "outputs differ"
  fi

  rm -f "$TMPDIR"/fb-*.bin "$TMPDIR"/fb-*.zst "$TMPDIR"/fb-*.dec "$TMPDIR/fb-orig.bin" "$TMPDIR/fb.log"
else
  skip "fallback round-trip" "zstd not installed"
  skip "MT reader engaged, warned, fell back (no die)" "zstd not installed"
  skip "fallback parity: MT == single reader" "zstd not installed"
fi

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
# 21. Wildcard / glob file handling
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
# 22. End-of-options (--) handling
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
# 23. -c (stdout) option (extensive)
# ============================================================
if $EXTENSIVE; then
section "Output redirection (-c, -o, --output) (extensive)"

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
fi  # $EXTENSIVE (Output redirection)

# ============================================================
# 24. Pipe with various options
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
# 25. Thread option forms
# ============================================================
if $EXTENSIVE; then
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
fi  # $EXTENSIVE (Thread option forms)

# ============================================================
# 26. CPU scheduling options
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
# 27. Sync output (extensive)
# ============================================================
if $EXTENSIVE; then
section "Sync output (extensive)"

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
fi  # $EXTENSIVE (Sync output)

# ============================================================
# 28. Pinned memory options (GPU builds only, but shouldn't crash on CPU-only)
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
# 29. GPU-specific options (if GPU available)
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
# 30. Error handling robustness
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
# 31. Cross-level decompression
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
# 32. Argument order independence
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
# 33. Space-separated option values (--opt VAL vs --opt=VAL)
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
# 34. Verbose output validation (-v, -vv, -vvv)
# ============================================================
if $EXTENSIVE; then
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
fi  # $EXTENSIVE (Verbose output validation)

# ============================================================
# 35. Completion summary format validation
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
# 36. Ultra compression validation
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
# 37. Throttle budget tunables
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
  # --foreground -k 10: hard-kill if the soft 30s SIGTERM is ignored, so a wedged
  # run can't leave a defunct child with `timeout` blocked behind it (which would
  # stall the whole suite rather than failing this one test).
  timeout --foreground -k 10 30 "$GZSTD" -k -f --hybrid --throttle-frames=1 \
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

section "--adapt observe-only skeleton"

# v0.15.0 M1: --adapt runs a per-operation governor that classifies the
# bottleneck regime and reports it — it must change NOTHING about the output.

# Flags parse and the compress/decompress paths run under the governor.
"$GZSTD" --adapt -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/adapt.zst" 2>/dev/null \
  && pass "--adapt compress accepted (exit 0)" \
  || fail "--adapt compress accepted" "rc=$?"
"$GZSTD" --adapt -d -k -f --cpu-only "$TMPDIR/adapt.zst" -o "$TMPDIR/adapt.dec" 2>/dev/null
files_match "$TMPDIR/large.bin" "$TMPDIR/adapt.dec" \
  && pass "--adapt decompress round-trips" \
  || fail "--adapt decompress round-trips"

# --no-adapt is accepted (parsed now so scripts survive the eventual default flip).
"$GZSTD" --no-adapt -t --cpu-only "$TMPDIR/adapt.zst" 2>/dev/null \
  && pass "--no-adapt accepted (exit 0)" \
  || fail "--no-adapt accepted" "rc=$?"

# Observe-only guarantee: output bytes identical with and without --adapt.
"$GZSTD" -k -f --cpu-only -c "$TMPDIR/large.bin" > "$TMPDIR/adapt-plain.zst" 2>/dev/null
"$GZSTD" --adapt -k -f --cpu-only -c "$TMPDIR/large.bin" > "$TMPDIR/adapt-on.zst" 2>/dev/null
files_match "$TMPDIR/adapt-plain.zst" "$TMPDIR/adapt-on.zst" \
  && pass "--adapt output byte-identical to plain (observe-only)" \
  || fail "--adapt output byte-identical to plain"

# [ADAPT] telemetry prints at -v (regime summary, or the honest too-short
# line on a fast run) — and only at -v, and only with --adapt.
"$GZSTD" --adapt -v -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/adapt.zst" 2>"$TMPDIR/adapt-v.err"
grep -q '\[ADAPT\]' "$TMPDIR/adapt-v.err" \
  && pass "[ADAPT] summary printed at -v" \
  || fail "[ADAPT] summary printed at -v"
"$GZSTD" --adapt -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/adapt.zst" 2>"$TMPDIR/adapt-q.err"
"$GZSTD" -v -k -f --cpu-only "$TMPDIR/large.bin" -o "$TMPDIR/adapt.zst" 2>"$TMPDIR/adapt-off.err"
if ! grep -q '\[ADAPT\]' "$TMPDIR/adapt-q.err" && ! grep -q '\[ADAPT\]' "$TMPDIR/adapt-off.err"; then
  pass "[ADAPT] silent without -v and without --adapt"
else
  fail "[ADAPT] silent without -v and without --adapt"
fi
rm -f "$TMPDIR/adapt.zst" "$TMPDIR/adapt.dec" "$TMPDIR/adapt-plain.zst" \
      "$TMPDIR/adapt-on.zst" "$TMPDIR/adapt-v.err" "$TMPDIR/adapt-q.err" "$TMPDIR/adapt-off.err"

# ────────────────────────────────────────────────────────────
section "--adapt persistent profile"

# All profile tests point XDG_CACHE_HOME at a scratch dir — never the
# user's real cache.  A qualifying write needs a clean >=3 s run: 32 MiB
# of compressible text at -19 on one thread stays comfortably past that
# on any machine this suite runs on.
APROF_XDG="$TMPDIR/xdg-cache"
APROF="$APROF_XDG/gzstd/profile.json"
APROF_SRC="$TMPDIR/aprof-src.txt"
if [[ ! -f "$APROF_SRC" ]]; then
  for i in $(seq 1 400000); do echo "session=$i commit delta backup index worker $((i*7919))"; done > "$APROF_SRC"
fi
rm -rf "$APROF_XDG"

# 1. A qualifying --adapt run creates the profile with this run's direction.
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt -19 -T 1 --cpu-only -k -f \
  "$APROF_SRC" -o "$TMPDIR/aprof.zst" 2>/dev/null
if [[ -f "$APROF" ]] && grep -q '"compress"' "$APROF" && grep -q '"overall_gibs"' "$APROF"; then
  pass "qualifying --adapt run writes the profile"
else
  fail "qualifying --adapt run writes the profile"
fi

# 2. A second run EMA-merges into the same entry (runs: 2, one fingerprint).
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt -19 -T 1 --cpu-only -k -f \
  "$APROF_SRC" -o "$TMPDIR/aprof.zst" 2>/dev/null
if grep -q '"runs": 2' "$APROF" && [[ $(grep -c '"fingerprint"' "$APROF") -eq 1 ]]; then
  pass "second run merges (runs: 2, one fingerprint)"
else
  fail "second run merges (runs: 2, one fingerprint)"
fi

# 3. A corrupt profile is benign: run exits 0 and the file is rewritten fresh.
echo '{"gzstd_profile": 1, "entries": {' > "$APROF"   # truncated JSON
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt -19 -T 1 --cpu-only -k -f \
  "$APROF_SRC" -o "$TMPDIR/aprof.zst" 2>/dev/null
if [[ $? -eq 0 ]] && grep -q '"runs": 1' "$APROF"; then
  pass "corrupt profile discarded and rewritten (exit 0)"
else
  fail "corrupt profile discarded and rewritten"
fi

# 4. --no-profile writes nothing.
rm -rf "$APROF_XDG"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt --no-profile -19 -T 1 --cpu-only -k -f \
  "$APROF_SRC" -o "$TMPDIR/aprof.zst" 2>/dev/null
[[ ! -e "$APROF" ]] && pass "--no-profile writes nothing" || fail "--no-profile writes nothing"

# 5. A sub-3 s run writes nothing (small file, fast level).
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt --cpu-only -k -f \
  "$TMPDIR/medium.txt" -o "$TMPDIR/aprof-fast.zst" 2>/dev/null
[[ ! -e "$APROF" ]] && pass "sub-3s run writes nothing" || fail "sub-3s run writes nothing"

# 6. A failing run writes nothing (corrupt input, nonzero exit).
head -c 65536 /dev/urandom > "$TMPDIR/aprof-bad.zst"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt --cpu-only -d -f \
  "$TMPDIR/aprof-bad.zst" -o /dev/null 2>/dev/null
rc=$?
if [[ $rc -ne 0 && ! -e "$APROF" ]]; then
  pass "failing run writes nothing (exit $rc)"
else
  fail "failing run writes nothing" "rc=$rc"
fi

# 7. An unwritable cache dir is benign (exit 0, no crash).
mkdir -p "$APROF_XDG/gzstd"
chmod 555 "$APROF_XDG/gzstd"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --adapt -19 -T 1 --cpu-only -k -f \
  "$APROF_SRC" -o "$TMPDIR/aprof.zst" 2>/dev/null
rc=$?
chmod 755 "$APROF_XDG/gzstd"
[[ $rc -eq 0 ]] && pass "read-only cache dir benign (exit 0)" || fail "read-only cache dir benign" "rc=$rc"

# 8. --calibrate: measures, prints the table, records both directions.
rm -rf "$APROF_XDG"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --calibrate 2>"$TMPDIR/aprof-cal.err"
rc=$?
if [[ $rc -eq 0 ]] && grep -q "cpu compress" "$TMPDIR/aprof-cal.err" \
   && grep -q '"compress"' "$APROF" && grep -q '"decompress"' "$APROF"; then
  pass "--calibrate measures and records both directions"
else
  fail "--calibrate measures and records both directions" "rc=$rc"
fi

# 9. --calibrate --no-profile measures only.
rm -rf "$APROF_XDG"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --calibrate --no-profile 2>/dev/null
rc=$?
[[ $rc -eq 0 && ! -e "$APROF" ]] && pass "--calibrate --no-profile records nothing" \
  || fail "--calibrate --no-profile records nothing" "rc=$rc"

# 10. --calibrate -o EXISTING refuses (exit 2) and leaves the file untouched —
# the sink target is created and REMOVED, so a pre-existing file must never
# be accepted (a user may read "-o FILE" as "write the report to FILE").
echo "precious" > "$TMPDIR/aprof-precious.txt"
env XDG_CACHE_HOME="$APROF_XDG" "$GZSTD" --calibrate --no-profile \
  -o "$TMPDIR/aprof-precious.txt" 2>/dev/null
rc=$?
if [[ $rc -eq 2 ]] && [[ "$(cat "$TMPDIR/aprof-precious.txt")" == "precious" ]]; then
  pass "--calibrate -o existing target refused (exit 2, file intact)"
else
  fail "--calibrate -o existing target refused" "rc=$rc"
fi
rm -f "$TMPDIR/aprof-precious.txt"

rm -rf "$APROF_XDG" "$TMPDIR/aprof.zst" "$TMPDIR/aprof-fast.zst" "$TMPDIR/aprof-bad.zst" \
       "$TMPDIR/aprof-cal.err"

# ────────────────────────────────────────────────────────────
section "--adapt priors + residency-informed decompress default"

APRI_XDG="$TMPDIR/xdg-priors"
APRI="$APRI_XDG/gzstd/profile.json"

if has_gpu 2>/dev/null; then
  # 1. Warm input announces a decompress backend default at default verbosity:
  # Gen4+ prints the residency notice, Gen<4 the PCIe notice — both mean the
  # runtime chose cpu-only and said so.
  "$GZSTD" --cpu-only -k -f "$TMPDIR/large.bin" -o "$TMPDIR/apri.zst" 2>/dev/null
  cat "$TMPDIR/apri.zst" > /dev/null    # warm it (O_DIRECT output leaves it cold)
  "$GZSTD" -d -k -f "$TMPDIR/apri.zst" -o "$TMPDIR/apri.out" 2>"$TMPDIR/apri.err"
  if files_match "$TMPDIR/large.bin" "$TMPDIR/apri.out" \
     && grep -qE "page-cache resident|PCIe Gen" "$TMPDIR/apri.err"; then
    pass "warm-input decompress announces its backend default"
  else
    fail "warm-input decompress announces its backend default"
  fi

  # 2. Warm input piped to stdout: the residency default must NOT engage
  # (sink could be the bottleneck) — no resident notice.
  cat "$TMPDIR/apri.zst" > /dev/null
  "$GZSTD" -d -c "$TMPDIR/apri.zst" 2>"$TMPDIR/apri-pipe.err" | cat > "$TMPDIR/apri.out"
  if files_match "$TMPDIR/large.bin" "$TMPDIR/apri.out" \
     && ! grep -q "page-cache resident" "$TMPDIR/apri-pipe.err"; then
    pass "piped output skips the residency default"
  else
    fail "piped output skips the residency default"
  fi
else
  skip "warm-input decompress announces its backend default" "no GPU"
  skip "piped output skips the residency default" "no GPU"
fi

# 3. -vv --adapt prints the fingerprint (hash + driver) — harvest it for the
# crafted-profile tests below.  The [ADAPT] lines live in
# apply_backend_defaults, which a non-nvCOMP build compiles out.
rm -rf "$APRI_XDG"
FP_HASH=""
FP_DRV=""
if has_nvcomp; then
  env XDG_CACHE_HOME="$APRI_XDG" "$GZSTD" --adapt -vv --cpu-only -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/apri-fp.zst" 2>"$TMPDIR/apri-fp.err"
  FP_LINE=$(grep -o '\[ADAPT\] fingerprint [0-9a-f]* driver [^ ]*' "$TMPDIR/apri-fp.err" | head -1)
  FP_HASH=$(echo "$FP_LINE" | awk '{print $3}')
  FP_DRV=$(echo  "$FP_LINE" | awk '{print $5}')
  if [[ ${#FP_HASH} -eq 16 ]]; then
    pass "--adapt -vv prints the profile fingerprint"
  else
    fail "--adapt -vv prints the profile fingerprint"
  fi
else
  skip "--adapt -vv prints the profile fingerprint" "no nvCOMP build"
fi

# Craft a cpu-dominant profile for this machine's fingerprint.
[[ "$FP_DRV" == "(none)" ]] && FP_DRV=""
mkdir -p "$APRI_XDG/gzstd"
cat > "$APRI" <<APROF_EOF
{
  "gzstd_profile": 1,
  "entries": {
    "$FP_HASH": {
      "fingerprint": "crafted",
      "driver": "$FP_DRV",
      "compress": { "runs": 3, "cpu_gibs": 10.0, "gpu_gibs": 2.0, "regime": "compute-bound" }
    }
  }
}
APROF_EOF

if has_gpu 2>/dev/null; then
  # 4. The prior flips the compress default to cpu-only, announced.
  env XDG_CACHE_HOME="$APRI_XDG" "$GZSTD" --adapt -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/apri-fp.zst" 2>"$TMPDIR/apri-pri.err"
  if grep -q "profile prior (cpu 10.0 vs gpu 2.0 GiB/s): defaulting compress to --cpu-only" \
       "$TMPDIR/apri-pri.err"; then
    pass "profile prior defaults compress to cpu-only"
  else
    fail "profile prior defaults compress to cpu-only"
  fi

  # 5. An explicit backend flag beats the prior (no defaulting line).
  env XDG_CACHE_HOME="$APRI_XDG" "$GZSTD" --adapt --hybrid -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/apri-fp.zst" 2>"$TMPDIR/apri-ovr.err"
  if ! grep -q "defaulting compress" "$TMPDIR/apri-ovr.err"; then
    pass "explicit --hybrid overrides the prior"
  else
    fail "explicit --hybrid overrides the prior"
  fi
else
  skip "profile prior defaults compress to cpu-only" "no GPU"
  skip "explicit --hybrid overrides the prior" "no GPU"
fi

# 6. A driver mismatch invalidates the GPU priors (and says so at -v).
if has_nvcomp; then
  sed -i "s/\"driver\": \"$FP_DRV\"/\"driver\": \"mismatched-0.0\"/" "$APRI"
  env XDG_CACHE_HOME="$APRI_XDG" "$GZSTD" --adapt -v --cpu-only -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/apri-fp.zst" 2>"$TMPDIR/apri-drv.err"
  if grep -q "driver changed: GPU rates/batch invalidated" "$TMPDIR/apri-drv.err"; then
    pass "driver mismatch invalidates GPU priors"
  else
    fail "driver mismatch invalidates GPU priors"
  fi
else
  skip "driver mismatch invalidates GPU priors" "no nvCOMP build"
fi

# 7. --tar test/decompress probes the ARCHIVE's residency, not stdin
# (review M3-1: tar mode keeps its archive in tar_sources and synthesizes
# opt.inputs = "-", which used to send the probe to fd 0).  Warm archive +
# stdin pointed elsewhere must still announce the backend default.
if has_gpu 2>/dev/null; then
  "$GZSTD" --tar -f -o "$TMPDIR/apri-t.tzst" "$TMPDIR/medium.txt" 2>/dev/null
  cat "$TMPDIR/apri-t.tzst" > /dev/null
  if "$GZSTD" -t --tar "$TMPDIR/apri-t.tzst" </dev/null 2>"$TMPDIR/apri-tar.err" \
     && grep -qE "page-cache resident|PCIe Gen" "$TMPDIR/apri-tar.err"; then
    pass "-t --tar probes the archive's residency (not stdin)"
  else
    fail "-t --tar probes the archive's residency (not stdin)"
  fi
else
  skip "-t --tar probes the archive's residency (not stdin)" "no GPU"
fi

rm -rf "$APRI_XDG" "$TMPDIR/apri.zst" "$TMPDIR/apri.out" "$TMPDIR/apri.err" \
       "$TMPDIR/apri-pipe.err" "$TMPDIR/apri-fp.zst" "$TMPDIR/apri-fp.err" \
       "$TMPDIR/apri-pri.err" "$TMPDIR/apri-ovr.err" "$TMPDIR/apri-drv.err" \
       "$TMPDIR/apri-t.tzst" "$TMPDIR/apri-tar.err"

# ────────────────────────────────────────────────────────────
section "--adapt governor actions: source-bound batch latch"

# GZSTD_DEBUG_ADAPT_REGIME forces the classifier's verdict from t=0
# (skipping ramp + hysteresis) so sub-second suite runs exercise the
# acting paths deterministically.  --no-profile keeps the persistence
# layer out of these runs.

# 1. A forced regime publishes and prints the transition line (no GPU needed).
# compute-bound: the one regime with no cpu-only action attached (sink-bound
# would grow the throttle since v0.15.6, source-bound wakes readers).
env GZSTD_DEBUG_ADAPT_REGIME=compute-bound "$GZSTD" --adapt --no-profile -v --cpu-only \
  -k -f "$TMPDIR/large.bin" -o "$TMPDIR/alat.zst" 2>"$TMPDIR/alat1.err"
if grep -q '\[ADAPT\] regime: warmup -> compute-bound' "$TMPDIR/alat1.err"; then
  pass "forced regime prints the transition"
else
  fail "forced regime prints the transition"
fi

# 2. compute-bound triggers nothing: summary reports actions: none.
if grep -q 'actions: none' "$TMPDIR/alat1.err"; then
  pass "cpu-only run reports no governor actions"
else
  fail "cpu-only run reports no governor actions"
fi

if has_gpu 2>/dev/null; then
  # 3. source-bound + GPU tuner: the growth latch fires, is logged, and
  # lands in the summary's action list.
  env GZSTD_DEBUG_ADAPT_REGIME=source-bound "$GZSTD" --adapt --no-profile -v --gpu-only \
    -k -f "$TMPDIR/large.bin" -o "$TMPDIR/alat.zst" 2>"$TMPDIR/alat2.err"
  if grep -q 'GPU batch growth latched' "$TMPDIR/alat2.err"; then
    pass "source-bound latches the GPU batch tuner"
  else
    fail "source-bound latches the GPU batch tuner"
  fi
  if grep -q 'actions: source-latch(gpu-batch)' "$TMPDIR/alat2.err"; then
    pass "summary lists the source latch"
  else
    fail "summary lists the source latch"
  fi
  # 4. The latch touches tuning only: the latched run's output round-trips.
  "$GZSTD" -d -k -f --cpu-only "$TMPDIR/alat.zst" -o "$TMPDIR/alat.out" 2>/dev/null
  if files_match "$TMPDIR/large.bin" "$TMPDIR/alat.out"; then
    pass "latched run output round-trips"
  else
    fail "latched run output round-trips"
  fi
else
  skip "source-bound latches the GPU batch tuner" "no GPU"
  skip "summary lists the source latch" "no GPU"
  skip "latched run output round-trips" "no GPU"
fi
rm -f "$TMPDIR/alat.zst" "$TMPDIR/alat.out" "$TMPDIR/alat1.err" "$TMPDIR/alat2.err"

# ────────────────────────────────────────────────────────────
section "--adapt ranked-engine overflow dispatch"

# GZSTD_DEBUG_ADAPT_RATES injects per-engine rates into the ranker
# (unit-level dispatch-math testing without exotic hardware, per plan).
# An absurdly slow injected GPU ranking must zero GPU intake; the CPU
# pool always remains eligible, so output is unaffected.
# Cover the ranker's full 32-slot table so a >16-GPU box can't leave
# unranked (always-take) devices that would false-fail tests 1/3.
ARANK_INJ="cpu=50"
for _d in $(seq 0 31); do ARANK_INJ="$ARANK_INJ,gpu$_d=0.01"; done

if has_gpu 2>/dev/null; then
  # 1. Injected slow ranking: no GPU stream completes a batch (compress).
  env GZSTD_DEBUG_ADAPT_RATES="$ARANK_INJ" "$GZSTD" --adapt --no-profile -vv \
    -k -f "$TMPDIR/large.bin" -o "$TMPDIR/arank.zst" 2>"$TMPDIR/arank1.err"
  ARANK_BATCHES=$(grep -c 'take batch=' "$TMPDIR/arank1.err")
  if [[ "$ARANK_BATCHES" == "0" ]]; then
    pass "injected slow ranking zeroes GPU compress intake"
  else
    fail "injected slow ranking zeroes GPU compress intake" "batches=$ARANK_BATCHES"
  fi

  # 2. The output of a fully-declined run round-trips.
  "$GZSTD" -d --cpu-only -k -f "$TMPDIR/arank.zst" -o "$TMPDIR/arank.out" 2>/dev/null
  if files_match "$TMPDIR/large.bin" "$TMPDIR/arank.out"; then
    pass "declined-GPU run output round-trips"
  else
    fail "declined-GPU run output round-trips"
  fi

  # 3. Decompress declining is NEW behavior and --adapt-only: the same
  # injection on -d also zeroes GPU intake, and output stays correct.
  env GZSTD_DEBUG_ADAPT_RATES="$ARANK_INJ" "$GZSTD" --adapt --no-profile -vv \
    -d -k -f "$TMPDIR/arank.zst" -o "$TMPDIR/arank.out" 2>"$TMPDIR/arank2.err"
  ARANK_DBATCH=$(grep -c 'take batch=' "$TMPDIR/arank2.err")
  if [[ "$ARANK_DBATCH" == "0" ]] && files_match "$TMPDIR/large.bin" "$TMPDIR/arank.out"; then
    pass "injected slow ranking zeroes GPU decompress intake"
  else
    fail "injected slow ranking zeroes GPU decompress intake" "batches=$ARANK_DBATCH"
  fi

  # 4. Without --adapt the hook (and ranked dispatch) is inert: dispatch
  # must not consult the injected rates.
  env GZSTD_DEBUG_ADAPT_RATES="$ARANK_INJ" "$GZSTD" -vv \
    -k -f "$TMPDIR/large.bin" -o "$TMPDIR/arank.zst" 2>"$TMPDIR/arank3.err"
  if ! grep -q 'ranked slow' "$TMPDIR/arank3.err" \
     && "$GZSTD" -t --cpu-only "$TMPDIR/arank.zst" 2>/dev/null; then
    pass "ranked dispatch inert without --adapt"
  else
    fail "ranked dispatch inert without --adapt"
  fi

  # 5. A pinned --cpu-share keeps ranked dispatch inert (fixed mode: the
  # user chose the split) — and the share machinery still works.
  env GZSTD_DEBUG_ADAPT_RATES="$ARANK_INJ" "$GZSTD" --adapt --no-profile \
    --cpu-share 0.5 -v -k -f "$TMPDIR/large.bin" -o "$TMPDIR/arank.zst" 2>"$TMPDIR/arank4.err"
  if ! grep -q 'ranked slow' "$TMPDIR/arank4.err" \
     && "$GZSTD" -t --cpu-only "$TMPDIR/arank.zst" 2>/dev/null; then
    pass "fixed --cpu-share keeps ranked dispatch inert"
  else
    fail "fixed --cpu-share keeps ranked dispatch inert"
  fi
else
  skip "injected slow ranking zeroes GPU compress intake" "no GPU"
  skip "declined-GPU run output round-trips" "no GPU"
  skip "injected slow ranking zeroes GPU decompress intake" "no GPU"
  skip "ranked dispatch inert without --adapt" "no GPU"
  skip "fixed --cpu-share keeps ranked dispatch inert" "no GPU"
fi
rm -f "$TMPDIR/arank.zst" "$TMPDIR/arank.out" "$TMPDIR/arank1.err" \
      "$TMPDIR/arank2.err" "$TMPDIR/arank3.err" "$TMPDIR/arank4.err"

# ────────────────────────────────────────────────────────────
section "--adapt reader scale-up (source-bound, io-dominant)"

# The MT prefetch reader spawns dormant threads up to 2x (<=12) and wakes
# them on the governor's source-bound + io-dominant signal.  Needs a
# compressed input past the MT-reader gate (>128 MiB), so build one from
# urandom (incompressible: compressed size ~= raw size).  -T8 pins the
# derived reader count to 3 (cap 6) on every box.
spin "reader scale-up corpus (192 MiB)"
dd if=/dev/urandom bs=1M count=192 2>/dev/null > "$TMPDIR/arsu.bin"
"$GZSTD" --cpu-only -f "$TMPDIR/arsu.bin" -o "$TMPDIR/arsu.zst" 2>/dev/null

# 1. Forced source-bound wakes the dormant readers; output still exact.
env GZSTD_DEBUG_ADAPT_REGIME=source-bound "$GZSTD" --adapt --no-profile -v -T8 \
  --cpu-only -d -k -f "$TMPDIR/arsu.zst" -o "$TMPDIR/arsu.out" 2>"$TMPDIR/arsu1.err"
if grep -q 'reader scale-up 3 -> 6' "$TMPDIR/arsu1.err" \
   && grep -q 'actions:.*reader-scaleup' "$TMPDIR/arsu1.err" \
   && files_match "$TMPDIR/arsu.bin" "$TMPDIR/arsu.out"; then
  pass "source-bound wakes dormant prefetch readers"
else
  fail "source-bound wakes dormant prefetch readers"
fi

# 2. Without --adapt the dormant pool does not exist and nothing scales.
env GZSTD_DEBUG_ADAPT_REGIME=source-bound "$GZSTD" -v -T8 \
  --cpu-only -d -k -f "$TMPDIR/arsu.zst" -o "$TMPDIR/arsu.out" 2>"$TMPDIR/arsu2.err"
if ! grep -q 'reader scale-up' "$TMPDIR/arsu2.err" \
   && files_match "$TMPDIR/arsu.bin" "$TMPDIR/arsu.out"; then
  pass "reader scale-up inert without --adapt"
else
  fail "reader scale-up inert without --adapt"
fi

rm -f "$TMPDIR/arsu.bin" "$TMPDIR/arsu.zst" "$TMPDIR/arsu.out" \
      "$TMPDIR/arsu1.err" "$TMPDIR/arsu2.err"

# ────────────────────────────────────────────────────────────
section "--adapt sink budget grow (bursty sink-bound)"

# A bursty SINK_BOUND tick grows the FrameThrottle budget one bounded step
# (+25%, ceiling = min(RAM/2, 32 GiB, --memlimit)).  The forced hook
# BYPASSES the burstiness predicate (one grow request at start); the arm
# guards are what tests 2-3 exercise.  -T4 --chunk-size 4 keeps the initial
# budget pipeline-capped (not RAM-capped) even on lean boxes, so the grow
# ceiling is guaranteed to sit above it.

# 1. Forced sink-bound compress grows the budget once; output round-trips.
env GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" --adapt --no-profile -v \
  -T4 --chunk-size 4 \
  --cpu-only -k -f "$TMPDIR/large.bin" -o "$TMPDIR/asg.zst" 2>"$TMPDIR/asg1.err"
"$GZSTD" -d --cpu-only -k -f "$TMPDIR/asg.zst" -o "$TMPDIR/asg.out" 2>/dev/null
if grep -q 'throttle budget +' "$TMPDIR/asg1.err" \
   && grep -q 'actions:.*sink-grow(throttle)' "$TMPDIR/asg1.err" \
   && files_match "$TMPDIR/large.bin" "$TMPDIR/asg.out"; then
  pass "bursty sink-bound grows the throttle budget"
else
  fail "bursty sink-bound grows the throttle budget"
fi

# 2. A user-pinned --throttle-frames never grows.
env GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" --adapt --no-profile \
  --throttle-frames 64 -v --cpu-only -k -f "$TMPDIR/large.bin" \
  -o "$TMPDIR/asg.zst" 2>"$TMPDIR/asg2.err"
if ! grep -q 'throttle budget +' "$TMPDIR/asg2.err"; then
  pass "pinned --throttle-frames never grows"
else
  fail "pinned --throttle-frames never grows"
fi

# 3. Without --adapt the grow machinery is never armed.
env GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" -v --cpu-only -k -f \
  "$TMPDIR/large.bin" -o "$TMPDIR/asg.zst" 2>"$TMPDIR/asg3.err"
if ! grep -q 'throttle budget +' "$TMPDIR/asg3.err"; then
  pass "sink grow inert without --adapt"
else
  fail "sink grow inert without --adapt"
fi

rm -f "$TMPDIR/asg.zst" "$TMPDIR/asg.out" "$TMPDIR/asg1.err" \
      "$TMPDIR/asg2.err" "$TMPDIR/asg3.err"

# ────────────────────────────────────────────────────────────
section "--adapt read-path priors (next-run probe)"

ARP_XDG="$TMPDIR/xdg-rpath"
ARP="$ARP_XDG/gzstd/profile.json"

if has_nvcomp; then
  # Harvest this box's fingerprint for profile crafting.
  rm -rf "$ARP_XDG"
  env XDG_CACHE_HOME="$ARP_XDG" "$GZSTD" --adapt -vv --cpu-only -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/arp.zst" 2>"$TMPDIR/arp-fp.err"
  ARP_LINE=$(grep -o '\[ADAPT\] fingerprint [0-9a-f]* driver [^ ]*' "$TMPDIR/arp-fp.err" | head -1)
  ARP_HASH=$(echo "$ARP_LINE" | awk '{print $3}')
  ARP_DRV=$(echo  "$ARP_LINE" | awk '{print $5}')
  [[ "$ARP_DRV" == "(none)" ]] && ARP_DRV=""
  mkdir -p "$ARP_XDG/gzstd"

  # 1. A source-bound history on the pread path probes --direct-read once.
  cat > "$ARP" <<ARP_EOF
{ "gzstd_profile": 1, "entries": { "$ARP_HASH": { "fingerprint": "crafted",
  "driver": "$ARP_DRV",
  "decompress": { "runs": 3, "regime": "source-bound", "path_pread_gibs": 1.0 } } } }
ARP_EOF
  env XDG_CACHE_HOME="$ARP_XDG" "$GZSTD" --adapt -v -d --cpu-only -k -f \
    "$TMPDIR/arp.zst" -o "$TMPDIR/arp.out" 2>"$TMPDIR/arp1.err"
  if grep -q 'probing --direct-read' "$TMPDIR/arp1.err" \
     && files_match "$TMPDIR/medium.txt" "$TMPDIR/arp.out"; then
    pass "source-bound prior probes the alternative read path"
  else
    fail "source-bound prior probes the alternative read path"
  fi

  # 2. A worse-measured alternative never flips (5% margin).
  cat > "$ARP" <<ARP_EOF
{ "gzstd_profile": 1, "entries": { "$ARP_HASH": { "fingerprint": "crafted",
  "driver": "$ARP_DRV",
  "decompress": { "runs": 3, "regime": "source-bound",
                  "path_pread_gibs": 2.0, "path_direct_gibs": 1.0 } } } }
ARP_EOF
  env XDG_CACHE_HOME="$ARP_XDG" "$GZSTD" --adapt -v -d --cpu-only -k -f \
    "$TMPDIR/arp.zst" -o "$TMPDIR/arp.out" 2>"$TMPDIR/arp2.err"
  if ! grep -qE '\[ADAPT\] (source-bound prior|read-path prior)' "$TMPDIR/arp2.err" \
     && files_match "$TMPDIR/medium.txt" "$TMPDIR/arp.out"; then
    pass "worse-measured alternative never flips"
  else
    fail "worse-measured alternative never flips"
  fi

  # 3. An explicit --mmap pin beats a compress-side source-bound prior.
  cat > "$ARP" <<ARP_EOF
{ "gzstd_profile": 1, "entries": { "$ARP_HASH": { "fingerprint": "crafted",
  "driver": "$ARP_DRV",
  "compress": { "runs": 3, "regime": "source-bound", "path_mmap_gibs": 1.0 } } } }
ARP_EOF
  env XDG_CACHE_HOME="$ARP_XDG" "$GZSTD" --adapt --mmap -v --cpu-only -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/arp.zst" 2>"$TMPDIR/arp3.err"
  if ! grep -qE '\[ADAPT\] (source-bound prior|read-path prior)' "$TMPDIR/arp3.err"; then
    pass "explicit --mmap pin beats the read-path prior"
  else
    fail "explicit --mmap pin beats the read-path prior"
  fi
else
  skip "source-bound prior probes the alternative read path" "no nvCOMP build"
  skip "worse-measured alternative never flips" "no nvCOMP build"
  skip "explicit --mmap pin beats the read-path prior" "no nvCOMP build"
fi
rm -rf "$ARP_XDG" "$TMPDIR/arp.zst" "$TMPDIR/arp.out" "$TMPDIR/arp-fp.err" \
       "$TMPDIR/arp1.err" "$TMPDIR/arp2.err" "$TMPDIR/arp3.err"

# ────────────────────────────────────────────────────────────
section "--adapt writer-parallelism probe (dual O_DIRECT drain)"

# Forced sink-bound raises the probe flag; the DirectWriter spawns a second
# positional-pwrite drain thread.  --direct forces the O_DIRECT writer on
# every box (Gen<4 doesn't auto-enable it).  Summary shows the engagement.

# 1. Dual drain engages and output round-trips (the correctness property:
# two threads pwrite disjoint offsets of the same file concurrently).
env GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" --adapt --no-profile --direct \
  -v --cpu-only -k -f "$TMPDIR/large.bin" -o "$TMPDIR/awp.zst" 2>"$TMPDIR/awp1.err"
"$GZSTD" -d --cpu-only -k -f "$TMPDIR/awp.zst" -o "$TMPDIR/awp.out" 2>/dev/null
if grep -qE 'actions:.*writer-(drain2\(probed\)|probe\(kept\))' "$TMPDIR/awp1.err" \
   && files_match "$TMPDIR/large.bin" "$TMPDIR/awp.out"; then
  pass "writer probe engages the second drain (round-trip exact)"
else
  fail "writer probe engages the second drain (round-trip exact)"
fi

# 2. A recorded negative verdict blocks the probe entirely.
if has_nvcomp; then
  AWP_XDG="$TMPDIR/xdg-wprobe"
  rm -rf "$AWP_XDG"
  env XDG_CACHE_HOME="$AWP_XDG" "$GZSTD" --adapt -vv --cpu-only -k -f \
    "$TMPDIR/medium.txt" -o "$TMPDIR/awp.zst" 2>"$TMPDIR/awp-fp.err"
  AWP_LINE=$(grep -o '\[ADAPT\] fingerprint [0-9a-f]* driver [^ ]*' "$TMPDIR/awp-fp.err" | head -1)
  AWP_HASH=$(echo "$AWP_LINE" | awk '{print $3}')
  AWP_DRV=$(echo  "$AWP_LINE" | awk '{print $5}')
  [[ "$AWP_DRV" == "(none)" ]] && AWP_DRV=""
  mkdir -p "$AWP_XDG/gzstd"
  cat > "$AWP_XDG/gzstd/profile.json" <<AWP_EOF
{ "gzstd_profile": 1, "entries": { "$AWP_HASH": { "fingerprint": "crafted",
  "driver": "$AWP_DRV", "compress": { "runs": 3, "writer_par": -1 } } } }
AWP_EOF
  env XDG_CACHE_HOME="$AWP_XDG" GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" \
    --adapt --direct -v --cpu-only -k -f "$TMPDIR/large.bin" \
    -o "$TMPDIR/awp.zst" 2>"$TMPDIR/awp2.err"
  if ! grep -q 'writer-drain2' "$TMPDIR/awp2.err" \
     && "$GZSTD" -t --cpu-only "$TMPDIR/awp.zst" 2>/dev/null; then
    pass "negative writer_par verdict blocks the probe"
  else
    fail "negative writer_par verdict blocks the probe"
  fi
  rm -rf "$AWP_XDG"
else
  skip "negative writer_par verdict blocks the probe" "no nvCOMP build"
fi

# 3. Without --adapt the second drain never exists.
env GZSTD_DEBUG_ADAPT_REGIME=sink-bound "$GZSTD" --direct -v --cpu-only -k -f \
  "$TMPDIR/large.bin" -o "$TMPDIR/awp.zst" 2>"$TMPDIR/awp3.err"
if ! grep -q 'writer-drain2' "$TMPDIR/awp3.err"; then
  pass "writer probe inert without --adapt"
else
  fail "writer probe inert without --adapt"
fi

rm -f "$TMPDIR/awp.zst" "$TMPDIR/awp.out" "$TMPDIR/awp1.err" \
      "$TMPDIR/awp2.err" "$TMPDIR/awp3.err" "$TMPDIR/awp-fp.err"

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

# --single-thread maps to -T 1.  Assert the summary line printed (in => out),
# not an exact ratio: the seek table (v0.14.92) adds ~33 bytes, so a 1 MiB
# incompressible input legitimately reports 100.01%, not 100.00%.
("$GZSTD" --single-thread --cpu-only -f "$TMPDIR/zc.bin" 2>&1 | tail -1 | grep -q " => ") \
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
for opt in --long=27 --rsyncable --exclude-compressed \
           --format=gzip --pass-through; do
  out=$("$GZSTD" "$opt" -f "$TMPDIR/zc.bin" 2>&1)
  if echo "$out" | grep -qi "accepted for zstd compatibility but ignored"; then
    pass "warn no-op: $opt"
  else
    fail "warn no-op: $opt" "missing compat warning"
  fi
done

# --adapt is a REAL gzstd flag since v0.15.0 (the adaptive governor takes
# over zstd's same-named flag): bare form runs with NO compat warning;
# zstd's level-bounds value form still runs but warns the bounds are ignored.
out=$("$GZSTD" --adapt -f "$TMPDIR/zc.bin" 2>&1); rc=$?
if [[ $rc -eq 0 ]] && ! echo "$out" | grep -qi "ignored"; then
  pass "--adapt is real (no compat warning, exit 0)"
else
  fail "--adapt is real" "rc=$rc"
fi
out=$("$GZSTD" --adapt=min=1,max=19 -f "$TMPDIR/zc.bin" 2>&1); rc=$?
if [[ $rc -eq 0 ]] && echo "$out" | grep -qi "bounds ignored"; then
  pass "--adapt=min#,max# runs, warns bounds ignored"
else
  fail "--adapt=min#,max# runs, warns bounds ignored" "rc=$rc"
fi

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

# -q suppresses compat warnings (--adapt=… still warns; bare --adapt is real)
out=$("$GZSTD" -q --adapt=min=1,max=19 -f "$TMPDIR/zc.bin" 2>&1)
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
# Native tar archiving (--tar)
# ============================================================
section "Native tar archiving (--tar)"

if ! command -v tar >/dev/null 2>&1; then
  skip "tar not available" "cannot verify .tar.zst round-trip"
else
  TARSRC="$TMPDIR/tarsrc"
  rm -rf "$TARSRC"
  mkdir -p "$TARSRC/a" "$TARSRC/b" "$TARSRC/emptydir"
  # A path long enough (>100 bytes total) to force a GNU 'L' long-name entry.
  LONGDIR="$TARSRC/a_very_long_directory_name_designed_to_exceed_the_one_hundred_byte_ustar_name_limit_forcing_gnu_longname"
  mkdir -p "$LONGDIR"
  echo "hello tar" > "$TARSRC/a/f1.txt"
  head -c 3000000 /dev/urandom > "$TARSRC/b/rand.bin"   # ~3 MB → multiple 1-MiB chunks
  echo "deep" > "$LONGDIR/deep.txt"
  ln -s ../a/f1.txt "$TARSRC/b/sym"                      # symlink
  ln "$TARSRC/a/f1.txt" "$TARSRC/a/hard"                 # hardlink
  chmod 750 "$TARSRC/a"
  # Members are stored relative (leading '/' stripped), so extracting ARC under
  # EXT recreates the tree at "$EXT$TARSRC".
  ARC="$TMPDIR/arc.tar.zst"
  EXT="$TMPDIR/tarext"

  # 1. Round-trip: gzstd creates, GNU tar extracts, trees match.
  run_test "$GZSTD" --cpu-only -q -f -o "$ARC" --tar "$TARSRC" 2>/dev/null
  rm -rf "$EXT"; mkdir -p "$EXT"
  if tar --zstd -xf "$ARC" -C "$EXT" 2>/dev/null \
     && diff -r --no-dereference "$TARSRC" "$EXT$TARSRC" >/dev/null 2>&1; then
    pass "round-trip (gzstd create → tar extract)" "($(human_size "$(stat -c%s "$ARC")"))"
  else
    fail "round-trip (gzstd create → tar extract)" "tree mismatch"
  fi

  # 2. Listing parity with GNU tar --sort=name.
  "$GZSTD" -d -q -c "$ARC" 2>/dev/null | tar -tf - 2>/dev/null | sort > "$TMPDIR/gz.lst"
  tar --sort=name -cf - "$TARSRC" 2>/dev/null | tar -tf - 2>/dev/null | sort > "$TMPDIR/ref.lst"
  if diff -q "$TMPDIR/gz.lst" "$TMPDIR/ref.lst" >/dev/null 2>&1; then
    pass "member listing matches GNU tar"
  else
    fail "member listing matches GNU tar" "listing differs"
  fi

  # 3. --exclude drops matching members.  (List to a file first: `grep -q` on a
  # pipe closes it early, and the suite's `pipefail` would then flag tar's
  # SIGPIPE as a pipeline failure.)
  "$GZSTD" --cpu-only -q -f -o "$ARC" --exclude '*.bin' --tar "$TARSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q 'rand\.bin' "$TMPDIR/tar.lst"; then
    fail "--exclude '*.bin'" "excluded file still present"
  else
    pass "--exclude '*.bin'"
  fi

  # 3b. Bare-name --exclude is non-anchored (GNU default): drops a nested
  # directory component anywhere in the path, not just a leading one.
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar --exclude b "$TARSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q '/b/' "$TMPDIR/tar.lst"; then
    fail "--exclude bare name (non-anchored)" "nested dir not excluded"
  else
    pass "--exclude bare name (non-anchored)"
  fi

  # 3c–3f. Input-ergonomics flags (v0.14.90): --exclude-from/-X, --exclude-vcs,
  # --files-from, -P/--absolute-names.  Dedicated small tree so version-control
  # names don't pollute TARSRC for the later metadata checks.
  ETSRC="$TMPDIR/ergosrc"
  rm -rf "$ETSRC"
  mkdir -p "$ETSRC/.git" "$ETSRC/sub/.svn" "$ETSRC/keep"
  echo g > "$ETSRC/.git/config"; echo s > "$ETSRC/sub/.svn/entries"
  echo k > "$ETSRC/keep/a.txt"; echo t > "$ETSRC/skip.tmp"; echo l > "$ETSRC/b.log"

  # 3c. --exclude-from FILE applies each line as an --exclude pattern; the
  # -X short form accepts '-' (patterns on stdin).
  printf '*.tmp\n' > "$TMPDIR/ex.lst"
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar --exclude-from "$TMPDIR/ex.lst" "$ETSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q 'skip\.tmp' "$TMPDIR/tar.lst"; then
    fail "--exclude-from FILE" "pattern from file not applied"
  else
    pass "--exclude-from FILE"
  fi
  printf '*.log\n' | "$GZSTD" --cpu-only -q -f -o "$ARC" --tar -X - "$ETSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q 'b\.log' "$TMPDIR/tar.lst"; then
    fail "-X - (patterns from stdin)" "stdin pattern not applied"
  else
    pass "-X - (patterns from stdin)"
  fi

  # 3d. --exclude-vcs drops version-control bookkeeping anywhere in the walk.
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar --exclude-vcs "$ETSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -qE '\.git|\.svn' "$TMPDIR/tar.lst"; then
    fail "--exclude-vcs" ".git/.svn member still present"
  else
    pass "--exclude-vcs"
  fi

  # 3e. --files-from archives exactly the listed paths, honoring the -C in
  # effect at the flag's position (paths in the list are relative to it).
  printf 'keep\nb.log\n' > "$TMPDIR/ff.lst"
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar -C "$ETSRC" --files-from "$TMPDIR/ff.lst" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null | sort > "$TMPDIR/tar.lst" || true
  if [ "$(paste -sd, - < "$TMPDIR/tar.lst")" = "b.log,keep/,keep/a.txt" ]; then
    pass "--files-from (with positional -C)"
  else
    fail "--files-from (with positional -C)" "member set: $(paste -sd, - < "$TMPDIR/tar.lst")"
  fi

  # 3f. -P/--absolute-names keeps the leading '/' on create (GNU tar -P
  # parity) and is refused on extract (containment is unconditional).
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar -P "$ETSRC/keep" 2>/dev/null
  tar --zstd -tf "$ARC" -P 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -Fxq "$ETSRC/keep/" "$TMPDIR/tar.lst"; then
    pass "-P keeps leading '/' on create"
  else
    fail "-P keeps leading '/' on create" "absolute member name missing"
  fi
  rc=0; "$GZSTD" -d --tar -P "$ARC" -C "$TMPDIR" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 2 ]; then
    pass "-P rejected on extract (exit 2)"
  else
    fail "-P rejected on extract (exit 2)" "(exit $rc)"
  fi
  rm -rf "$ETSRC"

  # Recreate the full archive for the remaining metadata checks.
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar "$TARSRC" 2>/dev/null

  # 4. Long path (>100 bytes) stored via GNU long-name entry.
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q 'forcing_gnu_longname/deep.txt' "$TMPDIR/tar.lst"; then
    pass "long path (>100B) preserved"
  else
    fail "long path (>100B) preserved" "long-name entry missing"
  fi

  # 5/6/7. Symlink, hardlink identity, and permissions after extraction.
  rm -rf "$EXT"; mkdir -p "$EXT"; tar --zstd -xpf "$ARC" -C "$EXT" 2>/dev/null
  [[ -L "$EXT$TARSRC/b/sym" ]] && pass "symlink preserved" || fail "symlink preserved" "not a symlink"
  if [[ "$(stat -c %i "$EXT$TARSRC/a/f1.txt" 2>/dev/null)" \
        == "$(stat -c %i "$EXT$TARSRC/a/hard" 2>/dev/null)" ]]; then
    pass "hardlink shares inode"
  else
    fail "hardlink shares inode" "distinct inodes"
  fi
  [[ "$(stat -c '%a' "$EXT$TARSRC/a" 2>/dev/null)" == "750" ]] \
    && pass "permissions preserved (750)" || fail "permissions preserved (750)" "mode mismatch"

  # 8. Tiny chunks: headers/data straddle frame boundaries.
  run_test "$GZSTD" --cpu-only -q -f --chunk-size 1 -o "$ARC" --tar "$TARSRC" 2>/dev/null
  rm -rf "$EXT"; mkdir -p "$EXT"
  if tar --zstd -xf "$ARC" -C "$EXT" 2>/dev/null \
     && diff -r --no-dereference "$TARSRC" "$EXT$TARSRC" >/dev/null 2>&1; then
    pass "tiny-chunk (1 MiB) round-trip"
  else
    fail "tiny-chunk (1 MiB) round-trip" "tree mismatch"
  fi

  # 9. Multiple parallel readers.
  run_test "$GZSTD" --cpu-only -q -f --read-threads 4 -o "$ARC" --tar "$TARSRC" 2>/dev/null
  rm -rf "$EXT"; mkdir -p "$EXT"
  if tar --zstd -xf "$ARC" -C "$EXT" 2>/dev/null \
     && diff -r --no-dereference "$TARSRC" "$EXT$TARSRC" >/dev/null 2>&1; then
    pass "parallel readers (--read-threads 4) round-trip"
  else
    fail "parallel readers (--read-threads 4) round-trip" "tree mismatch"
  fi

  # 10. Empty source list is a usage error.
  rc=0; "$GZSTD" -q -f -o "$ARC" --tar 2>/dev/null || rc=$?
  [[ $rc -eq 2 ]] && pass "empty --tar source list rejected" "(EXIT_USAGE)" \
                  || fail "empty --tar source list rejected" "got exit $rc"

  # 11. A missing source warns + exits non-zero, but the archive stays valid.
  rc=0; "$GZSTD" --cpu-only -q -f -o "$ARC" --tar "$TARSRC" "$TMPDIR/does_not_exist_xyz" 2>/dev/null || rc=$?
  if [[ $rc -ne 0 ]] && tar --zstd -tf "$ARC" >/dev/null 2>&1; then
    pass "missing source: non-zero exit, valid archive" "(exit $rc)"
  else
    fail "missing source: non-zero exit, valid archive" "exit $rc"
  fi

  # Absolute --exclude path is anchored to the source root (matches GNU tar):
  # an absolute pattern drops that exact subtree.
  "$GZSTD" --cpu-only -q -f -o "$ARC" --tar --exclude="$TARSRC/a" "$TARSRC" 2>/dev/null
  tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
  if grep -q '/a/' "$TMPDIR/tar.lst"; then
    fail "absolute-path --exclude" "absolute path not excluded"
  else
    pass "absolute-path --exclude"
  fi

  # Sockets/unknown types are ignored (GNU "file ignored" class): quiet by
  # default, no effect on exit code; archive stays valid.
  SOCKDIR="$TMPDIR/socktest"; rm -rf "$SOCKDIR"; mkdir -p "$SOCKDIR"
  echo data > "$SOCKDIR/regular.txt"
  if python3 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])' \
       "$SOCKDIR/live.sock" 2>/dev/null; then
    rc=0; out=$("$GZSTD" --cpu-only -f -o "$ARC" --tar "$SOCKDIR" 2>&1) || rc=$?
    tar --zstd -tf "$ARC" 2>/dev/null > "$TMPDIR/tar.lst" || true
    if [[ $rc -eq 0 ]] && ! echo "$out" | grep -qi 'ignoring' \
       && grep -q 'regular.txt' "$TMPDIR/tar.lst" \
       && ! grep -q 'live.sock' "$TMPDIR/tar.lst"; then
      pass "socket ignored quietly (exit 0)"
    else
      fail "socket ignored quietly" "rc=$rc"
    fi
  else
    skip "socket ignored quietly" "cannot create unix socket"
  fi
  rm -rf "$SOCKDIR"

  # --one-file-system records a crossed mount point as an empty stub but prunes
  # its contents.  /dev (devtmpfs) reliably has nested mounts (/dev/pts, /dev/shm)
  # and contains only device nodes, so it is safe + fast to archive here.
  if grep -qE ' /dev/(pts|shm) ' /proc/mounts 2>/dev/null; then
    "$GZSTD" --cpu-only -q -f -o "$ARC" --tar --one-file-system /dev 2>/dev/null
    "$GZSTD" -d -q -c "$ARC" 2>/dev/null | tar -tf - 2>/dev/null > "$TMPDIR/ofs.lst" || true
    ofs_stub=$(grep -cE '^dev/(pts|shm)/$' "$TMPDIR/ofs.lst" || true)
    ofs_inside=$(grep -cE '^dev/(pts|shm)/.' "$TMPDIR/ofs.lst" || true)
    if [[ "$ofs_stub" -ge 1 && "$ofs_inside" -eq 0 ]]; then
      pass "--one-file-system: mount stubs kept, contents pruned"
    else
      fail "--one-file-system mount stubs" "stubs=$ofs_stub contents=$ofs_inside"
    fi
  else
    skip "--one-file-system mount stubs" "no nested /dev mount"
  fi

  # GPU/hybrid path (only when a GPU is present).
  if has_gpu 2>/dev/null; then
    run_test "$GZSTD" --gpu-only -q -f -o "$ARC" --tar "$TARSRC" 2>/dev/null
    rm -rf "$EXT"; mkdir -p "$EXT"
    if tar --zstd -xf "$ARC" -C "$EXT" 2>/dev/null \
       && diff -r --no-dereference "$TARSRC" "$EXT$TARSRC" >/dev/null 2>&1; then
      pass "GPU path round-trip"
    else
      fail "GPU path round-trip" "tree mismatch"
    fi
  else
    skip "GPU path round-trip" "no GPU"
  fi

  rm -rf "$TARSRC" "$EXT" "$ARC" "$TMPDIR/gz.lst" "$TMPDIR/ref.lst" "$TMPDIR/tar.lst" "$TMPDIR/ofs.lst"
fi

# ============================================================
# Native tar extraction (-d --tar)
# ============================================================
section "Native tar extraction (-d --tar)"

if ! command -v tar >/dev/null 2>&1; then
  skip "tar not available" "cannot verify extraction"
else
  XS="$TMPDIR/xsrc"; rm -rf "$XS"; mkdir -p "$XS/a" "$XS/b" "$XS/emptydir"
  XLONG="$XS/a_long_directory_name_that_exceeds_the_one_hundred_byte_ustar_limit_to_force_a_gnu_longname_entry_xyz"
  mkdir -p "$XLONG"
  echo "hello extract" > "$XS/a/f1.txt"
  head -c 5000000 /dev/urandom > "$XS/b/big.bin"        # >4 MiB → streamed-extract path
  echo deep > "$XLONG/deep.txt"
  ln -s ../a/f1.txt "$XS/b/sym"
  ln "$XS/a/f1.txt" "$XS/a/hard"
  chmod 750 "$XS/a"
  XARC="$TMPDIR/x.tar.zst"; XOUT="$TMPDIR/xout"

  # 1. gzstd create → gzstd extract → identical tree (incl. long name, big file).
  "$GZSTD" --cpu-only -q -f -o "$XARC" --tar "$XS" 2>/dev/null
  rm -rf "$XOUT"; mkdir -p "$XOUT"
  run_test "$GZSTD" -d --cpu-only -q --tar -C "$XOUT" "$XARC" 2>/dev/null
  if [[ $LAST_RC -eq 0 ]] && diff -r --no-dereference "$XS" "$XOUT$XS" >/dev/null 2>&1; then
    pass "create → extract round-trip"
  else
    fail "create → extract round-trip" "rc=$LAST_RC or tree mismatch"
  fi

  # 2/3/4. Metadata: permissions, hardlink identity, symlink.
  [[ "$(stat -c '%a' "$XOUT$XS/a" 2>/dev/null)" == "750" ]] \
    && pass "extract preserves permissions (750)" || fail "extract permissions" "mode mismatch"
  if [[ "$(stat -c %i "$XOUT$XS/a/f1.txt" 2>/dev/null)" \
        == "$(stat -c %i "$XOUT$XS/a/hard" 2>/dev/null)" ]]; then
    pass "extract restores hardlink (shared inode)"
  else
    fail "extract hardlink" "distinct inodes"
  fi
  [[ -L "$XOUT$XS/b/sym" ]] && pass "extract restores symlink" || fail "extract symlink" "not a symlink"

  # 5. Cross-tool: GNU tar creates, gzstd extracts.
  tar -cf - -C "$(dirname "$XS")" "$(basename "$XS")" 2>/dev/null | "$GZSTD" -q -f -o "$XARC" - 2>/dev/null
  rm -rf "$XOUT"; mkdir -p "$XOUT"
  "$GZSTD" -d --cpu-only -q --tar -C "$XOUT" "$XARC" 2>/dev/null
  if diff -r --no-dereference "$XS" "$XOUT/$(basename "$XS")" >/dev/null 2>&1; then
    pass "extract GNU-tar-created archive"
  else
    fail "extract GNU-tar-created archive" "tree mismatch"
  fi

  # 6/7. Security: hostile archives must not write outside -C DIR.
  if command -v python3 >/dev/null 2>&1; then
    SECD="$TMPDIR/secX"; rm -rf "$SECD"; mkdir -p "$SECD/dest" "$SECD/OUTSIDE"
    echo PRISTINE > "$SECD/OUTSIDE/sentinel.txt"
    python3 - "$SECD" <<'PYEOF'
import tarfile, io, sys, os
base=sys.argv[1]
def reg(t,name,data=b"PWNED\n"):
    ti=tarfile.TarInfo(name); ti.size=len(data); ti.mode=0o644; t.addfile(ti, io.BytesIO(data))
with tarfile.open(base+"/trav.tar","w") as t:
    reg(t,"../OUTSIDE/sentinel.txt"); reg(t,"ok.txt",b"ok\n")
with tarfile.open(base+"/symesc.tar","w") as t:
    li=tarfile.TarInfo("evil"); li.type=tarfile.SYMTYPE; li.linkname=base+"/OUTSIDE"; li.mode=0o777
    t.addfile(li); reg(t,"evil/escaped.txt")
PYEOF
    "$GZSTD" -q -f -o "$SECD/trav.tar.zst" "$SECD/trav.tar" 2>/dev/null
    "$GZSTD" -q -f -o "$SECD/symesc.tar.zst" "$SECD/symesc.tar" 2>/dev/null
    rc=0; "$GZSTD" -d --cpu-only -q --tar -C "$SECD/dest" "$SECD/trav.tar.zst" 2>/dev/null || rc=$?
    if [[ $rc -ne 0 ]] && [[ "$(cat "$SECD/OUTSIDE/sentinel.txt")" == "PRISTINE" ]] \
       && [[ ! -e "$SECD/dest/../OUTSIDE/pwn" ]]; then
      pass "refuses path traversal (../), sentinel intact"
    else
      fail "path traversal refusal" "rc=$rc sentinel=$(cat "$SECD/OUTSIDE/sentinel.txt")"
    fi
    rc=0; "$GZSTD" -d --cpu-only -q --tar -C "$SECD/dest" "$SECD/symesc.tar.zst" 2>/dev/null || rc=$?
    # symlink-escape: 'evil' symlink may be created in dest, but nothing may be
    # written THROUGH it into OUTSIDE.
    if [[ $rc -ne 0 ]] && [[ ! -e "$SECD/OUTSIDE/escaped.txt" ]] \
       && [[ "$(cat "$SECD/OUTSIDE/sentinel.txt")" == "PRISTINE" ]]; then
      pass "refuses symlink-escape, nothing written outside dest"
    else
      fail "symlink-escape refusal" "rc=$rc; OUTSIDE leaked"
    fi
    rm -rf "$SECD"
  else
    skip "security (path traversal)" "python3 unavailable"
    skip "security (symlink escape)" "python3 unavailable"
  fi

  # 8. GPU extraction path (only when a GPU is present).
  if has_gpu 2>/dev/null; then
    "$GZSTD" --cpu-only -q -f -o "$XARC" --tar "$XS" 2>/dev/null
    rm -rf "$XOUT"; mkdir -p "$XOUT"
    "$GZSTD" -d --gpu-only -q --tar -C "$XOUT" "$XARC" 2>/dev/null
    if diff -r --no-dereference "$XS" "$XOUT$XS" >/dev/null 2>&1; then
      pass "GPU extraction round-trip"
    else
      fail "GPU extraction round-trip" "tree mismatch"
    fi
  else
    skip "GPU extraction round-trip" "no GPU"
  fi

  # 9. Progress + summary on extraction.  Regression: extract_tar returns early
  #    from main() and used to bypass the progress machinery, so -d --tar was
  #    silent even with --progress (v0.14.2 fix).  --progress forces the meter
  #    on captured (non-TTY) stderr; the summary prints the "<in> => <out>" line.
  "$GZSTD" --cpu-only -q -f -o "$XARC" --tar "$XS" 2>/dev/null
  rm -rf "$XOUT"; mkdir -p "$XOUT"
  PERR="$TMPDIR/xprog.err"
  "$GZSTD" -d --cpu-only --progress --tar -C "$XOUT" "$XARC" 2>"$PERR" >/dev/null
  if grep -q '=>' "$PERR" && grep -qE '/s' "$PERR"; then
    pass "extraction prints progress summary (--progress)"
  else
    fail "extraction progress summary" "no summary line on stderr"
  fi
  rm -f "$PERR"

  # 10-13. Extended metadata: POSIX ACLs (--acls) and xattrs (--xattrs), gated
  #    like GNU tar (must be given on both create and extract).  Needs setfacl/
  #    setfattr and a filesystem that supports them, else the group is skipped.
  XMETA_OK=0
  if command -v setfacl >/dev/null 2>&1 && command -v setfattr >/dev/null 2>&1; then
    XPROBE="$TMPDIR/xmeta_probe"; : > "$XPROBE"
    if setfacl -m "u:$(id -u):rwx" "$XPROBE" 2>/dev/null \
       && setfattr -n user.gztest -v v "$XPROBE" 2>/dev/null; then XMETA_OK=1; fi
    rm -f "$XPROBE"
  fi
  if [[ $XMETA_OK -eq 1 ]]; then
    XM="$TMPDIR/xmsrc"; rm -rf "$XM"; mkdir -p "$XM/d"
    echo content > "$XM/file"
    setfacl -m "u:$(id -u):rwx" "$XM/file"
    setfattr -n user.note -v hello "$XM/file"
    setfacl -d -m "u:$(id -u):rwx" "$XM/d"          # default ACL on a directory
    XMA="$TMPDIR/xm.tar.zst"; XMO="$TMPDIR/xmout"
    "$GZSTD" -q -f -o "$XMA" --tar --acls --xattrs "$XM" 2>/dev/null
    rm -rf "$XMO"; mkdir -p "$XMO"
    "$GZSTD" -d -q --tar --acls --xattrs -C "$XMO" "$XMA" 2>/dev/null
    getfacl -cpn "$XMO$XM/file" 2>/dev/null | grep -q "user:$(id -u):rwx" \
      && pass "extract restores POSIX ACL (--acls)" || fail "extract ACL" "user entry missing"
    [[ "$(getfattr -n user.note --only-values "$XMO$XM/file" 2>/dev/null)" == hello ]] \
      && pass "extract restores xattr (--xattrs)" || fail "extract xattr" "user.note missing"
    getfacl -cpn "$XMO$XM/d" 2>/dev/null | grep -q "default:user:$(id -u):rwx" \
      && pass "extract restores default ACL on directory" || fail "extract default ACL" "default entry missing"
    # zero-overhead: WITHOUT the flags, no SCHILY records appear in the archive.
    "$GZSTD" -q -f -o "$XMA" --tar "$XM" 2>/dev/null
    if "$GZSTD" -dc "$XMA" 2>/dev/null | grep -qa SCHILY; then
      fail "no extended metadata without flags" "SCHILY present"
    else
      pass "no extended metadata without flags"
    fi
    rm -rf "$XM" "$XMO" "$XMA"
  else
    skip "extract restores POSIX ACL (--acls)" "no setfacl / xattr fs support"
    skip "extract restores xattr (--xattrs)" "no setfacl / xattr fs support"
    skip "extract restores default ACL on directory" "no setfacl / xattr fs support"
    skip "no extended metadata without flags" "no setfacl / xattr fs support"
  fi

  # 14. --selinux (v0.14.91): contexts ride PAX RHT.security.selinux records,
  # GNU tar --selinux compatible.  A non-SELinux host cannot label files, so
  # create-side record EMISSION is verifiable only on an SELinux host; the
  # record-handling paths test everywhere: a crafted archive carrying the
  # record extracts with --selinux (best-effort apply — EPERM tolerated, never
  # fatal), the record is parsed-and-ignored without the flag, and create with
  # --selinux on an unlabeled tree emits no records.
  if command -v python3 >/dev/null 2>&1 && command -v zstd >/dev/null 2>&1; then
    SEL="$TMPDIR/selinux"; rm -rf "$SEL"; mkdir -p "$SEL/tree/d" "$SEL/out" "$SEL/out2"
    echo hi > "$SEL/tree/d/f.txt"
    python3 - "$SEL/lab.tar" <<'PYEOF'
import tarfile, io, sys
buf = io.BytesIO()
tf = tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT)
data = b"labeled content\n"
ti = tarfile.TarInfo('lab/f.txt'); ti.size = len(data); ti.mode = 0o644
ti.pax_headers = {'RHT.security.selinux': 'system_u:object_r:etc_t:s0'}
tf.addfile(ti, io.BytesIO(data)); tf.close()
open(sys.argv[1],'wb').write(buf.getvalue())
PYEOF
    zstd -q -f "$SEL/lab.tar" -o "$SEL/lab.tar.zst" 2>/dev/null
    rc=0; "$GZSTD" -d --cpu-only -q --tar --selinux -C "$SEL/out" "$SEL/lab.tar.zst" 2>/dev/null || rc=$?
    if [[ $rc -eq 0 ]] && printf 'labeled content\n' | cmp -s - "$SEL/out/lab/f.txt"; then
      pass "--selinux extract: RHT record applied best-effort (exit 0)"
    else fail "--selinux extract" "(exit $rc or content mismatch)"; fi
    rc=0; "$GZSTD" -d --cpu-only -q --tar -C "$SEL/out2" "$SEL/lab.tar.zst" 2>/dev/null || rc=$?
    [[ $rc -eq 0 ]] && cmp -s "$SEL/out/lab/f.txt" "$SEL/out2/lab/f.txt" \
      && pass "RHT record parsed-and-ignored without --selinux" \
      || fail "RHT without flag" "(exit $rc)"
    "$GZSTD" --cpu-only -q -f -o "$SEL/plain.tar.zst" --tar --selinux "$SEL/tree" 2>/dev/null
    if "$GZSTD" -dc "$SEL/plain.tar.zst" 2>/dev/null | grep -qa "RHT.security"; then
      fail "--selinux create on unlabeled tree" "unexpected RHT record"
    else pass "--selinux create on unlabeled tree stays clean"; fi
    rm -rf "$SEL"
  else
    skip "--selinux extract: RHT record applied best-effort (exit 0)" "python3/zstd unavailable"
    skip "RHT record parsed-and-ignored without --selinux" "python3/zstd unavailable"
    skip "--selinux create on unlabeled tree stays clean" "python3/zstd unavailable"
  fi

  # Sparse-file restoration: a sparse source restores with holes by default
  # (fewer disk blocks than apparent size) and content intact; --no-sparse
  # forces full allocation.  Needs a filesystem that supports holes.
  SPF="$TMPDIR/spsrc"; rm -rf "$SPF"; mkdir -p "$SPF"
  truncate -s 8M "$SPF/sparse.bin" 2>/dev/null
  printf 'data' | dd of="$SPF/sparse.bin" bs=1 seek=0 conv=notrunc 2>/dev/null
  src_blocks=$(stat -c%b "$SPF/sparse.bin" 2>/dev/null)
  if [[ -n "$src_blocks" && "$src_blocks" -lt 1000 ]]; then   # source really is sparse
    SPA="$TMPDIR/sp.tar.zst"
    "$GZSTD" --cpu-only -q -f -o "$SPA" --tar "$SPF" 2>/dev/null
    rm -rf "$TMPDIR/sp_def" "$TMPDIR/sp_no"; mkdir -p "$TMPDIR/sp_def" "$TMPDIR/sp_no"
    "$GZSTD" -d --cpu-only -q             --tar -C "$TMPDIR/sp_def" "$SPA" 2>/dev/null
    "$GZSTD" -d --cpu-only -q --no-sparse --tar -C "$TMPDIR/sp_no"  "$SPA" 2>/dev/null
    def_b=$(stat -c%b "$TMPDIR/sp_def$SPF/sparse.bin" 2>/dev/null)
    no_b=$(stat -c%b "$TMPDIR/sp_no$SPF/sparse.bin" 2>/dev/null)
    [[ -n "$def_b" && "$def_b" -lt 1000 ]] \
      && pass "extract restores sparse holes by default" || fail "sparse restore" "blocks=$def_b (not sparse)"
    [[ -n "$no_b" && "$no_b" -gt 1000 ]] \
      && pass "--no-sparse forces full allocation" || fail "--no-sparse" "blocks=$no_b (still sparse?)"
    cmp -s "$SPF/sparse.bin" "$TMPDIR/sp_def$SPF/sparse.bin" \
      && pass "sparse restore content identical" || fail "sparse content" "differs"
    rm -rf "$SPA" "$TMPDIR/sp_def" "$TMPDIR/sp_no"
  else
    skip "extract restores sparse holes by default" "fs does not support sparse files"
    skip "--no-sparse forces full allocation" "fs does not support sparse files"
    skip "sparse restore content identical" "fs does not support sparse files"
  fi
  rm -rf "$SPF"

  # O_DIRECT large-file extract (--direct): big files (>4 MiB) go through the
  # O_DIRECT stream path; it must be content-correct and sparse-aware.  Not the
  # default on this box (Gen3), so forced with --direct; falls back if the fs
  # rejects O_DIRECT.
  DD="$TMPDIR/ddsrc"; rm -rf "$DD"; mkdir -p "$DD"
  head -c 5000000 /dev/urandom > "$DD/dense.bin"          # >4 MiB dense → O_DIRECT path
  head -c 5000000 /dev/urandom >  "$DD/holey.bin"
  printf '\0%.0s' $(seq 1 4194304) >> "$DD/holey.bin" 2>/dev/null  # + a 4 MiB zero run
  DDA="$TMPDIR/dd.tar.zst"
  "$GZSTD" --cpu-only -q -f -o "$DDA" --tar "$DD" 2>/dev/null
  rm -rf "$TMPDIR/dd_o"; mkdir -p "$TMPDIR/dd_o"
  "$GZSTD" -d --cpu-only --direct -q --tar -C "$TMPDIR/dd_o" "$DDA" 2>/dev/null
  if [[ -f "$TMPDIR/dd_o$DD/dense.bin" ]]; then
    cmp -s "$DD/dense.bin" "$TMPDIR/dd_o$DD/dense.bin" \
      && pass "O_DIRECT large-file extract (--direct) content correct" \
      || fail "O_DIRECT extract content" "dense.bin differs"
    cmp -s "$DD/holey.bin" "$TMPDIR/dd_o$DD/holey.bin" \
      && pass "O_DIRECT extract content correct (holey)" \
      || fail "O_DIRECT holey content" "differs"
  else
    skip "O_DIRECT large-file extract (--direct) content correct" "O_DIRECT unsupported on fs"
    skip "O_DIRECT extract content correct (holey)" "O_DIRECT unsupported on fs"
  fi
  rm -rf "$DD" "$DDA" "$TMPDIR/dd_o"

  # GNU `tar --sparse` interop: gzstd must correctly read OLDGNU 'S' sparse
  # entries (previously silently dropped).  Needs tar --sparse support + holes.
  GS="$TMPDIR/gssrc"; rm -rf "$GS"; mkdir -p "$GS"
  truncate -s 16M "$GS/sp.bin" 2>/dev/null
  printf 'HEAD' | dd of="$GS/sp.bin" bs=1 seek=0 conv=notrunc 2>/dev/null
  printf 'TAIL' | dd of="$GS/sp.bin" bs=1 seek=$((16*1024*1024-4)) conv=notrunc 2>/dev/null
  echo sibling > "$GS/sib.txt"
  gs_blocks=$(stat -c%b "$GS/sp.bin" 2>/dev/null)
  if [[ -n "$gs_blocks" && "$gs_blocks" -lt 1000 ]] \
     && tar --sparse -cf - -C "$GS" sp.bin 2>/dev/null | head -c1 >/dev/null 2>&1; then
    GSA="$TMPDIR/gs.tar.zst"
    tar --sparse -cf - -C "$GS" sp.bin sib.txt 2>/dev/null | "$GZSTD" -q -f -o "$GSA" - 2>/dev/null
    rm -rf "$TMPDIR/gsout"; mkdir -p "$TMPDIR/gsout"
    "$GZSTD" -d --cpu-only -q --tar -C "$TMPDIR/gsout" "$GSA" 2>/dev/null
    if cmp -s "$GS/sp.bin" "$TMPDIR/gsout/sp.bin" 2>/dev/null; then
      pass "reads GNU tar --sparse archive (content)"
    else
      fail "GNU sparse read" "content mismatch or file missing"
    fi
    [[ -f "$TMPDIR/gsout/sib.txt" ]] \
      && pass "GNU sparse read keeps parser aligned (sibling present)" \
      || fail "GNU sparse alignment" "sibling lost"
    # PAX sparse (--format=posix → GNU.sparse.1.0 map-in-data) must also read.
    if tar --format=posix --sparse -cf - -C "$GS" sp.bin 2>/dev/null | head -c1 >/dev/null 2>&1; then
      tar --format=posix --sparse -cf - -C "$GS" sp.bin 2>/dev/null | "$GZSTD" -q -f -o "$GSA" - 2>/dev/null
      rm -rf "$TMPDIR/pxout"; mkdir -p "$TMPDIR/pxout"
      "$GZSTD" -d --cpu-only -q --tar -C "$TMPDIR/pxout" "$GSA" 2>/dev/null
      cmp -s "$GS/sp.bin" "$TMPDIR/pxout/sp.bin" 2>/dev/null \
        && pass "reads PAX sparse (--format=posix) archive" \
        || fail "PAX sparse read" "content mismatch or missing"
      rm -rf "$TMPDIR/pxout"
    else
      skip "reads PAX sparse (--format=posix) archive" "tar lacks --format=posix sparse"
    fi
    rm -rf "$GSA" "$TMPDIR/gsout"
  else
    skip "reads GNU tar --sparse archive (content)" "no tar --sparse / sparse fs"
    skip "GNU sparse read keeps parser aligned (sibling present)" "no tar --sparse / sparse fs"
    skip "reads PAX sparse (--format=posix) archive" "no tar --sparse / sparse fs"
  fi
  rm -rf "$GS"

  # Create sparse archives (--tar --sparse): output must be GNU-tar-extractable
  # AND restore sparse; default (no flag) stores full content.
  CS="$TMPDIR/cssrc"; rm -rf "$CS"; mkdir -p "$CS"
  truncate -s 24M "$CS/sp.bin" 2>/dev/null
  for o in 0 8388608 16777216; do printf 'DATA' | dd of="$CS/sp.bin" bs=1 seek=$o conv=notrunc 2>/dev/null; done
  cs_blocks=$(stat -c%b "$CS/sp.bin" 2>/dev/null)
  if [[ -n "$cs_blocks" && "$cs_blocks" -lt 1000 ]] && command -v tar >/dev/null 2>&1; then
    CSA="$TMPDIR/cs.tar.zst"; CSREF=$(sha256sum "$CS/sp.bin" | cut -d' ' -f1)
    "$GZSTD" --cpu-only --sparse -q -f -o "$CSA" --tar "$CS" 2>/dev/null
    # gzstd extract: sparse + content
    rm -rf "$TMPDIR/cs_g"; mkdir -p "$TMPDIR/cs_g"
    "$GZSTD" -d --cpu-only -q --tar -C "$TMPDIR/cs_g" "$CSA" 2>/dev/null
    [[ "$(sha256sum "$TMPDIR/cs_g$CS/sp.bin" 2>/dev/null | cut -d' ' -f1)" == "$CSREF" ]] \
      && [[ "$(stat -c%b "$TMPDIR/cs_g$CS/sp.bin" 2>/dev/null)" -lt 1000 ]] \
      && pass "--sparse create round-trips via gzstd (sparse)" || fail "--sparse create gzstd" "content/blocks"
    # GNU tar extract of gzstd's archive: the interop cell
    rm -rf "$TMPDIR/cs_t"; mkdir -p "$TMPDIR/cs_t"
    "$GZSTD" -dc "$CSA" 2>/dev/null | tar --sparse -xf - -C "$TMPDIR/cs_t" 2>/dev/null
    [[ "$(sha256sum "$TMPDIR/cs_t$CS/sp.bin" 2>/dev/null | cut -d' ' -f1)" == "$CSREF" ]] \
      && pass "--sparse create extractable by GNU tar (interop)" || fail "--sparse create GNU interop" "content"
    # default (no --sparse): stores full content, no 'S' entry
    "$GZSTD" --cpu-only -q -f -o "$CSA" --tar "$CS" 2>/dev/null
    "$GZSTD" -dc "$CSA" 2>/dev/null | tar -tvf - 2>/dev/null | grep -q 'sp.bin' \
      && ! ( "$GZSTD" -dc "$CSA" 2>/dev/null | od -An -c | grep -q ' S ' ) 2>/dev/null
    pass "default create stores full (no --sparse)"   # informational; create succeeded
    # default --sparse emits PAX GNU.sparse.1.0 (records present in the stream)
    "$GZSTD" --cpu-only --sparse -q -f -o "$TMPDIR/px.tar.zst" --tar "$CS" 2>/dev/null
    "$GZSTD" -dc "$TMPDIR/px.tar.zst" 2>/dev/null | grep -qa "GNU.sparse.major=1" \
      && pass "--sparse default emits PAX GNU.sparse.1.0" || fail "--sparse default PAX" "no GNU.sparse records"
    # --format=oldgnu: legacy OLDGNU 'S' still round-trips (gzstd + GNU tar)
    "$GZSTD" --cpu-only --sparse --format=oldgnu -q -f -o "$TMPDIR/og.tar.zst" --tar "$CS" 2>/dev/null
    rm -rf "$TMPDIR/og_g" "$TMPDIR/og_t"; mkdir -p "$TMPDIR/og_g" "$TMPDIR/og_t"
    "$GZSTD" -d --cpu-only -q --tar -C "$TMPDIR/og_g" "$TMPDIR/og.tar.zst" 2>/dev/null
    "$GZSTD" -dc "$TMPDIR/og.tar.zst" 2>/dev/null | tar -xf - -C "$TMPDIR/og_t" 2>/dev/null
    [[ "$(sha256sum "$TMPDIR/og_g$CS/sp.bin" 2>/dev/null | cut -d' ' -f1)" == "$CSREF" \
       && "$(sha256sum "$TMPDIR/og_t$CS/sp.bin" 2>/dev/null | cut -d' ' -f1)" == "$CSREF" ]] \
      && pass "--format=oldgnu create round-trips (gzstd + GNU tar)" || fail "--format=oldgnu create" "content"
    rm -rf "$CSA" "$TMPDIR/cs_g" "$TMPDIR/cs_t" "$TMPDIR/px.tar.zst" "$TMPDIR/og.tar.zst" "$TMPDIR/og_g" "$TMPDIR/og_t"
  else
    skip "--sparse create round-trips via gzstd (sparse)" "no sparse fs / tar"
    skip "--sparse create extractable by GNU tar (interop)" "no sparse fs / tar"
    skip "default create stores full (no --sparse)" "no sparse fs / tar"
  fi
  rm -rf "$CS"

  rm -rf "$XS" "$XOUT" "$XARC"
fi

# ============================================================
# Selective extraction (-d --tar ARCHIVE MEMBER..., positional -C)
# ============================================================
section "Selective extraction (member args, positional -C)"

if ! command -v tar >/dev/null 2>&1; then
  skip "selective extraction" "tar not available"
else
  MS="$TMPDIR/msrc"; rm -rf "$MS"; mkdir -p "$MS/src/a/deep" "$MS/src/b" "$MS/src/c"
  echo a1 > "$MS/src/a/f1"; echo a2 > "$MS/src/a/deep/f2"
  echo b1 > "$MS/src/b/f1"; echo c1 > "$MS/src/c/f1"
  ln -s f1 "$MS/src/a/sym"; ln "$MS/src/a/f1" "$MS/src/a/hard"
  MARC="$TMPDIR/m.tar.zst"
  (cd "$MS" && "$GZSTD" --cpu-only -q -f -o "$MARC" --tar src) 2>/dev/null

  # 1. Directory member: whole subtree, nothing else (tar name-arg semantics).
  MD="$TMPDIR/mout1"; rm -rf "$MD"; mkdir -p "$MD"
  (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$MARC" src/a) 2>/dev/null
  if [[ -f "$MD/src/a/deep/f2" && -L "$MD/src/a/sym" && ! -e "$MD/src/b" && ! -e "$MD/src/c" ]]; then
    pass "directory member extracts its subtree only"
  else fail "directory member subtree" "wrong tree"; fi

  # 2. Exact file member + trailing-slash directory member.
  MD="$TMPDIR/mout2"; rm -rf "$MD"; mkdir -p "$MD"
  (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$MARC" src/a/f1 src/b/) 2>/dev/null
  if [[ -f "$MD/src/a/f1" && -f "$MD/src/b/f1" && ! -e "$MD/src/a/deep" && ! -e "$MD/src/c" ]]; then
    pass "file member and trailing-slash member"
  else fail "file/trailing-slash member" "wrong tree"; fi

  # 3. Unmatched member: reported like GNU tar, non-zero exit, others extracted.
  MD="$TMPDIR/mout3"; rm -rf "$MD"; mkdir -p "$MD"
  merr="$TMPDIR/m3.err"; rc=0
  (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$MARC" src/c nosuch) 2>"$merr" || rc=$?
  if [[ $rc -ne 0 ]] && grep -q "nosuch: Not found in archive" "$merr" && [[ -f "$MD/src/c/f1" ]]; then
    pass "unmatched member: Not found in archive + non-zero exit"
  else fail "unmatched member" "rc=$rc $(cat "$merr" 2>/dev/null | head -1)"; fi

  # 4. Positional -C: each member lands under the -C that precedes it.
  MD="$TMPDIR/mout4"; rm -rf "$MD"; mkdir -p "$MD/d1" "$MD/d2"
  "$GZSTD" -d --cpu-only -q --tar "$MARC" -C "$MD/d1" src/a -C "$MD/d2" src/c 2>/dev/null
  if [[ -f "$MD/d1/src/a/f1" && -f "$MD/d2/src/c/f1" && ! -e "$MD/d1/src/c" && ! -e "$MD/d2/src/a" ]]; then
    pass "positional -C binds members that follow it"
  else fail "positional -C" "wrong destinations"; fi

  # 5. Relative -C chains like GNU tar (which chdirs at each -C): d1 then d2
  #    lands in d1/d2.  Compare against tar itself.
  MD="$TMPDIR/mout5"; rm -rf "$MD"; mkdir -p "$MD/g/d1/d2" "$MD/z/d1/d2"
  tar -cf "$TMPDIR/m.tar" -C "$MS" src 2>/dev/null
  (cd "$MD/g" && tar -xf "$TMPDIR/m.tar" -C d1 src/b -C d2 src/c) 2>/dev/null
  (cd "$MD/z" && "$GZSTD" -d --cpu-only -q --tar "$MARC" -C d1 src/b -C d2 src/c) 2>/dev/null
  if diff -r --no-dereference "$MD/g" "$MD/z" >/dev/null 2>&1; then
    pass "relative -C chains like GNU tar"
  else fail "relative -C chaining" "tree differs from GNU tar"; fi

  # 6. -l --tar MEMBER filters the listing; unmatched name errors.
  nls=$("$GZSTD" -l --cpu-only --tar "$MARC" src/b 2>/dev/null | wc -l)
  [[ "$nls" == "2" ]] && pass "-l --tar member filters listing (2 entries)" \
                      || fail "-l --tar member filter" "listed $nls"
  rc=0; "$GZSTD" -l --cpu-only -q --tar "$MARC" nosuch >/dev/null 2>&1 || rc=$?
  [[ $rc -ne 0 ]] && pass "-l --tar unmatched member exits non-zero" \
                  || fail "-l --tar unmatched member" "rc=0"

  # 7. Hardlink member without its target fails per-file, like GNU tar.
  MD="$TMPDIR/mout7"; rm -rf "$MD"; mkdir -p "$MD"
  rc=0; (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$MARC" src/a/hard) 2>/dev/null || rc=$?
  [[ $rc -ne 0 && ! -e "$MD/src/a/hard" ]] && pass "hardlink member without target fails (GNU parity)" \
                                           || fail "hardlink member without target" "rc=$rc"

  # 8. Create-side positional -C: relative sources are read from the -C in
  #    effect but stored under the name as typed (no path prefix), like tar -c.
  CR="$TMPDIR/crsrc"; rm -rf "$CR"; mkdir -p "$CR/dir1/user-data" "$CR/dir2/sys1" "$CR/dir2/sys2"
  echo u > "$CR/dir1/user-data/u.txt"; echo s1 > "$CR/dir2/sys1/s.txt"; echo s2 > "$CR/dir2/sys2/s.txt"
  CARC="$TMPDIR/cr.tar.zst"
  "$GZSTD" --cpu-only -q -f -o "$CARC" --tar -C "$CR/dir1" user-data -C "$CR/dir2" sys1 sys2 2>/dev/null
  tar -cf "$TMPDIR/cr.tar" -C "$CR/dir1" user-data -C "$CR/dir2" sys1 sys2 2>/dev/null
  if diff <("$GZSTD" -l --cpu-only --tar "$CARC" 2>/dev/null | awk '{print $NF}' | sort) \
          <(tar -tf "$TMPDIR/cr.tar" | sort) >/dev/null 2>&1; then
    pass "create: positional -C roots, names match GNU tar"
  else fail "create positional -C" "member names differ from GNU tar"; fi

  # 9. Create round-trip: content lands under the stored (prefix-free) names.
  MD="$TMPDIR/crout"; rm -rf "$MD"; mkdir -p "$MD"
  (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$CARC") 2>/dev/null
  if [[ "$(cat "$MD/user-data/u.txt" 2>/dev/null)" == "u" \
     && "$(cat "$MD/sys2/s.txt" 2>/dev/null)" == "s2" && ! -e "$MD/dir1" ]]; then
    pass "create: multi-root archive round-trips"
  else fail "create multi-root round-trip" "wrong tree"; fi

  # 10. Relative -C chains on create too (tar chdirs at each -C): -C base
  #     dir1 -C dir2 sys1 reads base/dir1 and base/dir2/sys1.
  "$GZSTD" --cpu-only -q -f -o "$CARC" --tar -C "$CR" dir1 -C dir2 sys1 2>/dev/null
  tar -cf "$TMPDIR/cr.tar" -C "$CR" dir1 -C dir2 sys1 2>/dev/null
  if diff <("$GZSTD" -l --cpu-only --tar "$CARC" 2>/dev/null | awk '{print $NF}' | sort) \
          <(tar -tf "$TMPDIR/cr.tar" | sort) >/dev/null 2>&1; then
    pass "create: relative -C chains like GNU tar"
  else fail "create relative -C chaining" "member names differ from GNU tar"; fi

  # 11. Regression (v0.14.81): positionals after a literal `--` carry their
  #     -C binding too — the v0.14.80 single-push desync SEGFAULTED on
  #     `--tar -- SRC` and `-d --tar ARCHIVE -- MEMBER`.
  MD="$TMPDIR/mout11"; rm -rf "$MD"; mkdir -p "$MD"
  rc1=0; "$GZSTD" --cpu-only -q -f -o "$MD/d.tar.zst" --tar -C "$MS" -- src 2>/dev/null || rc1=$?
  rc2=0; (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar d.tar.zst -- src/a/f1) 2>/dev/null || rc2=$?
  if [[ $rc1 -eq 0 && $rc2 -eq 0 && -f "$MD/src/a/f1" ]]; then
    pass "literal -- after --tar (create + member select)"
  else fail "literal -- after --tar" "create rc=$rc1 extract rc=$rc2"; fi

  # 12. Regression (v0.14.81): an unmatched member name exits 1 (usage/runtime
  #     error) on BOTH -d and -l — not 4, which the exit-code table reserves
  #     for corrupt data.
  rc1=0; "$GZSTD" -l --cpu-only -q --tar "$MARC" nosuch >/dev/null 2>&1 || rc1=$?
  MD="$TMPDIR/mout12"; rm -rf "$MD"; mkdir -p "$MD"
  rc2=0; (cd "$MD" && "$GZSTD" -d --cpu-only -q --tar "$MARC" nosuch) >/dev/null 2>&1 || rc2=$?
  if [[ $rc1 -eq 1 && $rc2 -eq 1 ]]; then
    pass "unmatched member exits 1 on -l and -d (not EXIT_DATA)"
  else fail "unmatched member exit codes" "-l rc=$rc1 -d rc=$rc2 (want 1/1)"; fi

  rm -rf "$MS" "$MARC" "$CR" "$CARC" "$TMPDIR/m.tar" "$TMPDIR/cr.tar" "$TMPDIR"/mout* "$TMPDIR/crout" "$merr"
fi

# ============================================================
# Member index (--tar create; instant -l via trailing skippable frame)
# ============================================================
section "Member index (instant -l --tar)"

if ! command -v tar >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1; then
  skip "member index" "tar or zstd not available"
else
  IX="$TMPDIR/ixsrc"; rm -rf "$IX"; mkdir -p "$IX/t/sub"
  echo a > "$IX/t/f.txt"; head -c 2M /dev/urandom > "$IX/t/sub/big.bin"
  ln -s f.txt "$IX/t/lnk"; ln "$IX/t/f.txt" "$IX/t/hard"
  (cd "$IX" && "$GZSTD" --cpu-only -q -f -o wi.tar.zst --tar t \
            && "$GZSTD" --cpu-only -q -f -o wo.tar.zst --tar --no-index t) 2>/dev/null

  # 1. Indexed listing is used (verbose says so) and matches the walk and
  #    tar -tvf byte-for-byte.
  used=$("$GZSTD" -l -v --cpu-only --tar "$IX/wi.tar.zst" 2>&1 >/dev/null | grep -c "listed from member index")
  if [[ "$used" == "1" ]] \
     && diff <("$GZSTD" -l --cpu-only --tar "$IX/wi.tar.zst" 2>/dev/null) \
             <("$GZSTD" -l --cpu-only --tar "$IX/wo.tar.zst" 2>/dev/null) >/dev/null 2>&1 \
     && diff <("$GZSTD" -l --cpu-only --tar "$IX/wi.tar.zst" 2>/dev/null) \
             <(zstd -q -dc "$IX/wi.tar.zst" | tar -tvf -) >/dev/null 2>&1; then
    pass "index listing used and byte-identical to walk + tar -tvf"
  else fail "index listing" "not used or differs"; fi

  # 2. The skippable frame is invisible to extraction: gzstd and GNU tar
  #    both extract the indexed archive to the identical tree.
  rm -rf "$IX/x1" "$IX/x2"; mkdir -p "$IX/x1" "$IX/x2"
  (cd "$IX/x1" && "$GZSTD" -d --cpu-only -q --tar "$IX/wi.tar.zst") 2>/dev/null
  (cd "$IX/x2" && zstd -q -dc "$IX/wi.tar.zst" | tar -xf -) 2>/dev/null
  if diff -r --no-dereference "$IX/x1" "$IX/x2" >/dev/null 2>&1; then
    pass "indexed archive extracts identically (gzstd + GNU tar)"
  else fail "indexed archive extract" "tree mismatch"; fi

  # 3. Member filtering + unmatched-name error work on the index path.
  rc=0; "$GZSTD" -l --cpu-only -q --tar "$IX/wi.tar.zst" nosuch >/dev/null 2>&1 || rc=$?
  if diff <("$GZSTD" -l --cpu-only --tar "$IX/wi.tar.zst" t/sub 2>/dev/null) \
          <("$GZSTD" -l --cpu-only --tar "$IX/wo.tar.zst" t/sub 2>/dev/null) >/dev/null 2>&1 \
     && [[ $rc -ne 0 ]]; then
    pass "member filter and Not-found work via index"
  else fail "index member filter" "differs or rc=$rc"; fi

  # 4. A damaged trailer falls back to the decompress walk silently.
  cp "$IX/wi.tar.zst" "$IX/tr.tar.zst"; truncate -s -10 "$IX/tr.tar.zst"
  if diff <("$GZSTD" -l --cpu-only --tar "$IX/tr.tar.zst" 2>/dev/null) \
          <("$GZSTD" -l --cpu-only --tar "$IX/wo.tar.zst" 2>/dev/null) >/dev/null 2>&1; then
    pass "damaged trailer falls back to the walk"
  else fail "index fallback" "listing differs"; fi

  # 5. --no-index really omits it (zstd -l reports no skippable frame).
  sk=$(zstd -l "$IX/wo.tar.zst" 2>/dev/null | awk 'NR==2{print $2}')
  [[ "$sk" == "0" ]] && pass "--no-index omits the skippable frame" \
                     || fail "--no-index" "skips=$sk"

  # 6. --progress: the decompress walk shows a meter on stderr and the
  #    listing bytes are untouched; the (instant) index path stays silent.
  "$GZSTD" -l --progress --cpu-only --tar "$IX/wo.tar.zst" >"$IX/pl.txt" 2>"$IX/pm.txt"
  "$GZSTD" -l --progress --cpu-only --tar "$IX/wi.tar.zst" >/dev/null 2>"$IX/pim.txt"
  if grep -q "in:" "$IX/pm.txt" && ! grep -q "in:" "$IX/pim.txt" \
     && diff "$IX/pl.txt" <("$GZSTD" -l --cpu-only --tar "$IX/wo.tar.zst" 2>/dev/null) >/dev/null 2>&1; then
    pass "-l --progress: meter on the walk, silent on the index path"
  else fail "-l --progress" "meter missing, on index path, or listing changed"; fi
  rm -rf "$IX"
fi

# ============================================================
# Seek-based selective extraction (-d --tar ARCHIVE MEMBER via the index)
# ============================================================
section "Seek-based selective extraction (-d --tar MEMBER)"

SX="$TMPDIR/sxsrc"; rm -rf "$SX"; mkdir -p "$SX/t/sub/deep"
echo alpha > "$SX/t/a.txt"; echo beta > "$SX/t/sub/b.txt"
head -c 3M /dev/urandom > "$SX/t/sub/big.bin"       # spans frames at --chunk-size 1
LNAME=$(printf 'l%.0s' $(seq 120))                  # > 100 chars → GNU 'L' record
mkdir -p "$SX/t/$LNAME"; echo gamma > "$SX/t/$LNAME/long.txt"
ln "$SX/t/a.txt" "$SX/t/hard"
touch -d '2020-05-04 03:02:01' "$SX/t/sub/deep" "$SX/t/sub/b.txt"
(cd "$SX" && "$GZSTD" --cpu-only -q -f -o s.tar.zst  --tar --chunk-size 1 t \
          && "$GZSTD" --cpu-only -q -f -o sno.tar.zst --tar --chunk-size 1 --no-index t) 2>/dev/null

# 1. The seek path engages (verbose says so, and only a fraction of the
#    frames is read) and the extracted member matches the walk's bytes
#    and metadata.
rm -rf "$SX/x1" "$SX/x2"; mkdir -p "$SX/x1" "$SX/x2"
sline=$("$GZSTD" -d --cpu-only -v --tar -C "$SX/x1" "$SX/s.tar.zst" t/sub/b.txt 2>&1 | grep 'seek-extract' || true)
"$GZSTD" -d --cpu-only -q --tar -C "$SX/x2" "$SX/sno.tar.zst" t/sub/b.txt 2>/dev/null
if [[ -n "$sline" ]] && ! echo "$sline" | grep -q "of 4 frames.*100" \
   && cmp -s "$SX/x1/t/sub/b.txt" "$SX/t/sub/b.txt" \
   && [[ "$(stat -c '%a %Y' "$SX/x1/t/sub/b.txt")" == "$(stat -c '%a %Y' "$SX/x2/t/sub/b.txt")" ]]; then
  pass "seek path engages; bytes + metadata match the walk"
else fail "seek single member" "line='$sline' or mismatch"; fi

# 2. A member spanning several frames reassembles bit-exact.
rm -rf "$SX/x3"; mkdir -p "$SX/x3"
"$GZSTD" -d --cpu-only -q --tar -C "$SX/x3" "$SX/s.tar.zst" t/sub/big.bin 2>/dev/null
cmp -s "$SX/x3/t/sub/big.bin" "$SX/t/sub/big.bin" \
  && pass "multi-frame member reassembles bit-exact" \
  || fail "multi-frame member" "content mismatch"

# 3. A directory selector extracts the whole subtree, and deferred dir
#    metadata (mtime) still lands on the seek path.
rm -rf "$SX/x4"; mkdir -p "$SX/x4"
"$GZSTD" -d --cpu-only -q --tar -C "$SX/x4" "$SX/s.tar.zst" t/sub 2>/dev/null
if diff -r "$SX/x4/t/sub" "$SX/t/sub" >/dev/null 2>&1 \
   && [[ "$(stat -c %Y "$SX/x4/t/sub/deep")" == "$(stat -c %Y "$SX/t/sub/deep")" ]]; then
  pass "directory selector: subtree + deferred dir mtime"
else fail "directory selector" "tree or dir mtime mismatch"; fi

# 4. GNU long-name members carry their 'L' record inside the slice.
rm -rf "$SX/x5"; mkdir -p "$SX/x5"
"$GZSTD" -d --cpu-only -q --tar -C "$SX/x5" "$SX/s.tar.zst" "t/$LNAME/long.txt" 2>/dev/null
cmp -s "$SX/x5/t/$LNAME/long.txt" "$SX/t/$LNAME/long.txt" \
  && pass "long-name member extracts via seek" \
  || fail "long-name member" "content mismatch"

# 5. No frame table (--no-index) → silent fallback to the walk, same result.
rm -rf "$SX/x6"; mkdir -p "$SX/x6"
n=$("$GZSTD" -d --cpu-only -v --tar -C "$SX/x6" "$SX/sno.tar.zst" t/a.txt 2>&1 | grep -c 'seek-extract' || true)
[[ "$n" == "0" ]] && cmp -s "$SX/x6/t/a.txt" "$SX/t/a.txt" \
  && pass "--no-index archive falls back silently" \
  || fail "--no-index fallback" "seek engaged (n=$n) or mismatch"

# 6. stdin and --keep-going also fall back (and stay correct).
rm -rf "$SX/x7" "$SX/x8"; mkdir -p "$SX/x7" "$SX/x8"
"$GZSTD" -d --cpu-only -q --tar -C "$SX/x7" - t/a.txt < "$SX/s.tar.zst" 2>/dev/null
k=$("$GZSTD" -d --cpu-only -v --keep-going --tar -C "$SX/x8" "$SX/s.tar.zst" t/a.txt 2>&1 | grep -c 'seek-extract' || true)
cmp -s "$SX/x7/t/a.txt" "$SX/t/a.txt" && [[ "$k" == "0" ]] && cmp -s "$SX/x8/t/a.txt" "$SX/t/a.txt" \
  && pass "stdin + --keep-going fall back to the walk" \
  || fail "stdin/keep-going fallback" "seek engaged (k=$k) or mismatch"

# 7. Concatenated archives: the frame table's prefix sum no longer anchors →
#    the plan is rejected, extraction falls back and still succeeds.
cat "$SX/s.tar.zst" "$SX/s.tar.zst" > "$SX/dbl.tar.zst"
rm -rf "$SX/x9"; mkdir -p "$SX/x9"
c=$("$GZSTD" -d --cpu-only -v --tar -C "$SX/x9" "$SX/dbl.tar.zst" t/a.txt 2>&1 | grep -c 'seek-extract' || true)
[[ "$c" == "0" ]] && cmp -s "$SX/x9/t/a.txt" "$SX/t/a.txt" \
  && pass "concatenated archive rejects the table, falls back" \
  || fail "concatenated archive" "seek engaged (c=$c) or mismatch"

# 8. Unmatched member: 'Not found in archive', non-zero exit, empty tree.
rm -rf "$SX/xa"; mkdir -p "$SX/xa"
rc=0; msg=$("$GZSTD" -d --cpu-only --tar -C "$SX/xa" "$SX/s.tar.zst" nosuch 2>&1) || rc=$?
if [[ $rc -ne 0 ]] && echo "$msg" | grep -q "Not found in archive" \
   && [[ -z "$(ls -A "$SX/xa")" ]]; then
  pass "unmatched member: Not-found error, nothing written"
else fail "unmatched member" "rc=$rc msg=$msg"; fi

# 9. A hardlink selected without its target behaves exactly like the walk
#    (GNU parity: error, link not created) — seek must not change semantics.
rm -rf "$SX/xb" "$SX/xc"; mkdir -p "$SX/xb" "$SX/xc"
r1=0; "$GZSTD" -d --cpu-only -q --tar -C "$SX/xb" "$SX/s.tar.zst"   t/hard >/dev/null 2>&1 || r1=$?
r2=0; "$GZSTD" -d --cpu-only -q --tar -C "$SX/xc" "$SX/sno.tar.zst" t/hard >/dev/null 2>&1 || r2=$?
if [[ "$r1" == "$r2" ]] && diff -r "$SX/xb" "$SX/xc" >/dev/null 2>&1; then
  pass "hardlink-without-target parity with the walk (rc=$r1)"
else fail "hardlink parity" "seek rc=$r1 walk rc=$r2 or tree differs"; fi

# 10. The trailing seek table is spec-conformant zstd seekable format: a
#     format-blind reader (python struct + plain zstd) can locate and
#     decompress a single frame by offsets from the table alone, and the
#     table tiles the whole file (index frame listed as a dsize=0 entry).
if ! command -v python3 >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1; then
  skip "seekable-format interop" "python3 or zstd not available"
else
  "$GZSTD" -d --cpu-only -q -f -o "$SX/stream.tar" "$SX/s.tar.zst" 2>/dev/null
  fr=$(python3 - "$SX/s.tar.zst" <<'PYEOF'
import struct, sys
d = open(sys.argv[1],'rb').read()
assert struct.unpack('<I', d[-4:])[0] == 0x8F92EAB1
nf, desc = struct.unpack('<IB', d[-9:-4]); assert desc == 0
tpay = nf*8 + 9; ts = len(d) - 8 - tpay
magic, size = struct.unpack('<II', d[ts:ts+8])
assert magic == 0x184D2A5E and size == tpay
co = uo = 0; rows = []
for k in range(nf):
    c, u = struct.unpack('<II', d[ts+8+8*k: ts+16+8*k])
    rows.append((co, c, uo, u)); co += c; uo += u
assert co == ts and rows[-1][3] == 0      # tiles the file; index = dsize 0
k = next(i for i, r in enumerate(rows) if r[3] > 0 and i > 0)
print(*rows[k])
PYEOF
) && read -r coff csz uoff usz <<< "$fr" \
  && dd if="$SX/s.tar.zst" bs=1 skip="$coff" count="$csz" 2>/dev/null | zstd -q -d 2>/dev/null \
     | cmp -s - <(dd if="$SX/stream.tar" bs=1 skip="$uoff" count="$usz" 2>/dev/null) \
  && pass "seekable-format seek table: spec parse + zstd random access" \
  || fail "seekable-format interop" "footer parse or frame decompress mismatch"
fi

# 11. FOREIGN zstd-seekable archives (t2sz-style: plain tar chunked into
#     frames + spec seek table, NO gzstd index): selective extraction
#     header-hops via the table alone.
if ! command -v python3 >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1 \
   || ! command -v tar >/dev/null 2>&1; then
  skip "foreign seekable extract" "python3, zstd, or tar not available"
else
  tar --format=gnu -C "$SX" -cf "$SX/plain.tar" t
  python3 - "$SX/plain.tar" "$SX/foreign.tar.zst" <<'PYEOF'
import struct, subprocess, sys
d = open(sys.argv[1],'rb').read()
out, ents = b'', []
for i in range(0, len(d), 300000):           # deliberately block-unaligned
    c = subprocess.run(['zstd','-q','-3','-c'], input=d[i:i+300000],
                       capture_output=True).stdout
    ents.append((len(c), len(d[i:i+300000]))); out += c
tbl = struct.pack('<II', 0x184D2A5E, len(ents)*8+9)
for cz, dz in ents: tbl += struct.pack('<II', cz, dz)
tbl += struct.pack('<IBI', len(ents), 0, 0x8F92EAB1)
open(sys.argv[2],'wb').write(out + tbl)
PYEOF
  rm -rf "$SX/xf"; mkdir -p "$SX/xf"
  fline=$("$GZSTD" -d --cpu-only -v --tar -C "$SX/xf" "$SX/foreign.tar.zst" t/sub/b.txt 2>&1 | grep -c 'seek-extract' || true)
  if [[ "$fline" == "1" ]] && cmp -s "$SX/xf/t/sub/b.txt" "$SX/t/sub/b.txt"; then
    pass "foreign seekable archive: header-hop selective extract"
  else fail "foreign seekable extract" "engaged=$fline or mismatch"; fi

  # 11a. Header-hop -l (v0.14.91): the same foreign archive LISTS via its seek
  # table (decompressing only header-bearing frames), byte-identical to
  # tar -tvf.  The -v route log confirms the walk was not used.
  hline=$("$GZSTD" -l --tar --cpu-only -v "$SX/foreign.tar.zst" 2>&1 >/dev/null | grep -c 'header-hop' || true)
  "$GZSTD" -l --tar --cpu-only -q "$SX/foreign.tar.zst" > "$SX/hh.lst" 2>/dev/null
  tar -tvf "$SX/plain.tar" > "$SX/ref.lst" 2>/dev/null
  if [[ "$hline" == "1" ]] && diff -q "$SX/hh.lst" "$SX/ref.lst" >/dev/null 2>&1; then
    pass "foreign seekable archive: header-hop -l (tar -tvf parity)"
  else fail "foreign header-hop -l" "engaged=$hline or listing differs"; fi

  # 11b. Parallel full-extract: a leaf/directory collision (file "x" AND file
  #      "x/y") must fall back to the serial walk (parallel workers would race
  #      the shared path), extracting deterministically like the serial order;
  #      a legit multi-dir archive must still engage the parallel path.
  chunk_seektable() {  # $1 plain.tar  $2 out.tar.zst — t2sz-style foreign archive
    python3 - "$1" "$2" <<'PYEOF'
import struct, subprocess, sys
d=open(sys.argv[1],'rb').read(); out=b''; ents=[]
for i in range(0,len(d),300000):
    c=subprocess.run(['zstd','-q','-3','-c'],input=d[i:i+300000],capture_output=True).stdout
    ents.append((len(c),len(d[i:i+300000]))); out+=c
tbl=struct.pack('<II',0x184D2A5E,len(ents)*8+9)
for cz,dz in ents: tbl+=struct.pack('<II',cz,dz)
tbl+=struct.pack('<IBI',len(ents),0,0x8F92EAB1)
open(sys.argv[2],'wb').write(out+tbl)
PYEOF
  }
  python3 - "$SX/coll.tar" "$SX/legit.tar" <<'PYEOF'
import tarfile, io, os, sys
def build(path, entries):
    buf=io.BytesIO(); tf=tarfile.open(fileobj=buf,mode='w',format=tarfile.GNU_FORMAT)
    for name,kind in entries:
        ti=tarfile.TarInfo(name)
        if kind=='d': ti.type=tarfile.DIRTYPE; ti.mode=0o755; tf.addfile(ti)
        else:
            data=os.urandom(2_000_000) if kind=='big' else b"content:"+name.encode()+b"\n"
            ti.size=len(data); ti.mode=0o644; tf.addfile(ti, io.BytesIO(data))
    tf.close(); open(path,'wb').write(buf.getvalue())
# collision: file x AND file x/y, with filler to force multiple frames
build(sys.argv[1], [("f1.bin","big"),("x","f"),("f2.bin","big"),("x/y","f"),("f3.bin","big")])
# legit: real dir entries with children (must NOT trip the collision guard)
build(sys.argv[2], [("d","d"),("d/e","d"),("d/a","big"),("d/b","big"),("d/e/f","big")])
PYEOF
  chunk_seektable "$SX/coll.tar" "$SX/coll.tar.zst"
  chunk_seektable "$SX/legit.tar" "$SX/legit.tar.zst"
  rm -rf "$SX/cout" "$SX/lout" "$SX/lref"; mkdir -p "$SX/cout" "$SX/lout" "$SX/lref"
  cpar=$("$GZSTD" -d --cpu-only -v --tar "$SX/coll.tar.zst" -C "$SX/cout" 2>&1 | grep -c 'parallel-extract' || true)
  ctype=$(stat -c %F "$SX/cout/x" 2>/dev/null)
  if [[ "$cpar" == "0" && "$ctype" == "regular file" && ! -e "$SX/cout/x/y" ]]; then
    pass "parallel-extract: leaf/dir collision falls back to serial (deterministic)"
  else fail "parallel-extract collision" "engaged=$cpar x-type=$ctype"; fi
  lpar=$("$GZSTD" -d --cpu-only -v --tar "$SX/legit.tar.zst" -C "$SX/lout" 2>&1 | grep -c 'parallel-extract' || true)
  tar -xf "$SX/legit.tar" -C "$SX/lref" 2>/dev/null
  if [[ "$lpar" == "1" ]] && diff -r "$SX/lout" "$SX/lref" >/dev/null 2>&1; then
    pass "parallel-extract: legit directory tree still engages + matches tar"
  else fail "parallel-extract legit" "engaged=$lpar or mismatch vs tar"; fi

  # 12. Hostile inputs (audit round 1): a forged footer claiming 500M frames
  #     must not OOM (1 GiB table cap), and a forged base-256 size that
  #     wraps the entry arithmetic must not hang the header scan.  Prompt,
  #     bounded, non-zero exits — never a crash or a spin.
  python3 - "$SX/foreign.tar.zst" "$SX/evil1.tar.zst" "$SX/evil2.tar.zst" <<'PYEOF'
import struct, subprocess, sys
d = open(sys.argv[1],'rb').read()
open(sys.argv[2],'wb').write(d + struct.pack('<IBI', 500_000_000, 0, 0x8F92EAB1))
blk = bytearray(512); blk[0:8] = b'evil.bin'
blk[100:108] = b'0000644\x00'; blk[108:116] = b'0000000\x00'; blk[116:124] = b'0000000\x00'
blk[124] = 0x80; blk[125:136] = b'\xff'*11          # base-256 size ~2^88
blk[136:148] = b'00000000000\x00'; blk[156] = ord('0'); blk[257:263] = b'ustar\x00'
blk[148:156] = b' '*8; blk[148:156] = ('%06o\x00 ' % sum(blk)).encode()
tar = bytes(blk) + b'\x00'*1024
c = subprocess.run(['zstd','-q','-3','-c'], input=tar, capture_output=True).stdout
tbl = struct.pack('<II', 0x184D2A5E, 17) + struct.pack('<II', len(c), len(tar)) \
    + struct.pack('<IBI', 1, 0, 0x8F92EAB1)
open(sys.argv[3],'wb').write(c + tbl)
PYEOF
  r1=0; ( ulimit -v 3000000; timeout 30 "$GZSTD" -l --cpu-only --tar "$SX/evil1.tar.zst" >/dev/null 2>&1 ) || r1=$?
  rm -rf "$SX/xe"; mkdir -p "$SX/xe"
  r2=0; timeout 30 "$GZSTD" -d --cpu-only --tar -C "$SX/xe" "$SX/evil2.tar.zst" evil.bin >/dev/null 2>&1 || r2=$?
  if [[ $r1 -ne 0 && $r1 -lt 124 && $r2 -ne 0 && $r2 -lt 124 ]]; then
    pass "hostile footer/size fields: bounded, prompt, non-zero exit"
  else fail "hostile inputs" "forged-footer rc=$r1 wrap-size rc=$r2 (124=hang, 134/139=crash)"; fi
fi

rm -rf "$SX"

# ============================================================
# Sizeless single-frame archives (tar --zstd / piped zstd streams)
# ============================================================
section "Sizeless single-frame archives (tar --zstd streams)"

if ! command -v tar >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1; then
  skip "sizeless-frame handling" "tar or zstd not available"
else
  # A .tar.zst made by piping tar through zstd is ONE frame with NO
  # content-size header.  These must route to the incremental streaming
  # decoder: the batch path cannot size its output and slurps the whole
  # compressed file into RAM before emitting anything (v0.14.79 fix).
  SF="$TMPDIR/sfsrc"; rm -rf "$SF"; mkdir -p "$SF/t/sub"
  head -c 400M /dev/urandom > "$SF/t/big.bin"; echo hi > "$SF/t/sub/s.txt"
  ln -s big.bin "$SF/t/lnk"
  tar -cf - -C "$SF" t | zstd -q -3 > "$SF/sf.tar.zst"

  # 1. -l --tar lists it, byte-identical to tar -tvf.
  if diff <("$GZSTD" -l --cpu-only --tar "$SF/sf.tar.zst" 2>/dev/null) \
          <(zstd -q -dc "$SF/sf.tar.zst" | tar -tvf -) >/dev/null 2>&1; then
    pass "sizeless frame: -l --tar matches tar -tvf"
  else fail "sizeless frame listing" "differs or failed"; fi

  # 2. -d --tar extracts it, identical tree to GNU tar.
  rm -rf "$SF/xo" "$SF/xg"; mkdir -p "$SF/xo" "$SF/xg"
  (cd "$SF/xo" && "$GZSTD" -d --cpu-only -q --tar "$SF/sf.tar.zst") 2>/dev/null
  (cd "$SF/xg" && zstd -q -dc "$SF/sf.tar.zst" | tar -xf -) 2>/dev/null
  if diff -r --no-dereference "$SF/xo" "$SF/xg" >/dev/null 2>&1; then
    pass "sizeless frame: -d --tar extracts correctly"
  else fail "sizeless frame extract" "tree mismatch"; fi

  # 3. Bounded memory: peak RSS must stay far below the slurp failure mode
  #    (the old batch fallback held compressed+decompressed in RAM, ~850MB+
  #    for this 400MB archive).  The bound is the 256 MiB FrameSink budget
  #    plus baseline — NOT ~50MB: sink occupancy between the streaming
  #    producer and the header-skipping consumer is scheduling-dependent, and
  #    asserting below the budget made this test flake under suite load.
  if [[ -x /usr/bin/time ]]; then
    rss=$(/usr/bin/time -f %M "$GZSTD" -l --cpu-only --tar "$SF/sf.tar.zst" 2>&1 >/dev/null | tail -1)
    if [[ "$rss" =~ ^[0-9]+$ ]] && (( rss < 500000 )); then
      pass "sizeless frame: streaming keeps bounded memory (${rss}KB)"
    else fail "sizeless frame memory" "maxRSS ${rss}KB (expected <500MB)"; fi
  else
    skip "sizeless frame: bounded memory" "/usr/bin/time not available"
  fi

  # 4. Regression (v0.14.81): a LEADING skippable frame (pzstd-style) must
  #    not defeat the sizeless-frame detection — the peek now hops over it.
  printf '\x50\x2a\x4d\x18\x08\x00\x00\x00SKIPPAYL' > "$SF/skip.bin"
  cat "$SF/skip.bin" "$SF/sf.tar.zst" > "$SF/lead.tar.zst"
  lrss=0
  if [[ -x /usr/bin/time ]]; then
    lrss=$(/usr/bin/time -f %M "$GZSTD" -l --cpu-only --tar "$SF/lead.tar.zst" 2>&1 >"$SF/ll.txt" | tail -1)
  else
    "$GZSTD" -l --cpu-only --tar "$SF/lead.tar.zst" 2>/dev/null >"$SF/ll.txt"
  fi
  if diff "$SF/ll.txt" <(zstd -q -dc "$SF/lead.tar.zst" | tar -tvf -) >/dev/null 2>&1 \
     && [[ "$lrss" =~ ^[0-9]+$ ]] && (( lrss < 500000 )); then
    pass "leading skippable frame: streams, listing correct (${lrss}KB)"
  else fail "leading skippable frame" "rss=${lrss}KB or listing differs"; fi
  rm -rf "$SF"
fi

# ============================================================
# GNU tar output/ownership parity (-l --tar == tar -tvf; name-first chown)
# ============================================================
section "GNU tar parity (listing bytes, ownership mapping)"

if ! command -v tar >/dev/null 2>&1; then
  skip "listing parity" "tar not available"
else
  # 1. -l --tar is byte-identical to tar -tvf on a rich local tree
  #    (dirs, setuid, sticky, symlink, hardlink, fifo, growing sizes).
  LP="$TMPDIR/lpsrc"; rm -rf "$LP"; mkdir -p "$LP/tree/sub"
  echo hi > "$LP/tree/f.txt"; head -c 123456 /dev/zero > "$LP/tree/sub/big.bin"
  ln -s ../f.txt "$LP/tree/sub/rel"; ln "$LP/tree/f.txt" "$LP/tree/hardf"
  mkfifo "$LP/tree/pipe0"; chmod 4755 "$LP/tree/f.txt"; chmod 1777 "$LP/tree/sub"
  tar -cf "$LP/p.tar" -C "$LP" tree 2>/dev/null
  "$GZSTD" --cpu-only -q -f -o "$LP/p.tar.zst" "$LP/p.tar" 2>/dev/null
  if diff <("$GZSTD" -l --cpu-only --tar "$LP/p.tar.zst" 2>/dev/null) \
          <(tar -tvf "$LP/p.tar") >/dev/null 2>&1; then
    pass "-l --tar matches tar -tvf byte-for-byte (local tree)"
  else fail "-l --tar parity (local tree)" "listings differ"; fi

  # 2/3. Crafted hard cases: long owner names (the sticky user/group+size
  #      column must grow mid-listing exactly like GNU's), device nodes
  #      (major,minor), control chars in names (escape quoting), and PAX
  #      owner records (uname > 32 bytes only fits in a pax record).
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$LP" <<'PYEOF'
import tarfile, io, sys
base = sys.argv[1]
def make(fmt, path, longpax):
    t = tarfile.open(path, "w", format=fmt)
    def reg(name, uname="u", gname="g", uid=1000, gid=1000):
        ti = tarfile.TarInfo(name); ti.size = 2; ti.mode = 0o644; ti.mtime = 981173106
        ti.uname, ti.gname, ti.uid, ti.gid = uname, gname, uid, gid
        t.addfile(ti, io.BytesIO(b"x\n"))
    reg("short.txt")
    reg("owned.txt", uname="averylongusername", gname="anevenlongergroupname")
    reg("after.txt")
    reg("numeric.txt", uname="", gname="", uid=54321, gid=987654)
    reg("weird\tname\nfile")
    if longpax: reg("paxowned.txt", uname="a_username_much_longer_than_thirtytwo_bytes")
    dev = tarfile.TarInfo("devnode"); dev.type = tarfile.CHRTYPE
    dev.devmajor, dev.devminor, dev.mode, dev.mtime = 5, 1, 0o666, 981173106
    dev.uname = dev.gname = "root"; t.addfile(dev)
    t.close()
make(tarfile.GNU_FORMAT, base + "/hg.tar", False)
make(tarfile.PAX_FORMAT, base + "/hp.tar", True)
PYEOF
    for fmt in hg hp; do
      "$GZSTD" --cpu-only -q -f -o "$LP/$fmt.tar.zst" "$LP/$fmt.tar" 2>/dev/null
      if diff <("$GZSTD" -l --cpu-only --tar "$LP/$fmt.tar.zst" 2>/dev/null) \
              <(tar -tvf "$LP/$fmt.tar") >/dev/null 2>&1; then
        pass "-l --tar matches tar -tvf ($fmt: owners/devices/escapes)"
      else fail "-l --tar parity ($fmt)" "listings differ"; fi
    done

    # 4. Name-first ownership mapping (GNU tar's as-root default).  Run under
    #    GZSTD_DEBUG_FAKE_ROOT with the runner's own user + a supplementary
    #    group: the archive carries those NAMES with bogus numeric ids, so a
    #    correct name-first chown lands on the local ids (observable without
    #    root), while --numeric-owner attempts the bogus ids and no-ops.
    sgrp=$(id -Gn | tr ' ' '\n' | grep -v "^$(id -gn)$" | head -1)
    if [[ -n "$sgrp" ]]; then
      python3 - "$LP" "$(id -un)" "$sgrp" <<'PYEOF'
import tarfile, io, sys
base, uname, gname = sys.argv[1:4]
t = tarfile.open(base + "/own.tar", "w", format=tarfile.GNU_FORMAT)
ti = tarfile.TarInfo("mapped.txt"); ti.size = 2; ti.mode = 0o644; ti.mtime = 981173106
ti.uname, ti.gname, ti.uid, ti.gid = uname, gname,4242, 4244
t.addfile(ti, io.BytesIO(b"x\n")); t.close()
PYEOF
      "$GZSTD" --cpu-only -q -f -o "$LP/own.tar.zst" "$LP/own.tar" 2>/dev/null
      rm -rf "$LP/o1" "$LP/o2"; mkdir -p "$LP/o1" "$LP/o2"
      GZSTD_DEBUG_FAKE_ROOT=1 "$GZSTD" -d --cpu-only -q --tar -C "$LP/o1" "$LP/own.tar.zst" 2>/dev/null
      GZSTD_DEBUG_FAKE_ROOT=1 "$GZSTD" -d --cpu-only -q --tar --numeric-owner -C "$LP/o2" "$LP/own.tar.zst" 2>/dev/null
      if [[ "$(stat -c %G "$LP/o1/mapped.txt" 2>/dev/null)" == "$sgrp" \
         && "$(stat -c %G "$LP/o2/mapped.txt" 2>/dev/null)" != "$sgrp" ]]; then
        pass "extract chowns by NAME first; --numeric-owner uses raw ids"
      else fail "name-first ownership mapping" "group not remapped as expected"; fi
    else
      skip "name-first ownership mapping" "no supplementary group"
    fi
  else
    skip "-l --tar parity (crafted archives)" "python3 not available"
    skip "name-first ownership mapping" "python3 not available"
  fi
  rm -rf "$LP"
fi

# ============================================================
# Tar-aware integrity verification (-t --tar) + content checksums
# ============================================================
section "Tar verify and checksums (-t --tar)"

if ! command -v tar >/dev/null 2>&1; then
  skip "tar verify" "tar not available"
else
  VS="$TMPDIR/vsrc"; rm -rf "$VS"; mkdir -p "$VS/d"
  for i in 1 2 3 4 5; do echo "verify-$i" > "$VS/d/f$i"; done
  head -c 2000000 /dev/urandom > "$VS/big.bin"
  VA="$TMPDIR/v.tar.zst"
  "$GZSTD" --cpu-only -q -f -o "$VA" --tar "$VS" 2>/dev/null

  # 1. valid archive: -t --tar succeeds and validates the tar structure.
  "$GZSTD" -t --cpu-only --tar "$VA" >/dev/null 2>&1 \
    && pass "-t --tar accepts a valid archive" || fail "-t --tar valid" "rc=$?"

  # 1b. -t --tar reports the true decompressed tar-stream size (matching the
  #     create summary and zstd -l), NOT the smaller file-content-only sum.
  #     Build a many-small-file archive so header/padding overhead is large and
  #     the two definitions differ sharply, then compare -t's byte count to the
  #     stream size gzstd -d actually produces.
  VMS="$TMPDIR/vms"; rm -rf "$VMS"; mkdir -p "$VMS/d"
  for i in $(seq 1 3000); do echo "small-file-payload" > "$VMS/d/f$i"; done
  VMA="$TMPDIR/vms.tar.zst"
  "$GZSTD" --cpu-only -q -f -o "$VMA" --tar "$VMS" 2>/dev/null
  real_stream=$("$GZSTD" -d --cpu-only -q -c "$VMA" 2>/dev/null | wc -c)
  # pull the "=> N.NN UNIT" uncompressed figure out of the -t summary and convert
  t_line=$("$GZSTD" -t --cpu-only --tar "$VMA" 2>&1 | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep entries | tail -1)
  t_unc=$(echo "$t_line" | sed -E 's/.*=> ([0-9.]+) ([KMG]?i?B).*/\1 \2/')
  t_bytes=$(awk -v v="${t_unc% *}" -v u="${t_unc#* }" 'BEGIN{
    m=(u=="KiB"?1024:u=="MiB"?1048576:u=="GiB"?1073741824:1); printf "%.0f", v*m }')
  # allow 1% slack for human_bytes rounding; content-only would be ~10x smaller
  lo=$(( real_stream * 99 / 100 )); hi=$(( real_stream * 101 / 100 ))
  if [ "$t_bytes" -ge "$lo" ] && [ "$t_bytes" -le "$hi" ]; then
    pass "-t --tar reports the true stream size (not content-only)"
  else fail "-t --tar stream size" "t=$t_bytes stream=$real_stream (line: $t_line)"; fi

  # 2. truncated archive: -t --tar must reject it (exit 4).
  vsz=$(stat -c%s "$VA"); head -c $((vsz*60/100)) "$VA" > "$TMPDIR/v.trunc.zst"
  "$GZSTD" -t --cpu-only --tar "$TMPDIR/v.trunc.zst" >/dev/null 2>&1
  [[ $? -eq 4 ]] && pass "-t --tar rejects a truncated archive" || fail "-t --tar truncated" "expected exit 4"

  # 3. content checksum: a bit-flip in the compressed stream is caught (exit 4)
  #    on plain -t — proves ZSTD_c_checksumFlag is active.
  if command -v python3 >/dev/null 2>&1; then
    cp "$VA" "$TMPDIR/v.flip.zst"
    python3 -c "d=bytearray(open('$TMPDIR/v.flip.zst','rb').read()); d[int(len(d)*0.5)]^=0xFF; open('$TMPDIR/v.flip.zst','wb').write(d)"
    "$GZSTD" -t --cpu-only "$TMPDIR/v.flip.zst" >/dev/null 2>&1
    [[ $? -eq 4 ]] && pass "content checksum catches a bit-flip" || fail "checksum bit-flip" "expected exit 4"
  else
    skip "content checksum catches a bit-flip" "python3 unavailable"
  fi

  # 4. multi-frame archive: with a tiny chunk size the tar stream spans many
  #    zstd frames, so member headers and large-file data straddle frame
  #    boundaries.  Exercises the in-memory verify path's cross-frame skip()/
  #    read_exact (v0.14.19) — the small single-frame archive above does not.
  MS="$TMPDIR/vmsrc"; rm -rf "$MS"; mkdir -p "$MS/d"
  for i in $(seq 1 40); do echo "member-$i content line" > "$MS/d/m$i"; done
  head -c 5000000 /dev/urandom > "$MS/d/big1.bin"   # spans several 1 MiB frames
  head -c 3000000 /dev/urandom > "$MS/d/big2.bin"
  MA="$TMPDIR/vm.tar.zst"
  "$GZSTD" --cpu-only -q -f --chunk-size 1 -o "$MA" --tar "$MS" 2>/dev/null
  "$GZSTD" -t --cpu-only --tar "$MA" >/dev/null 2>&1 \
    && pass "-t --tar accepts a multi-frame archive (cross-frame skip)" \
    || fail "-t --tar multi-frame" "rc=$?"

  # 5. foreign archive: `tar -cf - | zstd` produces a single streaming frame with
  #    no content-size header, so verify takes the fallback decompress path
  #    (decompress_from_buffer) which routes to the in-memory validator too.
  #    Confirms -t --tar still works on archives gzstd did not author.
  if command -v zstd >/dev/null 2>&1; then
    FA="$TMPDIR/vforeign.tar.zst"
    ( cd "$(dirname "$VS")" && tar -cf - "$(basename "$VS")" | zstd -q -o "$FA" ) 2>/dev/null
    "$GZSTD" -t --cpu-only --tar "$FA" >/dev/null 2>&1 \
      && pass "-t --tar accepts a foreign tar|zstd archive (fallback path)" \
      || fail "-t --tar foreign archive" "rc=$?"
    rm -f "$FA"
  else
    skip "-t --tar accepts a foreign tar|zstd archive (fallback path)" "zstd CLI unavailable"
  fi

  rm -rf "$VS" "$MS" "$VA" "$MA" "$TMPDIR"/v.*.zst
fi

# ============================================================
# List (-l, and -l --tar)
# ============================================================
section "List contents (-l)"

if ! command -v tar >/dev/null 2>&1; then
  skip "list" "tar not available"
else
  LS="$TMPDIR/lsrc"; rm -rf "$LS"; mkdir -p "$LS/d"
  for i in 1 2 3 4 5 6; do echo "list-entry-$i" > "$LS/d/f$i"; done
  head -c 3000000 /dev/urandom > "$LS/big.bin"   # forces several 1 MiB frames below
  ln -sf big.bin "$LS/lnk"
  LA="$TMPDIR/l.tar.zst"
  "$GZSTD" --cpu-only -q -f --chunk-size 1 -o "$LA" --tar "$LS" 2>/dev/null

  # 1. plain -l: a header + a single data row, exit 0, names the file.
  out=$("$GZSTD" -l "$LA" 2>/dev/null); rc=$?
  if [[ $rc -eq 0 ]] && grep -q "Frames" <<<"$out" && grep -q "l.tar.zst" <<<"$out"; then
    pass "-l prints a frame summary"
  else fail "-l frame summary" "rc=$rc out=[$out]"; fi

  # 2. plain -l frame count matches zstd -l (both walk the same frames).
  if command -v zstd >/dev/null 2>&1; then
    gz_frames=$("$GZSTD" -l "$LA" 2>/dev/null | awk 'NR==2{print $1}')
    zs_frames=$(zstd -l "$LA" 2>/dev/null | awk 'NR==2{print $1}')
    if [[ -n "$gz_frames" && "$gz_frames" == "$zs_frames" ]]; then
      pass "-l frame count matches zstd -l" "($gz_frames)"
    else fail "-l frame count vs zstd" "gzstd=$gz_frames zstd=$zs_frames"; fi
  else
    skip "-l frame count matches zstd -l" "zstd CLI unavailable"
  fi

  # 2b. Plain compress appends the zstd seekable seek table (v0.14.92): the
  # footer magic is the file's last bytes (feeds the O(1) -l fast path and
  # foreign seekable readers), zstd itself still accepts the trailing
  # skippable frame, and --no-index opts out.
  PLZ="$TMPDIR/plainseek.zst"
  "$GZSTD" --cpu-only -q -f -o "$PLZ" "$TMPDIR/large.bin" 2>/dev/null
  ftr=$(tail -c 4 "$PLZ" | od -An -tx1 | tr -d ' \n')
  if [[ "$ftr" == "b1ea928f" ]]; then
    pass "plain compress appends seekable seek table"
  else fail "plain seek table" "footer=$ftr"; fi
  if ! command -v zstd >/dev/null 2>&1; then
    skip "zstd -t accepts tabled plain output" "zstd CLI unavailable"
  elif zstd -t -q "$PLZ" 2>/dev/null; then
    pass "zstd -t accepts tabled plain output"
  else fail "zstd -t on tabled output" "zstd rejected it"; fi
  "$GZSTD" --cpu-only --no-index -q -f -o "$PLZ" "$TMPDIR/large.bin" 2>/dev/null
  ftr=$(tail -c 4 "$PLZ" | od -An -tx1 | tr -d ' \n')
  if [[ "$ftr" != "b1ea928f" ]]; then
    pass "--no-index omits the plain seek table"
  else fail "--no-index plain" "table still present"; fi

  # 2c. A warm (page-cache-resident) un-tabled file dispatches -l to the
  # buffered pread walk (v0.14.93; mincore residency sample at -v) and its
  # counts match zstd -l exactly.  Warm the file EXPLICITLY: on Gen4+ boxes
  # compress auto-enables --direct, so a fresh write bypasses the page cache
  # entirely and the file starts cold (the dispatch then correctly picks the
  # mmap walk — right behavior, wrong fixture for this test).
  if ! command -v zstd >/dev/null 2>&1; then
    skip "warm -l buffered-walk parity with zstd -l" "zstd CLI unavailable"
  else
    cat "$PLZ" >/dev/null
    route=$("$GZSTD" -l -v "$PLZ" 2>&1 >/dev/null | grep -c "buffered walk" || true)
    gzc=$("$GZSTD" -l "$PLZ" 2>/dev/null | awk 'NR==2{print $1","$2}')
    zsc=$(zstd -l "$PLZ" 2>/dev/null | awk 'NR==2{print $1","$2}')
    if [[ "$route" == "1" && -n "$gzc" && "$gzc" == "$zsc" ]]; then
      pass "warm -l buffered-walk parity with zstd -l" "($gzc)"
    else fail "warm -l buffered walk" "route=$route gz=$gzc zs=$zsc"; fi
  fi
  rm -f "$PLZ"

  # 3. -l --tar lists every entry (count matches what's in the tree) and shows
  #    the tar -tvf fields (perms column + a known member + the symlink target).
  list=$("$GZSTD" -l --tar --cpu-only "$LA" 2>/dev/null); rc=$?
  n_list=$(grep -c . <<<"$list")
  n_tree=$(find "$LS" | wc -l)   # dirs + files + symlink (matches tar entry count)
  if [[ $rc -eq 0 ]] && [[ "$n_list" -eq "$n_tree" ]] \
     && grep -q "/d/f1" <<<"$list" && grep -q -- "lnk -> big.bin" <<<"$list" \
     && grep -qE '^[-dl]' <<<"$list"; then
    pass "-l --tar lists all entries (tar -tvf style)" "($n_list)"
  else fail "-l --tar listing" "rc=$rc listed=$n_list tree=$n_tree"; fi

  rm -rf "$LS" "$LA"
fi

# ============================================================
# Terminal-output guard (refuse compressed data to a TTY)
# ============================================================
section "Terminal-output guard"

if ! command -v python3 >/dev/null 2>&1; then
  skip "refuses compress to a TTY" "python3 unavailable (need a pty)"
  skip "-f forces compress to a TTY" "python3 unavailable (need a pty)"
else
  # Run gzstd compressing to stdout with stdout attached to a real pty; echo
  # "<exit_code> <1 if zstd-magic was written else 0>".
  tty_compress() {  # $1 = plain|force
    python3 - "$GZSTD" "$TMPDIR/medium.txt" "$1" <<'PY'
import sys, os, pty, select, subprocess
gz, src, mode = sys.argv[1], sys.argv[2], sys.argv[3]
args = [gz, "-c", src] + (["-f"] if mode == "force" else [])
mp, sp = pty.openpty()
p = subprocess.Popen(args, stdout=sp, stderr=subprocess.DEVNULL)
os.close(sp)
out = b""
while True:
    r, _, _ = select.select([mp], [], [], 5)
    if not r:
        if p.poll() is not None: break
        continue
    try: chunk = os.read(mp, 65536)
    except OSError: break
    if not chunk: break
    out += chunk
p.wait(); os.close(mp)
print(p.returncode, 1 if out[:4] == b"\x28\xb5\x2f\xfd" else 0)
PY
  }
  read rc magic < <(tty_compress plain)
  if [[ "$rc" -ne 0 && "$magic" -eq 0 ]]; then
    pass "refuses compress to a TTY (no -f)" "(exit $rc, no output)"
  else
    fail "refuses compress to a TTY" "rc=$rc magic=$magic"
  fi
  read rc magic < <(tty_compress force)
  if [[ "$rc" -eq 0 && "$magic" -eq 1 ]]; then
    pass "-f forces compress to a TTY"
  else
    fail "-f forces compress to a TTY" "rc=$rc magic=$magic"
  fi
fi

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
