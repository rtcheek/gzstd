#!/usr/bin/env bash
#=====================================================================
# gzstd-benchmark.sh  Automated benchmark suite for gzstd
#
# Compares backends and settings end to end, compress AND decompress, and
# RECORDS WHAT EACH RUN ACTUALLY DID: every gzstd run writes --stats-json, and
# the tables show the GPU share of the bytes, the cards used and the CPU
# threads.  A label is a request; since the GPU startup guards a default or
# --hybrid run only starts its cards when they are expected to pay, so a row
# can ask for a GPU knob and run on the CPUs alone.  Such rows are flagged.
# A binary before v0.18.63 records GPU devices for compress only; with one, a
# decode row's GPU share is n/a and a --gpu-only decode that ran CPU decoders
# is flagged instead.
#
# Usage:
#   ./gzstd-benchmark.sh [options]
#
# Options:
#   --gzstd PATH      Path to gzstd binary (default: ./gzstd, else ./build/gzstd)
#   --data-dir DIR    Directory with test files (default: ./gzstd-testdata)
#   --output FILE     JSON output file (default: benchmark-results.json)
#   --iterations N    Runs per config, reports the median (default: 3)
#   --files PATTERN   Only test files matching glob (default: all .bin)
#   --quick           Reduced sweeps, 1 iteration
#   --list            Print the configurations and runs, then exit
#
#   --cpus LIST       CPU allowances for the baseline rows, e.g. "4,8,16,all"
#                     (default: all).  Each N runs under `taskset` on the first
#                     N CPUs this process may use -- a CPU MASK, not -T, because
#                     whether a default run starts its GPUs depends on the
#                     cores it has (on the 8-H100 server they pay below ~10).
#                     Sweeps run at the full allowance only.
#   --cold MODE       How inputs are made cold before EVERY timed run:
#                       evict        (default) scripts/drop_cache evicts each
#                                    input's pages (per file, rootless, other
#                                    users' cache untouched); gzstd runs with
#                                    the flags the row names and nothing else,
#                                    so the default row is a default run
#                       direct-read  add --direct-read to plain rows (the old
#                                    method; it changes the read path a
#                                    default run would choose); tar rows use
#                                    eviction instead, recorded per row
#                       warm         read each input before each run, never evict
#   --calibrated      Before the rows at each CPU allowance, run
#                     `gzstd --calibrate FILE FILE.zst` for each test file under
#                     that allowance, into this run's private cache, so default
#                     and hybrid rows start their GPUs at a price measured here
#                     instead of the flat 4.0 s guard.  Slow (minutes per file
#                     and allowance), and FILE should keep the CPU pool busy for
#                     several seconds (see `gzstd --help`, --calibrate).  Each
#                     calibration's report is kept beside the JSON output.
#
#   --tar             Also benchmark --tar: create (to /dev/null), extract and
#                     a one-member seek-extract, on two corpora -- the test
#                     files as a few large members, and many small files cut
#                     from one test file (--tar-many-files, generated
#                     once into DATA_DIR/tar-many) -- for the baseline rows,
#                     beside GNU tar + zstd for the same corpora
#   --tar-only        Only the --tar rows (implies --tar)
#   --tar-many-files N  Files in the many-small corpus (default: 40000, 32 KiB each)
#   --tar-dest DIR    Where extractions are written (default: a directory in
#                     this run's temp dir).  Uses a private child of DIR; existing
#                     contents are kept.  The extract time INCLUDES a sync of
#                     that filesystem, so a slow sink is charged, not hidden.
#   --tar-work DIR    Parent for this run's private archive directory (default: temp)
#   --no-gnu-tar      Skip the GNU tar + zstd baseline rows
#
#   --gpu-only        Only the GPU configs (--gpu-only rows and GPU sweeps)
#   --cpu-only        Only the CPU configs
#   --hybrid-only     Only the default and --hybrid rows
#   --sweep-batches   GPU batch sizes (--gpu-only)
#   --sweep-streams   GPU stream counts (--gpu-only)
#   --sweep-threads   CPU thread counts (--cpu-only -T)
#   --sweep-levels    Compression levels (--cpu-only)
#   --sweep-ultra     Ultra levels 20-22 (--cpu-only)
#   --sweep-throttle  --throttle-factor (cpu-only and gpu-only)
#   --sweep-matrix    backend x mmap x pinned (cpu-only and gpu-only)
#   --sweep-all       All sweeps
#   --adapt-warm      Add an adapt-warm row (see the CONFIGS block)
#   --no-decompress   Skip the plain decompress rows
#   --help            Show this help
#
# Baseline rows: default (no flags: the policy users get), cpu-only, gpu-only,
# hybrid, and adapt.  The GPU knob sweeps run under --gpu-only, which always
# starts the cards; the --hybrid versions of them were removed (v0.18.63)
# because a hybrid run declines its GPUs wherever its CPUs keep up, and then
# every knob setting measured the same CPU-only run.  To study a knob inside a
# hybrid run, force the cards on with `--hybrid --cpu-share X`.
#
# Plain rows write to /dev/null (the read + compute path is compared; every
# config shares the same writer).  Decompress reads the .bin.zst that
# gzstd-gendata.sh builds next to each .bin -- run gzstd-gendata.sh first.
# Every run gets a private XDG_CACHE_HOME, so no row reads or writes the
# machine's real --adapt profile or calibration; without --calibrated, hybrid
# and default rows therefore run on the flat 4.0 s GPU guard.
#=====================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#---------------------------------------------------------------------
# Defaults
#---------------------------------------------------------------------
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
LIST_ONLY=false
ADAPT_WARM=false
GPU_ONLY=false
CPU_ONLY=false
HYBRID_ONLY=false
SWEEP_BATCHES=false
SWEEP_STREAMS=false
SWEEP_THREADS=false
SWEEP_LEVELS=false
SWEEP_ULTRA=false
SWEEP_THROTTLE=false
SWEEP_MATRIX=false
DO_DECOMPRESS=true
CPUS_ARG="all"
COLD="evict"
CALIBRATED=false
DO_TAR=false
TAR_ONLY=false
TAR_MANY_FILES=40000
TAR_DEST=""
TAR_WORK=""
GNU_TAR=true

#---------------------------------------------------------------------
# Parse arguments
#---------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gzstd)       GZSTD="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --output)      OUTPUT="$2"; shift 2 ;;
    --iterations)  ITERATIONS="$2"; shift 2 ;;
    --files)       FILE_PATTERN="$2"; shift 2 ;;
    --quick)       QUICK=true; ITERATIONS=1; shift ;;
    --list)        LIST_ONLY=true; shift ;;
    --adapt-warm)  ADAPT_WARM=true; shift ;;
    --gpu-only)    GPU_ONLY=true; shift ;;
    --cpu-only)    CPU_ONLY=true; shift ;;
    --hybrid-only) HYBRID_ONLY=true; shift ;;
    --sweep-batches) SWEEP_BATCHES=true; shift ;;
    --sweep-streams) SWEEP_STREAMS=true; shift ;;
    --sweep-threads) SWEEP_THREADS=true; shift ;;
    --sweep-levels)  SWEEP_LEVELS=true; shift ;;
    --sweep-ultra)   SWEEP_ULTRA=true; shift ;;
    --sweep-throttle) SWEEP_THROTTLE=true; shift ;;
    --sweep-matrix)  SWEEP_MATRIX=true; shift ;;
    --sweep-all)
      SWEEP_BATCHES=true; SWEEP_STREAMS=true
      SWEEP_THREADS=true; SWEEP_LEVELS=true
      SWEEP_ULTRA=true; SWEEP_THROTTLE=true
      SWEEP_MATRIX=true; shift ;;
    --no-decompress) DO_DECOMPRESS=false; shift ;;
    --cpus)        CPUS_ARG="$2"; shift 2 ;;
    --cold)        COLD="$2"; shift 2 ;;
    --calibrated)  CALIBRATED=true; shift ;;
    --tar)         DO_TAR=true; shift ;;
    --tar-only)    DO_TAR=true; TAR_ONLY=true; shift ;;
    --tar-many-files) TAR_MANY_FILES="$2"; shift 2 ;;
    --tar-dest)    TAR_DEST="$2"; shift 2 ;;
    --tar-work)    TAR_WORK="$2"; shift 2 ;;
    --no-gnu-tar)  GNU_TAR=false; shift ;;
    --help|-h)
      # Print the header block from the title down to the closing ==== rule.
      awk 'NR>2 { if ($0 ~ /^#={10,}/) exit; print }' "$0"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

#---------------------------------------------------------------------
# Validation
#---------------------------------------------------------------------
if [[ ! -x "$GZSTD" ]]; then
  echo "ERROR: gzstd binary not found (tried ./gzstd and ./build/gzstd)"
  echo "       Use --gzstd PATH, set GZSTD env var, or build first."
  exit 1
fi
GZSTD=$(cd "$(dirname "$GZSTD")" && pwd)/$(basename "$GZSTD")   # absolute: runs cd nowhere, but be safe

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Test data directory '$DATA_DIR' not found."
  echo "       Run gzstd-gendata.sh first to generate test data."
  exit 1
fi
DATA_DIR=$(cd "$DATA_DIR" && pwd)

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --iterations needs a positive integer"; exit 2
fi
if ! [[ "$TAR_MANY_FILES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --tar-many-files needs a positive integer"; exit 2
fi
if ! $LIST_ONLY && ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to read and write benchmark statistics"; exit 2
fi

DROP_CACHE="$SCRIPT_DIR/scripts/drop_cache"
case "$COLD" in
  evict)
    if ! command -v python3 >/dev/null 2>&1 || [[ ! -f "$DROP_CACHE" ]]; then
      echo "NOTE: --cold evict needs python3 and scripts/drop_cache; using --cold direct-read"
      COLD="direct-read"
    fi ;;
  direct-read|warm) ;;
  *) echo "ERROR: --cold must be evict, direct-read or warm"; exit 2 ;;
esac
if $DO_TAR && [[ "$COLD" == direct-read && ! -f "$DROP_CACHE" ]]; then
  echo "ERROR: --tar with --cold direct-read needs scripts/drop_cache for its tar rows; use --cold warm instead"
  exit 2
fi

# Detect hardware
NCPU=$(nproc 2>/dev/null || echo 4)
HAS_GPU_BUILD=false
if "$GZSTD" --version 2>&1 | grep -q "nvCOMP"; then
  HAS_GPU_BUILD=true
fi

if $GPU_ONLY && ! $HAS_GPU_BUILD; then
  echo "ERROR: --gpu-only requested but gzstd built without nvCOMP"
  exit 1
fi

GPU_SUMMARY="none detected"
GPU_DRIVER=""
GPU_DETAILS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_DETAILS < <(
    nvidia-smi --query-gpu=index,name,memory.total \
      --format=csv,noheader 2>/dev/null || true)
  if [[ ${#GPU_DETAILS[@]} -gt 0 ]]; then
    GPU_SUMMARY="${#GPU_DETAILS[@]} detected"
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version \
                   --format=csv,noheader 2>/dev/null | head -1)
  fi
elif command -v lspci >/dev/null 2>&1; then
  mapfile -t GPU_DETAILS < <(
    lspci | grep -Ei 'vga|3d|display' | grep -Ei 'nvidia|amd|intel' || true)
  if [[ ${#GPU_DETAILS[@]} -gt 0 ]]; then
    GPU_SUMMARY="${#GPU_DETAILS[@]} detected"
  fi
fi

#---------------------------------------------------------------------
# CPU allowances (--cpus)
#---------------------------------------------------------------------
# The CPUs this process may use, expanded from a list like "0-63,128-191".
ALLOWED_CPUS=()
_allowed=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null || true)
[[ -z "$_allowed" ]] && _allowed="0-$((NCPU - 1))"
IFS=',' read -r -a _ranges <<< "$_allowed"
for _r in "${_ranges[@]}"; do
  if [[ "$_r" == *-* ]]; then
    for ((_c=${_r%-*}; _c<=${_r#*-}; _c++)); do ALLOWED_CPUS+=("$_c"); done
  else
    ALLOWED_CPUS+=("$_r")
  fi
done
NALLOWED=${#ALLOWED_CPUS[@]}

CPU_SET=()      # "all" or N, in the order given
for _n in ${CPUS_ARG//,/ }; do
  if [[ "$_n" == all ]]; then
    CPU_SET+=("all")
  elif [[ "$_n" =~ ^[1-9][0-9]*$ ]]; then
    if (( _n >= NALLOWED )); then
      echo "NOTE: --cpus $_n is not below the $NALLOWED CPUs allowed; running it as 'all'"
      CPU_SET+=("all")
    else
      CPU_SET+=("$_n")
    fi
  else
    echo "ERROR: --cpus takes counts and 'all', e.g. 4,8,16,all"; exit 2
  fi
done
# Drop repeats, keeping the first.
_seen=" "; _uniq=()
for _n in "${CPU_SET[@]}"; do
  [[ "$_seen" == *" $_n "* ]] && continue
  _seen+="$_n "; _uniq+=("$_n")
done
CPU_SET=("${_uniq[@]}")
[[ ${#CPU_SET[@]} -eq 0 ]] && CPU_SET=("all")

# cpu_mask N -> the taskset list for the first N allowed CPUs ("" for all).
cpu_mask() {
  [[ "$1" == all ]] && { echo ""; return; }
  local out="" i
  for ((i=0; i<$1; i++)); do out+="${out:+,}${ALLOWED_CPUS[$i]}"; done
  echo "$out"
}
cpu_label() { [[ "$1" == all ]] && echo "$NALLOWED" || echo "$1"; }

#---------------------------------------------------------------------
# ANSI / display helpers
#---------------------------------------------------------------------
BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
RED=$'\033[31m'
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

#---------------------------------------------------------------------
# Build configuration matrix
#---------------------------------------------------------------------
# Each config: "label|comp_flags|decomp_flags|opts".  opts (harness options,
# not gzstd flags, comma separated):
#   base    a baseline row: run at every --cpus allowance, and for --tar
#   warmup  one untimed run before the timed ones (adapt-warm)
#   comp-only  a compression-only setting; do not relabel the common decode input

CONFIGS=()

# default: no backend flag at all -- the policy every user gets, and the only
# row that can take the staged path (naming a backend rules staging out).  It
# was dropped in v0.13.42 as a duplicate of the hybrid rows; it no longer is.
if ! $GPU_ONLY && ! $CPU_ONLY; then
  CONFIGS+=("default|||base")
fi
if ! $GPU_ONLY && ! $HYBRID_ONLY; then
  CONFIGS+=("cpu-only|--cpu-only|--cpu-only|base")
fi
if $HAS_GPU_BUILD && ! $CPU_ONLY; then
  if ! $HYBRID_ONLY; then
    CONFIGS+=("gpu-only|--gpu-only|--gpu-only|base")
  fi
  if ! $GPU_ONLY; then
    CONFIGS+=("hybrid|--hybrid|--hybrid|base")
  fi
fi

# --adapt: the governor picks its own backend and tunes itself while running, so
# this is the row that answers "does --adapt actually beat the fixed backends on
# this box?".  Not gated on a GPU build: --adapt has plenty to do on a CPU-only
# machine (read path, writer parallelism, extract pool sizing).
#
# --no-profile is deliberate, and it is a TRADE.  It stops the run from writing
# calibration into a profile -- and every plain output goes to /dev/null, so a
# benchmark would teach a sink rate for a device that does not exist.  But it
# disables profile READING too, so this row measures --adapt WITHOUT its learned
# priors: read it as --adapt's FLOOR.  (Every run also has a private
# XDG_CACHE_HOME, below, so nothing reaches the machine's real profile anyway.)
if ! $GPU_ONLY && ! $HYBRID_ONLY && ! $CPU_ONLY; then
  CONFIGS+=("adapt|--adapt --no-profile|--adapt --no-profile|base")

  # --adapt-warm: the same governor allowed to KEEP what it learns, so the pair
  # brackets --adapt -- `adapt` is its floor, `adapt-warm` its steady state.  One
  # untimed priming run per input seeds the private profile first.  --adapt only
  # records a run of 3 s or longer, so on inputs that finish faster the priming
  # run records nothing and this row lands on the cold one: use bigger inputs.
  if $ADAPT_WARM; then
    CONFIGS+=("adapt-warm|--adapt|--adapt|base,warmup")
  fi
fi

# GPU batch sizes (--gpu-only always starts the cards)
if $SWEEP_BATCHES && $HAS_GPU_BUILD && ! $CPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then BATCH_SIZES="8 64 256"; else BATCH_SIZES="4 8 16 32 64 128 256 512 1024"; fi
  for b in $BATCH_SIZES; do
    CONFIGS+=("gpu-batch${b}|--gpu-only --gpu-batch $b|--gpu-only --gpu-batch $b|")
  done
fi

# GPU stream counts
if $SWEEP_STREAMS && $HAS_GPU_BUILD && ! $CPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then STREAM_COUNTS="1 3 6"; else STREAM_COUNTS="1 2 3 4 6 8"; fi
  for s in $STREAM_COUNTS; do
    CONFIGS+=("gpu-streams${s}|--gpu-only --gpu-streams $s|--gpu-only --gpu-streams $s|")
  done
fi

# Combined: batch x streams (both sweeps, not --quick)
if $SWEEP_BATCHES && $SWEEP_STREAMS && $HAS_GPU_BUILD && ! $CPU_ONLY && ! $HYBRID_ONLY && ! $QUICK; then
  for b in 16 32 64; do
    for s in 2 4 6; do
      CONFIGS+=("gpu-b${b}s${s}|--gpu-only --gpu-batch $b --gpu-streams $s|--gpu-only --gpu-batch $b --gpu-streams $s|")
    done
  done
fi

# CPU thread counts
if $SWEEP_THREADS && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then
    THREAD_COUNTS="1 $((NALLOWED/2)) $((NALLOWED-1))"
  else
    THREAD_COUNTS="1 2 4 8"
    [[ $NALLOWED -gt 8 ]] && THREAD_COUNTS="$THREAD_COUNTS 16"
    [[ $NALLOWED -gt 16 ]] && THREAD_COUNTS="$THREAD_COUNTS $((NALLOWED-1))"
  fi
  for t in $THREAD_COUNTS; do
    [[ $t -lt 1 || $t -ge $NALLOWED ]] && continue
    CONFIGS+=("cpu-T${t}|--cpu-only -T $t|--cpu-only -T $t|")
  done
fi

# Compression levels
if $SWEEP_LEVELS && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then LEVELS="1 6 19"; else LEVELS="1 3 6 9 12 15 19"; fi
  for l in $LEVELS; do
    CONFIGS+=("cpu-level${l}|--cpu-only -$l|--cpu-only|comp-only")
  done
fi

# Ultra levels (20-22): gzstd sizes chunks and guards RAM for these itself.
if $SWEEP_ULTRA && ! $GPU_ONLY && ! $HYBRID_ONLY; then
  if $QUICK; then ULTRA_LEVELS="22"; else ULTRA_LEVELS="20 21 22"; fi
  for l in $ULTRA_LEVELS; do
    CONFIGS+=("cpu-ultra${l}|--cpu-only --ultra -$l|--cpu-only|comp-only")
  done
fi

# Throttle slack factor: 1 is the tightest admissible pipeline (one in-flight
# frame per producer), 16 the loosest before the RAM cap dominates.
if $SWEEP_THROTTLE && ! $HYBRID_ONLY; then
  if $QUICK; then THROTTLE_FACTORS="1 4 16"; else THROTTLE_FACTORS="1 2 4 8 16"; fi
  for f in $THROTTLE_FACTORS; do
    if ! $GPU_ONLY; then
      CONFIGS+=("cpu-thr${f}|--cpu-only --throttle-factor $f|--cpu-only --throttle-factor $f|")
    fi
    if $HAS_GPU_BUILD && ! $CPU_ONLY; then
      CONFIGS+=("gpu-thr${f}|--gpu-only --throttle-factor $f|--gpu-only --throttle-factor $f|")
    fi
  done
fi

# backend x mmap x pinned: which "speed tricks" win on this hardware.  CPU rows
# skip the pinned axis (a GPU-side knob).  See v0.12.46 CHANGELOG.
if $SWEEP_MATRIX && ! $HYBRID_ONLY; then
  if ! $GPU_ONLY; then
    CONFIGS+=(
      "mtx-cpu-mmap|--cpu-only|--cpu-only|"
      "mtx-cpu-nommap|--cpu-only --no-mmap|--cpu-only --no-mmap|"
    )
  fi
  if $HAS_GPU_BUILD && ! $CPU_ONLY; then
    CONFIGS+=(
      "mtx-gpu-mmap-pin0|--gpu-only --pinned=off|--gpu-only --pinned=off|"
      "mtx-gpu-mmap-pin1|--gpu-only --pinned=on|--gpu-only --pinned=on|"
      "mtx-gpu-nommap-pin0|--gpu-only --pinned=off --no-mmap|--gpu-only --pinned=off --no-mmap|"
      "mtx-gpu-nommap-pin1|--gpu-only --pinned=on --no-mmap|--gpu-only --pinned=on --no-mmap|"
    )
  fi
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configurations to test. Check the filter options."
  exit 1
fi

# What a row asks of the GPU: "gpu-only", "gpu" (a GPU knob), or "none".  A
# compress row that asked and moved no bytes on a card measured something else,
# and a --gpu-only decode that ran CPU decoders did not decode on the GPU alone;
# both are flagged in the tables.
gpu_expectation() {
  if [[ "$1" == *--gpu-only* ]]; then echo gpu-only
  elif [[ "$1" == *--gpu-batch* || "$1" == *--gpu-streams* || "$1" == *--pinned* \
          || "$1" == *--gpu-devices* || "$1" == *--cpu-share* ]]; then echo gpu
  else echo none; fi
}
# row_flag KIND EXPECT GPU% THREADS DEVICES [SCOPE] -> a warning, or nothing.  With an older
# binary's decode record (GPU% n/a) a --gpu-only decode that ran CPU decoders
# is the only evidence there is.
row_flag() {
  if [[ "${6:-}" == excess ]] || { [[ "$3" =~ ^[0-9]+([.][0-9]+)?$ ]] \
     && awk -v s="$3" 'BEGIN { exit !(s > 100.000001) }'; }; then
    echo "GPU bytes exceed total"
  elif [[ "$3" == n/a ]]; then
    [[ "$2" == gpu-only && "$4" =~ ^[0-9]+$ && "$4" -gt 0 ]] && echo "decoded on CPUs"
  elif [[ "$5" =~ ^[0-9]+$ ]]; then
    if [[ "$2" != none && "$5" -eq 0 ]]; then
      echo "GPU not engaged"
    elif [[ "$2" == gpu-only && "$1" != compress && "$1" != tar-create ]] \
         && { [[ "${6:-}" == partial ]] || awk -v s="$3" 'BEGIN { exit !(s < 99.999999) }'; }; then
      echo "partly decoded on CPUs"
    fi
  fi
  return 0
}

#---------------------------------------------------------------------
# Collect test files and corpora
#---------------------------------------------------------------------
TEST_FILES=()
for f in "$DATA_DIR"/$FILE_PATTERN; do
  [[ -f "$f" ]] || continue
  # The corpus is .bin (compress input) + matching .bin.zst (decompress input).
  [[ "$f" == *.bin ]] || continue
  TEST_FILES+=("$f")
done

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No .bin test files found in $DATA_DIR matching '$FILE_PATTERN'"
  exit 1
fi

if $DO_DECOMPRESS && ! $TAR_ONLY; then
  missing=()
  for f in "${TEST_FILES[@]}"; do
    [[ -f "${f}.zst" ]] || missing+=("$(basename "${f}.zst")")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing decompress inputs: ${missing[*]}" >&2
    echo "       Run ./gzstd-gendata.sh to (re)build the .bin.zst alongside each .bin," >&2
    echo "       or pass --no-decompress to benchmark compression only." >&2
    exit 1
  fi
fi

file_size() { stat -Lc%s "$1" 2>/dev/null || echo 0; }

# --tar corpora.  "tar-large": the test files as members.  "tar-many": many
# small files cut from one test file (mixed or medium data when there is some), generated once into
# DATA_DIR/tar-many (a stamp marks a complete one, so an interrupted
# generation is redone, not used).
TAR_CORPORA=()
TAR_LARGE_MEMBERS=()
TAR_MANY_DIR="$DATA_DIR/tar-many"
TAR_MANY_STAMP="$TAR_MANY_DIR/.complete-$TAR_MANY_FILES"
TAR_MANY_OWNER="$TAR_MANY_DIR/.gzstd-benchmark-corpus"
if $DO_TAR; then
  for f in "${TEST_FILES[@]}"; do
    if [[ -L "$f" ]]; then
      echo "ERROR: --tar needs regular corpus members, not symlink inputs: $f" >&2
      exit 2
    fi
    TAR_LARGE_MEMBERS+=("$(basename "$f")")
  done
  TAR_CORPORA=("tar-large" "tar-many")
fi

ensure_tar_many() {
  # Cut from mixed or medium data when the corpus has it: the first file in
  # glob order is often high_compress.bin, which would make every member trivial.
  local src="${TEST_FILES[0]}" f
  for f in "${TEST_FILES[@]}"; do [[ "$(basename "$f")" == medium* ]] && src="$f"; done
  for f in "${TEST_FILES[@]}"; do [[ "$(basename "$f")" == mixed* ]] && src="$f"; done
  # A reserved name alone is not permission to delete an existing tree.  Only
  # replace a corpus carrying our ownership marker, including an incomplete one.
  if [[ -e "$TAR_MANY_DIR" || -L "$TAR_MANY_DIR" ]]; then
    if [[ -L "$TAR_MANY_DIR" || ! -d "$TAR_MANY_DIR" || -L "$TAR_MANY_OWNER" ]] \
       || [[ ! -f "$TAR_MANY_OWNER" ]] \
       || [[ "$(cat "$TAR_MANY_OWNER" 2>/dev/null)" != 'gzstd-benchmark tar-many v1' ]]; then
      echo "ERROR: $TAR_MANY_DIR already exists and is not a managed benchmark corpus; move it aside" >&2
      return 1
    fi
  fi
  local signature
  signature=$(python3 - "$src" <<'PY'
import json, os, sys
p = sys.argv[1]
s = os.stat(p)
print(json.dumps([p, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]))
PY
  )
  [[ -f "$TAR_MANY_STAMP" && "$(cat "$TAR_MANY_STAMP")" == "$signature" ]] && return 0
  echo "Generating the many-small-files corpus: $TAR_MANY_FILES files of 32 KiB from $(basename "$src") in $TAR_MANY_DIR ..."
  rm -rf -- "$TAR_MANY_DIR"
  mkdir -p "$TAR_MANY_DIR"
  printf '%s\n' 'gzstd-benchmark tar-many v1' > "$TAR_MANY_OWNER"
  python3 - "$src" "$TAR_MANY_DIR" "$TAR_MANY_FILES" <<'PY'
import os, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
# Sampled evenly across the whole source, not its first n*32 KiB: a generated
# file is laid out in segments (mixed.bin's first 64 MiB compress to 0.07%), and
# the corpus should carry what the whole file does.
CH, PER_DIR = 32 * 1024, 200
size = os.path.getsize(src)
with open(src, 'rb') as f:
    for i in range(n):
        d = os.path.join(out, 'd%05d' % (i // PER_DIR))
        if i % PER_DIR == 0:
            os.makedirs(d, exist_ok=True)
        off = (i * size // n) // CH * CH
        f.seek(max(0, min(off, size - CH)))
        b = f.read(CH)
        with open(os.path.join(d, 'f%05d' % (i % PER_DIR)), 'wb') as o:
            o.write(b)
PY
  printf '%s\n' "$signature" > "$TAR_MANY_STAMP"
}

# Members, -C directory and byte count of a corpus.
corpus_dir()     { [[ "$1" == tar-large ]] && echo "$DATA_DIR" || echo "$DATA_DIR"; }
corpus_members() {
  if [[ "$1" == tar-large ]]; then printf '%s\n' "${TAR_LARGE_MEMBERS[@]}"; else echo "tar-many"; fi
}
corpus_bytes() {
  if [[ "$1" == tar-large ]]; then
    local t=0 f; for f in "${TEST_FILES[@]}"; do t=$((t + $(file_size "$f"))); done; echo "$t"
  else
    find "$TAR_MANY_DIR" -type f ! -name '.complete-*' ! -name '.gzstd-benchmark-corpus' -printf '%s\n' 2>/dev/null | awk '{ t += $1 } END { print t + 0 }'
  fi
}
# The one member a seek-extract asks for: the middle one.
corpus_seek_member() {
  if [[ "$1" == tar-large ]]; then
    echo "${TAR_LARGE_MEMBERS[$(( ${#TAR_LARGE_MEMBERS[@]} / 2 ))]}"
  else
    local i=$(( TAR_MANY_FILES / 2 ))
    printf 'tar-many/d%05d/f%05d\n' $(( i / 200 )) $(( i % 200 ))
  fi
}

#---------------------------------------------------------------------
# The run list: one entry per timed row.
#   "cpus|cfg|kind|item"   cfg = index into CONFIGS, or "gnu" (GNU tar + zstd)
#   kind: compress | decompress | tar-create | tar-extract | tar-seek
#---------------------------------------------------------------------
JOBS=()
HAVE_GNU=false
if $DO_TAR && $GNU_TAR && command -v tar >/dev/null 2>&1 && command -v zstd >/dev/null 2>&1; then
  HAVE_GNU=true
fi
for cpus in "${CPU_SET[@]}"; do
  for ((ci=0; ci<${#CONFIGS[@]}; ci++)); do
    IFS='|' read -r _l _c _d _o <<< "${CONFIGS[$ci]}"
    is_base=false; [[ ",$_o," == *,base,* ]] && is_base=true
    # Sweeps run at the full allowance only.
    if ! $is_base && [[ "$cpus" != all ]]; then continue; fi
    if ! $TAR_ONLY; then
      for f in "${TEST_FILES[@]}"; do
        JOBS+=("$cpus|$ci|compress|$f")
        if $DO_DECOMPRESS && [[ ",$_o," != *,comp-only,* ]]; then
          JOBS+=("$cpus|$ci|decompress|$f")
        fi
      done
    fi
    if $DO_TAR && $is_base; then
      for c in "${TAR_CORPORA[@]}"; do
        JOBS+=("$cpus|$ci|tar-create|$c" "$cpus|$ci|tar-extract|$c" "$cpus|$ci|tar-seek|$c")
      done
    fi
  done
  if $HAVE_GNU; then
    for c in "${TAR_CORPORA[@]}"; do
      JOBS+=("$cpus|gnu|tar-create|$c" "$cpus|gnu|tar-extract|$c")
    done
  fi
done

#---------------------------------------------------------------------
# Banner
#---------------------------------------------------------------------
print_separator
echo "${BOLD}${WHITE} gzstd benchmark suite${RESET}"
print_separator
echo " ${WHITE}Binary${RESET}     : ${YELLOW}$GZSTD${RESET}"
_ver=$("$GZSTD" --version 2>&1 | head -1)
echo " ${WHITE}Version${RESET}    : ${CYAN}${_ver}${RESET}"
echo " ${WHITE}Data dir${RESET}   : ${YELLOW}$DATA_DIR${RESET}"
echo " ${WHITE}Iterations${RESET} : ${GREEN}$ITERATIONS${RESET}"
echo " ${WHITE}CPUs${RESET}       : ${GREEN}$NALLOWED allowed${RESET} ${DIM}($_allowed)${RESET};" \
     "rows at: ${GREEN}${CPU_SET[*]}${RESET}"
echo " ${WHITE}Cold inputs${RESET}: ${GREEN}$COLD${RESET}"
echo " ${WHITE}Calibrated${RESET} : ${GREEN}$CALIBRATED${RESET}"
echo " ${WHITE}GPU build${RESET}  : ${ORANGE}$HAS_GPU_BUILD${RESET}"
echo " ${WHITE}GPUs${RESET}       : ${YELLOW}$GPU_SUMMARY${RESET}"
[[ -n "$GPU_DRIVER" ]] && echo " ${WHITE}Driver${RESET}     : ${CYAN}$GPU_DRIVER${RESET}"
for gpu in "${GPU_DETAILS[@]}"; do
  echo " ${WHITE}GPU detail${RESET} : ${YELLOW}$gpu${RESET}"
done
if $DO_TAR; then
  echo " ${WHITE}--tar${RESET}      : ${GREEN}${TAR_CORPORA[*]}${RESET} ${DIM}(tar-many: $TAR_MANY_FILES files)${RESET};" \
       "GNU tar baseline: ${GREEN}$HAVE_GNU${RESET}"
fi
echo " ${WHITE}Output${RESET}     : ${YELLOW}$OUTPUT${RESET}"
print_separator
echo ""

echo "${BOLD}${WHITE}Configurations:${RESET} ${CYAN}${#CONFIGS[@]}${RESET}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r _l _c _d _o <<< "$cfg"
  printf "  ${YELLOW}%-20s${RESET} comp: %-40s decomp: %s\n" "$_l" "${_c:-(no flags)}" "${_d:-(no flags)}"
done
echo ""
echo "${BOLD}${WHITE}Test files:${RESET}"
for f in "${TEST_FILES[@]}"; do
  echo "  ${YELLOW}$(basename "$f")${RESET}  ${GREEN}$(numfmt --to=iec-i --suffix=B "$(file_size "$f")")${RESET}"
done
echo ""
echo "${BOLD}${WHITE}Timed rows:${RESET} ${CYAN}${#JOBS[@]}${RESET} (x $ITERATIONS iterations)"
if $LIST_ONLY; then
  for j in "${JOBS[@]}"; do
    IFS='|' read -r cpus ci kind item <<< "$j"
    if [[ "$ci" == gnu ]]; then lab="gnu-tar+zstd"; else IFS='|' read -r lab _c _d _o <<< "${CONFIGS[$ci]}"; fi
    printf "  %-6s %-20s %-12s %s\n" "$(cpu_label "$cpus")" "$lab" "$kind" "$(basename "$item")"
  done
  exit 0
fi
echo ""

#---------------------------------------------------------------------
# Run-time helpers
#---------------------------------------------------------------------
TMPDIR=$(mktemp -d /tmp/gzstd-bench.XXXXXX)
TAR_WORK_RUN=""; TAR_DEST_RUN=""; CAL_LOG_DIR=""
cleanup_benchmark() {
  local rc=$?
  if [[ -n "$TAR_DEST_RUN" ]]; then rm -rf -- "$TAR_DEST_RUN" || true; fi
  if [[ -n "$TAR_WORK_RUN" ]]; then rm -rf -- "$TAR_WORK_RUN" || true; fi
  rm -rf -- "$TMPDIR" || true
  return "$rc"
}
trap cleanup_benchmark EXIT
[[ -z "$TAR_WORK" ]] && TAR_WORK="$TMPDIR/tar-work"
[[ -z "$TAR_DEST" ]] && TAR_DEST="$TMPDIR/tar-dest"
if $DO_TAR; then
  mkdir -p "$TAR_WORK" "$TAR_DEST"
  TAR_WORK_RUN=$(mktemp -d "$TAR_WORK/gzstd-bench.XXXXXX")
  TAR_DEST_RUN=$(mktemp -d "$TAR_DEST/gzstd-bench.XXXXXX")
  TAR_WORK="$TAR_WORK_RUN"; TAR_DEST="$TAR_DEST_RUN"
fi

# Every run gets this private cache: no row reads or writes the machine's real
# --adapt profile or calibration (a /dev/null-sink benchmark once taught a real
# profile a sink rate for a device that does not exist).  --calibrated fills it.
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$XDG_CACHE_HOME"
STATS="$TMPDIR/stats.json"

# Make inputs cold (or warm) before a timed run.
evict_files() {
  [[ "$ROW_COLD" == evict ]] || return 0
  if ! python3 "$DROP_CACHE" "$@" >/dev/null; then
    echo "ERROR: input eviction failed; refusing to label the run cold" >&2
    return 1
  fi
}
evict_tree() {
  [[ "$ROW_COLD" == evict ]] || return 0
  if ! find "$@" -type f -print0 | xargs -0 -r python3 "$DROP_CACHE" >/dev/null; then
    echo "ERROR: input-tree eviction failed; refusing to label the run cold" >&2
    return 1
  fi
}
warm_paths() {
  [[ "$ROW_COLD" == warm ]] || return 0
  find -L "$@" -type f -print0 | xargs -0 -r cat >/dev/null
}
prepare_inputs() {
  if [[ ${#cold[@]} -gt 0 ]]; then evict_files "${cold[@]}"; fi
  if [[ ${#cold_tree[@]} -gt 0 ]]; then evict_tree "${cold_tree[@]}"; fi
  # Never with no paths: find with none walks the current directory.
  if [[ "$ROW_COLD" == warm ]] && (( ${#cold[@]} + ${#cold_tree[@]} > 0 )); then
    warm_paths "${cold[@]}" "${cold_tree[@]}"
  fi
}
# A GPU share for display: one decimal, so 0.1% is not 0 and 99.6% is not 100.
pct_show() {
  if [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then awk -v s="$1" 'BEGIN { printf "%.1f", s }'; else echo "$1"; fi
}

# run_timed SYNC_DIR CMD... -> elapsed seconds through command and optional sync;
# SYNC_DIR is empty for rows without an extraction destination.  Exit status goes to
# $TMPDIR/last_rc (this runs in a $(...) subshell).  stdout is discarded,
# stderr kept in last_stderr.log for the progress display and failures.
run_timed() {
  local flush_dir="$1"; shift
  local start end rc=0
  sync 2>/dev/null || true          # earlier writeback is not charged to this run
  rm -f "$TMPDIR/last_end"
  start=$(date +%s.%N)
  # Stamp the end after the command and its optional extraction flush, before
  # the 0.3 s progress poll can delay observation of completion.
  (
    r=0
    "$@" >/dev/null 2>"$TMPDIR/last_stderr.log" || r=$?
    if [[ -n "$flush_dir" ]]; then
      sync -f "$flush_dir" 2>>"$TMPDIR/last_stderr.log" || r=3
    fi
    date +%s.%N > "$TMPDIR/last_end"
    exit "$r"
  ) &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 0.3
    if [[ -s "$TMPDIR/last_stderr.log" ]]; then
      local stripped pct lab
      stripped=$(tail -c 256 "$TMPDIR/last_stderr.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
      pct=$(echo "$stripped" | grep -oP '(?<=out:)[0-9]+\.[0-9]+(?=%)' 2>/dev/null | tail -1 || true)
      lab="out"
      if [[ -z "$pct" ]]; then
        pct=$(echo "$stripped" | grep -oP '(?<=in:)[0-9]+\.[0-9]+(?=%)' 2>/dev/null | tail -1 || true)
        lab="in"
      fi
      if [[ -n "$pct" ]]; then
        CURRENT_GZSTD_PCT="$pct"; CURRENT_GZSTD_LABEL="$lab"; render_status_line
      fi
    fi
  done
  wait "$pid" 2>/dev/null || rc=$?
  end=$(cat "$TMPDIR/last_end" 2>/dev/null || date +%s.%N)
  CURRENT_GZSTD_PCT=""; CURRENT_GZSTD_LABEL=""; render_status_line
  echo "$rc" > "$TMPDIR/last_rc"
  if [[ $rc -ne 0 ]]; then
    echo "" >&2
    echo "  [FAILED rc=$rc] $*" >&2
    [[ -s "$TMPDIR/last_stderr.log" ]] && echo "  stderr: $(tr '\r' '\n' < "$TMPDIR/last_stderr.log" | grep -av '^\s*$' | tail -2 | tr '\n' ' ')" >&2
  fi
  awk -v s="$start" -v e="$end" 'BEGIN { printf "%.4f", e - s }'
}

# What the run did, from --stats-json: "gpu% devices threads in out scope".
# Scope compares integer byte counts, so rounding cannot hide partial CPU work.
# GPU share is of the uncompressed bytes (compress: read; decode: written).  No
# "gpu" object means no GPU pipeline ran: 0%.  Before v0.18.63 a decode's record
# never had one, whether or not its cards ran, so for an older binary a decode
# row says n/a.
run_stats() {
  [[ -s "$STATS" ]] || { echo "- - - 0 0 unknown"; return; }
  python3 - "$STATS" <<'PY' 2>/dev/null || echo "? ? ? 0 0 unknown"
import json, sys
d = json.load(open(sys.argv[1]))
comp = d.get("mode") == "compress"
try:
    ver = tuple(int(x) for x in str(d.get("version", "0")).split(".")[:3])
except ValueError:
    ver = (0,)
if not comp and not isinstance(d.get("gpu"), dict) and ver < (0, 18, 63):
    print("n/a n/a %s %d %d unknown" % (d.get("cpu", {}).get("threads", "?"),
                                d.get("input_bytes", 0), d.get("output_bytes", 0)))
    sys.exit()
g = d.get("gpu") if isinstance(d.get("gpu"), dict) else {}
devs = g.get("devices", []) or []
tot = d.get("input_bytes" if comp else "output_bytes", 0) or 0
on = sum(x.get("input_bytes" if comp else "output_bytes", 0) for x in devs)
share = 100.0 * on / tot if tot else 0.0
used = sum(1 for x in devs if x.get("frames", 0) > 0)
scope = "excess" if on > tot else "none" if on == 0 else "partial" if on < tot else "full"
print("%.6f %d %s %d %d %s" % (share, used, d.get("cpu", {}).get("threads", "?"),
                               d.get("input_bytes", 0), d.get("output_bytes", 0), scope))
PY
}

median() {
  printf '%s\n' "$@" | sort -g | awk '{ v[NR] = $1 }
    END { if (NR % 2) m = v[(NR + 1) / 2]; else m = (v[NR / 2] + v[NR / 2 + 1]) / 2;
          printf "%.4f", m }'
}

throughput_gibs() {
  awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.3f", b / 1073741824 / s; else print "0" }'
}

format_eta() {
  local secs="$1"
  (( secs < 0 )) && secs=0
  if (( secs < 60 )); then printf "%ds" "$secs"
  elif (( secs < 3600 )); then printf "%dm%02ds" "$((secs/60))" "$((secs%60))"
  else printf "%dh%02dm" "$((secs/3600))" "$(( (secs%3600)/60 ))"; fi
}

print_status() {
  local num="$1" total="$2" label="$3" item="$4" kind="$5" cpus="$6"
  local now el eta=0
  now=$(date +%s.%N)
  el=$(awk -v n="$now" -v s="$BENCH_START" 'BEGIN { printf "%.1f", n - s }')
  if (( num > 1 )); then
    eta=$(awk -v e="$el" -v n="$num" -v t="$total" 'BEGIN { printf "%d", e / (n - 1) * (t - n + 1) }')
  fi
  local line=""
  printf -v line "%s%s[%d/%d]%s %s%-17s%s %s@%-4s%s %-22s %s %s~%s%s" \
    "$BOLD" "$CYAN" "$num" "$total" "$RESET" "$YELLOW" "$label" "$RESET" \
    "$DIM" "$(cpu_label "$cpus")" "$RESET" "$item" "$kind" "$DIM" "$(format_eta "$eta")" "$RESET"
  STATUS_LINE="$line"; CURRENT_GZSTD_PCT=""; CURRENT_GZSTD_LABEL=""
  render_status_line
}

# Results accumulator (tab-separated)
RESULTS_FILE="$TMPDIR/results.tsv"
printf 'config\tcpus\tkind\titem\tbytes\tcomp_bytes\tmedian_secs\tthroughput_gibs\tratio_pct\tgpu_pct\tgpu_devices\tcpu_threads\tfailed\tgpu_expect\tflagged\tcold\n' \
  > "$RESULTS_FILE"

#---------------------------------------------------------------------
# Archives for the extract rows: built once per corpus, untimed, with default
# flags at the full allowance.
#---------------------------------------------------------------------
declare -A TAR_ARCHIVE TAR_BYTES
if $DO_TAR; then
  ensure_tar_many
  for c in "${TAR_CORPORA[@]}"; do
    TAR_BYTES[$c]=$(corpus_bytes "$c")
    TAR_ARCHIVE[$c]="$TAR_WORK/$c.tzst"
    if [[ ! -s "${TAR_ARCHIVE[$c]}" ]]; then
      echo "Building $c.tzst for the extract rows (untimed) ..."
      mapfile -t _m < <(corpus_members "$c")
      "$GZSTD" -q -f --tar -C "$(corpus_dir "$c")" "${_m[@]}" -o "${TAR_ARCHIVE[$c]}"
    fi
  done
  echo ""
fi

#---------------------------------------------------------------------
# Run
#---------------------------------------------------------------------
total_tests=${#JOBS[@]}
test_num=0
BENCH_START=$(date +%s.%N)
echo "${BOLD}Starting benchmark: ${total_tests} rows x ${ITERATIONS} iterations${RESET}"
print_separator

calibrated_at=" "
for job in "${JOBS[@]}"; do
  IFS='|' read -r cpus ci kind item <<< "$job"
  mask=$(cpu_mask "$cpus")
  PFX=()
  [[ -n "$mask" ]] && PFX=(taskset -c "$mask")

  # --calibrated: once per allowance, before its first row.
  if $CALIBRATED && [[ "$calibrated_at" != *" $cpus "* ]]; then
    calibrated_at+="$cpus "
    for f in "${TEST_FILES[@]}"; do
      clear_status_line
      # Kept beside the JSON (the temp dir goes at exit): what --calibrate found
      # is part of what the rows at this allowance were measured against.
      if [[ -z "$CAL_LOG_DIR" ]]; then CAL_LOG_DIR=$(mktemp -d "${OUTPUT%.json}.calibrate.XXXXXX"); fi
      cal_log="$CAL_LOG_DIR/$(cpu_label "$cpus")cpu-$(basename "$f").log"
      echo "  ${DIM}calibrating at $(cpu_label "$cpus") CPUs on $(basename "$f") -> $cal_log${RESET}"
      cal_in=("$f"); [[ -f "$f.zst" ]] && cal_in+=("$f.zst")
      "${PFX[@]}" "$GZSTD" --calibrate "${cal_in[@]}" > "$cal_log" 2>&1 \
        || echo "  ${YELLOW}calibration exited $? (see $cal_log)${RESET}"
    done
  fi

  if [[ "$ci" == gnu ]]; then
    label="gnu-tar+zstd"; comp_flags=""; decomp_flags=""; opts=""
  else
    IFS='|' read -r label comp_flags decomp_flags opts <<< "${CONFIGS[$ci]}"
  fi
  warmup=false; [[ ",$opts," == *,warmup,* ]] && warmup=true
  expect=none
  if [[ "$ci" != gnu ]]; then
    case "$kind" in
      compress|tar-create) expect=$(gpu_expectation "$comp_flags") ;;
      *)                   expect=$(gpu_expectation "$decomp_flags") ;;
    esac
  fi
  read_flags=()
  [[ "$COLD" == direct-read && ( "$kind" == compress || "$kind" == decompress ) ]] && read_flags=(--direct-read)
  ROW_COLD="$COLD"
  if [[ "$COLD" == direct-read && "$kind" == tar-* ]]; then ROW_COLD=evict; fi

  test_num=$((test_num + 1))
  item_name=$(basename "$item")
  print_status "$test_num" "$total_tests" "$label" "$item_name" "$kind" "$cpus"

  # Build the command and the inputs to make cold.
  # shellcheck disable=SC2206
  cflags=($comp_flags); dflags=($decomp_flags)
  cold=(); cold_tree=(); dest=""
  case "$kind" in
    compress)
      bytes=$(file_size "$item")
      cmd=("${PFX[@]}" "$GZSTD" "${read_flags[@]}" "${cflags[@]}" -f -o /dev/null --progress --stats-json "$STATS" "$item")
      cold=("$item") ;;
    decompress)
      bytes=$(file_size "$item")
      cmd=("${PFX[@]}" "$GZSTD" "${read_flags[@]}" "${dflags[@]}" -d -f -o /dev/null --progress --stats-json "$STATS" "$item.zst")
      cold=("$item.zst") ;;
    tar-create)
      bytes=${TAR_BYTES[$item]}
      mapfile -t members < <(corpus_members "$item")
      cdir=$(corpus_dir "$item")
      for m in "${members[@]}"; do cold_tree+=("$cdir/$m"); done
      if [[ "$ci" == gnu ]]; then
        zt="-T0"; [[ "$cpus" != all ]] && zt="-T$cpus"
        cmd=("${PFX[@]}" bash -c 'set -o pipefail; d=$1; z=$2; shift 2; tar -cf - -C "$d" "$@" | zstd -q "$z" -c >/dev/null' _ "$cdir" "$zt" "${members[@]}")
      else
        cmd=("${PFX[@]}" "$GZSTD" "${cflags[@]}" -f --progress --stats-json "$STATS" -o /dev/null --tar -C "$cdir" "${members[@]}")
      fi ;;
    tar-extract|tar-seek)
      dest="$TAR_DEST/x"
      arc=${TAR_ARCHIVE[$item]}
      cold=("$arc")
      if [[ "$kind" == tar-seek ]]; then
        member=$(corpus_seek_member "$item")
        if [[ "$item" == tar-large ]]; then bytes=$(file_size "$DATA_DIR/$member")
        else bytes=$(file_size "$TAR_MANY_DIR/${member#tar-many/}"); fi
        cmd=("${PFX[@]}" "$GZSTD" "${dflags[@]}" -d -f --progress --stats-json "$STATS" --tar "$arc" -C "$dest" "$member")
      else
        bytes=${TAR_BYTES[$item]}
        if [[ "$ci" == gnu ]]; then
          zt="-T0"; [[ "$cpus" != all ]] && zt="-T$cpus"
          cmd=("${PFX[@]}" bash -c 'set -o pipefail; zstd -q -d "$3" -c "$1" | tar -xf - -C "$2"' _ "$arc" "$dest" "$zt")
        else
          cmd=("${PFX[@]}" "$GZSTD" "${dflags[@]}" -d -f --progress --stats-json "$STATS" --tar "$arc" -C "$dest")
        fi
      fi ;;
  esac

  # Priming pass (adapt-warm): seed the private profile.  Input residency still
  # follows the row's cold/warm choice for priming and timed runs alike.
  if $warmup; then
    [[ -n "$dest" ]] && { rm -rf "$dest"; mkdir -p "$dest"; }
    prepare_inputs
    run_timed "$dest" "${cmd[@]}" >/dev/null || true
  fi

  times=(); shares=(); devs_max=0; thr_max=0; failed=0; flagged_runs=0; out_bytes=0
  for ((i=1; i<=ITERATIONS; i++)); do
    [[ -n "$dest" ]] && { rm -rf "$dest"; mkdir -p "$dest"; }
    prepare_inputs
    rm -f "$STATS"
    t=$(run_timed "$dest" "${cmd[@]}")
    [[ "$(cat "$TMPDIR/last_rc")" == 0 ]] || failed=$((failed + 1))
    times+=("$t")
    read -r share devs thr _in _out gpu_scope <<< "$(run_stats)"
    iteration_flag=$(row_flag "$kind" "$expect" "$share" "$thr" "$devs" "$gpu_scope")
    if [[ -n "$iteration_flag" ]]; then flagged_runs=$((flagged_runs + 1)); fi
    shares+=("$share")
    [[ "$devs" =~ ^[0-9]+$ ]] && (( devs > devs_max )) && devs_max=$devs
    [[ "$thr" =~ ^[0-9]+$ ]] && (( thr > thr_max )) && thr_max=$thr
    [[ "$_out" =~ ^[0-9]+$ ]] && out_bytes=$_out
  done
  [[ -n "$dest" ]] && rm -rf "$dest"

  med=$(median "${times[@]}")
  thr_g=$(throughput_gibs "$bytes" "$med")
  if [[ "${shares[0]}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then gpu_pct=$(median "${shares[@]}")
  else gpu_pct="${shares[0]}"; [[ "$gpu_pct" == n/a ]] && devs_max="n/a"; fi
  [[ "$ci" == gnu ]] && { gpu_pct="-"; devs_max="-"; thr_max="-"; }
  ratio="-"; comp_bytes=0
  case "$kind" in
    compress|tar-create)
      if (( out_bytes > 0 && bytes > 0 )); then
        comp_bytes=$out_bytes
        ratio=$(awk -v o="$out_bytes" -v b="$bytes" 'BEGIN { printf "%.2f", 100 * o / b }')
      fi ;;
    decompress) comp_bytes=$(file_size "$item.zst") ;;
    tar-extract|tar-seek) comp_bytes=$(file_size "${TAR_ARCHIVE[$item]}") ;;
  esac

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$(cpu_label "$cpus")" "$kind" "$item_name" "$bytes" "$comp_bytes" "$med" "$thr_g" \
    "$ratio" "$gpu_pct" "$devs_max" "$thr_max" "$failed" "$expect" "$flagged_runs" "$ROW_COLD" >> "$RESULTS_FILE"

  clear_status_line
  note=""
  flag=$(row_flag "$kind" "$expect" "$gpu_pct" "$thr_max" "$devs_max")
  if (( flagged_runs > 0 )); then flag="${flag:-some iterations did not run as labelled} ($flagged_runs/$ITERATIONS)"; fi
  [[ -n "$flag" ]] && note="  ${RED}${flag^^}${RESET}"
  (( failed > 0 )) && note+="  ${RED}${failed} FAILED${RESET}"
  gpu_show="$gpu_pct"; [[ "$gpu_pct" =~ ^[0-9]+([.][0-9]+)?$ ]] && gpu_show="$(pct_show "$gpu_pct")% x${devs_max}"
  printf "${CLEAR_LINE}  ${BOLD}${GREEN}✓${RESET} %-17s %s@%-4s%s %-22s %-11s ${BOLD}%8ss${RESET} %8s GiB/s  GPU %s%s\n" \
    "$label" "$DIM" "$(cpu_label "$cpus")" "$RESET" "$item_name" "$kind" "$med" "$thr_g" "$gpu_show" "$note"
done

total_elapsed=$(awk -v n="$(date +%s.%N)" -v s="$BENCH_START" 'BEGIN { printf "%.1f", n - s }')
printf "\r${CLEAR_LINE}"
print_separator
echo "${BOLD}${GREEN}✓ Benchmark complete${RESET}  ${total_tests} rows in ${total_elapsed}s"
echo ""

#---------------------------------------------------------------------
# Results tables, one per kind
#---------------------------------------------------------------------
print_separator
echo "${BOLD}${WHITE} BENCHMARK RESULTS${RESET}  ${DIM}(GPU% = share of the uncompressed bytes the cards moved; x = cards used; thr = CPU threads)${RESET}"
for kind in compress decompress tar-create tar-extract tar-seek; do
  awk -F'\t' -v k="$kind" 'NR>1 && $3==k { f=1 } END { exit !f }' "$RESULTS_FILE" || continue
  echo ""
  echo "${BOLD}${CYAN} ${kind^^}${RESET}"
  printf "${DIM}%-20s %5s %-22s %10s %9s %8s %7s %5s %3s %4s${RESET}\n" \
    "Config" "CPUs" "Item" "Size" "Time(s)" "GiB/s" "Ratio%" "GPU%" "x" "thr"
  awk -F'\t' -v k="$kind" 'NR>1 && $3==k' "$RESULTS_FILE" \
    | while IFS=$'\t' read -r cfg cpus knd item bytes cbytes secs thr ratio gpu devs threads failed expg flagged_count row_cold; do
      size_h=$(numfmt --to=iec-i --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B")
      flag=$(row_flag "$knd" "$expg" "$gpu" "$threads" "$devs")
      if (( flagged_count > 0 )); then flag="${flag:-some iterations did not run as labelled} ($flagged_count/$ITERATIONS)"; fi
      [[ -n "$flag" ]] && flag=" ${RED}${flag}${RESET}"
      [[ "$failed" != 0 ]] && flag+=" ${RED}${failed} failed${RESET}"
      printf "${YELLOW}%-20s${RESET} %5s ${WHITE}%-22s${RESET} %10s ${CYAN}%9.4f${RESET} ${GREEN}%8.3f${RESET} %7s %5s %3s %4s%s\n" \
        "$cfg" "$cpus" "$item" "$size_h" "$secs" "$thr" "$ratio" "$(pct_show "$gpu")" "$devs" "$threads" "$flag"
    done
done
echo ""

#---------------------------------------------------------------------
# Fastest config per kind, item and CPU allowance -- and where the GPUs engaged
#---------------------------------------------------------------------
print_separator
echo "${BOLD}${YELLOW} FASTEST PER ITEM AND CPU ALLOWANCE${RESET}"
awk -F'\t' 'NR>1 && $13==0 && $15==0 {
    key = $3 "\t" $4 "\t" $2
    if (!(key in best) || $8 + 0 > bestv[key]) { best[key] = $1; bestv[key] = $8 + 0; gpu[key] = $10 }
  }
  END { for (k in best) { split(k, p, "\t"); printf "%s\t%s\t%s\t%s\t%.3f\t%s\n", p[1], p[2], p[3], best[k], bestv[k], gpu[k] } }' \
  "$RESULTS_FILE" | sort -t$'\t' -k1,1 -k2,2 -k3,3n \
  | while IFS=$'\t' read -r kind item cpus cfg thr gpu; do
    [[ "$gpu" =~ ^[0-9]+([.][0-9]+)?$ ]] && gpu="$(pct_show "$gpu")%"
    printf "  %-12s %-22s @%-4s ${DIM}->${RESET} ${YELLOW}%-20s${RESET} ${GREEN}%8.3f GiB/s${RESET}  GPU %s\n" \
      "$kind" "$item" "$cpus" "$cfg" "$thr" "$gpu"
  done
echo ""

if awk -F'\t' 'NR>1 && ($1=="default" || $1=="hybrid") && $10!="n/a" { f=1 } END { exit !f }' "$RESULTS_FILE"; then
  echo "${BOLD}${ORANGE} WHERE THE DEFAULT AND --hybrid STARTED THEIR GPUS${RESET}  ${DIM}(GPU% by CPU allowance)${RESET}"
  awk -F'\t' 'NR>1 && ($1=="default" || $1=="hybrid") && $10!="n/a" {
      share = $10
      if (share ~ /^[0-9]+([.][0-9]+)?$/) share = sprintf("%.1f", share)
      printf "  %-8s %-12s %-22s @%-4s GPU %s%% x%s\n", $1, $3, $4, $2, share, $11
    }' \
    "$RESULTS_FILE"
  echo ""
fi

flagged=$(tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r cfg cpus knd item _b _c _s _t _r gpu _d threads _f expg flagged_count _cold; do
  f=$(row_flag "$knd" "$expg" "$gpu" "$threads" "$_d")
  if (( flagged_count > 0 )); then echo "  $cfg @$cpus $knd $item: ${f:-some iterations did not run as labelled} ($flagged_count/$ITERATIONS)"; fi
done || true)
if [[ -n "$flagged" ]]; then
  echo "${BOLD}${RED} Rows that did not run as their label asks:${RESET}"
  echo "$flagged"
  echo "  ${DIM}Their numbers are not the request's.  A -d --tar --gpu-only parallel extract decoding on the"
  echo "  CPUs is current behaviour (ROADMAP 45(g)); otherwise check the GPUs (nvidia-smi) and gzstd -v.${RESET}"
  echo ""
fi

#---------------------------------------------------------------------
# JSON output
#---------------------------------------------------------------------
echo "Writing JSON results to $OUTPUT..."
GZSTD_VER=$("$GZSTD" --version 2>&1 | head -1)
python3 - "$RESULTS_FILE" "$OUTPUT" "$GZSTD_VER" "$NALLOWED" "$HAS_GPU_BUILD" "$ITERATIONS" \
  "$COLD" "$CALIBRATED" "${CPU_SET[*]}" "$GPU_SUMMARY" "$GPU_DRIVER" <<'PY' 2>&1 || echo "  (JSON output failed)"
import json, sys
tsv, out, ver, ncpu, gpu_build, iters, cold, calibrated, cpus, gpus, driver = sys.argv[1:12]
rows = []
with open(tsv) as f:
    header = f.readline().rstrip('\n').split('\t')
    for line in f:
        v = line.rstrip('\n').split('\t')
        if len(v) != len(header):
            continue
        r = dict(zip(header, v))
        for k in ('bytes', 'comp_bytes', 'failed', 'flagged'):
            try: r[k] = int(r[k])
            except ValueError: pass
        for k in ('median_secs', 'throughput_gibs', 'ratio_pct', 'gpu_pct'):
            try: r[k] = float(r[k])
            except ValueError: r[k] = None
        for k in ('gpu_devices', 'cpu_threads', 'cpus'):
            try: r[k] = int(r[k])
            except ValueError: r[k] = None
        rows.append(r)
json.dump({
    'gzstd_version': ver,
    'system': {'cpus_allowed': int(ncpu), 'gpu_build': gpu_build == 'true',
               'gpus': gpus, 'driver': driver},
    'method': {'iterations': int(iters), 'cold': cold, 'calibrated': calibrated == 'true',
               'cpu_allowances': cpus.split()},
    'results': rows,
}, open(out, 'w'), indent=2)
print('  %d results written.' % len(rows))
PY

echo ""
print_separator
echo "${BOLD}${WHITE} Benchmark complete!${RESET}"
print_separator
