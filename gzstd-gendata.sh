#!/usr/bin/env bash
#======================================================================
# gzstd-gendata.sh  Generate test data for gzstd benchmarking
#
# Creates several .bin files with different compressibility characteristics,
# then compresses each to a matching .bin.zst with gzstd (multi-frame defaults).
# The .bin.zst are the decompress-benchmark inputs: gzstd-benchmark.sh reads them
# (and the .bin) and writes only /dev/null, so the corpus is built once here and
# the benchmark never writes to the test disk.  The .bin/.bin.zst pair is always
# (re)created together so they match.
#
# Usage: ./gzstd-gendata.sh [size_mib] [output_dir]
#   size_mib    Approximate size of each file in MiB (default: 512)
#   output_dir  Directory for test files (default: ./gzstd-testdata)
#   GZSTD_BIN   env: path to the gzstd binary (default: ./build/gzstd, then PATH)
#
# Requirements: bash, python3, dd, numfmt, stat, gzstd
#======================================================================
VERSION="0.13.51"
#set -euo pipefail

SIZE_MIB="${1:-512}"
OUTDIR="${2:-./gzstd-testdata}"
SIZE_BYTES=$(( SIZE_MIB * 1024 * 1024 ))

# gzstd binary used to build the matching .bin.zst decompress inputs.
GZSTD_BIN="${GZSTD_BIN:-./build/gzstd}"
if [ ! -x "$GZSTD_BIN" ]; then
  if command -v gzstd >/dev/null 2>&1; then
    GZSTD_BIN="$(command -v gzstd)"
  else
    echo "ERROR: gzstd binary not found (looked at ./build/gzstd and PATH)." >&2
    echo "       Set GZSTD_BIN=/path/to/gzstd — it builds the .bin.zst decompress" >&2
    echo "       inputs that gzstd-benchmark.sh needs." >&2
    exit 1
  fi
fi

# Number of parallel jobs — capped at 5 (one per file)
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
JOBS=$(( NPROC > 5 ? 5 : NPROC ))

# Temp dir for progress sidecar files (one per generator)
PROGDIR=$(mktemp -d /tmp/gzstd_prog.XXXXXX)
trap 'rm -rf "$PROGDIR"' EXIT

mkdir -p "$OUTDIR"

#----------------------------------------------------------------------
# ANSI helpers
#----------------------------------------------------------------------
ESC=$'\033'
RESET="${ESC}[0m"
BOLD="${ESC}[1m"
DIM="${ESC}[2m"

# Foreground colors
RED="${ESC}[31m"
GREEN="${ESC}[32m"
YELLOW="${ESC}[33m"
CYAN="${ESC}[36m"
WHITE="${ESC}[37m"
BRIGHT_WHITE="${ESC}[97m"

# Cursor movement
cursor_up()    { printf "${ESC}[%dA" "$1"; }   # move up N lines
cursor_col0()  { printf "${ESC}[0G";           }   # move to column 0
erase_line()   { printf "${ESC}[2K";           }   # erase current line
hide_cursor()  { printf "${ESC}[?25l";         }
show_cursor()  { printf "${ESC}[?25h";         }

# Restore cursor on Ctrl-C
trap 'show_cursor; echo ""; exit 130' INT TERM

#----------------------------------------------------------------------
# Header banner
#----------------------------------------------------------------------
echo ""
printf "${BOLD}${BRIGHT_WHITE}=== gzstd test data generator v${VERSION} ===${RESET}\n"
printf "${DIM}Output dir : ${RESET}${CYAN}%s${RESET}\n"  "$OUTDIR"
printf "${DIM}File size  : ${RESET}${CYAN}%s MiB each${RESET}\n" "$SIZE_MIB"
printf "${DIM}Parallel   : ${RESET}${CYAN}%s jobs${RESET}\n" "$JOBS"
printf "${DIM}gzstd bin  : ${RESET}${CYAN}%s${RESET}\n" "$GZSTD_BIN"

if python3 -c "import numpy" 2>/dev/null; then
  NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)")
  printf "${DIM}NumPy      : ${RESET}${GREEN}available (v%s) — fast path active${RESET}\n" "$NUMPY_VER"
else
  printf "${DIM}NumPy      : ${RESET}${YELLOW}not found — using array.array fallback${RESET}\n"
  printf "${DIM}             ${RESET}${DIM}(pip install numpy for ~5x boost on medium_compress)${RESET}\n"
fi
echo ""

#----------------------------------------------------------------------
# Python generator template.
# Each generator writes its bytes_written count to a sidecar file
# ($PROGDIR/<slug>) after every chunk so the progress loop can read it.
# The sidecar contains a single ASCII integer with a newline.
#----------------------------------------------------------------------

gen_high_compress() {
  local path="$1" size="$2" prog="$3"
  python3 -c "
import os
path  = '$path'
size  = $size
prog  = '$prog'
chunk = 64 << 20
line  = b'2025-03-01T12:34:56.789Z INFO  [worker-42] Processing request id=abc123 method=GET path=/api/v2/users status=200 duration=12ms\n'
buf   = (line * ((chunk // len(line)) + 1))[:chunk]
written = 0
with open(path, 'wb') as f:
    while written < size:
        n = min(chunk, size - written)
        f.write(buf[:n])
        written += n
        open(prog,'w').write(str(written))
"
}

gen_medium_compress() {
  local path="$1" size="$2" prog="$3"
  python3 -c "
import os, array
path  = '$path'
size  = $size
prog  = '$prog'
chunk = 64 << 20
words = (b'The server response data cache memory buffer thread process kernel '
         b'system network packet stream compress decompress algorithm benchmark '
         b'request error timeout configuration handler middleware serialize '
         b'deserialize allocate deallocate initialize terminate dispatch validate '
         b'authenticate authorize encrypt decrypt encode decode transform aggregate '
         b'distribute replicate synchronize coordinate optimize parallelize schedule '
         b'monitor analyze profile evaluate calibrate normalize standardize implement '
         b'integrate deploy maintain support update upgrade migrate refactor document annotate. ')
cbuf = (words * ((chunk // len(words)) + 1))[:chunk]
written = 0
toggle  = 0
with open(path, 'wb') as f:
    while written < size:
        n = min(chunk, size - written)
        if toggle % 3 != 0:
            # 2/3 compressible word sequences
            f.write(cbuf[:n])
        else:
            # 1/3 random printable ASCII (0x20-0x7e):
            # Use NumPy if available (SIMD vectorised, ~100x vs generator),
            # otherwise fall back to array module (~20x vs generator).
            raw = os.urandom(n)
            try:
                import numpy as np
                arr = np.frombuffer(raw, dtype=np.uint8)
                f.write((0x20 + (arr % 95)).tobytes())
            except ImportError:
                a = array.array('B', raw)
                for i in range(len(a)): a[i] = 0x20 + (a[i] % 95)
                f.write(a)
        written += n
        toggle  += 1
        open(prog,'w').write(str(written))
"
}

gen_low_compress() {
  local path="$1" size="$2" prog="$3"
  python3 -c "
import os
path  = '$path'
size  = $size
prog  = '$prog'
chunk = 64 << 20
pattern = os.urandom(4096)
pbuf    = (pattern * ((chunk // len(pattern)) + 1))[:chunk]
written = 0
toggle  = 0
with open(path, 'wb') as f:
    while written < size:
        n = min(chunk, size - written)
        if toggle % 10 != 0:
            f.write(os.urandom(n))
        else:
            f.write(pbuf[:n])
        written += n
        toggle  += 1
        open(prog,'w').write(str(written))
"
}

gen_mixed() {
  local path="$1" size="$2" prog="$3"
  python3 -c "
import os
path   = '$path'
size   = $size
prog   = '$prog'
chunk  = 64 << 20
record = b'{\"id\":12345,\"name\":\"user_42\",\"score\":87.50,\"active\":true,\"tags\":[\"alpha\",\"beta\",\"gamma\"]}\n'
cbuf   = (record * ((chunk // len(record)) + 1))[:chunk]
written = 0
toggle  = 0
with open(path, 'wb') as f:
    while written < size:
        n = min(chunk, size - written)
        if toggle % 2 == 0:
            f.write(cbuf[:n])
        else:
            f.write(os.urandom(n))
        written += n
        toggle  += 1
        open(prog,'w').write(str(written))
"
}

gen_zeros() {
  local path="$1" size="$2" prog="$3"
  # dd doesn't support sidecar progress, so we wrap it in Python too
  python3 -c "
import os
path  = '$path'
size  = $size
prog  = '$prog'
chunk = 64 << 20
buf   = b'\x00' * chunk
written = 0
with open(path, 'wb') as f:
    while written < size:
        n = min(chunk, size - written)
        f.write(buf[:n])
        written += n
        open(prog,'w').write(str(written))
"
}

#----------------------------------------------------------------------
# File registry — ordered list for display
# Each entry: "slug|label|path|progfile"
#----------------------------------------------------------------------
FILES=(
  "high_compress|high_compress.bin |(highly compressible)  |$OUTDIR/high_compress.bin|$PROGDIR/high_compress"
  "medium_compress|medium_compress.bin|(medium compressible)  |$OUTDIR/medium_compress.bin|$PROGDIR/medium_compress"
  "low_compress|low_compress.bin  |(low compressible)     |$OUTDIR/low_compress.bin|$PROGDIR/low_compress"
  "mixed|mixed.bin         |(mixed workload)       |$OUTDIR/mixed.bin|$PROGDIR/mixed"
  "zeros|zeros.bin         |(zero-filled)          |$OUTDIR/zeros.bin|$PROGDIR/zeros"
)
NUM_FILES=${#FILES[@]}

declare -A PIDS      # slug -> pid
declare -A DONE      # slug -> 0/1
declare -A EXIT_RC   # slug -> exit code

#----------------------------------------------------------------------
# Launch all generators
#----------------------------------------------------------------------
for entry in "${FILES[@]}"; do
  IFS='|' read -r slug fname annotation fpath prog <<< "$entry"
  echo 0 > "$prog"
  fn="gen_${slug}"
  "$fn" "$fpath" "$SIZE_BYTES" "$prog" &
  PIDS[$slug]=$!
  DONE[$slug]=0
done

#----------------------------------------------------------------------
# Progress rendering helpers
#----------------------------------------------------------------------

# Read bytes written from sidecar; return 0 if missing/unreadable
read_prog() {
  local prog="$1"
  local val
  val=$(cat "$prog" 2>/dev/null)
  echo "${val:-0}"
}

# Color a percentage value: red < 33%, yellow < 66%, green >= 66%, bright green = done
pct_color() {
  local pct="$1" done="$2"
  if   [ "$done" -eq 1 ];   then printf "${GREEN}"
  elif [ "$pct" -ge 66 ];   then printf "${YELLOW}"
  elif [ "$pct" -ge 33 ];   then printf "${CYAN}"
  else                           printf "${WHITE}"
  fi
}

# Human-readable GiB with one decimal
to_gib() {
  python3 -c "print(f'{$1/1073741824:.1f}')"
}

# Draw all 5 progress rows. Called repeatedly; uses cursor_up to redraw in place.
# First call: just print. Subsequent calls: move cursor up NUM_FILES lines first.
FIRST_DRAW=1

draw_progress() {
  local all_bytes_total=$(( NUM_FILES * SIZE_BYTES ))
  local all_bytes_done=0

  [ "$FIRST_DRAW" -eq 0 ] && cursor_up "$NUM_FILES"
  FIRST_DRAW=0

  for entry in "${FILES[@]}"; do
    IFS='|' read -r slug fname annotation fpath prog <<< "$entry"

    local written pct is_done pid rc
    written=$(read_prog "$prog")
    is_done=${DONE[$slug]}
    pid=${PIDS[$slug]}

    # Check if process finished (non-blocking)
    if [ "$is_done" -eq 0 ]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        rc=$?
        EXIT_RC[$slug]=$rc
        DONE[$slug]=1
        is_done=1
        written=$SIZE_BYTES
      fi
    fi

    pct=$(( written * 100 / SIZE_BYTES ))
    [ "$pct" -gt 100 ] && pct=100
    all_bytes_done=$(( all_bytes_done + written ))

    cursor_col0; erase_line
    local col
    col=$(pct_color "$pct" "$is_done")

    if [ "$is_done" -eq 1 ]; then
      local rc_val=${EXIT_RC[$slug]:-0}
      if [ "$rc_val" -eq 0 ]; then
        printf "  ${GREEN}✔${RESET}  %-20s ${DIM}%-23s${RESET}  ${GREEN}${BOLD}%3d%%${RESET}  ${DIM}%s GiB${RESET}\n" \
          "$fname" "$annotation" 100 "$(to_gib "$SIZE_BYTES")"
      else
        printf "  ${RED}✘${RESET}  %-20s ${DIM}%-23s${RESET}  ${RED}${BOLD}FAILED (exit %d)${RESET}\n" \
          "$fname" "$annotation" "$rc_val"
      fi
    else
      printf "  ${DIM}…${RESET}  %-20s ${DIM}%-23s${RESET}  ${col}${BOLD}%3d%%${RESET}  ${DIM}%s / %s GiB${RESET}\n" \
        "$fname" "$annotation" "$pct" "$(to_gib "$written")" "$(to_gib "$SIZE_BYTES")"
    fi
  done
}

#----------------------------------------------------------------------
# Live progress loop
#----------------------------------------------------------------------
hide_cursor
printf "${BOLD}${WHITE}Progress:${RESET}\n"
echo ""

# Print initial rows (all 0%)
draw_progress

# Poll until all done
while true; do
  sleep 0.25

  # Count how many are still running
  running=0
  for entry in "${FILES[@]}"; do
    IFS='|' read -r slug _ _ _ <<< "$entry"
    [ "${DONE[$slug]}" -eq 0 ] && running=$(( running + 1 ))
  done

  draw_progress

  [ "$running" -eq 0 ] && break
done

show_cursor

#----------------------------------------------------------------------
# Compress each .bin -> .bin.zst with gzstd (multi-frame defaults).
# These are the decompress-benchmark inputs.  Recreated every run so the
# .bin and .bin.zst always match.  gzstd keeps the source .bin by default and
# names the output <file>.bin.zst, so a bare `gzstd --overwrite <file>` is all
# we need.  Sequential: each gzstd run already uses every core.
#----------------------------------------------------------------------
echo ""
printf "${BOLD}${WHITE}Compressing → .bin.zst:${RESET}\n"
CFAILED=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r slug fname annotation fpath prog <<< "$entry"
  if [ "${EXIT_RC[$slug]:-0}" -ne 0 ]; then
    printf "  ${YELLOW}–${RESET}  %-20s ${DIM}skipped (generation failed)${RESET}\n" "$fname"
    continue
  fi
  printf "  ${DIM}…${RESET}  %-20s ${DIM}compressing…${RESET}\r" "$fname"
  # -k/--keep: gzstd keeps the source by default, but pass it explicitly so we still
  # retain the .bin if that default ever changes (we need both .bin and .bin.zst).
  if "$GZSTD_BIN" --overwrite -k "$fpath" >/dev/null 2>&1; then
    zsz=$(stat -c%s "${fpath}.zst" 2>/dev/null || echo 0)
    cursor_col0; erase_line
    printf "  ${GREEN}✔${RESET}  %-20s ${DIM}→ %s.bin.zst  %s${RESET}\n" \
      "$fname" "$slug" "$(numfmt --to=iec-i --suffix=B "$zsz" 2>/dev/null || echo "${zsz}B")"
  else
    cursor_col0; erase_line
    printf "  ${RED}✘${RESET}  %-20s ${RED}compress failed${RESET}\n" "$fname"
    CFAILED=$(( CFAILED + 1 ))
  fi
done

#----------------------------------------------------------------------
# Final summary
#----------------------------------------------------------------------
echo ""
FAILED=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r slug _ _ _ <<< "$entry"
  rc=${EXIT_RC[$slug]:-0}
  [ "$rc" -ne 0 ] && FAILED=$(( FAILED + 1 ))
done

[ "$FAILED" -gt 0 ] && printf "${RED}WARNING: %d generator(s) failed.${RESET}\n\n" "$FAILED" >&2
[ "${CFAILED:-0}" -gt 0 ] && printf "${RED}WARNING: %d compression(s) failed.${RESET}\n\n" "$CFAILED" >&2

printf "${BOLD}${GREEN}=== Test data ready in %s ===${RESET}\n" "$OUTDIR"
ls -lhS "$OUTDIR/"
echo ""

# Portable du (Linux vs macOS)
TOTAL=$(du -sb "$OUTDIR" 2>/dev/null | cut -f1 || du -sk "$OUTDIR" | awk '{print $1*1024}')
printf "Total: ${CYAN}%s${RESET}\n" "$(numfmt --to=iec-i --suffix=B "$TOTAL")"
