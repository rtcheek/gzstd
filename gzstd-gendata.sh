#!/usr/bin/env bash
#======================================================================
# gzstd-gendata.sh  Generate test data for gzstd benchmarking
#
# Creates several files with different compressibility characteristics
# to test gzstd performance across varied workloads.
#
# Usage: ./gzstd-gendata.sh [output_dir] [size_mib]
#   output_dir  Directory for test files (default: ./gzstd-testdata)
#   size_mib    Approximate size of each file in MiB (default: 512)
#======================================================================
set -euo pipefail

OUTDIR="${1:-./gzstd-testdata}"
SIZE_MIB="${2:-512}"
SIZE_BYTES=$(( SIZE_MIB * 1024 * 1024 ))

mkdir -p "$OUTDIR"
echo "=== gzstd test data generator ==="
echo "Output dir : $OUTDIR"
echo "File size  : ${SIZE_MIB} MiB each"
echo ""

#----------------------------------------------------------------------
# 1. Highly compressible: repeated log line (~0.01% ratio)
#----------------------------------------------------------------------
echo "[1/5] Generating highly compressible data (repeated pattern)..."
LINE='2025-03-01T12:34:56.789Z INFO  [worker-42] Processing request id=abc123 method=GET path=/api/v2/users status=200 duration=12ms'
yes "$LINE" | dd of="$OUTDIR/high_compress.bin" bs=1M count="$SIZE_MIB" iflag=fullblock status=none 2>/dev/null
truncate -s "$SIZE_BYTES" "$OUTDIR/high_compress.bin"
echo "  -> $(numfmt --to=iec-i --suffix=B $(stat -c%s "$OUTDIR/high_compress.bin"))"

#----------------------------------------------------------------------
# 2. Medium compressibility: printable ASCII with structure (~30-40%)
#    Base is /dev/urandom filtered to printable chars, then every other
#    64K block is replaced with repeated English-like word sequences.
#----------------------------------------------------------------------
echo "[2/5] Generating medium compressibility data (pseudo-text)..."
{
  # Pre-build a 64K compressible word block
  WORDS="The server response data cache memory buffer thread process kernel system network packet stream compress decompress algorithm benchmark request error timeout configuration handler middleware serialize deserialize allocate deallocate initialize terminate dispatch validate authenticate authorize encrypt decrypt encode decode transform aggregate distribute replicate synchronize coordinate optimize parallelize schedule monitor analyze profile evaluate calibrate normalize standardize implement integrate deploy maintain support update upgrade migrate refactor document annotate"
  WBLOCK=""
  while [ ${#WBLOCK} -lt 65536 ]; do
    WBLOCK="${WBLOCK}${WORDS}. "
  done
  WBLOCK="${WBLOCK:0:65536}"

  WRITTEN=0
  TOGGLE=0
  while [ "$WRITTEN" -lt "$SIZE_BYTES" ]; do
    REMAINING=$(( SIZE_BYTES - WRITTEN ))
    CHUNK=65536
    if [ "$CHUNK" -gt "$REMAINING" ]; then CHUNK="$REMAINING"; fi
    if [ $(( TOGGLE % 3 )) -ne 0 ]; then
      # 2/3 of blocks: compressible word sequences
      printf '%.s' "$WBLOCK" | dd bs="$CHUNK" count=1 status=none 2>/dev/null
    else
      # 1/3 of blocks: random printable ASCII
      dd if=/dev/urandom bs="$CHUNK" count=1 status=none 2>/dev/null | tr '\0-\37\177-\377' 'A-Za-z0-9 ,.'
    fi
    WRITTEN=$(( WRITTEN + CHUNK ))
    TOGGLE=$(( TOGGLE + 1 ))
  done
} > "$OUTDIR/medium_compress.bin"
truncate -s "$SIZE_BYTES" "$OUTDIR/medium_compress.bin"
echo "  -> $(numfmt --to=iec-i --suffix=B $(stat -c%s "$OUTDIR/medium_compress.bin"))"

#----------------------------------------------------------------------
# 3. Low compressibility: mostly random, occasional repeated blocks
#    (~85-95% ratio)
#----------------------------------------------------------------------
echo "[3/5] Generating low compressibility data (structured random)..."
{
  # Pre-generate a 4K repeatable pattern
  PATTERN=$(dd if=/dev/urandom bs=4096 count=1 status=none 2>/dev/null | base64 | head -c 4096)

  WRITTEN=0
  TOGGLE=0
  while [ "$WRITTEN" -lt "$SIZE_BYTES" ]; do
    REMAINING=$(( SIZE_BYTES - WRITTEN ))
    CHUNK=65536
    if [ "$CHUNK" -gt "$REMAINING" ]; then CHUNK="$REMAINING"; fi
    if [ $(( TOGGLE % 10 )) -ne 0 ]; then
      # 90%: pure random
      dd if=/dev/urandom bs="$CHUNK" count=1 status=none 2>/dev/null
    else
      # 10%: repeated pattern block (gives zstd something to match)
      REPS=$(( CHUNK / 4096 + 1 ))
      for ((r=0; r<REPS; r++)); do printf '%s' "$PATTERN"; done | dd bs="$CHUNK" count=1 status=none 2>/dev/null
    fi
    WRITTEN=$(( WRITTEN + CHUNK ))
    TOGGLE=$(( TOGGLE + 1 ))
  done
} > "$OUTDIR/low_compress.bin"
truncate -s "$SIZE_BYTES" "$OUTDIR/low_compress.bin"
echo "  -> $(numfmt --to=iec-i --suffix=B $(stat -c%s "$OUTDIR/low_compress.bin"))"

#----------------------------------------------------------------------
# 4. Mixed workload: alternating compressible and random 64K sections
#    (~50-60% ratio)
#----------------------------------------------------------------------
echo "[4/5] Generating mixed workload data..."
{
  # Pre-build a 64K JSON-like compressible block
  RECORD='{"id":12345,"name":"user_42","score":87.50,"active":true,"tags":["alpha","beta","gamma"]}'
  JBLOCK=""
  while [ ${#JBLOCK} -lt 65536 ]; do
    JBLOCK="${JBLOCK}${RECORD}"$'\n'
  done
  JBLOCK="${JBLOCK:0:65536}"

  WRITTEN=0
  TOGGLE=0
  while [ "$WRITTEN" -lt "$SIZE_BYTES" ]; do
    REMAINING=$(( SIZE_BYTES - WRITTEN ))
    CHUNK=65536
    if [ "$CHUNK" -gt "$REMAINING" ]; then CHUNK="$REMAINING"; fi
    if [ $(( TOGGLE % 2 )) -eq 0 ]; then
      # Compressible block
      printf '%s' "$JBLOCK" | dd bs="$CHUNK" count=1 status=none 2>/dev/null
    else
      # Random block
      dd if=/dev/urandom bs="$CHUNK" count=1 status=none 2>/dev/null
    fi
    WRITTEN=$(( WRITTEN + CHUNK ))
    TOGGLE=$(( TOGGLE + 1 ))
  done
} > "$OUTDIR/mixed.bin"
truncate -s "$SIZE_BYTES" "$OUTDIR/mixed.bin"
echo "  -> $(numfmt --to=iec-i --suffix=B $(stat -c%s "$OUTDIR/mixed.bin"))"

#----------------------------------------------------------------------
# 5. Zero-filled (extreme compression, tests overhead)
#----------------------------------------------------------------------
echo "[5/5] Generating zero-filled data (extreme compression)..."
dd if=/dev/zero of="$OUTDIR/zeros.bin" bs=1M count="$SIZE_MIB" status=none 2>/dev/null
echo "  -> $(numfmt --to=iec-i --suffix=B $(stat -c%s "$OUTDIR/zeros.bin"))"

echo ""
echo "=== Test data ready in $OUTDIR ==="
ls -lhS "$OUTDIR/"
echo ""
echo "Total: $(numfmt --to=iec-i --suffix=B $(du -sb "$OUTDIR" | cut -f1))"
