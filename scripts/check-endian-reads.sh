#!/usr/bin/env bash
# check-endian-reads.sh — fail on any HOST-ORDER read of an on-disk integer.
#
# gzstd is little-endian only (there is a static_assert in gzstd.cpp saying so),
# but every integer it reads from a file still goes through rd_le32/rd_le64 so
# the format handling is explicit rather than accidentally correct. This check
# is what keeps that true, and it exists because inspection did not:
#
#   v0.15.75  converted eighteen `memcpy(&v, p, 4)` sites to rd_le32 — and left
#             two behind that `pread` STRAIGHT INTO a uint32_t, because a grep
#             for "memcpy" cannot see a pread.
#   v0.15.76  a review found one more `memcpy(&m32, magic, 4)`. My own audit
#             pattern had been `memcpy\(&[a-z_]+,` — the DIGIT in `m32` fell
#             outside the character class, so the grep that was supposed to
#             prove the sweep complete silently skipped the one site it missed.
#
# Both misses came from hand-written patterns that looked thorough. This is the
# mechanical version: it matches the SHAPE (an address-of on a scalar passed to a
# byte-wise reader) with no assumption about the variable's name, and every
# exception has to be written down here rather than being invisible.
#
# Usage:  scripts/check-endian-reads.sh [file ...]     (default: gzstd.cpp)
set -uo pipefail

# Resolve any arguments against the CALLER's directory before moving to the repo
# root, so `scripts/check-endian-reads.sh /tmp/probe.cpp` works from anywhere.
files=()
for a in "$@"; do
  case "$a" in
    /*) files+=("$a") ;;
    *)  files+=("$PWD/$a") ;;
  esac
done
cd "$(dirname "$0")/.." || exit 2
(( ${#files[@]} )) || files=(gzstd.cpp)
status=0

# SHAPE, not names.
#
# The first draft of this check listed the readers by name (memcpy|pread|fread|…)
# and promptly failed its own test: the site it was written to catch is
# `pr(&magic, 4, pos)`, where `pr` is a LOCAL LAMBDA wrapping pread. Enumerating
# names reproduces, a third time, the mistake the whole file is about — assuming
# you know how the next one will be spelled.
#
# So match the shape instead: ANY call whose first argument is the address of a
# scalar and whose argument list contains a bare 4 or 8 (the width of the on-disk
# fields — a magic, a frame size, an offset). That catches memcpy, pread, fread
# and every wrapper anyone writes later, without naming any of them.
#
# \w+ for the identifier, NOT [a-z_]+: the narrower class is what let
# `memcpy(&m32, …)` through a hand-run audit, because of the digit.
pattern='\w+\s*\(\s*&\s*\w+\s*,[^;]*\b[48]\b'

# Deliberate exceptions, each justified. A line matching one of these is allowed.
# Keep the reason in the code as a comment too, so the exemption is visible where
# someone edits it and not only here.
declare -a ALLOW=(
  # Word-wise all-zero scans: the value is only ever compared against 0, which is
  # byte-order independent, and neither reads an on-disk integer FIELD.
  'std::memcpy(&w, p + i \* sizeof(size_t), sizeof(w));'
)

for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "check-endian-reads: no such file: $f" >&2; exit 2; }
  while IFS=: read -r lineno line; do
    [[ -n "$lineno" ]] || continue
    allowed=0
    for a in "${ALLOW[@]}"; do
      if [[ "$line" == *"${a//\\/}"* ]]; then allowed=1; break; fi
    done
    (( allowed )) && continue
    if (( status == 0 )); then
      echo "check-endian-reads: host-order read of an on-disk field — use rd_le32/rd_le64" >&2
      echo "  (if the value is genuinely byte-order independent, add it to ALLOW in $0" >&2
      echo "   WITH a reason, and comment the exemption at the site)" >&2
    fi
    echo "  $f:$lineno: $(echo "$line" | sed 's/^[[:space:]]*//')" >&2
    status=1
  done < <(grep -nE "$pattern" "$f" | grep -v '^\s*[0-9]*:\s*//')
done

if (( status == 0 )); then
  echo "check-endian-reads: OK — no host-order on-disk reads in ${files[*]}"
fi
exit $status
