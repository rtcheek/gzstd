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
# So match the shape instead: ANY call that takes the address of something AND
# states a width in its arguments — a bare 4 or 8 (the size of the on-disk fields:
# a magic, a frame size, an offset) or a `sizeof`. No callee names, no assumption
# about WHICH argument the destination is.
#
# FIVE patterns now, each of which looked complete when written:
#
#   1. `memcpy\(&[a-z_]+,`         — missed `memcpy(&m32, …)`: the DIGIT.
#   2. a sweep by eye              — missed two sites that `pread` into a uint32_t.
#   3. readers listed BY NAME      — missed `pr(&magic, 4, pos)`, a local lambda.
#   4. shape, but LITERAL width    — missed `pr(&magic, sizeof magic, pos)`.
#   5. shape, but FIRST ARGUMENT   — missed `pread(fd, &magic, 4, off)`, which is
#                                    the standard POSIX signature, while the
#                                    script's own comment claimed it caught pread.
#
# Widen on discovery. A false positive costs one ALLOW entry with a reason; a
# false negative ships a corrupt archive.
#
# WHAT THIS CANNOT DO — stated, because the last version of this comment claimed
# to catch `pread` and did not, and a check that overstates itself is worse than
# one that is merely narrow. It is a textual approximation of C++, not a parser.
# Known holes, each verified rather than guessed:
#
#   * A WIDTH HELD IN A VARIABLE: `pr(&magic, w4, 0)` is not matched, because the
#     width is what distinguishes a field read from every other out-parameter in
#     the file, and dropping that requirement matches hundreds of innocent calls.
#   * A read built through a struct overlay, a union, or `istream::read`.
#   * An address taken into a variable on one line and passed on another.
#
# So: a clean run means "no instance of the KNOWN SHAPES", not "no host-order
# read". This is a regression guard against the five forms that have actually
# occurred here, not a proof of absence — the `static_assert` in gzstd.cpp is what
# actually makes big-endian unreachable.
#
# \w+ for identifiers, NOT [a-z_]+: the narrower class is failure 1 above.
# Anchored on ARGUMENT POSITION: an address-of argument follows `(` or `,`. That
# is what distinguishes it from the logical `a && b`, which follows neither — and
# excluding `&&` matters, because without it every conjunction in the file matches
# and the check drowns in its own noise, which is how a lint gets switched off.
#
# Two earlier attempts at that exclusion were wrong in opposite directions: no
# guard at all (drowned), then a `[^&]&` guard that could not match `memcpy(&x, …)`
# because there is no character between the paren and the ampersand for it to
# consume — so it silently stopped catching the very first form on the list above.
#
# BOTH ORDERS.  The comments used to claim argument-order independence while the
# pattern required the width to FOLLOW the address, so
# `copy_n(src, 4, reinterpret_cast<char*>(&magic))` — width first, address inside
# a cast — was missed. That was failure ten. Two alternatives now, one per order.
addr='[(,][[:space:]]*&[[:space:]]*[A-Za-z_][A-Za-z0-9_.]*'
width='(\b[48]\b|sizeof)'
# The width-first arm additionally requires a CALL to have been opened, or
# `std::string * fields[4] = {&a, &b}` matches it — an array of pointers, whose
# `[4]` is the width and whose `, &b` is the address. The address-first arm needs
# no such prefix: it is already anchored on an argument separator.
pattern="(${addr}[^;]*${width}|\w+[[:space:]]*\([^;]*${width}[^;]*${addr})"

# Deliberate exceptions, in two lists, each entry justified.
#
# Note the asymmetry with the pattern above, which is the point: INCLUSION is by
# shape, because guessing how the next host-order read will be spelled has failed
# four times. EXCLUSION may be by name, because getting an exclusion wrong only
# produces a false positive — someone has to come here and justify it — whereas
# getting an inclusion wrong ships a corrupt archive. Wrong in the safe direction.

# Callees that cannot be reading an on-disk field, whatever their arguments look
# like. Keep this list SHORT and keep the reasons.
declare -a ALLOW_CALLEE=(
  'cudaMalloc'                   # &ptr out-param for a device allocation, not a read
  'cudaHostAlloc'                # same shape as cudaMalloc: &ptr out-param for a
                                 # page-locked HOST allocation. The width argument
                                 # is the allocation size, never an on-disk field.
  'cudaMemcpy'                   # device<->host copy; same endianness, no file
  'cudaMemcpyAsync'              # likewise — and it must be listed SEPARATELY:
                                 # exemptions are matched as `name(`, so the
                                 # cudaMemcpy entry above does NOT cover the
                                 # Async spelling.  v0.16.2 moved the drain's
                                 # readbacks onto their own stream and the
                                 # rename alone tripped this check.
  'localtime_r'                  # &time_t in, struct tm out; not a field read
  'nvmlDeviceGetHandleByIndex'   # &handle out-param from the NVML driver
  'getpwuid_r'                   # &passwd out-param; no on-disk integer field
  'getgrgid_r'                   # &group out-param; likewise
  'pthread_setaffinity_np'       # &cpu_set_t in; scheduling, not a field read
  'sched_getaffinity'            # &cpu_set_t OUT from the kernel; a CPU mask has
                                 # no on-disk representation and no byte order
  'strftime'                     # &tm in, formatted text out; no on-disk integer
)

# EXEMPTIONS SUBTRACT, THEY DO NOT EXCUSE.
#
# Both lists used to work by "if this appears anywhere on the line, skip the
# line" — and because a multi-line lambda folds into ONE logical line, a single
# permitted call anywhere inside it silenced every forbidden call in the same
# body:
#
#     outer([&] {
#       cudaMalloc(&p, 4);          // allowed …
#       pread(fd, &magic, 4, off);  // … and this went unreported
#     });
#
# So instead of skipping the line, each exemption is DELETED from the text and
# the remainder is re-tested. An allowed call can then only ever excuse itself.
# The deletion spans one level of nested parentheses, because `getpwuid_r((uid_t)
# uid, &pw, …)` has a cast in its arguments and a `[^)]*` scrub stopped at that
# inner `)`, leaving the rest of the call behind to match. Deeper nesting errs
# toward REPORTING, which is the right direction to be wrong in.

# Whole-text exemptions for things the callee list cannot express. Comment the
# exemption at the SITE as well, so it is visible where someone edits it.
declare -a ALLOW=(
  # Word-wise all-zero scan: the value is only ever compared against 0, which is
  # the same in every byte order, and it reads a data block, not an integer FIELD.
  'std::memcpy(&w, p + i * sizeof(size_t), sizeof(w));'
  # --calibrate corpus generation: WRITES a synthetic PRNG pattern into an
  # in-memory buffer that is then compressed as opaque bytes. Never read back as
  # an integer, never an on-disk field.
  'std::memcpy(&corpus[i], &x, 8);'
  # getrandom(2) filling a nonce: randomness has no byte order, and this reads
  # from the kernel rather than from a file. Used for quarantine/temp names.
  'reinterpret_cast<char *>(&value) + off'
)

# A call split across lines is still one call.  Fold each source file into LOGICAL
# lines — joining while parentheses are unbalanced — and report the line the call
# STARTED on, so `pread(fd,\n  &magic, 4, off);` is seen the same as the one-liner.
# Comment-only lines are dropped first so a `//` inside the fold cannot swallow the
# rest of the statement.
fold_logical_lines() {
  awk '
    { raw = $0
      sub(/^[[:space:]]*/, "", raw)
      if (raw ~ /^\/\//) { if (depth == 0) next; else raw = "" }
      line = $0
      if (depth == 0) { start = NR; acc = "" }
      acc = acc " " line
      n = gsub(/\(/, "(", line); m = gsub(/\)/, ")", line)
      depth += n - m
      if (depth <= 0) { depth = 0; print start ":" acc; acc = "" }
    }
  ' "$1"
}

for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "check-endian-reads: no such file: $f" >&2; exit 2; }
  while IFS=: read -r lineno line; do
    [[ -n "$lineno" ]] || continue
    # Subtract every exemption, then re-test what is left.
    scrub=$line
    for a in "${ALLOW[@]}"; do
      scrub=${scrub//"$a"/}
    done
    for c in "${ALLOW_CALLEE[@]}"; do
      # One level of nesting: `getpwuid_r((uid_t)uid, &pw, …)` has a cast in its
      # arguments, and a `[^)]*` scrub stopped at that inner `)` and left the rest
      # of the call behind to match.
      scrub=$(printf '%s' "$scrub" \
        | sed -E "s/(^|[^A-Za-z0-9_])${c}[[:space:]]*\([^()]*(\([^()]*\)[^()]*)*\)/\1/g")
    done
    printf '%s' "$scrub" | grep -qE "$pattern" || continue
    if (( status == 0 )); then
      echo "check-endian-reads: host-order read of an on-disk field — use rd_le32/rd_le64" >&2
      echo "  (if the value is genuinely byte-order independent, add it to ALLOW in $0" >&2
      echo "   WITH a reason, and comment the exemption at the site)" >&2
    fi
    echo "  $f:$lineno: $(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g' \
                          | cut -c1-160)" >&2
    status=1
  done < <(fold_logical_lines "$f" | grep -E "$pattern")
done

if (( status == 0 )); then
  echo "check-endian-reads: OK — no host-order on-disk reads in ${files[*]}"
fi
exit $status
