#!/usr/bin/env bash
#======================================================================
# gzstd-gds-check.sh — is this host ready for `gzstd --gds-only`?
#
# --gds-only reads the input straight from NVMe into GPU memory by
# peer-to-peer DMA, so the uncompressed bytes never enter host memory.  It
# is narrow: it needs four things from the platform, and when any one is
# missing cuFile quietly falls back to a host bounce buffer that returns
# the right bytes at nearly the same speed.  That silent fallback is the
# whole reason this script exists.
#
# THE RULE THIS SCRIPT IS BUILT ON: every static signal can lie.
#   * cuFileBufRegister returning success does NOT mean transfers are
#     routed peer-to-peer -- it succeeds in compat mode too.
#   * The nvidia-fs module being LOADED does not mean it WORKS.  A kernel
#     update can leave it loading cleanly while every BAR1 map fails.
#   * An aligned-transfer count does NOT prove routing; it counts
#     eligibility.
# So the authority here is a REAL READ: run one, and require the kernel
# module's own BAR1 map counter to MOVE.  Every other check below exists
# only to explain WHY that read failed.
#
# Usage: ./gzstd-gds-check.sh [options]
#   --path DIR      directory to test on -- use the filesystem you will
#                   actually read from (default: current directory)
#   --gzstd PATH    gzstd binary (default: $GZSTD_BIN, ./build/gzstd, then PATH)
#   -h, --help      this help
#
# Exit: 0 = ready (a real peer-to-peer read was observed)
#       1 = not ready (every reason is printed, with what to do)
#       2 = usage error
#
# Runs entirely as an ordinary user.  It writes one temporary file under
# --path and removes it.  Nothing is installed, loaded or changed.
#======================================================================
VERSION="1.0"
set -uo pipefail

GZ="${GZSTD_BIN:-}"
TESTDIR="."
while [ $# -gt 0 ]; do
  case "$1" in
    --path)  TESTDIR="${2:-}"; shift 2 ;;
    --gzstd) GZ="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gzstd-gds-check.sh: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

if [ -z "$GZ" ]; then
  if [ -x ./build/gzstd ]; then GZ=./build/gzstd
  else GZ="$(command -v gzstd 2>/dev/null || true)"; fi
fi

fail=0; warn=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
no()   { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
note() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; warn=1; }
info() { printf '        %s\n' "$*"; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

printf '\033[1mgzstd-gds-check.sh v%s\033[0m — GPUDirect Storage readiness\n' "$VERSION"

# ---------------------------------------------------------------- binary
hdr "0. gzstd binary"
if [ -z "$GZ" ] || ! [ -x "$GZ" ]; then
  no "no gzstd binary found (set GZSTD_BIN or use --gzstd)"
  echo; echo "Cannot continue without gzstd."; exit 1
fi
ok "$GZ ($("$GZ" --version 2>/dev/null | head -1))"
if "$GZ" -V 2>&1 | grep -qi nvcomp; then
  ok "built with nvCOMP (GPU support compiled in)"
else
  no "this binary is CPU-only (built with USE_NVCOMP=OFF) -- --gds-only cannot work"
  info "Rebuild with -DUSE_NVCOMP=ON, or use a GPU build."
fi

# ------------------------------------------------------------ gate 1: BAR1
hdr "1. GPU: does the PCI BAR1 aperture cover VRAM? (needs resizable BAR)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  note "nvidia-smi not found -- cannot check BAR1; the functional test below still decides"
else
  bar_bad=0; bar_seen=0
  while IFS=, read -r idx name mem; do
    idx="${idx// /}"; mem="${mem// MiB/}"; mem="${mem// /}"
    b1=$(nvidia-smi -i "$idx" -q 2>/dev/null | awk '/BAR1 Memory Usage/{f=1} f&&/Total/{gsub(/[^0-9]/,"",$0); print; exit}')
    [ -z "${b1:-}" ] && continue
    bar_seen=1
    if [ "$b1" -ge "$mem" ] 2>/dev/null; then
      ok "GPU$idx$name: BAR1 ${b1} MiB >= VRAM ${mem} MiB"
    else
      no "GPU$idx$name: BAR1 ${b1} MiB < VRAM ${mem} MiB -- cannot do peer-to-peer"
      bar_bad=1
    fi
  done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)
  if [ "$bar_bad" = 1 ]; then
    info "This is the usual disqualifier on consumer cards (often 256 MiB fixed)."
    info "Enable Resizable BAR / Above-4G Decoding in firmware if the card supports it;"
    info "many GeForce cards cannot qualify at all.  Use --direct-stage instead."
  fi
  [ "$bar_seen" = 0 ] && note "no GPUs reported by nvidia-smi"
fi

# -------------------------------------------------------- gate 2: nvidia-fs
hdr "2. nvidia-fs kernel module"
NVFS_VER=""
if [ -r /sys/module/nvidia_fs/version ]; then
  NVFS_VER="$(cat /sys/module/nvidia_fs/version 2>/dev/null)"
  ok "loaded, version $NVFS_VER"
elif modinfo nvidia_fs >/dev/null 2>&1; then
  NVFS_VER="$(modinfo nvidia_fs 2>/dev/null | awk '/^version:/{print $2}')"
  no "built (version ${NVFS_VER:-unknown}) but NOT LOADED"
  info "Load it with: sudo modprobe nvidia-fs"
  info "If it does not load at boot, your vendor's package may omit"
  info "/etc/modules-load.d/nvidia-fs.conf -- NVIDIA's own package ships it."
else
  no "not installed"
  info "Install nvidia-fs-dkms (NVIDIA's CUDA repo has it), then: sudo modprobe nvidia-fs"
fi

# The version gate.  nvidia-fs before 2.26.6 marks its shadow-buffer VMA
# VM_IO, and the kernel's check_vma_flags() returns -EFAULT for VM_IO, so
# pinning that buffer fails and every BAR1 map errors out.  Older kernels
# happened to satisfy the pin on a fast path that never consulted VMA
# flags; newer ones do not.  The module still loads and still reports a
# healthy version -- only the maps fail.
if [ -n "$NVFS_VER" ]; then
  if [ "$(printf '%s\n2.26.6\n' "$NVFS_VER" | sort -V | head -1)" = "$NVFS_VER" ] \
     && [ "$NVFS_VER" != "2.26.6" ]; then
    note "nvidia-fs $NVFS_VER is older than 2.26.6"
    info "Versions before 2.26.6 fail on kernels that enforce VMA flags during"
    info "page pinning: the module loads fine and every BAR1 map returns -EFAULT."
    info "If the functional test below fails, upgrade nvidia-fs first."
  else
    ok "version $NVFS_VER has the VM_IO shadow-buffer fix (>= 2.26.6)"
  fi
fi

if [ -r /proc/driver/nvidia-fs/stats ]; then
  ok "/proc/driver/nvidia-fs/stats readable (counters available)"
else
  no "/proc/driver/nvidia-fs/stats not readable -- cannot verify routing"
  info "Without this counter nothing can prove transfers are peer-to-peer."
fi

# ------------------------------------------------------- gate 3: userspace
hdr "3. cuFile userspace"
if ldconfig -p 2>/dev/null | grep -q 'libcufile\.so'; then
  ok "libcufile present: $(ldconfig -p 2>/dev/null | grep -m1 'libcufile\.so' | awk '{print $NF}')"
else
  no "libcufile not found -- install gds-tools / libcufile"
fi

# ------------------------------------------------ gate 4: filesystem + O_DIRECT
hdr "4. filesystem at --path ($TESTDIR)"
if ! [ -d "$TESTDIR" ] || ! [ -w "$TESTDIR" ]; then
  no "$TESTDIR is not a writable directory"
else
  fstype="$(stat -f -c %T "$TESTDIR" 2>/dev/null || echo unknown)"
  case "$fstype" in
    ext2/ext3|ext4|xfs) ok "filesystem type: $fstype (cuFile-supported)" ;;
    tmpfs|overlayfs|nfs|fuseblk|zfs|btrfs)
        no "filesystem type: $fstype -- cuFile will not do peer-to-peer here"
        info "Test against a local ext4/xfs mount on the NVMe you will actually read from." ;;
    *)  note "filesystem type: $fstype (unrecognised; the functional test decides)" ;;
  esac
  # cuFile silently requires O_DIRECT; a mount that refuses it cannot work.
  if command -v python3 >/dev/null 2>&1; then
    if python3 - "$TESTDIR" 2>/dev/null <<'PY'
import os,sys,tempfile
d=sys.argv[1]
fd=None;p=None
try:
    f,p=tempfile.mkstemp(dir=d); os.close(f)
    fd=os.open(p,os.O_RDONLY|os.O_DIRECT); sys.exit(0)
except Exception: sys.exit(1)
finally:
    if fd is not None: os.close(fd)
    if p: os.unlink(p)
PY
    then ok "O_DIRECT accepted here"
    else no "O_DIRECT refused on this filesystem -- cuFile cannot use it"; fi
  fi
fi

# ------------------------------------------------------- THE FUNCTIONAL TEST
hdr "5. THE AUTHORITY: does a real read actually go peer-to-peer?"
if [ "$fail" = 1 ]; then
  info "Skipped -- a gate above already rules it out.  Fix those first."
else
  b1_before="$(awk '/Bar1-map/{for(i=1;i<=NF;i++) if($i ~ /^ok=/){sub(/ok=/,"",$i); print $i; exit}}' \
               /proc/driver/nvidia-fs/stats 2>/dev/null)"
  tmp="$(mktemp "${TESTDIR%/}/.gdscheck.XXXXXX" 2>/dev/null)"
  if [ -z "$tmp" ]; then
    no "cannot create a temporary file in $TESTDIR"
  else
    # Big enough that the read is real; small enough to be polite on a busy box.
    head -c 67108864 /dev/urandom > "$tmp" 2>/dev/null
    out="$tmp.zst"; err="$tmp.err"
    "$GZ" -f --gds-only "$tmp" -o "$out" >/dev/null 2>"$err"; rc=$?
    b1_after="$(awk '/Bar1-map/{for(i=1;i<=NF;i++) if($i ~ /^ok=/){sub(/ok=/,"",$i); print $i; exit}}' \
                /proc/driver/nvidia-fs/stats 2>/dev/null)"
    if [ "$rc" -eq 0 ] && [ -n "${b1_before:-}" ] && [ -n "${b1_after:-}" ] \
       && [ "$b1_after" -gt "$b1_before" ] 2>/dev/null; then
      ok "a real read went peer-to-peer (BAR1 maps $b1_before -> $b1_after)"
    elif [ "$rc" -eq 0 ]; then
      no "gzstd ran, but the BAR1 map counter did not move ($b1_before -> $b1_after)"
      info "Transfers are bouncing through host memory -- this is the silent fallback."
    else
      no "gzstd --gds-only refused or failed (exit $rc)"
      sed -n '1,6p' "$err" | sed 's/^/        /'
    fi
    rm -f "$tmp" "$out" "$err"
  fi
fi

# ---------------------------------------------------------------- verdict
hdr "VERDICT"
if [ "$fail" = 0 ]; then
  printf '  \033[32mREADY\033[0m — --gds-only works on this host.\n'
  [ "$warn" = 1 ] && info "(warnings above are worth reading, but nothing blocks you)"
  exit 0
fi
printf '  \033[31mNOT READY\033[0m — --gds-only cannot do peer-to-peer here.\n\n'
cat <<'EOF'
  This is not a gzstd defect and usually is not fixable in software: the
  four requirements above are platform properties.  gzstd refuses rather
  than pretending, because a silent host bounce is the failure this mode
  exists to avoid.

  USE --direct-stage INSTEAD.  It is the portable ~95% of --gds-only:
  O_DIRECT reads landing straight in the GPU staging buffer, with the
  same device-side checksum.  It needs NONE of the four requirements
  above -- no cuFile, no nvidia-fs, no resizable BAR, no special
  filesystem -- and on measured hosts it recovers most of the host-CPU
  saving, because peer-to-peer DMA itself was only ever a small part of it.

      gzstd --direct-stage BIGFILE -o BIGFILE.zst
EOF
exit 1
