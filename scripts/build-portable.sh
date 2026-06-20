#!/usr/bin/env bash
# build-portable.sh — produce a maximally-portable static gzstd binary.
#
# Combines:
#   * Build inside an old-glibc container (Ubuntu 20.04 → glibc 2.31)
#   * Static-link libnvcomp / libcudart_static / libstdc++ / libgcc / libzstd
#   * NVIDIA's official nvCOMP tarball (which ships libnvcomp_static.a;
#     conda-forge does not)
#
# Result: build-portable/gzstd, ~60–80 MB, runs on any Linux x86_64 with
# glibc ≥ 2.31 and an NVIDIA driver providing libnvidia-ml.so.1.
#
# Usage:   scripts/build-portable.sh [--nvcomp-version X.Y.Z.W]
#          scripts/build-portable.sh --help
#
# Requirements on the build host:
#   * docker (with --gpus support not required for the build itself)
#   * curl
#   * ~150 MB free disk for the nvCOMP tarball + build artifacts

set -euo pipefail

# -------- defaults --------
NVCOMP_VERSION="${NVCOMP_VERSION:-5.2.0.10}"
CUDA_MAJOR="${CUDA_MAJOR:-12}"           # cuda12 or cuda13 variant of nvcomp
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.6.0-devel-ubuntu20.04}"
BUILD_DIR="${BUILD_DIR:-build-portable}"
NVCOMP_CACHE="${NVCOMP_CACHE:-$HOME/.cache/gzstd-build/nvcomp}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  --nvcomp-version VER   nvCOMP version to download (default: $NVCOMP_VERSION)
  --cuda-major N         CUDA major version variant (default: $CUDA_MAJOR; choose 12 or 13)
  --base-image IMAGE     Docker image for the build host (default: $BASE_IMAGE)
  --build-dir DIR        Output build directory (default: $BUILD_DIR)
  --clean                Wipe build directory before building
  --no-cache             Re-download nvCOMP even if cached
  --help                 Show this message

Environment variables override defaults (NVCOMP_VERSION, CUDA_MAJOR, BASE_IMAGE,
BUILD_DIR, NVCOMP_CACHE).
EOF
}

CLEAN=false
NO_CACHE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcomp-version) NVCOMP_VERSION="$2"; shift 2 ;;
    --cuda-major)     CUDA_MAJOR="$2"; shift 2 ;;
    --base-image)     BASE_IMAGE="$2"; shift 2 ;;
    --build-dir)      BUILD_DIR="$2"; shift 2 ;;
    --clean)          CLEAN=true; shift ;;
    --no-cache)       NO_CACHE=true; shift ;;
    --help|-h)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# -------- preflight --------
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }

# Resolve repo root from script location so this works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

[[ -f "$REPO_ROOT/CMakeLists.txt" ]] || {
  echo "ERROR: $REPO_ROOT does not look like the gzstd repo (no CMakeLists.txt)"; exit 1;
}

# -------- fetch nvCOMP --------
NVCOMP_BASENAME="nvcomp-linux-x86_64-${NVCOMP_VERSION}_cuda${CUDA_MAJOR}-archive"
NVCOMP_TARBALL="${NVCOMP_BASENAME}.tar.xz"
NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/${NVCOMP_TARBALL}"
NVCOMP_DIR="$NVCOMP_CACHE/$NVCOMP_BASENAME"

mkdir -p "$NVCOMP_CACHE"
if $NO_CACHE; then
  rm -rf "$NVCOMP_DIR" "$NVCOMP_CACHE/$NVCOMP_TARBALL"
fi

if [[ ! -f "$NVCOMP_DIR/lib/libnvcomp_static.a" ]]; then
  echo "Fetching nvCOMP $NVCOMP_VERSION (cuda$CUDA_MAJOR) ..."
  if [[ ! -f "$NVCOMP_CACHE/$NVCOMP_TARBALL" ]]; then
    curl -L --fail -o "$NVCOMP_CACHE/$NVCOMP_TARBALL" "$NVCOMP_URL"
  fi
  echo "Extracting ..."
  tar -xJf "$NVCOMP_CACHE/$NVCOMP_TARBALL" -C "$NVCOMP_CACHE"
  [[ -f "$NVCOMP_DIR/lib/libnvcomp_static.a" ]] || {
    echo "ERROR: libnvcomp_static.a not found after extraction; tarball layout changed?"
    exit 1
  }
else
  echo "Using cached nvCOMP at $NVCOMP_DIR"
fi

# -------- clean output dir if requested --------
if $CLEAN; then
  rm -rf "$REPO_ROOT/$BUILD_DIR"
fi

# -------- build --------
echo "Building inside $BASE_IMAGE ..."
docker run --rm \
  -v "$REPO_ROOT":/src \
  -v "$NVCOMP_DIR":/nvcomp:ro \
  -w /src \
  -e BUILD_DIR="$BUILD_DIR" \
  -e DEBIAN_FRONTEND=noninteractive \
  -e TZ=Etc/UTC \
  "$BASE_IMAGE" \
  bash -c '
    set -e
    # tzdata pulls in a timezone prompt under interactive frontends; pin it
    # before apt-get install so the package installs unattended.
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
    echo Etc/UTC > /etc/timezone
    apt-get update -qq
    # NB: not installing cmake from apt — Ubuntu 20.04 ships 3.16, our
    # CMakeLists.txt requires 3.18+.  We download an official Kitware
    # binary instead.
    # libacl1-dev/libattr1-dev: static POSIX-ACL libs for --acls (they ship
    # libacl.a / libattr.a).  --xattrs needs only glibc.  Without these the
    # binary still builds but --acls is a silent no-op, so the release smoke
    # test asserts ACL support is actually present.
    apt-get install -y -qq --no-install-recommends \
      build-essential libzstd-dev libacl1-dev libattr1-dev \
      pkg-config xz-utils ca-certificates curl
    CMAKE_VER=3.30.5
    curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-x86_64.tar.gz" \
      | tar -xz -C /opt
    export PATH="/opt/cmake-${CMAKE_VER}-linux-x86_64/bin:$PATH"
    cmake --version
    cmake -B "$BUILD_DIR" \
      -DBUILD_STATIC=ON \
      -DNVCOMP_ROOT=/nvcomp \
      -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/gzstd-cfg.log
    # The release binary must ship with --acls/--xattrs.  ACL support is
    # optional in CMake (libacl not found -> still builds), so assert it was
    # actually enabled rather than silently shipping a no-op --acls.
    grep -q "ACL/xattr support: enabled" /tmp/gzstd-cfg.log || {
      echo "ERROR: ACL/xattr support was NOT enabled (libacl1-dev missing in build container)"
      exit 1
    }
    cmake --build "$BUILD_DIR" -j"$(nproc)"
  '

# -------- verify --------
BIN="$REPO_ROOT/$BUILD_DIR/gzstd"
[[ -x "$BIN" ]] || { echo "ERROR: build did not produce $BIN"; exit 1; }

echo
echo "=== Result ==="
ls -lh "$BIN"
echo
echo "=== Runtime dependencies (should not include libnvcomp/libstdc++/libzstd/libnvidia-ml) ==="
ldd "$BIN" || true
echo
echo "=== Static linkage (each should read 'static' — none should be a dynamic dep) ==="
for lib in libzstd libnvcomp libstdc++ libacl libattr libnvidia-ml; do
  if ldd "$BIN" 2>/dev/null | grep -q "$lib"; then
    printf "  %-14s DYNAMIC  <-- not statically linked!\n" "$lib:"
  else
    printf "  %-14s static\n" "$lib:"
  fi
done

echo
echo "=== ACL/xattr support ==="
# The in-container configure asserted 'ACL/xattr support: enabled' (the build
# aborts otherwise), so reaching here means --acls/--xattrs are compiled in.
echo "  compiled in : yes (configure assertion passed)"
if command -v setfacl >/dev/null 2>&1 && command -v setfattr >/dev/null 2>&1; then
  probe="$(mktemp -d)"
  if setfacl -m "u:$(id -u):rwx" "$probe" 2>/dev/null \
     && setfattr -n user.gz -v v "$probe" 2>/dev/null; then
    mkdir -p "$probe/src" "$probe/out"; echo data > "$probe/src/f"
    setfacl  -m "u:$(id -u):rwx" "$probe/src/f" 2>/dev/null
    setfattr -n user.gz -v v "$probe/src/f" 2>/dev/null
    if "$BIN" --cpu-only -q -f -o "$probe/a.tar.zst" --tar --acls --xattrs "$probe/src" 2>/dev/null \
       && "$BIN" --cpu-only -d -q --tar --acls --xattrs -C "$probe/out" "$probe/a.tar.zst" 2>/dev/null; then
      F="$(find "$probe/out" -type f -name f | head -1)"
      if getfacl -cpn "$F" 2>/dev/null | grep -q "user:$(id -u):rwx" \
         && [ "$(getfattr -n user.gz --only-values "$F" 2>/dev/null)" = v ]; then
        echo "  live test   : OK (ACL + xattr restored)"
      else
        echo "  live test   : FAILED — ACL/xattr not restored"
      fi
    else
      echo "  live test   : skipped (binary did not run on this host)"
    fi
  else
    echo "  live test   : skipped (this filesystem doesn't support ACL/xattr)"
  fi
  rm -rf "$probe"
else
  echo "  live test   : skipped (setfacl/setfattr not installed)"
fi

echo
echo "Output: $BIN"
echo "Copy to any Linux x86_64 with glibc >= 2.31 (NVIDIA driver optional: GPU paths engage when present)."
