# Building gzstd

## Prerequisites

- C++17 compiler (GCC 7+ or Clang 5+)
- CMake 3.18+
- Zstd (headers + library): `conda install zstd` or `apt install libzstd-dev`

**For GPU support (optional):**
- CUDA Toolkit 11+
- nvCOMP: `conda install -c conda-forge nvcomp`
- NVIDIA driver (optional at runtime — NVML is dlopen'd; without a driver gzstd runs CPU-only)

## Quick Start (from scratch)

```bash
git clone https://github.com/rtcheek/gzstd.git
cd gzstd
cmake -B build
cmake --build build -j$(nproc)
./build/gzstd --version
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_NVCOMP` | ON | Enable GPU backend (auto-disables if CUDA/nvCOMP not found) |
| `BUILD_STATIC` | OFF | Link statically for portable binaries |

### Shared Build (default)

```bash
cmake -B build
cmake --build build -j$(nproc)
```

Requires all shared libraries (zstd, nvcomp, cudart) to be on `LD_LIBRARY_PATH` or in the RPATH at runtime. The RPATH is automatically configured for `$ORIGIN`, conda prefixes, and common system paths.

### Static Build (portable)

```bash
cmake -B build -DBUILD_STATIC=ON
cmake --build build -j$(nproc)
```

Links zstd, libstdc++, libgcc, and CUDA runtime statically. The resulting binary only needs glibc at runtime; the NVIDIA driver is optional (CUDA and NVML are discovered at runtime, with CPU-only fallback when absent).

**Note:** the conda-forge `libnvcomp` package ships shared libraries only — no `libnvcomp_static.a`. With BUILD_STATIC=ON against a conda env, CMake will fall back to `libnvcomp.so.5` and the binary will still depend on it at runtime. To get a fully self-contained binary, fetch NVIDIA's official tarball (next section) which includes `libnvcomp_static.a`.

### Maximally Portable Build (single self-contained binary)

For distribution, you want one binary that runs on essentially any Linux x86_64 machine from 2020+, with or without an NVIDIA driver — no nvCOMP install, no CUDA toolkit, no libstdc++ ABI mismatch. The wrapper script `scripts/build-portable.sh` does this in one command:

```bash
scripts/build-portable.sh
```

What it does:
1. Downloads NVIDIA's official `nvcomp-linux-x86_64-5.2.0.10_cuda12-archive.tar.xz` from `developer.download.nvidia.com` (cached at `~/.cache/gzstd-build/nvcomp/` for re-runs).
2. Spins up a Docker container running `nvidia/cuda:12.6.0-devel-ubuntu20.04` (glibc 2.31 floor — covers Ubuntu 20.04+, Debian 11+, RHEL 8+).
3. Builds with `BUILD_STATIC=ON`, pointing CMake at the official tarball's `lib/libnvcomp_static.a`.
4. Statically links nvCOMP, the CUDA runtime, libstdc++, libgcc, and libzstd into a single executable.

Output: `build-portable/gzstd`, ~60–80 MB.

**Runtime requirements on target:**
- glibc ≥ 2.31 (Ubuntu 20.04 / Debian 11 / RHEL 8 / Fedora 33+)
- NVIDIA driver — optional. Since v0.13.55 NVML is loaded via dlopen at runtime, so the binary starts (and runs CPU-only) on machines without any NVIDIA driver. With a driver present, GPU paths work as usual.
- That's it. No CUDA, no nvCOMP, no Python, no nothing.

**Customizing:**

```bash
# pin a specific nvCOMP version
scripts/build-portable.sh --nvcomp-version 5.1.0.6

# target CUDA 13 instead of 12
scripts/build-portable.sh --cuda-major 13

# use a different base image (e.g., target older glibc)
scripts/build-portable.sh --base-image nvidia/cuda:12.6.0-devel-ubuntu18.04

# rebuild from scratch
scripts/build-portable.sh --clean

# re-download nvCOMP (don't use cache)
scripts/build-portable.sh --no-cache

scripts/build-portable.sh --help
```

**Why nvCOMP from NVIDIA's CDN, not conda-forge:**
conda-forge's `libnvcomp` package omits the static archive. NVIDIA's official redistributable tarball at https://developer.download.nvidia.com/compute/nvcomp/redist/ includes both the `.so` and `libnvcomp_static.a` (~57 MB containing CUDA kernel code).

### Automated release builds (GitHub Actions)

`.github/workflows/release-portable.yml` runs the same recipe in CI on every tag push:

```bash
# Tag a release; the workflow builds and attaches the binary automatically.
git tag v0.13.2
git push origin v0.13.2
```

The workflow:
1. Checks out the repo on `ubuntu-22.04`
2. Restores the cached nvCOMP tarball (key includes the version, so a bump invalidates automatically)
3. Runs `scripts/build-portable.sh`
4. Smoke-tests the binary (CPU round-trip — GitHub-hosted runners don't have GPUs)
5. Packages `gzstd-<version>-linux-x86_64.tar.gz` with the binary + LICENSE + CHANGELOG.md
6. Generates a SHA-256 checksum file
7. On tag push: attaches both files to the GitHub release (creates the release if needed, with auto-generated release notes)
8. On manual run (Actions tab → Run workflow): uploads as a workflow artifact instead, so you can test the build without cutting a real release

Required repo settings: under Settings → Actions → General → Workflow permissions, "Read and write permissions" must be enabled (so the workflow can publish releases).

### Installing (bundling shared libs)

When `BUILD_STATIC` is ON and nvCOMP is shared, use the install target to bundle `libnvcomp.so` alongside the binary:

```bash
cmake --install build --prefix ./dist
```

This produces:

```
dist/
  bin/
    gzstd              # the binary
    libnvcomp.so.5     # symlink -> libnvcomp.so.5.x.x
    libnvcomp.so.5.x.x # real shared library
```

Copy the entire `dist/bin/` directory to the target machine. The `$ORIGIN` RPATH ensures the binary finds `libnvcomp.so.5` next to itself.

### CPU-Only Build

```bash
cmake -B build -DUSE_NVCOMP=OFF
cmake --build build -j$(nproc)
```

No CUDA or GPU dependencies required.

## Conda Environment Example

```bash
conda create -n gzstd -c conda-forge zstd nvcomp cudatoolkit
conda activate gzstd
cmake -B build
cmake --build build -j$(nproc)
```

## Deploying to Another Machine

**Option 1: Install target (recommended)**
```bash
cmake --install build --prefix ./dist
scp -r dist/bin/ user@target:~/gzstd/
```

**Option 2: Manual copy**
```bash
scp build/gzstd user@target:~/gzstd/
scp $CONDA_PREFIX/lib/libnvcomp.so.5 user@target:~/gzstd/
```

Both options work because the binary's RPATH includes `$ORIGIN`.

## Verifying the Build

```bash
# Check version and GPU support
./build/gzstd --version

# Run the test suite
./gzstd-test.sh

# Run benchmarks
./gzstd-benchmark.sh
```
