# Building gzstd

## Prerequisites

- C++17 compiler (GCC 7+ or Clang 5+)
- CMake 3.18+
- Zstd (headers + library): `conda install zstd` or `apt install libzstd-dev`

**For GPU support (optional):**
- CUDA Toolkit 11+
- nvCOMP: `conda install -c conda-forge nvcomp`
- NVIDIA driver with NVML

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

Links zstd, libstdc++, libgcc, and CUDA runtime statically. The resulting binary only needs glibc and the NVIDIA driver at runtime.

**Note:** nvCOMP does not ship a static library in the conda-forge package. CMake will warn if it falls back to the shared `libnvcomp.so`. In this case the binary still depends on `libnvcomp.so.5` at runtime.

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
