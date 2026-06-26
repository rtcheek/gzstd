// gpuverify.cu — GPU-side raw byte compare for `--verify --gpu-only`.
//
// gzstd.cpp is g++-compiled (CUDA runtime API + nvCOMP library only, no custom
// kernels), which keeps the main translation unit `LANGUAGES CXX`.  The one thing
// nvCOMP can't do for us is compare the just-decompressed output against the
// original input in VRAM, so that single kernel lives here and is compiled by
// nvcc (only when HAVE_NVCOMP).  gzstd.cpp calls the extern "C" launcher.
//
// The check is a raw byte compare, not a checksum: it is absolute (no XXH64
// collision window).  A faulting GPU's compressed bytes will almost never
// decompress back to the exact original, so a mismatch is the expected signal;
// the driver then discards and rebuilds CPU-only, same as a hard GPU fault.

#include <cuda_runtime.h>
#include <cstddef>

// Compare the first `sizes[chunk]` bytes of each chunk of `a` against `b`, where
// chunk N's data starts at `N * stride`.  On any mismatch, set *mismatch = 1 — a
// benign race (every writer stores the identical value 1), so no atomics needed.
__global__ void gzv_compare_kernel(const unsigned char * __restrict__ a,
                                   const unsigned char * __restrict__ b,
                                   const size_t * __restrict__ sizes,
                                   size_t stride, int * mismatch)
{
    const size_t chunk = blockIdx.y;                 // one chunk per block-row
    const size_t len   = sizes[chunk];
    const unsigned char * pa = a + chunk * stride;
    const unsigned char * pb = b + chunk * stride;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < len; i += (size_t)gridDim.x * blockDim.x) {
        if (pa[i] != pb[i]) { *mismatch = 1; return; }
    }
}

// Launch the compare on `stream` (ordered after the verify-decompress on the same
// stream).  `mismatch` is a device int the caller pre-zeroes and reads back.
extern "C" void gzv_launch_compare(const void * a, const void * b,
                                   const size_t * sizes, size_t n_chunks,
                                   size_t stride, int * mismatch,
                                   cudaStream_t stream)
{
    if (n_chunks == 0) return;
    const int      block = 256;
    const unsigned gx    = 256;   // blocks per chunk; threads stride across the chunk
    dim3 grid(gx, (unsigned)n_chunks, 1);
    gzv_compare_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char *)a, (const unsigned char *)b, sizes, stride, mismatch);
}
