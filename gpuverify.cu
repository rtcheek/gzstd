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

// One-shot capability probe for the current device: can this binary actually
// launch the compare kernel here?  The kernel only carries device images for the
// architectures it was compiled for (CMAKE_CUDA_ARCHITECTURES) plus a PTX
// fallback the driver JITs; on a card outside that range a launch fails with
// cudaErrorNoKernelImageForDevice.  We do a trivial 1-chunk / 0-byte launch
// (the len=0 loop never dereferences a/b) and report whether it ran, so the
// caller can quietly fall back to CPU verify instead of aborting mid-compress.
//
//   1  = kernel ran (GPU verify usable)
//   0  = no compatible image for this device
//  -1  = probe could not run (no device / allocation failure) — caller should
//        also fall back to CPU verify
//
// Any CUDA error is consumed here so a 0/-1 result doesn't poison later calls.
extern "C" int gzv_kernel_available(void)
{
    int    * d_mismatch = nullptr;
    size_t * d_sizes    = nullptr;
    if (cudaMalloc(&d_mismatch, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    if (cudaMalloc(&d_sizes, sizeof(size_t)) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(d_mismatch);
        return -1;
    }
    cudaMemset(d_sizes, 0, sizeof(size_t));   // sizes[0] = 0 -> kernel does no work
    gzv_compare_kernel<<<dim3(1, 1, 1), 1>>>(nullptr, nullptr, d_sizes, 0, d_mismatch);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync   = (launch == cudaSuccess) ? cudaDeviceSynchronize() : launch;
    cudaFree(d_sizes);
    cudaFree(d_mismatch);
    cudaGetLastError();                        // swallow any residual error
    return (launch == cudaSuccess && sync == cudaSuccess) ? 1 : 0;
}
