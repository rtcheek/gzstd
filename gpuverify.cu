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
// chunk N's data starts at `N * stride`.  On any mismatch, set *mismatch = 1.
//
// WRITTEN WITH atomicExch, NOT A PLAIN STORE.  Several blocks can find corruption
// at once, and this used to argue the race was benign because every writer stores
// the identical value.  Agreeing on the value does not make concurrent
// unsynchronised writes defined under the CUDA memory model -- it is a data race,
// and the compiler is entitled to assume races do not happen.  The store only
// executes on the mismatch path, which is the rare one, so the atomic costs
// nothing on a healthy run.
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
        if (pa[i] != pb[i]) { atomicExch(mismatch, 1); return; }
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

// ---------------------------------------------------------------------------
// XXH64 over VRAM-resident frames — the `--gds-only` content checksum.
//
// WHY THIS KERNEL HAS TO EXIST.  Every gzstd frame carries zstd's content
// checksum (low 32 bits of XXH64 of the UNCOMPRESSED frame), which is what makes
// a GPU archive self-verifying.  The normal GPU path computes it on the host
// during H2D staging, straight out of the Task's host buffer.  Under --gds-only
// there IS no host buffer: cuFileRead lands the bytes in VRAM and the CPU never
// sees them.  Hashing on the host would mean copying every byte back over PCIe,
// which is exactly the traffic --gds-only exists to remove, so the hash has to
// happen where the data already is.
//
// THE PARALLELISM CEILING IS FOUR, AND IT IS THE ALGORITHM'S, NOT OURS.  XXH64's
// main loop keeps four accumulators; accumulator j consumes bytes [32k+8j, +8)
// for every stripe k.  Each is a strictly sequential chain
//     acc = rotl(acc + in*P2, 31) * P1
// and rotl does not distribute over multiplication, so there is no closed form
// that composes two stripes — the chain cannot be split, tree-reduced, or
// skipped ahead.  Four independent chains per frame is the hard limit.
//
// WHAT THAT MEANS FOR WALL TIME.  A frame's hash costs (len/32) dependent
// rounds no matter how many threads are thrown at it, so the kernel's runtime
// scales with the CHUNK SIZE and is almost independent of how many frames are in
// the batch.  Frames all hash concurrently (a 258-frame batch is 1032 threads —
// nothing for any GPU), so bigger batches amortise a fixed cost and get cheaper
// per byte, while a bigger --chunk-mib gets linearly more expensive. Do not size
// this from a small batch and conclude it is slow. MEASURE at the batch you ship.
//
// Threads are assigned four-per-frame and consecutively, so a frame's four lanes
// read one contiguous 32-byte stripe (one memory sector) per iteration, and the
// four lanes of a frame always live in the same warp — which is what lets the
// final combine use __shfl_sync instead of shared memory.
// ---------------------------------------------------------------------------

static constexpr unsigned long long GZX_P1 = 11400714785074694791ULL;
static constexpr unsigned long long GZX_P2 = 14029467366897019727ULL;
static constexpr unsigned long long GZX_P3 =  1609587929392839161ULL;
static constexpr unsigned long long GZX_P4 =  9650029242287828579ULL;
static constexpr unsigned long long GZX_P5 =  2870177450012600261ULL;

__device__ __forceinline__ unsigned long long gzx_rotl(unsigned long long x, int r)
{ return (x << r) | (x >> (64 - r)); }

__device__ __forceinline__ unsigned long long gzx_round(unsigned long long acc,
                                                        unsigned long long in)
{ acc += in * GZX_P2; acc = gzx_rotl(acc, 31); acc *= GZX_P1; return acc; }

__device__ __forceinline__ unsigned long long gzx_merge(unsigned long long acc,
                                                        unsigned long long val)
{ acc ^= gzx_round(0ULL, val); acc = acc * GZX_P1 + GZX_P4; return acc; }

__device__ __forceinline__ unsigned long long gzx_avalanche(unsigned long long h)
{ h ^= h >> 33; h *= GZX_P2; h ^= h >> 29; h *= GZX_P3; h ^= h >> 32; return h; }

// Byte-wise little-endian assembly, mirroring xxh::rd8/rd4 in gzstd.cpp.  Used
// ONLY for the <32-byte tail, which starts at an arbitrary offset and therefore
// cannot be loaded as an aligned 64-bit word.
__device__ __forceinline__ unsigned long long gzx_rd8(const unsigned char * p)
{
    return  (unsigned long long)p[0]        | ((unsigned long long)p[1] << 8)
         | ((unsigned long long)p[2] << 16) | ((unsigned long long)p[3] << 24)
         | ((unsigned long long)p[4] << 32) | ((unsigned long long)p[5] << 40)
         | ((unsigned long long)p[6] << 48) | ((unsigned long long)p[7] << 56);
}
__device__ __forceinline__ unsigned int gzx_rd4(const unsigned char * p)
{
    return  (unsigned int)p[0]        | ((unsigned int)p[1] << 8)
         | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

// One frame per four consecutive threads; `out[chunk]` receives the low 32 bits
// of XXH64(frame, seed=0), which is precisely what zstd stores as the frame's
// Content_Checksum.
//
// PRECONDITION: `base` is 8-byte aligned and `stride` is a multiple of 8, so the
// main loop can use aligned 64-bit loads.  The tail path makes no alignment
// assumption.
//
// `base` is always cudaMalloc'd (256-byte aligned).  `stride` USED TO BE
// DESCRIBED HERE as "the GPU subchunk size, always a whole number of MiB", and
// that was true only of the compress-verify caller.  The DECOMPRESS verify
// caller passes alloc_decomp, which is the largest DECLARED frame size in the
// batch -- byte-granular, and odd for any archive whose frames are not whole
// MiB.  Four independently-compressed 1,000,003-byte frames concatenated is
// enough to produce it; no seek table required.  That caller now rounds the
// stride up to 8 explicitly, so the precondition holds again, but it holds
// because the caller maintains it and not because the shape guarantees it.
__global__ void gzx_xxh64_kernel(const unsigned char * __restrict__ base,
                                 size_t stride,
                                 const size_t * __restrict__ sizes,
                                 size_t n_chunks,
                                 unsigned int * __restrict__ out)
{
    const size_t   tid   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t   chunk = tid >> 2;          // four threads per frame
    const unsigned lane  = (unsigned)(tid & 3u);

    // NO EARLY RETURN.  __shfl_sync below needs every thread of the warp to take
    // part; a `return` for out-of-range threads would leave the trailing warp of
    // the last block short and make the exchange undefined.  Out-of-range threads
    // instead run the whole kernel over a zero-length frame and store nothing.
    const bool active = (chunk < n_chunks);
    size_t len = 0;
    if (active) len = sizes[chunk];

    const unsigned char * p    = base + (active ? chunk * stride : 0);
    const size_t          nblk = len >> 5;    // whole 32-byte stripes

    unsigned long long v;
    switch (lane) {                            // seed is always 0 here
        case 0:  v = GZX_P1 + GZX_P2; break;
        case 1:  v = GZX_P2;          break;
        case 2:  v = 0ULL;            break;
        default: v = 0ULL - GZX_P1;   break;
    }

    const unsigned long long * q =
        reinterpret_cast<const unsigned long long *>(p) + lane;

    // UNROLLED BY EIGHT FOR MEMORY-LEVEL PARALLELISM, NOT FOR LOOP OVERHEAD.
    // The accumulator chain is dependent but the LOADS are not, so the eight
    // __ldg's below issue back-to-back and their latencies overlap; the rolled
    // version stalled on each load in turn.  MEASURED on H100, 16 MiB frames:
    // 57 ms rolled vs the unrolled figure in the bench hook -- the arithmetic is
    // identical either way, so any difference is stall, not work.
    size_t k = 0;
    for (; k + 8 <= nblk; k += 8) {
        const unsigned long long a0 = __ldg(q +  0);
        const unsigned long long a1 = __ldg(q +  4);
        const unsigned long long a2 = __ldg(q +  8);
        const unsigned long long a3 = __ldg(q + 12);
        const unsigned long long a4 = __ldg(q + 16);
        const unsigned long long a5 = __ldg(q + 20);
        const unsigned long long a6 = __ldg(q + 24);
        const unsigned long long a7 = __ldg(q + 28);
        v = gzx_round(v, a0); v = gzx_round(v, a1);
        v = gzx_round(v, a2); v = gzx_round(v, a3);
        v = gzx_round(v, a4); v = gzx_round(v, a5);
        v = gzx_round(v, a6); v = gzx_round(v, a7);
        q += 32;
    }
    for (; k < nblk; ++k) { v = gzx_round(v, __ldg(q)); q += 4; }

    // Gather the four accumulators onto lane 0 of this group.  The group is four
    // consecutive threads and 4 divides the warp size, so a group never straddles
    // two warps and the source lanes are always in-warp.
    const unsigned wl    = threadIdx.x & 31u;
    const unsigned gbase = wl & ~3u;
    const unsigned long long v1 = __shfl_sync(0xffffffffu, v, gbase + 0, 32);
    const unsigned long long v2 = __shfl_sync(0xffffffffu, v, gbase + 1, 32);
    const unsigned long long v3 = __shfl_sync(0xffffffffu, v, gbase + 2, 32);
    const unsigned long long v4 = __shfl_sync(0xffffffffu, v, gbase + 3, 32);
    if (lane != 0) return;

    unsigned long long h;
    if (len >= 32) {
        h = gzx_rotl(v1, 1) + gzx_rotl(v2, 7) + gzx_rotl(v3, 12) + gzx_rotl(v4, 18);
        h = gzx_merge(h, v1); h = gzx_merge(h, v2);
        h = gzx_merge(h, v3); h = gzx_merge(h, v4);
    } else {
        h = GZX_P5;                            // seed + P5, seed == 0
    }
    h += (unsigned long long)len;

    // Tail: the same three loops as xxh::tail, in the same order.
    const unsigned char * t   = p + (nblk << 5);
    size_t                rem = len - (nblk << 5);
    while (rem >= 8) {
        h ^= gzx_round(0ULL, gzx_rd8(t));
        h  = gzx_rotl(h, 27) * GZX_P1 + GZX_P4;
        t += 8; rem -= 8;
    }
    if (rem >= 4) {
        h ^= (unsigned long long)gzx_rd4(t) * GZX_P1;
        h  = gzx_rotl(h, 23) * GZX_P2 + GZX_P3;
        t += 4; rem -= 4;
    }
    while (rem) {
        h ^= (unsigned long long)(*t) * GZX_P5;
        h  = gzx_rotl(h, 11) * GZX_P1;
        ++t; --rem;
    }
    h = gzx_avalanche(h);

    if (active) out[chunk] = (unsigned int)(h & 0xffffffffULL);
}

// Launch the batch hash on `stream`.  Ordered after whatever filled `base`
// (cuFileRead or H2D) by being on the same stream as that fill.
extern "C" void gzx_launch_xxh64(const void * base, size_t stride,
                                 const size_t * sizes, size_t n_chunks,
                                 unsigned int * out, cudaStream_t stream)
{
    if (n_chunks == 0) return;
    const int      block   = 128;                       // 32 frames per block
    const size_t   threads = n_chunks * 4;
    const unsigned grid    = (unsigned)((threads + block - 1) / block);
    gzx_xxh64_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char *)base, stride, sizes, n_chunks, out);
}

/*======================================================================
 gzp_finalize — stamp zstd frame headers/trailers IN VRAM
 -----------------------------------------------------------------------
 The ordinary compress drain copies each frame to the host and fixes it up
 there: set the Content_Checksum_flag in header byte 4, then append the
 4-byte XXH64-low32 trailer.  A pure peer-to-peer output path never brings
 the frame to the host, so the same two edits have to happen on the device.

 One thread per frame; the work is two byte writes, so the launch overhead
 dominates and a batched launch is the point -- 64 frames per batch would
 otherwise be 64 launches.
======================================================================*/
__global__ void gzp_finalize_kernel(unsigned char * __restrict__ pack,
                                    const unsigned long long * __restrict__ off,
                                    const unsigned long long * __restrict__ csz,
                                    const unsigned int * __restrict__ ck,
                                    int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned char * f = pack + off[i];
  const unsigned long long c = csz[i];
  if (c < 5) return;                  // not a well-formed frame; leave as-is
  f[4] |= 0x04;                       // Content_Checksum_flag
  const unsigned int v = ck[i];
  f[c + 0] = (unsigned char)( v        & 0xff);
  f[c + 1] = (unsigned char)((v >>  8) & 0xff);
  f[c + 2] = (unsigned char)((v >> 16) & 0xff);
  f[c + 3] = (unsigned char)((v >> 24) & 0xff);
}

extern "C" void gzp_launch_finalize(void * pack,
                                    const void * d_off, const void * d_csz,
                                    const void * d_ck, int n, void * stream)
{
  if (n <= 0) return;
  const int threads = 128;
  const int blocks  = (n + threads - 1) / threads;
  gzp_finalize_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (unsigned char *)pack,
      (const unsigned long long *)d_off,
      (const unsigned long long *)d_csz,
      (const unsigned int *)d_ck, n);
}
