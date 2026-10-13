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
// the batch while its blocks fit concurrently.  On the measured H100, batches
// of 8..256 frames were close in duration; a card with fewer resident blocks
// may need multiple waves.  Bigger batches amortise a fixed cost and get
// cheaper per byte until that point, while a bigger --chunk-mib gets linearly
// more expensive.  MEASURE at the batch and card you ship.
//
// THE FOUR CHAINS WERE WAITING ON MEMORY, NOT ON ARITHMETIC (v0.18.10).  Until
// then each frame got four threads and nothing else, and each thread loaded its
// own words: eight 8-byte loads in flight per thread, then eight dependent
// rounds, then the next eight loads.  MEASURED on an H100 at 16 MiB frames: 51-54
// ms per batch for 8..256 frames -- about 180 SM cycles per 32-byte stripe, where
// the same four chains with no loads at all (the arithmetic floor) take 8.7 ms.
// A --gpu-only decode of 4 GiB spent a third of its time there (ROADMAP 15).
//
// So a frame now gets a whole BLOCK.  All of its threads stream the frame into
// shared memory a tile ahead, in coalesced 8-byte loads, while threads 0..3 of
// warp 0 run the four chains out of shared memory.  Same chains, same order,
// bit-identical result; 12.0..12.9 ms for 8..256 16 MiB frames on an H100.
// The four compute lanes are still the algorithm's ceiling -- the block only
// keeps them fed.
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

// One BLOCK per frame (GZX_BLOCK threads); `out[chunk]` receives the low 32 bits
// of XXH64(frame, seed=0), which is precisely what zstd stores as the frame's
// Content_Checksum.
//
// PRECONDITION: `base` is 8-byte aligned and `stride` is a multiple of 8, so
// every frame starts on an 8-byte boundary and the tiles can be loaded as
// 64-bit words.  Only 8: the decompress caller's stride is 8-byte granular, so a
// 16-byte vector load would lack its required alignment on some frame starts.
// The tail path makes no alignment assumption.
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
//
// SIZES.  GZX_TILE stripes of 32 bytes per tile, two tiles (32 KiB of static
// shared memory, under the 48 KiB a block may declare on every card gzstd
// supports, Volta on).  MEASURED on an H100, 256 frames of 16 MiB: 512x256 12.9
// ms, 256x128 14.7 ms.  Turing occupancy is limited by the 32 KiB shared-memory
// allocation per block; this rewrite's speed on Turing has not been measured.
static constexpr int GZX_BLOCK = 256;
static constexpr int GZX_TILE  = 512;                  // stripes per tile
static constexpr int GZX_WORDS = GZX_TILE * 4;         // 64-bit words per tile

__global__ void __launch_bounds__(GZX_BLOCK)
gzx_xxh64_kernel(const unsigned char * __restrict__ base,
                 size_t stride,
                 const size_t * __restrict__ sizes,
                 size_t n_chunks,
                 unsigned int * __restrict__ out)
{
    __shared__ __align__(16) unsigned long long tile[2][GZX_WORDS];

    const size_t chunk = blockIdx.x;
    // The grid is exactly n_chunks blocks, so this never fires; if it did, the
    // WHOLE block would leave together, which is what keeps __syncthreads safe.
    if (chunk >= n_chunks) return;

    const size_t                len   = sizes[chunk];
    const unsigned char *       p     = base + chunk * stride;
    const unsigned long long *  src   = reinterpret_cast<const unsigned long long *>(p);
    const size_t                nblk  = len >> 5;                // whole 32-byte stripes
    const size_t                ntile = nblk / GZX_TILE;
    const unsigned              lane  = threadIdx.x;             // 0..3 compute

    unsigned long long v = 0;
    switch (lane) {                            // seed is always 0 here
        case 0:  v = GZX_P1 + GZX_P2; break;
        case 1:  v = GZX_P2;          break;
        case 2:  v = 0ULL;            break;
        case 3:  v = 0ULL - GZX_P1;   break;
        default: break;                        // a loader, not a chain
    }

    // Every thread loads a strided share of one tile; consecutive threads take
    // consecutive words, so each warp's load is one contiguous 256-byte run.
    auto fill = [&](size_t t, int b) {
        const unsigned long long * from = src + t * GZX_WORDS;
        unsigned long long *       to   = tile[b];
#pragma unroll 4
        for (int i = threadIdx.x; i < GZX_WORDS; i += GZX_BLOCK) to[i] = __ldg(from + i);
    };

    if (ntile) fill(0, 0);
    __syncthreads();
    for (size_t t = 0; t < ntile; ++t) {
        // The next tile's loads go out before this tile's rounds, so their
        // latency hides behind the chains instead of stalling them.
        if (t + 1 < ntile) fill(t + 1, (int)((t + 1) & 1));
        if (lane < 4) {
            // Accumulator j consumes word j of every stripe: stride 4 words.
            const unsigned long long * w = tile[t & 1] + lane;
#pragma unroll 16
            for (int j = 0; j < GZX_TILE; ++j) v = gzx_round(v, w[4 * j]);
        }
        // Both: this tile is free to be overwritten, and the next one is loaded.
        __syncthreads();
    }
    // The stripes after the last whole tile (fewer than GZX_TILE) straight from
    // global memory; at most one tile's worth of rounds, so not worth staging.
    if (lane < 4) {
        // Form an address only for a stripe that exists.  On an exact tile
        // boundary, a precomputed q would be beyond the last slot for lanes
        // 1..3 even though the remainder loop has no iterations.
        for (size_t k = ntile * GZX_TILE; k < nblk; ++k)
            v = gzx_round(v, __ldg(src + 4 * k + lane));
    }

    // Gather the four accumulators onto thread 0.  Lanes 0..3 are warp 0, and
    // every thread of warp 0 reaches this point, as __shfl_sync's full mask needs.
    const unsigned long long v1 = __shfl_sync(0xffffffffu, v, 0, 32);
    const unsigned long long v2 = __shfl_sync(0xffffffffu, v, 1, 32);
    const unsigned long long v3 = __shfl_sync(0xffffffffu, v, 2, 32);
    const unsigned long long v4 = __shfl_sync(0xffffffffu, v, 3, 32);
    if (threadIdx.x != 0) return;

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

    out[chunk] = (unsigned int)(h & 0xffffffffULL);
}

// Launch the batch hash on `stream`.  Ordered after whatever filled `base`
// (cuFileRead or H2D) by being on the same stream as that fill.
extern "C" void gzx_launch_xxh64(const void * base, size_t stride,
                                 const size_t * sizes, size_t n_chunks,
                                 unsigned int * out, cudaStream_t stream)
{
    if (n_chunks == 0) return;
    gzx_xxh64_kernel<<<(unsigned)n_chunks, GZX_BLOCK, 0, stream>>>(
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
