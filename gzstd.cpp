// gzstd.cpp — Hybrid CPU+GPU Zstd (adaptive share)
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2024-2026 rtcheek
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
static constexpr const char * GZSTD_VERSION = "0.14.43";
//
// Architecture overview:
//
//   COMPRESSION:
//   Input is read in large chunks and split into tasks (one Zstd frame each).
//   Tasks are enqueued in a TaskQueue and processed by parallel workers:
//     - CPU workers: call ZSTD_compress2() per chunk, one CCtx per thread
//     - GPU workers: batch chunks onto CUDA streams via nvCOMP batched API
//     - Hybrid mode: adaptive scheduler splits work between CPU+GPU based
//       on observed throughput (EMA-smoothed, updated every ~200ms)
//   A writer thread reassembles compressed frames in original sequence order.
//
//   DECOMPRESSION:
//   Input is read entirely into memory, then split into individual Zstd frames
//   using ZSTD_findFrameCompressedSize() and ZSTD_getFrameContentSize().
//   Multi-frame files are decompressed in parallel (CPU MT or GPU batched).
//   Single-frame files or files with unknown content sizes fall back to
//   streaming CPU decompression.
//
//   DATA FLOW:  input -> [frame splitter / chunker] -> TaskQueue
//                 -> [CPU workers | GPU workers] -> ResultStore
//                 -> [writer thread] -> output
//
// Piping support: stdin/stdout binary mode, SIGPIPE handling, auto-detect pipes
//
// Pretty, uniform verbose output (-vv / -vvv) for hybrid mode (CPU + GPU)
//  - Consistent tagging:
//      * GPU:   [GPU{dev}/S{stream}] ...
//      * CPU:   [CPU/T{thread}] ...
//      * RESCUE (CPU fallback): [RESCUE/T{thread}] ...
//  - -vv : per-worker summaries and per-GPU per-batch submit/complete lines
//  - -vvv: per-CPU-chunk lines

#include <zstd.h>
#include <zstd_errors.h>
#include <cstdio>
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>   // mallopt / M_MMAP_THRESHOLD — recycle frame buffers (see main)
#endif
#include <cstring>
#include <cstdint>
#include <climits>    // INT_MAX — used in RAM-frame caps; pulled in transitively only with nvCOMP
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <stdexcept>
#include <memory>
#include <functional>
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>        // opendir/readdir — tar-mode directory walk
#include <fnmatch.h>       // --exclude glob matching (GNU tar semantics)
#include <grp.h>           // getgrgid_r — gname for tar headers
#include <pwd.h>           // getpwuid_r — uname for tar headers
#include <sys/sysmacros.h> // major()/minor() — device-node tar headers
#include <sys/xattr.h>     // --xattrs: l/f getxattr/setxattr/listxattr (glibc)
#ifdef GZSTD_HAVE_ACL
#include <sys/acl.h>       // --acls: POSIX.1e ACLs (libacl)
#include <acl/libacl.h>    // acl_extended_file_nofollow, acl_equiv_mode, acl_to_any_text
#endif
#endif
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#ifdef _WIN32
 #include <io.h>
#endif
#ifdef HAVE_NVCOMP
 #include <cuda_runtime.h>
 #include <nvcomp/zstd.h>
 #ifdef HAVE_NVML
 #include <dlfcn.h>

/*======================================================================
 NVML runtime loader (v0.13.55)
 -----------------------------------------------------------------------
 NVML ships with the NVIDIA *driver* — there is no static archive, and
 linking the CUDA toolkit's stub still writes DT_NEEDED libnvidia-ml.so.1
 into the binary.  The ELF loader resolves DT_NEEDED at startup, before
 main(), so the "portable" release binary refused to start on any machine
 without the driver ("error while loading shared libraries") — exactly
 the machines a portable build must support.

 Load it with dlopen at first use instead: no link-time dependency at
 all.  With the driver present, behaviour is identical to linking.
 Without it, every wrapper returns GZ_NVML_UNAVAILABLE and the callers'
 existing fallbacks take over (free-VRAM device ranking, /sys PCIe-gen
 probe, util_scale staying at 1.0).

 The types below mirror nvml.h's stable v1 ABI for exactly the calls
 this file makes; the wrappers reuse the official function names so call
 sites compile unchanged.  If you add a new NVML call, extend GzNvmlApi
 and add a wrapper here.
======================================================================*/
typedef int nvmlReturn_t;
static constexpr nvmlReturn_t NVML_SUCCESS = 0;
static constexpr nvmlReturn_t GZ_NVML_UNAVAILABLE = 999;  // driver/library absent
typedef struct nvmlDevice_st * nvmlDevice_t;
typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;
typedef struct { unsigned long long total, free, used; } nvmlMemory_t;

struct GzNvmlApi {
  nvmlReturn_t (*Init)(void) = nullptr;
  nvmlReturn_t (*Shutdown)(void) = nullptr;
  nvmlReturn_t (*DeviceGetCount)(unsigned int *) = nullptr;
  nvmlReturn_t (*DeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *) = nullptr;
  nvmlReturn_t (*DeviceGetHandleByPciBusId)(const char *, nvmlDevice_t *) = nullptr;
  nvmlReturn_t (*DeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *) = nullptr;
  nvmlReturn_t (*DeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *) = nullptr;
  nvmlReturn_t (*DeviceGetCpuAffinity)(nvmlDevice_t, unsigned int, unsigned long *) = nullptr;
  nvmlReturn_t (*DeviceGetMaxPcieLinkGeneration)(nvmlDevice_t, unsigned int *) = nullptr;
};

// Resolve once, thread-safely (magic static).  RTLD_LOCAL keeps the driver's
// symbols out of our global namespace.
static const GzNvmlApi & gz_nvml()
{
  static const GzNvmlApi api = []{
    GzNvmlApi a{};
    void * h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h) h = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
    if (!h) return a;  // no driver: all pointers stay null
    auto sym = [&](const char * v2, const char * v1) -> void * {
      void * p = dlsym(h, v2);
      return p ? p : (v1 ? dlsym(h, v1) : nullptr);
    };
    a.Init        = (nvmlReturn_t(*)(void))sym("nvmlInit_v2", "nvmlInit");
    a.Shutdown    = (nvmlReturn_t(*)(void))sym("nvmlShutdown", nullptr);
    a.DeviceGetCount = (nvmlReturn_t(*)(unsigned int *))
        sym("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
    a.DeviceGetHandleByIndex = (nvmlReturn_t(*)(unsigned int, nvmlDevice_t *))
        sym("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
    a.DeviceGetHandleByPciBusId = (nvmlReturn_t(*)(const char *, nvmlDevice_t *))
        sym("nvmlDeviceGetHandleByPciBusId_v2", "nvmlDeviceGetHandleByPciBusId");
    a.DeviceGetUtilizationRates = (nvmlReturn_t(*)(nvmlDevice_t, nvmlUtilization_t *))
        sym("nvmlDeviceGetUtilizationRates", nullptr);
    a.DeviceGetMemoryInfo = (nvmlReturn_t(*)(nvmlDevice_t, nvmlMemory_t *))
        sym("nvmlDeviceGetMemoryInfo", nullptr);
    a.DeviceGetCpuAffinity = (nvmlReturn_t(*)(nvmlDevice_t, unsigned int, unsigned long *))
        sym("nvmlDeviceGetCpuAffinity", nullptr);
    a.DeviceGetMaxPcieLinkGeneration = (nvmlReturn_t(*)(nvmlDevice_t, unsigned int *))
        sym("nvmlDeviceGetMaxPcieLinkGeneration", nullptr);
    return a;
  }();
  return api;
}

// Same-name wrappers — every call site in this file compiles unchanged.
static inline nvmlReturn_t nvmlInit_v2(void)
{ auto & a = gz_nvml(); return a.Init ? a.Init() : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlShutdown(void)
{ auto & a = gz_nvml(); return a.Shutdown ? a.Shutdown() : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int * n)
{ auto & a = gz_nvml(); return a.DeviceGetCount ? a.DeviceGetCount(n) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int i, nvmlDevice_t * d)
{ auto & a = gz_nvml(); return a.DeviceGetHandleByIndex ? a.DeviceGetHandleByIndex(i, d) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int i, nvmlDevice_t * d)
{ return nvmlDeviceGetHandleByIndex(i, d); }
static inline nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char * id, nvmlDevice_t * d)
{ auto & a = gz_nvml(); return a.DeviceGetHandleByPciBusId ? a.DeviceGetHandleByPciBusId(id, d) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t d, nvmlUtilization_t * u)
{ auto & a = gz_nvml(); return a.DeviceGetUtilizationRates ? a.DeviceGetUtilizationRates(d, u) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t * mem)
{ auto & a = gz_nvml(); return a.DeviceGetMemoryInfo ? a.DeviceGetMemoryInfo(d, mem) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t d, unsigned int sz, unsigned long * set)
{ auto & a = gz_nvml(); return a.DeviceGetCpuAffinity ? a.DeviceGetCpuAffinity(d, sz, set) : GZ_NVML_UNAVAILABLE; }
static inline nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t d, unsigned int * gen)
{ auto & a = gz_nvml(); return a.DeviceGetMaxPcieLinkGeneration ? a.DeviceGetMaxPcieLinkGeneration(d, gen) : GZ_NVML_UNAVAILABLE; }
 #endif // HAVE_NVML
#endif

namespace fs = std::filesystem;

/*======================================================================
 Platform: binary mode for stdin/stdout, SIGPIPE handling
======================================================================*/
static void set_binary_mode(FILE * f)
{
#ifdef _WIN32
  _setmode(_fileno(f), _O_BINARY);
#else
  (void)f; // POSIX doesn't distinguish text/binary
#endif
}

/*======================================================================
 Temp file cleanup  remove partial output on error / signal
======================================================================*/
// Global path to the in-progress temp file. Empty when no temp file is active.
// Accessed from signal handlers, so we use a simple C string + atomic guard.
static char g_tmp_path[4096] = {};
static volatile sig_atomic_t g_tmp_active = 0;

static void cleanup_tmp_file()
{
  if (g_tmp_active && g_tmp_path[0] != '\0') {
    // Runs from the SIGINT/SIGTERM handler: use unlink() (async-signal-safe)
    // rather than std::remove(), whose libc wrapper isn't guaranteed safe to
    // call from a signal handler.  best-effort; ignore errors.
#ifndef _WIN32
    ::unlink(g_tmp_path);
#else
    std::remove(g_tmp_path);
#endif
    g_tmp_active = 0;
    g_tmp_path[0] = '\0';
  }
}

static void register_tmp_file(const std::string & path)
{
  if (path.size() < sizeof(g_tmp_path)) {
    std::strncpy(g_tmp_path, path.c_str(), sizeof(g_tmp_path) - 1);
    g_tmp_path[sizeof(g_tmp_path) - 1] = '\0';
    g_tmp_active = 1;
  }
}

static void clear_tmp_file()
{
  g_tmp_active = 0;
  g_tmp_path[0] = '\0';
}

// Signal handler for SIGINT / SIGTERM: clean up temp file, then re-raise
// to get the correct exit status (128 + signum) for the parent process.
static void signal_cleanup_handler(int signum)
{
  cleanup_tmp_file();
  // Restore default handler and re-raise so the shell sees the signal exit
  std::signal(signum, SIG_DFL);
  std::raise(signum);
}

static void setup_signal_handlers()
{
#ifndef _WIN32
  // Ignore SIGPIPE so writing to a closed pipe returns an error
  // instead of killing the process  (critical for: gzstd | head)
  std::signal(SIGPIPE, SIG_IGN);
#endif
  // Clean up temp files on interrupt / termination
  std::signal(SIGINT, signal_cleanup_handler);
  std::signal(SIGTERM, signal_cleanup_handler);

  // atexit covers die() and any other non-signal abnormal exit
  std::atexit(cleanup_tmp_file);
}

/*======================================================================
 Constants
======================================================================*/
static const size_t ONE_MIB = size_t(1024) * size_t(1024);
static const size_t DEFAULT_CHUNK_MIB = 16;
// A first frame larger than this is treated as a single-frame file (zstd /
// --sliding-window) and streamed directly from the file rather than buffered
// through the queue.  Sits well above gzstd's largest practical chunk
// (--ultra auto-bumps to 128 MiB) so genuinely multi-frame chunked inputs keep
// the parallel queue path (and the v0.13.1 multi-frame-oversize guard).
static const size_t SINGLE_FRAME_STREAM_MIN = size_t(256) * ONE_MIB;
#ifdef HAVE_NVCOMP
static const size_t GPU_SUBCHUNK_MAX = size_t(16) * ONE_MIB; // max GPU subchunk
static const size_t DEFAULT_GPU_BATCH_CAP = 8;    // per device  smaller batches launch sooner
static const size_t DEFAULT_GPU_DECOMP_BATCH_CAP = 16;  // sweet spot: amortizes kernel launch without starving writer
// Upper bound on the per-stream buffer when --gpu-batch is NOT user-pinned and
// the shared auto-tuner is active.  Sized to give the tuner real room to grow
// from DEFAULT_GPU_BATCH_CAP without allocating multi-GiB VRAM blocks per
// stream up front.  HARD_BATCH_CAP is the hard ceiling above this.
static const size_t AUTO_TUNE_BATCH_CEILING = 256;
static const double DEFAULT_GPU_MEM_FRACTION = 0.60; // fraction of free VRAM to use
static const size_t DEFAULT_GPU_STREAMS = 1;       // single stream avoids context-switch overhead
static const double GROW_CHECK_SEC = 0.3;  // auto-tune check interval (seconds)
static const size_t HARD_BATCH_CAP = 1024;        // per stream safety cap
#endif

/*======================================================================
 CLI
======================================================================*/
enum class Mode { COMPRESS, DECOMPRESS, TEST };
#ifdef HAVE_NVCOMP
enum class PinMode { AUTO, ON, OFF };
#endif
struct Options {
  Mode mode = Mode::COMPRESS;
  bool list_mode = false;    // -l/--list: list .zst frame info, or `-l --tar` archive contents
  int level = 3;             // CPU Zstd level
  bool level_user_set = false;
  bool fast_flag = false;
  bool best_flag = false;
  bool ultra = false;
  bool keep = true;          // --rm clears this: delete input after success
  bool force = false;
  bool unsafe_overwrite = false; // --overwrite: truncate target in place (no atomic tmp+rename)
  bool to_stdout = false;
  // Unified verbosity level:
  //   V_SILENT(0)  = -qq: suppress everything including errors
  //   V_ERROR(1)   = -q:  errors only
  //   V_DEFAULT(2) = normal: progress bar, completion summary
  //   V_VERBOSE(3) = -v:  informational messages
  //   V_DEBUG(4)   = -vv: per-worker/batch detail
  //   V_TRACE(5)   = -vvv: per-chunk debug trace
  int verbosity = 2;         // V_DEFAULT
  bool force_progress = false; // --progress: show progress even when stderr is not a TTY
  bool cpu_only = false;
  bool gpu_only = false;
  bool hybrid = false;
  // True if the user explicitly passed --cpu-only, --gpu-only, or --hybrid.
  // When false, apply_backend_defaults() picks based on mode + PCIe gen
  // (asymmetric mode: hybrid for compress; PCIe Gen3 → cpu-only for decompress).
  bool backend_user_set = false;
  // True if the user passed any flag that only makes sense in hybrid/GPU mode
  // (--gpu-batch, --gpu-streams, --gpu-devices, --gpu-mem-frac, --pinned/-no-pinned,
  // --cpu-share, --cpu-batch, --hybrid-floor, --hybrid-floor-factor).
  // apply_backend_defaults() promotes this to an implicit --hybrid when no
  // explicit backend flag was given, so asymmetric mode doesn't silently
  // route around the user's tuning intent.
  bool gpu_hybrid_tuning_seen = false;
  int cpu_threads = 0;       // 0=auto (capped at 96), -1=all threads (-T0)
  double cpu_share = -1;     // <0 adaptive (hybrid)
  size_t chunk_mib = DEFAULT_CHUNK_MIB;
  bool chunk_user_set = false;
  size_t cpu_queue_min = 0;   // min queue depth before CPU workers activate (0=no threshold)

  // Hybrid queue-floor controls (v0.12.12+, policy rewritten v0.13.5)
  //   AUTO    : "GPU first, CPU as surplus" — factor in [1.5, 4.0] based
  //             on CPU's measured share of aggregate throughput
  //   NOMINAL : factor = 1.0, floor = active_gpu_streams * gpu_batch_size
  //   OFF     : factor = 0.0, CPUs compete freely (pre-v0.12.12 behaviour)
  enum class HybridFloorMode { AUTO, NOMINAL, OFF };
  HybridFloorMode hybrid_floor_mode = HybridFloorMode::AUTO;
  // Manual override in [0.0, 4.0].  <0 means "unset; use mode above".
  double hybrid_floor_factor = -1.0;

  // Throttle tuning (v0.12.14+).
  //   throttle_factor : slack multiplier (default THROTTLE_SLACK_FACTOR when 0)
  //   throttle_frames : explicit frame cap.  -1 = auto (use formula);
  //                     0 = disabled (no throttle, no permit accounting);
  //                     >0 = explicit cap.  v0.12.48+.
  int throttle_factor = 0;
  int throttle_frames = -1;
  // Memory usage limit in MiB (zstd-compat `-M#` / `--memlimit` / `--memory`).
  // Decompress: passed to ZSTD_d_windowLogMax — frames requiring a larger
  // window are rejected.  Compress: caps the in-flight RAM budget in
  // compute_throttle_budget.  0 = unlimited (zstd's default).
  size_t mem_limit_mib = 0;
#ifdef HAVE_NVCOMP
  size_t gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  bool gpu_batch_user_set = false;
  double gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  size_t gpu_streams = 0;            // 0=auto (1 for compress, 2 for test/verify)
  int gpu_devices = 0;            // 0=auto (all GPUs, compress and decompress)
  PinMode pin_mode = PinMode::OFF;  // empirical: pinned cudaMemcpy is slower
                                    // on our typical workloads (see CHANGELOG
                                    // v0.12.45).  Opt-in with --pinned=on.
#endif
  std::string stats_json;
  int sparse_mode = -1;           // -1=auto (file:on, stdout:off), 0=off, 1=on
  bool sliding_window = false;    // --sliding-window: single-frame compression (cpu-only)
  bool sync_output = false;       // --sync-output: fsync before closing output file
  bool direct_io = false;         // --direct: use O_DIRECT (bypasses page cache)
  bool direct_io_user_set = false; // true if user passed --direct/--no-direct (suppresses the Gen4 decompress auto-default)
  bool use_mmap = true;           // --mmap=on/off: zero-copy mmap reader for regular-file inputs
  bool mmap_user_set = false;     // true if --mmap/--no-mmap given (suppresses the pre-6.4 large-file auto-gate)
  bool cold_read = false;         // --cold: posix_fadvise(DONTNEED) on input before read (benchmarking only)
  bool direct_read = false;       // --direct-read: O_DIRECT input — bypass the page cache (no populate/evict)
  size_t read_threads = 0;        // --read-threads N: buffered pooled-reader threads (0 = auto: 3)
  size_t write_threads = 0;       // --write-threads N: -d --tar extractor writer pool (0 = auto: min(threads,16))
  bool preallocate_output = false; // --preallocate / --no-preallocate: fallocate output upfront
                                   // (default off: fallocate'd unwritten extents make O_DIRECT
                                   //  writes pay an extent-conversion per write — measured ~4-6%
                                   //  slower on ext4/NVMe, both compress and decompress)
  bool verify = false;            // --verify: background CPU decompress-verify of every compressed
                                  //  frame (independent zstd -t round-trip) while compressing.  On
                                  //  any XXH64 mismatch, discard the output and rebuild CPU-only,
                                  //  retrying until it verifies clean.  Compress-only.
  int  verify_retries = 0;        // --verify-retries=N: max rebuild attempts on verify mismatch
                                  //  (0 = unlimited — keep trying until clean or interrupted)
  bool keep_going = false;        // --keep-going: on decompress, don't abort on a frame integrity
                                  //  error — write what decoded, record the damage, continue, then
                                  //  report which file(s)/byte ranges are affected.  Implies
                                  //  --cpu-only (the authoritative checksum validator).
  std::string input;              // current file being processed
  std::vector<std::string> inputs; // all positional args (multi-file)
  std::string output;             // explicit -o; empty = auto-derive per file

  // --tar: synthesize a GNU-format tar stream from tar_sources and feed it to
  // the compressor, reading member files in parallel (replaces `tar -cf - … |
  // gzstd`).  --tar ends option parsing: every token after it is an archive
  // member path (files to archive on create; archives to extract on -d --tar).
  bool tar_mode = false;
  std::vector<std::string> tar_sources;  // compress: files/dirs to archive; decompress: input archives (post-`--tar`)
  std::vector<std::string> tar_excludes; // --exclude PATTERN (fnmatch, repeatable)
  bool tar_numeric_owner = false;        // --numeric-owner: omit uname/gname lookup
  bool tar_one_file_system = false;      // --one-file-system: don't cross mount points
  bool tar_xattrs = false;               // --xattrs: store/restore extended attributes (SCHILY.xattr.*)
  bool tar_acls = false;                 // --acls: store/restore POSIX ACLs (SCHILY.acl.*); needs both on create AND extract
  std::string tar_dest = ".";            // -C/--directory: extraction root (decompress + --tar)
};

/*======================================================================
 Verbosity levels & exit codes
======================================================================*/
static constexpr int V_SILENT  = 0;  // -qq: suppress everything including errors
static constexpr int V_ERROR   = 1;  // -q:  errors only
static constexpr int V_DEFAULT = 2;  // normal: progress bar, completion summary
static constexpr int V_NORMAL  = 0;  // always shown (no -v needed)
static constexpr int V_VERBOSE = 3;  // -v:  informational messages
static constexpr int V_DEBUG   = 4;  // -vv: per-worker/batch detail
static constexpr int V_TRACE   = 5;  // -vvv: per-chunk debug trace

static constexpr int EXIT_OK       = 0;  // success
static constexpr int EXIT_ERROR    = 1;  // general runtime error (catch-all)
static constexpr int EXIT_USAGE    = 2;  // bad command-line usage
static constexpr int EXIT_IO       = 3;  // I/O error (disk full, read failure, permissions)
static constexpr int EXIT_DATA     = 4;  // data/compression error (corrupt input, integrity failure)
static constexpr int EXIT_GPU_FAIL = 5;  // reserved: all GPUs failed.  Since v0.13.54
                                         // unused — gpu_only_cpu_fallback finishes the
                                         // job on CPU (warning + exit 0) instead.
// --keep-going (decompress) completion codes.  A recovery run that produced
// output still exits non-zero so a script never mistakes it for a clean restore.
static constexpr int EXIT_DECOMP_UNVERIFIED = 6;  // finished; >=1 frame had a checksum mismatch
                                                  // (content present but UNVERIFIED), no data lost
static constexpr int EXIT_DECOMP_INCOMPLETE = 7;  // finished; >=1 frame could not fully decode
                                                  // (data missing/partial — outranks UNVERIFIED)

// Global verbosity for die() which doesn't take Options
static int g_verbosity = V_DEFAULT;

// True when stderr is a TTY and ANSI color codes should be emitted.
// Set once after parse_args(); never changes afterwards.
static bool g_color_stderr = false;

// Global flag: true when the progress bar is actively drawing on stderr.
// vlog() checks this to clear the progress line before printing so verbose
// messages don't overlap the progress bar.  Set by progress_loop, cleared
// when progress_loop exits.
static std::atomic<bool> g_progress_active{false};

// Emit a message to stderr if the current verbosity is >= min_level.
// Caller must include \n or \r in msg as appropriate.
// If the progress bar is active, clears the progress line first so
// the message prints cleanly on its own line, then the next progress
// tick redraws the bar.
// When stderr is a TTY, applies level-based ANSI color:
//   V_VERBOSE — default terminal color (most important, reads like plain text)
//   V_DEBUG   — dim (detail, visually recedes)
//   V_TRACE   — dark grey (fine-grained noise, lowest visual weight)
// Any leading [TAG] token is bolded as a visual anchor regardless of level.
// Numeric tokens (integers and decimals) are colorized from a cycling palette
// so they pop out from surrounding text.
// No codes are emitted when stderr is not a TTY (safe to redirect to logs).

// Determine the field color for the number at [num_start, num_end) in `s`.
// Looks back for a "keyword=" (or "keyword=[") pattern and forward for
// " keyword" suffixes (e.g. "159 batches").  Returns nullptr to fall through
// to the cycling palette.
//
// Field → color mapping:
//   in=       bright cyan    (input sizes)
//   out=      green          (output sizes)
//   seq=      bright blue    (sequence numbers, both in seq=[N..N])
//   batch=, tasks=, frames=, chunks=, waits=, count=
//             bright green   (counts)
//   N batches/frames/chunks/waits/tasks
//             bright green   (same counts, space-delimited)
//   h2d=, d2h=  gray        (timing labels, not meaningful data values)
static const char * vlog_field_color(const std::string & s,
                                     size_t num_start, size_t num_end)
{
  // --- look-back: find keyword= (or keyword=[) before this number ---
  size_t k = num_start;
  // Handle second number in seq=[N..N]: preceded by ".."
  if (k >= 2 && s[k-1] == '.' && s[k-2] == '.') {
    // skip back over first number and '[' to reach '='
    k -= 2;
    while (k > 0 && std::isdigit((unsigned char)s[k-1])) k--;
    if (k > 0 && s[k-1] == '[') k--;
  }
  // Skip optional '[' (seq=[N)
  if (k > 0 && s[k-1] == '[') k--;
  // Expect '='
  if (k > 0 && s[k-1] == '=') {
    k--;  // skip '='
    size_t kend = k;
    while (k > 0 && (std::isalnum((unsigned char)s[k-1]) || s[k-1] == '_')) k--;
    std::string kw = s.substr(k, kend - k);

    if (kw == "in")                              return "\033[1;96m";  // bright cyan
    if (kw == "out")                             return "\033[32m";    // green
    if (kw == "seq")                             return "\033[1;94m";  // bright blue
    if (kw == "h2d" || kw == "d2h")             return "";            // no color — plain text
    if (kw == "batch"  || kw == "tasks"  ||
        kw == "frames" || kw == "chunks" ||
        kw == "waits"  || kw == "count"  ||
        kw == "batches")                         return "\033[1;92m";  // bright green
  }

  // --- look-ahead: "N keyword" suffix (e.g. "159 batches,") ---
  if (num_end < s.size() && s[num_end] == ' ') {
    static const struct { const char * kw; } post[] = {
      {" batches"}, {" frames"}, {" chunks"}, {" waits"}, {" tasks"}, {" batch"},
    };
    for (auto & p : post) {
      size_t kl = std::strlen(p.kw);
      if (num_end + kl <= s.size() && s.compare(num_end, kl, p.kw) == 0)
        return "\033[1;92m";  // bright green
    }
  }

  return nullptr;  // use cycling palette
}

// Skip over an ANSI escape sequence starting at s[i] (must be '\033').
// Returns the index of the first character after the sequence.
static size_t vlog_skip_ansi(const std::string & s, size_t i)
{
  // Sequences we emit: \033[ <params> m
  if (i + 1 < s.size() && s[i + 1] == '[') {
    i += 2;
    while (i < s.size() && s[i] != 'm') ++i;
    if (i < s.size()) ++i;  // skip 'm'
  }
  return i;
}

// Colorize a [TAG] token:
//   '[' '/' ']'  — bold bright white
//   letters/text — word_color (bright yellow for GPU, bright magenta for CPU)
//   numbers      — cycling palette
// Pre-existing ANSI codes in `tag` are passed through unchanged.
static std::string vlog_colorize_tag(const std::string & tag, const char * word_color)
{
  static const char * palette[] = {
    "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[35m",
  };
  static constexpr size_t N = sizeof(palette) / sizeof(palette[0]);
  const char * PUNCT = "\033[1;97m";  // bold bright white

  std::string out;
  size_t i = 0, ci = 0;
  out += word_color;
  while (i < tag.size()) {
    if (tag[i] == '\033') {                    // pass through existing ANSI codes
      size_t j = vlog_skip_ansi(tag, i);
      out.append(tag, i, j - i);
      i = j;
    } else if (tag[i] == '[' || tag[i] == ']' || tag[i] == '/') {
      out += PUNCT;
      out += tag[i++];
      out += "\033[0m";
      out += word_color;
    } else if (std::isdigit((unsigned char)tag[i])
               || (tag[i] == '.' && i + 1 < tag.size()
                   && std::isdigit((unsigned char)tag[i + 1]))) {
      size_t start = i;
      while (i < tag.size() && (std::isdigit((unsigned char)tag[i]) || tag[i] == '.'))
        ++i;
      out += palette[ci++ % N];
      out.append(tag, start, i - start);
      out += "\033[0m";
      out += word_color;
    } else {
      out += tag[i++];
    }
  }
  return out;
}

// Walk `s` and colorize numeric tokens using keyword context:
//   in=N      bright cyan    out=N     green
//   seq=[N..N] bright blue   h2d=/d2h= gray
//   batch=/tasks=/frames=/etc. bright green
//   N batches/frames/chunks/waits  bright green
//   everything else  cycling palette (blue→cyan→green→yellow→purple)
// Pre-existing ANSI codes are passed through unchanged.
static std::string vlog_colorize_numbers(const std::string & s, const char * restore)
{
  static const char * palette[] = {
    "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[35m",
  };
  static constexpr size_t N = sizeof(palette) / sizeof(palette[0]);

  std::string out;
  out.reserve(s.size() * 2);
  size_t i = 0, ci = 0;
  while (i < s.size()) {
    if (s[i] == '\033') {
      size_t j = vlog_skip_ansi(s, i);
      out.append(s, i, j - i);
      i = j;
      continue;
    }
    // Only start a number token if not preceded by an alphanumeric — this prevents
    // digits embedded in identifiers like h2d, d2h, GPU0 from being colorized.
    bool num_start = (i == 0 || !std::isalnum((unsigned char)s[i - 1]))
                  && (std::isdigit((unsigned char)s[i])
                      || (s[i] == '.' && i + 1 < s.size()
                          && std::isdigit((unsigned char)s[i + 1])));
    if (num_start) {
      size_t start = i;
      while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.'))
        ++i;
      const char * fc = vlog_field_color(s, start, i);
      if (fc && fc[0] == '\0') {
        // Explicitly no color (e.g. h2d=/d2h=) — emit plain, no ANSI.
        out.append(s, start, i - start);
      } else {
        const char * col = fc ? fc : palette[ci++ % N];
        out += col;
        out.append(s, start, i - start);
        out += "\033[0m";
        out += restore;
      }
    } else {
      out += s[i++];
    }
  }
  return out;
}

static void vlog(int min_level, const Options & opt, const std::string & msg)
{
  if (opt.verbosity < min_level) return;
  if (g_progress_active.load(std::memory_order_relaxed))
    std::fprintf(stderr, "\r\033[K");  // clear progress line

  if (!g_color_stderr) {
    std::cerr << msg;
    return;
  }

  // All levels use default terminal color — numbers are colorized individually.
  const char * lvl_color = "";

  // Split off a leading [TAG] token to bold it as a visual scan anchor.
  std::string tag_part, body_part;
  if (!msg.empty() && msg[0] == '[') {
    auto close = msg.find(']');
    if (close != std::string::npos && close + 1 < msg.size()) {
      tag_part  = msg.substr(0, close + 1);
      body_part = msg.substr(close + 1);
    }
  }
  if (tag_part.empty()) body_part = msg;

  // Pick tag color based on prefix: GPU → bold bright yellow, CPU → bold bright magenta.
  const char * tag_color = "\033[1m";       // default: bold white
  if (tag_part.size() > 4) {               // at least "[GPU" or "[CPU"
    if (tag_part.compare(1, 3, "GPU") == 0) tag_color = "\033[1;93m";  // bold bright yellow
    else if (tag_part.compare(1, 3, "CPU") == 0) tag_color = "\033[1;95m";  // bold bright magenta
  }

  // Colorize tag: punctuation white, letters in tag_color, numbers in palette.
  std::string tag_col  = tag_part.empty() ? "" : vlog_colorize_tag(tag_part, tag_color);
  std::string body_col = vlog_colorize_numbers(body_part, lvl_color);

  // Assemble: tag | reset | body (numbers already colored).
  std::string out;
  out += lvl_color;
  if (!tag_col.empty()) {
    out += tag_col;
    out += "\033[0m";
    out += lvl_color;
  }
  out += body_col;

  // Strip trailing newline so reset lands before it (avoids stray color on blank lines).
  bool has_nl = !out.empty() && out.back() == '\n';
  if (has_nl) out.pop_back();
  std::cerr << out << "\033[0m";
  if (has_nl) std::cerr << '\n';
}

static void die(const std::string & msg, int code = EXIT_ERROR)
{
  if (g_verbosity >= V_ERROR) {
    std::cerr << "gzstd: ERROR: " << msg << "\n";
    if (g_tmp_active && g_tmp_path[0] != '\0')
      std::cerr << "gzstd: removing incomplete output: " << g_tmp_path << "\n";
  }
  std::exit(code);
}
static void die_usage(const std::string & msg)
{ die(msg, EXIT_USAGE); }
static void die_io(const std::string & msg)
{ die(msg, EXIT_IO); }
static void die_data(const std::string & msg)
{ die(msg, EXIT_DATA); }

// Short help — shown by -h and -?.  Groups flags by purpose with one-line
// descriptions.  Point users at --help for details and examples.
static void print_help()
{
  std::cout <<
"gzstd " << GZSTD_VERSION << " — hybrid CPU+GPU Zstd compression\n"
"\n"
"Usage: gzstd [options] [file ...]\n"
"\n"
"Operation:\n"
"  -d                  decompress\n"
"  -t                  test (verify integrity, no output)\n"
"  -l                  list .zst frame info (Frames/Sizes/Ratio/Check); with\n"
"                      --tar, list the archive contents (tar -tvf style)\n"
"  -k                  keep input after success (default)\n"
"  --rm                remove input after success\n"
"\n"
"Output:\n"
"  -c                  write to stdout\n"
"  -o, --output FILE   explicit output path\n"
"  -f                  overwrite existing file (atomic: .tmp + rename)\n"
"  --overwrite         overwrite in place (faster on ext4; non-atomic)\n"
"\n"
"Archive (tar):\n"
"  --tar SRC...        build one .tar.zst from the files/dirs listed after\n"
"                      --tar, reading members in parallel.  The tar options\n"
"                      below may also follow --tar.\n"
"  -d --tar ARC...     extract .tar.zst archive(s) (decompress + untar)\n"
"  -C, --directory DIR extraction root for -d --tar (default: current dir)\n"
"  --write-threads N   parallel file writers for -d --tar (default: min(-T,16))\n"
"  --exclude PATTERN   skip members matching PATTERN (glob; repeatable)\n"
"  --numeric-owner     store uid/gid only (no user/group name lookup)\n"
"  --one-file-system   do not descend into other mounted filesystems\n"
"  --xattrs            store/restore extended attributes (give on both create and extract)\n"
"  --acls              store/restore POSIX ACLs (give on both create and extract)\n"
"  --sparse            store sparse files compactly on create (GNU format; opt-in)\n"
"\n"
"Compression level:\n"
"  -1 .. -19           zstd level (default: 3)\n"
"  -20 .. -22          ultra levels (require --ultra)\n"
"  --fast / --best     aliases for -1 / -19\n"
"  --ultra             enable ultra levels (large window, more memory)\n"
"\n"
"Backend (auto: hybrid for compress; cpu-only for Gen3 decompress):\n"
"  --cpu-only          CPU multithreaded, no GPU\n"
"  --gpu-only          GPU only, no CPU workers\n"
"  --hybrid            CPU + GPU\n"
"  --sliding-window    single-frame max-ratio mode (implies --cpu-only)\n"
"\n"
"Tuning:\n"
"  -T, --threads N     CPU worker threads (0 = all cores)\n"
"  --chunk-size N      host I/O chunk size in MiB (default: 16)\n"
"  --throttle-factor N slack multiplier for in-flight cap (default: 4)\n"
"  --throttle-frames N explicit cap (-1=auto, 0=disabled)\n"
"  --no-throttle       alias for --throttle-frames=0 (benchmarking)\n";
#ifdef HAVE_NVCOMP
  std::cout <<
"  --gpu-batch N       GPU subchunks per stream (default: 8 compress,\n"
"                      16 decompress; decompress auto-scales by size)\n"
"  --gpu-streams N     CUDA streams per device (default: 1)\n"
"  --gpu-devices N     number of GPUs (0 = auto)\n"
"  --gpu-mem-frac X    fraction of free VRAM per device (default: 0.60)\n";
#endif
  std::cout <<
"\n"
"I/O:\n"
"  --[no-]sparse       sparse file writes (default: on for files)\n"
"  --[no-]mmap         zero-copy mmap reader for regular-file inputs\n"
"                      (on; auto-fread for inputs >4 GiB on kernel <6.4)\n"
"  --[no-]preallocate  fallocate output to expected size up front\n"
"                      (default: off; slower for O_DIRECT — see --preallocate)\n"
"  --direct            O_DIRECT output (bypass page cache; auto-on for\n"
"                      Gen4+ compress & decompress, --no-direct to force buffered)\n"
"  --sync-output       fsync output before exit\n"
"  --cold              drop input from page cache before reading\n"
"                      (BENCHMARKING ONLY: forces a cold-cache read)\n"
"  --direct-read       O_DIRECT input, single stream: bypass the page cache\n"
"                      for one-pass speedups and honest cold benchmarks\n"
"  --read-threads N    parallel readers for the BUFFERED input path (default:\n"
"                      auto 3..12 by -T; 1 = single; n/a with --direct-read/stdin)\n"
#ifdef HAVE_NVCOMP
"  --pinned MODE       pinned host buffers: auto|on|off (default: off)\n"
#endif
"\n"
"Integrity:\n"
"  --verify            decompress-verify every frame while compressing; on any\n"
"                      mismatch, rebuild CPU-only and retry until it verifies\n"
"  --verify-retries N  cap verify rebuild attempts (0 = unlimited; default)\n"
"  --keep-going        on -d, don't stop at a corrupt frame: recover what's\n"
"                      readable, then report which file(s) are damaged\n"
"\n"
"Logging:\n"
"  -v / -vv / -vvv     verbose / debug / trace\n"
"  -q / -qq            errors only / silent\n"
"  --progress          force progress meter in pipes\n"
"  --stats-json FILE   write run stats to FILE\n"
"\n"
"Misc:\n"
"  -h, -?              this short help\n"
"  --help              full help with details and examples\n"
"  -V, --version       version info\n"
"\n"
"Reads stdin when no file is given; stdin implies -c (stdout).\n"
"See `gzstd --help` for hybrid-scheduler details and full flag docs.\n";
}

// Long help — shown by --help.  Detailed descriptions, flag interactions,
// tuning notes, and runnable examples.
static void print_help_long()
{
  std::cout <<
"gzstd " << GZSTD_VERSION << " — hybrid CPU+GPU Zstd compression\n"
"\n"
"Usage: gzstd [options] [file ...]\n"
"\n"
"gzstd is a drop-in-compatible replacement for the zstd CLI that can run\n"
"compression and decompression simultaneously across CPU cores and one or\n"
"more CUDA-capable GPUs.  It produces standard .zst files that can be read\n"
"by any zstd implementation.\n"
"\n"
"With no file arguments (or file `-`), gzstd reads from stdin and writes\n"
"to stdout.  Pipes are fully supported:\n"
"    tar cf - dir  | gzstd > archive.tar.zst\n"
"    gzstd -d < archive.tar.zst | tar xf -\n"
"\n"
"============================================================\n"
" OPERATION\n"
"============================================================\n"
"  -d\n"
"     Decompress.  Input must be a valid zstd stream.\n"
"\n"
"  -t\n"
"     Test mode: decompress and verify integrity but write nothing.\n"
"     Exit code 0 on success, 4 on data error.  Defaults to 2 CUDA\n"
"     streams per device (vs. 1 for normal decompress) because there\n"
"     is no downstream writer to bottleneck on.\n"
"     With --tar (`-t --tar ARCHIVE`), ALSO validates the tar structure\n"
"     inside the zstd stream — every header checksum and member size, and\n"
"     that the archive is complete — catching a truncated or corrupt tar\n"
"     that plain -t (which only checks the zstd stream) would pass.\n"
"\n"
"  -l, --list\n"
"     List mode (no output written).  For a plain .zst file, prints the\n"
"     per-file frame summary — frame count, skippable frames, compressed\n"
"     and uncompressed sizes, ratio, and whether a content checksum is\n"
"     present (Frames/Skips/Compressed/Uncompressed/Ratio/Check), matching\n"
"     `zstd -l`.  Reads only frame headers, not the payload, so it is fast\n"
"     even on huge archives.\n"
"     With --tar (`-l --tar ARCHIVE`), lists the archive contents in\n"
"     `tar -tvf` style (mode, owner, size, mtime, name) by decompressing\n"
"     and parsing the tar headers in memory.\n"
"\n"
"  -k\n"
"     Keep input after success (this is the default).\n"
"\n"
"  --rm\n"
"     Remove input after a successful operation.  Implies -k=off.\n"
"\n"
"============================================================\n"
" OUTPUT\n"
"============================================================\n"
"  -c\n"
"     Write to stdout.  Implied when reading from stdin.\n"
"\n"
"  -o, --output FILE\n"
"     Explicit output path.  Overrides the default (input + .zst on\n"
"     compress, input with .zst stripped on decompress).\n"
"\n"
"  -f\n"
"     Force overwrite.  Writes to `<output>.gzstd.tmp` then renames\n"
"     atomically over the target.  On ext4 with large outputs the\n"
"     rename can stall while the journal flushes dirty pages; use\n"
"     --overwrite if that cost is unacceptable.\n"
"\n"
"  --overwrite\n"
"     Force overwrite in place (truncate target, no tmp + rename).\n"
"     Faster than -f on ext4 with large outputs, but NOT atomic: if\n"
"     gzstd is killed mid-run the target is left corrupt/partial.\n"
"     Implies -f.\n"
"\n"
"  --[no-]sparse\n"
"     Sparse output: skip writing all-zero blocks, leaving filesystem\n"
"     holes.  Default: on for regular-file output, off for stdout.\n"
"     On `-d --tar` extraction, on by default: a zero-filled region is\n"
"     restored as a hole (a 100 GiB-sparse/1 GiB-real file restores to\n"
"     ~1 GiB on disk, not 100 GiB).\n"
"     --no-sparse only disables this scan-for-zeros: it makes normal\n"
"     file entries fully allocated.  It does NOT affect entries the\n"
"     archive explicitly stored as sparse (created with `--tar --sparse`\n"
"     or GNU `tar --sparse`) — those are always restored with their\n"
"     recorded holes, like GNU tar.  --no-sparse means \"don't go looking\n"
"     for holes,\" not \"materialize holes the archive declared.\"\n"
"     On `--tar --sparse` (create), a sparse source file is stored\n"
"     compactly in GNU sparse format (opt-in; see --sparse under --tar).\n"
"\n"
"  --sync-output\n"
"     fsync the output file before exit.  Default: off.  Without\n"
"     this the OS flushes in the background; the data is durable on\n"
"     a clean shutdown but not guaranteed across power loss.\n"
"\n"
"  --verify\n"
"     While compressing, independently decompress-verify every frame on\n"
"     the CPU (the same round-trip as `gzstd -t`) using the XXH64\n"
"     (64-bit xxHash) content checksum each frame already carries.  zstd\n"
"     — and gzstd by default — only WRITE that checksum; it is checked at\n"
"     read time, never at creation.  That is fine for a CPU compressor\n"
"     (deterministic), but the GPU is an untrusted producer: a fault can\n"
"     emit wrong compressed bytes that surface only at restore, possibly\n"
"     after the source is gone.  --verify closes that gap.  It runs in a\n"
"     background thread pool that grows on demand and competes with the\n"
"     compress threads for CPU, so it costs throughput only when it has\n"
"     to; on a GPU run the otherwise-idle CPU absorbs it nearly free.\n"
"     On ANY mismatch the whole output is discarded and rebuilt CPU-only\n"
"     from the original input, retrying until it verifies clean.  Needs\n"
"     regular files for input and output (a pipe cannot be rewound to\n"
"     rebuild); over a pipe a mismatch is a fatal data error instead.\n"
"     Off by default.  Compress-only (decompression already validates).\n"
"\n"
"  --verify-retries N\n"
"     Cap the number of CPU-only rebuild attempts --verify makes on a\n"
"     persistent mismatch.  Default 0 = unlimited (keep trying until the\n"
"     archive verifies clean or you interrupt it) — the right default for\n"
"     an unattended backup.  Set a bound (e.g. for a cron job) so a truly\n"
"     faulty machine eventually gives up with a non-zero exit instead of\n"
"     looping forever.  Implies --verify.\n"
"\n"
"  --keep-going\n"
"     Recovery mode for DECOMPRESSING a damaged archive.  Normally a frame\n"
"     whose XXH64 (64-bit xxHash) content checksum does not match aborts\n"
"     the whole operation.  With --keep-going, gzstd instead writes what it\n"
"     could decode, records the damage, and continues — then reports which\n"
"     file(s) (with --tar) or output byte range(s) are affected.  It does\n"
"     NOT repair data: a checksum-mismatch frame is written in full but is\n"
"     UNVERIFIED (one or more bytes somewhere in it may be wrong); a frame\n"
"     that cannot be decoded at all yields only what decoded before the\n"
"     error.  Implies --cpu-only.  Existing files are overwritten as usual;\n"
"     to recover alongside the originals, extract into a fresh directory\n"
"     with -C DIR.  Exit codes: 6 = finished but some file(s) are\n"
"     unverified; 7 = finished but some data could not be decoded\n"
"     (incomplete).  Off by default.\n"
"\n"
"  --direct / --no-direct\n"
"     Use O_DIRECT (bypass page cache) vs. buffered I/O.  Default is\n"
"     buffered, EXCEPT on PCIe (PCI Express) Gen4+ hardware, where O_DIRECT is\n"
"     auto-enabled for BOTH compress and decompress: on fast-fabric\n"
"     boxes frame production outruns buffered writeback, so bypassing\n"
"     the page cache is a large win that scales with output volume\n"
"     (it regresses on smaller Gen<4 boxes, which stay buffered).\n"
"     Test mode writes nothing, so it is unaffected.  --no-direct\n"
"     forces buffered I/O (e.g. to benchmark the buffered baseline on\n"
"     Gen4).  O_DIRECT can expose the app to NVMe (NVM Express)\n"
"     garbage-collection (GC) stalls and journal commits.\n"
"\n"
"  --mmap / --no-mmap    (default: on, with a kernel auto-gate)\n"
"     Use a zero-copy memory-mapped reader for regular-file inputs.\n"
"     mmap lets workers read directly from the kernel's page cache —\n"
"     no fread + memcpy through a userspace buffer.  Pipes and stdin\n"
"     always fall back to fread regardless of this flag.\n"
"     AUTO-GATE: Linux kernels before 6.4 lack per-VMA (virtual memory\n"
"     area) locks, so a large mmap faulted by many worker threads\n"
"     serialises on the one\n"
"     mmap_lock — a fault-storm that scales with file size and cores.\n"
"     For inputs over 4 GiB on a <6.4 kernel, gzstd therefore defaults\n"
"     to fread; on 6.4+ kernels mmap stays on (zero-copy wins there).\n"
"     --mmap / --no-mmap force the choice and override the auto-gate.\n"
"\n"
"  --direct-read    (default: off)\n"
"     Read the input with O_DIRECT: transfer straight from disk into an\n"
"     aligned buffer, BYPASSING the page cache entirely (it neither\n"
"     reads from nor populates it).  Two uses: (1) a one-pass speedup —\n"
"     compressing a file you never re-read gains nothing from caching\n"
"     it, so skipping the populate + writeback overhead can be faster;\n"
"     (2) honest cold benchmarking with zero system impact — every run\n"
"     reads cold from disk, with no cache to drop, so no kcompactd\n"
"     compaction stall and no eviction of other users' cached data\n"
"     (unlike --cold, which drops the cache via fadvise).  Always a\n"
"     SINGLE stream: concurrent O_DIRECT reads contend on NVMe, so it\n"
"     cannot use --read-threads and cannot go through mmap.  On a\n"
"     big-RAM box the buffered path is usually faster (reads served from\n"
"     cache), so this is mainly a benchmarking / one-pass tool.\n"
"     Independent of --direct, which is O_DIRECT for OUTPUT.\n"
"\n"
"  --read-threads N    (default: 0 = auto)\n"
"     Number of parallel reader threads for the BUFFERED input path.  A\n"
"     single reader's per-frame copy and read syscalls can cap throughput\n"
"     on a fast box; spreading them over N threads lifts that ceiling.\n"
"     Applies to compress (the pooled reader — used when mmap is declined,\n"
"     e.g. a >4 GiB input on a <6.4 kernel, or --no-mmap) and to\n"
"     decompress (the parallel-prefetch frame reader), for seekable\n"
"     regular-file inputs.  0 = auto: clamp(threads/8, 3, 12).  1 = force\n"
"     a single reader.  Ignored for stdin/pipes and under --direct-read.\n"
"     Unrelated to --direct-read beyond being mutually exclusive with it.\n"
"\n"
"  --cold    (default: off)\n"
"     Drop the input from the page cache (posix_fadvise DONTNEED) before\n"
"     reading it, forcing a cold-cache read.  BENCHMARKING ONLY: unlike\n"
"     --direct-read (which bypasses the cache entirely), --cold evicts\n"
"     any already-cached pages, so it disturbs other users of that data.\n"
"\n"
"  --preallocate / --no-preallocate    (default: off)\n"
"     Preallocate the output file with `fallocate` to its expected\n"
"     final size before writes begin.  Off by default: with O_DIRECT\n"
"     output (the Gen4+ default), fallocate creates unwritten extents\n"
"     and each O_DIRECT write then pays an unwritten-to-written extent\n"
"     conversion — measured ~4-6%% slower on ext4/NVMe for both compress\n"
"     and decompress.  --preallocate forces it on for filesystems or\n"
"     workloads where reserving space up front helps (and for early\n"
"     out-of-space detection).  Only ever applies to the O_DIRECT paths\n"
"     where the expected size is known (input size / sum of frame sizes).\n"
"\n"
"============================================================\n"
" ARCHIVE (tar)\n"
"============================================================\n"
"  --tar SRC...\n"
"     Build a single .tar.zst archive from the listed files and\n"
"     directories, instead of piping `tar` into gzstd.  gzstd walks\n"
"     the tree itself and reads member files IN PARALLEL, feeding\n"
"     the same multithreaded CPU/GPU pipeline — so the archiving is\n"
"     no longer bottlenecked on tar's single-threaded reader.\n"
"     The result is a standard GNU-format tar\n"
"     stream (`tar --format=gnu`, the default on Ubuntu/Debian)\n"
"     wrapped in zstd, extractable with any tar+zstd:\n"
"         tar --zstd -xf out.tar.zst        # or:\n"
"         gzstd -d out.tar.zst | tar -xf -\n"
"     List the files and directories to archive AFTER --tar.  The\n"
"     tar-specific options (--exclude, --numeric-owner,\n"
"     --one-file-system) may appear after --tar too, mixed with the\n"
"     sources, e.g.:\n"
"         gzstd -o backup.tar.zst --tar --exclude '*/.cache/*' ~/docs\n"
"     A source whose name begins with '-' must be written ./-name (or\n"
"     placed after a literal `--`).  Output goes to -o FILE, or to\n"
"     stdout when piped.\n"
"     Stores regular files, directories, symlinks, hardlinks, FIFOs\n"
"     (named pipes), and device nodes, with permissions, owner, and\n"
"     modification time.  Long paths (>100 bytes) and large files\n"
"     (>8 GiB) use the standard GNU long-name and base-256 fields.\n"
"     Sockets and unknown file types are ignored (shown only at -v, like\n"
"     tar's --warning=no-file-ignored), and do not affect the exit code.\n"
"     A member that genuinely cannot be read (permissions, or it changes\n"
"     mid-read) is reported and the run exits non-zero, but the archive\n"
"     stays valid.\n"
"     Reads are spread over --read-threads workers (see CPU TUNING).\n"
"     Note: extraction is not yet built in; use tar to extract.\n"
"\n"
"  --exclude PATTERN\n"
"     Skip paths matching the shell-glob PATTERN (repeatable).  Matched\n"
"     against the filesystem path being walked, like GNU tar: a leading\n"
"     '/' anchors to the source root (--exclude=/proc drops only the\n"
"     top-level /proc, not var/proc), while a bare name is unanchored and\n"
"     matches that component anywhere (--exclude .cache drops every\n"
"     .cache).  '*' spans '/', and a directory match drops its whole\n"
"     subtree.  Examples: --exclude=/proc --exclude='/home/*/.cache'.\n"
"\n"
"  --numeric-owner\n"
"     Record numeric uid/gid only; skip user/group name resolution.\n"
"\n"
"  --one-file-system\n"
"     Stay on each source's filesystem: a directory on a different mount\n"
"     is recorded as an empty stub (so the mount point exists on restore)\n"
"     but its contents are not archived.  This is the safe way to back up\n"
"     `/` — it skips /proc, /sys, /dev, /run automatically.  List real\n"
"     partitions as separate sources to include them:\n"
"         sudo gzstd -o root.tar.zst --tar --one-file-system / /home /boot\n"
"\n"
"  --xattrs\n"
"     Store extended attributes (xattrs) — the user.*, security.*, etc.\n"
"     key/value pairs a file carries — as PAX (Portable Archive eXchange)\n"
"     extended-header `SCHILY.xattr.*` records, compatible with GNU tar.\n"
"     Off by default (gathering xattrs costs an extra syscall per member);\n"
"     must be given on BOTH create and extract, or it is ignored — exactly\n"
"     as GNU tar behaves.  Gathering runs in parallel across the worker pool.\n"
"\n"
"  --acls\n"
"     Store POSIX access control lists (ACLs) as PAX `SCHILY.acl.access`\n"
"     and `SCHILY.acl.default` text records, compatible with GNU tar.  Off\n"
"     by default; must be given on BOTH create and extract or it is ignored.\n"
"     Restoring an ACL is applied after permissions, so the ACL wins.\n"
"\n"
"  --sparse (on create)\n"
"     Detect each file's holes via SEEK_DATA/SEEK_HOLE (filesystem metadata,\n"
"     reads no hole bytes) and store it compactly in GNU OLDGNU sparse\n"
"     format — only the data segments plus a hole map.  Output is\n"
"     byte-identical to `tar --format=gnu --sparse`, so GNU tar extracts it\n"
"     and restores the holes too.  Opt-in, matching GNU tar: without\n"
"     --sparse a sparse file is stored with full content (Zstd still\n"
"     compresses the zeros, but the archive is not marked sparse).\n"
"     Extraction restores holes by default regardless; see --[no-]sparse.\n"
"\n"
"  -d --tar ARCHIVE...\n"
"     Extract one or more .tar.zst archives (decompress AND untar in one\n"
"     pass).  Decompression runs on the full parallel CPU/GPU pipeline and\n"
"     overlaps with file writes (small files written by a worker pool), so\n"
"     it is faster than `gzstd -d | tar -x` on many-small-file archives.\n"
"     Restores regular files, directories, symlinks, hardlinks, FIFOs and\n"
"     device nodes with permissions and modification time; owners are\n"
"     restored only when running as root.  Reads any standard tar inside\n"
"     zstd (GNU, ustar, or pax), not just gzstd's own.\n"
"     Extraction is SECURE BY CONSTRUCTION: members are written through the\n"
"     destination directory with O_NOFOLLOW on every path component, so a\n"
"     member symlink cannot be traversed to escape the destination, and\n"
"     `..` components are refused — the classic tar path-traversal and\n"
"     symlink-escape attacks are blocked.  A leading '/' is stripped\n"
"     (members extract relative to -C DIR).  Existing files are overwritten.\n"
"\n"
"  -C, --directory DIR\n"
"     Extraction root for -d --tar (default: current directory).  Must\n"
"     already exist.  Example:\n"
"         gzstd -d --tar -C /restore backup.tar.zst\n"
"\n"
"  --write-threads N\n"
"     Number of parallel file-writer threads for -d --tar extraction\n"
"     (default: min(worker threads, 16)).  Small files are written by this\n"
"     pool while decompression runs; raising it past ~16 usually plateaus\n"
"     on filesystem metadata/journal contention, but the optimum is\n"
"     hardware-dependent — tune it on the target box.\n"
"\n"
"============================================================\n"
" COMPRESSION LEVEL\n"
"============================================================\n"
"  -1 .. -19\n"
"     Standard zstd compression level.  Higher = better ratio,\n"
"     slower.  Default: 3.\n"
"\n"
"  -20 .. -22\n"
"     Ultra-high-ratio levels.  Require --ultra (guards against\n"
"     accidental 32-128 MiB window allocations per thread).\n"
"\n"
"  --fast\n"
"     Alias for -1.\n"
"\n"
"  --best\n"
"     Alias for -19.\n"
"\n"
"  --ultra\n"
"     Enable ultra levels.  Sets a large window (32-128 MiB depending\n"
"     on level) which can blow RAM budgets with many threads.\n"
"\n"
"============================================================\n"
" BACKEND SELECTION\n"
"============================================================\n"
"gzstd picks a backend automatically based on hardware and operation:\n"
"  * no GPU:                         --cpu-only\n"
"  * compress, GPU present:          --hybrid (CPU + GPU)\n"
"  * decompress, PCIe Gen<4 GPU:     --cpu-only  (asymmetric mode)\n"
"  * decompress, PCIe Gen4+ GPU:     --hybrid\n"
"\n"
"Asymmetric mode (v0.13.0+): on PCIe Gen3 hardware the D2H\n"
"(device-to-host) transfer cost makes hybrid decompress slower than\n"
"CPU MT (multithreaded) for every measured data type.  Default\n"
"decompress goes to CPU-only on Gen<4 to win that benchmark; compress\n"
"still uses hybrid because GPU compress wins across all PCIe\n"
"generations tested.  Override explicitly with one of these:\n"
"\n"
"  --cpu-only\n"
"     Force CPU-only.  Useful for baseline measurements or when the\n"
"     GPU is busy with other work.\n"
"\n"
"  --gpu-only\n"
"     Force GPU-only.  Fails if no GPU is available or all GPUs fail\n"
"     to initialize.  If a GPU faults mid-run, its output cannot be\n"
"     trusted, so the whole archive is discarded and rebuilt CPU-only\n"
"     from the original input.\n"
"\n"
"  --hybrid\n"
"     Enable CPU + GPU scheduling (default when a GPU is present).\n"
"     CPU workers pop single frames; GPU workers pop greedy batches.\n"
"     An EMA (exponential moving average)-based scheduler tracks\n"
"     observed throughput and adapts the CPU/GPU split over time.\n"
"\n"
"  --sliding-window\n"
"     Compress the whole file as a single frame with zstd's built-in\n"
"     multi-threaded mode (`ZSTD_c_nbWorkers`).  Maximum ratio on\n"
"     repetitive data (matches `zstd` exactly) because the\n"
"     sliding window carries context across the full file.\n"
"     Implies --cpu-only.  Decompression is single-threaded because\n"
"     one frame = one unit of work for any decompressor.\n"
"\n"
"============================================================\n"
" CPU TUNING\n"
"============================================================\n"
"  -T, --threads N\n"
"     CPU worker threads.  0 = all cores (auto-capped at 96).\n"
"\n"
"  --chunk-size N\n"
"     Host I/O chunk size in MiB (default: 16).  Each independent\n"
"     frame covers this many bytes of input.  Larger chunks = fewer\n"
"     frames, less bookkeeping, but less parallelism and coarser\n"
"     progress granularity.\n"
"\n"
"  --cpu-batch N  [hybrid only]\n"
"     Queue depth before CPU workers start popping.  Keeps frames\n"
"     stocked for GPU batch fills on fast-decompress workloads.\n"
"     Default: 0 (CPUs pop immediately).\n"
"\n"
"  --cpu-share X  [hybrid only]\n"
"     Fixed CPU share in [0..1].  Disables the EMA (exponential moving\n"
"     average) adaptation.\n"
"\n"
"  --throttle-factor N    [all modes]\n"
"     Slack multiplier for the in-flight frame budget.  Default: 4.\n"
"     Budget = parallelism * factor, capped at RAM/2, floor 32.\n"
"     Affects performance: too low and workers stall waiting for the\n"
"     writer to release permits during NVMe stalls (GC, journal commit);\n"
"     too high wastes RAM with no upside.  Bump to 8 or 16 if you see\n"
"     `source=pipeline` at -v and suspect the writer is bursty.\n"
"     No effect with -T 1 or --sliding-window (no parallelism to bound).\n"
"\n"
"  --throttle-frames N / --no-throttle    [all modes]\n"
"     Explicit in-flight frame cap.  Sentinel values:\n"
"       N >= 1  : explicit cap (bypasses formula).  -v shows source=user.\n"
"       N == 0  : DISABLE the throttle entirely — no permits, no lock,\n"
"                 no accounting.  Use for benchmarking the no-throttle\n"
"                 baseline.  At -v: `[THROTTLE] DISABLED`.\n"
"                 --no-throttle is an alias for --throttle-frames=0.\n"
"       N == -1 : auto (default; use the parallelism * factor formula).\n"
"     With huge inputs you can run out of memory (OOM) if every\n"
"     worker queues frames unbounded — the throttle exists for a\n"
"     reason.  Default is auto.\n"
"\n"
#ifdef HAVE_NVCOMP
"============================================================\n"
" GPU TUNING\n"
"============================================================\n"
"  --gpu-batch N\n"
"     Max GPU subchunks per CUDA stream.  Default: 8 for compress; for\n"
"     decompress 16, auto-scaled up by input size (64 above 10 GiB, 256\n"
"     above 75 GiB).  Each stream targets batches of up to N subchunks;\n"
"     the per-stream binary search may clamp lower if VRAM (GPU\n"
"     memory) is tight (reported at -v as `VRAM-fit: batch=X\n"
"     (requested N, ...)`).\n"
"\n"
"  --gpu-streams N\n"
"     CUDA streams per device (default: 1; 2 for -t verify).\n"
"     More streams = more kernel overlap at the cost of linearly\n"
"     more VRAM.\n"
"\n"
"  --gpu-devices N\n"
"     Number of GPUs to use.  0 = auto (all available GPUs, for both\n"
"     compress and decompress).\n"
"\n"
"  --gpu-mem-frac X\n"
"     Fraction of free VRAM per device to allocate (0.1..0.95,\n"
"     default: 0.60).  Split evenly across --gpu-streams.\n"
"\n"
"  --gpu-only\n"
"     GPU only, no CPU workers (error if no GPU available).\n"
"\n"
"  --pinned {auto|on|off}    (default: off)\n"
"     Control pinned (page-locked) host buffers for H2D\n"
"     (host-to-device) / D2H transfers.  Pinned cudaMemcpy uses a\n"
"     different DMA (direct memory access) path than pageable; on our\n"
"     typical workloads it has measured SLOWER than pageable (compress\n"
"     ~15%% slower, decompress ~2.4x slower) — the cost of locking pages\n"
"     and the extra mmap-or-vector copy outweighs any DMA savings when\n"
"     the input is already in the page cache.  Off is the default for\n"
"     that reason.  The infrastructure is plumbed and exposed in case\n"
"     this differs on your hardware/workload.\n"
"       off (default): no pinned buffers; data goes pageable -> device\n"
"              and device -> std::vector directly.  --no-pinned is an\n"
"              alias for this.\n"
"       auto:  ration to <=50%% of available RAM, summed across all GPU\n"
"              workers.  Streams that fit get pinned; streams that\n"
"              don't fall back to pageable.  Visible at -v as\n"
"              `[PINNED] H2D+D2H <size> reserved (shared per slot)`.\n"
"       on:    pin every stream regardless of system RAM.  May fail\n"
"              with cudaHostAlloc errors on memory-pressured boxes.\n"
"\n"
"  --throttle-factor N / --throttle-frames N    [all modes]\n"
"     Bound the in-flight frame budget shared by GPU + CPU workers and\n"
"     the writer.  Particularly relevant on GPU paths with multiple\n"
"     streams: with N GPUs * S streams * batch=B, peak permit demand\n"
"     can exceed the default budget and stall GPUs behind the writer.\n"
"     Bump --throttle-factor to 8 or 16, or set --throttle-frames\n"
"     directly, if -vv shows GPUs blocked on permit acquire.  See\n"
"     CPU TUNING above for full description.\n"
"\n"
#endif
"============================================================\n"
" HYBRID SCHEDULER\n"
"============================================================\n"
"  --hybrid-floor=MODE\n"
"     GPU queue-depth reservation mode:\n"
"       auto (default): factor in [1.5, 4.0] based on CPU's share of\n"
"                       aggregate throughput.  CPU contributing < 5%\n"
"                       of bytes → factor=4 (heavy lockout, hybrid ≈\n"
"                       gpu-only).  CPU contributing > 20% → factor=1.5\n"
"                       (CPU helps, GPU still reserved).\n"
"       nominal:        streams * gpu_batch_size (v0.12.9 behaviour).\n"
"       off:            no reservation; CPUs compete freely.\n"
"\n"
"  --hybrid-floor-factor X\n"
"     Override the auto scale with a fixed [0..4] multiplier.  Floor =\n"
"     X * active_gpu_streams * gpu_batch_size.  X=2.0 reserves two full\n"
"     GPU rounds ahead of CPU; X=0 lets CPU compete freely.\n"
"\n"
"  -M N, --memlimit=N, --memory=N\n"
"     Memory usage limit in MiB (zstd-compatible).  On decompression,\n"
"     frames requiring a window larger than N MiB are rejected (via\n"
"     ZSTD_d_windowLogMax).  On compression, caps the in-flight\n"
"     frame-throttle budget at roughly N MiB total — useful on shared\n"
"     machines where the default `min(pipeline, RAM/2)` is too generous.\n"
"\n"
"============================================================\n"
" LOGGING\n"
"============================================================\n"
"  -v         Informational messages (mmap choice, preallocation,\n"
"             thread count, etc.).\n"
"  -vv        Per-worker / per-batch detail.\n"
"  -vvv       Trace: per-chunk detail plus a PERFORMANCE BREAKDOWN\n"
"             on completion showing Reader / H2D / Kernel / D2H /\n"
"             CPU / Writer time and throughput.\n"
"  -q, --quiet       Errors only (suppresses progress bar and info).\n"
"  -qq, --silent     Suppress everything, including errors.\n"
"  --progress        Force the progress meter (e.g. when stderr is a\n"
"                    pipe and the TTY (terminal) detection would\n"
"                    suppress it).\n"
"  --no-progress     Suppress the progress meter.\n"
"  --stats-json FILE Write a machine-readable run summary to FILE.\n"
"\n"
"============================================================\n"
" EXIT CODES\n"
"============================================================\n"
"  0  Success\n"
"  1  Runtime error (out of memory, internal failure)\n"
"  2  Bad command-line usage\n"
"  3  I/O error (disk full, read failure, permissions)\n"
"  4  Data error (corrupt input, integrity check failure)\n"
"  5  Reserved: all GPUs failed (since v0.13.54 the run instead falls back\n"
"     to CPU with a warning and exits 0 — data is never left incomplete)\n"
"\n"
"============================================================\n"
" EXAMPLES\n"
"============================================================\n"
"  # Compress with defaults (auto-selects hybrid when a GPU is present)\n"
"  gzstd big.tar\n"
"\n"
"  # Decompress, overwriting any existing output\n"
"  gzstd -d -f big.tar.zst\n"
"\n"
"  # Overwrite in place (faster on ext4; non-atomic)\n"
"  gzstd -d --overwrite big.tar.zst\n"
"\n"
"  # Pipe compression with explicit level 9\n"
"  tar cf - ./src | gzstd -9 > src.tar.zst\n"
"\n"
"  # Build a .tar.zst directly, reading members in parallel (no `tar`)\n"
"  gzstd -o backup.tar.zst --tar --exclude '*/.cache/*' ~/docs ~/proj\n"
"\n"
"  # Extract a .tar.zst archive (decompress + untar in one parallel pass)\n"
"  gzstd -d --tar backup.tar.zst              # into the current directory\n"
"  gzstd -d --tar -C /restore backup.tar.zst  # into /restore (must exist)\n"
"\n"
"  # Preserve POSIX ACLs and extended attributes (give the flags on BOTH\n"
"  # create and extract, as GNU tar requires)\n"
"  gzstd -o home.tar.zst --tar --acls --xattrs ~user\n"
"  gzstd -d --tar --acls --xattrs -C /restore home.tar.zst\n"
"\n"
"  # Full-system backup: one archive, stay on each filesystem (skips /proc,\n"
"  # /sys, /dev; list real partitions as separate sources to include them)\n"
"  sudo gzstd -o root.tar.zst --tar --one-file-system / /home /boot\n"
"\n"
"  # CPU-only baseline at level 3 using all cores\n"
"  gzstd --cpu-only -T 0 big.tar\n"
"\n"
"  # GPU-only with 4 streams per device, 512-subchunk batches (per stream)\n"
"  gzstd --gpu-only --gpu-streams 4 --gpu-batch 512 big.tar\n"
"\n"
"  # Maximum ratio on repetitive data (single-frame, zstd-compatible)\n"
"  gzstd --sliding-window -19 repetitive.log\n"
"\n"
"  # Integrity check (no output), with trace-level perf breakdown\n"
"  gzstd -t -vvv big.tar.zst\n"
"\n"
"  # Stream through a pipe with forced progress on stderr\n"
"  gzstd --progress -d archive.tar.zst | tar xf -\n"
"\n"
"  # Write run statistics as JSON for later analysis\n"
"  gzstd --stats-json run.json big.tar\n"
"\n"
"For a condensed option list, run `gzstd -h`.\n";
}

static void print_version()
{
#ifdef HAVE_NVCOMP
  std::cout << "gzstd " << GZSTD_VERSION << " (CPU + nvCOMP) MT-CPU + Hybrid scheduling\n";
#else
  std::cout << "gzstd " << GZSTD_VERSION << " (CPU-only) MT compression\n";
#endif
}

// Convert a value string to integer/double with friendly errors.  std::sto*
// throw std::invalid_argument on garbage and std::out_of_range on overflow;
// letting those propagate to main() prints the raw what() and dumps core-like
// output instead of a usage hint.  Each helper validates the full string is
// consumed so "--gpu-streams=12abc" is rejected rather than silently truncated.
static uint64_t parse_u64_value(const std::string & pref, const std::string & v)
{
  try {
    size_t pos = 0;
    unsigned long long n = std::stoull(v, &pos);
    if (pos != v.size()) die_usage("invalid value for " + pref + ": " + v);
    return (uint64_t)n;
  } catch (const std::invalid_argument &) {
    die_usage("invalid value for " + pref + ": " + v);
  } catch (const std::out_of_range &) {
    die_usage("value out of range for " + pref + ": " + v);
  }
  return 0; // unreachable; die_usage exits
}
static int parse_int_value(const std::string & pref, const std::string & v)
{
  try {
    size_t pos = 0;
    int n = std::stoi(v, &pos);
    if (pos != v.size()) die_usage("invalid value for " + pref + ": " + v);
    return n;
  } catch (const std::invalid_argument &) {
    die_usage("invalid value for " + pref + ": " + v);
  } catch (const std::out_of_range &) {
    die_usage("value out of range for " + pref + ": " + v);
  }
  return 0;
}
// True if the whole string is a (optionally signed) base-10 integer.  Used to
// decide whether a separate token after a flag like `-T` is its value or the
// next argument — so `-T --cpu-only` / `-T file.zst` don't crash std::stoi.
static bool looks_like_int(const char * s)
{
  if (!s || !*s) return false;
  size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
  if (!s[i]) return false;                 // sign with no digits
  for (; s[i]; ++i) if (s[i] < '0' || s[i] > '9') return false;
  return true;
}
static double parse_double_value(const std::string & pref, const std::string & v)
{
  try {
    size_t pos = 0;
    double d = std::stod(v, &pos);
    if (pos != v.size()) die_usage("invalid value for " + pref + ": " + v);
    if (!std::isfinite(d)) die_usage("value not finite for " + pref + ": " + v);
    return d;
  } catch (const std::invalid_argument &) {
    die_usage("invalid value for " + pref + ": " + v);
  } catch (const std::out_of_range &) {
    die_usage("value out of range for " + pref + ": " + v);
  }
  return 0.0;
}

// Parse a --name=VALUE or --name VALUE argument as size_t.
// Returns true if this argument matched, advancing 'i' if needed.
static bool parse_num_arg(const std::string & name, int & i, int argc,
                          char ** argv, size_t & out, bool * was_set = nullptr)
{
  const std::string pref = "--" + name;
  std::string a = argv[i];

  // Form: --name=VALUE
  if (a.rfind(pref + "=", 0) == 0) {
    std::string v = a.substr(pref.size() + 1);
    if (v.empty()) die_usage("missing value for " + pref);
    out = (size_t)parse_u64_value(pref, v);
    if (was_set) *was_set = true;
    return true;
  }
  // Form: --name VALUE (next argv element)
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
    out = (size_t)parse_u64_value(pref, argv[++i]);
    if (was_set) *was_set = true;
    return true;
  }

  return false;
}
// Parse a --name=VALUE or --name VALUE argument as int.
static bool parse_int_arg(const std::string & name, int & i, int argc,
                          char ** argv, int & out)
{
  const std::string pref = "--" + name;
  std::string a = argv[i];

  if (a.rfind(pref + "=", 0) == 0) {
    std::string v = a.substr(pref.size() + 1);
    if (v.empty()) die_usage("missing value for " + pref);
    out = parse_int_value(pref, v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
    out = parse_int_value(pref, argv[++i]);
    return true;
  }

  return false;
}
// Parse a --name=VALUE or --name VALUE argument as string.
static bool parse_str_arg(const std::string & name, int & i, int argc,
                          char ** argv, std::string & out)
{
  const std::string pref = "--" + name;
  std::string a = argv[i];

  if (a.rfind(pref + "=", 0) == 0) {
    std::string v = a.substr(pref.size() + 1);
    if (v.empty()) die_usage("missing value for " + pref);
    out = v;
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
    out = argv[++i];
    return true;
  }

  return false;
}
#if __cplusplus >= 201703L
[[maybe_unused]]
#endif
// Parse a --name=VALUE or --name VALUE argument as double.
static bool parse_double_arg(const std::string & name, int & i, int argc,
                             char ** argv, double & out)
{
  const std::string pref = "--" + name;
  std::string a = argv[i];

  if (a.rfind(pref + "=", 0) == 0) {
    std::string v = a.substr(pref.size() + 1);
    if (v.empty()) die_usage("missing value for " + pref);
    out = parse_double_value(pref, v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
    out = parse_double_value(pref, argv[++i]);
    return true;
  }

  return false;
}

// zstd-compat: emit a single-line warning that a zstd option is accepted but
// ignored.  Prints to stderr at V_ERROR so -q silences it but default runs see
// it.  Used by the compat layer in parse_args for flags gzstd accepts for
// drop-in purposes but does not implement.
static int g_verbosity_for_compat = 2; // set at arg-parse start; used here
static void warn_ignored_zstd_opt(const std::string & opt_name,
                                   const std::string & reason = "")
{
  if (g_verbosity_for_compat < 2) return; // -q and -qq suppress
  std::string msg = "warning: " + opt_name
                  + " accepted for zstd compatibility but ignored";
  if (!reason.empty()) msg += " (" + reason + ")";
  std::fprintf(stderr, "gzstd: %s\n", msg.c_str());
}
// Eat a VALUE that follows a zstd long option, whether as `--opt VALUE`
// (separate argv) or `--opt=VALUE` (joined).  Returns true if the option name
// matched.  The value itself is discarded — this is for zstd flags we warn on.
static bool eat_zstd_value_opt(const std::string & name, int & i, int argc,
                                char ** argv)
{
  const std::string pref = "--" + name;
  std::string a = argv[i];
  if (a == pref) {
    if (i + 1 < argc) ++i; // consume value
    return true;
  }
  if (a.rfind(pref + "=", 0) == 0) return true;
  return false;
}

/*======================================================================
 Progress & metrics
======================================================================*/
struct Meter {
  std::atomic< uint64_t > read_bytes   { 0 };
  std::atomic< uint64_t > wrote_bytes  { 0 };
  std::atomic< uint64_t > tasks_done   { 0 };  // frames handed to writer (written in-order)
  std::atomic< uint64_t > total_frames { 0 };  // total frames to process (set by producer)
  std::atomic< uint64_t > total_out    { 0 };  // expected total output bytes (set when known)
  std::atomic< bool >     total_out_final { false }; // true once reader is done and total_out won't grow
  mutable std::atomic< uint64_t > read_elapsed_ms { 0 }; // elapsed ms when read completed (frozen rate)
  // Writer-state accounting (always on; reported at -v).  Answers "why isn't
  // the output device pegged?" — at any instant the output side is either
  // physically writing, idle because the next in-sequence frame is missing
  // while LATER frames sit buffered (a straggler: the pipeline's fault), or
  // idle with nothing buffered at all (upstream compute/read can't keep up).
  // disk_ns accrues on the AsyncWritePool worker; the two wait buckets accrue
  // on the writer thread, so percentages overlap and don't sum to 100.
  std::atomic< uint64_t > writer_disk_ns    { 0 }; // inside write_sparse (scan + write/seek)
  std::atomic< uint64_t > writer_iowrite_ns { 0 }; // subset: inside the write/seek syscalls only (rest = zero-scan)
  std::atomic< uint64_t > writer_hol_ns     { 0 }; // head-of-line: waiting while later frames sit buffered
  std::atomic< uint64_t > writer_hol_depth_ns { 0 }; // ∑ wait_ns × frames buffered behind the gap (ns·frames)
  std::atomic< uint64_t > writer_starved_ns { 0 }; // waiting with nothing buffered
  // Reader-state accounting, mirror of the writer's.  io = inside fread/pread;
  // parse = finding frame boundaries (decompress only — ZSTD_findFrameCompressedSize
  // / getFrameContentSize; zero for compress, which slices at fixed offsets);
  // copy = duplicating bytes into tasks (zero for the mmap and pooled zero-copy
  // readers — its presence IS the diagnosis); blocked = waiting for downstream
  // capacity (a pool buffer on compress, a bounded-queue push on decompress).
  // A reader near 100% of wall in io+parse+copy is the run's faucet; near 100%
  // in blocked means the CONSUMERS are the faucet and the reader is starved out.
  std::atomic< uint64_t > reader_io_ns      { 0 };
  std::atomic< uint64_t > reader_parse_ns   { 0 };
  std::atomic< uint64_t > reader_copy_ns    { 0 };
  std::atomic< uint64_t > reader_blocked_ns { 0 };
  std::atomic< int >      reader_threads    { 1 }; // io/parse/copy/blocked sum across these
  std::chrono::steady_clock::time_point t0 { std::chrono::steady_clock::now() };

  // Zero all counters and restart the clock — used when a GPU fault forces a
  // full CPU-only rebuild so the rebuild's progress/summary start fresh.
  void reset() {
    read_bytes = 0; wrote_bytes = 0; tasks_done = 0; total_frames = 0;
    total_out = 0; total_out_final = false; read_elapsed_ms = 0;
    writer_disk_ns = 0; writer_iowrite_ns = 0; writer_hol_ns = 0;
    writer_hol_depth_ns = 0; writer_starved_ns = 0;
    reader_io_ns = 0; reader_parse_ns = 0; reader_copy_ns = 0;
    reader_blocked_ns = 0; reader_threads = 1;
    t0 = std::chrono::steady_clock::now();
  }
};

// Interpret the Meter's writer-state accounting into a one-line diagnosis.
// Inputs are fractions of run wall time; they accrue on different threads,
// so they overlap and need not sum to 1.  Mirrors the manual triage order:
// a pegged sink is success; otherwise whichever idle state dominates names
// the culprit.  avg_stuck (mean frames buffered behind the missing one
// during head-of-line waits) separates a true straggler from healthy
// out-of-order jitter: N parallel workers buffer up to ~N frames in normal
// operation, so only depths well beyond the in-flight window indicate a
// pipeline pathology — small-depth HOL means the missing frame is simply
// still being computed, i.e. upstream-bound.  (Groundwork for --adapt:
// these are the regime signals.)
static const char * writer_verdict(double busy, double hol, double starved,
                                   double avg_stuck, double healthy_window)
{
  if (busy >= 0.80)
    return "output device saturated — the sink is the bottleneck (optimal for this device)";
  if (hol >= 0.25 && avg_stuck > healthy_window)
    return "stragglers — writer idled while far more frames sat buffered than the "
           "in-flight window; pipeline ordering, not the device, capped output";
  if (hol + starved >= 0.25)
    return "upstream-bound — compute/read could not fill the writer; the engines, "
           "not the device, capped output";
  if (busy >= 0.50)
    return "output device mostly busy — near sink-limited";
  return "no dominant writer-side state — output was capped upstream of the result store";
}
static bool is_stderr_tty() { return isatty(fileno(stderr)) != 0; }
static bool is_stdout_tty() { return isatty(fileno(stdout)) != 0; }
// Format a byte count as a human-readable string (e.g. "3.14 GiB").
static void human_bytes(double x, char * buf, size_t n)
{
  const char * units[] = { "B", "KiB", "MiB", "GiB", "TiB" };
  int k = 0;
  while (x >= 1024.0 && k < 4) {
    x /= 1024.0;
    ++k;
  }
  std::snprintf(buf, n, "%.2f %s", x, units[k]);
}

// Try to raise the calling thread's scheduling priority so that I/O-critical
// threads (reader, writer) aren't starved by the CPU worker pool.  Silently
// ignores failures (non-root, unsupported OS).
// Skip in GPU-only mode where there is no CPU pool contention  boosting I/O
// threads there just penalises the GPU worker's CUDA calls.
static void try_boost_io_priority(bool has_cpu_pool)
{
  if (!has_cpu_pool) return;
#ifdef __linux__
  if (nice(-5) == -1) { /* best-effort; ignore */ }
#endif
}

// Pin the calling thread to a specific CPU core.  Best-effort; silently
// ignores failures.  Used to keep reader/writer on dedicated cores so
// they aren't preempted by the CPU worker pool.
__attribute__((unused))
static void pin_thread_to_core(int core)
{
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
  (void)core;
#endif
}

// Returns the set of cores reserved for I/O threads (reader + writer).
// Workers should avoid these cores.  Returns {0, 1} on most systems.
static std::vector<int> get_io_cores()
{
  unsigned hw = std::thread::hardware_concurrency();
  if (hw <= 4) return {0};  // tiny system: share one core
  return {0, 1};            // dedicate two cores to I/O
}
// ANSI color/style codes used by the progress bar.
// All "in" metrics are cyan; all "out" metrics are green.
#define PR_RESET    "\033[0m"
#define PR_DIM      "\033[2m"
#define PR_CYAN     "\033[36m"       // cyan          — in label, size, rate
#define PR_CYAN_B   "\033[1;96m"     // bold bright cyan  — in pct
#define PR_GREEN    "\033[32m"       // green         — out label, size, rate
#define PR_GREEN_B  "\033[1;92m"     // bold bright green — out pct
#define PR_YELLOW_B "\033[1;93m"     // bold bright yellow — out pct when unknown

// Emit one progress line.  in_pct and out_pct are 0-100; pass -1 when unknown.
// Format: "in:<pct> <size> <rate>/s  out:<pct> <size> <rate>/s"
// All "in" tokens are cyan; all "out" tokens are green.
static void progress_emit_line(double in_pct, double out_pct,
                               const char * in_s, const char * out_s,
                               const char * in_rate_s, const char * out_rate_s)
{
  char in_pct_s[12], out_pct_s[12];
  const char * out_pct_color;
  if (in_pct  >= 0.0) std::snprintf(in_pct_s,  sizeof(in_pct_s),  "%.1f%%", in_pct);
  else                 std::snprintf(in_pct_s,  sizeof(in_pct_s),  "---");
  if (out_pct >= 0.0) {
    std::snprintf(out_pct_s, sizeof(out_pct_s), "%.1f%%", out_pct);
    out_pct_color = PR_GREEN_B;
  } else {
    std::snprintf(out_pct_s, sizeof(out_pct_s), "---");
    out_pct_color = PR_YELLOW_B;
  }

  // Buffer is larger than plain text to accommodate ANSI escape sequences.
  // Only the ##.#% values are bright/bold; labels, sizes, and rates use dim cyan/green.
  char line[512];
  int n = std::snprintf(line, sizeof(line),
      PR_CYAN "in:" PR_CYAN_B "%s" PR_RESET   // in label dim cyan + pct bold bright cyan
      PR_CYAN " %s %s/s"                      // in size + rate dim cyan
      "\033[90m | \033[0m"                     // dark grey separator
      PR_GREEN "out:" "%s" "%s" PR_RESET       // out label dim green + pct (bold green or yellow)
      PR_GREEN " %s %s/s" PR_RESET " ",        // out size + rate dim green
      in_pct_s, in_s, in_rate_s,
      out_pct_color, out_pct_s, out_s, out_rate_s);
  if (n < 0) n = 0;
  // \r overwrites the line; \033[K clears to end; PR_RESET before \033[K
  // prevents the erase from inheriting a background color.
  std::fprintf(stderr, "\r%.*s\033[K", std::min(n, (int)sizeof(line) - 1), line);
  std::fflush(stderr);
}
#undef PR_RESET
#undef PR_DIM
#undef PR_CYAN
#undef PR_CYAN_B
#undef PR_GREEN
#undef PR_GREEN_B
#undef PR_YELLOW_B

// Input size if knowable up front: a named regular file, OR a seekable stdin
// redirect ("gzstd < file", "gzstd -d < file.zst", "< /dev/sdX").  0 for a
// true pipe (tar -I gzstd, "cat | gzstd") or unknown.  Drives the progress
// denominator (real % instead of "---") and auto-progress (a known size ⇒
// NOT a true pipe, so the bar is worth showing).
static uint64_t known_input_size(const Options & opt, FILE * in)
{
  if (opt.input != "-") {
    std::error_code ec; uintmax_t z = fs::file_size(opt.input, ec);
    return ec ? 0 : (uint64_t)z;            // named regular file
  }
#ifndef _WIN32
  if (in) {                                 // stdin: is it a "< file" redirect?
    const int fd = ::fileno(in);
    struct stat st;
    if (::fstat(fd, &st) == 0) {
      if (S_ISREG(st.st_mode)) return (uint64_t)st.st_size;
      if (S_ISBLK(st.st_mode)) {            // block device: size via SEEK_END
        off_t cur = ::lseek(fd, 0, SEEK_CUR), end = ::lseek(fd, 0, SEEK_END);
        if (cur >= 0 && end > 0) { ::lseek(fd, cur, SEEK_SET); return (uint64_t)end; }
      }
    }
  }
#endif
  return 0;                                 // true pipe / unknown
}

static void progress_loop(const Options & opt, const Meter * m, uint64_t total_in, std::atomic< bool > * done_flag)
{
  // Progress bar at V_DEFAULT and V_VERBOSE only.
  // At V_DEBUG+, per-thread summaries provide progress; the bar would collide.
  // At V_DEFAULT, suppress if stderr is not a TTY OR if stdin is a TRUE pipe.
  // A redirected stdin from a real file ("gzstd -d < big.zst") has a known
  // size (total_in > 0), so it is NOT a true pipe — show the bar with real
  // percentages.  Only a genuine pipe (total_in == 0: tar -I gzstd, cat |
  // gzstd, terminal) is suppressed.  --progress overrides everything but the
  // V_DEBUG cap.
  if (opt.verbosity < V_DEFAULT || opt.verbosity >= V_DEBUG) return;
  if (opt.verbosity == V_DEFAULT && !opt.force_progress) {
    if (!is_stderr_tty()) return;
    if (opt.input == "-" && total_in == 0) return;
  }
  g_progress_active.store(true, std::memory_order_relaxed);
  using namespace std::chrono; using namespace std::chrono_literals;
  const bool is_test = (opt.mode == Mode::TEST);
  // out% denominator (total output) isn't known upfront — for decompress it
  // grows as the reader discovers frames, for compress it's an estimate from
  // the running ratio — so the raw percentage can dip when the estimate
  // jumps.  Clamp the displayed value to be monotonically non-decreasing: a
  // brief forward stall is acceptable, going backwards is not.
  double out_pct_floor = 0.0;
  while (!done_flag->load()) {
    std::this_thread::sleep_for(200ms);
    uint64_t in       = m->read_bytes.load();
    uint64_t out      = m->wrote_bytes.load();
    uint64_t t_out    = m->total_out.load();
    uint64_t t_done   = m->tasks_done.load();
    uint64_t t_frames = m->total_frames.load();
    auto dt      = steady_clock::now() - m->t0;
    double secs  = duration_cast< duration<double> >(dt).count();
    // Freeze input rate once reading is complete: dividing fixed read_bytes
    // by ever-growing elapsed time makes the rate decay after the reader stops.
    // Snapshot the elapsed time when we first see in >= total_in, then reuse it.
    double in_secs = secs;
    if (total_in > 0 && in >= total_in) {
      uint64_t frozen_ms = m->read_elapsed_ms.load(std::memory_order_relaxed);
      if (frozen_ms == 0) {
        uint64_t ms = (uint64_t)(secs * 1000.0);
        m->read_elapsed_ms.store(ms > 0 ? ms : 1, std::memory_order_relaxed);
        frozen_ms = ms > 0 ? ms : 1;
      }
      in_secs = double(frozen_ms) / 1000.0;
    }
    double in_rate  = in_secs > 0 ? double(in)  / in_secs : 0.0;
    double out_rate = secs > 0 ? double(out) / secs : 0.0;
    char in_s[64], out_s[64], rate_s[64], out_rate_s[64];
    human_bytes(double(in),  in_s,  sizeof(in_s));
    human_bytes(double(out), out_s, sizeof(out_s));
    human_bytes(in_rate,  rate_s,     sizeof(rate_s));
    human_bytes(out_rate, out_rate_s, sizeof(out_rate_s));

    // in_pct: how much input has been read (always known for regular files).
    double in_pct = (total_in > 0) ? std::min(100.0, 100.0 * double(in) / double(total_in)) : -1.0;

    // out_pct: tracks output completion.
    //   Reader done (total_out_final): exact wrote_bytes / total_out.
    //   Reader still running: total_out is partial (growing denominator would
    //     make the percentage jump high then drop).  Estimate the final total
    //     output from the current ratio: estimated = total_in × (total_out / read_bytes).
    //     This converges as more of the file is read and only increases
    //     monotonically in practice (uniform compressibility).
    //   Compression after reader, frames still in flight: tasks_done / total_frames.
    //   Unknown (pipe input, no data yet): show ---.
    bool out_final = m->total_out_final.load(std::memory_order_acquire);
    double out_pct;
    if (out_final && t_out > 0) {
      // Decompress, reader done: total_out is stable — smooth byte-level.
      out_pct = std::min(99.9, 100.0 * double(out) / double(t_out));
    } else if (t_frames > 0 && t_done < t_frames) {
      // Reader done (total_frames set), frames still in flight — frame-level.
      out_pct = std::min(99.9, 100.0 * double(t_done) / double(t_frames));
    } else if (t_frames > 0 && t_out > 0) {
      // All frames done, AIO still draining — byte-level.
      out_pct = std::min(99.9, 100.0 * double(out) / double(t_out));
    } else if (total_in > 0 && in > 0 && t_out > 0) {
      // Reader still running (t_frames not set yet): estimate total output
      // from current input/output ratio.  Converges as more data is read;
      // slightly conservative for compression (in-flight frames deflate the
      // ratio) which is better than an inflated percentage.
      double estimated_total = double(total_in) * (double(t_out) / double(in));
      out_pct = std::min(99.9, 100.0 * double(out) / std::max(1.0, estimated_total));
    } else {
      out_pct = -1.0;
    }
    // Monotonic clamp (see out_pct_floor declaration): never display a lower
    // out% than already shown.  Skips the unknown (-1) state.
    if (out_pct >= 0.0) {
      if (out_pct < out_pct_floor) out_pct = out_pct_floor;
      else                         out_pct_floor = out_pct;
    }

    if (is_test) {
      // Test mode: show verified bytes and decompression throughput.
      char line[512];
      if (in_pct >= 0.0)
        std::snprintf(line, sizeof(line),
            "\033[36min:\033[1;96m%.1f%%\033[0m %s  "
            "\033[32mverified:\033[1;92m%s\033[0m  "
            "\033[2m@ %s/s\033[0m ",
            in_pct, in_s, out_s, out_rate_s);
      else
        std::snprintf(line, sizeof(line),
            "\033[36min:\033[0m%s  "
            "\033[32mverified:\033[1;92m%s\033[0m  "
            "\033[2m@ %s/s\033[0m ",
            in_s, out_s, out_rate_s);
      std::fprintf(stderr, "\r%s\033[K", line);
      std::fflush(stderr);
    } else {
      progress_emit_line(in_pct, out_pct, in_s, out_s, rate_s, out_rate_s);
    }
  }
  // Final sample (no newline  the completion summary will overwrite this line)
  uint64_t in       = m->read_bytes.load();
  uint64_t out      = m->wrote_bytes.load();
  uint64_t t_out    = m->total_out.load();
  uint64_t t_done   = m->tasks_done.load();
  uint64_t t_frames = m->total_frames.load();
  auto dt      = std::chrono::steady_clock::now() - m->t0;
  double secs  = std::chrono::duration_cast< std::chrono::duration<double> >(dt).count();
  double in_secs = secs;
  {
    uint64_t frozen_ms = m->read_elapsed_ms.load(std::memory_order_relaxed);
    if (frozen_ms > 0) in_secs = double(frozen_ms) / 1000.0;
  }
  double in_rate  = in_secs > 0 ? double(in)  / in_secs : 0.0;
  double out_rate = secs > 0 ? double(out) / secs : 0.0;
  char in_s[64], out_s[64], rate_s[64], out_rate_s[64];
  human_bytes(double(in),  in_s,  sizeof(in_s));
  human_bytes(double(out), out_s, sizeof(out_s));
  human_bytes(in_rate,  rate_s,     sizeof(rate_s));
  human_bytes(out_rate, out_rate_s, sizeof(out_rate_s));
  double in_pct  = (total_in  > 0) ? std::min(100.0, 100.0 * double(in)  / double(total_in))  : -1.0;
  double out_pct;
  // Final line: reader is always done here, so total_out is stable
  if (opt.mode == Mode::DECOMPRESS && t_out > 0)
    out_pct = std::min(100.0, 100.0 * double(out) / double(t_out));
  else if (t_frames > 0)
    out_pct = std::min(100.0, 100.0 * double(t_done) / double(t_frames));
  else if (t_out > 0)
    out_pct = std::min(100.0, 100.0 * double(out)    / double(t_out));
  else
    out_pct = -1.0;
  progress_emit_line(in_pct, out_pct, in_s, out_s, rate_s, out_rate_s);
  // No \n here  the completion summary overwrites this line with \r
  g_progress_active.store(false, std::memory_order_relaxed);
}

/*======================================================================
 I/O helpers
======================================================================*/
static FILE * open_input(const std::string & path)
{
  if (path == "-") {
    set_binary_mode(stdin);
    return stdin;
  }
  FILE * f = std::fopen(path.c_str(), "rb");
  if (!f) die_io("cannot open input: " + path);
  return f;
}
// Open a temporary file for atomic write: write to .tmp, then rename on success.
static FILE * open_output_atomic(const std::string & out, std::string & tmp_path)
{
  tmp_path = out + ".gzstd.tmp";
  register_tmp_file(tmp_path);
  FILE * f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) die_io("cannot open temp output: " + tmp_path);
  std::setvbuf(f, nullptr, _IOFBF, 1 * 1024 * 1024);
  return f;
}
// Flush file data to disk (POSIX only).
static void fsync_file(FILE * f)
{
#if defined(_POSIX_VERSION)
  // Flush the stdio buffer to the fd FIRST: the output FILE* has a 1 MiB
  // full-buffer (setvbuf), so without this fsync() would sync only what has
  // already reached the fd and miss up to ~1 MiB of trailing output still in
  // the userspace buffer — defeating --sync-output's durability guarantee for
  // the tail (fclose flushes it afterward, but nothing fsyncs it then).
  std::fflush(f);
  int fd = fileno(f);
  fsync(fd);
#else
  (void)f;
#endif
}

// Robust fwrite that handles EINTR and short writes (pipes, signals)
static size_t robust_fwrite(const void * ptr, size_t size, FILE * f)
{
  const char * p = static_cast<const char *>(ptr);
  size_t remaining = size;
  while (remaining > 0) {
    errno = 0;  // don't let a stale EINTR from an earlier syscall loop us forever
    size_t w = std::fwrite(p, 1, remaining, f);
    if (w > 0) {
      p += w;
      remaining -= w;
    } else {
      // Check for EINTR (interrupted by signal); retry
      if (errno == EINTR) continue;
      // Real error (EPIPE, disk full, etc.)
      return size - remaining;
    }
  }
  return size;
}

/*======================================================================
 O_DIRECT writer  bypasses page cache for faster sequential writes
 -----------------------------------------------------------------------
 O_DIRECT requires aligned buffers and aligned write sizes.  We accumulate
 data in an aligned buffer and flush in aligned chunks.  The final partial
 block is handled by dropping O_DIRECT for the last write (via fallback
 to pwrite without the flag) or by ftruncate after aligned overwrite.

 Used only for regular file output.  Pipes, stdout, and device files
 fall back to standard fwrite via robust_fwrite.
======================================================================*/

/*======================================================================
 MmapRegion — RAII read-only memory-mapped file
 -----------------------------------------------------------------------
 Maps a regular file into the address space for zero-copy producer reads.
 Workers access the mapped pages directly via Task::view_ptr, avoiding
 the fread + memcpy overhead that serialises the single-threaded producer.
 MADV_SEQUENTIAL tells the kernel to read ahead aggressively.
======================================================================*/
#ifndef _WIN32
class MmapRegion {
public:
  MmapRegion() = default;
  ~MmapRegion() { reset(); }
  MmapRegion(const MmapRegion &) = delete;
  MmapRegion & operator=(const MmapRegion &) = delete;

  bool open(const char * path) {
    reset();
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size == 0) { ::close(fd); return false; }
    size_ = (size_t)st.st_size;
    ptr_ = (const char *)::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr_ == MAP_FAILED) { ptr_ = nullptr; size_ = 0; return false; }
    ::madvise((void *)ptr_, size_, MADV_SEQUENTIAL);
    return true;
  }

  void reset() {
    if (ptr_) { ::munmap((void *)ptr_, size_); ptr_ = nullptr; size_ = 0; }
  }

  const char * data() const { return ptr_; }
  size_t       size() const { return size_; }
  explicit operator bool() const { return ptr_ != nullptr; }

private:
  const char * ptr_ = nullptr;
  size_t       size_ = 0;
};

#endif // !_WIN32

#ifndef _WIN32
// FALLOC_FL_* may not be exposed by <fcntl.h> on all libc versions; the values
// are stable kernel ABI.  Used by DirectWriter::seek_forward to punch holes so
// sparse output and preallocation coexist (preallocate gives dense writes their
// extent-stall-free path; punch-hole restores sparseness over skipped zeros).
#ifdef __linux__
#ifndef FALLOC_FL_KEEP_SIZE
#define FALLOC_FL_KEEP_SIZE 0x01
#endif
#ifndef FALLOC_FL_PUNCH_HOLE
#define FALLOC_FL_PUNCH_HOLE 0x02
#endif
#endif
static inline uint64_t now_ns();   // defined below; used by DirectWriter write timing
// -v writer breakdown: time inside the O_DIRECT ::write syscall, kept at file
// scope so the [WRITER] summary can read it after the DirectWriter is destroyed
// (g_direct_writer is null by summary time).  Reset per output in open().
static std::atomic<uint64_t> g_odirect_write_ns{0};
static std::atomic<bool>     g_odirect_used{false};
class DirectWriter {
public:
  static constexpr size_t ALIGN = 4096;          // filesystem block alignment

  DirectWriter() = default;
  ~DirectWriter() { close(); }

  // Open file with O_DIRECT.  Returns false if O_DIRECT not supported
  // (caller should fall back to FILE*).
  bool open(const std::string & path) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0666);
    if (fd_ < 0) return false;
    // Allocate the aligned buffer pool.
    for (int i = 0; i < NBUF; ++i) {
      if (posix_memalign(&bufs_[i], ALIGN, BUF_CAP) != 0) {
        for (int j = 0; j < i; ++j) { ::free(bufs_[j]); bufs_[j] = nullptr; }
        ::close(fd_); fd_ = -1; return false;
      }
    }
    buf_used_ = 0; cur_bi_ = 0;
    logical_written_ = 0; phys_written_ = 0;
    werr_.store(false, std::memory_order_relaxed); wt_done_ = false;
    free_.clear(); for (int i = 1; i < NBUF; ++i) free_.push_back(i);  // 0 is the first fill buffer
    ops_.clear();
    g_odirect_write_ns.store(0, std::memory_order_relaxed);  // -v: per-output write-time
    g_odirect_used.store(true, std::memory_order_relaxed);
    wt_ = std::thread(&DirectWriter::writer_loop, this);
    return true;
  }

  // Bounce-copy into the current aligned buffer; when it fills (BUF_CAP, a
  // multiple of ALIGN), hand it to the write thread and take a free one.  Only
  // full buffers flush mid-stream, so there is no unaligned tail except at
  // finalize.  The copy runs here (caller thread); the ::write runs async.
  bool write(const void * data, size_t len) {
    if (fd_ < 0 || werr_.load(std::memory_order_relaxed)) return false;
    const char * src = static_cast<const char *>(data);
    while (len > 0) {
      size_t space = BUF_CAP - buf_used_;
      size_t copy = std::min(len, space);
      std::memcpy(static_cast<char*>(bufs_[cur_bi_]) + buf_used_, src, copy);
      buf_used_ += copy; src += copy; len -= copy;
      if (buf_used_ == BUF_CAP) {                 // full → hand off
        enqueue({OP_WRITE, cur_bi_, BUF_CAP, 0});
        logical_written_ += BUF_CAP;
        cur_bi_ = acquire_free();
        if (cur_bi_ < 0) return false;            // write error surfaced
        buf_used_ = 0;
      }
    }
    return !werr_.load(std::memory_order_relaxed);
  }

  // Finalize: enqueue the last partial buffer (aligned body O_DIRECT + sub-ALIGN
  // tail without O_DIRECT, handled in writer_loop), drain and join the write
  // thread, then truncate off any preallocation slack.
  bool finalize() {
    if (fd_ < 0) return false;
    if (!wt_.joinable()) return !werr_.load(std::memory_order_relaxed);  // never opened/already done
    if (buf_used_ > 0) {
      size_t tail = buf_used_ - (buf_used_ / ALIGN) * ALIGN;
      enqueue({OP_WRITE, cur_bi_, buf_used_, tail});
      logical_written_ += buf_used_;
      buf_used_ = 0;
    }
    { std::lock_guard<std::mutex> lk(mx_); wt_done_ = true; }
    cv_op_.notify_one();
    wt_.join();
    if (werr_.load(std::memory_order_relaxed)) return false;
    if (preallocated_ > 0 && logical_written_ < preallocated_) {
      if (::ftruncate(fd_, (off_t)logical_written_) != 0) { /* best-effort */ }
    }
    return true;
  }

  void close() {
    // Join the write thread if finalize() didn't (e.g. an error/abort path).
    if (wt_.joinable()) {
      { std::lock_guard<std::mutex> lk(mx_); wt_done_ = true; }
      cv_op_.notify_one();
      wt_.join();
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    for (int i = 0; i < NBUF; ++i) if (bufs_[i]) { ::free(bufs_[i]); bufs_[i] = nullptr; }
  }

  // Pre-allocate disk space to avoid per-write extent allocation overhead.
  // On NVMe, this can improve sequential write throughput by 2-4x because the
  // filesystem doesn't need to allocate new extents + journal on every write().
  // Falls back gracefully on filesystems that don't support fallocate.
  bool preallocate(uint64_t size) {
    if (fd_ < 0 || size == 0) return false;
#ifdef __linux__
    int ret = ::fallocate(fd_, 0, 0, (off_t)size);
    if (ret == 0) { preallocated_ = size; return true; }
    // EOPNOTSUPP / ENOSYS: filesystem doesn't support fallocate — not fatal
#endif
    return false;
  }

  // Logical output position (what's been enqueued + what's buffered).  Used by
  // write_sparse to keep sparse seeks O_DIRECT-aligned, so it must reflect the
  // caller's view, not the (lagging) physical write position.
  size_t total_bytes() const { return (size_t)logical_written_ + buf_used_; }
  int fd() const { return fd_; }
  uint64_t preallocated() const { return preallocated_; }

  // Seek forward by 'offset' bytes (sparse hole).  Flush the current buffer
  // (ALIGN-aligned here — write_sparse only seeks at aligned positions, see its
  // skip_ok guard), then enqueue an ordered SEEK so it lands after the buffered
  // data on the write thread.  The hole-punch + lseek happen there.
  bool seek_forward(size_t offset) {
    if (fd_ < 0 || werr_.load(std::memory_order_relaxed)) return false;
    if (buf_used_ > 0) {
      size_t tail = buf_used_ - (buf_used_ / ALIGN) * ALIGN;  // normally 0 (aligned seek point)
      enqueue({OP_WRITE, cur_bi_, buf_used_, tail});
      logical_written_ += buf_used_;
      cur_bi_ = acquire_free();
      if (cur_bi_ < 0) return false;
      buf_used_ = 0;
    }
    enqueue({OP_SEEK, -1, offset, 0});
    logical_written_ += offset;
    return !werr_.load(std::memory_order_relaxed);
  }

  // Direct pwrite  bypasses internal buffer.  For O_DIRECT, data and offset
  // must be ALIGN-aligned, and len should be a multiple of ALIGN (except for
  // the final write where we temporarily drop O_DIRECT).
  // If source data is not aligned, copies through the internal aligned buffer.
  bool direct_pwrite(const void * data, size_t len, off_t offset) {
    if (fd_ < 0) return false;

    // Check if source is aligned
    bool src_aligned = ((uintptr_t)data % ALIGN == 0);
    bool off_aligned = ((size_t)offset % ALIGN == 0);

    if (src_aligned && off_aligned) {
      // Fast path: source and offset are aligned
      return direct_pwrite_raw(data, len, offset);
    }

    // Slow path: copy through aligned buffer
    // Temporarily drop O_DIRECT and use regular pwrite
    int flags = fcntl(fd_, F_GETFL);
    if (flags >= 0) fcntl(fd_, F_SETFL, flags & ~O_DIRECT);
    bool ok = pwrite_all(static_cast<const char*>(data), len, offset);
    if (flags >= 0) fcntl(fd_, F_SETFL, flags);
    return ok;
  }

private:
  bool direct_pwrite_raw(const void * data, size_t len, off_t offset) {
    const char * p = static_cast<const char *>(data);

    // Aligned portion
    size_t aligned = (len / ALIGN) * ALIGN;
    if (aligned > 0) {
      if (!pwrite_all(p, aligned, offset)) return false;
      p += aligned;
      offset += (off_t)aligned;
      len -= aligned;
    }

    // Unaligned tail: temporarily drop O_DIRECT
    if (len > 0) {
      int flags = fcntl(fd_, F_GETFL);
      if (flags >= 0) fcntl(fd_, F_SETFL, flags & ~O_DIRECT);
      bool ok = pwrite_all(p, len, offset);
      if (flags >= 0) fcntl(fd_, F_SETFL, flags);  // restore O_DIRECT
      if (!ok) return false;
    }
    return true;
  }

private:
  static constexpr size_t BUF_CAP = 16 * 1024 * 1024;  // 16 MiB aligned buffer

  // Bare ::write loop, run on the background write thread (each op is timed in
  // writer_loop, so no timing here).
  bool write_raw(const void * data, size_t len) {
    const char * p = static_cast<const char *>(data);
    while (len > 0) {
      ssize_t w = ::write(fd_, p, len);
      if (w > 0) { p += w; len -= (size_t)w; }
      else if (w < 0) { if (errno == EINTR) continue; return false; }
      else return false;  // write() == 0 with len > 0: no progress possible
    }
    return true;
  }

  bool pwrite_all(const void * data, size_t len, off_t offset) {
    const char * p = static_cast<const char *>(data);
    while (len > 0) {
      ssize_t w = ::pwrite(fd_, p, len, offset);
      if (w > 0) {
        p += w;
        len -= (size_t)w;
        offset += w;
      } else if (w < 0) {
        if (errno == EINTR) continue;
        return false;
      } else {
        return false;  // pwrite() == 0 with len > 0: no progress possible
      }
    }
    return true;
  }

  // --- async write pipeline ---
  // The caller thread fills aligned buffers (bounce-copy + sparse zero-scan) and
  // a dedicated write thread ::writes them, so that CPU work overlaps the
  // O_DIRECT write instead of serializing on one thread (the win that brought
  // -d --tar extract to the device ceiling; this brings plain -d there too).
  static constexpr int NBUF = 4;            // 4 × 16 MiB = 64 MiB in flight
  int fd_ = -1;
  void * bufs_[NBUF] = {};
  int    cur_bi_ = -1;             // buffer the caller is currently filling
  size_t buf_used_ = 0;            // bytes in the current buffer
  uint64_t logical_written_ = 0;   // caller-side logical position (enqueued writes + seeks)
  uint64_t phys_written_ = 0;      // write-thread physical position (hole-punch offset)
  uint64_t preallocated_ = 0;
  enum OpKind { OP_WRITE, OP_SEEK };
  struct WOp { OpKind kind; int bi; size_t len; size_t tail; };
  std::mutex mx_;
  std::condition_variable cv_op_, cv_free_;
  std::deque<WOp> ops_;            // ordered fd ops for the write thread
  std::deque<int> free_;           // free buffer indices
  bool wt_done_ = false;
  std::atomic<bool> werr_{false};  // a write/seek failed on the write thread
  std::thread wt_;

  // Write thread: drains ops in order (writes + sparse seeks), keeping all fd
  // position changes sequential; returns each written buffer to the free list.
  void writer_loop() {
    for (;;) {
      WOp op;
      { std::unique_lock<std::mutex> lk(mx_);
        cv_op_.wait(lk, [&]{ return !ops_.empty() || wt_done_; });
        if (ops_.empty()) return;            // done and drained
        op = ops_.front(); ops_.pop_front(); }
      if (op.kind == OP_SEEK) {
#ifdef __linux__
        if (preallocated_ > 0 && op.len > 0)  // punch the skipped region back to a hole
          (void)::fallocate(fd_, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                            (off_t)phys_written_, (off_t)op.len);
#endif
        if (::lseek(fd_, (off_t)op.len, SEEK_CUR) < 0) werr_.store(true, std::memory_order_relaxed);
        phys_written_ += op.len;
      } else {
        char * b = static_cast<char*>(bufs_[op.bi]);
        size_t body = op.len - op.tail;       // aligned body (O_DIRECT) + optional sub-ALIGN tail
        uint64_t t0 = now_ns();
        bool ok = (body == 0) || write_raw(b, body);
        if (ok && op.tail > 0) {              // final unaligned tail: O_DIRECT can't do it
          int fl = ::fcntl(fd_, F_GETFL); if (fl >= 0) ::fcntl(fd_, F_SETFL, fl & ~O_DIRECT);
          ok = write_raw(b + body, op.tail);
          if (fl >= 0) ::fcntl(fd_, F_SETFL, fl);
        }
        g_odirect_write_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
        phys_written_ += op.len;
        if (!ok) werr_.store(true, std::memory_order_relaxed);
        { std::lock_guard<std::mutex> lk(mx_); free_.push_back(op.bi); }
        cv_free_.notify_one();
      }
    }
  }
  // Caller: take a free buffer to fill, blocking until the write thread frees
  // one (backpressure).  Returns -1 on write error.
  int acquire_free() {
    std::unique_lock<std::mutex> lk(mx_);
    cv_free_.wait(lk, [&]{ return !free_.empty() || werr_.load(std::memory_order_relaxed); });
    if (werr_.load(std::memory_order_relaxed)) return -1;
    int bi = free_.front(); free_.pop_front(); return bi;
  }
  void enqueue(WOp op) {
    { std::lock_guard<std::mutex> lk(mx_); ops_.push_back(op); }
    cv_op_.notify_one();
  }
};
#endif // _WIN32

/*======================================================================
 Auto chunk selection
======================================================================*/
// Choose a default chunk size for CPU compression.
// Regular files get larger chunks (fewer tasks, less overhead);
// pipes/streams use smaller chunks for lower latency.
// Resolve CPU thread count from Options.
//   -1 = all hardware threads (-T0 flag, like zstd)
//    0 = auto (capped at 96 for efficiency  diminishing returns beyond that)
//   >0 = user-specified exact count
static int resolve_cpu_threads(int opt_threads)
{
  unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  int io_reserved = (int)get_io_cores().size();  // cores pinned to reader/writer
  if (opt_threads == -1) {
    // -T0: use every thread except I/O-pinned cores
    return std::max(1, (int)hw - io_reserved);
  }
  if (opt_threads > 0)
    return opt_threads;                     // explicit -T N
  // Auto: use hw minus I/O cores, capped at 96
  int def = std::max(1, (int)hw - io_reserved);
  return std::min(def, 96);
}

// Query available host RAM (returns 0 if unknown)
static uint64_t get_available_ram_bytes()
{
#ifdef __linux__
  FILE * f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  uint64_t avail_kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::sscanf(line, "MemAvailable: %lu kB", &avail_kb) == 1) break;
  }
  std::fclose(f);
  return avail_kb * 1024ULL;
#else
  return 0;  // unknown on non-Linux
#endif
}

#ifdef HAVE_NVCOMP
/*======================================================================
 Pinned host-memory budget (--pinned auto)
 -----------------------------------------------------------------------
 Pinned (page-locked) memory speeds up cudaMemcpyAsync but is a global
 system resource: locked pages can't be swapped, and over-pinning starves
 other workloads (and on memory-pressured boxes, the kernel may refuse
 the allocation outright).

 AUTO mode rations pinning to <=50% of available system RAM, summed
 across ALL gpu workers (compress H2D + decompress D2H).  Streams that
 fit in the remaining budget get pinned; streams that don't (the
 unlucky ones) fall back to pageable memory.  Same applies if
 cudaHostAlloc fails for any reason — the worker silently uses
 pageable for that stream.

 PinMode::ON skips the budget check (user said "yes, pin everything").
 PinMode::OFF skips the entire path (no allocation attempt).

 The whole block is GPU-only: PinMode lives inside HAVE_NVCOMP in the
 Options struct, and these helpers are only ever called from compress/
 decompress nvcomp paths.
======================================================================*/
static std::atomic<uint64_t> g_pinned_bytes_reserved{0};
static std::atomic<uint64_t> g_pinned_bytes_budget{0};
static std::once_flag g_pinned_budget_init;

static void init_pinned_budget()
{
  uint64_t avail = get_available_ram_bytes();
  if (avail == 0) avail = 8ULL * 1024 * 1024 * 1024;  // assume 8 GiB if unknown
  g_pinned_bytes_budget.store(avail / 2, std::memory_order_relaxed);
}

// Try to reserve `want_bytes` from the global pinned-memory budget.
// Returns true if reserved (caller must release on failure or shutdown).
// Honors --pinned mode: ON always reserves, OFF always refuses, AUTO checks
// the budget atomically.
static bool try_reserve_pinned(uint64_t want_bytes, const Options & opt)
{
  if (opt.pin_mode == PinMode::OFF) return false;
  if (opt.pin_mode == PinMode::ON)  return true;  // user override
  std::call_once(g_pinned_budget_init, init_pinned_budget);
  uint64_t budget = g_pinned_bytes_budget.load(std::memory_order_relaxed);
  uint64_t cur = g_pinned_bytes_reserved.load(std::memory_order_acquire);
  while (true) {
    if (cur + want_bytes > budget) return false;
    if (g_pinned_bytes_reserved.compare_exchange_weak(
          cur, cur + want_bytes,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
    // cur was updated by the CAS, retry
  }
}

static void release_pinned(uint64_t bytes, const Options & opt)
{
  if (opt.pin_mode != PinMode::AUTO) return;  // ON/OFF didn't reserve
  g_pinned_bytes_reserved.fetch_sub(bytes, std::memory_order_release);
}
#endif // HAVE_NVCOMP

// Compute a throttle budget: how many frames can be buffered in ResultStore
// before producers must block waiting for the writer.
//
// v0.12.14: budget scales with the pipeline's actual producer parallelism
// (CPU threads + GPU streams * batch cap) times a slack factor, then clamped
// by available RAM.  This replaces the v0.12.13 hard 8 GiB byte ceiling,
// which was an arbitrary magic number tuned for our test systems.  The new
// formula scales naturally:
//
//   laptop  (8 CPU, no GPU)                ~32 frames   (floor; ~512 MiB)
//   workstn (24 CPU, 2*2*16 GPU pipeline)  ~352 frames  (~5.5 GiB)
//   server  (256 CPU, 8*4*64 GPU pipeline) ~9216 frames (~144 GiB)
//
// Rationale: every active producer needs a few frames ahead to stay busy
// across writer jitter, but there is no benefit to queuing hundreds of
// frames per producer — that just shifts work into RAM with no throughput
// gain.  SLACK_FACTOR=4 gives each producer ~4 frames of headroom, which
// matches typical pipeline-depth tuning in other staged designs.  The RAM
// cap (half of available) remains as a safety net.
//
// Floor is 32 frames so tiny-parallelism or very-large chunk_mib configs
// don't stall.  Fast-I/O systems are NOT slowed by the cap: the writer
// releases permits as fast as workers acquire, so nobody blocks.  Only
// slow-I/O systems see backpressure, which is exactly the desired
// behavior (keeps in-flight RAM bounded and producers in lockstep with
// disk).
static constexpr int THROTTLE_SLACK_FACTOR = 4;
static constexpr int THROTTLE_MIN_FRAMES = 32;

static int compute_throttle_budget(size_t frame_bytes,
                                   int pipeline_parallelism,
                                   int gpu_batch_floor,
                                   const Options & opt)
{
  size_t avail = get_available_ram_bytes();
  if (avail == 0) avail = 8ULL * 1024 * 1024 * 1024;  // assume 8 GiB if unknown
  frame_bytes = std::max<size_t>(frame_bytes, 1);
  pipeline_parallelism = std::max(pipeline_parallelism, 1);
  gpu_batch_floor = std::max(gpu_batch_floor, 0);

  int slack = opt.throttle_factor > 0 ? opt.throttle_factor : THROTTLE_SLACK_FACTOR;
  int pipeline_frames = pipeline_parallelism * slack;
  size_t ram_cap_frames = (avail / 2) / frame_bytes;
  // zstd-compat: --memlimit / --memory / -M# tightens the RAM cap on the
  // compress side to `mem_limit_mib` MiB total in-flight (matches the
  // spirit of zstd's memory cap flag, which is decompress-only in stock
  // zstd but extends naturally to compress for gzstd).
  if (opt.mem_limit_mib > 0) {
    size_t mem_cap_bytes = size_t(opt.mem_limit_mib) * ONE_MIB;
    size_t mem_cap_frames = std::max<size_t>(1, mem_cap_bytes / frame_bytes);
    if (mem_cap_frames < ram_cap_frames) ram_cap_frames = mem_cap_frames;
  }
  int ram_frames = (int)std::min<size_t>(ram_cap_frames, (size_t)INT_MAX);

  int frames;
  const char * source;
  if (opt.throttle_frames == 0) {
    // Explicit disable.  FrameThrottle ctor treats max<=0 as disabled
    // (acquire/release become no-ops, no lock taken).  This is for
    // benchmarking the no-throttle baseline.
    if (opt.verbosity >= V_VERBOSE) {
      vlog(V_VERBOSE, opt, "[THROTTLE] DISABLED (--throttle-frames=0)\n");
    }
    return 0;  // FrameThrottle(0) = disabled
  }
  if (opt.throttle_frames > 0) {
    frames = opt.throttle_frames;
    source = "user";
    // Guardrail: GPU workers greedy-acquire up to gpu_batch_cap permits per
    // stream.  If the budget is smaller than the sum of concurrent batches
    // that can be in flight, the first stream takes all permits, blocks
    // waiting for more, and the writer can't release any (nothing was
    // produced yet) — classic deadlock.  Clamp up to the safe floor.
    if (gpu_batch_floor > 0 && frames < gpu_batch_floor) {
      if (opt.verbosity >= V_ERROR) {
        std::ostringstream os;
        os << "warning: --throttle-frames=" << opt.throttle_frames
           << " is below the GPU batch floor (" << gpu_batch_floor
           << " = devices*streams*per_stream_batch); clamping to " << gpu_batch_floor
           << " to avoid deadlock.\n";
        vlog(V_ERROR, opt, os.str());
      }
      frames = gpu_batch_floor;
      source = "user+gpu-floor";
    }
  } else {
    frames = std::min(pipeline_frames, ram_frames);
    frames = std::max(frames, THROTTLE_MIN_FRAMES);
    if (gpu_batch_floor > 0 && frames < gpu_batch_floor) {
      // Defensive: the default formula should always clear this, but guard
      // against unusual configurations (tiny parallelism + huge batch_cap).
      frames = gpu_batch_floor;
    }
    source = (frames == THROTTLE_MIN_FRAMES && pipeline_frames < THROTTLE_MIN_FRAMES)
           ? "floor"
           : (pipeline_frames <= ram_frames ? "pipeline" : "ram");
  }

  // One-line summary at -v (informational).
  if (opt.verbosity >= V_VERBOSE) {
    char bud_s[32]; human_bytes(double(size_t(frames) * frame_bytes),
                                 bud_s, sizeof(bud_s));
    std::ostringstream os;
    os << "[THROTTLE] " << frames << " frames (" << bud_s
       << " in-flight max, source=" << source
       << ", parallelism=" << pipeline_parallelism
       << ", slack=" << slack << ")";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }
  // Detailed breakdown at -vv (debug).
  if (opt.verbosity >= V_DEBUG) {
    char ram_s[32]; human_bytes(double(avail), ram_s, sizeof(ram_s));
    std::ostringstream os;
    os << "[THROTTLE] detail: pipeline_cap=" << pipeline_frames
       << ", ram_cap=" << ram_frames
       << ", floor=" << THROTTLE_MIN_FRAMES
       << ", avail_ram=" << ram_s;
    vlog(V_DEBUG, opt, os.str() + "\n");
  }
  return frames;
}

// log_throttle_stats is defined below, after FrameThrottle.

/*======================================================================
 Ultra compression: window log helper
 -----------------------------------------------------------------------
 Zstd ultra levels (20-22) require explicitly setting ZSTD_c_windowLog
 to a value larger than the library default.  Without this, the library
 silently clamps the window to ~8 MiB, negating the purpose of ultra.

 Window sizes:
   level 20: windowLog 25 = 32 MiB
   level 21: windowLog 26 = 64 MiB
   level 22: windowLog 27 = 128 MiB

 The chunk size should be >= window size for ultra to be effective,
 since zstd can't use a window larger than the input chunk.
======================================================================*/

// Returns the required windowLog for the given compression level.
// Returns 0 if no special window is needed (level < 20 or !ultra).
static int ultra_window_log(int level, bool ultra)
{
  if (!ultra || level < 20) return 0;
  // level 20 -> 25, level 21 -> 26, level 22 -> 27
  return level + 5;
}

// Returns the minimum chunk size (in MiB) needed for ultra levels.
// The chunk must be >= window size for the compressor to use the full window.
// Returns 0 if no minimum is needed.
static size_t ultra_min_chunk_mib(int level, bool ultra)
{
  int wlog = ultra_window_log(level, ultra);
  if (wlog == 0) return 0;
  // windowLog N means window = 2^N bytes; convert to MiB
  return size_t(1) << (wlog - 20);  // 25->32, 26->64, 27->128
}

// Apply ultra window log to a CCtx.  Call after setting compression level.
// Returns the windowLog that was set, or 0 if none.
static int apply_ultra_cctx(ZSTD_CCtx * cctx, int level, bool ultra)
{
  int wlog = ultra_window_log(level, ultra);
  if (wlog > 0) {
    size_t st = ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, wlog);
    if (ZSTD_isError(st)) {
      // Non-fatal: log warning but continue with default window
      std::cerr << "gzstd: warning: ZSTD_c_windowLog(" << wlog
                << ") failed: " << ZSTD_getErrorName(st)
                << "  falling back to default window\n";
      return 0;
    }
  }
  return wlog;
}

// Pre-flight RAM check: estimate memory needed for N threads × chunk_mib.
// Each thread needs: input buffer (chunk) + output buffer (ZSTD_compressBound(chunk))
// plus queue overhead, result store, etc.
// If estimated usage exceeds 75% of available RAM, auto-reduce chunk_mib.
// Returns the (possibly reduced) chunk_mib.
static size_t check_ram_budget(int threads, size_t chunk_mib, const Options & opt)
{
  uint64_t avail = get_available_ram_bytes();
  if (avail == 0) return chunk_mib;  // can't determine  skip check

  size_t original_mib = chunk_mib;

  // Target: stay under 75% of available RAM
  uint64_t budget = avail * 75 / 100;

  while (chunk_mib >= 1) {
    size_t chunk_bytes = chunk_mib * ONE_MIB;
    // Per thread: input buf + output buf (compressBound ≈ chunk + chunk/256 + 64)
    // Ultra levels (20-22) require much larger CCtx internal state because
    // the window size grows to 32-128 MiB.  Each CCtx allocates roughly
    // 8 × windowSize for the hash tables + chain tables.
    size_t cctx_overhead = 0;
    {
      int wlog = ultra_window_log(opt.level, opt.ultra);
      if (wlog > 0) {
        size_t window_bytes = size_t(1) << wlog;
        cctx_overhead = window_bytes * 8;  // hash + chain tables
      }
    }
    size_t per_thread = chunk_bytes + chunk_bytes + (chunk_bytes >> 8) + 4096 + cctx_overhead;
    // Queue + result store can hold up to threads × 2 frames in flight
    size_t overhead = (size_t)threads * 2 * chunk_bytes;
    uint64_t est_total = (uint64_t)threads * per_thread + overhead;

    if (est_total <= budget) {
      if (chunk_mib < original_mib && opt.verbosity >= V_ERROR) {
        char est_s[64], avail_s[64];
        human_bytes(double(est_total), est_s, sizeof(est_s));
        human_bytes(double(avail), avail_s, sizeof(avail_s));
        std::cerr << "gzstd: note: reduced --chunk-size from " << original_mib
                  << " to " << chunk_mib << " MiB to fit in RAM ("
                  << est_s << " est, " << avail_s << " available)\n";
      }
      return chunk_mib;
    }
    // Halve and retry
    chunk_mib = chunk_mib / 2;
  }

  // Even 1 MiB doesn't fit  warn but proceed (let the OS handle it)
  if (opt.verbosity >= V_ERROR) {
    char avail_s[64];
    human_bytes(double(avail), avail_s, sizeof(avail_s));
    std::cerr << "gzstd: warning: very low RAM (" << avail_s
              << ")  compression may be slow or fail\n";
  }
  return 1;
}

/*======================================================================
 Queues and writer
======================================================================*/
// A chunk of work: compressed or uncompressed data with a sequence number.
// For decompression, decomp_size holds the expected decompressed size (from frame header).
// For mmap-backed compression, view_ptr/view_len point into the mapped region
// (zero-copy); data is empty.  Consumers use ptr()/len() uniformly.
// fwd: return a --direct-read zero-copy buffer to its pool (defined with DirectReadPool).
static void gz_direct_read_release(int slot);

struct Task {
  size_t seq = 0;
  std::vector<char> data;
  size_t decomp_size = 0;
  uint64_t out_off = 0;           // --keep-going: this frame's start offset in the decompressed
                                  //  output (prefix sum of decomp_size; set by the single reader)
  const char * view_ptr = nullptr;
  size_t       view_len = 0;
  int          direct_buf = -1;   // >=0: view_ptr aliases DirectReadPool slot; recycle on release


  const char * ptr() const { return view_ptr ? view_ptr : data.data(); }
  size_t       len() const { return view_ptr ? view_len : data.size(); }
  void release_input() {
    if (direct_buf >= 0) { gz_direct_read_release(direct_buf); direct_buf = -1; view_ptr = nullptr; view_len = 0; }
    else if (view_ptr)   { view_ptr = nullptr; view_len = 0; }
    else                 { std::vector<char>().swap(data); }
  }
};

// Thread-safe work queue for distributing chunks to compressor threads.
// Producer pushes chunks; workers pop one-at-a-time (CPU) or in batches (GPU).
// set_done() signals that no more work will be added.
// Nanosecond clock for performance instrumentation
static inline uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

/*======================================================================
 Performance counters for -vvv timing diagnosis
 -----------------------------------------------------------------------
 Thread-safe accumulation of wall-clock time spent in each pipeline phase.
 All times in nanoseconds (converted to seconds for display).

 USAGE: A stack-local PerfCounters is created in each top-level function
 (compress_cpu_mt, compress_nvcomp, decompress_cpu_mt, decompress_nvcomp)
 and g_perf is pointed to it.  Worker threads check `if (g_perf)` before
 recording.  After all workers join, print_summary() displays the results.

 IMPORTANT: GPU workers may have MULTIPLE completion paths (async poll
 and synchronous drain).  ALL paths must record to g_perf.  If counters
 show zero despite GPU tasks completing, check that every completion
 path includes g_perf recording  this has been a source of bugs.
======================================================================*/
struct PerfCounters {
  // CUDA init (tracks the maximum across all GPU workers  this is the
  // wall-clock time the main thread effectively waits for GPUs to be ready)
  std::atomic<uint64_t> cuda_init_max_ns{0};  // max single-device init time
  std::atomic<uint64_t> cuda_init_sum_ns{0};  // sum across all devices
  std::atomic<int>      cuda_init_count{0};   // number of devices

  // Reader / producer
  std::atomic<uint64_t> read_ns{0};        // wall time in fread / frame parsing
  std::atomic<uint64_t> read_bytes_total{0};

  // Queue wait
  std::atomic<uint64_t> queue_wait_ns{0};  // workers waiting for tasks
  std::atomic<uint64_t> queue_wait_count{0};

  // GPU H2D transfer
  std::atomic<uint64_t> h2d_ns{0};         // host-to-device transfer time
  std::atomic<uint64_t> h2d_bytes{0};
  std::atomic<uint64_t> h2d_count{0};

  // GPU compute (kernel)
  std::atomic<uint64_t> kernel_ns{0};      // nvCOMP kernel time (from CUDA events)
  std::atomic<uint64_t> kernel_count{0};

  // GPU D2H transfer
  std::atomic<uint64_t> d2h_ns{0};         // device-to-host transfer time
  std::atomic<uint64_t> d2h_bytes{0};
  std::atomic<uint64_t> d2h_count{0};

  // GPU total batch time (H2D + kernel + sync + D2H readback)
  std::atomic<uint64_t> gpu_batch_ns{0};
  std::atomic<uint64_t> gpu_batch_count{0};

  // CPU worker compute
  std::atomic<uint64_t> cpu_compute_ns{0};
  std::atomic<uint64_t> cpu_compute_count{0};
  std::atomic<uint64_t> cpu_compute_bytes{0};

  // Writer
  std::atomic<uint64_t> write_ns{0};       // wall time in fwrite
  std::atomic<uint64_t> write_bytes_total{0};
  std::atomic<uint64_t> writer_wait_ns{0}; // writer waiting for in-order results
  std::atomic<uint64_t> writer_wait_count{0};

  // Result store contention
  std::atomic<uint64_t> result_lock_ns{0}; // time holding result store mutex

  // Frames out of order (writer had to wait)
  std::atomic<uint64_t> out_of_order_waits{0};

  // Scheduler
  std::atomic<uint64_t> sched_cpu_tasks{0};
  std::atomic<uint64_t> sched_gpu_tasks{0};

  void print_summary(const char * label) const {
    auto ns_to_s = [](uint64_t ns) { return double(ns) / 1e9; };
    auto ns_to_ms = [](uint64_t ns) { return double(ns) / 1e6; };
    auto bytes_to_gib = [](uint64_t b) { return double(b) / (1024.0*1024*1024); };
    auto rate_gibs = [](uint64_t bytes, uint64_t ns) -> double {
      return (ns > 0) ? double(bytes) / (1024.0*1024*1024) / (double(ns)/1e9) : 0;
    };

    // Color tokens — empty strings when stderr is not a TTY.
    const char * RST = g_color_stderr ? "\033[0m"    : "";
    const char * HDR = g_color_stderr ? "\033[1;97m" : "";  // bold bright white  — header
    const char * LBL = g_color_stderr ? "\033[1;33m" : "";  // bold yellow        — row labels
    const char * TIM = g_color_stderr ? "\033[1;96m" : "";  // bold bright cyan   — seconds
    const char * SIZ = g_color_stderr ? "\033[1;94m" : "";  // bold bright blue   — GiB
    const char * THR = g_color_stderr ? "\033[1;92m" : "";  // bold bright green  — GiB/s
    const char * CNT = g_color_stderr ? "\033[95m"   : "";  // bright magenta     — counts
    const char * DIM = g_color_stderr ? "\033[2m"    : "";  // dim                — separators

    fprintf(stderr, "\n");
    fprintf(stderr, "  %sPERFORMANCE BREAKDOWN:%s %s%s%s\n",
            HDR, RST, LBL, label, RST);
    fprintf(stderr, "\n");

    if (cuda_init_count > 0)
      fprintf(stderr, "  %sCUDA init:%s        %s%8.3f s%s  "
                      "(%s%d%s devices, %s%.3f s%s each avg)\n",
              LBL, RST,
              TIM, ns_to_s(cuda_init_max_ns), RST,
              CNT, cuda_init_count.load(), RST,
              TIM, ns_to_s(cuda_init_sum_ns) / cuda_init_count, RST);

    fprintf(stderr, "  %s──────────────────────────────────────────────────────%s\n", DIM, RST);
    fprintf(stderr, "  %sReader:%s           %s%8.3f s%s  "
                    "(%s%6.2f GiB%s, %s%5.2f GiB/s%s)\n",
            LBL, RST,
            TIM, ns_to_s(read_ns), RST,
            SIZ, bytes_to_gib(read_bytes_total), RST,
            THR, rate_gibs(read_bytes_total, read_ns), RST);

    if (h2d_count > 0 || sched_gpu_tasks > 0)
      fprintf(stderr, "  %sH2D transfers:%s    %s%8.3f s%s  "
                      "(%s%6.2f GiB%s, %s%5.2f GiB/s%s) [%s%llu%s]\n",
              LBL, RST,
              TIM, ns_to_s(h2d_ns), RST,
              SIZ, bytes_to_gib(h2d_bytes), RST,
              THR, rate_gibs(h2d_bytes, h2d_ns), RST,
              CNT, (unsigned long long)h2d_count.load(), RST);

    if (kernel_count > 0 || sched_gpu_tasks > 0)
      fprintf(stderr, "  %sGPU kernel:%s       %s%8.3f s%s  "
                      "(%s%llu%s batches, %s%5.1f ms%s/batch)\n",
              LBL, RST,
              TIM, ns_to_s(kernel_ns), RST,
              CNT, (unsigned long long)kernel_count.load(), RST,
              TIM, (kernel_count > 0) ? ns_to_ms(kernel_ns) / kernel_count : 0.0, RST);

    if (d2h_count > 0 || sched_gpu_tasks > 0)
      fprintf(stderr, "  %sD2H transfers:%s    %s%8.3f s%s  "
                      "(%s%6.2f GiB%s, %s%5.2f GiB/s%s) [%s%llu%s]\n",
              LBL, RST,
              TIM, ns_to_s(d2h_ns), RST,
              SIZ, bytes_to_gib(d2h_bytes), RST,
              THR, rate_gibs(d2h_bytes, d2h_ns), RST,
              CNT, (unsigned long long)d2h_count.load(), RST);

    if (gpu_batch_count > 0 || sched_gpu_tasks > 0)
      fprintf(stderr, "  %sGPU batch total:%s  %s%8.3f s%s  "
                      "(%s%llu%s batches, %s%5.1f ms%s/batch)\n",
              LBL, RST,
              TIM, ns_to_s(gpu_batch_ns), RST,
              CNT, (unsigned long long)gpu_batch_count.load(), RST,
              TIM, (gpu_batch_count > 0) ? ns_to_ms(gpu_batch_ns) / gpu_batch_count : 0.0, RST);

    if (cpu_compute_count > 0)
      fprintf(stderr, "  %sCPU compute:%s      %s%8.3f s%s  "
                      "(%s%6.2f GiB%s, %s%llu%s chunks)\n",
              LBL, RST,
              TIM, ns_to_s(cpu_compute_ns), RST,
              SIZ, bytes_to_gib(cpu_compute_bytes), RST,
              CNT, (unsigned long long)cpu_compute_count.load(), RST);

    fprintf(stderr, "  %s──────────────────────────────────────────────────────%s\n", DIM, RST);
    fprintf(stderr, "  %sQueue wait:%s       %s%8.3f s%s  (%s%llu%s waits)\n",
            LBL, RST,
            TIM, ns_to_s(queue_wait_ns), RST,
            CNT, (unsigned long long)queue_wait_count.load(), RST);
    fprintf(stderr, "  %sWriter wait:%s      %s%8.3f s%s  (%s%llu%s waits)\n",
            LBL, RST,
            TIM, ns_to_s(writer_wait_ns), RST,
            CNT, (unsigned long long)writer_wait_count.load(), RST);
    fprintf(stderr, "  %sWriter I/O:%s       %s%8.3f s%s  "
                    "(%s%6.2f GiB%s, %s%5.2f GiB/s%s)\n",
            LBL, RST,
            TIM, ns_to_s(write_ns), RST,
            SIZ, bytes_to_gib(write_bytes_total), RST,
            THR, rate_gibs(write_bytes_total, write_ns), RST);
    if (result_lock_ns > 0)
      fprintf(stderr, "  %sResult lock:%s      %s%8.3f s%s\n",
              LBL, RST,
              TIM, ns_to_s(result_lock_ns), RST);

    fprintf(stderr, "  %s──────────────────────────────────────────────────────%s\n", DIM, RST);
    fprintf(stderr, "  %sScheduler:%s  CPU %s%llu%s tasks, GPU %s%llu%s tasks\n",
            LBL, RST,
            CNT, (unsigned long long)sched_cpu_tasks.load(), RST,
            CNT, (unsigned long long)sched_gpu_tasks.load(), RST);

    fprintf(stderr, "\n");
  }
};

// Global perf counters  only non-null when -vvv is active.
// All instrumentation checks `if (g_perf)` before recording.
// Set once in the top-level compress/decompress function BEFORE worker
// threads launch, and cleared AFTER all workers join.  Workers read this
// pointer from any thread  it's set-once-read-many so no synchronization
// needed beyond the thread-launch happens-before relationship.
static PerfCounters * g_perf = nullptr;

// Global DirectWriter pointer  set when O_DIRECT output is active.
// Writer thread checks this to decide between DirectWriter and fwrite.
#ifndef _WIN32
static DirectWriter * g_direct_writer = nullptr;
#endif

// Set when ANY GPU fault is detected during compression.  A faulting GPU is an
// unreliable narrator — we cannot trust which frames it already "finished" — so
// the entire GPU/hybrid attempt is abandoned and the compress driver rebuilds
// the whole output CPU-only from the (still-present) input.  See the compress
// restart path in the driver.
static std::atomic<bool> g_gpu_failed_restart{false};

// Set the instant a GPU faults mid-compress (distinct from g_gpu_failed_restart,
// which --verify also sets AFTER a pass to request a rebuild).  This one ABORTS
// the in-flight compress pass: the producer stops reading, CPU/GPU workers exit
// without compressing more, and the writer bails its in-order wait WITHOUT the
// "writer stuck" watchdog firing.  The whole (doomed) output is discarded and the
// driver rebuilds CPU-only — so there is no point finishing it, which is why the
// old rescue/fallback machinery (which re-compressed the tail only to throw it
// away) was deleted.  GPU-fault only: never set on decompress (a faulted GPU
// there is finished on CPU and the output is KEPT) nor by --verify.
static std::atomic<bool> g_gpu_aborted{false};

// --verify: a background decompress-verify worker found a frame whose compressed
// bytes do NOT round-trip (XXH64 mismatch / decode error).  Set alongside
// g_gpu_failed_restart so the driver's existing discard-and-rebuild path fires;
// this flag tells it the trigger was a verify failure (not a GPU fault) so it
// uses the data-error exit code and message on the unrecoverable (pipe) path.
// g_verify_failed_seq records the first failing frame's sequence number for the
// per-attempt log (a seq that repeats across rebuilds = a deterministic fault).
static std::atomic<bool>    g_verify_failed{false};
static std::atomic<int64_t> g_verify_failed_seq{-1};

// Test-only fault injection (read once from $GZSTD_DEBUG_FAIL_GPU_AFTER): make a
// GPU compress worker throw a simulated fault after it has delivered this many
// frames, to exercise the CPU-only restart path deterministically (real GPU
// faults are intermittent).  -1 (default, env unset) = disabled.
static int64_t g_debug_fail_gpu_after = -1;

// Test-only corruption injection (read once from $GZSTD_DEBUG_CORRUPT_FRAME):
// flip a byte in the compressed payload of the frame with this sequence number
// just before it reaches the writer, so --verify must catch it.  Exercises the
// verify→rebuild path deterministically.  -1 (default, env unset) = disabled.
// By default it fires ONCE (so the CPU rebuild then verifies clean); set
// $GZSTD_DEBUG_CORRUPT_PERSIST=1 to corrupt on EVERY pass (to exercise the
// --verify-retries give-up path).
static int64_t g_debug_corrupt_frame   = -1;
static bool    g_debug_corrupt_persist = false;

// --keep-going: collects every frame that failed to fully verify/decode during a
// recovery decompress, so the driver can report which output ranges (and, with
// --tar, which files) are damaged and pick the right exit code.  Only touched
// when opt.keep_going (which also forces the single-threaded, cpu-only path).
struct DamageReport {
  struct Range {
    uint64_t off;        // start offset in the decompressed output
    uint64_t produced;   // bytes actually recovered for this frame
    uint64_t declared;   // bytes the frame header said it should hold
  };
  std::mutex            m;
  std::vector<Range>    ranges;                  // one per damaged frame
  std::atomic<bool>     any{false};              // >=1 frame damaged
  std::atomic<bool>     incomplete{false};       // >=1 frame produced < declared (data lost)

  // Framing break: the reader lost frame sync (a corrupt frame header / declared
  // size overrun / truncation), so recovery stopped after the last good frame.
  // Everything past break_off is unrecoverable without a magic-scan resync, which
  // we deliberately don't do — so the recovery is INCOMPLETE (exit 7).
  std::atomic<bool>     framing_break{false};
  uint64_t              break_after_frames = 0;  // frames recovered before the break
  uint64_t              break_off = 0;           // bytes recovered before the break
  std::string           break_reason;

  void record(uint64_t off, uint64_t produced, uint64_t declared) {
    { std::lock_guard<std::mutex> lk(m); ranges.push_back({off, produced, declared}); }
    any.store(true, std::memory_order_relaxed);
    if (produced < declared) incomplete.store(true, std::memory_order_relaxed);
  }
  void set_framing_break(uint64_t after_frames, uint64_t off, const std::string & reason) {
    { std::lock_guard<std::mutex> lk(m); break_after_frames = after_frames; break_off = off; break_reason = reason; }
    framing_break.store(true, std::memory_order_relaxed);
    incomplete.store(true, std::memory_order_relaxed);   // data past the break is lost
  }
  bool damaged() const {   // anything to report?
    return any.load(std::memory_order_relaxed) || framing_break.load(std::memory_order_relaxed);
  }
  void reset() {
    std::lock_guard<std::mutex> lk(m);
    ranges.clear();
    any.store(false, std::memory_order_relaxed);
    incomplete.store(false, std::memory_order_relaxed);
    framing_break.store(false, std::memory_order_relaxed);
    break_after_frames = 0; break_off = 0; break_reason.clear();
  }
};
static DamageReport g_damage;

// --keep-going: print the post-run damage summary and pick the verdict wording.
// Plain output → byte ranges; --tar member mapping is added in report's caller
// chain (the extractor records member ranges).  Workers already printed a
// per-frame WARNING; this is the rollup.
static void report_keep_going_damage(const Options & opt) {
  std::lock_guard<std::mutex> lk(g_damage.m);
  const size_t n = g_damage.ranges.size();
  std::ostringstream os;
  if (n > 0)
    os << "WARNING: --keep-going recovered past " << n
       << " damaged frame(s); the output is NOT verified.\n";
  else
    os << "WARNING: --keep-going recovered the readable portion; the output is incomplete.\n";
  const size_t SHOW = 20;   // cap detail so heavily-rotted input doesn't flood
  for (size_t i = 0; i < n && i < SHOW; ++i) {
    const auto & r = g_damage.ranges[i];
    os << "  output bytes " << r.off << ".." << (r.off + r.declared);
    if (r.produced >= r.declared)
      os << ": checksum mismatch (content present, UNVERIFIED)\n";
    else
      os << ": could not decode — " << r.produced << " of " << r.declared
         << " bytes recovered\n";
  }
  if (n > SHOW) os << "  ... and " << (n - SHOW) << " more\n";
  if (g_damage.framing_break.load(std::memory_order_relaxed))
    os << "  RECOVERY STOPPED after " << g_damage.break_after_frames << " frame(s), "
       << g_damage.break_off << " bytes: " << g_damage.break_reason << "\n";
  if (g_damage.incomplete.load(std::memory_order_relaxed))
    os << "  result: INCOMPLETE — some data could not be decoded (exit "
       << EXIT_DECOMP_INCOMPLETE << ").\n";
  else
    os << "  result: complete but UNVERIFIED — one or more bytes may be wrong (exit "
       << EXIT_DECOMP_UNVERIFIED << ").\n";
  vlog(V_ERROR, opt, os.str());
}

// --keep-going + --tar: each extracted regular file's data range in the tar byte
// stream, recorded by the Extractor so the post-extraction report can name the
// file(s) overlapping a damaged frame.  Because the recovery decode emits every
// frame at its full declared length, the stream never desyncs, so these offsets
// line up exactly with the damage offsets regardless of error type.
struct MemberRanges {
  struct M { std::string path; uint64_t start; uint64_t size; };
  std::mutex          m;
  std::vector<M>      v;
  void add(const std::string & p, uint64_t start, uint64_t size) {
    std::lock_guard<std::mutex> lk(m); v.push_back({p, start, size});
  }
  void reset() { std::lock_guard<std::mutex> lk(m); v.clear(); }
};
static MemberRanges g_member_ranges;

// --keep-going + --tar: name the file(s) whose data overlaps a damaged frame and
// pick the verdict.  Returns the recovery exit code (EXIT_DECOMP_*), or EXIT_OK
// if nothing was damaged.
static int report_keep_going_damage_tar(const Options & opt) {
  if (!g_damage.damaged()) return EXIT_OK;
  std::vector<DamageReport::Range> dmg;
  { std::lock_guard<std::mutex> lk(g_damage.m); dmg = g_damage.ranges; }
  std::vector<MemberRanges::M> mem;
  { std::lock_guard<std::mutex> lk(g_member_ranges.m); mem = g_member_ranges.v; }

  // Worst overlapping severity per file (2 = data lost, 1 = unverified).
  std::vector<std::pair<std::string,int>> hits;
  for (const auto & f : mem) {
    int sev = 0;
    for (const auto & d : dmg) {
      const uint64_t a0 = f.start, a1 = f.start + f.size;       // member data range
      const uint64_t b0 = d.off,   b1 = d.off + d.declared;     // damaged frame range
      if (a0 < b1 && b0 < a1)                                   // ranges overlap
        sev = std::max(sev, (d.produced < d.declared) ? 2 : 1);
    }
    if (sev) hits.push_back({f.path, sev});
  }

  std::ostringstream os;
  os << "WARNING: --keep-going: ";
  if (!dmg.empty()) os << dmg.size() << " damaged frame(s); ";
  os << "the extracted files are NOT verified.\n";
  if (!dmg.empty() && hits.empty()) {
    os << "  (damage fell in tar metadata/padding, not within any file's data)\n";
  } else if (!hits.empty()) {
    os << "  damaged file(s):\n";
    const size_t SHOW = 50;
    for (size_t i = 0; i < hits.size() && i < SHOW; ++i)
      os << "    " << hits[i].first
         << (hits[i].second == 2 ? "  (INCOMPLETE — some data could not be decoded)"
                                 : "  (UNVERIFIED — one or more bytes may be wrong)") << "\n";
    if (hits.size() > SHOW) os << "    ... and " << (hits.size() - SHOW) << " more\n";
  }
  if (g_damage.framing_break.load(std::memory_order_relaxed))
    os << "  RECOVERY STOPPED after " << g_damage.break_after_frames << " frame(s), "
       << g_damage.break_off << " bytes: " << g_damage.break_reason << "\n"
       << "  files at/after that point were not extracted or are truncated.\n";
  os << "  tip: extract into a fresh directory with -C DIR to keep the originals.\n";
  vlog(V_ERROR, opt, os.str());
  return g_damage.incomplete.load(std::memory_order_relaxed)
       ? EXIT_DECOMP_INCOMPLETE : EXIT_DECOMP_UNVERIFIED;
}

// Fixed pool of page-aligned buffers for the zero-copy --direct-read path.
// The single O_DIRECT reader preads each chunk straight into a pooled buffer and
// hands it to a worker as a Task view (no per-chunk 16 MiB copy); the worker
// recycles it via release_input() -> gz_direct_read_release().  acquire() blocks
// when every buffer is in flight, so the pool doubles as producer backpressure
// (replacing the queue byte-cap, which is a no-op for zero-copy view tasks).
// One reader stream is deliberate: concurrent O_DIRECT reads *contend* on this
// class of NVMe (measured: 1 stream 4.5 GB/s, 4 streams 3.0 GB/s aggregate), so
// the win is a single uninterrupted stream that never stalls to memcpy.
#ifndef _WIN32
class DirectReadPool {
public:
  // want_thp: the MADV_HUGEPAGE + sparse-prefault dance below was built to
  // merge O_DIRECT DMA segments, and on a modern kernel it also speeds the
  // buffered pooled reader (huge-page-backed copy_to_user: 3.10 s vs 3.85 s
  // plain on the Gen3 workstation).  On pre-6.4 kernels it is poison for
  // buffered mode: THP doesn't engage, the sparse prefault leaves 1 page per
  // 2 MiB mapped, and every copy faults into a THP-eligible VMA (compaction
  // attempts on a fragmented box) — measured 2.14 GiB/s vs plain fread's 9.6
  // on the 256-core server.  Callers pass want_thp = o_direct (DMA always
  // needs it) || kernel >= 6.4.  The !want_thp path takes MADV_NOHUGEPAGE
  // explicitly (system THP=always must not re-enable the pathology) and
  // prefaults everything with a memset — deterministic, zero faults during
  // reads, no THP anywhere.
  bool init(size_t buf_size, size_t n_bufs, bool want_thp) {
    buf_size_ = buf_size;
    // Allocate the WHOLE pool as one big region (not n_bufs small ones) and slice it.
    // This is what cuts the O_DIRECT DMA segment count: many small posix_memalign
    // buffers land on the fragmented heap (16 MiB < our M_MMAP_THRESHOLD), so each
    // 16 MiB read is shattered by the driver's max_segments limit into ~340 KiB
    // requests; one large region (> the threshold) is a fresh mmap whose pages fault
    // in as long physically-contiguous runs on an unfragmented box, merging into a
    // few big segments so a pread reaches the device's max request size (measured on
    // the 256-core server: rareq-sz 340 KiB -> ~1230 KiB, ~max_sectors_kb).  The
    // 2 MiB alignment + MADV_HUGEPAGE additionally request THP backing where it's
    // healthy (belt-and-suspenders; measured AnonHugePages=0 on the 5.15 box, so the
    // win there is the contiguous allocation, not THP).  No memset: the first pread
    // faults each buffer, and workers read only view_len (== got).
    const size_t HP = 2 * 1024 * 1024;
    stride_ = ((buf_size + HP - 1) / HP) * HP;   // 2 MiB-aligned stride ⇒ every slice is 2 MiB-aligned
    const size_t total = stride_ * n_bufs;
    if (posix_memalign(&base_, HP, total) != 0 || !base_) { base_ = nullptr; return false; }
    if (want_thp) {
#ifdef MADV_HUGEPAGE
      (void)madvise(base_, total, MADV_HUGEPAGE);  // best-effort; no-op if THP is disabled
#endif
      // Pre-fault one byte per 2 MiB page via the normal write-fault path, which DOES
      // back a 2 MiB-aligned MADV_HUGEPAGE region with a huge page.  Necessary because
      // O_DIRECT's get_user_pages would otherwise fault the buffer in as 4 KiB pages
      // (THP never engages) and we'd keep the fragmented ~340 KiB-request behaviour.
      volatile char * vp = static_cast<volatile char *>(base_);
      for (size_t o = 0; o < total; o += HP) vp[o] = 0;
    } else {
      // Old-kernel buffered mode: plain pages, prefaulted (see init comment).
#ifdef MADV_NOHUGEPAGE
      (void)madvise(base_, total, MADV_NOHUGEPAGE);
#endif
      // Prefault capped at 4 GiB: the GPU-path pool can reach tens of GiB
      // and a full memset costs seconds of startup.  Beyond the cap, plain
      // 4 KiB first-touch faults amortize fine — it was the THP allocation
      // attempts that were toxic, and MADV_NOHUGEPAGE prevents those.
      std::memset(base_, 0, std::min<size_t>(total, size_t(4) << 30));
    }
    bufs_.reserve(n_bufs);
    free_.reserve(n_bufs);
    for (size_t i = 0; i < n_bufs; ++i) {
      bufs_.push_back(static_cast<char *>(base_) + i * stride_);
      free_.push_back((int)i);
    }
    return true;
  }
  // Blocks until a buffer is free; returns its slot index.
  int acquire() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&]{ return !free_.empty(); });
    int s = free_.back(); free_.pop_back();
    return s;
  }
  void release(int slot) {
    if (slot < 0) return;
    { std::lock_guard<std::mutex> lk(m_); free_.push_back(slot); }
    cv_.notify_one();
  }
  char * buf(int slot) { return bufs_[(size_t)slot]; }
  ~DirectReadPool() { if (base_) free(base_); }
private:
  void *              base_ = nullptr;   // single backing region (2 MiB-aligned, THP-advised)
  size_t              stride_ = 0;       // per-buffer stride (2 MiB-aligned)
  std::vector<char *> bufs_;
  std::vector<int>    free_;
  std::mutex          m_;
  std::condition_variable cv_;
  size_t              buf_size_ = 0;
};

// Set while a zero-copy --direct-read run is active (single producer sets it
// before workers start, clears it after they join); workers only ever call
// release() through it, which is internally locked.
static DirectReadPool * g_direct_read_pool = nullptr;
#endif

static void gz_direct_read_release(int slot) {
#ifndef _WIN32
  if (g_direct_read_pool) g_direct_read_pool->release(slot);
#else
  (void)slot;
#endif
}

class TaskQueue {
public:
  void push(Task && t)
  {
    std::unique_lock<std::mutex> lk(m_);
    // Bounded-queue backpressure (ROADMAP 7.8): block the producer so a consumer
    // slower than the reader can't buffer the entire input in RAM.  Two
    // independent caps, either may be 0 = off:
    //   max_depth_ — frame count (enough frames buffered to keep consumers fed)
    //   max_bytes_ — owned heap bytes (bounds RAM regardless of compressibility;
    //                a frame-count cap holds ~4× the RAM on incompressible input).
    // The `!q_.empty()` guard on the byte cap guarantees progress even when a
    // single frame exceeds the whole budget.  re_enqueue (push_front) bypasses
    // both, so it never blocks.
    while (!done_ &&
           ((max_depth_ > 0 && q_.size() >= max_depth_) ||
            (max_bytes_ > 0 && !q_.empty() && queued_bytes_ >= max_bytes_)))
      space_cv_.wait(lk);
    // done_ set mid-stream means a shutdown is in progress (GPU-fault abort): drop
    // the task instead of buffering it — the consumers have exited and the whole
    // output is being discarded, so buffering the rest of the input would only
    // grow memory unbounded.  Normal end-of-input calls set_done() AFTER the last
    // push, so this never drops a wanted frame.
    if (done_) return;
    queued_bytes_ += t.data.size();
    q_.push_back(std::move(t));
    ++total_tasks_;
    cv_.notify_all();      // wake all GPU workers waiting in pop_batch_greedy
    cpu_cv_.notify_one();  // wake one CPU worker waiting in pop_one_cpu
  }

  // Cap the number of queued (read-but-not-popped) frames for producer
  // backpressure.  0 = unbounded (default).  Set before the producer starts.
  void set_max_depth(size_t n)
  {
    std::lock_guard<std::mutex> lk(m_);
    max_depth_ = n;
  }

  // Cap the total owned (heap) bytes of queued frames for producer backpressure.
  // Complements set_max_depth: bounds RAM regardless of compressibility (a frame
  // cap holds far more RAM on incompressible input).  0 = unbounded.
  void set_max_bytes(size_t n)
  {
    std::lock_guard<std::mutex> lk(m_);
    max_bytes_ = n;
  }

  // Re-enqueue tasks that were popped but never processed (e.g., GPU trivial-skip,
  // VRAM failure).  Pushes to the FRONT of the deque in original sequence order.
  //
  // Why front, not back: the FrameThrottle is deadlock-free only when the FIFO
  // invariant holds — "the frame the writer needs next is always among the oldest
  // in-flight frames."  push_back breaks this: the writer needs frame 0, but it's
  // behind 1200+ higher-sequence frames.  Workers process those higher frames,
  // consuming all permits, and frame 0 can never be popped — classic circular wait.
  // push_front restores the invariant: re-enqueued frames (lowest seq) are popped
  // and processed first, so the writer can make progress and release permits.
  void re_enqueue(std::vector<Task> & batch)
  {
    if (batch.empty()) return;
    std::unique_lock<std::mutex> lk(m_);
    for (auto it = batch.rbegin(); it != batch.rend(); ++it) {
      queued_bytes_ += it->data.size();
      q_.push_front(std::move(*it));
    }
    batch.clear();
    cv_.notify_all();
    cpu_cv_.notify_one();
  }

  // Pop up to max_n tasks at once (used by GPU workers to fill a batch).
  // Returns false when the queue is drained (empty + done).
  // Pop up to max_n tasks at once (used by GPU workers to fill a batch).
  // If min_n > 0, waits until at least min_n tasks are queued (or producer done).
  bool pop_batch(size_t max_n, std::vector<Task> & out, size_t min_n = 0)
  {
    std::unique_lock<std::mutex> lk(m_);
    // Wait for at least min_n items (or done signal)
    while ((q_.size() < min_n && !done_) || (q_.empty() && !done_))
      cv_.wait(lk);
    if (q_.empty() && done_) return false;

    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i = 0; i < take; ++i) {
      out.push_back(take_front_locked());
    }
    return true;
  }

  // Greedy pop: wait until the queue has at least min_n items (default: max_n)
  // OR producer is done, then take everything available up to max_n.
  //
  // When min_n == max_n (default): maximizes batch size for GPU kernels where
  // per-launch overhead is expensive.  Used for compress.
  //
  // When min_n < max_n: starts processing sooner with partial batches.  Used
  // for decompress where 8 GPUs blocking for full batches serializes the
  // pipeline — each GPU drains the queue and others wait for the reader to
  // refill.  With min_n=1, GPUs grab whatever is available immediately.
  bool pop_batch_greedy(size_t max_n, std::vector<Task> & out, size_t min_n = 0)
  {
    if (min_n == 0) min_n = max_n;  // default: wait for full batch
    std::unique_lock<std::mutex> lk(m_);
    while (q_.size() < min_n && !done_)
      cv_.wait(lk);
    if (q_.empty() && done_) return false;

    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i = 0; i < take; ++i) {
      out.push_back(take_front_locked());
    }
    pop_signal_locked();
    return true;
  }

  // Pop a single task (used by CPU workers).
  bool pop_one(Task & t)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cpu_cv_.wait(lk);
    if (q_.empty() && done_) return false;

    t = take_front_locked();
    return true;
  }

  // Block until the queue has at least one task or is done.
  // Returns false when drained (caller should exit).
  //
  // Waits on cpu_cv_ (CPU channel) rather than cv_ (GPU batch channel) so
  // that push()'s targeted cpu_cv_.notify_one() wakes exactly one sleeping
  // CPU worker per frame.  Previously waited on cv_, which push notifies
  // via notify_all for GPU batch waiters — so every CPU worker in the pool
  // woke on every push, 21 of 22 then found the queue drained by the
  // winner and went back to sleep.  At ~8000 frames × 22 waiters that's
  // ~176k spurious wakeups per run; eliminating them collapses
  // worker-to-worker contention on m_ and removes the bulk of the CPU-side
  // variance at high thread counts.  set_done() notifies both CVs so
  // shutdown still wakes everyone.
  bool wait_for_work()
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cpu_cv_.wait(lk);
    return !(q_.empty() && done_);
  }

  // Non-blocking pop: returns 1 if got a task, 0 if empty, -1 if drained.
  int try_pop_one(Task & t)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty() && done_) return -1;
    if (q_.empty()) return 0;
    t = take_front_locked();
    pop_signal_locked();
    return 1;
  }

  // Non-blocking batch pop: takes up to max_n frames if available.
  // Returns number of frames taken (0 if queue was empty).
  // Never blocks  used by CPU workers in hybrid mode to avoid
  // competing with GPU workers for queue CV wakeups.
  size_t try_pop_batch(size_t max_n, std::vector<Task> & out)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty()) return 0;
    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i = 0; i < take; ++i) {
      out.push_back(take_front_locked());
    }
    return take;
  }

  // Non-blocking batch pop with minimum threshold.
  // Returns 0 (takes nothing) if queue has fewer than min_n items AND
  // producer is still running.  Once producer is done, takes whatever
  // remains (even if < min_n) so no frames are stranded.
  // This implements "--cpu-batch=N" semantics: don't give CPU any work
  // unless N frames are available, except at end-of-file.
  size_t try_pop_batch_min(size_t max_n, size_t min_n, std::vector<Task> & out)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty()) return 0;
    // If below threshold and producer still running, don't take anything
    if (q_.size() < min_n && !done_) return 0;
    // Take up to max_n
    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i = 0; i < take; ++i) {
      out.push_back(take_front_locked());
    }
    return take;
  }

  // Peek at the compression ratio of the front task without popping.
  // Returns the ratio (compressed/decompressed) or -1.0 if queue is empty.
  // Used by CPU decompress workers to detect trivially-compressed frames
  // (ratio < 2%) that are faster to decompress on CPU than GPU because
  // CPU avoids the PCIe D2H transfer overhead for near-zero output.
  double peek_front_ratio()
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty()) return -1.0;
    const Task & front = q_.front();
    if (front.decomp_size == 0) return -1.0;
    return double(front.len()) / double(front.decomp_size);
  }

  // Signal that no more tasks will be pushed (producer is finished).
  void set_done()
  {
    std::unique_lock<std::mutex> lk(m_);
    done_ = true;
    cv_.notify_all();
    cpu_cv_.notify_all();
    space_cv_.notify_all();  // wake a producer blocked on a full bounded queue
  }

  size_t total_tasks() const { return total_tasks_; }

  bool drained()
  {
    std::unique_lock<std::mutex> lk(m_);
    return q_.empty() && done_;
  }

  size_t size()
  {
    std::unique_lock<std::mutex> lk(m_);
    return q_.size();
  }

  // Wake ALL CPU workers blocking in wait_for_cpu().
  // Used for rare lifecycle events (set_done, set_gpu_ready, GPU worker exit)
  // where all CPUs must re-evaluate their state.
  void notify_cpu_waiters()
  {
    cpu_cv_.notify_all();
  }

  // Wake ONE CPU worker.  Used by the high-frequency gpu_got_data() path
  // to avoid thundering herd: each CPU pops one frame then loops back,
  // so waking all N threads just causes N-1 to contend on m_, check the
  // predicate, and go back to sleep.
  void notify_cpu_one()
  {
    cpu_cv_.notify_one();
  }

  // State snapshot passed to CPU worker predicates.  Lets the predicate
  // inspect queue state without calling back into TaskQueue (which would
  // deadlock since we already hold m_).
  struct QueueState {
    size_t depth;         // number of tasks in queue
    bool   done;          // producer has called set_done()
    double front_ratio;   // compressed/decompressed ratio of front task (-1.0 if empty or unknown)
  };

  // Blocking wait for CPU workers in hybrid mode.
  // Replaces poll+sleep(1ms) loops with a proper CV wait that wakes
  // instantly when: (a) a new task is pushed, (b) producer is done,
  // or (c) notify_cpu_waiters() is called (GPU released semaphore).
  //
  // The predicate receives a QueueState snapshot (computed under the lock)
  // and returns true when the CPU worker should attempt to pop.
  // Returns false when queue is drained (caller should exit).
  // Returns true when the predicate is satisfied (caller should try to pop).
  bool wait_for_cpu(std::function<bool(const QueueState &)> may_proceed)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (true) {
      if (q_.empty() && done_) return false;  // drained
      if (!q_.empty()) {
        QueueState qs { q_.size(), done_, front_ratio_locked() };
        if (may_proceed(qs)) return true;
      }
      cpu_cv_.wait(lk);
    }
  }

  // Park a tail-yielded GPU worker (should_gpu_take() == false, all its
  // streams idle) until something that could change the decision happens:
  // any pop shrinks the queue (take_front_locked notifies when a waiter is
  // registered), the queue drains, or a scheduler tick moves the EMA rates
  // (notify_gpu_yield_waiters).  Event-driven counterpart to wait_for_cpu —
  // no polling, no fixed sleeps.  The predicate receives a QueueState
  // snapshot under the lock and, like wait_for_cpu's, must not call back
  // into TaskQueue (deadlock).  Returns false when drained (caller exits),
  // true when the predicate says the GPU should take again.
  bool wait_for_gpu_yield(std::function<bool(const QueueState &)> may_take)
  {
    std::unique_lock<std::mutex> lk(m_);
    ++gpu_yield_waiters_;
    while (true) {
      if (q_.empty() && done_) { --gpu_yield_waiters_; return false; }
      QueueState qs { q_.size(), done_, front_ratio_locked() };
      if (may_take(qs)) { --gpu_yield_waiters_; return true; }
      cv_.wait(lk);
    }
  }

  // Wake GPU workers parked in wait_for_gpu_yield().  Called by the
  // scheduler tick after an EMA update — the only input to the yield
  // decision that changes without a queue event.  Takes the lock so a
  // notify can't slip between a waiter's predicate check and its wait.
  void notify_gpu_yield_waiters()
  {
    std::lock_guard<std::mutex> lk(m_);
    if (gpu_yield_waiters_ > 0) cv_.notify_all();
  }

  // Pop one task for a CPU worker, but only if may_proceed() is true.
  // Combines the wait and pop into a single lock acquisition to avoid
  // a race between checking the predicate and another thread popping
  // the same task.  Returns false when drained (caller should exit).
  bool pop_one_cpu(Task & t, std::function<bool(const QueueState &)> may_proceed)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (true) {
      if (q_.empty() && done_) return false;
      if (!q_.empty()) {
        QueueState qs { q_.size(), done_, front_ratio_locked() };
        if (may_proceed(qs)) {
          t = take_front_locked();
          return true;
        }
      }
      cpu_cv_.wait(lk);
    }
  }

  // Non-blocking CPU pop: try to pop one task if the predicate allows.
  // Returns:  1 = got a task,  0 = predicate not met or queue empty (retry later),
  //          -1 = drained (caller should exit).
  // Used by the acquire-before-pop pattern where the worker must not block
  // while holding a FrameThrottle permit.
  int try_pop_one_cpu(Task & t, std::function<bool(const QueueState &)> may_proceed)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty() && done_) return -1;
    if (!q_.empty()) {
      QueueState qs { q_.size(), done_, front_ratio_locked() };
      if (may_proceed(qs)) {
        t = take_front_locked();
        pop_signal_locked();
        return 1;
      }
    }
    return 0;
  }

private:
  // Peek at front task's compression ratio.  CALLER MUST HOLD m_.
  double front_ratio_locked() const {
    if (q_.empty()) return -1.0;
    const Task & front = q_.front();
    if (front.decomp_size == 0) return -1.0;
    return double(front.len()) / double(front.decomp_size);
  }

  // Wake a producer blocked in push() on a full bounded queue.  No-op when
  // unbounded (max_depth_ == 0).  CALLER MUST HOLD m_.  Called by the pop paths
  // a bounded (decompress) queue actually uses — try_pop_one, pop_batch_greedy,
  // try_pop_one_cpu — plus set_done().  Other pop methods only run on compress
  // queues, which are never bounded; if one is ever used with a bounded queue,
  // add a pop_signal_locked() call there too.
  void pop_signal_locked() { if (max_depth_ > 0 || max_bytes_ > 0) space_cv_.notify_all(); }

  // Dequeue the front task, keeping queued_bytes_ (owned heap held by the queue)
  // in sync.  CALLER MUST HOLD m_.  Centralizes the byte accounting so it can't
  // drift across the many pop sites.  Views (mmap; data.size()==0) contribute 0.
  Task take_front_locked() {
    queued_bytes_ -= q_.front().data.size();
    Task t = std::move(q_.front());
    q_.pop_front();
    // Every pop path lands here, so this is the one place a depth change
    // can wake a GPU parked in wait_for_gpu_yield().  Free integer check
    // when nothing is parked (the common case).
    if (gpu_yield_waiters_ > 0) cv_.notify_all();
    return t;
  }

  std::mutex              m_;
  std::condition_variable cv_;
  std::condition_variable cpu_cv_;   // dedicated CV for CPU workers (avoids spurious wakes from GPU pops)
  std::condition_variable space_cv_; // producer waits here when a bounded queue is full
  std::deque<Task>        q_;
  bool                    done_ = false;
  size_t                  max_depth_ = 0;  // 0 = unbounded (default); >0 caps queued frames (ROADMAP 7.8)
  size_t                  max_bytes_ = 0;  // 0 = unbounded; >0 caps queued owned bytes / RAM (7.8 follow-up)
  size_t                  queued_bytes_ = 0; // running sum of q_'s owned data.size() bytes
  int                     gpu_yield_waiters_ = 0; // GPUs parked in wait_for_gpu_yield (guarded by m_)
  std::atomic<size_t>     total_tasks_{0};
};

// Queue for "rescue" tasks: chunks that a failed GPU worker couldn't compress
// get re-routed here for CPU fallback compression.
class RescueQueue {
public:
  void push(Task && t)
  {
    std::unique_lock<std::mutex> lk(m_);
    q_.push_back(std::move(t));
    cv_.notify_one();
  }

  bool pop_one(Task & t)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cv_.wait(lk);
    if (q_.empty() && done_) return false;

    t = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  // Block until the queue has at least one task or is done.
  // Returns false when drained (caller should exit).
  bool wait_for_work()
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cv_.wait(lk);
    return !(q_.empty() && done_);
  }

  // Non-blocking pop: returns 1 if got a task, 0 if empty, -1 if drained.
  int try_pop_one(Task & t)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (q_.empty() && done_) return -1;
    if (q_.empty()) return 0;
    t = std::move(q_.front());
    q_.pop_front();
    return 1;
  }

  bool drained()
  {
    std::unique_lock<std::mutex> lk(m_);
    return q_.empty() && done_;
  }

  size_t size()
  {
    std::unique_lock<std::mutex> lk(m_);
    return q_.size();
  }

  void set_done()
  {
    std::unique_lock<std::mutex> lk(m_);
    done_ = true;
    cv_.notify_all();
  }

private:
  std::mutex              m_;
  std::condition_variable cv_;
  std::deque<Task>        q_;
  bool                    done_ = false;
};

// Holds compressed output frames indexed by sequence number, allowing the
// writer thread to emit them in order even though workers finish out-of-order.
// Output-frame buffer.  shared_ptr so that workers can hold a reference
// in a per-thread pool while another reference flows through the
// ResultStore → writer → AsyncWritePool pipeline.  When the writer drops
// its reference (after writing to disk), use_count returns to 1 (worker
// only) and the worker can reuse the buffer — eliminating the per-
// iteration allocation that caused the page-fault storm fixed in
// v0.13.7 for compress and v0.13.8 for decompress.
// Allocator that DEFAULT-initializes (not value-initializes) elements grown by
// resize()/the size-ctor.  For trivial T like char the grown region is left
// UNINITIALIZED instead of zeroed, eliminating the resize() memset that
// profiling pinned at ~16% on CPU decompress (every pooled buffer's first grow
// to the full decompressed frame size) and the smaller residual on the GPU
// direct-D2H readback paths.  Safe because every producer fully fills [0,size())
// before the buffer is read (ZSTD / cudaMemcpy / assign / memcpy) and no writer
// reads past size() — DirectWriter copies exactly size() bytes into its own
// aligned buffer, so a buffer's [size(),capacity()) tail never reaches disk.
template <typename T>
struct default_init_allocator : std::allocator<T> {
  using base = std::allocator<T>;
  using base::base;
  template <typename U> struct rebind { using other = default_init_allocator<U>; };
  template <typename U>
  void construct(U * p) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void *>(p)) U;   // default-init: no zero-fill for trivial U
  }
  template <typename U, typename... Args>
  void construct(U * p, Args &&... args) {
    std::allocator_traits<base>::construct(static_cast<base &>(*this), p, std::forward<Args>(args)...);
  }
};
using FrameVec = std::vector<char, default_init_allocator<char>>;
using FrameBuf = std::shared_ptr<FrameVec>;

/*======================================================================
 VerifyPool — background decompress-verify for --verify (compress only)
 ----------------------------------------------------------------------
 Every compressed frame is a complete, self-delimiting zstd frame carrying
 an XXH64 content checksum (CPU path via ZSTD_c_checksumFlag, GPU path via
 gpu_frame_add_checksum), so a streaming decompress validates it exactly
 like `gzstd -t` — no separate hashing needed.  The writer taps each
 finished frame into this pool (a shared_ptr copy, so the frame outlives
 the write until verified) and keeps going; verification runs off the
 critical path.

 Thread policy: start with ONE worker and add more only when the backlog
 persists, capped at hardware concurrency.  The workers are plain CPU
 decompressors that contend with the compress workers for cores, so on a
 CPU-bound run verification naturally steals time from compression — the
 backpressure lands exactly where it should — while on a GPU run the
 otherwise-idle CPU absorbs it nearly free.  A bounded queue blocks the
 writer (and thus the producers) if verification can't keep up.

 On ANY frame that fails to round-trip, it sets g_verify_failed (+ the
 first failing seq) and g_gpu_failed_restart, so the driver's existing
 discard-and-rebuild path fires.  See the compress driver's retry loop.
======================================================================*/
class VerifyPool {
public:
  VerifyPool(int max_threads, size_t max_queue)
    : max_threads_(std::max(1, max_threads)),
      max_queue_(std::max<size_t>(4, max_queue)),
      high_water_(std::max<size_t>(2, max_queue_ / 2)) {
    spawn_one();   // start very low — a single verify thread
  }
  ~VerifyPool() { finish(); }

  VerifyPool(const VerifyPool &) = delete;
  VerifyPool & operator=(const VerifyPool &) = delete;

  // Writer thread hands off a finished frame.  Blocks while the queue is full
  // (bounded) — that block propagates back as compress backpressure.  Cheap
  // no-op once a failure is known: no point verifying more, we're rebuilding.
  void submit(size_t seq, const FrameBuf & frame) {
    if (g_verify_failed.load(std::memory_order_relaxed)) return;
    bool grow = false;
    {
      std::unique_lock<std::mutex> lk(m_);
      not_full_.wait(lk, [&]{
        return q_.size() < max_queue_ || g_verify_failed.load(std::memory_order_relaxed);
      });
      if (g_verify_failed.load(std::memory_order_relaxed)) return;
      q_.push_back({seq, frame});
      grow = q_.size() > high_water_;
    }
    not_empty_.notify_one();
    if (grow) maybe_grow();
  }

  // No more frames will arrive.  Drains the queue, joins all workers, and
  // returns true iff every frame verified clean.
  bool finish() {
    {
      std::lock_guard<std::mutex> lk(m_);
      if (done_) return !g_verify_failed.load(std::memory_order_relaxed);
      done_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    {
      std::lock_guard<std::mutex> tlk(threads_m_);
      for (auto & t : threads_) if (t.joinable()) t.join();
      threads_.clear();
    }
    return !g_verify_failed.load(std::memory_order_relaxed);
  }

private:
  struct Item { size_t seq; FrameBuf frame; };

  void spawn_one() {
    std::lock_guard<std::mutex> tlk(threads_m_);
    if ((int)threads_.size() >= max_threads_) return;
    threads_.emplace_back([this]{ worker(); });
  }

  // Add a worker if the backlog warrants it and we're below the cap.  Called
  // off-lock from submit(); guarded by threads_m_ so growth is race-free.
  void maybe_grow() {
    {
      std::lock_guard<std::mutex> tlk(threads_m_);
      if ((int)threads_.size() >= max_threads_) return;
    }
    // Re-check backlog under the queue lock so we don't over-spawn on a blip.
    {
      std::lock_guard<std::mutex> lk(m_);
      if (q_.size() <= high_water_ || done_) return;
    }
    spawn_one();
  }

  void worker() {
    ZSTD_DStream * zds = ZSTD_createDStream();
    std::vector<char> scratch(ZSTD_DStreamOutSize());   // reused output sink (discarded)
    if (!zds) {   // allocation failure — treat as a verify failure, fail safe
      mark_failed((size_t)-1);
      return;
    }
    for (;;) {
      Item it;
      {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&]{
          return !q_.empty() || done_ || g_verify_failed.load(std::memory_order_relaxed);
        });
        if (g_verify_failed.load(std::memory_order_relaxed)) break;
        if (q_.empty()) { if (done_) break; else continue; }
        it = std::move(q_.front());
        q_.pop_front();
      }
      not_full_.notify_one();
      if (it.frame && !verify_one(zds, scratch, *it.frame)) {
        mark_failed(it.seq);
        break;
      }
    }
    ZSTD_freeDStream(zds);
  }

  // Streaming-decompress the whole frame, discarding output.  Returns false on
  // any decode error OR XXH64 checksum mismatch (zstd reports both as errors),
  // or on a truncated frame (last call still expecting input).
  static bool verify_one(ZSTD_DStream * zds, std::vector<char> & scratch, const FrameVec & f) {
    ZSTD_initDStream(zds);
    ZSTD_inBuffer in{ f.data(), f.size(), 0 };
    size_t last = 0;
    while (in.pos < in.size) {
      ZSTD_outBuffer out{ scratch.data(), scratch.size(), 0 };
      last = ZSTD_decompressStream(zds, &out, &in);
      if (ZSTD_isError(last)) return false;
    }
    return last == 0;   // 0 = frame(s) ended cleanly; >0 = truncated
  }

  void mark_failed(size_t seq) {
    int64_t expected = -1;
    g_verify_failed_seq.compare_exchange_strong(expected, (int64_t)seq);  // record first
    g_verify_failed.store(true, std::memory_order_relaxed);
    g_gpu_failed_restart.store(true);   // reuse the driver's discard-and-rebuild path
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  const int    max_threads_;
  const size_t max_queue_;
  const size_t high_water_;

  std::mutex                 m_;
  std::condition_variable    not_empty_, not_full_;
  std::deque<Item>           q_;
  bool                       done_ = false;

  std::mutex                 threads_m_;
  std::vector<std::thread>   threads_;
};

// Set by the compress driver while a --verify run is in flight; the writer's
// in-order drain taps each finished frame into it.  null = verification off.
static VerifyPool * g_verify_pool = nullptr;

struct ResultStore {
  std::mutex                                     m;
  std::condition_variable                        cv;
  std::unordered_map<size_t, FrameBuf>           data;           // seq -> compressed frame
  size_t                                         next_to_write = 0;
  size_t                                         total_tasks   = 0;
  bool                                           producer_done = false;
  bool                                           workers_done  = false;

  // --- Per-producer result slots (reduces mutex contention) ---
  // Each GPU worker and CPU worker group gets its own slot.
  // Producers push to their slot's local queue (own mutex, no contention).
  // Writer drains all slots into the shared map periodically.
  struct Slot {
    std::mutex slot_m;
    std::vector<std::pair<size_t, FrameBuf>> pending;  // (seq, data)
  };
  std::vector<std::unique_ptr<Slot>> slots;

  // Create N slots (call before launching workers)
  void init_slots(int n) {
    slots.resize(n);
    for (int i = 0; i < n; ++i)
      slots[i] = std::make_unique<Slot>();
  }

  // Producer: push a result to a specific slot (low contention  one producer per slot)
  void push_to_slot(int slot_id, size_t seq, FrameBuf frame) {
    if (slot_id >= 0 && slot_id < (int)slots.size()) {
      std::lock_guard<std::mutex> lk(slots[slot_id]->slot_m);
      slots[slot_id]->pending.emplace_back(seq, std::move(frame));
      // GPU path: no per-frame notify. Batch-completion notify handles it.
    } else {
      // CPU path: shared map, always notify.  A "targeted notify only when
      // seq == next_to_write" optimization was tried in v0.12.16 and removed
      // in v0.12.17 — it was not safe in hybrid mode: if a GPU batch is
      // in flight (no per-frame notify) and a CPU worker pushes an
      // out-of-order frame, the writer is left sleeping with data in the
      // shared map that it cannot see until the next GPU batch completes.
      // In pathological sequences (many CPU out-of-order pushes between
      // GPU batch boundaries) the writer stalls.  Per-CPU-push notify_one
      // is cheap enough — the writer is a single waiter, not a herd — so
      // correctness wins over the micro-optimization.
      std::lock_guard<std::mutex> lk(m);
      data.emplace(seq, std::move(frame));
      cv.notify_one();
    }
  }

  // Writer: drain all slots into the shared map.
  // CALLER MUST HOLD results.m (the shared lock).
  void drain_slots_locked() {
    for (auto & s : slots) {
      std::lock_guard<std::mutex> lk(s->slot_m);
      if (s->pending.empty()) continue;
      for (auto & p : s->pending)
        data.emplace(p.first, std::move(p.second));
      s->pending.clear();
    }
  }
};

// Writer thread: waits for compressed frames to appear in sequence order,
// then writes them to the output file.  This keeps the output deterministic
// (frames in the same order as the input) regardless of which worker finishes first.

/*======================================================================
 Frame Throttle (counting semaphore)
 -----------------------------------------------------------------------
 Bounds the number of frames in flight between "popped from queue" and
 "written to disk."  Workers acquire permits before popping; the writer
 releases permits after each frame is physically written.

 This is deadlock-free by construction: the task queue is FIFO, so the
 frame the writer needs next is always among the oldest in-flight frames.
 It was popped first and will finish processing -- the writer never waits
 for a frame that hasn't been popped yet while all permits are consumed.

 No byte-level tracking, no hysteresis, no writer-stalled flag.
======================================================================*/
class FrameThrottle {
public:
  explicit FrameThrottle(int max_in_flight = 512)
    : permits_(max_in_flight), max_(max_in_flight),
      disabled_(max_in_flight <= 0) {}

  bool disabled() const { return disabled_; }

  // Acquire n permits.  Blocks until enough are available (or done).
  // Greedy: takes whatever is available immediately, waits for the rest.
  // No-op when disabled (--throttle-frames=0): no lock, no accounting.
  void acquire(int n = 1) {
    if (n <= 0 || disabled_) return;
    std::unique_lock<std::mutex> lk(m_);
    bool waited = false;
    std::chrono::steady_clock::time_point t0;
    while (n > 0 && !done_) {
      if (permits_ <= 0 && !done_) {
        if (!waited) {
          waited = true;
          t0 = std::chrono::steady_clock::now();
        }
        cv_.wait(lk, [&] { return permits_ > 0 || done_; });
      }
      if (done_) break;
      int take = std::min(n, permits_);
      permits_ -= take;
      n -= take;
    }
    // Track peak observed in-flight (advisory; under the lock so consistent).
    int observed = max_ - permits_;
    int prev = peak_in_flight_.load(std::memory_order_relaxed);
    while (observed > prev &&
           !peak_in_flight_.compare_exchange_weak(prev, observed,
                                                  std::memory_order_relaxed)) {}
    if (waited) {
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - t0).count();
      block_count_.fetch_add(1, std::memory_order_relaxed);
      block_nanos_.fetch_add((uint64_t)ns, std::memory_order_relaxed);
    }
  }

  // Release n permits (called by writer after writing frames to disk).
  //
  // Wake exactly n waiters, not all: with 22+ workers saturated against the
  // throttle, notify_all on every single-frame release caused a thundering
  // herd (each aio write fan-out-wakes every worker, they contend on m_,
  // one wins, 21 go back to sleep).  At 1280 frames × 22 waiters that's
  // ~28k wasteful context switches and serializes workers on throttle.m_.
  // notify_one per permit released matches the actual capacity change and
  // keeps the pipeline unblocked without the convoy.
  void release(int n = 1) {
    if (n <= 0 || disabled_) return;
    {
      std::lock_guard<std::mutex> lk(m_);
      permits_ += n;
    }
    for (int i = 0; i < n; ++i) cv_.notify_one();
  }

  // Signal shutdown -- wake all blocked workers so they can exit.
  // No-op when disabled (no waiters to wake).
  void set_done() {
    if (disabled_) return;
    {
      std::lock_guard<std::mutex> lk(m_);
      done_ = true;
    }
    cv_.notify_all();
  }

  int in_flight() const {
    // Not locked -- advisory only (for stats/debug).
    return max_ - permits_;
  }

  int max_permits() const { return max_; }

  struct Stats {
    int max;
    int peak_in_flight;
    uint64_t block_count;
    uint64_t block_nanos;
  };
  Stats stats() const {
    return Stats{
      max_,
      peak_in_flight_.load(std::memory_order_relaxed),
      block_count_.load(std::memory_order_relaxed),
      block_nanos_.load(std::memory_order_relaxed),
    };
  }

  // -- Pool-drain notification (v0.13.10+) --
  // The writer calls notify_drain() after each buf.reset() (i.e. after
  // dropping a per-frame shared_ptr ref).  Per-worker bounded buffer
  // pools (cpu_worker, cpu_decomp_worker) wait on this CV when their
  // pool is full instead of yield-spinning, eliminating the sched_yield
  // syscall storm seen in v0.13.9 at high thread counts.
  //
  // Why a separate CV from permit-acquire (cv_):
  //   - permit-acquire is contended on m_; sharing the CV would force
  //     pool-waiters and permit-waiters to serialize on the same mutex.
  //   - notify_all on every shared_ptr drop on m_ would block release()
  //     for the duration of broadcasting.
  // drain_cv_ + drain_m_ are dedicated, so permit-acquire is unaffected.
  void notify_drain() {
    // notify_all because the writer doesn't know which worker's pool
    // slot just freed.  Wakes more than necessary but workers re-check
    // predicate via atomic shared_ptr use_count() — cheap.
    drain_cv_.notify_all();
  }

  template<class Predicate>
  void wait_for_drain(Predicate pred) {
    std::unique_lock<std::mutex> lk(drain_m_);
    // 10ms timeout is a safety net for any missed notify; in normal
    // operation the writer notifies on every frame drop so workers
    // wake promptly.
    drain_cv_.wait_for(lk, std::chrono::milliseconds(10), pred);
  }

private:
  std::mutex              drain_m_;
  std::condition_variable drain_cv_;
  int permits_;
  int max_;
  bool disabled_;        // true when constructed with max_in_flight <= 0
  bool done_ = false;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<int>      peak_in_flight_{0};
  std::atomic<uint64_t> block_count_{0};
  std::atomic<uint64_t> block_nanos_{0};
};

// In-order, byte-bounded hand-off of decompressed frames from the decompress
// writer thread to an in-process consumer — the `-t --tar` validator.  It
// replaces the kernel pipe the verifier used to drain: the validator parses tar
// headers by random access (skip() = pointer arithmetic over already-decompressed
// frames) instead of having every data byte copied through a pipe.  Bounded by a
// byte budget so a huge archive can't balloon RAM; push() blocks the producer
// when over budget (but always admits at least one frame to avoid a stall on a
// single oversized frame).
class FrameSink {
public:
  explicit FrameSink(size_t max_bytes) : max_bytes_(max_bytes) {}

  // Producer (writer_thread): enqueue one decompressed frame, in order.
  void push(FrameBuf f) {
    if (!f || f->empty()) return;                 // nothing to verify; drop
    const size_t sz = f->size();
    std::unique_lock<std::mutex> lk(m_);
    // Producer blocked here == the consumer (tar parser) is the bottleneck:
    // decompress is ready but the sink is full because extraction is behind.
    if (!(bytes_ + sz <= max_bytes_ || q_.empty())) {
      uint64_t t0 = now_ns();
      cv_prod_.wait(lk, [&] { return bytes_ + sz <= max_bytes_ || q_.empty(); });
      prod_wait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    bytes_ += sz;
    q_.push_back(std::move(f));
    cv_cons_.notify_one();
  }

  // Producer done: no more frames will arrive.
  void close() {
    { std::lock_guard<std::mutex> lk(m_); closed_ = true; }
    cv_cons_.notify_all();
  }

  // Consumer (validator/extractor): next frame in order, or null at EOF.
  FrameBuf pop() {
    std::unique_lock<std::mutex> lk(m_);
    // Consumer blocked here == the producer (decompress) is the bottleneck:
    // the tar parser is ready but no decompressed frame is available yet.
    if (q_.empty() && !closed_) {
      uint64_t t0 = now_ns();
      cv_cons_.wait(lk, [&] { return !q_.empty() || closed_; });
      cons_wait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
    if (q_.empty()) return nullptr;               // closed and drained
    FrameBuf f = std::move(q_.front());
    q_.pop_front();
    bytes_ -= f->size();
    cv_prod_.notify_one();
    return f;
  }

  // Diagnostics (-v): time the producer spent blocked on a full sink (consumer/
  // extraction is the bottleneck) vs the consumer spent blocked on an empty sink
  // (producer/decompress is the bottleneck).
  uint64_t producer_wait_ns() const { return prod_wait_ns_.load(std::memory_order_relaxed); }
  uint64_t consumer_wait_ns() const { return cons_wait_ns_.load(std::memory_order_relaxed); }

private:
  std::mutex m_;
  std::condition_variable cv_prod_, cv_cons_;
  std::deque<FrameBuf> q_;
  size_t bytes_ = 0;
  size_t max_bytes_;
  bool closed_ = false;
  std::atomic<uint64_t> prod_wait_ns_{0}, cons_wait_ns_{0};
};

// Set by verify_tar() / extract_tar() for the duration of one archive's
// decompress: when non-null the decompress writer_thread (and the streaming/
// fallback helpers) route finished frames here, to the in-process tar parser,
// instead of through a kernel pipe (mirrors the g_direct_writer global-sink
// pattern).  Used for both `-t --tar` (validate) and `-d --tar` (extract).
static FrameSink * g_tar_decomp_sink = nullptr;

// End-of-run throttle stats; one line at V_DEBUG.  Shows how often producers
// actually waited (saturation indicator) and peak in-flight frames vs budget.
static void log_throttle_stats(const FrameThrottle & t, const Options & opt,
                               const char * phase)
{
  if (opt.verbosity < V_DEBUG) return;
  if (t.disabled()) {
    vlog(V_DEBUG, opt, std::string("[THROTTLE] stats [") + phase
         + "]: DISABLED (--throttle-frames=0)\n");
    return;
  }
  auto s = t.stats();
  double saturation = s.max > 0 ? 100.0 * double(s.peak_in_flight) / double(s.max) : 0.0;
  double ms = double(s.block_nanos) / 1e6;
  std::ostringstream os;
  os << "[THROTTLE] stats [" << phase << "]: "
     << "peak=" << s.peak_in_flight << "/" << s.max
     << " (" << std::fixed << std::setprecision(1) << saturation << "%), "
     << "block_count=" << s.block_count
     << ", block_time=" << std::fixed << std::setprecision(2) << ms << "ms";
  vlog(V_DEBUG, opt, os.str() + "\n");
}

/*======================================================================
 Async write pool  double-buffered I/O to overlap compute with disk writes
 -----------------------------------------------------------------------
 The writer thread collects ready frames, then hands them to the write
 pool instead of writing synchronously.  The pool uses a background
 thread to flush data to disk while the writer thread goes back to
 collecting the next batch.  This mirrors zstd's AIO_WritePool design.

 Two slots alternate: while slot A is being written to disk by the
 background thread, slot B is being filled by the writer thread.
======================================================================*/
class AsyncWritePool {
public:
  explicit AsyncWritePool(FILE * out_file, DirectWriter * dw, bool sparse = true,
                          Meter * meter = nullptr, FrameThrottle * bp = nullptr,
                          bool progressive_sync = false)
    : out_(out_file), dw_(dw), sparse_(sparse), done_(false),
      progressive_sync_(progressive_sync), meter_(meter), bp_(bp)
  {
    worker_ = std::thread(&AsyncWritePool::worker_fn, this);
  }

  ~AsyncWritePool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      done_ = true;
    }
    cv_producer_.notify_one();
    if (worker_.joinable()) worker_.join();
  }

  // Submit a batch of data to be written asynchronously.
  // Takes ownership of the data via move.  Blocks if the previous
  // write hasn't finished yet (backpressure).
  void submit(std::vector<FrameBuf> && buffers) {
    // Offload the per-frame zero scan to here (the writer thread) — done BEFORE
    // the backpressure wait, so it overlaps the write worker draining the
    // previous batch instead of running serially on the write path.  Only when
    // sparse + O_DIRECT (the case where the worker would otherwise scan).
    std::vector<char> dense;
    if (sparse_ && dw_) {
      dense.resize(buffers.size());
      for (size_t i = 0; i < buffers.size(); ++i)
        dense[i] = (buffers[i] && !buffers[i]->empty())
                       ? (char)frame_dense(buffers[i]->data(), buffers[i]->size()) : (char)1;
    }
    std::unique_lock<std::mutex> lk(m_);
    // Wait for previous write to finish (backpressure)
    cv_consumer_.wait(lk, [this]{ return pending_.empty() || done_; });
    if (done_) return;
    pending_ = std::move(buffers);
    pending_dense_ = std::move(dense);
    lk.unlock();
    cv_producer_.notify_one();
  }

  // Wait for all queued writes to complete.
  void flush() {
    std::unique_lock<std::mutex> lk(m_);
    // Wait until the queued batch is both dequeued AND physically written.
    // Waiting only on pending_.empty() returned while the worker was still
    // writing the final batch (pending_ is emptied by the move, before the
    // write), so a write error on that last batch escaped the had_error()
    // check in writer_thread and the run reported success over corrupt output.
    cv_consumer_.wait(lk, [this]{ return pending_.empty() && !writing_; });
  }

  bool had_error() const { return error_; }

private:
  // Check if a region is all zeros (for sparse file support).
  // Uses size_t-wide comparisons for speed.
  static bool is_all_zero(const char * p, size_t len) {
    // Check in size_t chunks for speed.  memcpy into a size_t rather than an
    // unaligned reinterpret_cast: p is vector<char>::data() (not size_t-
    // aligned), so the cast is UB on strict-alignment targets.  With a constant
    // size this compiles to the same wide load on x86 (ROADMAP 7.6).
    size_t words = len / sizeof(size_t);
    for (size_t i = 0; i < words; ++i) {
      size_t w;
      std::memcpy(&w, p + i * sizeof(size_t), sizeof(w));
      if (w != 0) return false;
    }
    // Check remaining bytes
    for (size_t i = words * sizeof(size_t); i < len; ++i)
      if (p[i] != 0) return false;
    return true;
  }

  // "Dense" = the frame has no all-zero 4 KiB block, so write_sparse would find
  // nothing to skip and just write the whole thing.  Computed on the writer
  // thread (in submit) so the per-block zero scan overlaps the write instead of
  // running serially on the write thread.  Early-exits at the first zero block
  // (so genuinely sparse frames cost little here and fall through to the full
  // write_sparse).  SPARSE_BLOCK == 4096 (matches write_sparse).
  static bool frame_dense(const char * p, size_t len) {
    for (size_t off = 0; off + 4096 <= len; off += 4096)
      if (is_all_zero(p + off, 4096)) return false;
    return true;   // remainder (< 4096) is always written, never skipped
  }

  // Write a whole buffer with no zero-scan (dense frames, and the --no-sparse
  // fast path share this).  Records the I/O time like write_sparse's writes.
  bool write_whole(const char * data, size_t len) {
    uint64_t t0 = now_ns();
    bool ok = true;
#ifndef _WIN32
    if (dw_) ok = dw_->write(data, len);
    else
#endif
    if (out_) ok = (robust_fwrite(data, len, out_) == len);
    if (meter_) {
      meter_->writer_iowrite_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
      if (ok) meter_->wrote_bytes.fetch_add(len, std::memory_order_relaxed);
    }
    return ok;
  }

  // Write a buffer with sparse file optimization.
  // Scans for zero-filled blocks and seeks past them instead of writing.
  // The OS creates a sparse file  zero regions don't consume disk space
  // and aren't physically written, which can dramatically speed up output
  // for data with large zero runs (e.g., disk images, memory dumps).
  bool write_sparse(const char * data, size_t len, bool is_last_buffer = false) {
    static constexpr size_t SPARSE_BLOCK = 4096;  // check in page-sized blocks
    static constexpr size_t PROGRESS_INTERVAL = 16 * 1024 * 1024;  // 16 MiB

    // Time only the actual write/seek syscalls into writer_iowrite_ns; the rest
    // of write_sparse's time (tracked as writer_disk_ns by the caller) is the
    // single-threaded zero-scan.  The split shows whether the write thread is
    // device-I/O-bound or scan-bound (-v).
    auto timed_io = [&](auto && fn) -> bool {
      uint64_t t0 = now_ns();
      bool ok = fn();
      if (meter_) meter_->writer_iowrite_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
      return ok;
    };

    // Sparse disabled: write the whole buffer in one shot, no per-block zero
    // scan.  The scanning loop below calls is_all_zero on every block even when
    // sparse is off (to coalesce write runs), so skipping it here avoids a full
    // single-threaded pass over the data on the write thread's critical path.
    if (!sparse_) {
      (void)is_last_buffer;
#ifndef _WIN32
      if (dw_) { if (!timed_io([&]{ return dw_->write(data, len); })) return false; }
      else
#endif
      if (out_) { if (!timed_io([&]{ return robust_fwrite(data, len, out_) == len; })) return false; }
      if (meter_) meter_->wrote_bytes.fetch_add(len, std::memory_order_relaxed);
      return true;
    }

    size_t pos = 0;
    size_t bytes_since_progress = 0;
    while (pos < len) {
      size_t remain = len - pos;
      size_t block = std::min(remain, SPARSE_BLOCK);

      // Never sparse-skip the final block of the last buffer in a batch
      // the file must end with a physical write or it will be truncated.
      bool at_end = is_last_buffer && (pos + block >= len);

      // O_DIRECT writes require ALIGN-aligned file offsets.  A sparse seek is
      // only safe through the DirectWriter when the current logical position
      // is aligned (zero runs are ALIGN multiples, so alignment is preserved
      // across the skip).  Frames are usually 4 KiB multiples so this is
      // almost always true; odd-size frames just write their zeros instead of
      // seeking — correct output, merely less sparse.  Without this guard the
      // post-seek flush issues an O_DIRECT write at an unaligned offset and
      // fails with EINVAL (surfacing as a bogus "disk full?" error).
#ifndef _WIN32
      bool skip_ok = !dw_ || (dw_->total_bytes() % DirectWriter::ALIGN) == 0;
#else
      bool skip_ok = true;
#endif

      if (sparse_ && !at_end && skip_ok && block == SPARSE_BLOCK && is_all_zero(data + pos, block)) {
        // Coalesce consecutive zero blocks into a single skip, so the seek
        // (and, on a preallocated O_DIRECT file, the hole-punch inside
        // seek_forward) happens once per zero run instead of once per 4 KiB.
        size_t zero_end = pos + block;
        while (zero_end < len) {
          size_t nb = std::min(len - zero_end, SPARSE_BLOCK);
          bool nb_at_end = is_last_buffer && (zero_end + nb >= len);
          if (nb != SPARSE_BLOCK || nb_at_end || !is_all_zero(data + zero_end, nb))
            break;
          zero_end += nb;
        }
        size_t zero_len = zero_end - pos;
        if (dw_) {
          if (!timed_io([&]{ return dw_->seek_forward(zero_len); })) return false;
        } else if (out_) {
          if (!timed_io([&]{ return std::fseek(out_, (long)zero_len, SEEK_CUR) == 0; }))
            return false;
        }
        bytes_since_progress += zero_len;
        pos = zero_end;
        sparse_saved_ += zero_len;
      } else {
        // Find how many consecutive non-zero (or partial) blocks to write at once
        size_t write_end = pos + block;
        while (write_end < len) {
          size_t next_remain = len - write_end;
          size_t next_block = std::min(next_remain, SPARSE_BLOCK);
          if (next_block == SPARSE_BLOCK && is_all_zero(data + write_end, next_block))
            break;  // next block is zero  stop here, let the seek handle it
          write_end += next_block;
        }

        size_t to_write = write_end - pos;
        if (dw_) {
          if (!timed_io([&]{ return dw_->write(data + pos, to_write); })) return false;
        } else if (out_) {
          if (!timed_io([&]{ return robust_fwrite(data + pos, to_write, out_) == to_write; })) return false;
        }
        bytes_since_progress += to_write;
        pos = write_end;
      }

      if (meter_ && bytes_since_progress >= PROGRESS_INTERVAL) {
        meter_->wrote_bytes.fetch_add(bytes_since_progress, std::memory_order_relaxed);
        bytes_since_progress = 0;
      }
    }
    // Flush remaining progress
    if (meter_ && bytes_since_progress > 0)
      meter_->wrote_bytes.fetch_add(bytes_since_progress, std::memory_order_relaxed);
    return true;
  }

  void worker_fn() {
    while (true) {
      std::vector<FrameBuf> work;
      std::vector<char> work_dense;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_producer_.wait(lk, [this]{ return !pending_.empty() || done_; });
        if (done_ && pending_.empty()) return;
        work = std::move(pending_);
        work_dense = std::move(pending_dense_);
        writing_ = true;  // mark busy: flush() must wait until this batch is written, not just moved
      }
      cv_consumer_.notify_one();  // pending_ now empty: unblock submit() backpressure

      // Write all buffers (last buffer's final block always written to ensure file length)
      for (size_t bi = 0; bi < work.size(); ++bi) {
        auto & buf = work[bi];   // FrameBuf (shared_ptr<vector<char>>)
        uint64_t w_t0 = (g_perf || meter_) ? now_ns() : 0;
        bool is_last = (bi == work.size() - 1);
        // Dense frames (no zero blocks; scanned in submit) write whole with no
        // scan here; only frames with actual holes run the full write_sparse.
        bool dense = (bi < work_dense.size()) && work_dense[bi];
        bool wrote = dense ? write_whole(buf->data(), buf->size())
                           : write_sparse(buf->data(), buf->size(), is_last);
        if (!wrote) {
          // Clear writing_ and surface the error before returning, so a
          // flush() blocked on !writing_ wakes and the post-flush had_error()
          // check sees the failure (final-batch errors used to be lost here).
          {
            std::lock_guard<std::mutex> lk(m_);
            writing_ = false;
            error_ = true;
          }
          cv_consumer_.notify_one();
          return;
        }
        if (g_perf || meter_) {
          const uint64_t w_dt = now_ns() - w_t0;
          if (meter_) meter_->writer_disk_ns.fetch_add(w_dt, std::memory_order_relaxed);
          if (g_perf) {
            g_perf->write_ns.fetch_add(w_dt);
            g_perf->write_bytes_total.fetch_add(buf->size());
          }
        }
        // wrote_bytes is now updated incrementally within write_sparse()
        // Release one frame permit: writer made progress, blocked workers may unblock.
        if (bp_) bp_->release(1);
        // Drop our shared_ptr ref now (not at end of loop) so the worker pool
        // can recycle the buffer for its next frame as soon as we're done.
        // After the drop, notify pool-waiters: this is the moment use_count
        // transitions from 2 to 1 (worker becomes the sole owner), so the
        // worker's pool acquire predicate will now succeed.
        buf.reset();
        if (bp_) bp_->notify_drain();
      }

      // Batch fully written (handed to the OS): clear writing_ and wake any
      // flush() waiter.  flush() blocks on (pending_.empty() && !writing_) so
      // it can no longer return mid-write — which is what let final-batch I/O
      // errors slip past the had_error() check in writer_thread.
      {
        std::lock_guard<std::mutex> lk(m_);
        writing_ = false;
      }
      cv_consumer_.notify_one();

#ifdef __linux__
      // Progressive writeback: kick dirty pages to disk asynchronously after
      // each batch so they don't pile up and cause a stall on rename()/close().
      // Only for decompression — compression output is smaller and the CPU work
      // gives the kernel time to flush naturally.  Forcing writeback during
      // compression creates I/O contention with subsequent fwrite() calls,
      // triggering balance_dirty_pages throttling.
      if (progressive_sync_ && !dw_ && out_) {
        int fd = fileno(out_);
        off_t cur = ftello(out_);
        if (cur > sync_offset_) {
          ::sync_file_range(fd, sync_offset_, cur - sync_offset_,
                            SYNC_FILE_RANGE_WRITE);
          sync_offset_ = cur;
        }
      }
#endif
    }
  }

  FILE * out_;
  DirectWriter * dw_;
  bool sparse_;
  std::mutex m_;
  std::condition_variable cv_producer_;
  std::condition_variable cv_consumer_;
  std::vector<FrameBuf> pending_;
  std::vector<char>     pending_dense_;  // per-frame: 1 = no zero blocks → write whole, skip the scan
  std::thread worker_;
  bool done_;
  bool writing_ = false;  // true while the worker is physically writing a batch (guarded by m_)
  bool error_ = false;
  bool progressive_sync_ = false;
  uint64_t sparse_saved_ = 0;  // bytes skipped via sparse seek
  off_t sync_offset_ = 0;       // progressive writeback high-water mark
  Meter * meter_ = nullptr;     // for tracking physically-written bytes
  FrameThrottle * bp_ = nullptr;  // frame throttle (releases permits after writes)
};

static void writer_thread(FILE * out, ResultStore & results,
                          const Options & opt, Meter * m,
                          FrameThrottle * bp)
{
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  // pin_thread_to_core(1);              // disabled: hurts on loaded machines
  const bool skip_write = (opt.mode == Mode::TEST);
  // -t/-d --tar: hand decompressed frames to the in-process tar parser instead
  // of a kernel pipe.  Set by verify_tar()/extract_tar() for this archive.
  FrameSink * const sink = g_tar_decomp_sink;

  // Create async write pool for double-buffered I/O
  AsyncWritePool * aio = nullptr;
  std::unique_ptr<AsyncWritePool> aio_ptr;
  if (!skip_write && !sink) {
    // Sparse file support: skip zero-filled 4K blocks via seek.
    // --sparse forces on, --no-sparse forces off, default auto (file:on, stdout:off).
    bool enable_sparse;
    if (opt.sparse_mode == 1) {
      enable_sparse = true;  // --sparse: force on
    } else if (opt.sparse_mode == 0) {
      enable_sparse = false; // --no-sparse: force off
    } else {
      // Auto: enable for seekable file output (not pipes/stdout)
      enable_sparse = (g_direct_writer != nullptr)   // O_DIRECT file
                   || (out && out != stdout);          // regular file via fwrite
    }
    // Progressive writeback (sync_file_range) is enabled for decompression to
    // avoid the multi-second rename stall on ext4 data=ordered when the
    // tmp file's dirty pages must flush at commit time.  Skip it for
    // --overwrite (no tmp+rename) — the user explicitly opted out of
    // atomicity for speed, and the writeback hint just steals bandwidth
    // from fwrite.  Also skip for stdout (no rename).
    bool psync = (opt.mode == Mode::DECOMPRESS) && !opt.unsafe_overwrite
                 && (out != stdout);
#ifndef _WIN32
    aio_ptr = std::make_unique<AsyncWritePool>(out, g_direct_writer, enable_sparse, m, bp, psync);
    aio = aio_ptr.get();
#else
    aio_ptr = std::make_unique<AsyncWritePool>(out, nullptr, enable_sparse, m, bp, psync);
    aio = aio_ptr.get();
#endif
  }

  std::unique_lock<std::mutex> lk(results.m);

  while (true) {
    bool all_done = results.producer_done
                 && results.workers_done
                 && results.next_to_write >= results.total_tasks;

    uint64_t wait_t0 = g_perf ? now_ns() : 0;
    bool waited = false;

    // Wait for the next sequential frame.
    // On each wakeup, drain per-GPU slots into the shared map, then check
    // if the next sequential frame is available.  Producers call notify_all
    // on every frame delivery, so the writer wakes promptly.
    while (results.data.count(results.next_to_write) == 0 && !all_done
           && !g_gpu_aborted.load(std::memory_order_relaxed)) {
      results.drain_slots_locked();
      if (results.data.count(results.next_to_write) != 0) break;
      waited = true;
      // Classify this wait for the Meter's writer-state accounting: if any
      // LATER frame is already buffered, the writer is head-of-line blocked
      // on a straggler (pipeline ordering); if nothing is buffered at all,
      // upstream simply hasn't produced (starved).  Snapshot at sleep time.
      const bool seg_hol = !results.data.empty();
      const uint64_t seg_t0 = m ? now_ns() : 0;
      // Use timed wait to detect potential deadlocks: if all workers are done
      // but the next expected frame never arrives, something is wrong.
      if (results.workers_done) {
        results.cv.wait_for(lk, std::chrono::seconds(5));
        if (m) {
          const uint64_t seg = now_ns() - seg_t0;
          (seg_hol ? m->writer_hol_ns : m->writer_starved_ns)
              .fetch_add(seg, std::memory_order_relaxed);
          // Depth at wake: frames buffered behind the gap while we waited.
          if (seg_hol)
            m->writer_hol_depth_ns.fetch_add(seg * (uint64_t)results.data.size(),
                                             std::memory_order_relaxed);
        }
        results.drain_slots_locked();
        all_done = results.producer_done
                && results.workers_done
                && results.next_to_write >= results.total_tasks;
        if (!all_done && !g_gpu_aborted.load(std::memory_order_relaxed)
            && results.data.count(results.next_to_write) == 0) {
          // Workers are all done but we're still missing frames  this is a bug.
          // Log diagnostics and abort to avoid producing corrupt output.
          // (A GPU fault sets g_gpu_aborted, in which case missing frames are
          // EXPECTED — the doomed output is discarded and rebuilt — so we exit
          // cleanly below instead of treating it as an internal error.)
          std::ostringstream os;
          os << "internal error: writer stuck  workers_done but frame "
             << results.next_to_write << " of " << results.total_tasks
             << " missing (have " << results.data.size() << " buffered)";
          // Unlock before die() to avoid deadlock in cleanup
          lk.unlock();
          die(os.str());
        }
      } else {
        results.cv.wait(lk);
        if (m) {
          const uint64_t seg = now_ns() - seg_t0;
          (seg_hol ? m->writer_hol_ns : m->writer_starved_ns)
              .fetch_add(seg, std::memory_order_relaxed);
          // Depth at wake: frames buffered behind the gap while we waited.
          if (seg_hol)
            m->writer_hol_depth_ns.fetch_add(seg * (uint64_t)results.data.size(),
                                             std::memory_order_relaxed);
        }
      }
      results.drain_slots_locked();
      all_done = results.producer_done
              && results.workers_done
              && results.next_to_write >= results.total_tasks;
    }

    if (g_perf && waited) {
      g_perf->writer_wait_ns.fetch_add(now_ns() - wait_t0);
      g_perf->writer_wait_count.fetch_add(1);
      g_perf->out_of_order_waits.fetch_add(1);
    }
    if (all_done || g_gpu_aborted.load(std::memory_order_relaxed)) break;  // GPU-fault abort: stop, output is discarded

    // Batch drain: collect ALL consecutive ready frames
    std::vector<FrameBuf> batch;
    size_t batch_bytes = 0;
    const size_t batch_start_seq = results.next_to_write;   // seq of batch[0]
    while (true) {
      auto it = results.data.find(results.next_to_write);
      if (it == results.data.end()) break;
      batch_bytes += it->second->size();
      batch.push_back(std::move(it->second));
      results.data.erase(it);
      ++results.next_to_write;
    }

    if (batch.empty()) continue;

    // Count frames handed to writer for progress tracking.
    if (m) m->tasks_done.fetch_add(batch.size(), std::memory_order_relaxed);
    lk.unlock();

    // --verify tap: hand each finished frame to the background decompress-verify
    // pool BEFORE it is written (the pool holds a shared_ptr copy, so the frame
    // outlives the write until verified).  Done off the ResultStore lock so a
    // full verify queue applies backpressure to the writer, never stalls
    // producers behind the results mutex.  Frames are still standard zstd frames
    // on disk; this only reads them back in RAM.
    if (g_verify_pool) {
      for (size_t k = 0; k < batch.size(); ++k) {
        const size_t seq = batch_start_seq + k;
        // Test-only: flip a byte in this frame's compressed payload exactly once,
        // so --verify must catch it and trigger a rebuild (which, the hook having
        // fired, then verifies clean).  Corrupts the SAME buffer the writer emits,
        // so the on-disk frame is genuinely bad until rebuilt.
        if (g_debug_corrupt_frame >= 0 && seq == (size_t)g_debug_corrupt_frame
            && batch[k] && batch[k]->size() > 8) {
          static std::atomic<bool> corrupted_once{false};
          bool expect = false;
          if (g_debug_corrupt_persist || corrupted_once.compare_exchange_strong(expect, true))
            (*batch[k])[batch[k]->size() / 2] ^= 0xFF;
        }
        g_verify_pool->submit(seq, batch[k]);
      }
    }

    // Update total expected output for write drain progress.
    // For decompress: total_out is pre-set by stream_frames_to_queue (known from frame headers).
    // For compress: total_out starts at 0; accumulate here as compressed frames are collected
    //   since final compressed sizes aren't known upfront.
    if (m && opt.mode == Mode::COMPRESS) {
      m->total_out.fetch_add(batch_bytes, std::memory_order_relaxed);
    }

    // Submit to writer backend
    if (opt.verbosity >= V_DEBUG) {
      char bs[32];
      human_bytes(double(batch_bytes), bs, sizeof(bs));
      vlog(V_DEBUG, opt, std::string("[WRITER] submitting ") + std::to_string(batch.size())
           + " frames (" + bs + ")\n");
    }
    if (sink) {
      // -t/-d --tar: hand frames to the in-process tar parser (no kernel pipe).
      // Account per frame, NOT once per batch: when the (write-bound) Extractor
      // drains slowly, push() blocks and a big batch can take seconds to hand
      // off — a per-batch fetch_add then makes the progress bar sit, then jump
      // by the whole batch.  Counting each frame as it is accepted advances the
      // bar smoothly at the drain rate (the old aio path likewise updated every
      // 16 MiB inside the write).  Release the throttle permit per frame too, so
      // workers refill as the batch drains instead of all at once at the end.
      for (auto & fb : batch) {
        const size_t fsz = fb->size();
        sink->push(std::move(fb));
        if (m) m->wrote_bytes.fetch_add(fsz, std::memory_order_relaxed);
        if (bp) bp->release(1);
      }
    } else if (aio) {
      aio->submit(std::move(batch));
      if (aio->had_error()) die_io("async write failed (disk full?)");
    } else {
      // Test mode (no AIO): update wrote_bytes directly so the progress bar
      // shows decompressed bytes verified, not stuck at 0.
      if (m) m->wrote_bytes.fetch_add(batch_bytes, std::memory_order_relaxed);
    }
    // In normal mode, wrote_bytes is updated by the AIO worker after physical
    // write completes.  This ensures the progress bar reflects actual disk I/O.

    lk.lock();
  }

  // Flush remaining writes
  if (aio) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[WRITER] draining write queue...\n");
    aio->flush();
    if (aio->had_error()) die_io("async write failed (disk full?)");
  }
  // NB: the sink is NOT closed here.  writer_thread is not always the last
  // producer — the fallback/streaming decompress tail runs after wthr.join()
  // (see decompress_cpu_mt) and pushes more frames.  verify_tar closes the sink
  // once, after the whole decompress call returns.
}

/*======================================================================
 CPU compression helpers / workers
======================================================================*/
// Thread-local CCtx avoids repeated allocation for per-chunk compression
static thread_local ZSTD_CCtx * tl_cctx = nullptr;

// Compresses one frame into `out` (a reusable per-thread buffer) and returns the
// compressed size.  `out` is left sized to >= compressBound, NOT shrunk to csz —
// the caller uses the returned size, and [csz, out.size()) is undefined padding.
static inline size_t compress_one_cpu_frame(const void * src, size_t src_size, int level, bool ultra, FrameVec & out)
{
  if (!tl_cctx) {
    tl_cctx = ZSTD_createCCtx();
    if (!tl_cctx) die("failed to create ZSTD_CCtx");
  }
  // Set compression level explicitly on every call (level may differ per invocation)
  size_t st = ZSTD_CCtx_setParameter(tl_cctx, ZSTD_c_compressionLevel, level);
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));
  // Per-frame content checksum (XXH64 in the frame footer): makes every frame
  // self-verifying so decompress/-t catches silent corruption / bit-rot.  ~4
  // bytes/frame; standard zstd, verified by any decoder.
  ZSTD_CCtx_setParameter(tl_cctx, ZSTD_c_checksumFlag, 1);

  // Ultra levels (20-22) need an explicit windowLog for the large window to take effect.
  // Without this, zstd silently clamps to its default window (~8 MiB), defeating ultra.
  apply_ultra_cctx(tl_cctx, level, ultra);

  size_t bound = ZSTD_compressBound(src_size);
  // Grow-only: ZSTD writes [0,csz) and never reads the dst buffer, so the tail
  // [csz,bound) needs no initialization.  resize() value-initializes (zeroes)
  // the grown region — up to ~16 MiB per frame, which the profile pins as the
  // single largest gzstd-attributable CPU-compress cost (callgrind:
  // _M_default_append -> memset, ~7% of total, worst on compressible data).
  // Keeping the buffer at bound across frames makes resize() zero once on the
  // first frame, then no-op.  We return csz instead of shrinking `out` so the
  // buffer never re-grows (and re-zeroes) on the next call.
  if (out.size() < bound) out.resize(bound);
  size_t csz = ZSTD_compress2(tl_cctx, out.data(), out.size(), src, src_size);
  if (ZSTD_isError(csz)) die_data(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
  return csz;
}

// Apply --memlimit to a decompression context via ZSTD_d_windowLogMax.
// zstd's decompressor refuses to allocate window buffers larger than 2^wlog,
// so a user who passes `-M 50` gets streams with >64 MiB windows rejected
// (we use floor(log2(bytes)) — errs on the strict side of the user's cap).
// No-op when mem_limit_mib == 0 (zstd's default is windowLogMax=27).
static void apply_mem_limit_to_dctx(ZSTD_DCtx * dctx, const Options & opt)
{
  if (!dctx || opt.mem_limit_mib == 0) return;
  uint64_t bytes = uint64_t(opt.mem_limit_mib) * ONE_MIB;
  // floor(log2(bytes)) — clamped to zstd's accepted range [10, 31].
  int wlog = 0;
  for (uint64_t b = bytes; b >>= 1; ) ++wlog;
  if (wlog < 10) wlog = 10;   // zstd minimum
  if (wlog > 31) wlog = 31;   // zstd maximum (2 GiB window)
  size_t st = ZSTD_DCtx_setParameter(dctx, ZSTD_d_windowLogMax, wlog);
  if (ZSTD_isError(st) && opt.verbosity >= V_ERROR) {
    std::fprintf(stderr, "gzstd: warning: could not apply --memlimit (%s)\n",
                 ZSTD_getErrorName(st));
  }
}

// Streaming decompress from an in-memory buffer (used when stream_frames_to_queue
// already consumed stdin but couldn't parse frame boundaries).
static void decompress_from_buffer(const std::vector<char> & input,
                                   FILE * out, const Options & opt, Meter * m)
{
  const size_t chunk_bytes = 4 * ONE_MIB;
  std::vector<char> outbuf(chunk_bytes);
  ZSTD_DCtx * dctx = ZSTD_createDCtx();
  if (!dctx) die("failed to create ZSTD_DCtx");
  apply_mem_limit_to_dctx(dctx, opt);

  ZSTD_inBuffer zin { input.data(), input.size(), 0 };
  ZSTD_outBuffer zout { outbuf.data(), outbuf.size(), 0 };
  if (m) m->read_bytes.fetch_add(input.size());

  size_t ret = 0;
  while (zin.pos < zin.size) {
    ret = ZSTD_decompressStream(dctx, &zout, &zin);
    if (ZSTD_isError(ret))
      die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret)
               + "\n  (re-run with --keep-going to recover what is readable and report the damage)");
    if (zout.pos > 0) {
      if (g_tar_decomp_sink) {
        // -t/-d --tar streaming/fallback path: hand the chunk to the tar parser.
        g_tar_decomp_sink->push(std::make_shared<FrameVec>(outbuf.data(), outbuf.data() + zout.pos));
      } else if (opt.mode != Mode::TEST) {
#ifndef _WIN32
        if (g_direct_writer) {
          if (!g_direct_writer->write(outbuf.data(), zout.pos))
            die_io("direct write failed (disk full?)");
        } else
#endif
        if (out) {   // out is null only in sink mode (handled above); guard silences -Wnonnull
          size_t w = robust_fwrite(outbuf.data(), zout.pos, out);
          if (w != zout.pos) die_io("short write to output (broken pipe?)");
        }
      }
      if (m) m->wrote_bytes.fetch_add(zout.pos);
      zout.pos = 0;
    }
    if (ret == 0 && zin.pos < zin.size) {
      // Frame boundary in multi-frame stream; continue
    }
  }

  // ret > 0 means ZSTD expected more input  the stream was truncated mid-frame.
  if (ret > 0)
    die_data("truncated zstd stream (expected more data)");

  ZSTD_freeDCtx(dctx);
}

// Streaming decompress of a large single-frame input (zstd /
// --sliding-window) read directly from the FILE.  A single zstd frame can't be
// split across threads (nor GPU subchunks), so the old fallback buffered the
// whole compressed frame into one Task and decompressed it single-threaded
// only after the reader finished — serialising read and decompress, spiking
// memory to input+frame+output, and freezing the progress bar.  Streaming here
// overlaps read + decompress + write, keeps peak memory to a couple of I/O
// buffers, and lets the meter move.  Output uses the same DirectWriter / fwrite
// path as decompress_from_buffer.  ZSTD_decompressStream also decodes the rare
// trailing-frames-after-a-large-first-frame case correctly.  Used only for
// seekable inputs whose first frame exceeds SINGLE_FRAME_STREAM_MIN.
static void decompress_stream_from_file(FILE * in, FILE * out,
                                        const Options & opt, Meter * m)
{
  const size_t IO_CHUNK = 4 * ONE_MIB;
  std::vector<char> inbuf(IO_CHUNK);
  std::vector<char> outbuf(IO_CHUNK);
  ZSTD_DCtx * dctx = ZSTD_createDCtx();
  if (!dctx) die("failed to create ZSTD_DCtx");
  apply_mem_limit_to_dctx(dctx, opt);

  size_t ret = 0;  // last decompressStream hint; >0 at EOF means truncated
  for (;;) {
    size_t n = std::fread(inbuf.data(), 1, IO_CHUNK, in);
    if (n == 0) break;  // EOF
    if (m) m->read_bytes.fetch_add(n, std::memory_order_relaxed);
    ZSTD_inBuffer zin { inbuf.data(), n, 0 };
    while (zin.pos < zin.size) {
      ZSTD_outBuffer zout { outbuf.data(), outbuf.size(), 0 };
      ret = ZSTD_decompressStream(dctx, &zout, &zin);
      if (ZSTD_isError(ret)) {
        ZSTD_freeDCtx(dctx);
        die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret)
               + "\n  (re-run with --keep-going to recover what is readable and report the damage)");
      }
      if (zout.pos > 0) {
        if (g_tar_decomp_sink) {
          // -t/-d --tar single-frame path: hand the chunk to the tar parser.
          g_tar_decomp_sink->push(std::make_shared<FrameVec>(outbuf.data(), outbuf.data() + zout.pos));
        } else if (opt.mode != Mode::TEST) {
#ifndef _WIN32
          if (g_direct_writer) {
            if (!g_direct_writer->write(outbuf.data(), zout.pos))
              die_io("direct write failed (disk full?)");
          } else
#endif
          if (out) {   // out is null only in sink mode (handled above); guard silences -Wnonnull
            size_t w = robust_fwrite(outbuf.data(), zout.pos, out);
            if (w != zout.pos) die_io("short write to output (broken pipe?)");
          }
        }
        if (m) m->wrote_bytes.fetch_add(zout.pos, std::memory_order_relaxed);
      }
    }
  }
  if (ret > 0) {
    ZSTD_freeDCtx(dctx);
    die_data("truncated zstd stream (expected more data)");
  }
  ZSTD_freeDCtx(dctx);
}
struct CpuThreadStats {
  uint64_t tasks    = 0;
  uint64_t in_bytes = 0;
  uint64_t out_bytes = 0;
  double   comp_ms  = 0.0;
};

// Aggregated CPU stats across all threads.
struct CpuAgg {
  std::mutex                   m;
  uint64_t                     tasks    = 0;
  uint64_t                     in_bytes = 0;
  uint64_t                     out_bytes = 0;
  double                       comp_ms  = 0.0;
  int                          threads  = 0;
  std::vector<CpuThreadStats>  per_thread;
};

#ifdef HAVE_NVCOMP
/*======================================================================
 Hybrid scheduler  GPU-priority semaphore model
 -----------------------------------------------------------------------
 Coordinates CPU and GPU workers sharing a single task queue.

 DESIGN: GPUs get priority via a "gpus_waiting" semaphore:
   - GPU stream calls gpu_wants_data() before popping → increments counter
   - GPU stream calls gpu_got_data() after popping → decrements counter
   - CPU workers check should_cpu_take():
       * Before GPU ready: CPU runs wild (100%)
       * gpus_waiting > 0: CPU yields (GPU needs data)
       * gpus_waiting == 0: CPU takes (all GPUs busy processing)

 This ensures GPUs are always fed first.  CPUs handle overflow when all
 GPU streams are busy processing their batches.

 The tick thread runs for monitoring/logging but does NOT control
 scheduling  the semaphore handles that in real-time.

 QUEUE INTERACTION: GPU workers use pop_batch_greedy(min_n=1) so multiple
 GPUs can interleave — each grabs whatever is available up to max_n without
 waiting for a full batch (waiting for full batches serialises GPUs).
 CPU workers use non-blocking try_pop_batch to avoid competing with GPU
 for the queue's condition variable wakeups.
======================================================================*/
class HybridSched {
public:
  HybridSched(double override_share, int cpu_threads, int gpu_devices,
              const Options & opt)
    : opt_(opt), gpu_device_count_(gpu_devices),
      cpu_thread_count_(std::max(1, cpu_threads))
  {
    if (override_share >= 0.0) {
      fixed_mode_ = true;
      fixed_cpu_share_ = override_share;
    }
  }

  // Set the task queue pointer so gpu_got_data() can wake CPU workers.
  // Must be called before GPU workers start.
  void set_queue(TaskQueue * tq) { queue_ = tq; }

  // CPU checks this before taking from the queue.
  // If any GPU stream is waiting for data, CPU yields.
  bool should_cpu_take() const {
    if (fixed_mode_) {
      // If no GPU stream is currently active — GPUs haven't registered yet, or
      // they all failed/exited mid-run — the fixed share is moot: nothing
      // advances gpu_taken_, so the share cap would stall the main CPU workers
      // until the producer finishes (the drain fast-path then recovers, but the
      // whole production phase is wasted).  Let CPU run unrestricted instead.
      // Adaptive mode already handles this via the gpus_waiting_/floor path.
      // During a healthy run active_gpu_streams_ > 0, so this is a no-op there.
      if (active_gpu_streams_.load(std::memory_order_relaxed) == 0)
        return true;
      const uint64_t cpu = cpu_taken_.load(std::memory_order_relaxed);
      const uint64_t gpu = gpu_taken_.load(std::memory_order_relaxed);
      const uint64_t total = cpu + gpu + 1;
      return (double(cpu) / double(total)) < (fixed_cpu_share_ + 0.02);
    }

    // Before any GPU is ready: CPU runs wild
    if (!gpu_ready_.load(std::memory_order_acquire))
      return true;

    // If a GPU stream is waiting for data: CPU yields
    if (gpus_waiting_.load(std::memory_order_acquire) > 0)
      return false;

    // No GPU is waiting right now — but check queue depth reservation.
    // GPUs cycle through wants→got in microseconds; during the much longer
    // processing phase (milliseconds) gpus_waiting==0 and all CPUs flood
    // the queue.  The queue floor reserves enough tasks for the next GPU
    // batch cycle so GPUs never find the queue empty.
    return true;
  }

  // GPU checks this before popping a batch.  In fixed-share mode, GPU
  // yields when CPU is below its target share — symmetric counterpart
  // to should_cpu_take().
  //
  // Adaptive mode (compress only, v0.13.57): tail-aware intake.  The old
  // "GPU always takes" policy let a slow GPU set the run's makespan:
  // profiled on a Gen3 2-GPU box (GPU pool ~1.1 GiB/s vs CPU pool ~15),
  // the GPU grabbed batches from the near-empty queue and the whole run
  // waited ~2s per batch for it while the CPU pool sat idle — hybrid
  // finished 26% behind cpu-only.  Mid-run, greedy intake is harmless
  // (work conserves; both pools stay busy), so the GPU yields only at
  // the tail: when the queue holds less CPU-time of work than ~1.3x the
  // time this GPU batch would take, starting the batch would outlive
  // the CPU drain and stretch the makespan.  Frames are uniform
  // (chunk_mib), so frame counts divided by the EMA byte-rates compare
  // directly.  The check arms only once the producer is done — before
  // that, queue depth measures the reader, not remaining work, and a
  // streaming producer would starve the GPU (with mmap input the
  // producer finishes near t=0, so in the default path the check is
  // live for effectively the whole run).  It never engages during EMA
  // warm-up (both engines need 2 samples) or for decompress (different
  // economics — not validated there).  The first yield latches
  // tail_yield_, which zeroes cpu_queue_floor() — otherwise CPUs would
  // refuse the very frames the GPU just declined (reserved by the
  // floor) and the tail would strand.
  bool should_gpu_take() {
    if (fixed_mode_) {
      const uint64_t cpu = cpu_taken_.load(std::memory_order_relaxed);
      const uint64_t gpu = gpu_taken_.load(std::memory_order_relaxed);
      const uint64_t total = cpu + gpu + 1;
      // GPU takes when CPU has met (or exceeded) its share.  Same hysteresis
      // band as should_cpu_take so the ratio oscillates around the target
      // instead of one side starving when both check at the same moment.
      return (double(cpu) / double(total)) >= (fixed_cpu_share_ - 0.02);
    }
    if (opt_.mode != Mode::COMPRESS) return true;
    if (!producer_done_.load(std::memory_order_acquire)) return true;
    if (!queue_) return true;
    return should_gpu_take_at(queue_->size());
  }

  // Core of the adaptive tail decision for a known queue depth.  Split out
  // so the wait_for_gpu_yield predicate can evaluate it under the queue
  // lock without calling back into TaskQueue (deadlock).  Only reached
  // after the should_gpu_take() gates (adaptive, compress, producer done),
  // all of which are monotonic — once a worker parks, they can't unflip.
  bool should_gpu_take_at(size_t depth_now) {
    if (cpu_samples_.load(std::memory_order_relaxed) < 2 ||
        gpu_samples_.load(std::memory_order_relaxed) < 2) return true;
    const double c = cpu_rate_ema_.load(std::memory_order_relaxed);
    const double g = gpu_rate_ema_.load(std::memory_order_relaxed);
    if (c <= 0.0 || g <= 0.0) return true;
    const double depth = (double)depth_now;
    // CPUs refuse depths below --cpu-queue-min, so the GPU must stay the
    // drain of last resort there or the run hangs with both sides yielding.
    if (opt_.cpu_queue_min > 0 && depth <= (double)opt_.cpu_queue_min)
      return true;
    const double batch =
        (double)std::max<size_t>(1, gpu_batch_size_.load(std::memory_order_relaxed));
    const double streams =
        (double)std::max(1, active_gpu_streams_.load(std::memory_order_relaxed));
    // CPU-seconds of work left after this batch vs GPU-seconds to finish it
    // (per-stream rate = pool EMA / streams).
    const bool take = (depth - batch) / c >= 1.3 * batch * streams / g;
    if (!take && !tail_yield_.exchange(true, std::memory_order_acq_rel)) {
      // First yield: the floor is now zero — wake CPUs sleeping on it,
      // since no push will arrive to re-evaluate their predicate.
      // (Plain CV notify — safe even when called under the queue lock
      // from the wait_for_gpu_yield predicate.)
      queue_->notify_cpu_waiters();
    }
    return take;
  }

  // Producer finished: queue depth now equals remaining work, so the
  // tail-aware GPU intake check in should_gpu_take() can arm.
  void set_producer_done() {
    producer_done_.store(true, std::memory_order_release);
  }

  // Producer with a bounded queue depth (pooled reader: pool size) declares
  // its ceiling so update_queue_floor's clamp can engage.  0 = unbounded.
  void set_queue_depth_cap(size_t n) {
    queue_depth_cap_.store(n, std::memory_order_relaxed);
    refresh_queue_floor_();
  }

  // Minimum queue depth below which CPUs should not take work.
  // Reserves enough tasks for all GPU streams to fill their next batch.
  // Zero once the GPU has tail-yielded: the reservation exists to feed
  // GPU batches, and the GPU has declared it won't take another.
  size_t cpu_queue_floor() const {
    if (tail_yield_.load(std::memory_order_relaxed)) return 0;
    return gpu_queue_floor_.load(std::memory_order_relaxed);
  }

  // GPU workers call this once per stream after init
  void register_gpu_stream() {
    int n = active_gpu_streams_.fetch_add(1, std::memory_order_relaxed) + 1;
    update_queue_floor(n, gpu_batch_size_.load(std::memory_order_relaxed));
  }

  // GPU workers call this when a stream is shutting down (trivial-skip exit,
  // normal completion, or failure).  Decrements active_gpu_streams_ and
  // recalculates the queue floor so CPU workers aren't blocked by a floor
  // that reserves frames for GPUs that no longer exist.
  void unregister_gpu_stream() {
    int n = active_gpu_streams_.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (n < 0) n = 0;
    update_queue_floor(n, gpu_batch_size_.load(std::memory_order_relaxed));
    if (queue_) queue_->notify_cpu_waiters();
  }

  // GPU workers call this when the shared batch size changes
  void set_gpu_batch_size(size_t bs) {
    gpu_batch_size_.store(bs, std::memory_order_relaxed);
    update_queue_floor(active_gpu_streams_.load(std::memory_order_relaxed), bs);
  }

  // GPU stream calls this when it's ready and waiting for data
  void gpu_wants_data() { gpus_waiting_.fetch_add(1, std::memory_order_release); }

  // GPU stream calls this after it has taken a batch from the queue.
  // Decrements the semaphore.  Only wakes CPU workers when the last
  // waiting GPU gets data (gpus_waiting_ drops to 0) — before that,
  // should_cpu_take() returns false anyway, so waking CPUs is wasted
  // work that creates thundering-herd lock contention on TaskQueue::m_.
  void gpu_got_data() {
    int prev = gpus_waiting_.fetch_sub(1, std::memory_order_release);
    // Only wake when the last waiting GPU gets data (semaphore hits 0).
    // Before that, should_cpu_take() returns false anyway.
    // Use notify_one — each CPU pops one task then loops back; waking
    // all 96 CPUs just causes 95 to contend on TaskQueue::m_ and sleep.
    // Safety: GPU workers call notify_cpu_waiters() (all) on exit, so
    // stragglers always get woken for the drain path.
    if (prev == 1 && queue_) queue_->notify_cpu_one();
  }

  // Called by GPU workers once CUDA context is initialized
  void set_gpu_ready(int device_id = -1) {
    gpu_ready_.store(true, std::memory_order_release);
    if (queue_) queue_->notify_cpu_waiters();  // wake CPUs to re-evaluate
    if (opt_.verbosity >= V_VERBOSE) {
      if (device_id >= 0)
        vlog(V_VERBOSE, opt_, "[GPU" + std::to_string(device_id) + "] ready, semaphore scheduling active\n");
      else
        vlog(V_VERBOSE, opt_, "[GPU] ready, semaphore scheduling active\n");
    }
  }

  void mark_cpu_take(uint64_t n) { cpu_taken_.fetch_add(n, std::memory_order_relaxed); }
  void mark_gpu_take(uint64_t n) { gpu_taken_.fetch_add(n, std::memory_order_relaxed); }
  void add_cpu_bytes(uint64_t b) { cpu_bytes_.fetch_add(b, std::memory_order_relaxed); }
  void add_gpu_bytes(uint64_t b) { gpu_bytes_.fetch_add(b, std::memory_order_relaxed); }

  // Tick runs for monitoring/logging
  void tick() {
    if (fixed_mode_) return;
    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_tick_).count();
    if (secs < 0.5) return;
    last_tick_ = now;

    const uint64_t cpu_b = cpu_bytes_.exchange(0, std::memory_order_relaxed);
    const uint64_t gpu_b = gpu_bytes_.exchange(0, std::memory_order_relaxed);
    const double cpu_rate = cpu_b / std::max(1e-6, secs);
    const double gpu_rate = gpu_b / std::max(1e-6, secs);

    // Feed EMAs (alpha = 0.3) for AUTO-mode floor scaling.  Only count
    // samples that carried real work; idle ticks shouldn't collapse the EMA.
    if (cpu_b > 0) {
      double prev = cpu_rate_ema_.load(std::memory_order_relaxed);
      double sample = cpu_rate / 1e9;  // GiB/s
      double next = (prev == 0.0) ? sample : prev * 0.7 + sample * 0.3;
      cpu_rate_ema_.store(next, std::memory_order_relaxed);
      cpu_samples_.fetch_add(1, std::memory_order_relaxed);
    }
    if (gpu_b > 0) {
      double prev = gpu_rate_ema_.load(std::memory_order_relaxed);
      double sample = gpu_rate / 1e9;  // GiB/s
      double next = (prev == 0.0) ? sample : prev * 0.7 + sample * 0.3;
      gpu_rate_ema_.store(next, std::memory_order_relaxed);
      gpu_samples_.fetch_add(1, std::memory_order_relaxed);
    }
    refresh_queue_floor_();
    // EMA movement is the one yield-decision input with no queue event;
    // wake any GPU parked in wait_for_gpu_yield so it re-evaluates.
    if (queue_) queue_->notify_gpu_yield_waiters();

    if (opt_.verbosity >= V_DEBUG) {
      const char * YEL = g_color_stderr ? "\033[1;93m" : "";  // bright yellow — GPU
      const char * MAG = g_color_stderr ? "\033[1;95m" : "";  // bright magenta — CPU
      const char * RST = g_color_stderr ? "\033[0m"    : "";
      std::ostringstream os;
      os << std::fixed << std::setprecision(3)
         << "[HYBRID] tick "
         << MAG << "cpu_rate=" << (cpu_rate/1e9) << RST << " GiB/s"
         << "  " << YEL << "gpu_rate=" << (gpu_rate/1e9) << RST << " GiB/s"
         << "  " << YEL << "gpus_waiting=" << gpus_waiting_.load() << RST
         << "  " << MAG << "cpu_taken=" << cpu_taken_.load() << RST
         << "  " << YEL << "gpu_taken=" << gpu_taken_.load() << RST
         << "  queue_floor=" << gpu_queue_floor_.load()
         << "  floor_factor=" << resolve_factor_();
      vlog(V_DEBUG, opt_, os.str() + "\n");
    }

    cpu_taken_.store(0, std::memory_order_relaxed);
    gpu_taken_.store(0, std::memory_order_relaxed);
  }

  double target_share() const { return 0.5; }
  bool is_gpu_ready() const { return gpu_ready_.load(std::memory_order_acquire); }
  bool is_fixed_mode() const { return fixed_mode_; }
  bool any_gpu_active() const {
    return active_gpu_streams_.load(std::memory_order_relaxed) > 0;
  }
  int active_gpu_streams() const {
    return active_gpu_streams_.load(std::memory_order_relaxed);
  }
  size_t queue_depth_cap() const {
    return queue_depth_cap_.load(std::memory_order_relaxed);
  }

private:
  const Options & opt_;
  int gpu_device_count_ = 0;
  int cpu_thread_count_ = 1;
  bool fixed_mode_ = false;
  double fixed_cpu_share_ = -1.0;
  TaskQueue * queue_ = nullptr;  // for waking CPU workers from gpu_got_data()

  std::atomic<bool> gpu_ready_{false};
  std::atomic<int> gpus_waiting_{0};
  std::atomic<bool> producer_done_{false};  // arms the tail-aware GPU intake check
  std::atomic<bool> tail_yield_{false};     // GPU declined the tail; floor released

  // Queue-depth reservation: CPUs yield when depth <= floor.
  // Nominal floor = active_gpu_streams * gpu_batch_size.
  // In AUTO mode, scaled by GPU/CPU throughput ratio (see compute_factor_).
  std::atomic<int> active_gpu_streams_{0};
  std::atomic<size_t> gpu_batch_size_{8};   // initial, updated by auto-tuner
  std::atomic<size_t> gpu_queue_floor_{0};  // computed floor in effect

  // EMA throughput samples for AUTO mode (v0.12.12+).
  // Stored as bit-casted doubles so other paths can read lock-free.
  std::atomic<double> cpu_rate_ema_{0.0};  // GiB/s across all CPU threads
  std::atomic<double> gpu_rate_ema_{0.0};  // GiB/s across all GPU streams
  std::atomic<int>    cpu_samples_{0};     // warm-up counter
  std::atomic<int>    gpu_samples_{0};     // warm-up counter

  // Compute the floor scale factor for AUTO mode.
  //
  // Policy: "GPU first, CPU as surplus."  CPU can only take a frame when
  // the queue has at least factor * streams * batch frames buffered ahead
  // of it, guaranteeing GPU never finds the queue shallow when it asks
  // for its next batch.  The old per-worker-rate formula (v0.12.12) was
  // too small on high-core-count multi-GPU systems where the per-worker
  // GPU and CPU rates are similar — it produced factor ~0.15 and CPU
  // drained the queue, shrinking GPU batches and dragging hybrid below
  // --gpu-only.
  //
  // New formula keys off CPU's *share of aggregate throughput*:
  //   share < 5%  : CPU not contributing meaningfully → factor = 4.0
  //                 (heavy lockout; hybrid converges to gpu-only)
  //   share > 20% : CPU genuinely helping            → factor = 1.5
  //                 (still reserve one full batch + half, but let CPU work)
  //   in between  : linear interpolation
  //
  // Range [1.5, 4.0] guarantees the floor is always at least one full
  // GPU round, so a CPU pop can never leave the next GPU batch short.
  // Warm-up returns 2.0 — proactive reserve before any measurement.
  double compute_auto_factor_() const {
    int cs = cpu_samples_.load(std::memory_order_relaxed);
    int gs = gpu_samples_.load(std::memory_order_relaxed);
    if (cs < 2 || gs < 2) return 2.0;  // warm-up: proactive reserve
    double c_rate = cpu_rate_ema_.load(std::memory_order_relaxed);
    double g_rate = gpu_rate_ema_.load(std::memory_order_relaxed);
    double total = c_rate + g_rate;
    if (total <= 0.0) return 2.0;
    double cpu_share = c_rate / total;
    if (cpu_share >= 0.20) return 1.5;
    if (cpu_share <= 0.05) return 4.0;
    // Linear interp between (share=0.05, factor=4.0) and (share=0.20, factor=1.5)
    double t = (cpu_share - 0.05) / (0.20 - 0.05);
    return 4.0 - t * (4.0 - 1.5);
  }

  // Resolve the active floor factor from Options + runtime EMA state.
  double resolve_factor_() const {
    if (opt_.hybrid_floor_factor >= 0.0) return opt_.hybrid_floor_factor;
    switch (opt_.hybrid_floor_mode) {
      case Options::HybridFloorMode::OFF:     return 0.0;
      case Options::HybridFloorMode::NOMINAL: return 1.0;
      case Options::HybridFloorMode::AUTO:
      default:                                return compute_auto_factor_();
    }
  }

  void update_queue_floor(int streams, size_t batch) {
    size_t nominal = size_t(std::max(0, streams)) * batch;
    double factor = resolve_factor_();
    size_t floor = (size_t)(double(nominal) * factor + 0.5);
    // The floor predates the pooled reader, when mmap enqueued the whole
    // file and queue depth was effectively unbounded.  With a bounded
    // producer (pool size caps depth), GPU appetite (streams × batch) can
    // exceed the reader's supply rate so depth NEVER builds: any
    // substantial floor then locks the CPU pool out permanently — and the
    // AUTO mode latches (starved CPU ⇒ <5% share ⇒ factor=4 ⇒ starved CPU).
    // Measured on the 8-GPU server (v0.13.66 floor sweep): auto 15.74
    // GiB/s with GPU taking 92% of frames; even factor=0.5 locked at 90%;
    // floor OFF rebalanced to 37% GPU and 18.17 GiB/s ≈ cpu-only.  The
    // reservation has no purpose under continuous refill (GPUs pop with
    // min_n=1 and the auto-tuner adapts batch size), so a bounded producer
    // zeroes the AUTO floor; an explicit --hybrid-floor=nominal or
    // --hybrid-floor-factor is honored but still clamped to a quarter of
    // the ceiling so it cannot re-create the permanent lockout.
    const size_t cap = queue_depth_cap_.load(std::memory_order_relaxed);
    if (cap > 0) {
      const bool user_floor =
          opt_.hybrid_floor_factor >= 0.0 ||
          opt_.hybrid_floor_mode != Options::HybridFloorMode::AUTO;
      floor = user_floor ? std::min(floor, cap / 4) : 0;
    }
    gpu_queue_floor_.store(floor, std::memory_order_relaxed);
  }

  // Called from tick() so EMA changes immediately take effect.
  void refresh_queue_floor_() {
    update_queue_floor(active_gpu_streams_.load(std::memory_order_relaxed),
                       gpu_batch_size_.load(std::memory_order_relaxed));
  }

  std::atomic<uint64_t> cpu_taken_{0};
  std::atomic<uint64_t> gpu_taken_{0};
  std::atomic<uint64_t> cpu_bytes_{0};
  std::atomic<uint64_t> gpu_bytes_{0};
  std::atomic<size_t>   queue_depth_cap_{0};  // producer's depth ceiling (pooled reader); 0 = unbounded
  std::chrono::steady_clock::time_point last_tick_ = std::chrono::steady_clock::now();
};

// Background thread that periodically calls sched->tick() to update the
// adaptive CPU/GPU work-share ratio based on observed throughput.
static void tick_loop_fn(std::atomic<bool> & done, HybridSched * sched)
{
  using namespace std::chrono_literals;
  while (!done.load()) {
    std::this_thread::sleep_for(100ms);
    sched->tick();
  }
}
#endif

/*======================================================================
 CPU workers (normal + rescue)
======================================================================*/
struct RateMatchState {
  // GPU throughput (bytes/sec, smoothed)
  std::atomic<double> gpu_thr{0.0};     // GiB/s across all GPUs combined
  std::atomic<double> cpu_thr{0.0};     // GiB/s across all CPU threads combined

  // Accumulators for windowed measurement
  std::atomic<uint64_t> gpu_window_bytes{0};
  std::atomic<uint64_t> gpu_window_ns{0};
  std::atomic<uint64_t> cpu_window_bytes{0};
  std::atomic<uint64_t> cpu_window_ns{0};
  std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::now();
  std::mutex update_mtx;

  // CPU frame allowance: how many frames CPUs should take per cycle
  // Updated each time throughput is measured
  std::atomic<int> cpu_frame_allowance{4};  // start conservative
  std::atomic<int> cpu_frames_taken{0};     // frames taken in current cycle

  void report_gpu(uint64_t bytes, uint64_t ns) {
    gpu_window_bytes.fetch_add(bytes, std::memory_order_relaxed);
    gpu_window_ns.fetch_add(ns, std::memory_order_relaxed);
  }

  void report_cpu(uint64_t bytes, uint64_t ns) {
    cpu_window_bytes.fetch_add(bytes, std::memory_order_relaxed);
    cpu_window_ns.fetch_add(ns, std::memory_order_relaxed);
  }

  // Called periodically (e.g., every 0.5s) to update throughput estimates
  // and recalculate CPU frame allowance.
  void update(size_t gpu_batch_size, size_t avg_frame_bytes) {
    std::unique_lock<std::mutex> lk(update_mtx, std::try_to_lock);
    if (!lk.owns_lock()) return;

    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(
        now - last_update).count();
    if (secs < 0.3) return;
    last_update = now;

    uint64_t gb = gpu_window_bytes.exchange(0);
    uint64_t gn = gpu_window_ns.exchange(0);
    uint64_t cb = cpu_window_bytes.exchange(0);
    uint64_t cn = cpu_window_ns.exchange(0);

    // Smoothed throughput (EMA alpha=0.3)
    if (gn > 0) {
      double g = double(gb) / (double(gn) / 1e9) / 1e9;  // GiB/s
      double old = gpu_thr.load(std::memory_order_relaxed);
      gpu_thr.store(old * 0.7 + g * 0.3, std::memory_order_relaxed);
    }
    if (cn > 0) {
      double c = double(cb) / (double(cn) / 1e9) / 1e9;  // GiB/s
      double old = cpu_thr.load(std::memory_order_relaxed);
      cpu_thr.store(old * 0.7 + c * 0.3, std::memory_order_relaxed);
    }

    // Calculate CPU frame allowance: how many frames should CPUs process
    // while GPUs process one batch, so both finish at the same time
    double g_thr = gpu_thr.load(std::memory_order_relaxed);
    double c_thr = cpu_thr.load(std::memory_order_relaxed);
    if (g_thr > 0 && c_thr > 0 && avg_frame_bytes > 0) {
      // GPU batch time = batch_size * frame_size / gpu_throughput
      double gpu_batch_sec = double(gpu_batch_size * avg_frame_bytes) / (g_thr * 1e9);
      // CPU frames in that time = gpu_batch_time * cpu_throughput / frame_size
      int allowance = (int)(gpu_batch_sec * c_thr * 1e9 / double(avg_frame_bytes));
      allowance = std::max(1, std::min(allowance, 1000));  // sanity bounds
      cpu_frame_allowance.store(allowance, std::memory_order_relaxed);
    }
    cpu_frames_taken.store(0, std::memory_order_relaxed);
  }

  // Reset cycle counter (called when GPU batch completes)
  void reset_cycle() {
    cpu_frames_taken.store(0, std::memory_order_relaxed);
  }
};

static void cpu_worker(
  int worker_id,
  TaskQueue * tq,
  ResultStore * results,
  const Options * opt,
  Meter * m,
#ifdef HAVE_NVCOMP
  void * sched_ptr,
  RateMatchState * rate_match,
#endif
  CpuAgg * cpuagg,
  FrameThrottle * bp)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  // Per-thread reusable scratch buffer.  compress_one_cpu_frame's resize()
  // grows this once on the first iteration (paying ~bound bytes of page
  // faults) and reuses the same resident pages on every subsequent call.
  // Without this, each iteration allocates a fresh 16+ MiB vector, faults
  // every page, then frees it — and with 96 worker threads all doing this
  // in lockstep during GPU init, they serialize on mmap_lock and burn ~64%
  // of CPU cycles inside the kernel's page-fault handler.  See v0.13.7
  // CHANGELOG for the perf-record evidence.
  FrameVec scratch;

  // Per-thread bounded output-frame pool.  Same backpressure design as
  // cpu_decomp_worker (see comment there) — pool size derived from the
  // throttle so it auto-adapts; never grows; worker yields when full.
  // The compress path's per-push allocation is csz bytes (compressed
  // output size).  For compressible data csz is tiny and this fix is
  // ~free; for poorly compressible data csz ≈ src_size and the pool
  // prevents the same mmap_lock storm we fixed for decompress.
  const int n_workers = std::max(1, cpuagg ? cpuagg->threads : 1);
  const int throttle_budget = bp ? bp->max_permits() : 512;
  const int pool_size = std::max(2, throttle_budget / n_workers);
  std::vector<FrameBuf> out_pool(pool_size);
  for (auto & b : out_pool) b = std::make_shared<FrameVec>();
  uint64_t pool_wait_count = 0;

  auto acquire_out_buf = [&]() -> FrameBuf {
    while (true) {
      // GPU-fault abort: the writer has stopped draining, so no slot will ever
      // free — return any buffer; the caller exits at the next loop top.
      if (g_gpu_aborted.load(std::memory_order_relaxed)) return out_pool[0];
      for (auto & b : out_pool) {
        if (b.use_count() == 1) return b;
      }
      ++pool_wait_count;
      // Wait on the writer's drain signal instead of sched_yield.
      // Predicate checks for a free slot — wait_for() re-checks after
      // wake, so a single notify_all is sufficient even if 96 workers
      // race for ~5 slots.  The 10ms timeout in wait_for_drain is a
      // safety net for any missed notify.
      if (bp) {
        bp->wait_for_drain([&]{
          if (g_gpu_aborted.load(std::memory_order_relaxed)) return true;
          for (auto & b : out_pool) if (b.use_count() == 1) return true;
          return false;
        });
      } else {
        std::this_thread::yield();  // no throttle → fall back to yield
      }
    }
  };
  while (true) {
    if (g_gpu_aborted.load(std::memory_order_relaxed)) break;  // GPU fault: abort without draining the queue
    Task t;
    bool got_task = false;
#ifdef HAVE_NVCOMP
    if (sched) {
      // Hybrid mode: acquire-before-pop with release-before-sleep.
      //
      // Why this ordering matters:
      //   acquire-AFTER-pop deadlocks when a worker pops the frame the writer
      //   needs but can't get a permit (permits exhausted by ResultStore).
      //   acquire-BEFORE-pop with blocking wait hoards permits while sleeping.
      //
      // Solution: wait (no permit) → acquire → try_pop (non-blocking).
      // If try_pop fails (another worker grabbed it or predicate changed),
      // release the permit and retry.  Workers only hold permits while
      // actively processing — never while sleeping on the CV.
      auto may_take = [&](const TaskQueue::QueueState & qs) -> bool {
        // Drain-phase fast path: only when no GPU is left to do the work.
        // While any GPU stream is still active, honor the floor regardless
        // of phase — small files where mmap+page-cache finish the producer
        // in ~1s otherwise let CPU flood the post-done queue, shrinking
        // GPU batches and dragging hybrid below gpu-only.  When all GPU
        // streams have unregistered (failure or normal exit), CPU drains.
        if (qs.done && !sched->any_gpu_active()) return true;
        if (!sched->should_cpu_take()) return false;
        // Floor applies only in adaptive (non-fixed) mode.  Fixed-share
        // mode is the user's explicit intent and would deadlock here:
        // should_gpu_take requires cpu_share >= target - 2%, so locking
        // CPU out via the floor stops GPU too (both predicates fail).
        if (!sched->is_fixed_mode()) {
          // Reserve enough tasks for GPUs to fill their next batch cycle.
          // Without this, CPUs drain the queue during GPU processing and
          // GPUs find it empty when they come back for more.
          size_t floor = sched->cpu_queue_floor();
          if (floor > 0 && qs.depth <= floor) return false;
        }
        if (opt->cpu_queue_min > 0 && qs.depth < opt->cpu_queue_min) return false;
        return true;
      };
      bool drained = false;
      while (true) {
        // Step 1: wait for predicate (no permit held → no hoarding)
        if (!tq->wait_for_cpu(may_take)) { drained = true; break; }
        // Step 2: acquire permit (may block, but no task held → writer can progress)
        if (bp) bp->acquire(1);
        // Step 3: non-blocking pop (state may have changed since wait)
        int rc = tq->try_pop_one_cpu(t, may_take);
        if (rc == 1) { got_task = true; break; }   // success: have task + permit
        if (bp) bp->release(1);                     // release unused permit
        if (rc == -1) { drained = true; break; }    // queue drained
        // rc == 0: predicate changed between wait and try — retry
      }
      if (drained) break;
      if (got_task) sched->mark_cpu_take(1);
    } else
#endif
    {
      // Non-hybrid: same wait-acquire-try pattern to prevent permit hoarding.
      if (opt->cpu_queue_min > 0) {
        auto threshold_met = [&](const TaskQueue::QueueState & qs) -> bool {
          return qs.depth >= opt->cpu_queue_min || qs.done;
        };
        if (!tq->wait_for_cpu(threshold_met)) break;
        if (tq->drained() && tq->size() == 0) break;
      }
      bool drained = false;
      while (true) {
        if (!tq->wait_for_work()) { drained = true; break; }
        if (bp) bp->acquire(1);
        int rc = tq->try_pop_one(t);
        if (rc == 1) { got_task = true; break; }
        if (bp) bp->release(1);
        if (rc == -1) { drained = true; break; }
      }
      if (drained) break;
    }
    if (!got_task) break;

    // Permit already acquired above (before pop).
    // The writer releases this permit after the output is physically written.

    // Count mmap views here (their reader doesn't); --direct-read views
    // (direct_buf>=0) are already counted by the O_DIRECT reader — skip them.
    if (m && t.view_ptr && t.direct_buf < 0) m->read_bytes.fetch_add(t.len(), std::memory_order_relaxed);

    if (opt->verbosity >= V_DEBUG) {
      char in_s[32];
      human_bytes(double(t.len()), in_s, sizeof(in_s));
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] take seq=" << t.seq << " in=" << in_s;
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t csz = compress_one_cpu_frame(t.ptr(), t.len(), opt->level, opt->ultra, scratch);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast< std::chrono::duration<double, std::milli> >(t1 - t0).count();
    const size_t in_size = t.len();
    t.release_input();
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(uint64_t(ms * 1e6));
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(in_size);
      g_perf->sched_cpu_tasks.fetch_add(1);
    }
#ifdef HAVE_NVCOMP
    if (rate_match) {
      rate_match->report_cpu(in_size, uint64_t(ms * 1e6));
    }
#endif

    if (opt->verbosity >= V_DEBUG) {
      char in_s[32], out_s[32];
      human_bytes(double(in_size), in_s, sizeof(in_s));
      human_bytes(double(csz), out_s, sizeof(out_s));
      const double thr_gib = (ms > 0.0) ? (double)in_size / (ms/1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    // Deliver compressed bytes via a pooled buffer.  For poorly-compressible
    // output (csz a large fraction of the input) the memcpy below would copy a
    // near-full chunk per frame, so instead swap the scratch buffer straight
    // into the pooled FrameBuf (zero-copy) and take the pool slot's old buffer
    // as the next scratch.  For well-compressible output csz is small: the copy
    // is cheap and we keep pooled buffers right-sized (swapping there would
    // leave every slot carrying scratch's full compressBound capacity).
    // Threshold: output >= half the input.  (ROADMAP 7.4.)
    auto out_frame = acquire_out_buf();
    if (csz * 2 >= in_size) {
      std::swap(*out_frame, scratch);   // out_frame takes scratch's buffer (sized to bound)
      out_frame->resize(csz);           // shrink to the actual compressed size (shrink: no zeroing)
    } else {
      // assign() copy-constructs csz bytes straight from scratch — unlike
      // resize(csz)+memcpy it never value-initializes (zeroes) the grown region
      // only to overwrite it.  Same bytes, no wasted memset.
      out_frame->assign(scratch.data(), scratch.data() + csz);
    }
    results->push_to_slot(-1, t.seq, std::move(out_frame));
#ifdef HAVE_NVCOMP
    if (sched) sched->add_cpu_bytes(in_size);
#endif
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if (cpuagg->per_thread.size() <= (size_t)worker_id)
        cpuagg->per_thread.resize((size_t)worker_id + 1);
      auto & st = cpuagg->per_thread[(size_t)worker_id];
      st.tasks    += 1;
      st.in_bytes += in_size;
      st.out_bytes += csz;
      st.comp_ms  += ms;
      cpuagg->tasks    += 1;
      cpuagg->in_bytes += in_size;
      cpuagg->out_bytes += csz;
      cpuagg->comp_ms  += ms;
    }
    } // end single-frame processing block
  }

  if (opt->verbosity >= V_DEBUG) {
    CpuThreadStats st;
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if ((size_t)worker_id < cpuagg->per_thread.size())
        st = cpuagg->per_thread[(size_t)worker_id];
    }
    if (st.tasks == 0) {
      // Idle threads only shown at V_TRACE to avoid flooding
      if (opt->verbosity >= V_TRACE) {
        std::ostringstream os;
        os << "[CPU/T" << worker_id << "] idle (0 tasks)";
        vlog(V_TRACE, *opt, os.str() + "\n");
      }
    } else {
      const double thr_gib = (st.comp_ms > 0.0) ? (double)st.in_bytes / (st.comp_ms/1000.0) / 1e9 : 0.0;
      char in_s[32], out_s[32];
      human_bytes(double(st.in_bytes),  in_s,  sizeof(in_s));
      human_bytes(double(st.out_bytes), out_s, sizeof(out_s));
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] total tasks=" << st.tasks
         << " in=" << in_s << " out=" << out_s
         << " time=" << std::fixed << std::setprecision(2) << st.comp_ms << "ms"
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s"
         << " pool=" << pool_size << " waits=" << pool_wait_count;
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }
  }
}

/*======================================================================
 CPU decompression worker (for MT/hybrid decompress)
======================================================================*/
// Decompress a single Zstd frame on the CPU.  Each task contains one complete
// compressed frame and the expected decompressed size from the frame header.
// Uses a thread-local DCtx to avoid repeated allocation/destruction overhead.
static thread_local ZSTD_DCtx * tl_dctx = nullptr;

static void cpu_decomp_worker(
  int worker_id,
  TaskQueue * tq,
  ResultStore * results,
  const Options * opt,
  Meter * m,
#ifdef HAVE_NVCOMP
  void * sched_ptr,
  RateMatchState * rate_match,
#endif
  CpuAgg * cpuagg,
  FrameThrottle * bp)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  // Create a reusable decompression context for this thread
  if (!tl_dctx) {
    tl_dctx = ZSTD_createDCtx();
    if (!tl_dctx) die("failed to create ZSTD_DCtx");
    apply_mem_limit_to_dctx(tl_dctx, *opt);
  }

  // Per-thread decompress-buffer pool.  BOUNDED size; never grows past
  // pool_size.  When all slots are in flight (use_count > 1 — writer
  // still holds a ref), the worker yields until a slot becomes free.
  //
  // Pool size derived from the throttle budget so it auto-adapts to
  // --throttle-frames: pool_size = max(2, throttle_budget / N_workers).
  // The min of 2 guarantees pipelining (one frame in flight + one being
  // worked on); above that, the throttle's global cap is divided across
  // workers.  When writer is fast (frames drain quickly) the worker
  // never blocks; when writer is slow, the worker waits at acquire()
  // instead of allocating a fresh 16-64 MiB buffer.  This routes the
  // existing backpressure (FrameThrottle → writer-bound) into per-worker
  // backpressure without growing memory or storming mmap_lock.
  //
  // Discovered in v0.13.9 via perf record on cpu-only decompress:
  // unbounded pool grew to ~5 slots per worker on 96-worker decompress
  // because the writer (single-threaded, ~5 GiB/s on sparse zeros) could
  // never drain fast enough.  Each new slot was a fresh 64 MiB alloc;
  // 96 workers × ~5 fresh allocs × 4 KiB pages = ~2.3M page faults,
  // serialized on the per-process mmap_lock.
  const int n_workers = std::max(1, cpuagg ? cpuagg->threads : 1);
  const int throttle_budget = bp ? bp->max_permits() : 512;
  const int pool_size = std::max(2, throttle_budget / n_workers);
  std::vector<FrameBuf> decomp_pool(pool_size);
  for (auto & b : decomp_pool) b = std::make_shared<FrameVec>();
  uint64_t pool_wait_count = 0;  // telemetry — # of yields at -vv

  auto acquire_decomp_buf = [&]() -> FrameBuf {
    while (true) {
      for (auto & b : decomp_pool) {
        if (b.use_count() == 1) return b;  // only ref is ours; safe to reuse
      }
      // All slots in flight — writer hasn't drained any yet.  Wait on
      // the writer's drain signal (see FrameThrottle::wait_for_drain),
      // not sched_yield — yielding 96 workers in lockstep burned ~50s
      // of sys time on the v0.13.9 256-core decompress runs.
      ++pool_wait_count;
      if (bp) {
        bp->wait_for_drain([&]{
          for (auto & b : decomp_pool) if (b.use_count() == 1) return true;
          return false;
        });
      } else {
        std::this_thread::yield();  // no throttle → fall back to yield
      }
    }
  };

  while (true) {
    Task t{};
    bool got_task = false;
#ifdef HAVE_NVCOMP
    if (sched) {
      // Hybrid decompress: acquire-before-pop with release-before-sleep.
      // Same pattern as cpu_worker — see comments there.
      bool got_trivial = false;
      auto may_take = [&](const TaskQueue::QueueState & qs) -> bool {
        // Always allow trivially-compressed frames regardless of scheduler
        if (qs.front_ratio >= 0.0 && qs.front_ratio < 0.02) { got_trivial = true; return true; }
        got_trivial = false;
        // Drain-phase fast path: only when no GPU is left to do the work.
        // While any GPU stream is active, honor the floor regardless of phase
        // so small-file runs (producer finishes in <1s) don't let CPU flood
        // the post-done queue and shrink GPU batches.
        if (qs.done && !sched->any_gpu_active()) return true;
        // Normal scheduling: respect GPU priority and queue depth threshold
        if (!sched->should_cpu_take()) return false;
        // Floor only in adaptive mode — fixed-share would deadlock here.
        if (!sched->is_fixed_mode()) {
          size_t floor = sched->cpu_queue_floor();
          if (floor > 0 && qs.depth <= floor) return false;
        }
        if (opt->cpu_queue_min > 0 && qs.depth < opt->cpu_queue_min) return false;
        return true;
      };
      bool drained = false;
      while (true) {
        if (!tq->wait_for_cpu(may_take)) { drained = true; break; }
        if (bp) bp->acquire(1);
        int rc = tq->try_pop_one_cpu(t, may_take);
        if (rc == 1) { got_task = true; break; }
        if (bp) bp->release(1);
        if (rc == -1) { drained = true; break; }
      }
      if (drained) break;
      if (got_task) {
        sched->mark_cpu_take(1);
        if (got_trivial && opt->verbosity >= V_DEBUG) {
          double ratio = (t.decomp_size > 0)
                         ? double(t.len()) / double(t.decomp_size) : 0.0;
          std::ostringstream os;
          os << "[CPU/T" << worker_id << "] trivial frame (ratio="
             << std::fixed << std::setprecision(3) << (ratio * 100.0) << "%)";
          vlog(V_DEBUG, *opt, os.str() + "\n");
        }
      }
    } else
#endif
    {
      // Non-hybrid: same wait-acquire-try pattern.
      if (opt->cpu_queue_min > 0) {
        auto threshold_met = [&](const TaskQueue::QueueState & qs) -> bool {
          return qs.depth >= opt->cpu_queue_min || qs.done;
        };
        if (!tq->wait_for_cpu(threshold_met)) break;
        if (tq->drained() && tq->size() == 0) break;
      }
      bool drained = false;
      while (true) {
        if (!tq->wait_for_work()) { drained = true; break; }
        if (bp) bp->acquire(1);
        int rc = tq->try_pop_one(t);
        if (rc == 1) { got_task = true; break; }
        if (bp) bp->release(1);
        if (rc == -1) { drained = true; break; }
      }
      if (drained) break;
    }
    if (!got_task) break;

    // Permit already acquired above (before pop).

    // Per-task take line (mirrors cpu_worker for compress).  Fires at -vv.
    if (opt->verbosity >= V_DEBUG) {
      char in_s[32], ds_s[32];
      human_bytes(double(t.len()), in_s, sizeof(in_s));
      human_bytes(double(t.decomp_size), ds_s, sizeof(ds_s));
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] take seq=" << t.seq
         << " comp=" << in_s << " decomp=" << ds_s;
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    const auto t0_w = std::chrono::steady_clock::now();
    const size_t comp_size = t.len();
    size_t actual;

    // For oversized frames (e.g., single-frame files from zstd), use
    // streaming decompression with chunked output so the writer can make
    // progress and the progress bar updates incrementally.
    //
    // SAFETY: streaming reuses seq numbers (chunk_seq starts at t.seq and
    // ascends), which collides with adjacent frames' natural seqs in
    // multi-frame files (e.g., --ultra -22 produces 128 MiB frames, all
    // would stream and clobber each other).  Only safe when there is
    // exactly one frame total.  v0.13.1 fix.
    //
    // Only frame 0 can possibly be "the single frame", and a second pushed
    // task disproves single-frame instantly — so most oversize frames skip
    // the wait entirely (faster than the old wait-for-producer-done).  When
    // we do have to wait, it MUST be a timed wait that re-checks the task
    // count: the input queue is bounded (v0.13.29), so the producer can be
    // blocked in push() while every worker sits here — waiting on
    // producer_done alone deadlocked low-thread-count decompression of
    // multi-frame files with >64 MiB frames (fixed v0.13.54).
    static constexpr size_t STREAM_THRESHOLD = 64 * ONE_MIB;
    bool use_streaming = false;
    if (t.decomp_size > STREAM_THRESHOLD && t.seq == 0 && tq->total_tasks() <= 1) {
      std::unique_lock<std::mutex> lk(results->m);
      while (!results->producer_done && tq->total_tasks() <= 1)
        results->cv.wait_for(lk, std::chrono::milliseconds(20));
      use_streaming = (tq->total_tasks() == 1);
    }
    if (opt->keep_going) {
      // --keep-going recovery decode (one-shot).  The one-shot decoder writes the
      // WHOLE frame before validating the trailing checksum, so a checksum-mismatch
      // frame is recovered in full (unverified) and — crucially — keeps its exact
      // length, so nothing downstream (especially --tar member boundaries) desyncs.
      // We zero the buffer first so any tail a hard-corruption decode never reached
      // holds zeros, not stale pool data, and still emit the full declared length
      // to preserve alignment.  The DCtx is reused across frames, so reset it.
      ZSTD_DCtx_reset(tl_dctx, ZSTD_reset_session_only);
      auto out_buf = acquire_decomp_buf();
      try {
        out_buf->resize(t.decomp_size);
      } catch (const std::bad_alloc &) {
        die_data("frame " + std::to_string(t.seq) + " header claims "
                 + std::to_string(t.decomp_size) + " bytes; allocation failed — the frame header "
                 "is corrupt and --keep-going cannot recover past an un-allocatable frame");
      }
      std::memset(out_buf->data(), 0, out_buf->size());
      size_t rc = ZSTD_decompressDCtx(tl_dctx, out_buf->data(), out_buf->size(),
                                      t.ptr(), t.len());
      actual = t.decomp_size;            // always emit the full declared length (alignment)
      t.release_input();
      if (m) m->read_bytes.fetch_add(comp_size, std::memory_order_relaxed);
      if (ZSTD_isError(rc)) {
        const bool checksum = (ZSTD_getErrorCode(rc) == ZSTD_error_checksum_wrong);
        // checksum mismatch → full content present but UNVERIFIED.  Any other
        // error → the frame can't be trusted; keep the full-length buffer (valid
        // prefix + zeros) for alignment but report it as unrecovered.
        g_damage.record(t.out_off, checksum ? (uint64_t)t.decomp_size : 0, t.decomp_size);
        std::ostringstream os;
        os << "WARNING: --keep-going: frame " << t.seq << " (output bytes "
           << t.out_off << ".." << (t.out_off + t.decomp_size) << "): ";
        if (checksum) os << "checksum mismatch — content recovered but UNVERIFIED";
        else          os << "could not decode (" << ZSTD_getErrorName(rc) << ") — data unreliable";
        vlog(V_ERROR, *opt, os.str() + "\n");
      }
      results->push_to_slot(-1, t.seq, std::move(out_buf));
    } else if (use_streaming) {
      static constexpr size_t CHUNK = 16 * ONE_MIB;
      size_t n_chunks_est = (t.decomp_size + CHUNK - 1) / CHUNK;
      if (n_chunks_est < 1) n_chunks_est = 1;
      {
        std::lock_guard<std::mutex> lk(results->m);
        results->total_tasks += (n_chunks_est - 1);
      }
      results->cv.notify_all();

      ZSTD_inBuffer zin { t.ptr(), t.len(), 0 };
      actual = 0;
      size_t chunk_seq = t.seq;
      size_t prev_zin_pos = 0;
      for (;;) {
        // Reuse the same pool — streaming chunks are bounded by CHUNK,
        // which fits comfortably within the per-frame max we'd otherwise
        // see.  Buffers are recycled through the writer the same way.
        auto chunk = acquire_decomp_buf();
        chunk->resize(CHUNK);
        ZSTD_outBuffer zout { chunk->data(), chunk->size(), 0 };
        size_t zin_before = zin.pos;
        size_t ret = ZSTD_decompressStream(tl_dctx, &zout, &zin);
        if (ZSTD_isError(ret))
          die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret)
               + "\n  (re-run with --keep-going to recover what is readable and report the damage)");
        actual += zout.pos;
        if (m && zin.pos > prev_zin_pos) {
          m->read_bytes.fetch_add(zin.pos - prev_zin_pos, std::memory_order_relaxed);
          prev_zin_pos = zin.pos;
        }
        if (zout.pos > 0) {
          chunk->resize(zout.pos);
          // Each streamed chunk becomes an in-flight result frame, and the
          // writer releases one throttle permit per frame it writes.  We
          // entered holding exactly one permit (the pre-pop acquire): charge it
          // to the first chunk, then acquire one more per subsequent chunk so
          // acquires match the writer's releases.  Without this the writer
          // over-released by (actual_chunks - 1), drifting permits_ above the
          // cap.  Deadlock-free: chunks ascend from the lowest seq, so the
          // writer always drains the oldest first and frees a permit.
          if (chunk_seq != t.seq && bp) bp->acquire(1);
          results->push_to_slot(-1, chunk_seq, std::move(chunk));
          ++chunk_seq;
        }
        if (ret == 0) break;
        if (zin.pos == zin_before && zout.pos == 0)
          die_data("ZSTD decompressStream stalled: no progress");
      }
      size_t actual_chunks = chunk_seq - t.seq;
      if (actual_chunks != n_chunks_est) {
        std::lock_guard<std::mutex> lk(results->m);
        results->total_tasks += (actual_chunks - n_chunks_est);
        results->cv.notify_all();
      }
      t.release_input();
    } else {
      // Acquire a buffer from the per-thread pool.  First use grows it
      // (page faults); subsequent uses reuse resident pages and just
      // memset the prefix during resize.  The pool gives the buffer back
      // to us once the writer drops its shared_ptr ref.
      auto out_buf = acquire_decomp_buf();
      try {
        out_buf->resize(t.decomp_size);
      } catch (const std::bad_alloc &) {
        // A corrupt/hostile frame header can claim an absurd content size;
        // fail as a data error instead of an uncaught-exception abort.
        die_data("frame " + std::to_string(t.seq) + " header claims "
                 + std::to_string(t.decomp_size)
                 + " bytes; allocation failed (corrupt input?)");
      }
      actual = ZSTD_decompressDCtx(tl_dctx, out_buf->data(), out_buf->size(),
                                   t.ptr(), t.len());
      if (ZSTD_isError(actual))
        die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(actual)
                 + "\n  (re-run with --keep-going to recover what is readable and report the damage)");
      out_buf->resize(actual);
      t.release_input();
      if (m) m->read_bytes.fetch_add(comp_size, std::memory_order_relaxed);
      results->push_to_slot(-1, t.seq, std::move(out_buf));
    }

    const auto t1_w = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>>(t1_w - t0_w).count();
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(uint64_t(ms * 1e6));
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(actual);
      g_perf->sched_cpu_tasks.fetch_add(1);
    }
#ifdef HAVE_NVCOMP
    if (rate_match) {
      rate_match->report_cpu(actual, uint64_t(ms * 1e6));
    }
#endif

    if (opt->verbosity >= V_DEBUG) {
      char in_s[32], out_s[32];
      human_bytes(double(comp_size), in_s, sizeof(in_s));
      human_bytes(double(actual), out_s, sizeof(out_s));
      double thr_gib = (ms > 0.0) ? double(actual) / (ms / 1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    // (tasks_done is counted by the writer when frames are handed off in
    // order — counting here too double-counted CPU-decompressed frames and
    // skewed the progress bar's frame-level percentage.)

#ifdef HAVE_NVCOMP
    if (sched) sched->add_cpu_bytes(comp_size);
#endif
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if (cpuagg->per_thread.size() <= (size_t)worker_id)
        cpuagg->per_thread.resize((size_t)worker_id + 1);
      auto & st = cpuagg->per_thread[(size_t)worker_id];
      st.tasks    += 1;
      st.in_bytes += comp_size;
      st.out_bytes += actual;
      st.comp_ms  += ms;
    }
  }

  // Per-thread total summary (mirrors cpu_worker for compress).
  // -vv: workers that did real work; -vvv: also report idle workers.
  if (opt->verbosity >= V_DEBUG) {
    CpuThreadStats st;
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if ((size_t)worker_id < cpuagg->per_thread.size())
        st = cpuagg->per_thread[(size_t)worker_id];
    }
    if (st.tasks == 0) {
      if (opt->verbosity >= V_TRACE) {
        std::ostringstream os;
        os << "[CPU/T" << worker_id << "] idle (0 tasks)";
        vlog(V_TRACE, *opt, os.str() + "\n");
      }
    } else {
      double thr_gib = (st.comp_ms > 0.0)
                       ? double(st.out_bytes) / (st.comp_ms / 1000.0) / 1e9 : 0.0;
      char in_s[32], out_s[32];
      human_bytes(double(st.in_bytes),  in_s,  sizeof(in_s));
      human_bytes(double(st.out_bytes), out_s, sizeof(out_s));
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] total tasks=" << st.tasks
         << " comp=" << in_s << " decomp=" << out_s
         << " time=" << std::fixed << std::setprecision(2) << st.comp_ms << "ms"
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s"
         << " pool=" << pool_size << " waits=" << pool_wait_count;
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }
  }
}

#ifdef HAVE_NVCOMP
/*======================================================================
 GPU-only CPU fallback (v0.13.54)
 -----------------------------------------------------------------------
 All GPU workers in a --gpu-only run failed terminally (VRAM exhaustion,
 driver error — at init or mid-run).  The old behaviour either died with
 the queue half-processed or, worse, left the reader blocked forever on a
 bounded queue nobody drains.  Instead, the LAST failing GPU worker calls
 this to finish the job on a full CPU pool — maximum remaining throughput,
 and the output is byte-for-byte correct (CPU and GPU emit interchangeable
 zstd frames).  Blocks until the queue is drained; the caller (a dead GPU
 worker thread) then exits and the normal teardown proceeds.
======================================================================*/
static void gpu_only_cpu_fallback(bool decompress, TaskQueue * queue,
                                  ResultStore * results, const Options & opt,
                                  Meter * m, FrameThrottle * bp)
{
  int threads = resolve_cpu_threads(opt.cpu_threads);
  // Compress with a pending restart (g_gpu_failed_restart): this drain only
  // tears the pipeline down cleanly; its output is about to be DISCARDED and
  // rebuilt from scratch by the driver, so don't claim it's "complete and
  // correct".  Decompress (and any non-restart drain) keeps the original
  // finish-on-CPU semantics.
  if (!decompress && g_gpu_failed_restart.load()) {
    vlog(V_DEFAULT, opt,
         std::string("WARNING: GPU fault — draining the pipeline on CPU (")
         + std::to_string(threads) + " threads); this partial output will be "
         "discarded and the whole archive rebuilt.\n");
  } else {
    vlog(V_DEFAULT, opt,
         std::string("WARNING: all GPUs failed; finishing ")
         + (decompress ? "decompression" : "compression") + " on CPU ("
         + std::to_string(threads) + " threads).\n"
         "  Falling back for data safety: the remaining frames are processed "
         "on the CPU instead,\n  so the output is complete and correct — just "
         "without GPU acceleration.\n");
  }
  CpuAgg agg{};
  agg.threads = threads;
  agg.per_thread.resize((size_t)threads);
  std::vector<std::thread> pool;
  pool.reserve((size_t)threads);
  for (int i = 0; i < threads; ++i) {
    if (decompress)
      pool.emplace_back(cpu_decomp_worker, i, queue, results, &opt, m,
                        (void *)nullptr, (RateMatchState *)nullptr, &agg, bp);
    else
      pool.emplace_back(cpu_worker, i, queue, results, &opt, m,
                        (void *)nullptr, (RateMatchState *)nullptr, &agg, bp);
  }
  for (auto & th : pool) th.join();
}
#endif // HAVE_NVCOMP

// Peek at the first frame's decompressed size without consuming input.  Used by
// the parallel reader's gate AND the decompress dispatch (CPU and GPU) and
// extract_tar, so it must live outside the HAVE_NVCOMP block.  Returns the
// decomp size in bytes, or -1 if it can't be determined (stdin, seek failure,
// unknown content size).  Restores the input file position on every path.
static int64_t peek_first_frame_decomp_size(FILE * in)
{
  if (!in) return -1;
  long pos = std::ftell(in);
  if (pos < 0) return -1;  // not seekable (pipe / stdin)
  unsigned char buf[64];
  size_t n = std::fread(buf, 1, sizeof(buf), in);
  if (std::fseek(in, pos, SEEK_SET) != 0) return -1;
  if (n < 4) return -1;
  unsigned long long size = ZSTD_getFrameContentSize(buf, n);
  if (size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR)
    return -1;
  return (int64_t)size;
}

#ifndef _WIN32
// Sentinel: stream_frames_to_queue_mt bailed before pushing anything; caller
// must fall back to the single-threaded reader (which re-reads from offset 0).
static constexpr size_t MT_READER_BAIL = SIZE_MAX;

// Can we pread() the input in parallel?  fstat tells us what the fd really is,
// regardless of how it arrived: a named regular file / block device, OR a
// redirected stdin ("gzstd < file", "gzstd -d < file", even "< /dev/sdX") —
// anything S_ISREG or S_ISBLK and seekable.  Pipes/FIFOs/sockets/ttys — the
// "tar -I gzstd" stream, process substitution, a terminal — are NOT preadable
// and return -1, so the caller stays on the sequential stream reader.
// Returns an fd (caller closes it iff *owns) and sets *size; -1 if unpreadable.
// --direct-read is excluded (it owns the single O_DIRECT stream itself).
static int probe_preadable_input(const Options & opt, FILE * in,
                                 bool * owns, uint64_t * size)
{
  *owns = false; *size = 0;
  if (opt.direct_read) return -1;
  int fd;
  if (opt.input == "-") { fd = in ? ::fileno(in) : 0; }          // inherited stdin
  else { fd = ::open(opt.input.c_str(), O_RDONLY); if (fd < 0) return -1; *owns = true; }

  struct stat st;
  if (::fstat(fd, &st) != 0) { if (*owns) ::close(fd); return -1; }
  uint64_t sz = 0; bool ok = false;
  if (S_ISREG(st.st_mode)) {
    sz = (uint64_t)st.st_size; ok = (sz > 0);
  } else if (S_ISBLK(st.st_mode)) {
    // Block device: size isn't in st_size — query via SEEK_END, then restore
    // the offset (we may be borrowing the inherited stdin fd).
    off_t cur = ::lseek(fd, 0, SEEK_CUR);
    off_t end = ::lseek(fd, 0, SEEK_END);
    if (cur >= 0 && end > 0) { sz = (uint64_t)end; ok = true; ::lseek(fd, cur, SEEK_SET); }
  }
  // Seekability sanity — rejects anything that slipped through (ESPIPE etc.).
  if (ok && ::lseek(fd, 0, SEEK_CUR) == (off_t)-1) ok = false;
  if (!ok) { if (*owns) ::close(fd); return -1; }
  *size = sz;
  return fd;
}

/*======================================================================
 Parallel-prefetch Zstd frame producer (decompress)
 -----------------------------------------------------------------------
 K prefetch threads pread fixed-size BLOCKs of the file (claimed in order
 via a shared counter) into a bounded ring; one consumer parses frames
 in-place and pushes them.  A carry buffer bridges a frame (or skippable
 frame) that straddles a block boundary.  This parallelizes the read I/O,
 the dominant decompress-reader cost (v0.13.70 [READER]: io ~52%, copy
 ~37%, parse ~2.5% on the server); the per-frame copy stays on the
 consumer (a later zero-copy pass can remove it).

 Engaged only for a seekable regular file whose first frame is a normal
 known-size zstd frame (caller's peek).  Anything unusual mid-stream
 (unknown content size, malformed frame) is treated as a hard data error,
 exactly as the single-threaded reader treats truncation — it never hands
 off mid-stream.  Returns frames pushed, or MT_READER_BAIL if it could not
 even open the file (no frames pushed; caller uses the single reader).
======================================================================*/
static size_t stream_frames_to_queue_mt(
    int fd, bool owns_fd, uint64_t file_size, int n_readers,
    TaskQueue & queue, Meter * m, const Options & opt,
    size_t * max_frame_decomp_out, const std::atomic<bool> * abort,
    bool * fallback_out, std::vector<char> * raw_data_out)
{
  if (fd < 0) return MT_READER_BAIL;
  try_boost_io_priority(!opt.gpu_only);
  if (m) m->reader_threads.store(n_readers, std::memory_order_relaxed);

  // BLOCK comfortably larger than a typical frame so most frames sit fully
  // within one block (a straddling frame costs an extra small copy).  Frames
  // larger than BLOCK (e.g. huge --chunk-size) just span multiple blocks via
  // carry — correct, merely with more copying.
  const size_t BLOCK = (size_t)64 * ONE_MIB;
  const size_t STEP  = (size_t)4 * ONE_MIB;   // carry-growth increment
  const size_t n_blocks = (size_t)((file_size + BLOCK - 1) / BLOCK);
  const int    n_slots  = std::max(n_readers * 2, n_readers + 1);

  struct Slot { std::vector<char> buf; size_t len = 0; size_t idx = SIZE_MAX; bool ready = false; };
  std::vector<Slot> slots((size_t)n_slots);
  for (auto & s : slots) s.buf.resize(BLOCK);

  std::mutex mtx;
  std::condition_variable cv_filled, cv_freed;
  size_t consumed = 0;                 // blocks < consumed are free for reuse
  std::atomic<size_t> next_block{0};
  std::atomic<bool> failed{false};
  std::string fail_msg;
  bool stop = false;

  auto prefetch = [&]() {
    for (;;) {
      size_t i = next_block.fetch_add(1, std::memory_order_relaxed);
      if (i >= n_blocks) break;
      Slot & s = slots[i % (size_t)n_slots];
      {
        std::unique_lock<std::mutex> lk(mtx);
        // slot i%n last held block i-n; free once the consumer passed it
        cv_freed.wait(lk, [&]{ return stop || consumed + (size_t)n_slots > i; });
        if (stop) return;
      }
      const off_t  off  = (off_t)i * (off_t)BLOCK;
      const size_t want = std::min(BLOCK, (size_t)(file_size - (uint64_t)off));
      uint64_t t0 = m ? now_ns() : 0;
      size_t got = 0;
      bool io_err = false;
      while (got < want) {
        ssize_t r = ::pread(fd, s.buf.data() + got, want - got, off + (off_t)got);
        if (r < 0)  { io_err = true; break; }
        if (r == 0) break;            // file shrank under us; len<want flags it
        got += (size_t)r;
      }
      // Timing only — NOT m->read_bytes: the decompress workers count input
      // bytes per frame (cpu_decomp_worker / gpu workers), exactly as for the
      // single-threaded reader.  Counting here too double-counts the input
      // (the "423.78 GiB" display / progress-to-100%-at-half bug).
      if (m && got) m->reader_io_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
      if (g_perf && got) { g_perf->read_ns.fetch_add(now_ns() - t0);
                           g_perf->read_bytes_total.fetch_add(got); }
      {
        std::lock_guard<std::mutex> lk(mtx);
        if (io_err) { failed.store(true, std::memory_order_relaxed);
                      if (fail_msg.empty()) fail_msg = "pread failed (parallel decompress reader)";
                      stop = true; }
        else { s.idx = i; s.len = got; s.ready = true; }
      }
      cv_filled.notify_all();
      if (io_err) { cv_freed.notify_all(); return; }
    }
  };

  std::vector<std::thread> readers;
  readers.reserve((size_t)n_readers);
  for (int k = 0; k < n_readers; ++k) readers.emplace_back(prefetch);

  size_t seq = 0, max_frame_decomp = 0;
  std::vector<char> carry;            // straddling-frame bytes from prior block(s)
  uint64_t carry_origin = 0;          // absolute file offset of carry[0] (for the fallback offset)
  bool aborted = false;
  // Set when a mid-stream frame has no content-size header: parallel decode
  // needs known frame sizes, so we hand the rest of the file (from fallback_off)
  // to the caller's CPU streaming decoder instead of dying.  Matches the
  // single-threaded reader's mid-file fallback (which sets the same out-params).
  bool need_fallback = false;
  uint64_t fallback_off = 0;

  auto fail = [&](const std::string & why) {
    std::lock_guard<std::mutex> lk(mtx);
    if (fail_msg.empty()) fail_msg = why;
    failed.store(true, std::memory_order_relaxed);
    stop = true;
  };

  auto emit = [&](const char * p, size_t fc) -> bool {
    unsigned long long dz = ZSTD_getFrameContentSize(p, fc);
    if (dz == ZSTD_CONTENTSIZE_UNKNOWN || dz == ZSTD_CONTENTSIZE_ERROR) {
      // Complete frame but no content-size header (e.g. a zstd-streamed segment
      // concatenated after a gzstd archive).  Don't die: the caller computes the
      // file offset and falls back to the CPU streaming decoder for the tail.
      return false;
    }
    Task t; t.seq = seq++;
    uint64_t cp = m ? now_ns() : 0;
    t.data.assign(p, p + fc);
    if (m) m->reader_copy_ns.fetch_add(now_ns() - cp, std::memory_order_relaxed);
    t.decomp_size = (size_t)dz;
    if ((size_t)dz > max_frame_decomp) max_frame_decomp = (size_t)dz;
    uint64_t pb = m ? now_ns() : 0;
    queue.push(std::move(t));
    if (m) { m->reader_blocked_ns.fetch_add(now_ns() - pb, std::memory_order_relaxed);
             m->total_out.fetch_add((uint64_t)dz, std::memory_order_relaxed); }
    return true;
  };

  for (size_t cur = 0; cur < n_blocks && !aborted; ++cur) {
    if (abort && abort->load(std::memory_order_acquire)) { aborted = true; break; }
    Slot & s = slots[cur % (size_t)n_slots];
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv_filled.wait(lk, [&]{ return failed.load(std::memory_order_relaxed) || (s.ready && s.idx == cur); });
      if (failed.load(std::memory_order_relaxed)) { aborted = true; break; }
    }
    const char * data = s.buf.data();
    const size_t len  = s.len;
    size_t pos = 0;

    // (A) finish a frame/skippable carried from the previous block(s).
    if (!carry.empty()) {
      const size_t old = carry.size();
      // Is the carried frame a skippable frame?
      uint32_t magic = 0;
      if (carry.size() >= 4) std::memcpy(&magic, carry.data(), 4);
      const bool skippable = (carry.size() >= 4) && ((magic & 0xFFFFFFF0u) == 0x184D2A50u);
      size_t appended = 0, need = 0; bool resolved = false;
      if (skippable) {
        // need 8 bytes for the size field, then 8 + skip_size total
        while (carry.size() < 8 && appended < len) {
          size_t step = std::min(STEP, len - appended);
          carry.insert(carry.end(), data + appended, data + appended + step); appended += step;
        }
        if (carry.size() >= 8) {
          uint32_t ss = 0; std::memcpy(&ss, carry.data() + 4, 4);
          need = 8 + (size_t)ss;
          while (carry.size() < need && appended < len) {
            size_t step = std::min(STEP, len - appended);
            carry.insert(carry.end(), data + appended, data + appended + step); appended += step;
          }
          if (carry.size() >= need) { pos = need - old; carry.clear(); resolved = true; }  // skipped
        }
      } else {
        // Complete a normal frame that straddled into this block.  zstd has no
        // resumable frame-size parser, so each ZSTD_findFrameCompressedSize()
        // re-walks the whole carry from the start.  Growing the carry by a
        // FIXED STEP made that re-walk quadratic for frames much larger than a
        // BLOCK (--ultra / huge --chunk-size: a 128 MiB frame did ~16 re-parses
        // per 64 MiB block, each rescanning the full accumulated carry).  Grow
        // the step geometrically instead: a straddle-by-a-little still resolves
        // on the first small step, while a multi-block frame needs only
        // ~log2(BLOCK/STEP) parses per block.  Overshoot is bounded at 2x and
        // re-parsed in place by (B), so correctness is unchanged.
        size_t fc = 0;
        size_t step = STEP;
        while (appended <= len) {
          uint64_t ps = m ? now_ns() : 0;
          fc = ZSTD_findFrameCompressedSize(carry.data(), carry.size());
          if (m) m->reader_parse_ns.fetch_add(now_ns() - ps, std::memory_order_relaxed);
          if (!ZSTD_isError(fc) && fc <= carry.size()) {
            if (!emit(carry.data(), fc)) { need_fallback = true; fallback_off = carry_origin; aborted = true; }
            pos = (fc > old) ? (fc - old) : 0; carry.clear(); resolved = true; break;
          }
          if (appended >= len) break;       // exhausted this block; frame spans further
          size_t take = std::min(step, len - appended);
          carry.insert(carry.end(), data + appended, data + appended + take); appended += take;
          step = std::min(step * 2, BLOCK); // geometric growth, capped at one block
        }
      }
      if (aborted) break;
      if (!resolved) { pos = len; }         // whole block absorbed into carry; fetch next block
    }

    // (B) parse the rest of the block in place.
    while (!aborted && pos < len) {
      const char * p = data + pos; const size_t rem = len - pos;
      // A frame starting here begins at this absolute file offset.  Recorded so
      // a carry assigned below carries its origin (for section A's fallback),
      // and reused directly if this frame has no content-size header.
      carry_origin = (uint64_t)cur * BLOCK + pos;
      if (rem < 4) { carry.assign(p, p + rem); break; }
      uint32_t magic = 0; std::memcpy(&magic, p, 4);
      if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {          // skippable frame
        if (rem < 8) { carry.assign(p, p + rem); break; }
        uint32_t ss = 0; std::memcpy(&ss, p + 4, 4);
        size_t tot = 8 + (size_t)ss;
        if (tot > rem) { carry.assign(p, p + rem); break; }  // straddles
        pos += tot; continue;
      }
      uint64_t ps = m ? now_ns() : 0;
      size_t fc = ZSTD_findFrameCompressedSize(p, rem);
      if (m) m->reader_parse_ns.fetch_add(now_ns() - ps, std::memory_order_relaxed);
      if (ZSTD_isError(fc) || fc > rem) { carry.assign(p, p + rem); break; }  // straddles
      if (!emit(p, fc)) { need_fallback = true; fallback_off = carry_origin; aborted = true; break; }
      pos += fc;
    }

    {
      std::lock_guard<std::mutex> lk(mtx);
      consumed = cur + 1; s.ready = false;
    }
    cv_freed.notify_all();
  }

  // Tail: a non-empty carry after the last block means trailing bytes that
  // never formed a complete frame — truncation/corruption (the single reader
  // dies on the same condition).  A few trailing zero/pad bytes are tolerated.
  if (!aborted && !failed.load(std::memory_order_relaxed) && carry.size() > 8)
    fail("truncated zstd stream: " + std::to_string(carry.size())
         + " trailing bytes after " + std::to_string(seq) + " frames");

  { std::lock_guard<std::mutex> lk(mtx); stop = true; }
  cv_freed.notify_all(); cv_filled.notify_all();
  for (auto & th : readers) th.join();

  // Mid-stream unknown-size frame: hand the rest of the file to the caller's
  // CPU streaming decoder rather than dying.  Read [fallback_off, file_size)
  // from the (still-open) fd into raw_data; the frames already queued are
  // written first, then the caller appends this tail via decompress_from_buffer.
  // A genuine I/O error (failed) takes the die path below instead.
  if (need_fallback && !failed.load(std::memory_order_relaxed)) {
    if (raw_data_out && fallback_off < file_size) {
      raw_data_out->resize((size_t)(file_size - fallback_off));
      size_t got = 0;
      while (got < raw_data_out->size()) {
        ssize_t r = ::pread(fd, raw_data_out->data() + got,
                            raw_data_out->size() - got,
                            (off_t)(fallback_off + (uint64_t)got));
        if (r <= 0) break;     // short/erroring read: caller streams what we got
        got += (size_t)r;
      }
      raw_data_out->resize(got);
    }
    if (fallback_out) *fallback_out = true;
    if (owns_fd) ::close(fd);
    // The caller emits the single user-facing warning (shared with the
    // single-threaded reader's fallback) once the parsed frames are written.
    if (max_frame_decomp_out) *max_frame_decomp_out = max_frame_decomp;
    return seq;
  }

  if (owns_fd) ::close(fd);   // borrowed (inherited stdin) fd: leave it open

  if (failed.load(std::memory_order_relaxed)) {
    std::string why; { std::lock_guard<std::mutex> lk(mtx); why = fail_msg; }
    die_data(why.empty() ? "parallel decompress reader failed" : why);
  }
  if (max_frame_decomp_out) *max_frame_decomp_out = max_frame_decomp;
  return seq;
}
#endif  // !_WIN32

/*======================================================================
 Streaming Zstd frame producer
 -----------------------------------------------------------------------
 Reads compressed input incrementally and pushes individual Zstd frames
 into the task queue as they are discovered.  Workers can begin decom-
 pressing immediately  no need to read the entire file first.

 Uses ZSTD_findFrameCompressedSize() to find each frame boundary and
 ZSTD_getFrameContentSize() to determine the expected output size.
 If either fails (e.g. single-frame file with unknown content size),
 sets *fallback = true and returns the partially-read data in raw_data
 so the caller can fall back to streaming decompression.

 Returns:
   total number of frames pushed (0 if fallback triggered on first frame)
======================================================================*/
static size_t stream_frames_to_queue(
    FILE * in,
    TaskQueue & queue,
    Meter * m,
    const Options & opt,
    bool * fallback,
    std::vector<char> * raw_data,
    size_t * max_frame_decomp_out = nullptr,
    const std::atomic<bool> * abort = nullptr)
{
  size_t max_frame_decomp = 0;
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  *fallback = false;
  if (m) m->reader_threads.store(1, std::memory_order_relaxed);  // single sequential splitter

#ifndef _WIN32
  // Parallel-prefetch fast path: any preadable source — a named regular
  // file/block device OR a redirected stdin ("gzstd -d < big.zst") — whose
  // first frame is a normal known-size zstd frame.  fstat (probe_preadable_
  // input) decides preadability; pipes (tar -I gzstd), O_DIRECT, unknown-size
  // single-frame, and foreign archives stay on this single-threaded reader,
  // which also owns every fallback path.  peek runs on the FILE* (seekable
  // here, since fstat said so); the MT reader preads absolute offsets, so the
  // peek's position is irrelevant.
  {
    bool owns = false; uint64_t fsz = 0;
    int pfd = probe_preadable_input(opt, in, &owns, &fsz);
    if (pfd >= 0) {
      const int n_readers = opt.read_threads > 0 ? (int)opt.read_threads
                          : std::max(3, std::min(12, resolve_cpu_threads(opt.cpu_threads) / 8));
      if (n_readers > 1 && fsz > (uint64_t)(2 * 64 * ONE_MIB)
          && peek_first_frame_decomp_size(in) >= 0) {
        if (opt.verbosity >= V_VERBOSE)
          vlog(V_VERBOSE, opt, "[READER] parallel prefetch: " + std::to_string(n_readers)
               + " reader threads"
               + (opt.input == "-" ? " (stdin is a seekable file)" : "") + "\n");
        size_t r = stream_frames_to_queue_mt(pfd, owns, fsz, n_readers,
                                             queue, m, opt, max_frame_decomp_out, abort,
                                             fallback, raw_data);
        if (r != MT_READER_BAIL) return r;
        if (m) m->reader_threads.store(1, std::memory_order_relaxed);
      } else if (owns) {
        ::close(pfd);   // not engaging MT; release the fd we opened to probe
      }
    }
  }
#endif

  // Read buffer with offset-based tracking to avoid per-iteration shifts.
  // We compact (memmove) only when the unconsumed tail is too small to
  // hold a new read chunk, keeping the common path allocation-free.
  const size_t READ_CHUNK = 4 * ONE_MIB;
  std::vector<char> buf(READ_CHUNK * 2);
  size_t buf_len = 0;    // valid data in buf[0..buf_len)
  size_t buf_off = 0;    // parse cursor within buf

  size_t seq = 0;
  uint64_t out_off_acc = 0;   // --keep-going: running decompressed-output offset
  bool eof = false;

  // --direct-read: O_DIRECT input for decompress, mirroring the compress reader.
  // Read 4 KiB-aligned READ_CHUNK blocks into an aligned bounce buffer and copy into
  // the parse buffer (frame boundaries don't align to reads, so the bounce is
  // unavoidable — but it's the same copy fread does internally).  Bypasses the page
  // cache: honest-cold, no eviction.  Opens its own fd on opt.input — the FILE* `in`
  // is at offset 0 here (peek_first_frame_decomp_size rewinds) and is simply unused
  // for reading while O_DIRECT is active.  Falls back to fread if it can't be set up.
  struct DirectIn { int fd = -1; void * b = nullptr;
    ~DirectIn() { if (fd >= 0) ::close(fd); if (b) free(b); } } din;
  off_t din_off = 0;
  bool use_direct = false;
#ifndef _WIN32
  if (opt.direct_read && opt.input != "-" && fs::is_regular_file(opt.input)) {
    din.fd = ::open(opt.input.c_str(), O_RDONLY | O_DIRECT);
    if (din.fd >= 0 && posix_memalign(&din.b, 4096, READ_CHUNK) == 0 && din.b) {
      use_direct = true;
      vlog(V_VERBOSE, opt, "[DIRECT-READ] O_DIRECT input (page cache bypassed)\n");
    }
  }
#endif
  // Fill dst with up to READ_CHUNK bytes; returns count (0 = EOF).
  auto read_chunk = [&](char * dst) -> size_t {
#ifndef _WIN32
    if (use_direct) {
      ssize_t got = ::pread(din.fd, din.b, READ_CHUNK, din_off);  // offset & len 4 KiB-aligned
      if (got < 0) die_io("O_DIRECT read failed (--direct-read, decompress)");
      if (got == 0) return 0;
      std::memcpy(dst, din.b, (size_t)got);
      din_off += got;
      return (size_t)got;
    }
#endif
    return std::fread(dst, 1, READ_CHUNK, in);
  };

  while (!eof) {
    // Caller asked us to stop early (e.g. --gpu-only but the deferred bringup
    // thread found no CUDA device).  Return what we have; the caller errors.
    if (abort && abort->load(std::memory_order_acquire)) {
      if (max_frame_decomp_out) *max_frame_decomp_out = max_frame_decomp;
      return seq;
    }

    // Compact: move unconsumed tail to front if needed to make room for reading
    if (buf_off > 0) {
      size_t tail = buf_len - buf_off;
      if (tail > 0)
        std::memmove(buf.data(), buf.data() + buf_off, tail);
      buf_len = tail;
      buf_off = 0;
    }

    // Ensure buffer has room for a full read chunk
    if (buf_len + READ_CHUNK > buf.size())
      buf.resize(buf_len + READ_CHUNK);

    // Read more data from input (O_DIRECT bounce when --direct-read, else fread)
    uint64_t rd_t0 = (g_perf || m) ? now_ns() : 0;
    size_t n = read_chunk(buf.data() + buf_len);
    if ((g_perf || m) && n > 0) {
      const uint64_t r_dt = now_ns() - rd_t0;
      if (m) m->reader_io_ns.fetch_add(r_dt, std::memory_order_relaxed);
      if (g_perf) {
        g_perf->read_ns.fetch_add(r_dt);
        g_perf->read_bytes_total.fetch_add(n);
      }
    }
    buf_len += n;
    if (n == 0) eof = true;

    // Parse as many complete frames as we can from buf[buf_off..buf_len)
    while (buf_off < buf_len) {
      const char * ptr = buf.data() + buf_off;
      size_t remaining = buf_len - buf_off;

      // Need at least 4 bytes for magic number
      if (remaining < 4) break;

      // Skip skippable frames (magic 0x184D2A5?)
      {
        uint32_t magic = 0;
        std::memcpy(&magic, ptr, 4);
        if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {
          if (remaining < 8) break;  // need more data for skip size
          uint32_t skip_size = 0;
          std::memcpy(&skip_size, ptr + 4, 4);
          size_t total_skip = 8 + (size_t)skip_size;
          if (total_skip > remaining) break;  // need more data
          buf_off += total_skip;
          continue;
        }
      }

      // Try to find the compressed frame size (the sequential parse spine —
      // each frame's blocks must be walked to locate the next frame's start).
      uint64_t ps_t0 = m ? now_ns() : 0;
      size_t frame_comp = ZSTD_findFrameCompressedSize(ptr, remaining);
      if (m) m->reader_parse_ns.fetch_add(now_ns() - ps_t0, std::memory_order_relaxed);
      if (ZSTD_isError(frame_comp)) {
        if (!eof) {
          // Incomplete frame data -- need to read more.
          break;
        }
        // At EOF: no more data coming
        if (seq == 0) {
          // Never parsed any frame -- not a valid Zstd file
          *fallback = true;
          if (raw_data) {
            raw_data->assign(buf.data() + buf_off, buf.data() + buf_len);
          }
          return 0;
        }
        // Parsed some frames OK; trailing bytes cannot form a valid frame.
        // A few trailing null bytes could be padding; anything substantial
        // indicates truncation or corruption.
        if (remaining > 8) {
          // Significant trailing data that isn't a valid frame = truncated
          *fallback = false;  // not a format problem, it's a data problem
          if (opt.keep_going) {
            // Recovery: the reader can no longer find frame boundaries, so it
            // can't skip ahead.  Keep the frames already parsed (the pipeline
            // writes them), flag the framing break, and stop reading cleanly —
            // the driver prints the rollup and exits EXIT_DECOMP_INCOMPLETE.
            g_damage.set_framing_break((uint64_t)seq, out_off_acc,
                "frame structure corrupt after frame " + std::to_string(seq)
                + " — " + std::to_string(remaining)
                + " trailing bytes cannot form a valid frame, and gzstd does not "
                "scan to resync; everything past here is unrecoverable");
            if (max_frame_decomp_out) *max_frame_decomp_out = max_frame_decomp;
            return seq;
          }
          // Return frames parsed so far  the caller should still detect
          // the issue because the decompressed output will be incomplete.
          vlog(V_NORMAL, opt,
               "error: " + std::to_string(remaining)
               + " trailing bytes after frame " + std::to_string(seq)
               + " (truncated or corrupt input)\n");
          die_data("truncated zstd stream: " + std::to_string(remaining)
                   + " trailing bytes after " + std::to_string(seq) + " frames");
        }
        vlog(V_VERBOSE, opt,
             "warning: " + std::to_string(remaining)
             + " trailing bytes after frame " + std::to_string(seq)
             + " not a valid Zstd frame (ignored)\n");
        break;
      }

      // If the frame extends past what we\'ve read, wait for more data
      if (frame_comp > remaining) break;

      // Get decompressed size from the frame header
      unsigned long long frame_decomp = ZSTD_getFrameContentSize(ptr, remaining);
      if (frame_decomp == ZSTD_CONTENTSIZE_UNKNOWN ||
          frame_decomp == ZSTD_CONTENTSIZE_ERROR) {
        // Can\'t determine output size -- must fall back to streaming
        *fallback = true;
        if (raw_data) {
          // Reconstruct: unconsumed portion + rest of input
          raw_data->assign(buf.data() + buf_off, buf.data() + buf_len);
          std::vector<char> tail(READ_CHUNK);
          while (true) {
            size_t r = read_chunk(tail.data());
            if (r == 0) break;
            raw_data->insert(raw_data->end(), tail.data(), tail.data() + r);
          }
        }
        return seq;  // some frames may already be queued
      }

      // Complete frame with known sizes -- push to the queue
      Task t;
      t.seq = seq++;
      uint64_t cp_t0 = m ? now_ns() : 0;
      t.data.assign(ptr, ptr + frame_comp);   // copy compressed bytes out of the parse buffer
      if (m) m->reader_copy_ns.fetch_add(now_ns() - cp_t0, std::memory_order_relaxed);
      t.decomp_size = (size_t)frame_decomp;
      t.out_off = out_off_acc;                 // --keep-going: offset of this frame in the output
      out_off_acc += (uint64_t)frame_decomp;
      if ((size_t)frame_decomp > max_frame_decomp) max_frame_decomp = (size_t)frame_decomp;
      // push() blocks on the bounded queue when consumers can't keep up — that
      // wait is "the consumers are the bottleneck, not the reader" (vs io/parse/
      // copy, which are the reader itself).
      uint64_t pb_t0 = m ? now_ns() : 0;
      queue.push(std::move(t));
      if (m) m->reader_blocked_ns.fetch_add(now_ns() - pb_t0, std::memory_order_relaxed);
      if (m) m->total_out.fetch_add((uint64_t)frame_decomp, std::memory_order_relaxed);

      buf_off += frame_comp;

      if (opt.verbosity >= V_TRACE && ((seq - 1) % 1000 == 0 || frame_comp > 100 * ONE_MIB)) {
        char cs[32], ds[32];
        human_bytes(double(frame_comp), cs, sizeof(cs));
        human_bytes(double(frame_decomp), ds, sizeof(ds));
        std::ostringstream os;
        os << "[SPLIT] frame " << (seq - 1)
           << " comp=" << cs << " decomp=" << ds;
        vlog(V_TRACE, opt, os.str() + "\n");
      }
    }
  }

  // Any leftover bytes are trailing garbage  warn if non-empty
  size_t leftover = buf_len - buf_off;
  if (leftover > 0 && seq > 0) {
    vlog(V_VERBOSE, opt,
         "warning: " + std::to_string(leftover)
         + " trailing bytes after last Zstd frame (ignored)\n");
  }

  if (max_frame_decomp_out) *max_frame_decomp_out = max_frame_decomp;
  return seq;
}

/*======================================================================
 MT CPU-only decompression (multi-frame)
 -----------------------------------------------------------------------
 Launches worker threads first, then streams frames into the queue.
 Workers begin decompressing as soon as the first frame arrives 
 no waiting for the entire file to be read.
======================================================================*/
static void decompress_cpu_mt(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // ---- Performance instrumentation (active at -vvv) ----
  PerfCounters perf_local;
  if (opt.verbosity >= V_TRACE) g_perf = &perf_local;

  int threads = resolve_cpu_threads(opt.cpu_threads);


  CpuAgg cpuagg{};
  cpuagg.threads = threads;
  cpuagg.per_thread.resize((size_t)threads);

  TaskQueue queue;
  ResultStore results;
  FrameThrottle throttle(compute_throttle_budget(
      std::max<size_t>(1, opt.chunk_mib) * ONE_MIB, threads, 0, opt));
  // Disable throttle in test mode: no disk I/O means permits are never released.
  FrameThrottle * bp_ptr = (opt.mode == Mode::TEST) ? nullptr : &throttle;

  // Bound queued (read-but-not-popped) frames to pipeline depth so a consumer
  // slower than the reader can't buffer the whole compressed input in RAM
  // (ROADMAP 7.8).  Skip when throttling is explicitly disabled.
  if (opt.throttle_frames != 0) {
    int qslack = opt.throttle_factor > 0 ? opt.throttle_factor : THROTTLE_SLACK_FACTOR;
    const size_t qfloor = (size_t)std::max(THROTTLE_MIN_FRAMES, threads * qslack);
    queue.set_max_depth(qfloor);
    // Byte ceiling alongside the frame floor: bound queued RAM so incompressible
    // input (near-full-size compressed frames) can't hold ~4× the RAM a
    // compressible run does.  ~8 MiB/slot = half a standard 16 MiB frame (≈ half
    // the floor's worst-case bytes; measured throughput-neutral on Gen3 — should
    // be validated on knuth).  Soft cap — push() still admits a frame when empty.
    queue.set_max_bytes(qfloor * (8 * ONE_MIB));
  }

  // Start writer thread (outputs decompressed frames in original order)
  std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m, bp_ptr);

  // Start worker threads (they block on the queue until frames arrive)
  std::vector<std::thread> pool;
  pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) {
    pool.emplace_back(cpu_decomp_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
                      nullptr, nullptr,
#endif
                      &cpuagg, bp_ptr);
  }
  if (opt.verbosity >= V_VERBOSE)
    std::cerr << "[CPU] " << threads << " decompression threads online\n";

  // Stream frames from input directly into the queue.
  // Workers start decompressing as soon as the first frame is pushed.
  bool fallback = false;
  std::vector<char> raw_data;
  size_t n_frames = stream_frames_to_queue(in, queue, m, opt, &fallback, &raw_data);

  // Preallocate output file to avoid per-write extent allocation overhead.
  // total_out is known from zstd frame headers parsed by stream_frames_to_queue.
#ifndef _WIN32
  if (g_direct_writer && m && n_frames > 0 && opt.preallocate_output) {
    uint64_t total = m->total_out.load(std::memory_order_relaxed);
    if (total > 0 && g_direct_writer->preallocate(total)) {
      char sz[32]; human_bytes(double(total), sz, sizeof(sz));
      vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
    }
  }
#endif

  if (fallback && n_frames == 0) {
    // Could not parse any frames  shut down workers, fall back to streaming
    queue.set_done();
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.producer_done = true;
      results.total_tasks = 0;
      results.workers_done = true;
    }
    results.cv.notify_all();
    throttle.set_done();
    for (auto & th : pool) th.join();
    wthr.join();

    vlog(V_VERBOSE, opt,
         "cannot determine frame sizes; using streaming CPU decompress\n");
    decompress_from_buffer(raw_data, out, opt, m);
    return;
  }

  if (n_frames == 0) {
    // Empty file  clean shutdown
    queue.set_done();
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.producer_done = true;
      results.total_tasks = 0;
      results.workers_done = true;
    }
    results.cv.notify_all();
    throttle.set_done();
    for (auto & th : pool) th.join();
    wthr.join();
    return;
  }

  vlog(V_VERBOSE, opt,
       "[READER] streamed " + std::to_string(n_frames) + " frames for MT CPU decompress\n");

  // Signal that all frames have been enqueued
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
    if (m) {
      m->total_frames.store(results.total_tasks, std::memory_order_relaxed);
      m->total_out_final.store(true, std::memory_order_release);
    }
  }
  results.cv.notify_all();

  // Wait for workers, then writer.
  // Do NOT call throttle.set_done() before join: workers must respect
  // throttle while draining the queue, otherwise they buffer the entire
  // output in RAM (the done_ flag bypasses acquire()).
  for (auto & th : pool) th.join();
  throttle.set_done();  // safe now: all workers exited
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();
  wthr.join();
  log_throttle_stats(throttle, opt, "decompress-cpu");

  // Mid-file fallback tail: stream_frames_to_queue hit a frame with no
  // content-size header (e.g. `cat a.zst b.zst` where b came from zstd on a
  // pipe) and buffered the rest of the input in raw_data.  The parsed frames
  // are fully written now (writer joined), so append the tail via the CPU
  // streaming decoder.  Before v0.13.54 this tail was silently dropped —
  // truncated output with exit 0.
  if (fallback && !raw_data.empty()) {
    vlog(V_DEFAULT, opt,
         "warning: frame " + std::to_string(n_frames) + " onward has no "
         "content-size header (zstd streaming output); the parallel reader "
         "can't split it, so the remaining data is decompressed with the CPU "
         "streaming decoder (slower, but nothing is lost).\n");
    decompress_from_buffer(raw_data, out, opt, m);
  }

  if (g_perf) {
    g_perf->print_summary("CPU-ONLY DECOMPRESS");
    g_perf = nullptr;
  }
}

/*======================================================================
 MT CPU-only compression
======================================================================*/
static void compress_cpu_stream(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // Chunk size: 16 MiB default, or user-specified via --chunk-size
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;

  // Ultra levels need chunk >= window size to be effective.
  // Auto-increase if the user didn't explicitly set --chunk-size.
  {
    size_t ultra_min = ultra_min_chunk_mib(opt.level, opt.ultra);
    if (ultra_min > 0 && chosen_mib < ultra_min) {
      if (!opt.chunk_user_set) {
        vlog(V_VERBOSE, opt, "[ULTRA] auto-increasing chunk size from "
             + std::to_string(chosen_mib) + " to " + std::to_string(ultra_min)
             + " MiB (must be >= window size)\n");
        chosen_mib = ultra_min;
      } else {
        vlog(V_ERROR, opt, "warning: --chunk-size=" + std::to_string(chosen_mib)
             + " MiB is smaller than the ultra window ("
             + std::to_string(ultra_min) + " MiB). Compression ratio will suffer.\n");
      }
    }
  }

  // Bound the chunk by available RAM, exactly as the MT path does (the -T1
  // early-return in compress_cpu_mt skips its own check_ram_budget call).  A
  // no-op for normal chunk sizes; for an absurd --chunk-size it reduces the
  // chunk to fit instead of throwing an uncaught bad_alloc on the inbuf/outbuf
  // allocation below.
  chosen_mib = check_ram_budget(1, chosen_mib, opt);
  const size_t chunk_bytes = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // Get total input size for progress percentage
  uint64_t total_in = known_input_size(opt, in);

  // Preallocate output file: input size is a safe upper bound for compressed output.
#ifndef _WIN32
  if (g_direct_writer && total_in > 0 && opt.preallocate_output
      && g_direct_writer->preallocate(total_in)) {
    char sz[32]; human_bytes(double(total_in), sz, sizeof(sz));
    vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
  }
#endif

  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);

  ZSTD_CCtx * cctx = ZSTD_createCCtx();
  if (!cctx) die("failed to create ZSTD_CCtx");
  size_t st = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, opt.level);
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);  // self-verifying frames (see compress worker)

  // Ultra levels need explicit windowLog for large windows to take effect
  {
    int wlog = apply_ultra_cctx(cctx, opt.level, opt.ultra);
    if (wlog > 0)
      vlog(V_VERBOSE, opt, "[ULTRA] windowLog=" + std::to_string(wlog)
           + " (" + std::to_string(size_t(1) << (wlog - 20)) + " MiB window)\n");
  }

  std::vector< char > inbuf(chunk_bytes);
  std::vector< char > outbuf(ZSTD_compressBound(chunk_bytes));
  while (true) {
    uint64_t rd_t0 = g_perf ? now_ns() : 0;
    size_t n = std::fread(inbuf.data(), 1, chunk_bytes, in);
    if (g_perf && n > 0) {
      g_perf->read_ns.fetch_add(now_ns() - rd_t0);
      g_perf->read_bytes_total.fetch_add(n);
    }
    if (n == 0) break;
    if (m) m->read_bytes.fetch_add(n);
    uint64_t comp_t0 = g_perf ? now_ns() : 0;
    size_t csz = ZSTD_compress2(cctx, outbuf.data(), outbuf.size(), inbuf.data(), n);
    if (ZSTD_isError(csz)) die_data(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(now_ns() - comp_t0);
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(n);
    }
    uint64_t w_t0 = g_perf ? now_ns() : 0;
#ifndef _WIN32
    if (g_direct_writer) {
      if (!g_direct_writer->write(outbuf.data(), csz))
        die_io("direct write failed (disk full?)");
    } else
#endif
    {
      size_t w = robust_fwrite(outbuf.data(), csz, out);
      if (w != csz) die_io("short write to output (broken pipe?)");
    }
    if (g_perf) {
      g_perf->write_ns.fetch_add(now_ns() - w_t0);
      g_perf->write_bytes_total.fetch_add(csz);
    }
    if (m) m->wrote_bytes.fetch_add(csz);
  }

  ZSTD_freeCCtx(cctx);
  progress_done = true; progress_thr.join();
}

// Single-frame compression using zstd's built-in MT with sliding window.
// Produces one frame (like `zstd`) for maximum ratio on repetitive data.
// Decompression will be single-threaded (one frame = one unit of work).
static void compress_cpu_sliding_window(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  unsigned n_threads = (unsigned)std::thread::hardware_concurrency();
  if (opt.cpu_threads > 0) n_threads = (unsigned)opt.cpu_threads;
  else if (opt.cpu_threads == -1) n_threads = (unsigned)std::thread::hardware_concurrency();
  if (n_threads < 1) n_threads = 1;

  vlog(V_VERBOSE, opt, "[SLIDING-WINDOW] single-frame compression with "
       + std::to_string(n_threads) + " zstd worker threads\n");

  uint64_t total_in = known_input_size(opt, in);

  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);

  ZSTD_CCtx * cctx = ZSTD_createCCtx();
  if (!cctx) die("failed to create ZSTD_CCtx");

  size_t st;
  st = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, opt.level);
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(level): ") + ZSTD_getErrorName(st));
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);  // self-verifying frames (see compress worker)
  st = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, (int)n_threads);
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(nbWorkers): ") + ZSTD_getErrorName(st));
  if (total_in > 0) {
    st = ZSTD_CCtx_setPledgedSrcSize(cctx, total_in);
    if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setPledgedSrcSize: ") + ZSTD_getErrorName(st));
  }

  apply_ultra_cctx(cctx, opt.level, opt.ultra);

  const size_t READ_CHUNK = 4 * ONE_MIB;
  const size_t OUT_BUF = ZSTD_CStreamOutSize();
  std::vector<char> inbuf(READ_CHUNK);
  std::vector<char> outbuf(OUT_BUF);

  bool finished = false;
  while (!finished) {
    size_t n = std::fread(inbuf.data(), 1, READ_CHUNK, in);
    if (m && n > 0) m->read_bytes.fetch_add(n, std::memory_order_relaxed);
    bool last_chunk = (n < READ_CHUNK);

    ZSTD_inBuffer input = { inbuf.data(), n, 0 };
    ZSTD_EndDirective directive = last_chunk ? ZSTD_e_end : ZSTD_e_continue;

    do {
      ZSTD_outBuffer output = { outbuf.data(), outbuf.size(), 0 };
      size_t remaining = ZSTD_compressStream2(cctx, &output, &input, directive);
      if (ZSTD_isError(remaining))
        die_data(std::string("ZSTD_compressStream2: ") + ZSTD_getErrorName(remaining));

      if (output.pos > 0) {
#ifndef _WIN32
        if (g_direct_writer) {
          if (!g_direct_writer->write((const char *)output.dst, output.pos))
            die_io("direct write failed (disk full?)");
        } else
#endif
        {
          size_t w = robust_fwrite((const char *)output.dst, output.pos, out);
          if (w != output.pos) die_io("short write to output (broken pipe?)");
        }
        if (m) m->wrote_bytes.fetch_add(output.pos, std::memory_order_relaxed);
      }

      if (last_chunk && remaining == 0) finished = true;
    } while (input.pos < input.size || (last_chunk && !finished));
  }

  ZSTD_freeCCtx(cctx);
  progress_done = true; progress_thr.join();
}

// True if the running kernel has per-VMA locks (Linux >= 6.4).  Before 6.4 a
// single mmap_lock rwsem serialises every page fault in an address space, so many
// worker threads faulting a large mmap'd input contend hard — a sys-time storm
// that scales with file size and core count (see project_mmap_kernel_storm /
// CHANGELOG).  On 6.4+ faults are lock-free and mmap is a win.
static bool kernel_has_per_vma_locks()
{
#ifdef __linux__
  struct utsname u;
  if (uname(&u) != 0) return true;                 // unknown: assume modern, keep mmap
  int major = 0, minor = 0;
  if (std::sscanf(u.release, "%d.%d", &major, &minor) != 2) return true;
  return major > 6 || (major == 6 && minor >= 4);
#else
  return true;
#endif
}

// Above this input size, on a pre-6.4 kernel, fread beats mmap (the mmap_lock
// fault-storm dominates); below it mmap's zero-copy wins and the fault volume is
// small.  Heuristic, pre-6.4 only; --mmap / --no-mmap override the whole gate.
static const size_t MMAP_PREVMA_MAX_BYTES = size_t(4) * 1024 * ONE_MIB;  // 4 GiB

// Whether to mmap `path`, applying the pre-6.4 large-file auto-gate.  Evaluated
// before MmapReader::open() so a gated input never gets mapped at all.
static bool mmap_ok_for_input(const Options & opt, const std::string & path)
{
  if (opt.mmap_user_set) return true;              // explicit --mmap/--no-mmap: honour it
  if (kernel_has_per_vma_locks()) return true;     // >= 6.4: mmap is safe
  std::error_code ec;
  uintmax_t fsz = fs::file_size(path, ec);
  if (ec || fsz <= MMAP_PREVMA_MAX_BYTES) return true;  // small/unknown: mmap fine
  if (opt.verbosity >= V_VERBOSE)
    vlog(V_VERBOSE, opt, "[MMAP] pre-6.4 kernel + large input ("
         + std::to_string(size_t(fsz / ONE_MIB)) + " MiB): skipping mmap to avoid the "
           "mmap_lock fault-storm (override with --mmap)\n");
  return false;
}

#ifndef _WIN32
// --direct-read: SINGLE-stream O_DIRECT reader.  Concurrent O_DIRECT reads contend
// hard on this class of NVMe — measured on a 432 GiB file: 1 stream 4.5 GB/s, 4
// independent streams only ~3.0 GB/s aggregate (0.77 GB/s each).  So the fast path
// is one uninterrupted stream; the v0.13.46/47 multi-threaded readers were a
// mistake here.  A single dd stream already saturates the drive (4.5 GB/s), so the
// only job left is to not stall that stream — hence zero-copy:
//
//   • pool != nullptr (CPU path): pread straight into a pooled aligned buffer and
//     hand it to emit(buf, n, idx, slot); the Task aliases it as a view and the
//     worker recycles the slot on release_input().  No per-chunk 16 MiB memcpy —
//     that copy, contending for memory bandwidth with the compressors, was what
//     held the stream to ~1/3 of the drive (1.5 vs 4.5 GB/s).  pool->acquire()
//     blocks when all buffers are in flight, so it is also the producer
//     backpressure (the queue byte-cap is a no-op for zero-byte view tasks).
//   • pool == nullptr (GPU path): reuse one scratch buffer; emit copies (it splits
//     the host chunk into gpu subchunks anyway, so a single owning buffer per chunk
//     doesn't fit — keep the copy there, where PCIe dominates regardless).
//
// O_DIRECT transfers straight disk→buffer, bypassing the page cache (no populate,
// no eviction) — honest-cold, no kcompactd, no warm-cache skew.  It can't go
// through mmap (mmap IS the cache), so this is its own reader.  emit(buf,n,idx,slot)
// returns false to abort (GPU failure).  The final partial chunk rides O_DIRECT's
// EOF short read.  Returns false if O_DIRECT couldn't be set up (caller falls back
// to fread).  seq is the chunk index (file position): output stays ordered, RAM
// bounded.
//
// o_direct == false (v0.13.60): same machinery through the PAGE CACHE — pread
// into a pooled buffer, kernel readahead intact.  Exists because on the
// large-memory server the fread fallback's hidden second copy (kernel→scratch,
// then scratch→task 16 MiB assign) halved effective intake (9.6 → ~5.7 GiB/s
// measured on a 432 GiB input) and starved 96 workers; buffered-pooled reads
// keep the 9.6 and drop the assign.  Use it when mmap is unavailable (pre-6.4
// kernel gate, --no-mmap): mmap remains strictly better where it's safe
// (zero copies vs one), and O_DIRECT remains better only when the device's
// raw rate beats its buffered rate (not true on the Gen5 box: 4.5 vs 9.6).
// n_readers > 1 (buffered pool mode only, v0.13.63): N threads pull chunk
// indices from a shared atomic counter and pread the SAME fd.  This breaks
// the single-thread cold-destination copy wall (~3.5 GB/s node-local on the
// dual-socket server) while keeping the device stream effectively
// sequential — the offsets arrive near-ordered, so one readahead context
// keeps working and the device never sees the O_DIRECT-style random
// contention (probe on that box: 2 buffered dd streams = 17.4 GB/s
// aggregate vs 9.9 single; the O_DIRECT 1-stream rule does NOT carry over
// to page-cache reads).  Interleaved indices — NOT partitioned file
// regions — bound the queue's seq skew to ~N: a partitioned design floods
// the ResultStore with distant-seq frames, exhausts FrameThrottle permits
// and then the pool, and starves the region the writer needs (the
// re_enqueue FIFO-invariant deadlock).  O_DIRECT always stays 1 stream.
template <typename Emit>   // template so the lambda inlines (and -Wnonnull sees the
                           // pooled/aligned buffer as non-null)
static bool pooled_read_chunks(const std::string & path, size_t host_chunk,
                               Meter * m, DirectReadPool * pool, bool o_direct,
                               int n_readers, int borrowed_fd, Emit && emit)
{
  // borrowed_fd >= 0: pread an already-open fd (an inherited, seekable stdin)
  // — don't close it.  Otherwise open the path ourselves (and close at exit).
  const bool owns_fd = (borrowed_fd < 0);
  int fd = owns_fd ? ::open(path.c_str(), O_RDONLY | (o_direct ? O_DIRECT : 0))
                   : borrowed_fd;
  if (fd < 0) return false;                 // O_DIRECT unsupported → caller falls back to fread
#ifdef POSIX_FADV_SEQUENTIAL
  if (!o_direct)
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);  // widen kernel readahead
#endif
  const size_t ALIGN = 4096;
  const size_t cap = ((host_chunk + ALIGN - 1) / ALIGN) * ALIGN;
  if (o_direct || !pool) n_readers = 1;     // O_DIRECT contends; scratch path has one buffer
  if (n_readers < 1) n_readers = 1;
  if (m) m->reader_threads.store(n_readers, std::memory_order_relaxed);

  void * scratch = nullptr;                 // only used when pool == nullptr (copy path)
  if (!pool) {
    if (posix_memalign(&scratch, ALIGN, cap) != 0 || !scratch) { if (owns_fd) ::close(fd); return false; }
  }

  std::atomic<size_t> next_idx{0};
  std::atomic<bool>   done{false};
  std::atomic<bool>   io_error{false};
  auto read_loop = [&]() {
    for (;;) {
      if (done.load(std::memory_order_relaxed)) break;
      const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      int slot = -1;
      char * buf;
      if (pool) {                           // blocks when all bufs in flight = backpressure
        const uint64_t a_t0 = m ? now_ns() : 0;
        slot = pool->acquire(); buf = pool->buf(slot);
        if (m) m->reader_blocked_ns.fetch_add(now_ns() - a_t0, std::memory_order_relaxed);
      }
      else      { buf = static_cast<char *>(scratch); }
      uint64_t t0 = (g_perf || m) ? now_ns() : 0;
      ssize_t got = ::pread(fd, buf, cap, (off_t)idx * (off_t)cap);
      if (got < 0) {
        if (pool) pool->release(slot);
        io_error.store(true, std::memory_order_relaxed);
        done.store(true, std::memory_order_relaxed);
        break;
      }
      if (got == 0) {                       // at/past EOF (exact-multiple end, or a
        if (pool) pool->release(slot);      // racing thread's idx beyond the short read)
        done.store(true, std::memory_order_relaxed);
        break;
      }
      if (g_perf || m) {
        const uint64_t r_dt = now_ns() - t0;
        if (m) m->reader_io_ns.fetch_add(r_dt, std::memory_order_relaxed);
        if (g_perf) { g_perf->read_ns.fetch_add(r_dt); g_perf->read_bytes_total.fetch_add((uint64_t)got); }
      }
      if (m) m->read_bytes.fetch_add((uint64_t)got);
      // emit takes ownership of `slot` in the pool path (the Task carries it and the
      // worker releases it); the reader must not touch the buffer after this.
      bool cont = emit(static_cast<const char *>(buf), (size_t)got, idx, slot);
      if (!cont || (size_t)got < cap) {     // abort, or short read ⇒ this was the last chunk
        done.store(true, std::memory_order_relaxed);
        break;
      }
    }
  };

  if (n_readers == 1) {
    read_loop();
  } else {
    std::vector<std::thread> rthr;
    rthr.reserve((size_t)n_readers - 1);
    for (int i = 1; i < n_readers; ++i) rthr.emplace_back(read_loop);
    read_loop();
    for (auto & th : rthr) th.join();
  }
  if (!pool) free(scratch);
  if (owns_fd) ::close(fd);   // borrowed (inherited stdin) fd: leave it open
  if (io_error.load(std::memory_order_relaxed))
    die_io(o_direct ? "O_DIRECT read failed (--direct-read)" : "pread failed (pooled reader)");
  return true;
}

// Copy a raw byte range into a Task and enqueue it.  Its own (non-inlined)
// function so vector::assign is analysed with a provably non-null source — inlined
// into the threaded O_DIRECT reader, GCC can't prove the alloc/source non-null and
// trips a -Wnonnull false positive on assign's memmove.
__attribute__((noinline))
static void enqueue_direct_chunk(TaskQueue & q, size_t seq, const char * buf, size_t n)
{
  if (!buf || n == 0) return;
  Task t;
  t.seq = seq;
  t.data.assign(buf, buf + n);
  q.push(std::move(t));
}
#endif

#ifndef _WIN32
// Set when --tar skips a member (permissions, vanished, socket) or a file
// changed during read.  main() promotes it to a non-zero exit (GNU tar parity).
static std::atomic<bool> g_tar_had_errors{false};

/*======================================================================
 --tar: native parallel GNU-format tar archive creation
 ----------------------------------------------------------------------
 `tar -cf - … | gzstd` is bottlenecked by tar: it walks the tree and reads
 every member on a single thread before gzstd ever sees a byte.  Instead we
 synthesize the tar byte stream ourselves and read member files in parallel,
 feeding the existing TaskQueue → workers → writer pipeline (CPU and GPU).

 Two phases:
   1. build_layout() — single-threaded walk: lstat every member, apply
      --exclude, detect hardlinks, and compute each member's byte offset in
      the virtual tar stream (header length + data + 512-byte padding are all
      known from metadata alone).  Result: an ordered entry table and the
      total stream length.
   2. assemble() — parallel: split the virtual stream into chunk_size chunks,
      read the file ranges each chunk overlaps with concurrent preads, and
      push chunks to the queue.  Pushes are emitted in strict sequence order
      (a tiny reorder buffer) so the FrameThrottle's FIFO invariant holds
      (see TaskQueue::re_enqueue) — only the reads run out of order.

 Output is a standard GNU-format .tar.zst, extractable by `tar --zstd -xf`
 or `gzstd -d | tar -x`.  Format details mirror `tar --format=gnu` (Ubuntu's
 default): ustar base block + OLDGNU magic, 'L'/'K' long-name/long-link
 entries for paths > 100 bytes, and base-256 numeric fields for files > 8 GiB
 or uid/gid > 2^21.
======================================================================*/
namespace tarx {

static constexpr size_t TAR_BLK    = 512;
static constexpr size_t TAR_RECORD = 10240;  // GNU default blocking factor (20 blocks)

static inline uint64_t tar_round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// One archive member.  Offsets are into the virtual (uncompressed) tar stream.
struct TarEntry {
  std::string path;       // member name (dirs carry a trailing '/')
  std::string link;       // symlink target or hardlink referent ("" otherwise)
  std::string src;        // filesystem path to pread (regular files); also set for
                          // every member when --acls/--xattrs is on (gather source)
  std::string pax;        // serialized PAX SCHILY.* records ("" = no extended header)
  uint64_t    size      = 0;   // data bytes (0 for dir/symlink/hardlink/special)
  uint64_t    hdr_off   = 0;   // offset of the (first) header block
  uint64_t    data_off  = 0;   // offset of file data = hdr_off + hdr_len
  uint64_t    entry_end = 0;   // data_off + round_up(size, 512)
  uint32_t    hdr_len   = (uint32_t)TAR_BLK;  // pax 'x' block + 512 + optional L/K
  uint32_t    mode      = 0;
  uint64_t    uid = 0, gid = 0;
  int64_t     mtime     = 0;
  uint64_t    rdev      = 0;   // char/block device nodes
  char        typeflag  = '0';
  // --sparse create (OLDGNU 'S'): sparse_map = (logical offset, length) per data
  // segment; real_size = logical file size.  size becomes the STORED data length
  // (Σ segment lengths); hdr_len includes the sparse extension blocks.  Empty map
  // = not sparse.  data region holds the concatenated segments only.
  uint64_t    real_size = 0;
  std::vector<std::pair<uint64_t, uint64_t>> sparse_map;
};

struct TarLayout {
  std::vector<TarEntry> entries;
  std::unordered_map<uint64_t, std::string> unames, gnames;  // uid/gid → name (walk-populated)
  uint64_t total_size = 0;     // full padded virtual tar stream length
  bool     had_errors = false; // a member was skipped (permissions, vanished, socket)
};

// total header bytes for a member with the given name/link lengths.
static uint32_t header_len(size_t name_len, size_t link_len) {
  uint32_t n = (uint32_t)TAR_BLK;
  if (name_len > 100) n += (uint32_t)(TAR_BLK + tar_round_up(name_len + 1, TAR_BLK));
  if (link_len > 100) n += (uint32_t)(TAR_BLK + tar_round_up(link_len + 1, TAR_BLK));
  return n;
}

// Write `val` into an n-byte tar numeric field: octal ASCII (n-1 digits + NUL)
// when it fits, else GNU base-256 (0x80 marker + big-endian binary).
static void put_num(char * f, size_t n, uint64_t val) {
  bool fits = (3 * (n - 1) >= 64) || (val < (uint64_t(1) << (3 * (n - 1))));
  if (fits) {
    for (size_t i = n - 1; i-- > 0; ) { f[i] = char('0' + (val & 7)); val >>= 3; }
    f[n - 1] = '\0';
  } else {
    for (size_t i = n; i-- > 1; ) { f[i] = char(val & 0xff); val >>= 8; }
    f[0] = char(0x80);
  }
}

// Emit one 512-byte ustar/OLDGNU header block.
static void emit_block(char * h, const std::string & name, uint32_t mode,
                       uint64_t uid, uint64_t gid, uint64_t size, uint64_t mtime,
                       char typeflag, const std::string & link,
                       const std::string & uname, const std::string & gname,
                       uint64_t devmajor, uint64_t devminor) {
  std::memset(h, 0, TAR_BLK);
  std::memcpy(h, name.data(), std::min<size_t>(name.size(), 100));
  put_num(h + 100, 8,  mode & 07777);
  put_num(h + 108, 8,  uid);
  put_num(h + 116, 8,  gid);
  put_num(h + 124, 12, size);
  put_num(h + 136, 12, mtime);
  h[156] = typeflag;
  if (!link.empty()) std::memcpy(h + 157, link.data(), std::min<size_t>(link.size(), 100));
  std::memcpy(h + 257, "ustar  ", 8);   // OLDGNU magic+version: "ustar  \0"
  if (!uname.empty()) std::memcpy(h + 265, uname.data(), std::min<size_t>(uname.size(), 31));
  if (!gname.empty()) std::memcpy(h + 297, gname.data(), std::min<size_t>(gname.size(), 31));
  if (typeflag == '3' || typeflag == '4') {
    put_num(h + 329, 8, devmajor);
    put_num(h + 337, 8, devminor);
  }
  // Checksum: sum of all 512 bytes with the chksum field counted as spaces.
  std::memset(h + 148, ' ', 8);
  uint64_t sum = 0;
  for (size_t i = 0; i < TAR_BLK; ++i) sum += (unsigned char)h[i];
  char cs[8];
  std::snprintf(cs, sizeof(cs), "%06o", (unsigned)(sum & 0777777));
  std::memcpy(h + 148, cs, 6);
  h[154] = '\0';
  h[155] = ' ';
}

// Emit a GNU long-name ('L') or long-link ('K') pseudo-entry: a header naming
// "././@LongLink" whose data is the full string (NUL-terminated, padded).
static size_t emit_longlink(char * out, const std::string & s, char type) {
  emit_block(out, "././@LongLink", 0, 0, 0, s.size() + 1, 0, type, "", "", "", 0, 0);
  size_t datalen = s.size() + 1;
  std::memcpy(out + TAR_BLK, s.data(), s.size());
  size_t padded = tar_round_up(datalen, TAR_BLK);
  std::memset(out + TAR_BLK + s.size(), 0, padded - s.size());  // NUL + pad
  return TAR_BLK + padded;
}

// ---- PAX extended headers: xattrs (--xattrs) and POSIX ACLs (--acls) ----
// Stored as SCHILY.* records inside an 'x'-typeflag pseudo-entry preceding the
// member, matching GNU tar so archives interoperate.  Everything here is gated
// by the flags (build_layout only gathers when --acls/--xattrs is given), so the
// default path pays nothing.

// One PAX record: "<len> <key>=<value>\n", where <len> counts its own digits.
static std::string pax_record(const std::string & key, const std::string & val) {
  size_t fixed = key.size() + val.size() + 3;   // ' ', '=', '\n'
  size_t len = fixed + 1;                        // + at least one length digit
  for (;;) {
    size_t d = std::to_string(len).size();
    if (fixed + d == len) break;
    len = fixed + d;
  }
  return std::to_string(len) + " " + key + "=" + val + "\n";
}

// Append SCHILY.xattr.* records for `path` to `recs` (nofollow — the member's
// own xattrs, not a symlink target's).  When ACLs are stored separately, the
// system.posix_acl_* attributes are skipped here (they ride in the ACL records
// instead) to avoid duplication, exactly as GNU tar does.
static void gather_xattrs(const std::string & path, bool acls_separate, std::string & recs) {
  ssize_t ln = ::llistxattr(path.c_str(), nullptr, 0);
  if (ln <= 0) return;
  std::vector<char> names((size_t)ln);
  ln = ::llistxattr(path.c_str(), names.data(), names.size());
  if (ln <= 0) return;
  for (size_t p = 0; p < (size_t)ln; ) {
    std::string name(&names[p]);
    p += name.size() + 1;
    if (name.empty()) continue;
    if (acls_separate && (name == "system.posix_acl_access" ||
                          name == "system.posix_acl_default")) continue;
    ssize_t vl = ::lgetxattr(path.c_str(), name.c_str(), nullptr, 0);
    if (vl < 0) continue;
    std::string val((size_t)vl, '\0');
    vl = ::lgetxattr(path.c_str(), name.c_str(), val.data(), val.size());
    if (vl < 0) continue;
    val.resize((size_t)vl);
    recs += pax_record("SCHILY.xattr." + name, val);
  }
}

#ifdef GZSTD_HAVE_ACL
// Append SCHILY.acl.access / SCHILY.acl.default records.  Only non-trivial ACLs
// are stored (a trivial ACL just mirrors the mode bits — storing it would bloat
// every file), matching GNU tar.  Default ACLs apply to directories only.  Text
// is the comma-separated long form, which GNU tar's acl_from_text() reads.
static void gather_acls(const std::string & path, bool is_dir, std::string & recs) {
  if (::acl_extended_file_nofollow(path.c_str()) > 0) {
    acl_t acc = ::acl_get_file(path.c_str(), ACL_TYPE_ACCESS);
    if (acc) {
      char * txt = ::acl_to_any_text(acc, nullptr, ',', 0);
      if (txt) { recs += pax_record("SCHILY.acl.access", txt); ::acl_free(txt); }
      ::acl_free(acc);
    }
  }
  if (is_dir) {
    acl_t def = ::acl_get_file(path.c_str(), ACL_TYPE_DEFAULT);
    if (def) {
      if (::acl_entries(def) > 0) {
        char * txt = ::acl_to_any_text(def, nullptr, ',', 0);
        if (txt) { recs += pax_record("SCHILY.acl.default", txt); ::acl_free(txt); }
      }
      ::acl_free(def);
    }
  }
}
#endif

// Emit a PAX 'x' extended-header pseudo-entry whose data is `recs`.
static size_t emit_pax_block(char * out, const std::string & recs, const std::string & path) {
  std::string nm = "PaxHeaders/" + path;        // cosmetic; applies to the next header
  emit_block(out, nm, 0644, 0, 0, recs.size(), 0, 'x', "", "", "", 0, 0);
  std::memcpy(out + TAR_BLK, recs.data(), recs.size());
  size_t padded = tar_round_up(recs.size(), TAR_BLK);
  std::memset(out + TAR_BLK + recs.size(), 0, padded - recs.size());
  return TAR_BLK + padded;
}

// Synthesize a member's full header (PAX 'x' block + long-name/link blocks + base
// block) into `out` (caller sizes it ≥ e.hdr_len).  Returns bytes written.
// Fill the OLDGNU sparse map into a base header already written by emit_block
// ('S' typeflag, size = stored): 4 segments at offset 386, isextended at 482,
// realsize at 483; recompute the checksum; then emit 512-byte extension blocks
// (21 segments each) for any remaining segments.  Returns base + extension bytes.
static size_t emit_oldgnu_sparse(char * h, const TarEntry & e) {
  const auto & m = e.sparse_map;
  size_t n = m.size(), in_base = std::min<size_t>(4, n);
  for (size_t i = 0; i < in_base; ++i) {
    put_num(h + 386 + i * 24,      12, m[i].first);
    put_num(h + 386 + i * 24 + 12, 12, m[i].second);
  }
  h[482] = (n > 4) ? 1 : 0;                 // isextended
  put_num(h + 483, 12, e.real_size);
  std::memset(h + 148, ' ', 8);             // recompute checksum
  uint64_t sum = 0; for (size_t i = 0; i < TAR_BLK; ++i) sum += (unsigned char)h[i];
  char cs[8]; std::snprintf(cs, sizeof(cs), "%06o", (unsigned)(sum & 0777777));
  std::memcpy(h + 148, cs, 6); h[154] = '\0'; h[155] = ' ';
  size_t off = TAR_BLK, idx = in_base;
  while (idx < n) {
    char * xb = h + off; std::memset(xb, 0, TAR_BLK);
    size_t cnt = std::min<size_t>(21, n - idx);
    for (size_t i = 0; i < cnt; ++i) {
      put_num(xb + i * 24,      12, m[idx + i].first);
      put_num(xb + i * 24 + 12, 12, m[idx + i].second);
    }
    idx += cnt;
    xb[504] = (idx < n) ? 1 : 0;            // isextended
    off += TAR_BLK;
  }
  return off;
}

static size_t emit_header(const TarEntry & e, const TarLayout & lay, char * out) {
  static const std::string empty;
  size_t off = 0;
  if (!e.pax.empty()) off += emit_pax_block(out + off, e.pax, e.path);
  if (e.path.size() > 100) off += emit_longlink(out + off, e.path, 'L');
  if (e.link.size() > 100) off += emit_longlink(out + off, e.link, 'K');
  auto uit = lay.unames.find(e.uid);
  auto git = lay.gnames.find(e.gid);
  emit_block(out + off, e.path, e.mode, e.uid, e.gid, e.size, (uint64_t)e.mtime,
             e.typeflag, e.link,
             uit != lay.unames.end() ? uit->second : empty,
             git != lay.gnames.end() ? git->second : empty,
             (uint64_t)major((dev_t)e.rdev), (uint64_t)minor((dev_t)e.rdev));
  if (e.typeflag == 'S' && !e.sparse_map.empty())
    return off + emit_oldgnu_sparse(out + off, e);
  return off + TAR_BLK;
}

// ---- Phase 1: walk the source tree and lay out the virtual tar stream. ----
struct LayoutBuilder {
  const Options & opt;
  Meter * m;
  TarLayout lay;
  uint64_t off = 0;
  std::map<std::pair<uint64_t, uint64_t>, std::string> hardlinks;  // (dev,ino) → first member
  bool warned_leading_slash = false;

  // Two-pass walk: Pass A (enumerate) fills `pending_` in canonical order using
  // readdir's d_type (no lstat on leaves); Pass B (stat_pending) lstats every
  // entry IN PARALLEL — the cold-inode storm, ~20% of cold --tar wall time; see
  // project-tar-read-path; Pass C (finalize_entries) does the order-sensitive
  // bookkeeping (hardlinks, owners, offsets) serially on the stat'd results, so
  // the archive stays byte-for-byte identical to the old serial walk.
  struct Pending {
    std::string fspath, member;
    dev_t       boundary_dev;
    bool        is_dir;
    // filled by Pass B (all default-initialized so partial aggregate-init is clean):
    struct stat st {};
    std::string link {};         // readlink target (symlinks)
    int         stat_err = 0;    // 0 ok, else errno from lstat
    int         link_err = 0;    // 0 ok, else errno from readlink
    std::vector<std::pair<uint64_t, uint64_t>> sparse_map {};  // --sparse: data segments
  };
  std::vector<Pending> pending_;

  LayoutBuilder(const Options & o, Meter * mm) : opt(o), m(mm) {}

  void resolve_owner(uint64_t uid, uint64_t gid) {
    if (opt.tar_numeric_owner) return;
    if (lay.unames.find(uid) == lay.unames.end()) {
      std::string nm; char buf[1024]; struct passwd pw, *res = nullptr;
      if (getpwuid_r((uid_t)uid, &pw, buf, sizeof buf, &res) == 0 && res) nm = pw.pw_name;
      lay.unames.emplace(uid, std::move(nm));
    }
    if (lay.gnames.find(gid) == lay.gnames.end()) {
      std::string nm; char buf[1024]; struct group gr, *res = nullptr;
      if (getgrgid_r((gid_t)gid, &gr, buf, sizeof buf, &res) == 0 && res) nm = gr.gr_name;
      lay.gnames.emplace(gid, std::move(nm));
    }
  }

  void add(TarEntry e) {
    e.hdr_off   = off;
    e.hdr_len   = header_len(e.path.size(), e.link.size());
    // OLDGNU sparse: 4 segments fit in the base header; the rest spill into
    // 512-byte extension blocks (21 segments each) appended after it.
    if (!e.sparse_map.empty() && e.sparse_map.size() > 4)
      e.hdr_len += (uint32_t)(((e.sparse_map.size() - 4 + 20) / 21) * TAR_BLK);
    e.data_off  = off + e.hdr_len;
    e.entry_end = e.data_off + tar_round_up(e.size, TAR_BLK);
    off = e.entry_end;
    lay.entries.push_back(std::move(e));
  }

  // True if the path being walked is excluded.  Matches against the *filesystem
  // path* (absolute when the source is absolute), exactly like GNU tar — so
  // `--exclude=/proc` works when archiving `/`.  GNU --exclude defaults: '*'
  // spans '/' (no FNM_PATHNAME); FNM_LEADING_DIR so a directory pattern drops
  // its whole subtree; non-anchored — the pattern may match the full path OR any
  // component-suffix after a '/' (so bare `.cache` drops `a/.cache`).  A pattern
  // with a leading '/' can only match at the full path (its slash can't match a
  // mid-path suffix start), so `/tmp` is effectively anchored to the source root
  // and does NOT drop `var/tmp` — matching GNU tar.  Trailing '/' is ignored.
  bool excluded(const std::string & path) const {
    if (opt.tar_excludes.empty()) return false;
    std::string name = path;
    if (name.size() > 1 && name.back() == '/') name.pop_back();
    for (const std::string & pat : opt.tar_excludes) {
      std::string p = pat;
      if (p.size() > 1 && p.back() == '/') p.pop_back();
      for (size_t pos = 0; ; ) {
        if (fnmatch(p.c_str(), name.c_str() + pos, FNM_LEADING_DIR) == 0) return true;
        size_t slash = name.find('/', pos);
        if (slash == std::string::npos) break;
        pos = slash + 1;
      }
    }
    return false;
  }

  // A genuine failure (cannot stat/open/readlink, file changed mid-read): shown
  // by default and promoted to a non-zero exit, like GNU tar's errors.
  void warn_skip(const std::string & path, const std::string & why) {
    lay.had_errors = true;
    g_tar_had_errors.store(true, std::memory_order_relaxed);
    vlog(V_ERROR, opt, "gzstd: tar: skipping " + path + ": " + why + "\n");
  }

  // A benign skip (socket, unknown file type) — GNU tar's "file ignored"
  // warning class.  Expected on a full-system backup (sockets in /run, /tmp),
  // so it is informational only: shown at -v, never an error or exit change.
  // This makes the default behaviour equivalent to GNU tar's
  // --warning=no-file-ignored without needing a flag.
  void note_skip(const std::string & path, const std::string & why) {
    vlog(V_VERBOSE, opt, "gzstd: tar: ignoring " + path + ": " + why + "\n");
  }

  // fspath: filesystem path to stat/read.  member: name to store in the archive.
  // boundary_dev: (dev_t)-1 until --one-file-system fixes it to a source's root.
  // ---- Pass A: enumerate the tree into pending_ in canonical order. ----
  // dtype is the parent readdir's d_type for this entry (DT_UNKNOWN for the
  // top-level sources or filesystems that don't report it).  We avoid lstat
  // here: d_type tells us whether to recurse, and leaf metadata is deferred to
  // Pass B.  An lstat happens in Pass A only when unavoidable: DT_UNKNOWN (to
  // learn dir-ness) or --one-file-system on a directory (st_dev drives the
  // mount-crossing / descent decision, which must be known before recursing).
  void enumerate(const std::string & fspath, const std::string & member,
                 dev_t boundary_dev, int dtype) {
    if (excluded(fspath)) return;

    bool is_dir;
    struct stat ast {}; bool have_ast = false;
    if (dtype == DT_DIR) {
      is_dir = true;
    } else if (dtype == DT_UNKNOWN) {
      if (::lstat(fspath.c_str(), &ast) != 0) { warn_skip(fspath, std::strerror(errno)); return; }
      have_ast = true; is_dir = S_ISDIR(ast.st_mode);
    } else {
      is_dir = false;  // DT_REG/DT_LNK/DT_CHR/DT_BLK/DT_FIFO/DT_SOCK
    }

    if (!is_dir) {
      // Leaf.  The --one-file-system cross-check is deferred to Pass C (a leaf
      // is on its parent's filesystem unless it is itself a bind-mount, the rare
      // case Pass C still catches via the parallel-lstat st_dev).  No lstat here.
      pending_.push_back({fspath, member, boundary_dev, false});  // rest default-init
      return;
    }

    // Directory.  Under --one-file-system we need st_dev now to decide crossing
    // (before descending) and to seed a source root's boundary.
    bool cross = false;
    dev_t child_boundary = boundary_dev;
    if (opt.tar_one_file_system) {
      if (!have_ast) {
        if (::lstat(fspath.c_str(), &ast) != 0) { warn_skip(fspath, std::strerror(errno)); return; }
        have_ast = true;
      }
      if (boundary_dev != (dev_t)-1) cross = (ast.st_dev != boundary_dev);
      else                          child_boundary = ast.st_dev;  // source root sets the boundary
    }

    // Record the directory itself.  Skip the root placeholder (empty member):
    // an absolute source's members are stored leading-'/'-stripped with no root
    // entry, matching GNU tar (`/` → `proc/`, `home/...`, not `./proc`).
    if (!member.empty())
      pending_.push_back({fspath, member, boundary_dev, true});   // rest default-init

    if (cross) return;  // mount-point stub recorded; do not descend

    DIR * d = ::opendir(fspath.c_str());
    if (!d) { warn_skip(fspath, std::strerror(errno)); return; }
    std::vector<std::pair<std::string, int>> kids;  // (name, d_type)
    for (struct dirent * de; (de = ::readdir(d)); ) {
      if (std::strcmp(de->d_name, ".") == 0 || std::strcmp(de->d_name, "..") == 0) continue;
      kids.emplace_back(de->d_name, (int)de->d_type);
    }
    ::closedir(d);
    std::sort(kids.begin(), kids.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });  // canonical order
    std::string base = member;
    if (!base.empty() && base.back() == '/') base.pop_back();
    const std::string sep = (!fspath.empty() && fspath.back() == '/') ? "" : "/";
    for (const auto & k : kids)
      enumerate(fspath + sep + k.first,
                base.empty() ? k.first : base + "/" + k.first,
                child_boundary, k.second);
  }

  // --sparse: build a regular file's data-segment map from filesystem hole
  // metadata via SEEK_DATA/SEEK_HOLE (no data is read).  Leaves p.sparse_map
  // empty (→ stored normally) if the file has no holes, or if the fs/seek
  // doesn't support hole queries (graceful fallback to full storage).
  void probe_sparse(Pending & p) {
    int fd = ::open(p.fspath.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    const off_t size = p.st.st_size;
    std::vector<std::pair<uint64_t, uint64_t>> segs;
    off_t pos = 0; bool has_hole = false, ok = true;
    while (pos < size) {
      off_t data = ::lseek(fd, pos, SEEK_DATA);
      if (data < 0) {
        if (errno == ENXIO) has_hole = true;  // pos..EOF is a trailing hole
        else ok = false;                       // SEEK_DATA unsupported → fall back
        break;
      }
      if (data > pos) has_hole = true;          // hole before this data run
      off_t hole = ::lseek(fd, data, SEEK_HOLE);
      if (hole < 0) hole = size;
      if (hole > data) segs.emplace_back((uint64_t)data, (uint64_t)(hole - data));
      if (hole < size) has_hole = true;         // hole after this data run
      pos = hole;
    }
    ::close(fd);
    if (ok && has_hole) {
      // GNU tar terminates the segment map with a zero-length entry at the real
      // size; emit it too so GNU tar parses our archive correctly.
      segs.emplace_back((uint64_t)size, 0);
      p.sparse_map = std::move(segs);
    }
  }

  // ---- Pass B: lstat (+ readlink) every pending entry IN PARALLEL. ----
  // Disjoint slots, no shared state, so no locking — the cold-inode storm runs
  // concurrently.  errno is captured per-slot for Pass C's ordered warnings.
  void stat_pending() {
    const size_t n = pending_.size();
    if (n == 0) return;
    int nthreads = std::max(1, resolve_cpu_threads(opt.cpu_threads));
    nthreads = std::min<int>(nthreads, (int)n);
    std::atomic<size_t> idx{0};
    auto worker = [&] {
      for (size_t i = idx.fetch_add(1); i < n; i = idx.fetch_add(1)) {
        Pending & p = pending_[i];
        if (::lstat(p.fspath.c_str(), &p.st) != 0) { p.stat_err = errno; continue; }
        if (S_ISLNK(p.st.st_mode)) {
          std::vector<char> tgt((size_t)p.st.st_size + 1);
          ssize_t r = ::readlink(p.fspath.c_str(), tgt.data(), tgt.size());
          if (r < 0) p.link_err = errno;
          else       p.link.assign(tgt.data(), (size_t)r);
        }
        // --sparse: probe a regular file's hole structure via SEEK_DATA/SEEK_HOLE
        // (filesystem extent metadata — reads no data).  A file with real holes
        // gets >1 segment or a trailing hole; a fully-allocated file yields one
        // segment spanning [0,size) and is treated as non-sparse (single segment).
        if (opt.sparse_mode == 1 && S_ISREG(p.st.st_mode) && p.st.st_nlink <= 1
            && p.st.st_size > 0)
          probe_sparse(p);
      }
    };
    if (nthreads <= 1) worker();
    else {
      std::vector<std::thread> ths;
      for (int t = 0; t < nthreads; ++t) ths.emplace_back(worker);
      for (auto & t : ths) t.join();
    }
  }

  // ---- Pass C: order-sensitive bookkeeping on the stat'd results. ----
  // Iterates pending_ in canonical order, so hardlink first-occurrence, owner
  // resolution, and the offset prefix-sum are byte-for-byte identical to the old
  // serial walk — only the lstat I/O moved to the parallel Pass B.
  void finalize_entries() {
    for (Pending & p : pending_) {
      if (p.stat_err != 0) { warn_skip(p.fspath, std::strerror(p.stat_err)); continue; }
      const struct stat & st = p.st;

      if (p.is_dir) {
        TarEntry e;
        e.path = p.member;
        if (e.path.empty() || e.path.back() != '/') e.path += '/';
        e.mode = st.st_mode; e.uid = st.st_uid; e.gid = st.st_gid; e.mtime = st.st_mtime;
        e.typeflag = '5';
        if (opt.tar_xattrs || opt.tar_acls) e.src = p.fspath;
        resolve_owner(e.uid, e.gid); add(std::move(e));
        continue;
      }

      // Leaf --one-file-system cross-check (deferred from Pass A): skip a leaf on
      // a different filesystem than its boundary (a file bind-mount).
      if (opt.tar_one_file_system && p.boundary_dev != (dev_t)-1
          && st.st_dev != p.boundary_dev) continue;

      TarEntry e;
      e.path = p.member; e.mode = st.st_mode; e.uid = st.st_uid; e.gid = st.st_gid;
      e.mtime = st.st_mtime;
      if (opt.tar_xattrs || opt.tar_acls) e.src = p.fspath;

      if (S_ISREG(st.st_mode)) {
        if (st.st_nlink > 1) {
          auto key = std::make_pair((uint64_t)st.st_dev, (uint64_t)st.st_ino);
          auto it = hardlinks.find(key);
          if (it != hardlinks.end()) {
            e.typeflag = '1'; e.link = it->second; e.size = 0;
            resolve_owner(e.uid, e.gid); add(std::move(e)); continue;
          }
          hardlinks.emplace(key, p.member);  // first occurrence stores the data
        }
        e.typeflag = '0'; e.size = (uint64_t)st.st_size; e.src = p.fspath;
        // --sparse: this file has holes (Pass B built the segment map).  Store
        // as OLDGNU 'S': size = Σ segment lengths (data only); real_size = logical.
        if (!p.sparse_map.empty()) {
          e.typeflag = 'S';
          e.real_size = (uint64_t)st.st_size;
          e.sparse_map = std::move(p.sparse_map);
          uint64_t stored = 0; for (auto & s : e.sparse_map) stored += s.second;
          e.size = stored;
        }
      } else if (S_ISLNK(st.st_mode)) {
        if (p.link_err != 0) { warn_skip(p.fspath, std::strerror(p.link_err)); continue; }
        e.typeflag = '2'; e.link = p.link; e.size = 0;
      } else if (S_ISCHR(st.st_mode)) {
        e.typeflag = '3'; e.rdev = st.st_rdev;
      } else if (S_ISBLK(st.st_mode)) {
        e.typeflag = '4'; e.rdev = st.st_rdev;
      } else if (S_ISFIFO(st.st_mode)) {
        e.typeflag = '6';
      } else {
        note_skip(p.fspath, S_ISSOCK(st.st_mode) ? "socket" : "unknown file type");
        continue;
      }
      resolve_owner(e.uid, e.gid); add(std::move(e));
    }
    std::vector<Pending>().swap(pending_);  // release the transient enumeration buffer
  }

  // --acls/--xattrs: gather extended metadata for every member, then fold each
  // member's PAX 'x' block into the offset layout.  The gather (one acl_get_file
  // + llistxattr/lgetxattr per member) is the expensive part, so it runs in
  // parallel across the entry list — entries are independent.  The offset
  // recompute is a cheap serial pass afterward (pax block sizes are now known).
  // Only called when a flag is set, so the default path never touches this.
  void apply_extended_metadata() {
    const size_t n = lay.entries.size();
    if (n == 0) return;
    int nthreads = std::max(1, resolve_cpu_threads(opt.cpu_threads));
    nthreads = std::min<int>(nthreads, (int)n);
    std::atomic<size_t> idx{0};
    auto worker = [&] {
      for (size_t i = idx.fetch_add(1); i < n; i = idx.fetch_add(1)) {
        TarEntry & e = lay.entries[i];
        if (e.src.empty() || e.typeflag == '1') continue;  // hardlink ref shares the
                                                           // first occurrence's metadata
        std::string recs;
        if (opt.tar_xattrs) gather_xattrs(e.src, opt.tar_acls, recs);
#ifdef GZSTD_HAVE_ACL
        if (opt.tar_acls)   gather_acls(e.src, e.typeflag == '5', recs);
#endif
        e.pax = std::move(recs);
      }
    };
    if (nthreads <= 1) worker();
    else {
      std::vector<std::thread> ths;
      for (int t = 0; t < nthreads; ++t) ths.emplace_back(worker);
      for (auto & t : ths) t.join();
    }
    // Serial recompute: insert each entry's PAX block ahead of its header.
    uint64_t o = 0;
    for (TarEntry & e : lay.entries) {
      uint32_t pax_blk = e.pax.empty()
          ? 0u : (uint32_t)(TAR_BLK + tar_round_up(e.pax.size(), TAR_BLK));
      e.hdr_off   = o;
      e.hdr_len   = pax_blk + header_len(e.path.size(), e.link.size());
      e.data_off  = o + e.hdr_len;
      e.entry_end = e.data_off + tar_round_up(e.size, TAR_BLK);
      o = e.entry_end;
    }
    off = o;
  }

  void finalize() {
    // Two zero blocks (end-of-archive marker) then pad to a full record.
    lay.total_size = tar_round_up(off + 2 * TAR_BLK, TAR_RECORD);
  }
};

// Strip leading slashes from a command-line source so members are stored
// relative (GNU tar default).  Warns once.
static std::string strip_leading_slash(std::string s, const Options & opt, bool & warned) {
  size_t i = 0;
  while (i < s.size() && s[i] == '/') ++i;
  if (i > 0) {
    if (!warned) {
      vlog(V_ERROR, opt, "gzstd: tar: removing leading '/' from member names\n");
      warned = true;
    }
    s = s.substr(i);
  }
  return s;
}

static TarLayout build_layout(const Options & opt, Meter * m) {
  using _clk = std::chrono::steady_clock;
  auto _secs = [](_clk::time_point a, _clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
  };
  LayoutBuilder b(opt, m);
  if (opt.verbosity >= V_VERBOSE)
    vlog(V_VERBOSE, opt, "[TAR] scanning " + std::to_string(opt.tar_sources.size())
         + " source path(s)\n");
  // Pass A: serial enumeration (canonical order, no leaf lstat).
  auto t_enum0 = _clk::now();
  for (std::string src : opt.tar_sources) {
    // Normalize: drop trailing slashes (keep a lone "/"), strip leading slash.
    // An empty member (source was "/") means no root entry — children are stored
    // directly as "proc/", "home/...", matching GNU tar archiving "/".
    while (src.size() > 1 && src.back() == '/') src.pop_back();
    std::string member = strip_leading_slash(src, opt, b.warned_leading_slash);
    b.enumerate(src, member, (dev_t)-1, DT_UNKNOWN);
  }
  double enum_s = _secs(t_enum0, _clk::now());
  // Pass B: parallel lstat (the cold-inode storm).
  auto t_stat0 = _clk::now();
  b.stat_pending();
  double stat_s = _secs(t_stat0, _clk::now());
  // Pass C: serial finalize (hardlinks, owners, offsets — byte-stable order).
  b.finalize_entries();
  double walk_s = _secs(t_enum0, _clk::now());
  double gather_s = 0.0;
  if (opt.tar_xattrs || opt.tar_acls) {
    auto t_g0 = _clk::now();
    b.apply_extended_metadata();
    gather_s = _secs(t_g0, _clk::now());
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, std::string("[TAR] gathered ")
           + (opt.tar_acls ? "ACLs " : "") + (opt.tar_xattrs ? "xattrs " : "")
           + "(parallel)\n");
  }
  b.finalize();
  if (opt.verbosity >= V_VERBOSE) {
    char sz[64]; human_bytes(double(b.lay.total_size), sz, sizeof(sz));
    vlog(V_VERBOSE, opt, "[TAR] " + std::to_string(b.lay.entries.size())
         + " entries, " + sz + " uncompressed tar stream\n");
    // Phase split: enum = serial readdir tree scan; stat = parallel lstat pass
    // (Pass B); walk = enum+stat+finalize; gather = parallel ACL/xattr pass.
    char tbuf[160];
    std::snprintf(tbuf, sizeof(tbuf),
                  "[TIMING] layout walk=%.2fs (enum=%.2fs stat=%.2fs) gather=%.2fs\n",
                  walk_s, enum_s, stat_s, gather_s);
    vlog(V_VERBOSE, opt, tbuf);
  }
  return std::move(b.lay);
}

// ---- Phase 2: assemble the tar stream into seq-ordered chunks, in parallel. ----
// Reads run concurrently; a reorder buffer + single pusher emit chunks to the
// queue in strict sequence order so the FrameThrottle FIFO invariant holds.
static void assemble(TaskQueue & queue, const TarLayout & lay, size_t chunk_size,
                     const Options & opt, Meter * m) {
  const uint64_t total = lay.total_size;
  const size_t nchunks = chunk_size ? (size_t)((total + chunk_size - 1) / chunk_size) : 0;
  if (nchunks == 0) return;

  int hw = resolve_cpu_threads(opt.cpu_threads);
  int nreaders = opt.read_threads > 0 ? (int)opt.read_threads
                                      : std::max(3, std::min(12, hw / 8));
  nreaders = std::max(1, std::min<int>(nreaders, (int)nchunks));
  if (opt.verbosity >= V_VERBOSE)
    vlog(V_VERBOSE, opt, "[TAR] " + std::to_string(nreaders) + " parallel reader(s), "
         + std::to_string(nchunks) + " chunks\n");

  std::atomic<size_t> next_chunk{0};

  // Reorder buffer: readers deposit completed chunks here; a pusher drains them
  // to the queue in strict order.  Readers may only CLAIM a chunk once it is
  // within `window` of the push frontier (push_next) — this keeps the head
  // chunk always depositable (a deposit never blocks), so the head reader can't
  // be wedged behind later chunks.  window ≥ nreaders keeps every reader busy.
  const size_t window = std::max<size_t>(8, (size_t)nreaders * 2);
  std::mutex rb_mx;
  std::condition_variable rb_cv;        // signals: chunk arrived / frontier advanced
  std::map<size_t, Task> ready;
  size_t push_next = 0;

  // Assemble one chunk [a,b) into a zero-filled owning buffer.
  auto assemble_chunk = [&](size_t ci, std::string & cur_path, int & cur_fd, int & cur_errno) -> Task {
    uint64_t a = (uint64_t)ci * chunk_size;
    uint64_t b = std::min(total, a + chunk_size);
    size_t clen = (size_t)(b - a);
    Task t; t.seq = ci; t.data.assign(clen, 0);
    char * outp = t.data.data();
    // First entry whose data/padding could reach into this chunk (entry_end > a).
    auto it = std::upper_bound(lay.entries.begin(), lay.entries.end(), a,
                [](uint64_t v, const TarEntry & e) { return v < e.entry_end; });
    std::vector<char> hdrbuf;
    for (; it != lay.entries.end() && it->hdr_off < b; ++it) {
      const TarEntry & e = *it;
      // Header region.
      uint64_t he = e.hdr_off + e.hdr_len;
      if (e.hdr_off < b && he > a) {
        if (hdrbuf.size() < e.hdr_len) hdrbuf.resize(e.hdr_len);
        emit_header(e, lay, hdrbuf.data());
        uint64_t s = std::max<uint64_t>(e.hdr_off, a), en = std::min<uint64_t>(he, b);
        std::memcpy(outp + (s - a), hdrbuf.data() + (s - e.hdr_off), (size_t)(en - s));
      }
      // Data region (file bytes); padding stays zero.  For a sparse entry the
      // data region holds the concatenated data segments, so a virtual (stored)
      // offset maps to a file offset via the segment map; non-sparse is linear.
      if (e.size > 0) {
        uint64_t de = e.data_off + e.size;
        if (e.data_off < b && de > a) {
          if (cur_path != e.src) {
            if (cur_fd >= 0) ::close(cur_fd);
            cur_fd = ::open(e.src.c_str(), O_RDONLY);
            cur_errno = (cur_fd < 0) ? errno : 0;
            cur_path = e.src;
          }
          // Read `len` bytes from file offset `foff` into the chunk at virtual
          // offset `voff`.  On short read the remainder stays zero (GNU parity).
          auto read_seg = [&](off_t foff, uint64_t voff, size_t len) {
            size_t done = 0;
            if (cur_fd >= 0) {
              while (done < len) {
                ssize_t r = ::pread(cur_fd, outp + (voff - a) + done, len - done, foff + (off_t)done);
                if (r <= 0) break;
                done += (size_t)r;
              }
            }
            if (done < len) {
              g_tar_had_errors.store(true, std::memory_order_relaxed);
              if (cur_fd < 0)
                vlog(V_ERROR, opt, "gzstd: tar: " + e.src + ": cannot read ("
                     + std::strerror(cur_errno) + ")\n");
              else
                vlog(V_ERROR, opt, "gzstd: tar: " + e.src + ": file changed as we read it\n");
            }
          };
          uint64_t s = std::max<uint64_t>(e.data_off, a), en = std::min<uint64_t>(de, b);
          if (e.sparse_map.empty()) {
            read_seg((off_t)(s - e.data_off), s, (size_t)(en - s));
          } else {
            uint64_t ss = s - e.data_off, se = en - e.data_off, cum = 0;
            for (const auto & seg : e.sparse_map) {
              uint64_t seg_end = cum + seg.second;
              uint64_t lo = std::max(cum, ss), hi = std::min(seg_end, se);
              if (lo < hi) read_seg((off_t)(seg.first + (lo - cum)), e.data_off + lo, (size_t)(hi - lo));
              cum = seg_end;
              if (cum >= se) break;
            }
          }
        }
      }
    }
    if (m) m->read_bytes.fetch_add(clen);  // tar-stream bytes produced (progress %)
    return t;
  };

  auto reader = [&]() {
    std::string cur_path; int cur_fd = -1, cur_errno = 0;
    for (;;) {
      size_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
      if (ci >= nchunks) break;
      {  // throttle read-ahead to `window` chunks past the push frontier
        std::unique_lock<std::mutex> lk(rb_mx);
        rb_cv.wait(lk, [&] { return ci < push_next + window; });
      }
      Task t = assemble_chunk(ci, cur_path, cur_fd, cur_errno);  // concurrent preads (unlocked)
      {
        std::unique_lock<std::mutex> lk(rb_mx);
        ready.emplace(ci, std::move(t));
      }
      rb_cv.notify_all();
    }
    if (cur_fd >= 0) ::close(cur_fd);
  };

  // Pusher: drain `ready` to the queue in strict sequence order.
  auto pusher = [&]() {
    for (;;) {
      Task t;
      {
        std::unique_lock<std::mutex> lk(rb_mx);
        rb_cv.wait(lk, [&] { return !ready.empty() && ready.begin()->first == push_next; });
        t = std::move(ready.begin()->second);
        ready.erase(ready.begin());
        ++push_next;
      }
      rb_cv.notify_all();        // advance the frontier: unblock readers waiting to claim
      queue.push(std::move(t));  // may block on the queue's byte cap (backpressure)
      if (push_next >= nchunks) break;
    }
  };

  std::vector<std::thread> pool;
  pool.reserve((size_t)nreaders);
  for (int i = 0; i < nreaders; ++i) pool.emplace_back(reader);
  std::thread push_thr(pusher);
  for (auto & th : pool) th.join();
  push_thr.join();
}

/*======================================================================
 Extraction: parse the decompressed tar stream and write files securely.
 ----------------------------------------------------------------------
 The archive's zstd stream is decompressed by the existing parallel pipeline
 (writing the tar byte stream to a pipe); this consumer reads that pipe.  A
 single parser thread walks the serial tar stream (tar is inherently serial),
 handles directories/symlinks/specials inline, and dispatches regular-file
 writes to a worker pool — so file I/O runs in parallel with decompression.

 Security: every filesystem op goes through the destination dir fd using
 openat/mkdirat/symlinkat with O_NOFOLLOW on EVERY path component, so a member
 symlink can never be traversed to escape the destination (the classic
 symlink-escape CVE).  `..` components are rejected outright (path traversal).
======================================================================*/

// Read an octal or GNU base-256 tar numeric field.
static uint64_t get_num(const char * f, size_t n) {
  if ((unsigned char)f[0] & 0x80) {                  // GNU base-256, big-endian
    uint64_t v = (uint64_t)((unsigned char)f[0] & 0x7f);
    for (size_t i = 1; i < n; ++i) v = (v << 8) | (unsigned char)f[i];
    return v;
  }
  uint64_t v = 0; size_t i = 0;
  while (i < n && (f[i] == ' ' || f[i] == '\0')) ++i;
  for (; i < n && f[i] >= '0' && f[i] <= '7'; ++i) v = (v << 3) | (uint64_t)(f[i] - '0');
  return v;
}

// Read a NUL-terminated (or full-width) string field.
static std::string get_str(const char * f, size_t n) {
  size_t len = 0; while (len < n && f[len] != '\0') ++len;
  return std::string(f, len);
}

// Extended metadata restored under --xattrs/--acls (from PAX SCHILY.* records).
struct ExtMeta {
  std::vector<std::pair<std::string, std::string>> xattrs;  // (name, value)
  std::string acl_access, acl_default;                      // POSIX ACL text
  bool empty() const { return xattrs.empty() && acl_access.empty() && acl_default.empty(); }
};

// A parsed tar entry (metadata only).
struct InEntry {
  std::string name, linkname, uname, gname;   // uname/gname: owner/group names (-l --tar)
  uint64_t size = 0, uid = 0, gid = 0;   // size = bytes stored in the stream
  uint32_t mode = 0, devmajor = 0, devminor = 0;
  int64_t  mtime = 0;
  char     typeflag = '0';
  ExtMeta  ext;
  // GNU sparse ('S' / GNU.sparse.*): the file's data is stored as segments, not
  // a linear copy.  sparse_map = (logical offset, length) per stored segment;
  // real_size = the file's logical size (ftruncate target).
  bool      is_sparse = false;
  uint64_t  real_size = 0;
  std::vector<std::pair<uint64_t, uint64_t>> sparse_map;
  bool      sparse_map_in_data = false;  // PAX GNU.sparse.1.0: map is a text block
                                         // prefixed to the file data, not in sparse_map
};

// Buffered reader over a fd (the decompression pipe).
struct StreamReader {
  // Two interchangeable sources behind one interface: an fd (the extract pipe)
  // or an in-memory FrameSink (the `-t --tar` validator — see FrameSink).  In
  // sink mode `base` points straight into the current frame buffer (no copy)
  // and skip() advances past whole frames without touching their bytes; that is
  // the random-access win over the old "drain the pipe" path.
  int fd; std::vector<char> buf; size_t pos = 0, fill = 0; bool eof = false;
  FrameSink * src = nullptr;     // in-memory frame source (null = fd mode)
  FrameBuf cur;                  // current frame (sink mode); keeps `base` alive
  const char * base = nullptr;   // read cursor base: into buf (fd) or cur (sink)
  uint64_t consumed_total = 0;   // --keep-going: running offset in the tar byte stream
  explicit StreamReader(int f) : fd(f), buf(1 << 20) { base = buf.data(); }
  explicit StreamReader(FrameSink * s) : fd(-1), src(s) {}
  void refill() {
    pos = 0; fill = 0;
    if (src) {
      cur = src->pop();
      if (!cur) { eof = true; return; }
      base = cur->data(); fill = cur->size();
      return;
    }
    ssize_t r; do { r = ::read(fd, buf.data(), buf.size()); } while (r < 0 && errno == EINTR);
    if (r <= 0) { eof = true; return; }
    base = buf.data(); fill = (size_t)r;
  }
  size_t read_some(char * dst, size_t n) {
    if (pos == fill) { refill(); if (pos == fill) return 0; }
    size_t k = std::min(fill - pos, n);
    std::memcpy(dst, base + pos, k); pos += k; consumed_total += k; return k;
  }
  bool read_exact(char * dst, size_t n) {
    size_t got = 0; while (got < n) { size_t k = read_some(dst + got, n - got); if (!k) return false; got += k; }
    return true;
  }
  // Advance n bytes without copying: just move the cursor, refilling (which in
  // sink mode drops the spent frame and pulls the next) as buffers run out.
  bool skip(uint64_t n) {
    while (n) {
      if (pos == fill) { refill(); if (pos == fill) return false; }
      size_t k = (size_t)std::min<uint64_t>(n, fill - pos);
      pos += k; n -= k; consumed_total += k;
    }
    return true;
  }
  void drain() { while (true) { if (pos == fill) { refill(); if (pos == fill) return; } consumed_total += (fill - pos); pos = fill; } }  // consume to EOF (unblocks producer)
};

class Extractor {
public:
  // validate_only: parse and structurally verify the tar (header checksums,
  // sizes, completeness) WITHOUT writing any files — for `gzstd -t --tar`.
  // list_only: parse and print each entry (tar -tvf style) to stdout, write
  // nothing — for `gzstd -l --tar`.  Both are read-only walks (no writer pool).
  Extractor(int dest_fd, const Options & opt, Meter * m, bool validate_only = false,
            bool list_only = false)
    : dest_fd_(dest_fd), opt_(opt), m_(m), as_root_(::geteuid() == 0),
      validate_only_(validate_only), list_only_(list_only) {}
  ~Extractor() {
    // Defensive: stop_pool() normally joins the writer; ensure it's down before
    // freeing the buffers it writes from (e.g. an early-return/exception path).
    if (ld_writer_.joinable()) {
      { std::lock_guard<std::mutex> lk(ld_mx_); ld_done_ = true; } ld_jobcv_.notify_one();
      ld_writer_.join();
    }
    for (void * b : dbufs_) ::free(b);
  }

  bool had_error() const { return had_error_.load(std::memory_order_relaxed); }
  uint64_t validated_files() const { return vfiles_; }
  uint64_t validated_bytes() const { return vbytes_; }

  void run(int read_fd) {
    if (!read_only()) start_pool();
    StreamReader r(read_fd);
    parse(r);               // validates every header checksum; flags truncation
    r.drain();              // consume trailing padding so the decompressor can finish
    if (!read_only()) { stop_pool(); finish_deferred(); log_large_file_stats(); }
  }

  // Same as run() but reads decompressed frames straight from an in-memory
  // FrameSink (the `-t`/`-l --tar` fast path) instead of a pipe fd — skip() over
  // file data becomes pointer arithmetic, so the walk does O(files) work.
  void run_sink(FrameSink * src) {
    if (!read_only()) start_pool();
    StreamReader r(src);
    parse(r);
    r.drain();              // pop any trailing frames so the producer can finish
    if (!read_only()) { stop_pool(); finish_deferred(); log_large_file_stats(); }
  }

private:
  // -vv breakdown of the single-parse-thread large-file write path.  bufwait
  // dominant → WRITE-bound (writer can't keep up); fill dominant → the parse
  // thread's sink read + memcpy is the spine; scan dominant → zero-detect; the
  // writer's jobwait dominant → the parse thread can't feed it fast enough.
  void log_large_file_stats() {
    if (opt_.verbosity < V_DEBUG) return;
    uint64_t files = ld_files_.load(), bytes = ld_bytes_.load();
    if (files == 0) return;   // no large files in this archive
    auto ms = [](uint64_t ns) { return ns / 1000000.0; };
    char sz[48]; human_bytes(double(bytes), sz, sizeof(sz));
    char buf[320];
    std::snprintf(buf, sizeof(buf),
      "[UNTAR-LARGE] %llu large file(s) >4 MiB, %s via O_DIRECT (small files + dirs go to "
      "the writer pool, not counted here) | parse: fill=%.0fms scan=%.0fms "
      "bufwait=%.0fms (write-bound) | writer: write=%.0fms jobwait=%.0fms (parse-bound)\n",
      (unsigned long long)files, sz,
      ms(ld_fill_ns_.load()), ms(ld_scan_ns_.load()), ms(ld_bufwait_ns_.load()),
      ms(ld_write_ns_.load()), ms(ld_jobwait_ns_.load()));
    vlog(V_DEBUG, opt_, buf);
  }

  static constexpr uint64_t SMALL_FILE_MAX = 4 * 1024 * 1024;  // buffer+dispatch below this; stream above
  int dest_fd_; const Options & opt_; Meter * m_; bool as_root_;
  bool validate_only_ = false;
  bool list_only_ = false;
  bool read_only() const { return validate_only_ || list_only_; }  // no writer pool
  uint64_t vfiles_ = 0, vbytes_ = 0;  // validate/list counters (entries, logical bytes)
  std::atomic<bool> had_error_{false};

  // Format one entry as a `tar -tvf` line on stdout (for -l --tar).  perms,
  // owner/group (names if present in the header, else numeric), logical size,
  // mtime (local), name (+ link target).  Fixed fields ~fit 80 cols; the name
  // runs after, like every tar tool.
  void print_list_entry(const InEntry & e) {
    char perms[11];
    char t = '-';
    switch (e.typeflag) {
      case '5': t = 'd'; break; case '2': t = 'l'; break; case '1': t = 'h'; break;
      case '3': t = 'c'; break; case '4': t = 'b'; break; case '6': t = 'p'; break;
    }
    perms[0] = t;
    static const char rwx[9] = {'r','w','x','r','w','x','r','w','x'};
    for (int i = 0; i < 9; ++i) perms[1 + i] = (e.mode & (1u << (8 - i))) ? rwx[i] : '-';
    if (e.mode & 04000) perms[3] = (perms[3] == 'x') ? 's' : 'S';
    if (e.mode & 02000) perms[6] = (perms[6] == 'x') ? 's' : 'S';
    if (e.mode & 01000) perms[9] = (perms[9] == 'x') ? 't' : 'T';
    perms[10] = '\0';

    std::string owner = e.uname.empty() ? std::to_string(e.uid) : e.uname;
    std::string group = e.gname.empty() ? std::to_string(e.gid) : e.gname;
    std::string owngrp = owner + "/" + group;

    char date[24] = "";
    time_t tt = (time_t)e.mtime; struct tm tmv;
    if (localtime_r(&tt, &tmv)) std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", &tmv);

    uint64_t sz = e.is_sparse ? e.real_size : e.size;
    std::string nm = e.name;
    if (e.typeflag == '2' && !e.linkname.empty()) nm += " -> " + e.linkname;       // symlink
    else if (e.typeflag == '1' && !e.linkname.empty()) nm += " link to " + e.linkname;  // hardlink

    std::fprintf(stdout, "%s %-17s %12llu %s %s\n",
                 perms, owngrp.c_str(), (unsigned long long)sz, date, nm.c_str());
  }
  // Large-file (stream_large_direct) breakdown, -v only.  All on the single
  // parse thread except ld_write_ns_ / ld_jobwait_ns_ (the per-file writer):
  //   fill   = read member data from the sink into the aligned buffer (memcpy)
  //   scan   = zero-detect + build the O_DIRECT write plan
  //   bufwait= parse thread blocked waiting for the writer to free a buffer  → WRITE-bound
  //   write  = writer thread in pwrite
  //   jobwait= writer thread blocked waiting for a filled buffer            → PARSE-bound
  std::atomic<uint64_t> ld_fill_ns_{0}, ld_scan_ns_{0}, ld_bufwait_ns_{0};
  std::atomic<uint64_t> ld_write_ns_{0}, ld_jobwait_ns_{0};
  std::atomic<uint64_t> ld_bytes_{0}, ld_files_{0};
  // O_DIRECT large-file writer: a pool of reused 4 KiB-aligned buffers.  The
  // parse thread fills one (sink read + zero-detect) while a dedicated writer
  // thread O_DIRECTs another — pipelining read and write on a single stream
  // (concurrent O_DIRECT streams contend, so we keep exactly one writer).
  // Buffers: 16 MiB transfers (larger O_DIRECT writes sit closer to the device's
  // single-stream ceiling than 8 MiB) and a deeper queue (6, was 3) so the
  // writer doesn't starve when the parse thread briefly stalls.  96 MiB peak.
  static constexpr size_t DBUF_CAP = 16 * 1024 * 1024;
  static constexpr int    NDBUF    = 6;
  void * dbufs_[NDBUF] = {};

  // The writer is PERSISTENT across all large files (not spawned/joined per
  // file): at a file boundary the parse thread enqueues a FINALIZE job and moves
  // straight to the next entry, so the prior file's tail writes overlap the next
  // file's parse/read instead of stalling the parse thread on a per-file join.
  // The parse thread is the sole sink reader; the writer only touches dbufs_/fds.
  struct DJob {
    enum Kind { WRITE, FINALIZE } kind = WRITE;
    int fd = -1, bi = -1;
    std::vector<std::tuple<size_t, uint64_t, size_t>> runs;  // (buf_off, file_off, len)
    size_t tail_off = 0, tail_len = 0; uint64_t tail_fpos = 0;
    uint64_t final_size = 0;                                  // FINALIZE: ftruncate target
    uint32_t mode = 0; int64_t mtime = 0; uint64_t uid = 0, gid = 0; ExtMeta ext;
    std::string rel;
  };
  std::mutex ld_mx_; std::condition_variable ld_jobcv_, ld_freecv_;
  std::deque<DJob> ld_jobs_; std::deque<int> ld_freeq_;
  bool ld_done_ = false, ld_started_ = false, ld_nobuf_ = false;
  std::atomic<bool> ld_werr_{false};
  std::thread ld_writer_;

  // Deferred work applied after all data is written.
  struct Hard { std::string name, target; };
  struct DirMeta { std::string path; uint32_t mode; int64_t mtime; uint64_t uid, gid; ExtMeta ext; };
  std::vector<Hard> hardlinks_;
  std::vector<DirMeta> dirmeta_;

  void fail(const std::string & path, const std::string & why) {
    had_error_.store(true, std::memory_order_relaxed);
    vlog(V_ERROR, opt_, "gzstd: untar: " + path + ": " + why + "\n");
  }

  // ---- secure path resolution (O_NOFOLLOW on every component) ----
  // Returns the fd of rel's parent directory (caller closes) and sets `leaf`.
  // Refuses any '..' component; refuses to traverse a symlink.  create: mkdir
  // missing intermediates.
  int open_parent(const std::string & rel, std::string & leaf, bool create) {
    std::vector<std::string> comps;
    size_t i = 0;
    while (i < rel.size()) {
      size_t j = rel.find('/', i);
      std::string c = rel.substr(i, j == std::string::npos ? std::string::npos : j - i);
      if (!c.empty() && c != ".") {
        if (c == "..") return -1;                 // path traversal: refuse
        comps.push_back(std::move(c));
      }
      if (j == std::string::npos) break; else i = j + 1;
    }
    if (comps.empty()) return -1;
    leaf = comps.back();
    int cur = ::dup(dest_fd_);
    if (cur < 0) return -1;
    for (size_t k = 0; k + 1 < comps.size(); ++k) {
      int nx = ::openat(cur, comps[k].c_str(), O_DIRECTORY | O_NOFOLLOW | O_RDONLY | O_CLOEXEC);
      if (nx < 0 && create && errno == ENOENT) {
        if (::mkdirat(cur, comps[k].c_str(), 0755) != 0 && errno != EEXIST) { ::close(cur); return -1; }
        nx = ::openat(cur, comps[k].c_str(), O_DIRECTORY | O_NOFOLLOW | O_RDONLY | O_CLOEXEC);
      }
      ::close(cur);
      if (nx < 0) return -1;                       // ELOOP (symlink), ENOTDIR, … → refuse
      cur = nx;
    }
    return cur;
  }

  void set_meta_fd(int fd, uint32_t mode, int64_t mtime, uint64_t uid, uint64_t gid) {
    (void)::fchmod(fd, mode & 07777);
    struct timespec ts[2]; ts[0].tv_sec = (time_t)mtime; ts[0].tv_nsec = 0; ts[1] = ts[0];
    (void)::futimens(fd, ts);
    if (as_root_ && ::fchown(fd, (uid_t)uid, (gid_t)gid) != 0) { /* best-effort */ }
  }

  // Apply --xattrs/--acls metadata via a securely-opened fd (regular file or
  // directory).  Must run AFTER set_meta_fd: the access ACL overrides the mode's
  // group bits, so the ACL has to win.  Default ACLs (directories only) need a
  // path, so we go through /proc/self/fd to keep the secure-by-fd guarantee.
  // Failures (EPERM on system.*/security.*, filesystem without ACL support) are
  // best-effort and reported only at -v, like the device-node path.
  void apply_ext(int fd, bool is_dir, const ExtMeta & ext) {
    if (ext.empty()) return;
    for (const auto & kv : ext.xattrs) {
      if (::fsetxattr(fd, kv.first.c_str(), kv.second.data(), kv.second.size(), 0) != 0)
        vlog(V_VERBOSE, opt_, "gzstd: untar: xattr " + kv.first + ": " + std::strerror(errno) + "\n");
    }
#ifdef GZSTD_HAVE_ACL
    if (!ext.acl_access.empty()) {
      acl_t a = ::acl_from_text(ext.acl_access.c_str());
      if (a) {
        if (::acl_set_fd(fd, a) != 0)
          vlog(V_VERBOSE, opt_, std::string("gzstd: untar: acl(access): ") + std::strerror(errno) + "\n");
        ::acl_free(a);
      }
    }
    if (is_dir && !ext.acl_default.empty()) {
      acl_t a = ::acl_from_text(ext.acl_default.c_str());
      if (a) {
        std::string pp = "/proc/self/fd/" + std::to_string(fd);  // fd opened O_NOFOLLOW per component
        if (::acl_set_file(pp.c_str(), ACL_TYPE_DEFAULT, a) != 0)
          vlog(V_VERBOSE, opt_, std::string("gzstd: untar: acl(default): ") + std::strerror(errno) + "\n");
        ::acl_free(a);
      }
    }
#endif
  }

  // Securely create/replace a regular file; returns an open write fd or -1.
  // o_direct: add O_DIRECT to the leaf open (large-file extract fast path).  The
  // secure O_NOFOLLOW-per-component walk is unchanged; only the final flag differs.
  int create_file(const std::string & rel, uint32_t mode, bool o_direct = false) {
    std::string leaf; int pfd = open_parent(rel, leaf, true);
    if (pfd < 0) return -1;
    ::unlinkat(pfd, leaf.c_str(), 0);              // remove any existing file/symlink first
    int flags = O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC;
#ifdef O_DIRECT
    if (o_direct) flags |= O_DIRECT;
#endif
    int fd = ::openat(pfd, leaf.c_str(), flags, mode & 07777);
    ::close(pfd);
    return fd;
  }

  bool make_dir(const std::string & rel, uint32_t mode) {
    std::string leaf; int pfd = open_parent(rel, leaf, true);
    if (pfd < 0) return false;
    bool ok = true;
    if (::mkdirat(pfd, leaf.c_str(), mode & 07777) != 0) {
      if (errno == EEXIST) {
        struct stat st;
        if (::fstatat(pfd, leaf.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0 && !S_ISDIR(st.st_mode)) {
          ::unlinkat(pfd, leaf.c_str(), 0);        // a planted symlink/file where a dir belongs
          ok = (::mkdirat(pfd, leaf.c_str(), mode & 07777) == 0);
        }
      } else ok = false;
    }
    ::close(pfd);
    return ok;
  }

  bool make_symlink(const std::string & rel, const std::string & target,
                    uint64_t uid, uint64_t gid) {
    std::string leaf; int pfd = open_parent(rel, leaf, true);
    if (pfd < 0) return false;
    ::unlinkat(pfd, leaf.c_str(), 0);
    bool ok = (::symlinkat(target.c_str(), pfd, leaf.c_str()) == 0);
    if (ok && as_root_ && ::fchownat(pfd, leaf.c_str(), (uid_t)uid, (gid_t)gid, AT_SYMLINK_NOFOLLOW) != 0) { /* best-effort */ }
    ::close(pfd);
    return ok;
  }

  bool make_special(const std::string & rel, uint32_t mode, char type,
                    uint32_t major_, uint32_t minor_) {
    std::string leaf; int pfd = open_parent(rel, leaf, true);
    if (pfd < 0) return false;
    ::unlinkat(pfd, leaf.c_str(), 0);
    mode_t mt = (mode & 07777) | (type == '6' ? S_IFIFO : type == '3' ? S_IFCHR : S_IFBLK);
    dev_t dev = (type == '6') ? 0 : makedev(major_, minor_);
    bool ok = (::mknodat(pfd, leaf.c_str(), mt, dev) == 0);
    ::close(pfd);
    return ok;
  }

  // ---- writer pool (regular-file data writes, in parallel with decompression) ----
  struct Job { std::string rel; uint32_t mode; int64_t mtime; uint64_t uid, gid; std::vector<char> data; ExtMeta ext; };
  std::mutex q_m_; std::condition_variable q_cv_prod_, q_cv_cons_;
  std::deque<Job> q_; size_t q_bytes_ = 0; bool q_done_ = false;
  static constexpr size_t Q_MAX_BYTES = 256 * 1024 * 1024;
  std::vector<std::thread> writers_;

  void start_pool() {
    // --write-threads overrides; auto = min(worker threads, 16) — beyond ~16 the
    // gain plateaus on filesystem metadata/journal contention (tune per box).
    int n = opt_.write_threads > 0 ? (int)opt_.write_threads
                                   : std::min(resolve_cpu_threads(opt_.cpu_threads), 16);
    n = std::max(1, n);
    if (opt_.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt_, "[UNTAR] " + std::to_string(n) + " writer thread(s)\n");
    for (int i = 0; i < n; ++i) writers_.emplace_back([this] { writer_loop(); });
  }
  void stop_pool() {
    { std::lock_guard<std::mutex> lk(q_m_); q_done_ = true; } q_cv_cons_.notify_all();
    for (auto & t : writers_) t.join();
    writers_.clear();
    ld_stop_writer();   // drain + join the persistent large-file writer too
  }
  void enqueue(Job && j) {
    std::unique_lock<std::mutex> lk(q_m_);
    q_cv_prod_.wait(lk, [&] { return q_bytes_ < Q_MAX_BYTES || q_.empty(); });
    q_bytes_ += j.data.size();
    q_.push_back(std::move(j));
    lk.unlock(); q_cv_cons_.notify_one();
  }
  void writer_loop() {
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(q_m_);
        q_cv_cons_.wait(lk, [&] { return !q_.empty() || q_done_; });
        if (q_.empty()) return;
        j = std::move(q_.front()); q_.pop_front();
        q_bytes_ -= j.data.size();
        lk.unlock(); q_cv_prod_.notify_all();
      }
      write_small(j);
    }
  }
  // --sparse extract: write [data,data+n) at file offset `at`, leaving holes for
  // all-zero SPARSE_BLK runs (skip the pwrite).  The caller ftruncate()s to the
  // final size afterward so trailing holes and the exact length are preserved.
  static constexpr size_t SPARSE_BLK = 4096;
  static bool pwrite_sparse(int fd, off_t at, const char * data, size_t n) {
    size_t off = 0;
    while (off < n) {
      size_t chunk = std::min(SPARSE_BLK, n - off);
      bool zero = true;
      for (size_t k = 0; k < chunk; ++k) if (data[off + k]) { zero = false; break; }
      if (!zero) {
        size_t w = 0;
        while (w < chunk) {
          ssize_t r = ::pwrite(fd, data + off + w, chunk - w, at + (off_t)(off + w));
          if (r < 0) { if (errno == EINTR) continue; return false; }
          w += (size_t)r;
        }
      }
      off += chunk;
    }
    return true;
  }

  void write_small(const Job & j) {
    int fd = create_file(j.rel, j.mode);
    if (fd < 0) { fail(j.rel, std::strerror(errno)); return; }
    bool ok = true;
    if (opt_.sparse_mode != 0) {  // --sparse / auto (default on for file output); --no-sparse disables
      ok = pwrite_sparse(fd, 0, j.data.data(), j.data.size());
      if (ok && ::ftruncate(fd, (off_t)j.data.size()) != 0) ok = false;
    } else {
      size_t off = 0;
      while (off < j.data.size()) {
        ssize_t w = ::write(fd, j.data.data() + off, j.data.size() - off);
        if (w < 0) { if (errno == EINTR) continue; ok = false; break; }
        off += (size_t)w;
      }
    }
    if (!ok) fail(j.rel, std::strerror(errno));
    set_meta_fd(fd, j.mode, j.mtime, j.uid, j.gid);
    apply_ext(fd, /*is_dir=*/false, j.ext);
    ::close(fd);
    if (m_) m_->wrote_bytes.fetch_add(j.data.size(), std::memory_order_relaxed);
  }

  // ---- parsing ----
  // Normalize a member name to a safe relative path (strip leading '/' and './';
  // empty result means "skip").  '..' is rejected later in open_parent().
  static std::string norm(const std::string & raw) {
    std::string s = raw;
    size_t i = 0; while (i < s.size() && s[i] == '/') ++i; s = s.substr(i);
    while (s.rfind("./", 0) == 0) s = s.substr(2);
    return s;
  }

  void parse(StreamReader & r) {
    char blk[TAR_BLK];
    std::string pend_name, pend_link;     // GNU 'L'/'K' overrides for the next header
    std::string pax_path, pax_link; uint64_t pax_size = 0; int64_t pax_mtime = 0;
    bool pax_has_size = false, pax_has_mtime = false;
    bool pax_gnu_sparse = false;          // GNU.sparse.* records seen (PAX sparse format)
    int  pax_sp_major = 0, pax_sp_minor = 0;
    std::string pax_sp_name;
    uint64_t pax_sp_realsize = 0; bool pax_has_sp_realsize = false;
    std::vector<std::pair<uint64_t, uint64_t>> pax_sp_map;  // 0.0/0.1 segment map
    uint64_t pax_sp_pending_off = 0; bool pax_sp_have_off = false;  // 0.0 pairing
    ExtMeta pax_ext;                      // SCHILY.xattr.* / SCHILY.acl.* for next header
    int zero_run = 0;
    while (true) {
      if (!r.read_exact(blk, TAR_BLK)) {
        if (zero_run == 0) fail("(archive)", "truncated tar stream");
        return;
      }
      bool all_zero = true;
      for (size_t k = 0; k < TAR_BLK; ++k) if (blk[k]) { all_zero = false; break; }
      if (all_zero) { if (++zero_run >= 2) return; else continue; }
      zero_run = 0;

      // Verify the header checksum (also guards against reading garbage).
      uint64_t stored = get_num(blk + 148, 8);
      uint64_t sum = 0; for (size_t k = 0; k < TAR_BLK; ++k)
        sum += (k >= 148 && k < 156) ? (unsigned char)' ' : (unsigned char)blk[k];
      if (sum != stored) { fail("(archive)", "bad tar header checksum"); return; }

      char type = blk[156];
      uint64_t size = get_num(blk + 124, 12);

      // GNU long name / long link: data is the string for the NEXT header.
      if (type == 'L' || type == 'K') {
        std::vector<char> nm(size + 1, 0);
        if (!r.read_exact(nm.data(), size)) { fail("(archive)", "truncated long name"); return; }
        r.skip((TAR_BLK - (size % TAR_BLK)) % TAR_BLK);
        std::string s(nm.data(), size ? size - (nm[size - 1] == 0 ? 1 : 0) : 0);
        if (type == 'L') pend_name = s; else pend_link = s;
        continue;
      }
      // pax extended headers: "len key=value\n" records.
      if (type == 'x' || type == 'g') {
        std::vector<char> pd(size);
        if (!r.read_exact(pd.data(), size)) { fail("(archive)", "truncated pax header"); return; }
        r.skip((TAR_BLK - (size % TAR_BLK)) % TAR_BLK);
        size_t p = 0; std::string rec(pd.data(), pd.size());
        while (p < rec.size()) {
          size_t sp = rec.find(' ', p); if (sp == std::string::npos) break;
          int rlen = atoi(rec.substr(p, sp - p).c_str()); if (rlen <= 0) break;
          std::string kv = rec.substr(sp + 1, (size_t)rlen - (sp - p) - 2);  // drop trailing '\n'
          size_t eq = kv.find('=');
          if (eq != std::string::npos) {
            std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
            if (key == "path") pax_path = val;
            else if (key == "linkpath") pax_link = val;
            else if (key == "size") { pax_size = strtoull(val.c_str(), nullptr, 10); pax_has_size = true; }
            else if (key == "mtime") { pax_mtime = (int64_t)strtoll(val.c_str(), nullptr, 10); pax_has_mtime = true; }
            // --xattrs/--acls: collect SCHILY.* records only when the matching flag
            // is set, so without the flag they are parsed-and-ignored (GNU tar's behavior).
            else if (opt_.tar_xattrs && key.rfind("SCHILY.xattr.", 0) == 0)
              pax_ext.xattrs.emplace_back(key.substr(13), val);
            else if (opt_.tar_acls && key == "SCHILY.acl.access")  pax_ext.acl_access  = val;
            else if (opt_.tar_acls && key == "SCHILY.acl.default") pax_ext.acl_default = val;
            // PAX GNU sparse (0.0/0.1/1.0).  Capture the version, real size, real
            // name, and the segment map (however this version encodes it).
            else if (key.rfind("GNU.sparse.", 0) == 0) {
              pax_gnu_sparse = true;
              if      (key == "GNU.sparse.major") pax_sp_major = atoi(val.c_str());
              else if (key == "GNU.sparse.minor") pax_sp_minor = atoi(val.c_str());
              else if (key == "GNU.sparse.name")  pax_sp_name = val;
              else if (key == "GNU.sparse.realsize" || key == "GNU.sparse.size") {
                pax_sp_realsize = strtoull(val.c_str(), nullptr, 10); pax_has_sp_realsize = true;
              }
              else if (key == "GNU.sparse.map") {           // 0.1: "off,len,off,len,..."
                pax_sp_map.clear(); const char * s = val.c_str(); char * end = nullptr;
                while (*s) {
                  uint64_t o = strtoull(s, &end, 10); if (end == s || *end != ',') break; s = end + 1;
                  uint64_t l = strtoull(s, &end, 10); pax_sp_map.emplace_back(o, l);
                  if (*end == ',') s = end + 1; else s = end;
                }
              }
              else if (key == "GNU.sparse.offset") {        // 0.0: paired with numbytes
                pax_sp_pending_off = strtoull(val.c_str(), nullptr, 10); pax_sp_have_off = true;
              }
              else if (key == "GNU.sparse.numbytes" && pax_sp_have_off) {
                pax_sp_map.emplace_back(pax_sp_pending_off, strtoull(val.c_str(), nullptr, 10));
                pax_sp_have_off = false;
              }
            }
          }
          p += (size_t)rlen;
        }
        continue;
      }

      InEntry e;
      std::string name = get_str(blk, 100);
      std::string prefix = get_str(blk + 345, 155);
      if (!prefix.empty()) name = prefix + "/" + name;
      if (!pend_name.empty()) { name = pend_name; pend_name.clear(); }
      if (!pax_path.empty())  { name = pax_path;  pax_path.clear(); }
      e.name = name;
      e.linkname = get_str(blk + 157, 100);
      if (!pend_link.empty()) { e.linkname = pend_link; pend_link.clear(); }
      if (!pax_link.empty())  { e.linkname = pax_link;  pax_link.clear(); }
      e.mode = (uint32_t)get_num(blk + 100, 8);
      e.uid = get_num(blk + 108, 8);
      e.gid = get_num(blk + 116, 8);
      e.uname = get_str(blk + 265, 32);   // ustar owner/group names (for -l --tar)
      e.gname = get_str(blk + 297, 32);
      e.size = pax_has_size ? pax_size : size;
      e.mtime = pax_has_mtime ? pax_mtime : (int64_t)get_num(blk + 136, 12);
      e.typeflag = type;
      e.devmajor = (uint32_t)get_num(blk + 329, 8);
      e.devminor = (uint32_t)get_num(blk + 337, 8);
      e.ext = std::move(pax_ext); pax_ext = ExtMeta{};
      pax_has_size = pax_has_mtime = false;

      // GNU OLDGNU sparse ('S'): the sparse map is in the header (4 entries at
      // offset 386) + optional extension blocks; realsize at 483.  e.size is the
      // stored (compacted) data length; e.real_size is the logical size.
      if (type == 'S') {
        auto add_seg = [&](const char * p) {
          uint64_t off = get_num(p, 12), len = get_num(p + 12, 12);
          if (!(off == 0 && len == 0)) e.sparse_map.emplace_back(off, len);
        };
        for (int i = 0; i < 4; ++i) add_seg(blk + 386 + i * 24);
        e.real_size = get_num(blk + 483, 12);
        bool ext = (unsigned char)blk[482] != 0;
        char xb[TAR_BLK]; bool ok_ext = true;
        while (ext) {
          if (!r.read_exact(xb, TAR_BLK)) { fail(e.name, "truncated sparse extension"); ok_ext = false; break; }
          for (int i = 0; i < 21; ++i) add_seg(xb + i * 24);
          ext = (unsigned char)xb[504] != 0;
        }
        if (!ok_ext) return;
        e.is_sparse = true;
      } else if (pax_gnu_sparse) {
        // PAX GNU sparse.  The real name lives in GNU.sparse.name (the header
        // name is a placeholder); real size in GNU.sparse.realsize/size.  e.size
        // (the pax 'size' record) is the stored byte count to consume.
        if (!pax_sp_name.empty()) e.name = pax_sp_name;
        if (pax_has_sp_realsize)  e.real_size = pax_sp_realsize;
        if (pax_sp_major >= 1) {
          // 1.0: the segment map is a text block at the start of the file data.
          e.is_sparse = true; e.sparse_map_in_data = true;
        } else if (!pax_sp_map.empty()) {
          // 0.0 / 0.1: the map is fully described by the PAX records.
          e.is_sparse = true; e.sparse_map = pax_sp_map;
        } else {
          fail(e.name, "unrecognized PAX GNU sparse variant — skipped");
          uint64_t pad = (TAR_BLK - (e.size % TAR_BLK)) % TAR_BLK;
          if (e.size) { r.skip(e.size); r.skip(pad); }
          pax_gnu_sparse = false;
          continue;
        }
      }
      pax_gnu_sparse = false; pax_sp_major = pax_sp_minor = 0; pax_sp_name.clear();
      pax_sp_realsize = 0; pax_has_sp_realsize = false; pax_sp_map.clear();
      pax_sp_have_off = false;

      handle_entry(r, e);
    }
  }

  // True if any path component is "..", i.e. a parent-traversal escape attempt.
  static bool has_dotdot(const std::string & rel) {
    size_t i = 0;
    while (i < rel.size()) {
      size_t j = rel.find('/', i);
      std::string c = rel.substr(i, j == std::string::npos ? std::string::npos : j - i);
      if (c == "..") return true;
      if (j == std::string::npos) break; else i = j + 1;
    }
    return false;
  }

  void handle_entry(StreamReader & r, const InEntry & e) {
    std::string rel = norm(e.name);
    // data byte count present in the stream (for regular files); always skip it
    // after handling so the parser stays aligned even when we reject the member.
    uint64_t pad = (TAR_BLK - (e.size % TAR_BLK)) % TAR_BLK;

    // Read-only walks (`-t --tar` validate, `-l --tar` list): the header
    // checksum was already verified in parse(); count the member, list it if
    // requested, and consume its data so the parser stays aligned.  A short read
    // here (truncated archive) is caught by parse().
    if (read_only()) {
      if (list_only_) print_list_entry(e);
      ++vfiles_; vbytes_ += e.is_sparse ? e.real_size : e.size;
      if (e.size) { r.skip(e.size); r.skip(pad); }
      return;
    }

    // GNU sparse: place each stored segment at its logical offset, leaving holes.
    if (e.is_sparse) { extract_sparse(r, rel, e, pad); return; }

    if (rel.empty()) { if (e.size) { r.skip(e.size); r.skip(pad); } return; }
    // Path-traversal guard (the openat O_NOFOLLOW walk is the backstop; this gives
    // a clear message and refuses before touching the filesystem).
    if (has_dotdot(rel)) {
      fail(rel, "unsafe path (parent-directory traversal) — skipped");
      if (e.size) { r.skip(e.size); r.skip(pad); }
      return;
    }

    switch (e.typeflag) {
      case '0': case '\0': case '7': {            // regular file
        // --keep-going: record this file's data range in the tar stream (the
        // header is fully consumed, so consumed_total is the data start) so the
        // post-extraction report can name files that overlap a damaged frame.
        if (opt_.keep_going && e.size)
          g_member_ranges.add(rel, r.consumed_total, e.size);
        if (e.size <= SMALL_FILE_MAX) {
          Job j; j.rel = rel; j.mode = e.mode; j.mtime = e.mtime; j.uid = e.uid; j.gid = e.gid;
          j.ext = e.ext;
          j.data.resize(e.size);
          if (e.size && !r.read_exact(j.data.data(), e.size)) { fail(rel, "truncated file data"); return; }
          r.skip(pad);
          enqueue(std::move(j));
        } else {
          if (opt_.direct_io) stream_large_direct(r, rel, e);  // O_DIRECT (Gen4+ default)
          else                stream_large(r, rel, e);
          r.skip(pad);
        }
        break;
      }
      case '5':                                    // directory
        if (!make_dir(rel, e.mode)) fail(rel, "cannot create directory");
        else dirmeta_.push_back({rel, e.mode, e.mtime, e.uid, e.gid, e.ext});
        break;
      case '2':                                    // symlink
        if (!make_symlink(rel, e.linkname, e.uid, e.gid)) fail(rel, "cannot create symlink");
        break;
      case '1':                                    // hardlink (deferred: target must exist)
        hardlinks_.push_back({rel, norm(e.linkname)});
        break;
      case '3': case '4': case '6':                // char/block device, fifo
        if (!make_special(rel, e.mode, e.typeflag, e.devmajor, e.devminor)) {
          if (errno == EPERM) vlog(V_VERBOSE, opt_, "gzstd: untar: " + rel + ": need privilege for device/special; skipped\n");
          else fail(rel, "cannot create special file");
        }
        if (e.size) { r.skip(e.size); r.skip(pad); }
        break;
      case 'V':                                     // volume label/header: no file data
        if (e.size) { r.skip(e.size); r.skip(pad); }
        break;
      default:
        // Unknown/unsupported entry type (e.g. 'D' GNU incremental dumpdir,
        // 'M' multi-volume continuation).  Fail LOUDLY — do not silently drop
        // data; the user needs to know this archive isn't fully extracted.
        fail(rel, std::string("unsupported tar entry type '") + e.typeflag
                  + "' — skipped (data NOT extracted; use GNU tar)");
        if (e.size) { r.skip(e.size); r.skip(pad); }
        break;
    }
  }

  // PAX GNU.sparse.1.0: the segment map is a text block ("numblocks\noff\nlen\n…")
  // at the start of the file data, NUL-padded to a 512 multiple.  Read whole
  // 512-blocks (so map bytes consumed is always 512-aligned), parsing decimals.
  bool read_pax_sparse_map(StreamReader & r, std::vector<std::pair<uint64_t,uint64_t>> & map,
                           uint64_t & consumed) {
    map.clear(); consumed = 0;
    char blk[TAR_BLK]; std::string num;
    long long numblocks = -1; size_t need = 0, got = 0;
    uint64_t pend = 0; bool have = false, done = false;
    while (!done) {
      if (!r.read_exact(blk, TAR_BLK)) return false;
      consumed += TAR_BLK;
      for (size_t i = 0; i < TAR_BLK && !done; ++i) {
        char c = blk[i];
        if (c >= '0' && c <= '9') { num.push_back(c); continue; }
        if (c != '\n' || num.empty()) continue;
        uint64_t v = strtoull(num.c_str(), nullptr, 10); num.clear();
        if (numblocks < 0) { numblocks = (long long)v; need = 1 + 2 * (size_t)numblocks; got = 1;
                             if (numblocks == 0) done = true; }
        else { ++got;
               if (!have) { pend = v; have = true; }
               else { map.emplace_back(pend, v); have = false; }
               if (got >= need) done = true; }
      }
    }
    return numblocks >= 0;
  }

  // Restore a GNU sparse file: write each stored segment at its logical offset
  // (leaving holes between them), then ftruncate to the real size.  Consumes
  // exactly e.size bytes of stored data + padding from the stream.
  void extract_sparse(StreamReader & r, const std::string & rel, const InEntry & e, uint64_t pad) {
    int fd = create_file(rel, e.mode);
    if (fd < 0) { fail(rel, std::strerror(errno)); if (e.size) { r.skip(e.size); r.skip(pad); } return; }
    char buf[1 << 20]; bool ok = true; uint64_t consumed = 0;
    std::vector<std::pair<uint64_t,uint64_t>> data_map;
    const std::vector<std::pair<uint64_t,uint64_t>> * mapp = &e.sparse_map;
    if (e.sparse_map_in_data) {  // PAX 1.0: map prefixes the data
      uint64_t mc = 0;
      if (!read_pax_sparse_map(r, data_map, mc)) {
        fail(rel, "bad PAX sparse map"); ::close(fd); return;
      }
      consumed = mc; mapp = &data_map;
    }
    for (const auto & seg : *mapp) {
      uint64_t base = seg.first, len = seg.second, w = 0;
      while (w < len && ok) {
        size_t want = (size_t)std::min<uint64_t>(len - w, sizeof buf);
        size_t got = r.read_some(buf, want);
        if (!got) { ok = false; break; }
        size_t p = 0;
        while (p < got) {
          ssize_t n = ::pwrite(fd, buf + p, got - p, (off_t)(base + w + p));
          if (n < 0) { if (errno == EINTR) continue; ok = false; break; }
          p += (size_t)n;
        }
        w += got; consumed += got;
      }
      if (!ok) break;
    }
    if (consumed < e.size) r.skip(e.size - consumed);  // any trailing stored bytes
    r.skip(pad);
    if (ok && ::ftruncate(fd, (off_t)e.real_size) != 0) ok = false;
    if (!ok) fail(rel, "sparse write failed");
    set_meta_fd(fd, e.mode, e.mtime, e.uid, e.gid);
    apply_ext(fd, /*is_dir=*/false, e.ext);
    ::close(fd);
    if (m_) m_->wrote_bytes.fetch_add(e.real_size, std::memory_order_relaxed);
  }

  void stream_large(StreamReader & r, const std::string & rel, const InEntry & e) {
    int fd = create_file(rel, e.mode);
    if (fd < 0) { fail(rel, std::strerror(errno)); r.skip(e.size); return; }
    uint64_t left = e.size; char tmp[1 << 20]; bool ok = true;
    uint64_t fpos = 0;  // running file offset (for --sparse pwrite)
    while (left) {
      size_t want = (size_t)std::min<uint64_t>(left, sizeof tmp);
      size_t got = r.read_some(tmp, want);
      if (!got) { ok = false; break; }
      if (opt_.sparse_mode != 0) {
        if (!pwrite_sparse(fd, (off_t)fpos, tmp, got)) { ok = false; break; }
      } else {
        size_t off = 0;
        while (off < got) { ssize_t w = ::write(fd, tmp + off, got - off); if (w < 0) { if (errno == EINTR) continue; ok = false; break; } off += (size_t)w; }
      }
      if (!ok) break;
      fpos += got;
      left -= got;
      if (m_) m_->wrote_bytes.fetch_add(got, std::memory_order_relaxed);
    }
    if (!ok) fail(rel, "write failed");
    // --sparse skips trailing/interior zero blocks, so set the exact final size
    // (materializes trailing holes and the correct length).
    else if (opt_.sparse_mode != 0 && ::ftruncate(fd, (off_t)e.size) != 0) fail(rel, "ftruncate failed");
    set_meta_fd(fd, e.mode, e.mtime, e.uid, e.gid);
    apply_ext(fd, /*is_dir=*/false, e.ext);
    ::close(fd);
  }

  // Lazily allocate the aligned buffer pool and spawn the persistent writer on
  // the first large file.  Returns false if no aligned buffer could be had (the
  // caller then falls back to the buffered stream_large path for that file).
  bool ld_ensure_writer() {
    if (ld_started_) return !ld_nobuf_;
    ld_started_ = true;
    static constexpr size_t AL = 4096;
    for (int i = 0; i < NDBUF; ++i)
      if (!dbufs_[i] && posix_memalign(&dbufs_[i], AL, DBUF_CAP) != 0) dbufs_[i] = nullptr;
    int nbuf = 0; while (nbuf < NDBUF && dbufs_[nbuf]) ++nbuf;  // leading run we got
    if (nbuf == 0) { ld_nobuf_ = true; return false; }
    for (int i = 0; i < nbuf; ++i) ld_freeq_.push_back(i);
    ld_writer_ = std::thread([this] { ld_writer_loop(); });
    return true;
  }

  // Persistent writer thread: drains DJobs FIFO.  WRITE → O_DIRECT the aligned
  // runs of dbufs_[bi] to its fd (+ the unaligned tail with O_DIRECT cleared),
  // return the buffer.  FINALIZE → ftruncate to the exact size, set metadata,
  // close the fd.  FIFO ordering guarantees a file's WRITE jobs all complete
  // before its FINALIZE, and that earlier files finish before later ones.
  void ld_writer_loop() {
    auto pwrite_at = [](int fd, const char * p, size_t len, uint64_t at) {
      size_t d = 0;
      while (d < len) {
        ssize_t w = ::pwrite(fd, p + d, len - d, (off_t)(at + d));
        if (w < 0) { if (errno == EINTR) continue; return false; }
        d += (size_t)w;
      }
      return true;
    };
    for (;;) {
      DJob j;
      { std::unique_lock<std::mutex> lk(ld_mx_);
        if (ld_jobs_.empty() && !ld_done_) {
          uint64_t t0 = now_ns();
          ld_jobcv_.wait(lk, [&] { return !ld_jobs_.empty() || ld_done_; });
          ld_jobwait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);  // PARSE-bound
        }
        if (ld_jobs_.empty()) return;          // done and drained
        j = std::move(ld_jobs_.front()); ld_jobs_.pop_front(); }

      if (j.kind == DJob::FINALIZE) {
        // ftruncate fixes the exact size (materializes trailing holes / a tail
        // that a write error left short), then metadata, then close.
        if (::ftruncate(j.fd, (off_t)j.final_size) != 0 && !ld_werr_.load()) {
          fail(j.rel, "ftruncate failed"); ld_werr_.store(true);
        }
        set_meta_fd(j.fd, j.mode, j.mtime, j.uid, j.gid);
        apply_ext(j.fd, /*is_dir=*/false, j.ext);
        ::close(j.fd);
        continue;
      }

      const char * buf = static_cast<const char *>(dbufs_[j.bi]);
      bool ok = true;
      uint64_t wt0 = now_ns();
      for (auto & run : j.runs)
        if (!pwrite_at(j.fd, buf + std::get<0>(run), std::get<2>(run), std::get<1>(run))) { ok = false; break; }
      ld_write_ns_.fetch_add(now_ns() - wt0, std::memory_order_relaxed);
      if (ok && j.tail_len) {  // final sub-block tail: O_DIRECT can't do unaligned
#ifdef O_DIRECT
        int fl = ::fcntl(j.fd, F_GETFL); if (fl >= 0) ::fcntl(j.fd, F_SETFL, fl & ~O_DIRECT);
#endif
        if (!pwrite_at(j.fd, buf + j.tail_off, j.tail_len, j.tail_fpos)) ok = false;
      }
      { std::lock_guard<std::mutex> lk(ld_mx_); ld_freeq_.push_back(j.bi); }
      ld_freecv_.notify_one();
      // Don't return on error — keep draining so pending FINALIZE jobs still
      // close their fds (no leak); ftruncate is skipped once werr is set.
      if (!ok && !ld_werr_.load()) { fail(j.rel, "write failed"); ld_werr_.store(true); }
    }
  }

  // Join the persistent writer after the parse thread has enqueued every file's
  // jobs (called from stop_pool()).  Propagates a write error to had_error_.
  void ld_stop_writer() {
    if (!ld_started_ || ld_nobuf_) return;
    { std::lock_guard<std::mutex> lk(ld_mx_); ld_done_ = true; } ld_jobcv_.notify_one();
    if (ld_writer_.joinable()) ld_writer_.join();
    if (ld_werr_.load()) had_error_.store(true, std::memory_order_relaxed);
  }

  // O_DIRECT large-file extract (Gen4+ --direct): bypass the page cache so the
  // decompressor isn't throttled by buffered writeback.  Sparse-aware: all-zero
  // 4 KiB blocks are skipped as holes when sparse is on; runs of non-zero blocks
  // are coalesced into one aligned O_DIRECT write.  Data is handed to the
  // PERSISTENT writer (above) so the writes of this file overlap parsing of the
  // next; the parse thread is the sole sink reader and never blocks on a join.
  // Falls back to the buffered stream_large if O_DIRECT or the buffer pool is
  // unavailable (open failure, or a filesystem like tmpfs that lacks O_DIRECT).
  void stream_large_direct(StreamReader & r, const std::string & rel, const InEntry & e) {
    if (ld_werr_.load()) { r.skip(e.size); return; }   // a prior write failed: drain, skip
    int fd = create_file(rel, e.mode, /*o_direct=*/true);
    if (fd < 0) { stream_large(r, rel, e); return; }
    if (!ld_ensure_writer()) { ::close(fd); stream_large(r, rel, e); return; }
    static constexpr size_t AL = 4096;
    const bool sp = (opt_.sparse_mode != 0);
    auto blk_zero = [](const char * p) {
      const uint64_t * w = reinterpret_cast<const uint64_t *>(p);  // word-wise scan
      for (size_t i = 0; i < AL / sizeof(uint64_t); ++i) if (w[i]) return false;
      return true;
    };
    ld_files_.fetch_add(1, std::memory_order_relaxed);
    ld_bytes_.fetch_add(e.size, std::memory_order_relaxed);
    uint64_t left = e.size, fpos = 0; bool ok = true;
    while (left && ok) {
      // A write failed on another file: stop writing, but keep the sink aligned
      // by consuming this file's remaining bytes, so later entries parse cleanly.
      if (ld_werr_.load()) { r.skip(left); left = 0; ok = false; break; }
      int bi;
      { std::unique_lock<std::mutex> lk(ld_mx_);
        if (ld_freeq_.empty() && !ld_werr_.load()) {
          uint64_t t0 = now_ns();
          ld_freecv_.wait(lk, [&] { return !ld_freeq_.empty() || ld_werr_.load(); });
          ld_bufwait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);  // WRITE-bound
        }
        if (ld_werr_.load()) continue;         // re-check at loop top (drains via skip)
        bi = ld_freeq_.front(); ld_freeq_.pop_front(); }
      char * buf = static_cast<char *>(dbufs_[bi]);
      size_t fill = 0;
      uint64_t ft0 = now_ns();
      while (fill < DBUF_CAP && left) {
        size_t got = r.read_some(buf + fill, (size_t)std::min<uint64_t>(DBUF_CAP - fill, left));
        if (!got) { ok = false; break; }       // sink truncated
        fill += got; left -= got;
        if (m_) m_->wrote_bytes.fetch_add(got, std::memory_order_relaxed);
      }
      ld_fill_ns_.fetch_add(now_ns() - ft0, std::memory_order_relaxed);  // sink read + memcpy
      uint64_t st0 = now_ns();
      size_t aligned = (fill / AL) * AL, off = 0;
      DJob j; j.kind = DJob::WRITE; j.fd = fd; j.bi = bi; j.rel = rel;
      while (off < aligned) {
        if (sp && blk_zero(buf + off)) { fpos += AL; off += AL; continue; }  // hole
        size_t run = off;
        while (off < aligned && !(sp && blk_zero(buf + off))) off += AL;
        j.runs.emplace_back(run, fpos, off - run); fpos += (off - run);
      }
      j.tail_off = aligned; j.tail_len = fill - aligned; j.tail_fpos = fpos;
      fpos += j.tail_len;
      ld_scan_ns_.fetch_add(now_ns() - st0, std::memory_order_relaxed);  // zero-scan + plan
      { std::lock_guard<std::mutex> lk(ld_mx_); ld_jobs_.push_back(std::move(j)); }
      ld_jobcv_.notify_one();
    }
    if (!ok && !ld_werr_.load()) { fail(rel, "write failed"); ld_werr_.store(true); }
    // FINALIZE (ftruncate to e.size + metadata + close) runs on the writer after
    // this file's WRITE jobs drain — the parse thread does NOT wait for it.
    DJob f; f.kind = DJob::FINALIZE; f.fd = fd; f.final_size = e.size;
    f.mode = e.mode; f.mtime = e.mtime; f.uid = e.uid; f.gid = e.gid; f.ext = e.ext; f.rel = rel;
    { std::lock_guard<std::mutex> lk(ld_mx_); ld_jobs_.push_back(std::move(f)); }
    ld_jobcv_.notify_one();
  }

  void finish_deferred() {
    // Hardlinks: target file is now written.  Resolve both paths securely.
    for (const auto & h : hardlinks_) {
      std::string nleaf, tleaf;
      int npfd = open_parent(h.name, nleaf, true);
      int tpfd = (h.target.empty() ? -1 : open_parent(h.target, tleaf, false));
      if (npfd < 0 || tpfd < 0) { if (npfd >= 0) ::close(npfd); if (tpfd >= 0) ::close(tpfd); fail(h.name, "cannot create hardlink"); continue; }
      ::unlinkat(npfd, nleaf.c_str(), 0);
      if (::linkat(tpfd, tleaf.c_str(), npfd, nleaf.c_str(), 0) != 0) fail(h.name, std::string("hardlink: ") + std::strerror(errno));
      ::close(npfd); ::close(tpfd);
    }
    // Directory metadata in reverse order, so a parent's mtime isn't bumped by
    // writing its children after it.
    for (auto it = dirmeta_.rbegin(); it != dirmeta_.rend(); ++it) {
      std::string leaf; int pfd = open_parent(it->path, leaf, false);
      if (pfd < 0) continue;
      int dfd = ::openat(pfd, leaf.c_str(), O_DIRECTORY | O_NOFOLLOW | O_RDONLY | O_CLOEXEC);
      ::close(pfd);
      if (dfd < 0) continue;
      set_meta_fd(dfd, it->mode, it->mtime, it->uid, it->gid);
      apply_ext(dfd, /*is_dir=*/true, it->ext);
      ::close(dfd);
    }
  }
};

}  // namespace tarx
#endif  // !_WIN32

static void compress_cpu_mt(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // ---- Performance instrumentation (active at -vvv) ----
  PerfCounters perf_local;
  if (opt.verbosity >= V_TRACE) g_perf = &perf_local;

  int threads = resolve_cpu_threads(opt.cpu_threads);

  // Option A: if single-threaded, use simple streaming helper.  --tar has no
  // single FILE* to stream; it always uses the chunked queue path below (a
  // 1-worker pool still works), so the tar producer can feed it.
  if (threads == 1 && !opt.tar_mode) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[CPU] single-thread streaming path (-T 1)\n");
    compress_cpu_stream(in, out, opt, m);
    if (g_perf) {
      g_perf->print_summary("CPU-ONLY COMPRESS (single-thread)");
      g_perf = nullptr;
    }
    return;
  }

  CpuAgg cpuagg{};
  cpuagg.threads = threads;
  cpuagg.per_thread.resize((size_t)threads);

  // Determine chunk size (auto or user-specified)
  // Chunk size: 16 MiB default, or user-specified via --chunk-size
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;

  // Ultra levels need chunk >= window size to be effective.
  {
    size_t ultra_min = ultra_min_chunk_mib(opt.level, opt.ultra);
    if (ultra_min > 0 && chosen_mib < ultra_min) {
      if (!opt.chunk_user_set) {
        vlog(V_VERBOSE, opt, "[ULTRA] auto-increasing chunk size from "
             + std::to_string(chosen_mib) + " to " + std::to_string(ultra_min)
             + " MiB (must be >= window size)\n");
        chosen_mib = ultra_min;
      } else {
        vlog(V_ERROR, opt, "warning: --chunk-size=" + std::to_string(chosen_mib)
             + " MiB is smaller than the ultra window ("
             + std::to_string(ultra_min) + " MiB). Compression ratio will suffer.\n");
      }
    }
  }

  // Pre-flight: check we won't OOM  may reduce chunk size
  chosen_mib = check_ram_budget(threads, chosen_mib, opt);
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // Warn if RAM budget forced chunk below ultra minimum
  {
    size_t ultra_min = ultra_min_chunk_mib(opt.level, opt.ultra);
    if (ultra_min > 0 && chosen_mib < ultra_min) {
      vlog(V_ERROR, opt, "warning: RAM budget reduced chunk to "
           + std::to_string(chosen_mib) + " MiB, below ultra window ("
           + std::to_string(ultra_min) + " MiB). Consider reducing -T or adding RAM.\n");
    }
  }

  // Get total input size for progress percentage (unknown for pipes).
  // --tar: build the archive layout up front (single-threaded walk) so the
  // progress bar has a real denominator and the parallel assembler can run.
#ifndef _WIN32
  tarx::TarLayout tar_layout;
  bool tar_built = false;
  if (opt.tar_mode) { tar_layout = tarx::build_layout(opt, m); tar_built = true; }
  uint64_t total_in = tar_built ? tar_layout.total_size : known_input_size(opt, in);
#else
  uint64_t total_in = known_input_size(opt, in);
#endif

  // Preallocate output file: input size is a safe upper bound for compressed output.
#ifndef _WIN32
  if (g_direct_writer && total_in > 0 && opt.preallocate_output
      && g_direct_writer->preallocate(total_in)) {
    char sz[32]; human_bytes(double(total_in), sz, sizeof(sz));
    vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
  }
#endif

  // Start progress bar and ordered-writer threads
  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  TaskQueue queue;
  // Byte-cap the input queue: a pipe/stdin producer that outruns the workers
  // (fread reads frames into heap) must not buffer the whole input in RAM — the
  // compress-side analogue of the ROADMAP 7.8 decompress cap.  Bytes only (no
  // frame cap): mmap frames are zero-copy views (data.size()==0), so this is a
  // no-op for the common regular-file path and bounds only the fread path.
  // ~half a frame per slot; tunable via --throttle-factor.
  if (opt.throttle_frames != 0) {
    int qslack = opt.throttle_factor > 0 ? opt.throttle_factor : THROTTLE_SLACK_FACTOR;
    size_t qfloor = (size_t)std::max(THROTTLE_MIN_FRAMES, threads * qslack);
    queue.set_max_bytes(qfloor * (host_chunk / 2));
  }
  ResultStore results;
  // Size the throttle from the *resolved* chunk (host_chunk = chosen_mib),
  // not opt.chunk_mib: chosen_mib may have been auto-bumped for --ultra or
  // shrunk by check_ram_budget, and the RAM-cap term divides avail/2 by the
  // frame size — a stale 16 MiB there over- or under-shoots the in-flight cap
  // on ultra / low-RAM runs.  (ROADMAP 7.3.)
  FrameThrottle throttle(compute_throttle_budget(host_chunk, threads, 0, opt));
  std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m, &throttle);

  std::vector<std::thread> pool; pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) pool.emplace_back(cpu_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
    nullptr, nullptr,
#endif
    &cpuagg, &throttle);
  if (opt.verbosity >= V_VERBOSE) std::cerr << "[CPU] " << threads << " worker threads online\n";

  // Producer loop: read input in chunks and enqueue for compression.
  // For regular files, mmap the input for zero-copy: workers read directly
  // from the mapped pages, eliminating the fread + memcpy bottleneck that
  // serialises the single-threaded producer.  Pipes/stdin fall back to fread.
  try_boost_io_priority(true);
  std::atomic<size_t> seq{0};
#ifndef _WIN32
  MmapRegion mmap_region;
  DirectReadPool dr_pool;   // zero-copy --direct-read buffers; outlives the workers
  bool reader_done = false;
  // --tar: the parallel archive assembler IS the producer (synthesizes the tar
  // stream + reads member files concurrently, pushing chunks in sequence order).
  if (opt.tar_mode) {
    tarx::assemble(queue, tar_layout, host_chunk, opt, m);
    reader_done = true;
  }
  // Pooled zero-copy reader, shared by --direct-read (O_DIRECT) and the buffered
  // fallback below.  Sizes the pool once: enough to keep every worker fed plus a
  // read-ahead backlog, but no more than the file needs, capped so RAM stays sane
  // (a buffer is held only from pread until the worker finishes compressing, not
  // during write).
  bool dr_pool_ok = false, dr_pool_tried = false;
  // borrowed_fd >= 0: pread that already-open fd (a seekable stdin redirect),
  // sizing the pool from known_size; else read opt.input by path.
  auto run_pooled_reader = [&](bool o_direct, int borrowed_fd, uint64_t known_size) -> bool {
    const size_t cap = ((host_chunk + 4095) / 4096) * 4096;
    // Buffered mode fans the kernel copy out over several threads (the wall
    // is the per-thread cold-destination copy at ~2.5-3.5 GB/s, not the
    // device — see pooled_read_chunks).  Auto count scales with the worker
    // pool: measured optima are 3 on a 24-thread box and 12 on a 96-worker
    // dual-socket server (7.46 -> 18.74 GiB/s from 3 -> 12 there);
    // --read-threads overrides.
    const int n_readers = o_direct ? 1
                        : (opt.read_threads > 0 ? (int)opt.read_threads
                                                : std::max(3, std::min(12, threads / 8)));
    if (!dr_pool_tried) {
      dr_pool_tried = true;
      uint64_t fsz = known_size;
      if (borrowed_fd < 0) { std::error_code fec; uintmax_t z = fs::file_size(opt.input, fec); fsz = fec ? 0 : (uint64_t)z; }
      size_t file_chunks = (fsz > 0) ? (size_t)((fsz + cap - 1) / cap) : (size_t)threads;
      // 32 extra buffers per reader: at 12 readers the original
      // threads+128 sizing showed 15% blocked-on-pool — the readers, not
      // the device, were starving for buffers.
      size_t pool_n = std::min<size_t>((size_t)threads + 128 + 32 * (size_t)n_readers,
                                       std::min<size_t>(file_chunks + 1, 1024));
      if (pool_n < 8) pool_n = std::min<size_t>(8, std::max<size_t>(file_chunks + 1, 1));
      dr_pool_ok = dr_pool.init(cap, pool_n,
                                /*want_thp=*/o_direct || kernel_has_per_vma_locks());
      if (dr_pool_ok) g_direct_read_pool = &dr_pool;
      if (opt.verbosity >= V_VERBOSE)
        vlog(V_VERBOSE, opt,
             std::string(o_direct ? "[DIRECT-READ] O_DIRECT input (page cache bypassed)"
                                  : "[POOLED-READ] buffered input (page cache + readahead)")
             + (dr_pool_ok ? ", zero-copy pool " + std::to_string(pool_n) + " buffers\n"
                           : " (pool alloc failed; copy path)\n"));
    }
    return pooled_read_chunks(opt.input, host_chunk, m, dr_pool_ok ? &dr_pool : nullptr,
      o_direct, n_readers, borrowed_fd,
      [&](const char * buf, size_t n, size_t idx, int slot) {
        if (slot >= 0) {                              // zero-copy: Task aliases the pooled buffer
          Task t; t.seq = idx; t.view_ptr = buf; t.view_len = n; t.direct_buf = slot;
          queue.push(std::move(t));
        } else {                                      // copy fallback (pool unavailable)
          enqueue_direct_chunk(queue, idx, buf, n);   // seq deterministic by file position
        }
        return true;
      });
  };
  // --direct-read: O_DIRECT input (bypass the page cache).  Takes precedence over
  // mmap (mmap IS the page cache); falls through to fread if O_DIRECT can't open.
  if (opt.direct_read && opt.input != "-" && fs::is_regular_file(opt.input))
    reader_done = run_pooled_reader(true, -1, 0);
  if (!reader_done && opt.use_mmap && opt.input != "-" && fs::exists(opt.input)
      && fs::is_regular_file(opt.input)
      && mmap_ok_for_input(opt, opt.input)
      && mmap_region.open(opt.input.c_str())) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[MMAP] using zero-copy reader\n");
    const char * base = mmap_region.data();
    const size_t file_size = mmap_region.size();
    size_t off = 0;
    uint64_t mmap_rd_t0 = g_perf ? now_ns() : 0;
    while (off < file_size) {
      size_t n = std::min(host_chunk, file_size - off);
      Task t;
      t.seq = seq.fetch_add(1, std::memory_order_relaxed);
      t.view_ptr = base + off;
      t.view_len = n;
      queue.push(std::move(t));
      off += n;
    }
    if (g_perf) {
      g_perf->read_ns.fetch_add(now_ns() - mmap_rd_t0);
      g_perf->read_bytes_total.fetch_add(file_size);
    }
    reader_done = true;
  }
  // mmap declined (pre-6.4 kernel gate, --no-mmap, or open failure) but the
  // input is preadable — a named regular file/block device OR a seekable
  // stdin redirect ("gzstd < big.bin"): buffered zero-copy pooled reads
  // instead of the fread+copy fallback.  Same pool/view machinery as
  // --direct-read, through the page cache, with the kernel copy fanned out
  // over a few reader threads.  fstat (probe_preadable_input) decides;
  // pipes (tar -I gzstd) fall through to fread below.
  if (!reader_done) {
    bool owns = false; uint64_t psz = 0;
    int pfd = probe_preadable_input(opt, in, &owns, &psz);
    if (pfd >= 0) {
      reader_done = run_pooled_reader(false, pfd, psz);
      if (owns) ::close(pfd);   // pooled reader borrows the fd; we opened it
    }
  }
  if (!reader_done)
#endif
  {
    std::vector<char> host_in(host_chunk);
    while (true) {
      uint64_t rd_t0 = (g_perf || m) ? now_ns() : 0;
      size_t n = std::fread(host_in.data(), 1, host_chunk, in);
      if ((g_perf || m) && n > 0) {
        const uint64_t r_dt = now_ns() - rd_t0;
        if (m) m->reader_io_ns.fetch_add(r_dt, std::memory_order_relaxed);
        if (g_perf) {
          g_perf->read_ns.fetch_add(r_dt);
          g_perf->read_bytes_total.fetch_add(n);
        }
      }
      if (n == 0) break;
      if (m) m->read_bytes.fetch_add(n);

      Task t;
      t.seq = seq.fetch_add(1, std::memory_order_relaxed);
      const uint64_t c_t0 = m ? now_ns() : 0;
      t.data.assign(host_in.data(), host_in.data() + n);
      if (m) m->reader_copy_ns.fetch_add(now_ns() - c_t0, std::memory_order_relaxed);
      queue.push(std::move(t));
    }
  }

  // Signal workers that no more input is coming
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
    if (m) {
      m->total_frames.store(results.total_tasks, std::memory_order_relaxed);
      // NOTE: do NOT set total_out_final here.  Unlike decompress (where
      // total_out is summed from frame-header decomp_size during pre-scan),
      // compress's total_out is a running accumulator — it grows as workers
      // finish more frames.  Setting total_out_final = true at producer-done
      // makes the progress code use `wrote_bytes / total_out_so_far` (the
      // writer's catch-up percentage), which jumps to ~90% almost immediately.
      // The GPU compress path correctly leaves this unset; match that here.
    }
  }
  results.cv.notify_all();

  // Wait for all workers to finish, then signal the writer.
  // Do NOT call throttle.set_done() before join: workers must respect
  // throttle while draining the queue to avoid buffering entire output in RAM.
  for (auto & th : pool) th.join();
  g_direct_read_pool = nullptr;  // workers done: no more release() calls; safe to drop the pool
  throttle.set_done();  // safe now: all workers exited
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();

  // Wait for writer and progress threads to finish
  wthr.join();
  progress_done = true;
  progress_thr.join();
  log_throttle_stats(throttle, opt, "compress-cpu");

  if (g_perf) {
    g_perf->print_summary("CPU-ONLY COMPRESS");
    g_perf = nullptr;
  }
}

/*======================================================================
 GPU path (nvCOMP) + Hybrid
======================================================================*/
#ifdef HAVE_NVCOMP
// Per-device aggregate timing stats (accumulated across all streams).
struct DevStats {
  std::mutex m;
  double   h2d_ms   = 0.0;   // host-to-device transfer time
  double   comp_ms  = 0.0;   // GPU compression kernel time
  double   d2h_ms   = 0.0;   // device-to-host transfer time
  double   total_ms = 0.0;   // wall-clock total
  uint64_t in_bytes  = 0;
  uint64_t out_bytes = 0;
  uint64_t batches   = 0;
};

// Shared auto-tune state across all GPU workers.
// One worker acts as the "tuner" (first to complete a measurement window).
// All workers read shared_batch_size for their pop_batch_greedy calls.
// This prevents fast-settling GPUs from starving slow-settling ones.
struct SharedTuneState {
  std::atomic<size_t> batch_size{8};     // current batch size (all GPUs use this)
  std::atomic<bool>   locked{false};     // true when user specified --gpu-batch
  std::mutex          tune_mtx;          // protects tune logic (only one tuner at a time)

  // Tune algorithm state (protected by tune_mtx)
  enum class Phase { BASELINE, HALVE, DOUBLE, REFINE, SETTLED };
  Phase    phase = Phase::BASELINE;
  size_t   best_batch = 0;
  double   best_thr = 0.0;
  size_t   prev_batch = 0;
  double   prev_thr = 0.0;
  size_t   refine_lo = 0, refine_hi = 0;
  uint32_t refine_iters = 0;            // count refine steps to prevent oscillation
  uint32_t probe_count = 0;             // total probes (for alternating up/down)
  static constexpr uint32_t MAX_REFINE_ITERS = 6;  // settle after this many
  std::atomic<size_t> vram_ceiling{1024};          // max batch that fits in VRAM (set at init)
  uint32_t settle_ticks = 0;
  std::chrono::steady_clock::time_point last_tune = std::chrono::steady_clock::now();

  // Throughput accumulator (all GPUs add to this)
  std::atomic<uint64_t> window_bytes{0};
  std::atomic<uint64_t> window_ns{0};
  std::atomic<uint32_t> window_batches{0};

  static constexpr uint32_t MIN_BATCHES = 4;     // min batches across all GPUs before tuning
  static constexpr uint32_t PROBE_INTERVAL = 8;   // ticks in SETTLED before re-probing
  static constexpr double   TUNE_SEC = 0.3;       // min seconds between tune decisions
};

// Rate-matched dispatch: tracks CPU and GPU throughput to partition work
// so both finish at roughly the same time.  This minimizes out-of-order
// delivery to the writer.
//
// Usage:
//   - GPU workers call report_gpu() after each batch completion
//   - CPU workers call report_cpu() after each frame completion
//   - CPU workers call should_cpu_take() to check if they should grab work
//   - The dispatcher calculates how many CPU frames to allow per GPU batch cycle

// Per-stream stats for JSON export.
struct StreamStats {
  size_t   dev_index      = 0;
  size_t   stream_index   = 0;
  bool     pinned_h2d     = false;
  bool     pinned_d2h     = false;
  size_t   batch_capacity = 0;
  double   h2d_ms   = 0.0;
  double   comp_ms  = 0.0;
  double   d2h_ms   = 0.0;
  double   total_ms = 0.0;
  uint64_t in_bytes  = 0;
  uint64_t out_bytes = 0;
  uint64_t batches   = 0;
  uint64_t chunks    = 0;
};

// Collects per-stream stats from all devices for JSON output.
struct StatsSink {
  std::mutex m;
  std::vector<std::vector<StreamStats>> per_dev;
  explicit StatsSink(int n) : per_dev(size_t(n)) {}
};

using GetTempBytesSig = nvcompStatus_t (*)(size_t, size_t, nvcompBatchedZstdCompressOpts_t, size_t *, size_t);
using GetTempStreamSig= nvcompStatus_t (*)(size_t, size_t, nvcompBatchedZstdCompressOpts_t, size_t *, cudaStream_t);
static constexpr bool NVCOMP_GETTEMP_USES_BYTES = std::is_same<decltype(&nvcompBatchedZstdCompressGetTempSizeAsync), GetTempBytesSig>::value;

template<bool UsesBytes>
static inline nvcompStatus_t call_get_temp_size_async(size_t chunks, size_t gpu_chunk, nvcompBatchedZstdCompressOpts_t comp_opts, size_t * temp_out, cudaStream_t stream)
{
  if constexpr (UsesBytes) { (void)stream; return nvcompBatchedZstdCompressGetTempSizeAsync(chunks, gpu_chunk, comp_opts, temp_out, chunks*gpu_chunk); }
  else { return nvcompBatchedZstdCompressGetTempSizeAsync(chunks, gpu_chunk, comp_opts, temp_out, stream); }
}
static inline size_t get_nvcomp_temp_size(size_t chunks, size_t gpu_chunk, nvcompBatchedZstdCompressOpts_t comp_opts, cudaStream_t stream)
{
  size_t t=0; nvcompStatus_t st = call_get_temp_size_async<NVCOMP_GETTEMP_USES_BYTES>(chunks, gpu_chunk, comp_opts, &t, stream);
  if (st != nvcompSuccess) throw std::runtime_error("nvcompBatchedZstdCompressGetTempSizeAsync failed (status " + std::to_string(int(st)) + ")");
  return t;
}

// Per-stream GPU context: holds device buffers, CUDA events, and batch state
// for one CUDA stream on one device.
struct StreamCtx {
  cudaStream_t stream{};

  // Device memory: contiguous buffers split into per-subchunk slots
  void * d_in_base   = nullptr;   // input data (N * gpu_chunk bytes)
  void * d_out_base  = nullptr;   // compressed output (N * max_out_chunk bytes)
  void * d_temp      = nullptr;   // nvCOMP temporary workspace

  // Device arrays: pointers and sizes for nvCOMP batched API
  void **         d_in_ptrs    = nullptr;
  void **         d_out_ptrs   = nullptr;
  size_t *        d_in_sizes   = nullptr;
  size_t *        d_comp_sizes = nullptr;
  nvcompStatus_t* d_stats      = nullptr;
  size_t          temp_bytes_used = 0;

  // Pinned host memory for H2D + D2H transfers (optional).
  // ONE buffer per stream is sized to max(gpu_chunk, max_out_chunk) per
  // slot and used for both directions (input upload, then output readback).
  // h_io_slot_bytes is the per-slot stride so the D2H code can address slots.
  // Bytes tracked separately so we can release back to the global budget.
  void * h2d_pinned_base = nullptr;
  size_t h2d_pinned_bytes = 0;
  size_t h_io_slot_bytes = 0;  // per-slot stride within h2d_pinned_base
  void * d2h_pinned_base = nullptr;
  size_t d2h_pinned_bytes = 0;

  // Host-side vectors (mirroring device arrays for readback)
  std::vector<size_t>         h_in_sizes;
  std::vector<size_t>         h_comp_sizes;
  std::vector<uint32_t>       h_checksums;   // XXH64-low32 of each chunk's uncompressed data
  std::vector<nvcompStatus_t> h_stats;
  std::vector<Task>           batch;

  // Configuration and state
  size_t      gpu_chunk        = 0;
  size_t      max_out_chunk    = 0;
  size_t      per_stream_batch = 0;   // max subchunks per batch
  cudaEvent_t ev_h2d_begin{};
  cudaEvent_t ev_h2d_end{};
  cudaEvent_t ev_comp_end{};
  cudaEvent_t ev_d2h_end{};
  bool        busy   = false;         // true while a batch is in-flight
  size_t      filled = 0;             // subchunks in current batch
  size_t      delivered = 0;          // frames of the current batch already pushed to ResultStore
                                      // (lets the failure path rescue only the undelivered tail)
  StreamStats stats{};
  std::chrono::steady_clock::time_point last_adjust{ std::chrono::steady_clock::now() };

  // Recycled host output buffers for D2H readback — same pool as the decompress
  // path (ROADMAP 7.2): reuse a slot whose use_count()==1 (writer has drained
  // it) instead of a fresh make_shared per frame.  Compress output is the
  // *compressed* bytes (smaller than decomp), so fault pressure is lower, but
  // the pool still removes the per-frame allocation churn.  Grows lazily to a
  // cap; deadlock-free by the throttle FIFO argument (frames pushed in seq
  // order, writer drains the oldest and frees a slot).
  std::vector<FrameBuf> out_pool;
  uint64_t out_pool_waits = 0;
  FrameBuf acquire_out_buf(size_t cap, FrameThrottle * bp) {
    for (;;) {
      for (auto & b : out_pool)
        if (b.use_count() == 1) return b;
      if (out_pool.size() < cap) {
        out_pool.push_back(std::make_shared<FrameVec>());
        return out_pool.back();
      }
      ++out_pool_waits;
      if (bp) {
        bp->wait_for_drain([&]{
          for (auto & b : out_pool) if (b.use_count() == 1) return true;
          return false;
        });
      } else {
        std::this_thread::yield();
      }
    }
  }

  // (Per-stream EXPLORE/REFINE/SETTLE batch-size tuner removed v0.13.34: it was
  // dead code, superseded by the cross-GPU SharedTuneState hill-climb that all
  // streams/devices share.  Batch size now comes solely from shared_tune.)
};

static bool allocate_stream_buffers(StreamCtx & C, size_t per_stream_batch, size_t gpu_chunk, size_t max_out_chunk, nvcompBatchedZstdCompressOpts_t comp_opts, const Options & opt)
{
  C.per_stream_batch = per_stream_batch;
  C.gpu_chunk = gpu_chunk;
  C.max_out_chunk = max_out_chunk;

  // Pre-check: estimate total VRAM needed and compare to free memory.
  // cudaMalloc can hang on some drivers if the request exceeds VRAM,
  // so we fail fast here instead of letting the driver block.
  {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    // Get actual temp workspace size from nvCOMP (can be very large for big batches)
    size_t temp_est = 0;
    try {
      temp_est = get_nvcomp_temp_size(per_stream_batch, gpu_chunk, comp_opts, C.stream);
    } catch (...) {
      return false;  // nvCOMP can't even compute temp size for this config
    }

    size_t est_needed = per_stream_batch * gpu_chunk        // input
                      + per_stream_batch * max_out_chunk    // output
                      + temp_est                            // nvCOMP temp workspace
                      + per_stream_batch * (sizeof(void*)*2 + sizeof(size_t)*2 + sizeof(nvcompStatus_t));
    if (opt.verbosity >= V_DEBUG) {
      fprintf(stderr, "[VRAM check] batch=%zu gpu_chunk=%zu max_out=%zu temp=%zu MiB est=%zu MiB free=%zu MiB\n",
              per_stream_batch, gpu_chunk/ONE_MIB, max_out_chunk/ONE_MIB,
              temp_est/ONE_MIB, est_needed/ONE_MIB, free_b/ONE_MIB);
    }
    if (est_needed > free_b * 0.90) {
      return false;
    }
  }

  // Get temporary workspace size required by nvCOMP
  size_t temp_bytes = get_nvcomp_temp_size(per_stream_batch, gpu_chunk, comp_opts, C.stream);
  C.temp_bytes_used = temp_bytes;

  // Allocate device buffers for input, output, and metadata
  if (cudaMalloc(&C.d_in_base,    C.per_stream_batch * gpu_chunk)     != cudaSuccess) return false;
  if (cudaMalloc(&C.d_out_base,   C.per_stream_batch * max_out_chunk) != cudaSuccess) return false;
  if (temp_bytes > 0) {
    if (cudaMalloc(&C.d_temp, temp_bytes) != cudaSuccess) return false;
  }
  if (cudaMalloc(&C.d_in_ptrs,    sizeof(void*)          * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_out_ptrs,   sizeof(void*)          * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_in_sizes,   sizeof(size_t)         * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_comp_sizes, sizeof(size_t)         * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_stats,      sizeof(nvcompStatus_t) * C.per_stream_batch) != cudaSuccess) return false;
  // Optionally allocate pinned (page-locked) host memory for faster H2D
  // and D2H transfers.  ONE buffer per stream serves both directions:
  // the slot holds input pre-upload, then the same slot holds compressed
  // output post-D2H.  Sized to max(gpu_chunk, max_out_chunk) per slot so
  // either direction fits.  Safe because batches are serialised within a
  // stream (C.busy gates re-pop until D2H + result delivery complete).
  // AUTO mode rations to <=50% of system RAM via the global budget;
  // ON forces; OFF skips.  Streams that don't fit the budget (or fail
  // cudaHostAlloc) silently fall back to pageable memory.
  const size_t pin_slot_bytes = std::max(gpu_chunk, max_out_chunk);
  const size_t pin_h2d_bytes = C.per_stream_batch * pin_slot_bytes;
  C.h_io_slot_bytes = pin_slot_bytes;
  if (try_reserve_pinned(pin_h2d_bytes, opt)) {
    if (cudaHostAlloc(&C.h2d_pinned_base, pin_h2d_bytes,
                      cudaHostAllocDefault) != cudaSuccess) {
      C.h2d_pinned_base = nullptr;
      release_pinned(pin_h2d_bytes, opt);
      if (opt.verbosity >= V_VERBOSE) {
        char sz[32]; human_bytes(double(pin_h2d_bytes), sz, sizeof(sz));
        vlog(V_VERBOSE, opt, std::string("[PINNED] H2D/D2H alloc failed (")
             + sz + "); using pageable for this stream\n");
      }
    } else {
      C.h2d_pinned_bytes = pin_h2d_bytes;
      if (opt.verbosity >= V_VERBOSE) {
        char sz[32]; human_bytes(double(pin_h2d_bytes), sz, sizeof(sz));
        vlog(V_VERBOSE, opt, std::string("[PINNED] H2D+D2H ") + sz
             + " reserved (shared per slot)\n");
      }
    }
  } else if (opt.pin_mode == PinMode::AUTO) {
    if (opt.verbosity >= V_VERBOSE) {
      char sz[32]; human_bytes(double(pin_h2d_bytes), sz, sizeof(sz));
      char rem[32];
      uint64_t budget = g_pinned_bytes_budget.load();
      uint64_t reserved = g_pinned_bytes_reserved.load();
      human_bytes(double(budget > reserved ? budget - reserved : 0),
                  rem, sizeof(rem));
      vlog(V_VERBOSE, opt, std::string("[PINNED] H2D ") + sz
           + " skipped (auto budget exhausted, " + rem
           + " left); using pageable\n");
    }
  }
  C.d2h_pinned_base = nullptr;  // compress D2H uses exact-size copies (variable output)

  // Build pointer arrays: each slot points to its region of the large buffer
  std::vector<void*> h_in_ptrs(C.per_stream_batch);
  std::vector<void*> h_out_ptrs(C.per_stream_batch);
  for (size_t i = 0; i < C.per_stream_batch; ++i) {
    h_in_ptrs[i]  = static_cast<char*>(C.d_in_base)  + i * gpu_chunk;
    h_out_ptrs[i] = static_cast<char*>(C.d_out_base) + i * max_out_chunk;
  }

  // Upload pointer arrays to device
  if (cudaMemcpyAsync(C.d_in_ptrs, h_in_ptrs.data(),
                      sizeof(void*) * C.per_stream_batch,
                      cudaMemcpyHostToDevice, C.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(C.d_out_ptrs, h_out_ptrs.data(),
                      sizeof(void*) * C.per_stream_batch,
                      cudaMemcpyHostToDevice, C.stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(C.stream) != cudaSuccess) return false;

  // Allocate host-side vectors for sizes and statuses
  C.h_in_sizes.resize(C.per_stream_batch);
  C.h_comp_sizes.resize(C.per_stream_batch);
  C.h_checksums.resize(C.per_stream_batch);
  C.h_stats.resize(C.per_stream_batch);
  C.batch.reserve(C.per_stream_batch);

  // Create CUDA events for timing
  cudaEventCreateWithFlags(&C.ev_h2d_begin, cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_h2d_end,   cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_comp_end,  cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_d2h_end,   cudaEventDefault);

  C.stats.pinned_h2d = (C.h2d_pinned_base != nullptr);
  // The same buffer is reused for D2H (compress output readback).
  C.stats.pinned_d2h = (C.h2d_pinned_base != nullptr);
  C.stats.batch_capacity = per_stream_batch;
  return true;
}

static void free_stream_buffers_only(StreamCtx & C, const Options & opt)
{
  if (C.d_in_base) { cudaFree(C.d_in_base); }
  if (C.d_out_base) { cudaFree(C.d_out_base); }
  if (C.d_temp) { cudaFree(C.d_temp); }
  if (C.d_in_ptrs) { cudaFree(C.d_in_ptrs); }
  if (C.d_out_ptrs) { cudaFree(C.d_out_ptrs); }
  if (C.d_in_sizes) { cudaFree(C.d_in_sizes); }
  if (C.d_comp_sizes) { cudaFree(C.d_comp_sizes); }
  if (C.d_stats) { cudaFree(C.d_stats); }
  if (C.h2d_pinned_base) {
    cudaFreeHost(C.h2d_pinned_base);
    release_pinned(C.h2d_pinned_bytes, opt);
    C.h2d_pinned_bytes = 0;
  }
  if (C.d2h_pinned_base) {
    cudaFreeHost(C.d2h_pinned_base);
    release_pinned(C.d2h_pinned_bytes, opt);
    C.d2h_pinned_bytes = 0;
  }
  if (C.ev_h2d_begin) { cudaEventDestroy(C.ev_h2d_begin); }
  if (C.ev_h2d_end) { cudaEventDestroy(C.ev_h2d_end); }
  if (C.ev_comp_end) { cudaEventDestroy(C.ev_comp_end); }
  if (C.ev_d2h_end) { cudaEventDestroy(C.ev_d2h_end); }
  // Preserve accumulated per-stream stats (+ last_adjust) across the buffer
  // reallocation: C = StreamCtx{} would otherwise wipe the JSON stats.
  auto save_adjust = C.last_adjust;
  auto save_stats = C.stats;
  C = StreamCtx{};
  C.last_adjust = save_adjust;
  C.stats = save_stats;
}


// CUDA / nvCOMP error checkers  throw on failure so gpu_worker's catch block
// can route in-flight chunks to the rescue queue.
static void checkCuda(cudaError_t st, const char * msg)
{
  if (st != cudaSuccess)
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(st));
}

static void checkNvcomp(nvcompStatus_t st, const char * msg)
{
  if (st != nvcompSuccess)
    throw std::runtime_error(std::string(msg) + " (nvCOMP status " + std::to_string(int(st)) + ")");
}

/*----------------------------------------------------------------------
  GPU COMPRESS WORKER
  -----------------------------------------------------------------------
  One thread per GPU device.  Each thread owns one or more CUDA streams
  (StreamCtx).  The main loop has three phases:

  1. SUBMIT: Pop a batch from the task queue, upload to GPU (H2D),
     launch nvCOMP compress kernel, and record CUDA events for timing.

  2. POLL COMPLETIONS (async path): Non-blocking cudaStreamQuery checks
     if a stream has finished.  On success, reads back compressed sizes
     and data (D2H), records perf counters, and delivers results.

  3. SYNC DRAIN (synchronous path): When no new batch was submitted
     (!submitted_any)  e.g., queue is empty  we synchronously wait
     on busy streams via cudaStreamSynchronize instead of polling.
     This avoids spin-waiting when there's no new work to overlap with.

  IMPORTANT: Both completion paths (async poll and sync drain) MUST
  record to g_perf counters and per-device/per-stream stats.  If you
  add perf recording to one path, add it to the other too.  The sync
  drain path handles the majority of completions when batch sizes are
  small (N=1) because the GPU finishes before the next batch arrives.

  On failure: the catch block rescues in-flight chunks to the CPU
  rescue queue (hybrid mode) or dies (--gpu-only mode).
----------------------------------------------------------------------*/

// nvCOMP's batched zstd compressor does NOT add the per-frame content checksum
// that the CPU path emits (ZSTD_c_checksumFlag), so GPU/hybrid archives were not
// self-verifying.  We add it ourselves: zstd's content checksum is the low 32
// bits of XXH64(uncompressed_frame, 0).  ZSTD_XXH64 is exported by libzstd (it
// builds xxHash under the ZSTD_ namespace for its own checksums), so no extra
// dependency.  The checksum is computed at H2D staging (uncompressed data still
// on the host) and stitched onto the frame after D2H.
extern "C" unsigned long long ZSTD_XXH64(const void * input, size_t length,
                                         unsigned long long seed);

// Turn an nvCOMP-produced frame into a self-verifying one: set the
// Frame_Header_Descriptor's content-checksum flag (bit 2 of the byte after the
// 4-byte magic) and append the 4-byte XXH64-low32, little-endian — exactly what
// a zstd decoder (and `zstd -l`, and gzstd -t) expects.  The FrameBuf is a heap
// vector, so the append is free of the fixed-buffer overflow the GPU output
// slots would have.
static inline void gpu_frame_add_checksum(FrameVec & f, uint32_t ck) {
  if (f.size() < 5) return;                 // not a well-formed zstd frame; leave as-is
  f[4] |= 0x04;                             // Content_Checksum_flag
  f.push_back((char)(ck & 0xff));
  f.push_back((char)((ck >> 8) & 0xff));
  f.push_back((char)((ck >> 16) & 0xff));
  f.push_back((char)((ck >> 24) & 0xff));
}

static void gpu_worker(
  int device_id,
  int slot_index,   // positional index into per_dev/json_sink arrays
  Options opt,
  TaskQueue * queue,
  ResultStore * results,
  DevStats * devstats,
  StatsSink * json_sink,
  Meter * m,
  HybridSched * sched,
  std::atomic<bool> * any_gpu_failed,
  std::atomic<bool> * abort_on_failure,
  std::string * fatal_msg,
  std::atomic<bool> * gpu_started_flag,
  SharedTuneState * shared_tune,
  RateMatchState * rate_match,
  FrameThrottle * bp,
  std::atomic<int> * gpu_failures,   // terminal failures (init or mid-run), all workers
  int gpu_worker_count)              // total GPU workers spawned (for last-failure detection)
{
  (void)m; std::shared_ptr<std::vector<StreamCtx>> ctxs_ptr;
  void * vram_reserve = nullptr;
  size_t vram_reserve_bytes = 0;
  try {
    uint64_t init_t0 = g_perf ? now_ns() : 0;
    // -vv init-phase breakdown: distinguish context creation (driver-bound,
    // hard to reduce) from VRAM probing + cudaMalloc (potentially tunable).
    const uint64_t phase_t0 = now_ns();
    uint64_t probe_ns = 0, malloc_ns = 0;  // accumulated across streams
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    const size_t host_chunk_bytes = std::max<size_t>(1, opt.chunk_mib) * ONE_MIB;
    const size_t gpu_chunk = std::min(host_chunk_bytes, GPU_SUBCHUNK_MAX);
    nvcompBatchedZstdCompressOpts_t comp_opts = nvcompBatchedZstdCompressDefaultOpts; size_t max_out_chunk=0;
    checkNvcomp(nvcompBatchedZstdCompressGetMaxOutputChunkSize(gpu_chunk, comp_opts, &max_out_chunk), "nvcompBatchedZstdCompressGetMaxOutputChunkSize");
    const uint64_t ctx_done_ns = now_ns();  // cudaSetDevice forced context creation
    const size_t stream_count = std::max<size_t>(1, opt.gpu_streams);
    // --gpu-batch is per-stream (same semantics as decompress).  Each stream
    // targets up to gpu_batch_cap subchunks; VRAM safety comes from splitting
    // the memory budget across streams and clamping via binary search below.
    //
    // When the user does NOT pin --gpu-batch (gpu_batch_user_set==false), the
    // shared auto-tuner is active and tries to grow the batch.  We must
    // allocate enough buffer headroom for that growth — otherwise the pop
    // (clamped to per_stream_batch) caps the tuner at the initial value and
    // dynamic scaling is silently dead (was a long-standing bug; see
    // CHANGELOG v0.12.32).  Use HARD_BATCH_CAP as the binary-search ceiling
    // and let the VRAM check below pick the actual fit.
    size_t per_stream_cap = opt.gpu_batch_user_set
        ? std::max<size_t>(1, std::min(opt.gpu_batch_cap, HARD_BATCH_CAP))
        : AUTO_TUNE_BATCH_CEILING;
    double per_stream_frac = std::max(0.05, std::min(0.95, opt.gpu_mem_fraction / double(stream_count)));

    ctxs_ptr = std::make_shared<std::vector<StreamCtx>>(stream_count);
    auto & ctxs = *ctxs_ptr;
    for (size_t s=0; s<stream_count; ++s) {
      StreamCtx & C = ctxs[s];
      checkCuda(cudaStreamCreate(&C.stream), "cudaStreamCreate");
      // Calculate how many subchunks this stream can hold based on free VRAM.
      // nvCOMP temp workspace can be very large (e.g., 5 GiB for batch=200),
      // so we use binary search to find the largest batch that fits in VRAM.
      const uint64_t probe_t0 = now_ns();
      size_t free_b = 0, total_b = 0;
      if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b == 0) {
        C.per_stream_batch = 1;
      } else {
        // Binary search for largest batch that fits
        size_t lo = 1, hi = std::min(per_stream_cap, HARD_BATCH_CAP);
        size_t best = 1;
        while (lo <= hi) {
          size_t mid = lo + (hi - lo) / 2;
          // Estimate total VRAM for this batch size
          size_t temp_est = 0;
          try { temp_est = get_nvcomp_temp_size(mid, gpu_chunk, comp_opts, C.stream); }
          catch (...) { hi = mid - 1; continue; }
          size_t est = mid * (gpu_chunk + max_out_chunk)
                     + temp_est
                     + mid * (sizeof(void*)*2 + sizeof(size_t)*2 + sizeof(nvcompStatus_t));
          if (est <= static_cast<size_t>(free_b * per_stream_frac)) {
            best = mid;
            lo = mid + 1;
          } else {
            hi = mid - 1;
          }
        }
        C.per_stream_batch = best;
        if (best < per_stream_cap) {
          // Always report when user explicitly set batch size; otherwise only at -v
          int min_verb = opt.gpu_batch_user_set ? V_NORMAL : V_VERBOSE;
          if (opt.verbosity >= min_verb) {
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << s
               << "] VRAM-fit: batch=" << best << " (requested " << per_stream_cap
               << ", free=" << (free_b / ONE_MIB) << " MiB)";
            vlog(min_verb, opt, os.str() + "\n");
          }
        }
      }
      // Update shared tuner's VRAM ceiling (use minimum across all GPUs).
      // Use the binary search's maximum viable batch, not the chosen starting size.
      if (shared_tune) {
        // The binary search found 'best' as the largest that fits.
        // Use that as the ceiling (not C.per_stream_batch which is the default start).
        size_t vram_max = C.per_stream_batch;  // fallback
        // Re-query VRAM to estimate actual ceiling
        size_t free_b2 = 0, total_b2 = 0;
        if (cudaMemGetInfo(&free_b2, &total_b2) == cudaSuccess && free_b2 > 0) {
          // Estimate: each batch slot needs gpu_chunk + max_out_chunk + temp/batch
          size_t per_slot = gpu_chunk + max_out_chunk;
          if (per_slot > 0)
            vram_max = std::max(vram_max, static_cast<size_t>(free_b2 * 0.8) / per_slot);
        }
        size_t old_ceil = shared_tune->vram_ceiling.load();
        while (vram_max < old_ceil &&
               !shared_tune->vram_ceiling.compare_exchange_weak(old_ceil, vram_max));
      }
      probe_ns += now_ns() - probe_t0;
      const uint64_t malloc_t0 = now_ns();
      int vram_retries = 0;
      bool stream_init_failed = false;
      while (!allocate_stream_buffers(C, C.per_stream_batch, gpu_chunk, max_out_chunk, comp_opts, opt)) {
        // Free any partial allocations from the failed attempt
        free_stream_buffers_only(C, opt);
        if (C.per_stream_batch <= 1 || ++vram_retries > 10) {
          // Can't fit even batch=1 on this stream.  If we already initialized
          // one or more streams, stop adding more and run with what we have
          // (auto-decrement of --gpu-streams when VRAM is tight).  Only if
          // no streams succeeded do we skip the GPU entirely.
          stream_init_failed = true;
          if (C.stream) { cudaStreamDestroy(C.stream); C.stream = nullptr; }
          break;
        }
        C.per_stream_batch = std::max<size_t>(1, C.per_stream_batch/2);
        {
          int min_verb = opt.gpu_batch_user_set ? V_NORMAL : V_VERBOSE;
          if (opt.verbosity >= min_verb) {
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << s
               << "] VRAM insufficient, reducing batch to " << C.per_stream_batch;
            vlog(min_verb, opt, os.str() + "\n");
          }
        }
      }
      malloc_ns += now_ns() - malloc_t0;
      if (stream_init_failed) {
        // Auto-decrement stream count: keep streams [0..s) and stop here.
        if (s == 0) {
          // Couldn't fit even one stream — skip this GPU entirely.
          std::string skip_msg = "[GPU" + std::to_string(device_id)
              + "] insufficient VRAM for even 1 stream at batch=1  skipping device";
          vlog(V_ERROR, opt, skip_msg + "\n");
          *any_gpu_failed = true;
          *fatal_msg = skip_msg;
          int fails = gpu_failures
              ? gpu_failures->fetch_add(1, std::memory_order_acq_rel) + 1 : 1;
          // Wake writer and CPU workers in case they're waiting
          { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
          queue->notify_cpu_waiters();
          // Last GPU standing just failed in --gpu-only: finish on CPU
          // instead of stranding the queue (data safety over mode purity).
          if (opt.gpu_only && fails == gpu_worker_count)
            gpu_only_cpu_fallback(false, queue, results, opt, m, bp);
          return;
        }
        // At least one stream is usable — shrink ctxs and continue.
        vlog(V_DEFAULT, opt,
             "WARNING: [GPU" + std::to_string(device_id)
             + "] VRAM insufficient for " + std::to_string(stream_count)
             + " streams at batch=1; auto-reducing to " + std::to_string(s)
             + " stream" + (s == 1 ? "" : "s") + "\n");
        ctxs.resize(s);
        break;  // exit the stream-init loop
      }
      C.stats.dev_index = size_t(device_id);
      C.stats.stream_index = s;
      if (opt.verbosity >= V_DEBUG) {
        std::ostringstream os;
        os << "[GPU" << device_id << "/S" << s
           << "] subchunk=" << (gpu_chunk / ONE_MIB)
           << "MiB batch=" << C.per_stream_batch;
        vlog(V_DEBUG, opt, os.str() + "\n");
      }
    }

    bool producer_done_seen=false;
    if (g_perf) { uint64_t dt = now_ns() - init_t0; g_perf->cuda_init_sum_ns.fetch_add(dt); g_perf->cuda_init_count.fetch_add(1); uint64_t cur = g_perf->cuda_init_max_ns.load(); while (dt > cur && !g_perf->cuda_init_max_ns.compare_exchange_weak(cur, dt)); };
    if (opt.verbosity >= V_DEBUG) {
      std::ostringstream os;
      os << "[GPU" << device_id << "] init phases: ctx="
         << std::fixed << std::setprecision(0) << double(ctx_done_ns - phase_t0) / 1e6
         << "ms probe=" << double(probe_ns) / 1e6
         << "ms malloc=" << double(malloc_ns) / 1e6
         << "ms total=" << double(now_ns() - phase_t0) / 1e6 << "ms";
      vlog(V_DEBUG, opt, os.str() + "\n");
    }

    // VRAM reserve: hold enough memory to process half the batch size.
    // If another user grabs VRAM mid-run and a cudaMalloc fails, we free
    // the reserve and retry before giving up on this GPU.
    {
      size_t half_batch = std::max<size_t>(1, ctxs[0].per_stream_batch / 2);
      size_t per_frame = gpu_chunk + max_out_chunk + 4096;
      vram_reserve_bytes = half_batch * per_frame;
      if (cudaMalloc(&vram_reserve, vram_reserve_bytes) != cudaSuccess) {
        vram_reserve = nullptr;
        vram_reserve_bytes = 0;
        vlog(V_VERBOSE, opt, "[GPU" + std::to_string(device_id)
             + "] could not allocate VRAM reserve (non-fatal)\n");
      } else if (opt.verbosity >= V_DEBUG) {
        vlog(V_DEBUG, opt, "[GPU" + std::to_string(device_id)
             + "] VRAM reserve: " + std::to_string(vram_reserve_bytes / ONE_MIB) + " MiB\n");
      }
    }

    if (sched) {
      sched->set_gpu_ready(device_id);
      // Register each successfully-initialized stream so the scheduler
      // reserves enough queue depth for GPU batch demand.
      for (size_t s = 0; s < ctxs.size(); ++s)
        sched->register_gpu_stream();
    }
    double util_scale = 1.0;  // utilization scaling for batch size
    while (true) {
      bool submitted_any=false;
      // ---- Shared auto-tuner ----
      // All GPUs report throughput to SharedTuneState. Whichever worker
      // grabs the mutex first runs the tune logic for everyone.
      if (shared_tune && !shared_tune->locked.load()) {
        auto now = std::chrono::steady_clock::now();
        // Try to run tune logic (non-blocking mutex try_lock)
        if (shared_tune->window_batches.load(std::memory_order_relaxed) >= SharedTuneState::MIN_BATCHES) {
          std::unique_lock<std::mutex> lk(shared_tune->tune_mtx, std::try_to_lock);
          if (lk.owns_lock()) {
            double secs = std::chrono::duration_cast<std::chrono::duration<double>>(
                now - shared_tune->last_tune).count();
            if (secs >= SharedTuneState::TUNE_SEC) {
              shared_tune->last_tune = now;
              uint64_t bytes = shared_tune->window_bytes.exchange(0);
              uint64_t ns = shared_tune->window_ns.exchange(0);
              shared_tune->window_batches.store(0);
              double cur_thr = (ns > 0) ? double(bytes) / (double(ns)/1e9) / 1e9 : 0.0;
              size_t cur_batch = shared_tune->batch_size.load();

              auto & S = *shared_tune;
              if (S.phase == SharedTuneState::Phase::BASELINE) {
                S.best_batch = cur_batch;
                S.best_thr = cur_thr;
                S.prev_batch = cur_batch;
                S.prev_thr = cur_thr;
                // Try halving first
                size_t half = std::max<size_t>(1, cur_batch / 2);
                if (half < cur_batch) {
                  S.batch_size.store(half);
                  S.phase = SharedTuneState::Phase::HALVE;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] baseline=" << cur_batch << " (" << std::fixed
                       << std::setprecision(2) << cur_thr << " GiB/s) -> try " << half;
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                } else {
                  S.batch_size.store(std::min(cur_batch * 2, S.vram_ceiling.load()));
                  S.phase = SharedTuneState::Phase::DOUBLE;
                }
              } else if (S.phase == SharedTuneState::Phase::HALVE) {
                if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                if (cur_thr >= S.prev_thr * 0.98) {
                  // Halving helped  continue halving
                  S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                  size_t half = std::max<size_t>(1, cur_batch / 2);
                  if (half < cur_batch) { S.batch_size.store(half); }
                  else { S.batch_size.store(S.best_batch); S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                } else {
                  // Halving worse  revert to best, try doubling
                  S.prev_thr = S.best_thr; S.prev_batch = S.best_batch;
                  S.batch_size.store(S.best_batch);
                  S.phase = SharedTuneState::Phase::DOUBLE;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] halving worse, will try doubling (best=" << S.best_batch << ")";
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                }
              } else if (S.phase == SharedTuneState::Phase::DOUBLE) {
                // We're at best_batch measuring baseline before doubling, OR measuring after doubling
                if (cur_batch == S.best_batch) {
                  // At best  now try doubling
                  S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                  size_t dbl = std::min(cur_batch * 2, S.vram_ceiling.load());
                  if (dbl > cur_batch) { S.batch_size.store(dbl); }
                  else { S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                } else {
                  if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                  if (cur_thr >= S.prev_thr * 0.98) {
                    // Doubling helped  continue
                    S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                    size_t dbl = std::min(cur_batch * 2, S.vram_ceiling.load());
                    if (dbl > cur_batch) { S.batch_size.store(dbl); }
                    else { S.batch_size.store(S.best_batch); S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                  } else {
                    // Doubling worse  refine between best and current
                    if (std::abs((long)cur_batch - (long)S.best_batch) > 2) {
                      S.refine_lo = std::min(S.best_batch, cur_batch);
                      S.refine_hi = std::max(S.best_batch, cur_batch);
                      size_t mid = S.refine_lo + (S.refine_hi - S.refine_lo) / 2;
                      S.batch_size.store(mid);
                      S.phase = SharedTuneState::Phase::REFINE; S.refine_iters = 0;
                      if (opt.verbosity >= V_VERBOSE) {
                        std::ostringstream os;
                        os << "[AUTO-TUNE] refining [" << S.refine_lo << ".." << S.refine_hi
                           << "] trying " << mid;
                        vlog(V_VERBOSE, opt, os.str() + "\n");
                      }
                    } else {
                      S.batch_size.store(S.best_batch);
                      S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                      if (opt.verbosity >= V_VERBOSE) {
                        std::ostringstream os;
                        os << "[AUTO-TUNE] settled at batch=" << S.best_batch
                           << " (" << std::fixed << std::setprecision(2) << S.best_thr << " GiB/s)";
                        vlog(V_VERBOSE, opt, os.str() + "\n");
                      }
                    }
                  }
                }
              } else if (S.phase == SharedTuneState::Phase::REFINE) {
                ++S.refine_iters;
                if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                if (cur_batch < S.best_batch) S.refine_lo = cur_batch;
                else if (cur_batch > S.best_batch) S.refine_hi = cur_batch;
                if (S.refine_hi - S.refine_lo <= 2 || S.refine_iters >= SharedTuneState::MAX_REFINE_ITERS) {
                  S.batch_size.store(S.best_batch);
                  S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] refined, settled at batch=" << S.best_batch
                       << " (" << std::fixed << std::setprecision(2) << S.best_thr << " GiB/s)";
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                } else {
                  size_t mid = S.refine_lo + (S.refine_hi - S.refine_lo) / 2;
                  if (mid == cur_batch) mid++;
                  S.batch_size.store(mid);
                }
              } else {
                // SETTLED  continuous probing
                // Alternate between trying larger (+50%) and smaller (-25%)
                // to track changing data characteristics and find better batch sizes.
                ++S.settle_ticks;
                if (S.settle_ticks >= SharedTuneState::PROBE_INTERVAL) {
                  S.settle_ticks = 0;
                  // Update best throughput at current size
                  if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                  // Alternate: even ticks probe up, odd ticks probe down
                  ++S.probe_count;
                  bool probe_up = (S.probe_count % 2 == 0);
                  size_t probe;
                  if (probe_up) {
                    probe = std::min(S.best_batch + S.best_batch / 4, S.vram_ceiling.load());
                    if (probe <= S.best_batch) probe = S.best_batch + 1;
                  } else {
                    probe = std::max<size_t>(1, S.best_batch - S.best_batch / 4);
                    if (probe >= S.best_batch) probe = std::max<size_t>(1, S.best_batch - 1);
                  }
                  if (probe != S.best_batch && probe <= S.vram_ceiling.load()) {
                    S.prev_thr = cur_thr; S.prev_batch = S.best_batch;
                    S.batch_size.store(probe);
                    if (probe > S.best_batch) {
                      S.phase = SharedTuneState::Phase::DOUBLE;
                    } else {
                      S.phase = SharedTuneState::Phase::HALVE;
                    }
                    if (opt.verbosity >= V_VERBOSE) {
                      std::ostringstream os;
                      os << "[AUTO-TUNE] probe: " << S.best_batch << " -> " << probe
                         << " (" << std::fixed << std::setprecision(2) << cur_thr << " GiB/s)";
                      vlog(V_VERBOSE, opt, os.str() + "\n");
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Submit batches
      for (auto & C : ctxs) {
        if (C.busy) { continue; }
        // Fixed-share mode: GPU yields when CPU is below its target share.
        // Skip this stream's pop attempt; the outer worker loop will retry.
        // If the queue is already drained, propagate that so the worker can
        // exit instead of spinning forever on the share check.
        if (sched && !sched->should_gpu_take()) {
          if (queue->drained()) { producer_done_seen = true; continue; }
          // Adaptive tail yield (compress): CPUs are draining the rest.
          // Park on the queue CV until an event that could change the
          // decision: a pop shrinks the queue, the queue drains, or a
          // scheduler tick moves the EMAs.  No polling, no fixed sleeps.
          // Only park when every stream is idle — with a batch in flight
          // this loop must keep polling cudaStreamQuery, and the
          // completion paths below already block appropriately.  (Fixed
          // mode keeps its spin: the share check is designed to
          // oscillate per-batch.  Adaptive decompress never declines.)
          if (!sched->is_fixed_mode()) {
            bool any_busy = false;
            for (auto & X : ctxs) if (X.busy) { any_busy = true; break; }
            if (!any_busy &&
                !queue->wait_for_gpu_yield(
                    [&](const TaskQueue::QueueState & qs) {
                      return sched->should_gpu_take_at(qs.depth);
                    }))
              producer_done_seen = true;
          }
          continue;
        }
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        // Greedy pop: wait for a full batch (per_stream_batch) or producer done.
        // This maximizes GPU kernel efficiency by processing many chunks per launch.
        // Use shared batch size from auto-tuner (or per-stream if locked)
        size_t pop_n = (shared_tune && !shared_tune->locked.load())
                     ? shared_tune->batch_size.load(std::memory_order_relaxed)
                     : C.per_stream_batch;
        pop_n = std::min(pop_n, C.per_stream_batch);  // can't exceed allocated buffer
        // Keep scheduler's queue floor in sync with current batch size
        if (sched) sched->set_gpu_batch_size(pop_n);
        // Apply utilization scaling (updated after each batch completion)
        pop_n = std::max<size_t>(1, (size_t)(pop_n * util_scale));
        // Acquire frame permits BEFORE gpu_wants_data to avoid deadlock:
        // if we signal "GPU wants data" first, CPUs yield, but we then
        // block on throttle — nobody pops.
        if (bp) bp->acquire((int)pop_n);
        // Signal scheduler: GPU stream wants data (blocks CPU workers)
        if (sched) sched->gpu_wants_data();
        // Pop-batch minimum: when the user pinned --gpu-batch (locked), wait
        // for the full batch — they asked for it.  When auto-tuning, take
        // whatever the queue can give (min_n=1) so multiple GPUs don't
        // serialize behind a single producer.  pop_batch_greedy still returns
        // early at end-of-queue regardless (no deadlock).
        const bool locked_batch = shared_tune && shared_tune->locked.load();
        size_t comp_min_batch = locked_batch ? pop_n : 1;
        // Deadlock guard: a locked full-batch wait blocks while HOLDING this
        // batch's throttle permits.  Under a bounded queue (pooled reader),
        // the streams' aggregate locked demand can exceed the depth the
        // queue can ever present; the sleeping streams then sequester
        // enough permits to exhaust the throttle, CPUs can't pop the frames
        // the writer needs, and the run wedges (measured on the 8-GPU
        // server: --gpu-streams=16 --gpu-batch=64 → 128 streams × VRAM-fit
        // ~20 ≈ 2560 held permits, hang at 45%).  When aggregate locked
        // demand exceeds half the queue's depth ceiling, relax to min_n=1 —
        // the user's batch stays as the pop CAP.
        if (locked_batch && sched) {
          const size_t dcap = sched->queue_depth_cap();
          const size_t streams_n = (size_t)std::max(1, sched->active_gpu_streams());
          if (dcap > 0 && streams_n * pop_n * 2 > dcap) {
            comp_min_batch = 1;
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
              vlog(V_DEFAULT, opt,
                   "[GPU] locked --gpu-batch × streams exceeds queue capacity; "
                   "relaxing full-batch waits to avoid deadlock (batch stays the cap)\n");
          }
        }
        if (!queue->pop_batch_greedy(pop_n, C.batch, comp_min_batch)) {
          if (sched) sched->gpu_got_data();
          if (bp) bp->release((int)pop_n);  // release unused permits
          if (g_perf) { g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0); g_perf->queue_wait_count.fetch_add(1); }
          producer_done_seen = true; continue;
        }
        // Signal scheduler: GPU got its data (unblocks CPU workers)
        if (sched) sched->gpu_got_data();
        if (C.batch.empty()) {
          if (bp) bp->release((int)pop_n);  // release unused permits
          if (g_perf) { g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0); g_perf->queue_wait_count.fetch_add(1); }
          continue;
        }
        // Release excess permits if we got fewer frames than requested
        if (bp && (int)C.batch.size() < (int)pop_n)
          bp->release((int)pop_n - (int)C.batch.size());
        if (gpu_started_flag) { gpu_started_flag->store(true, std::memory_order_release); }
        if (sched) { sched->mark_gpu_take(C.batch.size()); }
        if (g_perf) g_perf->sched_gpu_tasks.fetch_add(C.batch.size());
        C.filled = C.batch.size();
        C.delivered = 0;

        if (m) {
          // mmap views are counted here (their reader doesn't); --direct-read
          // views (direct_buf >= 0) were already counted by the O_DIRECT
          // reader — skip them (mirrors cpu_worker).
          for (size_t i = 0; i < C.filled; ++i)
            if (C.batch[i].view_ptr && C.batch[i].direct_buf < 0)
              m->read_bytes.fetch_add(C.batch[i].len(), std::memory_order_relaxed);
        }

        // -vv: print take line
        if (opt.verbosity >= V_DEBUG) {
          size_t seq_lo = C.batch.front().seq;
          size_t seq_hi = C.batch.back().seq;
          uint64_t tin = 0;
          for (size_t i = 0; i < C.filled; ++i)
            tin += C.batch[i].len();
          char tin_s[32];
          human_bytes(double(tin), tin_s, sizeof(tin_s));
          std::ostringstream os;
          os << "[GPU" << device_id << "/S" << C.stats.stream_index
             << "] take batch=" << C.filled
             << " seq=[" << seq_lo << ".." << seq_hi << "]"
             << " in=" << tin_s;
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        cudaEventRecord(C.ev_h2d_begin, C.stream);
        // Upload each subchunk to its slot in the device input buffer
        for (size_t i = 0; i < C.filled; ++i) {
          const Task & t = C.batch[i];
          // Guard the slot bound: a Task larger than gpu_chunk would overflow
          // its device slot (illegal memory access).  The driver clamps the
          // host chunk to GPU_SUBCHUNK_MAX so this can't happen — fail loudly
          // rather than silently corrupt VRAM if that invariant ever breaks.
          if (t.len() > C.gpu_chunk)
            die("internal error: input chunk " + std::to_string(t.len())
                + " exceeds GPU subchunk slot " + std::to_string(C.gpu_chunk));
          void * d_dst = static_cast<char*>(C.d_in_base) + i * C.gpu_chunk;

          if (C.h2d_pinned_base) {
            void * h_src = static_cast<char*>(C.h2d_pinned_base) + i * C.h_io_slot_bytes;
            std::memcpy(h_src, t.ptr(), t.len());
            checkCuda(cudaMemcpyAsync(d_dst, h_src, t.len(),
                                      cudaMemcpyHostToDevice, C.stream),
                      "cudaMemcpyAsync(H2D pinned)");
          } else {
            checkCuda(cudaMemcpyAsync(d_dst, t.ptr(), t.len(),
                                      cudaMemcpyHostToDevice, C.stream),
                      "cudaMemcpyAsync(H2D)");
          }
          C.h_in_sizes[i] = t.len();
          // Compute the zstd content checksum now, while the uncompressed chunk
          // is still on the host (it may be released right after H2D below).
          C.h_checksums[i] = (uint32_t)ZSTD_XXH64(t.ptr(), t.len(), 0);
        }
        checkCuda(cudaMemcpyAsync(C.d_in_sizes, C.h_in_sizes.data(), sizeof(size_t)*C.filled, cudaMemcpyHostToDevice, C.stream), "cudaMemcpyAsync(d_in_sizes)");
        cudaEventRecord(C.ev_h2d_end, C.stream);

        // Release host-side input data  it's on the GPU now (and the per-frame
        // checksum was already computed above).  cudaMemcpyAsync with pageable
        // memory is host-synchronous, so the data is fully copied before the
        // function returns.  We no longer keep it alive for rescue (a GPU fault
        // now aborts and rebuilds from the original input), so release always —
        // freeing pooled-reader slots / mmap views as early as possible.
        for (size_t i = 0; i < C.filled; ++i)
          C.batch[i].release_input();

        checkNvcomp(nvcompBatchedZstdCompressAsync(
          (const void * const *)C.d_in_ptrs,
          (const size_t *)C.d_in_sizes,
          C.gpu_chunk,
          C.filled,
          C.d_temp,
          C.temp_bytes_used,
          (void * const *)C.d_out_ptrs,
          C.d_comp_sizes,
          nvcompBatchedZstdCompressDefaultOpts,
          C.d_stats,
          C.stream), "nvcompBatchedZstdCompressAsync");
        cudaEventRecord(C.ev_comp_end, C.stream);
        C.busy = true; submitted_any = true;
      }

      // ---- COMPLETION PATH 1: Async polling ----
      // Non-blocking check if GPU stream has finished.  This path runs when
      // we just submitted a new batch (submitted_any=true) and are checking
      // if previous batches completed while we were submitting.
      // NOTE: Must record to g_perf here  see also sync drain path below.
      for (auto & C : ctxs) {
        if (!C.busy) { continue; }
        cudaError_t q = cudaStreamQuery(C.stream);
        if (q == cudaSuccess) {
          uint64_t d2h_t0 = g_perf ? now_ns() : 0;
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes)");
          // Measure per-phase timing from CUDA events
          float h2d_ms = 0, comp_ms = 0;
          cudaEventElapsedTime(&h2d_ms,  C.ev_h2d_begin, C.ev_h2d_end);
          cudaEventElapsedTime(&comp_ms, C.ev_h2d_end,   C.ev_comp_end);

          if (g_perf) {
            g_perf->h2d_ns.fetch_add(uint64_t(h2d_ms * 1e6));
            g_perf->h2d_count.fetch_add(1);
            g_perf->kernel_ns.fetch_add(uint64_t(comp_ms * 1e6));
            g_perf->kernel_count.fetch_add(1);
          }

          uint64_t in_sum = 0, out_sum = 0;
          for (size_t i=0;i<C.filled;++i) {
            if (C.h_stats[i] != nvcompSuccess) throw std::runtime_error("nvCOMP per-chunk status != nvcompSuccess");
            const size_t csz = C.h_comp_sizes[i]; out_sum += csz; in_sum += C.h_in_sizes[i];
            auto h_out = C.acquire_out_buf(std::max<size_t>(2, C.per_stream_batch) * 2, bp);
            const void * d_src = static_cast<char*>(C.d_out_base)
                                 + i * C.max_out_chunk;
            if (C.h2d_pinned_base) {
              // Reuse the H2D pinned slot for D2H — input was already
              // consumed by the GPU, slot is free until next batch's pop.
              void * pin_slot = static_cast<char*>(C.h2d_pinned_base)
                                + i * C.h_io_slot_bytes;
              checkCuda(cudaMemcpy(pin_slot, d_src, csz,
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(D2H pinned shared slot)");
              // assign() copies straight from the pinned slot — no resize() zero-fill
              // (the copy would immediately overwrite it anyway).
              h_out->assign(static_cast<char*>(pin_slot),
                            static_cast<char*>(pin_slot) + csz);
            } else {
              h_out->resize(csz);  // direct D2H needs the dst pre-sized
              checkCuda(cudaMemcpy(h_out->data(), d_src, csz,
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(D2H exact)");
            }
            gpu_frame_add_checksum(*h_out, C.h_checksums[i]);  // make the frame self-verifying
            results->push_to_slot(slot_index, C.batch[i].seq, std::move(h_out));
            C.delivered = i + 1;  // a mid-loop throw rescues only [delivered..)
            // Test hook: deterministically simulate a GPU fault after N frames
            // (real illegal-access faults are intermittent) to exercise the
            // CPU-only rebuild path.  Disabled unless GZSTD_DEBUG_FAIL_GPU_AFTER set.
            if (g_debug_fail_gpu_after >= 0) {
              static std::atomic<int64_t> dbg_delivered{0};
              if (dbg_delivered.fetch_add(1) + 1 > g_debug_fail_gpu_after)
                throw std::runtime_error("simulated GPU fault (GZSTD_DEBUG_FAIL_GPU_AFTER)");
            }
          }
          double d2h_ms = double(now_ns() - d2h_t0) / 1e6;
          double tot_ms = double(h2d_ms) + double(comp_ms) + d2h_ms;
          if (g_perf) {
            g_perf->d2h_ns.fetch_add(uint64_t(d2h_ms * 1e6));
            g_perf->d2h_bytes.fetch_add(out_sum);
            g_perf->d2h_count.fetch_add(1);
            g_perf->h2d_bytes.fetch_add(in_sum);
            g_perf->gpu_batch_ns.fetch_add(uint64_t(tot_ms * 1e6));
            g_perf->gpu_batch_count.fetch_add(1);
          }
          // Accumulate into per-device stats
          {
            std::lock_guard<std::mutex> lk(devstats->m);
            devstats->h2d_ms  += h2d_ms;
            devstats->comp_ms += comp_ms;
            devstats->d2h_ms  += d2h_ms;
            devstats->total_ms += tot_ms;
            devstats->batches += 1;
          }
          #ifdef HAVE_NVCOMP
          if (sched) sched->add_gpu_bytes(in_sum);
          #endif
          // Accumulate into per-stream stats
          C.stats.h2d_ms   += h2d_ms;
          C.stats.comp_ms  += comp_ms;
          C.stats.d2h_ms   += d2h_ms;
          C.stats.total_ms += tot_ms;
          C.stats.in_bytes += in_sum;
          C.stats.out_bytes += out_sum;
          C.stats.batches  += 1;
          C.stats.chunks   += C.filled;

          // Report to shared auto-tuner (both completion paths)
          if (shared_tune && !shared_tune->locked.load()) {
            shared_tune->window_bytes.fetch_add(in_sum, std::memory_order_relaxed);
            shared_tune->window_ns.fetch_add(uint64_t(tot_ms * 1e6), std::memory_order_relaxed);
            shared_tune->window_batches.fetch_add(1, std::memory_order_relaxed);
          }
          if (rate_match) {
            rate_match->report_gpu(in_sum, uint64_t(tot_ms * 1e6));
            rate_match->reset_cycle();
            size_t avg_frame = (C.filled > 0) ? in_sum / C.filled : 16 * 1024 * 1024;
            size_t batch_sz = shared_tune ? shared_tune->batch_size.load() : C.filled;
            rate_match->update(batch_sz, avg_frame);
          }

          // -vv: batch completion line
          if (opt.verbosity >= V_DEBUG) {
            char in_s[32], out_s[32];
            human_bytes(double(in_sum), in_s, sizeof(in_s));
            human_bytes(double(out_sum), out_s, sizeof(out_s));
            double thr_gib = (tot_ms > 0.0)
                             ? double(in_sum) / (tot_ms / 1000.0) / 1e9 : 0.0;
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << C.stats.stream_index
               << "] done batch=" << C.filled
               << " in=" << in_s << " out=" << out_s
               << " h2d=" << std::fixed << std::setprecision(2) << h2d_ms
               << "ms comp=" << comp_ms
               << "ms d2h=" << d2h_ms
               << "ms tot=" << tot_ms
               << "ms thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
            vlog(V_DEBUG, opt, os.str() + "\n");
          }

          // Per-chunk trace at -vvv (compress async-poll path).
          if (opt.verbosity >= V_TRACE) {
            for (size_t i = 0; i < C.filled; ++i) {
              char cs[32], ds[32];
              human_bytes(double(C.batch[i].len()), ds, sizeof(ds));
              human_bytes(double(C.h_comp_sizes[i]), cs, sizeof(cs));
              std::ostringstream os;
              os << "[GPU" << device_id << "/S" << C.stats.stream_index
                 << "] chunk seq=" << C.batch[i].seq
                 << " in=" << ds << " out=" << cs;
              vlog(V_TRACE, opt, os.str() + "\n");
            }
          }

          C.busy = false;
          C.filled = 0;
          C.batch.clear();
          results->cv.notify_one();  // wake writer for batch
#ifdef HAVE_NVML
          {
            nvmlDevice_t dev;
            nvmlUtilization_t util;
            if (nvmlDeviceGetHandleByIndex(device_id, &dev) == NVML_SUCCESS &&
                nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
              util_scale = std::max(0.05, (100.0 - util.gpu) / 100.0);
            }
          }
#endif
        } else if (q != cudaErrorNotReady) { checkCuda(q, "cudaStreamQuery"); }
      }

      // If producer is done and all streams are idle, this device is finished
      if (producer_done_seen) {
        bool all_idle = true;
        for (auto & C : ctxs) {
          if (C.busy) { all_idle = false; break; }
        }
        if (all_idle) break;
      }
      // ---- COMPLETION PATH 2: Synchronous drain ----
      // When no new batch was submitted (queue empty or producer done),
      // block on busy streams instead of spin-polling.  This is the
      // primary completion path when batch sizes are small (N=1) or the
      // queue drains faster than GPU processes.
      // NOTE: Must record to g_perf here  see also async poll path above.
      //       If you add perf instrumentation to one path, ADD IT TO BOTH.
      if (!submitted_any) {
        bool blocked=false;
        for (auto & C: ctxs) {
          if (!C.busy) { continue; }
          checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize");
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses sync)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes sync)");
          // Measure per-phase timing (synchronous path)
          float h2d_ms = 0, comp_ms = 0;
          cudaEventElapsedTime(&h2d_ms,  C.ev_h2d_begin, C.ev_h2d_end);
          cudaEventElapsedTime(&comp_ms, C.ev_h2d_end,   C.ev_comp_end);
          if (g_perf) {
            g_perf->h2d_ns.fetch_add(uint64_t(h2d_ms * 1e6));
            g_perf->h2d_count.fetch_add(1);
            g_perf->kernel_ns.fetch_add(uint64_t(comp_ms * 1e6));
            g_perf->kernel_count.fetch_add(1);
          }
          uint64_t in_sum = 0, out_sum = 0;
          for (size_t i = 0; i < C.filled; ++i) {
            in_sum  += C.h_in_sizes[i];
            out_sum += C.h_comp_sizes[i];
          }
          uint64_t d2h_t0 = now_ns();
          for (size_t i = 0; i < C.filled; ++i) {
            // Per-chunk status check — the async-poll path always had this,
            // the sync drain was missing it: a failed chunk's comp_size is
            // garbage, so delivering it silently corrupted output (v0.13.54).
            if (C.h_stats[i] != nvcompSuccess)
              throw std::runtime_error("nvCOMP per-chunk status != nvcompSuccess");
            const size_t csz = C.h_comp_sizes[i];
            auto h_out = C.acquire_out_buf(std::max<size_t>(2, C.per_stream_batch) * 2, bp);
            const void * d_src = static_cast<char*>(C.d_out_base) + i * C.max_out_chunk;
            if (C.h2d_pinned_base) {
              // Reuse the H2D pinned slot for D2H (sync drain path).
              void * pin_slot = static_cast<char*>(C.h2d_pinned_base)
                                + i * C.h_io_slot_bytes;
              checkCuda(cudaMemcpy(pin_slot, d_src, csz, cudaMemcpyDeviceToHost),
                        "cudaMemcpy(D2H pinned shared slot sync)");
              // assign() copies straight from the pinned slot — no resize() zero-fill.
              h_out->assign(static_cast<char*>(pin_slot),
                            static_cast<char*>(pin_slot) + csz);
            } else {
              h_out->resize(csz);  // direct D2H needs the dst pre-sized
              checkCuda(cudaMemcpy(h_out->data(), d_src, csz, cudaMemcpyDeviceToHost),
                        "cudaMemcpy(D2H exact sync)");
            }
            gpu_frame_add_checksum(*h_out, C.h_checksums[i]);  // make the frame self-verifying
            results->push_to_slot(slot_index, C.batch[i].seq, std::move(h_out));
            C.delivered = i + 1;  // a mid-loop throw rescues only [delivered..)
            // Test hook: deterministically simulate a GPU fault after N frames
            // (real illegal-access faults are intermittent) to exercise the
            // CPU-only rebuild path.  Disabled unless GZSTD_DEBUG_FAIL_GPU_AFTER set.
            if (g_debug_fail_gpu_after >= 0) {
              static std::atomic<int64_t> dbg_delivered{0};
              if (dbg_delivered.fetch_add(1) + 1 > g_debug_fail_gpu_after)
                throw std::runtime_error("simulated GPU fault (GZSTD_DEBUG_FAIL_GPU_AFTER)");
            }
          }
          double d2h_ms = double(now_ns() - d2h_t0) / 1e6;
          if (g_perf) {
            g_perf->d2h_ns.fetch_add(uint64_t(d2h_ms * 1e6));
          }
          double tot_ms = double(h2d_ms) + double(comp_ms) + d2h_ms;
          {
            std::lock_guard<std::mutex> lk(devstats->m);
            devstats->h2d_ms  += h2d_ms;
            devstats->comp_ms += comp_ms;
            devstats->d2h_ms  += d2h_ms;
            devstats->total_ms += tot_ms;
            devstats->batches += 1;
          }
          #ifdef HAVE_NVCOMP
          if (sched) sched->add_gpu_bytes(in_sum);
          #endif
          if (g_perf) {
            g_perf->d2h_bytes.fetch_add(out_sum);
            g_perf->d2h_count.fetch_add(1);
            g_perf->h2d_bytes.fetch_add(in_sum);
            g_perf->gpu_batch_ns.fetch_add(uint64_t(tot_ms * 1e6));
            g_perf->gpu_batch_count.fetch_add(1);
          }
          // Accumulate per-stream stats (synchronous path)
          C.stats.h2d_ms   += h2d_ms;
          C.stats.comp_ms  += comp_ms;
          C.stats.d2h_ms   += d2h_ms;
          C.stats.total_ms += tot_ms;
          C.stats.in_bytes += in_sum;
          C.stats.out_bytes += out_sum;
          C.stats.batches  += 1;
          C.stats.chunks   += C.filled;

          // Report to shared auto-tuner (sync path)
          if (shared_tune && !shared_tune->locked.load()) {
            shared_tune->window_bytes.fetch_add(in_sum, std::memory_order_relaxed);
            shared_tune->window_ns.fetch_add(uint64_t(tot_ms * 1e6), std::memory_order_relaxed);
            shared_tune->window_batches.fetch_add(1, std::memory_order_relaxed);
          }
          if (rate_match) {
            rate_match->report_gpu(in_sum, uint64_t(tot_ms * 1e6));
            rate_match->reset_cycle();
            size_t avg_frame = (C.filled > 0) ? in_sum / C.filled : 16 * 1024 * 1024;
            size_t batch_sz = shared_tune ? shared_tune->batch_size.load() : C.filled;
            rate_match->update(batch_sz, avg_frame);
          }

          if (opt.verbosity >= V_DEBUG) {
            char in_s[32], out_s[32];
            human_bytes(double(in_sum), in_s, sizeof(in_s));
            human_bytes(double(out_sum), out_s, sizeof(out_s));
            double thr_gib = (tot_ms > 0.0)
                             ? double(in_sum) / (tot_ms / 1000.0) / 1e9 : 0.0;
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << C.stats.stream_index
               << "] done batch=" << C.filled
               << " in=" << in_s << " out=" << out_s
               << " h2d=" << std::fixed << std::setprecision(2) << h2d_ms
               << "ms comp=" << comp_ms
               << "ms d2h=" << d2h_ms
               << "ms tot=" << tot_ms
               << "ms thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
            vlog(V_DEBUG, opt, os.str() + "\n");
          }

          // Per-chunk trace at -vvv (compress sync-drain path).
          if (opt.verbosity >= V_TRACE) {
            for (size_t i = 0; i < C.filled; ++i) {
              char cs[32], ds[32];
              human_bytes(double(C.batch[i].len()), ds, sizeof(ds));
              human_bytes(double(C.h_comp_sizes[i]), cs, sizeof(cs));
              std::ostringstream os;
              os << "[GPU" << device_id << "/S" << C.stats.stream_index
                 << "] chunk seq=" << C.batch[i].seq
                 << " in=" << ds << " out=" << cs;
              vlog(V_TRACE, opt, os.str() + "\n");
            }
          }

          C.busy = false;
          C.filled = 0;
          C.batch.clear();
          results->cv.notify_one();  // wake writer for batch
#ifdef HAVE_NVML
          {
            nvmlDevice_t dev;
            nvmlUtilization_t util;
            if (nvmlDeviceGetHandleByIndex(device_id, &dev) == NVML_SUCCESS &&
                nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
              util_scale = std::max(0.05, (100.0 - util.gpu) / 100.0);
            }
          }
#endif
          blocked = true;
          break;
        }
        if (!blocked)
          std::this_thread::yield();
      }
    }

    // Export per-stream stats for JSON output
    if (json_sink) {
      std::lock_guard<std::mutex> lk(json_sink->m);
      auto & vec = json_sink->per_dev[size_t(slot_index)];
      for (size_t s = 0; s < ctxs.size(); ++s)
        vec.push_back(ctxs[s].stats);
    }
    if (opt.verbosity >= V_DEBUG) {
      for (auto & C : ctxs) {
        double thr_gib = (C.stats.total_ms>0.0)? double(C.stats.in_bytes)/(C.stats.total_ms/1000.0)/1e9 : 0.0;
        std::ostringstream os;
        os << "[GPU"<<device_id<<"/S"<<C.stats.stream_index<<"] total batches="<<C.stats.batches
           << " chunks="<<C.stats.chunks
           << " in="<<C.stats.in_bytes<<"B out="<<C.stats.out_bytes<<"B"
           << " time="<<std::fixed<<std::setprecision(2)<<C.stats.total_ms<<"ms"
           << " thr="<<std::fixed<<std::setprecision(2)<<thr_gib<<" GiB/s";
        vlog(V_DEBUG, opt, os.str() + "\n");
      }
    }

    // Free all GPU resources for this device
    if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
    for (auto & C : ctxs) {
      free_stream_buffers_only(C, opt);
      if (C.stream) cudaStreamDestroy(C.stream);
      C.stream = nullptr;
    }

    // Deregister streams so queue floor drops and CPU workers aren't blocked
    if (sched) {
      for (size_t s = 0; s < ctxs.size(); ++s)
        sched->unregister_gpu_stream();
    }

    // Wake all CPU workers: with notify_one in gpu_got_data(), CPUs that
    // were sleeping when this GPU exited might never get another wakeup.
    // This ensures all stragglers see the drain condition and exit.
    queue->notify_cpu_waiters();
  }
  catch (const std::exception & e) {
    // GPU fault — a faulted GPU is an unreliable narrator: we cannot trust any
    // frame it reported "done".  ABORT the whole compress pass and let the driver
    // rebuild CPU-only from the (still-present) input.  There is no point
    // finishing the doomed output, so we do NOT recompress the in-flight tail on
    // the CPU (that work would only be discarded) — the old rescue queue /
    // re-enqueue / gpu_only_cpu_fallback machinery was deleted.  We just raise
    // g_gpu_aborted (the producer stops reading, CPU/GPU workers exit, the writer
    // bails its in-order wait without finalizing), recycle inputs, and free the
    // GPU.  See g_gpu_aborted.
    *any_gpu_failed = true;
    g_gpu_failed_restart.store(true);
    g_gpu_aborted.store(true);
    *fatal_msg = std::string("[GPU") + std::to_string(device_id) + "] " + e.what();
    (void)gpu_failures; (void)gpu_worker_count;   // no longer needed on the abort path

    try {
      if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
      if (auto sp = std::weak_ptr<std::vector<StreamCtx>>(ctxs_ptr).lock()) {
        for (auto & C : *sp) {
          // Recycle any popped-but-undelivered (or mid-delivery) inputs so pooled
          // reader slots / mmap views don't leak; the output is discarded either
          // way, so nothing here is recompressed.
          for (auto & t : C.batch) t.release_input();
          C.batch.clear();
          free_stream_buffers_only(C, opt);
          if (C.stream) cudaStreamDestroy(C.stream);
        }
      }
    } catch (...) {}

    // Deregister streams so the queue floor drops and CPU workers aren't blocked.
    if (sched && ctxs_ptr) {
      for (size_t s = 0; s < ctxs_ptr->size(); ++s)
        sched->unregister_gpu_stream();
    }

    // Wake everything so it sees the abort and tears down cleanly: the writer's
    // in-order wait, CPU workers blocked popping the queue, the producer blocked
    // pushing a full bounded queue, workers blocked on the throttle, and any
    // rescue waiters (hybrid).  set_done() on the queue wakes both poppers and the
    // pusher; the workers then see g_gpu_aborted and exit without draining.
    { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
    queue->set_done();
    if (bp) bp->set_done();
  }
}

// Warm up CUDA primary contexts for the given devices IN PARALLEL.
// Each context init is ~0.6-1s on a datacenter GPU; doing them on one
// thread each overlaps the cost (~1s total for 8 vs ~5s serial).  The
// per-device primary context created here is reused by the GPU worker
// threads when they cudaSetDevice() later, so they start instantly.
//
// Only called for fixed-share mode (--cpu-share): there the user has
// asked for a precise CPU/GPU split, and a small input must not be
// drained entirely to CPU before the GPU finishes booting.  Adaptive
// mode skips this — deferring context creation to the worker threads
// gives the fastest possible startup (no progress-meter stall), and
// the GPU naturally catches up on any non-trivial input.
static void warm_gpu_contexts(const std::vector<int> & ids)
{
  std::vector<std::thread> warm;
  warm.reserve(ids.size());
  for (int id : ids) {
    warm.emplace_back([id]{
      if (cudaSetDevice(id) == cudaSuccess)
        cudaFree(nullptr);  // forces primary-context creation
    });
  }
  for (auto & t : warm) t.join();
}

// Select the best N GPUs by combining compute utilization and free VRAM.
// Lower utilization is better; ties broken by more free memory.
//
// When NVML is available and we want fewer GPUs than total, we first
// rank all devices via NVML (no CUDA contexts needed), then only call
// cudaSetDevice on the winners.  This avoids creating throwaway CUDA
// contexts on devices we won't use (~200ms each).
static std::vector<int> select_best_gpus(int total_devices, int want,
                                          const Options & opt)
{
  struct DevInfo {
    int id;               // CUDA device index
    size_t free_bytes;
    unsigned gpu_util;    // 0-100%, lower is better
    bool has_util;
  };

  want = std::min(want, total_devices);

  // Using every device — there is nothing to rank, so skip the probe.
  // The all-devices CUDA fallback below calls cudaSetDevice + cudaMemGetInfo
  // per device, which forces serial CUDA context creation on the main
  // thread (~0.6-1s per datacenter GPU → ~5s for 8 of them) BEFORE the
  // reader, progress meter, and worker pool start.  Returning the trivial
  // [0..N) list lets the GPU worker threads create their contexts in
  // parallel at startup instead, overlapping with the reader and (in
  // hybrid) the CPU pool.  v0.13.11.
  if (want >= total_devices) {
    std::vector<int> result;
    result.reserve(total_devices);
    for (int d = 0; d < total_devices; ++d) result.push_back(d);
    return result;
  }

#ifdef HAVE_NVML
  // Fast path: use NVML to rank without touching CUDA, then only probe winners
  if (want < total_devices) {
    bool nvml_ok = (nvmlInit_v2() == NVML_SUCCESS);
    if (nvml_ok) {
      // Build CUDA device index -> NVML info mapping.  cudaGetDeviceProperties
      // is lightweight  reads cached driver data, no context creation.
      struct NvmlInfo {
        int cuda_id;
        unsigned gpu_util;      // raw utilization 0-100
        unsigned score;         // effective score (util + NUMA penalty)
        size_t free_bytes;
        int numa_node;          // derived from CPU affinity, -1 if unknown
        bool has_data;
      };
      std::vector<NvmlInfo> infos(total_devices);

      for (int d = 0; d < total_devices; ++d) {
        infos[d] = {d, 100, 100, 0, -1, false};  // default: assume busy
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
          char pci_bus_id[32];
          snprintf(pci_bus_id, sizeof(pci_bus_id), "%08x:%02x:%02x.0",
                   prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
          nvmlDevice_t nvml_dev;
          if (nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_dev) == NVML_SUCCESS) {
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(nvml_dev, &util) == NVML_SUCCESS)
              infos[d].gpu_util = util.gpu;
            nvmlMemory_t mem;
            if (nvmlDeviceGetMemoryInfo(nvml_dev, &mem) == NVML_SUCCESS)
              infos[d].free_bytes = mem.free;
            // Get NUMA node via CPU affinity  first set bit determines node
            unsigned long cpuSet = 0;
            if (nvmlDeviceGetCpuAffinity(nvml_dev, 1, &cpuSet) == NVML_SUCCESS && cpuSet != 0) {
              // Find first set bit to determine which CPU group
              int first_cpu = __builtin_ctzl(cpuSet);
              // Simple heuristic: CPUs 0-63 = NUMA 0, 64-127 = NUMA 1, etc.
              // Works for 2-socket systems; for more sockets use first_cpu / (total_cpus / num_nodes)
              infos[d].numa_node = first_cpu / 64;
            }
            infos[d].has_data = true;
          }
        }
        infos[d].score = infos[d].gpu_util;
      }
      nvmlShutdown();

      // NUMA penalty: if any GPU on a NUMA node is busy (util > 10%),
      // add a penalty to all GPUs on that node to prefer the quieter node.
      // This avoids PCIe root complex contention from co-located busy GPUs.
      {
        // Find max utilization per NUMA node
        std::unordered_map<int, unsigned> numa_max_util;
        for (auto & info : infos) {
          if (info.numa_node >= 0 && info.has_data) {
            auto it = numa_max_util.find(info.numa_node);
            if (it == numa_max_util.end() || info.gpu_util > it->second)
              numa_max_util[info.numa_node] = info.gpu_util;
          }
        }
        // Apply penalty: half the max util on that node, capped so we don't
        // exceed 100. This makes a 0% GPU on a node with a 30% neighbor
        // score as 15%, losing to a 0% GPU on a fully idle node.
        for (auto & info : infos) {
          if (info.numa_node >= 0) {
            auto it = numa_max_util.find(info.numa_node);
            if (it != numa_max_util.end()) {
              unsigned neighbor_busy = it->second;
              // Don't penalize the busy GPU itself more (it already has high util)
              if (info.gpu_util < neighbor_busy) {
                unsigned penalty = neighbor_busy / 2;
                info.score = std::min(100u, info.gpu_util + penalty);
              }
            }
          }
        }
      }

      // Sort by effective score ascending, then free VRAM descending
      std::sort(infos.begin(), infos.end(),
                [](const NvmlInfo & a, const NvmlInfo & b) {
                  if (a.score != b.score) return a.score < b.score;
                  return a.free_bytes > b.free_bytes;
                });

      std::vector<int> result;
      for (int i = 0; i < want; ++i)
        result.push_back(infos[i].cuda_id);

      if (opt.verbosity >= V_VERBOSE && total_devices > 1) {
        std::ostringstream os;
        os << "[GPU] " << want << " device" << (want > 1 ? "s" : "") << " active";
        for (int i = 0; i < want; ++i) {
          os << (i ? ", " : ": ") << "GPU" << infos[i].cuda_id;
          os << " " << infos[i].gpu_util << "%";
          os << " (" << (infos[i].free_bytes / (1024*1024)) << " MiB)";
        }
        vlog(V_VERBOSE, opt, os.str() + "\n");
      }
      return result;
    }
  }
#endif

  // Fallback / all-devices path: use CUDA + optional NVML
  std::vector<DevInfo> devs;
  devs.reserve(total_devices);

#ifdef HAVE_NVML
  bool nvml_ok = (nvmlInit_v2() == NVML_SUCCESS);
#else
  bool nvml_ok = false;
#endif

  for (int d = 0; d < total_devices; ++d) {
    DevInfo info{d, 0, 0, false};
    if (cudaSetDevice(d) == cudaSuccess) {
      size_t free_b = 0, total_b = 0;
      if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
        info.free_bytes = free_b;

#ifdef HAVE_NVML
      if (nvml_ok) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
          char pci_bus_id[32];
          snprintf(pci_bus_id, sizeof(pci_bus_id), "%08x:%02x:%02x.0",
                   prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
          nvmlDevice_t nvml_dev;
          if (nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_dev) == NVML_SUCCESS) {
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(nvml_dev, &util) == NVML_SUCCESS) {
              info.gpu_util = util.gpu;
              info.has_util = true;
            }
          }
        }
      }
#endif
    }
    devs.push_back(info);
  }

#ifdef HAVE_NVML
  if (nvml_ok) nvmlShutdown();
#endif

  std::sort(devs.begin(), devs.end(),
            [](const DevInfo & a, const DevInfo & b) {
              if (a.has_util && b.has_util && a.gpu_util != b.gpu_util)
                return a.gpu_util < b.gpu_util;
              return a.free_bytes > b.free_bytes;
            });

  int n = std::min(want, (int)devs.size());
  std::vector<int> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i)
    result.push_back(devs[i].id);

  if (opt.verbosity >= V_VERBOSE && total_devices > 1) {
    std::ostringstream os;
    os << "[GPU] " << n << " device" << (n > 1 ? "s" : "") << " active";
    for (int i = 0; i < n; ++i) {
      os << (i ? ", " : ": ") << "GPU" << devs[i].id;
      if (devs[i].has_util) os << " " << devs[i].gpu_util << "%";
      os << " (" << (devs[i].free_bytes / (1024*1024)) << " MiB)";
    }
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }

  return result;
}

static void compress_nvcomp(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // ---- Performance instrumentation (active at -vvv) ----
  PerfCounters perf_local;
  if (opt.verbosity >= V_TRACE) g_perf = &perf_local;

  // ---- Detect GPU devices ----
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    if (opt.gpu_only)
      die_usage("GPU requested (--gpu-only) but no CUDA devices available");
    vlog(V_VERBOSE, opt, "[GPU] no devices found; falling back to MT CPU\n");
    compress_cpu_mt(in, out, opt, m);
    return;
  }

  // Apply --gpu-devices limit (0 = all for compress)
  const int total_hw_devices = device_count;
  if (opt.gpu_devices > 0 && opt.gpu_devices < device_count) {
    vlog(V_VERBOSE, opt, "[GPU] limiting to " + std::to_string(opt.gpu_devices)
         + " of " + std::to_string(device_count) + " GPU devices\n");
    device_count = opt.gpu_devices;
  }

  if (opt.verbosity >= V_VERBOSE && (opt.level_user_set || opt.fast_flag || opt.best_flag))
    vlog(V_VERBOSE, opt,
         "note: GPU uses fixed compression; CPU frames honor level "
         + std::to_string(opt.level) + "\n");

  // ---- Determine chunk size ----
  // 16 MiB default: matches GPU_SUBCHUNK_MAX, no splitting needed.
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;

  // Ultra levels need chunk >= window size for CPU workers in hybrid mode.
  // GPU workers use nvCOMP (fixed level), but CPU rescue/hybrid workers use zstd.
  {
    size_t ultra_min = ultra_min_chunk_mib(opt.level, opt.ultra);
    if (ultra_min > 0 && chosen_mib < ultra_min && !opt.gpu_only) {
      if (!opt.chunk_user_set) {
        vlog(V_VERBOSE, opt, "[ULTRA] auto-increasing chunk size from "
             + std::to_string(chosen_mib) + " to " + std::to_string(ultra_min)
             + " MiB (must be >= window size for CPU workers)\n");
        chosen_mib = ultra_min;
      } else {
        vlog(V_ERROR, opt, "warning: --chunk-size=" + std::to_string(chosen_mib)
             + " MiB is smaller than the ultra window ("
             + std::to_string(ultra_min) + " MiB). CPU compression ratio will suffer.\n");
      }
    }
  }

  // GPU slots are sized to gpu_chunk = min(host_chunk, GPU_SUBCHUNK_MAX) and the
  // GPU worker does NOT split a Task into subchunks, so a host chunk larger than
  // GPU_SUBCHUNK_MAX would overflow the device input slot on the H2D copy — a
  // device out-of-bounds write, i.e. an illegal memory access that wedges the
  // GPU.  Any GPU-capable run reaches this function (cpu-only never does), so
  // clamp here.  This overrides the ultra auto-bump above for hybrid: a >16 MiB
  // ultra window for the CPU rescue workers is not worth a GPU OOB.
  {
    const size_t gpu_cap_mib = GPU_SUBCHUNK_MAX / ONE_MIB;
    if (chosen_mib > gpu_cap_mib) {
      vlog(V_DEFAULT, opt, "warning: clamping chunk size "
           + std::to_string(chosen_mib) + " -> " + std::to_string(gpu_cap_mib)
           + " MiB (GPU subchunks cap at " + std::to_string(gpu_cap_mib) + " MiB)\n");
      chosen_mib = gpu_cap_mib;
    }
  }

  // Pre-flight: check we won't OOM (CPU rescue threads also need buffers)
  int cpu_threads_est = opt.cpu_only ? 0 : resolve_cpu_threads(opt.cpu_threads);
  chosen_mib = check_ram_budget(std::max(1, cpu_threads_est), chosen_mib, opt);
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // ---- Shared state for all workers ----
  TaskQueue   queue;
  ResultStore results;
  std::atomic<size_t> seq_counter{0};
  std::atomic<bool>   any_gpu_failed{false};
  std::atomic<bool>   abort_on_failure{ opt.gpu_only };
  std::atomic<bool>   gpu_started{false};

  // Select GPUs before allocating per-device arrays
  const uint64_t gpu_sel_t0 = now_ns();
  auto gpu_ids = select_best_gpus(total_hw_devices, device_count, opt);
  // Fixed-share mode: warm GPU contexts before the pipeline so a small
  // input isn't drained to CPU before the GPU registers (see
  // warm_gpu_contexts).  Adaptive mode defers for fastest startup.
  if (opt.cpu_share >= 0.0) warm_gpu_contexts(gpu_ids);
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[GPU] device selection: " << std::fixed << std::setprecision(1)
       << double(now_ns() - gpu_sel_t0) / 1e6 << " ms";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }
  const int gpu_count_early = (int)gpu_ids.size();

  std::vector<DevStats> per_dev(gpu_count_early);
  StatsSink json_sink(gpu_count_early);
  CpuAgg cpuagg{};
  cpuagg.threads = 0;

  // Get total input size for progress percentage (unknown for pipes).
  // --tar: build the archive layout up front (see compress_cpu_mt).
#ifndef _WIN32
  tarx::TarLayout tar_layout;
  bool tar_built = false;
  if (opt.tar_mode) { tar_layout = tarx::build_layout(opt, m); tar_built = true; }
  uint64_t total_in = tar_built ? tar_layout.total_size : known_input_size(opt, in);
#else
  uint64_t total_in = known_input_size(opt, in);
#endif

  // Preallocate output file: input size is a safe upper bound for compressed output.
#ifndef _WIN32
  if (g_direct_writer && total_in > 0 && opt.preallocate_output
      && g_direct_writer->preallocate(total_in)) {
    char sz[32]; human_bytes(double(total_in), sz, sizeof(sz));
    vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
  }
#endif

  // Frame throttle: bounds in-flight frames to prevent RAM exhaustion.
  // Budget scales with pipeline parallelism (CPU threads + GPU streams *
  // batch cap), capped by half of available RAM.  See
  // compute_throttle_budget header comment for rationale.
  //
  // Use the EFFECTIVE max batch the GPU side might pop, not just
  // opt.gpu_batch_cap.  When --gpu-batch is not user-pinned, the v0.12.32
  // auto-tuner can grow per-stream batches up to AUTO_TUNE_BATCH_CEILING
  // (256).  Sizing the throttle around opt.gpu_batch_cap (default 8) starves
  // the GPU side: every GPU pop acquires `pop_n` permits, and with 96 CPUs
  // each holding 1 permit, the budget runs out and GPUs block waiting for
  // CPU writers to drain.  Use the larger of the two so the budget actually
  // covers all the GPU permit demand.
  const size_t per_stream_budget = opt.gpu_batch_user_set
      ? std::max<size_t>(1, opt.gpu_batch_cap)
      : std::max<size_t>(opt.gpu_batch_cap, AUTO_TUNE_BATCH_CEILING);
  const int comp_gpu_batch_floor = gpu_count_early
      * (int)std::max<size_t>(1, opt.gpu_streams)
      * (int)per_stream_budget;
  const int comp_parallelism = cpu_threads_est + comp_gpu_batch_floor;
  // Byte-cap the input queue (compress-side ROADMAP 7.8 analogue): a pipe/stdin
  // producer that outruns the consumers reads frames into heap, so bound the
  // queued owned bytes.  No-op for the common mmap path (zero-copy views,
  // data.size()==0); bounds only fread.  Bytes only — no frame cap, so mmap can
  // run ahead freely.  Tunable via --throttle-factor.
  if (opt.throttle_frames != 0) {
    int qslack = opt.throttle_factor > 0 ? opt.throttle_factor : THROTTLE_SLACK_FACTOR;
    size_t qfloor = (size_t)std::max(THROTTLE_MIN_FRAMES, comp_parallelism * qslack);
    queue.set_max_bytes(qfloor * (host_chunk / 2));
  }
  FrameThrottle throttle(compute_throttle_budget(
      std::max<size_t>(1, chosen_mib) * ONE_MIB, comp_parallelism,
      comp_gpu_batch_floor, opt));

  // Start progress bar and ordered-writer threads
  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m, &throttle);

  // ---- Hybrid scheduler (adaptive CPU/GPU work-sharing) ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, resolve_cpu_threads(opt.cpu_threads),
        device_count, opt);
    sched = sched_ptr.get();
    sched->set_queue(&queue);
    tick_thr = std::thread(tick_loop_fn, std::ref(tick_done), sched);

    // Early banner in main() already printed mode + CPU share at -v.
    // Keep a -vv confirmation that the scheduler actually started.
    if (opt.verbosity >= V_DEBUG) {
      double share_pct = (opt.cpu_share >= 0.0)
                         ? (opt.cpu_share * 100.0)
                         : (sched->target_share() * 100.0);
      const char * mode_str = (opt.cpu_share >= 0.0) ? "% (fixed)" : "% (adaptive)";
      std::ostringstream os;
      os << std::fixed << std::setprecision(1)
         << "hybrid scheduler online: CPU share " << share_pct << mode_str;
      vlog(V_DEBUG, opt, os.str() + "\n");
    }
  }

  // (No CPU "rescue" pool: a GPU fault aborts the whole compress pass and the
  // driver rebuilds CPU-only from the original input — there is nothing to
  // rescue mid-run.  See g_gpu_aborted.)

  // In hybrid mode, start CPU pool BEFORE GPU workers so CPUs can compress
  // early chunks while GPUs are still initializing (CUDA context, memory alloc).
  // Once GPUs come online, the adaptive scheduler shifts work to them.
  // ---- CPU worker pool (started BEFORE GPU for early-start optimization) ----
  // In hybrid mode, CPUs begin compressing chunks immediately while GPUs are
  // still doing CUDA context init and memory allocation.
  std::vector<std::thread> cpu_pool;
  int cpu_threads = 0;
  RateMatchState rate_match_compress;
  if (sched) {
    cpu_threads = resolve_cpu_threads(opt.cpu_threads);
    cpuagg.threads = cpu_threads;
    cpuagg.per_thread.resize((size_t)cpu_threads);

    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[HYBRID] starting CPU pool: " << cpu_threads
         << " threads (GPUs initializing in background)";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
    for (int i = 0; i < cpu_threads; ++i)
      cpu_pool.emplace_back(cpu_worker, i, &queue, &results, &opt, m, (void*)sched, &rate_match_compress, &cpuagg, &throttle);
  }

  // ---- GPU workers (init CUDA context, allocate memory  CPUs already working) ----
  const int gpu_count = gpu_count_early;
  results.init_slots(gpu_count);  // per-GPU result slots (reduces lock contention)
  std::vector<std::thread> workers;
  workers.reserve(gpu_count);
  Options opt_for_workers = opt;
  opt_for_workers.chunk_mib = chosen_mib;
  std::vector<std::string> fatal_msgs(gpu_count);

  // Shared auto-tune state: all GPUs coordinate batch size through this
  SharedTuneState shared_tune;
  shared_tune.batch_size.store(opt.gpu_batch_cap);
  shared_tune.locked.store(opt.gpu_batch_user_set);
  // VRAM ceiling will be refined by the first worker's binary search

  // Counts GPUs that failed terminally (init or mid-run).  When the count
  // reaches gpu_count in --gpu-only mode, the last failing worker runs
  // gpu_only_cpu_fallback to finish the job on CPU (v0.13.54) — so the
  // producer keeps reading and the queue always has a consumer.
  std::atomic<int> gpu_failures{0};

  for (int i = 0; i < gpu_count; ++i) {
    workers.emplace_back(gpu_worker, gpu_ids[i], i, opt_for_workers,
                         &queue, &results,
                         &per_dev[size_t(i)], &json_sink, m, sched,
                         &any_gpu_failed, &abort_on_failure,
                         &fatal_msgs[size_t(i)], &gpu_started,
                         &shared_tune, &rate_match_compress,
                         &throttle, &gpu_failures, gpu_count);
  }
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[GPU] " << gpu_count << " device worker"
       << (gpu_count > 1 ? "s" : "") << " online";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }

  // Fixed-share: wait for at least one GPU stream to register (or for all
  // GPUs to fail init) before streaming frames.  warm_gpu_contexts only
  // creates the CUDA contexts; the GPU worker still does VRAM probe +
  // cudaMalloc + register_gpu_stream afterward.  On fast multi-CPU /
  // many-GPU machines the reader + CPU pool can otherwise drain a small
  // input via the drain-phase fast path (qs.done && !any_gpu_active)
  // before any GPU registers, silently collapsing the requested CPU/GPU
  // split to all-CPU.  Adaptive mode skips this — it promises no exact
  // split and wants the fastest possible start.  v0.13.14.
  if (sched && opt.cpu_share >= 0.0 && gpu_count > 0) {
    while (!sched->any_gpu_active()
           && gpu_failures.load(std::memory_order_acquire) < gpu_count) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // ---- Producer: read input, split into GPU-sized subchunks, enqueue ----
  try_boost_io_priority(!opt.gpu_only);
  const size_t gpu_chunk = std::min(host_chunk, GPU_SUBCHUNK_MAX);
#ifndef _WIN32
  MmapRegion mmap_region;
  DirectReadPool nv_pool;   // buffered zero-copy pool; must outlive all workers
  bool reader_done = false;
  // --tar: parallel assembler at GPU-chunk granularity (≤16 MiB frames feed the
  // nvCOMP batched path unchanged).  Pushes in sequence order.
  if (opt.tar_mode) {
    tarx::assemble(queue, tar_layout, gpu_chunk, opt, m);
    reader_done = true;
  }
  // --direct-read: O_DIRECT input (bypass page cache); splits each host chunk into
  // gpu_chunk subchunks like the other readers.  Precedence over mmap; falls
  // through to fread if O_DIRECT can't open.
  if (opt.direct_read && opt.input != "-" && fs::is_regular_file(opt.input)) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[DIRECT-READ] O_DIRECT input (page cache bypassed)\n");
    const size_t subs_per_host = (host_chunk + gpu_chunk - 1) / gpu_chunk;
    // GPU path keeps the copy: each host chunk is split into gpu_chunk subchunks, so
    // one owning buffer per host read doesn't map to one Task (pool == nullptr).
    // No early-abort on GPU failure: even if every GPU dies, the CPU
    // fallback (gpu_only_cpu_fallback / hybrid CPU pool) consumes the queue,
    // so the reader always streams the whole input.
    reader_done = pooled_read_chunks(opt.input, host_chunk, m, nullptr, /*o_direct=*/true,
      /*n_readers=*/1, /*borrowed_fd=*/-1,
      [&](const char * buf, size_t n_host, size_t idx, int /*slot*/) {
        // Deterministic, contiguous seq by file position: only the last (highest-
        // idx) chunk is partial, so chunk idx owns [idx*subs_per_host, +n_subs).
        size_t sub_off = 0, sub_i = 0;
        while (sub_off < n_host) {
          size_t sub_n = std::min(gpu_chunk, n_host - sub_off);
          enqueue_direct_chunk(queue, idx * subs_per_host + sub_i, buf + sub_off, sub_n);
          sub_off += sub_n; ++sub_i;
        }
        return true;
      });
  }
  if (!reader_done && opt.use_mmap && opt.input != "-" && fs::exists(opt.input)
      && fs::is_regular_file(opt.input)
      && mmap_ok_for_input(opt, opt.input)
      && mmap_region.open(opt.input.c_str())) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[MMAP] using zero-copy reader\n");
    const char * base = mmap_region.data();
    const size_t file_size = mmap_region.size();
    size_t off = 0;
    uint64_t mmap_rd_t0 = g_perf ? now_ns() : 0;
    while (off < file_size) {
      size_t n_host = std::min(host_chunk, file_size - off);
      size_t sub_off = 0;
      while (sub_off < n_host) {
        size_t sub_n = std::min(gpu_chunk, n_host - sub_off);
        Task t;
        t.seq = seq_counter.fetch_add(1, std::memory_order_relaxed);
        t.view_ptr = base + off + sub_off;
        t.view_len = sub_n;
        queue.push(std::move(t));
        sub_off += sub_n;
      }
      off += n_host;
    }
    if (g_perf) {
      g_perf->read_ns.fetch_add(now_ns() - mmap_rd_t0);
      g_perf->read_bytes_total.fetch_add(off);
    }
    reader_done = true;
  }
  // mmap declined (pre-6.4 gate, --no-mmap, open failure), regular file:
  // multi-thread buffered pooled reads at GPU-CHUNK granularity — one pool
  // buffer per Task, so the single-owner slot release works unchanged (the
  // host-chunk/subchunk split was an fread-efficiency artifact; pooled
  // preads don't need it, and seq == chunk idx stays dense).  Slots are
  // recycled on the batch-success, partial-rescue, and rescue-worker paths
  // (v0.13.65) and by CPU workers as before.  GPU batches hold slots from
  // pop to delivery, so the pool is sized far larger than the cpu-only
  // reader's — clamped to a quarter of available RAM (the FrameThrottle
  // bounds total in-flight output; this bounds in-flight input).
  // Preadable (named regular/block file OR a seekable stdin redirect)?  fstat
  // decides; pipes (tar -I gzstd) fall through to fread below.
  bool dr_owns = false; uint64_t dr_sz = 0;
  int dr_fd = reader_done ? -1 : probe_preadable_input(opt, in, &dr_owns, &dr_sz);
  if (dr_fd >= 0) {
    // Reader count scales with MACHINE parallelism, not the CPU pool:
    // cpu_threads is 0 in gpu-only mode, which silently capped gpu-only at
    // 3 readers (= 14.17 GiB/s measured) while the H100 pool had headroom.
    const int hw_par = resolve_cpu_threads(opt.cpu_threads);
    const int n_readers = opt.read_threads > 0 ? (int)opt.read_threads
                        : std::max(3, std::min(12, hw_par / 8));
    const size_t cap = ((gpu_chunk + 4095) / 4096) * 4096;
    const size_t file_chunks = (dr_sz > 0) ? (size_t)((dr_sz + cap - 1) / cap)
                                           : (size_t)cpu_threads;
    size_t pool_n = std::min<size_t>(
        (size_t)cpu_threads + 32 * (size_t)n_readers + 1024,  // CPU pool + reader lead + GPU window
        file_chunks + 1);
    const uint64_t ram_avail = get_available_ram_bytes();
    if (ram_avail > 0)
      pool_n = std::min<size_t>(pool_n, std::max<size_t>(8, (size_t)(ram_avail / 4 / cap)));
    if (pool_n < 8) pool_n = std::min<size_t>(8, std::max<size_t>(file_chunks + 1, 1));
    const bool pool_ok = nv_pool.init(cap, pool_n,
                                      /*want_thp=*/kernel_has_per_vma_locks());
    if (pool_ok) {
      g_direct_read_pool = &nv_pool;
      // Declare the depth ceiling: the queue can never hold more than
      // pool_n frames, so the GPU queue floor must clamp below it or the
      // CPU pool gets locked out (see update_queue_floor).
      if (sched) sched->set_queue_depth_cap(pool_n);
      if (opt.verbosity >= V_VERBOSE)
        vlog(V_VERBOSE, opt, "[POOLED-READ] buffered input (page cache + readahead), zero-copy pool "
             + std::to_string(pool_n) + " buffers\n");
      reader_done = pooled_read_chunks(opt.input, gpu_chunk, m, &nv_pool,
        /*o_direct=*/false, n_readers, dr_fd,
        [&](const char * buf, size_t n, size_t idx, int slot) {
          Task t; t.seq = idx; t.view_ptr = buf; t.view_len = n; t.direct_buf = slot;
          queue.push(std::move(t));
          return true;
        });
      if (!reader_done) g_direct_read_pool = nullptr;  // open failed; fread takes over
    }
    // pool alloc failure: fall through to fread+copy below.
    if (dr_owns) ::close(dr_fd);   // pooled reader borrows the fd; we opened it
  }
  if (!reader_done)
#endif
  {
    std::vector<char> host_in(host_chunk);
    while (true) {
      uint64_t rd_t0 = (g_perf || m) ? now_ns() : 0;
      size_t n_host = std::fread(host_in.data(), 1, host_chunk, in);
      if ((g_perf || m) && n_host > 0) {
        const uint64_t r_dt = now_ns() - rd_t0;
        if (m) m->reader_io_ns.fetch_add(r_dt, std::memory_order_relaxed);
        if (g_perf) {
          g_perf->read_ns.fetch_add(r_dt);
          g_perf->read_bytes_total.fetch_add(n_host);
        }
      }
      if (n_host == 0) break;
      if (m) m->read_bytes.fetch_add(n_host);
      size_t off = 0;
      const uint64_t c_t0 = m ? now_ns() : 0;
      while (off < n_host) {
        size_t sub_n = std::min(gpu_chunk, n_host - off);
        Task t;
        t.seq = seq_counter.fetch_add(1, std::memory_order_relaxed);
        t.data.assign(host_in.data() + off, host_in.data() + off + sub_n);
        queue.push(std::move(t));
        off += sub_n;
      }
      if (m) m->reader_copy_ns.fetch_add(now_ns() - c_t0, std::memory_order_relaxed);
    }
  }

  // Signal workers that all input has been read
  queue.set_done();
  if (sched) sched->set_producer_done();  // arm the tail-aware GPU intake check
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
    if (m) m->total_frames.store(results.total_tasks, std::memory_order_relaxed);
  }
  results.cv.notify_all();

  // ---- Teardown: join all threads in correct order ----
  // Do NOT call throttle.set_done() before join: workers must respect
  // throttle while draining the queue to avoid buffering entire output in RAM.
  for (auto & th : workers) th.join();
  throttle.set_done();  // safe now: all workers exited

  // Report GPU failures.  Even when ALL GPUs failed in --gpu-only mode the
  // job is already complete: the last failing worker ran
  // gpu_only_cpu_fallback (with its own warning), and the rescue pool
  // recompressed any in-flight batches.  Output integrity is intact — CPU and
  // GPU emit interchangeable zstd frames, and verify-on-write recompresses any
  // silently-corrupt GPU frame on the CPU — so the archive on disk is complete
  // and correct.  But every GPU faulting is a hardware/driver problem worth
  // surfacing, so we flag it for an EXIT_GPU_FAIL once the output is finalized.
  if (any_gpu_failed.load()) {
    // If a GPU faulted mid-run (g_gpu_failed_restart), this output is about to
    // be discarded and rebuilt CPU-only by the driver — say so plainly.  A
    // startup VRAM skip with no fault just continues on the remaining workers.
    const bool rebuilding = g_gpu_failed_restart.load();
    const char * suffix = rebuilding
        ? " (output will be discarded; rebuilding CPU-only)\n"
        : (abort_on_failure.load() ? " (other GPUs continuing)\n"
                                   : " (handled on CPU)\n");
    for (const auto & s : fatal_msgs)
      if (!s.empty())
        vlog(V_DEFAULT, opt, "WARNING: " + s + suffix);
  }

  // Join the hybrid CPU pool.
  queue.notify_cpu_waiters();
  if (!cpu_pool.empty())
    for (auto & th : cpu_pool) th.join();
#ifndef _WIN32
  g_direct_read_pool = nullptr;  // all release() callers joined; safe to drop the pool
#endif

  // Signal writer that all workers are done
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();

  // Stop adaptive tick thread
  if (sched) {
    tick_done = true;
    if (tick_thr.joinable()) tick_thr.join();
  }

  // Wait for writer and progress threads
  writer_thr.join();
  progress_done = true;
  progress_thr.join();
  log_throttle_stats(throttle, opt,
                     opt.hybrid ? "compress-hybrid" :
                     opt.gpu_only ? "compress-gpu" : "compress-nvcomp");

  if (g_perf) {
    g_perf->print_summary(opt.hybrid ? "HYBRID COMPRESS" :
                          opt.gpu_only ? "GPU-ONLY COMPRESS" : "COMPRESS");
    g_perf = nullptr;
  }
}

/*======================================================================
 GPU decompression worker
======================================================================*/
// Each batch: upload compressed frames to GPU, call nvCOMP batched decompress,
// download decompressed data.  Similar structure to gpu_worker for compression
// but using the decompress API.
static void gpu_decomp_worker(
  int device_id,
  int slot_index,
  Options opt,
  TaskQueue * queue,
  RescueQueue * rescue,
  ResultStore * results,
  Meter * m,
  HybridSched * sched,
  std::atomic<bool> * any_gpu_failed,
  std::atomic<bool> * abort_on_failure,
  std::string * fatal_msg,
  SharedTuneState * shared_tune,
  RateMatchState * rate_match,
  FrameThrottle * bp,
  std::atomic<int> * gpu_failures,   // terminal failures (init or mid-run), all workers
  int gpu_worker_count)              // total GPU workers spawned (for last-failure detection)
{
  (void)m;
  const size_t stream_count = std::max<size_t>(1, (size_t)opt.gpu_streams);

  // Per-stream state for decompression (declared before try so catch can rescue)
  struct DecompStreamCtx {
    cudaStream_t stream{};
    cudaEvent_t ev_begin{}, ev_end{};
    std::vector<Task> batch;
    bool busy = false;
    size_t filled = 0;
    size_t delivered = 0;   // frames of the current batch already pushed to ResultStore
                            // (lets the failure path re-enqueue only the undelivered tail)
    size_t stream_index = 0;

    // (Per-stream auto-tune tracking removed v0.13.34: dead code, superseded by
    // the shared SharedTuneState hill-climb — same as the compress StreamCtx.)

    // Pre-allocated device buffers (reused across batches)
    size_t alloc_batch = 0;     // how many slots are allocated
    size_t alloc_comp = 0;      // bytes per compressed slot
    size_t alloc_decomp = 0;    // bytes per decompressed slot
    void * d_comp_buf = nullptr;
    void * d_decomp_buf = nullptr;
    void * d_temp = nullptr;
    size_t temp_bytes = 0;
    void ** d_comp_ptrs = nullptr;
    void ** d_decomp_ptrs = nullptr;
    size_t * d_comp_sizes = nullptr;
    size_t * d_decomp_sizes = nullptr;
    size_t * d_actual_sizes = nullptr;
    nvcompStatus_t * d_statuses = nullptr;

    // Host-side arrays
    std::vector<void*> h_comp_ptrs, h_decomp_ptrs;
    std::vector<size_t> h_comp_sizes, h_decomp_sizes, h_actual;
    std::vector<nvcompStatus_t> h_statuses;

    // Pinned host buffer for D2H decompressed output (faster cudaMemcpy
    // than to pageable memory).  Sized batch_n * max_decomp; reused across
    // batches.  Allocated through the global pinned-RAM budget (--pinned).
    void * h_decomp_pinned = nullptr;
    size_t h_decomp_pinned_bytes = 0;

    // Recycled host output buffers for D2H readback.  Avoids a fresh
    // make_shared (+ a full-frame minor-fault storm) per frame on every batch
    // — the GPU analogue of the cpu_decomp_worker pool (ROADMAP 7.2).  A slot
    // with use_count()==1 has been drained by the writer and is free to reuse;
    // recycled slots keep their resident pages, so the steady state stops
    // faulting.  Grows lazily up to a cap; never preallocated.
    std::vector<FrameBuf> out_pool;
    uint64_t out_pool_waits = 0;

    // Acquire a recycled output buffer.  Grows the pool up to `cap` slots, then
    // waits on the writer's drain signal until a slot frees (so it never
    // allocates beyond cap).  Deadlock-free: a stream pushes its frames in
    // ascending seq order, so the writer always has the oldest in-flight frame
    // to write, drops its ref after writing, and frees a slot.
    FrameBuf acquire_out_buf(size_t cap, FrameThrottle * bp) {
      for (;;) {
        for (auto & b : out_pool)
          if (b.use_count() == 1) return b;
        if (out_pool.size() < cap) {
          out_pool.push_back(std::make_shared<FrameVec>());
          return out_pool.back();
        }
        ++out_pool_waits;
        if (bp) {
          bp->wait_for_drain([&]{
            for (auto & b : out_pool) if (b.use_count() == 1) return true;
            return false;
          });
        } else {
          std::this_thread::yield();
        }
      }
    }

    void free_device() {
      if (d_comp_buf)    { cudaFree(d_comp_buf);    d_comp_buf = nullptr; }
      if (d_decomp_buf)  { cudaFree(d_decomp_buf);  d_decomp_buf = nullptr; }
      if (d_temp)        { cudaFree(d_temp);         d_temp = nullptr; }
      if (d_comp_ptrs)   { cudaFree(d_comp_ptrs);   d_comp_ptrs = nullptr; }
      if (d_decomp_ptrs) { cudaFree(d_decomp_ptrs); d_decomp_ptrs = nullptr; }
      if (d_comp_sizes)  { cudaFree(d_comp_sizes);   d_comp_sizes = nullptr; }
      if (d_decomp_sizes){ cudaFree(d_decomp_sizes); d_decomp_sizes = nullptr; }
      if (d_actual_sizes){ cudaFree(d_actual_sizes); d_actual_sizes = nullptr; }
      if (d_statuses)    { cudaFree(d_statuses);     d_statuses = nullptr; }
      alloc_batch = 0; alloc_comp = 0; alloc_decomp = 0; temp_bytes = 0;
    }
    // Free pinned host buffer.  Caller must release the budget separately
    // (we don't have access to Options here).  Returns the byte count freed.
    size_t free_host_pinned() {
      size_t freed = 0;
      if (h_decomp_pinned) {
        cudaFreeHost(h_decomp_pinned);
        h_decomp_pinned = nullptr;
        freed = h_decomp_pinned_bytes;
        h_decomp_pinned_bytes = 0;
      }
      return freed;
    }

    // Ensure buffers are large enough for the given batch parameters.
    // Returns false on allocation failure.
    bool ensure_buffers(size_t batch_n, size_t max_comp, size_t max_decomp,
                        size_t needed_temp) {
      if (batch_n <= alloc_batch && max_comp <= alloc_comp
          && max_decomp <= alloc_decomp && needed_temp <= temp_bytes)
        return true;  // already big enough

      // Need to reallocate
      free_device();
      alloc_batch  = batch_n;
      alloc_comp   = max_comp;
      alloc_decomp = max_decomp;

      if (cudaMalloc(&d_comp_buf,     batch_n * max_comp)    != cudaSuccess) return false;
      if (cudaMalloc(&d_decomp_buf,   batch_n * max_decomp)  != cudaSuccess) return false;
      if (cudaMalloc(&d_comp_ptrs,    batch_n * sizeof(void*))   != cudaSuccess) return false;
      if (cudaMalloc(&d_decomp_ptrs,  batch_n * sizeof(void*))   != cudaSuccess) return false;
      if (cudaMalloc(&d_comp_sizes,   batch_n * sizeof(size_t))  != cudaSuccess) return false;
      if (cudaMalloc(&d_decomp_sizes, batch_n * sizeof(size_t))  != cudaSuccess) return false;
      if (cudaMalloc(&d_actual_sizes, batch_n * sizeof(size_t))  != cudaSuccess) return false;
      if (cudaMalloc(&d_statuses,     batch_n * sizeof(nvcompStatus_t)) != cudaSuccess) return false;
      if (needed_temp > 0) {
        if (cudaMalloc(&d_temp, needed_temp) != cudaSuccess) return false;
        temp_bytes = needed_temp;
      }

      // Set up pointer arrays (offsets are fixed for a given max_comp/max_decomp)
      h_comp_ptrs.resize(batch_n);
      h_decomp_ptrs.resize(batch_n);
      for (size_t i = 0; i < batch_n; ++i) {
        h_comp_ptrs[i]   = static_cast<char*>(d_comp_buf)   + i * max_comp;
        h_decomp_ptrs[i] = static_cast<char*>(d_decomp_buf) + i * max_decomp;
      }
      checkCuda(cudaMemcpy(d_comp_ptrs, h_comp_ptrs.data(),
                           batch_n * sizeof(void*), cudaMemcpyHostToDevice),
                "H2D decomp ptrs");
      checkCuda(cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(),
                           batch_n * sizeof(void*), cudaMemcpyHostToDevice),
                "H2D decomp ptrs");

      h_comp_sizes.resize(batch_n);
      h_decomp_sizes.resize(batch_n);
      h_actual.resize(batch_n);
      h_statuses.resize(batch_n);
      return true;
    }
  };

  std::vector<DecompStreamCtx> ctxs;  // declared here so catch block can rescue frames
  void * vram_reserve = nullptr;
  size_t vram_reserve_bytes = 0;
  try {
    uint64_t init_t0 = g_perf ? now_ns() : 0;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    // For decompress, gpu_batch_cap is PER STREAM (not divided across streams).
    // Kernel launch overhead dominates, so each stream needs large batches.
    // Compress divides across streams for VRAM management, but decompress
    // frames are small (compressed) so VRAM isn't the bottleneck.
    //
    // When --gpu-batch is NOT user-pinned, give the shared auto-tuner real
    // room to grow by sizing buffers up to max(opt.gpu_batch_cap,
    // AUTO_TUNE_BATCH_CEILING).  Decompress already auto-bumps gpu_batch_cap
    // to 256 for files >75 GiB (parse_args), so this only changes behaviour
    // for smaller files where the tuner was previously capped at 16/64.
    // The VRAM-fit halve loop below shrinks this if the GPU can't hold it.
    // Same fix as compress in v0.12.32 — see CHANGELOG.
    size_t per_stream_cap = opt.gpu_batch_user_set
        ? std::min(opt.gpu_batch_cap, HARD_BATCH_CAP)
        : std::max(opt.gpu_batch_cap, AUTO_TUNE_BATCH_CEILING);

    // We need to allocate per-stream buffers.  Unlike compression, we need
    // to handle variable decompressed sizes.  We pre-allocate based on the
    // chunk size (which upper-bounds both compressed and decompressed size)
    // and reuse across batches to avoid per-batch cudaMalloc overhead.

    const size_t host_chunk_bytes = std::max<size_t>(1, opt.chunk_mib) * ONE_MIB;

    ctxs.resize(stream_count);
    for (size_t s = 0; s < stream_count; ++s) {
      auto & C = ctxs[s];
      checkCuda(cudaStreamCreate(&C.stream), "cudaStreamCreate");
      cudaEventCreateWithFlags(&C.ev_begin, cudaEventDefault);
      cudaEventCreateWithFlags(&C.ev_end, cudaEventDefault);
      C.batch.reserve(per_stream_cap);
      C.stream_index = s;

      // Pre-allocate for the common case: batch_cap frames of chunk_size
      // The temp size query needs device pointers, so we estimate conservatively.
      // We'll resize later if a batch has larger frames.
      size_t init_comp = host_chunk_bytes;  // compressed <= original chunk
      size_t init_decomp = host_chunk_bytes;
      // Pre-allocate device buffers.  If batch size is too large for VRAM,
      // halve it until it fits.  This handles --gpu-batch=256 on GPUs with
      // limited VRAM (e.g., 10 GiB consumer GPUs vs 80+ GiB datacenter GPUs).
      bool stream_init_failed = false;
      {
        size_t est_temp = per_stream_cap * 1024;
        size_t try_batch = per_stream_cap;
        int vram_retries = 0;
        while (!C.ensure_buffers(try_batch, init_comp, init_decomp, est_temp)) {
          if (try_batch <= 1 || ++vram_retries > 10) {
            // Can't fit even batch=1 on this stream.  If we already have
            // working streams, stop adding more and run with what we have
            // (auto-decrement of --gpu-streams when VRAM is tight).
            stream_init_failed = true;
            if (C.ev_begin) { cudaEventDestroy(C.ev_begin); C.ev_begin = nullptr; }
            if (C.ev_end)   { cudaEventDestroy(C.ev_end);   C.ev_end = nullptr; }
            if (C.stream)   { cudaStreamDestroy(C.stream);  C.stream = nullptr; }
            break;
          }
          try_batch = std::max<size_t>(1, try_batch / 2);
          {
            int min_verb = opt.gpu_batch_user_set ? V_NORMAL : V_VERBOSE;
            if (opt.verbosity >= min_verb) {
              std::ostringstream os;
              os << "[GPU" << device_id << "/S" << s
                 << "] VRAM insufficient, reducing batch to " << try_batch;
              vlog(min_verb, opt, os.str() + "\n");
            }
          }
        }
        if (!stream_init_failed) {
          // Update per_stream_cap to the actual allocated size
          // (used by pop_batch_greedy to limit how many frames we grab)
          per_stream_cap = try_batch;
        }
      }
      if (stream_init_failed) {
        if (s == 0) {
          // Couldn't fit even one stream — skip this GPU entirely.
          std::string skip_msg = "[GPU" + std::to_string(device_id)
              + "] insufficient VRAM for even 1 stream at batch=1 ("
              + std::to_string(init_comp / ONE_MIB) + " MiB comp + "
              + std::to_string(init_decomp / ONE_MIB) + " MiB decomp per frame)  skipping device";
          vlog(V_ERROR, opt, skip_msg + "\n");
          *any_gpu_failed = true;
          *fatal_msg = skip_msg;
          int fails = gpu_failures
              ? gpu_failures->fetch_add(1, std::memory_order_acq_rel) + 1 : 1;
          { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
          queue->notify_cpu_waiters();
          // Last GPU standing just failed in --gpu-only: finish on CPU
          // instead of stranding the queue (data safety over mode purity).
          if (opt.gpu_only && fails == gpu_worker_count)
            gpu_only_cpu_fallback(true, queue, results, opt, m, bp);
          return;
        }
        vlog(V_DEFAULT, opt,
             "WARNING: [GPU" + std::to_string(device_id)
             + "] VRAM insufficient for " + std::to_string(stream_count)
             + " streams at batch=1; auto-reducing to " + std::to_string(s)
             + " stream" + (s == 1 ? "" : "s") + "\n");
        ctxs.resize(s);
        break;  // exit stream-init loop
      }

      if (opt.verbosity >= V_DEBUG) {
        std::ostringstream os;
        os << "[GPU" << device_id << "/S" << s
           << "] pre-alloc batch=" << per_stream_cap
           << " comp=" << (init_comp / ONE_MIB) << "MiB"
           << " decomp=" << (init_decomp / ONE_MIB) << "MiB";
        vlog(V_DEBUG, opt, os.str() + "\n");
      }
    }

    bool producer_done_seen = false;
    if (g_perf) { uint64_t dt = now_ns() - init_t0; g_perf->cuda_init_sum_ns.fetch_add(dt); g_perf->cuda_init_count.fetch_add(1); uint64_t cur = g_perf->cuda_init_max_ns.load(); while (dt > cur && !g_perf->cuda_init_max_ns.compare_exchange_weak(cur, dt)); };

    // VRAM reserve: hold enough memory to process half the batch size.
    // If another user grabs VRAM mid-run and a cudaMalloc fails (e.g., temp
    // buffer growth), we free the reserve and retry.  This avoids a hard
    // failure when VRAM pressure spikes during a long job.
    {
      // Reserve = half-batch worth of (comp + decomp + overhead) per stream
      size_t half_batch = std::max<size_t>(1, per_stream_cap / 2);
      size_t per_frame = host_chunk_bytes * 2 + 4096;  // comp + decomp + metadata
      vram_reserve_bytes = half_batch * per_frame;
      if (cudaMalloc(&vram_reserve, vram_reserve_bytes) != cudaSuccess) {
        vram_reserve = nullptr;
        vram_reserve_bytes = 0;
        vlog(V_VERBOSE, opt, "[GPU" + std::to_string(device_id)
             + "] could not allocate VRAM reserve (non-fatal)\n");
      } else if (opt.verbosity >= V_DEBUG) {
        vlog(V_DEBUG, opt, "[GPU" + std::to_string(device_id)
             + "] VRAM reserve: " + std::to_string(vram_reserve_bytes / ONE_MIB) + " MiB\n");
      }
    }

    // Signal scheduler that this GPU is ready for work
    if (sched) {
      sched->set_gpu_ready(device_id);
      for (size_t s = 0; s < ctxs.size(); ++s)
        sched->register_gpu_stream();
    }

    // Utilization scaling factor: 1.0 = idle, 0.1 = 90% busy.
    // Updated after each batch completion via NVML query.
    // Applied to batch size at next pop to match busy GPUs' completion time
    // with idle GPUs' completion time.
    double util_scale = 1.0;

    // Consecutive trivial-batch skips across all streams.  If we see too many
    // in a row without doing real GPU work, the remaining workload is
    // CPU-preferred (all trivially-compressed frames) and this GPU should
    // exit so CPU can drain the queue without GPU interference.
    int trivial_skip_streak = 0;
    const int trivial_skip_exit_threshold = (int)ctxs.size() * 4;

    while (true) {
      bool submitted_any = false;

      // ---- Shared auto-tuner (decompress) ----
      // All GPUs report throughput to SharedTuneState. Same logic as compress.
      // Whichever worker grabs the mutex first runs the tune decision.
      if (shared_tune && !shared_tune->locked.load()) {
        auto now = std::chrono::steady_clock::now();
        if (shared_tune->window_batches.load(std::memory_order_relaxed) >= SharedTuneState::MIN_BATCHES) {
          std::unique_lock<std::mutex> lk(shared_tune->tune_mtx, std::try_to_lock);
          if (lk.owns_lock()) {
            double secs = std::chrono::duration_cast<std::chrono::duration<double>>(
                now - shared_tune->last_tune).count();
            if (secs >= SharedTuneState::TUNE_SEC) {
              shared_tune->last_tune = now;
              uint64_t bytes = shared_tune->window_bytes.exchange(0);
              uint64_t ns = shared_tune->window_ns.exchange(0);
              shared_tune->window_batches.store(0);
              double cur_thr = (ns > 0) ? double(bytes) / (double(ns)/1e9) / 1e9 : 0.0;
              size_t cur_batch = shared_tune->batch_size.load();

              auto & S = *shared_tune;
              if (S.phase == SharedTuneState::Phase::BASELINE) {
                S.best_batch = cur_batch; S.best_thr = cur_thr;
                S.prev_batch = cur_batch; S.prev_thr = cur_thr;
                size_t half = std::max<size_t>(1, cur_batch / 2);
                if (half < cur_batch) {
                  S.batch_size.store(half);
                  S.phase = SharedTuneState::Phase::HALVE;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] baseline=" << cur_batch << " ("
                       << std::fixed << std::setprecision(2) << cur_thr << " GiB/s) -> try " << half;
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                } else {
                  S.batch_size.store(std::min(cur_batch * 2, S.vram_ceiling.load()));
                  S.phase = SharedTuneState::Phase::DOUBLE;
                }
              } else if (S.phase == SharedTuneState::Phase::HALVE) {
                if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                if (cur_thr >= S.prev_thr * 0.98) {
                  S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                  size_t half = std::max<size_t>(1, cur_batch / 2);
                  if (half < cur_batch) { S.batch_size.store(half); }
                  else { S.batch_size.store(S.best_batch); S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                } else {
                  S.prev_thr = S.best_thr; S.prev_batch = S.best_batch;
                  S.batch_size.store(S.best_batch);
                  S.phase = SharedTuneState::Phase::DOUBLE;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] halving worse, will try doubling (best=" << S.best_batch << ")";
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                }
              } else if (S.phase == SharedTuneState::Phase::DOUBLE) {
                if (cur_batch == S.best_batch) {
                  S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                  size_t dbl = std::min(cur_batch * 2, S.vram_ceiling.load());
                  if (dbl > cur_batch) { S.batch_size.store(dbl); }
                  else { S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                } else {
                  if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                  if (cur_thr >= S.prev_thr * 0.98) {
                    S.prev_thr = cur_thr; S.prev_batch = cur_batch;
                    size_t dbl = std::min(cur_batch * 2, S.vram_ceiling.load());
                    if (dbl > cur_batch) { S.batch_size.store(dbl); }
                    else { S.batch_size.store(S.best_batch); S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0; }
                  } else {
                    if (std::abs((long)cur_batch - (long)S.best_batch) > 2) {
                      S.refine_lo = std::min(S.best_batch, cur_batch);
                      S.refine_hi = std::max(S.best_batch, cur_batch);
                      size_t mid = S.refine_lo + (S.refine_hi - S.refine_lo) / 2;
                      S.batch_size.store(mid);
                      S.phase = SharedTuneState::Phase::REFINE; S.refine_iters = 0;
                      if (opt.verbosity >= V_VERBOSE) {
                        std::ostringstream os;
                        os << "[AUTO-TUNE] refining [" << S.refine_lo << ".." << S.refine_hi
                           << "] trying " << mid;
                        vlog(V_VERBOSE, opt, os.str() + "\n");
                      }
                    } else {
                      S.batch_size.store(S.best_batch);
                      S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                      if (opt.verbosity >= V_VERBOSE) {
                        std::ostringstream os;
                        os << "[AUTO-TUNE] settled at batch=" << S.best_batch
                           << " (" << std::fixed << std::setprecision(2) << S.best_thr << " GiB/s)";
                        vlog(V_VERBOSE, opt, os.str() + "\n");
                      }
                    }
                  }
                }
              } else if (S.phase == SharedTuneState::Phase::REFINE) {
                ++S.refine_iters;
                if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                if (cur_batch < S.best_batch) S.refine_lo = cur_batch;
                else if (cur_batch > S.best_batch) S.refine_hi = cur_batch;
                if (S.refine_hi - S.refine_lo <= 2 || S.refine_iters >= SharedTuneState::MAX_REFINE_ITERS) {
                  S.batch_size.store(S.best_batch);
                  S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                  if (opt.verbosity >= V_VERBOSE) {
                    std::ostringstream os;
                    os << "[AUTO-TUNE] refined, settled at batch=" << S.best_batch;
                    vlog(V_VERBOSE, opt, os.str() + "\n");
                  }
                } else {
                  size_t mid = S.refine_lo + (S.refine_hi - S.refine_lo) / 2;
                  if (mid == cur_batch) mid++;
                  S.batch_size.store(mid);
                }
              } else {
                ++S.settle_ticks;
                if (S.settle_ticks >= SharedTuneState::PROBE_INTERVAL) {
                  S.settle_ticks = 0;
                  if (cur_thr > S.best_thr) { S.best_thr = cur_thr; S.best_batch = cur_batch; }
                  ++S.probe_count;
                  bool probe_up = (S.probe_count % 2 == 0);
                  size_t probe;
                  if (probe_up) {
                    probe = std::min(S.best_batch + S.best_batch / 4, S.vram_ceiling.load());
                    if (probe <= S.best_batch) probe = S.best_batch + 1;
                  } else {
                    probe = std::max<size_t>(1, S.best_batch - S.best_batch / 4);
                    if (probe >= S.best_batch) probe = std::max<size_t>(1, S.best_batch - 1);
                  }
                  if (probe != S.best_batch && probe <= S.vram_ceiling.load()) {
                    S.prev_thr = cur_thr; S.prev_batch = S.best_batch;
                    S.batch_size.store(probe);
                    if (probe > S.best_batch)
                      S.phase = SharedTuneState::Phase::DOUBLE;
                    else
                      S.phase = SharedTuneState::Phase::HALVE;
                    if (opt.verbosity >= V_VERBOSE) {
                      std::ostringstream os;
                      os << "[AUTO-TUNE] probe: " << S.best_batch << " -> " << probe
                         << " (" << std::fixed << std::setprecision(2) << cur_thr << " GiB/s)";
                      vlog(V_VERBOSE, opt, os.str() + "\n");
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Submit batches
      for (auto & C : ctxs) {
        if (C.busy) continue;
        // Fixed-share mode: GPU yields when CPU is below its target share.
        // If the queue is already drained, propagate that so the worker can
        // exit instead of spinning forever on the share check.
        if (sched && !sched->should_gpu_take()) {
          if (queue->drained()) { producer_done_seen = true; continue; }
          // Adaptive tail yield (compress): CPUs are draining the rest.
          // Park on the queue CV until an event that could change the
          // decision: a pop shrinks the queue, the queue drains, or a
          // scheduler tick moves the EMAs.  No polling, no fixed sleeps.
          // Only park when every stream is idle — with a batch in flight
          // this loop must keep polling cudaStreamQuery, and the
          // completion paths below already block appropriately.  (Fixed
          // mode keeps its spin: the share check is designed to
          // oscillate per-batch.  Adaptive decompress never declines.)
          if (!sched->is_fixed_mode()) {
            bool any_busy = false;
            for (auto & X : ctxs) if (X.busy) { any_busy = true; break; }
            if (!any_busy &&
                !queue->wait_for_gpu_yield(
                    [&](const TaskQueue::QueueState & qs) {
                      return sched->should_gpu_take_at(qs.depth);
                    }))
              producer_done_seen = true;
          }
          continue;
        }
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        // Use shared batch size from auto-tuner, scaled by GPU utilization.
        // A GPU at 50% utilization gets half the batch → finishes at roughly
        // the same time as idle GPUs → results arrive in order for the writer.
        size_t pop_n = (shared_tune && !shared_tune->locked.load())
                     ? std::min(shared_tune->batch_size.load(std::memory_order_relaxed), per_stream_cap)
                     : per_stream_cap;
        // Keep scheduler's queue floor in sync with current batch size
        if (sched) sched->set_gpu_batch_size(pop_n);
        // Apply utilization scaling (updated after each batch completion)
        pop_n = std::max<size_t>(1, (size_t)(pop_n * util_scale));
        // Pop-batch minimum:
        //   - When the user pinned --gpu-batch (shared_tune->locked), honour
        //     it: wait for the full batch.  The pop still returns early at
        //     end-of-queue (pop_batch_greedy detects `done_` and takes
        //     whatever remains), so there's no deadlock — but during steady
        //     state, the GPU sees the batch size it asked for instead of
        //     getting tiny batches that defeat the user's intent.
        //   - When auto-tuning (unlocked), use a soft minimum of 4 frames.
        //     Don't block for the full batch (serializes multiple GPUs
        //     behind the reader); don't go as low as 1 either (tiny
        //     batches waste H2D/D2H overhead and poison the auto-tuner's
        //     throughput measurement).
        const bool locked_batch = shared_tune && shared_tune->locked.load();
        // gpu-only has no CPU relief valve for the in-order writer's
        // head-of-line frame: hybrid's CPU pool pops one low-seq frame at a
        // time, but rescue threads only fire on GPU failure.  A locked
        // full-batch wait in gpu-only lets streams sequester every throttle
        // permit (the budget is sized FROM device×streams×batch, so GPU
        // demand can drain the whole pool via the upfront acquire(pop_n)),
        // and the writer then wedges behind a head-of-line frame no stream
        // can pop — CONFIRMED on the 8-GPU server: -d --gpu-only
        // --gpu-batch=64 --gpu-streams=16 hangs at ~74% (hybrid completes).
        // So in gpu-only, treat --gpu-batch as a CAP, not a hard floor:
        // reuse the unlocked soft minimum so streams grab whatever is
        // queued and release the excess permits (handled just below)
        // instead of blocking on a full batch.  Hybrid keeps the honored
        // full-batch wait — its CPU pool prevents the wedge.
        const size_t decomp_min_batch = (locked_batch && !opt.gpu_only)
            ? pop_n
            : std::min<size_t>(pop_n, 4);
        // Acquire frame permits BEFORE gpu_wants_data to avoid deadlock.
        if (bp) bp->acquire((int)pop_n);
        // Signal scheduler: this GPU stream wants data (blocks CPU workers)
        if (sched) sched->gpu_wants_data();
        if (!queue->pop_batch_greedy(pop_n, C.batch, decomp_min_batch)) {
          if (sched) sched->gpu_got_data();
          if (bp) bp->release((int)pop_n);  // release unused permits
          if (g_perf) {
            g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0);
            g_perf->queue_wait_count.fetch_add(1);
          }
          producer_done_seen = true;
          continue;
        }
        // Signal scheduler: this GPU stream got its data (unblocks CPU workers)
        if (sched) sched->gpu_got_data();
        if (C.batch.empty()) {
          if (bp) bp->release((int)pop_n);  // release unused permits
          if (g_perf) {
            g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0);
            g_perf->queue_wait_count.fetch_add(1);
          }
        }
        // Release excess permits if we got fewer frames than requested
        if (bp && !C.batch.empty() && (int)C.batch.size() < (int)pop_n)
          bp->release((int)pop_n - (int)C.batch.size());
        if (C.batch.empty()) continue;

        // In hybrid mode, skip batches where every frame is trivially
        // compressed (ratio < 2%).  CPU decompresses these faster than GPU
        // because it avoids PCIe D2H overhead.  More importantly, GPU
        // batches of trivial frames cause head-of-line blocking: the GPU
        // holds early-sequence frames for ~200ms while CPU fills all
        // throttle permits waiting for the writer, freezing the pipeline.
        if (sched) {
          bool all_trivial = true;
          for (const auto & t : C.batch) {
            double ratio = (t.decomp_size > 0)
                ? double(t.len()) / double(t.decomp_size) : 1.0;
            if (ratio >= 0.02) { all_trivial = false; break; }
          }
          if (all_trivial) {
            if (opt.verbosity >= V_DEBUG)
              vlog(V_DEBUG, opt, "[GPU" + std::to_string(device_id)
                   + "/S" + std::to_string(C.stream_index)
                   + "] skipping trivial batch (" + std::to_string(C.batch.size())
                   + " frames, ratio<2%) -> re-enqueue for CPU\n");
            int held = (int)C.batch.size();
            queue->re_enqueue(C.batch);
            if (bp) bp->release(held);
            ++trivial_skip_streak;
            continue;
          }
          trivial_skip_streak = 0;  // reset on any non-trivial batch
        }

        if (g_perf) {
          g_perf->sched_gpu_tasks.fetch_add(C.batch.size());
        }
        if (sched) sched->mark_gpu_take(C.batch.size());
        C.filled = C.batch.size();
        uint64_t batch_t0 = now_ns();  // always record for auto-tuner (not just -vvv)

        // Determine max sizes for this batch
        size_t max_comp = 0, max_decomp = 0;
        for (size_t i = 0; i < C.filled; ++i) {
          max_comp   = std::max(max_comp,   C.batch[i].len());
          max_decomp = std::max(max_decomp, C.batch[i].decomp_size);
        }

        if (opt.verbosity >= V_DEBUG) {
          char in_s[32];
          uint64_t tin = 0;
          for (size_t i = 0; i < C.filled; ++i) tin += C.batch[i].len();
          human_bytes(double(tin), in_s, sizeof(in_s));
          size_t seq_lo = C.batch.front().seq;
          size_t seq_hi = C.batch.back().seq;
          std::ostringstream os;
          os << "[GPU" << device_id << "/S" << C.stream_index
             << "] take batch=" << C.filled
             << " seq=[" << seq_lo << ".." << seq_hi << "]"
             << " in=" << in_s;
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        // Get temp workspace size for this batch configuration.
        // We need a dummy ensure_buffers first if sizes grew, then query temp.
        // Use a two-pass approach: ensure base buffers, query temp, re-ensure with temp.
        if (!C.ensure_buffers(C.filled, max_comp, max_decomp, C.temp_bytes)) {
          throw std::runtime_error("GPU decomp: failed to allocate device buffers");
        }

        // (Re-)allocate pinned host D2H staging buffer if the per-batch
        // decompressed footprint grew.  Pinned -> pageable memcpy is faster
        // than direct device -> pageable for the decompressed results, and
        // future async-D2H overlap optimizations can build on this.
        // Honors the global --pinned budget; falls back to no pinned buffer.
        {
          size_t want = C.alloc_batch * C.alloc_decomp;
          if (want > 0 && want != C.h_decomp_pinned_bytes) {
            if (C.h_decomp_pinned) {
              cudaFreeHost(C.h_decomp_pinned);
              release_pinned(C.h_decomp_pinned_bytes, opt);
              C.h_decomp_pinned = nullptr;
              C.h_decomp_pinned_bytes = 0;
            }
            if (try_reserve_pinned(want, opt)) {
              if (cudaHostAlloc(&C.h_decomp_pinned, want, cudaHostAllocDefault)
                  != cudaSuccess) {
                C.h_decomp_pinned = nullptr;
                release_pinned(want, opt);
              } else {
                C.h_decomp_pinned_bytes = want;
                if (opt.verbosity >= V_VERBOSE) {
                  char sz[32]; human_bytes(double(want), sz, sizeof(sz));
                  vlog(V_VERBOSE, opt, std::string("[PINNED] D2H ") + sz
                       + " reserved (decompress)\n");
                }
              }
            }
          }
        }

        // Upload compressed data H2D (async into CUDA stream).
        // Per-frame cudaMemcpyAsync is efficient because CUDA batches them
        // internally in the stream  no host-side blocking between transfers.
        // A single large memcpy would require packing a contiguous host buffer
        // first, which is slower than letting CUDA handle the scatter.
        uint64_t h2d_t0 = (g_perf || opt.verbosity >= V_DEBUG) ? now_ns() : 0;
        uint64_t h2d_bytes_batch = 0;
        cudaEventRecord(C.ev_begin, C.stream);
        for (size_t i = 0; i < C.filled; ++i) {
          void * d_dst = static_cast<char*>(C.d_comp_buf) + i * C.alloc_comp;
          checkCuda(cudaMemcpyAsync(d_dst, C.batch[i].ptr(),
                                    C.batch[i].len(),
                                    cudaMemcpyHostToDevice, C.stream),
                    "cudaMemcpyAsync(H2D decomp)");
          h2d_bytes_batch += C.batch[i].len();
          C.h_comp_sizes[i]   = C.batch[i].len();
          C.h_decomp_sizes[i] = C.batch[i].decomp_size;
        }

        // Upload size arrays (async)
        checkCuda(cudaMemcpyAsync(C.d_comp_sizes, C.h_comp_sizes.data(),
                                  C.filled * sizeof(size_t),
                                  cudaMemcpyHostToDevice, C.stream), "H2D sizes");
        checkCuda(cudaMemcpyAsync(C.d_decomp_sizes, C.h_decomp_sizes.data(),
                                  C.filled * sizeof(size_t),
                                  cudaMemcpyHostToDevice, C.stream), "H2D sizes");

        // Compute total uncompressed bytes for this batch
        size_t total_decomp = 0;
        for (size_t i = 0; i < C.filled; ++i)
          total_decomp += C.h_decomp_sizes[i];

        // Get temp workspace size  sync is required by nvCOMP API
        nvcompBatchedZstdDecompressOpts_t decomp_opts{};
        size_t needed_temp = 0;
        checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize(pre-temp)");
        uint64_t h2d_elapsed_ns = (h2d_t0 > 0) ? now_ns() - h2d_t0 : 0;
        double h2d_ms_v = double(h2d_elapsed_ns) / 1e6;
        if (g_perf) {
          g_perf->h2d_ns.fetch_add(h2d_elapsed_ns);
          g_perf->h2d_bytes.fetch_add(h2d_bytes_batch);
          g_perf->h2d_count.fetch_add(1);
        }
        nvcompStatus_t tst = nvcompBatchedZstdDecompressGetTempSizeSync(
            (const void * const *)C.d_comp_ptrs,
            C.d_comp_sizes,
            C.filled,
            max_decomp,
            &needed_temp,
            total_decomp,
            decomp_opts,
            C.d_statuses,
            C.stream);
        if (tst != nvcompSuccess)
          throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSizeSync failed (status "
                                   + std::to_string(int(tst)) + ")");

        // Re-ensure buffers if temp grew
        if (needed_temp > C.temp_bytes) {
          if (!C.ensure_buffers(C.filled, max_comp, max_decomp, needed_temp)) {
            // Allocation failed  free VRAM reserve and retry
            if (vram_reserve) {
              vlog(V_VERBOSE, opt, "[GPU" + std::to_string(device_id)
                   + "] freeing VRAM reserve (" + std::to_string(vram_reserve_bytes / ONE_MIB)
                   + " MiB) to grow temp buffer\n");
              cudaFree(vram_reserve);
              vram_reserve = nullptr;
              vram_reserve_bytes = 0;
              if (!C.ensure_buffers(C.filled, max_comp, max_decomp, needed_temp))
                throw std::runtime_error("GPU decomp: failed to grow temp buffer (even after freeing reserve)");
            } else {
              throw std::runtime_error("GPU decomp: failed to grow temp buffer");
            }
          }
          // Re-upload pointer/size arrays since ensure_buffers may have reallocated
          for (size_t i = 0; i < C.filled; ++i) {
            void * d_dst = static_cast<char*>(C.d_comp_buf) + i * C.alloc_comp;
            checkCuda(cudaMemcpyAsync(d_dst, C.batch[i].ptr(),
                                      C.batch[i].len(),
                                      cudaMemcpyHostToDevice, C.stream),
                      "cudaMemcpyAsync(H2D decomp re-upload)");
          }
          checkCuda(cudaMemcpyAsync(C.d_comp_sizes, C.h_comp_sizes.data(),
                                    C.filled * sizeof(size_t),
                                    cudaMemcpyHostToDevice, C.stream), "H2D sizes");
          checkCuda(cudaMemcpyAsync(C.d_decomp_sizes, C.h_decomp_sizes.data(),
                                    C.filled * sizeof(size_t),
                                    cudaMemcpyHostToDevice, C.stream), "H2D sizes");
          checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize(re-upload)");
        }

        // Launch batched decompression
        uint64_t kern_t0 = (g_perf || opt.verbosity >= V_DEBUG) ? now_ns() : 0;

        // Save per-frame metadata needed by completion paths.
        //
        // Do NOT release the host-side compressed inputs here ("it's on the
        // GPU now"): if the kernel or a per-chunk status fails AFTER the
        // release, the catch block re-enqueues tasks whose data is gone —
        // the retry then decompresses 0 bytes and the rescue path is dead on
        // arrival.  Inputs are released per-frame after successful delivery
        // below (compressed frames are small, so holding them for the
        // kernel's duration costs little RAM).  v0.13.54.
        std::vector<size_t> batch_comp_sizes(C.filled);
        std::vector<size_t> batch_seqs(C.filled);
        for (size_t i = 0; i < C.filled; ++i) {
          batch_comp_sizes[i] = C.batch[i].len();
          batch_seqs[i] = C.batch[i].seq;
        }
        C.delivered = 0;

        checkNvcomp(nvcompBatchedZstdDecompressAsync(
            (const void * const *)C.d_comp_ptrs,
            C.d_comp_sizes,
            C.d_decomp_sizes,
            C.d_actual_sizes,
            C.filled,
            C.d_temp,
            C.temp_bytes,
            (void * const *)C.d_decomp_ptrs,
            decomp_opts,
            C.d_statuses,
            C.stream), "nvcompBatchedZstdDecompressAsync");

        cudaEventRecord(C.ev_end, C.stream);

        // Synchronize and read back results
        checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize(decomp)");
        uint64_t kern_elapsed_ns = (kern_t0 > 0) ? now_ns() - kern_t0 : 0;
        double comp_ms_v = double(kern_elapsed_ns) / 1e6;
        if (g_perf) {
          g_perf->kernel_ns.fetch_add(kern_elapsed_ns);
          g_perf->kernel_count.fetch_add(1);
        }

        // Read back statuses and actual sizes in bulk
        uint64_t d2h_t0 = (g_perf || opt.verbosity >= V_DEBUG) ? now_ns() : 0;
        uint64_t d2h_bytes_batch = 0;
        checkCuda(cudaMemcpy(C.h_statuses.data(), C.d_statuses,
                             C.filled * sizeof(nvcompStatus_t),
                             cudaMemcpyDeviceToHost), "D2H statuses");
        checkCuda(cudaMemcpy(C.h_actual.data(), C.d_actual_sizes,
                             C.filled * sizeof(size_t),
                             cudaMemcpyDeviceToHost), "D2H actual sizes");

        float batch_ms = 0;
        cudaEventElapsedTime(&batch_ms, C.ev_begin, C.ev_end);
        uint64_t out_sum = 0;

        // Download decompressed frames one at a time and deliver to writer
        // immediately.  Per-frame D2H has slightly more DMA overhead than a
        // single batch transfer, but it keeps the writer busy in parallel 
        // the writer can write frame 0 to disk while frame 1 is transferring.
        // A single batch D2H would stall the writer for the entire transfer.
        for (size_t i = 0; i < C.filled; ++i) {
          if (C.h_statuses[i] != nvcompSuccess)
            throw std::runtime_error("nvCOMP decompress per-chunk status != success");

          size_t actual = C.h_actual[i];
          // Integrity check: nvCOMP can report success yet produce a size that
          // disagrees with the frame header's declared content size (corrupt
          // input, or a transient device fault).  The CPU path gets this for
          // free — ZSTD_decompressDCtx rejects a frame whose output != declared
          // size — so without this guard the GPU path silently writes a short
          // /wrong frame and exits 0.  Throwing re-enqueues the undelivered tail
          // to the main queue (catch block); a CPU worker then re-decodes it and
          // either succeeds (transient glitch — run completes correctly) or dies
          // cleanly with a zstd data error, matching CPU-only and stock zstd.
          if (actual != C.batch[i].decomp_size)
            throw std::runtime_error(
                "GPU decompress size mismatch at frame "
                + std::to_string(batch_seqs[i]) + ": header declared "
                + std::to_string(C.batch[i].decomp_size)
                + " bytes, nvCOMP produced " + std::to_string(actual)
                + " (corrupt input?)");
          out_sum += actual;
          d2h_bytes_batch += actual;

          // Recycle a host buffer from this stream's pool rather than a fresh
          // per-frame allocation (ROADMAP 7.2).  Cap at two batches' worth so
          // the batch now completing and the previous one still draining at the
          // writer both fit without stalling.
          size_t out_pool_cap = std::max<size_t>(2, C.alloc_batch) * 2;
          auto h_out = C.acquire_out_buf(out_pool_cap, bp);
          const void * d_src = static_cast<char*>(C.d_decomp_buf) + i * C.alloc_decomp;
          if (C.h_decomp_pinned) {
            // Device -> pinned host slot, then copy pinned -> output vector.
            // Pinned cudaMemcpy uses a faster DMA path than pageable.
            void * pin_slot = static_cast<char*>(C.h_decomp_pinned)
                              + i * C.alloc_decomp;
            checkCuda(cudaMemcpy(pin_slot, d_src, actual,
                                 cudaMemcpyDeviceToHost), "D2H decomp pinned");
            // assign() copies from the pinned slot without the resize() zero-fill
            // the copy would immediately overwrite — and here `actual` is a FULL
            // decompressed frame (~16 MiB), so the saved memset is far from tiny
            // when the pool buffer has to grow (variable-frame-size input).
            h_out->assign(static_cast<char*>(pin_slot),
                          static_cast<char*>(pin_slot) + actual);
          } else {
            h_out->resize(actual);  // direct D2H needs the dst pre-sized
            checkCuda(cudaMemcpy(h_out->data(), d_src, actual,
                                 cudaMemcpyDeviceToHost), "D2H decomp data");
          }

          uint64_t rl_t0 = g_perf ? now_ns() : 0;
          results->push_to_slot(slot_index, batch_seqs[i], std::move(h_out));
          if (g_perf) g_perf->result_lock_ns.fetch_add(now_ns() - rl_t0);
          // Delivered: safe to free this frame's compressed input now.  A
          // mid-loop throw (bad status / failed D2H) re-enqueues only the
          // undelivered tail, whose inputs are still intact.
          C.batch[i].release_input();
          C.delivered = i + 1;
        }
        uint64_t d2h_elapsed_ns = (d2h_t0 > 0) ? now_ns() - d2h_t0 : 0;
        double d2h_ms_v = double(d2h_elapsed_ns) / 1e6;
        if (g_perf) {
          g_perf->d2h_ns.fetch_add(d2h_elapsed_ns);
          g_perf->d2h_bytes.fetch_add(d2h_bytes_batch);
          g_perf->d2h_count.fetch_add(1);
          g_perf->gpu_batch_ns.fetch_add(now_ns() - batch_t0);
          g_perf->gpu_batch_count.fetch_add(1);
        }

        // Track compressed bytes consumed for progress bar
        {
          uint64_t in_sum = 0;
          for (size_t i = 0; i < C.filled; ++i)
            in_sum += batch_comp_sizes[i];
          if (m) m->read_bytes.fetch_add(in_sum, std::memory_order_relaxed);
          // Report to shared auto-tuner
          if (shared_tune && !shared_tune->locked.load()) {
            shared_tune->window_bytes.fetch_add(in_sum, std::memory_order_relaxed);
            shared_tune->window_ns.fetch_add(now_ns() - batch_t0, std::memory_order_relaxed);
            shared_tune->window_batches.fetch_add(1, std::memory_order_relaxed);
          }
          // Report to rate-matcher and reset CPU cycle
          if (rate_match) {
            rate_match->report_gpu(out_sum, now_ns() - batch_t0);
            rate_match->reset_cycle();
            size_t avg_frame = (C.filled > 0) ? out_sum / C.filled : 16 * 1024 * 1024;
            size_t batch_sz = shared_tune ? shared_tune->batch_size.load() : C.filled;
            rate_match->update(batch_sz, avg_frame);
          }
        }

        if (sched) sched->add_gpu_bytes(out_sum);

        if (opt.verbosity >= V_DEBUG) {
          uint64_t in_sum_v = 0;
          for (size_t i = 0; i < C.filled; ++i) in_sum_v += batch_comp_sizes[i];
          char in_s[32], out_s[32];
          human_bytes(double(in_sum_v), in_s, sizeof(in_s));
          human_bytes(double(out_sum), out_s, sizeof(out_s));
          double tot_ms_v = double(now_ns() - batch_t0) / 1e6;
          double thr_gib = (tot_ms_v > 0.0)
                           ? double(out_sum) / (tot_ms_v / 1000.0) / 1e9 : 0.0;
          std::ostringstream os;
          os << "[GPU" << device_id << "/S" << C.stream_index
             << "] done batch=" << C.filled
             << " in=" << in_s << " out=" << out_s
             << " h2d=" << std::fixed << std::setprecision(2) << h2d_ms_v
             << "ms comp=" << comp_ms_v
             << "ms d2h=" << d2h_ms_v
             << "ms tot=" << tot_ms_v
             << "ms thr=" << thr_gib << " GiB/s";
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        // Per-chunk trace at -vvv.  v0.12.32 grew batches up to 256, so
        // the per-batch `[GPU/S] done batch=N` lines fire ~30x less often
        // than before.  Emit a per-chunk line at V_TRACE so the volume of
        // -vvv output actually matches the "trace" name.
        if (opt.verbosity >= V_TRACE) {
          for (size_t i = 0; i < C.filled; ++i) {
            char cs[32], ds[32];
            human_bytes(double(batch_comp_sizes[i]), cs, sizeof(cs));
            human_bytes(double(C.batch[i].decomp_size), ds, sizeof(ds));
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << C.stream_index
               << "] chunk seq=" << C.batch[i].seq
               << " comp=" << cs << " decomp=" << ds;
            vlog(V_TRACE, opt, os.str() + "\n");
          }
        }

        // Buffers are reused  no per-batch free needed
        C.busy = false;
        C.filled = 0;
        C.batch.clear();
        submitted_any = true;

        // Notify writer that a full batch of frames is now available
        results->cv.notify_one();

        // Update utilization scale for next batch
#ifdef HAVE_NVML
        {
          nvmlDevice_t dev;
          nvmlUtilization_t util;
          if (nvmlDeviceGetHandleByIndex(device_id, &dev) == NVML_SUCCESS &&
              nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
            util_scale = std::max(0.05, (100.0 - util.gpu) / 100.0);
          }
        }
#endif
      }

      // Exit if remaining workload is all trivially-compressed frames.
      // GPU is a net negative on that data (PCIe D2H overhead dominates,
      // and HOL-blocks the writer by holding early-sequence frames).
      // Let CPU workers drain the queue alone.
      if (trivial_skip_streak >= trivial_skip_exit_threshold) {
        if (opt.verbosity >= V_VERBOSE)
          vlog(V_VERBOSE, opt, "[GPU" + std::to_string(device_id)
               + "] all remaining work is trivially compressed; exiting to let CPU drain\n");
        break;
      }

      // Check termination
      if (producer_done_seen) {
        bool all_idle = true;
        for (auto & C : ctxs)
          if (C.busy) { all_idle = false; break; }
        if (all_idle) break;
      }
      if (!submitted_any) std::this_thread::yield();
    }

    // Cleanup
    if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
    for (auto & C : ctxs) {
      C.free_device();
      size_t freed = C.free_host_pinned();
      if (freed) release_pinned(freed, opt);
      if (C.ev_begin) cudaEventDestroy(C.ev_begin);
      if (C.ev_end)   cudaEventDestroy(C.ev_end);
      if (C.stream)   cudaStreamDestroy(C.stream);
    }

    // Deregister streams so queue floor drops and CPU workers aren't blocked
    if (sched) {
      for (size_t s = 0; s < ctxs.size(); ++s)
        sched->unregister_gpu_stream();
    }

    // Wake all CPU workers (see gpu_worker for rationale)
    queue->notify_cpu_waiters();
  }
  catch (const std::exception & e) {
    *any_gpu_failed = true;
    *fatal_msg = std::string("[GPU") + std::to_string(device_id) + "] " + e.what();

    // Rescue in-flight chunks back to queue so other GPUs can pick them up.
    // The batch was popped from the queue but not (fully) decompressed.
    try {
      if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
      for (auto & C : ctxs) {
        // Frames [0, delivered) already reached the ResultStore (and had
        // their inputs released) — re-enqueueing them would push empty,
        // duplicate-seq tasks.  Re-enqueue only the undelivered tail.
        if (C.delivered > 0 && C.delivered <= C.batch.size())
          C.batch.erase(C.batch.begin(),
                        C.batch.begin() + (ptrdiff_t)C.delivered);
        if (!C.batch.empty()) {
          // Capture the permit count before re_enqueue clears the batch.  The
          // next popper re-acquires one permit per frame, so release the GPU's
          // originals here or they leak (mirrors the in-loop re_enqueue at the
          // VRAM-retry path, which already releases).  The writer releases the
          // delivered frames' permits as it writes them.
          const int held = (int)C.batch.size();
          queue->re_enqueue(C.batch);
          if (bp) bp->release(held);
        }
        C.free_device();
        size_t freed = C.free_host_pinned();
        if (freed) release_pinned(freed, opt);
        if (C.ev_begin) { cudaEventDestroy(C.ev_begin); C.ev_begin = nullptr; }
        if (C.ev_end)   { cudaEventDestroy(C.ev_end);   C.ev_end = nullptr; }
        if (C.stream)   { cudaStreamDestroy(C.stream);   C.stream = nullptr; }
      }
    } catch (...) {}

    // Deregister streams so queue floor drops and CPU workers aren't blocked
    if (sched) {
      for (size_t s = 0; s < ctxs.size(); ++s)
        sched->unregister_gpu_stream();
    }

    {
      std::lock_guard<std::mutex> lk(results->m);
      results->cv.notify_all();
    }
    queue->notify_cpu_waiters();

    // Last GPU just failed in --gpu-only: finish decompression on CPU so the
    // re-enqueued frames (and everything still in the queue) have a consumer.
    int fails = gpu_failures
        ? gpu_failures->fetch_add(1, std::memory_order_acq_rel) + 1 : 1;
    if (opt.gpu_only && fails == gpu_worker_count)
      gpu_only_cpu_fallback(true, queue, results, opt, m, bp);
  }
}

/*======================================================================
 GPU/Hybrid decompression entry point
 -----------------------------------------------------------------------
 Launches GPU + CPU workers first, then streams frames into the queue.
 Workers begin decompressing as soon as the first frame arrives.
 Falls back to streaming CPU if frame sizes can't be determined.
======================================================================*/
static void decompress_nvcomp(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // ---- Performance instrumentation (active at -vvv) ----
  PerfCounters perf_local;
  if (opt.verbosity >= V_TRACE) g_perf = &perf_local;

  // ---- Detect GPU devices ----
  // Note: must use cudaGetDeviceCount (not NVML) because NVML sees all
  // physical devices regardless of CUDA_VISIBLE_DEVICES masking.
  //
  // cudaGetDeviceCount triggers the one-time CUDA driver init (cuInit):
  // ~2s on an 8-GPU box.  In HYBRID mode we defer it to a background
  // "GPU bringup" thread (see below) so the CPU pool can decompress
  // during that init instead of the main thread blocking here.  The
  // throttle is sized from a provisional device count (it's RAM-capped,
  // so over-estimating is harmless); the real count is detected in the
  // bringup thread before any GPU work.  v0.13.13.
  // Background GPU bringup applies only to ADAPTIVE hybrid (cpu_share < 0).
  // Fixed-share (--cpu-share) must keep the synchronous/inline bringup: the
  // GPU has to be warm before the reader starts, or a small input drains
  // entirely to CPU before the GPU registers and the explicit split is
  // silently ignored (the v0.13.11 regression).  In the inline path the CPU
  // pool is spawned but idle until the reader runs, and GPU bringup +
  // warm_gpu_contexts complete first, so the split is honored.
  const bool hybrid_overlap = opt.hybrid && !opt.cpu_only && !opt.gpu_only
                              && opt.cpu_share < 0.0;
  // gpu-only has no CPU pool to overlap cuInit with, but the *reader* is
  // useful cover work that the v0.13.13 restructure overlooked.  Defer the
  // cudaGetDeviceCount cuInit (~2-3s on an 8-GPU box) to the background
  // bringup thread and let the main thread stream frames into the queue
  // meanwhile, so GPU workers consume a warm queue the instant they come
  // online instead of starting cold after a synchronous stall.  v0.13.15.
  const bool gpu_only_overlap = opt.gpu_only;
  const bool defer_detect = hybrid_overlap || gpu_only_overlap;
  int device_count = 0;
  int total_hw_devices = 0;
  if (!defer_detect) {
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      if (opt.gpu_only)
        die_usage("GPU requested (--gpu-only) but no CUDA devices available");
      vlog(V_VERBOSE, opt, "[GPU] no devices found; falling back to MT CPU decompress\n");
      // Fall through with device_count=0; CPU pool will handle everything
    }
    // Apply --gpu-devices limit.  Default (0) = use all available GPUs.
    total_hw_devices = device_count;
    if (device_count > 0) {
      int target = opt.gpu_devices;
      if (target == 0) target = device_count;  // auto: use all GPUs
      if (target < device_count) {
        vlog(V_VERBOSE, opt, "[GPU] using " + std::to_string(target)
             + " of " + std::to_string(device_count) + " GPU devices"
             + (opt.gpu_devices == 0 ? " (auto)\n" : "\n"));
        device_count = target;
      }
    }
  } else {
    // Provisional count for throttle sizing; real detection happens in
    // the background bringup thread.  Generous (RAM-capped downstream).
    device_count = std::max(opt.gpu_devices, 8);
    total_hw_devices = device_count;
  }

  // ---- Shared state ----
  TaskQueue queue;
  RescueQueue rescue;
  ResultStore results;
  std::atomic<bool> any_gpu_failed{false};
  std::atomic<bool> abort_on_failure{ opt.gpu_only };
  // Set by the deferred bringup thread when --gpu-only is requested but no
  // CUDA device is found.  The reader checks it and stops streaming so main
  // can error out cleanly instead of buffering the whole file with no
  // consumer (the synchronous path errored instantly at detection time).
  std::atomic<bool> gpu_only_no_device{false};
  RateMatchState rate_match;

  // ---- Oversize first-frame peek (replaces the v0.12.22-v0.12.36 full
  // pre-scan that ran BEFORE worker spawn).  The pre-scan blocked worker
  // init for tens of seconds on large inputs, so users at -v/-vv/-vvv
  // saw nothing but [SPLIT] frame N lines until parsing finished.  We
  // restore the v0.12.21 "spawn workers first, parse concurrently"
  // architecture by checking only the first frame's decompressed size
  // up front — sufficient to detect zstd / --sliding-window
  // single-frame files (where frame 0 IS the whole file) without
  // touching the rest of the input.
  bool fallback = false;
  bool gpu_disabled_by_peek = false;  // oversize first frame → no GPU for this file
  std::vector<char> raw_data;  // populated by stream_frames_to_queue if it falls back
  int64_t first_frame_decomp = peek_first_frame_decomp_size(in);
  if (first_frame_decomp > (int64_t)GPU_SUBCHUNK_MAX && device_count > 0) {
    char sz[32]; human_bytes(double(first_frame_decomp), sz, sizeof(sz));
    vlog(V_NORMAL, opt,
         std::string("warning: first frame decompresses to ") + sz
         + " (GPU max: 16 MiB).\n"
         "  This file was likely compressed with --sliding-window or zstd.\n"
         "  Falling back to CPU-only decompression.\n");
    if (opt.gpu_only)
      vlog(V_NORMAL, opt, "  (--gpu-only ignored for this file)\n");
    // For hybrid the real device_count isn't known yet (deferred to the
    // bringup thread); record the decision as a flag the bringup thread
    // honors.  For non-hybrid, disable GPU directly.
    if (defer_detect) gpu_disabled_by_peek = true;
    else              device_count = 0;
  }

  // Early init banner for -v/-vv/-vvv users.  In hybrid the real GPU
  // count isn't known yet (detection is deferred to the bringup thread),
  // so report "detecting" rather than the provisional placeholder.
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[INIT] decompress: ";
    if (defer_detect) os << "GPUs detecting in background";
    else              os << device_count << " GPU(s) active";
    os << ", mode=";
    if (opt.gpu_only)      os << "gpu-only";
    else if (opt.cpu_only) os << "cpu-only";
    else if (opt.hybrid)   os << "hybrid";
    else                   os << "auto";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }

  // Frame throttle: scales with pipeline parallelism (see compress_nvcomp).
  // Same auto-tune-aware sizing as compress: when --gpu-batch is not pinned,
  // budget for the auto-tuner's potential growth so GPU pops don't starve
  // on permits when CPUs hold them in hybrid mode.
  const int decomp_cpu_threads_est = (device_count > 0 && opt.gpu_only && !gpu_disabled_by_peek) ? 0 : resolve_cpu_threads(opt.cpu_threads);
  const size_t decomp_per_stream_budget = opt.gpu_batch_user_set
      ? std::max<size_t>(1, opt.gpu_batch_cap)
      : std::max<size_t>(opt.gpu_batch_cap, AUTO_TUNE_BATCH_CEILING);
  const int decomp_gpu_batch_floor = device_count
      * (int)std::max<size_t>(1, opt.gpu_streams)
      * (int)decomp_per_stream_budget;
  const int decomp_parallelism = decomp_cpu_threads_est + decomp_gpu_batch_floor;
  FrameThrottle throttle(compute_throttle_budget(
      std::max<size_t>(1, opt.chunk_mib) * ONE_MIB, decomp_parallelism,
      decomp_gpu_batch_floor, opt));
  FrameThrottle * bp_ptr = (opt.mode == Mode::TEST) ? nullptr : &throttle;

  // Bound queued (read-but-not-popped) frames to pipeline depth so a slow GPU
  // consumer (D2H-bound) can't let the reader buffer the entire compressed
  // input in RAM — the gpu-only RSS blowup in ROADMAP 7.8.  Skip when
  // throttling is explicitly disabled.
  if (opt.throttle_frames != 0) {
    int qslack = opt.throttle_factor > 0 ? opt.throttle_factor : THROTTLE_SLACK_FACTOR;
    const size_t qfloor = (size_t)std::max(THROTTLE_MIN_FRAMES, decomp_parallelism * qslack);
    queue.set_max_depth(qfloor);
    // Byte ceiling — see decompress_cpu_mt: bounds queued RAM independent of
    // compressibility (~8 MiB/slot).  Soft cap; tune via --throttle-factor.
    queue.set_max_bytes(qfloor * (8 * ONE_MIB));
  }

  // ---- Writer thread (outputs decompressed data in order) ----
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m, bp_ptr);

  // ---- Hybrid scheduler ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid && device_count > 0) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, resolve_cpu_threads(opt.cpu_threads),
        device_count, opt);
    sched = sched_ptr.get();
    sched->set_queue(&queue);
    tick_thr = std::thread(tick_loop_fn, std::ref(tick_done), sched);
  }

  // ---- CPU decompression pool ----
  CpuAgg cpuagg{};
  std::vector<std::thread> cpu_pool;
  int cpu_threads = 0;
  // gpu_disabled_by_peek: gpu-only file with an oversize first frame falls
  // back to CPU; without sched (gpu-only has none) we must still spawn a pool.
  if (sched || (device_count <= 0) || gpu_disabled_by_peek) {
    cpu_threads = resolve_cpu_threads(opt.cpu_threads);
    cpuagg.threads = cpu_threads;
    cpuagg.per_thread.resize((size_t)cpu_threads);

    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[HYBRID] decompress: " << cpu_threads << " CPU threads";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
    for (int i = 0; i < cpu_threads; ++i)
      cpu_pool.emplace_back(cpu_decomp_worker, i, &queue, &results, &opt, m,
                            (void*)sched, &rate_match, &cpuagg, bp_ptr);
  }

  // ---- GPU decompression workers ----
  std::vector<int> gpu_ids;
  std::vector<std::thread> gpu_workers;
  std::vector<std::string> fatal_msgs;

  // Shared auto-tune state for decompress GPUs
  SharedTuneState shared_tune_decomp;
  shared_tune_decomp.batch_size.store(opt.gpu_batch_cap);
  shared_tune_decomp.locked.store(opt.gpu_batch_user_set);

  // Counts GPUs that failed terminally (init or mid-run).  When the count
  // reaches the worker count in --gpu-only mode, the last failing worker
  // runs gpu_only_cpu_fallback to finish on CPU (v0.13.54).
  std::atomic<int> gpu_failures{0};

  // GPU bringup: detect (when deferred — triggers the cuInit), select
  // devices, init result slots, and spawn GPU workers.  Runs on a background
  // thread when defer_detect (hybrid adaptive or gpu-only) so the CPU pool
  // (hybrid) or the reader (gpu-only) overlaps cuInit; inline otherwise.
  auto gpu_bringup = [&]() {
    if (defer_detect) {
      // Deferred device detection (the ~2s cuInit, off the critical path).
      int dc = 0;
      if (cudaGetDeviceCount(&dc) != cudaSuccess || dc <= 0) {
        if (opt.gpu_only) {
          // gpu-only with no GPU: tell the reader to stop so main can error
          // cleanly (the synchronous path used to die_usage here).
          gpu_only_no_device.store(true, std::memory_order_release);
          return;
        }
        // No GPU after all — the CPU pool (already running) does everything.
        vlog(V_VERBOSE, opt, "[GPU] no devices found; hybrid running CPU-only\n");
        return;
      }
      total_hw_devices = dc;
      int target = opt.gpu_devices;
      if (target == 0) target = dc;
      device_count = std::min(target, dc);
      if (gpu_disabled_by_peek) {
        vlog(V_VERBOSE, opt, "[GPU] disabled for this file (oversize frame); CPU-only\n");
        return;
      }
      // If the CPU pool already drained the whole file during cuInit, skip
      // GPU spawn entirely — no work left, and pointless context creation
      // would just delay process exit on small inputs.
      if (queue.drained() && queue.size() == 0) {
        vlog(V_VERBOSE, opt, "[GPU] file already decompressed by CPU during init; skipping GPU\n");
        return;
      }
    }
    if (device_count <= 0) return;

    const uint64_t gpu_sel_t0 = now_ns();
    gpu_ids = select_best_gpus(total_hw_devices, device_count, opt);
    if (opt.cpu_share >= 0.0) warm_gpu_contexts(gpu_ids);
    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[GPU] device selection: " << std::fixed << std::setprecision(1)
         << double(now_ns() - gpu_sel_t0) / 1e6 << " ms";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
    const int gpu_count = (int)gpu_ids.size();
    fatal_msgs.resize(gpu_count);
    // init_slots resizes ResultStore::slots, which the writer iterates in
    // drain_slots_locked under results.m — take that lock so the resize
    // can't race a concurrent drain (the writer is already running).
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.init_slots(gpu_count);
    }

    for (int i = 0; i < gpu_count; ++i) {
      gpu_workers.emplace_back(gpu_decomp_worker, gpu_ids[i], i, opt,
                               &queue, &rescue, &results, m, sched,
                               &any_gpu_failed, &abort_on_failure,
                               &fatal_msgs[size_t(i)],
                               &shared_tune_decomp, &rate_match,
                               bp_ptr, &gpu_failures, gpu_count);
    }
    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[GPU] " << (int)gpu_ids.size() << " device(s) online";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  };

  std::thread gpu_bringup_thr;
  if (defer_detect) {
    // Background bringup: main thread proceeds to the reader so cuInit +
    // context creation overlap with the CPU pool (hybrid) or the reader
    // filling the queue (gpu-only).
    gpu_bringup_thr = std::thread(gpu_bringup);
  } else if (device_count > 0) {
    // auto / fixed-share hybrid: bring up inline.
    gpu_bringup();
  }

  // Fixed-share decompress: same barrier as compress (see compress_nvcomp).
  // The inline bringup above spawned GPU workers but didn't wait for them
  // to register; without this, a small input drains to CPU before any GPU
  // registers and the explicit split is ignored.  Only applies to the
  // inline (fixed-share) path — adaptive runs hybrid_overlap with gpu
  // workers spawned on the background thread and promises no exact split.
  if (sched && opt.cpu_share >= 0.0 && !gpu_workers.empty()) {
    while (!sched->any_gpu_active()
           && gpu_failures.load(std::memory_order_acquire) < (int)gpu_workers.size()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // ---- Producer: stream frames into the queue while workers consume ----
  // v0.12.21 architecture: workers are already spawned and waiting on the
  // empty queue; they start decompressing as soon as the first frame is
  // pushed.  This removes the long "init lines invisible until pre-scan
  // finishes" delay that v0.12.22-v0.12.36 introduced.  Oversize detection
  // is handled up front by peek_first_frame_decomp_size.
  size_t max_frame_decomp = 0;
  size_t n_frames = stream_frames_to_queue(in, queue, m, opt, &fallback,
                                           &raw_data, &max_frame_decomp,
                                           &gpu_only_no_device);

  // Deferred bringup found no GPU for a --gpu-only request: error out as the
  // old synchronous path did, now that the reader has stopped streaming.
  if (gpu_only_no_device.load(std::memory_order_acquire)) {
    if (gpu_bringup_thr.joinable()) gpu_bringup_thr.join();
    die_usage("GPU requested (--gpu-only) but no CUDA devices available");
  }

  // Preallocate output file to avoid per-write extent allocation overhead.
#ifndef _WIN32
  if (g_direct_writer && m && n_frames > 0 && opt.preallocate_output) {
    uint64_t total = m->total_out.load(std::memory_order_relaxed);
    if (total > 0 && g_direct_writer->preallocate(total)) {
      char sz[32]; human_bytes(double(total), sz, sizeof(sz));
      vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
    }
  }
#endif

  if (fallback && n_frames == 0) {
    // Could not parse any frames  shut down workers, fall back to streaming
    queue.set_done();
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.producer_done = true;
      results.total_tasks = 0;
      results.workers_done = true;
    }
    results.cv.notify_all();
    throttle.set_done();
    // Join the background bringup thread BEFORE touching gpu_workers: it
    // populates that vector (data race otherwise), and leaving it joinable
    // made this std::thread's destructor call std::terminate — an abort on
    // every hybrid/gpu-only decompress of a streamed-zstd file (v0.13.54).
    if (gpu_bringup_thr.joinable()) gpu_bringup_thr.join();
    for (auto & th : gpu_workers) th.join();
    for (auto & th : cpu_pool) th.join();
    if (sched) { tick_done = true; if (tick_thr.joinable()) tick_thr.join(); }
    writer_thr.join();

    vlog(V_DEFAULT, opt,
         std::string("warning: frame sizes unknown (no content-size headers); "
         "falling back from ")
         + (opt.gpu_only ? "--gpu-only" : opt.hybrid ? "--hybrid" : "parallel")
         + " to CPU streaming decompress.\n"
         "  A single frame of unknown size cannot be split across workers; "
         "the CPU decoder\n  guarantees a complete, correct output — this is "
         "for data safety, just slower.\n");
    decompress_from_buffer(raw_data, out, opt, m);
    return;
  }

  vlog(V_VERBOSE, opt,
       "[READER] streamed " + std::to_string(n_frames) + " frames for GPU/hybrid decompress\n");

  // ---- Signal that all frames have been enqueued ----
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
    if (m) {
      m->total_frames.store(results.total_tasks, std::memory_order_relaxed);
      m->total_out_final.store(true, std::memory_order_release);
    }
  }
  results.cv.notify_all();

  // ---- Teardown ----
  // Join the background bringup thread FIRST: it populates gpu_workers, so
  // we must wait for it before iterating that vector.  (If CPU drained the
  // file during cuInit, the bringup thread may have spawned no GPU workers
  // or spawned ones that immediately hit the done+empty queue and exit.)
  if (gpu_bringup_thr.joinable()) gpu_bringup_thr.join();
  for (auto & th : gpu_workers) th.join();

  // Report GPU failures.  Even when ALL GPUs failed in --gpu-only mode the
  // job is already complete: the last failing worker ran
  // gpu_only_cpu_fallback (with its own warning) and drained the queue on
  // CPU, so the output is complete and correct.  Warn instead of dying.
  if (any_gpu_failed.load()) {
    int failed_count = 0;
    for (const auto & s : fatal_msgs)
      if (!s.empty()) ++failed_count;
    int total_gpus = (int)fatal_msgs.size();
    const char * suffix = (failed_count >= total_gpus && abort_on_failure.load())
        ? " (work completed on CPU)\n"
        : (abort_on_failure.load() ? " (other GPUs continuing)\n"
                                   : " (rescued to CPU/other GPUs)\n");
    for (const auto & s : fatal_msgs)
      if (!s.empty())
        vlog(V_DEFAULT, opt, "WARNING: " + s + suffix);
  }

  rescue.set_done();
  // After all GPUs have exited, ensure CPU workers are unblocked.
  // GPU unregister_gpu_stream already drops the floor and notifies, but
  // re-notify here as a safety net in case any CPU worker missed it.
  queue.notify_cpu_waiters();

  // Do NOT call throttle.set_done() before join: workers must respect
  // throttle while draining the queue to avoid buffering entire output in RAM.
  if (!cpu_pool.empty()) {
    auto t_cpu = std::chrono::steady_clock::now();
    for (auto & th : cpu_pool) th.join();
    if (opt.verbosity >= V_VERBOSE) {
      double ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(
          std::chrono::steady_clock::now() - t_cpu).count();
      if (ms > 500)
        vlog(V_VERBOSE, opt, "[CPU] pool join took " + std::to_string(int(ms)) + " ms\n");
    }
  }
  throttle.set_done();  // safe now: all workers exited

  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();

  if (sched) {
    tick_done = true;
    if (tick_thr.joinable()) tick_thr.join();
  }

  {
    auto t_wr = std::chrono::steady_clock::now();
    writer_thr.join();
    if (opt.verbosity >= V_VERBOSE) {
      double ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(
          std::chrono::steady_clock::now() - t_wr).count();
      if (ms > 500)
        vlog(V_VERBOSE, opt, "[WRITER] join took " + std::to_string(int(ms)) + " ms\n");
    }
  }

  // Mid-file fallback tail (see decompress_cpu_mt): a later frame had no
  // content-size header, so the reader buffered the rest of the input in
  // raw_data.  All parsed frames are written (writer joined); append the
  // tail via the CPU streaming decoder.  Previously dropped silently.
  if (fallback && !raw_data.empty()) {
    vlog(V_DEFAULT, opt,
         "warning: frame " + std::to_string(n_frames) + " onward has no "
         "content-size header (zstd streaming output); the parallel reader "
         "can't split it, so the remaining data is decompressed with the CPU "
         "streaming decoder (slower, but nothing is lost).\n");
    decompress_from_buffer(raw_data, out, opt, m);
  }

  log_throttle_stats(throttle, opt,
                     opt.hybrid ? "decompress-hybrid" :
                     opt.gpu_only ? "decompress-gpu" : "decompress-nvcomp");

  if (g_perf) {
    g_perf->print_summary(opt.hybrid ? "HYBRID DECOMPRESS" :
                          opt.gpu_only ? "GPU-ONLY DECOMPRESS" : "DECOMPRESS");
    g_perf = nullptr;
  }
}

#endif // HAVE_NVCOMP

/*======================================================================
 JSON writers (minimal)
======================================================================*/
#if __cplusplus >= 201703L
[[maybe_unused]]
#endif
static void write_stats_json_cpu_only(const std::string & path, const Options & opt, const Meter & meter, double elapsed_sec, const CpuAgg & cpuagg)
{
  std::ofstream js(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!js) { return; }
  uint64_t in_bytes = meter.read_bytes.load();
  uint64_t out_bytes = meter.wrote_bytes.load();
  const char * mode_str = (opt.mode == Mode::COMPRESS)    ? "compress"
                        : (opt.mode == Mode::DECOMPRESS)  ? "decompress"
                        :                                    "test";

  js << "{\n"
     << "  \"version\": \"" << GZSTD_VERSION << "\",\n"
     << "  \"mode\": \"" << mode_str << "\",\n"
     << "  \"elapsed_sec\": " << std::fixed << std::setprecision(6) << elapsed_sec << ",\n"
     << "  \"input_bytes\": " << in_bytes << ",\n"
     << "  \"output_bytes\": " << out_bytes << ",\n"
     << "  \"cpu\": { \"threads\": " << cpuagg.threads << " }\n"
     << "}\n";
}

/*======================================================================
 Main entry point
 -----------------------------------------------------------------------
 Control flow:
   1. Parse args, open input/output files
   2. Start progress bar thread (decompress/test modes)
   3. Dispatch to compression or decompression path:
      - Compress: compress_nvcomp() [GPU+hybrid] or compress_cpu_mt() [CPU-only]
      - Decompress: decompress_nvcomp() or decompress_cpu_mt()
   4. Join progress thread, print summary, clean up
======================================================================*/
static Options parse_args(int argc, char ** argv);
static void apply_backend_defaults(Options & opt);
static std::string derive_output(const std::string & input, Mode mode);

#ifndef _WIN32
// -d --tar: decompress each archive into a pipe and extract it via
// tarx::Extractor.  Reuses the full parallel decompressor (CPU/GPU) unchanged —
// the pipe is just a FILE* sink like stdout.  Returns an exit code.
static int extract_tar(const Options & opt, Meter * m)
{
  int dest_fd = ::open(opt.tar_dest.c_str(), O_DIRECTORY | O_RDONLY | O_CLOEXEC);
  if (dest_fd < 0)
    die_io("cannot open extraction directory '" + opt.tar_dest + "': " + std::strerror(errno));

  // Progress bar + completion summary.  This path bypasses main()'s
  // decompress progress setup (it returns early via extract_tar), so spawn the
  // same machinery here.  total_in = sum of compressed archive sizes drives the
  // input %; the decompress workers update the shared meter.  Unseekable/stdin
  // archives contribute 0 (unknown size), matching single-file stdin behavior.
  uint64_t total_in = 0;
  for (const std::string & arc : opt.tar_sources) {
    if (arc == "-") continue;
    std::error_code ec; uintmax_t z = fs::file_size(arc, ec);
    if (!ec) total_in += (uint64_t)z;
  }
  std::atomic<bool> prog_done{false};
  std::thread prog_thr(progress_loop, std::cref(opt), m, total_in, &prog_done);

  int rc = EXIT_OK;
  for (const std::string & arc : opt.tar_sources) {
    if (opt.keep_going) { g_damage.reset(); g_member_ranges.reset(); }  // damage reported per archive
    FILE * in = open_input(arc);   // dies on failure
    if (arc != "-") {              // validate zstd magic (skip for unseekable stdin)
      unsigned char magic[4] = {0};
      size_t nr = std::fread(magic, 1, 4, in);
      uint32_t mg = 0; if (nr == 4) std::memcpy(&mg, magic, 4);
      bool ok_magic = (mg == 0xFD2FB528u) || ((mg & 0xFFFFFFF0u) == 0x184D2A50u);
      if (!ok_magic) {
        vlog(V_ERROR, opt, "gzstd: untar: not a zstd archive: " + arc + "\n");
        if (in && in != stdin) std::fclose(in);
        rc = EXIT_DATA; continue;
      }
      std::rewind(in);
    }

    // Feed the decompressed tar stream to the Extractor in memory rather than
    // through a kernel pipe (same FrameSink path as verify, see verify_tar):
    // the decompressors push finished frames straight to the parser, which
    // reads file data out of them and writes it.  Removes the pipe's per-byte
    // kernel copy + syscalls; the parse stays serial and writes stay the
    // ceiling, so this is a tidy efficiency win, not a parallelization (see
    // ROADMAP for parallel-dispatch extract).  The budget bounds RAM when the
    // (write-bound) Extractor falls behind the decompressors.
    FrameSink sink(256 * ONE_MIB);
    g_tar_decomp_sink = &sink;

    // Pass no meter to the Extractor: the decompressor already counts the tar
    // stream it produces as wrote_bytes, and the Extractor writes that same
    // stream's file data to disk — counting both double-counts the output.
    tarx::Extractor ex(dest_fd, opt, /*meter=*/nullptr);
    std::thread exth([&] { ex.run_sink(&sink); });

    Options dopt = opt;
    dopt.sparse_mode = 0;
    dopt.unsafe_overwrite = true;

    // out is null: every output route is redirected to the sink while
    // g_tar_decomp_sink is set, so the FILE* is never touched.
    int64_t ff = (arc != "-") ? peek_first_frame_decomp_size(in) : -1;
    if (ff > (int64_t)SINGLE_FRAME_STREAM_MIN) {
      decompress_stream_from_file(in, nullptr, dopt, m);
    } else {
#ifdef HAVE_NVCOMP
      if (dopt.cpu_only) decompress_cpu_mt(in, nullptr, dopt, m);
      else               decompress_nvcomp(in, nullptr, dopt, m);
#else
      decompress_cpu_mt(in, nullptr, dopt, m);
#endif
    }
    sink.close();          // signal EOF to the Extractor after all frames pushed
    exth.join();
    g_tar_decomp_sink = nullptr;
    if (opt.verbosity >= V_DEBUG) {
      char b[200];
      std::snprintf(b, sizeof(b),
        "[SINK] producer waited %.0fms (extract-bound) | consumer waited %.0fms (decompress-bound)\n",
        sink.producer_wait_ns() / 1e6, sink.consumer_wait_ns() / 1e6);
      vlog(V_DEBUG, opt, b);
    }
    if (in && in != stdin) std::fclose(in);
    if (ex.had_error() && rc == EXIT_OK) rc = EXIT_ERROR;
    if (opt.keep_going) {
      int sev = report_keep_going_damage_tar(opt);   // names damaged files; 6/7 or OK
      if (sev > rc) rc = sev;                         // recovery verdict outranks a plain error
    }
  }
  ::close(dest_fd);

  // Stop the progress bar and print a zstd-style decompress summary (mirrors
  // main()'s DECOMPRESS summary).  in = total compressed read, out = total
  // extracted; the "output" label is the extraction directory.
  prog_done = true;
  if (prog_thr.joinable()) prog_thr.join();
  if (opt.verbosity >= V_DEFAULT) {
    uint64_t in_bytes  = m->read_bytes.load();
    uint64_t out_bytes = m->wrote_bytes.load();
    auto dt = std::chrono::steady_clock::now() - m->t0;
    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
    double rate = secs > 0 ? double(out_bytes) / secs : 0.0;
    char in_s[64], out_s[64], rate_s[64];
    human_bytes(double(in_bytes),  in_s,  sizeof(in_s));
    human_bytes(double(out_bytes), out_s, sizeof(out_s));
    human_bytes(rate, rate_s, sizeof(rate_s));
    std::string in_name = (opt.tar_sources.size() == 1)
        ? opt.tar_sources[0]
        : (std::to_string(opt.tar_sources.size()) + " archives");
    char summary[512];
    std::snprintf(summary, sizeof(summary),
      "%s : \033[36m%s\033[0m => \033[32m%s\033[0m, %s @ \033[32m%s/s\033[0m",
      in_name.c_str(), in_s, out_s, opt.tar_dest.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }
  return rc;
}

// -t --tar: decompress each archive and STRUCTURALLY validate the tar inside it
// (every header checksum, member sizes, completeness) without writing any files.
// Plain `-t` only proves the zstd stream decompresses; this also catches a
// truncated or corrupt tar wrapped in an intact zstd stream.  zstd already
// guarantees byte integrity end-to-end, and tar has no per-file data checksums,
// so this verifies "structurally valid, complete, would extract cleanly."
static int verify_tar(const Options & opt, Meter * m)
{
  // Progress bar (same machinery as extract_tar): total_in = sum of compressed
  // archive sizes drives the input %, fed by the decompressor's meter updates.
  uint64_t total_in = 0;
  for (const std::string & arc : opt.tar_sources) {
    if (arc == "-") continue;
    std::error_code ec; uintmax_t z = fs::file_size(arc, ec);
    if (!ec) total_in += (uint64_t)z;
  }
  std::atomic<bool> prog_done{false};
  std::thread prog_thr(progress_loop, std::cref(opt), m, total_in, &prog_done);

  // Result lines are collected and printed AFTER the progress thread stops, so
  // they don't collide with the bar.
  struct VRes { std::string name; bool bad; uint64_t files, bytes, comp; double secs; };
  std::vector<VRes> vres;

  int rc = EXIT_OK;
  for (const std::string & arc : opt.tar_sources) {
    FILE * in = open_input(arc);   // dies on failure
    if (arc != "-") {
      unsigned char magic[4] = {0};
      size_t nr = std::fread(magic, 1, 4, in);
      uint32_t mg = 0; if (nr == 4) std::memcpy(&mg, magic, 4);
      bool ok_magic = (mg == 0xFD2FB528u) || ((mg & 0xFFFFFFF0u) == 0x184D2A50u);
      if (!ok_magic) {
        vlog(V_ERROR, opt, "gzstd: test: not a zstd archive: " + arc + "\n");
        if (in && in != stdin) std::fclose(in);
        rc = EXIT_DATA; continue;
      }
      std::rewind(in);
    }

    // Hand the decompressed tar stream to the validator in memory instead of
    // through a kernel pipe: the validator skips over each member's data by
    // pointer arithmetic (FrameSink + StreamReader's sink mode) rather than
    // having every byte copied through a pipe and re-scanned.  This makes verify
    // decompress-bound (like plain -t) instead of single-pipe-drain bound.  The
    // budget bounds RAM if the validator ever falls behind the decompressors.
    FrameSink sink(256 * ONE_MIB);
    g_tar_decomp_sink = &sink;

    tarx::Extractor ex(-1, opt, /*meter=*/nullptr, /*validate_only=*/true);
    std::thread exth([&] { ex.run_sink(&sink); });

    Options dopt = opt;
    dopt.sparse_mode = 0;
    dopt.unsafe_overwrite = true;
    dopt.mode = Mode::DECOMPRESS;  // produce the tar stream (routed to the sink by
                                   // writer_thread / the streaming helpers); the
                                   // validate-only Extractor gives the test
                                   // semantics.  TEST mode would discard output.

    // A corrupt zstd stream makes the decompressor die_data(); that is itself a
    // test failure, surfaced as exit 4 — the desired behavior for `-t`.  out is
    // null: every output route is redirected to the sink while g_tar_decomp_sink
    // is set, so the FILE* is never touched.
    auto t_arc0 = std::chrono::steady_clock::now();
    int64_t ff = (arc != "-") ? peek_first_frame_decomp_size(in) : -1;
    if (ff > (int64_t)SINGLE_FRAME_STREAM_MIN) {
      decompress_stream_from_file(in, nullptr, dopt, m);
    } else {
#ifdef HAVE_NVCOMP
      if (dopt.cpu_only) decompress_cpu_mt(in, nullptr, dopt, m);
      else               decompress_nvcomp(in, nullptr, dopt, m);
#else
      decompress_cpu_mt(in, nullptr, dopt, m);
#endif
    }
    sink.close();          // idempotent: writer_thread also closes on normal exit
    exth.join();
    g_tar_decomp_sink = nullptr;
    if (opt.verbosity >= V_DEBUG) {
      char b[200];
      std::snprintf(b, sizeof(b),
        "[SINK] producer waited %.0fms (verify-bound) | consumer waited %.0fms (decompress-bound)\n",
        sink.producer_wait_ns() / 1e6, sink.consumer_wait_ns() / 1e6);
      vlog(V_DEBUG, opt, b);
    }
    if (in && in != stdin) std::fclose(in);

    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t_arc0).count();
    bool bad = ex.had_error();
    if (bad && rc == EXIT_OK) rc = EXIT_DATA;
    uint64_t comp = 0;
    if (arc != "-") {
      std::error_code ec; uintmax_t z = fs::file_size(arc, ec);
      if (!ec) comp = (uint64_t)z;
    }
    vres.push_back({arc == "-" ? "(stdin)" : arc, bad,
                    ex.validated_files(), ex.validated_bytes(), comp, secs});
  }

  // Stop the progress bar, then print the per-archive results on clean lines.
  prog_done = true;
  if (prog_thr.joinable()) prog_thr.join();
  if (opt.verbosity >= V_DEFAULT) {
    std::fprintf(stderr, "\r\033[K");  // clear any leftover progress line
    for (const VRes & v : vres) {
      char sz[64]; human_bytes(double(v.bytes), sz, sizeof(sz));
      char rate[64];
      human_bytes(v.secs > 0 ? double(v.bytes) / v.secs : 0.0, rate, sizeof(rate));
      const char * verdict = v.bad ? "tar structure invalid" : "tar structure valid";
      // Numbers colored to match the plain -t summary line: in=cyan, out=green,
      // ratio=bold, rate=green; entry count bold; OK bold-bright-green.
      const char * status  = v.bad ? "\033[31mCORRUPT\033[0m" : "\033[1;92mOK\033[0m";
      if (v.comp > 0 && v.bytes > 0) {
        char comp[64]; human_bytes(double(v.comp), comp, sizeof(comp));
        double pct = double(v.comp) / double(v.bytes) * 100.0;
        std::fprintf(stderr,
                     "%s : %s, \033[1m%llu\033[0m entries, \033[36m%s\033[0m => \033[32m%s\033[0m"
                     " (ratio: \033[1m%.1f%%\033[0m) @ \033[32m%s/s\033[0m — %s\n",
                     v.name.c_str(), status, (unsigned long long)v.files,
                     comp, sz, pct, rate, verdict);
      } else {
        std::fprintf(stderr,
                     "%s : %s, \033[1m%llu\033[0m entries, \033[32m%s\033[0m @ \033[32m%s/s\033[0m — %s\n",
                     v.name.c_str(), status, (unsigned long long)v.files,
                     sz, rate, verdict);
      }
    }
    std::fflush(stderr);
  }
  return rc;
}
#endif  // !_WIN32  (extract_tar + verify_tar)

// -l (plain): zstd-style frame summary for each .zst file — Frames / Skips /
// Compressed / Uncompressed / Ratio / Check / Filename.  Walks frame + block
// headers WITHOUT decompressing (ZSTD_findFrameCompressedSize skips block
// content); over an mmap only the header pages fault in, so it's fast even on a
// huge multi-frame file.  Uncompressed is summed from frame-header content sizes.
static int list_zst(const Options & opt)
{
  std::printf("%7s %6s %12s %14s %7s %6s  %s\n",
              "Frames", "Skips", "Compressed", "Uncompressed", "Ratio", "Check", "Filename");
  int rc = EXIT_OK;
  for (const std::string & fn : opt.inputs) {
    // Get a contiguous read-only view: mmap a regular file (lazy, zero-copy),
    // else read it all (stdin / mmap failure).
    const unsigned char * base = nullptr; size_t fsize = 0;
    void * mp = MAP_FAILED; size_t mp_len = 0; std::vector<char> buf;
#ifndef _WIN32
    if (fn != "-") {
      int fd = ::open(fn.c_str(), O_RDONLY);
      if (fd < 0) { vlog(V_ERROR, opt, "gzstd: list: " + fn + ": " + std::strerror(errno) + "\n"); rc = EXIT_DATA; continue; }
      struct stat st;
      if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
        mp_len = (size_t)st.st_size;
        mp = ::mmap(nullptr, mp_len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mp != MAP_FAILED) {
          base = (const unsigned char *)mp; fsize = mp_len;
          // The frame walk touches only block-header bytes (~one per 128 KiB),
          // which on a cold file is ~1M serial random page faults.  Hint a
          // forward sequential scan so the kernel prefetches in big reads
          // instead — turns random fault-per-header I/O into streaming reads.
          ::madvise(mp, mp_len, MADV_SEQUENTIAL);
        }
      }
      ::close(fd);
    }
#endif
    if (!base) {  // stdin or non-mmappable: read it all
      FILE * f = open_input(fn);
      char tmp[1 << 20]; size_t n;
      while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
      if (f && f != stdin) std::fclose(f);
      base = (const unsigned char *)buf.data(); fsize = buf.size();
    }

    uint64_t nframes = 0, nskips = 0, uncomp = 0;
    bool uncomp_known = true, has_check = false, ok = true;
    size_t pos = 0;
    while (pos + 4 <= fsize) {
      uint32_t magic; std::memcpy(&magic, base + pos, 4);
      if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {           // skippable frame
        if (pos + 8 > fsize) { ok = false; break; }
        uint32_t ssz; std::memcpy(&ssz, base + pos + 4, 4);
        pos += 8 + (size_t)ssz; ++nskips; continue;
      }
      size_t csz = ZSTD_findFrameCompressedSize(base + pos, fsize - pos);
      if (ZSTD_isError(csz) || csz == 0) { ok = false; break; }
      unsigned long long ucs = ZSTD_getFrameContentSize(base + pos, fsize - pos);
      if (ucs == ZSTD_CONTENTSIZE_UNKNOWN || ucs == ZSTD_CONTENTSIZE_ERROR) uncomp_known = false;
      else uncomp += ucs;
      // Frame_Header_Descriptor (byte after the 4-byte magic): bit 2 = content checksum.
      if (pos + 4 < fsize && (base[pos + 4] & 0x04)) has_check = true;
      pos += csz; ++nframes;
    }
#ifndef _WIN32
    if (mp != MAP_FAILED) ::munmap(mp, mp_len);
#endif

    char comp_h[32], unc_h[32];
    human_bytes(double(fsize), comp_h, sizeof comp_h);
    if (uncomp_known) human_bytes(double(uncomp), unc_h, sizeof unc_h);
    else std::snprintf(unc_h, sizeof unc_h, "?");
    double ratio = (uncomp_known && fsize > 0) ? double(uncomp) / double(fsize) : 0.0;
    char ratio_s[16];
    if (uncomp_known) std::snprintf(ratio_s, sizeof ratio_s, "%.3f", ratio);
    else std::snprintf(ratio_s, sizeof ratio_s, "-");
    std::printf("%7llu %6llu %12s %14s %7s %6s  %s\n",
                (unsigned long long)nframes, (unsigned long long)nskips,
                comp_h, unc_h, ratio_s, has_check ? "XXH64" : "None",
                fn == "-" ? "(stdin)" : fn.c_str());
    if (!ok) {
      vlog(V_ERROR, opt, "gzstd: list: " + fn + ": not a valid zstd stream (truncated or corrupt)\n");
      rc = EXIT_DATA;
    }
  }
  std::fflush(stdout);
  return rc;
}

#ifndef _WIN32
// -l --tar: list each archive's contents (tar -tvf style) without writing.
// Decompresses through the in-memory FrameSink (same fast path as verify); the
// Extractor in list mode prints each entry to stdout and skips the data.
static int list_tar(const Options & opt, Meter * m)
{
  int rc = EXIT_OK;
  for (const std::string & arc : opt.tar_sources) {
    FILE * in = open_input(arc);   // dies on failure
    if (arc != "-") {
      unsigned char magic[4] = {0};
      size_t nr = std::fread(magic, 1, 4, in);
      uint32_t mg = 0; if (nr == 4) std::memcpy(&mg, magic, 4);
      bool ok_magic = (mg == 0xFD2FB528u) || ((mg & 0xFFFFFFF0u) == 0x184D2A50u);
      if (!ok_magic) {
        vlog(V_ERROR, opt, "gzstd: list: not a zstd archive: " + arc + "\n");
        if (in && in != stdin) std::fclose(in);
        rc = EXIT_DATA; continue;
      }
      std::rewind(in);
    }
    FrameSink sink(256 * ONE_MIB);
    g_tar_decomp_sink = &sink;
    tarx::Extractor ex(-1, opt, /*meter=*/nullptr, /*validate_only=*/false, /*list_only=*/true);
    std::thread exth([&] { ex.run_sink(&sink); });

    Options dopt = opt;
    dopt.sparse_mode = 0; dopt.unsafe_overwrite = true; dopt.mode = Mode::DECOMPRESS;
    int64_t ff = (arc != "-") ? peek_first_frame_decomp_size(in) : -1;
    if (ff > (int64_t)SINGLE_FRAME_STREAM_MIN) {
      decompress_stream_from_file(in, nullptr, dopt, m);
    } else {
#ifdef HAVE_NVCOMP
      if (dopt.cpu_only) decompress_cpu_mt(in, nullptr, dopt, m);
      else               decompress_nvcomp(in, nullptr, dopt, m);
#else
      decompress_cpu_mt(in, nullptr, dopt, m);
#endif
    }
    sink.close();
    exth.join();
    g_tar_decomp_sink = nullptr;
    if (in && in != stdin) std::fclose(in);
    std::fflush(stdout);

    if (opt.verbosity >= V_DEFAULT) {  // footer summary (stderr, so stdout stays a clean listing)
      char sz[64]; human_bytes(double(ex.validated_bytes()), sz, sizeof sz);
      std::string pfx = opt.tar_sources.size() > 1 ? arc + ": " : "";
      std::fprintf(stderr, "%s%llu files, %s\n", pfx.c_str(),
                   (unsigned long long)ex.validated_files(), sz);
    }
    if (ex.had_error() && rc == EXIT_OK) rc = EXIT_DATA;
  }
  return rc;
}
#endif  // !_WIN32

int main(int argc, char ** argv)
{
  setup_signal_handlers();

  // Test-only: deterministic GPU fault injection (see g_debug_fail_gpu_after).
  if (const char * fa = std::getenv("GZSTD_DEBUG_FAIL_GPU_AFTER"))
    g_debug_fail_gpu_after = std::atoll(fa);

  // Test-only: deterministic frame-corruption injection (see g_debug_corrupt_frame).
  if (const char * cf = std::getenv("GZSTD_DEBUG_CORRUPT_FRAME"))
    g_debug_corrupt_frame = std::atoll(cf);
  if (const char * cp = std::getenv("GZSTD_DEBUG_CORRUPT_PERSIST"))
    g_debug_corrupt_persist = (std::atoll(cp) != 0);

  // Recycle frame buffers instead of mmap/munmap-ing them every chunk.
  // Our per-frame heap buffers are large (16 MiB default, 32 MiB ultra), so glibc
  // serves each via mmap (its dynamic threshold caps at 32 MiB and the 4-producer/
  // N-consumer hand-off pattern keeps the adaptation from engaging).  The killer is
  // not the page faults but the munmap on every free: tearing down a 16 MiB mapping
  // forces a TLB shootdown — an IPI to every other core — and that cost scales with
  // core count, so on a 256-core box it dominates (observed as many minutes of sys
  // time and the --direct-read reader stuck at ~1/3 of the drive's O_DIRECT
  // bandwidth; locally, 256-core-free, pinning the threshold still ~halved a 4 GiB
  // direct-read).  Pinning the mmap threshold above our frame size keeps these
  // buffers on the heap, where freed chunks return to the arena's bins and are
  // reused with no munmap and no shootdown; a high trim threshold stops glibc
  // handing the heap back to the OS between bursts only to re-grow it.  Peak RSS is
  // bounded by the in-flight cap (throttle + queue byte-cap).
#if defined(__GLIBC__)
  mallopt(M_MMAP_THRESHOLD, 128 * 1024 * 1024);  // frames (≤32 MiB) come from the heap, not mmap
  mallopt(M_TRIM_THRESHOLD, 256 * 1024 * 1024);  // keep the recycled heap resident
#endif

  Options opt = parse_args(argc, argv);

  // Asymmetric-mode default: PCIe Gen3 → cpu-only decompress; otherwise
  // hybrid for compress and Gen4+ decompress.  No-op if user passed an
  // explicit --cpu-only / --gpu-only / --hybrid.
  apply_backend_defaults(opt);

  // Early startup banner at -v+.  Printed BEFORE any heavy init (CUDA
  // device probe, file open, output preallocate) so users get immediate
  // feedback that gzstd is alive — important on loaded servers where
  // CUDA init can stall for several seconds.
  // Verbose output convention: every line uses [TAG] prefix with an
  // UPPERCASE tag, single space, sentence-case body.  See gzstd.cpp
  // verbose-output style.
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[STARTUP] gzstd " << GZSTD_VERSION << " ";
    switch (opt.mode) {
      case Mode::COMPRESS:   os << "COMPRESS"; break;
      case Mode::DECOMPRESS: os << "DECOMPRESS"; break;
      case Mode::TEST:       os << "TEST"; break;
    }
    if (opt.cpu_only)        os << " (cpu-only)";
    else if (opt.gpu_only)   os << " (gpu-only)";
    else if (opt.hybrid) {
      os << " (hybrid";
      if (opt.cpu_share >= 0.0) {
        os << ", CPU share " << std::fixed << std::setprecision(1)
           << (opt.cpu_share * 100.0) << "%";
      } else {
        os << ", CPU share adaptive";
      }
      os << ")";
    }
    else                     os << " (auto-select backend)";
    std::fprintf(stderr, "%s\n", os.str().c_str());
    std::fflush(stderr);  // unbuffer so it hits the terminal immediately
  }

  // --direct-read is a no-op when CREATING a --tar archive: members are read
  // through the parallel tar reader (buffered preads, gzstd.cpp:assemble), not
  // the single-file O_DIRECT input path (which also requires a regular-file
  // input, whereas --tar's input is a directory).  Warn instead of silently
  // ignoring it.
  if (opt.direct_read && opt.tar_mode && opt.mode == Mode::COMPRESS)
    vlog(V_DEFAULT, opt, "gzstd: warning: --direct-read has no effect with --tar "
                         "(archive members are read buffered); ignoring it\n");

  // --write-threads sizes the -d --tar extractor's parallel file-writer pool and
  // is consumed nowhere else (gzstd.cpp:Extractor::start_pool), so it is a no-op
  // for plain compress/decompress and for --tar create.  Warn rather than ignore.
  if (opt.write_threads > 0 && !(opt.tar_mode && opt.mode == Mode::DECOMPRESS))
    vlog(V_DEFAULT, opt, "gzstd: warning: --write-threads only applies to -d --tar "
                         "extraction (parallel file-writer pool); ignoring it\n");

#ifndef _WIN32
  // -l: list .zst frame info (plain) or archive contents (`-l --tar`).  Checked
  // before the extract/verify dispatch so it doesn't fall through to either.
  if (opt.list_mode) {
    if (opt.tar_mode) { Meter meter; return list_tar(opt, &meter); }
    return list_zst(opt);
  }
  // -d --tar: extract the archive(s) into -C DIR via the native parallel
  // extractor; bypasses the single-output per-file loop below.
  if (opt.tar_mode && opt.mode == Mode::DECOMPRESS) {
    Meter meter;
    return extract_tar(opt, &meter);
  }
  // -t --tar: decompress AND structurally validate the inner tar (header
  // checksums, completeness) without writing files — catches a corrupt/truncated
  // tar inside an otherwise-intact zstd stream, which plain -t misses.
  if (opt.tar_mode && opt.mode == Mode::TEST) {
    Meter meter;
    return verify_tar(opt, &meter);
  }
#endif

  int exit_code = EXIT_OK;

  for (size_t file_idx = 0; file_idx < opt.inputs.size(); ++file_idx) {
  // --- Per-file setup ---
  opt.input = opt.inputs[file_idx];
  if (opt.keep_going) g_damage.reset();   // damage is reported per input file

  // For multi-file with no explicit -o, derive output per file
  if (opt.inputs.size() > 1 && !opt.to_stdout) {
    opt.output = derive_output(opt.input, opt.mode);
  }

  // --tar synthesizes its input from tar_sources; there is no input FILE*.
  FILE * in = opt.tar_mode ? nullptr : open_input(opt.input);
#ifndef _WIN32
  // --cold (benchmarking only): drop the input file from the kernel's page
  // cache before reading so repeated benchmark iterations don't measure
  // memory-to-memory throughput.  POSIX_FADV_DONTNEED on a clean read-only
  // file evicts its pages from the page cache immediately; the next read
  // therefore goes to disk.
  if (opt.cold_read && opt.input != "-" && in != stdin) {
    int fd = fileno(in);
    if (fd >= 0) (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  }
#endif
  bool to_stdout = (opt.to_stdout || (!opt.output.empty() && opt.output == "stdout"));

  // When writing to stdout, force keep (can't delete stdin) and set binary mode
  if (to_stdout) {
    opt.keep = true;
    set_binary_mode(stdout);
  }

  std::string tmp;         // non-empty only when using atomic temp file (-f overwrite)
  bool use_atomic = false; // true when writing to .tmp then renaming
  FILE * out = nullptr;
  if (to_stdout) {
    out = stdout;
  } else {
    // Only a pre-existing REGULAR file is a clobber risk.  A special output
    // target (/dev/null, a device node, a fifo) is a deliberate sink, not
    // something we overwrite — never block on it and never register it for
    // cleanup-unlink (we must not delete /dev/null on failure).
    bool existed_before = fs::exists(opt.output);
    bool exists = existed_before && fs::is_regular_file(opt.output);
    if (exists && !opt.force) {
      die_io("output exists (use -f to overwrite): " + opt.output);
    }
    if (exists && opt.force) {
      bool is_regular = fs::is_regular_file(opt.output);
      if (is_regular && !opt.unsafe_overwrite) {
        out = open_output_atomic(opt.output, tmp);
        use_atomic = true;
        register_tmp_file(tmp);
      } else {
        // --overwrite: replace target in place.  On ext4 fopen("wb") has to
        // truncate the existing file, which means freeing every extent the
        // inode references — O(file_size) for huge files (10-30+s on 400+ GiB
        // outputs).  Unlink first instead: the inode is unreferenced
        // immediately and the extents free in the background.  fopen then
        // creates a fresh empty file in O(1).
        if (is_regular) {
          std::error_code ec_unlink;
          fs::remove(opt.output, ec_unlink);
          // If unlink failed (permissions, race), fall through to fopen "wb"
          // which will return its own error.
        }
        out = std::fopen(opt.output.c_str(), "wb");
        if (!out) die_io("cannot open output: " + opt.output);
        std::setvbuf(out, nullptr, _IOFBF, 1 * 1024 * 1024);
        if (is_regular) register_tmp_file(opt.output);
      }
    } else {
      out = std::fopen(opt.output.c_str(), "wb");
      if (!out) die_io("cannot open output: " + opt.output);
      std::setvbuf(out, nullptr, _IOFBF, 1 * 1024 * 1024);
      // Register for cleanup-on-failure only if we are CREATING a new file;
      // never register a pre-existing special target like /dev/null.
      if (!existed_before) register_tmp_file(opt.output);
    }
  }

  // O_DIRECT output: only when explicitly requested via --direct.
  // Default is buffered I/O (fwrite) which uses the OS page cache for
  // consistent throughput — O_DIRECT bypasses the cache and exposes the
  // application to NVMe GC stalls, journal commits, and writeback
  // contention from prior runs, causing 2-5× wall-time variance on
  // sequential large-file workloads.
#ifndef _WIN32
  std::unique_ptr<DirectWriter> direct_writer;
  if (opt.direct_io && !to_stdout && out != stdout) {
    std::string write_path = use_atomic ? tmp : opt.output;
    struct stat st;
    bool is_regular = (stat(write_path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
    if (is_regular) {
      auto dw = std::make_unique<DirectWriter>();
      if (dw->open(write_path)) {
        if (out) { std::fclose(out); out = nullptr; }
        direct_writer = std::move(dw);
        vlog(V_VERBOSE, opt, "[O_DIRECT] using O_DIRECT for output (--direct)\n");
      }
    }
  } else if (opt.direct_io && to_stdout && out == stdout) {
    do {
      int fd = fileno(stdout);
      if (fd < 0) break;
      struct stat st;
      if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) break;
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || (flags & O_APPEND)) break;
      char link_path[64];
      char real_path[4096];
      std::snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd);
      ssize_t len = readlink(link_path, real_path, sizeof(real_path) - 1);
      if (len <= 0) break;
      real_path[len] = '\0';
      if (std::strncmp(real_path, "/dev/", 5) == 0) break;
      if (std::strstr(real_path, "(deleted)")) break;
      auto dw = std::make_unique<DirectWriter>();
      if (dw->open(std::string(real_path))) {
        std::fflush(stdout);
        out = nullptr;
        direct_writer = std::move(dw);
        vlog(V_VERBOSE, opt, "[O_DIRECT] stdout redirect using O_DIRECT (--direct)\n");
      }
      // If open failed, fall through to normal fwrite via stdout
    } while (false);
  }
  DirectWriter * dw_ptr = direct_writer.get();
#else
  DirectWriter * dw_ptr = nullptr;
#endif

#ifndef _WIN32
  g_direct_writer = dw_ptr;
#endif

  Meter meter;
  if (opt.verbosity >= V_DEBUG && opt.mode == Mode::COMPRESS) {
    std::ostringstream os;
#ifdef HAVE_NVCOMP
    if (opt.cpu_only) {
      os << "compression level: " << opt.level
         << (opt.ultra && opt.level >= 20 ? " (ultra)" : "");
    } else if (opt.gpu_only) {
      os << "compression: GPU (nvCOMP, fixed level)";
    } else {
      os << "compression level: " << opt.level
         << (opt.ultra && opt.level >= 20 ? " (ultra)" : "")
         << " (CPU frames); GPU uses fixed nvCOMP level";
    }
#else
    os << "compression level: " << opt.level
       << (opt.ultra && opt.level >= 20 ? " (ultra)" : "");
#endif
    vlog(V_DEBUG, opt, os.str() + "\n");
  }

  // Progress bar for decompress/test modes
  std::atomic<bool> prog_done{false};
  std::thread prog_thr;
  uint64_t total_in_for_progress = 0;
  if (opt.mode == Mode::TEST || opt.mode == Mode::DECOMPRESS) {
    total_in_for_progress = known_input_size(opt, in);
    prog_thr = std::thread(progress_loop, std::cref(opt), &meter,
                           total_in_for_progress, &prog_done);
  }

  const auto t0 = std::chrono::steady_clock::now();
  if (opt.mode == Mode::COMPRESS) {
    // Warn if input looks like it's already zstd-compressed
    if (opt.input.size() > 4
        && opt.input.substr(opt.input.size() - 4) == ".zst"
        && opt.verbosity >= V_ERROR) {
      std::cerr << "gzstd: warning: " << opt.input
                << " already has .zst extension  compressing anyway\n"
                << "  hint: did you mean to decompress? use: gzstd -d "
                << opt.input << "\n";
    }
    // Compress dispatch with --verify + automatic rebuild-on-failure.
    //
    // A pass can fail two ways that both demand discarding the output and
    // rebuilding CPU-only from the original input: a GPU fault mid-run, or (with
    // --verify) a frame that fails to decompress-verify.  Both set
    // g_gpu_failed_restart.  We loop: run a pass, drain the verify pool, and if a
    // restart was flagged, reset the output and retry CPU-only — repeating until
    // it succeeds or (with --verify-retries) the cap is hit.  Without --verify
    // and without a GPU fault, the loop runs exactly once.
#ifndef HAVE_NVCOMP
    if (opt.gpu_only) die_usage("This binary was built without nvCOMP; --gpu-only cannot be satisfied");
    if (opt.hybrid) vlog(V_VERBOSE, opt, "[HYBRID] not available in CPU-only build; using MT CPU\n");
#endif
    auto run_one_pass = [&](const Options & po) {
#ifdef HAVE_NVCOMP
      if (po.cpu_only) {
        if (po.sliding_window) compress_cpu_sliding_window(in, out, po, &meter);
        else                   compress_cpu_mt(in, out, po, &meter);
      } else {
        compress_nvcomp(in, out, po, &meter);
      }
#else
      if (po.sliding_window) compress_cpu_sliding_window(in, out, po, &meter);
      else                   compress_cpu_mt(in, out, po, &meter);
#endif
    };

    Options pass_opt = opt;
    int rebuild_attempt = 0;
    for (;;) {
      g_gpu_failed_restart.store(false);
      g_gpu_aborted.store(false);   // clear the GPU-fault abort before each pass (incl. the CPU rebuild)
      g_verify_failed.store(false, std::memory_order_relaxed);
      g_verify_failed_seq.store(-1, std::memory_order_relaxed);

      // --verify: stand up a background decompress-verify pool for this pass.
      // It taps frames via g_verify_pool from the writer's in-order drain.
      std::unique_ptr<VerifyPool> vpool;
      if (opt.verify) {
        const int vmax = (opt.cpu_threads > 0)
            ? opt.cpu_threads
            : std::max(1, (int)std::thread::hardware_concurrency());
        vpool = std::make_unique<VerifyPool>(vmax, 256 /* max frames queued */);
        g_verify_pool = vpool.get();
      }

      run_one_pass(pass_opt);

      // Drain/join verify BEFORE reading the restart flag, so a mismatch that
      // surfaces only as the final frames are checked still triggers a rebuild.
      if (vpool) { vpool->finish(); g_verify_pool = nullptr; vpool.reset(); }

      if (!g_gpu_failed_restart.load()) break;   // clean — verified, or no verify and no fault

      const bool verify_trigger = g_verify_failed.load(std::memory_order_relaxed);

      // Recoverable only if we can re-read the input AND discard the output.
      // --tar re-reads its source paths; otherwise the fseek both TESTS and (as a
      // side effect) REWINDS the input for the rebuild.
      const bool in_rebuildable =
          opt.tar_mode || (in && in != stdin && std::fseek(in, 0, SEEK_SET) == 0);
      bool out_rebuildable;
#ifndef _WIN32
      if (dw_ptr) out_rebuildable = true;            // O_DIRECT regular file
      else
#endif
        out_rebuildable = (out && std::fseek(out, 0, SEEK_SET) == 0);  // seekable (incl. redirected stdout)

      if (!in_rebuildable || !out_rebuildable) {
        // Unrecoverable over a pipe: die loudly so a pipeline fails.
        if (verify_trigger) {
          die(std::string("--verify: a frame failed to verify and the ")
              + (!in_rebuildable ? "input" : "output")
              + " is a pipe/stream, so the output cannot be discarded and rebuilt.\n"
              "  The compressed bytes are CORRUPT; do not use them.  Re-run with\n"
              "  regular files for input and output so --verify can recover.", EXIT_DATA);
        }
        die(std::string("a GPU faulted during compression and the ")
            + (!in_rebuildable ? "input" : "output")
            + " is a pipe/stream — the partial output cannot be discarded and\n"
            "  rebuilt on the CPU, and a faulted GPU's output is NOT trustworthy.\n"
            "  Any bytes already sent downstream are INCOMPLETE/CORRUPT; do not use them.\n"
            "  Re-run with --cpu-only, or compress to/from regular files so a GPU\n"
            "  fault can be recovered automatically.", EXIT_GPU_FAIL);
      }

      ++rebuild_attempt;

      // --verify-retries cap (verify-triggered rebuilds only; a GPU fault rebuilds
      // CPU-only once and the CPU path doesn't re-fault).
      if (verify_trigger && opt.verify_retries > 0 && rebuild_attempt > opt.verify_retries) {
        die("--verify: output still failed to verify after "
            + std::to_string(opt.verify_retries)
            + " CPU rebuild attempt(s) — this machine looks unreliable and the\n"
            "  output is NOT trustworthy.  (Raise or clear --verify-retries to keep "
            "trying.)", EXIT_DATA);
      }

      // Loud per-attempt log: an unattended user sees progress, and a failing seq
      // that repeats across attempts marks a deterministic fault, not a transient.
      {
        std::ostringstream os;
        if (verify_trigger) {
          os << "WARNING: --verify caught a corrupt frame";
          const int64_t bad = g_verify_failed_seq.load(std::memory_order_relaxed);
          if (bad >= 0) os << " (sequence " << bad << ")";
          os << " — discarding output and rebuilding CPU-only";
        } else {
          os << "WARNING: a GPU faulted — discarding output and rebuilding CPU-only";
        }
        os << " from the original input (attempt " << rebuild_attempt << ")...\n";
        vlog(V_ERROR, opt, os.str());   // loud: an integrity event survives -q (only -qq silences)
      }

      // Reset the output to empty for the rebuild (input already rewound above).
#ifndef _WIN32
      if (dw_ptr) {
        std::string write_path = use_atomic ? tmp : opt.output;
        direct_writer.reset();                 // dtor joins writer thread, closes fd
        direct_writer = std::make_unique<DirectWriter>();
        if (!direct_writer->open(write_path))
          die_io("failed to reopen O_DIRECT output for CPU rebuild");
        dw_ptr = direct_writer.get();
        g_direct_writer = dw_ptr;
      } else
#endif
      if (out) {
        std::rewind(out);
        if (ftruncate(fileno(out), 0) != 0) { /* best-effort: rebuild overwrites */ }
      }
      meter.reset();

      // Anti-spin floor between full-archive rebuilds — NOT a coordination wait:
      // real rebuilds take real time; this only guards a pathological instant-fail
      // loop (e.g. a tiny input) from busy-spinning a core.
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Every rebuild is CPU-only — the trustworthy, deterministic path.
      pass_opt = opt;
      pass_opt.cpu_only = true; pass_opt.gpu_only = false; pass_opt.hybrid = false;
    }

    // CPU-only stats: written when the final (successful) pass ran on the CPU —
    // the original cpu-only path, any rebuild, or a no-NVCOMP build.  The nvCOMP
    // path writes its own stats internally.
#ifdef HAVE_NVCOMP
    const bool final_pass_cpu = pass_opt.cpu_only;
#else
    const bool final_pass_cpu = true;
#endif
    if (final_pass_cpu && !opt.stats_json.empty()) {
      CpuAgg agg{};
      agg.threads = (opt.cpu_threads > 0)
                    ? opt.cpu_threads
                    : std::max(1, int(std::thread::hardware_concurrency()) - 1);
      double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t0).count();
      write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, agg);
    }
  } else {
    // Decompression / test mode

    // Validate that input is actually a Zstd file by checking the magic number.
    // Zstd frames start with 0xFD2FB528.  Skippable frames start with 0x184D2A5?.
    // Catch this early to give a clear error instead of hanging or cryptic failures.
    if (opt.input != "-") {
      unsigned char magic[4] = {0};
      size_t nr = std::fread(magic, 1, 4, in);
      if (nr == 4) {
        uint32_t m32 = 0;
        std::memcpy(&m32, magic, 4);
        bool is_zstd = (m32 == 0xFD2FB528u);
        bool is_skip = ((m32 & 0xFFFFFFF0u) == 0x184D2A50u);
        if (!is_zstd && !is_skip) {
          die_data("not a Zstd file (bad magic: 0x"
              + ([&]{ char h[16]; snprintf(h, sizeof(h), "%08X", m32); return std::string(h); })()
              + "): " + opt.input
              + "\n  hint: did you mean to compress? use: gzstd " + opt.input);
        }
      }
      // Rewind so the actual decompressor sees the full stream
      std::rewind(in);
    }

    // A large single frame (zstd / --sliding-window) can't be split across
    // CPU threads or GPU subchunks.  Above SINGLE_FRAME_STREAM_MIN it's
    // effectively a single-frame file, so stream it straight from the FILE —
    // overlapping read/decompress/write with bounded memory — for every mode,
    // instead of buffering the whole frame through the queue.  Genuinely
    // multi-frame chunked inputs (frame0 below the threshold) keep their
    // parallel path.  Seekable input only (peek returns -1 on stdin).
    int64_t first_frame_decomp =
        (opt.input != "-") ? peek_first_frame_decomp_size(in) : -1;
    if (first_frame_decomp > (int64_t)SINGLE_FRAME_STREAM_MIN) {
      char sz[32]; human_bytes(double(first_frame_decomp), sz, sizeof(sz));
#ifdef HAVE_NVCOMP
      if (!opt.cpu_only)
        vlog(V_NORMAL, opt,
             std::string("warning: first frame decompresses to ") + sz
             + " (GPU max: 16 MiB).\n"
             "  This file was likely compressed with --sliding-window or zstd.\n"
             "  Decompressing on CPU (a single frame can't use the GPU).\n");
#endif
      vlog(V_VERBOSE, opt,
           std::string("[INIT] decompress: streaming single ") + sz + " frame on CPU\n");
      // total_out/total_out_final drive the progress bar's byte-level out%.
      meter.total_out.store((uint64_t)first_frame_decomp, std::memory_order_relaxed);
      meter.total_out_final.store(true, std::memory_order_release);
#ifndef _WIN32
      if (g_direct_writer && opt.preallocate_output
          && g_direct_writer->preallocate((uint64_t)first_frame_decomp)) {
        vlog(V_VERBOSE, opt, std::string("[FALLOCATE] preallocated ") + sz + " output\n");
      }
#endif
      decompress_stream_from_file(in, out, opt, &meter);
    } else
#ifdef HAVE_NVCOMP
    if (opt.cpu_only) {
      decompress_cpu_mt(in, out, opt, &meter);
    } else {
      decompress_nvcomp(in, out, opt, &meter);
    }
#else
    decompress_cpu_mt(in, out, opt, &meter);
#endif
    if (!opt.stats_json.empty()) {
      double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t0).count();
      CpuAgg dummy{};
      dummy.threads = 1;
      write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, dummy);
    }

    // --keep-going: summarize the damage and choose the recovery exit code.  The
    // workers already printed a per-frame WARNING for each; here we report the
    // affected file(s) (for --tar) or byte range(s) (plain .zst) and the verdict.
    if (opt.keep_going && g_damage.damaged()) {
      report_keep_going_damage(opt);
      const int sev = g_damage.incomplete.load(std::memory_order_relaxed)
          ? EXIT_DECOMP_INCOMPLETE : EXIT_DECOMP_UNVERIFIED;
      if (sev > exit_code) exit_code = sev;   // worst across multiple input files
    }
  }

  // ---- Test mode: report integrity results and exit ----
  if (opt.mode == Mode::TEST) {
    prog_done = true;
    if (prog_thr.joinable()) prog_thr.join();

    // Compute compressed vs decompressed sizes
    uint64_t comp_size = 0;
    if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
      comp_size = (uint64_t)fs::file_size(opt.input);
    else
      comp_size = meter.read_bytes.load();

    uint64_t decomp_size = meter.wrote_bytes.load();
    double pct = (decomp_size > 0)
                 ? (double)comp_size / (double)decomp_size * 100.0
                 : 0.0;

    if (opt.verbosity >= V_DEFAULT) {
      // Print completion summary on stderr (overwrites progress bar)
      auto dt = std::chrono::steady_clock::now() - meter.t0;
      double secs = std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
      double rate = secs > 0 ? double(decomp_size) / secs : 0.0;
      char comp_s[64], decomp_s[64], rate_s[64];
      human_bytes(double(comp_size), comp_s, sizeof(comp_s));
      human_bytes(double(decomp_size), decomp_s, sizeof(decomp_s));
      human_bytes(rate, rate_s, sizeof(rate_s));

      // Strip .zst extension from display name
      std::string base_name;
      if (opt.input == "-")
        base_name = "(stdin)";
      else if (opt.input.size() > 4
               && opt.input.substr(opt.input.size() - 4) == ".zst")
        base_name = opt.input.substr(0, opt.input.size() - 4);
      else
        base_name = opt.input;

      char summary[512];
      std::snprintf(summary, sizeof(summary),
        "%s : \033[1;92mOK\033[0m (\033[36m%s\033[0m => \033[32m%s\033[0m, ratio: \033[1m%.1f%%\033[0m) @ \033[32m%s/s\033[0m",
        base_name.c_str(), comp_s, decomp_s, pct, rate_s);
      std::fprintf(stderr, "\r%s\033[K\n", summary);
      std::fflush(stderr);
    }

    std::fclose(in);
    continue;  // next file
  }
  if (opt.mode == Mode::DECOMPRESS) {
    // Don't stop progress thread yet -- it shows write drain progress.
    // It will be stopped after finalize below.
  }

  double finalize_ms = 0, sync_ms = 0;

  // If DirectWriter is active (either explicit file or stdout-to-file O_DIRECT),
  // finalize it first regardless of to_stdout flag.
#ifndef _WIN32
  if (g_direct_writer) {
    if (opt.mode == Mode::COMPRESS && opt.verbosity >= V_DEFAULT) {
      uint64_t out_bytes = meter.wrote_bytes.load();
      if (out_bytes > 100 * ONE_MIB) {
        char sz[64];
        human_bytes(double(out_bytes), sz, sizeof(sz));
        std::fprintf(stderr, "\r[done] finalizing %s ...\033[K", sz);
        std::fflush(stderr);
      }
    }
    auto t_fin = std::chrono::steady_clock::now();
    if (!g_direct_writer->finalize())
      die_io("failed to finalize O_DIRECT output");
    if (opt.sync_output) {
      // --direct closed the FILE* and writes via DirectWriter's own fd, so the
      // fsync_file(out) path below never runs.  Flush the O_DIRECT fd here
      // (device write cache + the size metadata set by finalize's ftruncate) so
      // --direct --sync-output is actually durable (ROADMAP 7.5).
      int dfd = g_direct_writer->fd();
      if (dfd >= 0) ::fsync(dfd);
    }
    g_direct_writer = nullptr;
    direct_writer.reset();
    finalize_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(
        std::chrono::steady_clock::now() - t_fin).count();
  }
#endif

  if (!to_stdout) {
    if (out) {
      if (opt.sync_output) {
        fsync_file(out);
      }
      std::fclose(out);
    }
    if (in) std::fclose(in);  // null in --tar mode (no input FILE*)
    if (use_atomic) {
      auto t_rename = std::chrono::steady_clock::now();
      // Atomic overwrite: rename .tmp to final output
      std::error_code ec_rename; fs::rename(tmp, opt.output, ec_rename);
      if (ec_rename) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(opt.output, std::ios::binary | std::ios::trunc);
        if (!src || !dst) die_io("failed to finalize output file");
        dst << src.rdbuf(); src.close(); dst.close(); fs::remove(tmp);
      }
      double rename_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(
          std::chrono::steady_clock::now() - t_rename).count();
      if (opt.verbosity >= V_VERBOSE && rename_ms > 100)
        vlog(V_VERBOSE, opt, "[RENAME] atomic rename took " + std::to_string(int(rename_ms)) + " ms\n");
    }
    // Success: disarm cleanup (temp or direct output file is now final)
    clear_tmp_file();
    if (!opt.keep && opt.input != "-") {
      std::error_code ec_rm;
      fs::remove(opt.input, ec_rm);
    }
  } else {
    // Flush stdout to ensure all data reaches the downstream pipe
    std::fflush(stdout);
    if (in) std::fclose(in);  // null in --tar mode (no input FILE*)
  }

  // Stop decompress progress thread now that write drain is complete
  if (opt.mode == Mode::DECOMPRESS) {
    prog_done = true;
    if (prog_thr.joinable()) prog_thr.join();
  }

  // Print zstd-style completion summary with rate:
  //   input_file : ratio% (input_size => output_size, output_file) @ rate/s
  // The \r overwrites the progress bar; trailing spaces clear any leftover chars.
  if (opt.verbosity >= V_DEFAULT && opt.mode == Mode::COMPRESS) {
    uint64_t in_bytes  = meter.read_bytes.load();
    uint64_t out_bytes = meter.wrote_bytes.load();
    double ratio_pct = (in_bytes > 0) ? (double)out_bytes / (double)in_bytes * 100.0 : 0.0;
    auto dt = std::chrono::steady_clock::now() - meter.t0;
    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
    double rate = secs > 0 ? double(in_bytes) / secs : 0.0;
    char in_s[64], out_s[64], rate_s[64];
    human_bytes(double(in_bytes), in_s, sizeof(in_s));
    human_bytes(double(out_bytes), out_s, sizeof(out_s));
    human_bytes(rate, rate_s, sizeof(rate_s));
    std::string in_name  = (opt.input == "-") ? "(stdin)" : opt.input;
    std::string out_name = to_stdout ? "(stdout)" : opt.output;
    char summary[512];
    std::snprintf(summary, sizeof(summary),
      "%s : \033[1m%5.2f%%\033[0m (\033[36m%s\033[0m => \033[32m%s\033[0m, %s) @ \033[32m%s/s\033[0m",
      in_name.c_str(), ratio_pct, in_s, out_s, out_name.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }

  // Decompression summary
  if (opt.verbosity >= V_DEFAULT && opt.mode == Mode::DECOMPRESS) {
    uint64_t in_bytes  = meter.read_bytes.load();
    uint64_t out_bytes = meter.wrote_bytes.load();
    auto dt = std::chrono::steady_clock::now() - meter.t0;
    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
    double rate = secs > 0 ? double(out_bytes) / secs : 0.0;
    char in_s[64], out_s[64], rate_s[64];
    human_bytes(double(in_bytes), in_s, sizeof(in_s));
    human_bytes(double(out_bytes), out_s, sizeof(out_s));
    human_bytes(rate, rate_s, sizeof(rate_s));
    std::string in_name  = (opt.input == "-") ? "(stdin)" : opt.input;
    std::string out_name = to_stdout ? "(stdout)" : opt.output;
    char summary[512];
    std::snprintf(summary, sizeof(summary),
      "%s : \033[36m%s\033[0m => \033[32m%s\033[0m, %s @ \033[32m%s/s\033[0m",
      in_name.c_str(), in_s, out_s, out_name.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }

  // Print finalize/fsync timing (deferred until after summary line)
  if (opt.verbosity >= V_VERBOSE) {
    if (finalize_ms > 100)
      vlog(V_VERBOSE, opt, "[O_DIRECT] finalize took " + std::to_string(int(finalize_ms)) + " ms\n");
    if (sync_ms > 100)
      vlog(V_VERBOSE, opt, "[FSYNC] took " + std::to_string(int(sync_ms)) + " ms\n");
  }

  // Writer-state report: did this run peg the writer, and if not, whose
  // fault was it?  Always measured (the counters are cheap), printed at -v.
  // See the Meter comments for the three-state model.
  if (opt.verbosity >= V_VERBOSE && opt.mode != Mode::TEST) {
    auto wdt = std::chrono::steady_clock::now() - meter.t0;
    const double wall_ns =
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(wdt).count();
    if (wall_ns > 0) {
      const double busy    = double(meter.writer_disk_ns.load())    / wall_ns;
      const double hol     = double(meter.writer_hol_ns.load())     / wall_ns;
      const double starved = double(meter.writer_starved_ns.load()) / wall_ns;
      const uint64_t hol_ns = meter.writer_hol_ns.load();
      const double avg_stuck = hol_ns > 0
          ? double(meter.writer_hol_depth_ns.load()) / double(hol_ns) : 0.0;
      // Healthy in-flight window: N workers legitimately buffer up to ~N
      // out-of-order frames; GPU batches land tens-to-hundreds at once, so
      // allow ~2 batches when GPUs are in play (the user's pin if given,
      // else the auto-tuner's typical settle point).  Crude — refine when
      // --adapt consumes these signals.
      double gpu_window = 0.0;
#ifdef HAVE_NVCOMP
      if (!opt.cpu_only)
        gpu_window = 2.0 * double(opt.gpu_batch_user_set ? opt.gpu_batch_cap
                                                         : (size_t)128);
#endif
      const double healthy_window =
          2.0 * resolve_cpu_threads(opt.cpu_threads) + gpu_window;
      // Reader line first (input before output).  Only when a counting reader
      // ran — the mmap reader is zero-copy/near-instant and reports nothing.
      const uint64_t r_io = meter.reader_io_ns.load();
      const uint64_t r_ps = meter.reader_parse_ns.load();
      const uint64_t r_cp = meter.reader_copy_ns.load();
      const uint64_t r_bk = meter.reader_blocked_ns.load();
      if (r_io + r_ps + r_cp + r_bk > 0) {
        // Percentages are per reader thread (counters sum across them).
        const double r_n = std::max(1, meter.reader_threads.load());
        const double r_wall = wall_ns * r_n;
        char rline[256];
        std::snprintf(rline, sizeof(rline),
          "[READER] io %.1f%% | parse %.1f%% | task-copy %.1f%% | blocked-downstream %.1f%%  "
          "(per thread, %d reader%s, %.2f s run)\n",
          double(r_io) / r_wall * 100.0, double(r_ps) / r_wall * 100.0,
          double(r_cp) / r_wall * 100.0, double(r_bk) / r_wall * 100.0,
          (int)r_n, r_n > 1 ? "s" : "", wall_ns / 1e9);
        vlog(V_VERBOSE, opt, rline);
        // Reader-saturated (the reader is the faucet) vs blocked-downstream
        // (the reader is idle waiting for the GPU/CPU consumers — they are the
        // faucet, not the reader).  Whichever dominates names the bottleneck.
        const double r_busy    = double(r_io + r_ps + r_cp) / r_wall;
        const double r_blocked = double(r_bk) / r_wall;
        if (r_blocked > r_busy && r_blocked >= 0.30)
          vlog(V_VERBOSE, opt, "[READER] verdict: reader is NOT the bottleneck — it stalls "
               "pushing to a full queue; the decompress consumers (GPU/CPU) are the faucet\n");
        else if (r_busy >= 0.85) {
          const char * why =
              (r_cp >= r_io && r_cp >= r_ps)
                  ? "the per-frame copy dominates; a zero-copy frame reader would raise the ceiling"
              : (r_ps >= r_io)
                  ? "frame-boundary parsing dominates; the serial parse is the spine (needs a frame index to split)"
                  : "the device/syscall path is the faucet; a faster source or read path is the only lever";
          vlog(V_VERBOSE, opt, std::string("[READER] verdict: reader saturated — ") + why + "\n");
        }
      }
      char wline[224];
      std::snprintf(wline, sizeof(wline),
        "[WRITER] write-path busy %.1f%% | head-of-line %.1f%% (avg %.0f frames stuck) | "
        "starved %.1f%%  (of %.2f s run)\n",
        busy * 100.0, hol * 100.0, avg_stuck, starved * 100.0, wall_ns / 1e9);
      vlog(V_VERBOSE, opt, wline);
      // Split the "busy" time into actual device I/O vs the single-threaded
      // zero-scan (sparse detection), so "busy 97%" isn't mistaken for "device
      // saturated" when much of it is CPU scanning on the write thread.
      const uint64_t io_ns   = meter.writer_iowrite_ns.load();
      const uint64_t disk_ns = meter.writer_disk_ns.load();
      if (opt.verbosity >= V_DEBUG && disk_ns > 0 && io_ns <= disk_ns) {  // -vv: deep write-time split
        // io_ns = time in dw_->write (bounce-copy into the aligned buffer + the
        // ::write syscall); the DirectWriter reports the ::write time alone, so
        // bounce-copy = io_ns - write_syscall.  Splits the write thread's busy
        // time three ways: actual device write / aligned bounce-copy / zero-scan.
        char sline[300];
        bool odirect = false;
#ifndef _WIN32
        odirect = g_odirect_used.load(std::memory_order_relaxed);
#endif
        const double scan_pct = double(disk_ns - io_ns) / double(disk_ns) * 100.0;
        if (odirect) {
          uint64_t wr_ns = g_odirect_write_ns.load(std::memory_order_relaxed);
          if (wr_ns > io_ns) wr_ns = io_ns;   // clamp (different clocks/overlap)
          std::snprintf(sline, sizeof(sline),
            "[WRITER]   of busy: O_DIRECT ::write %.1f%% | bounce-copy→aligned %.1f%% | zero-scan %.1f%%"
            "  (copy+write are serial on ONE thread; extract pipelines them)\n",
            double(wr_ns) / double(disk_ns) * 100.0,
            double(io_ns - wr_ns) / double(disk_ns) * 100.0, scan_pct);
        } else {
          std::snprintf(sline, sizeof(sline),
            "[WRITER]   of busy: buffered write %.1f%% | zero-scan %.1f%%\n",
            double(io_ns) / double(disk_ns) * 100.0, scan_pct);
        }
        vlog(V_DEBUG, opt, sline);
      }
      vlog(V_VERBOSE, opt,
           std::string("[WRITER] verdict: ")
           + writer_verdict(busy, hol, starved, avg_stuck, healthy_window) + "\n");
    }
  }


  } // end for (file_idx)

#ifndef _WIN32
  // --tar: some members were skipped or changed mid-read (warnings already
  // printed) — report a non-zero exit like GNU tar, but the archive is valid.
  if (opt.tar_mode && g_tar_had_errors.load(std::memory_order_relaxed)
      && exit_code == EXIT_OK)
    exit_code = EXIT_ERROR;
#endif

  return exit_code;
}

// Derive the output filename for a given input file and mode.
// Compress: foo.tar -> foo.tar.zst
// Decompress: foo.tar.zst -> foo.tar  (or foo.bin -> foo.bin.out)
static std::string derive_output(const std::string & input, Mode mode)
{
  if (input == "-") return "stdout";
  if (mode == Mode::COMPRESS) return input + ".zst";
  // Decompress: strip .zst if present, otherwise append .out
  if (input.size() > 4 && input.substr(input.size() - 4) == ".zst")
    return input.substr(0, input.size() - 4);
  return input + ".out";
}

/*======================================================================
 PCIe link generation detection (asymmetric mode v0.13.0+)

 GPU compress consistently wins on the hardware tier we target, but
 PCIe Gen3 D2H transfer cost makes hybrid *decompress* slower than CPU
 MT for every data type measured on consumer Gen3 GPUs (RTX 20-series
 and similar).  Default decompress to --cpu-only on Gen3 hardware to
 avoid that pitfall; on Gen4+ datacenter GPUs hybrid still wins.

 Returns the minimum link generation across visible GPUs:
   1, 2, 3 → slow PCIe (D2H dominates decompress)
   4+      → fast PCIe (D2H cheap; hybrid wins)
   0       → undetectable (NVML/sysfs unavailable, no NVIDIA GPUs)

 NVML is the primary path; /sys/bus/pci/devices walk is the fallback
 when the binary was built without NVML.
======================================================================*/
static int detect_min_pcie_gen()
{
  int min_gen = -1;
#ifdef HAVE_NVML
  if (nvmlInit_v2() == NVML_SUCCESS) {
    unsigned dev_count = 0;
    if (nvmlDeviceGetCount_v2(&dev_count) == NVML_SUCCESS) {
      for (unsigned i = 0; i < dev_count; ++i) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS) continue;
        // Use Max not Curr: idle GPUs drop their link to Gen1 for power
        // management, so Curr would lie about a Gen3 GPU at rest.  Max
        // reports the negotiated hardware ceiling (factoring in slot too:
        // a Gen3 card in a Gen2 slot reports Max=Gen2, which is correct).
        unsigned int gen = 0;
        if (nvmlDeviceGetMaxPcieLinkGeneration(dev, &gen) == NVML_SUCCESS) {
          int g = (int)gen;
          if (min_gen < 0 || g < min_gen) min_gen = g;
        }
      }
    }
    nvmlShutdown();
  }
#endif
  if (min_gen > 0) return min_gen;

  // sysfs fallback: walk /sys/bus/pci/devices, find NVIDIA (0x10de),
  // parse max_link_speed (not current_link_speed — idle GPUs drop their
  // link to Gen1 for power management).
  //   "2.5 GT/s"  → Gen1
  //   "5.0 GT/s"  → Gen2
  //   "8.0 GT/s"  → Gen3
  //   "16.0 GT/s" → Gen4
  //   "32.0 GT/s" → Gen5
  std::error_code ec;
  fs::path bus("/sys/bus/pci/devices");
  if (!fs::exists(bus, ec)) return 0;
  for (const auto & entry : fs::directory_iterator(bus, ec)) {
    if (ec) break;
    std::ifstream vf(entry.path() / "vendor");
    std::string vstr;
    if (!vf || !std::getline(vf, vstr)) continue;
    while (!vstr.empty() && std::isspace((unsigned char)vstr.back())) vstr.pop_back();
    if (vstr != "0x10de") continue;
    std::ifstream sf(entry.path() / "max_link_speed");
    std::string sstr;
    if (!sf || !std::getline(sf, sstr)) continue;
    int gen = 0;
    if      (sstr.find("2.5 GT/s")  != std::string::npos) gen = 1;
    else if (sstr.find("5.0 GT/s")  != std::string::npos) gen = 2;
    else if (sstr.find("8.0 GT/s")  != std::string::npos) gen = 3;
    else if (sstr.find("16.0 GT/s") != std::string::npos) gen = 4;
    else if (sstr.find("32.0 GT/s") != std::string::npos) gen = 5;
    else if (sstr.find("64.0 GT/s") != std::string::npos) gen = 6;
    if (gen > 0 && (min_gen < 0 || gen < min_gen)) min_gen = gen;
  }
  return (min_gen > 0) ? min_gen : 0;
}

/*======================================================================
 Apply asymmetric-mode default backend selection.  Called after
 parse_args.  No-op if the user explicitly chose a backend.

   COMPRESS                : hybrid (GPU compress wins on all tiers)
   DECOMPRESS / TEST       : depends on PCIe link gen
     Gen<4                 : cpu-only (D2H cost > GPU benefit)
     Gen4+ or undetectable : hybrid
======================================================================*/
static void apply_backend_defaults(Options & opt)
{
#ifdef HAVE_NVCOMP
  // Promote tuning flags to implicit --hybrid: if the user passed any
  // GPU- or hybrid-only knob (--gpu-batch, --cpu-share, --hybrid-floor, etc.)
  // but no explicit backend flag, treat it as if they had asked for --hybrid.
  // Otherwise asymmetric mode would silently flip them to cpu-only on Gen3
  // and their tuning hint would do nothing.  Same precedent as
  // --sliding-window implying --cpu-only.
  if (!opt.backend_user_set && opt.gpu_hybrid_tuning_seen) {
    opt.hybrid = true;
    opt.backend_user_set = true;
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt,
           "[ASYMMETRIC] GPU/hybrid tuning flag passed; implying --hybrid "
           "(override with --cpu-only)\n");
  }

  // PCIe-gen probe — drives the --direct default (compress + decompress) and
  // the decompress backend default further down.
  int gen = detect_min_pcie_gen();

  // --direct default: O_DIRECT output is a large win on fast-fabric (PCIe Gen4+)
  // boxes for BOTH compress and decompress — frame production outruns buffered
  // writeback, scaling with output volume (win on large output, neutral on
  // tiny) — and a regression on Gen<4, which stay buffered.  Enable on Gen4+
  // unless the user passed --direct/--no-direct.  Skip test mode (writes
  // nothing).  Backend-independent (the win is the output write path), so it
  // runs before the backend_user_set return and covers cpu-only/hybrid/gpu-only
  // alike.  Harmless for pipe/stdout output: DirectWriter only activates for
  // regular-file targets.  See CHANGELOG / the --direct asymmetry finding.
  if (opt.mode != Mode::TEST && !opt.direct_io_user_set && gen >= 4) {
    opt.direct_io = true;
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "[O_DIRECT] PCIe Gen" + std::to_string(gen)
           + " detected; defaulting output to --direct (override with --no-direct)\n");
  }

  if (opt.backend_user_set) return;

  if (opt.mode == Mode::COMPRESS) { opt.hybrid = true; return; }
  if (gen > 0 && gen < 4) {
    opt.cpu_only = true;
    // Mirror the parse_args silencing: --cpu-batch in cpu-only mode causes
    // stop-and-go.  parse_args's own check ran before we flipped cpu_only,
    // so it missed this auto-flip path; redo it here.
    if (opt.cpu_queue_min > 0) {
      if (opt.verbosity >= V_ERROR)
        std::cerr << "gzstd: note: --cpu-batch is ignored in --cpu-only mode "
                     "(asymmetric default; override with --hybrid)\n";
      opt.cpu_queue_min = 0;
    }
    // Show this at default verbosity: users on Gen3 hardware otherwise see
    // no GPU activity and have no way to know the runtime made that choice.
    // Suppress for -l: a listing shouldn't announce a decompress backend (plain
    // -l doesn't even decompress).
    if (opt.verbosity >= V_DEFAULT && !opt.list_mode) {
      std::ostringstream os;
      os << "gzstd: PCIe Gen" << gen
         << " detected; defaulting decompress to --cpu-only "
            "(override with --hybrid or --gpu-only)\n";
      std::fprintf(stderr, "%s", os.str().c_str());
    }
  } else {
    opt.hybrid = true;
    if (gen >= 4 && opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[ASYMMETRIC] PCIe Gen" << gen
         << " detected; defaulting decompress to --hybrid";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  }
#else
  (void)opt;
#endif
}

// zstd/gzip compat: is `a` a bundled group of no-arg short flags like "-dcf"?
// True only when every char after a single leading '-' is one of the no-arg
// operation flags {d,t,k,f,c}.  Anything containing a value-taking flag
// (-o/-T/-M/-B/-D), a digit (numeric levels, -M#, -b#), a long option ("--…"),
// or the repeat flags (-vv/-vvv/-qq — v/q aren't in the set) returns false and
// is left for the exact-match loop to handle unchanged.  v/q are intentionally
// excluded so their repeat semantics survive; bundle verbosity flags separately.
static bool is_bundleable_short_group(const std::string & a)
{
  if (a.size() < 3 || a[0] != '-' || a[1] == '-') return false;
  for (size_t i = 1; i < a.size(); ++i) {
    char c = a[i];
    if (c != 'd' && c != 't' && c != 'k' && c != 'f' && c != 'c') return false;
  }
  return true;
}

/* parse_args at end */
static Options parse_args(int argc, char ** argv)
{
  Options opt;

  // zstd/gzip compat: expand bundled short-flag groups (-dc, -dkf, …) into
  // individual flags up front, so the match loop and all its value-flag
  // (argv[++i]) handling work unchanged.  Only all-no-arg-operation-flag groups
  // expand; value flags, digits, and -vv/-qq pass through untouched (see
  // is_bundleable_short_group).  xargs/xv own the rewritten argv for the rest
  // of this function; everything below operates on the expanded argc/argv.
  std::vector<std::string> xargs;
  xargs.reserve((size_t)std::max(argc, 1));
  if (argc > 0) xargs.push_back(argv[0] ? argv[0] : "gzstd");
  {
    bool end_of_opts = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i] ? argv[i] : "";
      if (!end_of_opts && a == "--") { end_of_opts = true; xargs.push_back(a); continue; }
      if (!end_of_opts && is_bundleable_short_group(a))
        for (size_t k = 1; k < a.size(); ++k) xargs.push_back(std::string("-") + a[k]);
      else
        xargs.push_back(std::move(a));
    }
  }
  std::vector<char *> xv;
  xv.reserve(xargs.size() + 1);
  for (auto & s : xargs) xv.push_back(const_cast<char *>(s.c_str()));
  argc = (int)xv.size();
  argv = xv.data();

  // Pre-scan for verbosity flags so zstd-compat warnings emitted during the
  // main parse below can be suppressed by `-q` / `-qq` regardless of the
  // order flags appear on the command line.
  g_verbosity_for_compat = 2; // V_DEFAULT
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-q" || a == "--quiet") g_verbosity_for_compat = 1;
    else if (a == "-qq" || a == "--silent") g_verbosity_for_compat = 0;
  }

  // Detect invocation name: if basename starts with "un" (e.g. ungzstd),
  // default to decompress mode (like gzip/gunzip, zstd/unzstd)
  if (argc > 0 && argv[0]) {
    std::string prog = argv[0];
    // Extract basename (strip directory path)
    auto slash = prog.find_last_of("/\\");
    std::string base = (slash != std::string::npos) ? prog.substr(slash + 1) : prog;
    if (base.size() >= 2 && base[0] == 'u' && base[1] == 'n') {
      opt.mode = Mode::DECOMPRESS;
    }
  }

  std::string pinned_value_tmp; // scratch buffer for --pinned VALUE parsing
  (void)pinned_value_tmp; // referenced only under HAVE_NVCOMP
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "-?") { print_help(); std::exit(EXIT_OK); }
    else if (a == "--help" || a == "-H") { print_help_long(); std::exit(EXIT_OK); }
    else if (a == "-V" || a == "--version") { print_version(); std::exit(EXIT_OK); }
    else if (a == "-d" || a == "--decompress" || a == "--uncompress")
      opt.mode = Mode::DECOMPRESS;
    else if (a == "-t" || a == "--test") opt.mode = Mode::TEST;
    else if (a == "-k" || a == "--keep") opt.keep = true;
    else if (a == "--rm") { opt.keep = false; }
    else if (a == "-f" || a == "--force") opt.force = true;
    else if (a == "--overwrite") { opt.force = true; opt.unsafe_overwrite = true; }
    else if (a == "--sparse") opt.sparse_mode = 1;
    else if (a == "--no-sparse") opt.sparse_mode = 0;
    else if (a == "--sync-output") opt.sync_output = true;
    else if (a == "--direct") { opt.direct_io = true; opt.direct_io_user_set = true; }
    else if (a == "--no-direct") { opt.direct_io = false; opt.direct_io_user_set = true; }
    else if (a == "--mmap" || a == "--mmap=on")  { opt.use_mmap = true;  opt.mmap_user_set = true; }
    else if (a == "--no-mmap" || a == "--mmap=off") { opt.use_mmap = false; opt.mmap_user_set = true; }
    else if (a == "--cold") opt.cold_read = true;
    else if (a == "--direct-read") opt.direct_read = true;
    else if (parse_num_arg("read-threads", i, argc, argv, opt.read_threads)) { }
    else if (parse_num_arg("write-threads", i, argc, argv, opt.write_threads)) { }
    // --tar introduces the archive sub-invocation: positionals that follow it
    // become archive members, and the tar-specific options (--exclude,
    // --numeric-owner, --one-file-system) may sit after --tar alongside them.
    // Flags still parse normally, so a source whose name begins with '-' must
    // be written ./-name (or placed after a literal `--`), as with real tar.
    else if (a == "--tar") opt.tar_mode = true;
    else if (a == "--exclude") {
      if (i + 1 >= argc) die_usage("missing value for --exclude");
      opt.tar_excludes.push_back(argv[++i]);
    }
    else if (a.rfind("--exclude=", 0) == 0) {
      std::string v = a.substr(10);
      if (v.empty()) die_usage("missing value for --exclude");
      opt.tar_excludes.push_back(v);
    }
    else if (a == "--numeric-owner") opt.tar_numeric_owner = true;
    else if (a == "--one-file-system") opt.tar_one_file_system = true;
    else if (a == "--xattrs") opt.tar_xattrs = true;
    else if (a == "--acls") opt.tar_acls = true;
    else if (a == "-C" || a == "--directory") {
      if (i + 1 >= argc) die_usage("missing value for " + a);
      opt.tar_dest = argv[++i];
    }
    else if (a.rfind("--directory=", 0) == 0) {
      opt.tar_dest = a.substr(12);
      if (opt.tar_dest.empty()) die_usage("missing value for --directory");
    }
    else if (a == "--preallocate" || a == "--preallocate=on")  opt.preallocate_output = true;
    else if (a == "--no-preallocate" || a == "--preallocate=off") opt.preallocate_output = false;
    else if (a == "--verify") opt.verify = true;
    else if (a == "--keep-going") opt.keep_going = true;
    else if (a.rfind("--verify-retries=", 0) == 0) {
      std::string v = a.substr(17);
      if (v.empty()) die_usage("missing value for --verify-retries");
      char * end = nullptr;
      long n = std::strtol(v.c_str(), &end, 10);
      if (*end != '\0' || n < 0) die_usage("--verify-retries must be a non-negative integer (0 = unlimited)");
      opt.verify = true;          // implied: you asked for a retry bound
      opt.verify_retries = (int)n;
    }
    else if (a == "-c" || a == "--stdout" || a == "--to-stdout") opt.to_stdout = true;
    else if (a == "-v" || a == "--verbose") opt.verbosity = V_VERBOSE;
    else if (a == "-vv") opt.verbosity = V_DEBUG;
    else if (a == "-vvv") opt.verbosity = V_TRACE;
    else if (a == "-q" || a == "--quiet") opt.verbosity = V_ERROR;
    else if (a == "-qq" || a == "--silent") opt.verbosity = V_SILENT;
    else if (a == "--progress") {
      opt.force_progress = true;
      if (opt.verbosity < V_DEFAULT) opt.verbosity = V_DEFAULT;
    }
    else if (a == "--no-progress") {
      opt.force_progress = false;
      if (opt.verbosity == V_DEFAULT) opt.verbosity = V_ERROR;
    }
    else if (a.size() >= 2 && a[0] == '-' && a[1] >= '0' && a[1] <= '9') {
      bool all_digits = true;
      for (size_t k = 1; k < a.size(); ++k) { if (a[k] < '0' || a[k] > '9') { all_digits = false; break; } }
      // Reject "-5x" etc. explicitly: this else-if already matched, so falling
      // through silently swallowed the malformed flag (and compressed at the
      // default level).  parse_int_value also catches overflow (-9999999...).
      if (!all_digits) die_usage("unknown option: " + a);
      int lvl = parse_int_value(a, a.substr(1));
      if (lvl < 1) die_usage("invalid compression level (must be 1..22)");
      if (lvl > 22) die_usage("invalid compression level (max 22)");
      opt.level = lvl; opt.level_user_set = true; continue;
    }
    else if (a == "--fast") { opt.fast_flag = true; opt.level = 1; opt.level_user_set = true; }
    else if (a.rfind("--fast=", 0) == 0) {
      // zstd-compat: --fast=# uses negative-ish fast levels.  Our backend
      // accepts 1..22, so clamp "--fast=N" to level 1 (fastest) and warn if N>1.
      std::string v = a.substr(7);
      int n = v.empty() ? 1 : parse_int_value("--fast", v);
      if (n < 1) n = 1;
      if (n > 1) warn_ignored_zstd_opt("--fast=" + v, "gzstd minimum level is 1");
      opt.fast_flag = true; opt.level = 1; opt.level_user_set = true;
    }
    else if (a == "--best") { opt.best_flag = true; opt.level = 19; opt.level_user_set = true; }
    else if (a == "--ultra") { opt.ultra = true; }
    else if (a == "--cpu-only") { opt.cpu_only = true; opt.backend_user_set = true; }
    else if (a == "--sliding-window") opt.sliding_window = true;
    else if (a == "--hybrid") { opt.hybrid = true; opt.backend_user_set = true; }
    else if (a.rfind("-T", 0) == 0 && a.size() > 2) {
      std::string val = a.substr(2);
      if (!val.empty() && val[0] == '=') val = val.substr(1);
      if (val.empty()) die_usage("missing value for -T");
      int th = parse_int_value("-T", val);  // graceful error on a bad attached value
      // -T0 means "use all available threads" (like zstd)
      opt.cpu_threads = (th == 0) ? -1 : th;
    }
    else if (a == "-T" || a == "--threads" || a.rfind("--threads=", 0) == 0) {
      // Value forms: `--threads=N` (attached), or `-T N` / `--threads N`
      // (separate).  Only consume the next token as the count when it actually
      // looks like an integer — a bare `-T` (e.g. `-T --cpu-only`, `-T file.zst`,
      // or a trailing `-T`) is tolerated and falls back to the default thread
      // count rather than crashing std::stoi on a non-numeric token.
      if (a.rfind("--threads=", 0) == 0) {
        int th = parse_int_value("--threads", a.substr(10));
        opt.cpu_threads = (th == 0) ? -1 : th;
      } else if (i + 1 < argc && looks_like_int(argv[i + 1])) {
        int th = parse_int_value(a, argv[++i]);
        opt.cpu_threads = (th == 0) ? -1 : th;
      }
      // else: no usable numeric value — leave cpu_threads at its default (auto)
    }
    else if (a == "-o" || a == "--output") {
      if (i + 1 >= argc) die_usage("missing value for " + a);
      opt.output = argv[++i];
      // -o implies not-stdout unless the path is literally "stdout" or "-"
      if (opt.output == "-") { opt.to_stdout = true; opt.output = "stdout"; }
    }
    else if (a.rfind("--output=", 0) == 0) {
      opt.output = a.substr(9);
      if (opt.output.empty()) die_usage("missing value for --output");
      if (opt.output == "-") { opt.to_stdout = true; opt.output = "stdout"; }
    }
    else if (parse_double_arg("cpu-share", i, argc, argv, opt.cpu_share)) { opt.gpu_hybrid_tuning_seen = true; }
    else if (parse_str_arg("stats-json", i, argc, argv, opt.stats_json)) {}
    else if (parse_num_arg("chunk-size", i, argc, argv, opt.chunk_mib, &opt.chunk_user_set)) {}
    else if (parse_num_arg("cpu-batch", i, argc, argv, opt.cpu_queue_min, nullptr)) { opt.gpu_hybrid_tuning_seen = true; }
    else if (a.rfind("--hybrid-floor=", 0) == 0) {
      std::string v = a.substr(15);
      std::transform(v.begin(), v.end(), v.begin(), ::tolower);
      if (v == "auto")         opt.hybrid_floor_mode = Options::HybridFloorMode::AUTO;
      else if (v == "nominal") opt.hybrid_floor_mode = Options::HybridFloorMode::NOMINAL;
      else if (v == "off")     opt.hybrid_floor_mode = Options::HybridFloorMode::OFF;
      else die_usage("invalid value for --hybrid-floor (expected auto|nominal|off)");
      opt.gpu_hybrid_tuning_seen = true;
    }
    else if (parse_double_arg("hybrid-floor-factor", i, argc, argv,
                              opt.hybrid_floor_factor)) {
      // v0.13.5+: cap raised from 1.0 to 4.0 to permit proactive batch
      // reservation (factor=N means N full GPU rounds queued ahead of CPU).
      if (opt.hybrid_floor_factor < 0.0 || opt.hybrid_floor_factor > 4.0)
        die_usage("--hybrid-floor-factor must be in [0.0, 4.0]");
      opt.gpu_hybrid_tuning_seen = true;
    }
    else if (parse_int_arg("throttle-factor", i, argc, argv, opt.throttle_factor)) {
      if (opt.throttle_factor < 1)
        die_usage("--throttle-factor must be >= 1");
    }
    else if (parse_int_arg("throttle-frames", i, argc, argv, opt.throttle_frames)) {
      // 0 = disabled (no throttle); -1 = auto (use formula); >0 = explicit cap.
      // Reject -2 and lower.
      if (opt.throttle_frames < -1)
        die_usage("--throttle-frames must be 0 (disabled), -1 (auto), or >= 1");
    }
    else if (a == "--no-throttle") opt.throttle_frames = 0;  // alias for --throttle-frames=0
#ifdef HAVE_NVCOMP
    else if (a == "--gpu-only") { opt.gpu_only = true; opt.backend_user_set = true; }
    else if (parse_num_arg("gpu-batch", i, argc, argv, opt.gpu_batch_cap)) { opt.gpu_batch_user_set = true; opt.gpu_hybrid_tuning_seen = true; }
    else if (parse_double_arg("gpu-mem-frac", i, argc, argv, opt.gpu_mem_fraction)) {
      // Reject obvious nonsense; warn-and-clamp the soft bounds so existing
      // scripts that pass slightly aggressive values still work but the user
      // learns why they didn't get what they asked for.
      if (opt.gpu_mem_fraction <= 0.0 || opt.gpu_mem_fraction >= 1.0)
        die_usage("--gpu-mem-frac must be in (0.0, 1.0)");
      if (opt.gpu_mem_fraction < 0.10 || opt.gpu_mem_fraction > 0.95) {
        double requested = opt.gpu_mem_fraction;
        opt.gpu_mem_fraction = (requested < 0.10) ? 0.10 : 0.95;
        if (opt.verbosity >= V_ERROR) {
          std::ostringstream os;
          os << "gzstd: warning: --gpu-mem-frac=" << requested
             << " outside safe range [0.10, 0.95]; clamping to "
             << opt.gpu_mem_fraction << "\n";
          std::fprintf(stderr, "%s", os.str().c_str());
        }
      }
      opt.gpu_hybrid_tuning_seen = true;
    }
    else if (parse_num_arg("gpu-streams", i, argc, argv, opt.gpu_streams)) { opt.gpu_hybrid_tuning_seen = true; }
    else if (parse_int_arg("gpu-devices", i, argc, argv, opt.gpu_devices)) { opt.gpu_hybrid_tuning_seen = true; }
    else if (a == "--no-pinned") { opt.pin_mode = PinMode::OFF; opt.gpu_hybrid_tuning_seen = true; }
    else if (parse_str_arg("pinned", i, argc, argv, pinned_value_tmp)) {
      std::transform(pinned_value_tmp.begin(), pinned_value_tmp.end(),
                     pinned_value_tmp.begin(), ::tolower);
      if (pinned_value_tmp == "auto") opt.pin_mode = PinMode::AUTO;
      else if (pinned_value_tmp == "on") opt.pin_mode = PinMode::ON;
      else if (pinned_value_tmp == "off") opt.pin_mode = PinMode::OFF;
      else die_usage("invalid value for --pinned (expected auto|on|off)");
      opt.gpu_hybrid_tuning_seen = true;
    }
#else
    else if (a.rfind("--gpu-", 0) == 0 || a == "--gpu-only") {
      // Ignored on CPU-only builds
    }
#endif
    else if (a == "--") {
      // End of options — everything after this is a literal path (archive
      // sources in --tar mode, input files otherwise).
      for (++i; i < argc; ++i)
        (opt.tar_mode ? opt.tar_sources : opt.inputs).push_back(argv[i]);
    }
    // === zstd-compat layer: accept flags we don't implement so gzstd can
    // serve as a drop-in replacement.  Real aliases are handled above; below
    // is silent or warning acceptance for the rest.  Keep in sync with zstd
    // --help. ===
    //
    // Silent no-ops: zstd defaults we already match, so the flag is a no-op.
    else if (a == "--asyncio" || a == "--no-asyncio") {}
    else if (a == "--check" || a == "--no-check") {}
    else if (a == "--format=zstd") {}
    else if (a == "--no-dictID") {}
    else if (a == "--compress-literals" || a == "--no-compress-literals") {}
    else if (a == "--row-match-finder" || a == "--no-row-match-finder") {}
    else if (a == "--mmap-dict" || a == "--no-mmap-dict") {}
    else if (a.rfind("--stream-size=", 0) == 0) {}
    else if (a.rfind("--size-hint=", 0) == 0) {}
    else if (a.rfind("--target-compressed-block-size=", 0) == 0) {}
    else if (a.rfind("--auto-threads=", 0) == 0) {}
    // Real mapping: --single-thread ≈ -T 1
    else if (a == "--single-thread") { opt.cpu_threads = 1; }
    // Warn no-ops: zstd features gzstd does not implement.
    else if (a == "--adapt" || a.rfind("--adapt=", 0) == 0) {
      warn_ignored_zstd_opt("--adapt", "gzstd does not adapt level during compression");
    }
    else if (a == "--long" || a.rfind("--long=", 0) == 0) {
      warn_ignored_zstd_opt(a, "use --ultra for large window levels");
    }
    else if (a.rfind("--patch-from=", 0) == 0) {
      warn_ignored_zstd_opt("--patch-from", "dictionary/diff not supported");
    }
    else if (a == "--patch-from") {
      warn_ignored_zstd_opt("--patch-from", "dictionary/diff not supported");
      if (i + 1 < argc) ++i;
    }
    else if (a == "--rsyncable") {
      warn_ignored_zstd_opt("--rsyncable");
    }
    else if (a == "--exclude-compressed") {
      warn_ignored_zstd_opt("--exclude-compressed");
    }
    else if (a == "--format=gzip" || a == "--format=xz"
          || a == "--format=lzma" || a == "--format=lz4") {
      warn_ignored_zstd_opt(a, "gzstd emits only zstd format");
    }
    else if (a == "--pass-through" || a == "--no-pass-through") {
      warn_ignored_zstd_opt(a);
    }
    else if (a == "-r" || a == "--recursive") {
      warn_ignored_zstd_opt("-r/--recursive", "recurse into directories yourself");
    }
    else if (a == "-l" || a == "--list") {
      opt.list_mode = true;
      opt.mode = Mode::DECOMPRESS;   // a read-only op; main dispatches list before any decompress
    }
    else if (eat_zstd_value_opt("filelist", i, argc, argv)) {
      warn_ignored_zstd_opt("--filelist");
    }
    else if (eat_zstd_value_opt("output-dir-flat", i, argc, argv)
          || eat_zstd_value_opt("output-dir-mirror", i, argc, argv)) {
      warn_ignored_zstd_opt(a, "output dir flags not supported");
    }
    else if (eat_zstd_value_opt("trace", i, argc, argv)) {
      warn_ignored_zstd_opt("--trace");
    }
    // Dictionary flags (all warn-no-op)
    else if (a == "-D" || a == "--dict" || a == "--dictionary") {
      warn_ignored_zstd_opt(a, "dictionary compression not supported");
      if (i + 1 < argc) ++i;
    }
    else if (a.rfind("--dict=", 0) == 0 || a.rfind("--dictionary=", 0) == 0) {
      warn_ignored_zstd_opt(a, "dictionary compression not supported");
    }
    // Training mode (warn; zstd exits early with these — gzstd will proceed
    // as normal compression, producing no dictionary)
    else if (a == "--train" || a.rfind("--train-", 0) == 0
          || a.rfind("--maxdict", 0) == 0 || a.rfind("--dictID", 0) == 0) {
      warn_ignored_zstd_opt(a, "dictionary training not supported");
    }
    // Memory limit (zstd-compat `-M#` / `-M N` / `--memlimit[=N]` /
    // `--memory[=N]`, value in MiB).  Applied to decompress via
    // ZSTD_d_windowLogMax and to compress as a throttle-budget cap.
    else if (a.size() > 2 && a[0] == '-' && a[1] == 'M'
          && (a[2] >= '0' && a[2] <= '9')) {
      opt.mem_limit_mib = (size_t)parse_u64_value("-M", a.substr(2));
    }
    else if (a == "-M") {
      if (i + 1 >= argc) die_usage("missing value for -M");
      opt.mem_limit_mib = (size_t)parse_u64_value("-M", argv[++i]);
    }
    else if (a.rfind("--memlimit=", 0) == 0) {
      opt.mem_limit_mib = (size_t)parse_u64_value("--memlimit", a.substr(11));
    }
    else if (a == "--memlimit") {
      if (i + 1 >= argc) die_usage("missing value for --memlimit");
      opt.mem_limit_mib = (size_t)parse_u64_value("--memlimit", argv[++i]);
    }
    else if (a.rfind("--memory=", 0) == 0) {
      opt.mem_limit_mib = (size_t)parse_u64_value("--memory", a.substr(9));
    }
    else if (a == "--memory") {
      if (i + 1 >= argc) die_usage("missing value for --memory");
      opt.mem_limit_mib = (size_t)parse_u64_value("--memory", argv[++i]);
    }
    // Job size: `-B#` in compression (bytes).  gzstd uses --chunk-size in MiB
    // with different semantics (one frame per chunk).  Warn.
    else if (a.size() > 2 && a[0] == '-' && a[1] == 'B'
          && (a[2] >= '0' && a[2] <= '9')) {
      warn_ignored_zstd_opt(a, "use --chunk-size N (MiB) for frame size");
    }
    else if (a == "-B") {
      warn_ignored_zstd_opt("-B", "use --chunk-size N (MiB)");
      if (i + 1 < argc) ++i;
    }
    // Benchmark mode (zstd's -b/-e/-i/-S and --priority=rt).  gzstd has its
    // own benchmark harness (gzstd-benchmark.sh); warn and no-op.
    else if ((a.size() > 2 && (a[0] == '-')
           && (a[1] == 'b' || a[1] == 'e' || a[1] == 'i')
           && (a[2] >= '0' && a[2] <= '9'))
          || a == "-S"
          || a.rfind("--priority=", 0) == 0) {
      warn_ignored_zstd_opt(a, "benchmark mode not implemented; see gzstd-benchmark.sh");
    }
    else if (a.size() > 1 && a[0] == '-' && a != "-") {
      die_usage("unknown option: " + a);
    }
    else { (opt.tar_mode ? opt.tar_sources : opt.inputs).push_back(a); }
  }
  if (opt.inputs.empty()) opt.inputs.push_back("-");

  // -o with multiple files is ambiguous
  if (!opt.output.empty() && opt.inputs.size() > 1)
    die_usage("-o/--output cannot be used with multiple input files");

  // Set opt.input to first file for backward compat in single-file paths
  opt.input = opt.inputs[0];

  // When reading from stdin, default to stdout output (pipe-friendly, like gzip/zstd)
  if (opt.input == "-" && opt.output.empty()) opt.to_stdout = true;

  // When writing to stdout (-c), always keep the input file (can't delete stdin,
  // and deleting a named file when output goes to stdout matches gzip behavior)
  if (opt.to_stdout) opt.keep = true;

  if (opt.tar_mode) {
#ifdef _WIN32
    die_usage("--tar is not supported on this platform");
#endif
    if (opt.sliding_window)
      die_usage("--tar is incompatible with --sliding-window");
    // Sources/archives follow --tar; anything in inputs leaked in before it
    // (opt.inputs is otherwise just the "-" placeholder added below).
    for (const std::string & s : opt.inputs)
      if (s != "-")
        die_usage(std::string("--tar paths must come after --tar (found '") + s + "' before it)");
    if (opt.tar_sources.empty())
      die_usage(opt.mode == Mode::COMPRESS
                ? "--tar requires at least one file or directory to archive"
                : opt.mode == Mode::TEST
                ? "-t --tar requires at least one archive to test"
                : "-d --tar requires at least one archive to extract");
    if (opt.mode == Mode::DECOMPRESS || opt.mode == Mode::TEST) {
      // Extraction/test: the post-`--tar` positionals are the input archive(s);
      // the first becomes opt.input so the existing single-input plumbing works.
      if (!opt.tar_excludes.empty())
        die_usage("--exclude applies to creation (--tar), not extraction/test");
      opt.input = opt.tar_sources[0];
    }
    opt.keep = true;  // never delete archive sources / input archives
  }
  else if (!opt.tar_excludes.empty() || opt.tar_numeric_owner || opt.tar_one_file_system)
    die_usage("--exclude/--numeric-owner/--one-file-system require --tar");

  if (opt.sliding_window) {
    if (opt.mode != Mode::COMPRESS)
      die_usage("--sliding-window only applies to compression");
    if (opt.gpu_only)
      die_usage("--sliding-window is incompatible with --gpu-only (GPU cannot use sliding window context)");
    if (opt.hybrid)
      die_usage("--sliding-window is incompatible with --hybrid (GPU cannot use sliding window context)");
    if (!opt.cpu_only) {
      vlog(V_NORMAL, opt, "warning: --sliding-window implies --cpu-only (GPU cannot use sliding window context)\n");
      opt.cpu_only = true;
      opt.backend_user_set = true;
    }
  }

  if (opt.verify && opt.mode != Mode::COMPRESS) {
    // Decompression already validates every frame's checksum as it runs, so
    // --verify is redundant there — accept it quietly so scripts that always
    // pass --verify still work, but say it's a no-op.
    vlog(V_VERBOSE, opt, "note: --verify applies to compression; decompression already validates checksums\n");
    opt.verify = false;
  }

  if (opt.keep_going) {
    if (opt.mode != Mode::DECOMPRESS) {
      // Recovery only makes sense when extracting (-d).  -t in particular would
      // otherwise record damage but still print "OK"; accept quietly for scripts.
      vlog(V_VERBOSE, opt, "note: --keep-going applies to decompression (-d); ignored otherwise\n");
      opt.keep_going = false;
    } else {
      if (!opt.cpu_only) {
        // The CPU decoder is the authoritative checksum validator and the recovery
        // path lives there; recovery is rare and not perf-critical, so route the
        // whole run through it rather than also threading it through nvCOMP.
        vlog(V_VERBOSE, opt, "note: --keep-going implies --cpu-only (the authoritative checksum validator)\n");
        opt.cpu_only = true;
        opt.backend_user_set = true;
      }
      // Force the single-threaded reader so each frame's output offset is a simple
      // prefix sum (needed to map damage to byte ranges / tar members); the
      // parallel readers don't process frames in a single offset-accumulating order.
      opt.read_threads = 1;
    }
  }

#ifdef HAVE_NVCOMP
  if (opt.gpu_batch_cap == 0) opt.gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  // Decompress benefits massively from large batches  nvCOMP kernel launch
  // has significant per-launch overhead (Huffman table setup,
  // scratch allocation).  But very large batches delay the writer since it
  // can't start until the batch finishes.  Auto-tune for ~4 batches total
  // to balance kernel efficiency with GPU-writer overlap.
  // Decompress batch: start based on file size, auto-tuner refines from there.
  // Small files: start low (16) so the tuner converges quickly.
  // Large files (>75 GiB): start high (256)  benchmarks show H100s perform
  // well at large batch sizes, and starting low wastes minutes exploring upward.
  if (!opt.gpu_batch_user_set && (opt.mode == Mode::DECOMPRESS || opt.mode == Mode::TEST)) {
    uint64_t input_size = 0;
    if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
      input_size = (uint64_t)fs::file_size(opt.input);
    if (input_size > 75ULL * 1024 * 1024 * 1024) {
      opt.gpu_batch_cap = 256;
    } else if (input_size > 10ULL * 1024 * 1024 * 1024) {
      opt.gpu_batch_cap = 64;
    } else {
      opt.gpu_batch_cap = DEFAULT_GPU_DECOMP_BATCH_CAP;  // 16
    }
  }
  if (opt.gpu_streams == 0) {
    // Auto: 2 streams for verify (-t) where no write bottleneck exists,
    // 1 stream for compress/decompress (larger batches win over overlap)
    opt.gpu_streams = (opt.mode == Mode::TEST) ? 2 : DEFAULT_GPU_STREAMS;
  }
  if (!(opt.gpu_mem_fraction > 0.0 && opt.gpu_mem_fraction < 1.0)) opt.gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  if (opt.gpu_mem_fraction < 0.10) opt.gpu_mem_fraction = 0.10;
  if (opt.gpu_mem_fraction > 0.95) opt.gpu_mem_fraction = 0.95;
#endif
  if (opt.chunk_mib == 0) opt.chunk_mib = DEFAULT_CHUNK_MIB;
  if (opt.chunk_user_set && opt.chunk_mib > 1024) {
    if (opt.verbosity >= V_ERROR)
      std::cerr << "gzstd: warning: --chunk-size=" << opt.chunk_mib
                << " is very large (max recommended: 1024 MiB)\n";
  }
  if (opt.mode == Mode::TEST) { opt.to_stdout = true; opt.output = "stdout"; }
  else if (opt.to_stdout && opt.output.empty()) {
    opt.output = "stdout";
  }
  else if (!opt.to_stdout && opt.output.empty() && opt.inputs.size() == 1) {
    // Single file, no -o: auto-derive output name
    opt.output = derive_output(opt.input, opt.mode);
  }
  // Multi-file with no -o: output is derived per-file in main loop

  // Refuse to write compressed (binary) data to an interactive terminal, like
  // zstd/gzip.  Catches the common slip of forgetting -o (e.g. `gzstd --tar
  // ~/backup` with no -o would otherwise spray a .tar.zst at the screen).
  // Compression only — decompressed output to a terminal is fine.  -f overrides.
  if (opt.mode == Mode::COMPRESS && opt.to_stdout && !opt.force && is_stdout_tty())
    die_usage("refusing to write compressed data to the terminal "
              "(use -o FILE, redirect stdout, or -f to force)");

  if (opt.gpu_only && (opt.cpu_only || opt.hybrid)) die_usage("--gpu-only cannot be combined with --cpu-only or --hybrid");
  if (opt.cpu_only && opt.hybrid) die_usage("--cpu-only cannot be combined with --hybrid");
  if (opt.level >= 20 && opt.level <= 22 && !opt.ultra) die_usage("levels 20..22 require --ultra (zstd-compatible behavior)");

  // --cpu-batch is a hybrid-only tuning knob.  In --cpu-only mode it causes
  // a stop-and-go pattern (all threads idle until queue depth >= N, then stampede)
  // that wastes CPU time and hammers the kernel with CV contention.
  if (opt.cpu_only && opt.cpu_queue_min > 0) {
    if (opt.verbosity >= V_ERROR)
      std::cerr << "gzstd: note: --cpu-batch is ignored in --cpu-only mode (hybrid-only option)\n";
    opt.cpu_queue_min = 0;
  }

  // Backend default selection moved to apply_backend_defaults() so we can
  // run PCIe-generation detection (asymmetric mode: hybrid for compress,
  // CPU-only for Gen3 decompress where D2H cost dwarfs GPU benefit).

  // Auto-lower verbosity when used as a pipe (both stdin and stdout are non-TTY)
  // but only if the user hasn't explicitly set verbosity via flags.
  // V_DEFAULT stays V_DEFAULT (keeps progress on a TTY stderr), but if stderr
  // is also not a TTY we let progress_loop decide (it checks is_stderr_tty).
  // We do NOT auto-quiet here; the progress_loop already handles the TTY check.

  // Sync global verbosity for die() (which has no access to Options)
  g_verbosity = opt.verbosity;

  // Cache TTY check once; used by vlog() to gate ANSI color on verbose output.
  g_color_stderr = is_stderr_tty();

  return opt;
}
