// gzstd.cpp  Hybrid CPU+GPU Zstd (adaptive share)
static constexpr const char * GZSTD_VERSION = "0.9.83";
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
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <deque>
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
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
#endif
#include <unistd.h>
#include <fcntl.h>
#ifdef _WIN32
 #include <io.h>
#endif
#ifdef HAVE_NVCOMP
 #include <cuda_runtime.h>
 #include <nvcomp/zstd.h>
 #ifdef HAVE_NVML
 #include <nvml.h>
 #endif
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
    std::remove(g_tmp_path);   // best-effort; ignore errors
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
static const size_t DEFAULT_CHUNK_MIB = 32;
#ifdef HAVE_NVCOMP
static const size_t GPU_SUBCHUNK_MAX = size_t(16) * ONE_MIB; // max GPU subchunk
static const size_t DEFAULT_GPU_BATCH_CAP = 8;    // per device  smaller batches launch sooner
static const size_t DEFAULT_GPU_DECOMP_BATCH_CAP = 256; // decompress benefits from large batches (less per-launch overhead)
static const double DEFAULT_GPU_MEM_FRACTION = 0.60; // fraction of free VRAM to use
static const size_t DEFAULT_GPU_STREAMS = 1;       // single stream avoids context-switch overhead
static const size_t AUTO_HOST_CHUNK_MIN_MIB = 32;
static const size_t AUTO_HOST_CHUNK_MAX_MIB = 512;
static const double GROW_CHECK_SEC = 1.0;
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
  int level = 3;             // CPU Zstd level
  bool level_user_set = false;
  bool fast_flag = false;
  bool best_flag = false;
  bool ultra = false;
  bool keep = true;
  bool remove_input = false; // --rm: delete input after success
  bool force = false;
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
  int cpu_threads = 0;       // 0=auto (capped at 96), -1=all threads (-T0)
  double cpu_share = -1;     // <0 adaptive (hybrid)
  size_t chunk_mib = DEFAULT_CHUNK_MIB;
  bool chunk_user_set = false;
  size_t cpu_backlog = 0;    // queue depth before CPU pops (hybrid)
  size_t cpu_batch = 1;      // frames per CPU worker pop (default 1, higher reduces queue contention)
#ifdef HAVE_NVCOMP
  size_t gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  bool gpu_batch_user_set = false;
  double gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  size_t gpu_streams = DEFAULT_GPU_STREAMS;
  int gpu_devices = 0;            // 0=auto (all for compress, 1 for decompress)
  PinMode pin_mode = PinMode::AUTO;
#endif
  std::string stats_json;
  int sparse_mode = -1;           // -1=auto (file:on, stdout:off), 0=off, 1=on
  std::string input;              // current file being processed
  std::vector<std::string> inputs; // all positional args (multi-file)
  std::string output;             // explicit -o; empty = auto-derive per file
};

/*======================================================================
 Verbosity levels & exit codes
======================================================================*/
static constexpr int V_SILENT  = 0;  // -qq: suppress everything including errors
static constexpr int V_ERROR   = 1;  // -q:  errors only
static constexpr int V_DEFAULT = 2;  // normal: progress bar, completion summary
static constexpr int V_VERBOSE = 3;  // -v:  informational messages
static constexpr int V_DEBUG   = 4;  // -vv: per-worker/batch detail
static constexpr int V_TRACE   = 5;  // -vvv: per-chunk debug trace

static constexpr int EXIT_OK       = 0;  // success
static constexpr int EXIT_ERROR    = 1;  // runtime / I/O / compression error
static constexpr int EXIT_USAGE    = 2;  // bad command-line usage

// Global verbosity for die() which doesn't take Options
static int g_verbosity = V_DEFAULT;

// Emit a message to stderr if the current verbosity is >= min_level.
// Caller must include \n or \r in msg as appropriate.
static void vlog(int min_level, const Options & opt, const std::string & msg)
{
  if (opt.verbosity >= min_level)
    std::cerr << msg;
}

static void die(const std::string & msg, int code = EXIT_ERROR)
{
  if (g_verbosity >= V_ERROR)
    std::cerr << "gzstd: " << msg << "\n";
  std::exit(code);
}
static void die_usage(const std::string & msg)
{ die(msg, EXIT_USAGE); }

static void print_help()
{
  std::cout <<
"gzstd " << GZSTD_VERSION << " - Hybrid CPU+GPU Zstd compression\n"
"\n"
"Usage: gzstd [options] [file ...]\n"
"\n"
"If no file is given (or file is '-'), reads from stdin.\n"
"When reading from stdin, output goes to stdout (implies -c).\n"
"Compatible with pipes:  tar cf - dir | gzstd > archive.tar.zst\n"
"                        gzstd -d < archive.tar.zst | tar xf -\n"
"\n"
"Options:\n"
" -d Decompress\n"
" -t Test compressed file (verify integrity)\n"
" -k Keep input file after operation (default)\n"
" --rm Remove input file after successful operation\n"
" -f Force overwrite of output file\n"
" --[no-]sparse Enable/disable sparse file support (skip zero blocks)\n"
"               Default: enabled for file output, disabled for stdout\n"
" -c Write to stdout\n"
" -o, --output FILE  Explicit output path\n"
" -1 .. -19 Compression level (CPU zstd; default: 3)\n"
" -20 .. -22 Stronger levels (require --ultra; use more memory)\n"
" --fast Alias for a fast CPU level (maps to -1)\n"
" --ultra Enable ultra levels (-20..-22); higher memory usage\n"
" --best Alias for a strong CPU level (maps to -19)\n"
" -v / -vv / -vvv Verbose / more verbose / debug-trace\n"
" -q, --quiet Errors only (suppress progress and info)\n"
" -qq, --silent Suppress ALL output including errors\n"
" --progress Force progress meter (even in pipes)\n"
" --no-progress Suppress progress meter\n"
" --chunk-size N Host I/O chunk size in MiB (default: auto; was 32)\n"
" --stats-json <f> Write run statistics (JSON)\n"
" --cpu-only Force CPU-only path (multithreaded, no GPU)\n"
" --hybrid Enable hybrid CPU+GPU scheduling (default with GPU)\n"
" -T, --threads N  CPU worker threads (0=all cores, auto=96 max). -T N or -T# [CPU-only/hybrid]\n"
" --cpu-batch N    Frames per CPU worker pop (default: 1, higher reduces queue contention)\n"
" --cpu-share X Fixed CPU share [0..1], disables adaptation (hybrid)\n"
" --cpu-backlog N Min queue depth before CPU pops (hybrid; 0=off)\n";
#ifdef HAVE_NVCOMP
  std::cout <<
" --gpu-batch N Max GPU subchunks per device (default: 16)\n"
" --gpu-mem-frac X Fraction of free VRAM per device (0.1..0.95, def: 0.60)\n"
" --gpu-streams N CUDA streams per device (default: 3)\n"
" --gpu-devices N Number of GPUs to use (0=auto: all for compress, 1 for decompress)\n"
" --gpu-only GPU only, no CPU workers (error if GPU unavailable)\n"
" --pinned {auto|on|off} Control pinned host buffers (default: auto)\n"
" --no-pinned Alias for --pinned=off\n";
#endif
  std::cout <<
" -h, --help Show this help\n"
" -V, --version Version info\n"
"\n"
"Exit codes:\n"
"  0  Success\n"
"  1  Runtime error (I/O, compression, GPU failure)\n"
"  2  Bad command-line usage\n"
"\n"
"Progress is shown when stderr is a TTY. Use --progress to force it in pipes.\n";
}

static void print_version()
{
#ifdef HAVE_NVCOMP
  std::cout << "gzstd " << GZSTD_VERSION << " (CPU + nvCOMP) MT-CPU + Hybrid scheduling\n";
#else
  std::cout << "gzstd " << GZSTD_VERSION << " (CPU-only) MT compression\n";
#endif
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
    if (v.empty()) die("missing value for " + pref);
    out = std::stoull(v);
    if (was_set) *was_set = true;
    return true;
  }
  // Form: --name VALUE (next argv element)
  else if (a == pref) {
    if (i + 1 >= argc) die("missing value for " + pref);
    out = std::stoull(argv[++i]);
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
    if (v.empty()) die("missing value for " + pref);
    out = std::stoi(v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die("missing value for " + pref);
    out = std::stoi(argv[++i]);
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
    if (v.empty()) die("missing value for " + pref);
    out = v;
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die("missing value for " + pref);
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
    if (v.empty()) die("missing value for " + pref);
    out = std::stod(v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die("missing value for " + pref);
    out = std::stod(argv[++i]);
    return true;
  }

  return false;
}

/*======================================================================
 Progress & metrics
======================================================================*/
struct Meter {
  std::atomic< uint64_t > read_bytes { 0 };
  std::atomic< uint64_t > wrote_bytes{ 0 };
  std::atomic< uint64_t > tasks_done { 0 };
  std::chrono::steady_clock::time_point t0 { std::chrono::steady_clock::now() };
};
static bool is_stderr_tty() { return isatty(fileno(stderr)) != 0; }
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
static void progress_emit_line(double pct, const char * in_s, const char * out_s, const char * rate_s)
{
  char line[256];
  int n = 0;
  if (pct >= 0.0) { n = std::snprintf(line, sizeof(line), "[%.1f%%] in:%s out:%s in_rate:%s/s ", pct, in_s, out_s, rate_s); }
  else { n = std::snprintf(line, sizeof(line), " in:%s out:%s in_rate:%s/s ", in_s, out_s, rate_s); }
  if (n < 0) { n = 0; }
  size_t len = (size_t)std::min(n, (int)sizeof(line) - 1);
  std::fprintf(stderr, "\r%.*s\033[K", (int)len, line);
  std::fflush(stderr);
}

static void progress_loop(const Options & opt, const Meter * m, uint64_t total_in, std::atomic< bool > * done_flag)
{
  // Progress bar at V_DEFAULT and V_VERBOSE only.
  // At V_DEBUG+, per-thread summaries provide progress; the bar would collide.
  // At V_DEFAULT, suppress if stderr is not a TTY OR if stdin is a pipe.
  // --progress (force_progress) overrides TTY/pipe checks but not V_DEBUG+.
  if (opt.verbosity < V_DEFAULT || opt.verbosity >= V_DEBUG) return;
  if (opt.verbosity == V_DEFAULT && !opt.force_progress) {
    if (!is_stderr_tty()) return;
    if (opt.input == "-" && !isatty(fileno(stdin))) return;
  }
  using namespace std::chrono; using namespace std::chrono_literals;
  while (!done_flag->load()) {
    std::this_thread::sleep_for(200ms);
    uint64_t in  = m->read_bytes.load();
    uint64_t out = m->wrote_bytes.load();
    auto dt      = steady_clock::now() - m->t0;
    double secs  = duration_cast< duration<double> >(dt).count();
    double in_rate = secs>0 ? double(in)/secs : 0.0;
    char in_s[64], out_s[64], rate_s[64];
    human_bytes(double(in),  in_s,  sizeof(in_s));
    human_bytes(double(out), out_s, sizeof(out_s));
    human_bytes(in_rate,     rate_s, sizeof(rate_s));
    double pct = (total_in > 0) ? (100.0 * double(in) / double(total_in)) : -1.0;
    progress_emit_line(pct, in_s, out_s, rate_s);
  }
  // Final sample (no newline  the completion summary will overwrite this line)
  uint64_t in  = m->read_bytes.load();
  uint64_t out = m->wrote_bytes.load();
  auto dt      = std::chrono::steady_clock::now() - m->t0;
  double secs  = std::chrono::duration_cast< std::chrono::duration<double> >(dt).count();
  double in_rate = secs>0 ? double(in)/secs : 0.0;
  char in_s[64], out_s[64], rate_s[64];
  human_bytes(double(in),  in_s,  sizeof(in_s));
  human_bytes(double(out), out_s, sizeof(out_s));
  human_bytes(in_rate,     rate_s, sizeof(rate_s));
  double pct = (total_in > 0) ? (100.0 * double(in) / double(total_in)) : -1.0;
  progress_emit_line(pct, in_s, out_s, rate_s);
  // No \n here  the completion summary overwrites this line with \r
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
  if (!f) die("cannot open input: " + path);
  return f;
}
// Open a temporary file for atomic write: write to .tmp, then rename on success.
static FILE * open_output_atomic(const std::string & out, std::string & tmp_path)
{
  tmp_path = out + ".gzstd.tmp";
  register_tmp_file(tmp_path);
  FILE * f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) die("cannot open temp output: " + tmp_path);
  return f;
}
// Flush file data to disk (POSIX only).
static void fsync_file(FILE * f)
{
#if defined(_POSIX_VERSION)
  int fd = fileno(f);
  fsync(fd);
#else
  (void)f;
#endif
}
// Check whether a FILE* refers to a regular file (not a pipe/socket/etc.)
static bool is_regular_file_stream(FILE * f)
{
  if (!f) return false;
  int fd = fileno(f);
  struct stat st;
  if (fstat(fd, &st) != 0) return false;
  return S_ISREG(st.st_mode);
}

// Robust fwrite that handles EINTR and short writes (pipes, signals)
static size_t robust_fwrite(const void * ptr, size_t size, FILE * f)
{
  const char * p = static_cast<const char *>(ptr);
  size_t remaining = size;
  while (remaining > 0) {
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
#ifndef _WIN32
class DirectWriter {
public:
  static constexpr size_t ALIGN = 4096;          // filesystem block alignment
  static constexpr size_t FLUSH_SIZE = 4 * 1024 * 1024;  // flush every 4 MiB

  DirectWriter() = default;
  ~DirectWriter() { close(); }

  // Open file with O_DIRECT.  Returns false if O_DIRECT not supported
  // (caller should fall back to FILE*).
  bool open(const std::string & path) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0666);
    if (fd_ < 0) return false;
    // Allocate aligned buffer
    if (posix_memalign(&buf_, ALIGN, BUF_CAP) != 0) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    buf_used_ = 0;
    total_written_ = 0;
    return true;
  }

  // Write data  accumulates in aligned buffer, flushes in aligned chunks
  bool write(const void * data, size_t len) {
    if (fd_ < 0) return false;
    const char * src = static_cast<const char *>(data);
    while (len > 0) {
      size_t space = BUF_CAP - buf_used_;
      size_t copy = std::min(len, space);
      std::memcpy(static_cast<char*>(buf_) + buf_used_, src, copy);
      buf_used_ += copy;
      src += copy;
      len -= copy;
      if (buf_used_ >= FLUSH_SIZE) {
        if (!flush_aligned()) return false;
      }
    }
    return true;
  }

  // Finalize  flush remaining data (may need non-aligned final write)
  bool finalize() {
    if (fd_ < 0) return false;
    if (buf_used_ == 0) return true;

    size_t aligned = (buf_used_ / ALIGN) * ALIGN;
    size_t tail = buf_used_ - aligned;

    // Write aligned portion with O_DIRECT
    if (aligned > 0) {
      if (!write_all(buf_, aligned)) return false;
      total_written_ += aligned;
    }

    // Write remaining tail without O_DIRECT
    if (tail > 0) {
      // Remove O_DIRECT flag for the final unaligned write
      int flags = fcntl(fd_, F_GETFL);
      if (flags >= 0) fcntl(fd_, F_SETFL, flags & ~O_DIRECT);
      const char * tail_ptr = static_cast<char*>(buf_) + aligned;
      if (!write_all(tail_ptr, tail)) return false;
      total_written_ += tail;
    }

    buf_used_ = 0;
    return true;
  }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (buf_) {
      free(buf_);
      buf_ = nullptr;
    }
  }

  size_t total_bytes() const { return total_written_ + buf_used_; }
  int fd() const { return fd_; }

  // Seek forward by 'offset' bytes  for sparse file support.
  // Flushes the internal buffer first to keep fd position and buffer in sync,
  // then seeks the fd forward.  After this call, the next write() will
  // resume at the new position.
  bool seek_forward(size_t offset) {
    if (fd_ < 0) return false;
    // Flush any buffered data first
    if (buf_used_ > 0) {
      if (!flush_aligned()) return false;
      // flush_aligned may leave a small unaligned tail in the buffer.
      // For sparse seek we need the buffer fully empty so the fd position
      // is authoritative.  Write any remaining tail without O_DIRECT.
      if (buf_used_ > 0) {
        int flags = fcntl(fd_, F_GETFL);
        if (flags >= 0) fcntl(fd_, F_SETFL, flags & ~O_DIRECT);
        if (!write_all(buf_, buf_used_)) return false;
        if (flags >= 0) fcntl(fd_, F_SETFL, flags);  // restore O_DIRECT
        total_written_ += buf_used_;
        buf_used_ = 0;
      }
    }
    // Now seek the fd forward
    if (lseek(fd_, (off_t)offset, SEEK_CUR) < 0) return false;
    total_written_ += offset;
    return true;
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

  bool flush_aligned() {
    // Flush as much as possible in aligned chunks
    size_t aligned = (buf_used_ / ALIGN) * ALIGN;
    if (aligned == 0) return true;
    if (!write_all(buf_, aligned)) return false;
    total_written_ += aligned;
    // Move leftover tail to front
    size_t tail = buf_used_ - aligned;
    if (tail > 0)
      std::memmove(buf_, static_cast<char*>(buf_) + aligned, tail);
    buf_used_ = tail;
    return true;
  }

  bool write_all(const void * data, size_t len) {
    const char * p = static_cast<const char *>(data);
    while (len > 0) {
      ssize_t w = ::write(fd_, p, len);
      if (w > 0) {
        p += w;
        len -= (size_t)w;
      } else if (w < 0) {
        if (errno == EINTR) continue;
        return false;
      }
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
      }
    }
    return true;
  }

  int fd_ = -1;
  void * buf_ = nullptr;
  size_t buf_used_ = 0;
  size_t total_written_ = 0;
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
  if (opt_threads == -1)
    return (int)hw;                         // -T0: use every thread
  if (opt_threads > 0)
    return opt_threads;                     // explicit -T N
  // Auto: use hw-1 (leave one for I/O), capped at 96
  int def = (hw > 1) ? (int)(hw - 1) : 1;
  return std::min(def, 96);
}

static size_t auto_chunk_mib_cpu(FILE * in, const Options & opt)
{
  (void)opt;
  if (!is_regular_file_stream(in))
    return 32;  // pipes: smaller chunks for lower latency

  // Scale chunk size with file size for better CPU efficiency.
  // Larger chunks = fewer tasks = less scheduling overhead.
  // But not too large or single-thread latency suffers.
  uint64_t fsize = 0;
  struct stat st;
  if (fstat(fileno(in), &st) == 0 && S_ISREG(st.st_mode))
    fsize = (uint64_t)st.st_size;

  if (fsize > 100ULL * 1024 * 1024 * 1024)  return 128;  // > 100 GiB
  if (fsize > 10ULL * 1024 * 1024 * 1024)   return 96;   // > 10 GiB
  if (fsize > 1ULL * 1024 * 1024 * 1024)    return 64;   // > 1 GiB
  return 32;                                               // <= 1 GiB
}
#ifdef HAVE_NVCOMP
#if __cplusplus >= 201703L
[[maybe_unused]]
#endif
static size_t auto_chunk_mib_gpu(FILE * in, const Options & opt, int device_count)
{
  (void)in;
  size_t base = 16;
  size_t target = std::max<size_t>(1, (size_t)device_count) * std::max<size_t>(1, opt.gpu_batch_cap) * base;
  size_t chosen = std::max<size_t>(AUTO_HOST_CHUNK_MIN_MIB, std::min<size_t>(target, AUTO_HOST_CHUNK_MAX_MIB));
  return chosen;
}
#endif

/*======================================================================
 Queues and writer
======================================================================*/
// A chunk of work: compressed or uncompressed data with a sequence number.
// For decompression, decomp_size holds the expected decompressed size (from frame header).
struct Task { size_t seq; std::vector<char> data; size_t decomp_size = 0; };

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

    fprintf(stderr, "\n\n");
    fprintf(stderr, "  PERFORMANCE BREAKDOWN: %-36s \n", label);
    fprintf(stderr, "\n");

    if (cuda_init_count > 0)
      fprintf(stderr, "  CUDA init:        %8.3f s  (%d devices, %.3f s each avg)    \n",
              ns_to_s(cuda_init_max_ns), cuda_init_count.load(),
              ns_to_s(cuda_init_sum_ns) / cuda_init_count);

    fprintf(stderr, "                                                              \n");
    fprintf(stderr, "  Reader:           %8.3f s  (%6.2f GiB, %5.2f GiB/s)      \n",
            ns_to_s(read_ns), bytes_to_gib(read_bytes_total),
            rate_gibs(read_bytes_total, read_ns));

    if (h2d_count > 0) {
      fprintf(stderr, "  H2D transfers:    %8.3f s  (%6.2f GiB, %5.2f GiB/s) [%llu] \n",
              ns_to_s(h2d_ns), bytes_to_gib(h2d_bytes),
              rate_gibs(h2d_bytes, h2d_ns),
              (unsigned long long)h2d_count.load());
    }
    if (kernel_count > 0) {
      fprintf(stderr, "  GPU kernel:       %8.3f s  (%llu batches, %5.1f ms/batch)   \n",
              ns_to_s(kernel_ns),
              (unsigned long long)kernel_count.load(),
              (kernel_count > 0) ? ns_to_ms(kernel_ns) / kernel_count : 0.0);
    }
    if (d2h_count > 0) {
      fprintf(stderr, "  D2H transfers:    %8.3f s  (%6.2f GiB, %5.2f GiB/s) [%llu] \n",
              ns_to_s(d2h_ns), bytes_to_gib(d2h_bytes),
              rate_gibs(d2h_bytes, d2h_ns),
              (unsigned long long)d2h_count.load());
    }
    if (gpu_batch_count > 0) {
      fprintf(stderr, "  GPU batch total:  %8.3f s  (%llu batches, %5.1f ms/batch)   \n",
              ns_to_s(gpu_batch_ns),
              (unsigned long long)gpu_batch_count.load(),
              (gpu_batch_count > 0) ? ns_to_ms(gpu_batch_ns) / gpu_batch_count : 0.0);
    }
    if (cpu_compute_count > 0) {
      fprintf(stderr, "  CPU compute:      %8.3f s  (%6.2f GiB, %llu chunks)       \n",
              ns_to_s(cpu_compute_ns), bytes_to_gib(cpu_compute_bytes),
              (unsigned long long)cpu_compute_count.load());
    }

    fprintf(stderr, "                                                              \n");
    fprintf(stderr, "  Queue wait:       %8.3f s  (%llu waits)                    \n",
            ns_to_s(queue_wait_ns), (unsigned long long)queue_wait_count.load());
    fprintf(stderr, "  Writer wait:      %8.3f s  (%llu waits, %llu out-of-order)  \n",
            ns_to_s(writer_wait_ns), (unsigned long long)writer_wait_count.load(),
            (unsigned long long)out_of_order_waits.load());
    fprintf(stderr, "  Writer I/O:       %8.3f s  (%6.2f GiB, %5.2f GiB/s)      \n",
            ns_to_s(write_ns), bytes_to_gib(write_bytes_total),
            rate_gibs(write_bytes_total, write_ns));
    if (result_lock_ns > 0)
      fprintf(stderr, "  Result lock:      %8.3f s                                \n",
              ns_to_s(result_lock_ns));

    fprintf(stderr, "                                                              \n");
    fprintf(stderr, "  Scheduler:  CPU %llu tasks, GPU %llu tasks                    \n",
            (unsigned long long)sched_cpu_tasks.load(),
            (unsigned long long)sched_gpu_tasks.load());

    fprintf(stderr, "\n\n");
  }
};

// Global perf counters  only non-null when -vvv is active.
// All instrumentation checks `if (g_perf)` before recording.
static PerfCounters * g_perf = nullptr;

// Global DirectWriter pointer  set when O_DIRECT output is active.
// Writer thread checks this to decide between DirectWriter and fwrite.
#ifndef _WIN32
static DirectWriter * g_direct_writer = nullptr;
#endif

class TaskQueue {
public:
  void push(Task && t)
  {
    std::unique_lock<std::mutex> lk(m_);
    q_.push_back(std::move(t));
    ++total_tasks_;
    cv_.notify_one();
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
      out.push_back(std::move(q_.front()));
      q_.pop_front();
    }
    return true;
  }

  // Greedy pop: wait until the queue has max_n items OR producer is done,
  // then take everything available up to max_n.  This maximizes batch size
  // for GPU kernels where per-launch overhead is expensive.
  bool pop_batch_greedy(size_t max_n, std::vector<Task> & out)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.size() < max_n && !done_)
      cv_.wait(lk);
    if (q_.empty() && done_) return false;

    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i = 0; i < take; ++i) {
      out.push_back(std::move(q_.front()));
      q_.pop_front();
    }
    return true;
  }

  // Pop a single task (used by CPU workers).
  bool pop_one(Task & t)
  {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cv_.wait(lk);
    if (q_.empty() && done_) return false;

    t = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  // Signal that no more tasks will be pushed (producer is finished).
  void set_done()
  {
    std::unique_lock<std::mutex> lk(m_);
    done_ = true;
    cv_.notify_all();
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

private:
  std::mutex              m_;
  std::condition_variable cv_;
  std::deque<Task>        q_;
  bool                    done_ = false;
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
struct ResultStore {
  std::mutex                                     m;
  std::condition_variable                        cv;
  std::unordered_map<size_t, std::vector<char>>  data;           // seq -> compressed frame
  size_t                                         next_to_write = 0;
  size_t                                         total_tasks   = 0;
  bool                                           producer_done = false;
  bool                                           workers_done  = false;
};

// Writer thread: waits for compressed frames to appear in sequence order,
// then writes them to the output file.  This keeps the output deterministic
// (frames in the same order as the input) regardless of which worker finishes first.
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
  explicit AsyncWritePool(FILE * out_file, DirectWriter * dw, bool sparse = true)
    : out_(out_file), dw_(dw), sparse_(sparse), done_(false)
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
  void submit(std::vector<std::vector<char>> && buffers) {
    std::unique_lock<std::mutex> lk(m_);
    // Wait for previous write to finish (backpressure)
    cv_consumer_.wait(lk, [this]{ return pending_.empty() || done_; });
    if (done_) return;
    pending_ = std::move(buffers);
    lk.unlock();
    cv_producer_.notify_one();
  }

  // Wait for all queued writes to complete.
  void flush() {
    std::unique_lock<std::mutex> lk(m_);
    cv_consumer_.wait(lk, [this]{ return pending_.empty(); });
  }

  bool had_error() const { return error_; }

private:
  // Check if a region is all zeros (for sparse file support).
  // Uses size_t-wide comparisons for speed.
  static bool is_all_zero(const char * p, size_t len) {
    // Check in size_t chunks for speed
    const size_t * wp = reinterpret_cast<const size_t *>(p);
    size_t words = len / sizeof(size_t);
    for (size_t i = 0; i < words; ++i)
      if (wp[i] != 0) return false;
    // Check remaining bytes
    for (size_t i = words * sizeof(size_t); i < len; ++i)
      if (p[i] != 0) return false;
    return true;
  }

  // Write a buffer with sparse file optimization.
  // Scans for zero-filled blocks and seeks past them instead of writing.
  // The OS creates a sparse file  zero regions don't consume disk space
  // and aren't physically written, which can dramatically speed up output
  // for data with large zero runs (e.g., disk images, memory dumps).
  bool write_sparse(const char * data, size_t len, bool is_last_buffer = false) {
    static constexpr size_t SPARSE_BLOCK = 4096;  // check in page-sized blocks

    size_t pos = 0;
    while (pos < len) {
      size_t remain = len - pos;
      size_t block = std::min(remain, SPARSE_BLOCK);

      // Never sparse-skip the final block of the last buffer in a batch 
      // the file must end with a physical write or it will be truncated.
      bool at_end = is_last_buffer && (pos + block >= len);

      if (sparse_ && !at_end && block == SPARSE_BLOCK && is_all_zero(data + pos, block)) {
        // Skip this zero block via seek
        if (dw_) {
          if (!dw_->seek_forward(block)) return false;
        } else if (out_) {
          if (std::fseek(out_, (long)block, SEEK_CUR) != 0)
            return false;
        }
        pos += block;
        sparse_saved_ += block;
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
          if (!dw_->write(data + pos, to_write)) return false;
        } else if (out_) {
          size_t w = robust_fwrite(data + pos, to_write, out_);
          if (w != to_write) return false;
        }
        pos = write_end;
      }
    }
    return true;
  }

  void worker_fn() {
    while (true) {
      std::vector<std::vector<char>> work;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_producer_.wait(lk, [this]{ return !pending_.empty() || done_; });
        if (done_ && pending_.empty()) return;
        work = std::move(pending_);
      }
      cv_consumer_.notify_one();  // signal that pending_ is now empty

      // Write all buffers (last buffer's final block always written to ensure file length)
      for (size_t bi = 0; bi < work.size(); ++bi) {
        auto & buf = work[bi];
        uint64_t w_t0 = g_perf ? now_ns() : 0;
        bool is_last = (bi == work.size() - 1);
        if (!write_sparse(buf.data(), buf.size(), is_last)) {
          error_ = true;
          return;
        }
        if (g_perf) {
          g_perf->write_ns.fetch_add(now_ns() - w_t0);
          g_perf->write_bytes_total.fetch_add(buf.size());
        }
      }
    }
  }

  FILE * out_;
  DirectWriter * dw_;
  bool sparse_;
  std::mutex m_;
  std::condition_variable cv_producer_;
  std::condition_variable cv_consumer_;
  std::vector<std::vector<char>> pending_;
  std::thread worker_;
  bool done_;
  bool error_ = false;
  uint64_t sparse_saved_ = 0;  // bytes skipped via sparse seek
};

static void writer_thread(FILE * out, ResultStore & results,
                          const Options & opt, Meter * m)
{
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  const bool skip_write = (opt.mode == Mode::TEST);

  // Create async write pool for double-buffered I/O
  AsyncWritePool * aio = nullptr;
  std::unique_ptr<AsyncWritePool> aio_ptr;
  if (!skip_write) {
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
#ifndef _WIN32
    aio_ptr = std::make_unique<AsyncWritePool>(out, g_direct_writer, enable_sparse);
#else
    aio_ptr = std::make_unique<AsyncWritePool>(out, nullptr, enable_sparse);
#endif
    aio = aio_ptr.get();
  }

  std::unique_lock<std::mutex> lk(results.m);

  while (true) {
    bool all_done = results.producer_done
                 && results.workers_done
                 && results.next_to_write >= results.total_tasks;

    uint64_t wait_t0 = g_perf ? now_ns() : 0;
    bool waited = false;

    // Wait for the next sequential frame
    while (results.data.count(results.next_to_write) == 0 && !all_done) {
      waited = true;
      results.cv.wait(lk);
      all_done = results.producer_done
              && results.workers_done
              && results.next_to_write >= results.total_tasks;
    }

    if (g_perf && waited) {
      g_perf->writer_wait_ns.fetch_add(now_ns() - wait_t0);
      g_perf->writer_wait_count.fetch_add(1);
      g_perf->out_of_order_waits.fetch_add(1);
    }
    if (all_done) break;

    // Batch drain: collect ALL consecutive ready frames
    std::vector<std::vector<char>> batch;
    size_t batch_bytes = 0;
    while (true) {
      auto it = results.data.find(results.next_to_write);
      if (it == results.data.end()) break;
      batch_bytes += it->second.size();
      batch.push_back(std::move(it->second));
      results.data.erase(it);
      ++results.next_to_write;
    }

    if (batch.empty()) continue;
    lk.unlock();

    // Submit to async write pool (non-blocking unless previous write still in flight)
    if (aio) {
      if (opt.verbosity >= V_DEBUG) {
        char bs[32];
        human_bytes(double(batch_bytes), bs, sizeof(bs));
        vlog(V_DEBUG, opt, std::string("[WRITER] submitting ") + std::to_string(batch.size())
             + " frames (" + bs + ")\n");
      }
      aio->submit(std::move(batch));
      if (aio->had_error()) die("async write failed (disk full?)");
    }
    if (m) m->wrote_bytes.fetch_add(batch_bytes);

    lk.lock();
  }

  // Flush remaining async writes
  if (aio) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "writer: flushing remaining output to disk...\n");
    aio->flush();
    if (aio->had_error()) die("async write failed (disk full?)");
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "writer: flush complete\n");
  }
}

/*======================================================================
 CPU compression helpers / workers
======================================================================*/
// Thread-local CCtx avoids repeated allocation for per-chunk compression
static thread_local ZSTD_CCtx * tl_cctx = nullptr;

static inline void compress_one_cpu_frame(const void * src, size_t src_size, int level, std::vector< char > & out)
{
  if (!tl_cctx) {
    tl_cctx = ZSTD_createCCtx();
    if (!tl_cctx) die("failed to create ZSTD_CCtx");
  }
  // Set compression level explicitly on every call (level may differ per invocation)
  size_t st = ZSTD_CCtx_setParameter(tl_cctx, ZSTD_c_compressionLevel, level);
  if (ZSTD_isError(st)) die(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));

  size_t bound = ZSTD_compressBound(src_size);
  out.resize(bound);
  size_t csz = ZSTD_compress2(tl_cctx, out.data(), out.size(), src, src_size);
  if (ZSTD_isError(csz)) die(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
  out.resize(csz);
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

  ZSTD_inBuffer zin { input.data(), input.size(), 0 };
  ZSTD_outBuffer zout { outbuf.data(), outbuf.size(), 0 };
  if (m) m->read_bytes.fetch_add(input.size());

  while (zin.pos < zin.size) {
    size_t ret = ZSTD_decompressStream(dctx, &zout, &zin);
    if (ZSTD_isError(ret))
      die(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret));
    if (zout.pos > 0) {
      if (opt.mode != Mode::TEST) {
#ifndef _WIN32
        if (g_direct_writer) {
          if (!g_direct_writer->write(outbuf.data(), zout.pos))
            die("direct write failed (disk full?)");
        } else
#endif
        {
          size_t w = robust_fwrite(outbuf.data(), zout.pos, out);
          if (w != zout.pos) die("short write to output (broken pipe?)");
        }
      }
      if (m) m->wrote_bytes.fetch_add(zout.pos);
      zout.pos = 0;
    }
    if (ret == 0 && zin.pos < zin.size) {
      // Frame boundary in multi-frame stream; continue
    }
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
 Hybrid scheduler  single definition placed before workers
======================================================================*/
class HybridSched {
public:
  HybridSched(double override_share, int /*cpu_threads*/, int gpu_devices,
              const Options & opt)
    : opt_(opt), gpu_device_count_(gpu_devices)
  {
    if (override_share >= 0.0) {
      fixed_mode_ = true;
      fixed_cpu_share_ = override_share;
    }
  }

  // CPU checks this before taking from the queue.
  // If any GPU stream is waiting for data, CPU yields.
  bool should_cpu_take(size_t /*queue_depth*/ = 0) const {
    if (fixed_mode_) {
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

    // No GPU is waiting  CPU can take
    return true;
  }

  // GPU stream calls this when it's ready and waiting for data
  void gpu_wants_data() { gpus_waiting_.fetch_add(1, std::memory_order_release); }

  // GPU stream calls this after it has taken a batch from the queue
  void gpu_got_data()   { gpus_waiting_.fetch_sub(1, std::memory_order_release); }

  // Called by GPU workers once CUDA context is initialized
  void set_gpu_ready() {
    gpu_ready_.store(true, std::memory_order_release);
    if (opt_.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt_, "GPU ready  semaphore scheduling active\n");
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

    if (opt_.verbosity >= V_DEBUG) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(3)
         << "hybrid: tick cpu_rate=" << (cpu_rate/1e9) << " GiB/s"
         << " gpu_rate=" << (gpu_rate/1e9) << " GiB/s"
         << " gpus_waiting=" << gpus_waiting_.load()
         << " cpu_taken=" << cpu_taken_.load()
         << " gpu_taken=" << gpu_taken_.load();
      std::cerr << os.str() << "\n";
    }

    cpu_taken_.store(0, std::memory_order_relaxed);
    gpu_taken_.store(0, std::memory_order_relaxed);
  }

  double target_share() const { return 0.5; }
  bool is_gpu_ready() const { return gpu_ready_.load(std::memory_order_acquire); }

private:
  const Options & opt_;
  int gpu_device_count_ = 0;
  bool fixed_mode_ = false;
  double fixed_cpu_share_ = -1.0;

  std::atomic<bool> gpu_ready_{false};
  std::atomic<int> gpus_waiting_{0};

  std::atomic<uint64_t> cpu_taken_{0};
  std::atomic<uint64_t> gpu_taken_{0};
  std::atomic<uint64_t> cpu_bytes_{0};
  std::atomic<uint64_t> gpu_bytes_{0};
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
static void cpu_worker(
  int worker_id,
  TaskQueue * tq,
  ResultStore * results,
  const Options * opt,
  Meter * m,
#ifdef HAVE_NVCOMP
  void * sched_ptr,
#endif
  CpuAgg * cpuagg)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  if (opt->verbosity >= V_TRACE) {
    std::ostringstream os;
    os << "[CPU/T" << worker_id << "] online";
    vlog(V_TRACE, *opt, os.str() + "\n");
  }

  while (true) {
#ifdef HAVE_NVCOMP
    if (sched) {
      if (!sched->should_cpu_take()) {
        if (tq->drained()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    }
#endif
    if (opt->cpu_backlog > 0) {
      size_t qsz = tq->size();
      if (qsz < opt->cpu_backlog) {
        if (tq->drained()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    }

    Task t;
    if (!tq->pop_one(t)) break;
#ifdef HAVE_NVCOMP
    if (sched) sched->mark_cpu_take(1);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> out_frame;
    compress_one_cpu_frame(t.data.data(), t.data.size(), opt->level, out_frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast< std::chrono::duration<double, std::milli> >(t1 - t0).count();
    const size_t csz = out_frame.size();
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(uint64_t(ms * 1e6));
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(t.data.size());
      g_perf->sched_cpu_tasks.fetch_add(1);
    }

    if (opt->verbosity >= V_DEBUG) {
      char in_s[32], out_s[32];
      human_bytes(double(t.data.size()), in_s, sizeof(in_s));
      human_bytes(double(csz), out_s, sizeof(out_s));
      const double thr_gib = (ms > 0.0) ? (double)t.data.size() / (ms/1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    if (m) m->tasks_done.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(results->m);
      results->data.emplace(t.seq, std::move(out_frame));
    }
    results->cv.notify_one();
#ifdef HAVE_NVCOMP
    if (sched) sched->add_cpu_bytes(t.data.size());
#endif
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if (cpuagg->per_thread.size() <= (size_t)worker_id)
        cpuagg->per_thread.resize((size_t)worker_id + 1);
      auto & st = cpuagg->per_thread[(size_t)worker_id];
      st.tasks    += 1;
      st.in_bytes += t.data.size();
      st.out_bytes += csz;
      st.comp_ms  += ms;
      cpuagg->tasks    += 1;
      cpuagg->in_bytes += t.data.size();
      cpuagg->out_bytes += csz;
      cpuagg->comp_ms  += ms;
    }
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
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] total tasks=" << st.tasks
         << " in=" << st.in_bytes << "B out=" << st.out_bytes << "B"
         << " time=" << std::fixed << std::setprecision(2) << st.comp_ms << "ms"
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }
  }
}

static void cpu_worker_rescue(
  int worker_id,
  RescueQueue * rq,
  ResultStore * results,
  const Options * opt,
  Meter * /*m*/,
  CpuAgg * cpuagg)
{
  if (opt->verbosity >= V_TRACE) {
    std::ostringstream os;
    os << "[RESCUE/T" << worker_id << "] online";
    vlog(V_TRACE, *opt, os.str() + "\n");
  }
  while (true) {
    Task t;
    if (!rq->pop_one(t)) break;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> out_frame;
    compress_one_cpu_frame(t.data.data(), t.data.size(), opt->level, out_frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>>(t1 - t0).count();
    const size_t csz = out_frame.size();

    if (opt->verbosity >= V_TRACE) {
      char in_s[32], out_s[32];
      human_bytes(double(t.data.size()), in_s, sizeof(in_s));
      human_bytes(double(csz), out_s, sizeof(out_s));
      double thr_gib = (ms > 0.0) ? (double)t.data.size() / (ms / 1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[RESCUE/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_TRACE, *opt, os.str() + "\n");
    }

    // Deliver compressed frame to the result store
    {
      std::lock_guard<std::mutex> lk(results->m);
      results->data.emplace(t.seq, std::move(out_frame));
    }
    results->cv.notify_one();

    // Update per-thread stats
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if (cpuagg->per_thread.size() <= (size_t)worker_id)
        cpuagg->per_thread.resize((size_t)worker_id + 1);
      auto & st = cpuagg->per_thread[(size_t)worker_id];
      st.tasks    += 1;
      st.in_bytes += t.data.size();
      st.out_bytes += csz;
      st.comp_ms  += ms;
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
#endif
  CpuAgg * cpuagg)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  // Create a reusable decompression context for this thread
  if (!tl_dctx) {
    tl_dctx = ZSTD_createDCtx();
    if (!tl_dctx) die("failed to create ZSTD_DCtx");
  }

  if (opt->verbosity >= V_TRACE) {
    std::ostringstream os;
    os << "[CPU-D/T" << worker_id << "] online";
    vlog(V_TRACE, *opt, os.str() + "\n");
  }

  while (true) {
#ifdef HAVE_NVCOMP
    // In hybrid mode: CPU runs wild before GPU ready, then yields to GPU
    // unless queue is overflowing or CPU measured faster
    if (sched) {
      if (!sched->should_cpu_take(tq->size())) {
        if (tq->drained()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    }
#endif

    // Grab one or more frames depending on cpu_batch setting
    std::vector<Task> tasks;
    if (opt->cpu_batch > 1) {
      if (!tq->pop_batch(opt->cpu_batch, tasks, opt->cpu_batch)) break;
      if (tasks.empty()) {
        if (tq->drained()) break;
        continue;
      }
#ifdef HAVE_NVCOMP
      if (sched) sched->mark_cpu_take(tasks.size());
#endif
    } else {
      Task t;
      if (!tq->pop_one(t)) break;
#ifdef HAVE_NVCOMP
      if (sched) sched->mark_cpu_take(1);
#endif
      tasks.push_back(std::move(t));
    }

    // Process all frames in this batch
    for (auto & t : tasks) {
    const auto t0_w = std::chrono::steady_clock::now();

    std::vector<char> out_buf(t.decomp_size);
    const size_t comp_size = t.data.size();
    size_t actual = ZSTD_decompressDCtx(tl_dctx, out_buf.data(), out_buf.size(),
                                        t.data.data(), t.data.size());
    if (ZSTD_isError(actual))
      die(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(actual));
    out_buf.resize(actual);

    { std::vector<char>().swap(t.data); }

    const auto t1_w = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>>(t1_w - t0_w).count();
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(uint64_t(ms * 1e6));
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(actual);
      g_perf->sched_cpu_tasks.fetch_add(1);
    }

    if (opt->verbosity >= V_DEBUG) {
      char in_s[32], out_s[32];
      human_bytes(double(comp_size), in_s, sizeof(in_s));
      human_bytes(double(actual), out_s, sizeof(out_s));
      double thr_gib = (ms > 0.0) ? double(actual) / (ms / 1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[CPU-D/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    if (m) m->read_bytes.fetch_add(comp_size);

    {
      std::lock_guard<std::mutex> lk(results->m);
      results->data.emplace(t.seq, std::move(out_buf));
    }
    results->cv.notify_one();

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
    } // end for (auto & t : tasks)
  }
}

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
    Meter * /*m*/,
    const Options & opt,
    bool * fallback,
    std::vector<char> * raw_data)
{
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  *fallback = false;

  // Read buffer with offset-based tracking to avoid per-iteration shifts.
  // We compact (memmove) only when the unconsumed tail is too small to
  // hold a new read chunk, keeping the common path allocation-free.
  const size_t READ_CHUNK = 4 * ONE_MIB;
  std::vector<char> buf(READ_CHUNK * 2);
  size_t buf_len = 0;    // valid data in buf[0..buf_len)
  size_t buf_off = 0;    // parse cursor within buf

  size_t seq = 0;
  bool eof = false;

  while (!eof) {
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

    // Read more data from input
    uint64_t rd_t0 = g_perf ? now_ns() : 0;
    size_t n = std::fread(buf.data() + buf_len, 1, READ_CHUNK, in);
    if (g_perf && n > 0) {
      g_perf->read_ns.fetch_add(now_ns() - rd_t0);
      g_perf->read_bytes_total.fetch_add(n);
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

      // Try to find the compressed frame size
      size_t frame_comp = ZSTD_findFrameCompressedSize(ptr, remaining);
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
        // Parsed some frames OK; trailing bytes cannot form a valid frame
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
            size_t r = std::fread(tail.data(), 1, READ_CHUNK, in);
            if (r == 0) break;
            raw_data->insert(raw_data->end(), tail.data(), tail.data() + r);
          }
        }
        return seq;  // some frames may already be queued
      }

      // Complete frame with known sizes -- push to the queue
      Task t;
      t.seq = seq++;
      t.data.assign(ptr, ptr + frame_comp);
      t.decomp_size = (size_t)frame_decomp;
      queue.push(std::move(t));

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

  // Start writer thread (outputs decompressed frames in original order)
  std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m);

  // Start worker threads (they block on the queue until frames arrive)
  std::vector<std::thread> pool;
  pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) {
    pool.emplace_back(cpu_decomp_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
                      nullptr,
#endif
                      &cpuagg);
  }
  if (opt.verbosity >= V_VERBOSE)
    std::cerr << "[CPU-D] " << threads << " decompression threads online\n";

  // Stream frames from input directly into the queue.
  // Workers start decompressing as soon as the first frame is pushed.
  bool fallback = false;
  std::vector<char> raw_data;
  size_t n_frames = stream_frames_to_queue(in, queue, m, opt, &fallback, &raw_data);

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
    for (auto & th : pool) th.join();
    wthr.join();
    return;
  }

  vlog(V_VERBOSE, opt,
       "streamed " + std::to_string(n_frames) + " frames for MT CPU decompress\n");

  // Signal that all frames have been enqueued
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
  }
  results.cv.notify_all();

  // Wait for workers, then writer
  for (auto & th : pool) th.join();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();
  wthr.join();

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
  // Choose chunk size
  size_t chosen_mib = opt.chunk_mib;
  if (!opt.chunk_user_set) {
    chosen_mib = auto_chunk_mib_cpu(in, opt);
    vlog(V_VERBOSE, opt,
         std::string("auto-chunk (CPU): ") + std::to_string(chosen_mib) + " MiB\n");
  }
  const size_t chunk_bytes = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // Get total input size for progress percentage
  uint64_t total_in = 0;
  if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
    total_in = (uint64_t)fs::file_size(opt.input);

  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);

  ZSTD_CCtx * cctx = ZSTD_createCCtx();
  if (!cctx) die("failed to create ZSTD_CCtx");
  size_t st = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, opt.level);
  if (ZSTD_isError(st)) die(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));

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
    if (ZSTD_isError(csz)) die(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
    if (g_perf) {
      g_perf->cpu_compute_ns.fetch_add(now_ns() - comp_t0);
      g_perf->cpu_compute_count.fetch_add(1);
      g_perf->cpu_compute_bytes.fetch_add(n);
    }
    uint64_t w_t0 = g_perf ? now_ns() : 0;
#ifndef _WIN32
    if (g_direct_writer) {
      if (!g_direct_writer->write(outbuf.data(), csz))
        die("direct write failed (disk full?)");
    } else
#endif
    {
      size_t w = robust_fwrite(outbuf.data(), csz, out);
      if (w != csz) die("short write to output (broken pipe?)");
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

static void compress_cpu_mt(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  // ---- Performance instrumentation (active at -vvv) ----
  PerfCounters perf_local;
  if (opt.verbosity >= V_TRACE) g_perf = &perf_local;

  int threads = resolve_cpu_threads(opt.cpu_threads);

  // Option A: if single-threaded, use simple streaming helper
  if (threads == 1) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "CPU MT requested with 1 thread; using single-thread streaming path\n");
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
  size_t chosen_mib = opt.chunk_mib;
  if (!opt.chunk_user_set) {
    chosen_mib = auto_chunk_mib_cpu(in, opt);
    vlog(V_VERBOSE, opt,
         std::string("auto-chunk (CPU-MT): ") + std::to_string(chosen_mib) + " MiB\n");
  }
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // Get total input size for progress percentage (unknown for pipes)
  uint64_t total_in = 0;
  if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
    total_in = (uint64_t)fs::file_size(opt.input);

  // Start progress bar and ordered-writer threads
  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  TaskQueue queue;
  ResultStore results;
  std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m);

  std::vector<std::thread> pool; pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) pool.emplace_back(cpu_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
    nullptr,
#endif
    &cpuagg);
  if (opt.verbosity >= V_VERBOSE) std::cerr << "[CPU] " << threads << " worker threads online\n";

  // Producer loop: read input in chunks and enqueue for compression
  try_boost_io_priority(true);  // CPU-only: always has worker pool
  std::vector<char> host_in(host_chunk);
  std::atomic<size_t> seq{0};
  while (true) {
    uint64_t rd_t0 = g_perf ? now_ns() : 0;
    size_t n = std::fread(host_in.data(), 1, host_chunk, in);
    if (g_perf && n > 0) {
      g_perf->read_ns.fetch_add(now_ns() - rd_t0);
      g_perf->read_bytes_total.fetch_add(n);
    }
    if (n == 0) break;
    if (m) m->read_bytes.fetch_add(n);

    Task t;
    t.seq = seq.fetch_add(1, std::memory_order_relaxed);
    t.data.assign(host_in.data(), host_in.data() + n);
    queue.push(std::move(t));
  }

  // Signal workers that no more input is coming
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
  }
  results.cv.notify_all();

  // Wait for all workers to finish, then signal the writer
  for (auto & th : pool) th.join();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();

  // Wait for writer and progress threads to finish
  wthr.join();
  progress_done = true;
  progress_thr.join();

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

  // Pinned host memory for faster H2D transfers (optional)
  void * h2d_pinned_base = nullptr;
  void * d2h_pinned_base = nullptr;

  // Host-side vectors (mirroring device arrays for readback)
  std::vector<size_t>         h_in_sizes;
  std::vector<size_t>         h_comp_sizes;
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
  StreamStats stats{};
  std::chrono::steady_clock::time_point last_adjust{ std::chrono::steady_clock::now() };
};

static bool allocate_stream_buffers(StreamCtx & C, size_t per_stream_batch, size_t gpu_chunk, size_t max_out_chunk, nvcompBatchedZstdCompressOpts_t comp_opts, const Options & opt)
{
  C.per_stream_batch = per_stream_batch;
  C.gpu_chunk = gpu_chunk;
  C.max_out_chunk = max_out_chunk;

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
  // Optionally allocate pinned (page-locked) host memory for faster H2D transfers
  bool want_pin = (opt.pin_mode != PinMode::OFF);
  if (want_pin) {
    if (cudaHostAlloc(&C.h2d_pinned_base, C.per_stream_batch * gpu_chunk,
                      cudaHostAllocDefault) != cudaSuccess)
      C.h2d_pinned_base = nullptr;  // fall back to pageable memory
    C.d2h_pinned_base = nullptr;    // D2H uses exact-size copies
  }

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
  C.h_stats.resize(C.per_stream_batch);
  C.batch.reserve(C.per_stream_batch);

  // Create CUDA events for timing
  cudaEventCreateWithFlags(&C.ev_h2d_begin, cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_h2d_end,   cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_comp_end,  cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_d2h_end,   cudaEventDefault);

  C.stats.pinned_h2d = (C.h2d_pinned_base != nullptr);
  C.stats.pinned_d2h = false;
  C.stats.batch_capacity = per_stream_batch;
  return true;
}

static void free_stream_buffers_only(StreamCtx & C)
{
  if (C.d_in_base) { cudaFree(C.d_in_base); }
  if (C.d_out_base) { cudaFree(C.d_out_base); }
  if (C.d_temp) { cudaFree(C.d_temp); }
  if (C.d_in_ptrs) { cudaFree(C.d_in_ptrs); }
  if (C.d_out_ptrs) { cudaFree(C.d_out_ptrs); }
  if (C.d_in_sizes) { cudaFree(C.d_in_sizes); }
  if (C.d_comp_sizes) { cudaFree(C.d_comp_sizes); }
  if (C.d_stats) { cudaFree(C.d_stats); }
  if (C.h2d_pinned_base) { cudaFreeHost(C.h2d_pinned_base); }
  if (C.d2h_pinned_base) { cudaFreeHost(C.d2h_pinned_base); }
  if (C.ev_h2d_begin) { cudaEventDestroy(C.ev_h2d_begin); }
  if (C.ev_h2d_end) { cudaEventDestroy(C.ev_h2d_end); }
  if (C.ev_comp_end) { cudaEventDestroy(C.ev_comp_end); }
  if (C.ev_d2h_end) { cudaEventDestroy(C.ev_d2h_end); }
  C = StreamCtx{};
}

static bool try_grow_stream(StreamCtx & C, size_t desired_batch, nvcompBatchedZstdCompressOpts_t comp_opts, const Options & opt)
{
  if (desired_batch <= C.per_stream_batch) return false;
  StreamCtx tmp; tmp.stream = C.stream;
  try {
    if (!allocate_stream_buffers(tmp, desired_batch, C.gpu_chunk,
                                 C.max_out_chunk, comp_opts, opt))
      return false;
  } catch (...) {
    return false;
  }

  // Swap all buffers from the new (larger) allocation into the live context,
  // then free the old (smaller) allocation.
  std::swap(C.d_in_base,       tmp.d_in_base);
  std::swap(C.d_out_base,      tmp.d_out_base);
  std::swap(C.d_temp,          tmp.d_temp);
  std::swap(C.d_in_ptrs,       tmp.d_in_ptrs);
  std::swap(C.d_out_ptrs,      tmp.d_out_ptrs);
  std::swap(C.d_in_sizes,      tmp.d_in_sizes);
  std::swap(C.d_comp_sizes,    tmp.d_comp_sizes);
  std::swap(C.d_stats,         tmp.d_stats);
  std::swap(C.h2d_pinned_base, tmp.h2d_pinned_base);
  std::swap(C.d2h_pinned_base, tmp.d2h_pinned_base);
  std::swap(C.h_in_sizes,      tmp.h_in_sizes);
  std::swap(C.h_comp_sizes,    tmp.h_comp_sizes);
  std::swap(C.h_stats,         tmp.h_stats);
  std::swap(C.batch,           tmp.batch);
  std::swap(C.per_stream_batch, tmp.per_stream_batch);
  std::swap(C.gpu_chunk,       tmp.gpu_chunk);
  std::swap(C.max_out_chunk,   tmp.max_out_chunk);
  std::swap(C.temp_bytes_used, tmp.temp_bytes_used);
  std::swap(C.ev_h2d_begin,    tmp.ev_h2d_begin);
  std::swap(C.ev_h2d_end,      tmp.ev_h2d_end);
  std::swap(C.ev_comp_end,     tmp.ev_comp_end);
  std::swap(C.ev_d2h_end,      tmp.ev_d2h_end);

  C.stats.pinned_h2d    = (C.h2d_pinned_base != nullptr);
  C.stats.pinned_d2h    = false;
  C.stats.batch_capacity = C.per_stream_batch;

  tmp.stream = nullptr;
  free_stream_buffers_only(tmp);
  return true;
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

static void gpu_worker(
  int device_id,
  int slot_index,   // positional index into per_dev/json_sink arrays
  Options opt,
  TaskQueue * queue,
  RescueQueue * rescue,
  ResultStore * results,
  DevStats * devstats,
  StatsSink * json_sink,
  Meter * m,
  HybridSched * sched,
  std::atomic<bool> * any_gpu_failed,
  std::atomic<bool> * abort_on_failure,
  std::string * fatal_msg,
  std::atomic<bool> * gpu_started_flag)
{
  (void)m; std::shared_ptr<std::vector<StreamCtx>> ctxs_ptr;
  try {
    uint64_t init_t0 = g_perf ? now_ns() : 0;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    const size_t host_chunk_bytes = std::max<size_t>(1, opt.chunk_mib) * ONE_MIB;
    const size_t gpu_chunk = std::min(host_chunk_bytes, GPU_SUBCHUNK_MAX);
    nvcompBatchedZstdCompressOpts_t comp_opts = nvcompBatchedZstdCompressDefaultOpts; size_t max_out_chunk=0;
    checkNvcomp(nvcompBatchedZstdCompressGetMaxOutputChunkSize(gpu_chunk, comp_opts, &max_out_chunk), "nvcompBatchedZstdCompressGetMaxOutputChunkSize");
    const size_t stream_count = std::max<size_t>(1, opt.gpu_streams);
    size_t per_stream_cap = std::max<size_t>(1, (opt.gpu_batch_cap + stream_count - 1) / stream_count);
    per_stream_cap = std::min(per_stream_cap, HARD_BATCH_CAP);
    double per_stream_frac = std::max(0.05, std::min(0.95, opt.gpu_mem_fraction / double(stream_count)));

    ctxs_ptr = std::make_shared<std::vector<StreamCtx>>(stream_count);
    auto & ctxs = *ctxs_ptr;
    for (size_t s=0; s<stream_count; ++s) {
      StreamCtx & C = ctxs[s];
      checkCuda(cudaStreamCreate(&C.stream), "cudaStreamCreate");
      // Calculate how many subchunks this stream can hold based on free VRAM
      size_t free_b = 0, total_b = 0;
      if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b == 0) {
        C.per_stream_batch = std::max<size_t>(1, per_stream_cap);
      } else {
        const size_t budget   = static_cast<size_t>(free_b * per_stream_frac);
        const size_t per_in   = gpu_chunk;
        const size_t per_out  = max_out_chunk;
        const size_t per_meta = sizeof(void*) * 2 + sizeof(size_t) * 2 + sizeof(nvcompStatus_t);

        size_t bsz = budget / std::max<size_t>(1, per_in + per_out + per_meta);
        if (bsz == 0) bsz = 1;
        bsz = std::min(bsz, per_stream_cap);
        bsz = std::min(bsz, HARD_BATCH_CAP);
        if (bsz == 0) bsz = 1;
        C.per_stream_batch = bsz;
      }
      while (!allocate_stream_buffers(C, C.per_stream_batch, gpu_chunk, max_out_chunk, comp_opts, opt)) {
        if (C.per_stream_batch==1) throw std::runtime_error("insufficient GPU memory for per-stream batch=1");
        C.per_stream_batch = std::max<size_t>(1, C.per_stream_batch/2);
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
    while (true) {
      bool submitted_any=false;
      // Adjust/grow if room
      for (auto & C: ctxs) {
        if (C.busy) { continue; }
        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::duration<double>>(now - C.last_adjust).count();
        if (secs < GROW_CHECK_SEC) { continue; }
        size_t free_b=0,total_b=0;
        if (cudaMemGetInfo(&free_b,&total_b)==cudaSuccess && free_b>0) {
          size_t target = std::min<size_t>(
              std::min<size_t>(C.per_stream_batch * 2, per_stream_cap), HARD_BATCH_CAP);
          if (try_grow_stream(C, target, comp_opts, opt)) {
            if (opt.verbosity >= V_DEBUG) {
              std::ostringstream os;
              os << "[GPU" << device_id
                 << "] stream grew to batch=" << C.per_stream_batch;
              vlog(V_DEBUG, opt, os.str() + "\n");
            }
          }
        }
        C.last_adjust = now;
      }

      // Submit batches
      for (auto & C : ctxs) {
        if (C.busy) { continue; }
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        if (!queue->pop_batch(C.per_stream_batch, C.batch)) {
          if (g_perf) { g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0); g_perf->queue_wait_count.fetch_add(1); }
          producer_done_seen = true; continue;
        }
        if (C.batch.empty()) {
          if (g_perf) { g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0); g_perf->queue_wait_count.fetch_add(1); }
          continue;
        }
        if (gpu_started_flag) { gpu_started_flag->store(true, std::memory_order_release); }
        if (sched) { sched->mark_gpu_take(C.batch.size()); }
        if (g_perf) g_perf->sched_gpu_tasks.fetch_add(C.batch.size());
        C.filled = C.batch.size();

        // -vv: print take line
        if (opt.verbosity >= V_DEBUG) {
          size_t seq_lo = C.batch.front().seq;
          size_t seq_hi = C.batch.back().seq;
          uint64_t tin = 0;
          for (size_t i = 0; i < C.filled; ++i)
            tin += C.batch[i].data.size();
          char tin_s[32];
          human_bytes(double(tin), tin_s, sizeof(tin_s));
          std::ostringstream os;
          os << "[GPU" << device_id << "/S" << C.stats.stream_index
             << "] take N=" << C.filled
             << " seq=[" << seq_lo << ".." << seq_hi << "]"
             << " in=" << tin_s;
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        cudaEventRecord(C.ev_h2d_begin, C.stream);
        // Upload each subchunk to its slot in the device input buffer
        for (size_t i = 0; i < C.filled; ++i) {
          const Task & t = C.batch[i];
          void * d_dst = static_cast<char*>(C.d_in_base) + i * C.gpu_chunk;

          if (C.h2d_pinned_base) {
            // Copy to pinned staging buffer, then async H2D (faster for large transfers)
            void * h_src = static_cast<char*>(C.h2d_pinned_base) + i * C.gpu_chunk;
            std::memcpy(h_src, t.data.data(), t.data.size());
            checkCuda(cudaMemcpyAsync(d_dst, h_src, t.data.size(),
                                      cudaMemcpyHostToDevice, C.stream),
                      "cudaMemcpyAsync(H2D pinned)");
          } else {
            // Direct pageable H2D (simpler but slower)
            checkCuda(cudaMemcpyAsync(d_dst, t.data.data(), t.data.size(),
                                      cudaMemcpyHostToDevice, C.stream),
                      "cudaMemcpyAsync(H2D)");
          }
          C.h_in_sizes[i] = t.data.size();
        }
        checkCuda(cudaMemcpyAsync(C.d_in_sizes, C.h_in_sizes.data(), sizeof(size_t)*C.filled, cudaMemcpyHostToDevice, C.stream), "cudaMemcpyAsync(d_in_sizes)");
        cudaEventRecord(C.ev_h2d_end, C.stream);

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
        cudaEventRecord(C.ev_d2h_end, C.stream);
        C.busy = true; submitted_any = true;
      }

      // Poll completions
      for (auto & C : ctxs) {
        if (!C.busy) { continue; }
        cudaError_t q = cudaStreamQuery(C.stream);
        if (q == cudaSuccess) {
          uint64_t d2h_t0 = g_perf ? now_ns() : 0;
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes)");
          // Measure per-phase timing from CUDA events
          float h2d_ms = 0, comp_ms = 0, d2h_ms = 0, tot_ms = 0;
          cudaEventElapsedTime(&h2d_ms,  C.ev_h2d_begin, C.ev_h2d_end);
          cudaEventElapsedTime(&comp_ms, C.ev_h2d_end,   C.ev_comp_end);
          cudaEventElapsedTime(&d2h_ms,  C.ev_comp_end,  C.ev_d2h_end);
          cudaEventElapsedTime(&tot_ms,  C.ev_h2d_begin, C.ev_d2h_end);

          if (g_perf) {
            g_perf->h2d_ns.fetch_add(uint64_t(h2d_ms * 1e6));
            g_perf->h2d_count.fetch_add(1);
            g_perf->kernel_ns.fetch_add(uint64_t(comp_ms * 1e6));
            g_perf->kernel_count.fetch_add(1);
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

          uint64_t in_sum = 0, out_sum = 0;
          for (size_t i=0;i<C.filled;++i) {
            if (C.h_stats[i] != nvcompSuccess) throw std::runtime_error("nvCOMP per-chunk status != nvcompSuccess");
            const size_t csz = C.h_comp_sizes[i]; out_sum += csz; in_sum += C.h_in_sizes[i];
            std::vector<char> h_out(csz);
            const void * d_src = static_cast<char*>(C.d_out_base)
                                 + i * C.max_out_chunk;
            checkCuda(cudaMemcpy(h_out.data(), d_src, csz,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(D2H exact)");
            {
              std::lock_guard<std::mutex> lk(results->m);
              results->data.emplace(C.batch[i].seq, std::move(h_out));
            }
            results->cv.notify_one();
          }
          if (g_perf) {
            g_perf->d2h_ns.fetch_add(now_ns() - d2h_t0);
            g_perf->d2h_bytes.fetch_add(out_sum);
            g_perf->d2h_count.fetch_add(1);
            g_perf->h2d_bytes.fetch_add(in_sum);
            g_perf->gpu_batch_ns.fetch_add(uint64_t(tot_ms * 1e6));
            g_perf->gpu_batch_count.fetch_add(1);
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

          // -vv: batch completion line
          if (opt.verbosity >= V_DEBUG) {
            char in_s[32], out_s[32];
            human_bytes(double(in_sum), in_s, sizeof(in_s));
            human_bytes(double(out_sum), out_s, sizeof(out_s));
            double thr_gib = (tot_ms > 0.0)
                             ? double(in_sum) / (tot_ms / 1000.0) / 1e9 : 0.0;
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << C.stats.stream_index
               << "] done N=" << C.filled
               << " in=" << in_s << " out=" << out_s
               << " h2d=" << std::fixed << std::setprecision(2) << h2d_ms
               << "ms comp=" << comp_ms
               << "ms d2h=" << d2h_ms
               << "ms tot=" << tot_ms
               << "ms thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
            vlog(V_DEBUG, opt, os.str() + "\n");
          }

          C.busy = false;
          C.filled = 0;
          C.batch.clear();
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
      if (!submitted_any) {
        bool blocked=false;
        for (auto & C: ctxs) {
          if (!C.busy) { continue; }
          checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize");
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses sync)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes sync)");
          // Measure per-phase timing (synchronous path)
          float h2d_ms = 0, comp_ms = 0, d2h_ms = 0, tot_ms = 0;
          cudaEventElapsedTime(&h2d_ms,  C.ev_h2d_begin, C.ev_h2d_end);
          cudaEventElapsedTime(&comp_ms, C.ev_h2d_end,   C.ev_comp_end);
          cudaEventElapsedTime(&d2h_ms,  C.ev_comp_end,  C.ev_d2h_end);
          cudaEventElapsedTime(&tot_ms,  C.ev_h2d_begin, C.ev_d2h_end);
          {
            std::lock_guard<std::mutex> lk(devstats->m);
            devstats->h2d_ms  += h2d_ms;
            devstats->comp_ms += comp_ms;
            devstats->d2h_ms  += d2h_ms;
            devstats->total_ms += tot_ms;
            devstats->batches += 1;
          }
          uint64_t in_sum = 0, out_sum = 0;
          for (size_t i = 0; i < C.filled; ++i) {
            in_sum  += C.h_in_sizes[i];
            out_sum += C.h_comp_sizes[i];
          }
          {
            // D2H exact copies and release to writer (synchronous path)
            for (size_t i = 0; i < C.filled; ++i) {
              const size_t csz = C.h_comp_sizes[i];
              std::vector<char> h_out(csz);
              const void * d_src = static_cast<char*>(C.d_out_base) + i * C.max_out_chunk;
              checkCuda(cudaMemcpy(h_out.data(), d_src, csz, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H exact sync)");
              {
                std::lock_guard<std::mutex> lk(results->m);
                results->data.emplace(C.batch[i].seq, std::move(h_out));
              }
              results->cv.notify_one();
            }
          }
          #ifdef HAVE_NVCOMP
          if (sched) sched->add_gpu_bytes(in_sum);
          #endif
          // Accumulate per-stream stats (synchronous path)
          C.stats.h2d_ms   += h2d_ms;
          C.stats.comp_ms  += comp_ms;
          C.stats.d2h_ms   += d2h_ms;
          C.stats.total_ms += tot_ms;
          C.stats.in_bytes += in_sum;
          C.stats.out_bytes += out_sum;
          C.stats.batches  += 1;
          C.stats.chunks   += C.filled;

          if (opt.verbosity >= V_DEBUG) {
            char in_s[32], out_s[32];
            human_bytes(double(in_sum), in_s, sizeof(in_s));
            human_bytes(double(out_sum), out_s, sizeof(out_s));
            double thr_gib = (tot_ms > 0.0)
                             ? double(in_sum) / (tot_ms / 1000.0) / 1e9 : 0.0;
            std::ostringstream os;
            os << "[GPU" << device_id << "/S" << C.stats.stream_index
               << "] done N=" << C.filled
               << " in=" << in_s << " out=" << out_s
               << " h2d=" << std::fixed << std::setprecision(2) << h2d_ms
               << "ms comp=" << comp_ms
               << "ms d2h=" << d2h_ms
               << "ms tot=" << tot_ms
               << "ms thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
            vlog(V_DEBUG, opt, os.str() + "\n");
          }

          C.busy = false;
          C.filled = 0;
          C.batch.clear();
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
    for (auto & C : ctxs) {
      free_stream_buffers_only(C);
      if (C.stream) cudaStreamDestroy(C.stream);
      C.stream = nullptr;
    }
  }
  catch (const std::exception & e) {
    // GPU failure: record error, rescue any in-flight chunks to CPU fallback
    *any_gpu_failed = true;
    *fatal_msg = std::string("[GPU") + std::to_string(device_id) + "] " + e.what();

    try {
      if (rescue) {
        if (auto sp = std::weak_ptr<std::vector<StreamCtx>>(ctxs_ptr).lock()) {
          for (auto & C : *sp) {
            // Re-enqueue in-flight chunks to the rescue (CPU) queue
            if (C.busy && !C.batch.empty()) {
              for (size_t i = 0; i < C.filled; ++i)
                rescue->push(Task{ C.batch[i].seq, C.batch[i].data });
            }
            free_stream_buffers_only(C);
            if (C.stream) cudaStreamDestroy(C.stream);
          }
        }
      }
    } catch (...) {}

    // Wake writer in case it's waiting on results
    {
      std::lock_guard<std::mutex> lk(results->m);
      results->cv.notify_all();
    }
  }
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
        os << "GPU selection: using device";
        if (want > 1) os << "s";
        for (int i = 0; i < want; ++i) {
          os << (i ? "," : " ") << infos[i].cuda_id;
          os << " (" << infos[i].gpu_util << "% util";
          if (infos[i].score != infos[i].gpu_util)
            os << ", score=" << infos[i].score << " (NUMA penalty)";
          os << ", N" << infos[i].numa_node;
          os << ", " << (infos[i].free_bytes / (1024*1024)) << " MiB free)";
        }
        if (want < (int)infos.size()) {
          os << "  skipped:";
          for (int i = want; i < (int)infos.size(); ++i)
            os << " " << infos[i].cuda_id << "("
               << infos[i].gpu_util << "%"
               << (infos[i].score != infos[i].gpu_util
                   ? ",s=" + std::to_string(infos[i].score) : "")
               << ",N" << infos[i].numa_node
               << "," << (infos[i].free_bytes / (1024*1024)) << "MiB)";
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
    os << "GPU selection: using device";
    if (n > 1) os << "s";
    for (int i = 0; i < n; ++i) {
      os << (i ? "," : " ") << devs[i].id;
      os << " (";
      if (devs[i].has_util) os << devs[i].gpu_util << "% util, ";
      os << (devs[i].free_bytes / (1024*1024)) << " MiB free)";
    }
    if (n < (int)devs.size()) {
      os << "  skipped:";
      for (int i = n; i < (int)devs.size(); ++i) {
        os << " " << devs[i].id << "(";
        if (devs[i].has_util) os << devs[i].gpu_util << "%,";
        os << (devs[i].free_bytes / (1024*1024)) << "MiB)";
      }
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
      die("GPU requested (--gpu-only) but no CUDA devices available");
    vlog(V_VERBOSE, opt, "nvCOMP: no GPUs found; falling back to MT CPU\n");
    compress_cpu_mt(in, out, opt, m);
    return;
  }

  // Apply --gpu-devices limit (0 = all for compress)
  const int total_hw_devices = device_count;
  if (opt.gpu_devices > 0 && opt.gpu_devices < device_count) {
    vlog(V_VERBOSE, opt, "limiting to " + std::to_string(opt.gpu_devices)
         + " of " + std::to_string(device_count) + " GPU devices\n");
    device_count = opt.gpu_devices;
  }

  if (opt.verbosity >= V_VERBOSE && (opt.level_user_set || opt.fast_flag || opt.best_flag))
    vlog(V_VERBOSE, opt,
         "note: GPU uses fixed compression; CPU frames honor level "
         + std::to_string(opt.level) + "\n");

  // ---- Determine chunk size ----
  size_t chosen_mib = opt.chunk_mib;
  if (!opt.chunk_user_set) {
    chosen_mib = auto_chunk_mib_gpu(in, opt, device_count);
    std::ostringstream os;
    os << "auto-chunk (GPU," << device_count << " devices): " << chosen_mib << " MiB";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  // ---- Shared state for all workers ----
  TaskQueue   queue;
  RescueQueue rescue;
  ResultStore results;
  std::atomic<size_t> seq_counter{0};
  std::atomic<bool>   any_gpu_failed{false};
  std::atomic<bool>   abort_on_failure{ opt.gpu_only };
  std::atomic<bool>   gpu_started{false};

  // Select GPUs before allocating per-device arrays
  auto gpu_ids = select_best_gpus(total_hw_devices, device_count, opt);
  const int gpu_count_early = (int)gpu_ids.size();

  std::vector<DevStats> per_dev(gpu_count_early);
  StatsSink json_sink(gpu_count_early);
  CpuAgg cpuagg{};
  cpuagg.threads = 0;

  // Get total input size for progress percentage (unknown for pipes)
  uint64_t total_in = 0;
  if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
    total_in = (uint64_t)fs::file_size(opt.input);

  // Start progress bar and ordered-writer threads
  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m);

  // ---- Hybrid scheduler (adaptive CPU/GPU work-sharing) ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, /*cpu_threads*/0, device_count, opt);
    sched = sched_ptr.get();
    tick_thr = std::thread(tick_loop_fn, std::ref(tick_done), sched);

    if (opt.verbosity >= V_VERBOSE) {
      double share_pct = (opt.cpu_share >= 0.0)
                         ? (opt.cpu_share * 100.0)
                         : (sched->target_share() * 100.0);
      const char * mode_str = (opt.cpu_share >= 0.0) ? "% (fixed)" : "% (adaptive)";
      std::ostringstream os;
      os << std::fixed << std::setprecision(1)
         << "Using hybrid mode: CPU share " << share_pct << mode_str;
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  }

  // ---- Rescue pool: CPU fallback threads for GPU failures ----
  std::vector<std::thread> rescue_pool;
  {
    unsigned ths = std::max(1u, std::thread::hardware_concurrency() / 2);
    rescue_pool.reserve(ths);
    for (unsigned i = 0; i < ths; ++i)
      rescue_pool.emplace_back(cpu_worker_rescue, (int)i, &rescue, &results, &opt, m, &cpuagg);

    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[RESCUE] " << ths << " rescue threads online";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  }

  // In hybrid mode, start CPU pool BEFORE GPU workers so CPUs can compress
  // early chunks while GPUs are still initializing (CUDA context, memory alloc).
  // Once GPUs come online, the adaptive scheduler shifts work to them.
  // ---- CPU worker pool (started BEFORE GPU for early-start optimization) ----
  // In hybrid mode, CPUs begin compressing chunks immediately while GPUs are
  // still doing CUDA context init and memory allocation.
  std::vector<std::thread> cpu_pool;
  int cpu_threads = 0;
  if (sched) {
    cpu_threads = resolve_cpu_threads(opt.cpu_threads);
    cpuagg.threads = cpu_threads;
    cpuagg.per_thread.resize((size_t)cpu_threads);

    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "hybrid: starting CPU pool: " << cpu_threads
         << " threads (GPUs initializing in background)";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
    for (int i = 0; i < cpu_threads; ++i)
      cpu_pool.emplace_back(cpu_worker, i, &queue, &results, &opt, m, (void*)sched, &cpuagg);
  }

  // ---- GPU workers (init CUDA context, allocate memory  CPUs already working) ----
  const int gpu_count = gpu_count_early;
  std::vector<std::thread> workers;
  workers.reserve(gpu_count);
  Options opt_for_workers = opt;
  opt_for_workers.chunk_mib = chosen_mib;
  std::vector<std::string> fatal_msgs(gpu_count);

  for (int i = 0; i < gpu_count; ++i) {
    workers.emplace_back(gpu_worker, gpu_ids[i], i, opt_for_workers,
                         &queue, &rescue, &results,
                         &per_dev[size_t(i)], &json_sink, m, sched,
                         &any_gpu_failed, &abort_on_failure,
                         &fatal_msgs[size_t(i)], &gpu_started);
  }
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[GPU] " << gpu_count << " device worker"
       << (gpu_count > 1 ? "s" : "") << " online";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }

  // ---- Producer: read input, split into GPU-sized subchunks, enqueue ----
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  std::vector<char> host_in(host_chunk);
  while (true) {
    uint64_t rd_t0 = g_perf ? now_ns() : 0;
    size_t n_host = std::fread(host_in.data(), 1, host_chunk, in);
    if (g_perf && n_host > 0) {
      g_perf->read_ns.fetch_add(now_ns() - rd_t0);
      g_perf->read_bytes_total.fetch_add(n_host);
    }
    if (n_host == 0) break;
    if (m) m->read_bytes.fetch_add(n_host);

    // Split each host chunk into subchunks that fit in GPU memory
    const size_t gpu_chunk = std::min(host_chunk, GPU_SUBCHUNK_MAX);
    size_t off = 0;
    while (off < n_host) {
      size_t sub_n = std::min(gpu_chunk, n_host - off);
      Task t;
      t.seq = seq_counter.fetch_add(1, std::memory_order_relaxed);
      t.data.assign(host_in.data() + off, host_in.data() + off + sub_n);
      queue.push(std::move(t));
      off += sub_n;
    }

    if (abort_on_failure.load() && any_gpu_failed.load()) break;
  }

  // Signal workers that all input has been read
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
  }
  results.cv.notify_all();

  // ---- Teardown: join all threads in correct order ----
  for (auto & th : workers) th.join();

  // Check for GPU failures (fatal in --gpu-only mode)
  if (abort_on_failure.load() && any_gpu_failed.load()) {
    progress_done = true;
    progress_thr.join();
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.workers_done = true;
      results.cv.notify_all();
    }
    writer_thr.join();
    std::string msg = "GPU path failed (--gpu-only).";
    for (const auto & s : fatal_msgs)
      if (!s.empty()) { msg += " "; msg += s; }
    die(msg);
  }

  // Drain rescue queue and join CPU pool
  rescue.set_done();
  for (auto & th : rescue_pool) th.join();
  if (!cpu_pool.empty())
    for (auto & th : cpu_pool) th.join();

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
  Options opt,
  TaskQueue * queue,
  RescueQueue * rescue,
  ResultStore * results,
  Meter * m,
  HybridSched * sched,
  std::atomic<bool> * any_gpu_failed,
  std::atomic<bool> * abort_on_failure,
  std::string * fatal_msg)
{
  (void)m;
  try {
    uint64_t init_t0 = g_perf ? now_ns() : 0;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    const size_t stream_count = std::max<size_t>(1, opt.gpu_streams);
    // For decompress, gpu_batch_cap is PER STREAM (not divided across streams).
    // Kernel launch overhead dominates, so each stream needs large batches.
    // Compress divides across streams for VRAM management, but decompress
    // frames are small (compressed) so VRAM isn't the bottleneck.
    const size_t per_stream_cap = std::min(opt.gpu_batch_cap, HARD_BATCH_CAP);

    // We need to allocate per-stream buffers.  Unlike compression, we need
    // to handle variable decompressed sizes.  We pre-allocate based on the
    // chunk size (which upper-bounds both compressed and decompressed size)
    // and reuse across batches to avoid per-batch cudaMalloc overhead.

    const size_t host_chunk_bytes = std::max<size_t>(1, opt.chunk_mib) * ONE_MIB;

    // Per-stream state for decompression
    struct DecompStreamCtx {
      cudaStream_t stream{};
      cudaEvent_t ev_begin{}, ev_end{};
      std::vector<Task> batch;
      bool busy = false;
      size_t filled = 0;
      size_t stream_index = 0;

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

    std::vector<DecompStreamCtx> ctxs(stream_count);
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
      // Estimate temp size  we'll get the real size on first batch and grow if needed
      {
        size_t est_temp = per_stream_cap * 1024;  // small initial estimate
        C.ensure_buffers(per_stream_cap, init_comp, init_decomp, est_temp);
      }

      if (opt.verbosity >= V_DEBUG) {
        std::ostringstream os;
        os << "[GPU-D" << device_id << "/S" << s
           << "] pre-alloc batch=" << per_stream_cap
           << " comp=" << (init_comp / ONE_MIB) << "MiB"
           << " decomp=" << (init_decomp / ONE_MIB) << "MiB";
        vlog(V_DEBUG, opt, os.str() + "\n");
      }
    }

    bool producer_done_seen = false;
    if (g_perf) { uint64_t dt = now_ns() - init_t0; g_perf->cuda_init_sum_ns.fetch_add(dt); g_perf->cuda_init_count.fetch_add(1); uint64_t cur = g_perf->cuda_init_max_ns.load(); while (dt > cur && !g_perf->cuda_init_max_ns.compare_exchange_weak(cur, dt)); };
    // Signal scheduler that this GPU is ready for work
    if (sched) sched->set_gpu_ready();

    while (true) {
      bool submitted_any = false;

      // Submit batches
      for (auto & C : ctxs) {
        if (C.busy) continue;
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        // Signal scheduler: this GPU stream wants data (blocks CPU workers)
        if (sched) sched->gpu_wants_data();
        if (!queue->pop_batch_greedy(per_stream_cap, C.batch)) {
          if (sched) sched->gpu_got_data();
          if (g_perf) {
            g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0);
            g_perf->queue_wait_count.fetch_add(1);
          }
          producer_done_seen = true;
          continue;
        }
        // Signal scheduler: this GPU stream got its data (unblocks CPU workers)
        if (sched) sched->gpu_got_data();
        if (g_perf && C.batch.empty()) {
          g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0);
          g_perf->queue_wait_count.fetch_add(1);
        }
        if (C.batch.empty()) continue;
        if (g_perf) {
          g_perf->sched_gpu_tasks.fetch_add(C.batch.size());
        }
        if (sched) sched->mark_gpu_take(C.batch.size());
        C.filled = C.batch.size();
        uint64_t batch_t0 = g_perf ? now_ns() : 0;

        // Determine max sizes for this batch
        size_t max_comp = 0, max_decomp = 0;
        for (size_t i = 0; i < C.filled; ++i) {
          max_comp   = std::max(max_comp,   C.batch[i].data.size());
          max_decomp = std::max(max_decomp, C.batch[i].decomp_size);
        }

        if (opt.verbosity >= V_DEBUG) {
          char in_s[32];
          uint64_t tin = 0;
          for (size_t i = 0; i < C.filled; ++i) tin += C.batch[i].data.size();
          human_bytes(double(tin), in_s, sizeof(in_s));
          std::ostringstream os;
          os << "[GPU-D" << device_id << "/S" << C.stream_index
             << "] take N=" << C.filled << " in=" << in_s;
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        // Get temp workspace size for this batch configuration.
        // We need a dummy ensure_buffers first if sizes grew, then query temp.
        // Use a two-pass approach: ensure base buffers, query temp, re-ensure with temp.
        if (!C.ensure_buffers(C.filled, max_comp, max_decomp, C.temp_bytes)) {
          throw std::runtime_error("GPU decomp: failed to allocate device buffers");
        }

        // Upload compressed data H2D (async)
        uint64_t h2d_t0 = g_perf ? now_ns() : 0;
        uint64_t h2d_bytes_batch = 0;
        cudaEventRecord(C.ev_begin, C.stream);
        for (size_t i = 0; i < C.filled; ++i) {
          void * d_dst = static_cast<char*>(C.d_comp_buf) + i * C.alloc_comp;
          checkCuda(cudaMemcpyAsync(d_dst, C.batch[i].data.data(),
                                    C.batch[i].data.size(),
                                    cudaMemcpyHostToDevice, C.stream),
                    "cudaMemcpyAsync(H2D decomp)");
          h2d_bytes_batch += C.batch[i].data.size();
          C.h_comp_sizes[i]   = C.batch[i].data.size();
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
        if (g_perf) {
          g_perf->h2d_ns.fetch_add(now_ns() - h2d_t0);
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
          if (!C.ensure_buffers(C.filled, max_comp, max_decomp, needed_temp))
            throw std::runtime_error("GPU decomp: failed to grow temp buffer");
          // Re-upload pointer/size arrays since ensure_buffers may have reallocated
          for (size_t i = 0; i < C.filled; ++i) {
            void * d_dst = static_cast<char*>(C.d_comp_buf) + i * C.alloc_comp;
            checkCuda(cudaMemcpyAsync(d_dst, C.batch[i].data.data(),
                                      C.batch[i].data.size(),
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
        uint64_t kern_t0 = g_perf ? now_ns() : 0;
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
        if (g_perf) {
          g_perf->kernel_ns.fetch_add(now_ns() - kern_t0);
          g_perf->kernel_count.fetch_add(1);
        }

        // Read back statuses and actual sizes in bulk
        uint64_t d2h_t0 = g_perf ? now_ns() : 0;
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

        for (size_t i = 0; i < C.filled; ++i) {
          if (C.h_statuses[i] != nvcompSuccess)
            throw std::runtime_error("nvCOMP decompress per-chunk status != success");

          size_t actual = C.h_actual[i];
          out_sum += actual;
          d2h_bytes_batch += actual;

          // Download decompressed data
          std::vector<char> h_out(actual);
          const void * d_src = static_cast<char*>(C.d_decomp_buf) + i * C.alloc_decomp;
          checkCuda(cudaMemcpy(h_out.data(), d_src, actual,
                               cudaMemcpyDeviceToHost), "D2H decomp data");

          // Deliver to result store (wrote_bytes tracked by writer thread)
          uint64_t rl_t0 = g_perf ? now_ns() : 0;
          {
            std::lock_guard<std::mutex> lk(results->m);
            results->data.emplace(C.batch[i].seq, std::move(h_out));
          }
          if (g_perf) g_perf->result_lock_ns.fetch_add(now_ns() - rl_t0);
          results->cv.notify_one();
        }

        if (g_perf) {
          g_perf->d2h_ns.fetch_add(now_ns() - d2h_t0);
          g_perf->d2h_bytes.fetch_add(d2h_bytes_batch);
          g_perf->d2h_count.fetch_add(1);
          g_perf->gpu_batch_ns.fetch_add(now_ns() - batch_t0);
          g_perf->gpu_batch_count.fetch_add(1);
        }

        // Track compressed bytes consumed for progress bar
        {
          uint64_t in_sum = 0;
          for (size_t i = 0; i < C.filled; ++i)
            in_sum += C.batch[i].data.size();
          if (m) m->read_bytes.fetch_add(in_sum);
        }

        if (sched) sched->add_gpu_bytes(out_sum);

        if (opt.verbosity >= V_DEBUG) {
          char out_s[32];
          human_bytes(double(out_sum), out_s, sizeof(out_s));
          double thr_gib = (batch_ms > 0.0)
                           ? double(out_sum) / (batch_ms / 1000.0) / 1e9 : 0.0;
          std::ostringstream os;
          os << "[GPU-D" << device_id << "/S" << C.stream_index
             << "] done N=" << C.filled
             << " out=" << out_s
             << " ms=" << std::fixed << std::setprecision(2) << batch_ms
             << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        // Buffers are reused  no per-batch free needed
        C.busy = false;
        C.filled = 0;
        C.batch.clear();
        submitted_any = true;
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
    for (auto & C : ctxs) {
      C.free_device();
      if (C.ev_begin) cudaEventDestroy(C.ev_begin);
      if (C.ev_end)   cudaEventDestroy(C.ev_end);
      if (C.stream)   cudaStreamDestroy(C.stream);
    }
  }
  catch (const std::exception & e) {
    *any_gpu_failed = true;
    *fatal_msg = std::string("[GPU-D") + std::to_string(device_id) + "] " + e.what();

    // Rescue in-flight chunks back to CPU
    try {
      if (rescue) {
        // No in-flight since we synchronize per-batch, but rescue unprocessed
      }
    } catch (...) {}

    {
      std::lock_guard<std::mutex> lk(results->m);
      results->cv.notify_all();
    }
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
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    if (opt.gpu_only)
      die("GPU requested (--gpu-only) but no CUDA devices available");
    vlog(V_VERBOSE, opt, "nvCOMP: no GPUs found; falling back to MT CPU decompress\n");
    // Fall through with device_count=0; CPU pool will handle everything
  }

  // Apply --gpu-devices limit.
  // Default (0) = 1 GPU for decompress: D2H transfer of the full
  // uncompressed data saturates PCIe; multiple GPUs share bandwidth
  // and are slower than a single GPU with full link speed.
  const int total_hw_devices = device_count;
  if (device_count > 0) {
    int target = opt.gpu_devices;
    if (target == 0) target = std::min(2, device_count);  // auto: 2 GPUs balances kernel parallelism vs PCIe contention
    if (target < device_count) {
      vlog(V_VERBOSE, opt, "decompress: using " + std::to_string(target)
           + " of " + std::to_string(device_count) + " GPU devices"
           + (opt.gpu_devices == 0 ? " (auto  PCIe bandwidth optimal)\n" : "\n"));
      device_count = target;
    }
  }

  // ---- Shared state ----
  TaskQueue queue;
  RescueQueue rescue;
  ResultStore results;
  std::atomic<bool> any_gpu_failed{false};
  std::atomic<bool> abort_on_failure{ opt.gpu_only };

  // ---- Writer thread (outputs decompressed data in order) ----
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m);

  // ---- Hybrid scheduler ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, 0, device_count, opt);
    sched = sched_ptr.get();
    tick_thr = std::thread(tick_loop_fn, std::ref(tick_done), sched);
  }

  // ---- CPU decompression pool (early start) ----
  CpuAgg cpuagg{};
  std::vector<std::thread> cpu_pool;
  int cpu_threads = 0;
  if (sched || (device_count <= 0)) {
    cpu_threads = resolve_cpu_threads(opt.cpu_threads);
    cpuagg.threads = cpu_threads;
    cpuagg.per_thread.resize((size_t)cpu_threads);

    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "hybrid decompress: " << cpu_threads << " CPU threads";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
    for (int i = 0; i < cpu_threads; ++i)
      cpu_pool.emplace_back(cpu_decomp_worker, i, &queue, &results, &opt, m,
                            (void*)sched, &cpuagg);
  }

  // ---- GPU decompression workers ----
  std::vector<int> gpu_ids;
  std::vector<std::thread> gpu_workers;
  std::vector<std::string> fatal_msgs;
  if (device_count > 0) {
    gpu_ids = select_best_gpus(total_hw_devices, device_count, opt);
    const int gpu_count = (int)gpu_ids.size();
    fatal_msgs.resize(gpu_count);
    for (int i = 0; i < gpu_count; ++i) {
      gpu_workers.emplace_back(gpu_decomp_worker, gpu_ids[i], opt,
                               &queue, &rescue, &results, m, sched,
                               &any_gpu_failed, &abort_on_failure,
                               &fatal_msgs[size_t(i)]);
    }
    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[GPU-D] " << (int)gpu_ids.size() << " device(s) online";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  }

  // ---- Stream frames from input into the queue ----
  // Workers start decompressing as soon as the first frame is pushed.
  bool fallback = false;
  std::vector<char> raw_data;
  size_t n_frames = stream_frames_to_queue(in, queue, m, opt, &fallback, &raw_data);

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
    for (auto & th : gpu_workers) th.join();
    for (auto & th : cpu_pool) th.join();
    if (sched) { tick_done = true; if (tick_thr.joinable()) tick_thr.join(); }
    writer_thr.join();

    vlog(V_VERBOSE, opt,
         "frame sizes unknown; falling back to CPU streaming decompress\n");
    decompress_from_buffer(raw_data, out, opt, m);
    return;
  }

  vlog(V_VERBOSE, opt,
       "streamed " + std::to_string(n_frames) + " frames for GPU/hybrid decompress\n");

  // ---- Signal that all frames have been enqueued ----
  queue.set_done();
  {
    std::lock_guard<std::mutex> lk(results.m);
    results.producer_done = true;
    results.total_tasks = queue.total_tasks();
  }
  results.cv.notify_all();

  // ---- Teardown ----
  for (auto & th : gpu_workers) th.join();

  if (abort_on_failure.load() && any_gpu_failed.load()) {
    {
      std::lock_guard<std::mutex> lk(results.m);
      results.workers_done = true;
      results.cv.notify_all();
    }
    writer_thr.join();
    std::string msg = "GPU decompress failed (--gpu-only).";
    for (const auto & s : fatal_msgs)
      if (!s.empty()) { msg += " "; msg += s; }
    die(msg);
  }

  rescue.set_done();
  if (!cpu_pool.empty())
    for (auto & th : cpu_pool) th.join();

  {
    std::lock_guard<std::mutex> lk(results.m);
    results.workers_done = true;
  }
  results.cv.notify_all();

  if (sched) {
    tick_done = true;
    if (tick_thr.joinable()) tick_thr.join();
  }

  writer_thr.join();

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
static std::string derive_output(const std::string & input, Mode mode);
int main(int argc, char ** argv)
{
  setup_signal_handlers();

  Options opt = parse_args(argc, argv);

  int exit_code = EXIT_OK;

  for (size_t file_idx = 0; file_idx < opt.inputs.size(); ++file_idx) {
  // --- Per-file setup ---
  opt.input = opt.inputs[file_idx];

  // For multi-file with no explicit -o, derive output per file
  if (opt.inputs.size() > 1 && !opt.to_stdout) {
    opt.output = derive_output(opt.input, opt.mode);
  }

  FILE * in = open_input(opt.input);
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
    bool exists = fs::exists(opt.output);
    if (exists && !opt.force) {
      die("output exists (use -f to overwrite): " + opt.output);
    }
    if (exists && opt.force) {
      // Atomic overwrite: write to .tmp, rename on success.
      // Skip for device/special files (e.g., /dev/null)  can't create .tmp there.
      bool is_regular = fs::is_regular_file(opt.output);
      if (is_regular) {
        out = open_output_atomic(opt.output, tmp);
        use_atomic = true;
      } else {
        out = std::fopen(opt.output.c_str(), "wb");
        if (!out) die("cannot open output: " + opt.output);
      }
    } else {
      // Direct write: write to final name, delete on failure
      out = std::fopen(opt.output.c_str(), "wb");
      if (!out) die("cannot open output: " + opt.output);
      register_tmp_file(opt.output); // arm cleanup to delete on failure
    }
  }

  // Try O_DIRECT for regular file output (bypasses page cache for ~2-3x write speed).
  // Falls back gracefully: pipes, stdout, device files, and filesystems without
  // O_DIRECT support all use standard fwrite.
#ifndef _WIN32
  std::unique_ptr<DirectWriter> direct_writer;
  if (!to_stdout && out != stdout) {
    // Check if output is a regular file (not /dev/null, pipe, etc.)
    std::string write_path = use_atomic ? tmp : opt.output;
    struct stat st;
    bool is_regular = (stat(write_path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
    if (is_regular) {
      auto dw = std::make_unique<DirectWriter>();
      if (dw->open(write_path)) {
        // Close the FILE*  DirectWriter owns the fd now
        if (out) { std::fclose(out); out = nullptr; }
        direct_writer = std::move(dw);
        vlog(V_VERBOSE, opt, "using O_DIRECT for output (bypass page cache)\n");
      }
      // If O_DIRECT open fails (filesystem doesn't support it), keep using FILE*
    }
  }
  DirectWriter * dw_ptr = direct_writer.get();
#else
  DirectWriter * dw_ptr = nullptr;
#endif

#ifndef _WIN32
  g_direct_writer = dw_ptr;
#endif

  Meter meter; vlog(V_VERBOSE, opt, "gzstd starting...\n");
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
    if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
      total_in_for_progress = (uint64_t)fs::file_size(opt.input);
    prog_thr = std::thread(progress_loop, std::cref(opt), &meter,
                           total_in_for_progress, &prog_done);
  }

  const auto t0 = std::chrono::steady_clock::now();
  if (opt.mode == Mode::COMPRESS) {
#ifdef HAVE_NVCOMP
    if (opt.cpu_only) {
      compress_cpu_mt(in, out, opt, &meter);
      if (!opt.stats_json.empty()) {
        CpuAgg agg{};
        agg.threads = (opt.cpu_threads > 0)
                      ? opt.cpu_threads
                      : std::max(1, int(std::thread::hardware_concurrency()) - 1);
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t0).count();
        write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, agg);
      }
    } else {
      compress_nvcomp(in, out, opt, &meter);
    }
#else
    if (opt.gpu_only) die_usage("This binary was built without nvCOMP; --gpu-only cannot be satisfied");
    if (opt.hybrid) vlog(V_VERBOSE, opt, "Hybrid requested but not available in CPU-only build; using MT CPU.\n");
    compress_cpu_mt(in, out, opt, &meter);
    if (!opt.stats_json.empty()) {
      CpuAgg agg{};
      agg.threads = (opt.cpu_threads > 0)
                    ? opt.cpu_threads
                    : std::max(1, int(std::thread::hardware_concurrency()) - 1);
      double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t0).count();
      write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, agg);
    }
#endif
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
          die("not a Zstd file (bad magic: 0x"
              + ([&]{ char h[16]; snprintf(h, sizeof(h), "%08X", m32); return std::string(h); })()
              + "): " + opt.input
              + "\n  hint: did you mean to compress? use: gzstd " + opt.input);
        }
      }
      // Rewind so the actual decompressor sees the full stream
      std::rewind(in);
    }

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
        "%s : OK (%s => %s, ratio: %.1f%%) @ %s/s",
        base_name.c_str(), comp_s, decomp_s, pct, rate_s);
      std::fprintf(stderr, "\r%s\033[K\n", summary);
      std::fflush(stderr);
    }

    std::fclose(in);
    continue;  // next file
  }
  if (opt.mode == Mode::DECOMPRESS) {
    prog_done = true;
    if (prog_thr.joinable()) prog_thr.join();
    // Summary is printed after fsync below so throughput includes flush time
  }

  if (!to_stdout) {
    // Flush buffered writes to disk.  For large decompressed files this
    // can take several seconds as the kernel writes out dirty pages.
    // Show a summary-style status line so the user knows what's happening.
    if (opt.verbosity >= V_DEFAULT && opt.mode == Mode::DECOMPRESS) {
      uint64_t in_bytes  = meter.read_bytes.load();
      uint64_t out_bytes = meter.wrote_bytes.load();
      char in_s[64], out_s[64];
      human_bytes(double(in_bytes), in_s, sizeof(in_s));
      human_bytes(double(out_bytes), out_s, sizeof(out_s));
      std::string in_name = (opt.input == "-") ? "(stdin)" : opt.input;
      char flush_line[512];
      std::snprintf(flush_line, sizeof(flush_line),
        "%s : %s => %s, flushing to disk...",
        in_name.c_str(), in_s, out_s);
      std::fprintf(stderr, "\r%s\033[K", flush_line);
      std::fflush(stderr);
    }
    // Finalize DirectWriter (flush remaining data, handle unaligned tail)
#ifndef _WIN32
    if (g_direct_writer) {
      if (!g_direct_writer->finalize())
        die("failed to finalize O_DIRECT output");
      g_direct_writer = nullptr;
      direct_writer.reset();  // closes the fd
    }
#endif
    if (out) {
      fsync_file(out);
      std::fclose(out);
    }
    std::fclose(in);
    if (use_atomic) {
      // Atomic overwrite: rename .tmp to final output
      std::error_code ec_rename; fs::rename(tmp, opt.output, ec_rename);
      if (ec_rename) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(opt.output, std::ios::binary | std::ios::trunc);
        if (!src || !dst) die("failed to finalize output file");
        dst << src.rdbuf(); src.close(); dst.close(); fs::remove(tmp);
      }
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
    std::fclose(in);
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
      "%s : %5.2f%% (%s => %s, %s) @ %s/s",
      in_name.c_str(), ratio_pct, in_s, out_s, out_name.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }

  // Decompression summary (printed after fsync so throughput includes flush time)
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
      "%s : %s => %s, %s @ %s/s",
      in_name.c_str(), in_s, out_s, out_name.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }

  vlog(V_VERBOSE, opt, "done.\n");

  } // end for (file_idx)

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

/* parse_args at end */
static Options parse_args(int argc, char ** argv)
{
  Options opt;

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

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { print_help(); std::exit(EXIT_OK); }
    else if (a == "-V" || a == "--version") { print_version(); std::exit(EXIT_OK); }
    else if (a == "-d") opt.mode = Mode::DECOMPRESS;
    else if (a == "-t") opt.mode = Mode::TEST;
    else if (a == "-k") opt.keep = true;
    else if (a == "--rm") { opt.remove_input = true; opt.keep = false; }
    else if (a == "-f") opt.force = true;
    else if (a == "--sparse") opt.sparse_mode = 1;
    else if (a == "--no-sparse") opt.sparse_mode = 0;
    else if (a == "-c") opt.to_stdout = true;
    else if (a == "-v") opt.verbosity = V_VERBOSE;
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
      if (all_digits) {
        int lvl = std::stoi(a.substr(1));
        if (lvl < 1) die_usage("invalid compression level (must be 1..22)");
        if (lvl >= 20 && lvl <= 22 && !opt.ultra) { die_usage("levels 20..22 require --ultra (zstd-compatible behavior)"); }
        if (lvl > 22) die_usage("invalid compression level (max 22)");
        opt.level = lvl; opt.level_user_set = true; continue;
      }
    }
    else if (a == "--fast") { opt.fast_flag = true; opt.level = 1; opt.level_user_set = true; }
    else if (a == "--best") { opt.best_flag = true; opt.level = 19; opt.level_user_set = true; }
    else if (a == "--ultra") { opt.ultra = true; }
    else if (a == "--cpu-only") opt.cpu_only = true;
    else if (a == "--hybrid") opt.hybrid = true;
    else if (a.rfind("-T", 0) == 0 && a.size() > 2) {
      std::string val = a.substr(2);
      if (!val.empty() && val[0] == '=') val = val.substr(1);
      if (val.empty()) die_usage("missing value for -T");
      int th = std::stoi(val);
      // -T0 means "use all available threads" (like zstd)
      opt.cpu_threads = (th == 0) ? -1 : th;
    }
    else if (a == "-T" || a == "--threads") {
      int th = 0;
      if (a == "-T" && i + 1 < argc)
        th = std::stoi(argv[++i]);
      else if (!parse_int_arg("threads", i, argc, argv, th))
        die_usage("missing value for --threads");
      opt.cpu_threads = (th == 0) ? -1 : th;
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
    else if (parse_double_arg("cpu-share", i, argc, argv, opt.cpu_share)) {}
    else if (parse_str_arg("stats-json", i, argc, argv, opt.stats_json)) {}
    else if (parse_num_arg("chunk-size", i, argc, argv, opt.chunk_mib, &opt.chunk_user_set)) {}
    else if (parse_num_arg("cpu-backlog", i, argc, argv, opt.cpu_backlog, nullptr)) {}
    else if (parse_num_arg("cpu-batch", i, argc, argv, opt.cpu_batch, nullptr)) {}
#ifdef HAVE_NVCOMP
    else if (a == "--gpu-only") opt.gpu_only = true;
    else if (parse_num_arg("gpu-batch", i, argc, argv, opt.gpu_batch_cap)) { opt.gpu_batch_user_set = true; }
    else if (parse_double_arg("gpu-mem-frac", i, argc, argv, opt.gpu_mem_fraction)) {}
    else if (parse_num_arg("gpu-streams", i, argc, argv, opt.gpu_streams)) {}
    else if (parse_int_arg("gpu-devices", i, argc, argv, opt.gpu_devices)) {}
    else if (a == "--no-pinned") { opt.pin_mode = PinMode::OFF; }
    else if (a.rfind("--pinned", 0) == 0) {
      std::string v;
      if (parse_str_arg("pinned", i, argc, argv, v) || a.find('=') != std::string::npos) {
        if (a.find('=') != std::string::npos) v = a.substr(a.find('=') + 1);
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v == "auto") opt.pin_mode = PinMode::AUTO;
        else if (v == "on") opt.pin_mode = PinMode::ON;
        else if (v == "off") opt.pin_mode = PinMode::OFF;
        else die_usage("invalid value for --pinned (expected auto|on|off)");
      }
    }
#else
    else if (a.rfind("--gpu-", 0) == 0 || a == "--gpu-only") {
      // Ignored on CPU-only builds
    }
#endif
    else { opt.inputs.push_back(a); }
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

#ifdef HAVE_NVCOMP
  if (opt.gpu_batch_cap == 0) opt.gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  // Decompress benefits massively from large batches  nvCOMP kernel launch
  // Decompress kernel has significant per-launch overhead (Huffman table setup,
  // Decompress kernel has significant per-launch overhead (Huffman table setup,
  // scratch allocation).  But very large batches delay the writer since it
  // can't start until the batch finishes.  Auto-tune for ~4 batches total
  // to balance kernel efficiency with GPU-writer overlap.
  if (!opt.gpu_batch_user_set && (opt.mode == Mode::DECOMPRESS || opt.mode == Mode::TEST)) {
    uint64_t input_size = 0;
    if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
      input_size = (uint64_t)fs::file_size(opt.input);
    if (input_size > 0) {
      size_t chunk = std::max<size_t>(1, opt.chunk_mib) * ONE_MIB;
      size_t est_frames = std::max<size_t>(1, (size_t)(input_size / chunk) + 1);
      size_t target_batches = 4;
      size_t auto_batch = std::max<size_t>(16, est_frames / target_batches);
      auto_batch = std::min(auto_batch, (size_t)512);  // auto caps at 512; user --gpu-batch can go to 1024
      opt.gpu_batch_cap = auto_batch;
    } else {
      opt.gpu_batch_cap = 64;  // default for pipes/unknown size
    }
  }
  if (opt.gpu_streams == 0) opt.gpu_streams = DEFAULT_GPU_STREAMS;
  if (!(opt.gpu_mem_fraction > 0.0 && opt.gpu_mem_fraction < 1.0)) opt.gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  if (opt.gpu_mem_fraction < 0.10) opt.gpu_mem_fraction = 0.10;
  if (opt.gpu_mem_fraction > 0.95) opt.gpu_mem_fraction = 0.95;
#endif
  if (opt.chunk_mib == 0) opt.chunk_mib = DEFAULT_CHUNK_MIB;
  if (opt.mode == Mode::TEST) { opt.to_stdout = true; opt.output = "stdout"; }
  else if (opt.to_stdout && opt.output.empty()) {
    opt.output = "stdout";
  }
  else if (!opt.to_stdout && opt.output.empty() && opt.inputs.size() == 1) {
    // Single file, no -o: auto-derive output name
    opt.output = derive_output(opt.input, opt.mode);
  }
  // Multi-file with no -o: output is derived per-file in main loop
  if (opt.gpu_only && (opt.cpu_only || opt.hybrid)) die_usage("--gpu-only cannot be combined with --cpu-only or --hybrid");
  if (opt.cpu_only && opt.hybrid) die_usage("--cpu-only cannot be combined with --hybrid");

#ifdef HAVE_NVCOMP
  // Default to hybrid mode when nvCOMP is available, unless the user
  // explicitly chose --gpu-only or --cpu-only.  This ensures CPU compression
  // levels (-1..-19) always have an effect and CPUs can work during GPU init.
  if (!opt.cpu_only && !opt.gpu_only && !opt.hybrid) {
    opt.hybrid = true;
  }
#endif

  // Auto-lower verbosity when used as a pipe (both stdin and stdout are non-TTY)
  // but only if the user hasn't explicitly set verbosity via flags.
  // V_DEFAULT stays V_DEFAULT (keeps progress on a TTY stderr), but if stderr
  // is also not a TTY we let progress_loop decide (it checks is_stderr_tty).
  // We do NOT auto-quiet here; the progress_loop already handles the TTY check.

  // Sync global verbosity for die() (which has no access to Options)
  g_verbosity = opt.verbosity;

  return opt;
}
