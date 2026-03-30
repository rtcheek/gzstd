// gzstd.cpp  Hybrid CPU+GPU Zstd (adaptive share)
static constexpr const char * GZSTD_VERSION = "0.11.31";
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
#include <functional>
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
static const size_t DEFAULT_CHUNK_MIB = 16;
#ifdef HAVE_NVCOMP
static const size_t GPU_SUBCHUNK_MAX = size_t(16) * ONE_MIB; // max GPU subchunk
static const size_t DEFAULT_GPU_BATCH_CAP = 8;    // per device  smaller batches launch sooner
static const size_t DEFAULT_GPU_DECOMP_BATCH_CAP = 16;  // sweet spot: amortizes kernel launch without starving writer
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
  size_t cpu_queue_min = 0;   // min queue depth before CPU workers activate (0=no threshold)
#ifdef HAVE_NVCOMP
  size_t gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  bool gpu_batch_user_set = false;
  double gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  size_t gpu_streams = 0;            // 0=auto (1 for compress, 2 for test/verify)
  int gpu_devices = 0;            // 0=auto (all for compress, 1 for decompress)
  PinMode pin_mode = PinMode::AUTO;
#endif
  std::string stats_json;
  int sparse_mode = -1;           // -1=auto (file:on, stdout:off), 0=off, 1=on
  bool sync_output = false;       // --sync-output: fsync before closing output file
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
static constexpr int V_NORMAL  = 0;  // always shown (no -v needed)
static constexpr int V_VERBOSE = 3;  // -v:  informational messages
static constexpr int V_DEBUG   = 4;  // -vv: per-worker/batch detail
static constexpr int V_TRACE   = 5;  // -vvv: per-chunk debug trace

static constexpr int EXIT_OK       = 0;  // success
static constexpr int EXIT_ERROR    = 1;  // general runtime error (catch-all)
static constexpr int EXIT_USAGE    = 2;  // bad command-line usage
static constexpr int EXIT_IO       = 3;  // I/O error (disk full, read failure, permissions)
static constexpr int EXIT_DATA     = 4;  // data/compression error (corrupt input, integrity failure)
static constexpr int EXIT_GPU_FAIL = 5;  // all GPUs failed (VRAM exhaustion, driver error)

// Global verbosity for die() which doesn't take Options
static int g_verbosity = V_DEFAULT;

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
static void vlog(int min_level, const Options & opt, const std::string & msg)
{
  if (opt.verbosity >= min_level) {
    if (g_progress_active.load(std::memory_order_relaxed)) {
      std::fprintf(stderr, "\r\033[K");  // clear progress line
    }
    std::cerr << msg;
  }
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
" --sync-output Fsync output file before exit (default: off; OS flushes in background)\n"
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
" --chunk-size N Host I/O chunk size in MiB (default: 16)\n"
" --stats-json <f> Write run statistics (JSON)\n"
" --cpu-only Force CPU-only path (multithreaded, no GPU)\n"
" --hybrid Enable hybrid CPU+GPU scheduling (default with GPU)\n"
" -T, --threads N  CPU worker threads (0=all cores, auto=96 max). -T N or -T# [CPU-only/hybrid]\n"
" --cpu-batch N    Queue depth before CPUs start popping (hybrid only; keeps frames\n"
"                  stocked for GPU batch fills). Ignored in --cpu-only. Default: 0\n"
" --cpu-share X Fixed CPU share [0..1], disables adaptation (hybrid)\n"
" --cpu-backlog N Secondary queue threshold for CPU workers (hybrid; 0=off)\n";
#ifdef HAVE_NVCOMP
  std::cout <<
" --gpu-batch N Max GPU subchunks per device (default: 16)\n"
" --gpu-mem-frac X Fraction of free VRAM per device (0.1..0.95, def: 0.60)\n"
" --gpu-streams N CUDA streams per device (default: 1; 2 for -t verify)\n"
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
"  1  Runtime error (out of memory, internal failure)\n"
"  2  Bad command-line usage\n"
"  3  I/O error (disk full, read failure, permissions)\n"
"  4  Data error (corrupt input, integrity check failure)\n"
"  5  All GPUs failed (VRAM exhaustion, driver error)\n"
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
    if (v.empty()) die_usage("missing value for " + pref);
    out = std::stoull(v);
    if (was_set) *was_set = true;
    return true;
  }
  // Form: --name VALUE (next argv element)
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
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
    if (v.empty()) die_usage("missing value for " + pref);
    out = std::stoi(v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
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
    out = std::stod(v);
    return true;
  }
  else if (a == pref) {
    if (i + 1 >= argc) die_usage("missing value for " + pref);
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
  std::atomic< uint64_t > total_out  { 0 };  // expected total output bytes (set when known)
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
static void progress_emit_line(double pct, const char * in_s, const char * out_s,
                               const char * in_rate_s, const char * out_rate_s)
{
  char line[256];
  int n = 0;
  if (pct >= 0.0) { n = std::snprintf(line, sizeof(line), "[%.1f%%] in:%s out:%s | in:%s/s out:%s/s ", pct, in_s, out_s, in_rate_s, out_rate_s); }
  else { n = std::snprintf(line, sizeof(line), " in:%s out:%s | in:%s/s out:%s/s ", in_s, out_s, in_rate_s, out_rate_s); }
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
  g_progress_active.store(true, std::memory_order_relaxed);
  using namespace std::chrono; using namespace std::chrono_literals;
  const bool is_test = (opt.mode == Mode::TEST);
  while (!done_flag->load()) {
    std::this_thread::sleep_for(200ms);
    uint64_t in  = m->read_bytes.load();
    uint64_t out = m->wrote_bytes.load();
    uint64_t t_out = m->total_out.load();
    auto dt      = steady_clock::now() - m->t0;
    double secs  = duration_cast< duration<double> >(dt).count();
    double in_rate = secs>0 ? double(in)/secs : 0.0;
    double out_rate = secs>0 ? double(out)/secs : 0.0;
    char in_s[64], out_s[64], rate_s[64], out_rate_s[64];
    human_bytes(double(in),  in_s,  sizeof(in_s));
    human_bytes(double(out), out_s, sizeof(out_s));
    human_bytes(in_rate,     rate_s, sizeof(rate_s));
    human_bytes(out_rate,    out_rate_s, sizeof(out_rate_s));
    double pct = (total_in > 0) ? (100.0 * double(in) / double(total_in)) : -1.0;

    if (is_test) {
      // Test mode: show verified bytes and decompression throughput.
      // out = decompressed bytes verified (no disk I/O).
      char line[256];
      if (pct >= 0.0)
        std::snprintf(line, sizeof(line), "[%.1f%%] in:%s verified:%s @ %s/s ", pct, in_s, out_s, out_rate_s);
      else
        std::snprintf(line, sizeof(line), " in:%s verified:%s @ %s/s ", in_s, out_s, out_rate_s);
      std::fprintf(stderr, "\r%s\033[K", line);
      std::fflush(stderr);
    } else if (total_in > 0 && in >= total_in && out > 0) {
      // All input consumed: switch to showing write drain progress
      char wr_s[64];
      human_bytes(out_rate, wr_s, sizeof(wr_s));
      // Show write drain percentage if we know the expected output size
      if (t_out > 0 && out < t_out) {
        // AIO still writing  show progress
        double write_pct = std::min(99.9, 100.0 * double(out) / double(t_out));
        char line[256];
        std::snprintf(line, sizeof(line),
            "[%.1f%%] writing: %s / ", write_pct, out_s);
        char tot_s[64];
        human_bytes(double(t_out), tot_s, sizeof(tot_s));
        std::fprintf(stderr, "\r%s%s @ %s/s\033[K", line, tot_s, wr_s);
        std::fflush(stderr);
      } else if (t_out > 0) {
        // AIO finished but file not yet closed/fsynced  show flushing
        char tot_s[64];
        human_bytes(double(t_out), tot_s, sizeof(tot_s));
        std::fprintf(stderr, "\r[done] flushing %s to disk... (%.0fs)\033[K", tot_s, secs);
        std::fflush(stderr);
      } else {
        progress_emit_line(pct, in_s, out_s, rate_s, out_rate_s);
      }
    } else {
      progress_emit_line(pct, in_s, out_s, rate_s, out_rate_s);
    }
  }
  // Final sample (no newline  the completion summary will overwrite this line)
  uint64_t in  = m->read_bytes.load();
  uint64_t out = m->wrote_bytes.load();
  auto dt      = std::chrono::steady_clock::now() - m->t0;
  double secs  = std::chrono::duration_cast< std::chrono::duration<double> >(dt).count();
  double in_rate = secs>0 ? double(in)/secs : 0.0;
  double out_rate = secs>0 ? double(out)/secs : 0.0;
  char in_s[64], out_s[64], rate_s[64], out_rate_s[64];
  human_bytes(double(in),  in_s,  sizeof(in_s));
  human_bytes(double(out), out_s, sizeof(out_s));
  human_bytes(in_rate,     rate_s, sizeof(rate_s));
  human_bytes(out_rate,    out_rate_s, sizeof(out_rate_s));
  double pct = (total_in > 0) ? (100.0 * double(in) / double(total_in)) : -1.0;
  progress_emit_line(pct, in_s, out_s, rate_s, out_rate_s);
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
    size_t per_thread = chunk_bytes + chunk_bytes + (chunk_bytes >> 8) + 4096;
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

    if (h2d_count > 0 || sched_gpu_tasks > 0) {
      fprintf(stderr, "  H2D transfers:    %8.3f s  (%6.2f GiB, %5.2f GiB/s) [%llu] \n",
              ns_to_s(h2d_ns), bytes_to_gib(h2d_bytes),
              rate_gibs(h2d_bytes, h2d_ns),
              (unsigned long long)h2d_count.load());
    }
    if (kernel_count > 0 || sched_gpu_tasks > 0) {
      fprintf(stderr, "  GPU kernel:       %8.3f s  (%llu batches, %5.1f ms/batch)   \n",
              ns_to_s(kernel_ns),
              (unsigned long long)kernel_count.load(),
              (kernel_count > 0) ? ns_to_ms(kernel_ns) / kernel_count : 0.0);
    }
    if (d2h_count > 0 || sched_gpu_tasks > 0) {
      fprintf(stderr, "  D2H transfers:    %8.3f s  (%6.2f GiB, %5.2f GiB/s) [%llu] \n",
              ns_to_s(d2h_ns), bytes_to_gib(d2h_bytes),
              rate_gibs(d2h_bytes, d2h_ns),
              (unsigned long long)d2h_count.load());
    }
    if (gpu_batch_count > 0 || sched_gpu_tasks > 0) {
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
    fprintf(stderr, "  Writer wait:      %8.3f s  (%llu waits)                    \n",
            ns_to_s(writer_wait_ns), (unsigned long long)writer_wait_count.load());
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

class TaskQueue {
public:
  void push(Task && t)
  {
    std::unique_lock<std::mutex> lk(m_);
    q_.push_back(std::move(t));
    ++total_tasks_;
    cv_.notify_all();      // wake all GPU workers waiting in pop_batch_greedy
    cpu_cv_.notify_one();  // wake one CPU worker waiting in pop_one_cpu
  }

  // Re-enqueue tasks that were popped but never processed (e.g., GPU VRAM failure).
  // Does NOT increment total_tasks_ since these were already counted on first push.
  void re_enqueue(std::vector<Task> & batch)
  {
    if (batch.empty()) return;
    std::unique_lock<std::mutex> lk(m_);
    for (auto & t : batch)
      q_.push_back(std::move(t));
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
      out.push_back(std::move(q_.front()));
      q_.pop_front();
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
      out.push_back(std::move(q_.front()));
      q_.pop_front();
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
    return double(front.data.size()) / double(front.decomp_size);
  }

  // Signal that no more tasks will be pushed (producer is finished).
  void set_done()
  {
    std::unique_lock<std::mutex> lk(m_);
    done_ = true;
    cv_.notify_all();
    cpu_cv_.notify_all();
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

  // Wake all CPU workers blocking in wait_for_cpu().
  // Called externally when scheduling conditions change (e.g., GPU releases
  // the semaphore via gpu_got_data, or the queue is drained).
  void notify_cpu_waiters()
  {
    cpu_cv_.notify_all();
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
          t = std::move(q_.front());
          q_.pop_front();
          return true;
        }
      }
      cpu_cv_.wait(lk);
    }
  }

private:
  // Peek at front task's compression ratio.  CALLER MUST HOLD m_.
  double front_ratio_locked() const {
    if (q_.empty()) return -1.0;
    const Task & front = q_.front();
    if (front.decomp_size == 0) return -1.0;
    return double(front.data.size()) / double(front.decomp_size);
  }

  std::mutex              m_;
  std::condition_variable cv_;
  std::condition_variable cpu_cv_;   // dedicated CV for CPU workers (avoids spurious wakes from GPU pops)
  std::deque<Task>        q_;
  bool                    done_ = false;
  std::atomic<size_t>     total_tasks_{0};
};

// Sequential frame dispatcher: assigns contiguous frame ranges to GPU workers
// in round-robin order, so results arrive at the writer in long sequential runs.
// GPU 0 gets batch 0, GPU 1 gets batch 1, ..., then GPU 0 gets batch N, etc.
// This minimizes out-of-order delivery to the writer thread.
//
// Usage: each GPU worker calls pop_my_batch(my_slot) which blocks until
// it's that slot's turn, then pops a contiguous batch from the queue.
class SequentialDispatcher {
public:
  explicit SequentialDispatcher(int num_slots)
    : num_slots_(num_slots), next_slot_(0) {}

  // Block until it's this slot's turn, then pop a batch from the queue.
  // Returns false when queue is drained.
  bool pop_my_batch(int slot, size_t batch_size, TaskQueue & queue,
                    std::vector<Task> & out)
  {
    // Wait for our turn
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&]{ return slot == next_slot_ || queue.drained(); });
    if (queue.drained() && queue.size() == 0) return false;
    lk.unlock();

    // Pop from queue (we have the turn  other GPUs are waiting)
    bool ok = queue.pop_batch_greedy(batch_size, out);

    // Advance to next slot
    {
      std::lock_guard<std::mutex> lk2(mtx_);
      next_slot_ = (next_slot_ + 1) % num_slots_;
    }
    cv_.notify_all();
    return ok;
  }

  // Wake all waiters (called when queue is drained)
  void notify_done() { cv_.notify_all(); }

private:
  int num_slots_;
  int next_slot_;
  std::mutex mtx_;
  std::condition_variable cv_;
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

  // --- Per-producer result slots (reduces mutex contention) ---
  // Each GPU worker and CPU worker group gets its own slot.
  // Producers push to their slot's local queue (own mutex, no contention).
  // Writer drains all slots into the shared map periodically.
  struct Slot {
    std::mutex slot_m;
    std::vector<std::pair<size_t, std::vector<char>>> pending;  // (seq, data)
  };
  std::vector<std::unique_ptr<Slot>> slots;

  // Create N slots (call before launching workers)
  void init_slots(int n) {
    slots.resize(n);
    for (int i = 0; i < n; ++i)
      slots[i] = std::make_unique<Slot>();
  }

  // Producer: push a result to a specific slot (low contention  one producer per slot)
  void push_to_slot(int slot_id, size_t seq, std::vector<char> && frame) {
    if (slot_id >= 0 && slot_id < (int)slots.size()) {
      std::lock_guard<std::mutex> lk(slots[slot_id]->slot_m);
      slots[slot_id]->pending.emplace_back(seq, std::move(frame));
      // GPU path: no per-frame notify. Batch-completion notify handles it.
    } else {
      // CPU fallback: push to shared map and notify (CPU frames are infrequent)
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
 Writer backpressure  prevents CPU workers from flooding the writer
 -----------------------------------------------------------------------
 Tracks the gap between bytes produced (decompressed) and bytes physically
 written to disk.  CPU workers block on a CV when the gap exceeds a
 high-water mark, and wake instantly when it drops below a low-water mark.

 GPU workers are NEVER throttled  their batches are already in-flight
 on the GPU and can't be stopped mid-kernel.  Only CPU workers check
 backpressure before popping a new task.

 The AIO worker calls mark_written() after each physical write, which
 wakes blocked CPU workers if the backlog has dropped enough.

 High/low water marks use a hysteresis band to prevent oscillation:
   - CPU blocks when backlog > high_water (e.g., 4 GiB)
   - CPU wakes when backlog < low_water (e.g., 2 GiB)
 This keeps 2-4 GiB of decompressed data buffered  enough to keep the
 writer continuously fed without exhausting RAM on huge files.
======================================================================*/
class WriterBackpressure {
public:
  // Default: 4 GiB high / 2 GiB low.  Tuned for NVMe at ~2 GiB/s:
  // 2 GiB low-water = ~1 second of writer runway before starvation.
  explicit WriterBackpressure(uint64_t high = 4ULL * 1024 * 1024 * 1024,
                              uint64_t low  = 2ULL * 1024 * 1024 * 1024)
    : high_water_(high), low_water_(low) {}

  // Called by workers after producing a decompressed frame.
  // Lightweight atomic add  no lock, no CV interaction.
  void mark_produced(uint64_t bytes) {
    produced_.fetch_add(bytes, std::memory_order_relaxed);
  }

  // Called by AIO worker after physically writing data to disk.
  // If backlog drops below low-water, wakes all blocked CPU workers.
  void mark_written(uint64_t bytes) {
    written_.fetch_add(bytes, std::memory_order_release);
    uint64_t backlog = produced_.load(std::memory_order_relaxed)
                     - written_.load(std::memory_order_acquire);
    if (backlog <= low_water_) {
      std::lock_guard<std::mutex> lk(m_);
      cv_.notify_all();
    }
  }

  // Called by CPU workers before popping a new task.
  // Blocks if backlog exceeds high-water mark.  Returns immediately
  // if backlog is acceptable or if done flag is set.
  void wait_if_backlogged() {
    uint64_t backlog = produced_.load(std::memory_order_relaxed)
                     - written_.load(std::memory_order_acquire);
    if (backlog <= high_water_) return;  // fast path: no contention

    // Slow path: block until writer catches up
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [this] {
      uint64_t bl = produced_.load(std::memory_order_relaxed)
                  - written_.load(std::memory_order_acquire);
      return bl <= low_water_ || done_.load(std::memory_order_relaxed);
    });
  }

  // Signal that all work is complete  wake any blocked workers so they can exit.
  void set_done() {
    done_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(m_);
    cv_.notify_all();
  }

  uint64_t backlog() const {
    return produced_.load(std::memory_order_relaxed)
         - written_.load(std::memory_order_acquire);
  }

private:
  std::atomic<uint64_t> produced_{0};
  std::atomic<uint64_t> written_{0};
  uint64_t high_water_;
  uint64_t low_water_;
  std::atomic<bool> done_{false};
  std::mutex m_;
  std::condition_variable cv_;
};

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
                          Meter * meter = nullptr, WriterBackpressure * bp = nullptr)
    : out_(out_file), dw_(dw), sparse_(sparse), done_(false), meter_(meter), bp_(bp)
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
        // Update wrote_bytes after physical write (not after submit)
        // so the progress bar reflects actual disk I/O completion.
        if (meter_) meter_->wrote_bytes.fetch_add(buf.size(), std::memory_order_relaxed);
        // Notify backpressure: writer made progress, CPU workers may unblock.
        if (bp_) bp_->mark_written(buf.size());
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
  Meter * meter_ = nullptr;     // for tracking physically-written bytes
  WriterBackpressure * bp_ = nullptr;  // for throttling CPU workers
};

static void writer_thread(FILE * out, ResultStore & results,
                          const Options & opt, Meter * m,
                          WriterBackpressure * bp)
{
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  // pin_thread_to_core(1);              // disabled: hurts on loaded machines
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
    aio_ptr = std::make_unique<AsyncWritePool>(out, g_direct_writer, enable_sparse, m, bp);
    aio = aio_ptr.get();
#else
    aio_ptr = std::make_unique<AsyncWritePool>(out, nullptr, enable_sparse, m, bp);
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
    while (results.data.count(results.next_to_write) == 0 && !all_done) {
      results.drain_slots_locked();
      if (results.data.count(results.next_to_write) != 0) break;
      waited = true;
      // Use timed wait to detect potential deadlocks: if all workers are done
      // but the next expected frame never arrives, something is wrong.
      if (results.workers_done) {
        results.cv.wait_for(lk, std::chrono::seconds(5));
        results.drain_slots_locked();
        all_done = results.producer_done
                && results.workers_done
                && results.next_to_write >= results.total_tasks;
        if (!all_done && results.data.count(results.next_to_write) == 0) {
          // Workers are all done but we're still missing frames  this is a bug.
          // Log diagnostics and abort to avoid producing corrupt output.
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
    if (aio) {
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
      vlog(V_VERBOSE, opt, "writer: draining write queue...\n");
    aio->flush();
    if (aio->had_error()) die_io("async write failed (disk full?)");
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
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));

  size_t bound = ZSTD_compressBound(src_size);
  out.resize(bound);
  size_t csz = ZSTD_compress2(tl_cctx, out.data(), out.size(), src, src_size);
  if (ZSTD_isError(csz)) die_data(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
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

  size_t ret = 0;
  while (zin.pos < zin.size) {
    ret = ZSTD_decompressStream(dctx, &zout, &zin);
    if (ZSTD_isError(ret))
      die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret));
    if (zout.pos > 0) {
      if (opt.mode != Mode::TEST) {
#ifndef _WIN32
        if (g_direct_writer) {
          if (!g_direct_writer->write(outbuf.data(), zout.pos))
            die_io("direct write failed (disk full?)");
        } else
#endif
        {
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

 QUEUE INTERACTION: GPU workers use blocking pop_batch_greedy (waits for
 full batch).  CPU workers use non-blocking try_pop_batch to avoid
 competing with GPU for the queue's condition variable wakeups.
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

  // Set the task queue pointer so gpu_got_data() can wake CPU workers.
  // Must be called before GPU workers start.
  void set_queue(TaskQueue * tq) { queue_ = tq; }

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

  // GPU stream calls this after it has taken a batch from the queue.
  // Decrements the semaphore and wakes CPU workers so they can
  // immediately check should_cpu_take() instead of sleeping.
  void gpu_got_data() {
    gpus_waiting_.fetch_sub(1, std::memory_order_release);
    if (queue_) queue_->notify_cpu_waiters();
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
  TaskQueue * queue_ = nullptr;  // for waking CPU workers from gpu_got_data()

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

  // CPU workers call this: should I take a frame right now?
  bool cpu_may_take() {
    int taken = cpu_frames_taken.load(std::memory_order_relaxed);
    int allowed = cpu_frame_allowance.load(std::memory_order_relaxed);
    if (taken < allowed) {
      cpu_frames_taken.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    return false;
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
  WriterBackpressure * bp)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  while (true) {
    // Backpressure: block if writer can't keep up (prevents RAM exhaustion)
    if (bp) bp->wait_if_backlogged();

    Task t;
#ifdef HAVE_NVCOMP
    if (sched) {
      // Hybrid mode: block on queue CV until scheduler allows CPU to take
      // and queue depth meets threshold.  Wakes instantly when:
      //   - producer pushes a task (push -> cpu_cv_.notify)
      //   - GPU releases semaphore (gpu_got_data -> notify_cpu_waiters)
      //   - producer is done (set_done -> cpu_cv_.notify_all)
      auto may_take = [&](const TaskQueue::QueueState & qs) -> bool {
        if (!sched->should_cpu_take()) return false;
        if (opt->cpu_queue_min > 0 && qs.depth < opt->cpu_queue_min
            && !qs.done) return false;
        return true;
      };
      if (!tq->pop_one_cpu(t, may_take)) break;
      sched->mark_cpu_take(1);
    } else
#endif
    {
      // Non-hybrid (--cpu-only): block on queue CV with threshold predicate
      if (opt->cpu_queue_min > 0) {
        auto threshold_met = [&](const TaskQueue::QueueState & qs) -> bool {
          return qs.depth >= opt->cpu_queue_min || qs.done;
        };
        if (!tq->wait_for_cpu(threshold_met)) break;
        if (tq->drained() && tq->size() == 0) break;
      }
      if (!tq->pop_one(t)) break;
    }

    // cpu_backlog check: if the queue is below the backlog threshold,
    // re-enqueue and wait.  This is a secondary throttle separate from
    // cpu_queue_min (which gates the initial pop).
    // Note: cpu_backlog is rarely used; cpu_queue_min is the primary mechanism.
    if (opt->cpu_backlog > 0 && tq->size() < opt->cpu_backlog && !tq->drained()) {
      // Put it back and wait for more frames
      tq->push(std::move(t));
      auto backlog_met = [&](const TaskQueue::QueueState & qs) -> bool {
        return qs.depth >= opt->cpu_backlog || qs.done;
      };
      if (!tq->wait_for_cpu(backlog_met)) break;
      continue;  // re-enter the loop to pop properly
    }

    {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> out_frame;
    compress_one_cpu_frame(t.data.data(), t.data.size(), opt->level, out_frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast< std::chrono::duration<double, std::milli> >(t1 - t0).count();
    const size_t csz = out_frame.size();
    const size_t in_size = t.data.size();  // save before releasing
    { std::vector<char>().swap(t.data); }  // release input immediately
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

    if (m) m->tasks_done.fetch_add(1);
    results->push_to_slot(-1, t.seq, std::move(out_frame));
    if (bp) bp->mark_produced(csz);  // track output for writer backpressure
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
  CpuAgg * cpuagg,
  WriterBackpressure * bp)
{
  while (true) {
    if (bp) bp->wait_if_backlogged();
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
    results->push_to_slot(-1, t.seq, std::move(out_frame));
    if (bp) bp->mark_produced(csz);

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
  RateMatchState * rate_match,
#endif
  CpuAgg * cpuagg,
  WriterBackpressure * bp)
{
#ifdef HAVE_NVCOMP
  HybridSched * sched = static_cast<HybridSched*>(sched_ptr);
#endif
  // Create a reusable decompression context for this thread
  if (!tl_dctx) {
    tl_dctx = ZSTD_createDCtx();
    if (!tl_dctx) die("failed to create ZSTD_DCtx");
  }

  while (true) {
    // Writer backpressure: block if the writer is overwhelmed.
    // This prevents CPU workers from producing decompressed data faster
    // than the NVMe can write, which would exhaust RAM and cause massive
    // kernel writeback pressure (the sys time problem).
    // GPUs are never throttled  their batches are already in-flight.
    if (bp) bp->wait_if_backlogged();

    Task t;
#ifdef HAVE_NVCOMP
    if (sched) {
      // Hybrid decompress: block on queue CV until scheduler allows CPU to take
      // OR a trivially-compressed frame (ratio < 2%) is at the front of the queue.
      // Trivial frames are faster on CPU (no PCIe D2H overhead), so we bypass
      // the GPU-priority semaphore for them.
      bool got_trivial = false;
      auto may_take = [&](const TaskQueue::QueueState & qs) -> bool {
        // Always allow trivially-compressed frames regardless of scheduler
        if (qs.front_ratio >= 0.0 && qs.front_ratio < 0.02) { got_trivial = true; return true; }
        got_trivial = false;
        // Normal scheduling: respect GPU priority and queue depth threshold
        if (!sched->should_cpu_take()) return false;
        if (opt->cpu_queue_min > 0 && qs.depth < opt->cpu_queue_min
            && !qs.done) return false;
        return true;
      };
      if (!tq->pop_one_cpu(t, may_take)) break;
      sched->mark_cpu_take(1);
      if (got_trivial && opt->verbosity >= V_DEBUG) {
        double ratio = (t.decomp_size > 0)
                       ? double(t.data.size()) / double(t.decomp_size) : 0.0;
        std::ostringstream os;
        os << "[CPU-D/T" << worker_id << "] trivial frame (ratio="
           << std::fixed << std::setprecision(3) << (ratio * 100.0) << "%)";
        vlog(V_DEBUG, *opt, os.str() + "\n");
      }
    } else
#endif
    {
      // Non-hybrid (--cpu-only): block on queue CV with threshold predicate
      if (opt->cpu_queue_min > 0) {
        auto threshold_met = [&](const TaskQueue::QueueState & qs) -> bool {
          return qs.depth >= opt->cpu_queue_min || qs.done;
        };
        if (!tq->wait_for_cpu(threshold_met)) break;
        if (tq->drained() && tq->size() == 0) break;
      }
      if (!tq->pop_one(t)) break;
    }
    {
    const auto t0_w = std::chrono::steady_clock::now();

    std::vector<char> out_buf(t.decomp_size);
    const size_t comp_size = t.data.size();
    size_t actual = ZSTD_decompressDCtx(tl_dctx, out_buf.data(), out_buf.size(),
                                        t.data.data(), t.data.size());
    if (ZSTD_isError(actual))
      die_data(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(actual));
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
      os << "[CPU-D/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_DEBUG, *opt, os.str() + "\n");
    }

    if (m) m->read_bytes.fetch_add(comp_size);

    results->push_to_slot(-1, t.seq, std::move(out_buf));
    if (bp) bp->mark_produced(actual);  // track backpressure for writer throttling

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
    } // end single-frame processing block
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
    Meter * m,
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
        // Parsed some frames OK; trailing bytes cannot form a valid frame.
        // A few trailing null bytes could be padding; anything substantial
        // indicates truncation or corruption.
        if (remaining > 8) {
          // Significant trailing data that isn't a valid frame = truncated
          *fallback = false;  // not a format problem, it's a data problem
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
  WriterBackpressure backpressure;
  // Disable backpressure in test mode: no disk I/O, so mark_written() is never
  // called and CPU workers would block at the high-water mark forever.
  WriterBackpressure * bp_ptr = (opt.mode == Mode::TEST) ? nullptr : &backpressure;

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
    backpressure.set_done();
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
    backpressure.set_done();
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
  backpressure.set_done();  // wake any CPU workers blocked on backpressure
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
  // Chunk size: 16 MiB default, or user-specified via --chunk-size
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;
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
  if (ZSTD_isError(st)) die_data(std::string("ZSTD_CCtx_setParameter(level) error: ") + ZSTD_getErrorName(st));

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
  // Chunk size: 16 MiB default, or user-specified via --chunk-size
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;
  // Pre-flight: check we won't OOM  may reduce chunk size
  chosen_mib = check_ram_budget(threads, chosen_mib, opt);
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
  WriterBackpressure backpressure;
  std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m, &backpressure);

  std::vector<std::thread> pool; pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) pool.emplace_back(cpu_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
    nullptr, nullptr,
#endif
    &cpuagg, &backpressure);
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
  backpressure.set_done();  // wake any backlogged workers so they can exit
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

  // Throughput-aware auto-tuning: continuously searches for the optimal
  // batch size using a hill-climbing/binary-search approach.
  //
  // States:
  //   EXPLORE: Try a new batch size, measure throughput
  //   SETTLE:  Found a good size, use it.  Periodically probe to detect
  //            changes in data characteristics (e.g., mixed tar archives).
  //
  // The tuner tracks throughput (GiB/s) at each tested batch size and
  // binary-searches between known-good and known-bad sizes.
  enum class TuneState { EXPLORE, REFINE, SETTLE };
  TuneState   tune_state = TuneState::EXPLORE;
  size_t      tune_lo = 0;             // lower bound of search range
  size_t      tune_hi = 0;             // upper bound (set to per_stream_cap)
  size_t      tune_best_batch = 0;     // best batch size found so far
  double      tune_best_thr = 0.0;     // throughput at best batch size
  size_t      tune_prev_batch = 0;     // batch size we're measuring now
  double      tune_prev_thr = 0.0;     // throughput at previous batch size
  uint32_t    tune_batches_at_size = 0; // how many batches processed at current size
  uint32_t    tune_settle_count = 0;   // ticks spent in SETTLE state
  static constexpr uint32_t TUNE_MIN_BATCHES = 2;   // min ticks before judging throughput
  static constexpr uint32_t TUNE_PROBE_INTERVAL = 15; // ticks in SETTLE before probing
  bool        tune_tried_down = false;  // have we tried halving yet?
  bool        tune_tried_up = false;    // have we tried doubling yet?
  size_t      refine_lo = 0;            // lower bound for refinement binary search
  size_t      refine_hi = 0;            // upper bound for refinement binary search
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
  // Preserve auto-tune state across buffer reallocation.
  // C = StreamCtx{} would wipe tune_prev_batch, tune_best_thr, etc.
  auto save_state = C.tune_state;
  auto save_lo = C.tune_lo;
  auto save_hi = C.tune_hi;
  auto save_best_batch = C.tune_best_batch;
  auto save_best_thr = C.tune_best_thr;
  auto save_prev_batch = C.tune_prev_batch;
  auto save_prev_thr = C.tune_prev_thr;
  auto save_batches = C.tune_batches_at_size;
  auto save_settle = C.tune_settle_count;
  auto save_tried_down = C.tune_tried_down;
  auto save_tried_up = C.tune_tried_up;
  auto save_refine_lo = C.refine_lo;
  auto save_refine_hi = C.refine_hi;
  auto save_adjust = C.last_adjust;
  auto save_stats = C.stats;
  C = StreamCtx{};
  C.tune_state = save_state;
  C.tune_lo = save_lo;
  C.tune_hi = save_hi;
  C.tune_best_batch = save_best_batch;
  C.tune_best_thr = save_best_thr;
  C.tune_prev_batch = save_prev_batch;
  C.tune_prev_thr = save_prev_thr;
  C.tune_batches_at_size = save_batches;
  C.tune_settle_count = save_settle;
  C.tune_tried_down = save_tried_down;
  C.tune_tried_up = save_tried_up;
  C.refine_lo = save_refine_lo;
  C.refine_hi = save_refine_hi;
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
  std::atomic<bool> * gpu_started_flag,
  SharedTuneState * shared_tune,
  RateMatchState * rate_match,
  WriterBackpressure * bp)
{
  (void)m; std::shared_ptr<std::vector<StreamCtx>> ctxs_ptr;
  void * vram_reserve = nullptr;
  size_t vram_reserve_bytes = 0;
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
      // Calculate how many subchunks this stream can hold based on free VRAM.
      // nvCOMP temp workspace can be very large (e.g., 5 GiB for batch=200),
      // so we use binary search to find the largest batch that fits in VRAM.
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
      int vram_retries = 0;
      while (!allocate_stream_buffers(C, C.per_stream_batch, gpu_chunk, max_out_chunk, comp_opts, opt)) {
        // Free any partial allocations from the failed attempt
        free_stream_buffers_only(C);
        if (C.per_stream_batch <= 1 || ++vram_retries > 10) {
          // Can't fit even batch=1 or too many retries  skip this GPU entirely.
          // Other GPUs (or CPU in hybrid) will handle the work.
          std::string skip_msg = "[GPU" + std::to_string(device_id)
              + "] insufficient VRAM for even batch=1  skipping device"
              + (vram_retries > 10 ? " (retry limit)" : "");
          vlog(V_ERROR, opt, skip_msg + "\n");
          *any_gpu_failed = true;
          *fatal_msg = skip_msg;
          // Clean up streams we already created
          for (size_t cs = 0; cs <= s; ++cs) {
            if (ctxs[cs].stream) { cudaStreamDestroy(ctxs[cs].stream); ctxs[cs].stream = nullptr; }
          }
          // Wake writer in case it's waiting
          { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
          return;
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

    if (sched) sched->set_gpu_ready(device_id);
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
                    os << "[auto-tune] baseline=" << cur_batch << " (" << std::fixed
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
                    os << "[auto-tune] halving worse, will try doubling (best=" << S.best_batch << ")";
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
                        os << "[auto-tune] refining [" << S.refine_lo << ".." << S.refine_hi
                           << "] trying " << mid;
                        vlog(V_VERBOSE, opt, os.str() + "\n");
                      }
                    } else {
                      S.batch_size.store(S.best_batch);
                      S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                      if (opt.verbosity >= V_VERBOSE) {
                        std::ostringstream os;
                        os << "[auto-tune] settled at batch=" << S.best_batch
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
                    os << "[auto-tune] refined, settled at batch=" << S.best_batch
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
                      os << "[auto-tune] probe: " << S.best_batch << " -> " << probe
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
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        // Signal scheduler: GPU stream wants data (blocks CPU workers)
        if (sched) sched->gpu_wants_data();
        // Greedy pop: wait for a full batch (per_stream_batch) or producer done.
        // This maximizes GPU kernel efficiency by processing many chunks per launch.
        // Use shared batch size from auto-tuner (or per-stream if locked)
        size_t pop_n = (shared_tune && !shared_tune->locked.load())
                     ? shared_tune->batch_size.load(std::memory_order_relaxed)
                     : C.per_stream_batch;
        pop_n = std::min(pop_n, C.per_stream_batch);  // can't exceed allocated buffer
        // Apply utilization scaling (updated after each batch completion)
        pop_n = std::max<size_t>(1, (size_t)(pop_n * util_scale));
        // Backpressure: wait if writer is overwhelmed before grabbing more work
        if (bp) bp->wait_if_backlogged();
        if (!queue->pop_batch_greedy(pop_n, C.batch)) {
          if (sched) sched->gpu_got_data();
          if (g_perf) { g_perf->queue_wait_ns.fetch_add(now_ns() - qw_t0); g_perf->queue_wait_count.fetch_add(1); }
          producer_done_seen = true; continue;
        }
        // Signal scheduler: GPU got its data (unblocks CPU workers)
        if (sched) sched->gpu_got_data();
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

        // Release host-side input data  it's on the GPU now.
        // cudaMemcpyAsync with pageable memory is host-synchronous, so
        // the data is fully copied before the function returns.
        // In hybrid mode, keep data alive for potential rescue on GPU failure.
        // In gpu-only mode, no rescue  safe to release immediately.
        if (!rescue) {
          for (size_t i = 0; i < C.filled; ++i)
            { std::vector<char>().swap(C.batch[i].data); }
        }

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
            results->push_to_slot(slot_index, C.batch[i].seq, std::move(h_out));
          }
          // Track GPU compressed output for writer backpressure
          if (bp) bp->mark_produced(out_sum);
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
            uint64_t d2h_t0 = g_perf ? now_ns() : 0;
            for (size_t i = 0; i < C.filled; ++i) {
              const size_t csz = C.h_comp_sizes[i];
              std::vector<char> h_out(csz);
              const void * d_src = static_cast<char*>(C.d_out_base) + i * C.max_out_chunk;
              checkCuda(cudaMemcpy(h_out.data(), d_src, csz, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H exact sync)");
              results->push_to_slot(slot_index, C.batch[i].seq, std::move(h_out));
            }
            if (g_perf) {
              g_perf->d2h_ns.fetch_add(now_ns() - d2h_t0);
            }
          }
          // Track GPU compressed output for writer backpressure (sync path)
          if (bp) bp->mark_produced(out_sum);
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
      if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
      if (auto sp = std::weak_ptr<std::vector<StreamCtx>>(ctxs_ptr).lock()) {
        for (auto & C : *sp) {
          if (C.busy && !C.batch.empty()) {
            if (rescue) {
              // Hybrid mode: push to CPU rescue queue
              for (size_t i = 0; i < C.filled; ++i)
                rescue->push(Task{ C.batch[i].seq, C.batch[i].data });
              C.batch.clear();
            } else {
              // GPU-only mode: push back to main queue for other GPUs
              queue->re_enqueue(C.batch);
            }
          }
          free_stream_buffers_only(C);
          if (C.stream) cudaStreamDestroy(C.stream);
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
        os << "GPUs: " << want << " device" << (want > 1 ? "s" : "") << " active";
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
    os << "GPUs: " << n << " device" << (n > 1 ? "s" : "") << " active";
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
  // 16 MiB default: matches GPU_SUBCHUNK_MAX, no splitting needed.
  size_t chosen_mib = opt.chunk_user_set ? opt.chunk_mib : 16;
  // Pre-flight: check we won't OOM (CPU rescue threads also need buffers)
  int cpu_threads_est = opt.cpu_only ? 0 : resolve_cpu_threads(opt.cpu_threads);
  chosen_mib = check_ram_budget(std::max(1, cpu_threads_est), chosen_mib, opt);
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

  // Writer backpressure: prevents workers from producing compressed data
  // faster than the NVMe can write.  Same mechanism as decompress backpressure.
  WriterBackpressure backpressure;

  // Start progress bar and ordered-writer threads
  std::atomic<bool> progress_done{false};
  std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m, &backpressure);

  // ---- Hybrid scheduler (adaptive CPU/GPU work-sharing) ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, /*cpu_threads*/0, device_count, opt);
    sched = sched_ptr.get();
    sched->set_queue(&queue);
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
      rescue_pool.emplace_back(cpu_worker_rescue, (int)i, &rescue, &results, &opt, m, &cpuagg, &backpressure);

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
  RateMatchState rate_match_compress;
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
      cpu_pool.emplace_back(cpu_worker, i, &queue, &results, &opt, m, (void*)sched, &rate_match_compress, &cpuagg, &backpressure);
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

  for (int i = 0; i < gpu_count; ++i) {
    workers.emplace_back(gpu_worker, gpu_ids[i], i, opt_for_workers,
                         &queue, &rescue, &results,
                         &per_dev[size_t(i)], &json_sink, m, sched,
                         &any_gpu_failed, &abort_on_failure,
                         &fatal_msgs[size_t(i)], &gpu_started,
                         &shared_tune, &rate_match_compress,
                         &backpressure);
  }
  if (opt.verbosity >= V_VERBOSE) {
    std::ostringstream os;
    os << "[GPU] " << gpu_count << " device worker"
       << (gpu_count > 1 ? "s" : "") << " online";
    vlog(V_VERBOSE, opt, os.str() + "\n");
  }

  // ---- Producer: read input, split into GPU-sized subchunks, enqueue ----
  try_boost_io_priority(!opt.gpu_only);  // only boost when CPU pool competes
  // pin_thread_to_core(0);              // disabled: hurts on loaded machines
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

    // Note: we do NOT abort the read loop on single GPU failures.
    // Surviving GPUs continue processing.  The post-join check handles
    // the "all GPUs failed" case after workers are joined.
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
  backpressure.set_done();  // wake any backlogged workers so they can exit
  for (auto & th : workers) th.join();

  // Check for GPU failures
  if (any_gpu_failed.load()) {
    // Count how many GPUs actually failed
    int failed_count = 0;
    for (const auto & s : fatal_msgs)
      if (!s.empty()) ++failed_count;
    int total_gpus = (int)fatal_msgs.size();

    if (abort_on_failure.load() && failed_count >= total_gpus) {
      // ALL GPUs failed in --gpu-only mode  fatal
      backpressure.set_done();  // unblock any waiting workers
      progress_done = true;
      progress_thr.join();
      {
        std::lock_guard<std::mutex> lk(results.m);
        results.workers_done = true;
        results.cv.notify_all();
      }
      writer_thr.join();
      std::string msg = "all GPUs failed (--gpu-only).";
      for (const auto & s : fatal_msgs)
        if (!s.empty()) { msg += " "; msg += s; }
      die(msg, EXIT_GPU_FAIL);
    } else {
      // Some GPUs failed but others are working (or hybrid mode)
      for (const auto & s : fatal_msgs)
        if (!s.empty())
          vlog(V_NORMAL, opt, "WARNING: " + s
               + (abort_on_failure.load() ? " (other GPUs continuing)\n" : " (rescued to CPU)\n"));
    }
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
  WriterBackpressure * bp)
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
    size_t stream_index = 0;

    // Auto-tune tracking
    std::chrono::steady_clock::time_point last_adjust{ std::chrono::steady_clock::now() };
    uint64_t tune_in_bytes = 0;      // bytes processed since last tune check
    double   tune_elapsed_ms = 0.0;  // ms elapsed since last tune check
    enum class TuneState { EXPLORE, REFINE, SETTLE };
    TuneState tune_state = TuneState::EXPLORE;
    size_t    tune_lo = 0, tune_hi = 0;
    size_t    tune_best_batch = 0;
    double    tune_best_thr = 0.0;
    size_t    tune_prev_batch = 0;
    double    tune_prev_thr = 0.0;
    uint32_t  tune_batches_at_size = 0;
    uint32_t  tune_settle_count = 0;
    // Constants for tuner (not static  local classes can't have static members)
    uint32_t TUNE_MIN_BATCHES = 2;
    uint32_t TUNE_PROBE_INTERVAL = 15;

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
    size_t per_stream_cap = std::min(opt.gpu_batch_cap, HARD_BATCH_CAP);

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
      {
        size_t est_temp = per_stream_cap * 1024;
        size_t try_batch = per_stream_cap;
        int vram_retries = 0;
        while (!C.ensure_buffers(try_batch, init_comp, init_decomp, est_temp)) {
          if (try_batch <= 1 || ++vram_retries > 10) {
            // Can't fit even batch=1  skip this GPU entirely.
            // Other GPUs (or CPU in hybrid) will handle the work.
            std::string skip_msg = "[GPU" + std::to_string(device_id)
                + "] insufficient VRAM for even batch=1 ("
                + std::to_string(init_comp / ONE_MIB) + " MiB comp + "
                + std::to_string(init_decomp / ONE_MIB) + " MiB decomp per frame)  skipping device";
            vlog(V_ERROR, opt, skip_msg + "\n");
            *any_gpu_failed = true;
            *fatal_msg = skip_msg;
            // Clean up streams/events we already created
            for (size_t cs = 0; cs <= s; ++cs) {
              if (ctxs[cs].ev_begin) { cudaEventDestroy(ctxs[cs].ev_begin); ctxs[cs].ev_begin = nullptr; }
              if (ctxs[cs].ev_end)   { cudaEventDestroy(ctxs[cs].ev_end);   ctxs[cs].ev_end = nullptr; }
              if (ctxs[cs].stream)   { cudaStreamDestroy(ctxs[cs].stream);   ctxs[cs].stream = nullptr; }
            }
            // Wake writer in case it's waiting
            { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
            return;
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
        // Update per_stream_cap to the actual allocated size
        // (used by pop_batch_greedy to limit how many frames we grab)
        per_stream_cap = try_batch;
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
    if (sched) sched->set_gpu_ready(device_id);

    // Utilization scaling factor: 1.0 = idle, 0.1 = 90% busy.
    // Updated after each batch completion via NVML query.
    // Applied to batch size at next pop to match busy GPUs' completion time
    // with idle GPUs' completion time.
    double util_scale = 1.0;

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
                    os << "[auto-tune-D] baseline=" << cur_batch << " ("
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
                    os << "[auto-tune-D] halving worse, will try doubling (best=" << S.best_batch << ")";
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
                      S.batch_size.store(S.refine_lo + (S.refine_hi - S.refine_lo) / 2);
                      S.phase = SharedTuneState::Phase::REFINE; S.refine_iters = 0;
                    } else {
                      S.batch_size.store(S.best_batch);
                      S.phase = SharedTuneState::Phase::SETTLED; S.settle_ticks = 0;
                    }
                    if (opt.verbosity >= V_VERBOSE) {
                      std::ostringstream os;
                      os << "[auto-tune-D] settled at batch=" << S.best_batch
                         << " (" << std::fixed << std::setprecision(2) << S.best_thr << " GiB/s)";
                      vlog(V_VERBOSE, opt, os.str() + "\n");
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
                    os << "[auto-tune-D] refined, settled at batch=" << S.best_batch;
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
                      os << "[auto-tune-D] probe: " << S.best_batch << " -> " << probe
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
        C.batch.clear();
        uint64_t qw_t0 = g_perf ? now_ns() : 0;
        // Signal scheduler: this GPU stream wants data (blocks CPU workers)
        if (sched) sched->gpu_wants_data();
        // Use shared batch size from auto-tuner, scaled by GPU utilization.
        // A GPU at 50% utilization gets half the batch → finishes at roughly
        // the same time as idle GPUs → results arrive in order for the writer.
        size_t pop_n = (shared_tune && !shared_tune->locked.load())
                     ? std::min(shared_tune->batch_size.load(std::memory_order_relaxed), per_stream_cap)
                     : per_stream_cap;
        // Apply utilization scaling (updated after each batch completion)
        pop_n = std::max<size_t>(1, (size_t)(pop_n * util_scale));
        // Backpressure: wait if writer is overwhelmed before grabbing more work.
        // This prevents GPUs from flooding the result store with decompressed
        // data faster than the NVMe can write.
        if (bp) bp->wait_if_backlogged();
        if (!queue->pop_batch_greedy(pop_n, C.batch)) {
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
        uint64_t batch_t0 = now_ns();  // always record for auto-tuner (not just -vvv)

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
          os << "[GPU" << device_id << "/S" << C.stream_index
             << "] take N=" << C.filled << " in=" << in_s;
          vlog(V_DEBUG, opt, os.str() + "\n");
        }

        // Get temp workspace size for this batch configuration.
        // We need a dummy ensure_buffers first if sizes grew, then query temp.
        // Use a two-pass approach: ensure base buffers, query temp, re-ensure with temp.
        if (!C.ensure_buffers(C.filled, max_comp, max_decomp, C.temp_bytes)) {
          throw std::runtime_error("GPU decomp: failed to allocate device buffers");
        }

        // Upload compressed data H2D (async into CUDA stream).
        // Per-frame cudaMemcpyAsync is efficient because CUDA batches them
        // internally in the stream  no host-side blocking between transfers.
        // A single large memcpy would require packing a contiguous host buffer
        // first, which is slower than letting CUDA handle the scatter.
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

        // Release host-side compressed data  it's on the GPU now.
        // Save per-frame metadata needed by completion paths.
        std::vector<size_t> batch_comp_sizes(C.filled);
        std::vector<size_t> batch_seqs(C.filled);
        for (size_t i = 0; i < C.filled; ++i) {
          batch_comp_sizes[i] = C.batch[i].data.size();
          batch_seqs[i] = C.batch[i].seq;
          { std::vector<char>().swap(C.batch[i].data); }
        }

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

        // Download decompressed frames one at a time and deliver to writer
        // immediately.  Per-frame D2H has slightly more DMA overhead than a
        // single batch transfer, but it keeps the writer busy in parallel 
        // the writer can write frame 0 to disk while frame 1 is transferring.
        // A single batch D2H would stall the writer for the entire transfer.
        for (size_t i = 0; i < C.filled; ++i) {
          if (C.h_statuses[i] != nvcompSuccess)
            throw std::runtime_error("nvCOMP decompress per-chunk status != success");

          size_t actual = C.h_actual[i];
          out_sum += actual;
          d2h_bytes_batch += actual;

          std::vector<char> h_out(actual);
          const void * d_src = static_cast<char*>(C.d_decomp_buf) + i * C.alloc_decomp;
          checkCuda(cudaMemcpy(h_out.data(), d_src, actual,
                               cudaMemcpyDeviceToHost), "D2H decomp data");

          uint64_t rl_t0 = g_perf ? now_ns() : 0;
          results->push_to_slot(slot_index, batch_seqs[i], std::move(h_out));
          if (g_perf) g_perf->result_lock_ns.fetch_add(now_ns() - rl_t0);
        }
        // Track GPU output for writer backpressure accounting.
        // GPUs are never throttled (batches are in-flight), but their output
        // counts toward the backlog so CPU workers throttle correctly.
        if (bp) bp->mark_produced(out_sum);

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
            in_sum += batch_comp_sizes[i];
          if (m) m->read_bytes.fetch_add(in_sum);
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
          char out_s[32];
          human_bytes(double(out_sum), out_s, sizeof(out_s));
          double thr_gib = (batch_ms > 0.0)
                           ? double(out_sum) / (batch_ms / 1000.0) / 1e9 : 0.0;
          std::ostringstream os;
          os << "[GPU" << device_id << "/S" << C.stream_index
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
      if (C.ev_begin) cudaEventDestroy(C.ev_begin);
      if (C.ev_end)   cudaEventDestroy(C.ev_end);
      if (C.stream)   cudaStreamDestroy(C.stream);
    }
  }
  catch (const std::exception & e) {
    *any_gpu_failed = true;
    *fatal_msg = std::string("[GPU") + std::to_string(device_id) + "] " + e.what();

    // Rescue in-flight chunks back to queue so other GPUs can pick them up.
    // The batch was popped from the queue but never decompressed.
    try {
      if (vram_reserve) { cudaFree(vram_reserve); vram_reserve = nullptr; }
      for (auto & C : ctxs) {
        if (!C.batch.empty()) {
          queue->re_enqueue(C.batch);
        }
        C.free_device();
        if (C.ev_begin) { cudaEventDestroy(C.ev_begin); C.ev_begin = nullptr; }
        if (C.ev_end)   { cudaEventDestroy(C.ev_end);   C.ev_end = nullptr; }
        if (C.stream)   { cudaStreamDestroy(C.stream);   C.stream = nullptr; }
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
      die_usage("GPU requested (--gpu-only) but no CUDA devices available");
    vlog(V_VERBOSE, opt, "nvCOMP: no GPUs found; falling back to MT CPU decompress\n");
    // Fall through with device_count=0; CPU pool will handle everything
  }

  // Apply --gpu-devices limit.
  // Default (0) = 1 GPU for decompress: D2H transfer of the full
  // Use all available GPUs for decompression. Utilization-scaled batch sizing
  // (via NVML queries in select_best_gpus) handles partially-loaded GPUs.
  // Previously capped at 2 GPUs due to PCIe contention concerns, but the
  // auto-tuner and utilization scaling handle this better dynamically.
  const int total_hw_devices = device_count;
  if (device_count > 0) {
    int target = opt.gpu_devices;
    if (target == 0) target = device_count;  // auto: use all GPUs
    if (target < device_count) {
      vlog(V_VERBOSE, opt, "decompress: using " + std::to_string(target)
           + " of " + std::to_string(device_count) + " GPU devices"
           + (opt.gpu_devices == 0 ? " (auto)\n" : "\n"));
      device_count = target;
    }
  }

  // ---- Shared state ----
  TaskQueue queue;
  RescueQueue rescue;
  ResultStore results;
  std::atomic<bool> any_gpu_failed{false};
  std::atomic<bool> abort_on_failure{ opt.gpu_only };
  RateMatchState rate_match;

  // Writer backpressure: prevents CPU workers from producing decompressed data
  // faster than the NVMe can write.  Without this, 96 CPU threads + 8 GPUs
  // all flooding the writer causes 19 minutes of sys time (kernel writeback).
  // GPUs call mark_produced() but are never throttled (batches in-flight).
  // CPU workers call wait_if_backlogged() before popping a new task.
  // Disabled in test mode: no disk I/O means mark_written() is never called,
  // so workers would block at the high-water mark forever.
  WriterBackpressure backpressure;
  WriterBackpressure * bp_ptr = (opt.mode == Mode::TEST) ? nullptr : &backpressure;

  // ---- Writer thread (outputs decompressed data in order) ----
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m, bp_ptr);

  // ---- Hybrid scheduler ----
  std::unique_ptr<HybridSched> sched_ptr;
  HybridSched * sched = nullptr;
  std::atomic<bool> tick_done{false};
  std::thread tick_thr;

  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
    sched_ptr = std::make_unique<HybridSched>(
        opt.cpu_share, 0, device_count, opt);
    sched = sched_ptr.get();
    sched->set_queue(&queue);
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


  if (device_count > 0) {
    gpu_ids = select_best_gpus(total_hw_devices, device_count, opt);
    const int gpu_count = (int)gpu_ids.size();
    fatal_msgs.resize(gpu_count);
    results.init_slots(gpu_count);  // per-GPU result slots (reduces lock contention)

    for (int i = 0; i < gpu_count; ++i) {
      gpu_workers.emplace_back(gpu_decomp_worker, gpu_ids[i], i, opt,
                               &queue, &rescue, &results, m, sched,
                               &any_gpu_failed, &abort_on_failure,
                               &fatal_msgs[size_t(i)],
                               &shared_tune_decomp, &rate_match,
                               bp_ptr);
    }
    if (opt.verbosity >= V_VERBOSE) {
      std::ostringstream os;
      os << "[GPU] " << (int)gpu_ids.size() << " device(s) online";
      vlog(V_VERBOSE, opt, os.str() + "\n");
    }
  }

  // ---- Stream frames from input into the queue ----
  // Workers start decompressing as soon as the first frame is pushed.
  // Pin reader to core 0 so it isn't preempted by the worker pool.
  // pin_thread_to_core(0);              // disabled: hurts on loaded machines
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
    backpressure.set_done();
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
    // Count how many GPUs actually failed
    int failed_count = 0;
    for (const auto & s : fatal_msgs)
      if (!s.empty()) ++failed_count;
    int total_gpus = (int)fatal_msgs.size();

    if (failed_count >= total_gpus) {
      // ALL GPUs failed  fatal in --gpu-only mode
      {
        std::lock_guard<std::mutex> lk(results.m);
        results.workers_done = true;
        results.cv.notify_all();
      }
      writer_thr.join();
      std::string msg = "all GPUs failed (--gpu-only).";
      for (const auto & s : fatal_msgs)
        if (!s.empty()) { msg += " "; msg += s; }
      die(msg, EXIT_GPU_FAIL);
    } else {
      // Some GPUs failed but others are still working
      for (const auto & s : fatal_msgs)
        if (!s.empty())
          vlog(V_NORMAL, opt, "WARNING: " + s + " (other GPUs continuing)\n");
    }
  }

  rescue.set_done();
  backpressure.set_done();  // wake any CPU workers blocked on backpressure
  if (!cpu_pool.empty()) {
    auto t_cpu = std::chrono::steady_clock::now();
    for (auto & th : cpu_pool) th.join();
    if (opt.verbosity >= V_VERBOSE) {
      double ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(
          std::chrono::steady_clock::now() - t_cpu).count();
      if (ms > 500)
        vlog(V_VERBOSE, opt, "CPU pool join: " + std::to_string(int(ms)) + " ms\n");
    }
  }

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
        vlog(V_VERBOSE, opt, "writer join: " + std::to_string(int(ms)) + " ms\n");
    }
  }

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
      die_io("output exists (use -f to overwrite): " + opt.output);
    }
    if (exists && opt.force) {
      // Atomic overwrite: write to .tmp, rename on success.
      // Skip for device/special files (e.g., /dev/null)  can't create .tmp there.
      bool is_regular = fs::is_regular_file(opt.output);
      if (is_regular) {
        out = open_output_atomic(opt.output, tmp);
        use_atomic = true;
        register_tmp_file(tmp);  // clean up .tmp on failure (original preserved)
      } else {
        out = std::fopen(opt.output.c_str(), "wb");
        if (!out) die_io("cannot open output: " + opt.output);
      }
    } else {
      // Direct write: write to final name, delete on failure
      out = std::fopen(opt.output.c_str(), "wb");
      if (!out) die_io("cannot open output: " + opt.output);
      register_tmp_file(opt.output); // arm cleanup to delete on failure
    }
  }

  // Try O_DIRECT for regular file output (bypasses page cache for ~2-3x write speed).
  // Falls back gracefully: pipes, stdout, device files, and filesystems without
  // O_DIRECT support all use standard fwrite.
#ifndef _WIN32
  std::unique_ptr<DirectWriter> direct_writer;
  if (!to_stdout && out != stdout) {
    // Explicit output file: use O_DIRECT if it's a regular file
    std::string write_path = use_atomic ? tmp : opt.output;
    struct stat st;
    bool is_regular = (stat(write_path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
    if (is_regular) {
      auto dw = std::make_unique<DirectWriter>();
      if (dw->open(write_path)) {
        if (out) { std::fclose(out); out = nullptr; }
        direct_writer = std::move(dw);
        vlog(V_VERBOSE, opt, "using O_DIRECT for output (bypass page cache)\n");
      }
    }
  } else if (to_stdout && out == stdout) {
    // stdout path: check if stdout is redirected to a regular file.
    // If so, reopen with O_DIRECT for ~2× write speed.
    // Fall back silently on any issue (append mode, unsupported fs, etc.)
    do {
      int fd = fileno(stdout);
      if (fd < 0) break;
      struct stat st;
      if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) break;
      // Skip if O_APPEND is set (O_DIRECT + O_APPEND is undefined on Linux)
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || (flags & O_APPEND)) break;
      // Get the file path from /proc/self/fd/N
      char link_path[64];
      char real_path[4096];
      std::snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd);
      ssize_t len = readlink(link_path, real_path, sizeof(real_path) - 1);
      if (len <= 0) break;
      real_path[len] = '\0';
      // Sanity: skip /dev/null, /dev/*, deleted files
      if (std::strncmp(real_path, "/dev/", 5) == 0) break;
      if (std::strstr(real_path, "(deleted)")) break;
      // Try opening with O_DIRECT
      auto dw = std::make_unique<DirectWriter>();
      if (dw->open(std::string(real_path))) {
        // Success: close stdout's FILE*, the DirectWriter owns the fd now
        std::fflush(stdout);
        // Don't fclose(stdout)  it's special. Just stop using it.
        out = nullptr;
        direct_writer = std::move(dw);
        vlog(V_VERBOSE, opt, "stdout is a regular file  using O_DIRECT (bypass page cache)\n");
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
    if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input))
      total_in_for_progress = (uint64_t)fs::file_size(opt.input);
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
          die_data("not a Zstd file (bad magic: 0x"
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
    std::fclose(in);
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
        vlog(V_VERBOSE, opt, "atomic rename: " + std::to_string(int(rename_ms)) + " ms\n");
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
      "%s : %5.2f%% (%s => %s, %s) @ %s/s",
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
      "%s : %s => %s, %s @ %s/s",
      in_name.c_str(), in_s, out_s, out_name.c_str(), rate_s);
    std::fprintf(stderr, "\r%s\033[K\n", summary);
    std::fflush(stderr);
  }

  // Print finalize/fsync timing (deferred until after summary line)
  if (opt.verbosity >= V_VERBOSE) {
    if (finalize_ms > 100)
      vlog(V_VERBOSE, opt, "DirectWriter finalize: " + std::to_string(int(finalize_ms)) + " ms\n");
    if (sync_ms > 100)
      vlog(V_VERBOSE, opt, "fsync: " + std::to_string(int(sync_ms)) + " ms\n");
  }


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
    else if (a == "--sync-output") opt.sync_output = true;
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
    else if (a == "-T" || a == "--threads" || a.rfind("--threads=", 0) == 0) {
      int th = 0;
      if (a.rfind("--threads=", 0) == 0) {
        th = std::stoi(a.substr(10));
      } else if (a == "-T" && i + 1 < argc) {
        th = std::stoi(argv[++i]);
      } else if (a == "--threads" && i + 1 < argc) {
        th = std::stoi(argv[++i]);
      } else {
        die_usage("missing value for " + a);
      }
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
    else if (parse_num_arg("cpu-batch", i, argc, argv, opt.cpu_queue_min, nullptr)) {}
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
    else if (a == "--") {
      // End of options  everything after this is a filename
      for (++i; i < argc; ++i)
        opt.inputs.push_back(argv[i]);
    }
    else if (a.size() > 1 && a[0] == '-' && a != "-") {
      die_usage("unknown option: " + a);
    }
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
