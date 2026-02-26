// gzstd.cpp  Hybrid CPU+GPU Zstd (adaptive share), v0.9.31
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
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <stdexcept>
#include <memory>
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef _WIN32
 #include <io.h>
 #include <fcntl.h>
#endif
#ifdef HAVE_NVCOMP
 #include <cuda_runtime.h>
 #include <nvcomp/zstd.h>
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
static constexpr const char * GZSTD_VERSION = "0.9.31";

static const size_t ONE_MIB = size_t(1024) * size_t(1024);
static const size_t DEFAULT_CHUNK_MIB = 32;
#ifdef HAVE_NVCOMP
static const size_t GPU_SUBCHUNK_MAX = size_t(16) * ONE_MIB; // max GPU subchunk
static const size_t DEFAULT_GPU_BATCH_CAP = 16;   // per device (split across streams)
static const double DEFAULT_GPU_MEM_FRACTION = 0.60; // fraction of free VRAM to use
static const size_t DEFAULT_GPU_STREAMS = 3;      // streams per device
static const size_t AUTO_HOST_CHUNK_MIN_MIB = 32;
static const size_t AUTO_HOST_CHUNK_MAX_MIB = 512;
static const double GROW_CHECK_SEC = 1.0;
static const size_t HARD_BATCH_CAP = 256;         // per stream safety cap
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
  int cpu_threads = 0;       // 0=auto
  double cpu_share = -1;     // <0 adaptive (hybrid)
  size_t chunk_mib = DEFAULT_CHUNK_MIB;
  bool chunk_user_set = false;
  size_t cpu_backlog = 0;    // queue depth before CPU pops (hybrid)
#ifdef HAVE_NVCOMP
  size_t gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
  double gpu_mem_fraction = DEFAULT_GPU_MEM_FRACTION;
  size_t gpu_streams = DEFAULT_GPU_STREAMS;
  PinMode pin_mode = PinMode::AUTO;
#endif
  std::string stats_json;
  std::string input;
  std::string output;
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

// Emit a message to stderr if the current verbosity is >= min_level
static void vlog(int min_level, const Options & opt, const std::string & msg)
{ if (opt.verbosity >= min_level) std::cerr << msg << "\n"; }

static void die(const std::string & msg, int code = EXIT_ERROR)
{ if (g_verbosity >= V_ERROR) std::cerr << "gzstd: " << msg << "\n"; std::exit(code); }
static void die_usage(const std::string & msg)
{ die(msg, EXIT_USAGE); }

static void print_help()
{
  std::cout <<
"gzstd " << GZSTD_VERSION << " - Hybrid CPU+GPU Zstd compression\n"
"\n"
"Usage: gzstd [options] [file]\n"
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
" --cpu-only Force CPU path (multithreaded)\n"
" --hybrid Enable hybrid CPU+GPU scheduling (shared queue)\n"
" -T, --threads N  CPU worker threads (0=auto). Forms -T N or -T# are accepted [CPU-only/hybrid]\n"
" --cpu-share X Fixed CPU share [0..1], disables adaptation (hybrid)\n"
" --cpu-backlog N Min queue depth before CPU pops (hybrid; 0=off)\n";
#ifdef HAVE_NVCOMP
  std::cout <<
" --gpu-batch N Max GPU subchunks per device (default: 16)\n"
" --gpu-mem-frac X Fraction of free VRAM per device (0.1..0.95, def: 0.60)\n"
" --gpu-streams N CUDA streams per device (default: 3)\n"
" --gpu-only Error out if GPU path becomes unavailable\n"
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

static bool parse_num_arg(const std::string & name, int & i, int argc, char ** argv, size_t & out, bool * was_set = nullptr)
{
  const std::string pref = "--" + name; std::string a = argv[i];
  if (a.rfind(pref + "=", 0) == 0) { std::string v = a.substr(pref.size() + 1); if (v.empty()) die("missing value for " + pref); out = std::stoull(v); if (was_set) *was_set = true; return true; }
  else if (a == pref) { if (i + 1 >= argc) die("missing value for " + pref); out = std::stoull(argv[++i]); if (was_set) *was_set = true; return true; }
  return false;
}
static bool parse_int_arg(const std::string & name, int & i, int argc, char ** argv, int & out)
{
  const std::string pref = "--" + name; std::string a = argv[i];
  if (a.rfind(pref + "=", 0) == 0) { std::string v = a.substr(pref.size() + 1); if (v.empty()) die("missing value for " + pref); out = std::stoi(v); return true; }
  else if (a == pref) { if (i + 1 >= argc) die("missing value for " + pref); out = std::stoi(argv[++i]); return true; }
  return false;
}
static bool parse_str_arg(const std::string & name, int & i, int argc, char ** argv, std::string & out)
{
  const std::string pref = "--" + name; std::string a = argv[i];
  if (a.rfind(pref + "=", 0) == 0) { std::string v = a.substr(pref.size() + 1); if (v.empty()) die("missing value for " + pref); out = v; return true; }
  else if (a == pref) { if (i + 1 >= argc) die("missing value for " + pref); out = argv[++i]; return true; }
  return false;
}
#if __cplusplus >= 201703L
[[maybe_unused]]
#endif
static bool parse_double_arg(const std::string & name, int & i, int argc, char ** argv, double & out)
{
  const std::string pref = "--" + name; std::string a = argv[i];
  if (a.rfind(pref + "=", 0) == 0) { std::string v = a.substr(pref.size() + 1); if (v.empty()) die("missing value for " + pref); out = std::stod(v); return true; }
  else if (a == pref) { if (i + 1 >= argc) die("missing value for " + pref); out = std::stod(argv[++i]); return true; }
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
static void human_bytes(double x, char * buf, size_t n)
{ const char * u[] = {"B","KiB","MiB","GiB","TiB"}; int k=0; while (x>=1024.0 && k<4) { x/=1024.0; ++k; } std::snprintf(buf, n, "%.2f %s", x, u[k]); }

static void progress_emit_line(double pct, const char * in_s, const char * out_s, const char * rate_s, size_t & last_len)
{
  char line[256];
  int n = 0;
  if (pct >= 0.0) { n = std::snprintf(line, sizeof(line), "[%.1f%%] in:%s out:%s in_rate:%s/s ", pct, in_s, out_s, rate_s); }
  else { n = std::snprintf(line, sizeof(line), " in:%s out:%s in_rate:%s/s ", in_s, out_s, rate_s); }
  if (n < 0) { n = 0; }
  size_t len = (size_t)std::min(n, (int)sizeof(line) - 1);
  std::fprintf(stderr, "\r%.*s", (int)len, line);
  if (last_len > len) { std::fprintf(stderr, "%*s", (int)(last_len - len), ""); }
  std::fflush(stderr);
  last_len = len;
}

static void progress_loop(const Options & opt, const Meter * m, uint64_t total_in, std::atomic< bool > * done_flag)
{
  // Progress requires V_DEFAULT(2) or higher.
  // At V_DEFAULT, suppress if stderr is not a TTY OR if stdin is a pipe
  // (reading from a pipe = no known total size, progress is noise).
  // --progress (force_progress) overrides both checks.
  // At V_VERBOSE+, the user explicitly asked for output, so always show.
  if (opt.verbosity < V_DEFAULT) return;
  if (opt.verbosity == V_DEFAULT && !opt.force_progress) {
    if (!is_stderr_tty()) return;
    if (opt.input == "-" && !isatty(fileno(stdin))) return;
  }
  using namespace std::chrono; using namespace std::chrono_literals;
  size_t last_len = 0;
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
    progress_emit_line(pct, in_s, out_s, rate_s, last_len);
  }
  // Final sample
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
  size_t last_len2 = 0;
  progress_emit_line(pct, in_s, out_s, rate_s, last_len2);
  std::fprintf(stderr, "\n");
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
static FILE * open_output_atomic(const std::string & out, std::string & tmp_path)
{ tmp_path = out + ".gzstd.tmp"; register_tmp_file(tmp_path); FILE * f = std::fopen(tmp_path.c_str(), "wb"); if (!f) die("cannot open temp output: " + tmp_path); return f; }
static void fsync_file(FILE * f)
{
#if defined(_POSIX_VERSION)
  int fd = fileno(f); fsync(fd);
#else
  (void)f;
#endif
}
static bool is_regular_file_stream(FILE * f)
{ if (!f) return false; int fd = fileno(f); struct stat st; if (fstat(fd, &st) != 0) return false; return S_ISREG(st.st_mode); }

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
 Auto chunk selection
======================================================================*/
static size_t auto_chunk_mib_cpu(FILE * in, const Options & opt)
{ (void)opt; return is_regular_file_stream(in) ? size_t(64) : size_t(32); }
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
struct Task { size_t seq; std::vector<char> data; };

class TaskQueue {
public:
  void push(Task && t) { std::unique_lock<std::mutex> lk(m_); q_.push_back(std::move(t)); ++total_tasks_; cv_.notify_one(); }
  bool pop_batch(size_t max_n, std::vector<Task> & out) {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cv_.wait(lk);
    if (q_.empty() && done_) return false;
    const size_t take = std::min(max_n, q_.size());
    out.reserve(out.size() + take);
    for (size_t i=0;i<take;++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
    return true;
  }
  bool pop_one(Task & t) {
    std::unique_lock<std::mutex> lk(m_);
    while (q_.empty() && !done_) cv_.wait(lk);
    if (q_.empty() && done_) return false;
    t = std::move(q_.front()); q_.pop_front(); return true;
  }
  void set_done() { std::unique_lock<std::mutex> lk(m_); done_ = true; cv_.notify_all(); }
  size_t total_tasks() const { return total_tasks_; }
  bool drained() { std::unique_lock<std::mutex> lk(m_); return q_.empty() && done_; }
  size_t size() { std::unique_lock<std::mutex> lk(m_); return q_.size(); }
private:
  std::mutex m_; std::condition_variable cv_; std::deque<Task> q_; bool done_ = false; std::atomic<size_t> total_tasks_{0};
};

class RescueQueue {
public:
  void push(Task && t) { std::unique_lock<std::mutex> lk(m_); q_.push_back(std::move(t)); cv_.notify_one(); }
  bool pop_one(Task & t) { std::unique_lock<std::mutex> lk(m_); while (q_.empty() && !done_) cv_.wait(lk); if (q_.empty() && done_) return false; t = std::move(q_.front()); q_.pop_front(); return true; }
  bool drained() { std::unique_lock<std::mutex> lk(m_); return q_.empty() && done_; }
  size_t size() { std::unique_lock<std::mutex> lk(m_); return q_.size(); }
  void set_done() { std::unique_lock<std::mutex> lk(m_); done_ = true; cv_.notify_all(); }
private:
  std::mutex m_; std::condition_variable cv_; std::deque<Task> q_; bool done_ = false;
};

struct ResultStore { std::mutex m; std::condition_variable cv; std::unordered_map<size_t, std::vector<char>> data; size_t next_to_write = 0; size_t total_tasks = 0; bool producer_done = false; bool workers_done = false; };

static void writer_thread(FILE * out, ResultStore & results, const Options & opt, Meter * m)
{
  (void)opt; std::unique_lock<std::mutex> lk(results.m);
  while (true) {
    while (results.data.count(results.next_to_write) == 0 && !(results.producer_done && results.workers_done && results.next_to_write >= results.total_tasks)) results.cv.wait(lk);
    if (results.producer_done && results.workers_done && results.next_to_write >= results.total_tasks) break;
    auto it = results.data.find(results.next_to_write); if (it == results.data.end()) continue;
    const std::vector<char> & buf = it->second;
    lk.unlock();
    size_t w = robust_fwrite(buf.data(), buf.size(), out);
    if (w != buf.size()) die("short write to output (broken pipe?)");
    if (m) m->wrote_bytes.fetch_add(w);
    lk.lock(); results.data.erase(it); ++results.next_to_write;
  }
}

/*======================================================================
 CPU compression helpers / workers
======================================================================*/
static inline void compress_one_cpu_frame(const void * src, size_t src_size, int level, std::vector< char > & out)
{
  size_t bound = ZSTD_compressBound(src_size);
  out.resize(bound);
  size_t csz = ZSTD_compress(out.data(), out.size(), src, src_size, level);
  if (ZSTD_isError(csz)) die(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
  out.resize(csz);
}

static void decompress_stream(FILE * in, FILE * out, const Options & opt, Meter * m = nullptr)
{
  size_t chosen_mib = opt.chunk_mib;
  if (!opt.chunk_user_set) { chosen_mib = auto_chunk_mib_cpu(in, opt); vlog(V_VERBOSE,opt, std::string("auto-chunk (decompress): ") + std::to_string(chosen_mib) + " MiB"); }
  const size_t chunk_bytes = std::max<size_t>(1, chosen_mib) * ONE_MIB;
  std::vector<char> inbuf(chunk_bytes);
  std::vector<char> outbuf(chunk_bytes);
  ZSTD_DCtx * dctx = ZSTD_createDCtx(); if (!dctx) die("failed to create ZSTD_DCtx");
  ZSTD_inBuffer zin { nullptr, 0, 0 };
  ZSTD_outBuffer zout { outbuf.data(), outbuf.size(), 0 };
  while (true) {
    if (zin.pos == zin.size) {
      zin.size = std::fread(inbuf.data(), 1, chunk_bytes, in);
      zin.pos = 0; zin.src = inbuf.data();
      if (zin.size == 0) break;
      if (m) m->read_bytes.fetch_add(zin.size);
    }
    size_t ret = ZSTD_decompressStream(dctx, &zout, &zin);
    if (ZSTD_isError(ret)) die(std::string("ZSTD decompress error: ") + ZSTD_getErrorName(ret));
    if (zout.pos > 0) {
      if (opt.mode != Mode::TEST) {
        size_t w = robust_fwrite(outbuf.data(), zout.pos, out);
        if (w != zout.pos) die("short write to output (broken pipe?)");
      }
      if (m) m->wrote_bytes.fetch_add(zout.pos);
      zout.pos = 0;
    }
  }
  ZSTD_freeDCtx(dctx);
}

struct CpuThreadStats { uint64_t tasks=0; uint64_t in_bytes=0; uint64_t out_bytes=0; double comp_ms=0.0; };
struct CpuAgg { std::mutex m; uint64_t tasks=0; uint64_t in_bytes=0; uint64_t out_bytes=0; double comp_ms=0.0; int threads=0; std::vector<CpuThreadStats> per_thread; };

#ifdef HAVE_NVCOMP
/*======================================================================
 Hybrid scheduler  single definition placed before workers
======================================================================*/
class HybridSched {
public:
  HybridSched(double override_share, int /*cpu_threads*/, int /*gpu_devices*/, const Options & opt)
  : opt_(opt) {
    if (override_share >= 0.0) { set_target_share(override_share); adaptive_ = false; }
    else { set_target_share(0.25); adaptive_ = true; }
    window_cpu_taken_.store(0); window_gpu_taken_.store(0); cpu_bytes_.store(0); gpu_bytes_.store(0); last_tick_ = std::chrono::steady_clock::now();
  }
  bool should_cpu_take() const {
    const uint64_t gpu = window_gpu_taken_.load(std::memory_order_relaxed);
    if (gpu == 0) { return false; }
    const uint64_t cpu = window_cpu_taken_.load(std::memory_order_relaxed);
    const uint64_t total = cpu + gpu + 1;
    const double used = double(cpu) / double(total);
    const double tgt = share_.load(std::memory_order_relaxed) / 1000.0;
    return used < (tgt + 0.02);
  }
  void mark_cpu_take(uint64_t n){ window_cpu_taken_.fetch_add(n, std::memory_order_relaxed);}
  void mark_gpu_take(uint64_t n){ window_gpu_taken_.fetch_add(n, std::memory_order_relaxed);}
  void add_cpu_bytes(uint64_t b){ cpu_bytes_.fetch_add(b, std::memory_order_relaxed);}
  void add_gpu_bytes(uint64_t b){ gpu_bytes_.fetch_add(b, std::memory_order_relaxed);}
  void tick() {
    if (!adaptive_) { return; }
    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_tick_).count();
    if (secs < 0.8) { return; }
    last_tick_ = now;
    const uint64_t cpu_b = cpu_bytes_.exchange(0, std::memory_order_relaxed);
    const uint64_t gpu_b = gpu_bytes_.exchange(0, std::memory_order_relaxed);
    const double cpu_rate = cpu_b / std::max(1e-6, secs);
    const double gpu_rate = gpu_b / std::max(1e-6, secs);
    double tgt = (cpu_rate + gpu_rate > 0.0) ? (cpu_rate / (cpu_rate + gpu_rate)) : 0.25;
    if (tgt < 0.05) { tgt = 0.05; }
    if (tgt > 0.80) { tgt = 0.80; }
    const double cur = share_.load(std::memory_order_relaxed) / 1000.0;
    const double ema = 0.8 * cur + 0.2 * tgt;
    set_target_share(ema);
    window_cpu_taken_.store(0, std::memory_order_relaxed);
    window_gpu_taken_.store(0, std::memory_order_relaxed);
    if (opt_.verbosity >= V_DEBUG) {
      std::ostringstream os; os << std::fixed << std::setprecision(3)
        << "hybrid: tick cpu_rate=" << (cpu_rate/1e9) << " GiB/s"
        << " gpu_rate=" << (gpu_rate/1e9) << " GiB/s"
        << " target_cpu_share=" << (share_.load()/10.0) << "%";
      std::cerr << os.str() << "\n";
    }
  }
  double target_share() const { return share_.load() / 1000.0; }
private:
  void set_target_share(double s){ if (s<0.0) s=0.0; if (s>1.0) s=1.0; share_.store(uint32_t(std::lround(s*1000.0)), std::memory_order_relaxed);}
  const Options & opt_;
  std::atomic<uint32_t> share_{250}; // permille
  std::atomic<uint64_t> window_cpu_taken_{0}, window_gpu_taken_{0};
  std::atomic<uint64_t> cpu_bytes_{0}, gpu_bytes_{0};
  std::chrono::steady_clock::time_point last_tick_;
  bool adaptive_ = true;
};

static void tick_loop_fn(std::atomic<bool> & done, HybridSched * sched)
{ using namespace std::chrono_literals; while (!done.load()) { std::this_thread::sleep_for(200ms); sched->tick(); } }
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
  if (opt->verbosity >= V_VERBOSE) { std::ostringstream os; os << "[CPU/T" << worker_id << "] online"; vlog(V_VERBOSE, *opt, os.str()); }

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

    Task t; if (!tq->pop_one(t)) break;
#ifdef HAVE_NVCOMP
    if (sched) sched->mark_cpu_take(1);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> out_frame;
    compress_one_cpu_frame(t.data.data(), t.data.size(), opt->level, out_frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast< std::chrono::duration<double, std::milli> >(t1 - t0).count();
    const size_t csz = out_frame.size();

    if (opt->verbosity >= V_TRACE) {
      char in_s[32], out_s[32];
      human_bytes(double(t.data.size()), in_s, sizeof(in_s));
      human_bytes(double(csz), out_s, sizeof(out_s));
      const double thr_gib = (ms > 0.0) ? (double)t.data.size() / (ms/1000.0) / 1e9 : 0.0;
      std::ostringstream os;
      os << "[CPU/T" << worker_id << "] seq=" << t.seq
         << " in=" << in_s << " out=" << out_s
         << " ms=" << std::fixed << std::setprecision(2) << ms
         << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
      vlog(V_TRACE, *opt, os.str());
    }

    if (m) m->tasks_done.fetch_add(1);
    { std::lock_guard<std::mutex> lk(results->m); results->data.emplace(t.seq, std::move(out_frame)); }
    results->cv.notify_one();
#ifdef HAVE_NVCOMP
    if (sched) sched->add_cpu_bytes(t.data.size());
#endif
    {
      std::lock_guard<std::mutex> lk(cpuagg->m);
      if (cpuagg->per_thread.size() <= (size_t)worker_id) cpuagg->per_thread.resize((size_t)worker_id + 1);
      auto & st = cpuagg->per_thread[(size_t)worker_id];
      st.tasks += 1; st.in_bytes += t.data.size(); st.out_bytes += csz; st.comp_ms += ms;
      cpuagg->tasks += 1; cpuagg->in_bytes += t.data.size(); cpuagg->out_bytes += csz; cpuagg->comp_ms += ms;
    }
  }

  if (opt->verbosity >= V_DEBUG) {
    CpuThreadStats st; { std::lock_guard<std::mutex> lk(cpuagg->m); if ((size_t)worker_id < cpuagg->per_thread.size()) st = cpuagg->per_thread[(size_t)worker_id]; }
    const double thr_gib = (st.comp_ms > 0.0) ? (double)st.in_bytes / (st.comp_ms/1000.0) / 1e9 : 0.0;
    std::ostringstream os;
    os << "[CPU/T" << worker_id << "] total tasks=" << st.tasks
       << " in=" << st.in_bytes << "B out=" << st.out_bytes << "B"
       << " time=" << std::fixed << std::setprecision(2) << st.comp_ms << "ms"
       << " thr=" << std::fixed << std::setprecision(2) << thr_gib << " GiB/s";
    vlog(V_DEBUG, *opt, os.str());
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
  if (opt->verbosity >= V_VERBOSE) { std::ostringstream os; os << "[RESCUE/T" << worker_id << "] online"; vlog(V_VERBOSE, *opt, os.str()); }
  while (true) {
    Task t; if (!rq->pop_one(t)) break;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> out_frame; compress_one_cpu_frame(t.data.data(), t.data.size(), /*level*/3, out_frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration_cast< std::chrono::duration<double, std::milli> >(t1 - t0).count();
    const size_t csz = out_frame.size();
    if (opt->verbosity >= V_TRACE) {
      char in_s[32], out_s[32]; human_bytes(double(t.data.size()), in_s, sizeof(in_s)); human_bytes(double(csz), out_s, sizeof(out_s)); double thr_gib=(ms>0.0)?(double)t.data.size()/(ms/1000.0)/1e9:0.0; std::ostringstream os; os<<"[RESCUE/T"<<worker_id<<"] seq="<<t.seq<<" in="<<in_s<<" out="<<out_s<<" ms="<<std::fixed<<std::setprecision(2)<<ms<<" thr="<<std::fixed<<std::setprecision(2)<<thr_gib<<" GiB/s"; vlog(V_TRACE, *opt, os.str()); }
    { std::lock_guard<std::mutex> lk(results->m); results->data.emplace(t.seq, std::move(out_frame)); }
    results->cv.notify_one();
    { std::lock_guard<std::mutex> lk(cpuagg->m); if (cpuagg->per_thread.size() <= (size_t)worker_id) cpuagg->per_thread.resize((size_t)worker_id + 1); auto & st = cpuagg->per_thread[(size_t)worker_id]; st.tasks += 1; st.in_bytes += t.data.size(); st.out_bytes += csz; st.comp_ms += ms; }
  }
}

/*======================================================================
 MT CPU-only compression
======================================================================*/
static void compress_cpu_stream(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  size_t chosen_mib = opt.chunk_mib;
  if (!opt.chunk_user_set) { chosen_mib = auto_chunk_mib_cpu(in, opt); vlog(V_VERBOSE,opt, std::string("auto-chunk (CPU): ") + std::to_string(chosen_mib) + " MiB"); }
  const size_t chunk_bytes = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  uint64_t total_in = 0; if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input)) total_in = (uint64_t)fs::file_size(opt.input);
  std::atomic<bool> progress_done{false}; std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);

  std::vector< char > inbuf(chunk_bytes);
  std::vector< char > outbuf(ZSTD_compressBound(chunk_bytes));
  while (true) {
    size_t n = std::fread(inbuf.data(), 1, chunk_bytes, in);
    if (n == 0) break;
    if (m) m->read_bytes.fetch_add(n);
    size_t csz = ZSTD_compress(outbuf.data(), outbuf.size(), inbuf.data(), n, opt.level);
    if (ZSTD_isError(csz)) die(std::string("ZSTD error: ") + ZSTD_getErrorName(csz));
    size_t w = robust_fwrite(outbuf.data(), csz, out);
    if (w != csz) die("short write to output (broken pipe?)");
    if (m) m->wrote_bytes.fetch_add(w);
  }

  progress_done = true; progress_thr.join();
}

static void compress_cpu_mt(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  int threads = opt.cpu_threads; if (threads <= 0) { unsigned hw = std::max(1u, std::thread::hardware_concurrency()); threads = (hw>1)? int(hw-1) : 1; }

  // Option A: if single-threaded, use simple streaming helper
  if (threads == 1) {
    if (opt.verbosity >= V_VERBOSE)
      vlog(V_VERBOSE, opt, "CPU MT requested with 1 thread; using single-thread streaming path");
    compress_cpu_stream(in, out, opt, m);
    return;
  }

  CpuAgg cpuagg{}; cpuagg.threads = threads; cpuagg.per_thread.resize((size_t)threads);
  size_t chosen_mib = opt.chunk_mib; if (!opt.chunk_user_set) { chosen_mib = auto_chunk_mib_cpu(in, opt); vlog(V_VERBOSE,opt, std::string("auto-chunk (CPU-MT): ") + std::to_string(chosen_mib) + " MiB"); }
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  uint64_t total_in = 0; if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input)) total_in = (uint64_t)fs::file_size(opt.input);
  std::atomic<bool> progress_done{false}; std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);

  TaskQueue queue; ResultStore results; std::thread wthr(writer_thread, out, std::ref(results), std::cref(opt), m);

  std::vector<std::thread> pool; pool.reserve((size_t)threads);
  Options opt_copy = opt;
  for (int i = 0; i < threads; ++i) pool.emplace_back(cpu_worker, i, &queue, &results, &opt_copy, m,
#ifdef HAVE_NVCOMP
    nullptr,
#endif
    &cpuagg);

  std::vector<char> host_in(host_chunk); std::atomic<size_t> seq{0};
  while (true) {
    size_t n = std::fread(host_in.data(), 1, host_chunk, in);
    if (n == 0) break;
    if (m) m->read_bytes.fetch_add(n);
    Task t; t.seq = seq.fetch_add(1, std::memory_order_relaxed); t.data.assign(host_in.data(), host_in.data()+n);
    queue.push(std::move(t));
  }
  queue.set_done(); { std::lock_guard<std::mutex> lk(results.m); results.producer_done = true; results.total_tasks = queue.total_tasks(); } results.cv.notify_all();

  for (auto & th : pool) { th.join(); } { std::lock_guard<std::mutex> lk(results.m); results.workers_done = true; } results.cv.notify_all();

  wthr.join(); progress_done = true; progress_thr.join();
}

/*======================================================================
 GPU path (nvCOMP) + Hybrid
======================================================================*/
#ifdef HAVE_NVCOMP
struct DevStats { std::mutex m; double h2d_ms=0.0, comp_ms=0.0, d2h_ms=0.0, total_ms=0.0; uint64_t in_bytes=0, out_bytes=0; uint64_t batches=0; };
struct StreamStats { size_t dev_index=0, stream_index=0; bool pinned_h2d=false, pinned_d2h=false; size_t batch_capacity=0; double h2d_ms=0.0, comp_ms=0.0, d2h_ms=0.0, total_ms=0.0; uint64_t in_bytes=0, out_bytes=0; uint64_t batches=0; uint64_t chunks=0; };
struct StatsSink { std::mutex m; std::vector<std::vector<StreamStats>> per_dev; explicit StatsSink(int n): per_dev(size_t(n)){} };

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

struct StreamCtx {
  cudaStream_t stream{};
  void * d_in_base=nullptr, * d_out_base=nullptr, * d_temp=nullptr;
  void ** d_in_ptrs=nullptr; void ** d_out_ptrs=nullptr; size_t * d_in_sizes=nullptr; size_t * d_comp_sizes=nullptr; nvcompStatus_t* d_stats=nullptr;
  size_t temp_bytes_used=0;
  void * h2d_pinned_base=nullptr; void * d2h_pinned_base=nullptr;
  std::vector<size_t> h_in_sizes; std::vector<size_t> h_comp_sizes; std::vector<nvcompStatus_t> h_stats; std::vector<Task> batch;
  size_t gpu_chunk=0, max_out_chunk=0, per_stream_batch=0; cudaEvent_t ev_h2d_begin{}, ev_h2d_end{}, ev_comp_end{}, ev_d2h_end{}; bool busy=false; size_t filled=0;
  StreamStats stats{}; std::chrono::steady_clock::time_point last_adjust{ std::chrono::steady_clock::now() };
};

static bool allocate_stream_buffers(StreamCtx & C, size_t per_stream_batch, size_t gpu_chunk, size_t max_out_chunk, nvcompBatchedZstdCompressOpts_t comp_opts, const Options & opt)
{
  C.per_stream_batch = per_stream_batch; C.gpu_chunk=gpu_chunk; C.max_out_chunk=max_out_chunk; size_t temp_bytes = get_nvcomp_temp_size(per_stream_batch, gpu_chunk, comp_opts, C.stream); C.temp_bytes_used = temp_bytes;
  if (cudaMalloc(&C.d_in_base,  C.per_stream_batch * gpu_chunk) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_out_base, C.per_stream_batch * max_out_chunk) != cudaSuccess) return false;
  if (temp_bytes>0) { if (cudaMalloc(&C.d_temp, temp_bytes) != cudaSuccess) return false; }
  if (cudaMalloc(&C.d_in_ptrs,  sizeof(void*) * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_out_ptrs, sizeof(void*) * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_in_sizes,  sizeof(size_t) * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_comp_sizes,sizeof(size_t) * C.per_stream_batch) != cudaSuccess) return false;
  if (cudaMalloc(&C.d_stats,     sizeof(nvcompStatus_t) * C.per_stream_batch) != cudaSuccess) return false;
  bool want_pin = (opt.pin_mode != PinMode::OFF);
  if (want_pin) {
    if (cudaHostAlloc(&C.h2d_pinned_base, C.per_stream_batch * gpu_chunk, cudaHostAllocDefault) != cudaSuccess) C.h2d_pinned_base = nullptr;
    C.d2h_pinned_base = nullptr; // exact-size D2H
  }
  std::vector<void*> h_in_ptrs(C.per_stream_batch), h_out_ptrs(C.per_stream_batch);
  for (size_t i=0;i<C.per_stream_batch;++i){ h_in_ptrs[i]=static_cast<char*>(C.d_in_base)+i*gpu_chunk; h_out_ptrs[i]=static_cast<char*>(C.d_out_base)+i*max_out_chunk; }
  if (cudaMemcpyAsync(C.d_in_ptrs,  h_in_ptrs.data(),  sizeof(void*)*C.per_stream_batch, cudaMemcpyHostToDevice, C.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(C.d_out_ptrs, h_out_ptrs.data(), sizeof(void*)*C.per_stream_batch, cudaMemcpyHostToDevice, C.stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(C.stream) != cudaSuccess) return false;
  C.h_in_sizes.resize(C.per_stream_batch); C.h_comp_sizes.resize(C.per_stream_batch); C.h_stats.resize(C.per_stream_batch); C.batch.reserve(C.per_stream_batch);
  cudaEventCreateWithFlags(&C.ev_h2d_begin, cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_h2d_end,   cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_comp_end,  cudaEventDefault);
  cudaEventCreateWithFlags(&C.ev_d2h_end,   cudaEventDefault);
  C.stats.pinned_h2d = (C.h2d_pinned_base != nullptr); C.stats.pinned_d2h = false; C.stats.batch_capacity = per_stream_batch;
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
  try { if (!allocate_stream_buffers(tmp, desired_batch, C.gpu_chunk, C.max_out_chunk, comp_opts, opt)) return false; }
  catch(...) { return false; }
  std::swap(C.d_in_base, tmp.d_in_base); std::swap(C.d_out_base, tmp.d_out_base); std::swap(C.d_temp, tmp.d_temp);
  std::swap(C.d_in_ptrs,tmp.d_in_ptrs); std::swap(C.d_out_ptrs,tmp.d_out_ptrs);
  std::swap(C.d_in_sizes,tmp.d_in_sizes); std::swap(C.d_comp_sizes,tmp.d_comp_sizes); std::swap(C.d_stats,tmp.d_stats);
  std::swap(C.h2d_pinned_base,tmp.h2d_pinned_base); std::swap(C.d2h_pinned_base,tmp.d2h_pinned_base);
  std::swap(C.h_in_sizes,tmp.h_in_sizes); std::swap(C.h_comp_sizes,tmp.h_comp_sizes); std::swap(C.h_stats,tmp.h_stats);
  std::swap(C.batch,tmp.batch); std::swap(C.per_stream_batch,tmp.per_stream_batch);
  std::swap(C.gpu_chunk,tmp.gpu_chunk); std::swap(C.max_out_chunk,tmp.max_out_chunk); std::swap(C.temp_bytes_used,tmp.temp_bytes_used);
  std::swap(C.ev_h2d_begin,tmp.ev_h2d_begin); std::swap(C.ev_h2d_end,tmp.ev_h2d_end); std::swap(C.ev_comp_end,tmp.ev_comp_end); std::swap(C.ev_d2h_end,tmp.ev_d2h_end);
  C.stats.pinned_h2d = (C.h2d_pinned_base != nullptr); C.stats.pinned_d2h = false; C.stats.batch_capacity = C.per_stream_batch;
  tmp.stream = nullptr; free_stream_buffers_only(tmp);
  return true;
}

static void checkCuda(cudaError_t st, const char * msg)
{ if (st != cudaSuccess) throw std::runtime_error(std::string(msg)+": "+cudaGetErrorString(st)); }
static void checkNvcomp(nvcompStatus_t st, const char * msg)
{ if (st != nvcompSuccess) throw std::runtime_error(std::string(msg)+" (nvCOMP status "+std::to_string(int(st))+")"); }

static void gpu_worker(
  int device_id,
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
      size_t free_b=0,total_b=0; if (cudaMemGetInfo(&free_b,&total_b)!=cudaSuccess || free_b==0) { C.per_stream_batch=std::max<size_t>(1, per_stream_cap); }
      else {
        const size_t budget = static_cast<size_t>(free_b * per_stream_frac);
        const size_t per_in = gpu_chunk; const size_t per_out = max_out_chunk; const size_t per_meta = sizeof(void*)*2 + sizeof(size_t)*2 + sizeof(nvcompStatus_t);
        size_t bsz = budget / std::max<size_t>(1, (per_in+per_out+per_meta));
        if (bsz==0) { bsz=1; }
        bsz = std::min(bsz, per_stream_cap); bsz = std::min(bsz, HARD_BATCH_CAP); if (bsz==0) { bsz=1; }
        C.per_stream_batch = bsz;
      }
      while (!allocate_stream_buffers(C, C.per_stream_batch, gpu_chunk, max_out_chunk, comp_opts, opt)) {
        if (C.per_stream_batch==1) throw std::runtime_error("insufficient GPU memory for per-stream batch=1");
        C.per_stream_batch = std::max<size_t>(1, C.per_stream_batch/2);
      }
      C.stats.dev_index = size_t(device_id); C.stats.stream_index = s;
      if (opt.verbosity >= V_VERBOSE) { std::ostringstream os; os << "[GPU"<<device_id<<"/S"<<s<<"] subchunk="<<(gpu_chunk/ONE_MIB)<<"MiB batch="<<C.per_stream_batch; vlog(V_VERBOSE,opt, os.str()); }
    }

    bool producer_done_seen=false;
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
          size_t target = std::min<size_t>(std::min<size_t>(C.per_stream_batch*2, per_stream_cap), HARD_BATCH_CAP);
          if (try_grow_stream(C, target, comp_opts, opt)) { if (opt.verbosity>=V_DEBUG) { std::ostringstream os; os<<"[GPU"<<device_id<<"] stream grew to batch="<<C.per_stream_batch; vlog(V_DEBUG,opt, os.str()); } }
        }
        C.last_adjust = now;
      }

      // Submit batches
      for (auto & C : ctxs) {
        if (C.busy) { continue; }
        C.batch.clear();
        if (!queue->pop_batch(C.per_stream_batch, C.batch)) { producer_done_seen = true; continue; }
        if (C.batch.empty()) { continue; }
        if (gpu_started_flag) { gpu_started_flag->store(true, std::memory_order_release); }
        if (sched) { sched->mark_gpu_take(C.batch.size()); }
        C.filled = C.batch.size();

        // -vv: print take line
        if (opt.verbosity >= V_DEBUG) {
          size_t seq_lo = C.batch.front().seq, seq_hi = C.batch.back().seq; uint64_t tin=0; for (size_t i=0;i<C.filled;++i) tin += C.batch[i].data.size(); char tin_s[32]; human_bytes(double(tin), tin_s, sizeof(tin_s)); std::ostringstream os; os << "[GPU"<<device_id<<"/S"<<C.stats.stream_index<<"] take N="<<C.filled<<" seq=["<<seq_lo<<".."<<seq_hi<<"] in="<<tin_s; vlog(V_DEBUG,opt, os.str()); }

        cudaEventRecord(C.ev_h2d_begin, C.stream);
        for (size_t i=0;i<C.filled;++i) {
          const Task & t = C.batch[i]; void * d_dst = static_cast<char*>(C.d_in_base) + i * C.gpu_chunk;
          if (C.h2d_pinned_base) { void * h_src = static_cast<char*>(C.h2d_pinned_base) + i * C.gpu_chunk; std::memcpy(h_src, t.data.data(), t.data.size()); checkCuda(cudaMemcpyAsync(d_dst, h_src, t.data.size(), cudaMemcpyHostToDevice, C.stream), "cudaMemcpyAsync(H2D pinned)"); }
          else { checkCuda(cudaMemcpyAsync(d_dst, t.data.data(), t.data.size(), cudaMemcpyHostToDevice, C.stream), "cudaMemcpyAsync(H2D)"); }
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
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes)");
          float h2d_ms=0, comp_ms=0, d2h_ms=0, tot_ms=0; cudaEventElapsedTime(&h2d_ms, C.ev_h2d_begin, C.ev_h2d_end); cudaEventElapsedTime(&comp_ms, C.ev_h2d_end, C.ev_comp_end); cudaEventElapsedTime(&d2h_ms, C.ev_comp_end, C.ev_d2h_end); cudaEventElapsedTime(&tot_ms, C.ev_h2d_begin, C.ev_d2h_end);
          { std::lock_guard<std::mutex> lk(devstats->m); devstats->h2d_ms+=h2d_ms; devstats->comp_ms+=comp_ms; devstats->d2h_ms+=d2h_ms; devstats->total_ms+=tot_ms; devstats->batches+=1; }

          uint64_t in_sum=0, out_sum=0;
          for (size_t i=0;i<C.filled;++i) {
            if (C.h_stats[i] != nvcompSuccess) throw std::runtime_error("nvCOMP per-chunk status != nvcompSuccess");
            const size_t csz = C.h_comp_sizes[i]; out_sum += csz; in_sum += C.h_in_sizes[i];
            std::vector<char> h_out(csz); const void * d_src = static_cast<char*>(C.d_out_base) + i * C.max_out_chunk; checkCuda(cudaMemcpy(h_out.data(), d_src, csz, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H exact)");
            { std::lock_guard<std::mutex> lk(results->m); results->data.emplace(C.batch[i].seq, std::move(h_out)); }
            results->cv.notify_one();
          }
          #ifdef HAVE_NVCOMP
          if (sched) sched->add_gpu_bytes(in_sum);
          #endif
          { C.stats.h2d_ms += h2d_ms; C.stats.comp_ms += comp_ms; C.stats.d2h_ms += d2h_ms; C.stats.total_ms += tot_ms; C.stats.in_bytes += in_sum; C.stats.out_bytes += out_sum; C.stats.batches += 1; C.stats.chunks += C.filled; }

          // -vv: done line
          if (opt.verbosity >= V_DEBUG) {
            char in_s[32], out_s[32]; human_bytes(double(in_sum), in_s, sizeof(in_s)); human_bytes(double(out_sum), out_s, sizeof(out_s)); double thr_gib = (tot_ms>0.0)? double(in_sum)/(tot_ms/1000.0)/1e9 : 0.0; std::ostringstream os; os<<"[GPU"<<device_id<<"/S"<<C.stats.stream_index<<"] done N="<<C.filled<<" in="<<in_s<<" out="<<out_s<<" h2d="<<std::fixed<<std::setprecision(2)<<h2d_ms<<"ms comp="<<comp_ms<<"ms d2h="<<d2h_ms<<"ms tot="<<tot_ms<<"ms thr="<<std::fixed<<std::setprecision(2)<<thr_gib<<" GiB/s"; vlog(V_DEBUG,opt, os.str()); }

          C.busy=false; C.filled=0; C.batch.clear();
        } else if (q != cudaErrorNotReady) { checkCuda(q, "cudaStreamQuery"); }
      }

      if (producer_done_seen) { bool all_idle=true; for (auto & C: ctxs) { if (C.busy){ all_idle=false; break; } } if (all_idle) break; }
      if (!submitted_any) {
        bool blocked=false;
        for (auto & C: ctxs) {
          if (!C.busy) { continue; }
          checkCuda(cudaStreamSynchronize(C.stream), "cudaStreamSynchronize");
          checkCuda(cudaMemcpy(C.h_stats.data(), C.d_stats, sizeof(nvcompStatus_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H statuses sync)");
          checkCuda(cudaMemcpy(C.h_comp_sizes.data(), C.d_comp_sizes, sizeof(size_t)*C.filled, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H comp_sizes sync)");
          float h2d_ms=0, comp_ms=0, d2h_ms=0, tot_ms=0; cudaEventElapsedTime(&h2d_ms, C.ev_h2d_begin, C.ev_h2d_end); cudaEventElapsedTime(&comp_ms, C.ev_h2d_end, C.ev_comp_end); cudaEventElapsedTime(&d2h_ms, C.ev_comp_end, C.ev_d2h_end); cudaEventElapsedTime(&tot_ms, C.ev_h2d_begin, C.ev_d2h_end);
          { std::lock_guard<std::mutex> lk(devstats->m); devstats->h2d_ms+=h2d_ms; devstats->comp_ms+=comp_ms; devstats->d2h_ms+=d2h_ms; devstats->total_ms+=tot_ms; devstats->batches+=1; }
          uint64_t in_sum=0, out_sum=0; for (size_t i=0;i<C.filled;++i) { in_sum += C.h_in_sizes[i]; out_sum += C.h_comp_sizes[i]; }
          {
            // D2H exact copies and release to writer (synchronous path)
            for (size_t i = 0; i < C.filled; ++i) {
              const size_t csz = C.h_comp_sizes[i];
              std::vector<char> h_out(csz);
              const void * d_src = static_cast<char*>(C.d_out_base) + i * C.max_out_chunk;
              checkCuda(cudaMemcpy(h_out.data(), d_src, csz, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H exact sync)");
              { std::lock_guard<std::mutex> lk(results->m); results->data.emplace(C.batch[i].seq, std::move(h_out)); }
              results->cv.notify_one();
            }
          }
          #ifdef HAVE_NVCOMP
          if (sched) sched->add_gpu_bytes(in_sum);
          #endif
          { C.stats.h2d_ms += h2d_ms; C.stats.comp_ms += comp_ms; C.stats.d2h_ms += d2h_ms; C.stats.total_ms += tot_ms; C.stats.in_bytes += in_sum; C.stats.out_bytes += out_sum; C.stats.batches += 1; C.stats.chunks += C.filled; }
          if (opt.verbosity >= V_DEBUG) { char in_s[32], out_s[32]; human_bytes(double(in_sum), in_s, sizeof(in_s)); human_bytes(double(out_sum), out_s, sizeof(out_s)); double thr_gib = (tot_ms>0.0)? double(in_sum)/(tot_ms/1000.0)/1e9 : 0.0; std::ostringstream os; os<<"[GPU"<<device_id<<"/S"<<C.stats.stream_index<<"] done N="<<C.filled<<" in="<<in_s<<" out="<<out_s<<" h2d="<<std::fixed<<std::setprecision(2)<<h2d_ms<<"ms comp="<<comp_ms<<"ms d2h="<<d2h_ms<<"ms tot="<<tot_ms<<"ms thr="<<std::fixed<<std::setprecision(2)<<thr_gib<<" GiB/s"; vlog(V_DEBUG,opt, os.str()); }
          C.busy=false; C.filled=0; C.batch.clear(); blocked=true; break;
        }
        if (!blocked) { std::this_thread::yield(); }
      }
    }

    if (json_sink) { std::lock_guard<std::mutex> lk(json_sink->m); auto & vec = json_sink->per_dev[size_t(device_id)]; for (size_t s=0;s<ctxs.size();++s) vec.push_back(ctxs[s].stats); }
    if (opt.verbosity >= V_DEBUG) {
      for (auto & C : ctxs) {
        double thr_gib = (C.stats.total_ms>0.0)? double(C.stats.in_bytes)/(C.stats.total_ms/1000.0)/1e9 : 0.0;
        std::ostringstream os;
        os << "[GPU"<<device_id<<"/S"<<C.stats.stream_index<<"] total batches="<<C.stats.batches
           << " chunks="<<C.stats.chunks
           << " in="<<C.stats.in_bytes<<"B out="<<C.stats.out_bytes<<"B"
           << " time="<<std::fixed<<std::setprecision(2)<<C.stats.total_ms<<"ms"
           << " thr="<<std::fixed<<std::setprecision(2)<<thr_gib<<" GiB/s";
        vlog(V_DEBUG, opt, os.str());
      }
    }

    for (auto & C : ctxs) { free_stream_buffers_only(C); if (C.stream) cudaStreamDestroy(C.stream); C.stream=nullptr; }
  }
  catch (const std::exception & e) {
    *any_gpu_failed = true; *fatal_msg = std::string("[GPU")+std::to_string(device_id)+"] "+e.what();
    try {
      if (rescue) {
        if (auto sp = std::weak_ptr<std::vector<StreamCtx>>(ctxs_ptr).lock()) {
          for (auto & C : *sp) {
            if (C.busy && !C.batch.empty()) { for (size_t i=0;i<C.filled;++i) rescue->push(Task{ C.batch[i].seq, C.batch[i].data }); }
            free_stream_buffers_only(C); if (C.stream) cudaStreamDestroy(C.stream);
          }
        }
      }
    } catch(...){}
    { std::lock_guard<std::mutex> lk(results->m); results->cv.notify_all(); }
  }
}

static void compress_nvcomp(FILE * in, FILE * out, const Options & opt, Meter * m)
{
  int device_count=0; if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) { if (opt.gpu_only) die("GPU requested (--gpu-only) but no CUDA devices available"); vlog(V_VERBOSE,opt, "nvCOMP: no GPUs found; falling back to MT CPU"); compress_cpu_mt(in, out, opt, m); return; }
  if (opt.verbosity >= V_VERBOSE && (opt.level_user_set || opt.fast_flag || opt.best_flag)) vlog(V_VERBOSE,opt, "note: GPU backend ignores CPU level flags; CPU frames in hybrid still honor level");

  size_t chosen_mib = opt.chunk_mib; if (!opt.chunk_user_set) { chosen_mib = auto_chunk_mib_gpu(in, opt, device_count); std::ostringstream os; os<<"auto-chunk (GPU,"<<device_count<<" devices): "<<chosen_mib<<" MiB"; vlog(V_VERBOSE,opt, os.str()); }
  const size_t host_chunk = std::max<size_t>(1, chosen_mib) * ONE_MIB;

  TaskQueue queue; RescueQueue rescue; ResultStore results; std::atomic<size_t> seq_counter{0};
  std::atomic<bool> any_gpu_failed{false}; std::atomic<bool> abort_on_failure{ opt.gpu_only };
  std::atomic<bool> gpu_started{false};
  std::vector<DevStats> per_dev(device_count); StatsSink json_sink(device_count); CpuAgg cpuagg{}; cpuagg.threads = 0;
  uint64_t total_in = 0; if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input)) total_in = (uint64_t)fs::file_size(opt.input);
  std::atomic<bool> progress_done{false}; std::thread progress_thr(progress_loop, std::cref(opt), m, total_in, &progress_done);
  std::thread writer_thr(writer_thread, out, std::ref(results), std::cref(opt), m);

  std::unique_ptr<HybridSched> sched_ptr; HybridSched * sched=nullptr; std::atomic<bool> tick_done{false}; std::thread tick_thr;
  if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) { sched_ptr = std::make_unique<HybridSched>(opt.cpu_share, /*cpu_threads*/0, device_count, opt); sched=sched_ptr.get(); tick_thr = std::thread(tick_loop_fn, std::ref(tick_done), sched); if (opt.verbosity>=V_VERBOSE) { std::ostringstream os; os<<std::fixed<<std::setprecision(1)<<"Using hybrid mode: CPU share "<<((opt.cpu_share>=0.0)?(opt.cpu_share*100.0):(sched->target_share()*100.0))<<((opt.cpu_share>=0.0)?"% (fixed)":"% (adaptive)"); vlog(V_VERBOSE,opt, os.str()); } }

  // Rescue pool
  std::vector<std::thread> rescue_pool; { unsigned ths = std::max(1u, std::thread::hardware_concurrency()/2); rescue_pool.reserve(ths); for (unsigned i=0;i<ths;++i) rescue_pool.emplace_back(cpu_worker_rescue, (int)i, &rescue, &results, &opt, m, &cpuagg); }

  // GPU workers
  std::vector<std::thread> workers; workers.reserve(device_count); Options opt_for_workers = opt; opt_for_workers.chunk_mib = chosen_mib; std::vector<std::string> fatal_msgs(device_count);
  for (int dev=0; dev<device_count; ++dev) workers.emplace_back(gpu_worker, dev, opt_for_workers, &queue, &rescue, &results, &per_dev[size_t(dev)], &json_sink, m, sched, &any_gpu_failed, &abort_on_failure, &fatal_msgs[size_t(dev)], &gpu_started);

  // Warm GPUs before starting CPU pool
  std::vector<std::thread> cpu_pool; int cpu_threads=0;
  if (sched) {
    if (opt.verbosity>=V_VERBOSE) vlog(V_VERBOSE,opt, "hybrid: warming GPUs for up to 500ms before starting CPU threads");
    auto t0w = std::chrono::steady_clock::now();
    while (!gpu_started.load(std::memory_order_acquire)) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); if (std::chrono::steady_clock::now() - t0w > std::chrono::milliseconds(500)) break; }
    cpu_threads = opt.cpu_threads; if (cpu_threads<=0) { unsigned hw=std::max(1u,std::thread::hardware_concurrency()); unsigned def=std::max(1u,hw/3); if (def>32u) def=32u; cpu_threads = def; }
    if (!gpu_started.load(std::memory_order_acquire)) { if (opt.verbosity>=V_VERBOSE) vlog(V_VERBOSE,opt, "hybrid: GPUs not active after warm-up; limiting CPU threads to 8"); if (cpu_threads>8) cpu_threads=8; }
    cpuagg.threads = cpu_threads; cpuagg.per_thread.resize((size_t)cpu_threads);
    if (opt.verbosity>=V_VERBOSE) { std::ostringstream os; os<<"hybrid: starting CPU pool: "<<cpu_threads<<" threads"; vlog(V_VERBOSE,opt, os.str()); }
    for (int i=0;i<cpu_threads;++i) cpu_pool.emplace_back(cpu_worker, i, &queue, &results, &opt, m, (void*)sched, &cpuagg);
  }

  // Producer: read host chunks, split into <= gpu_chunk subchunks
  std::vector<char> host_in(host_chunk);
  while (true) {
    size_t n_host = std::fread(host_in.data(), 1, host_chunk, in);
    if (n_host == 0) break;
    if (m) m->read_bytes.fetch_add(n_host);
    size_t off=0; const size_t gpu_chunk = std::min(host_chunk, GPU_SUBCHUNK_MAX);
    while (off < n_host) { size_t sub_n = std::min(gpu_chunk, n_host - off); Task t; t.seq = seq_counter.fetch_add(1, std::memory_order_relaxed); t.data.assign(host_in.data()+off, host_in.data()+off+sub_n); queue.push(std::move(t)); off += sub_n; }
    if (abort_on_failure.load() && any_gpu_failed.load()) break;
  }
  queue.set_done(); { std::lock_guard<std::mutex> lk(results.m); results.producer_done = true; results.total_tasks = queue.total_tasks(); } results.cv.notify_all();

  for (auto & th : workers) th.join();
  if (abort_on_failure.load() && any_gpu_failed.load()) { progress_done = true; progress_thr.join(); { std::lock_guard<std::mutex> lk(results.m); results.workers_done = true; results.cv.notify_all(); } writer_thr.join(); std::string msg = "GPU path failed (--gpu-only)."; for (const auto & s : fatal_msgs) if (!s.empty()) { msg += " "; msg += s; } die(msg); }

  rescue.set_done(); for (auto & th : rescue_pool) th.join();
  if (!cpu_pool.empty()) for (auto & th : cpu_pool) th.join();
  { std::lock_guard<std::mutex> lk(results.m); results.workers_done = true; } results.cv.notify_all();
  if (sched) { tick_done = true; if (tick_thr.joinable()) tick_thr.join(); }

  writer_thr.join(); progress_done = true; progress_thr.join();
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
  uint64_t in_bytes=meter.read_bytes.load(), out_bytes=meter.wrote_bytes.load();
  js << "{\n  \"version\": \"" << GZSTD_VERSION << "\",\n  \"mode\": \"" << (opt.mode==Mode::COMPRESS?"compress":opt.mode==Mode::DECOMPRESS?"decompress":"test") << "\",\n  \"elapsed_sec\": " << std::fixed << std::setprecision(6) << elapsed_sec << ",\n  \"input_bytes\": " << in_bytes << ",\n  \"output_bytes\": " << out_bytes << ",\n  \"cpu\": { \"threads\": " << cpuagg.threads << " }\n}\n";
}

/*======================================================================
 Main
======================================================================*/
static Options parse_args(int argc, char ** argv);
int main(int argc, char ** argv)
{
  setup_signal_handlers();

  Options opt = parse_args(argc, argv);
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
      // Atomic overwrite: write to .tmp, rename on success
      out = open_output_atomic(opt.output, tmp);
      use_atomic = true;
    } else {
      // Direct write: write to final name, delete on failure
      out = std::fopen(opt.output.c_str(), "wb");
      if (!out) die("cannot open output: " + opt.output);
      register_tmp_file(opt.output); // arm cleanup to delete on failure
    }
  }
  Meter meter; vlog(V_VERBOSE, opt, "gzstd starting...");
  if (opt.verbosity >= V_DEBUG && opt.mode == Mode::COMPRESS) {
    std::ostringstream os;
#ifdef HAVE_NVCOMP
    os << "compression level (CPU path): " << opt.level
       << (opt.ultra && opt.level >= 20 ? " (ultra)" : "");
    if (!opt.cpu_only && !opt.gpu_only && opt.hybrid) {
      os << "  [note: GPU backend ignores level; CPU frames use this level]";
    }
#else
    os << "compression level: " << opt.level
       << (opt.ultra && opt.level >= 20 ? " (ultra)" : "");
#endif
    vlog(V_DEBUG, opt, os.str());
  }

  std::atomic<bool> prog_done{false}; std::thread prog_thr; uint64_t total_in_for_progress=0;
  if (opt.mode == Mode::TEST || opt.mode == Mode::DECOMPRESS) { if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input)) total_in_for_progress = (uint64_t)fs::file_size(opt.input); prog_thr = std::thread(progress_loop, std::cref(opt), &meter, total_in_for_progress, &prog_done); }

  const auto t0 = std::chrono::steady_clock::now();
  if (opt.mode == Mode::COMPRESS) {
#ifdef HAVE_NVCOMP
    if (opt.cpu_only) {
      compress_cpu_mt(in, out, opt, &meter);
      if (!opt.stats_json.empty()) { CpuAgg agg{}; agg.threads = (opt.cpu_threads>0?opt.cpu_threads: std::max(1,int(std::thread::hardware_concurrency())-1)); double elapsed = std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::steady_clock::now()-t0).count(); write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, agg); }
    } else {
      compress_nvcomp(in, out, opt, &meter);
    }
#else
    if (opt.gpu_only) die_usage("This binary was built without nvCOMP; --gpu-only cannot be satisfied");
    if (opt.hybrid) vlog(V_VERBOSE,opt, "Hybrid requested but not available in CPU-only build; using MT CPU.");
    compress_cpu_mt(in, out, opt, &meter);
    if (!opt.stats_json.empty()) { CpuAgg agg{}; agg.threads = (opt.cpu_threads>0?opt.cpu_threads: std::max(1,int(std::thread::hardware_concurrency())-1)); double elapsed = std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::steady_clock::now()-t0).count(); write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, agg); }
#endif
  } else {
    decompress_stream(in, out, opt, &meter);
    if (!opt.stats_json.empty()) { double elapsed = std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::steady_clock::now()-t0).count(); CpuAgg dummy{}; dummy.threads=1; write_stats_json_cpu_only(opt.stats_json, opt, meter, elapsed, dummy); }
  }

  if (opt.mode == Mode::TEST) { prog_done = true; if (prog_thr.joinable()) prog_thr.join(); uint64_t comp_size = 0; if (opt.input != "-" && fs::exists(opt.input) && fs::is_regular_file(opt.input)) comp_size = (uint64_t)fs::file_size(opt.input); else comp_size = meter.read_bytes.load(); uint64_t decomp_size = meter.wrote_bytes.load(); double pct = (decomp_size > 0) ? (double)comp_size / (double)decomp_size * 100.0 : 0.0; if (opt.verbosity >= V_DEFAULT) { std::string base_name; if (opt.input == "-") base_name = "(stdin)"; else if (opt.input.size()>4 && opt.input.substr(opt.input.size()-4) == ".zst") base_name = opt.input.substr(0, opt.input.size()-4); else base_name = opt.input; std::cout << base_name << "  : " << "compressed size:" << comp_size << " bytes, " << "uncompressed size:" << decomp_size << " bytes, " << "ratio: " << std::fixed << std::setprecision(1) << pct << "%\n"; } std::fclose(in); return EXIT_OK; }
  if (opt.mode == Mode::DECOMPRESS) { prog_done = true; if (prog_thr.joinable()) prog_thr.join(); }

  if (!to_stdout) {
    fsync_file(out); std::fclose(out); std::fclose(in);
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
    if (!opt.keep && opt.input != "-") { std::error_code ec_rm; fs::remove(opt.input, ec_rm); }
  } else {
    // Flush stdout to ensure all data reaches the downstream pipe
    std::fflush(stdout);
    std::fclose(in);
  }

  vlog(V_VERBOSE, opt, "done.");
  return EXIT_OK;
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
    else if (a == "-c") opt.to_stdout = true;
    else if (a == "-v") opt.verbosity = V_VERBOSE;
    else if (a == "-vv") opt.verbosity = V_DEBUG;
    else if (a == "-vvv") opt.verbosity = V_TRACE;
    else if (a == "-q" || a == "--quiet") opt.verbosity = V_ERROR;
    else if (a == "-qq" || a == "--silent") opt.verbosity = V_SILENT;
    else if (a == "--progress") { opt.force_progress = true; if (opt.verbosity < V_DEFAULT) opt.verbosity = V_DEFAULT; }
    else if (a == "--no-progress") { opt.force_progress = false; if (opt.verbosity == V_DEFAULT) opt.verbosity = V_ERROR; }
    else if (a.size() >= 2 && a[0] == '-') {
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
    else if (a.rfind("-T", 0) == 0 && a.size() > 2) { int th = std::stoi(a.substr(2)); opt.cpu_threads = th; }
    else if (a == "-T" || a == "--threads") { int th = 0; if (a == "-T" && i + 1 < argc) th = std::stoi(argv[++i]); else if (!parse_int_arg("threads", i, argc, argv, th)) die_usage("missing value for --threads"); opt.cpu_threads = th; }
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
#ifdef HAVE_NVCOMP
    else if (a == "--gpu-only") opt.gpu_only = true;
    else if (parse_num_arg("gpu-batch", i, argc, argv, opt.gpu_batch_cap)) {}
    else if (parse_double_arg("gpu-mem-frac", i, argc, argv, opt.gpu_mem_fraction)) {}
    else if (parse_num_arg("gpu-streams", i, argc, argv, opt.gpu_streams)) {}
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
    else { if (opt.input.empty()) opt.input = a; else die_usage("multiple input files not supported yet"); }
  }
  if (opt.input.empty()) opt.input = "-";

  // When reading from stdin, default to stdout output (pipe-friendly, like gzip/zstd)
  if (opt.input == "-" && opt.output.empty()) opt.to_stdout = true;

  // When writing to stdout (-c), always keep the input file (can't delete stdin,
  // and deleting a named file when output goes to stdout matches gzip behavior)
  if (opt.to_stdout) opt.keep = true;

#ifdef HAVE_NVCOMP
  if (opt.gpu_batch_cap == 0) opt.gpu_batch_cap = DEFAULT_GPU_BATCH_CAP;
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
  else if (!opt.to_stdout && opt.output.empty()) {
    if (opt.mode == Mode::COMPRESS) opt.output = (opt.input == "-") ? "stdout" : (opt.input + ".zst");
    else {
      if (opt.input.size() > 4 && opt.input.substr(opt.input.size() - 4) == ".zst") opt.output = opt.input.substr(0, opt.input.size() - 4);
      else opt.output = opt.input + ".out";
    }
  }
  if (opt.gpu_only && (opt.cpu_only || opt.hybrid)) die_usage("--gpu-only cannot be combined with --cpu-only or --hybrid");
  if (opt.cpu_only && opt.hybrid) die_usage("--cpu-only cannot be combined with --hybrid");

  // Auto-lower verbosity when used as a pipe (both stdin and stdout are non-TTY)
  // but only if the user hasn't explicitly set verbosity via flags.
  // V_DEFAULT stays V_DEFAULT (keeps progress on a TTY stderr), but if stderr
  // is also not a TTY we let progress_loop decide (it checks is_stderr_tty).
  // We do NOT auto-quiet here; the progress_loop already handles the TTY check.

  // Sync global verbosity for die() (which has no access to Options)
  g_verbosity = opt.verbosity;

  return opt;
}
